#!/usr/bin/env python3
"""Classical vs quantum-kernel SVMs on Iris (one-vs-rest) with leak-free repeated stratified CV.

Every model runs through the same per-fold pipeline, so the comparison is like-for-like:
    SelectKBest + MinMaxScaler fitted on the training fold only
    -> kernel fitted on the training fold (trainable kernels are optimised here)
    -> K_train and K_test computed once each
    -> SVC(kernel="precomputed"); predictions and AUC both come from the same decision scores

Usage:
    uv run python src/QSVC.py                      # noise-free
    uv run python src/QSVC.py --noise 2.0          # add N(0, 2.0) noise to the raw features
    uv run python src/QSVC.py --noise 0 0.5 1 2    # noise sweep, one set of outputs per level
    uv run python src/compare_noise.py             # degradation + paired analysis of the sweep
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import pauli_feature_map, z_feature_map, zz_feature_map
from qiskit_machine_learning.kernels import FidelityStatevectorKernel, TrainableFidelityStatevectorKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.optimizers import COBYLA, SPSA
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils.parallel import Parallel, delayed

log = logging.getLogger("qsvc")

N_FEATURES = 2  # selected features == qubits
PROBLEMS = {"Setosa_vs_Rest": 0, "Versicolor_vs_Rest": 1, "Virginica_vs_Rest": 2}
METRICS = ["accuracy", "precision", "recall", "f1", "auc", "train_accuracy", "train_f1", "fit_time", "predict_time"]
COLORS = {"Classical": "tab:blue", "Fixed Quantum": "tab:orange", "Trainable Quantum": "tab:red"}
OPTIMIZERS = {
    "SPSA": lambda maxiter: SPSA(maxiter=maxiter, learning_rate=0.1, perturbation=0.01),
    "COBYLA": lambda maxiter: COBYLA(maxiter=maxiter),
}

Kernel = Callable[[np.ndarray, np.ndarray | None], np.ndarray]  # (A, B) -> K(A, B); B=None means K(A, A)
KernelFactory = Callable[[np.ndarray, np.ndarray], Kernel]  # fitted on (X_train, y_train) of one fold


@dataclass(frozen=True)
class Model:
    name: str
    family: str
    fit_kernel: KernelFactory
    feature_map: str = "-"
    optimizer: str = "-"


# --- kernels ---------------------------------------------------------------------------------

def classical_kernel(metric: str, **params) -> KernelFactory:
    def fit(X: np.ndarray, y: np.ndarray) -> Kernel:
        # sklearn's gamma="scale", computed on the training fold
        gamma = {"gamma": 1.0 / (X.shape[1] * X.var())} if metric in ("rbf", "poly") else {}
        return partial(pairwise_kernels, metric=metric, **params, **gamma)

    return fit


def fixed_quantum_kernel(feature_map: QuantumCircuit) -> KernelFactory:
    kernel = FidelityStatevectorKernel(feature_map=feature_map)  # exact, no shot noise; data-independent
    return lambda X, y: kernel.evaluate


def trainable_quantum_kernel(circuit: QuantumCircuit, theta: ParameterVector,
                             make_optimizer: Callable[[], object], seed: int) -> KernelFactory:
    def fit(X: np.ndarray, y: np.ndarray) -> Kernel:
        # Reseed per fit: initial point and SPSA perturbations no longer depend on run order.
        algorithm_globals.random_seed = seed
        kernel = TrainableFidelityStatevectorKernel(feature_map=circuit, training_parameters=theta)
        trainer = QuantumKernelTrainer(kernel, loss="svc_loss", optimizer=make_optimizer())
        return trainer.fit(X, y).quantum_kernel.evaluate

    return fit


def trainable_feature_map(n: int, *, entangle: bool, rx_after_encoding: bool = False,
                          final_rz: bool = False) -> tuple[QuantumCircuit, ParameterVector]:
    """RY(θ) -> RZ(x) [-> RX(θ)] [-> CX·RZ(x_i·x_j)·CX] -> RY(θ) [-> RZ(θ)] on every qubit."""
    x = ParameterVector("x", n)
    theta = ParameterVector("θ", n * (2 + rx_after_encoding + final_rz))
    th = iter(theta)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(next(th), q)
    for q in range(n):
        qc.rz(x[q], q)
        if rx_after_encoding:
            qc.rx(next(th), q)
    if entangle:
        for q in range(n - 1):
            qc.cx(q, q + 1)
            qc.rz(x[q] * x[q + 1], q + 1)
            qc.cx(q, q + 1)
    for q in range(n):
        qc.ry(next(th), q)
        if final_rz:
            qc.rz(next(th), q)
    return qc, theta


def build_models(maxiter: int, seed: int) -> list[Model]:
    classical = {"rbf": {}, "poly": {"degree": 3, "coef0": 1.0}, "linear": {}}  # coef0=0 cannot fit a middle class
    fixed = {
        "ZZ": zz_feature_map(N_FEATURES, reps=2),
        "Z": z_feature_map(N_FEATURES, reps=2),
        "Pauli_Y_ZZ": pauli_feature_map(N_FEATURES, reps=2, paulis=["Y", "ZZ"]),  # ["Z","ZZ"] would equal ZZ
    }
    trainable = {
        "Z": trainable_feature_map(N_FEATURES, entangle=False),
        "ZZ": trainable_feature_map(N_FEATURES, entangle=True, final_rz=True),
        "Pauli": trainable_feature_map(N_FEATURES, entangle=True, rx_after_encoding=True, final_rz=True),
    }
    return (
        [Model(name, "Classical", classical_kernel(name, **params)) for name, params in classical.items()]
        + [Model(f"Fixed_{name}", "Fixed Quantum", fixed_quantum_kernel(fm), feature_map=name)
           for name, fm in fixed.items()]
        + [Model(f"Trainable_{fm}_{opt}", "Trainable Quantum",
                 trainable_quantum_kernel(circuit, theta, partial(make_opt, maxiter), seed),
                 feature_map=fm, optimizer=opt)
           for fm, (circuit, theta) in trainable.items() for opt, make_opt in OPTIMIZERS.items()]
    )


# --- evaluation ------------------------------------------------------------------------------

def load_data(noise: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    iris = load_iris()
    X = iris.data
    if noise > 0:
        X = X + np.random.default_rng(seed).normal(0.0, noise, X.shape)
    return X, iris.target, np.array(iris.feature_names)


def run_fold(model: Model, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray,
             train: np.ndarray, test: np.ndarray) -> dict:
    prep = make_pipeline(SelectKBest(f_classif, k=N_FEATURES), MinMaxScaler(feature_range=(0, np.pi)))
    X_tr = prep.fit_transform(X[train], y[train])
    X_te = prep.transform(X[test])

    start = time.perf_counter()
    kernel = model.fit_kernel(X_tr, y[train])
    K_tr = kernel(X_tr, None)
    svc = SVC(kernel="precomputed").fit(K_tr, y[train])
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    scores = svc.decision_function(kernel(X_te, X_tr))
    predict_time = time.perf_counter() - start

    pred = (scores > 0).astype(int)  # identical to svc.predict for binary SVC, without a second kernel pass
    train_pred = (svc.decision_function(K_tr) > 0).astype(int)
    y_te = y[test]
    return {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0),
        "auc": roc_auc_score(y_te, scores),
        "train_accuracy": accuracy_score(y[train], train_pred),
        "train_f1": f1_score(y[train], train_pred, zero_division=0),
        "fit_time": fit_time,
        "predict_time": predict_time,
        "features": ", ".join(feature_names[prep[0].get_support()]),
    }


def run_cv(models: list[Model], X: np.ndarray, y: np.ndarray, feature_names: np.ndarray, *,
           n_splits: int, n_repeats: int, seed: int, n_jobs: int) -> pd.DataFrame:
    # Stratifying on the 3-class label keeps every one-vs-rest problem stratified and gives all
    # problems and models the same folds, so model differences are paired per fold.
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    folds = list(cv.split(X, y))
    rows = []
    with Parallel(n_jobs=n_jobs) as parallel:  # folds are independent; workers are reused across models
        for problem, target in PROBLEMS.items():
            y_bin = (y == target).astype(int)
            for model in models:
                results = parallel(delayed(run_fold)(model, X, y_bin, feature_names, train, test)
                                   for train, test in folds)
                rows += [{"problem": problem, "family": model.family, "model": model.name,
                          "feature_map": model.feature_map, "optimizer": model.optimizer, "fold": i, **r}
                         for i, r in enumerate(results)]
                acc = [r["accuracy"] for r in results]
                log.info("%-18s %-22s acc %.3f +/- %.3f  fit %.2fs/fold", problem, model.name,
                         np.mean(acc), np.std(acc, ddof=1), np.mean([r["fit_time"] for r in results]))
    return pd.DataFrame(rows)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    keys = ["problem", "family", "model", "feature_map", "optimizer"]
    summary = folds.groupby(keys, sort=False)[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


# --- output ----------------------------------------------------------------------------------

def report(summary: pd.DataFrame) -> None:
    cols = ["family", "model", "accuracy_mean", "accuracy_std", "f1_mean", "auc_mean",
            "train_accuracy_mean", "fit_time_mean", "predict_time_mean"]
    for problem, df in summary.groupby("problem", sort=False):
        print(f"\n=== {problem}: mean over CV folds (accuracy_std = std over folds) ===")
        print(df.sort_values("accuracy_mean", ascending=False)[cols].to_string(index=False, float_format="%.3f"))
        best = df.loc[df.groupby("family", sort=False)["accuracy_mean"].idxmax()]
        print("Best per family: " + " | ".join(
            f"{r.family}: {r.model} {r.accuracy_mean:.3f} +/- {r.accuracy_std:.3f}" for r in best.itertuples()))


def plot(summary: pd.DataFrame, path: Path) -> None:
    problems = list(summary["problem"].unique())
    fig, axes = plt.subplots(1, len(problems) + 1, figsize=(6 * (len(problems) + 1), 6))
    for ax, problem in zip(axes, problems):
        df = summary[summary["problem"] == problem]
        ax.barh(df["model"], df["accuracy_mean"], xerr=df["accuracy_std"], capsize=3,
                color=df["family"].map(COLORS))
        ax.set(title=problem, xlabel="CV accuracy (mean ± std)", xlim=(0, 1.05))
        ax.invert_yaxis()
    ax = axes[-1]
    for family, df in summary.groupby("family", sort=False):
        ax.scatter(df["fit_time_mean"], df["accuracy_mean"], label=family, color=COLORS[family])
    ax.set(xscale="log", xlabel="mean fit time per fold [s]", ylabel="CV accuracy", title="Accuracy vs fit time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--noise", type=float, nargs="+", default=[0.0],
                   help="std(s) of Gaussian noise on raw features; several values run a sweep (0 = none)")
    p.add_argument("--splits", type=int, default=5, help="CV folds")
    p.add_argument("--repeats", type=int, default=3, help="CV repetitions")
    p.add_argument("--maxiter", type=int, default=50, help="optimizer iterations for trainable kernels")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores, 1 = clean timings)")
    p.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args.out.mkdir(parents=True, exist_ok=True)
    models = build_models(args.maxiter, args.seed)

    # Same seed at every level: identical folds, and the noise is the same draw scaled by sigma,
    # so differences between levels come from sigma alone.
    for noise in args.noise:
        log.info("--- noise sigma = %g ---", noise)
        X, y, feature_names = load_data(noise, args.seed)
        folds = run_cv(models, X, y, feature_names,
                       n_splits=args.splits, n_repeats=args.repeats, seed=args.seed, n_jobs=args.jobs)
        summary = summarize(folds)
        print(f"\n##### noise sigma = {noise:g} #####")
        report(summary)

        tag = f"noise{noise:g}"
        folds.to_csv(args.out / f"folds_{tag}.csv", index=False, float_format="%.6f")
        summary.to_csv(args.out / f"summary_{tag}.csv", index=False, float_format="%.6f")
        plot(summary, args.out / f"comparison_{tag}.png")
        log.info("Wrote folds_%s.csv, summary_%s.csv, comparison_%s.png to %s", tag, tag, tag, args.out.resolve())


if __name__ == "__main__":
    main()
