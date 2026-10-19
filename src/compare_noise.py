#!/usr/bin/env python3
"""Noise-sweep analysis of the per-fold results written by QSVC.py.

Reads every <results>/folds_noise<sigma>.csv (same seed => same folds at every level) and writes:
    noise_degradation.csv  best-of-family and family-mean accuracy/AUC per problem and sigma
    noise_paired.csv       best fixed / trainable quantum model vs best classical, paired per fold
    noise_degradation.png  accuracy and AUC vs sigma per family, with chance levels

Paired test: corrected resampled t-test (Nadeau & Bengio 2003, Bouckaert & Frank 2004). Folds of
repeated k-fold CV share training data, so a plain paired t-test or Wilcoxon understates the variance.
Best-of-family is picked on the same CV scores, so it is optimistic; family means are reported too.

Usage:
    uv run python src/compare_noise.py [--results results] [--splits 5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from QSVC import COLORS

MAJORITY = 2 / 3  # every one-vs-rest Iris problem is 50 vs 100 samples
KEYS = ["sigma", "problem", "family"]


def load_folds(results: Path) -> pd.DataFrame:
    paths = sorted(results.glob("folds_noise*.csv"))
    if not paths:
        raise SystemExit(f"No folds_noise*.csv in {results.resolve()}; run QSVC.py first.")
    return pd.concat(pd.read_csv(p).assign(sigma=float(p.stem.removeprefix("folds_noise"))) for p in paths)


def degradation(folds: pd.DataFrame) -> pd.DataFrame:
    per_model = folds.groupby([*KEYS, "model"], sort=False)[["accuracy", "auc"]].agg(["mean", "std"])
    per_model.columns = [f"{metric}_{stat}" for metric, stat in per_model.columns]
    per_model = per_model.reset_index()
    best = per_model.loc[per_model.groupby(KEYS)["accuracy_mean"].idxmax()]
    family_mean = (per_model.groupby(KEYS)[["accuracy_mean", "auc_mean"]].mean()
                   .add_prefix("family_").reset_index())
    return best.merge(family_mean, on=KEYS).sort_values(KEYS).reset_index(drop=True)


def corrected_ttest(d: np.ndarray, test_train_ratio: float) -> tuple[float, float]:
    """Nadeau-Bengio corrected resampled t-test on paired fold differences: (t, two-sided p)."""
    mean, var = d.mean(), d.var(ddof=1)
    if var == 0:  # every fold gives the same difference
        return (0.0, 1.0) if mean == 0 else (float(np.sign(mean) * np.inf), 0.0)
    t = mean / np.sqrt((1 / len(d) + test_train_ratio) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=len(d) - 1))


def paired(folds: pd.DataFrame, best: pd.DataFrame, test_train_ratio: float) -> pd.DataFrame:
    rows = []
    for (sigma, problem), b in best.groupby(["sigma", "problem"]):
        acc = folds[(folds.sigma == sigma) & (folds.problem == problem)].pivot(
            index="fold", columns="model", values="accuracy")
        classical = b.loc[b.family == "Classical", "model"].item()
        for _, q in b[b.family != "Classical"].iterrows():
            d = (acc[q.model] - acc[classical]).to_numpy()
            t, p = corrected_ttest(d, test_train_ratio)
            rows.append({"sigma": sigma, "problem": problem, "quantum_family": q.family,
                         "quantum": q.model, "classical": classical, "mean_diff": d.mean(),
                         "std_diff": d.std(ddof=1), "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                         "losses": int((d < 0).sum()), "t": t, "p": p})
    return pd.DataFrame(rows)


def plot(best: pd.DataFrame, path: Path) -> None:
    problems = list(best.problem.unique())
    fig, axes = plt.subplots(2, len(problems), figsize=(5 * len(problems), 8), sharex=True, squeeze=False)
    offsets = dict(zip(COLORS, (-0.02, 0.0, 0.02)))  # keep overlapping error bars readable
    for col, problem in enumerate(problems):
        for row, (metric, chance, label) in enumerate([("accuracy", MAJORITY, "majority class"),
                                                       ("auc", 0.5, "chance")]):
            ax = axes[row, col]
            for family, df in best[best.problem == problem].groupby("family", sort=False):
                ax.errorbar(df.sigma + offsets[family], df[f"{metric}_mean"], yerr=df[f"{metric}_std"],
                            marker="o", capsize=3, color=COLORS[family], label=family)
            ax.axhline(chance, color="gray", linestyle="--", linewidth=1, label=label)
            ax.set(title=f"{problem}: best-of-family {metric}", ylabel=f"CV {metric} (mean ± std)",
                   xlabel="noise σ" if row else None)
            ax.grid(alpha=0.3)
    axes[0, 0].legend()
    axes[1, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--splits", type=int, default=5, help="k used by QSVC.py (sets test/train ratio)")
    args = p.parse_args(argv)

    folds = load_folds(args.results)
    best = degradation(folds)
    pairs = paired(folds, best, test_train_ratio=1 / (args.splits - 1))

    pd.set_option("display.width", 250)
    print("\nBest-of-family CV accuracy by noise level (family mean in brackets):")
    table = best.assign(cell=best.apply(
        lambda r: f"{r.accuracy_mean:.3f} +/- {r.accuracy_std:.3f} [{r.family_accuracy_mean:.3f}]", axis=1))
    print(table.pivot(index=["problem", "family"], columns="sigma", values="cell").to_string())
    print("\nBest quantum vs best classical, paired per fold (corrected resampled t-test):")
    print(pairs.to_string(index=False, float_format="%.3f"))

    best.to_csv(args.results / "noise_degradation.csv", index=False, float_format="%.6f")
    pairs.to_csv(args.results / "noise_paired.csv", index=False, float_format="%.6f")
    plot(best, args.results / "noise_degradation.png")
    print(f"\nWrote noise_degradation.csv, noise_paired.csv, noise_degradation.png to {args.results.resolve()}")


if __name__ == "__main__":
    main()
