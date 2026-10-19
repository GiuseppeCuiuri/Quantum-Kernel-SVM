"""Self-checks for src/. Run: uv run python tests/test_qsvc.py (pytest also collects them)."""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import compare_noise as cn  # noqa: E402
import QSVC as q  # noqa: E402

X, Y, NAMES = q.load_data(noise=2.0, seed=42)
Y_BIN = (Y == 1).astype(int)
TRAIN, TEST = np.arange(150)[np.arange(150) % 5 != 0], np.arange(150)[np.arange(150) % 5 == 0]
MODELS = {m.name: m for m in q.build_models(maxiter=5, seed=42)}


def test_test_fold_cannot_leak_into_preprocessing_or_training():
    corrupted = X.copy()
    corrupted[TEST] = np.random.default_rng(0).normal(0, 100, corrupted[TEST].shape)
    for name in ("rbf", "Fixed_ZZ", "Trainable_Z_SPSA"):
        clean = q.run_fold(MODELS[name], X, Y_BIN, NAMES, TRAIN, TEST)
        dirty = q.run_fold(MODELS[name], corrupted, Y_BIN, NAMES, TRAIN, TEST)
        assert clean["features"] == dirty["features"], name
        assert clean["train_accuracy"] == dirty["train_accuracy"], name


def test_trainable_kernel_is_reproducible():
    a = q.run_fold(MODELS["Trainable_ZZ_SPSA"], X, Y_BIN, NAMES, TRAIN, TEST)
    b = q.run_fold(MODELS["Trainable_ZZ_SPSA"], X, Y_BIN, NAMES, TRAIN, TEST)
    assert (a["accuracy"], a["auc"]) == (b["accuracy"], b["auc"])


def test_corrected_ttest():
    d = np.array([0.02, -0.01, 0.03, 0.0, 0.05, 0.01, -0.02, 0.04])
    t, p = cn.corrected_ttest(d, test_train_ratio=0.0)  # no correction == plain paired t-test
    ref = stats.ttest_1samp(d, 0.0)
    assert np.isclose(t, ref.statistic) and np.isclose(p, ref.pvalue)
    assert cn.corrected_ttest(d, test_train_ratio=0.25)[1] > p  # correction widens the variance
    assert cn.corrected_ttest(np.zeros(5), 0.25) == (0.0, 1.0)


if __name__ == "__main__":
    test_test_fold_cannot_leak_into_preprocessing_or_training()
    test_trainable_kernel_is_reproducible()
    test_corrected_ttest()
    print("ok")
