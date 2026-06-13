"""Tests for the FREUID metric bundle (freuid/metrics.py).

Pure-CPU, dependency-light. Runs under pytest *and* directly:

    uv run python -m pytest tests/test_metrics.py
    uv run python tests/test_metrics.py        # no pytest required

Note: AuDET is a trapezoid integral over the DET curve and is sensitive to
BPCER ties on tiny samples, so the separable fixture below is sized to behave
like real data (many distinct scores, no degenerate ties).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import metrics  # noqa: E402


# Genuine (0) clearly below attacks (1): perfectly separable, distinct scores.
_GEN = np.linspace(0.01, 0.40, 100)
_ATK = np.linspace(0.60, 0.99, 100)
Y_SEP = np.r_[np.zeros(100), np.ones(100)].astype(int)
S_SEP = np.r_[_GEN, _ATK]


def test_det_curve_is_sorted_and_bounded():
    bpcer, apcer, thr = metrics.det_curve(Y_SEP, S_SEP)
    assert bpcer.shape == apcer.shape == thr.shape
    assert np.all(np.diff(bpcer) >= -1e-12)
    assert bpcer.min() >= -1e-12 and bpcer.max() <= 1 + 1e-12
    assert apcer.min() >= -1e-12 and apcer.max() <= 1 + 1e-12


def test_error_rates_endpoints():
    apcer, bpcer = metrics.error_rates(Y_SEP, S_SEP, threshold=2.0)
    assert apcer == 1.0 and bpcer == 0.0
    apcer, bpcer = metrics.error_rates(Y_SEP, S_SEP, threshold=-1.0)
    assert apcer == 0.0 and bpcer == 1.0


def test_perfect_separation_is_near_optimal():
    r = metrics.freuid_score(Y_SEP, S_SEP)
    assert r.audet < 0.05
    assert r.apcer_at_1pct_bpcer < 1e-9
    assert r.freuid < 0.05
    assert r.eer is not None and r.eer < 1e-6


def test_inverted_scores_are_much_worse():
    good = metrics.freuid_score(Y_SEP, S_SEP)
    bad = metrics.freuid_score(Y_SEP, 1.0 - S_SEP)
    assert bad.audet > good.audet
    assert bad.freuid > good.freuid
    assert bad.freuid > 0.5


def test_freuid_matches_harmonic_identity():
    r = metrics.freuid_score(Y_SEP, S_SEP)
    g_audet = 1.0 - r.audet
    g_apcer = 1.0 - r.apcer_at_1pct_bpcer
    expected = 1.0 - (2.0 * g_audet * g_apcer / (g_audet + g_apcer))
    assert abs(r.freuid - expected) < 1e-9
    assert abs(r.g_audet - g_audet) < 1e-12
    assert abs(r.g_apcer - g_apcer) < 1e-12


def test_apcer_at_bpcer_returns_threshold_meeting_budget():
    apcer, thr = metrics.apcer_at_bpcer(Y_SEP, S_SEP, target_bpcer=0.01)
    assert thr is not None
    got_apcer, got_bpcer = metrics.error_rates(Y_SEP, S_SEP, thr)
    assert got_bpcer <= 0.01 + 1e-9
    assert abs(got_apcer - apcer) < 1e-9


def test_scores_are_floats_in_range():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    y[0], y[1] = 0, 1
    s = rng.random(200)
    r = metrics.freuid_score(y, s)
    for v in (r.freuid, r.audet, r.apcer_at_1pct_bpcer):
        assert 0.0 <= v <= 1.0


def test_bootstrap_metric_brackets_point_estimate():
    r = metrics.bootstrap_metric(Y_SEP, S_SEP, n_boot=100, seed=0)
    fr = r["freuid"]
    assert fr["lo"] <= fr["point"] + 1e-9
    assert fr["point"] - 1e-9 <= fr["hi"]
    assert r["n_boot"] == 100


# --------------------------------------------------------------------------
# Vectorised det_curve must stay bit-identical to the naive per-threshold loop.
# --------------------------------------------------------------------------
def _ref_det_curve(y_true, y_score):
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(y_score, dtype=np.float64)
    thresholds = np.unique(s)
    thresholds = np.concatenate(([1.0 + 1e-12], thresholds, [-1e-12]))
    bl, al = [], []
    for t in thresholds:
        pred = s >= float(t)
        att = y == 1
        gen = y == 0
        al.append(float(np.mean(~pred[att])) if att.any() else 0.0)
        bl.append(float(np.mean(pred[gen])) if gen.any() else 0.0)
    order = np.argsort(bl)
    return np.asarray(bl)[order], np.asarray(al)[order], thresholds[order]


def test_det_curve_matches_reference_loop_exactly():
    rng = np.random.default_rng(123)
    for kind in ("uniform", "ties", "binary"):
        for n in (6, 50, 777):
            y = rng.integers(0, 2, size=n)
            y[0], y[1] = 0, 1
            if kind == "uniform":
                s = rng.random(n)
            elif kind == "ties":
                s = rng.integers(0, 5, size=n) / 4.0
            else:
                s = rng.integers(0, 2, size=n).astype(float)
            rb, ra, rt = _ref_det_curve(y, s)
            vb, va, vt = metrics.det_curve(y, s)
            assert np.array_equal(rb, vb), f"bpcer differs ({kind}, n={n})"
            assert np.array_equal(ra, va), f"apcer differs ({kind}, n={n})"
            assert np.array_equal(rt, vt), f"thresholds differ ({kind}, n={n})"


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
