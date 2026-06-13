"""FREUID competition metrics: DET curve, AuDET, APCER@1%BPCER, FREUID score.

Scores are continuous fraud probabilities in [0, 1] (higher = more likely fraud).
Labels: 0 = genuine/bona-fide, 1 = attack/fraud.

Official combination (lower FREUID is better):

    g_audet = 1 - AuDET
    g_apcer = 1 - APCER@1%BPCER
    FREUID  = 1 - 2 * g_audet * g_apcer / (g_audet + g_apcer)

AuDET is the area under the DET curve in (BPCER, APCER) error space.

The DET curve is built with a vectorised ``searchsorted`` (O(n log n)) that is
bit-for-bit identical to the naive per-threshold loop (see tests/test_metrics.py),
and ``freuid_score`` builds it once and reuses it for AuDET / APCER / EER.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    freuid: float
    audet: float
    apcer_at_1pct_bpcer: float
    g_audet: float
    g_apcer: float
    eer: float | None = None
    threshold_at_1pct_bpcer: float | None = None


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(y_score, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if y.size == 0:
        raise ValueError("empty inputs")
    return y, s


def error_rates(y_true, y_score, threshold: float) -> tuple[float, float]:
    """Return (APCER, BPCER) at a fixed score threshold.

    APCER: attack classified as genuine (score < threshold).
    BPCER: genuine classified as attack (score >= threshold).
    """
    y, s = _as_arrays(y_true, y_score)
    pred_fraud = s >= threshold
    attacks = y == 1
    genuine = y == 0
    apcer = float(np.mean(~pred_fraud[attacks])) if attacks.any() else 0.0
    bpcer = float(np.mean(pred_fraud[genuine])) if genuine.any() else 0.0
    return apcer, bpcer


def det_curve(
    y_true,
    y_score,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute DET points over unique score thresholds.

    Returns ``(bpcer, apcer, thresholds)`` sorted by ascending BPCER.

    Vectorised: for a threshold ``t``, ``APCER(t)`` is the fraction of attack
    scores strictly below ``t`` and ``BPCER(t)`` the fraction of genuine scores
    at or above ``t``. Both are exact integer counts (via ``searchsorted`` on the
    sorted per-class scores) divided by the class size, so the floating-point
    results are identical to evaluating ``error_rates`` at every threshold.
    """
    y, s = _as_arrays(y_true, y_score)
    thresholds = np.unique(s)
    thresholds = np.concatenate(([1.0 + 1e-12], thresholds, [-1e-12]))

    attack_scores = s[y == 1]
    genuine_scores = s[y == 0]
    n_att = attack_scores.size
    n_gen = genuine_scores.size

    if n_att:
        sa = np.sort(attack_scores)
        # # attacks with score < t  ==  np.mean(~(s >= t) over attacks)
        apcer = np.searchsorted(sa, thresholds, side="left").astype(np.float64) / n_att
    else:
        apcer = np.zeros(thresholds.shape, dtype=np.float64)

    if n_gen:
        sg = np.sort(genuine_scores)
        # # genuine with score >= t  ==  n_gen - (# genuine with score < t)
        n_below = np.searchsorted(sg, thresholds, side="left")
        bpcer = (n_gen - n_below).astype(np.float64) / n_gen
    else:
        bpcer = np.zeros(thresholds.shape, dtype=np.float64)

    order = np.argsort(bpcer)
    return bpcer[order], apcer[order], thresholds[order]


# --------------------------------------------------------------------------
# Curve-derived quantities (the ``_from_curve`` variants avoid recomputing the
# DET curve when several metrics are needed together).
# --------------------------------------------------------------------------
def _audet_from_curve(bpcer: np.ndarray, apcer: np.ndarray) -> float:
    if bpcer.size < 2:
        return float(apcer[-1]) if apcer.size else 1.0
    return float(np.clip(np.trapezoid(apcer, bpcer), 0.0, 1.0))


def _apcer_at_bpcer_from_curve(
    bpcer: np.ndarray, apcer: np.ndarray, thresholds: np.ndarray, target_bpcer: float
) -> tuple[float, float | None]:
    eligible = np.where(bpcer <= target_bpcer)[0]
    if eligible.size:
        idx = int(eligible[-1])  # highest threshold still meeting BPCER budget
    else:
        idx = int(np.argmin(bpcer))
    return float(apcer[idx]), float(thresholds[idx])


def _eer_from_curve(bpcer: np.ndarray, apcer: np.ndarray) -> float | None:
    diff = np.abs(bpcer - apcer)
    if diff.size == 0:
        return None
    idx = int(np.argmin(diff))
    return float((bpcer[idx] + apcer[idx]) / 2.0)


def audet(y_true, y_score) -> float:
    """Area under the DET curve (error area; lower is better)."""
    bpcer, apcer, _ = det_curve(y_true, y_score)
    return _audet_from_curve(bpcer, apcer)


def apcer_at_bpcer(
    y_true,
    y_score,
    target_bpcer: float = 0.01,
) -> tuple[float, float | None]:
    """APCER at the operating point closest to ``target_bpcer``.

    Returns ``(apcer, threshold)``. When BPCER never reaches the target,
    uses the point with minimum BPCER above target, else the lowest-BPCER point.
    """
    bpcer, apcer, thresholds = det_curve(y_true, y_score)
    return _apcer_at_bpcer_from_curve(bpcer, apcer, thresholds, target_bpcer)


def eer(y_true, y_score) -> float | None:
    """Equal error rate (optional diagnostic)."""
    bpcer, apcer, _ = det_curve(y_true, y_score)
    return _eer_from_curve(bpcer, apcer)


def freuid_score(y_true, y_score) -> MetricResult:
    """Compute the full FREUID metric bundle (DET curve built once)."""
    bpcer, apcer, thresholds = det_curve(y_true, y_score)
    area = _audet_from_curve(bpcer, apcer)
    apcer_1, thr = _apcer_at_bpcer_from_curve(bpcer, apcer, thresholds, 0.01)
    g_audet = 1.0 - area
    g_apcer = 1.0 - apcer_1
    denom = g_audet + g_apcer
    if denom <= 0:
        freuid = 1.0
    else:
        freuid = 1.0 - (2.0 * g_audet * g_apcer / denom)
    return MetricResult(
        freuid=float(freuid),
        audet=float(area),
        apcer_at_1pct_bpcer=float(apcer_1),
        g_audet=float(g_audet),
        g_apcer=float(g_apcer),
        eer=_eer_from_curve(bpcer, apcer),
        threshold_at_1pct_bpcer=thr,
    )


# --------------------------------------------------------------------------
# Uncertainty: stratified bootstrap confidence intervals
# --------------------------------------------------------------------------
def bootstrap_metric(
    y_true,
    y_score,
    n_boot: int = 500,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Stratified bootstrap CIs for FREUID / AuDET / APCER@1%BPCER.

    Resampling is done *within* each class so the genuine/attack balance — and
    therefore the BPCER grid that APCER@1%BPCER depends on — is preserved across
    bootstrap replicates. Returns mean and [lo, hi] percentile interval per
    metric, plus the point estimate on the full sample.
    """
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(y_score, dtype=np.float64)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)

    point = freuid_score(y, s)
    if pos.size == 0 or neg.size == 0:
        return {
            "point": {"freuid": point.freuid, "audet": point.audet,
                      "apcer_at_1pct_bpcer": point.apcer_at_1pct_bpcer},
            "n_boot": 0,
            "note": "single-class sample; CIs undefined",
        }

    fr: list[float] = []
    au: list[float] = []
    ap: list[float] = []
    for _ in range(n_boot):
        bi = np.concatenate((
            rng.choice(pos, size=pos.size, replace=True),
            rng.choice(neg, size=neg.size, replace=True),
        ))
        r = freuid_score(y[bi], s[bi])
        fr.append(r.freuid)
        au.append(r.audet)
        ap.append(r.apcer_at_1pct_bpcer)

    lo_pct = (1.0 - ci) / 2.0 * 100.0
    hi_pct = (1.0 + ci) / 2.0 * 100.0

    def _ci(point_val: float, samples: list[float]) -> dict:
        arr = np.asarray(samples, dtype=np.float64)
        return {
            "point": float(point_val),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "lo": float(np.percentile(arr, lo_pct)),
            "hi": float(np.percentile(arr, hi_pct)),
        }

    return {
        "freuid": _ci(point.freuid, fr),
        "audet": _ci(point.audet, au),
        "apcer_at_1pct_bpcer": _ci(point.apcer_at_1pct_bpcer, ap),
        "n_boot": n_boot,
        "ci": ci,
    }
