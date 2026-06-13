"""FREUID competition metrics: DET curve, AuDET, APCER@1%BPCER, FREUID score.

Scores are continuous fraud probabilities in [0, 1] (higher = more likely fraud).
Labels: 0 = genuine/bona-fide, 1 = attack/fraud.

Official combination (lower FREUID is better):

    g_audet = 1 - AuDET
    g_apcer = 1 - APCER@1%BPCER
    FREUID  = 1 - 2 * g_audet * g_apcer / (g_audet + g_apcer)

AuDET is the area under the DET curve in (BPCER, APCER) error space.
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
    """
    y, s = _as_arrays(y_true, y_score)
    thresholds = np.unique(s)
    thresholds = np.concatenate(([1.0 + 1e-12], thresholds, [-1e-12]))
    bpcer_list: list[float] = []
    apcer_list: list[float] = []
    for t in thresholds:
        apcer, bpcer = error_rates(y, s, float(t))
        bpcer_list.append(bpcer)
        apcer_list.append(apcer)
    order = np.argsort(bpcer_list)
    bpcer = np.asarray(bpcer_list, dtype=np.float64)[order]
    apcer = np.asarray(apcer_list, dtype=np.float64)[order]
    thresholds = thresholds[order]
    return bpcer, apcer, thresholds


def audet(y_true, y_score) -> float:
    """Area under the DET curve (error area; lower is better)."""
    bpcer, apcer, _ = det_curve(y_true, y_score)
    if bpcer.size < 2:
        return float(apcer[-1]) if apcer.size else 1.0
    return float(np.clip(np.trapezoid(apcer, bpcer), 0.0, 1.0))


def apcer_at_bpcer(
    y_true,
    y_score,
    target_bpcer: float = 0.01,
) -> tuple[float, float | None]:
    """APCER at the operating point closest to ``target_bpcer``.

    Returns ``(apcer, threshold)``. When BPCER never reaches the target,
    uses the point with minimum BPCER above target, else the lowest-BPCER point.
    """
    y, s = _as_arrays(y_true, y_score)
    bpcer, apcer, thresholds = det_curve(y, s)

    eligible = np.where(bpcer <= target_bpcer)[0]
    if eligible.size:
        idx = int(eligible[-1])  # highest threshold still meeting BPCER budget
    else:
        idx = int(np.argmin(bpcer))
    return float(apcer[idx]), float(thresholds[idx])


def eer(y_true, y_score) -> float | None:
    """Equal error rate (optional diagnostic)."""
    bpcer, apcer, _ = det_curve(y_true, y_score)
    diff = np.abs(bpcer - apcer)
    if diff.size == 0:
        return None
    idx = int(np.argmin(diff))
    return float((bpcer[idx] + apcer[idx]) / 2.0)


def freuid_score(y_true, y_score) -> MetricResult:
    """Compute the full FREUID metric bundle."""
    area = audet(y_true, y_score)
    apcer_1, thr = apcer_at_bpcer(y_true, y_score, target_bpcer=0.01)
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
        eer=eer(y_true, y_score),
        threshold_at_1pct_bpcer=thr,
    )
