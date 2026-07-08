"""Validation pipeline: leakage-safe splits, group CV, leave-one-type-out,
FREUID scoring with bootstrap confidence intervals, and per-type breakdowns.

A model is plugged in as one of:
  * ``score_fn(frame) -> scores``                 (already-trained scorer)
  * ``fit_predict_fn(train_df, val_df) -> scores``(refit per fold)

Everything funnels through the FREUID metric bundle (``freuid.metrics``) and
reports uncertainty, because with this dataset a single point estimate is not
trustworthy on its own.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, io, metrics, splits


ScoreFn = Callable[[pd.DataFrame], Sequence[float]]
FitPredictFn = Callable[[pd.DataFrame, pd.DataFrame], Sequence[float]]


@dataclass
class FoldResult:
    fold: int | str
    n_train: int
    n_val: int
    metrics: metrics.MetricResult
    val_types: list[str] = field(default_factory=list)
    val_fraud_rate: float | None = None


@dataclass
class ValidationReport:
    holdout: dict[str, dict | None] = field(default_factory=dict)
    cv_folds: list[FoldResult] = field(default_factory=list)
    cv_summary: dict[str, float] = field(default_factory=dict)
    loto: list[dict] = field(default_factory=list)
    per_type: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "holdout": self.holdout,
            "cv_summary": self.cv_summary,
            "cv_folds": [
                {
                    "fold": f.fold,
                    "n_train": f.n_train,
                    "n_val": f.n_val,
                    "val_fraud_rate": f.val_fraud_rate,
                    "val_types": f.val_types,
                    "freuid": f.metrics.freuid,
                    "audet": f.metrics.audet,
                    "apcer_at_1pct_bpcer": f.metrics.apcer_at_1pct_bpcer,
                    "eer": f.metrics.eer,
                }
                for f in self.cv_folds
            ],
            "leave_one_type_out": self.loto,
            "per_type": self.per_type,
            "warnings": self.warnings,
        }


def _metric_dict(m: metrics.MetricResult) -> dict:
    return {
        "freuid": m.freuid,
        "audet": m.audet,
        "apcer_at_1pct_bpcer": m.apcer_at_1pct_bpcer,
        "eer": m.eer,
    }


class ValidationPipeline:
    """End-to-end local validation built on leakage-safe group splits."""

    def __init__(
        self,
        splits_dir: Path | None = None,
        rebuild: bool = False,
        split_config: splits.SplitConfig | None = None,
        bootstrap: int = 500,
    ) -> None:
        self.splits_dir = Path(splits_dir) if splits_dir else config.SPLITS_DIR
        self.bootstrap = bootstrap
        if rebuild or not (self.splits_dir / "labeled_with_split.csv").exists():
            df, manifest = splits.build_splits(cfg=split_config)
            splits.save_splits(df, manifest, self.splits_dir)
        self.table = splits.load_splits(self.splits_dir)
        self.manifest = splits.load_manifest(self.splits_dir)

    # --- partitions --------------------------------------------------------
    def _part(self, name: str) -> pd.DataFrame:
        return self.table[self.table[splits.SPLIT_COL] == name].copy()

    @property
    def train(self) -> pd.DataFrame:
        return self._part("train")

    @property
    def val(self) -> pd.DataFrame:
        return self._part("val")

    @property
    def test(self) -> pd.DataFrame:
        return self._part("test")

    def cv_folds(self) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        return list(splits.iter_cv_folds(self.table))

    # --- scoring -----------------------------------------------------------
    @staticmethod
    def evaluate_frame(frame: pd.DataFrame, y_score: Sequence[float]) -> metrics.MetricResult:
        y_true = frame[config.LABEL_COL].to_numpy()
        return metrics.freuid_score(y_true, y_score)

    def evaluate_holdout(
        self,
        score_fn: ScoreFn,
        splits_to_score: Sequence[str] = ("val", "test"),
    ) -> ValidationReport:
        """Score hold-out partitions, with bootstrap CIs and per-type breakdown."""
        report = ValidationReport()
        for split_name in splits_to_score:
            part = self._part(split_name)
            if part.empty:
                report.holdout[split_name] = None
                report.warnings.append(f"Hold-out split '{split_name}' is empty.")
                continue
            scores = np.asarray(list(score_fn(part)), dtype=np.float64)
            y = part[config.LABEL_COL].to_numpy()
            entry = _metric_dict(metrics.freuid_score(y, scores))
            if self.bootstrap:
                entry["ci"] = metrics.bootstrap_metric(y, scores, n_boot=self.bootstrap)
            report.holdout[split_name] = entry
            report.per_type[split_name] = self._per_type(part, scores)
        return report

    def _per_type(self, frame: pd.DataFrame, scores: np.ndarray) -> dict:
        out: dict[str, dict] = {}
        if config.TYPE_COL not in frame.columns:
            return out
        scores = np.asarray(scores, dtype=np.float64)
        types = frame[config.TYPE_COL].astype(str).to_numpy()
        y = frame[config.LABEL_COL].to_numpy()
        for t in sorted(set(types)):
            m = types == t
            if m.sum() < 2 or len(set(y[m])) < 2:
                out[t] = {"n": int(m.sum()), "note": "too few / single-class"}
                continue
            out[t] = _metric_dict(metrics.freuid_score(y[m], scores[m]))
            out[t]["n"] = int(m.sum())
        return out

    def run_cv(self, fit_predict_fn: FitPredictFn) -> ValidationReport:
        """Group-aware CV on train+val. ``fit_predict_fn(train, val) -> scores``."""
        report = ValidationReport()
        freuid_scores: list[float] = []
        audet_scores: list[float] = []
        for fold_idx, (tr, va) in enumerate(self.cv_folds()):
            if va.empty or tr.empty:
                report.warnings.append(f"Skipping empty fold {fold_idx}.")
                continue
            scores = np.asarray(list(fit_predict_fn(tr, va)), dtype=np.float64)
            if scores.shape[0] != len(va):
                raise ValueError(f"Fold {fold_idx}: expected {len(va)} scores, got {scores.shape[0]}")
            result = self.evaluate_frame(va, scores)
            freuid_scores.append(result.freuid)
            audet_scores.append(result.audet)
            report.cv_folds.append(FoldResult(
                fold=fold_idx,
                n_train=len(tr),
                n_val=len(va),
                metrics=result,
                val_types=sorted(va[config.TYPE_COL].astype(str).unique().tolist()),
                val_fraud_rate=float(va[config.LABEL_COL].mean()),
            ))
        if freuid_scores:
            fr = np.asarray(freuid_scores)
            au = np.asarray(audet_scores)
            report.cv_summary = {
                "freuid_mean": float(fr.mean()),
                "freuid_std": float(fr.std()),
                "freuid_min": float(fr.min()),
                "freuid_max": float(fr.max()),
                "audet_mean": float(au.mean()),
                "audet_std": float(au.std()),
                "n_folds": float(len(freuid_scores)),
            }
        return report

    def run_leave_one_type_out(self, fit_predict_fn: FitPredictFn) -> list[dict]:
        """Train on all-but-one type, evaluate on the held-out type. Cross-type
        generalization stress test (the pessimistic, type-shift scenario)."""
        loto: list[dict] = []
        for t, tr, va in splits.iter_type_holdout(self.table):
            if va.empty or tr.empty or len(set(va[config.LABEL_COL])) < 2:
                loto.append({"type": t, "n_val": int(len(va)), "note": "skipped"})
                continue
            scores = np.asarray(list(fit_predict_fn(tr, va)), dtype=np.float64)
            res = self.evaluate_frame(va, scores)
            entry = {"type": t, "n_train": int(len(tr)), "n_val": int(len(va))}
            entry.update(_metric_dict(res))
            loto.append(entry)
        if loto:
            vals = [e["freuid"] for e in loto if "freuid" in e]
            if vals:
                arr = np.asarray(vals)
                loto.append({"type": "__summary__", "freuid_mean": float(arr.mean()),
                             "freuid_std": float(arr.std()), "n_types": len(vals)})
        return loto

    # --- diagnostics -------------------------------------------------------
    def sanity_check(self) -> dict:
        """Leakage / balance diagnostics. Raises if a hard invariant is broken."""
        splits.assert_no_leakage(self.table)  # hard fail on any leakage

        out: dict = {"strategy": self.manifest.strategy,
                     "manifest_warnings": self.manifest.warnings,
                     "n_groups": self.manifest.n_groups,
                     "n_nontrivial_groups": self.manifest.n_nontrivial_groups,
                     "partitions": {}}
        for split_name in ("train", "val", "test"):
            out["partitions"][split_name] = splits._split_stats(self.table, split_name)

        # fraud-rate balance across CV folds (should be tight)
        rates = self.manifest.fold_fraud_rates
        if rates:
            arr = np.asarray(rates)
            out["cv_fraud_rate_spread"] = {
                "min": float(arr.min()), "max": float(arr.max()),
                "std": float(arr.std()), "folds": len(rates),
            }

        # explicit duplicate-leakage report from artifacts (defense in depth)
        pairs = splits._duplicate_pairs_from_artifacts()
        split_map = dict(zip(self.table[config.ID_COL].astype(str), self.table[splits.SPLIT_COL]))
        leaked = [
            {"a": a, "b": b, "split_a": split_map.get(a), "split_b": split_map.get(b)}
            for a, b in pairs
            if split_map.get(a) and split_map.get(b) and split_map[a] != split_map[b]
        ]
        out["duplicate_leakage_pairs"] = leaked
        out["duplicate_leakage_count"] = len(leaked)

        type_sets = {
            s: set(self.table.loc[self.table[splits.SPLIT_COL] == s, config.TYPE_COL].astype(str))
            for s in ("train", "val", "test")
        }
        out["type_overlap"] = {
            "train_val": sorted(type_sets["train"] & type_sets["val"]),
            "train_test": sorted(type_sets["train"] & type_sets["test"]),
            "val_test": sorted(type_sets["val"] & type_sets["test"]),
        }
        return out

    def save_report(self, report: ValidationReport, name: str = "validation_report.json") -> Path:
        return io.save_json(name, report.to_dict())
