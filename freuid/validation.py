"""Validation pipeline: load splits, run CV, and score with FREUID metrics."""
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


@dataclass
class ValidationReport:
    holdout: dict[str, metrics.MetricResult | None] = field(default_factory=dict)
    cv_folds: list[FoldResult] = field(default_factory=list)
    cv_summary: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        def _metric(m: metrics.MetricResult | None) -> dict | None:
            if m is None:
                return None
            return {
                "freuid": m.freuid,
                "audet": m.audet,
                "apcer_at_1pct_bpcer": m.apcer_at_1pct_bpcer,
                "eer": m.eer,
            }

        return {
            "holdout": {k: _metric(v) for k, v in self.holdout.items()},
            "cv_summary": self.cv_summary,
            "cv_folds": [
                {
                    "fold": f.fold,
                    "n_train": f.n_train,
                    "n_val": f.n_val,
                    "val_types": f.val_types,
                    "metrics": _metric(f.metrics),
                }
                for f in self.cv_folds
            ],
            "warnings": self.warnings,
        }


class ValidationPipeline:
    """End-to-end local validation using group-aware splits."""

    def __init__(
        self,
        splits_dir: Path | None = None,
        rebuild: bool = False,
        split_config: splits.SplitConfig | None = None,
    ) -> None:
        self.splits_dir = Path(splits_dir) if splits_dir else config.SPLITS_DIR
        if rebuild or not (self.splits_dir / "labeled_with_split.csv").exists():
            df, manifest = splits.build_splits(cfg=split_config)
            splits.save_splits(df, manifest, self.splits_dir)
        self.table = splits.load_splits(self.splits_dir)
        self.manifest = splits.load_manifest(self.splits_dir)

    @property
    def train(self) -> pd.DataFrame:
        return self.table[self.table["split"] == "train"].copy()

    @property
    def val(self) -> pd.DataFrame:
        return self.table[self.table["split"] == "val"].copy()

    @property
    def test(self) -> pd.DataFrame:
        return self.table[self.table["split"] == "test"].copy()

    def cv_folds(self) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        return list(splits.iter_cv_folds(self.table))

    @staticmethod
    def evaluate_frame(frame: pd.DataFrame, y_score: Sequence[float]) -> metrics.MetricResult:
        y_true = frame[config.LABEL_COL].to_numpy()
        return metrics.freuid_score(y_true, y_score)

    def evaluate_holdout(
        self,
        score_fn: ScoreFn,
        splits_to_score: Sequence[str] = ("val", "test"),
    ) -> ValidationReport:
        """Score hold-out partitions with a user-provided model/score function."""
        report = ValidationReport()
        for split_name in splits_to_score:
            part = self.table[self.table["split"] == split_name]
            if part.empty:
                report.holdout[split_name] = None
                report.warnings.append(f"Hold-out split '{split_name}' is empty.")
                continue
            scores = np.asarray(list(score_fn(part)), dtype=np.float64)
            report.holdout[split_name] = self.evaluate_frame(part, scores)
        return report

    def run_cv(
        self,
        fit_predict_fn: FitPredictFn,
    ) -> ValidationReport:
        """Run group-aware CV on train+val.

        ``fit_predict_fn(train_df, val_df)`` must return fraud scores aligned with
        ``val_df`` rows.
        """
        report = ValidationReport()
        freuid_scores: list[float] = []
        for fold_idx, (tr, va) in enumerate(self.cv_folds()):
            if va.empty or tr.empty:
                report.warnings.append(f"Skipping empty fold {fold_idx}.")
                continue
            scores = np.asarray(list(fit_predict_fn(tr, va)), dtype=np.float64)
            if scores.shape[0] != len(va):
                raise ValueError(
                    f"Fold {fold_idx}: expected {len(va)} scores, got {scores.shape[0]}"
                )
            result = self.evaluate_frame(va, scores)
            freuid_scores.append(result.freuid)
            report.cv_folds.append(
                FoldResult(
                    fold=fold_idx,
                    n_train=len(tr),
                    n_val=len(va),
                    metrics=result,
                    val_types=sorted(va[config.TYPE_COL].astype(str).unique().tolist()),
                )
            )
        if freuid_scores:
            report.cv_summary = {
                "freuid_mean": float(np.mean(freuid_scores)),
                "freuid_std": float(np.std(freuid_scores)),
                "n_folds": float(len(freuid_scores)),
            }
        return report

    def sanity_check(self) -> dict:
        """Quick leakage / balance diagnostics."""
        out: dict = {"manifest_warnings": self.manifest.warnings, "partitions": {}}
        id_col = config.ID_COL
        for split_name in ("train", "val", "test"):
            part = self.table[self.table["split"] == split_name]
            out["partitions"][split_name] = splits._split_stats(part, split_name)

        # duplicate leakage check
        pairs = splits._duplicate_pairs_from_artifacts()
        if pairs:
            split_map = dict(zip(self.table[id_col].astype(str), self.table["split"]))
            leaked = [
                {"a": a, "b": b, "split_a": split_map.get(a), "split_b": split_map.get(b)}
                for a, b in pairs
                if split_map.get(a) and split_map.get(b) and split_map[a] != split_map[b]
            ]
            out["duplicate_leakage_pairs"] = leaked
            out["duplicate_leakage_count"] = len(leaked)
        else:
            out["duplicate_leakage_pairs"] = []
            out["duplicate_leakage_count"] = 0

        type_sets = {
            s: set(self.table.loc[self.table["split"] == s, config.TYPE_COL].astype(str))
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
