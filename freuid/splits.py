"""Group-aware train / val / test splits and cross-validation folds.

Design goals (see HANDOFF):
  - No random sample-level split: hold out whole document ``type`` groups.
  - Near-duplicate pairs stay in the same partition (union-find on dup graph).
  - Types linked by cross-type near-dups are merged before assignment.
  - Stratify by fraud rate and group size when assigning types to splits.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from . import config, io


@dataclass
class SplitConfig:
    val_type_fraction: float = 0.15
    test_type_fraction: float = 0.15
    n_cv_folds: int = 5
    random_state: int = 42
    min_types_per_split: int = 1


@dataclass
class SplitManifest:
    config: SplitConfig
    split_col: str = "split"
    group_col: str = "split_group"
    type_col: str = config.TYPE_COL
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    n_cv_folds: int = 0
    type_assignments: dict[str, str] = field(default_factory=dict)
    type_component_sizes: dict[str, int] = field(default_factory=dict)
    split_stats: dict[str, dict] = field(default_factory=dict)
    cv_folds: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for x in self.parent:
            root = self.find(x)
            groups.setdefault(root, []).append(x)
        return groups


def _duplicate_pairs_from_artifacts() -> list[tuple[str, str]]:
    payload = io.load_json("duplicates.json") or {}
    pairs: list[tuple[str, str]] = []
    for key in ("near_duplicate_pairs", "exact_duplicate_groups"):
        chunk = payload.get(key, [])
        if key.endswith("groups"):
            for group in chunk:
                ids = [str(x) for x in group]
                for i in range(1, len(ids)):
                    pairs.append((ids[0], ids[i]))
        else:
            for row in chunk:
                pairs.append((str(row["a"]), str(row["b"])))
    return pairs


def merge_types_by_duplicates(df: pd.DataFrame, pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
    """Attach ``type_component`` — types linked by near-dups share an id."""
    out = df.copy()
    id_to_type = dict(zip(out[config.ID_COL].astype(str), out[config.TYPE_COL].astype(str)))
    uf = UnionFind()
    for t in out[config.TYPE_COL].astype(str).unique():
        uf.add(t)
    for a, b in pairs:
        ta, tb = id_to_type.get(a), id_to_type.get(b)
        if ta and tb:
            uf.union(ta, tb)
    comp_map: dict[str, str] = {}
    for i, (_, members) in enumerate(sorted(uf.components().items(), key=lambda kv: kv[0])):
        comp_id = f"comp_{i:04d}"
        for t in members:
            comp_map[t] = comp_id
    out["type_component"] = out[config.TYPE_COL].astype(str).map(comp_map)
    return out


def build_split_groups(df: pd.DataFrame, pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
    """Attach ``split_group`` so near-duplicate ids never cross partitions."""
    out = df.copy()
    ids = out[config.ID_COL].astype(str).tolist()
    uf = UnionFind()
    for i in ids:
        uf.add(i)
    for a, b in pairs:
        if a in uf.parent and b in uf.parent:
            uf.union(a, b)
    root_map = {i: uf.find(i) for i in ids}
    out[config.TYPE_COL] = out[config.TYPE_COL].astype(str)
    group_meta: dict[str, str] = {}
    for root, members in uf.components().items():
        if root not in ids:
            continue
        types = out.loc[out[config.ID_COL].astype(str).isin(members), config.TYPE_COL]
        group_meta[root] = str(types.mode().iloc[0]) if not types.empty else "unknown"
    out["split_group"] = out[config.ID_COL].astype(str).map(root_map)
    out["split_group_type"] = out["split_group"].map(group_meta)
    return out


def _component_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [config.TYPE_COL, "type_component", config.LABEL_COL, config.ID_COL]
    g = (
        df.groupby("type_component", as_index=False)
        .agg(
            n=(config.ID_COL, "count"),
            n_fraud=(config.LABEL_COL, "sum"),
            types=(config.TYPE_COL, lambda s: sorted(set(map(str, s)))),
        )
    )
    g["fraud_rate"] = g["n_fraud"] / g["n"].clip(lower=1)
    return g.sort_values("n", ascending=False).reset_index(drop=True)


def _assign_components(
    components: pd.DataFrame,
    cfg: SplitConfig,
) -> dict[str, str]:
    """Greedy stratified assignment of type components to train/val/test."""
    rng = np.random.default_rng(cfg.random_state)
    rows = components.sample(frac=1.0, random_state=cfg.random_state).reset_index(drop=True)
    targets = {
        "train": 1.0 - cfg.val_type_fraction - cfg.test_type_fraction,
        "val": cfg.val_type_fraction,
        "test": cfg.test_type_fraction,
    }
    totals = {"train": 0, "val": 0, "test": 0}
    frauds = {"train": 0, "val": 0, "test": 0}
    assignments: dict[str, str] = {}

    for _, row in rows.iterrows():
        comp = str(row["type_component"])
        n, n_fraud = int(row["n"]), int(row["n_fraud"])
        best_split = "train"
        best_cost = float("inf")
        for split in ("train", "val", "test"):
            proj_total = totals[split] + n
            proj_fraud = frauds[split] + n_fraud
            proj_rate = proj_fraud / max(proj_total, 1)
            global_total = sum(totals.values()) + n
            global_fraud = sum(frauds.values()) + n_fraud
            global_rate = global_fraud / max(global_total, 1)
            size_penalty = abs((totals[split] + n) / max(global_total, 1) - targets[split])
            rate_penalty = abs(proj_rate - global_rate)
            empty_bonus = -0.05 if totals[split] == 0 else 0.0
            cost = size_penalty + 0.35 * rate_penalty + empty_bonus
            if cost < best_cost:
                best_cost = cost
                best_split = split
        assignments[comp] = best_split
        totals[best_split] += n
        frauds[best_split] += n_fraud

    # shuffle tie-break reproducibility without changing totals much
    _ = rng
    return assignments


def _split_stats(df: pd.DataFrame, split_name: str) -> dict:
    part = df[df["split"] == split_name]
    if part.empty:
        return {"n": 0, "n_fraud": 0, "fraud_rate": None, "n_types": 0, "n_groups": 0}
    return {
        "n": int(len(part)),
        "n_fraud": int((part[config.LABEL_COL] == config.POSITIVE_LABEL).sum()),
        "fraud_rate": float(part[config.LABEL_COL].mean()),
        "n_types": int(part[config.TYPE_COL].nunique()),
        "n_groups": int(part["split_group"].nunique()),
        "n_countries": int(part["country"].nunique()) if "country" in part else None,
    }


def build_cv_folds(df: pd.DataFrame, cfg: SplitConfig) -> list[dict]:
    """Group K-fold on train+val using ``type_component`` as the group key."""
    from sklearn.model_selection import GroupKFold

    pool = df[df["split"].isin(("train", "val"))].copy()
    if pool.empty:
        return []
    n_groups = pool["type_component"].nunique()
    n_splits = min(cfg.n_cv_folds, n_groups)
    if n_splits < 2:
        return []

    gkf = GroupKFold(n_splits=n_splits)
    groups = pool["type_component"].astype(str).to_numpy()
    ids = pool[config.ID_COL].astype(str).to_numpy()
    folds: list[dict] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(pool, groups=groups)):
        folds.append(
            {
                "fold": fold_idx,
                "train_ids": ids[tr_idx].tolist(),
                "val_ids": ids[va_idx].tolist(),
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "val_types": sorted(pool.iloc[va_idx][config.TYPE_COL].astype(str).unique().tolist()),
            }
        )
    return folds


def build_splits(
    df: pd.DataFrame | None = None,
    cfg: SplitConfig | None = None,
    duplicate_pairs: Iterable[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, SplitManifest]:
    """Return labeled dataframe with ``split`` column and a manifest."""
    cfg = cfg or SplitConfig()
    base = df if df is not None else io.load_labels()
    if config.TYPE_COL not in base.columns:
        raise ValueError(f"Missing {config.TYPE_COL} column in labels table")

    pairs = list(duplicate_pairs) if duplicate_pairs is not None else _duplicate_pairs_from_artifacts()
    labeled_ids = set(base[config.ID_COL].astype(str))
    pairs = [(a, b) for a, b in pairs if a in labeled_ids and b in labeled_ids]

    with_groups = build_split_groups(base, pairs)
    with_components = merge_types_by_duplicates(with_groups, pairs)
    components = _component_table(with_components)
    comp_assign = _assign_components(components, cfg)

    manifest = SplitManifest(config=cfg, n_cv_folds=cfg.n_cv_folds)
    if len(components) < 3:
        manifest.warnings.append("Few type components; val/test may be tiny.")

    type_to_split = {}
    for _, row in components.iterrows():
        comp = str(row["type_component"])
        split = comp_assign[comp]
        for t in row["types"]:
            type_to_split[str(t)] = split
    manifest.type_assignments = type_to_split
    manifest.type_component_sizes = {
        str(r["type_component"]): int(r["n"]) for _, r in components.iterrows()
    }

    out = with_components.copy()
    out["split"] = out[config.TYPE_COL].astype(str).map(type_to_split)
    if out["split"].isna().any():
        missing = int(out["split"].isna().sum())
        manifest.warnings.append(f"{missing} rows had unknown type; defaulted to train.")
        out["split"] = out["split"].fillna("train")

    for split_name in ("train", "val", "test"):
        manifest.split_stats[split_name] = _split_stats(out, split_name)
        manifest.__setattr__(f"n_{split_name}", manifest.split_stats[split_name]["n"])

    manifest.cv_folds = build_cv_folds(out, cfg)
    if len(manifest.cv_folds) < cfg.n_cv_folds:
        manifest.warnings.append(
            f"Only {len(manifest.cv_folds)} CV folds possible "
            f"({components['type_component'].nunique()} type components)."
        )
    return out, manifest


def save_splits(df: pd.DataFrame, manifest: SplitManifest, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else config.SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        config.ID_COL,
        config.PATH_COL,
        config.LABEL_COL,
        config.IS_DIGITAL_COL,
        config.TYPE_COL,
        "country",
        "doc_type",
        "split_group",
        "type_component",
        "split",
    ]
    cols = [c for c in cols if c in df.columns]
    slim = df[cols].copy()
    slim.to_csv(out_dir / "labeled_with_split.csv", index=False)
    for split_name in ("train", "val", "test"):
        part = slim[slim["split"] == split_name]
        part.to_csv(out_dir / f"{split_name}.csv", index=False)

    payload = asdict(manifest)
    payload["config"] = asdict(manifest.config)
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2))
    for fold in manifest.cv_folds:
        fold_idx = fold["fold"]
        (out_dir / f"cv_fold_{fold_idx}.json").write_text(json.dumps(fold, indent=2))
    return out_dir


def load_splits(out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = Path(out_dir) if out_dir else config.SPLITS_DIR
    path = out_dir / "labeled_with_split.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split table: {path}. Run scripts/09_build_splits.py.")
    return pd.read_csv(path)


def load_manifest(out_dir: Path | None = None) -> SplitManifest:
    out_dir = Path(out_dir) if out_dir else config.SPLITS_DIR
    path = out_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing split manifest: {path}.")
    raw = json.loads(path.read_text())
    raw["config"] = SplitConfig(**raw["config"])
    return SplitManifest(**raw)


def iter_cv_folds(df: pd.DataFrame | None = None) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield ``(train_df, val_df)`` for each group-aware CV fold on train+val."""
    table = df if df is not None else load_splits()
    manifest = load_manifest()
    id_col = config.ID_COL
    for fold in manifest.cv_folds:
        tr_ids = set(fold["train_ids"])
        va_ids = set(fold["val_ids"])
        pool = table[table["split"].isin(("train", "val"))]
        yield (
            pool[pool[id_col].astype(str).isin(tr_ids)].copy(),
            pool[pool[id_col].astype(str).isin(va_ids)].copy(),
        )
