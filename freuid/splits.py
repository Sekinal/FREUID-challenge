"""Leakage-safe train / val / test splits and cross-validation folds.

Why this exists
---------------
The dataset has only ~5 document ``type`` groups, so holding out whole types
(the old strategy) yields just a handful of high-variance configurations and a
validation score that does not track the leaderboard. Worse, near-duplicate
images (a genuine document and its tampered twin, shared blank templates) leak
across folds unless they are pinned to the same partition.

This module fixes both:

  * **Grouping** — every image is assigned a ``split_group`` = its connected
    component in the exact+near duplicate graph (``freuid.dedup``). Near-dups
    therefore never straddle a fold or the train/test boundary.
  * **Stratified group splitting** — ``StratifiedGroupKFold`` assigns whole
    groups to train/val/test and CV folds while balancing the ``type x label``
    distribution, so every fold sees a representative fraud rate and type mix.
  * **Leave-one-type-out (LOTO)** — a secondary, derivable stress test that
    trains on 4 types and evaluates on the 5th, measuring cross-type
    generalization (the pessimistic case if the hidden test shifts types).
  * **Hard assertions** — ``assert_no_leakage`` fails loudly if any group or id
    crosses a partition boundary.

Folds are stored as a ``cv_fold`` column (and derived ``type`` for LOTO) rather
than giant id-list JSONs, so the manifest stays small.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from . import config, io
from .dedup import UnionFind

CV_FOLD_COL = "cv_fold"
GROUP_COL = "split_group"
SPLIT_COL = "split"


@dataclass
class SplitConfig:
    # Fractions are of the labeled set (groups kept intact), not of types.
    # Names kept for backward compatibility with scripts/09_build_splits.py.
    val_type_fraction: float = 0.15
    test_type_fraction: float = 0.15
    n_cv_folds: int = 5
    random_state: int = 42
    min_types_per_split: int = 1
    strategy: str = "stratified_group"   # "stratified_group" | "type_holdout"
    near_dup_threshold: int = 10          # 256-bit pHash; see freuid.dedup (metadata)
    stratify_on_type: bool = True


@dataclass
class SplitManifest:
    config: SplitConfig
    strategy: str = "stratified_group"
    split_col: str = SPLIT_COL
    group_col: str = GROUP_COL
    type_col: str = config.TYPE_COL
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    n_cv_folds: int = 0
    n_groups: int = 0
    n_nontrivial_groups: int = 0
    type_assignments: dict[str, str] = field(default_factory=dict)
    split_stats: dict[str, dict] = field(default_factory=dict)
    cv_folds: list[dict] = field(default_factory=list)          # metadata only (no ids)
    type_holdout_folds: list[dict] = field(default_factory=list)  # LOTO metadata
    fold_fraud_rates: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Duplicate pairs -> groups
# --------------------------------------------------------------------------
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


def assign_groups(df: pd.DataFrame, pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
    """Attach ``split_group`` = connected-component id over the duplicate graph.

    Images with no duplicates form singleton groups (their own id), so the
    grouping degrades gracefully to per-image when no duplicates are known.
    """
    out = df.copy()
    ids = out[config.ID_COL].astype(str).tolist()
    uf = UnionFind()
    for i in ids:
        uf.add(i)
    id_set = set(ids)
    for a, b in pairs:
        if a in id_set and b in id_set:
            uf.union(a, b)
    root_map = {i: uf.find(i) for i in ids}
    # Stable, compact component ids ordered by descending size then root.
    comps: dict[str, list[str]] = {}
    for i in ids:
        comps.setdefault(root_map[i], []).append(i)
    ordered = sorted(comps.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    root_to_gid = {root: f"grp_{k:07d}" for k, (root, _) in enumerate(ordered)}
    out[GROUP_COL] = out[config.ID_COL].astype(str).map(lambda i: root_to_gid[root_map[i]])
    return out


# --------------------------------------------------------------------------
# Stratification + stratified group splitting
# --------------------------------------------------------------------------
def _strat_codes(df: pd.DataFrame, stratify_on_type: bool) -> np.ndarray:
    label = df[config.LABEL_COL].astype(int).astype(str)
    if stratify_on_type and config.TYPE_COL in df.columns:
        key = df[config.TYPE_COL].astype(str) + "|" + label
    else:
        key = label
    return pd.factorize(key)[0]


def _peel_holdout(
    pool: pd.DataFrame, frac: float, cfg: SplitConfig, seed_offset: int
) -> np.ndarray:
    """Return positional indices (into ``pool``) of a stratified, group-disjoint
    hold-out of size ~``frac``, via the first fold of a StratifiedGroupKFold.

    Returns an empty array when ``frac <= 0`` or there are too few groups to
    split (the caller then leaves those rows in the larger partition)."""
    from sklearn.model_selection import StratifiedGroupKFold

    n_groups = pool[GROUP_COL].nunique()
    if frac <= 0 or n_groups < 2:
        return np.empty(0, dtype=int)
    k = max(2, min(round(1.0 / frac), n_groups))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg.random_state + seed_offset)
    y = _strat_codes(pool, cfg.stratify_on_type)
    groups = pool[GROUP_COL].to_numpy()
    _, holdout = next(iter(sgkf.split(np.zeros(len(pool)), y, groups)))
    return holdout


def _assign_stratified_group(df: pd.DataFrame, cfg: SplitConfig) -> tuple[pd.Series, pd.Series, list[str]]:
    """Return (split_series, cv_fold_series, warnings)."""
    from sklearn.model_selection import StratifiedGroupKFold

    warnings: list[str] = []
    idx = df.index.to_numpy()
    split = pd.Series("train", index=df.index, dtype=object)
    cv_fold = pd.Series(-1, index=df.index, dtype=int)

    n_groups = df[GROUP_COL].nunique()
    if n_groups < 5:
        warnings.append(f"Only {n_groups} duplicate-components; splits will be coarse.")

    # 1) peel off test
    pool = df
    test_pos = _peel_holdout(pool, cfg.test_type_fraction, cfg, seed_offset=0)
    test_idx = pool.index.to_numpy()[test_pos]
    split.loc[test_idx] = "test"

    # 2) peel off val from the remainder
    rest = df.loc[split == "train"]
    val_frac_adj = cfg.val_type_fraction / max(1.0 - cfg.test_type_fraction, 1e-9)
    val_pos = _peel_holdout(rest, val_frac_adj, cfg, seed_offset=1)
    val_idx = rest.index.to_numpy()[val_pos]
    split.loc[val_idx] = "val"

    # 3) CV folds on the train+val pool (group-disjoint, stratified)
    pool_mask = split.isin(("train", "val"))
    pool_df = df.loc[pool_mask]
    n_pool_groups = pool_df[GROUP_COL].nunique()
    if n_pool_groups < 2:
        warnings.append(f"Too few pool groups ({n_pool_groups}); no CV folds built.")
        return split, cv_fold, warnings
    n_folds = max(2, min(cfg.n_cv_folds, n_pool_groups))
    if n_folds < cfg.n_cv_folds:
        warnings.append(f"Only {n_folds} CV folds possible ({n_pool_groups} pool groups).")
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=cfg.random_state)
    y = _strat_codes(pool_df, cfg.stratify_on_type)
    groups = pool_df[GROUP_COL].to_numpy()
    pool_index = pool_df.index.to_numpy()
    for f, (_, va) in enumerate(sgkf.split(np.zeros(len(pool_df)), y, groups)):
        cv_fold.loc[pool_index[va]] = f
    _ = idx
    return split, cv_fold, warnings


def _assign_type_holdout(df: pd.DataFrame, cfg: SplitConfig) -> tuple[pd.Series, pd.Series, list[str]]:
    """Legacy strategy: hold out whole document types (high variance, few folds)."""
    from sklearn.model_selection import GroupKFold

    warnings = ["Using legacy type_holdout strategy: few groups, high variance."]
    types = df[config.TYPE_COL].astype(str)
    uniq = sorted(types.unique())
    rng = np.random.default_rng(cfg.random_state)
    order = rng.permutation(uniq)
    n_val = max(1, round(len(uniq) * cfg.val_type_fraction))
    n_test = max(1, round(len(uniq) * cfg.test_type_fraction))
    val_types = set(order[:n_val])
    test_types = set(order[n_val:n_val + n_test])
    split = pd.Series("train", index=df.index, dtype=object)
    split[types.isin(val_types)] = "val"
    split[types.isin(test_types)] = "test"

    cv_fold = pd.Series(-1, index=df.index, dtype=int)
    pool_mask = split.isin(("train", "val"))
    pool_df = df.loc[pool_mask]
    pool_types = pool_df[config.TYPE_COL].astype(str)
    n_folds = max(2, min(cfg.n_cv_folds, pool_types.nunique()))
    gkf = GroupKFold(n_splits=n_folds)
    pool_index = pool_df.index.to_numpy()
    for f, (_, va) in enumerate(gkf.split(np.zeros(len(pool_df)), groups=pool_types.to_numpy())):
        cv_fold.loc[pool_index[va]] = f
    return split, cv_fold, warnings


# --------------------------------------------------------------------------
# Stats + leakage assertions
# --------------------------------------------------------------------------
def _split_stats(df: pd.DataFrame, split_name: str) -> dict:
    part = df[df[SPLIT_COL] == split_name] if SPLIT_COL in df.columns else df
    if part.empty:
        return {"n": 0, "n_fraud": 0, "fraud_rate": None, "n_types": 0, "n_groups": 0}
    return {
        "n": int(len(part)),
        "n_fraud": int((part[config.LABEL_COL] == config.POSITIVE_LABEL).sum()),
        "fraud_rate": float(part[config.LABEL_COL].mean()),
        "n_types": int(part[config.TYPE_COL].nunique()) if config.TYPE_COL in part else 0,
        "n_groups": int(part[GROUP_COL].nunique()) if GROUP_COL in part else 0,
        "n_countries": int(part["country"].nunique()) if "country" in part else None,
    }


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Raise AssertionError if any duplicate-group or id crosses a partition."""
    ids = df[config.ID_COL].astype(str)
    assert ids.is_unique, "duplicate image ids in split table"

    # every group lives in exactly one split
    per_group_splits = df.groupby(GROUP_COL)[SPLIT_COL].nunique()
    bad = per_group_splits[per_group_splits > 1]
    assert bad.empty, f"{len(bad)} duplicate-groups straddle train/val/test (leakage)"

    # CV folds: val groups disjoint from train groups in every fold
    pool = df[df[SPLIT_COL].isin(("train", "val"))]
    if CV_FOLD_COL in pool.columns and (pool[CV_FOLD_COL] >= 0).any():
        for f in sorted(pool[CV_FOLD_COL].unique()):
            if f < 0:
                continue
            val_g = set(pool.loc[pool[CV_FOLD_COL] == f, GROUP_COL])
            tr_g = set(pool.loc[pool[CV_FOLD_COL] != f, GROUP_COL])
            overlap = val_g & tr_g
            assert not overlap, f"CV fold {f}: {len(overlap)} groups leak between train/val"


# --------------------------------------------------------------------------
# Public build / save / load
# --------------------------------------------------------------------------
def build_splits(
    df: pd.DataFrame | None = None,
    cfg: SplitConfig | None = None,
    duplicate_pairs: Iterable[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, SplitManifest]:
    cfg = cfg or SplitConfig()
    base = df if df is not None else io.load_labels()
    if config.TYPE_COL not in base.columns:
        raise ValueError(f"Missing {config.TYPE_COL} column in labels table")
    base = base.reset_index(drop=True)

    pairs = list(duplicate_pairs) if duplicate_pairs is not None else _duplicate_pairs_from_artifacts()
    labeled_ids = set(base[config.ID_COL].astype(str))
    pairs = [(a, b) for a, b in pairs if a in labeled_ids and b in labeled_ids]

    out = assign_groups(base, pairs)

    if cfg.strategy == "type_holdout":
        split, cv_fold, warns = _assign_type_holdout(out, cfg)
    else:
        split, cv_fold, warns = _assign_stratified_group(out, cfg)
    out[SPLIT_COL] = split
    out[CV_FOLD_COL] = cv_fold

    assert_no_leakage(out)  # fail fast on any leakage

    n_folds_assigned = int(cv_fold.max()) + 1 if (cv_fold >= 0).any() else 0
    manifest = SplitManifest(config=cfg, strategy=cfg.strategy, n_cv_folds=n_folds_assigned)
    manifest.n_groups = int(out[GROUP_COL].nunique())
    manifest.n_nontrivial_groups = int((out.groupby(GROUP_COL).size() > 1).sum())
    manifest.warnings = warns

    for split_name in ("train", "val", "test"):
        manifest.split_stats[split_name] = _split_stats(out, split_name)
        setattr(manifest, f"n_{split_name}", manifest.split_stats[split_name]["n"])

    # per-split type assignment summary (which types appear where, and how much)
    if config.TYPE_COL in out.columns:
        ta: dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            for t in out.loc[out[SPLIT_COL] == split_name, config.TYPE_COL].astype(str).unique():
                ta.setdefault(t, "")
                ta[t] = (ta[t] + "," + split_name).strip(",")
        manifest.type_assignments = ta

    # CV fold metadata (no id lists) + fraud-rate balance
    pool = out[out[SPLIT_COL].isin(("train", "val"))]
    fold_rates: list[float] = []
    for f in sorted(x for x in pool[CV_FOLD_COL].unique() if x >= 0):
        va = pool[pool[CV_FOLD_COL] == f]
        tr = pool[pool[CV_FOLD_COL] != f]
        fold_rates.append(float(va[config.LABEL_COL].mean()))
        manifest.cv_folds.append({
            "fold": int(f),
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "val_fraud_rate": float(va[config.LABEL_COL].mean()),
            "val_types": sorted(va[config.TYPE_COL].astype(str).unique().tolist()) if config.TYPE_COL in va else [],
        })
    manifest.fold_fraud_rates = fold_rates

    # leave-one-type-out metadata (group-aware; matches iter_type_holdout)
    if config.TYPE_COL in out.columns:
        for t, tr, held in iter_type_holdout(out):
            manifest.type_holdout_folds.append({
                "type": t,
                "n_train": int(len(tr)),
                "n_val": int(len(held)),
                "val_fraud_rate": float(held[config.LABEL_COL].mean()),
            })

    return out, manifest


def save_splits(df: pd.DataFrame, manifest: SplitManifest, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else config.SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        config.ID_COL, config.PATH_COL, config.LABEL_COL, config.IS_DIGITAL_COL,
        config.TYPE_COL, "country", "doc_type", GROUP_COL, SPLIT_COL, CV_FOLD_COL,
    ]
    cols = [c for c in cols if c in df.columns]
    slim = df[cols].copy()
    slim.to_csv(out_dir / "labeled_with_split.csv", index=False)
    for split_name in ("train", "val", "test"):
        slim[slim[SPLIT_COL] == split_name].to_csv(out_dir / f"{split_name}.csv", index=False)

    payload = asdict(manifest)
    payload["config"] = asdict(manifest.config)
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, default=str))
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
    known = set(SplitConfig.__dataclass_fields__)
    raw["config"] = SplitConfig(**{k: v for k, v in raw.get("config", {}).items() if k in known})
    mknown = set(SplitManifest.__dataclass_fields__)
    return SplitManifest(**{k: v for k, v in raw.items() if k in mknown})


def iter_cv_folds(df: pd.DataFrame | None = None) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield ``(train_df, val_df)`` per CV fold, derived from the ``cv_fold`` column."""
    table = df if df is not None else load_splits()
    pool = table[table[SPLIT_COL].isin(("train", "val"))]
    folds = sorted(int(f) for f in pool[CV_FOLD_COL].unique() if f >= 0)
    for f in folds:
        yield (
            pool[pool[CV_FOLD_COL] != f].copy(),
            pool[pool[CV_FOLD_COL] == f].copy(),
        )


def iter_type_holdout(df: pd.DataFrame | None = None) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(type, train_df, val_df)`` for leave-one-type-out evaluation.

    Leakage-safe: ``val`` is the rows of the held type, and ``train`` excludes
    *every duplicate group that touches the held type* (not merely the held type's
    rows). A duplicate component spanning two types is therefore never split
    across train/val.
    """
    table = df if df is not None else load_splits()
    types = table[config.TYPE_COL].astype(str)
    has_groups = GROUP_COL in table.columns
    for t in sorted(types.unique()):
        held = table[types == t]
        if has_groups:
            tainted = set(held[GROUP_COL])
            train = table[(types != t) & (~table[GROUP_COL].isin(tainted))]
        else:
            train = table[types != t]
        yield t, train.copy(), held.copy()
