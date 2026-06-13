"""Tests for leakage-safe splitting (freuid/splits.py).

Synthetic, fast, no images. Runs under pytest *or* directly:

    uv run python tests/test_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, splits  # noqa: E402


def _frame(n_per_type: int = 400, n_types: int = 5, dup_frac: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Build a labeled frame with a known near-duplicate pair structure."""
    rng = np.random.default_rng(seed)
    types = [f"C{t}/DL" for t in range(n_types)]
    rows = []
    k = 0
    for t in types:
        for _ in range(n_per_type):
            rows.append({
                config.ID_COL: f"img{k:06d}",
                config.PATH_COL: f"{t}/img{k:06d}.jpeg",
                config.LABEL_COL: int(rng.random() < 0.4),
                config.IS_DIGITAL_COL: True,
                config.TYPE_COL: t,
                "country": t.split("/")[0],
                "doc_type": t.split("/")[1],
            })
            k += 1
    df = pd.DataFrame(rows)
    # craft duplicate pairs within type (so grouping has something to merge)
    pairs = []
    ids = df[config.ID_COL].tolist()
    n_dup = int(len(ids) * dup_frac)
    for i in range(0, n_dup, 2):
        if i + 1 < len(ids):
            pairs.append((ids[i], ids[i + 1]))
    return df, pairs


def test_no_group_leakage_across_splits_and_folds():
    df, pairs = _frame()
    out, manifest = splits.build_splits(df, cfg=splits.SplitConfig(n_cv_folds=5), duplicate_pairs=pairs)
    # build_splits already calls assert_no_leakage; call again explicitly
    splits.assert_no_leakage(out)
    # every duplicate pair is in the same split
    split_map = dict(zip(out[config.ID_COL], out[splits.SPLIT_COL]))
    for a, b in pairs:
        assert split_map[a] == split_map[b], f"dup pair {a},{b} leaked across splits"


def test_all_three_partitions_present_and_disjoint():
    df, pairs = _frame()
    out, _ = splits.build_splits(df, duplicate_pairs=pairs)
    counts = out[splits.SPLIT_COL].value_counts()
    for name in ("train", "val", "test"):
        assert counts.get(name, 0) > 0, f"empty split: {name}"
    # ids partition cleanly
    assert out[config.ID_COL].is_unique


def test_fraud_rate_is_balanced_across_cv_folds():
    df, pairs = _frame()
    out, manifest = splits.build_splits(df, cfg=splits.SplitConfig(n_cv_folds=5), duplicate_pairs=pairs)
    rates = np.asarray(manifest.fold_fraud_rates)
    assert len(rates) >= 2
    # stratified folds: fraud rate should not swing wildly
    assert rates.std() < 0.08, f"fold fraud-rate std too high: {rates.std():.3f}"


def test_cv_folds_cover_pool_without_overlap():
    df, pairs = _frame()
    out, _ = splits.build_splits(df, cfg=splits.SplitConfig(n_cv_folds=5), duplicate_pairs=pairs)
    seen = []
    for tr, va in splits.iter_cv_folds(out):
        # train/val disjoint by id
        assert set(tr[config.ID_COL]) & set(va[config.ID_COL]) == set()
        seen.append(set(va[config.ID_COL]))
    # val folds are disjoint and together cover the whole train+val pool
    pool_ids = set(out.loc[out[splits.SPLIT_COL].isin(("train", "val")), config.ID_COL])
    union = set().union(*seen)
    assert union == pool_ids
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            assert seen[i].isdisjoint(seen[j])


def test_leave_one_type_out_covers_every_type_without_group_leak():
    df, pairs = _frame(n_types=5)
    out, _ = splits.build_splits(df, duplicate_pairs=pairs)
    held_types = []
    for t, tr, va in splits.iter_type_holdout(out):
        assert set(va[config.TYPE_COL]) == {t}
        assert t not in set(tr[config.TYPE_COL])
        # no duplicate group may appear in both LOTO train and val
        assert set(tr[splits.GROUP_COL]) & set(va[splits.GROUP_COL]) == set()
        held_types.append(t)
    assert sorted(held_types) == sorted(out[config.TYPE_COL].unique())


def test_loto_excludes_cross_type_duplicate_groups():
    # force a duplicate component that spans two types, then check LOTO honours it
    df, _ = _frame(n_types=2, n_per_type=50)
    ids_by_type = {t: df.loc[df[config.TYPE_COL] == t, config.ID_COL].tolist()
                   for t in df[config.TYPE_COL].unique()}
    t0, t1 = sorted(ids_by_type)
    cross_pair = (ids_by_type[t0][0], ids_by_type[t1][0])  # links the two types
    out, _ = splits.build_splits(df, duplicate_pairs=[cross_pair])
    # the two cross-type images share a group
    g = dict(zip(out[config.ID_COL], out[splits.GROUP_COL]))
    assert g[cross_pair[0]] == g[cross_pair[1]]
    for t, tr, va in splits.iter_type_holdout(out):
        assert set(tr[splits.GROUP_COL]) & set(va[splits.GROUP_COL]) == set()


def test_deterministic_given_seed():
    df, pairs = _frame()
    a, _ = splits.build_splits(df, cfg=splits.SplitConfig(random_state=7), duplicate_pairs=pairs)
    b, _ = splits.build_splits(df, cfg=splits.SplitConfig(random_state=7), duplicate_pairs=pairs)
    assert (a[splits.SPLIT_COL].to_numpy() == b[splits.SPLIT_COL].to_numpy()).all()
    assert (a[splits.CV_FOLD_COL].to_numpy() == b[splits.CV_FOLD_COL].to_numpy()).all()


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
