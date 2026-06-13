"""Tests for the dedup engine (freuid/dedup.py).

Operates on synthetic 64-bit pHash integers, so no images / imagehash needed.
Validates LSH-banding completeness (the pigeonhole guarantee), exact grouping,
and union-find component formation. Runs under pytest or directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import dedup  # noqa: E402


def _flip_bits(value: int, n: int, rng) -> int:
    positions = rng.choice(64, size=n, replace=False)
    for p in positions:
        value ^= (1 << int(p))
    return value


def test_union_find_basic():
    uf = dedup.UnionFind()
    for x in "abcde":
        uf.add(x)
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.connected("a", "c")
    assert not uf.connected("a", "d")
    comps = uf.components()
    assert any(set(v) == {"a", "b", "c"} for v in comps.values())


def test_exact_duplicate_grouping():
    stems = ["x", "y", "z"]
    md5 = {"x": "AA", "y": "AA", "z": "BB"}  # x,y identical bytes
    phash = {"x": 1, "y": 1 << 40, "z": 999}  # different perceptual hashes
    res = dedup.find_duplicates(stems, md5, phash, threshold=0)
    assert res.n_exact_duplicate_groups == 1
    # x and y must end up in the same component via the exact-dup union
    cmap = dedup.component_map(res)
    assert cmap["x"] == cmap["y"]


def test_lsh_finds_all_pairs_within_threshold():
    """Brute-force vs LSH must agree on the near-duplicate set (completeness)."""
    rng = np.random.default_rng(0)
    threshold = 8
    n = 300
    base = rng.integers(0, 1 << 63, size=n, dtype=np.int64).astype(object)
    phash = {f"s{i}": int(base[i]) for i in range(n)}
    # inject some controlled near-dups at known distances
    for i in range(0, 40, 2):
        phash[f"s{i+1}"] = _flip_bits(phash[f"s{i}"], rng.integers(0, threshold + 1), rng)
    stems = list(phash)
    md5 = {s: s for s in stems}  # all unique bytes

    res = dedup.find_duplicates(stems, md5, phash, threshold=threshold)
    lsh_pairs = {(p["a"], p["b"]) for p in res.near_duplicate_pairs}

    # brute force ground truth
    vals = {s: phash[s] for s in stems}
    brute = set()
    for i in range(n):
        for j in range(i + 1, n):
            a, b = stems[i], stems[j]
            if bin(vals[a] ^ vals[b]).count("1") <= threshold:
                brute.add((a, b) if a < b else (b, a))

    assert lsh_pairs == brute, f"LSH missed {brute - lsh_pairs}, extra {lsh_pairs - brute}"
    assert not res.incomplete


def test_pairs_beyond_threshold_not_linked():
    phash = {"a": 0, "b": (1 << 20) - 1}  # 20 bits set -> distance 20
    res = dedup.find_duplicates(["a", "b"], {"a": "1", "b": "2"}, phash, threshold=8)
    assert res.n_near_duplicate_pairs == 0
    cmap = dedup.component_map(res)
    assert cmap["a"] != cmap["b"]


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
