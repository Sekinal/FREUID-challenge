"""Scalable exact + near-duplicate detection for leakage-safe splits.

The whole point of dedup here is *grouping*: near-duplicate images (a genuine
document and its tampered twin, or the same blank template shot twice) must land
in the same CV fold / train-test partition, or the validation score leaks.

Design:
  - Exact duplicates via MD5 of file bytes (cheap, catches byte-identical files).
  - Near-duplicates via 64-bit perceptual hash (pHash). To avoid the O(n^2)
    all-pairs comparison on the full release (~70k images), candidate pairs are
    generated with LSH banding: a pair within Hamming distance ``threshold`` is
    guaranteed to collide in at least one of ``threshold + 1`` equal-width bands
    (pigeonhole), so we only verify within-bucket candidates.
  - Confirmed exact + near pairs are unioned into connected components; the
    component id is the leakage-safe grouping key used by ``freuid.splits``.

Hashing is parallelised with a process pool; JPEG decode is the bottleneck, so
images are drafted down before hashing (consistently, so distances stay stable).
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# NOTE: imagehash / PIL are imported lazily inside the hashing functions so that
# importing this module for UnionFind / component logic (e.g. from freuid.splits)
# does not require the image stack.

# 256-bit pHash (hash_size=16). Calibrated on the full release: a 64-bit pHash
# is too coarse for these same-template ID documents (distinct docs collide even
# at distance 0), whereas at 256 bits true near-dup twins sit at distance <=~10
# while unrelated same-type docs stay far apart.
HASH_SIZE = 16
PHASH_BITS = HASH_SIZE * HASH_SIZE  # 256
DEFAULT_THRESHOLD = 10
# Buckets up to this size are verified exhaustively (vectorised). A bucket above
# this is pathological (a near-constant LSH band across tens of thousands of
# images); rather than silently skip it — which would break the pigeonhole
# completeness guarantee and let a near-dup leak — we mark the result INCOMPLETE
# and the caller fails closed. With real pHashes this is never hit.
HARD_CAP = 50_000


@dataclass
class DedupResult:
    n_images: int
    phash_threshold: int
    exact_duplicate_groups: list[list[str]] = field(default_factory=list)
    near_duplicate_pairs: list[dict] = field(default_factory=list)
    components: dict[str, list[str]] = field(default_factory=dict)
    skipped_buckets: int = 0
    n_candidate_pairs: int = 0
    incomplete: bool = False  # True if any bucket exceeded HARD_CAP (guarantee voided)

    @property
    def n_exact_duplicate_groups(self) -> int:
        return len(self.exact_duplicate_groups)

    @property
    def n_near_duplicate_pairs(self) -> int:
        return len(self.near_duplicate_pairs)

    @property
    def n_nontrivial_components(self) -> int:
        return sum(1 for ids in self.components.values() if len(ids) > 1)

    def to_json(self, label_map: dict[str, int] | None = None) -> dict:
        sizes = sorted((len(v) for v in self.components.values()), reverse=True)
        payload = {
            "n_images": self.n_images,
            "phash_threshold": self.phash_threshold,
            "n_exact_duplicate_groups": self.n_exact_duplicate_groups,
            "exact_duplicate_groups": self.exact_duplicate_groups,
            "n_near_duplicate_pairs": self.n_near_duplicate_pairs,
            "near_duplicate_pairs": self.near_duplicate_pairs,
            "n_candidate_pairs": self.n_candidate_pairs,
            "n_nontrivial_components": self.n_nontrivial_components,
            "largest_component": sizes[0] if sizes else 0,
            "component_size_histogram": _size_histogram(sizes),
            "skipped_buckets": self.skipped_buckets,
            "incomplete": self.incomplete,
        }
        if label_map is not None:
            payload["n_label_conflicts"] = sum(
                1 for p in self.near_duplicate_pairs if p.get("label_conflict")
            )
        return payload


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
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def connected(self, a: str, b: str) -> bool:
        return self.find(a) == self.find(b)

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for x in self.parent:
            groups[self.find(x)].append(x)
        return dict(groups)


def _size_histogram(sizes: Sequence[int]) -> dict[str, int]:
    hist: dict[str, int] = defaultdict(int)
    for s in sizes:
        if s <= 1:
            key = "1"
        elif s == 2:
            key = "2"
        elif s <= 5:
            key = "3-5"
        elif s <= 10:
            key = "6-10"
        else:
            key = "11+"
        hist[key] += 1
    return dict(hist)


def _hash_one(args: tuple[str, int, int]) -> tuple[str, str, int] | None:
    """Worker: return (stem, md5_hex, phash_int) for one image, or None.

    ``phash_int`` is the full ``hash_size**2``-bit perceptual hash packed into a
    Python int (arbitrary precision), MSB-first.
    """
    import imagehash
    from PIL import Image

    path_str, draft_size, hash_size = args
    path = Path(path_str)
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        md5_hex = h.hexdigest()
        with Image.open(path) as im:
            im.draft("L", (draft_size, draft_size))  # fast JPEG downscale-decode
            ph = imagehash.phash(im.convert("L"), hash_size=hash_size)
    except Exception:
        return None
    value = 0
    for bit in ph.hash.flatten():
        value = (value << 1) | int(bool(bit))
    return path.stem, md5_hex, value


def _popcount_rows(arr) -> "np.ndarray":
    """Row-wise popcount of a (m, W) uint64 array -> (m,) int64 set-bit counts."""
    import numpy as np

    a = np.ascontiguousarray(arr, dtype=np.uint64)
    if a.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    flat = a.reshape(a.shape[0], -1)
    return np.unpackbits(flat.view(np.uint8), axis=1).sum(axis=1).astype(np.int64)


def _int_to_words(v: int, n_words: int) -> list[int]:
    """Split a Python int into ``n_words`` little-endian 64-bit words."""
    mask = (1 << 64) - 1
    return [(v >> (64 * k)) & mask for k in range(n_words)]


def _band_masks(n_bits: int, n_bands: int) -> list[tuple[int, int]]:
    """Return (shift, mask) per band, splitting ``n_bits`` as evenly as possible."""
    base, extra = divmod(n_bits, n_bands)
    masks: list[tuple[int, int]] = []
    shift = 0
    for b in range(n_bands):
        width = base + (1 if b < extra else 0)
        masks.append((shift, (1 << width) - 1))
        shift += width
    return masks


def compute_hashes(
    paths: Iterable[Path],
    max_workers: int | None = None,
    draft_size: int = 128,
    hash_size: int = HASH_SIZE,
) -> tuple[list[str], dict[str, str], dict[str, int]]:
    """Parallel hash of all images. Returns (stems, md5_by_stem, phash_by_stem).

    ``phash_by_stem`` maps stem -> full ``hash_size**2``-bit pHash as a Python int.
    """
    path_strs = [str(p) for p in paths]
    args = [(s, draft_size, hash_size) for s in path_strs]
    md5_by: dict[str, str] = {}
    phash_by: dict[str, int] = {}
    stems: list[str] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(_hash_one, args, chunksize=64):
            if res is None:
                continue
            stem, md5_hex, value = res
            stems.append(stem)
            md5_by[stem] = md5_hex
            phash_by[stem] = value
    return stems, md5_by, phash_by


def find_duplicates(
    stems: Sequence[str],
    md5_by: dict[str, str],
    phash_by: dict[str, int],
    threshold: int = DEFAULT_THRESHOLD,
    label_map: dict[str, int] | None = None,
    bits: int = PHASH_BITS,
) -> DedupResult:
    """Build exact groups, near-dup pairs (LSH-banded), and components."""
    uf = UnionFind()
    for s in stems:
        uf.add(s)

    # --- exact duplicates (identical bytes) ---
    by_md5: dict[str, list[str]] = defaultdict(list)
    for s in stems:
        by_md5[md5_by[s]].append(s)
    exact_groups = [sorted(ids) for ids in by_md5.values() if len(ids) > 1]
    for group in exact_groups:
        for other in group[1:]:
            uf.union(group[0], other)

    # --- near duplicates via LSH banding (pigeonhole-complete) ---
    # A pair within Hamming<=threshold collides in >=1 of (threshold+1) equal
    # bands, so verifying within-bucket candidates is complete -- UNLESS a bucket
    # is skipped, which we refuse to do silently (HARD_CAP -> incomplete flag).
    import numpy as np

    stem_arr = list(stems)
    n_words = (bits + 63) // 64
    full = [phash_by[s] for s in stem_arr]
    hash_words = np.array([_int_to_words(v, n_words) for v in full], dtype=np.uint64)
    hash_words = hash_words.reshape(len(stem_arr), n_words)

    n_bands = threshold + 1
    masks = _band_masks(bits, n_bands)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in range(len(stem_arr)):
        v = full[i]
        for bi, (shift, mask) in enumerate(masks):
            buckets[(bi, (v >> shift) & mask)].append(i)

    seen_pairs: set[tuple[str, str]] = set()
    near_pairs: list[dict] = []
    skipped = 0
    n_candidates = 0
    for members in buckets.values():
        m = len(members)
        if m < 2:
            continue
        if m > HARD_CAP:
            skipped += 1
            continue
        mem = np.asarray(members, dtype=np.int64)
        hs = hash_words[mem]  # (m, W)
        for ii in range(m - 1):
            x = hs[ii] ^ hs[ii + 1:]   # (m-ii-1, W)
            d = _popcount_rows(x)
            n_candidates += int(d.size)
            hits = np.nonzero(d <= threshold)[0]
            for h in hits:
                a = stem_arr[members[ii]]
                b = stem_arr[members[ii + 1 + int(h)]]
                pair = (a, b) if a < b else (b, a)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                uf.union(a, b)
                rec = {"a": pair[0], "b": pair[1], "phash_dist": int(d[h])}
                if label_map is not None:
                    la, lb = label_map.get(pair[0]), label_map.get(pair[1])
                    rec["label_a"], rec["label_b"] = la, lb
                    rec["label_conflict"] = (la is not None and lb is not None and la != lb)
                near_pairs.append(rec)
    near_pairs.sort(key=lambda r: r["phash_dist"])

    return DedupResult(
        n_images=len(stems),
        phash_threshold=threshold,
        exact_duplicate_groups=exact_groups,
        near_duplicate_pairs=near_pairs,
        components=uf.components(),
        skipped_buckets=skipped,
        n_candidate_pairs=n_candidates,
        incomplete=skipped > 0,
    )


def component_map(result: DedupResult) -> dict[str, str]:
    """Map each image stem -> stable component id (``dup_XXXXXX``)."""
    mapping: dict[str, str] = {}
    roots = sorted(result.components.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for i, (_root, members) in enumerate(roots):
        cid = f"dup_{i:06d}"
        for m in members:
            mapping[m] = cid
    return mapping
