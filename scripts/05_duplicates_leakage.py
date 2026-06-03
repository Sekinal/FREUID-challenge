#!/usr/bin/env python
"""Exact + near-duplicate detection and a train/test leakage scaffold.

- Exact duplicates via MD5 of file bytes.
- Near-duplicates via perceptual hashes (pHash, dHash, aHash); pairs with
  Hamming distance <= threshold are flagged. Conflicting labels on a duplicate
  pair are highlighted (a real data-quality risk).
- Leakage: if a separate test image directory ever ships, the same pHash index
  is reused to find train<->test overlaps. With only the public sample present,
  this is reported as "test set hidden".

Writes ``artifacts/duplicates.json`` and a montage of the closest pairs.

    uv run scripts/05_duplicates_leakage.py
"""
from __future__ import annotations

import hashlib
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, io, viz  # noqa: E402

PHASH_THRESHOLD = 8  # Hamming distance on a 64-bit pHash considered "near-dup"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hashes(path: Path) -> dict:
    im = Image.open(path).convert("RGB")
    return {
        "phash": imagehash.phash(im),
        "dhash": imagehash.dhash(im),
        "ahash": imagehash.average_hash(im),
    }


def main() -> None:
    viz.set_style()
    labels = io.load_labels()
    label_map = dict(zip(labels[config.ID_COL].astype(str), labels[config.LABEL_COL]))

    paths = io.list_images()
    if not paths:
        sys.exit("No images found. Run scripts/00_download.py first.")

    # exact duplicates
    by_md5: dict[str, list[str]] = defaultdict(list)
    h = {}
    for p in paths:
        by_md5[md5(p)].append(p.stem)
        h[p.stem] = hashes(p)
    exact_groups = [ids for ids in by_md5.values() if len(ids) > 1]

    # near duplicates (all pairs — fine for sample sizes; swap to BK-tree at scale)
    ids = [p.stem for p in paths]
    near = []
    for a, b in itertools.combinations(ids, 2):
        d = h[a]["phash"] - h[b]["phash"]
        if d <= PHASH_THRESHOLD:
            near.append({
                "a": a, "b": b, "phash_dist": int(d),
                "dhash_dist": int(h[a]["dhash"] - h[b]["dhash"]),
                "label_a": label_map.get(a), "label_b": label_map.get(b),
                "label_conflict": label_map.get(a) != label_map.get(b),
            })
    near.sort(key=lambda r: r["phash_dist"])

    result = {
        "n_images": len(paths),
        "phash_threshold": PHASH_THRESHOLD,
        "n_exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_groups": exact_groups,
        "n_near_duplicate_pairs": len(near),
        "near_duplicate_pairs": near,
        "n_label_conflicts": sum(r["label_conflict"] for r in near),
        "leakage": {
            "test_images_present": False,
            "note": "Public release ships only train_sample; the hidden test set "
                    "cannot be checked locally. Re-run when test images are available.",
        },
    }
    out = io.save_json("duplicates.json", result)

    # montage of the closest near-duplicate pairs
    top = near[:6]
    if top:
        idx = io.image_index()
        fig, axes = viz.plt.subplots(len(top), 2, figsize=(5, 2.4 * len(top)))
        axes = axes.reshape(len(top), 2)
        for row, pair in zip(axes, top):
            for ax, key in zip(row, ("a", "b")):
                im = Image.open(idx[pair[key]]).convert("RGB")
                im.thumbnail((220, 220))
                ax.imshow(im)
                ax.axis("off")
                ax.set_title(f"{pair[key][:8]} (lab={pair['label_'+key]})", fontsize=7)
            row[0].set_ylabel(f"pHash={pair['phash_dist']}", fontsize=8)
        fig.suptitle("Closest near-duplicate pairs", fontweight="bold")
        viz.save_fig(fig, "duplicate_pairs.png")

    print(f"images={len(paths)}  exact_groups={len(exact_groups)}  "
          f"near_pairs={len(near)}  label_conflicts={result['n_label_conflicts']}")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
