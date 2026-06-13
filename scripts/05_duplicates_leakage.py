#!/usr/bin/env python
"""Exact + near-duplicate detection at full-dataset scale (leakage units).

Scales to the full release (~70k images) via parallel hashing and LSH-banded
near-duplicate candidate generation (see ``freuid.dedup``). Writes
``artifacts/duplicates.json`` (consumed by ``freuid.splits`` to keep near-dups
in the same partition) and a montage of the closest pairs. When ``public_test``
images are present, also reports train<->test near-duplicate leakage.

    uv run scripts/05_duplicates_leakage.py
    uv run scripts/05_duplicates_leakage.py --threshold 8 --workers 16 --limit 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, dedup, io  # noqa: E402  (viz imported lazily for the montage)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-scale duplicate / leakage detection.")
    p.add_argument("--threshold", type=int, default=dedup.DEFAULT_THRESHOLD,
                   help="Max pHash Hamming distance for a near-duplicate (256-bit).")
    p.add_argument("--hash-size", type=int, default=dedup.HASH_SIZE,
                   help="pHash side length; bits = hash_size**2 (16 -> 256-bit).")
    p.add_argument("--workers", type=int, default=None, help="Hashing process pool size.")
    p.add_argument("--limit", type=int, default=None, help="Cap #images (debug).")
    p.add_argument("--draft", type=int, default=128, help="JPEG draft decode size.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = io.load_labels()
    label_map = {
        str(k): int(v)
        for k, v in zip(labels[config.ID_COL].astype(str), labels[config.LABEL_COL])
        if v == v  # drop NaN
    }

    paths = io.list_images()
    if not paths:
        sys.exit("No images found. Run scripts/00_download.py first.")
    if args.limit:
        paths = paths[: args.limit]

    bits = args.hash_size * args.hash_size
    print(f"[hash] {len(paths):,} images (workers={args.workers or 'auto'}, draft={args.draft}, "
          f"hash_size={args.hash_size}/{bits}-bit, threshold={args.threshold})")
    stems, md5_by, phash_by = dedup.compute_hashes(
        paths, max_workers=args.workers, draft_size=args.draft, hash_size=args.hash_size)
    print(f"[hash] done: {len(stems):,} hashed ({len(paths) - len(stems)} failed)")

    result = dedup.find_duplicates(
        stems, md5_by, phash_by, threshold=args.threshold, label_map=label_map, bits=bits,
    )
    if result.incomplete:
        sys.exit(
            f"[FATAL] {result.skipped_buckets} LSH bucket(s) exceeded the hard cap; "
            "the near-duplicate guarantee is voided and downstream splits could leak. "
            "Re-run with a smaller --threshold or raise dedup.HARD_CAP after inspection."
        )
    payload = result.to_json(label_map=label_map)

    # --- optional train <-> public_test leakage ---
    test_root = config.PUBLIC_TEST_DIR
    if test_root.exists():
        test_paths = [p for p in test_root.rglob("*") if p.suffix.lower() in config.IMAGE_EXTENSIONS]
        if args.limit:
            test_paths = test_paths[: args.limit]
        print(f"[leakage] hashing {len(test_paths):,} public_test images")
        t_stems, _t_md5, t_phash = dedup.compute_hashes(
            test_paths, max_workers=args.workers, draft_size=args.draft, hash_size=args.hash_size)
        train_set = set(stems)
        all_stems = list(stems) + [s for s in t_stems if s not in train_set]
        merged_phash = dict(phash_by)
        for s in t_stems:
            merged_phash.setdefault(s, t_phash[s])
        cross = dedup.find_duplicates(
            all_stems, {s: "" for s in all_stems}, merged_phash, threshold=args.threshold, bits=bits)
        is_test = set(t_stems)
        leak_pairs = [
            p for p in cross.near_duplicate_pairs
            if (p["a"] in is_test) != (p["b"] in is_test)  # exactly one side is test
        ]
        payload["train_test_leakage"] = {
            "n_public_test": len(t_stems),
            "n_train_test_near_pairs": len(leak_pairs),
            "examples": leak_pairs[:25],
        }
        print(f"[leakage] train<->public_test near-dup pairs: {len(leak_pairs):,}")
    else:
        payload["train_test_leakage"] = {"note": "public_test not present"}

    out = io.save_json("duplicates.json", payload)

    # montage of the closest near-duplicate pairs (skipped if matplotlib absent)
    top = result.near_duplicate_pairs[:6]
    try:
        from freuid import viz
    except ModuleNotFoundError:
        viz = None
        print("[skip] montage: matplotlib not installed")
    if top and viz is not None:
        viz.set_style()
        idx = io.image_index()
        fig, axes = viz.plt.subplots(len(top), 2, figsize=(5, 2.4 * len(top)))
        axes = axes.reshape(len(top), 2)
        for row, pair in zip(axes, top):
            for ax, key in zip(row, ("a", "b")):
                stem = pair[key]
                if stem in idx:
                    im = Image.open(idx[stem]).convert("RGB")
                    im.thumbnail((220, 220))
                    ax.imshow(im)
                ax.axis("off")
                ax.set_title(f"{stem[:8]} (lab={pair.get('label_' + key)})", fontsize=7)
            row[0].set_ylabel(f"pHash={pair['phash_dist']}", fontsize=8)
        fig.suptitle("Closest near-duplicate pairs", fontweight="bold")
        viz.save_fig(fig, "duplicate_pairs.png")

    print(
        f"[done] images={result.n_images:,}  exact_groups={result.n_exact_duplicate_groups:,}  "
        f"near_pairs={result.n_near_duplicate_pairs:,}  "
        f"nontrivial_components={result.n_nontrivial_components:,}  "
        f"largest={payload['largest_component']}  "
        f"candidates={result.n_candidate_pairs:,}  skipped_buckets={result.skipped_buckets}"
    )
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
