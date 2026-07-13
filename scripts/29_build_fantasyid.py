#!/usr/bin/env python
"""Build a FantasyID labeled frame (id,label,abs_path) by path-keyword inference.

Robust/best-effort: scans the extracted FantasyID tree, labels images whose path
clearly indicates attack (1) or bonafide (0); skips ambiguous ones. Exits nonzero
if too few confident labels (so the chain skips FantasyID).

    python3 scripts/29_build_fantasyid.py --root data/aux/fantasyid --out artifacts/fantasyid.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config  # noqa: E402

ATTACK = ("attack", "fake", "morph", "manip", "inpaint", "swap", "forg", "tamper", "spoof")
BONA = ("bonafide", "bona_fide", "bona-fide", "genuine", "/real", "live", "original", "positive", "pristine")
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def label_for(path_str: str):
    p = path_str.lower()
    a = any(k in p for k in ATTACK)
    b = any(k in p for k in BONA)
    if a and not b:
        return 1
    if b and not a:
        return 0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/aux/fantasyid")
    ap.add_argument("--out", default="artifacts/fantasyid.csv")
    ap.add_argument("--min-rows", type=int, default=200)
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for f in root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in EXT:
            continue
        lab = label_for(str(f))
        if lab is None:
            continue
        rows.append({config.ID_COL: f"fantasyid_{f.stem}", config.LABEL_COL: lab, "abs_path": str(f)})

    df = pd.DataFrame(rows).drop_duplicates(config.ID_COL)
    if len(df) < args.min_rows:
        print(f"[fid] only {len(df)} confidently-labeled images (<{args.min_rows}); SKIP FantasyID")
        sys.exit(1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[fid] wrote {args.out}: {len(df):,} rows "
          f"(attack={int(df[config.LABEL_COL].sum()):,} bonafide={int((df[config.LABEL_COL]==0).sum()):,})")


if __name__ == "__main__":
    main()
