#!/usr/bin/env python3
"""Day-13 runbook step 3: build the two FINAL full submissions.

Combines (a) the frozen public-test predictions already on Kaggle and (b) the
fresh private-test predictions from infer.py into complete sample_submission
CSVs, then prints sha256 checksums for the README mapping.

    python3 scripts/day13_build_submissions.py \
        --private-preds-public submissions/priv_public.csv \
        --private-preds-robust submissions/priv_robust.csv
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def build(sample, frozen_public_csv, private_csv, out_path):
    sub = pd.read_csv(sample)
    id_col, label_col = sub.columns[0], sub.columns[1]
    pub = pd.read_csv(frozen_public_csv).set_index("id")
    pubcol = pub.columns[0]
    pub = pub[pub[pubcol] != 0.5][pubcol]          # only the real public rows
    priv = pd.read_csv(private_csv).set_index("id")["label"]
    overlap = pub.index.intersection(priv.index)
    assert len(overlap) == 0, f"public/private id overlap: {len(overlap)}"
    merged = pd.concat([pub, priv])
    sub[label_col] = sub[id_col].map(merged).fillna(0.5)
    n_real = (sub[label_col] != 0.5).sum()
    sub.to_csv(out_path, index=False)
    sha = hashlib.sha256(Path(out_path).read_bytes()).hexdigest()
    print(f"[build] {out_path}: {len(sub):,} rows, {n_real:,} predicted "
          f"({len(pub):,} public + {len(priv):,} private), fill={len(sub)-n_real:,}")
    print(f"[sha256] {sha}  {Path(out_path).name}")
    return sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/extracted/sample_submission.csv")
    ap.add_argument("--frozen-public", default="submissions/sub_bm1024.csv",
                    help="submission whose public rows are frozen for the PUBLIC pick")
    ap.add_argument("--frozen-robust", default="submissions/sub_slot2v3_1024_ens3.csv")
    ap.add_argument("--private-preds-public", required=True)
    ap.add_argument("--private-preds-robust", required=True)
    args = ap.parse_args()

    build(args.sample, args.frozen_public, args.private_preds_public,
          "submissions/FINAL_public_bm1024.csv")
    build(args.sample, args.frozen_robust, args.private_preds_robust,
          "submissions/FINAL_robust_slot2v3.csv")
    print("\nNext: kaggle competitions submit each CSV, then SELECT BOTH as "
          "final picks on the Kaggle submissions page, then paste the sha256 "
          "values into docker/README.md and the reply template.")


if __name__ == "__main__":
    main()
