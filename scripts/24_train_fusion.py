#!/usr/bin/env python
"""Train the fusion model: EffNetV2 + frozen-CLIP + region-noise, with IDNet aux.

Leakage-safe splits (scripts/09) + optional IDNet aux mix-in + optional
leave-one-type-out eval. Frozen CLIP(768)+region(27) features are cached by id
(artifacts/fusion_features.npz) and extracted on demand. Use --no-fusion for the
ablation (backbone only) to measure whether the streams actually help.

    uv run scripts/24_train_fusion.py --smoke                       # fast pipeline check
    uv run scripts/24_train_fusion.py --aux --epochs 3              # in-dist leakage-safe
    uv run scripts/24_train_fusion.py --aux --loto-type MAURITIUS/ID --epochs 3
    uv run scripts/24_train_fusion.py --aux --loto-type MAURITIUS/ID --no-fusion --epochs 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import aux_data, config, fusion, io, validation  # noqa: E402

CACHE = config.ARTIFACTS_DIR / "fusion_features.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--backbone", default="tf_efficientnetv2_m.in21k_ft_in1k")
    p.add_argument("--aux", action="store_true", help="mix in IDNet-2025 aux data")
    p.add_argument("--aux-roots", default="data/aux/idnet2025")
    p.add_argument("--max-aux", type=int, default=30000)
    p.add_argument("--loto-type", default="", help="hold this FREUID type out of train -> test")
    p.add_argument("--no-fusion", action="store_true", help="ablation: backbone only")
    p.add_argument("--loss", default="focal", choices=["focal", "bce"])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--feat-batch", type=int, default=128)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--rebuild-splits", action="store_true")
    return p.parse_args()


def load_cache():
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return list(d["ids"].astype(str)), d["X"].astype(np.float32)
    return [], np.zeros((0, fusion.FUSION_DIM), np.float32)


def get_features(frame, args):
    """Return feats[len(frame), 795] aligned to frame rows (by id), extracting+caching misses."""
    ids = frame[config.ID_COL].astype(str).tolist()
    cache_ids, cache_X = load_cache()
    id2idx = {i: k for k, i in enumerate(cache_ids)}
    missing = sorted({i for i in ids if i not in id2idx})
    if missing:
        mframe = frame.drop_duplicates(config.ID_COL).set_index(config.ID_COL).loc[missing].reset_index()
        print(f"[feat] extracting {len(missing):,} new features...")
        Xnew, _ = fusion.extract_fusion_features(mframe["abs_path"].astype(str).tolist(),
                                                 batch_size=args.feat_batch, workers=args.workers)
        cache_ids = cache_ids + mframe[config.ID_COL].astype(str).tolist()
        cache_X = np.vstack([cache_X, Xnew])
        np.savez_compressed(CACHE, ids=np.array(cache_ids), X=cache_X)
        id2idx = {i: k for k, i in enumerate(cache_ids)}
        print(f"[feat] cache now {len(cache_ids):,} ids")
    return cache_X[[id2idx[i] for i in ids]]


def resolve(frame):
    out = io.attach_image_paths(frame)
    return out[out["path_exists"]].reset_index(drop=True)


def main():
    args = parse_args()
    pipe = validation.ValidationPipeline(rebuild=args.rebuild_splits)
    train, val, test = resolve(pipe.train), resolve(pipe.val), resolve(pipe.test)

    # leave-one-type-out: pull the held type out of train/val, make it the test set
    if args.loto_type:
        full = pd.concat([train, val, test], ignore_index=True)
        test = full[full[config.TYPE_COL] == args.loto_type].reset_index(drop=True)
        train = train[train[config.TYPE_COL] != args.loto_type].reset_index(drop=True)
        val = val[val[config.TYPE_COL] != args.loto_type].reset_index(drop=True)
        print(f"[loto] held out {args.loto_type}: test={len(test):,}  train={len(train):,}")

    if args.smoke:
        train = train.sample(min(len(train), 1500), random_state=0).reset_index(drop=True)
        val = val.sample(min(len(val), 600), random_state=0).reset_index(drop=True)
        test = test.sample(min(len(test), 600), random_state=0).reset_index(drop=True)
        args.epochs = 1

    if args.aux:
        roots = [config.REPO_ROOT / r for r in args.aux_roots.split(",")]
        aux = aux_data.load_idnet_frame(roots)
        aux = aux[aux["abs_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
        if args.max_aux and len(aux) > args.max_aux:
            aux = aux.sample(args.max_aux, random_state=0).reset_index(drop=True)
        if args.smoke:
            aux = aux.sample(min(len(aux), 800), random_state=0).reset_index(drop=True)
        train = pd.concat([train, aux], ignore_index=True)
        print(f"[aux] mixed in {len(aux):,} IDNet rows -> train={len(train):,}")

    print(f"[data] train={len(train):,} val={len(val):,} test={len(test):,} "
          f"fraud-rate train={train[config.LABEL_COL].mean():.3f}")

    train_feats = get_features(train, args)
    val_feats = get_features(val, args)
    test_feats = get_features(test, args)

    cfg = fusion.FusionConfig(backbone=args.backbone, img_size=args.img_size,
                              epochs=args.epochs, batch_size=args.batch_size,
                              use_fusion=not args.no_fusion, loss_type=args.loss,
                              num_workers=args.workers)
    tag = ("nofusion" if args.no_fusion else "fusion") + (
        f"_loto_{args.loto_type.replace('/', '-')}" if args.loto_type else "")
    eval_sets = {"test": (test, test_feats), "val": (val, val_feats)}
    res = fusion.train_fusion(train, train_feats, eval_sets, cfg, save_name=f"fusion_{tag}")
    res["tag"] = tag
    io.save_json(f"fusion_result_{tag}.json", res)
    print(f"\n[done] {tag}  best={res['best'].get('metrics')}")


if __name__ == "__main__":
    main()
