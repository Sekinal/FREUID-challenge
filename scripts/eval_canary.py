"""Canary tripwire: score a checkpoint on the ONLY in-domain captured FREUID
images we have (the 20 is_digital==False rows, ~90% Mauritius). Any model that
can't rank these is dead on the private set (which emphasises captured docs).

NOTE ON CONTAMINATION: valid only for checkpoints that did NOT train on these
ids. The Mauritius-LOTO model (holds out Mauritius) sees ~18 of the 20 as clean
OOD test; a --train-all model memorised them, so its canary number is meaningless.

    ocr-not-needed; uses the torch training env:
    python3 scripts/eval_canary.py --checkpoint runs/robust_val_maurLOTO.pt
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, data, fusion  # noqa: E402
from freuid.metrics import freuid_score  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def captured_frame():
    df = pd.read_csv(config.DATA_DIR / "extracted" / "train_labels.csv") \
        if (config.DATA_DIR / "extracted" / "train_labels.csv").exists() \
        else pd.read_csv("/root/freuid/data/extracted/train_labels.csv")
    cap = df[df["is_digital"] == False].copy()
    root = Path("/root/freuid/data/extracted/train/train")
    cap["abs_path"] = cap["id"].map(lambda i: str(root / f"{i}.jpeg"))
    cap[config.LABEL_COL] = cap["label"].astype(int)
    return cap.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    cfgd = ckpt["cfg"]
    backbone = cfgd.get("backbone", "tf_efficientnetv2_m.in21k_ft_in1k")
    use_fusion = cfgd.get("use_fusion", False)
    img_size = cfgd.get("img_size", 384)
    model = fusion.FusionModel(backbone, use_fusion=use_fusion).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cap = captured_frame()
    feats = np.zeros((len(cap), fusion.FUSION_DIM), np.float32)
    tf = data.build_transforms(data.DataConfig(img_size=img_size, train=False))
    loader = DataLoader(fusion.FusionDataset(cap, feats, tf),
                        batch_size=args.batch_size, num_workers=4)
    scores = []
    with torch.no_grad():
        for x, ff, _ in loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logit = model(x.to(dev).to(memory_format=torch.channels_last), ff.to(dev))
            scores.append(torch.sigmoid(logit.float()).cpu().numpy())
    s = np.concatenate(scores)
    cap["score"] = s
    y = cap[config.LABEL_COL].to_numpy()

    print(f"\n=== CANARY: {len(cap)} captured images | ckpt={Path(args.checkpoint).name} ===")
    for _, r in cap.sort_values("score").iterrows():
        flag = "" if (r["score"] > 0.5) == bool(r[config.LABEL_COL]) else "  <-- WRONG"
        print(f"  {r['type']:14s} label={int(r[config.LABEL_COL])} score={r['score']:.3f}{flag}")
    gm = s[y == 1].mean(); bm = s[y == 0].mean()
    print(f"\n  fraud mean={gm:.3f}  genuine mean={bm:.3f}  separation={gm-bm:+.3f}")
    if len(set(y)) == 2:
        print(f"  AUC={roc_auc_score(y, s):.3f}   FREUID={freuid_score(y, s).freuid:.4f}")
    print(f"  TRIPWIRE: {'PASS' if gm - bm > 0.15 else 'FAIL'} (captured fraud must score above captured genuine)")


if __name__ == "__main__":
    main()
