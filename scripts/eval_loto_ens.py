"""Evalua checkpoints (y su ensemble prob-avg y rank-avg) sobre las filas de un
type LOTO de train_labels.csv. Mide si el seed-ensemble mejora el OOD."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, data, fusion
from freuid.metrics import freuid_score

def predict(ckpt_path, df, dev):
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfgd = ckpt["cfg"]
    model = fusion.FusionModel(cfgd.get("backbone", "tf_efficientnetv2_m.in21k_ft_in1k"),
                               use_fusion=cfgd.get("use_fusion", False)).to(dev)
    model.load_state_dict(ckpt["model"]); model.eval()
    tf = data.build_transforms(data.DataConfig(img_size=cfgd.get("img_size", 384), train=False))
    feats = np.zeros((len(df), fusion.FUSION_DIM), np.float32)
    loader = DataLoader(fusion.FusionDataset(df, feats, tf), batch_size=16, num_workers=8)
    out = []
    with torch.no_grad():
        for x, ff, _ in loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                lg = model(x.to(dev).to(memory_format=torch.channels_last), ff.to(dev))
            out.append(torch.sigmoid(lg.float()).cpu().numpy())
    return np.concatenate(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--loto-type", default="MOZAMBIQUE/DL")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv("/root/freuid/data/extracted/train_labels.csv")
    df = df[df["type"] == a.loto_type].copy()
    root = Path("/root/freuid/data/extracted/train/train")
    df["abs_path"] = df["id"].map(lambda i: str(root / (str(i) + ".jpeg")))
    df[config.LABEL_COL] = df["label"].astype(int)
    df = df.reset_index(drop=True)
    y = df[config.LABEL_COL].to_numpy()
    print(f"[loto-ens] {a.loto_type}: {len(df)} imgs, fraude={y.mean():.3f}")
    preds = []
    for cp in a.checkpoints.split(","):
        s = predict(cp, df, dev)
        preds.append(s)
        print(f"  {Path(cp).name:32s} FREUID={freuid_score(y, s).freuid:.4f}")
    if len(preds) > 1:
        pa = np.mean(preds, axis=0)
        ranks = [pd.Series(p).rank(pct=True).to_numpy() for p in preds]
        ra = np.mean(ranks, axis=0)
        print(f"  {'ENSEMBLE prob-avg':32s} FREUID={freuid_score(y, pa).freuid:.4f}")
        print(f"  {'ENSEMBLE rank-avg':32s} FREUID={freuid_score(y, ra).freuid:.4f}")
    print("LOTO_ENS_DONE")

if __name__ == "__main__":
    main()
