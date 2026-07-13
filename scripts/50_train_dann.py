#!/usr/bin/env python
"""Domain-Adversarial (DANN) training for country-invariant fraud features.

A gradient-reversal domain (country) classifier forces the backbone to learn
features that DON'T reveal the country -> better generalization to UNSEEN
countries (the private test's focus). LOCO-validated. No pseudo-labels.

    python3 scripts/50_train_dann.py --loto-type MOZAMBIQUE/DL --epochs 3 --save-name dann_loto
    python3 scripts/50_train_dann.py --train-all --epochs 3 --save-name dann_all
"""
from __future__ import annotations
import argparse, math, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, timm, torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import aux_data, config, data, metrics, io, validation  # noqa
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd): ctx.lambd = lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lambd * g, None


class DANN(nn.Module):
    def __init__(self, backbone, n_dom):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        d = self.backbone.num_features
        self.fraud = nn.Linear(d, 1)
        self.domain = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_dom))
    def forward(self, x, lambd=0.0):
        f = self.backbone(x)
        return self.fraud(f).squeeze(1), self.domain(GRL.apply(f, lambd))


class DS(Dataset):
    def __init__(self, frame, size, train, dom_map):
        self.p = frame["abs_path"].astype(str).tolist()
        self.y = frame[config.LABEL_COL].to_numpy().astype(np.float32)
        self.dom = frame["_dom"].to_numpy().astype(np.int64)
        tf = [T.Resize((size, size))]
        if train: tf.append(T.RandomHorizontalFlip(0.5))
        tf += [T.ToTensor(), T.Normalize(data.IMAGENET_MEAN, data.IMAGENET_STD)]
        self.tf = T.Compose(tf)
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        try:
            im = Image.open(self.p[i]); im.load(); x = self.tf(im.convert("RGB"))
        except Exception: x = torch.zeros(3, 384, 384)
        return x, self.y[i], self.dom[i]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="convnextv2_large.fcmae_ft_in22k_in1k")
    p.add_argument("--init-from", default="runs/mae_cnxv2L.pt")
    p.add_argument("--img-size", type=int, default=384); p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--loto-type", default=""); p.add_argument("--train-all", action="store_true")
    p.add_argument("--max-aux", type=int, default=30000); p.add_argument("--workers", type=int, default=12)
    p.add_argument("--save-name", required=True); p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def resolve(f): o = io.attach_image_paths(f); return o[o["path_exists"]].reset_index(drop=True)


def main():
    a = parse(); dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    pipe = validation.ValidationPipeline(rebuild=False)
    full = pd.concat([resolve(pipe.train), resolve(pipe.val), resolve(pipe.test)], ignore_index=True)
    test = None
    if a.loto_type:
        test = full[full[config.TYPE_COL] == a.loto_type].reset_index(drop=True)
        pool = full[full[config.TYPE_COL] != a.loto_type].reset_index(drop=True)
        val = pool.sample(min(5000, len(pool)//10), random_state=0); train = pool.drop(val.index).reset_index(drop=True); val = val.reset_index(drop=True)
    else:
        val = full.sample(min(5000, len(full)//10), random_state=0); train = full.drop(val.index).reset_index(drop=True); val = val.reset_index(drop=True)
    # domain = country (from type); aux gets its own domain buckets
    if a.max_aux:
        aux = aux_data.load_idnet_frame([config.REPO_ROOT/"data/aux/idnet2025"])
        aux = aux[aux["abs_path"].map(lambda p: Path(p).exists())]
        if len(aux) > a.max_aux: aux = aux.sample(a.max_aux, random_state=0)
        train = pd.concat([train, aux], ignore_index=True)
    for fr in (train, val):
        fr["_dom"] = fr[config.TYPE_COL].astype(str)
    doms = sorted(train["_dom"].unique()); dmap = {d: i for i, d in enumerate(doms)}
    train["_dom"] = train["_dom"].map(dmap); val["_dom"] = val["_dom"].map(lambda d: dmap.get(d, 0))
    if a.smoke: train = train.sample(1500, random_state=0); val = val.sample(500, random_state=0); a.epochs = 1
    print(f"[dann] train={len(train):,} val={len(val):,} test={0 if test is None else len(test):,} domains={len(doms)}", flush=True)
    tl = DataLoader(DS(train, a.img_size, True, dmap), batch_size=a.batch_size, shuffle=True, drop_last=True,
                    num_workers=a.workers, pin_memory=True, persistent_workers=True)
    model = DANN(a.backbone, len(doms)).to(dev)
    if a.init_from and Path(a.init_from).exists():
        ck = torch.load(a.init_from, map_location=dev, weights_only=False)
        miss, unexp = model.load_state_dict(ck["model"], strict=False)
        print(f"[dann] warm-start {a.init_from} missing={len(miss)} unexpected={len(unexp)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs*len(tl))
    bce = nn.BCEWithLogitsLoss(); ce = nn.CrossEntropyLoss()
    tf_e = T.Compose([T.Resize((a.img_size, a.img_size)), T.ToTensor(), T.Normalize(data.IMAGENET_MEAN, data.IMAGENET_STD)])

    def predict(frame):
        frame = frame.copy(); frame["_dom"] = 0
        ld = DataLoader(DS(frame, a.img_size, False, dmap), batch_size=a.batch_size, num_workers=a.workers, pin_memory=True)
        model.eval(); out = []
        with torch.no_grad():
            for x, _, _ in ld:
                with torch.autocast("cuda", dtype=torch.bfloat16): out.append(torch.sigmoid(model(x.to(dev))[0].float()).cpu())
        return torch.cat(out).numpy()

    tot = a.epochs*len(tl); step = 0; best = {"audet": math.inf}
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); rf = rd = 0.0
        for bi, (x, y, dm) in enumerate(tl):
            lambd = 2./(1.+math.exp(-10*step/tot))-1   # ramp 0->1
            x = x.to(dev, non_blocking=True); y = y.to(dev); dm = dm.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lf, ld_ = model(x, lambd); loss = bce(lf, y) + ce(ld_, dm)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            rf += bce(lf.detach(), y).item(); rd += ce(ld_.detach(), dm).item()
            if bi % 100 == 0: print(f"  ep{ep} it{bi}/{len(tl)} fraud={rf/(bi+1):.3f} dom={rd/(bi+1):.3f} lambd={lambd:.2f} ({(bi+1)*a.batch_size/(time.time()-t0):.0f}im/s)", flush=True)
        m = metrics.freuid_score(val[config.LABEL_COL].to_numpy(), predict(val))
        tline = ""
        if test is not None:
            tm = metrics.freuid_score(test[config.LABEL_COL].to_numpy(), predict(test)); tline = f" || LOTO-test FREUID={tm.freuid:.4f}"
        print(f"  [ep{ep}] val FREUID={m.freuid:.4f} AuDET={m.audet:.4f}{tline}", flush=True)
        if m.audet < best["audet"]:
            best = {"epoch": ep, "audet": m.audet, "loto": (tm.freuid if test is not None else None)}
            torch.save({"model": model.state_dict(), "backbone": a.backbone, "img_size": a.img_size}, config.RUNS_DIR/f"{a.save_name}.pt")
    if test is not None: print(f"[PROXY] {a.save_name} loto={a.loto_type} CROSS-COUNTRY-FREUID={best['loto']:.4f}", flush=True)
    print(f"[dann] done best={best}", flush=True)


if __name__ == "__main__":
    main()
