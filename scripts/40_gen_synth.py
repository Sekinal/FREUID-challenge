#!/usr/bin/env python
"""Generate DIVERSE synthetic ID fraud from real FREUID genuines (multi-editor).

Only genuine (label 0) FREUID images are edited (real bona-fides untouched).
Several distinct editors are mixed so the model learns *manipulation* cues, not a
single generator fingerprint:
  - faceswap : InsightFace inswapper_128 (portrait substitution)
  - sd       : SD2 diffusion inpaint of the face region (GenAI portrait)
  - splice   : classical copy of another doc's face + Poisson blend (no neural fp)
Every output gets random anti-fingerprint post-processing (JPEG recompress, mild
noise/blur) to wash generator-specific high-frequency artifacts.

    python3 scripts/40_gen_synth.py --n 24 --out-dir data/synth/sample --methods faceswap,sd,splice
"""
from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config  # noqa: E402

INSWAPPER = "models_synth/inswapper_128.onnx"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--out-dir", default="data/synth/sample")
    p.add_argument("--methods", default="faceswap,sd,splice")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-csv", default="")
    return p.parse_args()


def genuine_paths():
    df = pd.read_csv(config.LABELS_CSV)
    g = df[df[config.LABEL_COL] == 0]
    idx = {p.stem: str(p) for p in config.IMAGES_DIR.rglob("*.jpeg")}
    paths = [idx[i] for i in g[config.ID_COL].astype(str) if i in idx]
    return paths


def expand(b, w, h, m=0.25):
    x0, y0, x1, y1 = b
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, int(x0 - bw * m)); y0 = max(0, int(y0 - bh * m))
    x1 = min(w, int(x1 + bw * m)); y1 = min(h, int(y1 + bh * m))
    return x0, y0, x1, y1


def sdxl_inpaint(sd, crop_bgr, mask_pil_1024, prompt):
    """Inpaint a BGR crop with SDXL at 1024; return BGR crop at original size."""
    from PIL import Image
    ch, cw = crop_bgr.shape[:2]
    cpil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)).resize((1024, 1024))
    gen = sd(prompt=prompt, image=cpil, mask_image=mask_pil_1024,
             num_inference_steps=22, guidance_scale=6.5, strength=0.99).images[0]
    return cv2.cvtColor(np.array(gen.resize((cw, ch))), cv2.COLOR_RGB2BGR)


def post_process(img, rng):
    """Anti-fingerprint: random JPEG recompress + mild noise/blur."""
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.4:
        n = rng.normal(0, rng.uniform(2, 8), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    q = int(rng.integers(55, 96))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed); random.seed(args.seed)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = genuine_paths()
    print(f"[synth] {len(paths):,} genuine sources | methods={methods}")

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    swapper = None
    if "faceswap" in methods:
        import insightface
        swapper = insightface.model_zoo.get_model(INSWAPPER, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    sd = None
    if "sd" in methods or "text" in methods:
        import torch
        from diffusers import AutoPipelineForInpainting
        sd = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16, variant="fp16").to("cuda")
        sd.set_progress_bar_config(disable=True)

    ocr = None
    if "text" in methods:
        import easyocr
        ocr = easyocr.Reader(["en"], gpu=True)

    rows, made, tries = [], 0, 0
    while made < args.n and tries < args.n * 6:
        tries += 1
        src = random.choice(paths)
        img = cv2.imread(src)
        if img is None:
            continue
        h, w = img.shape[:2]
        faces = app.get(img)
        if not faces:
            continue
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        method = random.choice(methods)
        res = None
        try:
            if method == "faceswap":
                # source identity = a different genuine doc's face
                for _ in range(5):
                    o = cv2.imread(random.choice(paths))
                    of = app.get(o) if o is not None else []
                    if of:
                        res = swapper.get(img, face, of[0], paste_back=True); break
            elif method == "splice":
                o = None
                for _ in range(5):
                    cand = cv2.imread(random.choice(paths)); of = app.get(cand) if cand is not None else []
                    if of:
                        o, ofb = cand, of[0].bbox.astype(int); break
                if o is not None:
                    x0, y0, x1, y1 = expand(face.bbox.astype(int), w, h, 0.1)
                    sx0, sy0, sx1, sy1 = expand(ofb, o.shape[1], o.shape[0], 0.1)
                    patch = cv2.resize(o[sy0:sy1, sx0:sx1], (x1 - x0, y1 - y0))
                    mask = np.full(patch.shape[:2], 255, np.uint8)
                    center = ((x0 + x1) // 2, (y0 + y1) // 2)
                    res = cv2.seamlessClone(patch, img, mask, center, cv2.NORMAL_CLONE)
            elif method == "sd":   # GenAI diffusion regeneration of the FACE region
                from PIL import Image, ImageDraw
                x0, y0, x1, y1 = expand(face.bbox.astype(int), w, h, 0.3)
                crop = img[y0:y1, x0:x1]
                mask = Image.new("L", (1024, 1024), 0)
                ImageDraw.Draw(mask).ellipse([180, 140, 844, 940], fill=255)
                newcrop = sdxl_inpaint(sd, crop, mask,
                    "a realistic id document passport portrait photo of a person, frontal, neutral")
                res = img.copy(); res[y0:y1, x0:x1] = newcrop
            elif method == "text":  # GenAI multimodal: diffusion-edit a TEXT field
                from PIL import Image
                results = ocr.readtext(img) if ocr else []
                cand = []
                for (bbox, text, conf) in results:
                    xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
                    bx = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                    bw_, bh_ = bx[2] - bx[0], bx[3] - bx[1]
                    if bw_ > 60 and 12 < bh_ < 120 and conf > 0.3:
                        cand.append(bx)
                if cand:
                    bx = random.choice(cand)
                    x0, y0, x1, y1 = expand(bx, w, h, 0.6)
                    crop = img[y0:y1, x0:x1]
                    mh, mw = crop.shape[:2]
                    mask = Image.new("L", (1024, 1024), 0)
                    from PIL import ImageDraw
                    ty0 = int((bx[1] - y0) / mh * 1024); ty1 = int((bx[3] - y0) / mh * 1024)
                    tx0 = int((bx[0] - x0) / mw * 1024); tx1 = int((bx[2] - x0) / mw * 1024)
                    ImageDraw.Draw(mask).rectangle([tx0, ty0, tx1, ty1], fill=255)
                    newcrop = sdxl_inpaint(sd, crop, mask,
                        "a realistic id document text field with printed characters, high quality")
                    res = img.copy(); res[y0:y1, x0:x1] = newcrop
        except Exception as e:
            print(f"  [skip] {method}: {str(e)[:60]}")
            continue
        if res is None:
            continue
        res = post_process(res, rng)
        sid = f"synth_{method}_{made:05d}"
        fp = out / f"{sid}.jpg"
        cv2.imwrite(str(fp), res)
        rows.append({config.ID_COL: sid, config.LABEL_COL: 1, "abs_path": str(fp), "method": method})
        made += 1
        if made % 10 == 0:
            print(f"  [synth] {made}/{args.n}")

    df = pd.DataFrame(rows)
    print("[synth] por metodo:", df["method"].value_counts().to_dict() if len(df) else "0")
    if args.out_csv:
        df[[config.ID_COL, config.LABEL_COL, "abs_path"]].to_csv(args.out_csv, index=False)
        print(f"[synth] csv -> {args.out_csv}")
    print(f"[done] {made} synthetic frauds -> {out}")


if __name__ == "__main__":
    main()
