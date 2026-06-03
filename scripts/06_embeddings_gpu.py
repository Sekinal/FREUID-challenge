#!/usr/bin/env python
"""Extract image embeddings on the GPU with a pretrained vision encoder.

Default encoder is OpenCLIP ViT-B/32 (laion2b); set ``FREUID_EMB_MODEL`` /
``FREUID_EMB_PRETRAINED`` to override. Embeddings are L2-normalised and saved to
``embeddings/embeddings.npy`` with aligned ids and metadata. Batched and
device-aware so it scales from the 13-image sample to the full release.

    uv run scripts/06_embeddings_gpu.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freuid import config, io  # noqa: E402

MODEL = os.environ.get("FREUID_EMB_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("FREUID_EMB_PRETRAINED", "laion2b_s34b_b79k")
BATCH = int(os.environ.get("FREUID_EMB_BATCH", "32"))


def main() -> None:
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}"
          + (f" — {torch.cuda.get_device_name(0)}" if device == "cuda" else ""))

    paths = io.list_images()
    if not paths:
        sys.exit("No images found. Run scripts/00_download.py first.")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL, pretrained=PRETRAINED
    )
    model = model.to(device).eval()

    ids, embs = [], []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(paths), BATCH):
            batch_paths = paths[start:start + BATCH]
            tensors, batch_ids = [], []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception as exc:  # noqa: BLE001
                    print(f"[skip] {p.name}: {exc}")
                    continue
                tensors.append(preprocess(img))
                batch_ids.append(p.stem)
            if not tensors:
                continue
            x = torch.stack(tensors).to(device)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                feats = model.encode_image(x)
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            embs.append(feats.cpu().numpy())
            ids.extend(batch_ids)
            print(f"  embedded {len(ids)}/{len(paths)}")

    matrix = np.concatenate(embs, axis=0).astype(np.float32)
    config.EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EMB_DIR / "embeddings.npy", matrix)
    (config.EMB_DIR / "ids.json").write_text(json.dumps(ids))

    meta = {
        "model": MODEL, "pretrained": PRETRAINED, "device": device,
        "n": int(matrix.shape[0]), "dim": int(matrix.shape[1]),
        "seconds": round(time.time() - t0, 2),
    }
    io.save_json("embeddings_meta.json", meta)
    print(f"[done] {meta['n']}x{meta['dim']} embeddings in {meta['seconds']}s "
          f"-> {config.EMB_DIR/'embeddings.npy'}")


if __name__ == "__main__":
    main()
