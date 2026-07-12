"""Pre-resize all FREUID train + public_test images to an 896px cache so training
is not bottlenecked decoding 1585x1000 JPEGs on 12 cores."""
import os, csv, time
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

ROOT = "/root/freuid/data/extracted"
OUT = "/root/freuid/data/cache896"
MAX = 896
os.makedirs(OUT, exist_ok=True)


def all_paths():
    ps = []
    for r in csv.DictReader(open(os.path.join(ROOT, "train_labels.csv"))):
        ps.append(os.path.join(ROOT, "train", "train", r["id"] + ".jpeg"))
    pt = os.path.join(ROOT, "public_test", "public_test")
    for f in os.listdir(pt):
        ps.append(os.path.join(pt, f))
    return ps


def work(p):
    o = os.path.join(OUT, Path(p).stem + ".jpg")
    if os.path.exists(o):
        return 0
    try:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        s = MAX / max(w, h)
        if s < 1:
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        im.save(o, "JPEG", quality=92)
        return 1
    except Exception:
        return -1


if __name__ == "__main__":
    ps = all_paths()
    print(f"{len(ps)} images -> {OUT}", flush=True)
    t0 = time.time(); done = 0; new = 0
    with ProcessPoolExecutor(max_workers=12) as ex:
        for r in ex.map(work, ps, chunksize=64):
            done += 1; new += (r == 1)
            if done % 10000 == 0:
                print(f"  {done}/{len(ps)}  {done/(time.time()-t0):.0f} img/s", flush=True)
    print(f"CACHE_DONE {done} imgs ({new} new) in {(time.time()-t0)/60:.1f}m", flush=True)
