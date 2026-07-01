"""Run PP-OCRv5 (Latin + Arabic passes) over a labeled sample of FREUID train
images, cache raw OCR to JSONL, and write a features CSV for modeling.

Usage:
  ocr-venv/bin/python scripts/ocr_build.py --per 250 --server \
      --out artifacts/consistency
"""
import argparse, csv, json, os, random, sys, time
from collections import defaultdict

sys.path.insert(0, "/root/freuid")
from freuid import consistency as C  # our feature extractor

ROOT = "/root/freuid/data/extracted"
TR = os.path.join(ROOT, "train", "train")


def poly_to_xywh(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return [float(min(xs)), float(min(ys)), float(max(xs) - min(xs)), float(max(ys) - min(ys))]


def run_pass(ocr, path):
    """Return list of {text, score, box} from a PaddleOCR 3.x predict() call."""
    out = []
    try:
        res = ocr.predict(path)
    except Exception as e:
        return out, str(e)
    for r in res:
        d = r if isinstance(r, dict) else getattr(r, "json", {}).get("res", {})
        texts = d.get("rec_texts", []); scores = d.get("rec_scores", [])
        polys = d.get("rec_polys", d.get("dt_polys", []))
        for i, t in enumerate(texts):
            box = poly_to_xywh(polys[i]) if i < len(polys) else None
            out.append({"text": t, "score": float(scores[i]) if i < len(scores) else 0.0, "box": box})
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=250, help="images per (type,label)")
    ap.add_argument("--server", action="store_true", help="use server rec models (slower, better)")
    ap.add_argument("--out", default="/root/freuid/artifacts/consistency")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-hpi", action="store_true", help="disable High-Performance Inference")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)

    rows = list(csv.DictReader(open(os.path.join(ROOT, "train_labels.csv"))))
    bytl = defaultdict(list)
    for r in rows:
        bytl[(r["type"], r["label"])].append(r["id"])
    sample = []
    for (typ, lab), ids in sorted(bytl.items()):
        for i in random.sample(ids, min(args.per, len(ids))):
            sample.append((i, typ, int(lab)))
    random.shuffle(sample)
    print(f"[build] {len(sample)} images, per={args.per}, server={args.server}", flush=True)

    from paddleocr import PaddleOCR
    kw = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
              use_textline_orientation=False, enable_hpi=not args.no_hpi)
    if args.server:
        kw.update(text_detection_model_name="PP-OCRv5_server_det")
    t0 = time.time()
    lat = PaddleOCR(lang="en",  # -> PP-OCRv6 latin det+rec
                    text_recognition_model_name="PP-OCRv5_server_rec" if args.server else None,
                    **kw)
    ara = PaddleOCR(lang="ar", **kw)  # -> arabic_PP-OCRv5 rec
    print(f"[build] readers ready in {time.time()-t0:.1f}s", flush=True)

    raw_f = open(os.path.join(args.out, "ocr_raw.jsonl"), "w")
    feat_path = os.path.join(args.out, "features.csv")
    fcsv = None
    t0 = time.time(); done = 0
    for i, (iid, typ, lab) in enumerate(sample):
        p = os.path.join(TR, iid + ".jpeg")
        lt, e1 = run_pass(lat, p)
        ar, e2 = run_pass(ara, p)
        # The Arabic model re-reads Latin text too; keep only tokens that carry
        # real Arabic-script content (letters U+0600-06FF or Arabic-Indic digits)
        # so a Latin date from the Arabic pass can't masquerade as a cross-script
        # counterpart.
        def _is_arabic(t):
            return any("؀" <= c <= "ۿ" or c in "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹" for c in t)
        for it in lt: it["script"] = "latin"
        ar = [it for it in ar if _is_arabic(it.get("text", ""))]
        for it in ar: it["script"] = "arabic"
        items = lt + ar
        raw_f.write(json.dumps({"id": iid, "type": typ, "label": lab, "items": items}) + "\n")
        feats = C.extract_features(items)
        feats.update({"id": iid, "type": typ, "label": lab})
        if fcsv is None:
            cols = ["id", "type", "label"] + C.FEATURE_NAMES
            fcsv = csv.DictWriter(open(feat_path, "w"), fieldnames=cols)
            fcsv.writeheader()
        fcsv.writerow({k: feats[k] for k in (["id", "type", "label"] + C.FEATURE_NAMES)})
        done += 1
        if done % 100 == 0:
            rate = done / (time.time() - t0)
            print(f"[build] {done}/{len(sample)}  {rate:.2f} img/s  eta {(len(sample)-done)/rate/60:.1f}m", flush=True)
    raw_f.close()
    print(f"[build] DONE {done} imgs in {(time.time()-t0)/60:.1f}m -> {feat_path}", flush=True)
    print("OCR_BUILD_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
