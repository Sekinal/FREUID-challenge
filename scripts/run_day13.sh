#!/bin/bash
# ============================================================================
# DAY-13 RUNBOOK — private test released -> final submissions
# Run on the GPU box from /root/freuid. Weights are FROZEN; inference only.
#
#   PRIV_DIR=data/private_test/... bash scripts/run_day13.sh
# ============================================================================
set -e
cd "$(dirname "$0")/.."
PRIV_DIR=${PRIV_DIR:?set PRIV_DIR to the extracted private-test image dir}

echo "===== 1) Radiografia del dataset privado ====="
python3 - <<PY
from pathlib import Path
from PIL import Image
import random
paths = [p for p in Path("$PRIV_DIR").iterdir() if p.suffix.lower() in
         {".jpeg",".jpg",".png",".webp",".bmp",".tif",".tiff"}]
print(f"imagenes: {len(paths):,}")
sizes = []
for p in random.sample(paths, min(300, len(paths))):
    with Image.open(p) as im:
        sizes.append(im.size)
ws = sorted(w for w,h in sizes); hs = sorted(h for w,h in sizes)
print(f"width  min/med/max: {ws[0]}/{ws[len(ws)//2]}/{ws[-1]}")
print(f"height min/med/max: {hs[0]}/{hs[len(hs)//2]}/{hs[-1]}")
PY

echo "===== 2) Inferencia ROBUSTA (3 seeds + TTA, ~2h para 135k) ====="
python3 -u infer.py --data-dir "$PRIV_DIR" --out submissions/priv_robust.csv \
  --checkpoints runs/slot2v3_1024_a.pt,runs/slot2v3_1024_b.pt,runs/slot2v3_1024_c.pt \
  --batch-size 32 --workers 10

echo "===== 3) Inferencia PUBLICA (1 seed + TTA, ~50min) ====="
python3 -u infer.py --data-dir "$PRIV_DIR" --out submissions/priv_public.csv \
  --checkpoints runs/fusion_nofusion_all_noaug_s4.pt \
  --batch-size 32 --workers 10

echo "===== 4) Construir CSVs finales completos + checksums ====="
python3 scripts/day13_build_submissions.py \
  --private-preds-public submissions/priv_public.csv \
  --private-preds-robust submissions/priv_robust.csv

echo "===== 5) Submit a Kaggle ====="
kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai \
  -f submissions/FINAL_public_bm1024.csv -m "FINAL pick 1: public specialist effM@1024 (frozen weights, private rows added)"
kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai \
  -f submissions/FINAL_robust_slot2v3.csv -m "FINAL pick 2: robust 3-seed ensemble effM@1024 capture-aug (frozen weights, private rows added)"

echo ""
echo "##########################################################"
echo "#  MANUAL RESTANTE:                                      #"
echo "#  1. Kaggle web -> Submissions -> SELECT these 2 picks  #"
echo "#  2. Pegar sha256 en docker/README.md + reply template  #"
echo "#  3. Reply en el thread fijado (REPLY_TEMPLATE.txt)     #"
echo "##########################################################"
echo "DAY13_DONE_MARKER"
