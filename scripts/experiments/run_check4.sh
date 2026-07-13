#!/bin/bash
# CHEQUEO 4: OOD del campeon publico. Config identico a bm1024 pero LOTO Mozambique.
set -e
cd /root/freuid
unset FREUID_IMG_CACHE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u scripts/24_train_fusion.py --loto-type "MOZAMBIQUE/DL" --no-fusion --loss bce \
  --img-size 1024 --no-capture-aug --batch-size 12 --workers 10 --epochs 3 --seed 4
echo "===== resultado json ====="
ls -t artifacts/fusion_result_*loto*MOZAMBIQUE* 2>/dev/null | head -2
cat $(ls -t artifacts/fusion_result_*loto*MOZAMBIQUE* 2>/dev/null | head -1) 2>/dev/null
echo "CHECK4_DONE_MARKER"
