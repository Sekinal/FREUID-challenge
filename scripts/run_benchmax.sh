#!/bin/bash
# Benchmaxxer: pure in-distribution public-LB overfit.
# all-5 types, NO capture-aug (keep the fragile digital artifact), NO aux, plain BCE,
# res-768, EffNetV2-M, channels_last, 896px cache, 3-seed ensemble + flip-TTA.
# No submit (gated separately).
set -e
cd /root/freuid
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FREUID_IMG_CACHE=/root/freuid/data/cache896
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
EP=${EP:-3}; BS=${BS:-24}; SZ=${SZ:-768}
mkdir -p submissions
for S in 0 1 2; do
  echo "===== TRAIN seed $S ep=$EP bs=$BS sz=$SZ ($(date +%H:%M:%S)) ====="
  python3 -u scripts/24_train_fusion.py --train-all --no-fusion --loss bce \
    --img-size $SZ --no-capture-aug --batch-size $BS --workers 10 --epochs $EP --seed $S
done
for S in 0 1 2; do
  echo "===== PREDICT seed $S ($(date +%H:%M:%S)) ====="
  python3 -u scripts/25_predict_fusion.py \
    --checkpoint runs/fusion_nofusion_all_noaug_s${S}.pt --tta --out submissions/bm_s${S}.csv
done
python3 - <<'PY'
import pandas as pd, numpy as np, glob
fs = sorted(glob.glob("submissions/bm_s*.csv"))
dfs = [pd.read_csv(f) for f in fs]
base = dfs[0].copy(); col = base.columns[1]
base[col] = np.mean([d[col].values for d in dfs], axis=0)
base.to_csv("submissions/bm_ensemble.csv", index=False)
print(f"[ensemble] {len(fs)} seeds -> submissions/bm_ensemble.csv rows={len(base)}")
print(base[col].describe())
PY
echo "BENCHMAX_TRAIN_DONE_MARKER"
