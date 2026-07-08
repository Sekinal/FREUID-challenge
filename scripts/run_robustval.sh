set -e
cd /root/freuid
export FREUID_IMG_CACHE=/root/freuid/data/cache896 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
mkdir -p logs
echo "[$(date +%H:%M:%S)] ROBUST-VAL start (Mauritius-LOTO, diverse data, 384)"
python3 -u scripts/30_train2.py --loto-type "MAURITIUS/ID" \
  --backbone tf_efficientnetv2_m.in21k_ft_in1k --img-size 384 --batch-size 48 --workers 10 \
  --loss bce --aug-strength heavy \
  --aux --aux-roots data/aux/idnet2025 --max-aux 60000 \
  --extra-csv artifacts/extra_diverse.csv \
  --epochs 3 --save-name robust_val_maurLOTO
echo "ROBUSTVAL_DONE_MARKER"
