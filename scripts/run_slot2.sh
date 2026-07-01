set -e
cd /root/freuid
export FREUID_IMG_CACHE=/root/freuid/data/cache896 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
echo "[$(date +%H:%M:%S)] SLOT2 train-all robust (diverse DIGITAL, heavy aug, 768, NO scanned)"
python3 -u scripts/30_train2.py --train-all \
  --backbone tf_efficientnetv2_m.in21k_ft_in1k --img-size 768 --batch-size 24 --workers 10 \
  --loss bce --aug-strength heavy \
  --aux --aux-roots data/aux/idnet2025 --max-aux 60000 \
  --extra-csv artifacts/extra_diverse.csv \
  --epochs 3 --save-name slot2_robust_all768
echo "SLOT2_DONE_MARKER"
