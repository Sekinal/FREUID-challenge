#!/bin/bash
# Overnight FREUID experiment chain. Failure-tolerant: a crashing experiment
# is logged and the chain continues. Review with scripts/23_summary.py.
cd /root/freuid
PY=.venv/bin/python
LOG=overnight.log
mkdir -p logs submissions
exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo "[$(date)] OVERNIGHT CHAIN START"
echo "=================================================================="

run_exp () {  # name, args...
  local name="$1"; shift
  echo "------------------------------------------------------------------"
  echo "[$(date)] >>> $name"
  echo "    args: $*"
  $PY scripts/13_train_with_aux.py --run-dir "runs/$name" "$@" > "logs/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date)] <<< $name OK"
    tr '\r' '\n' < "logs/$name.log" | grep -E "val  FREUID|test FREUID" | tail -2
  else
    echo "[$(date)] !!! $name FAILED (exit $rc) — ver logs/$name.log"
    tr '\r' '\n' < "logs/$name.log" | tail -8
  fi
}

# --- soft-wait for extra IDNet data (max ~15 min) ---
for i in $(seq 1 90); do
  grep -q "EXTRA IDNet listo" dl_idnet_extra.log 2>/dev/null && { echo "[$(date)] extra IDNet listo"; break; }
  sleep 10
done

AUX1="data/aux/idnet2025"
AUX2="data/aux/idnet2025,data/aux/idnet2025_scanned"

# ================= FASE 1 — BARRIDO (3 paises train, test=Mozambique) =================
run_exp e1_convnext_aug        --model convnext_base.fb_in22k_ft_in1k --img-size 384 \
        --aug domain --loss bce   --aux-dirs "$AUX1" --max-aux 30000 \
        --epochs 3 --batch-size 128 --lr 5e-5

run_exp e2_convnext_aug_focal  --model convnext_base.fb_in22k_ft_in1k --img-size 384 \
        --aug domain --loss focal --aux-dirs "$AUX1" --max-aux 30000 \
        --epochs 3 --batch-size 128 --lr 5e-5

run_exp e3_convnext_moreaux    --model convnext_base.fb_in22k_ft_in1k --img-size 384 \
        --aug domain --loss focal --aux-dirs "$AUX2" --max-aux 45000 \
        --epochs 3 --batch-size 128 --lr 5e-5

run_exp e4_dinov2_aug_focal    --model vit_base_patch14_dinov2.lvd142m --img-size 224 \
        --aug domain --loss focal --aux-dirs "$AUX1" --max-aux 30000 \
        --epochs 3 --batch-size 192 --lr 5e-5

run_exp e5_effnetv2m_aug_focal --model tf_efficientnetv2_m.in21k_ft_in1k --img-size 384 \
        --aug domain --loss focal --aux-dirs "$AUX1" --max-aux 30000 \
        --epochs 3 --batch-size 96 --lr 1e-4

echo "=================================================================="
echo "[$(date)] BARRIDO terminado. Resumen:"
$PY scripts/23_summary.py
echo "=================================================================="

# ================= FASE 2 — MODELOS FINALES (los 5 paises) =================
run_exp f1_convnext_all  --model convnext_base.fb_in22k_ft_in1k --img-size 384 \
        --aug domain --loss focal --aux-dirs "$AUX2" --max-aux 60000 \
        --epochs 4 --batch-size 128 --lr 5e-5 --train-all

run_exp f2_dinov2_all    --model vit_base_patch14_dinov2.lvd142m --img-size 224 \
        --aug domain --loss focal --aux-dirs "$AUX1" --max-aux 40000 \
        --epochs 4 --batch-size 192 --lr 5e-5 --train-all

# ================= FASE 3 — SUBMISSIONS (no se suben) =================
echo "[$(date)] Generando submissions..."
# (a) mejor modelo 3-paises (con score local honesto en Mozambique)
$PY scripts/22_ensemble_submission.py --checkpoints runs/e3_convnext_moreaux/best.pt \
    --out submissions/sub_convnext_3country.csv > logs/sub_3country.log 2>&1 \
    && echo "  sub_convnext_3country.csv OK" || echo "  sub_3country FAILED"
# (b) modelo final convnext (todos los paises)
$PY scripts/22_ensemble_submission.py --checkpoints runs/f1_convnext_all/best.pt --no-score \
    --out submissions/sub_f1_convnext_all.csv > logs/sub_f1.log 2>&1 \
    && echo "  sub_f1_convnext_all.csv OK" || echo "  sub_f1 FAILED"
# (c) ENSEMBLE final convnext + dinov2 (todos los paises)
$PY scripts/22_ensemble_submission.py \
    --checkpoints runs/f1_convnext_all/best.pt,runs/f2_dinov2_all/best.pt --no-score \
    --out submissions/sub_ensemble_all.csv > logs/sub_ensemble.log 2>&1 \
    && echo "  sub_ensemble_all.csv OK" || echo "  sub_ensemble FAILED"

echo "=================================================================="
echo "[$(date)] CHAIN COMPLETO. Resumen final:"
$PY scripts/23_summary.py
echo "Submissions generadas (NO subidas):"; ls -la submissions/*.csv 2>/dev/null
echo "[$(date)] FIN"
echo "=================================================================="
