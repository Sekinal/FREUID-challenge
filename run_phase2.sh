#!/bin/bash
# Phase 2: runs AFTER the overnight chain finishes (no GPU conflict).
#   1) no-aux ablation  -> measures IDNet's real contribution
#   2) pseudo-labeling   -> self-train on public_test (target domain)
cd /root/freuid
PY=.venv/bin/python
exec > >(tee -a phase2.log) 2>&1
echo "=================================================================="
echo "[$(date)] PHASE2 esperando a que termine la cadena 'overnight'..."
while tmux has-session -t overnight 2>/dev/null; do sleep 60; done
echo "[$(date)] cadena terminada. Empezando fase 2."
mkdir -p logs submissions

# 1) Ablacion SIN aux (ConvNeXt + aug + focal, solo FREUID) -> comparar vs e2
echo "[$(date)] >>> e0_noaux (ablacion sin IDNet)"
$PY scripts/13_train_with_aux.py --model convnext_base.fb_in22k_ft_in1k --img-size 384 \
    --aug domain --loss focal --max-aux 0 --epochs 3 --batch-size 128 --lr 5e-5 \
    --run-dir runs/e0_noaux > logs/e0_noaux.log 2>&1 \
    && { echo "[$(date)] e0_noaux OK"; tr '\r' '\n' < logs/e0_noaux.log | grep -E "val  FREUID|test FREUID" | tail -2; } \
    || { echo "[$(date)] e0_noaux FAILED"; tail -6 logs/e0_noaux.log; }

# 2) Pseudo-labeling del public_test (teacher = finales all-5 + fallback e3)
echo "[$(date)] >>> pseudo-labeling"
TEACH="runs/f1_convnext_all/best.pt,runs/f2_dinov2_all/best.pt,runs/e3_convnext_moreaux/best.pt"
$PY scripts/24_pseudolabel.py --teacher "$TEACH" \
    --model convnext_base.fb_in22k_ft_in1k --img-size 384 --aug domain --loss focal \
    --epochs 4 --batch-size 128 --lr 5e-5 --aux-dirs data/aux/idnet2025 --max-aux 40000 \
    --run-dir runs/pl_convnext --out submissions/sub_pseudolabel.csv \
    > logs/pl.log 2>&1 \
    && { echo "[$(date)] pseudolabel OK"; tr '\r' '\n' < logs/pl.log | grep -E "pseudo-labels|holdout val|wrote" | tail -3; } \
    || { echo "[$(date)] pseudolabel FAILED"; tail -10 logs/pl.log; }

echo "=================================================================="
echo "[$(date)] PHASE2 COMPLETO. Resumen:"
$PY scripts/23_summary.py
echo "Submissions:"; ls -la submissions/*.csv 2>/dev/null
echo "[$(date)] FIN PHASE2"
echo "=================================================================="
