#!/bin/bash
# Round 2: build on the new best (pseudo-label, 0.17166). Train-only tonight
# (today's 5/5 Kaggle slots are used); generate candidate CSVs for tomorrow.
cd /root/freuid
PY=python3
export KAGGLE_API_TOKEN=$(cat .kaggle/access_token)
exec > >(tee -a round2.log) 2>&1
mkdir -p logs submissions artifacts
echo "=================================================================="
echo "[$(date)] ROUND2 START"
BASE=runs/fusion_nofusion_30k.pt
PL=runs/fusion_nofusion_all_pl.pt        # new best (0.17166)

# --- iterate self-training: relabel public_test with the WINNER -> sharper labels ---
echo "[$(date)] >>> re-pseudo-label with the 0.172 model"
$PY scripts/27_make_pseudolabels.py --checkpoint $PL --lo 0.05 --hi 0.95 \
    --out artifacts/pseudo2.csv > logs/r2_plmake.log 2>&1
grep "kept" logs/r2_plmake.log | tail -1

echo "[$(date)] >>> train PL2 (30k IDNet + pseudo2)"
$PY scripts/30_train2.py --train-all --aux --max-aux 30000 --extra-csv artifacts/pseudo2.csv \
    --epochs 3 --save-name fusion_nofusion_all_pl2 > logs/r2_pl2.log 2>&1 \
    && { echo "[$(date)] PL2 OK"; grep "val FREUID" logs/r2_pl2.log | tail -1; } \
    || { echo "[$(date)] PL2 FAILED"; tail -6 logs/r2_pl2.log; }

# --- candidate CSVs for TOMORROW (no submit; quota exhausted today) ---
echo "[$(date)] >>> generando CSVs candidatas (sin subir)"
PL2=runs/fusion_nofusion_all_pl2.pt
[ -f $PL2 ] && $PY scripts/28_ensemble_submit.py --checkpoints $PL2 \
    --out submissions/sub_r2_pl2.csv > logs/r2_sub_pl2.log 2>&1 || true
$PY scripts/28_ensemble_submit.py --checkpoints $PL,$BASE \
    --out submissions/sub_r2_pl_base.csv > logs/r2_sub_plbase.log 2>&1 || true
[ -f $PL2 ] && $PY scripts/28_ensemble_submit.py --checkpoints $PL2,$BASE \
    --out submissions/sub_r2_pl2_base.csv > logs/r2_sub_pl2base.log 2>&1 || true
[ -f $PL2 ] && $PY scripts/28_ensemble_submit.py --checkpoints $PL2,$PL,$BASE \
    --out submissions/sub_r2_pl2_pl_base.csv > logs/r2_sub_all.log 2>&1 || true

echo "=================================================================="
echo "[$(date)] ROUND2 DONE. Candidatas para mañana:"
ls -la submissions/sub_r2_*.csv submissions/sub_dp_2stage.csv submissions/sub_dp_fantasyid.csv 2>/dev/null
echo "=================================================================="
