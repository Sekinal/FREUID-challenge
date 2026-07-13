#!/bin/bash
# Data plan: add data that HELPS without noise. Failure-tolerant chain.
#   #1 pseudo-label public_test (target domain)   #2 2-stage pretrain->finetune
#   #4 FantasyID minority (if frame exists)        #3 ensemble + TTA -> submit
cd /root/freuid
PY=python3
export KAGGLE_API_TOKEN=$(cat .kaggle/access_token)
exec > >(tee -a dataplan.log) 2>&1
mkdir -p logs submissions artifacts
echo "=================================================================="
echo "[$(date)] DATA PLAN START"
BASE=runs/fusion_nofusion_30k.pt   # the 0.187/0.178 base (EffNetV2-M, all-5 + 30k IDNet)

# ============ #1 PSEUDO-LABEL ============
echo "[$(date)] >>> #1 pseudo-labels from base + train"
$PY scripts/27_make_pseudolabels.py --checkpoint $BASE --lo 0.05 --hi 0.95 \
    --out artifacts/pseudo.csv > logs/dp_pl_make.log 2>&1
$PY scripts/30_train2.py --train-all --aux --max-aux 30000 --extra-csv artifacts/pseudo.csv \
    --epochs 3 --save-name fusion_nofusion_all_pl > logs/dp_pl_train.log 2>&1 \
    && { echo "[$(date)] PL OK"; grep "val FREUID" logs/dp_pl_train.log | tail -1; } \
    || { echo "[$(date)] PL FAILED"; tail -6 logs/dp_pl_train.log; }

# ============ #2 TWO-STAGE pretrain -> finetune ============
echo "[$(date)] >>> #2a stageA pretrain (FREUID + ALL IDNet digital+scanned)"
$PY scripts/30_train2.py --train-all --aux \
    --aux-roots data/aux/idnet2025,data/aux/idnet2025_scanned --max-aux 0 \
    --epochs 2 --lr 1e-4 --save-name fusion_nofusion_stageA > logs/dp_stageA.log 2>&1 \
    && echo "[$(date)] stageA OK" || { echo "[$(date)] stageA FAILED"; tail -6 logs/dp_stageA.log; }
echo "[$(date)] >>> #2b stageB finetune (FREUID-only, lr 2e-5, init from A)"
if [ -f runs/fusion_nofusion_stageA.pt ]; then
  $PY scripts/30_train2.py --train-all --epochs 3 --lr 2e-5 \
      --init-from runs/fusion_nofusion_stageA.pt --save-name fusion_nofusion_all_stageB > logs/dp_stageB.log 2>&1 \
      && { echo "[$(date)] stageB OK"; grep "val FREUID" logs/dp_stageB.log | tail -1; } \
      || { echo "[$(date)] stageB FAILED"; tail -6 logs/dp_stageB.log; }
fi

# ============ #4 FANTASYID minority (best-effort) ============
# wait for the background download/extract, then build the labeled frame
for i in $(seq 1 30); do tmux has-session -t dlfid 2>/dev/null && sleep 20 || break; done
$PY scripts/29_build_fantasyid.py --root data/aux/fantasyid --out artifacts/fantasyid.csv > logs/dp_fid_build.log 2>&1 || true
cat logs/dp_fid_build.log 2>/dev/null | tail -2
if [ -f artifacts/fantasyid.csv ]; then
  echo "[$(date)] >>> #4 FantasyID minority"
  $PY scripts/30_train2.py --train-all --aux --max-aux 30000 --extra-csv artifacts/fantasyid.csv \
      --epochs 3 --save-name fusion_nofusion_all_fid > logs/dp_fid.log 2>&1 \
      && { echo "[$(date)] FID OK"; grep "val FREUID" logs/dp_fid.log | tail -1; } \
      || { echo "[$(date)] FID FAILED"; tail -6 logs/dp_fid.log; }
else
  echo "[$(date)] #4 FantasyID frame ausente -> skip"
fi

# ============ #3 ENSEMBLE + SUBMIT ============
echo "[$(date)] >>> #3 ensemble + submissions"
ENS="$BASE"
for m in fusion_nofusion_all_pl fusion_nofusion_all_stageB fusion_nofusion_all_fid; do
  [ -f runs/$m.pt ] && ENS="$ENS,runs/$m.pt"
done
echo "[$(date)] ensemble members: $ENS"

# SUBMIT 1 (slot del dia): gran ensemble + TTA
$PY scripts/28_ensemble_submit.py --checkpoints "$ENS" \
    --out submissions/sub_dp_ensemble.csv --submit \
    -m "ensemble 30k+PL+2stage(+fantasyID) +TTA" > logs/dp_sub_ens.log 2>&1
grep -E "wrote|Successfully|error|400|429" logs/dp_sub_ens.log | tail -3

# SUBMIT 2 (slot del dia): pseudo-label model + TTA (la palanca de datos on-domain)
if [ -f runs/fusion_nofusion_all_pl.pt ]; then
  $PY scripts/28_ensemble_submit.py --checkpoints runs/fusion_nofusion_all_pl.pt \
      --out submissions/sub_dp_pseudolabel.csv --submit \
      -m "pseudo-label public_test + 30k IDNet +TTA" > logs/dp_sub_pl.log 2>&1
  grep -E "wrote|Successfully|error|400|429" logs/dp_sub_pl.log | tail -3
fi

# CSVs listos para mañana (NO submit): 2-stage y fantasyid solos
[ -f runs/fusion_nofusion_all_stageB.pt ] && $PY scripts/28_ensemble_submit.py \
    --checkpoints runs/fusion_nofusion_all_stageB.pt --out submissions/sub_dp_2stage.csv > logs/dp_sub_2stage.log 2>&1 || true
[ -f runs/fusion_nofusion_all_fid.pt ] && $PY scripts/28_ensemble_submit.py \
    --checkpoints runs/fusion_nofusion_all_fid.pt --out submissions/sub_dp_fantasyid.csv > logs/dp_sub_fid.log 2>&1 || true

echo "=================================================================="
echo "[$(date)] DATA PLAN COMPLETO"
echo "Submissions generadas:"; ls -la submissions/sub_dp_*.csv 2>/dev/null
echo "[$(date)] FIN"
echo "=================================================================="
