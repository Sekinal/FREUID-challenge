#!/bin/bash
# Double training: (A) differentiable RANKING surrogate (soft-AUC) loss,
# (B) REINFORCE RL fine-tune. Both ConvNeXtV2-L on the winning data recipe.
# Also retries the failed 3-MAE ensemble. Submits the 3 (3 slots free).
cd /root/freuid
PY=python3
export KAGGLE_API_TOKEN=$(cat .kaggle/access_token 2>/dev/null)
exec > >(tee -a doubletrain.log) 2>&1
mkdir -p logs submissions
echo "=================================================================="
echo "[$(date)] DOUBLETRAIN START"
PE=artifacts/pseudo_ens.csv
M0=runs/fusion_nofusion_all_maeFT.pt
MM=runs/fusion_nofusion_all_maeFT_effM.pt
MS=runs/fusion_nofusion_all_maeFT_synth.pt

# ---- A) ranking-surrogate (soft-AUC) ----
echo "[$(date)] >>> rank-surrogate"
$PY scripts/30_train2.py --train-all --backbone convnextv2_large.fcmae_ft_in22k_in1k \
    --init-from runs/mae_cnxv2L.pt --loss rank --batch-size 32 --aux --max-aux 30000 \
    --extra-csv "$PE" --epochs 3 --save-name fusion_nofusion_all_rank > logs/dt_rank.log 2>&1 \
  && { echo "[$(date)] rank OK"; grep "val FREUID" logs/dt_rank.log | tail -1; } \
  || { echo "[$(date)] rank FAILED"; tail -6 logs/dt_rank.log; }

# ---- B) REINFORCE RL fine-tune (warm-start from maeFT 0.156) ----
echo "[$(date)] >>> RL (REINFORCE)"
$PY scripts/41_train_rl.py --init-from "$M0" --backbone convnextv2_large.fcmae_ft_in22k_in1k \
    --batch-size 32 --max-aux 30000 --extra-csv "$PE" --epochs 2 --lr 1e-5 \
    --save-name rl_maeFT > logs/dt_rl.log 2>&1 \
  && { echo "[$(date)] RL OK"; grep "val FREUID" logs/dt_rl.log | tail -1; } \
  || { echo "[$(date)] RL FAILED"; tail -6 logs/dt_rl.log; }

# ---- CSVs (heavy TTA) ----
RANK=runs/fusion_nofusion_all_rank.pt; RL=runs/rl_maeFT.pt
[ -f "$RANK" ] && $PY scripts/31_polish_submit.py --checkpoints "$RANK" --out submissions/sub_rank.csv > logs/dt_c_rank.log 2>&1 && echo "  sub_rank OK"
[ -f "$RL" ]   && $PY scripts/31_polish_submit.py --checkpoints "$RL"   --out submissions/sub_rl.csv   > logs/dt_c_rl.log 2>&1 && echo "  sub_rl OK"
$PY scripts/31_polish_submit.py --checkpoints "$M0,$MM,$MS" --rank-avg --out submissions/sub_mae3.csv > logs/dt_c_mae3.log 2>&1 && echo "  sub_mae3 OK"

# ---- submit (uses free slots) ----
sub () { kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai -f "$1" -m "$2" 2>&1 | grep -E "Successfully|error|exceeded" | tail -1; }
echo "[$(date)] submitting..."
[ -f submissions/sub_rank.csv ] && { echo "[$(date)] >>rank"; sub submissions/sub_rank.csv "ranking surrogate soft-AUC loss + heavy TTA"; }
[ -f submissions/sub_rl.csv ]   && { echo "[$(date)] >>rl";   sub submissions/sub_rl.csv "REINFORCE RL fine-tune + heavy TTA"; }
[ -f submissions/sub_mae3.csv ] && { echo "[$(date)] >>mae3"; sub submissions/sub_mae3.csv "rank-avg 3 diverse MAE"; }
sleep 100
echo "[$(date)] scores:"; kaggle competitions submissions -c the-freuid-challenge-2026-ijcai-ecai 2>/dev/null | head -7 | sed -E "s/  +/ | /g" | cut -c1-120
echo "[$(date)] DOUBLETRAIN DONE"
echo "=================================================================="
