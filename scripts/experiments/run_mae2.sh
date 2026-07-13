#!/bin/bash
# Squeeze the MAE direction (domain-MAE -> finetune got the new best 0.15589).
#  P1: stronger MAE pretrain (longer, finer patches, higher mask) on 2 backbones
#  P2: finetune each (winning recipe)   P3: MAE-ensemble CSVs for next reset
cd /root/freuid
PY=python3
exec > >(tee -a mae2.log) 2>&1
mkdir -p logs submissions artifacts
echo "=================================================================="
echo "[$(date)] MAE2 START"

PE=artifacts/pseudo_ens.csv
PL=runs/fusion_nofusion_all_pl.pt
MAEFT0=runs/fusion_nofusion_all_maeFT.pt   # current best (0.15589)

run () { local name="$1"; shift
  echo "[$(date)] >>> $name"
  $PY scripts/30_train2.py "$@" --save-name "$name" > "logs/m2_$name.log" 2>&1 \
    && { echo "[$(date)] $name OK"; grep -E "init|val FREUID" "logs/m2_$name.log" | tail -2; } \
    || { echo "[$(date)] $name FAILED"; tail -8 "logs/m2_$name.log"; }
}

# ============ PHASE 1: stronger MAE pretraining (finer patch 16, mask 0.6) ============
echo "[$(date)] >>> MAE-L (ConvNeXtV2-L, 5 ep, patch16 mask0.6)"
$PY scripts/33_mae_pretrain.py --backbone convnextv2_large.fcmae_ft_in22k_in1k \
    --epochs 5 --batch-size 32 --patch 16 --mask-ratio 0.6 --out runs/mae2_cnxL.pt > logs/m2_mae_L.log 2>&1 \
    && echo "[$(date)] MAE-L OK ($(grep -c save logs/m2_mae_L.log) saves)" || { echo "[$(date)] MAE-L FAILED"; tail -6 logs/m2_mae_L.log; }

echo "[$(date)] >>> MAE-effM (EffNetV2-M, 6 ep, patch16 mask0.6)"
$PY scripts/33_mae_pretrain.py --backbone tf_efficientnetv2_m.in21k_ft_in1k \
    --epochs 6 --batch-size 64 --patch 16 --mask-ratio 0.6 --out runs/mae2_effM.pt > logs/m2_mae_effM.log 2>&1 \
    && echo "[$(date)] MAE-effM OK" || { echo "[$(date)] MAE-effM FAILED"; tail -6 logs/m2_mae_effM.log; }

# ============ PHASE 2: finetune each MAE backbone (winning recipe) ============
[ -f runs/mae2_cnxL.pt ] && run fusion_nofusion_all_maeFT_L --train-all \
    --backbone convnextv2_large.fcmae_ft_in22k_in1k --batch-size 32 --loss bce \
    --init-from runs/mae2_cnxL.pt --aux --max-aux 30000 --extra-csv "$PE" --epochs 3
[ -f runs/mae2_effM.pt ] && run fusion_nofusion_all_maeFT_effM --train-all \
    --backbone tf_efficientnetv2_m.in21k_ft_in1k --batch-size 96 --loss focal \
    --init-from runs/mae2_effM.pt --aux --max-aux 30000 --extra-csv "$PE" --epochs 3

# ============ PHASE 3: MAE-ensemble CSVs (heavy TTA + rank-avg; NO submit) ============
echo "[$(date)] >>> CSVs (MAE-ensemble)"
MFTL=runs/fusion_nofusion_all_maeFT_L.pt; MFTM=runs/fusion_nofusion_all_maeFT_effM.pt
gen () { local out="$1"; local rank="$2"; shift 2; local cks=""
  for c in "$@"; do [ -f "$c" ] && cks="$cks,$c"; done; cks="${cks#,}"
  [ -z "$cks" ] && return
  $PY scripts/31_polish_submit.py --checkpoints "$cks" $rank --out "submissions/$out" \
      > "logs/m2_$out.log" 2>&1 && echo "  $out OK" || echo "  $out FAILED"; }
gen sub_m2_maeFT_L.csv    ""  $MFTL
gen sub_m2_maeFT_effM.csv ""  $MFTM
# MAE ensembles (diverse MAE backbones + the proven 0.156)
gen sub_m2_mae_ens.csv      --rank-avg  $MFTL $MFTM $MAEFT0
gen sub_m2_mae_ens_pl.csv   --rank-avg  $MFTL $MFTM $MAEFT0 $PL
gen sub_m2_maeL_mae0.csv    --rank-avg  $MFTL $MAEFT0

echo "=================================================================="
echo "[$(date)] MAE2 DONE."; ls -la submissions/sub_m2_*.csv 2>/dev/null
echo "=================================================================="
