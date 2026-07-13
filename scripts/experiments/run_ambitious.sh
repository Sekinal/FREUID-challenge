#!/bin/bash
# Ambitious A+B: break below 0.172. Failure-tolerant; Part A delivers even if B fails.
#  A: strong backbones (ConvNeXtV2-L, EffNetV2-XL) + teacher-ENSEMBLE pseudo-labels
#  B: domain MAE pretrain (ConvNeXtV2-L) -> finetune
cd /root/freuid
PY=python3
exec > >(tee -a ambitious.log) 2>&1
mkdir -p logs submissions artifacts
echo "=================================================================="
echo "[$(date)] AMBITIOUS START"

PL=runs/fusion_nofusion_all_pl.pt
PL2S=runs/fusion_nofusion_all_pl_2stage.pt
EFFL=runs/fusion_nofusion_all_effL_pl.pt
SWIN=runs/fusion_nofusion_all_swin.pt

run () { local name="$1"; shift
  echo "[$(date)] >>> $name"
  $PY scripts/30_train2.py "$@" --save-name "$name" > "logs/a_$name.log" 2>&1 \
    && { echo "[$(date)] $name OK"; grep "val FREUID" "logs/a_$name.log" | tail -1; } \
    || { echo "[$(date)] $name FAILED"; tail -6 "logs/a_$name.log"; }
}

# ---- A0: teacher-ensemble pseudo-labels (cleaner than single 0.172 teacher) ----
echo "[$(date)] >>> A0 teacher-ensemble pseudo-labels"
$PY scripts/32_ens_pseudolabels.py --teachers $PL,$PL2S,$EFFL,$SWIN \
    --lo 0.04 --hi 0.96 --out artifacts/pseudo_ens.csv > logs/a_pseudoens.log 2>&1
grep kept logs/a_pseudoens.log | tail -1

# ---- A1/A2: strong backbones + ensemble pseudo ----
run fusion_nofusion_all_cnxv2L --train-all --backbone convnextv2_large.fcmae_ft_in22k_in1k \
    --batch-size 32 --loss bce  --aux --max-aux 30000 --extra-csv artifacts/pseudo_ens.csv --epochs 3
run fusion_nofusion_all_effXL  --train-all --backbone tf_efficientnetv2_xl.in21k_ft_in1k \
    --batch-size 32 --loss focal --aux --max-aux 30000 --extra-csv artifacts/pseudo_ens.csv --epochs 3

# ---- B1: domain MAE pretrain (ConvNeXtV2-L) ----
echo "[$(date)] >>> B1 MAE domain pretrain (ConvNeXtV2-L)"
$PY scripts/33_mae_pretrain.py --backbone convnextv2_large.fcmae_ft_in22k_in1k \
    --epochs 3 --batch-size 24 --out runs/mae_cnxv2L.pt > logs/a_mae.log 2>&1 \
    && { echo "[$(date)] MAE OK"; grep "save\|done" logs/a_mae.log | tail -1; } \
    || { echo "[$(date)] MAE FAILED"; tail -6 logs/a_mae.log; }

# ---- B2: finetune from MAE-pretrained backbone ----
if [ -f runs/mae_cnxv2L.pt ]; then
  run fusion_nofusion_all_maeFT --train-all --backbone convnextv2_large.fcmae_ft_in22k_in1k \
      --batch-size 32 --loss bce --init-from runs/mae_cnxv2L.pt \
      --aux --max-aux 30000 --extra-csv artifacts/pseudo_ens.csv --epochs 3
fi

# ---- Submissions (heavy TTA + rank-avg; NO submit, cupo manana) ----
echo "[$(date)] >>> generando CSVs (heavy TTA + rank-avg)"
CNX=runs/fusion_nofusion_all_cnxv2L.pt; XL=runs/fusion_nofusion_all_effXL.pt
MFT=runs/fusion_nofusion_all_maeFT.pt
gen () { local out="$1"; local rank="$2"; shift 2; local cks=""
  for c in "$@"; do [ -f "$c" ] && cks="$cks,$c"; done; cks="${cks#,}"
  [ -z "$cks" ] && return
  $PY scripts/31_polish_submit.py --checkpoints "$cks" $rank --out "submissions/$out" \
      > "logs/a_$out.log" 2>&1 && echo "  $out OK" || echo "  $out FAILED"; }
gen sub_a_cnxv2L.csv  ""          $CNX
gen sub_a_effXL.csv   ""          $XL
gen sub_a_maeFT.csv   ""          $MFT
gen sub_a_ens_new.csv     --rank-avg  $CNX $XL $MFT $PL
gen sub_a_ens_allbest.csv --rank-avg  $PL $PL2S $EFFL $SWIN $CNX $XL $MFT

echo "=================================================================="
echo "[$(date)] AMBITIOUS DONE. Candidatas:"; ls -la submissions/sub_a_*.csv 2>/dev/null
echo "=================================================================="
