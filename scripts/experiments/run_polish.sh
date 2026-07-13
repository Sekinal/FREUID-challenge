#!/bin/bash
# Pre-submit polish: (3) 3rd diverse backbone (Swin, BCE), then
# (1)+(2) regenerate candidates with HEAVY TTA + RANK-average ensembling.
cd /root/freuid
PY=python3
exec > >(tee -a polish.log) 2>&1
mkdir -p logs submissions artifacts
echo "=================================================================="
echo "[$(date)] POLISH START"

BASE=runs/fusion_nofusion_30k.pt
PL=runs/fusion_nofusion_all_pl.pt
PL2=runs/fusion_nofusion_all_pl2.pt
PL3=runs/fusion_nofusion_all_pl3.pt
EFFL=runs/fusion_nofusion_all_effL_pl.pt
SWIN=runs/fusion_nofusion_all_swin.pt

# ---- (3) 3rd diverse backbone: Swin-Base/384 with BCE (focal collapses non-effnet) ----
echo "[$(date)] >>> train Swin backbone (BCE + pseudo3)"
$PY scripts/30_train2.py --train-all --backbone swin_base_patch4_window12_384.ms_in22k_ft_in1k \
    --batch-size 48 --loss bce --aux --max-aux 30000 --extra-csv artifacts/pseudo3.csv \
    --epochs 3 --save-name fusion_nofusion_all_swin > logs/p_swin.log 2>&1 \
    && { echo "[$(date)] Swin OK"; grep "val FREUID" logs/p_swin.log | tail -1; } \
    || { echo "[$(date)] Swin FAILED"; tail -8 logs/p_swin.log; }

# ---- (1)+(2) regenerate candidates: heavy TTA + rank-average ----
echo "[$(date)] >>> regenerando candidatas (heavy TTA + rank-avg)"
gen () { local out="$1"; local rank="$2"; shift 2; local cks=""
  for c in "$@"; do [ -f "$c" ] && cks="$cks,$c"; done; cks="${cks#,}"
  [ -z "$cks" ] && { echo "  $out: sin checkpoints"; return; }
  $PY scripts/31_polish_submit.py --checkpoints "$cks" $rank --out "submissions/$out" \
      > "logs/p_$out.log" 2>&1 && echo "  $out OK ($cks)" || { echo "  $out FAILED"; tail -4 logs/p_$out.log; }
}

# singles (heavy TTA, no rank needed)
gen sub_p_effL.csv   ""          $EFFL
gen sub_p_pl3.csv    ""          $PL3
gen sub_p_2stage.csv ""          runs/fusion_nofusion_all_pl_2stage.pt
gen sub_p_swin.csv   ""          $SWIN
# rank-averaged strong ensembles (diverse backbones)
gen sub_p_ens_strong.csv  --rank-avg  $PL $PL3 $EFFL $SWIN
gen sub_p_ens_3div.csv    --rank-avg  $PL3 $EFFL $SWIN
gen sub_p_ens_max.csv     --rank-avg  $PL $PL2 $PL3 $EFFL $SWIN

echo "=================================================================="
echo "[$(date)] POLISH DONE. Candidatas mejoradas:"
ls -la submissions/sub_p_*.csv 2>/dev/null
echo "=================================================================="
