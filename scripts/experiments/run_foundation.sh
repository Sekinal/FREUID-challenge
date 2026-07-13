#!/bin/bash
# Foundation-model generalization recipe (from PAD-ID-Card winners + gen. paper):
#  frozen DINOv2-L / CLIP-L linear-probe, then SCORE-FUSION with our EffNetV2 CNN.
#  No IDNet (paper: synthetic bona-fides HURT generalization). Pseudo-labels kept.
cd /root/freuid
PY=python3
exec > >(tee -a foundation.log) 2>&1
mkdir -p logs submissions artifacts
echo "=================================================================="
echo "[$(date)] FOUNDATION START"

PL=runs/fusion_nofusion_all_pl.pt          # best CNN (EffNetV2-M, 0.172)
XL=runs/fusion_nofusion_all_effXL.pt       # EffNetV2-XL
PE=artifacts/pseudo_ens.csv                # teacher-ensemble pseudo-labels

# ensure pseudo_ens exists
[ -f "$PE" ] || $PY scripts/32_ens_pseudolabels.py --teachers "$PL,runs/fusion_nofusion_all_pl_2stage.pt,runs/fusion_nofusion_all_effL_pl.pt,runs/fusion_nofusion_all_swin.pt" --lo 0.04 --hi 0.96 --out "$PE" > logs/f_pe.log 2>&1

run () { local name="$1"; shift
  echo "[$(date)] >>> $name"
  $PY scripts/30_train2.py "$@" --save-name "$name" > "logs/f_$name.log" 2>&1 \
    && { echo "[$(date)] $name OK"; grep -E "freeze|val FREUID" "logs/f_$name.log" | tail -2; } \
    || { echo "[$(date)] $name FAILED"; tail -8 "logs/f_$name.log"; }
}

# ---- Exp A: DINOv2-Large frozen linear-probe @518 (paper's best generalizer) ----
run fusion_nofusion_all_dinoP --train-all --backbone vit_large_patch14_dinov2.lvd142m \
    --img-size 518 --freeze-backbone --batch-size 48 --lr 1e-3 --loss bce \
    --extra-csv "$PE" --max-aux 0 --epochs 5

# ---- Exp B: CLIP-Large frozen linear-probe @224 (Track-1 winner backbone) ----
run fusion_nofusion_all_clipP --train-all --backbone vit_large_patch14_clip_224.openai \
    --img-size 224 --freeze-backbone --batch-size 96 --lr 1e-3 --loss bce \
    --extra-csv "$PE" --max-aux 0 --epochs 5

# ---- Score-fusion submissions (heavy TTA + rank-avg; NO submit, cupo Jun 25) ----
echo "[$(date)] >>> generando CSVs score-fusion (CNN + ViT)"
DINO=runs/fusion_nofusion_all_dinoP.pt; CLIP=runs/fusion_nofusion_all_clipP.pt
gen () { local out="$1"; local rank="$2"; shift 2; local cks=""
  for c in "$@"; do [ -f "$c" ] && cks="$cks,$c"; done; cks="${cks#,}"
  [ -z "$cks" ] && return
  $PY scripts/31_polish_submit.py --checkpoints "$cks" $rank --out "submissions/$out" \
      > "logs/f_$out.log" 2>&1 && echo "  $out OK" || echo "  $out FAILED"; }
# individuales (ver fuerza propia)
gen sub_f_dinoP.csv  ""  $DINO
gen sub_f_clipP.csv  ""  $CLIP
# score-fusion CNN+ViT (la receta que bajo EER 15.6->8.2)
gen sub_f_dino_pl.csv      --rank-avg  $DINO $PL
gen sub_f_clip_pl.csv      --rank-avg  $CLIP $PL
gen sub_f_dino_clip_pl.csv --rank-avg  $DINO $CLIP $PL
gen sub_f_all.csv          --rank-avg  $DINO $CLIP $PL $XL

echo "=================================================================="
echo "[$(date)] FOUNDATION DONE. Candidatas:"; ls -la submissions/sub_f_*.csv 2>/dev/null
echo "=================================================================="
