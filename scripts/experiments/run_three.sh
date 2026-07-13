#!/bin/bash
# Three new ideas overnight: (1) higher resolution 512, (2) soft pseudo-labels,
# (3) SRM forensic noise-residual stream. CSVs only (cupo Jun 27 agotado -> Jun 28).
cd /root/freuid
PY=python3
exec > >(tee -a three.log) 2>&1
mkdir -p logs submissions artifacts
echo "=================================================================="
echo "[$(date)] THREE START"
MAE=runs/mae_cnxv2L.pt; PE=artifacts/pseudo_ens.csv

# ---- generate soft pseudo-labels (for idea 2) ----
echo "[$(date)] >>> soft pseudo-labels"
$PY scripts/43_soft_pseudo.py --teacher runs/fusion_nofusion_all_maeFT.pt \
    --out artifacts/soft_pseudo.csv > logs/t_soft.log 2>&1 && grep soft logs/t_soft.log | tail -1

run () { local name="$1"; shift
  echo "[$(date)] >>> $name"
  $PY scripts/30_train2.py "$@" --save-name "$name" > "logs/t_$name.log" 2>&1 \
    && { echo "[$(date)] $name OK"; grep "val FREUID" "logs/t_$name.log" | tail -1; } \
    || { echo "[$(date)] $name FAILED"; tail -6 "logs/t_$name.log"; }
}

# ---- IDEA 1: higher resolution 512 (warm-start from the 0.156 model -> adapt to 512) ----
run fusion_nofusion_all_res512 --train-all --backbone convnextv2_large.fcmae_ft_in22k_in1k \
    --init-from runs/fusion_nofusion_all_maeFT.pt --img-size 512 --batch-size 16 --loss bce --lr 3e-5 \
    --aux --max-aux 30000 --extra-csv "$PE" --epochs 2

# ---- IDEA 2: soft pseudo-labels ----
run fusion_nofusion_all_softpl --train-all --backbone convnextv2_large.fcmae_ft_in22k_in1k \
    --init-from "$MAE" --img-size 384 --batch-size 32 --loss bce \
    --aux --max-aux 30000 --extra-csv artifacts/soft_pseudo.csv --epochs 3

# ---- IDEA 3: SRM forensic stream (self-contained: trains + writes CSV) ----
echo "[$(date)] >>> SRM forensic stream"
$PY scripts/42_train_srm.py --backbone convnextv2_base.fcmae_ft_in22k_in1k --img-size 384 \
    --epochs 4 --batch-size 48 --max-aux 30000 --extra-csv "$PE" \
    --save-name srm_cnxB --out submissions/sub_srm.csv > logs/t_srm.log 2>&1 \
  && { echo "[$(date)] SRM OK"; grep "val FREUID\|wrote" logs/t_srm.log | tail -2; } \
  || { echo "[$(date)] SRM FAILED"; tail -8 logs/t_srm.log; }

# ---- CSVs for res512 + softpl (heavy TTA) ----
R=runs/fusion_nofusion_all_res512.pt; S=runs/fusion_nofusion_all_softpl.pt
[ -f "$R" ] && $PY scripts/31_polish_submit.py --checkpoints "$R" --out submissions/sub_res512.csv > logs/t_c_res.log 2>&1 && echo "  sub_res512 OK"
[ -f "$S" ] && $PY scripts/31_polish_submit.py --checkpoints "$S" --out submissions/sub_softpl.csv > logs/t_c_soft.log 2>&1 && echo "  sub_softpl OK"

echo "=================================================================="
echo "[$(date)] THREE DONE. Candidatas (subir en reset Jun 28):"
ls -la submissions/sub_res512.csv submissions/sub_softpl.csv submissions/sub_srm.csv 2>/dev/null
echo "=================================================================="
