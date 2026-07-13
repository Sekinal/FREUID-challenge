#!/bin/bash
# Synthetic-data experiment: 4 diverse editors (3 GenAI + 1 classical) -> fraud
# from real FREUID genuines, mixed (capped) into the MAE recipe (best = 0.15589).
cd /root/freuid
PY=python3
exec > >(tee -a synth.log) 2>&1
mkdir -p data/synth/full logs submissions artifacts
echo "=================================================================="
echo "[$(date)] SYNTH START"

gen () { local n="$1" meth="$2" seed="$3"
  echo "[$(date)] >>> gen $meth (n=$n)"
  $PY scripts/40_gen_synth.py --n "$n" --methods "$meth" --seed "$seed" \
      --out-dir data/synth/full --out-csv "artifacts/synth_$meth.csv" > "logs/s_gen_$meth.log" 2>&1 \
    && echo "[$(date)] $meth OK ($(grep -c done logs/s_gen_$meth.log))" \
    || { echo "[$(date)] $meth FAILED"; tail -4 logs/s_gen_$meth.log; }
}

# ---- Phase 1: generate diverse synthetic fraud (GenAI-weighted) ----
gen 4000 faceswap 10
gen 1500 splice   11
gen 1200 sd       12
gen 800  text     13

# ---- combine synth + pseudo_ens into one extra-csv ----
$PY - <<"PYEOF"
import pandas as pd, glob
from pathlib import Path
parts=[]
for f in glob.glob("artifacts/synth_*.csv"):
    parts.append(pd.read_csv(f))
pe=Path("artifacts/pseudo_ens.csv")
if pe.exists(): parts.append(pd.read_csv(pe))
df=pd.concat(parts, ignore_index=True).drop_duplicates("id")
df.to_csv("artifacts/extra_synth.csv", index=False)
synth=sum(1 for p in parts[:-1] for _ in [0]) if False else None
print(f"[combine] extra_synth.csv: {len(df):,} rows (synth+pseudo)")
PYEOF
echo "synth generados: $(find data/synth/full -name '*.jpg' | wc -l)"

# ---- Phase 2: finetune MAE backbone + 30k IDNet + (pseudo + synth) ----
echo "[$(date)] >>> train maeFT + synth"
INIT=runs/mae_cnxv2L.pt; [ -f "$INIT" ] || INIT=""
$PY scripts/30_train2.py --train-all --backbone convnextv2_large.fcmae_ft_in22k_in1k \
    ${INIT:+--init-from $INIT} --batch-size 32 --loss bce --aux --max-aux 30000 \
    --extra-csv artifacts/extra_synth.csv --epochs 3 \
    --save-name fusion_nofusion_all_maeFT_synth > logs/s_train.log 2>&1 \
  && { echo "[$(date)] train OK"; grep "val FREUID" logs/s_train.log | tail -1; } \
  || { echo "[$(date)] train FAILED"; tail -8 logs/s_train.log; }

# ---- Phase 3: submission CSVs (heavy TTA + rank-avg; NO submit, cupo Jun 27) ----
M=runs/fusion_nofusion_all_maeFT_synth.pt; M0=runs/fusion_nofusion_all_maeFT.pt
[ -f "$M" ] && $PY scripts/31_polish_submit.py --checkpoints "$M" \
    --out submissions/sub_synth_maeFT.csv > logs/s_sub.log 2>&1 && echo "  sub_synth_maeFT OK"
[ -f "$M" ] && $PY scripts/31_polish_submit.py --checkpoints "$M,$M0" --rank-avg \
    --out submissions/sub_synth_ens.csv > logs/s_sub2.log 2>&1 && echo "  sub_synth_ens OK"

echo "=================================================================="
echo "[$(date)] SYNTH DONE."; ls -la submissions/sub_synth_*.csv 2>/dev/null
echo "=================================================================="
