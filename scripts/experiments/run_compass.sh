#!/bin/bash
# Overnight "compass" plan. Waits for round2, then:
#  P1: validate the LOTO proxy against KNOWN LB scores (base/PL/60k)
#  P2: rank NEW candidates (PL3, EffNetV2-L+PL, PL+2stage) by the proxy + train submission models
#  P3: build strong-model ensemble CSVs for tomorrow
cd /root/freuid
PY=python3
exec > >(tee -a compass.log) 2>&1
mkdir -p logs submissions artifacts
echo "=================================================================="
echo "[$(date)] COMPASS START -- esperando round2..."
for i in $(seq 1 120); do tmux has-session -t round2 2>/dev/null && sleep 30 || break; done
echo "[$(date)] round2 terminado. PL2 = $(ls runs/fusion_nofusion_all_pl2.pt 2>/dev/null || echo MISSING)"

LOTO="MOZAMBIQUE/DL"   # fixed held-out country (hardest cross-country signal)
run () {  # name, args...
  local name="$1"; shift
  echo "[$(date)] >>> $name"
  $PY scripts/30_train2.py "$@" --save-name "$name" > "logs/c_$name.log" 2>&1 \
    && { echo "[$(date)] $name OK"; grep -E "PROXY|val FREUID" "logs/c_$name.log" | tail -2; } \
    || { echo "[$(date)] $name FAILED"; tail -6 "logs/c_$name.log"; }
}

echo "################ PHASE 1: validar la brujula (LOTO vs LB conocido) ################"
run loto_base --loto-type "$LOTO" --aux --max-aux 30000 --epochs 3
run loto_pl   --loto-type "$LOTO" --aux --max-aux 30000 --extra-csv artifacts/pseudo.csv --epochs 3
run loto_60k  --loto-type "$LOTO" --aux --max-aux 60000 --epochs 3

echo "################ PHASE 1b: correlacion proxy vs LB ################"
$PY - <<"PYEOF"
import re, pathlib
known = {"loto_base":0.187, "loto_pl":0.172, "loto_60k":0.220}  # public LB
rows=[]
for n in known:
    t=pathlib.Path(f"logs/c_{n}.log")
    proxy=None
    if t.exists():
        m=re.findall(r"CROSS-COUNTRY-FREUID=([0-9.]+)", t.read_text())
        if m: proxy=float(m[-1])
    rows.append((n, proxy, known[n]))
print(f"{'recipe':<12}{'LOTO-proxy':>12}{'known-LB':>10}")
for n,p,lb in rows:
    print(f"{n:<12}{(f'{p:.4f}' if p else 'NA'):>12}{lb:>10.3f}")
ok=[r for r in rows if r[1] is not None]
if len(ok)>=2:
    by_proxy=[r[0] for r in sorted(ok,key=lambda r:r[1])]
    by_lb   =[r[0] for r in sorted(ok,key=lambda r:r[2])]
    print("rank by proxy:", by_proxy)
    print("rank by LB   :", by_lb)
    print("COMPASS TRUSTWORTHY:" , "YES (ranking matches)" if by_proxy==by_lb else "PARTIAL/NO (ranking differs)")
PYEOF

echo "################ PHASE 2: candidatos nuevos (proxy + modelo submission) ################"
# fresh pseudo-labels from the latest best (PL2 if present, else PL)
TEACHER=runs/fusion_nofusion_all_pl2.pt; [ -f "$TEACHER" ] || TEACHER=runs/fusion_nofusion_all_pl.pt
$PY scripts/27_make_pseudolabels.py --checkpoint $TEACHER --lo 0.05 --hi 0.95 --out artifacts/pseudo3.csv > logs/c_pl3make.log 2>&1
grep kept logs/c_pl3make.log | tail -1

# --- candidate A: PL3 (multi-round self-training) ---
run loto_pl3 --loto-type "$LOTO" --aux --max-aux 30000 --extra-csv artifacts/pseudo3.csv --epochs 3
run fusion_nofusion_all_pl3 --train-all --aux --max-aux 30000 --extra-csv artifacts/pseudo3.csv --epochs 3
# --- candidate B: EffNetV2-L + PL (strong 2nd backbone for ensemble diversity) ---
run loto_effL_pl --loto-type "$LOTO" --backbone tf_efficientnetv2_l.in21k_ft_in1k --batch-size 64 --aux --max-aux 30000 --extra-csv artifacts/pseudo3.csv --epochs 3
run fusion_nofusion_all_effL_pl --train-all --backbone tf_efficientnetv2_l.in21k_ft_in1k --batch-size 64 --aux --max-aux 30000 --extra-csv artifacts/pseudo3.csv --epochs 3
# --- candidate C: PL + 2-stage (pretrain IDNet -> finetune FREUID+pseudo) ---
run loto_pl_2stage --loto-type "$LOTO" --init-from runs/fusion_nofusion_stageA.pt --aux --max-aux 30000 --extra-csv artifacts/pseudo3.csv --lr 3e-5 --epochs 3
run fusion_nofusion_all_pl_2stage --train-all --init-from runs/fusion_nofusion_stageA.pt --aux --max-aux 30000 --extra-csv artifacts/pseudo3.csv --lr 3e-5 --epochs 3

echo "################ PHASE 2b: ranking de candidatos por proxy ################"
$PY - <<"PYEOF"
import re, glob, pathlib
print(f"{'candidate':<20}{'LOTO-proxy(cross-country FREUID, lower=better)':>20}")
res=[]
for t in sorted(glob.glob("logs/c_loto_*.log")):
    txt=pathlib.Path(t).read_text()
    m=re.findall(r"CROSS-COUNTRY-FREUID=([0-9.]+)", txt)
    if m: res.append((pathlib.Path(t).stem.replace("c_",""), float(m[-1])))
for n,p in sorted(res,key=lambda r:r[1]):
    print(f"{n:<20}{p:>12.4f}")
PYEOF

echo "################ PHASE 3: ensembles fuertes (CSVs para manana, NO submit) ################"
BASE=runs/fusion_nofusion_30k.pt; PL=runs/fusion_nofusion_all_pl.pt
PL2=runs/fusion_nofusion_all_pl2.pt; PL3=runs/fusion_nofusion_all_pl3.pt
EFFL=runs/fusion_nofusion_all_effL_pl.pt; PL2S=runs/fusion_nofusion_all_pl_2stage.pt
ens () { local out="$1"; shift; local cks=""; for c in "$@"; do [ -f "$c" ] && cks="$cks,$c"; done; cks="${cks#,}";
  [ -n "$cks" ] && $PY scripts/28_ensemble_submit.py --checkpoints "$cks" --out "submissions/$out" > "logs/c_$out.log" 2>&1 && echo "  $out <- $cks"; }
ens sub_c_pl3.csv $PL3
ens sub_c_effL.csv $EFFL
ens sub_c_pl_2stage.csv $PL2S
ens sub_c_ens_strong.csv $PL $PL3 $EFFL
ens sub_c_ens_pl_pl3.csv $PL $PL3
ens sub_c_ens_pl3_effL.csv $PL3 $EFFL

echo "=================================================================="
echo "[$(date)] COMPASS DONE. CSVs candidatas:"; ls -la submissions/sub_c_*.csv submissions/sub_r2_*.csv 2>/dev/null
echo "=================================================================="
