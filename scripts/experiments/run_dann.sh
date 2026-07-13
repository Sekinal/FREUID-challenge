#!/bin/bash
cd /root/freuid
exec > >(tee -a dann.log) 2>&1
echo "[$(date)] DANN START"
echo "[$(date)] >>> LOCO-DANN (valida invariancia de pais vs baseline 0.4023)"
python3 scripts/50_train_dann.py --loto-type "MOZAMBIQUE/DL" --backbone convnextv2_large.fcmae_ft_in22k_in1k \
    --init-from runs/mae_cnxv2L.pt --img-size 384 --batch-size 32 --max-aux 30000 --epochs 3 \
    --save-name dann_loto > logs/dann_loto.log 2>&1 \
  && { echo "[$(date)] LOCO-DANN OK"; grep "CROSS-COUNTRY\|val FREUID" logs/dann_loto.log | tail -2; } \
  || { echo "[$(date)] LOCO-DANN FAILED"; tail -6 logs/dann_loto.log; }
echo "[$(date)] >>> DANN all-5 (modelo privado candidato)"
python3 scripts/50_train_dann.py --train-all --backbone convnextv2_large.fcmae_ft_in22k_in1k \
    --init-from runs/mae_cnxv2L.pt --img-size 384 --batch-size 32 --max-aux 30000 --epochs 3 \
    --save-name dann_all > logs/dann_all.log 2>&1 \
  && { echo "[$(date)] DANN-all OK"; grep "val FREUID" logs/dann_all.log | tail -1; } \
  || { echo "[$(date)] DANN-all FAILED"; tail -6 logs/dann_all.log; }
echo "[$(date)] DANN DONE. VEREDICTO LOCO:"; grep -h CROSS-COUNTRY logs/dann_loto.log | tail -1
