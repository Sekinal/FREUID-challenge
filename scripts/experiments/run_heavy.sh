#!/bin/bash
cd /root/freuid
python3 -u scripts/24_train_fusion.py --train-all --aux --max-aux 30000 --no-fusion --img-size 384 --batch-size 96 --epochs 3 --aug-strength heavy > logs/train_heavy.log 2>&1
python3 -u scripts/25_predict_fusion.py --checkpoint runs/fusion_nofusion_all_heavy.pt --submit --message "ISOLATED: heavy capture aug only (else=0.187 baseline, no TTA)" > logs/submit_heavy.log 2>&1
echo ALLDONE > logs/heavy_done
