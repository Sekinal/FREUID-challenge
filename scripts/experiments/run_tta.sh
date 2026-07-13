#!/bin/bash
cd /root/freuid
python3 -u scripts/25_predict_fusion.py --checkpoint runs/fusion_nofusion_all.pt --tta --submit --message "ITER2: TTA (hflip avg) on the 0.187 baseline model" > logs/submit_tta.log 2>&1
echo ALLDONE > logs/tta_done
