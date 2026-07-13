#!/bin/bash
cd /root/freuid
cp -n runs/fusion_nofusion_all.pt runs/fusion_nofusion_30k.pt
python3 -u scripts/24_train_fusion.py --train-all --aux --max-aux 60000 --no-fusion --img-size 384 --batch-size 96 --epochs 3 > logs/train_iter3.log 2>&1
python3 -u scripts/25_predict_fusion.py --checkpoint runs/fusion_nofusion_all.pt --tta --submit --message "ITER3: 60k digital IDNet (2x data) + TTA, default aug" > logs/submit_iter3.log 2>&1
echo ALLDONE > logs/iter3_done
