cd /root/freuid
echo "--- SMOKE soft pseudo ---"
python3 scripts/43_soft_pseudo.py --teacher runs/fusion_nofusion_all_maeFT.pt --out artifacts/soft_smoke.csv 2>&1 | grep -E "soft|Error" | tail -2; rm -f artifacts/soft_smoke.csv
echo "--- SMOKE SRM (mini) ---"
python3 scripts/42_train_srm.py --backbone convnextv2_base.fcmae_ft_in22k_in1k --smoke --batch-size 8 --max-aux 0 --save-name srm_smoke --out submissions/srm_smoke.csv 2>&1 | grep -E "srm\]|val FREUID|wrote|Error|Trace" | tail -3; rm -f runs/srm_smoke.pt submissions/srm_smoke.csv
echo "--- SMOKE res512 (mini) ---"
python3 scripts/30_train2.py --train-all --backbone convnextv2_large.fcmae_ft_in22k_in1k --init-from runs/mae_cnxv2L.pt --img-size 512 --batch-size 8 --max-aux 0 --extra-csv artifacts/pseudo_ens.csv --smoke --save-name res_smoke 2>&1 | grep -E "val FREUID|loss=|Error|Trace|OutOfMem" | tail -2; rm -f runs/res_smoke.pt
echo "SMOKE3 DONE"
