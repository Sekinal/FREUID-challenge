#!/bin/bash
cd /root/freuid
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "[$(date)] IDSpace Mobile_images (0.65GB, captura movil con labels)..."
cd data/aux/idspace
/root/freuid/.venv/bin/hf download cactuslab/IDSpace Mobile_images.tar --repo-type dataset --local-dir . 2>&1 | tail -1 || python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(\"cactuslab/IDSpace\",\"Mobile_images.tar\",repo_type=\"dataset\",local_dir=\".\")" 2>&1 | tail -1
tar -xf Mobile_images.tar 2>/dev/null && rm -f Mobile_images.tar
echo "[$(date)] estructura IDSpace mobile:"; find . -maxdepth 3 -type d | head -20
echo "[$(date)] imgs:"; find . -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | wc -l
echo "[$(date)] === MIDV-500 (capturas reales)..."
cd /root/freuid/data/aux/midv
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\"Noaman/midv500\",repo_type=\"dataset\",local_dir=\".\")" 2>&1 | tail -1
find . -type f \( -iname "*.tif" -o -iname "*.jpg" -o -iname "*.png" \) 2>/dev/null | wc -l
echo "[$(date)] CAPTURE DL DONE"
