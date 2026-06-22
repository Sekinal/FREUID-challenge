#!/bin/bash
set -e
cd /root/freuid
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p data/aux/idnet2025
cd data/aux/idnet2025
for C in EST SVK ESP; do
  echo "[$(date)] descargando $C.tar.gz ..."
  /root/freuid/.venv/bin/hf download cactuslab/IDNet-2025 ${C}.tar.gz --repo-type dataset --local-dir .
  echo "[$(date)] extrayendo $C ..."
  tar -xzf ${C}.tar.gz && rm -f ${C}.tar.gz
done
echo "[$(date)] IDNet listo. Estructura:"
find . -maxdepth 2 -type d | head -40
echo "total imagenes idnet:"; find . -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l
