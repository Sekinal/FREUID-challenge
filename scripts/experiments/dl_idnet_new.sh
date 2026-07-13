#!/bin/bash
set -e
cd /root/freuid
mkdir -p data/aux/idnet2025
cd data/aux/idnet2025
for C in EST SVK ESP GRC RUS; do
  echo "[$(date)] downloading $C ..."
  hf download cactuslab/IDNet-2025 ${C}.tar.gz --repo-type dataset --local-dir . || { echo "FAILED $C"; continue; }
  echo "[$(date)] extracting $C ..."
  tar -xzf ${C}.tar.gz && rm -f ${C}.tar.gz
done
echo "[$(date)] IDNet done. images: $(find . -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)"
echo IDNET_DONE > /root/freuid/logs/idnet_done
