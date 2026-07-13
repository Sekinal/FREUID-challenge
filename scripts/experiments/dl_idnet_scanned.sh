#!/bin/bash
set -e
cd /root/freuid
mkdir -p data/aux/idnet2025_scanned
cd data/aux/idnet2025_scanned
for C in EST ESP SVK GRC RUS; do
  echo "[$(date)] scanned $C ..."
  hf download cactuslab/IDNet-2025 ${C}_scanned.tar.gz --repo-type dataset --local-dir . 2>/dev/null && { tar -xzf ${C}_scanned.tar.gz && rm -f ${C}_scanned.tar.gz; echo "  ok $C"; } || echo "  no scanned for $C"
done
echo "[$(date)] scanned images: $(find . -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)"
echo SCANNED_DONE > /root/freuid/logs/scanned_done
