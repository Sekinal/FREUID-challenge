#!/bin/bash
set -e
cd /root/freuid
# Paises digitales nuevos -> al root existente
mkdir -p data/aux/idnet2025 data/aux/idnet2025_scanned
cd data/aux/idnet2025
for C in GRC RUS; do
  echo "[$(date)] digital $C ..."
  /root/freuid/.venv/bin/hf download cactuslab/IDNet-2025 ${C}.tar.gz --repo-type dataset --local-dir . >/dev/null 2>&1
  tar -xzf ${C}.tar.gz && rm -f ${C}.tar.gz
done
# Versiones scanned (print-and-capture) -> root separado
cd /root/freuid/data/aux/idnet2025_scanned
for C in EST ESP; do
  echo "[$(date)] scanned $C ..."
  /root/freuid/.venv/bin/hf download cactuslab/IDNet-2025 ${C}_scanned.tar.gz --repo-type dataset --local-dir . >/dev/null 2>&1
  tar -xzf ${C}_scanned.tar.gz && rm -f ${C}_scanned.tar.gz
done
echo "[$(date)] EXTRA IDNet listo."
echo "digital dirs:"; ls /root/freuid/data/aux/idnet2025
echo "scanned dirs:"; ls /root/freuid/data/aux/idnet2025_scanned
