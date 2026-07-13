#!/bin/bash
set -e
cd /root/freuid/data/aux/fantasyid
echo "[$(date)] descargando FantasyID.tgz (2.5GB)..."
wget -q "https://zenodo.org/records/17063366/files/FantasyID.tgz?download=1" -O FantasyID.tgz
echo "[$(date)] extrayendo..."
tar -xzf FantasyID.tgz && rm -f FantasyID.tgz
echo "[$(date)] FantasyID listo"
find . -maxdepth 3 -type d | head -30
