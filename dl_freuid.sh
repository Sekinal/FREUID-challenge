#!/bin/bash
set -e
cd /root/freuid
export KAGGLE_API_TOKEN=$(cat .kaggle/access_token)
mkdir -p data/raw data/extracted
echo "[$(date)] descargando dataset FREUID..."
.venv/bin/kaggle competitions download -c the-freuid-challenge-2026-ijcai-ecai -p data/raw
echo "[$(date)] descarga lista, extrayendo..."
.venv/bin/python -c "import zipfile,glob; z=glob.glob(\"data/raw/*.zip\")[0]; zipfile.ZipFile(z).extractall(\"data/extracted\"); print(\"extraido\", z)"
echo "[$(date)] FREUID listo. imagenes:"; find data/extracted -name "*.jpeg" | wc -l
