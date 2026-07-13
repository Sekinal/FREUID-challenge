#!/bin/bash
cd /root/freuid
echo "[$(date)] instalando easyocr..."
pip install --break-system-packages --quiet easyocr 2>&1 | tail -1
echo "[$(date)] prefetch SDXL-inpaint (~7GB)..."
python3 -c "
import warnings; warnings.filterwarnings(\"ignore\")
from diffusers import AutoPipelineForInpainting
import torch
p=AutoPipelineForInpainting.from_pretrained(\"diffusers/stable-diffusion-xl-1.0-inpainting-0.1\", torch_dtype=torch.float16, variant=\"fp16\")
print(\"SDXL-inpaint OK\")
" 2>&1 | tail -2
echo "[$(date)] prefetch easyocr models..."
python3 -c "import easyocr; r=easyocr.Reader([\"en\"], gpu=True); print(\"easyocr OK\")" 2>&1 | tail -1
echo "[$(date)] EDITORS2 DONE"
