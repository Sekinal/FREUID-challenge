#!/bin/bash
cd /root/freuid
echo "[$(date)] bajando inswapper_128.onnx..."
python3 -c "from huggingface_hub import hf_hub_download; import shutil; p=hf_hub_download(\"ezioruan/inswapper_128.onnx\",\"inswapper_128.onnx\"); shutil.copy(p,\"models_synth/inswapper_128.onnx\"); print(\"inswapper OK\")" 2>&1 | tail -2
echo "[$(date)] prefetch SD2-inpaint..."
python3 -c "
import warnings; warnings.filterwarnings(\"ignore\")
from diffusers import AutoPipelineForInpainting
import torch
p=AutoPipelineForInpainting.from_pretrained(\"stabilityai/stable-diffusion-2-inpainting\", torch_dtype=torch.float16)
print(\"SD2-inpaint OK\")
" 2>&1 | tail -2
echo "[$(date)] EDITORS DONE"
