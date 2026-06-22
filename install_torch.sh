#!/bin/bash
cd /root/freuid
echo "[$(date)] instalando torch torchvision timm..."
.venv/bin/pip install torch torchvision timm albumentations 2>&1 | tail -5
.venv/bin/python -c "import torch; print(\"torch\", torch.__version__, \"cuda?\", torch.cuda.is_available())"
echo "[$(date)] torch listo"
