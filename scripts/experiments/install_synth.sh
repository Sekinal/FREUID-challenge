#!/bin/bash
P="pip install --break-system-packages --quiet"
echo "[$(date)] instalando opencv..."; $P opencv-python-headless 2>&1 | tail -1
echo "[$(date)] instalando diffusers stack..."; $P diffusers transformers accelerate safetensors 2>&1 | tail -1
echo "[$(date)] instalando insightface..."; $P insightface onnxruntime-gpu 2>&1 | tail -1
echo "[$(date)] verificando..."
python3 -c "import cv2; print(\"opencv\", cv2.__version__)" 2>&1 | tail -1
python3 -c "import diffusers; print(\"diffusers\", diffusers.__version__)" 2>&1 | tail -1
python3 -c "import insightface, onnxruntime; print(\"insightface OK, ort\", onnxruntime.__version__)" 2>&1 | tail -1
echo "[$(date)] INSTALL DONE"
