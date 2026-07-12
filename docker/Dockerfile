# FREUID Challenge 2026 — reproducibility image (no-network sandbox contract)
#
# Runs inference on /data (read-only, flat image files) and writes
# /submissions/submission.csv. All weights are embedded; no downloads at runtime.
#
# Build (from repo root, after placing the released weights in docker/weights/):
#   docker build -t freuid2026-eliastsj -f docker/Dockerfile .
# Run (sandbox contract):
#   docker run --rm --network none --gpus all --shm-size 2g \
#     -v /path/to/images:/data:ro -v /path/to/out:/submissions \
#     freuid2026-eliastsj

FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime

RUN pip install --no-cache-dir --break-system-packages timm==1.0.27 pandas==2.2.3 pillow==11.0.0

WORKDIR /app
COPY infer.py /app/infer.py
COPY docker/weights/ /app/weights/

# FREUID_MODEL: "robust" (default; 3-seed capture-robust ensemble, the primary
# final submission) or "public" (single-seed public-LB specialist).
ENV FREUID_MODEL=robust
ENV FREUID_BATCH=16
ENV FREUID_WORKERS=4

COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
