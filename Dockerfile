# Inference engine image — deployed on Render (see render.yaml), and runs
# unchanged on any container host (Azure Container Apps, Cloud Run, …).
FROM python:3.14-slim

# Hosts such as Render inject PORT; 7860 is the local default.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# CPU-only PyTorch first: the default wheel bundles CUDA (~2 GB) that a CPU
# host never uses. requirements.txt then sees torch as already satisfied.
COPY backend/requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY --chown=app:app backend/ .
USER app

EXPOSE 7860
# --proxy-headers: the platform's proxy is the TCP peer, so take the visitor's
# IP from X-Forwarded-For — otherwise every visitor shares one rate-limit bucket.
CMD ["sh", "-c", "exec uvicorn inference_engine.api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips '*'"]
