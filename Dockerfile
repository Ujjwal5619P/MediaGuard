# Build from the repository root: docker build -f Dockerfile -t mediaguard-api .
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MEDIAGUARD_HOST=0.0.0.0 \
    MEDIAGUARD_PORT=8000 \
    MEDIAGUARD_LIGHTWEIGHT_MODE=true \
    MEDIAGUARD_AUTO_DOWNLOAD_MODELS=true \
    MEDIAGUARD_MODELS_DIR=/tmp/mediaguard/models \
    MEDIAGUARD_DATA_DIR=/tmp/mediaguard/data \
    MEDIAGUARD_DB_PATH=/tmp/mediaguard/data/mediaguard.db \
    MEDIAGUARD_TEMP_DIR=/tmp/mediaguard/tmp \
    MEDIAGUARD_ARTIFACTS_DIR=/tmp/mediaguard/artifacts \
    HF_HOME=/tmp/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements-render.txt ./requirements-render.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-render.txt

COPY backend/app ./app
COPY backend/run_server.py ./run_server.py

EXPOSE 8000
CMD ["python", "run_server.py"]
