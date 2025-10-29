# Multi-stage Dockerfile for RunPod Serverless Deployment
# Target: <3GB container, <2s cold start with FlashBoot
# Optimized for: NVIDIA T4/A100 GPUs

# ==============================================================================
# Stage 1: Builder - Install dependencies and compile
# ==============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m pip install --upgrade pip setuptools wheel
RUN python3.11 -m venv /opt/venv

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
WORKDIR /app
COPY pyproject.toml ./

# Install Python dependencies (excluding heavy optional deps)
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    transformers==4.38.0 \
    anthropic==0.18.1 \
    fastapi==0.109.2 \
    uvicorn[standard]==0.27.1 \
    pydantic==2.6.1 \
    pydantic-settings==2.1.0 \
    numpy==1.26.4 \
    pandas==2.2.0 \
    alpaca-py==0.20.2 \
    sentence-transformers==2.3.1 \
    faiss-gpu==1.7.4 \
    python-dotenv==1.0.1 \
    python-json-logger==2.0.7 \
    pyyaml==6.0.1 \
    requests==2.31.0

# Install TA library (pure Python version to avoid compilation issues)
RUN pip install --no-cache-dir ta==0.11.0

# Optional: Install TimesFM if available (may increase image size)
# RUN pip install --no-cache-dir timesfm[torch] || echo "TimesFM not available"

# ==============================================================================
# Stage 2: Runtime - Minimal production image
# ==============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # GPU visibility
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # RunPod specific
    RUNPOD_POD_ID=${RUNPOD_POD_ID:-unknown} \
    # Application settings
    APP_ENV=production \
    LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    # Model settings
    MODEL_CACHE_DIR=/app/model_cache \
    HF_HOME=/app/model_cache/huggingface \
    TRANSFORMERS_CACHE=/app/model_cache/transformers \
    TORCH_HOME=/app/model_cache/torch

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY deployment/api/ /app/deployment/api/
COPY deployment/runpod/ /app/deployment/runpod/

# Create directories
RUN mkdir -p \
    /app/model_cache \
    /app/logs \
    /app/data_cache \
    && chmod -R 777 /app/model_cache /app/logs /app/data_cache

# Download and cache models during build (reduces cold start time)
# This pre-loads the models into the container
RUN python3.11 -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    model_name='yiyanghkust/finbert-tone'; \
    print('Downloading FinBERT...'); \
    AutoTokenizer.from_pretrained(model_name, cache_dir='/app/model_cache/transformers'); \
    AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='/app/model_cache/transformers'); \
    print('FinBERT cached successfully')" || echo "Warning: Could not pre-cache FinBERT"

RUN python3.11 -c "from sentence_transformers import SentenceTransformer; \
    model_name='sentence-transformers/all-mpnet-base-v2'; \
    print('Downloading sentence transformer...'); \
    SentenceTransformer(model_name, cache_folder='/app/model_cache/sentence_transformers'); \
    print('Sentence transformer cached successfully')" || echo "Warning: Could not pre-cache sentence transformer"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden by RunPod)
CMD ["python3.11", "-u", "deployment/runpod/handler.py"]
