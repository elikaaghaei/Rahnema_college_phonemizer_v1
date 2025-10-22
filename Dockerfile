# =============================================================================
# GE2PE Persian Diacritization API - Production Dockerfile
# =============================================================================
# Base image: Python 3.11 slim for smaller size
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr (better for Docker logs)
ENV PYTHONUNBUFFERED=1

# Disable pip cache to reduce image size
ENV PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install system dependencies for PyTorch and FastAPI
# - build-essential: C/C++ compilers for some Python packages
# - curl: For healthchecks
# - git: May be needed for some pip installs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy all requirements files first (better caching)
COPY api/requirements.txt /app/api/requirements.txt
COPY data/requirements.txt /app/data/requirements.txt
COPY model/requirements.txt /app/model/requirements.txt

# Install Python dependencies
# Install API deps first (most likely to change)
RUN pip install --upgrade pip && \
    pip install -r /app/api/requirements.txt && \
    pip install -r /app/data/requirements.txt && \
    pip install -r /app/model/requirements.txt

# Copy application code
# (Doing this after pip install allows better layer caching)
COPY api/ /app/api/
COPY data/ /app/data/
COPY model/ /app/model/
COPY GE2PE/ /app/GE2PE/

# Create directory for model checkpoints
RUN mkdir -p /app/GE2PE/content

# Expose port for FastAPI
EXPOSE 8000

# Health check - ping the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: start FastAPI with uvicorn
# --host 0.0.0.0: Allow external connections
# --port 8000: Listen on port 8000
# --workers 1: Single worker (increase for production)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# =============================================================================
# TEACHING: Docker Best Practices for Python ML Projects
# =============================================================================
#
# 📚 DOCKERFILE STRUCTURE:
#
# 1. BASE IMAGE (line 6):
#    - python:3.11-slim = Python 3.11 with minimal OS packages
#    - ~120MB vs ~900MB for full python:3.11
#    - Trade-off: may need to install some system libs manually
#
# 2. ENVIRONMENT VARIABLES (lines 9-12):
#    - PYTHONUNBUFFERED=1: Print output immediately (don't buffer)
#    - PIP_NO_CACHE_DIR=1: Don't cache pip downloads (saves ~100MB)
#
# 3. SYSTEM DEPENDENCIES (lines 18-22):
#    - build-essential: Needed for compiling Python packages (numpy, etc.)
#    - --no-install-recommends: Don't install suggested packages
#    - rm -rf /var/lib/apt/lists/*: Clean up apt cache (saves ~50MB)
#
# 4. LAYER CACHING (lines 25-35):
#    - Copy requirements.txt FIRST, then pip install
#    - Copy app code LAST
#    - Why? If code changes, only that layer rebuilds (not pip install)
#    - Saves 5-10 minutes on rebuilds!
#
# 5. HEALTHCHECK (lines 45-46):
#    - Docker/K8s can auto-restart if health fails
#    - Checks /health endpoint every 30 seconds
#
# 6. CMD (line 53):
#    - JSON array format ["cmd", "arg1", "arg2"]
#    - Better than shell form: faster startup, proper signal handling
#    - --workers 1: Single process (increase to 4 for production)
#
# ⚠️ THREE COMMON DOCKER PITFALLS:
#
# 1. HUGE IMAGE SIZE:
#    - Problem: 2-3GB image because of full Python base + pip cache
#    - Solution: Use slim base, multi-stage builds, .dockerignore
#    - Check size: docker images | grep phonemizer
#    - Good size: <1GB, Great: <500MB
#
# 2. GPU ACCESS NOT WORKING:
#    - Problem: torch.cuda.is_available() returns False in container
#    - Solution: Use nvidia-docker runtime
#    - Build: docker build -t phonemizer .
#    - Run: docker run --gpus all -p 8000:8000 phonemizer
#    - Check: nvidia-smi inside container
#
# 3. MISSING SYSTEM LIBRARIES:
#    - Problem: "ImportError: libgomp.so.1: cannot open shared object"
#    - Cause: PyTorch/NumPy need system libs not in slim image
#    - Solution: Install build-essential or specific lib (libgomp1)
#    - Debug: docker run -it phonemizer /bin/bash, then import torch
#
# 🎯 NEXT STEPS FOR PRODUCTION:
#
# 1. KUBERNETES DEPLOYMENT:
#    - Create k8s manifests: deployment.yaml, service.yaml, ingress.yaml
#    - Use ConfigMaps for environment variables
#    - Use Secrets for API keys, model paths
#    - Auto-scaling: HPA based on CPU/memory or custom metrics (requests/sec)
#    - Example: kubectl apply -f k8s/
#
# 2. MLOPS INTEGRATION:
#    - MLflow: Track experiments, log metrics, version models
#    - DVC: Version large datasets and model checkpoints
#    - Kubeflow: End-to-end ML pipelines on Kubernetes
#    - Airflow: Schedule training jobs, data preprocessing
#
# 3. MONITORING & OBSERVABILITY:
#    - Prometheus: Scrape /metrics endpoint (add prometheus_client)
#    - Grafana: Visualize metrics (latency, throughput, errors)
#    - Jaeger: Distributed tracing for request flows
#    - ELK Stack: Centralized logging (Elasticsearch, Logstash, Kibana)
#    - Alerts: PagerDuty/Slack for high error rates or latency spikes
#
# 💡 OPTIMIZATION TIPS:
#    - Multi-stage build: Build wheels in one stage, copy to runtime stage
#    - Model quantization: Convert FP32 → FP16/INT8 (2-4x speedup)
#    - Batch inference: Queue requests, process in batches
#    - Redis cache: Distributed caching across multiple containers
#    - CDN: Cache responses at edge for frequently requested texts
