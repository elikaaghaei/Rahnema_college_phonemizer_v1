# 🐳 Docker Deployment Guide - GE2PE API

Complete guide for deploying the GE2PE Persian diacritization API using Docker.

## Quick Start

### Method 1: Using the automated script

```bash
./docker-run.sh
```

### Method 2: Using Docker Compose (Recommended)

```bash
docker-compose up --build
```

### Method 3: Manual Docker commands

```bash
# Build
docker build -t ge2pe-phonemizer .

# Run
docker run -d -p 8000:8000 --name ge2pe-api ge2pe-phonemizer

# With GPU
docker run -d -p 8000:8000 --gpus all --name ge2pe-api ge2pe-phonemizer
```

## Verification Checklist

After starting the container, verify:

```bash
# 1. Check container is running
docker ps | grep ge2pe

# 2. Check health endpoint
curl http://localhost:8000/health

# 3. Test diacritization
curl -X POST http://localhost:8000/diacritize \
  -H "Content-Type: application/json" \
  -d '{"texts": "سلام دنیا"}'
```

## Docker Files Overview

| File | Purpose |
|------|---------|
| `Dockerfile` | Main image definition with multi-layer caching |
| `docker-compose.yml` | Service orchestration with resource limits |
| `.dockerignore` | Excludes unnecessary files from build context |
| `docker-run.sh` | Automated build and deployment script |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GE2PE_MODEL_PATH` | `/app/GE2PE/content/checkpoint-320` | Path to model checkpoint |
| `USE_GPU` | `auto` | GPU usage: `true`, `false`, or `auto` |
| `PYTHONUNBUFFERED` | `1` | Unbuffered Python output for logs |

## Volume Mounts

Mount local directories to persist data or enable hot-reload:

```yaml
volumes:
  # Model checkpoints
  - ./checkpoints:/app/GE2PE/content
  
  # Dataset (for training)
  - ./data:/app/data
  
  # Live code reload (development only)
  - ./api:/app/api
```

## Common Commands

### Build and Run

```bash
# Build only
docker-compose build

# Start in foreground
docker-compose up

# Start in background
docker-compose up -d

# Rebuild and start
docker-compose up --build
```

### Logs and Debugging

```bash
# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f phonemizer

# Execute command in running container
docker-compose exec phonemizer /bin/bash

# Check resource usage
docker stats ge2pe-api
```

### Stop and Clean

```bash
# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Remove everything including images
docker-compose down --rmi all
```

## GPU Support

### Prerequisites

1. Install NVIDIA Docker runtime:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. Verify GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Running with GPU

```bash
# Docker run
docker run --gpus all -p 8000:8000 ge2pe-phonemizer

# Docker Compose (auto-detects)
docker-compose up
```

## Production Deployment

### Resource Limits

Adjust in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'      # 4 CPU cores
      memory: 8G       # 8GB RAM
    reservations:
      cpus: '1.0'
      memory: 2G
```

### Multiple Workers

Edit `Dockerfile` CMD:

```dockerfile
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Behind Nginx

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Kubernetes Deployment

### Basic Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ge2pe-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ge2pe
  template:
    metadata:
      labels:
        app: ge2pe
    spec:
      containers:
      - name: api
        image: ge2pe-phonemizer:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            memory: "4Gi"
            cpu: "2000m"
          requests:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: ge2pe-service
spec:
  selector:
    app: ge2pe
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs ge2pe-api

# Common issues:
# 1. Port already in use
sudo lsof -i :8000
# 2. Permission issues
ls -la ./checkpoints
# 3. Missing dependencies
docker exec -it ge2pe-api pip list
```

### Out of memory

```bash
# Check memory usage
docker stats ge2pe-api

# Increase limit in docker-compose.yml
memory: 8G  # Increase from 4G
```

### Model not loading

```bash
# Check model path
docker exec -it ge2pe-api ls -la /app/GE2PE/content/

# Mount local checkpoint
# In docker-compose.yml:
volumes:
  - ./my-checkpoint:/app/GE2PE/content
```

## Image Size Optimization

Current image: ~1.5GB (with PyTorch)

### Further optimization:

1. **Multi-stage build**:
```dockerfile
# Build stage
FROM python:3.11 as builder
RUN pip wheel --no-cache-dir torch

# Runtime stage
FROM python:3.11-slim
COPY --from=builder /wheels /wheels
RUN pip install /wheels/*
```

2. **Use smaller base**:
```dockerfile
FROM python:3.11-alpine  # ~50MB base (but may need compilation)
```

3. **Remove build tools after install**:
```dockerfile
RUN apt-get remove -y build-essential && apt-get autoremove -y
```

## Monitoring Integration

### Prometheus metrics

Add to `api/main.py`:
```python
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

### Health checks

Docker Compose already includes:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Build and Push Docker Image
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build image
        run: docker build -t ge2pe:${{ github.sha }} .
      - name: Push to registry
        run: docker push ge2pe:${{ github.sha }}
```

## Support

For issues:
1. Check logs: `docker logs ge2pe-api`
2. Verify health: `curl localhost:8000/health`
3. Test API: `curl -X POST localhost:8000/diacritize -d '{"texts":"test"}'`
