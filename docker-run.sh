#!/bin/bash
# =============================================================================
# Quick Docker Run Script for GE2PE API
# =============================================================================

set -e  # Exit on error

echo "============================================================"
echo "🐳 GE2PE Diacritization API - Docker Deployment"
echo "============================================================"

# Configuration
IMAGE_NAME="ge2pe-phonemizer"
CONTAINER_NAME="ge2pe-api"
PORT="8000"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

success "Docker is installed"

# Build the image
info "Building Docker image..."
docker build -t $IMAGE_NAME .
success "Image built: $IMAGE_NAME"

# Stop and remove existing container if running
if docker ps -a | grep -q $CONTAINER_NAME; then
    info "Stopping existing container..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

# Check for GPU support
if command -v nvidia-smi &> /dev/null; then
    info "NVIDIA GPU detected. Running with GPU support..."
    GPU_FLAG="--gpus all"
else
    warning "No GPU detected. Running on CPU..."
    GPU_FLAG=""
fi

# Run the container
info "Starting container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8000 \
    $GPU_FLAG \
    -e USE_GPU=auto \
    --restart unless-stopped \
    $IMAGE_NAME

success "Container started: $CONTAINER_NAME"

# Wait for container to be healthy
info "Waiting for API to be ready..."
sleep 5

# Check health
if curl -f http://localhost:$PORT/health &> /dev/null; then
    success "API is healthy!"
else
    warning "API may still be starting. Check logs with: docker logs $CONTAINER_NAME"
fi

echo ""
echo "============================================================"
echo "📡 API Endpoints:"
echo "============================================================"
echo "  Main API:     http://localhost:$PORT"
echo "  Health:       http://localhost:$PORT/health"
echo "  Metrics:      http://localhost:$PORT/metrics"
echo "  Swagger Docs: http://localhost:$PORT/docs"
echo ""
echo "============================================================"
echo "🔧 Useful Commands:"
echo "============================================================"
echo "  View logs:    docker logs -f $CONTAINER_NAME"
echo "  Stop:         docker stop $CONTAINER_NAME"
echo "  Restart:      docker restart $CONTAINER_NAME"
echo "  Remove:       docker rm -f $CONTAINER_NAME"
echo "  Shell access: docker exec -it $CONTAINER_NAME /bin/bash"
echo "============================================================"
echo ""

# Test the API
info "Testing API with sample request..."
TEST_RESPONSE=$(curl -s -X POST http://localhost:$PORT/diacritize \
    -H "Content-Type: application/json" \
    -d '{"texts": "سلام"}' 2>/dev/null || echo "Request failed")

if [[ $TEST_RESPONSE == *"results"* ]]; then
    success "API test successful!"
    echo "Response: $TEST_RESPONSE"
else
    warning "API test failed or model not loaded"
    echo "Run this to check: docker logs $CONTAINER_NAME"
fi

echo ""
success "Deployment complete! 🚀"
