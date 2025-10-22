# GE2PE Diacritization API

Production-ready FastAPI service for Persian text diacritization using GE2PE model.

## Features

- ✅ REST API with FastAPI
- ✅ Automatic OpenAPI documentation
- ✅ GPU support with automatic detection
- ✅ LRU caching for repeated queries
- ✅ Batch processing
- ✅ Health check and metrics endpoints
- ✅ CORS enabled

## Installation

```bash
cd api
pip install -r requirements.txt
```

## Environment Variables

- `GE2PE_MODEL_PATH`: Path to trained model checkpoint (default: `./GE2PE/content/checkpoint-320`)
- `USE_GPU`: Force GPU usage (`true`/`false`/`auto`, default: `auto`)

## Running the Server

### Basic Usage

```bash
python api/main.py
```

### With Arguments

```bash
python api/main.py --host 0.0.0.0 --port 8000 --reload
```

### Using uvicorn directly

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### 1. Diacritize Text (POST /diacritize)

**Request:**
```json
{
  "texts": "سلام دنیا",
  "batch_size": 10,
  "use_rules": false,
  "use_dict": false
}
```

Or with multiple texts:
```json
{
  "texts": ["سلام دنیا", "این یک تست است"],
  "batch_size": 10
}
```

**Response:**
```json
{
  "results": ["سَلام دُنیا"],
  "count": 1,
  "processing_time_ms": 45.2
}
```

### 2. Health Check (GET /health)

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "gpu_available": true
}
```

### 3. Metrics (GET /metrics)

**Response:**
```json
{
  "total_requests": 150,
  "diacritize_requests": 145,
  "errors": 2,
  "uptime_seconds": 3600.5
}
```

### 4. API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

Run the test client:

```bash
# Start the server first
python api/main.py

# In another terminal, run tests
python api/test_client.py
```

## Docker Deployment

### Build Image

```bash
docker build -t ge2pe-api .
```

### Run Container

```bash
# CPU only
docker run -p 8000:8000 ge2pe-api

# With GPU
docker run --gpus all -p 8000:8000 ge2pe-api
```

## Example Usage

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/diacritize",
    json={"texts": "سلام دنیا"}
)

print(response.json())
# {"results": ["سَلام دُنیا"], "count": 1, ...}
```

### cURL

```bash
curl -X POST "http://localhost:8000/diacritize" \
  -H "Content-Type: application/json" \
  -d '{"texts": "سلام دنیا"}'
```

### JavaScript (fetch)

```javascript
fetch('http://localhost:8000/diacritize', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({texts: 'سلام دنیا'})
})
.then(r => r.json())
.then(data => console.log(data.results));
```

## Performance

- Single text: ~20-50ms (cached: <1ms)
- Batch of 10: ~100-200ms
- LRU cache: 1000 entries
- GPU speedup: 2-3x vs CPU

## Troubleshooting

### Model not loading
- Check `GE2PE_MODEL_PATH` points to valid checkpoint
- Install missing dependencies: `pip install parsivar`
- Check logs for detailed error messages

### GPU not detected
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA-enabled PyTorch
- Use `USE_GPU=false` to force CPU mode

### Connection refused
- Ensure server is running: `python api/main.py`
- Check port is not in use: `lsof -i :8000`
- Try different port: `--port 8001`

## License

Same as GE2PE project
