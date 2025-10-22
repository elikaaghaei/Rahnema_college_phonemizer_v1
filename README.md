# 🎯 GE2PE Persian Diacritization System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elikaaghaei/Rahnema_college_phonemizer_v1/blob/main/GE2PE_Colab_Demo.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Complete Persian diacritization and phonemization system using T5-based GE2PE model with production-ready FastAPI service.

## 🚀 Quick Start - Google Colab (Recommended)

**بدون نیاز به نصب، فقط یک کلیک!**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elikaaghaei/Rahnema_college_phonemizer_v1/blob/main/GE2PE_Colab_Demo_Fixed.ipynb)

مزایا:
- ✅ GPU رایگان
- ✅ بدون نیاز به فضای دیسک
- ✅ اجرا در 2 دقیقه
- ✅ UI زیبا با Gradio (بدون نیاز به authtoken)
- ✅ استفاده از GE2PE واقعی

## 📋 Features

- 🎯 **Dataset Loader**: Persian text preprocessing with normalization
- 🔤 **Tokenizer**: Character and word-level tokenization with vocabulary management
- 🏋️ **Training Script**: Complete training loop with validation and checkpointing
- 🌐 **FastAPI Service**: Production-ready REST API with caching
- 🐳 **Docker Support**: Complete containerization with GPU support
- 📊 **Monitoring**: Health checks, metrics, and request tracking
- 🧪 **Testing**: Comprehensive test suite included

## 🏗️ Project Structure

```
.
├── api/                    # FastAPI service
│   ├── main.py            # Main API endpoints
│   ├── main_mock.py       # Mock API for testing
│   ├── test_client.py     # Test suite
│   └── README.md          # API documentation
├── data/                   # Dataset and tokenization
│   ├── loader.py          # Dataset loader
│   ├── tokenizer.py       # Persian tokenizer
│   └── requirements.txt
├── model/                  # Training scripts
│   ├── train_ge2pe.py     # Training loop
│   └── requirements.txt
├── GE2PE/                  # GE2PE model
│   └── GE2PE.py           # Model class
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker orchestration
├── GE2PE_Colab_Demo.ipynb # Google Colab demo
└── README.md              # This file
```

## 🛠️ Installation

### Method 1: Local Setup (Minimal - 3GB)

```bash
# Clone repository
git clone https://github.com/elikaaghaei/Rahnema_college_phonemizer_v1.git
cd Rahnema_college_phonemizer_v1

# Install minimal dependencies (CPU)
pip install --no-cache-dir \
  torch --index-url https://download.pytorch.org/whl/cpu \
  transformers fastapi uvicorn pydantic

# Run mock API
python api/main_mock.py

# Access API
open http://localhost:8000/docs
```

### Method 2: Full Setup (5-6GB)

```bash
# Install all dependencies
pip install -r api/requirements.txt
pip install -r data/requirements.txt
pip install -r model/requirements.txt
pip install parsivar

# Run full API
python api/main.py
```

### Method 3: Docker (7-8GB)

```bash
# Build and run
docker-compose up --build

# Or use the automated script
./docker-run.sh

# Access API
open http://localhost:8000/docs
```

## 📖 Usage

### Python API

```python
import requests

# Diacritize text
response = requests.post(
    "http://localhost:8000/diacritize",
    json={"texts": "سلام دنیا"}
)

print(response.json())
# {"results": ["سَلام دُنیا"], "count": 1, "processing_time_ms": 45.2}
```

### cURL

```bash
curl -X POST "http://localhost:8000/diacritize" \
  -H "Content-Type: application/json" \
  -d '{"texts": "سلام دنیا"}'
```

### Dataset Loader

```python
from data.loader import PhonemizerDataset, train_val_split

# Load dataset
dataset = PhonemizerDataset(
    'phonemizer _dataset_v1.csv/phonemizer _dataset_v1.csv',
    mode='char',
    max_samples=1000
)

# Split
train, val = train_val_split(dataset, val_frac=0.2)
print(f"Train: {len(train)}, Val: {len(val)}")
```

### Tokenizer

```python
from data.tokenizer import PersianTokenizer

# Create tokenizer
tokenizer = PersianTokenizer()

# Build vocabulary
tokenizer.build_vocab(['سلام', 'دنیا'], mode='char')

# Encode/Decode
text = "سلام"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
print(f"{text} → {encoded} → {decoded}")
```

## 🧪 Testing

### Run Test Suite

```bash
# API tests
python api/test_client.py

# Unit tests (if available)
pytest tests/
```

### Quick Verification

```bash
# Health check
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics

# Test diacritization
curl -X POST http://localhost:8000/diacritize \
  -H "Content-Type: application/json" \
  -d '{"texts": "تست"}'
```

## 🐳 Docker Deployment

### Quick Start

```bash
# Using docker-compose (recommended)
docker-compose up --build

# Using automated script
./docker-run.sh

# Manual Docker commands
docker build -t ge2pe-phonemizer .
docker run -d -p 8000:8000 --name ge2pe-api ge2pe-phonemizer
```

### With GPU

```bash
# Install nvidia-docker first
docker run -d -p 8000:8000 --gpus all ge2pe-phonemizer
```

See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for detailed instructions.

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/metrics` | GET | Request statistics |
| `/diacritize` | POST | Main diacritization endpoint |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

## 🎓 Training

```bash
# Run training script
python model/train_ge2pe.py

# Or use Google Colab for free GPU
# See GE2PE_Colab_Demo.ipynb
```

## 💾 Space Requirements

| Setup | Disk Space | RAM | GPU |
|-------|-----------|-----|-----|
| Minimal (CPU) | 3-4 GB | 4 GB | No |
| Standard | 5-6 GB | 8 GB | Optional |
| Docker | 7-8 GB | 8 GB | Optional |
| **Google Colab** | **0 GB** | **12 GB** | **✅ Free** |

## 🔧 Configuration

### Environment Variables

```bash
# Model path
export GE2PE_MODEL_PATH=/path/to/checkpoint

# GPU usage
export USE_GPU=auto  # auto, true, or false

# API port
export PORT=8000
```

### Docker Volumes

Mount local directories:

```yaml
volumes:
  - ./checkpoints:/app/GE2PE/content
  - ./data:/app/data
  - ./api:/app/api  # for hot-reload
```

## 📚 Documentation

- [API Documentation](api/README.md)
- [Docker Guide](DOCKER_GUIDE.md)
- [GE2PE Model](GE2PE/README.md)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project uses the GE2PE model which has its own license. See [LICENSE](LICENSE).

## 🙏 Acknowledgments

- GE2PE model: Original implementation
- Rahnema College: Project support
- Transformers library: Hugging Face
- FastAPI: Modern web framework

## 📧 Contact

For questions or issues:
- GitHub Issues: https://github.com/elikaaghaei/Rahnema_college_phonemizer_v1/issues
- Email: [Your email]

## 🎯 Next Steps

- [ ] Fine-tune model on larger dataset
- [ ] Add more evaluation metrics
- [ ] Create Gradio web interface
- [ ] Telegram bot integration
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline with GitHub Actions

---

**Made with ❤️ for Persian NLP**

[![Star this repo](https://img.shields.io/github/stars/elikaaghaei/Rahnema_college_phonemizer_v1?style=social)](https://github.com/elikaaghaei/Rahnema_college_phonemizer_v1)
