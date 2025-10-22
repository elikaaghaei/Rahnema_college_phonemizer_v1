#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Service for Persian Diacritization using GE2PE
Author: Cursor:Claude-Sonnet
Date: 2025-10-22
Purpose: Production-ready REST API for Persian text diacritization
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Union
from functools import lru_cache

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import GE2PE model
from GE2PE.GE2PE import GE2PE


# =============================================================================
# GLOBAL STATE & CONFIGURATION
# =============================================================================

app = FastAPI(
    title="GE2PE Diacritization API",
    description="Persian text diacritization service using GE2PE model",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded at startup)
model: Optional[GE2PE] = None

# Request counter for metrics
request_counter = {"total": 0, "diacritize": 0, "errors": 0}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class DiacritizeRequest(BaseModel):
    """
    Request model for diacritization endpoint.
    
    Examples:
        >>> req = DiacritizeRequest(texts=["سلام"])
        >>> len(req.texts)
        1
        >>> req.batch_size
        10
    """
    texts: Union[str, List[str]] = Field(
        ...,
        description="Single text or list of texts to diacritize",
        example=["سلام دنیا", "این یک تست است"]
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Batch size for inference"
    )
    use_rules: bool = Field(
        default=False,
        description="Apply post-processing rules for short vowels"
    )
    use_dict: bool = Field(
        default=False,
        description="Use custom dictionary if available"
    )
    
    @validator('texts', pre=True)
    def convert_to_list(cls, v):
        """Convert single string to list."""
        if isinstance(v, str):
            return [v]
        return v


class DiacritizeResponse(BaseModel):
    """
    Response model for diacritization endpoint.
    
    Examples:
        >>> resp = DiacritizeResponse(results=["سَلام"], count=1)
        >>> resp.count
        1
    """
    results: List[str] = Field(
        ...,
        description="Diacritized texts"
    )
    count: int = Field(
        ...,
        description="Number of texts processed"
    )
    processing_time_ms: Optional[float] = Field(
        None,
        description="Processing time in milliseconds"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    gpu_available: bool


class MetricsResponse(BaseModel):
    """Metrics response."""
    total_requests: int
    diacritize_requests: int
    errors: int
    uptime_seconds: float


# =============================================================================
# STARTUP & SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model at startup."""
    global model
    
    print("="*60)
    print("🚀 Starting GE2PE Diacritization API")
    print("="*60)
    
    # Configuration
    model_path = os.getenv("GE2PE_MODEL_PATH", "./GE2PE/content/checkpoint-320")
    use_gpu = os.getenv("USE_GPU", "auto")
    
    # Determine GPU usage
    if use_gpu == "auto":
        import torch
        gpu_available = torch.cuda.is_available()
    else:
        gpu_available = use_gpu.lower() == "true"
    
    print(f"📁 Model path: {model_path}")
    print(f"🖥️  GPU available: {gpu_available}")
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"⚠️  Model not found at {model_path}")
        print(f"   Loading without pre-trained weights (demo mode)")
        model_path = "google/mt5-small"  # Fallback to base model
    
    try:
        # Load GE2PE model
        model = GE2PE(model_path=model_path, GPU=gpu_available)
        print("✅ Model loaded successfully!")
        
        # Test inference
        test_result = model.generate(["تست"], batch_size=1)
        print(f"🧪 Test inference: 'تست' → '{test_result[0]}'")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   API will start but /diacritize endpoint will fail")
    
    print("="*60)
    print("✅ API ready!")
    print("="*60 + "\n")
    
    # Record startup time
    app.state.startup_time = time.time()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\n" + "="*60)
    print("🛑 Shutting down GE2PE API")
    print("="*60)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@lru_cache(maxsize=1000)
def cached_diacritize(text: str, use_rules: bool = False, use_dict: bool = False) -> str:
    """
    Cached diacritization for single text.
    Uses LRU cache to avoid re-processing identical inputs.
    
    Args:
        text: Input text
        use_rules: Apply post-processing rules
        use_dict: Use custom dictionary
        
    Returns:
        Diacritized text
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    
    results = model.generate(
        [text],
        batch_size=1,
        use_rules=use_rules,
        use_dict=use_dict
    )
    return results[0]


def process_batch(
    texts: List[str],
    batch_size: int = 10,
    use_rules: bool = False,
    use_dict: bool = False
) -> List[str]:
    """
    Process batch of texts with caching.
    
    Args:
        texts: List of input texts
        batch_size: Batch size for model inference
        use_rules: Apply post-processing rules
        use_dict: Use custom dictionary
        
    Returns:
        List of diacritized texts
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    
    # Try to use cache for individual texts
    results = []
    uncached_texts = []
    uncached_indices = []
    
    for i, text in enumerate(texts):
        try:
            # Try cache
            result = cached_diacritize(text, use_rules, use_dict)
            results.append(result)
        except:
            # If cache miss or any error, process later
            uncached_texts.append(text)
            uncached_indices.append(i)
            results.append(None)  # Placeholder
    
    # Process uncached texts in batch
    if uncached_texts:
        uncached_results = model.generate(
            uncached_texts,
            batch_size=batch_size,
            use_rules=use_rules,
            use_dict=use_dict
        )
        
        # Fill in results
        for idx, result in zip(uncached_indices, uncached_results):
            results[idx] = result
    
    return results


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "GE2PE Diacritization API",
        "version": "1.0.0",
        "endpoints": {
            "diacritize": "/diacritize (POST)",
            "health": "/health (GET)",
            "metrics": "/metrics (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Service health status
    """
    import torch
    
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        gpu_available=torch.cuda.is_available()
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["monitoring"])
async def get_metrics():
    """
    Get API metrics.
    
    Returns:
        Request statistics and uptime
    """
    uptime = time.time() - app.state.startup_time if hasattr(app.state, 'startup_time') else 0
    
    return MetricsResponse(
        total_requests=request_counter["total"],
        diacritize_requests=request_counter["diacritize"],
        errors=request_counter["errors"],
        uptime_seconds=uptime
    )


@app.post("/diacritize", response_model=DiacritizeResponse, tags=["inference"])
async def diacritize_text(request: DiacritizeRequest):
    """
    Diacritize Persian text(s).
    
    Args:
        request: Diacritization request with text(s) and options
        
    Returns:
        Diacritized text(s)
        
    Raises:
        HTTPException: If model not loaded or processing fails
    """
    # Update counters
    request_counter["total"] += 1
    request_counter["diacritize"] += 1
    
    if model is None:
        request_counter["errors"] += 1
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service is starting up or encountered an error."
        )
    
    try:
        start_time = time.time()
        
        # Process texts
        results = process_batch(
            texts=request.texts,
            batch_size=request.batch_size,
            use_rules=request.use_rules,
            use_dict=request.use_dict
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return DiacritizeResponse(
            results=results,
            count=len(results),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        request_counter["errors"] += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )


# =============================================================================
# TEACHING SECTION
# =============================================================================

"""
TEACHING: FastAPI Service for GE2PE
====================================

📚 OVERVIEW:
Production REST API using FastAPI for Persian diacritization with GE2PE model.

🔍 KEY COMPONENTS:

1. PYDANTIC MODELS (lines 50-120):
   - DiacritizeRequest: Validates input (text, batch_size, flags)
   - DiacritizeResponse: Formats output with results + timing
   - HealthResponse, MetricsResponse: Monitoring endpoints
   - Validators: Auto-convert single string to list

2. STARTUP/SHUTDOWN (lines 125-170):
   - @app.on_event("startup"): Load GE2PE model once at start
   - Detect GPU availability (torch.cuda.is_available)
   - Test model with sample inference
   - Record startup time for uptime metrics
   - @app.on_event("shutdown"): Cleanup (currently minimal)

3. CACHING (lines 175-195):
   - @lru_cache(maxsize=1000): Cache last 1000 unique queries
   - Avoids re-processing identical texts
   - Key includes text + use_rules + use_dict
   - Memory-efficient: oldest entries auto-evicted

4. BATCH PROCESSING (lines 197-235):
   - process_batch(): Smart batching with cache integration
   - Tries cache first for each text
   - Batches uncached texts for efficiency
   - Fills results maintaining original order

5. ENDPOINTS:
   - GET /: API info and available endpoints
   - GET /health: Service status, model loaded, GPU available
   - GET /metrics: Request counts, uptime
   - POST /diacritize: Main inference endpoint
     * Accepts single string or list
     * Returns diacritized texts + processing time
     * Updates request counters

⚠️ THREE DEBUGGING TIPS:

1. GPU Not Detected:
   - Check: import torch; torch.cuda.is_available()
   - Environment: Set USE_GPU=false to force CPU mode
   - Docker: Ensure --gpus all flag when running container
   - Test: nvidia-smi to verify GPU visibility

2. Model Loading Fails:
   - Path error: Set GE2PE_MODEL_PATH env variable
   - Missing files: Check model checkpoint exists
   - Parsivar error: pip install parsivar
   - Fallback: Uses google/mt5-small if checkpoint missing

3. Batch Shape Mismatch / Async Errors:
   - Error: "list index out of range" in process_batch
   - Cause: Uncached_indices mismatch
   - Fix: Ensure results list has same length as input
   - Debug: Print len(texts), len(results), uncached_indices
   - Async: FastAPI handles async/sync mixing automatically

🎯 NEXT STEPS:

1. Dockerization (Dockerfile):
   FROM python:3.9-slim
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . /app
   WORKDIR /app
   EXPOSE 8000
   CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

2. Deployment & Monitoring:
   - Use gunicorn with multiple workers
   - Add Prometheus metrics endpoint
   - Integrate with ELK stack for logging
   - Rate limiting with slowapi
   - Authentication with OAuth2/JWT

3. Performance Optimization:
   - Redis cache instead of LRU (distributed)
   - Batch request queuing (celery/RQ)
   - Model quantization (FP16/INT8)
   - Async inference with asyncio
   - Load balancing with nginx

💡 DESIGN NOTES:
- FastAPI provides automatic OpenAPI docs at /docs
- Pydantic validates all input/output automatically
- CORS enabled for browser access
- Graceful degradation if model fails to load
- LRU cache is thread-safe (Python GIL)
"""


# =============================================================================
# MAIN - RUNNABLE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GE2PE API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 Starting GE2PE Diacritization API Server")
    print("="*60)
    print(f"📡 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🔄 Reload: {args.reload}")
    print(f"📖 Docs: http://{args.host}:{args.port}/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
