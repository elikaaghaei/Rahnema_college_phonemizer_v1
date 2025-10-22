#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mock FastAPI for testing structure without dependencies
"""

from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="GE2PE Mock API")

class DiacritizeRequest(BaseModel):
    texts: Union[str, List[str]]
    batch_size: int = 10
    use_rules: bool = False

class DiacritizeResponse(BaseModel):
    results: List[str]
    count: int
    processing_time_ms: float = 0.0

@app.get("/")
async def root():
    return {"message": "GE2PE Mock API", "status": "mock mode"}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": False, "gpu_available": False}

@app.post("/diacritize", response_model=DiacritizeResponse)
async def diacritize(req: DiacritizeRequest):
    # Mock: just add some diacritics randomly
    texts = req.texts if isinstance(req.texts, list) else [req.texts]
    results = [f"{t} [MOCK-DIACRITIZED]" for t in texts]
    return DiacritizeResponse(results=results, count=len(results))

if __name__ == "__main__":
    import uvicorn
    print("🎭 Mock API (no real model)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
