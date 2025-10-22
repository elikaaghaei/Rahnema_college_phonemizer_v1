#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test client for GE2PE API
Run this after starting the server to test endpoints
"""

import requests
import json
from typing import List


API_BASE_URL = "http://localhost:8000"


def test_health():
    """Test /health endpoint."""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200


def test_metrics():
    """Test /metrics endpoint."""
    print("\n" + "="*60)
    print("Testing /metrics endpoint")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/metrics")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200


def test_diacritize_single():
    """Test /diacritize with single text."""
    print("\n" + "="*60)
    print("Testing /diacritize with single text")
    print("="*60)
    
    payload = {
        "texts": "سلام دنیا",
        "batch_size": 10,
        "use_rules": False,
        "use_dict": False
    }
    
    response = requests.post(f"{API_BASE_URL}/diacritize", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Input: {payload['texts']}")
        print(f"Output: {result['results'][0]}")
        print(f"Processing time: {result['processing_time_ms']} ms")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_diacritize_batch():
    """Test /diacritize with batch of texts."""
    print("\n" + "="*60)
    print("Testing /diacritize with batch")
    print("="*60)
    
    texts = [
        "سلام دنیا",
        "این یک تست است",
        "زبان فارسی زیباست"
    ]
    
    payload = {
        "texts": texts,
        "batch_size": 10,
        "use_rules": False
    }
    
    response = requests.post(f"{API_BASE_URL}/diacritize", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Processed {result['count']} texts")
        print(f"Processing time: {result['processing_time_ms']} ms")
        
        for i, (inp, out) in enumerate(zip(texts, result['results']), 1):
            print(f"\n{i}. Input:  {inp}")
            print(f"   Output: {out}")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 GE2PE API Test Suite")
    print("="*60)
    print(f"API URL: {API_BASE_URL}")
    
    tests = [
        ("Health Check", test_health),
        ("Metrics", test_metrics),
        ("Diacritize Single", test_diacritize_single),
        ("Diacritize Batch", test_diacritize_batch),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--url":
        API_BASE_URL = sys.argv[2]
    
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API server!")
        print(f"   Please ensure the server is running at {API_BASE_URL}")
        print("\n   Start the server with:")
        print("   python api/main.py")
        print()
