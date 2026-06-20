# velocity-ctr-service

## Overview

`velocity-ctr-service` is a high-throughput, low-latency click-through rate (CTR) prediction engine built for mobile advertising. It combines synthetic ad-tech data generation, a lightweight Scikit-Learn inference pipeline, and a FastAPI-powered prediction endpoint to showcase an engineering-ready architecture for real-time systems.

This project is designed to clearly demonstrate:

- sub-10ms inference potential for request handling
- clean separation between feature lookup, preprocessing, and model inference
- robust deployment-ready API structure

## Key Architectural Highlights

- **FastAPI inference endpoint**: Minimal request overhead with startup-loaded model state and a streamlined prediction flow.
- **Mock low-latency feature store**: A fast in-memory dictionary simulates user profiling lookups for `device_os` and `app_category`.
- **Decoupled Scikit-Learn pipeline**: A `ColumnTransformer` plus one-hot encoding and logistic regression pipeline is serialized to `ctr_model.pkl` for runtime reuse.
- **Startup model initialization**: The model loads once at startup using FastAPI lifecycle events so each request only runs inference.
- **Synthetic data training**: `train.py` generates a realistic ad-tech dataset and trains on a mix of device, ad placement, app category, and hourly context.

## Repository Structure

- `train.py` — synthetic data generation, model training, and `ctr_model.pkl` export
- `app.py` — FastAPI server, startup model loading, in-memory feature retrieval, prediction endpoint
- `benchmark.py` — sequential latency benchmark with client-side timing and professional console reporting
- `README.md` — project summary, performance metrics, and quick-start guide

## Tech Stack

- Python 3.11+ (recommended)
- FastAPI for high-performance HTTP serving
- Scikit-Learn for inference pipeline and model training
- Pandas / NumPy for feature handling and benchmarking
- Joblib for model serialization
- Uvicorn as the ASGI production server

## Local Benchmark Performance

The service was benchmarked using `benchmark.py` against the running FastAPI endpoint.

| Metric                        | Value    |
|------------------------------|----------|
| Total Requests Processed     | 200      |
| Average Latency              | 3.18 ms  |
| Median Latency (p50)         | 3.00 ms  |
| p99 Latency                  | 6.00 ms  |

## Quick Start

Follow these steps to run the service locally.

### 1. Activate the virtual environment

```powershell
cd c:\Users\Karthik Reddy\velocity-ctr-service
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

If `requirements.txt` exists:

```powershell
pip install -r requirements.txt
```

If not, install the core dependencies directly:

```powershell
pip install fastapi uvicorn scikit-learn pandas numpy joblib requests
```

### 3. Train the model

```powershell
python train.py --rows 50000 --out ctr_model.pkl
```

### 4. Run the FastAPI server

```powershell
uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1
```

> If port `8000` is unavailable, use `8001` or another open port.

### 5. Run the benchmark

```powershell
python benchmark.py
```

## Example Request

```powershell
curl -X POST "http://127.0.0.1:8001/predict_ctr" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user_101","ad_id":"ad_999","ad_position":"banner"}'
```

## Why This Project Stands Out

This repository is not only a working prediction service, but also a strong example of practical machine learning engineering:

- clean separation between model training and inference
- efficient runtime design for real-time scoring
- clear benchmarking methodology for latency validation
- ready-to-present structure for recruiters and hiring managers

## Next Steps

Potential extensions for a production-ready release include:

- replacing the mock feature store with a real-time feature store (Redis, Cassandra, etc.)
- adding async request batching or worker autoscaling
- deploying in Docker or Kubernetes for horizontal scaling
- adding unit tests and CI/CD pipelines
