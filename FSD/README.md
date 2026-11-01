# FSD: Full-stack integration scaffold

This folder is designed to be copied into the root of the `lane-detection-in-adverse-weather` repository. It adds a small browser UI, a FastAPI prediction service, and an optional Docker Compose setup for single-image inference.

## Connect it to the model

The API expects the existing project package under `src/lane_detection/` and imports `get_device`, `load_model`, `infer_frame`, and `draw_lanes` from `lane_detection.lane_ufld`. It expects a trained checkpoint at `models/best.pt` by default. Set `MODEL_PATH` if your checkpoint is elsewhere.

The checkpoint is not included. Place your trained checkpoint at `models/best.pt` in the repository root, or update `MODEL_PATH` in the environment.

## Run with Docker Compose

From the repository root after copying this folder there:

```powershell
docker compose -f FSD/compose.yaml up --build
```

Open <http://localhost:8080>. The Compose backend uses CPU PyTorch by default. GPU deployment needs a CUDA-compatible image and host runtime.

## Run without Docker

From the repository root, create and activate your Python environment, install the project dependencies and PyTorch build appropriate for your machine, then install the FastAPI serving dependencies:

```powershell
python -m pip install -r FSD/backend/requirements.txt
$env:PYTHONPATH = "src"
$env:MODEL_PATH = "models/best.pt"
python -m uvicorn app.main:app --app-dir FSD/backend --host 127.0.0.1 --port 8000
```

Serve `FSD/frontend` with any static web server and set `API_BASE_URL` in `frontend/app.js` to `http://127.0.0.1:8000`. For the Compose setup the UI uses the same-origin `/api` proxy.

## API

- `GET /health`: process health
- `GET /ready`: model readiness
- `POST /predict`: multipart form upload with field name `file`; returns JSON containing lane points and a base64 PNG overlay

The API is a demo scaffold for single-image uploads. It does not implement accounts, persistent image storage, video streaming, or production monitoring.
