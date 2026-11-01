from __future__ import annotations

import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# This file lives at <project-root>/FSD/backend/app/main.py.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

try:
    from lane_detection.lane_ufld import draw_lanes, get_device, infer_frame, load_model
except ImportError as exc:
    raise RuntimeError(
        "Could not import the project inference code. Ensure FSD is inside the repository root "
        "and src/lane_detection/lane_ufld.py is present."
    ) from exc


MODEL_PATH = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "best.pt")))
MODEL_ENHANCE = os.getenv("MODEL_ENHANCE", "auto").strip().lower()
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024)
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
STATE: dict[str, Any] = {"model": None, "device": None, "enhance": False}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model checkpoint not found at {MODEL_PATH}. Set MODEL_PATH or place the trained checkpoint there."
        )
    device = get_device()
    model, checkpoint_metadata = load_model(str(MODEL_PATH), device)
    if MODEL_ENHANCE == "on":
        enhance = True
    elif MODEL_ENHANCE == "off":
        enhance = False
    else:
        enhance = bool(checkpoint_metadata.get("enhance", False))
    STATE.update(model=model, device=device, enhance=enhance)
    yield
    STATE.update(model=None, device=None)


app = FastAPI(title="Lane Detection API", version="0.1.0", lifespan=lifespan)
origins = [value.strip() for value in os.getenv("CORS_ORIGINS", "http://localhost:8080").split(",") if value.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ready"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")

    encoded = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="The uploaded file could not be decoded as an image")

    try:
        lanes = infer_frame(STATE["model"], frame, STATE["enhance"], STATE["device"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Lane inference failed") from exc

    overlay = draw_lanes(frame.copy(), lanes)
    success, png = cv2.imencode(".png", overlay)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode the annotated image")

    points = [
        [[float(x), float(y)] for x, y in np.asarray(lane).reshape(-1, 2)]
        for lane in lanes
    ]
    return {
        "filename": file.filename,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "lanes": points,
        "annotated_image_png_base64": base64.b64encode(png.tobytes()).decode("ascii"),
    }
