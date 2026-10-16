# CPU-first development image for the lane detection project.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Libraries required by OpenCV wheels on Debian slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

# requirements.txt intentionally leaves PyTorch to be installed per machine.
RUN python -m pip install --upgrade pip && \
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install -r requirements.txt

COPY . .

# Current command-line modules import lane_core/lane_ufld as sibling modules.
ENV PYTHONPATH="/app/src/lane_detection:/app/src"

RUN mkdir -p /app/data /app/runs /app/pred_out /app/models

# Checks the installed runtime. The serving app is not implemented yet.
CMD ["python", "-c", "import cv2, torch; print('Docker environment ready'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('OpenCV:', cv2.__version__)"]
