# Docker setup: Lane Detection in Adverse Weather

This is a CPU-first reproducible Python environment for this repository. It does not pretend to provide a web server: `serving/app.py` is currently empty, so a browser UI will not start until the FastAPI application is implemented.

## Files

- `Dockerfile`: Python 3.11, CPU PyTorch/torchvision, OpenCV shared libraries, and project dependencies.
- `.dockerignore`: prevents datasets, checkpoints, and generated output from being copied into the image.
- `docker-compose.yml`: build/run shortcuts and host-mounted paths for data, runs, predictions, and models.

## 1. Put these files in the repository root

Copy `Dockerfile`, `.dockerignore`, and `docker-compose.yml` beside `requirements.txt` and `README.md`.

## 2. Build and test the image

From the repository root in PowerShell or the VS Code terminal:

```powershell
docker compose build
docker compose run --rm lane-detection
```

The check should print PyTorch/OpenCV versions and `CUDA available: False`. That is expected for this CPU image.

## 3. Run a prediction using a trained checkpoint

Create/ensure a host folder `runs/main` and place your trained checkpoint at `runs/main/best.pt`. Put an input image at the repository root named `sample.jpg`, then run:

```powershell
docker compose run --rm lane-detection python /app/src/lane_detection/lane_ufld.py predict --ckpt /app/runs/main/best.pt --source /app/sample.jpg --out /app/pred_out/sample_pred.png
```

The output appears in your local `pred_out` directory. The checkpoint must be from this model implementation.

## 4. Predict over a folder of images

Put images in `data/my_images/` and run:

```powershell
docker compose run --rm lane-detection python /app/src/lane_detection/predict_folder.py --ckpt /app/runs/main/best.pt --input /app/data/my_images --out_dir /app/pred_out --recursive
```

This saves image overlays, a contact sheet, and a `results.csv` under `pred_out/`.

## 5. Dataset index and small training smoke test

CULane data is not included in the repository. After downloading/extracting the dataset under `data/culane/` in the expected structure, create a small index:

```powershell
docker compose run --rm lane-detection python /app/src/lane_detection/lane_core.py index --root /app/data/culane --out /app/data/index.json --n_train 1000 --n_val 200 --n_test 200
```

Then run a minimal smoke test:

```powershell
docker compose run --rm lane-detection python /app/src/lane_detection/lane_ufld.py train --index /app/data/index.json --out_dir /app/runs/smoke --epochs 1 --batch 2 --workers 0 --max_iters 2
```

This only checks the training path; it is not meaningful model training. A CPU image is fine for import checks and tiny smoke tests, but not efficient for full training.

## 6. Practical commands

Rebuild after changing dependencies:

```powershell
docker compose build --no-cache
```

Run an interactive shell inside the environment:

```powershell
docker compose run --rm lane-detection bash
```

Remove the image when no longer needed:

```powershell
docker image rm lane-detection-adverse-weather:cpu
```

## Limitations and notes

- The image is CPU-only and will not use the host NVIDIA GPU. GPU support needs Docker's NVIDIA runtime/toolkit and a CUDA-compatible PyTorch image.
- Your current `requirements.txt` lists both `opencv-python` and `opencv-contrib-python`. They overlap; to avoid wheel conflicts, ideally keep only one (usually `opencv-contrib-python` if its extra modules are needed), then rebuild.
- `serving/app.py` is empty right now. Do not expect `uvicorn serving.app:app` to work until you add the FastAPI app object and routes.
- Dataset files and checkpoints are excluded from the build context on purpose. Store them on the host and use the mounted folders.
