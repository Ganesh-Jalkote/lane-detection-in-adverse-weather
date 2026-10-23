# Lane Detection in Adverse Weather

A research prototype for lane detection under night-time, foggy, and rainy conditions. The current implemented path uses a PyTorch ResNet-18 row-anchor detector inspired by Ultra-Fast Lane Detection (UFLD), with OpenCV preprocessing and synthetic weather augmentation.

> **Project status:** prototype / work in progress. The CULane-oriented training and inference code is implemented in `src/lane_detection/lane_ufld.py` and `lane_core.py`. The planned BDD100K/FoggyLane adapters and FastAPI serving app are not implemented yet. Metrics in this repository are preliminary and should not be interpreted as validated real-world performance.

## Current implementation

- ResNet-18 backbone and row-anchor lane classification.
- CULane annotation parsing and index creation.
- Geometric augmentation and synthetic fog/rain/night transforms.
- Optional CLAHE, adaptive gamma correction, and dark-channel-prior dehazing.
- Training, checkpoint resume, CULane-style approximate evaluation, image/video inference, ONNX and TorchScript export.
- Folder prediction overlays and per-image CSV output.

## Repository layout

```
src/lane_detection/
  lane_core.py       # image processing, labels, augmentation, metrics, index builder
  lane_ufld.py       # model, train/eval/predict/export CLI
  predict_folder.py  # folder-level inference and CSV output
  check_lists.py     # inspect CULane split-list distribution
data/README.md       # dataset notes and current documented split
docs/results/        # sample prediction artifacts
tests/               # lightweight CPU tests
```

Several planned files under `datasets/`, `preprocessing/`, and `serving/` are placeholders. BDD100K and FoggyLane support, a web API, and a complete end-to-end package interface should be considered future work.

## Setup

Use Python 3.10 or 3.11 for the most conservative compatibility baseline, and create a virtual environment.

### 1. Install PyTorch

Choose **one** command based on your environment; consult the [official PyTorch selector](https://pytorch.org/get-started/locally/) if your CUDA version differs.

CPU:
```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

NVIDIA GPU (example CUDA 12.4 build):
```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Install project dependencies

```bash
python -m pip install -r requirements.txt
```

The requirements use `opencv-contrib-python` as the single OpenCV wheel; do not install `opencv-python` alongside it in the same environment.

## CULane workflow

Download CULane and its list archives from the [official project page](https://xingangpan.github.io/projects/CULane.html), and extract them locally. Dataset images and generated indexes are not committed to Git.

### Build an index

Run from the repository root:

```bash
python src/lane_detection/lane_core.py index --root /path/to/culane --out index.json --night_only
```

On Windows PowerShell, replace the path with the local CULane extraction directory, for example `D:\\datasets\\CULane`. The index builder expects CULane image folders and annotation/list files. Review its printed image root, list directory, and split counts before training.

### Train

```bash
python src/lane_detection/lane_ufld.py train --index index.json --out_dir runs/night --epochs 20 --enhance
```

For a quick smoke test, use `--max_iters 2 --workers 0 --batch 2`. This is only a pipeline check, not a meaningful training run. Training may download ImageNet weights; if that fails, the code falls back to random initialization, which can materially change results.

### Evaluate

```bash
python src/lane_detection/lane_ufld.py eval --index index.json --ckpt runs/night/best.pt --n_eval 100
```

The evaluator reports a **CULane-style approximation**, not the official CULane evaluation executable. Use the official evaluator for benchmark claims where feasible. Synthetic weather evaluation can be enabled with `--synthetic`, but synthetic results must be reported separately from real-weather test results.

### Predict one image or video

```bash
python src/lane_detection/lane_ufld.py predict --ckpt runs/night/best.pt --source path/to/road.jpg --out prediction.jpg
python src/lane_detection/lane_ufld.py predict --ckpt runs/night/best.pt --source path/to/drive.mp4 --out prediction.mp4
```

### Predict a folder

```bash
python src/lane_detection/predict_folder.py --ckpt runs/night/best.pt --input path/to/images --out_dir pred_out --recursive --limit 30
```

This writes overlays and `results.csv`. The CSV's FPS is based on measured per-image inference function time and is not end-to-end application/video throughput. Record the hardware, checkpoint, image set, enhancement setting, and whether CPU or GPU was used when sharing results.

## Evaluation protocol

- Keep test frames out of training, augmentation-parameter selection, and threshold calibration. Select thresholds using training/validation data and freeze them before testing.
- Compare enhancement on/off using the same checkpoint and exactly the same held-out images.
- Report precision, recall, F1, latency, and FPS per condition; separate real weather from synthetic weather.
- Do not describe target metrics as achieved metrics. The sample outputs in `docs/results/` are illustrative and are not, by themselves, a validated benchmark.
- Lane safety is not established by these experiments. This prototype is for research/demo use, not direct deployment in a vehicle.

## Tests

Install dependencies, then run:

```bash
python -m pytest -q
```

## License and data

Check the repository license and the individual dataset terms before use or redistribution. Dataset files, model checkpoints, and generated caches should not be committed unless licensing and repository policy explicitly permit it.
