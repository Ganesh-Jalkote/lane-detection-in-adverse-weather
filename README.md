# Lane Detection in Adverse Weather

Robust, real-time lane detection for rain, fog, and night-time driving conditions on Indian roads — a B.Tech Semester V minor project (MIT Vishwaprayag University, School of Computing).

## Problem

Camera-based lane detection is a core function of ADAS (Advanced Driver-Assistance Systems), but most published models and commercial systems are trained and benchmarked almost entirely on clear, daylight conditions. On the CULane benchmark, published detectors show F1-scores falling from ~90-93% in normal daylight to just 66-74% at night. Maharashtra's own highways — the Mumbai-Pune Expressway, the Mumbai-Nagpur Samruddhi Mahamarg — carry heavy night traffic through fog-prone Western Ghat terrain, and MoRTH data shows night-time accidents are markedly more severe than daytime ones.

This project targets that gap directly: a lane detection system trained and evaluated specifically on rain, fog, and night conditions, rather than reporting a single aggregate score that hides how badly a model degrades in exactly the conditions where it matters most.

## Approach

A two-stage pipeline:

1. **Image enhancement pre-processing** — CLAHE or Retinex-based low-light enhancement (night), a lightweight dehazing step (fog), and glare/reflection suppression (rain).
2. **Lightweight lane-detection model** — a ResNet-18-based row-anchor detector in the style of Ultra-Fast Lane Detection (UFLD), chosen for real-time speed on modest hardware.

The model is trained on CULane (night/dazzle-light subset) and BDD100K (rain/fog-tagged subset), augmented with locally captured and synthetically generated adverse-condition footage.

## Target metrics

| Metric | Target |
|---|---|
| Overall F1-score | ≥ 80% |
| Per-condition F1-score (rain/fog/night) | ≥ 75% |
| Inference speed | ≥ 15 FPS |
| F1 improvement over unmodified baseline (adverse conditions) | 15-20 percentage points |

## Repository structure

```
lane-detection-in-adverse-weather/
├── configs/              # Dataset paths, hyperparameters, model config
├── data/                 # Dataset download instructions (raw/processed data is gitignored)
├── docs/                 # Project synopsis, reports, reference notes
├── serving/              # Web interface for serving the trained model
├── src/
│   └── lane_detection/   # Core package: datasets, preprocessing, model, training, evaluation
├── requirements.txt
├── LICENSE
└── README.md
```

`notebooks/`, `models/`, `scripts/`, and `tests/` will be added as the project progresses — they're not yet tracked since they're currently empty.

## Getting started

```bash
git clone https://github.com/Ganesh-Jalkote/lane-detection-in-adverse-weather.git
cd lane-detection-in-adverse-weather

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Datasets

Datasets are not committed to this repo (see `.gitignore`). Download instructions and expected folder layout are in [`data/README.md`](data/README.md):

- [CULane](https://xingangpan.github.io/projects/CULane.html) — night/dazzle-light subset
- [BDD100K](https://bdd-data.berkeley.edu/) — rain/fog-tagged subset

## Usage

_To be added as the pipeline is implemented — training, evaluation, and serving commands will go here._

## Roadmap

| Phase | Focus | Deadline |
|---|---|---|
| 1 | Problem Statement & Project Synopsis | 14 Sep 2026 |
| 2 | Model Development & Integration | 2 Oct 2026 |
| 3 | Full-Stack Integration | 23 Oct 2026 |
| 4 | Deployment & Reproducibility | 6 Nov 2026 |
| 5 | Project Documentation | 15 Nov 2026 |

## Tech stack

Python · PyTorch · OpenCV · ONNX / TorchScript · FastAPI 

## References

Key sources are listed in the [project synopsis](docs/) — including MoRTH road accident data, CULane/BDD100K dataset papers, and UFLD/SCNN/CLRNet benchmark results.

## License

See [LICENSE](LICENSE).
