# Dataset setup and evaluation notes

Raw dataset images and generated indexes are not committed to Git. Download datasets from their official sources and review their licenses and usage terms before use or redistribution.

## Current implementation status

**Implemented:** CULane-oriented indexing and annotation parsing in `src/lane_detection/lane_core.py`; row-anchor targets for the current ResNet-18 model; optional synthetic fog, rain and night augmentation.

**Not implemented yet:** dedicated BDD100K and FoggyLane adapters. The presence of a dataset in this document does not mean it has been integrated or used in a completed experiment. BDD100K and FoggyLane results must not be claimed until their loaders, annotation conversion, split logic and evaluations are implemented and run.

## Dataset sources

- **CULane:** https://xingangpan.github.io/projects/CULane.html
- **BDD100K:** https://bdd-data.berkeley.edu/
- **FoggyLane:** verify the dataset/paper source, license and annotation format before use.

## CULane annotations

CULane uses one `*.lines.txt` file per image. Each line describes a lane as a sequence of pixel-coordinate `x y` pairs. Split lists such as `train_gt.txt`, `val_gt.txt` and files under `test_split/` are used by the current index builder.

The model maps these polylines to row-anchor targets: for each predefined image row and each of four lane slots, it predicts a column bin or the “no lane” class.

## Build an index

From the repository root:

```bash
python src/lane_detection/lane_core.py index --root /path/to/culane --out index.json --night_only
```

On Windows PowerShell, for example:

```powershell
python src/lane_detection/lane_core.py index --root "D:\datasets\CULane" --out index.json --night_only
```

Review the printed image root, list directory, and train/validation/test counts. The generated index contains local paths and is not portable to another machine without rebuilding it.

## Weather augmentation and enhancement

The current code supports synthetic fog, rain and night augmentation, plus optional CLAHE, adaptive gamma correction and dark-channel-prior dehazing. Synthetic weather is not a substitute for evaluation on real adverse-weather footage. Keep real and synthetic results in separate tables.

The current enhancement heuristic is a general preprocessing heuristic, not a verified weather classifier. Do not describe it as reliable rain detection or glare/reflection suppression without implementing and testing those functions.

## Evaluation protocol

- Keep the final test split out of training and all parameter/threshold selection.
- Choose brightness thresholds using training/validation data, freeze them, and only then evaluate the test set.
- Compare enhancement on/off with the same checkpoint and the same held-out images.
- Report lane F1, precision, recall, latency and FPS separately for each condition, including sample count and hardware.
- The current metric code is a CULane-style approximation. Use the official CULane evaluation tool for benchmark claims where feasible.
- Values in README tables are targets, not achieved results. Treat saved sample outputs as preliminary unless a reproducible evaluation run verifies them.

## Data hygiene

Do not commit raw images, personal footage, generated caches or model checkpoints without checking size and license constraints. Record dataset version, split-generation method, random seed and annotation-conversion version for each experiment.
