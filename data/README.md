# `data/` — Datasets for Lane Detection in Adverse Weather

This folder holds everything dataset-related for the project: where the data comes from, how it is laid out on disk, how annotations are converted into one common format, and how to reproduce the splits we train on.

> **The raw images are NOT committed to Git.** CULane alone is ~30 GB. `data/` is covered by `.gitignore`; only small, reproducible files (index/list files, scripts, this README) are tracked.

---

## 1. Datasets used

We train and evaluate on **real-world images only** (no synthetic data).

| Dataset | Used for | Conditions | Notes |
|---|---|---|---|
| **CULane** | Primary training and evaluation | Night, dazzle-light, shadow, normal | ~133K frames in total. We use the **night subset** (and its dazzle/shadow test categories). |
| **BDD100K** | Rain / fog / night diversity | Rainy, foggy, night (via weather and time-of-day tags) | Filtered using the `weather` and `timeofday` attributes in its label JSON. |
| **FoggyLane** (optional) | Extra real foggy-driving images | Fog | Candidate source for the fog stage ([arXiv 2504.06121](https://arxiv.org/abs/2504.06121)). |

Training follows a staged strategy: **clear → night → fog/rain**. Night is trained first, and fog/rain data is added in later runs.

### Where to get them

- **CULane** — <https://xingangpan.github.io/projects/CULane.html> (also mirrored on Kaggle)
- **BDD100K** — <https://doc.bdd100k.com> / <https://bdd-data.berkeley.edu> (also on Kaggle)
- **FoggyLane** — see the paper linked above

Please check each dataset's license and terms of use before use or redistribution. These datasets are intended for research and non-commercial use.

---

## 2. Splits we actually train on

| Split | Source | Frames |
|---|---|---|
| Night train | CULane (night only) | **9,593** |
| Night val | CULane (night only) | **527** |
| Test | CULane real test frames, per category | normal, dazzle-light, night, shadow |

Dazzle-light and shadow are reported separately because they are the hardest cases in CULane.

---

## 3. Annotation formats

The two main datasets use different schemas. Reconciling them is handled in the data-loading code under `src/lane_detection/`.

- **CULane** — one `*.lines.txt` per image. Each line is one lane, given as a flat list of `x y` points (pixel coordinates). Split lists are plain text files under `list/`.
- **BDD100K** — JSON labels. Lane markings are polylines (with lane-style and direction attributes), and each image carries `weather` (e.g. rainy, foggy, clear) and `timeofday` (e.g. night) tags that we use for filtering.

Both are converted to **a single row-anchor representation** (for the Ultra-Fast-Lane-Detection-style, ResNet-18-based model): for a fixed set of image rows, the x-position of each lane, or "no lane" if it is absent.

---

## 4. Preprocessing

Condition-specific enhancement is applied before the detector (toggle with `--enhance` in training and eval):

- **Night / low light:** CLAHE and gamma correction
- **Fog:** lightweight dehazing (e.g. dark channel prior)
- **Rain:** glare/reflection suppression

Training augmentations are limited to geometric and photometric changes applied to the real images.

---

## 5. Reproducing the data setup

1. Download the datasets you need (keep to the minimum — start with CULane night).
2. Place them under `data/` (or edit the dataset paths in `configs/`).
3. Build the night index:
   ```bash
   python -m lane_detection.data.build_index --dataset culane --condition night
   ```
   *(Replace with the actual script/module name in `src/lane_detection/` if it differs.)*
4. Run a quick sanity check by drawing annotations onto a few frames before training.

---

## 6. Evaluation metrics

- **F1 at an IoU threshold** (CULane-style lane matching), reported per condition
- **Mean IoU**

Targets: overall F1 ≥ 80%, per-condition F1 ≥ 75%, and ≥ 15 FPS at inference.

---

## 7. Notes

- Never commit raw datasets, checkpoints, or generated caches. Share trained weights through GitHub Releases.
- If you add a new dataset (e.g. extra fog or rain data), document its source, license, and annotation format here and add the conversion step to the loader.
