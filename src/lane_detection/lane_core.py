"""
lane_core.py  -  torch-free building blocks for the adverse-weather lane detector
(UFLD-style row-anchor classifier on ResNet-18, trained on CULane).

Everything here only needs numpy / opencv / scipy, so it can be tested without a GPU.

CLI (build the train/val/test index from a CULane folder):
    python lane_core.py index --root /kaggle/input/<culane-folder> --out index.json
"""
import argparse
import glob
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

cv2.setNumThreads(0)

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
IMG_W0, IMG_H0 = 1640, 590          # original CULane frame size
W, H = 800, 288                     # network input size
GRID = 200                          # column bins (class index GRID == "no lane")
NUM_LANES = 4                       # slots: [far-left, left, right, far-right]
ROW_ANCHORS = np.array([121, 131, 141, 150, 160, 170, 180, 189, 199, 209,
                        219, 228, 238, 248, 258, 267, 277, 287])   # in 288-px space
NUM_ROWS = len(ROW_ANCHORS)
COL_SAMPLE = np.linspace(0, W - 1, GRID)
COL_W = float(COL_SAMPLE[1] - COL_SAMPLE[0])

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

WEATHER_KINDS = ["fog", "rain", "night"]
NIGHT_BRIGHTNESS = 70.0             # mean gray (0-255) below this => "night" tag for train sampling

# CULane test-split files -> our condition names
SPLIT_NAMES = {"test0_normal": "normal", "test2_hlight": "dazzle", "test8_night": "night",
               "test3_shadow": "shadow"}


# ----------------------------------------------------------------------------
# reading images / annotations
# ----------------------------------------------------------------------------
def lines_path(img_path):
    return os.path.splitext(img_path)[0] + ".lines.txt"


def load_image(path):
    """Fast half-res JPEG decode, resized to the network size (BGR uint8, HxWx3)."""
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def load_lanes(img_path, net=False):
    """Lanes from CULane *.lines.txt -> list of (n,2) float arrays [x,y].
    net=False: original 1640x590 coords; net=True: 800x288 network coords."""
    lanes = []
    p = lines_path(img_path)
    if not os.path.exists(p):
        return lanes
    with open(p) as f:
        for line in f:
            v = line.split()
            if len(v) < 4:
                continue
            a = np.array(v, dtype=np.float64)
            a = a[: len(a) // 2 * 2].reshape(-1, 2)
            if net:
                a = a * np.array([W / IMG_W0, H / IMG_H0])
            lanes.append(a)
    return lanes


def to_input(img_bgr):
    """BGR uint8 HxWx3 -> normalised float32 CHW (RGB)."""
    x = img_bgr[..., ::-1].astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


# ----------------------------------------------------------------------------
# labels (row-anchor classification targets)
# ----------------------------------------------------------------------------
def _clean(lane):
    lane = lane[np.argsort(lane[:, 1], kind="stable")]
    _, idx = np.unique(lane[:, 1], return_index=True)
    return lane[np.sort(idx)]


def _x_at_bottom(lane):
    k = min(len(lane), 4)
    pts = lane[-k:]
    if np.ptp(pts[:, 1]) < 1:
        return float(pts[-1, 0])
    a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
    return float(np.clip(a * (H - 1) + b, -2 * W, 3 * W))


def assign_slots(lanes):
    """lanes (net coords) -> list of NUM_LANES entries (lane array or None).
    slot1 = nearest lane left of car centre, slot0 = next one further left,
    slot2 = nearest right, slot3 = next further right."""
    cleaned = []
    for l in lanes:
        if len(l) >= 2:
            l = _clean(l)
            if len(l) >= 2:
                cleaned.append(l)
    left = sorted([l for l in cleaned if _x_at_bottom(l) < W / 2], key=_x_at_bottom, reverse=True)
    right = sorted([l for l in cleaned if _x_at_bottom(l) >= W / 2], key=_x_at_bottom)
    slots = [None] * NUM_LANES
    if len(left) > 0: slots[1] = left[0]
    if len(left) > 1: slots[0] = left[1]
    if len(right) > 0: slots[2] = right[0]
    if len(right) > 1: slots[3] = right[1]
    return slots


def make_labels(lanes_net):
    """-> int64 array (NUM_ROWS, NUM_LANES); value in [0,GRID-1] = column bin, GRID = no lane."""
    labels = np.full((NUM_ROWS, NUM_LANES), GRID, dtype=np.int64)
    for s, lane in enumerate(assign_slots(lanes_net)):
        if lane is None:
            continue
        ys, xs = lane[:, 1], lane[:, 0]
        for r, ya in enumerate(ROW_ANCHORS):
            if ys[0] <= ya <= ys[-1]:
                x = float(np.interp(ya, ys, xs))
                if 0 <= x <= W - 1:
                    labels[r, s] = min(int(round(x / COL_W)), GRID - 1)
    return labels


# ----------------------------------------------------------------------------
# augmentation: geometry
# ----------------------------------------------------------------------------
def random_affine(img, lanes, rng, rot=6.0, scale=(0.9, 1.1), shift=(0.08, 0.05)):
    angle = rng.uniform(-rot, rot)
    sc = rng.uniform(*scale)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, sc)
    M[0, 2] += rng.uniform(-shift[0], shift[0]) * W
    M[1, 2] += rng.uniform(-shift[1], shift[1]) * H
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    out = [l @ M[:, :2].T + M[:, 2] for l in lanes]
    return img, out


# ----------------------------------------------------------------------------
# augmentation: synthetic adverse weather (applied on 800x288 BGR uint8)
# ----------------------------------------------------------------------------
def add_fog(img, rng):
    """Atmospheric scattering model I = J*t + A*(1-t) with a depth proxy that grows towards the horizon."""
    h, w = img.shape[:2]
    beta = rng.uniform(1.0, 3.5)
    A = rng.uniform(0.75, 0.95)
    y = np.arange(h, dtype=np.float32)[:, None]
    depth = 0.15 + np.clip((h - y) / (h - 100.0), 0, 1)            # (h,1)
    t = np.exp(-beta * depth)[:, :, None]                          # (h,1,1)
    patch = cv2.resize(rng.uniform(0.85, 1.15, (4, 8)).astype(np.float32), (w, h),
                       interpolation=cv2.INTER_CUBIC)[:, :, None]  # patchy density
    t = np.clip(t * patch, 0.02, 1.0)
    f = img.astype(np.float32) / 255.0
    out = f * t + A * (1 - t)
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.2))
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def add_rain(img, rng):
    h, w = img.shape[:2]
    f = img.astype(np.float32) / 255.0 * rng.uniform(0.65, 0.9)
    f = 0.6 * f + 0.4 * f.mean(2, keepdims=True)                   # desaturate
    layer = np.zeros((h, w), np.float32)
    ang = np.deg2rad(rng.uniform(-12, 12))
    for _ in range(int(rng.integers(300, 900))):
        x0, y0, L = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(8, 25))
        cv2.line(layer, (x0, y0), (x0 + int(L * np.sin(ang)), y0 + int(L * np.cos(ang))),
                 float(rng.uniform(0.4, 1.0)), 1)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
    f = f + 0.35 * layer[..., None]
    f = f * 0.9 + 0.1 * rng.uniform(0.5, 0.8)                      # wet-air haze
    f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.3, 0.9))
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def add_night(img, rng):
    f = img.astype(np.float32) / 255.0
    f = f ** rng.uniform(1.8, 3.2)
    f[..., 0] *= 1.10                                              # slightly blue (BGR)
    f[..., 2] *= 0.90
    f = f + rng.normal(0, rng.uniform(0.005, 0.03), f.shape).astype(np.float32)
    s = rng.uniform(0.0, 0.9)
    if s > 0.3:
        f = cv2.GaussianBlur(f, (0, 0), s)
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def apply_weather(img, kind, rng):
    return {"fog": add_fog, "rain": add_rain, "night": add_night}[kind](img, rng)


# ----------------------------------------------------------------------------
# stage-1 enhancement (CLAHE + adaptive gamma + light dehaze)  -- "pre-processing block"
# ----------------------------------------------------------------------------
def dehaze_dcp(img, omega=0.9, t0=0.25, patch=15):
    f = img.astype(np.float32) / 255.0
    k = np.ones((patch, patch), np.uint8)
    dark = cv2.erode(f.min(2), k)
    n = max(1, int(dark.size * 0.001))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    A = np.clip(f.reshape(-1, 3)[idx].mean(0), 0.5, 1.0)
    t = 1 - omega * cv2.erode((f / A).min(2), k)
    t = np.clip(cv2.GaussianBlur(t, (0, 0), patch / 3), t0, 1.0)
    J = (f - A) / t[..., None] + A
    return np.clip(J * 255, 0, 255).astype(np.uint8)


def enhance_image(img, haze_thresh=0.45):
    """Adaptive: brighten if dark, CLAHE always, dehaze if the dark-channel says it is hazy."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    m = L.mean() / 255.0
    if m < 0.35:
        gamma = float(np.clip(np.log(0.45) / np.log(max(m, 0.02)), 0.35, 1.0))
        L = cv2.LUT(L, (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8))
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    out = cv2.cvtColor(cv2.merge([L, lab[..., 1], lab[..., 2]]), cv2.COLOR_LAB2BGR)
    dark_mean = cv2.erode(out.min(2), np.ones((15, 15), np.uint8)).mean() / 255.0
    if dark_mean > haze_thresh:
        out = dehaze_dcp(out)
    return out


# ----------------------------------------------------------------------------
# samples
# ----------------------------------------------------------------------------
def train_sample(path, rng, weather_p=0.5, enhance=False):
    img = load_image(path)
    lanes = load_lanes(path, net=True)
    img, lanes = random_affine(img, lanes, rng)
    if rng.random() < weather_p:
        img = apply_weather(img, WEATHER_KINDS[int(rng.integers(len(WEATHER_KINDS)))], rng)
    if enhance:
        img = enhance_image(img)
    return to_input(img), make_labels(lanes)


def eval_image(path, kind=None, seed=0, enhance=False):
    img = load_image(path)
    if kind:
        img = apply_weather(img, kind, np.random.default_rng(seed))
    if enhance:
        img = enhance_image(img)
    return to_input(img)


# ----------------------------------------------------------------------------
# decoding + metric
# ----------------------------------------------------------------------------
def decode(logits):
    """logits: np (GRID+1, NUM_ROWS, NUM_LANES) -> list of (n,2) lanes in ORIGINAL 1640x590 coords."""
    z = logits[:GRID] - logits[:GRID].max(0, keepdims=True)
    p = np.exp(z)
    p /= p.sum(0, keepdims=True)
    loc = (p * np.arange(GRID)[:, None, None]).sum(0)               # (R, L)
    valid = logits.argmax(0) != GRID
    lanes = []
    for l in range(NUM_LANES):
        rows = valid[:, l]
        if rows.sum() > 2:
            xs = loc[rows, l] * COL_W * IMG_W0 / W
            ys = ROW_ANCHORS[rows] * IMG_H0 / H
            lanes.append(np.stack([xs, ys], 1))
    return lanes


def _draw(lane, width=15, scale=0.5):
    canvas = np.zeros((int(IMG_H0 * scale), int(IMG_W0 * scale)), np.uint8)
    pts = np.round(lane * scale).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, 1, thickness=width)
    return canvas.astype(bool)


def f1_counts(preds, gts, iou_thr=0.5):
    """CULane-style: lanes drawn 30 px wide (15 px at half-res), Hungarian match, IoU>0.5 => TP."""
    from scipy.optimize import linear_sum_assignment
    if not preds and not gts:
        return 0, 0, 0
    if not preds:
        return 0, 0, len(gts)
    if not gts:
        return 0, len(preds), 0
    P, G = [_draw(p) for p in preds], [_draw(g) for g in gts]
    iou = np.zeros((len(P), len(G)))
    for i, a in enumerate(P):
        for j, b in enumerate(G):
            u = np.logical_or(a, b).sum()
            iou[i, j] = np.logical_and(a, b).sum() / u if u else 0.0
    r, c = linear_sum_assignment(-iou)
    tp = int((iou[r, c] > iou_thr).sum())
    return tp, len(P) - tp, len(G) - tp


def f1_score(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


# ----------------------------------------------------------------------------
# dataset indexing
# ----------------------------------------------------------------------------
def _bfs_find(root, pred, max_depth=3):
    frontier = [(root, 0)]
    while frontier:
        d, depth = frontier.pop(0)
        if pred(d):
            return d
        if depth < max_depth:
            try:
                for e in sorted(os.scandir(d), key=lambda e: e.name):
                    if e.is_dir(follow_symlinks=True):
                        frontier.append((e.path, depth + 1))
            except (PermissionError, FileNotFoundError):
                pass
    return None


def _brightness(p):
    im = cv2.imread(p, cv2.IMREAD_REDUCED_GRAYSCALE_4)
    return float(im.mean()) if im is not None else -1.0


def _read_list(path):
    out = []
    with open(path) as f:
        for line in f:
            t = line.split()
            if t:
                out.append(t[0])
    return out


def _take_existing(rel_paths, image_root, n, rng):
    rel_paths = list(rel_paths)
    rng.shuffle(rel_paths)
    out = []
    for r in rel_paths:
        p = os.path.join(image_root, r.lstrip("/"))
        if os.path.exists(p) and os.path.exists(lines_path(p)):
            out.append(p)
            if len(out) >= n:
                break
    return out


def _tag(paths, cache):
    todo = [p for p in paths if p not in cache]
    if todo:
        with ThreadPoolExecutor(8) as ex:
            for p, b in zip(todo, ex.map(_brightness, todo)):
                cache[p] = b
    return [(p, cache[p]) for p in paths]


def build_index(root, out, n_train=30000, n_val=2000, n_test=1500, seed=0, night_only=False, night_thresh=0.0):
    """night_only=True: train/val contain ONLY real frames that look like night (mean gray < threshold).
    The threshold is auto-calibrated on the official test8_night frames unless night_thresh > 0."""
    rng = random.Random(seed)
    BIG = 10 ** 9
    image_root = _bfs_find(root, lambda d: any(n.startswith("driver_") for n in os.listdir(d)))
    if image_root is None:
        raise SystemExit(f"No driver_* folders found under {root}. Check the path / extraction.")
    list_dir = _bfs_find(root, lambda d: os.path.isfile(os.path.join(d, "train_gt.txt")))
    print(f"image root: {image_root}\nlist dir  : {list_dir}")

    index = {"image_root": image_root, "test": {}, "night_only": night_only}
    cache = {}
    pool_val = []
    if list_dir:
        pool_train = _take_existing(_read_list(os.path.join(list_dir, "train_gt.txt")), image_root, BIG, rng)
        vf = os.path.join(list_dir, "val_gt.txt")
        if os.path.isfile(vf):
            pool_val = _take_existing(_read_list(vf), image_root, BIG, rng)
        split_dir = os.path.join(list_dir, "test_split")
        for fn, cond in SPLIT_NAMES.items():
            sf = os.path.join(split_dir, fn + ".txt")
            if os.path.isfile(sf):
                index["test"][cond] = _take_existing(_read_list(sf), image_root, n_test, rng)
    else:
        print("WARNING: official list files not found -> falling back to a clip-level random split.")
        allp = sorted(glob.glob(os.path.join(image_root, "driver_*", "*", "*.jpg")))
        allp = [p for p in allp if os.path.exists(lines_path(p))]
        clips = sorted({os.path.dirname(p) for p in allp})
        rng.shuffle(clips)
        nt, nv = max(1, int(0.15 * len(clips))), max(1, int(0.05 * len(clips)))
        test_c, val_c, train_c = set(clips[:nt]), set(clips[nt:nt + nv]), set(clips[nt + nv:])
        by = lambda cs: [p for p in allp if os.path.dirname(p) in cs]
        pool_train, pool_val, te = by(train_c), by(val_c), by(test_c)
        for lst in (pool_train, pool_val, te):
            rng.shuffle(lst)
        thr0 = night_thresh or NIGHT_BRIGHTNESS
        te = _tag(te[: n_test * 4], cache)
        index["test"]["normal"] = [p for p, b in te if b >= thr0][:n_test]
        index["test"]["night"] = [p for p, b in te if 0 <= b < thr0][:n_test]

    # --- night threshold ---------------------------------------------------------------
    # Never tune a preprocessing/sampling threshold from held-out test frames. Use the
    # explicit user-supplied value, or the documented fixed default; calibrate externally
    # on train/validation data only if a dataset-specific threshold is needed.
    thr = float(night_thresh) if night_thresh > 0 else float(NIGHT_BRIGHTNESS)
    index["night_thresh"] = thr
    print(f"night threshold (mean gray < thr): {thr:.0f}")
    if night_thresh <= 0:
        print("using fixed default threshold; do not tune this value using test-set images")

    # --- train / val ------------------------------------------------------------------------------
    if night_only:
        pool_train = [p for p, b in _tag(pool_train, cache) if 0 <= b < thr]
        pool_val = [p for p, b in _tag(pool_val, cache) if 0 <= b < thr]
    if pool_val:
        val_paths, train_paths = pool_val, pool_train
    else:
        print("official val images not present -> using held-out train clips (~5%) as validation")
        clips = sorted({os.path.dirname(p) for p in pool_train})
        rng.shuffle(clips)
        val_clips = set(clips[: max(1, int(0.05 * len(clips)))])
        val_paths = [p for p in pool_train if os.path.dirname(p) in val_clips]
        train_paths = [p for p in pool_train if os.path.dirname(p) not in val_clips]
    train_paths, val_paths = train_paths[:n_train], val_paths[:n_val]
    index["train"] = [[p, b] for p, b in _tag(train_paths, cache)]
    index["val"] = val_paths

    ntr = len(index["train"])
    night = sum(1 for _, b in index["train"] if 0 <= b < thr)
    kind = "night-only" if night_only else f"night-tagged {night}, {100 * night / max(1, ntr):.1f}%"
    print(f"train {ntr} ({kind}), val {len(index['val'])}, test " + ", ".join(f"{k}={len(v)}" for k, v in index["test"].items()))
    if not index["train"]:
        raise SystemExit("0 training images found - did you extract the driver_* archives AND the annotations (*.lines.txt)?")
    with open(out, "w") as f:
        json.dump(index, f)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index")
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="index.json")
    p.add_argument("--n_train", type=int, default=30000)
    p.add_argument("--n_val", type=int, default=2000)
    p.add_argument("--n_test", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--night_only", action="store_true", help="keep only real night-looking frames in train/val")
    p.add_argument("--night_thresh", type=float, default=0.0, help="mean-gray threshold for 'night' (0 = auto-calibrate)")
    a = ap.parse_args()
    build_index(a.root, a.out, a.n_train, a.n_val, a.n_test, a.seed, a.night_only, a.night_thresh)
