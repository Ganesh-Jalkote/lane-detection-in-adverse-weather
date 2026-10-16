"""
predict_folder.py - run the trained lane model on EVERY image in a folder and save the overlays.

    python predict_folder.py --ckpt "..\\runs\\night_enh\\last.pt" --input "M:\\some\\folder" --out_dir "..\\pred_out"
    python predict_folder.py --ckpt "..\\runs\\night_enh\\last.pt" --index "..\\index.json" --cond night --n 30

Works with any dashcam images (any size). If an image is a CULane frame (1640x590) that has its
*.lines.txt next to it, the ground truth is drawn as white lines and TP/FP/FN are printed.
Needs lane_ufld.py and lane_core.py in the same folder; does not modify them.
"""
import argparse
import csv
import json
import os
import random
import time

import cv2
import numpy as np

import lane_core as lc
import lane_ufld as lu

EXT = (".jpg", ".jpeg", ".png", ".bmp")
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]      # BGR: red, green, blue, yellow


def find_images(path, recursive, skip_dir):
    if os.path.isfile(path):
        return [path]
    out = []
    if recursive:
        for dp, _, fns in os.walk(path):
            out += [os.path.join(dp, f) for f in fns if f.lower().endswith(EXT)]
    else:
        out = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(EXT)]
    skip = os.path.abspath(skip_dir)
    return sorted(p for p in out if not os.path.abspath(p).startswith(skip))


def draw(frame, lanes, gts=None, text=None):
    out = frame.copy()
    t = max(2, out.shape[1] // 400)
    for g in gts or []:
        cv2.polylines(out, [np.round(g).astype(np.int32).reshape(-1, 1, 2)], False, (255, 255, 255), max(1, t // 2), cv2.LINE_AA)
    for i, lane in enumerate(lanes):
        pts = np.round(lane).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, COLORS[i % 4], t, cv2.LINE_AA)
        for x, y in pts[:, 0]:
            cv2.circle(out, (int(x), int(y)), t + 1, COLORS[i % 4], -1)
    if text:
        cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, max(0.7, out.shape[1] / 1800), (0, 255, 255), 2, cv2.LINE_AA)
    return out


def contact_sheet(imgs, cols=2, width=820):
    tiles = []
    for im in imgs:
        h = int(round(im.shape[0] * width / im.shape[1]))
        tiles.append(cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA))
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="e.g. ..\\runs\\night_enh\\last.pt")
    ap.add_argument("--input", default="", help="folder of images (or a single image)")
    ap.add_argument("--index", default="", help="index.json: use frames from its TEST split instead of --input")
    ap.add_argument("--cond", default="night", help="with --index: night, normal, dazzle, shadow")
    ap.add_argument("--n", type=int, default=30, help="with --index: how many random test frames (0 = all in the index)")
    ap.add_argument("--seed", type=int, default=0, help="with --index: different seed = different frames")
    ap.add_argument("--out_dir", default="pred_out")
    ap.add_argument("--recursive", action="store_true", help="also look in sub-folders")
    ap.add_argument("--every", type=int, default=1, help="use every Nth image (handy for video-frame folders)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N images (0 = all)")
    ap.add_argument("--sheet_n", type=int, default=8, help="how many results to put on the contact sheet")
    ap.add_argument("--enhance", choices=["auto", "on", "off"], default="auto")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    index_mode = bool(a.index)
    if index_mode:
        test = json.load(open(a.index))["test"]
        if a.cond not in test or not test[a.cond]:
            raise SystemExit(f"no '{a.cond}' test frames in {a.index}; available: {[k for k, v in test.items() if v]}")
        paths = list(test[a.cond])
        random.Random(a.seed).shuffle(paths)
        paths = paths[: a.n] if a.n else paths
    else:
        if not a.input:
            raise SystemExit("give either --input <folder> or --index index.json")
        paths = find_images(a.input, a.recursive, a.out_dir)[:: max(1, a.every)]
        if a.limit:
            paths = paths[: a.limit]
    if not paths:
        raise SystemExit(f"no images ({', '.join(EXT)}) found in {a.input}")
    print(f"{len(paths)} images")

    device = lu.get_device()
    model, targs = lu.load_model(a.ckpt, device)
    enhance = targs.get("enhance", False) if a.enhance == "auto" else a.enhance == "on"
    print(f"device {device}  enhance {enhance}")

    first = cv2.imread(paths[0])
    if first is not None:
        lu.infer_frame(model, first, enhance, device)                 # warm-up, excluded from timing

    rows, tiles, used = [], [], set()
    tp_all = fp_all = fn_all = 0
    t_all = 0.0
    for k, p in enumerate(paths):
        frame = cv2.imread(p)
        if frame is None:
            print("skip (cannot read):", p)
            continue
        t0 = time.time()
        lanes = lu.infer_frame(model, frame, enhance, device)
        ms = (time.time() - t0) * 1000
        t_all += ms

        gts, tp, fp, fn, text = None, "", "", "", None
        if frame.shape[:2] == (lc.IMG_H0, lc.IMG_W0) and os.path.exists(lc.lines_path(p)):
            gts = lc.load_lanes(p, net=False)
            tp, fp, fn = lc.f1_counts(lanes, gts)
            tp_all += tp; fp_all += fp; fn_all += fn
            text = f"TP {tp}  FP {fp}  FN {fn}   (white = ground truth)"
        img = draw(frame, lanes, gts, text)

        name = os.path.splitext(os.path.basename(p))[0] + "_pred.jpg"
        if index_mode:                                                 # many clips share the names 00000.jpg, ...
            name = f"{a.cond}_{k:03d}_{os.path.basename(os.path.dirname(p))}_{name}"
        if name in used:
            name = f"{k:05d}_{name}"
        used.add(name)
        cv2.imwrite(os.path.join(a.out_dir, name), img)
        if len(tiles) < a.sheet_n:
            tiles.append(img)
        rows.append([p, name, len(lanes), f"{ms:.1f}", tp, fp, fn])
        print(f"{k + 1:4d}/{len(paths)}  lanes {len(lanes)}  {ms:5.1f} ms  {text or ''}  {os.path.basename(p)}")

    with open(os.path.join(a.out_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "output", "lanes_found", "ms", "TP", "FP", "FN"])
        w.writerows(rows)
    if tiles:
        cv2.imwrite(os.path.join(a.out_dir, "sheet.jpg"), contact_sheet(tiles))
    n = max(1, len(rows))
    print(f"\ndone: {len(rows)} images, avg {t_all / n:.1f} ms/image ({1000 * n / max(t_all, 1e-9):.1f} FPS, model inference only)")
    if tp_all + fp_all + fn_all:
        print(f"F1 over images that had ground truth: {lc.f1_score(tp_all, fp_all, fn_all) * 100:.1f}  (TP {tp_all} FP {fp_all} FN {fn_all})")
    print("saved to", os.path.abspath(a.out_dir), "(sheet.jpg = overview, results.csv = per-image table)")


if __name__ == "__main__":
    main()
