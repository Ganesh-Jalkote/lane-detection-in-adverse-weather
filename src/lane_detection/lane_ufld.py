"""
lane_ufld.py - train / evaluate / export an Ultra-Fast-Lane-Detection style model (ResNet-18, row anchors)
for lane detection in adverse weather. Needs lane_core.py next to it.

    python lane_ufld.py train   --index index.json --out_dir runs/main --epochs 20
    python lane_ufld.py eval    --index index.json --ckpt runs/main/best.pt
    python lane_ufld.py export  --ckpt runs/main/best.pt
    python lane_ufld.py predict --ckpt runs/main/best.pt --source road.jpg --out out.png
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import lane_core as lc


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class UFLD(nn.Module):
    """ResNet-18 backbone -> 1x1 conv (512->8) -> FC -> (GRID+1, rows, lanes) logits."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        try:
            r = torchvision.models.resnet18(weights=weights)
        except Exception as e:  # e.g. no internet to fetch ImageNet weights
            print("WARNING: could not load ImageNet weights (", e, ") -> training from scratch")
            r = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.pool = nn.Conv2d(512, 8, 1)
        feat = 8 * (lc.H // 32) * (lc.W // 32)          # 8*9*25 = 1800
        self.cls = nn.Sequential(nn.Linear(feat, 2048), nn.ReLU(inplace=True),
                                 nn.Linear(2048, (lc.GRID + 1) * lc.NUM_ROWS * lc.NUM_LANES))

    def forward(self, x):
        f = self.pool(self.backbone(x)).flatten(1)
        return self.cls(f).view(-1, lc.GRID + 1, lc.NUM_ROWS, lc.NUM_LANES)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):                    # (B,C,R,L), (B,R,L)
        logp = F.log_softmax(logits.float(), dim=1)
        logp_t = logp.gather(1, labels.unsqueeze(1)).squeeze(1)
        return (-((1 - logp_t.exp()) ** self.gamma) * logp_t).mean()


# ----------------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------------
class TrainSet(Dataset):
    def __init__(self, paths, weather_p, enhance):
        self.paths, self.weather_p, self.enhance = paths, weather_p, enhance

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x, y = lc.train_sample(self.paths[i], np.random.default_rng(), self.weather_p, self.enhance)
        return torch.from_numpy(x), torch.from_numpy(y)


class ValSet(Dataset):
    """Real validation frames. (Only if synthetic=True: deterministic mix clean/fog/rain/night.)"""
    KINDS = [None, "fog", "rain", "night"]

    def __init__(self, paths, enhance, synthetic=False):
        self.paths, self.enhance, self.synthetic = paths, enhance, synthetic

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        x = lc.eval_image(p, self.KINDS[i % 4] if self.synthetic else None, seed=i, enhance=self.enhance)
        y = lc.make_labels(lc.load_lanes(p, net=True))
        return torch.from_numpy(x), torch.from_numpy(y)


class EvalSet(Dataset):
    def __init__(self, paths, kind, enhance):
        self.paths, self.kind, self.enhance = paths, kind, enhance

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.from_numpy(lc.eval_image(self.paths[i], self.kind, seed=i, enhance=self.enhance)), i


def _worker_init(_):
    cv2.setNumThreads(0)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UFLD(pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck.get("args", {})


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, crit, device, amp):
    model.eval()
    tot, n, correct, cnt, lane_correct, lane_cnt = 0.0, 0, 0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp):
            out = model(x)
        tot += crit(out, y).item() * x.size(0)
        n += x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        cnt += y.numel()
        m = y < lc.GRID
        lane_correct += ((pred - y).abs() <= 2)[m].sum().item()      # within +-2 bins (~8 px) on real lane cells
        lane_cnt += m.sum().item()
    model.train()
    return tot / max(n, 1), correct / max(cnt, 1), lane_correct / max(lane_cnt, 1)


def train(a):
    device = get_device()
    amp = device.type == "cuda"
    if amp:
        torch.backends.cudnn.benchmark = True             # fixed input size -> faster convolutions
    print(f"device: {device}  amp: {amp}")
    os.makedirs(a.out_dir, exist_ok=True)
    idx = json.load(open(a.index))
    lc.NIGHT_BRIGHTNESS = idx.get("night_thresh", lc.NIGHT_BRIGHTNESS)
    print(f"train images: {len(idx['train'])}  val images: {len(idx['val'])}  "
          f"night-only index: {idx.get('night_only', False)}  synthetic weather prob: {a.weather_p}")

    paths = [p for p, _ in idx["train"]]
    weights = [a.night_boost if 0 <= b < lc.NIGHT_BRIGHTNESS else 1.0 for _, b in idx["train"]]
    train_ds = TrainSet(paths, a.weather_p, a.enhance)
    sampler = WeightedRandomSampler(weights, num_samples=len(paths), replacement=True)
    # Windows starts every worker as a fresh process that imports torch (+CUDA DLLs, ~1-2 GB of virtual memory each),
    # so keep the TOTAL number of worker processes small: train workers stay alive, val uses only 2 short-lived ones.
    workers = a.workers if a.workers >= 0 else min(os.cpu_count() or 2, 6)
    pin = device.type == "cuda"
    kw = dict(num_workers=workers, pin_memory=pin, worker_init_fn=_worker_init)
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    train_dl = DataLoader(train_ds, batch_size=a.batch, sampler=sampler, drop_last=True, **kw)
    val_paths = idx["val"][: a.n_val]
    val_dl = DataLoader(ValSet(val_paths, a.enhance, a.weather_p > 0), batch_size=a.batch, shuffle=False,
                        num_workers=min(workers, 2), pin_memory=pin, worker_init_fn=_worker_init)

    model = UFLD(pretrained=True).to(device).to(memory_format=torch.channels_last)
    if a.init:                                            # fine-tune from an earlier run (e.g. night model -> fog/rain data)
        model.load_state_dict(torch.load(a.init, map_location=device, weights_only=False)["model"])
        print("initialised weights from", a.init)
    crit = SoftmaxFocalLoss(2.0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    steps_per_epoch = len(train_dl)
    total = a.epochs * steps_per_epoch
    warm = max(1, min(300, total // 10))
    lr_at = lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))) * 0.99 + 0.01
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    start_ep, step, best = 0, 0, float("inf")
    last = os.path.join(a.out_dir, "last.pt")
    if os.path.exists(last) and not a.fresh:
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start_ep, step, best = ck["epoch"] + 1, ck["step"], ck["best"]
        for _ in range(step):
            sched.step()
        print(f"resumed from {last}: epoch {start_ep}, step {step}")

    def save(path, ep, full):
        d = {"model": model.state_dict(), "epoch": ep, "step": step, "best": best,
             "args": {k: v for k, v in vars(a).items() if k != "fn"}}
        if full:                                           # optimizer state only needed for resuming
            d.update(opt=opt.state_dict(), scaler=scaler.state_dict())
        torch.save(d, path)

    t0 = time.time()
    model.train()
    for ep in range(start_ep, a.epochs):
        run, nb, te = 0.0, 0, time.time()
        for x, y in train_dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                out = model(x)
            loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run += loss.item(); nb += 1
            if nb % 50 == 0:
                print(f"  ep {ep} it {nb}/{steps_per_epoch} loss {run / nb:.4f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"{nb * a.batch / (time.time() - te):.0f} img/s", flush=True)
            if a.max_iters and nb >= a.max_iters:
                break
        vl, vacc, vlane = validate(model, val_dl, crit, device, amp) if len(val_paths) else (run / max(nb, 1), 0, 0)
        print(f"== epoch {ep + 1}/{a.epochs}  train {run / max(nb, 1):.4f}  val {vl:.4f}  "
              f"val-acc {vacc:.3f}  val-lane-acc(+-2bins) {vlane:.3f}  epoch {time.time() - te:.0f}s", flush=True)
        if vl < best:
            best = vl
            save(os.path.join(a.out_dir, "best.pt"), ep, False)
            print("   new best -> best.pt")
        save(last, ep, True)
        if a.max_iters:
            break
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            print(f"time limit {a.max_minutes} min reached - stopped after epoch {ep + 1}. Re-run the same command to resume.")
            break
    print("done. best val loss", best)


# ----------------------------------------------------------------------------
# evaluate (per-condition F1, CULane-style)
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_paths(model, paths, kind, enhance, device, batch=32, workers=2):
    dl = DataLoader(EvalSet(paths, kind, enhance), batch_size=batch, num_workers=workers, worker_init_fn=_worker_init)
    preds = []
    for x, _ in dl:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(x.to(device))
        out = out.float().cpu().numpy()
        preds += [lc.decode(o) for o in out]
    return preds


def evaluate(a):
    device = get_device()
    model, targs = load_model(a.ckpt, device)
    enhance = targs.get("enhance", False) if a.enhance == "auto" else a.enhance == "on"
    idx = json.load(open(a.index))
    conds = {}                                            # name -> (paths, synthetic kind)
    for c, ps in idx["test"].items():
        if ps:
            conds[c + " (real)"] = (ps[: a.n_eval], None)
    base = idx["test"].get("normal", [])[: a.n_eval]
    for k in lc.WEATHER_KINDS:
        if base and a.synthetic:
            conds[f"{k} (synthetic on normal)"] = (base, k)
    res = {}
    print(f"enhance={enhance}   F1 @ IoU>0.5, CULane-style (python approximation of the official tool)\n")
    for name, (ps, kind) in conds.items():
        t = time.time()
        preds = predict_paths(model, ps, kind, enhance, device, a.batch)
        tp = fp = fn = 0
        for p, pr in zip(ps, preds):
            a1, b1, c1 = lc.f1_counts(pr, lc.load_lanes(p, net=False))
            tp += a1; fp += b1; fn += c1
        f1 = lc.f1_score(tp, fp, fn)
        res[name] = {"f1": f1, "tp": tp, "fp": fp, "fn": fn, "n": len(ps)}
        print(f"{name:34s} n={len(ps):5d}  F1={f1 * 100:5.1f}   (TP {tp} FP {fp} FN {fn})  [{time.time() - t:.0f}s]", flush=True)
    out = os.path.join(os.path.dirname(a.ckpt) or ".", f"results_enhance_{'on' if enhance else 'off'}.json")
    json.dump(res, open(out, "w"), indent=2)
    print("saved", out)


# ----------------------------------------------------------------------------
# export
# ----------------------------------------------------------------------------
def export(a):
    model, targs = load_model(a.ckpt, torch.device("cpu"))
    out_dir = a.out_dir or os.path.dirname(a.ckpt) or "."
    dummy = torch.zeros(1, 3, lc.H, lc.W)
    onnx_path = os.path.join(out_dir, "lane_ufld.onnx")
    kwargs = dict(input_names=["image"], output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=17)
    try:
        torch.onnx.export(model, dummy, onnx_path, dynamo=False, **kwargs)
    except TypeError:                                     # older torch without the dynamo kwarg
        torch.onnx.export(model, dummy, onnx_path, **kwargs)
    print("wrote", onnx_path)
    ts_path = os.path.join(out_dir, "lane_ufld.torchscript.pt")
    torch.jit.trace(model, dummy).save(ts_path)
    print("wrote", ts_path)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        o = sess.run(None, {"image": dummy.numpy()})[0]
        d = np.abs(o - model(dummy).detach().numpy()).max()
        print(f"onnxruntime check OK, max abs diff vs torch: {d:.2e}")
        x = dummy.numpy(); t = time.time()
        for _ in range(10):
            sess.run(None, {"image": x})
        print(f"CPU latency (onnxruntime): {(time.time() - t) / 10 * 1000:.0f} ms / frame")
    except ImportError:
        print("(pip install onnxruntime to verify the export and measure CPU FPS)")
    json.dump({"input": [1, 3, lc.H, lc.W], "enhance": targs.get("enhance", False), "grid": lc.GRID,
               "num_rows": lc.NUM_ROWS, "num_lanes": lc.NUM_LANES, "row_anchors": lc.ROW_ANCHORS.tolist(),
               "mean": lc.MEAN.tolist(), "std": lc.STD.tolist(), "orig_size": [lc.IMG_W0, lc.IMG_H0]},
              open(os.path.join(out_dir, "model_meta.json"), "w"), indent=2)


# ----------------------------------------------------------------------------
# predict (single image / video) - this is also what the FastAPI endpoint should call
# ----------------------------------------------------------------------------
@torch.no_grad()
def infer_frame(model, frame, enhance, device):
    """frame: BGR uint8 of any size. Returns lanes in the frame's own pixel coordinates."""
    fh, fw = frame.shape[:2]
    ch = min(fh, int(round(fw * lc.IMG_H0 / lc.IMG_W0)))   # keep CULane aspect: use the bottom part of the frame
    y0 = fh - ch
    crop = cv2.resize(frame[y0:], (lc.W, lc.H), interpolation=cv2.INTER_AREA)
    if enhance:
        crop = lc.enhance_image(crop)
    x = torch.from_numpy(lc.to_input(crop)).unsqueeze(0).to(device)
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        out = model(x)
    lanes = lc.decode(out.float().cpu().numpy()[0])
    sx, sy = fw / lc.IMG_W0, ch / lc.IMG_H0
    return [l * np.array([sx, sy]) + np.array([0, y0]) for l in lanes]


def draw_lanes(frame, lanes):
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for i, l in enumerate(lanes):
        for pt in l:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, colors[i % 4], -1)
    return frame


def predict(a):
    device = get_device()
    model, targs = load_model(a.ckpt, device)
    enhance = targs.get("enhance", False) if a.enhance == "auto" else a.enhance == "on"
    if a.source.lower().endswith((".jpg", ".jpeg", ".png")):
        frame = cv2.imread(a.source)
        cv2.imwrite(a.out, draw_lanes(frame, infer_frame(model, frame, enhance, device)))
        print("wrote", a.out)
        return
    cap = cv2.VideoCapture(a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n, t = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vw.write(draw_lanes(frame, infer_frame(model, frame, enhance, device)))
        n += 1
    vw.release()
    print(f"wrote {a.out}  ({n} frames, {n / (time.time() - t):.1f} FPS incl. video I/O)")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train")
    p.add_argument("--index", default="index.json")
    p.add_argument("--out_dir", default="runs/main")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weather_p", type=float, default=0.0, help="prob. of SYNTHETIC fog/rain/night per training image (default 0 = real images only)")
    p.add_argument("--init", default="", help="start from the weights of an earlier best.pt (fine-tuning)")
    p.add_argument("--enhance", action="store_true", help="CLAHE/gamma/dehaze pre-processing block (train + inference)")
    p.add_argument("--night_boost", type=float, default=3.0, help="sampling weight of dark (night) frames")
    p.add_argument("--workers", type=int, default=-1, help="data-loading processes (default: min(cpu cores, 6))")
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--max_minutes", type=float, default=0, help="stop gracefully after N minutes (resume by re-running)")
    p.add_argument("--max_iters", type=int, default=0, help="debug: iterations per epoch, then stop")
    p.add_argument("--fresh", action="store_true", help="ignore an existing last.pt")
    p.set_defaults(fn=train)

    p = sub.add_parser("eval")
    p.add_argument("--index", default="index.json")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_eval", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--enhance", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--synthetic", action="store_true", help="also score synthetic fog/rain/night on normal frames (off = real images only)")
    p.set_defaults(fn=evaluate)

    p = sub.add_parser("export")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", default="")
    p.set_defaults(fn=export)

    p = sub.add_parser("predict")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="pred_out.png")
    p.add_argument("--enhance", choices=["auto", "on", "off"], default="auto")
    p.set_defaults(fn=predict)

    args = ap.parse_args()
    args.fn(args)
