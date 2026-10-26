"""Ukur latensi, FPS, parameter, dan GFLOPs. Jalankan di PERANGKAT TARGET (CPU / Jetson / GPU).

Dua pengukuran:
  forward : hanya forward jaringan (tanpa pre/post-process)
  e2e     : model.predict() lengkap (letterbox + forward + NMS) pada gambar acak

Contoh:
  python scripts/benchmark_latency.py --weights runs/s2_p2_960/seed0/weights/best.pt --imgsz 960 --device cpu
  python scripts/benchmark_latency.py --cfg cfg/models/yolov8-p2.yaml --scale s --imgsz 640 --device 0 --half
Catatan Jetson: untuk angka deployment, export ke TensorRT (yolo export format=engine) lalu ukur dengan
mesin itu; angka PyTorch di sini hanya perbandingan relatif antar konfigurasi.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sodhelmet  # noqa: F401,E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def stats(ms):
    ms = sorted(ms)
    return {"mean_ms": statistics.mean(ms), "p50_ms": ms[len(ms) // 2], "p95_ms": ms[int(len(ms) * 0.95) - 1],
            "fps_mean": 1000 / statistics.mean(ms)}  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights")
    ap.add_argument("--cfg", help="YAML model (alternatif dari --weights; bobot acak, cukup untuk latensi)")
    ap.add_argument("--scale", default="s")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--no-e2e", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not (a.weights or a.cfg):
        ap.error("butuh --weights atau --cfg")

    if a.cfg:
        import re
        import shutil
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / f"yolov8{a.scale}-{re.sub(r'[^a-z0-9-]', '', Path(a.cfg).stem.replace('yolov8-', ''))}.yaml"
        shutil.copy(a.cfg, tmp)
        yolo = YOLO(str(tmp))
    else:
        yolo = YOLO(a.weights)
    dev = torch.device("cpu" if a.device == "cpu" else f"cuda:{a.device}" if a.device.isdigit() else a.device)
    net = yolo.model.to(dev).eval()
    if a.half and dev.type == "cuda":
        net = net.half()
    x = torch.rand(a.batch, 3, a.imgsz, a.imgsz, device=dev, dtype=torch.half if (a.half and dev.type == "cuda") else torch.float32)

    out = {
        "imgsz": a.imgsz, "device": str(dev), "half": bool(a.half), "batch": a.batch,
        "params_M": sum(p.numel() for p in net.parameters()) / 1e6,
        "gflops": float(get_flops(net, a.imgsz)),
    }  # fmt: skip
    with torch.inference_mode():
        for _ in range(a.warmup):
            net(x)
        sync(dev)
        ms = []
        for _ in range(a.runs):
            t = time.perf_counter()
            net(x)
            sync(dev)
            ms.append((time.perf_counter() - t) * 1000)
    out["forward"] = stats(ms)

    if not a.no_e2e:
        img = (np.random.rand(a.imgsz, a.imgsz, 3) * 255).astype(np.uint8)
        kw = dict(imgsz=a.imgsz, device=a.device, half=a.half, verbose=False, conf=0.25)
        for _ in range(max(5, a.warmup // 4)):
            yolo.predict(img, **kw)
        ms = []
        for _ in range(max(20, a.runs // 2)):
            t = time.perf_counter()
            yolo.predict(img, **kw)
            ms.append((time.perf_counter() - t) * 1000)
        out["e2e"] = stats(ms)

    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
