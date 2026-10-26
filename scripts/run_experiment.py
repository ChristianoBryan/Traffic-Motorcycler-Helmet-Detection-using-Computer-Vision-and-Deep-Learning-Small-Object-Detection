"""Jalankan eksperimen dari cfg/experiments.yaml (atau kombinasi ad-hoc dengan --grid).

Contoh:
  python scripts/run_experiment.py --list
  python scripts/run_experiment.py --data data/helmet.yaml --stage 0
  python scripts/run_experiment.py --data data/helmet.yaml --exp s1_base_960 s2_p2_960 --seeds 0 1 2
  python scripts/run_experiment.py --data data/helmet.yaml --grid arch=base,p2 imgsz=640,960 loss=ciou,piou2
  python scripts/run_experiment.py --data data/helmet.yaml --stage 2 --dry-run
"""

from __future__ import annotations
 
import argparse
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sodhelmet.experiment import grid_experiments, load_experiments, run_one  # noqa: E402
 
 
def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        if v in ("True", "False"):
            v = v == "True"
        out[k] = v
    return out
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/helmet.yaml")
    ap.add_argument("--config", default=None, help="path experiments.yaml (default cfg/experiments.yaml)")
    ap.add_argument("--stage", type=int, nargs="*", help="nomor tahap (0-7)")
    ap.add_argument("--exp", nargs="*", help="nama eksperimen")
    ap.add_argument("--grid", nargs="*", help="kombinasi ad-hoc, mis. arch=base,p2 imgsz=640,960 loss=ciou,piou2 attn=none,simam")
    ap.add_argument("--seeds", type=int, nargs="*", help="menimpa seeds di config")
    ap.add_argument("--scale", choices=list("nslmx"), help="menimpa skala model (default dari config)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--pretrained", help="mis. yolov8s.pt atau none")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--override", nargs="*", help="argumen Ultralytics tambahan, mis. patience=30 cos_lr=True")
    ap.add_argument("--overwrite", action="store_true",
                    help="timpa folder hasil lama (default: run baru otomatis masuk folder baru _r2, _r3, ...)")  # fmt: skip
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="hanya cetak konfigurasi & tulis YAML model")
    a = ap.parse_args()
 
    exps = load_experiments(a.config)
    if a.grid:
        base = next(iter(exps.values()))
        spec = {kv.split("=")[0]: kv.split("=")[1].split(",") for kv in a.grid}
        base = {**base, "attn": "none"}
        exps = grid_experiments(spec, base)
        chosen = list(exps)
    elif a.list:
        for n, c in exps.items():
            print(f"[tahap {c['stage']}] {n:16s} arch={c['arch']:8s} imgsz={c['imgsz']:<5d} attn={c['attn']:5s} loss={c['loss']}")
        return
    else:
        chosen = list(exps)
        if a.stage is not None:
            chosen = [n for n in chosen if exps[n]["stage"] in a.stage]
        if a.exp:
            chosen = [n for n in exps if n in a.exp]
        if not (a.stage is not None or a.exp):
            ap.error("pilih --stage, --exp, atau --grid (atau --list)")
    if not chosen:
        ap.error("tidak ada eksperimen yang cocok")
 
    overrides = parse_overrides(a.override)
    for n in chosen:
        cfg = exps[n]
        if a.scale:
            cfg["scale"] = a.scale
        if a.epochs:
            cfg["epochs"] = a.epochs
        if a.batch:
            cfg["batch"] = a.batch
        if a.pretrained:
            cfg["pretrained"] = a.pretrained
        for seed in a.seeds if a.seeds else cfg["seeds"]:
            run_one(cfg, a.data, a.project, seed, a.device, a.workers, overrides, dry_run=a.dry_run, overwrite=a.overwrite)
 
 
if __name__ == "__main__":
    main()

#640 : python scripts/run_experiment.py --data data/helmet.yaml --exp s0_baseline_640 --epochs 10 --batch 8 --device 0
#960 : python scripts/run_experiment.py --data data/helmet.yaml --exp s1_base_960 --epochs 10 --batch 4 --device 0

"""
Tahap   | Konfigurasi       | Tujuan
0       | YOLOv8s, imgsz 640| Baseline
1       | imgsz 960/1280    | Lihat dampak resolusi (biasanya besar di 1080p)
2       | + head P2         | Dampak layer deteksi kecil
3       | + PIoU (atau NWD) | Dampak loss
4       | + GFPN            | Hanya jika latensi masih aman
"""

""" 
    Images count (After Augmentation) :
    Dataset Awal  : 440 (356, 51, 33)
    Dataset add 1 : 2352 (2166, 186, 0)
    Dataset add 2 : 5293 (5001, 217, 75)
    Total Split : 2792 (6372, 313 , 192)

    Annotation Amount (Before Augmentation):
    Dataset Awal : 513, 408
    Dataset add 1 : 2024, 1285
    Dataset add 2 : 3068, 1790
    Total* : 5605, 3483

    *hasn't been recounted
    **edited for balance
"""