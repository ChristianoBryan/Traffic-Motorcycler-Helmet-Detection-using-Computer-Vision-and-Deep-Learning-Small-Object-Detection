"""Logika bersama: membaca experiments.yaml, membangun YAML model (arsitektur + atensi), menjalankan training."""

from __future__ import annotations

import copy
import itertools
import json
import time
from pathlib import Path

import yaml

import sodhelmet  # noqa: F401  (mendaftarkan modul custom + patch loss)
from sodhelmet.losses import LOSS_CFG, set_loss_config
from sodhelmet.modules import ATTENTION_BLOCKS

ROOT = Path(__file__).resolve().parents[1]
ARCHS = {"base": "yolov8-base.yaml", "p2": "yolov8-p2.yaml", "p2-gfpn": "yolov8-p2-gfpn.yaml"}
LOSS_DEFAULT = {"iou": "ciou", "nwd_ratio": 0.0, "nwd_c": 12.8, "piou_lambda": 1.2}


def _resolve(v, best):
    if isinstance(v, str) and v.startswith("$best."):
        return copy.deepcopy(best[v.split(".", 1)[1]])
    if isinstance(v, dict):
        return {k: _resolve(x, best) for k, x in v.items()}
    return v


def load_experiments(path: str | Path | None = None) -> dict:
    """Kembalikan {nama: konfigurasi lengkap} setelah defaults dan $best diterapkan."""
    path = Path(path or ROOT / "cfg" / "experiments.yaml")
    spec = yaml.safe_load(path.read_text())
    best, defaults = spec["best"], spec["defaults"]
    out = {}
    for name, e in spec["experiments"].items():
        e = {k: _resolve(v, best) for k, v in e.items()}
        cfg = copy.deepcopy(defaults)
        loss = {**LOSS_DEFAULT, **cfg.pop("loss", {}), **e.pop("loss", {})}
        cfg.update(e)
        cfg["loss"], cfg["name"] = loss, name
        out[name] = cfg
    return out


def grid_experiments(spec: dict[str, list], base: dict) -> dict:
    """Eksperimen ad-hoc dari kombinasi nilai, mis. {arch:[base,p2], imgsz:[640,960], loss:[ciou,piou2]}."""
    keys = list(spec)
    out = {}
    for vals in itertools.product(*(spec[k] for k in keys)):
        cfg = copy.deepcopy(base)
        parts = []
        for k, v in zip(keys, vals):
            if k == "loss":
                cfg["loss"] = {**LOSS_DEFAULT, **_LOSS_ALIASES[v]}
            else:
                cfg[k] = int(v) if k == "imgsz" else v
            parts.append(f"{k}-{v}")
        cfg["name"], cfg["stage"] = "grid_" + "_".join(parts), None
        out[cfg["name"]] = cfg
    return out


_LOSS_ALIASES = {
    "ciou": {"iou": "ciou"},
    "piou": {"iou": "piou"},
    "piou2": {"iou": "piou2"},
    "nwd": {"iou": "ciou", "nwd_ratio": 1.0},
    "ciou+nwd": {"iou": "ciou", "nwd_ratio": 0.5},
    "piou2+nwd": {"iou": "piou2", "nwd_ratio": 0.5},
}


def build_model_yaml(cfg: dict, out_dir: Path) -> Path:
    """Tulis YAML model final. Nama file memuat skala (yolov8s-...) agar Ultralytics membaca skala yang benar."""
    d = yaml.safe_load((ROOT / "cfg" / "models" / ARCHS[cfg["arch"]]).read_text())
    attn = cfg.get("attn", "none")
    if attn != "none":
        blk = ATTENTION_BLOCKS[attn].__name__
        d["head"] = [[f, n, (blk if m == "C2f" else m), a] for f, n, m, a in d["head"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg["arch"] + ("" if attn == "none" else f"-{attn}")
    path = out_dir / f"yolov8{cfg['scale']}-{tag}.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False))
    return path


def _pakai(d: Path) -> bool:
    """True jika folder ini sudah berisi hasil sebuah run (bukan sekadar folder kosong dari --dry-run)."""
    return any((d / f).exists() for f in ("weights", "results.csv", "result.json"))


def pilih_nama_run(project: str, name: str, seed: int, overwrite: bool) -> str:
    """Nama folder run yang belum terpakai: s0_baseline_640, lalu s0_baseline_640_r2, _r3, ... (kecuali overwrite)."""
    if overwrite:
        return name
    run_name, n = name, 1
    while _pakai(Path(project) / run_name / f"seed{seed}"):
        n += 1
        run_name = f"{name}_r{n}"
    return run_name


def run_one(cfg: dict, data: str, project: str, seed: int, device: str, workers: int, overrides: dict, dry_run=False,
            overwrite=False):
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    run_name = pilih_nama_run(project, cfg["name"], seed, overwrite)
    if run_name != cfg["name"]:
        print(f"folder hasil {cfg['name']}/seed{seed} sudah terpakai -> run ini disimpan di {run_name}/seed{seed}")
    run_dir = Path(project) / run_name / f"seed{seed}"
    model_yaml = build_model_yaml(cfg, run_dir / "model")
    set_loss_config(**cfg["loss"])
    print(f"\n=== {cfg['name']} | seed {seed} | {model_yaml.name} | imgsz {cfg['imgsz']} | loss {dict(LOSS_CFG)}")
    if dry_run:
        return None

    model = YOLO(str(model_yaml))
    pre = cfg.get("pretrained", "none")
    use_pre = pre not in (None, "none", "None", False)
    t0 = time.time()
    # Bobot pretrained HARUS lewat argumen `pretrained=` milik train(). Memanggil model.load() lalu
    # pretrained=False membuat trainer membangun ulang model dari YAML dan membuang bobotnya.
    args = dict(
        data=data, imgsz=cfg["imgsz"], epochs=cfg["epochs"], batch=cfg["batch"], seed=seed, deterministic=True,
        project=str(Path(project).resolve() / run_name), name=f"seed{seed}", exist_ok=True,
        device=device, workers=workers, pretrained=str(pre) if use_pre else False, plots=False,
    )  # fmt: skip
    args.update(overrides)
    model.train(**args)
    train_min = (time.time() - t0) / 60

    # lepas memori GPU dari training sebelum validasi akhir (GPU kecil, mis. 2 GB, bisa kehabisan memori)
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best = YOLO(str(Path(args["project"]) / args["name"] / "weights" / "best.pt"))
    val_kw = dict(data=data, imgsz=cfg["imgsz"], device=device, plots=False, verbose=False)
    try:
        m = best.val(batch=min(cfg["batch"], 4), **val_kw)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        print("validasi akhir kehabisan memori GPU, ulangi dengan batch=1")
        gc.collect()
        torch.cuda.empty_cache()
        m = best.val(batch=1, **val_kw)
    n_params = sum(p.numel() for p in best.model.parameters())
    
    # --- TAMBAHKAN PRINT INI UNTUK MEMASTIKAN BERJALAN ---
    print("Mengekstrak map50_s dan map50_m...")
    maps_list = list(m.box.maps) if hasattr(m.box.maps, "__iter__") else []
    print(f"Isi maps_list: {maps_list}")
    
    result = {
        "name": run_name, "base_name": cfg["name"], "seed": seed, "stage": cfg.get("stage"), "arch": cfg["arch"], "imgsz": cfg["imgsz"],
        "attn": cfg.get("attn", "none"), "loss": cfg["loss"], "epochs": args["epochs"],
        "precision": float(m.box.mp), "recall": float(m.box.mr), "map50": float(m.box.map50),
        "map50_95": float(m.box.map), 
        "map50_s": float(maps_list[0]) if len(maps_list) > 0 else 0.0,
        "map50_m": float(maps_list[1]) if len(maps_list) > 1 else 0.0,
        "params_M": n_params / 1e6, "gflops": float(get_flops(best.model, cfg["imgsz"])),
        "train_minutes": train_min,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print("hasil:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()})
    return result