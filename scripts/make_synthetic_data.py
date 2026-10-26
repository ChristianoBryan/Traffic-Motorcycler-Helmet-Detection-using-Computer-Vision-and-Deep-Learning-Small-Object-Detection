"""Buat dataset sintetis kecil (kotak berwarna di latar acak) untuk uji pipeline TANPA data asli.

  python scripts/make_synthetic_data.py --out /tmp/synth --n-train 24 --n-val 8 --size 640
Hasilnya format YOLO (images/ + labels/) dan <out>/data.yaml. BUKAN untuk mengukur akurasi.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    for split, n in (("train", a.n_train), ("val", a.n_val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = rng.integers(60, 140, (a.size, a.size, 3), dtype=np.uint8)
            lines = []
            for _ in range(rng.integers(2, 12)):
                bw, bh = rng.integers(8, 40, 2)  # objek kecil
                x, y = rng.integers(0, a.size - bw), rng.integers(0, a.size - bh)
                img[y : y + bh, x : x + bw] = (230, 200, 30)
                lines.append(f"0 {(x + bw / 2) / a.size:.6f} {(y + bh / 2) / a.size:.6f} {bw / a.size:.6f} {bh / a.size:.6f}")
            Image.fromarray(img).save(out / "images" / split / f"{i:03d}.jpg", quality=92)
            (out / "labels" / split / f"{i:03d}.txt").write_text("\n".join(lines) + "\n")
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: helmet\n")
    print("dataset sintetis:", out / "data.yaml")


if __name__ == "__main__":
    main()
