"""Evaluasi detektor helm: inferensi biasa ATAU tiling SAHI, dengan metrik per-ukuran objek.

Kedua mode memakai pencocokan & perhitungan AP yang sama (gaya validator Ultralytics), jadi
angka inferensi biasa vs SAHI bisa dibandingkan langsung.

Contoh:
  python scripts/evaluate.py --weights runs/s2_p2_960/seed0/weights/best.pt --data data/helmet.yaml --imgsz 960
  python scripts/evaluate.py --weights best.pt --data data/helmet.yaml --sahi --slice 640 --overlap 0.2 --imgsz 640

Keluaran: P, R, mAP50, mAP50-95 (semua objek) + recall@IoU0.5 (conf>=--conf) per kelompok ukuran
(sisi akar luas, dalam piksel gambar ASLI: tiny/small/medium/large menurut --size-bins).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sodhelmet  # noqa: F401,E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.metrics import ap_per_class, box_iou  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(data_yaml: str, split: str) -> list[Path]:
    d = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(d.get("path", Path(data_yaml).parent))
    if not root.is_absolute():
        root = (Path(data_yaml).parent / root).resolve()
    src = d[split]
    srcs = src if isinstance(src, list) else [src]
    files: list[Path] = []
    for s in srcs:
        p = Path(s) if Path(s).is_absolute() else root / s
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
        elif p.suffix == ".txt":
            files += [Path(x.strip()) if Path(x.strip()).is_absolute() else root / x.strip() for x in p.read_text().splitlines() if x.strip()]
        else:
            raise FileNotFoundError(f"split {split!r}: tidak menemukan {p}")
    return files


def label_path(img: Path) -> Path:
    parts = list(img.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_gt(img: Path, w: int, h: int):
    lp = label_path(img)
    if not lp.exists() or lp.stat().st_size == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    a = np.loadtxt(lp, ndmin=2, dtype=np.float32)[:, :5]
    cls = a[:, 0].astype(np.int64)
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1), cls


def match_predictions(gt, gt_cls, pb, pcls, iouv):
    """Matriks benar (n_pred, n_iouv) dengan pencocokan greedy 1-ke-1 seperti validator Ultralytics."""
    correct = np.zeros((len(pb), len(iouv)), dtype=bool)
    if len(gt) == 0 or len(pb) == 0:
        return correct, np.zeros((len(pb),), dtype=np.float32) - 1
    iou = box_iou(torch.from_numpy(gt), torch.from_numpy(pb)).numpy() * (gt_cls[:, None] == pcls[None, :])
    gt_of_pred = np.full(len(pb), -1.0, np.float32)  # GT yang cocok pada IoU 0.5 (untuk recall per ukuran)
    for i, t in enumerate(iouv):
        x = np.nonzero(iou >= t)
        if x[0].shape[0]:
            m = np.concatenate([np.stack(x, 1), iou[x[0], x[1]][:, None]], 1)
            if x[0].shape[0] > 1:
                m = m[m[:, 2].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
            if i == 0:
                gt_of_pred[m[:, 1].astype(int)] = m[:, 0]
    return correct, gt_of_pred


def _side(b):
    return np.sqrt(np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None))


def ap50_in_range(records, lo, hi):
    """AP50 hanya untuk objek ber-ukuran [lo, hi) piksel (sisi = akar luas kotak), gaya COCO.

    GT di luar rentang diabaikan. Prediksi yang cocok dengan GT di luar rentang, atau tidak cocok dengan
    GT mana pun tetapi ukurannya di luar rentang, diabaikan (tidak dihitung sebagai salah). None bila
    tidak ada GT dalam rentang.
    """
    tps, confs, pclss, tcls = [], [], [], []
    for gt, gcls, pb, pc, pk in records:
        gvalid = (_side(gt) >= lo) & (_side(gt) < hi) if len(gt) else np.zeros(0, bool)
        tcls.append(gcls[gvalid])
        if len(pb) == 0:
            continue
        order = np.argsort(-pc)
        pb, pc, pk = pb[order], pc[order], pk[order]
        pin = (_side(pb) >= lo) & (_side(pb) < hi)
        keep, tp = pin.copy(), np.zeros(len(pb), bool)
        if len(gt):
            iou = box_iou(torch.from_numpy(gt), torch.from_numpy(pb)).numpy() * (gcls[:, None] == pk[None, :])
            taken = np.zeros(len(gt), bool)
            for j in range(len(pb)):
                col = np.where(taken, 0.0, iou[:, j])
                cv = np.where(gvalid, col, 0.0)  # GT dalam rentang diutamakan
                if cv.max() >= 0.5:
                    taken[cv.argmax()] = True
                    tp[j], keep[j] = True, True
                    continue
                ci = np.where(~gvalid, col, 0.0)
                if ci.max() >= 0.5:  # cocok dengan GT di luar rentang -> abaikan
                    taken[ci.argmax()] = True
                    keep[j] = False
        tps.append(tp[keep]), confs.append(pc[keep]), pclss.append(pk[keep])
    tcls = np.concatenate(tcls) if tcls else np.zeros(0, np.int64)
    if len(tcls) == 0:
        return None
    if not tps or sum(len(t) for t in tps) == 0:
        return 0.0
    tp = np.concatenate(tps)[:, None]
    res = ap_per_class(tp, np.concatenate(confs), np.concatenate(pclss), tcls, plot=False)
    return float(res[5][:, 0].mean())


def build_predictor(args):
    if args.sahi:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        det = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=args.weights, confidence_threshold=args.min_conf, device=args.device,
            image_size=args.imgsz,
        )  # fmt: skip

        def predict(img_path):
            r = get_sliced_prediction(
                str(img_path), det, slice_height=args.slice, slice_width=args.slice,
                overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
                perform_standard_pred=not args.no_full_pass, postprocess_type=args.postprocess,
                postprocess_match_threshold=args.match_thr, verbose=0, auto_slice_resolution=False,
            )  # fmt: skip
            ps = r.object_prediction_list
            if not ps:
                return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
            return (
                np.array([p.bbox.to_xyxy() for p in ps], np.float32),
                np.array([p.score.value for p in ps], np.float32),
                np.array([p.category.id for p in ps], np.int64),
            )

        return predict
    model = YOLO(args.weights)

    def predict(img_path):
        r = model.predict(str(img_path), imgsz=args.imgsz, conf=args.min_conf, iou=args.nms_iou, device=args.device,
                          verbose=False, max_det=1000)[0]  # fmt: skip
        b = r.boxes
        return b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(np.int64)

    return predict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640, help="ukuran input model (per-irisan bila --sahi)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-conf", type=float, default=0.001, help="ambang rendah untuk menghitung AP")
    ap.add_argument("--conf", type=float, default=0.25, help="ambang untuk recall per ukuran")
    ap.add_argument("--nms-iou", type=float, default=0.6)
    ap.add_argument("--size-bins", type=float, nargs=3, default=[16, 32, 96], help="batas sisi (px) tiny|small|medium|large")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--sahi", action="store_true")
    ap.add_argument("--slice", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--postprocess", default="GREEDYNMM", help="GREEDYNMM|NMM|NMS|LSNMS")
    ap.add_argument("--match-thr", type=float, default=0.5)
    ap.add_argument("--no-full-pass", action="store_true", help="SAHI tanpa inferensi gambar penuh tambahan")
    ap.add_argument("--out", help="simpan hasil ke JSON")
    args = ap.parse_args()

    from PIL import Image

    images = list_images(args.data, args.split)
    if args.max_images:
        images = images[: args.max_images]
    predict = build_predictor(args)
    iouv = np.linspace(0.5, 0.95, 10)
    tp, conf, pcls, tcls = [], [], [], []
    records = []  # (gt, gt_cls, pred_boxes, conf, pred_cls) per gambar, untuk AP50 per ukuran
    names = ["tiny", "small", "medium", "large"]
    edges = np.array(args.size_bins)
    n_gt = np.zeros(4, int)
    n_hit = np.zeros(4, int)
    t0 = time.time()
    for k, img in enumerate(images):
        with Image.open(img) as im:
            w, h = im.size
        gt, gcls = load_gt(img, w, h)
        pb, pc, pk = predict(img)
        correct, gt_of_pred = match_predictions(gt, gcls, pb, pk, iouv)
        tp.append(correct), conf.append(pc), pcls.append(pk), tcls.append(gcls)
        records.append((gt, gcls, pb, pc, pk))
        if len(gt):
            side = np.sqrt((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))
            bucket = np.searchsorted(edges, side, side="right")  # 0..3
            hit = np.zeros(len(gt), bool)
            sel = (pc >= args.conf) & (gt_of_pred >= 0)
            hit[gt_of_pred[sel].astype(int)] = True
            for b in range(4):
                n_gt[b] += int((bucket == b).sum())
                n_hit[b] += int(hit[bucket == b].sum())
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(images)} gambar", flush=True)
    secs = time.time() - t0

    tp, conf, pcls, tcls = (np.concatenate(x, 0) for x in (tp, conf, pcls, tcls))
    if len(tp) == 0 or tp.shape[0] == 0:
        p = r = map50 = map5095 = 0.0
    else:
        res = ap_per_class(tp, conf, pcls, tcls, plot=False)
        p, r, ap_all = float(res[2].mean()), float(res[3].mean()), res[5]
        map50, map5095 = float(ap_all[:, 0].mean()), float(ap_all.mean())
    e = args.size_bins
    ranges = {
        f"tiny_<{e[0]:g}px": (0, e[0]), f"small_{e[0]:g}-{e[1]:g}px": (e[0], e[1]),
        f"medium_{e[1]:g}-{e[2]:g}px": (e[1], e[2]), f"large_>={e[2]:g}px": (e[2], float("inf")),
        f"gabungan_tiny+small_<{e[1]:g}px": (0, e[1]), "semua_ukuran": (0, float("inf")),
    }  # fmt: skip
    ap50_by_size = {k: ap50_in_range(records, lo, hi) for k, (lo, hi) in ranges.items()}
    out = {
        "weights": args.weights, "mode": "sahi" if args.sahi else "plain", "imgsz": args.imgsz,
        "ap50_by_size": ap50_by_size,
        "images": len(images), "gt_boxes": int(len(tcls)), "precision": p, "recall": r,
        "map50": map50, "map50_95": map5095, "seconds_per_image": secs / max(len(images), 1),
        "recall_by_size": {names[b]: (n_hit[b] / n_gt[b] if n_gt[b] else None) for b in range(4)},
        "gt_by_size": {names[b]: int(n_gt[b]) for b in range(4)},
    }  # fmt: skip
    if args.sahi:
        out["sahi"] = {"slice": args.slice, "overlap": args.overlap, "postprocess": args.postprocess}
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()