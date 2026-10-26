"""Loss regresi kotak untuk objek kecil: PIoU / PIoU v2 dan NWD (dapat dicampur dengan CIoU).

Cara kerja: ``BboxLoss.forward`` milik Ultralytics diganti (monkey-patch) dengan versi yang
memilih jenis loss dari ``LOSS_CFG``. Bagian DFL tidak diubah.

Konfigurasi:
  iou        : "ciou" (default Ultralytics) | "piou" | "piou2"
  nwd_ratio  : 0..1. Loss akhir = (1 - r) * loss_iou + r * (1 - NWD). r=1 -> NWD murni.
  nwd_c      : konstanta C pada NWD, dalam PIKSEL gambar input (mengikuti dataset).
  piou_lambda: lambda pada PIoU v2.

Catatan: penugasan label (TAL assigner) tetap memakai CIoU standar.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from ultralytics.utils import loss as ul_loss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist

LOSS_CFG = {"iou": "ciou", "nwd_ratio": 0.0, "nwd_c": 12.8, "piou_lambda": 1.2}
_ORIG_FORWARD = ul_loss.BboxLoss.forward


def set_loss_config(iou: str = "ciou", nwd_ratio: float = 0.0, nwd_c: float = 12.8, piou_lambda: float = 1.2) -> dict:
    if iou not in {"ciou", "piou", "piou2"}:
        raise ValueError(f"iou harus ciou|piou|piou2, bukan {iou!r}")
    if not 0.0 <= nwd_ratio <= 1.0:
        raise ValueError("nwd_ratio harus di antara 0 dan 1")
    LOSS_CFG.update(iou=iou, nwd_ratio=float(nwd_ratio), nwd_c=float(nwd_c), piou_lambda=float(piou_lambda))
    return dict(LOSS_CFG)


def piou_loss(pred: torch.Tensor, target: torch.Tensor, v2: bool = False, lam: float = 1.2, eps: float = 1e-7):
    """Powerful-IoU (Liu et al., 2023). Kotak xyxy, bentuk (N,4) -> loss (N,1).

    L_v1 = (1 - IoU) + (1 - exp(-P^2)),  P = ((dw1+dw2)/w_gt + (dh1+dh2)/h_gt) / 4
    L_v2 = u(x) * L_v1,  u(x) = 3x exp(-x^2),  x = lam * exp(-P)
    """
    px1, py1, px2, py2 = pred.chunk(4, -1)
    tx1, ty1, tx2, ty2 = target.chunk(4, -1)
    pw, ph = (px2 - px1).clamp(min=eps), (py2 - py1).clamp(min=eps)
    tw, th = (tx2 - tx1).clamp(min=eps), (ty2 - ty1).clamp(min=eps)

    inter = (px2.minimum(tx2) - px1.maximum(tx1)).clamp(0) * (py2.minimum(ty2) - py1.maximum(ty1)).clamp(0)
    iou = inter / (pw * ph + tw * th - inter + eps)

    dw1, dw2 = (px1 - tx1).abs(), (px2 - tx2).abs()
    dh1, dh2 = (py1 - ty1).abs(), (py2 - ty2).abs()
    p = ((dw1 + dw2) / tw + (dh1 + dh2) / th) / 4
    l_v1 = (1.0 - iou) + (1.0 - torch.exp(-(p**2)))
    if not v2:
        return l_v1
    x = lam * torch.exp(-p)
    return 3.0 * x * torch.exp(-(x**2)) * l_v1


def nwd_loss(pred_px: torch.Tensor, target_px: torch.Tensor, c: float = 12.8, eps: float = 1e-7):
    """1 - NWD (Wang et al., 2021). Kotak xyxy dalam piksel, (N,4) -> (N,1)."""
    pcx, pcy = (pred_px[:, 0:1] + pred_px[:, 2:3]) / 2, (pred_px[:, 1:2] + pred_px[:, 3:4]) / 2
    tcx, tcy = (target_px[:, 0:1] + target_px[:, 2:3]) / 2, (target_px[:, 1:2] + target_px[:, 3:4]) / 2
    pw, ph = pred_px[:, 2:3] - pred_px[:, 0:1], pred_px[:, 3:4] - pred_px[:, 1:2]
    tw, th = target_px[:, 2:3] - target_px[:, 0:1], target_px[:, 3:4] - target_px[:, 1:2]
    w2_sq = (pcx - tcx) ** 2 + (pcy - tcy) ** 2 + ((pw - tw) / 2) ** 2 + ((ph - th) / 2) ** 2
    return 1.0 - torch.exp(-torch.sqrt(w2_sq + eps) / c)


def box_regression_loss(pred, target, stride_px=None, cfg: dict | None = None):
    """Loss kotak per-sampel (N,1). ``pred``/``target`` dalam satuan grid; ``stride_px`` (N,1) ke piksel."""
    cfg = cfg or LOSS_CFG
    if cfg["iou"] == "ciou":
        base = 1.0 - bbox_iou(pred, target, xywh=False, CIoU=True)
    else:
        base = piou_loss(pred, target, v2=cfg["iou"] == "piou2", lam=cfg["piou_lambda"])
    r = cfg["nwd_ratio"]
    if r <= 0:
        return base
    if stride_px is None:
        stride_px = torch.ones_like(pred[:, :1])
    nwd = nwd_loss(pred * stride_px, target * stride_px, c=cfg["nwd_c"])
    return (1.0 - r) * base + r * nwd


def _bbox_loss_forward(
    self,
    pred_dist,
    pred_bboxes,
    anchor_points,
    target_bboxes,
    target_scores,
    target_scores_sum,
    fg_mask,
    imgsz=None,
    stride=None,
):
    if LOSS_CFG["iou"] == "ciou" and LOSS_CFG["nwd_ratio"] <= 0:
        return _ORIG_FORWARD(
            self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,
            imgsz, stride,
        )  # fmt: skip

    fg = fg_mask.nonzero(as_tuple=True)  # (batch_idx, anchor_idx)
    weight = target_scores[fg].sum(-1, keepdim=True)
    stride_fg = stride[fg[1]] if stride is not None else None  # stride: (num_anchors, 1)
    box_l = box_regression_loss(pred_bboxes[fg], target_bboxes[fg], stride_fg)
    loss_iou = (box_l * weight).sum() / target_scores_sum

    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg].view(-1, self.dfl_loss.reg_max), target_ltrb[fg]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:  # reg_max <= 1 (L1), tidak dipakai YOLOv8 standar
        target_ltrb = bbox2dist(anchor_points, target_bboxes) * stride
        pd = pred_dist * stride
        loss_dfl = (F.l1_loss(pd[fg], target_ltrb[fg], reduction="none").mean(-1, keepdim=True) * weight).sum()
        loss_dfl = loss_dfl / target_scores_sum
    return loss_iou, loss_dfl


def patch_bbox_loss() -> None:
    ul_loss.BboxLoss.forward = _bbox_loss_forward
