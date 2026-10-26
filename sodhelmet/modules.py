"""Modul atensi dan blok C2f+atensi untuk eksperimen deteksi helm (objek kecil).

Semua atensi menjaga jumlah kanal (c -> c), jadi bisa disisipkan di mana saja.
Blok ``C2f_<ATT>`` = C2f standar YOLOv8 + atensi pada keluaran (setelah cv2).
Penempatan seragam untuk semua jenis atensi, supaya perbandingan antar atensi adil.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f


class SimAM(nn.Module):
    """SimAM (Yang et al., 2021): atensi 3D tanpa parameter tambahan."""

    def __init__(self, channels: int | None = None, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        n = max(h * w - 1, 1)
        d = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        v = d.sum(dim=(2, 3), keepdim=True) / n
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return x * torch.sigmoid(e_inv)


class ECA(nn.Module):
    """ECA-Net (Wang et al., 2020): atensi kanal, hanya satu Conv1d kecil."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(-1, -2)  # (b,1,c)
        y = torch.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)  # (b,c,1,1)
        return x * y


class _HSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.nn.functional.relu6(x + 3.0) / 6.0


class CoordAtt(nn.Module):
    """Coordinate Attention (Hou et al., 2021): meng-encode posisi horizontal & vertikal."""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mip = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mip, 1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = _HSwish()
        self.conv_h = nn.Conv2d(mip, channels, 1)
        self.conv_w = nn.Conv2d(mip, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_h = self.pool_h(x)  # (b,c,h,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b,c,w,1)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w.permute(0, 1, 3, 2)))
        return x * a_w * a_h


class EMA(nn.Module):
    """Efficient Multi-scale Attention (Ouyang et al., 2023).

    ``factor`` = jumlah grup; otomatis diturunkan bila kanal tidak habis dibagi.
    """

    def __init__(self, channels: int, factor: int = 8):
        super().__init__()
        while factor > 1 and channels % factor:
            factor //= 2
        self.groups = factor
        cg = channels // factor
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(cg, cg)
        self.conv1x1 = nn.Conv2d(cg, cg, 1)
        self.conv3x3 = nn.Conv2d(cg, cg, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.groups
        gx = x.reshape(b * g, -1, h, w)
        x_h = self.pool_h(gx)
        x_w = self.pool_w(gx).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(gx * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(gx)
        x11 = self.softmax(self.agp(x1).reshape(b * g, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * g, c // g, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * g, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * g, c // g, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * g, 1, h, w)
        return (gx * weights.sigmoid()).reshape(b, c, h, w)


# ---------------------------------------------------------------- blok C2f + atensi
class C2f_SimAM(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.att = SimAM(c2)

    def forward(self, x):
        return self.att(super().forward(x))

    def forward_split(self, x):
        return self.att(super().forward_split(x))


class C2f_ECA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.att = ECA(c2)

    def forward(self, x):
        return self.att(super().forward(x))

    def forward_split(self, x):
        return self.att(super().forward_split(x))


class C2f_CA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.att = CoordAtt(c2)

    def forward(self, x):
        return self.att(super().forward(x))

    def forward_split(self, x):
        return self.att(super().forward_split(x))


class C2f_EMA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.att = EMA(c2)

    def forward(self, x):
        return self.att(super().forward(x))

    def forward_split(self, x):
        return self.att(super().forward_split(x))


ATTENTION_BLOCKS = {
    "simam": C2f_SimAM,
    "ca": C2f_CA,
    "eca": C2f_ECA,
    "ema": C2f_EMA,
}
