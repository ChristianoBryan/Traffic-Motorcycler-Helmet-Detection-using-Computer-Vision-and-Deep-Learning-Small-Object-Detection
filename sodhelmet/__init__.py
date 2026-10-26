"""sodhelmet: ekstensi YOLOv8 untuk deteksi helm berukuran kecil.

``import sodhelmet`` harus dilakukan SEBELUM membangun model dari YAML atau memuat
checkpoint (.pt) hasil training, karena checkpoint menyimpan referensi ke kelas di sini.
"""

from .losses import LOSS_CFG, patch_bbox_loss, set_loss_config
from .modules import ATTENTION_BLOCKS, ECA, EMA, CoordAtt, SimAM
from .register import register_modules

register_modules()
patch_bbox_loss()

__all__ = [
    "ATTENTION_BLOCKS", "ECA", "EMA", "CoordAtt", "SimAM", "LOSS_CFG", "set_loss_config",
]  # fmt: skip
