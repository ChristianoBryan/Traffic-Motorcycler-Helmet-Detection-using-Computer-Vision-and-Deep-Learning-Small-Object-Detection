"""Mendaftarkan modul custom ke parser YAML Ultralytics (``parse_model``).

``parse_model`` menyimpan daftar modul yang dikenali (``base_modules`` dan ``repeat_modules``)
sebagai variabel lokal, jadi tidak bisa ditambah dari luar. Solusi yang tidak mengubah
paket Ultralytics di disk: ambil source-nya, sisipkan modul kita ke kedua daftar itu,
lalu compile ulang di namespace ``ultralytics.nn.tasks``.
"""

from __future__ import annotations

import inspect
import re

from ultralytics.nn import tasks

from .modules import ATTENTION_BLOCKS

_DONE = False


def register_modules() -> None:
    global _DONE
    if _DONE:
        return

    for cls in ATTENTION_BLOCKS.values():
        setattr(tasks, cls.__name__, cls)
    tasks._SOD_BLOCKS = tuple(ATTENTION_BLOCKS.values())

    src = inspect.getsource(tasks.parse_model)
    new_src, n1 = re.subn(r"(base_modules\s*=\s*frozenset\(\s*\{)", r"\1*_SOD_BLOCKS, ", src, count=1)
    new_src, n2 = re.subn(r"(repeat_modules\s*=\s*frozenset\([^\n]*\n\s*\{)", r"\1*_SOD_BLOCKS, ", new_src, count=1)
    if n1 != 1 or n2 != 1:
        raise RuntimeError(
            "Struktur parse_model di versi Ultralytics ini berbeda dari yang diuji (8.4.178). "
            "Jalankan: pip install ultralytics==8.4.178"
        )
    exec(compile(new_src, "<sodhelmet.parse_model>", "exec"), tasks.__dict__)  # rebinding tasks.parse_model
    _DONE = True
