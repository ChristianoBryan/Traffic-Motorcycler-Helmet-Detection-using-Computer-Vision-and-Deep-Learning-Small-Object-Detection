# Referensi: kode asli SOD-YOLOv8

Salinan apa adanya dari https://github.com/Boshrakh/Yolov8-Small-Object-Detection-Arial-Images
(paper: Khalili & Smyth, "SOD-YOLOv8", Sensors 2024, 24(19), 6209). Folder ini HANYA untuk dibaca; tidak dipakai
oleh proyek. Cache `__pycache__` dan `.ipynb_checkpoints` dibuang.

Mengapa tidak dipakai langsung:
- Repo hanya berisi potongan fork Ultralytics (`nn/modules`, `utils`, `cfg`), bukan proyek yang bisa dijalankan.
- `nn/modules/block.py` memakai `from .bra_legacy import *` dan memanggil `EMA`, tetapi keduanya TIDAK ada di repo.
- Kodenya terikat pada versi Ultralytics lama; menimpa paket `ultralytics` dengannya merusak versi baru.

Yang diambil dari sini (dan dibandingkan): rumus PIoU/PIoU v2 (`utils/metrics.py`; implementasi baru di
`sodhelmet/losses.py` selisihnya ~1e-7), gagasan head P2 dan neck lintas-skala (`cfg/model/yolov8-p2.yaml`,
`yolov8s-GFPN_ema.yaml`). Lisensi kode Ultralytics: AGPL-3.0.
