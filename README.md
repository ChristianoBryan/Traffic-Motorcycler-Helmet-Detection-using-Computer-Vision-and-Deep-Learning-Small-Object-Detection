# Deteksi Helm Objek Kecil dengan YOLOv8 (kerangka eksperimen tugas akhir)

Proyek ini menata ulang repo [SOD-YOLOv8](https://github.com/Boshrakh/Yolov8-Small-Object-Detection-Arial-Images)
menjadi kerangka eksperimen yang bisa dijalankan di atas **Ultralytics resmi** (versi teruji: `8.4.178`).
Semua tahap rencana Anda tersedia: baseline → resolusi → head P2 → PIoU/NWD → GFPN → atensi → kombinasi penuh,
plus inferensi **SAHI**, metrik per-ukuran objek, dan pengukur **latensi**.

## Isi proyek

```
cfg/models/        yolov8-base.yaml · yolov8-p2.yaml · yolov8-p2-gfpn.yaml
cfg/experiments.yaml   semua eksperimen (tahap 0–6) + blok `best` untuk diisi
sodhelmet/         modules.py (SimAM, ECA, CoordAtt, EMA, blok C2f_<atensi>)
                   losses.py  (PIoU, PIoU v2, NWD, campuran)  register.py  experiment.py
scripts/           run_experiment.py · evaluate.py (plain/SAHI) · benchmark_latency.py
                   summarize_results.py · make_synthetic_data.py
data/helmet.yaml   template dataset (isi path & kelas Anda)
reference/         salinan kode asli repo (hanya untuk dibaca)
```

## Mulai cepat (Google Colab / Kaggle, GPU gratis)

```bash
pip install -r requirements.txt
# letakkan dataset YOLO Anda, lalu sesuaikan data/helmet.yaml
python scripts/run_experiment.py --list                       # lihat semua eksperimen
python scripts/run_experiment.py --data data/helmet.yaml --stage 0 1     # baseline + resolusi
python scripts/run_experiment.py --data data/helmet.yaml --stage 2       # + head P2
python scripts/summarize_results.py --runs runs                          # tabel ablation
```

Tes cepat tanpa data asli (CPU, beberapa menit):
```bash
python scripts/make_synthetic_data.py --out /tmp/synth --n-train 16 --n-val 6 --size 320
python scripts/run_experiment.py --data /tmp/synth/data.yaml --grid arch=p2 imgsz=320 loss=piou2 attn=simam \
  --scale n --epochs 2 --batch 4 --pretrained none --device cpu --workers 0
```

## Rencana eksperimen

| Tahap | Eksperimen (`--stage`) | Tujuan |
|---|---|---|
| 0 | `s0_baseline_640` — YOLOv8s, 640 | Baseline |
| 1 | `s1_base_960`, `s1_base_1280` | Dampak resolusi |
| 2 | `s2_p2_640/960/1280` | Dampak head P2 (dan interaksinya dengan resolusi) |
| 3 | `s3_piou`, `s3_piou2`, `s3_nwd`, `s3_ciou_nwd`, `s3_piou2_nwd` | Dampak loss |
| 4 | `s4_p2gfpn` | GFPN-lite, hanya jika latensi masih aman |
| 5 | `s5_simam`, `s5_ca`, `s5_eca`, `s5_ema` | Atensi (opsional, urutan prioritas: SimAM → CA → ECA → EMA) |
| 6 | `s6_full`, `s6_full_nwd` | Kombinasi penuh semua komponen |

**Tahap 3–6 memakai blok `best:` di `cfg/experiments.yaml`** (resolusi, arsitektur, loss, atensi terbaik).
Isi ulang blok itu setelah melihat hasil tahap sebelumnya; nilai awalnya hanya contoh.

Kombinasi bebas (tanpa mengedit YAML):
```bash
python scripts/run_experiment.py --data data/helmet.yaml \
  --grid arch=base,p2 imgsz=640,960 loss=ciou,piou2,nwd attn=none,simam --seeds 0 1 2
# loss: ciou | piou | piou2 | nwd | ciou+nwd | piou2+nwd     attn: none | simam | ca | eca | ema
```

## SAHI dan evaluasi per-ukuran

```bash
# inferensi biasa
python scripts/evaluate.py --weights runs/s2_p2_960/seed0/weights/best.pt --data data/helmet.yaml --imgsz 960 --device 0
# SAHI: irisan 640 px, overlap 20%
python scripts/evaluate.py --weights runs/s0_baseline_640/seed0/weights/best.pt --data data/helmet.yaml \
  --sahi --slice 640 --overlap 0.2 --imgsz 640 --device 0
```
Kedua mode memakai perhitungan AP yang sama, jadi angkanya bisa dibandingkan. Keluaran juga memuat
**recall per kelompok ukuran** (tiny/small/medium/large, dalam piksel gambar asli; atur `--size-bins`).
SAHI hanya dipakai saat inferensi; modelnya sama dengan yang dilatih biasa.

## Latensi (ukur di perangkat target)

```bash
python scripts/benchmark_latency.py --weights runs/s2_p2_960/seed0/weights/best.pt --imgsz 960 --device cpu
python scripts/benchmark_latency.py --cfg cfg/models/yolov8-p2-gfpn.yaml --scale s --imgsz 960 --device 0 --half
```
Melaporkan forward-saja dan end-to-end (mean/p50/p95 ms, FPS), parameter, dan GFLOPs. Untuk Jetson, export ke
TensorRT lalu ukur ulang di sana; angka PyTorch hanya untuk membandingkan konfigurasi secara relatif.

## Yang perlu Anda ketahui (batasan & keputusan desain)

- **Komponen berikut adalah pilihan implementasi saya, bukan salinan paper:**
  - *GFPN-lite*: node neck menggabungkan fitur dari tiga skala (atas + backbone + bawah) dengan blok C2f. Ini bukan
    GFPN DAMO-YOLO penuh (tanpa CSPStage / log2n-link). Tulis di laporan sebagai "neck bergaya GFPN".
  - *Atensi* ditempatkan **setelah `cv2` di setiap blok C2f pada head** (seragam untuk keempat jenis atensi agar adil).
    Repo asli menyisipkan EMA di bottleneck kedua; itu berbeda.
- **PIoU**: rumus identik dengan repo asli (diuji, selisih ≈1e-7). Nilai `piou_lambda=1.2` mengikuti repo; belum saya
  verifikasi terhadap paper, jadi anggap sebagai hiperparameter untuk di-tuning.
- **NWD**: konstanta `nwd_c=12.8` adalah nilai dari paper NWD untuk dataset berbeda. Hitungannya dalam **piksel input
  training**, jadi nilai idealnya bergantung dataset dan resolusi. Pertimbangkan mencoba 6–16.
- Penugasan label (TAL assigner) tetap CIoU bawaan; hanya loss regresi kotak yang diganti.
- **Transfer bobot**: `yolov8s.pt` hanya mengisi lapisan yang bentuknya sama. Varian P2/GFPN/atensi sebagian dilatih
  dari nol, jadi bandingkan dengan mengingat hal ini (atau latih semua dengan `--pretrained none`).
- Beda 0,3–0,5% mAP biasanya masih dalam noise antar seed. Pakai `--seeds 0 1 2` dan laporkan mean ± std
  (`summarize_results.py`).
- Loss custom hanya berlaku pada training 1 GPU/CPU. Training multi-GPU (DDP) memulai proses baru tanpa patch.
- Lisensi: Ultralytics AGPL-3.0; kode asal SOD-YOLOv8 oleh Khalili & Smyth (Sensors 2024). Sitasi paper tersebut,
  Ultralytics YOLOv8, SAHI, NWD, PIoU, dan paper atensi yang Anda pakai.

## Status verifikasi (yang benar-benar sudah dijalankan)

- 15 kombinasi arsitektur × atensi: model terbangun dan forward-pass benar (CPU).
- PIoU/PIoU v2 cocok dengan repo asli; NWD dan semua campuran loss: gradien valid.
- Training 1 siklus penuh di CPU (GFPN + EMA + PIoU2 + NWD, data sintetis): berjalan sampai selesai, `best.pt`
  tersimpan dan bisa dimuat ulang.
- Training 45 epoch di data sintetis (YOLOv8n + SimAM + PIoU2/NWD): model belajar mendeteksi. `evaluate.py` mode biasa
  cocok dengan validator Ultralytics pada bobot yang sama (mAP50 0,9427 vs 0,9425; mAP50-95 0,7498 vs 0,751).
  Mode SAHI dan `benchmark_latency.py` juga berjalan. Itu data mainan: angkanya tidak punya arti untuk helm.
- **Belum** dijalankan: training pada dataset helm asli dan GPU (sandbox hanya CPU), jadi **tidak ada angka akurasi**
  dalam proyek ini; semua hasil harus Anda ukur sendiri.
