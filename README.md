# Portofolio AI for Medicine

Kumpulan proyek machine learning untuk data medis: sinyal EEG, citra X-ray, dan CT 3D.
Setiap proyek dirancang sesuai sumber daya yang tersedia, dan keputusan itu didokumentasikan.

## Proyek

| # | Proyek | Data | Metode | Tempat jalan | Status |
|---|--------|------|--------|--------------|--------|
| 01 | [Deteksi kejang dari EEG](01-eeg-seizure-detection/) | CHB-MIT (subset 1 pasien) | Ekstraksi fitur + Random Forest | Laptop (CPU) | Selesai v1: 7/7 kejang terdeteksi, ~0,5 alarm palsu/jam |
| 02 | [Klasifikasi X-ray dada](02-chest-xray-classification/) | Dataset X-ray di Kaggle | CNN transfer learning + Grad-CAM | Kaggle GPU, analisis di laptop | Rencana |
| 03 | [Segmentasi organ CT](03-ct-segmentation-monai/) | Subset kecil TotalSegmentator | UNet 3D (MONAI) | Google Colab | Rencana |

## Strategi komputasi

Laptop yang dipakai: Core i3-1215U, RAM 8 GB, GPU Intel UHD (tanpa CUDA), sisa disk sekitar 138 GB.
Karena itu:

- **Laptop** dipakai untuk ML klasik (scikit-learn), pemrosesan data kecil, analisis hasil, dan menulis kode.
- **Kaggle / Colab** dipakai untuk semua pelatihan deep learning. Dataset dibaca langsung di platform, tidak diunduh ke laptop.
- **Tidak ada dataset penuh di disk lokal.** Hanya subset yang dibutuhkan, dan folder `data/` tidak masuk git.

Batasan ini ikut ditulis di setiap proyek, karena memilih metode yang sesuai sumber daya adalah bagian dari pekerjaan.

## Yang ditunjukkan portofolio ini

- Pemisahan data yang benar untuk data medis (split per rekaman/pasien, bukan per sampel acak) supaya tidak terjadi *data leakage*.
- Metrik yang relevan secara klinis: sensitivitas, spesifisitas, dan PR-AUC untuk kelas yang tidak seimbang, bukan hanya akurasi.
- Interpretabilitas: feature importance (proyek 01), Grad-CAM (proyek 02).
- Batasan dan risiko setiap model ditulis terang-terangan.

## Disclaimer

Semua proyek ini untuk tujuan belajar dan riset. Tidak ada model di sini yang divalidasi untuk penggunaan klinis.
