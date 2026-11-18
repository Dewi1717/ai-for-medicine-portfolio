# 02 — Klasifikasi X-ray Dada (rencana)

**Tempat jalan:** pelatihan di Kaggle Notebook (GPU), analisis hasil di laptop.

## Rencana

1. Buat notebook di Kaggle, tambahkan dataset X-ray dada langsung dari Kaggle (jangan diunduh ke laptop).
   Cek dulu ketersediaan dataset dan kuota GPU mingguan Kaggle saat mulai.
2. Transfer learning dari model pra-latih kecil (mis. ResNet18 / EfficientNet-B0) dengan PyTorch.
3. Split per pasien jika metadata pasien tersedia, supaya gambar pasien yang sama tidak muncul di train dan test.
4. Evaluasi: sensitivitas, spesifisitas, ROC-AUC, confusion matrix.
5. Grad-CAM pada beberapa contoh benar dan salah, untuk melihat apakah model melihat paru atau artefak (label, tulisan, alat).
6. Unduh ke laptop hanya `predictions.csv`, metrik, dan gambar Grad-CAM untuk ditulis di README ini.

## Yang disimpan di folder ini

- `notebook.ipynb` (salinan dari Kaggle, dengan output)
- `results/` berisi metrik dan gambar kecil
- README dengan hasil, contoh Grad-CAM, dan batasan
