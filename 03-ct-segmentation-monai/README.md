# 03 — Segmentasi Organ pada CT 3D (rencana)

**Tempat jalan:** Google Colab (GPU). Tidak ada data CT yang diunduh ke laptop.

## Rencana

1. Ikuti tutorial MONAI tentang segmentasi 3D di Colab terlebih dahulu.
2. Ambil subset kecil TotalSegmentator (puluhan volume, bukan seluruh dataset) dan simpan di Google Drive.
3. Pilih 1–3 organ saja (mis. hati, limpa, ginjal) supaya pelatihan muat di GPU gratis.
4. Latih UNet 3D dengan patch (crop) untuk menghemat memori GPU.
5. Evaluasi dengan Dice per organ, tampilkan potongan slice prediksi vs ground truth.
6. Simpan checkpoint di Google Drive, karena sesi Colab bisa terputus.

## Yang disimpan di folder ini

- Notebook Colab (dengan output)
- `results/` berisi tabel Dice dan beberapa gambar slice
- README dengan hasil dan batasan (subset kecil, organ terbatas)
