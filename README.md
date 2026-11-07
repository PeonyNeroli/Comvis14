# UTS Computer Vision 🔎

Pengolahan citra motif batik Indonesia: grayscale, enhancement, evaluasi MSE dan PSNR, lalu ekstraksi fitur dengan LBP multi-skala, SIFT (Bag of Visual Words), dan MobileNetV2.

## Kelompok 14

| NIM | Nama |
| --- | --- |
| 187221097 | Oksaviyana Alfarezi |
| 187231077 | Raditya Nauval Ramadhan Putra Wibowo |
| 187231088 | Dimas Akbar Hataputra |
| 187231075 | Dzaky Muttaqi Sunaryadi |
| 187231093 | Amir Gymnastiar |

## Alur program

**Bagian A: pengolahan citra dan evaluasi**

1. **Input:** citra dibaca per folder kelas, lalu di-resize ke 256×256 piksel.
2. **Grayscale:** citra warna diubah menjadi satu kanal keabuan. Citra ini menjadi acuan evaluasi.
3. **Enhancement:** tiga metode dijalankan terpisah pada citra grayscale yang sama:
   - Histogram Equalization (HE)
   - Contrast Stretching (persentil 2 dan 98 dipetakan ke 0 dan 255)
   - CLAHE (`clipLimit` 2.0, blok 8×8)
4. **Filter:** median atau Gaussian 3×3 tersedia, tetapi **tidak dipakai** pada hasil akhir (`FILTER = None`). Kedua filter sudah dicoba dan sama-sama menaikkan MSE terhadap citra grayscale.
5. **Evaluasi:** MSE dan PSNR dihitung per citra antara grayscale dan hasil enhancement, lalu dirata-ratakan per metode dan per kelas.

**Bagian B: ekstraksi fitur** (dari tiap hasil enhancement)

| Metode | Jumlah fitur | Yang ditangkap |
| --- | --- | --- |
| LBP multi-skala | 80 | Tekstur lokal, gabungan histogram pola uniform dari 4 skala (P, R) = (8, 1), (16, 2), (24, 3), (24, 5) |
| SIFT + Bag of Visual Words | 256 | Bentuk lokal, dense SIFT tiap 8 piksel, kamus 256 visual word dari k-means |
| MobileNetV2 | 1280 | Pola hasil belajar dari ImageNet (bobot pretrained, lapisan klasifikasi dibuang, tanpa pelatihan ulang) |

Program tidak melakukan klasifikasi. Fitur disimpan ke `output/fitur.npz` dan divisualisasikan per kelas.

## Dataset

[Indonesian Batik Motifs](https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs) dari Kaggle (20 kelas, 983 file). Dataset tidak disertakan di repo.

Program memakai 5 kelas yang dinamai menurut motif: ceplok, kawung, megamendung, parang, dan tambal (239 citra). Kelas lain banyak dinamai menurut daerah, sehingga satu kelas berisi banyak motif berbeda. File `batik-megamendung/49.jpg` bukan citra dan dilewati otomatis.

## Cara menjalankan

1. Unduh dataset dan ekstrak ke folder `dataset/`, satu subfolder per kelas:

   ```
   dataset/
   ├── batik-ceplok/
   ├── batik-kawung/
   ├── batik-megamendung/
   ├── batik-parang/
   └── batik-tambal/
   ```

2. Buat virtual environment dan pasang pustaka:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```

   `--extra-index-url` memasang PyTorch versi CPU yang jauh lebih kecil. Saat pertama dijalankan, bobot MobileNetV2 (sekitar 14 MB) diunduh otomatis.

3. Jalankan program:

   ```bash
   python uts.py
   ```

   Kalau memakai VS Code, pilih interpreter `.venv` lewat **Python: Select Interpreter**. Jika tidak, akan muncul error `No module named 'cv2'`.

## Pengaturan

Semua pengaturan ada di konstanta bagian atas `uts.py`.

| Konstanta | Nilai | Keterangan |
| --- | --- | --- |
| `CLASSES` | 5 kelas motif | Daftar subfolder yang dipakai; `None` = semua kelas |
| `SIZE` | 256 | Ukuran resize citra |
| `FILTER` | `None` | `None`, `"median"`, atau `"gaussian"` |
| `KSIZE` | 3 | Ukuran kernel filter |
| `LBP_SCALES` | (8,1), (16,2), (24,3), (24,5) | Skala LBP |
| `SIFT_STEP`, `SIFT_SIZES` | 8, (8, 16) | Jarak grid dan ukuran titik dense SIFT |
| `VOCAB` | 256 | Jumlah visual word |
| `KMEANS_SAMPLE` | 300 | Deskriptor acak per citra untuk k-means |

## Hasil

Program mencetak tabel ke terminal dan menyimpan hasil ke `output/`:

| File | Isi |
| --- | --- |
| `A_<kelas>.png` | Input, grayscale, tiga enhancement beserta MSE/PSNR, dan histogram (satu contoh per kelas) |
| `B_<kelas>.png` | Peta LBP, histogram LBP multi-skala, histogram visual word SIFT, dan fitur MobileNetV2 |
| `mse_psnr.csv` | MSE dan PSNR tiap citra untuk tiap enhancement |
| `fitur.npz` | Vektor fitur LBP, SIFT, dan MobileNetV2 semua citra, beserta label kelas |

Rata-rata MSE dan PSNR (239 citra, tanpa filter):

| Enhancement | Rata-rata MSE | Rata-rata PSNR (dB) |
| --- | --- | --- |
| HE | 1951.29 | 17.00 |
| Contrast Stretching | 613.37 | 24.21* |
| CLAHE | 561.38 | 21.90 |

\* 4 citra tidak berubah sama sekali (MSE = 0, PSNR tak hingga) dan dikeluarkan dari rata-rata PSNR.

## Isi repo

| Path | Isi |
| --- | --- |
| `uts.py` | Program utama |
| `requirements.txt` | Daftar pustaka |
| `UTS COMVIS_14.pdf` | Laporan UTS |
| `UTS_14.zip` | Berkas pengumpulan |
| `output (median)/` | Hasil versi sebelumnya yang masih memakai median filter |
