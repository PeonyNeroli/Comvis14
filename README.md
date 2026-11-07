# UTS Computer Vision 🔎

Pengolahan citra motif batik Indonesia: grayscale dan enhancement dievaluasi dengan MSE dan PSNR, lalu fitur SIFT dan LBP dari enhancement terbaik diklasifikasi dengan CNN dan MobileNetV2.

## Kelompok 14

| NIM | Nama |
| --- | --- |
| 187221097 | Oksaviyana Alfarezi |
| 187231077 | Raditya Nauval Ramadhan Putra Wibowo |
| 187231088 | Dimas Akbar Hataputra |
| 187231075 | Dzaky Muttaqi Sunaryadi |
| 187231093 | Amir Gymnastiar |

## Alur program

**A. Enhancement dan evaluasi**

1. **Input:** citra dibaca per folder kelas, lalu di-resize ke 256×256 piksel.
2. **Grayscale:** citra warna diubah menjadi satu kanal keabuan. Citra ini menjadi acuan evaluasi.
3. **Enhancement:** tiga metode dijalankan terpisah pada citra grayscale yang sama:
   - Histogram Equalization (HE)
   - Contrast Stretching (persentil 2 dan 98 dipetakan ke 0 dan 255)
   - CLAHE (`clipLimit` 2.0, blok 8×8)
4. **Tanpa filter:** median dan Gaussian 3×3 masih tersedia di program, tetapi tidak dipakai (`FILTER = None`), karena keduanya membuat citra lebih buruk (MSE naik).
5. **Evaluasi:** MSE dan PSNR dihitung per citra antara grayscale dan hasil enhancement, lalu dirata-ratakan per metode dan per kelas.
6. **Pemilihan:** enhancement dengan MSE rata-rata terendah dipakai untuk tahap berikutnya. Hasilnya **CLAHE**.

**B. Ekstraksi fitur** (dari citra CLAHE)

Kedua fitur dibentuk sebagai peta 3 kanal berukuran 128×128 agar bisa masuk ke CNN dan MobileNetV2.

| Metode | Cara membentuk peta fitur |
| --- | --- |
| Metode 1: SIFT | Deskriptor dense SIFT (RootSIFT, 128 angka) di grid 32×32 titik, diringkas ke 3 komponen PCA. PCA dibangun dari citra latih saja. |
| Metode 2: LBP | Peta LBP uniform dari 3 skala (P, R) = (8, 1), (16, 2), (24, 3), satu skala per kanal. |

**C. Klasifikasi** (4 output)

| Output | Fitur | Model |
| --- | --- | --- |
| 1 | SIFT | CNN kecil, dilatih dari nol |
| 2 | SIFT | MobileNetV2 pretrained ImageNet, di-fine-tune |
| 3 | LBP | CNN kecil, dilatih dari nol |
| 4 | LBP | MobileNetV2 pretrained ImageNet, di-fine-tune |

Data dibagi 80/20 (191 latih, 48 uji) dengan proporsi kelas yang sama. Tiap model dilatih 20 epoch dengan Adam dan augmentasi flip horizontal.

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

3. Jalankan program (sekitar 3–4 menit di CPU):

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
| `LBP_SCALES` | (8,1), (16,2), (24,3) | Skala LBP, satu per kanal |
| `SIFT_STEP`, `SIFT_SIZE` | 8, 4 | Jarak grid dan ukuran titik dense SIFT |
| `DL_SIZE` | 128 | Ukuran peta fitur yang masuk ke model |
| `TEST_SIZE` | 0.2 | Porsi data uji |
| `EPOCHS`, `BATCH` | 20, 16 | Lama pelatihan dan ukuran batch |
| `LR` | CNN 1e-3, MobileNetV2 1e-4 | Learning rate Adam |
| `SEED` | 0 | Seed acak agar hasil bisa diulang |

## Hasil

**Rata-rata MSE dan PSNR** (239 citra, tanpa filter):

| Enhancement | Rata-rata MSE | Rata-rata PSNR (dB) |
| --- | --- | --- |
| HE | 1951.29 | 17.00 |
| Contrast Stretching | 613.37 | 24.21* |
| **CLAHE** | **561.38** | 21.90 |

\* 4 citra tidak berubah sama sekali (MSE = 0, PSNR tak hingga) dan dikeluarkan dari rata-rata PSNR.

**Klasifikasi** (48 citra uji; tebakan acak = 0.20):

| Output | Fitur + model | Akurasi | Macro F1 |
| --- | --- | --- | --- |
| 1 | SIFT + CNN | 0.583 | 0.584 |
| 2 | SIFT + MobileNetV2 | **0.625** | **0.625** |
| 3 | LBP + CNN | 0.479 | 0.434 |
| 4 | LBP + MobileNetV2 | 0.562 | 0.561 |

**File keluaran** di `output/`:

| File | Isi |
| --- | --- |
| `A_<kelas>.png` | Input, grayscale, tiga enhancement beserta MSE/PSNR, dan histogram (satu contoh per kelas) |
| `B_<kelas>.png` | Citra CLAHE, peta SIFT, peta LBP, dan tiap kanal LBP untuk contoh yang sama |
| `mse_psnr.csv` | MSE dan PSNR tiap citra untuk tiap enhancement |
| `fitur.npz` | Peta fitur SIFT dan LBP semua citra, label kelas, dan indeks data latih/uji |

Terminal juga mencetak loss latih tiap 5 epoch, recall per kelas, dan confusion matrix untuk keempat output.

## Isi repo

| Path | Isi |
| --- | --- |
| `uts.py` | Program utama |
| `requirements.txt` | Daftar pustaka |
| `UTS COMVIS_14.pdf` | Laporan UTS (versi sebelumnya) |
| `UTS_14.zip` | Berkas pengumpulan (versi sebelumnya) |
| `output (median)/` | Hasil versi lama yang masih memakai median filter |
