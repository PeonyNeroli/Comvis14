# UTS Computer Vision 🔎

Pipeline pengolahan citra pada dataset motif batik:

- **A:** input -> grayscale -> enhancement (Histogram Equalization, Contrast Stretching, CLAHE) -> filter (median / gaussian) -> evaluasi MSE & PSNR
- **B:** ekstraksi fitur dengan 2 metode, **LBP multi-skala** dan **SIFT (Bag of Visual Words)**, dibandingkan lewat akurasi SVM (data latih/uji 80/20, parameter SVM di-tuning dengan 5-fold CV pada data latih)

## Menjalankan

```bash
pip install opencv-python scikit-image scikit-learn matplotlib
python uts.py
```

Dataset tidak disertakan di repo. Unduh "Indonesian Batik Motifs" dari Kaggle https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs dan ekstrak ke folder `dataset/` (satu subfolder per kelas). Secara default dipakai 5 kelas motif (ceplok, kawung, megamendung, parang, tambal); ubah `CLASSES` di `uts.py` untuk kelas lain, atau `None` untuk semua. Hasil (gambar, `mse_psnr.csv`, `fitur.npz`) tersimpan di `output/`.

Filter dan parameter lain diatur lewat konstanta di bagian atas `uts.py`.
