# UTS Computer Vision 🔎

Pipeline pengolahan citra pada dataset motif batik:

- **A:** input -> grayscale -> enhancement (Histogram Equalization, Contrast Stretching, CLAHE) -> filter (median / gaussian) -> evaluasi MSE & PSNR
- **B:** ekstraksi fitur dengan 2 metode, **HOG** dan **LBP**, dibandingkan lewat akurasi SVM (5-fold CV)

## Menjalankan

```bash
pip install opencv-python scikit-image scikit-learn matplotlib
python uts.py
```

Dataset tidak disertakan di repo. Unduh "Indonesian Batik Motifs" dari Kaggle https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs dan ekstrak ke folder `dataset/` (satu subfolder per kelas). Hasil (gambar, `mse_psnr.csv`, `fitur.npz`) tersimpan di `output/`.

Filter dan parameter lain diatur lewat konstanta di bagian atas `uts.py`.
