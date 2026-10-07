"""UTS Computer Vision
A: input -> grayscale -> enhancement (HE / Contrast Stretching / CLAHE) -> filter -> MSE & PSNR
B: ekstraksi fitur dengan 2 metode: HOG dan LBP
"""
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA, OUT = Path("dataset"), Path("output")
SIZE = 256            # semua citra di-resize ke SIZE x SIZE
FILTER = "median"     # "median" atau "gaussian"
KSIZE = 3             # ukuran kernel filter (ganjil)
LBP_P, LBP_R = 8, 1   # jumlah tetangga dan radius LBP


def contrast_stretch(g, lo_pct=2, hi_pct=98):
    lo, hi = np.percentile(g, (lo_pct, hi_pct))
    if hi == lo:
        return g.copy()
    return np.clip((g.astype(np.float32) - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)


ENHANCE = {
    "HE": cv2.equalizeHist,
    "Contrast Stretching": contrast_stretch,
    "CLAHE": cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply,
}


def smooth(img):
    return cv2.medianBlur(img, KSIZE) if FILTER == "median" else cv2.GaussianBlur(img, (KSIZE, KSIZE), 0)


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a, b):
    m = mse(a, b)
    return float("inf") if m == 0 else 10 * np.log10(255 ** 2 / m)


def hog_feat(img, visualize=False):
    return hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=visualize)


def lbp_map(img):
    return local_binary_pattern(img, LBP_P, LBP_R, method="uniform")


def lbp_feat(img):
    # histogram pola uniform: LBP_P + 2 bin
    return np.histogram(lbp_map(img), bins=LBP_P + 2, range=(0, LBP_P + 2), density=True)[0]


def save_figures(name, bgr, gray, results):
    """results: {metode: (enhanced, filtered)}"""
    fig, ax = plt.subplots(3, 4, figsize=(16, 12))
    ax[0, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); ax[0, 0].set_title("Input")
    ax[1, 0].imshow(gray, cmap="gray"); ax[1, 0].set_title("Grayscale")
    ax[2, 0].hist(gray.ravel(), 256, range=(0, 256)); ax[2, 0].set_title("Histogram Grayscale")
    for j, (m, (enh, flt)) in enumerate(results.items(), 1):
        ax[0, j].imshow(enh, cmap="gray"); ax[0, j].set_title(m)
        ax[1, j].imshow(flt, cmap="gray")
        ax[1, j].set_title(f"{m} + {FILTER}\nMSE={mse(gray, flt):.1f}  PSNR={psnr(gray, flt):.2f} dB")
        ax[2, j].hist(enh.ravel(), 256, range=(0, 256)); ax[2, j].set_title(f"Histogram {m}")
    for a in ax[:2].ravel():
        a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / f"A_{name}.png", dpi=100); plt.close(fig)

    fig, ax = plt.subplots(3, 4, figsize=(16, 12))
    for i, (m, (_, flt)) in enumerate(results.items()):
        ax[i, 0].imshow(flt, cmap="gray"); ax[i, 0].set_title(f"{m} + {FILTER}")
        h = hog_feat(flt, visualize=True)[1]
        ax[i, 1].imshow(h, cmap="gray", vmax=h.max() * 0.3); ax[i, 1].set_title("HOG")
        ax[i, 2].imshow(lbp_map(flt), cmap="gray"); ax[i, 2].set_title("LBP")
        ax[i, 3].bar(range(LBP_P + 2), lbp_feat(flt)); ax[i, 3].set_title("Histogram LBP")
        for a in ax[i, :3]:
            a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / f"B_{name}.png", dpi=100); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    rows, labels = [], []
    feats = {(m, f): [] for m in ENHANCE for f in ("HOG", "LBP")}

    for cls in sorted(p for p in DATA.iterdir() if p.is_dir()):
        for k, path in enumerate(sorted(cls.glob("*.jpg"))):
            bgr = cv2.imread(str(path))
            if bgr is None:
                print("skip (tidak terbaca):", path)
                continue
            bgr = cv2.resize(bgr, (SIZE, SIZE))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            results = {}
            for m, fn in ENHANCE.items():
                enh = fn(gray)
                flt = smooth(enh)
                results[m] = (enh, flt)
                rows.append([cls.name, path.name, m, mse(gray, flt), psnr(gray, flt)])
                feats[m, "HOG"].append(hog_feat(flt))
                feats[m, "LBP"].append(lbp_feat(flt))
            labels.append(cls.name)
            if k == 0:
                save_figures(cls.name, bgr, gray, results)

    # A: evaluasi MSE & PSNR per citra + rata-rata
    with open(OUT / "mse_psnr.csv", "w", newline="") as f:
        csv.writer(f).writerows([["kelas", "file", "metode", "mse", "psnr"]] + rows)
    print(f"\n{len(labels)} citra, filter = {FILTER} {KSIZE}x{KSIZE}")
    print(f"\n{'Metode':<22}{'Rata-rata MSE':>15}{'Rata-rata PSNR (dB)':>22}")
    for m in ENHANCE:
        r = [x for x in rows if x[2] == m]
        print(f"{m:<22}{np.mean([x[3] for x in r]):>15.2f}{np.mean([x[4] for x in r]):>22.2f}")

    # B: simpan fitur + bandingkan kedua metode lewat akurasi SVM (5-fold CV)
    np.savez_compressed(OUT / "fitur.npz", label=labels,
                        **{f"{f}_{m.replace(' ', '_')}": np.array(v) for (m, f), v in feats.items()})
    print(f"\n{'Metode':<22}{'Fitur':<7}{'Dimensi':>9}{'Akurasi SVM':>14}")
    for (m, f), v in feats.items():
        X = np.array(v)
        acc = cross_val_score(make_pipeline(StandardScaler(), SVC()), X, labels, cv=5).mean()
        print(f"{m:<22}{f:<7}{X.shape[1]:>9}{acc:>14.3f}")
    print(f"\nGambar, mse_psnr.csv, dan fitur.npz tersimpan di {OUT}/")


def selfcheck():
    a, b = np.zeros((4, 4), np.uint8), np.full((4, 4), 255, np.uint8)
    assert mse(a, a) == 0 and psnr(a, a) == float("inf")
    assert mse(a, b) == 255 ** 2 and abs(psnr(a, b)) < 1e-9
    r = np.random.default_rng(0).integers(0, 256, (32, 32), dtype=np.uint8)
    assert abs(psnr(r, smooth(r)) - cv2.PSNR(r, smooth(r))) < 1e-6
    assert contrast_stretch(np.arange(100, 164, dtype=np.uint8).reshape(8, 8)).max() == 255
    assert abs(lbp_feat(r).sum() - 1) < 1e-9


if __name__ == "__main__":
    selfcheck()
    main()
