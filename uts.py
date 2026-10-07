"""UTS Computer Vision
A: input -> grayscale -> enhancement (HE / Contrast Stretching / CLAHE) -> filter -> MSE & PSNR
B: ekstraksi fitur dengan 2 metode: LBP multi-skala dan SIFT (Bag of Visual Words)
"""
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from skimage.feature import local_binary_pattern
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA, OUT = Path("dataset"), Path("output")
# kelas yang dipakai (nama subfolder di DATA); None = semua subfolder
CLASSES = ["batik-ceplok", "batik-kawung", "batik-megamendung", "batik-parang", "batik-tambal"]
SIZE = 256            # semua citra di-resize ke SIZE x SIZE
FILTER = "median"     # "median" atau "gaussian"
KSIZE = 3             # ukuran kernel filter (ganjil)
LBP_SCALES = ((8, 1), (16, 2), (24, 3), (24, 5))   # (jumlah tetangga, radius) tiap skala LBP
SIFT_STEP, SIFT_SIZES = 8, (8, 16)   # jarak grid dan ukuran keypoint dense SIFT
VOCAB = 256           # jumlah visual word (cluster k-means)
TEST_SIZE = 0.2       # porsi data uji
SVM_GRID = [          # kandidat parameter SVM, dipilih lewat 5-fold CV pada data latih
    {"svc__kernel": ["rbf"], "svc__C": [0.1, 1, 10, 100, 1000], "svc__gamma": ["scale", 1e-4, 1e-3, 1e-2, 1e-1]},
    {"svc__kernel": ["linear"], "svc__C": [0.01, 0.1, 1, 10]},
]


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


def lbp_feat(img):
    # gabungan histogram pola uniform (P + 2 bin) dari tiap skala
    return np.concatenate([
        np.histogram(local_binary_pattern(img, p, r, method="uniform"), bins=p + 2, range=(0, p + 2), density=True)[0]
        for p, r in LBP_SCALES])


def sift_desc(img):
    # dense SIFT: deskriptor dihitung pada grid teratur, bukan pada keypoint terdeteksi (lebih cocok untuk tekstur)
    kp = [cv2.KeyPoint(float(x), float(y), float(s)) for s in SIFT_SIZES
          for y in range(s, img.shape[0] - s, SIFT_STEP) for x in range(s, img.shape[1] - s, SIFT_STEP)]
    d = cv2.SIFT_create().compute(img, kp)[1]
    return np.sqrt(d / (d.sum(1, keepdims=True) + 1e-7))  # RootSIFT


def bovw_feats(imgs, train_idx):
    """Histogram visual word per citra. Kamus (k-means) dibangun hanya dari citra latih."""
    desc = Parallel(n_jobs=-1)(delayed(sift_desc)(im) for im in imgs)
    rng = np.random.default_rng(0)
    sample = np.vstack([desc[i][rng.choice(len(desc[i]), 300, replace=False)] for i in train_idx])
    km = MiniBatchKMeans(VOCAB, random_state=0, n_init=3, batch_size=4096).fit(sample)
    h = np.array([np.bincount(km.predict(d), minlength=VOCAB) for d in desc], dtype=np.float64)
    return np.sqrt(h / h.sum(1, keepdims=True))


def save_fig_a(name, bgr, gray, results):
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


def save_fig_b(name, i, imgs, feats):
    fig, ax = plt.subplots(3, 4, figsize=(18, 12))
    for r, m in enumerate(ENHANCE):
        img = imgs[m][i]
        ax[r, 0].imshow(img, cmap="gray"); ax[r, 0].set_title(f"{m} + {FILTER}")
        p, rad = LBP_SCALES[0]
        ax[r, 1].imshow(local_binary_pattern(img, p, rad, method="uniform"), cmap="gray")
        ax[r, 1].set_title(f"LBP (P={p}, R={rad})")
        ax[r, 2].bar(range(len(feats[m, "LBP"][i])), feats[m, "LBP"][i]); ax[r, 2].set_title("Histogram LBP multi-skala")
        ax[r, 3].bar(range(VOCAB), feats[m, "SIFT"][i], width=1); ax[r, 3].set_title("Histogram visual word SIFT")
        ax[r, 0].axis("off"); ax[r, 1].axis("off")
    fig.tight_layout(); fig.savefig(OUT / f"B_{name}.png", dpi=100); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    rows, labels, first = [], [], {}
    imgs = {m: [] for m in ENHANCE}

    for cls in sorted(p for p in DATA.iterdir() if p.is_dir() and (CLASSES is None or p.name in CLASSES)):
        for path in sorted(cls.glob("*.jpg")):
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
                imgs[m].append(flt)
                rows.append([cls.name, path.name, m, mse(gray, flt), psnr(gray, flt)])
            if cls.name not in first:
                first[cls.name] = len(labels)
                save_fig_a(cls.name, bgr, gray, results)
            labels.append(cls.name)

    # A: evaluasi MSE & PSNR per citra + rata-rata
    with open(OUT / "mse_psnr.csv", "w", newline="") as f:
        csv.writer(f).writerows([["kelas", "file", "metode", "mse", "psnr"]] + rows)
    print(f"\n{len(labels)} citra, {len(first)} kelas, filter = {FILTER} {KSIZE}x{KSIZE}")
    print(f"\n{'Metode':<22}{'Rata-rata MSE':>15}{'Rata-rata PSNR (dB)':>22}")
    for m in ENHANCE:
        r = [x for x in rows if x[2] == m]
        print(f"{m:<22}{np.mean([x[3] for x in r]):>15.2f}{np.mean([x[4] for x in r]):>22.2f}")

    # B: ekstraksi fitur, lalu bandingkan kedua metode lewat akurasi SVM.
    # Data dibagi latih/uji; kamus SIFT dan parameter SVM hanya memakai data latih, akurasi akhir dari data uji.
    labels = np.array(labels)
    tr, te = train_test_split(np.arange(len(labels)), test_size=TEST_SIZE, stratify=labels, random_state=42)
    feats = {}
    for m in ENHANCE:
        feats[m, "LBP"] = np.array(Parallel(n_jobs=-1)(delayed(lbp_feat)(im) for im in imgs[m]))
        feats[m, "SIFT"] = bovw_feats(imgs[m], tr)
    np.savez_compressed(OUT / "fitur.npz", label=labels, train_idx=tr, test_idx=te,
                        **{f"{f}_{m.replace(' ', '_')}": v for (m, f), v in feats.items()})
    for name, i in first.items():
        save_fig_b(name, i, imgs, feats)

    print(f"\n{len(tr)} citra latih, {len(te)} citra uji")
    print(f"\n{'Metode':<22}{'Fitur':<7}{'Dimensi':>9}{'Akurasi CV':>12}{'Akurasi Uji':>13}  Parameter SVM terbaik")
    for (m, f), X in feats.items():
        g = GridSearchCV(make_pipeline(StandardScaler(), SVC()), SVM_GRID, cv=5, n_jobs=-1).fit(X[tr], labels[tr])
        best = ", ".join(f"{k[5:]}={val}" for k, val in g.best_params_.items())
        print(f"{m:<22}{f:<7}{X.shape[1]:>9}{g.best_score_:>12.3f}{g.score(X[te], labels[te]):>13.3f}  {best}")
    print(f"\nGambar, mse_psnr.csv, dan fitur.npz tersimpan di {OUT}/")


def selfcheck():
    a, b = np.zeros((4, 4), np.uint8), np.full((4, 4), 255, np.uint8)
    assert mse(a, a) == 0 and psnr(a, a) == float("inf")
    assert mse(a, b) == 255 ** 2 and abs(psnr(a, b)) < 1e-9
    r = np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8)
    assert abs(psnr(r, smooth(r)) - cv2.PSNR(r, smooth(r))) < 1e-6
    assert contrast_stretch(np.arange(100, 164, dtype=np.uint8).reshape(8, 8)).max() == 255
    f = lbp_feat(r)
    assert len(f) == sum(p + 2 for p, _ in LBP_SCALES) and abs(f.sum() - len(LBP_SCALES)) < 1e-9
    d = sift_desc(r)
    assert d.shape[1] == 128 and np.allclose((d ** 2).sum(1), 1, atol=1e-3)


if __name__ == "__main__":
    selfcheck()
    main()
