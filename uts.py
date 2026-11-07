"""UTS Computer Vision
A: input -> grayscale -> enhancement (HE / Contrast Stretching / CLAHE) -> MSE & PSNR
   -> pilih enhancement terbaik (MSE rata-rata terendah)
B: ekstraksi fitur dari enhancement terbaik: SIFT (metode 1) dan LBP (metode 2), masing-masing berupa peta fitur 3 kanal
C: klasifikasi tiap peta fitur dengan CNN (dilatih dari nol) dan MobileNetV2 (pretrained ImageNet, fine-tuning)
   -> Output 1: SIFT + CNN, Output 2: SIFT + MobileNetV2, Output 3: LBP + CNN, Output 4: LBP + MobileNetV2
"""
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from joblib import Parallel, delayed
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision.models import mobilenet_v2

DATA, OUT = Path("dataset"), Path("output")
# kelas yang dipakai (nama subfolder di DATA); None = semua subfolder
CLASSES = ["batik-ceplok", "batik-kawung", "batik-megamendung", "batik-parang", "batik-tambal"]
SIZE = 256            # semua citra di-resize ke SIZE x SIZE
FILTER = None         # None (tanpa filter), "median", atau "gaussian"
KSIZE = 3             # ukuran kernel filter (ganjil), dipakai bila FILTER tidak None
LBP_SCALES = ((8, 1), (16, 2), (24, 3))   # (jumlah tetangga, radius); satu skala = satu kanal peta LBP
SIFT_STEP, SIFT_SIZE = 8, 4               # jarak grid dan ukuran titik dense SIFT (jendela deskriptor ~6x ukuran titik)
DL_SIZE = 128         # ukuran peta fitur yang masuk ke CNN dan MobileNetV2
TEST_SIZE = 0.2       # porsi data uji
EPOCHS, BATCH = 20, 16
LR = {"CNN": 1e-3, "MobileNetV2": 1e-4}   # learning rate Adam per model
SEED = 0


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
    if FILTER is None:
        return img
    return cv2.medianBlur(img, KSIZE) if FILTER == "median" else cv2.GaussianBlur(img, (KSIZE, KSIZE), 0)


def tag(m):
    return f"{m} + {FILTER}" if FILTER else m


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a, b):
    m = mse(a, b)
    return float("inf") if m == 0 else 10 * np.log10(255 ** 2 / m)


def lbp_map(img):
    """Peta LBP 3 kanal (satu kanal per skala), nilai 0..1, ukuran DL_SIZE."""
    chans = [local_binary_pattern(img, p, r, method="uniform") / (p + 1) for p, r in LBP_SCALES]
    return np.stack([cv2.resize(c.astype(np.float32), (DL_SIZE, DL_SIZE), interpolation=cv2.INTER_AREA)
                     for c in chans])


def sift_grid(img):
    """Dense SIFT: deskriptor RootSIFT 128 angka di tiap titik grid -> array (baris, kolom, 128)."""
    pos = range(SIFT_STEP // 2, img.shape[0], SIFT_STEP)
    kp = [cv2.KeyPoint(float(x), float(y), float(SIFT_SIZE)) for y in pos for x in pos]
    d = cv2.SIFT_create().compute(img, kp)[1]
    assert len(d) == len(kp), "ada titik SIFT yang dibuang OpenCV"
    d = np.sqrt(d / (d.sum(1, keepdims=True) + 1e-7))
    return d.reshape(len(pos), len(pos), 128)


def sift_maps(imgs, train_idx):
    """Peta SIFT 3 kanal: 128 angka per titik grid diringkas ke 3 komponen PCA (dibangun dari citra latih saja)."""
    grids = Parallel(n_jobs=-1)(delayed(sift_grid)(im) for im in imgs)
    train = np.vstack([grids[i].reshape(-1, 128) for i in train_idx])
    pca = PCA(3, random_state=SEED).fit(train)
    z = pca.transform(train)
    mu, sd = z.mean(0), z.std(0)
    maps = []
    for g in grids:
        m = (pca.transform(g.reshape(-1, 128)) - mu) / sd                 # z-score per komponen
        m = np.clip((m + 3) / 6, 0, 1).reshape(g.shape[0], g.shape[1], 3)  # ke rentang 0..1
        maps.append(cv2.resize(m.astype(np.float32), (DL_SIZE, DL_SIZE), interpolation=cv2.INTER_LINEAR))
    return np.stack(maps).transpose(0, 3, 1, 2), pca.explained_variance_ratio_.sum()


def small_cnn(n_classes):
    """CNN sederhana yang dilatih dari nol: 3 blok konvolusi, lalu global average pooling."""
    def block(i, o):
        return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*block(3, 16), *block(16, 32), *block(32, 64),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))


def mobilenet(n_classes):
    """MobileNetV2 pretrained ImageNet; lapisan klasifikasi diganti untuk kelas batik, lalu seluruh jaringan di-fine-tune."""
    net = mobilenet_v2(weights="IMAGENET1K_V2")
    net.classifier[1] = nn.Linear(net.last_channel, n_classes)
    return net


MODELS = {"CNN": small_cnn, "MobileNetV2": mobilenet}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def train_eval(name, X, y, tr, te, n_classes):
    """Latih model `name` pada X[tr] lalu kembalikan prediksi untuk X[te]."""
    torch.manual_seed(SEED)
    model = MODELS[name](n_classes)
    X = (torch.from_numpy(X).float() - MEAN) / STD
    y = torch.from_numpy(y)
    opt = torch.optim.Adam(model.parameters(), lr=LR[name])
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.from_numpy(tr)[torch.randperm(len(tr))]
        total = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            xb = X[idx]
            flip = torch.rand(len(idx)) < 0.5                       # augmentasi: flip horizontal acak
            xb[flip] = xb[flip].flip(-1)
            opt.zero_grad()
            loss = loss_fn(model(xb), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if ep % 5 == 0:
            print(f"    epoch {ep:>2}: loss latih {total / len(tr):.3f}")
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[te[i:i + 32]]).argmax(1) for i in range(0, len(te), 32)]).numpy()


def save_fig_a(name, bgr, gray, results):
    """results: {metode: (enhanced, filtered)}"""
    fig, ax = plt.subplots(3, 4, figsize=(16, 12))
    ax[0, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); ax[0, 0].set_title("Input")
    ax[1, 0].imshow(gray, cmap="gray"); ax[1, 0].set_title("Grayscale")
    ax[2, 0].hist(gray.ravel(), 256, range=(0, 256)); ax[2, 0].set_title("Histogram Grayscale")
    for j, (m, (enh, flt)) in enumerate(results.items(), 1):
        ax[0, j].imshow(enh, cmap="gray"); ax[0, j].set_title(m)
        ax[1, j].imshow(flt, cmap="gray")
        ax[1, j].set_title(f"{tag(m)}\nMSE={mse(gray, flt):.1f}  PSNR={psnr(gray, flt):.2f} dB")
        ax[2, j].hist(enh.ravel(), 256, range=(0, 256)); ax[2, j].set_title(f"Histogram {m}")
    for a in ax[:2].ravel():
        a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / f"A_{name}.png", dpi=100); plt.close(fig)


def save_fig_b(name, best, img, maps):
    """Citra hasil enhancement terbaik, peta SIFT, peta LBP, dan tiap kanal LBP."""
    fig, ax = plt.subplots(1, 6, figsize=(24, 4.5))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title(tag(best))
    ax[1].imshow(maps["SIFT"].transpose(1, 2, 0)); ax[1].set_title("Peta SIFT (3 komponen PCA)")
    ax[2].imshow(maps["LBP"].transpose(1, 2, 0)); ax[2].set_title("Peta LBP (3 skala sebagai RGB)")
    for k, (p, r) in enumerate(LBP_SCALES):
        ax[3 + k].imshow(maps["LBP"][k], cmap="gray"); ax[3 + k].set_title(f"LBP P={p}, R={r}")
    for a in ax:
        a.axis("off")
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
    print(f"\n{len(labels)} citra, {len(first)} kelas, filter = {f'{FILTER} {KSIZE}x{KSIZE}' if FILTER else 'tanpa filter'}")
    # citra yang tidak berubah sama sekali (MSE = 0) punya PSNR tak hingga; dikeluarkan dari rata-rata PSNR
    fin = lambda v: [x for x in v if np.isfinite(x)]
    mean_mse = {}
    print(f"\n{'Metode':<22}{'Rata-rata MSE':>15}{'Rata-rata PSNR (dB)':>22}{'Citra MSE=0':>13}")
    for m in ENHANCE:
        r = [x for x in rows if x[2] == m]
        mean_mse[m] = np.mean([x[3] for x in r])
        print(f"{m:<22}{mean_mse[m]:>15.2f}{np.mean(fin([x[4] for x in r])):>22.2f}{sum(x[3] == 0 for x in r):>13}")

    # sebaran PSNR per metode + jumlah citra yang PSNR-nya tertinggi pada metode itu
    n = len(ENHANCE)
    winners = [max(rows[i:i + n], key=lambda x: x[4])[2] for i in range(0, len(rows), n)]
    print(f"\n{'Metode':<22}{'Std PSNR':>10}{'PSNR min':>10}{'PSNR maks':>11}{'Citra PSNR tertinggi':>22}")
    for m in ENHANCE:
        ps = fin([x[4] for x in rows if x[2] == m])
        print(f"{m:<22}{np.std(ps):>10.2f}{min(ps):>10.2f}{max(ps):>11.2f}{winners.count(m):>22}")

    print("\nRata-rata per kelas (MSE / PSNR dB)")
    print(f"{'Kelas':<20}" + "".join(f"{m:>24}" for m in ENHANCE))
    for c in first:
        cells = []
        for m in ENHANCE:
            r = [x for x in rows if x[0] == c and x[2] == m]
            cells.append(f"{np.mean([x[3] for x in r]):.2f} / {np.mean(fin([x[4] for x in r])):.2f}")
        print(f"{c:<20}" + "".join(f"{v:>24}" for v in cells))

    best = min(mean_mse, key=mean_mse.get)
    print(f"\nEnhancement terbaik (MSE rata-rata terendah): {best}")

    # B: peta fitur SIFT dan LBP dari enhancement terbaik
    classes = list(first)
    y = np.array([classes.index(c) for c in labels])
    tr, te = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=42)
    sift, var = sift_maps(imgs[best], tr)
    maps = {"SIFT": sift, "LBP": np.stack(Parallel(n_jobs=-1)(delayed(lbp_map)(im) for im in imgs[best]))}
    np.savez_compressed(OUT / "fitur.npz", label=np.array(labels), train_idx=tr, test_idx=te,
                        **{f: v.astype(np.float16) for f, v in maps.items()})
    for name, i in first.items():
        save_fig_b(name, best, imgs[best][i], {f: v[i] for f, v in maps.items()})
    print(f"Peta SIFT: grid {sift_grid(imgs[best][0]).shape[0]}x{sift_grid(imgs[best][0]).shape[1]}, "
          f"3 komponen PCA menjelaskan {var:.1%} variasi deskriptor")
    print(f"Peta fitur masuk ke model: 3 x {DL_SIZE} x {DL_SIZE}")

    # C: klasifikasi tiap peta fitur dengan CNN dan MobileNetV2
    print(f"\n{len(tr)} citra latih, {len(te)} citra uji, {EPOCHS} epoch")
    results, out_no = [], 0
    for f in ("SIFT", "LBP"):
        for name in MODELS:
            out_no += 1
            print(f"\nOutput {out_no}: {f} + {name}")
            pred = train_eval(name, maps[f], y, tr, te, len(classes))
            results.append((out_no, f, name, pred))

    print(f"\n{'Output':<8}{'Fitur':<7}{'Model':<14}{'Akurasi Uji':>12}{'Macro F1 Uji':>14}")
    for out_no, f, name, pred in results:
        print(f"{out_no:<8}{f:<7}{name:<14}{(pred == y[te]).mean():>12.3f}"
              f"{f1_score(y[te], pred, average='macro'):>14.3f}")

    print("\nRecall per kelas (data uji)")
    print(f"{'Output':<30}" + "".join(f"{c[6:]:>13}" for c in classes))
    cms = [(f"{o}. {f} + {name}", confusion_matrix(y[te], pred, labels=range(len(classes))))
           for o, f, name, pred in results]
    for label, cm in cms:
        print(f"{label:<30}" + "".join(f"{v:>13.2f}" for v in cm.diagonal() / cm.sum(1)))

    print("\nConfusion matrix (data uji; baris = kelas asli, kolom = prediksi)")
    for label, cm in cms:
        print(f"\n{label}")
        print(f"{'':<20}" + "".join(f"{c[6:]:>13}" for c in classes))
        for c, row in zip(classes, cm):
            print(f"{c:<20}" + "".join(f"{v:>13}" for v in row))
    print(f"\nGambar, mse_psnr.csv, dan fitur.npz tersimpan di {OUT}/")


def selfcheck():
    a, b = np.zeros((4, 4), np.uint8), np.full((4, 4), 255, np.uint8)
    assert mse(a, a) == 0 and psnr(a, a) == float("inf")
    assert mse(a, b) == 255 ** 2 and abs(psnr(a, b)) < 1e-9
    r = np.random.default_rng(0).integers(0, 256, (SIZE, SIZE), dtype=np.uint8)
    assert abs(psnr(r, cv2.medianBlur(r, 3)) - cv2.PSNR(r, cv2.medianBlur(r, 3))) < 1e-6
    assert contrast_stretch(np.arange(100, 164, dtype=np.uint8).reshape(8, 8)).max() == 255
    m = lbp_map(r)
    assert m.shape == (3, DL_SIZE, DL_SIZE) and 0 <= m.min() and m.max() <= 1
    assert sift_grid(r).shape[2] == 128
    x = torch.zeros(2, 3, DL_SIZE, DL_SIZE)
    assert all(MODELS[k](5).eval()(x).shape == (2, 5) for k in MODELS)


if __name__ == "__main__":
    selfcheck()
    main()
