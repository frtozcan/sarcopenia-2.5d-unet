"""
ADIM 1 — On Isleme (Preprocessing)
=====================================
Egitim_25d klasorundeki ham npy dosyalarini kontrol eder,
istatistikleri hesaplar ve egitim icin hazirlik raporu uretir.

Calistirma:
  python 1_onisleme.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────
# Proje kok klasoru: SARKOPENI_DIR ortam degiskeniyle degistirilebilir
BASE_DIR = Path(os.environ.get("SARKOPENI_DIR", r"E:\sarkopeni"))

EGITIM_DIR = BASE_DIR / "output" / "egitim_25d"
RAPOR_DIR  = EGITIM_DIR / "rapor"
RAPOR_DIR.mkdir(exist_ok=True)

IMG_BOYUT = 512   # model giris boyutu


# ═══════════════════════════════════════════
# 1. DOSYA SAYIMI VE BOYUT KONTROLU
# ═══════════════════════════════════════════
def dosya_kontrol():
    sonuc = {}
    for bolme in ["egitim", "test"]:
        x_dosyalar = sorted((EGITIM_DIR / bolme / "X").glob("*_X.npy"))
        y_dosyalar = sorted((EGITIM_DIR / bolme / "Y").glob("*_Y.npy"))

        print("\n[{}]".format(bolme.upper()))
        print("  X dosya sayisi : {}".format(len(x_dosyalar)))
        print("  Y dosya sayisi : {}".format(len(y_dosyalar)))

        if len(x_dosyalar) != len(y_dosyalar):
            print("  UYARI: X ve Y sayisi eslesmiyor!")

        # İlk dosyayı kontrol et
        if x_dosyalar:
            x0 = np.load(str(x_dosyalar[0]))
            y0 = np.load(str(y_dosyalar[0]))
            print("  X shape        : {}  dtype: {}".format(x0.shape, x0.dtype))
            print("  Y shape        : {}  dtype: {}".format(y0.shape, y0.dtype))
            print("  X min/max      : {:.4f} / {:.4f}".format(x0.min(), x0.max()))
            print("  Y benzersiz    : {}".format(np.unique(y0)))
            print("  Y pozitif oran : {:.2f}%".format(
                100 * y0.mean()))

        sonuc[bolme] = {
            "x_sayisi": len(x_dosyalar),
            "y_sayisi": len(y_dosyalar),
        }
    return sonuc


# ═══════════════════════════════════════════
# 2. MASKE DAĞILIM ANALİZİ
# ═══════════════════════════════════════════
def maske_dagilim_analizi(max_ornek=200):
    """
    Eğitim setindeki maskelerin pozitif piksel oranlarını analiz eder.
    Sınıf dengesizliği varsa class_weight hesaplar.
    """
    print("\n[MASKE DAGILIM ANALİZİ]")
    x_dosyalar = sorted((EGITIM_DIR / "egitim" / "X").glob("*_X.npy"))

    # Rastgele örnekle (hepsi çok uzun sürebilir)
    np.random.seed(42)
    secilen = np.random.choice(
        len(x_dosyalar),
        min(max_ornek, len(x_dosyalar)),
        replace=False)

    pozitif_oranlar = []
    y_dosyalar = sorted((EGITIM_DIR / "egitim" / "Y").glob("*_Y.npy"))

    for idx in tqdm(secilen, desc="Analiz ediliyor"):
        y = np.load(str(y_dosyalar[idx]))
        pozitif_oranlar.append(float(y.mean()))

    pozitif_oranlar = np.array(pozitif_oranlar)
    ort_pozitif = pozitif_oranlar.mean()
    ort_negatif = 1 - ort_pozitif

    print("  Ortalama pozitif piksel orani : {:.2f}%".format(
        100 * ort_pozitif))
    print("  Ortalama negatif piksel orani : {:.2f}%".format(
        100 * ort_negatif))

    # Class weight hesapla
    if ort_pozitif > 0:
        pos_weight = round(ort_negatif / ort_pozitif, 2)
        print("  Onerilen pos_weight           : {}".format(pos_weight))
        print("  (Egitimde BCE loss icin kullan)")
    else:
        pos_weight = 1.0

    # Histogram
    plt.figure(figsize=(8, 4))
    plt.hist(pozitif_oranlar * 100, bins=30, color="#378ADD", edgecolor="white")
    plt.xlabel("Pozitif piksel orani (%)")
    plt.ylabel("Ornek sayisi")
    plt.title("Kas maskesi piksel dagilimi (n={})".format(len(secilen)))
    plt.tight_layout()
    plt.savefig(RAPOR_DIR / "maske_dagilim.png", dpi=120)
    plt.close()
    print("  Grafik: {}".format(RAPOR_DIR / "maske_dagilim.png"))

    return pos_weight


# ═══════════════════════════════════════════
# 3. GÖRSEL ÖN İZLEME
# ═══════════════════════════════════════════
def gorsel_on_izleme(n=6):
    """
    Rastgele n örnek için X ve Y görselini yan yana gösterir.
    """
    print("\n[GORSEL ON IZLEME]")
    x_dosyalar = sorted((EGITIM_DIR / "egitim" / "X").glob("*_X.npy"))
    y_dosyalar = sorted((EGITIM_DIR / "egitim" / "Y").glob("*_Y.npy"))

    np.random.seed(0)
    secilen = np.random.choice(len(x_dosyalar), min(n, len(x_dosyalar)),
                               replace=False)

    fig, axes = plt.subplots(n, 5, figsize=(20, n * 4))
    basliklar = ["Onceki CT", "Merkez CT", "Sonraki CT", "Kas Maskesi", "Birlesik"]

    for satir, idx in enumerate(secilen):
        x = np.load(str(x_dosyalar[idx]))  # (512, 512, 3)
        y = np.load(str(y_dosyalar[idx]))  # (512, 512, 1)

        dosya_adi = x_dosyalar[idx].stem.replace("_X", "")

        axes[satir, 0].imshow(x[:, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[satir, 1].imshow(x[:, :, 1], cmap="gray", vmin=0, vmax=1)
        axes[satir, 2].imshow(x[:, :, 2], cmap="gray", vmin=0, vmax=1)
        axes[satir, 3].imshow(y[:, :, 0], cmap="Greens", vmin=0, vmax=1)
        axes[satir, 4].imshow(x[:, :, 1], cmap="gray", vmin=0, vmax=1)
        axes[satir, 4].imshow(y[:, :, 0], cmap="Reds", alpha=0.4, vmin=0, vmax=1)

        for col in range(5):
            axes[satir, col].axis("off")
            if satir == 0:
                axes[satir, col].set_title(basliklar[col], fontsize=10)

        axes[satir, 0].set_ylabel(dosya_adi[:25], fontsize=7, rotation=0,
                                  labelpad=120)

    plt.suptitle("2.5D Egitim Verisi On Izleme", fontsize=12)
    plt.tight_layout()
    plt.savefig(RAPOR_DIR / "onizleme.png", dpi=100)
    plt.close()
    print("  Gorsel: {}".format(RAPOR_DIR / "onizleme.png"))


# ═══════════════════════════════════════════
# 4. HASTA BAZLI İSTATİSTİK
# ═══════════════════════════════════════════
def hasta_istatistik():
    print("\n[HASTA BAZLI ISTATISTIK]")
    satirlar = []

    for bolme in ["egitim", "test"]:
        x_dosyalar = sorted((EGITIM_DIR / bolme / "X").glob("*_X.npy"))
        hasta_sayac = {}
        for f in x_dosyalar:
            # dosya adi: "hasta_001_z0131_X.npy" → hasta: "hasta_001"
            parcalar = f.stem.replace("_X", "").rsplit("_z", 1)
            hasta    = parcalar[0] if len(parcalar) == 2 else f.stem
            hasta_sayac[hasta] = hasta_sayac.get(hasta, 0) + 1

        for hasta, sayac in sorted(hasta_sayac.items()):
            satirlar.append({
                "bolme": bolme, "hasta": hasta, "dilim_sayisi": sayac
            })
        print("  {}: {} hasta, {} dilim".format(
            bolme.upper(), len(hasta_sayac), sum(hasta_sayac.values())))

    df = pd.DataFrame(satirlar)
    df.to_excel(RAPOR_DIR / "hasta_istatistik.xlsx", index=False)
    print("  Excel: {}".format(RAPOR_DIR / "hasta_istatistik.xlsx"))
    return df


# ═══════════════════════════════════════════
# ANA
# ═══════════════════════════════════════════
def main():
    print("=" * 55)
    print("ON ISLEME RAPORU")
    print("Egitim klasoru:", EGITIM_DIR)
    print("=" * 55)

    # 1. Dosya kontrolü
    dosya_kontrol()

    # 2. Maske dağılımı + class weight
    pos_weight = maske_dagilim_analizi(max_ornek=300)

    # 3. Görsel önizleme
    gorsel_on_izleme(n=6)

    # 4. Hasta istatistikleri
    hasta_istatistik()

    # Rapor özeti
    print("\n" + "=" * 55)
    print("ON ISLEME TAMAMLANDI")
    print("Egitim scriptine gececek pos_weight: {}".format(pos_weight))
    print("Bu degeri 2_egitim.py icerisindeki")
    print("POS_WEIGHT = {} satirina yazin".format(pos_weight))
    print("=" * 55)

    return pos_weight


if __name__ == "__main__":
    main()