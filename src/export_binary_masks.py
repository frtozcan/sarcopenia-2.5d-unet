"""
Ikili (siyah-beyaz) maske goruntuleri
=====================================
sonuc klasorunde gercek maske / tahmin paneli kaydedilmis her test hastasi icin
merkez dilimin (test betigindeki ile ayni: Dice'i en yuksek dilim)
  - gercek maskesini
  - model tahminini
maske = beyaz (255), geri kalan her yer = siyah (0) olacak sekilde kaydeder.

Cikti: <SARKOPENI_DIR>\\sonuc\\ikili_maske\\<hasta>_gercek_maske_ikili.png
                                        <hasta>_tahmin_maskesi_ikili.png
Boyut 1024x1024 (diger panellerle ayni), en yakin komsu buyutme (keskin kenar).

Tum test hastalari icin uretmek isterseniz TUM_HASTALAR = True yapin.
"""
import os
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Proje kok klasoru: SARKOPENI_DIR ortam degiskeniyle degistirilebilir
BASE_DIR = Path(os.environ.get("SARKOPENI_DIR", r"E:\sarkopeni"))

MODEL_YOLU   = str(BASE_DIR / "modeller" / "unet25d_512_20260928_1700" / "en_iyi.keras")
EGITIM_DIR   = BASE_DIR / "output" / "egitim_25d"
SONUC_DIR    = BASE_DIR / "sonuc"
CIKTI_DIR    = SONUC_DIR / "ikili_maske"
ESIK         = 0.5
BOYUT        = 1024
TUM_HASTALAR = False


def dice(yg, yt, s=1e-6):
    yg = yg > 0.5; yt = yt > 0.5
    return (2 * np.sum(yg & yt) + s) / (yg.sum() + yt.sum() + s)


def ikili_kaydet(maske, yol):
    img = ((maske > 0.5).astype(np.uint8)) * 255
    img = cv2.resize(img, (BOYUT, BOYUT), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(yol), img)


def main():
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    model = tf.keras.models.load_model(MODEL_YOLU, compile=False)
    CIKTI_DIR.mkdir(parents=True, exist_ok=True)

    gruplar = {}
    for xf in sorted((EGITIM_DIR / "test" / "X").glob("*_X.npy")):
        hasta = xf.stem.replace("_X", "").rsplit("_z", 1)[0]
        gruplar.setdefault(hasta, []).append(xf)

    if TUM_HASTALAR:
        secilen = sorted(gruplar)
    else:
        secilen = sorted({h for h in gruplar
                          if (SONUC_DIR / "{}_2_gercek_maske.png".format(h[:35])).exists()
                          or (SONUC_DIR / "{}_3_tahmin_maskesi.png".format(h[:35])).exists()})
    print("Islenecek hasta:", len(secilen))

    for hasta in secilen:
        en_iyi = None
        for xf in gruplar[hasta]:
            x  = np.load(str(xf)).astype(np.float32)
            yf = xf.parent.parent / "Y" / xf.name.replace("_X.npy", "_Y.npy")
            yg = np.load(str(yf))[:, :, 0]
            yt = model.predict(x[np.newaxis], verbose=0)[0, :, :, 0] > ESIK
            d  = dice(yg, yt)
            if en_iyi is None or d > en_iyi[0]:
                en_iyi = (d, yg, yt, xf.stem)
        d, yg, yt, ad = en_iyi
        ikili_kaydet(yg, CIKTI_DIR / "{}_gercek_maske_ikili.png".format(hasta[:35]))
        ikili_kaydet(yt, CIKTI_DIR / "{}_tahmin_maskesi_ikili.png".format(hasta[:35]))
        print("  {:28s} dilim {}  Dice {:.3f}  -> kaydedildi".format(
            hasta, ad.rsplit("_z", 1)[1].replace("_X", ""), d))

    print("Bitti:", CIKTI_DIR)


if __name__ == "__main__":
    main()
