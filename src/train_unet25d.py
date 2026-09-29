"""
ADIM 2 — 2.5D U-Net Egitimi (Duzeltilmis)
============================================
TensorFlow / Keras — GPU (CUDA)
Giris : (512, 512, 3)  — onceki, merkez, sonraki CT kesimleri
Cikis : (512, 512, 1)  — merkez kesitin kas maskesi

Calistirma:
  python 2_egitim.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────
# GPU AYARLARI
# ─────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError(
        "GPU bulunamadi! CUDA kurulu mu? "
        "'nvidia-smi' komutunu kontrol edin.")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("Kullanilan GPU:")
for g in gpus:
    print(" ", g)

# ─────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────
# Proje kok klasoru: SARKOPENI_DIR ortam degiskeniyle degistirilebilir
BASE_DIR = Path(os.environ.get("SARKOPENI_DIR", r"E:\sarkopeni"))

EGITIM_DIR   = BASE_DIR / "output" / "egitim_25d"
MODEL_DIR    = BASE_DIR / "modeller"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_BOYUT    = 512
BATCH_SIZE   = 2       # VRAM yetmezse 1 yap
EPOCHS       = 80
TEMEL_FILTRE = 32
LR           = 1e-4
POS_WEIGHT   = 8.0     # 1_onisleme.py'nin verdigi degeri buraya yaz


# ═══════════════════════════════════════════
# VERİ YÜKLEYİCİ
# ═══════════════════════════════════════════
class SarcopeniDataset(keras.utils.Sequence):
    def __init__(self, bolme, batch_size, augment=False, karistir=True):
        self.batch_size = batch_size
        self.augment    = augment
        self.karistir   = karistir

        self.x_dosyalar = sorted(
            (EGITIM_DIR / bolme / "X").glob("*_X.npy"))
        self.y_dosyalar = sorted(
            (EGITIM_DIR / bolme / "Y").glob("*_Y.npy"))

        assert len(self.x_dosyalar) == len(self.y_dosyalar), \
            "X ve Y dosya sayisi eslesmiyor!"

        self.indeksler = np.arange(len(self.x_dosyalar))
        if self.karistir:
            np.random.shuffle(self.indeksler)

        print("[{}] {} ornek".format(bolme.upper(), len(self.x_dosyalar)))

    def __len__(self):
        return int(np.ceil(len(self.x_dosyalar) / self.batch_size))

    def __getitem__(self, batch_idx):
        bas  = batch_idx * self.batch_size
        bitis = min(bas + self.batch_size, len(self.x_dosyalar))
        idx_list = self.indeksler[bas:bitis]

        X_b, Y_b = [], []
        for idx in idx_list:
            x = np.load(str(self.x_dosyalar[idx])).astype(np.float32)
            y = np.load(str(self.y_dosyalar[idx])).astype(np.float32)
            if self.augment:
                x, y = self._augment(x, y)
            X_b.append(x)
            Y_b.append(y)

        return np.array(X_b), np.array(Y_b)

    def on_epoch_end(self):
        if self.karistir:
            np.random.shuffle(self.indeksler)

    def _augment(self, x, y):
        if np.random.rand() > 0.5:
            x = np.fliplr(x); y = np.fliplr(y)
        if np.random.rand() > 0.5:
            x = np.flipud(x); y = np.flipud(y)
        return x, y


# ═══════════════════════════════════════════
# U-NET MİMARİSİ
# ═══════════════════════════════════════════
def conv_blok(x, filtre, dropout=0.0):
    x = layers.Conv2D(filtre, 3, padding="same",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filtre, 3, padding="same",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def unet_25d(img_boyut=512, giris_kanal=3, temel=32):
    giris = keras.Input(shape=(img_boyut, img_boyut, giris_kanal),
                        name="giris_25d")

    # Encoder
    c1 = conv_blok(giris, temel);           p1 = layers.MaxPooling2D(2)(c1)
    c2 = conv_blok(p1,    temel*2,  0.1);   p2 = layers.MaxPooling2D(2)(c2)
    c3 = conv_blok(p2,    temel*4,  0.1);   p3 = layers.MaxPooling2D(2)(c3)
    c4 = conv_blok(p3,    temel*8,  0.2);   p4 = layers.MaxPooling2D(2)(c4)

    # Bottleneck
    c5 = conv_blok(p4, temel*16, 0.3)

    # Decoder
    u6 = layers.Conv2DTranspose(temel*8, 2, strides=2, padding="same")(c5)
    u6 = layers.Concatenate()([u6, c4]);    c6 = conv_blok(u6, temel*8)

    u7 = layers.Conv2DTranspose(temel*4, 2, strides=2, padding="same")(c6)
    u7 = layers.Concatenate()([u7, c3]);    c7 = conv_blok(u7, temel*4)

    u8 = layers.Conv2DTranspose(temel*2, 2, strides=2, padding="same")(c7)
    u8 = layers.Concatenate()([u8, c2]);    c8 = conv_blok(u8, temel*2)

    u9 = layers.Conv2DTranspose(temel, 2, strides=2, padding="same")(c8)
    u9 = layers.Concatenate()([u9, c1]);    c9 = conv_blok(u9, temel)

    cikis = layers.Conv2D(1, 1, activation="sigmoid",
                          name="maske_cikis")(c9)
    return keras.Model(giris, cikis, name="unet_25d_512")


# ═══════════════════════════════════════════
# KAYIP FONKSİYONLARI
# ═══════════════════════════════════════════

@tf.autograph.experimental.do_not_convert
def dice_kayip(y_gercek, y_tahmin, smooth=1e-6):
    yg = tf.reshape(tf.cast(y_gercek, tf.float32), [-1])
    yt = tf.reshape(tf.cast(y_tahmin, tf.float32), [-1])
    k  = tf.reduce_sum(yg * yt)
    return 1.0 - (2.0*k + smooth) / (
        tf.reduce_sum(yg) + tf.reduce_sum(yt) + smooth)


@tf.autograph.experimental.do_not_convert
def dice_skoru(y_gercek, y_tahmin, smooth=1e-6):
    yg = tf.reshape(tf.cast(y_gercek, tf.float32), [-1])
    yt = tf.reshape(tf.cast(y_tahmin > 0.5, tf.float32), [-1])
    k  = tf.reduce_sum(yg * yt)
    return (2.0*k + smooth) / (
        tf.reduce_sum(yg) + tf.reduce_sum(yt) + smooth)


@tf.autograph.experimental.do_not_convert
def iou_skoru(y_gercek, y_tahmin, smooth=1e-6):
    yg = tf.reshape(tf.cast(y_gercek, tf.float32), [-1])
    yt = tf.reshape(tf.cast(y_tahmin > 0.5, tf.float32), [-1])
    k  = tf.reduce_sum(yg * yt)
    b  = tf.reduce_sum(yg) + tf.reduce_sum(yt) - k
    return (k + smooth) / (b + smooth)


def birlesik_kayip_olustur(pos_weight: float):
    """
    Ağırlıklı BCE + Dice loss.
    tf.where yerine doğrudan piksel bazlı ağırlık matrisi kullanılır
    (shape uyumsuzluğu hatasını önler).
    """
    pw = float(pos_weight)

    @tf.autograph.experimental.do_not_convert
    def kayip(y_gercek, y_tahmin):
        yg  = tf.cast(y_gercek, tf.float32)
        yt  = tf.cast(y_tahmin, tf.float32)

        # Piksel başına ağırlık: pozitif piksel → pw, negatif piksel → 1
        agirlik = yg * (pw - 1.0) + 1.0   # shape: (B, H, W, 1)

        # Piksel başına BCE
        eps = 1e-7
        yt_k = tf.clip_by_value(yt, eps, 1.0 - eps)
        bce  = -(yg * tf.math.log(yt_k) +
                 (1.0 - yg) * tf.math.log(1.0 - yt_k))   # (B, H, W, 1)

        bce_agirlikli = tf.reduce_mean(agirlik * bce)
        return 0.5 * bce_agirlikli + 0.5 * dice_kayip(yg, yt)

    kayip.__name__ = "birlesik_kayip"
    return kayip


# ═══════════════════════════════════════════
# EĞİTİM
# ═══════════════════════════════════════════
def egit():
    zaman   = datetime.now().strftime("%Y%m%d_%H%M")
    model_adi = "unet25d_512_{}".format(zaman)
    model_kayit = MODEL_DIR / model_adi
    model_kayit.mkdir(exist_ok=True)

    print("=" * 60)
    print("2.5D U-Net Egitimi — 512x512 — GPU")
    print("Model klasoru:", model_kayit)
    print("=" * 60)

    # Veri
    egitim_ds = SarcopeniDataset("egitim", BATCH_SIZE,
                                  augment=True,  karistir=True)
    val_ds    = SarcopeniDataset("test",   BATCH_SIZE,
                                  augment=False, karistir=False)

    # Model
    with tf.device("/GPU:0"):
        model = unet_25d(IMG_BOYUT, giris_kanal=3, temel=TEMEL_FILTRE)

    model.summary()
    print("\nToplam parametre: {:,}".format(model.count_params()))

    kayip_fn = birlesik_kayip_olustur(POS_WEIGHT)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss=kayip_fn,
        metrics=[dice_skoru, iou_skoru]
    )

    # Callback'ler
    cb = [
        keras.callbacks.ModelCheckpoint(
            str(model_kayit / "en_iyi.keras"),
            monitor="val_dice_skoru",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_dice_skoru",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_skoru",
            mode="max",
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            str(model_kayit / "egitim_log.csv")
        ),
    ]

    # Eğitim
    gecmis = model.fit(
        egitim_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cb,
        verbose=1
    )

    # Grafik
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Egitim — {}".format(model_adi))

    axes[0].plot(gecmis.history["loss"],         label="Egitim")
    axes[0].plot(gecmis.history["val_loss"],     label="Validasyon")
    axes[0].set_title("Kayip"); axes[0].legend()

    axes[1].plot(gecmis.history["dice_skoru"],     label="Egitim")
    axes[1].plot(gecmis.history["val_dice_skoru"], label="Validasyon")
    axes[1].set_title("Dice Skoru"); axes[1].legend()

    axes[2].plot(gecmis.history["iou_skoru"],     label="Egitim")
    axes[2].plot(gecmis.history["val_iou_skoru"], label="Validasyon")
    axes[2].set_title("IoU Skoru"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(model_kayit / "egitim_grafigi.png", dpi=130)
    plt.close()

    # Meta
    en_iyi = int(np.argmax(gecmis.history["val_dice_skoru"]))
    meta   = {
        "model_adi":        model_adi,
        "img_boyut":        IMG_BOYUT,
        "batch_size":       BATCH_SIZE,
        "epochs_calisilan": len(gecmis.history["loss"]),
        "en_iyi_epoch":     en_iyi + 1,
        "val_dice":         round(gecmis.history["val_dice_skoru"][en_iyi], 4),
        "val_iou":          round(gecmis.history["val_iou_skoru"][en_iyi],  4),
        "val_loss":         round(gecmis.history["val_loss"][en_iyi],       4),
        "pos_weight":       POS_WEIGHT,
        "temel_filtre":     TEMEL_FILTRE,
        "lr":               LR,
    }
    with open(model_kayit / "egitim_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("EGİTİM TAMAMLANDI")
    print("  En iyi epoch  : {}".format(meta["en_iyi_epoch"]))
    print("  Val Dice      : {}".format(meta["val_dice"]))
    print("  Val IoU       : {}".format(meta["val_iou"]))
    print("  Model         : {}".format(model_kayit / "en_iyi.keras"))
    print("=" * 60)

    return model, model_kayit


if __name__ == "__main__":
    egit()