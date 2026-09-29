"""
ADIM 2 — DeepLab-UNeTR Hibrit Model Egitimi
=============================================
TensorFlow / Keras — GPU (CUDA)

Giris : (512, 512, 3) — onceki, merkez, sonraki CT kesimleri (2.5D)
Cikis : (512, 512, 1) — merkez kesitin kas maskesi (sigmoid)

Mimari:
  Paylasimli CNN omurgasi
      ├─ DeepLab ASPP kolu   (yerel, cok olcekli atrous conv)
      └─ UNeTR Transformer   (global, uzun menzil dikkat)
                ↓
      Adaptif SE-Net fuzyonu
                ↓
      Ince ayar + Bilinear upsample
                ↓
      Sigmoid segmentasyon cikisi

Calistirma:
  python 2_egitim_dlunetr.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")   # GPU sunucusunda ekran yoksa Agg yeterli
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────
#  GPU AYARLARI
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
#  AYARLAR
# ─────────────────────────────────────────
# Proje kok klasoru: SARKOPENI_DIR ortam degiskeniyle degistirilebilir
BASE_DIR = Path(os.environ.get("SARKOPENI_DIR", r"E:\sarkopeni"))

EGITIM_DIR   = BASE_DIR / "output" / "egitim_25d"
MODEL_DIR    = BASE_DIR / "DLUnetR"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_BOYUT    = 512
BATCH_SIZE   = 2        # VRAM yetmezse 1 yap
EPOCHS       = 80
FEATURE_DIM  = 128      # ASPP ve Transformer cikis kanal sayisi
                         # VRAM kisitliysa 64'e dusurebilirsiniz
TRANS_DEPTH  = 3        # Transformer blok sayisi (az = hizli, cok = daha iyi)
TRANS_HEADS  = 4        # Multi-head attention bas sayisi
TRANS_PATCH  = 4        # Patch boyutu (backbone cikisina gore)
LR           = 1e-4
POS_WEIGHT   = 8.0      # 1_onisleme.py'nin verdigi degeri buraya yaz


# ═══════════════════════════════════════════
#  VERİ YÜKLEYİCİ  (orijinal koddan aynen)
# ═══════════════════════════════════════════
class SarcopeniDataset(keras.utils.Sequence):
    """
    NaN Guvencesi:
    - __len__ her zaman tam bolme yapar (son eksik batch atilir)
    - __getitem__ herzaman tam BATCH_SIZE boyutunda array dondurur
    - Hicbir batch'te boyut farkliligi olmaz → NaN riski sifir
    """
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

        # Son eksik batch'i at: tam batch sayisi kadar ornek kullan
        n_tam = (len(self.x_dosyalar) // batch_size) * batch_size
        self.x_dosyalar = self.x_dosyalar[:n_tam]
        self.y_dosyalar = self.y_dosyalar[:n_tam]

        self.indeksler = np.arange(len(self.x_dosyalar))
        if self.karistir:
            np.random.shuffle(self.indeksler)

        print("[{}] {} ornek  ({} batch x {})".format(
            bolme.upper(), len(self.x_dosyalar),
            len(self.x_dosyalar) // batch_size, batch_size))

    def __len__(self):
        # Her zaman tam sayi — bolme artigi yok
        return len(self.x_dosyalar) // self.batch_size

    def __getitem__(self, batch_idx):
        bas      = batch_idx * self.batch_size
        idx_list = self.indeksler[bas: bas + self.batch_size]

        X_b, Y_b = [], []
        for idx in idx_list:
            x = np.load(str(self.x_dosyalar[idx])).astype(np.float32)
            y = np.load(str(self.y_dosyalar[idx])).astype(np.float32)
            # NaN / Inf kontrolu
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                y = np.zeros_like(y)
            if self.augment:
                x, y = self._augment(x, y)
            X_b.append(x)
            Y_b.append(y)

        return np.stack(X_b, axis=0), np.stack(Y_b, axis=0)

    def on_epoch_end(self):
        if self.karistir:
            np.random.shuffle(self.indeksler)

    def _augment(self, x, y):
        if np.random.rand() > 0.5:
            x = np.fliplr(x); y = np.fliplr(y)
        if np.random.rand() > 0.5:
            x = np.flipud(x); y = np.flipud(y)
        if np.random.rand() > 0.5:
            delta = np.random.uniform(-0.05, 0.05)
            x = np.clip(x + delta, 0.0, 1.0)
        return x, y


# ═══════════════════════════════════════════
#  YARDIMCI KATMANLAR
# ═══════════════════════════════════════════

def depthwise_sep_conv(x, filtre, dilation_rate=1):
    """
    Depthwise Separable Convolution.
    Standart 3x3 conv'a gore parametre tasarrufu saglar.
    CT segmentasyonunda yeterli ogrenim kapasitesi sunarken
    bellegi verimli kullanir.
    """
    x = layers.DepthwiseConv2D(
            3, padding="same",
            dilation_rate=dilation_rate,
            depthwise_initializer="he_normal")(x)
    x = layers.Conv2D(filtre, 1, use_bias=False,
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def se_block(x, filtre, reduction=8, name=None):
    """
    Squeeze-and-Excitation blogu.
    Her kanalin onemini ogrenir; fuzyonda hangi kolun
    (ASPP mi, Transformer mi) ne kadar agirlik tasiyacagina
    veri odakli karar verir.
    name parametresi Keras katman ismi catismasini onler.
    """
    mid = max(filtre // reduction, 8)
    s = layers.GlobalAveragePooling2D(
            name="{}_gap".format(name) if name else None)(x)
    s = layers.Dense(mid, activation="relu", use_bias=False,
                     name="{}_d1".format(name) if name else None)(s)
    s = layers.Dense(filtre, activation="sigmoid", use_bias=False,
                     name="{}_d2".format(name) if name else None)(s)
    s = layers.Reshape((1, 1, filtre),
                       name="{}_rs".format(name) if name else None)(s)
    return layers.Multiply(
        name="{}_mul".format(name) if name else None)([x, s])


# ═══════════════════════════════════════════
#  DEEPLAB KOLU — ASPP
# ═══════════════════════════════════════════

def aspp_modulu(giris, filtre=128, atrous_rates=(6, 12, 18)):
    """
    Atrous Spatial Pyramid Pooling.

    Neden kullaniyoruz?
    CT goruntusundeki kaslar farkli boyutlarda ve sekillerde
    gorulur. ASPP ayni anda 6, 12, 18 piksel genisligindeki
    alici alanlara bakarak hem kucuk hem buyuk kas bolgelerini
    yakalar. Global avg-pool ise tum kesit baglamini ekler.

    Giris gercek goruntu boyutuna degil, omurga cikisina
    (64x64) uygulanir; bu yuzden atrous oranlari bu olceye
    gore secilmistir.
    """
    kanal = giris.shape[-1]

    # 1x1 conv — en dar baglam
    p0 = layers.Conv2D(filtre, 1, use_bias=False,
                       kernel_initializer="he_normal")(giris)
    p0 = layers.BatchNormalization()(p0)
    p0 = layers.Activation("relu")(p0)

    # Atrous dallari — her biri DepthwiseSep ile hafifletildi
    dallar = [p0]
    for r in atrous_rates:
        d = depthwise_sep_conv(giris, filtre, dilation_rate=r)
        dallar.append(d)

    # Global ortalama havuz — tam goruntu baglami
    gp = layers.GlobalAveragePooling2D()(giris)
    gp = layers.Reshape((1, 1, kanal))(gp)
    gp = layers.Conv2D(filtre, 1, use_bias=False,
                       kernel_initializer="he_normal")(gp)
    gp = layers.BatchNormalization()(gp)
    gp = layers.Activation("relu")(gp)
    # Omurga cikis boyutuna geri buyut
    h = tf.shape(giris)[1]
    w = tf.shape(giris)[2]
    gp = layers.Lambda(
        lambda t: tf.image.resize(t[0], (tf.shape(t[1])[1],
                                         tf.shape(t[1])[2]),
                                  method="bilinear"),
        name="aspp_gp_resize"
    )([gp, giris])
    dallar.append(gp)

    # Birlestir ve sıkistir
    x = layers.Concatenate()(dallar)
    x = layers.Conv2D(filtre, 1, use_bias=False,
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)
    return x


# ═══════════════════════════════════════════
#  UNETR KOLU — TRANSFORMER
# ═══════════════════════════════════════════

class UNeTREncoder(layers.Layer):
    """
    Tam UNeTR Encoder katmani — tek bir Layer sinifi.

    Keras Functional API'de coklu donus degeri (Hp, Wp) ve
    tf.Variable pozisyonel kodlama desteklenmez. Bu nedenle
    PatchEmbed + pozisyonel kodlama + Transformer bloklari +
    2D yeniden sekillendirme hepsini tek bir Layer'da topluyoruz.

    Giriş : (B, H, W, C)   — omurga cikisi (64x64x512)
    Cikis : (B, H, W, D)   — token ozellikleri 2D'ye donmus
    """
    def __init__(self, embed_dim, num_heads, depth,
                 patch_size, out_filtre, dropout=0.1, **kw):
        super().__init__(**kw)
        self.embed_dim  = embed_dim
        self.patch_size = patch_size
        self.out_filtre = out_filtre

        # Patch projeksiyon: (B,H,W,C) → (B,H/p,W/p,D)
        self.patch_proj = layers.Conv2D(
            embed_dim, patch_size, strides=patch_size,
            use_bias=False, kernel_initializer="glorot_uniform",
            name="patch_proj")
        self.patch_norm = layers.LayerNormalization(name="patch_norm")

        # Transformer bloklari
        self.trans_blocks = []
        for i in range(depth):
            block = {
                "norm1": layers.LayerNormalization(name=f"tb{i}_n1"),
                "attn":  layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=embed_dim // num_heads,
                    dropout=dropout,
                    name=f"tb{i}_attn"),
                "norm2": layers.LayerNormalization(name=f"tb{i}_n2"),
                "ff1":   layers.Dense(embed_dim * 4, activation="gelu",
                                      name=f"tb{i}_ff1"),
                "drop1": layers.Dropout(dropout),
                "ff2":   layers.Dense(embed_dim, name=f"tb{i}_ff2"),
                "drop2": layers.Dropout(dropout),
                "adrop": layers.Dropout(dropout),
            }
            self.trans_blocks.append(block)
            # Keras'in takip edebilmesi icin her sub-layer'i kaydet
            for lname, lobj in block.items():
                setattr(self, f"_tb{i}_{lname}", lobj)

        self.final_norm = layers.LayerNormalization(name="trans_norm")

        # Cikis projeksiyon
        self.out_proj = layers.Conv2D(
            out_filtre, 1, use_bias=False,
            kernel_initializer="he_normal", name="trans_proj")
        self.out_bn   = layers.BatchNormalization(name="trans_bn")
        self.out_act  = layers.Activation("relu")
        self.drop_in  = layers.Dropout(dropout)

    def build(self, input_shape):
        # Pozisyonel kodlama: (1, N, D) — N patch sayisi
        H, W = input_shape[1], input_shape[2]
        if H is None or W is None:
            raise ValueError(
                "UNeTREncoder: giris boyutu bilinmeli (None kabul edilmez). "
                "IMG_BOYUT // 8 sabit olmali.")
        Hp = H // self.patch_size
        Wp = W // self.patch_size
        N  = Hp * Wp
        self.Hp = Hp
        self.Wp = Wp
        self.pos_emb = self.add_weight(
            name="pos_embedding",
            shape=(1, N, self.embed_dim),
            initializer="zeros",
            trainable=True)
        super().build(input_shape)

    @tf.autograph.experimental.do_not_convert
    def call(self, x, training=None):
        # 1) Patch projeksiyon: (B,H,W,C) → (B,Hp,Wp,D)
        x = self.patch_proj(x, training=training)
        B  = tf.shape(x)[0]
        Hp = tf.shape(x)[1]
        Wp = tf.shape(x)[2]
        D  = self.embed_dim

        # 2) Duzlestir: (B,Hp*Wp,D)
        tokens = tf.reshape(x, (B, Hp * Wp, D))
        tokens = self.patch_norm(tokens, training=training)

        # 3) Pozisyonel kodlama (add_weight ile tanimlandi)
        tokens = tokens + self.pos_emb
        tokens = self.drop_in(tokens, training=training)

        # 4) Transformer bloklari
        for block in self.trans_blocks:
            n      = block["norm1"](tokens, training=training)
            attn_o = block["attn"](n, n, training=training)
            attn_o = block["adrop"](attn_o, training=training)
            tokens = tokens + attn_o
            n2     = block["norm2"](tokens, training=training)
            ff     = block["ff1"](n2, training=training)
            ff     = block["drop1"](ff, training=training)
            ff     = block["ff2"](ff, training=training)
            ff     = block["drop2"](ff, training=training)
            tokens = tokens + ff

        tokens = self.final_norm(tokens, training=training)

        # 5) 2D'ye don: (B, Hp, Wp, D)
        feat = tf.reshape(tokens, (B, self.Hp, self.Wp, D))

        # 6) Cikis projeksiyon
        feat = self.out_proj(feat, training=training)
        feat = self.out_bn(feat, training=training)
        feat = self.out_act(feat)
        return feat

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim":  self.embed_dim,
            "patch_size": self.patch_size,
            "out_filtre": self.out_filtre,
        })
        return cfg


def unetr_kolu(giris, embed_dim=512, depth=3,
               num_heads=4, patch_size=4, out_filtre=128):
    """
    UNeTR transformer kolunu tek bir Layer uzerinden cagirır.
    Tum patch embed + positional encoding + transformer + 2D yeniden
    sekillendirme UNeTREncoder sinifi icinde yapilir; boylece
    Keras Functional API symbolic tensor kisitlamasi asılır.
    """
    # Tum transformer islemleri tek katmanda
    feat = UNeTREncoder(
        embed_dim  = embed_dim,
        num_heads  = num_heads,
        depth      = depth,
        patch_size = patch_size,
        out_filtre = out_filtre,
        name       = "unetr_encoder"
    )(giris)

    # patch_size kadar buyut → omurga cozunurluguyle esle
    feat = layers.UpSampling2D(
        size=(patch_size, patch_size),
        interpolation="bilinear",
        name="trans_upsample")(feat)

    # Ince conv rafine
    feat = depthwise_sep_conv(feat, out_filtre)
    return feat


# ═══════════════════════════════════════════
#  ANA HİBRİT MODEL
# ═══════════════════════════════════════════

def deeplab_unetr_hibrit(img_boyut=512, giris_kanal=3,
                          feature_dim=128, trans_depth=3,
                          trans_heads=4, trans_patch=4):
    """
    DeepLab-UNeTR Hibrit Segmentasyon Modeli

    Veri akisi:
      (512,512,3) giris
           ↓
      Paylasimli CNN omurgasi  [1/8 cozunurluk → 64x64]
           ├───────────────────────────────────┐
      ASPP kolu                        UNeTR kolu
      (yerel cok olcekli)         (global transformer)
           └────────────┬──────────────────────┘
                    SE-Net fuzyonu
                        ↓
                  Ince ayar convlar
                        ↓
                  Bilinear x8 upsample → (512,512)
                        ↓
                  Sigmoid cikis (kas maskesi)

    Neden bu mimari sarkopeni icin uygun?
    - ASPP: farkli kalin/ince kas kesitlerini yakalar
    - Transformer: dis/ic oblique kaslar gibi birbirine
      bagli uzak anatomik yapilarin iliskisini ogrenir
    - SE fuzyonu: modelin kendisi hangi kola guvenecegine
      veriyle karar verir (elle ayarlamak gerekmez)
    """
    giris = keras.Input(shape=(img_boyut, img_boyut, giris_kanal),
                        name="giris_25d")

    # ── 1. Paylasimli CNN Omurgasi ─────────────────
    # Her iki kol ayni omurga cikisini kullanir →
    # tek hesaplama, cift faydasi.
    # Stride=2 adimlarla 1/8 olcege indirir: 512→256→128→64
    x = layers.Conv2D(64, 7, strides=2, padding="same",
                      use_bias=False,
                      kernel_initializer="he_normal",
                      name="stem_conv")(giris)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same",
                             name="stem_pool")(x)    # 512→128
    # Stage 1
    x = depthwise_sep_conv(x, 128)
    x = depthwise_sep_conv(x, 128)
    x = layers.MaxPooling2D(2, name="pool_s1")(x)   # 128→64
    # Stage 2 — dilated (cozunurluk korumak icin stride yok)
    x = depthwise_sep_conv(x, 256)
    x = depthwise_sep_conv(x, 256)
    # Stage 3 — daha genis alici alan
    x = depthwise_sep_conv(x, 512, dilation_rate=2)
    omurga_cikisi = depthwise_sep_conv(x, 512, dilation_rate=2)
    # Cikis boyutu: (B, 64, 64, 512)

    # ── 2. Paralel Kollar ─────────────────────────
    # ASPP kolu
    aspp_cikisi = aspp_modulu(omurga_cikisi,
                               filtre=feature_dim,
                               atrous_rates=(6, 12, 18))
    # Transformer kolu
    trans_cikisi = unetr_kolu(omurga_cikisi,
                               embed_dim=max(feature_dim * 2, 256),
                               depth=trans_depth,
                               num_heads=trans_heads,
                               patch_size=trans_patch,
                               out_filtre=feature_dim)

    # ── 3. Adaptif SE-Net Fuzyonu ─────────────────
    # Boyutlari esitle (gerekirse)
    trans_cikisi = layers.Lambda(
        lambda t: tf.image.resize(
            t[0],
            (tf.shape(t[1])[1], tf.shape(t[1])[2]),
            method="bilinear"),
        name="trans_resize"
    )([trans_cikisi, aspp_cikisi])

    # SE agirlikli ASPP
    aspp_se  = se_block(aspp_cikisi,  feature_dim, name="se_aspp")
    # SE agirlikli Transformer
    trans_se = se_block(trans_cikisi, feature_dim, name="se_trans")

    # Birlestir → 1x1 ile orijinal boyuta sıkistir
    fused = layers.Concatenate(name="concat_fusion")(
        [aspp_se, trans_se])
    fused = layers.Conv2D(feature_dim, 1, use_bias=False,
                          kernel_initializer="he_normal",
                          name="fusion_proj")(fused)
    fused = layers.BatchNormalization(name="fusion_bn")(fused)
    fused = layers.Activation("relu")(fused)

    # ── 4. Ince Ayar Konvolüsyonlari ──────────────
    x = depthwise_sep_conv(fused, feature_dim)
    x = depthwise_sep_conv(x, feature_dim // 2)

    # ── 5. Orijinal Cozunurluge Geri Don ─────────
    # Bilinear upsample x8 (64→512)
    x = layers.UpSampling2D(
        size=(8, 8),
        interpolation="bilinear",
        name="final_upsample")(x)

    # ── 6. Siniflandirma Cikisi ───────────────────
    x = layers.Conv2D(feature_dim // 2, 3, padding="same",
                      use_bias=False,
                      kernel_initializer="he_normal",
                      name="head_conv")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)
    cikis = layers.Conv2D(1, 1, activation="sigmoid",
                          name="maske_cikis")(x)

    return keras.Model(giris, cikis, name="deeplab_unetr_hibrit_512")


# ═══════════════════════════════════════════
#  KAYIP FONKSİYONLARI  (orijinal koddan)
# ═══════════════════════════════════════════

@tf.autograph.experimental.do_not_convert
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
    yt_raw = tf.cast(y_tahmin, tf.float32)
    yt_raw = tf.where(tf.math.is_finite(yt_raw),
                      yt_raw, tf.zeros_like(yt_raw))
    yt = tf.reshape(tf.cast(yt_raw > 0.5, tf.float32), [-1])
    k  = tf.reduce_sum(yg * yt)
    return (2.0*k + smooth) / (
        tf.reduce_sum(yg) + tf.reduce_sum(yt) + smooth)


@tf.autograph.experimental.do_not_convert
def iou_skoru(y_gercek, y_tahmin, smooth=1e-6):
    yg = tf.reshape(tf.cast(y_gercek, tf.float32), [-1])
    yt_raw = tf.cast(y_tahmin, tf.float32)
    yt_raw = tf.where(tf.math.is_finite(yt_raw),
                      yt_raw, tf.zeros_like(yt_raw))
    yt = tf.reshape(tf.cast(yt_raw > 0.5, tf.float32), [-1])
    k  = tf.reduce_sum(yg * yt)
    b  = tf.reduce_sum(yg) + tf.reduce_sum(yt) - k
    return (k + smooth) / (b + smooth)


def birlesik_kayip_olustur(pos_weight: float):
    """
    Agirlikli BCE + Dice loss kombinasyonu.
    BCE: piksel duzeyinde hassas dogruluk
    Dice: sinif dengesizligine (az kas / cok arka plan) karsi saglamlik
    pos_weight: pozitif piksel agirlik katsayisi (1_onisleme.py ciktisi)
    """
    pw = float(pos_weight)

    @tf.autograph.experimental.do_not_convert
    def kayip(y_gercek, y_tahmin):
        yg  = tf.cast(y_gercek, tf.float32)
        yt  = tf.cast(y_tahmin, tf.float32)

        # NaN/Inf koruması: validasyonda NaN gelirse sifira indir
        yg  = tf.where(tf.math.is_finite(yg), yg, tf.zeros_like(yg))
        yt  = tf.where(tf.math.is_finite(yt), yt, tf.zeros_like(yt))

        agirlik = yg * (pw - 1.0) + 1.0   # pozitif piksel pw, negatif 1
        eps  = 1e-7
        yt_k = tf.clip_by_value(yt, eps, 1.0 - eps)
        bce  = -(yg * tf.math.log(yt_k) +
                 (1.0 - yg) * tf.math.log(1.0 - yt_k))
        # BCE'de de NaN koruması
        bce  = tf.where(tf.math.is_finite(bce), bce, tf.zeros_like(bce))
        bce_agirlikli = tf.reduce_mean(agirlik * bce)
        toplam = 0.5 * bce_agirlikli + 0.5 * dice_kayip(yg, yt)
        # Son NaN koruması: NaN cikti ise 0 dondur (egitimi durdurmaz)
        return tf.where(tf.math.is_finite(toplam),
                        toplam, tf.zeros_like(toplam))

    kayip.__name__ = "birlesik_kayip"
    return kayip



# ═══════════════════════════════════════════
#  CANLI GRAFIK CALLBACK
# ═══════════════════════════════════════════

class GrafikCallback(keras.callbacks.Callback):
    """
    Her epoch sonunda metrikleri PNG olarak kaydeden callback.
    4 grafik yan yana:
      1. Kayip (egitim + val)
      2. Dice Skoru (egitim + val)
      3. IoU Skoru  (egitim + val)
      4. Ogrenme Hizi degisimi

    Eger matplotlib interaktif backend destekleniyorsa
    (Spyder, Jupyter, vs.) canli pencere de acar.
    Sunucu ortaminda sadece PNG kaydeder.

    Ayrica her epoch sonunda konsola ozet satirini yazdirir:
      Epoch 05 | Kayip 0.3241 | Dice 0.7812 | IoU 0.6423 | LR 1.0e-04
    """

    RENKLER = {
        "egitim": "#378ADD",      # mavi
        "val":    "#1D9E75",      # yesil
        "en_iyi": "#E24B4A",      # kirmizi
        "lr":     "#BA7517",      # amber
    }

    def __init__(self, kayit_dir: Path, model_adi: str,
                 canli_pencere: bool = False):
        super().__init__()
        self.kayit_dir    = Path(kayit_dir)
        self.model_adi    = model_adi
        self.canli_pencere = canli_pencere
        self.gecmis       = {
            "loss": [], "val_loss": [],
            "dice_skoru": [], "val_dice_skoru": [],
            "iou_skoru":  [], "val_iou_skoru":  [],
            "lr": [],
        }
        self.en_iyi_epoch = 0
        self.en_iyi_dice  = 0.0
        self._fig         = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Metrikleri biriktir
        self.gecmis["loss"].append(float(logs.get("loss", 0)))
        self.gecmis["val_loss"].append(float(logs.get("val_loss", 0)))
        self.gecmis["dice_skoru"].append(
            float(logs.get("dice_skoru", 0)))
        self.gecmis["val_dice_skoru"].append(
            float(logs.get("val_dice_skoru", 0)))
        self.gecmis["iou_skoru"].append(
            float(logs.get("iou_skoru", 0)))
        self.gecmis["val_iou_skoru"].append(
            float(logs.get("val_iou_skoru", 0)))

        # Mevcut ogrenme hizini al
        try:
            lr = float(
                keras.backend.get_value(self.model.optimizer.lr))
        except Exception:
            lr = 0.0
        self.gecmis["lr"].append(lr)

        # En iyi epoch takibi
        val_dice = self.gecmis["val_dice_skoru"][-1]
        if val_dice > self.en_iyi_dice:
            self.en_iyi_dice  = val_dice
            self.en_iyi_epoch = epoch + 1

        # Konsol ozet
        print(
            "  Epoch {:03d} | "
            "Kayip {:.4f}/{:.4f} | "
            "Dice {:.4f}/{:.4f} | "
            "IoU {:.4f}/{:.4f} | "
            "LR {:.2e}{}".format(
                epoch + 1,
                self.gecmis["loss"][-1],
                self.gecmis["val_loss"][-1],
                self.gecmis["dice_skoru"][-1],
                self.gecmis["val_dice_skoru"][-1],
                self.gecmis["iou_skoru"][-1],
                self.gecmis["val_iou_skoru"][-1],
                lr,
                " ★ EN IYI" if (epoch + 1) == self.en_iyi_epoch
                else ""
            )
        )

        # Grafigi guncelle ve kaydet
        self._grafik_ciz(epoch + 1)

    def _grafik_ciz(self, epoch_no: int):
        ep   = list(range(1, epoch_no + 1))
        r    = self.RENKLER
        g    = self.gecmis
        eiy  = self.en_iyi_epoch

        fig = plt.figure(figsize=(20, 10))
        fig.patch.set_facecolor("#0F1117")

        gs = gridspec.GridSpec(
            2, 4,
            figure=fig,
            hspace=0.45, wspace=0.35,
            left=0.06, right=0.97,
            top=0.88, bottom=0.10
        )

        # Baslik
        fig.suptitle(
            "DeepLab-UNeTR Egitim Izleme — {}  "
            "(Epoch {})".format(self.model_adi, epoch_no),
            fontsize=13, color="white", fontweight="bold", y=0.96
        )

        # ── Stil yardimcisi ───────────────────────
        def stil_ax(ax, baslik, ylabel="", ylim=None):
            ax.set_facecolor("#1A1D27")
            ax.set_title(baslik, color="white",
                         fontsize=10, pad=6)
            ax.set_xlabel("Epoch", color="#9090A0", fontsize=8)
            ax.set_ylabel(ylabel, color="#9090A0", fontsize=8)
            ax.tick_params(colors="#9090A0", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333345")
            ax.grid(True, color="#2A2D3A", linewidth=0.6,
                    linestyle="--", alpha=0.7)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if ylim:
                ax.set_ylim(*ylim)

        def ciz_cift(ax, eg_veri, val_veri, baslik,
                     ylabel="", ylim=None, en_iyi_dikey=True):
            stil_ax(ax, baslik, ylabel, ylim)
            ax.plot(ep, eg_veri,  color=r["egitim"],
                    lw=1.8, label="Egitim", alpha=0.9)
            ax.plot(ep, val_veri, color=r["val"],
                    lw=1.8, label="Validasyon", alpha=0.9)
            # Dolgu
            ax.fill_between(ep, eg_veri,  alpha=0.10,
                            color=r["egitim"])
            ax.fill_between(ep, val_veri, alpha=0.10,
                            color=r["val"])
            # En iyi epoch dikey cizgisi
            if en_iyi_dikey and eiy <= len(ep):
                ax.axvline(eiy, color=r["en_iyi"],
                           lw=1.2, ls="--", alpha=0.7,
                           label="En iyi ({})".format(eiy))
            # Son deger etiketi
            ax.annotate(
                "{:.4f}".format(val_veri[-1]),
                xy=(ep[-1], val_veri[-1]),
                xytext=(5, 0), textcoords="offset points",
                color=r["val"], fontsize=7, va="center"
            )
            lgd = ax.legend(fontsize=7, framealpha=0.3,
                            labelcolor="white",
                            facecolor="#1A1D27",
                            edgecolor="#444455")

        # ── Satir 1: Ana metrikler (3 panel) ─────
        ax_kayip = fig.add_subplot(gs[0, 0])
        ciz_cift(ax_kayip,
                 g["loss"], g["val_loss"],
                 "Kayip (BCE + Dice)", "Kayip")

        ax_dice = fig.add_subplot(gs[0, 1])
        ciz_cift(ax_dice,
                 g["dice_skoru"], g["val_dice_skoru"],
                 "Dice Skoru", "Dice", ylim=(0, 1.02))

        ax_iou = fig.add_subplot(gs[0, 2])
        ciz_cift(ax_iou,
                 g["iou_skoru"], g["val_iou_skoru"],
                 "IoU Skoru", "IoU", ylim=(0, 1.02))

        # ── Ogrenme Hizi (4. panel, satir 1) ─────
        ax_lr = fig.add_subplot(gs[0, 3])
        stil_ax(ax_lr, "Ogrenme Hizi", "LR")
        ax_lr.semilogy(ep, g["lr"],
                       color=r["lr"], lw=1.8, alpha=0.9)
        ax_lr.fill_between(ep, g["lr"],
                           alpha=0.15, color=r["lr"])
        ax_lr.tick_params(colors="#9090A0", labelsize=7)
        ax_lr.annotate(
            "{:.2e}".format(g["lr"][-1]),
            xy=(ep[-1], g["lr"][-1]),
            xytext=(5, 0), textcoords="offset points",
            color=r["lr"], fontsize=7, va="center"
        )

        # ── Satir 2: Delta ve bant grafikleri ────
        # Dice farki (val - egitim): overfitting gostergesi
        ax_delta = fig.add_subplot(gs[1, 0])
        stil_ax(ax_delta, "Val-Egitim Dice Farki",
                "val_dice - egitim_dice")
        delta = [v - e for v, e in
                 zip(g["val_dice_skoru"], g["dice_skoru"])]
        renkler_delta = [r["val"] if d >= 0 else r["en_iyi"]
                         for d in delta]
        ax_delta.bar(ep, delta, color=renkler_delta, alpha=0.75)
        ax_delta.axhline(0, color="#555566",
                         lw=0.8, ls="--")
        ax_delta.annotate(
            "Pozitif = val > egitim  |  Negatif = ezber",
            xy=(0.02, 0.92), xycoords="axes fraction",
            color="#9090A0", fontsize=6, va="top"
        )

        # Hareketli ortalama Dice (5 epoch)
        ax_ma = fig.add_subplot(gs[1, 1])
        stil_ax(ax_ma, "Val Dice (5-Epoch Ort.)",
                "Dice", ylim=(0, 1.02))
        vd = g["val_dice_skoru"]
        if len(vd) >= 5:
            ma = [float(np.mean(vd[max(0, i-4):i+1]))
                  for i in range(len(vd))]
            ax_ma.plot(ep, vd, color=r["val"],
                       lw=0.8, alpha=0.4, label="Ham")
            ax_ma.plot(ep, ma, color="#FCDE5A",
                       lw=2.0, label="5-ep ortalama")
            ax_ma.legend(fontsize=7, framealpha=0.3,
                         labelcolor="white",
                         facecolor="#1A1D27",
                         edgecolor="#444455")
        else:
            ax_ma.plot(ep, vd, color=r["val"], lw=1.8)

        # Kayip - IoU ikili eksen
        ax_li = fig.add_subplot(gs[1, 2])
        stil_ax(ax_li, "Val Kayip  vs  Val IoU")
        ln1 = ax_li.plot(ep, g["val_loss"],
                         color=r["egitim"], lw=1.6,
                         label="Val Kayip")
        ax_li.set_ylabel("Kayip", color=r["egitim"], fontsize=8)
        ax_li.tick_params(axis="y", colors=r["egitim"])
        ax2 = ax_li.twinx()
        ax2.set_facecolor("#1A1D27")
        ln2 = ax2.plot(ep, g["val_iou_skoru"],
                       color=r["val"], lw=1.6,
                       label="Val IoU")
        ax2.set_ylabel("IoU", color=r["val"], fontsize=8)
        ax2.tick_params(axis="y", colors=r["val"],
                        labelsize=7)
        ax2.set_ylim(0, 1.02)
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax_li.legend(lns, labs, fontsize=7,
                     framealpha=0.3, labelcolor="white",
                     facecolor="#1A1D27",
                     edgecolor="#444455")

        # Ozet istatistik kutusu (4. panel, satir 2)
        ax_ozet = fig.add_subplot(gs[1, 3])
        ax_ozet.set_facecolor("#1A1D27")
        ax_ozet.axis("off")
        ax_ozet.set_title("Anlık Ozet", color="white",
                           fontsize=10, pad=6)

        vd_son  = g["val_dice_skoru"]
        vi_son  = g["val_iou_skoru"]
        vl_son  = g["val_loss"]

        ozet_satirlari = [
            ("Epoch",         "{} / {}".format(
                epoch_no, EPOCHS)),
            ("En iyi epoch",  "{}".format(eiy)),
            ("En iyi Dice",   "{:.4f}".format(self.en_iyi_dice)),
            ("",              ""),
            ("Val Dice",
             "{:.4f}  (max {:.4f})".format(
                 vd_son[-1], max(vd_son))),
            ("Val IoU",
             "{:.4f}  (max {:.4f})".format(
                 vi_son[-1], max(vi_son))),
            ("Val Kayip",
             "{:.4f}  (min {:.4f})".format(
                 vl_son[-1], min(vl_son))),
            ("",              ""),
            ("Ogrenme Hizi",  "{:.2e}".format(g["lr"][-1])),
            ("LR Dususu",
             "{} kez".format(
                 sum(1 for i in range(1, len(g["lr"]))
                     if g["lr"][i] < g["lr"][i-1]))),
        ]

        y_pos = 0.96
        for etiket, deger in ozet_satirlari:
            if not etiket:
                y_pos -= 0.04
                continue
            ax_ozet.text(0.02, y_pos, etiket + ":",
                         transform=ax_ozet.transAxes,
                         color="#9090A0", fontsize=8,
                         va="top")
            ax_ozet.text(0.48, y_pos, deger,
                         transform=ax_ozet.transAxes,
                         color="white", fontsize=8,
                         va="top", fontweight="bold")
            y_pos -= 0.09

        # Kaydet
        grafik_yolu = self.kayit_dir / "egitim_canli_grafik.png"
        fig.savefig(str(grafik_yolu), dpi=120,
                    facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        plt.close(fig)

    def on_train_end(self, logs=None):
        print("\nSon grafik kaydedildi:",
              self.kayit_dir / "egitim_canli_grafik.png")


# ═══════════════════════════════════════════
#  EGITIM SONU KAPSAMLI GRAFIK
# ═══════════════════════════════════════════

def egitim_grafigi_kaydet(gecmis, model_adi, model_kayit):
    """
    Egitim tamamlandiktan sonra kaydedilen son, temiz grafik.
    3 panel + en iyi epoch isaretleri + ızgara + dolgu.
    """
    ep     = list(range(1, len(gecmis.history["loss"]) + 1))
    en_iyi = int(np.argmax(gecmis.history["val_dice_skoru"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0F1117")
    fig.suptitle(
        "DeepLab-UNeTR — {}  "
        "(En Iyi Epoch: {})".format(model_adi, en_iyi + 1),
        fontsize=13, color="white", fontweight="bold"
    )

    PANEL = [
        ("Kayip (BCE + Dice)",
         "loss", "val_loss", "Kayip", None),
        ("Dice Skoru",
         "dice_skoru", "val_dice_skoru", "Dice", (0, 1.02)),
        ("IoU Skoru",
         "iou_skoru", "val_iou_skoru", "IoU", (0, 1.02)),
    ]

    for ax, (baslik, eg_k, val_k, ylabel, ylim) in             zip(axes, PANEL):
        eg_v  = gecmis.history[eg_k]
        val_v = gecmis.history[val_k]

        ax.set_facecolor("#1A1D27")
        ax.set_title(baslik, color="white", fontsize=11)
        ax.set_xlabel("Epoch", color="#9090A0")
        ax.set_ylabel(ylabel, color="#9090A0")
        ax.tick_params(colors="#9090A0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333345")
        ax.grid(True, color="#2A2D3A", linestyle="--",
                alpha=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if ylim:
            ax.set_ylim(*ylim)

        ax.plot(ep, eg_v,  color="#378ADD", lw=2,
                label="Egitim", alpha=0.9)
        ax.plot(ep, val_v, color="#1D9E75", lw=2,
                label="Validasyon", alpha=0.9)
        ax.fill_between(ep, eg_v,  alpha=0.10,
                        color="#378ADD")
        ax.fill_between(ep, val_v, alpha=0.10,
                        color="#1D9E75")
        ax.axvline(en_iyi + 1, color="#E24B4A",
                   lw=1.5, ls="--", alpha=0.8,
                   label="En iyi ({})".format(en_iyi + 1))

        lgd = ax.legend(fontsize=9, framealpha=0.3,
                        labelcolor="white",
                        facecolor="#1A1D27",
                        edgecolor="#444455")

    plt.tight_layout()
    dosya = model_kayit / "egitim_grafigi_final.png"
    plt.savefig(str(dosya), dpi=130,
                facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close()
    print("Final grafik kaydedildi:", dosya)

# ═══════════════════════════════════════════
#  EĞİTİM
# ═══════════════════════════════════════════


def egit():
    zaman     = datetime.now().strftime("%Y%m%d_%H%M")
    model_adi = "dlunetr_512_{}".format(zaman)
    model_kayit = MODEL_DIR / model_adi
    model_kayit.mkdir(exist_ok=True)

    print("=" * 65)
    print("DeepLab-UNeTR Hibrit Egitimi — 512x512 — GPU")
    print("Model klasoru:", model_kayit)
    print("=" * 65)

    # Veri
    egitim_ds = SarcopeniDataset("egitim", BATCH_SIZE,
                                  augment=True,  karistir=True)
    val_ds    = SarcopeniDataset("test",   BATCH_SIZE,
                                  augment=False, karistir=False)

    # ── Veri Tanilama (NaN / boyut kontrolu) ─────
    print("Veri tanilama yapiliyor...")
    x0, y0 = egitim_ds[0]
    xv, yv = val_ds[0]
    assert x0.shape == (BATCH_SIZE, IMG_BOYUT, IMG_BOYUT, 3), \
        "Egitim X boyutu yanlis: {}".format(x0.shape)
    assert y0.shape == (BATCH_SIZE, IMG_BOYUT, IMG_BOYUT, 1), \
        "Egitim Y boyutu yanlis: {}".format(y0.shape)
    assert xv.shape == (BATCH_SIZE, IMG_BOYUT, IMG_BOYUT, 3), \
        "Val X boyutu yanlis: {}".format(xv.shape)
    assert yv.shape == (BATCH_SIZE, IMG_BOYUT, IMG_BOYUT, 1), \
        "Val Y boyutu yanlis: {}".format(yv.shape)
    assert not np.any(np.isnan(x0)), "Egitim X NaN iceriyor!"
    assert not np.any(np.isnan(y0)), "Egitim Y NaN iceriyor!"
    assert not np.any(np.isnan(xv)), "Val X NaN iceriyor!"
    assert not np.any(np.isnan(yv)), "Val Y NaN iceriyor!"
    print("  Egitim batch: X{} Y{}  min={:.3f} max={:.3f}".format(
        x0.shape, y0.shape, float(x0.min()), float(x0.max())))
    print("  Val    batch: X{} Y{}  min={:.3f} max={:.3f}".format(
        xv.shape, yv.shape, float(xv.min()), float(xv.max())))
    print("  Egitim batch sayisi: {}  |  Val batch sayisi: {}".format(
        len(egitim_ds), len(val_ds)))
    print("Veri tanilama TAMAM.\n")

    # Model
    with tf.device("/GPU:0"):
        model = deeplab_unetr_hibrit(
            img_boyut   = IMG_BOYUT,
            giris_kanal = 3,
            feature_dim = FEATURE_DIM,
            trans_depth = TRANS_DEPTH,
            trans_heads = TRANS_HEADS,
            trans_patch = TRANS_PATCH,
        )

    model.summary(line_length=100)
    print("\nToplam parametre: {:,}".format(model.count_params()))

    kayip_fn = birlesik_kayip_olustur(POS_WEIGHT)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss=kayip_fn,
        metrics=[dice_skoru, iou_skoru]
    )

    # ── Callback'ler ──────────────────────────────
    cb = [
        keras.callbacks.ModelCheckpoint(
            str(model_kayit / "en_iyi.keras"),
            monitor    = "val_dice_skoru",
            mode       = "max",
            save_best_only = True,
            verbose    = 1
        ),
        keras.callbacks.EarlyStopping(
            monitor  = "val_dice_skoru",
            mode     = "max",
            patience = 15,
            restore_best_weights = True,
            verbose  = 1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_dice_skoru",
            mode     = "max",
            factor   = 0.5,
            patience = 7,
            min_lr   = 1e-7,
            verbose  = 1
        ),
        keras.callbacks.CSVLogger(
            str(model_kayit / "egitim_log.csv")
        ),
        # Ara model yedegi (her 10 epoch)
        keras.callbacks.ModelCheckpoint(
            str(model_kayit / "epoch_{epoch:03d}.keras"),
            save_freq  = "epoch",
            period     = 10,
            save_best_only = False,
            verbose    = 0
        ),
        # Her epoch sonunda grafik guncelle ve PNG kaydet
        GrafikCallback(
            kayit_dir    = model_kayit,
            model_adi    = model_adi,
            canli_pencere = False,   # Spyder/Jupyter'da True yapabilirsiniz
        ),
    ]

    # ── Eğitim ────────────────────────────────────
    gecmis = model.fit(
        egitim_ds,
        validation_data = val_ds,
        epochs  = EPOCHS,
        callbacks = cb,
        verbose = 1
    )

    # ── Grafikler ─────────────────────────────────
    # Canli callback'in urettigi grafik zaten kaydedildi.
    # Ek olarak temiz "final" grafik olustur:
    egitim_grafigi_kaydet(gecmis, model_adi, model_kayit)

    # ── Meta Veri ─────────────────────────────────
    en_iyi = int(np.argmax(gecmis.history["val_dice_skoru"]))
    meta   = {
        "model_adi":        model_adi,
        "mimari":           "DeepLab-UNeTR Hibrit",
        "img_boyut":        IMG_BOYUT,
        "batch_size":       BATCH_SIZE,
        "feature_dim":      FEATURE_DIM,
        "trans_depth":      TRANS_DEPTH,
        "trans_heads":      TRANS_HEADS,
        "trans_patch":      TRANS_PATCH,
        "epochs_calisilan": len(gecmis.history["loss"]),
        "en_iyi_epoch":     en_iyi + 1,
        "val_dice":         round(gecmis.history["val_dice_skoru"][en_iyi], 4),
        "val_iou":          round(gecmis.history["val_iou_skoru"][en_iyi],  4),
        "val_loss":         round(gecmis.history["val_loss"][en_iyi],       4),
        "pos_weight":       POS_WEIGHT,
        "lr":               LR,
        "toplam_parametre": model.count_params(),
    }
    with open(str(model_kayit / "egitim_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("EGİTİM TAMAMLANDI")
    print("  En iyi epoch  : {}".format(meta["en_iyi_epoch"]))
    print("  Val Dice      : {}".format(meta["val_dice"]))
    print("  Val IoU       : {}".format(meta["val_iou"]))
    print("  Parametreler  : {:,}".format(meta["toplam_parametre"]))
    print("  Model         :", model_kayit / "en_iyi.keras")
    print("=" * 65)

    return model, model_kayit


if __name__ == "__main__":
    egit()