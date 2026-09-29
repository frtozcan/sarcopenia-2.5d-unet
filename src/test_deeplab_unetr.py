"""
ADIM 3 -- DeepLab-UNeTR Model Testi ve Degerlendirme
=====================================================
Her hasta icin 4 panelli OpenCV gorseli:
  1. Orijinal CT (merkez kesit)
  2. Gercek maske (yesil overlay)
  3. Tahmin maskesi (turuncu overlay)
  4. Ortusme (yesil=TP, kirmizi=FP, mavi=FN)

Bos SMI nedeni: ilgili hastalarin DICOM dosyasinda
PatientSize etiketi bos -> boy bilinemez -> SMI hesaplanamaz.
Bu hastalar icin boy bilgisi elle girilebilir (BOY_MANUEL sozlugu).

Calistirma:
  python 3_test_dlunetr.py

  ya da belirli model icin:
  python src/test_deeplab_unetr.py --model "<SARKOPENI_DIR>/DLUnetR/dlunetr_512_.../en_iyi.keras"
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -----------------------------------------
#  GPU AYARLARI
# -----------------------------------------
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# -----------------------------------------
#  AYARLAR
# -----------------------------------------
# Proje kok klasoru: SARKOPENI_DIR ortam degiskeniyle degistirilebilir
BASE_DIR = Path(os.environ.get("SARKOPENI_DIR", r"E:\sarkopeni"))

DLUNE_DIR  = BASE_DIR / "DLUnetR"        # model arama koku
EGITIM_DIR = BASE_DIR / "output" / "egitim_25d"
NPY_DIR    = BASE_DIR / "output" / "npy"
ESIK       = 0.5   # binary esik (threshold analizi sonucuna gore guncelleyin)

# Boy bilgisi DICOM'da olmayan hastalar icin manuel giris
# Ornek: "hasta_klasor_adi": 175
BOY_MANUEL = {
    # "hasta_klasor_adi": 172,   # DICOM'da boyu olmayan hastalar icin (cm)
}


# ===========================================
#  KAYIP / METRIK -- MODEL YUKLEMEK ICIN GEREKLI
#  (DeepLab-UNeTR egitiminde kullanilan fonksiyonlarin ayni)
# ===========================================

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


@tf.autograph.experimental.do_not_convert
def birlesik_kayip(y_gercek, y_tahmin, smooth=1e-6):
    """
    Egitimde pos_weight ile olusturulan kayip fonksiyonunun
    test sirasinda model.load icin gerekli basitlestirilmis hali.
    """
    yg = tf.cast(y_gercek, tf.float32)
    yt = tf.cast(y_tahmin, tf.float32)
    k  = tf.reduce_sum(yg * yt)
    return 1.0 - (2.0*k + smooth) / (
        tf.reduce_sum(yg) + tf.reduce_sum(yt) + smooth)


# ===========================================
#  MODEL BULMA
# ===========================================

def son_modeli_bul(model_dir: Path) -> Path:
    """DLUnetR klasorundeki en son egitilmis modeli bulur."""
    modeller = sorted(model_dir.glob("dlunetr_*/en_iyi.keras"))
    if not modeller:
        raise FileNotFoundError(
            "{} icinde 'dlunetr_*/en_iyi.keras' bulunamadi.\n"
            "Once 2_egitim_dlunetr.py calistirin.".format(model_dir))
    print("Bulunan model:", modeller[-1])
    return modeller[-1]


# ===========================================
#  METRIKLER (numpy, piksel bazli)
# ===========================================

def metrikleri_hesapla(y_gercek, y_tahmin, esik=0.5):
    yg = (y_gercek.flatten() > 0.5).astype(np.float32)
    yt = (y_tahmin.flatten() > esik).astype(np.float32)
    tp = float(np.sum(yg * yt))
    fp = float(np.sum((1 - yg) * yt))
    fn = float(np.sum(yg * (1 - yt)))
    tn = float(np.sum((1 - yg) * (1 - yt)))
    s  = 1e-6
    return {
        "dice":       round((2*tp+s) / (2*tp+fp+fn+s), 4),
        "iou":        round((tp+s)   / (tp+fp+fn+s),   4),
        "hassasiyet": round((tp+s)   / (tp+fn+s),      4),
        "ozgulluk":   round((tn+s)   / (tn+fp+s),      4),
        "kesinlik":   round((tp+s)   / (tp+fp+s),      4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ===========================================
#  THRESHOLD ANALIZI
# ===========================================

def threshold_analizi(y_gercek_list, y_tahmin_list,
                       kayit_dir: Path, esikler=None):
    if esikler is None:
        esikler = np.arange(0.1, 0.95, 0.05)

    dice_ort, iou_ort = [], []
    for esik in esikler:
        d_list, i_list = [], []
        for yg, yt in zip(y_gercek_list, y_tahmin_list):
            m = metrikleri_hesapla(yg, yt, esik=float(esik))
            d_list.append(m["dice"])
            i_list.append(m["iou"])
        dice_ort.append(float(np.mean(d_list)))
        iou_ort.append(float(np.mean(i_list)))

    en_iyi_idx  = int(np.argmax(dice_ort))
    en_iyi_esik = float(esikler[en_iyi_idx])
    en_iyi_dice = float(dice_ort[en_iyi_idx])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(esikler, dice_ort, "b-o", ms=4, label="Dice")
    ax.plot(esikler, iou_ort,  "g-s", ms=4, label="IoU")
    ax.axvline(en_iyi_esik, color="red", ls="--",
               label="En iyi esik = {:.2f} (Dice={:.4f})".format(
                   en_iyi_esik, en_iyi_dice))
    ax.set_xlabel("Esik degeri")
    ax.set_ylabel("Skor")
    ax.set_title("Threshold Analizi -- DeepLab-UNeTR")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(kayit_dir / "threshold_analizi.png"), dpi=130)
    plt.close()
    print("  En iyi esik: {:.2f}  ->  Dice={:.4f}".format(
        en_iyi_esik, en_iyi_dice))
    return en_iyi_esik, en_iyi_dice


# ===========================================
#  SMI KARSILASTIRMA
# ===========================================

def smi_karsilastir(hasta_key, tahmin_maskeleri, spacing_mm, boy_cm):
    SMI_ESIK = {"e": 49.0, "k": 31.0}
    meta_yolu  = NPY_DIR / hasta_key / "meta.json"
    gercek_smi = None
    cinsiyet   = "e"

    if meta_yolu.exists():
        with open(str(meta_yolu), encoding="utf-8") as f:
            meta = json.load(f)
        gercek_smi = meta.get("smi")
        cinsiyet   = (meta.get("cinsiyet") or "E").lower()
        if not boy_cm:
            boy_cm = meta.get("boy_cm")

    if not boy_cm:
        boy_cm = BOY_MANUEL.get(hasta_key)

    if not tahmin_maskeleri or not boy_cm:
        return {
            "gercek_smi": gercek_smi, "tahmin_smi": None,
            "fark": None, "fark_pct": None,
            "gercek_sark": None, "tahmin_sark": None,
            "smi_esik": None,
        }

    px_cm2     = (spacing_mm[0] / 10.0) * (spacing_mm[1] / 10.0)
    alanlar    = [float(m.sum()) * px_cm2
                  for m in tahmin_maskeleri.values()]
    ort_alan   = float(np.mean(alanlar))
    tahmin_smi = round(ort_alan / (boy_cm / 100.0) ** 2, 2)
    esik_val   = SMI_ESIK.get(cinsiyet, 49.0)
    fark       = round(tahmin_smi - gercek_smi, 2) if gercek_smi else None
    fark_pct   = (round(abs(fark) / gercek_smi * 100, 1)
                  if (fark and gercek_smi) else None)

    return {
        "gercek_smi":  gercek_smi,
        "tahmin_smi":  tahmin_smi,
        "fark":        fark,
        "fark_pct":    fark_pct,
        "smi_esik":    esik_val,
        "gercek_sark": ("EVET" if (gercek_smi and gercek_smi < esik_val)
                        else "HAYIR"),
        "tahmin_sark": "EVET" if tahmin_smi < esik_val else "HAYIR",
    }


# ===========================================
#  4 PANELLI GORSEL -- OpenCV
# ===========================================

def gorsel_goster_opencv(hasta_key, gorsel_dir,
                          x_ornekler, y_gercek_list,
                          y_tahmin_list, metrik_list,
                          smi_bilgi, max_gorsel=3):
    """
    L3-1, L3 merkez, L3+1 dilimleri icin 4 panelli gorsel olusturur.
    OpenCV penceresi acar ve PNG olarak kaydeder.
    Herhangi bir tus: sonraki hasta  |  ESC: cikis
    """
    n = min(max_gorsel, len(x_ornekler))
    if n == 0:
        return True

    panel_w, panel_h = 800, 800
    bosluk = 6

    # SMI baslik metni
    smi_str = ""
    if smi_bilgi.get("gercek_smi") and smi_bilgi.get("tahmin_smi"):
        smi_str = (
            "  SMI: Gercek={:.1f}  Tahmin={:.1f}  Fark={:.1f}".format(
                smi_bilgi["gercek_smi"],
                smi_bilgi["tahmin_smi"],
                smi_bilgi.get("fark") or 0))

    ana_baslik = "{}{}".format(
        hasta_key.replace("_", " ").upper(), smi_str)
    panel_basliklar = [
        "Orijinal CT",
        "Gercek Maske",
        "Tahmin Maskesi",
        "Ortusme (Y=TP R=FP B=FN)",
    ]
    etiketler = ["L3 - 1", "L3 (merkez)", "L3 + 1"]

    satir_listesi = []

    for satir in range(n):
        x         = x_ornekler[satir]
        yg        = y_gercek_list[satir][:, :, 0]
        yt_olasil = y_tahmin_list[satir][:, :, 0]
        yt        = (yt_olasil > ESIK).astype(np.float32)
        m         = metrik_list[satir]

        ct_u8  = (x[:, :, 1] * 255).astype(np.uint8)   # merkez dilim
        ct_bgr = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)
        ct_bgr = cv2.resize(ct_bgr, (panel_w, panel_h))

        def overlay(maske, renk_bgr, alfa=0.5):
            base   = ct_bgr.copy()
            m_r    = cv2.resize((maske * 255).astype(np.uint8),
                                (panel_w, panel_h))
            renkli = np.zeros_like(base)
            renkli[m_r > 127] = renk_bgr
            return cv2.addWeighted(base, 1.0, renkli, alfa, 0)

        p1 = ct_bgr.copy()
        p2 = overlay(yg, [0, 200, 50],  alfa=0.5)    # gercek -- yesil
        p3 = overlay(yt, [0, 140, 255], alfa=0.5)    # tahmin -- turuncu

        # Ortusme paneli
        tp_m = ((yg > 0.5) & (yt > 0.5)).astype(np.uint8)
        fp_m = ((yg < 0.5) & (yt > 0.5)).astype(np.uint8)
        fn_m = ((yg > 0.5) & (yt < 0.5)).astype(np.uint8)
        p4_r = cv2.resize(ct_bgr.copy(), (panel_w, panel_h))
        for maske, renk in [(tp_m, [0, 220, 50]),
                             (fp_m, [0, 50, 220]),
                             (fn_m, [200, 50, 50])]:
            m_r    = cv2.resize((maske * 255).astype(np.uint8),
                                (panel_w, panel_h))
            renkli = np.zeros_like(p4_r)
            renkli[m_r > 127] = renk
            p4_r = cv2.addWeighted(p4_r, 1.0, renkli, 0.55, 0)

        # Yazılar
        cv2.putText(
            p3,
            "Dice:{:.3f}  IoU:{:.3f}".format(m["dice"], m["iou"]),
            (8, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(
            p4_r,
            "TP:{}  FP:{}  FN:{}".format(m["tp"], m["fp"], m["fn"]),
            (8, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
            0.70, (255, 255, 255), 2, cv2.LINE_AA)

        etiket = (etiketler[satir] if satir < len(etiketler)
                  else "Dilim {}".format(satir + 1))
        cv2.putText(p1, etiket, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (200, 220, 255), 2, cv2.LINE_AA)

        # Model etiketi
        cv2.putText(p1, "DeepLab-UNeTR Hibrit",
                    (8, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (150, 200, 255), 1, cv2.LINE_AA)

        bos = np.full((panel_h, bosluk, 3), 40, dtype=np.uint8)
        satir_listesi.append(
            np.hstack([p1, bos, p2, bos, p3, bos, p4_r]))

    # Dikey birlestir
    bos_satir = np.full(
        (bosluk, satir_listesi[0].shape[1], 3), 40, dtype=np.uint8)
    gorsel = satir_listesi[0]
    for s in satir_listesi[1:]:
        gorsel = np.vstack([gorsel, bos_satir, s])

    # Panel basliklari
    pb_bant = np.full((60, gorsel.shape[1], 3), 30, dtype=np.uint8)
    x_pos = 0
    for pb in panel_basliklar:
        cx = x_pos + panel_w // 2
        cv2.putText(pb_bant, pb, (cx - len(pb) * 4, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80,
                    (180, 210, 255), 2, cv2.LINE_AA)
        x_pos += panel_w + bosluk

    # Ana baslik
    ab_bant = np.full((70, gorsel.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(ab_bant, ana_baslik, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (220, 230, 255), 2, cv2.LINE_AA)

    # Alt bilgi
    alt_bant = np.full((45, gorsel.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(
        alt_bant,
        "Herhangi bir tusa basin: sonraki hasta  |  ESC: cikis",
        (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
        (150, 150, 150), 2, cv2.LINE_AA)

    final = np.vstack([ab_bant, pb_bant, gorsel, alt_bant])

    # Kaydet
    kayit_yolu = gorsel_dir / "{}_l3.png".format(hasta_key[:35])
    cv2.imwrite(str(kayit_yolu), final)
    print("  Gorsel kaydedildi:", kayit_yolu.name)

    # Goster
    pencere = "DLUNeTR Testi - {}".format(hasta_key[:40])
    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(pencere,
                     min(final.shape[1], 1920),
                     min(final.shape[0], 1040))
    cv2.imshow(pencere, final)
    tus = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(pencere)

    return tus != 27   # ESC = dur


# ===========================================
#  EXCEL RAPORU
# ===========================================

def excel_rapor_kaydet(df, dosya_yolu):
    wb = Workbook()
    ws = wb.active
    ws.title = "Test Sonuclari"

    BASLIK = PatternFill("solid", fgColor="1F4E79")
    IYI    = PatternFill("solid", fgColor="E2EFDA")
    ORTA   = PatternFill("solid", fgColor="FFF2CC")
    KOTU   = PatternFill("solid", fgColor="FCE4D6")
    BF     = Font(bold=True, color="FFFFFF", size=10)
    NF     = Font(size=10)
    C      = Alignment(horizontal="center", vertical="center")
    L      = Alignment(horizontal="left",   vertical="center")
    k      = Side(style="thin", color="BFBFBF")
    KN     = Border(left=k, right=k, top=k, bottom=k)

    SUTUNLAR = [
        ("No",             "no",          5,  "c"),
        ("Hasta",          "hasta",       26, "l"),
        ("Mimari",         "mimari",      20, "l"),
        ("Dilim Sayisi",   "dilim_say",   13, "c"),
        ("Dice",           "dice",         9, "c"),
        ("IoU",            "iou",          9, "c"),
        ("Hassasiyet",     "hassasiyet",  13, "c"),
        ("Ozgulluk",       "ozgulluk",    13, "c"),
        ("Kesinlik",       "kesinlik",    13, "c"),
        ("Gercek SMI",     "gercek_smi",  13, "c"),
        ("Tahmin SMI",     "tahmin_smi",  13, "c"),
        ("SMI Fark",       "fark",        11, "c"),
        ("Fark %",         "fark_pct",    10, "c"),
        ("Gercek Sark.",   "gercek_sark", 14, "c"),
        ("Tahmin Sark.",   "tahmin_sark", 14, "c"),
        ("Sinif Eslesti?", "eslesti",     14, "c"),
    ]

    ws.row_dimensions[1].height = 22
    for col_idx, (bas, _, gen, _) in enumerate(SUTUNLAR, start=1):
        h = ws.cell(row=1, column=col_idx, value=bas)
        h.fill = BASLIK; h.font = BF
        h.alignment = C; h.border = KN
        ws.column_dimensions[get_column_letter(col_idx)].width = gen

    for satir_no, (_, veri) in enumerate(df.iterrows(), start=2):
        ws.row_dimensions[satir_no].height = 17
        dice  = float(veri.get("dice") or 0)
        dolgu = IYI if dice >= 0.85 else ORTA if dice >= 0.70 else KOTU

        degerler = [satir_no - 1]
        for _, alan, _, _ in SUTUNLAR[1:]:
            degerler.append(veri.get(alan, ""))

        for col_idx, (deger, (_, _, _, hiz)) in enumerate(
                zip(degerler, SUTUNLAR), start=1):
            c = ws.cell(row=satir_no, column=col_idx, value=deger)
            c.font = NF; c.fill = dolgu; c.border = KN
            c.alignment = L if hiz == "l" else C

    # Ortalama satiri
    son = len(df) + 2
    ws.row_dimensions[son].height = 20
    ws.cell(row=son, column=1, value="ORTALAMA").font = \
        Font(bold=True, size=10)
    for col_idx, (_, alan, _, _) in enumerate(SUTUNLAR[1:], start=2):
        if alan in ("dice", "iou", "hassasiyet", "ozgulluk",
                    "kesinlik", "fark", "fark_pct"):
            col_data = df[alan].dropna()
            if len(col_data) > 0:
                c = ws.cell(row=son, column=col_idx,
                            value=round(float(col_data.mean()), 4))
                c.font = Font(bold=True, size=10)
                c.alignment = C

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = "A1:{}{}".format(
        get_column_letter(len(SUTUNLAR)), len(df) + 1)
    wb.save(str(dosya_yolu))
    print("  Excel:", dosya_yolu)


# ===========================================
#  ANA TEST DONGUSU
# ===========================================

def test_et(model_yolu: Path):
    model_klasor = model_yolu.parent
    zaman        = datetime.now().strftime("%Y%m%d_%H%M")
    sonuc_dir    = model_klasor / "test_sonuclari"
    gorsel_dir   = sonuc_dir / "gorseller"
    sonuc_dir.mkdir(parents=True, exist_ok=True)
    gorsel_dir.mkdir(exist_ok=True)

    print("=" * 65)
    print("DeepLab-UNeTR Hibrit -- Model Testi")
    print("Model   :", model_yolu)
    print("Sonuclar:", sonuc_dir)
    print("=" * 65)

    # Model meta (pos_weight)
    meta_yolu  = model_klasor / "egitim_meta.json"
    pos_weight = 8.0
    model_adi  = model_klasor.name
    if meta_yolu.exists():
        with open(str(meta_yolu), encoding="utf-8") as f:
            meta_egitim = json.load(f)
        pos_weight = meta_egitim.get("pos_weight", 8.0)
        model_adi  = meta_egitim.get("model_adi", model_adi)
        print("Egitim meta: pos_weight={}  Val Dice={}".format(
            pos_weight, meta_egitim.get("val_dice", "?")))

    # Model yukle
    print("Model yukleniyor...")
    model = keras.models.load_model(
        str(model_yolu),
        custom_objects={
            "birlesik_kayip": birlesik_kayip,
            "dice_skoru":     dice_skoru,
            "iou_skoru":      iou_skoru,
        }
    )
    print("Model yuklendi. Giris: {}  Parametre: {:,}".format(
        model.input_shape, model.count_params()))

    # Veri dosyalari
    x_dosyalar = sorted(
        (EGITIM_DIR / "test" / "X").glob("*_X.npy"))
    y_dosyalar = sorted(
        (EGITIM_DIR / "test" / "Y").glob("*_Y.npy"))

    if not x_dosyalar:
        raise FileNotFoundError(
            "{} icinde *_X.npy bulunamadi!".format(
                EGITIM_DIR / "test" / "X"))

    print("Test ornegi: {}  |  Tahmini hasta: ~{}".format(
        len(x_dosyalar), len(x_dosyalar) // 20))

    # Hasta bazli grupla
    hasta_gruplari = {}
    for xf, yf in zip(x_dosyalar, y_dosyalar):
        parcalar = xf.stem.replace("_X", "").rsplit("_z", 1)
        hasta    = parcalar[0] if len(parcalar) == 2 else xf.stem
        hasta_gruplari.setdefault(hasta, []).append((xf, yf))

    print("Test hastasi: {}".format(len(hasta_gruplari)))

    # Threshold analizi icin ilk 200 ornek
    print("\nThreshold analizi icin olasılıklar toplanıyor...")
    th_gercek, th_tahmin = [], []
    for xf, yf in list(zip(x_dosyalar, y_dosyalar))[:200]:
        x = np.load(str(xf)).astype(np.float32)
        y = np.load(str(yf)).astype(np.float32)
        p = model.predict(x[np.newaxis, ...], verbose=0)[0]
        th_gercek.append(y)
        th_tahmin.append(p)

    en_iyi_esik, en_iyi_dice = threshold_analizi(
        th_gercek, th_tahmin, sonuc_dir)
    print("  -> Kullanilacak esik: {}  "
          "(En iyi: {:.2f}; ESIK degiskenini guncelleyebilirsiniz)".format(
              ESIK, en_iyi_esik))

    # Ana test dongusu
    sonuclar = []

    for hasta_key, dosyalar in tqdm(hasta_gruplari.items(),
                                     desc="Test ediliyor"):
        x_ornekler, y_gercek_list = [], []
        y_tahmin_list, metrik_list = [], []
        tahmin_maskeleri = {}

        spacing_mm = [1.3672, 1.3672, 3.27]
        boy_cm     = BOY_MANUEL.get(hasta_key)
        meta_h     = NPY_DIR / hasta_key / "meta.json"

        if meta_h.exists():
            with open(str(meta_h), encoding="utf-8") as f:
                mh = json.load(f)
            spacing_mm = mh.get("spacing_mm", spacing_mm)
            if not boy_cm:
                boy_cm = mh.get("boy_cm")

        for xf, yf in dosyalar:
            x        = np.load(str(xf)).astype(np.float32)
            y        = np.load(str(yf)).astype(np.float32)
            y_tahmin = model.predict(x[np.newaxis, ...], verbose=0)[0]
            metr     = metrikleri_hesapla(y, y_tahmin, ESIK)

            parcalar = xf.stem.replace("_X", "").rsplit("_z", 1)
            z_idx    = int(parcalar[1]) if len(parcalar) == 2 else 0
            tahmin_maskeleri[z_idx] = (
                y_tahmin[:, :, 0] > ESIK).astype(np.float32)

            x_ornekler.append(x)
            y_gercek_list.append(y)
            y_tahmin_list.append(y_tahmin)
            metrik_list.append(metr)

        def ort(k):
            return round(float(np.mean([m[k] for m in metrik_list])), 4)

        smi = smi_karsilastir(hasta_key, tahmin_maskeleri,
                               spacing_mm, boy_cm)

        eslesti = None
        if smi.get("gercek_sark") and smi.get("tahmin_sark"):
            eslesti = ("EVET"
                       if smi["gercek_sark"] == smi["tahmin_sark"]
                       else "HAYIR")

        sonuclar.append({
            "hasta":       hasta_key,
            "mimari":      "DeepLab-UNeTR",
            "dilim_say":   len(dosyalar),
            "dice":        ort("dice"),
            "iou":         ort("iou"),
            "hassasiyet":  ort("hassasiyet"),
            "ozgulluk":    ort("ozgulluk"),
            "kesinlik":    ort("kesinlik"),
            "gercek_smi":  smi.get("gercek_smi"),
            "tahmin_smi":  smi.get("tahmin_smi"),
            "fark":        smi.get("fark"),
            "fark_pct":    smi.get("fark_pct"),
            "gercek_sark": smi.get("gercek_sark"),
            "tahmin_sark": smi.get("tahmin_sark"),
            "eslesti":     eslesti,
        })

        print("  {:25s}  Dice:{:.3f}  IoU:{:.3f}  "
              "SMI gercek:{:6}  tahmin:{:6}  {}".format(
            hasta_key[:25], ort("dice"), ort("iou"),
            str(smi.get("gercek_smi") or "?"),
            str(smi.get("tahmin_smi") or "?"),
            eslesti or ""))

        # L3-1, L3 merkez, L3+1
        n_dilim     = len(x_ornekler)
        dice_list_h = [m["dice"] for m in metrik_list]
        l3_pos      = int(np.argmax(dice_list_h))
        bas_idx     = max(0, l3_pos - 1)
        bit_idx     = min(n_dilim, l3_pos + 2)

        devam = gorsel_goster_opencv(
            hasta_key, gorsel_dir,
            x_ornekler[bas_idx:bit_idx],
            y_gercek_list[bas_idx:bit_idx],
            y_tahmin_list[bas_idx:bit_idx],
            metrik_list[bas_idx:bit_idx],
            smi, max_gorsel=3)

        if not devam:
            print("\nKullanici ESC ile cikti.")
            break

    # DataFrame & Excel
    df = pd.DataFrame(sonuclar)
    excel_yolu = sonuc_dir / "test_sonuclari_dlunetr.xlsx"
    excel_rapor_kaydet(df, excel_yolu)

    # Ozet
    print("\n" + "=" * 65)
    print("TEST SONUCLARI -- {} hasta  [DeepLab-UNeTR Hibrit]".format(
        len(df)))
    print("-" * 65)
    print("  Dice       : {:.4f} +/- {:.4f}  "
          "(min:{:.4f}  max:{:.4f})".format(
        df["dice"].mean(), df["dice"].std(),
        df["dice"].min(),  df["dice"].max()))
    print("  IoU        : {:.4f} +/- {:.4f}".format(
        df["iou"].mean(), df["iou"].std()))
    print("  Hassasiyet : {:.4f}".format(df["hassasiyet"].mean()))
    print("  Ozgulluk   : {:.4f}".format(df["ozgulluk"].mean()))
    print("  Kesinlik   : {:.4f}".format(df["kesinlik"].mean()))

    # Metrik ozet grafik
    fig, ax = plt.subplots(figsize=(8, 5))
    metrik_adlari = ["Dice", "IoU", "Hassasiyet", "Ozgulluk", "Kesinlik"]
    metrik_ort = [df["dice"].mean(), df["iou"].mean(),
                  df["hassasiyet"].mean(), df["ozgulluk"].mean(),
                  df["kesinlik"].mean()]
    metrik_std = [df["dice"].std(), df["iou"].std(),
                  df["hassasiyet"].std(), df["ozgulluk"].std(),
                  df["kesinlik"].std()]
    renkler = ["steelblue", "seagreen", "darkorange",
               "mediumpurple", "crimson"]
    bars = ax.bar(metrik_adlari, metrik_ort,
                  yerr=metrik_std, capsize=5,
                  color=renkler, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, metrik_ort):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                "{:.3f}".format(val),
                ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("DeepLab-UNeTR Hibrit -- Test Seti Metrikleri\n"
                 "(n={} hasta, esik={})".format(len(df), ESIK))
    ax.set_ylabel("Skor (ortalama +/- std)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(sonuc_dir / "metrik_ozet.png"), dpi=130)
    plt.close()

    # SMI karsilastirma
    df_smi = df.dropna(subset=["gercek_smi", "tahmin_smi"])
    if len(df_smi) >= 3:
        r, p = stats.pearsonr(df_smi["gercek_smi"],
                               df_smi["tahmin_smi"])
        eslesti_n = (df_smi["eslesti"] == "EVET").sum()

        print("\n  SMI Karsilastirma ({} hasta):".format(len(df_smi)))
        print("  Korelasyon (r)   : {:.4f}  p={:.4f}".format(r, p))
        print("  Ort. SMI farki   : {:.2f} cm2/m2".format(
            df_smi["fark"].abs().mean()))
        print("  Sinif eslesmesi  : {}/{} ({:.1f}%)".format(
            eslesti_n, len(df_smi),
            100 * eslesti_n / len(df_smi)))

        ort_smi  = (df_smi["gercek_smi"] + df_smi["tahmin_smi"]) / 2
        fark_smi = df_smi["tahmin_smi"] - df_smi["gercek_smi"]
        of = fark_smi.mean(); sd = fark_smi.std()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "SMI Karsilastirma -- DeepLab-UNeTR Hibrit", fontsize=13)

        # Scatter
        axes[0].scatter(df_smi["gercek_smi"], df_smi["tahmin_smi"],
                        s=80, alpha=0.85, color="#378ADD",
                        edgecolors="none", zorder=3)
        for _, row in df_smi.iterrows():
            axes[0].annotate(
                row["hasta"].replace("_", " ")[:15],
                (row["gercek_smi"], row["tahmin_smi"]),
                fontsize=6, alpha=0.6,
                xytext=(4, 2), textcoords="offset points")
        mn = df_smi["gercek_smi"].min() - 3
        mx = df_smi["gercek_smi"].max() + 3
        axes[0].plot([mn, mx], [mn, mx], "k--",
                     lw=1.2, alpha=0.5, label="y=x")
        axes[0].set_xlabel("Gercek SMI (cm2/m2)", fontsize=11)
        axes[0].set_ylabel("Tahmin SMI (cm2/m2)", fontsize=11)
        axes[0].set_title(
            "Scatter -- r={:.4f}  p={:.4f}".format(r, p))
        axes[0].legend()

        # Bland-Altman
        axes[1].scatter(ort_smi, fark_smi,
                        s=80, alpha=0.85, color="#1D9E75",
                        edgecolors="none", zorder=3)
        for _, row in df_smi.iterrows():
            o = (row["gercek_smi"] + row["tahmin_smi"]) / 2
            f = row["tahmin_smi"] - row["gercek_smi"]
            axes[1].annotate(
                row["hasta"].replace("_", " ")[:15],
                (o, f), fontsize=6, alpha=0.6,
                xytext=(4, 2), textcoords="offset points")
        axes[1].axhline(of, color="black", lw=1.5,
                        label="Ort: {:.2f}".format(of))
        axes[1].axhline(of + 1.96 * sd, color="#E24B4A",
                        lw=1.2, ls="--",
                        label="+1.96SD: {:.2f}".format(of + 1.96 * sd))
        axes[1].axhline(of - 1.96 * sd, color="#E24B4A",
                        lw=1.2, ls="--",
                        label="-1.96SD: {:.2f}".format(of - 1.96 * sd))
        axes[1].axhline(0, color="gray", lw=0.8, ls=":")
        axes[1].set_xlabel("Ortalama SMI (cm2/m2)", fontsize=11)
        axes[1].set_ylabel(
            "Fark: Tahmin - Gercek (cm2/m2)", fontsize=11)
        axes[1].set_title("Bland-Altman Analizi")
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(str(sonuc_dir / "smi_karsilastirma.png"), dpi=140)
        plt.close()
        print("  Bland-Altman grafigi kaydedildi.")

    # JSON ozet
    ozet = {
        "tarih":            zaman,
        "model":            str(model_yolu),
        "mimari":           "DeepLab-UNeTR Hibrit",
        "hasta_sayisi":     len(df),
        "esik":             ESIK,
        "en_iyi_esik":      round(en_iyi_esik, 2),
        "ortalama_dice":    round(float(df["dice"].mean()), 4),
        "std_dice":         round(float(df["dice"].std()),  4),
        "ortalama_iou":     round(float(df["iou"].mean()),  4),
        "ortalama_hass":    round(float(df["hassasiyet"].mean()), 4),
        "ortalama_ozg":     round(float(df["ozgulluk"].mean()), 4),
    }
    with open(str(sonuc_dir / "test_ozet.json"), "w",
              encoding="utf-8") as f:
        json.dump(ozet, f, indent=2, ensure_ascii=False)

    print("\n  Ciktilar:")
    print("  Excel          :", excel_yolu)
    print("  Metrik grafik  :", sonuc_dir / "metrik_ozet.png")
    print("  Threshold grfk :", sonuc_dir / "threshold_analizi.png")
    print("  Gorseller      :", gorsel_dir)
    print("=" * 65)
    return df


# ===========================================
#  KOMUT SATIRI
# ===========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepLab-UNeTR model testi")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Test edilecek .keras dosya yolu. "
             "Verilmezse DLUnetR klasoeruendeki son model kullanilir.")
    args = parser.parse_args()

    if args.model:
        model_yolu = Path(args.model)
        if not model_yolu.exists():
            raise FileNotFoundError(
                "Model bulunamadi: {}".format(model_yolu))
    else:
        model_yolu = son_modeli_bul(DLUNE_DIR)

    test_et(model_yolu)