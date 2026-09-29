"""
ADIM 3 — Model Testi ve Degerlendirme (v6 - merkez dilim, panel panel secimli kayit)
============================================================
Pencere 1: 4 panelli ana goruntu (CT, Gercek, Tahmin, Ortusme)
Pencere 2: Ortusme yakinlastirma penceresi
  - Mouse scroll yukari     : zoom in
  - Mouse scroll asagi      : zoom out
  - Mouse sol tik + surukle : pan (kaydirma)
  - 1 / 2 / 3               : zoom penceresinde dilim sec
  - S                       : goruntuyu SONUC klasorune kaydet, sonraki hasta
  - N veya BOSLUK           : kaydetmeden gec, sonraki hasta
  - ESC                     : goruntulemeyi bitir (kalan hastalar
                              ekransiz hesaplanir, Excel yine yazilir)
Tum hastalarin goruntuleri ayrica TEST_DIR/gorseller'e arsivlenir.

Calistirma:
  python src/test_unet25d_interactive.py
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ─────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────
# Proje kok klasoru: SARKOPENI_DIR ortam degiskeniyle degistirilebilir
BASE_DIR = Path(os.environ.get("SARKOPENI_DIR", r"E:\sarkopeni"))

MODEL_YOLU = str(BASE_DIR / "modeller" / "unet25d_512_20260928_1700" / "en_iyi.keras")

EGITIM_DIR = BASE_DIR / "output" / "egitim_25d"
NPY_DIR    = BASE_DIR / "output" / "npy"
TEST_DIR   = BASE_DIR / "output" / "test_sonuclari_20260928"
# Secilen (S tusu) goruntuler buraya kaydedilir
SONUC_DIR  = BASE_DIR / "sonuc"
# True: secilen panellerin Ingilizce etiketli kopyasi da SONUC_DIR/etiketli'ye yazilir
ETIKETLI_KAYIT = True
ESIK       = 0.5

BOY_MANUEL = {
    # "hasta_klasor_adi": 172,   # DICOM'da boyu olmayan hastalar icin (cm)
}


# ═══════════════════════════════════════════
# OZEL NESNELER
# ═══════════════════════════════════════════
@tf.autograph.experimental.do_not_convert
def dice_skoru(y_gercek, y_tahmin, smooth=1e-6):
    yg = tf.reshape(tf.cast(y_gercek, tf.float32), [-1])
    yt = tf.reshape(tf.cast(y_tahmin > 0.5, tf.float32), [-1])
    k  = tf.reduce_sum(yg * yt)
    return (2.0*k + smooth) / (tf.reduce_sum(yg) + tf.reduce_sum(yt) + smooth)

@tf.autograph.experimental.do_not_convert
def iou_skoru(y_gercek, y_tahmin, smooth=1e-6):
    yg = tf.reshape(tf.cast(y_gercek, tf.float32), [-1])
    yt = tf.reshape(tf.cast(y_tahmin > 0.5, tf.float32), [-1])
    k  = tf.reduce_sum(yg * yt)
    b  = tf.reduce_sum(yg) + tf.reduce_sum(yt) - k
    return (k + smooth) / (b + smooth)

@tf.autograph.experimental.do_not_convert
def birlesik_kayip(y_gercek, y_tahmin, smooth=1e-6):
    yg = tf.cast(y_gercek, tf.float32)
    yt = tf.cast(y_tahmin, tf.float32)
    k  = tf.reduce_sum(yg * yt)
    return 1.0 - (2.0*k + smooth) / (tf.reduce_sum(yg) + tf.reduce_sum(yt) + smooth)


# ═══════════════════════════════════════════
# METRIKLER
# ═══════════════════════════════════════════
def metrikleri_hesapla(y_gercek, y_tahmin, esik=0.5):
    yg = (y_gercek.flatten() > 0.5).astype(np.float32)
    yt = (y_tahmin.flatten() > esik).astype(np.float32)
    tp = float(np.sum(yg * yt))
    fp = float(np.sum((1-yg) * yt))
    fn = float(np.sum(yg * (1-yt)))
    tn = float(np.sum((1-yg) * (1-yt)))
    s  = 1e-6
    return {
        "dice":       round((2*tp+s) / (2*tp+fp+fn+s), 4),
        "iou":        round((tp+s)   / (tp+fp+fn+s),   4),
        "hassasiyet": round((tp+s)   / (tp+fn+s),      4),
        "ozgulluk":   round((tn+s)   / (tn+fp+s),      4),
        "kesinlik":   round((tp+s)   / (tp+fp+s),      4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ═══════════════════════════════════════════
# SMI KARSILASTIRMA
# ═══════════════════════════════════════════
def smi_karsilastir(hasta_key, tahmin_maskeleri, spacing_mm, boy_cm):
    SMI_ESIK = {"e": 49.0, "k": 31.0}
    meta_yolu  = NPY_DIR / hasta_key / "meta.json"
    gercek_smi = None
    cinsiyet   = "e"

    if meta_yolu.exists():
        with open(meta_yolu, encoding="utf-8") as f:
            meta = json.load(f)
        gercek_smi = meta.get("smi")
        cinsiyet   = (meta.get("cinsiyet") or "E").lower()
        if not boy_cm:
            boy_cm = meta.get("boy_cm")

    if not boy_cm:
        boy_cm = BOY_MANUEL.get(hasta_key)

    if not tahmin_maskeleri or not boy_cm:
        return {"gercek_smi": gercek_smi, "tahmin_smi": None,
                "fark": None, "fark_pct": None,
                "gercek_sark": None, "tahmin_sark": None,
                "smi_esik": None}

    px_cm2     = (spacing_mm[0] / 10.0) * (spacing_mm[1] / 10.0)
    alanlar    = [float(m.sum()) * px_cm2 for m in tahmin_maskeleri.values()]
    ort_alan   = float(np.mean(alanlar))
    tahmin_smi = round(ort_alan / (boy_cm / 100.0) ** 2, 2)
    esik_val   = SMI_ESIK.get(cinsiyet, 49.0)
    fark       = round(tahmin_smi - gercek_smi, 2) if gercek_smi else None
    fark_pct   = round(abs(fark) / gercek_smi * 100, 1) if (fark and gercek_smi) else None

    return {
        "gercek_smi":  gercek_smi,
        "tahmin_smi":  tahmin_smi,
        "fark":        fark,
        "fark_pct":    fark_pct,
        "smi_esik":    esik_val,
        "gercek_sark": "EVET" if (gercek_smi and gercek_smi < esik_val) else "HAYIR",
        "tahmin_sark": "EVET" if tahmin_smi < esik_val else "HAYIR",
    }


# ═══════════════════════════════════════════
# ZOOM PENCERESI — mouse scroll + pan
# ═══════════════════════════════════════════
class ZoomPenceresi:
    """
    Tam cozunurluklu ortusme goruntusunu gosterir.
    Mouse scroll: zoom in/out
    Sol tik + surukle: pan (kaydirma)
    """
    def __init__(self, pencere_adi, goruntu):
        self.pencere_adi = pencere_adi
        self.goruntu     = goruntu          # tam boyutlu BGR goruntu
        self.H, self.W   = goruntu.shape[:2]

        # Goruntuleme durumu
        self.zoom    = 1.0                  # mevcut zoom carpani
        self.zoom_min = 0.2
        self.zoom_max = 10.0
        self.zoom_adim = 0.15

        # Sol ust kose — goruntulenecek bolgenin baslangici (piksel)
        self.ox = 0
        self.oy = 0

        # Pan durumu
        self.surukleme = False
        self.son_x = 0
        self.son_y = 0

        # Pencere boyutu
        self.pen_w = min(900, self.W)
        self.pen_h = min(900, self.H)

        cv2.namedWindow(self.pencere_adi, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.pencere_adi, self.pen_w, self.pen_h)
        cv2.setMouseCallback(self.pencere_adi, self._mouse_callback)
        self._goster()

    def _sinirla(self):
        """Offset sınırlarını kontrol et."""
        zoomed_w = int(self.W * self.zoom)
        zoomed_h = int(self.H * self.zoom)
        gorunen_w = int(self.pen_w / self.zoom)
        gorunen_h = int(self.pen_h / self.zoom)
        self.ox = max(0, min(self.ox, self.W - gorunen_w))
        self.oy = max(0, min(self.oy, self.H - gorunen_h))

    def _goster(self):
        """Mevcut zoom ve offset ile goruntu goster."""
        gorunen_w = int(self.pen_w / self.zoom)
        gorunen_h = int(self.pen_h / self.zoom)

        # Sinir kontrolu
        gorunen_w = min(gorunen_w, self.W)
        gorunen_h = min(gorunen_h, self.H)
        self.ox   = max(0, min(self.ox, self.W - gorunen_w))
        self.oy   = max(0, min(self.oy, self.H - gorunen_h))

        # Goruntu parcasini crop et
        bolge = self.goruntu[
            self.oy : self.oy + gorunen_h,
            self.ox : self.ox + gorunen_w
        ]

        # Pencere boyutuna buyut
        gosterim = cv2.resize(bolge, (self.pen_w, self.pen_h),
                              interpolation=cv2.INTER_LINEAR)

        # Zoom oranini sol ust koseye yaz
        cv2.putText(gosterim,
                    "Zoom: {:.1f}x  |  Scroll: zoom  |  Sol tik+surukle: pan".format(
                        self.zoom),
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (220, 220, 100), 2, cv2.LINE_AA)

        if getattr(self, "mesaj", ""):
            cv2.putText(gosterim, self.mesaj, (10, self.pen_h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)

        cv2.imshow(self.pencere_adi, gosterim)

    def gorunen_kesit(self, uzun_kenar=1024):
        """Zoom penceresinde o an gorunen bolgeyi (yazisiz) dondurur."""
        gorunen_w = min(int(self.pen_w / self.zoom), self.W)
        gorunen_h = min(int(self.pen_h / self.zoom), self.H)
        ox = max(0, min(self.ox, self.W - gorunen_w))
        oy = max(0, min(self.oy, self.H - gorunen_h))
        bolge = self.goruntu[oy:oy + gorunen_h, ox:ox + gorunen_w]
        olcek = uzun_kenar / max(bolge.shape[:2])
        return cv2.resize(bolge, (int(bolge.shape[1] * olcek), int(bolge.shape[0] * olcek)),
                          interpolation=cv2.INTER_LINEAR)

    def _mouse_callback(self, olay, x, y, flags, param):
        # Scroll — zoom
        if olay == cv2.EVENT_MOUSEWHEEL:
            # flags > 0 yukari (zoom in), < 0 asagi (zoom out)
            eski_zoom = self.zoom
            if flags > 0:
                self.zoom = min(self.zoom_max, self.zoom * (1 + self.zoom_adim))
            else:
                self.zoom = max(self.zoom_min, self.zoom / (1 + self.zoom_adim))

            # Zoom merkezini mouse pozisyonuna sabitle
            gercek_x = self.ox + x / eski_zoom
            gercek_y = self.oy + y / eski_zoom
            self.ox = int(gercek_x - x / self.zoom)
            self.oy = int(gercek_y - y / self.zoom)
            self._sinirla()
            self._goster()

        # Sol tik basi — pan baslat
        elif olay == cv2.EVENT_LBUTTONDOWN:
            self.surukleme = True
            self.son_x = x
            self.son_y = y

        # Sol tik birak — pan durdur
        elif olay == cv2.EVENT_LBUTTONUP:
            self.surukleme = False

        # Surukle
        elif olay == cv2.EVENT_MOUSEMOVE and self.surukleme:
            dx = int((self.son_x - x) / self.zoom)
            dy = int((self.son_y - y) / self.zoom)
            self.ox += dx
            self.oy += dy
            self.son_x = x
            self.son_y = y
            self._sinirla()
            self._goster()


# ═══════════════════════════════════════════
# ANA GORSEL — 4 panel
# ═══════════════════════════════════════════
def ortusme_goruntu_olustur(ct_bgr, yg, yt, boyut):
    """
    Tam cozunurluklu ortusme goruntusunu olusturur.
    yesil=TP, kirmizi(BGR)=FP, mavi=FN
    """
    H, W  = yg.shape
    ct_r  = cv2.resize(ct_bgr, (boyut, boyut))
    sonuc = ct_r.copy()

    tp = ((yg > 0.5) & (yt > 0.5)).astype(np.uint8)
    fp = ((yg < 0.5) & (yt > 0.5)).astype(np.uint8)
    fn = ((yg > 0.5) & (yt < 0.5)).astype(np.uint8)

    for msk, renk in [(tp, [0, 220, 50]),
                       (fp, [0, 50, 220]),
                       (fn, [200, 50, 50])]:
        msk_r  = cv2.resize((msk * 255).astype(np.uint8), (boyut, boyut))
        renkli = np.zeros_like(sonuc)
        renkli[msk_r > 127] = renk
        sonuc  = cv2.addWeighted(sonuc, 1.0, renkli, 0.6, 0)

    return sonuc


def gorsel_goster(hasta_key, gorsel_dir,
                  x_ornekler, y_gercek_list,
                  y_tahmin_list, metrik_list,
                  smi_bilgi, max_gorsel=3, goster=True, merkez=1):
    """
    Yalnizca MERKEZ dilimin 4 goruntusunu ayri paneller olarak gosterir:
      [1] Orijinal CT  [2] Gercek maske  [3] Tahmin maskesi  [4] Ortusme
    Tuslar:
      1 / 2 / 3 / 4 : o paneli SONUC klasorune ayri dosya olarak kaydet
      A             : 4 panelin hepsini kaydet
      5 / Z         : zoom penceresinde o an gorunen kesiti kaydet
      N / Bosluk / Enter : sonraki hasta
      ESC           : goruntulemeyi bitir (kalan hastalar ekransiz hesaplanir)
    Kaydedilen dosyalar yazisiz, 1024x1024 temiz goruntulerdir.
    """
    if len(x_ornekler) == 0:
        return True
    merkez = max(0, min(merkez, len(x_ornekler) - 1))

    PANEL, KAYIT, BOSLUK = 600, 1024, 6
    ANA_PEN = "Merkez dilim — {}".format(hasta_key[:35])
    ZOM_PEN = "Ortusme ZOOM — {}".format(hasta_key[:35])

    x  = x_ornekler[merkez]
    yg = y_gercek_list[merkez][:, :, 0]
    yt = (y_tahmin_list[merkez][:, :, 0] > ESIK).astype(np.float32)
    m  = metrik_list[merkez]
    ct_bgr = cv2.cvtColor((x[:, :, 1] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def katman(maske, renk, boyut, alfa=0.5):
        base  = cv2.resize(ct_bgr, (boyut, boyut))
        msk_r = cv2.resize((maske * 255).astype(np.uint8), (boyut, boyut))
        ren   = np.zeros_like(base)
        ren[msk_r > 127] = renk
        return cv2.addWeighted(base, 1.0, ren, alfa, 0)

    def paneller(boyut):
        return [cv2.resize(ct_bgr, (boyut, boyut)),
                katman(yg, [0, 200, 50], boyut),
                katman(yt, [0, 140, 255], boyut),
                ortusme_goruntu_olustur(ct_bgr, yg, yt, boyut)]

    temiz   = paneller(KAYIT)          # kaydedilecek yazisiz goruntuler
    adlar   = ["1_orijinal_ct", "2_gercek_maske", "3_tahmin_maskesi", "4_ortusme"]
    # Ingilizce panel etiketleri (makale icin)
    ing_ad  = ["Original CT", "Ground Truth", "Prediction",
               "Overlap (Green: TP, Red: FP, Blue: FN)"]
    etiket  = ["[{}] {}".format(i + 1, a) for i, a in enumerate(ing_ad)]
    kaydedilen = set()

    smi_str = ""
    if smi_bilgi.get("gercek_smi") and smi_bilgi.get("tahmin_smi"):
        smi_str = "   SMI: GT={:.1f}  Pred={:.1f}  Diff={:+.1f}".format(
            smi_bilgi["gercek_smi"], smi_bilgi["tahmin_smi"],
            smi_bilgi.get("fark", 0) or 0)
    baslik = "{}   Dice: {:.3f}  IoU: {:.3f}{}".format(
        hasta_key.replace("_", " ").upper(), m["dice"], m["iou"], smi_str)

    def cerceve():
        gorunen = []
        for i, p in enumerate(paneller(PANEL)):
            p = p.copy()
            cv2.putText(p, etiket[i], (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (180, 220, 255), 2, cv2.LINE_AA)
            if i in kaydedilen:
                cv2.rectangle(p, (3, 3), (PANEL - 4, PANEL - 4), (0, 220, 255), 4)
                cv2.putText(p, "SAVED", (10, PANEL - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
            gorunen.append(p)
        bos = np.full((PANEL, BOSLUK, 3), 40, dtype=np.uint8)
        satir = np.hstack([gorunen[0], bos, gorunen[1], bos, gorunen[2], bos, gorunen[3]])
        ab = np.full((60, satir.shape[1], 3), 20, dtype=np.uint8)
        cv2.putText(ab, baslik, (12, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (220, 235, 255), 2, cv2.LINE_AA)
        alt = np.full((42, satir.shape[1], 3), 20, dtype=np.uint8)
        cv2.putText(alt, "1-4: save panel (4 also saves zoom)  |  A: save all + zoom  |  5 / Z: save zoom only  |  "
                    "N / Space / Enter: next patient  |  ESC: finish viewing",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (140, 140, 140), 1, cv2.LINE_AA)
        return np.vstack([ab, satir, alt])

    # Arsiv kaydi (her hasta, yazili 4'lu goruntu)
    cv2.imwrite(str(gorsel_dir / "{}_merkez.png".format(hasta_key[:35])), cerceve())
    if not goster:
        return True

    def kaydet(i):
        SONUC_DIR.mkdir(parents=True, exist_ok=True)
        yol = SONUC_DIR / "{}_{}.png".format(hasta_key[:35], adlar[i])
        cv2.imwrite(str(yol), temiz[i])
        # Gercek maske / tahmin secildiyse siyah-beyaz ikili halini de kaydet
        if i in (1, 2):
            (SONUC_DIR / "ikili_maske").mkdir(exist_ok=True)
            mk = yg if i == 1 else yt
            ikili = cv2.resize(((mk > 0.5).astype(np.uint8)) * 255, (KAYIT, KAYIT),
                               interpolation=cv2.INTER_NEAREST)
            ad2 = "gercek_maske_ikili" if i == 1 else "tahmin_maskesi_ikili"
            cv2.imwrite(str(SONUC_DIR / "ikili_maske" / "{}_{}.png".format(hasta_key[:35], ad2)), ikili)
        if ETIKETLI_KAYIT:
            (SONUC_DIR / "etiketli").mkdir(exist_ok=True)
            e = temiz[i].copy()
            cv2.rectangle(e, (0, 0), (KAYIT, 64), (0, 0, 0), -1)
            cv2.putText(e, ing_ad[i], (20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(SONUC_DIR / "etiketli" / yol.name), e)
        kaydedilen.add(i)
        print("  KAYDEDILDI ->", yol.name)

    cv2.namedWindow(ANA_PEN, cv2.WINDOW_NORMAL)
    final = cerceve()
    cv2.resizeWindow(ANA_PEN, min(final.shape[1], 1920), min(final.shape[0], 700))
    cv2.imshow(ANA_PEN, final)
    zp = ZoomPenceresi(ZOM_PEN, temiz[3])
    zoom_sayac = [0]

    def zoom_kaydet():
        SONUC_DIR.mkdir(parents=True, exist_ok=True)
        zoom_sayac[0] += 1
        yol = SONUC_DIR / "{}_5_ortusme_zoom{}.png".format(hasta_key[:35], zoom_sayac[0])
        kesit = zp.gorunen_kesit()
        cv2.imwrite(str(yol), kesit)
        if ETIKETLI_KAYIT:
            (SONUC_DIR / "etiketli").mkdir(exist_ok=True)
            e = kesit.copy()
            cv2.rectangle(e, (0, 0), (e.shape[1], 64), (0, 0, 0), -1)
            cv2.putText(e, "Overlap, magnified (Green: TP, Red: FP, Blue: FN)", (20, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(SONUC_DIR / "etiketli" / yol.name), e)
        print("  KAYDEDILDI (zoom) ->", yol.name)
        zp.mesaj = "SAVED: zoom view {}".format(zoom_sayac[0])
        zp._goster()
    print("  [1-4] paneli kaydet  [A] hepsi  [5/Z] zoom kesitini kaydet  [N/Bosluk/Enter] sonraki  [ESC] bitir")

    while True:
        tus = cv2.waitKey(50) & 0xFF
        if tus in (ord('1'), ord('2'), ord('3'), ord('4')):
            kaydet(tus - ord('1'))
            if tus == ord('4'):
                zoom_kaydet()       # ortusme secilince zoom gorunumu de kaydedilir
            cv2.imshow(ANA_PEN, cerceve())
        elif tus in (ord('5'), ord('z'), ord('Z')):
            zoom_kaydet()
        elif tus in (ord('a'), ord('A')):
            for i in range(4):
                kaydet(i)
            zoom_kaydet()
            cv2.imshow(ANA_PEN, cerceve())
        elif tus in (ord('n'), ord('N'), 32, 13):
            if not kaydedilen:
                print("  gecildi (kaydedilmedi)")
            break
        elif tus == 27:
            cv2.destroyAllWindows()
            return False
        if cv2.getWindowProperty(ANA_PEN, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return False

    cv2.destroyAllWindows()
    return True



# ═══════════════════════════════════════════
# EXCEL RAPORU
# ═══════════════════════════════════════════
def excel_rapor_kaydet(df, dosya_yolu):
    wb = Workbook(); ws = wb.active; ws.title = "Test Sonuclari"
    BASLIK = PatternFill("solid", fgColor="1F4E79")
    IYI    = PatternFill("solid", fgColor="E2EFDA")
    ORTA   = PatternFill("solid", fgColor="FFF2CC")
    KOTU   = PatternFill("solid", fgColor="FCE4D6")
    BF = Font(bold=True, color="FFFFFF", size=10)
    NF = Font(size=10)
    C  = Alignment(horizontal="center", vertical="center")
    L  = Alignment(horizontal="left",   vertical="center")
    kn = Side(style="thin", color="BFBFBF")
    KN = Border(left=kn, right=kn, top=kn, bottom=kn)

    SUTUNLAR = [
        ("No",             "no",          5,  "c"),
        ("Hasta",          "hasta",       26, "l"),
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
        h.fill = BASLIK; h.font = BF; h.alignment = C; h.border = KN
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

    son = len(df) + 2
    ws.cell(row=son, column=1, value="ORTALAMA").font = Font(bold=True, size=10)
    for col_idx, (_, alan, _, _) in enumerate(SUTUNLAR[1:], start=2):
        if alan in ("dice","iou","hassasiyet","ozgulluk","kesinlik","fark","fark_pct"):
            cd = df[alan].dropna()
            if len(cd) > 0:
                c = ws.cell(row=son, column=col_idx,
                            value=round(float(cd.mean()), 4))
                c.font = Font(bold=True, size=10); c.alignment = C

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = "A1:{}{}".format(
        get_column_letter(len(SUTUNLAR)), len(df)+1)
    wb.save(str(dosya_yolu))
    print("  Excel:", dosya_yolu)


# ═══════════════════════════════════════════
# ANA TEST DONGUSU
# ═══════════════════════════════════════════
def test_et(model_yolu):
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    gorsel_dir = TEST_DIR / "gorseller"
    gorsel_dir.mkdir(exist_ok=True)

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    print("Model yukleniyor:", model_yolu)
    model = tf.keras.models.load_model(
        model_yolu,
        custom_objects={
            "dice_skoru":     dice_skoru,
            "iou_skoru":      iou_skoru,
            "birlesik_kayip": birlesik_kayip,
            "kayip":          birlesik_kayip,
        }
    )
    print("Model yuklendi. Giris:", model.input_shape)

    x_dosyalar = sorted((EGITIM_DIR / "test" / "X").glob("*_X.npy"))
    y_dosyalar = sorted((EGITIM_DIR / "test" / "Y").glob("*_Y.npy"))

    hasta_gruplari = {}
    for xf, yf in zip(x_dosyalar, y_dosyalar):
        parcalar = xf.stem.replace("_X", "").rsplit("_z", 1)
        hasta    = parcalar[0] if len(parcalar) == 2 else xf.stem
        if hasta not in hasta_gruplari:
            hasta_gruplari[hasta] = []
        hasta_gruplari[hasta].append((xf, yf))

    print("Test ornegi: {}  |  Test hastasi: {}".format(
        len(x_dosyalar), len(hasta_gruplari)))

    sonuclar = []
    goster   = True

    for hasta_key, dosyalar in tqdm(hasta_gruplari.items(), desc="Test"):
        x_ornekler, y_gercek_list = [], []
        y_tahmin_list, metrik_list = [], []
        tahmin_maskeleri = {}

        spacing_mm = [1.3672, 1.3672, 3.27]
        boy_cm     = BOY_MANUEL.get(hasta_key)
        meta_yolu  = NPY_DIR / hasta_key / "meta.json"
        if meta_yolu.exists():
            with open(meta_yolu, encoding="utf-8") as f:
                meta = json.load(f)
            spacing_mm = meta.get("spacing_mm", spacing_mm)
            if not boy_cm:
                boy_cm = meta.get("boy_cm")

        for xf, yf in dosyalar:
            x = np.load(str(xf)).astype(np.float32)
            y = np.load(str(yf)).astype(np.float32)
            y_tahmin = model.predict(x[np.newaxis, ...], verbose=0)[0]
            metr     = metrikleri_hesapla(y, y_tahmin, ESIK)
            parcalar = xf.stem.replace("_X", "").rsplit("_z", 1)
            z_idx    = int(parcalar[1]) if len(parcalar) == 2 else 0
            tahmin_maskeleri[z_idx] = (y_tahmin[:,:,0] > ESIK).astype(np.float32)
            x_ornekler.append(x); y_gercek_list.append(y)
            y_tahmin_list.append(y_tahmin); metrik_list.append(metr)

        ort = lambda k: round(float(np.mean([m[k] for m in metrik_list])), 4)
        smi = smi_karsilastir(hasta_key, tahmin_maskeleri, spacing_mm, boy_cm)
        eslesti = None
        if smi["gercek_sark"] and smi["tahmin_sark"]:
            eslesti = "EVET" if smi["gercek_sark"] == smi["tahmin_sark"] else "HAYIR"

        sonuclar.append({
            "hasta": hasta_key, "dilim_say": len(dosyalar),
            "dice": ort("dice"), "iou": ort("iou"),
            "hassasiyet": ort("hassasiyet"), "ozgulluk": ort("ozgulluk"),
            "kesinlik": ort("kesinlik"),
            "gercek_smi": smi.get("gercek_smi"), "tahmin_smi": smi.get("tahmin_smi"),
            "fark": smi.get("fark"), "fark_pct": smi.get("fark_pct"),
            "gercek_sark": smi.get("gercek_sark"), "tahmin_sark": smi.get("tahmin_sark"),
            "eslesti": eslesti,
        })

        print("  {:25s}  Dice:{:.3f}  IoU:{:.3f}  SMI:{:6}->{:6}  {}".format(
            hasta_key[:25], ort("dice"), ort("iou"),
            str(smi.get("gercek_smi") or "?"),
            str(smi.get("tahmin_smi") or "?"),
            eslesti or ""))

        # L3-1, L3, L3+1 secimi
        n_dilim = len(x_ornekler)
        l3_pos  = int(np.argmax([m["dice"] for m in metrik_list]))
        bas     = max(0, l3_pos - 1)
        bit     = min(n_dilim, l3_pos + 2)

        devam = gorsel_goster(
            hasta_key, gorsel_dir,
            x_ornekler[bas:bit], y_gercek_list[bas:bit],
            y_tahmin_list[bas:bit], metrik_list[bas:bit],
            smi, max_gorsel=3, goster=goster, merkez=l3_pos - bas)

        if goster and not devam:
            print("\nESC: goruntuleme kapatildi, kalan hastalar ekransiz hesaplaniyor.")
            goster = False

    df = pd.DataFrame(sonuclar)
    excel_rapor_kaydet(df, TEST_DIR / "test_sonuclari.xlsx")

    print("\n" + "=" * 65)
    print("TEST SONUCLARI — {} hasta".format(len(df)))
    print("-" * 65)
    print("  Dice       : {:.4f} ± {:.4f}".format(df["dice"].mean(), df["dice"].std()))
    print("  IoU        : {:.4f} ± {:.4f}".format(df["iou"].mean(), df["iou"].std()))
    print("  Hassasiyet : {:.4f}".format(df["hassasiyet"].mean()))
    print("  Ozgulluk   : {:.4f}".format(df["ozgulluk"].mean()))
    print("  Kesinlik   : {:.4f}".format(df["kesinlik"].mean()))

    df_smi = df.dropna(subset=["gercek_smi", "tahmin_smi"])
    if len(df_smi) >= 3:
        r, p = stats.pearsonr(df_smi["gercek_smi"], df_smi["tahmin_smi"])
        eslesti_n = (df_smi["eslesti"] == "EVET").sum()
        print("\n  SMI ({} hasta): r={:.4f}  Ort.fark:{:.2f}  Sinif:{}/{} ({:.1f}%)".format(
            len(df_smi), r, df_smi["fark"].abs().mean(),
            eslesti_n, len(df_smi), 100*eslesti_n/len(df_smi)))

        ort_smi  = (df_smi["gercek_smi"] + df_smi["tahmin_smi"]) / 2
        fark_smi = df_smi["tahmin_smi"] - df_smi["gercek_smi"]
        of = fark_smi.mean(); sd = fark_smi.std()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Manual vs. Automated SMI", fontsize=13)
        axes[0].scatter(df_smi["gercek_smi"], df_smi["tahmin_smi"],
                        s=80, alpha=0.85, color="#378ADD", edgecolors="none")

        mn, mx = df_smi["gercek_smi"].min()-3, df_smi["gercek_smi"].max()+3
        axes[0].plot([mn,mx],[mn,mx],"k--",lw=1.2,alpha=0.5,label="y=x")
        axes[0].set_xlabel("Manual SMI (cm$^2$/m$^2$)"); axes[0].set_ylabel("Automated SMI (cm$^2$/m$^2$)")
        axes[0].set_title("Pearson r = {:.3f}, p {}".format(r, "< 0.001" if p < 0.001 else "= {:.3f}".format(p))); axes[0].legend()
        axes[1].scatter(ort_smi, fark_smi, s=80, alpha=0.85,
                        color="#1D9E75", edgecolors="none")
        axes[1].axhline(of, color="black", lw=1.5, label="Mean: {:.2f}".format(of))
        axes[1].axhline(of+1.96*sd, color="#E24B4A", lw=1.2, ls="--",
                        label="+1.96 SD: {:.2f}".format(of+1.96*sd))
        axes[1].axhline(of-1.96*sd, color="#E24B4A", lw=1.2, ls="--",
                        label="-1.96 SD: {:.2f}".format(of-1.96*sd))
        axes[1].set_xlabel("Mean of manual and automated SMI (cm$^2$/m$^2$)"); axes[1].set_ylabel("Automated - manual SMI (cm$^2$/m$^2$)")
        axes[1].set_title("Bland-Altman plot"); axes[1].legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(TEST_DIR / "smi_karsilastirma.png", dpi=140)
        plt.show(); plt.close()

    print("\n  Excel  :", TEST_DIR / "test_sonuclari.xlsx")
    print("  Gorsel :", TEST_DIR / "gorseller")
    print("=" * 65)
    return df


if __name__ == "__main__":
    df = test_et(MODEL_YOLU)