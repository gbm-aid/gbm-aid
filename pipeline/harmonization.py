"""
Harmonizasyon ve segmentasyon fonksiyonlari icin yer tutucu (stub) modul.
Mert (segmentasyon/N4/Z-score) ve Nisa (ComBat/neuroHarmonize) kendi
fonksiyonlarini yazdiginda, bu dosyadaki stub'lar gercek implementasyonla
degistirilecek. Fonksiyon imzalari (isim, parametre, donus tipi) ekip ici
sozlesme olarak burada sabitlenmistir.
"""


def apply_n4_bias_correction(image_path: str) -> str:
    """
    Mert tarafindan doldurulacak.
    N4BiasFieldCorrection uygular, islenmis dosyanin yolunu doner.
    """
    raise NotImplementedError("Mert tarafindan implemente edilecek")


def apply_zscore_normalization(image_path: str, source: str) -> str:
    """
    Mert tarafindan doldurulacak.
    Kaynak-bazli (source: 'TCGA'|'UPenn'|'LUMIERE') Z-score normalizasyonu uygular.
    """
    raise NotImplementedError("Mert tarafindan implemente edilecek")


def apply_combat_harmonization(features: dict, source: str) -> dict:
    """
    Nisa tarafindan doldurulacak.
    neuroHarmonize harmonizationApply() ile out-of-sample duzeltme uygular.
    Bilinmeyen kaynak icin UPenn-benzeri referans kategoriye atanir.
    """
    raise NotImplementedError("Nisa tarafindan implemente edilecek")


def get_segmentation_mask(patient_id: str, scan_id: str) -> dict:
    """
    Mert tarafindan doldurulacak.
    Hazir maske varsa onu, yoksa nnU-Net pretrained ile uretilen maskeyi doner.
    Dusuk guven skorunda 'low_confidence' uyari etiketi icerir.
    """
    raise NotImplementedError("Mert tarafindan implemente edilecek")
