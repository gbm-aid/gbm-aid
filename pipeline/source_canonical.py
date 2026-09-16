"""Kaynak adı kanonikleştirme -- GÖRÜNTÜ KÜTÜPHANESİNDEN BAĞIMSIZ.

NEDEN AYRI BİR MODÜL (2026-09-14, Barış onayı: "b1 i uygula, sac a dokunma")
---------------------------------------------------------------------------
Bu üç şey (`SOURCE_ALIASES`, `canonical_source()`, `_canonical_source_series()`)
2026-09-14'e kadar `pipeline/harmonization.py` içinde yaşıyordu. Üçü de SAF
string/pandas işi yapar, görüntüye HİÇ dokunmaz -- ama `harmonization.py`
modül başında `import SimpleITK as sitk` yaptığı için, bu üç yardımcıyı
almak isteyen HER modül 104 MB'lık bir görüntü kütüphanesini de yüklemek
zorunda kalıyordu.

SOMUT SONUÇ (2026-09-14'te ÖLÇÜLDÜ, varsayılmadı): Windows **Smart App
Control** o gün `_SimpleITK.cp310-win_amd64.pyd`'yi engellemeye başladı
(204 blok olayı, `Microsoft-Windows-CodeIntegrity/Operational`, Id
3033/3077/3118). `pipeline/cox_model.py`'nin TEK BİR import satırı
(`from pipeline.harmonization import _canonical_source_series,
canonical_source`) yüzünden şunların HEPSİ import edilemez hâle geldi:

    pipeline.cox_model · pipeline.xgboost_model
    api.predict · api.analyze_patient · api.similar · api.main

Yani **tahmin yolu, hiç ihtiyaç duymadığı hâlde görüntü işlemeye bağımlıydı.**
`/predict` radyomik değerleri DB'den okur; görüntü açmaz, yazmaz.
Smart App Control bu kusuru ORTAYA ÇIKARDI, yaratmadı.

KANIT -- ayrıştırmanın sayıları DEĞİŞTİRMEDİĞİ ölçüldü
------------------------------------------------------
Değişiklikten önce, `sys.modules`'a sahte (boş) bir SimpleITK enjekte
edilerek gerçek tahminler koşuldu. Beş kilitli regresyon pininin BEŞİ de
**bit-birebir** çıktı (fark 0.0e+00):

    UCSF-PDGM-167  risk -0.7096787591025546 · HR 0.49180215905468494
                   XGBoost P 0.7318906784057617
    TCGA-06-5412   risk  0.7551730099099314 · XGBoost P 0.2470972239971161
    model_arm = 'v3b_lowvar_v2amgmt'

EMSAL (aynı desen, aynı gerekçe)
--------------------------------
`api/similar.py:174` faiss'i modül seviyesinde DEĞİL fonksiyon içinde
import eder -- 2026-08-19'da faiss `.pyd`'si aynı şekilde Smart App
Control tarafından bloklandığı için. Bkz. `log/2026-08-19.md`.

⚠️ NE DEĞİŞMEDİ
---------------
* Kod **birebir** taşındı -- tek karakter değiştirilmedi.
* `pipeline.harmonization` bu üç adı **yeniden ihraç eder**; eski
  `from pipeline.harmonization import canonical_source` import'ları
  AYNEN çalışır (`pipeline/segmentation.py`, `tools/*.py`, testler).
* `SOURCE_ALIASES` **aynı sözlük NESNESİ** olarak paylaşılır --
  `tests/test_harmonization.py:1084` o sözlüğü yerinde mutasyona
  uğratıp geri alıyor (sentetik `"future-cohort"` enjeksiyonu); nesne
  kimliği korunmasaydı o test SESSİZCE anlamsızlaşırdı.
* N4 / Z-score / PyRadiomics yolu SimpleITK'ya GERÇEKTEN muhtaçtır ve
  blok sürdüğü sürece kapalı kalır -- bu ayrıştırma onu KURTARMAZ,
  yalnız tahmin/servis yolunu ondan bağımsızlaştırır.

⚠️ Review-gate YAPILMADI (E3) -- bu değişiklik ne ilgili ekip üyesinin
fiziksel incelemesinden, ne codex çapraz incelemesinden, ne de resmî
gözden geçirmeden geçti. "Production-ready" DEĞİLDİR.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

SOURCE_ALIASES = {
    "tcga": "TCGA",
    "tcga-gbm": "TCGA",
    "upenn": "UPenn",
    "upenn-gbm": "UPenn",
    "lumiere": "LUMIERE",
    # 2026-08-18 (Barış onayı) -- UCSF-PDGM tam kohort C32 çıkarımı.
    # UPenn'inkiyle ÖZDEŞ desen. DİKKAT: bu kanonik değer `COX_TRAINING_
    # ALLOWED_SOURCES = {"UPenn", "LUMIERE"}` (pipeline/cox_model.py)
    # tarafından BAĞIMSIZ, AYRI bir whitelist ile kontrol edilir --
    # SOURCE_ALIASES'e eklenmesi UCSF'i Cox eğitim havuzuna otomatik
    # SOKMAZ (2026-08-15 kararınca UCSF rolü: HARİCİ TEST 1, eğitime
    # GİRMEZ). Bu satır yalnız `canonical_source()`/N4/Z-score/ComBat
    # guard'larının UCSF'i TANIMASINI sağlar.
    "ucsf": "UCSF",
    "ucsf-pdgm": "UCSF",
}


def canonical_source(source: str) -> str:
    """Kaynak adını kanonik TCGA/UPenn/LUMIERE değerine dönüştür."""

    normalized = source.strip().lower()
    try:
        return SOURCE_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(SOURCE_ALIASES.values())))
        raise ValueError(
            f"Desteklenmeyen kaynak: {source!r}. Beklenen: {allowed}"
        ) from exc


def _canonical_source_series(source: pd.Series) -> pd.Series:
    """`canonical_source()`'ı bir `pd.Series`'in HER satırına uygula.

    Neden gerekli (2026-08-12 CRITICAL fix, bkz. decisions/2026-08-11-
    apply-combat-harmonization-true-source-guard.md "[2026-08-12]
    REVIEW-GATE" / "FIX-2"): `apply_combat_harmonization()`'ın TCGA
    guard'ı önceden kendi ad-hoc `.strip().str.upper() == "TCGA"`
    normalizasyonunu yapıyordu -- bu, `SOURCE_ALIASES`'te zaten tanımlı
    `"TCGA-GBM"` gibi alias'ları YAKALAMIYORDU (`"TCGA-GBM".strip()
    .upper()` = `"TCGA-GBM"` != `"TCGA"`). `dataset_sources.source_name`
    DB kolonunun GERÇEK değeri literal `"TCGA-GBM"` (bkz.
    `pipeline/cox_model.py` satır ~98 `WHERE ds.source_name IN
    ('UPenn-GBM', 'LUMIERE', 'TCGA-GBM')`), düz `"TCGA"` DEĞİL --
    normalize edilmemiş guard bu değeri sessizce geçirirdi.

    Bu fonksiyon tek kanonik kaynağı (`SOURCE_ALIASES` / `canonical_source()`)
    kullanarak her satırı normalize eder. Tanınmayan bir kaynak adı
    (yazım hatası, hiç bilinmeyen bir kaynak vb.) SESSİZCE geçilmez --
    proje kuralı "sessiz fallback yasak" gereği `ValueError` fırlatılır
    (hangi index'te olduğu hata mesajına eklenir).
    """

    def _safe_canonical(index_value: Any, raw_value: Any) -> str:
        try:
            return canonical_source(str(raw_value))
        except ValueError as exc:
            raise ValueError(
                f"true_dataset_source[{index_value!r}] = {raw_value!r} "
                "tanınmayan bir kaynak adı -- guard bunu sessizce "
                "geçemez (mimari kural: sessiz fallback/varsayım yasak). "
                "SOURCE_ALIASES içine eklenmeli veya veri kaynağındaki "
                f"yazım hatası düzeltilmeli. Orijinal hata: {exc}"
            ) from exc

    return pd.Series(
        [_safe_canonical(idx, val) for idx, val in source.items()],
        index=source.index,
        dtype=object,
    )
