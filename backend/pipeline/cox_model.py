"""GBM-AID Cox PHM + elastic-net + nested CV + SHAP entegrasyonu.

.. warning::

   **BU DOCSTRING'İN ALTINDAKİ "Kapsam" BÖLÜMÜ 2026-08-09 TARİHLİDİR ve
   ARTIK GEÇERSİZDİR.** Metin tarihsel kayıt olarak korunuyor (wiki hard
   rule #3: çelişki silinmez, işaretlenir), ama aşağıdaki sayı ve
   yöntemler CLAUDE.md'nin KİLİTLİ kurallarıyla çelişiyor. *Kodun
   davranışı DOĞRUDUR* -- sorun yalnız bu açıklama metnindedir
   (düzeltme 2026-08-28).

   ================= ==================================================
   Aşağıda yazan     GÜNCEL KİLİTLİ GERÇEK
   ================= ==================================================
   UPenn+LUMIERE     **YALNIZ UPenn**, fiili eğitim havuzu
   ``~721`` eğitim   **611 hasta / 585 ölüm olayı**. LUMIERE eğitime
                     GİRMEZ (``vital_status`` 0/91). "UPenn+LUMIERE
                     ~721 ile eğitildi" ifadesi raporda YASAKTIR.
   Harici test       Harici test **UCSF-PDGM 295 / 169** (2026-08-18).
   ``TCGA 50``       TCGA harici testten ÇIKARILDI (GDC'de
                     ``gtr_over90percent`` %0, ``idh1_status`` %0).
   ``LASSO (L1)``    CV-taranmış **elastic net + bootstrap stabilite
                     seçimi**; düz LASSO DEĞİL.
   ``107 özellik``   107 = bölge başına **çıkarım** sayısı. Birincil
                     **modelleme uzayı WT-only x 93** (ICC>=0,60
                     stabilite filtresi). Bu ikisi FARKLI şeydir.
   ``5-fold CV``     **nested-CV ZORUNLU** (seçim her fold İÇİNDE,
                     sızıntı önleme).
   ================= ==================================================

   Kaynak: CLAUDE.md "KRİTİK MİMARİ KURALLAR";
   ``decisions/2026-08-13-bolge-stratejisi-ve-modelleme-protokolu.md``;
   ``decisions/2026-08-18-tek-model-ucsf-harici-test-k15-kapanisi.md``.

Kapsam (raw/plan/plan.txt satır 426-438, Nisa/modeling-agent Hafta 3
görevleri) -- **2026-08-09 TARİHLİ, GEÇERSİZ, tarihsel kayıt**:

1. 107 radyomik özellik üzerinde LASSO (L1) ile klinik açıdan anlamlı
   alt küme seçimi (overfitting + çoklu doğrusal bağlantı riski).
2. lifelines ``CoxPHFitter`` ile UPenn+LUMIERE (~721) EĞİTİM setinin
   kurulması -- TCGA bu sete HİÇ dahil edilmez.
3. ``source`` kovaryatının (UPenn/LUMIERE) dummy değişken olarak
   tasarıma eklenmesi -- kurumsal (batch) etki kontrolü.
4. 5-fold stratified CV -- YALNIZ UPenn+LUMIERE eğitim havuzu içinde
   (hiperparametre/iç doğrulama), ardından TCGA'nın 50 hastasında
   (modelin hiç görmediği) harici test.
5. Event-sayısı (ölüm/nüks) doğrulama sorgusu.
6. SHAP entegrasyon noktası (iskelet -- bu dosyanın yazıldığı tarihte
   GERÇEK ÇALIŞTIRILMAZ, bkz. aşağıdaki kapsam sınırı).

ÖNEMLİ KAPSAM SINIRI (bu dosya yazıldığı tarihte, 2026-08-09
-- **AŞILDI: her iki blokaj da çözüldü, gerçek eğitim 2026-08-15'ten
beri koşuluyor; metin tarihsel kayıt olarak duruyor**):
Bu modüldeki fonksiyonlar GERÇEK ~721 (UPenn+LUMIERE) eğitimini
BAŞLATMAK için DEĞİL -- kod iskeleti + mekanik doğrulama içindir.
Gerçek eğitim şunlar tamamlanmadan başlamamalı:

  1. ``combat_parameters``'ın gerçek UPenn+LUMIERE fit çıktısı (bkz.
     decisions/2026-08-09-combat-fit-iki-blokaj-region-ve-pyradiomics-parametre.md
     -- Blokaj 2 hâlâ açık: LUMIERE'in PyRadiomics özellikleri
     binWidth=25/normalize=False/voxelArrayShift=0 ile yeniden
     çıkarılıyor).
  2. LUMIERE'in event/vital_status alanı (bkz.
     decisions/2026-08-06-lumiere-event-vital-status-eksik.md) -- bu
     dosyanın yazıldığı tarihte LUMIERE'in ``patients`` tablosunda
     ``vital_status`` TAMAMEN NULL (91/91, canlı SELECT ile
     doğrulandı), yani LUMIERE şu an gerçek bir Cox eğitimine
     event=0/1 sağlayamaz.

Bu iki blokaj çözülene kadar bu modül SADECE (a) sentetik veriyle
birim testi, (b) TCGA'nın kendi (harici test seti, asla eğitim
DEĞİL) canonical verisiyle MEKANİK smoke-test için kullanılmalı --
bkz. ``tools/smoke_test_cox_model.py``.

GİRİŞ NOKTASI (2026-08-13, TAKİP-3'te sertleştirildi -- bkz.
decisions/2026-08-12-cox-model-source-guard-canonicalization.md
"[2026-08-13] TAKİP-3"): Eğitim çerçevesi kurmak isteyen HER DIŞ
çağıran (Hafta 3'ün gerçek eğitim script'i dahil) ``build_training_
frame()``'i kullanmalı -- bu, whitelist guard'ı (``assert_training_
pool_sources()``) ile eğitim tablosu kurucusunu (artık PRIVATE
``_assemble_training_frame()``) doğru sırada zorunlu birlikte çağıran
TEK resmi giriş noktasıdır. Dürüst kapsam: varsayılan ve whitelist-
DARALTMA kullanımında (``allowed_sources`` `COX_TRAINING_ALLOWED_
SOURCES`'ın bir alt kümesi verilirse) TCGA/tanınmayan bir kaynak kod
seviyesinde reddedilir, ve `allowed_sources`'ın kendisini `COX_
TRAINING_ALLOWED_SOURCES`'ın ÖTESİNE GENİŞLETMEK (örn. TCGA eklemek)
de artık ayrıca reddedilir (Codex HIGH-2 bulgusu, TAKİP-3'te
kapatıldı). `_assemble_training_frame()`'e doğrudan erişim artık
private (alt çizgi ön-ekli) olduğu için KAZARA kullanım pratik olarak
imkânsız hale geldi; ama BİLİNÇLİ bypass (örn.
`tools/smoke_test_cox_model.py`'nin TCGA-ile-mekanik-test amacı) hâlâ
mümkün ve belgeli -- "guard hiçbir koşulda atlanamaz" gibi mutlak bir
iddia bu yüzden YANLIŞ olurdu, burada iddia edilmiyor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

# 2026-09-14 (B1, Barış onayı) -- ESKİ HÂLİ:
#   from pipeline.harmonization import _canonical_source_series, canonical_source
# `pipeline.harmonization` modül başında `import SimpleITK as sitk` yapar.
# Bu iki SAF yardımcıyı oradan almak, tahmin yolunun tamamını (cox_model →
# api/predict → api/analyze_patient → api/similar) 104 MB'lık bir GÖRÜNTÜ
# kütüphanesine bağımlı kılıyordu. 2026-09-14'te Windows Smart App Control
# `_SimpleITK.cp310-win_amd64.pyd`'yi engelleyince o gizli bağımlılık tüm
# servis yolunu düşürdü. `/predict` radyomiği DB'den okur, görüntü açmaz.
# Ölçüldü: beş kilitli regresyon pini de ayrıştırmadan sonra BİT-BİREBİR
# aynı (fark 0.0e+00). Ayrıntı: `pipeline/source_canonical.py` docstring'i.
from pipeline.source_canonical import _canonical_source_series, canonical_source

COX_TRAINING_ALLOWED_SOURCES = {"UPenn", "LUMIERE"}
DEFAULT_SOURCE_REFERENCE = "UPenn"


# =====================================================================
# C4 -- ICC>=0,60 stabilite filtresi (2026-08-14, Nisa)
# =====================================================================
#
# KİLİTLİ KARAR: decisions/2026-08-13-bolge-stratejisi-ve-modelleme-
# protokolu.md Bolum 4.1b -- "ICC >= 0,60 esigiyle filtre uygulanir ->
# 93 ozellik (14 shape + 4 first-order + 75 texture)."
#
# TÜRETME (bu listenin nasıl üretildiği, tekrarlanabilir olsun diye
# belgeleniyor -- prespesifikasyon disiplini gereği bu liste HER
# KOŞUDA yeniden hesaplanmaz, burada donmuş sabit olarak durur):
#   Kaynak dosya : artifacts/week3/pyradiomics/
#                  probe_perturbation_icc_per_feature.csv
#                  (Mert, 2026-08-13, C32 varyantı, yoğunluk ölçeği
#                  pertürbasyonu [0,4x/1,0x/2,5x] + sentetik bias-field,
#                  3 kaynak [TCGA/UPenn/LUMIERE] x 5 tarama = 15 tarama)
#   Filtre       : extraction_variant == "C32"
#   Agregasyon   : groupby(feature_class, feature_name)["icc_2_1"].mean()
#                  -- 3 kaynağın ORTALAMASI (karar dosyasının "3 kaynak
#                  ortalaması" ifadesiyle birebir tutarlı)
#   Eşik         : icc_2_1 >= 0,60
#   Doğrulama    : 2026-08-14'te bu modülü yazan ajan tarafından
#                  yukarıdaki agregasyon GERÇEKTEN çalıştırıldı (bkz.
#                  log/2026-08-14.md) -- sonuç tam olarak 93 satır
#                  (14 shape + 4 first-order + 75 texture) verdi,
#                  eleneni first-order'daki 14 mutlak-ölçek özellikle
#                  BİREBİR eşleşti (Energy/TotalEnergy/RMS/Variance/
#                  Mean/Median/Minimum/Maximum/Range/10Percentile/
#                  90Percentile/MeanAbsoluteDeviation/
#                  RobustMeanAbsoluteDeviation/InterquartileRange).
#                  Karar dosyasındaki kompozisyon (14+4+75=93) BOZULMADI
#                  -- eğer ölçüm dosyası değişir ve bu kompozisyon
#                  bozulursa, bu listeyi YENİDEN türetmeden önce
#                  Barış'a DURDUR-ve-raporla (CLAUDE.md/görev talimatı
#                  gereği "sayıyı zorlama").
#
# NOT -- İSİMLENDİRME: bu isimler PyRadiomics'in `original_<sınıf>_
# <özellik>` sözleşmesiyle birebir aynı (örn. "original_shape_
# Sphericity") -- `radiomics` tablosundaki `shape_features`/
# `first_order_features`/`texture_features` JSONB sözlüklerinin
# anahtarlarıyla DOĞRUDAN eşleşir (bkz. tools/smoke_test_cox_model.py
# CANONICAL_KEY_COUNTS + bu dosyanın aşağıdaki
# `pivot_radiomics_long_to_wide()` fonksiyonu).
STABLE_FEATURES_ICC60: tuple[str, ...] = (
    "original_firstorder_Entropy",
    "original_firstorder_Kurtosis",
    "original_firstorder_Skewness",
    "original_firstorder_Uniformity",
    "original_glcm_Autocorrelation",
    "original_glcm_ClusterProminence",
    "original_glcm_ClusterShade",
    "original_glcm_ClusterTendency",
    "original_glcm_Contrast",
    "original_glcm_Correlation",
    "original_glcm_DifferenceAverage",
    "original_glcm_DifferenceEntropy",
    "original_glcm_DifferenceVariance",
    "original_glcm_Id",
    "original_glcm_Idm",
    "original_glcm_Idmn",
    "original_glcm_Idn",
    "original_glcm_Imc1",
    "original_glcm_Imc2",
    "original_glcm_InverseVariance",
    "original_glcm_JointAverage",
    "original_glcm_JointEnergy",
    "original_glcm_JointEntropy",
    "original_glcm_MCC",
    "original_glcm_MaximumProbability",
    "original_glcm_SumAverage",
    "original_glcm_SumEntropy",
    "original_glcm_SumSquares",
    "original_gldm_DependenceEntropy",
    "original_gldm_DependenceNonUniformity",
    "original_gldm_DependenceNonUniformityNormalized",
    "original_gldm_DependenceVariance",
    "original_gldm_GrayLevelNonUniformity",
    "original_gldm_GrayLevelVariance",
    "original_gldm_HighGrayLevelEmphasis",
    "original_gldm_LargeDependenceEmphasis",
    "original_gldm_LargeDependenceHighGrayLevelEmphasis",
    "original_gldm_LargeDependenceLowGrayLevelEmphasis",
    "original_gldm_LowGrayLevelEmphasis",
    "original_gldm_SmallDependenceEmphasis",
    "original_gldm_SmallDependenceHighGrayLevelEmphasis",
    "original_gldm_SmallDependenceLowGrayLevelEmphasis",
    "original_glrlm_GrayLevelNonUniformity",
    "original_glrlm_GrayLevelNonUniformityNormalized",
    "original_glrlm_GrayLevelVariance",
    "original_glrlm_HighGrayLevelRunEmphasis",
    "original_glrlm_LongRunEmphasis",
    "original_glrlm_LongRunHighGrayLevelEmphasis",
    "original_glrlm_LongRunLowGrayLevelEmphasis",
    "original_glrlm_LowGrayLevelRunEmphasis",
    "original_glrlm_RunEntropy",
    "original_glrlm_RunLengthNonUniformity",
    "original_glrlm_RunLengthNonUniformityNormalized",
    "original_glrlm_RunPercentage",
    "original_glrlm_RunVariance",
    "original_glrlm_ShortRunEmphasis",
    "original_glrlm_ShortRunHighGrayLevelEmphasis",
    "original_glrlm_ShortRunLowGrayLevelEmphasis",
    "original_glszm_GrayLevelNonUniformity",
    "original_glszm_GrayLevelNonUniformityNormalized",
    "original_glszm_GrayLevelVariance",
    "original_glszm_HighGrayLevelZoneEmphasis",
    "original_glszm_LargeAreaEmphasis",
    "original_glszm_LargeAreaHighGrayLevelEmphasis",
    "original_glszm_LargeAreaLowGrayLevelEmphasis",
    "original_glszm_LowGrayLevelZoneEmphasis",
    "original_glszm_SizeZoneNonUniformity",
    "original_glszm_SizeZoneNonUniformityNormalized",
    "original_glszm_SmallAreaEmphasis",
    "original_glszm_SmallAreaHighGrayLevelEmphasis",
    "original_glszm_SmallAreaLowGrayLevelEmphasis",
    "original_glszm_ZoneEntropy",
    "original_glszm_ZonePercentage",
    "original_glszm_ZoneVariance",
    "original_ngtdm_Busyness",
    "original_ngtdm_Coarseness",
    "original_ngtdm_Complexity",
    "original_ngtdm_Contrast",
    "original_ngtdm_Strength",
    "original_shape_Elongation",
    "original_shape_Flatness",
    "original_shape_LeastAxisLength",
    "original_shape_MajorAxisLength",
    "original_shape_Maximum2DDiameterColumn",
    "original_shape_Maximum2DDiameterRow",
    "original_shape_Maximum2DDiameterSlice",
    "original_shape_Maximum3DDiameter",
    "original_shape_MeshVolume",
    "original_shape_MinorAxisLength",
    "original_shape_Sphericity",
    "original_shape_SurfaceArea",
    "original_shape_SurfaceVolumeRatio",
    "original_shape_VoxelVolume",
)

if len(STABLE_FEATURES_ICC60) != 93:  # pragma: no cover -- prespesifikasyon koruması
    raise RuntimeError(
        f"STABLE_FEATURES_ICC60 93 özellik içermeli, {len(STABLE_FEATURES_ICC60)} "
        "bulundu -- bu liste elle donduruldu (bkz. yukarıdaki TÜRETME notu), "
        "beklenmedik bir değişiklik/typo var demektir. DURDUR ve Barış'a raporla."
    )

# Elenen 14 mutlak-ölçek first-order özelliği -- SADECE dokümantasyon/
# regresyon-testi amaçlı (STABLE_FEATURES_ICC60'ın KENDİSİ zaten bunları
# içermiyor, bu liste "içermediğini" test etmek için var).
EXCLUDED_FIRST_ORDER_ABSOLUTE_SCALE_FEATURES: tuple[str, ...] = (
    "original_firstorder_Energy",
    "original_firstorder_TotalEnergy",
    "original_firstorder_RootMeanSquared",
    "original_firstorder_Variance",
    "original_firstorder_Maximum",
    "original_firstorder_Median",
    "original_firstorder_10Percentile",
    "original_firstorder_Mean",
    "original_firstorder_Range",
    "original_firstorder_90Percentile",
    "original_firstorder_MeanAbsoluteDeviation",
    "original_firstorder_RobustMeanAbsoluteDeviation",
    "original_firstorder_Minimum",
    "original_firstorder_InterquartileRange",
)


# =====================================================================
# C1 -- Bölge (ROI) isimlendirme + uzun->geniş pivot (2026-08-14, Nisa)
# =====================================================================
#
# Bağlam: decisions/2026-08-13-bolge-stratejisi-ve-modelleme-protokolu.md
# Bölüm 3 -- çıkarım UPenn/LUMIERE için NC/ED/ET + türetilmiş WT/TC,
# TCGA için native WT/TC üretir (henüz koşulmadı, bkz. CLAUDE.md
# "PyRadiomics C32" uyarısı -- bu fonksiyon SADECE altyapı, C32 verisi
# DB'ye yazılana kadar gerçek veriyle ÇALIŞTIRILMAMALI).
#
# Kaynak-bazlı ham bölge adı -> kanonik (WT/TC/NC/ED/ET) eşlemesi.
# UPenn/LUMIERE'in türetilmiş WT/TC satırları AKTIF-GOREVLER.md'nin
# A2 görev tanımına göre "WT_derived"/"TC_derived" olarak etiketlenecek
# (TCGA'nın native WT/TC'siyle karışmasın diye) -- bu eşleme tablosu bu
# ayrımı burada, tek yerde çözer. TCGA'nın kendi WT/TC'si DEĞİŞTİRİLMEDEN
# (birincil modelde kırpma YOK, bkz. karar dosyası Bölüm 2) kullanılır.
REGION_NAME_ALIASES: dict[str, dict[str, str]] = {
    "UPenn": {
        "NC": "NC",
        "ED": "ED",
        "ET": "ET",
        "WT_derived": "WT",
        "TC_derived": "TC",
    },
    "LUMIERE": {
        # DeepBraTumIA maskelerinin kendi native etiketleri -- UPenn'in
        # NC/ED/ET'sinden FARKLI (bkz. bu dosyayı çalıştıran görevin
        # kendi talimatı).
        "Necrosis": "NC",
        "Edema": "ED",
        "Contrast-enhancing": "ET",
        "WT_derived": "WT",
        "TC_derived": "TC",
    },
    "TCGA": {
        # TCGA'da yalnız WT/TC var, ET/NC/ED HİÇ üretilmez (kilitli
        # karar, bkz. CLAUDE.md "TCGA'da ET hiçbir zaman üretilmeyecek").
        "WT": "WT",
        "TC": "TC",
    },
    # 2026-09-12 EKLENDİ (canlı bug -- rag-agent turu, 8/8 UCSF hastasında
    # `/predict` "REGION_NAME_ALIASES kaynağı tanımıyor: 'UCSF'" ile 500
    # patlıyordu). Kök neden: `pipeline/harmonization.py::SOURCE_ALIASES`
    # 2026-08-18'de "ucsf-pdgm" -> "UCSF" eşlemesini eklemişti (N4/Z-score/
    # ComBat guard'ları için), ama bu sözlük hiç güncellenmemişti --
    # `canonical_source()` "UCSF" döndürüyor, burada karşılığı yoktu.
    # Canlı DB'de doğrulandı (readonly SELECT, 2026-09-12): UCSF'in
    # `radiomics.tumor_region` değerleri UPenn'inkiyle BİREBİR AYNI (NC/
    # ED/ET/WT_derived/TC_derived, segmentation_tool =
    # "UCSF-PDGM-PyRadiomics-107-C32", 295 hasta x 5 bölge) -- yani
    # eşleme UPenn'in bloğunun mekanik kopyası, yeni bir bölge semantiği
    # icat edilmedi. Bu blok SADECE EKLENDİ -- UPenn/LUMIERE/TCGA
    # blokları BİREBİR aynı kaldı (additif değişiklik, bkz.
    # tests/test_cox_model.py::test_canonicalize_region_label_accepts_ucsf_source).
    # NOT: UCSF `COX_TRAINING_ALLOWED_SOURCES = {"UPenn", "LUMIERE"}`'e
    # eklenmedi -- UCSF harici test setidir (CLAUDE.md 2026-08-18), bu
    # değişiklik UCSF'i Cox eğitim/CV havuzuna SOKMAZ, yalnız
    # `canonicalize_region_label()`/`pivot_radiomics_long_to_wide()`
    # zincirinin (dolayısıyla `api/predict.py` tahmin yolunun) UCSF'i
    # tanımasını sağlar.
    "UCSF": {
        "NC": "NC",
        "ED": "ED",
        "ET": "ET",
        "WT_derived": "WT",
        "TC_derived": "TC",
    },
}

PRIMARY_REGIONS: tuple[str, ...] = ("WT",)
SENSITIVITY_REGIONS_WT_TC: tuple[str, ...] = ("WT", "TC")
SENSITIVITY_REGIONS_TC_ONLY: tuple[str, ...] = ("TC",)
EXPLORATORY_REGIONS_NC_ED_ET: tuple[str, ...] = ("NC", "ED", "ET")


def canonicalize_region_label(source: str, raw_region: str) -> str:
    """Kaynağa özgü ham `tumor_region` değerini kanonik etikete çevirir.

    Kanonik etiketler: `"WT"`, `"TC"`, `"NC"`, `"ED"`, `"ET"`.
    `source`, `pipeline.harmonization.canonical_source()` ile önce
    kanonikleştirilir (ham DB alias'ları -- `"UPenn-GBM"` gibi -- kabul
    edilir). Tanınmayan (kaynak, ham-bölge) çifti SESSİZCE geçilmez --
    `ValueError` fırlatılır (proje kuralı: sessiz fallback yasak, ayrıca
    bu görevin kendi talimatı: "tanınmayan bölge/kaynak exception
    fırlatmalı").
    """

    canonical = canonical_source(source)
    try:
        source_region_map = REGION_NAME_ALIASES[canonical]
    except KeyError as exc:  # pragma: no cover -- canonical_source zaten
        # sadece TCGA/UPenn/LUMIERE döndürür, REGION_NAME_ALIASES bu
        # üçünü de tanımlıyor -- bu dal teorik olarak ulaşılamaz, yine
        # de sessiz KeyError yerine açık ValueError'a çeviriyoruz.
        raise ValueError(
            f"REGION_NAME_ALIASES kaynağı tanımıyor: {canonical!r}"
        ) from exc

    try:
        return source_region_map[raw_region]
    except KeyError as exc:
        raise ValueError(
            f"REGION_NAME_ALIASES[{canonical!r}] ham bölge adını "
            f"tanımıyor: {raw_region!r}. Bilinen bölgeler: "
            f"{sorted(source_region_map)}. Bu, ya bir veri hatası "
            "(yazım hatası / beklenmeyen segmentation_tool çıktısı) "
            "ya da REGION_NAME_ALIASES'in güncellenmesi gereken yeni "
            "bir bölge adı anlamına gelir -- sessizce NC/ED/ET/WT/TC'den "
            "birine eşlenmez."
        ) from exc


class RegionPivotError(RuntimeError):
    """`pivot_radiomics_long_to_wide()` sırasında çakışma/eksik-bölge hatası."""


@dataclass
class RegionPivotReport:
    """`pivot_radiomics_long_to_wide()` çıktısının şeffaf denetim izi."""

    regions_requested: list[str]
    n_input_rows: int
    n_candidate_patients: int
    n_output_patients: int
    dropped_patients_missing_region: dict[str, int]


def pivot_radiomics_long_to_wide(
    long_frame: pd.DataFrame,
    regions: Iterable[str],
    *,
    patient_col: str = "patient_id",
    source_col: str = "source",
    region_col: str = "tumor_region",
    feature_dict_cols: tuple[str, str, str] = (
        "shape_features",
        "first_order_features",
        "texture_features",
    ),
    on_missing_region: str = "drop",
) -> tuple[pd.DataFrame, RegionPivotReport]:
    """`radiomics` tablosunun UZUN formatını hasta-başına GENİŞ matrise çevirir.

    Girdi (`long_frame`): her satır bir (hasta, kaynak, bölge) -- yani
    `radiomics` tablosunun `(scan_id, segmentation_tool, tumor_region)`
    satırlarının `mr_scans`/`patients` ile join edilip `scan_id`'nin
    `patient_id`'ye indirgendiği, ÇAĞIRAN TARAFIN önceden hazırladığı
    bir DataFrame. Bu fonksiyon DB'ye BAĞLANMAZ, SQL YAZMAZ -- saf bir
    veri dönüşümüdür (test edilebilirlik + görev talimatı: "sadece
    ALTYAPI kodu, DB'ye HİÇBİR YAZMA YOK" gereği; SELECT/join mantığı
    ayrı, bu fonksiyonu çağıran bir script/db-agent görevinin işi).

    Bir hastanın aynı `(patient_id, kanonik_bölge)` çifti için BİRDEN
    FAZLA satırı varsa (örn. iki farklı `segmentation_tool` aynı bölge
    için satır üretmiş) -- ortalama alma/rastgele seçim YAPILMAZ,
    `RegionPivotError` fırlatılır (çağıran taraf hangi `segmentation_tool`
    kullanılacağını ÖNCEDEN, SQL'de netleştirmeli).

    Bir hastada istenen `regions`'tan biri eksikse (`on_missing_region`):
      - `"drop"` (varsayılan): o hasta çıktıdan düşürülür, kaç hastanın
        hangi bölge eksikliğinden düştüğü `RegionPivotReport`'a
        SAYILARAK raporlanır (CLAUDE.md "sessizce filtrelenmez" ilkesi
        -- rapor var, gizli değil). `TrainingFrameReport`'un desenini
        izler (bkz. `_assemble_training_frame`).
      - `"raise"`: İLK eksik-bölgeli hastada `RegionPivotError`
        fırlatılır (kısmi/eksik bir eğitim çerçevesinin sessizce
        oluşmasını istemeyen çağıranlar için, örn. duyarlılık kollarında
        "hiçbir hasta kaybetmeyelim" politikası).
      SESSİZ NaN doldurma HİÇBİR MODDA yapılmaz (görev talimatı).

    Çıktı kolon adları bölge-önekli: `f"{kanonik_bölge}__{özellik_adı}"`
    (örn. `"WT__original_shape_Sphericity"`) -- çok-bölgeli kollarda
    (örn. WT+TC) aynı isimli özelliklerin çakışmasını önler.

    NOT -- eksik özellik anahtarı (bir bölge satırı var ama JSONB
    sözlüğü PyRadiomics'in beklenen tam anahtar kümesini içermiyor,
    örn. başarısız/kısmi extraction): bu fonksiyon bunu KONTROL ETMEZ
    -- yalnız bölge-düzeyinde varlık/yokluk kontrol edilir. Çağıran
    taraf (gerçek Hafta 3 script'i) `tools/smoke_test_cox_model.py`'nin
    `CANONICAL_KEY_COUNTS` desenini izleyerek satır-düzeyinde tam
    anahtar sayısı doğrulaması AYRICA yapmalı -- bu AÇIK bir kapsam
    sınırı, sessizce varsayılmıyor.
    """

    regions = list(regions)
    if not regions:
        raise ValueError("regions boş olamaz.")
    if on_missing_region not in {"drop", "raise"}:
        raise ValueError(
            "on_missing_region 'drop' ya da 'raise' olmalı, alındı: "
            f"{on_missing_region!r}"
        )

    required_cols = {patient_col, source_col, region_col, *feature_dict_cols}
    missing_cols = required_cols - set(long_frame.columns)
    if missing_cols:
        raise ValueError(f"long_frame eksik kolon(lar): {sorted(missing_cols)}")

    n_input_rows = len(long_frame)
    working = long_frame.copy()
    # Tüm satırlar kanonikleştirilir (yalnız istenen `regions` DEĞİL) --
    # eşleme tablosunun eksik/hatalı olduğu bir satır, o satır sonradan
    # filtrelenecek olsa bile SESSİZCE geçilmemeli (fail-loud ilkesi).
    working["_canonical_region"] = [
        canonicalize_region_label(src, region)
        for src, region in zip(working[source_col], working[region_col])
    ]

    all_patients_full = sorted(long_frame[patient_col].unique())
    n_candidate_patients = len(all_patients_full)

    filtered = working[working["_canonical_region"].isin(regions)].copy()

    duplicate_mask = filtered.duplicated(
        subset=[patient_col, "_canonical_region"], keep=False
    )
    if duplicate_mask.any():
        duplicate_counts = (
            filtered.loc[duplicate_mask, "_canonical_region"].value_counts().to_dict()
        )
        raise RegionPivotError(
            "long_frame'de aynı (hasta, kanonik bölge) çifti için birden "
            "fazla satır var -- pivot ÇAKIŞMASI, ortalama alma/rastgele "
            "seçim YAPILMAZ. Çağıran taraf önceden tekilleştirmeli "
            "(muhtemel neden: birden fazla segmentation_tool aynı hasta+"
            f"bölge için satır üretmiş). Bölge kırılımı: {duplicate_counts}"
        )

    groups = {pid: rows for pid, rows in filtered.groupby(patient_col)}

    wide_records: dict[str, dict[str, float]] = {}
    dropped_missing_region: dict[str, int] = {region: 0 for region in regions}

    for patient_id in all_patients_full:
        patient_rows = groups.get(patient_id)
        present_regions = (
            set(patient_rows["_canonical_region"]) if patient_rows is not None else set()
        )
        missing = [region for region in regions if region not in present_regions]
        if missing:
            if on_missing_region == "raise":
                raise RegionPivotError(
                    f"Hasta {patient_id!r} istenen bölge(ler)den eksik: "
                    f"{missing} (on_missing_region='raise')."
                )
            for region in missing:
                dropped_missing_region[region] += 1
            continue

        patient_features: dict[str, float] = {}
        for _, row in patient_rows.iterrows():
            prefix = f"{row['_canonical_region']}__"
            for dict_col in feature_dict_cols:
                feature_dict = row[dict_col] or {}
                for feature_name, value in feature_dict.items():
                    patient_features[f"{prefix}{feature_name}"] = value
        wide_records[patient_id] = patient_features

    wide_frame = pd.DataFrame.from_dict(wide_records, orient="index")
    wide_frame.index.name = patient_col
    wide_frame = wide_frame.sort_index()

    report = RegionPivotReport(
        regions_requested=regions,
        n_input_rows=n_input_rows,
        n_candidate_patients=n_candidate_patients,
        n_output_patients=len(wide_records),
        dropped_patients_missing_region=dropped_missing_region,
    )
    return wide_frame, report


class CoxTrainingLeakageError(RuntimeError):
    """Eğitim/CV havuzuna yasak bir kaynak (örn. TCGA) sızdı.

    `fit_combat_harmonization()`'daki `ComBatFitLeakageError` ile aynı
    mimari kuralın Cox tarafındaki karşılığı -- CLAUDE.md: "Cox PHM +
    XGBoost eğitimi SADECE UPenn-GBM + LUMIERE ile yapılır."
    """


@dataclass
class TrainingFrameReport:
    """`_assemble_training_frame()` çıktısının şeffaf denetim izi."""

    n_input_rows: int
    n_output_rows: int
    dropped_missing_event: int
    dropped_missing_duration: int
    dropped_missing_features: int
    dropped_unrecognized_vital_status: dict[str, int]
    sources: dict[str, int]
    # 2026-08-15 EKLENDİ (klinik kovaryat desteği): `passthrough_columns`
    # (örn. yaş/cinsiyet/GTR/IDH gibi ZORUNLU klinik kovaryatlar --
    # elastic-net/stabilite seçiminden GEÇMEZ, ayrı bir eksik-değer
    # sayacı gerekir çünkü bunlar `feature_columns`'DAN AYRI bir kavram
    # (radyomik "aday" değil, her zaman modelde kalan düzeltme terimi).
    # Varsayılan 0 -- `passthrough_columns` verilmezse eski davranış
    # BİREBİR korunur, mevcut çağıranlar/testler ETKİLENMEZ.
    dropped_missing_passthrough: int = 0
    # 2026-08-19 EKLENDİ (bölge-farkında eğitim havuzu kapısı, Barış
    # talimatı -- bkz. gunluk-rapor-2026-08-18.md §Riskler-1 ve
    # `tools/train_cox_week3.py::check_training_pool_counts`):
    # SAYI yetmiyor, KİMLİK de gerekiyor. `v2c_mgmt_spline_wttc`
    # (WT+TC) koşusunda havuz 611/585 yerine 609/583 ölçüldü ve
    # kapı DOĞRU şekilde durdu; ama "hangi 2 hasta düştü" sorusunu
    # yanıtlamak için canlı DB'ye AYRI bir sorgu gerekti. Bu alan o
    # denetim izini raporun İÇİNE koyar: gerekçe -> düşen patient_id
    # listesi (sıralı). Anahtarlar `dropped_*` sayaçlarıyla BİREBİR
    # eşleşir: "unrecognized_vital_status" / "missing_duration" /
    # "missing_features" / "missing_passthrough". Bir gerekçede hiç
    # düşen yoksa o anahtar SÖZLÜKTE BULUNMAZ (boş liste yazılmaz).
    # Varsayılan boş sözlük -- mevcut çağıranlar/testler ETKİLENMEZ.
    dropped_patient_ids: dict[str, list[str]] = field(default_factory=dict)


def verify_event_counts(connection) -> pd.DataFrame:
    """"Event-sayısı (ölüm/nüks) doğrulama sorgusu" -- plan.txt madde 436.

    GERÇEK, canlı (readonly) sorgudur -- bu fonksiyon smoke-test/iskelet
    DEĞİL, istatistiksel güç iddiasını gerçek sayıya bağlamak için
    doğrudan kullanılabilir. `connection` çağıran tarafın
    `db_connection.get_connection(readonly=True)` ile açtığı bağlantı
    olmalı (bu modül DB bağlantı yönetimini kendi üstlenmez).
    """

    from psycopg2.extras import RealDictCursor

    cursor = connection.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(
            """
            SELECT ds.source_name AS source,
                   p.vital_status,
                   COUNT(*) AS n_patients,
                   COUNT(*) FILTER (WHERE p.survival_days IS NOT NULL) AS n_with_duration
            FROM patients p
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE ds.source_name IN ('UPenn-GBM', 'LUMIERE', 'TCGA-GBM')
            GROUP BY ds.source_name, p.vital_status
            ORDER BY ds.source_name, n_patients DESC
            """
        )
        rows = cursor.fetchall()
    finally:
        cursor.close()
    return pd.DataFrame(rows)


def _validate_allowed_sources(allowed_sources: set[str]) -> None:
    """`allowed_sources`'ın `COX_TRAINING_ALLOWED_SOURCES`'ın GENİŞLETİLMEMİŞ
    bir alt kümesi olduğunu doğrula (2026-08-13, Codex HIGH-2 bulgusu --
    bkz. decisions/2026-08-12-cox-model-source-guard-canonicalization.md
    "[2026-08-13] TAKİP-3").

    Bağlam: `assert_training_pool_sources()`/`build_training_frame()`'in
    `allowed_sources` parametresi bir SABİT değil, çağıranın geçirebildiği
    bir parametredir. Bu kasıtlı -- örn. sadece UPenn-only ya da sadece
    LUMIERE-only alt-küme testleri için whitelist'i DARALTMAK meşru bir
    kullanım. Ama bu esneklik aynı zamanda bir çağıranın (yanlışlıkla ya
    da kötü niyetle) `allowed_sources={"UPenn", "LUMIERE", "TCGA"}` gibi
    bir değer geçirip whitelist'i GENİŞLETMESİNE de izin veriyordu -- bu
    durumda guard'ın kendisi TCGA'yı sessizce KABUL ederdi, CLAUDE.md'nin
    en üst mimari kuralını ("TCGA eğitime KESİNLİKLE girmez") ihlal
    ederdi. Bu fonksiyon SADECE daraltmaya izin verir, genişletmeyi kod
    seviyesinde reddeder: `allowed_sources - COX_TRAINING_ALLOWED_SOURCES`
    boş değilse (yani `allowed_sources` içinde `COX_TRAINING_ALLOWED_
    SOURCES`'ta olmayan bir eleman varsa -- TCGA dahil, ya da tamamen
    hayali bir kaynak adı) `CoxTrainingLeakageError` fırlatır.
    """

    invalid = allowed_sources - COX_TRAINING_ALLOWED_SOURCES
    if invalid:
        raise CoxTrainingLeakageError(
            "allowed_sources parametresi COX_TRAINING_ALLOWED_SOURCES "
            f"({sorted(COX_TRAINING_ALLOWED_SOURCES)}) kaynaklarının bir "
            "ALT KÜMESİ olmalı -- bu parametre whitelist'i sadece "
            "DARALTMAK (örn. sadece UPenn ya da sadece LUMIERE alt-küme "
            "testleri) için var, GENİŞLETMEK için değil. Kabul edilmeyen "
            f"kaynak(lar): {sorted(invalid)}. TCGA (veya tanınmayan başka "
            "bir kaynak) allowed_sources'a asla EKLENEMEZ -- CLAUDE.md "
            "kritik mimari kural."
        )


class ComBatIdentityAssumptionError(RuntimeError):
    """ComBat'ın Cox eğitim zincirinde ÖZDEŞLİK dönüşümü olduğu varsayımı bozuldu.

    Bkz. decisions/2026-08-13-combat-ozdeslik-bulgusu-ve-kapsami.md
    "TESPİT EDİLEN MAYIN" bölümü.
    """


def assert_combat_identity_for_training_pool(
    patients_frame: pd.DataFrame,
    *,
    source_col: str = "source",
    reference_batch: str = DEFAULT_SOURCE_REFERENCE,
) -> None:
    """C5 -- ComBat'ın eğitim havuzuna ÖZDEŞLİK uyguladığı varsayımını doğrula.

    BAĞLAM (2026-08-13, kilitli bulgu -- bkz. decisions/2026-08-13-
    combat-ozdeslik-bulgusu-ve-kapsami.md): `neuroHarmonize` referans-
    batch tasarımında referans batch için `gamma_star=0`/`delta_star=1`
    olduğundan, ComBat referans batch'in KENDİ verisine cebirsel olarak
    HİÇBİR değişiklik uygulamaz (ampirik doğrulandı: maksimum fark 0,0).
    Bugünkü konfigürasyonda eğitim havuzu YALNIZ UPenn (LUMIERE olay
    verisi sağlayamıyor, bkz. decisions/2026-08-06-lumiere-event-vital-
    status-eksik.md) ve ComBat referans batch'i de UPenn -- yani eğitim
    özellikleri fiilen HAM (C32-harmonize) özellikler, TCGA (harici
    test, ComBat'sız) ile AYNI uzayda. Bu, "eğitim/test asimetrisi YOK"
    hükmünün TEK dayanağı.

    KIRILGANLIK: bu özdeşlik yalnız `eğitim havuzundaki TÜM kaynaklar ==
    reference_batch` olduğu sürece geçerli. Biri ileride (a) LUMIERE'i
    (event verisi bulunursa/RANO-PD tabanlı bir endpoint'e geçilirse)
    eğitim havuzuna sokarsa, veya (b) `reference_batch`'i UPenn'den
    başka bir şeye çevirirse, ComBat artık eğitim verisine GERÇEK bir
    dönüşüm uygular -- ama TCGA (harici test) hâlâ HİÇBİR ComBat
    dönüşümü almadığından (CLAUDE.md: "ComBat harmonizasyon modeli
    SADECE UPenn+LUMIERE'i görür, TCGA HARİÇ"), eğitim/test uzay
    ASİMETRİSİ doğar -- Codex'in CRITICAL uyarısı aynen gerçekleşir.

    Bu fonksiyon `patients_frame[source_col]`'daki (kanonikleştirilmiş)
    TÜM kaynakların `reference_batch`'e eşit olduğunu assert eder.
    Eşit değilse SESSİZCE devam edilmez -- `ComBatIdentityAssumptionError`
    fırlatılır.

    ÖNEMLİ SINIR (dürüstçe belirtiliyor): bu fonksiyon ComBat'ın
    GERÇEKTEN özdeşlik uyguladığını yeniden hesaplayıp KANITLAMAZ (bu,
    `pipeline.harmonization.fit_combat_harmonization()`'ın kendi işi,
    ayrı bir modül) -- yalnız yukarıdaki KANITLANMIŞ cebirsel özdeşliğin
    ÖN KOŞULUNU (eğitim havuzu == reference_batch) kod seviyesinde
    zorlar. `reference_batch` parametresinin GERÇEKTEN üretimdeki
    `fit_combat_harmonization(reference_batch=...)` çağrısıyla aynı
    değeri taşıdığını doğrulamak ÇAĞIRANIN sorumluluğundadır (bu
    fonksiyon `pipeline.harmonization`'a bağımlı DEĞİL, dairesel import
    riski yok -- yalnız bu modülün zaten import ettiği
    `canonical_source`/`_canonical_source_series` kullanılır).
    """

    if source_col not in patients_frame.columns:
        raise ValueError(f"patients_frame '{source_col}' kolonu içermeli.")

    try:
        canonical_series = _canonical_source_series(patients_frame[source_col])
    except ValueError as exc:
        raise ComBatIdentityAssumptionError(
            "ComBat özdeşlik guard'ı çalıştırılamadı -- "
            f"patients_frame['{source_col}'] içinde `SOURCE_ALIASES`'in "
            f"tanımadığı bir kaynak adı bulundu. Orijinal hata: {exc}"
        ) from exc

    canonical_reference = canonical_source(reference_batch)
    sources_present = set(canonical_series.unique())
    non_reference_sources = sources_present - {canonical_reference}
    if non_reference_sources:
        raise ComBatIdentityAssumptionError(
            "ComBat'ın Cox eğitim zincirinde ÖZDEŞLİK dönüşümü olduğu "
            f"varsayımı BOZULDU: eğitim havuzunda referans batch "
            f"({canonical_reference!r}) DIŞINDA kaynak(lar) var: "
            f"{sorted(non_reference_sources)}. Bu durumda ComBat artık "
            "özdeşlik değil GERÇEK bir dönüşüm uygular, ama TCGA (harici "
            "test) hiçbir ComBat dönüşümü almıyor (CLAUDE.md: ComBat "
            "modeli SADECE UPenn+LUMIERE'i görür) -- bu eğitim/test uzay "
            "ASİMETRİSİ yaratır (bkz. decisions/2026-08-13-combat-"
            "ozdeslik-bulgusu-ve-kapsami.md 'TESPİT EDİLEN MAYIN'). "
            "Sessizce devam EDİLEMEZ -- ya reference_batch bu kaynağı da "
            "kapsayacak şekilde yeniden değerlendirilmeli (ve ComBat'ın "
            "artık özdeşlik OLMAYAN gerçek etkisi TCGA'ya karşı ayrıca "
            "doğrulanmalı), ya da bu kaynak eğitim havuzundan çıkarılmalı."
        )


def assert_training_pool_sources(
    covariates: pd.DataFrame,
    *,
    source_col: str = "source",
    allowed_sources: set[str] = COX_TRAINING_ALLOWED_SOURCES,
) -> None:
    """CV/fit'e TCGA (veya başka bir yasak kaynak) sızmadığını doğrula.

    Not (2026-08-13): Hafta 3 gerçek eğitim script'i DAHİL, bu modülün
    DIŞINDAN gelen çağıranlar bu fonksiyonu `_assemble_training_frame()`
    ile AYRI AYRI ELLE sıralamak yerine `build_training_frame()`'i
    (bu ikisini doğru sırada zorunlu birlikte çağıran tek resmi giriş
    noktası) KULLANMALI -- bkz. decisions/2026-08-12-cox-model-source-
    guard-canonicalization.md "[2026-08-13] TAKİP-2" bölümü. Bu
    fonksiyon geriye dönük uyumluluk için ayrı çağrılabilir kalmaya
    devam eder (örn. testlerde), ama üretim kodu için tercih edilen
    yol `build_training_frame()`'dir.

    `pipeline.harmonization.fit_combat_harmonization()`'daki sızıntı
    korumasıyla aynı desen -- Cox eğitim havuzu için tekrarlanıyor
    çünkü ComBat ve Cox ayrı işlem adımları, ikisinin de kendi
    bağımsız korumasına ihtiyacı var (birinin koruması diğerini
    kapsamaz).

    Not (2026-08-12, `fit_combat_harmonization()`'ın FIX-3'ünde
    düzeltilen sorunla BİREBİR AYNI hata sınıfı -- bkz.
    decisions/2026-08-12-cox-model-source-guard-canonicalization.md):
    bu guard önceden ``covariates[source_col]``'ı HAM literal string
    olarak ``COX_TRAINING_ALLOWED_SOURCES = {"UPenn", "LUMIERE"}`` ile
    karşılaştırıyordu. `dataset_sources.source_name` DB kolonunun
    GERÇEK değeri literal ``"UPenn-GBM"``/``"LUMIERE"``/``"TCGA-GBM"``
    olduğundan (bkz. bu dosyanın `verify_event_counts()` sorgusu,
    satır ~98), bu ham karşılaştırma MEŞRU UPenn verisini de
    yanlışlıkla reddederdi (fail-safe ama üretimi kıran bir hata,
    yanlış-pozitif). Guard artık
    `pipeline.harmonization._canonical_source_series()` (aynı
    `SOURCE_ALIASES`/`canonical_source()` normalizasyonunu ComBat
    guard'larıyla PAYLAŞIR, kod tekrarı yok) ile normalize edilmiş
    değerler üzerinden çalışır: ``"UPenn-GBM"``, ``"upenn-gbm"``,
    ``" UPenn-GBM "`` gibi varyantlar kanonik ``"UPenn"``'e eşlenip
    KABUL edilir; ``"TCGA"``, ``"TCGA-GBM"``, ``"tcga-gbm"`` gibi tüm
    TCGA varyantları hâlâ REDDEDİLİR. Whitelist mantığı KORUNDU:
    `SOURCE_ALIASES`'in hiç tanımadığı bir kaynak adı (yazım hatası,
    tamamen bilinmeyen bir kohort) da whitelist gereği REDDEDİLİR
    (sessiz kabul yasak) -- `_canonical_source_series()`'in fırlattığı
    `ValueError` burada yakalanıp `CoxTrainingLeakageError`'a çevrilir.

    `allowed_sources` parametresi zaten kanonik değerler taşımalı
    (varsayılan `COX_TRAINING_ALLOWED_SOURCES` bu şekildedir) --
    fonksiyon bu parametreyi kanonikleştirmez, çağıranın sorumluluğu.

    Not (2026-08-13, Codex HIGH-2 bulgusu kapatıldı -- bkz.
    decisions/2026-08-12-cox-model-source-guard-canonicalization.md
    "[2026-08-13] TAKİP-3"): `allowed_sources` bir SABİT değil, bir
    parametre olduğu için önceden bir çağıran
    `allowed_sources={"UPenn", "LUMIERE", "TCGA"}` geçirip whitelist'i
    GENİŞLETEBİLİRDİ -- bu durumda guard TCGA'yı SESSİZCE kabul ederdi.
    Bu fonksiyon artık ÖNCE `_validate_allowed_sources()` ile
    `allowed_sources`'ın `COX_TRAINING_ALLOWED_SOURCES`'ın bir ALT
    KÜMESİ olduğunu zorunlu kılar -- parametrenin var olma amacı
    (whitelist'i DARALTMAK, örn. sadece UPenn-only test) korunur,
    sadece GENİŞLETME (TCGA veya tanınmayan bir kaynak eklemek)
    reddedilir.
    """

    _validate_allowed_sources(allowed_sources)

    if source_col not in covariates.columns:
        raise ValueError(f"covariates '{source_col}' kolonu içermeli.")

    try:
        canonical_series = _canonical_source_series(covariates[source_col])
    except ValueError as exc:
        raise CoxTrainingLeakageError(
            "Cox eğitim/CV havuzu SADECE "
            f"{sorted(allowed_sources)} kaynaklarını görebilir "
            "(whitelist). "
            f"covariates['{source_col}'] içinde `SOURCE_ALIASES`'in "
            "tanımadığı bir kaynak adı bulundu -- whitelist mantığı "
            "gereği tanınmayan her şey reddedilir (sessiz kabul "
            f"yasak). Orijinal hata: {exc}"
        ) from exc

    sources_present = set(canonical_series.unique())
    forbidden = sources_present - allowed_sources
    if forbidden:
        raise CoxTrainingLeakageError(
            "Cox eğitim/CV havuzu SADECE "
            f"{sorted(allowed_sources)} kaynaklarını görebilir. "
            "Yasak kaynak(lar) (kanonikleştirilmiş) bulundu: "
            f"{sorted(forbidden)}. TCGA eğitime/CV'ye ASLA girmemeli "
            "-- CLAUDE.md kritik mimari kural (bkz. "
            "evaluate_external_test() TCGA için ayrı yol)."
        )


def build_training_frame(
    feature_frame: pd.DataFrame,
    patients_frame: pd.DataFrame,
    *,
    source_col: str = "source",
    duration_col: str = "survival_days",
    event_col: str = "event",
    vital_status_col: str = "vital_status",
    deceased_labels: Iterable[str] = ("DECEASED",),
    # "CENSORED" 2026-08-14'te EKLENDİ (koordinatör): canlı DB'de UPenn'in
    # `vital_status` alanı ÜÇ değer taşıyor -- DECEASED 585 / ALIVE 17 /
    # CENSORED 9. Önceki varsayılan yalnız ("ALIVE",) olduğu için 9 CENSORED
    # hasta "tanınmayan statü" sayılıp SESSİZCE düşürülüyordu ve eğitim havuzu
    # 611 yerine 602 oluyordu (gerçek koşuda 611/585 gate'i bunu yakaladı).
    # Sağkalım analizinde sansürlenmiş hasta ATILMAZ: olay=0 olarak modele
    # girer ve takip süresi boyunca "olay gözlenmedi" bilgisini taşır.
    # ALIVE ve CENSORED'ın ikisi de "takip sonuna kadar olay yok" demektir.
    censored_labels: Iterable[str] = ("ALIVE", "CENSORED"),
    source_reference: str = DEFAULT_SOURCE_REFERENCE,
    allowed_sources: set[str] = COX_TRAINING_ALLOWED_SOURCES,
    check_combat_identity: bool = True,
    passthrough_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, TrainingFrameReport]:
    """Cox eğitim çerçevesi kurmak için TEK resmi giriş noktası.

    `passthrough_columns` (2026-08-15 EKLENDİ, klinik kovaryat desteği):
    `patients_frame`'den, dummy-encoding YAPILMADAN, OLDUĞU GİBİ Cox
    çerçevesine taşınacak kolon adları (örn. `clinical_age`,
    `clinical_gender_male`, `clinical_gtr_y`, `clinical_gtr_missing`,
    `clinical_idh_mutant` -- bkz. `tools/train_cox_week3.py`'nin klinik
    kovaryat inşa fonksiyonları). Bunlar `feature_columns` GİBİ elastic-
    net/stabilite seçimine SOKULMAZ -- çağıran taraf bunları
    `run_nested_cv()`/`select_features_lasso()`'nun `extra_columns`
    parametresine verir (radyomik `feature_columns`'DAN AYRI bir kavram).
    Bu fonksiyon SADECE bu kolonları `feature_frame`/`patients_frame`
    join'inden Cox çerçevesine SESSİZCE KAYBETMEDEN taşır ve eksik
    değerli satırları (radyomik özellik eksikliğiyle AYNI disiplinle)
    düşürüp `TrainingFrameReport.dropped_missing_passthrough`'a sayar.
    Varsayılan `None` -- eski davranış BİREBİR korunur.

    Bağlam (2026-08-13, reviewer bulgusu -- bkz.
    decisions/2026-08-12-cox-model-source-guard-canonicalization.md
    "[2026-08-13] TAKİP-2" bölümü): `assert_training_pool_sources()`
    (whitelist guard'ı) ile `_assemble_training_frame()` (eğitim
    tablosunu kuran, dummy-encoding yapan fonksiyon) BİLİNÇLİ olarak
    ayrı iki fonksiyondu -- `_assemble_training_frame()` whitelist/
    TCGA-reddi UYGULAMAZ (kasıtlı kapsam sınırı, docstring'inde
    açıkça yazılı). Bu ayrım kendi başına YANLIŞ değil, ama
    `assert_training_pool_sources()`'ın hiçbir üretim çağıranı
    OLMADIĞI (grep ile doğrulandı, yalnız testlerde çağrılıyor)
    gerçeğiyle birleşince SÜREGELEN bir risk oluşturuyordu: Hafta 3'ün
    gerçek eğitim script'ini yazan biri sadece
    `_assemble_training_frame()`'i çağırıp guard'ı çağırmayı UNUTURSA, TCGA verisi
    `source_TCGA` dummy kolonuyla SESSİZCE eğitim havuzuna girebilirdi
    -- CLAUDE.md'nin en üst mimari kuralının ("Cox PHM + XGBoost
    eğitimi SADECE UPenn-GBM + LUMIERE ile yapılır, TCGA KESİNLİKLE
    girmez") kodda değil sadece disiplin/hafızada ZORLANMASI anlamına
    gelirdi.

    Bu fonksiyon iki alt fonksiyonu DOĞRU SIRADA (önce guard, sonra
    assemble) ZORUNLU birlikte çağırır -- guard, `_assemble_training_
    frame()`'e hiç ulaşılmadan önce çalışır ve yasak bir kaynak varsa
    `CoxTrainingLeakageError` fırlatıp fonksiyonu erken sonlandırır.
    Hafta 3'ün gerçek eğitim script'i DAHİL, bu modülün DIŞINDAN çağıran
    HER KOD bu fonksiyonu KULLANMALI -- `assert_training_pool_sources()`/
    `_assemble_training_frame()`'i AYRI AYRI doğrudan çağırmak yerine.

    DÜRÜST KAPSAM (2026-08-13, Codex HIGH-1/HIGH-2 sertleştirmesi --
    bkz. decisions/2026-08-12-cox-model-source-guard-canonicalization.md
    "[2026-08-13] TAKİP-3"): "guard atlanamaz / sızıntı imkânsız" gibi
    MUTLAK bir iddia YANLIŞ olurdu, burada iddia edilmiyor. Doğru
    kapsam iki parçalı:
    (1) Varsayılan ve whitelist-DARALTMA kullanımında (`allowed_sources`
        `COX_TRAINING_ALLOWED_SOURCES`'ın bir alt kümesi verilirse,
        `_validate_allowed_sources()` bunu zorunlu kılar -- GENİŞLETMEK,
        örn. `allowed_sources={"UPenn","LUMIERE","TCGA"}`, artık ayrıca
        reddedilir) TCGA/tanınmayan bir kaynak kod seviyesinde
        REDDEDİLİR.
    (2) `_assemble_training_frame()`'e (artık private, alt çizgi
        ön-ekli) doğrudan erişim KAZARA olmaz hale geldi -- ama
        BİLİNÇLİ bypass (örn. `tools/smoke_test_cox_model.py`'nin
        TCGA-ile-mekanik-test amacı, kendi docstring'inde NEDEN
        belgeli) Python'da hâlâ MÜMKÜN ve KASITLI olarak engellenmedi
        (private isimlendirme bir erişim-kontrol mekanizması DEĞİL,
        bir "bunu doğrudan çağırma" sözleşmesidir).

    `assert_training_pool_sources()` (guard) `patients_frame`
    ÜZERİNDE, `_assemble_training_frame()`'in yaptığı `feature_frame`
    ile inner join'den ÖNCE çalışır -- yani yasak bir kaynak,
    `feature_frame`'de o hastaya ait satır olmasa bile (join onu zaten
    düşürecek olsa bile) SESSİZCE atlanmaz, join'e hiç ulaşılmadan
    LOUD bir şekilde reddedilir (CLAUDE.md ilkesi: sessiz filtreleme
    yasak, "inner join zaten temizler" gibi örtük bir varsayıma
    GÜVENİLMEZ).

    ÖNEMLİ İSTİSNA -- bu fonksiyon HER durumda kullanılmamalı:
    `tools/smoke_test_cox_model.py` gibi BİLİNÇLİ olarak TCGA verisiyle
    SADECE kod mekaniğini (DB->frame->LASSO->CV->external-eval->SHAP)
    test eden, üretilen hiçbir modeli gerçek Cox modeli olarak
    KULLANMAYAN/kaydetmeyen script'ler bu fonksiyonu KULLANMAMALI --
    bu fonksiyon TCGA'yı her koşulda reddeder, böyle bir mekanik
    smoke-test'i de engeller. Bu durumlarda `_assemble_training_frame()`
    hâlâ doğrudan (guard'sız, bilinçli olarak) çağrılabilir, ama
    çağıran tarafın docstring'inde NEDEN guard'ın atlandığı AÇIKÇA
    belgelenmeli (örnek: `tools/smoke_test_cox_model.py`'nin kendi
    docstring'i).

    Parametreler ve dönüş değeri `_assemble_training_frame()` ile
    BİREBİR AYNI (bu fonksiyon ona bir ön-kontrol ekleyen ince bir
    sarmalayıcıdır) -- `allowed_sources` ayrıca `assert_training_pool_
    sources()`'a geçirilir (varsayılan `COX_TRAINING_ALLOWED_SOURCES`,
    zaten kanonik). `assert_training_pool_sources()` bu değeri
    `_validate_allowed_sources()` ile doğrular -- bu fonksiyona
    `COX_TRAINING_ALLOWED_SOURCES`'ı GENİŞLETEN bir `allowed_sources`
    (örn. TCGA eklenmiş) geçirmek de artık `CoxTrainingLeakageError`
    ile reddedilir (parametrenin amacı sadece DARALTMAK, GENİŞLETMEK
    değil).

    C5 (2026-08-14, Nisa -- bkz. decisions/2026-08-13-combat-ozdeslik-
    bulgusu-ve-kapsami.md "ZORUNLU KORUMA"): whitelist guard'ından SONRA
    (TCGA zaten reddedildikten sonra), `check_combat_identity=True`
    (varsayılan) iken `assert_combat_identity_for_training_pool()` de
    çalışır -- eğitim havuzundaki TÜM kaynakların `source_reference`
    (varsayılan `"UPenn"`, ComBat'ın `reference_batch`'iyle TUTARLI
    tutulması gereken AYNI parametre, bkz. `_assemble_training_frame()`
    docstring'indeki `source_reference` notu) ile aynı olduğunu assert
    eder. LUMIERE (whitelist'te İZİNLİ ama şu an olay verisi
    sağlayamadığı için havuzda fiilen YOK, bkz. decisions/2026-08-06-
    lumiere-event-vital-status-eksik.md) ileride gerçekten havuza
    girerse, bu guard `ComBatIdentityAssumptionError` ile patlar --
    whitelist guard'ı bunu YAKALAMAZ (LUMIERE zaten whitelist'te), bu
    yüzden AYRI bir guard gerekiyordu. `check_combat_identity=False`
    SADECE ComBat'a hiç bağımlı olmayan/onu bilinçli bypass eden
    (örn. `tools/smoke_test_cox_model.py` tarzı mekanik test)
    senaryolar için var -- gerçek Hafta 3 eğitim script'i bunu
    ASLA `False` yapmamalı.
    """

    assert_training_pool_sources(
        patients_frame, source_col=source_col, allowed_sources=allowed_sources
    )
    if check_combat_identity:
        assert_combat_identity_for_training_pool(
            patients_frame, source_col=source_col, reference_batch=source_reference
        )
    return _assemble_training_frame(
        feature_frame,
        patients_frame,
        source_col=source_col,
        duration_col=duration_col,
        event_col=event_col,
        vital_status_col=vital_status_col,
        deceased_labels=deceased_labels,
        censored_labels=censored_labels,
        source_reference=source_reference,
        passthrough_columns=passthrough_columns,
    )


def _assemble_training_frame(
    feature_frame: pd.DataFrame,
    patients_frame: pd.DataFrame,
    *,
    source_col: str = "source",
    duration_col: str = "survival_days",
    event_col: str = "event",
    vital_status_col: str = "vital_status",
    deceased_labels: Iterable[str] = ("DECEASED",),
    # "CENSORED" 2026-08-14'te EKLENDİ (koordinatör): canlı DB'de UPenn'in
    # `vital_status` alanı ÜÇ değer taşıyor -- DECEASED 585 / ALIVE 17 /
    # CENSORED 9. Önceki varsayılan yalnız ("ALIVE",) olduğu için 9 CENSORED
    # hasta "tanınmayan statü" sayılıp SESSİZCE düşürülüyordu ve eğitim havuzu
    # 611 yerine 602 oluyordu (gerçek koşuda 611/585 gate'i bunu yakaladı).
    # Sağkalım analizinde sansürlenmiş hasta ATILMAZ: olay=0 olarak modele
    # girer ve takip süresi boyunca "olay gözlenmedi" bilgisini taşır.
    # ALIVE ve CENSORED'ın ikisi de "takip sonuna kadar olay yok" demektir.
    censored_labels: Iterable[str] = ("ALIVE", "CENSORED"),
    source_reference: str = DEFAULT_SOURCE_REFERENCE,
    passthrough_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, TrainingFrameReport]:
    """Radyomik özellik matrisi + hasta klinik verisini Cox eğitim çerçevesine dönüştür.

    PRIVATE (2026-08-13, Codex HIGH-1 bulgusu kapatıldı -- bkz.
    decisions/2026-08-12-cox-model-source-guard-canonicalization.md
    "[2026-08-13] TAKİP-3"): bu fonksiyon eskiden (`assemble_training_
    frame()` adıyla) public'ti; whitelist/TCGA-reddi UYGULAMADIĞI
    (aşağıdaki "Not" bölümüne bkz.) hâlde hiçbir erişim kısıtlaması
    yoktu -- bir çağıran `build_training_frame()`'i unutup bunu
    doğrudan çağırırsa aynı sızıntı sınıfı (TCGA'nın `source_TCGA`
    dummy kolonuyla sessizce eğitim havuzuna girmesi) geri gelirdi. Alt
    çizgi ön-ekiyle private yapıldı -- KAZARA `from pipeline.cox_model
    import assemble_training_frame` artık `ImportError` verir (eski isim
    modülde YOK). Bu isimlendirme Python'da GERÇEK bir erişim kontrolü
    DEĞİL (istenirse `_assemble_training_frame` adıyla hâlâ import
    edilebilir) -- sadece "bunu doğrudan çağırma" sözleşmesini kod
    seviyesinde daha görünür/zor-gözden-kaçırılır kılar.

    Not (2026-08-13): bu fonksiyon whitelist/TCGA-reddi UYGULAMAZ
    (kasıtlı kapsam sınırı, aşağıdaki "Not" bölümüne bkz.). Hafta 3
    gerçek eğitim script'i DAHİL, bu modülün DIŞINDAN gelen çağıranlar
    bu fonksiyonu doğrudan çağırmak yerine `build_training_frame()`'i
    (guard + assemble'ı doğru sırada zorunlu birlikte çağıran tek
    resmi giriş noktası) KULLANMALI -- bkz. decisions/2026-08-12-cox-
    model-source-guard-canonicalization.md "[2026-08-13] TAKİP-2"
    bölümü. Bu fonksiyonu guard'sız doğrudan çağırmanın TEK meşru
    kullanım örneği bilinçli mekanik smoke-test'tir (bkz.
    `tools/smoke_test_cox_model.py`'nin kendi docstring'i) -- gerçek
    model üretimi/kaydı İÇİN DEĞİL.

    Parametreler
    ------------
    feature_frame : pandas.DataFrame
        Index = patient_id, kolonlar = (ComBat-harmonize edilmiş)
        özellik adları. Bu fonksiyon harmonizasyon YAPMAZ -- girdinin
        zaten uygun şekilde hazırlanmış olduğunu VARSAYAR.
    patients_frame : pandas.DataFrame
        Index = patient_id, en az `source_col`, `duration_col`,
        `vital_status_col` kolonlarını içermeli.
    deceased_labels / censored_labels : Iterable[str]
        `vital_status_col`'un hangi ham değerlerinin event=1 (ölüm)
        / event=0 (sansürlü) sayılacağı -- AÇIKÇA verilmeli, örtük
        varsayım YAPILMAZ. Bu listelerde OLMAYAN bir değer (örn.
        UPenn'in "Lost to Follow-up"/"Deceased - uncertain date of
        death" ham metinleri, bkz.
        decisions/2026-07-19-tcga-mgmt-idh1-survival-days-acik.md)
        o satırı SESSİZCE event=0 YAPMAZ -- satır DROP edilir ve
        `TrainingFrameReport.dropped_unrecognized_vital_status`'a
        sayılır (çağıran taraf bunu görüp bilinçli bir eşleme
        kararı vermeli).
    source_reference : str
        `source_col`'dan üretilen dummy değişkenlerde referans
        (drop edilen) seviye -- varsayılan "UPenn", ComBat'ın
        referans-batch seçimiyle TUTARLI tutulur (mimari kural,
        bkz. pipeline.harmonization.fit_combat_harmonization).
        Varsayılan `"UPenn"` zaten kanoniktir; ham bir alias
        (`"UPenn-GBM"` gibi) verilirse de fonksiyon bunu otomatik
        kanonikleştirir (aşağıya bkz.) -- `pd.get_dummies()`'in ürettiği
        kanonik `source_UPenn` kolonuyla eşleşmesi için.

    Not (2026-08-13, `assert_training_pool_sources()`'ın FIX-3/FIX-4
    deseniyle AYNI hata sınıfının önlenmesi -- bkz.
    decisions/2026-08-12-cox-model-source-guard-canonicalization.md
    "Çağıranlar kontrolü" bölümünde işaretlenen MEDIUM takip riski):
    `patients_frame[source_col]` DB'den HAM (`"UPenn-GBM"`, `"LUMIERE"`
    gibi alias) gelebilir. Bu fonksiyon `pd.get_dummies()`'e vermeden
    ÖNCE `merged[source_col]`'ü
    `pipeline.harmonization._canonical_source_series()` ile kanonikleştirir
    (`assert_training_pool_sources()`'ın kullandığı AYNI paylaşımlı
    yardımcı -- iki normalizasyon mantığı YOK). Bunun sonucu:
    - Üretilen dummy kolonlar her zaman kanonik adlarla
      (`source_UPenn`, `source_LUMIERE`, ...) gelir, `source_reference`
      (varsayılan `"UPenn"`, kanonik) ile TUTARLI eşleşir -- ham
      `"source_UPenn-GBM"` kolonunun `"source_UPenn"` referansıyla
      eşleşmeyip `ValueError` fırlatması riski YOK EDİLDİ.
    - `TrainingFrameReport.sources` de artık kanonik anahtarlarla
      raporlanır (ham `"UPenn-GBM"`/`"upenn-gbm"` gibi varyantlar ayrı
      ayrı sayılıp yanıltıcı bir dağılım GÖSTERMEZ).
    - `SOURCE_ALIASES`'in TANIMADIĞI bir kaynak adı (yazım hatası,
      bilinmeyen kohort) SESSİZCE geçilmez -- `_canonical_source_series()`
      `ValueError` fırlatır, bu fonksiyon onu YUTMAZ/başka bir değere
      DÖNÜŞTÜRMEZ, olduğu gibi çağırana yayılır (proje kuralı: sessiz
      fallback yasak).
    - Bu fonksiyon KENDİSİ whitelist/TCGA-reddi UYGULAMAZ (bu,
      `assert_training_pool_sources()`'ın işi, ayrıca ve ÖNCE
      çağrılmalı -- bu fonksiyonun docstring'i zaten bunu ima ediyordu,
      DEĞİŞMEDİ). Yani TCGA gibi kanonikleştirilebilen ama yasak bir
      kaynak buraya sızarsa (örn. `assert_training_pool_sources()`
      atlanırsa) bu fonksiyon onu SESSİZCE kabul edip `source_TCGA`
      dummy kolonu üretir -- bu KASITLI bir kapsam sınırıdır, whitelist
      sorumluluğu bu fonksiyona GENİŞLETİLMEDİ (görev talimatı: "Task
      #18'in source_LUMIERE=0 kararı gibi başka mantığa dokunma").

    Döndürür
    --------
    (frame, report)
        ``frame``: Cox'a hazır DataFrame (özellik kolonları + source
        dummy kolonları + `duration_col` + `event_col`).
        ``report``: kaç satırın hangi gerekçeyle düşürüldüğünün
        şeffaf dökümü -- CLAUDE.md'nin "sessizce filtrelenmez" ilkesi.
    """

    required_patient_cols = {source_col, duration_col, vital_status_col}
    missing_cols = required_patient_cols - set(patients_frame.columns)
    if missing_cols:
        raise ValueError(f"patients_frame eksik kolon(lar): {sorted(missing_cols)}")

    merged = feature_frame.join(patients_frame, how="inner")
    n_input_rows = len(merged)

    # 2026-08-13 FIX (bkz. docstring'in "Not" bölümü + decisions/2026-08-12-
    # cox-model-source-guard-canonicalization.md "Çağıranlar kontrolü"):
    # `source_col` DB'den ham (`"UPenn-GBM"` gibi) gelebilir -- bunu
    # `pd.get_dummies()`'e vermeden ÖNCE kanonikleştiriyoruz, aksi halde
    # üretilen "source_UPenn-GBM" dummy kolonu aşağıdaki
    # `reference_dummy_col = f"source_{source_reference}"` (varsayılan
    # "source_UPenn") ile EŞLEŞMEZ ve ValueError fırlatılır.
    # `_canonical_source_series()` `assert_training_pool_sources()` ile
    # PAYLAŞIMLI aynı yardımcı (kod tekrarı yok) -- tanınmayan bir kaynak
    # adı için SESSİZCE geçmez, ValueError fırlatır (proje kuralı: sessiz
    # fallback yasak). Bu fonksiyon whitelist/TCGA-reddi UYGULAMAZ (o,
    # ayrı ve önce çağrılması gereken `assert_training_pool_sources()`'ın
    # işi) -- burada yalnız TUTARLI kanonik isimlendirme sağlanır.
    merged[source_col] = _canonical_source_series(merged[source_col])

    deceased_set = set(deceased_labels)
    censored_set = set(censored_labels)
    recognized = deceased_set | censored_set

    unrecognized_mask = ~merged[vital_status_col].isin(recognized)
    unrecognized_counts = (
        merged.loc[unrecognized_mask, vital_status_col]
        .fillna("<NULL>")
        .value_counts()
        .to_dict()
    )
    dropped_unrecognized = int(unrecognized_mask.sum())
    # 2026-08-19: SAYININ yanina KIMLIK de kaydedilir (bkz.
    # `TrainingFrameReport.dropped_patient_ids` alan yorumu).
    dropped_patient_ids: dict[str, list[str]] = {}
    if dropped_unrecognized:
        dropped_patient_ids["unrecognized_vital_status"] = sorted(
            str(pid) for pid in merged.index[unrecognized_mask]
        )
    merged = merged.loc[~unrecognized_mask].copy()

    merged[event_col] = merged[vital_status_col].isin(deceased_set).astype(int)

    missing_duration_mask = merged[duration_col].isna()
    dropped_missing_duration = int(missing_duration_mask.sum())
    if dropped_missing_duration:
        dropped_patient_ids["missing_duration"] = sorted(
            str(pid) for pid in merged.index[missing_duration_mask]
        )
    merged = merged.loc[~missing_duration_mask].copy()

    feature_columns = list(feature_frame.columns)
    missing_feature_mask = merged[feature_columns].isna().any(axis=1)
    dropped_missing_features = int(missing_feature_mask.sum())
    if dropped_missing_features:
        dropped_patient_ids["missing_features"] = sorted(
            str(pid) for pid in merged.index[missing_feature_mask]
        )
    merged = merged.loc[~missing_feature_mask].copy()

    # 2026-08-15 EKLENDİ (klinik kovaryat desteği): `passthrough_columns`
    # radyomik `feature_columns` DEĞİL ama AYNI "eksikse sessizce
    # doldurma/atma yerine DÜŞÜR ve SAY" disipliniyle ele alınır. Çağıran
    # taraf (bkz. `tools/train_cox_week3.py`) zaten GTR gibi bilinçli
    # eksik-gösterge kolonları (`clinical_gtr_missing`) üretip NaN'ı
    # ORTADAN KALDIRDIĞI için bu adım pratikte GTR'de hiç satır
    # düşürmez -- ama yaş/cinsiyet gibi (TCGA'da ~%3 eksik olabilen)
    # kolonlarda gerçek bir güvenlik ağı sağlar.
    passthrough_columns = list(passthrough_columns or [])
    missing_passthrough_columns = set(passthrough_columns) - set(merged.columns)
    if missing_passthrough_columns:
        raise ValueError(
            f"patients_frame passthrough_columns icin eksik kolon(lar): "
            f"{sorted(missing_passthrough_columns)}"
        )
    if passthrough_columns:
        missing_passthrough_mask = merged[passthrough_columns].isna().any(axis=1)
        dropped_missing_passthrough = int(missing_passthrough_mask.sum())
        if dropped_missing_passthrough:
            dropped_patient_ids["missing_passthrough"] = sorted(
                str(pid) for pid in merged.index[missing_passthrough_mask]
            )
        merged = merged.loc[~missing_passthrough_mask].copy()
    else:
        dropped_missing_passthrough = 0

    source_dummies = pd.get_dummies(
        merged[source_col], prefix="source", drop_first=False
    )
    # 2026-08-13: `source_reference` de kanonikleştirilir -- `merged[source_col]`
    # artık yukarıda kanonik hale getirildiğinden (dummy kolonlar
    # `source_UPenn`/`source_LUMIERE` gibi kanonik adlarla üretiliyor),
    # çağıran taraf `source_reference="UPenn-GBM"` gibi ham bir alias
    # geçerse `f"source_{source_reference}"` hiçbir zaman eşleşmezdi --
    # `pipeline.harmonization.fit_combat_harmonization()`'ın FIX-5'inde
    # (`reference_batch` kanonikleştirmesi) düzeltilen AYNI hata sınıfı,
    # burada da önlenir.
    canonical_source_reference = canonical_source(source_reference)
    reference_dummy_col = f"source_{canonical_source_reference}"
    if reference_dummy_col not in source_dummies.columns:
        raise ValueError(
            f"source_reference={source_reference!r} (kanonik: "
            f"{canonical_source_reference!r}) merged['{source_col}'] "
            f"içinde bulunamadı: {sorted(merged[source_col].unique())}"
        )
    source_dummies = source_dummies.drop(columns=[reference_dummy_col])

    frame_parts = [merged[feature_columns]]
    if passthrough_columns:
        frame_parts.append(merged[passthrough_columns])
    frame_parts.append(source_dummies)
    frame_parts.append(merged[[duration_col, event_col]])
    frame = pd.concat(frame_parts, axis=1)

    report = TrainingFrameReport(
        n_input_rows=n_input_rows,
        n_output_rows=len(frame),
        dropped_missing_event=0,  # event _assemble_training_frame içinde her zaman türetilir
        dropped_missing_duration=dropped_missing_duration,
        dropped_missing_features=dropped_missing_features,
        dropped_unrecognized_vital_status=unrecognized_counts,
        sources=merged[source_col].value_counts().to_dict(),
        dropped_missing_passthrough=dropped_missing_passthrough,
        dropped_patient_ids=dropped_patient_ids,
    )
    return frame, report


def select_features_lasso(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    duration_col: str = "survival_days",
    event_col: str = "event",
    extra_columns: list[str] | None = None,
    penalizer: float = 0.1,
    l1_ratio: float = 1.0,
    coefficient_tolerance: float = 1e-4,
    extra_column_penalizer: float | None = None,
) -> tuple[list[str], Any, pd.Series]:
    """lifelines `CoxPHFitter` + L1 (LASSO) penaltı ile özellik seçimi.

    `extra_column_penalizer` (2026-08-15 EKLENDİ, klinik kovaryat
    desteği): `None` (varsayılan) iken davranış DEĞİŞMEZ. Verilirse
    `extra_columns` `penalizer` YERİNE bu (genelde daha zayıf/sıfır)
    ceza ile fit edilir -- bkz. `_build_penalizer_argument()` docstring'i.

    Not (deneyle doğrulandı, bkz. tests/test_cox_model.py): lifelines'ın
    L1 penaltısı gerçek coordinate-descent LASSO (örn. glmnet) gibi
    katsayıları TAM SIFIRA indirmiyor -- gürültü özellikleri genelde
    1e-6..1e-8 mertebesinde kalıyor, sinyal özellikleri 1e-2..1'de.
    Bu yüzden `coefficient_tolerance` varsayılanı 1e-4 (1e-8 DEĞİL) --
    çok düşük bir tolerans neredeyse hiçbir özelliği elemez.
    `penalizer` çok küçükse (örn. 0.05) LASSO pratikte hiçbir şeyi
    seçip elemez, çok büyükse (örn. >0.7, veri/sinyal büyüklüğüne
    bağlı) gerçek sinyali de siler -- gerçek 107-özellik verisiyle
    çalışırken bu ikisi CV ile (ayrı bir grid-search sarmalayıcısıyla,
    bu iskelet kapsamında YOK) taranmalı, sabit varsayılan DEĞERLERE
    körü körüne güvenilmemeli.

    `l1_ratio=1.0` (varsayılan) saf LASSO'dur (lifelines'ın Elastic
    Net parametrizasyonunda). `extra_columns` (örn. source dummy'leri)
    LASSO'nun KENDİSİNE dahil edilir ama seçim çıktısına (dönen
    `selected_features` listesine) DAHİL EDİLMEZ -- kurumsal/batch
    kovaryatı "seçilecek bir radyomik özellik" değil, her zaman
    modelde tutulması istenen bir düzeltme terimi.

    Not: lifelines `penalizer`/`l1_ratio`'yu CV ile taramak (gerçek
    hiperparametre optimizasyonu) bu fonksiyonun kapsamında DEĞİL --
    bu fonksiyon TEK BİR sabit (penalizer, l1_ratio) için fit eder.
    Hiperparametre taraması `run_stratified_cv()`'nin dışına, ayrı bir
    grid-search sarmalayıcısına bırakılmalı (henüz yazılmadı, iskelet
    kapsamı dışı).
    """

    from lifelines import CoxPHFitter

    extra_columns = extra_columns or []
    model_columns = feature_columns + extra_columns + [duration_col, event_col]
    missing = set(model_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"frame eksik kolon(lar): {sorted(missing)}")

    penalizer_argument = _build_penalizer_argument(
        feature_columns,
        extra_columns,
        penalizer=penalizer,
        extra_column_penalizer=extra_column_penalizer,
    )
    cox = CoxPHFitter(penalizer=penalizer_argument, l1_ratio=l1_ratio)
    cox.fit(frame[model_columns], duration_col=duration_col, event_col=event_col)

    coefficients = cox.params_
    selected = [
        name
        for name in feature_columns
        if abs(coefficients.get(name, 0.0)) > coefficient_tolerance
    ]
    return selected, cox, coefficients


def run_stratified_cv(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    duration_col: str = "survival_days",
    event_col: str = "event",
    extra_columns: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 42,
    penalizer: float = 0.1,
    l1_ratio: float = 1.0,
) -> pd.DataFrame:
    """5-fold (varsayılan) stratified CV -- event durumuna göre stratify.

    "Stratified" burada `event_col` (ölüm/sansür) oranının her fold'da
    korunması anlamına gelir -- survival analizinde standart pratik
    (event sayısı az olduğunda fold'lar arası dengesizliği önler).
    Plan.txt'nin "5-fold stratified CV YALNIZ UPenn+LUMIERE eğitim
    havuzu içinde" talimatına uymak ÇAĞIRANIN sorumluluğunda --
    bu fonksiyon `frame`'in içeriğini DENETLEMEZ (TCGA sızıntısını
    yakalamak için `frame` `build_training_frame()`'in çıktısı OLMALI
    -- bu, `assert_training_pool_sources()`'ı
    `_assemble_training_frame()`'den ÖNCE zorunlu çağıran tek resmi giriş noktası, bkz.
    decisions/2026-08-12-cox-model-source-guard-canonicalization.md
    "[2026-08-13] TAKİP-2").

    Döndürür
    --------
    pandas.DataFrame
        Her satır bir fold: `fold`, `n_train`, `n_test`,
        `n_events_train`, `n_events_test`, `c_index`.
    """

    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from sklearn.model_selection import StratifiedKFold

    extra_columns = extra_columns or []
    model_columns = feature_columns + extra_columns

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []
    for fold_index, (train_idx, test_idx) in enumerate(
        splitter.split(frame, frame[event_col])
    ):
        train_frame = frame.iloc[train_idx]
        test_frame = frame.iloc[test_idx]

        cox = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        cox.fit(
            train_frame[model_columns + [duration_col, event_col]],
            duration_col=duration_col,
            event_col=event_col,
        )

        test_partial_hazard = cox.predict_partial_hazard(test_frame[model_columns])
        c_index = concordance_index(
            test_frame[duration_col],
            -test_partial_hazard,
            test_frame[event_col],
        )

        results.append(
            {
                "fold": fold_index,
                "n_train": len(train_frame),
                "n_test": len(test_frame),
                "n_events_train": int(train_frame[event_col].sum()),
                "n_events_test": int(test_frame[event_col].sum()),
                "c_index": c_index,
            }
        )

    return pd.DataFrame(results)


# =====================================================================
# C2 + C3 -- Nested-CV sarmalayıcısı + elastic-net/stabilite grid
# taraması (2026-08-14, Nisa)
# =====================================================================
#
# Bağlam: decisions/2026-08-13-bolge-stratejisi-ve-modelleme-protokolu.md
# Bölüm 4.2-4.3. `run_stratified_cv()` (yukarıda) özellik seçimini
# çağırana bırakır (tipik kullanım: `select_features_lasso()` tüm
# havuzda BİR KEZ çalıştırılıp sabit bir `feature_columns` listesi
# geçirilir) -- bu KLASİK SEÇİM SIZINTISIDIR (Ambroise & McLachlan
# 2002), EPV=5,6'da kozmetik değil. Aşağıdaki `run_nested_cv()` özellik
# seçimini (elastic-net hiperparametre taraması + bootstrap stabilite
# seçimi dahil) HER DIŞ FOLD'UN İÇİNDE tekrarlar.

DEFAULT_L1_RATIO_GRID: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0)
DEFAULT_PENALIZER_GRID: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0)
DEFAULT_STABILITY_FREQUENCY_THRESHOLD: float = 0.6


@dataclass
class NestedCVFoldResult:
    """`run_nested_cv()`'nin bir dış fold'unun tam denetim izi.

    `selected_features` her fold için AYRI raporlanır (görev talimatı:
    "fold başına seçilen özellik listesi raporlanmalı, stabilite
    gözlemi için") -- fold'lar arası özellik seçimi ne kadar
    değişkense (EPV düşükken beklenir), stabilite o kadar zayıftır;
    bu saklanmaz, dışarı verilir.
    """

    fold: int
    n_train: int
    n_test: int
    n_events_train: int
    n_events_test: int
    best_penalizer: float
    best_l1_ratio: float
    inner_mean_c_index: float
    n_selected_features_elastic_net: int
    n_selected_features_final: int
    selected_features: list[str]
    c_index: float


def standardize_columns_fold_safe(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame | None,
    columns: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Z-score standardizasyonu -- istatistik SADECE `train_frame`'den
    fit edilir, `test_frame`'e (verilmişse) AYNI ortalama/std uygulanır.

    2026-08-15 EKLENDİ (klinik kovaryat desteği, görev talimatı: "Eksik
    veri her DIŞ FOLD İÇİNDE ele alınır ... Standardizasyon (scaler) da
    fold içinde fit edilmeli"). Amaç yaş gibi sürekli klinik kovaryatları
    (radyomik özelliklerin C32 Z-score harmonizasyonuyla aynı büyüklük
    mertebesine getirmek + sayısal kararlılık) `test_frame`'e SIZDIRMADAN
    ölçeklemek.

    NOT (istatistiksel dürüstlük -- bkz. bu fonksiyonu çağıran
    `run_nested_cv()`'nin docstring notu): `tools/train_cox_week3.py`
    klinik kovaryatları `extra_columns` ile (varsayılan
    `extra_column_penalizer=0.0`, yani PENALİZE EDİLMEZ) modele sokuyor.
    Penalize EDİLMEYEN bir kovaryat için Cox'un partial-likelihood'u ve
    üretilen C-index (concordance sıralaması) o kovaryatın DOĞRUSAL
    (afin) yeniden ölçeklenmesine karşı DEĞİŞMEZDİR -- yani bu fonksiyon
    C-index sonucunu MATEMATİKSEL OLARAK DEĞİŞTİRMEZ (kanıt: sabit bir
    afin dönüşüm sadece exp(linear_predictor)'ı TÜM gözlemlerde AYNI
    çarpanla ölçekler, bu da concordance sıralamasını korur). Bu
    fonksiyon YİNE DE görev talimatının açık isteği + sayısal kararlılık
    + katsayı yorumlanabilirliği için UYGULANIYOR -- "gereksiz" değil,
    ama "C-index'i değiştiren kritik bir düzeltme" de DEĞİL; bu ayrım
    raporlarda AÇIKÇA belirtilmeli (penalize edilen bir klinik kovaryat
    senaryosunda bu artık DOĞRU OLMAZDI).

    `columns` boş/`None` ise girdiler DEĞİŞTİRİLMEDEN döner (no-op).
    Std=0 olan bir kolon (sabit değer, örn. tek-sınıflı bir dış-fold alt
    kümesi) 1.0'a yuvarlanır -- sıfıra bölme SESSİZCE NaN üretmez.
    """

    if not columns:
        return train_frame, test_frame

    train_frame = train_frame.copy()
    means = train_frame[columns].mean()
    stds = train_frame[columns].std(ddof=0)
    stds = stds.where(stds > 0, 1.0)

    train_frame[columns] = (train_frame[columns] - means) / stds
    if test_frame is not None:
        test_frame = test_frame.copy()
        test_frame[columns] = (test_frame[columns] - means) / stds
    return train_frame, test_frame


def _build_penalizer_argument(
    feature_columns: list[str],
    extra_columns: list[str],
    *,
    penalizer: float,
    extra_column_penalizer: float | None,
) -> float | np.ndarray:
    """`penalizer` skaler mi yoksa özellik-başına dizi mi olacağına karar
    verir.

    2026-08-15 EKLENDİ (klinik kovaryat desteği, Codex önerisi -- görev
    talimatı: "klinik blok zayıf-penalize edilir, radyomik blok normal
    penalize"). `extra_column_penalizer=None` (varsayılan) iken davranış
    ESKİSİYLE BİREBİR AYNI (tek skaler `penalizer`, `CoxPHFitter` içinde
    TÜM kovaryatlara -- feature_columns + extra_columns -- eşit uygulanır)
    -- mevcut hiçbir çağıran/test ETKİLENMEZ.

    `extra_column_penalizer` verilirse (örn. `0.0` -- standart pratik:
    glmnet'in `penalty.factor=0` ile "zorunlu/forced-in" kovaryatları
    cezalandırmama yaklaşımıyla AYNI), `feature_columns` `penalizer`
    ile, `extra_columns` `extra_column_penalizer` ile cezalandırılan bir
    `numpy.ndarray` döner. SIRA KRİTİK: lifelines'ın dizi-penalizer'ı
    `CoxPHFitter.fit()`'e verilen DataFrame'in kolon SIRASIYLA hizalanır
    (ampirik doğrulandı, bkz. bu değişikliğin test dosyasındaki
    `test_build_penalizer_argument_array_order_matches_lifelines_params_`
    ordering testi) -- bu yüzden dönen dizi DAİMA
    `[penalizer]*len(feature_columns) + [extra_column_penalizer]*len(extra_columns)`
    sırasında, `select_features_lasso()`'nun `model_columns = feature_columns
    + extra_columns + [...]` sırasıyla TUTARLI kurulur.
    """

    if extra_column_penalizer is None:
        return penalizer
    return np.array(
        [penalizer] * len(feature_columns) + [extra_column_penalizer] * len(extra_columns)
    )


def _score_hyperparameters_via_inner_cv(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    duration_col: str,
    event_col: str,
    extra_columns: list[str],
    penalizer: float,
    l1_ratio: float,
    inner_splits: int,
    seed: int,
    extra_column_penalizer: float | None = None,
) -> float:
    """Bir (penalizer, l1_ratio) çifti için İÇ CV ortalama C-index'i.

    Her iç fold'da `select_features_lasso()` İÇ-eğitim alt-kümesinde
    yeniden fit edilir (seçim burada da tekrarlanır -- dış fold'un
    hiperparametre seçimi de kendi içinde sızıntısız olmalı) ve
    doğrudan o fit edilmiş (regularize) `CoxPHFitter`'ın
    `predict_partial_hazard()`'ı ile iç-test üzerinde skorlanır --
    AYRICA bir unpenalized refit YAPILMAZ (gereksiz karmaşıklık,
    elastic-net'in kendisi zaten regularize edilmiş bir risk skoru
    üretiyor).

    Bir iç fold'da eğitim ya da test alt-kümesinde HİÇ event yoksa
    (küçük n'de olası) o fold C-index'e KATILMAZ (tanımsız, sessizce
    0/1'e yuvarlanmaz) -- `evaluate_external_test()`'in bootstrap'taki
    "event yoksa atla" deseniyle TUTARLI.

    Hiçbir fold geçerli skor üretmezse `float("nan")` döner -- çağıran
    (`run_nested_cv()`) NaN skorları grid karşılaştırmasında ELER.
    """

    from lifelines.exceptions import ConvergenceError
    from lifelines.utils import concordance_index
    from numpy.linalg import LinAlgError
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    scores: list[float] = []
    for inner_train_idx, inner_test_idx in splitter.split(train_frame, train_frame[event_col]):
        inner_train = train_frame.iloc[inner_train_idx]
        inner_test = train_frame.iloc[inner_test_idx]
        if inner_train[event_col].sum() == 0 or inner_test[event_col].sum() == 0:
            continue
        try:
            _, cox, _ = select_features_lasso(
                inner_train,
                feature_columns,
                duration_col=duration_col,
                event_col=event_col,
                extra_columns=extra_columns,
                penalizer=penalizer,
                l1_ratio=l1_ratio,
                extra_column_penalizer=extra_column_penalizer,
            )
        except (ConvergenceError, LinAlgError):
            # MEDIUM daraltma (Codex 3. tur, task-mt06mqzi-k0gb5j): ÖNCEDEN
            # `except Exception` -- lifelines kaynak incelemesi (2026-08-19)
            # gösterdi ki `CoxPHFitter.fit()` "infs or NaNs"/tekil matris gibi
            # SAYISAL yakınsama sorunlarını `lifelines.exceptions.
            # ConvergenceError`'a SARAR; bunun dışındaki `ValueError`'lar
            # (örn. `l1_ratio` aralık dışı, yanlış sütun kümesi) lifelines'ın
            # KENDİSİ TARAFINDAN OLDUĞU GİBİ yeniden fırlatılır -- yani bunlar
            # bir "kırılgan yakınsama" değil, KOD/GİRDİ HATASI işaretidir ve
            # artık SESSİZCE yutulmuyor. `LinAlgError` da ayrıca yakalanır
            # (bazı sürüm/yollarda lifelines'a ulaşmadan önce fırlayabilir).
            # ⚠️ atlanan fold sayısı/nedeni kaydı VE minimum geçerli-fold
            # eşiği (Codex'in istediği diğer iki alt-madde) bu turda BİLİNÇLİ
            # olarak YAPILMADI -- görev talimatı bu dosyada SADECE except
            # daraltmasına izin verdi, "mevcut çağıranların davranışını
            # değiştiren varsayılan EKLEME" kısıtı bu iki alt-maddeyi
            # (kayıt + fail-closed eşiği DAVRANIŞ değişikliği sayılır) kapsam
            # dışı bırakıyor. Bkz. görev raporu.
            continue
        model_columns = feature_columns + extra_columns
        partial_hazard = cox.predict_partial_hazard(inner_test[model_columns])
        c_index = concordance_index(
            inner_test[duration_col], -partial_hazard, inner_test[event_col]
        )
        scores.append(c_index)

    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _bootstrap_stability_selection(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    duration_col: str,
    event_col: str,
    extra_columns: list[str],
    penalizer: float,
    l1_ratio: float,
    n_bootstrap: int,
    frequency_threshold: float,
    seed: int,
    extra_column_penalizer: float | None = None,
) -> tuple[list[str], pd.Series]:
    """Bootstrap + seçim-sıklığı eşiği (decisions/.../Bölüm 4.2).

    `train_frame`'den B (`n_bootstrap`) kez yerine-koyarak (with
    replacement) örnekle, HER örnekte `select_features_lasso()`'yu
    (verilen sabit `penalizer`/`l1_ratio` ile -- bu ikisi zaten dış
    fold'un iç-CV'siyle seçildi, burada TEKRAR taranmaz) çalıştır, her
    özelliğin kaç örnekte seçildiğini say. `frequency_threshold`
    (varsayılan 0,6 -- karar dosyası bu değeri KİLİTLEMEDİ, elle
    seçilmiş makul bir varsayılan; ampirik olarak DOĞRULANMASI GEREKİR,
    bkz. bu fonksiyonun çağrıldığı yerdeki "Barış'a soru" notu)
    ÜZERİNDEKİ özellikler "stabil" kabul edilir.

    Event'siz bir bootstrap örneği (küçük n'de olası) o örneği
    SESSİZCE atlar (`evaluate_external_test()`'in bootstrap'ıyla AYNI
    desen). TÜM örnekler event'siz kalırsa (aşırı küçük n/olay oranı)
    `RuntimeError` fırlatılır -- sessizce boş bir stabilite kümesi
    DÖNÜLMEZ.
    """

    from lifelines.exceptions import ConvergenceError
    from numpy.linalg import LinAlgError

    rng = np.random.default_rng(seed)
    n = len(train_frame)
    selection_counts = pd.Series(0, index=feature_columns, dtype=int)
    n_valid_bootstrap = 0

    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        sample = train_frame.iloc[sample_idx]
        if sample[event_col].sum() == 0:
            continue
        try:
            selected, _, _ = select_features_lasso(
                sample,
                feature_columns,
                duration_col=duration_col,
                event_col=event_col,
                extra_columns=extra_columns,
                penalizer=penalizer,
                l1_ratio=l1_ratio,
                extra_column_penalizer=extra_column_penalizer,
            )
        except (ConvergenceError, LinAlgError):
            # MEDIUM daraltma -- yukarıdaki `_score_hyperparameters_via_
            # inner_cv()`'deki AYNI gerekçe/kısıt (kayıt + fail-closed eşiği
            # bu turda BİLİNÇLİ olarak YAPILMADI, sadece except daraltıldı).
            continue
        if selected:
            selection_counts.loc[selected] += 1
        n_valid_bootstrap += 1

    if n_valid_bootstrap == 0:
        raise RuntimeError(
            "Bootstrap stabilite seçiminde hiçbir örnekleme geçerli "
            "(event içeren + lifelines'ın fit edebildiği) sonuç "
            "üretmedi -- n_bootstrap artırılmalı ya da event oranı/n "
            "kontrol edilmeli. Sessizce boş bir stabilite kümesi "
            "DÖNDÜRÜLMEDİ."
        )

    frequency = selection_counts / n_valid_bootstrap
    stable_features = sorted(frequency[frequency >= frequency_threshold].index.tolist())
    return stable_features, frequency


def run_nested_cv(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    duration_col: str = "survival_days",
    event_col: str = "event",
    extra_columns: list[str] | None = None,
    outer_splits: int = 5,
    inner_splits: int = 5,
    seed: int = 42,
    l1_ratio_grid: tuple[float, ...] = DEFAULT_L1_RATIO_GRID,
    penalizer_grid: tuple[float, ...] = DEFAULT_PENALIZER_GRID,
    stability_selection: bool = True,
    n_bootstrap_stability: int = 200,
    stability_frequency_threshold: float = DEFAULT_STABILITY_FREQUENCY_THRESHOLD,
    extra_column_penalizer: float | None = None,
    clinical_standardize_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Nested-CV: özellik seçimi (elastic-net grid + stabilite) HER DIŞ
    FOLD'UN İÇİNDE tekrarlanır -- `run_stratified_cv()`'nin (yukarıda)
    aksine, tüm havuzda BİR KEZ seçilmiş sabit bir `feature_columns`
    ALMAZ; `feature_columns` burada ADAY havuzu (örn.
    `STABLE_FEATURES_ICC60`, 93 özellik), seçim SONUCU DEĞİL.

    Her dış fold'da:
      1. İÇ CV (`inner_splits`, `event_col`'a göre stratified) ile
         `l1_ratio_grid` x `penalizer_grid` TARANIR (C3) -- her
         kombinasyon `_score_hyperparameters_via_inner_cv()` ile
         puanlanır (iç-test ortalama C-index), NaN puanlar (hiç geçerli
         iç fold yoksa) ELENİR. En yüksek puanlı kombinasyon
         `best_penalizer`/`best_l1_ratio` olarak seçilir.
      2. Bu en iyi hiperparametrelerle, dış-eğitim'in TAMAMINDA
         `select_features_lasso()` çağrılır (elastic-net seçimi).
      3. `stability_selection=True` (varsayılan) ise, AYNI
         hiperparametrelerle dış-eğitim üzerinde bootstrap stabilite
         seçimi (`_bootstrap_stability_selection()`) çalışır; final
         özellik kümesi = stabilite eşiğini geçenler (BOŞ çıkarsa
         elastic-net'in seçtiği kümeye geri düşülür -- decisions/...
         Bölüm 4.2: "ikisi birlikte, biri diğerinin yerini TUTMAZ" ama
         stabilite HİÇBİR özelliği geçiremezse tamamen boş bir modelle
         devam edilemez).
      4. Final özellik kümesiyle (+ `extra_columns`) dış-eğitimin
         TAMAMINDA yeni bir `CoxPHFitter` (aynı best hiperparametrelerle)
         fit edilir, dış-test'te C-index hesaplanır.

    Döndürür
    --------
    pandas.DataFrame
        Her satır bir dış fold -- `NestedCVFoldResult` alanları
        (fold, n_train, n_test, n_events_train, n_events_test,
        best_penalizer, best_l1_ratio, inner_mean_c_index,
        n_selected_features_elastic_net, n_selected_features_final,
        selected_features, c_index).

    AÇIK TASARIM SORUSU (Barış'a soru, bkz. rapor): `stability_
    frequency_threshold=0,6` varsayılanı decisions/2026-08-13-bolge-
    stratejisi-ve-modelleme-protokolu.md'de SAYISAL olarak KİLİTLENMEDİ
    (yalnız "bootstrap + seçim sıklığı eşiği" diye tarif edildi) -- bu
    fonksiyonun yazarı (bugünkü ajan) makul bir literatür-standardı
    (Meinshausen & Bühlmann 2010 stabilite seçiminde tipik aralık
    0,6-0,9) seçti, ama GERÇEK 93-özellik/585-olay veriyle ampirik
    olarak DOĞRULANMADI (bu görev kapsamında sentetik veriyle test
    edildi, gerçek veri henüz DB'de yok). Gerçek eğitim öncesi bu eşik
    duyarlılık analiziyle (örn. 0,5/0,6/0,7 karşılaştırması) teyit
    edilmeli, körü körüne varsayılana güvenilmemeli.
    🔧 2026-09-14 (kalem 4) SAYI DÜZELTMESİ — YALNIZ METİN, davranış
    DEĞİŞMEDİ: ~~"603-olay"~~ -> **585-olay**. 630/603 UPenn'in *kayıtlı*
    sayısıdır; FİİLİ eğitim havuzu **611 hasta / 585 ölüm olayı**'dır
    (19 hastada yalnız post-op T1ce var, hazır maske yok -> radyomik
    satırı üretilemiyor). Bkz. CLAUDE.md kilitli değerler +
    `decisions/2026-08-13-bolge-stratejisi-ve-modelleme-protokolu.md`
    "SAYI DÜZELTMESİ (2026-08-14)". Bu dosyada "603" başka HİÇBİR yerde
    geçmiyordu (ölçüldü: tek eşleşme bu satırdı) -- yani sabit/eşik
    DEĞİL, salt anlatım metniydi.
    ⚠️ Yukarıdaki *"gerçek veri henüz DB'de yok"* ifadesi de bugün
    BAYATTIR (2026-08-13'ten beri C32 verisi DB'de); eşik duyarlılığının
    fiilen ne zaman/hangi konfigürasyonda koşulduğu
    `ZORUNLU-BEYANLAR.md` A12'de kayıtlıdır. Metin tarihsel kayıt olarak
    KORUNDU (wiki hard rule #3).

    `extra_column_penalizer` (2026-08-15 EKLENDİ, klinik kovaryat
    desteği): `None` (varsayılan) -- eski davranış BİREBİR korunur.
    Verilirse `extra_columns`'a (örn. klinik kovaryatlar) `feature_
    columns`'DAN FARKLI (genelde zayıf/sıfır) bir ceza uygulanır --
    bkz. `_build_penalizer_argument()`. Hem iç-CV grid taramasında hem
    dış-fold final fit'inde kullanılır.

    `clinical_standardize_columns` (2026-08-15 EKLENDİ, klinik kovaryat
    desteği -- görev talimatı: "Standardizasyon (scaler) da fold içinde
    fit edilmeli"): `None`/boş -- eski davranış BİREBİR korunur.
    Verilirse bu kolonlar (örn. `["clinical_age"]`) `standardize_
    columns_fold_safe()` ile HER DIŞ FOLD'UN train alt-kümesinden fit
    edilip AYNI dönüşüm test alt-kümesine uygulanır (sızıntı yok) --
    bu tek seviyede yapılır (iç-CV/bootstrap kendi İÇLERİNDE AYRICA
    yeniden standardize ETMEZ); bkz. `standardize_columns_fold_safe()`
    docstring'indeki matematiksel gerekçe (penalize edilmeyen bir
    kovaryat için bu C-index'i DEĞİŞTİRMEZ, ama görev talimatının açık
    isteği + sayısal kararlılık için uygulanıyor).
    """

    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from sklearn.model_selection import StratifiedKFold

    extra_columns = extra_columns or []
    clinical_standardize_columns = clinical_standardize_columns or []

    outer_splitter = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    fold_results: list[NestedCVFoldResult] = []

    for fold_index, (train_idx, test_idx) in enumerate(
        outer_splitter.split(frame, frame[event_col])
    ):
        train_frame = frame.iloc[train_idx]
        test_frame = frame.iloc[test_idx]
        train_frame, test_frame = standardize_columns_fold_safe(
            train_frame, test_frame, clinical_standardize_columns
        )

        best_score = float("-inf")
        best_penalizer = penalizer_grid[0]
        best_l1_ratio = l1_ratio_grid[0]
        any_valid_score = False
        for penalizer in penalizer_grid:
            for l1_ratio in l1_ratio_grid:
                score = _score_hyperparameters_via_inner_cv(
                    train_frame,
                    feature_columns,
                    duration_col=duration_col,
                    event_col=event_col,
                    extra_columns=extra_columns,
                    penalizer=penalizer,
                    l1_ratio=l1_ratio,
                    inner_splits=inner_splits,
                    seed=seed,
                    extra_column_penalizer=extra_column_penalizer,
                )
                if not np.isnan(score) and score > best_score:
                    best_score = score
                    best_penalizer = penalizer
                    best_l1_ratio = l1_ratio
                    any_valid_score = True

        if not any_valid_score:
            raise RuntimeError(
                f"Dış fold {fold_index}: iç CV grid taramasındaki HİÇBİR "
                "(penalizer, l1_ratio) kombinasyonu geçerli bir skor "
                "üretmedi -- grid/inner_splits/n gözden geçirilmeli. "
                "Sessizce ilk grid değerine düşülmedi."
            )

        en_selected, _, _ = select_features_lasso(
            train_frame,
            feature_columns,
            duration_col=duration_col,
            event_col=event_col,
            extra_columns=extra_columns,
            penalizer=best_penalizer,
            l1_ratio=best_l1_ratio,
            extra_column_penalizer=extra_column_penalizer,
        )

        if stability_selection:
            stable_features, _ = _bootstrap_stability_selection(
                train_frame,
                feature_columns,
                duration_col=duration_col,
                event_col=event_col,
                extra_columns=extra_columns,
                penalizer=best_penalizer,
                l1_ratio=best_l1_ratio,
                n_bootstrap=n_bootstrap_stability,
                frequency_threshold=stability_frequency_threshold,
                seed=seed + fold_index,
                extra_column_penalizer=extra_column_penalizer,
            )
            final_features = stable_features if stable_features else en_selected
        else:
            final_features = en_selected

        # 2026-08-15 GEVŞETİLDİ (klinik kovaryat desteği -- görev
        # talimatı: `clinical_base` kolu `feature_columns=[]` ile
        # çağrılabilir, yani radyomik ADAY havuzu baştan boş -- bu
        # durumda `final_features`'IN boş olması bir HATA değil,
        # BEKLENEN davranıştır). `extra_columns=[]` olan TÜM eski
        # çağıranlarda (mevcut testler dahil) davranış BİREBİR AYNI --
        # `final_features` boşsa `extra_columns` de boş olduğundan hata
        # yine fırlar.
        if not final_features and not extra_columns:
            raise RuntimeError(
                f"Dış fold {fold_index}: elastic-net/stabilite seçimi "
                "hiçbir özellik bırakmadı (ve extra_columns/klinik "
                "kovaryat da yok) -- grid/eşik gözden geçirilmeli. "
                "Sessizce boş bir modelle devam edilmedi."
            )

        final_model_columns = final_features + extra_columns
        final_penalizer_argument = _build_penalizer_argument(
            final_features,
            extra_columns,
            penalizer=best_penalizer,
            extra_column_penalizer=extra_column_penalizer,
        )
        cox_final = CoxPHFitter(penalizer=final_penalizer_argument, l1_ratio=best_l1_ratio)
        cox_final.fit(
            train_frame[final_model_columns + [duration_col, event_col]],
            duration_col=duration_col,
            event_col=event_col,
        )
        test_partial_hazard = cox_final.predict_partial_hazard(test_frame[final_model_columns])
        c_index = concordance_index(
            test_frame[duration_col], -test_partial_hazard, test_frame[event_col]
        )

        fold_results.append(
            NestedCVFoldResult(
                fold=fold_index,
                n_train=len(train_frame),
                n_test=len(test_frame),
                n_events_train=int(train_frame[event_col].sum()),
                n_events_test=int(test_frame[event_col].sum()),
                best_penalizer=best_penalizer,
                best_l1_ratio=best_l1_ratio,
                inner_mean_c_index=best_score if any_valid_score else float("nan"),
                n_selected_features_elastic_net=len(en_selected),
                n_selected_features_final=len(final_features),
                selected_features=final_features,
                c_index=c_index,
            )
        )

    return pd.DataFrame([result.__dict__ for result in fold_results])


def evaluate_external_test(
    fitted_model,
    external_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    duration_col: str = "survival_days",
    event_col: str = "event",
    extra_columns: list[str] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """TCGA (harici test) üzerinde C-index + bootstrap güven aralığı.

    plan.txt madde 435: "n=50 küçük olduğundan güven aralığını da
    raporla" -- bu yüzden nokta tahmini yerine percentile-bootstrap CI
    döner.

    AÇIK TASARIM SORUSU (çözülmedi, ekip kararı gerekiyor, bkz.
    decisions/2026-08-09-combat-fit-iki-blokaj-region-ve-pyradiomics-parametre.md'deki
    TCGA-referans-apply tartışmasıyla AYNI SINIF sorun):
    Eğer `fitted_model` `source` dummy kovaryatı İÇERİYORSA (bkz.
    `_assemble_training_frame`), TCGA için bu kovaryatın hangi değeri
    kullanılmalı? TCGA eğitimde HİÇ görülmedi, `extra_columns`'daki
    source dummy'lerinin TCGA satırlarında ne olacağı bu fonksiyona
    ÖRTÜK bir varsayılan (örn. hepsini 0/referans yapmak) UYGULANMAZ --
    çağıran taraf `external_frame`'e bu kolonları AÇIKÇA doldurarak
    geçmeli. Varsayılan olarak 0 (referans batch UPenn'miş gibi)
    doldurmak, ComBat'ın TCGA-referans-apply'ının cebirsel özdeşlik
    olduğu bulgusuyla AYNI YAKLAŞIM olur ama bunun Cox risk skorları
    için de istatistiksel olarak geçerli olup olmadığı AYRICA
    doğrulanmalı, burada VARSAYILMADI.
    """

    from lifelines.utils import concordance_index

    extra_columns = extra_columns or []
    model_columns = feature_columns + extra_columns
    missing = set(model_columns) - set(external_frame.columns)
    if missing:
        raise ValueError(
            f"external_frame eksik kolon(lar): {sorted(missing)} -- "
            "TCGA'da source dummy kolonları varsa AÇIKÇA doldurulmalı "
            "(bkz. bu fonksiyonun docstring'indeki açık tasarım sorusu)."
        )

    partial_hazard = fitted_model.predict_partial_hazard(external_frame[model_columns])
    point_estimate = concordance_index(
        external_frame[duration_col], -partial_hazard, external_frame[event_col]
    )

    rng = np.random.default_rng(seed)
    n = len(external_frame)
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        sample = external_frame.iloc[sample_idx]
        sample_hazard = partial_hazard.iloc[sample_idx]
        if sample[event_col].sum() == 0:
            continue  # bootstrap örneğinde hiç event yoksa C-index tanımsız, atla
        bootstrap_scores.append(
            concordance_index(sample[duration_col], -sample_hazard, sample[event_col])
        )

    alpha = 1 - confidence_level
    if bootstrap_scores:
        lower = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))
    else:
        lower = upper = float("nan")

    return {
        "n_patients": n,
        "n_events": int(external_frame[event_col].sum()),
        "c_index": float(point_estimate),
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence_level": confidence_level,
        "n_bootstrap_valid": len(bootstrap_scores),
        "n_bootstrap_requested": n_bootstrap,
    }


def compute_shap_values(
    fitted_model,
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    n_background: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """SHAP entegrasyon noktası -- ÇAĞRILABİLİR ama bu Hafta 3 iskelet
    kapsamında GERÇEK PRODÜKSİYON VERİSİYLE çalıştırılmaz (yalnız
    sentetik veriyle birim testi, bkz. tests/test_cox_model.py).

    lifelines `CoxPHFitter` doğrudan scikit-learn arayüzünde olmadığı
    için `shap.Explainer`'a `predict_partial_hazard`'ı saran bir
    fonksiyon veriliyor (KernelExplainer tabanlı, model-agnostik --
    yavaş ama lifelines ile uyumlu tek genel yöntem).
    """

    import shap

    background = frame[feature_columns].sample(
        n=min(n_background, len(frame)), random_state=seed
    )

    def _predict(data: np.ndarray) -> np.ndarray:
        as_frame = pd.DataFrame(data, columns=feature_columns)
        return fitted_model.predict_partial_hazard(as_frame).to_numpy()

    explainer = shap.KernelExplainer(_predict, background)
    shap_values = explainer.shap_values(frame[feature_columns], silent=True)
    return pd.DataFrame(shap_values, columns=feature_columns, index=frame.index)
