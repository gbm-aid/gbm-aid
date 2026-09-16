"""Hazır segmentasyon maskelerinden bölge hacmi (mm3) hesapla.

Yalnız hacim/şekil ön-yazımı için kullanılır; PyRadiomics'in tam 107 özellik
çıkarımı (Hafta 3, Mert) bu modülün kapsamında değildir. Etiket sözleşmesi
gerçek NAS verisinden (SimpleITK ile) doğrulanmıştır:

- TCGA whole.nii.gz / core.nii.gz: binary maske, foreground=1 -> WT / TC
- UPenn images_segm / automated_segm: BraTS etiketleri 1=NC, 2=ED, 4=ET
- LUMIERE DeepBraTumIA native/segmentation/<mod>_seg_mask.nii.gz (ANA KAYNAK,
  Hafta 2 kararı): 1=Necrosis (Necrotic_NonEnhancing), 2=Contrast-enhancing
  (Enhancing_Core), 3=Edema (Edema_Compartment) -- ham LUMIERE görüntüsüyle
  aynı geometride.
  DÜZELTME (2026-08-09, Mert/imaging-agent, GÖREV 3 sırasında bulundu):
  önceki sürüm burada 1=Contrast-enhancing/2=Necrosis yazıyordu (TERSİ) --
  bu YANLIŞTI. Canlı `radiomics` tablosundaki (segmentation_tool=
  'DeepBraTumIA', Hafta 1'de harici CSV'den yüklenen, bu modülle
  ÜRETİLMEMİŞ) 3 farklı hasta/vizit için NAS'taki maskeler doğrudan
  SimpleITK ile sayılıp DB hacimleriyle karşılaştırıldı: label=1 sistematik
  olarak DB'nin "Necrosis" hacmiyle (fark <100mm³), label=2 DB'nin
  "Contrast-enhancing" hacmiyle eşleşti (fark <500mm³) -- eskiden yazan
  "measured_volumes_in_mm3.json ile çapraz doğrulandı" iddiası bu üç
  örnekte DOĞRULANAMADI (o JSON'a bu oturumda erişilmedi, önceki doğrulama
  yöntemi/kapsamı belirsiz). Eski (yanlış) sözleşmeyle bu modül üzerinden
  DB'ye YAZILMIŞ hiçbir satır YOK (mevcut 'DeepBraTumIA' satırları harici
  CSV kaynaklı, bu dict'i hiç kullanmadı) -- yani düzeltme geriye dönük
  bir veri bozulmasını GERİ ALMIYOR, yalnız bu modülün GELECEKTEKİ
  kullanımını düzeltiyor.
- LUMIERE HD-GLIO-AUTO registered/segmentation.nii.gz (yalnız DUYARLILIK
  ANALİZİ, ana modele girmez): 1=Non-enhancing, 2=Contrast-enhancing
- UCSF-PDGM (2026-08-18, Barış onayı -- tam kohort C32 çıkarımı):
  `<ID>_tumor_segmentation.nii.gz`, etiketler UPenn ile BİREBİR AYNI:
  1=NC (Necrotic), 2=ED (Edema), 4=ET (Enhancing Tumor). Bugünün pilot
  doğrulamasında (25 hasta, 15 hasta yoğunluk yönü) LUMIERE-tipi bir
  takas GÖRÜLMEDİ -- şema UPenn/BraTS ile aynı. Girdi görüntüsü
  KESİNLİKLE `<ID>_T1c.nii.gz` (ham) -- `<ID>_T1c_bias.nii.gz`
  (UCSF'in KENDİ N4 düzeltmesi) KULLANILMAZ; bizim kendi N4'ümüz ham
  görüntüye uygulanır, aksi halde çifte bias-field düzeltmesi riski
  doğar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pipeline.lazy_sitk import sitk  # 2026-09-16: TEMBEL import (SAC blogu) -- bkz. pipeline/lazy_sitk.py

REGION_LABELS_BY_MASK_SOURCE: dict[str, dict[str, int]] = {
    "provided_whole_tumor": {"WT": 1},
    "provided_tumor_core": {"TC": 1},
    "upenn_expert": {"NC": 1, "ED": 2, "ET": 4},
    "upenn_automated_approx": {"NC": 1, "ED": 2, "ET": 4},
    "lumiere_deepbratumia_native": {
        "Necrosis": 1,
        "Contrast-enhancing": 2,
        "Edema": 3,
    },
    "lumiere_hd_glio_fallback": {"Non-enhancing": 1, "Contrast-enhancing": 2},
    # 2026-08-18 (Barış onayı) -- UCSF-PDGM tam kohort. UPenn ile
    # ÖZDEŞ etiket kontratı (bkz. modül-üstü docstring): NC=1/ED=2/ET=4.
    "ucsf_native": {"NC": 1, "ED": 2, "ET": 4},
}


class UnknownMaskSourceError(KeyError):
    """`mask_source` için bölge/etiket sözleşmesi tanımlı değil."""


def compute_region_volumes_mm3(
    mask_path: str | Path,
    label_map: dict[str, int],
) -> dict[str, float]:
    """Bir maske dosyasındaki her etiket için fiziksel hacmi (mm3) hesapla.

    Hacim = etiketli voksel sayısı x spacing'in üç ekseninin çarpımı.
    """

    mask = sitk.ReadImage(str(mask_path))
    spacing = mask.GetSpacing()
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

    array = sitk.GetArrayViewFromImage(mask)
    volumes: dict[str, float] = {}
    for region_name, label_value in label_map.items():
        voxel_count = int((array == label_value).sum())
        volumes[region_name] = voxel_count * voxel_volume_mm3
    return volumes


def compute_volumes_for_mask_source(
    mask_path: str | Path,
    mask_source: str,
) -> dict[str, float]:
    """`resolve_ready_mask()`'ın döndürdüğü `mask_source` anahtarına göre hacim üret."""

    try:
        label_map = REGION_LABELS_BY_MASK_SOURCE[mask_source]
    except KeyError as exc:
        raise UnknownMaskSourceError(
            f"Bilinmeyen mask_source, etiket sözleşmesi tanımlanmamış: {mask_source!r}"
        ) from exc
    return compute_region_volumes_mm3(mask_path, label_map)


# ============================================================
# A2 (2026-08-14) -- Türetilmiş WT/TC bölgeleri
# ============================================================
#
# Karar: decisions/2026-08-13-bolge-stratejisi-ve-modelleme-protokolu.md
# Bölüm 3 ("Çıkarımda 5 bölge") -- UPenn/LUMIERE için NC/ED/ET'e ek olarak
# WT = NC∪ED∪ET, TC = NC∪ET türetilir (TCGA için DEĞİŞTİRİLMEZ, zaten
# native WT/TC üretiyor -- bu modülün kapsamı DIŞINDA).
#
# ZORUNLU GUARD (decisions/2026-08-14-hd-glio-fallback-kanonik-vektorden-
# haric.md "AÇIK İŞ" bölümü): türetme YALNIZ üç mask_source için
# tanımlıdır. `REGION_LABELS_BY_MASK_SOURCE`'ta hâlâ canlı duran
# `lumiere_hd_glio_fallback` (ödem etiketi YOK, yalnız Non-enhancing/
# Contrast-enhancing) girdisi buraya BİLİNÇLİ OLARAK dahil EDİLMEDİ --
# jenerik "mevcut tüm etiketlerin birleşimi = WT" yazılsaydı ödemsiz bir
# "WT" sessizce üretilir, 2026-08-14 kararı delinirdi.

DERIVED_REGION_ROLE_ALIASES: dict[str, dict[str, str]] = {
    "upenn_expert": {"NC": "NC", "ED": "ED", "ET": "ET"},
    "upenn_automated_approx": {"NC": "NC", "ED": "ED", "ET": "ET"},
    "lumiere_deepbratumia_native": {
        "Necrosis": "NC",
        "Edema": "ED",
        "Contrast-enhancing": "ET",
    },
    # 2026-08-18 (Barış onayı) -- UCSF-PDGM tam kohort. NC/ED/ET zaten
    # UPenn'deki rol adlarıyla birebir (kimlik eşleme), WT_derived/
    # TC_derived türetmesi UPenn'dekiyle AYNI voksel-birleşimi kuralını
    # kullanır (bkz. DERIVED_REGION_COMPONENT_ROLES, değiştirilmedi).
    "ucsf_native": {"NC": "NC", "ED": "ED", "ET": "ET"},
}

# WT = NC ∪ ED ∪ ET, TC = NC ∪ ET (Menze 2015 PMID 25494501, Bakas 2017
# PMID 28872634, Bakas UPenn-GBM 2022 PMID 35906241 -- bkz. karar dosyası
# Bölüm 1).
DERIVED_REGION_COMPONENT_ROLES: dict[str, tuple[str, ...]] = {
    "WT_derived": ("NC", "ED", "ET"),
    "TC_derived": ("NC", "ET"),
}

WT_TC_DERIVATION_ALLOWED_MASK_SOURCES = frozenset(DERIVED_REGION_ROLE_ALIASES)


class UnsupportedWTTCDerivationError(ValueError):
    """WT/TC türetmesi bu `mask_source` için tanımlı DEĞİL (2026-08-14 kararı)."""


def build_derived_region_masks(
    mask_path: str | Path,
    mask_source: str,
) -> dict[str, sitk.Image]:
    """`WT_derived` (NC∪ED∪ET) ve `TC_derived` (NC∪ET) ikili (0/1) maskelerini üret.

    Döndürülen `sitk.Image`'lar `mask_path`'in geometrisini (`CopyInformation`)
    taşır, değerleri yalnız 0/1'dir (PyRadiomics'e `label=1` ile verilmeye
    hazır). Bu fonksiyon dosyaya YAZMAZ -- bellekte kalır, çağıran taraf
    isterse `extractor.execute(image_path, derived_image, label=1)` ile
    doğrudan kullanabilir (PyRadiomics `maskFilepath` için hem dosya yolu
    hem `SimpleITK.Image` kabul eder).

    Hata
    ----
    UnsupportedWTTCDerivationError
        `mask_source` izinli üç değerden (`upenn_expert`,
        `upenn_automated_approx`, `lumiere_deepbratumia_native`) biri
        DEĞİLSE -- sessiz/jenerik bir varsayım YAPILMAZ (2026-08-14
        kararı, bkz. modül üstü not).
    """

    if mask_source not in WT_TC_DERIVATION_ALLOWED_MASK_SOURCES:
        raise UnsupportedWTTCDerivationError(
            f"WT/TC türetmesi mask_source={mask_source!r} için tanımlı "
            "DEĞİL (2026-08-14 kararı: decisions/2026-08-14-hd-glio-"
            "fallback-kanonik-vektorden-haric.md). İzinli mask_source'lar: "
            f"{sorted(WT_TC_DERIVATION_ALLOWED_MASK_SOURCES)}."
        )

    role_aliases = DERIVED_REGION_ROLE_ALIASES[mask_source]
    label_map = REGION_LABELS_BY_MASK_SOURCE[mask_source]
    role_to_label: dict[str, int] = {
        role: label_map[region_name] for region_name, role in role_aliases.items()
    }

    mask = sitk.ReadImage(str(mask_path))
    array = sitk.GetArrayFromImage(mask)

    derived: dict[str, sitk.Image] = {}
    for derived_region, roles in DERIVED_REGION_COMPONENT_ROLES.items():
        included_labels = [role_to_label[role] for role in roles]
        binary = np.isin(array, included_labels).astype(np.uint8)
        derived_image = sitk.GetImageFromArray(binary)
        derived_image.CopyInformation(mask)
        derived[derived_region] = derived_image

    return derived


def derived_region_has_voxels(derived_image: sitk.Image) -> bool:
    """Türetilmiş ikili maskenin en az 1 voksel içerip içermediğini kontrol et.

    Yapısal olarak WT/TC her zaman en az bir bileşenden geliyor olsa da,
    (nadiren) TÜM bileşen etiketleri 0-voksel ise (örn. hem NC hem ED hem
    ET etiketi mevcut değilse) türetilmiş bölge de boş kalabilir --
    çağıran taraf bunu native bölgelerdeki `LABEL_YOK_0_VOXEL` durumuyla
    AYNI şekilde ele almalı (sessizce PyRadiomics'e geçmemeli).
    """

    array = sitk.GetArrayViewFromImage(derived_image)
    return bool(np.any(array != 0))
