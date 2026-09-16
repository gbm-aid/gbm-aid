"""TCGA whole/core maskeleriyle PyRadiomics 107 özellik çıkarımı (C32), radiomics'e INSERT.

KARAR (2026-08-13, bkz. decisions/2026-08-13-pyradiomics-c32-bincount-
karari.md): bu script artık **C32** sözleşmesiyle çalışır -- N4 bias-field
düzeltmesi + T1ce-özel (bu script zaten yalnız T1ce işler, modalite
filtresi doğal) kaynak-bazlı Z-score + `binCount=32` (binWidth=25 DEĞİL).
Eski davranış (ham NAS görüntüsü + binWidth=25 varsayılanı,
`segmentation_tool='TCGA-ground-truth'`) SİLİNMEDİ/DEĞİŞTİRİLMEDİ -- ayrı
bir `segmentation_tool` adıyla (`TCGA-ground-truth-C32`) YENİ satırlar
INSERT edilir, eski satırlar baseline olarak korunur (versiyonlu ayrım,
Codex'in önerisi).

Neden bu değişiklik gerekti (2026-08-13 bulgu zinciri, özet -- tam detay
karar dosyasında):
1. Reviewer bulgusu (CRITICAL): bu script (ve UPenn/LUMIERE eşleniği) N4+
   Z-score çıktısını HİÇ KULLANMIYORDU, ham NAS görüntüsünden doğrudan
   PyRadiomics çalıştırıyordu -- `pipeline/harmonization.py` docstring'indeki
   "değişmez işlem sırası" (N4->Zscore->PyRadiomics->ComBat) ile çelişiyordu.
2. Barış'ın 15-taramalık karşılaştırması: ham vs doğru-sıralı arasında
   first-order+texture'ın ~%98-100'ü farklı; kök neden binWidth=25'in
   Z-skorlanmış aralıkta (~[-1,+3]) yalnız ~2 etkin gri-seviye bin
   üretmesi (GLCM/GLRLM/GLSZM/NGTDM/GLDM'yi dejenere ediyor).
3. Ayırt edici perturbasyon testi (`tools/probe_perturbation_stability.py`,
   68.480 satır): texture ICC'si binCount'a geçince (C32/C64≈0,97) A/B'den
   (≈0,33) KESİN OLARAK daha kararlı; C32, küçük ROI'lerde (<500 voksel)
   C64'ten daha kararlı (C64 bir özellikte ICC=-0,677 negatif çıktı) --
   **C32 seçildi**.

AÇIK RİSK (karar dosyasında da kayıtlı): first-order'ın mutlak-ölçek
özellikleri (Energy/TotalEnergy/RMS/Variance/Mean) hiçbir varyantta
(A/B/C32/C64) mutlak-yoğunluk-ölçeğine karşı kararlı değil (ICC≈0,08-0,32)
-- bunun ComBat'ın feature-seviyesi düzeltmesiyle telafi edilmesi
BEKLENİYOR ama TCGA ComBat kapsamı dışında olduğu için (2026-08-09 kararı)
bu risk TCGA için ayrıca değerlendirilmeli.

Ayıklanan özellik sınıfları (shape+firstorder+glcm+glrlm+glszm+ngtdm+gldm =
14+18+75=107) `decisions/2026-08-07-107-ozellik-formulu.md` ile uyumludur.
PyRadiomics ayarları artık extractor çağrısında AÇIKÇA veriliyor
(binCount=32, normalize=False, imageTypes={'Original':{}}) -- 2026-08-13
Codex bulgusu: boş `RadiomicsFeatureExtractor()` çağrısında `binWidth`
extractor'ın `settings` sözlüğünde HİÇ GÖRÜNMÜYORDU (özellik sınıfı içine
gömülü varsayılan), sürüm değişince sessizce değişebilirdi -- artık
görünür. Ortamın kendi paket sürümleri `requirements-pyradiomics.txt`'te
pinlenmiştir.

TCGA görüntüleri zaten %100 1x1x1mm izotropik olduğu için (hafta2_durum_
ozeti, Hafta 1 bulgusu) resampling uygulanmaz -- ama görüntü/maske
geometrisi yine de PyRadiomics'e geçmeden ÖNCE açıkça doğrulanır
(mimari kural: sessiz fallback yasak, `validate_image_mask_geometry`).

İşlem sırası (mimari kural, `pipeline/harmonization.py` ile tutarlı):
  Pass 1: her tarama için görüntü/maske çözümle + doğrula, N4 bias-field
          düzeltmesi uygula (ham görüntü DEĞİŞTİRİLMEZ, türetilmiş çıktı
          `--processed-root` altına atomik yazılır).
  Pass 2: Pass 1'in TÜM N4 çıktılarından TEK bir T1ce-özel, kaynak-bazlı
          (yalnız TCGA) Z-score istatistiği fit edilir
          (`fit_source_zscore_statistics`, `--stats-path`'e yazılır --
          üretim `artifacts/week2/zscore/source_stats.json`'a
          DOKUNULMAZ, ayrı/versiyonlu dosya).
  Pass 3: her N4 çıktısı Pass 2'nin istatistiğiyle Z-score normalize
          edilir, PyRadiomics C32 ile çalıştırılır, `--apply` verilirse
          `segmentation_tool='TCGA-ground-truth-C32'` ile INSERT edilir.

Var olan `TCGA-ground-truth` (eski A-yöntemi) satırlarına HİÇ DOKUNULMAZ
(UPDATE yok, farklı segmentation_tool -- UNIQUE(scan_id, segmentation_tool,
tumor_region) çakışması olmaz).

Bu script YALNIZCA Python 3.10 ortamında (pyradiomics 3.0.1,
`requirements-pyradiomics.txt`) çalışır.

KISIT (2026-08-13 görev talimatı): bu turda TAM KOHORT YENİDEN ÇIKARIMI
YAPILMAZ -- kod hazırlanmıştır ama `--apply` ile gerçek bir koşu bu
oturumda ÇALIŞTIRILMADI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).absolute().parents[1]
# NOT (2026-08-14, B1 tam kohort koşusu sırasında bulundu/düzeltildi):
# `.resolve()` DEĞİL `.absolute()` KULLANILIYOR -- `.resolve()` bu
# makinede `subst X:` ile kurulan ASCII-safe sürücüyü GERÇEK (Türkçe
# karakterli, `Barış` içeren) yola çözümleyip SimpleITK'nin NIfTI
# yazıcısının bilinen path bug'ına (bkz. takim/mert.md operasyonel notu)
# sessizce geri düşürüyordu -- N4 çıktısı yazılamayıp `RuntimeError`
# fırlatıyordu (CANLI görüldü: B1 TCGA koşusu ilk denemede tüm taramalarda
# N4 yazımında başarısız oldu). `run_pyradiomics_lumiere.py` zaten
# `.absolute()` kullanıyordu (bugünkü başarılı 55dk'lık preflight koşusu
# bunun kanıtı) -- bu script/UPenn eşleniği tutarlı hale getirildi.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from psycopg2.extras import RealDictCursor  # noqa: E402

from db_connection import get_connection  # noqa: E402
from pipeline.harmonization import (  # noqa: E402
    apply_n4_bias_correction,
    apply_zscore_normalization,
    fit_source_zscore_statistics_if_needed,
    get_fitted_zscore_record,
    n4_output_path,
)
from pipeline.resampling import validate_image_mask_geometry  # noqa: E402
from pipeline.segmentation import _find_nifti, resolve_nas_path  # noqa: E402

CANONICAL_SOURCE = "TCGA"
SEGMENTATION_TOOL_LEGACY = "TCGA-ground-truth"  # A-yöntemi -- DOKUNULMAZ, baseline
SEGMENTATION_TOOL_C32 = "TCGA-ground-truth-C32"  # bu script artık BUNU üretir
REGION_MASK_FILE = {
    "WT": ("whole.nii.gz", "whole.nii"),
    "TC": ("core.nii.gz", "core.nii"),
}

OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "week3" / "pyradiomics"
DEFAULT_PROCESSED_ROOT = OUTPUT_ROOT / "c32_n4_zscore"
# 2026-08-13 H5 düzeltmesi (review-gate BLOKE bulgusu): ÖNCEDEN üç script
# (TCGA/UPenn/LUMIERE) AYNI `source_stats_t1ce_c32.json` dosyasını
# PAYLAŞIYORDU -- kilitsiz oku-değiştir-tam-dosya-yaz deseniyle, iki
# script Pass 2'de eşzamanlı/örtüşen çalışırsa biri diğerinin az önce
# yazdığı kaydı (BAŞKA bir kaynağın kaydı olsa bile, çünkü TÜM payload
# okunup TÜM payload yeniden yazılıyor) ezebilirdi. Artık HER script
# kendi kaynağına özel AYRI bir dosya kullanıyor -- kaynak-bazlı paralel
# koşu artık güvenli. İSTENİRSE `--stats-path` ile eskisi gibi paylaşımlı
# bir dosyaya da yönlendirilebilir (bu artık BİLİNÇLİ bir tercih, varsayılan
# DEĞİL).
DEFAULT_ZSCORE_STATS_PATH = OUTPUT_ROOT / "source_stats_t1ce_c32_tcga.json"
DEFAULT_REPORT = OUTPUT_ROOT / "tcga_pyradiomics_c32.csv"

# A1.1 (2026-08-14, Codex review TUR 2 -- "image_count gate"): TCGA'nın
# beklenen tam kohort N4-üretilebilir tarama sayısı.
# DİKKAT -- BU DEĞER CANLI DB İLE İKİ KEZ ÖLÇÜLDÜ, İLK VARSAYIM
# (154) YANLIŞTI: "154" TCGA-GBM'in TÜM modalitelerinin (T1ce 39 +
# T1 39 + T2 36 + FLAIR 40 = 154) toplamı -- `hafta2_durum_ozeti.md`'deki
# "154 tarama" ifadesi buna atıfta bulunuyor, bu script'in `_fetch_tcga_
# t1ce_scans()`'i ise YALNIZ T1ce filtreler. Canlı `SELECT COUNT(*) ...
# WHERE ds.source_name='TCGA-GBM' AND ms.modality='T1ce'` = **39** (2026-
# 08-14'te doğrulandı, `_fetch_tcga_t1ce_scans()`'in KENDİ sorgusuyla
# birebir aynı JOIN yapısı kullanılarak). 39, eski `TCGA-ground-truth`
# DB yazımının (78 satır = 39 hasta x 2 bölge [WT,TC]) hasta sayısıyla da
# TUTARLI. TCGA'nın hazır whole/core maskesi %100 mevcut olduğu için
# (mask/geometri anomalisi beklenmiyor) 39'un TAMAMI N4-üretilebilir
# olmalı -- YÜKSEK GÜVENİLİRLİK. Operatör yine de `--expected-image-count`
# ile override edebilir.
DEFAULT_EXPECTED_IMAGE_COUNT = 39

# A1.5 (2026-08-14): provenance manifest sabitleri.
EXPECTED_PYRADIOMICS_VERSION = "3.0.1"
ZSCORE_SCOPE = "t1ce_source"

FIELDNAMES = [
    "patient_id",
    "scan_id",
    "region",
    "status",
    "voxel_volume",
    "legacy_volume",
    "volume_match",
]


def _make_extractor():
    from radiomics import featureextractor

    logging.getLogger("radiomics").setLevel(logging.ERROR)
    # TÜM parametreler AÇIKÇA veriliyor (2026-08-13 Codex bulgusu: boş
    # extractor çağrısında binWidth settings'te görünmüyor) -- hiçbir
    # örtük varsayılana güvenilmiyor.
    extractor = featureextractor.RadiomicsFeatureExtractor(
        binCount=32,
        normalize=False,
    )
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()
    for cls in ("shape", "firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm"):
        extractor.enableFeatureClassByName(cls)
    return extractor


def _split_features(result: dict) -> tuple[dict, dict, dict]:
    shape, first_order, texture = {}, {}, {}
    for key, value in result.items():
        if not key.startswith("original_"):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if key.startswith("original_shape_"):
            shape[key] = numeric
        elif key.startswith("original_firstorder_"):
            first_order[key] = numeric
        elif key.startswith(("original_glcm_", "original_glrlm_", "original_glszm_", "original_ngtdm_", "original_gldm_")):
            texture[key] = numeric
    return shape, first_order, texture


def _n4_params_signature() -> dict[str, str]:
    """N4 parametrelerinin (env'den okunan) imzası -- A1.2 cache anahtarı."""

    return {
        "n4_iterations": os.environ.get("GBMAID_N4_ITERATIONS", "50,50,30,20"),
        "n4_shrink_factor": os.environ.get("GBMAID_N4_SHRINK_FACTOR", "4"),
    }


def _n4_cache_meta_path(cached_path: Path) -> Path:
    return cached_path.with_name(cached_path.name + ".n4meta.json")


def _input_content_hash(input_path: Path) -> str:
    """Girdi dosyasının GERÇEK bayt içeriğinin sha256'sı (A1.2).

    Yalnız YOLA değil, İÇERİĞE bakar -- aynı yoldaki dosya farklı bir
    NAS senkronizasyonu/düzeltmesiyle değişmiş olsa bile cache bunu
    fark eder (yol-tabanlı hash tek başına bunu YAKALAYAMAZ).
    """

    hasher = hashlib.sha256()
    with input_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _n4_cache_hit(cached_path: Path, *, input_path: Path) -> bool:
    """H4 + A1.2 düzeltmesi -- N4 çıktısı diskte VAR mı, TAM OKUNABİLİR mi,
    SONLU + SIFIR-DIŞI içerik taşıyor mu VE AYNI N4 parametreleri + AYNI
    girdi içeriğiyle mi üretilmiş, kontrol et.

    2026-08-14 Codex review TUR 2 bulgusu (A1.2, bkz. decisions/2026-08-13-
    pyradiomics-c32-bincount-karari.md Riskler #1): önceki sürüm yalnız
    `ImageFileReader.ReadImageInformation()` ile HEADER okuyordu -- gövdesi
    bozuk (örn. NaN/Inf dolu veya tamamen sıfır) ama header'ı sağlam bir
    dosya sessizce "cache hit" sayılabiliyordu; ayrıca cache anahtarı
    (`n4_output_path()` -> `_derived_output_path()`) yalnız girdi YOLUNU
    hashliyordu -- N4 parametreleri (iterasyon/shrink_factor) değişse veya
    girdi dosyasının GERÇEK içeriği değişse bile ESKİ N4 çıktısı sessizce
    yeniden kullanılabiliyordu. `apply_n4_bias_correction()`'ın kendisi
    (production fonksiyonu, başka çağıranları var) DEĞİŞTİRİLMEDEN, bu
    fonksiyon SCRIPT tarafında bir SIDECAR meta dosyasıyla
    (`<n4çıktısı>.n4meta.json`, `_write_n4_cache_meta()`) bu iki boşluğu
    kapatıyor.
    """

    if not cached_path.is_file():
        return False

    meta_path = _n4_cache_meta_path(cached_path)
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("n4_params") != _n4_params_signature():
            return False
        if meta.get("input_sha256") != _input_content_hash(input_path):
            return False
    except (json.JSONDecodeError, OSError, KeyError):
        return False

    try:
        import SimpleITK as sitk

        image = sitk.ReadImage(str(cached_path))
        array = sitk.GetArrayViewFromImage(image)
        if array.size == 0:
            return False
        if not np.isfinite(array).all():
            return False
        if not np.any(array != 0):
            return False
    except Exception:  # noqa: BLE001
        return False
    return True


def _write_n4_cache_meta(cached_path: Path, input_path: Path) -> None:
    """A1.2 -- cache-hit doğrulamasının okuduğu sidecar'ı YAZ (atomik)."""

    meta_path = _n4_cache_meta_path(cached_path)
    payload = {
        "n4_params": _n4_params_signature(),
        "input_sha256": _input_content_hash(input_path),
        "input_path": str(input_path),
    }
    temporary = meta_path.with_name(f".{meta_path.name}.{uuid.uuid4().hex}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, meta_path)
    finally:
        temporary.unlink(missing_ok=True)


def _apply_n4_cached(image_path: str) -> str:
    """N4 çıktısı zaten diskte (geçerli biçimde, AYNI parametre/girdiyle)
    VARSA yeniden hesaplama; değilse hesapla VE A1.2 sidecar'ını yaz.

    2026-08-13 H4 düzeltmesi (review-gate BLOKE bulgusu, en yüksek
    kaldıraçlı düzeltme): `apply_n4_bias_correction()`'ın kendisi HİÇ
    değiştirilmedi (production fonksiyonu, başka çağıranları var,
    davranış değişikliği regresyon riski taşırdı) -- cache kontrolü
    burada, script tarafında AYRI bir katman olarak eklendi. Bu, Pass 1
    kesintiye uğrayıp restart edildiğinde önceden hesaplanmış N4
    çıktılarının YENİDEN hesaplanmasını önler (B1'in kök nedeni).
    """

    input_path = Path(image_path)
    cached_path = n4_output_path(image_path)
    if _n4_cache_hit(cached_path, input_path=input_path):
        return str(cached_path)
    result = apply_n4_bias_correction(image_path)
    _write_n4_cache_meta(Path(result), input_path)
    return result


def _normalized_pyradiomics_version() -> str:
    import radiomics

    raw = str(getattr(radiomics, "__version__", "")).strip()
    return raw[1:] if raw.lower().startswith("v") else raw


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _extractor_settings_signature(extractor) -> str:
    """Extractor'ın GERÇEK runtime ayarlarının kanonik JSON hash'i.

    2026-08-14 GENİŞLETME (Barış onayı, Codex review MEDIUM-1): ilk
    manifest şeması yalnız `bin_count`/`normalize`'ı KENDİ KENDİNE beyan
    ediyordu -- bu, 2026-08-13'ün 'sessiz varsayılan' hata sınıfına
    (`binWidth` boş extractor'da `settings`'te HİÇ GÖRÜNMÜYORDU) karşı
    yetersiz. Bu imza `settings` + `enabledImagetypes` + `enabledFeatures`
    ÜÇÜNÜ birlikte hash'ler -- extractor'ın GERÇEKTEN nasıl yapılandırıldığı
    (tüm ayarlar + hangi görüntü tiplerinin/özellik sınıflarının açık
    olduğu) tek bir değere bağlanır, sürüm/parametre kayması olursa hash
    değişir.
    """

    canonical = {
        "settings": extractor.settings,
        "enabledImagetypes": extractor.enabledImagetypes,
        "enabledFeatures": extractor.enabledFeatures,
    }
    payload = json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_provenance_manifest(
    *,
    extractor,
    report_path: Path,
    rows_out: list[dict[str, object]],
    n4_applied: bool,
    stats_path: Path,
    zscore_stats_image_count: int | None,
    expected_image_count: int | None,
    full_cohort_ids: set[object],
    covered_id_field: str,
    partial_run: bool,
    offset: int,
    limit: int | None,
) -> Path:
    """A1.5 (2026-08-14, GENİŞLETİLMİŞ şema) -- C32 rapor CSV'sinin YANINA
    provenance manifest'i yaz.

    Neden ZORUNLU (Codex review, 2026-08-14): eski A-yöntemi CSV'siyle
    yeni C32 CSV'sinin BAŞLIKLARI BİREBİR AYNI -- şemadan C32 olup
    olmadığı anlaşılmıyor. DB yazıcıları bu manifest'i ARTIK ZORUNLU
    tutuyor, yoksa/uyuşmazsa exit 2 veriyor.

    GENİŞLETME GEREKÇESİ (Barış, Codex review MEDIUM-1): ilk şema
    manifest'in KENDİ KENDİNİ BEYAN etmesine dayanıyordu -- `report_sha256`
    yalnız manifest'i CSV'ye bağlıyordu, N4 çıktısına/kullanılan Z-score
    istatistiğine/gerçek extractor ayarlarına bağlamıyordu; bayat/yanlış
    bir stats dosyasıyla üretilmiş bir CSV de "geçerli" manifest alabilirdi.
    Artık `zscore_stats_sha256`/`zscore_stats_image_count` (A1.1 gate'in
    aynı bilgisi, downstream'e taşınıyor) ve `extractor_settings_sha256`
    (sessiz-varsayılan sınıfına karşı) da manifest'e giriyor. Hiçbir alan
    elle sabit YAZILMAZ -- runtime nesnelerinden (extractor, stats dosyası,
    CSV, script dosyasının kendisi) üretilir; üretilemiyorsa manifest
    YAZILMAZ, hata verilir.

    `run_complete`, BU çağrının `--offset/--limit` kullanıp kullanmadığına
    DEĞİL, `report_path`'teki BİRİKMİŞ satırların (idempotent-resume
    deseni gereği önceki parçalı koşuların satırları da `rows_out`'ta
    birikir) `full_cohort_ids` (DB'den offset/limit UYGULANMADAN ÖNCE
    çekilen TAM liste) ile TAM örtüşüp örtüşmediğine bakar -- parçalı
    koşularda son parça `--offset/--limit` vermeden çalıştırılmasa bile
    (örn. son parça da bir dilim ise) birikmiş CSV'nin GERÇEKTEN tam
    kohortu kapsayıp kapsamadığı doğru yansıtılır. `zscore_stats_
    image_count != expected_image_count` durumunda da (yalnız
    `--skip-image-count-gate` ile Pass 3'e ulaşılmışsa mümkün, aksi halde
    gate zaten reddeder) `run_complete=False` yazılır.
    """

    actual_version = _normalized_pyradiomics_version()
    if actual_version != EXPECTED_PYRADIOMICS_VERSION:
        raise RuntimeError(
            f"pyradiomics sürüm uyuşmazlığı: kurulu={actual_version!r}, "
            f"beklenen={EXPECTED_PYRADIOMICS_VERSION!r} -- provenance "
            "manifest'i YAZILMADI (2026-08-13 kararı: pyradiomics==3.0.1 "
            "pinli, sürüm sürüklenmesi sessizce kabul edilmez)."
        )

    bin_count = extractor.settings.get("binCount")
    normalize = extractor.settings.get("normalize")
    if bin_count != 32 or normalize is not False:
        raise RuntimeError(
            "Extractor ayarları C32 sözleşmesiyle uyuşmuyor "
            f"(binCount={bin_count!r}, normalize={normalize!r}) -- "
            "provenance manifest'i YAZILMADI."
        )

    if not report_path.is_file():
        raise RuntimeError(
            f"Rapor CSV'si bulunamadı, manifest yazılamıyor: {report_path}"
        )
    if not stats_path.is_file():
        raise RuntimeError(
            f"Z-score stats dosyası bulunamadı, manifest yazılamıyor: {stats_path}"
        )

    report_sha256 = _file_sha256(report_path)
    with report_path.open("r", newline="", encoding="utf-8") as handle:
        report_rows_list = list(csv.DictReader(handle))
    report_rows = len(report_rows_list)

    status_distribution: dict[str, int] = {}
    covered_ids: set[str] = set()
    for row in report_rows_list:
        status_key = str(row.get("status", ""))
        status_distribution[status_key] = status_distribution.get(status_key, 0) + 1
        identifier = row.get(covered_id_field)
        if identifier not in (None, ""):
            covered_ids.add(str(identifier))

    missing_ids = full_cohort_ids - covered_ids
    image_count_mismatch = (
        expected_image_count is not None
        and zscore_stats_image_count is not None
        and zscore_stats_image_count != expected_image_count
    )
    run_complete = (not missing_ids) and not image_count_mismatch

    # `.absolute()`, `.resolve()` DEGIL -- K12 kurali (bkz.
    # tests/test_no_resolve_path_regression.py): `.resolve()` Windows'ta
    # `subst X:` eslemesini Turkce karakterli gercek yola geri cozer ve
    # ITK'nin NIfTI yazimini bozar. Burada yalniz `.name`/sha256 icin
    # kullanildigi halde desenin kopyalanmasi 2026-08-14'te tam kohort
    # kosusunu cokertti -- desen kaynagindan temizlendi. Davranis
    # DEGISMEDI: ayni dosya, ayni bayt, ayni sha256, ayni `.name`.
    generator_path = Path(__file__).absolute()

    manifest = {
        "extraction_contract": "C32",
        "bin_count": bin_count,
        "normalize": normalize,
        "n4_applied": n4_applied,
        "zscore_scope": ZSCORE_SCOPE,
        "pyradiomics_version": actual_version,
        "extractor_settings_sha256": _extractor_settings_signature(extractor),
        "image_types": sorted(extractor.enabledImagetypes.keys()),
        "enabled_feature_classes": sorted(extractor.enabledFeatures.keys()),
        "zscore_stats_path": str(stats_path),
        "zscore_stats_sha256": _file_sha256(stats_path),
        "zscore_stats_source": CANONICAL_SOURCE,
        "zscore_stats_image_count": zscore_stats_image_count,
        "expected_image_count": expected_image_count,
        "generator_script": generator_path.name,
        "generator_script_sha256": _file_sha256(generator_path),
        "cohort": CANONICAL_SOURCE,
        "report_sha256": report_sha256,
        "report_rows": report_rows,
        "status_distribution": status_distribution,
        "run_complete": run_complete,
        "missing_id_count": len(missing_ids),
        "partial_run": partial_run,
        "offset": offset if partial_run else None,
        "limit": limit if partial_run else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = report_path.with_name(report_path.name + ".provenance.json")
    temporary = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest_path


def _fetch_tcga_t1ce_scans(connection) -> list[dict]:
    """`tools/compute_segmentation_volumes.py::_fetch_tcga_t1ce_scans` ile BİREBİR aynı sorgu.

    Eski `run_pyradiomics_tcga.py`'nin aksine, bu script artık `radiomics`
    tablosundaki (eski tool'a bağlı) "pending NULL" satırlarına değil,
    doğrudan `mr_scans`'e bağlı -- C32 için hiçbir ön-yazılmış satıra
    ihtiyaç yok, kendi kendine yeter.
    """

    cursor = connection.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(
            "SELECT ms.scan_id, ms.patient_id, ms.file_path "
            "FROM mr_scans AS ms "
            "JOIN patients AS p ON p.patient_id = ms.patient_id "
            "JOIN dataset_sources AS ds ON ds.source_id = p.source_id "
            "WHERE ds.source_name = 'TCGA-GBM' AND ms.modality = 'T1ce' "
            "ORDER BY ms.scan_id"
        )
        return list(cursor.fetchall())
    finally:
        cursor.close()


def _fetch_legacy_volumes(connection) -> dict[tuple[int, str], float]:
    """Eski `TCGA-ground-truth` (A-yöntemi) hacimlerini çapraz-kontrol için önceden yükle."""

    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT scan_id, tumor_region, tumor_volume_mm3 FROM radiomics "
            "WHERE segmentation_tool = %s",
            (SEGMENTATION_TOOL_LEGACY,),
        )
        result: dict[tuple[int, str], float] = {}
        for scan_id, region, volume in cursor.fetchall():
            if volume is not None:
                result[(scan_id, region)] = float(volume)
        return result
    finally:
        cursor.close()


def _insert_c32_row(
    connection,
    *,
    scan_id: int,
    region: str,
    volume_mm3: float | None,
    shape: dict,
    first_order: dict,
    texture: dict,
) -> bool:
    """INSERT dener (`segmentation_tool='TCGA-ground-truth-C32'`), gerçekten
    yeni satır yazıldıysa True döner (idempotent -- ON CONFLICT DO NOTHING)."""

    surface_area = shape.get("original_shape_SurfaceArea")
    entropy = first_order.get("original_firstorder_Entropy")
    contrast = texture.get("original_glcm_Contrast")
    cursor = connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO radiomics "
            "(scan_id, segmentation_tool, tumor_region, tumor_volume_mm3, "
            "shape_features, first_order_features, texture_features, "
            "surface_area, entropy, contrast, feature_source) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'computed') "
            "ON CONFLICT (scan_id, segmentation_tool, tumor_region) DO NOTHING "
            "RETURNING radiomics_id",
            (
                scan_id,
                SEGMENTATION_TOOL_C32,
                region,
                volume_mm3,
                json.dumps(shape),
                json.dumps(first_order),
                json.dumps(texture),
                surface_area,
                entropy,
                contrast,
            ),
        )
        return cursor.fetchone() is not None
    finally:
        cursor.close()


def _load_done_keys(report_path: Path) -> tuple[list[dict[str, object]], set[tuple[str, str]]]:
    rows: list[dict[str, object]] = []
    done: set[tuple[str, str]] = set()
    if not report_path.is_file():
        return rows, done
    with report_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
            if row.get("status") in ("OK", "YAZILDI"):
                done.add((row["scan_id"], row["region"]))
    return rows, done


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "TCGA whole/core maskelerinden C32 (N4+T1ce-özel Z-score+binCount=32) "
            "PyRadiomics 107 özellik üret, segmentation_tool='TCGA-ground-truth-C32' "
            "ile INSERT et (eski 'TCGA-ground-truth' baseline'ına DOKUNMAZ)."
        )
    )
    parser.add_argument("--apply", action="store_true", help="Belirtilmezse DB'ye yazılmaz (dry-run varsayılan).")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "İşlenecek listenin (scan_id sırasına göre) BAŞINDAN kaç kaydın "
            "atlanacağı -- 2026-08-13 B1 düzeltmesi: --limit TEK BAŞINA her "
            "zaman listenin PREFIX'ini alıyordu (`scans[:limit]`), bu yüzden "
            "parçalı/kesintili koşularda her seferinde limit'i ARTIRMAK "
            "gerekiyordu. --offset ile birlikte doğal bir chunking sağlanır: "
            "örn. `--offset 300 --limit 300` bir sonraki 300'lük dilimi işler. "
            "(N4 çıktı cache'i sayesinde -- H4 düzeltmesi -- zaten-işlenmiş "
            "taramalar üzerinden tekrar geçmek artık ucuz, bu yüzden --offset "
            "ZORUNLU değil, yalnız kolaylık.)"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="--offset'ten sonraki ilk N taramayla sınırla.")
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=DEFAULT_ZSCORE_STATS_PATH,
        help="C32 T1ce-özel Z-score fit çıktısı (üretim source_stats.json DEĞİL).",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="N4/Z-score ara çıktılarının yazılacağı kök (GBMAID_PROCESSED_ROOT).",
    )
    parser.add_argument(
        "--fresh-fit",
        action="store_true",
        help="Var olan --stats-path'i silip TAZE fit et (varsayılan: dosya varsa üzerine ekler/günceller).",
    )
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--expected-image-count",
        type=int,
        default=None,
        help=(
            "A1.1 (2026-08-14) image_count gate -- Pass 3'e geçmeden ÖNCE, "
            "stats dosyasındaki donmuş istatistiğin image_count'unun bu "
            f"değere eşit olması ZORUNLU (TCGA için önerilen: {DEFAULT_EXPECTED_IMAGE_COUNT}, "
            "bkz. script docstring'i/DEFAULT_EXPECTED_IMAGE_COUNT gerekçesi). "
            "Eşit değilse koşu REDDEDİLİR (sessiz devam YOK). Belirtilmezse "
            "ve --skip-image-count-gate de verilmezse script Pass 3'e HİÇ "
            "geçmez (exit 2)."
        ),
    )
    parser.add_argument(
        "--skip-image-count-gate",
        action="store_true",
        help="A1.1 gate'ini BİLİNÇLİ olarak atla (yalnız küçük smoke test/keşif koşuları için).",
    )
    args = parser.parse_args()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    os.environ["GBMAID_PROCESSED_ROOT"] = str(args.processed_root)
    # KRİTİK (smoke testte bulundu): apply_zscore_normalization() stats
    # dosyasını YALNIZ bu ortam değişkeninden okur (fit_source_zscore_
    # statistics()'e verilen stats_path= parametresi apply() tarafını
    # ETKİLEMEZ) -- probe_perturbation_stability.py'nin deseniyle tutarlı.
    os.environ["GBMAID_ZSCORE_STATS_PATH"] = str(args.stats_path)

    if args.fresh_fit and args.stats_path.is_file():
        args.stats_path.unlink()

    rows_out, already_done = _load_done_keys(args.report)
    if already_done:
        print(f"Idempotent devam: {len(already_done)} (scan,bölge) zaten OK/YAZILDI, atlanacak.", flush=True)

    extractor = _make_extractor()
    connection = get_connection(readonly=True)
    try:
        scans = _fetch_tcga_t1ce_scans(connection)
        legacy_volumes = _fetch_legacy_volumes(connection)
    finally:
        connection.close()

    # A1.5 -- offset/limit UYGULANMADAN ÖNCE, TAM kohortun scan_id kümesi
    # (manifest'in `run_complete` hesabı için).
    full_cohort_scan_ids = {str(scan["scan_id"]) for scan in scans}

    if args.offset:
        scans = scans[args.offset :]
    if args.limit:
        scans = scans[: args.limit]

    t_start = time.time()

    def flush() -> None:
        # B2 düzeltmesi (2026-08-13 review-gate BLOKE bulgusu): ÖNCEDEN bu
        # script'te `flush()` diye bir kavram YOKTU, rapor yalnız `main()`'in
        # en SONUNDA (tüm Pass 3 bittikten sonra) tek seferde yazılıyordu --
        # UPenn/LUMIERE'deki periyodik-flush deseninin AKSİNE. Artık üçü de
        # tutarlı: Pass 3 sırasında periyodik + en sonda kesin flush.
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows_out)

    # ---- Pass 1: çözümle + doğrula + N4 ----
    resolved_scans: list[dict[str, object]] = []
    n4_paths: list[str] = []
    for scan in scans:
        scan_id = scan["scan_id"]
        patient_id = scan["patient_id"]
        pending_regions = [
            region for region in REGION_MASK_FILE if (str(scan_id), region) not in already_done
        ]
        if not pending_regions:
            continue

        try:
            image_path = resolve_nas_path(scan["file_path"])
        except FileNotFoundError as exc:
            for region in pending_regions:
                rows_out.append(
                    {
                        "patient_id": patient_id,
                        "scan_id": scan_id,
                        "region": region,
                        "status": f"GORUNTU_YOK:{exc}"[:200],
                        "voxel_volume": "",
                        "legacy_volume": "",
                        "volume_match": "",
                    }
                )
            continue

        region_masks: dict[str, Path] = {}
        for region in pending_regions:
            names = REGION_MASK_FILE[region]
            mask_path = _find_nifti(image_path.parent, names)
            if mask_path is None:
                rows_out.append(
                    {
                        "patient_id": patient_id,
                        "scan_id": scan_id,
                        "region": region,
                        "status": "MASKE_YOK",
                        "voxel_volume": "",
                        "legacy_volume": "",
                        "volume_match": "",
                    }
                )
                continue
            # Mimari kural: geometri eşleşmiyorsa PyRadiomics'e GEÇME, açık hata ver.
            geometry = validate_image_mask_geometry(image_path, mask_path)
            if not geometry["geometry_match"]:
                rows_out.append(
                    {
                        "patient_id": patient_id,
                        "scan_id": scan_id,
                        "region": region,
                        "status": "GEOMETRI_UYUSMUYOR",
                        "voxel_volume": "",
                        "legacy_volume": "",
                        "volume_match": "",
                    }
                )
                continue
            region_masks[region] = mask_path

        if not region_masks:
            continue

        try:
            n4_path = _apply_n4_cached(str(image_path))
        except Exception as exc:  # noqa: BLE001
            for region in region_masks:
                rows_out.append(
                    {
                        "patient_id": patient_id,
                        "scan_id": scan_id,
                        "region": region,
                        "status": f"HATA:N4:{type(exc).__name__}:{exc}"[:200],
                        "voxel_volume": "",
                        "legacy_volume": "",
                        "volume_match": "",
                    }
                )
            continue

        n4_paths.append(n4_path)
        resolved_scans.append(
            {
                "patient_id": patient_id,
                "scan_id": scan_id,
                "n4_path": n4_path,
                "region_masks": region_masks,
            }
        )
        elapsed = time.time() - t_start
        print(f"[N4 tamam] {patient_id} scan_id={scan_id}, gecen={elapsed:.0f}s", flush=True)

    # ---- Pass 2: TEK T1ce-özel, kaynak-bazlı (TCGA) Z-score fit ----
    # 2026-08-13 B3 düzeltmesi (review-gate BLOKE bulgusu, CRITICAL): ÖNCEDEN
    # bu blok KOŞULSUZ `fit_source_zscore_statistics()` çağırıyordu -- kısmi
    # bir koşu restart edildiğinde `n4_paths` yalnız KALAN (henüz OK/YAZILDI
    # olmayan) taramaları içerdiğinden, bu SESSİZCE aynı kaynağın TEK kanonik
    # istatistiğini FARKLI bir (mean,std) ile ÜZERİNE YAZARDI -- aynı kaynağın
    # hastaları hangi partide işlendiğine göre farklı normalize edilirdi.
    # `fit_source_zscore_statistics_if_needed()` artık stats dosyasında
    # ZATEN geçerli bir kayıt varsa (force=True/--fresh-fit verilmedikçe)
    # REFİT ETMİYOR, mevcut değeri REUSE ediyor -- tüm restart'lar arasında
    # TEK bir dondurulmuş istatistik kullanılır (bkz. harmonization.py
    # docstring'indeki ödünleşim notu).
    # `scope=ZSCORE_SCOPE` (2026-09-13, karar 28): fit edilen istatistik
    # dosyasinin kokune `"zscore_scope": "t1ce_source"` beyani YAZILIR.
    # `apply_zscore_normalization()`'in kapsam guard'i bu beyani ZORUNLU
    # kilar -- beyansiz ya da havuzlanmis (`source_pooled_all_modalities`)
    # bir dosya ile normalize etmeyi fail-closed reddeder.
    fit_record = fit_source_zscore_statistics_if_needed(
        n4_paths,
        CANONICAL_SOURCE,
        stats_path=args.stats_path,
        force=args.fresh_fit,
        scope=ZSCORE_SCOPE,
    )
    if fit_record is not None:
        print(
            f"Z-score fit ({CANONICAL_SOURCE}, T1ce-özel, n={len(n4_paths)}): "
            f"mean={fit_record['mean']:.6f} std={fit_record['std']:.6f}",
            flush=True,
        )
    else:
        print(
            f"REUSE: {CANONICAL_SOURCE} için stats dosyasında zaten geçerli bir "
            "istatistik var, refit ATLANDI (B3 guard) -- tüm kohort AYNI "
            "(mean,std) ile normalize edilecek. Zorla yeniden fit için "
            "--fresh-fit kullanın.",
            flush=True,
        )

    # ---- A1.1 (2026-08-14) image_count gate -- Pass 3 başlamadan ZORUNLU ----
    # Codex review TUR 2 bulgusu (bkz. decisions/2026-08-13-pyradiomics-c32-
    # bincount-karari.md Riskler #1): B3 guard'ı "süreç öldü -> restart"ı
    # kapattı ama --offset/--limit ile başlayan İLK koşu bir ALT-KÜMEDEN
    # istatistik fit edip kanonikleştirebilir -- sonraki parçalar bunu
    # sessizce REUSE eder. Tek çare: Pass 3'e geçmeden ÖNCE, donmuş
    # istatistiğin `image_count`'unun BEKLENEN TAM KOHORT sayısına eşit
    # olduğunu doğrulamak. Eşit değilse KOŞU REDDEDİLİR (sessiz devam YOK).
    gate_record = (
        fit_record
        if fit_record is not None
        else get_fitted_zscore_record(CANONICAL_SOURCE, stats_path=args.stats_path)
    )
    if args.skip_image_count_gate:
        print(
            "UYARI: --skip-image-count-gate verildi -- image_count kontrolü "
            "BİLİNÇLİ OLARAK ATLANDI (operatör override, A1.1 guard devre dışı).",
            flush=True,
        )
    else:
        if args.expected_image_count is None:
            print(
                "HATA: --expected-image-count verilmedi (ve --skip-image-count-gate "
                "de verilmedi) -- A1.1 guard Pass 3'e GEÇEMEZ. Beklenen tam kohort "
                f"sayısını AÇIKÇA belirtin (TCGA için önerilen: {DEFAULT_EXPECTED_IMAGE_COUNT}) "
                "veya bilinçli bir smoke test ise --skip-image-count-gate kullanın.",
                file=sys.stderr,
            )
            return 2
        actual_count = gate_record["image_count"] if gate_record else None
        if actual_count != args.expected_image_count:
            print(
                "HATA: image_count gate REDDETTİ -- stats dosyasındaki donmuş "
                f"istatistiğin image_count={actual_count!r}, beklenen tam kohort="
                f"{args.expected_image_count}. Bu, istatistiğin bir ALT-KÜMEDEN fit "
                "edilip kanonikleştirildiği anlamına gelebilir (A1.1 riski). Koşu "
                f"REDDEDİLDİ, Pass 3 ÇALIŞTIRILMADI. stats_path={args.stats_path}. "
                "Bilinçli olarak devam etmek için --fresh-fit ile TAM kohortu "
                "yeniden fit edin veya --skip-image-count-gate kullanın.",
                file=sys.stderr,
            )
            return 2
        print(
            f"image_count gate GEÇTİ: {actual_count} == beklenen "
            f"{args.expected_image_count}.",
            flush=True,
        )

    # ---- Pass 3: Z-score normalize + C32 PyRadiomics + (varsa) INSERT ----
    connection = get_connection(readonly=not args.apply)
    ok_count = fail_count = write_count = 0
    try:
        for idx, scan in enumerate(resolved_scans, start=1):
            patient_id = scan["patient_id"]
            scan_id = scan["scan_id"]
            try:
                zscore_path = apply_zscore_normalization(scan["n4_path"], CANONICAL_SOURCE)
            except Exception as exc:  # noqa: BLE001
                for region in scan["region_masks"]:
                    rows_out.append(
                        {
                            "patient_id": patient_id,
                            "scan_id": scan_id,
                            "region": region,
                            "status": f"HATA:ZSCORE:{type(exc).__name__}:{exc}"[:200],
                            "voxel_volume": "",
                            "legacy_volume": "",
                            "volume_match": "",
                        }
                    )
                    fail_count += 1
                continue

            for region, mask_path in scan["region_masks"].items():
                geometry = validate_image_mask_geometry(zscore_path, mask_path)
                if not geometry["geometry_match"]:
                    rows_out.append(
                        {
                            "patient_id": patient_id,
                            "scan_id": scan_id,
                            "region": region,
                            "status": "GEOMETRI_UYUSMUYOR_ZSCORE",
                            "voxel_volume": "",
                            "legacy_volume": "",
                            "volume_match": "",
                        }
                    )
                    fail_count += 1
                    continue

                try:
                    result = extractor.execute(str(zscore_path), str(mask_path))
                    shape, first_order, texture = _split_features(result)
                except Exception as exc:  # noqa: BLE001
                    rows_out.append(
                        {
                            "patient_id": patient_id,
                            "scan_id": scan_id,
                            "region": region,
                            "status": f"HATA:{type(exc).__name__}:{exc}"[:200],
                            "voxel_volume": "",
                            "legacy_volume": "",
                            "volume_match": "",
                        }
                    )
                    fail_count += 1
                    continue

                voxel_volume = shape.get("original_shape_VoxelVolume")
                legacy_volume = legacy_volumes.get((scan_id, region))
                volume_match = ""
                if voxel_volume is not None and legacy_volume is not None:
                    volume_match = abs(float(voxel_volume) - legacy_volume) < 1.0

                status = "OK"
                if args.apply:
                    inserted = _insert_c32_row(
                        connection,
                        scan_id=scan_id,
                        region=region,
                        volume_mm3=voxel_volume,
                        shape=shape,
                        first_order=first_order,
                        texture=texture,
                    )
                    status = "YAZILDI" if inserted else "ATLANDI_ZATEN_VAR"
                    if inserted:
                        write_count += 1

                rows_out.append(
                    {
                        "patient_id": patient_id,
                        "scan_id": scan_id,
                        "region": region,
                        "status": status,
                        "voxel_volume": voxel_volume,
                        "legacy_volume": legacy_volume if legacy_volume is not None else "",
                        "volume_match": volume_match,
                    }
                )
                ok_count += 1
                print(f"{patient_id} {region}: {status} (vol_match={volume_match})", flush=True)

            if idx % args.progress_every == 0:
                if args.apply:
                    connection.commit()
                # B2 düzeltmesi: periyodik rapor flush -- UPenn/LUMIERE ile
                # tutarlı, Pass 3 uzun sürerken kesinti anındaki ilerlemeyi
                # diske yazar (öncesinde bu script'te HİÇ yoktu, rapor
                # yalnız en sonda tek seferde yazılıyordu).
                flush()
                elapsed = time.time() - t_start
                print(
                    f"[Pass3 C32] [{idx}/{len(resolved_scans)}] tarama işlendi, "
                    f"ok={ok_count} fail={fail_count} yazildi={write_count}, "
                    f"gecen={elapsed:.0f}s, rapor guncellendi: {args.report}",
                    flush=True,
                )

        if args.apply:
            connection.commit()
    finally:
        connection.close()

    flush()

    partial_run = bool(args.offset) or (args.limit is not None)
    manifest_path = _write_provenance_manifest(
        extractor=extractor,
        report_path=args.report,
        rows_out=rows_out,
        n4_applied=True,
        stats_path=args.stats_path,
        zscore_stats_image_count=(
            gate_record["image_count"] if gate_record else None
        ),
        expected_image_count=args.expected_image_count,
        full_cohort_ids=full_cohort_scan_ids,
        covered_id_field="scan_id",
        partial_run=partial_run,
        offset=args.offset,
        limit=args.limit,
    )

    print(
        f"\nBitti. {len(resolved_scans)} tarama işlendi. ok={ok_count} fail={fail_count} "
        f"yazildi={write_count}. Rapor: {args.report}\n"
        f"Provenance manifest: {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
