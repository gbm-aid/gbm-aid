"""LUMIERE DeepBraTumIA maskeleriyle kanonik PyRadiomics 107 özellik çıkarımı (C32).

KARAR (2026-08-13, bkz. decisions/2026-08-13-pyradiomics-c32-bincount-
karari.md): bu script artık **C32** sözleşmesiyle çalışır -- N4 bias-field
düzeltmesi + T1ce-özel (bu script zaten yalnız T1ce/CT1 işler, modalite
filtresi doğal) kaynak-bazlı Z-score + `binCount=32` (binWidth=25 DEĞİL,
Bulgu 2'nin `binWidth=25`'i de artık geçersiz -- ikisi de aynı kök nedenin,
Z-skorlanmış aralıkta sabit binWidth'in dejenerasyona yol açmasının,
farklı belirtileriydi). Eski davranış (ham NAS görüntüsü + binWidth=25)
`lumiere_pyradiomics_applied.csv`/`lumiere_pyradiomics_resampled.csv`'de
KORUNDU, bu script onları ÜZERİNE YAZMAZ -- varsayılan çıktı ayrı bir
dosyaya (`lumiere_pyradiomics_c32.csv`) yazılır.

Neden bu değişiklik gerekti (2026-08-13 bulgu zinciri -- tam detay karar
dosyasında, bkz. `tools/run_pyradiomics_tcga.py` docstring'i, aynı gerekçe
üç script'te tekrar): Reviewer bulgusu (N4+Z-score hiç kullanılmıyordu) +
Barış'ın 15-taramalık ölçümü + ayırt edici perturbasyon testi (texture ICC
C32'de ≈0,97, A/B'de ≈0,33).

Bağlam (bkz. decisions/2026-08-09-combat-fit-iki-blokaj-region-ve-
pyradiomics-parametre.md, Bulgu 2 -- ARTIK C32 ile birlikte KAPANDI): eski
LUMIERE precomputed DeepBraTumIA radyomikleri binWidth=5/normalize=True/
normalizeScale=100/voxelArrayShift=300 ile üretilmişti, TCGA/UPenn'in eski
binWidth=25/normalize=False varsayılanlarıyla UYUŞMUYORDU. C32 kararıyla
artık ÜÇ kaynak da (TCGA/UPenn/LUMIERE) BİREBİR AYNI extractor ayarlarıyla
(binCount=32, normalize=False, yalnız 'original' görüntü tipi) VE aynı N4+
T1ce-özel-Zscore ön-işlemesiyle çıkarılıyor.

Kapsam: LUMIERE'in DeepBraTumIA native maskeleri (Necrosis/Edema/
Contrast-enhancing -- UPenn'in NC/ED/ET'ine 1:1 karşılık gelir, bkz.
pipeline/radiomics_volume.py::REGION_LABELS_BY_MASK_SOURCE). Kanonik
görüntü TCGA/UPenn ile TUTARLI olacak şekilde YALNIZ T1ce (LUMIERE'de CT1).

HD-GLIO-AUTO fallback'e (`mask_source == "lumiere_hd_glio_fallback"`)
DÜŞÜLMEZ -- bu araç yalnız duyarlılık analizi içindir, ana kanonik
vektöre KARIŞTIRILMAZ (mimari kural). Böyle bir vizit
`hd_glio_fallback_excluded` olarak işaretlenip atlanır, PyRadiomics
çalıştırılmaz.

Segmentasyon aracı adı (bu script kendisi DB'ye yazmıyor -- ama `tools/
write_lumiere_pyradiomics_to_db.py`'nin, ayrı takip görevi olarak,
kullanması gereken ad): `LUMIERE-PyRadiomics-107-C32` -- mevcut
`DeepBraTumIA`/`HD-GLIO-AUTO`/`LUMIERE-PyRadiomics-107` (A-yöntemi, eski
binWidth=25) precomputed satırlarına dokunmaz, ayrı bir segmentation_tool.

RESAMPLING ENTEGRASYONU (Task #19'dan DEVRALINDI, DAVRANIŞI DEĞİŞMEDİ):
Görüntü/maske fiziksel geometrisi uyuşmadığında PyRadiomics'e SESSİZCE
geçilmez -- açıkça `pipeline/resampling.py::resample_image_and_mask()` ile
ortak 1mm ızgaraya taşınıp yeniden denenir (mimari kural: sessiz fallback
yasak). Bu geometri-kurtarma adımı N4/Z-score'DAN ÖNCE, HAM görüntü/maske
üzerinde çalışır (N4/Zscore geometriyi DEĞİŞTİRMEZ, `CopyInformation` ile
grid korunur -- bu yüzden sıralama önemsiz, ama kurtarma mantığının
BASİTLİĞİ için ham görüntüde yapılmaya devam ediyor). PyRadiomics'in kendi
(daha sıkı) dahili tolerance kontrolü extraction sırasında hâlâ
reddedebilir -- bu durumda AYNI resample fonksiyonu, artık Z-score
görüntüsü + (varsa zaten bir kez resample edilmiş) maske üzerinde BİR KEZ
daha denenir (resample_image_and_mask salt geometrik bir dönüşümdür,
Z-skorlanmış girdi üzerinde de doğru çalışır).

İşlem sırası (mimari kural, `pipeline/harmonization.py` ile tutarlı):
  Pass 1: her tarama için hazır maske çöz, HD-GLIO fallback'i dışla,
          geometri uyuşmazlığında resample-kurtarma dene, N4 bias-field
          düzeltmesi uygula.
  Pass 2: Pass 1'in TÜM N4 çıktılarından TEK bir T1ce-özel, kaynak-bazlı
          (yalnız LUMIERE) Z-score istatistiği fit edilir (`--stats-path`'e
          yazılır -- üretim `artifacts/week2/zscore/source_stats.json`'a
          DOKUNULMAZ).
  Pass 3: her N4 çıktısı Pass 2'nin istatistiğiyle Z-score normalize
          edilir, PyRadiomics C32 ile çalıştırılır (gerekirse ikinci
          seviye resample-kurtarma denenir), sonuç CSV'ye yazılır.

DB'YE HİÇBİR YAZMA YAPILMAZ (yalnız SELECT, salt-okunur). CSV'yi
Supabase'e yazmak db-agent'ın ayrı dry-run/apply protokolüdür.

KISIT (2026-08-13 görev talimatı): bu turda TAM KOHORT YENİDEN ÇIKARIMI
YAPILMAZ -- kod hazırlanmıştır ama gerçek bir koşu bu oturumda
ÇALIŞTIRILMADI (yalnız küçük `--limit` smoke testi).

Bu script yalnız Python 3.10 ortamında (pyradiomics 3.0.1,
`requirements-pyradiomics.txt`) çalışır.
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from psycopg2.extras import RealDictCursor  # noqa: E402

from db_connection import get_connection  # noqa: E402
from pipeline.harmonization import (  # noqa: E402
    apply_n4_bias_correction,
    apply_zscore_normalization,
    fit_source_zscore_statistics_if_needed,
    get_fitted_zscore_record,
    n4_output_path,
)
from pipeline.radiomics_volume import (  # noqa: E402
    REGION_LABELS_BY_MASK_SOURCE,
    build_derived_region_masks,
    derived_region_has_voxels,
)
from pipeline.resampling import resample_image_and_mask  # noqa: E402
from pipeline.segmentation import (  # noqa: E402
    ReadyMaskNotFoundError,
    resolve_nas_path,
    resolve_ready_mask,
)

CANONICAL_SOURCE = "LUMIERE"
CANONICAL_MASK_SOURCE = "lumiere_deepbratumia_native"
# Bu script kendisi DB'ye yazmaz (yalnız CSV) -- ama `tools/
# write_lumiere_pyradiomics_to_db.py`'nin (ayrı takip görevi) hangi
# `segmentation_tool` adını kullanması gerektiğini burada belgeliyoruz.
SEGMENTATION_TOOL_C32_FOR_DB_WRITE = "LUMIERE-PyRadiomics-107-C32"

OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "week3" / "pyradiomics"
DEFAULT_PROCESSED_ROOT = OUTPUT_ROOT / "c32_n4_zscore"
# 2026-08-13 H5 düzeltmesi (review-gate BLOKE bulgusu): ÖNCEDEN TCGA/UPenn/
# LUMIERE scriptleri AYNI dosyayı paylaşıyordu (kilitsiz oku-tam-dosya-yaz
# deseniyle) -- artık her script kendi kaynağına özel AYRI bir dosya
# kullanıyor, kaynak-bazlı paralel koşu güvenli hale geldi.
DEFAULT_ZSCORE_STATS_PATH = OUTPUT_ROOT / "source_stats_t1ce_c32_lumiere.json"
DEFAULT_REPORT = OUTPUT_ROOT / "lumiere_pyradiomics_c32.csv"

# Geometri-kurtarma (resample_image_and_mask) çıktıları -- Task #19'un
# `lumiere_resampled_for_pyradiomics` kökünden AYRI (farklı amaç: burada
# N4/Zscore SONRASI görüntüler de resample edilebilir, karışmasın).
DEFAULT_RESAMPLE_ROOT = OUTPUT_ROOT / "lumiere_resampled_for_pyradiomics_c32"

# A1.1 (2026-08-14, Codex review TUR 2 -- "image_count gate"): LUMIERE'nin
# beklenen tam kohort N4-üretilebilir tarama sayısı.
# ✅ KESİNLEŞTİRİLDİ (2026-08-14, `--pass1-only` PREFLIGHT koşusuyla --
# TAM kohort, PRODUCTION yollarıyla, TAM koşunun kullanacağı AYNI DB
# sorgusu + NAS çözümleme + HD-GLIO eleme kurallarıyla, kesintisiz, 3294s
# ~55dk): `lumiere_pyradiomics_c32.csv.pass1_preflight.json` ->
# `total_scans_considered=599`, `n4_reached_count=580`, kategori dökümü:
# `GORUNTU_YOK=8` (NAS'ta dosya yok) + `MASKE_YOK_VEYA_GEOMETRI:
# resample_denendi_basarisiz=6` (geometri-kurtarma denendi başarısız
# oldu) + `label_yok_all_regions=5` (maske çözüldü ama TÜM hedef bölgeler
# -- native+türetilmiş -- 0-voksel) = 19 düşen tarama; 599-19=580,
# TAM ÖRTÜŞÜYOR (gizemli/açıklanamayan kayıp YOK). `resampled_recovered_
# count=357` (kohortun >%60'ı geometri-kurtarmadan geçti -- Task #19'un
# ~%40 oranıyla aynı mertebede, LUMIERE'de nadir bir yol DEĞİL).
# Önceki `591` tahmini (eski A-yöntemi script'inin `lumiere_pyradiomics_
# resampled.csv`'sinden) SAPTI çünkü FARKLI/DAR bir tanımla ölçülmüştü:
# yalnız "maske ÇÖZÜMLENDİ mi" (mask_source resolution) sayıyordu --
# görüntüsü olmayan (GORUNTU_YOK=8), geometrisi kurtarılamayan (6) ve
# tüm-bölgeleri-boş (5) taramaları bu C32 script'indeki gibi AYRI
# kategorilere ayırıp N4'TEN ÖNCE elemiyordu. Sapmanın nedeni yukarıdaki
# kategori dökümüyle TAM açıklanıyor -- gizemli/açıklanamayan bir kayıp
# YOK, 591 basitçe daha az kesin bir ölçümdü.
DEFAULT_EXPECTED_IMAGE_COUNT = 580

# A1.5 (2026-08-14): provenance manifest sabitleri.
EXPECTED_PYRADIOMICS_VERSION = "3.0.1"
ZSCORE_SCOPE = "t1ce_source"

FIELDNAMES = [
    "patient_id",
    "scan_id",
    "timepoint_label",
    "region",
    "mask_source",
    "status",
    "voxel_volume",
    "db_volume",
    "volume_match",
    "volume_diff_mm3",
    "warning",
    "resampled",
    "shape_features_json",
    "first_order_features_json",
    "texture_features_json",
    "surface_area",
    "entropy",
    "contrast",
]


def _make_extractor():
    from radiomics import featureextractor

    logging.getLogger("radiomics").setLevel(logging.ERROR)
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
        elif key.startswith(
            ("original_glcm_", "original_glrlm_", "original_glszm_", "original_ngtdm_", "original_gldm_")
        ):
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
    """Girdi dosyasının GERÇEK bayt içeriğinin sha256'sı (A1.2) -- bkz.
    `run_pyradiomics_tcga.py`'deki AYNI fonksiyonun docstring'i."""

    hasher = hashlib.sha256()
    with input_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _n4_cache_hit(cached_path: Path, *, input_path: Path) -> bool:
    """H4 + A1.2 düzeltmesi -- N4 çıktısı diskte VAR mı, TAM OKUNABİLİR mi,
    SONLU + SIFIR-DIŞI içerik taşıyor mu VE AYNI N4 parametreleri + AYNI
    girdi içeriğiyle mi üretilmiş, kontrol et.

    2026-08-14 Codex review TUR 2 bulgusu (A1.2) -- bkz. `run_pyradiomics_
    tcga.py`'deki AYNI fonksiyonun docstring'i, tam gerekçe orada. NOT:
    LUMIERE'de `input_path` -- HAM görüntü DEĞİL, gerekirse Pass 1'in
    geometri-kurtarma (resample) adımından geçmiş bir görüntü olabilir --
    içerik hash'i bu YENİDEN ÜRETİLEN dosyanın GERÇEK bayt içeriğine
    bakar, bu yüzden resample çıktısı değişirse (örn. resample parametreleri
    değişirse) cache doğru şekilde miss verir.
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
    kaldıraçlı düzeltme -- bkz. `run_pyradiomics_tcga.py`'deki AYNI
    fonksiyonun docstring'i, gerekçe orada tekrar yazılmadı). Pass 1
    kesintiye uğrayıp restart edildiğinde önceden hesaplanmış N4
    çıktılarının YENİDEN hesaplanmasını önler. NOT: LUMIERE'de girdi
    `resolved_image_path` -- HAM görüntü DEĞİL, gerekirse Pass 1'in
    geometri-kurtarma (resample) adımından geçmiş bir görüntü olabilir --
    `n4_output_path()` bu yolun kendisini hash'lediği için sorun yok
    (resample çıktısı da deterministik bir dosya yolu, önceki koşuyla
    AYNI dosya yolunu üretir).
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
    """Bkz. `run_pyradiomics_tcga.py`'deki AYNI fonksiyonun docstring'i."""

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
    full_cohort_ids: set[str],
    covered_id_field: str,
    partial_run: bool,
    offset: int,
    limit: int | None,
) -> Path:
    """A1.5 (2026-08-14, GENİŞLETİLMİŞ şema) -- bkz. `run_pyradiomics_tcga.py`
    'deki AYNI fonksiyonun docstring'i, tam gerekçe orada tekrar yazılmadı."""

    actual_version = _normalized_pyradiomics_version()
    if actual_version != EXPECTED_PYRADIOMICS_VERSION:
        raise RuntimeError(
            f"pyradiomics sürüm uyuşmazlığı: kurulu={actual_version!r}, "
            f"beklenen={EXPECTED_PYRADIOMICS_VERSION!r} -- provenance "
            "manifest'i YAZILMADI (pyradiomics==3.0.1 pinli)."
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


def _fetch_lumiere_t1ce_scans(connection, modality: str) -> list[dict]:
    cursor = connection.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(
            """
            SELECT ms.scan_id, ms.patient_id, ms.file_path, ms.timepoint_label
            FROM mr_scans ms
            JOIN patients p ON p.patient_id = ms.patient_id
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE ds.source_name = 'LUMIERE' AND ms.modality = %s
            ORDER BY ms.patient_id, ms.scan_id
            """,
            (modality,),
        )
        return list(cursor.fetchall())
    finally:
        cursor.close()


def _fetch_existing_volumes(connection) -> dict[tuple[int, str], float]:
    """Mevcut DeepBraTumIA precomputed hacimlerini (aynı scan_id/region) önceden yükle."""

    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT r.scan_id, r.tumor_region, r.tumor_volume_mm3
            FROM radiomics r
            JOIN mr_scans ms ON ms.scan_id = r.scan_id
            JOIN patients p ON p.patient_id = ms.patient_id
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE ds.source_name = 'LUMIERE' AND r.segmentation_tool = 'DeepBraTumIA'
            """
        )
        result: dict[tuple[int, str], float] = {}
        for scan_id, region, volume in cursor.fetchall():
            if volume is not None:
                result[(scan_id, region)] = float(volume)
        return result
    finally:
        cursor.close()


def _load_done_keys(report_path: Path) -> tuple[list[dict[str, object]], set[tuple[str, int, str]]]:
    rows: list[dict[str, object]] = []
    done: set[tuple[str, int, str]] = set()
    if not report_path.is_file():
        return rows, done
    with report_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
            if row.get("status") == "OK":
                done.add((row["patient_id"], int(row["scan_id"]), row["region"]))
    return rows, done


def _blank_row(
    *,
    patient_id: str,
    scan_id: int,
    timepoint_label: str,
    region: str,
    mask_source: str,
    status: str,
    warning: str = "",
    resampled: object = "",
) -> dict[str, object]:
    return {
        "patient_id": patient_id,
        "scan_id": scan_id,
        "timepoint_label": timepoint_label,
        "region": region,
        "mask_source": mask_source,
        "status": status[:250],
        "voxel_volume": "",
        "db_volume": "",
        "volume_match": "",
        "volume_diff_mm3": "",
        "warning": warning,
        "resampled": resampled,
        "shape_features_json": "",
        "first_order_features_json": "",
        "texture_features_json": "",
        "surface_area": "",
        "entropy": "",
        "contrast": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "LUMIERE DeepBraTumIA maskelerinden C32 (N4+T1ce-özel Z-score+binCount=32) "
            "PyRadiomics 107 özellik üret (yalnız CSV, DB'ye yazmaz)."
        )
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--resample-output-root",
        type=Path,
        default=DEFAULT_RESAMPLE_ROOT,
        help="resample_image_and_mask() çıktılarının yazılacağı kök klasör.",
    )
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
        help="Var olan --stats-path'i silip TAZE fit et.",
    )
    parser.add_argument("--modality", default="T1ce", help="Kanonik tek-modalite seçimi (varsayılan T1ce/CT1).")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "İşlenecek tarama listesinin BAŞINDAN kaç kaydın atlanacağı -- "
            "2026-08-13 B1 düzeltmesi: --limit tek başına her zaman listenin "
            "PREFIX'ini alıyordu, parçalı koşularda limit'i her seferinde "
            "ARTIRMAK gerekiyordu. --offset ile doğal chunking sağlanır "
            "(örn. --offset 300 --limit 300). N4 cache'i (H4) sayesinde "
            "zaten-işlenmiş kayıtlar üzerinden tekrar geçmek artık ucuz."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="--offset'ten sonraki ilk N taramayla sınırla.")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--max-volume-drift",
        type=float,
        default=0.10,
        help="resample_image_and_mask() için izin verilen azami hacim sapma oranı.",
    )
    parser.add_argument(
        "--expected-image-count",
        type=int,
        default=None,
        help=(
            "A1.1 (2026-08-14) image_count gate -- Pass 3'e geçmeden ÖNCE, "
            "stats dosyasındaki donmuş istatistiğin image_count'unun bu "
            "değere eşit olması ZORUNLU (LUMIERE için KESİNLEŞMİŞ değer: "
            f"{DEFAULT_EXPECTED_IMAGE_COUNT}, 2026-08-14 tam-kohort Pass-1-only "
            "preflight koşusuyla ÖLÇÜLDÜ -- bkz. script docstring'i/"
            "DEFAULT_EXPECTED_IMAGE_COUNT gerekçesi). Eşit değilse koşu "
            "REDDEDİLİR (sessiz devam YOK). Belirtilmezse ve "
            "--skip-image-count-gate de verilmezse script Pass 3'e HİÇ "
            "geçmez (exit 2)."
        ),
    )
    parser.add_argument(
        "--skip-image-count-gate",
        action="store_true",
        help="A1.1 gate'ini BİLİNÇLİ olarak atla (yalnız küçük smoke test/keşif koşuları için).",
    )
    parser.add_argument(
        "--pass1-only",
        action="store_true",
        help=(
            "GÖREV 2 (2026-08-14) PREFLIGHT modu -- yalnız Pass 1 (hazır "
            "maske çözümle + HD-GLIO fallback'i ele + geometri-kurtarma + N4) "
            "çalışır. Pass 2 (Z-score fit) ve Pass 3 (PyRadiomics extraction) "
            "HİÇ ÇALIŞTIRILMAZ, DB'ye HİÇBİR YAZMA YAPILMAZ (bu script zaten "
            "CSV-only). Amaç: gerçek N4-üretilebilir tarama sayısını, TAM "
            "koşunun kullanacağı AYNI DB sorgusu + NAS çözümleme + HD-GLIO "
            "eleme kurallarıyla ölçmek (bkz. A1.1 image_count gate) -- "
            "`DEFAULT_EXPECTED_IMAGE_COUNT=580` bu bayrakla 2026-08-14'te "
            "TAM kohortta ÖLÇÜLÜP dondurulmuştur; NAS/DB verisi değişirse "
            "(örn. eksik DeepBraTumIA maskeleri sonradan eklenirse) bu "
            "bayrak yeniden çalıştırılıp sabit ELLE güncellenmelidir. Sonuç "
            "`<rapor>.pass1_preflight.json`'a yazılır (kategori dökümü + "
            "N4'e ulaşan sayı). N4 çıktıları normal --processed-root'a "
            "yazılır -- bu koşu BOŞA GİTMEZ, tam kohort koşusunun Aşama 1'i "
            "olarak cache'den kullanılır (A1.2 N4 cache)."
        ),
    )
    args = parser.parse_args()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    os.environ["GBMAID_RESAMPLED_ROOT"] = str(args.resample_output_root)
    os.environ["GBMAID_PROCESSED_ROOT"] = str(args.processed_root)
    # KRİTİK (smoke testte TCGA scriptinde bulundu, burada da geçerli):
    # apply_zscore_normalization() stats dosyasını YALNIZ bu ortam
    # değişkeninden okur (fit_source_zscore_statistics()'e verilen
    # stats_path= parametresi apply() tarafını ETKİLEMEZ).
    os.environ["GBMAID_ZSCORE_STATS_PATH"] = str(args.stats_path)

    if args.fresh_fit and args.stats_path.is_file():
        args.stats_path.unlink()

    rows_out, already_done = _load_done_keys(args.report)
    if already_done:
        print(f"Idempotent devam: {len(already_done)} (hasta,scan,bölge) zaten OK, atlanacak.", flush=True)

    extractor = _make_extractor()
    connection = get_connection(readonly=True)
    try:
        scans = _fetch_lumiere_t1ce_scans(connection, args.modality)
        existing_volumes = _fetch_existing_volumes(connection)
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

    # ---- Pass 1: hazır maske çöz + geometri-kurtarma + N4 ----
    resolved_scans: list[dict[str, object]] = []
    n4_paths: list[str] = []
    processed_pass1 = 0
    resampled_recovered_count = 0
    # GÖREV 2 (2026-08-14) PREFLIGHT -- her taramanın Pass 1'de HANGİ
    # kategoriye düştüğünü (scan-seviyesinde, TEK sayım) izler. `--pass1-
    # only` modunda kategori dökümü olarak raporlanır; normal koşularda da
    # (ek maliyeti sıfıra yakın) hesaplanır, `--pass1-only` DIŞINDA
    # kullanılmaz ama gelecekte teşhis için faydalı olabilir.
    pass1_outcomes: Counter[str] = Counter()

    for scan in scans:
        patient_id = scan["patient_id"]
        scan_id = scan["scan_id"]
        timepoint_label = scan["timepoint_label"]
        processed_pass1 += 1

        try:
            image_path = resolve_nas_path(scan["file_path"])
        except FileNotFoundError as exc:
            rows_out.append(
                _blank_row(
                    patient_id=patient_id,
                    scan_id=scan_id,
                    timepoint_label=timepoint_label,
                    region="-",
                    mask_source="",
                    status=f"GORUNTU_YOK:{exc}",
                )
            )
            pass1_outcomes["GORUNTU_YOK"] += 1
            continue

        try:
            resolved = resolve_ready_mask(
                source="LUMIERE",
                image_path=image_path,
                patient_id=patient_id,
                scan_id=scan_id,
                raise_on_geometry_mismatch=False,
            )
        except (ReadyMaskNotFoundError, ValueError) as exc:
            rows_out.append(
                _blank_row(
                    patient_id=patient_id,
                    scan_id=scan_id,
                    timepoint_label=timepoint_label,
                    region="-",
                    mask_source="",
                    status=f"MASKE_YOK_VEYA_GEOMETRI:{type(exc).__name__}:{exc}",
                )
            )
            pass1_outcomes["MASKE_YOK_VEYA_GEOMETRI:maske_bulunamadi"] += 1
            continue

        mask_source = resolved["mask_source"]
        if mask_source != CANONICAL_MASK_SOURCE:
            # HD-GLIO-AUTO fallback -- ana kanonik vektöre karıştırılmaz.
            rows_out.append(
                _blank_row(
                    patient_id=patient_id,
                    scan_id=scan_id,
                    timepoint_label=timepoint_label,
                    region="-",
                    mask_source=mask_source,
                    status="hd_glio_fallback_excluded",
                    warning="; ".join(resolved["warnings"]),
                )
            )
            pass1_outcomes["hd_glio_fallback_excluded"] += 1
            continue

        resolved_image_path = Path(resolved["image_path"])
        mask_path = Path(resolved["mask_path"])
        mask_labels = set(resolved["geometry"]["mask"]["labels"])
        scan_resampled = False
        scan_warnings = list(resolved["warnings"])

        if not resolved["geometry"]["geometry_match"]:
            try:
                resample_result = resample_image_and_mask(
                    resolved_image_path, mask_path, max_volume_drift=args.max_volume_drift
                )
            except Exception as exc:  # noqa: BLE001
                rows_out.append(
                    _blank_row(
                        patient_id=patient_id,
                        scan_id=scan_id,
                        timepoint_label=timepoint_label,
                        region="-",
                        mask_source=mask_source,
                        status=f"MASKE_YOK_VEYA_GEOMETRI:resample_denendi_basarisiz:{type(exc).__name__}:{exc}",
                        warning="; ".join(scan_warnings),
                    )
                )
                pass1_outcomes["MASKE_YOK_VEYA_GEOMETRI:resample_denendi_basarisiz"] += 1
                continue

            if resample_result["status"] != "PASS":
                detail = (
                    f"output_geometry_match={resample_result['output_geometry_match']} "
                    f"labels_preserved={resample_result['labels_preserved']} "
                    f"volume_drift={resample_result['volume_drift_fraction']:.4f}"
                )
                rows_out.append(
                    _blank_row(
                        patient_id=patient_id,
                        scan_id=scan_id,
                        timepoint_label=timepoint_label,
                        region="-",
                        mask_source=mask_source,
                        status="MASKE_YOK_VEYA_GEOMETRI:resample_denendi_basarisiz",
                        warning=f"{'; '.join(scan_warnings)}; {detail}",
                    )
                )
                pass1_outcomes["MASKE_YOK_VEYA_GEOMETRI:resample_denendi_basarisiz"] += 1
                continue

            resolved_image_path = Path(resample_result["image_output_path"])
            mask_path = Path(resample_result["mask_output_path"])
            mask_labels = {float(v) for v in resample_result["output_mask"]["labels"]}
            scan_resampled = True
            resampled_recovered_count += 1
            scan_warnings.append(
                "geometri uyuşmazlığı nedeniyle 1mm ortak ızgaraya resample edildi "
                f"(volume_drift={resample_result['volume_drift_fraction']:.4f})"
            )

        label_map = REGION_LABELS_BY_MASK_SOURCE[mask_source]

        # A2 (2026-08-14) -- native Necrosis/Contrast-enhancing/Edema'ya ek
        # olarak türetilmiş WT_derived/TC_derived de pending_regions'a
        # eklenir. Bu noktada `mask_source` GARANTİ OLARAK
        # `CANONICAL_MASK_SOURCE` ("lumiere_deepbratumia_native") -- yukarıda
        # (satır ~700) HD-GLIO fallback zaten `continue` ile elendi. Yine de
        # sessiz varsayım yerine AÇIKÇA assert edilir (mimari kural,
        # 2026-08-14 kararı: WT/TC türetmesi yalnız izinli üç mask_source
        # için tanımlı).
        if mask_source != CANONICAL_MASK_SOURCE:
            raise AssertionError(
                f"Beklenmedik mask_source WT/TC türetme noktasına ulaştı: "
                f"{mask_source!r} -- HD-GLIO fallback guard'ı yukarıda zaten "
                "elemiş olmalıydı (2026-08-14 kararı)."
            )
        derived_pending = {
            region: None
            for region in ("WT_derived", "TC_derived")
            if (patient_id, scan_id, region) not in already_done
        }

        pending_regions = {
            region: label_value
            for region, label_value in label_map.items()
            if (patient_id, scan_id, region) not in already_done
        }
        pending_regions.update(derived_pending)
        if not pending_regions:
            pass1_outcomes["already_done_all_regions"] += 1
            continue

        label_yok_regions = [
            r for r, lv in pending_regions.items() if lv is not None and float(lv) not in mask_labels
        ]
        if derived_pending:
            derived_masks_pass1 = build_derived_region_masks(mask_path, mask_source)
            for region in derived_pending:
                if not derived_region_has_voxels(derived_masks_pass1[region]):
                    label_yok_regions.append(region)
        for region in label_yok_regions:
            rows_out.append(
                _blank_row(
                    patient_id=patient_id,
                    scan_id=scan_id,
                    timepoint_label=timepoint_label,
                    region=region,
                    mask_source=mask_source,
                    status="LABEL_YOK_0_VOXEL",
                    warning="; ".join(scan_warnings),
                    resampled=scan_resampled,
                )
            )
        remaining_regions = {r: lv for r, lv in pending_regions.items() if r not in label_yok_regions}
        if not remaining_regions:
            pass1_outcomes["label_yok_all_regions"] += 1
            continue

        try:
            n4_path = _apply_n4_cached(str(resolved_image_path))
        except Exception as exc:  # noqa: BLE001
            for region in remaining_regions:
                rows_out.append(
                    _blank_row(
                        patient_id=patient_id,
                        scan_id=scan_id,
                        timepoint_label=timepoint_label,
                        region=region,
                        mask_source=mask_source,
                        status=f"HATA:N4:{type(exc).__name__}:{exc}",
                        warning="; ".join(scan_warnings),
                        resampled=scan_resampled,
                    )
                )
            pass1_outcomes["HATA:N4"] += 1
            continue

        n4_paths.append(n4_path)
        resolved_scans.append(
            {
                "patient_id": patient_id,
                "scan_id": scan_id,
                "timepoint_label": timepoint_label,
                "mask_path": mask_path,
                "mask_source": mask_source,
                "n4_path": n4_path,
                "regions": remaining_regions,
                "scan_resampled": scan_resampled,
                "warnings": scan_warnings,
            }
        )
        pass1_outcomes["N4_REACHED"] += 1

        if processed_pass1 % args.progress_every == 0:
            elapsed = time.time() - t_start
            print(
                f"[Pass1 N4] [{processed_pass1}/{len(scans)}] tarama işlendi, "
                f"gecen={elapsed:.0f}s",
                flush=True,
            )

    if args.pass1_only:
        # GÖREV 2 (2026-08-14) PREFLIGHT -- Pass 2 (Z-score fit) VE Pass 3
        # (PyRadiomics extraction) HİÇ ÇALIŞTIRILMAZ, DB'ye YAZILMAZ (bu
        # script zaten CSV-only). N4 çıktıları normal `--processed-root`'a
        # yazıldı (A1.2 cache'li) -- bu koşu boşa gitmiyor, tam kohort
        # koşusunun Aşama 1'i olarak cache'den kullanılacak.
        elapsed = time.time() - t_start
        summary = {
            "cohort": CANONICAL_SOURCE,
            "mode": "pass1_only_preflight",
            "total_scans_considered": len(scans),
            "n4_reached_count": len(resolved_scans),
            "category_breakdown": dict(pass1_outcomes),
            "resampled_recovered_count": resampled_recovered_count,
            "offset": args.offset,
            "limit": args.limit,
            "elapsed_seconds": round(elapsed, 1),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        summary_path = args.report.with_name(args.report.name + ".pass1_preflight.json")
        temporary = summary_path.with_name(f".{summary_path.name}.{uuid.uuid4().hex}")
        try:
            temporary.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            os.replace(temporary, summary_path)
        finally:
            temporary.unlink(missing_ok=True)

        # Pass 1'in kendi HATA/durum satırları da (rapor tutarlılığı için)
        # yazılır -- ama bu bir CSV "sonuç" raporu DEĞİL, yalnız Pass 1
        # sırasında biriken hata/atlama satırlarının dökümü.
        with args.report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows_out)

        print(
            f"\nPASS1-ONLY PREFLIGHT bitti (gecen={elapsed:.0f}s). "
            f"Toplam tarama={len(scans)}, N4'e ULAŞAN={len(resolved_scans)}, "
            f"kategori dökümü={dict(pass1_outcomes)}.\n"
            f"Özet: {summary_path}\n"
            f"(Pass 1 hata/atlama satırları -- sonuç DEĞİL): {args.report}",
            flush=True,
        )
        return 0

    # ---- Pass 2: TEK T1ce-özel, kaynak-bazlı (LUMIERE) Z-score fit ----
    # 2026-08-13 B3 düzeltmesi (review-gate BLOKE bulgusu) -- bkz.
    # `run_pyradiomics_tcga.py`'nin AYNI bloğundaki tam gerekçe. Özet:
    # koşulsuz refit, kısmi-restart senaryosunda kaynağın tek kanonik
    # istatistiğini partiye göre SESSİZCE değiştirirdi. Artık zaten geçerli
    # bir kayıt varsa (force=True/--fresh-fit verilmedikçe) REUSE edilir.
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
            "istatistik var, refit ATLANDI (B3 guard). Zorla yeniden fit için "
            "--fresh-fit kullanın.",
            flush=True,
        )

    # ---- A1.1 (2026-08-14) image_count gate -- Pass 3 başlamadan ZORUNLU ----
    # Bkz. `run_pyradiomics_tcga.py`'deki AYNI gate'in docstring'i, tam
    # gerekçe orada tekrar yazılmadı.
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
                "sayısını AÇIKÇA belirtin (LUMIERE için KESİNLEŞMİŞ değer: "
                f"{DEFAULT_EXPECTED_IMAGE_COUNT}, 2026-08-14 tam-kohort preflight "
                "ile ÖLÇÜLDÜ -- bkz. script docstring'i) veya bilinçli bir smoke "
                "test ise --skip-image-count-gate kullanın.",
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

    # ---- Pass 3: Z-score normalize + C32 PyRadiomics (ikinci-seviye resample-kurtarma dahil) ----
    ok_count = fail_count = 0
    processed_pass3 = 0

    def flush() -> None:
        with args.report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows_out)

    for scan in resolved_scans:
        patient_id = scan["patient_id"]
        scan_id = scan["scan_id"]
        timepoint_label = scan["timepoint_label"]
        mask_path = scan["mask_path"]
        mask_source = scan["mask_source"]
        scan_resampled = scan["scan_resampled"]
        scan_warnings = list(scan["warnings"])
        processed_pass3 += 1

        try:
            zscore_path = apply_zscore_normalization(scan["n4_path"], CANONICAL_SOURCE)
        except Exception as exc:  # noqa: BLE001
            for region in scan["regions"]:
                rows_out.append(
                    _blank_row(
                        patient_id=patient_id,
                        scan_id=scan_id,
                        timepoint_label=timepoint_label,
                        region=region,
                        mask_source=mask_source,
                        status=f"HATA:ZSCORE:{type(exc).__name__}:{exc}",
                        warning="; ".join(scan_warnings),
                        resampled=scan_resampled,
                    )
                )
                fail_count += 1
            continue

        for region, label_value in scan["regions"].items():
            try:
                if label_value is None:
                    # A2 (2026-08-14) -- WT_derived/TC_derived: native bir
                    # etiket DEĞİL, bellekte üretilmiş ikili maske (label=1).
                    derived_masks = build_derived_region_masks(mask_path, mask_source)
                    result = extractor.execute(
                        str(zscore_path), derived_masks[region], label=1
                    )
                else:
                    result = extractor.execute(str(zscore_path), str(mask_path), label=label_value)
            except Exception as exc:  # noqa: BLE001
                if scan_resampled or "geometry mismatch" not in str(exc).lower():
                    rows_out.append(
                        _blank_row(
                            patient_id=patient_id,
                            scan_id=scan_id,
                            timepoint_label=timepoint_label,
                            region=region,
                            mask_source=mask_source,
                            status=f"HATA:{type(exc).__name__}:{exc}",
                            warning="; ".join(scan_warnings),
                            resampled=scan_resampled,
                        )
                    )
                    fail_count += 1
                    continue

                # İkinci-seviye kurtarma: PyRadiomics'in kendi dahili
                # tolerance kontrolü Z-score görüntüsünü reddetti --
                # AYNI resample fonksiyonunu Z-score+maske üzerinde BİR
                # KEZ daha dene (salt geometrik dönüşüm, Z-skorlanmış
                # girdide de doğru çalışır).
                try:
                    resample_result = resample_image_and_mask(
                        Path(zscore_path), mask_path, max_volume_drift=args.max_volume_drift
                    )
                except Exception as exc2:  # noqa: BLE001
                    rows_out.append(
                        _blank_row(
                            patient_id=patient_id,
                            scan_id=scan_id,
                            timepoint_label=timepoint_label,
                            region=region,
                            mask_source=mask_source,
                            status=(
                                f"HATA:{type(exc).__name__}:{exc} | "
                                f"resample_denendi_basarisiz:{type(exc2).__name__}:{exc2}"
                            ),
                            warning="; ".join(scan_warnings),
                            resampled=False,
                        )
                    )
                    fail_count += 1
                    continue

                if resample_result["status"] != "PASS":
                    detail = (
                        f"output_geometry_match={resample_result['output_geometry_match']} "
                        f"labels_preserved={resample_result['labels_preserved']} "
                        f"volume_drift={resample_result['volume_drift_fraction']:.4f}"
                    )
                    rows_out.append(
                        _blank_row(
                            patient_id=patient_id,
                            scan_id=scan_id,
                            timepoint_label=timepoint_label,
                            region=region,
                            mask_source=mask_source,
                            status="HATA:resample_denendi_basarisiz",
                            warning=f"{'; '.join(scan_warnings)}; {detail}",
                            resampled=False,
                        )
                    )
                    fail_count += 1
                    continue

                retry_zscore_path = Path(resample_result["image_output_path"])
                retry_mask_path = Path(resample_result["mask_output_path"])
                retry_mask_labels = {float(v) for v in resample_result["output_mask"]["labels"]}
                scan_resampled = True
                resampled_recovered_count += 1
                scan_warnings.append(
                    "PyRadiomics dahili tolerance hatası sonrası 1mm ortak ızgaraya "
                    f"resample edildi (volume_drift={resample_result['volume_drift_fraction']:.4f})"
                )

                # HIGH düzeltmesi (2026-08-14, Codex review A3 -- TUR sonucu):
                # ÖNCEDEN `retry_zscore_path`/`retry_mask_path` yalnız BU
                # bölgenin extraction çağrısında kullanılıyordu -- döngünün
                # üst kapsamındaki `zscore_path`/`mask_path` (bir sonraki
                # `region` iterasyonunun İLK denemesinin kullandığı
                # değişkenler) GÜNCELLENMİYORDU. Sonuç: aynı taramanın
                # SONRAKİ bölgeleri yine ESKİ (geometri-uyumsuz) yollarla
                # deneniyordu VE `scan_resampled` artık `True` olduğu için
                # (yukarıdaki `if scan_resampled or ...:` koşulu) yeniden
                # kurtarma denenmeden doğrudan `HATA` satırına düşüyordu --
                # geometri-kurtarmasına ihtiyaç duyan bir vizitte 5 bölgeden
                # yalnız döngüde İLK hata veren bölge kurtuluyordu,
                # WT_derived/TC_derived dahil kalanlar eksik kalıyordu.
                # Düzeltme: retry yolları TARAMA SEVİYESİNE yükseltiliyor --
                # kalan TÜM bölgeler (native + türetilmiş) bu düzeltilmiş
                # geometriyle doğrudan İLK denemede çalışır, aynı resample
                # işi bölge başına TEKRARLANMAZ.
                zscore_path = str(retry_zscore_path)
                mask_path = retry_mask_path

                if label_value is None:
                    # A2 (2026-08-14) -- ikinci-seviye kurtarma sonrası
                    # türetilmiş bölge: NATIVE `retry_mask_path`'ten (henüz
                    # etiketleri KORUNMUŞ çok-etiketli maske) YENİDEN inşa
                    # edilir -- `retry_mask_labels` (native etiket kümesi)
                    # burada DOĞRUDAN kullanılamaz, çünkü türetilmiş bölge
                    # tek bir etikete değil bileşen BİRLEŞİMİNE karşılık
                    # gelir.
                    retry_derived_masks = build_derived_region_masks(retry_mask_path, mask_source)
                    if not derived_region_has_voxels(retry_derived_masks[region]):
                        rows_out.append(
                            _blank_row(
                                patient_id=patient_id,
                                scan_id=scan_id,
                                timepoint_label=timepoint_label,
                                region=region,
                                mask_source=mask_source,
                                status="LABEL_YOK_0_VOXEL",
                                warning="; ".join(scan_warnings),
                                resampled=scan_resampled,
                            )
                        )
                        fail_count += 1
                        continue
                    try:
                        result = extractor.execute(
                            str(retry_zscore_path), retry_derived_masks[region], label=1
                        )
                    except Exception as exc3:  # noqa: BLE001
                        rows_out.append(
                            _blank_row(
                                patient_id=patient_id,
                                scan_id=scan_id,
                                timepoint_label=timepoint_label,
                                region=region,
                                mask_source=mask_source,
                                status=f"HATA:resample_sonrasi_hala_basarisiz:{type(exc3).__name__}:{exc3}",
                                warning="; ".join(scan_warnings),
                                resampled=scan_resampled,
                            )
                        )
                        fail_count += 1
                        continue
                    # `result` başarıyla üretildi -- aşağıdaki ortak başarı
                    # yoluna düşer (bu iç `if` bloğu erken `continue`
                    # ETMEDİĞİ için native kolun geri kalanı ATLANIR).
                elif float(label_value) not in retry_mask_labels:
                    rows_out.append(
                        _blank_row(
                            patient_id=patient_id,
                            scan_id=scan_id,
                            timepoint_label=timepoint_label,
                            region=region,
                            mask_source=mask_source,
                            status="LABEL_YOK_0_VOXEL",
                            warning="; ".join(scan_warnings),
                            resampled=scan_resampled,
                        )
                    )
                    fail_count += 1
                    continue
                else:
                    try:
                        result = extractor.execute(
                            str(retry_zscore_path), str(retry_mask_path), label=label_value
                        )
                    except Exception as exc3:  # noqa: BLE001
                        rows_out.append(
                            _blank_row(
                                patient_id=patient_id,
                                scan_id=scan_id,
                                timepoint_label=timepoint_label,
                                region=region,
                                mask_source=mask_source,
                                status=f"HATA:resample_sonrasi_hala_basarisiz:{type(exc3).__name__}:{exc3}",
                                warning="; ".join(scan_warnings),
                                resampled=scan_resampled,
                            )
                        )
                        fail_count += 1
                        continue

            # Ortak başarı yolu -- ilk denemede ya da resample-sonrası
            # retry'de başarılı olan `result` buraya düşer.
            shape, first_order, texture = _split_features(result)
            voxel_volume = shape.get("original_shape_VoxelVolume")
            db_volume = existing_volumes.get((scan_id, region))
            volume_match = ""
            volume_diff = ""
            if voxel_volume is not None and db_volume is not None:
                diff = abs(float(voxel_volume) - float(db_volume))
                volume_diff = f"{diff:.4f}"
                volume_match = diff < 1.0

            rows_out.append(
                {
                    "patient_id": patient_id,
                    "scan_id": scan_id,
                    "timepoint_label": timepoint_label,
                    "region": region,
                    "mask_source": mask_source,
                    "status": "OK",
                    "voxel_volume": voxel_volume,
                    "db_volume": db_volume if db_volume is not None else "",
                    "volume_match": volume_match,
                    "volume_diff_mm3": volume_diff,
                    "warning": "; ".join(scan_warnings),
                    "resampled": scan_resampled,
                    "shape_features_json": json.dumps(shape),
                    "first_order_features_json": json.dumps(first_order),
                    "texture_features_json": json.dumps(texture),
                    "surface_area": shape.get("original_shape_SurfaceArea", ""),
                    "entropy": first_order.get("original_firstorder_Entropy", ""),
                    "contrast": texture.get("original_glcm_Contrast", ""),
                }
            )
            ok_count += 1

        if processed_pass3 % args.progress_every == 0:
            flush()
            elapsed = time.time() - t_start
            print(
                f"[Pass3 C32] [{processed_pass3}/{len(resolved_scans)}] tarama işlendi, "
                f"ok={ok_count} fail={fail_count}, gecen={elapsed:.0f}s, rapor guncellendi: {args.report}",
                flush=True,
            )

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
        f"\nBitti. {len(resolved_scans)} tarama C32 için işlendi (Pass1'de {processed_pass1} tarama "
        f"gözden geçirildi). ok={ok_count} fail={fail_count} "
        f"resample_ile_kurtarilan={resampled_recovered_count}. Rapor: {args.report}\n"
        f"Provenance manifest: {manifest_path}\n"
        f"NOT: DB'ye yazım için önerilen segmentation_tool adı: "
        f"{SEGMENTATION_TOOL_C32_FOR_DB_WRITE!r} (write_lumiere_pyradiomics_to_db.py "
        f"güncellemesi bu görevin kapsamı dışında, ayrı takip)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
