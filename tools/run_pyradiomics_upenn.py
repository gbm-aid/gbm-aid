"""UPenn hazır expert/automated maskeleriyle PyRadiomics 107 özellik çıkarımı (C32).

KARAR (2026-08-13, bkz. decisions/2026-08-13-pyradiomics-c32-bincount-
karari.md): bu script artık **C32** sözleşmesiyle çalışır -- N4 bias-field
düzeltmesi + T1ce-özel (bu script zaten yalnız T1ce işler, modalite
filtresi doğal) kaynak-bazlı Z-score + `binCount=32` (binWidth=25 DEĞİL).
Eski davranış (ham NAS görüntüsü + binWidth=25 varsayılanı) `upenn_
pyradiomics_applied.csv`'de KORUNDU, bu script onu ÜZERİNE YAZMAZ --
varsayılan çıktı ayrı bir dosyaya (`upenn_pyradiomics_c32.csv`) yazılır.

Neden bu değişiklik gerekti (2026-08-13 bulgu zinciri, özet -- tam detay
karar dosyasında): Reviewer bulgusu (bu script N4+Z-score çıktısını hiç
kullanmıyordu) + Barış'ın 15-taramalık ölçümü (binWidth=25'in Z-skorlanmış
aralıkta yalnız ~2 gri-seviye bin üretmesi) + ayırt edici perturbasyon
testi (texture ICC'si C32'de ≈0,97, A/B'de ≈0,33) -- bkz.
`tools/run_pyradiomics_tcga.py` docstring'i (aynı gerekçe, üç script'te
tekrar).

Karar (bkz. decisions/2026-08-07-upenn-144-vs-107-uyumsuzlugu.md): UPenn'in
mevcut 144-özellik CaPTk verisi kanonik 107 ile isim eşlemesiyle çözülemeyecek
kadar farklı (farklı motor/parametre) -- bu yüzden UPenn'de de PyRadiomics
BAŞTAN çalıştırılıyor. Mevcut CaPTk verisi silinmez/değişmez.

Bu script yalnız Python 3.10 ortamında (pyradiomics 3.0.1,
`requirements-pyradiomics.txt`) çalışır -- proje ana ortamında (README.md:
3.10.18) pyradiomics MSVC Build Tools derleme adımı gerektirdiği için ayrı
bir venv (`.venv310_pyradiomics`) kullanılır.

PyRadiomics ayarları extractor çağrısında AÇIKÇA veriliyor (binCount=32,
normalize=False, imageTypes={'Original':{}}) -- 2026-08-13 Codex bulgusu:
boş `RadiomicsFeatureExtractor()` çağrısında `binWidth` extractor'ın
`settings` sözlüğünde HİÇ GÖRÜNMÜYORDU, sürüm değişince sessizce
değişebilirdi. shape+firstorder+glcm+glrlm+glszm+ngtdm+gldm = 14+18+75=107.

Hasta seçimi: `decisions/2026-08-06-upenn-dedup-11-21.md` kararınca ana kayıt
`_11` (pre-treatment) T1ce'dir. 630 UPenn hastasının 611'inde pre-treatment
T1ce var; kalan 19 hastada YALNIZCA post-op (`_21`) T1ce bulunuyor (NAS/DB'de
pre-treatment kaydı hiç yok). Bu 19 hasta için post-op'a AÇIKÇA ETİKETLENMİŞ
("timepoint_used=post-op_fallback_no_pretreatment") biçimde düşülür --
sessiz fallback değildir, CSV'de görünür.

İşlem sırası (mimari kural, `pipeline/harmonization.py` ile tutarlı):
  Pass 1: her hasta için görüntü çözümle, hazır maske çöz (expert/automated),
          geometri doğrula, N4 bias-field düzeltmesi uygula.
  Pass 2: Pass 1'in TÜM N4 çıktılarından TEK bir T1ce-özel, kaynak-bazlı
          (yalnız UPenn) Z-score istatistiği fit edilir (`--stats-path`'e
          yazılır -- üretim `artifacts/week2/zscore/source_stats.json`'a
          DOKUNULMAZ, ayrı/versiyonlu dosya, TCGA/LUMIERE'in kendi C32
          fit'leriyle AYNI dosyada (kaynak-anahtarlı) paylaşılabilir).
  Pass 3: her N4 çıktısı Pass 2'nin istatistiğiyle Z-score normalize
          edilir, PyRadiomics C32 ile çalıştırılır, sonuç CSV'ye yazılır.

DB'YE HİÇBİR YAZMA YAPILMAZ (yalnız SELECT, salt-okunur bağlantı) -- bu
script öncekiyle AYNI, yalnız CSV üretir. Supabase'e uygulama db-agent'ın
ayrı dry-run/apply protokolüyle yapılacaktır. Bu turda `tools/
write_upenn_pyradiomics_to_db.py`'nin `SEGMENTATION_TOOL` sabiti C32 için
GÜNCELLENMEDİ (bu script'in kapsamı dışında, ayrı takip görevi olarak
kayıtlı) -- önerilen ad `UPenn-PyRadiomics-107-C32` (bkz. karar dosyası).

Var olan `tumor_volume_mm3` (CaPTk-automatic/CaPTk-corrected) PyRadiomics'in
kendi hesapladığı VoxelVolume ile çapraz karşılaştırılır (yalnız bilgi
amaçlı -- geometri Z-score'dan etkilenmediği için hâlâ anlamlı).

KISIT (2026-08-13 görev talimatı): bu turda TAM KOHORT YENİDEN ÇIKARIMI
YAPILMAZ -- kod hazırlanmıştır ama gerçek bir koşu bu oturumda
ÇALIŞTIRILMADI (yalnız küçük `--limit` smoke testi).
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
# NOT (2026-08-14, B1 sırasında TCGA eşleniğinde bulunan bug -- bkz. o
# dosyadaki AYNI yorum, tam gerekçe orada): `.resolve()` DEĞİL
# `.absolute()` -- `subst X:` ASCII-safe sürücüsünü gerçek Türkçe
# karakterli yola geri düşürüp SimpleITK NIfTI yazım hatasına yol
# açıyordu.
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
from pipeline.radiomics_volume import (  # noqa: E402
    REGION_LABELS_BY_MASK_SOURCE,
    WT_TC_DERIVATION_ALLOWED_MASK_SOURCES,
    build_derived_region_masks,
    derived_region_has_voxels,
)
from pipeline.resampling import validate_image_mask_geometry  # noqa: E402
from pipeline.segmentation import (  # noqa: E402
    ImageMaskGeometryError,
    ReadyMaskNotFoundError,
    _upenn_mask,
    resolve_nas_path,
)

CANONICAL_SOURCE = "UPenn"
MASK_SOURCE_TO_TOOL = {
    "upenn_expert": "CaPTk-corrected",
    "upenn_automated_approx": "CaPTk-automatic",
}
# Bu script kendisi DB'ye yazmaz (yalnız CSV) -- ama `tools/
# write_upenn_pyradiomics_to_db.py`'nin (ayrı takip görevi) hangi
# `segmentation_tool` adını kullanması gerektiğini burada belgeliyoruz.
SEGMENTATION_TOOL_C32_FOR_DB_WRITE = "UPenn-PyRadiomics-107-C32"

OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "week3" / "pyradiomics"
DEFAULT_PROCESSED_ROOT = OUTPUT_ROOT / "c32_n4_zscore"
# 2026-08-13 H5 düzeltmesi (review-gate BLOKE bulgusu): ÖNCEDEN TCGA/UPenn/
# LUMIERE scriptleri AYNI `source_stats_t1ce_c32.json` dosyasını paylaşıyordu
# (kilitsiz oku-tam-dosya-yaz deseniyle) -- artık her script kendi kaynağına
# özel AYRI bir dosya kullanıyor, kaynak-bazlı paralel koşu güvenli hale
# geldi. `--stats-path` ile isteğe bağlı paylaşımlı bir dosyaya dönülebilir.
DEFAULT_ZSCORE_STATS_PATH = OUTPUT_ROOT / "source_stats_t1ce_c32_upenn.json"
DEFAULT_REPORT = OUTPUT_ROOT / "upenn_pyradiomics_c32.csv"

# A1.1 (2026-08-14, Codex review TUR 2 -- "image_count gate"): UPenn'in
# beklenen tam kohort N4-üretilebilir hasta sayısı. 630 DB hastasının 19'u
# YALNIZ post-op (_21) T1ce'ye sahip ve bu 19'un HİÇBİRİ için images_segm/
# automated_segm altında hazır bir maske dosyası YOK (2026-08-14'te
# `artifacts/week3/pyradiomics/upenn_pyradiomics_applied.csv`'nin eski
# (A-yöntemi) koşusundan CANLI DOĞRULANDI -- 19/19 post-op hasta
# `MASKE_YOK:ReadyMaskNotFoundError` ile başarısız, `_upenn_mask()`
# `scan_key = image_path.parent.name` üzerinden ARAR, `_21` klasörü için
# hiçbir zaman bir segm dosyası bulunamıyor -- YAPISAL/KALICI bir veri
# eksikliği, geçici bir NAS/DB sorunu DEĞİL). Bu 19 hasta `_upenn_mask()`
# `ReadyMaskNotFoundError` fırlattığı için N4'e HİÇ ULAŞMIYOR (Pass 1'de
# MASKE_YOK satırıyla `continue` ediliyor) -- yani bu script'in Pass 1'i
# N4'ü YALNIZ 611 hasta için üretebilir, 630 için DEĞİL. YÜKSEK
# GÜVENİLİRLİK (gerçek veriyle doğrulandı) -- yine de operatör
# `--expected-image-count` ile override edebilir.
DEFAULT_EXPECTED_IMAGE_COUNT = 611

# A1.5 (2026-08-14): provenance manifest sabitleri.
EXPECTED_PYRADIOMICS_VERSION = "3.0.1"
ZSCORE_SCOPE = "t1ce_source"

FIELDNAMES = [
    "patient_id",
    "scan_id",
    "timepoint_used",
    "region",
    "mask_source",
    "status",
    "voxel_volume",
    "db_volume",
    "db_tool",
    "volume_match",
    "volume_diff_mm3",
    "warning",
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
    tcga.py`'deki AYNI fonksiyonun docstring'i, tam gerekçe orada.
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
    çıktılarının YENİDEN hesaplanmasını önler.
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


def _fetch_upenn_patient_scans(connection) -> list[dict]:
    """630 UPenn hastasının her biri için tercih edilen T1ce taramasını döndür.

    Öncelik: pre-treatment (_11). Yoksa post-op (_21) -- açıkça etiketlenir.

    DETERMİNİZM (A1.3, 2026-08-14 -- Codex review TUR 2 bulgusu): `ORDER BY`
    eskiden yalnız `(patient_id, timepoint_önceliği)` içeriyordu. Bir hastanın
    AYNI öncelik grubunda birden çok T1ce taraması varsa `DISTINCT ON` hangisini
    tutacağını PostgreSQL'in fiziksel satır sırası belirlerdi -- yani seçim
    koşudan koşuya değişebilirdi. `--offset/--limit` ile PARÇALI koştuğumuz için
    bu, parçalar arasında tarama atlanmasına/tekrarlanmasına yol açabilirdi.
    `ms.scan_id` son kırıcı olarak eklendi; seçim artık tamamen deterministik.

    (Ölçüm 2026-08-14, canlı DB: 671 (hasta × öncelik-grubu) grubunun
    hepsinde tam 1 tarama var -- yani MEVCUT veride fiili bir tie YOK, bu
    düzeltme veri değiştiğinde ısırmasını önleyen bir garanti. Dönen satır
    sırası zaten `patient_id`'ye göre deterministikti, offset/limit dilimlemesi
    o yüzden bugün de tutarlıydı.)

    Dönen satır sırası: `patient_id` artan (offset/limit dilimlemesi stabil).
    """

    cursor = connection.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(
            """
            SELECT DISTINCT ON (p.patient_id)
                p.patient_id, ms.scan_id, ms.file_path, ms.timepoint_label
            FROM patients p
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            LEFT JOIN mr_scans ms
                ON ms.patient_id = p.patient_id AND ms.modality = 'T1ce'
            WHERE ds.source_name = 'UPenn-GBM'
            ORDER BY
                p.patient_id,
                CASE ms.timepoint_label
                    WHEN 'pre-treatment' THEN 0
                    WHEN 'post-op' THEN 1
                    ELSE 2
                END,
                ms.scan_id ASC NULLS LAST
            """
        )
        return list(cursor.fetchall())
    finally:
        cursor.close()


def _fetch_existing_volumes(connection) -> dict[tuple[int, str, str], float]:
    """UPenn T1ce scan_id'leri için mevcut CaPTk hacimlerini önceden yükle."""

    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT r.scan_id, r.segmentation_tool, r.tumor_region, r.tumor_volume_mm3
            FROM radiomics r
            JOIN mr_scans ms ON ms.scan_id = r.scan_id
            JOIN patients p ON p.patient_id = ms.patient_id
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE ds.source_name = 'UPenn-GBM' AND ms.modality = 'T1ce'
            """
        )
        result: dict[tuple[int, str, str], float] = {}
        for scan_id, tool, region, volume in cursor.fetchall():
            if volume is not None:
                result[(scan_id, tool, region)] = float(volume)
        return result
    finally:
        cursor.close()


def _load_done_keys(report_path: Path) -> tuple[list[dict[str, object]], set[tuple[str, str]]]:
    """Önceki koşudan başarıyla tamamlanmış (patient_id, region) anahtarlarını oku."""

    rows: list[dict[str, object]] = []
    done: set[tuple[str, str]] = set()
    if not report_path.is_file():
        return rows, done
    with report_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
            if row.get("status") == "OK":
                done.add((row["patient_id"], row["region"]))
    return rows, done


def _blank_row(*, patient_id, scan_id, timepoint_used, region, mask_source, status, warning="") -> dict[str, object]:
    return {
        "patient_id": patient_id,
        "scan_id": scan_id,
        "timepoint_used": timepoint_used,
        "region": region,
        "mask_source": mask_source,
        "status": status[:200] if isinstance(status, str) else status,
        "voxel_volume": "",
        "db_volume": "",
        "db_tool": "",
        "volume_match": "",
        "volume_diff_mm3": "",
        "warning": warning,
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
            "UPenn hazır maskelerinden C32 (N4+T1ce-özel Z-score+binCount=32) "
            "PyRadiomics 107 özellik üret (yalnız CSV, DB'ye yazmaz)."
        )
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "İşlenecek hasta listesinin BAŞINDAN kaç kaydın atlanacağı -- "
            "2026-08-13 B1 düzeltmesi: --limit tek başına her zaman listenin "
            "PREFIX'ini alıyordu, parçalı koşularda limit'i her seferinde "
            "ARTIRMAK gerekiyordu. --offset ile doğal chunking sağlanır "
            "(örn. --offset 300 --limit 300). N4 cache'i (H4) sayesinde "
            "zaten-işlenmiş kayıtlar üzerinden tekrar geçmek artık ucuz."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="--offset'ten sonraki ilk N hastayla sınırla.")
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
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--expected-image-count",
        type=int,
        default=None,
        help=(
            "A1.1 (2026-08-14) image_count gate -- Pass 3'e geçmeden ÖNCE, "
            "stats dosyasındaki donmuş istatistiğin image_count'unun bu "
            f"değere eşit olması ZORUNLU (UPenn için önerilen: {DEFAULT_EXPECTED_IMAGE_COUNT}, "
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
    # KRİTİK (smoke testte TCGA scriptinde bulundu, burada da geçerli):
    # apply_zscore_normalization() stats dosyasını YALNIZ bu ortam
    # değişkeninden okur (fit_source_zscore_statistics()'e verilen
    # stats_path= parametresi apply() tarafını ETKİLEMEZ).
    os.environ["GBMAID_ZSCORE_STATS_PATH"] = str(args.stats_path)

    if args.fresh_fit and args.stats_path.is_file():
        args.stats_path.unlink()

    rows_out, already_done = _load_done_keys(args.report)
    if already_done:
        print(f"Idempotent devam: {len(already_done)} (hasta,bölge) zaten OK, atlanacak.", flush=True)

    extractor = _make_extractor()
    connection = get_connection(readonly=True)
    try:
        patients = _fetch_upenn_patient_scans(connection)
        existing_volumes = _fetch_existing_volumes(connection)
    finally:
        connection.close()

    # A1.5 -- offset/limit UYGULANMADAN ÖNCE, TAM kohortun patient_id
    # kümesi (manifest'in `run_complete` hesabı için; UPenn'de patient_id
    # kullanılır çünkü T1CE_TARAMASI_YOK satırlarında scan_id boş kalabilir).
    full_cohort_patient_ids = {str(patient["patient_id"]) for patient in patients}

    if args.offset:
        patients = patients[args.offset :]
    if args.limit:
        patients = patients[: args.limit]

    t_start = time.time()

    # ---- Pass 1: çözümle + doğrula + N4 ----
    resolved_patients: list[dict[str, object]] = []
    n4_paths: list[str] = []
    processed_pass1 = 0

    for patient in patients:
        patient_id = patient["patient_id"]
        scan_id = patient["scan_id"]
        file_path = patient["file_path"]
        timepoint_label = patient["timepoint_label"]
        processed_pass1 += 1

        if scan_id is None or file_path is None:
            rows_out.append(
                _blank_row(
                    patient_id=patient_id,
                    scan_id="",
                    timepoint_used="",
                    region="-",
                    mask_source="",
                    status="T1CE_TARAMASI_YOK",
                )
            )
            continue

        timepoint_used = timepoint_label
        if timepoint_label == "post-op":
            timepoint_used = "post-op_fallback_no_pretreatment"

        try:
            image_path = resolve_nas_path(file_path)
        except FileNotFoundError as exc:
            rows_out.append(
                _blank_row(
                    patient_id=patient_id,
                    scan_id=scan_id,
                    timepoint_used=timepoint_used,
                    region="-",
                    mask_source="",
                    status=f"GORUNTU_YOK:{exc}",
                )
            )
            continue

        try:
            mask_path, mask_source, mask_warnings = _upenn_mask(image_path)
        except (ReadyMaskNotFoundError, ValueError) as exc:
            rows_out.append(
                _blank_row(
                    patient_id=patient_id,
                    scan_id=scan_id,
                    timepoint_used=timepoint_used,
                    region="-",
                    mask_source="",
                    status=f"MASKE_YOK:{type(exc).__name__}:{exc}",
                )
            )
            continue

        geometry = validate_image_mask_geometry(image_path, mask_path)
        if not geometry["geometry_match"]:
            rows_out.append(
                _blank_row(
                    patient_id=patient_id,
                    scan_id=scan_id,
                    timepoint_used=timepoint_used,
                    region="-",
                    mask_source=mask_source,
                    status="GEOMETRI_UYUSMUYOR",
                    warning="; ".join(mask_warnings),
                )
            )
            continue

        mask_labels = set(geometry["mask"]["labels"])
        label_map = REGION_LABELS_BY_MASK_SOURCE[mask_source]
        db_tool = MASK_SOURCE_TO_TOOL[mask_source]

        # A2 (2026-08-14) -- native NC/ED/ET'e ek olarak türetilmiş
        # WT_derived/TC_derived de pending_regions'a eklenir. Sentinel
        # değer `None` -- Pass 3'te bunun native bir etiket DEĞİL,
        # `build_derived_region_masks()` ile bellekte üretilecek bir
        # ikili maske olduğunu ayırt etmek için kullanılır. Guard:
        # `mask_source` burada ZATEN yalnız `upenn_expert`/
        # `upenn_automated_approx` olabilir (`_upenn_mask()`'in kendi
        # sözleşmesi), yani `WT_TC_DERIVATION_ALLOWED_MASK_SOURCES`
        # kontrolü burada FİİLEN her zaman geçer -- yine de sessiz
        # varsayım yerine AÇIKÇA assert edilir (mimari kural).
        if mask_source not in WT_TC_DERIVATION_ALLOWED_MASK_SOURCES:
            raise AssertionError(
                f"_upenn_mask() beklenmedik bir mask_source döndürdü: "
                f"{mask_source!r} -- WT/TC türetmesi tanımsız (2026-08-14 kararı)."
            )
        derived_pending = {
            region: None
            for region in ("WT_derived", "TC_derived")
            if (patient_id, region) not in already_done
        }

        pending_regions = {
            region: label_value
            for region, label_value in label_map.items()
            if (patient_id, region) not in already_done
        }
        pending_regions.update(derived_pending)
        if not pending_regions:
            continue

        label_yok_regions = [
            r for r, lv in pending_regions.items() if lv is not None and float(lv) not in mask_labels
        ]
        # Türetilmiş bölgeler için "0-voksel" kontrolü ayrı yapılır --
        # native etiket kontrolü gibi tek bir label'a değil, bileşen
        # birleşimine bakar (bkz. `derived_region_has_voxels()`).
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
                    timepoint_used=timepoint_used,
                    region=region,
                    mask_source=mask_source,
                    status="LABEL_YOK_0_VOXEL",
                    warning="; ".join(mask_warnings),
                )
            )
        remaining_regions = {r: lv for r, lv in pending_regions.items() if r not in label_yok_regions}
        if not remaining_regions:
            continue

        try:
            n4_path = _apply_n4_cached(str(image_path))
        except Exception as exc:  # noqa: BLE001
            for region in remaining_regions:
                rows_out.append(
                    _blank_row(
                        patient_id=patient_id,
                        scan_id=scan_id,
                        timepoint_used=timepoint_used,
                        region=region,
                        mask_source=mask_source,
                        status=f"HATA:N4:{type(exc).__name__}:{exc}",
                        warning="; ".join(mask_warnings),
                    )
                )
            continue

        n4_paths.append(n4_path)
        resolved_patients.append(
            {
                "patient_id": patient_id,
                "scan_id": scan_id,
                "timepoint_used": timepoint_used,
                "mask_path": mask_path,
                "mask_source": mask_source,
                "db_tool": db_tool,
                "regions": remaining_regions,
                "n4_path": n4_path,
                "warnings": mask_warnings,
            }
        )

        if processed_pass1 % args.progress_every == 0:
            elapsed = time.time() - t_start
            print(
                f"[Pass1 N4] [{processed_pass1}/{len(patients)}] hasta işlendi, "
                f"gecen={elapsed:.0f}s",
                flush=True,
            )

    # ---- Pass 2: TEK T1ce-özel, kaynak-bazlı (UPenn) Z-score fit ----
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
                f"sayısını AÇIKÇA belirtin (UPenn için önerilen: {DEFAULT_EXPECTED_IMAGE_COUNT}) "
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

    # ---- Pass 3: Z-score normalize + C32 PyRadiomics ----
    ok_count = fail_count = 0
    processed_pass3 = 0

    def flush() -> None:
        with args.report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows_out)

    for patient in resolved_patients:
        patient_id = patient["patient_id"]
        scan_id = patient["scan_id"]
        timepoint_used = patient["timepoint_used"]
        mask_path = patient["mask_path"]
        mask_source = patient["mask_source"]
        db_tool = patient["db_tool"]
        warnings_joined = "; ".join(patient["warnings"])
        processed_pass3 += 1

        try:
            zscore_path = apply_zscore_normalization(patient["n4_path"], CANONICAL_SOURCE)
        except Exception as exc:  # noqa: BLE001
            for region in patient["regions"]:
                rows_out.append(
                    _blank_row(
                        patient_id=patient_id,
                        scan_id=scan_id,
                        timepoint_used=timepoint_used,
                        region=region,
                        mask_source=mask_source,
                        status=f"HATA:ZSCORE:{type(exc).__name__}:{exc}",
                        warning=warnings_joined,
                    )
                )
                fail_count += 1
            continue

        geometry = validate_image_mask_geometry(zscore_path, mask_path)
        if not geometry["geometry_match"]:
            for region in patient["regions"]:
                rows_out.append(
                    _blank_row(
                        patient_id=patient_id,
                        scan_id=scan_id,
                        timepoint_used=timepoint_used,
                        region=region,
                        mask_source=mask_source,
                        status="GEOMETRI_UYUSMUYOR_ZSCORE",
                        warning=warnings_joined,
                    )
                )
                fail_count += 1
            continue

        for region, label_value in patient["regions"].items():
            try:
                if label_value is None:
                    # A2 (2026-08-14) -- WT_derived/TC_derived: native bir
                    # etiket DEĞİL, bellekte üretilmiş ikili maske (label=1).
                    # `mask_path` burada Pass 1'den beri DEĞİŞMEDİ (UPenn'de
                    # LUMIERE'deki gibi bir resample-retry adımı YOK), bu
                    # yüzden yeniden inşa etmek güvenli/ucuz.
                    derived_masks = build_derived_region_masks(mask_path, mask_source)
                    result = extractor.execute(
                        str(zscore_path), derived_masks[region], label=1
                    )
                else:
                    result = extractor.execute(str(zscore_path), str(mask_path), label=label_value)
                shape, first_order, texture = _split_features(result)
                voxel_volume = shape.get("original_shape_VoxelVolume")
                db_volume = existing_volumes.get((scan_id, db_tool, region))
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
                        "timepoint_used": timepoint_used,
                        "region": region,
                        "mask_source": mask_source,
                        "status": "OK",
                        "voxel_volume": voxel_volume,
                        "db_volume": db_volume if db_volume is not None else "",
                        "db_tool": db_tool,
                        "volume_match": volume_match,
                        "volume_diff_mm3": volume_diff,
                        "warning": warnings_joined,
                        "shape_features_json": json.dumps(shape),
                        "first_order_features_json": json.dumps(first_order),
                        "texture_features_json": json.dumps(texture),
                        "surface_area": shape.get("original_shape_SurfaceArea", ""),
                        "entropy": first_order.get("original_firstorder_Entropy", ""),
                        "contrast": texture.get("original_glcm_Contrast", ""),
                    }
                )
                ok_count += 1
            except Exception as exc:  # noqa: BLE001
                rows_out.append(
                    _blank_row(
                        patient_id=patient_id,
                        scan_id=scan_id,
                        timepoint_used=timepoint_used,
                        region=region,
                        mask_source=mask_source,
                        status=f"HATA:{type(exc).__name__}:{exc}",
                        warning=warnings_joined,
                    )
                )
                fail_count += 1

        if processed_pass3 % args.progress_every == 0:
            flush()
            elapsed = time.time() - t_start
            print(
                f"[Pass3 C32] [{processed_pass3}/{len(resolved_patients)}] hasta işlendi, "
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
        full_cohort_ids=full_cohort_patient_ids,
        covered_id_field="patient_id",
        partial_run=partial_run,
        offset=args.offset,
        limit=args.limit,
    )

    print(
        f"\nBitti. {len(resolved_patients)} hasta C32 için işlendi (Pass1'de {processed_pass1} hasta "
        f"gözden geçirildi). ok={ok_count} fail={fail_count}. Rapor: {args.report}\n"
        f"Provenance manifest: {manifest_path}\n"
        f"NOT: DB'ye yazım için önerilen segmentation_tool adı: "
        f"{SEGMENTATION_TOOL_C32_FOR_DB_WRITE!r} (write_upenn_pyradiomics_to_db.py "
        f"güncellemesi bu görevin kapsamı dışında, ayrı takip)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
