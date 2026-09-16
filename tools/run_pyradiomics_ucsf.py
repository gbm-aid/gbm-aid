"""UCSF-PDGM tam kohort C32 PyRadiomics 107 özellik çıkarımı.

KARAR (2026-08-18, Barış onayı): bugünün 5-hasta pilotunun (0 hata, 107/107
özellik) tam kohorta genişletilmesi. C32 sözleşmesine BİREBİR uyar (bkz.
decisions/2026-08-13-pyradiomics-c32-bincount-karari.md): N4 bias-field
düzeltmesi + T1ce-özel (bu script yalnız T1c işler) kaynak-bazlı (SADECE
UCSF'in KENDİ kohortu -- UPenn'inki KULLANILMAZ) Z-score + `binCount=32`
(binWidth DEĞİL) + `pyradiomics==3.0.1` pinli. Extractor ayarları AÇIKÇA
verilir (bkz. `_make_extractor()`), aynı desen `run_pyradiomics_upenn.py`/
`run_pyradiomics_tcga.py` ile.

Bölge sözleşmesi: UPenn ile AYNI 5-bölge kontratı (NC/ED/ET + WT_derived +
TC_derived), `pipeline/radiomics_volume.py::REGION_LABELS_BY_MASK_SOURCE
["ucsf_native"]` (NC=1/ED=2/ET=4) ve `DERIVED_REGION_ROLE_ALIASES
["ucsf_native"]` üzerinden (2026-08-18 eklendi, UPenn'le KİMLİK eşlemesi).

Girdi görüntüsü KESİNLİKLE `<ID>_T1c.nii.gz` (ham) -- `<ID>_T1c_bias.nii.gz`
(UCSF'in KENDİ N4 çıktısı) ASLA KULLANILMAZ (çifte bias-field düzeltmesi
riski, bkz. `pipeline/radiomics_volume.py` modül-üstü docstring).

UPenn/LUMIERE/TCGA'nın aksine UCSF NAS/DB'de DEĞİL -- yerel diskte
(`PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5/`) indirilmiş paket olarak durur.
Bu yüzden bu script `pipeline/segmentation.py::resolve_ready_mask()`'ı
KULLANMAZ (o modül NAS/DB yol çözümlemesi içindir) -- hasta listesi ve
dosya yolları `tools/freeze_ucsf_cohort.py`'nin ürettiği DONDURULMUŞ CSV'den
okunur (`--frozen-cohort`).

DB'YE HİÇBİR YAZMA YAPILMAZ (bu script salt CSV üretir). Önerilen
`segmentation_tool` adı DB yazımı için: `UCSF-PDGM-PyRadiomics-107-C32`.

Hasta seçimi/kohort dondurma kuralı: bkz. `tools/freeze_ucsf_cohort.py`
docstring'i ve `gbm-aid mert/artifacts/week3/ucsf_cohort/
ucsf_frozen_cohort_2026-08-18_*.csv.manifest.json`.
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

REPO_ROOT = Path(__file__).absolute().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# DİKKAT (aynı bug sınıfı run_pyradiomics_upenn.py/tcga.py'de belgelendi):
# proje kökü Türkçe karakter ("Barış") içerdiği için SimpleITK bu yolları
# OKUYAMIYOR ("Unable to open ... for reading", canlı doğrulandı 2026-08-18
# smoke testinde). Çözüm AYNI: `subst` ile ASCII-safe sürücü harfleri
# (`X:` = gbm-aid mert, `Y:` = proje kökü) -- ama X:/Y: birbirinin
# parent/child'ı OLARAK GÖRÜNMÜYOR (subst her ikisini de bağımsız kök
# yapıyor), bu yüzden `parents[N]` ile Y:'yi X:'ten TÜRETEMEYİZ. Proje
# kökü (PKG klasörünün bulunduğu yer) `--disk-root` ile AÇIKÇA verilir,
# varsayılan `Y:\` (mevcut subst kaydı, `subst` komutuyla doğrulanmalı).
DEFAULT_DISK_ROOT = Path("Y:/")

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

CANONICAL_SOURCE = "UCSF"
MASK_SOURCE = "ucsf_native"
SEGMENTATION_TOOL_C32_FOR_DB_WRITE = "UCSF-PDGM-PyRadiomics-107-C32"

OUTPUT_ROOT = REPO_ROOT / "artifacts" / "week3" / "pyradiomics"
DEFAULT_PROCESSED_ROOT = OUTPUT_ROOT / "c32_n4_zscore_ucsf"
DEFAULT_ZSCORE_STATS_PATH = OUTPUT_ROOT / "source_stats_t1ce_c32_ucsf.json"
DEFAULT_REPORT = OUTPUT_ROOT / "ucsf_pyradiomics_c32.csv"
DEFAULT_FROZEN_COHORT = (
    REPO_ROOT / "artifacts" / "week3" / "ucsf_cohort" / "ucsf_frozen_cohort_2026-08-18_295.csv"
)

EXPECTED_PYRADIOMICS_VERSION = "3.0.1"
ZSCORE_SCOPE = "t1ce_source"

FIELDNAMES = [
    "patient_id",
    "region",
    "mask_source",
    "status",
    "voxel_volume",
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
    return {
        "n4_iterations": os.environ.get("GBMAID_N4_ITERATIONS", "50,50,30,20"),
        "n4_shrink_factor": os.environ.get("GBMAID_N4_SHRINK_FACTOR", "4"),
    }


def _n4_cache_meta_path(cached_path: Path) -> Path:
    return cached_path.with_name(cached_path.name + ".n4meta.json")


def _input_content_hash(input_path: Path) -> str:
    hasher = hashlib.sha256()
    with input_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _n4_cache_hit(cached_path: Path, *, input_path: Path) -> bool:
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
    n4_applied: bool,
    stats_path: Path,
    zscore_stats_image_count: int | None,
    expected_image_count: int | None,
    full_cohort_ids: set[str],
    partial_run: bool,
    offset: int,
    limit: int | None,
    frozen_cohort_path: Path,
) -> Path:
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
        raise RuntimeError(f"Rapor CSV'si bulunamadı, manifest yazılamıyor: {report_path}")
    if not stats_path.is_file():
        raise RuntimeError(f"Z-score stats dosyası bulunamadı, manifest yazılamıyor: {stats_path}")

    report_sha256 = _file_sha256(report_path)
    with report_path.open("r", newline="", encoding="utf-8") as handle:
        report_rows_list = list(csv.DictReader(handle))
    report_rows = len(report_rows_list)

    status_distribution: dict[str, int] = {}
    covered_ids: set[str] = set()
    for row in report_rows_list:
        status_key = str(row.get("status", ""))
        status_distribution[status_key] = status_distribution.get(status_key, 0) + 1
        identifier = row.get("patient_id")
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
        "frozen_cohort_csv": str(frozen_cohort_path),
        "frozen_cohort_csv_sha256": _file_sha256(frozen_cohort_path),
        "segmentation_tool_for_db_write": SEGMENTATION_TOOL_C32_FOR_DB_WRITE,
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


def _load_frozen_cohort(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: r["ID"])
    return rows


def _load_done_keys(report_path: Path) -> tuple[list[dict[str, object]], set[tuple[str, str]]]:
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


def _blank_row(*, patient_id, region, mask_source, status, warning="") -> dict[str, object]:
    return {
        "patient_id": patient_id,
        "region": region,
        "mask_source": mask_source,
        "status": status[:200] if isinstance(status, str) else status,
        "voxel_volume": "",
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
            "UCSF-PDGM hazır maskelerinden C32 (N4+T1ce-özel Z-score+binCount=32) "
            "PyRadiomics 107 özellik üret (yalnız CSV, DB'ye yazmaz)."
        )
    )
    parser.add_argument("--frozen-cohort", type=Path, default=DEFAULT_FROZEN_COHORT)
    parser.add_argument(
        "--disk-root",
        type=Path,
        default=DEFAULT_DISK_ROOT,
        help=(
            "`PKG - UCSF-PDGM Version 5` klasörünün bulunduğu ASCII-safe kök "
            "(varsayılan Y:\\, mevcut `subst` kaydı -- proje kökü Türkçe "
            "karakter içerdiği için SimpleITK doğrudan C:\\Users\\...\\Barış\\... "
            "yolunu OKUYAMIYOR, canlı doğrulandı)."
        ),
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stats-path", type=Path, default=DEFAULT_ZSCORE_STATS_PATH)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--fresh-fit", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--expected-image-count", type=int, default=None)
    parser.add_argument("--skip-image-count-gate", action="store_true")
    args = parser.parse_args()

    disk_base = args.disk_root / "PKG - UCSF-PDGM Version 5" / "UCSF-PDGM-v5"
    if not disk_base.is_dir():
        print(
            f"HATA: --disk-root ({args.disk_root}) altında UCSF paket klasörü "
            f"bulunamadı: {disk_base}. `subst` çıktısını kontrol edin.",
            file=sys.stderr,
        )
        return 2

    args.report.parent.mkdir(parents=True, exist_ok=True)
    os.environ["GBMAID_PROCESSED_ROOT"] = str(args.processed_root)
    os.environ["GBMAID_ZSCORE_STATS_PATH"] = str(args.stats_path)

    if args.fresh_fit and args.stats_path.is_file():
        args.stats_path.unlink()

    rows_out, already_done = _load_done_keys(args.report)
    if already_done:
        print(f"Idempotent devam: {len(already_done)} (hasta,bölge) zaten OK, atlanacak.", flush=True)

    extractor = _make_extractor()
    patients = _load_frozen_cohort(args.frozen_cohort)
    full_cohort_patient_ids = {str(p["ID"]) for p in patients}

    if args.offset:
        patients = patients[args.offset :]
    if args.limit:
        patients = patients[: args.limit]

    t_start = time.time()

    # ---- Pass 1: çözümle + doğrula + N4 ----
    resolved_patients: list[dict[str, object]] = []
    n4_paths: list[str] = []
    processed_pass1 = 0

    label_map = REGION_LABELS_BY_MASK_SOURCE[MASK_SOURCE]
    if MASK_SOURCE not in WT_TC_DERIVATION_ALLOWED_MASK_SOURCES:
        raise AssertionError(
            f"MASK_SOURCE={MASK_SOURCE!r} WT/TC türetmesi için tanımsız -- "
            "pipeline/radiomics_volume.py::DERIVED_REGION_ROLE_ALIASES eksik."
        )

    for patient in patients:
        patient_id = patient["ID"]
        folder = patient["folder"]
        t1c_file = patient["t1c_file"]
        seg_file = patient["seg_file"]
        processed_pass1 += 1

        image_path = disk_base / folder / t1c_file
        mask_path = disk_base / folder / seg_file

        if not image_path.is_file():
            rows_out.append(
                _blank_row(
                    patient_id=patient_id,
                    region="-",
                    mask_source="",
                    status=f"GORUNTU_YOK:{image_path}",
                )
            )
            continue
        if not mask_path.is_file():
            rows_out.append(
                _blank_row(
                    patient_id=patient_id,
                    region="-",
                    mask_source="",
                    status=f"MASKE_YOK:{mask_path}",
                )
            )
            continue

        geometry = validate_image_mask_geometry(image_path, mask_path)
        if not geometry["geometry_match"]:
            rows_out.append(
                _blank_row(
                    patient_id=patient_id,
                    region="-",
                    mask_source=MASK_SOURCE,
                    status="GEOMETRI_UYUSMUYOR",
                )
            )
            continue

        mask_labels = set(geometry["mask"]["labels"])

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
        if derived_pending:
            derived_masks_pass1 = build_derived_region_masks(mask_path, MASK_SOURCE)
            for region in derived_pending:
                if not derived_region_has_voxels(derived_masks_pass1[region]):
                    label_yok_regions.append(region)
        for region in label_yok_regions:
            rows_out.append(
                _blank_row(
                    patient_id=patient_id,
                    region=region,
                    mask_source=MASK_SOURCE,
                    status="LABEL_YOK_0_VOXEL",
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
                        region=region,
                        mask_source=MASK_SOURCE,
                        status=f"HATA:N4:{type(exc).__name__}:{exc}",
                    )
                )
            continue

        n4_paths.append(n4_path)
        resolved_patients.append(
            {
                "patient_id": patient_id,
                "mask_path": mask_path,
                "regions": remaining_regions,
                "n4_path": n4_path,
            }
        )

        if processed_pass1 % args.progress_every == 0:
            elapsed = time.time() - t_start
            print(
                f"[Pass1 N4] [{processed_pass1}/{len(patients)}] hasta işlendi, gecen={elapsed:.0f}s",
                flush=True,
            )

    # ---- Pass 2: TEK T1ce-özel, kaynak-bazlı (UCSF'in KENDİ kohortu) Z-score fit ----
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
            "istatistik var, refit ATLANDI. Zorla yeniden fit için --fresh-fit kullanın.",
            flush=True,
        )

    gate_record = (
        fit_record
        if fit_record is not None
        else get_fitted_zscore_record(CANONICAL_SOURCE, stats_path=args.stats_path)
    )
    if args.skip_image_count_gate:
        print(
            "UYARI: --skip-image-count-gate verildi -- image_count kontrolü "
            "BİLİNÇLİ OLARAK ATLANDI (operatör override).",
            flush=True,
        )
    else:
        if args.expected_image_count is None:
            print(
                "HATA: --expected-image-count verilmedi (ve --skip-image-count-gate "
                "de verilmedi) -- Pass 3'e GEÇEMEZ. Dondurulmuş kohort büyüklüğünü "
                "AÇIKÇA belirtin veya bilinçli bir smoke test ise "
                "--skip-image-count-gate kullanın.",
                file=sys.stderr,
            )
            return 2
        actual_count = gate_record["image_count"] if gate_record else None
        if actual_count != args.expected_image_count:
            print(
                "HATA: image_count gate REDDETTİ -- stats dosyasındaki donmuş "
                f"istatistiğin image_count={actual_count!r}, beklenen tam kohort="
                f"{args.expected_image_count}. Koşu REDDEDİLDİ, Pass 3 ÇALIŞTIRILMADI. "
                f"stats_path={args.stats_path}. Bilinçli devam için --fresh-fit ile "
                "TAM kohortu yeniden fit edin veya --skip-image-count-gate kullanın.",
                file=sys.stderr,
            )
            return 2
        print(f"image_count gate GEÇTİ: {actual_count} == beklenen {args.expected_image_count}.", flush=True)

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
        mask_path = patient["mask_path"]
        processed_pass3 += 1

        try:
            zscore_path = apply_zscore_normalization(patient["n4_path"], CANONICAL_SOURCE)
        except Exception as exc:  # noqa: BLE001
            for region in patient["regions"]:
                rows_out.append(
                    _blank_row(
                        patient_id=patient_id,
                        region=region,
                        mask_source=MASK_SOURCE,
                        status=f"HATA:ZSCORE:{type(exc).__name__}:{exc}",
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
                        region=region,
                        mask_source=MASK_SOURCE,
                        status="GEOMETRI_UYUSMUYOR_ZSCORE",
                    )
                )
                fail_count += 1
            continue

        for region, label_value in patient["regions"].items():
            try:
                if label_value is None:
                    derived_masks = build_derived_region_masks(mask_path, MASK_SOURCE)
                    result = extractor.execute(str(zscore_path), derived_masks[region], label=1)
                else:
                    result = extractor.execute(str(zscore_path), str(mask_path), label=label_value)
                shape, first_order, texture = _split_features(result)
                voxel_volume = shape.get("original_shape_VoxelVolume")

                rows_out.append(
                    {
                        "patient_id": patient_id,
                        "region": region,
                        "mask_source": MASK_SOURCE,
                        "status": "OK",
                        "voxel_volume": voxel_volume,
                        "warning": "",
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
                        region=region,
                        mask_source=MASK_SOURCE,
                        status=f"HATA:{type(exc).__name__}:{exc}",
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
        n4_applied=True,
        stats_path=args.stats_path,
        zscore_stats_image_count=(gate_record["image_count"] if gate_record else None),
        expected_image_count=args.expected_image_count,
        full_cohort_ids=full_cohort_patient_ids,
        partial_run=partial_run,
        offset=args.offset,
        limit=args.limit,
        frozen_cohort_path=args.frozen_cohort,
    )

    print(
        f"\nBitti. {len(resolved_patients)} hasta C32 için işlendi (Pass1'de {processed_pass1} hasta "
        f"gözden geçirildi). ok={ok_count} fail={fail_count}. Rapor: {args.report}\n"
        f"Provenance manifest: {manifest_path}\n"
        f"NOT: DB'ye yazım için önerilen segmentation_tool adı: "
        f"{SEGMENTATION_TOOL_C32_FOR_DB_WRITE!r}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
