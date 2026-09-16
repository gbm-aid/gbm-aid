"""Doğrulama Adım A: HAM (ham NAS) vs DOĞRU-SIRALI (N4->T1ce-only Z-score)
PyRadiomics 107-özellik karşılaştırması, küçük bir örneklemde (varsayılan
5 TCGA + 5 UPenn-GBM + 5 LUMIERE).

BAĞLAM (bkz. 2026-08-13 görev talimatı): `run_pyradiomics_tcga.py`/
`run_pyradiomics_upenn.py`/`run_pyradiomics_lumiere.py` hiçbiri N4+Z-score
çıktısını KULLANMIYOR -- hepsi `resolve_nas_path()` ile ham NAS görüntüsünden
doğrudan PyRadiomics çalıştırıyor. Bu, `harmonization.py`'nin dokümante
edilmiş "değişmez işlem sırası" (N4->Z-score->PyRadiomics->ComBat) ile
çelişiyor. AYRICA `fit_source_zscore_statistics()` üretimde kaynak-başına
TEK bir (mean,std) fit ediyor, MODALİTEDEN BAĞIMSIZ (T1/T1ce/T2/FLAIR aynı
havuzda) -- bu script'in "DOĞRU-SIRALI" kolu, fonksiyon imzasını
DEĞİŞTİRMEDEN, çağıran tarafta T1ce-only bir alt-küme ile fit ederek bu
ikinci sorunu da izole eder (Codex'in önerdiği "minimum invaziv düzeltme").

Bu script:
1. DB'den (readonly) her kaynaktan N adet, T1ce modaliteli, hazır maskesi
   olan tarama seçer (varsayılan olarak zaten `radiomics` tablosunda
   OK/dolu satırı olanlardan -- kolaylık için).
2. Her tarama için HAM PyRadiomics vektörünü üretir: mevcut
   run_pyradiomics_*.py scriptleriyle BİREBİR AYNI extractor ayarları
   (binWidth=25, normalize=False, shape+firstorder+glcm+glrlm+glszm+
   ngtdm+gldm) ve BİREBİR AYNI maske-çözümleme mantığı (TCGA: whole/core
   ayrı dosyalar; UPenn/LUMIERE: `resolve_ready_mask()` çok-etiketli
   maske).
3. Her taramaya N4 bias-field düzeltmesi uygular, SADECE bu örneklemin
   T1ce görüntüleriyle (kaynak başına) bir Z-score fit'i yapar (üretim
   fonksiyonları DEĞİŞTİRİLMEDEN, `image_paths` listesi bu script
   tarafından T1ce-only filtrelenip geçiriliyor), Z-score normalize edilmiş
   görüntüden AYNI extractor ayarlarıyla PyRadiomics çalıştırır.
4. HAM ve DOĞRU-SIRALI vektörlerini karşılaştırıp uzun-format bir CSV
   (`validation_raw_vs_harmonized_sample.csv`) + kaynak x özellik-sınıfı
   özet CSV'si (`validation_raw_vs_harmonized_summary.csv`) üretir.

KISITLAR (görev talimatı):
- DB'ye HİÇBİR YAZMA yapılmaz (`get_connection(readonly=True)`).
- Üretim Z-score istatistik dosyasına (`artifacts/week2/zscore/
  source_stats.json`) DOKUNULMAZ -- bu script kendi TEMP stats dosyasını
  (`GBMAID_ZSCORE_STATS_PATH` ortam değişkeniyle, script'in kendi çıktı
  klasörü altında) kullanır, üretim N4/Z-score görüntü klasörlerine
  (`artifacts/week2/n4`, `artifacts/week2/zscore`) de yazmaz -- kendi ayrı
  `validation_n4_zscore_tmp/` klasörünü kullanır.
- Görüntü/maske geometrisi eşleşmiyorsa PyRadiomics'e GEÇİLMEZ, açık hata
  ile o tarama/bölge atlanır (mimari kural, sessiz fallback yasak).

Bu script yalnız Python 3.10 ortamında (pyradiomics 3.0.1) çalışır --
`tools/run_pyradiomics_tcga.py` ile aynı ortam kısıtı.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from psycopg2.extras import RealDictCursor  # noqa: E402

from db_connection import get_connection  # noqa: E402
from pipeline.harmonization import (  # noqa: E402
    apply_n4_bias_correction,
    apply_zscore_normalization,
    canonical_source,
    fit_source_zscore_statistics,
)
from pipeline.radiomics_volume import REGION_LABELS_BY_MASK_SOURCE  # noqa: E402
from pipeline.resampling import validate_image_mask_geometry  # noqa: E402
from pipeline.segmentation import (  # noqa: E402
    ImageMaskGeometryError,
    ReadyMaskNotFoundError,
    _find_nifti,
    resolve_nas_path,
    resolve_ready_mask,
)

OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "week3" / "pyradiomics"
TMP_ROOT = OUTPUT_ROOT / "validation_n4_zscore_tmp"
TEMP_STATS_PATH = OUTPUT_ROOT / "validation_zscore_stats_TEMP.json"

TCGA_REGION_MASK_FILE = {
    "WT": ("whole.nii.gz", "whole.nii"),
    "TC": ("core.nii.gz", "core.nii"),
}

SOURCE_DB_NAME = {
    "TCGA": "TCGA-GBM",
    "UPenn": "UPenn-GBM",
    "LUMIERE": "LUMIERE",
}

FEATURE_CLASS_PREFIXES = {
    "shape": ("original_shape_",),
    "first_order": ("original_firstorder_",),
    "texture": (
        "original_glcm_",
        "original_glrlm_",
        "original_glszm_",
        "original_ngtdm_",
        "original_gldm_",
    ),
}


def _make_extractor():
    """"A varyantı" (mevcut/eski üretim davranışı) -- DAVRANIŞ DEĞİŞTİRİLMEDİ.

    2026-08-13 Codex bulgusu: boş `RadiomicsFeatureExtractor()` çağrısında
    `binWidth` extractor'ın `settings` sözlüğünde HİÇ GÖRÜNMÜYOR (özellik
    sınıfının içine gömülü `kwargs.get('binWidth', 25)` varsayılanı) --
    kodu okuyan biri hangi binWidth'in kullanıldığını göremiyordu. Bu
    fonksiyon hâlâ TAMAMEN AYNI A-varyantı ayarlarını üretir (binWidth=25,
    normalize=False, yalnız 'original' görüntü tipi) -- amaç davranışı
    C32'ye çevirmek DEĞİL, örtük varsayılanı görünür kılmak (bkz.
    decisions/2026-08-13-pyradiomics-c32-bincount-karari.md "Sessiz
    varsayılan" bölümü).
    """

    from radiomics import featureextractor

    logging.getLogger("radiomics").setLevel(logging.ERROR)
    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=25,
        normalize=False,
    )
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()
    for cls in ("shape", "firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm"):
        extractor.enableFeatureClassByName(cls)
    return extractor


def _feature_class(name: str) -> str:
    for cls, prefixes in FEATURE_CLASS_PREFIXES.items():
        if name.startswith(prefixes):
            return cls
    return "unknown"


def _extract_numeric(result: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in result.items():
        if not key.startswith("original_"):
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _fetch_candidates(connection, source_db_name: str, limit: int) -> list[dict]:
    cursor = connection.cursor(cursor_factory=RealDictCursor)
    try:
        if source_db_name == "TCGA-GBM":
            cursor.execute(
                """
                SELECT DISTINCT ON (p.patient_id) p.patient_id, ms.scan_id,
                       ms.file_path, ms.modality
                FROM radiomics r
                JOIN mr_scans ms ON ms.scan_id = r.scan_id
                JOIN patients p ON p.patient_id = ms.patient_id
                JOIN dataset_sources ds ON ds.source_id = p.source_id
                WHERE ds.source_name = %s AND r.segmentation_tool = 'TCGA-ground-truth'
                  AND r.shape_features IS NOT NULL AND r.shape_features != '{}'::jsonb
                ORDER BY p.patient_id, ms.scan_id
                LIMIT %s
                """,
                (source_db_name, limit),
            )
        elif source_db_name == "UPenn-GBM":
            cursor.execute(
                """
                SELECT DISTINCT ON (p.patient_id) p.patient_id, ms.scan_id,
                       ms.file_path, ms.modality
                FROM radiomics r
                JOIN mr_scans ms ON ms.scan_id = r.scan_id
                JOIN patients p ON p.patient_id = ms.patient_id
                JOIN dataset_sources ds ON ds.source_id = p.source_id
                WHERE ds.source_name = %s AND ms.modality = 'T1ce'
                  AND r.shape_features IS NOT NULL AND r.shape_features != '{}'::jsonb
                ORDER BY p.patient_id, ms.scan_id
                LIMIT %s
                """,
                (source_db_name, limit),
            )
        elif source_db_name == "LUMIERE":
            cursor.execute(
                """
                SELECT DISTINCT ON (p.patient_id) p.patient_id, ms.scan_id,
                       ms.file_path, ms.modality
                FROM radiomics r
                JOIN mr_scans ms ON ms.scan_id = r.scan_id
                JOIN patients p ON p.patient_id = ms.patient_id
                JOIN dataset_sources ds ON ds.source_id = p.source_id
                WHERE ds.source_name = %s AND ms.modality = 'T1ce'
                  AND r.segmentation_tool = 'DeepBraTumIA'
                  AND r.shape_features IS NOT NULL AND r.shape_features != '{}'::jsonb
                ORDER BY p.patient_id, ms.scan_id
                LIMIT %s
                """,
                (source_db_name, limit),
            )
        else:
            raise ValueError(f"Bilinmeyen kaynak: {source_db_name}")
        return list(cursor.fetchall())
    finally:
        cursor.close()


def _resolve_regions(
    *, canonical: str, image_path: Path, patient_id: str, scan_id: int
) -> tuple[Path, dict[str, dict[str, object]]]:
    """Kaynağa uygun bölge->{mask_path,label} sözlüğünü döndür.

    Döner: (radyomik_için_kullanılacak_görüntü_yolu, {region: {"mask_path":..,
    "label":.., "mask_source":..}}).

    TCGA whole/core AYRI dosyalardır (tek çok-etiketli maske DEĞİL) -- bu
    yüzden `resolve_ready_mask()` (yalnız TEK maske döndürür) yerine
    `tools/run_pyradiomics_tcga.py` ile BİREBİR AYNI `_find_nifti()` mantığı
    kullanılır. UPenn/LUMIERE için `resolve_ready_mask()` + çok-etiketli
    `REGION_LABELS_BY_MASK_SOURCE` kullanılır (mevcut scriptlerle aynı).
    """

    if canonical == "TCGA":
        regions: dict[str, dict[str, object]] = {}
        for region, names in TCGA_REGION_MASK_FILE.items():
            mask_path = _find_nifti(image_path.parent, names)
            if mask_path is None:
                continue
            geometry = validate_image_mask_geometry(image_path, mask_path)
            if not geometry["geometry_match"]:
                raise ImageMaskGeometryError(
                    f"TCGA {region} görüntü/maske geometrisi eşleşmiyor: "
                    f"{image_path} vs {mask_path}"
                )
            regions[region] = {
                "mask_path": mask_path,
                "label": 1,
                "mask_source": "provided_whole_tumor" if region == "WT" else "provided_tumor_core",
            }
        if not regions:
            raise ReadyMaskNotFoundError(
                f"TCGA hazır whole/core maskesi bulunamadı: {image_path.parent}"
            )
        return image_path, regions

    resolved = resolve_ready_mask(
        source=canonical,
        image_path=image_path,
        patient_id=patient_id,
        scan_id=scan_id,
        raise_on_geometry_mismatch=True,
    )
    mask_source = resolved["mask_source"]
    if canonical == "LUMIERE" and mask_source != "lumiere_deepbratumia_native":
        raise ReadyMaskNotFoundError(
            f"LUMIERE ana kaynak (DeepBraTumIA native) bulunamadı, "
            f"HD-GLIO fallback'e düşüldü ({mask_source}) -- bu duyarlılık "
            "analizi aracı, kanonik vektöre karıştırılmaz (mimari kural)."
        )
    resolved_image = Path(resolved["image_path"])
    mask_path = Path(resolved["mask_path"])
    mask_labels = {float(v) for v in resolved["geometry"]["mask"]["labels"]}
    label_map = REGION_LABELS_BY_MASK_SOURCE[mask_source]
    regions = {}
    for region, label in label_map.items():
        if float(label) not in mask_labels:
            continue
        regions[region] = {"mask_path": mask_path, "label": label, "mask_source": mask_source}
    if not regions:
        raise ReadyMaskNotFoundError(
            f"{canonical} maskesinde beklenen hiçbir etiket bulunamadı: {mask_path}"
        )
    return resolved_image, regions


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Doğrulama Adım A: küçük örneklemde HAM vs N4+T1ce-only-Zscore "
            "PyRadiomics 107-özellik karşılaştırması (DB'ye yazmaz)."
        )
    )
    parser.add_argument("--n-per-source", type=int, default=5)
    parser.add_argument(
        "--candidate-pool-multiplier",
        type=int,
        default=4,
        help=(
            "DB'den n-per-source * bu çarpan kadar aday çek -- LUMIERE'de "
            "bilinen görüntü/maske geometri uyuşmazlığı oranı (~%%40, "
            "Task #19) yüzünden bazı adaylar 'resolve' aşamasında ATLANIR; "
            "her kaynak için TAM n-per-source başarılı tarama toplanana "
            "kadar fazladan adaylar denenir (sessiz fallback değil, "
            "yalnız daha büyük bir aday havuzu)."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=OUTPUT_ROOT / "validation_raw_vs_harmonized_sample.csv",
    )
    parser.add_argument(
        "--summary-report",
        type=Path,
        default=OUTPUT_ROOT / "validation_raw_vs_harmonized_summary.csv",
    )
    parser.add_argument(
        "--stats-report",
        type=Path,
        default=OUTPUT_ROOT / "validation_zscore_fit_stats.json",
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    # Üretim klasörleriyle KARIŞMASIN diye N4/Z-score çıktıları ayrı bir kök
    # altına yazılır; TEMP_STATS_PATH de üretim source_stats.json'ından ayrı.
    os.environ["GBMAID_PROCESSED_ROOT"] = str(TMP_ROOT)
    os.environ["GBMAID_ZSCORE_STATS_PATH"] = str(TEMP_STATS_PATH)
    if TEMP_STATS_PATH.is_file():
        TEMP_STATS_PATH.unlink()

    extractor = _make_extractor()
    connection = get_connection(readonly=True)
    try:
        candidates_by_source: dict[str, list[dict]] = {}
        for canonical, db_name in SOURCE_DB_NAME.items():
            candidates_by_source[canonical] = _fetch_candidates(
                connection, db_name, args.n_per_source * args.candidate_pool_multiplier
            )
    finally:
        connection.close()

    for canonical, rows in candidates_by_source.items():
        print(f"{canonical}: {len(rows)} aday hasta seçildi.", flush=True)
        for row in rows:
            print(f"  - {row['patient_id']} scan_id={row['scan_id']}", flush=True)

    # ---- Pass 1: HAM PyRadiomics + N4 (her kaynak/tarama için) ----
    scan_records: list[dict[str, object]] = []
    n4_paths_by_source: dict[str, list[str]] = {"TCGA": [], "UPenn": [], "LUMIERE": []}
    skipped: list[dict[str, str]] = []

    success_count_by_source: dict[str, int] = {"TCGA": 0, "UPenn": 0, "LUMIERE": 0}

    t_start = time.time()
    for canonical, rows in candidates_by_source.items():
        for row in rows:
            if success_count_by_source[canonical] >= args.n_per_source:
                # Bu kaynak için hedef sayıda BAŞARILI (resolve+HAM+N4 tamam)
                # tarama zaten toplandı -- kalan adaylar (varsa) hiç
                # denenmez. Bu bir "silinen veri" değil, sadece fazladan
                # çekilmiş aday havuzunun kullanılmayan kısmı.
                break
            patient_id = row["patient_id"]
            scan_id = row["scan_id"]
            try:
                image_path = resolve_nas_path(row["file_path"])
                radiomics_image_path, regions = _resolve_regions(
                    canonical=canonical,
                    image_path=image_path,
                    patient_id=patient_id,
                    scan_id=scan_id,
                )
            except (
                FileNotFoundError,
                ReadyMaskNotFoundError,
                ImageMaskGeometryError,
                ValueError,
            ) as exc:
                skipped.append(
                    {
                        "patient_id": patient_id,
                        "scan_id": str(scan_id),
                        "source": canonical,
                        "stage": "resolve",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"ATLANDI (resolve): {canonical} {patient_id} scan_id={scan_id}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            ham_features_by_region: dict[str, dict[str, float]] = {}
            for region, spec in regions.items():
                try:
                    result = extractor.execute(
                        str(radiomics_image_path),
                        str(spec["mask_path"]),
                        label=int(spec["label"]),
                    )
                    ham_features_by_region[region] = _extract_numeric(result)
                except Exception as exc:  # noqa: BLE001
                    skipped.append(
                        {
                            "patient_id": patient_id,
                            "scan_id": str(scan_id),
                            "source": canonical,
                            "stage": f"ham_pyradiomics:{region}",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(
                        f"ATLANDI (HAM pyradiomics): {canonical} {patient_id} "
                        f"scan_id={scan_id} region={region}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )

            if not ham_features_by_region:
                continue

            try:
                n4_path = apply_n4_bias_correction(str(radiomics_image_path))
            except Exception as exc:  # noqa: BLE001
                skipped.append(
                    {
                        "patient_id": patient_id,
                        "scan_id": str(scan_id),
                        "source": canonical,
                        "stage": "n4",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"ATLANDI (N4): {canonical} {patient_id} scan_id={scan_id}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            n4_paths_by_source[canonical].append(n4_path)
            scan_records.append(
                {
                    "patient_id": patient_id,
                    "scan_id": scan_id,
                    "source": canonical,
                    "radiomics_image_path": str(radiomics_image_path),
                    "n4_path": n4_path,
                    "regions": regions,
                    "ham_features_by_region": ham_features_by_region,
                }
            )
            success_count_by_source[canonical] += 1
            elapsed = time.time() - t_start
            print(
                f"[HAM+N4 tamam] {canonical} {patient_id} scan_id={scan_id} "
                f"({len(ham_features_by_region)} bölge), gecen={elapsed:.0f}s",
                flush=True,
            )

    # ---- Pass 2: T1ce-only, kaynak-bazlı Z-score fit ----
    fit_stats: dict[str, dict[str, object]] = {}
    for canonical, n4_paths in n4_paths_by_source.items():
        if not n4_paths:
            print(f"UYARI: {canonical} için hiç N4 çıktısı yok, Z-score fit ATLANIYOR.", flush=True)
            continue
        record = fit_source_zscore_statistics(
            n4_paths, canonical, stats_path=TEMP_STATS_PATH
        )
        fit_stats[canonical] = record
        print(
            f"Z-score fit ({canonical}, T1ce-only, n={len(n4_paths)}): "
            f"mean={record['mean']:.6f} std={record['std']:.6f} "
            f"voxel_count={record['voxel_count']}",
            flush=True,
        )

    args.stats_report.parent.mkdir(parents=True, exist_ok=True)
    args.stats_report.write_text(
        json.dumps(
            {
                "note": (
                    "T1ce-only, bu 15-taramalik ornneklemle fit edilmis "
                    "kaynak-bazli Z-score istatistigi (uretim source_stats."
                    "json DEGIL, karsilastirma amacli TEMP)."
                ),
                "fit_stats": fit_stats,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ---- Pass 3: Z-score normalize + DOĞRU-SIRALI PyRadiomics ----
    for record in scan_records:
        canonical = record["source"]
        n4_path = record["n4_path"]
        patient_id = record["patient_id"]
        scan_id = record["scan_id"]
        if canonical not in fit_stats:
            print(
                f"ATLANDI (zscore-apply, fit yok): {canonical} {patient_id} "
                f"scan_id={scan_id}",
                flush=True,
            )
            record["harmonized_features_by_region"] = {}
            continue
        try:
            zscore_path = apply_zscore_normalization(n4_path, canonical)
        except Exception as exc:  # noqa: BLE001
            skipped.append(
                {
                    "patient_id": patient_id,
                    "scan_id": str(scan_id),
                    "source": canonical,
                    "stage": "zscore_apply",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"ATLANDI (Z-score apply): {canonical} {patient_id} "
                f"scan_id={scan_id}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            record["harmonized_features_by_region"] = {}
            continue

        record["zscore_path"] = zscore_path
        harmonized_by_region: dict[str, dict[str, float]] = {}
        regions = record["regions"]
        for region, spec in regions.items():
            if region not in record["ham_features_by_region"]:
                continue
            geometry = validate_image_mask_geometry(zscore_path, spec["mask_path"])
            if not geometry["geometry_match"]:
                skipped.append(
                    {
                        "patient_id": patient_id,
                        "scan_id": str(scan_id),
                        "source": canonical,
                        "stage": f"zscore_geometry:{region}",
                        "reason": "N4/Zscore sonrası görüntü/maske geometrisi eşleşmiyor",
                    }
                )
                print(
                    f"ATLANDI (Zscore geometri): {canonical} {patient_id} "
                    f"scan_id={scan_id} region={region}",
                    flush=True,
                )
                continue
            try:
                result = extractor.execute(
                    str(zscore_path), str(spec["mask_path"]), label=int(spec["label"])
                )
                harmonized_by_region[region] = _extract_numeric(result)
            except Exception as exc:  # noqa: BLE001
                skipped.append(
                    {
                        "patient_id": patient_id,
                        "scan_id": str(scan_id),
                        "source": canonical,
                        "stage": f"harmonized_pyradiomics:{region}",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"ATLANDI (harmonized pyradiomics): {canonical} {patient_id} "
                    f"scan_id={scan_id} region={region}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        record["harmonized_features_by_region"] = harmonized_by_region
        print(
            f"[DOGRU-SIRALI tamam] {canonical} {patient_id} scan_id={scan_id} "
            f"({len(harmonized_by_region)} bölge)",
            flush=True,
        )

    # ---- Karşılaştırma + rapor ----
    long_rows: list[dict[str, object]] = []
    summary_acc: dict[tuple[str, str], list[float]] = {}

    for record in scan_records:
        canonical = record["source"]
        patient_id = record["patient_id"]
        scan_id = record["scan_id"]
        ham_by_region = record["ham_features_by_region"]
        harmonized_by_region = record.get("harmonized_features_by_region", {})
        for region, ham_features in ham_by_region.items():
            harmonized_features = harmonized_by_region.get(region, {})
            all_feature_names = sorted(set(ham_features) | set(harmonized_features))
            for feature_name in all_feature_names:
                feature_class = _feature_class(feature_name)
                value_raw = ham_features.get(feature_name)
                value_harmonized = harmonized_features.get(feature_name)
                abs_diff = ""
                pct_diff = ""
                if value_raw is not None and value_harmonized is not None:
                    abs_diff = abs(value_harmonized - value_raw)
                    if value_raw != 0:
                        pct_diff = abs_diff / abs(value_raw) * 100.0
                    elif value_harmonized == 0:
                        pct_diff = 0.0
                    else:
                        pct_diff = ""  # 0'a bölme, tanımsız, boş bırak
                long_rows.append(
                    {
                        "patient_id": patient_id,
                        "scan_id": scan_id,
                        "source": canonical,
                        "region": region,
                        "feature_class": feature_class,
                        "feature_name": feature_name,
                        "value_raw_i": value_raw if value_raw is not None else "",
                        "value_harmonized_ii": (
                            value_harmonized if value_harmonized is not None else ""
                        ),
                        "abs_diff": abs_diff,
                        "pct_diff": pct_diff,
                    }
                )
                if isinstance(pct_diff, float):
                    summary_acc.setdefault((canonical, feature_class), []).append(pct_diff)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "patient_id",
            "scan_id",
            "source",
            "region",
            "feature_class",
            "feature_name",
            "value_raw_i",
            "value_harmonized_ii",
            "abs_diff",
            "pct_diff",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(long_rows)

    summary_rows = []
    for (canonical, feature_class), values in sorted(summary_acc.items()):
        summary_rows.append(
            {
                "source": canonical,
                "feature_class": feature_class,
                "n_values": len(values),
                "mean_pct_diff": statistics.mean(values),
                "median_pct_diff": statistics.median(values),
                "max_pct_diff": max(values),
                "min_pct_diff": min(values),
            }
        )
    with args.summary_report.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "source",
            "feature_class",
            "n_values",
            "mean_pct_diff",
            "median_pct_diff",
            "max_pct_diff",
            "min_pct_diff",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    skipped_report = OUTPUT_ROOT / "validation_skipped.csv"
    with skipped_report.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["patient_id", "scan_id", "source", "stage", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(skipped)

    print(
        f"\nBitti. {len(scan_records)} tarama işlendi, {len(long_rows)} özellik-satırı "
        f"karşılaştırıldı, {len(skipped)} adım atlandı.\n"
        f"Ana rapor: {args.report}\n"
        f"Özet rapor: {args.summary_report}\n"
        f"Atlanan adımlar: {skipped_report}\n"
        f"Z-score fit istatistikleri: {args.stats_report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
