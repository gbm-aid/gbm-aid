"""Ayirt edici test (2026-08-13 gorev talimati): binWidth/pipeline-sirasi
kararini beslemek icin, mevcut 15 taramalik ornseklemde (validate_raw_vs_
harmonized_sample.py ile AYNI 5 TCGA + 5 UPenn-GBM + 5 LUMIERE taramasi)
kontrollu yogunluk-olcegi ve bias-field perturbasyonlari altinda dort
extraction varyantinin (A/B/C32/C64) ozellik-kararliligini olcer.

Varyantlar (Codex talimati, "sessiz varsayilan" riskine dusmemek icin HER
varyantta extractor ayarlari ACIKCA set edilir -- binWidth/binCount,
normalize=False, imageTypes={'Original': {}}):
  A   : ham (perturbe edilmis) goruntu + binWidth=25 (mevcut uretim davranisi)
  B   : N4 -> binWidth=25 (Z-score YOK)
  C32 : N4 -> T1ce-only kaynak-bazli Z-score -> binCount=32
  C64 : N4 -> T1ce-only kaynak-bazli Z-score -> binCount=64

Perturbasyonlar (her tarama icin, TEK N4/Zscore-fit ile 4 kosul):
  scale0.4 : global yogunluk carpani 0.4x (ham goruntu uzerinde)
  scale1.0 : perturbasyonsuz baseline (Z-score fit istatistigi BU
             kosuldan, kaynak-bazli/T1ce-only, uretilir)
  scale2.5 : global yogunluk carpani 2.5x
  bias     : scale1.0 uzerine kontrollu, yumusak, cok-eksenli carpimsal
             bias-field (N4'un duzeltmesi gereken turden -- dogrusal
             gradyan, rastgele degil, deterministik/tekrarlanabilir)

Z-score fit stratejisi -- kaynak-bazli, SABIT (uretim tasarimiyla tutarli):
her kaynagin 5 taramasinin SADECE scale1.0 (baseline, bias'siz) N4
ciktilarindan T1ce-only bir (mean,std) fit edilir (`fit_source_zscore_
statistics()`, ayni `validate_raw_vs_harmonized_sample.py`'nin T1ce-only
cagirma deseni). Bu SABIT model, o kaynagin TUM perturbasyon kosullarina
(0.4x/1.0x/2.5x/bias) aynen uygulanir -- boylece test edilen soru net:
"tek bir taramanin yogunluk olcegi kaynagin fit edildigi tipik olcekten
sapinca, kaynak-bazli SABIT Z-score bunu duzeltebiliyor mu?"

KISITLAR (gorev talimati):
- DB'ye HICBIR YAZMA yapilmaz (`get_connection(readonly=True)`).
- Uretim `artifacts/week2/zscore/source_stats.json`'a DOKUNULMAZ -- ayri
  kaynak-bazli TEMP stats dosyasi kullanilir.
- Uretim run_pyradiomics_*.py scriptleri DEGISTIRILMEZ, bu ayri bir
  test scripti.
- Goruntu/maske geometrisi eslesmiyorsa PyRadiomics'e GECILMEZ (mimari
  kural) -- perturbasyonlar SADECE piksel yogunlugunu degistirir, grid
  (size/spacing/origin/direction) HIC degismez, bu yuzden geometri
  kontrolu bir kez (resolve asamasinda) yapilir, her perturbasyon icin
  TEKRAR EDILMEZ (geregi yok, CopyInformation ile grid aynen korunur).

Uzun-suren-arka-plan-sureci riskine karsi (bkz. takim/mert.md operasyonel
notu): bu script `--sources` ile kaynak-bazli parcalara bolunebilir
(orn. once TCGA, sonra UPenn, sonra LUMIERE, ayri cagrilarla) ve HER
taramanin bolgeleri tamamlaninca sonuc satirlari HEMEN diske flush edilir
(`--report` CSV'ye append modunda) -- kismi ilerleme her zaman diskte.

Bu script yalniz Python 3.10 ortaminda (pyradiomics 3.0.1) calisir.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import SimpleITK as sitk  # noqa: E402
from psycopg2.extras import RealDictCursor  # noqa: E402

from db_connection import get_connection  # noqa: E402
from pipeline.harmonization import (  # noqa: E402
    apply_n4_bias_correction,
    apply_zscore_normalization,
    fit_source_zscore_statistics,
    write_image_atomic,
)
from pipeline.resampling import validate_image_mask_geometry  # noqa: E402
from pipeline.radiomics_volume import REGION_LABELS_BY_MASK_SOURCE  # noqa: E402
from pipeline.segmentation import (  # noqa: E402
    ImageMaskGeometryError,
    ReadyMaskNotFoundError,
    _find_nifti,
    resolve_nas_path,
    resolve_ready_mask,
)

OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "week3" / "pyradiomics"
TMP_ROOT = OUTPUT_ROOT / "probe_perturbation_tmp"

TCGA_REGION_MASK_FILE = {
    "WT": ("whole.nii.gz", "whole.nii"),
    "TC": ("core.nii.gz", "core.nii"),
}

# AYNI 15 tarama, `validate_raw_vs_harmonized_sample.py`'nin 2026-08-13
# kosusunda secilen (source, patient_id, scan_id) uclusu -- tekrar DB
# aday-havuzu secimi YAPILMAZ, birebir ayni orneklem kullanilir.
FIXED_SAMPLE: list[tuple[str, str, int]] = [
    ("TCGA", "TCGA-02-0003", 2),
    ("TCGA", "TCGA-02-0006", 6),
    ("TCGA", "TCGA-02-0009", 9),
    ("TCGA", "TCGA-02-0011", 13),
    ("TCGA", "TCGA-02-0027", 17),
    ("UPenn", "UPENN-GBM-00001", 156),
    ("UPenn", "UPENN-GBM-00002", 160),
    ("UPenn", "UPENN-GBM-00003", 164),
    ("UPenn", "UPENN-GBM-00004", 168),
    ("UPenn", "UPENN-GBM-00005", 172),
    ("LUMIERE", "Patient-001", 2839),
    ("LUMIERE", "Patient-004", 2895),
    ("LUMIERE", "Patient-014", 3155),
    ("LUMIERE", "Patient-015", 3167),
    ("LUMIERE", "Patient-018", 3247),
]

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

PERTURBATIONS = ["scale0.4", "scale1.0", "scale2.5", "bias"]

REPORT_FIELDNAMES = [
    "source",
    "patient_id",
    "scan_id",
    "region",
    "roi_voxel_count",
    "perturbation",
    "extraction_variant",
    "feature_class",
    "feature_name",
    "value",
]


def _feature_class(name: str) -> str:
    for cls, prefixes in FEATURE_CLASS_PREFIXES.items():
        if name.startswith(prefixes):
            return cls
    return "unknown"


def _make_extractor(*, bin_width: int | None, bin_count: int | None):
    from radiomics import featureextractor

    logging.getLogger("radiomics").setLevel(logging.ERROR)
    kwargs: dict[str, object] = {"normalize": False}
    if bin_width is not None:
        kwargs["binWidth"] = bin_width
    if bin_count is not None:
        kwargs["binCount"] = bin_count
    extractor = featureextractor.RadiomicsFeatureExtractor(**kwargs)
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()
    for cls in ("shape", "firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm"):
        extractor.enableFeatureClassByName(cls)
    return extractor


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


def _fetch_file_path(connection, patient_id: str, scan_id: int) -> tuple[str, str]:
    cursor = connection.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(
            """
            SELECT ms.file_path, ms.modality
            FROM mr_scans ms
            WHERE ms.patient_id = %s AND ms.scan_id = %s
            """,
            (patient_id, scan_id),
        )
        row = cursor.fetchone()
    finally:
        cursor.close()
    if row is None:
        raise LookupError(f"Tarama DB'de bulunamadi: {patient_id} scan_id={scan_id}")
    return row["file_path"], row["modality"]


def _resolve_regions(
    *, canonical: str, image_path: Path, patient_id: str, scan_id: int
) -> tuple[Path, dict[str, dict[str, object]]]:
    """`validate_raw_vs_harmonized_sample.py`'deki BIREBIR ayni mantik."""

    if canonical == "TCGA":
        regions: dict[str, dict[str, object]] = {}
        for region, names in TCGA_REGION_MASK_FILE.items():
            mask_path = _find_nifti(image_path.parent, names)
            if mask_path is None:
                continue
            geometry = validate_image_mask_geometry(image_path, mask_path)
            if not geometry["geometry_match"]:
                raise ImageMaskGeometryError(
                    f"TCGA {region} goruntu/maske geometrisi eslesmiyor: "
                    f"{image_path} vs {mask_path}"
                )
            regions[region] = {
                "mask_path": mask_path,
                "label": 1,
                "mask_source": "provided_whole_tumor" if region == "WT" else "provided_tumor_core",
            }
        if not regions:
            raise ReadyMaskNotFoundError(
                f"TCGA hazir whole/core maskesi bulunamadi: {image_path.parent}"
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
            f"LUMIERE ana kaynak (DeepBraTumIA native) bulunamadi -- bu "
            f"test kanonik ana-kaynak kararina ({mask_source} degil) uyar."
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
            f"{canonical} maskesinde beklenen hicbir etiket bulunamadi: {mask_path}"
        )
    return resolved_image, regions


def _roi_voxel_count(mask_path: Path, label: int) -> int:
    mask = sitk.ReadImage(str(mask_path))
    array = sitk.GetArrayViewFromImage(mask)
    return int(np.count_nonzero(array == label))


def _make_bias_field(shape: tuple[int, int, int]) -> np.ndarray:
    """Deterministik, yumusak, cok-eksenli carpimsal bias alani.

    N4'un duzeltmeye calistigi turden dusuk-frekansli bir gradyan --
    Z ekseninde baskin (gercek coil-inhomojenitesine benzer, S-I
    ekseni), X ekseninde daha kucuk ikincil bileşen. Deger araligi
    yaklasik [0.3, 1.7] -- kucuk bir 25 binWidth ile bile ayirt
    edilebilecek buyuklukte, ama hala "yumusak"/dusuk-frekansli.
    """

    z, y, x = shape
    zz, _, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, z),
        np.linspace(-1.0, 1.0, y),
        np.linspace(-1.0, 1.0, x),
        indexing="ij",
    )
    field = 1.0 + 0.5 * zz + 0.2 * xx
    return field.astype(np.float32)


def _perturb_image(base_image: sitk.Image, perturbation: str) -> sitk.Image:
    array = sitk.GetArrayFromImage(base_image).astype(np.float32, copy=True)
    if perturbation == "scale0.4":
        out = array * np.float32(0.4)
    elif perturbation == "scale1.0":
        out = array
    elif perturbation == "scale2.5":
        out = array * np.float32(2.5)
    elif perturbation == "bias":
        field = _make_bias_field(array.shape)
        out = array * field
    else:
        raise ValueError(f"Bilinmeyen perturbasyon: {perturbation}")
    image = sitk.GetImageFromArray(out)
    image.CopyInformation(base_image)
    return image


def _write_temp_nifti(image: sitk.Image, label: str) -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = TMP_ROOT / f"{label}_{uuid.uuid4().hex}.nii.gz"
    write_image_atomic(image, output_path)
    return output_path


def _append_rows(report_path: Path, rows: list[dict[str, object]], *, write_header: bool) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if report_path.is_file() and not write_header else "w"
    with report_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ayirt edici test: 15 taramalik ornseklemde yogunluk-olcegi + "
            "bias-field perturbasyonlarina karsi A/B/C32/C64 ozellik "
            "kararliligi (DB'ye yazmaz)."
        )
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="TCGA,UPenn,LUMIERE",
        help="Virgulle ayrilmis kaynak alt-kumesi (parca-parca kosu icin).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=OUTPUT_ROOT / "probe_perturbation_stability_long.csv",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Rapor dosyasini bastan yaz (varsayilan: append, kaldigi yerden devam).",
    )
    args = parser.parse_args()

    requested_sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    for source in requested_sources:
        if source not in SOURCE_DB_NAME:
            raise ValueError(f"Bilinmeyen kaynak: {source!r}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    # KRITIK (2026-08-13'te kosu-sonrasi kesfedildi, bu satir eksikti):
    # bu ortam degiskeni ayarlanmazsa apply_n4_bias_correction()/
    # apply_zscore_normalization() N4/Zscore ciktilarini VARSAYILAN
    # `artifacts/harmonized/<hash>` kokune yazar -- bu, uretim `artifacts/
    # week2/n4`/`artifacts/week2/zscore` klasorlerinden AYRI ama paylasimli/
    # scratch bir varsayilan (baska eski smoke-testlerin de kullandigi).
    # Kendi ayri TMP_ROOT'umuza yazdirmak icin acikca set ediyoruz --
    # `validate_raw_vs_harmonized_sample.py`'nin deseniyle tutarli. (Bu
    # script'in ilk kosusunda bu satir EKSIKTI, 60 scratch N4/Zscore ciktisi
    # `artifacts/harmonized/`'e yazildi -- kosu sonrasi hash-eslesmesiyle
    # dogrulanip TEMIZLENDI, uretim verisine dokunulmadi, bkz. log/2026-08-13.md.)
    os.environ["GBMAID_PROCESSED_ROOT"] = str(TMP_ROOT)

    # Zaten islenmis (source,patient_id,scan_id) uclulerini atlamak icin
    # mevcut raporu oku (idempotent devam etme -- uzun-surme riskine karsi).
    already_done: set[tuple[str, str, int]] = set()
    if args.report.is_file() and not args.fresh:
        with args.report.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                already_done.add(
                    (row["source"], row["patient_id"], int(row["scan_id"]))
                )
        print(
            f"Mevcut rapor bulundu, {len(already_done)} tarama-satiri zaten "
            f"islenmis gorunuyor (kaba kontrol, asagida scan-bazinda tekrar "
            f"dogrulanacak): {args.report}",
            flush=True,
        )

    write_header = args.fresh or not args.report.is_file()

    extractor_a = _make_extractor(bin_width=25, bin_count=None)
    extractor_b = _make_extractor(bin_width=25, bin_count=None)
    extractor_c32 = _make_extractor(bin_width=None, bin_count=32)
    extractor_c64 = _make_extractor(bin_width=None, bin_count=64)

    t_start = time.time()

    for canonical in requested_sources:
        scans = [row for row in FIXED_SAMPLE if row[0] == canonical]
        stats_path = OUTPUT_ROOT / f"probe_perturbation_zscore_stats_TEMP_{canonical}.json"
        if stats_path.is_file():
            stats_path.unlink()
        os.environ["GBMAID_ZSCORE_STATS_PATH"] = str(stats_path)

        print(f"\n=== Kaynak: {canonical} ({len(scans)} tarama) ===", flush=True)

        connection = get_connection(readonly=True)
        try:
            resolved_scans: list[dict[str, object]] = []
            for _, patient_id, scan_id in scans:
                scan_ids_done = {
                    sid for src, pid, sid in already_done if src == canonical and pid == patient_id
                }
                if scan_id in scan_ids_done:
                    print(
                        f"  ATLANDI (zaten raporda): {patient_id} scan_id={scan_id}",
                        flush=True,
                    )
                    continue
                try:
                    file_path, modality = _fetch_file_path(connection, patient_id, scan_id)
                    image_path = resolve_nas_path(file_path)
                    radiomics_image_path, regions = _resolve_regions(
                        canonical=canonical,
                        image_path=image_path,
                        patient_id=patient_id,
                        scan_id=scan_id,
                    )
                except (
                    LookupError,
                    FileNotFoundError,
                    ReadyMaskNotFoundError,
                    ImageMaskGeometryError,
                    ValueError,
                ) as exc:
                    print(
                        f"  ATLANDI (resolve): {patient_id} scan_id={scan_id}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                voxel_counts = {
                    region: _roi_voxel_count(spec["mask_path"], int(spec["label"]))
                    for region, spec in regions.items()
                }
                resolved_scans.append(
                    {
                        "patient_id": patient_id,
                        "scan_id": scan_id,
                        "image_path": radiomics_image_path,
                        "regions": regions,
                        "voxel_counts": voxel_counts,
                    }
                )
        finally:
            connection.close()

        if not resolved_scans:
            print(f"  {canonical}: islenecek yeni tarama yok.", flush=True)
            continue

        # ---- Faz 1: baseline (scale1.0) N4, Z-score fit icin ----
        baseline_n4_paths: list[str] = []
        for scan in resolved_scans:
            base_image = sitk.Cast(
                sitk.ReadImage(str(scan["image_path"])), sitk.sitkFloat32
            )
            scan["base_image"] = base_image
            tmp_path = _write_temp_nifti(
                base_image, f"{canonical}_{scan['patient_id']}_baseline"
            )
            n4_path = apply_n4_bias_correction(str(tmp_path))
            scan["n4_baseline_path"] = n4_path
            baseline_n4_paths.append(n4_path)
            elapsed = time.time() - t_start
            print(
                f"  [baseline N4] {scan['patient_id']} scan_id={scan['scan_id']} "
                f"gecen={elapsed:.0f}s",
                flush=True,
            )

        fit_record = fit_source_zscore_statistics(
            baseline_n4_paths, canonical, stats_path=stats_path
        )
        print(
            f"  Z-score fit ({canonical}, T1ce-only baseline, n="
            f"{len(baseline_n4_paths)}): mean={fit_record['mean']:.6f} "
            f"std={fit_record['std']:.6f}",
            flush=True,
        )

        # ---- Faz 2: 4 perturbasyon kosulu x 4 extraction varyanti ----
        for scan in resolved_scans:
            patient_id = scan["patient_id"]
            scan_id = scan["scan_id"]
            regions = scan["regions"]
            voxel_counts = scan["voxel_counts"]
            base_image = scan["base_image"]

            rows: list[dict[str, object]] = []

            for perturbation in PERTURBATIONS:
                perturbed_image = _perturb_image(base_image, perturbation)

                # ---- A: ham (perturbe) goruntu, binWidth=25 ----
                for region, spec in regions.items():
                    try:
                        result = extractor_a.execute(
                            perturbed_image, str(spec["mask_path"]), label=int(spec["label"])
                        )
                        features = _extract_numeric(result)
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"    HATA (A) {patient_id} {perturbation} {region}: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        continue
                    for feature_name, value in features.items():
                        rows.append(
                            {
                                "source": canonical,
                                "patient_id": patient_id,
                                "scan_id": scan_id,
                                "region": region,
                                "roi_voxel_count": voxel_counts[region],
                                "perturbation": perturbation,
                                "extraction_variant": "A",
                                "feature_class": _feature_class(feature_name),
                                "feature_name": feature_name,
                                "value": value,
                            }
                        )

                # ---- N4 (scale1.0 icin baseline'i tekrar kullan) ----
                if perturbation == "scale1.0":
                    n4_path = scan["n4_baseline_path"]
                else:
                    tmp_path = _write_temp_nifti(
                        perturbed_image, f"{canonical}_{patient_id}_{perturbation}"
                    )
                    try:
                        n4_path = apply_n4_bias_correction(str(tmp_path))
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"    HATA (N4) {patient_id} {perturbation}: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        continue

                n4_image = sitk.ReadImage(n4_path)
                # ---- B: N4, binWidth=25, Z-score YOK ----
                for region, spec in regions.items():
                    try:
                        result = extractor_b.execute(
                            n4_image, str(spec["mask_path"]), label=int(spec["label"])
                        )
                        features = _extract_numeric(result)
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"    HATA (B) {patient_id} {perturbation} {region}: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        continue
                    for feature_name, value in features.items():
                        rows.append(
                            {
                                "source": canonical,
                                "patient_id": patient_id,
                                "scan_id": scan_id,
                                "region": region,
                                "roi_voxel_count": voxel_counts[region],
                                "perturbation": perturbation,
                                "extraction_variant": "B",
                                "feature_class": _feature_class(feature_name),
                                "feature_name": feature_name,
                                "value": value,
                            }
                        )

                # ---- C: N4 -> Z-score (SABIT baseline model) -> C32/C64 ----
                try:
                    zscore_path = apply_zscore_normalization(n4_path, canonical)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"    HATA (zscore) {patient_id} {perturbation}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                zscore_image = sitk.ReadImage(zscore_path)
                for extractor, variant_name in (
                    (extractor_c32, "C32"),
                    (extractor_c64, "C64"),
                ):
                    for region, spec in regions.items():
                        try:
                            result = extractor.execute(
                                zscore_image, str(spec["mask_path"]), label=int(spec["label"])
                            )
                            features = _extract_numeric(result)
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"    HATA ({variant_name}) {patient_id} "
                                f"{perturbation} {region}: {type(exc).__name__}: {exc}",
                                flush=True,
                            )
                            continue
                        for feature_name, value in features.items():
                            rows.append(
                                {
                                    "source": canonical,
                                    "patient_id": patient_id,
                                    "scan_id": scan_id,
                                    "region": region,
                                    "roi_voxel_count": voxel_counts[region],
                                    "perturbation": perturbation,
                                    "extraction_variant": variant_name,
                                    "feature_class": _feature_class(feature_name),
                                    "feature_name": feature_name,
                                    "value": value,
                                }
                            )

                elapsed = time.time() - t_start
                print(
                    f"    [{perturbation}] tamam, gecen={elapsed:.0f}s, "
                    f"birikmis satir={len(rows)}",
                    flush=True,
                )

            _append_rows(args.report, rows, write_header=write_header)
            write_header = False
            elapsed = time.time() - t_start
            print(
                f"  [tarama tamam+flush] {patient_id} scan_id={scan_id} "
                f"({len(rows)} satir), gecen={elapsed:.0f}s -> {args.report}",
                flush=True,
            )

    print(f"\nBitti. Rapor: {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
