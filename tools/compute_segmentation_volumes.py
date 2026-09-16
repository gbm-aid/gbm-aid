"""TCGA hazır whole/core maskelerinden WT/TC hacmi hesapla, radiomics'e ön-yaz.

Kapsam: yalnız TCGA-ground-truth. UPenn (CaPTk precomputed) ve LUMIERE
(DeepBraTumIA/HD-GLIO precomputed) zaten kendi hacimleriyle birlikte
Supabase'e yüklü -- bu script onlara dokunmaz.

Var olan (scan_id, 'TCGA-ground-truth', tumor_region) satırları (TCGA-02-0003
ve TCGA-02-0006, Hafta 2'de elle işlendi) hiçbir koşulda INSERT edilmez;
script bunları atlar. Yeni satırlar shape_features/first_order_features/
texture_features = NULL bırakılır -- Mert'in Hafta 3 PyRadiomics çıktısı
bu satırları (scan_id, segmentation_tool, tumor_region) anahtarıyla UPDATE
edecek (hafta2_durum_ozeti.md, madde 4).

Varsayılan mod --dry-run: DB'ye hiçbir yazma yapmaz, yalnız CSV rapor üretir.
Gözden geçirme sonrası --apply ile gerçek INSERT çalıştırılır.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from psycopg2.extras import RealDictCursor  # noqa: E402

from db_connection import get_connection  # noqa: E402
from pipeline.harmonization import canonical_source  # noqa: E402
from pipeline.radiomics_volume import compute_volumes_for_mask_source  # noqa: E402
from pipeline.resampling import validate_image_mask_geometry  # noqa: E402
from pipeline.segmentation import _find_nifti, resolve_nas_path  # noqa: E402

SEGMENTATION_TOOL = "TCGA-ground-truth"
REGION_MASK_SOURCE = {
    "WT": "provided_whole_tumor",
    "TC": "provided_tumor_core",
}


def _fetch_tcga_t1ce_scans(connection) -> list[dict]:
    cursor = connection.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(
            "SELECT ms.scan_id, ms.patient_id, ms.file_path, ds.source_name AS source "
            "FROM mr_scans AS ms "
            "JOIN patients AS p ON p.patient_id = ms.patient_id "
            "JOIN dataset_sources AS ds ON ds.source_id = p.source_id "
            "WHERE ds.source_name = 'TCGA-GBM' AND ms.modality = 'T1ce' "
            "ORDER BY ms.scan_id"
        )
        return list(cursor.fetchall())
    finally:
        cursor.close()


def _existing_regions(connection, scan_id: int) -> set[str]:
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT tumor_region FROM radiomics "
            "WHERE scan_id = %s AND segmentation_tool = %s",
            (scan_id, SEGMENTATION_TOOL),
        )
        return {row[0] for row in cursor.fetchall()}
    finally:
        cursor.close()


def _resolve_region_mask(image_path: Path, region: str) -> Path | None:
    if region == "WT":
        return _find_nifti(image_path.parent, ("whole.nii.gz", "whole.nii"))
    if region == "TC":
        return _find_nifti(image_path.parent, ("core.nii.gz", "core.nii"))
    raise ValueError(f"Bilinmeyen bölge: {region}")


def _insert_volume_row(connection, *, scan_id: int, region: str, volume_mm3: float) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO radiomics "
            "(scan_id, segmentation_tool, tumor_region, tumor_volume_mm3, feature_source) "
            "VALUES (%s, %s, %s, %s, 'computed') "
            "ON CONFLICT (scan_id, segmentation_tool, tumor_region) DO NOTHING",
            (scan_id, SEGMENTATION_TOOL, region, volume_mm3),
        )
    finally:
        cursor.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TCGA whole/core maske hacimlerini hesapla ve radiomics'e ön-yaz."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Belirtilmezse hiçbir DB yazımı yapılmaz (dry-run varsayılan).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "week2" / "volumes" / "tcga_volumes.csv",
        help="CSV rapor çıktı yolu.",
    )
    args = parser.parse_args()

    connection = get_connection(readonly=not args.apply)
    rows: list[dict[str, object]] = []
    try:
        scans = _fetch_tcga_t1ce_scans(connection)
        for scan in scans:
            scan_id = scan["scan_id"]
            patient_id = scan["patient_id"]
            already_done = _existing_regions(connection, scan_id)
            try:
                image_path = resolve_nas_path(scan["file_path"])
            except FileNotFoundError as exc:
                rows.append(
                    {
                        "patient_id": patient_id,
                        "scan_id": scan_id,
                        "region": "-",
                        "status": f"GORUNTU_YOK: {exc}",
                        "volume_mm3": "",
                    }
                )
                continue

            for region, mask_source in REGION_MASK_SOURCE.items():
                if region in already_done:
                    rows.append(
                        {
                            "patient_id": patient_id,
                            "scan_id": scan_id,
                            "region": region,
                            "status": "ZATEN_VAR_ATLANDI",
                            "volume_mm3": "",
                        }
                    )
                    continue

                mask_path = _resolve_region_mask(image_path, region)
                if mask_path is None:
                    rows.append(
                        {
                            "patient_id": patient_id,
                            "scan_id": scan_id,
                            "region": region,
                            "status": "MASKE_YOK",
                            "volume_mm3": "",
                        }
                    )
                    continue

                geometry = validate_image_mask_geometry(image_path, mask_path)
                if not geometry["geometry_match"]:
                    rows.append(
                        {
                            "patient_id": patient_id,
                            "scan_id": scan_id,
                            "region": region,
                            "status": "GEOMETRI_UYUSMUYOR",
                            "volume_mm3": "",
                        }
                    )
                    continue

                volumes = compute_volumes_for_mask_source(mask_path, mask_source)
                volume_mm3 = volumes[region]

                status = "DRY_RUN"
                if args.apply:
                    _insert_volume_row(
                        connection, scan_id=scan_id, region=region, volume_mm3=volume_mm3
                    )
                    status = "YAZILDI"

                rows.append(
                    {
                        "patient_id": patient_id,
                        "scan_id": scan_id,
                        "region": region,
                        "status": status,
                        "volume_mm3": f"{volume_mm3:.4f}",
                    }
                )

        if args.apply:
            connection.commit()
    finally:
        connection.close()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["patient_id", "scan_id", "region", "status", "volume_mm3"]
        )
        writer.writeheader()
        writer.writerows(rows)

    written = sum(1 for row in rows if row["status"] in ("YAZILDI", "DRY_RUN"))
    print(f"{len(rows)} satır işlendi, {written} hacim hesaplandı. Rapor: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
