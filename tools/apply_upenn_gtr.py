"""UPenn patients.gtr_over90percent kolonu ekleme + doldurma.

Kaynak: UPENN-GBM_clinical_info_v2.1 database e girecek.csv (671 satir =
611 preop [_11] + 60 post-op [_21]). Karar (Baris, 2026-08-15, olcumle
dogrulandi + decisions/2026-08-06-upenn-dedup-11-21.md'nin kilitli _11
onceligiyle tutarli):

  - GTR_over90percent TARAMA-seviyesi bir alan, sadece _11 (baseline/preop)
    satirinda anlamli. _21 satirlarinin TAMAMI (60/60, istisnasiz) GTR =
    'Not Applicable' -- bu bir hasta bulgusu DEGIL, satir-seviyesi yer
    tutucu (olculdu: _21'lerin Time_since_baseline_preop'u hep >0/takip
    gunu, _11'lerin hepsi 0).
  - Deger eslemesi:
      Y             -> 'Y'
      N             -> 'N'
      Not Available -> NULL   (gercekten eksik veri, _11'de)
      (sadece _21'i olan 19 hasta, UPENN-GBM-00612..00630, hic _11 yok)
                    -> NULL   (GTR bilinmiyor)
  - LUMIERE/TCGA hastalarina DOKUNULMAZ, NULL kalir.

Beklenen (630 UPenn hastasi): Y=362, N=211, NULL=57 (38 Not Available +
19 sadece-_21).

Varsayilan: DRY-RUN (DB'ye yazilmaz, --apply verilmeden ALTER TABLE/UPDATE
calismaz). Sema degisikligi (ALTER TABLE ADD COLUMN) SADECE --apply modunda
ve SADECE kolon henuz yoksa calisir.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from db_connection import get_connection  # noqa: E402

REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_CSV = REPO_ROOT / "raw" / "veri" / "UPENN-GBM_clinical_info_v2.1 database e girecek.csv"
DEFAULT_REPORT = PROJECT_ROOT / "artifacts" / "week3" / "upenn_gtr_report.csv"

GTR_VALUE_MAP = {
    "Y": "Y",
    "N": "N",
    "Not Available": None,
    "Not Applicable": None,  # yalniz _21'de gorulur, satir-seviyesi yer tutucu
}


def load_gtr_values(csv_path: Path) -> dict[str, str | None]:
    """base_id (patient_id, suffix atilmis) -> gtr degeri (None dahil).

    Kural: _11 (preop) satiri VARSA onun degeri kullanilir (Not Available
    dahil -> NULL). _11 YOKSA (sadece _21 olan 19 hasta) NULL.
    """
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    by_base: dict[str, dict[str, dict]] = {}
    for row in raw_rows:
        full_id = row["ID"].strip()
        if "_" not in full_id:
            continue
        base_id, suffix = full_id.rsplit("_", 1)
        by_base.setdefault(base_id, {})[suffix] = row

    result: dict[str, str | None] = {}
    for base_id, variants in by_base.items():
        row_11 = variants.get("11")
        if row_11 is not None:
            raw_val = row_11.get("GTR_over90percent", "").strip()
            result[base_id] = GTR_VALUE_MAP.get(raw_val, None)
        else:
            # sadece _21 var -> GTR bilinmiyor
            result[base_id] = None
    return result


def column_exists(connection) -> bool:
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'patients' AND column_name = 'gtr_over90percent'
            """
        )
        return cursor.fetchone() is not None
    finally:
        cursor.close()


def ensure_column(connection) -> str:
    """ALTER TABLE calistirir (sadece --apply modunda cagrilir). Idempotent."""
    if column_exists(connection):
        return "ZATEN_VAR"
    cursor = connection.cursor()
    try:
        cursor.execute("ALTER TABLE patients ADD COLUMN gtr_over90percent TEXT;")
        connection.commit()
        return "EKLENDI"
    finally:
        cursor.close()


def _update_row(connection, *, patient_id: str, gtr_value: str | None) -> bool:
    cursor = connection.cursor()
    try:
        cursor.execute(
            "UPDATE patients SET gtr_over90percent = %s WHERE patient_id = %s",
            (gtr_value, patient_id),
        )
        return cursor.rowcount == 1
    finally:
        cursor.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="UPenn patients.gtr_over90percent kolon ekleme + doldurma (630 hasta beklenen)."
    )
    parser.add_argument("--apply", action="store_true", help="Belirtilmezse DB'ye yazilmaz (dry-run varsayilan).")
    parser.add_argument("--limit", type=int, default=None, help="Sadece ilk N hastayi isle (kucuk olcek testi).")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    gtr_values = load_gtr_values(args.csv)

    connection = get_connection(readonly=not args.apply)
    rows_out: list[dict[str, object]] = []
    schema_action = "DRY_RUN_ATLANDI"
    try:
        if args.apply:
            schema_action = ensure_column(connection)
        else:
            schema_action = "ZATEN_VAR" if column_exists(connection) else "EKLENECEK (dry-run, calistirilmadi)"

        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT p.patient_id
            FROM patients p
            JOIN dataset_sources ds ON p.source_id = ds.source_id
            WHERE ds.source_name = 'UPenn-GBM'
            ORDER BY p.patient_id
            """
        )
        db_patient_ids = [r[0] for r in cursor.fetchall()]
        cursor.close()

        if args.limit is not None:
            db_patient_ids = db_patient_ids[: args.limit]

        for patient_id in db_patient_ids:
            if patient_id not in gtr_values:
                out = {
                    "patient_id": patient_id,
                    "csv_gtr_raw": "",
                    "new_gtr_value": "",
                    "db_action": "ATLANDI_CSV_ID_BULUNAMADI",
                }
                rows_out.append(out)
                continue

            new_val = gtr_values[patient_id]
            out = {
                "patient_id": patient_id,
                "csv_gtr_raw": new_val if new_val is not None else "NULL",
                "new_gtr_value": new_val if new_val is not None else "NULL",
                "db_action": "",
            }

            try:
                if args.apply:
                    updated = _update_row(connection, patient_id=patient_id, gtr_value=new_val)
                    connection.commit()
                    out["db_action"] = "YAZILDI" if updated else "ATLANDI_PATIENT_ID_BULUNAMADI"
                else:
                    out["db_action"] = "DRY_RUN"
            except Exception as exc:  # noqa: BLE001
                connection.rollback()
                out["db_action"] = f"HATA:{type(exc).__name__}:{exc}"[:150]

            rows_out.append(out)
    finally:
        connection.close()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["patient_id", "csv_gtr_raw", "new_gtr_value", "db_action"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    action_counts = Counter(r["db_action"] for r in rows_out)
    value_counts = Counter(r["new_gtr_value"] for r in rows_out)
    print(f"SEMA DURUMU: {schema_action}")
    print(f"TOPLAM ISLENEN HASTA: {len(rows_out)}")
    print(f"db_action dagilimi: {dict(action_counts)}")
    print(f"Yeni deger dagilimi: {dict(value_counts)}")
    print(f"Rapor: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
