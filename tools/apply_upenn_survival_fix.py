"""UPenn survival_days / vital_status duzeltmesi -- patients tablosu UPDATE.

Kapsam: scratch_upenn_survival_fix_dryrun.py (proje koku) ile ayni hesaplama
mantigini kullanir (COALESCE: Survival_from_surgery_days_UPDATED yoksa
Survival_Censor'a dus; Lost to Follow-up / Deceased - uncertain date of
death -> CENSORED), ama --apply verildiginde gercekten UPDATE calistirir.

Satir-bazli commit/rollback deseni write_upenn_pyradiomics_to_db.py ile
ayni (bkz. decisions/2026-08-09-... -- tek-transaction commit yuzunden
1 satirlik hatanin TUM basarili satirlari sessizce rollback ettigi bug
bir onceki oturumda burada yakalanmisti, o yuzden bu script satir-bazli
commit kullanir).

Varsayilan: DRY-RUN (DB'ye yazilmaz). --apply verilmeden calistirilirsa
sadece rapor CSV'si uretilir, DB'ye dokunulmaz.
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

from db_connection import get_connection  # noqa: E402

REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_CSV = REPO_ROOT / "raw" / "veri" / "UPENN-GBM_clinical_info_v2.1 database e girecek.csv"
DEFAULT_REPORT = PROJECT_ROOT / "artifacts" / "week2" / "upenn_survival_fix_report.csv"

VITAL_STATUS_MAP = {
    "Deceased": "DECEASED",
    "Alive": "ALIVE",
    "Lost to Follow-up": "CENSORED",
    "Deceased - uncertain date of death": "CENSORED",
}


def _safe_int(val: str | None) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def normalize_vital_status(val: str) -> str:
    return VITAL_STATUS_MAP.get(val, val)


def compute_survival_days(row: dict) -> int | None:
    primary = _safe_int(row["Survival_from_surgery_days_UPDATED"])
    if primary is not None:
        return primary
    return _safe_int(row["Survival_Censor"])


def load_corrected_values(csv_path: Path) -> dict[str, dict]:
    with csv_path.open(encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    by_base: dict[str, dict[str, dict]] = {}
    for row in raw_rows:
        full_id = row["ID"]
        if "_" not in full_id:
            continue
        base_id, suffix = full_id.rsplit("_", 1)
        by_base.setdefault(base_id, {})[suffix] = row

    corrected = {}
    for base_id, variants in by_base.items():
        row = variants.get("11") or variants.get("21")
        corrected[base_id] = {
            "survival_days": compute_survival_days(row),
            "vital_status": normalize_vital_status(row["Survival_Status"]),
            "raw_status": row["Survival_Status"],
        }
    return corrected


def _update_row(connection, *, patient_id: str, vital_status: str, survival_days: int | None) -> bool:
    """UPDATE dener, gercekten bir satir degistiyse True doner."""
    cursor = connection.cursor()
    try:
        cursor.execute(
            "UPDATE patients SET vital_status = %s, survival_days = %s WHERE patient_id = %s",
            (vital_status, survival_days, patient_id),
        )
        return cursor.rowcount == 1
    finally:
        cursor.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="UPenn patients.survival_days/vital_status duzeltmesi (27+10 hasta beklenen)."
    )
    parser.add_argument("--apply", action="store_true", help="Belirtilmezse DB'ye yazilmaz (dry-run varsayilan).")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    corrected = load_corrected_values(args.csv)

    connection = get_connection(readonly=not args.apply)
    rows_out: list[dict[str, object]] = []
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT p.patient_id, p.vital_status, p.survival_days
            FROM patients p
            JOIN dataset_sources ds ON p.source_id = ds.source_id
            WHERE ds.source_name = 'UPenn-GBM'
            """
        )
        db_rows = {r[0]: {"vital_status": r[1], "survival_days": r[2]} for r in cursor.fetchall()}
        cursor.close()

        for patient_id, cur_vals in db_rows.items():
            new_vals = corrected.get(patient_id)
            if new_vals is None:
                continue
            changed = (cur_vals["vital_status"] != new_vals["vital_status"]) or (
                cur_vals["survival_days"] != new_vals["survival_days"]
            )
            if not changed:
                continue

            out = {
                "patient_id": patient_id,
                "before_vital": cur_vals["vital_status"],
                "after_vital": new_vals["vital_status"],
                "before_days": cur_vals["survival_days"],
                "after_days": new_vals["survival_days"],
                "raw_status": new_vals["raw_status"],
                "db_action": "",
            }

            try:
                if args.apply:
                    updated = _update_row(
                        connection,
                        patient_id=patient_id,
                        vital_status=new_vals["vital_status"],
                        survival_days=new_vals["survival_days"],
                    )
                    connection.commit()  # satir-bazli commit: bir hata diger satirlari bozmasin
                    out["db_action"] = "YAZILDI" if updated else "ATLANDI_PATIENT_ID_BULUNAMADI"
                else:
                    out["db_action"] = "DRY_RUN"
            except Exception as exc:  # noqa: BLE001
                connection.rollback()  # bu satirin transaction'ini temizle, sonraki satirlar etkilenmesin
                out["db_action"] = f"HATA:{type(exc).__name__}:{exc}"[:150]

            rows_out.append(out)
    finally:
        connection.close()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "patient_id", "before_vital", "after_vital", "before_days", "after_days",
            "raw_status", "db_action",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    counts = Counter(r["db_action"] for r in rows_out)
    vital_changed = sum(1 for r in rows_out if r["before_vital"] != r["after_vital"])
    days_changed = sum(1 for r in rows_out if r["before_days"] != r["after_days"])
    print(f"TOPLAM ETKILENEN SATIR: {len(rows_out)}")
    print(f"vital_status degisen: {vital_changed}, survival_days degisen: {days_changed}")
    print(f"Dagilim: {dict(counts)}")
    print(f"Rapor: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
