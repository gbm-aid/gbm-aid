"""mr_scans.harmonization_status UPDATE -- N4+Z-score harmonizasyon sonuclarinin DB'ye yazilmasi.

Kapsam: TCGA-GBM, UPenn-GBM VE LUMIERE. [2026-08-12 GUNCELLEME] LUMIERE
ilk yazilis sirasinda ("Kapsam: SADECE TCGA-GBM ve UPenn-GBM") kasitli
olarak DISI birakilmisti (imaging-agent sureci henuz bitmemisti). Simdi
LUMIERE tam kohort (2396/2396 scan_id) cozumlendi -- 2152 n4_zscore_ok +
244 skipped_not_evaluable (kalici/bilincli, harmonizasyon UYGULANMADI,
bu script tarafindan hic islenmez cunku status filtresi zaten sadece
n4_zscore_ok'u kapsiyor). ALLOWED_SOURCES bu yuzden LUMIERE'i de icerecek
sekilde genisletildi (db-agent, Baris onayiyla, decisions/ ve
AKTIF-GOREVLER.md'de kayitli).

Girdi: gbm-aid mert/artifacts/week2/harmonization_log.csv
(kolonlar: scan_id,patient_id,dataset_source,status,hata_mesaji,timestamp,detay)
Bu CSV imaging-agent (Mert) tarafindan uretiliyor/append ediliyor; bu script
SADECE OKUR, hicbir sekilde uzerine yazmaz/duzenlemez.

Deger karari (mr_scans_harmonization_status_check CHECK constraint):
    CHECK (harmonization_status = ANY (ARRAY['raw','n4','zscore','combat']))
CSV'deki "n4_zscore_ok" durumu hem N4 bias-field duzeltmesi HEM Z-score
normalizasyonunun tamamlandigini ifade ediyor. Constraint'in izin verdigi
kume icinde bu iki adimi "n4"den daha ileri temsil eden tek deger "zscore"
(raw -> n4 -> zscore -> combat sirali pipeline varsayimi, DB'de zaten
kullanilan "n4" tek basina N4-only anlamina geliyor). Bu yuzden hedef deger
sabit olarak "zscore" secildi -- yeni bir deger EKLENMEDI (constraint
degistirilmedi, ONAY gerektiren sema degisikligi bu scriptin kapsami
DISINDA).

Satir-bazli commit/rollback deseni apply_upenn_survival_fix.py ile AYNI.

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
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from db_connection import get_connection  # noqa: E402

DEFAULT_CSV = PROJECT_ROOT / "artifacts" / "week2" / "harmonization_log.csv"
DEFAULT_REPORT = PROJECT_ROOT / "artifacts" / "week2" / "harmonization_status_db_update_report.csv"

# Bu scriptin kapsamindaki kaynaklar -- [2026-08-12] LUMIERE eklendi (kohort tamamlandi).
ALLOWED_SOURCES = {"TCGA-GBM", "UPenn-GBM", "LUMIERE"}
SOURCE_STATUS = "n4_zscore_ok"
TARGET_HARMONIZATION_STATUS = "zscore"


def load_csv_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    filtered = [
        row
        for row in rows
        if row["dataset_source"] in ALLOWED_SOURCES and row["status"] == SOURCE_STATUS
    ]
    skipped_sources = Counter(
        row["dataset_source"] for row in rows if row["dataset_source"] not in ALLOWED_SOURCES
    )
    if skipped_sources:
        print(f"[bilgi] Kapsam disi birakilan kaynaklar (CSV'de var, DOKUNULMADI): {dict(skipped_sources)}")
    return filtered


def _update_row(connection, *, scan_id: int, new_status: str) -> bool:
    """UPDATE dener, gercekten bir satir degistiyse True doner."""
    cursor = connection.cursor()
    try:
        cursor.execute(
            "UPDATE mr_scans SET harmonization_status = %s WHERE scan_id = %s",
            (new_status, scan_id),
        )
        return cursor.rowcount == 1
    finally:
        cursor.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "mr_scans.harmonization_status UPDATE -- TCGA-GBM+UPenn-GBM+LUMIERE icin "
            "n4_zscore_ok -> 'zscore' (TCGA+UPenn icin 2838 scan_id [154+2684] daha once "
            "--apply ile yazildi/tamamlandi; LUMIERE icin 2152 scan_id ayrica bekleniyor, "
            "kalan 244 LUMIERE scan_id 'skipped_not_evaluable' bu scriptin filtresine hic "
            "girmiyor)."
        )
    )
    parser.add_argument("--apply", action="store_true", help="Belirtilmezse DB'ye yazilmaz (dry-run varsayilan).")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    csv_rows = load_csv_rows(args.csv)
    scan_ids = [int(row["scan_id"]) for row in csv_rows]
    csv_by_scan_id = {int(row["scan_id"]): row for row in csv_rows}

    connection = get_connection(readonly=not args.apply)
    rows_out: list[dict[str, object]] = []
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT m.scan_id, m.patient_id, ds.source_name, m.harmonization_status
            FROM mr_scans m
            JOIN patients p ON m.patient_id = p.patient_id
            JOIN dataset_sources ds ON p.source_id = ds.source_id
            WHERE m.scan_id = ANY(%s)
            """,
            (scan_ids,),
        )
        db_rows = {r[0]: {"patient_id": r[1], "source_name": r[2], "harmonization_status": r[3]} for r in cursor.fetchall()}
        cursor.close()

        missing_in_db = [sid for sid in scan_ids if sid not in db_rows]
        if missing_in_db:
            print(f"[UYARI] CSV'de olup mr_scans'ta bulunamayan scan_id sayisi: {len(missing_in_db)} -> {missing_in_db[:20]}")

        for sid in scan_ids:
            db_row = db_rows.get(sid)
            csv_row = csv_by_scan_id[sid]
            if db_row is None:
                rows_out.append(
                    {
                        "scan_id": sid,
                        "patient_id": csv_row["patient_id"],
                        "dataset_source": csv_row["dataset_source"],
                        "before_status": None,
                        "after_status": TARGET_HARMONIZATION_STATUS,
                        "db_action": "ATLANDI_SCAN_ID_DB_DE_YOK",
                    }
                )
                continue

            if db_row["source_name"] != csv_row["dataset_source"]:
                rows_out.append(
                    {
                        "scan_id": sid,
                        "patient_id": csv_row["patient_id"],
                        "dataset_source": csv_row["dataset_source"],
                        "before_status": db_row["harmonization_status"],
                        "after_status": TARGET_HARMONIZATION_STATUS,
                        "db_action": f"ATLANDI_KAYNAK_UYUSMAZLIGI_DB={db_row['source_name']}",
                    }
                )
                continue

            before_status = db_row["harmonization_status"]
            if before_status == TARGET_HARMONIZATION_STATUS:
                rows_out.append(
                    {
                        "scan_id": sid,
                        "patient_id": csv_row["patient_id"],
                        "dataset_source": csv_row["dataset_source"],
                        "before_status": before_status,
                        "after_status": TARGET_HARMONIZATION_STATUS,
                        "db_action": "DEGISIKLIK_YOK_ZATEN_ZSCORE",
                    }
                )
                continue

            out = {
                "scan_id": sid,
                "patient_id": csv_row["patient_id"],
                "dataset_source": csv_row["dataset_source"],
                "before_status": before_status,
                "after_status": TARGET_HARMONIZATION_STATUS,
                "db_action": "",
            }

            try:
                if args.apply:
                    updated = _update_row(connection, scan_id=sid, new_status=TARGET_HARMONIZATION_STATUS)
                    connection.commit()  # satir-bazli commit: bir hata diger satirlari bozmasin
                    out["db_action"] = "YAZILDI" if updated else "ATLANDI_SCAN_ID_BULUNAMADI"
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
        fieldnames = ["scan_id", "patient_id", "dataset_source", "before_status", "after_status", "db_action"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    counts = Counter(r["db_action"] for r in rows_out)
    by_source = Counter(
        (r["dataset_source"], r["db_action"]) for r in rows_out if r["db_action"] in ("DRY_RUN", "YAZILDI")
    )
    print(f"TOPLAM CSV SATIRI (TCGA+UPenn, n4_zscore_ok): {len(csv_rows)}")
    print(f"TOPLAM ISLENEN SATIR: {len(rows_out)}")
    print(f"Dagilim (db_action): {dict(counts)}")
    print(f"Kaynak x aksiyon kirilimi: {dict(by_source)}")
    print(f"Hedef harmonization_status: '{TARGET_HARMONIZATION_STATUS}'")
    print(f"Rapor: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
