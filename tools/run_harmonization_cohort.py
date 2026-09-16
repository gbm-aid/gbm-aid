"""Kohort-çapında N4 bias-field düzeltmesi + kaynak-bazlı Z-score orkestratörü.

Kapsam: `mr_scans` tablosundaki TÜM taramalar (5234). Kaynağa göre üç ayrı
alt-kohort (TCGA/UPenn/LUMIERE) sırayla işlenir; her kaynak için:

  Faz A (N4): her taramanın NAS görüntüsüne `apply_n4_bias_correction`
  uygulanır. LUMIERE'de görüntü zaten 1mm izotropik değilse (bkz.
  decisions/2026-07-19-lumiere-voxel-spacing-kritik-bulgu.md), N4'ten ÖNCE
  `pipeline/resampling.py::resample_image_only` ile ortak 1mm ızgaraya
  taşınır -- gerekçe: N4'ün shrink-factor tabanlı çok-çözünürlüklü alanı,
  girdi son derece anizotropik (örn. 640x640x24 @ 0.36x0.36x6mm) olduğunda
  güvenilir bias-field fit üretmeyebilir; ayrıca downstream mask eşleme ve
  kohort Z-score istatistiği için ortak geometri gereklidir.

  Faz B (fit): Faz A'da üretilen TÜM N4 çıktılarından kaynak-bazlı tek bir
  Z-score istatistiği (`fit_source_zscore_statistics`) fit edilir ve
  `artifacts/week2/zscore/source_stats.json`'a yazılır (smoke test
  istatistiğinin -- `smoke_source_stats.json`, image_count=1 -- ÜZERİNE
  YAZILMAZ, ayrı dosya).

  Faz C (Z-score): her N4 çıktısına `apply_zscore_normalization` uygulanır.

LUMIERE'in bilinen not_evaluable hastaları bu batch'e DAHİL EDİLMEZ (bkz.
decisions/2026-08-07-lumiere-ana-maske-karari.md,
hafta2_durum_ozeti.md madde 10). AÇIK TUTARSIZLIK NOTU: kaynak belgede bu
grup "7 hasta" olarak toplanıyor ama tek tek isim listesi 8 farklı hasta
içeriyor (6 geometri-uyuşmazlığı diye etiketlenen alt listede aslında 7 isim
var: Patient-027/032/036/037/045/075/073, + 1 görüntü-eksik: Patient-025).
Güvenli taraf seçildi: adı geçen 8 hastanın TAMAMI hariç tutuldu. Bu sayı
farkı Barış'a ayrıca bildirilmelidir.

Çıktı log dosyası (`--report`, varsayılan
`artifacts/week2/harmonization_log.csv`) idempotent'tir: script yeniden
başlatılırsa, log'da zaten `n4_zscore_ok` olan `scan_id`'ler atlanır. Ayrıca
N4 aşaması, `apply_n4_bias_correction`'ın deterministik/atomik çıktı yoluna
göre diskte zaten var olan sonuçları da (log'dan bağımsız olarak) yeniden
hesaplamadan tanır.

DB'YE HİÇBİR UPDATE/INSERT YAPILMAZ (yalnız SELECT, salt-okunur bağlantı).
`mr_scans.harmonization_status` güncellemesi Barış'ın (db-agent) bu CSV'yi
kullanarak yapacağı ayrı bir görevdir.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# NOT: `.resolve()` KULLANMA -- bu makinede `subst X: "C:\Users\Barış\..."`
# ile ASCII sürücü haritalandı (kullanıcı adı Türkçe karakter içerdiği için
# ITK/SimpleITK `C:\Users\Barış\...` altına yazamıyor). `Path.resolve()`
# Windows'ta subst eşlemesini gerçek yola geri çözer (`GetFinalPathName`
# davranışı) ve bu ITK yazma hatasını GERİ GETİRİR -- canlı doğrulandı, bkz.
# log/2026-08-09.md. `.absolute()` sürücü harfini korur, yalnız bunu kullan.
PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from psycopg2.extras import RealDictCursor  # noqa: E402

from db_connection import get_connection  # noqa: E402
from pipeline.harmonization import (  # noqa: E402
    _derived_output_path,
    _nifti_stem,
    apply_n4_bias_correction,
    apply_zscore_normalization,
    canonical_source,
    fit_source_zscore_statistics,
)
from pipeline.resampling import (  # noqa: E402
    _single_output_directory,
    is_isotropic_spacing,
    resample_image_only,
)
from pipeline.segmentation import resolve_nas_path  # noqa: E402

# hafta2_durum_ozeti.md madde 10 -- ham liste 8 isim içeriyor ("7 hasta" olarak
# özetlenmiş olsa da). Güvenli taraf: tamamı hariç tutuluyor, bkz. modül
# docstring'i.
LUMIERE_NOT_EVALUABLE_PATIENTS = {
    "Patient-025",  # görüntü eksik
    "Patient-027",  # geometri/eksen uyuşmazlığı
    "Patient-032",
    "Patient-036",
    "Patient-037",
    "Patient-045",
    "Patient-073",
    "Patient-075",
}

REPORT_FIELDNAMES = [
    "scan_id",
    "patient_id",
    "dataset_source",
    "status",
    "hata_mesaji",
    "timestamp",
    "detay",
]

WEEK2_ROOT = PROJECT_ROOT / "artifacts" / "week2"
N4_ROOT = WEEK2_ROOT / "n4"
ZSCORE_ROOT = WEEK2_ROOT / "zscore"
LUMIERE_RESAMPLED_ROOT = WEEK2_ROOT / "lumiere_resampled_1mm"
STATS_PATH = ZSCORE_ROOT / "source_stats.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fetch_all_scans(connection) -> list[dict]:
    cursor = connection.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(
            """
            SELECT ms.scan_id, ms.patient_id, ms.modality, ms.file_path,
                   ds.source_name AS source
            FROM mr_scans ms
            JOIN patients p ON p.patient_id = ms.patient_id
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE ds.source_name IN ('TCGA-GBM', 'UPenn-GBM', 'LUMIERE')
            ORDER BY ds.source_name, ms.patient_id, ms.scan_id
            """
        )
        return list(cursor.fetchall())
    finally:
        cursor.close()


def _load_existing_report(report_path: Path) -> list[dict[str, str]]:
    if not report_path.is_file():
        return []
    with report_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_report(report_path: Path, rows: list[dict[str, object]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = report_path.with_name(f".{report_path.name}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, report_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kohort-çapında N4 + kaynak-bazlı Z-score orkestratörü (yalnız dosya/log, DB'ye yazmaz)."
    )
    parser.add_argument("--report", type=Path, default=WEEK2_ROOT / "harmonization_log.csv")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["TCGA-GBM", "UPenn-GBM", "LUMIERE"],
        help="Yalnız belirtilen kaynak(lar)ı işle (varsayılan: üçü de).",
    )
    parser.add_argument("--limit-per-source", type=int, default=None, help="Test için kaynak başına ilk N tarama.")
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    connection = get_connection(readonly=True)
    try:
        all_scans = _fetch_all_scans(connection)
    finally:
        connection.close()

    by_source: dict[str, list[dict]] = {}
    for scan in all_scans:
        by_source.setdefault(scan["source"], []).append(scan)

    existing_rows = _load_existing_report(args.report)
    rows_by_scan: dict[int, dict[str, object]] = {int(r["scan_id"]): r for r in existing_rows if r.get("scan_id")}
    already_ok = {
        scan_id for scan_id, row in rows_by_scan.items() if row.get("status") == "n4_zscore_ok"
    }
    already_skipped = {
        scan_id for scan_id, row in rows_by_scan.items() if row.get("status") == "skipped_not_evaluable"
    }
    print(
        f"Mevcut log: {len(already_ok)} n4_zscore_ok, {len(already_skipped)} skipped_not_evaluable "
        f"(idempotent devam).",
        flush=True,
    )

    def upsert_row(*, scan_id: int, patient_id: str, source: str, status: str, error: str = "", detail: str = "") -> None:
        rows_by_scan[scan_id] = {
            "scan_id": scan_id,
            "patient_id": patient_id,
            "dataset_source": source,
            "status": status,
            "hata_mesaji": error[:300],
            "timestamp": _now_iso(),
            "detay": detail[:200],
        }

    def flush_report() -> None:
        ordered = sorted(rows_by_scan.values(), key=lambda r: int(r["scan_id"]))
        _write_report(args.report, ordered)

    t_start = time.time()
    totals = {"ok": 0, "skip": 0, "fail": 0}

    for source in args.sources:
        scans = by_source.get(source, [])
        if args.limit_per_source:
            scans = scans[: args.limit_per_source]
        print(f"\n=== Kaynak: {source} ({len(scans)} tarama) ===", flush=True)
        patient_id_by_scan = {s["scan_id"]: s["patient_id"] for s in scans}

        # --- Faz A: N4 (+ LUMIERE için gerekirse önce 1mm resample) ---
        n4_outputs: dict[int, str] = {}
        os.environ["GBMAID_PROCESSED_ROOT"] = str(N4_ROOT)
        os.environ["GBMAID_RESAMPLED_ROOT"] = str(LUMIERE_RESAMPLED_ROOT)

        processed = 0
        n4_reused = 0
        n4_fresh = 0
        for scan in scans:
            scan_id = scan["scan_id"]
            patient_id = scan["patient_id"]

            if source == "LUMIERE" and patient_id in LUMIERE_NOT_EVALUABLE_PATIENTS:
                upsert_row(
                    scan_id=scan_id,
                    patient_id=patient_id,
                    source=source,
                    status="skipped_not_evaluable",
                    detail="LUMIERE not_evaluable hasta listesi (geometri uyuşmazlığı/görüntü eksik)",
                )
                totals["skip"] += 1
                processed += 1
                continue

            try:
                image_path = resolve_nas_path(scan["file_path"])
            except FileNotFoundError as exc:
                upsert_row(scan_id=scan_id, patient_id=patient_id, source=source, status="failed", error=f"GORUNTU_YOK:{exc}")
                totals["fail"] += 1
                processed += 1
                continue

            n4_input_path = image_path
            detail = ""
            try:
                if source == "LUMIERE" and not is_isotropic_spacing(image_path):
                    detail = "resampled_1mm_before_n4"
                    predicted_resampled = _single_output_directory(image_path, None) / (
                        f"{_nifti_stem(image_path)}_1mm.nii.gz"
                    )
                    if predicted_resampled.is_file():
                        n4_input_path = predicted_resampled
                    else:
                        resample_result = resample_image_only(image_path, output_dir=None)
                        n4_input_path = Path(resample_result["image_output_path"])

                # Diskte deterministik/atomik N4 çıktısı zaten varsa yeniden hesaplama
                # (idempotent resume) -- bu, log dosyasının durumundan BAĞIMSIZ, disk
                # gerçeğine dayanır; Faz B/C her koşuda TÜM kaynak N4 kümesi üzerinde
                # çalışsın diye önceden tamamlanmış taramalar da n4_outputs'a eklenir.
                predicted_n4 = _derived_output_path(n4_input_path, "n4")
                if predicted_n4.is_file():
                    n4_outputs[scan_id] = str(predicted_n4)
                    n4_reused += 1
                else:
                    n4_output = apply_n4_bias_correction(str(n4_input_path))
                    n4_outputs[scan_id] = n4_output
                    n4_fresh += 1
            except Exception as exc:  # noqa: BLE001
                upsert_row(
                    scan_id=scan_id,
                    patient_id=patient_id,
                    source=source,
                    status="failed",
                    error=f"N4_HATASI:{type(exc).__name__}:{exc}",
                    detail=detail,
                )
                totals["fail"] += 1

            processed += 1
            if processed % args.progress_every == 0:
                flush_report()
                elapsed = time.time() - t_start
                print(
                    f"  [Faz A/N4 {source}] {processed}/{len(scans)} islendi "
                    f"(yeni={n4_fresh}, diskten-tekrar-kullanildi={n4_reused}), gecen={elapsed:.0f}s",
                    flush=True,
                )

        flush_report()
        print(f"  Faz A tamam: {source} icin {len(n4_outputs)} yeni N4 ciktisi.", flush=True)

        if not n4_outputs:
            print(f"  {source}: yeni N4 çıktısı yok, Faz B/C atlanıyor.", flush=True)
            continue

        # --- Faz B: kaynak-bazlı Z-score istatistiği fit et ---
        try:
            record = fit_source_zscore_statistics(
                list(n4_outputs.values()), source, stats_path=STATS_PATH
            )
            print(f"  Faz B tamam: {source} stats -> mean={record['mean']:.4f} std={record['std']:.4f} (n_img={record['image_count']})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  Faz B HATA ({source}): {type(exc).__name__}: {exc}", flush=True)
            for scan_id in n4_outputs:
                patient_id = patient_id_by_scan[scan_id]
                upsert_row(
                    scan_id=scan_id,
                    patient_id=patient_id,
                    source=source,
                    status="failed",
                    error=f"ZSCORE_FIT_HATASI:{type(exc).__name__}:{exc}",
                )
                totals["fail"] += 1
            flush_report()
            continue

        # --- Faz C: Z-score normalize et ---
        os.environ["GBMAID_PROCESSED_ROOT"] = str(ZSCORE_ROOT)
        os.environ["GBMAID_ZSCORE_STATS_PATH"] = str(STATS_PATH)
        zprocessed = 0
        for scan_id, n4_path in n4_outputs.items():
            patient_id = patient_id_by_scan[scan_id]
            try:
                apply_zscore_normalization(n4_path, source)
                upsert_row(scan_id=scan_id, patient_id=patient_id, source=source, status="n4_zscore_ok")
                totals["ok"] += 1
            except Exception as exc:  # noqa: BLE001
                upsert_row(
                    scan_id=scan_id,
                    patient_id=patient_id,
                    source=source,
                    status="failed",
                    error=f"ZSCORE_APPLY_HATASI:{type(exc).__name__}:{exc}",
                )
                totals["fail"] += 1
            zprocessed += 1
            if zprocessed % args.progress_every == 0:
                flush_report()
                print(f"  [Faz C/Zscore {source}] {zprocessed}/{len(n4_outputs)} işlendi", flush=True)

        flush_report()
        print(f"  Faz C tamam: {source}.", flush=True)

    flush_report()
    elapsed = time.time() - t_start
    print(
        f"\nBitti. Toplam ok={totals['ok']} skip={totals['skip']} fail={totals['fail']} "
        f"(bu koşuda), gecen={elapsed:.0f}s. Rapor: {args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
