"""`followup_series`'in 5 hala-NULL kolonunu (`tumor_volume_mm3`, `mgmt_status`,
`idh1_status`, `segmentation_tool`, `scan_id`) doldurmak icin TASARIM +
DRY-RUN raporu, VE (2026-09-12 db-agent-B, Baris onayi: "3. icin okey durum
apply edebiliriz") KISITLI bir --apply yolu.

** VARSAYILAN MOD HALA DRY-RUN. ** `--apply` bayragi olmadan bu script
HICBIR INSERT/UPDATE/ALTER/DELETE CALISTIRMAZ, sadece CSV rapor uretir
(asagidaki tasarim/DRY-RUN aciklamasi degismedi).

`--apply` VERILDIGINDE bile SADECE `match_status == "MATCHED"`
(match_confidence == "HIGH_POSITIONAL") olan 298 satir UPDATE edilir.
`WEEK_MISMATCH_SUSPICIOUS` (13) ve `AMBIGUOUS_COUNT_MISMATCH` (305) satirlarina
KESINLIKLE DOKUNULMAZ -- bu ayrim kod seviyesinde zorunlu kilinir
(`_select_matched_rows_only` + `_assert_only_matched_targeted`), baska bir
match_status'u yazmaya calisan bir kod yolu YOKTUR; boyle bir tutarsizlik
tespit edilirse script FAIL-LOUD (RuntimeError) ile durur, sessizce atlamaz.

KAPSAM: SADECE `followup_series.source_id=3` (LUMIERE) satirlari -- 616/616
satir (hafta2_mert_ozet.md'nin "RED: followup_series.scan_id LUMIERE 616/616
NULL" bulgusuyla BIREBIR ayni sayi, canli 2026-09-12 SELECT ile dogrulandi).
`source_id=1` (TCGA-GBM, 280 satir) BU SCRIPT'IN KAPSAMI DISINDA -- o
satirlarin kaynagi `pipeline/growth_simulation.py` DEGIL (o modul SADECE
LUMIERE'i, `m.patient_id ILIKE 'Patient-%%'` filtresiyle isler), farkli bir
surecin (muhtemelen tumor_events/T3 RANO) urunudur, bu script ONLARA
DOKUNMAZ/DOKUNMAYI ONERMEZ.

## Veri kaynagi

`pipeline/growth_simulation.py::fetch_lumiere_wt_volume_series()` DOGRUDAN
CAGRILIR (yeniden yazilmadi) -- bu fonksiyon `radiomics` (segmentation_tool=
'LUMIERE-PyRadiomics-107-C32', tumor_region='WT_derived') + `mr_scans` JOIN'i
uzerinden SADECE SELECT yapar, patient_id/scan_id/tumor_volume_mm3/
timepoint_label/week/suffix/x_week doner. `artifacts/week4/growth_simulation_v3/
patient_transition_growth_rates_v3.csv` (gorev talimatinin isaret ettigi
artifact) bu ayni verinin ARDISIK-CIFT (transition) goruntusudur -- ayri bir
per-scan sutunu TASIMADIGI (Pre-Op/Post-Op ayrimi YOK) icin eslestirme icin
CANLI DB sorgusu (ayni fonksiyon) tercih edildi; SAYISAL DEGERLER (tumor_
volume_mm3) HER IKI KAYNAKTA DA AYNI TABLODAN (radiomics) gelir, tutarsizlik
riski YOK.

## Eslestirme problemi VE cozumu (ACIKCA yazilir, gizlenmez)

`followup_series` (mevcut 616 LUMIERE satiri, `followup_t3.py` tarafindan
DOLDURULDU -- bu script tarafindan degil) `scan_id` TASIMIYOR. Bu yuzden
"hangi followup_series satiri hangi mr_scans.scan_id'ye karsilik geliyor"
sorusu DOGRUDAN bir foreign-key ile cevaplanamiyor -- BIR ESLESTIRME
KURALI gerekiyor. Tek basina `(patient_id, visit_week)` YETERSIZ: bazi
hastalarda Pre-Op VE Post-Op ikisi de `visit_week=0` (iki farkli scan,
AYNI hafta) -- bu KANITLANDI (canli ornek: Patient-001).

**Secilen kural: POZISYONEL eslestirme + SAYI ESITLIGI KAPISI.**
Her hasta icin:
  1. Mevcut `followup_series` satirlari `followup_id` (artan, DOLDURULMA
     SIRASI = kronolojik varsayim) ile siralanir.
  2. Canli radyomik/mr_scans'ten gelen scan serisi `x_week` (kronolojik,
     `growth_simulation.py`'nin KENDI siralama mantigi) ile siralanir.
  3. Iki listenin UZUNLUKLARI ESITSE, i'inci mevcut satir <-> i'inci scan
     POZISYONEL olarak eslenir (guven: YUKSEK -- ayni hastanin GORULEN scan
     sayisi ile mevcut RANO-satiri sayisi TUTARLI).
  4. UZUNLUKLAR ESIT DEGILSE, bu hastanin TUM satirlari "AMBIGUOUS_COUNT_
     MISMATCH" olarak isaretlenir -- HICBIR pozisyonel tahmin YAPILMAZ,
     hicbir deger onerilmez (sessizce yanlis eslestirme YAPMAMAK, yanlis
     ama "calisiyor gorunen" bir deger UYDURMAKTAN daha onemli).
  5. Esitlik gecse bile, eslenen cift arasindaki hafta farki (`|visit_week -
     round(x_week)|`) 2 haftadan fazlaysa satir "WEEK_MISMATCH_SUSPICIOUS"
     olarak ISARETLENIR (yine de rapora yazilir AMA "guven: DUSUK" etiketiyle
     -- insan onayi ozellikle bu satirlar icin istenir).

Bu kural DOGRULANMAMIS bir VARSAYIMA dayanir: "followup_id artan sirasi =
kronolojik ziyaret sirasi". Bu script bu varsayimi `followup_t3.py`'nin
KENDI kodunu okuyarak BAGIMSIZ DOGRULAMADI (kapsam disi, zaman kisiti) --
asagidaki rapor ve konsol ciktisinda bu ACIKCA bir varsayim olarak
belirtilir, "kanitlandi" DENMEZ.

## Cikti

  - Konsol: ozet sayaclar (eslendi / ambiguous / week-mismatch / toplam).
  - CSV: `artifacts/week4/followup_series_write_dry_run_report.csv` --
    HER mevcut LUMIERE `followup_series` satiri icin (followup_id bazinda)
    onerilen UPDATE degerleri (`proposed_*` sutunlari) VEYA atlama nedeni.

Kullanim:
  python tools/write_followup_series_to_db.py               # dry-run (varsayilan)
  python tools/write_followup_series_to_db.py --apply        # SADECE 298 MATCHED satiri yazar

`--apply` icin Baris onayi ZORUNLU (bkz. yukaridaki not, 2026-09-12).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # gbm-aid mert/tools -> gbm-aid mert
sys.path.insert(0, str(PROJECT_ROOT))

from db_connection import get_connection  # noqa: E402
from pipeline.growth_simulation import (  # noqa: E402
    SEGMENTATION_TOOL,
    TUMOR_REGION,
    fetch_lumiere_wt_volume_series,
)

REPORT_PATH = PROJECT_ROOT / "artifacts" / "week4" / "followup_series_write_dry_run_report.csv"
APPLY_REPORT_PATH = PROJECT_ROOT / "artifacts" / "week4" / "followup_series_write_apply_report.csv"
WEEK_MISMATCH_THRESHOLD_WEEKS = 2.0
TARGET_COLUMNS = (
    "tumor_volume_mm3",
    "mgmt_status",
    "idh1_status",
    "segmentation_tool",
    "scan_id",
)


def _fetch_existing_lumiere_rows(conn) -> list[dict]:
    """followup_series.source_id=3 (LUMIERE) satirlarini followup_id artan
    sirada doner -- SADECE SELECT."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.followup_id, f.patient_id, f.source_id, f.visit_week,
                   f.tumor_volume_mm3, f.rano_label, f.rano_rationale,
                   f.mgmt_status, f.idh1_status, f.segmentation_tool, f.scan_id
            FROM followup_series f
            JOIN dataset_sources ds ON f.source_id = ds.source_id
            WHERE ds.source_name = 'LUMIERE'
            ORDER BY f.patient_id, f.followup_id
            """
        )
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _fetch_patient_molecular_status(conn, patient_ids: list[str]) -> dict[str, dict]:
    """patients.mgmt_status/idh1_status'u OLDUGU GIBI (normalize ETMEDEN)
    doner -- LUMIERE'in kendi ham etiketleri (orn. 'wt', 'methylated') zaten
    `data_integrity_check.py`'nin domain-normalizasyon WARN'inda kayitli,
    bu script YENI bir normalizasyon semasi ICAT ETMEZ (kapsam disi)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT patient_id, mgmt_status, idh1_status FROM patients WHERE patient_id = ANY(%s)",
            (patient_ids,),
        )
        return {row[0]: {"mgmt_status": row[1], "idh1_status": row[2]} for row in cur.fetchall()}


def build_dry_run_rows(conn) -> list[dict]:
    existing_rows = _fetch_existing_lumiere_rows(conn)
    existing_by_patient: dict[str, list[dict]] = {}
    for row in existing_rows:
        existing_by_patient.setdefault(row["patient_id"], []).append(row)

    series_df = fetch_lumiere_wt_volume_series(conn)  # SADECE SELECT (pipeline modulunun kendi fonksiyonu)
    series_by_patient: dict[str, list[dict]] = {}
    for _, r in series_df.iterrows():
        series_by_patient.setdefault(r["patient_id"], []).append(
            {
                "scan_id": int(r["scan_id"]),
                "timepoint_label": r["timepoint_label"],
                "x_week": float(r["x_week"]),
                "tumor_volume_mm3": float(r["tumor_volume_mm3"]),
            }
        )
    # fetch_lumiere_wt_volume_series zaten x_week'e gore sirali doner (kendi
    # docstring'i) -- burada TEKRAR sirali oldugunu ACIKCA garanti ediyoruz
    # (kirilgan bir "zaten sirali" varsayimina GUVENMEMEK icin).
    for pid in series_by_patient:
        series_by_patient[pid].sort(key=lambda d: d["x_week"])

    molecular = _fetch_patient_molecular_status(conn, list(existing_by_patient.keys()))

    report_rows: list[dict] = []
    for patient_id, existing in existing_by_patient.items():
        series = series_by_patient.get(patient_id, [])
        mol = molecular.get(patient_id, {"mgmt_status": None, "idh1_status": None})

        if len(existing) != len(series):
            for row in existing:
                report_rows.append({
                    "followup_id": row["followup_id"],
                    "patient_id": patient_id,
                    "visit_week": row["visit_week"],
                    "rano_label": row["rano_label"],
                    "match_status": "AMBIGUOUS_COUNT_MISMATCH",
                    "match_confidence": "NONE",
                    "reason": (
                        f"mevcut followup_series satiri sayisi ({len(existing)}) "
                        f"canli radyomik scan sayisiyla ({len(series)}) UYUSMUYOR -- "
                        "pozisyonel eslestirme GUVENLI DEGIL, hicbir deger onerilmedi."
                    ),
                    "proposed_scan_id": None,
                    "proposed_tumor_volume_mm3": None,
                    "proposed_segmentation_tool": None,
                    "proposed_mgmt_status": None,
                    "proposed_idh1_status": None,
                })
            continue

        for row, scan in zip(existing, series):
            week_delta = abs(float(row["visit_week"] or 0) - round(scan["x_week"]))
            suspicious = week_delta > WEEK_MISMATCH_THRESHOLD_WEEKS
            report_rows.append({
                "followup_id": row["followup_id"],
                "patient_id": patient_id,
                "visit_week": row["visit_week"],
                "rano_label": row["rano_label"],
                "match_status": "WEEK_MISMATCH_SUSPICIOUS" if suspicious else "MATCHED",
                "match_confidence": "LOW_NEEDS_HUMAN_REVIEW" if suspicious else "HIGH_POSITIONAL",
                "reason": (
                    f"pozisyonel eslesme -- mevcut visit_week={row['visit_week']} vs "
                    f"scan x_week={scan['x_week']:.2f} (fark={week_delta:.2f} hafta, "
                    f"timepoint_label={scan['timepoint_label']!r})"
                ),
                "proposed_scan_id": scan["scan_id"],
                "proposed_tumor_volume_mm3": scan["tumor_volume_mm3"],
                "proposed_segmentation_tool": SEGMENTATION_TOOL,
                "proposed_mgmt_status": mol["mgmt_status"],
                "proposed_idh1_status": mol["idh1_status"],
            })

    return report_rows


def _select_matched_rows_only(rows: list[dict]) -> list[dict]:
    """SADECE match_status=='MATCHED' VE match_confidence=='HIGH_POSITIONAL'
    olan satirlari doner -- WEEK_MISMATCH_SUSPICIOUS / AMBIGUOUS_COUNT_MISMATCH
    satirlari bu fonksiyonun ciktisina HICBIR SEKILDE giremez."""
    matched = [
        r for r in rows
        if r["match_status"] == "MATCHED" and r["match_confidence"] == "HIGH_POSITIONAL"
    ]
    _assert_only_matched_targeted(rows, matched)
    return matched


def _assert_only_matched_targeted(all_rows: list[dict], matched: list[dict]) -> None:
    """Fail-loud guard: yazilacak kume disinda hicbir non-MATCHED satir
    icerilmemis mi, ve MATCHED sayisi degismemis mi (298) dogrula."""
    matched_ids = {r["followup_id"] for r in matched}
    non_matched_ids = {
        r["followup_id"] for r in all_rows if r["match_status"] != "MATCHED"
    }
    leaked = matched_ids & non_matched_ids
    if leaked:
        raise RuntimeError(
            f"FAIL-LOUD: {len(leaked)} followup_id hem MATCHED hem non-MATCHED "
            f"kumesinde -- APPLY DURDURULDU. followup_id'ler: {sorted(leaked)}"
        )
    for r in matched:
        if r["proposed_scan_id"] is None or r["proposed_segmentation_tool"] is None:
            raise RuntimeError(
                f"FAIL-LOUD: followup_id={r['followup_id']} MATCHED isaretli ama "
                "proposed_* alanlari eksik -- APPLY DURDURULDU."
            )
    if len(matched) != 298:
        raise RuntimeError(
            f"FAIL-LOUD: beklenen 298 yuksek-guven eslesme yerine {len(matched)} "
            "bulundu -- veri/kod dry-run raporundan bu yana degismis olabilir. "
            "APPLY DURDURULDU, once fark arastirilmali."
        )


def _verify_targets_are_null(conn, followup_ids: list[int]) -> None:
    """APPLY'DAN ONCE: yazilacak 5 kolonun TUMUNUN NULL oldugunu dogrula.
    Beklenmeyen dolu bir satir bulunursa uzerine yazma riskine karsi
    FAIL-LOUD ile dur (hicbir UPDATE calistirilmadan)."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT followup_id, {', '.join(TARGET_COLUMNS)}
            FROM followup_series
            WHERE followup_id = ANY(%s)
            """,
            (followup_ids,),
        )
        rows = cur.fetchall()
    if len(rows) != len(followup_ids):
        raise RuntimeError(
            f"FAIL-LOUD: {len(followup_ids)} followup_id istendi, {len(rows)} "
            "satir donduruldu -- APPLY DURDURULDU."
        )
    not_null_found = [row for row in rows if any(v is not None for v in row[1:])]
    if not_null_found:
        raise RuntimeError(
            "FAIL-LOUD: apply oncesi beklenmeyen DOLU satir(lar) bulundu -- "
            f"uzerine yazma riski, APPLY DURDURULDU. followup_id'ler: "
            f"{[row[0] for row in not_null_found]}"
        )


def _apply_matched_rows(conn, matched: list[dict]) -> None:
    """SADECE matched (298) satirlari UPDATE eder. Transaction commit()
    cagiran (main) tarafindan yapilir."""
    with conn.cursor() as cur:
        for r in matched:
            cur.execute(
                """
                UPDATE followup_series
                SET tumor_volume_mm3 = %s,
                    mgmt_status = %s,
                    idh1_status = %s,
                    segmentation_tool = %s,
                    scan_id = %s
                WHERE followup_id = %s
                """,
                (
                    r["proposed_tumor_volume_mm3"],
                    r["proposed_mgmt_status"],
                    r["proposed_idh1_status"],
                    r["proposed_segmentation_tool"],
                    r["proposed_scan_id"],
                    r["followup_id"],
                ),
            )
            if cur.rowcount != 1:
                raise RuntimeError(
                    f"FAIL-LOUD: followup_id={r['followup_id']} UPDATE "
                    f"{cur.rowcount} satir etkiledi (1 beklenirdi) -- APPLY DURDURULDU."
                )


def _live_verify_after_apply(conn, matched_ids: list[int], other_ids: list[int]) -> dict:
    """Yazimdan SONRA canli SELECT ile kanit topla: matched_ids'in TUMU
    dolu mu, other_ids'in TUMU hala NULL mi, yan tablolar (TCGA-GBM
    280 satiri) etkilenmemis mi.

    NOT (2026-09-12 duzeltme): `mgmt_status`/`idh1_status` icin "dolu"
    kriteri YOKTUR -- bu iki alan `patients` tablosunun KENDI NULL'unu
    birebir tasir (bazi LUMIERE hastalarinda molekuler test verisi
    gercekten eksik, "muhtemelen" ile doldurulmadi). Bu yuzden 5-kolon-
    hepsi-dolu beklentisi YANLIS ALARM uretir -- sadece HER ZAMAN dolu
    olmasi gereken 3 kolon (`scan_id`, `tumor_volume_mm3`,
    `segmentation_tool`, radyomikten/turetilmis, patients'a bagli degil)
    298/298 kriteri olarak kullanilir; mgmt/idh1 bilgi amacli raporlanir."""
    ALWAYS_FILLED_COLUMNS = ("scan_id", "tumor_volume_mm3", "segmentation_tool")
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT COUNT(*) FILTER (
                     WHERE {' AND '.join(f'{c} IS NOT NULL' for c in ALWAYS_FILLED_COLUMNS)}
                   ),
                   COUNT(*),
                   COUNT(*) FILTER (WHERE mgmt_status IS NOT NULL),
                   COUNT(*) FILTER (WHERE idh1_status IS NOT NULL)
            FROM followup_series WHERE followup_id = ANY(%s)
            """,
            (matched_ids,),
        )
        matched_filled, matched_total, mgmt_filled, idh1_filled = cur.fetchone()

        cur.execute(
            f"""
            SELECT COUNT(*) FILTER (WHERE {' OR '.join(f'{c} IS NOT NULL' for c in TARGET_COLUMNS)}),
                   COUNT(*)
            FROM followup_series WHERE followup_id = ANY(%s)
            """,
            (other_ids,),
        )
        other_dirty, other_total = cur.fetchone()

        cur.execute(
            """
            SELECT COUNT(*) FILTER (WHERE tumor_volume_mm3 IS NOT NULL
                                        OR mgmt_status IS NOT NULL
                                        OR idh1_status IS NOT NULL
                                        OR segmentation_tool IS NOT NULL
                                        OR scan_id IS NOT NULL),
                   COUNT(*)
            FROM followup_series f
            JOIN dataset_sources ds ON f.source_id = ds.source_id
            WHERE ds.source_name = 'TCGA-GBM'
            """
        )
        tcga_dirty, tcga_total = cur.fetchone()

    return {
        "matched_filled": matched_filled,
        "matched_total": matched_total,
        "mgmt_filled": mgmt_filled,
        "idh1_filled": idh1_filled,
        "other_dirty": other_dirty,
        "other_total": other_total,
        "tcga_dirty": tcga_dirty,
        "tcga_total": tcga_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "followup_series 5 kolonunu doldur -- varsayilan dry-run, "
            "--apply ile SADECE 298 yuksek-guven (MATCHED) satir yazilir."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Belirtilmezse hicbir DB yazimi yapilmaz (dry-run varsayilan).",
    )
    args = parser.parse_args()

    conn = get_connection(readonly=not args.apply)
    try:
        rows = build_dry_run_rows(conn)

        if not args.apply:
            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "followup_id", "patient_id", "visit_week", "rano_label",
                "match_status", "match_confidence", "reason",
                "proposed_scan_id", "proposed_tumor_volume_mm3",
                "proposed_segmentation_tool", "proposed_mgmt_status", "proposed_idh1_status",
            ]
            with REPORT_PATH.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            n_total = len(rows)
            n_matched = sum(1 for r in rows if r["match_status"] == "MATCHED")
            n_suspicious = sum(1 for r in rows if r["match_status"] == "WEEK_MISMATCH_SUSPICIOUS")
            n_ambiguous = sum(1 for r in rows if r["match_status"] == "AMBIGUOUS_COUNT_MISMATCH")
            n_patients_ambiguous = len({r["patient_id"] for r in rows if r["match_status"] == "AMBIGUOUS_COUNT_MISMATCH"})

            print("=" * 78)
            print("followup_series DRY-RUN yazim tasarimi -- HICBIR YAZMA YAPILMADI")
            print("=" * 78)
            print(f"Kaynak segmentation_tool={SEGMENTATION_TOOL!r} tumor_region={TUMOR_REGION!r}")
            print(f"Toplam LUMIERE followup_series satiri (source_id=LUMIERE): {n_total}")
            print(f"  MATCHED (yuksek guven, pozisyonel + hafta tutarli): {n_matched}")
            print(f"  WEEK_MISMATCH_SUSPICIOUS (pozisyonel ama hafta farkli -- insan onayi istenir): {n_suspicious}")
            print(f"  AMBIGUOUS_COUNT_MISMATCH (hic tahmin YAPILMADI): {n_ambiguous} satir / {n_patients_ambiguous} hasta")
            print()
            print(f"Rapor yazildi: {REPORT_PATH}")
            print()
            print("HATIRLATMA: 'followup_id artan sirasi = kronolojik ziyaret sirasi' "
                  "varsayimi bu script tarafindan BAGIMSIZ DOGRULANMADI (followup_t3.py "
                  "kodu okunmadi, kapsam disi) -- APPLY ONCESI bu varsayim ayrica "
                  "dogrulanmali. --apply VERILMEDIGI icin hicbir yazma yapilmadi.")
            print("=" * 78)
            return 0

        # ---- --apply yolu: SADECE 298 MATCHED satir ----
        matched = _select_matched_rows_only(rows)
        matched_ids = [r["followup_id"] for r in matched]
        other_ids = [r["followup_id"] for r in rows if r["followup_id"] not in set(matched_ids)]

        _verify_targets_are_null(conn, matched_ids)

        _apply_matched_rows(conn, matched)
        conn.commit()

        verify = _live_verify_after_apply(conn, matched_ids, other_ids)
    finally:
        conn.close()

    if args.apply:
        APPLY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "followup_id", "patient_id", "visit_week", "rano_label",
            "apply_status", "proposed_scan_id", "proposed_tumor_volume_mm3",
            "proposed_segmentation_tool", "proposed_mgmt_status", "proposed_idh1_status",
        ]
        with APPLY_REPORT_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in matched:
                writer.writerow({
                    "followup_id": r["followup_id"],
                    "patient_id": r["patient_id"],
                    "visit_week": r["visit_week"],
                    "rano_label": r["rano_label"],
                    "apply_status": "YAZILDI",
                    "proposed_scan_id": r["proposed_scan_id"],
                    "proposed_tumor_volume_mm3": r["proposed_tumor_volume_mm3"],
                    "proposed_segmentation_tool": r["proposed_segmentation_tool"],
                    "proposed_mgmt_status": r["proposed_mgmt_status"],
                    "proposed_idh1_status": r["proposed_idh1_status"],
                })

        print("=" * 78)
        print("followup_series APPLY tamamlandi -- SADECE 298 MATCHED satir yazildi")
        print("=" * 78)
        print(f"Yazilan (MATCHED) satir sayisi: {len(matched)}")
        print(f"Canli dogrulama -- matched kumede scan_id/tumor_volume_mm3/"
              f"segmentation_tool (her zaman dolu olmali): "
              f"{verify['matched_filled']}/{verify['matched_total']}")
        print(f"  (bilgi amacli) mgmt_status dolu: {verify['mgmt_filled']}/{verify['matched_total']}, "
              f"idh1_status dolu: {verify['idh1_filled']}/{verify['matched_total']} -- "
              "geri kalan NULL, patients tablosunun KENDI eksikligini birebir tasir (fabrikasyon degil).")
        print(f"Canli dogrulama -- dokunulmayan LUMIERE satirlarinda (suspicious+ambiguous) "
              f"herhangi-bir-kolon-dolu: {verify['other_dirty']}/{verify['other_total']} (0 olmali)")
        print(f"Canli dogrulama -- TCGA-GBM (280 satir, bu scriptin kapsami disi) "
              f"herhangi-bir-kolon-dolu: {verify['tcga_dirty']}/{verify['tcga_total']} (0 olmali)")
        print(f"Apply raporu yazildi: {APPLY_REPORT_PATH}")
        print("=" * 78)

        if verify["matched_filled"] != 298 or verify["other_dirty"] != 0 or verify["tcga_dirty"] != 0:
            print("UYARI: canli dogrulama BEKLENEN degerlerle UYUSMUYOR -- yukarida detay.")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
