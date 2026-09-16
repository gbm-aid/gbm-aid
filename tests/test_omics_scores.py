"""`pipeline/omics_scores.py` regresyon testleri.

Zorunlu regresyon: üç fonksiyon da `omics_profiles`'ın ham değerlerinden
başlayarak `molecular_scores`'taki mevcut 48 satırı yeniden üretmeli.
Tolerans: tmz/aggr için tam eşleşme (2 ondalık yuvarlama payı), dna_repair
için ±0.01.

İKİ KATMANLI regresyon (2026-08-18, Codex çapraz incelemesi BOŞLUK 2
düzeltmesi):
  1. **Fixture-tabanlı** (`test_*_reproduces_fixture_48_of_48`) — DB
     GEREKTİRMEZ, ASLA skip edilmez. `tests/fixtures/omics_scores_
     regression_48.json`'a karşı çalışır (canlı DB'den BİR KEZ üretildi,
     `FIXTURE_SHA256` ile bütünlüğü kilitli). `dna_repair_score`'un
     dondurulmuş referans sabitlerinin (`DNA_REPAIR_RAW_MEAN_V1_0`,
     `DNA_REPAIR_RAW_STD_V1_0`) kaymasına karşı TEK/ASIL koruma BUDUR —
     DB erişilemese/CI'da olmasa BİLE bu test koşar ve kırmızı yanar.
  2. **Canlı-DB** (`test_*_reproduces_db_48_of_48_live`) — DB'ye
     erişilemezse skip edilir (repo genelindeki smoke-test
     konvansiyonu). Amacı: fixture'ın kendisinin BAYATLAMADIĞINI
     (canlı DB ile hâlâ örtüştüğünü) ayrıca doğrulamak — fixture'ın
     YERİNE geçmez, ÜSTÜNE eklenir.

Bu dosya `pipeline/cox_model.py`/`tools/train_cox_week3.py`'yi import
ETMEZ (modeling-agent çakışma uyarısı, AKTIF-GOREVLER.md 2026-08-18).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.omics_scores import (  # noqa: E402
    classify_aggressiveness,
    classify_dna_repair,
    classify_tmz_resistance,
    compute_aggressiveness_score,
    compute_dna_repair_score,
    compute_tmz_resistance_score,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "omics_scores_regression_48.json"

# `generate_omics_fixture.py` (scratchpad, canlı DB'ye karşı 2026-08-18'de
# BİR KEZ çalıştırıldı) tarafından üretildi. Bu sabit fixture dosyasının
# SESSİZCE değiştirilmediğini kilitler -- fixture değişirse (kasıtlı
# güncelleme DAHİL) bu sabit de BİLİNÇLİ olarak güncellenmeli.
FIXTURE_SHA256 = "ce6bad97bb2e399c04b526530f8309195fe3dc27bb883f6e83d63bfda36b562a"


# ---------------------------------------------------------------------------
# Saf birim testleri (DB gerektirmez) — formüllerin sözleşmesini kilitler
# ---------------------------------------------------------------------------


def test_tmz_resistance_clips_to_0_100():
    # Aşırı uç girdi: skor formülsel olarak 100'ü aşar, clip[0,100] tutmalı.
    row = {
        "mgmt_methylation": -1.0,  # -1*25 = +25 katkı (met negatif -> skor artar)
        "msh6_methylation": 1.0,
        "mlh1_methylation": 1.0,
        "pms2_methylation": 1.0,
        "msh6_expression": 0.0,  # max(0,5-0)*3 = 15
    }
    score = compute_tmz_resistance_score(row)
    assert 0.0 <= score <= 100.0


def test_aggressiveness_clips_to_0_100():
    row = {
        "egfr_expression": 999.0,  # min(.,12) ile sınırlanır
        "egfr_cnv": 999.0,
        "pten_methylation": 999.0,
        "cdkn2a_cnv": -999.0,
        "tp53_methylation": -999.0,
    }
    score = compute_aggressiveness_score(row)
    assert score == 100.0  # clip üst sınıra oturmalı


def test_dna_repair_clips_to_0_100():
    row = {f"{g}_methylation": 999.0 for g in ("msh2", "mlh1", "brca1", "brca2", "chek2")}
    row.update({f"{g}_expression": -999.0 for g in ("msh2", "mlh1", "brca1", "brca2", "chek2")})
    score = compute_dna_repair_score(row)
    assert 0.0 <= score <= 100.0


def test_classify_tmz_resistance_boundaries():
    assert classify_tmz_resistance(0.0) == "sensitive"
    assert classify_tmz_resistance(46.5009) == "sensitive"  # <=p33 dahil
    assert classify_tmz_resistance(46.51) == "intermediate"
    assert classify_tmz_resistance(48.39) == "intermediate"  # <=p66 dahil
    assert classify_tmz_resistance(48.40) == "resistant"
    assert classify_tmz_resistance(100.0) == "resistant"


def test_classify_dna_repair_boundaries():
    assert classify_dna_repair(0.0) == "impaired"
    assert classify_dna_repair(41.9847) == "impaired"
    assert classify_dna_repair(41.99) == "intermediate"
    assert classify_dna_repair(59.0642) == "intermediate"
    assert classify_dna_repair(59.07) == "intact"
    assert classify_dna_repair(100.0) == "intact"


def test_classify_aggressiveness_fixed_thresholds():
    assert classify_aggressiveness(44.99) == "low"
    assert classify_aggressiveness(45.0) == "intermediate"
    assert classify_aggressiveness(64.99) == "intermediate"
    assert classify_aggressiveness(65.0) == "high"
    assert classify_aggressiveness(79.99) == "high"
    assert classify_aggressiveness(80.0) == "very_high"


def test_classify_aggressiveness_low_matches_function_contract():
    """`classify_aggressiveness()` score<45 için 'low' döner (mimari +
    Excel Data Dictionary'nin 4-kategori tanımıyla tutarlı)."""
    assert classify_aggressiveness(10.0) == "low"


# ---------------------------------------------------------------------------
# 2026-08-18 Codex çapraz incelemesi BUG 1 -- classify_* SAVUNMACI YUVARLAMA
# regresyon testleri. Her biri Codex'in bulduğu/istediği somut sınır
# değeriyle, ham (yuvarlanmamış) float DOĞRUDAN classify_*'a verilerek
# çağrılır -- compute_* devre dışı, yalnız classify_*'ın kendi savunması
# test ediliyor.
# ---------------------------------------------------------------------------


def test_classify_tmz_resistance_defends_against_unrounded_float_near_p33():
    # 46.504 yuvarlanmamış -> p33=46.5009'u aşar gibi görünür ama
    # round(46.504,2)=46.5 <= 46.5009 -> "sensitive" olmalı.
    assert classify_tmz_resistance(46.504) == "sensitive"


def test_classify_tmz_resistance_defends_against_unrounded_float_near_p66():
    # 48.3905 yuvarlanmamış -> p66=48.3900'ü aşar gibi görünür ("resistant")
    # ama round(48.3905,2)=48.39 <= 48.3900 -> "intermediate" olmalı
    # (canlı vaka: TCGA-26-5134, bkz. pipeline/omics_scores.py docstring).
    assert classify_tmz_resistance(48.3905) == "intermediate"


def test_classify_aggressiveness_defends_against_unrounded_float_near_45():
    # 44.996 yuvarlanmamış -> 45'in altında görünür ("low") ama
    # round(44.996,2)=45.0 -> "intermediate" olmalı.
    assert classify_aggressiveness(44.996) == "intermediate"


def test_classify_dna_repair_defends_against_unrounded_float_near_p33():
    # 41.984 yuvarlanmamış -> round(41.984,2)=41.98 <= p33=41.9847 -> "impaired".
    assert classify_dna_repair(41.984) == "impaired"


def test_classify_dna_repair_defends_against_unrounded_float_near_p66():
    # 59.0642 tam sınır değeri, yuvarlanınca 59.06 <= p66=59.0642 -> "intermediate".
    assert classify_dna_repair(59.0642) == "intermediate"


# ---------------------------------------------------------------------------
# Fixture bütünlüğü — bu, DB'siz regresyonun GÜVENİLİRLİĞİNİN ön koşulu.
# ---------------------------------------------------------------------------


def test_fixture_file_exists_and_matches_recorded_sha256():
    assert FIXTURE_PATH.is_file(), f"Fixture dosyası eksik: {FIXTURE_PATH}"
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    actual_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert actual_sha256 == FIXTURE_SHA256, (
        "Fixture dosyası değişmiş (sha256 uyuşmuyor) -- bu ya SESSİZ bir "
        "bozulma ya da bilinçli bir güncelleme; ikinci durumda "
        "FIXTURE_SHA256 sabiti de BİLİNÇLİ olarak güncellenmeli. "
        f"beklenen={FIXTURE_SHA256} gerçek={actual_sha256}"
    )


def _load_fixture_records() -> list[dict]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    records = payload["records"]
    assert len(records) == 48, (
        f"Fixture'da beklenen 48 hasta, {len(records)} bulundu -- fixture bozuk."
    )
    return records


@pytest.fixture(scope="module")
def fixture_omics_rows():
    return _load_fixture_records()


def test_tmz_resistance_score_reproduces_fixture_48_of_48(fixture_omics_rows):
    """DB GEREKTİRMEZ, ASLA skip edilmez -- bkz. modül docstring'i."""
    mismatches = []
    for row in fixture_omics_rows:
        expected_score = float(row["tmz_resistance_score"])
        expected_class = row["tmz_class"]
        computed_score = compute_tmz_resistance_score(row)
        computed_class = classify_tmz_resistance(computed_score)
        if abs(computed_score - expected_score) > 0.01 or computed_class != expected_class:
            mismatches.append(
                (row["patient_id"], computed_score, expected_score, computed_class, expected_class)
            )
    assert not mismatches, f"tmz_resistance_score/tmz_class uyuşmazlığı (fixture): {mismatches}"


def test_aggressiveness_score_reproduces_fixture_48_of_48(fixture_omics_rows):
    """DB GEREKTİRMEZ, ASLA skip edilmez -- bkz. modül docstring'i."""
    mismatches = []
    for row in fixture_omics_rows:
        expected_score = float(row["aggressiveness_score"])
        expected_class = row["aggr_class"]
        computed_score = compute_aggressiveness_score(row)
        computed_class = classify_aggressiveness(computed_score)
        if abs(computed_score - expected_score) > 0.01 or computed_class != expected_class:
            mismatches.append(
                (row["patient_id"], computed_score, expected_score, computed_class, expected_class)
            )
    assert not mismatches, f"aggressiveness_score/aggr_class uyuşmazlığı (fixture): {mismatches}"


def test_dna_repair_score_reproduces_fixture_48_of_48_within_tolerance(fixture_omics_rows):
    """DB GEREKTİRMEZ, ASLA skip edilmez -- `DNA_REPAIR_RAW_MEAN_V1_0`/
    `DNA_REPAIR_RAW_STD_V1_0` referans sabitlerinin kaymasına karşı TEK
    koruma budur (bkz. modül docstring'i, Codex BOŞLUK 2)."""
    mismatches = []
    for row in fixture_omics_rows:
        expected_score = float(row["dna_repair_score"])
        expected_class = row["repair_class"]
        computed_score = compute_dna_repair_score(row)
        computed_class = classify_dna_repair(computed_score)
        if abs(computed_score - expected_score) > 0.01 or computed_class != expected_class:
            mismatches.append(
                (row["patient_id"], computed_score, expected_score, computed_class, expected_class)
            )
    assert not mismatches, (
        f"dna_repair_score/repair_class uyuşmazlığı (fixture, ±0.01 tolerans): {mismatches}"
    )


# ---------------------------------------------------------------------------
# Canlı DB: molecular_scores_aggr_class_check artık 'low'u kabul ediyor mu?
# (2026-08-18 migrasyonu, Barış onayı -- savepoint-izole, TAM ROLLBACK,
# kalıcı satır BIRAKMAZ). Erişilemezse skip -- bu testin TEK amacı DB
# constraint'inin canlı durumunu doğrulamak, fixture testleriyle
# İLGİSİZ/bağımsız.
# ---------------------------------------------------------------------------


def test_aggr_class_check_constraint_accepts_low_live():
    try:
        import db_connection

        conn = db_connection.get_connection(readonly=False)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gerçek DB'ye erişilemedi, canlı constraint testi atlandı: {exc}")

    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT pg_get_constraintdef(oid) FROM pg_constraint
            WHERE conrelid='molecular_scores'::regclass
              AND conname='molecular_scores_aggr_class_check'
            """
        )
        (constraint_def,) = cur.fetchone()
        assert "'low'" in constraint_def, (
            f"CHECK constraint 'low' içermiyor -- migrasyon geri mi alındı? {constraint_def}"
        )

        # savepoint-izole gerçek INSERT denemesi -- TAM ROLLBACK ile biter
        cur.execute("SAVEPOINT before_low_insert_test")
        cur.execute("SELECT source_id FROM dataset_sources LIMIT 1")
        (some_source_id,) = cur.fetchone()
        fake_patient_id = "TEST-AGGR-LOW-PYTEST-ROLLBACK-ONLY"
        cur.execute(
            "INSERT INTO patients (patient_id, source_id, has_mr, has_omics) "
            "VALUES (%s, %s, FALSE, TRUE)",
            (fake_patient_id, some_source_id),
        )
        cur.execute(
            "INSERT INTO molecular_scores (patient_id, aggressiveness_score, aggr_class, score_version) "
            "VALUES (%s, 10.0, 'low', 'v1.0-test')",
            (fake_patient_id,),
        )
        cur.execute("ROLLBACK TO SAVEPOINT before_low_insert_test")
    finally:
        conn.rollback()  # TAM rollback -- kalıcı iz YOK
        cur.close()
        conn.close()


# ---------------------------------------------------------------------------
# Canlı DB regresyon testi -- fixture'ın BAYATLAMADIĞINI (canlı DB ile hâlâ
# örtüştüğünü) doğrular. Erişilemezse skip -- fixture testlerinin YERİNE
# GEÇMEZ, onların ÜSTÜNE eklenir (bkz. modül docstring'i).
# ---------------------------------------------------------------------------


def _get_live_omics_rows():
    """omics_profiles JOIN molecular_scores, tüm 48 satır. Erişilemezse
    pytest.skip fırlatır (repo genelindeki smoke-test konvansiyonu,
    bkz. tests/test_api_predict.py)."""
    try:
        import db_connection  # proje kökündeki .env üzerinden bağlanır

        conn = db_connection.get_connection(readonly=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gerçek DB'ye erişilemedi, canlı-DB regresyon testi atlandı: {exc}")

    cur = conn.cursor()
    cur.execute(
        """
        SELECT o.patient_id,
               o.mgmt_methylation, o.msh2_methylation, o.msh6_methylation,
               o.mlh1_methylation, o.pms2_methylation, o.egfr_methylation,
               o.pten_methylation, o.tp53_methylation, o.cdkn2a_methylation,
               o.brca1_methylation, o.brca2_methylation, o.chek2_methylation,
               o.mgmt_expression, o.msh2_expression, o.msh6_expression,
               o.mlh1_expression, o.pms2_expression, o.egfr_expression,
               o.pten_expression, o.tp53_expression, o.cdkn2a_expression,
               o.brca1_expression, o.brca2_expression, o.chek2_expression,
               o.mgmt_cnv, o.egfr_cnv, o.pten_cnv, o.cdkn2a_cnv, o.tp53_cnv,
               m.tmz_resistance_score, m.tmz_class,
               m.aggressiveness_score, m.aggr_class,
               m.dna_repair_score, m.repair_class
        FROM omics_profiles o
        JOIN molecular_scores m ON m.patient_id = o.patient_id
        ORDER BY o.patient_id
        """
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


@pytest.fixture(scope="module")
def live_omics_rows():
    rows = _get_live_omics_rows()
    if len(rows) != 48:
        pytest.fail(
            f"Beklenen 48 hasta, DB'den {len(rows)} geldi -- kohort DEĞİŞMİŞ "
            "olabilir, regresyon testi bunu SESSİZCE geçemez."
        )
    return rows


def test_tmz_resistance_score_reproduces_db_48_of_48_live(live_omics_rows):
    mismatches = []
    for row in live_omics_rows:
        expected_score = float(row["tmz_resistance_score"])
        expected_class = row["tmz_class"]
        computed_score = compute_tmz_resistance_score(row)
        computed_class = classify_tmz_resistance(computed_score)
        if abs(computed_score - expected_score) > 0.01 or computed_class != expected_class:
            mismatches.append(
                (row["patient_id"], computed_score, expected_score, computed_class, expected_class)
            )
    assert not mismatches, f"tmz_resistance_score/tmz_class uyuşmazlığı (canlı DB): {mismatches}"


def test_aggressiveness_score_reproduces_db_48_of_48_live(live_omics_rows):
    mismatches = []
    for row in live_omics_rows:
        expected_score = float(row["aggressiveness_score"])
        expected_class = row["aggr_class"]
        computed_score = compute_aggressiveness_score(row)
        computed_class = classify_aggressiveness(computed_score)
        if abs(computed_score - expected_score) > 0.01 or computed_class != expected_class:
            mismatches.append(
                (row["patient_id"], computed_score, expected_score, computed_class, expected_class)
            )
    assert not mismatches, f"aggressiveness_score/aggr_class uyuşmazlığı (canlı DB): {mismatches}"


def test_dna_repair_score_reproduces_db_48_of_48_within_tolerance_live(live_omics_rows):
    mismatches = []
    for row in live_omics_rows:
        expected_score = float(row["dna_repair_score"])
        expected_class = row["repair_class"]
        computed_score = compute_dna_repair_score(row)
        computed_class = classify_dna_repair(computed_score)
        if abs(computed_score - expected_score) > 0.01 or computed_class != expected_class:
            mismatches.append(
                (row["patient_id"], computed_score, expected_score, computed_class, expected_class)
            )
    assert not mismatches, (
        f"dna_repair_score/repair_class uyuşmazlığı (canlı DB, ±0.01 tolerans): {mismatches}"
    )
