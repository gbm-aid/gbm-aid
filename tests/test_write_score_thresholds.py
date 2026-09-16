"""`tools/write_score_thresholds.py` -- saf mantık testleri (DB gerektirmez).

2026-08-18 Codex çapraz incelemesi BOŞLUK 3 düzeltmesi: idempotent
atlamadan önce mevcut satırın geçerliliğini doğrulayan `_validate_existing_row()`
fonksiyonunun kendisini kilitler.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_DIR / "write_score_thresholds.py"

spec = importlib.util.spec_from_file_location("write_score_thresholds", MODULE_PATH)
write_score_thresholds = importlib.util.module_from_spec(spec)
sys.modules["write_score_thresholds"] = write_score_thresholds
spec.loader.exec_module(write_score_thresholds)  # type: ignore[union-attr]

_validate_existing_row = write_score_thresholds._validate_existing_row

TARGET = {"score_version": "v1.0", "score_name": "tmz_resistance_score", "p33_cutoff": 46.5009, "p66_cutoff": 48.3900, "cohort_snapshot_n": 48}


def test_validate_existing_row_accepts_valid_matching_row():
    existing = {"p33_cutoff": 46.5009, "p66_cutoff": 48.39, "cohort_snapshot_n": 48}
    assert _validate_existing_row(TARGET, existing) == []


def test_validate_existing_row_accepts_valid_but_different_calibration():
    # p33<p66, ikisi de [0,100], cohort_snapshot_n dogru -- GECERLI ama
    # farkli kalibrasyon (main() bunun icin UYARI basar, FATAL degil).
    existing = {"p33_cutoff": 40.0, "p66_cutoff": 60.0, "cohort_snapshot_n": 48}
    assert _validate_existing_row(TARGET, existing) == []


def test_validate_existing_row_rejects_p33_greater_than_p66():
    existing = {"p33_cutoff": 60.0, "p66_cutoff": 40.0, "cohort_snapshot_n": 48}
    problems = _validate_existing_row(TARGET, existing)
    assert any("p33_cutoff" in p and "p66_cutoff" in p for p in problems)


def test_validate_existing_row_rejects_out_of_range_p33():
    existing = {"p33_cutoff": -5.0, "p66_cutoff": 60.0, "cohort_snapshot_n": 48}
    problems = _validate_existing_row(TARGET, existing)
    assert any("p33_cutoff" in p for p in problems)


def test_validate_existing_row_rejects_out_of_range_p66():
    existing = {"p33_cutoff": 40.0, "p66_cutoff": 150.0, "cohort_snapshot_n": 48}
    problems = _validate_existing_row(TARGET, existing)
    assert any("p66_cutoff" in p for p in problems)


def test_validate_existing_row_rejects_wrong_cohort_snapshot_n():
    existing = {"p33_cutoff": 46.5009, "p66_cutoff": 48.39, "cohort_snapshot_n": 611}
    problems = _validate_existing_row(TARGET, existing)
    assert any("cohort_snapshot_n" in p for p in problems)


def test_validate_existing_row_reports_all_problems_simultaneously():
    existing = {"p33_cutoff": 200.0, "p66_cutoff": -10.0, "cohort_snapshot_n": 0}
    problems = _validate_existing_row(TARGET, existing)
    # p33>p66, p33 out-of-range, p66 out-of-range, cohort_snapshot_n yanlis -> 4 sorun
    assert len(problems) == 4


def test_rows_to_write_excludes_aggressiveness_score():
    names = {r["score_name"] for r in write_score_thresholds.ROWS_TO_WRITE}
    assert names == {"tmz_resistance_score", "dna_repair_score"}
    assert "aggressiveness_score" not in names
