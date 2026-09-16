"""``GET /model_performance`` (api/model_performance.py) -- birim + gerçek
artefakt duman testleri.

Görev talimatı: paralel-38 (backend-agent-X1, 2026-09-16). Tek doğruluk
kaynağı `artifacts/week3/cox_model/*` + `artifacts/week4/xgboost_v2a_mgmt_
reduce_collinearity_0912/*` dosyalarıdır -- bu modül HİÇBİR sayıyı
gömmez/yuvarlamaz, hepsini bu dosyalardan okur. Testler İKİ katmanlıdır:
(1) sentetik geçici CSV/JSON ile saf mantık (DB'siz, artefakt'sız),
(2) gerçek `artifacts/` dosyalarıyla duman testi (dosyalar YOKSA/taşınmışsa
`pytest.skip`, mevcut desenle AYNI).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

import api.model_performance as mp_module


# =====================================================================
# 1) Saf mantık birim testleri -- yardımcı fonksiyonlar
# =====================================================================


def test_typed_cell_converts_numeric_strings_to_float():
    assert mp_module._typed_cell("0.6592") == pytest.approx(0.6592)
    assert mp_module._typed_cell("") is None
    assert mp_module._typed_cell("611/585") == "611/585"
    assert mp_module._typed_cell("TAMAM") == "TAMAM"


def test_parse_ci_bracket_extracts_two_floats():
    assert mp_module._parse_ci_bracket("[0.6529, 0.7599]") == [
        pytest.approx(0.6529),
        pytest.approx(0.7599),
    ]


def test_parse_ci_bracket_returns_none_for_non_string_or_unmatched():
    assert mp_module._parse_ci_bracket(None) is None
    assert mp_module._parse_ci_bracket(0.65) is None
    assert mp_module._parse_ci_bracket("not-a-range") is None


def test_read_csv_rows_raises_when_file_missing(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(mp_module.ModelPerformanceArtifactError):
        mp_module._read_csv_rows(missing)


def test_read_json_raises_when_file_missing(tmp_path: Path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(mp_module.ModelPerformanceArtifactError):
        mp_module._read_json(missing)


def test_read_csv_rows_raises_when_file_unparseable_is_actually_empty(tmp_path: Path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises(mp_module.ModelPerformanceArtifactError):
        mp_module._read_csv_rows(empty_csv)


def test_read_json_raises_when_file_is_invalid_json(tmp_path: Path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(mp_module.ModelPerformanceArtifactError):
        mp_module._read_json(bad_json)


# =====================================================================
# 2) build_model_performance_block() -- sentetik geçici artefaktlarla
#    (gerçek DB/artifacts'a bağımlı DEĞİL)
# =====================================================================


def _write_synthetic_artifacts(tmp_path: Path) -> None:
    variants_csv = tmp_path / "variants.csv"
    with variants_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "variant", "rol", "secim_durumu", "egitim_havuzu",
                "harici_c_index", "ci_alt", "ci_ust", "harici_n",
                "harici_olay", "ic_cv_ort", "ic_cv_std", "aday_ozellik",
            ]
        )
        writer.writerow(
            ["v_a", "varyant", "duyarlilik_kolu", "611/585", "0.66", "0.62", "0.70", "295", "169", "0.64", "0.03", "93"]
        )
        writer.writerow(
            ["v_b", "varyant", "SECILEN_BIRINCIL_MODEL", "611/585", "0.65", "0.61", "0.70", "295", "169", "0.66", "0.03", "54"]
        )

    extended_csv = tmp_path / "extended.csv"
    with extended_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "variant", "gate", "uno_c_tau365", "uno_c_tau365_ci",
                "td_auc_365", "td_auc_365_ci", "uno_c_tau548", "uno_c_tau548_ci",
                "td_auc_548", "td_auc_548_ci", "uno_c_tau730", "uno_c_tau730_ci",
                "td_auc_730", "td_auc_730_ci", "n_failed_resamples", "n_valid_min",
            ]
        )
        # yalnız v_a icin satir -- v_b KASITLI OLARAK eksik birakildi,
        # eslesme-bulunamadi yolunu test etmek icin.
        writer.writerow(
            [
                "v_a", "GECTI", "0.70", "[0.65, 0.75]", "0.75", "[0.68, 0.81]",
                "0.67", "[0.62, 0.72]", "0.71", "[0.64, 0.78]", "0.66", "[0.62, 0.70]",
                "0.71", "[0.63, 0.79]", "0", "1000",
            ]
        )

    calibration_csv = tmp_path / "calibration.csv"
    with calibration_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metrik", "t_gun", "deger", "ci_alt", "ci_ust", "kaynak"])
        writer.writerow(["kalibrasyon_egimi", "", "0.685", "0.470", "0.900", "test-kaynak"])

    run_metadata_json = tmp_path / "run_metadata.json"
    run_metadata_json.write_text(
        json.dumps({"xgboost_auc_out_of_fold": {"auc": 0.72, "ci_lower": 0.68, "ci_upper": 0.76}}),
        encoding="utf-8",
    )

    ucsf_json = tmp_path / "ucsf_external.json"
    ucsf_json.write_text(
        json.dumps(
            {
                "variant": "v_b",
                "n_upenn_training_patients": 611,
                "n_upenn_training_events": 585,
                "ucsf_auc": {"auc": 0.72, "ci_lower": 0.65, "ci_upper": 0.79},
                "reporting_rule_note": "test-note",
            }
        ),
        encoding="utf-8",
    )


def test_build_model_performance_block_with_synthetic_artifacts(tmp_path: Path, monkeypatch):
    _write_synthetic_artifacts(tmp_path)
    monkeypatch.setattr(mp_module, "COX_VARIANTS_CSV", tmp_path / "variants.csv")
    monkeypatch.setattr(mp_module, "COX_EXTERNAL_METRICS_CSV", tmp_path / "extended.csv")
    monkeypatch.setattr(mp_module, "COX_CALIBRATION_CSV", tmp_path / "calibration.csv")
    monkeypatch.setattr(mp_module, "XGBOOST_RUN_METADATA_JSON", tmp_path / "run_metadata.json")
    monkeypatch.setattr(mp_module, "XGBOOST_UCSF_EXTERNAL_JSON", tmp_path / "ucsf_external.json")

    result = mp_module.build_model_performance_block()

    assert len(result["cox_variants"]) == 2
    by_variant = {row["variant"]: row for row in result["cox_variants"]}

    # secilen_mi -- yalniz "SECILEN" alt dizesi tasiyan satir True olmali.
    assert by_variant["v_a"]["secilen_mi"] is False
    assert by_variant["v_b"]["secilen_mi"] is True

    # Sayisal alanlar HAM float olarak gelmeli, string DEGIL.
    assert by_variant["v_a"]["harici_c_index"] == pytest.approx(0.66)
    assert isinstance(by_variant["v_a"]["harici_c_index"], float)

    # td_auc birlestirmesi -- v_a icin VAR, v_b icin YOK (drift senaryosu).
    assert by_variant["v_a"]["td_auc"]["tau730"]["td_auc"] == pytest.approx(0.71)
    assert by_variant["v_a"]["td_auc"]["tau730"]["td_auc_ci"] == [
        pytest.approx(0.63), pytest.approx(0.79),
    ]
    assert by_variant["v_b"]["td_auc"] is None
    assert "bulunamadi" in by_variant["v_b"]["td_auc_not_available_reason"]

    assert result["calibration"][0]["metrik"] == "kalibrasyon_egimi"
    assert result["calibration"][0]["deger"] == pytest.approx(0.685)

    assert result["xgboost"]["variant"] == "v_b"
    assert result["xgboost"]["oof_auc"]["auc"] == pytest.approx(0.72)
    assert result["xgboost"]["ucsf_external_auc"]["auc"] == pytest.approx(0.72)
    assert result["xgboost"]["n_upenn_training_patients"] == 611

    # Kaynak artefakt yollari yanitta seffaf olmali.
    assert set(result["source_artifacts"]) == {
        "cox_variants_csv", "cox_external_metrics_ci_csv", "cox_calibration_csv",
        "xgboost_run_metadata_json", "xgboost_ucsf_external_evaluation_json",
    }


def test_get_model_performance_returns_503_when_artifact_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(mp_module, "COX_VARIANTS_CSV", tmp_path / "does_not_exist.csv")

    with pytest.raises(HTTPException) as excinfo:
        mp_module.get_model_performance()
    assert excinfo.value.status_code == 503


# =====================================================================
# 3) Gerçek artefakt duman testi -- dosyalar taşınmışsa/yoksa ATLANIR.
# =====================================================================


def test_model_performance_real_artifact_smoke_has_eight_cox_variants_and_selected_v3b():
    if not mp_module.COX_VARIANTS_CSV.is_file():
        pytest.skip(f"Gercek artefakt bulunamadi, smoke test atlandi: {mp_module.COX_VARIANTS_CSV}")

    result = mp_module.build_model_performance_block()

    assert len(result["cox_variants"]) == 8
    by_variant = {row["variant"]: row for row in result["cox_variants"]}
    assert "v3b_lowvar_v2amgmt" in by_variant
    assert by_variant["v3b_lowvar_v2amgmt"]["secilen_mi"] is True
    # Yalniz TEK satir secilen_mi=True olmali.
    assert sum(1 for row in result["cox_variants"] if row["secilen_mi"]) == 1

    # v2c/v3c egitim havuzu 609/583, digerleri 611/585 (CLAUDE.md kilitli
    # ayrimi) -- burada DUZELTILMEDEN oldugu gibi tasinmali.
    assert by_variant["v2c_mgmt_spline_wttc"]["egitim_havuzu"] == "609/583"
    assert by_variant["v3c_lowvar_wttc_mgmt_nospline"]["egitim_havuzu"] == "609/583"
    assert by_variant["v1_referans"]["egitim_havuzu"] == "611/585"

    assert result["xgboost"]["variant"] == "v2a_mgmt"
    assert result["xgboost"]["oof_auc"] is not None
    assert result["xgboost"]["ucsf_external_auc"] is not None
