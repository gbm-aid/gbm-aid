"""``GET /model_curves`` (api/model_curves.py) -- birim + gercek artefakt
duman testleri.

Gorev talimati: paralel-40 (backend-agent-Y1, 2026-09-16, Baris onayi).
Tek dogruluk kaynagi `artifacts/week3/cox_model/week3_v3b_calibration_365
.csv` + `week3_v3b_calibration_summary.csv` +
`week3_v3b_lowvar_v2amgmt_final_coefficients.csv` dosyalaridir -- bu modul
HICBIR sayiyi gommez/yuvarlamaz, hepsini bu dosyalardan okur. Testler IKI
katmanlidir: (1) sentetik gecici CSV'lerle saf mantik (DB'siz,
artefakt'siz), (2) gercek `artifacts/` dosyalariyla duman testi (dosyalar
YOKSA/tasinmissa `pytest.skip`, `test_api_model_performance.py`'deki AYNI
desen).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api.model_curves as mc_module


# =====================================================================
# 1) Saf mantik birim testleri -- yardimci fonksiyonlar
# =====================================================================


def test_typed_cell_converts_numeric_strings_to_float():
    assert mc_module._typed_cell("0.685") == pytest.approx(0.685)
    assert mc_module._typed_cell("") is None
    assert mc_module._typed_cell("test-kaynak") == "test-kaynak"


def test_as_whole_int_converts_integral_float():
    assert mc_module._as_whole_int(365.0, field_name="t_gun") == 365
    assert mc_module._as_whole_int(1.0, field_name="decile") == 1


def test_as_whole_int_raises_for_non_integral_float():
    with pytest.raises(mc_module.ModelCurvesArtifactError):
        mc_module._as_whole_int(1.5, field_name="decile")


def test_as_whole_int_raises_for_non_numeric():
    with pytest.raises(mc_module.ModelCurvesArtifactError):
        mc_module._as_whole_int(None, field_name="decile")
    with pytest.raises(mc_module.ModelCurvesArtifactError):
        mc_module._as_whole_int("metin", field_name="decile")


def test_read_csv_rows_raises_when_file_missing(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(mc_module.ModelCurvesArtifactError):
        mc_module._read_csv_rows(missing)


def test_read_csv_rows_raises_when_file_empty(tmp_path: Path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises(mc_module.ModelCurvesArtifactError):
        mc_module._read_csv_rows(empty_csv)


# =====================================================================
# 2) build_model_curves_block() -- sentetik gecici artefaktlarla
#    (gercek DB/artifacts'a bagimli DEGIL)
# =====================================================================


def _write_synthetic_calibration_365(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["decile", "n", "olay", "tahmin_S365_ort", "gozlenen_S365_KM", "KM_alt", "KM_ust"]
        )
        # Kasitli olarak TERS sirada -- sort mantigini test etmek icin.
        writer.writerow(["2", "29", "19", "0.4172", "0.5341", "0.3189", "0.7087"])
        writer.writerow(["1", "30", "17", "0.2647", "0.3626", "0.1553", "0.5755"])


def _write_synthetic_calibration_summary(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metrik", "t_gun", "deger", "ci_alt", "ci_ust", "kaynak"])
        writer.writerow(["kalibrasyon_egimi", "", "0.685", "0.470", "0.900", "test-kaynak-notu"])
        writer.writerow(["decile_mae", "365", "0.06925", "", "", "test-mae-notu"])


def _write_synthetic_coefficients(path: Path, *, n_radiomic: int = 16, n_clinical: int = 8) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "variant", "covariate", "coef", "exp(coef)", "se(coef)",
                "coef lower 95%", "coef upper 95%", "exp(coef) lower 95%",
                "exp(coef) upper 95%", "z", "p",
            ]
        )
        for i in range(n_radiomic):
            writer.writerow(
                [
                    "v3b_lowvar_v2amgmt", f"WT__original_feature_{i}", "-0.1", "0.9",
                    "0.05", "-0.2", "0.0", "0.82", "1.0", "-2.0", "0.045",
                ]
            )
        for i in range(n_clinical):
            writer.writerow(
                [
                    "v3b_lowvar_v2amgmt", f"clinical_feature_{i}", "0.2", "1.22",
                    "0.06", "0.08", "0.32", "1.08", "1.38", "3.3", "0.001",
                ]
            )


def _write_all_synthetic_artifacts(tmp_path: Path) -> dict[str, Path]:
    calibration_365 = tmp_path / "calibration_365.csv"
    calibration_summary = tmp_path / "calibration_summary.csv"
    coefficients = tmp_path / "coefficients.csv"
    _write_synthetic_calibration_365(calibration_365)
    _write_synthetic_calibration_summary(calibration_summary)
    _write_synthetic_coefficients(coefficients)
    return {
        "calibration_365": calibration_365,
        "calibration_summary": calibration_summary,
        "coefficients": coefficients,
    }


def test_build_model_curves_block_with_synthetic_artifacts(tmp_path: Path, monkeypatch):
    paths = _write_all_synthetic_artifacts(tmp_path)
    monkeypatch.setattr(mc_module, "COX_CALIBRATION_365_CSV", paths["calibration_365"])
    monkeypatch.setattr(mc_module, "COX_CALIBRATION_SUMMARY_CSV", paths["calibration_summary"])
    monkeypatch.setattr(mc_module, "COX_V3B_COEFFICIENTS_CSV", paths["coefficients"])

    result = mc_module.build_model_curves_block()

    calibration = result["calibration"]
    assert calibration["variant"] == "v3b_lowvar_v2amgmt"
    assert calibration["t_days"] == 365
    assert isinstance(calibration["t_days"], int)
    assert calibration["slope"]["value"] == pytest.approx(0.685)
    assert calibration["slope"]["ci_lower"] == pytest.approx(0.470)
    assert calibration["slope"]["ci_upper"] == pytest.approx(0.900)
    assert calibration["slope"]["source_note"] == "test-kaynak-notu"
    assert calibration["decile_mae"] == pytest.approx(0.06925)

    # Girdi ters sirada verildi -- cikti decile'a gore ARTAN sirali olmali.
    assert [d["decile"] for d in calibration["deciles"]] == [1, 2]
    first = calibration["deciles"][0]
    assert first["n"] == 30
    assert first["events"] == 17
    assert first["predicted_s365"] == pytest.approx(0.2647)
    assert first["observed_km"] == pytest.approx(0.3626)
    assert first["km_lower"] == pytest.approx(0.1553)
    assert first["km_upper"] == pytest.approx(0.5755)

    coefficients = result["coefficients"]
    assert coefficients["variant"] == "v3b_lowvar_v2amgmt"
    assert coefficients["n_covariates"] == 24
    assert len(coefficients["rows"]) == 24
    radiomic_rows = [r for r in coefficients["rows"] if r["kind"] == "radiomic"]
    clinical_rows = [r for r in coefficients["rows"] if r["kind"] == "clinical"]
    assert len(radiomic_rows) == 16
    assert len(clinical_rows) == 8
    assert all(r["covariate"].startswith("WT__") for r in radiomic_rows)
    assert all(not r["covariate"].startswith("WT__") for r in clinical_rows)

    sample = coefficients["rows"][0]
    assert sample["coef"] == pytest.approx(-0.1)
    assert sample["hr"] == pytest.approx(0.9)
    assert sample["hr_ci_lower"] == pytest.approx(0.82)
    assert sample["hr_ci_upper"] == pytest.approx(1.0)

    assert set(result["source_artifacts"]) == {
        "calibration_csv", "calibration_summary_csv", "coefficients_csv",
    }
    assert len(result["notes"]) >= 3


def test_build_model_curves_block_raises_when_covariate_count_wrong(tmp_path: Path, monkeypatch):
    paths = _write_all_synthetic_artifacts(tmp_path)
    # 15 radyomik + 8 klinik = 23 -- 24 SOZLESMESINI BOZAR.
    _write_synthetic_coefficients(paths["coefficients"], n_radiomic=15, n_clinical=8)
    monkeypatch.setattr(mc_module, "COX_CALIBRATION_365_CSV", paths["calibration_365"])
    monkeypatch.setattr(mc_module, "COX_CALIBRATION_SUMMARY_CSV", paths["calibration_summary"])
    monkeypatch.setattr(mc_module, "COX_V3B_COEFFICIENTS_CSV", paths["coefficients"])

    with pytest.raises(mc_module.ModelCurvesArtifactError):
        mc_module.build_model_curves_block()


def test_build_model_curves_block_raises_when_variant_name_mismatched(tmp_path: Path, monkeypatch):
    paths = _write_all_synthetic_artifacts(tmp_path)
    bad_coefficients = tmp_path / "bad_coefficients.csv"
    with bad_coefficients.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "variant", "covariate", "coef", "exp(coef)", "se(coef)",
                "coef lower 95%", "coef upper 95%", "exp(coef) lower 95%",
                "exp(coef) upper 95%", "z", "p",
            ]
        )
        writer.writerow(
            ["baska_varyant", "WT__x", "-0.1", "0.9", "0.05", "-0.2", "0.0", "0.82", "1.0", "-2.0", "0.045"]
        )
    monkeypatch.setattr(mc_module, "COX_CALIBRATION_365_CSV", paths["calibration_365"])
    monkeypatch.setattr(mc_module, "COX_CALIBRATION_SUMMARY_CSV", paths["calibration_summary"])
    monkeypatch.setattr(mc_module, "COX_V3B_COEFFICIENTS_CSV", bad_coefficients)

    with pytest.raises(mc_module.ModelCurvesArtifactError):
        mc_module.build_model_curves_block()


def test_get_model_curves_returns_503_when_artifact_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(mc_module, "COX_CALIBRATION_365_CSV", tmp_path / "does_not_exist.csv")

    with pytest.raises(HTTPException) as excinfo:
        mc_module.get_model_curves()
    assert excinfo.value.status_code == 503


def test_model_curves_endpoint_via_testclient_with_synthetic_artifacts(tmp_path: Path, monkeypatch):
    """`api.main`'i import edip TestClient ile gercek HTTP yolunu (router
    kaydi dahil) dogrular -- port 8000'deki calisan sunucuya DOKUNMAZ,
    ASGI uygulamasina bellek-ici cagri yapar."""

    paths = _write_all_synthetic_artifacts(tmp_path)
    monkeypatch.setattr(mc_module, "COX_CALIBRATION_365_CSV", paths["calibration_365"])
    monkeypatch.setattr(mc_module, "COX_CALIBRATION_SUMMARY_CSV", paths["calibration_summary"])
    monkeypatch.setattr(mc_module, "COX_V3B_COEFFICIENTS_CSV", paths["coefficients"])

    import api.main as main_module

    client = TestClient(main_module.app)
    response = client.get("/model_curves")
    assert response.status_code == 200
    body = response.json()
    assert body["calibration"]["variant"] == "v3b_lowvar_v2amgmt"
    assert body["coefficients"]["n_covariates"] == 24


# =====================================================================
# 3) Gercek artefakt duman testi -- dosyalar tasinmissa/yoksa ATLANIR.
# =====================================================================


def test_model_curves_real_artifact_smoke_has_24_covariates_16_plus_8():
    if not mc_module.COX_CALIBRATION_365_CSV.is_file():
        pytest.skip(f"Gercek artefakt bulunamadi, smoke test atlandi: {mc_module.COX_CALIBRATION_365_CSV}")
    if not mc_module.COX_V3B_COEFFICIENTS_CSV.is_file():
        pytest.skip(f"Gercek artefakt bulunamadi, smoke test atlandi: {mc_module.COX_V3B_COEFFICIENTS_CSV}")

    result = mc_module.build_model_curves_block()

    assert result["calibration"]["variant"] == "v3b_lowvar_v2amgmt"
    assert result["calibration"]["t_days"] == 365
    assert len(result["calibration"]["deciles"]) == 10
    assert result["calibration"]["slope"]["value"] == pytest.approx(0.685)
    assert result["calibration"]["slope"]["ci_lower"] == pytest.approx(0.470)
    assert result["calibration"]["slope"]["ci_upper"] == pytest.approx(0.900)

    coefficients = result["coefficients"]
    assert coefficients["n_covariates"] == 24
    n_radiomic = sum(1 for r in coefficients["rows"] if r["kind"] == "radiomic")
    n_clinical = sum(1 for r in coefficients["rows"] if r["kind"] == "clinical")
    assert n_radiomic == 16
    assert n_clinical == 8
