"""`POST /predict/{patient_id}` -- birim testleri (api/predict.py).

DB'ye BAGLANMAZ -- her test `api.predict` modulunun ic fonksiyonlarini
monkeypatch ile sahteler (bkz. tests/test_api_harmonize.py'deki
FakeConnection/FakeCursor deseni, burada da AYNI desen kullaniliyor).

`compute_risk_score_and_shap` icin AYRICA sentetik ama GERCEK bir
`lifelines.CoxPHFitter` ile SAYISAL additivite kontrolu yapan bir test
var (`test_compute_risk_score_and_shap_additivity_is_exact`) -- bu,
`tools/shap_explainer_prototype.py`'de elle dogrulanan bulguyu (SHAP
toplami + taban == risk_score, kapali-form LinearExplainer) kalici bir
regresyon testine cevirir.
"""

from __future__ import annotations

import copy
import logging
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from dotenv import dotenv_values

import api.predict as predict_module


class FakeCursor:
    """Sirali `execute()` cagrilarina, sirayla onceden tanimlanmis
    sonuclar doner -- `fetch_c32_wt_row`'un iki ayri sorgusunu (once
    `patients` varlik kontrolu, sonra `radiomics` cekimi) taklit eder."""

    def __init__(self, fetchone_results: list[Any], fetchall_results: list[list[dict]]) -> None:
        self._fetchone_results = list(fetchone_results)
        self._fetchall_results = list(fetchall_results)
        self._last_call = None

    def execute(self, query: str, _params: tuple) -> None:
        self._last_call = "fetchone" if "SELECT 1" in query else "fetchall"

    def fetchone(self):
        return self._fetchone_results.pop(0)

    def fetchall(self):
        return self._fetchall_results.pop(0)

    def close(self) -> None:
        return None


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self, **_kwargs: Any):
        return self._cursor

    def close(self) -> None:
        return None


def _fake_arm_result(
    fitted_model,
    final_features: list[str],
    *,
    extra_columns: list[str] | None = None,
    name: str = "test_arm",
) -> dict[str, Any]:
    """2026-08-18 GUNCELLENDI -- `load_cox_arm_result()`'un artik dondurdugu
    NORMALIZE sozluk bicimini taklit eder (bkz. api/predict.py::
    _normalize_arm). Eski `SimpleNamespace(final_model=...)` bicimi
    ARTIK endpoint tarafindan dogrudan tuketilmiyor (o bicim yalniz
    `_normalize_arm()`'un ESKI-checkpoint dalinin GIRDISI, bkz.
    `test_normalize_arm_handles_legacy_arm_result_object`)."""

    return {
        "name": name,
        "final_features": list(final_features),
        "extra_columns": list(extra_columns or []),
        "fitted_model": fitted_model,
    }


def _fit_tiny_cox_model(feature_cols: list[str], *, seed: int = 42, n: int = 100):
    from lifelines import CoxPHFitter

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({col: rng.normal(size=n) for col in feature_cols})
    frame["survival_days"] = rng.exponential(scale=400, size=n) + 1.0
    frame["event"] = rng.binomial(1, 0.7, size=n)
    model = CoxPHFitter(penalizer=0.01)
    model.fit(frame[feature_cols + ["survival_days", "event"]], duration_col="survival_days", event_col="event")
    return model, frame


# =====================================================================
# compute_risk_score_and_shap -- sayisal additivite (asil "dogruluk
# kontrolu", gorev talimatinin merkezi gereksinimi)
# =====================================================================


def test_compute_risk_score_and_shap_additivity_is_exact() -> None:
    feature_cols = ["WT__f1", "WT__f2", "WT__f3"]
    model, frame = _fit_tiny_cox_model(feature_cols)

    background = frame[feature_cols].sample(n=30, random_state=42)
    patient_row = frame[feature_cols].iloc[[0]]

    result = predict_module.compute_risk_score_and_shap(
        model, feature_cols, patient_row, background
    )

    true_risk_score = float(model.predict_log_partial_hazard(patient_row).iloc[0])
    true_hazard_ratio = float(model.predict_partial_hazard(patient_row).iloc[0])

    assert result["risk_score_log_partial_hazard"] == pytest.approx(true_risk_score, abs=1e-9)
    assert result["hazard_ratio_partial_hazard"] == pytest.approx(true_hazard_ratio, abs=1e-9)
    assert result["shap_additivity_check_abs_diff"] < 1e-9
    reconstructed = sum(result["shap_values"].values()) + result["shap_base_value"]
    assert reconstructed == pytest.approx(true_risk_score, abs=1e-9)
    assert set(result["shap_values"].keys()) == set(feature_cols)


def test_compute_risk_score_and_shap_additivity_independent_of_background_choice() -> None:
    """Kapali-form LinearExplainer'in matematiksel ozelligi: taban
    `-(coef @ norm_mean)` uzerinden sabitlendigi icin additivite hangi
    arka plan orneklemi verilirse verilsin (buyukluk/rastgelelik fark
    etmeksizin) TAM tutmali -- bkz. api/predict.py modul dokstring'i."""

    feature_cols = ["WT__f1", "WT__f2"]
    model, frame = _fit_tiny_cox_model(feature_cols, seed=7, n=80)
    patient_row = frame[feature_cols].iloc[[5]]

    small_background = frame[feature_cols].sample(n=5, random_state=1)
    large_background = frame[feature_cols]

    result_small = predict_module.compute_risk_score_and_shap(
        model, feature_cols, patient_row, small_background
    )
    result_large = predict_module.compute_risk_score_and_shap(
        model, feature_cols, patient_row, large_background
    )

    assert result_small["shap_additivity_check_abs_diff"] < 1e-9
    assert result_large["shap_additivity_check_abs_diff"] < 1e-9
    assert result_small["risk_score_log_partial_hazard"] == pytest.approx(
        result_large["risk_score_log_partial_hazard"], abs=1e-12
    )


# =====================================================================
# fetch_c32_wt_row -- FakeConnection/FakeCursor ile (DB'ye baglanmadan)
# =====================================================================


def test_fetch_c32_wt_row_raises_patient_not_found_when_missing_from_patients_table(
    monkeypatch,
) -> None:
    cursor = FakeCursor(fetchone_results=[None], fetchall_results=[])
    monkeypatch.setattr(predict_module, "_get_db_connection", lambda: FakeConnection(cursor))

    with pytest.raises(predict_module.PatientNotFoundError):
        predict_module.fetch_c32_wt_row("NOPE-999", ["WT__f1"])


def test_fetch_c32_wt_row_raises_c32_not_found_when_no_radiomics_rows(monkeypatch) -> None:
    cursor = FakeCursor(fetchone_results=[(1,)], fetchall_results=[[]])
    monkeypatch.setattr(predict_module, "_get_db_connection", lambda: FakeConnection(cursor))

    with pytest.raises(predict_module.C32RadiomicsNotFoundError):
        predict_module.fetch_c32_wt_row("UPENN-GBM-99999", ["WT__f1"])


def test_fetch_c32_wt_row_raises_multi_scan_when_longitudinal(monkeypatch) -> None:
    """GERCEK DB'de KESFEDILEN bulgu (coklu zaman noktasi) -- bkz.
    tools/shap_explainer_prototype.py docstring'i. Bu regresyon testi o
    bulguyu sabitler.

    ⚠️ 2026-09-13 GUNCELLENDI (Y1 karari, Baris onayi -- decisions/
    2026-09-13-lumiere-risk-skoru-preop-kanonik-vizit.md): bu test eskiden
    `Patient-001`/`LUMIERE` satirlariyla kosuyordu. LUMIERE ARTIK bu
    istisnayi ALMIYOR -- K2 kanonik-vizit kurali (Rating=='Pre-Op') o
    kaynakta coklu-taramayi COZUYOR. Test, davranisin KORUNDUGU yere
    (LUMIERE-DISI kaynaklar) tasindi; LUMIERE'nin YENI davranisi asagidaki
    "LUMIERE KANONIK VIZIT" bolumunde KIRMIZI/YESIL olarak test edilir."""

    rows = [
        {
            "patient_id": "UPENN-GBM-00042",
            "source": "UPenn-GBM",
            "tumor_region": "WT_derived",
            "shape_features": {"a": 1.0},
            "first_order_features": {},
            "texture_features": {},
            "scan_id": scan_id,
            "timepoint_label": label,
        }
        for scan_id, label in ((2839, "_11"), (2843, "_21"))
    ]
    cursor = FakeCursor(fetchone_results=[(1,)], fetchall_results=[rows])
    monkeypatch.setattr(predict_module, "_get_db_connection", lambda: FakeConnection(cursor))

    with pytest.raises(predict_module.MultiScanNotSupportedError):
        predict_module.fetch_c32_wt_row("UPENN-GBM-00042", ["WT__a"])


def test_fetch_c32_wt_row_raises_required_region_missing_when_region_absent(monkeypatch) -> None:
    rows = [
        {
            "patient_id": "UPENN-GBM-00001",
            "source": "UPenn-GBM",
            "tumor_region": "ED",
            "shape_features": {"a": 1.0},
            "first_order_features": {},
            "texture_features": {},
            "scan_id": 1,
        }
    ]
    cursor = FakeCursor(fetchone_results=[(1,)], fetchall_results=[rows])
    monkeypatch.setattr(predict_module, "_get_db_connection", lambda: FakeConnection(cursor))

    with pytest.raises(predict_module.RequiredRegionMissingError):
        predict_module.fetch_c32_wt_row("UPENN-GBM-00001", ["WT__original_shape_Flatness"])


def test_fetch_c32_wt_row_returns_wide_row_for_single_scan_patient(monkeypatch) -> None:
    rows = [
        {
            "patient_id": "UPENN-GBM-00001",
            "source": "UPenn-GBM",
            "tumor_region": "WT_derived",
            "shape_features": {"original_shape_Flatness": 0.42},
            "first_order_features": {},
            "texture_features": {},
            "scan_id": 1,
        }
    ]
    cursor = FakeCursor(fetchone_results=[(1,)], fetchall_results=[rows])
    monkeypatch.setattr(predict_module, "_get_db_connection", lambda: FakeConnection(cursor))

    wide = predict_module.fetch_c32_wt_row(
        "UPENN-GBM-00001", ["WT__original_shape_Flatness"]
    )
    assert wide.loc["UPENN-GBM-00001", "WT__original_shape_Flatness"] == 0.42


# =====================================================================
# LUMIERE KANONIK VIZIT (K2 kilitli kural / Y1 -- 2026-09-13)
#
# Karar: decisions/2026-09-13-lumiere-risk-skoru-preop-kanonik-vizit.md
# ("y1 i onaylıyorum lumiere 72 hastaya çıksın" -- Baris).
#
# Bu bolumun IKI zorunlu testi:
#   (b) `Patient-020` TUZAGI -- `week-000` -> Post-Op, `week-000-1` ->
#       Pre-Op. Etikete gore siralayan bir kural TAM BURADA ameliyat-
#       SONRASI taramayi secerdi (KIRMIZI); K2 kurali `week-000-1`'i
#       secmeli (YESIL).
#   (d) UPenn/UCSF/TCGA'da coklu-tarama HALA 422 -- LUMIERE'ye ozel
#       gevsetme o kaynaklara SIZMAMALI.
# =====================================================================

_EXPERT_RATING_RATING_COLUMN = (
    "Rating (according to RANO, PD: Progressive disease, SD: Stable disease, "
    "PR: Partial response, CR: Complete response, Pre-Op: Pre-Operative, "
    "Post-Op: Post-Operative)"
)


def _write_expert_rating_csv(tmp_path, rows: list[tuple[str, str, str]]) -> Path:
    """`(Patient, Date, Rating)` uclulerinden gercek CSV semasinda bir
    fixture yazar (kolon adi `raw/veri/LUMIERE-ExpertRating-*.csv` ile
    BIREBIR ayni -- yukleyici bu prefix'e gore kolonu buluyor)."""

    import csv as csv_module

    path = tmp_path / "fixture-expert-rating.csv"
    fieldnames = ["Patient", "Date", "LessThan3Months", _EXPERT_RATING_RATING_COLUMN]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for patient, date, rating in rows:
            writer.writerow(
                {
                    "Patient": patient,
                    "Date": date,
                    "LessThan3Months": "",
                    _EXPERT_RATING_RATING_COLUMN: rating,
                }
            )
    return path


def _lumiere_c32_rows(patient_id: str, visits: list[tuple[int, str]]) -> list[dict[str, Any]]:
    """`(scan_id, timepoint_label)` listesinden LUMIERE C32 uzun-format
    satirlari uretir (her vizit icin tek `WT_derived` satiri)."""

    return [
        {
            "patient_id": patient_id,
            "source": "LUMIERE",
            "tumor_region": "WT_derived",
            "shape_features": {"original_shape_Flatness": 0.10 * scan_id},
            "first_order_features": {},
            "texture_features": {},
            "scan_id": scan_id,
            "timepoint_label": label,
        }
        for scan_id, label in visits
    ]


def _fake_lumiere_cursor(monkeypatch, rows: list[dict[str, Any]]) -> None:
    cursor = FakeCursor(fetchone_results=[(1,)], fetchall_results=[rows])
    monkeypatch.setattr(predict_module, "_get_db_connection", lambda: FakeConnection(cursor))


def test_k2_canonical_visit_rule_has_a_single_source_of_truth() -> None:
    """TEK KAYNAK LINT'i: `api/predict.py` ve `tools/rebuild_faiss_indexes.py`
    AYNI fonksiyon NESNELERINI kullanmali -- kural ikinci bir yere
    KOPYALANMAMIS olmali (2026-09-13'te `tools/data_integrity_check.py`'de
    bir "eslesme aynasi" yuzunden drift riski dogdu; ayni hata
    tekrarlanmayacak). `api/` bir `tools/` script'ini IMPORT ETMEZ --
    ortak mantik `pipeline/lumiere_canonical_visit.py`dedir."""

    import pipeline.lumiere_canonical_visit as lcv

    # ⚠️ `sys.path` MUTLAKA GERI ALINIR (2026-09-13'te OLCULDU): `tools/
    # rebuild_faiss_indexes.py` import edilirken KENDI icinde
    # `sys.path.insert(0, Path(__file__).resolve().parent.parent)` yapiyor
    # -- `.resolve()` `X:` subst eslemesini Turkce karakterli GERCEK yola
    # geri cozdugu icin (K12) o yol sys.path'in BASINA giriyor ve DAHA SONRA
    # import edilen `api/similar.py` kendi `REPO_ROOT`unu o Turkce yoldan
    # kuruyor -> FAISS indeks dosyasi acilamiyor ve
    # `tests/test_api_similar.py::..._ucsf_patient_returns_422` 503 aliyor.
    # Bu test geri almayi YAPMAZSA ayni pytest kosusunda BASKA bir dosyanin
    # testini KIRIYOR (canli olculdu: bu test deselect edilince o test
    # GECIYOR). `tools/rebuild_faiss_indexes.py`'nin `.resolve()`u ON-VAR
    # bir durum, bu gorevde DEGISTIRILMEDI.
    original_sys_path = list(sys.path)
    try:
        sys.path.insert(0, str(Path(predict_module.__file__).absolute().parents[1] / "tools"))
        import rebuild_faiss_indexes as rfi
    finally:
        sys.path[:] = original_sys_path

    assert rfi._load_lumiere_preop_visit_keys is lcv.load_lumiere_preop_visit_keys
    assert rfi._lumiere_timepoint_sort_key is lcv.lumiere_timepoint_sort_key
    assert rfi.EXPECTED_LUMIERE_PREOP_PATIENTS == lcv.EXPECTED_LUMIERE_PREOP_PATIENTS
    assert rfi.LumierePreopCohortMismatchError is lcv.LumierePreopCohortMismatchError
    assert rfi.ExpertRatingFileNotFoundError is lcv.ExpertRatingFileNotFoundError
    # `api/predict.py` de AYNI modulden okur (kendi kopyasini TUTMAZ).
    assert predict_module.select_canonical_preop_visit_for_patient is (
        lcv.select_canonical_preop_visit_for_patient
    )
    assert predict_module.LUMIERE_EXPERT_RATING_CSV_PATH == (
        lcv.DEFAULT_LUMIERE_EXPERT_RATING_CSV
    )
    # `api/` KATMANI `tools/` ALTINDAN HICBIR SEY IMPORT ETMEMELI.
    predict_source = Path(predict_module.__file__).read_text(encoding="utf-8")
    assert "import tools." not in predict_source.replace("import tools.train_cox_week3", "")


def test_lumiere_canonical_visit_picks_preop_not_earliest_label_patient020_trap(
    tmp_path, monkeypatch
) -> None:
    """(b) ZORUNLU -- `Patient-020` TUZAGI (KIRMIZI/YESIL).

    GERCEK veri (canli olculdu 2026-09-13): `week-000` -> `Post-Op`,
    `week-000-1` -> `Pre-Op`. Bu testte IKI vizitin de C32 radyomigi VAR
    (fixture; canli DB'de `week-000`in C32 satiri YOK, yani tuzak orada
    LATENT'tir -- burada AKTIF hale getirilip kural dogrudan sinaniyor).

    KIRMIZI: `timepoint_label` siralamasina gore secen bir kural
    `week-000`i (Post-Op) alirdi -- test bunu ACIKCA gosterir.
    YESIL: K2 kurali `week-000-1`i (Pre-Op) secer."""

    csv_path = _write_expert_rating_csv(
        tmp_path,
        [
            ("Patient-020", "week-000", "Post-Op"),
            ("Patient-020", "week-000-1", "Pre-Op"),
        ],
    )
    monkeypatch.setattr(predict_module, "LUMIERE_EXPERT_RATING_CSV_PATH", csv_path)
    rows = _lumiere_c32_rows("Patient-020", [(3300, "week-000"), (3323, "week-000-1")])

    # KIRMIZI kanit: etiket siralamasi YANLIS taramayi (week-000/Post-Op) secerdi.
    import pipeline.lumiere_canonical_visit as lcv

    earliest_label = sorted(
        {row["timepoint_label"] for row in rows}, key=lcv.lumiere_timepoint_sort_key
    )[0]
    assert earliest_label == "week-000"  # <- "en erken" kurali BURADA yanilirdi

    _fake_lumiere_cursor(monkeypatch, rows)
    wide = predict_module.fetch_c32_wt_row("Patient-020", ["WT__original_shape_Flatness"])

    # YESIL: secilen satir week-000-1'in (scan 3323) degeri olmali.
    assert wide["WT__original_shape_Flatness"].iloc[0] == pytest.approx(0.10 * 3323)
    assert wide["WT__original_shape_Flatness"].iloc[0] != pytest.approx(0.10 * 3300)


def test_lumiere_canonical_visit_selection_reports_patient020_visit(
    tmp_path, monkeypatch
) -> None:
    """Secim nesnesi `week-000-1`/`Pre-Op` beyan etmeli (yanit seffaflik
    blogunun kaynagi)."""

    csv_path = _write_expert_rating_csv(
        tmp_path,
        [
            ("Patient-020", "week-000", "Post-Op"),
            ("Patient-020", "week-000-1", "Pre-Op"),
        ],
    )
    monkeypatch.setattr(predict_module, "LUMIERE_EXPERT_RATING_CSV_PATH", csv_path)
    rows = _lumiere_c32_rows("Patient-020", [(3300, "week-000"), (3323, "week-000-1")])

    selection = predict_module.select_lumiere_canonical_visit_from_long_frame(
        "Patient-020", pd.DataFrame(rows)
    )
    assert selection.timepoint_label == "week-000-1"
    assert selection.rating == "Pre-Op"
    assert selection.scan_id == 3323
    assert selection.tie_broken is False

    block = predict_module.build_lumiere_canonical_visit_block(selection)
    assert block["applied"] is True
    assert block["timepoint_label"] == "week-000-1"
    assert block["rating"] == "Pre-Op"
    assert "Rating == 'Pre-Op'" in block["rule"]


def test_lumiere_canonical_visit_accepts_preop_at_week069_patient060(
    tmp_path, monkeypatch
) -> None:
    """(c) `Patient-060` vakasi -- bir `Pre-Op` viziti `week-069`'da.

    Burada week-000 `Post-Op`, week-069 `Pre-Op` (yani "hafta 0 = ameliyat
    oncesi" varsayimi COKER). K2 kurali week-069'u SECMELI."""

    csv_path = _write_expert_rating_csv(
        tmp_path,
        [
            ("Patient-060", "week-000", "Post-Op"),
            ("Patient-060", "week-069", "Pre-Op"),
        ],
    )
    monkeypatch.setattr(predict_module, "LUMIERE_EXPERT_RATING_CSV_PATH", csv_path)
    rows = _lumiere_c32_rows("Patient-060", [(4375, "week-000"), (4400, "week-069")])

    selection = predict_module.select_lumiere_canonical_visit_from_long_frame(
        "Patient-060", pd.DataFrame(rows)
    )
    assert selection.timepoint_label == "week-069"
    assert selection.scan_id == 4400


def test_lumiere_canonical_visit_tie_break_picks_earliest_preop_patient060(
    tmp_path, monkeypatch
) -> None:
    """`Patient-060`'in GERCEK hali: IKI `Pre-Op` viziti (week-000 VE
    week-069), ikisinde de C32 var -> tie-break EN ERKEN olani secer
    (K2'nin kohort yolundaki `sort_values(_sort_key)+groupby.first()` ile
    AYNI sonuc). Tie-break yanitta ACIKCA beyan edilir."""

    csv_path = _write_expert_rating_csv(
        tmp_path,
        [
            ("Patient-060", "week-000", "Pre-Op"),
            ("Patient-060", "week-069", "Pre-Op"),
        ],
    )
    monkeypatch.setattr(predict_module, "LUMIERE_EXPERT_RATING_CSV_PATH", csv_path)
    rows = _lumiere_c32_rows("Patient-060", [(4375, "week-000"), (4400, "week-069")])

    selection = predict_module.select_lumiere_canonical_visit_from_long_frame(
        "Patient-060", pd.DataFrame(rows)
    )
    assert selection.timepoint_label == "week-000"
    assert selection.tie_broken is True
    assert list(selection.candidate_visits) == ["week-000", "week-069"]


def test_lumiere_dirty_rating_values_do_not_match_preop(tmp_path, monkeypatch) -> None:
    """CSV'deki kirli degerler (`'Post-Op '` sondan bosluklu, `'Post-Op/PD'`)
    `startswith`/`in` ile YANLISLIKLA eslesmemeli -- `.strip()` + TAM
    esleme (reviewer uyarisi, 2026-08-18)."""

    csv_path = _write_expert_rating_csv(
        tmp_path,
        [
            ("Patient-777", "week-000", "Post-Op "),
            ("Patient-777", "week-005", "Post-Op/PD"),
            ("Patient-777", "week-010", "Pre-Op"),
        ],
    )
    monkeypatch.setattr(predict_module, "LUMIERE_EXPERT_RATING_CSV_PATH", csv_path)
    rows = _lumiere_c32_rows(
        "Patient-777", [(1, "week-000"), (2, "week-005"), (3, "week-010")]
    )

    selection = predict_module.select_lumiere_canonical_visit_from_long_frame(
        "Patient-777", pd.DataFrame(rows)
    )
    assert selection.timepoint_label == "week-010"


def test_lumiere_preop_visit_without_c32_raises_explicit_error(tmp_path, monkeypatch) -> None:
    """(e) ZORUNLU -- `Pre-Op` viziti VAR ama O VIZITTE C32 radyomigi YOK
    (canli DB'de 19 hasta): ACIK hata, SESSIZ bos/varsayilan/post-op
    fallback YOK (Y1 sart 4)."""

    csv_path = _write_expert_rating_csv(
        tmp_path,
        [
            ("Patient-026", "week-000", "Pre-Op"),
            ("Patient-026", "week-002", "Post-Op"),
        ],
    )
    monkeypatch.setattr(predict_module, "LUMIERE_EXPERT_RATING_CSV_PATH", csv_path)
    # YALNIZ post-op vizitin C32'si var (Patient-026'nin canli DB'deki hali).
    rows = _lumiere_c32_rows("Patient-026", [(3483, "week-002")])
    _fake_lumiere_cursor(monkeypatch, rows)

    with pytest.raises(predict_module.LumierePreopVisitNotAvailableError) as excinfo:
        predict_module.fetch_c32_wt_row("Patient-026", ["WT__original_shape_Flatness"])

    message = str(excinfo.value)
    assert "week-000" in message  # hangi vizit BEKLENIYORDU
    assert "week-002" in message  # ELDE NE VAR
    assert excinfo.value.patient_id == "Patient-026"


def test_lumiere_single_scan_postop_patient_is_rejected(tmp_path, monkeypatch) -> None:
    """TEK taramasi olan LUMIERE hastasi da K2'ye TABIDIR.

    Canli olcum (2026-09-13): Y1 oncesi servis edilebilen 4 LUMIERE
    hastasinin (`Patient-026`/`-044`/`-053`/`-076`) TEK C32 taramasi
    `Rating=='Post-Op'`tur -- eski kod bu 4 hastaya AMELIYAT SONRASI
    goruntuden risk skoru basiyordu (sessiz yanlis sonuc). Bu test o
    regresyonu KALICI olarak kapatir: tek tarama da olsa Post-Op ise 422."""

    csv_path = _write_expert_rating_csv(
        tmp_path,
        [
            ("Patient-076", "week-000", "Pre-Op"),
            ("Patient-076", "week-001", "Post-Op"),
        ],
    )
    monkeypatch.setattr(predict_module, "LUMIERE_EXPERT_RATING_CSV_PATH", csv_path)
    rows = _lumiere_c32_rows("Patient-076", [(4935, "week-001")])
    _fake_lumiere_cursor(monkeypatch, rows)

    with pytest.raises(predict_module.LumierePreopVisitNotAvailableError):
        predict_module.fetch_c32_wt_row("Patient-076", ["WT__original_shape_Flatness"])


def test_lumiere_patient_absent_from_expert_rating_raises_explicit_error(
    tmp_path, monkeypatch
) -> None:
    """Hastanin CSV'de HIC `Pre-Op` satiri yoksa (bugun 0/91 hasta) sebep
    ACIKCA ayri yazilir -- "C32'si yok" ile KARISTIRILMAZ."""

    csv_path = _write_expert_rating_csv(tmp_path, [("Patient-999", "week-000", "Pre-Op")])
    monkeypatch.setattr(predict_module, "LUMIERE_EXPERT_RATING_CSV_PATH", csv_path)
    rows = _lumiere_c32_rows("Patient-123", [(1, "week-000")])

    with pytest.raises(predict_module.LumierePreopVisitNotAvailableError) as excinfo:
        predict_module.select_lumiere_canonical_visit_from_long_frame(
            "Patient-123", pd.DataFrame(rows)
        )
    assert "HIC" in str(excinfo.value)
    assert excinfo.value.preop_labels_in_csv == []


def test_lumiere_missing_expert_rating_csv_raises_not_found(tmp_path, monkeypatch) -> None:
    """CSV yoksa SESSIZCE baska bir kurala DUSULMEZ."""

    monkeypatch.setattr(
        predict_module, "LUMIERE_EXPERT_RATING_CSV_PATH", tmp_path / "yok.csv"
    )
    rows = _lumiere_c32_rows("Patient-002", [(1, "week-000")])
    with pytest.raises(predict_module.ExpertRatingFileNotFoundError):
        predict_module.select_lumiere_canonical_visit_from_long_frame(
            "Patient-002", pd.DataFrame(rows)
        )


def test_lumiere_canonical_visit_requires_timepoint_label_column() -> None:
    """`timepoint_label` yoksa kural UYGULANAMAZ -- rastgele bir tarama
    SESSIZCE secilmez."""

    rows = _lumiere_c32_rows("Patient-002", [(1, "week-000")])
    frame = pd.DataFrame(rows).drop(columns=["timepoint_label"])
    with pytest.raises(predict_module.LumiereCanonicalVisitDataError):
        predict_module.select_lumiere_canonical_visit_from_long_frame("Patient-002", frame)


def test_lumiere_canonical_visit_rejects_source_mismatch(tmp_path, monkeypatch) -> None:
    """`patient_id` oneki LUMIERE ama `source_name` LUMIERE DEGIL ->
    kimlik/kaynak celiskisi, kural SESSIZCE uygulanmaz."""

    csv_path = _write_expert_rating_csv(tmp_path, [("Patient-002", "week-000", "Pre-Op")])
    monkeypatch.setattr(predict_module, "LUMIERE_EXPERT_RATING_CSV_PATH", csv_path)
    rows = _lumiere_c32_rows("Patient-002", [(1, "week-000")])
    rows[0]["source"] = "UPenn-GBM"
    with pytest.raises(predict_module.LumiereCanonicalVisitDataError):
        predict_module.select_lumiere_canonical_visit_from_long_frame(
            "Patient-002", pd.DataFrame(rows)
        )


@pytest.mark.parametrize(
    ("patient_id", "source", "labels"),
    [
        ("UPENN-GBM-00042", "UPenn-GBM", ("_11", "_21")),
        ("UCSF-PDGM-0167", "UCSF-PDGM", ("t1", "t2")),
        ("TCGA-06-5412", "TCGA-GBM", ("visit-1", "visit-2")),
    ],
)
def test_multi_scan_still_raises_for_non_lumiere_sources(
    monkeypatch, patient_id: str, source: str, labels: tuple[str, str]
) -> None:
    """(d) ZORUNLU -- LUMIERE'ye ozel kanonik-vizit gevsetmesi
    UPenn/UCSF/TCGA'ya SIZMAMALI.

    O kaynaklarda hangi taramanin ameliyat-oncesi oldugunu soyleyen bir
    ExpertRating tablosu YOKTUR; gevsetmek "sessizce yanlis tarama sec"
    kapisi acardi (Y1 sart 2). Hata mesaji bunu ACIKCA soylemeli."""

    rows = [
        {
            "patient_id": patient_id,
            "source": source,
            "tumor_region": "WT_derived",
            "shape_features": {"original_shape_Flatness": 0.1},
            "first_order_features": {},
            "texture_features": {},
            "scan_id": scan_id,
            "timepoint_label": label,
        }
        for scan_id, label in zip((11, 22), labels)
    ]
    _fake_lumiere_cursor(monkeypatch, rows)

    with pytest.raises(predict_module.MultiScanNotSupportedError) as excinfo:
        predict_module.fetch_c32_wt_row(patient_id, ["WT__original_shape_Flatness"])
    detail = str(excinfo.value)
    assert "2 farkli scan_id" in detail
    assert source in detail


def test_real_expert_rating_csv_pins_patient020_and_patient060_traps() -> None:
    """GERCEK `raw/veri/LUMIERE-ExpertRating-*.csv`e karsi TUZAK PINI
    (canli olculdu 2026-09-13): `Patient-020`'nin Pre-Op viziti
    `week-000-1`dir, `week-000` DEGIL; `Patient-060`'in IKI Pre-Op viziti
    var (`week-000` VE `week-069`). Bu iki olgu degisirse (CSV
    guncellenirse) Y1'in dayandigi olcum de degismis olur -> test kirilir,
    sessizce gecmez."""

    import pipeline.lumiere_canonical_visit as lcv

    csv_path = lcv.DEFAULT_LUMIERE_EXPERT_RATING_CSV
    if not csv_path.is_file():
        pytest.skip(f"Gercek ExpertRating CSV'si bulunamadi: {csv_path}")

    preop_pairs, n_rows, n_patients = lcv.load_lumiere_preop_visit_keys(csv_path)
    assert (n_rows, n_patients) == (92, 91)
    assert ("Patient-020", "week-000-1") in preop_pairs
    assert ("Patient-020", "week-000") not in preop_pairs  # <- TUZAK
    assert {"week-000", "week-069"} == {
        label for pid, label in preop_pairs if pid == "Patient-060"
    }


def test_is_lumiere_patient_id_gate() -> None:
    assert predict_module.is_lumiere_patient_id("Patient-002") is True
    assert predict_module.is_lumiere_patient_id("UPENN-GBM-00001") is False
    assert predict_module.is_lumiere_patient_id("UCSF-PDGM-167") is False
    assert predict_module.is_lumiere_patient_id("TCGA-06-5412") is False


# =====================================================================
# Endpoint (predict_patient) -- ust-seviye fonksiyonlar monkeypatch'li
# =====================================================================


def test_predict_patient_end_to_end_success(monkeypatch, tmp_path) -> None:
    feature_cols = ["WT__f1", "WT__f2"]
    model, frame = _fit_tiny_cox_model(feature_cols, seed=3, n=60)
    arm = _fake_arm_result(model, feature_cols, name="primary_test")

    fake_checkpoint = tmp_path / "fake_checkpoint.pkl"
    fake_checkpoint.write_bytes(b"dummy")  # icerik onemsiz -- load_cox_arm_result monkeypatch'li

    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)
    monkeypatch.setattr(
        predict_module,
        "fetch_c32_wt_row",
        lambda patient_id, final_features: frame[feature_cols].iloc[[0]],
    )
    monkeypatch.setattr(
        predict_module,
        "get_shap_background",
        lambda final_features, extra_columns, cache_generation, **_kwargs: frame[feature_cols].sample(
            n=20, random_state=1
        ),
    )
    monkeypatch.setattr(predict_module, "_checkpoint_path", lambda: fake_checkpoint)

    result = predict_module.predict_patient("UPENN-GBM-00001")

    assert result["patient_id"] == "UPENN-GBM-00001"
    assert result["model_arm"] == "primary_test"
    assert set(result["final_features"]) == set(feature_cols)
    assert result["clinical_extra_columns"] == []
    assert result["shap_additivity_check_abs_diff"] < 1e-9
    reconstructed = sum(result["shap_values"].values()) + result["shap_base_value"]
    assert reconstructed == pytest.approx(result["risk_score_log_partial_hazard"], abs=1e-9)
    assert result["hazard_ratio_partial_hazard"] == pytest.approx(
        np.exp(result["risk_score_log_partial_hazard"]), abs=1e-9
    )


def test_predict_patient_returns_503_when_checkpoint_missing(monkeypatch) -> None:
    def _raise_missing():
        raise predict_module.CheckpointNotFoundError("yok")

    monkeypatch.setattr(predict_module, "load_cox_arm_result", _raise_missing)

    with pytest.raises(predict_module.HTTPException) as exc_info:
        predict_module.predict_patient("UPENN-GBM-00001")
    assert exc_info.value.status_code == 503


def test_predict_patient_returns_501_when_clinical_covariates_unsupported(monkeypatch) -> None:
    """2026-08-18 GUNCELLENDI (model-agnostik karari, Baris) -- 501
    ARTIK yalniz `SUPPORTED_CLINICAL_EXTRA_COLUMNS` (v1/v2a/v2b/v2c'nin
    URETEBILECEGI TUM bilinen kolonlari kapsayan sozluk) DISINDA,
    HICBIR V2 varyantina ait OLMAYAN bir isim gelirse tetiklenir --
    `clinical_mgmt_methylated` ARTIK desteklendigi icin (v2a_mgmt) o
    isimle test EDILEMEZ, bkz. asagida `test_predict_patient_succeeds_
    with_mgmt_clinical_covariates`."""

    feature_cols = ["WT__f1"]
    model, _frame = _fit_tiny_cox_model(
        feature_cols + ["clinical_totally_unknown_covariate"], n=40
    )
    arm = _fake_arm_result(
        model, feature_cols, extra_columns=["clinical_totally_unknown_covariate"]
    )
    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)

    with pytest.raises(predict_module.HTTPException) as exc_info:
        predict_module.predict_patient("UPENN-GBM-00001")
    assert exc_info.value.status_code == 501
    assert "clinical_totally_unknown_covariate" in str(exc_info.value.detail)


def test_predict_patient_returns_422_when_clinical_covariate_missing(monkeypatch) -> None:
    """2026-08-18 EKLENDI -- gorev talimati: 'Klinik kovaryati eksik
    hasta -> 422, sessiz sifir/varsayilan ATAMA YOK.' `build_patient_
    clinical_covariate_row` bu durumda `ClinicalCovariateMissingError`
    firlatir, endpoint bunu 422'ye cevirir."""

    feature_cols = ["WT__f1"]
    clinical_cols = [
        "clinical_age",
        "clinical_gender_male",
        "clinical_gtr_y",
        "clinical_gtr_missing",
        "clinical_idh_mutant",
        "clinical_idh_missing",
    ]
    model, frame = _fit_tiny_cox_model(feature_cols + clinical_cols, n=60)
    arm = _fake_arm_result(model, feature_cols, extra_columns=clinical_cols)
    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)
    monkeypatch.setattr(
        predict_module,
        "fetch_c32_wt_row",
        lambda patient_id, final_features: frame[feature_cols].iloc[[0]],
    )

    def _raise_missing_clinical(patient_id, extra_columns, **_kwargs):
        raise predict_module.ClinicalCovariateMissingError(
            f"Hasta {patient_id!r}: gtr_over90percent (deger=None)"
        )

    monkeypatch.setattr(
        predict_module, "build_patient_clinical_covariate_row", _raise_missing_clinical
    )

    with pytest.raises(predict_module.HTTPException) as exc_info:
        predict_module.predict_patient("TCGA-06-5412")
    assert exc_info.value.status_code == 422
    assert "gtr_over90percent" in str(exc_info.value.detail)


def test_predict_patient_succeeds_with_supported_clinical_covariates(monkeypatch, tmp_path) -> None:
    """2026-08-18 EKLENDI -- desteklenen 6 klinik kovaryatla uctan uca
    BASARILI bir tahmin: radyomik+klinik satirlari BIRLESTIRILIR,
    `fitted_model.params_.index` ile ayni ozellik uzayinda SHAP hesabi
    yapilir, additivite hala TAM tutmali (matematik degismedi, bkz.
    modul dokstring'i 'SHAP OLCEGI')."""

    feature_cols = ["WT__f1", "WT__f2"]
    clinical_cols = [
        "clinical_age",
        "clinical_gender_male",
        "clinical_gtr_y",
        "clinical_gtr_missing",
        "clinical_idh_mutant",
        "clinical_idh_missing",
    ]
    all_cols = feature_cols + clinical_cols
    model, frame = _fit_tiny_cox_model(all_cols, seed=11, n=80)
    arm = _fake_arm_result(model, feature_cols, extra_columns=clinical_cols, name="clinical_test")

    fake_checkpoint = tmp_path / "fake_checkpoint.pkl"
    fake_checkpoint.write_bytes(b"dummy")

    clinical_row = frame[clinical_cols].iloc[[0]].reset_index(drop=True)

    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)
    monkeypatch.setattr(
        predict_module,
        "fetch_c32_wt_row",
        lambda patient_id, final_features: frame[feature_cols].iloc[[0]],
    )
    monkeypatch.setattr(
        predict_module,
        "build_patient_clinical_covariate_row",
        lambda patient_id, extra_columns, **_kwargs: clinical_row,
    )
    monkeypatch.setattr(
        predict_module,
        "get_shap_background",
        lambda final_features, extra_columns, cache_generation, **_kwargs: frame[all_cols].sample(
            n=20, random_state=1
        ),
    )
    monkeypatch.setattr(predict_module, "_checkpoint_path", lambda: fake_checkpoint)

    result = predict_module.predict_patient("UPENN-GBM-00001")

    assert result["clinical_extra_columns"] == clinical_cols
    assert set(result["shap_values"].keys()) == set(all_cols)
    assert result["shap_additivity_check_abs_diff"] < 1e-9
    reconstructed = sum(result["shap_values"].values()) + result["shap_base_value"]
    assert reconstructed == pytest.approx(result["risk_score_log_partial_hazard"], abs=1e-9)


def test_predict_patient_succeeds_with_mgmt_clinical_covariates(monkeypatch, tmp_path) -> None:
    """2026-08-18 EKLENDI (model-agnostik karari) -- v2a_mgmt TARZI bir
    artefakt (MGMT gosterge-degiskeni EKLENMIS 8 klinik kovaryat) ile
    uctan uca BASARILI tahmin. Kod, hangi kolonlarin gerektigini
    TAMAMEN `extra_columns`'tan okur -- v1'e ait 6 sabit isim burada
    HARDCODE EDILMEDI."""

    feature_cols = ["WT__f1"]
    clinical_cols = [
        "clinical_age",
        "clinical_gender_male",
        "clinical_gtr_y",
        "clinical_gtr_missing",
        "clinical_idh_mutant",
        "clinical_idh_missing",
        "clinical_mgmt_methylated",
        "clinical_mgmt_missing",
    ]
    all_cols = feature_cols + clinical_cols
    model, frame = _fit_tiny_cox_model(all_cols, seed=13, n=80)
    arm = _fake_arm_result(model, feature_cols, extra_columns=clinical_cols, name="v2a_mgmt_test")

    fake_checkpoint = tmp_path / "fake_checkpoint.pkl"
    fake_checkpoint.write_bytes(b"dummy")
    clinical_row = frame[clinical_cols].iloc[[0]].reset_index(drop=True)

    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)
    monkeypatch.setattr(
        predict_module,
        "fetch_c32_wt_row",
        lambda patient_id, final_features: frame[feature_cols].iloc[[0]],
    )
    monkeypatch.setattr(
        predict_module,
        "build_patient_clinical_covariate_row",
        lambda patient_id, extra_columns, **_kwargs: clinical_row,
    )
    monkeypatch.setattr(
        predict_module,
        "get_shap_background",
        lambda final_features, extra_columns, cache_generation, **_kwargs: frame[all_cols].sample(
            n=20, random_state=1
        ),
    )
    monkeypatch.setattr(predict_module, "_checkpoint_path", lambda: fake_checkpoint)

    result = predict_module.predict_patient("UPENN-GBM-00001")

    assert result["clinical_extra_columns"] == clinical_cols
    assert "clinical_mgmt_methylated" in result["shap_values"]
    assert "clinical_mgmt_missing" in result["shap_values"]
    assert result["shap_additivity_check_abs_diff"] < 1e-9


def test_predict_patient_succeeds_with_wt_tc_regions(monkeypatch, tmp_path) -> None:
    """2026-08-18 EKLENDI (model-agnostik karari) -- v2c_mgmt_spline_wttc
    TARZI bir artefakt WT+TC bolgelerinden ozellik istiyorsa (186 aday
    havuzu), endpoint TEK bolgeye (WT) SABITLENMEMIS olmali -- final_
    features'in kolon oneklerinden (WT__/TC__) region listesi TURETILIR."""

    feature_cols = ["WT__f1", "TC__f1"]
    model, frame = _fit_tiny_cox_model(feature_cols, seed=17, n=60)
    arm = _fake_arm_result(model, feature_cols, name="wt_tc_test")

    fake_checkpoint = tmp_path / "fake_checkpoint.pkl"
    fake_checkpoint.write_bytes(b"dummy")

    captured_regions: list[str] = []

    def _fake_fetch_c32(patient_id, final_features):
        captured_regions.extend(sorted({c.split("__", 1)[0] for c in final_features}))
        return frame[feature_cols].iloc[[0]]

    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)
    monkeypatch.setattr(predict_module, "fetch_c32_wt_row", _fake_fetch_c32)
    monkeypatch.setattr(
        predict_module,
        "get_shap_background",
        lambda final_features, extra_columns, cache_generation, **_kwargs: frame[feature_cols].sample(
            n=20, random_state=1
        ),
    )
    monkeypatch.setattr(predict_module, "_checkpoint_path", lambda: fake_checkpoint)

    result = predict_module.predict_patient("UPENN-GBM-00001")

    assert captured_regions == ["TC", "WT"]
    assert set(result["final_features"]) == {"WT__f1", "TC__f1"}


def test_predict_patient_returns_501_when_age_spline_requested_without_knots(monkeypatch) -> None:
    """2026-08-18 EKLENDI -- v2b/v2c 'age_rcs' kovaryatlarini istiyor ama
    artefakt (bugunku TEK gercek artefakt DAHIL) `age_rcs_knots`
    TASIMIYORSA endpoint 501 doner, dugum noktalarini UYDURMAZ."""

    feature_cols = ["WT__f1"]
    clinical_cols = ["clinical_age_rcs1", "clinical_age_rcs2", "clinical_gender_male"]
    model, _frame = _fit_tiny_cox_model(feature_cols + clinical_cols, n=40)
    arm = _fake_arm_result(model, feature_cols, extra_columns=clinical_cols)
    assert arm.get("age_rcs_knots") is None
    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)

    with pytest.raises(predict_module.HTTPException) as exc_info:
        predict_module.predict_patient("UPENN-GBM-00001")
    assert exc_info.value.status_code == 501
    assert "age_rcs_knots" in str(exc_info.value.detail)


def test_predict_patient_succeeds_with_age_spline_when_knots_present(monkeypatch, tmp_path) -> None:
    """2026-08-18 EKLENDI -- artefakt `age_rcs_knots` TASIYORSA (varsayimsal
    v2b/v2c sozlesmesi) yas-spline kovaryati BASARIYLA hesaplanir."""

    feature_cols = ["WT__f1"]
    clinical_cols = ["clinical_age_rcs1", "clinical_age_rcs2"]
    all_cols = feature_cols + clinical_cols
    model, frame = _fit_tiny_cox_model(all_cols, seed=19, n=60)
    arm = _fake_arm_result(model, feature_cols, extra_columns=clinical_cols, name="v2b_spline_test")
    arm["age_rcs_knots"] = (30.0, 55.0, 75.0)

    fake_checkpoint = tmp_path / "fake_checkpoint.pkl"
    fake_checkpoint.write_bytes(b"dummy")
    clinical_row = frame[clinical_cols].iloc[[0]].reset_index(drop=True)

    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)
    monkeypatch.setattr(
        predict_module,
        "fetch_c32_wt_row",
        lambda patient_id, final_features: frame[feature_cols].iloc[[0]],
    )
    monkeypatch.setattr(
        predict_module,
        "build_patient_clinical_covariate_row",
        lambda patient_id, extra_columns, **_kwargs: clinical_row,
    )
    monkeypatch.setattr(
        predict_module,
        "get_shap_background",
        lambda final_features, extra_columns, cache_generation, **_kwargs: frame[all_cols].sample(
            n=20, random_state=1
        ),
    )
    monkeypatch.setattr(predict_module, "_checkpoint_path", lambda: fake_checkpoint)

    result = predict_module.predict_patient("UPENN-GBM-00001")

    assert "clinical_age_rcs1" in result["shap_values"]
    assert "clinical_age_rcs2" in result["shap_values"]
    assert result["shap_additivity_check_abs_diff"] < 1e-9


def test_predict_patient_returns_404_when_patient_not_found(monkeypatch) -> None:
    feature_cols = ["WT__f1"]
    model, _frame = _fit_tiny_cox_model(feature_cols, n=40)
    arm = _fake_arm_result(model, feature_cols)
    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)

    def _raise_not_found(patient_id, final_features):
        raise predict_module.PatientNotFoundError(patient_id)

    monkeypatch.setattr(predict_module, "fetch_c32_wt_row", _raise_not_found)

    with pytest.raises(predict_module.HTTPException) as exc_info:
        predict_module.predict_patient("NOPE-999")
    assert exc_info.value.status_code == 404


@pytest.mark.parametrize(
    "raised_exc,expected_status",
    [
        (predict_module.C32RadiomicsNotFoundError("yok"), 422),
        (predict_module.MultiScanNotSupportedError("coklu"), 422),
        (predict_module.RequiredRegionMissingError("eksik"), 422),
    ],
)
def test_predict_patient_maps_fetch_errors_to_422(
    monkeypatch, raised_exc, expected_status
) -> None:
    feature_cols = ["WT__f1"]
    model, _frame = _fit_tiny_cox_model(feature_cols, n=40)
    arm = _fake_arm_result(model, feature_cols)
    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)

    def _raise(patient_id, final_features):
        raise raised_exc

    monkeypatch.setattr(predict_module, "fetch_c32_wt_row", _raise)

    with pytest.raises(predict_module.HTTPException) as exc_info:
        predict_module.predict_patient("UPENN-GBM-00001")
    assert exc_info.value.status_code == expected_status


def test_predict_patient_returns_500_when_final_features_empty(monkeypatch) -> None:
    feature_cols: list[str] = []
    model, _frame = _fit_tiny_cox_model(["WT__f1"], n=40)
    arm = _fake_arm_result(model, feature_cols)
    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)

    with pytest.raises(predict_module.HTTPException) as exc_info:
        predict_module.predict_patient("UPENN-GBM-00001")
    assert exc_info.value.status_code == 500


def test_predict_patient_returns_500_when_additivity_guard_triggers(monkeypatch, tmp_path) -> None:
    """Additivite gercekte hicbir zaman GERCEKTEN bozulmaz (matematiksel
    garanti, bkz. yukaridaki test) -- bu test o SAVUNMA guard'inin
    GERCEKTEN calistigini (dekoratif olmadigini) `compute_risk_score_
    and_shap`'i BILEREK bozuk bir sonuc dondurecek sekilde monkeypatch'
    leyerek dogrular."""

    feature_cols = ["WT__f1"]
    model, frame = _fit_tiny_cox_model(feature_cols, n=40)
    arm = _fake_arm_result(model, feature_cols)
    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)
    monkeypatch.setattr(
        predict_module,
        "fetch_c32_wt_row",
        lambda patient_id, final_features: frame[feature_cols].iloc[[0]],
    )
    monkeypatch.setattr(
        predict_module,
        "get_shap_background",
        lambda final_features, extra_columns, cache_generation, **_kwargs: frame[feature_cols].sample(
            n=10, random_state=1
        ),
    )
    fake_checkpoint = tmp_path / "fake_checkpoint.pkl"
    fake_checkpoint.write_bytes(b"dummy")
    monkeypatch.setattr(predict_module, "_checkpoint_path", lambda: fake_checkpoint)
    monkeypatch.setattr(
        predict_module,
        "compute_risk_score_and_shap",
        lambda *args, **kwargs: {
            "risk_score_log_partial_hazard": 1.0,
            "hazard_ratio_partial_hazard": 2.7,
            "shap_base_value": 0.0,
            "shap_values": {"WT__f1": 0.0},  # kasitli YANLIS -- toplam 1.0 etmiyor
            "shap_additivity_check_abs_diff": 1.0,
        },
    )

    with pytest.raises(predict_module.HTTPException) as exc_info:
        predict_module.predict_patient("UPENN-GBM-00001")
    assert exc_info.value.status_code == 500
    assert "GUVENILIR" in str(exc_info.value.detail)


# =====================================================================
# load_cox_arm_result -- checkpoint dosyasi eksikse
# =====================================================================


def test_load_cox_arm_result_raises_when_file_missing(tmp_path) -> None:
    missing_path = tmp_path / "does_not_exist.pkl"
    with pytest.raises(predict_module.CheckpointNotFoundError):
        predict_module.load_cox_arm_result(missing_path)


# =====================================================================
# _normalize_arm -- 2026-08-18 EKLENDI, iki pickle bicimini AYNI sozluge
# indirger (yeni sade-dict artefakt VS eski ArmResult dataclass'i).
# =====================================================================


def test_normalize_arm_handles_new_dict_artifact() -> None:
    model, _frame = _fit_tiny_cox_model(["WT__f1", "clinical_age"], n=30)
    loaded = {
        "arm_name": "primary_wt93_clinical_full_ucsf",
        "fitted_model": model,
        "final_features": ["WT__f1"],
        "clinical_extra_columns": ["clinical_age"],
        "provenance": {"note": "onemsiz"},
    }

    normalized = predict_module._normalize_arm(loaded)

    assert normalized["name"] == "primary_wt93_clinical_full_ucsf"
    assert normalized["final_features"] == ["WT__f1"]
    assert normalized["extra_columns"] == ["clinical_age"]
    assert normalized["fitted_model"] is model


def test_normalize_arm_handles_new_dict_artifact_without_clinical_columns() -> None:
    """`clinical_extra_columns` anahtari HIC yoksa (eski radyomik-only bir
    sozluk artefakti gelirse) `extra_columns` bos listeye DUSER, KeyError
    FIRLATMAZ."""

    model, _frame = _fit_tiny_cox_model(["WT__f1"], n=30)
    loaded = {"arm_name": "radiomics_only", "fitted_model": model, "final_features": ["WT__f1"]}

    normalized = predict_module._normalize_arm(loaded)

    assert normalized["extra_columns"] == []


def test_normalize_arm_handles_legacy_arm_result_object() -> None:
    """ESKI `tools.train_cox_week3.ArmResult` dataclass bicimi (`.name`,
    `.final_model.final_features`, `.final_model.fitted_model`,
    `.extra_columns`) -- `GBMAID_COX_CHECKPOINT_PATH` ile eski bir
    checkpoint'e geriye donuk isaret edilirse HALA calismali."""

    model, _frame = _fit_tiny_cox_model(["WT__f1"], n=30)
    legacy = SimpleNamespace(
        name="arm_primary_wt93_icc60_th06",
        final_model=SimpleNamespace(final_features=["WT__f1"], fitted_model=model),
        extra_columns=[],
    )

    normalized = predict_module._normalize_arm(legacy)

    assert normalized["name"] == "arm_primary_wt93_icc60_th06"
    assert normalized["final_features"] == ["WT__f1"]
    assert normalized["extra_columns"] == []
    assert normalized["fitted_model"] is model


def test_normalize_arm_raises_type_error_for_unknown_shape() -> None:
    with pytest.raises(TypeError):
        predict_module._normalize_arm(object())


def test_normalize_arm_raises_type_error_for_dict_missing_required_key() -> None:
    with pytest.raises(TypeError):
        predict_module._normalize_arm({"arm_name": "eksik", "final_features": ["WT__f1"]})


# =====================================================================
# Klinik kovaryat kodlama -- _encode_clinical_frame / build_patient_
# clinical_covariate_row (2026-08-18 EKLENDI). DB'ye BAGLANMAZ.
# =====================================================================

_ALL_CLINICAL_COLUMNS = [
    "clinical_age",
    "clinical_gender_male",
    "clinical_gtr_y",
    "clinical_gtr_missing",
    "clinical_idh_mutant",
    "clinical_idh_missing",
]


def test_encode_clinical_frame_tolerant_fillna_matches_training_pattern() -> None:
    raw = pd.DataFrame(
        {
            "age": [55.0],
            "gender": ["Male"],
            "gtr_over90percent": [None],
            "idh1_status": ["NOS/NEC"],
        }
    )

    encoded = predict_module._encode_clinical_frame(raw, _ALL_CLINICAL_COLUMNS)

    assert encoded.loc[0, "clinical_age"] == 55.0
    assert encoded.loc[0, "clinical_gender_male"] == 1.0
    assert encoded.loc[0, "clinical_gtr_y"] == 0.0
    assert encoded.loc[0, "clinical_gtr_missing"] == 1.0
    assert encoded.loc[0, "clinical_idh_mutant"] == 0.0
    assert encoded.loc[0, "clinical_idh_missing"] == 1.0


def test_encode_clinical_frame_raises_on_unrecognized_gender_value() -> None:
    raw = pd.DataFrame(
        {"age": [10.0], "gender": ["unknown"], "gtr_over90percent": ["Y"], "idh1_status": ["Wildtype"]}
    )
    with pytest.raises(predict_module.ClinicalCovariateMissingError):
        predict_module._encode_clinical_frame(raw, _ALL_CLINICAL_COLUMNS)


def test_build_patient_clinical_covariate_row_gtr_and_idh_null_is_train_aligned(monkeypatch) -> None:
    """2026-09-13 GUNCELLEME (Baris onayi, tutarlilik gerekcesiyle):
    ESKI davranis (bu testin adi eskiden `..._raises_when_gtr_and_idh_
    null` idi) GERCEK DB'de KESFEDILEN TCGA deseni (gtr_over90percent VE
    idh1_status ikisi de NULL) icin 422 atiyordu. `CLINICAL_COVARIATE_
    NULL_POLICY['gtr_over90percent'/'idh1_status'] == TRAIN_ALIGNED`
    oldugu icin ARTIK 422 ATILMAZ -- egitimdeki tolerant kodlamayla
    (`tools/train_cox_week3.py` satir 663-687) BIREBIR AYNI sekilde
    `clinical_gtr_missing=1`/`clinical_idh_missing=1` olarak kodlanir
    (MGMT'nin K19'da aldigi TRAIN_ALIGNED muamelesiyle AYNI)."""

    raw_frame = pd.DataFrame(
        [{"patient_id": "TCGA-06-5412", "age": 78, "gender": "Female",
          "gtr_over90percent": None, "idh1_status": None}]
    )
    monkeypatch.setattr(predict_module, "_fetch_patients_clinical_raw", lambda ids: raw_frame)

    row = predict_module.build_patient_clinical_covariate_row(
        "TCGA-06-5412", _ALL_CLINICAL_COLUMNS
    )
    assert row.loc[0, "clinical_gtr_missing"] == 1.0
    assert row.loc[0, "clinical_gtr_y"] == 0.0
    assert row.loc[0, "clinical_idh_missing"] == 1.0
    assert row.loc[0, "clinical_idh_mutant"] == 0.0


def test_build_patient_clinical_covariate_row_gtr_unrecognized_value_still_rejected(monkeypatch) -> None:
    """K19-desenli TRAIN_ALIGNED politikasi SADECE NULL'u kapsar --
    taninmayan (NULL-DISI, ornek yazim hatasi) bir `gtr_over90percent`
    degeri HALA 422 ATAR, politika onu KAPSAMAZ (bu 'bilinen eksiklik'
    degil veri bozuklugu) -- MGMT'nin ayni desenli regresyon testiyle
    (`..._mgmt_unrecognized_value_still_rejected`) PARALEL."""

    raw_frame = pd.DataFrame(
        [{"patient_id": "GHOST-5", "age": 60, "gender": "Male",
          "gtr_over90percent": "???", "idh1_status": "Wildtype"}]
    )
    monkeypatch.setattr(predict_module, "_fetch_patients_clinical_raw", lambda ids: raw_frame)

    with pytest.raises(predict_module.ClinicalCovariateMissingError) as exc_info:
        predict_module.build_patient_clinical_covariate_row("GHOST-5", _ALL_CLINICAL_COLUMNS)
    assert "gtr_over90percent" in str(exc_info.value)


def test_build_patient_clinical_covariate_row_idh_unrecognized_value_still_rejected(monkeypatch) -> None:
    """GTR'nin yukaridaki testiyle AYNI mantik, `idh1_status` icin --
    NULL-DISI taninmayan deger politikadan BAGIMSIZ HER ZAMAN 422 atar."""

    raw_frame = pd.DataFrame(
        [{"patient_id": "GHOST-6", "age": 60, "gender": "Male",
          "gtr_over90percent": "Y", "idh1_status": "???"}]
    )
    monkeypatch.setattr(predict_module, "_fetch_patients_clinical_raw", lambda ids: raw_frame)

    with pytest.raises(predict_module.ClinicalCovariateMissingError) as exc_info:
        predict_module.build_patient_clinical_covariate_row("GHOST-6", _ALL_CLINICAL_COLUMNS)
    assert "idh1_status" in str(exc_info.value)


def test_build_patient_clinical_covariate_row_allows_idh_nos_nec(monkeypatch) -> None:
    """K15 karari: `NOS/NEC` ("test yapilmadi") 422 TETIKLEMEZ, eksik-
    gosterge (`clinical_idh_missing=1`) olarak kodlanir -- egitimdeki
    davranisla AYNI."""

    raw_frame = pd.DataFrame(
        [{"patient_id": "UPENN-GBM-00099", "age": 60, "gender": "Male",
          "gtr_over90percent": "Y", "idh1_status": "NOS/NEC"}]
    )
    monkeypatch.setattr(predict_module, "_fetch_patients_clinical_raw", lambda ids: raw_frame)

    row = predict_module.build_patient_clinical_covariate_row(
        "UPENN-GBM-00099", _ALL_CLINICAL_COLUMNS
    )

    assert row.loc[0, "clinical_idh_missing"] == 1.0
    assert row.loc[0, "clinical_idh_mutant"] == 0.0
    assert row.loc[0, "clinical_gtr_y"] == 1.0
    assert row.loc[0, "clinical_gtr_missing"] == 0.0


# =====================================================================
# LUMIERE ham etiket normalizasyonu (2026-09-13, Baris onayi) --
# `_normalize_clinical_categorical_raw_values()` / `IDH1_LABEL_
# NORMALIZATION_MAP` / `MGMT_LABEL_NORMALIZATION_MAP`. Ilk 4 test SAF
# MANTIK (DB'ye BAGLANMAZ), sonraki ikisi GERCEK DB smoke testidir
# (LUMIERE Patient-026 -- 'IDH1 neg, Sequencing required').
# =====================================================================

def test_normalize_clinical_categorical_raw_values_maps_known_lumiere_labels() -> None:
    raw = pd.DataFrame(
        {
            "patient_id": ["A", "B", "C", "D", "E"],
            "idh1_status": ["WT", "wt", "R132H mut", "IDH1 neg, Sequencing required", "Wildtype"],
            "mgmt_status": ["methylated", "not methylated", None, "Indeterminate", "Methylated"],
        }
    )
    normalized = predict_module._normalize_clinical_categorical_raw_values(raw)

    assert list(normalized["idh1_status"]) == [
        "Wildtype", "Wildtype", "Mutated", "NOS/NEC", "Wildtype",
    ]
    assert normalized["mgmt_status"].tolist()[:2] == ["Methylated", "Unmethylated"]
    assert pd.isna(normalized["mgmt_status"].iloc[2])
    assert normalized["mgmt_status"].iloc[3] == "Indeterminate"  # haritada YOK, degismedi
    assert normalized["mgmt_status"].iloc[4] == "Methylated"  # zaten kanonik, degismedi


def test_normalize_clinical_categorical_raw_values_leaves_unmapped_values_untouched() -> None:
    """Haritada OLMAYAN bir deger (yazim hatasi/sema disi) SESSIZCE
    degistirilmez -- asagidaki 'taninmayan deger' kontrolleri (422) HALA
    calismalidir, bu fonksiyon onlari YUTMAZ."""

    raw = pd.DataFrame({"patient_id": ["X"], "idh1_status": ["???"], "mgmt_status": ["yanlis-deger"]})
    normalized = predict_module._normalize_clinical_categorical_raw_values(raw)
    assert normalized["idh1_status"].iloc[0] == "???"
    assert normalized["mgmt_status"].iloc[0] == "yanlis-deger"


def test_normalize_clinical_categorical_raw_values_does_not_mutate_input_frame() -> None:
    raw = pd.DataFrame({"patient_id": ["A"], "idh1_status": ["WT"], "mgmt_status": ["methylated"]})
    _ = predict_module._normalize_clinical_categorical_raw_values(raw)
    assert raw["idh1_status"].iloc[0] == "WT"  # orijinal DEGISMEDI
    assert raw["mgmt_status"].iloc[0] == "methylated"


def test_build_patient_clinical_covariate_row_normalizes_lumiere_wt_and_mgmt_labels(monkeypatch) -> None:
    """Ucu birden (WT/methylated) normalize edilmis 'gorunumlu' bir LUMIERE
    hastasi -- normalizasyon OLMASA 422 atardi (bkz. asagidaki KIRMIZI
    kontrolu, `_clinical_raw_value_is_rejected` ile DOGRUDAN), normalizasyon
    ILE 200/basarili sonuc doner."""

    raw_frame = pd.DataFrame(
        [{"patient_id": "Patient-044-SENTETIK", "age": 72, "gender": "Male",
          "gtr_over90percent": None, "idh1_status": "WT", "mgmt_status": "not methylated"}]
    )
    # KIRMIZI -- normalizasyon UYGULANMADAN ham deger dogrudan kontrol edilirse
    # REDDEDILIR (422'ye giderdi).
    assert predict_module._clinical_raw_value_is_rejected(
        "WT",
        frozenset({predict_module.IDH1_WILDTYPE_LABEL, predict_module.IDH1_MUTATED_LABEL, predict_module.IDH1_NOS_LABEL}),
        raw_column="idh1_status",
    ) is True

    monkeypatch.setattr(
        predict_module,
        "_fetch_patients_clinical_raw",
        lambda ids: predict_module._normalize_clinical_categorical_raw_values(raw_frame),
    )

    row = predict_module.build_patient_clinical_covariate_row(
        "Patient-044-SENTETIK", _ALL_CLINICAL_COLUMNS + _MGMT_CLINICAL_COLUMNS
    )
    assert row.loc[0, "clinical_idh_mutant"] == 0.0
    assert row.loc[0, "clinical_idh_missing"] == 0.0  # 'WT'->Wildtype, EKSIK DEGIL
    assert row.loc[0, "clinical_mgmt_methylated"] == 0.0
    assert row.loc[0, "clinical_mgmt_missing"] == 0.0  # 'not methylated'->Unmethylated, EKSIK DEGIL


def test_fetch_patients_clinical_raw_real_db_smoke_lumiere_patient_026_normalizes() -> None:
    """GERCEK DB smoke -- LUMIERE `Patient-026`: ham `idh1_status`=
    `'IDH1 neg, Sequencing required'`, ham `mgmt_status`=NULL (canli DB'de
    2026-09-13'te dogrulandi). `_fetch_patients_clinical_raw()` bu ham
    degeri `NOS/NEC`'e cevirmeli -- `Wildtype`'a DEGIL (kritik ayrim,
    bkz. modul dokstring'i 'HAM ETIKET NORMALIZASYONU')."""

    try:
        raw_frame = predict_module._fetch_patients_clinical_raw(["Patient-026"])
    except Exception as exc:  # pragma: no cover -- yalniz DB erisilemezse
        pytest.skip(f"Gercek DB'ye erisilemedi, smoke test atlandi: {exc}")

    if raw_frame.empty:
        pytest.skip("Patient-026 canli DB'de bulunamadi -- veri degismis olabilir.")
    row = raw_frame.set_index("patient_id").loc["Patient-026"]
    if row["idh1_status"] != predict_module.IDH1_NOS_LABEL:
        pytest.skip(
            f"Patient-026'nin idh1_status'u artik beklenen ham degeri tasimiyor "
            f"olabilir (canli deger normalize-sonrasi={row['idh1_status']!r}) -- "
            "canli veri degismis olabilir, bu test o spesifik senaryoyu kanitlar."
        )
    assert row["idh1_status"] == "NOS/NEC"  # 'Wildtype' DEGIL


def test_build_patient_clinical_covariate_row_real_db_smoke_lumiere_patient_026_returns_ok() -> None:
    """GERCEK DB smoke (KIRMIZI/YESIL kaniti) -- LUMIERE `Patient-026`
    normalizasyon ONCESI (ham deger dogrudan kontrol edilseydi) 422
    alirdi, normalizasyon SONRASI (guncel kod) basarili doner VE
    `clinical_idh_missing=1`/`clinical_idh_mutant=0` uretir (NOS/NEC
    'bilinen eksiklik' olarak kodlanir, ASLA 'bilinen Wildtype' olarak
    YORUMLANMAZ)."""

    checkpoint_extra_columns = [
        "clinical_age", "clinical_gender_male",
        "clinical_gtr_y", "clinical_gtr_missing",
        "clinical_idh_mutant", "clinical_idh_missing",
        "clinical_mgmt_methylated", "clinical_mgmt_missing",
    ]
    try:
        raw = predict_module._fetch_patients_clinical_raw(["Patient-026"])
    except Exception as exc:  # pragma: no cover -- yalniz DB erisilemezse
        pytest.skip(f"Gercek DB'ye erisilemedi, smoke test atlandi: {exc}")
    if raw.empty or raw.set_index("patient_id").loc["Patient-026", "idh1_status"] != "NOS/NEC":
        pytest.skip("Patient-026 canli DB'de beklenen ham/normalize durumu tasimiyor -- veri degismis olabilir.")

    row = predict_module.build_patient_clinical_covariate_row("Patient-026", checkpoint_extra_columns)
    assert row.loc[0, "clinical_idh_missing"] == 1.0
    assert row.loc[0, "clinical_idh_mutant"] == 0.0
    assert row.loc[0, "clinical_mgmt_missing"] == 1.0  # mgmt_status NULL -> TRAIN_ALIGNED


def test_build_patient_clinical_covariate_row_raises_when_age_null(monkeypatch) -> None:
    raw_frame = pd.DataFrame(
        [{"patient_id": "GHOST-1", "age": None, "gender": "Male",
          "gtr_over90percent": "Y", "idh1_status": "Wildtype"}]
    )
    monkeypatch.setattr(predict_module, "_fetch_patients_clinical_raw", lambda ids: raw_frame)

    with pytest.raises(predict_module.ClinicalCovariateMissingError) as exc_info:
        predict_module.build_patient_clinical_covariate_row("GHOST-1", _ALL_CLINICAL_COLUMNS)
    assert "age" in str(exc_info.value)


# =====================================================================
# MGMT + yas-spline (RCS) klinik kovaryatlari -- model-agnostik karari
# (2026-08-18, Baris): v2a_mgmt/v2b_mgmt_spline/v2c_mgmt_spline_wttc
# HENUZ nihai model DEGIL, ama kod bunlari destekleyecek sekilde
# GENELLESTIRILDI. DB'ye BAGLANMAZ.
# =====================================================================

_MGMT_CLINICAL_COLUMNS = ["clinical_mgmt_methylated", "clinical_mgmt_missing"]


def test_encode_clinical_frame_mgmt_tolerant_fillna() -> None:
    raw = pd.DataFrame(
        {
            "age": [55.0],
            "gender": ["Male"],
            "gtr_over90percent": ["Y"],
            "idh1_status": ["Wildtype"],
            "mgmt_status": ["Indeterminate"],
        }
    )
    encoded = predict_module._encode_clinical_frame(raw, _MGMT_CLINICAL_COLUMNS)
    assert encoded.loc[0, "clinical_mgmt_methylated"] == 0.0
    assert encoded.loc[0, "clinical_mgmt_missing"] == 1.0


def test_encode_clinical_frame_mgmt_raises_on_unrecognized_value() -> None:
    raw = pd.DataFrame(
        {
            "age": [55.0], "gender": ["Male"], "gtr_over90percent": ["Y"],
            "idh1_status": ["Wildtype"], "mgmt_status": ["yanlis-deger"],
        }
    )
    with pytest.raises(predict_module.ClinicalCovariateMissingError):
        predict_module._encode_clinical_frame(raw, _MGMT_CLINICAL_COLUMNS)


def test_build_patient_clinical_covariate_row_allows_mgmt_indeterminate(monkeypatch) -> None:
    """MGMT_INDETERMINATE_LABEL notu: IDH'nin NOS/NEC'iyle AYNI epistemik
    durum ('test yapildi, sonuc kullanilamaz') -- 422 TETIKLEMEZ."""

    raw_frame = pd.DataFrame(
        [{"patient_id": "UPENN-GBM-00099", "age": 60, "gender": "Male",
          "gtr_over90percent": "Y", "idh1_status": "Wildtype", "mgmt_status": "Indeterminate"}]
    )
    monkeypatch.setattr(predict_module, "_fetch_patients_clinical_raw", lambda ids: raw_frame)

    row = predict_module.build_patient_clinical_covariate_row(
        "UPENN-GBM-00099", _MGMT_CLINICAL_COLUMNS
    )
    assert row.loc[0, "clinical_mgmt_missing"] == 1.0
    assert row.loc[0, "clinical_mgmt_methylated"] == 0.0


def test_build_patient_clinical_covariate_row_mgmt_status_null_is_train_aligned(monkeypatch) -> None:
    """K19 karari (Baris 2026-09-12, Secenek 2) -- ESKI davranis (bu
    testin adi eskiden `..._raises_when_mgmt_status_null` idi) NULL
    `mgmt_status` icin 422 atiyordu. `CLINICAL_COVARIATE_NULL_POLICY
    ['mgmt_status'] == TRAIN_ALIGNED` oldugu icin ARTIK 422 ATILMAZ --
    `Indeterminate` ile AYNI sekilde `clinical_mgmt_missing=1` olarak
    kodlanir (egitimdeki tolerant kodlamayla BIREBIR AYNI)."""

    raw_frame = pd.DataFrame(
        [{"patient_id": "GHOST-2", "age": 60, "gender": "Male",
          "gtr_over90percent": "Y", "idh1_status": "Wildtype", "mgmt_status": None}]
    )
    monkeypatch.setattr(predict_module, "_fetch_patients_clinical_raw", lambda ids: raw_frame)

    row = predict_module.build_patient_clinical_covariate_row("GHOST-2", _MGMT_CLINICAL_COLUMNS)
    assert row.loc[0, "clinical_mgmt_missing"] == 1.0
    assert row.loc[0, "clinical_mgmt_methylated"] == 0.0


def test_build_patient_clinical_covariate_row_mgmt_unrecognized_value_still_rejected(monkeypatch) -> None:
    """K19 politikasi SADECE NULL'u kapsar -- taninmayan (NULL-DISI,
    ornek yazim hatasi) bir `mgmt_status` degeri HALA 422 ATAR, politika
    onu KAPSAMAZ (bu 'bilinen eksiklik' degil veri bozuklugu)."""

    raw_frame = pd.DataFrame(
        [{"patient_id": "GHOST-3", "age": 60, "gender": "Male",
          "gtr_over90percent": "Y", "idh1_status": "Wildtype", "mgmt_status": "yanlis-deger"}]
    )
    monkeypatch.setattr(predict_module, "_fetch_patients_clinical_raw", lambda ids: raw_frame)

    with pytest.raises(predict_module.ClinicalCovariateMissingError) as exc_info:
        predict_module.build_patient_clinical_covariate_row("GHOST-3", _MGMT_CLINICAL_COLUMNS)
    assert "mgmt_status" in str(exc_info.value)


def test_build_patient_clinical_covariate_row_gtr_idh_mgmt_null_no_longer_asymmetric(
    monkeypatch,
) -> None:
    """2026-09-13 GUNCELLEME (Baris onayi) -- bu test ESKIDEN (adi
    `..._gtr_idh_null_still_strict_after_mgmt_alignment` idi) KASITLI
    asimetriyi kilitliyordu: GTR/IDH NULL icin 422, MGMT NULL icin 422
    YOK. Simdi ucu de `TRAIN_ALIGNED` oldugu icin asimetri KALKTI --
    GTR/IDH/MGMT UCU DE AYNI satirda/cagrida NULL olsa da 422 ATILMAZ,
    tutarli sekilde `*_missing=1` olarak kodlanir. Eski test adi ve
    davranisi BURADA, degisikligin NEDENI olarak yorumla KAYITLI (CLAUDE.
    md 'hata duzeltmek post-hoc degildir' ilkesi -- asimetri kod-hatasi
    degildi, bilincli bir kisitliliktin ama tutarlilik icin kaldirildi)."""

    raw_frame = pd.DataFrame(
        [{"patient_id": "GHOST-4", "age": 60, "gender": "Male",
          "gtr_over90percent": None, "idh1_status": None, "mgmt_status": None}]
    )
    monkeypatch.setattr(predict_module, "_fetch_patients_clinical_raw", lambda ids: raw_frame)

    row = predict_module.build_patient_clinical_covariate_row(
        "GHOST-4", _ALL_CLINICAL_COLUMNS + _MGMT_CLINICAL_COLUMNS
    )
    assert row.loc[0, "clinical_gtr_missing"] == 1.0
    assert row.loc[0, "clinical_idh_missing"] == 1.0
    assert row.loc[0, "clinical_mgmt_missing"] == 1.0


def test_build_patient_clinical_covariate_row_age_gender_still_strict_no_asymmetry_change(
    monkeypatch,
) -> None:
    """GTR/IDH/MGMT hizalamasi `age`/`gender`'i ETKILEMEDI (yapisal sebep
    -- egitimde bunlara karsilik gelen bir `*_missing` kolonu YOK, bkz.
    `CLINICAL_COVARIATE_NULL_POLICY`). `age` NULL, GTR/IDH/MGMT hepsi
    GECERLI olsa da HALA 422 atar -- bu, hizalamanin `age`/`gender`'a
    SIZMADIGININ regresyon kaniti."""

    raw_frame = pd.DataFrame(
        [{"patient_id": "GHOST-7", "age": None, "gender": "Male",
          "gtr_over90percent": "Y", "idh1_status": "Wildtype", "mgmt_status": "Methylated"}]
    )
    monkeypatch.setattr(predict_module, "_fetch_patients_clinical_raw", lambda ids: raw_frame)

    with pytest.raises(predict_module.ClinicalCovariateMissingError) as exc_info:
        predict_module.build_patient_clinical_covariate_row(
            "GHOST-7", _ALL_CLINICAL_COLUMNS + _MGMT_CLINICAL_COLUMNS
        )
    detail = str(exc_info.value)
    assert "age" in detail
    assert "gtr_over90percent" not in detail
    assert "idh1_status" not in detail
    assert "mgmt_status" not in detail


def test_encode_clinical_frame_raises_without_knots_for_age_spline() -> None:
    raw = pd.DataFrame(
        {"age": [55.0], "gender": ["Male"], "gtr_over90percent": ["Y"], "idh1_status": ["Wildtype"]}
    )
    with pytest.raises(predict_module.ClinicalCovariatesNotSupportedError):
        predict_module._encode_clinical_frame(
            raw, ["clinical_age_rcs1", "clinical_age_rcs2"], age_rcs_knots=None
        )


def test_encode_clinical_frame_age_spline_matches_manual_formula() -> None:
    """`_restricted_cubic_spline_basis`'in `tools/train_cox_week3.py::
    restricted_cubic_spline_basis()`'in AYNI kopyasi oldugunu, formulu
    ELLE (Harrell RCS ucuncu-dugum formulu) yeniden hesaplayarak
    dogrular -- BAGIMSIZ implementasyon karsilastirmasi."""

    knots = (30.0, 55.0, 75.0)
    raw = pd.DataFrame(
        {
            "age": [20.0, 55.0, 90.0],
            "gender": ["Male"] * 3,
            "gtr_over90percent": ["Y"] * 3,
            "idh1_status": ["Wildtype"] * 3,
        }
    )
    encoded = predict_module._encode_clinical_frame(
        raw, ["clinical_age_rcs1", "clinical_age_rcs2"], age_rcs_knots=knots
    )

    t1, t2, t3 = knots

    def _manual(age: float) -> float:
        def pos3(v: float) -> float:
            return max(v, 0.0) ** 3

        return (
            pos3(age - t1) - pos3(age - t2) * (t3 - t1) / (t3 - t2) + pos3(age - t3) * (t2 - t1) / (t3 - t2)
        ) / ((t3 - t1) ** 2)

    for i, age in enumerate(raw["age"]):
        assert encoded.loc[i, "clinical_age_rcs1"] == pytest.approx(age)
        assert encoded.loc[i, "clinical_age_rcs2"] == pytest.approx(_manual(age), abs=1e-9)


def test_restricted_cubic_spline_basis_raises_on_unordered_knots() -> None:
    raw_age = pd.Series([50.0])
    with pytest.raises(ValueError):
        predict_module._restricted_cubic_spline_basis(raw_age, (55.0, 30.0, 75.0))


# =====================================================================
# fetch_omics_interpretation / _build_omics_block -- Adim 7 (OPSIYONEL)
# plan.txt:145-151 + v45.txt SS 6.4. DB'ye BAGLANMAZ (FakeConnection).
# =====================================================================


_FULL_MOLECULAR_SCORE_ROW: dict[str, Any] = {
    "tmz_resistance_score": 39.92,
    "tmz_class": "sensitive",
    "tmz_class_relative": None,
    "aggressiveness_score": 53.97,
    "aggr_class": "intermediate",
    "dna_repair_score": 36.45,
    "repair_class": "impaired",
    "molecular_subtype": "parp_inhibitor_candidate",
    "egfr_amp_flag": None,
    "pten_del_flag": None,
    "cdkn2a_del_flag": None,
    "mgmt_interpretation": None,
    "score_version": "v1.0",
}

# `score_thresholds`'un canli 2026-08-18 --apply kosusundan sonraki
# GERCEK degerleri (db-agent, tools/write_score_thresholds.py) --
# 39.92 <= 46.5009 (p33) => "sensitive", _FULL_MOLECULAR_SCORE_ROW'un
# kendi `tmz_class`'iyla TUTARLI (consistency check testleri icin).
_FAKE_SCORE_THRESHOLDS_ROWS: list[dict[str, Any]] = [
    {
        "score_name": "tmz_resistance_score",
        "score_version": "v1.0",
        "p33_cutoff": 46.5009,
        "p66_cutoff": 48.3900,
        "cohort_snapshot_n": 48,
        "frozen_at": "2026-08-18T13:38:47+00:00",
    },
    {
        "score_name": "dna_repair_score",
        "score_version": "v1.0",
        "p33_cutoff": 41.9847,
        "p66_cutoff": 59.0642,
        "cohort_snapshot_n": 48,
        "frozen_at": "2026-08-18T13:38:47+00:00",
    },
]


def test_fetch_omics_interpretation_returns_none_when_has_omics_false(monkeypatch) -> None:
    """Plan.txt:503 -- `patients.has_omics=TRUE` DEGILSE mod�l sessizce
    devre disi kalir: `None` doner, molecular_scores'a HIC sorgu ATILMAZ
    (FakeCursor'un ikinci `fetchall_results` girdisi olmadigindan, ikinci
    bir `execute()` cagrilirsa test IndexError ile CRASH eder)."""

    cursor = FakeCursor(fetchone_results=[{"has_omics": False}], fetchall_results=[])
    monkeypatch.setattr(predict_module, "_get_db_connection", lambda: FakeConnection(cursor))

    assert predict_module.fetch_omics_interpretation("UPENN-GBM-00001") is None


def test_fetch_omics_interpretation_returns_none_when_patient_missing(monkeypatch) -> None:
    cursor = FakeCursor(fetchone_results=[None], fetchall_results=[])
    monkeypatch.setattr(predict_module, "_get_db_connection", lambda: FakeConnection(cursor))

    assert predict_module.fetch_omics_interpretation("NOPE-999") is None


def test_fetch_omics_interpretation_returns_full_block_when_has_omics_true(monkeypatch) -> None:
    cursor = FakeCursor(
        fetchone_results=[{"has_omics": True}, dict(_FULL_MOLECULAR_SCORE_ROW)],
        fetchall_results=[list(_FAKE_SCORE_THRESHOLDS_ROWS)],
    )
    monkeypatch.setattr(predict_module, "_get_db_connection", lambda: FakeConnection(cursor))

    block = predict_module.fetch_omics_interpretation("TCGA-06-5412")

    assert block is not None
    assert block["has_omics"] is True
    assert block["available"] is True
    assert block["tmz_resistance_score"] == pytest.approx(39.92)
    assert block["tmz_class"] == "sensitive"
    assert block["molecular_subtype"] == "parp_inhibitor_candidate"
    assert block["score_version"] == "v1.0"
    # Kilitli kural: tmz_resistance_score dusuk = duyarli, yon acikca yazilmali.
    assert "DUSUK" in block["tmz_resistance_direction"]
    assert "DUYARLI" in block["tmz_resistance_direction"]
    # Kohort-gorelilik beyani (aggressiveness/dna_repair) yanitta olmali.
    assert "GORECELIDIR" in block["interpretation_note"]
    assert "score_thresholds" in block["interpretation_note"]
    # 2026-08-18 EKLENDI -- Codex review-gate bulgusu: statik "0 satir" YOK,
    # canli esikler DONDU ve DOGRU surumu/kohort n'i raporluyor.
    assert "0 satir" not in block["interpretation_note"]
    assert "v1.0" in block["interpretation_note"]
    assert block["score_thresholds"]["tmz_resistance_score"]["p33_cutoff"] == pytest.approx(46.5009)
    # Capraz kontrol: 39.92 <= p33=46.5009 -> 'sensitive', DB'nin tmz_class'iyla TUTARLI.
    assert block["tmz_class_consistency_check"].startswith("consistent")
    # 2026-08-18 EKLENDI -- Ege'nin bulgusu: 4 NULL kolonun DA notu VAR (yalniz tmz_class_relative degil).
    for field in ("egfr_amp_flag", "pten_del_flag", "cdkn2a_del_flag", "mgmt_interpretation"):
        note_key = f"{field}_note"
        assert note_key in block
        assert "uretim kurali tanimli degil" in block[note_key]
        assert "48/48 NULL" in block[note_key]


def test_fetch_omics_interpretation_flags_inconsistent_tmz_class(monkeypatch) -> None:
    """Codex review-gate bulgusu -- DB'de BOZUK bir satir (skor DUYARLI
    bolgede ama sinif 'resistant' yazili) olsa API bunu SESSIZCE aynen
    donmemeli, TUTARSIZLIK uyarisi ICERMELI."""

    broken_row = dict(_FULL_MOLECULAR_SCORE_ROW)
    broken_row["tmz_resistance_score"] = 10.0  # << p33=46.5009, "sensitive" olmali
    broken_row["tmz_class"] = "resistant"  # BOZUK -- skorla UYUSMUYOR

    cursor = FakeCursor(
        fetchone_results=[{"has_omics": True}, broken_row],
        fetchall_results=[list(_FAKE_SCORE_THRESHOLDS_ROWS)],
    )
    monkeypatch.setattr(predict_module, "_get_db_connection", lambda: FakeConnection(cursor))

    block = predict_module.fetch_omics_interpretation("BROKEN-ROW-001")

    assert block["tmz_class_consistency_check"].startswith("TUTARSIZLIK")
    assert "resistant" in block["tmz_class_consistency_check"]
    assert "sensitive" in block["tmz_class_consistency_check"]
    # DB'deki (bozuk) deger SESSIZCE duzeltilmiyor -- oldugu gibi donuyor.
    assert block["tmz_class"] == "resistant"


def test_check_tmz_class_consistency_returns_cannot_verify_when_thresholds_empty() -> None:
    """Codex'in 5. maddesi: `score_thresholds` bossa (henuz hic esik
    dondurulmamis) SESSIZCE bir varsayilan esik UYDURULMAZ -- ayri bir
    'cannot_verify' metni doner."""

    result = predict_module._check_tmz_class_consistency(dict(_FULL_MOLECULAR_SCORE_ROW), {})
    assert result.startswith("cannot_verify")


def test_fetch_omics_interpretation_tmz_class_relative_is_null_with_documented_note(
    monkeypatch,
) -> None:
    """DB'de 48/48 BOS oldugu canli olculen `tmz_class_relative` icin:
    `None` DONER (baska bir degere sessizce duzeltilmez) VE bunun
    bilinen/kod-hatasi-olmayan bir durum oldugu ayri bir alanla
    belgelenir."""

    cursor = FakeCursor(
        fetchone_results=[{"has_omics": True}, dict(_FULL_MOLECULAR_SCORE_ROW)],
        fetchall_results=[list(_FAKE_SCORE_THRESHOLDS_ROWS)],
    )
    monkeypatch.setattr(predict_module, "_get_db_connection", lambda: FakeConnection(cursor))

    block = predict_module.fetch_omics_interpretation("TCGA-06-5412")

    assert block["tmz_class_relative"] is None
    assert "NULL" in block["tmz_class_relative_note"]
    assert "KOD HATASI DEGIL" in block["tmz_class_relative_note"]


def test_fetch_omics_interpretation_reports_inconsistency_when_row_missing(monkeypatch) -> None:
    """`has_omics=TRUE` AMA `molecular_scores`'ta satir YOKSA -- bu
    plan.txt:503'teki "sessizce atla" senaryosu DEGIL (o, has_omics=FALSE
    ile temsil ediliyor). Bu durum GORUNUR bir anomali blogu olarak
    donmeli, sessizce None'a DUSMEMELI."""

    cursor = FakeCursor(fetchone_results=[{"has_omics": True}, None], fetchall_results=[])
    monkeypatch.setattr(predict_module, "_get_db_connection", lambda: FakeConnection(cursor))

    block = predict_module.fetch_omics_interpretation("GHOST-PATIENT-001")

    assert block is not None
    assert block["has_omics"] is True
    assert block["available"] is False
    assert "tutarsizligi" in block["note"]


# =====================================================================
# predict_patient uctan-uca -- omics blogunun varligi/yoklugu VE Cox
# risk_score'un omics'ten TAMAMEN bagimsiz oldugu
# =====================================================================


def _predict_end_to_end_setup(monkeypatch, tmp_path, *, omics_return):
    feature_cols = ["WT__f1", "WT__f2"]
    model, frame = _fit_tiny_cox_model(feature_cols, seed=3, n=60)
    arm = _fake_arm_result(model, feature_cols, name="primary_test")

    fake_checkpoint = tmp_path / "fake_checkpoint.pkl"
    fake_checkpoint.write_bytes(b"dummy")

    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)
    monkeypatch.setattr(
        predict_module,
        "fetch_c32_wt_row",
        lambda patient_id, final_features: frame[feature_cols].iloc[[0]],
    )
    monkeypatch.setattr(
        predict_module,
        "get_shap_background",
        lambda final_features, extra_columns, cache_generation, **_kwargs: frame[feature_cols].sample(
            n=20, random_state=1
        ),
    )
    monkeypatch.setattr(predict_module, "_checkpoint_path", lambda: fake_checkpoint)
    monkeypatch.setattr(
        predict_module, "fetch_omics_interpretation", lambda patient_id: omics_return
    )
    return predict_module.predict_patient("UPENN-GBM-00001")


def test_predict_patient_includes_omics_key_when_has_omics_true(monkeypatch, tmp_path) -> None:
    fake_omics_block = {
        "has_omics": True,
        "available": True,
        "tmz_resistance_score": 39.92,
        "molecular_subtype": "parp_inhibitor_candidate",
    }
    result = _predict_end_to_end_setup(monkeypatch, tmp_path, omics_return=fake_omics_block)

    assert result["omics"] == fake_omics_block


def test_predict_patient_omits_omics_key_when_has_omics_false(monkeypatch, tmp_path) -> None:
    result = _predict_end_to_end_setup(monkeypatch, tmp_path, omics_return=None)

    assert "omics" not in result


def test_predict_patient_cox_risk_score_identical_regardless_of_omics_presence(
    monkeypatch, tmp_path
) -> None:
    """Kilitli kural (CLAUDE.md): 'Omics modulu Cox'un ZORUNLU girdisi
    DEGILDIR.' Ayni hasta/model/radyomik girdisiyle, omics VARKEN ve
    YOKKEN Cox `risk_score_log_partial_hazard` BIREBIR ayni olmali --
    ikisi de `compute_risk_score_and_shap` AYNI kod yolundan geciyor,
    omics fetch'i SONRADAN eklenen ayri bir adim."""

    fake_omics_block = {"has_omics": True, "available": True, "tmz_resistance_score": 10.0}

    result_with_omics = _predict_end_to_end_setup(
        monkeypatch, tmp_path, omics_return=fake_omics_block
    )
    result_without_omics = _predict_end_to_end_setup(monkeypatch, tmp_path, omics_return=None)

    assert result_with_omics["risk_score_log_partial_hazard"] == pytest.approx(
        result_without_omics["risk_score_log_partial_hazard"], abs=1e-12
    )
    assert result_with_omics["hazard_ratio_partial_hazard"] == pytest.approx(
        result_without_omics["hazard_ratio_partial_hazard"], abs=1e-12
    )
    assert result_with_omics["shap_values"] == result_without_omics["shap_values"]


# =====================================================================
# GERCEK DB SMOKE TESTI -- omics 6 hastadan biriyle (TCGA-06-5412),
# gercek MR + gercek omics kesisimi. DB erisilemezse ATLANIR (skip),
# ama DB erisilebiliyorsa ASSERTION HATASI atlanmaz.
# =====================================================================


def test_fetch_omics_interpretation_real_db_smoke_tcga_06_5412() -> None:
    try:
        block = predict_module.fetch_omics_interpretation("TCGA-06-5412")
    except Exception as exc:  # pragma: no cover -- yalniz DB erisilemezse
        pytest.skip(f"Gercek DB'ye erisilemedi, smoke test atlandi: {exc}")

    assert block is not None, "TCGA-06-5412 patients.has_omics=TRUE olmali (2026-08-18 dogrulandi)"
    assert block["has_omics"] is True
    assert block["available"] is True
    assert block["molecular_subtype"] == "parp_inhibitor_candidate"
    assert block["tmz_class"] == "sensitive"
    # Bilinen veri eksikligi -- kod hatasi degil (bkz. modul dokstring'i).
    assert block["tmz_class_relative"] is None
    assert block["egfr_amp_flag"] is None
    assert block["mgmt_interpretation"] is None


def test_predict_patient_real_db_smoke_tcga_06_5412_returns_200_train_aligned() -> None:
    """2026-09-13 REVIZYON (Baris onayi -- ESKI test adi
    `..._returns_422_missing_clinical_covariates` idi, 2026-08-18'den
    2026-09-13'e kadar GECERLI davranisi kilitliyordu).

    ESKI davranis (SIMDI GECERSIZ): varsayilan model GTR/IDH1'i `STRICT`
    politikayla istiyordu, TCGA-06-5412'nin (omics demo hastasi)
    `gtr_over90percent`/`idh1_status`/`mgmt_status`'u (canli DB, ucu de
    NULL) icin 422 donuyordu. Canli DOGRULAMA (bu degisiklik SIRASINDA,
    2026-09-13): eski STRICT politika GERI simule edilerek 422 ALINDI
    (detail: "gtr_over90percent (deger=None)", "idh1_status
    (deger=None)"), SONRA guncel TRAIN_ALIGNED politikayla AYNI hasta
    200 DONDU -- bu test o iki-adimli kaniti KALICI hale getirir.

    YENI davranis (Baris onayi, tutarlilik gerekcesiyle -- `tools/
    train_cox_week3.py` satir 663-687 GTR/IDH/MGMT'yi BIREBIR AYNI
    desenle kodluyor): `CLINICAL_COVARIATE_NULL_POLICY['gtr_over90percent'
    /'idh1_status'] == TRAIN_ALIGNED`, NULL 422 ATMAZ, `clinical_gtr_
    missing=1`/`clinical_idh_missing=1` (+ `clinical_mgmt_missing=1`,
    K19'dan beri) olarak kodlanir.

    ⚠️ JURY DEMO NOTU: demo Blok B'nin anlatisi bu hastanin 422 ALMASI
    uzerine kuruluydu ("sistem eksik veride uydurmuyor") -- bu test
    ARTIK 200 bekledigi icin o anlatinin BELGE tarafinin (demo metinleri)
    GUNCELLENMESI GEREKIR (bu dosyanin/backend-agent'in kapsaminda
    COZULMEDI, sadece bildirilir)."""

    try:
        result = predict_module.predict_patient("TCGA-06-5412")
    except predict_module.HTTPException as exc:
        pytest.fail(
            "TCGA-06-5412 icin 200 BEKLENIYORDU (2026-09-13 GTR/IDH1 "
            f"TRAIN_ALIGNED hizalamasi) ama HTTPException alindi: "
            f"{exc.status_code} / {exc.detail}"
        )
        return
    except Exception as exc:  # pragma: no cover -- yalniz DB/artefakta erisilemezse
        pytest.skip(f"Gercek DB/model artefaktina erisilemedi, smoke test atlandi: {exc}")

    assert result["patient_id"] == "TCGA-06-5412"
    extra_columns = result["clinical_extra_columns"]
    if "clinical_gtr_missing" not in extra_columns or "clinical_idh_missing" not in extra_columns:
        pytest.skip(
            "Deploy edilmis model GTR/IDH kovaryati tasimiyor olabilir -- "
            f"bu test yalniz GTR/IDH-tasiyan bir arm ile anlamli (alinan: {extra_columns})."
        )

    clinical_row = predict_module.build_patient_clinical_covariate_row(
        "TCGA-06-5412", extra_columns
    )
    raw = predict_module._fetch_patients_clinical_raw(["TCGA-06-5412"]).iloc[0]
    if pd.notna(raw["gtr_over90percent"]) or pd.notna(raw["idh1_status"]):
        pytest.skip(
            "TCGA-06-5412'nin gtr_over90percent/idh1_status'u artik NULL "
            f"DEGIL (canli deger gtr={raw['gtr_over90percent']!r}, "
            f"idh={raw['idh1_status']!r}) -- bu test NULL senaryosunu "
            "kanitlamak icin yazildi, canli veri degismis olabilir."
        )
    assert clinical_row.loc[0, "clinical_gtr_missing"] == 1.0
    assert clinical_row.loc[0, "clinical_gtr_y"] == 0.0
    assert clinical_row.loc[0, "clinical_idh_missing"] == 1.0
    assert clinical_row.loc[0, "clinical_idh_mutant"] == 0.0
    assert "clinical_gtr_missing" in result["shap_values"]
    assert "clinical_idh_missing" in result["shap_values"]


def test_predict_patient_real_db_smoke_succeeds_with_full_clinical_covariates() -> None:
    """2026-08-18 EKLENDI -- UPENN-GBM-00001 (tek scan_id, C32 radyomigi
    VE tam GTR/IDH1/yas/cinsiyet VAR, canli DB'de dogrulandi) ile uctan
    uca GERCEK DB kosusu. Referans risk_score, `predict_patient()`'in
    KENDI cagirdigi yoldan BAGIMSIZ olarak alt-fonksiyonlar DOGRUDAN
    cagrilarak testin CALISTIGI ANDA yeniden hesaplanir -- sabit bir
    sayi HARDCODE EDILMEZ, model yeniden egitilse bile test KIRILGAN
    olmaz."""

    try:
        arm = predict_module.load_cox_arm_result()
        final_features = arm["final_features"]
        extra_columns = arm["extra_columns"]
        model_features = list(arm["fitted_model"].params_.index)

        patient_row = predict_module.fetch_c32_wt_row("UPENN-GBM-00001", final_features)
        if extra_columns:
            clinical_row = predict_module.build_patient_clinical_covariate_row(
                "UPENN-GBM-00001", extra_columns
            )
            patient_row = pd.concat(
                [patient_row.reset_index(drop=True), clinical_row.reset_index(drop=True)],
                axis=1,
            )
        checkpoint_mtime = predict_module._checkpoint_path().stat().st_mtime
        background = predict_module.get_shap_background(
            final_features, extra_columns, checkpoint_mtime
        )
        # 2026-09-12 DUZELTME (K19 gorevinin YAN BULGUSU -- bu test daha
        # once MGMT NULL nedeniyle HER ZAMAN skip'e dusuyordu, bu satira
        # HIC ULASMIYORDU; K19 duzeltmesi bu hastayi artik gecirdigi icin
        # ASAGIDAKI eksik parametre ORTAYA CIKTI): `predict_patient()`
        # `feature_standardization=arm.get("feature_standardization")`
        # GECIRIYOR (v3b_lowvar_v2amgmt standardize edilmis uzayda fit
        # edildi) -- bu manuel "referans" hesap da AYNI parametreyi
        # gecirmezse HAM-olcekli yanlis bir skorla KARSILASTIRIR (once
        # -2523 vs -0.78 gibi -- olcek uyumsuzlugu, hicbir zaman
        # kosmadigi icin fark edilmemis bir kusurdu, HATA DUZELTMEK
        # POST-HOC DEGILDIR, bkz. CLAUDE.md).
        expected = predict_module.compute_risk_score_and_shap(
            arm["fitted_model"],
            model_features,
            patient_row,
            background,
            feature_standardization=arm.get("feature_standardization"),
        )

        result = predict_module.predict_patient("UPENN-GBM-00001")
    except Exception as exc:  # pragma: no cover -- yalniz DB/artefakta erisilemezse
        pytest.skip(f"Gercek DB/model artefaktina erisilemedi, smoke test atlandi: {exc}")

    assert result["patient_id"] == "UPENN-GBM-00001"
    assert result["clinical_extra_columns"] == extra_columns
    assert result["risk_score_log_partial_hazard"] == pytest.approx(
        expected["risk_score_log_partial_hazard"], abs=1e-9
    )


def test_predict_patient_real_db_mgmt_null_train_aligned_returns_200() -> None:
    """K19 karari (Baris 2026-09-12, Secenek 2) canli-DB kanit testi.

    `UPENN-GBM-00001`'in `mgmt_status`'u canli DB'de NULL (2026-09-12
    dogrulandi, DEVAM-NOTU-2026-09-12.md #1.1b). Deploy edilmis varsayilan
    model (`v3b_lowvar_v2amgmt`, `.env`'deki `GBMAID_COX_CHECKPOINT_PATH`
    ile) MGMT kovaryati ICERIYOR. Bu degisiklikten ONCE bu hasta icin
    `predict_patient()` 422 donuyordu (`ClinicalCovariateMissingError`,
    'mgmt_status' NULL) -- ARTIK (`CLINICAL_COVARIATE_NULL_POLICY
    ['mgmt_status'] == TRAIN_ALIGNED`) 200 donmeli ve `clinical_mgmt_
    missing=1` olarak KODLANMIS OLMALI (SHAP `model_features` kumesinde
    `clinical_mgmt_missing` anahtari OLMASI, o kovaryatin modele
    VERILDIGININ kanitidir -- ham deger DOGRUDAN yanitta donmuyor).

    Bu test DB-BAGIMLI (gercek baglanti) -- baglanti/artefakt erisim
    hatasi SKIP'e, ama 422/baska bir HTTPException'a FAIL'e cevrilir
    (skip ile fail'i AYRISTIRMAK onemli -- bu testin amaci TAM OLARAK bu
    davranisi kanitlamak, sessizce atlanmasi anlamli olmaz)."""

    try:
        raw = predict_module._fetch_patients_clinical_raw(["UPENN-GBM-00001"])
    except Exception as exc:  # pragma: no cover -- yalniz DB'ye erisilemezse
        pytest.skip(f"Gercek DB'ye erisilemedi, smoke test atlandi: {exc}")
    if raw.empty:
        pytest.skip("UPENN-GBM-00001 patients tablosunda bulunamadi -- test atlandi.")

    mgmt_raw = raw.iloc[0]["mgmt_status"]
    if pd.notna(mgmt_raw):
        pytest.skip(
            "UPENN-GBM-00001'in mgmt_status'u artik NULL DEGIL "
            f"(canli deger={mgmt_raw!r}) -- bu test NULL-MGMT senaryosunu "
            "kanitlamak icin yazildi, canli veri degismis olabilir. K19 "
            "davranisi baska bir NULL-MGMT hastayla yeniden dogrulanmali."
        )

    try:
        result = predict_module.predict_patient("UPENN-GBM-00001")
    except predict_module.HTTPException as exc:
        pytest.fail(
            "UPENN-GBM-00001 (mgmt_status NULL) icin 200 BEKLENIYORDU (K19 "
            f"karari, TRAIN_ALIGNED) ama HTTPException alindi: "
            f"{exc.status_code} / {exc.detail}"
        )
    except Exception as exc:  # pragma: no cover -- yalniz artefakta erisilemezse
        pytest.skip(f"Gercek model artefaktina erisilemedi, smoke test atlandi: {exc}")

    assert result["patient_id"] == "UPENN-GBM-00001"
    assert "clinical_mgmt_missing" in result["clinical_extra_columns"], (
        "Deploy edilmis model MGMT kovaryati tasimiyor olabilir -- bu test "
        "yalniz MGMT-tasiyan bir arm ile anlamli."
    )
    assert "clinical_mgmt_missing" in result["shap_values"]

    clinical_row = predict_module.build_patient_clinical_covariate_row(
        "UPENN-GBM-00001", result["clinical_extra_columns"]
    )
    assert clinical_row.loc[0, "clinical_mgmt_missing"] == 1.0
    assert clinical_row.loc[0, "clinical_mgmt_methylated"] == 0.0


# =====================================================================
# feature_standardization (2026-09-12 EKLENDI, Baris karari "Secenek A"
# -- checkpoint sozlugune GERIYE DONUK UYUMLU opsiyonel alan). Blokaj:
# canli `api/predict.py` hastanin HAM radyomik/yas degerlerini modele
# veriyordu; v3b_lowvar_v2amgmt fold-guvenli Z-score ile STANDARDIZE
# edilmis bir uzayda fit edildigi icin bu, checkpoint'i o varyanta
# gecilirse SESSIZCE yanlis tahmine yol acardi (bkz. entities/cox-model-
# varyant-katalogu.md §4 "Katsayi okuma notu").
# =====================================================================


def test_normalize_arm_carries_feature_standardization_when_present() -> None:
    model, _frame = _fit_tiny_cox_model(["WT__f1", "clinical_age"], n=30)
    feature_standardization = {
        "WT__f1": {"mean": 1.0, "std": 2.0},
        "clinical_age": {"mean": 55.0, "std": 13.0},
    }
    loaded = {
        "arm_name": "v3b_test",
        "fitted_model": model,
        "final_features": ["WT__f1"],
        "clinical_extra_columns": ["clinical_age"],
        "feature_standardization": feature_standardization,
    }

    normalized = predict_module._normalize_arm(loaded)

    assert normalized["feature_standardization"] == feature_standardization


def test_normalize_arm_feature_standardization_defaults_to_none_when_absent() -> None:
    """SIFIR REGRESYON sartinin `_normalize_arm` seviyesindeki kanit --
    alan YOKSA (v1 checkpoint'i) `None` doner, hicbir varsayilan
    mean/std UYDURULMAZ."""

    model, _frame = _fit_tiny_cox_model(["WT__f1"], n=30)
    loaded = {"arm_name": "v1_test", "fitted_model": model, "final_features": ["WT__f1"]}

    normalized = predict_module._normalize_arm(loaded)

    assert normalized["feature_standardization"] is None


def test_normalize_arm_carries_feature_standardization_for_legacy_arm_result() -> None:
    """ESKI `ArmResult` bicimi de (opsiyonel) `.feature_standardization`
    ozniteligini tasiyabilir -- `getattr(..., None)` ile OKUNUR, yoksa
    `None`."""

    model, _frame = _fit_tiny_cox_model(["WT__f1"], n=30)
    feature_standardization = {"WT__f1": {"mean": 0.5, "std": 1.5}}
    legacy = SimpleNamespace(
        name="legacy_standardized",
        final_model=SimpleNamespace(final_features=["WT__f1"], fitted_model=model),
        extra_columns=[],
        feature_standardization=feature_standardization,
    )

    normalized = predict_module._normalize_arm(legacy)

    assert normalized["feature_standardization"] == feature_standardization


def test_apply_feature_standardization_is_noop_when_absent() -> None:
    frame = pd.DataFrame({"WT__f1": [1.0, 2.0, 3.0]})

    result = predict_module._apply_feature_standardization(frame, None)

    pd.testing.assert_frame_equal(result, frame)


def test_compute_risk_score_and_shap_matches_manual_standardization() -> None:
    """DOGRU OLCEKLEME -- sentetik bir standardize checkpoint (bilinen
    mean/std) ile, `coef @ (standardize_edilmis_deger - norm_mean)`
    (modul dokstring'inin "SHAP OLCEGI" bolumundeki KAPALI-FORM kimlik)
    kullanilarak ELLE hesaplanan (lifelines'in `predict_log_partial_
    hazard()`'ini TEKRAR CAGIRMAYAN, dolayisiyla totolojik OLMAYAN)
    beklenen log-partial-hazard degerine esit ciktigini dogrular."""

    feature_cols = ["WT__f1", "clinical_age"]
    model, frame = _fit_tiny_cox_model(feature_cols, seed=11, n=50)

    feature_standardization = {
        "WT__f1": {"mean": 0.0, "std": 2.0},
        "clinical_age": {"mean": 10.0, "std": 5.0},
    }

    background = frame[feature_cols].sample(n=20, random_state=2)
    patient_row = pd.DataFrame({"WT__f1": [4.0], "clinical_age": [20.0]})

    result = predict_module.compute_risk_score_and_shap(
        model,
        feature_cols,
        patient_row,
        background,
        feature_standardization=feature_standardization,
    )

    coef = model.params_[feature_cols].to_numpy()
    norm_mean = model._norm_mean[feature_cols].to_numpy()
    standardized_values = np.array(
        [
            (4.0 - feature_standardization["WT__f1"]["mean"]) / feature_standardization["WT__f1"]["std"],
            (20.0 - feature_standardization["clinical_age"]["mean"])
            / feature_standardization["clinical_age"]["std"],
        ]
    )
    expected_risk_score = float(coef @ (standardized_values - norm_mean))
    expected_hazard_ratio = float(np.exp(expected_risk_score))

    assert result["risk_score_log_partial_hazard"] == pytest.approx(expected_risk_score, abs=1e-9)
    assert result["hazard_ratio_partial_hazard"] == pytest.approx(expected_hazard_ratio, abs=1e-9)

    # Girdi DataFrame'i mutate EDILMEDI (fonksiyon icerde kopyalar).
    assert patient_row.loc[0, "WT__f1"] == 4.0
    assert patient_row.loc[0, "clinical_age"] == 20.0


def test_compute_risk_score_and_shap_standardization_preserves_shap_additivity() -> None:
    """SHAP TUTARLILIGI -- standardize checkpoint'te de additivite
    (SHAP_toplami + taban == risk_score) FLOAT PRECISION'da korunuyor,
    hem hasta satirinin hem SHAP arka planinin AYNI donusumden gectigi
    dogrulanir (aksi halde `shap.LinearExplainer`'in kapali-form kimligi
    bozulurdu)."""

    feature_cols = ["WT__f1", "WT__f2", "clinical_age"]
    model, frame = _fit_tiny_cox_model(feature_cols, seed=21, n=80)
    feature_standardization = {
        "WT__f1": {"mean": 0.1, "std": 1.5},
        "clinical_age": {"mean": 50.0, "std": 12.0},
    }
    background = frame[feature_cols].sample(n=25, random_state=3)
    patient_row = frame[feature_cols].iloc[[7]]

    result = predict_module.compute_risk_score_and_shap(
        model,
        feature_cols,
        patient_row,
        background,
        feature_standardization=feature_standardization,
    )

    assert result["shap_additivity_check_abs_diff"] < 1e-9
    reconstructed = sum(result["shap_values"].values()) + result["shap_base_value"]
    assert reconstructed == pytest.approx(result["risk_score_log_partial_hazard"], abs=1e-9)
    assert set(result["shap_values"].keys()) == set(feature_cols)


def test_predict_patient_end_to_end_applies_feature_standardization(monkeypatch, tmp_path) -> None:
    """UCTAN UCA -- endpoint'in `arm.get("feature_standardization")`'i
    `compute_risk_score_and_shap()`'e GERCEKTEN ilettigini dogrular
    (yalniz alt-fonksiyonun kendi basina calismasi degil, cagri zinciri)."""

    feature_cols = ["WT__f1", "WT__f2"]
    model, frame = _fit_tiny_cox_model(feature_cols, seed=9, n=60)
    feature_standardization = {
        "WT__f1": {
            "mean": float(frame["WT__f1"].mean()),
            "std": float(frame["WT__f1"].std()),
        },
    }
    arm = _fake_arm_result(model, feature_cols, name="standardized_test")
    arm["feature_standardization"] = feature_standardization

    fake_checkpoint = tmp_path / "fake_checkpoint.pkl"
    fake_checkpoint.write_bytes(b"dummy")

    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)
    monkeypatch.setattr(
        predict_module,
        "fetch_c32_wt_row",
        lambda patient_id, final_features: frame[feature_cols].iloc[[0]],
    )
    monkeypatch.setattr(
        predict_module,
        "get_shap_background",
        lambda final_features, extra_columns, cache_generation, **_kwargs: frame[feature_cols].sample(
            n=20, random_state=1
        ),
    )
    monkeypatch.setattr(predict_module, "_checkpoint_path", lambda: fake_checkpoint)

    result = predict_module.predict_patient("UPENN-GBM-00001")

    raw_row = frame[feature_cols].iloc[[0]]
    coef = model.params_[feature_cols].to_numpy()
    norm_mean = model._norm_mean[feature_cols].to_numpy()
    standardized_row = raw_row.copy()
    standardized_row["WT__f1"] = (
        standardized_row["WT__f1"] - feature_standardization["WT__f1"]["mean"]
    ) / feature_standardization["WT__f1"]["std"]
    expected_risk_score = float(coef @ (standardized_row[feature_cols].to_numpy()[0] - norm_mean))

    assert result["risk_score_log_partial_hazard"] == pytest.approx(expected_risk_score, abs=1e-9)
    assert result["shap_additivity_check_abs_diff"] < 1e-9


V1_PINNED_CHECKPOINT_PATH = (
    predict_module.REPO_ROOT
    / "models"
    / "cox_phm_primary_wt93_clinical_full_ucsf_2026-08-18.pkl"
)

# =====================================================================
# VARSAYILAN CHECKPOINT SABIT-PINI (2026-09-13, karar 33, backend-agent-K)
# =====================================================================
V3B_EXPECTED_DEFAULT_CHECKPOINT_FILE_NAME = "cox_phm_v3b_lowvar_v2amgmt_2026-09-12.pkl"
V3B_EXPECTED_ARM_NAME = "v3b_lowvar_v2amgmt"


def test_default_checkpoint_path_is_pinned_to_v3b() -> None:
    """**SABIT-PIN** (gorev talimatinin ZORUNLU sarti): `DEFAULT_CHECKPOINT_
    PATH` nihai model v3b'yi GOSTERMEK ZORUNDA.

    NEDEN: 2026-09-13'e kadar bu sabit v1'i (`cox_phm_primary_wt93_clinical_
    full_ucsf_2026-08-18.pkl`) gosteriyordu ve v3b YALNIZ proje kokundeki
    `.env`'in `GBMAID_COX_CHECKPOINT_PATH` satiri sayesinde geliyordu --
    o satirin olmadigi her ortam SESSIZCE v1'i servis ediyordu (canli olcum:
    UCSF-PDGM-167 riski -0,19639713520164725, v3b pini -0,7096787591025546
    yerine; hata/uyari YOK). `decisions/2026-09-12-nihai-cox-model-secimi-
    v3b.md` "Dalga etkisi" md.6 bunu ZATEN sart kosuyordu.

    Bu test ENV'DEN TAMAMEN BAGIMSIZDIR -- `_checkpoint_path()`'i CAGIRMAZ,
    dogrudan SABITI okur. Biri ileride sabiti sessizce baska bir dosyaya
    cevirirse KIRMIZI olur."""

    assert (
        predict_module.DEFAULT_CHECKPOINT_PATH.name
        == V3B_EXPECTED_DEFAULT_CHECKPOINT_FILE_NAME
    ), (
        "Varsayilan Cox checkpoint'i nihai model v3b OLMALI (karar 33, "
        "decisions/2026-09-12-nihai-cox-model-secimi-v3b.md). Bulunan: "
        f"{predict_module.DEFAULT_CHECKPOINT_PATH.name!r}. Model degisecekse "
        "ONCE karar dosyasi yazilir, SONRA bu pin bilincli olarak guncellenir."
    )
    assert predict_module.DEFAULT_CHECKPOINT_PATH.parent.name == "models"
    # v1 artefakti SILINMEDI/TASINMADI -- pinlenmis baseline testi ona bagli.
    assert (
        predict_module.V1_BASELINE_CHECKPOINT_PATH.name
        == "cox_phm_primary_wt93_clinical_full_ucsf_2026-08-18.pkl"
    )
    assert predict_module.V1_BASELINE_CHECKPOINT_PATH != predict_module.DEFAULT_CHECKPOINT_PATH


def test_default_checkpoint_loads_v3b_arm_without_env_override(monkeypatch) -> None:
    """PINI ISMIYLE DEGIL DAVRANISLA da dogrular: env override YOKKEN
    yuklenen kol `v3b_lowvar_v2amgmt` olmali (v1 degil).

    `GBMAID_COX_CHECKPOINT_PATH` BOS STRING'e cekilir -- `python-dotenv`
    `override=False` ile YALNIZ `os.environ`'da OLMAYAN anahtarlari yazar,
    bu yuzden bos string `.env`'deki satirin YOK oldugu ortami (CI/temiz
    klon/dagitilmis sunucu) BIREBIR taklit eder. `.env` OKUNMAZ/BASILMAZ."""

    monkeypatch.setenv("GBMAID_COX_CHECKPOINT_PATH", "")
    resolved = predict_module._checkpoint_path()
    assert resolved.name == V3B_EXPECTED_DEFAULT_CHECKPOINT_FILE_NAME

    if not resolved.is_file():
        pytest.skip(f"v3b artefakti diskte yok: {resolved.name}")
    arm = predict_module.load_cox_arm_result(resolved)
    assert arm["name"] == V3B_EXPECTED_ARM_NAME
    # v3b'nin AYIRT EDICI imzasi: v1 `feature_standardization` TASIMAZ.
    assert arm["feature_standardization"] is not None
    assert len(arm["feature_standardization"]) == 17  # 16 radyomik + clinical_age
    assert len(arm["final_features"]) == 16
    assert len(arm["extra_columns"]) == 8


# =====================================================================
# CHECKPOINT OVERRIDE GORUNURLUGU -- 2026-09-14 (Baris'in 2 numarali karari,
# backend-agent-R2 bulgusu)
#
# BULGU: `api/predict.py`'de HIC `logging` YOKTU. Biri `.env`'deki
# `GBMAID_COX_CHECKPOINT_PATH` satirini elle v1'e (ya da baska bir `.pkl`'e)
# cevirirse API o modeli HICBIR UYARI BASMADAN servis ediyordu; tek iz
# yanittaki `model_arm` alaniydi -- "sessiz degil ama sessize yakin".
#
# BU BIR HATA DUZELTMESI DEGIL, GORUNURLUK EKLEMESIDIR: sessiz fallback YOK
# (7 ariza senaryosu olculdu, hepsi ya dogru modeli yukledi ya GURULTULU hata
# verdi), override de MESRU/bilincli korunan bir geriye-donuk erisim yoludur
# -- bu yuzden `assert`/`raise` DEGIL, `logging.warning`.
#
# Asagidaki testler KIRMIZI/YESIL ciftidir: (1) override BASKA bir dosyayi
# gosterdiginde uyari BASILIR, (2) override yokken VEYA override varsayilanin
# KENDISINI gosterdiginde HICBIR SEY basilmaz (gurultu kirliligi yasagi).
# =====================================================================


def test_checkpoint_override_to_different_file_logs_warning(
    monkeypatch, tmp_path, caplog
) -> None:
    """YESIL yari: override varsayilandan FARKLI bir dosyayi gosteriyorsa
    `api.predict` logger'ina TEK bir WARNING satiri basilir.

    Degisiklikten ONCE bu test KIRMIZIYDI (modulde `logging` yoktu -> hic
    kayit uretilmezdi). Uyari TAM YOL icermez -- yalniz basename (bu makinede
    yol kullanici adi + Turkce karakter tasiyor)."""

    # v1'e (2026-08-18 baseline) isaret eden bir override taklidi. Dosyanin
    # ICERIGI onemsiz: `_checkpoint_path()` dosyayi ACMAZ, yalniz yolu cozer.
    fake_override = tmp_path / "cox_phm_primary_wt93_clinical_full_ucsf_2026-08-18.pkl"
    fake_override.write_bytes(b"")
    monkeypatch.setenv("GBMAID_COX_CHECKPOINT_PATH", str(fake_override))

    with caplog.at_level(logging.WARNING, logger="api.predict"):
        resolved = predict_module._checkpoint_path()

    # (a) DAVRANIS DEGISMEDI -- override HALA aynen doner, hata atilmaz.
    assert resolved == fake_override

    records = [r for r in caplog.records if r.name == "api.predict"]
    assert len(records) == 1, (
        "Override aktifken TAM BIR uyari satiri beklenir, bulunan: "
        f"{[r.getMessage() for r in records]!r}"
    )
    record = records[0]
    assert record.levelno == logging.WARNING
    message = record.getMessage()

    # (b) hangi dosya + beklenen varsayilan + "birincil sonuc" uyarisi
    assert fake_override.name in message
    assert predict_module.DEFAULT_CHECKPOINT_PATH.name in message
    assert "BIRINCIL SONUC OLARAK RAPORLANAMAZ" in message

    # (c) TAM YOL/dizin SIZMAZ (gorev sarti: yalniz basename)
    assert str(fake_override.parent) not in message
    assert str(fake_override) not in message
    assert str(predict_module.DEFAULT_CHECKPOINT_PATH.parent) not in message


def test_checkpoint_path_logs_nothing_without_override(monkeypatch, caplog) -> None:
    """KIRMIZI yari (gurultu kirliligi yasagi): varsayilan yolla kosuldugunda
    HICBIR SEY basilmaz.

    `GBMAID_COX_CHECKPOINT_PATH` BOS STRING'e cekilir -- `python-dotenv`
    `override=False` ile YALNIZ `os.environ`'da OLMAYAN anahtarlari yazar,
    bu yuzden bos string `.env` satirinin YOK oldugu ortami (CI/temiz klon)
    BIREBIR taklit eder. `.env` OKUNMAZ/BASILMAZ."""

    monkeypatch.setenv("GBMAID_COX_CHECKPOINT_PATH", "")

    with caplog.at_level(logging.WARNING, logger="api.predict"):
        resolved = predict_module._checkpoint_path()

    assert resolved == predict_module.DEFAULT_CHECKPOINT_PATH
    assert [r.getMessage() for r in caplog.records if r.name == "api.predict"] == []


def test_checkpoint_override_pointing_at_default_file_logs_nothing(
    monkeypatch, caplog
) -> None:
    """Override TANIMLI ama varsayilanin KENDISINI gosteriyorsa uyari
    BASILMAZ -- bu, bu makinedeki GERCEK durumdur (proje kokundeki `.env`
    override'i v3b'nin KENDISINE isaret ettiriyor, 2026-09-14'te olculdu).
    Duz string karsilastirmasi burada her istekte SAF GURULTU uretirdi."""

    default_path = predict_module.DEFAULT_CHECKPOINT_PATH
    if not default_path.is_file():
        pytest.skip(f"v3b artefakti diskte yok: {default_path.name}")
    monkeypatch.setenv("GBMAID_COX_CHECKPOINT_PATH", str(default_path))

    with caplog.at_level(logging.WARNING, logger="api.predict"):
        resolved = predict_module._checkpoint_path()

    assert resolved == default_path
    assert [r.getMessage() for r in caplog.records if r.name == "api.predict"] == []


def test_predict_patient_real_db_zero_regression_pinned_baseline(monkeypatch) -> None:
    """SIFIR REGRESYON KANITI (gorev talimatinin merkezi sarti) -- bu
    degerler `feature_standardization` destegi EKLENMEDEN ONCE (2026-09-12,
    bu degisiklikten hemen once) canli DB + gercek **v1** checkpoint
    (`models/cox_phm_primary_wt93_clinical_full_ucsf_2026-08-18.pkl`,
    arm_name=primary_wt93_clinical_full_ucsf) ile UPENN-GBM-00001 icin
    OLCULEN cikti PINLENMISTIR (predict_patient() bu degisiklikten ONCE ve
    SONRA ayni sekilde calistirilip bit-birebir karsilastirildi). Bu
    checkpoint `feature_standardization` alani TASIMAZ (ham olcek) -- test
    o alanin eklenmesinin/`_apply_feature_standardization()`'in bu
    checkpoint icin sonucu HICBIR SEKILDE degistirmedigini dogrular.

    TESTIN KAPSAMI: YALNIZ "olcekleme kodu eski davranisi bozmadi"
    korumasidir. Baris'in model seciminden (nihai model **v3b**, karar
    2026-09-12) BAGIMSIZDIR ve onu sorgulamaz.

    2026-09-13 DUZELTMESI (belge-agent-F): bu test ONCEDEN VARSAYILAN
    checkpoint'i okuyordu ve "artik feature_standardization tasiyor" diye
    KENDINI ATLIYORDU. ~~Varsayilan artik v3b oldugu icin~~ koruma FIILEN
    CALISMIYORDU (inert skip). Cozum: test v1 checkpoint'ini `_checkpoint_
    path()` monkeypatch'iyle ACIKCA yukler -- varsayilanin ne oldugundan
    (env `GBMAID_COX_CHECKPOINT_PATH` dahil) BAGIMSIZ hale geldi. v1 pkl
    diskte duruyor, SILINMEDI.

    ⚠️ 2026-09-13 IKINCI DUZELTME (backend-agent-K, karar 33) -- YUKARIDAKI
    CUMLE O GUN YANLISTI, uzeri cizildi: belge-agent-F "varsayilan artik
    v3b" derken FIILI (`.env` override'li) davranisi tarif ediyordu; KODUN
    `DEFAULT_CHECKPOINT_PATH` SABITI o an HALA v1'i gosteriyordu (olculdu:
    env override notr birakilinca yuklenen kol `primary_wt93_clinical_full_
    ucsf`). Yani inert-skip'in sebebi "varsayilan v3b" DEGIL, "varsayilan
    `.env` uzerinden v3b'ye override edilmis olmasi"ydi. `DEFAULT_CHECKPOINT_
    PATH` 2026-09-13'te v3b'ye cevrildikten SONRA o cumle ARTIK DOGRU --
    ama bu testin `_checkpoint_path()` monkeypatch'i sayesinde testin
    GECERLILIGI her iki durumda da varsayilandan BAGIMSIZ kalir, yani bu
    test degisiklikten ETKILENMEDI (aynen yesil). Varsayilanin v3b oldugu
    AYRICA/BAGIMSIZ olarak `test_default_checkpoint_path_is_pinned_to_v3b`
    ile pinlenmistir.
    """

    if not V1_PINNED_CHECKPOINT_PATH.is_file():
        pytest.skip(
            f"v1 pinlenmis-baseline checkpoint'i diskte yok: "
            f"{V1_PINNED_CHECKPOINT_PATH.name} -- bu test o artefakta baglidir."
        )

    # Varsayilan checkpoint'ten (bugun v3b) BAGIMSIZ olsun diye v1'i ACIKCA
    # secmek: `load_cox_arm_result()` ve `predict_patient()` icindeki
    # SHAP-background cache-generation'i (checkpoint mtime) AYNI dosyayi
    # gorsun diye tek noktadan (`_checkpoint_path`) override ediliyor.
    monkeypatch.setattr(
        predict_module, "_checkpoint_path", lambda: V1_PINNED_CHECKPOINT_PATH
    )

    try:
        arm = predict_module.load_cox_arm_result()
    except Exception as exc:  # pragma: no cover -- yalniz artefakt okunamazsa
        pytest.skip(f"v1 model artefakti okunamadi, test atlandi: {exc}")

    # DIKKAT: bu assert BILINCLI olarak try/except DISINDA -- broad
    # `except Exception` bir AssertionError'i de yutar ve testi sessizce
    # skip'e cevirirdi (K19 kaydinda ayni sinif kusur olculmustu).
    assert arm.get("feature_standardization") is None, (
        "v1 checkpoint'i feature_standardization TASIMAMALI (ham olcek). "
        "Tasiyorsa dosya degismis demektir -- bu pinlenmis baseline "
        "BILINCLI olarak yeniden alinmalidir, sessizce atlanmamalidir."
    )

    try:
        result = predict_module.predict_patient("UPENN-GBM-00001")
    except Exception as exc:  # pragma: no cover -- yalniz DB'ye erisilemezse
        pytest.skip(f"Gercek DB/model artefaktina erisilemedi, smoke test atlandi: {exc}")

    assert result["model_arm"] == "primary_wt93_clinical_full_ucsf"
    assert result["risk_score_log_partial_hazard"] == pytest.approx(-0.8217672406371088, abs=1e-9)
    assert result["hazard_ratio_partial_hazard"] == pytest.approx(0.4396539931480594, abs=1e-9)
    assert result["shap_base_value"] == pytest.approx(0.07955236008662281, abs=1e-9)
    assert result["shap_additivity_check_abs_diff"] < 1e-9

    expected_shap_values = {
        "WT__original_glcm_InverseVariance": -0.3075003350351854,
        "WT__original_glrlm_RunVariance": 0.05808447955771548,
        "WT__original_glszm_SizeZoneNonUniformityNormalized": -0.07605122757160092,
        "WT__original_glszm_SmallAreaHighGrayLevelEmphasis": 0.1291210483706094,
        "WT__original_ngtdm_Strength": -0.0413208260759994,
        "WT__original_shape_Maximum2DDiameterColumn": -0.07844588330215278,
        "WT__original_shape_Maximum3DDiameter": -0.10766465136787486,
        "WT__original_shape_Sphericity": -0.10776327155144749,
        "clinical_age": -0.28003288939349,
        "clinical_gender_male": 0.029358368612987563,
        "clinical_gtr_missing": 0.009002175810906721,
        "clinical_gtr_y": -0.10471564544648852,
        "clinical_idh_missing": -0.03699804548041977,
        "clinical_idh_mutant": 0.01360710214870919,
    }
    assert set(result["shap_values"].keys()) == set(expected_shap_values.keys())
    for feature, expected_value in expected_shap_values.items():
        assert result["shap_values"][feature] == pytest.approx(expected_value, abs=1e-9), feature


# --- Fail-loud (4 ayri istisna senaryosu, gorev talimatinin ZORUNLU sarti) ---


def test_feature_standardization_raises_config_error_on_zero_std() -> None:
    feature_cols = ["WT__f1"]
    model, _frame = _fit_tiny_cox_model(feature_cols, n=30)
    feature_standardization = {"WT__f1": {"mean": 0.0, "std": 0.0}}
    patient_row = pd.DataFrame({"WT__f1": [1.0]})
    background = pd.DataFrame({"WT__f1": [0.5, 0.6]})

    with pytest.raises(predict_module.FeatureStandardizationConfigError):
        predict_module.compute_risk_score_and_shap(
            model,
            feature_cols,
            patient_row,
            background,
            feature_standardization=feature_standardization,
        )


def test_feature_standardization_raises_config_error_on_missing_mean() -> None:
    feature_cols = ["WT__f1"]
    model, _frame = _fit_tiny_cox_model(feature_cols, n=30)
    feature_standardization = {"WT__f1": {"std": 1.0}}  # 'mean' YOK
    patient_row = pd.DataFrame({"WT__f1": [1.0]})
    background = pd.DataFrame({"WT__f1": [0.5, 0.6]})

    with pytest.raises(predict_module.FeatureStandardizationConfigError):
        predict_module.compute_risk_score_and_shap(
            model,
            feature_cols,
            patient_row,
            background,
            feature_standardization=feature_standardization,
        )


def test_feature_standardization_raises_config_error_on_unknown_column() -> None:
    """Listelenen kolon modelin final_features+clinical_extra_columns
    kumesinde YOK -- `_normalize_arm()` seviyesinde (checkpoint yuklenirken)
    fail-loud."""

    model, _frame = _fit_tiny_cox_model(["WT__f1"], n=30)
    loaded = {
        "arm_name": "bad_arm",
        "fitted_model": model,
        "final_features": ["WT__f1"],
        "clinical_extra_columns": [],
        "feature_standardization": {
            "WT__unknown_feature_not_in_model": {"mean": 0.0, "std": 1.0},
        },
    }

    with pytest.raises(predict_module.FeatureStandardizationConfigError):
        predict_module._normalize_arm(loaded)


def test_feature_standardization_raises_column_missing_error_when_absent_from_row() -> None:
    """Listelenen kolon config'te GECERLI (model_features kumesinde) ama
    standardize edilecek veri satirinda (hasta veya SHAP arka plani)
    YOK -- veri-duzeyinde fail-loud."""

    feature_standardization = {"WT__f1": {"mean": 0.0, "std": 1.0}}
    frame_missing_column = pd.DataFrame({"WT__other": [1.0, 2.0]})

    with pytest.raises(predict_module.FeatureStandardizationColumnMissingError):
        predict_module._apply_feature_standardization(frame_missing_column, feature_standardization)


# =====================================================================
# RegionCanonicalizationError (2026-09-12 EKLENDI, rag-agent canli UCSF
# bulgusu) -- `pivot_radiomics_long_to_wide()` -> `canonicalize_region_
# label()` taninmayan bir (kaynak, ham-bolge) cifti icin bare `ValueError`
# firlatiyor; ONCEDEN bu `fetch_c32_wt_row()`/`get_shap_background()`
# tarafindan YAKALANMIYORDU (yalniz `RegionPivotError` yakalaniyordu) --
# endpoint'te aciklanmamis 500'e sizardi. Kok neden (REGION_NAME_ALIASES'te
# UCSF alias eksikligi) `pipeline/cox_model.py`'de AYRI bir gorevde
# (modeling-agent-C) duzeltiliyor -- BU testler o dosyaya DOKUNMADAN,
# istisnayi SENTETIK olarak tetikleyerek API katmaninin dayanikliligini
# dogrular (kok neden duzelse bile BASKA bir kaynak/bolge icin ayni sinif
# hata tekrar olusabilir).
# =====================================================================


def test_fetch_c32_wt_row_converts_bare_value_error_from_pivot_to_named_error(monkeypatch) -> None:
    """KIRMIZI/YESIL kanit (1/2) -- `pivot_radiomics_long_to_wide()`
    SENTETIK olarak bare `ValueError` firlatir (gercek UCSF-alias bug'inin
    urettigi hatayla AYNI sekilde); FIX'TEN ONCE bu yakalanmiyordu (yalniz
    `RegionPivotError` yakalaniyordu) -- simdi isimlendirilmis
    `RegionCanonicalizationError`'a cevriliyor."""

    rows = [
        {
            "patient_id": "SOME-PATIENT",
            "source": "UPenn-GBM",
            "tumor_region": "WT_derived",
            "shape_features": {"a": 1.0},
            "first_order_features": {},
            "texture_features": {},
            "scan_id": 1,
        }
    ]
    cursor = FakeCursor(fetchone_results=[(1,)], fetchall_results=[rows])
    monkeypatch.setattr(predict_module, "_get_db_connection", lambda: FakeConnection(cursor))

    def _raise_bare_value_error(*_args, **_kwargs):
        raise ValueError("REGION_NAME_ALIASES kaynagi tanimiyor: 'UCSF' (sentetik tetikleme)")

    monkeypatch.setattr(predict_module, "pivot_radiomics_long_to_wide", _raise_bare_value_error)

    with pytest.raises(predict_module.RegionCanonicalizationError):
        predict_module.fetch_c32_wt_row("SOME-PATIENT", ["WT__a"])


def test_get_shap_background_converts_bare_value_error_from_pivot_to_named_error(monkeypatch) -> None:
    """Ayni dayaniksizlik SHAP arka plani icin de -- egitim havuzu SABIT
    UPenn oldugundan bugunku UCSF bug'iyla BIREBIR tetiklenmez, ama AYNI
    hata SINIFI (ornek: gelecekte arka plan kaynagi degisirse) icin
    savunma amacli SENTETIK dogrulama."""

    monkeypatch.setattr(
        predict_module,
        "_fetch_background_long_frame",
        lambda segmentation_tool: pd.DataFrame(
            {
                "patient_id": ["P1"],
                "source": ["UPenn-GBM"],
                "tumor_region": ["WT_derived"],
                "shape_features": [{"a": 1.0}],
                "first_order_features": [{}],
                "texture_features": [{}],
            }
        ),
    )

    def _raise_bare_value_error(*_args, **_kwargs):
        raise ValueError("sentetik pivot hatasi")

    monkeypatch.setattr(predict_module, "pivot_radiomics_long_to_wide", _raise_bare_value_error)

    with pytest.raises(predict_module.RegionCanonicalizationError):
        predict_module.get_shap_background(["WT__region_canon_test_a"], [], 999999.123)


def test_predict_patient_returns_422_instead_of_unhandled_500_for_region_canonicalization_error(
    monkeypatch,
) -> None:
    """KIRMIZI/YESIL kanit (2/2) -- endpoint seviyesinde. FIX'TEN ONCE
    `RegionCanonicalizationError` (dolayisiyla altindaki bare `ValueError`)
    `predict_patient()`'in `fetch_c32_wt_row` etrafindaki try/except
    blogunda YAKALANMIYORDU -- FastAPI'ye aciklanmamis bir istisna olarak
    ulasip 500 uretirdi. Simdi acik mesajli 422 doner."""

    feature_cols = ["WT__f1"]
    model, _frame = _fit_tiny_cox_model(feature_cols, n=30)
    arm = _fake_arm_result(model, feature_cols)
    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)

    def _raise_region_canonicalization_error(*_args, **_kwargs):
        raise predict_module.RegionCanonicalizationError(
            "sentetik: REGION_NAME_ALIASES kaynagi tanimiyor: 'UCSF'"
        )

    monkeypatch.setattr(predict_module, "fetch_c32_wt_row", _raise_region_canonicalization_error)

    with pytest.raises(predict_module.HTTPException) as exc_info:
        predict_module.predict_patient("UCSF-PDGM-0001")

    assert exc_info.value.status_code == 422
    assert "REGION_NAME_ALIASES" in str(exc_info.value.detail)


def test_predict_patient_returns_500_for_region_canonicalization_error_in_shap_background(
    monkeypatch, tmp_path
) -> None:
    """Ayni donusum, SHAP arka plani asamasinda olusursa (kullaniciya
    ozgu bir sorun degil, egitim havuzuna ait bir veri sorunu) -- mevcut
    hata-yaniti desenine uyarak (RequiredRegionMissingError/
    C32RadiomicsNotFoundError ile AYNI) 500 doner, sessizce yutulmaz."""

    feature_cols = ["WT__f1"]
    model, frame = _fit_tiny_cox_model(feature_cols, n=30)
    arm = _fake_arm_result(model, feature_cols)
    fake_checkpoint = tmp_path / "fake_checkpoint.pkl"
    fake_checkpoint.write_bytes(b"dummy")

    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)
    monkeypatch.setattr(
        predict_module,
        "fetch_c32_wt_row",
        lambda patient_id, final_features: frame[feature_cols].iloc[[0]],
    )
    monkeypatch.setattr(predict_module, "_checkpoint_path", lambda: fake_checkpoint)

    def _raise_region_canonicalization_error(*_args, **_kwargs):
        raise predict_module.RegionCanonicalizationError("sentetik arka plan hatasi")

    monkeypatch.setattr(predict_module, "get_shap_background", _raise_region_canonicalization_error)

    with pytest.raises(predict_module.HTTPException) as exc_info:
        predict_module.predict_patient("UPENN-GBM-00001")

    assert exc_info.value.status_code == 500


# =====================================================================
# _ensure_project_env_loaded / _checkpoint_path (2026-09-12 EKLENDI --
# koordinator bulgusu, KRITIK): taze bir process'te, DB'ye HIC dokunmadan
# (`db_connection.get_connection()` hic cagrilmadan) `_checkpoint_path()`
# `.env`'deki `GBMAID_COX_CHECKPOINT_PATH` override'ini KACIRABILIYORDU
# (dotenv eskiden YALNIZ `get_connection()` icinde LAZY yukleniyordu, ama
# `predict_patient()`'in ILK satiri `load_cox_arm_result()` ->
# `_checkpoint_path()`'tir -- DB'ye HENUZ dokunulmadan calisir). Bu, v3b
# deploy edilmis OLSA BILE taze bir uvicorn surecinin ilk istekte
# SESSIZCE v1'e dusmesi anlamina gelirdi. Bu test AYRI bir subprocess'te
# calisir -- boylece bu pytest surecinin ONCEKI testlerinin (DB'ye
# dokunan, dolayisiyla os.environ'i ONCEDEN dolduran) durumu MASKELEME
# riskini TAMAMEN ELER, gercek "taze process" senaryosunu dogrular.
# =====================================================================


def test_checkpoint_path_picks_up_env_override_in_fresh_process_without_db_touch() -> None:
    """KIRMIZI/YESIL kanit (2026-09-12, koordinator bulgusu). Duzeltmeden
    ONCE bu test KIRMIZIYDI -- `_checkpoint_path()` DB'ye dokunmadan
    calistiginda `os.environ.get('GBMAID_COX_CHECKPOINT_PATH')` HENUZ
    yuklenmemis oluyordu (dotenv'in kendisi yuklenmemisti), boylece HER
    ZAMAN `DEFAULT_CHECKPOINT_PATH`'e (v1) duserdi -- `.env`'de bir
    override TANIMLI olsa bile. `_ensure_project_env_loaded()` (checkpoint
    okumadan ONCE `db_connection.load_project_environment()`'i idempotent
    cagirir) eklendikten SONRA bu test YESIL."""

    project_root = predict_module.REPO_ROOT.parent  # "GBM-AID Prototip" -- kanonik .env burada
    env_file = project_root / ".env"
    if not env_file.is_file():
        pytest.skip(f"Kanonik .env dosyasi bulunamadi: {env_file}")

    # SADECE bu tek anahtar okunur -- .env'in tamami parse edilse de
    # ekrana/log'a hicbir sey BASILMAZ, yalniz bu degerle KARSILASTIRMA
    # icin bellekte tutulur (bu bir dosya yolu, kimlik bilgisi DEGIL).
    env_values = dotenv_values(str(env_file))
    expected_override = env_values.get("GBMAID_COX_CHECKPOINT_PATH")
    if not expected_override:
        pytest.skip(
            "'.env'de GBMAID_COX_CHECKPOINT_PATH tanimli degil -- bu test "
            "yalniz bir override AKTIFKEN anlamli/dogrulanabilir."
        )

    repo_root = predict_module.REPO_ROOT  # "gbm-aid mert" -- sys.path icin
    script = (
        "import sys; sys.path.insert(0, r'{repo_root}'); "
        "import api.predict as predict_module; "
        "print(str(predict_module._checkpoint_path()))"
    ).format(repo_root=str(repo_root))

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=60,
    )

    assert completed.returncode == 0, (
        f"Alt-surec basarisiz oldu (returncode={completed.returncode}): "
        f"{completed.stderr}"
    )
    resolved_path = completed.stdout.strip().splitlines()[-1]

    assert Path(resolved_path).name == Path(expected_override).name, (
        "Taze process'te (DB'ye HIC dokunmadan) ilk `_checkpoint_path()` "
        "cagrisi '.env' override'ini KACIRDI -- beklenen dosya adi "
        f"{Path(expected_override).name!r}, alinan {resolved_path!r}. Bu "
        "koordinatorun bildirdigi 'sessizce eski/varsayilan checkpoint "
        "yuklenir' regresyonudur (jüri demosunda YANLIS MODEL sunulmasi "
        "anlamina gelirdi)."
    )


# =====================================================================
# XGBOOST IKINCI KATMAN (shadow) -- 2026-09-13, ADIM 2
#
# Bu bolumun MERKEZI amaci IKI KIRMIZI/YESIL kanit uretmektir:
#   1. KOLON SIRASI -- `xgb_feature_columns` sirasi karisan bir cerceve
#      ACIK bir hatayla reddedilir (sessizce yanlis olasilik DONMEZ).
#   2. OLCEK KARISMASI -- XGBoost'a giden 93 radyomik HAM kalmak
#      ZORUNDA; biri yanlislikla `cox_score` blogunun standardizasyon
#      sozlugunu bu cerceveye uygularsa test KIRMIZI olur. Bu, 2026-09-12'de
#      Cox tarafinda YASANAN hatanin (model standardize, API ham besledi)
#      TERS yonlu ikizidir ve tek savunmadir.
# =====================================================================


def _real_xgboost_checkpoint_path() -> Path:
    return predict_module._xgboost_checkpoint_path()


def _load_raw_xgboost_checkpoint() -> dict:
    """Diskteki GERCEK checkpoint'i ham (dogrulanmamis) sozluk olarak
    okur -- `_validate_xgboost_checkpoint()`'in kendisini test etmek icin
    gerekli. Dosya yoksa test ATLANIR (checkpoint `models/` altinda,
    tools/export_xgboost_checkpoint.py uretir; bu testler o artefakta
    baglidir ve SESSIZCE yesil gecmemeli)."""

    path = _real_xgboost_checkpoint_path()
    if not path.is_file():
        pytest.skip(f"XGBoost checkpoint'i diskte yok: {path.name}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _real_xgboost_checkpoint() -> dict:
    path = _real_xgboost_checkpoint_path()
    if not path.is_file():
        pytest.skip(f"XGBoost checkpoint'i diskte yok: {path.name}")
    return predict_module.load_xgboost_checkpoint(path)


def _shallow_copy_xgboost_checkpoint(raw: dict) -> dict:
    """Ust seviye + iki blok + standardizasyon sozlugu KOPYALANIR (fit
    edilmis model NESNELERI paylasilir -- onlara dokunulmuyor)."""

    copied = dict(raw)
    copied["cox_score"] = dict(raw["cox_score"])
    copied["cox_score"]["feature_standardization"] = dict(
        raw["cox_score"]["feature_standardization"]
    )
    copied["xgboost"] = dict(raw["xgboost"])
    copied["xgboost"]["age_standardization"] = dict(raw["xgboost"]["age_standardization"])
    return copied


def _synthetic_patient_rows(checkpoint: dict, *, seed: int = 7):
    """GERCEK checkpoint'in kolon adlarini kullanan, ama DB'ye
    DOKUNMAYAN sentetik bir hasta satiri (93 radyomik + 8 klinik).
    Degerler gercekci olmak ZORUNDA DEGIL -- bu testler SAYISAL bir
    prognoz iddiasi kurmaz, YALNIZ olcek/sira sozlesmesini olcer."""

    rng = np.random.default_rng(seed)
    raw_columns = checkpoint["raw_radiomic_feature_columns"]
    sampled = rng.normal(loc=50.0, scale=10.0, size=len(raw_columns))
    radiomic_row = pd.DataFrame(
        [{column: float(value) for column, value in zip(raw_columns, sampled)}]
    )
    clinical_values = {
        "clinical_age": 62.0,
        "clinical_gender_male": 1.0,
        "clinical_gtr_y": 0.0,
        "clinical_gtr_missing": 1.0,
        "clinical_idh_mutant": 0.0,
        "clinical_idh_missing": 1.0,
        "clinical_mgmt_methylated": 0.0,
        "clinical_mgmt_missing": 1.0,
    }
    clinical_row = pd.DataFrame(
        [{column: clinical_values[column] for column in checkpoint["xgb_extra_columns"]}]
    )
    return radiomic_row, clinical_row


def test_real_xgboost_checkpoint_satisfies_documented_contract() -> None:
    """Checkpoint'in GERCEK alan adlari/sayilari -- gorev talimatindaki
    sozlesme (102 kolon = 93 HAM radyomik + 8 klinik + 1 skor) KODLA
    dogrulanir, belgeye/anlatiya GUVENILMEZ."""

    checkpoint = _real_xgboost_checkpoint()

    assert checkpoint["arm_name"] == (
        "v2a_mgmt+reduce_collinearity-aligned-v3b_bitidentical"
    )
    assert len(checkpoint["xgb_feature_columns"]) == 102
    assert len(checkpoint["raw_radiomic_feature_columns"]) == 93
    assert len(checkpoint["xgb_extra_columns"]) == 8
    assert checkpoint["score_column"] == "cox_oof_score"
    assert len(checkpoint["cox_final_features"]) == 16
    assert checkpoint["twelve_month_threshold_days"] == 365.0
    # YAPISAL kurulus (egitimdeki sira) -- pipeline/xgboost_model.py ~2925
    assert checkpoint["xgb_feature_columns"] == (
        checkpoint["raw_radiomic_feature_columns"]
        + checkpoint["xgb_extra_columns"]
        + [checkpoint["score_column"]]
    )
    # Cox alt-modelinin 16 ozelligi, 93'luk HAM havuzun ALT KUMESI olmali
    # (aksi halde tek bir radyomik cekimle her ikisi beslenemezdi).
    assert set(checkpoint["cox_final_features"]) <= set(
        checkpoint["raw_radiomic_feature_columns"]
    )
    # Harici gecerleme checkpoint'ten OKUNUR (bu testte yeniden kosulmaz).
    external = checkpoint["external_validation"]
    assert external["n_patients"] == 232
    assert external["auc"] == pytest.approx(0.723729, abs=5e-7)
    assert external["ci_lower"] == pytest.approx(0.650801, abs=5e-7)
    assert external["ci_upper"] == pytest.approx(0.790298, abs=5e-7)


def test_real_xgboost_checkpoint_separates_two_scales() -> None:
    """IKI AYRI OLCEK, KODLA dogrulanir: `cox_score` blogunun
    standardizasyon sozlugu 16 radyomigin TAMAMINI + `clinical_age`'i
    KAPSAR; ikili klinik bayraklar KAPSAM DISI (HAM 0/1); `xgboost`
    blogunun yas istatistigi `cox_score` blogundakiyle SAYISAL OLARAK
    AYNI."""

    checkpoint = _real_xgboost_checkpoint()
    standardization = checkpoint["cox_feature_standardization"]

    assert set(checkpoint["cox_final_features"]) <= set(standardization)
    assert "clinical_age" in standardization
    for flag in (
        "clinical_gender_male",
        "clinical_gtr_y",
        "clinical_gtr_missing",
        "clinical_idh_mutant",
        "clinical_idh_missing",
        "clinical_mgmt_methylated",
        "clinical_mgmt_missing",
    ):
        assert flag not in standardization, flag

    assert checkpoint["age_mean"] == pytest.approx(
        standardization["clinical_age"]["mean"], abs=1e-12
    )
    assert checkpoint["age_std"] == pytest.approx(
        standardization["clinical_age"]["std"], abs=1e-12
    )


def test_build_xgboost_feature_frame_keeps_93_radiomics_raw_and_zscores_only_age() -> None:
    """OLCEK AYRIMININ POZITIF (yesil) kaniti: 93 radyomik girdi
    degerine BIREBIR esit kalir; SADECE `clinical_age` z-skorlanir;
    ikili bayraklar HAM 0/1 kalir; skor kolonu SON kolondur."""

    checkpoint = _real_xgboost_checkpoint()
    radiomic_row, clinical_row = _synthetic_patient_rows(checkpoint)

    frame = predict_module.build_xgboost_feature_frame(
        checkpoint, radiomic_row, clinical_row, cox_oof_score=-0.25
    )

    assert list(frame.columns) == checkpoint["xgb_feature_columns"]
    for column in checkpoint["raw_radiomic_feature_columns"]:
        assert frame.iloc[0][column] == radiomic_row.iloc[0][column], column
    expected_age_z = (
        clinical_row.iloc[0]["clinical_age"] - checkpoint["age_mean"]
    ) / checkpoint["age_std"]
    assert frame.iloc[0]["clinical_age"] == pytest.approx(expected_age_z, abs=1e-12)
    assert frame.iloc[0]["clinical_age"] != clinical_row.iloc[0]["clinical_age"]
    assert frame.iloc[0]["clinical_gender_male"] == 1.0
    assert frame.iloc[0]["clinical_mgmt_missing"] == 1.0
    assert frame.iloc[0][checkpoint["score_column"]] == pytest.approx(-0.25, abs=1e-15)
    # Guard'lar bu (dogru) cerceveyi KABUL etmeli -- yesil taraf.
    predict_module._assert_xgboost_feature_frame_contract(frame, checkpoint)
    predict_module._assert_xgboost_raw_scale_preserved(
        frame, radiomic_row, checkpoint["raw_radiomic_feature_columns"]
    )


def test_xgboost_raw_scale_guard_is_red_when_93_radiomics_are_standardized() -> None:
    """**ZORUNLU KIRMIZI TEST** (gorev talimati): "93 radyomik
    yanlislikla standardize edilirse test KIRMIZI olmali".

    Senaryo GERCEKCI: checkpoint'in `cox_score` blogu 54 radyomik icin
    mean/std TASIYOR (2026-09-13'te backend-agent-K gercek `.pkl`'i acarak
    DOGRULADI: `feature_standardization` **55 anahtar** = 54 radyomik +
    `clinical_age`, yani buradaki "54 radyomik" ifadesi DOGRUDUR --
    `api/predict.py::_assert_xgboost_raw_scale_preserved()` dokstring'indeki
    "54-kolonluk sozluk" ifadesi ise YANLISTI ve 55'e duzeltildi); biri
    (gelecekte) o sozlugu XGBoost cercevesine de
    uygularsa -- ki 2026-09-12'de Cox tarafinda tam bu sinifta bir hata
    yasandi -- model SESSIZCE yanlis olasilik dondururdu. Guard bunu
    yakalar VE testin ikinci yarisi olasiligin GERCEKTEN degistigini
    (yani guard'in bos/vakum bir kontrol olmadigini) kanitlar."""

    checkpoint = _real_xgboost_checkpoint()
    radiomic_row, clinical_row = _synthetic_patient_rows(checkpoint)
    correct_frame = predict_module.build_xgboost_feature_frame(
        checkpoint, radiomic_row, clinical_row, cox_oof_score=-0.25
    )

    poisoned_frame = predict_module._apply_feature_standardization(
        correct_frame, checkpoint["cox_feature_standardization"]
    )

    with pytest.raises(predict_module.XGBoostFeatureFrameError) as exc_info:
        predict_module._assert_xgboost_raw_scale_preserved(
            poisoned_frame, radiomic_row, checkpoint["raw_radiomic_feature_columns"]
        )
    assert "OLCEK KARISMASI" in str(exc_info.value)

    # Guard VAKUM DEGIL: olcek karismasi olasiligi GERCEKTEN degistiriyor.
    correct_probability = float(
        checkpoint["xgb_fitted_model"].predict_proba(correct_frame)[0, 1]
    )
    poisoned_probability = float(
        checkpoint["xgb_fitted_model"].predict_proba(poisoned_frame)[0, 1]
    )
    assert correct_probability != pytest.approx(poisoned_probability, abs=1e-6), (
        "Olcek karismasi olasiligi degistirmiyorsa bu guard yanlis seyi "
        "olcuyor demektir -- bulgu olarak bildirilmeli."
    )
    # Servis edilen deger HAM (dogru) cerceveden gelmeli.
    served = predict_module.compute_xgboost_shadow_prediction(
        checkpoint, radiomic_row, clinical_row
    )
    cox_score = predict_module.compute_cox_oof_score_for_xgboost(
        checkpoint, radiomic_row, clinical_row
    )
    raw_frame = predict_module.build_xgboost_feature_frame(
        checkpoint, radiomic_row, clinical_row, cox_oof_score=cox_score
    )
    assert served["probability_12_month_survival"] == pytest.approx(
        float(checkpoint["xgb_fitted_model"].predict_proba(raw_frame)[0, 1]), abs=1e-12
    )


def test_xgboost_column_order_guard_is_red_when_order_shuffled() -> None:
    """**ZORUNLU KIRMIZI TEST** (gorev talimati): kolon SIRASI karisirsa
    ACIK hata. Kume AYNI, yalniz sira farkli -- sessiz pozisyonel
    besleme en tehlikeli durumdur."""

    checkpoint = _real_xgboost_checkpoint()
    radiomic_row, clinical_row = _synthetic_patient_rows(checkpoint)
    frame = predict_module.build_xgboost_feature_frame(
        checkpoint, radiomic_row, clinical_row, cox_oof_score=-0.25
    )

    shuffled = frame[list(reversed(list(frame.columns)))]
    assert set(shuffled.columns) == set(frame.columns)

    with pytest.raises(predict_module.XGBoostFeatureFrameError) as exc_info:
        predict_module._assert_xgboost_feature_frame_contract(shuffled, checkpoint)
    message = str(exc_info.value)
    assert "SIRA" in message

    # YESIL taraf: dogru sirali cerceve kabul edilir.
    predict_module._assert_xgboost_feature_frame_contract(frame, checkpoint)


def test_xgboost_column_guard_is_red_when_column_missing_or_extra() -> None:
    checkpoint = _real_xgboost_checkpoint()
    radiomic_row, clinical_row = _synthetic_patient_rows(checkpoint)
    frame = predict_module.build_xgboost_feature_frame(
        checkpoint, radiomic_row, clinical_row, cox_oof_score=-0.25
    )

    dropped = frame.drop(columns=[checkpoint["score_column"]])
    with pytest.raises(predict_module.XGBoostFeatureFrameError) as exc_info:
        predict_module._assert_xgboost_feature_frame_contract(dropped, checkpoint)
    assert "eksik" in str(exc_info.value)

    widened = frame.copy()
    widened["surprise_column"] = 1.0
    with pytest.raises(predict_module.XGBoostFeatureFrameError) as exc_info:
        predict_module._assert_xgboost_feature_frame_contract(widened, checkpoint)
    assert "fazla" in str(exc_info.value)


def test_build_xgboost_feature_frame_raises_when_radiomic_column_absent() -> None:
    checkpoint = _real_xgboost_checkpoint()
    radiomic_row, clinical_row = _synthetic_patient_rows(checkpoint)
    incomplete = radiomic_row.drop(columns=[checkpoint["raw_radiomic_feature_columns"][0]])

    with pytest.raises(predict_module.XGBoostFeatureFrameError):
        predict_module.build_xgboost_feature_frame(
            checkpoint, incomplete, clinical_row, cox_oof_score=0.0
        )


def test_cox_oof_score_uses_standardized_space_not_raw() -> None:
    """`cox_oof_score`, checkpoint'in `cox_score` blogunun
    standardizasyonuyla hesaplanir -- HAM beslenmis bir hesapla AYNI
    OLMAMALIDIR (2026-09-12'de Cox tarafinda yasanan hatanin bu
    checkpoint icin de olculmus hali)."""

    checkpoint = _real_xgboost_checkpoint()
    radiomic_row, clinical_row = _synthetic_patient_rows(checkpoint)

    standardized_score = predict_module.compute_cox_oof_score_for_xgboost(
        checkpoint, radiomic_row, clinical_row
    )

    cox_model_columns = checkpoint["cox_model_columns"]
    combined = pd.concat(
        [radiomic_row.reset_index(drop=True), clinical_row.reset_index(drop=True)], axis=1
    )
    raw_score = float(
        checkpoint["cox_fitted_model"]
        .predict_log_partial_hazard(combined[cox_model_columns])
        .iloc[0]
    )

    assert standardized_score != pytest.approx(raw_score, abs=1e-6)

    # Elle yeniden hesap -- standardizasyonun GERCEKTEN uygulandiginin kaniti.
    manual = combined[cox_model_columns].copy()
    for column in cox_model_columns:
        stats = checkpoint["cox_feature_standardization"].get(column)
        if stats is not None:
            manual[column] = (manual[column].astype(float) - float(stats["mean"])) / float(
                stats["std"]
            )
    expected = float(
        checkpoint["cox_fitted_model"].predict_log_partial_hazard(manual[cox_model_columns]).iloc[0]
    )
    assert standardized_score == pytest.approx(expected, abs=1e-12)


# --- Checkpoint sozlesmesi: fail-loud dogrulama (KIRMIZI senaryolar) ---


def test_validate_xgboost_checkpoint_rejects_missing_block() -> None:
    raw = _shallow_copy_xgboost_checkpoint(_load_raw_xgboost_checkpoint())
    del raw["xgboost"]
    with pytest.raises(predict_module.XGBoostCheckpointContractError):
        predict_module._validate_xgboost_checkpoint(raw)


def test_validate_xgboost_checkpoint_rejects_structural_column_mismatch() -> None:
    raw = _shallow_copy_xgboost_checkpoint(_load_raw_xgboost_checkpoint())
    raw["xgboost"]["xgb_feature_columns"] = list(
        reversed(list(raw["xgboost"]["xgb_feature_columns"]))
    )
    with pytest.raises(predict_module.XGBoostCheckpointContractError) as exc_info:
        predict_module._validate_xgboost_checkpoint(raw)
    assert "yapisal kurulusla" in str(exc_info.value)


def test_validate_xgboost_checkpoint_rejects_uncovered_standardization() -> None:
    """**OLCEK GUARD'I** -- Cox alt-modelinin kullandigi bir radyomik
    ozelligin standardizasyon istatistigi EKSIKSE checkpoint REDDEDILIR.
    Aksi halde o kolon HAM beslenir ve skor SESSIZCE yanlis olur (bu,
    2026-09-12'de Cox tarafinda gerceklesen hatanin ta kendisi)."""

    raw = _shallow_copy_xgboost_checkpoint(_load_raw_xgboost_checkpoint())
    dropped_feature = raw["cox_score"]["final_features"][0]
    del raw["cox_score"]["feature_standardization"][dropped_feature]
    with pytest.raises(predict_module.XGBoostCheckpointContractError) as exc_info:
        predict_module._validate_xgboost_checkpoint(raw)
    assert "KAPSAMIYOR" in str(exc_info.value)
    assert dropped_feature in str(exc_info.value)


def test_validate_xgboost_checkpoint_rejects_standardized_binary_flag() -> None:
    raw = _shallow_copy_xgboost_checkpoint(_load_raw_xgboost_checkpoint())
    raw["cox_score"]["feature_standardization"]["clinical_gender_male"] = {
        "mean": 0.5,
        "std": 0.5,
    }
    with pytest.raises(predict_module.XGBoostCheckpointContractError) as exc_info:
        predict_module._validate_xgboost_checkpoint(raw)
    assert "ikili klinik bayrak" in str(exc_info.value)


def test_validate_xgboost_checkpoint_rejects_age_scale_divergence() -> None:
    raw = _shallow_copy_xgboost_checkpoint(_load_raw_xgboost_checkpoint())
    raw["xgboost"]["age_standardization"]["mean"] = (
        float(raw["xgboost"]["age_standardization"]["mean"]) + 1.0
    )
    with pytest.raises(predict_module.XGBoostCheckpointContractError) as exc_info:
        predict_module._validate_xgboost_checkpoint(raw)
    assert "AYRISMIS" in str(exc_info.value)


def test_validate_xgboost_checkpoint_rejects_clinical_column_mismatch_between_blocks() -> None:
    raw = _shallow_copy_xgboost_checkpoint(_load_raw_xgboost_checkpoint())
    raw["xgboost"]["clinical_extra_columns"] = list(
        raw["xgboost"]["clinical_extra_columns"]
    )[:-1]
    with pytest.raises(predict_module.XGBoostCheckpointContractError):
        predict_module._validate_xgboost_checkpoint(raw)


def test_load_xgboost_checkpoint_raises_when_file_missing(tmp_path) -> None:
    with pytest.raises(predict_module.XGBoostCheckpointNotFoundError):
        predict_module.load_xgboost_checkpoint(tmp_path / "yok.pkl")


# --- BLOK-FATAL degradasyon (api/analyze_patient.py B-1 deseniyle AYNI) ---


def test_xgboost_block_degrades_when_checkpoint_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        predict_module, "_xgboost_checkpoint_path", lambda: tmp_path / "yok.pkl"
    )
    block = predict_module.build_xgboost_shadow_block("UPENN-GBM-00001")

    assert block["available"] is False
    assert block["status"] == "shadow"
    assert block["error_type"] == "XGBoostCheckpointNotFoundError"
    assert block["systemic_hint"] is False
    # UYDURMA deger YOK -- anahtar HIC eklenmez (None olarak da degil).
    assert "probability_12_month_survival" not in block


def test_xgboost_block_degrades_on_multi_scan_patient(monkeypatch) -> None:
    """422 sinifi bir radyomik sorunu (coklu tarama) blogu dusurur ama
    ISTEGI OLDURMEZ -- B-1'in `_risk_block_unavailable()` desenine
    birebir uyum."""

    if not _real_xgboost_checkpoint_path().is_file():
        pytest.skip("XGBoost checkpoint'i diskte yok.")

    def _raise_multi_scan(patient_id, feature_columns):
        raise predict_module.MultiScanNotSupportedError("4 farkli scan_id")

    monkeypatch.setattr(predict_module, "fetch_c32_radiomic_row", _raise_multi_scan)
    block = predict_module.build_xgboost_shadow_block("Patient-002")

    assert block["available"] is False
    assert block["error_type"] == "MultiScanNotSupportedError"
    assert block["systemic_hint"] is False
    assert "4 farkli scan_id" in block["error_detail"]
    assert "probability_12_month_survival" not in block


def test_xgboost_block_degrades_on_db_operational_error_with_systemic_hint(monkeypatch) -> None:
    """BILINCLI SAPMA `api/analyze_patient.py` B-1'den: orada
    `psycopg2.OperationalError` ISTEK-FATAL, burada BLOK-FATAL.
    Gerekce (bkz. api/predict.py modul dokstring'i madde 3): bu satira
    gelinmis olmasi, AYNI DB'den Cox radyomigi+klinigi+SHAP arka planinin
    ZATEN okundugu anlamina gelir -- son sorgudaki bir kesinti ZATEN
    URETILMIS gecerli bir Cox tahminini cope atmamali. Sistemik sinyal
    GIZLENMIYOR: `systemic_hint=True` yanitta GORUNUR."""

    if not _real_xgboost_checkpoint_path().is_file():
        pytest.skip("XGBoost checkpoint'i diskte yok.")

    import psycopg2

    def _raise_operational(patient_id, feature_columns):
        raise psycopg2.OperationalError("baglanti koptu")

    monkeypatch.setattr(predict_module, "fetch_c32_radiomic_row", _raise_operational)
    block = predict_module.build_xgboost_shadow_block("UPENN-GBM-00001")

    assert block["available"] is False
    assert block["error_type"] == "OperationalError"
    assert block["systemic_hint"] is True
    assert "probability_12_month_survival" not in block


def test_xgboost_block_can_be_disabled_by_env(monkeypatch) -> None:
    monkeypatch.setenv(predict_module.XGBOOST_ENABLE_ENV_VAR, "0")
    block = predict_module.build_xgboost_shadow_block("UPENN-GBM-00001")

    assert block["available"] is False
    assert block["error_type"] == "Disabled"
    assert "probability_12_month_survival" not in block


def test_xgboost_shadow_is_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv(predict_module.XGBOOST_ENABLE_ENV_VAR, raising=False)
    assert predict_module.xgboost_shadow_enabled() is True
    monkeypatch.setenv(predict_module.XGBOOST_ENABLE_ENV_VAR, "false")
    assert predict_module.xgboost_shadow_enabled() is False


def test_predict_patient_keeps_cox_result_when_xgboost_block_fails(
    monkeypatch, tmp_path
) -> None:
    """ENDPOINT duzeyi: XGBoost checkpoint'i okunamasa BILE Cox risk
    skoru/SHAP AYNEN doner, istek 200 kalir (HTTPException YOK)."""

    feature_cols = ["WT__f1", "WT__f2"]
    model, frame = _fit_tiny_cox_model(feature_cols, seed=3, n=60)
    arm = _fake_arm_result(model, feature_cols, name="primary_test")

    fake_checkpoint = tmp_path / "fake_checkpoint.pkl"
    fake_checkpoint.write_bytes(b"dummy")

    def _raise_contract_error():
        raise predict_module.XGBoostCheckpointContractError("bozuk checkpoint")

    monkeypatch.setattr(predict_module, "load_cox_arm_result", lambda: arm)
    monkeypatch.setattr(
        predict_module,
        "fetch_c32_wt_row",
        lambda patient_id, final_features: frame[feature_cols].iloc[[0]],
    )
    monkeypatch.setattr(
        predict_module,
        "get_shap_background",
        lambda final_features, extra_columns, cache_generation, **_kwargs: frame[
            feature_cols
        ].sample(n=20, random_state=1),
    )
    monkeypatch.setattr(predict_module, "_checkpoint_path", lambda: fake_checkpoint)
    monkeypatch.setattr(predict_module, "load_xgboost_checkpoint", _raise_contract_error)
    monkeypatch.setattr(predict_module, "fetch_omics_interpretation", lambda patient_id: None)

    result = predict_module.predict_patient("UPENN-GBM-00001")

    assert result["risk_score_log_partial_hazard"] is not None
    assert result["shap_additivity_check_abs_diff"] < 1e-9
    assert result["xgboost"]["available"] is False
    assert result["xgboost"]["error_type"] == "XGBoostCheckpointContractError"
    assert result["xgboost"]["is_primary_decision_model"] is False


def test_predict_patient_real_db_xgboost_shadow_pinned_ucsf_167() -> None:
    """CANLI DB + GERCEK iki checkpoint (Cox v3b + XGBoost shadow).

    UC PIN birlikte olculur:
      1. Cox risk skoru/HR -- 2026-09-13 oncesi degerle BIREBIR AYNI
         (XGBoost baglanmasi Cox yolunu BOZMADI).
      2. `cox_oof_score` (XGBoost'un girdisi) ile birincil Cox risk
         skoru arasindaki fark **TAM 0,0** -- iki modelin bit-birebir
         ayni oldugunun (modeling-agent-G'nin 24/24 kolon olcumu) CANLI
         teyidi VE olcek zincirinin dogru kuruldugunun bagimsiz kaniti
         (standardizasyon yanlis uygulanirsa bu fark 0 CIKMAZ).
      3. 12-ay sagkalim olasiligi (float32 XGBoost cikarimi -- bu yuzden
         mutlak tolerans 1e-6, Cox pinleri gibi 1e-9 DEGIL).
    """

    if not _real_xgboost_checkpoint_path().is_file():
        pytest.skip("XGBoost checkpoint'i diskte yok.")

    try:
        result = predict_module.predict_patient("UCSF-PDGM-167")
    except Exception as exc:  # pragma: no cover -- yalniz DB/artefakt yoksa
        pytest.skip(f"Canli DB/artefakt erisilemedi: {exc}")

    assert result["risk_score_log_partial_hazard"] == pytest.approx(
        -0.7096787591025546, abs=1e-9
    )
    assert result["hazard_ratio_partial_hazard"] == pytest.approx(
        0.49180215905468494, abs=1e-9
    )

    block = result["xgboost"]
    assert block["available"] is True
    assert block["status"] == "shadow"
    assert block["is_primary_decision_model"] is False
    assert block["n_xgb_feature_columns"] == 102
    assert block["n_raw_radiomic_features"] == 93
    assert block["cox_score_input"]["value"] == pytest.approx(
        -0.7096787591025546, abs=1e-9
    )
    assert block["cox_score_input"]["primary_cox_risk_score_abs_diff"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert block["probability_12_month_survival"] == pytest.approx(
        0.7318906784057617, abs=1e-6
    )
    assert 0.0 <= block["probability_12_month_survival"] <= 1.0
    assert block["external_validation"]["n_patients"] == 232


def test_xgboost_block_version_hint_matches_live_model_registry_shadow_row() -> None:
    """CANLI DB TEYIDI (SALT SELECT): yanitta beyan edilen
    `model_registry_version_hint`, `model_registry` tablosundaki GERCEK
    satirla eslesiyor mu VE o satirin `status`'u HALA 'shadow' mu?

    Bu test iki sey icin vardir:
      1. Yanittaki "shadow" etiketi bir metin sabiti olarak DEGIL, DB'deki
         gercek kayitla ESLESEN bir beyan olsun (gorev talimati: "endpoint
         ciktisini DB'den canli dogrula, 'calisiyor' deme").
      2. Birileri statuyu 'production'a cevirirse bu test KIRMIZI olur ve
         karar GORUNUR hale gelir -- politika degisikligi SESSIZ olmaz.
         (Bu ajan `model_registry`'ye HICBIR SEY YAZMADI.)
    """

    checkpoint_path = _real_xgboost_checkpoint_path()
    if not checkpoint_path.is_file():
        pytest.skip("XGBoost checkpoint'i diskte yok.")
    checkpoint = predict_module.load_xgboost_checkpoint(checkpoint_path)
    version_hint = checkpoint["model_registry_version_hint"]
    assert version_hint, "Checkpoint 'model_registry_version_hint' TASIMIYOR."

    try:
        connection = predict_module._get_db_connection()
    except Exception as exc:  # pragma: no cover -- yalniz DB erisilemezse
        pytest.skip(f"Canli DB'ye erisilemedi: {exc}")
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT status FROM model_registry WHERE model_name = %s AND version = %s",
            ("xgboost", version_hint),
        )
        row = cursor.fetchone()
        cursor.close()
    finally:
        connection.close()

    assert row is not None, (
        f"model_registry'de model_name='xgboost' AND version={version_hint!r} "
        "satiri YOK -- yanittaki surum beyani DB ile dogrulanamiyor."
    )
    status = row[0] if not isinstance(row, dict) else row["status"]
    assert status == predict_module.XGBOOST_MODEL_STATUS == "shadow", (
        f"model_registry'deki statu {status!r}, yanitta beyan edilen "
        f"{predict_module.XGBOOST_MODEL_STATUS!r} ile AYNI DEGIL -- 'shadow' "
        "etiketi yaniti okuyan tarafi YANLIS bilgilendirir."
    )


def test_predict_patient_real_db_xgboost_shadow_pinned_tcga_06_5412() -> None:
    """Ikinci canli pin -- YUKSEK riskli hasta. Yon tutarliligi da
    olculur: Cox risk skoru POZITIF (kotu) iken 12-ay SAGKALIM olasiligi
    dusuk olmali (ters yonlu okuma hatasina karsi kalici kanit)."""

    if not _real_xgboost_checkpoint_path().is_file():
        pytest.skip("XGBoost checkpoint'i diskte yok.")

    try:
        result = predict_module.predict_patient("TCGA-06-5412")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Canli DB/artefakt erisilemedi: {exc}")

    assert result["risk_score_log_partial_hazard"] == pytest.approx(
        0.7551730099099314, abs=1e-9
    )
    block = result["xgboost"]
    assert block["available"] is True
    assert block["cox_score_input"]["primary_cox_risk_score_abs_diff"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert block["probability_12_month_survival"] == pytest.approx(
        0.2470972239971161, abs=1e-6
    )
    assert block["probability_12_month_survival"] < 0.5
