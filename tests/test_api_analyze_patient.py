"""``POST /analyze_patient`` -- birim testleri (api/analyze_patient.py).

DB'ye BAGLANMAZ -- her test `api.analyze_patient` icinden cagirilan UC
alt-modulu (`api.predict`, `api.similar`, `pipeline.rag_pipeline`)
monkeypatch ile sahteler. Amac: bu dosyanin KENDI mantigini (zincirleme +
graceful degradation + JSON birlestirme) test etmek -- alt modullerin
kendi ic mantigi zaten `tests/test_api_predict.py` / `tests/test_api_
similar.py` / `tests/test_rag_pipeline.py`'de ayrica test edildi, burada
TEKRAR EDILMEZ.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import pytest
from fastapi import HTTPException

import api.analyze_patient as analyze_module
import pipeline.growth_simulation as growth_simulation_pipeline
from pipeline.rag_pipeline import RagPipelineResult
from pipeline.rag_query import PubMedQueryResult


# =====================================================================
# growth_simulation mock altyapisi -- gercek DB'ye BAGLANMAZ (predict/
# similar/rag ile AYNI test felsefesi). `_build_growth_simulation_block()`
# artik canli DB'ye baglaniyor (2026-09-13) -- burada `db_connection_
# module.get_connection` VE `growth_simulation_module.fetch_lumiere_*`
# DOGRUDAN sahtelenir, gercek psycopg2 baglantisi ASLA acilmaz.
# =====================================================================


_SERIES_COLUMNS = [
    "patient_id", "scan_id", "timepoint_label", "tumor_volume_mm3",
    "surface_area", "entropy", "sphericity", "glcm_joint_entropy",
    "week", "suffix", "x_week",
]
_RANO_COLUMNS = [
    "followup_id", "patient_id", "visit_week", "rano_label",
    "rano_rationale", "rano_label_raw",
]


class _FakeConnection:
    """`db_connection_module.get_connection()`'in sahte donusu -- `.close()`
    disinda hicbir sey yapmaz, gercek psycopg2 baglantisi ASLA acilmaz."""

    def close(self) -> None:
        pass


def _empty_series_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_SERIES_COLUMNS)


def _empty_rano_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_RANO_COLUMNS)


def _build_synthetic_lumiere_series(
    n_patients: int = 7, weeks: list[int] | None = None
) -> pd.DataFrame:
    """7 sentetik LUMIERE hastasi icin lojistik-sekilli, GERCEKTEN
    yakinsayan bir WT hacim serisi uretir (canli DB'ye BAGLANMAZ, ama
    `pipeline/growth_simulation.py`'nin GERCEK `logistic()`/`curve_fit`
    mantigiyla dogrulanmis -- bkz. bu dosyanin gorev-onu manuel
    dogrulamasi). Weeks varsayilani [0,4,8,12,16], deterministik seed=7
    gurultu.

    `weeks` 2026-09-13'te (G1) eklendi -- VARSAYILANI DEGISMEDI, yani
    mevcut testlerin sayisal beklentileri bu parametreden ETKILENMEZ.
    Amaci: son ziyareti 26,07 haftadan SONRA olan (gercek LUMIERE'de
    tipik) bir seri kurup projeksiyon ufkunun geriye donuk olmadigini
    test edebilmek."""

    rows: list[dict[str, Any]] = []
    weeks = [0, 4, 8, 12, 16] if weeks is None else list(weeks)
    rng = np.random.default_rng(7)
    patient_ids = [f"Patient-{900 + i}" for i in range(1, n_patients + 1)]
    for idx, pid in enumerate(patient_ids):
        k = 4000 + idx * 200
        r = 0.12 + idx * 0.01
        t0 = 8.0
        for w in weeks:
            vol = float(growth_simulation_pipeline.logistic(np.array([float(w)]), k, r, t0)[0])
            vol = vol * (1 + rng.normal(0, 0.01))
            rows.append(
                {
                    "patient_id": pid,
                    "scan_id": f"{pid}-{w}",
                    "timepoint_label": f"week-{w:03d}",
                    "tumor_volume_mm3": vol,
                    "surface_area": vol * 0.5,
                    "entropy": 3.0,
                    "sphericity": 0.7,
                    "glcm_joint_entropy": 5.0,
                    "week": w,
                    "suffix": 0,
                    "x_week": float(w),
                }
            )
    return pd.DataFrame(rows)


def _build_synthetic_rano_df(patient_ids: list[str], *, rano_label: str = "PD") -> pd.DataFrame:
    rows = [
        {
            "followup_id": i,
            "patient_id": pid,
            "visit_week": 16,
            "rano_label": rano_label,
            "rano_rationale": None,
        }
        for i, pid in enumerate(patient_ids)
    ]
    df = pd.DataFrame(rows)
    df["rano_label_raw"] = df["rano_label"]
    return df


def _patch_growth_simulation_db(
    monkeypatch, *, series_df: pd.DataFrame, rano_df: pd.DataFrame
) -> None:
    monkeypatch.setattr(
        analyze_module.db_connection_module,
        "get_connection",
        lambda readonly=True: _FakeConnection(),
    )
    monkeypatch.setattr(
        analyze_module.growth_simulation_module,
        "fetch_lumiere_wt_volume_series",
        lambda conn: series_df,
    )
    monkeypatch.setattr(
        analyze_module.growth_simulation_module,
        "fetch_lumiere_rano_labels",
        lambda conn: rano_df,
    )


def _patch_growth_simulation_not_in_lumiere(monkeypatch) -> None:
    """Varsayilan: hasta LUMIERE serisinde YOK -- diger blokları test eden
    ESKI testler bu varsayilanla degismeden gecer (autouse fixture)."""

    _patch_growth_simulation_db(monkeypatch, series_df=_empty_series_df(), rano_df=_empty_rano_df())


@pytest.fixture(autouse=True)
def _default_growth_simulation_not_available(monkeypatch) -> None:
    """Growth-simulation testleri DISINDAKI HER test icin varsayilan --
    gercek DB'ye ASLA baglanmaz, hasta LUMIERE serisinde bulunamadigi icin
    `available: False` doner. Growth-simulation'a OZEL testler bunu KENDI
    icinde YENIDEN patch'ler (bkz. asagidaki testler)."""

    _patch_growth_simulation_not_in_lumiere(monkeypatch)


# =====================================================================
# XGBoost (shadow) ikinci katman -- 2026-09-13 (backend-agent-J).
# `api.predict.predict_patient()` yanitinin `xgboost` alani ARTIK
# `/analyze_patient` ciktisina `xgboost_12mo_survival` adiyla TASINIYOR.
# Asagidaki sahte blok, `api/predict.py::build_xgboost_shadow_block()`'in
# GERCEK alan adlarini birebir taklit eder (backend-agent-G'nin
# 2026-09-13 canli olcumunden alinan sayilarla) -- ama bu dosya DB'ye
# BAGLANMAZ, degerler sahtedir/sabittir.
# =====================================================================

_FAKE_XGB_PROBABILITY = 0.7318906784057617
_FAKE_XGB_COX_OOF_SCORE = -0.7096787591025546


def _fake_xgboost_block(*, available: bool = True) -> dict[str, Any]:
    if not available:
        # `api/predict.py::_xgboost_block_unavailable()` sozlesmesi --
        # `probability_12_month_survival` anahtari HIC YOK.
        return {
            "available": False,
            "status": "shadow",
            "is_primary_decision_model": False,
            "error_type": "XGBoostCheckpointNotFoundError",
            "error_detail": "XGBoost (shadow) checkpoint'i bulunamadi (test).",
            "systemic_hint": False,
            "not_available_reason": (
                "XGBoost (shadow) ikinci katman olasiligi URETILEMEDI: "
                "XGBoost (shadow) checkpoint'i bulunamadi (test)."
            ),
            "status_note": "GOLGE (shadow) modundadir (test).",
        }
    return {
        "available": True,
        "status": "shadow",
        "is_primary_decision_model": False,
        "status_note": "GOLGE (shadow) modundadir (test).",
        "model_arm": "v2a_mgmt+reduce_collinearity-aligned-v3b_bitidentical",
        "model_registry_version_hint": (
            "v2a_mgmt+reduce_collinearity-aligned-v3b_bitidentical_2026-09-12"
        ),
        "probability_12_month_survival": _FAKE_XGB_PROBABILITY,
        "direction_note": (
            "`probability_12_month_survival` ... YUKSEK deger IYI prognozdur. "
            "DIKKAT: Cox `risk_score_log_partial_hazard` ile TERS yonludur."
        ),
        "target_definition_note": "Hedef `target_12mo_survival` (test).",
        "twelve_month_threshold_days": 365.0,
        "cox_score_input": {
            "score_column": "cox_oof_score",
            "value": _FAKE_XGB_COX_OOF_SCORE,
        },
        "external_validation": {"auc": 0.7237294917967186, "n_patients": 232},
    }


def _fake_cox_result(
    *, with_omics: bool = False, xgboost_block: dict[str, Any] | None | str = "default"
) -> dict[str, Any]:
    """`xgboost_block`: `"default"` -> mutlu-yol sahte blok; `None` ->
    `xgboost` anahtari yanitta HIC OLMAZ (sozlesme ihlali dali); sozluk ->
    aynen kullanilir."""

    result: dict[str, Any] = {
        "patient_id": "TEST-001",
        "model_arm": "test_arm",
        "final_features": ["WT__f1", "WT__f2"],
        "clinical_extra_columns": ["clinical_age"],
        "risk_score_log_partial_hazard": 0.42,
        "hazard_ratio_partial_hazard": 1.52,
        "shap_base_value": 0.1,
        "shap_values": {"WT__f1": 0.2, "WT__f2": 0.12, "clinical_age": 0.0},
        "shap_additivity_check_abs_diff": 1e-12,
    }
    if xgboost_block == "default":
        result["xgboost"] = _fake_xgboost_block()
    elif isinstance(xgboost_block, dict):
        result["xgboost"] = xgboost_block
    if with_omics:
        result["omics"] = {"has_omics": True, "available": True, "tmz_resistance_score": 33.6}
    return result


def _fake_similar_result() -> dict[str, Any]:
    return {
        "patient_id": "TEST-001",
        "clinical_radiomics_faiss": {
            "index_size": 722,
            "k_requested": 10,
            "k_returned": 10,
            "filters_applied": {},
            "warnings": [],
            "results": [{"patient_id": "N1", "l2_distance": 12.3}],
        },
    }


def _fake_rag_result(*, available: bool = True) -> RagPipelineResult:
    if not available:
        return RagPipelineResult(available=False, reason="PubMed erisilemedi/sonuc yok: test")
    query = PubMedQueryResult(
        base_query="glioblastoma",
        expansion_terms=("MGMT methylated",),
        full_query="glioblastoma MGMT methylated",
        pubmed_boolean_query="glioblastoma[Title/Abstract]",
        semantic_query_text="glioblastoma MGMT methylated",
        mgmt_bucket="methylated",
        idh1_bucket="wildtype",
        conditions=(),
        unevaluated_notes=(),
    )
    return RagPipelineResult(
        available=True,
        reason=None,
        query=query,
        retrieved_chunks=(),
        sanitized_context=None,
        llm_summary_available=False,
        llm_reason="LLM yapilandirilmadi (test)",
        notes=("test-note",),
        used_cache=False,
        sanitization_violation=False,
    )


def _patch_happy_path(monkeypatch, *, with_omics: bool = False) -> None:
    monkeypatch.setattr(
        analyze_module.predict_module,
        "predict_patient",
        lambda patient_id: _fake_cox_result(with_omics=with_omics),
    )
    monkeypatch.setattr(
        analyze_module.similar_module,
        "get_similar_patients",
        lambda patient_id, k, idh1_status, mgmt_status: _fake_similar_result(),
    )
    monkeypatch.setattr(
        analyze_module.rag_pipeline_module,
        "generate_literature_summary",
        lambda patient_id: _fake_rag_result(),
    )


# =====================================================================
# 1) Mutlu yol -- tum bloklar birlesiyor
# =====================================================================


def test_analyze_patient_happy_path_merges_all_blocks(monkeypatch) -> None:
    _patch_happy_path(monkeypatch, with_omics=True)

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="TEST-001")
    )

    assert result["patient_id"] == "TEST-001"

    risk = result["risk"]
    assert risk["available"] is True
    assert risk["shap_available"] is True
    assert risk["omics_available"] is True
    assert risk["risk_score_log_partial_hazard"] == 0.42
    assert risk["hazard_ratio_partial_hazard"] == 1.52
    assert risk["shap_values"] == {"WT__f1": 0.2, "WT__f2": 0.12, "clinical_age": 0.0}
    assert "risk_score_interpretation_note" in risk
    assert "OLASILIK" in risk["risk_score_interpretation_note"]
    assert risk["omics"]["tmz_resistance_score"] == 33.6

    assert result["similar_patients"]["available"] is True
    assert result["similar_patients"]["clinical_radiomics_faiss"]["index_size"] == 722

    lit = result["literature"]
    assert lit["available"] is True
    assert lit["llm_summary_available"] is False
    assert lit["query"]["pubmed_boolean_query"] == "glioblastoma[Title/Abstract]"

    # growth_simulation artik CANLI hesaplanir (autouse fixture: hasta
    # LUMIERE serisinde YOK -> available:False, ACIK sebep) -- ayrintili
    # dallar asagidaki OZEL testlerde kanitlanir.
    assert result["growth_simulation"]["available"] is False
    assert "TEST-001" in result["growth_simulation"]["not_available_reason"]
    assert "LUMIERE" in result["growth_simulation"]["not_available_reason"]
    # 2026-09-13 (backend-agent-J) GUNCELLENDI -- eski hali:
    #     assert result["xgboost_recurrence"] == {
    #         "available": False,
    #         "not_available_reason": analyze_module.XGBOOST_NOT_AVAILABLE_REASON,
    #     }
    # O assert, ARTIK YANLIS olan iki seyi kilitliyordu: (1) blogun her zaman
    # `available:false` sabiti dondugunu (XGBoost 2026-09-13 ADIM 2'de canli
    # `/predict` zincirine BAGLANDI), (2) `xgboost_recurrence` adini
    # ("recurrence" = nuks, oysa egitilen hedef 12-ay SAGKALIM sinifi).
    # Test SILINMEDI, gercege gore yeniden yazildi (CLAUDE.md "hata
    # duzeltmek post-hoc degildir").
    xgb = result[analyze_module.XGBOOST_BLOCK_KEY]
    assert analyze_module.XGBOOST_BLOCK_KEY == "xgboost_12mo_survival"
    assert xgb["available"] is True
    assert xgb["probability_12_month_survival"] == _FAKE_XGB_PROBABILITY
    assert xgb["status"] == "shadow"
    assert xgb["is_primary_decision_model"] is False
    assert "TERS" in xgb["direction_note"]
    # Eski/yanlis ad UST SEVIYEDE YOK (temiz kesme) ama blogun ICINDE
    # gorunur bir iz olarak duruyor.
    assert "xgboost_recurrence" not in result
    assert xgb["previous_block_name"] == "xgboost_recurrence"


def test_analyze_patient_omits_omics_key_when_absent(monkeypatch) -> None:
    _patch_happy_path(monkeypatch, with_omics=False)

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="TEST-001")
    )

    assert "omics" not in result["risk"]


# =====================================================================
# 2) Cox adimi -- 2026-09-13 DUZELTME (B-1 mimari kusuru, Baris onayi):
#    SADECE hasta HIC YOK (404) veya DB'ye HIC ULASILAMIYOR (psycopg2.
#    OperationalError -> 503) ISTEK-FATAL'dir (yukari yayilir). Digger
#    TUM durumlar (422/501/500/503-checkpoint, beklenmedik Exception)
#    BLOK-FATAL'dir -- `risk` bloğu `available:False` doner, istek 200
#    devam eder, DIGER bloklar BAGIMSIZ uretilir. Eski testler (422/503
#    icin `pytest.raises(HTTPException)` bekleyen) TAM DA duzeltilen
#    hatayi test ediyordu -- asagida YENI davranisa gore GUNCELLENDI
#    (silinmedi, gerekcesiyle degistirildi -- CLAUDE.md "hata duzeltmek
#    post-hoc degildir" ilkesi).
# =====================================================================


def test_analyze_patient_propagates_404_when_patient_not_found(monkeypatch) -> None:
    def _raise(patient_id: str):
        raise HTTPException(status_code=404, detail=f"Hasta bulunamadi: {patient_id}")

    monkeypatch.setattr(analyze_module.predict_module, "predict_patient", _raise)

    with pytest.raises(HTTPException) as exc_info:
        analyze_module.analyze_patient(analyze_module.AnalyzePatientRequest(patient_id="NOPE"))
    assert exc_info.value.status_code == 404


def test_analyze_patient_propagates_503_when_db_unreachable(monkeypatch) -> None:
    """DB'ye HIC ULASILAMIYOR (`psycopg2.OperationalError`) -- ISTEK-FATAL,
    503'e cevrilip yukari yayilir (FAISS/buyume-simulasyonu da AYNI DB'yi
    okudugu icin kismi bir '200' yaniltici olurdu)."""

    def _raise(patient_id: str):
        raise psycopg2.OperationalError("connection timed out (test)")

    monkeypatch.setattr(analyze_module.predict_module, "predict_patient", _raise)

    with pytest.raises(HTTPException) as exc_info:
        analyze_module.analyze_patient(analyze_module.AnalyzePatientRequest(patient_id="X"))
    assert exc_info.value.status_code == 503
    assert "OperationalError" in exc_info.value.detail


def test_analyze_patient_risk_block_degrades_on_422_multi_scan(monkeypatch) -> None:
    """KIRMIZI/YESIL kaniti (B-1) -- `Patient-002` gibi coklu-tarama
    (LUMIERE) hastalarda Cox 422 dondugunde ARTIK istek COMPLE dusmez:
    `risk` bloğu `available:False` olur ama FAISS/RAG/buyume-simulasyonu
    BAGIMSIZ olarak uretilmeye devam eder (200 doner)."""

    def _raise(patient_id: str):
        raise HTTPException(status_code=422, detail="Coklu-tarama, C32 radyomigi belirsiz")

    monkeypatch.setattr(analyze_module.predict_module, "predict_patient", _raise)
    monkeypatch.setattr(
        analyze_module.similar_module,
        "get_similar_patients",
        lambda pid, k, idh1_status, mgmt_status: _fake_similar_result(),
    )
    monkeypatch.setattr(
        analyze_module.rag_pipeline_module,
        "generate_literature_summary",
        lambda pid: _fake_rag_result(),
    )

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="Patient-002")
    )

    risk = result["risk"]
    assert risk["available"] is False
    assert risk["error_status_code"] == 422
    assert risk["shap_available"] is False
    assert risk["omics_available"] is False
    assert "Coklu-tarama" in risk["error_detail"]

    # Diger bloklar ETKILENMEDI -- pipeline DURMADI.
    assert result["similar_patients"]["available"] is True
    assert result["literature"]["available"] is True

    summary = result["summary"]
    assert summary["all_blocks_available"] is False
    unavailable_names = {b["block"] for b in summary["unavailable_blocks"]}
    assert "risk" in unavailable_names
    assert "growth_simulation" in unavailable_names  # autouse: LUMIERE'de yok


def test_analyze_patient_risk_block_degrades_on_503_checkpoint_missing(monkeypatch) -> None:
    """Model artefakti diskte yok (`CheckpointNotFoundError` -> 503) --
    bu DB-erisim sorunuyla KARISTIRILMAZ, BLOK-FATAL'dir (checkpoint disk
    dosyasidir, DB'den BAGIMSIZ)."""

    def _raise(patient_id: str):
        raise HTTPException(status_code=503, detail="Model artefakti bulunamadi")

    monkeypatch.setattr(analyze_module.predict_module, "predict_patient", _raise)
    monkeypatch.setattr(
        analyze_module.similar_module,
        "get_similar_patients",
        lambda pid, k, idh1_status, mgmt_status: _fake_similar_result(),
    )
    monkeypatch.setattr(
        analyze_module.rag_pipeline_module,
        "generate_literature_summary",
        lambda pid: _fake_rag_result(),
    )

    result = analyze_module.analyze_patient(analyze_module.AnalyzePatientRequest(patient_id="X"))

    risk = result["risk"]
    assert risk["available"] is False
    assert risk["error_status_code"] == 503
    assert "artefakti" in risk["error_detail"]
    assert result["similar_patients"]["available"] is True


def test_analyze_patient_risk_block_degrades_on_unexpected_exception(monkeypatch) -> None:
    def _raise(patient_id: str):
        raise RuntimeError("beklenmeyen bir Cox hatasi")

    monkeypatch.setattr(analyze_module.predict_module, "predict_patient", _raise)
    monkeypatch.setattr(
        analyze_module.similar_module,
        "get_similar_patients",
        lambda pid, k, idh1_status, mgmt_status: _fake_similar_result(),
    )
    monkeypatch.setattr(
        analyze_module.rag_pipeline_module,
        "generate_literature_summary",
        lambda pid: _fake_rag_result(),
    )

    result = analyze_module.analyze_patient(analyze_module.AnalyzePatientRequest(patient_id="X"))

    risk = result["risk"]
    assert risk["available"] is False
    assert risk["error_status_code"] is None
    assert "RuntimeError" in risk["error_detail"]
    assert result["similar_patients"]["available"] is True


# =====================================================================
# 3) FAISS benzer-hasta -- OPSIYONEL, graceful degradation
# =====================================================================


def test_analyze_patient_similar_block_degrades_on_422_ucsf(monkeypatch) -> None:
    monkeypatch.setattr(
        analyze_module.predict_module, "predict_patient", lambda pid: _fake_cox_result()
    )
    monkeypatch.setattr(
        analyze_module.rag_pipeline_module,
        "generate_literature_summary",
        lambda pid: _fake_rag_result(),
    )

    def _raise(patient_id, k, idh1_status, mgmt_status):
        raise HTTPException(
            status_code=422,
            detail=f"{patient_id!r} clinical_radiomics_faiss v1 indeksinde YOK (UCSF).",
        )

    monkeypatch.setattr(analyze_module.similar_module, "get_similar_patients", _raise)

    # KRITIK: pipeline DURMAZ -- 422 whole-endpoint exception OLARAK
    # FIRLATILMAZ, blok-seviyesinde gorunur kalir.
    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="UCSF-PDGM-190")
    )

    assert result["similar_patients"]["available"] is False
    assert result["similar_patients"]["error_status_code"] == 422
    assert "UCSF" in result["similar_patients"]["error_detail"]
    # Risk bloğu ETKİLENMEDİ.
    assert result["risk"]["risk_score_log_partial_hazard"] == 0.42


def test_analyze_patient_similar_block_degrades_on_503_missing_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(
        analyze_module.predict_module, "predict_patient", lambda pid: _fake_cox_result()
    )
    monkeypatch.setattr(
        analyze_module.rag_pipeline_module,
        "generate_literature_summary",
        lambda pid: _fake_rag_result(),
    )

    def _raise(patient_id, k, idh1_status, mgmt_status):
        raise HTTPException(status_code=503, detail="FAISS artefaktlari eksik")

    monkeypatch.setattr(analyze_module.similar_module, "get_similar_patients", _raise)

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="X")
    )
    assert result["similar_patients"]["available"] is False
    assert result["similar_patients"]["error_status_code"] == 503


def test_analyze_patient_similar_block_degrades_on_unexpected_exception(monkeypatch) -> None:
    monkeypatch.setattr(
        analyze_module.predict_module, "predict_patient", lambda pid: _fake_cox_result()
    )
    monkeypatch.setattr(
        analyze_module.rag_pipeline_module,
        "generate_literature_summary",
        lambda pid: _fake_rag_result(),
    )

    def _raise(patient_id, k, idh1_status, mgmt_status):
        raise RuntimeError("beklenmeyen bir hata")

    monkeypatch.setattr(analyze_module.similar_module, "get_similar_patients", _raise)

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="X")
    )
    assert result["similar_patients"]["available"] is False
    assert result["similar_patients"]["error_status_code"] is None
    assert "RuntimeError" in result["similar_patients"]["error_detail"]


# =====================================================================
# 4) RAG literatur -- OPSIYONEL, graceful degradation (sozlesme "asla
#    firlatmaz" dese de savunmaci try/except test edilir)
# =====================================================================


def test_analyze_patient_literature_block_not_available(monkeypatch) -> None:
    monkeypatch.setattr(
        analyze_module.predict_module, "predict_patient", lambda pid: _fake_cox_result()
    )
    monkeypatch.setattr(
        analyze_module.similar_module,
        "get_similar_patients",
        lambda pid, k, idh1_status, mgmt_status: _fake_similar_result(),
    )
    monkeypatch.setattr(
        analyze_module.rag_pipeline_module,
        "generate_literature_summary",
        lambda pid: _fake_rag_result(available=False),
    )

    result = analyze_module.analyze_patient(analyze_module.AnalyzePatientRequest(patient_id="X"))
    assert result["literature"]["available"] is False
    assert "PubMed" in result["literature"]["reason"]


def test_analyze_patient_literature_block_degrades_on_unexpected_exception(monkeypatch) -> None:
    monkeypatch.setattr(
        analyze_module.predict_module, "predict_patient", lambda pid: _fake_cox_result()
    )
    monkeypatch.setattr(
        analyze_module.similar_module,
        "get_similar_patients",
        lambda pid, k, idh1_status, mgmt_status: _fake_similar_result(),
    )

    def _raise(patient_id: str):
        raise RuntimeError("rag pipeline sozlesmesine ragmen beklenmedik hata")

    monkeypatch.setattr(analyze_module.rag_pipeline_module, "generate_literature_summary", _raise)

    result = analyze_module.analyze_patient(analyze_module.AnalyzePatientRequest(patient_id="X"))
    assert result["literature"]["available"] is False
    assert "RuntimeError" in result["literature"]["reason"]
    # Diger bloklar ETKILENMEDI.
    assert result["similar_patients"]["available"] is True
    assert result["risk"]["risk_score_log_partial_hazard"] == 0.42


# =====================================================================
# 5) Buyume simulasyonu -- 2026-09-13 canli zincire baglandi (OPSIYONEL,
#    graceful degradation). Gercek DB'ye baglanmaz -- `db_connection_
#    module.get_connection` + `growth_simulation_module.fetch_lumiere_*`
#    sahtelenir (predict/similar/rag ile AYNI test felsefesi).
# =====================================================================


def _patch_happy_path_without_growth_simulation_default(monkeypatch) -> None:
    """Cox/similar/rag icin mutlu-yol sahteleri -- growth_simulation KENDI
    testinde AYRICA patch'lenir (autouse varsayilanini override eder)."""

    monkeypatch.setattr(
        analyze_module.predict_module, "predict_patient", lambda pid: _fake_cox_result()
    )
    monkeypatch.setattr(
        analyze_module.similar_module,
        "get_similar_patients",
        lambda pid, k, idh1_status, mgmt_status: _fake_similar_result(),
    )
    monkeypatch.setattr(
        analyze_module.rag_pipeline_module,
        "generate_literature_summary",
        lambda pid: _fake_rag_result(),
    )


def test_analyze_patient_growth_simulation_not_available_when_patient_not_in_lumiere(
    monkeypatch,
) -> None:
    """KIRMIZI/YESIL kaniti -- demo hastalari `UCSF-PDGM-167` ve
    `TCGA-06-5412` LUMIERE kohortunda DEGIL (canli DB'de 2026-09-13'te
    dogrulandi: ikisi de 0 satir). Bu test o durumu (bos seri) sahte DB
    ile REPRODUCE eder -- `available: False` + patient_id/LUMIERE gecen
    ACIK bir sebep bekleniyor, SESSIZ bos sonuc/uydurma projeksiyon YOK."""

    _patch_happy_path_without_growth_simulation_default(monkeypatch)
    _patch_growth_simulation_not_in_lumiere(monkeypatch)

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="UCSF-PDGM-167")
    )

    growth = result["growth_simulation"]
    assert growth["available"] is False
    assert "UCSF-PDGM-167" in growth["not_available_reason"]
    assert "LUMIERE" in growth["not_available_reason"]
    # Risk/similar/literature bloklari ETKILENMEDI.
    assert result["risk"]["risk_score_log_partial_hazard"] == 0.42


def test_analyze_patient_growth_simulation_not_available_insufficient_visits(
    monkeypatch,
) -> None:
    """Hasta LUMIERE serisinde VAR ama < 3 ziyaretle (`min_visits=3`
    varsayilani) fit hic DENENMEZ -- `fit_status='insufficient_visits'`,
    ACIK sebep + n_visits doner."""

    _patch_happy_path_without_growth_simulation_default(monkeypatch)

    series_df = pd.DataFrame(
        [
            {
                "patient_id": "Patient-999", "scan_id": "Patient-999-0",
                "timepoint_label": "week-000", "tumor_volume_mm3": 1000.0,
                "surface_area": 500.0, "entropy": 3.0, "sphericity": 0.7,
                "glcm_joint_entropy": 5.0, "week": 0, "suffix": 0, "x_week": 0.0,
            },
            {
                "patient_id": "Patient-999", "scan_id": "Patient-999-4",
                "timepoint_label": "week-004", "tumor_volume_mm3": 1200.0,
                "surface_area": 600.0, "entropy": 3.0, "sphericity": 0.7,
                "glcm_joint_entropy": 5.0, "week": 4, "suffix": 0, "x_week": 4.0,
            },
        ]
    )
    _patch_growth_simulation_db(monkeypatch, series_df=series_df, rano_df=_empty_rano_df())

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="Patient-999")
    )

    growth = result["growth_simulation"]
    assert growth["available"] is False
    assert growth["fit_status"] == "insufficient_visits"
    assert growth["n_visits"] == 2
    assert "fit edilemedi" in growth["not_available_reason"]


def test_analyze_patient_growth_simulation_available_with_full_projection(monkeypatch) -> None:
    """KIRMIZI/YESIL kaniti -- 7 sentetik LUMIERE hastasi (5 ziyaret, PD
    RANO grubu, n=7>=6 -> interpretable=True) icin `Patient-901` GERCEK
    degerlerle (`available: True`, fit + 6-aylik projeksiyon araligi)
    doner. Sayisal degerler bu testin YAZILMASINDAN once gercek
    `pipeline/growth_simulation.py` fonksiyonlariyla manuel dogrulandi
    (bkz. gorev-onu Bash dogrulamasi, 2026-09-13).

    GUNCELLEME 2026-09-13 (G1 projeksiyon-ufku duzeltmesi) -- eski
    beklentiler SILINMEDI, asagida kayitli: bu test ilk yazildiginda
    `v_median == 5138.52` / `pct_median == 78.46` bekliyordu. O degerler
    HATALI ufukla (`t_target = 6*4,345 = 26,07` hafta, hastanin ILK
    taramasindan itibaren MUTLAK) uretilmisti; oysa `v0` SON ziyaretin
    hacmi -- ufuk ile referans AYNI zaman eksenine oturmuyordu. Gercek
    LUMIERE hastalarinda son ziyaret 26,07'den SONRA oldugu icin
    karsilastirma GERIYE DONUK hale geliyordu (olculen etki
    `Patient-028`: alt uc %+32,96 -> %-0,47, ISARET DEGISIYOR).
    Duzeltmeden sonra `from_week = 16,0` -> `t_target_week = 42,07` ve
    beklentiler `v_median == 5300,8868…` / `pct_median == 84,10…` olarak
    GUNCELLENDI. Her iki deger de gercek kodla olculdu, uydurulmadi.
    Bkz. `decisions/2026-09-13-projeksiyon-ufku-duzeltmesi.md`."""

    _patch_happy_path_without_growth_simulation_default(monkeypatch)

    series_df = _build_synthetic_lumiere_series(n_patients=7)
    patient_ids = sorted(series_df["patient_id"].unique())
    rano_df = _build_synthetic_rano_df(patient_ids, rano_label="PD")
    _patch_growth_simulation_db(monkeypatch, series_df=series_df, rano_df=rano_df)

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="Patient-901")
    )

    growth = result["growth_simulation"]
    assert growth["available"] is True
    assert growth["n_visits"] == 5
    assert growth["fit_status"] == "ok"
    assert growth["best_model"] == "gompertz"
    assert growth["best_r"] == pytest.approx(0.058759, abs=1e-4)
    assert "SIMULASYONDUR" in growth["simulation_interpretation_note"]

    assert growth["rano_group"] == "PD"
    assert growth["rano_group_n"] == 7
    assert growth["rano_group_interpretable"] is True

    projection = growth["projection"]
    assert projection is not None
    assert projection["v0"] == pytest.approx(2879.336, rel=1e-3)
    assert projection["months"] == 6.0
    assert projection["r_median"] == pytest.approx(0.146561, abs=1e-4)
    assert projection["v_median"] == pytest.approx(5300.8868055446765, rel=1e-9)
    assert projection["pct_median"] == pytest.approx(84.10101987123443, rel=1e-9)
    assert projection["v_low"] < projection["v_median"] < projection["v_high"]

    # --- G1 DUZELTMESININ KIRMIZI/YESIL ASSERT'LERI (2026-09-13) ---------
    # Sentetik serinin SON ziyareti hafta 16; `v0` o ziyaretin hacmi.
    # Projeksiyon ufku SON ZIYARETTEN SONRA olmak ZORUNDA.
    last_visit_week = 16.0
    assert projection["from_week"] == pytest.approx(last_visit_week)
    assert projection["t_target_week"] == pytest.approx(
        last_visit_week + 6.0 * growth_simulation_pipeline._WEEKS_PER_MONTH
    )
    # Eski (hatali) davranis geri gelirse t_target_week = 26,07 olur ve
    # asagidaki assert KIRMIZI'ya doner (26,07 > 16 olmasina ragmen
    # `from_week` 0 dondugu icin yukaridaki iki assert de kirilir; bu
    # assert ise ufkun referans ana gore ILERIDE oldugunu dogrudan kodlar).
    assert projection["t_target_week"] > last_visit_week, (
        "Projeksiyon ufku son ziyaretten ONCE/uzerinde -- geriye donuk "
        "karsilastirma regresyonu (G1)."
    )
    assert projection["t_target_week"] == pytest.approx(42.07)

    # Diger bloklar ETKILENMEDI.
    assert result["risk"]["risk_score_log_partial_hazard"] == 0.42
    assert result["similar_patients"]["available"] is True

    # summary -- 2026-09-13 (backend-agent-J) GUNCELLENDI. Eski hali:
    #     assert summary["all_blocks_available"] is False
    #     assert summary["unavailable_blocks"] == [
    #         {"block": "xgboost_recurrence",
    #          "reason": analyze_module.XGBOOST_NOT_AVAILABLE_REASON}
    #     ]
    # Eski beklenti, XGBoost blogunun KALICI OLARAK uretilemedigi
    # varsayimina dayaniyordu; blok artik `/predict`ten TASINIYOR ve bu
    # mutlu-yol sahtesinde `available:true` -- dolayisiyla HICBIR blok
    # unavailable DEGIL. Test silinmedi, gercege gore yeniden yazildi.
    summary = result["summary"]
    assert summary["unavailable_blocks"] == []
    assert summary["all_blocks_available"] is True
    assert result[analyze_module.XGBOOST_BLOCK_KEY]["available"] is True


def test_analyze_patient_growth_projection_horizon_is_after_last_visit_g1(
    monkeypatch,
) -> None:
    """G1 KIRMIZI/YESIL -- son ziyareti hafta **38** olan (yani 26,07'den
    SONRA, gercek LUMIERE'de tipik) bir seride projeksiyon ufku son
    ziyaretten SONRA olmak ZORUNDA.

    Bu, 2026-09-13'te duzeltilen hatanin DOGRUDAN regresyon testidir:
    duzeltmeden once `t_target = 6*4,345 = 26,07` idi ve `v0` hafta-38
    ziyaretinin hacmiydi -> yuzde degisim *'egrinin 12 hafta ONCEKI
    degeri vs bugunun gozlemi'* olan GERIYE DONUK bir karsilastirmaydi.
    Eski davranis geri gelirse `t_target_week` 26,07 doner ve asagidaki
    assert'ler KIRMIZI olur.

    Not: bu test SADECE buyume-simulasyonu blogunu dogrular; `risk`
    blogu uzerine LUMIERE varsayimi KURULMAZ (Y1 sonrasi LUMIERE risk
    skoru alabilecek)."""

    _patch_happy_path_without_growth_simulation_default(monkeypatch)

    late_weeks = [0, 8, 15, 21, 25, 30, 35, 38]
    series_df = _build_synthetic_lumiere_series(n_patients=7, weeks=late_weeks)
    patient_ids = sorted(series_df["patient_id"].unique())
    rano_df = _build_synthetic_rano_df(patient_ids, rano_label="PD")
    _patch_growth_simulation_db(monkeypatch, series_df=series_df, rano_df=rano_df)

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="Patient-901")
    )
    growth = result["growth_simulation"]
    assert growth["available"] is True
    projection = growth["projection"]
    assert projection is not None

    last_visit_week = float(max(late_weeks))
    assert last_visit_week == 38.0
    # `v0` SON ziyaretin hacmi oldugu icin ufuk da oradan baslar.
    assert projection["from_week"] == pytest.approx(last_visit_week)
    assert projection["t_target_week"] == pytest.approx(
        last_visit_week + 6.0 * growth_simulation_pipeline._WEEKS_PER_MONTH
    )
    assert projection["t_target_week"] == pytest.approx(64.07)
    # ESKI (hatali) davranis: 26,07 < 38 -> bu assert KIRMIZI olurdu.
    assert projection["t_target_week"] > last_visit_week, (
        "Projeksiyon ufku son ziyaretten ONCE -- G1 regresyonu geri geldi."
    )
    assert projection["v0"] == pytest.approx(
        float(
            series_df[series_df["patient_id"] == "Patient-901"]
            .sort_values("x_week")
            .iloc[-1]["tumor_volume_mm3"]
        )
    )


def test_analyze_patient_growth_simulation_projection_not_available_when_group_too_small(
    monkeypatch,
) -> None:
    """RANO grubu `interpretable=False` (n<6, 2026-09-11 Baris karari) ise
    projeksiyon araligi URETILMEZ ama fit nokta-tahmini (best_r) HALA
    dondurulur -- kismi ama SESSIZ-OLMAYAN bir sonuc."""

    _patch_happy_path_without_growth_simulation_default(monkeypatch)

    series_df = _build_synthetic_lumiere_series(n_patients=3)  # n=3 < min_group_n=6
    patient_ids = sorted(series_df["patient_id"].unique())
    rano_df = _build_synthetic_rano_df(patient_ids, rano_label="PD")
    _patch_growth_simulation_db(monkeypatch, series_df=series_df, rano_df=rano_df)

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="Patient-901")
    )

    growth = result["growth_simulation"]
    assert growth["available"] is True
    assert growth["best_r"] is not None
    assert growth["rano_group"] == "PD"
    assert growth["rano_group_n"] == 3
    assert growth["rano_group_interpretable"] is False
    assert growth["projection"] is None
    assert "interpretable" in growth["projection_not_available_reason"]


def test_analyze_patient_growth_simulation_no_rano_group_still_returns_fit(monkeypatch) -> None:
    """Hasta LUMIERE serisinde VAR, fit basarili, ama gecerli bir RANO
    yanit etiketi (PD/SD/PR/CR) YOK -- projeksiyon URETILEMEZ ama fit
    nokta-tahmini HALA doner."""

    _patch_happy_path_without_growth_simulation_default(monkeypatch)

    series_df = _build_synthetic_lumiere_series(n_patients=7)
    _patch_growth_simulation_db(monkeypatch, series_df=series_df, rano_df=_empty_rano_df())

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="Patient-901")
    )

    growth = result["growth_simulation"]
    assert growth["available"] is True
    assert growth["fit_status"] == "ok"
    assert growth["rano_group"] is None
    assert growth["projection"] is None
    assert "RANO" in growth["projection_not_available_reason"]


def test_analyze_patient_growth_simulation_degrades_on_db_connection_failure(monkeypatch) -> None:
    """DB baglantisi kurulamazsa (canli olcumde her zaman OLMASI beklenmez,
    ama savunmaci kod yolu) pipeline DURMAZ -- `available: False` + hata
    metni tasiyan bir blok doner, HTTPException FIRLATILMAZ."""

    _patch_happy_path_without_growth_simulation_default(monkeypatch)

    def _raise_connection_error(readonly=True):
        raise RuntimeError("DB'ye erisilemedi (test)")

    monkeypatch.setattr(
        analyze_module.db_connection_module, "get_connection", _raise_connection_error
    )

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="Patient-901")
    )

    growth = result["growth_simulation"]
    assert growth["available"] is False
    assert "RuntimeError" in growth["not_available_reason"]
    # Diger bloklar ETKILENMEDI.
    assert result["risk"]["risk_score_log_partial_hazard"] == 0.42


# =====================================================================
# 6) XGBoost (shadow) ikinci katman -- 2026-09-13 (backend-agent-J).
#    `predict_patient()` yanitinin `xgboost` blogu `/analyze_patient`
#    ciktisina `xgboost_12mo_survival` adiyla TASINIR; hicbir deger
#    burada YENIDEN HESAPLANMAZ. Hata felsefesi B-1 ile AYNI: BLOK-FATAL
#    (`available:false` + sebep), istek 200 devam eder, UYDURMA olasilik
#    ASLA dondurulmez.
# =====================================================================


def test_analyze_patient_xgboost_block_is_propagated_verbatim(monkeypatch) -> None:
    """Blok `/predict` yanitindan AYNEN tasinir: kaynak blogun HER
    anahtari, AYNI degerle ciktida olmali (yeniden hesap YOK, secici
    kopyalama YOK -- eski `_build_risk_block()` hatasi tam da secici
    kopyalamaydi). Ayrica kaynak sozluk MUTASYONA UGRAMAMALI."""

    source_block = _fake_xgboost_block()
    source_snapshot = dict(source_block)
    _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        analyze_module.predict_module,
        "predict_patient",
        lambda pid: _fake_cox_result(xgboost_block=source_block),
    )

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="UCSF-PDGM-167")
    )

    xgb = result[analyze_module.XGBOOST_BLOCK_KEY]
    for key, value in source_snapshot.items():
        assert xgb[key] == value, f"{key!r} tasinirken degisti/kayboldu"
    # Kaynak sozluk DEGISMEDI (sig kopya alindi).
    assert source_block == source_snapshot
    # Orkestrasyon metadata'si EKLENDI (kaynagin alanlarini EZMEDEN).
    assert xgb["propagated_from"] == "api.predict.predict_patient()['xgboost']"
    assert "yeniden hesaplamaz" in xgb["propagation_note"].lower()


def test_analyze_patient_xgboost_metadata_does_not_overwrite_predict_values(
    monkeypatch,
) -> None:
    """Eklenen metadata anahtarlari kaynakta ZATEN varsa EZILMEZ --
    `api/predict.py`'nin kendi degeri kazanir (ileride o dosya ayni adli
    bir alan eklerse sessiz ustune-yazma olmasin)."""

    source_block = _fake_xgboost_block()
    source_block["propagated_from"] = "predict.py'nin KENDI degeri"
    _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        analyze_module.predict_module,
        "predict_patient",
        lambda pid: _fake_cox_result(xgboost_block=source_block),
    )

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="X")
    )
    assert (
        result[analyze_module.XGBOOST_BLOCK_KEY]["propagated_from"]
        == "predict.py'nin KENDI degeri"
    )


def test_analyze_patient_xgboost_shadow_fields_survive_propagation(monkeypatch) -> None:
    """`status="shadow"`, `is_primary_decision_model=False` ve
    `direction_note` GORUNUR KALMALI -- arayuz bu ucunu gostermek
    ZORUNDA (yuksek olasilik = IYI prognoz, Cox riskinin TERSI)."""

    _patch_happy_path(monkeypatch)

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="X")
    )
    xgb = result[analyze_module.XGBOOST_BLOCK_KEY]
    assert xgb["status"] == "shadow"
    assert xgb["is_primary_decision_model"] is False
    assert "IYI prognoz" in xgb["direction_note"]
    assert "TERS" in xgb["direction_note"]
    assert xgb["status_note"]
    # `model_registry` statusu bu dosyadan DEGISTIRILMEZ -- servis edilen
    # deger her zaman predict.py'nin sabitidir.
    assert analyze_module.predict_module.XGBOOST_MODEL_STATUS == "shadow"


def test_analyze_patient_xgboost_block_fatal_is_propagated_without_probability(
    monkeypatch,
) -> None:
    """BLOK-FATAL: `/predict` blogu `available:false` dondurduyse o hali
    AYNEN tasinir -- istek 200 kalir, Cox/FAISS/RAG ETKILENMEZ, UYDURMA
    olasilik YOK (`probability_12_month_survival` anahtari HIC YOK) ve
    blok `summary.unavailable_blocks`'a DUSER."""

    _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        analyze_module.predict_module,
        "predict_patient",
        lambda pid: _fake_cox_result(xgboost_block=_fake_xgboost_block(available=False)),
    )

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="X")
    )

    xgb = result[analyze_module.XGBOOST_BLOCK_KEY]
    assert xgb["available"] is False
    assert "probability_12_month_survival" not in xgb  # uydurma/None YOK
    assert xgb["error_type"] == "XGBoostCheckpointNotFoundError"
    assert xgb["status"] == "shadow"
    # Istek OLMEDI, diger bloklar uretildi.
    assert result["risk"]["available"] is True
    assert result["risk"]["risk_score_log_partial_hazard"] == 0.42
    assert result["similar_patients"]["available"] is True

    summary = result["summary"]
    assert summary["all_blocks_available"] is False
    unavailable = {b["block"]: b["reason"] for b in summary["unavailable_blocks"]}
    assert analyze_module.XGBOOST_BLOCK_KEY in unavailable
    assert "URETILEMEDI" in unavailable[analyze_module.XGBOOST_BLOCK_KEY]


def test_analyze_patient_xgboost_unavailable_when_cox_block_fatal(monkeypatch) -> None:
    """Cox BLOK-FATAL (422 coklu-tarama) ise `predict_patient()` yaniti HIC
    OLUSMAZ -> tasinacak XGBoost blogu da yoktur. Bagimlilik zinciri ACIKCA
    yazilir, olasilik burada YENIDEN HESAPLANMAZ ve UYDURULMAZ; istek yine
    200 doner."""

    def _raise(patient_id: str):
        raise HTTPException(status_code=422, detail="Coklu-tarama, C32 radyomigi belirsiz")

    _patch_happy_path(monkeypatch)
    monkeypatch.setattr(analyze_module.predict_module, "predict_patient", _raise)

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="Patient-002")
    )

    assert result["risk"]["available"] is False
    xgb = result[analyze_module.XGBOOST_BLOCK_KEY]
    assert xgb["available"] is False
    assert "probability_12_month_survival" not in xgb
    assert "422" in xgb["not_available_reason"]
    assert "Coklu-tarama" in xgb["not_available_reason"]
    assert "YENIDEN" in xgb["not_available_reason"]  # tek hesap noktasi ilkesi
    assert xgb["status"] == "shadow"
    assert xgb["is_primary_decision_model"] is False

    unavailable_names = {b["block"] for b in result["summary"]["unavailable_blocks"]}
    assert {"risk", analyze_module.XGBOOST_BLOCK_KEY} <= unavailable_names


def test_analyze_patient_xgboost_unavailable_on_unexpected_cox_exception(monkeypatch) -> None:
    """Cox tarafinda beklenmedik bir `Exception` olursa XGBoost blogu da
    BLOK-FATAL doner (sessiz `KeyError` ile istek DUSMEZ)."""

    def _raise(patient_id: str):
        raise RuntimeError("beklenmeyen bir Cox hatasi")

    _patch_happy_path(monkeypatch)
    monkeypatch.setattr(analyze_module.predict_module, "predict_patient", _raise)

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="X")
    )
    xgb = result[analyze_module.XGBOOST_BLOCK_KEY]
    assert xgb["available"] is False
    assert "RuntimeError" in xgb["not_available_reason"]
    assert "probability_12_month_survival" not in xgb


def test_analyze_patient_xgboost_missing_key_in_predict_response_is_blok_fatal(
    monkeypatch,
) -> None:
    """SOZLESME IHLALI: `predict_patient()` 200 dondu ama yanitta `xgboost`
    anahtari YOK (orn. birisi `api/predict.py`'den blogu kaldirdi) --
    sessiz `KeyError`/kayip alan YERINE gorunur bir BLOK-FATAL doner."""

    _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        analyze_module.predict_module,
        "predict_patient",
        lambda pid: _fake_cox_result(xgboost_block=None),
    )

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="X")
    )
    xgb = result[analyze_module.XGBOOST_BLOCK_KEY]
    assert xgb["available"] is False
    assert "sozlesme ihlali" in xgb["not_available_reason"]
    assert "probability_12_month_survival" not in xgb
    # Cox blogu ETKILENMEDI.
    assert result["risk"]["available"] is True


def test_analyze_patient_xgboost_available_without_direction_note_is_blok_fatal(
    monkeypatch,
) -> None:
    """FAIL-CLOSED: `available:true` ama `direction_note` (veya olasilik/
    statu alanlari) EKSIK ise olasilik SERVIS EDILMEZ. Gerekce: o not
    olmadan 'yuksek olasilik = IYI prognoz' bilgisi kaybolur ve deger Cox
    risk skoruyla AYNI yonde okunabilir -- bugun yakalanan 'etiket !=
    gercek' hatasinin ayni sinifi."""

    broken = _fake_xgboost_block()
    del broken["direction_note"]
    _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        analyze_module.predict_module,
        "predict_patient",
        lambda pid: _fake_cox_result(xgboost_block=broken),
    )

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="X")
    )
    xgb = result[analyze_module.XGBOOST_BLOCK_KEY]
    assert xgb["available"] is False
    assert "direction_note" in xgb["not_available_reason"]
    assert "probability_12_month_survival" not in xgb


def test_analyze_patient_xgboost_block_without_available_key_is_blok_fatal(
    monkeypatch,
) -> None:
    """`available` alani hic yoksa blok AVAILABLE SAYILMAZ (belirsizlik
    lehte yorumlanmaz)."""

    broken = _fake_xgboost_block()
    del broken["available"]
    _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        analyze_module.predict_module,
        "predict_patient",
        lambda pid: _fake_cox_result(xgboost_block=broken),
    )

    result = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="X")
    )
    xgb = result[analyze_module.XGBOOST_BLOCK_KEY]
    assert xgb["available"] is False
    assert "`available` alani YOK" in xgb["not_available_reason"]


def test_analyze_patient_old_xgboost_recurrence_key_is_gone(monkeypatch) -> None:
    """TEMIZ KESME regresyonu -- eski/yanlis `xgboost_recurrence` adi UST
    SEVIYEDE hicbir dalda GERI GELMEMELI (ne mutlu yolda, ne Cox
    dustugunde). 'recurrence' nuks/progresyon ima ediyordu; egitilen hedef
    `target_12mo_survival`dir."""

    _patch_happy_path(monkeypatch)
    happy = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="X")
    )
    assert "xgboost_recurrence" not in happy
    assert "xgboost_recurrence" not in {b["block"] for b in happy["summary"]["unavailable_blocks"]}

    def _raise(patient_id: str):
        raise HTTPException(status_code=422, detail="422 (test)")

    monkeypatch.setattr(analyze_module.predict_module, "predict_patient", _raise)
    degraded = analyze_module.analyze_patient(
        analyze_module.AnalyzePatientRequest(patient_id="X")
    )
    assert "xgboost_recurrence" not in degraded
    assert analyze_module.XGBOOST_BLOCK_KEY in degraded
    # Eski ad yalniz blogun ICINDE, aciklamali bir iz olarak durur.
    assert degraded[analyze_module.XGBOOST_BLOCK_KEY]["previous_block_name"] == (
        "xgboost_recurrence"
    )
    assert "YANLISTI" in degraded[analyze_module.XGBOOST_BLOCK_KEY]["previous_block_name_note"]


# =====================================================================
# 7) Request govdesi -- plan.txt:169 sozlesmesi ("patient_id" govde alani)
# =====================================================================


def test_analyze_patient_request_model_parses_patient_id() -> None:
    request = analyze_module.AnalyzePatientRequest(**{"patient_id": "TCGA-02-0003"})
    assert request.patient_id == "TCGA-02-0003"
