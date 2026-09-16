"""``GET /patients`` (api/patients.py) -- birim + gerçek DB duman testleri.

Görev talimatı: paralel-36 (backend-agent-W1, 2026-09-15/16). Diğer test
dosyalarının deseniyle AYNI: endpoint fonksiyonları FastAPI/TestClient
ÜZERİNDEN DEĞİL, doğrudan Python fonksiyonu olarak çağrılır (bkz.
`tests/test_api_predict.py`, `api.predict.predict_patient(...)` deseni).
Bu yüzden `Query(...)` varsayılanlı parametreler HER test çağrısında
AÇIKÇA geçirilir -- FastAPI'nin dependency-injection katmanı devrede
OLMADIĞI için `Query(8, ...)` nesnesinin kendisi varsayılan olarak
kalmaz (bu bir hata DEĞİL, doğrudan-çağrı testlerinin bilinen bir
kısıtıdır).

Gerçek-DB testleri (`_real_db_smoke_` sonekli) CLAUDE.md'nin KİLİTLİ
sayılarını (611/585, 295/169, 39/38, 91/72) REGRESYON PİNİ olarak
kullanır -- DB'ye erişilemezse `pytest.skip` ile atlanır (mevcut desen,
`test_api_predict.py`'deki `real_db_smoke` testleriyle AYNI).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import api.patients as patients_module


# =====================================================================
# 1) Birim testleri -- DB YOK
# =====================================================================


def test_known_source_names_match_source_roles_keys():
    assert set(patients_module.KNOWN_SOURCE_NAMES) == set(patients_module.SOURCE_ROLES)
    # ZORUNLU-BEYANLAR.md B8 rol tablosuyla birebir.
    assert patients_module.SOURCE_ROLES["UPenn-GBM"] == "Eğitim"
    assert patients_module.SOURCE_ROLES["UCSF-PDGM"] == "Harici test"
    assert patients_module.SOURCE_ROLES["TCGA-Omics"] == "Opsiyonel omics modülü"


def test_validate_source_filter_accepts_known_rejects_unknown():
    patients_module._validate_source_filter(None)
    patients_module._validate_source_filter("UPenn-GBM")
    with pytest.raises(patients_module.UnknownSourceFilterError):
        patients_module._validate_source_filter("Upenn-gbm-typo")


def test_timepoint_whitelist_imported_not_copied_has_no_lumiere_key():
    # Görev talimatının açık uyarısı: LUMIERE bu sözlükte YOK (K2 kanonik-
    # vizit kuralı statik bir etiket kümesi DEĞİLDİR).
    assert patients_module.SEGMENTATION_TOOL_UPENN_C32 in patients_module.ALLOWED_TIMEPOINT_LABELS_BY_TOOL
    assert (
        patients_module.SEGMENTATION_TOOL_LUMIERE_C32
        not in patients_module.ALLOWED_TIMEPOINT_LABELS_BY_TOOL
    )
    assert patients_module.ALLOWED_TIMEPOINT_LABELS_BY_TOOL[
        patients_module.SEGMENTATION_TOOL_UPENN_C32
    ] == frozenset({"pre-treatment"})
    assert patients_module.ALLOWED_TIMEPOINT_LABELS_BY_TOOL[
        patients_module.SEGMENTATION_TOOL_UCSF_C32
    ] == frozenset({"baseline"})


def test_row_is_servisable_false_when_age_or_gender_missing():
    row = {"age": None, "gender": "Male", "source": "UPenn-GBM"}
    assert patients_module._row_is_servisable(row, frozenset()) is False

    row2 = {"age": 55, "gender": None, "source": "UPenn-GBM"}
    assert patients_module._row_is_servisable(row2, frozenset()) is False


def test_row_is_servisable_lumiere_membership():
    row_in = {
        "age": 60,
        "gender": "Male",
        "source": patients_module.LUMIERE_SOURCE_NAME,
        "patient_id": "Patient-028",
    }
    row_out = {
        "age": 60,
        "gender": "Male",
        "source": patients_module.LUMIERE_SOURCE_NAME,
        "patient_id": "Patient-999",
    }
    canonical_ids = frozenset({"Patient-028"})
    assert patients_module._row_is_servisable(row_in, canonical_ids) is True
    assert patients_module._row_is_servisable(row_out, canonical_ids) is False


def test_row_is_servisable_upenn_uses_has_c32_flag():
    tool = patients_module.SEGMENTATION_TOOL_UPENN_C32
    row_true = {
        "age": 60,
        "gender": "Male",
        "source": "UPenn-GBM",
        f"_has_c32__{tool}": True,
    }
    row_false = {
        "age": 60,
        "gender": "Male",
        "source": "UPenn-GBM",
        f"_has_c32__{tool}": False,
    }
    assert patients_module._row_is_servisable(row_true, frozenset()) is True
    assert patients_module._row_is_servisable(row_false, frozenset()) is False


def test_row_is_servisable_unknown_source_is_false():
    row = {"age": 60, "gender": "Male", "source": "TCGA-Omics"}
    assert patients_module._row_is_servisable(row, frozenset()) is False


def test_build_where_clause_defaults_to_true_with_no_filters():
    where_sql, params = patients_module._build_where_clause(
        source=None,
        gender=None,
        idh1_status=None,
        mgmt_status=None,
        gtr_over90percent=None,
        servis_edilebilir=None,
        lumiere_servisable_ids=frozenset(),
    )
    assert where_sql == "TRUE"
    assert params == {}


def test_build_where_clause_source_filter_uses_exact_param():
    where_sql, params = patients_module._build_where_clause(
        source="UPenn-GBM",
        gender=None,
        idh1_status=None,
        mgmt_status=None,
        gtr_over90percent=None,
        servis_edilebilir=None,
        lumiere_servisable_ids=frozenset(),
    )
    assert "ds.source_name = %(source)s" in where_sql
    assert params["source"] == "UPenn-GBM"


def test_build_where_clause_servis_edilebilir_true_includes_lumiere_ids():
    lumiere_ids = frozenset({"Patient-028", "Patient-060"})
    where_sql, params = patients_module._build_where_clause(
        source=None,
        gender=None,
        idh1_status=None,
        mgmt_status=None,
        gtr_over90percent=None,
        servis_edilebilir=True,
        lumiere_servisable_ids=lumiere_ids,
    )
    assert "p.patient_id = ANY(%(lumiere_ids)s)" in where_sql
    assert set(params["lumiere_ids"]) == lumiere_ids
    assert not where_sql.strip().startswith("NOT")


def test_build_where_clause_servis_edilebilir_false_negates_predicate():
    where_sql, _params = patients_module._build_where_clause(
        source=None,
        gender=None,
        idh1_status=None,
        mgmt_status=None,
        gtr_over90percent=None,
        servis_edilebilir=False,
        lumiere_servisable_ids=frozenset(),
    )
    assert where_sql.strip().startswith("NOT")


def test_compute_lumiere_servisable_patient_ids_empty_long_frame(monkeypatch):
    monkeypatch.setattr(
        patients_module,
        "_fetch_lumiere_c32_long_frame",
        lambda conn: pd.DataFrame(columns=["patient_id", "timepoint_label", "scan_id"]),
    )
    ids, warning = patients_module._compute_lumiere_servisable_patient_ids(conn=None)
    assert ids == frozenset()
    assert warning is not None and "C32" in warning


def test_compute_lumiere_servisable_patient_ids_graceful_on_cohort_mismatch(monkeypatch):
    fake_frame = pd.DataFrame(
        {
            "patient_id": ["Patient-001"],
            "timepoint_label": ["week-000"],
            "scan_id": [1],
        }
    )
    monkeypatch.setattr(
        patients_module, "_fetch_lumiere_c32_long_frame", lambda conn: fake_frame
    )

    def _raise_mismatch(*args: Any, **kwargs: Any):
        raise patients_module.LumierePreopCohortMismatchError("sayı tutmadı")

    monkeypatch.setattr(
        patients_module, "select_lumiere_preop_canonical_visits", _raise_mismatch
    )
    ids, warning = patients_module._compute_lumiere_servisable_patient_ids(conn=None)
    assert ids == frozenset()
    assert warning is not None
    assert "null" in warning or "bilinmiyor" in warning


def test_list_patients_unknown_source_raises_422():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        patients_module.list_patients(
            source="does-not-exist",
            gender=None,
            idh1_status=None,
            mgmt_status=None,
            gtr_over90percent=None,
            servis_edilebilir=None,
            limit=8,
            offset=0,
        )
    assert excinfo.value.status_code == 422


# =====================================================================
# 2) Gerçek DB duman testleri -- CLAUDE.md kilitli sayılarının regresyon
#    pini. DB'ye erişilemezse ATLANIR (mevcut desen).
# =====================================================================


def _call_list_patients(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "source": None,
        "gender": None,
        "idh1_status": None,
        "mgmt_status": None,
        "gtr_over90percent": None,
        "servis_edilebilir": None,
        "limit": 8,
        "offset": 0,
    }
    params.update(overrides)
    return patients_module.list_patients(**params)


def test_list_patients_real_db_smoke_counters_match_locked_values():
    try:
        response = _call_list_patients(limit=1)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gercek DB'ye erisilemedi, smoke test atlandi: {exc}")

    by_source = {row["source"]: row for row in response["counters"]["by_source"]}

    upenn = by_source["UPenn-GBM"]
    assert upenn["n_registered"] == 630
    assert upenn["n_train_pool"] == 611
    assert upenn["n_events_train_pool"] == 585

    ucsf = by_source["UCSF-PDGM"]
    assert ucsf["n_registered"] == 295
    assert ucsf["n_external_test_pool"] == 295
    assert ucsf["n_events_external_test"] == 169

    lumiere = by_source[patients_module.LUMIERE_SOURCE_NAME]
    assert lumiere["n_registered"] == 91
    assert lumiere["n_c32_radiomics_any_visit"] == 90
    # K2 hesaplaması bu makinede BAŞARISIZ olursa (CSV yolu bulunamazsa)
    # 72 yerine 0 dönebilir -- bu durumda `warnings` alanı BOŞ OLAMAZ.
    if lumiere["n_faiss_servisable_canonical_preop"] != 72:
        assert response["warnings"], (
            "LUMIERE kanonik-72 sayısı 72 DEĞİL ama warnings BOŞ -- "
            "sessiz bir bozulma var."
        )
    else:
        assert lumiere["n_faiss_servisable_canonical_preop"] == 72

    tcga = by_source["TCGA-GBM"]
    assert tcga["n_c32_radiomics"] == 39
    assert tcga["n_predict_servisable"] == 38

    omics = by_source["TCGA-Omics"]
    assert omics["n_has_omics_total"] == 48


def test_list_patients_real_db_smoke_source_filter_returns_only_that_source():
    try:
        response = _call_list_patients(source="UCSF-PDGM", limit=5)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gercek DB'ye erisilemedi, smoke test atlandi: {exc}")

    assert response["pagination"]["total_matching"] == 295
    for row in response["patients"]:
        assert row["source"] == "UCSF-PDGM"
        assert row["role"] == "Harici test"


def test_list_patients_real_db_smoke_pagination_offset_beyond_total_is_empty():
    try:
        response = _call_list_patients(source="TCGA-Omics", limit=8, offset=10_000)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gercek DB'ye erisilemedi, smoke test atlandi: {exc}")

    assert response["patients"] == []
    assert response["pagination"]["total_matching"] >= 0


def test_list_patients_real_db_smoke_servisable_filter_upenn_matches_611():
    try:
        response = _call_list_patients(
            source="UPenn-GBM", servis_edilebilir=True, limit=1
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gercek DB'ye erisilemedi, smoke test atlandi: {exc}")

    assert response["pagination"]["total_matching"] == 611
