"""``GET /patient/{patient_id}/volume_series`` (api/volume_series.py) --
birim + gerçek DB duman testleri.

Görev talimatı: paralel-38 (backend-agent-X1, 2026-09-16, "SITE AŞAMA
3-A"). Diğer test dosyalarının deseniyle AYNI desen (`tests/test_api_
patients.py`): endpoint fonksiyonları doğrudan Python fonksiyonu olarak
çağrılır. Gerçek-DB testleri erişilemezse `pytest.skip` ile atlanır.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from fastapi import HTTPException

import api.volume_series as volume_series_module


class _FakeConn:
    """`_get_db_connection()`'ın döndürdüğü nesnenin bu kod yolunda
    kullanılan TEK yöntemi `.close()`."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


# =====================================================================
# 1) Birim testleri -- DB YOK (fonksiyonlar monkeypatch ile izole edilir)
# =====================================================================


def test_build_volume_series_block_db_connection_error_is_graceful(monkeypatch):
    def _raise_connect():
        raise RuntimeError("baglanti reddedildi")

    monkeypatch.setattr(volume_series_module, "_get_db_connection", _raise_connect)

    result = volume_series_module.build_volume_series_block("Patient-001")

    assert result["available"] is False
    assert result["patient_id"] == "Patient-001"
    assert result["n_visits"] == 0
    assert result["visits"] == []
    assert "DB baglantisi kurulamadi" in result["not_available_reason"]


def test_build_volume_series_block_series_fetch_error_is_graceful(monkeypatch):
    monkeypatch.setattr(volume_series_module, "_get_db_connection", lambda: _FakeConn())

    def _raise_fetch(conn):
        raise RuntimeError("sorgu patladi")

    monkeypatch.setattr(
        volume_series_module.growth_simulation_module,
        "fetch_lumiere_wt_volume_series",
        _raise_fetch,
    )

    result = volume_series_module.build_volume_series_block("Patient-001")

    assert result["available"] is False
    assert "hacim serisi okuma" in result["not_available_reason"]


def test_build_volume_series_block_non_lumiere_patient_not_available(monkeypatch):
    monkeypatch.setattr(volume_series_module, "_get_db_connection", lambda: _FakeConn())
    empty_df = pd.DataFrame(
        columns=["patient_id", "scan_id", "x_week", "tumor_volume_mm3"]
    )
    monkeypatch.setattr(
        volume_series_module.growth_simulation_module,
        "fetch_lumiere_wt_volume_series",
        lambda conn: empty_df,
    )

    result = volume_series_module.build_volume_series_block("TCGA-02-0047")

    assert result["available"] is False
    assert result["n_visits"] == 0
    assert result["visits"] == []
    assert "LUMIERE" in result["not_available_reason"]
    assert "uydurulmus" in result["not_available_reason"]


def test_build_volume_series_block_merges_rano_by_scan_id_and_sorts_by_week(monkeypatch):
    monkeypatch.setattr(volume_series_module, "_get_db_connection", lambda: _FakeConn())

    series_df = pd.DataFrame(
        {
            "patient_id": ["Patient-001", "Patient-001", "Patient-002"],
            "scan_id": [2, 1, 9],
            "x_week": [4.0, 0.0, 0.0],
            "tumor_volume_mm3": [150.0, 100.0, 500.0],
        }
    )
    monkeypatch.setattr(
        volume_series_module.growth_simulation_module,
        "fetch_lumiere_wt_volume_series",
        lambda conn: series_df,
    )

    # Yalnız scan_id=1 için bir RANO eşleşmesi var -- scan_id=2 eşleşmeden
    # kalmalı (rano=None), scan_id=9 farklı hastaya ait, filtrelenmeli.
    rano_df = pd.DataFrame(
        {
            "patient_id": ["Patient-001", "Patient-002"],
            "scan_id": [1, 9],
            "rano_label": ["Pre-Op", "PD"],
            "match_type": ["unique", "unique"],
        }
    )
    monkeypatch.setattr(
        volume_series_module, "pair_lumiere_visits_with_rano", lambda conn: rano_df
    )

    result = volume_series_module.build_volume_series_block("Patient-001")

    assert result["available"] is True
    assert result["not_available_reason"] is None
    assert result["n_visits"] == 2
    # Sıra week'e göre artan olmalı (x_week=0.0 önce, 4.0 sonra) -- girdi
    # DataFrame'i KASITLI OLARAK ters sırada verildi.
    assert [v["week"] for v in result["visits"]] == [0.0, 4.0]
    assert result["visits"][0]["volume_mm3"] == 100.0
    assert result["visits"][0]["rano"] == "Pre-Op"
    assert result["visits"][0]["rano_match_type"] == "unique"
    # scan_id=2 için RANO eşleşmesi YOK -- uydurulmuş bir etiket KONULMAZ.
    assert result["visits"][1]["rano"] is None
    assert result["visits"][1]["rano_match_type"] is None


def test_build_volume_series_block_rano_join_failure_still_returns_series(monkeypatch, caplog):
    """RANO eşleştirmesi patlarsa hacim serisi YİNE DE dönmeli (opsiyonel
    zenginleştirme) -- sessizce yutulmaz (warning loglanır) ama istek
    başarısız OLMAZ."""

    monkeypatch.setattr(volume_series_module, "_get_db_connection", lambda: _FakeConn())
    series_df = pd.DataFrame(
        {
            "patient_id": ["Patient-001"],
            "scan_id": [1],
            "x_week": [0.0],
            "tumor_volume_mm3": [100.0],
        }
    )
    monkeypatch.setattr(
        volume_series_module.growth_simulation_module,
        "fetch_lumiere_wt_volume_series",
        lambda conn: series_df,
    )

    def _raise_rano(conn):
        raise RuntimeError("rano eslestirme patladi")

    monkeypatch.setattr(volume_series_module, "pair_lumiere_visits_with_rano", _raise_rano)

    with caplog.at_level("WARNING"):
        result = volume_series_module.build_volume_series_block("Patient-001")

    assert result["available"] is True
    assert result["n_visits"] == 1
    assert result["visits"][0]["rano"] is None
    assert any("RANO eslestirmesi basarisiz" in rec.message for rec in caplog.records)


def test_get_volume_series_raises_404_when_patient_missing(monkeypatch):
    monkeypatch.setattr(volume_series_module, "_get_db_connection", lambda: _FakeConn())
    monkeypatch.setattr(
        volume_series_module, "_fetch_patient_exists", lambda conn, patient_id: False
    )

    with pytest.raises(HTTPException) as excinfo:
        volume_series_module.get_volume_series("NOPE-999")
    assert excinfo.value.status_code == 404


def test_get_volume_series_raises_503_when_existence_check_db_fails(monkeypatch):
    def _raise_connect():
        raise RuntimeError("DB cokmus")

    monkeypatch.setattr(volume_series_module, "_get_db_connection", _raise_connect)

    with pytest.raises(HTTPException) as excinfo:
        volume_series_module.get_volume_series("Patient-001")
    assert excinfo.value.status_code == 503


def test_get_volume_series_delegates_to_build_block_when_patient_exists(monkeypatch):
    monkeypatch.setattr(volume_series_module, "_get_db_connection", lambda: _FakeConn())
    monkeypatch.setattr(
        volume_series_module, "_fetch_patient_exists", lambda conn, patient_id: True
    )

    sentinel: dict[str, Any] = {"available": True, "patient_id": "Patient-001", "n_visits": 0, "visits": [], "not_available_reason": None}
    monkeypatch.setattr(
        volume_series_module,
        "build_volume_series_block",
        lambda patient_id: sentinel,
    )

    result = volume_series_module.get_volume_series("Patient-001")
    assert result is sentinel


# =====================================================================
# 2) Gerçek DB duman testleri -- DB'ye erişilemezse ATLANIR (mevcut desen).
# =====================================================================


def test_volume_series_real_db_smoke_lumiere_patient_has_series():
    try:
        result = volume_series_module.build_volume_series_block("Patient-001")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gercek DB'ye erisilemedi, smoke test atlandi: {exc}")

    if not result["available"]:
        pytest.skip(
            "Patient-001 canli DB'de artik seri uretmiyor (veri degismis "
            f"olabilir): {result.get('not_available_reason')}"
        )
    assert result["n_visits"] > 0
    assert len(result["visits"]) == result["n_visits"]
    weeks = [v["week"] for v in result["visits"]]
    assert weeks == sorted(weeks)  # artan sirada olmali
    for visit in result["visits"]:
        assert isinstance(visit["volume_mm3"], float)
        assert visit["rano"] is None or isinstance(visit["rano"], str)


def test_volume_series_real_db_smoke_non_lumiere_patient_not_available():
    try:
        result = volume_series_module.build_volume_series_block("TCGA-02-0047")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gercek DB'ye erisilemedi, smoke test atlandi: {exc}")

    assert result["available"] is False
    assert result["visits"] == []
    assert "LUMIERE" in result["not_available_reason"]


def test_volume_series_real_db_smoke_unknown_patient_404():
    try:
        with pytest.raises(HTTPException) as excinfo:
            volume_series_module.get_volume_series("NOPE-DOES-NOT-EXIST-999")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gercek DB'ye erisilemedi, smoke test atlandi: {exc}")
    assert excinfo.value.status_code == 404
