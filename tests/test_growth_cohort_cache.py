"""`api/analyze_patient.py` -- buyume simulasyonu KOHORT ONBELLEGI testleri.

NEDEN VAR (2026-09-17, Baris onayi)
===================================
Canli sunucuda (https://gbm-aid.tech) olculen sorun:

    POST /analyze_patient, LUMIERE hastasi    -> 58,7 / 70,9 / 64,5 sn
    POST /analyze_patient, LUMIERE disi hasta ->  4,8 /  8,1 sn
    POST /predict                             ->  2,2 sn
    GET  /patient/{id}/similar                ->  1,4 sn

Fark tamamen `_build_growth_simulation_block()`'tan geliyordu: blok HER
istekte TUM LUMIERE kohortunun buyume egrilerini sifirdan fit ediyordu.
IKINCI cagri da yavas kaldigi icin (70,9 sn) "demo oncesi bir kez cagirip
isitalim" yaklasimi YAPISAL OLARAK calismiyordu -- isinacak bir sey yoktu.

Bu dosya onbellegin DAVRANISINI kilitler. Ozellikle iki sinifi ayirir:
  * ONBELLEGE ALINAN: iki DB okumasi + `fit_patient_growth_curves` (pahali)
  * ALINMAYAN: RANO grup turevleri ve hastaya ozel adimlar (ucuz; ayrica
    cagri SIRALARI davranisin parcasi -- erken hesaplanirlarsa bos kohortta
    `KeyError: 'fit_status'` verirler, bu 2026-09-17'de olculdu).

Gercek DB'ye BAGLANMAZ: `tests/test_api_analyze_patient.py`'nin sentetik
kohort yardimcilari yeniden kullanilir (kopyalanmaz).
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

import api.analyze_patient as analyze_module

# Kardes test modulunun sentetik kohort yardimcilari -- YENIDEN YAZILMADI.
from test_api_analyze_patient import (  # type: ignore[import-not-found]
    _FakeConnection,
    _build_synthetic_lumiere_series,
    _build_synthetic_rano_df,
)

_LUMIERE_PATIENT = "Patient-901"
_OTHER_LUMIERE_PATIENT = "Patient-902"


class _CallCounters:
    """Kohort geneli adimlarin KAC KEZ calistigini sayar."""

    def __init__(self) -> None:
        self.connection = 0
        self.fetch_series = 0
        self.fetch_rano = 0
        self.fit = 0


def _patch_cohort_with_counters(
    monkeypatch, *, fit_delay_seconds: float = 0.0
) -> _CallCounters:
    """Sentetik 7 hastalik LUMIERE kohortunu sayaclarla baglar."""

    counters = _CallCounters()
    series_df = _build_synthetic_lumiere_series()
    patient_ids = sorted(series_df["patient_id"].unique().tolist())
    rano_df = _build_synthetic_rano_df(patient_ids, rano_label="PD")

    real_fit = analyze_module.growth_simulation_module.fit_patient_growth_curves

    def _fake_connection(readonly: bool = True) -> Any:
        counters.connection += 1
        return _FakeConnection()

    def _fake_series(conn: Any) -> Any:
        counters.fetch_series += 1
        return series_df

    def _fake_rano(conn: Any) -> Any:
        counters.fetch_rano += 1
        return rano_df

    def _counting_fit(frame: Any) -> Any:
        counters.fit += 1
        if fit_delay_seconds:
            time.sleep(fit_delay_seconds)
        return real_fit(frame)

    monkeypatch.setattr(
        analyze_module.db_connection_module, "get_connection", _fake_connection
    )
    monkeypatch.setattr(
        analyze_module.growth_simulation_module,
        "fetch_lumiere_wt_volume_series",
        _fake_series,
    )
    monkeypatch.setattr(
        analyze_module.growth_simulation_module, "fetch_lumiere_rano_labels", _fake_rano
    )
    monkeypatch.setattr(
        analyze_module.growth_simulation_module,
        "fit_patient_growth_curves",
        _counting_fit,
    )
    return counters


# ---------------------------------------------------------------- 1) TEMEL


def test_kohort_fit_iki_istek_icin_yalniz_BIR_KEZ_kosar(monkeypatch) -> None:
    """Asil kazanc bu: ayni surecte ikinci istek kohortu YENIDEN FIT ETMEZ.

    KIRMIZI/YESIL kaniti: onbellek yokken bu sayac 2 olurdu (canli olcumde
    ikinci cagri 70,9 sn surmustu).
    """

    counters = _patch_cohort_with_counters(monkeypatch)

    first = analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)
    second = analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)

    assert first["available"] is True
    assert second["available"] is True
    assert counters.fit == 1
    assert counters.fetch_series == 1
    assert counters.fetch_rano == 1
    assert counters.connection == 1


def test_onbellek_FARKLI_hasta_icin_de_paylasilir(monkeypatch) -> None:
    """Onbellek KOHORT genelidir -- hasta basina degil."""

    counters = _patch_cohort_with_counters(monkeypatch)

    first = analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)
    second = analyze_module._build_growth_simulation_block(_OTHER_LUMIERE_PATIENT)

    assert first["available"] is True
    assert second["available"] is True
    # Ayni kohort, FARKLI hastalar -> yine tek fit.
    assert counters.fit == 1
    # Ama donen sonuc hastaya OZELDIR, onbellekten "ayni yanit" gelmez.
    assert first["projection"] != second["projection"]


def test_onbellekli_sonuc_onbelleksiz_sonucla_BIREBIR_AYNI(monkeypatch) -> None:
    """SAYISAL ETKI YOK guvencesi -- bu yamanin en kritik iddiasi.

    Ayni sentetik kohort, onbellek ACIK ve KAPALI iken kosulur; donen
    projeksiyon sozlugu BIREBIR ayni olmalidir.
    """

    _patch_cohort_with_counters(monkeypatch)
    with_cache = analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)

    analyze_module.reset_growth_cohort_cache()
    monkeypatch.setenv("GBMAID_GROWTH_CACHE_DISABLED", "1")
    without_cache = analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)

    assert with_cache == without_cache


# ----------------------------------------------------------- 2) KAPATMA/TTL


def test_GBMAID_GROWTH_CACHE_DISABLED_onbellegi_kapatir(monkeypatch) -> None:
    """Kacis kapisi: eski davranis bir ortam degiskeniyle geri gelir."""

    counters = _patch_cohort_with_counters(monkeypatch)
    monkeypatch.setenv("GBMAID_GROWTH_CACHE_DISABLED", "1")

    analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)
    analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)

    assert counters.fit == 2


def test_TTL_dolunca_yeniden_hesaplanir(monkeypatch) -> None:
    """TTL VARSAYILAN OLARAK YOKTUR ama acildiginda gercekten calisir."""

    counters = _patch_cohort_with_counters(monkeypatch)
    monkeypatch.setenv("GBMAID_GROWTH_CACHE_TTL_SECONDS", "0.05")

    analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)
    assert counters.fit == 1

    time.sleep(0.08)
    analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)
    assert counters.fit == 2


def test_TTL_varsayilani_YOKTUR(monkeypatch) -> None:
    """Gerekce: bir TTL, juri demosunun ORTASINDA 55 saniyelik yeniden
    hesaba dusme riski yaratirdi."""

    monkeypatch.delenv("GBMAID_GROWTH_CACHE_TTL_SECONDS", raising=False)
    assert analyze_module._growth_cache_ttl_seconds() == 0.0


def test_gecersiz_TTL_degeri_SESSIZCE_yutulmaz_ama_COKMEZ(monkeypatch) -> None:
    """Bozuk deger -> TTL yok kabul edilir + uyari loglanir (cokme YOK)."""

    monkeypatch.setenv("GBMAID_GROWTH_CACHE_TTL_SECONDS", "abc")
    assert analyze_module._growth_cache_ttl_seconds() == 0.0


# -------------------------------------------------------------- 3) HATA YOLU


def test_HATA_onbellege_ALINMAZ(monkeypatch) -> None:
    """Gecici bir DB kesintisi KALICI bir 'kullanilamaz' durumuna donusmemeli.

    Bu, onbelleklerin klasik tuzagi -- burada acikca test ediliyor.
    """

    def _failing_connection(readonly: bool = True) -> Any:
        raise RuntimeError("DB gecici olarak erisilemez")

    monkeypatch.setattr(
        analyze_module.db_connection_module, "get_connection", _failing_connection
    )

    failed = analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)
    assert failed["available"] is False
    assert "DB baglantisi kurulamadi" in failed["not_available_reason"]

    # DB geri geliyor -- onbellek hatayi TUTMAMALI.
    counters = _patch_cohort_with_counters(monkeypatch)
    recovered = analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)

    assert recovered["available"] is True
    assert counters.fit == 1


def test_DB_OKUMA_hatasinin_metni_KORUNDU(monkeypatch) -> None:
    """Hangi asamada patladigi yanitta AYNEN korunur (asama ayrimi kaybolmasin)."""

    monkeypatch.setattr(
        analyze_module.db_connection_module,
        "get_connection",
        lambda readonly=True: _FakeConnection(),
    )

    def _boom(conn: Any) -> Any:
        raise RuntimeError("okuma patladi")

    monkeypatch.setattr(
        analyze_module.growth_simulation_module,
        "fetch_lumiere_wt_volume_series",
        _boom,
    )

    result = analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)
    assert result["available"] is False
    assert "Beklenmeyen hata (DB okuma)" in result["not_available_reason"]


def test_EGRI_FIT_hatasinin_metni_KORUNDU(monkeypatch) -> None:
    _patch_cohort_with_counters(monkeypatch)

    def _boom(frame: Any) -> Any:
        raise RuntimeError("fit patladi")

    monkeypatch.setattr(
        analyze_module.growth_simulation_module, "fit_patient_growth_curves", _boom
    )

    result = analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)
    assert result["available"] is False
    assert "Beklenmeyen hata (egri fit)" in result["not_available_reason"]


# ------------------------------------------------------------ 4) ES ZAMANLILIK


def test_es_zamanli_iki_istek_kohortu_IKI_KEZ_fit_ETMEZ(monkeypatch) -> None:
    """uvicorn sync endpoint'leri is parcacigi havuzunda kosar -- cift
    kontrollu kilit olmasaydi iki istek ayni anda 55 saniyelik fit'i
    BASLATIRDI."""

    counters = _patch_cohort_with_counters(monkeypatch, fit_delay_seconds=0.2)
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def _worker() -> None:
        try:
            barrier.wait(timeout=5)
            results.append(
                analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)
            )
        except BaseException as exc:  # pragma: no cover -- test tanisi icin
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"is parcaciklarinda hata: {errors!r}"
    assert len(results) == 2
    assert all(r["available"] is True for r in results)
    assert counters.fit == 1
    assert results[0] == results[1]


# --------------------------------------------------------- 5) SIFIRLAMA/IZOLASYON


def test_reset_growth_cohort_cache_gercekten_bosaltir(monkeypatch) -> None:
    """`conftest.py`'deki autouse fixture buna dayaniyor -- testler arasi
    kirlenmenin tek koruyucusu bu fonksiyon."""

    counters = _patch_cohort_with_counters(monkeypatch)

    analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)
    assert counters.fit == 1

    analyze_module.reset_growth_cohort_cache()
    analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)
    assert counters.fit == 2


def test_onbellek_testler_arasinda_SIZMAZ(monkeypatch) -> None:
    """Bu test, kendinden onceki testlerin doldurdugu onbellegi GORMEMELI.

    (Guvence `conftest.py::_reset_growth_cohort_cache` autouse fixture'indan
    gelir; burada acikca dogrulaniyor.)
    """

    counters = _patch_cohort_with_counters(monkeypatch)
    analyze_module._build_growth_simulation_block(_LUMIERE_PATIENT)
    assert counters.fit == 1, "onceki testin onbellegi sizmis olurdu -> 0 olurdu"
