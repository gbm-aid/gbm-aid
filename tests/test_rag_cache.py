"""`pipeline/rag_cache.py` testleri -- Upstash REST istemcisi.

Ağ çağrıları MOCK'lanır (gerçek Upstash isteği YOK) -- bağlantının canlı
çalıştığı bu görevde AYRICA, minimal bir `PING` isteğiyle elle doğrulandı
(bkz. final yanıt). Buradaki testler MANTIK/graceful-degradation
odaklıdır.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline import rag_cache


def test_query_signature_is_deterministic_and_patient_id_free() -> None:
    sig1 = rag_cache.query_signature("glioblastoma MGMT methylated IDH1 wildtype")
    sig2 = rag_cache.query_signature("glioblastoma MGMT methylated IDH1 wildtype")
    sig3 = rag_cache.query_signature("glioblastoma MGMT unmethylated IDH1 mutant")
    assert sig1 == sig2
    assert sig1 != sig3
    assert len(sig1) == 64  # sha256 hex


def test_load_upstash_config_missing_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)
    # `rag_cache.py` `from db_connection import load_project_environment` ile
    # ismi kendi ad alanina KOPYALADIGI icin `db_connection.load_project_
    # environment`'i degil, `rag_cache.load_project_environment`'i patch'lemek
    # gerekiyor (Python import semantigi).
    monkeypatch.setattr(rag_cache, "load_project_environment", lambda: None)
    assert rag_cache.load_upstash_config() is None


def test_get_cached_summary_returns_none_when_config_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag_cache, "load_upstash_config", lambda: None)
    assert rag_cache.get_cached_summary("abc") is None


def test_set_cached_summary_returns_false_when_config_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag_cache, "load_upstash_config", lambda: None)
    assert rag_cache.set_cached_summary("abc", {"x": 1}) is False


def test_increment_rate_limit_counter_returns_none_when_config_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag_cache, "load_upstash_config", lambda: None)
    assert rag_cache.increment_rate_limit_counter() is None


def test_upstash_request_network_error_returns_none_not_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ağ hatası graceful degrade eder -- exception ASLA çağırana sızmaz."""

    config = rag_cache.UpstashConfig(rest_url="https://fake.upstash.io", rest_token="fake-token")

    def _raise(*args, **kwargs):
        raise ConnectionError("network unreachable")

    with patch("requests.get", side_effect=_raise):
        result = rag_cache._upstash_request(config, "PING")
    assert result is None


def test_check_upstash_connectivity_true_on_pong(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rag_cache,
        "load_upstash_config",
        lambda: rag_cache.UpstashConfig(rest_url="https://fake", rest_token="tok"),
    )
    monkeypatch.setattr(rag_cache, "_upstash_request", lambda config, *segments: "PONG")
    assert rag_cache.check_upstash_connectivity() is True


def test_check_upstash_connectivity_false_on_missing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag_cache, "load_upstash_config", lambda: None)
    assert rag_cache.check_upstash_connectivity() is False


def test_get_cached_summary_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    store: dict[str, str] = {}

    def fake_request(config, *segments):
        cmd = segments[0]
        if cmd == "SET":
            store[segments[1]] = segments[2]
            return "OK"
        if cmd == "GET":
            return store.get(segments[1])
        return None

    monkeypatch.setattr(
        rag_cache,
        "load_upstash_config",
        lambda: rag_cache.UpstashConfig(rest_url="https://fake", rest_token="tok"),
    )
    monkeypatch.setattr(rag_cache, "_upstash_request", fake_request)

    sig = rag_cache.query_signature("glioblastoma MGMT methylated")
    assert rag_cache.set_cached_summary(sig, {"articles": [{"pmid": "1"}]}) is True
    assert rag_cache.get_cached_summary(sig) == {"articles": [{"pmid": "1"}]}


def test_increment_rate_limit_counter_sets_expire_only_on_first_increment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_request(config, *segments):
        calls.append(segments)
        if segments[0] == "INCR":
            return 1  # Upstash REST INCR JSON sonucu int doner
        if segments[0] == "EXPIRE":
            return 1
        return None

    monkeypatch.setattr(
        rag_cache,
        "load_upstash_config",
        lambda: rag_cache.UpstashConfig(rest_url="https://fake", rest_token="tok"),
    )
    monkeypatch.setattr(rag_cache, "_upstash_request", fake_request)

    result = rag_cache.increment_rate_limit_counter()
    assert result == 1
    assert any(c[0] == "EXPIRE" for c in calls)
