"""``GET /health`` -- Redis/Upstash alanı birim testleri (G3 düzeltmesi,
2026-09-11, api/main.py).

ÖNCEKİ HATA (10+ gündür açık bulgu): `/health` `redis.from_url(REDIS_URL)`
çağırıyordu ama `.env`'de `REDIS_URL` HİÇ tanımlı değil (yalnız Upstash
REST kimlik bilgileri var, bkz. `pipeline/rag_cache.py`) -- bu yüzden
`redis` alanı HER ZAMAN "error" dönüyordu, konfigürasyon eksikliği ile
gerçek bir Upstash erişilemezliği AYIRT EDİLEMİYORDU. Bu dosya üç durumu
AYRI AYRI doğrular: yapılandırılmamış / erişilebilir / erişilemez.
"""

from __future__ import annotations

import api.main as api_main


def test_health_redis_not_configured_when_upstash_env_missing(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "load_upstash_config", lambda: None)
    monkeypatch.setattr(
        api_main, "get_db_connection", lambda: (_ for _ in ()).throw(RuntimeError("db kapali - test"))
    )

    result = api_main.health_check()

    assert result["redis"].startswith("not_configured")


def test_health_redis_ok_when_upstash_ping_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "load_upstash_config", lambda: object())
    monkeypatch.setattr(api_main, "check_upstash_connectivity", lambda: True)
    monkeypatch.setattr(
        api_main, "get_db_connection", lambda: (_ for _ in ()).throw(RuntimeError("db kapali - test"))
    )

    result = api_main.health_check()

    assert result["redis"] == "ok"


def test_health_redis_error_when_upstash_ping_fails(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "load_upstash_config", lambda: object())
    monkeypatch.setattr(api_main, "check_upstash_connectivity", lambda: False)
    monkeypatch.setattr(
        api_main, "get_db_connection", lambda: (_ for _ in ()).throw(RuntimeError("db kapali - test"))
    )

    result = api_main.health_check()

    assert result["redis"].startswith("error")


def test_health_redis_error_when_upstash_check_raises(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "load_upstash_config", lambda: object())

    def _raise() -> bool:
        raise RuntimeError("ag hatasi - test")

    monkeypatch.setattr(api_main, "check_upstash_connectivity", _raise)
    monkeypatch.setattr(
        api_main, "get_db_connection", lambda: (_ for _ in ()).throw(RuntimeError("db kapali - test"))
    )

    # health_check HICBIR ZAMAN cokmemeli -- beklenmedik bir hata bile
    # "error: ..." metnine donusturulur, exception YUKARI SIZMAZ.
    result = api_main.health_check()

    assert result["redis"].startswith("error")
