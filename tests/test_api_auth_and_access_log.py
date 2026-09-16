"""Minimal Auth stub + `access_log` yazımı -- birim testleri (G2, Hafta 5
kapanışı, api/main.py).

DB'ye BAĞLANMAZ -- `_write_access_log()` her testte `db_connection.
get_connection`'ı (dolaylı olarak `api.main.get_connection`) monkeypatch
eder. ASGI middleware'in KENDİSİ (`access_control_and_logging_
middleware`) `starlette.requests.Request` NESNESİ elle kurularak + sahte
bir `call_next` ile DOĞRUDAN çağrılır -- bu makinede `httpx` kurulu
olmadığı için `fastapi.testclient.TestClient` KULLANILAMIYOR (bkz. bu
görevin final yanıtı, "bilinen sınırlar"); bu yaklaşım repo'nun genel
test felsefesiyle (endpoint fonksiyonlarını DOĞRUDAN çağırma, bkz.
tests/test_api_harmonize.py) tutarlıdır.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from starlette.requests import Request
from starlette.responses import Response

import api.main as api_main


def _make_request(path: str, method: str = "GET", headers: dict[str, str] | None = None) -> Request:
    headers = headers or {}
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()],
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, query: str, params: tuple) -> None:
        self.executed.append((query, params))

    def close(self) -> None:
        return None


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.committed = False
        self.closed = False

    def cursor(self, **_kwargs: Any) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        self.closed = True


# =====================================================================
# 1) _is_authorized -- salt-mantık, DB/HTTP gerektirmez
# =====================================================================


def test_is_authorized_true_when_no_api_key_configured(monkeypatch) -> None:
    monkeypatch.delenv("GBMAID_API_KEY", raising=False)
    assert api_main._is_authorized(None) is True
    assert api_main._is_authorized("herhangi-bir-sey") is True


def test_is_authorized_requires_exact_match_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("GBMAID_API_KEY", "gizli-anahtar")
    assert api_main._is_authorized("gizli-anahtar") is True
    assert api_main._is_authorized("yanlis-anahtar") is False
    assert api_main._is_authorized(None) is False


def test_is_authorized_empty_string_env_treated_as_unset(monkeypatch) -> None:
    # ".env"de yanlışlıkla `GBMAID_API_KEY=` bırakılması auth'u
    # SESSİZCE etkisiz bırakmamalı gibi görünse de -- bu, "tanımsız"
    # ile AYNI (geliştirme modu) davranışa bilinçli olarak eşlenir,
    # bkz. `_api_key_env()` docstring'i.
    monkeypatch.setenv("GBMAID_API_KEY", "")
    assert api_main._api_key_env() is None
    assert api_main._is_authorized(None) is True


def test_is_authorized_uses_constant_time_comparison(monkeypatch) -> None:
    # Codex capraz inceleme (2026-09-11) HIGH bulgusu: `==` string
    # karsilastirmasi zamanlama-saldirisina acikti (Python `str.__eq__`
    # ilk uyusmayan karakterde erken cikar, dogru anahtarin uzunlugu/on
    # eki byte-byte zamanlama farkiyla sizdirilabilir). Duzeltme
    # `secrets.compare_digest` kullanir -- bu test DOGRUDAN o fonksiyonun
    # cagrildigini dogrular (davranissal sonuc zaten asagidaki
    # dogru/yanlis/None testleriyle sabit).
    calls: list[tuple[bytes, bytes]] = []
    real_compare_digest = api_main.secrets.compare_digest

    def _spy(a: bytes, b: bytes) -> bool:
        calls.append((a, b))
        return real_compare_digest(a, b)

    monkeypatch.setenv("GBMAID_API_KEY", "gizli-anahtar")
    monkeypatch.setattr(api_main.secrets, "compare_digest", _spy)

    assert api_main._is_authorized("gizli-anahtar") is True
    assert len(calls) == 1
    assert calls[0] == (b"gizli-anahtar", b"gizli-anahtar")


def test_is_authorized_correct_key_returns_true(monkeypatch) -> None:
    monkeypatch.setenv("GBMAID_API_KEY", "gizli-anahtar")
    assert api_main._is_authorized("gizli-anahtar") is True


def test_is_authorized_wrong_key_returns_false(monkeypatch) -> None:
    monkeypatch.setenv("GBMAID_API_KEY", "gizli-anahtar")
    assert api_main._is_authorized("yanlis-anahtar") is False


def test_is_authorized_none_key_returns_false_without_raising(monkeypatch) -> None:
    # `secrets.compare_digest` `None` alirsa `TypeError` firlatir --
    # `_is_authorized` bunu ASLA disariya sizdirmamali, once `None`
    # guard'i devreye girmeli.
    monkeypatch.setenv("GBMAID_API_KEY", "gizli-anahtar")
    assert api_main._is_authorized(None) is False


def test_is_authorized_empty_provided_key_returns_false_without_raising(monkeypatch) -> None:
    monkeypatch.setenv("GBMAID_API_KEY", "gizli-anahtar")
    assert api_main._is_authorized("") is False


# =====================================================================
# 2) _extract_patient_id_from_path
# =====================================================================


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/predict/TCGA-06-5412", "TCGA-06-5412"),
        ("/patient/UPENN-GBM-00022/similar", "UPENN-GBM-00022"),
        ("/patient/UPENN-GBM-00022/status", "UPENN-GBM-00022"),
        ("/patient/UPENN-GBM-00022/harmonize", "UPENN-GBM-00022"),
        ("/analyze_patient", None),  # patient_id govdede, yolda DEGIL
        ("/health", None),
        ("/", None),
    ],
)
def test_extract_patient_id_from_path(path: str, expected: str | None) -> None:
    assert api_main._extract_patient_id_from_path(path) == expected


# =====================================================================
# 3) _write_access_log -- INSERT sözleşmesi + graceful DB-hatası
# =====================================================================


def test_write_access_log_inserts_single_row_and_commits(monkeypatch) -> None:
    cursor = FakeCursor()
    connection = FakeConnection(cursor)
    monkeypatch.setattr(api_main, "get_connection", lambda readonly=False: connection)

    api_main._write_access_log(action="GET /predict/X -> 200", patient_id="X")

    assert len(cursor.executed) == 1
    query, params = cursor.executed[0]
    assert "INSERT INTO access_log" in query
    assert params == (None, None, "X", "GET /predict/X -> 200")
    assert connection.committed is True
    assert connection.closed is True


def test_write_access_log_never_raises_when_db_unavailable(monkeypatch, caplog) -> None:
    def _raise(readonly: bool = False):
        raise RuntimeError("DB erisilemiyor - test")

    monkeypatch.setattr(api_main, "get_connection", _raise)

    with caplog.at_level("WARNING", logger="api.access_log"):
        api_main._write_access_log(action="GET /health -> 200", patient_id=None)

    assert "yazılamadı" in caplog.text


def test_write_access_log_does_not_include_request_body() -> None:
    # Sözlesme testi: fonksiyon imzası SADECE action+patient_id kabul
    # eder -- istek gövdesi/hasta verisi TASIYACAK bir parametre YOK.
    import inspect

    params = set(inspect.signature(api_main._write_access_log).parameters)
    assert params == {"action", "patient_id"}


# =====================================================================
# 4) access_control_and_logging_middleware -- uçtan uca (ASGI Request
#    elle kuruldu, TestClient/httpx GEREKMEDEN)
# =====================================================================


def test_middleware_allows_request_and_logs_when_auth_disabled(monkeypatch) -> None:
    monkeypatch.delenv("GBMAID_API_KEY", raising=False)
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(
        api_main,
        "_write_access_log",
        lambda action, patient_id: logged.append({"action": action, "patient_id": patient_id}),
    )

    async def fake_call_next(_request: Request) -> Response:
        return Response(status_code=200)

    request = _make_request("/predict/TCGA-06-5412", method="POST")
    response = asyncio.run(
        api_main.access_control_and_logging_middleware(request, fake_call_next)
    )

    assert response.status_code == 200
    assert len(logged) == 1
    assert logged[0]["patient_id"] == "TCGA-06-5412"
    assert logged[0]["action"] == "POST /predict/TCGA-06-5412 -> 200"


def test_middleware_rejects_request_with_401_when_api_key_missing(monkeypatch) -> None:
    monkeypatch.setenv("GBMAID_API_KEY", "gizli-anahtar")
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(
        api_main,
        "_write_access_log",
        lambda action, patient_id: logged.append({"action": action, "patient_id": patient_id}),
    )

    async def fake_call_next(_request: Request) -> Response:  # pragma: no cover -- CAGRILMAMALI
        raise AssertionError("call_next 401 durumunda CAGRILMAMALI")

    request = _make_request("/predict/X")
    response = asyncio.run(
        api_main.access_control_and_logging_middleware(request, fake_call_next)
    )

    assert response.status_code == 401
    # Reddedilen denemeler de access_log'a YAZILIR (guvenlik denetim izi).
    assert len(logged) == 1
    assert "-> 401" in logged[0]["action"]


def test_middleware_allows_request_with_correct_api_key(monkeypatch) -> None:
    monkeypatch.setenv("GBMAID_API_KEY", "gizli-anahtar")
    monkeypatch.setattr(api_main, "_write_access_log", lambda action, patient_id: None)

    async def fake_call_next(_request: Request) -> Response:
        return Response(status_code=200)

    request = _make_request("/predict/X", headers={"X-API-Key": "gizli-anahtar"})
    response = asyncio.run(
        api_main.access_control_and_logging_middleware(request, fake_call_next)
    )
    assert response.status_code == 200


def test_middleware_logs_and_reraises_when_call_next_raises_unhandled_exception(
    monkeypatch,
) -> None:
    # Codex capraz inceleme (2026-09-11) MEDIUM bulgusu: `call_next`'ten
    # (HTTPException DISINDA) beklenmeyen bir istisna firlarsa eskiden
    # `_write_access_log` HIC CAGRILMIYORDU -- guvenlik denetim izinde
    # 500'e giden istekler GORUNMEZ kaliyordu. Duzeltme: call_next
    # try/finally ile sarilir, log finally'de yazilir, istisna AYNEN
    # yeniden firlatilir (ServerErrorMiddleware'e ulasmasi icin --
    # davranis DEGISMEZ, sadece log eklenir).
    monkeypatch.delenv("GBMAID_API_KEY", raising=False)
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(
        api_main,
        "_write_access_log",
        lambda action, patient_id: logged.append({"action": action, "patient_id": patient_id}),
    )

    async def fake_call_next_raises(_request: Request) -> Response:
        raise RuntimeError("beklenmeyen ham istisna - test")

    request = _make_request("/predict/TCGA-06-5412", method="POST")

    with pytest.raises(RuntimeError, match="beklenmeyen ham istisna - test"):
        asyncio.run(
            api_main.access_control_and_logging_middleware(
                request, fake_call_next_raises
            )
        )

    assert len(logged) == 1
    assert logged[0]["patient_id"] == "TCGA-06-5412"
    assert "POST /predict/TCGA-06-5412" in logged[0]["action"]
    assert "500" in logged[0]["action"]


def test_middleware_never_logs_request_body(monkeypatch) -> None:
    """CLAUDE.md/gorev talimati: 'ISTEK GOVDESI/hasta verisi YAZILMAZ'.
    Middleware, `_write_access_log`'u SADECE `action`/`patient_id` ile
    cagirir -- govde okuma/iletme YOK (imza kontrolu, bkz. yukaridaki
    `test_write_access_log_does_not_include_request_body`)."""

    calls: list[tuple] = []
    monkeypatch.setattr(
        api_main,
        "_write_access_log",
        lambda action, patient_id: calls.append((action, patient_id)),
    )
    monkeypatch.delenv("GBMAID_API_KEY", raising=False)

    async def fake_call_next(_request: Request) -> Response:
        return Response(status_code=200)

    request = _make_request("/analyze_patient", method="POST")
    asyncio.run(api_main.access_control_and_logging_middleware(request, fake_call_next))

    assert len(calls) == 1
    action, patient_id = calls[0]
    assert isinstance(action, str)
    assert patient_id is None  # /analyze_patient -- govdede, yolda DEGIL
