"""Hafta 5 — Redis (Upstash) önbelleği: sorgu-imzası bazlı RAG önbelleği +
PubMed rate-limit sayacı (`raw/mimari/v45.txt` Bölüm 8.3, plan.txt satır
544).

⚠️ ÖNEMLİ BULGU (bu görevde ölçüldü, 2026-08-18): `.env`'de `REDIS_URL`
(protokol-tabanlı `rediss://` bağlantı dizesi) YOK -- yalnız
`UPSTASH_REDIS_REST_URL` + `UPSTASH_REDIS_REST_TOKEN` (Upstash'in HTTP
REST arayüzü, `raw/mimari/v45.txt` satır 545'te "Upstash Redis (REST)"
olarak zaten adlandırılmış). Bu modül bu yüzden `redis` paketini (protokol
istemcisi, `requirements.txt`'te zaten pinli, `api/main.py`'nin health-
check'i onu kullanıyor) DEĞİL, Upstash'in REST API'sini (`requests` ile)
kullanır. `api/main.py`'deki `redis.from_url(REDIS_URL)` health-check'i
mevcut `.env` ile ÇALIŞMAZ (`REDIS_URL` `None`) -- bu AYRI bir bulgu,
backend-agent'e/Barış'a bildirilmeli, bu görevin kapsamı DIŞINDA
düzeltilmedi.

Bağlantı canlı ÖLÇÜLDÜ (bu görevde, minimal `PING` isteğiyle): Upstash
REST erişilebilir, `PONG` döndü. Detay: final yanıt / log/2026-08-18.md.

Graceful degradation: Redis'e erişilemezse (ağ hatası, kimlik hatası,
zaman aşımı) bu modülün HİÇBİR fonksiyonu exception FIRLATMAZ -- önbellek
`None`/rate-limit sayacı `None` döner, çağıran taraf (`pipeline/
rag_pipeline.py`) bunu "önbellek yok, PubMed'e doğrudan git" olarak
yorumlar. Önbellek bir PERFORMANS katmanıdır, doğruluk/erişim GARANTİSİ
DEĞİLDİR -- bu yüzden sessiz düşüş (silent fallback) burada kabul
edilebilir (CLAUDE.md'nin "FAISS/PubMed sonucu sessizce boş dönmez"
kuralı ÖNBELLEĞE değil, ASIL veri kaynağına uygulanıyor)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

from db_connection import load_project_environment

_DEFAULT_TIMEOUT_SECONDS = 5
_DEFAULT_SUMMARY_TTL_SECONDS = 7 * 24 * 3600  # 1 hafta -- literatür yavaş değişir
_RATE_LIMIT_WINDOW_SECONDS = 1  # NCBI limiti saniye bazlı (3/sn veya 10/sn)


@dataclass(frozen=True)
class UpstashConfig:
    rest_url: str
    rest_token: str


def load_upstash_config() -> UpstashConfig | None:
    """`.env`'den Upstash REST kimlik bilgilerini okur. Eksikse `None`
    döner (hata FIRLATMAZ) -- çağıran taraf önbelleği pas geçer."""

    load_project_environment()
    url = os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    if not url or not token:
        return None
    return UpstashConfig(rest_url=url, rest_token=token)


def query_signature(text: str) -> str:
    """Sorgu-imzası: PubMed sorgu metninin sha256'sı -- HASTA KİMLİĞİNDEN
    BAĞIMSIZ (v45.txt satır 1341-1343: "benzer profilli farklı hastalar
    arasında da önbellek isabeti sağlar"). Girdi `patient_id` İÇERMEMELİDİR
    -- çağıran taraf `PubMedQueryResult.full_query` gibi kimlik-taşımayan
    bir metin vermelidir."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _upstash_request(config: UpstashConfig, *segments: str) -> Any | None:
    """Upstash REST komutunu çalıştırır. HERHANGİ bir hata (ağ/timeout/
    HTTP hata kodu/JSON ayrıştırma) `None` döner -- exception YUKARI
    SIZDIRILMAZ (bu modülün graceful-degradation sözleşmesi)."""

    try:
        import requests

        path = "/".join(requests.utils.quote(s, safe="") for s in segments)
        response = requests.get(
            f"{config.rest_url}/{path}",
            headers={"Authorization": f"Bearer {config.rest_token}"},
            timeout=_DEFAULT_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            return None
        return response.json().get("result")
    except Exception:
        return None


def check_upstash_connectivity() -> bool:
    """Upstash REST bağlantısını `PING` ile ölçer. Bu görevde ÇALIŞTIĞI
    doğrulandı (bkz. modül dokstring'i) -- bu fonksiyon canlı/tekrar
    doğrulama için tutulur, testlerde `requests`'i mock'lar."""

    config = load_upstash_config()
    if config is None:
        return False
    result = _upstash_request(config, "PING")
    return result == "PONG"


def get_cached_summary(signature: str) -> dict[str, Any] | None:
    """Önbellekte `rag:summary:<signature>` anahtarı varsa JSON olarak
    döner, yoksa/erişilemezse `None`."""

    config = load_upstash_config()
    if config is None:
        return None
    raw = _upstash_request(config, "GET", f"rag:summary:{signature}")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def set_cached_summary(
    signature: str,
    payload: dict[str, Any],
    *,
    ttl_seconds: int = _DEFAULT_SUMMARY_TTL_SECONDS,
) -> bool:
    """Önbelleğe `rag:summary:<signature>` -> `payload` (JSON, `ttl_seconds`
    ile sona erer) yazar. Başarısızlıkta `False` döner, exception ATMAZ."""

    config = load_upstash_config()
    if config is None:
        return False
    serialized = json.dumps(payload, ensure_ascii=False)
    result = _upstash_request(
        config, "SET", f"rag:summary:{signature}", serialized, "EX", str(ttl_seconds)
    )
    return result == "OK"


def increment_rate_limit_counter(
    bucket_key: str = "pubmed_requests_this_second",
    *,
    window_seconds: int = _RATE_LIMIT_WINDOW_SECONDS,
) -> int | None:
    """PubMed istek sayacını artırır (`INCR` + ilk artırımda `EXPIRE`).
    Döner: artırım sonrası sayaç değeri, veya Redis erişilemezse `None`
    (best-effort -- asıl limit koruması Biopython'ın kendi throttling'i +
    NCBI sunucu tarafı limitidir, bkz. `pipeline/rag_pubmed.py`)."""

    config = load_upstash_config()
    if config is None:
        return None
    key = f"rag:ratelimit:{bucket_key}"
    raw_value = _upstash_request(config, "INCR", key)
    if raw_value is None:
        return None
    try:
        new_value = int(raw_value)
    except (TypeError, ValueError):
        return None
    if new_value == 1:
        _upstash_request(config, "EXPIRE", key, str(window_seconds))
    return new_value
