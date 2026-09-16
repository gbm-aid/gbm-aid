"""Hafta 5 — SAĞLAYICI-BAĞIMSIZ LLM adaptör katmanı (2026-09-14, rag-agent-P5).

NEDEN BU DOSYA VAR
==================
`pipeline/rag_llm.py::_invoke_llm_provider()` 2026-09-13'e kadar **her
durumda** `LLMNotConfiguredError` fırlatıyordu -- gerçek sağlayıcı çağrı
kodu HİÇ YAZILMAMIŞTI. Bu dosya o boşluğu doldurur.

🔴 **SAĞLAYICI KARARI BARIŞ'TA VE HENÜZ VERİLMEDİ** (hafta5 §2.1, D3).
Bu yüzden kod TEK bir sağlayıcıya bağlanmaz: `GBMAID_LLM_PROVIDER` ortam
değişkeni (`openai` | `anthropic`) hangi adaptörün koşacağını seçer.
Barış hangi anahtarı verirse versin **yeniden yazım gerekmez** -- yalnız
`.env`'de iki satır değişir.

GÜVENLİK SÖZLEŞMESİ (bu dosyanın değişmez kuralları)
====================================================
1. **API anahtarı hiçbir dataclass'ta SAKLANMAZ.** `LLMProviderConfig`
   yalnız anahtarın OKUNACAĞI ORTAM DEĞİŞKENİNİN ADINI taşır
   (`api_key_env_var`), değerini değil. Gerekçe yapısal: `api/
   analyze_patient.py::_build_literature_block()` sonuç ağacına
   `dataclasses.asdict()` uygular; config nesnesi o ağaca bir gün
   yanlışlıkla eklenirse, anahtar alanı OLMADIĞI için HTTP yanıtına
   sızamaz. "Dikkat ederiz" yerine "orada yok".
2. **Anahtar hiçbir log/exception mesajına GİRMEZ.** Hata mesajları
   yalnız DEĞİŞKEN ADINI söyler (`OPENAI_API_KEY tanimli degil`), değeri
   asla. `describe_config()` bilinçli olarak anahtar-içermeyen tek
   satırlık bir özet döner ve loglanacak TEK biçimdir.
   (2026-09-13'te bir ajan DB şifresini terminale bastı; redaksiyon
   regex'i URL'ye gömülü şifreyi kaçırdı. Buradaki çözüm regex değil:
   değer hiç taşınmıyor.)
3. **Varsayılan KAPALI.** `GBMAID_LLM_ENABLED` tanımsız/`false` iken
   `load_llm_config()` `LLMDisabledError` fırlatır ve çağrı zinciri
   2026-09-13'teki davranışın BİREBİR AYNISINA döner.

MODEL ADI SABİTLEME (`latest` alias YASAK)
==========================================
`GBMAID_LLM_MODEL` zorunludur ve kodda GÖMÜLÜ VARSAYILANI YOKTUR --
raporda "hangi model sürümüyle koşuldu" sorusunun cevabı `.env`'den ve
`run` çıktısından okunabilmelidir (yeniden üretilebilirlik).
`assert_model_is_pinned()` içinde `latest` içeren her model adı REDDEDİLİR.

⚠️ **ÖLÇÜLMÜŞ NÜANS -- "tarihli snapshot" iki sağlayıcıda AYNI ŞEY DEĞİL:**
  * **OpenAI:** tarihli snapshot'lar gerçektir (`...-2024-08-06` gibi) ve
    `-latest` alias'ları vardır -> alias YASAĞI burada birebir anlamlıdır.
  * **Anthropic:** güncel model kimlikleri ZATEN sabittir ve tarih soneki
    ALMAZ (`claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5`).
    Bunlara tarih eklemek YANLIŞ olur (kaynak: `claude-api` skill'i,
    "use only the exact model ID strings ... never append date suffixes").
Bu nedenle kapı "tarih ZORUNLU" demez, **"alias YASAK"** der. Fark
raporda beyan edilmelidir.

TEST EDİLEBİLİRLİK / BU DOSYANIN AÇIK RİSKİ
===========================================
Sağlayıcı SDK'ları (`openai`, `anthropic`) bu makinede **KURULU DEĞİL**
(2026-09-14'te ölçüldü) ve bu görevde gerçek ağ çağrısı YASAK. Bu yüzden
SDK'ya dokunan kod MÜMKÜN OLDUĞUNCA İNCE tutulmuş, istek gövdesi üretimi
ve yanıt çözümlemesi SAF fonksiyonlara ayrılmıştır
(`build_*_request` / `parse_*_response`) -- bunlar SDK olmadan test
edilir ve testlidir.
🔴 **TEST EDİLMEYEN TEK ŞERİT:** `_call_openai` / `_call_anthropic`
içindeki ~6 satırlık SDK çağrısının KENDİSİ (istemci kurulumu + metot
adı). Bu şerit ancak Barış anahtarı verdikten sonraki İLK GERÇEK
KOŞUDA doğrulanabilir. "Yazıldı" != "doğrulandı" -- raporda böyle geçer.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_logger = logging.getLogger("pipeline.rag.llm")

# --------------------------------------------------------------------------
# Ortam değişkeni adları -- TEK yerde, string tekrarı yok.
# --------------------------------------------------------------------------
ENV_ENABLED = "GBMAID_LLM_ENABLED"
ENV_PROVIDER = "GBMAID_LLM_PROVIDER"
ENV_MODEL = "GBMAID_LLM_MODEL"
ENV_TIMEOUT = "GBMAID_LLM_TIMEOUT_SECONDS"
ENV_MAX_RETRIES = "GBMAID_LLM_MAX_RETRIES"
ENV_MAX_OUTPUT_TOKENS = "GBMAID_LLM_MAX_OUTPUT_TOKENS"
ENV_TEMPERATURE = "GBMAID_LLM_TEMPERATURE"
ENV_EFFORT = "GBMAID_LLM_EFFORT"

PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
SUPPORTED_PROVIDERS: tuple[str, ...] = (PROVIDER_OPENAI, PROVIDER_ANTHROPIC)

#: Hangi sağlayıcı hangi ortam değişkeninden anahtar okur. DEĞER burada
#: değil, yalnız ADI.
API_KEY_ENV_BY_PROVIDER: Mapping[str, str] = {
    PROVIDER_OPENAI: "OPENAI_API_KEY",
    PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
}

# MUHAFAZAKÂR varsayılanlar -- Ege'nin hesabında 10 USD SERT LİMİT var
# (görev talimatı). Hiçbiri "bol keseden" seçilmedi:
#   * 60 s  : tek bir kısa özet için fazlasıyla yeterli; asılı kalan bir
#             istek isteği 10 dk boyunca tutmasın diye SDK varsayılanı
#             (10 dk) BİLİNÇLİ olarak kısaltıldı.
#   * 1     : toplam EN FAZLA 2 deneme. Her yeniden deneme PARA demek;
#             SDK varsayılanı 2 (=3 deneme) bu bütçede fazla.
#   * 900   : ~200 kelimelik Türkçe özet + JSON zarfı rahat sığar;
#             kaçak bir "sonsuz üretim" faturayı büyütmesin diye tavan.
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_MAX_OUTPUT_TOKENS = 900

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
_ENV_FILE: Path = _PROJECT_ROOT / ".env"
_dotenv_lock = threading.Lock()
_dotenv_loaded = False


class LLMConfigError(RuntimeError):
    """LLM yapılandırması EKSİK/GEÇERSİZ. Mesaj yalnız DEĞİŞKEN ADI
    söyler, hiçbir koşulda anahtar DEĞERİ içermez."""


class LLMDisabledError(LLMConfigError):
    """`GBMAID_LLM_ENABLED` açık değil -- bu bir HATA DEĞİL, KASITLI
    kapalılık. Ayrı tip olmasının sebebi: çağıran taraf "yapılandırma
    bozuk" ile "bilinçli kapalı"yı ayırt edebilsin."""


class LLMProviderError(RuntimeError):
    """Sağlayıcı çağrısı BAŞARISIZ (ağ/kota/HTTP/SDK yok). Bu bir
    TEMELLENDİRME ihlali DEĞİLDİR -- `CitationGroundingError` ile
    karıştırılmamalıdır, çünkü ikisi pipeline'da FARKLI şekilde ele
    alınır (bkz. `pipeline/rag_pipeline.py`)."""


@dataclass(frozen=True)
class LLMProviderConfig:
    """LLM çağrısının TÜM ayarları -- **API anahtarı HARİÇ** (bkz. modül
    dokstring'i, güvenlik sözleşmesi madde 1)."""

    provider: str
    model: str
    api_key_env_var: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float | None = None
    effort: str | None = None


@dataclass(frozen=True)
class ProviderResponse:
    """Sağlayıcıdan dönen HAM yanıtın sağlayıcı-bağımsız karşılığı.
    `input_tokens`/`output_tokens` maliyet ölçümü içindir (ilk gerçek
    koşuda "token/süre/maliyet" raporlanacak, görev talimatı ÇIKTI-4)."""

    text: str
    model_name: str
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


# ==========================================================================
# 1) Yapılandırma yükleme
# ==========================================================================


def _load_dotenv_once() -> None:
    """Proje KÖKÜNDEKİ `.env`'i bir kez yükler (CLAUDE.md: kanonik `.env`
    proje kökündedir, `gbm-aid mert/.env` DEĞİL). Dosya yoksa SESSİZCE
    geçer -- LLM kapalı çalışmak meşru bir durumdur, `db_connection.py`
    gibi `FileNotFoundError` fırlatmak burada yanlış olurdu."""

    global _dotenv_loaded
    if _dotenv_loaded:
        return
    with _dotenv_lock:
        if _dotenv_loaded:
            return
        if _ENV_FILE.is_file():
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=_ENV_FILE, override=False)
            except Exception:  # dotenv yoksa/bozuksa ortam değişkenleri yine okunur
                _logger.debug("dotenv yuklenemedi; yalniz os.environ kullanilacak")
        _dotenv_loaded = True


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on", "evet"}


def _read_int(env: Mapping[str, str], key: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise LLMConfigError(f"{key} tamsayi olmali, cozulemedi.") from exc
    if not (minimum <= value <= maximum):
        raise LLMConfigError(f"{key} {minimum}-{maximum} araliginda olmali (okunan: {value}).")
    return value


def _read_float(env: Mapping[str, str], key: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise LLMConfigError(f"{key} sayi olmali, cozulemedi.") from exc
    if not (minimum <= value <= maximum):
        raise LLMConfigError(f"{key} {minimum}-{maximum} araliginda olmali (okunan: {value}).")
    return value


def assert_model_is_pinned(model: str) -> None:
    """Model adı SABİT bir sürümü göstermeli -- hareketli alias YASAK.

    ⚠️ Kapı "tarih ZORUNLU" DEMEZ (modül dokstring'indeki nüans: Anthropic
    kimlikleri tarih soneki almaz, eklemek yanlış olur). Kapı **alias**
    reddeder: adında `latest` geçen her model. Boş ad da reddedilir --
    kodda GÖMÜLÜ VARSAYILAN YOKTUR."""

    cleaned = (model or "").strip()
    if not cleaned:
        raise LLMConfigError(
            f"{ENV_MODEL} bos. Kodda GOMULU varsayilan model YOKTUR "
            "(yeniden uretilebilirlik: rapora hangi surumun kosuldugu "
            "yazilacak) -- .env'de acikca belirtilmelidir."
        )
    if "latest" in cleaned.lower():
        raise LLMConfigError(
            f"{ENV_MODEL}='{cleaned}' bir HAREKETLI ALIAS. Yeniden "
            "uretilebilirlik icin sabit bir surum kimligi gerekir "
            "(OpenAI'de tarihli snapshot, Anthropic'te tarihsiz ama SABIT "
            "model kimligi -- Anthropic kimliklerine tarih EKLENMEZ)."
        )


def load_llm_config(env: Mapping[str, str] | None = None) -> LLMProviderConfig:
    """`.env` + ortamdan `LLMProviderConfig` üretir.

    Fırlatır:
      * `LLMDisabledError` -- `GBMAID_LLM_ENABLED` açık değil (VARSAYILAN).
      * `LLMConfigError`   -- sağlayıcı/model/anahtar eksik veya geçersiz.

    ⚠️ Anahtarın VARLIĞI kontrol edilir, DEĞERİ okunmaz/dönmez/loglanmaz.
    """

    if env is None:
        _load_dotenv_once()
        env = os.environ

    if not _truthy(env.get(ENV_ENABLED)):
        raise LLMDisabledError(
            f"LLM cagrisi KAPALI ({ENV_ENABLED} tanimsiz/false). Bu bir hata "
            "degil, varsayilan durumdur -- acmak icin .env'de "
            f"{ENV_ENABLED}=true yapilmalidir."
        )

    provider = (env.get(ENV_PROVIDER) or "").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise LLMConfigError(
            f"{ENV_PROVIDER}='{provider}' desteklenmiyor. Desteklenenler: "
            f"{', '.join(SUPPORTED_PROVIDERS)}."
        )

    model = (env.get(ENV_MODEL) or "").strip()
    assert_model_is_pinned(model)

    api_key_env_var = API_KEY_ENV_BY_PROVIDER[provider]
    if not (env.get(api_key_env_var) or "").strip():
        raise LLMConfigError(
            f"{api_key_env_var} tanimli degil veya bos (saglayici: {provider}). "
            "Anahtari .env'e BARIS ekler; bu kod yalnizca VARLIGINI kontrol "
            "eder, degerini okumaz/loglamaz."
        )

    temperature_raw = (env.get(ENV_TEMPERATURE) or "").strip()
    temperature = (
        _read_float(env, ENV_TEMPERATURE, 0.0, minimum=0.0, maximum=2.0)
        if temperature_raw
        else None
    )
    effort = (env.get(ENV_EFFORT) or "").strip() or None

    return LLMProviderConfig(
        provider=provider,
        model=model,
        api_key_env_var=api_key_env_var,
        timeout_seconds=_read_float(
            env, ENV_TIMEOUT, DEFAULT_TIMEOUT_SECONDS, minimum=1.0, maximum=600.0
        ),
        max_retries=_read_int(env, ENV_MAX_RETRIES, DEFAULT_MAX_RETRIES, minimum=0, maximum=5),
        max_output_tokens=_read_int(
            env, ENV_MAX_OUTPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS, minimum=64, maximum=8192
        ),
        temperature=temperature,
        effort=effort,
    )


def describe_config(config: LLMProviderConfig) -> str:
    """Loglanabilir TEK biçim. Anahtar İÇERMEZ -- yalnız değişken ADI
    geçer, değeri asla."""

    return (
        f"provider={config.provider} model={config.model} "
        f"api_key_env={config.api_key_env_var} "
        f"timeout={config.timeout_seconds}s retries={config.max_retries} "
        f"max_output_tokens={config.max_output_tokens} "
        f"temperature={config.temperature} effort={config.effort}"
    )


def resolve_api_key(config: LLMProviderConfig, env: Mapping[str, str] | None = None) -> str:
    """Anahtarı ÇAĞRI ANINDA ortamdan okur. Dönen değer YALNIZ SDK
    istemcisine verilir -- loglanmaz, dataclass'a konmaz, dönüş
    nesnelerine girmez."""

    if env is None:
        _load_dotenv_once()
        env = os.environ
    key = (env.get(config.api_key_env_var) or "").strip()
    if not key:
        raise LLMConfigError(f"{config.api_key_env_var} tanimli degil veya bos.")
    return key


# ==========================================================================
# 2) SAF fonksiyonlar -- istek gövdesi + yanıt çözümleme (SDK'sız TEST EDİLİR)
# ==========================================================================


def build_openai_request(
    config: LLMProviderConfig, *, system_prompt: str, user_prompt: str
) -> dict[str, Any]:
    """OpenAI Chat Completions istek gövdesi.

    ⚠️ `temperature` YALNIZ açıkça ayarlandıysa gönderilir: bazı güncel
    modeller bu parametreyi REDDEDİYOR (HTTP 400). "Her ihtimale karşı
    gönder" yaklaşımı burada sessiz bir kırılma üretirdi."""

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": config.max_output_tokens,
        # Yanit JSON olmali -- prompt'ta da "JSON" kelimesi geciyor
        # (OpenAI json_object modu bunu ZORUNLU kilar).
        "response_format": {"type": "json_object"},
    }
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    return payload


def build_anthropic_request(
    config: LLMProviderConfig, *, system_prompt: str, user_prompt: str
) -> dict[str, Any]:
    """Anthropic Messages istek gövdesi.

    ⚠️ `system` TOP-LEVEL alandır (messages içinde DEĞİL) -- Anthropic
    sözleşmesi böyle. `max_tokens` ZORUNLUDUR.
    ⚠️ `temperature` yalnız açıkça ayarlandıysa gönderilir: güncel
    modellerde (4.6+) bu parametre KALDIRILDI ve 400 döner."""

    payload: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_output_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    if config.effort:
        payload["output_config"] = {"effort": config.effort}
    return payload


def _as_dict(raw: Any) -> dict[str, Any]:
    """SDK yanıtını (pydantic modeli) düz sözlüğe indirger. İki SDK da
    `model_dump()` sağlar; `to_dict()` yedek yoldur. Çözümlemeyi SÖZLÜK
    üzerinden yapmak, SDK kurulu OLMADAN test yazabilmemizin sebebidir."""

    if isinstance(raw, dict):
        return raw
    for attr in ("model_dump", "to_dict"):
        fn = getattr(raw, attr, None)
        if callable(fn):
            try:
                value = fn()
            except Exception:  # pragma: no cover -- SDK surum farki
                continue
            if isinstance(value, dict):
                return value
    raise LLMProviderError(
        f"Saglayici yaniti sozluge cevrilemedi (tip: {type(raw).__name__}). "
        "SDK surumu beklenmedik bir yanit nesnesi dondurdu."
    )


def parse_openai_response(raw: Any) -> ProviderResponse:
    """OpenAI yanıtından metin + model adı + token sayıları çıkarır.
    Eksik/boş metin SESSİZCE "" olarak geçirilmez -- `LLMProviderError`."""

    data = _as_dict(raw)
    choices = data.get("choices") or []
    if not choices:
        raise LLMProviderError("OpenAI yaniti bos: 'choices' YOK/bos.")
    message = (choices[0] or {}).get("message") or {}
    text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        raise LLMProviderError(
            "OpenAI yanitinda metin YOK (choices[0].message.content bos). "
            f"finish_reason={(choices[0] or {}).get('finish_reason')!r}"
        )
    usage = data.get("usage") or {}
    return ProviderResponse(
        text=text,
        model_name=str(data.get("model") or ""),
        stop_reason=(choices[0] or {}).get("finish_reason"),
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
    )


def parse_anthropic_response(raw: Any) -> ProviderResponse:
    """Anthropic yanıtından metin + model adı + token sayıları çıkarır.

    ⚠️ `content` bir BLOK LİSTESİDİR ve metin dışı bloklar da içerebilir
    (`thinking` vb.) -- yalnız `type == "text"` blokları birleştirilir.
    Bunu atlayıp `content[0].text` okumak, düşünme açıkken sessizce
    patlardı."""

    data = _as_dict(raw)
    blocks = data.get("content") or []
    if not isinstance(blocks, list):
        raise LLMProviderError("Anthropic yanitinda 'content' liste degil.")
    parts = [
        b.get("text", "")
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    text = "".join(parts)
    if not text.strip():
        raise LLMProviderError(
            "Anthropic yanitinda metin blogu YOK. "
            f"stop_reason={data.get('stop_reason')!r}"
        )
    usage = data.get("usage") or {}
    return ProviderResponse(
        text=text,
        model_name=str(data.get("model") or ""),
        stop_reason=data.get("stop_reason"),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )


# ==========================================================================
# 3) Gerçek SDK çağrıları -- İNCE ŞERİT (test edilmeyen tek yer)
# ==========================================================================


def _call_openai(
    config: LLMProviderConfig, api_key: str, *, system_prompt: str, user_prompt: str
) -> ProviderResponse:
    try:
        import openai  # lazy: paket kurulu olmasa da modul import edilebilir
    except ImportError as exc:
        raise LLMProviderError(
            "'openai' paketi kurulu degil. Kurulum: pip install openai "
            "(surum .env kurulum notunda pinlenmelidir)."
        ) from exc

    client = openai.OpenAI(
        api_key=api_key,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )
    try:
        raw = client.chat.completions.create(
            **build_openai_request(config, system_prompt=system_prompt, user_prompt=user_prompt)
        )
    except Exception as exc:  # ag/kota/HTTP -- mesaj anahtar ICERMEZ
        raise LLMProviderError(f"OpenAI cagrisi basarisiz: {type(exc).__name__}: {exc}") from exc
    return parse_openai_response(raw)


def _call_anthropic(
    config: LLMProviderConfig, api_key: str, *, system_prompt: str, user_prompt: str
) -> ProviderResponse:
    try:
        import anthropic  # lazy
    except ImportError as exc:
        raise LLMProviderError(
            "'anthropic' paketi kurulu degil. Kurulum: pip install anthropic "
            "(surum .env kurulum notunda pinlenmelidir)."
        ) from exc

    client = anthropic.Anthropic(
        api_key=api_key,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )
    try:
        raw = client.messages.create(
            **build_anthropic_request(
                config, system_prompt=system_prompt, user_prompt=user_prompt
            )
        )
    except Exception as exc:  # ag/kota/HTTP -- mesaj anahtar ICERMEZ
        raise LLMProviderError(
            f"Anthropic cagrisi basarisiz: {type(exc).__name__}: {exc}"
        ) from exc
    return parse_anthropic_response(raw)


#: Sağlayıcı adı -> uygulama. Yeni bir sağlayıcı eklemek = buraya bir
#: satır + `API_KEY_ENV_BY_PROVIDER`'a bir satır. `rag_llm.py` DEĞİŞMEZ.
_DISPATCH = {
    PROVIDER_OPENAI: _call_openai,
    PROVIDER_ANTHROPIC: _call_anthropic,
}


def invoke_provider(
    config: LLMProviderConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    env: Mapping[str, str] | None = None,
) -> ProviderResponse:
    """Seçilen sağlayıcıyı çağırır.

    Hangi sağlayıcı/model seçildiği AÇIKÇA loglanır (`describe_config`
    -- anahtar İÇERMEZ, görev talimatı: "kod açıkça loglasın, anahtarı
    değil yalnız sağlayıcı adını ve model adını")."""

    handler = _DISPATCH.get(config.provider)
    if handler is None:  # pragma: no cover -- load_llm_config zaten dogrular
        raise LLMConfigError(f"Desteklenmeyen saglayici: {config.provider}")

    _logger.info("LLM cagrisi baslatiliyor -- %s", describe_config(config))
    api_key = resolve_api_key(config, env)
    response = handler(config, api_key, system_prompt=system_prompt, user_prompt=user_prompt)
    _logger.info(
        "LLM cagrisi tamamlandi -- provider=%s model=%s stop_reason=%s "
        "input_tokens=%s output_tokens=%s",
        config.provider,
        response.model_name or config.model,
        response.stop_reason,
        response.input_tokens,
        response.output_tokens,
    )
    return response
