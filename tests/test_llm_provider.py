"""`pipeline/llm_provider.py` testleri -- 2026-09-14, rag-agent-P5.

🔴 **HİÇBİR GERÇEK AĞ İSTEĞİ YOK.** `openai`/`anthropic` paketleri bu
makinede kurulu DEĞİL ve görev talimatı gerçek çağrıyı YASAKLIYOR. Bu
yüzden testler SAF katmanı zorlar: yapılandırma yükleme + istek gövdesi
üretimi + yanıt çözümleme. SDK'ya dokunan ince şerit (`_call_openai` /
`_call_anthropic` içindeki istemci kurulumu) **kasıtlı olarak test
edilmemiştir** ve bu, raporda AÇIK RİSK olarak beyan edilir -- "yazıldı"
ile "doğrulandı" burada aynı şey değildir.

Testlerin hepsi `env=` parametresiyle ÇALIŞIR: gerçek `.env` veya
`os.environ` OKUNMAZ, yani bir geliştiricinin makinesinde anahtar tanımlı
olması test sonucunu DEĞİŞTİRMEZ (ve testler anahtar sızdıramaz).
"""

from __future__ import annotations

import pytest

from pipeline import llm_provider
from pipeline.llm_provider import (
    ENV_ENABLED,
    ENV_MAX_OUTPUT_TOKENS,
    ENV_MAX_RETRIES,
    ENV_MODEL,
    ENV_PROVIDER,
    ENV_TEMPERATURE,
    ENV_TIMEOUT,
    LLMConfigError,
    LLMDisabledError,
    LLMProviderConfig,
    LLMProviderError,
    assert_model_is_pinned,
    build_anthropic_request,
    build_openai_request,
    describe_config,
    load_llm_config,
    parse_anthropic_response,
    parse_openai_response,
    resolve_api_key,
)

# Testlerde kullanilan SAHTE anahtar -- gercek bir anahtar DEGIL, hicbir
# yere gonderilmiyor. Sizinti testleri bu dizeyi arar.
_FAKE_KEY = "sk-TEST-ONLY-NOT-A-REAL-KEY-0000"


def _env(**overrides) -> dict[str, str]:
    base = {
        ENV_ENABLED: "true",
        ENV_PROVIDER: "openai",
        ENV_MODEL: "gpt-test-2026-01-01",
        "OPENAI_API_KEY": _FAKE_KEY,
    }
    base.update({k: v for k, v in overrides.items() if v is not None})
    for key, value in overrides.items():
        if value is None:
            base.pop(key, None)
    return base


# =====================================================================
# 1) VARSAYILAN KAPALI -- bugunku canli davranis DEGISMEMELI
# =====================================================================


def test_llm_is_disabled_by_default_when_flag_absent() -> None:
    """🔴 EN ÖNEMLİ TEST: bayrak hiç tanımlı değilken LLM KAPALI.
    Bu, "bugün canlı sistemde hiçbir şey bozulmamalı" şartının kod
    seviyesindeki karşılığıdır."""

    env = _env()
    env.pop(ENV_ENABLED)
    with pytest.raises(LLMDisabledError):
        load_llm_config(env)


@pytest.mark.parametrize("value", ["false", "0", "no", "", "  ", "off", "FALSE"])
def test_llm_stays_disabled_for_falsy_flag_values(value: str) -> None:
    with pytest.raises(LLMDisabledError):
        load_llm_config(_env(**{ENV_ENABLED: value}))


@pytest.mark.parametrize("value", ["true", "1", "yes", "on", "TRUE", "Evet"])
def test_flag_truthy_values_enable_config_loading(value: str) -> None:
    config = load_llm_config(_env(**{ENV_ENABLED: value}))
    assert config.provider == "openai"


def test_disabled_error_is_a_config_error_subclass() -> None:
    """`LLMDisabledError` ayrı bir tiptir ama `LLMConfigError`'dan türer --
    "bilinçli kapalı" ile "yapılandırma bozuk" ayırt edilebilsin, ama tek
    bir `except LLMConfigError` ikisini de yakalayabilsin."""

    assert issubclass(LLMDisabledError, LLMConfigError)


# =====================================================================
# 2) Saglayici / model / anahtar kapilari
# =====================================================================


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_both_providers_are_supported(provider: str) -> None:
    env = _env(**{ENV_PROVIDER: provider})
    env["ANTHROPIC_API_KEY"] = _FAKE_KEY
    config = load_llm_config(env)
    assert config.provider == provider
    assert config.api_key_env_var == {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }[provider]


def test_unknown_provider_is_rejected_and_lists_supported_ones() -> None:
    with pytest.raises(LLMConfigError) as exc:
        load_llm_config(_env(**{ENV_PROVIDER: "mistral"}))
    message = str(exc.value)
    assert "openai" in message and "anthropic" in message


def test_provider_name_is_case_insensitive_and_trimmed() -> None:
    assert load_llm_config(_env(**{ENV_PROVIDER: "  OpenAI  "})).provider == "openai"


def test_missing_api_key_is_rejected_naming_the_variable_not_the_value() -> None:
    env = _env()
    env.pop("OPENAI_API_KEY")
    with pytest.raises(LLMConfigError) as exc:
        load_llm_config(env)
    assert "OPENAI_API_KEY" in str(exc.value)


def test_blank_api_key_counts_as_missing() -> None:
    with pytest.raises(LLMConfigError):
        load_llm_config(_env(**{"OPENAI_API_KEY": "   "}))


# --- model adi: hareketli alias YASAK --------------------------------


@pytest.mark.parametrize(
    "model",
    ["gpt-4o-latest", "chatgpt-4o-latest", "claude-LATEST", "  some-latest-model  "],
)
def test_moving_alias_model_names_are_rejected(model: str) -> None:
    """Yeniden üretilebilirlik kapısı: `latest` alias'ı raporda "hangi
    sürümle koştuk" sorusunu cevapsız bırakır."""

    with pytest.raises(LLMConfigError) as exc:
        assert_model_is_pinned(model)
    assert "ALIAS" in str(exc.value).upper()


def test_empty_model_is_rejected_there_is_no_hardcoded_default() -> None:
    with pytest.raises(LLMConfigError) as exc:
        assert_model_is_pinned("")
    assert ENV_MODEL in str(exc.value)


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o-2024-08-06",  # OpenAI: tarihli snapshot
        "claude-opus-5",  # Anthropic: tarihsiz AMA SABIT kimlik
        "claude-sonnet-5",
    ],
)
def test_pinned_model_names_pass_including_anthropic_undated_ids(model: str) -> None:
    """⚠️ Kapı "tarih ZORUNLU" demez, "alias YASAK" der. Anthropic model
    kimliklerine tarih EKLENMEZ (eklemek yanlış olur) -- bu yüzden
    `claude-opus-5` GEÇERLİ bir sabit kimliktir. Bu testin varlığı, ileride
    biri kapıyı "tarih zorunlu"ya çevirmeye kalkarsa onu durdurur."""

    assert_model_is_pinned(model)  # hata firlatmamali


def test_load_llm_config_applies_the_model_pin_gate() -> None:
    with pytest.raises(LLMConfigError):
        load_llm_config(_env(**{ENV_MODEL: "gpt-4o-latest"}))


# =====================================================================
# 3) ANAHTAR SIZINTISI -- en kritik guvenlik testleri
# =====================================================================


def test_config_object_does_not_store_the_api_key_anywhere() -> None:
    """🔴 Yapısal garanti: anahtar `LLMProviderConfig`'te ALAN OLARAK
    YOKTUR. `api/analyze_patient.py` sonuç ağacına `dataclasses.asdict()`
    uygular; config bir gün o ağaca yanlışlıkla girerse bile anahtar
    HTTP yanıtına sızamaz -- çünkü orada yok."""

    config = load_llm_config(_env())
    assert _FAKE_KEY not in repr(config)
    assert _FAKE_KEY not in str(config)
    assert not any(_FAKE_KEY in str(v) for v in vars(config).values())
    assert "api_key" not in vars(config), "config anahtarin KENDISINI tasimamali"
    assert config.api_key_env_var == "OPENAI_API_KEY"


def test_describe_config_is_loggable_and_contains_no_key() -> None:
    """Loglanan TEK biçim. Sağlayıcı adı ve model adı GÖRÜNÜR (görev
    talimatı bunu açıkça istiyor), anahtar GÖRÜNMEZ."""

    text = describe_config(load_llm_config(_env()))
    assert "provider=openai" in text
    assert "model=gpt-test-2026-01-01" in text
    assert "OPENAI_API_KEY" in text  # degiskenin ADI -- degeri degil
    assert _FAKE_KEY not in text


def test_missing_key_error_message_contains_no_value() -> None:
    config = load_llm_config(_env())
    with pytest.raises(LLMConfigError) as exc:
        resolve_api_key(config, {})
    assert _FAKE_KEY not in str(exc.value)


def test_resolve_api_key_reads_from_the_named_variable_at_call_time() -> None:
    config = load_llm_config(_env())
    assert resolve_api_key(config, {"OPENAI_API_KEY": _FAKE_KEY}) == _FAKE_KEY


def test_built_requests_never_carry_the_api_key() -> None:
    """İstek gövdesinde anahtar bulunmaz -- anahtar SDK istemcisine
    verilir, gövdeye değil. (Gövde loglanırsa/kaydedilirse diye.)"""

    config = load_llm_config(_env())
    for builder in (build_openai_request, build_anthropic_request):
        body = builder(config, system_prompt="S", user_prompt="U")
        assert _FAKE_KEY not in str(body)


# =====================================================================
# 4) Muhafazakar varsayilanlar + sinir dogrulama (10 USD sert limit)
# =====================================================================


def test_conservative_defaults_when_tuning_vars_are_absent() -> None:
    config = load_llm_config(_env())
    assert config.timeout_seconds == llm_provider.DEFAULT_TIMEOUT_SECONDS == 60.0
    assert config.max_retries == llm_provider.DEFAULT_MAX_RETRIES == 1
    assert config.max_output_tokens == llm_provider.DEFAULT_MAX_OUTPUT_TOKENS == 900
    assert config.temperature is None, "temperature acikca ayarlanmadikca GONDERILMEZ"


def test_tuning_vars_are_read_when_present() -> None:
    config = load_llm_config(
        _env(
            **{
                ENV_TIMEOUT: "12.5",
                ENV_MAX_RETRIES: "0",
                ENV_MAX_OUTPUT_TOKENS: "256",
                ENV_TEMPERATURE: "0.2",
            }
        )
    )
    assert (config.timeout_seconds, config.max_retries) == (12.5, 0)
    assert (config.max_output_tokens, config.temperature) == (256, 0.2)


@pytest.mark.parametrize(
    "key,value",
    [
        (ENV_MAX_RETRIES, "99"),  # her yeniden deneme PARA
        (ENV_MAX_RETRIES, "-1"),
        (ENV_MAX_OUTPUT_TOKENS, "999999"),  # kacak uretim = fatura
        (ENV_MAX_OUTPUT_TOKENS, "1"),
        (ENV_TIMEOUT, "0"),
        (ENV_TIMEOUT, "99999"),
    ],
)
def test_out_of_range_tuning_values_are_rejected_not_silently_clamped(
    key: str, value: str
) -> None:
    """Sessizce kırpmak (clamp) bir "sessiz düzeltme"dir: kullanıcı 99
    yazıp 5 alırsa faturayı yanlış öngörür. Fail-loud."""

    with pytest.raises(LLMConfigError) as exc:
        load_llm_config(_env(**{key: value}))
    assert key in str(exc.value)


@pytest.mark.parametrize("key", [ENV_MAX_RETRIES, ENV_MAX_OUTPUT_TOKENS, ENV_TIMEOUT])
def test_non_numeric_tuning_values_are_rejected(key: str) -> None:
    with pytest.raises(LLMConfigError):
        load_llm_config(_env(**{key: "cok"}))


# =====================================================================
# 5) Istek govdesi (SAF -- SDK gerekmez)
# =====================================================================


def _config(**overrides) -> LLMProviderConfig:
    base = dict(
        provider="openai",
        model="m-2026-01-01",
        api_key_env_var="OPENAI_API_KEY",
        max_output_tokens=321,
    )
    base.update(overrides)
    return LLMProviderConfig(**base)


def test_openai_request_shape() -> None:
    body = build_openai_request(_config(), system_prompt="SYS", user_prompt="USR")
    assert body["model"] == "m-2026-01-01"
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]
    assert body["max_completion_tokens"] == 321
    assert body["response_format"] == {"type": "json_object"}


def test_anthropic_request_puts_system_at_top_level_not_in_messages() -> None:
    """⚠️ Anthropic sözleşmesi: `system` TOP-LEVEL alandır. Onu
    `messages`'a koymak sessiz bir davranış farkı üretirdi."""

    body = build_anthropic_request(
        _config(provider="anthropic", api_key_env_var="ANTHROPIC_API_KEY"),
        system_prompt="SYS",
        user_prompt="USR",
    )
    assert body["system"] == "SYS"
    assert body["messages"] == [{"role": "user", "content": "USR"}]
    assert body["max_tokens"] == 321, "Anthropic'te max_tokens ZORUNLU alandir"
    assert "system" not in str(body["messages"])


@pytest.mark.parametrize("builder", [build_openai_request, build_anthropic_request])
def test_temperature_is_omitted_unless_explicitly_configured(builder) -> None:
    """🔴 Ölçülmüş gerekçe: güncel modellerin bir kısmı `temperature`
    parametresini REDDEDİYOR (HTTP 400). "Her ihtimale karşı gönder"
    yaklaşımı sessiz bir kırılma üretirdi."""

    assert "temperature" not in builder(_config(), system_prompt="S", user_prompt="U")
    body = builder(_config(temperature=0.3), system_prompt="S", user_prompt="U")
    assert body["temperature"] == 0.3


def test_effort_is_only_sent_for_anthropic_when_configured() -> None:
    cfg = _config(provider="anthropic", api_key_env_var="ANTHROPIC_API_KEY", effort="low")
    body = build_anthropic_request(cfg, system_prompt="S", user_prompt="U")
    assert body["output_config"] == {"effort": "low"}
    assert "output_config" not in build_anthropic_request(
        _config(provider="anthropic", api_key_env_var="ANTHROPIC_API_KEY"),
        system_prompt="S",
        user_prompt="U",
    )


# =====================================================================
# 6) Yanit cozumleme (SAF -- SDK gerekmez)
# =====================================================================


def test_parse_openai_response_extracts_text_model_and_usage() -> None:
    payload = {
        "model": "m-2026-01-01",
        "choices": [{"message": {"content": '{"summary_tr": "x"}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 111, "completion_tokens": 22},
    }
    parsed = parse_openai_response(payload)
    assert parsed.text == '{"summary_tr": "x"}'
    assert (parsed.model_name, parsed.stop_reason) == ("m-2026-01-01", "stop")
    assert (parsed.input_tokens, parsed.output_tokens) == (111, 22)


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {},
        {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
        {"choices": [{"message": {}, "finish_reason": "stop"}]},
    ],
)
def test_parse_openai_response_fails_loud_on_empty_text(payload: dict) -> None:
    """Boş metni sessizce `""` olarak geçirmek, aşağıdaki JSON
    çözümleyicide anlamsız bir hataya dönüşürdü. Burada fail-loud."""

    with pytest.raises(LLMProviderError):
        parse_openai_response(payload)


def test_parse_anthropic_response_joins_only_text_blocks() -> None:
    """⚠️ `content` blok listesidir ve metin-dışı bloklar (ör. `thinking`)
    içerebilir. `content[0].text` okumak düşünme açıkken sessizce
    patlardı -- bu test o hatayı kalıcı olarak engeller."""

    payload = {
        "model": "claude-test",
        "content": [
            {"type": "thinking", "thinking": "IC MUHAKEME -- metin DEGIL"},
            {"type": "text", "text": '{"summary_tr":'},
            {"type": "text", "text": ' "x"}'},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 9, "output_tokens": 8},
    }
    parsed = parse_anthropic_response(payload)
    assert parsed.text == '{"summary_tr": "x"}'
    assert "MUHAKEME" not in parsed.text
    assert (parsed.input_tokens, parsed.output_tokens) == (9, 8)


@pytest.mark.parametrize(
    "payload",
    [
        {"content": []},
        {"content": [{"type": "thinking", "thinking": "sadece dusunme"}]},
        {"content": [{"type": "text", "text": "   "}]},
    ],
)
def test_parse_anthropic_response_fails_loud_on_no_text(payload: dict) -> None:
    with pytest.raises(LLMProviderError):
        parse_anthropic_response(payload)


def test_parse_accepts_pydantic_like_objects_via_model_dump() -> None:
    """SDK'lar pydantic modeli döndürür. Çözümleme SÖZLÜK üzerinden
    yapıldığı için `model_dump()` yeterlidir -- SDK kurulu olmadan test
    edebilmemizin sebebi budur."""

    class _FakeSdkResponse:
        def model_dump(self):
            return {
                "model": "m",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            }

    assert parse_openai_response(_FakeSdkResponse()).text == "ok"


def test_unconvertible_response_object_fails_loud() -> None:
    class _Opaque:
        pass

    with pytest.raises(LLMProviderError):
        parse_openai_response(_Opaque())


# =====================================================================
# 7) Sevk (dispatch) -- hangi saglayici secildi
# =====================================================================


def test_invoke_provider_dispatches_to_the_configured_provider(monkeypatch) -> None:
    """Sağlayıcı seçimi `GBMAID_LLM_PROVIDER`'dan gelir ve `_DISPATCH`
    tablosu üzerinden çözülür -- `rag_llm.py` sağlayıcı adını HİÇ
    bilmez. Yeni sağlayıcı eklemek bu tabloya bir satır demektir."""

    seen: list[str] = []

    def _fake(config, api_key, *, system_prompt, user_prompt):
        seen.append(config.provider)
        assert api_key == _FAKE_KEY
        return llm_provider.ProviderResponse(text="ok", model_name=config.model)

    monkeypatch.setitem(llm_provider._DISPATCH, "anthropic", _fake)
    env = _env(**{ENV_PROVIDER: "anthropic"})
    env["ANTHROPIC_API_KEY"] = _FAKE_KEY

    response = llm_provider.invoke_provider(
        load_llm_config(env), system_prompt="S", user_prompt="U", env=env
    )
    assert seen == ["anthropic"]
    assert response.text == "ok"


def test_provider_selection_is_logged_without_the_key(monkeypatch, caplog) -> None:
    """Görev talimatı: "hangi sağlayıcının seçildiğini kod AÇIKÇA
    loglasın (anahtarı değil, yalnız sağlayıcı adını ve model adını)."""

    monkeypatch.setitem(
        llm_provider._DISPATCH,
        "openai",
        lambda c, k, *, system_prompt, user_prompt: llm_provider.ProviderResponse(
            text="ok", model_name=c.model
        ),
    )
    env = _env()
    with caplog.at_level("INFO", logger="pipeline.rag.llm"):
        llm_provider.invoke_provider(
            load_llm_config(env), system_prompt="S", user_prompt="U", env=env
        )

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "provider=openai" in logged
    assert "gpt-test-2026-01-01" in logged
    assert _FAKE_KEY not in logged
