"""`pipeline/rag_llm.py` PMID TEMELLENDİRME (grounding) kapısı testleri
-- 2026-09-13, rag-agent-C (Barış onayı / karar 6).

HİÇBİR GERÇEK API ÇAĞRISI YOK. Sahte LLM yanıtı, `rag_llm._invoke_llm_
provider`'ın monkeypatch'lenmesiyle üretilir -- yani testler guard'ı
DOĞRUDAN değil, GERÇEK public yol (`call_llm_summarize`) üzerinden
zorlar. Bu bilinçlidir: "guard atlanabilir mi" sorusunun cevabı ancak
public yol üzerinden alınabilir.

Test omurgası (görev talimatı):
  * KIRMIZI: retrieved kümede OLMAYAN bir PMID -> `FabricatedPmidError`.
  * YEŞİL: yalnız retrieved kümedeki PMID'ler -> özet DÖNER.
  * TAUTOLOJİ KONTROLÜ: guard'ın gerçekten bir şey yakaladığını
    kanıtlamak için, guard'ı YAPMAYAN bir "no-op" referans uygulama
    aynı sahte yanıtla koşulur ve o uygulamanın uydurmayı GEÇİRDİĞİ
    gösterilir (bkz. `tests/test_xgboost_model.py::test_old_misaligned_
    pipeline_...` deseni -- "yeni yolun çözdüğü somut sorunu KANITLA").
"""

from __future__ import annotations

import pytest

from pipeline import rag_llm
from pipeline.rag_llm import (
    CitationGroundingError,
    FabricatedPmidError,
    LLMNotConfiguredError,
    LLMSummaryResult,
    UncitedSummaryError,
    assert_citations_are_grounded,
    call_llm_summarize,
)
from pipeline.rag_sanitize import PatientAttributeProfile, build_sanitized_context

# Gerçek PubMed'den gelmiş gibi davranan, AMA bu testte tamamen yerel olan
# sahte retrieved chunk'lar. (Ağ yok, DB yok.)
_RETRIEVED = (
    ("31562769", "Temozolomide in elderly glioblastoma", "MGMT methylation predicted benefit."),
    ("29260225", "MGMT promoter methylation review", "Methylated tumors showed longer survival."),
)
_ALLOWED = {"31562769", "29260225"}

# Bize HİÇ verilmemiş, modelin "uydurduğu" PMID.
_FABRICATED = "99999999"


def _context():
    attrs = PatientAttributeProfile(
        age_years=62,
        gender="Male",
        kps_score=80,
        mgmt_status_bucket="methylated",
        idh1_status_bucket="wildtype",
        risk_category="orta",
    )
    return build_sanitized_context(attrs, _RETRIEVED)


def _fake_result(cited, summary_text="Ozet metni.") -> LLMSummaryResult:
    return LLMSummaryResult(
        summary_text=summary_text,
        cited_pmids=tuple(cited),
        model_name="fake-model-for-tests",
    )


def _patch_provider(monkeypatch, result: LLMSummaryResult) -> None:
    """Sahte LLM yanıtı enjekte eder -- `_invoke_llm_provider` ağa ÇIKMAZ,
    zaten burada hiç çağrılmayan gerçek gövdesinin yerine geçer."""

    monkeypatch.setattr(rag_llm, "_invoke_llm_provider", lambda prompt: result)


# =====================================================================
# 0) Ön koşul: guard'sız dünya nasıl görünüyordu (ve fixture geçerliliği)
# =====================================================================


def test_fixture_precondition_fabricated_pmid_is_really_outside_retrieved_set() -> None:
    """ÖN KOŞUL: testin 'uydurma' dediği PMID GERÇEKTEN retrieved kümenin
    DIŞINDA mı? Bu doğrulanmazsa aşağıdaki KIRMIZI test, yanlış sebeple
    (örn. fixture'ı yanlış yazdığımız için) geçebilirdi."""

    allowed = rag_llm.allowed_pmids(_context())
    assert allowed == _ALLOWED, f"retrieved PMID kumesi beklenenden farkli: {sorted(allowed)}"
    assert _FABRICATED not in allowed


def test_provider_is_not_configured_so_no_network_call_can_happen() -> None:
    """Yapısal güvence: monkeypatch YOKKEN `call_llm_summarize` hâlâ
    `LLMNotConfiguredError` fırlatıyor -- yani bu test dosyası hiçbir
    koşulda gerçek bir sağlayıcıya gitmez."""

    with pytest.raises(LLMNotConfiguredError):
        call_llm_summarize(_context())


# =====================================================================
# 1) KIRMIZI -- uydurma PMID reddedilmeli
# =====================================================================


def test_call_llm_summarize_rejects_fabricated_pmid(monkeypatch) -> None:
    """KIRMIZI: sahte LLM yanıtı, retrieved kümede OLMAYAN bir PMID
    içeriyor -> `FabricatedPmidError`. Mesaj HEM uydurulan PMID'yi HEM
    beklenen kümeyi söylemeli (jüri sorusu: "bu kaynağı nereden
    buldunuz")."""

    _patch_provider(monkeypatch, _fake_result(["31562769", _FABRICATED]))

    with pytest.raises(FabricatedPmidError) as exc:
        call_llm_summarize(_context())

    message = str(exc.value)
    assert _FABRICATED in message, "hata mesaji UYDURULAN PMID'yi soylemiyor"
    for pmid in sorted(_ALLOWED):
        assert pmid in message, f"hata mesaji BEKLENEN kumeyi tam soylemiyor ({pmid} yok)"
    # Sessiz kirpma YASAK: guard bir "temizlenmis" sonuc DONDURMEZ.
    assert isinstance(exc.value, CitationGroundingError)


def test_fabricated_pmid_hidden_only_in_summary_text_is_also_rejected(monkeypatch) -> None:
    """Model, uydurduğu PMID'yi `cited_pmids` listesine KOYMAYIP yalnız
    özet METNİNE yazarsa da yakalanmalı -- `summary_text` de jüriye
    gösterilen bir yüzeydir."""

    _patch_provider(
        monkeypatch,
        _fake_result(["31562769"], summary_text=f"Benzer bulgular bildirildi (PMID: {_FABRICATED})."),
    )

    with pytest.raises(FabricatedPmidError) as exc:
        call_llm_summarize(_context())
    assert _FABRICATED in str(exc.value)


def test_malformed_citation_is_rejected_not_silently_cleaned(monkeypatch) -> None:
    """Rakama çözülemeyen bir alıntı ("PMID yok", "abc123") sessizce
    temizlenmez -- doğrulanamayan alıntı = temellendirilmemiş alıntı."""

    _patch_provider(monkeypatch, _fake_result(["31562769", "abc123"]))

    with pytest.raises(FabricatedPmidError) as exc:
        call_llm_summarize(_context())
    assert "abc123" in str(exc.value)


# =====================================================================
# 2) "Sıfır alıntı" kararı -- REDDEDİLİR (gerekçe guard dokstring'inde)
# =====================================================================


def test_zero_citations_is_rejected(monkeypatch) -> None:
    """Sözleşme "PMID-referanslı özet" olduğu için SIFIR alıntı KABUL
    EDİLMEZ. Bu, guard'ın engellemeye çalıştığı tehlikeli davranışın
    (parametrik hafızadan konuşma) tam olarak işaretidir."""

    _patch_provider(monkeypatch, _fake_result([], summary_text="Genel bir ozet, kaynak yok."))

    with pytest.raises(UncitedSummaryError) as exc:
        call_llm_summarize(_context())
    message = str(exc.value)
    assert "sifir alinti" in message.lower() or "HIC PMID" in message
    for pmid in sorted(_ALLOWED):
        assert pmid in message


def test_empty_retrieved_context_is_rejected_even_with_zero_citations(monkeypatch) -> None:
    """Bağlamda HİÇ literatür yoksa, "0 alıntı normaldir" diye sessizce
    geçilmez -- temellendirilebilir özet MANTIKEN imkânsızdır."""

    attrs = PatientAttributeProfile(
        age_years=62,
        gender="Female",
        kps_score=90,
        mgmt_status_bucket="unknown",
        idh1_status_bucket="unknown",
        risk_category=None,
    )
    empty_context = build_sanitized_context(attrs, ())
    _patch_provider(monkeypatch, _fake_result([]))

    with pytest.raises(UncitedSummaryError) as exc:
        call_llm_summarize(empty_context)
    assert "0" in str(exc.value)


# =====================================================================
# 3) YEŞİL -- temiz yanıt geçmeli
# =====================================================================


def test_call_llm_summarize_accepts_fully_grounded_summary(monkeypatch) -> None:
    """YEŞİL: tüm alıntılar retrieved kümenin İÇİNDE -> özet DÖNER,
    değiştirilmeden."""

    clean = _fake_result(
        ["31562769", "29260225"],
        summary_text="MGMT metilasyonu sagkalimla iliskili (PMID: 29260225).",
    )
    _patch_provider(monkeypatch, clean)

    result = call_llm_summarize(_context())
    assert result is clean
    assert result.cited_pmids == ("31562769", "29260225")
    assert result.disclaimer.startswith("Bu özet bir Klinik Karar Destek")


def test_pmid_prefix_and_whitespace_are_normalized_symmetrically(monkeypatch) -> None:
    """'PMID: 31562769' gibi biçimler HER İKİ tarafta AYNI fonksiyonla
    normalize edilir -> yanlış-pozitif ret ÜRETMEZ. (Asimetrik
    normalizasyon sahte 'eşleşmedi' üretirdi.)"""

    _patch_provider(monkeypatch, _fake_result(["PMID: 31562769", "  29260225 "]))
    result = call_llm_summarize(_context())
    assert result.cited_pmids == ("PMID: 31562769", "  29260225 ")


def test_legitimate_numbers_in_summary_text_are_not_treated_as_pmids(monkeypatch) -> None:
    """Yanlış pozitif kontrolü: özet metnindeki meşru sayılar (n=295,
    yıl, p-değeri) PMID SAYILMAZ -- guard fail-closed olduğu için bir
    yanlış pozitif = özetin tamamen engellenmesi demektir."""

    _patch_provider(
        monkeypatch,
        _fake_result(
            ["31562769"],
            summary_text="2019 yilinda n=295 hastada p<0.001 bulundu; 10-20 mg doz araligi.",
        ),
    )
    assert call_llm_summarize(_context()).cited_pmids == ("31562769",)


# =====================================================================
# 4) TAUTOLOJİ KONTROLÜ -- guard GERÇEKTEN bir şey yakalıyor mu?
# =====================================================================


def test_anti_tautology_a_guardless_reference_path_lets_the_fabrication_through(monkeypatch) -> None:
    """TAUTOLOJİ KONTROLÜ (1/2) -- "testin boşa geçmediği" kanıtı.

    Aynı sahte yanıt, guard'ı UYGULAMAYAN bir referans yoldan (2026-09-13
    öncesi davranışın birebir karşılığı: sağlayıcı sonucunu DOĞRUDAN
    döndür) geçirilir. O yol uydurma PMID'yi SORUNSUZ GEÇİRİR. Yani
    KIRMIZI testin başarısı, "her şey zaten hata veriyor"dan DEĞİL,
    guard'ın kendisinden geliyor.

    Bu, `tests/test_xgboost_model.py::test_old_misaligned_pipeline_...`
    deseninin birebir uygulamasıdır: yeni kapının çözdüğü SOMUT sorunu
    kanıtla."""

    context = _context()
    fake = _fake_result(["31562769", _FABRICATED])

    def _guardless_call(ctx):
        # 2026-09-13 ONCESI davranis: prompt kapisi VAR, PMID kapisi YOK.
        prompt = rag_llm.build_llm_prompt(ctx)
        rag_llm.assert_no_identifiers_leaked(prompt)
        return fake  # <-- guard YOK: uydurma PMID dogrudan kullaniciya giderdi

    leaked = _guardless_call(context)
    assert _FABRICATED in leaked.cited_pmids, (
        "Guard'siz referans yol bile uydurmayi gecirmedi -- bu durumda "
        "KIRMIZI testin neyi kanitladigi BELIRSIZ olurdu."
    )

    # AYNI yanit, GERCEK yoldan gecirilince REDDEDILIYOR.
    _patch_provider(monkeypatch, fake)
    with pytest.raises(FabricatedPmidError):
        call_llm_summarize(context)


def test_anti_tautology_b_guard_is_not_unconditionally_raising(monkeypatch) -> None:
    """TAUTOLOJİ KONTROLÜ (2/2): guard "her zaman patlayan" bir kontrol
    DEĞİL. Tek bir PMID'yi uydurmadan geçerliye çevirmek, AYNI çağrıyı
    KIRMIZI'dan YEŞİL'e döndürür -- yani ayrım gerçekten alıntının
    kendisinden geliyor, başka bir yan etkiden değil."""

    context = _context()

    _patch_provider(monkeypatch, _fake_result(["31562769", _FABRICATED]))
    with pytest.raises(FabricatedPmidError):
        call_llm_summarize(context)

    # TEK degisiklik: uydurma PMID -> retrieved kumedeki gercek PMID.
    _patch_provider(monkeypatch, _fake_result(["31562769", "29260225"]))
    assert call_llm_summarize(context).cited_pmids == ("31562769", "29260225")


def test_anti_tautology_c_guard_cannot_be_skipped_by_any_public_flag() -> None:
    """MİMARİ ŞART: guard'ı atlayan bir konfigürasyon bayrağı/parametre
    OLMAMALI. İki yapısal kanıt:
      (a) `call_llm_summarize` TEK parametre alır (context) -- "skip"
          benzeri bir anahtar yok.
      (b) Modülde guard'ı koşullu kılabilecek bir ENV/bayrak sabiti yok."""

    import ast
    import inspect
    import textwrap

    sig = inspect.signature(call_llm_summarize)
    assert list(sig.parameters) == ["context"], (
        f"call_llm_summarize imzasi degismis: {sig} -- guard'i kosullu kilabilecek "
        "bir parametre EKLENMIS olabilir."
    )

    # AST ile: guard cagrisi fonksiyon GOVDESININ EN UST seviyesinde, duz bir
    # ifade olarak bulunmali -- bir `if`/`try`/`for` blogunun ICINDE degil.
    # (Metin/girinti kontrolu dokstring'deki aciklamayla karisiyordu.)
    tree = ast.parse(textwrap.dedent(inspect.getsource(call_llm_summarize)))
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)

    def _is_guard_call(node) -> bool:
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "assert_citations_are_grounded"
        )

    top_level_guard = [st for st in func.body if _is_guard_call(st)]
    assert len(top_level_guard) == 1, (
        "PMID temellendirme kapisi `call_llm_summarize` govdesinin EN UST "
        "seviyesinde duz bir cagri olarak BULUNAMADI (kaldirilmis veya "
        "kosullu bir bloga alinmis olabilir)."
    )

    # Ayrica: guard `return`DEN ONCE gelmeli.
    guard_index = func.body.index(top_level_guard[0])
    return_indexes = [i for i, st in enumerate(func.body) if isinstance(st, ast.Return)]
    assert return_indexes and guard_index < min(return_indexes), (
        "PMID kapisi `return`den SONRA -- yani hic calismiyor olabilir."
    )

    forbidden = ("GBMAID_LLM_SKIP", "skip_citation", "verify_citations", "SKIP_PMID")
    module_source = inspect.getsource(rag_llm)
    for token in forbidden:
        assert token not in module_source, f"guard'i atlatabilecek bayrak bulundu: {token}"


# =====================================================================
# 5) Guard doğrudan çağrıldığında da aynı sözleşme (birim seviyesi)
# =====================================================================


def test_assert_citations_are_grounded_is_a_pure_gate_returning_none() -> None:
    """`assert_no_identifiers_leaked` deseniyle tutarlılık: temiz girdide
    `None` döner (bir "temizlenmiş" nesne DEĞİL) -- yani guard'ın sessiz
    bir düzeltme yapması YAPISAL OLARAK mümkün değil."""

    assert assert_citations_are_grounded(_fake_result(["31562769"]), _context()) is None


# =====================================================================
# 6) 2026-09-14 (rag-agent-P5) -- GERÇEK SAĞLAYICI ÇAĞRISI EKLENDİ
#
# 🔴 Bu bölümdeki HİÇBİR test ağa çıkmaz. Mühür iki katlı:
#   (1) `llm_provider._DISPATCH` sahte bir fonksiyonla değiştirilir --
#       SDK import'una BİLE sıra gelmez (paketler zaten kurulu değil);
#   (2) anahtar olarak açıkça SAHTE bir dize verilir.
# =====================================================================

from pipeline import llm_provider  # noqa: E402  (bolum-yerel, bilincli)
from pipeline.llm_provider import LLMProviderError, ProviderResponse  # noqa: E402
from pipeline.rag_llm import (  # noqa: E402
    LLMResponseFormatError,
    build_llm_system_prompt,
    parse_llm_json_payload,
)

_FAKE_KEY_FOR_TESTS = "sk-TEST-ONLY-NOT-A-REAL-KEY-0000"


def _enable_fake_provider(monkeypatch, responder) -> None:
    """LLM'i AÇIK duruma getirir ama sağlayıcıyı sahteyle değiştirir.

    `responder(config, api_key, system_prompt=..., user_prompt=...)` bir
    `ProviderResponse` döndürmeli (veya hata fırlatmalı)."""

    monkeypatch.setenv("GBMAID_LLM_ENABLED", "true")
    monkeypatch.setenv("GBMAID_LLM_PROVIDER", "openai")
    monkeypatch.setenv("GBMAID_LLM_MODEL", "fake-model-2026-01-01")
    monkeypatch.setenv("OPENAI_API_KEY", _FAKE_KEY_FOR_TESTS)
    monkeypatch.setitem(llm_provider._DISPATCH, "openai", responder)


def _json_response(summary: str, pmids) -> ProviderResponse:
    import json as _json

    return ProviderResponse(
        text=_json.dumps({"summary_tr": summary, "cited_pmids": list(pmids)}),
        model_name="fake-model-2026-01-01",
        stop_reason="stop",
        input_tokens=123,
        output_tokens=45,
    )


# --- 6a) JSON zarfi cozumleme (saf) ----------------------------------


def test_parse_llm_json_payload_happy_path() -> None:
    summary, pmids = parse_llm_json_payload(
        '{"summary_tr": "Ozet.", "cited_pmids": ["31562769"]}'
    )
    assert summary == "Ozet."
    assert pmids == ("31562769",)


def test_parse_llm_json_payload_strips_markdown_fence() -> None:
    """Çit soymak içeriği DEĞİŞTİRMEZ, yalnız zarfı açar -- bu yüzden
    "sessiz düzeltme" yasağını ihlal etmez."""

    text = '```json\n{"summary_tr": "Ozet.", "cited_pmids": ["31562769"]}\n```'
    assert parse_llm_json_payload(text)[0] == "Ozet."


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "duz metin, JSON degil",
        '["liste", "nesne degil"]',
        '{"cited_pmids": ["31562769"]}',
        '{"summary_tr": "   ", "cited_pmids": []}',
        '{"summary_tr": "x", "cited_pmids": "31562769"}',
        '{"summary_tr": "x", "cited_pmids": [{"pmid": 1}]}',
    ],
)
def test_parse_llm_json_payload_rejects_malformed_output(text: str) -> None:
    with pytest.raises(LLMResponseFormatError):
        parse_llm_json_payload(text)


def test_format_error_is_not_a_citation_grounding_error() -> None:
    """🔴 TİP AYRIMI KASITLIDIR: `rag_pipeline` biçim hatasını (teknik
    arıza -> literatür bloğu ayakta kalır) ile uydurma alıntıyı (doğruluk
    ihlali -> blok düşer) FARKLI ele alır. İkisi tek tip olsaydı bu ayrım
    sessizce kaybolurdu."""

    assert not issubclass(LLMResponseFormatError, CitationGroundingError)
    assert not issubclass(CitationGroundingError, LLMResponseFormatError)


def test_empty_cited_pmids_is_left_to_the_grounding_gate_not_the_parser() -> None:
    """Boş alıntı listesi PARSER'da reddedilmez -- `UncitedSummaryError`
    kapısının işidir. İki yerde reddetmek, hangi kapının reddettiğini
    belirsizleştirirdi."""

    assert parse_llm_json_payload('{"summary_tr": "x", "cited_pmids": []}') == ("x", ())


# --- 6b) Sistem prompt'u ---------------------------------------------


def test_system_prompt_demands_turkish_json_and_no_fabricated_sources() -> None:
    system = build_llm_system_prompt()
    lowered = system.lower()
    assert "turkce" in lowered or "türkçe" in lowered
    assert "summary_tr" in system and "cited_pmids" in system
    assert "json" in lowered
    assert "uydurma" in lowered


def test_system_prompt_carries_no_patient_data() -> None:
    """Sistem prompt'u SABİTTİR ve hasta verisi taşımaz -- sanitizasyon
    kapısının konusu veri taşıyan KULLANICI prompt'udur."""

    system = build_llm_system_prompt()
    rag_llm.assert_no_identifiers_leaked(system)
    assert build_llm_system_prompt() == system  # deterministik


# --- 6c) UCTAN UCA, SAHTE SAGLAYICI ILE ------------------------------


def test_end_to_end_with_fake_provider_returns_grounded_turkish_summary(monkeypatch) -> None:
    """YEŞİL: sahte sağlayıcı yalnız retrieved kümedeki PMID'leri
    alıntılıyor -> özet DÖNER, token/süre ölçümleri taşınır."""

    captured: dict[str, str] = {}

    def _responder(config, api_key, *, system_prompt, user_prompt):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        captured["model"] = config.model
        return _json_response("Turkce ozet metni.", ["31562769", "29260225"])

    _enable_fake_provider(monkeypatch, _responder)
    result = call_llm_summarize(_context())

    assert result.summary_text == "Turkce ozet metni."
    assert result.cited_pmids == ("31562769", "29260225")
    assert result.model_name == "fake-model-2026-01-01"
    assert (result.input_tokens, result.output_tokens) == (123, 45)
    assert result.latency_seconds is not None and result.latency_seconds >= 0
    assert "summary_tr" in captured["system"]
    assert "[PMID:31562769]" in captured["user"]


def test_end_to_end_fake_provider_hallucination_is_rejected(monkeypatch) -> None:
    """KIRMIZI: sahte sağlayıcı, kendisine VERİLMEYEN bir PMID
    alıntılıyor -> `FabricatedPmidError`. Guard, sağlayıcı eklendikten
    SONRA da KOŞULSUZ çalışıyor."""

    _enable_fake_provider(
        monkeypatch,
        lambda c, k, *, system_prompt, user_prompt: _json_response(
            "Ozet.", ["31562769", _FABRICATED]
        ),
    )
    with pytest.raises(FabricatedPmidError) as exc:
        call_llm_summarize(_context())
    assert _FABRICATED in str(exc.value)


def test_end_to_end_fake_provider_zero_citations_is_rejected(monkeypatch) -> None:
    """KIRMIZI: sıfır alıntı -> `UncitedSummaryError` (sözleşme
    "PMID-referanslı özet")."""

    _enable_fake_provider(
        monkeypatch,
        lambda c, k, *, system_prompt, user_prompt: _json_response("Kaynaksiz ozet.", []),
    )
    with pytest.raises(UncitedSummaryError):
        call_llm_summarize(_context())


def test_end_to_end_fake_provider_malformed_json_is_rejected(monkeypatch) -> None:
    _enable_fake_provider(
        monkeypatch,
        lambda c, k, *, system_prompt, user_prompt: ProviderResponse(
            text="Bu bir JSON degil.", model_name="fake-model-2026-01-01"
        ),
    )
    with pytest.raises(LLMResponseFormatError):
        call_llm_summarize(_context())


def test_end_to_end_provider_technical_failure_surfaces_as_provider_error(monkeypatch) -> None:
    def _boom(config, api_key, *, system_prompt, user_prompt):
        raise LLMProviderError("kota asildi (sahte)")

    _enable_fake_provider(monkeypatch, _boom)
    with pytest.raises(LLMProviderError):
        call_llm_summarize(_context())


# --- 6d) VARSAYILAN KAPALI -- bugunku davranis DEGISMEDI -------------


def test_disabled_flag_reproduces_the_pre_2026_09_14_behaviour(monkeypatch) -> None:
    """🔴 EN ÖNEMLİ REGRESYON TESTİ: `GBMAID_LLM_ENABLED` kapalıyken
    `call_llm_summarize` -- sağlayıcı kodu YAZILMADAN ÖNCEKİ hâliyle
    birebir aynı şekilde -- `LLMNotConfiguredError` fırlatır."""

    monkeypatch.setenv("GBMAID_LLM_ENABLED", "false")
    monkeypatch.setenv("GBMAID_LLM_PROVIDER", "openai")
    monkeypatch.setenv("GBMAID_LLM_MODEL", "fake-model-2026-01-01")
    monkeypatch.setenv("OPENAI_API_KEY", _FAKE_KEY_FOR_TESTS)

    with pytest.raises(LLMNotConfiguredError):
        call_llm_summarize(_context())


def test_enabled_but_misconfigured_still_degrades_as_not_configured(monkeypatch) -> None:
    """Açık ama model adı hareketli alias -> yine `LLMNotConfiguredError`
    (pipeline'ın mevcut graceful-degradation dalı korunur), ama mesaj
    NEDENİ söyler. Sessiz bir "varsayılan modele düş" YOK."""

    monkeypatch.setenv("GBMAID_LLM_ENABLED", "true")
    monkeypatch.setenv("GBMAID_LLM_PROVIDER", "openai")
    monkeypatch.setenv("GBMAID_LLM_MODEL", "gpt-4o-latest")
    monkeypatch.setenv("OPENAI_API_KEY", _FAKE_KEY_FOR_TESTS)

    with pytest.raises(LLMNotConfiguredError) as exc:
        call_llm_summarize(_context())
    assert "ALIAS" in str(exc.value).upper()


def test_no_fake_api_key_leaks_into_any_error_message(monkeypatch) -> None:
    """Anahtar hiçbir hata mesajına girmez -- sahte anahtarla ölçülür."""

    def _boom(config, api_key, *, system_prompt, user_prompt):
        assert api_key == _FAKE_KEY_FOR_TESTS  # saglayiciya ULASIYOR
        raise LLMProviderError("HTTP 401 (sahte)")

    _enable_fake_provider(monkeypatch, _boom)
    with pytest.raises(LLMProviderError) as exc:
        call_llm_summarize(_context())
    assert _FAKE_KEY_FOR_TESTS not in str(exc.value)
