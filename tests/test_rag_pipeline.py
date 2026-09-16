"""`pipeline/rag_pipeline.py` testleri -- graceful degradation sözleşmesi
(plan.txt satır 550-551: "RAG/LLM hata verirse pipeline durmaz").

TÜM testler DB/PubMed/Upstash'i MOCK'lar -- gerçek ağ isteği YOK. Amaç:
`generate_literature_summary()` HİÇBİR aşamada exception fırlatmadığını
kanıtlamak."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pipeline import rag_pipeline
from pipeline.rag_embed_index import IndexedChunk, TempRagIndexError
from pipeline.rag_pubmed import PubMedArticle, PubMedFetchError
from pipeline.rag_query import PatientNotFoundError, RawPatientProfile


def _profile(**overrides) -> RawPatientProfile:
    base = dict(
        patient_id="TCGA-00-0000",
        age=60,
        gender="Male",
        kps_score=80,
        mgmt_status_raw="Methylated",
        idh1_status_raw="Wildtype",
        has_omics=False,
        egfr_cnv=None,
        egfr_expression=None,
        pten_cnv=None,
        pten_methylation=None,
        mgmt_methylation=None,
        mgmt_expression=None,
        molecular_subtype=None,
    )
    base.update(overrides)
    return RawPatientProfile(**base)


def test_patient_not_found_degrades_gracefully() -> None:
    with patch(
        "pipeline.rag_pipeline.fetch_raw_patient_profile",
        side_effect=PatientNotFoundError("nope"),
    ):
        result = rag_pipeline.generate_literature_summary("DOES-NOT-EXIST")

    assert result.available is False
    assert "nope" in result.reason


def test_db_connection_error_degrades_gracefully_not_raises() -> None:
    with patch(
        "pipeline.rag_pipeline.fetch_raw_patient_profile",
        side_effect=RuntimeError("connection refused"),
    ):
        result = rag_pipeline.generate_literature_summary("TCGA-00-0000")

    assert result.available is False
    assert "DB hatasi" in result.reason


def test_pubmed_empty_result_degrades_gracefully() -> None:
    with patch(
        "pipeline.rag_pipeline.fetch_raw_patient_profile", return_value=_profile()
    ), patch("pipeline.rag_pipeline.get_cached_summary", return_value=None), patch(
        "pipeline.rag_pipeline.set_cached_summary", return_value=True
    ), patch(
        "pipeline.rag_pipeline.fetch_pubmed_articles",
        side_effect=PubMedFetchError("sonuc bulunamadi"),
    ):
        result = rag_pipeline.generate_literature_summary("TCGA-00-0000")

    assert result.available is False
    assert "PubMed" in result.reason
    assert result.query is not None  # sorgu yine de uretilmis olmali


def test_pubmed_unexpected_exception_does_not_propagate() -> None:
    with patch(
        "pipeline.rag_pipeline.fetch_raw_patient_profile", return_value=_profile()
    ), patch("pipeline.rag_pipeline.get_cached_summary", return_value=None), patch(
        "pipeline.rag_pipeline.fetch_pubmed_articles", side_effect=ValueError("boom")
    ):
        result = rag_pipeline.generate_literature_summary("TCGA-00-0000")

    assert result.available is False
    assert "beklenmeyen hata" in result.reason


def test_articles_with_no_abstracts_degrade_gracefully() -> None:
    articles = [PubMedArticle(pmid="1", title="No abstract here", abstract="")]
    with patch(
        "pipeline.rag_pipeline.fetch_raw_patient_profile", return_value=_profile()
    ), patch("pipeline.rag_pipeline.get_cached_summary", return_value=None), patch(
        "pipeline.rag_pipeline.set_cached_summary", return_value=True
    ), patch("pipeline.rag_pipeline.fetch_pubmed_articles", return_value=articles):
        result = rag_pipeline.generate_literature_summary("TCGA-00-0000")

    assert result.available is False
    assert "abstract" in result.reason


def test_embedding_index_failure_degrades_gracefully() -> None:
    articles = [PubMedArticle(pmid="1", title="T", abstract="Glioblastoma survival study text.")]
    with patch(
        "pipeline.rag_pipeline.fetch_raw_patient_profile", return_value=_profile()
    ), patch("pipeline.rag_pipeline.get_cached_summary", return_value=None), patch(
        "pipeline.rag_pipeline.set_cached_summary", return_value=True
    ), patch("pipeline.rag_pipeline.fetch_pubmed_articles", return_value=articles), patch(
        "pipeline.rag_pipeline.build_temp_faiss_index",
        side_effect=TempRagIndexError("bos"),
    ):
        result = rag_pipeline.generate_literature_summary("TCGA-00-0000")

    assert result.available is False
    assert "indeksi" in result.reason


def test_successful_pipeline_returns_available_with_no_llm_summary() -> None:
    articles = [
        PubMedArticle(pmid="1", title="MGMT methylation study", abstract="Glioblastoma MGMT methylation predicts survival.")
    ]
    with patch(
        "pipeline.rag_pipeline.fetch_raw_patient_profile", return_value=_profile()
    ), patch("pipeline.rag_pipeline.get_cached_summary", return_value=None), patch(
        "pipeline.rag_pipeline.set_cached_summary", return_value=True
    ), patch("pipeline.rag_pipeline.fetch_pubmed_articles", return_value=articles):
        result = rag_pipeline.generate_literature_summary("TCGA-00-0000", top_k=3)

    assert result.available is True
    assert result.reason is None
    assert len(result.retrieved_chunks) >= 1
    assert result.sanitized_context is not None
    # LLM gercekten CAGRILMADI (gorev kapsami) -- ozet HER ZAMAN yok.
    assert result.llm_summary_available is False
    assert result.llm_reason is not None
    assert "TCGA-00-0000" not in str(result.sanitized_context.patient_context_text)


def test_uses_cache_when_available_skips_pubmed_call() -> None:
    cached_payload = {
        "articles": [
            {"pmid": "1", "title": "Cached study", "abstract": "Glioblastoma survival radiomics cached abstract."}
        ]
    }
    with patch(
        "pipeline.rag_pipeline.fetch_raw_patient_profile", return_value=_profile()
    ), patch("pipeline.rag_pipeline.get_cached_summary", return_value=cached_payload), patch(
        "pipeline.rag_pipeline.fetch_pubmed_articles"
    ) as mock_fetch:
        result = rag_pipeline.generate_literature_summary("TCGA-00-0000", top_k=3)

    mock_fetch.assert_not_called()
    assert result.available is True
    assert result.used_cache is True


def test_sanitization_violation_degrades_gracefully_instead_of_raising() -> None:
    """Kasıtlı olarak sanitizasyon kapısını ATLATMAYA çalışan bozuk bir
    girdi simüle edilir (retrieved chunk metnine bir hasta ID'si
    sızdırılırsa dahi) -- pipeline yine de exception FIRLATMAZ."""

    articles = [
        PubMedArticle(
            pmid="1",
            title="Study",
            abstract="Glioblastoma survival radiomics.",
        )
    ]
    with patch(
        "pipeline.rag_pipeline.fetch_raw_patient_profile", return_value=_profile()
    ), patch("pipeline.rag_pipeline.get_cached_summary", return_value=None), patch(
        "pipeline.rag_pipeline.set_cached_summary", return_value=True
    ), patch("pipeline.rag_pipeline.fetch_pubmed_articles", return_value=articles), patch(
        "pipeline.rag_pipeline.build_sanitized_context",
        side_effect=__import__(
            "pipeline.rag_sanitize", fromlist=["SanitizationViolationError"]
        ).SanitizationViolationError("simulated leak"),
    ):
        result = rag_pipeline.generate_literature_summary("TCGA-00-0000")

    assert result.available is False
    assert "KRITIK" in result.reason


# ---------------------------------------------------------------------------
# REGRESYON — 2026-08-18 çapraz doğrulama (reviewer) CRITICAL 2: sızıntı
# GİRİŞİMİ ile sıradan "veri yok" durumu artık AYRI/programatik olarak
# ayırt edilebiliyor ve ayrı loglanıyor.
# ---------------------------------------------------------------------------


def test_sanitization_violation_sets_dedicated_flag_not_just_reason_string() -> None:
    """`sanitization_violation=True` -- çağıran taraf `reason` metnini
    STRING-EŞLEŞTİRMEDEN (örn. 'KRITIK' öneki) bu durumu programatik
    olarak ayırt edebilmeli (reviewer CRITICAL 2)."""

    articles = [PubMedArticle(pmid="1", title="Study", abstract="Glioblastoma survival radiomics.")]
    with patch(
        "pipeline.rag_pipeline.fetch_raw_patient_profile", return_value=_profile()
    ), patch("pipeline.rag_pipeline.get_cached_summary", return_value=None), patch(
        "pipeline.rag_pipeline.set_cached_summary", return_value=True
    ), patch("pipeline.rag_pipeline.fetch_pubmed_articles", return_value=articles), patch(
        "pipeline.rag_pipeline.build_sanitized_context",
        side_effect=__import__(
            "pipeline.rag_sanitize", fromlist=["SanitizationViolationError"]
        ).SanitizationViolationError("simulated leak"),
    ):
        result = rag_pipeline.generate_literature_summary("TCGA-00-0000")

    assert result.sanitization_violation is True


def test_ordinary_pubmed_failure_does_not_set_sanitization_violation_flag() -> None:
    """Sıradan bir 'PubMed'de sonuç yok' durumu YANLIŞLIKLA sızıntı
    girişimi olarak işaretlenmemeli (yanlış-pozitif regresyonu)."""

    with patch(
        "pipeline.rag_pipeline.fetch_raw_patient_profile", return_value=_profile()
    ), patch("pipeline.rag_pipeline.get_cached_summary", return_value=None), patch(
        "pipeline.rag_pipeline.set_cached_summary", return_value=True
    ), patch(
        "pipeline.rag_pipeline.fetch_pubmed_articles",
        side_effect=PubMedFetchError("sonuc bulunamadi"),
    ):
        result = rag_pipeline.generate_literature_summary("TCGA-00-0000")

    assert result.sanitization_violation is False


def test_successful_pipeline_run_does_not_set_sanitization_violation_flag() -> None:
    articles = [
        PubMedArticle(pmid="1", title="MGMT methylation study", abstract="Glioblastoma MGMT methylation predicts survival.")
    ]
    with patch(
        "pipeline.rag_pipeline.fetch_raw_patient_profile", return_value=_profile()
    ), patch("pipeline.rag_pipeline.get_cached_summary", return_value=None), patch(
        "pipeline.rag_pipeline.set_cached_summary", return_value=True
    ), patch("pipeline.rag_pipeline.fetch_pubmed_articles", return_value=articles):
        result = rag_pipeline.generate_literature_summary("TCGA-00-0000", top_k=3)

    assert result.available is True
    assert result.sanitization_violation is False


def test_sanitization_violation_is_logged_separately_via_stdlib_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sızıntı girişimi AYRI/filtrelenebilir bir kayıt olarak `logging` ile
    düşülmeli (reviewer CRITICAL 2: `pipeline/` dizininde hiç `logging`
    çağrısı olmaması bir bulgu olarak işaretlenmişti)."""

    articles = [PubMedArticle(pmid="1", title="Study", abstract="Glioblastoma survival radiomics.")]
    with patch(
        "pipeline.rag_pipeline.fetch_raw_patient_profile", return_value=_profile()
    ), patch("pipeline.rag_pipeline.get_cached_summary", return_value=None), patch(
        "pipeline.rag_pipeline.set_cached_summary", return_value=True
    ), patch("pipeline.rag_pipeline.fetch_pubmed_articles", return_value=articles), patch(
        "pipeline.rag_pipeline.build_sanitized_context",
        side_effect=__import__(
            "pipeline.rag_sanitize", fromlist=["SanitizationViolationError"]
        ).SanitizationViolationError("simulated leak"),
    ):
        with caplog.at_level("ERROR", logger="pipeline.rag.security"):
            rag_pipeline.generate_literature_summary("TCGA-00-0000")

    assert len(caplog.records) == 1
    assert caplog.records[0].name == "pipeline.rag.security"
    assert caplog.records[0].levelname == "ERROR"
    # Ham sizdirilan deger loglanmamali -- yalniz olayin GERCEKLESTIGI.
    assert "simulated leak" in caplog.text  # exc mesaji (zaten ham deger icermiyor) gorunur olmali


def test_ordinary_failures_do_not_emit_security_log_records(caplog: pytest.LogCaptureFixture) -> None:
    """Yanlış-pozitif regresyonu: sıradan hatalar güvenlik logger'ını
    TETİKLEMEMELİ (yalnız gerçek sanitizasyon ihlalleri tetiklemeli)."""

    with patch(
        "pipeline.rag_pipeline.fetch_raw_patient_profile", return_value=_profile()
    ), patch("pipeline.rag_pipeline.get_cached_summary", return_value=None), patch(
        "pipeline.rag_pipeline.set_cached_summary", return_value=True
    ), patch(
        "pipeline.rag_pipeline.fetch_pubmed_articles",
        side_effect=PubMedFetchError("sonuc bulunamadi"),
    ):
        with caplog.at_level("ERROR", logger="pipeline.rag.security"):
            rag_pipeline.generate_literature_summary("TCGA-00-0000")

    assert len(caplog.records) == 0


# =====================================================================
# 2026-09-14 (rag-agent-P5) -- İŞ 2 (CitationGroundingError yakalaması)
#                              + İŞ 3 (Türkçe özet + ORİJİNAL abstract)
#
# 🔴 AĞ YOK: sağlayıcı `llm_provider._DISPATCH` üzerinden sahteyle
# değiştirilir, anahtar açıkça SAHTE bir dizedir.
# =====================================================================

import json as _json  # noqa: E402

from pipeline import llm_provider, rag_llm  # noqa: E402
from pipeline.llm_provider import LLMProviderError, ProviderResponse  # noqa: E402
from pipeline.rag_llm import (  # noqa: E402
    CitationGroundingError,
    LLMNotConfiguredError,
)

_PMID_A = "31562769"
_PMID_B = "29260225"
_FABRICATED_PMID = "99999999"
_FAKE_KEY_FOR_TESTS = "sk-TEST-ONLY-NOT-A-REAL-KEY-0000"

# Kasitli olarak REDAKTE EDILECEK bir kurum adi iceriyor ("The Cancer
# Genome Atlas") -- boylece "orijinal abstract redakte EDILMEDEN
# tasiniyor mu" sorusu OLCULEBILIR hale geliyor.
_ARTICLES = [
    PubMedArticle(
        pmid=_PMID_A,
        title="MGMT methylation and survival",
        abstract=(
            "Glioblastoma patients with MGMT promoter methylation showed longer "
            "overall survival. Data were drawn from The Cancer Genome Atlas "
            "resource and validated in an independent cohort."
        ),
    ),
    PubMedArticle(
        pmid=_PMID_B,
        title="Radiomics in glioblastoma",
        abstract=(
            "Radiomic shape and texture features predicted overall survival in "
            "glioblastoma. Elastic net selection was applied with nested "
            "cross-validation."
        ),
    ),
]


def _enable_fake_provider(monkeypatch, responder) -> None:
    monkeypatch.setenv("GBMAID_LLM_ENABLED", "true")
    monkeypatch.setenv("GBMAID_LLM_PROVIDER", "openai")
    monkeypatch.setenv("GBMAID_LLM_MODEL", "fake-model-2026-01-01")
    monkeypatch.setenv("OPENAI_API_KEY", _FAKE_KEY_FOR_TESTS)
    monkeypatch.setitem(llm_provider._DISPATCH, "openai", responder)


def _responder_citing(pmids, summary="Turkce ozet metni."):
    def _fn(config, api_key, *, system_prompt, user_prompt):
        return ProviderResponse(
            text=_json.dumps({"summary_tr": summary, "cited_pmids": list(pmids)}),
            model_name="fake-model-2026-01-01",
            stop_reason="stop",
            input_tokens=150,
            output_tokens=60,
        )

    return _fn


def _run(patient_id: str = "TCGA-00-0000"):
    with patch(
        "pipeline.rag_pipeline.fetch_raw_patient_profile", return_value=_profile()
    ), patch("pipeline.rag_pipeline.get_cached_summary", return_value=None), patch(
        "pipeline.rag_pipeline.set_cached_summary", return_value=True
    ), patch("pipeline.rag_pipeline.fetch_pubmed_articles", return_value=_ARTICLES):
        return rag_pipeline.generate_literature_summary(patient_id, top_k=4)


# ---------------------------------------------------------------------
# A) YAPISAL KANIT -- eski `except` bu hatayi YAKALAYAMAZDI
# ---------------------------------------------------------------------


def test_citation_grounding_error_is_not_caught_by_the_old_except_clause() -> None:
    """🔴 KIRMIZI/YEŞİL kanıtının yapısal hâli: 2026-09-13'e kadar bu
    modül YALNIZ `LLMNotConfiguredError` yakalıyordu. `CitationGrounding
    Error` onun alt tipi OLMADIĞI için, sağlayıcı açıldığı gün bir
    halüsinasyon `generate_literature_summary`'den DIŞARI sızar ve
    "HİÇBİR ZAMAN exception fırlatmaz" sözleşmesini kırardı.

    Bu test, biri yeni `except` dalını silerse onu tek başına
    yakalayamaz (silme davranış testleriyle yakalanır) -- ama eklemenin
    NEDEN gerekli olduğunu kalıcı olarak belgeler."""

    assert not issubclass(CitationGroundingError, LLMNotConfiguredError)
    assert not issubclass(LLMNotConfiguredError, CitationGroundingError)


# ---------------------------------------------------------------------
# B) YESIL -- basarili ozet
# ---------------------------------------------------------------------


def test_successful_llm_summary_returns_turkish_summary_and_sources(monkeypatch) -> None:
    _enable_fake_provider(monkeypatch, _responder_citing([_PMID_A]))
    result = _run()

    assert result.available is True
    assert result.llm_summary_available is True
    assert result.llm_reason is None
    assert result.summary_tr == "Turkce ozet metni."
    assert result.llm_cited_pmids == (_PMID_A,)
    assert result.llm_model_name == "fake-model-2026-01-01"
    assert result.llm_disclaimer and "Klinik Karar Destek" in result.llm_disclaimer
    assert (result.llm_input_tokens, result.llm_output_tokens) == (150, 60)
    assert result.llm_latency_seconds is not None
    assert result.citation_grounding_violation is False


def test_sources_carry_the_original_unredacted_abstract_but_the_prompt_does_not(
    monkeypatch,
) -> None:
    """🔴 G3'ün ÖZÜ ve aynı zamanda güvenlik ayrımının ölçümü:
    `sources[].abstract_original` PubMed'den geldiği gibi (redakte
    EDİLMEMİŞ) durur -- jüri Türkçe özetle karşılaştırabilsin; ama LLM'e
    giden `sanitized_context` içinde AYNI kurum adı REDAKTE edilmiştir."""

    _enable_fake_provider(monkeypatch, _responder_citing([_PMID_A]))
    result = _run()

    source_a = {s.pmid: s for s in result.sources}[_PMID_A]
    assert "The Cancer Genome Atlas" in source_a.abstract_original
    assert source_a.title_original == "MGMT methylation and survival"
    assert source_a.retrieved_chunk_count >= 1

    sanitized_text = " ".join(
        s.text for s in result.sanitized_context.literature_snippets
    )
    assert "The Cancer Genome Atlas" not in sanitized_text, (
        "Sanitizasyon kurum adini redakte etmeliydi -- bu test, orijinal "
        "abstract'in redakte EDILMEMIS olmasinin bir sanitizasyon kacagi "
        "DEGIL, bilincli bir yanit alani oldugunu kanitlar."
    )


def test_sources_are_populated_even_when_the_llm_is_disabled() -> None:
    """G3 LLM'den BAĞIMSIZ çalışır: özet üretilmese bile kaynakların
    orijinal abstract'ları yanıtta durur."""

    result = _run()
    assert result.llm_summary_available is False
    assert result.summary_tr is None
    assert len(result.sources) >= 1
    assert all(s.abstract_original for s in result.sources)


def test_source_order_follows_top_k_similarity_not_alphabetical() -> None:
    result = _run()
    first_retrieved_pmid = result.retrieved_chunks[0].pmid
    assert result.sources[0].pmid == first_retrieved_pmid


# ---------------------------------------------------------------------
# C) KIRMIZI -- halusinasyon: blok duser, ISTEK COKMEZ
# ---------------------------------------------------------------------


def test_fabricated_pmid_makes_the_block_unavailable_without_raising(monkeypatch) -> None:
    """🔴 İŞ 2'nin ana testi: uydurma PMID -> blok `available:false`,
    `generate_literature_summary` HİÇBİR exception FIRLATMAZ (istek 200
    devam eder), uydurma özet SERVİS EDİLMEZ."""

    _enable_fake_provider(monkeypatch, _responder_citing([_PMID_A, _FABRICATED_PMID]))
    result = _run()  # firlatirsa test zaten hata verir

    assert result.available is False
    assert result.citation_grounding_violation is True
    assert result.llm_summary_available is False
    assert result.summary_tr is None, "UYDURMA OZET SERVIS EDILEMEZ"
    assert result.llm_cited_pmids == ()
    assert _FABRICATED_PMID in result.reason
    assert "KRITIK" in result.reason


def test_fabricated_pmid_block_still_carries_retrieval_evidence(monkeypatch) -> None:
    """Blok düşse de BİLGİ KAYBOLMAZ: ne getirildiği (`sources`,
    `retrieved_chunks`, `query`) görünür kalır -- jüri "ne getirdiniz,
    neyi reddettiniz" sorusunun ikisini de sorabilir."""

    _enable_fake_provider(monkeypatch, _responder_citing([_FABRICATED_PMID]))
    result = _run()

    assert result.available is False
    assert result.query is not None
    assert len(result.retrieved_chunks) >= 1
    assert len(result.sources) >= 1
    assert result.sanitized_context is not None


def test_zero_citations_also_rejects_the_summary(monkeypatch) -> None:
    """`UncitedSummaryError` de bir `CitationGroundingError`'dır -> aynı
    blok-fatal dal."""

    _enable_fake_provider(monkeypatch, _responder_citing([]))
    result = _run()

    assert result.available is False
    assert result.citation_grounding_violation is True
    assert result.summary_tr is None


def test_grounding_violation_writes_a_filterable_security_log(monkeypatch, caplog) -> None:
    """`sanitization_violation` ile AYNI desen: ihlal ayrı/filtrelenebilir
    bir `pipeline.rag.security` kaydına düşer."""

    _enable_fake_provider(monkeypatch, _responder_citing([_FABRICATED_PMID]))
    with caplog.at_level("ERROR", logger="pipeline.rag.security"):
        _run()

    security_records = [
        r for r in caplog.records if r.name == "pipeline.rag.security"
    ]
    assert security_records, "guvenlik logu YAZILMADI"
    assert any("temellendirme" in r.getMessage().lower() for r in security_records)


def test_grounding_violation_is_distinguishable_from_sanitization_violation(
    monkeypatch,
) -> None:
    """İki ihlal AYRI bayraklarla taşınır -- çağıran taraf `reason`
    metnini STRING-EŞLEŞTİRMEDEN hangisi olduğunu bilebilir."""

    _enable_fake_provider(monkeypatch, _responder_citing([_FABRICATED_PMID]))
    result = _run()

    assert result.citation_grounding_violation is True
    assert result.sanitization_violation is False


# ---------------------------------------------------------------------
# D) TEKNIK ARIZA -- literaturu GECERSIZ KILMAZ
# ---------------------------------------------------------------------


def test_provider_network_failure_keeps_the_block_available(monkeypatch) -> None:
    """(c) dalı: ağ/kota hatası bir DOĞRULUK ihlali değildir -- getirilen
    abstract'lar hâlâ gösterilebilir, blok ayakta kalır."""

    def _boom(config, api_key, *, system_prompt, user_prompt):
        raise LLMProviderError("kota asildi (sahte)")

    _enable_fake_provider(monkeypatch, _boom)
    result = _run()

    assert result.available is True
    assert result.citation_grounding_violation is False
    assert result.llm_summary_available is False
    assert result.summary_tr is None
    assert "LLM cagrisi basarisiz" in result.llm_reason
    assert len(result.sources) >= 1


def test_malformed_provider_json_keeps_the_block_available(monkeypatch) -> None:
    _enable_fake_provider(
        monkeypatch,
        lambda c, k, *, system_prompt, user_prompt: ProviderResponse(
            text="Bu JSON degil.", model_name="fake-model-2026-01-01"
        ),
    )
    result = _run()

    assert result.available is True
    assert result.llm_summary_available is False
    assert "LLMResponseFormatError" in result.llm_reason


def test_unexpected_exception_from_llm_layer_does_not_propagate(monkeypatch) -> None:
    """Modülün "asla fırlatmaz" sözleşmesindeki SESSİZ AÇIK: 2026-09-13'te
    LLM adımının etrafında geniş bir `except` YOKTU."""

    monkeypatch.setattr(
        rag_pipeline,
        "call_llm_summarize",
        lambda ctx: (_ for _ in ()).throw(TimeoutError("soket zaman asimi")),
    )
    result = _run()

    assert result.available is True
    assert result.llm_summary_available is False
    assert "TimeoutError" in result.llm_reason


# ---------------------------------------------------------------------
# E) VARSAYILAN KAPALI -- bugunku canli davranis DEGISMEDI
# ---------------------------------------------------------------------


def test_with_llm_disabled_the_pipeline_behaves_exactly_as_before(monkeypatch) -> None:
    """🔴 "Bugün canlı sistemde hiçbir şey bozulmamalı" şartının pipeline
    seviyesindeki karşılığı."""

    monkeypatch.setenv("GBMAID_LLM_ENABLED", "false")
    result = _run()

    assert result.available is True
    assert result.reason is None
    assert result.llm_summary_available is False
    assert result.llm_reason is not None and "KAPALI" in result.llm_reason
    assert result.summary_tr is None
    assert result.citation_grounding_violation is False


def test_no_fake_key_appears_anywhere_in_the_serialized_result(monkeypatch) -> None:
    """`api/analyze_patient.py` sonucu `dataclasses.asdict()` ile
    serileştirip HTTP yanıtına koyuyor -- anahtarın o ağaçta OLMADIĞI
    burada ölçülür."""

    import dataclasses

    _enable_fake_provider(monkeypatch, _responder_citing([_PMID_A]))
    payload = dataclasses.asdict(_run())
    assert _FAKE_KEY_FOR_TESTS not in str(payload)
    assert "api_key" not in str(payload)
