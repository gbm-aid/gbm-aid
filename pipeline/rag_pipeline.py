"""Hafta 5 — RAG orkestrasyonu: hasta profili -> PubMed sorgusu ->
makale çekme -> chunk/embed/geçici FAISS -> sanitizasyon -> (LLM iskeleti,
gerçek çağrı YOK).

Graceful degradation sözleşmesi (plan.txt satır 550-551, Sultan'ın
Hafta 5 görevi -- bu modül o sözleşmeye UYAR): `generate_literature_
summary()` HİÇBİR ZAMAN exception fırlatmaz. Her aşamadaki hata
yakalanır, `RagPipelineResult(available=False, reason=...)` ile
döner -- çağıran orkestrasyon (`POST /analyze_patient`) bunu "literatür
özeti üretilemedi" notuyla devam ettirebilir, pipeline ÇÖKMEZ.

`RagPipelineResult` BİLİNÇLİ OLARAK `patient_id` alanı TAŞIMAZ -- çağıran
taraf zaten `patient_id`'yi biliyor (fonksiyonu onunla çağırdı); bu
sonuç nesnesi yalnızca RAG'ın kendi çıktısını taşır, ileride
loglanırsa/serileştirilirse kimlikle birlikte saklanma riski BURADA
azaltılmış olur (nihai loglama kararı çağıran tarafa aittir).

⚠️ 2026-08-18 ÇAPRAZ DOĞRULAMA DÜZELTMESİ (reviewer CRITICAL 2): bir
`SanitizationViolationError` (gerçek bir kimlik-sızıntı GİRİŞİMİ) ile
sıradan bir "PubMed'de sonuç yok" durumu ÖNCEDEN aynı kanaldan
(`available=False, reason=...`), aynı önem seviyesinde dönüyordu --
`reason` string'indeki "KRITIK" öneki dışında hiçbir ayırt edici sinyal
yoktu. `RagPipelineResult` artık ayrı bir `sanitization_violation: bool`
alanı taşır VE ihlal yakalandığında `logging` ile (stdlib, `pipeline.rag.
security` logger'ı, `ERROR` seviyesi) AYRI/filtrelenebilir bir kayıt
düşülür -- ham sızdırılan değer LOGLANMAZ (yalnız hangi desen/kategorinin
yakalandığı, `SanitizationViolationError` mesajı zaten ham değeri
İÇERMİYOR, bkz. `pipeline/rag_sanitize.py`). Bu şemanın (henüz) tüketicisi
yok (`POST /analyze_patient` yazılmadı) -- geriye dönük kırılma YOK.

⚠️ 2026-09-14 (rag-agent-P5) — İKİ EKLEME.

**(1) `CitationGroundingError` YAKALANIYOR (kritik).** 2026-09-13'e kadar
bu modül `call_llm_summarize` çevresinde YALNIZ `LLMNotConfiguredError`
yakalıyordu. Gerçek sağlayıcı açıldığı gün bir halüsinasyon (uydurma
PMID) modülün "HİÇBİR ZAMAN exception fırlatmaz" sözleşmesini kırıp TÜM
pipeline'ı çökertirdi. Artık üç dal var ve ÜÇÜ DE FARKLI şey ifade eder:

  a) `LLMNotConfiguredError` -> **DAVRANIŞ DEĞİŞMEDİ.** `available=True`,
     `llm_summary_available=False`. LLM kapalıyken (bugünkü varsayılan)
     sistem bugünkü hâliyle BİREBİR aynı davranır.
  b) `CitationGroundingError` (uydurma/temellendirilmemiş PMID) ->
     **BLOK-FATAL**, `available=False` + `citation_grounding_violation=
     True` + güvenlik logu. İstek 200 devam eder ve blok
     `summary.unavailable_blocks`'a düşer (`api/analyze_patient.py::
     _build_summary` `available` alanına bakar). **Uydurma özet ASLA
     servis edilmez.**
     📌 Neden `available=False` (ve neden `sanitization_violation` ile
     AYNI desen): bu bir "veri yok" durumu değil, bir **doğruluk
     ihlalidir**. Aynı dosyadaki sanitizasyon-ihlali dalı da retrieval
     başarılı olduğu hâlde `available=False` döner; farklı davranmak iki
     guard'ı tutarsız kılardı.
     📌 Ama **bilgi KAYBOLMAZ**: bu dalda `query`, `retrieved_chunks`,
     `sanitized_context` ve `sources` (orijinal abstract'lar) DOLU döner
     -- yalnız uydurma ÖZET tutulmaz. Jüri "ne getirdiniz, neyi
     reddettiniz" sorusunun ikisine de cevap bulabilir.
  c) Diğer HER hata (sağlayıcı ağ/kota hatası, biçim hatası, beklenmedik
     istisna) -> `available=True`, `llm_summary_available=False`.
     Gerekçe: teknik bir arıza, getirilen literatürü GEÇERSİZ KILMAZ --
     abstract'lar hâlâ gösterilebilir. (b)'den farkı budur.
     ⚠️ Bu dal aynı zamanda modülün "asla fırlatmaz" sözleşmesindeki
     SESSİZ BİR AÇIĞI kapatır: 2026-09-13'te LLM adımının etrafında geniş
     bir `except` YOKTU; sağlayıcı açıldığında bir `TimeoutError` bile
     sözleşmeyi kırardı.

**(2) G3 — Türkçe özet + ORİJİNAL abstract yan yana** (Barış kararı):
`RagPipelineResult.summary_tr` Türkçe özeti, `sources[].abstract_original`
her kaynağın PubMed'den gelen İngilizce abstract'ını taşır.
🔴 **GÜVENLİK AYRIMI -- karıştırılmamalı:** `sources[].abstract_original`
**ASLA prompt'a girmez**; LLM'e giden metin her zaman `sanitized_context`
içindeki redakte edilmiş parçalardır. Orijinal abstract yalnız KENDİ
kullanıcımıza/jürimize dönen yanıtta bulunur. Bu meşrudur çünkü abstract
zaten YAYIMLANMIŞ, herkese açık literatürdür ve HASTAMIZA ait hiçbir
kimlik taşımaz -- sanitizasyonun amacı hasta kimliğinin DIŞARI (harici
sağlayıcıya) çıkmasını engellemektir, kendi ekranımızda yayın metnini
sansürlemek değil.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pipeline.rag_cache import (
    get_cached_summary,
    query_signature,
    set_cached_summary,
)
from pipeline.rag_embed_index import (
    IndexedChunk,
    RetrievedChunk,
    TempRagIndexError,
    build_temp_faiss_index,
    chunk_text,
    retrieve_top_k,
)
from pipeline.rag_llm import (
    CitationGroundingError,
    LLMNotConfiguredError,
    LLMSummaryResult,
    call_llm_summarize,
)
from pipeline.rag_pubmed import PubMedArticle, PubMedFetchError, fetch_pubmed_articles
from pipeline.rag_query import (
    PatientNotFoundError,
    PubMedQueryResult,
    RawPatientProfile,
    normalize_idh1_bucket,
    normalize_mgmt_bucket,
    build_pubmed_query,
    fetch_raw_patient_profile,
)
from pipeline.rag_sanitize import (
    PatientAttributeProfile,
    SanitizationViolationError,
    SanitizedLLMContext,
    build_sanitized_context,
)

_DEFAULT_RETMAX = 30
_DEFAULT_TOP_K = 7

# Güvenlik-ilgili olayların (yalnız sızıntı GİRİŞİMLERİ -- sıradan hata
# değil) AYRI/filtrelenebilir kaydı için (reviewer CRITICAL 2). Ham
# sızdırılan değer buraya YAZILMAZ -- `SanitizationViolationError` mesajı
# zaten ham değeri içermiyor (bkz. pipeline/rag_sanitize.py).
_security_logger = logging.getLogger("pipeline.rag.security")


@dataclass(frozen=True)
class LiteratureSource:
    """G3 (Barış kararı, 2026-09-14): jürinin Türkçe özeti kaynağıyla
    KARŞILAŞTIRABİLMESİ için her kaynağın PubMed'den gelen **orijinal**
    (İngilizce, redakte EDİLMEMİŞ) başlık ve abstract'ı.

    🔴 Bu metin LLM'e GİTMEZ (bkz. modül dokstring'i madde 2) -- prompt'a
    yalnız `sanitized_context` içindeki redakte parçalar girer.
    `retrieved_chunk_count`: bu kaynaktan top-k'ya kaç parça girdi
    (kaynağın sonuca ne kadar katkı verdiğini gösterir)."""

    pmid: str
    title_original: str
    abstract_original: str
    retrieved_chunk_count: int


@dataclass(frozen=True)
class RagPipelineResult:
    available: bool
    reason: str | None
    query: PubMedQueryResult | None = None
    retrieved_chunks: tuple[RetrievedChunk, ...] = field(default_factory=tuple)
    sanitized_context: SanitizedLLMContext | None = None
    llm_summary_available: bool = False
    llm_reason: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    used_cache: bool = False
    # --- G3: Turkce ozet + ORIJINAL abstract'lar (2026-09-14) ----------
    #: Modelin urettigi TURKCE ozet. `llm_summary_available` False iken
    #: HER ZAMAN None -- "yarim/uydurma ozet" servis edilmez.
    summary_tr: str | None = None
    #: Ozetin alintiladigi (ve temellendirme kapisindan GECEN) PMID'ler.
    llm_cited_pmids: tuple[str, ...] = field(default_factory=tuple)
    llm_model_name: str | None = None
    llm_disclaimer: str | None = None
    llm_input_tokens: int | None = None
    llm_output_tokens: int | None = None
    llm_latency_seconds: float | None = None
    #: Getirilen kaynaklarin orijinal (Ingilizce) baslik+abstract'lari.
    #: LLM ozeti uretilmese de DOLUDUR -- literatur yine gosterilebilir.
    sources: tuple[LiteratureSource, ...] = field(default_factory=tuple)
    # reviewer CRITICAL 2 deseninin PMID-temellendirme karsiligi: uydurma
    # bir alinti yakalandiginda `reason` metnini STRING-ESLESTIRMEDEN
    # programatik olarak ayirt edilebilsin.
    citation_grounding_violation: bool = False
    # reviewer CRITICAL 2: gercek bir sanitizasyon-ihlali GIRISIMI ile
    # siradan bir "veri yok" durumunu AYIRT ETMEK icin -- cagiran taraf
    # (orn. POST /analyze_patient, Sultan'ın kapsamı) bunu `reason` metnini
    # STRING-ESLESTIRMEDEN (orn. "KRITIK" onekiyle) programatik olarak
    # kontrol edebilmeli.
    sanitization_violation: bool = False


def _build_attribute_profile(profile: RawPatientProfile) -> PatientAttributeProfile:
    """`RawPatientProfile` (kimlik taşır) -> `PatientAttributeProfile`
    (kimlik TAŞIMAZ). Risk kategorisi bu modülde HESAPLANMAZ -- Cox risk
    skoru `pipeline/cox_model.py`/`api/predict.py`'nin sorumluluğu (bu
    görev o dosyalara DOKUNMUYOR, AKTIF-GOREVLER.md çakışma uyarısı);
    ileride orkestrasyon (`POST /analyze_patient`, Sultan) risk skorunu
    hesaplayıp `risk_category` olarak BU fonksiyona değil, doğrudan
    `PatientAttributeProfile`'a enjekte edebilir -- bu modül şimdilik
    `None` bırakır."""

    return PatientAttributeProfile(
        age_years=profile.age,
        gender=profile.gender,
        kps_score=profile.kps_score,
        mgmt_status_bucket=normalize_mgmt_bucket(profile.mgmt_status_raw),
        idh1_status_bucket=normalize_idh1_bucket(profile.idh1_status_raw),
        risk_category=None,
        has_omics=profile.has_omics,
        molecular_subtype=profile.molecular_subtype,
    )


def _articles_to_chunks(articles: list[PubMedArticle]) -> list[IndexedChunk]:
    chunks: list[IndexedChunk] = []
    for article in articles:
        if not article.abstract:
            continue
        for piece in chunk_text(article.abstract):
            chunks.append(IndexedChunk(pmid=article.pmid, title=article.title, chunk_text=piece))
    return chunks


def _build_sources(
    articles: list[PubMedArticle], retrieved: list[RetrievedChunk]
) -> tuple[LiteratureSource, ...]:
    """G3: top-k'ya giren PMID'ler için ORİJİNAL başlık + abstract.

    Sıra, top-k listesindeki İLK GÖRÜLME sırasıdır (en benzer parça
    önce) -- alfabetik/rastgele değil; jüriye gösterilen sıra, modelin
    gördüğü öncelik sırasıyla aynı olsun diye.

    ⚠️ Abstract'ı DB/önbellekte bulunamayan bir PMID sessizce ATLANMAZ:
    boş bir abstract ile listelenir, çünkü "kaynak getirildi ama metni
    gösterilemiyor" ile "kaynak hiç getirilmedi" farklı şeylerdir."""

    by_pmid = {a.pmid: a for a in articles}
    counts: dict[str, int] = {}
    order: list[str] = []
    for chunk in retrieved:
        if chunk.pmid not in counts:
            counts[chunk.pmid] = 0
            order.append(chunk.pmid)
        counts[chunk.pmid] += 1

    sources: list[LiteratureSource] = []
    for pmid in order:
        article = by_pmid.get(pmid)
        sources.append(
            LiteratureSource(
                pmid=pmid,
                title_original=article.title if article else "",
                abstract_original=article.abstract if article else "",
                retrieved_chunk_count=counts[pmid],
            )
        )
    return tuple(sources)


def generate_literature_summary(
    patient_id: str,
    *,
    retmax: int = _DEFAULT_RETMAX,
    top_k: int = _DEFAULT_TOP_K,
) -> RagPipelineResult:
    """Uçtan uca RAG akışı. DB'ye SADECE readonly SELECT
    (`fetch_raw_patient_profile`). Gerçek LLM çağrısı YAPILMAZ (bkz.
    `pipeline/rag_llm.py`) -- `llm_summary_available` her zaman `False`
    döner, `llm_reason` bunun NEDENİNİ açıklar."""

    notes: list[str] = []

    # 1) Hasta profili + PubMed sorgusu
    try:
        profile = fetch_raw_patient_profile(patient_id)
    except PatientNotFoundError as exc:
        return RagPipelineResult(available=False, reason=str(exc))
    except Exception as exc:  # DB baglanti hatasi vb. -- pipeline COKMEZ
        return RagPipelineResult(
            available=False, reason=f"Hasta profili okunamadi (DB hatasi): {exc!r}"
        )

    query_result = build_pubmed_query(profile)
    notes.extend(query_result.unevaluated_notes)

    attrs = _build_attribute_profile(profile)

    # 2) Onbellek kontrolu -- sorgu-imzasi HASTA KIMLIGI ICERMEZ
    #    (pubmed_boolean_query zaten oznitelik-duzeyinde, bkz. rag_query.py).
    #    Onbellek anahtari BOOLEAN sorguya gore kurulur -- Entrez'e GERCEKTEN
    #    giden budur, ayni boolean sorgu ayni sonucu uretir.
    signature = query_signature(query_result.pubmed_boolean_query)
    cached = get_cached_summary(signature)
    used_cache = False
    articles: list[PubMedArticle]
    if cached is not None and isinstance(cached.get("articles"), list):
        try:
            articles = [
                PubMedArticle(pmid=a["pmid"], title=a["title"], abstract=a["abstract"])
                for a in cached["articles"]
            ]
            used_cache = True
            notes.append("PubMed sonuclari onbellekten (Upstash) okundu.")
        except (KeyError, TypeError):
            articles = []  # bozuk onbellek girdisi -- asagida yeniden cekilecek
    else:
        articles = []

    # 3) PubMed'den cek (onbellek bos/gecersizse)
    if not articles:
        try:
            articles = fetch_pubmed_articles(query_result.pubmed_boolean_query, retmax=retmax)
        except PubMedFetchError as exc:
            return RagPipelineResult(
                available=False,
                reason=f"PubMed erisilemedi/sonuc yok: {exc}",
                query=query_result,
                notes=tuple(notes),
            )
        except Exception as exc:  # beklenmeyen bir hata -- pipeline COKMEZ
            return RagPipelineResult(
                available=False,
                reason=f"PubMed cekme asamasinda beklenmeyen hata: {exc!r}",
                query=query_result,
                notes=tuple(notes),
            )
        cache_payload = {
            "articles": [
                {"pmid": a.pmid, "title": a.title, "abstract": a.abstract} for a in articles
            ]
        }
        if not set_cached_summary(signature, cache_payload):
            notes.append("Onbellege yazilamadi (Upstash erisilemez olabilir) -- performans notu, hata degil.")

    # 4) Chunk + embed + gecici FAISS + top-k
    chunks = _articles_to_chunks(articles)
    if not chunks:
        return RagPipelineResult(
            available=False,
            reason="Cekilen makalelerin hicbirinde kullanilabilir abstract yok.",
            query=query_result,
            notes=tuple(notes),
        )

    try:
        rag_index = build_temp_faiss_index(chunks)
        retrieved = retrieve_top_k(
            rag_index,
            query_result.semantic_query_text,
            k=top_k,
        )
    except TempRagIndexError as exc:
        return RagPipelineResult(
            available=False,
            reason=f"Gecici RAG indeksi kurulamadi/sorgulanamadi: {exc}",
            query=query_result,
            notes=tuple(notes),
        )
    except Exception as exc:  # embedding modeli vb. -- pipeline COKMEZ
        return RagPipelineResult(
            available=False,
            reason=f"Embedding/indeksleme asamasinda beklenmeyen hata: {exc!r}",
            query=query_result,
            notes=tuple(notes),
        )

    # 5) Sanitizasyon (yapisal olarak atlanamaz -- tek giris noktasi)
    try:
        sanitized_context = build_sanitized_context(
            attrs,
            [(c.pmid, c.title, c.chunk_text) for c in retrieved],
        )
    except SanitizationViolationError as exc:
        # reviewer CRITICAL 2: gercek bir sizinti GIRISIMI -- siradan bir
        # hata degil, AYRI/filtrelenebilir bir guvenlik kaydi dusulur (ham
        # sizdirilan deger LOGLANMAZ, exc mesaji zaten onu icermiyor).
        _security_logger.error(
            "RAG sanitizasyon kapisi bir kimlik/kurum sizinti GIRISIMINI "
            "ENGELLEDI (LLM baglamina eklenmedi): %s",
            exc,
        )
        return RagPipelineResult(
            available=False,
            reason=(
                f"KRITIK: sanitizasyon kapisi bir kimlik/kurum deseni "
                f"YAKALADI, LLM baglamina EKLENMEDI: {exc}"
            ),
            query=query_result,
            notes=tuple(notes),
            sanitization_violation=True,
        )

    # G3: orijinal (Ingilizce) abstract'lar -- LLM'e GITMEZ, yalniz yanitta.
    sources = _build_sources(articles, retrieved)

    # 6) LLM cagrisi. Uc dalin gerekcesi modul dokstring'inde (madde 1).
    llm_reason: str | None
    summary: LLMSummaryResult | None = None
    try:
        summary = call_llm_summarize(sanitized_context)
        llm_reason = None
    except LLMNotConfiguredError as exc:
        # (a) LLM KAPALI/yapilandirilmamis -- 2026-09-13 davranisinin BIREBIR AYNISI.
        llm_reason = str(exc)
    except CitationGroundingError as exc:
        # (b) DOGRULUK IHLALI -- uydurma/temellendirilmemis PMID.
        #     BLOK-FATAL: uydurma ozet SERVIS EDILMEZ, istek 200 devam eder.
        _security_logger.error(
            "RAG PMID temellendirme kapisi bir UYDURMA/temellendirilmemis "
            "alintiyi ENGELLEDI (ozet servis EDILMEDI): %s",
            exc,
        )
        return RagPipelineResult(
            available=False,
            reason=(
                f"KRITIK: LLM ozeti PMID temellendirme kapisindan GECEMEDI, "
                f"ozet REDDEDILDI (uydurma ozet servis edilmez): {exc}"
            ),
            query=query_result,
            retrieved_chunks=tuple(retrieved),
            sanitized_context=sanitized_context,
            llm_summary_available=False,
            llm_reason=str(exc),
            notes=tuple(notes),
            used_cache=used_cache,
            sources=sources,
            citation_grounding_violation=True,
        )
    except Exception as exc:
        # (c) TEKNIK ariza (ag/kota/bicim/beklenmedik) -- literaturu
        #     GECERSIZ KILMAZ. Modulun "asla firlatmaz" sozlesmesi burada
        #     korunur.
        llm_reason = f"LLM cagrisi basarisiz: {type(exc).__name__}: {exc}"

    return RagPipelineResult(
        available=True,
        reason=None,
        query=query_result,
        retrieved_chunks=tuple(retrieved),
        sanitized_context=sanitized_context,
        llm_summary_available=summary is not None,
        llm_reason=llm_reason,
        notes=tuple(notes),
        used_cache=used_cache,
        summary_tr=summary.summary_text if summary else None,
        llm_cited_pmids=tuple(summary.cited_pmids) if summary else (),
        llm_model_name=summary.model_name if summary else None,
        llm_disclaimer=summary.disclaimer if summary else None,
        llm_input_tokens=summary.input_tokens if summary else None,
        llm_output_tokens=summary.output_tokens if summary else None,
        llm_latency_seconds=summary.latency_seconds if summary else None,
        sources=sources,
    )
