"""Hafta 5 — PubMed Entrez ile makale çekme (`raw/mimari/v45.txt` Bölüm
8.1 adım 5, plan.txt satır 539).

Teknoloji: Biopython (`Bio.Entrez`) -- v45.txt Bölüm 8.3'ün kilitli teknoloji
seçimi ("PubMed Entrez (Biopython)"). `NCBI_API_KEY` proje kökündeki
`.env`'den `db_connection.load_project_environment()` ile okunur, hiçbir
zaman kodda/logda görünmez.

Rate limit (NCBI politikası): API anahtarsız 3 istek/sn, anahtarlı
10 istek/sn. `Bio.Entrez` anahtar varsa bunu otomatik uygular
(`Entrez.api_key` set edildiğinde kütüphane içi throttling'i buna göre
ayarlar). Bu modül ek olarak `pipeline/rag_cache.py`'nin rate-limit
sayacını (Redis/Upstash) KULLANIR (varsa) -- Redis erişilemezse
sessizce/sınırsız devam eder (rate limit sayaç katmanı best-effort'tur,
PubMed'in kendi sunucu tarafı limiti asıl güvencedir).

Hata yönetimi: PubMed'e erişilemezse (ağ hatası, HTTP hatası, boş sonuç,
zaman aşımı) `PubMedFetchError` fırlatılır -- ÇAĞIRAN TARAF (`pipeline/
rag_pipeline.py`) bunu yakalayıp "literatür özeti üretilemedi" ile devam
eder (plan.txt satır 550-551 graceful-degradation sözleşmesi). Bu modülün
kendisi SESSİZCE boş liste DÖNMEZ -- CLAUDE.md: "FAISS sonucu boşsa
sessizce boş dönme" ilkesi burada PubMed'e de uygulanır.
"""

from __future__ import annotations

from dataclasses import dataclass
from http.client import HTTPException
from xml.parsers.expat import ExpatError

from db_connection import load_project_environment


class PubMedFetchError(RuntimeError):
    """PubMed'e erişilemedi / sonuç ayrıştırılamadı / boş sonuç geldi --
    sessizce yutulmaz, çağıran taraf bunu yakalayıp AÇIKÇA raporlamalı."""


# NCBI Entrez politikası e-posta ister (kimlik doğrulama DEĞİL, kötüye
# kullanımda NCBI'nin ulaşabileceği bir iletişim adresi). Gerçek bir
# kişisel e-posta .env'de TANIMLI DEĞİL -- proje-düzeyi, izlenmeyen bir
# yer tutucu kullanılıyor. Barış gerçek bir iletişim adresi tanımlamak
# isterse `NCBI_ENTREZ_EMAIL` .env değişkeni EKLENEBİLİR (bu modül varsa
# onu okur, yoksa yer tutucuya düşer) -- .env'e credential GÖMÜLMEDİ.
_DEFAULT_ENTREZ_EMAIL = "gbm-aid-rag@research.local"

_DEFAULT_RETMAX = 30


@dataclass(frozen=True)
class PubMedArticle:
    pmid: str
    title: str
    abstract: str


def _configure_entrez() -> None:
    import os

    from Bio import Entrez

    load_project_environment()
    Entrez.email = os.environ.get("NCBI_ENTREZ_EMAIL", _DEFAULT_ENTREZ_EMAIL)
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key
    # Anahtar YOKSA Entrez.api_key set EDİLMEZ -- Biopython bu durumda
    # kendiliğinden 3/sn'lik anahtarsız limite düşer (kütüphanenin kendi
    # davranışı, burada ekstra kod GEREKMEZ).


def _extract_abstract(article_xml: dict) -> str:
    """PubmedArticle XML kaydından abstract metnini çıkarır. Yapılandırılmış
    abstract (Background/Methods/Results/Conclusion bölümlü) parçaları
    boşlukla birleştirilir; abstract yoksa boş string döner (hata
    DEĞİL -- bazı makalelerin (editorial/letter) abstract'ı olmaz)."""

    try:
        medline = article_xml["MedlineCitation"]
        article = medline["Article"]
        abstract_node = article.get("Abstract")
        if not abstract_node:
            return ""
        pieces = abstract_node.get("AbstractText", [])
        return " ".join(str(p) for p in pieces).strip()
    except (KeyError, TypeError):
        return ""


def fetch_pubmed_articles(
    query: str,
    *,
    retmax: int = _DEFAULT_RETMAX,
) -> list[PubMedArticle]:
    """`query` ile PubMed'de arama yapar, ilk `retmax` (varsayılan 30)
    makalenin PMID + başlık + abstract'ını döner.

    Boş arama sonucu SESSİZCE boş liste DÖNMEZ -- `PubMedFetchError`
    fırlatır (açık "literatür bulunamadı" durumu, CLAUDE.md ilkesiyle
    tutarlı -- FAISS'in "benzer hasta bulunamadı" kuralının PubMed
    karşılığı)."""

    try:
        from Bio import Entrez
    except ImportError as exc:  # pragma: no cover -- kurulum eksikse
        raise PubMedFetchError(
            "Biopython kurulu degil (`pip install biopython==1.85`) -- "
            "PubMed sorgusu yapilamadi."
        ) from exc

    _configure_entrez()

    try:
        with Entrez.esearch(db="pubmed", term=query, retmax=retmax, sort="relevance") as handle:
            search_result = Entrez.read(handle)
    except (OSError, HTTPException, ExpatError, RuntimeError) as exc:
        raise PubMedFetchError(f"PubMed esearch basarisiz: {exc!r}") from exc

    id_list = search_result.get("IdList", [])
    if not id_list:
        raise PubMedFetchError(
            f"PubMed sorgusu icin sonuc bulunamadi (query={query!r}) -- "
            "acik durum, sessizce bos liste DONULMEDI."
        )

    try:
        with Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml") as handle:
            fetch_result = Entrez.read(handle)
    except (OSError, HTTPException, ExpatError, RuntimeError) as exc:
        raise PubMedFetchError(f"PubMed efetch basarisiz: {exc!r}") from exc

    articles: list[PubMedArticle] = []
    for article_xml in fetch_result.get("PubmedArticle", []):
        try:
            pmid = str(article_xml["MedlineCitation"]["PMID"])
            title = str(article_xml["MedlineCitation"]["Article"].get("ArticleTitle", ""))
        except (KeyError, TypeError):
            continue
        abstract = _extract_abstract(article_xml)
        articles.append(PubMedArticle(pmid=pmid, title=title, abstract=abstract))

    if not articles:
        raise PubMedFetchError(
            f"PubMed {len(id_list)} PMID dondurdu ama hicbiri ayristirilamadi "
            f"(query={query!r}) -- acik hata, sessiz bos liste DONULMEDI."
        )

    return articles
