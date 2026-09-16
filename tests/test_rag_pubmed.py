"""`pipeline/rag_pubmed.py` testleri -- hata yönetimi MOCK'lanır (canlı
PubMed'e bu test dosyasında istek ATILMAZ; canlı doğrulama görev
sırasında ELLE, az sayıda istekle yapıldı -- bkz. final yanıt)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.rag_pubmed import PubMedFetchError, fetch_pubmed_articles


def _fake_entrez_module(*, search_ids: list[str], fetch_articles: list[dict]):
    fake_entrez = MagicMock()
    fake_entrez.esearch.return_value.__enter__ = lambda self: self
    fake_entrez.esearch.return_value.__exit__ = lambda *a: False

    fake_handle = MagicMock()
    fake_entrez.esearch.return_value = fake_handle
    fake_handle.__enter__.return_value = fake_handle
    fake_handle.__exit__.return_value = False

    fake_efetch_handle = MagicMock()
    fake_efetch_handle.__enter__.return_value = fake_efetch_handle
    fake_efetch_handle.__exit__.return_value = False

    fake_entrez.esearch.return_value = fake_handle
    fake_entrez.efetch.return_value = fake_efetch_handle

    def fake_read(handle):
        if handle is fake_handle:
            return {"IdList": search_ids}
        if handle is fake_efetch_handle:
            return {"PubmedArticle": fetch_articles}
        raise AssertionError("unexpected handle")

    fake_entrez.read.side_effect = fake_read
    return fake_entrez


def test_fetch_pubmed_articles_empty_search_result_raises_explicit_error() -> None:
    fake_entrez = _fake_entrez_module(search_ids=[], fetch_articles=[])
    with patch("pipeline.rag_pubmed._configure_entrez", lambda: None), patch.dict(
        "sys.modules", {"Bio": MagicMock(Entrez=fake_entrez)}
    ):
        with pytest.raises(PubMedFetchError, match="sonuc bulunamadi"):
            fetch_pubmed_articles("glioblastoma nonsense query with zero hits")


def test_fetch_pubmed_articles_network_error_raises_pubmed_fetch_error() -> None:
    fake_entrez = MagicMock()
    fake_entrez.esearch.side_effect = ConnectionError("network down")

    with patch("pipeline.rag_pubmed._configure_entrez", lambda: None), patch.dict(
        "sys.modules", {"Bio": MagicMock(Entrez=fake_entrez)}
    ):
        with pytest.raises(PubMedFetchError):
            fetch_pubmed_articles("glioblastoma")


def test_fetch_pubmed_articles_success_parses_title_and_abstract() -> None:
    article_xml = {
        "MedlineCitation": {
            "PMID": "12345",
            "Article": {
                "ArticleTitle": "A study of glioblastoma",
                "Abstract": {"AbstractText": ["Background text.", "Conclusion text."]},
            },
        }
    }
    fake_entrez = _fake_entrez_module(search_ids=["12345"], fetch_articles=[article_xml])

    with patch("pipeline.rag_pubmed._configure_entrez", lambda: None), patch.dict(
        "sys.modules", {"Bio": MagicMock(Entrez=fake_entrez)}
    ):
        articles = fetch_pubmed_articles("glioblastoma")

    assert len(articles) == 1
    assert articles[0].pmid == "12345"
    assert articles[0].title == "A study of glioblastoma"
    assert "Background text." in articles[0].abstract
    assert "Conclusion text." in articles[0].abstract


def test_fetch_pubmed_articles_missing_abstract_returns_empty_string_not_error() -> None:
    article_xml = {
        "MedlineCitation": {
            "PMID": "999",
            "Article": {"ArticleTitle": "An editorial with no abstract"},
        }
    }
    fake_entrez = _fake_entrez_module(search_ids=["999"], fetch_articles=[article_xml])

    with patch("pipeline.rag_pubmed._configure_entrez", lambda: None), patch.dict(
        "sys.modules", {"Bio": MagicMock(Entrez=fake_entrez)}
    ):
        articles = fetch_pubmed_articles("glioblastoma")

    assert len(articles) == 1
    assert articles[0].abstract == ""
