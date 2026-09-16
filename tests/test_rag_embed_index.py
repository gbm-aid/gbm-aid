"""`pipeline/rag_embed_index.py` testleri -- chunking, embedding, geçici
FAISS indeksi. Model bir kez indirilip process-içi cache'lenir (`sentence-
transformers/all-MiniLM-L6-v2`, ~88MB, ilk testte indirilir/yüklenir)."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.rag_embed_index import (
    IndexedChunk,
    TempRagIndexError,
    build_temp_faiss_index,
    chunk_text,
    embed_texts,
    retrieve_top_k,
)


def test_chunk_text_empty_returns_empty_list() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short_text_single_chunk() -> None:
    chunks = chunk_text("Glioblastoma is an aggressive brain tumor.")
    assert len(chunks) == 1


def test_chunk_text_long_text_produces_multiple_chunks_near_target() -> None:
    # ~1200 kelimelik tekrarlı bir metin -- >200 token'lık birkaç parça beklenir.
    long_text = " ".join(["glioblastoma radiomics survival prognosis MGMT IDH1"] * 300)
    chunks = chunk_text(long_text, target_tokens=200)
    assert len(chunks) >= 2
    # Her parca (son haric) hedefe yakin/uzerinde token sayisi tasimali.
    from pipeline.rag_embed_index import _get_model

    tokenizer = _get_model().tokenizer
    for c in chunks[:-1]:
        n_tokens = len(tokenizer.encode(c, add_special_tokens=False))
        assert n_tokens >= 200 * 0.8  # esnek alt sinir


def test_chunk_text_default_target_stays_within_model_max_seq_length() -> None:
    """G2 (2026-09-11) regresyon testi: `all-MiniLM-L6-v2` `max_seq_length`
    = 256. VARSAYILAN `target_tokens` (artik 200, eski deger 300 idi) ile
    uretilen parcalar, `[CLS]`/`[SEP]` OZEL TOKEN'LARI DAHIL (gercek
    `model.encode` cagrisinin yaptigi gibi) 256'yi ASMAMALI -- aksi halde
    kuyruk sessizce kirpilir (G2'nin orijinal bulgusu)."""

    from pipeline.rag_embed_index import _MODEL_MAX_SEQ_LENGTH, _get_model

    long_text = " ".join(
        ["Glioblastoma MGMT methylation IDH1 wildtype radiomics survival prognosis temozolomide"] * 200
    )
    chunks = chunk_text(long_text)  # varsayilan target_tokens kullanilir
    assert len(chunks) >= 2

    tokenizer = _get_model().tokenizer
    for c in chunks:
        n_tokens_with_special = len(tokenizer.encode(c, add_special_tokens=True))
        assert n_tokens_with_special <= _MODEL_MAX_SEQ_LENGTH, (
            f"Parca ozel token'larla {n_tokens_with_special} token -- "
            f"max_seq_length ({_MODEL_MAX_SEQ_LENGTH}) asildi, embedding "
            f"kuyrugu sessizce kirpilir."
        )


def test_embed_texts_empty_returns_correct_shape() -> None:
    vectors = embed_texts([])
    assert vectors.shape == (0, 384)


def test_embed_texts_returns_expected_dimension() -> None:
    vectors = embed_texts(["glioblastoma MGMT methylated survival"])
    assert vectors.shape == (1, 384)
    assert vectors.dtype == np.float32


def test_build_temp_faiss_index_empty_raises() -> None:
    with pytest.raises(TempRagIndexError):
        build_temp_faiss_index([])


def test_retrieve_top_k_self_query_distance_near_zero() -> None:
    """Self-query testi (CLAUDE.md Temel Süreç md.2): kendi vektörünü
    sorgulayan bir metin, kendi parçasına ~0 mesafede dönmeli."""

    chunks = [
        IndexedChunk(pmid="1", title="A", chunk_text="Glioblastoma MGMT methylation predicts survival."),
        IndexedChunk(pmid="2", title="B", chunk_text="PTEN deletion activates the PI3K AKT pathway in cancer."),
        IndexedChunk(pmid="3", title="C", chunk_text="Radiomics features correlate with tumor grade."),
    ]
    rag_index = build_temp_faiss_index(chunks)
    results = retrieve_top_k(rag_index, "Glioblastoma MGMT methylation predicts survival.", k=3)

    assert len(results) == 3
    assert results[0].pmid == "1"
    assert results[0].l2_distance < 1e-3


def test_retrieve_top_k_returns_fewer_than_k_when_not_enough_chunks() -> None:
    chunks = [IndexedChunk(pmid="1", title="A", chunk_text="Glioblastoma survival radiomics.")]
    rag_index = build_temp_faiss_index(chunks)
    results = retrieve_top_k(rag_index, "glioblastoma", k=7)
    assert len(results) == 1


def test_retrieve_top_k_orders_by_ascending_distance() -> None:
    chunks = [
        IndexedChunk(pmid="1", title="A", chunk_text="Completely unrelated topic about cooking recipes."),
        IndexedChunk(pmid="2", title="B", chunk_text="Glioblastoma MGMT methylation predicts survival outcome."),
        IndexedChunk(pmid="3", title="C", chunk_text="A different unrelated topic about weather patterns."),
    ]
    rag_index = build_temp_faiss_index(chunks)
    results = retrieve_top_k(rag_index, "MGMT methylation glioblastoma survival", k=3)
    distances = [r.l2_distance for r in results]
    assert distances == sorted(distances)
    assert results[0].pmid == "2"
