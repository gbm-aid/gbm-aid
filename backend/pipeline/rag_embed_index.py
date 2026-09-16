"""Hafta 5 — chunking + embedding + geçici FAISS RAG indeksi
(`raw/mimari/v45.txt` Bölüm 8.1 adım 6-8, 8.3; plan.txt satır 540).

Teknoloji (v45.txt Bölüm 8.3, kilitli): `SentenceTransformer
(all-MiniLM-L6-v2)` embedding + `FAISS (geçici indeks)` -- "her sorgu
için yenilenir, kalıcı RAG DB tutulmaz". Bu modül DİSKE HİÇBİR ŞEY
YAZMAZ (RAG indeksi `clinical_radiomics_faiss`/`molecular_omics_faiss`
gibi kalıcı ARTİFACT DEĞİLDİR -- bellek-içi, tek-sorguluk).

Boyut: `all-MiniLM-L6-v2` embedding boyutu 384 -- `clinical_radiomics_
faiss` (93) / `molecular_omics_faiss` (11) ile KARIŞTIRILMAMALI, bu ayrı
bir indeks/uzay (RAG literatür parçaları, hasta-benzerlik indeksleri
DEĞİL).

Vektör sayısı mertebesi (CLAUDE.md "Temel Süreç" md.1): 20-30 makale x
birkaç parça = birkaç yüz vektör -- Flat index yeterli, IVF/HNSW GEREKMEZ.

⚠️ 2026-09-11 DÜZELTMESİ (G2, RAG güvenlik kapanışı görevi): `chunk_text`
hedefi (eski varsayılan 300 token) modelin `max_seq_length`'inden (256,
`tokenizer.model_max_length` ile doğrulandı) YÜKSEKTİ -- 256-300+ token
arası KUYRUK, `embed_texts`'in `[CLS]`/`[SEP]` eklemesiyle birlikte
embedding'e HİÇ girmeden sessizce kırpılıyordu (hata yok, yalnız
retrieval kalitesi bozuluyordu). Varsayılan 200'e ÇEKİLDİ (örtüşmeli/
overlap bölme yerine -- gerekçe `chunk_text` dokstring'inde).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# `all-MiniLM-L6-v2`'nin `max_seq_length` = 256 (tokenizer.model_max_length
# ile doğrulandı, 2026-09-11). ESKİ varsayılan (300) bu sınırın ÜZERİNDEYDİ
# -- `chunk_text` kendi tokenizer'ıyla "en az 300 token" hedefliyordu (kelime
# eklemeye devam edip 300'ü GEÇTİĞİ anda kesiyordu, yani üretilen parça
# genelde 300'den biraz FAZLA oluyordu), ama `embed_texts` (`model.encode`)
# aynı tokenizer'a `[CLS]`/`[SEP]` özel token'larını da EKLİYOR. Sonuç:
# 256-300+ token aralığındaki KUYRUK, embedding'e hiç girmeden SESSİZCE
# kırpılıyordu (hata YOK, yalnız retrieval kalitesi bozuluyordu).
#
# SEÇİM (örtüşmeli/overlap bölme yerine): hedefi 256'nın YETERİNCE altına
# çekmek. Gerekçe: (1) bu modülün literatür parçaları retrieval-bağımsızdır
# -- her PubMed chunk'ı kendi başına bir kanıt parçası, komşu parçalarla
# BAĞLAM SÜREKLİLİĞİ gerektirmiyor (v45.txt Bölüm 8.1: "en yakın 5-7 parça"
# ayrı ayrı skorlanıyor) -- bu yüzden overlap'in tipik faydası (parça sınırı
# ortasında kalan bir cümleyi iki parçada da temsil etmek) burada düşük
# değerli; (2) hedefi düşürmek TEK SATIRLIK, test edilmesi kolay bir
# değişiklik, overlap ise pencere/adım parametreleri + ek testler gerektirir
# -- eşit derecede doğru ama gereksiz karmaşıklık. 200 seçildi: greedy
# döngünün "hedefi geçtiği anda kes" davranışının ürettiği taşma (tek bir
# kelime birden çok alt-token'a bölünebilir) + 2 özel token payı bırakır;
# `tests/test_rag_embed_index.py` bunu gerçek tokenizer ile (özel token'lar
# DAHİL) 256 sınırına karşı doğrular.
_DEFAULT_CHUNK_TARGET_TOKENS = 200
_EMBEDDING_DIM = 384
_MODEL_MAX_SEQ_LENGTH = 256

_model_lock = threading.Lock()
_model_cache: dict[str, object] = {}


def _get_model(model_name: str = _DEFAULT_MODEL_NAME):
    """Tekil (singleton) model yükleyici -- her çağrıda yeniden model
    indirmemek/yüklememek için process-içi cache (thread-safe)."""

    with _model_lock:
        if model_name not in _model_cache:
            from sentence_transformers import SentenceTransformer

            _model_cache[model_name] = SentenceTransformer(model_name)
        return _model_cache[model_name]


def chunk_text(
    text: str,
    *,
    target_tokens: int = _DEFAULT_CHUNK_TARGET_TOKENS,
    model_name: str = _DEFAULT_MODEL_NAME,
) -> list[str]:
    """Metni ~`target_tokens` token'lık parçalara böler.

    Token sayımı, embedding modelinin KENDİ tokenizer'ıyla yapılır (yaklaşık
    bir kelime-sayımı sezgiselinin aksine) -- zaten yüklü olan modelin
    tokenizer'ı yeniden kullanılır, ek bağımlılık GEREKMEZ. Boş/çok kısa
    metin (örn. abstract'sız makale) için boş liste döner (hata değil --
    çağıran taraf bu makaleyi indekse KATMAZ).

    ⚠️ 2026-09-11 (G2 düzeltmesi): `target_tokens` varsayılanı (200),
    modelin `max_seq_length`'inden (256) BİLEREK düşük tutulur -- bu
    fonksiyon `add_special_tokens=False` ile SAYAR ama `embed_texts`
    (`model.encode`) gerçek çağrıda `[CLS]`/`[SEP]` EKLER, ayrıca greedy
    döngü hedefi AŞTIĞI anda kestiği için üretilen parça `target_tokens`'ı
    bir miktar GEÇEBİLİR. `target_tokens`'ı 256'ya YAKIN/ÜZERİNDE elle
    override eden bir çağıran, bu payı kendi sorumluluğunda kaybeder --
    varsayılanın DIŞINA çıkmak isteyen kod `_MODEL_MAX_SEQ_LENGTH`'i
    hesaba katmalıdır."""

    text = text.strip()
    if not text:
        return []

    model = _get_model(model_name)
    tokenizer = model.tokenizer
    words = text.split(" ")

    chunks: list[str] = []
    current_words: list[str] = []
    for word in words:
        current_words.append(word)
        candidate = " ".join(current_words)
        n_tokens = len(tokenizer.encode(candidate, add_special_tokens=False))
        if n_tokens >= target_tokens:
            chunks.append(candidate)
            current_words = []
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


@dataclass(frozen=True)
class IndexedChunk:
    pmid: str
    title: str
    chunk_text: str


def embed_texts(texts: list[str], *, model_name: str = _DEFAULT_MODEL_NAME) -> np.ndarray:
    """`texts` listesini `(n, 384)` float32 embedding matrisine çevirir."""

    if not texts:
        return np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
    model = _get_model(model_name)
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


class TempRagIndexError(RuntimeError):
    """Geçici RAG indeksi boş/kurulamadı -- sessizce boş sonuç DÖNÜLMEZ."""


@dataclass
class TempRagIndex:
    """Bellek-içi, tek-sorguluk FAISS Flat indeksi + eşlik eden metadata.
    DİSKE YAZILMAZ (v45.txt Bölüm 8.3: "her sorgu için yenilenir")."""

    index: object  # faiss.IndexFlatL2
    chunks: tuple[IndexedChunk, ...]
    model_name: str = _DEFAULT_MODEL_NAME


def build_temp_faiss_index(
    indexed_chunks: list[IndexedChunk],
    *,
    model_name: str = _DEFAULT_MODEL_NAME,
) -> TempRagIndex:
    """`indexed_chunks` listesinden bellek-içi Flat L2 FAISS indeksi kurar.

    Boş girdi `TempRagIndexError` fırlatır -- CLAUDE.md "FAISS sonucu
    boşsa sessizce boş dönme" ilkesi (burada: girdi boşsa indeks HİÇ
    kurulmaz, açık hata)."""

    if not indexed_chunks:
        raise TempRagIndexError(
            "Indekslenecek hicbir parca yok (chunking sonrasi bos liste) -- "
            "gecici RAG indeksi KURULMADI."
        )

    import faiss

    texts = [c.chunk_text for c in indexed_chunks]
    vectors = embed_texts(texts, model_name=model_name)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return TempRagIndex(index=index, chunks=tuple(indexed_chunks), model_name=model_name)


@dataclass(frozen=True)
class RetrievedChunk:
    pmid: str
    title: str
    chunk_text: str
    l2_distance: float


def retrieve_top_k(
    rag_index: TempRagIndex,
    query_text: str,
    *,
    k: int = 7,
) -> list[RetrievedChunk]:
    """Hasta profil metnine (`query_text`) en yakın `k` parçayı döner
    (v45.txt Bölüm 8.1 adım 8: "en yakın 5-7 parça").

    `rag_index` boşsa (indeks hiç kurulmadıysa) çağıran taraf zaten
    `TempRagIndexError` almış olur (bkz. `build_temp_faiss_index`); bu
    fonksiyon `k > mevcut_parça_sayısı` durumunda mevcut TÜM parçaları
    (mesafeye göre sıralı) döner, hata FIRLATMAZ."""

    n_chunks = len(rag_index.chunks)
    if n_chunks == 0:
        raise TempRagIndexError("Indeks bos -- sorgulanamiyor.")

    effective_k = min(k, n_chunks)
    query_vector = embed_texts([query_text], model_name=rag_index.model_name)
    distances, indices = rag_index.index.search(query_vector, effective_k)

    results: list[RetrievedChunk] = []
    for dist, idx in zip(distances.ravel(), indices.ravel()):
        if idx == -1:
            continue
        chunk = rag_index.chunks[idx]
        results.append(
            RetrievedChunk(
                pmid=chunk.pmid,
                title=chunk.title,
                chunk_text=chunk.chunk_text,
                l2_distance=float(dist),
            )
        )
    return results
