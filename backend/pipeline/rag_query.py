"""Hafta 5 — hasta profilinden koşullu PubMed sorgusu üretimi.

Kaynak: `raw/plan/plan.txt` satır 534-538, `raw/mimari/v45.txt` Bölüm 8.1
("Koşullu RAG Sorgu Genişleme Mantığı", satır 1290-1296).

Temel şablon (v45.txt satır 1281, plan.txt satır 535-536):
    "glioblastoma MGMT [status] IDH1 [status] radiomics survival [year]"

Koşullu genişletme (v45.txt satır 1290-1293):
    EGFR amplifikasyonu varsa      -> "EGFR amplification glioblastoma
                                        treatment response prognosis"
    PTEN delesyonu varsa           -> "PTEN deletion PI3K AKT pathway
                                        glioblastoma resistance"
    MGMT unmethylated + yüksek RNA -> "MGMT unmethylated high expression
                                        temozolomide resistance mechanism"

⚠️ AÇIK BULGU (görev talimatında da işaretlenmişti, db-agent'in
2026-08-18 bulgusuyla ikinci kez doğrulandı): `molecular_scores.
egfr_amp_flag`/`pten_del_flag`/`mgmt_interpretation` DB'de 48/48 NULL --
kaynak veri sözlüğünde (`genom_verileri.xlsx`, 11 sayfa) bu alanların
ÜRETİM KURALI TANIMLI DEĞİL (db-agent, log/2026-08-18.md). Bu modül bu
yüzden mimarinin öngördüğü kanonik bayrakları KULLANAMAZ; iki yol izler:

  1) EGFR amp / PTEN del  -> `omics_profiles.egfr_cnv`/`pten_cnv` HAM
     kolonlarının İŞARETİNE (pozitif=kazanç yönü, negatif=kayıp yönü)
     dayanan bir BEST-EFFORT proxy kullanılır. Bu bir MAGNİTÜD eşiği
     UYDURMAZ (örn. "CNV>0.3 ise amplifikasyon" gibi bir sayı İCAT
     EDİLMEDİ) -- yalnız işaret yönü okunur, ki bu yön zaten
     `pipeline/omics_scores.py::compute_aggressiveness_score()`'ta AYNI
     sözleşmeyle (`max(0, egfr_cnv)`, `max(0, -cdkn2a_cnv)`) kullanılıyor
     ve db-agent tarafından 48/48 hastada doğrulanmış durumda. Yine de bu
     KLİNİK OLARAK DOĞRULANMIŞ bir amplifikasyon/delesyon ÇAĞRISI DEĞİLDİR
     -- yanıtta `is_proxy=True` ile AÇIKÇA işaretlenir.
  2) MGMT unmethylated + YÜKSEK EKSPRESYON -- "yüksek" için kaynakta
     (ne v45.txt'te ne Excel Data Dictionary'de) HİÇBİR MUTLAK/GÖRECELİ
     eşik TANIMLI DEĞİL. Bu modül burada eşik UYDURMAZ: bu koşul her
     zaman `evaluable=False` olarak işaretlenir, ilgili genişletme terimi
     sorguya EKLENMEZ, sebep açık metinle döner.

MGMT/IDH1 sözlük tutarsızlığı (AYRI bir bulgu, bu modülün taşıdığı bir
kapsam DEĞİL -- Cox pipeline'ının `api/predict.py`'sinden BAĞIMSIZ, ayrı
ve GEVŞEK bir normalizasyon burada uygulanır): canlı `patients` tablosunda
`mgmt_status`/`idh1_status` en az 3 farklı kaynağın vokabülerini taşıyor
("Methylated"/"methylated"/"not methylated", "Wildtype"/"wt"/"WT"/
"R132H mut"/"IDH1 neg, Sequencing required"/"NOS/NEC") -- `api/predict.py`
bunların yalnız DAR bir alt kümesini ("Methylated"/"Unmethylated"/
"Indeterminate", "Wildtype"/"Mutated"/"NOS/NEC") tanır ve geri kalanında
`ClinicalCovariateMissingError` fırlatır (Cox modeli için doğru davranış
-- kesinlik gerektirir). RAG sorgu üretimi ise Cox'un AKSİNE bir PubMed
ARAMA TERİMİ ürettiği için (klinik karar değil) daha GEVŞEK bir eşleme
kullanır ve `normalize_mgmt_bucket`/`normalize_idh1_bucket` içinde
AÇIKÇA belgelenir; DB'nin kendi vokabüler tutarsızlığı bu modülün
İCADI DEĞİLDİR, ayrı bir bulgu olarak raporlanmalıdır (db-agent'e).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras

from db_connection import get_connection


class PatientNotFoundError(RuntimeError):
    """`patient_id` `patients` tablosunda yok -- sessizce boş profil
    ÜRETİLMEZ."""


# ---------------------------------------------------------------------------
# 1) Ham hasta profili -- SADECE readonly SELECT.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawPatientProfile:
    """DB'den ÇEKİLEN ham alanlar -- bu tip `patient_id` TAŞIR, LLM'e
    ASLA doğrudan verilmez (bkz. `pipeline/rag_sanitize.py`). Yalnız sorgu
    üretimi ve `PatientAttributeProfile`'a dönüşüm için kullanılır."""

    patient_id: str
    age: int | None
    gender: str | None
    kps_score: int | None
    mgmt_status_raw: str | None
    idh1_status_raw: str | None
    has_omics: bool
    egfr_cnv: float | None
    egfr_expression: float | None
    pten_cnv: float | None
    pten_methylation: float | None
    mgmt_methylation: float | None
    mgmt_expression: float | None
    molecular_subtype: str | None


def fetch_raw_patient_profile(patient_id: str) -> RawPatientProfile:
    """`patients` (+ `has_omics=TRUE` ise `omics_profiles`,
    `molecular_scores.molecular_subtype`) tablolarından readonly SELECT."""

    conn = get_connection(readonly=True)
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT patient_id, age, gender, kps_score, mgmt_status,
                   idh1_status, has_omics
            FROM patients
            WHERE patient_id = %s
            """,
            (patient_id,),
        )
        patient_row = cur.fetchone()
        if patient_row is None:
            cur.close()
            raise PatientNotFoundError(
                f"{patient_id!r} 'patients' tablosunda yok -- RAG sorgusu "
                "üretilemiyor."
            )

        omics_row: dict[str, Any] | None = None
        subtype: str | None = None
        if patient_row["has_omics"]:
            cur.execute(
                """
                SELECT egfr_cnv, egfr_expression, pten_cnv, pten_methylation,
                       mgmt_methylation, mgmt_expression
                FROM omics_profiles
                WHERE patient_id = %s
                """,
                (patient_id,),
            )
            omics_row = cur.fetchone()

            cur.execute(
                "SELECT molecular_subtype FROM molecular_scores WHERE patient_id = %s",
                (patient_id,),
            )
            subtype_row = cur.fetchone()
            subtype = subtype_row["molecular_subtype"] if subtype_row else None
        cur.close()
    finally:
        conn.close()

    def _f(row: dict[str, Any] | None, key: str) -> float | None:
        if row is None or row.get(key) is None:
            return None
        return float(row[key])

    return RawPatientProfile(
        patient_id=patient_row["patient_id"],
        age=patient_row["age"],
        gender=patient_row["gender"],
        kps_score=patient_row["kps_score"],
        mgmt_status_raw=patient_row["mgmt_status"],
        idh1_status_raw=patient_row["idh1_status"],
        has_omics=bool(patient_row["has_omics"]),
        egfr_cnv=_f(omics_row, "egfr_cnv"),
        egfr_expression=_f(omics_row, "egfr_expression"),
        pten_cnv=_f(omics_row, "pten_cnv"),
        pten_methylation=_f(omics_row, "pten_methylation"),
        mgmt_methylation=_f(omics_row, "mgmt_methylation"),
        mgmt_expression=_f(omics_row, "mgmt_expression"),
        molecular_subtype=subtype,
    )


# ---------------------------------------------------------------------------
# 2) Gevşek MGMT/IDH1 vokabüler normalizasyonu (RAG-özel, Cox'tan BAĞIMSIZ)
# ---------------------------------------------------------------------------

_MGMT_METHYLATED_TOKENS = {"methylated"}
_MGMT_UNMETHYLATED_TOKENS = {"unmethylated", "not methylated"}

_IDH1_WILDTYPE_TOKENS = {"wildtype", "wt"}
_IDH1_MUTANT_TOKENS = {"mutated", "mutant", "r132h mut"}
# "IDH1 neg, Sequencing required" ve "NOS/NEC" -> sonuçsuz/bilinmiyor.


def normalize_mgmt_bucket(raw: str | None) -> str:
    """Döner: "methylated" | "unmethylated" | "unknown".

    ⚠️ Bu eşleme Cox modelinin `api/predict.py::MGMT_*_LABEL` sabitleriyle
    AYNI DEĞİLDİR -- RAG sorgusu bir arama terimi ürettiği için daha
    geniş bir vokabüler kümesini (örn. küçük harfli "methylated") de
    "methylated" sayar. Cox pipeline'ı bu genişletmeyi KULLANMAZ/
    KULLANMAMALI (kesinlik farkı, docstring'de yukarıda açıklandı)."""

    if raw is None:
        return "unknown"
    normalized = raw.strip().lower()
    if normalized in _MGMT_METHYLATED_TOKENS:
        return "methylated"
    if normalized in _MGMT_UNMETHYLATED_TOKENS:
        return "unmethylated"
    return "unknown"


def normalize_idh1_bucket(raw: str | None) -> str:
    """Döner: "wildtype" | "mutant" | "unknown"."""

    if raw is None:
        return "unknown"
    normalized = raw.strip().lower()
    if normalized in _IDH1_WILDTYPE_TOKENS:
        return "wildtype"
    if normalized in _IDH1_MUTANT_TOKENS:
        return "mutant"
    return "unknown"


# ---------------------------------------------------------------------------
# 3) Koşullu genişletme değerlendirmesi
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionEvaluation:
    name: str
    evaluable: bool
    triggered: bool
    is_proxy: bool
    note: str


def _evaluate_egfr_amplification(profile: RawPatientProfile) -> ConditionEvaluation:
    if not profile.has_omics or profile.egfr_cnv is None:
        return ConditionEvaluation(
            "egfr_amplification", evaluable=False, triggered=False, is_proxy=False,
            note="Omics verisi yok (has_omics=False veya egfr_cnv NULL) -- degerlendirilemiyor.",
        )
    triggered = profile.egfr_cnv > 0
    return ConditionEvaluation(
        "egfr_amplification",
        evaluable=True,
        triggered=triggered,
        is_proxy=True,
        note=(
            f"PROXY: egfr_cnv={profile.egfr_cnv} isaretine dayanir "
            "(molecular_scores.egfr_amp_flag kaynakta NULL/uretilmemis -- "
            "db-agent 2026-08-18). Bu KLINIK OLARAK DOGRULANMIS bir "
            "amplifikasyon CAGRISI DEGILDIR, yalniz CNV yon okumasidir."
        ),
    )


def _evaluate_pten_deletion(profile: RawPatientProfile) -> ConditionEvaluation:
    if not profile.has_omics or profile.pten_cnv is None:
        return ConditionEvaluation(
            "pten_deletion", evaluable=False, triggered=False, is_proxy=False,
            note="Omics verisi yok (has_omics=False veya pten_cnv NULL) -- degerlendirilemiyor.",
        )
    triggered = profile.pten_cnv < 0
    return ConditionEvaluation(
        "pten_deletion",
        evaluable=True,
        triggered=triggered,
        is_proxy=True,
        note=(
            f"PROXY: pten_cnv={profile.pten_cnv} isaretine dayanir "
            "(molecular_scores.pten_del_flag kaynakta NULL/uretilmemis -- "
            "db-agent 2026-08-18). Klinik olarak dogrulanmis bir delesyon "
            "CAGRISI DEGILDIR."
        ),
    )


def _evaluate_mgmt_unmethylated_high_expression(profile: RawPatientProfile) -> ConditionEvaluation:
    mgmt_bucket = normalize_mgmt_bucket(profile.mgmt_status_raw)
    if mgmt_bucket != "unmethylated":
        return ConditionEvaluation(
            "mgmt_unmethylated_high_expression", evaluable=True, triggered=False, is_proxy=False,
            note=f"MGMT durumu '{mgmt_bucket}' -- kosul tetiklenmedi (unmethylated degil).",
        )
    return ConditionEvaluation(
        "mgmt_unmethylated_high_expression",
        evaluable=False,
        triggered=False,
        is_proxy=False,
        note=(
            "MGMT unmethylated AMA 'yuksek ekspresyon' icin kaynakta "
            "(v45.txt, genom_verileri.xlsx Data Dictionary) HICBIR mutlak/"
            "goreceli esik TANIMLI DEGIL -- bu kosul UYDURULMUS bir esikle "
            "DEGERLENDIRILMEDI, sorguya EKLENMEDI. Karar Baris'a acik soru."
        ),
    )


# ---------------------------------------------------------------------------
# 4) Sorgu inşası
# ---------------------------------------------------------------------------
#
# ⚠️ 2026-08-18 ÇAPRAZ DOĞRULAMA DÜZELTMESİ (reviewer HIGH 3, canlı Entrez
# ile ölçüldü -- bu görevin ilk sürümünün tahmin ettiğinden DAHA KÖTÜ):
# mimarinin literal şablonu ("glioblastoma MGMT [status] IDH1 [status]
# radiomics survival [year]", v45.txt satır 1281) TEK BAŞINA (genişletme
# YOK, yıl YOK) bile 0 sonuç veriyor -- sorun yalnızca "çok fazla ANDlanmış
# terim" değil, çok-kelimeli durum ifadelerinin ("IDH1 wildtype", "MGMT
# methylated") tırnaksız/alan-etiketsiz haliyle PubMed'in otomatik terim
# eşlemesine düşmemesi. Canlı ölçülen sayılar (2026-08-18, retmax=5):
#
#   'glioblastoma MGMT methylated IDH1 wildtype radiomics survival'  -> 0
#   'glioblastoma[Title/Abstract] AND radiomics[...] AND survival[...]' -> 217
#   + AND MGMT[Title/Abstract]                                        -> 42
#   + AND IDH1[Title/Abstract]                                        -> 7
#   + AND (MGMT[Title/Abstract] OR IDH1[Title/Abstract])               -> 48
#   + AND MGMT[...] AND IDH1[...] (ikisi birden ZORUNLU)               -> 0
#
# DÜZELTME (bu modülde UYGULANDI): `PubMedQueryResult` artık İKİ AYRI sorgu
# taşır:
#   - `pubmed_boolean_query`: gerçekten Entrez'e GÖNDERİLEN sorgu.
#     `[Title/Abstract]` alan etiketli, TEK-kelimelik gen adlarıyla (durum
#     kelimesi -- methylated/wildtype -- OLMADAN, çünkü o kelimeler
#     eklenince yukarıdaki ölçümde sonuç sıfıra düşüyor) kurulur. MGMT ve
#     IDH1 ikisi de biliniyorsa OR ile birleştirilir (ikisini birden ZORUNLU
#     kılmak sıfır sonuca yol açıyor, ölçüldü).
#   - `semantic_query_text`: mimarinin literal/betimleyici sözlüğünü
#     (durum kelimeleri + EGFR/PTEN/MGMT-yüksek-ekspresyon genişletme
#     ifadeleri DAHİL) TAŞIR ama PubMed'e GÖNDERİLMEZ -- yalnız `pipeline/
#     rag_embed_index.py::retrieve_top_k`'nin embedding-tabanlı yeniden
#     sıralamasında (hastaya özgü kişiselleştirme burada, boolean arama
#     KATMANINDA değil, semantik KATMANDA yapılır) kullanılır.
#
# 🔴 Bu, `raw/mimari/v45.txt` satır 1281'in LİTERAL şablonundan BİLİNÇLİ bir
# SAPMADIR -- mimari otoritesi gereği burada YAZILI KARAR haline
# GETİRİLMEDİ (`concepts/mimari-revizyonlari.md`'ye yazılmadı), yalnızca
# ADAY olarak İŞARETLENDİ; Barış'ın onayı olmadan kalıcı revizyon
# SAYILMAZ. `expansion_terms`/`base_query` alanları geriye dönük uyumluluk
# için KORUNDU (artık yalnızca semantik metnin bileşenleri, boolean sorguya
# GİRMİYORLAR).

_EGFR_EXPANSION_TERM = "EGFR amplification glioblastoma treatment response prognosis"
_PTEN_EXPANSION_TERM = "PTEN deletion PI3K AKT pathway glioblastoma resistance"
_MGMT_HIGH_EXP_EXPANSION_TERM = (
    "MGMT unmethylated high expression temozolomide resistance mechanism"
)

_CORE_BOOLEAN_TERMS = (
    "glioblastoma[Title/Abstract]",
    "radiomics[Title/Abstract]",
    "survival[Title/Abstract]",
)


@dataclass(frozen=True)
class PubMedQueryResult:
    base_query: str
    expansion_terms: tuple[str, ...]
    full_query: str  # GERİYE DÖNÜK UYUMLULUK İÇİN KORUNDU -- artık semantic_query_text ile AYNI (PubMed'e GÖNDERİLMEZ, bkz. pubmed_boolean_query)
    pubmed_boolean_query: str  # Entrez'e GERÇEKTEN gönderilen sorgu (alan-etiketli, sıfır-sonuç-önleyici tasarım)
    semantic_query_text: str  # embedding-tabanlı yeniden sıralama icin (base_query + expansion_terms, PubMed'e GİTMEZ)
    mgmt_bucket: str
    idh1_bucket: str
    conditions: tuple[ConditionEvaluation, ...]
    unevaluated_notes: tuple[str, ...]


def build_pubmed_query(
    profile: RawPatientProfile,
    *,
    year: int | None = None,
) -> PubMedQueryResult:
    """v45.txt Bölüm 8.1 şablonunun SEMANTİK/betimleyici sözlüğünü korur,
    ama Entrez'e giden BOOLEAN sorguyu ayrı ve sıfır-sonuç-vermeyecek
    şekilde kurar (bkz. modülün bu bölümdeki üst dokstring'i -- reviewer
    HIGH 3 düzeltmesi).

    `year` verilmezse UTC bugünkü yıl kullanılır (yalnız `semantic_query_
    text`'e eklenir -- `pubmed_boolean_query`'ye EKLENMEZ, çünkü çıplak yıl
    metni PubMed'de bir tarih FİLTRESİ değil bir metin terimi gibi
    davranıyor ve nadiren abstract'ta geçiyor, ölçülen ek bir sıfır-sonuç
    riski)."""

    resolved_year = year if year is not None else datetime.now(timezone.utc).year

    mgmt_bucket = normalize_mgmt_bucket(profile.mgmt_status_raw)
    idh1_bucket = normalize_idh1_bucket(profile.idh1_status_raw)

    # --- semantik/betimleyici metin (mimarinin literal sözlüğü, KORUNDU) ---
    base_terms = ["glioblastoma"]
    unevaluated_notes: list[str] = []
    if mgmt_bucket == "unknown":
        unevaluated_notes.append(
            "MGMT durumu bilinmiyor (NULL veya taninmayan deger) -- temel "
            "sorgudan MGMT niteleyicisi CIKARILDI (mimarinin literal "
            "sablonundan bilincli sapma, belirsiz bilgi ILERI SURULMEDI)."
        )
    else:
        base_terms.append(f"MGMT {mgmt_bucket}")

    if idh1_bucket == "unknown":
        unevaluated_notes.append(
            "IDH1 durumu bilinmiyor (NULL veya taninmayan deger) -- temel "
            "sorgudan IDH1 niteleyicisi CIKARILDI."
        )
    else:
        base_terms.append(f"IDH1 {idh1_bucket}")

    base_terms.append("radiomics survival")
    base_terms.append(str(resolved_year))
    base_query = " ".join(base_terms)

    egfr_eval = _evaluate_egfr_amplification(profile)
    pten_eval = _evaluate_pten_deletion(profile)
    mgmt_high_exp_eval = _evaluate_mgmt_unmethylated_high_expression(profile)
    conditions = (egfr_eval, pten_eval, mgmt_high_exp_eval)

    expansion_terms: list[str] = []
    if egfr_eval.evaluable and egfr_eval.triggered:
        expansion_terms.append(_EGFR_EXPANSION_TERM)
    if pten_eval.evaluable and pten_eval.triggered:
        expansion_terms.append(_PTEN_EXPANSION_TERM)
    if mgmt_high_exp_eval.evaluable and mgmt_high_exp_eval.triggered:
        expansion_terms.append(_MGMT_HIGH_EXP_EXPANSION_TERM)  # şu an hiçbir zaman evaluable=True dönmez

    for cond in conditions:
        if not cond.evaluable:
            unevaluated_notes.append(f"{cond.name}: {cond.note}")

    semantic_query_text = " ".join([base_query, *expansion_terms]).strip()

    # --- Entrez'e GERÇEKTEN gönderilen boolean sorgu (sıfır-sonuç-önleyici) ---
    boolean_parts = list(_CORE_BOOLEAN_TERMS)
    gene_terms: list[str] = []
    if mgmt_bucket != "unknown":
        gene_terms.append("MGMT[Title/Abstract]")
    if idh1_bucket != "unknown":
        gene_terms.append("IDH1[Title/Abstract]")
    if len(gene_terms) == 1:
        boolean_parts.append(gene_terms[0])
    elif len(gene_terms) >= 2:
        boolean_parts.append("(" + " OR ".join(gene_terms) + ")")
    # Kosullu genisletme -- boolean sorguya TEK-kelimelik gen adiyla eklenir
    # (coklu-kelimeli tanimlayici ifadeyle DEGIL, ayni sifir-sonuc riski).
    if egfr_eval.evaluable and egfr_eval.triggered:
        boolean_parts.append("EGFR[Title/Abstract]")
    if pten_eval.evaluable and pten_eval.triggered:
        boolean_parts.append("PTEN[Title/Abstract]")
    pubmed_boolean_query = " AND ".join(boolean_parts)

    return PubMedQueryResult(
        base_query=base_query,
        expansion_terms=tuple(expansion_terms),
        full_query=semantic_query_text,
        pubmed_boolean_query=pubmed_boolean_query,
        semantic_query_text=semantic_query_text,
        mgmt_bucket=mgmt_bucket,
        idh1_bucket=idh1_bucket,
        conditions=conditions,
        unevaluated_notes=tuple(unevaluated_notes),
    )


def build_pubmed_query_for_patient(patient_id: str, *, year: int | None = None) -> PubMedQueryResult:
    """`fetch_raw_patient_profile` + `build_pubmed_query` -- kolaylık
    sarmalayıcısı (DB'ye SADECE readonly SELECT)."""

    profile = fetch_raw_patient_profile(patient_id)
    return build_pubmed_query(profile, year=year)
