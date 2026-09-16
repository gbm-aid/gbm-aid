"""`pipeline/rag_query.py` testleri -- saf mantık (DB gerektirmez, `RawPatientProfile`
elle inşa edilir) + gerçek DB'ye karşı bir smoke testi (mevcutsa)."""

from __future__ import annotations

import pytest

from pipeline.rag_query import (
    RawPatientProfile,
    build_pubmed_query,
    normalize_idh1_bucket,
    normalize_mgmt_bucket,
)


def _profile(**overrides) -> RawPatientProfile:
    base = dict(
        patient_id="TCGA-00-0000",
        age=60,
        gender="Male",
        kps_score=80,
        mgmt_status_raw=None,
        idh1_status_raw=None,
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


# ---------------------------------------------------------------------------
# 1) MGMT/IDH1 normalizasyonu -- gevşek vokabüler eşlemesi
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Methylated", "methylated"),
        ("methylated", "methylated"),
        ("Unmethylated", "unmethylated"),
        ("not methylated", "unmethylated"),
        ("Indeterminate", "unknown"),
        (None, "unknown"),
        ("garbage-value", "unknown"),
    ],
)
def test_normalize_mgmt_bucket(raw, expected) -> None:
    assert normalize_mgmt_bucket(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Wildtype", "wildtype"),
        ("wt", "wildtype"),
        ("WT", "wildtype"),
        ("Mutated", "mutant"),
        ("R132H mut", "mutant"),
        ("NOS/NEC", "unknown"),
        ("IDH1 neg, Sequencing required", "unknown"),
        (None, "unknown"),
    ],
)
def test_normalize_idh1_bucket(raw, expected) -> None:
    assert normalize_idh1_bucket(raw) == expected


# ---------------------------------------------------------------------------
# 2) Temel şablon -- mimarinin literal örneğiyle birebir (v45.txt satır 1281)
# ---------------------------------------------------------------------------


def test_base_query_matches_architecture_template_when_both_known() -> None:
    profile = _profile(mgmt_status_raw="Methylated", idh1_status_raw="Wildtype")
    result = build_pubmed_query(profile, year=2024)
    assert result.base_query == "glioblastoma MGMT methylated IDH1 wildtype radiomics survival 2024"
    assert result.full_query == result.base_query  # omics yok -> genisletme YOK
    assert result.unevaluated_notes  # egfr/pten "omics yok" notlari beklenir


def test_unknown_mgmt_and_idh1_are_dropped_from_base_query_with_note() -> None:
    profile = _profile(mgmt_status_raw=None, idh1_status_raw=None)
    result = build_pubmed_query(profile, year=2024)
    assert "MGMT" not in result.base_query
    assert "IDH1" not in result.base_query
    assert result.base_query == "glioblastoma radiomics survival 2024"
    notes_text = " ".join(result.unevaluated_notes)
    assert "MGMT durumu bilinmiyor" in notes_text
    assert "IDH1 durumu bilinmiyor" in notes_text


# ---------------------------------------------------------------------------
# 3) Koşullu genişletme -- EGFR/PTEN proxy (sadece işaret, eşik UYDURULMADI)
# ---------------------------------------------------------------------------


def test_egfr_amplification_proxy_triggers_on_positive_cnv() -> None:
    profile = _profile(has_omics=True, egfr_cnv=0.274, pten_cnv=None)
    result = build_pubmed_query(profile, year=2024)
    assert "EGFR amplification glioblastoma treatment response prognosis" in result.expansion_terms
    egfr_cond = next(c for c in result.conditions if c.name == "egfr_amplification")
    assert egfr_cond.is_proxy is True
    assert egfr_cond.triggered is True


def test_egfr_amplification_proxy_does_not_trigger_on_negative_cnv() -> None:
    profile = _profile(has_omics=True, egfr_cnv=-0.1, pten_cnv=None)
    result = build_pubmed_query(profile, year=2024)
    assert "EGFR amplification glioblastoma treatment response prognosis" not in result.expansion_terms


def test_pten_deletion_proxy_triggers_on_negative_cnv() -> None:
    profile = _profile(has_omics=True, pten_cnv=-0.269, egfr_cnv=None)
    result = build_pubmed_query(profile, year=2024)
    assert "PTEN deletion PI3K AKT pathway glioblastoma resistance" in result.expansion_terms


def test_egfr_and_pten_not_evaluable_without_omics() -> None:
    profile = _profile(has_omics=False)
    result = build_pubmed_query(profile, year=2024)
    assert result.expansion_terms == ()
    egfr_cond = next(c for c in result.conditions if c.name == "egfr_amplification")
    pten_cond = next(c for c in result.conditions if c.name == "pten_deletion")
    assert egfr_cond.evaluable is False
    assert pten_cond.evaluable is False


def test_mgmt_unmethylated_high_expression_never_invents_threshold() -> None:
    """Görev talimatı: 'eşik uydurma' -- bu koşul HER ZAMAN evaluable=False
    döner (kaynakta 'yüksek ekspresyon' için hiçbir eşik tanımlı değil),
    genişletme terimi sorguya ASLA eklenmez."""

    profile = _profile(
        mgmt_status_raw="Unmethylated", has_omics=True, mgmt_expression=999.0
    )
    result = build_pubmed_query(profile, year=2024)
    mgmt_cond = next(c for c in result.conditions if c.name == "mgmt_unmethylated_high_expression")
    assert mgmt_cond.evaluable is False
    assert "MGMT unmethylated high expression temozolomide resistance mechanism" not in result.expansion_terms
    assert "esik" in mgmt_cond.note.lower() or "eşik" in mgmt_cond.note


def test_mgmt_unmethylated_high_expression_not_triggered_when_methylated() -> None:
    profile = _profile(mgmt_status_raw="Methylated", has_omics=True, mgmt_expression=999.0)
    result = build_pubmed_query(profile, year=2024)
    mgmt_cond = next(c for c in result.conditions if c.name == "mgmt_unmethylated_high_expression")
    assert mgmt_cond.evaluable is True
    assert mgmt_cond.triggered is False


def test_full_example_matches_architecture_scenario_3_style() -> None:
    """v45.txt satır 1295-1296 örneğine yakın bir senaryo (MGMT unmethylated,
    IDH1 wildtype, EGFR amp + PTEN del proxy tetiklenir; 'high expression'
    kolu UYDURULMADIĞI için eklenmez -- mimarinin ÖRNEĞİNDEN kasıtlı sapma,
    yukarıdaki testle belgelenmiştir)."""

    profile = _profile(
        mgmt_status_raw="Unmethylated",
        idh1_status_raw="Wildtype",
        has_omics=True,
        egfr_cnv=0.5,
        pten_cnv=-0.3,
    )
    result = build_pubmed_query(profile, year=2024)
    assert result.base_query == "glioblastoma MGMT unmethylated IDH1 wildtype radiomics survival 2024"
    assert "EGFR amplification glioblastoma treatment response prognosis" in result.expansion_terms
    assert "PTEN deletion PI3K AKT pathway glioblastoma resistance" in result.expansion_terms
    assert len(result.expansion_terms) == 2


# ---------------------------------------------------------------------------
# 4) REGRESYON — 2026-08-18 çapraz doğrulama (reviewer) HIGH 3: mimarinin
#    literal şablonu canlı Entrez'de TEK BAŞINA (genişletme/yıl olmadan
#    bile) 0 sonuç veriyordu. `pubmed_boolean_query` artık AYRI, alan-
#    etiketli ve sıfır-sonuç-önleyici bir tasarımla kuruluyor; bu testler
#    o tasarımın DAVRANIŞINI (canlı ölçümle DOĞRULANMIŞ desen) kilitliyor.
#    `semantic_query_text`/`base_query`/`full_query` (mimarinin literal
#    sözlüğü) DEĞİŞMEDİ -- yalnız artık PubMed'e GÖNDERİLMİYOR.
# ---------------------------------------------------------------------------


def test_pubmed_boolean_query_uses_field_tags_and_never_ANDs_both_status_words() -> None:
    """Canlı ölçüldü (2026-08-18): 'glioblastoma[Title/Abstract] AND
    MGMT[Title/Abstract] AND methylated[Title/Abstract] AND IDH1[...] AND
    radiomics[...] AND survival[...]' -> 0 sonuç. `pubmed_boolean_query`
    durum kelimelerini (methylated/wildtype) HİÇ TAŞIMAMALI."""

    profile = _profile(mgmt_status_raw="Methylated", idh1_status_raw="Wildtype")
    result = build_pubmed_query(profile, year=2024)

    assert "methylated" not in result.pubmed_boolean_query.lower()
    assert "wildtype" not in result.pubmed_boolean_query.lower()
    assert "[Title/Abstract]" in result.pubmed_boolean_query
    assert "MGMT[Title/Abstract]" in result.pubmed_boolean_query
    assert "IDH1[Title/Abstract]" in result.pubmed_boolean_query
    # ikisi birden ZORUNLU AND degil, OR ile birlestirilmis olmali
    assert "OR" in result.pubmed_boolean_query


def test_pubmed_boolean_query_single_gene_uses_plain_and_no_or() -> None:
    profile = _profile(mgmt_status_raw="Methylated", idh1_status_raw=None)
    result = build_pubmed_query(profile, year=2024)
    assert "MGMT[Title/Abstract]" in result.pubmed_boolean_query
    assert "IDH1" not in result.pubmed_boolean_query
    assert "OR" not in result.pubmed_boolean_query


def test_pubmed_boolean_query_core_terms_are_field_tagged() -> None:
    profile = _profile(mgmt_status_raw=None, idh1_status_raw=None)
    result = build_pubmed_query(profile, year=2024)
    assert result.pubmed_boolean_query == (
        "glioblastoma[Title/Abstract] AND radiomics[Title/Abstract] "
        "AND survival[Title/Abstract]"
    )


def test_pubmed_boolean_query_does_not_contain_bare_year() -> None:
    """Çıplak yıl PubMed'de bir tarih FİLTRESİ değil, nadiren eşleşen bir
    metin terimi gibi davranıyor (ek sıfır-sonuç riski, canlı ölçüldü) --
    `pubmed_boolean_query`'ye HİÇ EKLENMEZ (yalnız `semantic_query_text`'te
    kalır)."""

    profile = _profile(mgmt_status_raw="Methylated", idh1_status_raw="Wildtype")
    result = build_pubmed_query(profile, year=2024)
    assert "2024" not in result.pubmed_boolean_query
    assert "2024" in result.semantic_query_text


def test_pubmed_boolean_query_expansion_uses_single_gene_word_not_full_phrase() -> None:
    profile = _profile(has_omics=True, egfr_cnv=0.5, pten_cnv=-0.3)
    result = build_pubmed_query(profile, year=2024)
    assert "EGFR[Title/Abstract]" in result.pubmed_boolean_query
    assert "PTEN[Title/Abstract]" in result.pubmed_boolean_query
    # cok-kelimeli betimleyici ifade boolean sorguya GIRMEMELI
    assert "amplification" not in result.pubmed_boolean_query.lower()
    assert "pathway" not in result.pubmed_boolean_query.lower()


def test_semantic_query_text_still_carries_full_architecture_vocabulary() -> None:
    """Mimarinin literal sözlüğü (durum kelimeleri + betimleyici genişletme
    ifadeleri) KAYBOLMADI -- yalnız artık `semantic_query_text`'te (embedding
    yeniden-sıralaması için), `pubmed_boolean_query`'de DEĞİL."""

    profile = _profile(
        mgmt_status_raw="Unmethylated",
        idh1_status_raw="Wildtype",
        has_omics=True,
        egfr_cnv=0.5,
    )
    result = build_pubmed_query(profile, year=2024)
    assert "MGMT unmethylated" in result.semantic_query_text
    assert "IDH1 wildtype" in result.semantic_query_text
    assert "EGFR amplification glioblastoma treatment response prognosis" in result.semantic_query_text
    assert result.semantic_query_text == result.full_query  # full_query geriye-donuk takma ad
