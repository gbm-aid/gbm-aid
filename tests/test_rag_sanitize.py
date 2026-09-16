"""Hafta 5 — Prompt Sanitizasyon Katmanı testleri (`pipeline/rag_sanitize.py`).

EN KRİTİK TEST DOSYASI (görev talimatı): hasta ID'lerinin sızmadığını
HER kaynak formatı için AYRI kanıtlar -- TCGA-NN-NNNN, UPENN-GBM-NNNNN,
UCSF-PDGM-NNN, Patient-NNN (LUMIERE).
"""

from __future__ import annotations

import pytest

from pipeline.rag_sanitize import (
    PatientAttributeProfile,
    SanitizationViolationError,
    assert_no_identifiers_leaked,
    assert_patient_context_has_no_digit_clusters,
    build_sanitized_context,
    render_patient_context_text,
    sanitize_free_text,
)

# ---------------------------------------------------------------------------
# 1) Her hasta-ID formatı AYRI test edilir (canlı DB'de doğrulanmış formatlar,
#    2026-08-18: TCGA-02-0047, UPENN-GBM-00123, UCSF-PDGM-190, Patient-001).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "patient_id",
    [
        "TCGA-02-0047",
        "TCGA-06-5412",
        "UPENN-GBM-00123",
        "UPENN-GBM-00566",
        "UCSF-PDGM-190",
        "UCSF-PDGM-115",
        "Patient-001",
        "Patient-060",
    ],
)
def test_sanitize_free_text_removes_patient_id(patient_id: str) -> None:
    text = f"Hasta {patient_id}, 62 yaşında, IDH-wildtype, MGMT-metile."
    cleaned, counts = sanitize_free_text(text)

    assert patient_id not in cleaned
    assert sum(v for k, v in counts.items() if k.startswith("patient_id")) >= 1


@pytest.mark.parametrize(
    "patient_id",
    [
        "TCGA-02-0047",
        "UPENN-GBM-00123",
        "UCSF-PDGM-190",
        "Patient-001",
    ],
)
def test_assert_no_identifiers_leaked_raises_on_raw_id(patient_id: str) -> None:
    text = f"Hasta {patient_id} 2009 tarihinde tarandı."
    with pytest.raises(SanitizationViolationError):
        assert_no_identifiers_leaked(text)


@pytest.mark.parametrize(
    "patient_id",
    [
        "TCGA-02-0047",
        "UPENN-GBM-00123",
        "UCSF-PDGM-190",
        "Patient-001",
    ],
)
def test_assert_no_identifiers_leaked_passes_after_sanitization(patient_id: str) -> None:
    text = f"Hasta {patient_id} 2009 tarihinde tarandı, UCSF'de tedavi gördü."
    cleaned, _ = sanitize_free_text(text)
    # sanitize sonrası kapı ARTIK exception fırlatmamalı.
    assert_no_identifiers_leaked(cleaned)


# ---------------------------------------------------------------------------
# 2) Tarih desenleri
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "date_text",
    [
        "2009-03-15",
        "03/15/2009",
        "March 15, 2009",
        "15 March 2009",
        "March 2009",
        "2009",
    ],
)
def test_sanitize_free_text_removes_dates(date_text: str) -> None:
    text = f"Tarama tarihi: {date_text}."
    cleaned, counts = sanitize_free_text(text)
    assert date_text not in cleaned
    assert counts.get("date", 0) >= 1


# ---------------------------------------------------------------------------
# 3) Kurum/kaynak adları
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "institution",
    ["TCGA", "UPenn", "UCSF", "LUMIERE", "University of Pennsylvania"],
)
def test_sanitize_free_text_removes_institution_names(institution: str) -> None:
    text = f"Hasta {institution} kohortundan alındı."
    cleaned, counts = sanitize_free_text(text)
    assert institution not in cleaned
    assert counts.get("institution", 0) >= 1


# ---------------------------------------------------------------------------
# 4) Birincil savunma -- tip-düzeyi allowlist: `PatientAttributeProfile`
#    hiçbir şekilde patient_id/tarih/kurum ALANI TAŞIYAMAZ.
# ---------------------------------------------------------------------------


def test_patient_attribute_profile_has_no_identifying_fields() -> None:
    field_names = set(PatientAttributeProfile.__dataclass_fields__.keys())
    forbidden = {"patient_id", "scan_date", "source_dataset", "center", "id", "source"}
    assert field_names.isdisjoint(forbidden), (
        f"PatientAttributeProfile kimlik-taşıyan alan(lar) içeriyor: "
        f"{field_names & forbidden}"
    )


def test_patient_attribute_profile_rejects_unknown_kwarg() -> None:
    with pytest.raises(TypeError):
        PatientAttributeProfile(  # type: ignore[call-arg]
            age_years=62,
            gender="Male",
            kps_score=90,
            mgmt_status_bucket="methylated",
            idh1_status_bucket="wildtype",
            risk_category="orta",
            patient_id="TCGA-06-5412",  # <- yapısal olarak KABUL EDİLMEMELİ
        )


def test_render_patient_context_text_never_contains_id_even_if_smuggled_via_subtype() -> None:
    # molecular_subtype gibi serbest-metin alanlarına kazara bir ID
    # yazılırsa dahi (örn. yanlış bir DB satırı) render fonksiyonu bunu
    # olduğu gibi geçirir -- BU YÜZDEN render'ın YALNIZ sabit-şablonlu
    # alanları kullandığı, id/tarih/kurum ALANI HİÇ OKUMADIĞI ayrıca
    # doğrulanır (aşağıdaki pozitif test).
    attrs = PatientAttributeProfile(
        age_years=62,
        gender="Male",
        kps_score=90,
        mgmt_status_bucket="methylated",
        idh1_status_bucket="wildtype",
        risk_category="orta",
        has_omics=True,
        molecular_subtype="parp_inhibitor_candidate",
    )
    text = render_patient_context_text(attrs)
    assert "TCGA" not in text
    assert "UPENN" not in text.upper()
    assert "Patient-" not in text


# ---------------------------------------------------------------------------
# 5) İkincil savunma -- `build_sanitized_context` uçtan uca kapı testi.
# ---------------------------------------------------------------------------


def test_build_sanitized_context_end_to_end_no_leak() -> None:
    attrs = PatientAttributeProfile(
        age_years=62,
        gender="Female",
        kps_score=80,
        mgmt_status_bucket="unmethylated",
        idh1_status_bucket="wildtype",
        risk_category="yüksek",
        has_omics=False,
    )
    literature = [
        ("12345678", "A study of glioblastoma TCGA-02-0047 outcomes", "Patients from UCSF in 2009 showed..."),
    ]
    ctx = build_sanitized_context(attrs, literature)

    assert "TCGA-02-0047" not in ctx.literature_snippets[0].title
    assert "UCSF" not in ctx.literature_snippets[0].text
    assert "2009" not in ctx.literature_snippets[0].text
    assert "TCGA" not in ctx.patient_context_text
    assert "62 yaşında" in ctx.patient_context_text
    assert "MGMT unmethylated" in ctx.patient_context_text
    assert "IDH1 wildtype" in ctx.patient_context_text


def test_sanitize_free_text_does_not_leak_the_stripped_value_via_counts() -> None:
    text = "TCGA-02-0047"
    _, counts = sanitize_free_text(text)
    # counts sözlüğü SADECE sayı taşımalı, orijinal ID değerini DEĞİL.
    for value in counts.values():
        assert isinstance(value, int)
    assert "TCGA-02-0047" not in str(counts)


# ---------------------------------------------------------------------------
# 6) REGRESYON — 2026-08-18 çapraz doğrulama (reviewer) CRITICAL 1: canlı
#    kanıtlanmış 4 bypass senaryosu. Bunların HİÇBİRİ tekrar geçmemeli --
#    ne redaksiyondan (`sanitize_free_text`) ne de son-kontrol kapısından
#    (`assert_no_identifiers_leaked`).
# ---------------------------------------------------------------------------


def test_bypass_regression_underscore_separator() -> None:
    text = "TCGA_06_5412 patient"
    cleaned, counts = sanitize_free_text(text)
    assert "TCGA_06_5412" not in cleaned
    assert "5412" not in cleaned
    assert sum(counts.values()) >= 1
    assert_no_identifiers_leaked(cleaned)  # kapı da artık gecmemeli -- exception YOK demek gecti


def test_bypass_regression_no_separator() -> None:
    text = "TCGA065412 patient"
    cleaned, counts = sanitize_free_text(text)
    assert "TCGA065412" not in cleaned
    assert "5412" not in cleaned
    assert_no_identifiers_leaked(cleaned)


def test_bypass_regression_compact_iso_date() -> None:
    text = "20090315 scan"
    cleaned, counts = sanitize_free_text(text)
    assert "20090315" not in cleaned
    assert counts.get("date", 0) >= 1


def test_bypass_regression_bare_id_detached_from_institution_token() -> None:
    text = "06-5412 from TCGA cohort"
    cleaned, counts = sanitize_free_text(text)
    assert "TCGA" not in cleaned
    assert "06-5412" not in cleaned
    assert "5412" not in cleaned
    assert_no_identifiers_leaked(cleaned)


# ---------------------------------------------------------------------------
# 2026-08-19'da ACILAN, 2026-09-11'DE KAPATILAN 4 residuel bypass.
#
# GECMIS (xfail donemi): reviewer capraz dogrulamasi (2026-08-19) dort
# residuel bypass'i CANLI olarak dogrulamisti -- hem `sanitize_free_text()`
# rakam obegini metinde BIRAKIYORDU hem de `assert_no_identifiers_leaked()`
# ayni desen setini paylastigi icin exception FIRLATMIYORDU. O donemde bu
# 4 form + Unicode kapi bulgusu icin TOPLAM 9 `xfail(strict=True)` test
# ornekli (parametrized) nobette tutulmustu (reviewer'in acik onerisi:
# "en azindan xfail/skip isaretli regresyon testleri ekleyin").
#
# 2026-09-11 KAPANIS: `pipeline/rag_sanitize.py`'deki 3 yapisal degisiklik
# (ayrac sinifini `{0,3}` tekrara VE Unicode tireye genislet + token-
# listesinden ILERI/TERS regex uret + bare-residual yakinligini cumle
# sinirina da bak) butun 9 ornegi GERCEK PASS yapti. Asagidaki testler
# ARTIK xfail DEGIL -- gercek regresyon testleridir. `strict=True`'nin
# vaat ettigi gibi, bu donusum XPASS->HATA uzerinden ZORLA fark edildi
# (bkz. log/2026-09-11.md).
# ---------------------------------------------------------------------------

_RESIDUAL_BYPASSES = [
    pytest.param("TCGA--06--5412 patient", "5412", id="cift-ayrac"),
    pytest.param("5412-06-TCGA from cohort", "5412", id="ters-sira"),
    pytest.param("06-5412 from" + " " * 90 + "TCGA cohort", "06-5412", id="pencere-kenari"),
    pytest.param("TCGA－06－5412 patient", "5412", id="unicode-tam-genislik-tire"),
]


@pytest.mark.parametrize("text,leaked_fragment", _RESIDUAL_BYPASSES)
def test_residual_bypass_redaction_now_closed(text: str, leaked_fragment: str) -> None:
    """Redaksiyon katmani: rakam obegi metinde KALMAMALI (2026-09-11
    oncesi KALIYORDU -- eski xfail: `test_residual_bypass_redaction_
    known_open`)."""

    cleaned, counts = sanitize_free_text(text)
    assert leaked_fragment not in cleaned
    assert sum(counts.values()) >= 1


@pytest.mark.parametrize("text,leaked_fragment", _RESIDUAL_BYPASSES)
def test_residual_bypass_gate_rejects_raw_form_now_closed(text: str, leaked_fragment: str) -> None:
    """Kapi katmani: `assert_no_identifiers_leaked`, HAM (sanitize
    edilmemis) 4 residuel formu da dogrudan YAKALAR -- redaksiyon
    ATLANIP dogrudan kapiya gelen bozuk bir kod yolu simulasyonu.

    NOT (test tasarim degisikligi): eski surum (`test_residual_bypass_
    gate_known_open`) bu kapiyi SANITIZE EDILMIS metin uzerinde test
    ediyordu -- o senaryonun onkosulu "redaksiyon ID'yi silmede
    basarisiz oluyor, kapi da onu yakalamiyor" idi. Redaksiyon artik
    ID'yi TAM siliyor, yani sanitize-sonrasi metinde yakalanacak bir sey
    KALMIYOR (bu, ayri asagidaki `test_residual_bypass_gate_passes_on_
    cleaned_text_now_that_redaction_is_complete` ile pozitif olarak
    dogrulaniyor). Kapinin BAGIMSIZ degerini olcmenin doğru yeri HAM
    metindir -- redaksiyon ATLANIRSA/basarisiz olursa kapi yine de
    yakalamali; bu asagida dogrulaniyor."""

    with pytest.raises(SanitizationViolationError):
        assert_no_identifiers_leaked(text)


@pytest.mark.parametrize("text,leaked_fragment", _RESIDUAL_BYPASSES)
def test_residual_bypass_gate_passes_on_cleaned_text_now_that_redaction_is_complete(
    text: str, leaked_fragment: str
) -> None:
    """Redaksiyon TAM oldugu icin sanitize-sonrasi metin kapiyi
    tetiklememeli (aksi halde redaksiyon her zaman kapiyi da tetikleyen
    bir sonsuz dongu olurdu)."""

    cleaned, _counts = sanitize_free_text(text)
    assert_no_identifiers_leaked(cleaned)  # exception firlatmamali


# --- BIRINCIL KAPININ GERCEK KAPSAMI (2026-08-19 OLCUMU, 2026-09-11
#     UNICODE DUZELTMESIYLE TAMAMLANDI) ----------------------------------
#
# 2026-08-19 olcumu: birincil kapi (`assert_patient_context_has_no_digit_
# clusters`) dort residuel formdan UCUNU yakaliyordu, Unicode tam-genislik
# tire (U+FF0D) formunu KACIRIYORDU (ayrac karakter sinifinda olmadigi
# icin rakam dizisi "06" + "5412" diye bolunuyor, kalan "5412" tek basina
# min-5-karakter kuralini saglamiyordu). 2026-09-11: `_DIGIT_CLUSTER_
# PATTERN` ayni Unicode tire kumesini (`_UNICODE_DASH_CHARS`) alarak bu
# ayrimi kapatti -- desen KODU hala BAGIMSIZ (enumere listeden ayri), ama
# karakter KAPSAMI artik `_SEP_CLASS` ile senkron.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("TCGA--06--5412 patient", id="cift-ayrac"),
        pytest.param("5412-06-TCGA from cohort", id="ters-sira"),
        pytest.param("06-5412 from" + " " * 90 + "TCGA cohort", id="pencere-kenari"),
        pytest.param("TCGA－06－5412 patient", id="unicode-tam-genislik-tire"),
    ],
)
def test_patient_context_gate_catches_all_four_residual_forms_now_closed(text: str) -> None:
    """Birincil kapi artik DORT residuel formun TAMAMINI yakaliyor (2026-
    08-19'da uc/dort idi, eski xfail: `test_patient_context_gate_misses_
    unicode_dash_form_known_open`)."""

    with pytest.raises(SanitizationViolationError):
        assert_patient_context_has_no_digit_clusters(text)


def test_patient_context_gate_still_catches_novel_ascii_id_shape() -> None:
    """POZITIF KONTROL: kapinin ASIL degeri -- enumere desen listesinde
    OLMAYAN, tamamen yeni bir ASCII ID sekli yine de yakalaniyor."""

    with pytest.raises(SanitizationViolationError):
        assert_patient_context_has_no_digit_clusters("NEWSITE-4471-0093 hasta")


def test_bypass_regression_gate_alone_also_rejects_all_four_raw_forms() -> None:
    """Kapı (`assert_no_identifiers_leaked`), SANITIZE EDİLMEMİŞ ham
    metinlerde de artık bu 4 formu YAKALAR (redaksiyon atlanıp doğrudan
    kapıya gelen bozuk bir kod yolu simülasyonu)."""

    raw_forms = [
        "TCGA_06_5412 patient",
        "TCGA065412 patient",
    ]
    for raw in raw_forms:
        with pytest.raises(SanitizationViolationError):
            assert_no_identifiers_leaked(raw)


# ---------------------------------------------------------------------------
# 2026-09-11 EKLENDI -- YENI KARSIT-ORNEKLER (gorev talimati: en az 5).
# Bilinen 4 residuel formun VARYASYONLARI -- ayni kok neden siniflarini
# (ayrac tekrari, Unicode tire cesitliligi, pencere mesafesi, karisik
# ayrac, ters-sira+Unicode kombinasyonu) FARKLI somut girdilerle test eder.
# DURUSTLUK: bu, "her yeni kimlik seklini" KAPSADIGI anlamina GELMEZ --
# yalnizca BILINEN 4 form sinifinin ek somut orneklerle dogrulanmasidir.
# ---------------------------------------------------------------------------


def test_new_counterexample_triple_separator() -> None:
    """Ucuz ayrac tekrari (`---`) -- cift ayracin (`--`) bir adim otesi,
    `_SEP`'in {0,3} ust siniri icinde kaliyor mu diye test eder."""

    text = "TCGA---06---5412 patient"
    cleaned, _counts = sanitize_free_text(text)
    assert "5412" not in cleaned
    with pytest.raises(SanitizationViolationError):
        assert_no_identifiers_leaked(text)  # ham metin kapiyi tetiklemeli
    assert_no_identifiers_leaked(cleaned)  # temizlenmis metin tetiklememeli


def test_new_counterexample_unicode_minus_sign() -> None:
    """U+2212 (MINUS SIGN) -- tam-genislik tireden (U+FF0D) FARKLI bir
    Unicode tire karakteri; `_UNICODE_DASH_CHARS` kumesinin tek bir
    karaktere ozel olmadigini dogrular."""

    text = "TCGA−06−5412 patient"
    cleaned, _counts = sanitize_free_text(text)
    assert "5412" not in cleaned
    with pytest.raises(SanitizationViolationError):
        assert_no_identifiers_leaked(text)
    assert_no_identifiers_leaked(cleaned)


def test_new_counterexample_120_char_distance_from_institution() -> None:
    """Pencere-kenari testinin (90 karakter) OTESINDE bir mesafe (120
    karakter) -- cumle-sinirli kural karakter sayisindan BAGIMSIZ oldugu
    icin sabit pencereyi (80) sirf buyutmekten farkli olarak burada da
    calismali."""

    text = "06-5412 from" + " " * 120 + "TCGA cohort, no sentence break in between"
    cleaned, _counts = sanitize_free_text(text)
    assert "06-5412" not in cleaned
    assert "5412" not in cleaned
    assert_no_identifiers_leaked(cleaned)


def test_new_counterexample_mixed_separator_chars() -> None:
    """Karisik ayrac: ilk ayrac ASCII tire, ikincisi alt cizgi
    (`TCGA-06_5412`) -- `_SEP`'in HER iki taraf icin de BAGIMSIZ
    esletigini (ayni karakterin tekrarini ZORUNLU KILMADIGINI) dogrular."""

    text = "TCGA-06_5412 patient"
    cleaned, _counts = sanitize_free_text(text)
    assert "5412" not in cleaned
    with pytest.raises(SanitizationViolationError):
        assert_no_identifiers_leaked(text)
    assert_no_identifiers_leaked(cleaned)


def test_new_counterexample_reversed_order_with_unicode_dash() -> None:
    """Ters-sira VE Unicode tire kombinasyonu (iki kok nedenin BIRLIKTE
    gorulmesi) -- '_build_id_patterns' otomatik-ters-sira uretiminin
    Unicode ayrac genislemesiyle BAGIMSIZ calistigini dogrular."""

    text = "5412－06－TCGA from cohort"
    cleaned, _counts = sanitize_free_text(text)
    assert "5412" not in cleaned
    with pytest.raises(SanitizationViolationError):
        assert_no_identifiers_leaked(text)
    assert_no_identifiers_leaked(cleaned)


# ---------------------------------------------------------------------------
# 2026-09-11 EKLENDI -- YANLIS-POZITIF KONTROLLERI. Gorev talimati: meşru
# metin (PMID'ler, yillar, istatistikler) REDAKTE EDILMEMELI. Bu testler,
# yukaridaki genisletmelerin (ayrac tekrari + cumle-sinirli yakinlik)
# MESRU icerigi YOK ETMEDIGINI somut olarak gosterir.
# ---------------------------------------------------------------------------


def test_false_positive_pmid_not_redacted() -> None:
    text = "See PMID: 12345678 for methodological details."
    cleaned, counts = sanitize_free_text(text)
    assert "12345678" in cleaned
    assert sum(counts.values()) == 0


def test_false_positive_p_value_and_sample_size_not_redacted() -> None:
    text = "Median overall survival was 14.6 months (p=0,0026, n=295)."
    cleaned, counts = sanitize_free_text(text)
    assert "p=0,0026" in cleaned
    assert "n=295" in cleaned
    assert sum(counts.values()) == 0


def test_false_positive_stats_in_same_sentence_as_institution_not_redacted() -> None:
    """2026-09-11'de eklenen cumle-sinirli yakinlik kuralinin YAN ETKISI
    OLMADIGINI gosterir: kurum adi VE mesru istatistik AYNI cumlede
    gecse bile, istatistik bare-ID SEKLINDE (rakam-ayrac-rakam) OLMADIGI
    surece redakte edilmez."""

    text = "The TCGA cohort showed p=0,0026 and n=295 in this analysis."
    cleaned, counts = sanitize_free_text(text)
    assert "p=0,0026" in cleaned
    assert "n=295" in cleaned
    # yalniz "TCGA" kurum adi redakte edilir, sayisal icerik DEGIL.
    assert counts.get("institution", 0) == 1
    assert sum(v for k, v in counts.items() if k.startswith("patient_id")) == 0


def test_known_tradeoff_dose_range_near_institution_is_overredacted() -> None:
    """BILINEN ODUNLESIM (yeni bulgu DEGIL -- 2026-08-18 tasarımının
    dogal sonucu, bu gorevde GENISLETILMEDI): bir doz/orneklem ARALIGI
    ("100-200") bare-ID ile AYNI sekle sahip oldugu icin kurum adiyla
    ayni cumlede/yakininda gecerse yine de redakte edilir. Bu testin
    amaci bunu GIZLEMEDEN belgelemek -- "her turlu mesru istatistik
    korunur" iddiasi KURULMAZ, yalnizca hyphen ICERMEYEN istatistikler
    (yukaridaki testler) icin korunma garanti edilir."""

    text = "In the TCGA cohort, the dose ranged 100-200 mg daily."
    cleaned, _counts = sanitize_free_text(text)
    assert "100-200" not in cleaned  # bilinçli kabul edilen asiri-redaksiyon


# ---------------------------------------------------------------------------
# 7) Bağımsız/katı kapı (`assert_patient_context_has_no_digit_clusters`) --
#    reviewer'ın "kapı redaksiyonla AYNI regex'i paylaşmasın" isteği.
# ---------------------------------------------------------------------------


def test_digit_cluster_gate_rejects_novel_unforeseen_id_shape() -> None:
    """Bu kapı ENUMERE edilmiş bir format listesine bakmadığı için hiç
    görülmemiş/gelecekte ortaya çıkacak bir ID şeklini de (örn. hayali
    "NEWSRC-77-8899") yakalayabilir -- `sanitize_free_text`'in desen
    listesine hiç eklenmemiş olsa BİLE."""

    with pytest.raises(SanitizationViolationError):
        assert_patient_context_has_no_digit_clusters("NEWSRC-77-8899 hasta bilgisi")


def test_digit_cluster_gate_passes_on_real_rendered_patient_context() -> None:
    """Gerçek `render_patient_context_text()` çıktısı ASLA bu kapıyı
    tetiklememeli (yanlış pozitif regresyonu)."""

    attrs = PatientAttributeProfile(
        age_years=62,
        gender="Male",
        kps_score=90,
        mgmt_status_bucket="methylated",
        idh1_status_bucket="wildtype",
        risk_category="orta",
    )
    text = render_patient_context_text(attrs)
    assert_patient_context_has_no_digit_clusters(text)  # exception atmamali


def test_digit_cluster_gate_not_applied_to_literature_free_text_to_avoid_false_positives() -> None:
    """Tasarım kararı: geniş rakam-öbeği kapısı literatür metnine
    UYGULANMAZ -- aksi halde meşru bilimsel içerik (p-değeri, doz aralığı)
    yanlış pozitif üretirdi. Bu test o riski SOMUT göstermek için var
    (üretim kodunda bu kapı literatür metnine hiç çağrılmıyor zaten,
    bkz. `build_sanitized_context`)."""

    legitimate_scientific_text = "Median overall survival was 14.6-18.2 months (p<0.0001, n=1200-1500)."
    with pytest.raises(SanitizationViolationError):
        # KASITLI OLARAK burada cagirilirsa yanlis pozitif verecegini kanitlar --
        # bu yuzden build_sanitized_context bu metne bu kapiyi UYGULAMAZ.
        assert_patient_context_has_no_digit_clusters(legitimate_scientific_text)


# ===========================================================================
# 2026-09-13 (G2) — IKINCI NESIL BYPASS'LAR.
#
# BAGLAM: 2026-09-11 (G1) dort residuel bypass'i kapatmisti, AMA bunu
# AYRAC LISTESINI GENISLETEREK yapmisti (`_SEP_CLASS`'a 8 Unicode tire
# eklemek, tekrar sinirini {0,1}'den {0,3}'e cikarmak). 2026-09-13
# olcumu, ayni KOK NEDEN SINIFININ bir adim otesinde hala acik oldugunu
# canli gosterdi -- yani "listeye karakter ekleme" yaklasimi esigi
# oteliyordu, sorunu cozmuyordu:
#
#   dort-ayrac    'TCGA----06----5412'  -> {0,3} siniri asildi, SIZDI
#   nokta-ayrac   'TCGA.06.5412'        -> '.' listede yok,      SIZDI
#   nbsp          'TCGA\xa006\xa05412'  -> U+00A0 listede yok,   SIZDI
#   soft-hyphen   'TCGA\xad06\xad5412'  -> U+00AD listede yok,   SIZDI
#   sifir-genislik'TCGA​06​5412' -> U+200B listede yok, SIZDI
#   tam-genislik-harf 'TCGA-06-5412' (U+FF34..) -> harfler tam-genislik,
#                     KURUM TOKEN'I BILE eslesmiyordu,           SIZDI
#
# YAPISAL DUZELTME (regex yamasi DEGIL): girdi artik once
# `normalize_for_detection()` ile NFKC + Unicode KATEGORI katlamasindan
# (Cf dusur, Pd/Sm tire -> '-', Zs bosluk -> ' ') geciyor ve esleşmeler
# indeks haritasiyla ORIJINAL koordinatlara geri yaziliyor. Artik
# "hangi karakter listeye eklendi" sorusu YOK.
# ===========================================================================

_SECOND_GEN_BYPASSES = [
    pytest.param("TCGA----06----5412 underwent resection", id="dort-ayrac"),
    pytest.param("TCGA.06.5412 underwent resection", id="nokta-ayrac"),
    pytest.param("TCGA 06 5412 underwent resection", id="nbsp-ayrac"),
    pytest.param("TCGA­06­5412 underwent resection", id="soft-hyphen"),
    pytest.param("TCGA​06​5412 underwent resection", id="sifir-genislik-ayrac"),
    pytest.param("TCGA-06-54​12 underwent resection", id="sifir-genislik-rakam-ici"),
    pytest.param("ＴＣＧＡ-06-5412 underwent resection", id="tam-genislik-harf"),
    pytest.param(
        "TCGA-０６-５４１２ underwent resection",
        id="tam-genislik-rakam",
    ),
    pytest.param("UPENN--GBM--00566 underwent resection", id="upenn-cift-ayrac"),
    pytest.param("UCSF PDGM 190 underwent resection", id="ucsf-nbsp"),
]


@pytest.mark.parametrize("text", _SECOND_GEN_BYPASSES)
def test_second_gen_bypass_redaction(text: str) -> None:
    """Redaksiyon katmani: ID'nin rakam govdesi metinde KALMAMALI.

    ⚠️ TEST TASARIM NOTU (2026-09-13'te yakalanan olcum hatasi): bu
    metinlerin soneki bilinerek ' underwent resection' -- ONCEKI olcumde
    sonek ' patient' idi ve '5412 patient' dizisi TERS-SIRA LUMIERE-
    Patient desenine KAZAYLA eslesip '5412'yi rediyordu. Yani koruma
    desenden degil TEST METNININ KELIMESINDEN geliyordu ve bypass
    'kapali' gorunuyordu. Notr sonek bu artefakti ortadan kaldirir."""

    cleaned, counts = sanitize_free_text(text)
    assert "5412" not in cleaned
    assert "00566" not in cleaned
    assert "５４１２" not in cleaned  # tam-genislik rakamlar
    assert sum(counts.values()) >= 1


@pytest.mark.parametrize("text", _SECOND_GEN_BYPASSES)
def test_second_gen_bypass_gate_rejects_raw_form(text: str) -> None:
    """Kapi katmani: redaksiyon ATLANSA/basarisiz olsa bile ham form
    yakalanmali (iki katmanin AYNI ANDA korlesmemesi kurali)."""

    with pytest.raises(SanitizationViolationError):
        assert_no_identifiers_leaked(text)


@pytest.mark.parametrize("text", _SECOND_GEN_BYPASSES)
def test_second_gen_bypass_gate_passes_after_redaction(text: str) -> None:
    """Redaksiyon tam oldugu icin temizlenmis metin kapiyi tetiklememeli
    (aksi halde her ozet fail-closed engellenirdi)."""

    cleaned, _counts = sanitize_free_text(text)
    assert_no_identifiers_leaked(cleaned)


# --- BAGIMSIZ ISKELET KAPISI (2026-09-13) ----------------------------------


def test_skeleton_patterns_are_not_inverted_by_uppercasing() -> None:
    """🔴 REGRESYON NOBETI -- 2026-09-13'te CANLI YAKALANAN HATA.

    Iskelet desenleri token listesinden uretilirken `"".join(tokens)
    .upper()` kullanilmisti; bu, regex PARCALARINI da buyutup `\\d{2}`
    -> `\\D{2}` yapiyordu (RAKAM -> RAKAM OLMAYAN). Sonuc: `TCGA\\D{6}`
    HEM gercek ID'yi ('TCGA065412') KACIRIYOR HEM meşru metni ('TCGA
    cohort study') eslestirip fail-closed kapiyi bosuna tetikliyordu --
    yani sozde 'bagimsiz kapi' sessizce ISE YARAMAZ hale gelmisti.
    Bu, 2026-08-18'in 'tek zayiflik iki savunmayi birden dusurdu'
    hatasinin birebir tekrariydi ve ancak OLCUMLE fark edildi.

    Test hem YONU (gercek ID eslesir) hem TERSINI (mesru metin
    eslesmez) dogrular -- yalniz biri yeterli DEGILDIR."""

    from pipeline.rag_sanitize import _SKELETON_PATTERNS, identifier_skeleton

    # (a) hicbir desen `\D` (rakam-olmayan) icermemeli
    for name, pattern in _SKELETON_PATTERNS:
        assert "\\D" not in pattern.pattern, (
            f"{name} deseni ters cevrilmis: {pattern.pattern}"
        )

    # (b) gercek ID iskeleti eslesmeli
    assert any(p.search(identifier_skeleton("TCGA-06-5412")) for _n, p in _SKELETON_PATTERNS)
    # (c) mesru metin eslesmemeli
    assert not any(
        p.search(identifier_skeleton("the TCGA cohort study")) for _n, p in _SKELETON_PATTERNS
    )


@pytest.mark.parametrize(
    "text",
    [
        "TCGA.06.5412",
        "TCGA--06--5412",
        "TCGA​06​5412",
        "5412-06-TCGA",
        "UPENN GBM 00566",
        "UCSF..PDGM..190",
    ],
)
def test_skeleton_gate_catches_any_separator_shape(text: str) -> None:
    """Iskelet kapisi ayraclarin TAMAMINI siyirdigi icin ayrac sinifindaki
    / tekrar sinirindaki BIR HATA bu kapiyi kor edemez -- enumere
    katmandan BAGIMSIZ tespit mekanizmasi budur."""

    from pipeline.rag_sanitize import _SKELETON_PATTERNS, identifier_skeleton

    assert any(p.search(identifier_skeleton(text)) for _n, p in _SKELETON_PATTERNS)


@pytest.mark.parametrize(
    "legit",
    [
        "the TCGA cohort study",
        "TCGA 2008, 206 patients were profiled",
        "recurrent GBM 2019 cohort",
        "in each patient 12 months of follow-up",
        "of 416 consecutive patients",
        "573 patients were randomly assigned",
    ],
)
def test_skeleton_gate_does_not_block_legitimate_text(legit: str) -> None:
    """Iskelet kapisi FAIL-CLOSED'dir: yanlis pozitif = ozetin TAMAMEN
    engellenmesi. Bu yuzden mesru literatur dili uzerinde AYRICA test
    edilir -- BU TEST YALNIZ ISKELET KATMANINI olcer (enumere katmanin
    ayri, ONCEDEN VAR OLAN davranisi asagida belgelendi).

    'patient 12' / 'GBM 2019' gibi diziler, `LUMIERE-Patient` ve
    `generic-source-numeric` spec'leri iskelet kapisina DAHIL EDILSEYDI
    eslesirdi (2026-09-13'te olculdu: 41 metinlik korpusta 6 yanlis
    pozitif + kurgusal karsit-ornekler: 'recurrent GBM 2019 cohort',
    'multifocal GBM 450 lesions'). Bu yuzden o iki spec iskelet kapisi
    DISINDA birakildi -- ayrac-capali enumere katmanda KAPSANMAYA
    DEVAM EDIYORLAR."""

    from pipeline.rag_sanitize import _SKELETON_PATTERNS, identifier_skeleton

    skeleton = identifier_skeleton(legit)
    hits = [name for name, p in _SKELETON_PATTERNS if p.search(skeleton)]
    assert hits == [], f"iskelet kapisi mesru metni engelledi: {hits}"


@pytest.mark.parametrize(
    "legit,matched_fragment",
    [
        ("in each patient 12 months of follow-up", "patient 12"),
        ("recurrent GBM 2019 cohort", "recurrent GBM 2019"),
    ],
)
def test_known_preexisting_enumerated_layer_false_positive(
    legit: str, matched_fragment: str
) -> None:
    """⚠️ ONCEDEN VAR OLAN YANLIS POZITIF -- G2'de OLCULDU, GIZLENMIYOR.

    Enumere katmanin iki genis spec'i mesru literatur dilinde eslesiyor:
      `LUMIERE-Patient`      -> 'patient 12' ('patient 12 months')
      `generic-source-numeric`-> 'recurrent GBM 2019'
    Bu, G2'nin GETIRDIGI bir gerileme DEGILDIR: 2026-09-11 (G1) donemi
    desenleri yeniden kurulup ayni cumlelerde test edildi, IKISI DE
    ESLESIYORDU (2026-09-13 olcumu).

    ⚠️ URETIM YOLUNDA ETKISI OLCULDU = 0: `build_sanitized_context`
    kapiyi HAM metne degil REDAKSIYONDAN GECMIS metne uygular; redaksiyon
    bu parcalari zaten kaldirdigi icin kapi tetiklenmiyor. 41 metinlik
    korpusta uretim yolunda engellenen metin sayisi 0/41 olculdu.
    Bu test davranisi DONDURUR ki ileride biri kapiyi ham metne
    uygularsa (veya redaksiyon sirasi degisirse) bunun bir maliyeti
    oldugu gorunur olsun."""

    with pytest.raises(SanitizationViolationError):
        assert_no_identifiers_leaked(legit)

    # Uretim yolu (once redaksiyon, sonra kapi) ENGELLENMEMELI:
    cleaned, _counts = sanitize_free_text(legit)
    assert_no_identifiers_leaked(cleaned)


# --- NORMALIZASYON CIKTIYI BOZMAMALI ---------------------------------------


def test_redaction_output_is_not_normalized_legit_unicode_preserved() -> None:
    """Normalizasyon YALNIZ tespit icindir; kullaniciya donen metin
    ORIJINAL karakterleri korur (indeks haritasi tasariminin amaci budur).

    Aksi halde sanitizasyon, mesru Unicode icerigi (Turkce karakterler,
    '≥', tam-genislik noktalama) sessizce ASCII'ye dusurup ozeti
    bozardi."""

    text = "Sağkalım ≥ 14,6 ay; genişletilmiş analiz — TCGA-06-5412 haric."
    cleaned, _counts = sanitize_free_text(text)

    assert "5412" not in cleaned          # ID gitti
    assert "Sağkalım" in cleaned          # Turkce karakter korundu
    assert "≥" in cleaned                 # matematik sembolu korundu
    assert "—" in cleaned                 # em-dash korundu (ayrac degil, icerik)


def test_normalization_index_map_is_consistent() -> None:
    """`normalize_for_detection` haritasi, normalize metnin her karakteri
    icin ORIJINAL metinde gecerli bir indeks vermeli (span geri-yazma
    dogrulugunun on kosulu)."""

    from pipeline.rag_sanitize import normalize_for_detection

    raw = "TCGA​－06­－5412 ＡＢ ≥"
    norm, index_map = normalize_for_detection(raw)
    assert len(norm) == len(index_map)
    assert all(0 <= i < len(raw) for i in index_map)
    # harita monoton olmali (span geri-yazma bunu varsayar)
    assert index_map == sorted(index_map)


# --- YANLIS POZITIF: GERCEKCI ABSTRACT DILI (2026-09-13 olcumu) -------------


@pytest.mark.parametrize(
    "sentence,must_survive",
    [
        ("Patients received temozolomide 75 mg per square meter daily.", "75 mg"),
        ("Median survival was 14.6 months versus 12.1 months.", "14.6"),
        ("The hazard ratio was 0.63 (95% CI 0.52-0.75, p<0.001).", "0.52-0.75"),
        ("MGMT promoter was methylated in 92 of 206 assessable cases.", "92 of 206"),
        ("IDH1 mutations at codon 132 were found in 445 tumors.", "codon 132"),
        ("EGFR amplification was detected in 178 of 390 tumors.", "178 of 390"),
        ("Dice coefficients were 0.92, 0.87 and 0.84 respectively.", "0.92"),
        ("A 25% increase in perpendicular diameters of at least 10 mm.", "10 mm"),
        ("Adjuvant temozolomide 150 to 200 mg for 5 days per 28-day cycle.", "150 to 200 mg"),
        ("Inference required 45-90 seconds per case on a 24 GB GPU.", "45-90 seconds"),
    ],
)
def test_false_positive_legitimate_clinical_content_survives(
    sentence: str, must_survive: str
) -> None:
    """Asiri redaksiyon da bir HATADIR -- ozet anlamsizlasir. Doz, olcum,
    gen adi, istatistik ve %95 GA araliklari KURUM ADI GECMEYEN mesru
    cumlelerde bozulmadan kalmali.

    Olcum (2026-09-13): 12 gercekci GBM abstract'i (6.002 karakter)
    uzerinde redakte edilen oran %0,933 ve TAMAMI kasitli hedeflerdi
    (yil/tarih + 'The Cancer Genome Atlas') -- hicbir doz/istatistik/
    gen adi redakte edilmedi. G2 degisikligi bu orani DEGISTIRMEDI
    (once/sonra birebir ayni)."""

    cleaned, counts = sanitize_free_text(sentence)
    assert must_survive in cleaned
    assert sum(v for k, v in counts.items() if k.startswith("patient_id")) == 0
