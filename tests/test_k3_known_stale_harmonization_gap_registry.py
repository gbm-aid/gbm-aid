"""K3 `KnownStaleHarmonizationGap` registry'si icin TAMAMEN DB-BAGIMSIZ,
sentetik, kalici pytest paketi.

Codex 2026-08-28 capraz-inceleme bulgusu #2'nin duzeltmesi: logda iddia
edilen "6 sentetik senaryo"nun repoda kalici bir karsiligi yoktu. Bu dosya
o 6 senaryoyu (a-f) DB gerektirmeden, `tools/data_integrity_check.py`'nin
SAF (DB-bagimsiz) fonksiyonlarini (`_evaluate_c32_harmonization_gap`,
`_build_registry_by_tool`, `check_registry_expiry`,
`recompute_expected_row_identity_sha256`) dogrudan cagirarak sabitler:

  (a) registry'deki satir canli veriyle (kimlik+icerik) esler -> GECER (INFO)
  (b) kimlik hash'i uyusmuyor -> CRITICAL
  (c) icerik hash'i uyusmuyor (kimlik ayni) -> CRITICAL
  (d) registry'de KAYITLI OLMAYAN bir aracta bayat satir -> YAKALANIR (CRITICAL)
  (e) mukerrer registry girdisi (ayni segmentation_tool iki kez) -> HATA (ValueError)
  (f) `review_by` gecmis -> UYARI (WARN, varsayilan) / HATA (CRITICAL, --strict-expiry)

Ayrica: `expected_row_identity_sha256`'nin BAGIMSIZ yeniden-uretilebilirligi
(Codex bulgu #5) ve gercek LUMIERE-PyRadiomics-107-C32 registry girdisinin
yerel CSV artifact'iyla (canli DB bu oturumda ERISILEMEDI, bkz.
data_integrity_check.py'deki ilgili yorum) tutarliligi ayri testlerle
kilitlenir.

Bu dosya hicbir DB baglantisi ACMAZ -- `conn` fixture'i YOKTUR, hicbir test
`psycopg2`/canli SELECT kullanmaz.

2026-09-12 EKLENDI (Codex/reviewer bagimsiz dogrulamasinin 4 sartindan
#2 ve #3'un duzeltmesi -- bkz. log/2026-09-12.md, db-agent):
  - (f) senaryosunun TAM sinir davranisini (today==review_by VE
    today==review_by+1) kilitleyen iki YENI test eklendi -- onceki testler
    yalniz "cok eski" (2020-01-01) veya "cok uzak gelecek" (2099-01-01)
    review_by degerleriyle calisiyordu, `effective_today > review_by`
    (KESIN BUYUKTUR, >=  DEGIL) karsilastirmasinin tam sinirini
    KANITLAMIYORDU.
  - Hash normalizasyon testleri (`_normalize_scalar_for_hash`,
    `_normalize_json_dict_for_hash`) icin BAGIMSIZ (elle hesaplanmis,
    ayni fonksiyondan TURETILMEMIS) beklenen-deger sabitleriyle testler
    eklendi -- onceki tek test (`test_...`) beklenen hash'i AYNI
    fonksiyondan uretiyordu (totolojik, hicbir regresyonu yakalayamazdi).
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import data_integrity_check as dic  # noqa: E402

PROJECT_TOOLS_DIR = Path(__file__).resolve().parents[1]
REAL_LUMIERE_CSV = (
    PROJECT_TOOLS_DIR / "artifacts" / "week3" / "pyradiomics" / "lumiere_pyradiomics_c32.csv"
)


# ---------------------------------------------------------------------------
# Sentetik gap fabrikasi -- kucuk, hizli, canli LUMIERE baseline'indan
# TAMAMEN BAGIMSIZ (o baseline'i degistirmeden test edebilmek icin).
# ---------------------------------------------------------------------------

def _make_synthetic_gap_and_matching_rows(*, tool: str = "SYNTH-TOOL-C32", review_by: str = "2099-01-01"):
    """Kendi kendine tutarli (self-consistent) kucuk bir
    `KnownStaleHarmonizationGap` + o gap'in TAM olarak kabul edecegi
    (kimlik+icerik birebir uyumlu) `tool_rows` listesi uretir. Testler bu
    ciftin BASLANGICTA (a) senaryosunu (tam eslesme) saglamasini,
    sonra KASITLI bozulmalarla (b)/(c) senaryolarini KIRMIZI/YESIL
    kanitlamasini kullanir."""
    scan_id_to_patient = {101: "Patient-A", 102: "Patient-A", 201: "Patient-B"}
    regions = frozenset({"WT_derived", "TC_derived"})

    identity_rows = [
        (tool, scan_id, patient_id, region)
        for scan_id, patient_id in scan_id_to_patient.items()
        for region in regions
    ]
    row_identity_hash = dic._hash_row_identities(identity_rows)
    scan_hash = dic._hash_scan_id_set(scan_id_to_patient.keys())
    patient_hash = dic._hash_patient_id_set(scan_id_to_patient.values())

    content_rows = [
        (tool, scan_id, patient_id, region, 100.0, 20.0, 3.0, 4.0, "computed",
         {"shape_a": 1.5}, {"fo_a": 2.5}, {"tex_a": 3.5})
        for (tool_, scan_id, patient_id, region) in identity_rows
    ]
    content_hash = dic._hash_row_content(content_rows)

    gap = dic.KnownStaleHarmonizationGap(
        segmentation_tool=tool,
        expected_n_scans=len(scan_id_to_patient),
        expected_n_rows=len(identity_rows),
        expected_n_patients=len(set(scan_id_to_patient.values())),
        expected_scan_ids=frozenset(scan_id_to_patient.keys()),
        expected_patient_ids=frozenset(scan_id_to_patient.values()),
        expected_row_identity_sha256=row_identity_hash,
        expected_scan_id_set_sha256=scan_hash,
        expected_patient_id_set_sha256=patient_hash,
        expected_scan_id_to_patient_id=dict(scan_id_to_patient),
        expected_tumor_regions=regions,
        expected_content_sha256=content_hash,
        decision_ref="test fixture -- gercek bir K3 karari DEGIL",
        review_by=review_by,
    )
    return gap, content_rows


# ---------------------------------------------------------------------------
# (a) registry satiri canli veriyle (kimlik+icerik) esler -> GECER
# ---------------------------------------------------------------------------

def test_scenario_a_matching_rows_pass_as_info():
    gap, content_rows = _make_synthetic_gap_and_matching_rows()
    findings = dic._evaluate_c32_harmonization_gap(gap.segmentation_tool, content_rows, gap)
    assert findings, "hic bulgu uretilmedi"
    assert all(f.severity == "INFO" for f in findings), findings
    assert any("KIMLIK VE ICERIK bakimindan BIREBIR" in f.message for f in findings), findings


# ---------------------------------------------------------------------------
# (b) kimlik hash'i uyusmuyor -> CRITICAL
# ---------------------------------------------------------------------------

def test_scenario_b_identity_mismatch_is_critical_RED_GREEN():
    gap, content_rows = _make_synthetic_gap_and_matching_rows()

    # YESIL (bozulmamis durum): once dogrula ki hicbir CRITICAL YOK.
    baseline_findings = dic._evaluate_c32_harmonization_gap(gap.segmentation_tool, content_rows, gap)
    assert not any(f.severity == "CRITICAL" for f in baseline_findings), (
        "KIRMIZI/YESIL kanitin YESIL ucu bozuk -- bozulmamis veri zaten CRITICAL uretiyor"
    )

    # KIRMIZI: bir satirin scan_id'sini registry'de OLMAYAN bir degere
    # degistir (kimlik kumesi degisir, icerik yapisi ayni kalir).
    broken_rows = list(content_rows)
    tool, scan_id, patient_id, region, *rest = broken_rows[0]
    broken_rows[0] = (tool, 999999, patient_id, region, *rest)

    broken_findings = dic._evaluate_c32_harmonization_gap(gap.segmentation_tool, broken_rows, gap)
    critical = [f for f in broken_findings if f.severity == "CRITICAL"]
    assert critical, "kimlik hash'i degistigi halde CRITICAL uretilmedi -- KIRMIZI/YESIL testi basarisiz"
    assert any("KIMLIGI UYUSMUYOR" in f.message for f in critical), critical

    # GERI AL (revert) -- orijinal veriyle tekrar YESIL oldugunu dogrula.
    reverted_findings = dic._evaluate_c32_harmonization_gap(gap.segmentation_tool, content_rows, gap)
    assert not any(f.severity == "CRITICAL" for f in reverted_findings), (
        "geri alma sonrasi hala CRITICAL var -- test durumu kirletmis olabilir"
    )


# ---------------------------------------------------------------------------
# (c) icerik hash'i uyusmuyor (kimlik AYNI) -> CRITICAL
# ---------------------------------------------------------------------------

def test_scenario_c_content_mismatch_with_identity_match_is_critical_RED_GREEN():
    gap, content_rows = _make_synthetic_gap_and_matching_rows()

    baseline_findings = dic._evaluate_c32_harmonization_gap(gap.segmentation_tool, content_rows, gap)
    assert not any(f.severity == "CRITICAL" for f in baseline_findings), (
        "KIRMIZI/YESIL kanitin YESIL ucu bozuk"
    )

    # KIRMIZI: KIMLIK alanlarina (ilk 4) DOKUNMADAN sadece bir radyomik
    # DEGERI (tumor_volume_mm3) degistir -- kimlik hash'i AYNI kalmali,
    # icerik hash'i FARKLILASMALI.
    broken_rows = list(content_rows)
    (tool, scan_id, patient_id, region, volume, surface_area, entropy,
     contrast, feature_source, shape, first_order, texture) = broken_rows[0]
    broken_rows[0] = (
        tool, scan_id, patient_id, region, volume + 500.0, surface_area,
        entropy, contrast, feature_source, shape, first_order, texture,
    )

    # On-kosul: kimlik hash'i GERCEKTEN ayni kaldi mi (test kendi kendini dogrulasin).
    identity_before = dic._hash_row_identities([r[:4] for r in content_rows])
    identity_after = dic._hash_row_identities([r[:4] for r in broken_rows])
    assert identity_before == identity_after, "test kurulumu hatali -- kimlik de degismis"

    broken_findings = dic._evaluate_c32_harmonization_gap(gap.segmentation_tool, broken_rows, gap)
    critical = [f for f in broken_findings if f.severity == "CRITICAL"]
    assert critical, "icerik degistigi halde CRITICAL uretilmedi"
    assert any("ICERIGI UYUSMUYOR" in f.message for f in critical), critical
    # Bu, kimlik-mismatch mesajindan AYRI/farkli bir mesaj olmali (iki
    # farkli drift sinifi -- Codex bulgu #1'in ozeti).
    assert not any("KIMLIGI UYUSMUYOR" in f.message for f in critical), critical

    reverted_findings = dic._evaluate_c32_harmonization_gap(gap.segmentation_tool, content_rows, gap)
    assert not any(f.severity == "CRITICAL" for f in reverted_findings), (
        "geri alma sonrasi hala CRITICAL var"
    )


# ---------------------------------------------------------------------------
# (d) registry'de KAYITLI OLMAYAN aracta bayat satir -> YAKALANIR (CRITICAL)
# ---------------------------------------------------------------------------

def test_scenario_d_unregistered_tool_with_raw_rows_is_critical():
    gap, content_rows = _make_synthetic_gap_and_matching_rows()
    # gap=None -- bu arac icin HICBIR registry kaydi yok.
    findings = dic._evaluate_c32_harmonization_gap(gap.segmentation_tool, content_rows, gap=None)
    critical = [f for f in findings if f.severity == "CRITICAL"]
    assert critical, "kayitsiz aractaki raw-status satiri YAKALANMADI"
    assert any("kayitli bir istisna YOK" in f.message for f in critical), critical


def test_scenario_d_no_rows_no_gap_is_info_not_critical():
    """Bos taraf: satir da yok, kayit da yok -- bu bir SORUN DEGIL."""
    findings = dic._evaluate_c32_harmonization_gap("HICBIR-SATIRI-OLMAYAN-ARAC", [], gap=None)
    assert all(f.severity == "INFO" for f in findings), findings


def test_scenario_d_registry_entry_but_now_zero_live_rows_is_critical_bayatlama():
    """Registry'de kayit VAR ama artik canli DB'de HICBIR satir yok --
    bu 'iyilesme' de olabilir ama SESSIZCE dusurulmemeli (CRITICAL, insan
    onayiyla registry'den cikarilmali)."""
    gap, _ = _make_synthetic_gap_and_matching_rows()
    findings = dic._evaluate_c32_harmonization_gap(gap.segmentation_tool, [], gap=gap)
    critical = [f for f in findings if f.severity == "CRITICAL"]
    assert critical, "registry kaydi bayatladiginda (0 canli satir) CRITICAL uretilmedi"
    assert any("BAYATLADI" in f.message for f in critical), critical


# ---------------------------------------------------------------------------
# (e) mukerrer registry girdisi -> HATA
# ---------------------------------------------------------------------------

def test_scenario_e_duplicate_registry_entry_raises_RED_GREEN():
    gap, _ = _make_synthetic_gap_and_matching_rows()

    # YESIL: tekil liste sorunsuz.
    result = dic._build_registry_by_tool([gap])
    assert result == {gap.segmentation_tool: gap}

    # KIRMIZI: ayni segmentation_tool'u IKI KEZ ekle -- sessiz
    # dict-comprehension yerine ACIK hata beklenir.
    with pytest.raises(ValueError, match="mukerrer segmentation_tool"):
        dic._build_registry_by_tool([gap, gap])

    # Import-zamani self-check'in de (gercek global registry uzerinde)
    # ayni korumayi ZATEN uyguladigini dogrula (regresyon guvencesi).
    dic._build_registry_by_tool()  # gercek KNOWN_STALE_HARMONIZATION_GAPS -- patlamamali


# ---------------------------------------------------------------------------
# (f) review_by gecmis -> UYARI (varsayilan) / HATA (--strict-expiry)
# ---------------------------------------------------------------------------

def test_scenario_f_expired_review_by_is_warn_by_default_RED_GREEN():
    gap, _ = _make_synthetic_gap_and_matching_rows(review_by="2020-01-01")
    today = date(2026, 9, 11)

    findings = dic.check_registry_expiry(gaps=[gap], today=today, strict=False)
    assert findings, "expiry kontrolu hic bulgu uretmedi"
    assert not any(f.severity == "CRITICAL" for f in findings), (
        "varsayilan (strict=False) modda suresi gecmis kayit CRITICAL uretmemeli"
    )
    assert any(f.severity == "WARN" for f in findings), findings
    assert any("GECTI" in f.message for f in findings), findings


def test_scenario_f_expired_review_by_is_critical_with_strict_flag():
    gap, _ = _make_synthetic_gap_and_matching_rows(review_by="2020-01-01")
    today = date(2026, 9, 11)

    findings = dic.check_registry_expiry(gaps=[gap], today=today, strict=True)
    critical = [f for f in findings if f.severity == "CRITICAL"]
    assert critical, "--strict-expiry ile suresi gecmis kayit CRITICAL'e yukselmedi"


def test_scenario_f_not_yet_expired_review_by_is_info():
    gap, _ = _make_synthetic_gap_and_matching_rows(review_by="2099-01-01")
    today = date(2026, 9, 11)

    findings = dic.check_registry_expiry(gaps=[gap], today=today, strict=False)
    assert all(f.severity == "INFO" for f in findings), findings


def test_scenario_f_boundary_today_equals_review_by_is_info_not_warn():
    """SINIR (2026-09-12 EKLENDI): kod `if effective_today > review_by`
    kullaniyor -- yani today == review_by GUNU HENUZ 'gecmedi' sayilir
    (INFO basmali, WARN/CRITICAL DEGIL). Bu, `check_registry_expiry()`'nin
    kendi yorumunda ('bugun 2026-09-19 VEYA onumuzdeki 8 gun icinde
    calistirilirsa hicbir WARN/CRITICAL degismez') ZATEN iddia edilen
    davranisin, o gunun KENDISINDE de gecerli oldugunu kanitlar -- onceki
    testler bu TAM gunu hic sinamiyordu (yalniz cok once/cok sonraki
    review_by degerleriyle calisiyordu)."""
    gap, _ = _make_synthetic_gap_and_matching_rows(review_by="2026-09-19")
    today = date(2026, 9, 19)

    findings = dic.check_registry_expiry(gaps=[gap], today=today, strict=False)
    assert findings, "sinir gunu icin hic bulgu uretilmedi"
    assert all(f.severity == "INFO" for f in findings), (
        f"today == review_by GUNUNDE hala INFO beklenir (kod > kullaniyor, >= degil): {findings}"
    )
    assert not any("GECTI" in f.message for f in findings), findings

    # strict=True modda da AYNI gun icin hala INFO olmali (henuz gecmedi).
    strict_findings = dic.check_registry_expiry(gaps=[gap], today=today, strict=True)
    assert all(f.severity == "INFO" for f in strict_findings), strict_findings


def test_scenario_f_boundary_today_is_review_by_plus_one_day_is_warn_default_and_critical_strict():
    """SINIR (2026-09-12 EKLENDI): review_by'dan TAM BIR GUN sonra (ilk
    'gecmis' gun) varsayilan modda WARN, `--strict-expiry` ile CRITICAL
    basmali -- bu, `days_over == 1` sinirini kilitler (onceki testler
    review_by="2020-01-01" gibi COK eski tarihlerle calisiyordu, ilk gecis
    gununu degil)."""
    gap, _ = _make_synthetic_gap_and_matching_rows(review_by="2026-09-19")
    today = date(2026, 9, 20)

    default_findings = dic.check_registry_expiry(gaps=[gap], today=today, strict=False)
    assert any(f.severity == "WARN" for f in default_findings), default_findings
    assert not any(f.severity == "CRITICAL" for f in default_findings), (
        f"today == review_by+1 gununde varsayilan modda CRITICAL OLMAMALI: {default_findings}"
    )
    assert any("1 gun once" in f.message for f in default_findings), default_findings

    strict_findings = dic.check_registry_expiry(gaps=[gap], today=today, strict=True)
    assert any(f.severity == "CRITICAL" for f in strict_findings), (
        f"today == review_by+1 gununde --strict-expiry ile CRITICAL basmali: {strict_findings}"
    )


def test_real_registry_is_empty_after_t4_closure_2026_09_13():
    """2026-09-13 GUNCELLEMESI (db-agent-E): bu test ONCEDEN
    ("...does_not_flip_red_this_week") GERCEK `KNOWN_STALE_HARMONIZATION_GAPS`
    registry'sinin (LUMIERE-PyRadiomics-107-C32 kaydiyla) `review_by`
    tarihinden (2026-09-19) once hala INFO bastigini dogruluyordu. T4
    migrasyonu bu kaydi KAPATTI (canli DB'de dogrulandi -- bu 53 taramanin
    TUMU artik harmonization_status='zscore_t1ce_c32', raw-status+C32
    kombinasyonu 0 satir): kayit ARTIK GEREKSIZ oldugu icin
    KNOWN_STALE_HARMONIZATION_GAPS'ten CIKARILDI (kendiliginden degil,
    AKTIF-GOREVLER.md/log/2026-09-13.md'de acikca raporlandi), tarihsel
    hash-regresyon sabiti olarak
    `dic._CLOSED_GAP_LUMIERE_PYRADIOMICS_107_C32_2026_09_13` altinda ayrica
    saklaniyor (bkz. test'ler asagida).

    Bu test artik BOS registry'nin `check_registry_expiry()`'yi hicbir
    bulgu URETMEDEN (bos liste) gecirdigini kilitler -- eger ileride BU
    listeye YENI bir gap eklenirse (baska bir arac/kohort icin), bu test
    KIRILIR ve bakim yapan kisiyi bu testi (ve varsa review_by beklentisini)
    GUNCELLEMEYE ZORLAR (sessizce yesile donmez)."""
    today = date(2026, 9, 13)
    assert dic.KNOWN_STALE_HARMONIZATION_GAPS == [], (
        "KNOWN_STALE_HARMONIZATION_GAPS artik BOS olmali (K3, T4 ile kapandi) -- "
        "eger buraya YENI bir kayit eklendiyse bu test ACIKCA guncellenmeli."
    )
    findings = dic.check_registry_expiry(today=today, strict=False)
    assert findings == [], (
        "bos registry icin check_registry_expiry() bos bulgu listesi donmeli: "
        + str(findings)
    )


# ---------------------------------------------------------------------------
# Finding #5 -- expected_row_identity_sha256 BAGIMSIZ yeniden-uretilebilir
# ---------------------------------------------------------------------------

def test_expected_row_identity_sha256_is_independently_reproducible_synthetic():
    gap, _ = _make_synthetic_gap_and_matching_rows()
    recomputed = dic.recompute_expected_row_identity_sha256(gap)
    assert recomputed == gap.expected_row_identity_sha256


def test_expected_row_identity_sha256_is_independently_reproducible_for_real_registry():
    """Gercek/aktif registry (`KNOWN_STALE_HARMONIZATION_GAPS`) icin AYNI
    koruma -- bu, `_assert_known_stale_gaps_self_consistent()`'in import-
    zamaninda zaten yaptigi kontrolun ACIK bir pytest karsiligidir
    (import-zamani hatasi pytest toplama asamasinda gizli kalabilir, bu
    test onu GORUNUR kilar).

    2026-09-13 GUNCELLEMESI: `KNOWN_STALE_HARMONIZATION_GAPS` T4 kapanisi
    sonrasi BOS liste -- bu dongu artik hicbir gap uzerinde calismiyor
    (bu bir HATA DEGIL, kayitli tasarim). Asagidaki AYRI test
    (`..._for_closed_lumiere_gap`) ayni korumayi TARIHSEL sabit uzerinde
    yapar, boylece bu dosyanin coverage'i BOSALMAZ."""
    assert dic.KNOWN_STALE_HARMONIZATION_GAPS == []
    for gap in dic.KNOWN_STALE_HARMONIZATION_GAPS:
        recomputed = dic.recompute_expected_row_identity_sha256(gap)
        assert recomputed == gap.expected_row_identity_sha256, gap.segmentation_tool


def test_expected_row_identity_sha256_is_independently_reproducible_for_closed_lumiere_gap():
    """2026-09-13 EKLENDI (db-agent-E): yukaridaki testin gercek
    KNOWN_STALE_HARMONIZATION_GAPS listesi uzerinde artik hicbir kayit
    kalmadigi icin (K3 kapandi) bos donmesinin, `expected_row_identity_
    sha256`'nin BAGIMSIZ yeniden-uretilebilirligi korumasini SESSIZCE
    ZAYIFLATMAMASI icin -- ayni kontrol, KAPANMIS/tarihsel
    `_CLOSED_GAP_LUMIERE_PYRADIOMICS_107_C32_2026_09_13` sabiti uzerinde
    ACIKCA tekrarlanir (bu sabit zaten import-zamaninda
    `_assert_known_stale_gaps_self_consistent()` tarafindan da dogrulanir,
    burada AYRICA gorunur/pytest-raporlanabilir kilinir)."""
    gap = dic._CLOSED_GAP_LUMIERE_PYRADIOMICS_107_C32_2026_09_13
    recomputed = dic.recompute_expected_row_identity_sha256(gap)
    assert recomputed == gap.expected_row_identity_sha256, gap.segmentation_tool


def test_recompute_row_identity_breaks_loudly_if_registry_hand_edited_RED_GREEN():
    """KIRMIZI/YESIL kanit: registry'nin `expected_scan_id_to_patient_id`
    alani elle bozulursa (bir hastayi degistir), yeniden-uretilen hash
    ARTIK UYUSMAMALI -- yani bu koruma GERCEKTEN bir seyi test ediyor,
    her zaman True donen bir 'tautology' DEGIL."""
    gap, _ = _make_synthetic_gap_and_matching_rows()
    assert dic.recompute_expected_row_identity_sha256(gap) == gap.expected_row_identity_sha256

    tampered_mapping = dict(gap.expected_scan_id_to_patient_id)
    a_scan_id = next(iter(tampered_mapping))
    tampered_mapping[a_scan_id] = "Patient-TAMPERED"
    tampered_gap = dataclasses.replace(
        gap, expected_scan_id_to_patient_id=tampered_mapping
    )
    assert dic.recompute_expected_row_identity_sha256(tampered_gap) != gap.expected_row_identity_sha256


# ---------------------------------------------------------------------------
# Gercek LUMIERE-PyRadiomics-107-C32 registry girdisi -- yerel CSV
# artifact'iyla capraz-dogrulama (CANLI DB bu oturumda ERISILEMEDI --
# bkz. data_integrity_check.py KNOWN_STALE_HARMONIZATION_GAPS[0].
# expected_content_sha256 yorumu). Bu test DB GEREKTIRMEZ, sadece diskteki
# artifact dosyasina bakar -- dosya yoksa (baska bir ortamda calistirilirsa)
# ACIKCA skip eder, sessizce PASS DEMEZ.
# ---------------------------------------------------------------------------

def _load_real_lumiere_c32_rows_for_known_gap(gap) -> list[tuple]:
    known_scan_ids = set(gap.expected_scan_id_to_patient_id.keys())
    with REAL_LUMIERE_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matched = [row for row in rows if int(row["scan_id"]) in known_scan_ids]

    def _sanitize(data: dict) -> dict:
        clean = {}
        for key, value in data.items():
            if isinstance(value, float) and not math.isfinite(value):
                clean[key] = None
            else:
                clean[key] = value
        return clean

    def _parse(raw: str) -> dict:
        return json.loads(raw) if raw else {}

    full_rows = []
    for row in matched:
        scan_id = int(row["scan_id"])
        patient_id = row["patient_id"]
        region = row["region"]
        status = row["status"]
        if status == "LABEL_YOK_0_VOXEL":
            volume, surface_area, entropy, contrast = 0, 0, None, None
            shape, first_order, texture = {}, {}, {}
        else:
            volume = float(row["voxel_volume"]) if row["voxel_volume"] else None
            surface_area = float(row["surface_area"]) if row["surface_area"] else None
            entropy = float(row["entropy"]) if row["entropy"] else None
            contrast = float(row["contrast"]) if row["contrast"] else None
            shape = _sanitize(_parse(row["shape_features_json"]))
            first_order = _sanitize(_parse(row["first_order_features_json"]))
            texture = _sanitize(_parse(row["texture_features_json"]))
        full_rows.append((
            gap.segmentation_tool, scan_id, patient_id, region, volume,
            surface_area, entropy, contrast, "computed", shape, first_order, texture,
        ))
    return full_rows


@pytest.mark.skipif(not REAL_LUMIERE_CSV.exists(), reason="yerel artifact CSV'si bu ortamda yok")
def test_real_lumiere_c32_registry_matches_local_csv_artifact_identity_and_content():
    """CANLI DB YERINE (bu oturumda erisilemedi) yerel artifact CSV'sini
    kullanarak K3 registry girdisinin (LUMIERE-PyRadiomics-107-C32) hem
    KIMLIK hem ICERIK hash'ini dogrular. `lumiere_c32_db_write_report.csv`
    265/265 satirin `db_action=YAZILDI` oldugunu ve repo genelinde
    `radiomics` tablosuna hicbir UPDATE yolunun bulunmadigini (yalniz INSERT)
    kanitladigi icin bu CSV'nin canli DB icerigiyle BIREBIR ayni olmasi
    beklenir.

    2026-09-13 GUNCELLEMESI (db-agent-E): kaynak artik AKTIF
    `dic.KNOWN_STALE_HARMONIZATION_GAPS` DEGIL -- T4 migrasyonu bu kaydi
    KAPATTI (canli DB'de dogrulandi, bkz. registry tanimindaki "2026-09-13
    KAPANIS" yorumu) ve kayit tarihsel `dic._CLOSED_GAP_LUMIERE_
    PYRADIOMICS_107_C32_2026_09_13` sabitine TASINDI. Bu test o tarihsel
    sabiti kullanir -- CSV'nin (donmus, 2026-09-11 tarihli) icerigi hala
    KIMLIK+ICERIK bakimindan bu sabitle BIREBIR eslesmelidir (K3 kapanisi
    RADYOMIK VERIYI degistirmedi, sadece harmonization_status metadata
    bayragini duzeltti). Bu, gercek `python tools/data_integrity_check.py
    --verbose` canli kosusunun YERINE GECMEZ, DB erisilebilir olunca o kosu
    AYRICA yapilmalidir."""
    gap = dic._CLOSED_GAP_LUMIERE_PYRADIOMICS_107_C32_2026_09_13
    full_rows = _load_real_lumiere_c32_rows_for_known_gap(gap)
    assert len(full_rows) == gap.expected_n_rows

    findings = dic._evaluate_c32_harmonization_gap(gap.segmentation_tool, full_rows, gap)
    critical = [f for f in findings if f.severity == "CRITICAL"]
    assert not critical, (
        "yerel CSV artifact'i tarihsel registry sabitinin kimlik/icerik "
        f"baseline'iyla UYUSMUYOR -- {critical}"
    )
    assert any(f.severity == "INFO" for f in findings), findings


# ---------------------------------------------------------------------------
# HASH NORMALIZASYONU -- BAGIMSIZ (elle hesaplanmis) beklenen degerlerle
# (2026-09-12 EKLENDI, Codex/reviewer bagimsiz dogrulamasinin #3 sartinin
# duzeltmesi). Onceki durum: `_normalize_json_dict_for_hash` icin TEK test
# vardi ve beklenen hash'i AYNI fonksiyonu (`_hash_row_content`) tekrar
# cagirarak turetiyordu -- yani fonksiyon KENDI KENDINI dogruluyordu
# (totolojik): fonksiyon degisse de, beklenen deger ONUNLA BIRLIKTE
# degisecegi icin test HICBIR ZAMAN kirmizi yanmazdi. Asagidaki testler
# beklenen degerleri LITERAL (elle yazilmis) string'ler olarak sabitler --
# `repr(float(...))`'un CPython semantigi bagimsiz olarak dogrulanabilir
# (Python REPL'de tek satirda hesaplanip literal olarak buraya yapistirildi).
# ---------------------------------------------------------------------------

def test_normalize_scalar_for_hash_none_is_literal_NULL_string():
    assert dic._normalize_scalar_for_hash(None) == "NULL"


def test_normalize_scalar_for_hash_int_matches_hardcoded_repr():
    # repr(float(5)) Python semantigiyle BAGIMSIZ olarak "5.0" -- fonksiyon
    # tekrar cagrilmadan, elle yazilmis literal.
    assert dic._normalize_scalar_for_hash(5) == "5.0"


def test_normalize_scalar_for_hash_decimal_precision_loss_matches_hardcoded_repr():
    """Decimal precision-loss senaryosu: `Decimal('1.23456789012345678901')`
    (float'in ~17 anlamli hane hassasiyetini asan bir deger) `float()`'a
    cevrildiginde hassasiyet KAYBEDER. Beklenen deger burada `repr(float(...))`
    CAGRILARAK degil, Python'un float yuvarlama semantiginin BAGIMSIZ bilinen
    sonucu olarak elle yazildi (bkz. bu test dosyasinin ust seviye yorumu)."""
    value = Decimal("1.23456789012345678901")
    assert dic._normalize_scalar_for_hash(value) == "1.2345678901234567"


def test_normalize_scalar_for_hash_decimal_and_float_same_value_hash_identically():
    """Decimal(2.5) ve float(2.5) TAM temsil edilebilir bir deger oldugu
    icin (ikili tabanda kesin) ikisi de AYNI kanonik string'i uretmeli --
    NUMERIC (Decimal) / float tip farkinin kendisi YANLIS-POZITIF CRITICAL
    URETMEMELI (bu, `_normalize_scalar_for_hash`'in var olma sebebidir)."""
    assert (
        dic._normalize_scalar_for_hash(Decimal("2.5"))
        == dic._normalize_scalar_for_hash(2.5)
        == "2.5"
    )


def test_normalize_scalar_for_hash_non_numeric_string_passthrough():
    assert dic._normalize_scalar_for_hash("feature_source_computed") == "feature_source_computed"


def test_normalize_json_dict_for_hash_none_and_empty_dict_are_equivalent_by_design():
    """KASITLI ESDEGERLIK (2026-09-12'de ACIKCA belgelendi -- Codex/reviewer
    #3 sarti): `radiomics.shape_features`/`first_order_features`/
    `texture_features` kolonlarinda None (SQL NULL) ve {} (bos JSONB) AYNI
    anlama gelir -- 'bu bolge/tarama icin ozellik hesaplanmadi' (orn.
    LABEL_YOK_0_VOXEL durumu, bkz. bu dosyadaki
    `_load_real_lumiere_c32_rows_for_known_gap`'in `shape, first_order,
    texture = {}, {}, {}` atamasi). Bu yuzden ikisinin AYNI kanonik metne
    ("{}") normalize edilmesi bir HATA DEGIL, BILINCLI bir tasarim
    kararidir -- bu test bu kararı acikca kilitler (aksi halde bir
    regresyon None'i farkli bir sey olarak ele almaya baslarsa fark
    edilmeyebilir)."""
    assert (
        dic._normalize_json_dict_for_hash(None)
        == dic._normalize_json_dict_for_hash({})
        == "{}"
    )


def test_normalize_json_dict_for_hash_key_order_independent_matches_hardcoded_json():
    """Anahtar sirasindan bagimsizlik + int/float/bool normalizasyonu --
    beklenen deger `json.dumps(..., sort_keys=True, separators=(',', ':'))`
    DAVRANISININ BAGIMSIZ bilinen ciktisi olarak elle yazildi (fonksiyon
    tekrar cagrilarak TURETILMEDI)."""
    data = {"b_feature": 2, "a_feature": 1.5, "c_flag": True}
    result = dic._normalize_json_dict_for_hash(data)
    assert result == '{"a_feature":1.5,"b_feature":2.0,"c_flag":true}'


def test_normalize_json_dict_for_hash_non_finite_float_becomes_null():
    """NaN/Inf JSON'da GECERSIZ literal uretebilir -- fonksiyon bunlari
    ACIKCA None'a cevirmeli (kanonik metinde 'null')."""
    data = {"entropy_like": float("nan"), "contrast_like": float("inf")}
    result = dic._normalize_json_dict_for_hash(data)
    assert result == '{"contrast_like":null,"entropy_like":null}'


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Codex/reviewer 2026-09-12 MEDIUM bulgusu #3: `_normalize_json_dict_for_hash` "
        "SADECE ust-seviye (top-level) sayisal degerleri normalize ediyor -- IC ICE "
        "(nested) bir dict icindeki Decimal/NaN degerleri normalize ETMIYOR (kod "
        "'elif isinstance(value, (int, float))' kontrolu nested dict'i YAKALAMIYOR, "
        "'else: normalized[key] = value' dalina duser, oldugu gibi birakilir). "
        "Decimal JSON-serilestirilemez oldugu icin bu genellikle `json.dumps()` "
        "asamasinda TypeError firlatir (bu test bu HATAYI 'beklenen basarisizlik' "
        "olarak isaretler). DUZELTILDIGINDE (nested dict/list icin recursive "
        "normalizasyon eklendiginde) bu test XPASS olur -- strict=True oldugu icin "
        "XPASS bir FAILURE'a donusur, bu da xfail isaretinin KALDIRILMASI "
        "GEREKTIGINI acikca bildirir (sessizce yesile donmez)."
    ),
)
def test_normalize_json_dict_for_hash_nested_decimal_and_nan_not_currently_sanitized_RED():
    nested_with_raw_types = {"outer_feature": {"sub_a": Decimal("1.500"), "sub_b": float("nan")}}
    nested_expected_sanitized = {"outer_feature": {"sub_a": 1.5, "sub_b": None}}

    result_raw = dic._normalize_json_dict_for_hash(nested_with_raw_types)
    result_expected = dic._normalize_json_dict_for_hash(nested_expected_sanitized)
    assert result_raw == result_expected
