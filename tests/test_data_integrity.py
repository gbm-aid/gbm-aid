"""tools/data_integrity_check.py icin pytest sarmalayicisi.

TAMAMEN SALT-OKUNUR -- bu test dosyasi hicbir INSERT/UPDATE/ALTER/DELETE
calistirmaz, tools/data_integrity_check.py'nin check_* fonksiyonlarini canli
DB'ye karsi (readonly baglanti) cagirir.

Not: Bu testler CANLI DB'YE BAGIMLIDIR (fixture olarak sabitlenmis bir CSV/JSON
degil) -- kasitli: gorevin amaci "bugun DB'ye ne yazildi, hala tutarli mi"
sorusunu her calistirmada YENIDEN sormak. DATABASE bağlantısı yoksa (.env
okunamiyor veya baglanti kurulamiyorsa) testler ACIKCA FAIL eder (sessizce
skip edilmez) -- bu bir bilincli tercih: veri butunlugu testinin DB'siz
"gectigini" iddia etmek yaniltici olur.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import data_integrity_check as dic  # noqa: E402


@pytest.fixture(scope="module")
def conn():
    connection = dic.connect_readonly()
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def all_findings(conn):
    return dic.run_all_checks(conn)


def _critical(findings, category: str | None = None):
    return [
        f for f in findings
        if f.severity == "CRITICAL" and (category is None or f.category == category)
    ]


# ---------------------------------------------------------------------------
# 1) Kilitli sayilar
# ---------------------------------------------------------------------------

def test_locked_table_count(conn):
    findings = dic.check_locked_numbers(conn)
    assert not _critical(findings), _critical(findings)


def test_locked_patients_total_and_by_source(conn):
    findings = dic.check_locked_numbers(conn)
    msgs = [f.message for f in findings if f.category == "kilitli-sayi"]
    assert any("patients toplam=1311 -- OK" in m for m in msgs), msgs


def test_locked_cox_training_pool_611_585(conn):
    findings = dic.check_locked_numbers(conn)
    msgs = [f.message for f in findings if f.category == "kilitli-sayi"]
    assert any("Cox egitim havuzu=611/585 -- OK" in m for m in msgs), (
        "Kilitli deger degisti mi? Eski/yanlis rakamlar (721 havuz, 603 olay) "
        "CLAUDE.md'ye gore KULLANILMAZ. " + str(msgs)
    )


def test_locked_ucsf_external_test_295_169(conn):
    findings = dic.check_locked_numbers(conn)
    msgs = [f.message for f in findings if f.category == "kilitli-sayi"]
    assert any("UCSF harici test=295/169 -- OK" in m for m in msgs), msgs


def test_locked_radiomics_c32_counts(conn):
    findings = dic.check_locked_numbers(conn)
    assert not _critical(findings, "kilitli-sayi"), _critical(findings, "kilitli-sayi")


# ---------------------------------------------------------------------------
# 2) Referans butunlugu
# ---------------------------------------------------------------------------

def test_no_orphan_records(conn):
    findings = dic.check_referential_integrity(conn)
    crit = _critical(findings)
    assert not crit, f"Yetim kayit bulundu: {crit}"


# ---------------------------------------------------------------------------
# 3) Benzersizlik ihlalleri
# ---------------------------------------------------------------------------

def test_no_uniqueness_violations(conn):
    findings = dic.check_uniqueness(conn)
    crit = _critical(findings)
    assert not crit, f"Benzersizlik ihlali bulundu: {crit}"


# ---------------------------------------------------------------------------
# 4) NULL/eksik tarama -- bilinen eksiklikler GECER, yeni eksiklikler KIRMIZI yanar
# ---------------------------------------------------------------------------

def test_known_null_gaps_within_expected_range(conn):
    findings = dic.check_null_completeness(conn)
    crit = _critical(findings)
    assert not crit, (
        "Bilinen eksikliklerden biri BEKLENEN aralik disina cikti -- bu ya bir "
        "iyilesme (kayit tamamlandi) ya da yeni bir regresyon olabilir, "
        "KNOWN_NULL_GAPS registry'sini gozden gecir: " + str(crit)
    )


def test_combat_parameters_still_empty_or_flagged(conn):
    """combat_parameters 0 satir olmali (ComBat ozdeslik donusumu, hic fit
    edilmedi -- decisions/2026-08-13-combat-ozdeslik-bulgusu-ve-kapsami.md).
    Eger artik 0'dan farkliysa bu bir mimari degisiklik sinyalidir (LUMIERE
    egitime girdi mi? reference_batch degisti mi?) -- testin kendisi
    FAIL ETMEZ (bilgilendirme), ama WARN uretmesini dogrular."""
    findings = dic.check_null_completeness(conn)
    combat_findings = [f for f in findings if "combat_parameters" in f.message]
    assert combat_findings, "combat_parameters kontrolu hic calismamis."


def test_unexpected_null_scan_is_informational_only(conn):
    """scan_unexpected_nulls SADECE WARN/INFO uretmeli, asla exit code'u
    etkileyecek CRITICAL uretmemeli (bu fonksiyon kesif amacli, gating degil)."""
    findings = dic.scan_unexpected_nulls(conn, {gap.label for gap in dic.KNOWN_NULL_GAPS})
    assert all(f.severity != "CRITICAL" for f in findings), findings


# ---------------------------------------------------------------------------
# 5) Deger alani (domain) kontrolleri
# ---------------------------------------------------------------------------

def test_vital_status_domain(conn):
    rows = dic._q(conn, "select distinct vital_status from patients where vital_status is not null")
    values = {r[0] for r in rows}
    assert values <= {"DECEASED", "ALIVE", "CENSORED"}, values


def test_gender_domain(conn):
    rows = dic._q(conn, "select distinct gender from patients where gender is not null")
    values = {r[0] for r in rows}
    assert values <= {"Male", "Female"}, values


def test_gtr_over90percent_domain(conn):
    rows = dic._q(conn, "select distinct gtr_over90percent from patients where gtr_over90percent is not null")
    values = {r[0] for r in rows}
    assert values <= {"Y", "N"}, values


def test_harmonization_status_domain(conn):
    # 2026-09-13 GUNCELLEMESI (db-agent-E): eski hard-coded literal set
    # {"raw", "n4", "zscore", "combat"} T4 migrasyonundan (db-agent-D,
    # ayni gun) ONCEKI bir isimlendirmeydi ve hicbir zaman canli
    # degerlerle (raw/zscore_pooled_legacy/zscore_t1ce_c32) eslesmedi.
    # T4, `tools/data_integrity_check.py::DOMAIN_ALLOWED["mr_scans.
    # harmonization_status"]`'i ZATEN 6 degerli (4 eski + 2 yeni,
    # CHECK constraint'le birebir) dogru sekilde guncellemisti -- bu
    # test SADECE o TEK otoriter kaynagi kullanacak sekilde duzeltildi
    # (kendi ayri/bayat kopyasini tutmak yerine), boylece ikisi bir
    # daha birbirinden SESSIZCE ayrisamaz.
    rows = dic._q(conn, "select distinct harmonization_status from mr_scans")
    values = {r[0] for r in rows}
    assert values <= dic.DOMAIN_ALLOWED["mr_scans.harmonization_status"], values


def test_no_negative_or_absurd_survival_days(conn):
    rows = dic._q(conn, "select patient_id, survival_days from patients where survival_days < 0 or survival_days > 10000")
    assert not rows, rows


def test_no_out_of_range_age(conn):
    rows = dic._q(conn, "select patient_id, age from patients where age < 0 or age > 120")
    assert not rows, rows


def test_idh1_and_mgmt_domain_normalization_findings_surface(conn):
    """Bilinen bulgu (2026-08-18 taramasi): LUMIERE idh1_status/mgmt_status
    kanonik olmayan degerler kullaniyor ('wt'/'WT', 'methylated'/'not
    methylated' -- UPenn/UCSF'in Title-case sozlugunden farkli). Bu test
    CRITICAL uretmez (henuz cozulmedi, Cox/XGBoost kod tarafi ayri
    dogrulanmali) ama WARN uretilmesini ZORUNLU kilar -- sessizce
    kaybolmasin diye.

    2026-09-13 GUNCELLEME (Baris onayi, LUMIERE normalizasyon gorevi --
    Secenek B: okuma-zamani normalizasyon, DB'ye DOKUNULMADI): bu WARN'lar
    HALA cikiyor (kaynak veri BILINCLI olarak HAM birakildi, provenance
    korunuyor) -- bu ARTIK gurultu DEGIL, bkz. asagidaki YENI test ('bilinen/
    normalize-edilen' mesaji ayrica dogrulaniyor). Kaynak: decisions/
    2026-09-13-lumiere-idh-mgmt-etiket-normalizasyonu.md."""
    findings = dic.check_domain_values(conn)
    warn_msgs = [f.message for f in findings if f.severity == "WARN" and f.category == "domain-normalizasyon"]
    assert warn_msgs, "idh1/mgmt normalizasyon WARN'lari beklenirken bulunamadi -- kaynak verisi normalize mi edildi?"


def test_idh1_and_mgmt_known_normalized_values_are_contextualized(conn):
    """2026-09-13 YENI (Baris onayi): canli DB'deki BILINEN kanonik-olmayan
    LUMIERE degerleri ('WT'/'wt'/'R132H mut'/'IDH1 neg, Sequencing required',
    'methylated'/'not methylated') artik `api/predict.py`'nin okuma-zamani
    normalizasyonundan GECIYOR -- WARN mesaji bunu ACIKCA belirtmeli
    ('okuma-zamani normalizasyonu' / 'servis 422 ATMAZ'), sessizce eski
    'dogrulanmali' diliyle KALMAMALI (aksi halde denetim BAYATLAR)."""
    findings = dic.check_domain_values(conn)
    warn_msgs = [f.message for f in findings if f.severity == "WARN" and f.category == "domain-normalizasyon"]
    contextualized = [m for m in warn_msgs if "okuma-zamani normalizasyonu" in m]
    assert contextualized, (
        "En az bir WARN mesaji 'bilinen/normalize-edilen' baglamini icermeli -- "
        f"bulunan mesajlar: {warn_msgs}"
    )


def test_idh1_domain_normalization_flags_genuinely_unknown_value_distinctly(monkeypatch, conn):
    """Saf-mantik regresyonu (DB'ye YAZMAZ): `KNOWN_NORMALIZED_IDH1_RAW_VALUES`
    haritasinin KAPSAMADIGI, tamamen YENI/beklenmedik bir ham deger
    (`'???-yeni-sema-disi-deger'`) `dic._q()` sahte-donus degeriyle
    SIMULE edilir -- bu deger 'bilinen/normalize-edilen' mesaji DEGIL, 'YENI'
    (bilinmeyen) mesaji tetiklemeli. Boylece iki dal (bilinen vs gercekten
    yeni) birbirinden AYRISIK kalir -- ileride biri digerine sessizce
    KARISMAZ."""

    # DOMAIN_ALLOWED donguleri ilk basta calisiyor -- onlari bos/etkisiz
    # birakmak icin ayri bir sahte uygulamiyoruz, sadece idh1/mgmt sorgularini
    # hedefliyoruz; DOMAIN_ALLOWED dongusu gercek `conn`'a gidecek (readonly,
    # zaten zararsiz).
    real_q = dic._q

    def _patched_q(c, sql, *args, **kwargs):
        if "p.idh1_status, count(*)" in sql:
            return [("LUMIERE", "???-yeni-sema-disi-deger", 1)]
        if "p.mgmt_status, count(*)" in sql:
            return []
        return real_q(c, sql, *args, **kwargs)

    monkeypatch.setattr(dic, "_q", _patched_q)
    findings = dic.check_domain_values(conn)
    warn_msgs = [f.message for f in findings if f.severity == "WARN" and f.category == "domain-normalizasyon"]
    unknown_msgs = [m for m in warn_msgs if "YENI deger" in m and "???-yeni-sema-disi-deger" in m]
    assert unknown_msgs, f"Beklenmeyen: simule edilen yeni deger 'YENI' olarak isaretlenmedi -- {warn_msgs}"


# ---------------------------------------------------------------------------
# 6) Kaynaklar arasi tutarlilik
# ---------------------------------------------------------------------------

def test_patient_id_prefix_matches_source(conn):
    findings = dic.check_cross_source_consistency(conn)
    crit = _critical(findings, "kaynak-tutarliligi")
    assert not crit, f"patient_id oneki / kaynak uyumsuzlugu: {crit}"


def test_segmentation_tool_matches_source_no_cross_contamination(conn):
    findings = dic.check_cross_source_consistency(conn)
    crit = [f for f in findings if f.severity == "CRITICAL" and "segmentation_tool" in f.message]
    assert not crit, f"Capraz-kaynak segmentation_tool sizintisi: {crit}"


# ---------------------------------------------------------------------------
# 7) Ek bulgular -- bilinen (henuz cozulmemis) sorunlar, bilerek KIRMIZI
# ---------------------------------------------------------------------------

def test_dataset_sources_n_patients_matches_actual_KNOWN_FAILING(conn):
    """BILINEN SORUN (2026-08-18 tarandi, henuz DUZELTILMEDI):
    dataset_sources.n_patients TCGA-GBM'de 260 (gercek 261), TCGA-Omics'te 48
    (gercek 34, muhtemelen omics_profiles sayisiyla karistirilmis). Bu test
    BILINCLI OLARAK xfail -- duzeltildiginde bu test PASS'e donecek ve
    xfail(strict=True) sayesinde bu da ACIKCA gorulecek (sessiz gecmeyecek)."""
    mismatches = dic._q(conn, """
        select ds.source_id, ds.source_name, ds.n_patients, count(p.patient_id) actual
        from dataset_sources ds left join patients p on p.source_id = ds.source_id
        group by 1, 2, 3
        having ds.n_patients != count(p.patient_id)
    """)
    if mismatches:
        pytest.xfail(f"BILINEN SORUN henuz duzeltilmedi (db-agent bulgusu, 2026-08-18): {mismatches}")
    # mismatches bossa (duzeltilmisse) bu satira ulasilir ve test GERCEKTEN GECER
    assert not mismatches


def test_no_c32_radiomics_on_raw_status_scans_KNOWN_FAILING(conn):
    """BILINEN SORUN (2026-08-18 tarandi, kok nedeni belgelendi, K3 bilinen-
    istisna registry'sine tasindi -- bkz. check_c32_harmonization_status_gap()):
    LUMIERE'de 53 tarama (7 hasta) mr_scans.harmonization_status='raw' iken
    LUMIERE-PyRadiomics-107-C32 (N4+zscore-ON-KOSULLU sozlesme) altinda 265
    radyomik satiri var. Bu test KASITLI OLARAK registry-FARKINDA DEGIL --
    'K3'un onayladigi bir istisna olsa BILE, ham SQL seviyesinde HERHANGI
    bir raw-status+C32 satiri varsa bunu ACIKCA bildir' diye SADE bir
    regresyon sinyali saglar (registry-tabanli check_c32_harmonization_
    status_gap() ile TAMAMLAYICI, onun YERINE GECMEZ).

    2026-09-11 DUZELTME (Codex bulgu #6, K13/M1 dersi): suffix
    `LIKE '%%-C32'` YERINE `LOCKED_RADIOMICS_C32`'nin KAPALI IN(...) listesi
    kullanilir -- `-C32-v2` gibi yeniden adlandirilmis bir varyant LIKE ile
    SESSIZCE bu taramanin disinda kalirdi."""
    rows = dic._q(conn, """
        select r.segmentation_tool, count(*)
        from radiomics r join mr_scans m on r.scan_id = m.scan_id
        where m.harmonization_status = 'raw'
          and r.segmentation_tool = ANY(%s)
        group by 1
    """, (list(dic.LOCKED_RADIOMICS_C32.keys()),))
    if rows:
        pytest.xfail(f"BILINEN SORUN henuz cozulmedi (db-agent bulgusu, 2026-08-18): {rows}")
    assert not rows


# ---------------------------------------------------------------------------
# Genel entegrasyon -- tum kategorilerin en az bir kez calistigini dogrula
# ---------------------------------------------------------------------------

def test_run_all_checks_produces_findings_in_every_category(all_findings):
    categories = {f.category for f in all_findings}
    expected_categories = {
        "kilitli-sayi", "referans-butunlugu", "benzersizlik", "bilinen-eksiklik",
        "domain", "kaynak-tutarliligi", "ek-bulgu",
    }
    missing = expected_categories - categories
    assert not missing, f"Beklenen kategoriler hic bulgu uretmemis: {missing}"


def test_print_report_exit_code_matches_critical_presence(all_findings, capsys):
    exit_code = dic.print_report(all_findings, verbose=False)
    has_critical = any(f.severity == "CRITICAL" for f in all_findings)
    assert exit_code == (1 if has_critical else 0)
