"""13 tablolu Supabase semasi icin TAMAMEN SALT-OKUNUR veri butunlugu test paketi.

Gorevlendirme: raw/plan/plan.txt Hafta 6 -- "Veri butunlugu testleri: 13 tabloda
NULL/eksik kayit taramasi, source_id tutarliligi, (patient_id, scan_date,
modality) benzersizlik ihlali kontrolu" (Baris).

KESIN KURAL: Bu script hicbir INSERT/UPDATE/ALTER/DELETE calistirmaz. Baglanti
`default_transaction_read_only=on` ile acilir (savunma katmani -- yanlislikla
bir yazma denenirse DB seviyesinde reddedilir).

Kategoriler (CLAUDE.md gorev tanimiyla birebir):
  1) Kilitli sayilar (regresyon kontrolu)
  2) Referans butunlugu (orphan kontrolu -- FK'lar zaten var ama savunmaci
     dogrulama, ileride bir FK kaldirilirsa bu script hala yakalar)
  3) Benzersizlik ihlalleri (UNIQUE constraint'ler zaten var ama savunmaci
     dogrulama)
  4) NULL/eksik tarama -- bilinen/kabul edilmis eksiklikler KNOWN_NULL_GAPS
     registry'sinde tanimli, bunlarin DISINDA bir NULL bulunursa (yeni/
     beklenmeyen) CRITICAL olarak isaretlenir
  5) Deger alani (domain) kontrolleri
  6) Kaynaklar arasi tutarlilik (patient_id onek <-> source, segmentation_tool
     <-> source)
  7) Ek bulgular -- kod calisirken bagimsiz olarak kesfedilen, kayitli
     bulgularin disinda kalan somut tutarsizliklar. n_patients metadata
     sapmasi TARIHSEL bir ornektir -- 2026-08-19'da BULUNDU VE DUZELTILDI
     (TCGA-GBM 260->261, TCGA-Omics 48->34), 2026-09-12 canli kosusunda
     CRITICAL-0 dogrulandi; kontrol SAVUNMACI olarak (regresyona karsi)
     kalicidir, "acik bulgu" DEGILDIR (bkz. check_additional_findings()
     docstring'i). harmonization_status='raw' + C32 radyomik satiri
     kombinasyonu ise KIMLIK-seviyeli bilinen-istisna registry'sine
     tasindi (bkz. check_c32_harmonization_status_gap()).

Kullanim:
  python tools/data_integrity_check.py            # ozet + detay raporu basar
  python tools/data_integrity_check.py --verbose   # her INFO bulgusunu da basar

Exit code: 0 = CRITICAL bulgu yok, 1 = en az bir CRITICAL bulgu var (CI'a
baglanabilir). WARN/INFO bulgular exit code'u ETKILEMEZ (bilgilendirme
amacli, insan gozden gecirmesi gerekir).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # gbm-aid mert/tools -> proje koku
ENV_PATH = PROJECT_ROOT / ".env"

SEVERITY_ORDER = {"CRITICAL": 0, "WARN": 1, "INFO": 2}

# CI UYUMLULUK NOTU (2026-09-12, Codex/reviewer bagimsiz dogrulamasinin #1
# sartinin duzeltmesi): `ENV_PATH` git repo'sunun (`gbm-aid mert/`) BIR ust
# dizinindedir (`PROJECT_ROOT` = "GBM-AID Prototip/", repo koku degil, bkz.
# .github/workflows/tests.yml basindaki backend-agent notu: "repo koku
# `gbm-aid mert/` klasorudur"). Yani bir GitHub Actions checkout'unda bu
# dosya HICBIR ZAMAN VAR OLAMAZ -- `.env` repoya commit edilmez ve ust
# dizin CI workspace'inde bulunmaz. Bu yuzden `load_env()` ONCE mevcut
# OS ortam degiskenlerine (CI'da GitHub Secrets'tan `env:` ile enjekte
# edilir) bakar, SADECE eksik olan anahtarlar icin (yerel gelistirme
# ortaminda) `.env` dosyasina BASVURUR -- dosya yoksa sessizce atlanir
# (hata firlatmaz), CI'da zaten env degiskenleri set edilmis olacagi icin
# bu BEKLENEN bir daldir.
REQUIRED_ENV_KEYS = (
    "SUPABASE_HOST", "SUPABASE_PORT", "SUPABASE_DB",
    "SUPABASE_USER", "SUPABASE_PASSWORD",
)


@dataclass
class Finding:
    severity: str  # CRITICAL | WARN | INFO
    category: str
    message: str

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_ORDER:
            raise ValueError(f"Gecersiz severity: {self.severity!r}")


# ---------------------------------------------------------------------------
# .env / baglanti
# ---------------------------------------------------------------------------

def load_env() -> dict:
    """Once os.environ (CI'da GitHub Secrets -> `env:` ile enjekte edilir),
    SONRA (varsa) yerel `.env` dosyasi ile EKSIK anahtarlari tamamlar. `.env`
    dosyasi CI'da YOKTUR (repo disinda) -- bu BEKLENEN bir durumdur, hata
    firlatilmaz. Sonunda REQUIRED_ENV_KEYS'ten biri hala eksikse ACIKCA
    KeyError firlatilir (sessizce yaniltici bir baglanti denemesi yerine)."""
    env: dict = dict(os.environ)
    if ENV_PATH.is_file():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip())
    missing = [k for k in REQUIRED_ENV_KEYS if not env.get(k)]
    if missing:
        raise KeyError(
            f"Baglanti bilgisi eksik: {missing}. Yerel gelistirmede proje "
            f"kokundeki .env dosyasi ({ENV_PATH}) kontrol edilmeli; CI'da "
            "GitHub repo Secrets'in workflow adiminda `env:` olarak "
            "enjekte edildigi dogrulanmali (bkz. .github/workflows/"
            "data-integrity.yml)."
        )
    return env


def connect_readonly(env: Optional[dict] = None):
    """SADECE readonly baglanti. Bu fonksiyon hicbir --apply/yazma parametresi
    ALMAZ -- bu script'in tasarim geregi tek yetkisi budur."""
    env = env or load_env()
    conn = psycopg2.connect(
        host=env["SUPABASE_HOST"], port=env["SUPABASE_PORT"], dbname=env["SUPABASE_DB"],
        user=env["SUPABASE_USER"], password=env["SUPABASE_PASSWORD"],
        options="-c default_transaction_read_only=on",
    )
    return conn


def _q(conn, sql: str, params=None) -> list:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# 1) KILITLI SAYILAR (CLAUDE.md + gorev talimati, 2026-08-18 anlik goruntu)
# ---------------------------------------------------------------------------

LOCKED_TOTAL_TABLES = 13

LOCKED_PATIENTS_BY_SOURCE = {
    "UPenn-GBM": 630,
    "LUMIERE": 91,
    "TCGA-GBM": 261,
    "TCGA-Omics": 34,
    "UCSF-PDGM": 295,
}
LOCKED_PATIENTS_TOTAL = 1311

# Cox egitim havuzu: UPenn-GBM, WT_derived (WT-only) C32 radyomigi olan hastalar.
LOCKED_COX_TRAINING_N = 611
LOCKED_COX_TRAINING_EVENTS = 585

# UCSF harici test: UCSF-PDGM, WT_derived C32 radyomigi olan hastalar.
LOCKED_UCSF_EXTERNAL_N = 295
LOCKED_UCSF_EXTERNAL_EVENTS = 169

# radiomics C32 nesli: {segmentation_tool: (satir_sayisi, distinct_scan_id)}
LOCKED_RADIOMICS_C32 = {
    "UPenn-PyRadiomics-107-C32": (3055, 611),
    "LUMIERE-PyRadiomics-107-C32": (2920, 585),
    "UCSF-PDGM-PyRadiomics-107-C32": (1475, 295),
    "TCGA-ground-truth-C32": (78, 39),
}

LOCKED_OMICS_PROFILES = 48
LOCKED_MOLECULAR_SCORES = 48
LOCKED_SCORE_THRESHOLDS = 2
# 2026-09-12 GUNCELLEMESI (db-agent, Gorev B): model_registry 0 -> 3.
# Baris'in nihai Cox model karari (decisions/2026-09-12-nihai-cox-model-
# secimi-v3b.md) sonrasi ilk 3 kayit yazildi: cox_phm/v3b_lowvar_v2amgmt
# (shadow), nnunet/Dataset002_BRATS19-zenodo-11582627 (shadow),
# omics_scorer/v1.0 (production).
# 2026-09-12 IKINCI GUNCELLEME (db-agent-C): model_registry 3 -> 4.
# XGBoost uretim kosusu (v2a_mgmt+reduce_collinearity, aligned, v3b ile
# bit-birebir Cox recete-sadakati kanitli) 16:50'de bitti, 4. kayit
# yazildi: model_id=10, xgboost/v2a_mgmt+reduce_collinearity-aligned-
# v3b_bitidentical_2026-09-12 (shadow -- api/analyze_patient.py XGBoost'u
# canli SERVE ETMIYOR, available:False sabit). Kaynak:
# tools/write_model_registry_xgboost.py.
LOCKED_MODEL_REGISTRY = 4

# ---------------------------------------------------------------------------
# SERVIS EDILEBILIR LUMIERE POPULASYONU (Y1 karari, 2026-09-13, Baris onayi:
# "y1 i onaylıyorum lumiere 72 hastaya çıksın")
#
# ⚠️ BULGU (backend-agent-I, 2026-09-13): Y1 gorev talimati bu dosyada
# "servis edilebilir LUMIERE = 4" diye bir BEKLENTININ VAR OLDUGUNU
# varsayiyordu ve 4 -> 72 guncellenmesini istiyordu. CANLI OLCUM: boyle bir
# beklenti bu dosyada (veya baska bir yerde) HIC YOKTU -- `grep` ile
# dogrulandi, Y1 oncesi kosuda da CRITICAL 0 / WARN 11 idi, yani
# "guncellenmezse yeni bir WARN/CRITICAL dogar" ONGORUSU GERCEKLESMEDI.
# Bu blok bu yuzden bir GUNCELLEME degil, YENI bir guard olarak eklendi:
# K2/Y1'in kilitli sayilari (92/91/73/72) canli DB'den her kosuda
# DOGRULANIR, boylece bir daha "servis edilen populasyon belgelenen
# populasyondan sessizce ayrisma" durumu olusmaz.
#
# 🔑 KURAL BU DOSYAYA KOPYALANMADI: secim mantigi
# `pipeline/lumiere_canonical_visit.py`'den (TEK KAYNAK) IMPORT edilir --
# `api/predict.py` ve `tools/rebuild_faiss_indexes.py` da AYNI modulu
# kullanir. (Bu dosyadaki `KNOWN_NORMALIZED_*` sozlukleri bilincli bir
# "eslesme aynasi"dir ve kendi yorumunda drift riski olarak BELGELENMISTIR
# -- ayni hata BURADA tekrarlanmiyor.)
#
# CI NOTU: `raw/veri/LUMIERE-ExpertRating-*.csv` proje kokunde, yani repo
# kokunun (`gbm-aid mert/`) BIR UST dizinindedir -> GitHub Actions
# checkout'unda BULUNMAZ. Bu yuzden CSV/import erisilemezse bulgu
# **WARN**'dir ("kontrol YAPILAMADI"), ASLA CRITICAL degil -- workflow
# `--strict-expiry` ile kosuyor ve eksik bir girdi dosyasi isi
# kirmiziya cevirmemeli. Sayilar OLCULEBILIP tutmuyorsa CRITICAL.
LOCKED_LUMIERE_PREOP_RATED_ROWS = 92
LOCKED_LUMIERE_PREOP_RATED_PATIENTS = 91
LOCKED_LUMIERE_PREOP_C32_VISITS = 73
LOCKED_LUMIERE_SERVICEABLE_PATIENTS = 72  # (Y1 oncesi FIILI deger: 4)
LOCKED_LUMIERE_PREOP_WITHOUT_C32_PATIENTS = 19

LUMIERE_C32_SEGMENTATION_TOOL = "LUMIERE-PyRadiomics-107-C32"


def check_lumiere_serviceable_population(conn) -> list[Finding]:
    """K2/Y1'in kilitli LUMIERE sayilarini CANLI DB'ye karsi dogrular.

    Olculen: (a) CSV'de `Rating=='Pre-Op'` olan (hasta, vizit) cifti sayisi
    ve ayri hasta sayisi, (b) bu vizitlerden C32 radyomigi DE olanlar
    (= `/predict`'in servis edebilecegi LUMIERE hastalari), (c) Pre-Op
    vizitinde C32'si olmayanlar (acik 422 alan hastalar).

    Secim kurali BU DOSYADA YENIDEN YAZILMAZ -- tek kaynak modulden
    (`pipeline.lumiere_canonical_visit`) import edilir.
    """

    findings: list[Finding] = []

    repo_code_root = Path(__file__).resolve().parents[1]  # "gbm-aid mert"
    if str(repo_code_root) not in sys.path:
        sys.path.insert(0, str(repo_code_root))
    try:
        from pipeline.lumiere_canonical_visit import (  # noqa: PLC0415
            EXPECTED_LUMIERE_PREOP_PATIENTS,
            ExpertRatingFileNotFoundError,
            load_lumiere_preop_visit_keys,
        )
    except Exception as exc:  # ImportError/bagimlilik -- SESSIZ GECILMEZ
        return [Finding("WARN", "lumiere-servis-populasyonu",
            f"`pipeline.lumiere_canonical_visit` import EDILEMEDI ({type(exc).__name__}: "
            "{exc}). K2/Y1'in servis edilebilir LUMIERE sayisi (kilitli 72) "
            "bu kosuda DOGRULANAMADI -- 'OK' ile KARISTIRILMAMALI.".format(exc=exc))]

    if EXPECTED_LUMIERE_PREOP_PATIENTS != LOCKED_LUMIERE_SERVICEABLE_PATIENTS:
        findings.append(Finding("CRITICAL", "lumiere-servis-populasyonu",
            f"`pipeline.lumiere_canonical_visit.EXPECTED_LUMIERE_PREOP_PATIENTS`="
            f"{EXPECTED_LUMIERE_PREOP_PATIENTS} ile bu dosyadaki kilitli deger "
            f"{LOCKED_LUMIERE_SERVICEABLE_PATIENTS} AYRISMIS -- iki taraf elle "
            "senkronize edilmeli (K2/Y1 karari degistiyse karar dosyasi da "
            "guncellenmis olmali)."))

    try:
        preop_pairs, n_rows, n_patients = load_lumiere_preop_visit_keys()
    except ExpertRatingFileNotFoundError as exc:
        return findings + [Finding("WARN", "lumiere-servis-populasyonu",
            f"LUMIERE ExpertRating CSV'si okunamadi ({exc}). K2/Y1'in servis "
            f"edilebilir LUMIERE sayisi (kilitli {LOCKED_LUMIERE_SERVICEABLE_PATIENTS}) "
            "bu kosuda DOGRULANAMADI. CI'da (GitHub Actions) bu BEKLENEN bir "
            "daldir -- `raw/` dizini repo kokunun bir ust dizinindedir, "
            "checkout'a girmez. 'OK' ile KARISTIRILMAMALI.")]

    if (n_rows, n_patients) != (
        LOCKED_LUMIERE_PREOP_RATED_ROWS, LOCKED_LUMIERE_PREOP_RATED_PATIENTS
    ):
        findings.append(Finding("CRITICAL", "lumiere-servis-populasyonu",
            f"LUMIERE ExpertRating CSV'sinde `Rating=='Pre-Op'`: {n_rows} cift / "
            f"{n_patients} hasta; kilitli deger "
            f"{LOCKED_LUMIERE_PREOP_RATED_ROWS}/{LOCKED_LUMIERE_PREOP_RATED_PATIENTS}. "
            "K2/Y1'in dayandigi olcum degismis -- karar dosyasi gozden gecirilmeli."))
    else:
        findings.append(Finding("INFO", "lumiere-servis-populasyonu",
            f"LUMIERE Pre-Op etiketi: {n_rows} cift / {n_patients} hasta -- OK."))

    db_pairs = _q(conn, """
        select ms.patient_id, ms.timepoint_label
        from radiomics r
        join mr_scans ms on ms.scan_id = r.scan_id
        where r.segmentation_tool = %s
        group by 1, 2
    """, (LUMIERE_C32_SEGMENTATION_TOOL,))
    matched = [(pid, label) for pid, label in db_pairs if (pid, label) in preop_pairs]
    n_matched_visits = len(set(matched))
    serviceable = {pid for pid, _label in matched}
    without_c32 = {pid for pid, _label in preop_pairs} - serviceable

    if n_matched_visits != LOCKED_LUMIERE_PREOP_C32_VISITS:
        findings.append(Finding("CRITICAL", "lumiere-servis-populasyonu",
            f"Pre-Op VE C32 radyomigi olan vizit sayisi {n_matched_visits}, "
            f"kilitli deger {LOCKED_LUMIERE_PREOP_C32_VISITS}."))
    else:
        findings.append(Finding("INFO", "lumiere-servis-populasyonu",
            f"Pre-Op + C32 vizit sayisi {n_matched_visits} -- OK."))

    if len(serviceable) != LOCKED_LUMIERE_SERVICEABLE_PATIENTS:
        findings.append(Finding("CRITICAL", "lumiere-servis-populasyonu",
            f"Servis edilebilir (kanonik Pre-Op vizitinde C32'si olan) LUMIERE "
            f"hasta sayisi {len(serviceable)}, kilitli deger "
            f"{LOCKED_LUMIERE_SERVICEABLE_PATIENTS} (Y1 karari, 2026-09-13). "
            "`/predict` bu hastalara risk skoru verir; sayi degistiyse karar "
            "dosyasi + ZORUNLU-BEYANLAR guncellenmeli."))
    else:
        findings.append(Finding("INFO", "lumiere-servis-populasyonu",
            f"Servis edilebilir LUMIERE hasta sayisi {len(serviceable)} -- OK "
            "(Y1 oncesi 4; o 4 hastanin tek C32 taramasi Post-Op'tu ve ARTIK "
            "acik 422 aliyor -- kume degisimi, genisleme degil)."))

    if len(without_c32) != LOCKED_LUMIERE_PREOP_WITHOUT_C32_PATIENTS:
        findings.append(Finding("CRITICAL", "lumiere-servis-populasyonu",
            f"Pre-Op vizitinde C32'si OLMAYAN LUMIERE hasta sayisi "
            f"{len(without_c32)}, kilitli deger "
            f"{LOCKED_LUMIERE_PREOP_WITHOUT_C32_PATIENTS} (bunlar `/predict`'te "
            "ACIK 422 alir, sessizce post-op taramaya DUSULMEZ)."))
    else:
        findings.append(Finding("INFO", "lumiere-servis-populasyonu",
            f"Pre-Op'ta C32'si olmayan (acik 422 alan) LUMIERE hastasi "
            f"{len(without_c32)} -- OK."))

    return findings


def check_locked_numbers(conn) -> list[Finding]:
    findings: list[Finding] = []

    # 2026-09-13 DUZELTME (db-agent-D, T1 uygulamasi sirasinda bulundu):
    # sorgu ONCEDEN table_type filtresi UYGULAMIYORDU -- information_schema.
    # tables VIEW'lari da sayar, bu yuzden T1'in `radiomics_c32` VIEW'i
    # (K13 karari, decisions/2026-09-11-radiomics-c32-view.md -- 2026-09-14'te
    # "-TASLAK" eki kaldirildi, eski ad artik yok, bkz. K23)
    # eklendiginde bu kontrol 14 vs 13 diye YANLIS-POZITIF CRITICAL
    # basiyordu -- LOCKED_TOTAL_TABLES kavramsal olarak hep "BASE TABLE"
    # sayisiydi (13-tablolu sema CLAUDE.md'de), VIEW'lar bu sayimin
    # KAPSAMI DISINDA. `table_type='BASE TABLE'` filtresi eklendi; VIEW
    # sayisi AYRICA (ayri, INFO seviyeli) raporlanir -- sessizce yutulmaz.
    n_tables = _q(conn,
        "select count(*) from information_schema.tables "
        "where table_schema='public' and table_type='BASE TABLE'")[0][0]
    n_views = _q(conn,
        "select count(*) from information_schema.tables "
        "where table_schema='public' and table_type='VIEW'")[0][0]
    if n_tables != LOCKED_TOTAL_TABLES:
        findings.append(Finding("CRITICAL", "kilitli-sayi",
            f"public semada {n_tables} BASE TABLE var, kilitli deger {LOCKED_TOTAL_TABLES}."))
    else:
        findings.append(Finding("INFO", "kilitli-sayi",
            f"BASE TABLE sayisi {n_tables} -- OK. (Ayrica {n_views} VIEW var -- "
            "kilitli sayinin kapsami disinda, ayri bilgi: 2026-09-13'te "
            "eklenen `radiomics_c32` K13 VIEW'ini kapsar.)"))

    total = _q(conn, "select count(*) from patients")[0][0]
    if total != LOCKED_PATIENTS_TOTAL:
        findings.append(Finding("CRITICAL", "kilitli-sayi",
            f"patients toplam={total}, kilitli deger={LOCKED_PATIENTS_TOTAL}."))
    else:
        findings.append(Finding("INFO", "kilitli-sayi", f"patients toplam={total} -- OK."))

    by_source = dict(_q(conn, """
        select ds.source_name, count(*) from patients p
        join dataset_sources ds on p.source_id = ds.source_id
        group by 1
    """))
    for src, expected in LOCKED_PATIENTS_BY_SOURCE.items():
        actual = by_source.get(src)
        if actual != expected:
            findings.append(Finding("CRITICAL", "kilitli-sayi",
                f"patients[{src}]={actual}, kilitli deger={expected}."))
        else:
            findings.append(Finding("INFO", "kilitli-sayi", f"patients[{src}]={actual} -- OK."))
    unexpected_sources = set(by_source) - set(LOCKED_PATIENTS_BY_SOURCE)
    if unexpected_sources:
        findings.append(Finding("WARN", "kilitli-sayi",
            f"KAYITLI OLMAYAN kaynaklar bulundu: {sorted(unexpected_sources)} -- "
            "kilitli listeye eklenmesi gerekebilir (yeni kohort mu?)."))

    cox_n, cox_events = _q(conn, """
        select count(distinct p.patient_id) n_total,
               count(distinct p.patient_id) filter (where p.vital_status = 'DECEASED') n_deceased
        from patients p
        join dataset_sources ds on p.source_id = ds.source_id
        join mr_scans m on m.patient_id = p.patient_id
        join radiomics r on r.scan_id = m.scan_id
            and r.segmentation_tool = 'UPenn-PyRadiomics-107-C32'
            and r.tumor_region = 'WT_derived'
        where ds.source_name = 'UPenn-GBM'
    """)[0]
    if (cox_n, cox_events) != (LOCKED_COX_TRAINING_N, LOCKED_COX_TRAINING_EVENTS):
        findings.append(Finding("CRITICAL", "kilitli-sayi",
            f"Cox egitim havuzu={cox_n}/{cox_events}, kilitli deger="
            f"{LOCKED_COX_TRAINING_N}/{LOCKED_COX_TRAINING_EVENTS}."))
    else:
        findings.append(Finding("INFO", "kilitli-sayi",
            f"Cox egitim havuzu={cox_n}/{cox_events} -- OK."))

    ucsf_n, ucsf_events = _q(conn, """
        select count(distinct p.patient_id) n_total,
               count(distinct p.patient_id) filter (where p.vital_status = 'DECEASED') n_deceased
        from patients p
        join dataset_sources ds on p.source_id = ds.source_id
        join mr_scans m on m.patient_id = p.patient_id
        join radiomics r on r.scan_id = m.scan_id
            and r.segmentation_tool = 'UCSF-PDGM-PyRadiomics-107-C32'
            and r.tumor_region = 'WT_derived'
        where ds.source_name = 'UCSF-PDGM'
    """)[0]
    if (ucsf_n, ucsf_events) != (LOCKED_UCSF_EXTERNAL_N, LOCKED_UCSF_EXTERNAL_EVENTS):
        findings.append(Finding("CRITICAL", "kilitli-sayi",
            f"UCSF harici test={ucsf_n}/{ucsf_events}, kilitli deger="
            f"{LOCKED_UCSF_EXTERNAL_N}/{LOCKED_UCSF_EXTERNAL_EVENTS}."))
    else:
        findings.append(Finding("INFO", "kilitli-sayi",
            f"UCSF harici test={ucsf_n}/{ucsf_events} -- OK."))

    rad_rows = _q(conn, """
        select segmentation_tool, count(*), count(distinct scan_id)
        from radiomics
        where segmentation_tool = ANY(%s)
        group by 1
    """, (list(LOCKED_RADIOMICS_C32.keys()),))
    rad_actual = {r[0]: (r[1], r[2]) for r in rad_rows}
    for tool, expected in LOCKED_RADIOMICS_C32.items():
        actual = rad_actual.get(tool)
        if actual != expected:
            findings.append(Finding("CRITICAL", "kilitli-sayi",
                f"radiomics[{tool}] rows/scans={actual}, kilitli deger={expected}."))
        else:
            findings.append(Finding("INFO", "kilitli-sayi", f"radiomics[{tool}]={actual} -- OK."))

    for label, sql, expected in [
        ("omics_profiles", "select count(*) from omics_profiles", LOCKED_OMICS_PROFILES),
        ("molecular_scores", "select count(*) from molecular_scores", LOCKED_MOLECULAR_SCORES),
        ("score_thresholds", "select count(*) from score_thresholds", LOCKED_SCORE_THRESHOLDS),
        ("model_registry", "select count(*) from model_registry", LOCKED_MODEL_REGISTRY),
    ]:
        actual = _q(conn, sql)[0][0]
        if actual != expected:
            findings.append(Finding("CRITICAL", "kilitli-sayi",
                f"{label}={actual}, kilitli deger={expected}."))
        else:
            findings.append(Finding("INFO", "kilitli-sayi", f"{label}={actual} -- OK."))

    return findings


# ---------------------------------------------------------------------------
# 2) REFERANS BUTUNLUGU (savunmaci -- FK'lar zaten var, DB seviyesinde 0
#    garanti, ama script kendi bagimsiz sorgusuyla da dogrular)
# ---------------------------------------------------------------------------

ORPHAN_CHECKS = [
    ("mr_scans.patient_id -> patients", """
        select count(*) from mr_scans m left join patients p on m.patient_id = p.patient_id
        where p.patient_id is null
    """),
    ("radiomics.scan_id -> mr_scans", """
        select count(*) from radiomics r left join mr_scans m on r.scan_id = m.scan_id
        where m.scan_id is null
    """),
    ("patients.source_id -> dataset_sources", """
        select count(*) from patients p left join dataset_sources ds on p.source_id = ds.source_id
        where ds.source_id is null
    """),
    ("omics_profiles.patient_id -> patients", """
        select count(*) from omics_profiles o left join patients p on o.patient_id = p.patient_id
        where p.patient_id is null
    """),
    ("molecular_scores.patient_id -> patients", """
        select count(*) from molecular_scores ms left join patients p on ms.patient_id = p.patient_id
        where p.patient_id is null
    """),
    ("followup_series.patient_id -> patients", """
        select count(*) from followup_series f left join patients p on f.patient_id = p.patient_id
        where p.patient_id is null
    """),
    ("followup_series.source_id -> dataset_sources", """
        select count(*) from followup_series f left join dataset_sources ds on f.source_id = ds.source_id
        where ds.source_id is null
    """),
    ("followup_series.scan_id -> mr_scans (NULL olabilir, dolu ise gecerli olmali)", """
        select count(*) from followup_series f left join mr_scans m on f.scan_id = m.scan_id
        where f.scan_id is not null and m.scan_id is null
    """),
    ("treatments.patient_id -> patients", """
        select count(*) from treatments t left join patients p on t.patient_id = p.patient_id
        where p.patient_id is null
    """),
    ("tumor_events.patient_id -> patients", """
        select count(*) from tumor_events te left join patients p on te.patient_id = p.patient_id
        where p.patient_id is null
    """),
    ("access_log.patient_id -> patients (NULL olabilir, dolu ise gecerli olmali)", """
        select count(*) from access_log a left join patients p on a.patient_id = p.patient_id
        where a.patient_id is not null and p.patient_id is null
    """),
]


def check_referential_integrity(conn) -> list[Finding]:
    findings: list[Finding] = []
    for label, sql in ORPHAN_CHECKS:
        n = _q(conn, sql)[0][0]
        if n > 0:
            findings.append(Finding("CRITICAL", "referans-butunlugu",
                f"{label}: {n} yetim kayit bulundu."))
        else:
            findings.append(Finding("INFO", "referans-butunlugu", f"{label}: 0 yetim kayit -- OK."))
    return findings


# ---------------------------------------------------------------------------
# 3) BENZERSIZLIK IHLALLERI (savunmaci -- UNIQUE constraint'ler zaten var)
# ---------------------------------------------------------------------------

UNIQUENESS_CHECKS = [
    ("patients.patient_id (PK)", """
        select patient_id, count(*) from patients group by 1 having count(*) > 1
    """),
    ("mr_scans (patient_id, modality, timepoint_label)", """
        select patient_id, modality, timepoint_label, count(*) from mr_scans
        group by 1, 2, 3 having count(*) > 1
    """),
    ("radiomics (scan_id, segmentation_tool, tumor_region)", """
        select scan_id, segmentation_tool, tumor_region, count(*) from radiomics
        group by 1, 2, 3 having count(*) > 1
    """),
    ("model_registry (model_name, version)", """
        select model_name, version, count(*) from model_registry
        group by 1, 2 having count(*) > 1
    """),
    ("dataset_sources.source_name", """
        select source_name, count(*) from dataset_sources group by 1 having count(*) > 1
    """),
]


def check_uniqueness(conn) -> list[Finding]:
    findings: list[Finding] = []
    for label, sql in UNIQUENESS_CHECKS:
        rows = _q(conn, sql)
        if rows:
            findings.append(Finding("CRITICAL", "benzersizlik",
                f"{label}: {len(rows)} mukerrer grup bulundu -- ornek: {rows[:5]}"))
        else:
            findings.append(Finding("INFO", "benzersizlik", f"{label}: mukerrer yok -- OK."))
    return findings


# ---------------------------------------------------------------------------
# 4) NULL/EKSIK TARAMA -- bilinen/kabul edilmis eksiklikler
# ---------------------------------------------------------------------------
# Her giris: (etiket, sql -> (null_count, total_count), beklenen (null,total)
# EXACT esitlik degil -- kayitli/kabul edilmis durum bir ARALIK olarak
# tanimlanir (min_null_pct, max_null_pct) cunku kohort kucuk degisikliklerle
# (yeni hasta eklenmesi vb.) kayabilir; asil onemli olan "bu sutun/kaynak
# kombinasyonu HALA buyuk olcude bos mu" sorusu.

@dataclass
class KnownGap:
    label: str
    sql: str
    min_null_pct: float
    max_null_pct: float
    note: str


KNOWN_NULL_GAPS = [
    KnownGap(
        "patients.kps_score (UPenn-GBM)",
        """select count(*) filter (where p.kps_score is null), count(*)
           from patients p join dataset_sources ds on p.source_id = ds.source_id
           where ds.source_name = 'UPenn-GBM'""",
        80.0, 95.0,
        "Kaynakta yok, 3 yolla dogrulandi (~%12 dolu, %88 NULL beklenir).",
    ),
    KnownGap(
        "patients.mgmt_status (TCGA-GBM)",
        """select count(*) filter (where p.mgmt_status is null), count(*)
           from patients p join dataset_sources ds on p.source_id = ds.source_id
           where ds.source_name = 'TCGA-GBM'""",
        99.0, 100.0,
        "TCGA-GBM'de %0 dolu (kaynakta yok).",
    ),
    KnownGap(
        "patients.idh1_status (TCGA-GBM)",
        """select count(*) filter (where p.idh1_status is null), count(*)
           from patients p join dataset_sources ds on p.source_id = ds.source_id
           where ds.source_name = 'TCGA-GBM'""",
        99.0, 100.0,
        "TCGA-GBM'de %0 dolu (kaynakta yok).",
    ),
    KnownGap(
        "molecular_scores.tmz_class_relative",
        "select count(*) filter (where tmz_class_relative is null), count(*) from molecular_scores",
        99.0, 100.0,
        "48/48 NULL -- kaynakta uretim kurali yok, 'deprecated' olarak "
        "belgelendi (pipeline/omics_scores.py modul-sonu notu).",
    ),
    KnownGap(
        "molecular_scores.egfr_amp_flag",
        "select count(*) filter (where egfr_amp_flag is null), count(*) from molecular_scores",
        99.0, 100.0,
        "48/48 NULL -- kaynakta URETICI formul/esik yok (db-agent arastirmasi, 2026-08-18).",
    ),
    KnownGap(
        "molecular_scores.pten_del_flag",
        "select count(*) filter (where pten_del_flag is null), count(*) from molecular_scores",
        99.0, 100.0,
        "48/48 NULL -- kaynakta URETICI formul/esik yok.",
    ),
    KnownGap(
        "molecular_scores.cdkn2a_del_flag",
        "select count(*) filter (where cdkn2a_del_flag is null), count(*) from molecular_scores",
        99.0, 100.0,
        "48/48 NULL -- kaynakta URETICI formul/esik yok.",
    ),
    KnownGap(
        "molecular_scores.mgmt_interpretation",
        "select count(*) filter (where mgmt_interpretation is null), count(*) from molecular_scores",
        99.0, 100.0,
        "48/48 NULL -- kaynakta URETICI formul/esik yok.",
    ),
    KnownGap(
        "followup_series.tumor_volume_mm3",
        "select count(*) filter (where tumor_volume_mm3 is null), count(*) from followup_series",
        30.0, 67.0,
        "2026-09-12'de Baris'in onayiyla 298/896 satir YAZILDI (%33,3 dolu "
        "-> %66,7 NULL) -- canli SELECT ile db-agent tarafindan 2026-09-13'te "
        "birebir dogrulandi (598 NULL / 896 toplam = %66,74). Eski sabit "
        "(%99-100 NULL) bu yazimdan ONCEKI 'Mert henuz hesapliyor' durumunu "
        "yansitiyordu, artik BAYAT. ALT SINIR (%30,0) KOR bir sekilde "
        "%66,7'ye SABITLENMEDI: 13 supheli + 305 belirsiz satirin (toplam "
        "318) ILERIDE yazilip yazilmayacagi HALA ACIK karar (Baris SADECE "
        "298 yuksek-guvenli satiri onayladi) -- bu 318 satirin TAMAMI "
        "ileride yazilirsa NULL orani en fazla ~%31,25'e (280/896) duser, "
        "bu da HALA BEKLENEN bir durumdur, CRITICAL basmamali. UST SINIR "
        "(%67,0) ise GUNCEL durumu (%66,74) kucuk bir yuvarlama payiyla "
        "kilitler -- oranin YUKARI cikmasi (veri SILINMESI veya var olan "
        "degerlerin NULL'a geri donmesi) beklenmez, boyle bir sapma hala "
        "CRITICAL olarak yakalanmalidir.",
    ),
    KnownGap(
        "patients.vital_status (LUMIERE)",
        """select count(*) filter (where p.vital_status is null), count(*)
           from patients p join dataset_sources ds on p.source_id = ds.source_id
           where ds.source_name = 'LUMIERE'""",
        99.0, 100.0,
        "LUMIERE olay verisi saglamiyor (vital_status 0/91), canli DB + "
        "orijinal yayinla dogrulandi.",
    ),
]


def check_null_completeness(conn) -> list[Finding]:
    findings: list[Finding] = []

    for gap in KNOWN_NULL_GAPS:
        null_n, total_n = _q(conn, gap.sql)[0]
        pct = (100.0 * null_n / total_n) if total_n else 0.0
        if gap.min_null_pct <= pct <= gap.max_null_pct:
            findings.append(Finding("INFO", "bilinen-eksiklik",
                f"{gap.label}: NULL {null_n}/{total_n} (%{pct:.1f}) -- BEKLENEN aralikta "
                f"[%{gap.min_null_pct}-%{gap.max_null_pct}]. {gap.note}"))
        else:
            findings.append(Finding("CRITICAL", "beklenmeyen-degisiklik",
                f"{gap.label}: NULL {null_n}/{total_n} (%{pct:.1f}) -- BEKLENEN aralik disinda "
                f"[%{gap.min_null_pct}-%{gap.max_null_pct}]. Bilinen eksiklik profili DEGISMIS "
                "olabilir (iyilesme veya yeni bir regresyon) -- incele."))

    # combat_parameters 0 satir -- kabul edilmis (ComBat ozdeslik donusumu, hic
    # fit edilmedi). Bu bir "NULL orani" degil, tablo-seviyesi bos durum --
    # ayrica ele aliniyor.
    n = _q(conn, "select count(*) from combat_parameters")[0][0]
    if n == 0:
        findings.append(Finding("INFO", "bilinen-eksiklik",
            "combat_parameters: 0 satir -- BEKLENEN (ComBat referans-batch=UPenn, "
            "egitim havuzu=UPenn, ozdeslik donusumu -- hic fit edilmedi, "
            "decisions/2026-08-13-combat-ozdeslik-bulgusu-ve-kapsami.md)."))
    else:
        findings.append(Finding("WARN", "bilgi",
            f"combat_parameters artik {n} satir iceriyor -- daha once 0'di, "
            "muhtemelen ComBat gercekten fit edildi (LUMIERE egitime girdi veya "
            "reference_batch degisti mi? guard kontrol edilmeli)."))

    return findings


def scan_unexpected_nulls(conn, tracked_labels: set[str]) -> list[Finding]:
    """Kayitli KNOWN_NULL_GAPS listesinin DISINDA, gorev talimatinin adini
    verdigi kritik kolonlarda NULL var mi -- kesif amacli, GATING DEGIL
    (INFO/WARN olarak raporlanir, exit code'u etkilemez). Amac: bilinen
    listeye eklenmemis YENI bir eksiklik kesfedilirse insan gozden
    gecirmesine sunulsun."""
    findings: list[Finding] = []
    candidates = [
        ("patients.age", "select count(*) filter (where age is null), count(*) from patients"),
        ("patients.gender", "select count(*) filter (where gender is null), count(*) from patients"),
        ("patients.vital_status (tum kaynaklar)",
         "select count(*) filter (where vital_status is null), count(*) from patients"),
        ("patients.survival_days", "select count(*) filter (where survival_days is null), count(*) from patients"),
        ("patients.mgmt_status (tum kaynaklar)",
         "select count(*) filter (where mgmt_status is null), count(*) from patients"),
        ("patients.idh1_status (tum kaynaklar)",
         "select count(*) filter (where idh1_status is null), count(*) from patients"),
        ("patients.gtr_over90percent", "select count(*) filter (where gtr_over90percent is null), count(*) from patients"),
        ("patients.diagnosis_type", "select count(*) filter (where diagnosis_type is null), count(*) from patients"),
        ("mr_scans.scan_date", "select count(*) filter (where scan_date is null), count(*) from mr_scans"),
    ]
    for label, sql in candidates:
        if label in tracked_labels:
            continue
        null_n, total_n = _q(conn, sql)[0]
        if null_n == 0:
            continue
        pct = 100.0 * null_n / total_n if total_n else 0.0
        findings.append(Finding("WARN", "kesif-null-taramasi",
            f"{label}: NULL {null_n}/{total_n} (%{pct:.1f}) -- KNOWN_NULL_GAPS "
            "registry'sinde YOK, gorev talimatinin bilinen-eksiklik listesinde de "
            "belirtilmemis. Yeni bir bulgu olabilir, insan incelemesi onerilir."))
    return findings


# ---------------------------------------------------------------------------
# 5) DEGER ALANI (DOMAIN) KONTROLLERI
# ---------------------------------------------------------------------------

DOMAIN_ALLOWED = {
    "patients.vital_status": {"DECEASED", "ALIVE", "CENSORED"},
    "patients.gender": {"Male", "Female"},
    "patients.gtr_over90percent": {"Y", "N"},
    # 2026-09-13 GUNCELLENDI (T4 migrasyonu, db-agent-D): CHECK constraint
    # SAF GENISLETME olarak degisti (mr_scans_harmonization_status_check),
    # 2 yeni deger eklendi -- eski 4 deger KALDI (geriye donuk uyumluluk).
    # Canli olcum (2026-09-13, migrasyon SONRASI): raw=191,
    # zscore_t1ce_c32=1530, zscore_pooled_legacy=3808, zscore=0, n4=0,
    # combat=0. Kaynak: decisions/2026-09-11-harmonization-status-kalici-
    # cozum.md (KARARLASTI, 2026-09-13; 2026-09-14'te dosya adindan
    # "-TASLAK" eki kaldirildi, eski ad artik yok, bkz. K23).
    "mr_scans.harmonization_status": {
        "raw", "n4", "zscore", "combat",
        "zscore_pooled_legacy", "zscore_t1ce_c32",
    },
}

# idh1_status icin mimari-belgeli kanonik set (v45.txt satir 313: "546
# Wildtype, 19 Mutated, 106 NOS/NEC" -- UPenn icin). mgmt_status icin
# kanonik set UPenn/UCSF'in kullandigi Title-case desenden alindi (v45.txt'de
# ayri bir enum listesi YOK, bu yuzden bu liste "gozlemsel kanonik" -- LUMIERE
# farkli bir sozluk kullaniyorsa bu bir NORMALIZASYON bulgusu olarak
# raporlanir, sessizce yutulmaz).
IDH1_CANONICAL = {"Wildtype", "Mutated", "NOS/NEC"}
MGMT_CANONICAL = {"Methylated", "Unmethylated", "Indeterminate"}

# 2026-09-13 EKLENDI (Baris onayi, LUMIERE normalizasyon gorevi) -- BU
# ASAGIDAKI IKI SOZLUK, `api/predict.py::IDH1_LABEL_NORMALIZATION_MAP`/
# `MGMT_LABEL_NORMALIZATION_MAP`'in BAGIMSIZ (import EDILMEYEN, elle
# senkronize edilen) bir KOPYASIDIR -- BU dosya (`data_integrity_check.py`)
# hafif/bagimsiz kalmasi icin BILINCLI olarak `api/` paketini (FastAPI/
# shap/lifelines bagimliliklari) import ETMEZ. Otorite kaynagi (gercek
# davranisi belirleyen) HER ZAMAN `api/predict.py`'deki orijinaldir --
# BU kopya SADECE denetim/rapor baglami icindir, HICBIR veri DONUSTURMEZ.
# Ikisi DRIFT ederse (biri guncellenip digeri unutulursa) asagidaki
# `check_domain_values()` "bilinen/normalize-edilen" ile "GERCEKTEN yeni/
# kapsam-disi" degerleri AYIRT EDEMEZ -- bu YENI bir manuel-senkronizasyon
# riskidir, `decisions/2026-09-13-lumiere-idh-mgmt-etiket-normalizasyonu.md`
# icinde ACIKCA belirtilmistir.
KNOWN_NORMALIZED_IDH1_RAW_VALUES = {
    "WT": "Wildtype", "wt": "Wildtype", "R132H mut": "Mutated",
    "IDH1 neg, Sequencing required": "NOS/NEC",
}
KNOWN_NORMALIZED_MGMT_RAW_VALUES = {
    "methylated": "Methylated", "not methylated": "Unmethylated",
}


def check_domain_values(conn) -> list[Finding]:
    findings: list[Finding] = []

    for col, allowed in DOMAIN_ALLOWED.items():
        table, column = col.split(".")
        rows = _q(conn, f"select {column}, count(*) from {table} group by 1")
        bad = {v: c for v, c in rows if v is not None and v not in allowed}
        if bad:
            findings.append(Finding("CRITICAL", "domain",
                f"{col}: izin verilmeyen degerler bulundu: {bad} (izinli set: {sorted(allowed)})."))
        else:
            findings.append(Finding("INFO", "domain", f"{col}: tum degerler izinli set icinde -- OK."))

    idh1_rows = _q(conn, """
        select ds.source_name, p.idh1_status, count(*)
        from patients p join dataset_sources ds on p.source_id = ds.source_id
        where p.idh1_status is not null
        group by 1, 2
    """)
    non_canonical = [(s, v, c) for s, v, c in idh1_rows if v not in IDH1_CANONICAL]
    if non_canonical:
        unknown = [(s, v, c) for s, v, c in non_canonical if v not in KNOWN_NORMALIZED_IDH1_RAW_VALUES]
        known = [(s, v, c) for s, v, c in non_canonical if v in KNOWN_NORMALIZED_IDH1_RAW_VALUES]
        if unknown:
            # BILINMEYEN (haritada olmayan) deger(ler) var -- bu GERCEK/YENI
            # bir bulgu, sessizce gecistirilmez.
            findings.append(Finding("WARN", "domain-normalizasyon",
                f"patients.idh1_status: kanonik olmayan (v45.txt satir 313'te belgeli "
                f"{sorted(IDH1_CANONICAL)} disinda) VE bilinen normalizasyon "
                f"haritasinda (KNOWN_NORMALIZED_IDH1_RAW_VALUES) OLMAYAN YENI deger(ler) "
                f"bulundu: {unknown}. Bunlar `api/predict.py`'nin okuma-zamani "
                "normalizasyonundan GECMEZ -- Cox/XGBoost kovaryat kodlamasinin "
                "bunlari sessizce 'bilinmiyor' saymadigi, gercek 422 aldigi "
                "dogrulanmali."))
        if known:
            # 2026-09-13, Baris onayi (decisions/2026-09-13-lumiere-idh-mgmt-
            # etiket-normalizasyonu.md): bu degerler BILINEN/beklenen -- kaynak
            # veri (patients tablosu) BILINCLI olarak HAM birakildi (provenance
            # korunuyor), `api/predict.py::_fetch_patients_clinical_raw()`
            # OKUMA-ZAMANINDA kanonik forma cevirir. Bu WARN GURULTU DEGIL --
            # "kaynakta ham/kanonik-olmayan deger var" ifadesi HALA DOGRU,
            # sadece SERVIS KATMANI bunu ZATEN ele aliyor.
            findings.append(Finding("WARN", "domain-normalizasyon",
                f"patients.idh1_status: kanonik olmayan AMA BILINEN/normalize-edilen "
                f"deger(ler) bulundu: {known}. Kaynak veri BILINCLI olarak HAM "
                "birakildi (provenance) -- `api/predict.py`'nin okuma-zamani "
                "normalizasyonu (`IDH1_LABEL_NORMALIZATION_MAP`) bunlari kanonik "
                "forma cevirir, servis 422 ATMAZ. DB'ye toplu UPDATE UYGULANMADI "
                "(bilincli tasarim karari -- bkz. decisions/2026-09-13-lumiere-idh-"
                "mgmt-etiket-normalizasyonu.md)."))
    else:
        findings.append(Finding("INFO", "domain-normalizasyon",
            "patients.idh1_status: tum dolu degerler kanonik sette -- OK."))

    mgmt_rows = _q(conn, """
        select ds.source_name, p.mgmt_status, count(*)
        from patients p join dataset_sources ds on p.source_id = ds.source_id
        where p.mgmt_status is not null
        group by 1, 2
    """)
    non_canonical_mgmt = [(s, v, c) for s, v, c in mgmt_rows if v not in MGMT_CANONICAL]
    if non_canonical_mgmt:
        unknown_mgmt = [
            (s, v, c) for s, v, c in non_canonical_mgmt if v not in KNOWN_NORMALIZED_MGMT_RAW_VALUES
        ]
        known_mgmt = [
            (s, v, c) for s, v, c in non_canonical_mgmt if v in KNOWN_NORMALIZED_MGMT_RAW_VALUES
        ]
        if unknown_mgmt:
            findings.append(Finding("WARN", "domain-normalizasyon",
                f"patients.mgmt_status: kanonik olmayan (gozlemsel set {sorted(MGMT_CANONICAL)} "
                f"disinda) VE bilinen normalizasyon haritasinda (KNOWN_NORMALIZED_MGMT_"
                f"RAW_VALUES) OLMAYAN YENI deger(ler) bulundu: {unknown_mgmt}. Bunlar "
                "`api/predict.py`'nin okuma-zamani normalizasyonundan GECMEZ -- kod "
                "tarafinda gercek 422 aldigi dogrulanmali."))
        if known_mgmt:
            # 2026-09-13, Baris onayi -- bkz. idh1_status'un ANALOG blogu yukarida.
            findings.append(Finding("WARN", "domain-normalizasyon",
                f"patients.mgmt_status: kanonik olmayan AMA BILINEN/normalize-edilen "
                f"deger(ler) bulundu: {known_mgmt}. Kaynak veri BILINCLI olarak HAM "
                "birakildi (provenance) -- `api/predict.py`'nin okuma-zamani "
                "normalizasyonu (`MGMT_LABEL_NORMALIZATION_MAP`) bunlari kanonik "
                "forma cevirir, servis 422 ATMAZ. DB'ye toplu UPDATE UYGULANMADI "
                "(bilincli tasarim karari -- bkz. decisions/2026-09-13-lumiere-idh-"
                "mgmt-etiket-normalizasyonu.md)."))
    else:
        findings.append(Finding("INFO", "domain-normalizasyon",
            "patients.mgmt_status: tum dolu degerler kanonik sette -- OK."))

    bad_survival = _q(conn, "select patient_id, survival_days from patients where survival_days < 0 or survival_days > 10000")
    if bad_survival:
        findings.append(Finding("CRITICAL", "domain",
            f"patients.survival_days: negatif veya >10000 gun bulundu: {bad_survival[:10]}"))
    else:
        findings.append(Finding("INFO", "domain", "patients.survival_days: negatif/absurd deger yok -- OK."))

    bad_age = _q(conn, "select patient_id, age from patients where age < 0 or age > 120")
    if bad_age:
        findings.append(Finding("CRITICAL", "domain",
            f"patients.age: [0,120] araligi disinda deger bulundu: {bad_age[:10]}"))
    else:
        findings.append(Finding("INFO", "domain", "patients.age: [0,120] araliginda -- OK."))

    return findings


# ---------------------------------------------------------------------------
# 6) KAYNAKLAR ARASI TUTARLILIK
# ---------------------------------------------------------------------------

PATIENT_ID_PREFIX_RULES = {
    "UCSF-PDGM": "UCSF-PDGM-%",
    "UPenn-GBM": "UPENN-GBM-%",
    "LUMIERE": "Patient-%",
    # TCGA-GBM ve TCGA-Omics ikisi de 'TCGA-' onekini PAYLASIR (ayni kurumun
    # iki farkli kullanim amacli tablosu) -- ayri ayri degil, BIRLIKTE kontrol edilir.
}

SEGMENTATION_TOOL_TO_SOURCE = {
    "CaPTk-automatic": "UPenn-GBM",
    "CaPTk-corrected": "UPenn-GBM",
    "UPenn-PyRadiomics-107": "UPenn-GBM",
    "UPenn-PyRadiomics-107-C32": "UPenn-GBM",
    "DeepBraTumIA": "LUMIERE",
    "HD-GLIO-AUTO": "LUMIERE",
    "LUMIERE-PyRadiomics-107": "LUMIERE",
    "LUMIERE-PyRadiomics-107-C32": "LUMIERE",
    "TCGA-ground-truth": "TCGA-GBM",
    "TCGA-ground-truth-C32": "TCGA-GBM",
    "UCSF-PDGM-PyRadiomics-107-C32": "UCSF-PDGM",
}


def check_cross_source_consistency(conn) -> list[Finding]:
    findings: list[Finding] = []

    for source, pattern in PATIENT_ID_PREFIX_RULES.items():
        wrong_source = _q(conn, """
            select count(*) from patients p join dataset_sources ds on p.source_id = ds.source_id
            where ds.source_name = %s and p.patient_id not like %s
        """, (source, pattern))[0][0]
        leaked_into_others = _q(conn, """
            select count(*) from patients p join dataset_sources ds on p.source_id = ds.source_id
            where ds.source_name != %s and p.patient_id like %s
        """, (source, pattern))[0][0]
        if wrong_source or leaked_into_others:
            findings.append(Finding("CRITICAL", "kaynak-tutarliligi",
                f"{source} ({pattern}): {wrong_source} hasta bu kaynakta ama onek "
                f"uymuyor, {leaked_into_others} hasta baska kaynakta ama bu oneki tasiyor."))
        else:
            findings.append(Finding("INFO", "kaynak-tutarliligi",
                f"{source} patient_id onegi ({pattern}) -- OK, kaynak-disi sizinti yok."))

    tcga_prefix_wrong = _q(conn, """
        select count(*) from patients p join dataset_sources ds on p.source_id = ds.source_id
        where ds.source_name in ('TCGA-GBM','TCGA-Omics') and p.patient_id not like 'TCGA-%'
    """)[0][0]
    tcga_prefix_leak = _q(conn, """
        select count(*) from patients p join dataset_sources ds on p.source_id = ds.source_id
        where ds.source_name not in ('TCGA-GBM','TCGA-Omics') and p.patient_id like 'TCGA-%'
    """)[0][0]
    if tcga_prefix_wrong or tcga_prefix_leak:
        findings.append(Finding("CRITICAL", "kaynak-tutarliligi",
            f"TCGA-* onegi: {tcga_prefix_wrong} TCGA-kaynakli hasta oneksiz, "
            f"{tcga_prefix_leak} TCGA-onekli hasta TCGA-disi kaynakta."))
    else:
        findings.append(Finding("INFO", "kaynak-tutarliligi", "TCGA-* patient_id onegi -- OK."))

    tool_source_rows = _q(conn, """
        select r.segmentation_tool, ds.source_name, count(*)
        from radiomics r
        join mr_scans m on r.scan_id = m.scan_id
        join patients p on m.patient_id = p.patient_id
        join dataset_sources ds on p.source_id = ds.source_id
        group by 1, 2
    """)
    for tool, actual_source, n in tool_source_rows:
        expected_source = SEGMENTATION_TOOL_TO_SOURCE.get(tool)
        if expected_source is None:
            findings.append(Finding("WARN", "kaynak-tutarliligi",
                f"radiomics.segmentation_tool='{tool}' SEGMENTATION_TOOL_TO_SOURCE "
                f"eslemesinde YOK (kaynak={actual_source}, {n} satir) -- yeni bir "
                "arac mi eklendi, kilitli listeye eklenmeli."))
        elif expected_source != actual_source:
            findings.append(Finding("CRITICAL", "kaynak-tutarliligi",
                f"radiomics.segmentation_tool='{tool}' beklenen kaynak="
                f"'{expected_source}' ama '{actual_source}' kaynagindaki hastalarda "
                f"da bulundu ({n} satir) -- capraz-kaynak arac sizintisi."))
        else:
            findings.append(Finding("INFO", "kaynak-tutarliligi",
                f"radiomics.segmentation_tool='{tool}' <-> kaynak='{actual_source}' "
                f"({n} satir) -- OK, beklenen eslesmeyle uyumlu."))
    covered_tools = {t for t, _, _ in tool_source_rows}
    for tool, expected_source in SEGMENTATION_TOOL_TO_SOURCE.items():
        if tool not in covered_tools:
            findings.append(Finding("INFO", "kaynak-tutarliligi",
                f"radiomics.segmentation_tool='{tool}' hic satir uretmemis (bu calisma "
                "anda hicbir sorun degil, bilgi amacli)."))
    return findings


# ---------------------------------------------------------------------------
# 6b) K3 BILINEN-ISTISNA REGISTRY -- mr_scans.harmonization_status='raw' +
#     C32 radyomik satiri kombinasyonu icin KIMLIK-seviyeli (sayi DEGIL)
#     sabitlenmis istisna kaydi.
#
#     GECICI COZUM -- KALICI DEGIL. Kalici cozum radyomik cikarim BASINA
#     preprocessing provenance'tir (orn. `radiomics` tablosuna
#     `n4_applied`, `zscore_scope`, `bin_count` kolonlari/ayri bir
#     provenance tablosu eklenmesi) -- bu, `mr_scans.harmonization_status`
#     tek-kolon tasariminin genel-goruntu pipeline'i
#     (run_harmonization_cohort.py) ile radyomik-ozel C32 pipeline'ini
#     (run_pyradiomics_lumiere.py) AYNI kolonda karistirmasi sorununu
#     KOKTEN cozer. `zscore_t1ce_c32` gibi yeni bir CHECK/enum degeri
#     eklemek ise SEMA DEGISIKLIGI'dir (CLAUDE.md KESIN SINIRLAR #4) --
#     ANCAK sema incelemesi + Baris'in ACIK onayi ile yapilabilir, burada
#     YAPILMADI (bkz. BEKLEYEN-KARARLAR.md K3, "B2'yi bekliyor").
#
#     DUZELTME (2026-08-19, Codex spesifikasyonuyla netlestirildi):
#     `harmonization_status` kolonunu HICBIR script YAZMIYOR -- ikisi de
#     salt-okunur baglanti aciyor:
#       - run_harmonization_cohort.py:39-40 (docstring): "DB'YE HICBIR
#         UPDATE/INSERT YAPILMAZ ... `mr_scans.harmonization_status`
#         guncellemesi Baris'in (db-agent) ... yapacagi AYRI bir
#         gorevdir." + :170 `get_connection(readonly=True)`.
#       - run_pyradiomics_lumiere.py:684 `get_connection(readonly=True)`.
#     Bu, K3'un 2026-08-18 kapanis kaydindaki (AKTIF-GOREVLER.md,
#     imaging-agent satiri) "bu kolonu YALNIZ run_harmonization_cohort.py
#     gunceller" ifadesinin DUZELTMESIDIR -- dogrusu "HICBIRI
#     guncellemiyor, guncelleme onayli AYRI bir db-agent gorevidir."
#
#     Neden LOCKED_RADIOMICS_C32 ile AYNI kapali anahtar kumesi
#     kullaniliyor (K13 dersi, BEKLEYEN-KARARLAR.md): suffix
#     `LIKE '%-C32'` yerine kapali `IN (...)` listesi -- `-C32-v2` gibi
#     yeniden adlandirilmis bir varyant LIKE ile sessizce KACARDI (K3
#     incelemesi M1 bulgusu).
#
#     KAPSAM BEYANI (Codex 2026-08-28 CRITICAL bulgusu #1'in duzeltmesi,
#     2026-09-11 db-agent):
#       - KIMLIK hash'i (`expected_row_identity_sha256`, `_hash_row_identities`)
#         SADECE 4 alani kanitlar: segmentation_tool + scan_id + patient_id +
#         tumor_region. "Bu SATIR KUMESI degismedi" der, "bu satirlarin
#         DEGERLERI degismedi" DEMEZ.
#       - ICERIK hash'i (`expected_content_sha256`, `_hash_row_content`, YENI)
#         `radiomics` tablosunun KENDI kolonlarindaki KABUL EDILEN degerleri
#         kanitlar: tumor_volume_mm3, surface_area, entropy, contrast,
#         feature_source, shape_features, first_order_features,
#         texture_features (107 ozellik JSONB olarak bu 3 alanda tutulur).
#         Ikisi BIRLIKTE: "satir kumesi VE o satirlarin degerleri baseline'la
#         BIREBIR ayni" iddiasini kanitlar.
#       - PROVENANCE (N4 uygulandi mi / Z-score kapsami / binCount) BU
#         KONTROLUN KAPSAMI DISINDADIR -- CLAUDE.md'nin kendi notu geregi
#         (`radiomics` tablosunda n4_applied/zscore_scope/bin_count KOLONU
#         YOK, bu bilgi YALNIZ CSV yaninda ayri bir `.provenance.json`
#         dosyasinda tutulur, DB'de degil). Bu script SADECE readonly SQL
#         calistirir, diskteki provenance manifest'ini OKUMAZ. Yani icerik
#         hash'i "DB'deki radyomik SAYISAL DEGERLER degismedi" der,
#         "bu degerler GERCEKTEN N4+T1ce-Z-score+binCount=32 ile uretildi"
#         DEMEZ (o iddia zaten K3'un kendi kok-neden arastirmasinin -- tek
#         provenance kosusu + n4meta.json kaniti -- konusu, burada TEKRAR
#         DOGRULANMAZ). Bu SINIR gizlenmiyor, burada ACIKCA yaziliyor.
#       - KAPSAM/UPenn: `check_c32_harmonization_status_gap()` DORT C32
#         aracinin (`LOCKED_RADIOMICS_C32` kapali listesi -- UPenn, LUMIERE,
#         UCSF, TCGA) HEPSINI tarar. 2026-09-13'e KADAR YALNIZ LUMIERE-
#         PyRadiomics-107-C32'nin KNOWN_STALE_HARMONIZATION_GAPS'te bir
#         istisna KAYDI vardi -- 2026-09-13'te T4 migrasyonu bu istisnayi
#         KAPATTI (bkz. registry tanimlamasinin ustundeki "2026-09-13
#         KAPANIS" yorumu), simdi bu liste BOS. Bu "UPenn kapsam disi"
#         ANLAMINA GELMIYORDU/GELMEZ, tam tersi: UPenn (veya TCGA/UCSF/
#         LUMIERE) icin BOYLE bir raw-status+C32 satiri BULUNURSA (henuz
#         hic bulunmadi, 2026-08-18 canli olcumu VE 2026-09-13 canli
#         olcumu 0 gosterdi) registry'de kaydi olmadigi icin DOGRUDAN
#         CRITICAL basar (asagidaki "Bilinen istisna YOK" dalina duser) --
#         yani artik TUM DORT arac AYNI sifir-tolerans kontrole tabidir.
# ---------------------------------------------------------------------------

def _hash_row_identities(rows: list[tuple]) -> str:
    """rows: (segmentation_tool, scan_id[int], patient_id[str], tumor_region[str])
    listesi (4-tuple -- daha uzun bir tuple/list verilirse ilk 4 alan
    kullanilir cagiran tarafindan onceden dilimlenmis olmalidir). Siralama
    Python'un dogal tuple siralamasiyla yapilir (scan_id NUMERIK olarak
    siralanir, sonra formatlanir) -- baseline hash'lerin hesaplandigi
    yontemle (scratchpad k3_probe.py, 2026-08-19 canli olcum) BIREBIR AYNI
    olmalidir, aksi halde INFO/CRITICAL ayrimi hatali tetiklenir."""
    ordered = sorted(rows)
    joined = "|".join(f"{a},{b},{c},{d}" for a, b, c, d in ordered)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _hash_scan_id_set(scan_ids) -> str:
    ordered = sorted(set(scan_ids))
    joined = "|".join(str(x) for x in ordered)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _hash_patient_id_set(patient_ids) -> str:
    ordered = sorted(set(patient_ids))
    joined = "|".join(ordered)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _normalize_scalar_for_hash(value) -> str:
    """NUMERIC (psycopg2 -> Decimal) ve float degerleri AYNI normalizasyon
    yolundan gecirir (ikisini de `float()`'a cevirip `repr()` alir) --
    Decimal/float tip farki YUZUNDEN yanlis-pozitif CRITICAL uretilmesin
    diye. `repr(float(x))` Python'da kisa-yol/round-trip garantili tek
    temsildir (CPython >=3.1); NUMERIC kolonu insert edilirken zaten
    `repr(float(...))`'un ONDALIK KARSILIGI yazildigi icin (bkz.
    write_lumiere_pyradiomics_to_db.py::_insert_row, psycopg2'nin float
    adapteri) bu ceviri KAYIPSIZDIR."""
    if value is None:
        return "NULL"
    try:
        return repr(float(value))
    except (TypeError, ValueError):
        return str(value)


def _normalize_json_dict_for_hash(data) -> str:
    """JSONB (shape/first_order/texture_features) icin ANAHTAR SIRASINDAN
    BAGIMSIZ, sayisal tip farkindan (int vs float vs Decimal) BAGIMSIZ bir
    kanonik JSON metni uretir. `json.dumps(..., sort_keys=True)` anahtar
    sirasini sabitler; her sayisal yapragi `float()`'a cevirmek Postgres'in
    jsonb yeniden-yazma bicimlendirmesinin (orn. `1` vs `1.0`) hash'i
    kirmasini engeller."""
    if not data:
        return "{}"
    normalized = {}
    for key, value in data.items():
        if isinstance(value, bool):
            normalized[key] = value  # bool, int'in alt sinifi -- once kontrol edilmeli
        elif isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                normalized[key] = None
            else:
                normalized[key] = float(value)
        else:
            normalized[key] = value
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _hash_row_content(rows: list[tuple]) -> str:
    """rows: (segmentation_tool, scan_id, patient_id, tumor_region,
    tumor_volume_mm3, surface_area, entropy, contrast, feature_source,
    shape_features, first_order_features, texture_features) 12-tuple
    listesi -- `radiomics` tablosunun KABUL EDILEN satirlarinin GERCEK
    DEGERLERINI kanitlar (bkz. bu bolumun basindaki KAPSAM BEYANI). Kimlik
    alanlarina (scan_id, tumor_region) gore siralanir (deterministik
    sonuc), her satir '\\n' ile ayrilir (JSON icindeki '|' karakteriyle
    kaza yaratmasin diye satir-ici alanlar '|' ile, satirlar '\\n' ile
    ayriliyor)."""
    ordered = sorted(rows, key=lambda r: (r[1], r[3]))
    parts = []
    for (tool, scan_id, patient_id, region, volume, surface_area, entropy,
         contrast, feature_source, shape, first_order, texture) in ordered:
        piece = "|".join([
            str(tool), str(scan_id), str(patient_id), str(region),
            _normalize_scalar_for_hash(volume),
            _normalize_scalar_for_hash(surface_area),
            _normalize_scalar_for_hash(entropy),
            _normalize_scalar_for_hash(contrast),
            str(feature_source) if feature_source is not None else "NULL",
            _normalize_json_dict_for_hash(shape),
            _normalize_json_dict_for_hash(first_order),
            _normalize_json_dict_for_hash(texture),
        ])
        parts.append(piece)
    joined = "\n".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class KnownStaleHarmonizationGap:
    """Tek bir bilinen, arastirilmis, kok-nedeni belgelenmis
    'harmonization_status=raw + C32 radyomik satiri' kombinasyonunu
    KIMLIK seviyesinde (sayi DEGIL) kilitler. Sayi-tabanli bir esik (orn.
    '<=265 satir ise OK') KIRILGAN olurdu -- bugun onaylanan TAM 265
    satirin SETI onaylandi, baska (farkli hastalardan) 265 satir
    ONAYLANMADI. Kimlik hash'i bu farki yakalar, sayi karsilastirmasi
    yakalamaz (bkz. log/2026-08-19.md 13:50 notu: `check_training_pool_
    counts` sinifinin yasadigi tam bu tur bir fail-open riski).

    2026-09-11 EKLENEN ALANLAR (Codex 2026-08-28 bulgu #1 + #5 duzeltmesi):
      - `expected_content_sha256`: kimlik hash'inin KAPSAMADIGI radyomik
        DEGERLERI (bkz. modul-seviyesi KAPSAM BEYANI) kanitlayan ikinci,
        BAGIMSIZ hash. Kimlik hash'inden FARKLI olarak bu deger 265x107
        ozellik pratik olarak literal EMBED edilemeyecegi icin (kimlik
        hash'inin scan_id/patient_id kumeleri gibi) import-zamaninda
        BAGIMSIZ yeniden uretilemez -- bu BILINCLI/belgelenmis bir sinirdir
        (asagida `expected_content_sha256` alaninin yorumuna bakin).
      - `expected_scan_id_to_patient_id` + `expected_tumor_regions`:
        `expected_row_identity_sha256`'nin (kimlik hash'i) SIFIR-DB ile
        BAGIMSIZ yeniden uretilebilmesini saglar (Codex bulgu #5) --
        `_build_expected_identity_rows()` bu iki alandan capraz-carpim
        (her scan_id x her tumor_region) ile tam satir kimligi listesini
        YENIDEN INSA eder, `_assert_known_stale_gaps_self_consistent()`
        bunu `expected_row_identity_sha256` ile karsilastirir (import-zamani
        assert, DB GEREKTIRMEZ). VARSAYIM: bu gap'teki HER scan_id AYNI
        `expected_tumor_regions` kumesini tasir -- bu varsayim LUMIERE
        girisinde CANLI CSV/DB yazim raporundan (2026-09-11, 53 scan x
        5 bolge = 265 satir, HICBIR sapma yok) DOGRULANDI. Varsayim
        tutmazsa (`expected_n_rows` ile capraz-carpim uzunlugu
        UYUSMAZSA) import ACIKCA patlar, sessizce yanlis deger uretmez."""

    segmentation_tool: str
    expected_n_scans: int
    expected_n_rows: int
    expected_n_patients: int
    # Literal kumeler -- hash'in NE'yi ozetledigini dogrulamak (modul
    # import-zamaninda self-check) VE uyusmazlik halinde eklenen/cikan
    # scan_id/patient_id DELTALARINI raporlayabilmek icin AYRICA saklanir
    # (yalnizca hash yeterli olsaydi delta raporlanamazdi).
    expected_scan_ids: frozenset
    expected_patient_ids: frozenset
    expected_row_identity_sha256: str
    expected_scan_id_set_sha256: str
    expected_patient_id_set_sha256: str
    # YENI (2026-09-11, Codex bulgu #5): kimlik hash'inin BAGIMSIZ
    # yeniden-uretimi icin gereken literal kaynak veri.
    expected_scan_id_to_patient_id: dict
    expected_tumor_regions: frozenset
    # YENI (2026-09-11, Codex bulgu #1): radyomik DEGERLERI kanitlayan
    # ikinci hash -- bkz. modul-seviyesi KAPSAM BEYANI + yukaridaki
    # docstring notu (bagimsiz yeniden-uretilemez, opak baseline).
    expected_content_sha256: str
    decision_ref: str
    review_by: str


# Baseline'lar 2026-08-19'da CANLI DB'den readonly SELECT ile olculdu
# (scratchpad k3_probe.py, db-agent). Bu sabitler o olcumun DONDURULMUS
# goruntusudur -- kohort degisirse (yeni C32 kosusu, yeniden-isleme vb.)
# registry BILEREK BAYATLAR ve asagidaki kontrol CRITICAL basar (sessizce
# INFO kalmaz).
#
# 2026-09-13 KAPANIS (db-agent-E, Baris'in onayiyla) -- LUMIERE-PyRadiomics-
# 107-C32 KAYDI ASAGIDAKI AKTIF LISTEDEN CIKARILDI (KENDILIGINDEN DEGIL,
# ACIKCA raporlanan bir karar): T4 migrasyonu mr_scans.harmonization_status
# degerlerini guncelledi (canli dagilim: raw=191, zscore_t1ce_c32=1530,
# zscore_pooled_legacy=3808, toplam 5529 -- koordinator BAGIMSIZ dogruladi).
# Bu, bu kaydin belgeledigi 53 tarama/265 satir/7 hastalik "raw-status AMA
# gercekte C32-sozlesmeli" durumunu KAPATTI: canli SELECT ile dogrulandi --
# bu 53 scan_id'nin TAMAMI artik dogru sekilde harmonization_status=
# 'zscore_t1ce_c32' (`select harmonization_status, count(*) from mr_scans
# where scan_id = ANY(<53 scan_id>) group by 1` -> tek satir
# ('zscore_t1ce_c32', 53)); LOCKED_RADIOMICS_C32['LUMIERE-PyRadiomics-107-
# C32'] (2920 satir/585 tarama) DEGISMEDI -- yalniz METADATA bayragi
# duzeldi, radyomik VERI degismedi. `review_by="2026-09-19"` bu yuzden
# ARTIK GEREKSIZ: gozden gecirilecek "gecici" bir durum kalmadi, kayit
# gelecege TASINMADI, KAPATILDI. Orijinal literal veriler (hash'leriyle
# birlikte) TARIHSEL bir regresyon sabiti olarak asagida
# `_CLOSED_GAP_LUMIERE_PYRADIOMICS_107_C32_2026_09_13` altinda SAKLANIR
# (hash fonksiyonlarinin gercek 265 satirlik LUMIERE C32 verisi uzerinde
# doğru calistigini kanitlayan cross-check testleri icin) -- bu sabit
# ARTIK KNOWN_STALE_HARMONIZATION_GAPS'in (aktif tarama listesinin)
# PARCASI DEGIL, canli taramada hicbir davranis tetiklemez. Detay:
# log/2026-09-13.md, AKTIF-GOREVLER.md (db-agent-E satiri).
KNOWN_STALE_HARMONIZATION_GAPS: list[KnownStaleHarmonizationGap] = []

_CLOSED_GAP_LUMIERE_PYRADIOMICS_107_C32_2026_09_13 = (
    KnownStaleHarmonizationGap(
        segmentation_tool="LUMIERE-PyRadiomics-107-C32",
        expected_n_scans=53,
        expected_n_rows=265,
        expected_n_patients=7,
        expected_scan_ids=frozenset({
            3487, 3491, 3495, 3695, 3699, 3703, 3707, 3711, 3715, 3719,
            3723, 3727, 3731, 3735, 3739, 3743, 3839, 3843, 3847, 3851,
            3855, 3859, 3863, 3867, 3871, 3875, 4047, 4051, 4055, 4059,
            4827, 4831, 4835, 4839, 4843, 4847, 4851, 4855, 4859, 4863,
            4867, 4871, 4875, 4879, 4883, 4887, 4891, 4895, 4915, 4919,
            4923, 4927, 4931,
        }),
        expected_patient_ids=frozenset({
            "Patient-027", "Patient-032", "Patient-036", "Patient-037",
            "Patient-045", "Patient-073", "Patient-075",
        }),
        expected_row_identity_sha256=(
            "2b8aff2a8535e5218b39f583b98ee389ea1a3ab72095b1b6a7c3de6ce247c69d"
        ),
        expected_scan_id_set_sha256=(
            "abe737de8da24a0de25ecf47afc58fcb1ea62dfc6c93fcdb564d172a69a6f53c"
        ),
        expected_patient_id_set_sha256=(
            "9f5121d46d07bc8fe508a50a5f950264356185ffad5b5bcbc4a809f7fb69a605"
        ),
        # 2026-09-11 EKLENDI (Codex bulgu #5) -- `expected_row_identity_sha256`'nin
        # BAGIMSIZ yeniden-uretimi icin gereken literal kaynak: 53 scan_id -> hasta
        # eslemesi + 5 bolgelik sabit kume. Kaynak: yerel
        # `artifacts/week3/pyradiomics/lumiere_pyradiomics_c32.csv` +
        # `lumiere_c32_db_write_report.csv` (265/265 satir `db_action=YAZILDI`,
        # 2026-09-11'de db-agent tarafindan capraz-kontrol edildi -- CANLI DB
        # bu oturumda erisilemedi, bkz. `expected_content_sha256` yorumu).
        expected_scan_id_to_patient_id={
            3487: "Patient-027", 3491: "Patient-027", 3495: "Patient-027",
            3695: "Patient-032", 3699: "Patient-032", 3703: "Patient-032",
            3707: "Patient-032", 3711: "Patient-032", 3715: "Patient-032",
            3719: "Patient-032", 3723: "Patient-032", 3727: "Patient-032",
            3731: "Patient-032", 3735: "Patient-032", 3739: "Patient-032",
            3743: "Patient-032",
            3839: "Patient-036", 3843: "Patient-036", 3847: "Patient-036",
            3851: "Patient-037", 3855: "Patient-037", 3859: "Patient-037",
            3863: "Patient-037", 3867: "Patient-037", 3871: "Patient-037",
            3875: "Patient-037",
            4047: "Patient-045", 4051: "Patient-045", 4055: "Patient-045",
            4059: "Patient-045",
            4827: "Patient-073", 4831: "Patient-073", 4835: "Patient-073",
            4839: "Patient-073", 4843: "Patient-073", 4847: "Patient-073",
            4851: "Patient-073", 4855: "Patient-073", 4859: "Patient-073",
            4863: "Patient-073", 4867: "Patient-073", 4871: "Patient-073",
            4875: "Patient-073", 4879: "Patient-073", 4883: "Patient-073",
            4887: "Patient-073", 4891: "Patient-073", 4895: "Patient-073",
            4915: "Patient-075", 4919: "Patient-075", 4923: "Patient-075",
            4927: "Patient-075", 4931: "Patient-075",
        },
        expected_tumor_regions=frozenset({
            "Contrast-enhancing", "Edema", "Necrosis", "TC_derived", "WT_derived",
        }),
        # 2026-09-11 EKLENDI (Codex bulgu #1) -- `radiomics` tablosunun KABUL
        # EDILEN 265 satirinin GERCEK DEGERLERINI (tumor_volume_mm3,
        # surface_area, entropy, contrast, feature_source, shape/first_order/
        # texture_features) kanitlayan ikinci hash. KAYNAK: yerel
        # `lumiere_pyradiomics_c32.csv` (bu CSV'nin `write_lumiere_pyradiomics_
        # to_db.py::_insert_row()` tarafindan HANGI donusumle DB'ye yazildigi
        # birebir taklit edilerek, 2026-09-11'de db-agent tarafindan hesaplandi;
        # 265/265 satirin `lumiere_c32_db_write_report.csv`'de `db_action=
        # YAZILDI` oldugu ve `radiomics` tablosuna bu satirlar icin baska HICBIR
        # UPDATE yolunun bulunmadigi (repo-genelinde `UPDATE radiomics` sifir
        # eslesme) dogrulanarak CSV'nin DB icerigiyle BIREBIR ayni olmasi
        # beklenir).
        # ⚠️ CANLI DOGRULAMA YAPILAMADI: bu oturumda Supabase pooler'a baglanti
        # kurulamadi (psycopg2 OperationalError: EAUTHQUERY/ECIRCUITBREAKER,
        # 2026-09-11) -- bu deger DB erisilebilir olur olmaz
        # `python tools/data_integrity_check.py --verbose` ile TEYIT EDILMELI
        # (bu satirin INFO mu CRITICAL mi bastigina bakilarak); CRITICAL
        # basarsa bu OTOMATIK OLARAK "gercek bir icerik degisikligi" ANLAMINA
        # GELMEZ -- once bu hash'in olcum yontemi (Decimal/float normalizasyonu,
        # NUMERIC kolon hassasiyeti) canli satirla elle karsilastirilarak
        # dogrulanmali, SONRA registry guncellenmeli.
        expected_content_sha256=(
            "679ba3ffd9db29912e32a0570159eaf79c9a990856343947f148c52c3b28b348"
        ),
        decision_ref=(
            "K3 (BEKLEYEN-KARARLAR.md) kapanis kaydi -- AKTIF-GOREVLER.md "
            "imaging-agent satiri (2026-08-18) + log/2026-08-18.md ~satir "
            "1180-1206: kok neden (a) bayrak BAYAT, (b) sozlesme ihlali "
            "DEGIL. Kanit: tek provenance kosusu "
            "(lumiere_pyradiomics_c32.csv.provenance.json) n4_applied:true, "
            "zscore_scope:'t1ce_source', bin_count:32; 2/7 hastada (Patient-032/"
            "073) gercek NAS-kaynakli .n4meta.json; 265/265 satir DB'de "
            "resampled=True/volume_drift=0.0000 ile Task #19 resample-kurtarma "
            "yolundan gectigini gosteriyor. KAPANIS (2026-09-13, db-agent-E): "
            "T4 migrasyonu bu 53 taramanin harmonization_status'unu duzeltti "
            "(canli dogrulandi), K3 istisnasi bu yuzden ARTIK GEREKLI DEGIL -- "
            "bu obje KNOWN_STALE_HARMONIZATION_GAPS aktif listesinden CIKARILIP "
            "SADECE tarihsel hash-regresyon sabiti olarak burada tutuluyor."
        ),
        review_by="2026-09-19",  # KAPANIS SONRASI ARTIK GOZDEN GECIRME
        # GEREKTIRMEZ (bu obje aktif taramada kullanilmiyor) -- tarih TARIHSEL
        # kayit olarak DEGISTIRILMEDI, yeni bir review_by ATANMADI.
    )
)


def _build_expected_identity_rows(gap: "KnownStaleHarmonizationGap") -> list[tuple]:
    """`expected_row_identity_sha256`'nin (kimlik hash'i) registry'nin KENDI
    literal alanlarindan (Codex bulgu #5 duzeltmesi) BAGIMSIZ olarak, HICBIR
    DB baglantisi olmadan yeniden uretilmesini saglar: her
    `expected_scan_id_to_patient_id` girdisini (scan_id -> patient_id) her
    `expected_tumor_regions` elemaniyla CARPAR (bu gap'teki HER scan_id'nin
    AYNI bolge kumesini tasidigi varsayimi -- bkz. dataclass docstring'i).
    Donen liste `_hash_row_identities()`'e DOGRUDAN verilebilir 4-tuple
    formatindadir."""
    rows: list[tuple] = []
    for scan_id, patient_id in gap.expected_scan_id_to_patient_id.items():
        for region in gap.expected_tumor_regions:
            rows.append((gap.segmentation_tool, scan_id, patient_id, region))
    return rows


def recompute_expected_row_identity_sha256(gap: "KnownStaleHarmonizationGap") -> str:
    """Codex 2026-08-28 bulgu #5 icin PUBLIC arac fonksiyonu -- bir
    registry girdisinin `expected_row_identity_sha256` degerini KENDI
    literal alanlarindan (DB'siz) yeniden hesaplar. `tools/
    data_integrity_check.py --recompute-k3-hashes` CLI bayragi ve
    `tests/test_k3_known_stale_harmonization_gap_registry.py` bunu
    kullanir."""
    return _hash_row_identities(_build_expected_identity_rows(gap))


def _build_registry_by_tool(
    gaps: Optional[list["KnownStaleHarmonizationGap"]] = None,
) -> dict[str, "KnownStaleHarmonizationGap"]:
    """`{segmentation_tool: gap}` sozlugunu ACIK bir benzersizlik kontroluyle
    kurar (Codex bulgu #3 duzeltmesi). Duz bir dict-comprehension mukerrer
    bir `segmentation_tool` girildiginde SESSIZCE sonuncuyu tutar --
    birinci kaydin (orn. eski/yanlis bir hash'in) fark edilmeden
    ORTADAN KAYBOLMASINA yol acar. Burada bunun yerine ACIK bir hata
    firlatilir."""
    gaps = KNOWN_STALE_HARMONIZATION_GAPS if gaps is None else gaps
    tools_seen = [g.segmentation_tool for g in gaps]
    duplicates = sorted({t for t in tools_seen if tools_seen.count(t) > 1})
    if duplicates:
        raise ValueError(
            f"KNOWN_STALE_HARMONIZATION_GAPS: mukerrer segmentation_tool "
            f"kaydi bulundu: {duplicates}. Bir sozluk (dict) kurulurken bu "
            "SESSIZCE sonuncu kaydi tutar, birincisini KAYBEDER -- bu "
            "ACIKCA yasaklanir, once mukerrer girdilerden biri silinmeli "
            "veya birlestirilmeli."
        )
    return {g.segmentation_tool: g for g in gaps}


def _assert_known_stale_gaps_self_consistent() -> None:
    """Import-zamaninda calisan SAF kod self-check'i (DB gerektirmez) --
    registry'deki LITERAL scan_id/patient_id kumeleri ile yaninda kayitli
    hash sabitlerinin BIRBIRIYLE tutarli oldugunu dogrular. Amac: bu
    dosyayi elle duzenleyen biri kumeyi guncelleyip hash'i guncellemeyi
    unutursa (veya tam tersi), CANLI DB'ye hic baglanmadan, import anda
    sessiz degil GURULTULU sekilde patlasin.

    2026-09-11 EKLENEN KONTROLLER (Codex 2026-08-28 bulgu #3 + #5):
      - `_build_registry_by_tool()` cagrilarak mukerrer `segmentation_tool`
        kaydi import-zamaninda YAKALANIR (daha once yalniz kullanim
        noktasindaki sessiz dict-comprehension vardi).
      - `expected_scan_id_to_patient_id`'nin anahtar/deger kumeleri
        `expected_scan_ids`/`expected_patient_ids` ile CAPRAZ dogrulanir.
      - `expected_row_identity_sha256`, `_build_expected_identity_rows()`
        ile YENIDEN INSA edilip HASH'I TEKRAR HESAPLANARAK kayitli
        sabitle karsilastirilir (Codex bulgu #5 -- artik "opak, yeniden
        uretilemez" DEGIL).

    2026-09-13 EKLENDI (db-agent-E, K3 kapanisi): `KNOWN_STALE_HARMONIZATION_
    GAPS` T4 sonrasi BOS liste oldugu icin (aktif istisna kalmadi) bu
    fonksiyon ARTIK ayrica `_CLOSED_GAP_LUMIERE_PYRADIOMICS_107_C32_2026_09_13`
    TARIHSEL sabitini de AYNI kontrollerle dogrular -- kapatildi diye bu
    obje "elle yazilmis, dogrulanmamis" bir kalinti haline GELMEZ."""
    _build_registry_by_tool()  # yalniz yan-etkisi icin: mukerrer varsa patlar

    for gap in (
        *KNOWN_STALE_HARMONIZATION_GAPS,
        _CLOSED_GAP_LUMIERE_PYRADIOMICS_107_C32_2026_09_13,
    ):
        if len(gap.expected_scan_ids) != gap.expected_n_scans:
            raise AssertionError(
                f"KNOWN_STALE_HARMONIZATION_GAPS[{gap.segmentation_tool}]: "
                f"expected_scan_ids uzunlugu ({len(gap.expected_scan_ids)}) "
                f"expected_n_scans ({gap.expected_n_scans}) ile UYUSMUYOR.")
        if len(gap.expected_patient_ids) != gap.expected_n_patients:
            raise AssertionError(
                f"KNOWN_STALE_HARMONIZATION_GAPS[{gap.segmentation_tool}]: "
                f"expected_patient_ids uzunlugu ({len(gap.expected_patient_ids)}) "
                f"expected_n_patients ({gap.expected_n_patients}) ile UYUSMUYOR.")
        if _hash_scan_id_set(gap.expected_scan_ids) != gap.expected_scan_id_set_sha256:
            raise AssertionError(
                f"KNOWN_STALE_HARMONIZATION_GAPS[{gap.segmentation_tool}]: "
                "expected_scan_ids ile expected_scan_id_set_sha256 TUTARSIZ "
                "(transkripsiyon hatasi olabilir -- registry elle bozulmus).")
        if _hash_patient_id_set(gap.expected_patient_ids) != gap.expected_patient_id_set_sha256:
            raise AssertionError(
                f"KNOWN_STALE_HARMONIZATION_GAPS[{gap.segmentation_tool}]: "
                "expected_patient_ids ile expected_patient_id_set_sha256 TUTARSIZ "
                "(transkripsiyon hatasi olabilir -- registry elle bozulmus).")
        if set(gap.expected_scan_id_to_patient_id.keys()) != set(gap.expected_scan_ids):
            raise AssertionError(
                f"KNOWN_STALE_HARMONIZATION_GAPS[{gap.segmentation_tool}]: "
                "expected_scan_id_to_patient_id anahtarlari expected_scan_ids "
                "ile UYUSMUYOR (registry elle bozulmus olabilir).")
        if set(gap.expected_scan_id_to_patient_id.values()) != set(gap.expected_patient_ids):
            raise AssertionError(
                f"KNOWN_STALE_HARMONIZATION_GAPS[{gap.segmentation_tool}]: "
                "expected_scan_id_to_patient_id degerleri expected_patient_ids "
                "ile UYUSMUYOR (registry elle bozulmus olabilir).")
        rebuilt_rows = _build_expected_identity_rows(gap)
        if len(rebuilt_rows) != gap.expected_n_rows:
            raise AssertionError(
                f"KNOWN_STALE_HARMONIZATION_GAPS[{gap.segmentation_tool}]: "
                f"expected_scan_id_to_patient_id x expected_tumor_regions carpimi "
                f"{len(rebuilt_rows)} satir uretti, expected_n_rows="
                f"{gap.expected_n_rows} ile UYUSMUYOR -- 'her scan_id ayni bolge "
                "kumesini tasir' VARSAYIMI bu girdi icin GECERSIZ olabilir, "
                "registry'ye scan-basina bolge kumesi eklenmeli.")
        if recompute_expected_row_identity_sha256(gap) != gap.expected_row_identity_sha256:
            raise AssertionError(
                f"KNOWN_STALE_HARMONIZATION_GAPS[{gap.segmentation_tool}]: "
                "expected_scan_id_to_patient_id x expected_tumor_regions'tan "
                "YENIDEN INSA EDILEN kimlik hash'i, kayitli "
                "expected_row_identity_sha256 ile UYUSMUYOR (Codex bulgu #5 "
                "korumasi -- registry elle bozulmus olabilir).")


_assert_known_stale_gaps_self_consistent()


# check_c32_harmonization_status_gap()'in canli-DB satirlarindan bekledigi
# tuple sirasi -- SQL SELECT sirasi bununla BIREBIR ayni tutulmali.
_C32_GAP_ROW_FIELDS = (
    "segmentation_tool", "scan_id", "patient_id", "tumor_region",
    "tumor_volume_mm3", "surface_area", "entropy", "contrast",
    "feature_source", "shape_features", "first_order_features", "texture_features",
)


def _evaluate_c32_harmonization_gap(
    tool: str,
    tool_rows: list[tuple],
    gap: Optional["KnownStaleHarmonizationGap"],
) -> list[Finding]:
    """`check_c32_harmonization_status_gap()`'in DB-BAGIMSIZ karar mantigi
    (Codex 2026-08-28 bulgu #2 duzeltmesi -- bu ayirim sentetik/kalici
    pytest testlerinin CANLI DB OLMADAN bu mantigi dogrudan
    cagirabilmesini saglar, bkz.
    tests/test_k3_known_stale_harmonization_gap_registry.py).

    `tool_rows`: her biri `_C32_GAP_ROW_FIELDS` sirasinda 12-tuple
    (segmentation_tool, scan_id, patient_id, tumor_region,
    tumor_volume_mm3, surface_area, entropy, contrast, feature_source,
    shape_features, first_order_features, texture_features). Bos liste ise
    'bu arac icin raw-status+C32 satiri yok' anlamina gelir.

      - hic raw-status satiri yoksa -> OK (registry'de kayitliysa ve artik
        0 ise BAYATLAMIS OLABILECEGI ayrica bildirilir -- sessizce
        dusurulmez).
      - raw-status satiri var AMA registry'de kayit YOKSA -> CRITICAL
        (K3 istisnasi YALNIZ LUMIERE-PyRadiomics-107-C32 icin gecerli;
        UPenn/TCGA/UCSF de bu koldan gecer -- bkz. modul basindaki
        KAPSAM BEYANI).
      - raw-status satiri var VE registry'de kayit VARSA -> UC KATMANLI
        karsilastirma: (1) KIMLIK hash'i (segmentation_tool+scan_id+
        patient_id+tumor_region) uyusuyor mu, (2) ICERIK hash'i (radyomik
        DEGERLER) uyusuyor mu. Ikisi de tutarsa INFO; kimlik tutup icerik
        tutmazsa AYRI bir CRITICAL (deger degisikligi -- KIMLIK
        sabitliginin YAKALAYAMADIGI tam da bu sinif, Codex bulgu #1);
        kimlik tutmazsa (icerikten BAGIMSIZ) mevcut delta/siniflandirma
        mantigi calisir.
    """
    findings: list[Finding] = []

    if not tool_rows:
        if gap is not None:
            findings.append(Finding("CRITICAL", "ek-bulgu",
                f"K3 registry: '{tool}' icin bilinen bir raw-status istisna kaydi "
                "VAR ama canli DB'de artik HICBIR raw-status satiri YOK (0 satir). "
                "Bu MESRU bir duzelme (harmonization_status guncellendi) OLABILIR "
                "AMA registry BAYATLADI -- once dogrula, sonra bu girdiyi "
                "KNOWN_STALE_HARMONIZATION_GAPS'ten CIKAR (sessiz birakma, K3 "
                "kapanis notunu guncelle)."))
        else:
            findings.append(Finding("INFO", "ek-bulgu",
                f"harmonization_status='raw' + segmentation_tool='{tool}' "
                "kombinasyonu yok -- OK."))
        return findings

    if gap is None:
        # Bilinen istisna YOK -- K3 istisnasi YALNIZ
        # LUMIERE-PyRadiomics-107-C32 icin kayitli. Baska HERHANGI bir
        # onayli C32 aracinda (UPenn/TCGA/UCSF DAHIL) raw-status satiri
        # bulunursa bu YENI bir bulgu olabilir, sessizce yutulmaz.
        scan_ids = sorted({r[1] for r in tool_rows})
        patient_ids = sorted({r[2] for r in tool_rows})
        findings.append(Finding("CRITICAL", "ek-bulgu",
            f"mr_scans.harmonization_status='raw' olan taramalarda C32-sozlesmeli "
            f"radyomik satirlari bulundu: tool='{tool}' n_scans={len(scan_ids)} "
            f"n_rows={len(tool_rows)} n_patients={len(patient_ids)}. K3'un bilinen "
            "istisnasi YALNIZ 'LUMIERE-PyRadiomics-107-C32' icin gecerlidir -- bu "
            "arac icin kayitli bir istisna YOK. C32 sozlesmesi N4+T1ce-ozel "
            "Z-score'u ON KOSUL sayiyor (decisions/2026-08-13-pyradiomics-c32-"
            "bincount-karari.md) -- ya status guncellenmemis (metadata gecikmesi) "
            "ya da GERCEKTEN harmonize-edilmemis goruntuden cikarilmis (sozlesme "
            f"ihlali). scan_id ornekleri: {scan_ids[:10]} patient_id ornekleri: "
            f"{patient_ids[:10]}. imaging-agent/Mert'e sorulmali, kok neden ayirt "
            "edilene kadar ComBat/Cox/FAISS'e SOKULMAMALI."))
        return findings

    # Bilinen istisna VAR -- once KIMLIK, sonra ICERIK seviyesinde
    # karsilastirma yapilir (Codex bulgu #1: ikisi FARKLI iddialar).
    identity_rows = [r[:4] for r in tool_rows]
    live_scan_ids = sorted({r[1] for r in tool_rows})
    live_patient_ids = sorted({r[2] for r in tool_rows})
    live_row_hash = _hash_row_identities(identity_rows)
    live_scan_hash = _hash_scan_id_set(live_scan_ids)
    live_patient_hash = _hash_patient_id_set(live_patient_ids)
    live_counts = (len(live_scan_ids), len(tool_rows), len(live_patient_ids))
    expected_counts = (gap.expected_n_scans, gap.expected_n_rows, gap.expected_n_patients)

    identity_match = (
        live_row_hash == gap.expected_row_identity_sha256
        and live_scan_hash == gap.expected_scan_id_set_sha256
        and live_patient_hash == gap.expected_patient_id_set_sha256
        and live_counts == expected_counts
    )

    if not identity_match:
        baseline_scan_set = gap.expected_scan_ids
        baseline_patient_set = gap.expected_patient_ids
        live_scan_set = set(live_scan_ids)
        live_patient_set = set(live_patient_ids)
        added_scans = sorted(live_scan_set - baseline_scan_set)
        removed_scans = sorted(baseline_scan_set - live_scan_set)
        added_patients = sorted(live_patient_set - baseline_patient_set)
        removed_patients = sorted(baseline_patient_set - live_patient_set)

        if added_scans and not removed_scans:
            direction = "BUYUDU (yeni scan/hasta eklenmis)"
        elif removed_scans and not added_scans:
            direction = ("KUCULDU (bazi scan/hastalar artik raw-status DEGIL -- "
                         "MESRU yeniden-isleme/duzelme OLABILIR, ama baseline "
                         "yenilemesi ACIK inceleme gerektirir)")
        elif added_scans and removed_scans:
            direction = "DEGISTI (hem ekleme hem cikarma var -- takas/yeniden-isleme)"
        else:
            direction = ("AYNI KUME ama satir-kimlik hash'i FARKLI (scan/hasta "
                         "kumeleri degismedi, muhtemelen tumor_region kompozisyonu "
                         "degisti -- tam liste elle karsilastirilmali)")

        findings.append(Finding("CRITICAL", "ek-bulgu",
            f"K3 bilinen-istisna KIMLIGI UYUSMUYOR: tool='{tool}' canli DB kimligi "
            f"kayitli baseline'dan FARKLI -- siniflandirma: {direction}. canli "
            f"n_scans={live_counts[0]} n_rows={live_counts[1]} "
            f"n_patients={live_counts[2]} (beklenen {expected_counts}). "
            f"eklenen scan_id={added_scans} cikan scan_id={removed_scans} "
            f"eklenen patient_id={added_patients} cikan patient_id={removed_patients}. "
            "Baseline yenileme (registry guncelleme) ACIK inceleme + onay gerektirir "
            "-- sessizce INFO'ya dusurulmedi."))
        return findings

    # Kimlik tutuyor -- simdi ICERIK (radyomik DEGERLER) kontrol edilir.
    # Bu, kimlik hash'inin KAPSAMADIGI sinifi yakalar: ayni satirlar, AMA
    # degerleri degismis (Codex 2026-08-28 CRITICAL bulgu #1).
    live_content_hash = _hash_row_content(tool_rows)
    if live_content_hash == gap.expected_content_sha256:
        findings.append(Finding("INFO", "ek-bulgu",
            f"K3 bilinen-istisna: tool='{tool}' raw-status C32 satirlari KIMLIK "
            f"VE ICERIK bakimindan BIREBIR kayitli baseline ile eslesiyor -- "
            f"n_scans={live_counts[0]} n_rows={live_counts[1]} "
            f"n_patients={live_counts[2]}, kimlik-hash={live_row_hash[:12]}... "
            f"icerik-hash={live_content_hash[:12]}... ({gap.decision_ref}). GECICI "
            f"istisna -- gozden gecirme tarihi: {gap.review_by}. Kalici cozum HENUZ "
            "uygulanmadi (bkz. bu bolumun basindaki modul-seviyesi not: "
            "preprocessing provenance). NOT: icerik hash'i SADECE radiomics "
            "tablosunun kendi kolonlarini kanitlar, N4/Z-score PROVENANCE'ini "
            "DEGIL (bkz. modul basindaki KAPSAM BEYANI)."))
    else:
        findings.append(Finding("CRITICAL", "ek-bulgu",
            f"K3 bilinen-istisna ICERIGI UYUSMUYOR: tool='{tool}' satir KIMLIGI "
            f"(scan_id/patient_id/tumor_region kumesi) baseline ile AYNI ama "
            f"kabul edilen satirlarin RADYOMIK DEGERLERI (tumor_volume_mm3/"
            "surface_area/entropy/contrast/shape_features/first_order_features/"
            f"texture_features) FARKLI -- canli icerik-hash={live_content_hash[:16]}... "
            f"beklenen={gap.expected_content_sha256[:16]}.... Bu, kimlik hash'inin "
            "TEK BASINA YAKALAYAMAYACAGI bir drift sinifidir (Codex 2026-08-28 "
            "CRITICAL bulgu #1) -- once bu farkin MESRU mu (bilincli yeniden-"
            "isleme) yoksa bir REGRESYON mu oldugu arastirilmali, SONRA registry "
            "guncellenmeli. ⚠️ Eger bu CRITICAL, registry'nin ILK canli "
            "dogrulamasindaysa (2026-09-11'de db-agent DB'ye baglanamadigi icin "
            "expected_content_sha256 yerel CSV'den olculmustu) ONCE olcum "
            "yonteminin (Decimal/float normalizasyonu) canli satirla ELLE "
            "karsilastirilmasi gerekir -- otomatik olarak 'veri bozuldu' "
            "SONUCUNA ATLANMAMALI."))

    return findings


def check_c32_harmonization_status_gap(conn) -> list[Finding]:
    """K3 -- mr_scans.harmonization_status='raw' + C32 radyomik satiri
    kombinasyonu icin KIMLIK + ICERIK seviyesinde (sayi DEGIL) bilinen-
    istisna kontrolu. Karar mantigi `_evaluate_c32_harmonization_gap()`'te
    (DB-bagimsiz, testlenebilir); bu fonksiyon SADECE canli SELECT'i
    calistirip sonucu arac-bazinda gruplar.

    Suffix `LIKE '%-C32'` YERINE LOCKED_RADIOMICS_C32'nin KAPALI anahtar
    kumesi kullanilir (K13 dersi).
    """
    registry_by_tool = _build_registry_by_tool()  # mukerrer varsa ACIKCA patlar

    rows = _q(conn, """
        select r.segmentation_tool, r.scan_id, m.patient_id, r.tumor_region,
               r.tumor_volume_mm3, r.surface_area, r.entropy, r.contrast,
               r.feature_source, r.shape_features, r.first_order_features,
               r.texture_features
        from radiomics r
        join mr_scans m on r.scan_id = m.scan_id
        where m.harmonization_status = 'raw'
          and r.segmentation_tool = ANY(%s)
    """, (list(LOCKED_RADIOMICS_C32.keys()),))
    raw_rows_by_tool: dict[str, list[tuple]] = {}
    for row in rows:
        tool = row[0]
        raw_rows_by_tool.setdefault(tool, []).append(tuple(row))

    findings: list[Finding] = []
    for tool in LOCKED_RADIOMICS_C32:  # kapali, deterministik sira
        tool_rows = raw_rows_by_tool.get(tool, [])
        gap = registry_by_tool.get(tool)
        findings += _evaluate_c32_harmonization_gap(tool, tool_rows, gap)
    return findings


# Butun MESRU segmentation_tool degerleri -- canli DB'de 2026-08-19'da
# dogrulandi (11 deger: nesil-1 precompute 4 + nesil-2 A-yontemi 3 +
# nesil-3 C32 4). SEGMENTATION_TOOL_TO_SOURCE zaten bu 11 aracin
# TAMAMINI iceriyor (kaynak eslemesi icin de kullanilir) -- ayri bir
# liste TUTULMUYOR, tek otoriter kaynak SEGMENTATION_TOOL_TO_SOURCE'tur.
KNOWN_SEGMENTATION_TOOLS = frozenset(SEGMENTATION_TOOL_TO_SOURCE.keys())


def check_segmentation_tool_whitelist(conn) -> list[Finding]:
    """BAGIMSIZ kontrol (K3 Codex spesifikasyonu madde 7) -- kayitsiz/
    yeniden-adlandirilmis bir segmentation_tool varyanti kapali IN(...)
    listelerini (LOCKED_RADIOMICS_C32 dahil) sessizce atlayamasin.
    check_cross_source_consistency() zaten benzer bir WARN uretiyor
    (kaynak eslemesi eksikse) -- bu fonksiyon AYRI ve daha SERT
    (CRITICAL): amac yalnizca 'bu deger hic kayitli mi' sorusu, kaynak
    eslemesi degil. Iki katman kasitli (RAG bypass ikili-kapi deseniyle
    tutarli) -- biri kacsa digeri yakalar."""
    findings: list[Finding] = []
    distinct_tools = {r[0] for r in _q(conn, "select distinct segmentation_tool from radiomics")}
    unknown = sorted(distinct_tools - KNOWN_SEGMENTATION_TOOLS)
    if unknown:
        findings.append(Finding("CRITICAL", "ek-bulgu",
            f"radiomics.segmentation_tool: KAYITSIZ/beyaz-liste-disi deger(ler) "
            f"bulundu: {unknown}. KNOWN_SEGMENTATION_TOOLS ({len(KNOWN_SEGMENTATION_TOOLS)} "
            "mesru arac) disinda -- yeniden adlandirilmis bir varyant (orn. "
            "'-C32-v2') olabilir, kapali IN(...) listelerini sessizce atlar. Once bu "
            "aracin ne oldugu arastirilmali, sonra ilgili kilitli listelere (LOCKED_"
            "RADIOMICS_C32 ve/veya SEGMENTATION_TOOL_TO_SOURCE) ACIKCA eklenmeli."))
    else:
        findings.append(Finding("INFO", "ek-bulgu",
            f"radiomics.segmentation_tool: canli {len(distinct_tools)} deger, hepsi "
            f"KNOWN_SEGMENTATION_TOOLS beyaz listesinde ({len(KNOWN_SEGMENTATION_TOOLS)} "
            "mesru arac) -- OK."))
    return findings


# ---------------------------------------------------------------------------
# 7) EK BULGULAR -- calisma sirasinda bagimsiz kesfedilen somut tutarsizliklar
# ---------------------------------------------------------------------------

def check_additional_findings(conn) -> list[Finding]:
    """
    n_patients UZLASMA NOTU (2026-09-12, Codex/reviewer bagimsiz dogrulamasinin
    #4 sartinin duzeltmesi): bu fonksiyonun modul dosyasinin BASINDAKI genel
    aciklamasi (kategori 7, "orn. n_patients metadata sapmasi...") bu kontrolu
    HALA "acik/cozulmemis" bir ornek gibi sunuyordu -- ESKI/YANILTICI. Gercek
    durum: bu CRITICAL 2026-08-19 13:35'te BULUNDU VE DUZELTILDI (log/2026-08-19.md
    ayni saat damgasi): TCGA-GBM 260->261, TCGA-Omics 48->34 (48'in `omics_
    profiles` satir sayisiyla karismasi kok nedeniydi). `sql-dogrulama-protokolu`
    birebir izlendi (dry-run -> Baris onayi -> guard'li --apply [BEKLENEN eski
    degerle WHERE kosulu] -> ayri readonly baglantidan canli SELECT dogrulamasi).
    2026-09-12'de bu fonksiyon canli DB'ye karsi YENIDEN calistirildi (db-agent,
    K3 4-sart turu) -- **5/5 kaynakta 0 uyusmazlik** (asagidaki `mismatches`
    sorgusu bos donuyor, CRITICAL-0). Yani bu artik "canli/aktif bir bulgu"
    DEGIL, "gecmiste yakalanip kapatilan, hala SAVUNMACI olarak calisan bir
    regresyon kontrolu"dur -- fonksiyon BURADA KALIR (silinmez), cunku yeni bir
    kaynak eklenirse veya n_patients elle bozulursa TEKRAR CRITICAL basar.
    Modul basindaki genel kategori-7 aciklamasi bu notla TUTARLI okunmalidir;
    orada "sapma bulunabilir" ORNEGI verilirken burada "su an sapma YOK, ama
    kontrol surekli calisir" netligi saglanir.
    """
    findings: list[Finding] = []

    # (a) dataset_sources.n_patients alani ile patients tablosundaki GERCEK
    # kayit sayisi arasinda fark var mi -- bu alan bir metadata/ozet
    # kolonudur, patients'tan TURETILMEDEN elle/ayri bir surecte
    # doldurulmus olabilir, bu yuzden CANLI veriyle senkron olmasi gerekir.
    # (2026-08-19'da TAM BU KONTROLLE 2 satir sapma bulunup duzeltildi --
    # yukaridaki fonksiyon docstring'ine bkz.)
    mismatches = _q(conn, """
        select ds.source_id, ds.source_name, ds.n_patients, count(p.patient_id) actual
        from dataset_sources ds left join patients p on p.source_id = ds.source_id
        group by 1, 2, 3
        having ds.n_patients != count(p.patient_id)
    """)
    if mismatches:
        findings.append(Finding("CRITICAL", "ek-bulgu",
            f"dataset_sources.n_patients alani GERCEK patients sayisiyla UYUSMUYOR: "
            f"{mismatches} (source_id, source_name, n_patients, actual_count). "
            "Bu bir ozet/metadata alani -- yaniltici olabilir, raporlarda/dashboard'da "
            "kullaniliyorsa yanlis sayi gosterebilir."))
    else:
        findings.append(Finding("INFO", "ek-bulgu", "dataset_sources.n_patients tum kaynaklarda actual ile uyumlu -- OK."))

    # (b) mr_scans.harmonization_status='raw' + '*-C32' radyomik satiri
    # kombinasyonu ARTIK burada DEGIL -- K3 bilinen-istisna registry'sine
    # tasindi (bkz. check_c32_harmonization_status_gap() ve
    # check_segmentation_tool_whitelist(), asagida, kategori 6 ile 7
    # arasinda). Gerekce: LUMIERE-PyRadiomics-107-C32'nin 53 tarama/265
    # satir/7 hastalik durumu 2026-08-18'de KOK NEDENI BELGELENMIS bilinen
    # bir bulgu (K3, log/2026-08-18.md ~1180) -- her koşuda sahte CRITICAL
    # basmak yerine KIMLIK-seviyeli (sayi degil) bir istisna kaydiyla INFO'ya
    # indirildi, AMA yeni/farkli bir ihlal (baska bir arac, farkli scan/hasta
    # kumesi) hala CRITICAL olarak yakalanir. Detay: modul-seviyesi
    # `KnownStaleHarmonizationGap` docstring'i.

    return findings


# ---------------------------------------------------------------------------
# 6c) K3 REGISTRY EXPIRY KAPISI (Codex 2026-08-28 bulgu #4 duzeltmesi)
# ---------------------------------------------------------------------------
# ONCEKI DURUM: "GECICI istisna" etiketi SADECE INFO mesaji icindeki bir
# metin parcasiydi -- `review_by` tarihi GECSE BILE hicbir davranis
# degismiyordu (sonsuza kadar sessizce INFO kalabilirdi).
#
# DAVRANIS TANIMI (2026-09-11'de BILINCLI SECILDI -- bugun review_by'dan
# (2026-09-19) 8 gun ONCE, bu script BUGUN veya onumuzdeki 8 gun icinde
# calistirilirsa hicbir WARN/CRITICAL UYARISI DEGISMEZ, `check_registry_
# expiry` INFO basar):
#   - Varsayilan (CLI `--strict-expiry` VERILMEDEN): suresi gecmis bir
#     kayit WARN basar -- exit code'u ETKILEMEZ, mevcut CI/otomasyonun
#     BUGUNDEN ITIBAREN aniden kirmiziya donmesi ISTENMEDI (bu SESSIZ bir
#     tolerans DEGIL -- WARN insan gozden gecirmesini ACIKCA talep eder,
#     sadece exit code'u kirmiziya CEVIRMEZ).
#   - `--strict-expiry` ACIKCA verilirse: suresi gecmis kayit CRITICAL'e
#     yukselir (exit code 1) -- bu, "gecici" etiketinin bir GUN gercekten
#     ZORUNLU hale getirilmesini isteyen bir insanin (orn. haftalik CI
#     job'u) kullanacagi bayraktir, gunluk calistirmanin VARSAYILANI
#     DEGILDIR.
def check_registry_expiry(
    *,
    gaps: Optional[list["KnownStaleHarmonizationGap"]] = None,
    today: Optional[date] = None,
    strict: bool = False,
) -> list[Finding]:
    """DB GEREKTIRMEZ (saf tarih karsilastirmasi) -- KNOWN_STALE_HARMONIZATION_GAPS
    registry'sindeki (veya test amacli verilen `gaps` listesindeki) her
    `review_by` tarihinin GECIP GECMEDIGINI kontrol eder. `gaps`/`today`
    parametreleri testlerin GERCEK tarihe veya global registry'ye
    BAGIMLI OLMADAN sentetik senaryo kurabilmesi icin var."""
    findings: list[Finding] = []
    effective_today = today or date.today()
    for gap in (gaps if gaps is not None else KNOWN_STALE_HARMONIZATION_GAPS):
        review_by = date.fromisoformat(gap.review_by)
        if effective_today > review_by:
            days_over = (effective_today - review_by).days
            severity = "CRITICAL" if strict else "WARN"
            findings.append(Finding(severity, "registry-expiry",
                f"K3 bilinen-istisna '{gap.segmentation_tool}': gozden gecirme "
                f"tarihi {gap.review_by} GECTI ({days_over} gun once, bugun "
                f"{effective_today.isoformat()}) -- 'GECICI' etiketi artik sadece "
                "metin degil, bu kaydin insan tarafindan TEKRAR gozden "
                "gecirilmesi gerekiyor (kalici cozum: preprocessing "
                "provenance kolonlari/tablosu, bkz. modul basindaki not). "
                + ("--strict-expiry ile CRITICAL'e yukseltildi (exit code 1)."
                   if strict else
                   "Varsayilan modda SADECE WARN (exit code'u ETKILEMEZ) -- "
                   "CRITICAL'e yukseltmek icin --strict-expiry kullanin.")))
        else:
            days_left = (review_by - effective_today).days
            findings.append(Finding("INFO", "registry-expiry",
                f"K3 bilinen-istisna '{gap.segmentation_tool}': gozden gecirme "
                f"tarihine {days_left} gun var ({gap.review_by})."))
    return findings


# ---------------------------------------------------------------------------
# Orkestrasyon
# ---------------------------------------------------------------------------

def run_all_checks(conn, *, strict_expiry: bool = False) -> list[Finding]:
    tracked_labels = {gap.label for gap in KNOWN_NULL_GAPS}
    all_findings: list[Finding] = []
    all_findings += check_locked_numbers(conn)
    all_findings += check_lumiere_serviceable_population(conn)
    all_findings += check_referential_integrity(conn)
    all_findings += check_uniqueness(conn)
    all_findings += check_null_completeness(conn)
    all_findings += scan_unexpected_nulls(conn, tracked_labels)
    all_findings += check_domain_values(conn)
    all_findings += check_cross_source_consistency(conn)
    all_findings += check_c32_harmonization_status_gap(conn)
    all_findings += check_segmentation_tool_whitelist(conn)
    all_findings += check_registry_expiry(strict=strict_expiry)
    all_findings += check_additional_findings(conn)
    return all_findings


def print_report(findings: list[Finding], verbose: bool) -> int:
    by_sev: dict[str, list[Finding]] = {"CRITICAL": [], "WARN": [], "INFO": []}
    for f in findings:
        by_sev[f.severity].append(f)

    print("=" * 78)
    print("VERI BUTUNLUGU RAPORU -- gbm-aid Supabase (13 tablo, SALT-OKUNUR)")
    print("=" * 78)
    print(f"CRITICAL: {len(by_sev['CRITICAL'])}  |  WARN: {len(by_sev['WARN'])}  |  INFO: {len(by_sev['INFO'])}")
    print()

    if by_sev["CRITICAL"]:
        print("-" * 78)
        print("CRITICAL (exit code'u kirmiziya cevirir):")
        print("-" * 78)
        for f in by_sev["CRITICAL"]:
            print(f"  [CRITICAL][{f.category}] {f.message}")
        print()

    if by_sev["WARN"]:
        print("-" * 78)
        print("WARN (bilgilendirme, exit code'u ETKILEMEZ, insan gozden gecirmeli):")
        print("-" * 78)
        for f in by_sev["WARN"]:
            print(f"  [WARN][{f.category}] {f.message}")
        print()

    if verbose and by_sev["INFO"]:
        print("-" * 78)
        print("INFO (--verbose):")
        print("-" * 78)
        for f in by_sev["INFO"]:
            print(f"  [INFO][{f.category}] {f.message}")
        print()

    print("=" * 78)
    if by_sev["CRITICAL"]:
        print(f"SONUC: BASARISIZ -- {len(by_sev['CRITICAL'])} CRITICAL bulgu var.")
    else:
        print("SONUC: GECTI -- CRITICAL bulgu yok.")
    print("=" * 78)

    return 1 if by_sev["CRITICAL"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="INFO seviyesindeki tum bulgulari da bas")
    parser.add_argument(
        "--strict-expiry", action="store_true",
        help=(
            "K3 bilinen-istisna registry'sinde suresi gecmis (review_by < bugun) "
            "kayitlari WARN yerine CRITICAL'e yukselt (exit code'u etkiler). "
            "VARSAYILAN KAPALI -- gunluk calistirmalarin review_by gunu aniden "
            "kirmiziya donmemesi icin (bkz. check_registry_expiry() yorumu)."
        ),
    )
    args = parser.parse_args()

    conn = connect_readonly()
    try:
        findings = run_all_checks(conn, strict_expiry=args.strict_expiry)
    finally:
        conn.close()

    return print_report(findings, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
