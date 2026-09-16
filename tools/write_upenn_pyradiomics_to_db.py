"""upenn_pyradiomics_c32.csv (`run_pyradiomics_upenn.py` C32 çıktısı) → radiomics INSERT.

Kapsam: `run_pyradiomics_upenn.py` tarafından üretilen kanonik 107-özellik
CSV'sini `radiomics` tablosuna yeni satırlar olarak yazar. Mevcut UPenn
CaPTk-automatic/CaPTk-corrected (144-özellik, precomputed) satırlarına
DOKUNULMAZ -- ayrı bir segmentation_tool ('UPenn-PyRadiomics-107-C32') ile
UNIQUE(scan_id, segmentation_tool, tumor_region) çakışması olmadan
eklenir (bkz. decisions/2026-08-07-upenn-144-vs-107-uyumsuzlugu.md).

C32 GEÇİŞİ (A4, 2026-08-14):
`segmentation_tool` artık `UPenn-PyRadiomics-107-C32`. Eski
`UPenn-PyRadiomics-107` (1833 satır, A-yöntemi: ham görüntü + binWidth=25,
N4/Z-score ATLANMIŞ) satırları versiyonlu baseline olarak KORUNUR, bu script
onlara DOKUNMAZ (bkz. decisions/2026-08-13-pyradiomics-c32-bincount-karari.md).

SESSİZ-BAŞARISIZLIK GUARD'LARI (H6 + Codex review 2026-08-14 bulguları;
GUARD 6 karar 27 / 2026-09-13 ile EKLENDİ -- "DÖRT GUARD" ifadesi artık BAYAT,
BEŞ tane var: 1, 2, 3, 4, 6. "GUARD 5" bu script'te YOKTUR; o numara UCSF
yazıcısının scan_id-eşleme kapısına aittir, numaralar kasten çakıştırılmadı):

GUARD 1 -- C32 provenance. CSV'nin yanında `<csv_adı>.provenance.json` ZORUNLU;
  `extraction_contract/bin_count/normalize/n4_applied/zscore_scope` beklenen
  değerleri tutmalı ve `report_sha256` CSV'nin gerçek hash'iyle eşleşmeli.
  Dosya adındaki 'c32' yalnızca UYARI'dır, kapı DEĞİLDİR -- eski A-yöntemi
  CSV'siyle yeni C32 CSV'sinin BAŞLIKLARI BİREBİR AYNI (ölçüldü), şemadan
  ayırt edilemiyor ve dosya adı yeniden adlandırmayla delinir.
  `--skip-provenance-check` bypass'ı YALNIZ dry-run'da geçerlidir; `--apply`
  ile birlikte KULLANILAMAZ.

GUARD 2 -- yazım sonucu, ÜÇ DURUMLU. `--apply` modunda: hiçbir satır INSERT
  aşamasına ulaşmadıysa exit 2; INSERT sırasında DB hatası olduysa exit 2;
  yazılabilir satırların tamamı zaten DB'deyse bu hata DEĞİL (`IDEMPOTENT_NOOP`,
  exit 0 -- `--require-new-writes` ile exit 2 yapılabilir).
  Gerekçe: `ON CONFLICT DO NOTHING` çakışmada hata FIRLATMAZ.

GUARD 3 -- `mask_source` doğrulaması. `status` alanına körü körüne güvenilmez;
  izin verilmeyen bir maske kaynağından gelen satır kanonik C32 tool adı altına
  GİRMEZ (satır yazılmaz, koşu exit 2).

GUARD 4 -- kaynak CSV'deki gerçek çıkarım başarısızlıkları zararsız "atlandı"
  sayılmaz; operatör sayıyı `--accept-extraction-failures N` ile AÇIKÇA kabul
  etmeden yükleme yapılamaz. Gerekçe: çıkarımın büyük bölümü çökmüş bir koşu
  eskiden "başarıyla yüklendi" görünüp exit 0 veriyordu.

GUARD 6 -- TIMEPOINT SERT FİLTRESİ (2026-09-13, karar 27; db-agent-G).
  **BYPASS BAYRAĞI YOKTUR -- bilinçli olarak.**
  `run_pyradiomics_upenn.py::_fetch_upenn_patient_scans()` 630 UPenn hastasının
  HEPSİNİ çeker; pre-treatment (`_11`) T1ce'si olmayan hastada POST-OP (`_21`)
  taramaya DÜŞER ve bunu yalnız `timepoint_used=
  "post-op_fallback_no_pretreatment"` diye ETİKETLER -- yani eleme değil,
  işaretleme yapar. Bugün (2026-09-13 ölçümü) o 19 hastanın 38 satırı SADECE
  hazır maske bulunamadığı için (`MASKE_YOK:ReadyMaskNotFoundError`) GUARD 4'e
  takılıyor. `_21` maskeleri bir gün NAS'a eklenirse aynı satırlar `status=OK` +
  `mask_source=upenn_expert` ile gelir, GUARD 3 ve GUARD 4'ten SORUNSUZ geçer ve
  post-op radyomik SESSİZCE kanonik `UPenn-PyRadiomics-107-C32` vektörüne, oradan
  `train_cox_week3.py`'nin eğitim havuzuna girer (`train_cox_week3.py` yalnız
  `segmentation_tool`'a bakar, timepoint'e BAKMAZ) -> **kilitli 611 hasta / 585
  ölüm olayı sayısı bozulur** (630/603'e doğru sürüklenir).
  Bu yüzden: `timepoint_used != "pre-treatment"` olan ve AKSİ HALDE INSERT
  EDİLECEK OLAN her satır `HATA_TIMEPOINT_POST_OP` ile reddedilir ve koşu
  **exit 3** ile durur (diğer guard'ların exit 2'sinden AYRI kod -- otomasyonda
  "611 kilidi tehdit edildi" durumu ayırt edilebilsin). Sessiz atlama DEĞİL --
  operatör görür.
  `timepoint_used` kolonunun kendisi CSV'de YOKSA koşu daha döngüye girmeden
  exit 2 verir (fail-closed): kolonu olmayan bir CSV'de post-op satır olup
  olmadığı DOĞRULANAMAZ.
  Post-op radyomik gerçekten istenirse yolu AYRI bir `segmentation_tool` adıdır
  (örn. `UPenn-PyRadiomics-107-C32-POSTOP`), bu script DEĞİL.
  NOT (kasıtlı sıralama): GUARD 6, GUARD 3/GUARD 4'ten SONRA çalışır. Böylece
  bugünkü 38 post-op satırı (hepsi `MASKE_YOK`) eskisi gibi `HATA_CSV_STATUS`
  kalır ve 2026-08-14'teki gerçek yazım koşusunun komutu
  (`--accept-extraction-failures 38`) AYNI sonucu üretmeye devam eder --
  geçmiş koşu yeniden üretilebilir kalır. Post-op satırların sayısı yine de
  statüden BAĞIMSIZ olarak her koşuda ayrıca BASILIR (aşağıdaki
  "POST-OP FALLBACK ÖZETİ").

Satır durumları (kaynak CSV'nin `status` kolonu):
- OK                    -> INSERT (gerçek 107-özellik + hacim)
- LABEL_YOK_0_VOXEL     -> INSERT (tumor_volume_mm3=0, boş JSON'lar --
                           decisions/2026-08-07-107-ozellik-formulu.md ile tutarlı)
- T1CE_TARAMASI_YOK     -> ATLANIR, `SKIPPED_VERI_YOK` (veri yok, pipeline
                           hatası DEĞİL; sayılır ve raporlanır)
- HATA:* / GORUNTU_YOK:* / MASKE_YOK:* / GEOMETRI_UYUSMUYOR* / bilinmeyen
                        -> `HATA_CSV_STATUS`, GUARD 4'e sayılır (BAŞARISIZLIK)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import sys
from pathlib import Path

import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db_connection import get_connection  # noqa: E402

SEGMENTATION_TOOL = "UPenn-PyRadiomics-107-C32"
LEGACY_SEGMENTATION_TOOL = "UPenn-PyRadiomics-107"  # A-yöntemi baseline -- DOKUNULMAZ

# F3 (kalem 5, 2026-09-13): `LEGACY_SEGMENTATION_TOOL` eskiden TANIMLI ama HİÇ
# KULLANILMAYAN ölü bir sabitti -- kodu okuyan onu aktif bir guard sanabiliyordu.
# Artık gerçek bir çapa: bu modül eski A-yöntemi baseline'ının (1833 satır, ham
# görüntü + binWidth=25) tool adı altına ASLA yazmamalı. İki koşul de import
# anında doğrulanır; bir düzenleme `SEGMENTATION_TOOL`'u yanlışlıkla eski ada
# çevirirse modül hiç yüklenmez (fail-closed).
assert SEGMENTATION_TOOL != LEGACY_SEGMENTATION_TOOL, (
    f"SEGMENTATION_TOOL eski baseline adına eşit olamaz ({LEGACY_SEGMENTATION_TOOL!r}) -- "
    "bu script versiyonlu A-yöntemi satırlarına DOKUNMAMALI."
)
assert SEGMENTATION_TOOL == f"{LEGACY_SEGMENTATION_TOOL}-C32", (
    f"SEGMENTATION_TOOL beklenen C32 adı değil: {SEGMENTATION_TOOL!r} != "
    f"{LEGACY_SEGMENTATION_TOOL + '-C32'!r}"
)

DEFAULT_CSV = PROJECT_ROOT / "artifacts" / "week3" / "pyradiomics" / "upenn_pyradiomics_c32.csv"

# --- Rapor varsayılanları: MODA GÖRE AYRI (İŞ 3 / karar 25, 2026-09-13) --------
# Eskiden tek varsayılan vardı ve o `_db_write_report.csv` idi -- yani DRY-RUN
# varsayılan argümanlarla koşulduğunda GERÇEK YAZIM raporunun üzerine yazıyordu
# (UCSF'te olanın ters yönü). İki mod için iki ayrı varsayılan: hiçbir zaman
# aynı dosyaya düşemezler.
_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "week3" / "pyradiomics"
DEFAULT_WRITE_REPORT = _ARTIFACT_DIR / "upenn_pyradiomics_c32_db_write_report.csv"
DEFAULT_DRYRUN_REPORT = _ARTIFACT_DIR / "upenn_pyradiomics_c32_db_dryrun.csv"

# --- GUARD 6: timepoint sert filtresi (karar 27, 2026-09-13) -------------------
# Gerekçe modül-üstü docstring'de. Değer `run_pyradiomics_upenn.py`'nin ürettiği
# `timepoint_used` kolonundan gelir; bugünkü ölçüm (tam kohort CSV'si, 3104
# satır): pre-treatment=3066, post-op_fallback_no_pretreatment=38.
TIMEPOINT_COLUMN = "timepoint_used"
CANONICAL_TIMEPOINT = "pre-treatment"

# --- C32 provenance sözleşmesi (GUARD 1) -------------------------------------
# NEDEN DOSYA ADI YETMİYOR (Codex review, 2026-08-14, HIGH):
# Eski A-yöntemi CSV'siyle yeni C32 CSV'sinin BAŞLIKLARI BİREBİR AYNI (ölçüldü) --
# şemadan C32 olup olmadığı ANLAŞILAMIYOR. Dosya adı kontrolü ise yeniden
# adlandırmayla trivial biçimde delinir. Bu yüzden çıkarım script'i CSV'nin
# yanına bir provenance manifest'i yazar ve bu yazıcı onu ZORUNLU tutar.
PROVENANCE_SUFFIX = ".provenance.json"
EXPECTED_PROVENANCE = {
    "extraction_contract": "C32",
    "bin_count": 32,
    "normalize": False,
    "n4_applied": True,
    "zscore_scope": "t1ce_source",
    "pyradiomics_version": "3.0.1",
    "image_types": ["Original"],
}
# Boş/eksik olmaması gereken runtime bağları. Değerleri koşudan koşuya değişir,
# bu yüzden içerik değil VARLIK doğrulanır -- ama yokluğu manifest'in elle
# uydurulduğuna işarettir (Codex review 2026-08-14, MEDIUM-1).
REQUIRED_PROVENANCE_FIELDS = (
    "extractor_settings_sha256",
    "zscore_stats_sha256",
    "generator_script_sha256",
    "cohort",
)
# Manifest ŞEMASI KAPALI DEĞİL: buradaki alanların dışında ek alanlar (örn.
# `missing_id_count`) serbesttir, doğrulama onlara dokunmaz.

# Kanonik C32 vektörüne girmesine izin verilen maske kaynakları (GUARD 3).
# UPenn maskeleri tek çok-etiketli dosyada (NC=1/ED=2/ET=4) -- WT/TC türetmesi
# yapısal olarak güvenli. Başka bir kaynak buraya girerse fail-loud.
ALLOWED_MASK_SOURCES = frozenset({"upenn_expert", "upenn_automated_approx"})

# --- Kaynak CSV `status` sınıflandırması (GUARD 4) ----------------------------
# `run_pyradiomics_upenn.py`'nin ürettiği statü sözlüğünden türetildi (grep ile
# çıkarıldı, tahmin DEĞİL): GEOMETRI_UYUSMUYOR, GEOMETRI_UYUSMUYOR_ZSCORE,
# LABEL_YOK_0_VOXEL, T1CE_TARAMASI_YOK, GORUNTU_YOK:*, HATA:*, MASKE_YOK:*.
#
# NEDEN (Codex review 2026-08-14, HIGH): eskiden bunların HEPSİ zararsız
# `SKIPPED_*` sayılıyordu -- çıkarımın büyük bölümü çökmüş bir koşu bile
# "başarıyla yüklendi" görünüp exit 0 veriyordu. Artık gerçek çıkarım
# başarısızlıkları AYRI sayılır ve operatörün sayıyı AÇIKÇA kabul etmesi
# gerekir (`--accept-extraction-failures N`).
BENIGN_STATUSES = frozenset({"LABEL_YOK_0_VOXEL"})  # DB'ye 0-hacim satırı olarak yazılır
DATA_GAP_STATUSES = frozenset({"T1CE_TARAMASI_YOK"})  # veri yok, pipeline hatası DEĞİL
# Geri kalan HER ŞEY (HATA:*, GORUNTU_YOK:*, MASKE_YOK:*, GEOMETRI_UYUSMUYOR*,
# bilinmeyenler) gerçek çıkarım başarısızlığıdır.

# GUARD 2 yalnız INSERT SIRASINDA oluşan DB hatalarını saymalı; çıkarım-kaynaklı
# başarısızlıklar (GUARD 4) ayrı sayılır, çift sayım olmasın.
#
# F4 (kalem 5, 2026-09-13): `HATA_BAGLANTI` eskiden bu modülde ÜRETİLEMEYEN ölü
# bir önekti (retry/timeout deseni yalnız LUMIERE yazıcısında vardı). Aşağıdaki
# `_run_row_with_timeout()` + retry bloğu LUMIERE'den birebir taşındı, artık üç
# önek de gerçekten üretilebilir. `HATA_TIMEOUT` da GUARD 2'ye SAYILIR --
# LUMIERE'de bu eksikti (`"HATA_TIMEOUT:...".startswith(("HATA:","HATA_BAGLANTI"))`
# False döndüğü için asılan satırlar GUARD 2'nin hata sayacına GİRMİYORDU, yani
# fail-open'dı; aynı düzeltme LUMIERE yazıcısına da uygulandı).
DB_ERROR_PREFIXES = ("HATA:", "HATA_BAGLANTI", "HATA_TIMEOUT")

# F4 -- LUMIERE yazıcısından taşındı (bkz. write_lumiere_pyradiomics_to_db.py'nin
# 2026-08-13 NOT'u): Windows'ta libpq keepalive parametreleri kısmen yok sayıldığı
# için `cursor.execute()`/`commit()` SÜRESİZ bloke olabiliyor (istisna hiç
# fırlamıyor). Her deneme için TEK KULLANIMLIK executor açılır; paylaşımlı havuz
# KULLANILMAZ, aksi halde tek bir asılı satır havuzu kalıcı olarak tıkar.
# Asılı thread Python'da gerçekten kill EDİLEMEZ -- bilinen/belgeli sınır
# (decisions/2026-08-12-backend-harmonize-endpoint-timeout-hardening.md).
_ROW_TIMEOUT_SECONDS = 45


def _run_row_with_timeout(func, *args, **kwargs):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="upenn-row-write")
    try:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=_ROW_TIMEOUT_SECONDS)
    finally:
        executor.shutdown(wait=False)


def validate_timepoint_column(fieldnames) -> list[str]:
    """GUARD 6, birinci katman: `timepoint_used` kolonu CSV'de VAR MI?

    Kolon yoksa fail-CLOSED: post-op satır olup olmadığı doğrulanamaz, bu yüzden
    koşu hiç başlamaz. `row.get(...) is None` sessizce "sorun yok" demek OLURDU.
    """
    names = list(fieldnames or [])
    if TIMEPOINT_COLUMN not in names:
        return [
            f"kaynak CSV'de `{TIMEPOINT_COLUMN}` kolonu YOK (bulunan kolonlar: "
            f"{names[:8]}{'...' if len(names) > 8 else ''}). GUARD 6 post-op "
            "fallback satırlarını ayırt EDEMEZ -- kilitli 611 hasta / 585 olay "
            "havuzu yapısal olarak korunamaz, koşu reddedildi."
        ]
    return []


def classify_timepoint(row) -> str | None:
    """GUARD 6, ikinci katman. None = kanonik pre-treatment; aksi halde red nedeni.

    Boş/eksik değer de REDDEDİLİR (fail-closed) -- "bilinmiyor" pre-treatment
    sayılmaz.
    """
    raw = row.get(TIMEPOINT_COLUMN)
    value = (raw or "").strip() if isinstance(raw, str) else raw
    if value == CANONICAL_TIMEPOINT:
        return None
    if not value:
        return "BOS_VEYA_EKSIK"
    return str(value)[:60]


def _parse_json(raw: str) -> dict:
    if not raw:
        return {}
    return json.loads(raw)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.name + PROVENANCE_SUFFIX)


def validate_c32_provenance(csv_path: Path, *, require_complete: bool = False) -> list[str]:
    """C32 provenance manifest'ini doğrula; sorunları döndürür (boş liste = temiz).

    Manifest'i `run_pyradiomics_upenn.py` koşu sonunda CSV'nin yanına yazar.
    `report_sha256` alanı, manifest'in BU CSV'ye ait olduğunu kanıtlar --
    eski bir CSV'yi C32 adıyla yeniden adlandırmak yetmez, manifest'in
    hash'i tutmaz.
    """

    manifest_path = provenance_path(csv_path)
    if not manifest_path.exists():
        return [
            f"provenance manifest'i YOK: {manifest_path.name}. Bu CSV'nin C32 "
            "sözleşmesiyle (N4 + T1ce-özel Z-score + binCount=32) üretildiği "
            "DOĞRULANAMIYOR."
        ]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [f"provenance manifest'i okunamadı/bozuk: {type(exc).__name__}: {exc}"]
    if not isinstance(manifest, dict):
        # Geçerli JSON ama nesne değil (liste/string/null) -- kontrolsüz
        # AttributeError yerine açıklayıcı hata (Codex review 2026-08-14, LOW).
        return [f"provenance manifest'i JSON nesnesi değil: {type(manifest).__name__}"]

    problems: list[str] = []
    for key, expected in EXPECTED_PROVENANCE.items():
        actual = manifest.get(key)
        if actual != expected:
            problems.append(f"{key}: beklenen {expected!r}, bulunan {actual!r}")

    for key in REQUIRED_PROVENANCE_FIELDS:
        if not manifest.get(key):
            problems.append(
                f"`{key}` alanı yok/boş -- manifest gerçek koşuya bağlanmamış "
                "(elle uydurulmuş olabilir)"
            )

    # A1.1'in garantisini DB sınırına taşı: Z-score istatistiği TÜM kohortu
    # gördü mü? Parçalı koşuda alt-kümeden fit edilmiş istatistik buradan geçemez.
    stats_n = manifest.get("zscore_stats_image_count")
    expected_n = manifest.get("expected_image_count")
    # DİKKAT: `isinstance(x, int)` KULLANMA -- Python'da `bool`, `int` alt sınıfıdır,
    # `true` bir sayaç değeri olarak kabul edilirdi (fail-open). Codex A3, LOW.
    if type(stats_n) is not int or type(expected_n) is not int or stats_n <= 0 or expected_n <= 0:
        problems.append(
            "`zscore_stats_image_count` / `expected_image_count` POZİTİF TAMSAYI değil "
            f"(bulunan: {stats_n!r} / {expected_n!r}) -- Z-score istatistiğinin tam "
            "kohorttan fit edildiği DOĞRULANAMIYOR"
        )
    elif stats_n != expected_n:
        problems.append(
            f"Z-score istatistiği tam kohorttan fit EDİLMEMİŞ: "
            f"zscore_stats_image_count={stats_n} != expected_image_count={expected_n}. "
            "Alt-kümeden fit edilmiş istatistikle normalize edilmiş veri DB'ye giremez."
        )

    # `run_complete` yalnız gerçek yazımda (--apply) zorunlu; dry-run'da kısmi
    # bir parçayı incelemek meşru.
    if require_complete and manifest.get("run_complete") is not True:
        problems.append(
            f"`run_complete` true DEĞİL (bulunan: {manifest.get('run_complete')!r}) -- "
            "kısmi/tamamlanmamış bir koşunun çıktısı DB'ye YAZILAMAZ. "
            "Tüm parçalar aynı --report'a yazılıp koşu tamamlanmalı."
        )

    recorded_hash = manifest.get("report_sha256")
    if not recorded_hash:
        problems.append("manifest'te `report_sha256` alanı yok (CSV'ye bağlanamıyor)")
    elif recorded_hash != _sha256(csv_path):
        problems.append(
            "CSV içeriği manifest'le EŞLEŞMİYOR -- dosya yeniden adlandırılmış "
            "veya değiştirilmiş olabilir (sha256 uyuşmuyor)"
        )
    return problems


# --- Rapor dosya adı disiplini (İŞ 3 / karar 25, 2026-09-13) -------------------
# NEDEN: UCSF'in GERÇEK yazım koşusu varsayılan `--report` yolu değiştirilmeden
# koşuldu, bu yüzden tek yazım kanıtı `ucsf_c32_db_dryrun.csv` ADLI dosyada kaldı
# ve sonraki bir dry-run onu üzerine yazsa kanıt KAYBOLACAKTI. Artık `--apply`
# verildiğinde rapor adında `_write` ZORUNLU. Üç yazıcıda AYNI kural.
WRITE_REPORT_MARKER = "_write"


def validate_limit_and_apply(limit, apply: bool) -> list[str]:
    """F5 (kalem 5): `--limit` + `--apply` yasağı. Boş liste = temiz.

    Ayrı bir fonksiyon olması bilinçli: DB'ye hiç bağlanmadan birim testle
    doğrulanabilsin (bkz. tests/test_c32_writer_guards.py). Eskiden bu kontrol
    HİÇ YOKTU; `--limit 50 --apply` kohortun ilk 50 satırını yazıp `PARTIAL_RUN`
    basıyor ve **exit 0** veriyordu -- otomasyon bunu "başarılı yükleme" sayardı.
    """
    if limit is not None and apply:
        return [
            "--limit ile --apply BİRLİKTE kullanılamaz. --limit yalnız dry-run smoke "
            "testi içindir; kısmi bir yükleme 'başarıyla tamamlandı' (exit 0) görünüp "
            "kohortu SESSİZCE eksik bırakırdı. Tam yükleme için --limit'i kaldır."
        ]
    return []


def validate_report_path(report_path: Path, *, apply: bool) -> list[str]:
    """`--apply` ile rapor adı arasındaki tutarlılığı doğrular (boş liste = temiz)."""
    name = report_path.name.lower()
    if apply and WRITE_REPORT_MARKER not in name:
        return [
            f"--apply verildi ama --report dosya adında '{WRITE_REPORT_MARKER}' YOK: "
            f"{report_path.name}. Gerçek yazım kanıtı 'dryrun' adlı bir dosyaya "
            "yazılırsa sonraki bir dry-run onu üzerine yazar ve KANIT KAYBOLUR "
            "(2026-08-18 UCSF olayı, karar 25). Örnek: "
            f"{report_path.with_name(report_path.stem + '_write_report' + report_path.suffix).name}"
        ]
    return []


def _write_run_args_sidecar(report_path: Path, args, *, csv_sha256: str, summary: dict) -> Path:
    """F6 (kalem 5): koşunun CLI argümanlarını rapor CSV'sinin yanına JSON olarak yaz.

    NEDEN: rapor CSV'si yalnız satır sonuçlarını içeriyordu; hangi eşiklerle
    (özellikle kabul edilen `--accept-extraction-failures` değeriyle) koşulduğu
    HİÇBİR yere yazılmıyordu -- "38 başarısızlık kabul edildi mi, kaçı?" sorusu
    artefakta bakarak yanıtlanamıyordu. CSV'ye başlık satırı eklemek DictReader
    tüketicilerini bozacağı için yan dosya (sidecar) seçildi.
    """
    import datetime as _dt

    sidecar = report_path.with_name(report_path.name + ".run_args.json")
    payload = {
        "script": Path(__file__).name,
        "segmentation_tool": SEGMENTATION_TOOL,
        "legacy_segmentation_tool_untouched": LEGACY_SEGMENTATION_TOOL,
        "written_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "resolved_args": {
            "apply": bool(args.apply),
            "csv": str(args.csv),
            "report": str(args.report),
            "limit": args.limit,
            "skip_provenance_check": bool(args.skip_provenance_check),
            "require_new_writes": bool(args.require_new_writes),
            "accept_extraction_failures": args.accept_extraction_failures,
        },
        "thresholds": {"accept_extraction_failures": args.accept_extraction_failures},
        "source_csv_sha256": csv_sha256,
        "summary": summary,
    }
    sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return sidecar


def _sanitize_floats(data: dict) -> dict:
    """NaN/Infinity değerlerini None'a çevir -- json.dumps bunları literal NaN/Infinity
    yazar, bu geçerli JSON değil, Postgres jsonb reddeder (InvalidTextRepresentation)."""
    import math

    clean = {}
    for key, value in data.items():
        if isinstance(value, float) and not math.isfinite(value):
            clean[key] = None
        else:
            clean[key] = value
    return clean


def _insert_row(connection, *, scan_id: int, region: str, volume, surface_area, entropy, contrast, shape, first_order, texture) -> bool:
    """INSERT dener, gerçekten yeni bir satır yazıldıysa True döner.

    ON CONFLICT DO NOTHING bir çakışmada hatasız ama 0 etkilenen satırla
    döner -- cursor.rowcount kontrol edilmezse bu durum "başarı" ile
    (gerçek INSERT) ayırt edilemez, çağıran yanlışlıkla YAZILDI diyebilir.
    """
    cursor = connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO radiomics "
            "(scan_id, segmentation_tool, tumor_region, tumor_volume_mm3, "
            "surface_area, entropy, contrast, shape_features, first_order_features, "
            "texture_features, feature_source) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'computed') "
            "ON CONFLICT (scan_id, segmentation_tool, tumor_region) DO NOTHING",
            (
                scan_id,
                SEGMENTATION_TOOL,
                region,
                volume,
                surface_area,
                entropy,
                contrast,
                json.dumps(shape),
                json.dumps(first_order),
                json.dumps(texture),
            ),
        )
        return cursor.rowcount == 1
    finally:
        cursor.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"upenn_pyradiomics_c32.csv'yi radiomics tablosuna INSERT et (segmentation_tool='{SEGMENTATION_TOOL}')."
    )
    parser.add_argument("--apply", action="store_true", help="Belirtilmezse DB'ye yazılmaz (dry-run varsayılan).")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,  # MODA GÖRE çözülür, bkz. DEFAULT_*_REPORT
        help="Verilmezse moda göre seçilir: --apply ile DEFAULT_WRITE_REPORT, "
        "dry-run'da DEFAULT_DRYRUN_REPORT (karar 25).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="SMOKE TEST için ilk N satırla sınırla. F5 (2026-09-13): --apply ile "
        "BİRLİKTE KULLANILAMAZ -- kısmi yükleme exit 0 ile 'başarılı' görünürdü.",
    )
    parser.add_argument(
        "--skip-provenance-check",
        action="store_true",
        help="GUARD 1'i bilinçli olarak devre dışı bırak. SADECE manifest üretimi henüz "
        "devreye girmemişken, operatörün CSV'nin kaynağını elle doğruladığı durumda.",
    )
    parser.add_argument(
        "--require-new-writes",
        action="store_true",
        help="GUARD 2: tüm satırlar zaten DB'de olsa bile (idempotent no-op) hata say. "
        "İlk yüklemede kullanılır.",
    )
    parser.add_argument(
        "--accept-extraction-failures",
        type=int,
        default=0,
        help="GUARD 4: kaynak CSV'de kaç çıkarım başarısızlığı (HATA/GORUNTU_YOK/"
        "MASKE_YOK/GEOMETRI_UYUSMUYOR) bilinçli olarak kabul edildiği. Gerçek sayı "
        "bundan büyükse koşu reddedilir -- operatör sayıyı GÖRMEDEN yükleme yapamaz.",
    )
    args = parser.parse_args()

    # Rapor varsayılanını MODA GÖRE çöz (karar 25) -- bkz. DEFAULT_*_REPORT notu.
    if args.report is None:
        args.report = DEFAULT_WRITE_REPORT if args.apply else DEFAULT_DRYRUN_REPORT

    # HIGH-2 (Codex review 2026-08-14): provenance bypass'ı `--apply` ile
    # BİRLİKTE kullanılamaz. Aksi halde tek bayrak, C32 kapısının tamamını
    # (yanlış sözleşme + yanlış hash + değiştirilmiş CSV dahil) kaldırırdı.
    if args.skip_provenance_check and args.apply:
        print(
            "HATA: --skip-provenance-check ile --apply BİRLİKTE kullanılamaz.\n"
            "      Bypass yalnız dry-run'da (inceleme amaçlı) geçerlidir; gerçek DB\n"
            "      yazımı provenance manifest'i olmadan YAPILAMAZ. Manifest'i\n"
            "      `run_pyradiomics_upenn.py` koşu sonunda üretir -- o iş bitene\n"
            "      kadar C32 yüklemesi BLOKELİDİR (bilinçli blokaj).",
            file=sys.stderr,
        )
        return 2

    # F5 (kalem 5, 2026-09-13) -- kısmi yükleme exit 0 ile "başarılı" görünürdü.
    limit_problems = validate_limit_and_apply(args.limit, args.apply)
    if limit_problems:
        print("HATA: " + "\n      ".join(limit_problems), file=sys.stderr)
        return 2

    # İŞ 3 / karar 25 -- gerçek yazım kanıtı 'dryrun' adlı bir dosyaya yazılamaz.
    report_problems = validate_report_path(args.report, apply=args.apply)
    if report_problems:
        print("HATA: " + "\n      ".join(report_problems), file=sys.stderr)
        return 2
    if not args.apply and WRITE_REPORT_MARKER in args.report.name.lower():
        print(
            f"UYARI: dry-run raporu '{WRITE_REPORT_MARKER}' içeren bir ada yazılıyor "
            f"({args.report.name}) -- varsa GERÇEK yazım kanıtının ÜZERİNE yazar. "
            "Ayrı bir '_dryrun' adı kullanmayı düşün.",
            file=sys.stderr,
        )

    if not args.csv.exists():
        print(f"HATA: --csv dosyası bulunamadı: {args.csv}", file=sys.stderr)
        return 2

    # GUARD 1 -- yanlış/eski CSV'nin C32 tool adı altında yazılmasını engelle.
    # Dosya adı yalnızca ucuz bir ön uyarı; ASIL kapı provenance manifest'idir
    # (eski ve yeni CSV başlıkları birebir aynı, şemadan ayırt edilemiyor).
    if "c32" not in args.csv.name.lower():
        print(
            f"UYARI: --csv dosya adında 'c32' geçmiyor: {args.csv.name}", file=sys.stderr
        )
    provenance_problems = validate_c32_provenance(args.csv, require_complete=args.apply)
    if provenance_problems:
        if args.skip_provenance_check:
            print(
                "UYARI: GUARD 1 --skip-provenance-check ile atlandı. Sorunlar:\n  - "
                + "\n  - ".join(provenance_problems),
                file=sys.stderr,
            )
        else:
            print(
                f"HATA: C32 provenance doğrulaması BAŞARISIZ ({args.csv.name}).\n  - "
                + "\n  - ".join(provenance_problems)
                + f"\n      Bu script segmentation_tool='{SEGMENTATION_TOOL}' ile yazıyor;\n"
                f"      yanlış kaynaklı veri bu ad altına GİRMEMELİ.\n"
                f"      Manifest'i `run_pyradiomics_upenn.py` koşu sonunda üretir.\n"
                f"      Bilinçli istisna: --skip-provenance-check",
                file=sys.stderr,
            )
            return 2

    with args.csv.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        csv_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    # GUARD 6, birinci katman -- kolon yoksa koşu HİÇ BAŞLAMAZ (fail-closed).
    timepoint_column_problems = validate_timepoint_column(csv_fieldnames)
    if timepoint_column_problems:
        print(
            "HATA (GUARD 6): " + "\n      ".join(timepoint_column_problems)
            + "\n      Bu guard'ın BYPASS BAYRAĞI YOKTUR (karar 27, 2026-09-13).",
            file=sys.stderr,
        )
        return 2

    if args.limit:
        rows = rows[: args.limit]

    # GUARD 6 bilgilendirme -- post-op fallback satırları STATÜDEN BAĞIMSIZ sayılır.
    # Bugün bunların hepsi `MASKE_YOK` olduğu için GUARD 4'e takılıyor; maskeler
    # ileride eklenirse aşağıdaki satır-bazlı GUARD 6 onları yakalayacak. Her iki
    # durumda operatör SAYIYI GÖRÜR.
    postop_rows = [r for r in rows if classify_timepoint(r) is not None]
    postop_patients = sorted({str(r.get("patient_id", "")) for r in postop_rows})

    connection = get_connection(readonly=not args.apply)
    rows_out: list[dict[str, object]] = []
    try:
        for row in rows:
            status = row["status"]
            out = {
                "patient_id": row["patient_id"],
                "scan_id": row["scan_id"],
                "region": row["region"],
                "csv_status": status,
                "db_action": "",
            }
            if status in DATA_GAP_STATUSES:
                # Veri yokluğu -- pipeline hatası değil, ayrı sayılır.
                out["db_action"] = f"SKIPPED_VERI_YOK:{status[:80]}"
                rows_out.append(out)
                continue
            if status != "OK" and status not in BENIGN_STATUSES:
                # GUARD 4 -- gerçek çıkarım başarısızlığı. Zararsız "SKIPPED" DEĞİL.
                out["db_action"] = f"HATA_CSV_STATUS:{status[:80]}"
                rows_out.append(out)
                continue

            # GUARD 3 (Codex review 2026-08-14, HIGH) -- yazıcı `status`'e körü körüne
            # güvenmemeli. `status=OK` ama izin verilmeyen bir maske kaynağından gelen
            # satır, kanonik C32 tool adı altına GİRMEMELİ. Bu sessizce ATLANMAZ:
            # aşağıda toplanıp koşu sonunda exit 2'ye çevrilir (fail-loud).
            mask_source = (row.get("mask_source") or "").strip()
            if mask_source not in ALLOWED_MASK_SOURCES:
                out["db_action"] = f"HATA_IZINSIZ_MASK_SOURCE:{mask_source[:60]!r}"
                rows_out.append(out)
                continue

            # GUARD 6 (karar 27, 2026-09-13) -- SERT TIMEPOINT FİLTRESİ, bypass YOK.
            # Buraya gelen satır GUARD 3 ve GUARD 4'ten GEÇMİŞTİR, yani aksi halde
            # kanonik C32 vektörüne INSERT EDİLECEKTİ. Post-op fallback bir tarama
            # buradan GEÇEMEZ: `train_cox_week3.py` yalnız `segmentation_tool`'a
            # baktığı için bu satır doğrudan Cox eğitim havuzuna girer ve kilitli
            # 611/585 bozulur. Sessizce atlanmaz -- koşu sonunda exit 2.
            timepoint_problem = classify_timepoint(row)
            if timepoint_problem is not None:
                out["db_action"] = f"HATA_TIMEPOINT_POST_OP:{timepoint_problem}"
                rows_out.append(out)
                continue

            scan_id = int(row["scan_id"])
            if status == "LABEL_YOK_0_VOXEL":
                volume, surface_area, entropy, contrast = 0, 0, None, None
                shape, first_order, texture = {}, {}, {}
            else:
                volume = float(row["voxel_volume"]) if row["voxel_volume"] else None
                surface_area = float(row["surface_area"]) if row["surface_area"] else None
                entropy = float(row["entropy"]) if row["entropy"] else None
                contrast = float(row["contrast"]) if row["contrast"] else None
                shape = _sanitize_floats(_parse_json(row["shape_features_json"]))
                first_order = _sanitize_floats(_parse_json(row["first_order_features_json"]))
                texture = _sanitize_floats(_parse_json(row["texture_features_json"]))

            # F4 (kalem 5, 2026-09-13) -- LUMIERE yazıcısındaki satır-timeout +
            # retry deseni buraya TAŞINDI. Eskiden UPenn tarafında ne timeout ne
            # yeniden-bağlanma vardı: Supabase pooler'ın "server closed the
            # connection unexpectedly" hatası tüm koşuyu (rapor yazılmadan)
            # çökertebiliyor, süresiz bloke olan bir satır ise sonsuza kadar
            # asılı kalabiliyordu. `HATA_BAGLANTI` öneki de bu yüzden ÖLÜ idi.
            for attempt in (1, 2):
                try:
                    if args.apply:
                        def _do_insert_and_commit(conn=connection):
                            inserted_ = _insert_row(
                                conn,
                                scan_id=scan_id,
                                region=row["region"],
                                volume=volume,
                                surface_area=surface_area,
                                entropy=entropy,
                                contrast=contrast,
                                shape=shape,
                                first_order=first_order,
                                texture=texture,
                            )
                            conn.commit()  # satır-bazlı commit: bir hata diğer satırları bozmasın
                            return inserted_

                        inserted = _run_row_with_timeout(_do_insert_and_commit)
                        out["db_action"] = "YAZILDI" if inserted else "ATLANDI_ZATEN_VAR"
                    else:
                        out["db_action"] = "DRY_RUN"
                    break
                except concurrent.futures.TimeoutError:
                    # Asılı thread gerçekten kill EDİLEMEZ (belgeli sınır) -- eski
                    # bağlantıyı beklemeden terk et, yeni bağlantıyla bir kez daha dene.
                    try:
                        connection.close()
                    except Exception:  # noqa: BLE001
                        pass
                    if attempt == 2:
                        out["db_action"] = f"HATA_TIMEOUT:{_ROW_TIMEOUT_SECONDS}s_asildi"[:150]
                        break
                    connection = get_connection(readonly=not args.apply)
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
                    try:
                        connection.close()
                    except Exception:  # noqa: BLE001
                        pass
                    if attempt == 2:
                        out["db_action"] = f"HATA_BAGLANTI:{type(exc).__name__}:{exc}"[:150]
                        break
                    connection = get_connection(readonly=not args.apply)
                except Exception as exc:  # noqa: BLE001
                    try:
                        connection.rollback()  # bu satırın transaction'ını temizle
                    except Exception:  # noqa: BLE001
                        pass
                    out["db_action"] = f"HATA:{type(exc).__name__}:{exc}"[:150]
                    break

            rows_out.append(out)
    finally:
        try:
            connection.close()
        except Exception:  # noqa: BLE001
            pass

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["patient_id", "scan_id", "region", "csv_status", "db_action"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    from collections import Counter

    counts = Counter(r["db_action"] for r in rows_out)
    partial_run = args.limit is not None
    if partial_run:
        # F5: `--limit` + `--apply` artık reddediliyor, yani buraya yalnız dry-run
        # düşebilir -- kısmi bir YÜKLEME exit 0 ile geçemez.
        print(f"PARTIAL_RUN (yalnız dry-run): --limit={args.limit} verildi, CSV'nin yalnız ilk {args.limit} satırı işlendi.")
    print(f"{len(rows_out)} satır işlendi. Dağılım: {dict(counts)}")

    bad_timepoint = sum(v for k, v in counts.items() if k.startswith("HATA_TIMEPOINT_POST_OP"))

    # F6 -- koşunun argümanları/eşikleri rapor CSV'sinin yanına JSON olarak yazılır.
    sidecar = _write_run_args_sidecar(
        args.report,
        args,
        csv_sha256=_sha256(args.csv),
        summary={
            "rows_processed": len(rows_out),
            "db_action_counts": dict(counts),
            "postop_fallback_rows_in_csv": len(postop_rows),
            "postop_fallback_patients_in_csv": len(postop_patients),
            "guard6_rejected_rows": bad_timepoint,
        },
    )
    print(f"Rapor: {args.report}")
    print(f"Koşu argümanları (F6): {sidecar}")

    # GUARD 6 bilgilendirme bloğu -- post-op satırları HER koşuda basılır.
    print(
        f"POST-OP FALLBACK ÖZETİ (GUARD 6): kaynak CSV'de {len(postop_rows)} satır / "
        f"{len(postop_patients)} hasta `{TIMEPOINT_COLUMN}` != '{CANONICAL_TIMEPOINT}'. "
        f"Bunlardan {bad_timepoint} satır GUARD 6 tarafından REDDEDİLDİ "
        f"(geri kalanı GUARD 3/GUARD 4'e daha önce takılmış)."
    )
    if postop_patients:
        print(f"  post-op fallback hastalar: {postop_patients[:25]}{'...' if len(postop_patients) > 25 else ''}")

    # GUARD 4 sonucu -- kaynak CSV'deki gerçek çıkarım başarısızlıkları.
    csv_failures = sum(v for k, v in counts.items() if k.startswith("HATA_CSV_STATUS"))
    data_gaps = sum(v for k, v in counts.items() if k.startswith("SKIPPED_VERI_YOK"))
    if data_gaps:
        print(f"NOT: {data_gaps} satır 'veri yok' (T1CE_TARAMASI_YOK) -- pipeline hatası değil.")
    if csv_failures:
        # Operatörün karar verebilmesi için ham string yerine KATEGORİ dökümü:
        # "26 hata" tek başına eyleme dönüşmez, "16'sı görüntü yok / 6'sı boş
        # maske" dönüşür. Kategori = statünün ilk ':' öncesi öneki.
        kategori: Counter = Counter()
        etkilenen_scan: dict[str, set] = {}
        for satir in rows_out:
            eylem = str(satir["db_action"])
            if not eylem.startswith("HATA_CSV_STATUS:"):
                continue
            k = eylem[len("HATA_CSV_STATUS:"):].split(":")[0]
            kategori[k] += 1
            etkilenen_scan.setdefault(k, set()).add(satir["scan_id"])
        print("ÇIKARIM BAŞARISIZLIKLARI (kategori dökümü):")
        for k, v in kategori.most_common():
            print(f"  {k:34s} satır={v:4d}  benzersiz scan={len(etkilenen_scan[k]):4d}")

    # GUARD 6 sonucu -- kilitli 611/585 havuzunu YAPISAL olarak korur.
    # Bilinçli olarak GUARD 4'ün exit'inden ÖNCE: bu, `--accept-extraction-failures`
    # gibi bir operatör eşiğiyle KABUL EDİLEBİLİR bir durum DEĞİL.
    if bad_timepoint:
        print(
            f"HATA (GUARD 6): {bad_timepoint} satır post-op fallback taramadan geliyor "
            f"(`{TIMEPOINT_COLUMN}` != '{CANONICAL_TIMEPOINT}') ve DB'ye YAZILMADI.\n"
            f"      Bu satırlar GUARD 3 ve GUARD 4'ten GEÇTİ -- yani bu guard olmasa\n"
            f"      kanonik '{SEGMENTATION_TOOL}' vektörüne SESSİZCE girecek, oradan\n"
            f"      `train_cox_week3.py`'nin eğitim havuzuna karışacaklardı\n"
            f"      (o script yalnız `segmentation_tool`'a bakar, timepoint'e BAKMAZ).\n"
            f"      Sonuç: kilitli 611 hasta / 585 ölüm olayı sayısı BOZULURDU.\n"
            f"      BYPASS BAYRAĞI YOKTUR. Post-op radyomik gerçekten isteniyorsa yolu\n"
            f"      AYRI bir segmentation_tool adıdır (örn. '{SEGMENTATION_TOOL}-POSTOP'),\n"
            f"      bu script DEĞİL. Bkz. karar 27 / CLAUDE.md 611-585 kilidi.",
            file=sys.stderr,
        )
        return 3

    if csv_failures > args.accept_extraction_failures:
        print(
            f"HATA: kaynak CSV'de {csv_failures} çıkarım BAŞARISIZLIĞI var "
            f"(kabul edilen üst sınır: {args.accept_extraction_failures}).\n"
            f"      Bunlar zararsız 'atlandı' DEĞİL -- çıkarım o taramalarda ÇÖKTÜ.\n"
            f"      Eksik kohortu 'başarıyla yüklendi' saymamak için koşu reddedildi.\n"
            f"      Her kategorinin KAYNAK-VERİ kusuru mu (yeniden koşmakla düzelmez)\n"
            f"      yoksa pipeline hatası mı (düzeltilebilir) olduğunu belirle; kaynak-veri\n"
            f"      kusuruysa sayıyı karar dosyasına onaylı baseline olarak yaz ve geç:\n"
            f"      --accept-extraction-failures {csv_failures}",
            file=sys.stderr,
        )
        return 2

    # GUARD 3 sonucu -- izinsiz maske kaynağı bulunduysa koşu BAŞARISIZ sayılır.
    bad_mask = sum(v for k, v in counts.items() if k.startswith("HATA_IZINSIZ_MASK_SOURCE"))
    if bad_mask:
        print(
            f"HATA: {bad_mask} satır izin verilmeyen bir `mask_source` taşıyor ve DB'ye\n"
            f"      YAZILMADI. İzinli kaynaklar: {sorted(ALLOWED_MASK_SOURCES)}.\n"
            f"      CSV yanlış kohorttan gelmiş olabilir -- kaynağı doğrulanmadan\n"
            f"      bu yazım 'tamamlandı' sayılmamalı.",
            file=sys.stderr,
        )
        return 2

    # GUARD 2 -- "başarılı koşu, 0 satır" sessiz başarısızlığı, ÜÇ DURUMLU
    # (Codex review 2026-08-14, MEDIUM): eski hâli yalnız `written==0 and
    # skipped_existing>0`e bakıyordu; bu hem idempotent yeniden koşuyu hata
    # sayıyor hem de "tüm satırlar HATA aldı, hiç çakışma yok" durumunu exit 0
    # ile geçiriyordu.
    if args.apply:
        written = counts.get("YAZILDI", 0)
        skipped_existing = counts.get("ATLANDI_ZATEN_VAR", 0)
        # Yalnız INSERT SIRASINDA oluşan DB hataları (GUARD 4'ün CSV-kaynaklı
        # başarısızlıklarıyla çift sayılmasın diye kesin önek eşleşmesi).
        errors = sum(v for k, v in counts.items() if k.startswith(DB_ERROR_PREFIXES))
        attempted = written + skipped_existing + errors  # INSERT'e ulaşan satırlar

        if attempted == 0:
            print(
                "HATA: --apply verildi ama HİÇBİR satır INSERT aşamasına ULAŞMADI "
                f"(işlenen {len(rows_out)} satırın tamamı atlandı).\n"
                "      Muhtemel neden: boş/yanlış CSV veya beklenmeyen `status` değerleri.",
                file=sys.stderr,
            )
            return 2
        if errors:
            print(
                f"HATA: {errors} satır INSERT sırasında hata aldı (yazılan: {written}).\n"
                "      Rapor CSV'sindeki `db_action` kolonunda 'HATA' ile başlayan\n"
                "      satırlara bak; doğrulanmadan 'tamamlandı' sayma.",
                file=sys.stderr,
            )
            return 2
        if written == 0:
            # Tüm yazılabilir satırlar zaten DB'de -- gerçek bir yazma başarısızlığı
            # DEĞİL, idempotent no-op. İlk yüklemede bu beklenmez.
            message = (
                f"IDEMPOTENT_NOOP: yazılabilir {skipped_existing} satırın tamamı zaten "
                f"DB'de ((scan_id, '{SEGMENTATION_TOOL}', region) çakışması). Yeni satır yazılmadı."
            )
            if args.require_new_writes:
                print("HATA: " + message + "\n      --require-new-writes verildiği için başarısız sayıldı.", file=sys.stderr)
                return 2
            print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
