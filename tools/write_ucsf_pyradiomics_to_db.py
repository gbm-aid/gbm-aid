"""ucsf_pyradiomics_c32.csv (`run_pyradiomics_ucsf.py` C32 çıktısı) -> radiomics INSERT.

Kapsam: `tools/write_upenn_pyradiomics_to_db.py`'nin GUARD 1-4 desenini VE
gerçek INSERT yolunu BİREBİR takip eder
(NOT 2026-09-13: eski metin "DÖRT GUARD deseni" diyordu; UPenn'e karar 27 ile
GUARD 6 eklendiği için o ifade bayatladı -- burada kastedilen GUARD 1-4'tür.
GUARD 6 UCSF'te YOKTUR, gerekçesi aşağıda) (C32 provenance / üç-durumlu yazım
sonucu / mask_source doğrulaması / çıkarım-başarısızlığı kabul eşiği),
UCSF'e uyarlanmış.

~~**BU SCRIPT'İN GÜNCEL (2026-08-18, ikinci tur) KOŞUSUNDA `--apply`
ÇALIŞTIRILMAMIŞTIR -- yalnız dry-run gösterildi, Barış onayı BEKLENİYOR.**~~

🔴 **DÜZELTME (2026-09-13, db-agent-G, karar 24) -- ÜSTTEKİ ÇİZİLİ CÜMLE
YANLIŞTI.** `--apply` **2026-08-18 13:00'te Barış'ın onayıyla KOŞULDU** ve
`radiomics` tablosuna **1475 satır** yazıldı (295 hasta × 5 bölge).
Kanıt (üçü bağımsız):
  1. `log/2026-08-18.md` (13:00 girdisi) -- onay ve koşu kaydı.
  2. Canlı SELECT (bugün, 2026-09-13, salt-okunur):
     `SELECT count(*) FROM radiomics WHERE segmentation_tool =
     'UCSF-PDGM-PyRadiomics-107-C32'` -> **1475**; 295 benzersiz hasta,
     295 benzersiz `scan_id`.
  3. Yazım raporu CSV'si: 1475 satırın tamamı `db_action='YAZILDI'`
     (dosya adı yanıltıcı, bkz. aşağıdaki RAPOR ADI notu).
**Eski metin neden SİLİNMEDİ:** wiki hard rule #3 -- çelişki silinmez,
işaretlenir. Bu dokstring operatörün gördüğü BİRİNCİL güvenlik sinyalidir;
"--apply çalıştırılmamıştır" diye okumak, yazımın idempotent olduğunu
bilmeyen birini "bir kez daha koşsam da olur" yanılgısına götürebilirdi.

📌 **RAPOR ADI NOTU (karar 25, 2026-09-13):** gerçek yazım koşusu varsayılan
`--report` yolu DEĞİŞTİRİLMEDEN koşuldu, bu yüzden yazım kanıtı
`ucsf_c32_db_dryrun.csv` ADLI dosyaya düştü (içeriği 1475 `YAZILDI` satırı --
bir dry-run çıktısı DEĞİL) ve önceki dry-run çıktısı üzerine yazılarak
KAYBOLDU. Bugün: (a) varsayılan rapor adı `ucsf_c32_db_write_report.csv`
yapıldı, (b) `--apply` verildiğinde rapor adında `_write` ZORUNLU kılındı
(üç yazıcıda aynı kural), (c) eski dosya SİLİNMEDİ, doğru adla da kopyalandı.

GUARD 5 -- scan_id EŞLEMESİ (2026-08-18 db-agent UCSF onboarding'i UYGULADI,
bu guard artık GEÇİYOR): `dataset_sources`'ta `UCSF-PDGM` satırı, `patients`'ta
295 UCSF hastası, `mr_scans`'ta 295 T1ce taraması VAR (canlı doğrulandı).
Guard yine de KOD İÇİNDE KALIR -- ileride onboarding'in bir şekilde
bozulması/geri alınması ihtimaline karşı fail-loud kontrol, sessizce
atlanmaz.

Kalan dört guard (UPenn eşleniğiyle AYNI mantık, aynı gerekçe -- tekrar
yazılmadı, bkz. `write_upenn_pyradiomics_to_db.py` docstring'i):
GUARD 1 provenance / GUARD 2 üç-durumlu yazım sonucu / GUARD 3 mask_source
doğrulaması (`ucsf_native`) / GUARD 4 çıkarım-başarısızlığı kabul eşiği.

**LABEL_YOK_0_VOXEL POLİTİKASI -- koordinatörün "13 boş NC satırı
ATLANMALI" önerisi CANLI SORGUYLA ÇÜRÜTÜLDÜ, UYGULANMADI (2026-08-18):**
Koordinatörün gerekçesi "LUMIERE Necrosis 582/585, boş bölgeler
YAZILMAMIŞ" idi. Canlı SELECT (`radiomics` tablosu, iki bağımsız sorgu):
  - LUMIERE-PyRadiomics-107(-C32) altında Necrosis'te tumor_volume_mm3=0/
    NULL VE shape_features='{}' olan **57 satır GERÇEKTEN VAR** (DB'de
    yazılı) -- yani LABEL_YOK_0_VOXEL satırları ATLANMIYOR, TAM TERSİNE
    UPenn'le AYNI konvansiyonla (volume=0, boş JSON) INSERT ediliyor.
  - 582 (Necrosis) vs 585 (distinct scan_id) farkı, o 3 scan_id için
    Necrosis'in DB'de HİÇ satırı olmaması demek (ne OK ne LABEL_YOK) --
    bu, o 3 kayıt için kaynak CSV'de gerçek bir çıkarım BAŞARISIZLIĞI
    (HATA/MASKE_YOK/GEOMETRI_UYUSMUYOR, GUARD 4'e göre INSERT'e hiç
    girmeyen kategori) olduğunu gösterir -- "boş bölge dışlama" POLİTİKASI
    DEĞİL. UPenn'de de aynı konvansiyon doğrulandı (NC=3/ET=6/
    TC_derived=2 satır volume=0/boş JSON, 611/611 tam -- hiç eksik yok).
Sonuç: UCSF'in 13 `LABEL_YOK_0_VOXEL` (yalnız NC) satırı, kurulu
UPenn/LUMIERE konvansiyonuyla TUTARLI olarak `BENIGN_STATUSES`'ta kalır ve
INSERT edilir (volume=0, boş JSON) -- **"ATLANDI_BOS_BOLGE" davranışı
UYGULANMADI.** Barış/koordinatör farklı karar verirse `BENIGN_STATUSES`'tan
çıkarmak tek satırlık bir değişikliktir, ama bu o zaman UPenn/LUMIERE'in
MEVCUT DB verisiyle DE tutarsızlık yaratır (geriye dönük düzeltme gerekir)
-- bu ayrıca beyan edilmeli.

GUARD 6 (UPenn'in post-op timepoint sert filtresi) BU SCRIPT'TE **YOKTUR ve
GEREKMEZ** (2026-09-13, db-agent-G, karar 27) -- bu bir EKSİKLİK DEĞİL, TASARIM:
  - UCSF kohortu `tools/freeze_ucsf_cohort.py` ile hasta başına TEK preoperatif
    T1ce olacak şekilde dondurulmuştur; kaynak CSV'de `timepoint` kolonu HİÇ
    YOKTUR (ölçüldü, 2026-09-13: kolonlar patient_id/region/mask_source/status/
    voxel_volume/warning/*_json/surface_area/entropy/contrast).
  - `_resolve_scan_ids()` hasta başına tek `mr_scans` satırı çözer (295/295).
  - UCSF **HARİCİ TEST** setidir; Cox EĞİTİM havuzuna hiç girmez, dolayısıyla
    kilitli 611 hasta / 585 olay sayısı bu yoldan tehdit EDİLEMEZ.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from db_connection import get_connection  # noqa: E402

SEGMENTATION_TOOL = "UCSF-PDGM-PyRadiomics-107-C32"
DEFAULT_CSV = PROJECT_ROOT / "artifacts" / "week3" / "pyradiomics" / "ucsf_pyradiomics_c32.csv"

# --- Rapor varsayılanları: MODA GÖRE AYRI (İŞ 3 / karar 25, 2026-09-13) --------
# ESKİ HÂLİ: tek varsayılan, `ucsf_c32_db_dryrun.csv`. Gerçek `--apply` koşusu
# varsayılanı değiştirmediği için yazım kanıtı "dryrun" adlı dosyaya düştü ve
# önceki dry-run çıktısını üzerine yazdı. Tek varsayılanı `_write_report` yapmak
# hatayı TERS YÖNE çevirirdi (bir dry-run yazım kanıtını silerdi) -- bu yüzden
# varsayılan MODA GÖRE ayrıldı. İki dosya hiçbir zaman aynı olamaz.
_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "week3" / "pyradiomics"
DEFAULT_WRITE_REPORT = _ARTIFACT_DIR / "ucsf_c32_db_write_report.csv"
# NOT: `ucsf_c32_db_dryrun.csv` ADI BİLİNÇLİ OLARAK KULLANILMIYOR -- o dosya
# 2026-08-18'in GERÇEK YAZIM kanıtını taşıyor (1475 `YAZILDI` satırı) ve
# tarihsel kayıt olarak SİLİNMEYECEK/ÜZERİNE YAZILMAYACAK.
DEFAULT_DRYRUN_REPORT = _ARTIFACT_DIR / "ucsf_c32_db_dryrun_latest.csv"

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
REQUIRED_PROVENANCE_FIELDS = (
    "extractor_settings_sha256",
    "zscore_stats_sha256",
    "generator_script_sha256",
    "cohort",
)

ALLOWED_MASK_SOURCES = frozenset({"ucsf_native"})

BENIGN_STATUSES = frozenset({"LABEL_YOK_0_VOXEL"})
DATA_GAP_STATUSES = frozenset(set())  # UCSF'in "veri yok" kavramı yok -- kohort zaten disk-tam filtrelendi
# `HATA_TIMEOUT` (2026-09-13): UPenn/LUMIERE yazıcılarıyla tuple'ı HİZALI tutmak
# için eklendi. Bu script'te satır-timeout deseni YOK, yani önek bugün
# ÜRETİLEMEZ -- ileride retry deseni buraya da taşınırsa GUARD 2'nin sessizce
# fail-open olmaması için önceden hizalandı (LUMIERE'de tam bu eksiklik vardı).
DB_ERROR_PREFIXES = ("HATA:", "HATA_BAGLANTI", "HATA_TIMEOUT")

# GUARD 6 (timepoint sert filtresi) BU SCRIPT'TE YOKTUR ve GEREKMEZ
# (2026-09-13, db-agent-G, karar 27): UCSF kohortu hasta başına TEK preoperatif
# T1ce ile dondurulmuştur (`freeze_ucsf_cohort.py`), kaynak CSV'de `timepoint`
# kolonu HİÇ YOKTUR (ölçüldü: kolonlar patient_id/region/mask_source/status/...)
# ve `_resolve_scan_ids()` hasta başına tek `mr_scans` satırı çözer. Ayrıca UCSF
# HARİCİ TEST setidir, Cox EĞİTİM havuzuna hiç girmez -- 611/585 kilidi bu yoldan
# tehdit edilemez. Bu not, "UPenn'de var, burada yok" farkının EKSİKLİK değil
# TASARIM olduğunu kayda geçirir.


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.name + PROVENANCE_SUFFIX)


def validate_c32_provenance(csv_path: Path, *, require_complete: bool = False) -> list[str]:
    manifest_path = provenance_path(csv_path)
    if not manifest_path.exists():
        return [f"provenance manifest'i YOK: {manifest_path.name}."]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [f"provenance manifest'i okunamadı/bozuk: {type(exc).__name__}: {exc}"]
    if not isinstance(manifest, dict):
        return [f"provenance manifest'i JSON nesnesi değil: {type(manifest).__name__}"]

    problems: list[str] = []
    for key, expected in EXPECTED_PROVENANCE.items():
        actual = manifest.get(key)
        if actual != expected:
            problems.append(f"{key}: beklenen {expected!r}, bulunan {actual!r}")
    for key in REQUIRED_PROVENANCE_FIELDS:
        if not manifest.get(key):
            problems.append(f"`{key}` alanı yok/boş -- manifest gerçek koşuya bağlanmamış olabilir")

    stats_n = manifest.get("zscore_stats_image_count")
    expected_n = manifest.get("expected_image_count")
    if type(stats_n) is not int or type(expected_n) is not int or stats_n <= 0 or expected_n <= 0:
        problems.append(
            "`zscore_stats_image_count` / `expected_image_count` POZİTİF TAMSAYI değil "
            f"(bulunan: {stats_n!r} / {expected_n!r})"
        )
    elif stats_n != expected_n:
        problems.append(
            f"Z-score istatistiği tam kohorttan fit EDİLMEMİŞ: "
            f"zscore_stats_image_count={stats_n} != expected_image_count={expected_n}."
        )

    if require_complete and manifest.get("run_complete") is not True:
        problems.append(f"`run_complete` true DEĞİL (bulunan: {manifest.get('run_complete')!r}).")

    recorded_hash = manifest.get("report_sha256")
    if not recorded_hash:
        problems.append("manifest'te `report_sha256` alanı yok")
    elif recorded_hash != _sha256(csv_path):
        problems.append("CSV içeriği manifest'le EŞLEŞMİYOR (sha256 uyuşmuyor)")
    return problems


# --- Rapor dosya adı disiplini (İŞ 3 / karar 25, 2026-09-13) -------------------
# Bu olayın KAYNAĞI tam olarak bu script'tir: gerçek yazım koşusu varsayılan
# `--report` yolunu değiştirmediği için tek yazım kanıtı `ucsf_c32_db_dryrun.csv`
# adlı dosyada kaldı ve sonraki bir dry-run onu üzerine yazsa KANIT KAYBOLACAKTI.
WRITE_REPORT_MARKER = "_write"


def validate_limit_and_apply(limit, apply: bool) -> list[str]:
    """F5 (kalem 5): `--limit` + `--apply` yasağı. Boş liste = temiz.

    Ayrı fonksiyon olması bilinçli: DB'ye hiç bağlanmadan birim testle
    doğrulanabilsin (tests/test_c32_writer_guards.py).
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
    (özellikle `--accept-extraction-failures` ve `--skip-scan-id-gate` ile)
    koşulduğu HİÇBİR yere yazılmıyordu. CSV'ye başlık satırı eklemek DictReader
    tüketicilerini bozacağı için yan dosya (sidecar) seçildi.
    """
    import datetime as _dt

    sidecar = report_path.with_name(report_path.name + ".run_args.json")
    payload = {
        "script": Path(__file__).name,
        "segmentation_tool": SEGMENTATION_TOOL,
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
            "skip_scan_id_gate": bool(args.skip_scan_id_gate),
        },
        "thresholds": {"accept_extraction_failures": args.accept_extraction_failures},
        "source_csv_sha256": csv_sha256,
        "summary": summary,
    }
    sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return sidecar


def _parse_json(raw: str) -> dict:
    if not raw:
        return {}
    return json.loads(raw)


def _sanitize_floats(data: dict) -> dict:
    """NaN/Infinity değerlerini None'a çevir -- json.dumps bunları literal NaN/Infinity
    yazar, bu geçerli JSON değil, Postgres jsonb reddeder (InvalidTextRepresentation).
    `write_upenn_pyradiomics_to_db.py`'nin AYNI fonksiyonu, birebir kopya."""
    import math

    clean = {}
    for key, value in data.items():
        if isinstance(value, float) and not math.isfinite(value):
            clean[key] = None
        else:
            clean[key] = value
    return clean


def _insert_row(
    connection,
    *,
    scan_id: int,
    region: str,
    volume,
    surface_area,
    entropy,
    contrast,
    shape,
    first_order,
    texture,
) -> bool:
    """INSERT dener, gerçekten yeni bir satır yazıldıysa True döner.

    `write_upenn_pyradiomics_to_db.py::_insert_row()` ile BİREBİR AYNI --
    yalnız `SEGMENTATION_TOOL` bu modülün UCSF sabitine bağlı.
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


def _resolve_scan_ids(connection, patient_ids: set[str]) -> dict[str, int]:
    """UCSF `patient_id` (CSV'deki, örn. 'UCSF-PDGM-0115') -> `mr_scans.scan_id`.

    GUARD 5 -- 2026-08-18'de canlı doğrulandı: `patients` tablosunda UCSF
    kaynaklı 0 satır var, bu yüzden bu sorgu BEKLENEN OLARAK boş sözlük
    döndürür. Sessizce "0 eşleşme, devam" DENMEZ -- `main()` bunu ayrı bir
    fail-loud guard olarak ele alır.
    """

    if not patient_ids:
        return {}
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT ms.patient_id, ms.scan_id
            FROM mr_scans ms
            JOIN patients p ON p.patient_id = ms.patient_id
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE ds.source_name = 'UCSF-PDGM' AND ms.modality = 'T1ce'
              AND ms.patient_id = ANY(%s)
            """,
            (list(patient_ids),),
        )
        return {row[0]: row[1] for row in cursor.fetchall()}
    finally:
        cursor.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            f"ucsf_pyradiomics_c32.csv'yi radiomics tablosuna INSERT et "
            f"(segmentation_tool='{SEGMENTATION_TOOL}'). "
            "DURUM (2026-09-13 düzeltmesi, karar 24): bu yükleme 2026-08-18 13:00'te "
            "Barış onayıyla ZATEN YAPILDI -- canlı DB'de 1475 satır var. Script "
            "idempotenttir (ON CONFLICT DO NOTHING), yeniden --apply IDEMPOTENT_NOOP "
            "verir. Eski 'yalnız dry-run, --apply ÇALIŞTIRILMAMALI' metni YANLIŞTI."
        )
    )
    parser.add_argument("--apply", action="store_true", help="Belirtilmezse DB'ye yazılmaz (dry-run varsayılan).")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,  # MODA GÖRE çözülür, bkz. DEFAULT_*_REPORT ve aşağısı
        help="Verilmezse moda göre seçilir: --apply ile DEFAULT_WRITE_REPORT, "
        "dry-run'da DEFAULT_DRYRUN_REPORT (karar 25 -- tek bir varsayılan iki modda "
        "paylaşıldığı için yazım kanıtı üzerine yazılıyordu).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="SMOKE TEST için ilk N satırla sınırla. F5 (2026-09-13): --apply ile "
        "BİRLİKTE KULLANILAMAZ -- kısmi yükleme exit 0 ile 'başarılı' görünürdü.",
    )
    parser.add_argument("--skip-provenance-check", action="store_true")
    parser.add_argument("--require-new-writes", action="store_true")
    parser.add_argument("--accept-extraction-failures", type=int, default=0)
    parser.add_argument(
        "--skip-scan-id-gate",
        action="store_true",
        help=(
            "GUARD 5'i bilinçli atla -- YALNIZ UCSF onboarding'i (dataset_sources/"
            "patients/mr_scans) tamamlandıktan SONRA, DB'de gerçek scan_id eşlemesi "
            "varken kullanılmalı. Bugünkü koşuda kullanılmamalı (eşleme 0)."
        ),
    )
    args = parser.parse_args()

    # Rapor varsayılanını MODA GÖRE çöz (karar 25) -- bkz. DEFAULT_*_REPORT notu.
    if args.report is None:
        args.report = DEFAULT_WRITE_REPORT if args.apply else DEFAULT_DRYRUN_REPORT

    if args.skip_provenance_check and args.apply:
        print(
            "HATA: --skip-provenance-check ile --apply BİRLİKTE kullanılamaz.",
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
            "Ayrı bir '_dryrun' adı kullanmayı düşün (örn. --report ucsf_c32_db_dryrun_v2.csv).",
            file=sys.stderr,
        )

    if not args.csv.exists():
        print(f"HATA: --csv dosyası bulunamadı: {args.csv}", file=sys.stderr)
        return 2

    if "c32" not in args.csv.name.lower():
        print(f"UYARI: --csv dosya adında 'c32' geçmiyor: {args.csv.name}", file=sys.stderr)
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
                + "\n  - ".join(provenance_problems),
                file=sys.stderr,
            )
            return 2

    with args.csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]

    connection = get_connection(readonly=not args.apply)
    try:
        # GUARD 5 -- scan_id eşlemesi (2026-08-18 bulgusu: bu BOŞ dönecek).
        patient_ids = {row["patient_id"] for row in rows if row.get("patient_id")}
        scan_id_by_patient = _resolve_scan_ids(connection, patient_ids)
        unresolved = patient_ids - set(scan_id_by_patient)

        if unresolved and not args.skip_scan_id_gate:
            print(
                f"HATA (GUARD 5): {len(unresolved)}/{len(patient_ids)} UCSF hastası için "
                "`mr_scans.scan_id` eşlemesi YOK (dataset_sources/patients/mr_scans'ta "
                "UCSF-PDGM henüz onboard EDİLMEMİŞ -- 2026-08-18'de canlı doğrulandı: "
                "dataset_sources={TCGA-GBM,UPenn-GBM,LUMIERE,TCGA-Omics}, UCSF YOK).\n"
                "      Bu YAPISAL bir blokaj -- CSV/kod hatası DEĞİL. `radiomics.scan_id` "
                "NOT NULL olduğu için hiçbir satır INSERT EDİLEMEZ.\n"
                "      Önce (ayrı, onaylı bir şema/veri görevi): dataset_sources'a "
                "UCSF-PDGM eklenmeli, 295 hastanın patients+mr_scans satırları "
                "yazılmalı. Bu script o iş bitmeden --apply İLE ÇALIŞTIRILAMAZ.\n"
                "      Bilinçli atlamak (yalnız onboarding SONRASI) için: "
                "--skip-scan-id-gate",
                file=sys.stderr,
            )
            return 2

        rows_out: list[dict[str, object]] = []
        for row in rows:
            status = row["status"]
            patient_id = row["patient_id"]
            out = {
                "patient_id": patient_id,
                "scan_id": "",
                "region": row["region"],
                "csv_status": status,
                "db_action": "",
            }
            if status in DATA_GAP_STATUSES:
                out["db_action"] = f"SKIPPED_VERI_YOK:{status[:80]}"
                rows_out.append(out)
                continue
            if status != "OK" and status not in BENIGN_STATUSES:
                # GUARD 4 -- gerçek çıkarım başarısızlığı (zararsız "atlandı" DEĞİL).
                out["db_action"] = f"HATA_CSV_STATUS:{status[:80]}"
                rows_out.append(out)
                continue

            mask_source = (row.get("mask_source") or "").strip()
            if mask_source not in ALLOWED_MASK_SOURCES:
                # GUARD 3 -- izin verilmeyen maske kaynağı.
                out["db_action"] = f"HATA_IZINSIZ_MASK_SOURCE:{mask_source[:60]!r}"
                rows_out.append(out)
                continue

            scan_id = scan_id_by_patient.get(patient_id)
            if scan_id is None:
                out["db_action"] = "HATA_SCAN_ID_YOK"
                rows_out.append(out)
                continue
            out["scan_id"] = scan_id

            # LABEL_YOK_0_VOXEL POLİTİKASI (bkz. modül-üstü docstring) --
            # UPenn/LUMIERE ile TUTARLI: volume=0, boş JSON, yine de INSERT.
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

            try:
                if args.apply:
                    inserted = _insert_row(
                        connection,
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
                    connection.commit()  # satır-bazlı commit: bir hata diğer satırları bozmasın
                    out["db_action"] = "YAZILDI" if inserted else "ATLANDI_ZATEN_VAR"
                else:
                    out["db_action"] = "DRY_RUN"
            except Exception as exc:  # noqa: BLE001
                connection.rollback()  # bu satırın transaction'ını temizle, sonraki satırlar etkilenmesin
                out["db_action"] = f"HATA:{type(exc).__name__}:{exc}"[:150]

            rows_out.append(out)

        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["patient_id", "scan_id", "region", "csv_status", "db_action"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)

        counts = Counter(r["db_action"] for r in rows_out)
        mode = "APPLY" if args.apply else "DRY-RUN"
        if args.limit is not None:
            print(f"PARTIAL_RUN (yalnız dry-run): --limit={args.limit} verildi.")
        print(f"{len(rows_out)} satır işlendi ({mode}). Dağılım: {dict(counts)}")

        # F6 -- koşunun argümanları/eşikleri rapor CSV'sinin yanına JSON olarak yazılır.
        sidecar = _write_run_args_sidecar(
            args.report,
            args,
            csv_sha256=_sha256(args.csv),
            summary={
                "rows_processed": len(rows_out),
                "db_action_counts": dict(counts),
                "scan_ids_resolved": len(scan_id_by_patient),
                "scan_ids_unresolved": len(unresolved),
            },
        )
        print(f"Rapor: {args.report}")
        print(f"Koşu argümanları (F6): {sidecar}")
        print(
            f"scan_id eşlemesi: {len(scan_id_by_patient)}/{len(patient_ids)} hasta "
            f"çözüldü, {len(unresolved)} çözülemedi."
        )

        # Bölge kırılımı -- yalnız bilgi amaçlı (DRY_RUN/YAZILDI/ATLANDI_ZATEN_VAR olan satırlar).
        writable_actions = {"DRY_RUN", "YAZILDI", "ATLANDI_ZATEN_VAR"}
        region_breakdown: Counter = Counter(
            r["region"] for r in rows_out if r["db_action"] in writable_actions
        )
        if region_breakdown:
            print(f"Bölge kırılımı (yazılabilir satırlar): {dict(region_breakdown)}")

        csv_failures = sum(v for k, v in counts.items() if k.startswith("HATA_CSV_STATUS"))
        if csv_failures:
            print(f"UYARI: kaynak CSV'de {csv_failures} çıkarım başarısızlığı (HATA_CSV_STATUS).")
        if csv_failures > args.accept_extraction_failures:
            print(
                f"HATA: kaynak CSV'de {csv_failures} çıkarım BAŞARISIZLIĞI var "
                f"(kabul edilen üst sınır: {args.accept_extraction_failures}).\n"
                f"      Bunlar zararsız 'atlandı' DEĞİL. Bilinçli kabul için:\n"
                f"      --accept-extraction-failures {csv_failures}",
                file=sys.stderr,
            )
            return 2

        bad_mask = sum(v for k, v in counts.items() if k.startswith("HATA_IZINSIZ_MASK_SOURCE"))
        if bad_mask:
            print(
                f"HATA: {bad_mask} satır izin verilmeyen bir `mask_source` taşıyor ve DB'ye "
                f"YAZILMADI. İzinli kaynaklar: {sorted(ALLOWED_MASK_SOURCES)}.",
                file=sys.stderr,
            )
            return 2

        # GUARD 2 -- "başarılı koşu, 0 satır" sessiz başarısızlığı, ÜÇ DURUMLU
        # (write_upenn_pyradiomics_to_db.py ile AYNI mantık).
        if args.apply:
            written = counts.get("YAZILDI", 0)
            skipped_existing = counts.get("ATLANDI_ZATEN_VAR", 0)
            errors = sum(v for k, v in counts.items() if k.startswith(DB_ERROR_PREFIXES))
            attempted = written + skipped_existing + errors

            if attempted == 0:
                print(
                    "HATA: --apply verildi ama HİÇBİR satır INSERT aşamasına ULAŞMADI "
                    f"(işlenen {len(rows_out)} satırın tamamı atlandı).",
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
                message = (
                    f"IDEMPOTENT_NOOP: yazılabilir {skipped_existing} satırın tamamı zaten "
                    f"DB'de ((scan_id, '{SEGMENTATION_TOOL}', region) çakışması). Yeni satır yazılmadı."
                )
                if args.require_new_writes:
                    print("HATA: " + message + "\n      --require-new-writes verildiği için başarısız sayıldı.", file=sys.stderr)
                    return 2
                print(message)
    finally:
        connection.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
