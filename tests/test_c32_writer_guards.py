"""C32 `radiomics` yazıcılarının guard'ları için BİRİM testleri.

Kapsam: `tools/write_upenn_pyradiomics_to_db.py`,
`tools/write_lumiere_pyradiomics_to_db.py`,
`tools/write_ucsf_pyradiomics_to_db.py`.

NEDEN BU DOSYA VAR (karar 26, 2026-09-13, db-agent-G):
Bu üç yazıcının DÖRT guard'ının (+ bugün eklenen GUARD 6'nın) TEK kanıtı
2026-08-14/2026-08-18'deki canlı koşulardı -- `tests/` altında bu dosyalar için
SIFIR test vardı. Yani bir refactor guard'lardan birini sessizce devre dışı
bıraksa hiçbir şey ötmezdi. Özellikle tehlikeli olan, guard'ların
**fail-open**a dönmesi: örneğin `zscore_stats_image_count` kontrolünde
`isinstance(x, int)` kullanılsa `true` (bool, int'in ALT SINIFI) geçerli bir
sayaç sayılırdı.

🔴 TESTLER DB'YE BAĞLANMAZ. Yalnız `tmp_path` altındaki geçici dosyalar ve saf
fonksiyonlar kullanılır; `main()` HİÇ çağrılmaz, `--apply` HİÇ verilmez,
`get_connection()` HİÇ çalıştırılmaz. (Guard'ların `main()` içindeki
bağlanışları F5/İŞ-3 için saf fonksiyonlara çıkarıldı: `validate_limit_and_apply`,
`validate_report_path` -- tam olarak testin DB'ye dokunmaması için.)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import write_lumiere_pyradiomics_to_db as lumiere_writer  # noqa: E402
from tools import write_ucsf_pyradiomics_to_db as ucsf_writer  # noqa: E402
from tools import write_upenn_pyradiomics_to_db as upenn_writer  # noqa: E402

ALL_WRITERS = [
    pytest.param(upenn_writer, id="upenn"),
    pytest.param(lumiere_writer, id="lumiere"),
    pytest.param(ucsf_writer, id="ucsf"),
]
LEGACY_WRITERS = [pytest.param(upenn_writer, id="upenn"), pytest.param(lumiere_writer, id="lumiere")]


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def _make_csv(tmp_path: Path, name: str = "fake_pyradiomics_c32.csv") -> Path:
    """İçeriği önemsiz, ama hash'i gerçek olan bir CSV."""
    csv_path = tmp_path / name
    csv_path.write_text("patient_id,region,status\nX-1,WT_derived,OK\n", encoding="utf-8")
    return csv_path


def _clean_manifest(writer, csv_path: Path) -> dict:
    """İlgili yazıcının GEÇMESİ gereken, tam/temiz bir provenance manifest'i."""
    manifest = dict(writer.EXPECTED_PROVENANCE)
    for key in writer.REQUIRED_PROVENANCE_FIELDS:
        manifest[key] = f"dummy-{key}"
    manifest["zscore_stats_image_count"] = 42
    manifest["expected_image_count"] = 42
    manifest["run_complete"] = True
    manifest["report_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    return manifest


def _write_manifest(writer, csv_path: Path, manifest) -> Path:
    path = writer.provenance_path(csv_path)
    if isinstance(manifest, str):
        path.write_text(manifest, encoding="utf-8")
    else:
        path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# GUARD 1 -- validate_c32_provenance()
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_provenance_clean_manifest_passes(writer, tmp_path):
    """Kontrol grubu: temiz manifest HİÇ sorun üretmemeli (aksi halde aşağıdaki
    kırmızı testler 'her şey başarısız' olduğu için yanlış yere güven verirdi)."""
    csv_path = _make_csv(tmp_path)
    _write_manifest(writer, csv_path, _clean_manifest(writer, csv_path))
    assert writer.validate_c32_provenance(csv_path, require_complete=True) == []


@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_provenance_missing_manifest_is_rejected(writer, tmp_path):
    csv_path = _make_csv(tmp_path)  # manifest YAZILMADI
    problems = writer.validate_c32_provenance(csv_path)
    assert len(problems) == 1
    assert "provenance manifest'i YOK" in problems[0]


@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_provenance_corrupt_json_is_rejected(writer, tmp_path):
    csv_path = _make_csv(tmp_path)
    _write_manifest(writer, csv_path, "{ bu gecerli JSON degil")
    problems = writer.validate_c32_provenance(csv_path)
    assert len(problems) == 1
    assert "okunamadı/bozuk" in problems[0]


@pytest.mark.parametrize("writer", ALL_WRITERS)
@pytest.mark.parametrize("payload", ["[1, 2, 3]", '"bir string"', "null", "17"])
def test_provenance_json_not_object_is_rejected(writer, tmp_path, payload):
    """Geçerli JSON ama nesne DEĞİL -- kontrolsüz AttributeError yerine açıklayıcı red."""
    csv_path = _make_csv(tmp_path)
    _write_manifest(writer, csv_path, payload)
    problems = writer.validate_c32_provenance(csv_path)
    assert len(problems) == 1
    assert "JSON nesnesi değil" in problems[0]


@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_provenance_wrong_bin_count_is_rejected(writer, tmp_path):
    """C32 sözleşmesinin kalbi: binCount=32. binWidth dönemine geri kayış yakalanmalı."""
    csv_path = _make_csv(tmp_path)
    manifest = _clean_manifest(writer, csv_path)
    manifest["bin_count"] = 25
    _write_manifest(writer, csv_path, manifest)
    problems = writer.validate_c32_provenance(csv_path)
    assert any("bin_count" in p and "32" in p for p in problems), problems


@pytest.mark.parametrize("writer", ALL_WRITERS)
@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("extraction_contract", "A"),
        ("normalize", True),
        ("n4_applied", False),
        ("zscore_scope", "source_pooled"),
        ("pyradiomics_version", "3.1.0"),
        ("image_types", ["Original", "Wavelet"]),
    ],
)
def test_provenance_each_contract_field_is_gated(writer, tmp_path, key, bad_value):
    """C32 sözleşmesinin HER alanı gerçekten kapı -- yalnız bin_count değil."""
    csv_path = _make_csv(tmp_path)
    manifest = _clean_manifest(writer, csv_path)
    manifest[key] = bad_value
    _write_manifest(writer, csv_path, manifest)
    problems = writer.validate_c32_provenance(csv_path)
    assert any(p.startswith(f"{key}:") for p in problems), problems


@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_provenance_sha256_mismatch_is_rejected(writer, tmp_path):
    """Eski bir CSV'yi C32 adıyla yeniden adlandırmak YETMEZ."""
    csv_path = _make_csv(tmp_path)
    manifest = _clean_manifest(writer, csv_path)
    manifest["report_sha256"] = "0" * 64
    _write_manifest(writer, csv_path, manifest)
    problems = writer.validate_c32_provenance(csv_path)
    assert any("sha256 uyuşmuyor" in p for p in problems), problems


@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_provenance_sha256_mismatch_after_csv_edited(writer, tmp_path):
    """Manifest yazıldıktan SONRA CSV değiştirilirse de yakalanmalı."""
    csv_path = _make_csv(tmp_path)
    _write_manifest(writer, csv_path, _clean_manifest(writer, csv_path))
    csv_path.write_text("patient_id,region,status\nX-1,WT_derived,OK\nX-2,ET,OK\n", encoding="utf-8")
    problems = writer.validate_c32_provenance(csv_path)
    assert any("sha256 uyuşmuyor" in p for p in problems), problems


@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_provenance_missing_report_sha256_is_rejected(writer, tmp_path):
    csv_path = _make_csv(tmp_path)
    manifest = _clean_manifest(writer, csv_path)
    del manifest["report_sha256"]
    _write_manifest(writer, csv_path, manifest)
    problems = writer.validate_c32_provenance(csv_path)
    assert any("report_sha256" in p for p in problems), problems


@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_provenance_missing_runtime_field_is_rejected(writer, tmp_path):
    """Elle uydurulmuş manifest belirtisi: runtime bağlarının boş olması."""
    csv_path = _make_csv(tmp_path)
    for key in writer.REQUIRED_PROVENANCE_FIELDS:
        manifest = _clean_manifest(writer, csv_path)
        manifest[key] = ""
        _write_manifest(writer, csv_path, manifest)
        problems = writer.validate_c32_provenance(csv_path)
        assert any(f"`{key}` alanı yok/boş" in p for p in problems), (key, problems)


@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_provenance_run_complete_false_blocks_apply_only(writer, tmp_path):
    """`run_complete=false`: --apply'da RED, dry-run'da (inceleme) SERBEST."""
    csv_path = _make_csv(tmp_path)
    manifest = _clean_manifest(writer, csv_path)
    manifest["run_complete"] = False
    _write_manifest(writer, csv_path, manifest)

    apply_problems = writer.validate_c32_provenance(csv_path, require_complete=True)
    assert any("run_complete" in p for p in apply_problems), apply_problems

    assert writer.validate_c32_provenance(csv_path, require_complete=False) == []


@pytest.mark.parametrize("writer", ALL_WRITERS)
@pytest.mark.parametrize("truthy", ["true", 1, "42"])
def test_provenance_run_complete_must_be_literal_true(writer, tmp_path, truthy):
    """`is not True` kontrolü: 'true' (string) veya 1 kabul EDİLMEMELİ."""
    csv_path = _make_csv(tmp_path)
    manifest = _clean_manifest(writer, csv_path)
    manifest["run_complete"] = truthy
    _write_manifest(writer, csv_path, manifest)
    problems = writer.validate_c32_provenance(csv_path, require_complete=True)
    assert any("run_complete" in p for p in problems), (truthy, problems)


@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_provenance_image_count_mismatch_is_rejected(writer, tmp_path):
    """Z-score istatistiği alt-kümeden fit edilmişse DB'ye giremez."""
    csv_path = _make_csv(tmp_path)
    manifest = _clean_manifest(writer, csv_path)
    manifest["zscore_stats_image_count"] = 100
    manifest["expected_image_count"] = 295
    _write_manifest(writer, csv_path, manifest)
    problems = writer.validate_c32_provenance(csv_path)
    assert any("tam kohorttan fit EDİLMEMİŞ" in p for p in problems), problems
    assert any("100" in p and "295" in p for p in problems), problems


# --------------------------------------------------------------------------- #
# 🔴 BOOL FAIL-OPEN REGRESYONU (Codex A3) -- en kritik birim test
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("writer", ALL_WRITERS)
@pytest.mark.parametrize(
    "stats_n,expected_n",
    [
        (True, True),      # ikisi de bool
        (True, 42),        # yalnız sayaç bool
        (42, True),        # yalnız beklenen bool
        (False, False),    # bool False (int olarak 0)
    ],
)
def test_provenance_bool_image_count_is_rejected_fail_closed(writer, tmp_path, stats_n, expected_n):
    """`isinstance(x, int)` KULLANILMADIĞINI kanıtlar (`type(x) is int` kullanılıyor).

    Python'da `bool`, `int`in ALT SINIFIDIR: `isinstance(True, int) is True`.
    Eğer kontrol `isinstance` ile yazılmış olsaydı, manifest'te
    `"zscore_stats_image_count": true` yazması yeterdi ve `True == True`
    olduğundan eşitlik kontrolünü de GEÇERDİ -- yani "Z-score tam kohorttan fit
    edildi" guard'ı TAMAMEN fail-open olurdu. Bu test o regresyonu kilitler.
    """
    # Önce öncülü ispatla: bool gerçekten int alt sınıfı, yani naif kontrol geçer.
    assert isinstance(True, int) is True
    assert (True == True) is True  # noqa: E712  -- eşitlik kontrolü de geçerdi

    csv_path = _make_csv(tmp_path)
    manifest = _clean_manifest(writer, csv_path)
    manifest["zscore_stats_image_count"] = stats_n
    manifest["expected_image_count"] = expected_n
    _write_manifest(writer, csv_path, manifest)

    problems = writer.validate_c32_provenance(csv_path)
    assert any("POZİTİF TAMSAYI değil" in p for p in problems), (stats_n, expected_n, problems)


@pytest.mark.parametrize("writer", ALL_WRITERS)
@pytest.mark.parametrize(
    "stats_n,expected_n",
    [(0, 0), (-1, -1), ("42", "42"), (42.0, 42.0), (None, None), (42, None)],
)
def test_provenance_non_positive_or_non_int_image_count_is_rejected(writer, tmp_path, stats_n, expected_n):
    """Sıfır/negatif/string/float/None sayaçlar da reddedilir (fail-closed)."""
    csv_path = _make_csv(tmp_path)
    manifest = _clean_manifest(writer, csv_path)
    manifest["zscore_stats_image_count"] = stats_n
    manifest["expected_image_count"] = expected_n
    _write_manifest(writer, csv_path, manifest)
    problems = writer.validate_c32_provenance(csv_path)
    assert any("POZİTİF TAMSAYI değil" in p for p in problems), (stats_n, expected_n, problems)


# --------------------------------------------------------------------------- #
# GUARD 6 -- UPenn post-op timepoint sert filtresi (karar 27)
# --------------------------------------------------------------------------- #
def test_upenn_timepoint_column_missing_is_rejected():
    """Kolon YOKSA fail-CLOSED: post-op satır var mı DOĞRULANAMAZ, koşu reddedilir."""
    fieldnames = ["patient_id", "scan_id", "region", "mask_source", "status"]
    problems = upenn_writer.validate_timepoint_column(fieldnames)
    assert len(problems) == 1
    assert "timepoint_used` kolonu YOK" in problems[0]
    assert "611" in problems[0]


@pytest.mark.parametrize("fieldnames", [None, []])
def test_upenn_timepoint_column_empty_header_is_rejected(fieldnames):
    """Boş/None başlık (boş CSV) da geçmemeli."""
    assert upenn_writer.validate_timepoint_column(fieldnames) != []


def test_upenn_timepoint_column_present_passes():
    fieldnames = ["patient_id", "scan_id", "timepoint_used", "region", "status"]
    assert upenn_writer.validate_timepoint_column(fieldnames) == []


@pytest.mark.parametrize("value", ["pre-treatment", "  pre-treatment  "])
def test_upenn_classify_timepoint_accepts_pre_treatment(value):
    assert upenn_writer.classify_timepoint({"timepoint_used": value}) is None


def test_upenn_classify_timepoint_rejects_postop_fallback():
    """Karar 27'nin tam senaryosu: `_21` maskeleri NAS'a eklenirse gelecek değer."""
    reason = upenn_writer.classify_timepoint(
        {"timepoint_used": "post-op_fallback_no_pretreatment"}
    )
    assert reason == "post-op_fallback_no_pretreatment"


@pytest.mark.parametrize("value", ["post-op", "Pre-Treatment", "pretreatment", "PRE-TREATMENT"])
def test_upenn_classify_timepoint_is_strict_not_fuzzy(value):
    """Büyük/küçük harf veya tire varyantı SESSİZCE kabul edilmez."""
    assert upenn_writer.classify_timepoint({"timepoint_used": value}) is not None


@pytest.mark.parametrize("row", [{"timepoint_used": ""}, {"timepoint_used": "   "}, {}, {"timepoint_used": None}])
def test_upenn_classify_timepoint_empty_is_rejected(row):
    """'Bilinmiyor' pre-treatment SAYILMAZ (fail-closed)."""
    assert upenn_writer.classify_timepoint(row) == "BOS_VEYA_EKSIK"


def test_guard6_is_upenn_only_by_design():
    """LUMIERE/UCSF'te GUARD 6 YOKTUR ve bu bir EKSİKLİK DEĞİL, TASARIM.

    LUMIERE longitudinal (çok-vizitli) olmak ZORUNDA (büyüme simülasyonu + T3
    RANO + FAISS) ve Cox eğitimine hiç girmez; UCSF hasta başına tek preop
    taramayla dondurulmuş HARİCİ TEST setidir. Bu test, gerekçenin dokstring'den
    sessizce silinmesini engeller.
    """
    assert not hasattr(lumiere_writer, "classify_timepoint")
    assert not hasattr(ucsf_writer, "classify_timepoint")
    for writer in (lumiere_writer, ucsf_writer):
        doc = writer.__doc__ or ""
        assert "GUARD 6" in doc, f"{writer.__name__} dokstring'inde GUARD 6 gerekçesi YOK"


# --------------------------------------------------------------------------- #
# F5 -- --limit + --apply reddi
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_limit_with_apply_is_rejected(writer):
    problems = writer.validate_limit_and_apply(50, True)
    assert len(problems) == 1
    assert "--limit ile --apply" in problems[0]


@pytest.mark.parametrize("writer", ALL_WRITERS)
@pytest.mark.parametrize("limit,apply", [(50, False), (None, True), (None, False), (0, True)])
def test_limit_apply_allowed_combinations(writer, limit, apply):
    """`--limit 0` bilinçli olarak `None` DEĞİL: argparse 0 verirse de reddedilir."""
    expected_clean = not (limit is not None and apply)
    assert (writer.validate_limit_and_apply(limit, apply) == []) is expected_clean


# --------------------------------------------------------------------------- #
# İŞ 3 / karar 25 -- rapor dosya adı disiplini
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_apply_with_dryrun_report_name_is_rejected(writer):
    """2026-08-18 UCSF olayının birebir senaryosu."""
    problems = writer.validate_report_path(Path("ucsf_c32_db_dryrun.csv"), apply=True)
    assert len(problems) == 1
    assert "_write" in problems[0]
    assert "KANIT KAYBOLUR" in problems[0]


@pytest.mark.parametrize("writer", ALL_WRITERS)
@pytest.mark.parametrize(
    "name",
    ["upenn_c32_db_write_report.csv", "lumiere_c32_db_write_report.csv", "UCSF_C32_DB_WRITE_REPORT.CSV"],
)
def test_apply_with_write_report_name_passes(writer, name):
    assert writer.validate_report_path(Path(name), apply=True) == []


@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_dryrun_report_name_is_never_gated(writer):
    """Dry-run'da ad kuralı KAPI DEĞİL (yalnız uyarı) -- mevcut akışları bozmamak için."""
    assert writer.validate_report_path(Path("her_hangi_bir_ad.csv"), apply=False) == []
    assert writer.validate_report_path(Path("x_write_report.csv"), apply=False) == []


@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_write_report_marker_is_identical_across_writers(writer):
    """Üç yazıcıda AYNI kural -- birinde kayarsa bu test öter."""
    assert writer.WRITE_REPORT_MARKER == "_write"


@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_mode_dependent_report_defaults_can_never_collide(writer):
    """Karar 25'in KÖK NEDENİ: tek varsayılanın iki mod tarafından paylaşılması.

    UCSF'te varsayılan `ucsf_c32_db_dryrun.csv` idi -> gerçek yazım kanıtı
    "dryrun" adlı dosyaya düştü. Tek varsayılanı `_write_report` yapmak hatayı
    TERS yöne çevirirdi (bir dry-run yazım kanıtını silerdi -- UPenn/LUMIERE'de
    varsayılan tam olarak buydu). Artık iki AYRI varsayılan var ve:
      - yazım varsayılanı `--apply` kapısından GEÇER,
      - dry-run varsayılanı `_write` İÇERMEZ (yazım kanıtını ezemez),
      - ikisi ASLA aynı dosya olamaz.
    """
    assert writer.DEFAULT_WRITE_REPORT != writer.DEFAULT_DRYRUN_REPORT
    assert writer.validate_report_path(writer.DEFAULT_WRITE_REPORT, apply=True) == []
    assert writer.WRITE_REPORT_MARKER not in writer.DEFAULT_DRYRUN_REPORT.name.lower()
    assert "dryrun" in writer.DEFAULT_DRYRUN_REPORT.name.lower()


def test_ucsf_dryrun_default_does_not_overwrite_2026_08_18_write_evidence():
    """2026-08-18'in GERÇEK yazım kanıtı `ucsf_c32_db_dryrun.csv` adlı dosyada.

    Bu dosya tarihsel kayıt olarak SİLİNMEZ/ÜZERİNE YAZILMAZ (wiki hard rule #4),
    bu yüzden dry-run varsayılanı o ADI KULLANAMAZ.
    """
    assert ucsf_writer.DEFAULT_DRYRUN_REPORT.name != "ucsf_c32_db_dryrun.csv"
    assert ucsf_writer.DEFAULT_WRITE_REPORT.name == "ucsf_c32_db_write_report.csv"


# --------------------------------------------------------------------------- #
# F3 -- LEGACY_SEGMENTATION_TOOL artık ölü kod değil, gerçek çapa
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("writer", LEGACY_WRITERS)
def test_legacy_tool_name_anchor(writer):
    """Bu script eski A-yöntemi baseline satırlarına (binWidth=25) YAZMAMALI."""
    assert writer.SEGMENTATION_TOOL != writer.LEGACY_SEGMENTATION_TOOL
    assert writer.SEGMENTATION_TOOL == f"{writer.LEGACY_SEGMENTATION_TOOL}-C32"


@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_segmentation_tool_is_c32(writer):
    assert writer.SEGMENTATION_TOOL.endswith("-C32")


# --------------------------------------------------------------------------- #
# F4 / GUARD 2 -- DB_ERROR_PREFIXES fail-open düzeltmesi
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_db_error_prefixes_cover_timeout(writer):
    """🔴 LUMIERE'de gerçek bir fail-open vardı: `HATA_TIMEOUT` tuple'da YOKTU.

    GUARD 2 `errors` sayacı `k.startswith(DB_ERROR_PREFIXES)` ile hesaplanıyor;
    `HATA_TIMEOUT:45s_asildi` eski tuple'la EŞLEŞMİYORDU, yani asılan satırlar
    hata sayılmıyor ve koşu (yazılan > 0 olduğu için) exit 0 veriyordu.
    """
    assert "HATA_TIMEOUT" in writer.DB_ERROR_PREFIXES
    assert "HATA_TIMEOUT:45s_asildi".startswith(writer.DB_ERROR_PREFIXES)
    assert "HATA_BAGLANTI:OperationalError:x".startswith(writer.DB_ERROR_PREFIXES)
    assert "HATA:ValueError:x".startswith(writer.DB_ERROR_PREFIXES)
    # Çift sayım olmamalı: GUARD 4'ün CSV-kaynaklı statüleri GUARD 2'ye GİRMEZ.
    assert not "HATA_CSV_STATUS:MASKE_YOK".startswith(writer.DB_ERROR_PREFIXES)
    assert not "HATA_IZINSIZ_MASK_SOURCE:'x'".startswith(writer.DB_ERROR_PREFIXES)


def test_upenn_guard6_action_is_not_counted_as_db_error():
    """GUARD 6'nın `HATA_TIMEPOINT_POST_OP` eylemi GUARD 2'nin DB hata sayacına
    KARIŞMAMALI (ayrı guard, ayrı exit kodu 3)."""
    assert not "HATA_TIMEPOINT_POST_OP:post-op_fallback_no_pretreatment".startswith(
        upenn_writer.DB_ERROR_PREFIXES
    )


def test_upenn_has_row_timeout_like_lumiere():
    """F4: satır-timeout + retry deseni artık UPenn'de de var (eskiden YOKTU)."""
    assert upenn_writer._ROW_TIMEOUT_SECONDS == lumiere_writer._ROW_TIMEOUT_SECONDS
    assert callable(upenn_writer._run_row_with_timeout)


# --------------------------------------------------------------------------- #
# F6 -- koşu argümanlarının sidecar JSON'a yazılması
# --------------------------------------------------------------------------- #
class _FakeArgs:
    apply = False
    csv = Path("fake_c32.csv")
    report = Path("fake_report.csv")
    limit = None
    ok_only = False
    skip_provenance_check = False
    require_new_writes = False
    accept_extraction_failures = 38
    skip_scan_id_gate = False


@pytest.mark.parametrize("writer", ALL_WRITERS)
def test_run_args_sidecar_records_accepted_failure_threshold(writer, tmp_path):
    """F6: kabul edilen `--accept-extraction-failures` eşiği artefakta YAZILMALI.

    Eskiden rapor CSV'si yalnız satır sonuçlarını içeriyordu; "kaç başarısızlık
    bilinçli kabul edildi?" sorusu artefakta bakarak YANITLANAMIYORDU.
    """
    args = _FakeArgs()
    args.report = tmp_path / "x_write_report.csv"
    sidecar = writer._write_run_args_sidecar(
        args.report, args, csv_sha256="deadbeef", summary={"rows_processed": 3}
    )
    assert sidecar.name.endswith(".run_args.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["thresholds"]["accept_extraction_failures"] == 38
    assert payload["resolved_args"]["accept_extraction_failures"] == 38
    assert payload["source_csv_sha256"] == "deadbeef"
    assert payload["segmentation_tool"] == writer.SEGMENTATION_TOOL
    assert payload["summary"] == {"rows_processed": 3}
    assert "argv" in payload and "written_at_utc" in payload
