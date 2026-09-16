"""LINT TESTİ: görüntü yazan modüllerde `Path(__file__).resolve()` YASAK.

## Neden bu test var (2026-08-14)

Bu makinede kullanıcı adı Türkçe karakter içeriyor
(`C:\\Users\\Barış\\...`) ve **ITK/SimpleITK non-ASCII yollara NIfTI
YAZAMIYOR**. Çözüm olarak `subst X: "C:\\Users\\Barış\\Desktop\\GBM-AID
Prototip\\gbm-aid mert"` ile ASCII bir sürücü haritalandı.

**Tuzak:** `Path.resolve()` Windows'ta `subst` eşlemesini GERÇEK yola geri
çözer (`GetFinalPathName` davranışı) -- yani `X:\\...` yeniden
`C:\\Users\\Barış\\...` olur ve ITK yazma hatası GERİ GELİR.
`Path.absolute()` ise sürücü harfini KORUR.

Canlı ölçüm (2026-08-14, X: üzerinden):
    Path('tools/x.py').resolve().parents[1]   -> C:\\Users\\Barış\\...   ✗
    Path('tools/x.py').absolute().parents[1]  -> X:\\                    ✓

## Bu testin var olma sebebi: bilgi YAYILMADI

Bu tuzak **2026-08-09'da keşfedildi** (bkz. `log/2026-08-09.md`) ve
`pipeline/harmonization.py`'de düzeltildi, uyarı yorumu yazıldı. Ama diğer
dosyalara YAYILMADI -- sonradan yazılan `run_pyradiomics_{tcga,upenn}.py`
aynı `.resolve()` desenini yeniden kopyaladı ve **2026-08-14'te B1 tam
kohort koşusunun ilk denemesini çökertti** (tüm taramalarda
`ITK ERROR: nifti library failed to write image`).

Ders: yorum satırındaki bilgi yayılmaz, **çalıştırılabilir kural yayılır.**
Bu test o kuraldır.

## Test NE denetler

Yalnız **görüntü yazma yeteneği olan** modülleri (SimpleITK'yı doğrudan ya
da N4/Z-score/resample yardımcıları üzerinden kullananlar). Bu modüllerde
`__file__` üzerinde `.resolve()` çağrısı -- **NEREDE OLURSA OLSUN** --
yasaktır.

CSV/DB-only araçlar (`write_*_to_db.py`, `train_cox_week3.py`,
`db_connection.py` ...) kapsam DIŞIDIR: Python'un kendi dosya G/Ç'si
Türkçe yollarda sorunsuz çalışır, sorun yalnız ITK'ya özgüdür.

## 2026-09-13 DERİNLEŞTİRME (karar 30) -- lint YALAN SÖYLÜYORDU

Eski `_resolve_violations()` iki ayrı biçimde SIĞDI:

1. Yalnız `tree.body`'yi (MODÜL seviyesi) tarıyordu -- fonksiyon/metot/
   sınıf **içindeki** atamalar hiç görülmüyordu.
2. Yalnız **BÜYÜK HARFLİ** hedef adlarını (`PROJECT_ROOT` gibi "yol
   sabiti konvansiyonu") kabul ediyordu -- `generator_path` gibi
   küçük harfli yerel değişkenler filtreleniyordu.

Sonuç: test *"temiz"* raporlarken diskte **dört gerçek ihlal** duruyordu
(2026-09-13'te ölçüldü, hepsi `_build_manifest()` gibi fonksiyonların
İÇİNDE, küçük harfli `generator_path` hedefiyle):

    tools/run_pyradiomics_lumiere.py:437   generator_path = Path(__file__).resolve()
    tools/run_pyradiomics_tcga.py:464      (aynı)
    tools/run_pyradiomics_ucsf.py:307      (aynı)
    tools/run_pyradiomics_upenn.py:411     (aynı)

Pratik etkisi sınırlıydı (bu yol yalnız `_file_sha256()` ve `.name` için
kullanılıyordu, ITK'ya hiç gitmiyordu) **ama lint yalan söylüyordu** --
ve bu testin var olma sebebi tam olarak "desen kopyalanır" riskiydi:
yanlış desen dosyada durduğu sürece bir sonraki script onu yeniden
kopyalar.

**Düzeltme (a):** tarama artık `ast.walk()` ile TÜM ağacı gezer ve
atama-hedefi adına HİÇ bakmaz. Dahası, artık `ast.Assign` ile de
sınırlı değil: `__file__` alıcı zincirinde olan HER `.resolve()`
**çağrısı** (fonksiyon argümanı içinde, f-string içinde, dönüş
ifadesinde, zincirlenmiş `.parents[1]` ardında ...) yakalanır.

**Düzeltme (c):** dört satır `.absolute()` yapıldı -- muafiyet
İŞARETLENMEDİ. Gerekçe: `.absolute()` bu dört yerde **davranışı hiç
değiştirmiyor** (aynı dosya, aynı bayt → aynı `_file_sha256()`; aynı
`.name`; manifest'e tam yol string'i HİÇ yazılmıyor -- ölçüldü), yani
muafiyetin bedeli sıfır ama faydası negatifti: yanlış desen dosyada
kalsa kopyalanmaya devam ederdi. Muafiyet mekanizması yalnız
`.absolute()`'ın GERÇEKTEN yanlış sonuç ürettiği (böyle bir yer henüz
bulunmadı) durumlar için saklı tutuluyor.

## Bilinçli istisna -- ve kötüye kullanımın ZORLAŞTIRILMASI

Gerekirse ilgili satıra `# lint-allow-resolve: <gerekçe>` yorumu eklenir.
İki ek fren var (2026-09-13):

1. **Gerekçe ZORUNLU ve boş olamaz** -- `# lint-allow-resolve:` tek
   başına (gerekçesiz) muafiyet SAYILMAZ, ihlal olarak raporlanır.
   `MIN_ALLOW_REASON_LENGTH` karakterden kısa gerekçe de reddedilir.
2. **Muafiyet BÜTÇESİ sıfırdır** (`EXPECTED_ALLOW_MARKER_COUNT = 0`).
   Yeni bir muafiyet eklemek, bu sabiti de AÇIKÇA artırmayı gerektirir
   -- yani sessizce eklenemez, diff'te görünür ve review'a düşer.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# `.absolute()` kullanılmalı, `.resolve()` DEĞİL -- bkz. modül docstring'i.
PROJECT_ROOT = Path(__file__).absolute().parents[1]

SCAN_DIRS = ("tools", "pipeline")

# Bir modülü "görüntü yazabilir" sayan işaretler.
#
# 2026-09-12 EKLEME (Barış onayı, K12 kapsam boşluğu): `write_mask_overlay_
# preview` ve `fit_source_zscore_statistics` eklendi. Kanıtlanmış boşluk:
# `tools/mask_overlay.py` ve `tools/fit_source_zscore.py` bu iki eklenene
# kadar eski 5 işaretçiden HİÇBİRİNİ içermiyordu -- bu yüzden hiç
# taranmıyorlardı ve `mask_overlay.py`'deki gerçek `.resolve()` ihlali
# 2026-08-14'ten 2026-09-12'ye kadar SESSİZCE kaldı (bkz. log/2026-09-12.md).
# İkisi de doğrulanmış gerçek import/çağrı işaretleridir (yorum satırı
# DEĞİL): `write_mask_overlay_preview` -- `pipeline/resampling.py`'de
# tanımlı, SimpleITK tabanlı PNG/overlay yazıcısı; `fit_source_zscore_
# statistics` -- `pipeline/harmonization.py`'de tanımlı, `sitk.ReadImage`
# ile görüntü okuyan fonksiyon (kendisi NIfTI YAZMAZ, ama K12 tutarlılığı
# için aynı kurala tabi tutulması Barış'ın kararıdır).
IMAGE_WRITING_MARKERS = (
    "SimpleITK",
    "apply_n4_bias_correction",
    "apply_zscore_normalization",
    "resample_image_and_mask",
    "build_derived_region_masks",
    "write_mask_overlay_preview",
    "fit_source_zscore_statistics",
)

ALLOW_MARKER = "# lint-allow-resolve:"

# Muafiyet kötüye kullanımını zorlaştıran iki fren (2026-09-13, karar 30).
# Bkz. modül docstring'i "Bilinçli istisna" bölümü.
MIN_ALLOW_REASON_LENGTH = 10
EXPECTED_ALLOW_MARKER_COUNT = 0


def _image_writing_modules() -> list[Path]:
    """Görüntü yazma yeteneği olan modülleri bul."""

    found: list[Path] = []
    for directory in SCAN_DIRS:
        root = PROJECT_ROOT / directory
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(marker in text for marker in IMAGE_WRITING_MARKERS):
                found.append(path)
    return found


def _has_valid_allow_marker(lines: list[str], line_no: int) -> bool:
    """Satırda/bir üst satırda GEÇERLİ (gerekçeli) bir muafiyet işareti var mı?

    Gerekçesiz ya da `MIN_ALLOW_REASON_LENGTH`'ten kısa gerekçeli işaret
    GEÇERSİZDİR -- drive-by muafiyet eklemeyi zorlaştırır (karar 30).
    """

    nearby = lines[max(0, line_no - 2) : line_no]
    for line in nearby:
        if ALLOW_MARKER not in line:
            continue
        reason = line.split(ALLOW_MARKER, 1)[1].strip()
        if len(reason) >= MIN_ALLOW_REASON_LENGTH:
            return True
    return False


def _resolve_violations(path: Path) -> list[tuple[int, str]]:
    """`__file__` üzerindeki HER `.resolve()` çağrısını bul -- nerede olursa olsun.

    2026-09-13 derinleştirmesi (karar 30, bkz. modül docstring'i): eski
    sürüm yalnız `tree.body` (modül seviyesi) + BÜYÜK HARFLİ atama
    hedeflerini tarıyordu ve fonksiyon-içi `generator_path = Path(__file__)
    .resolve()` gibi dört gerçek ihlali KAÇIRIYORDU.

    Artık `ast.walk()` ile tüm ağaç gezilir ve atama yerine **çağrının
    kendisi** aranır: `func` bir `.resolve` attribute erişimiyse ve alıcı
    zincirinde `__file__` varsa ihlaldir. Bu, atama olmayan kullanımları
    da yakalar (fonksiyon argümanı, f-string, `return`, zincirlenmiş
    `.parents[1]` vb.).
    """

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:  # pragma: no cover -- bozuk dosya
        pytest.fail(f"{path.name} parse edilemedi: {exc}")

    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):  # TÜM ağaç -- fonksiyon/metot içi DAHİL
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "resolve":
            continue
        # `.resolve()`'un alıcısı (receiver) zincirinde `__file__` var mı?
        # Örn. `Path(__file__).resolve()`, `Path(__file__).parent.resolve()`.
        receiver_has_dunder_file = any(
            isinstance(inner, ast.Name) and inner.id == "__file__"
            for inner in ast.walk(node.func)
        )
        if not receiver_has_dunder_file:
            continue
        line_no = node.lineno
        if _has_valid_allow_marker(lines, line_no):
            continue
        segment = ast.get_source_segment(text, node) or ".resolve() çağrısı"
        violations.append((line_no, segment))
    return sorted(set(violations))


def _allow_marker_lines(path: Path) -> list[int]:
    """Dosyadaki GEÇERLİ muafiyet işaretlerinin satır numaraları."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    found: list[int] = []
    for index, line in enumerate(lines, start=1):
        if ALLOW_MARKER not in line:
            continue
        reason = line.split(ALLOW_MARKER, 1)[1].strip()
        if len(reason) >= MIN_ALLOW_REASON_LENGTH:
            found.append(index)
    return found


def test_image_writing_modules_do_not_use_resolve() -> None:
    """Görüntü yazan hiçbir modül `Path(__file__).resolve()` kullanmamalı."""

    modules = _image_writing_modules()
    assert modules, (
        "Hiç görüntü-yazan modül bulunamadı -- tarama yolu bozulmuş olabilir. "
        f"Bakılan: {[str(PROJECT_ROOT / d) for d in SCAN_DIRS]}"
    )

    failures: list[str] = []
    for module in modules:
        for line_no, snippet in _resolve_violations(module):
            failures.append(
                f"  {module.relative_to(PROJECT_ROOT)}:{line_no}  ->  {snippet}"
            )

    assert not failures, (
        "`Path(__file__).resolve()` görüntü yazan modüllerde YASAK.\n"
        "Windows'ta `subst` eşlemesini gerçek (Türkçe karakterli) yola geri "
        "çözer ve ITK'nın NIfTI yazımını bozar -- 2026-08-14'te B1 tam kohort "
        "koşusunun ilk denemesini çökertti.\n"
        "DÜZELTME: `.resolve()` yerine `.absolute()` kullan.\n"
        "Bilinçli istisna gerekiyorsa satırın üstüne "
        f"`{ALLOW_MARKER} <gerekçe>` yorumu ekle.\n"
        "İhlaller:\n" + "\n".join(failures)
    )


def test_pyradiomics_runners_use_absolute() -> None:
    """DÖRT C32 çıkarım script'i açıkça `.absolute()` kullanmalı (pozitif kontrol).

    Yukarıdaki test 'yasak deseni yok' der; bu test 'doğru desen VAR' der.
    İkisi birlikte, birinin sessizce silinmesini de yakalar.

    2026-09-13 (karar 30, düzeltme (b)): `run_pyradiomics_ucsf.py` bu
    listede EKSİKTİ -- UCSF 2026-08-18'de harici test setini devraldı ve
    kendi C32 çıkarım script'i yazıldı, ama pozitif kontrole hiç
    eklenmedi. Yani UCSF runner'ının `PROJECT_ROOT`'u sessizce
    `.resolve()`'a dönse bu test fark etmezdi.
    """

    runners = [
        "run_pyradiomics_tcga.py",
        "run_pyradiomics_upenn.py",
        "run_pyradiomics_lumiere.py",
        "run_pyradiomics_ucsf.py",
    ]
    # 2026-09-13 EK BULGU (karar 30): eski assert LİTERAL
    # `"PROJECT_ROOT = Path(__file__).absolute()"` string'ini arıyordu.
    # `run_pyradiomics_ucsf.py` kök sabitini `REPO_ROOT` diye
    # adlandırıyor (satır 52) -- yani UCSF listeye eklenseydi bile test
    # YANLIŞ-NEGATİF değil, YANLIŞ-POZİTİF verecekti. Kontrol artık
    # isimden bağımsız bir regex ile yapılıyor: kök sabiti hangi adla
    # tanımlanırsa tanımlansın `.absolute()` KULLANMALI.
    root_pattern = re.compile(
        r"^(?P<name>[A-Z_][A-Z0-9_]*)\s*=\s*Path\(__file__\)\.absolute\(\)",
        re.MULTILINE,
    )
    for name in runners:
        path = PROJECT_ROOT / "tools" / name
        assert path.is_file(), f"{name} bulunamadı"
        text = path.read_text(encoding="utf-8", errors="replace")
        assert root_pattern.search(text), (
            f"{name}: modül seviyesinde `<KOK_SABITI> = "
            "Path(__file__).absolute()` deseni YOK. "
            "Bu script NIfTI yazıyor; `.resolve()` ITK yazma hatasına yol açar."
        )
        # 2026-09-13 (karar 30): manifest'in `generator_script`/
        # `generator_script_sha256` alanlarını besleyen yol da
        # `.absolute()` olmalı -- eskiden `.resolve()`'du ve derinleşmemiş
        # lint bunu göremiyordu.
        assert "generator_path = Path(__file__).absolute()" in text, (
            f"{name}: `generator_path` `.absolute()` ile kurulmuyor "
            "(2026-09-13'te `.resolve()`'dan çevrildi, karar 30)."
        )


# ============================================================
# KIRMIZI/YEŞİL -- lint'in KENDİSİNİ test et (2026-09-13, karar 30)
# ============================================================
#
# Derinleştirilmiş `_resolve_violations()` gerçekten fonksiyon-İÇİ
# ihlalleri görüyor mu? Yukarıdaki iki test yalnız "repo temiz" der --
# eski (sığ) lint de "temiz" DİYORDU. Aşağıdaki testler dedektörün
# kendisini sentetik girdiyle ölçer: kaçırma (false negative) hatasını
# yakalayan tek katman budur.

_UNMARKED_FUNCTION_LOCAL = '''\
"""SimpleITK içerir -- tarama kapsamına girsin."""
from pathlib import Path


def build_manifest():
    generator_path = Path(__file__).resolve()
    return generator_path.name
'''

_MARKED_FUNCTION_LOCAL = '''\
"""SimpleITK içerir -- tarama kapsamına girsin."""
from pathlib import Path


def build_manifest():
    # lint-allow-resolve: yalniz sha256/name icin, ITK'ya gitmiyor
    generator_path = Path(__file__).resolve()
    return generator_path.name
'''

_REASONLESS_MARKER = '''\
"""SimpleITK içerir -- tarama kapsamına girsin."""
from pathlib import Path


def build_manifest():
    # lint-allow-resolve:
    generator_path = Path(__file__).resolve()
    return generator_path.name
'''

_NON_ASSIGNMENT_USE = '''\
"""SimpleITK içerir -- tarama kapsamına girsin."""
from pathlib import Path


def build_manifest():
    return str(Path(__file__).resolve().parents[1] / "out.nii.gz")
'''

_MODULE_LEVEL_LOWERCASE = '''\
"""SimpleITK içerir -- tarama kapsamına girsin."""
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
'''


def test_lint_catches_unmarked_function_local_resolve(tmp_path: Path) -> None:
    """KIRMIZI: muafiyet işareti OLMAYAN fonksiyon-içi `.resolve()` yakalanır.

    Bu, 2026-09-13'te bulunan gerçek kaçırmanın (dört
    `tools/run_pyradiomics_*.py` satırı) birebir sentetik kopyasıdır.
    ESKİ sığ lint bu girdiyi TEMİZ raporluyordu.
    """

    module = tmp_path / "fake_image_writer.py"
    module.write_text(_UNMARKED_FUNCTION_LOCAL, encoding="utf-8")

    violations = _resolve_violations(module)
    assert len(violations) == 1, f"Fonksiyon-içi ihlal kaçırıldı: {violations}"
    line_no, snippet = violations[0]
    assert line_no == 6, line_no
    assert ".resolve()" in snippet


def test_lint_catches_non_assignment_resolve(tmp_path: Path) -> None:
    """KIRMIZI: atamaya BAĞLI OLMAYAN `.resolve()` kullanımı da yakalanır."""

    module = tmp_path / "fake_image_writer.py"
    module.write_text(_NON_ASSIGNMENT_USE, encoding="utf-8")

    assert len(_resolve_violations(module)) == 1


def test_lint_catches_module_level_lowercase_name(tmp_path: Path) -> None:
    """KIRMIZI: küçük harfli modül-seviyesi yol değişkeni de yakalanır.

    Eski lint `name.isupper()` filtresi yüzünden bunu da kaçırıyordu.
    """

    module = tmp_path / "fake_image_writer.py"
    module.write_text(_MODULE_LEVEL_LOWERCASE, encoding="utf-8")

    assert len(_resolve_violations(module)) == 1


def test_lint_respects_marked_resolve(tmp_path: Path) -> None:
    """YEŞİL: gerekçeli muafiyet işareti olan satır ihlal SAYILMAZ."""

    module = tmp_path / "fake_image_writer.py"
    module.write_text(_MARKED_FUNCTION_LOCAL, encoding="utf-8")

    assert _resolve_violations(module) == []


def test_lint_rejects_reasonless_allow_marker(tmp_path: Path) -> None:
    """Gerekçesiz `# lint-allow-resolve:` muafiyet SAYILMAZ (kötüye kullanım freni)."""

    module = tmp_path / "fake_image_writer.py"
    module.write_text(_REASONLESS_MARKER, encoding="utf-8")

    assert len(_resolve_violations(module)) == 1, (
        "Gerekçesiz muafiyet işareti kabul edildi -- drive-by muafiyet "
        "eklemek mümkün, fren çalışmıyor."
    )


def test_allow_marker_budget_is_unchanged() -> None:
    """Muafiyet BÜTÇESİ: taranan modüllerde kaç muafiyet var?

    Şu an sıfır (dört `.resolve()` satırı `.absolute()` yapıldı, muafiyet
    İŞARETLENMEDİ). Yeni bir muafiyet eklemek `EXPECTED_ALLOW_MARKER_COUNT`
    sabitini de AÇIKÇA artırmayı gerektirir -- sessizce eklenemez, diff'te
    görünür ve review-gate'e düşer.
    """

    markers: list[str] = []
    for module in _image_writing_modules():
        for line_no in _allow_marker_lines(module):
            markers.append(f"  {module.relative_to(PROJECT_ROOT)}:{line_no}")

    assert len(markers) == EXPECTED_ALLOW_MARKER_COUNT, (
        f"Muafiyet sayısı {len(markers)}, beklenen "
        f"{EXPECTED_ALLOW_MARKER_COUNT}. Yeni bir `{ALLOW_MARKER}` "
        "eklendiyse gerekçesini karar dosyasına yazın ve "
        "`EXPECTED_ALLOW_MARKER_COUNT`'u bilinçli olarak güncelleyin.\n"
        "Bulunanlar:\n" + "\n".join(markers)
    )
