"""pytest oturum yapilandirmasi -- ITK'nin Turkce-yol kisitini kalici cozer.

NEDEN VAR (2026-09-13, karar 40)
================================
Bu depoda 14 test AYLARCA "bilinen ITK Turkce-yol hatasi, cozulemez" diye
gecildi:

    test_segmentation                      4
    test_new_patient_segmentation          4
    test_harmonization                     3
    test_lumiere_deepbratumia_resolution   2
    test_resampling                        1

Hepsi ayni tek sebepten kirmiziydi:

    ** ERROR (nifti_image_write_engine): cannot open output file
       'C:\\Users\\Barış\\AppData\\Local\\Temp\\pytest-of-Barış\\pytest-N\\...'

## Kok neden zinciri (ÖLÇÜLDÜ, varsayilmadi -- 2026-09-13)

1. ITK/SimpleITK'nin NIfTI yazicisi (znzlib) yol string'ini C `fopen()`'a
   verir. Bu makinede ANSI kod sayfasi cp1254'tur ama Python yol'u UTF-8
   olarak gecirir; `ş` (U+015F) iki byte'a acilir ve kod sayfasinda yanlis
   cozulur -> ITK var olmayan bir yolu acmaya calisir.
   Olculen kanit (bu depoda, 2026-09-13):
       C:\\Users\\Barış\\AppData\\Local\\Temp\\_probe.nii   -> RuntimeError
       C:\\pytmp_gbmaid\\_probe.nii                        -> YAZILDI (608 byte)
2. pytest'in VARSAYILAN gecici dizini iki yerde `ş` tasir: kullanici
   profili (`C:\\Users\\Barış\\...`) VE `pytest-of-<kullanici adi>` klasor
   adi (`pytest-of-Barış`).
3. `tempfile.gettempdir()` bu makinede 8.3 KISA adi dondurur
   (`C:\\Users\\BAR~1\\AppData\\Local\\Temp` -- ASCII!), ama pytest onu
   `getbasetemp()` icinde `.resolve()` eder ve Turkce hale GERI DONER.
   Bu yuzden "kisa ad ASCII, sorun yok" cikarimi YANLIStir.
4. `subst X:` de cozum DEGIL: pytest yolu `.resolve()` ettigi icin
   `--basetemp=X:/_pytmp` yeniden `C:\\Users\\Barış\\...` olur (ayni
   K12 tuzagi, bkz. `tests/test_no_resolve_path_regression.py`).
5. `USERNAME=baris` TEK BASINA yetmez: `pytest-of-baris` ASCII olur ama
   ust dizin (`C:\\Users\\Barış\\...Temp`) hala `ş` tasir.

## Secilen cozum ve NEDEN bu (alternatifler olculdu)

`pytest_configure` icinde, YALNIZ gerekliyse, iki ortam degiskeni ayarlanir:

    PYTEST_DEBUG_TEMPROOT = <ASCII, yazilabilir kok>   (zincirin 1. halkasi)
    LOGNAME               = <kullanici adinin ASCII'si> (zincirin 2. halkasi)

Ikisi birlikte pytest'in `getbasetemp()` zincirini tamamen ASCII yapar:
`<ASCII kok>\\pytest-of-Baris\\pytest-N\\<test adi>`.

### Neden `--basetemp` DEGIL (bilincli tercih)
`--basetemp` verildiginde pytest:
  * verilen dizini her kosuda `rm_rf` ile SILER,
  * numarali dizin (`pytest-N`) + kilit + eski kosu temizligi mekanizmasini
    TAMAMEN atlar.
Bu depoda ayni anda birden fazla terminal/ajan pytest kosuyor
(`AKTIF-GOREVLER.md` protokolu bunun icin var). Sabit bir `--basetemp` ile
iki es zamanli kosu birbirinin gecici dosyalarini kosu ortasinda silerdi ->
aciklanamayan, tekrarlanamayan kirmizilar. `PYTEST_DEBUG_TEMPROOT` ise
pytest'in kilitli/numarali/otomatik-temizlenen mekanizmasini KORUR.

### Neden proje ICINDE bir dizin DEGIL
Proje yolunun kendisi Turkce (`C:\\Users\\Barış\\Desktop\\...`), yani proje
altindaki ASCII adli bir alt dizin de ASCII OLMAZ. Ayrica proje ici bir
gecici dizin git'e/yedeklere karisir (`gbm-aid mert/` bir git deposudur).
Secilen koklerin HEPSI proje agacinin DISINDADIR; asagidaki
`_is_outside_repo()` bunu her kosuda dogrular.

## Linux / CI / baska makine icin ZARARSIZLIK (iki kapili koruma)

Bu kanca su IKI kosul birlikte saglanmazsa HICBIR SEY yapmaz:
  (a) `sys.platform == "win32"`  -- sorun ITK'nin Windows ANSI kod sayfasi
      donusumune ozgudur; Linux/macOS UTF-8 dosya adlarini sorunsuz acar,
  (b) pytest'in kullanacagi VARSAYILAN yol gercekten ASCII-DISI.
Yani Linux'ta, CI'da ve ASCII kullanici adi olan Windows makinelerinde
davranis BIT BIT ayni kalir -- kod yolu hic girilmez. Kullanici adi da
hicbir yere GOMULMEZ: `getpass.getuser()`'dan turetilir.

Ayrica kullanici `--basetemp` ya da `PYTEST_DEBUG_TEMPROOT` verdiyse kanca
GERI CEKILIR (acik niyet ezilmez).

## SESSIZ BASARISIZLIK YOK

Ne yapildigi (ya da neden yapilamadigi) her kosuda pytest basligina
yazilir -- `pytest_report_header`. Yazilabilir ASCII kok bulunamazsa kanca
sessizce pes ETMEZ: bir `UserWarning` yukseltir ve basliga ACIKCA "14 ITK
testi yine kirmizi olabilir" notunu duser.
"""

from __future__ import annotations

import getpass
import os
import sys
import tempfile
import unicodedata
import warnings
from pathlib import Path

# Bu dosyanin bulundugu dizin = kod koku ("gbm-aid mert").
# K12: `.resolve()` DEGIL `.absolute()` -- subst eslemesini korur.
# (bkz. tests/test_no_resolve_path_regression.py)
_CODE_ROOT = Path(__file__).absolute().parent

# Turkce harfler NFKD ile tam cozulmez (`ı` U+0131 hic ayrismaz, silinirdi);
# once acik esleme, sonra NFKD artiklarini temizle.
_TURKISH_TO_ASCII = str.maketrans(
    {
        "ı": "i", "İ": "I", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G",
        "ü": "u", "Ü": "U", "ö": "o", "Ö": "O", "ç": "c", "Ç": "C",
    }
)

# Aday ASCII kokler, tercih sirasiyla. Hicbiri sabit yazilmis tam yol
# DEGIL: surucu harfi/ozel klasorler ortam degiskenlerinden okunur.
_CANDIDATE_ENV_ROOTS = (
    ("SystemDrive", "pytest-tmp"),   # ornek: C:\pytest-tmp
    ("PUBLIC", "pytest-tmp"),        # ornek: C:\Users\Public\pytest-tmp
    ("ProgramData", "pytest-tmp"),   # ornek: C:\ProgramData\pytest-tmp
)

# Baslikta gosterilecek not; `pytest_configure` doldurur.
_HEADER_NOTES: list[str] = []


def _asciify(name: str) -> str:
    """Bir adi ASCII'ye indir (`Barış` -> `Baris`). Bos kalirsa `user`."""

    folded = name.translate(_TURKISH_TO_ASCII)
    folded = unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode("ascii")
    kept = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in folded)
    return kept or "user"


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover -- getpass'in calismadigi ortamlar
        return "unknown"


def _pytest_default_rootdir() -> Path:
    """pytest'in `getbasetemp()` icinde KULLANACAGI kok dizini birebir uret.

    `_pytest.tmpdir.TempPathFactory.getbasetemp()` ile ayni adimlar:
    `PYTEST_DEBUG_TEMPROOT` ya da `tempfile.gettempdir()`, ardindan
    `.resolve()`, ardindan `pytest-of-<get_user()>`.

    `.resolve()` BURADA ZORUNLU ve dogrudur: 8.3 kisa adi (`BAR~1`) ASCII
    goruntugu icin `.absolute()` ile bakmak yanlis "sorun yok" kararina yol
    acar (olculdu, bkz. modul docstring'i madde 3). Bu cagri `__file__`
    uzerinde DEGIL, gecici dizin string'i uzerindedir -- K12 yasagi
    goruntu yazan modullerdeki `Path(__file__).resolve()` desenine dairdir.
    """

    from_env = os.environ.get("PYTEST_DEBUG_TEMPROOT")
    temproot = Path(from_env or tempfile.gettempdir()).resolve()
    return temproot / f"pytest-of-{_current_user()}"


def _is_outside_repo(path: Path) -> bool:
    """Yol proje/kod agacinin DISINDA mi? (git'e ve yedeklere karismasin)"""

    try:
        path.absolute().relative_to(_CODE_ROOT)
    except ValueError:
        pass
    else:
        return False
    # Proje koku = kod kokunun bir ustu.
    try:
        path.absolute().relative_to(_CODE_ROOT.parent)
    except ValueError:
        return True
    return False


def _usable_ascii_root() -> Path | None:
    """Ilk ASCII + yazilabilir + proje DISI adayi dondur; yoksa None.

    "Yazilabilir" VARSAYILMAZ: gercekten mkdir + dosya yaz + sil denenir.
    """

    for env_var, leaf in _CANDIDATE_ENV_ROOTS:
        base = os.environ.get(env_var)
        if not base:
            continue
        if env_var == "SystemDrive":
            base = base.rstrip("\\/") + os.sep
        candidate = Path(base) / leaf
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not str(resolved).isascii():
            continue
        if not _is_outside_repo(resolved):
            continue
        try:
            resolved.mkdir(parents=True, exist_ok=True)
            probe = resolved / ".pytest-write-probe"
            probe.write_text("ok", encoding="ascii")
            probe.unlink()
        except OSError:
            continue
        return resolved
    return None


def pytest_configure(config) -> None:
    """Gerekliyse pytest'in gecici dizin zincirini ASCII'ye tasi."""

    # (a) Sorun Windows/ITK'ya ozgu -- baska yerde hic devreye girme.
    if sys.platform != "win32":
        return

    # Kullanicinin acik niyetini ezme.
    if getattr(config.option, "basetemp", None):
        _HEADER_NOTES.append(
            "tmp: --basetemp acikca verildi, ASCII yonlendirmesi UYGULANMADI."
        )
        return
    if os.environ.get("PYTEST_DEBUG_TEMPROOT"):
        _HEADER_NOTES.append(
            "tmp: PYTEST_DEBUG_TEMPROOT ortamda zaten ayarli, "
            "ASCII yonlendirmesi UYGULANMADI."
        )
        return

    # (b) Varsayilan yol gercekten ASCII-disi mi?
    default_rootdir = _pytest_default_rootdir()
    if str(default_rootdir).isascii():
        return  # ASCII kullanici adi -> yapacak is yok, davranis degismez.

    ascii_root = _usable_ascii_root()
    if ascii_root is None:
        tried = [f"%{var}%\\{leaf}" for var, leaf in _CANDIDATE_ENV_ROOTS]
        message = (
            "pytest gecici dizini ASCII-DISI ("
            f"{default_rootdir}) ve yazilabilir bir ASCII kok BULUNAMADI "
            f"(denenenler: {', '.join(tried)}). ITK/SimpleITK non-ASCII "
            "yollara NIfTI YAZAMAZ -- goruntu yazan 14 test yine kirmizi "
            "olabilir. Elle cozum: pytest'i "
            "`--basetemp=C:/pytest-tmp/gbm-aid` ile kosun."
        )
        warnings.warn(message, UserWarning, stacklevel=1)
        _HEADER_NOTES.append("tmp: ASCII kok BULUNAMADI -- " + message)
        return

    ascii_user = _asciify(_current_user())
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(ascii_root)
    # `getpass.getuser()` sirayla LOGNAME, USER, LNAME, USERNAME okur.
    # Windows'ta LOGNAME bir POSIX kalintisidir ve isletim sistemi onu
    # KULLANMAZ; bu yuzden `USERNAME` yerine LOGNAME ayarliyoruz -- boylece
    # pytest ASCII kullanici adini gorur, Windows'un kendi `USERNAME`'i
    # DOKUNULMADAN kalir.
    os.environ["LOGNAME"] = ascii_user

    _HEADER_NOTES.append(
        f"tmp: varsayilan yol ASCII-disi ({default_rootdir}) -> "
        f"PYTEST_DEBUG_TEMPROOT={ascii_root}, LOGNAME={ascii_user} "
        "(ITK non-ASCII yola NIfTI yazamaz; karar 40)"
    )


def pytest_report_header(config) -> list[str]:
    """Ne yapildigini her kosuda GORUNUR kil (sessiz sihir yok)."""

    return list(_HEADER_NOTES)


# =====================================================================
# LLM ORTAM IZOLASYONU -- 2026-09-14 (Baris'in anahtari .env'ye yazildi)
# =====================================================================
# NEDEN VAR: `tests/test_rag_llm.py::test_provider_is_not_configured_so_
# no_network_call_can_happen` kendi dokstring'inde su GARANTIYI veriyordu:
# "bu test dosyasi hicbir kosulda gercek bir saglayiciya gitmez".
# O garanti, ortamda LLM degiskeni OLMAMASINA dayaniyordu -- yani
# ortama bagli, kirilgan bir guvenceydi. 2026-09-14'te `.env`'ye gercek
# bir OpenAI anahtari yazilinca garanti DUSTU ve 3 test kirmizi oldu.
#
# RISK (teorik degil): test paketi 1145 test topluyor. Ortamdan beslenen
# gercek bir anahtarla, monkeypatch'i atlayan/ileride eklenen tek bir
# test GERCEK agla konusabilir ve Ege'nin 10 USD SERT LIMITINI sessizce
# yiyebilir. "Sessiz fallback yasak" kuralinin para tarafi.
#
# COZUM: her teste girerken LLM ortam degiskenleri KALDIRILIR. Bir test
# saglayiciyi gercekten istiyorsa kendi `monkeypatch.setenv`'iyle ACIKCA
# geri koyar -- yani niyet kodda GORUNUR olur, ortamdan sizmaz.
# `.env` DOSYASINA DOKUNULMAZ; yalnizca test surecinin ortami temizlenir.
import pytest as _pytest

_LLM_ENV_KEYS = (
    "GBMAID_LLM_ENABLED",
    "GBMAID_LLM_PROVIDER",
    "GBMAID_LLM_MODEL",
    "GBMAID_LLM_TIMEOUT_SECONDS",
    "GBMAID_LLM_MAX_RETRIES",
    "GBMAID_LLM_MAX_OUTPUT_TOKENS",
    "GBMAID_LLM_TEMPERATURE",
    "GBMAID_LLM_EFFORT",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


@_pytest.fixture(autouse=True)
def _isolate_llm_environment(monkeypatch):
    """Her testte LLM ortamini TEMIZLE -- kazara gercek API cagrisi olmasin.

    Testler saglayiciyi istiyorsa `monkeypatch.setenv(...)` ile kendisi
    kurar. Bu fixture `.env` dosyasini DEGISTIRMEZ, yalnizca pytest
    surecinin `os.environ`'ini duzenler ve test bitince geri alir.
    """

    for _key in _LLM_ENV_KEYS:
        monkeypatch.delenv(_key, raising=False)

    # `os.environ` temizligi TEK BASINA YETMIYOR (2026-09-14'te olculdu):
    # `pipeline/llm_provider.py:186` proje kokundeki `.env`'yi KENDI
    # yukluyor (`_load_dotenv_once()` -> `load_dotenv(override=False)`).
    # monkeypatch degiskenleri silince o yukleyici bosluk goruyor ve
    # dosyadan GERI DOLDURUYOR -- yani anahtar sizmaya devam ediyordu.
    # Modulun KENDI "bir kez yukle" bayragini kullaniyoruz (ozel bir
    # mekanizma icat etmiyoruz); monkeypatch test bitince geri aliyor.
    try:
        from pipeline import llm_provider as _llm_provider
    except Exception:  # modul yoksa/import edilemiyorsa izolasyon gereksiz
        return
    monkeypatch.setattr(_llm_provider, "_dotenv_loaded", True, raising=False)


# =====================================================================
# BUYUME SIMULASYONU KOHORT ONBELLEGI IZOLASYONU -- 2026-09-17
# =====================================================================
# NEDEN VAR: `api/analyze_patient.py` 2026-09-17'de kohort geneli buyume
# turevlerini (DB okuma + TUM LUMIERE kohortunun fit'i + RANO grup
# dagilimi) SUREC-ICI bir onbellege aldi -- canli sunucuda olculen 58-71
# saniyelik gecikmenin tek kaynagi oydu (LUMIERE disi hastalarda ayni
# cagri 4,8-8,1 sn).
#
# YAN ETKI OLCULDU, VARSAYILMADI: onbellek eklenir eklenmez
# `tests/test_api_analyze_patient.py` 30 testin 16'si KIRMIZI oldu; tek
# tek kosuldugunda hepsi YESILDI. Sebep test kirlenmesiydi -- o dosyadaki
# testler `growth_simulation_module.fetch_lumiere_*`'i monkeypatch edip
# HER TEST ICIN FARKLI sahte kohort veriyor, ama ilk testin kohortu
# onbellege girince sonrakiler onu goruyordu.
#
# COZUM: her testin BASINDA ve SONUNDA onbellek bosaltilir. Bir test
# onbellek davranisinin KENDISINI olcuyorsa (bkz.
# `tests/test_growth_cohort_cache.py`) kendi icinde tekrar doldurur --
# yani niyet kodda GORUNUR olur.
#
# ⚠️ MODULU ZORLA IMPORT ETMEZ: `api.analyze_patient` agir bir moduldur
# (shap/xgboost zinciri). Yalnizca ZATEN yuklenmisse sifirlar -- boylece
# onunla ilgisi olmayan testlere hicbir maliyet binmez.


@_pytest.fixture(autouse=True)
def _reset_growth_cohort_cache():
    """Testler arasi buyume-kohortu onbellegini sifirla."""

    def _reset() -> None:
        module = sys.modules.get("api.analyze_patient")
        if module is None:
            return
        resetter = getattr(module, "reset_growth_cohort_cache", None)
        if resetter is not None:
            resetter()

    _reset()
    yield
    _reset()
