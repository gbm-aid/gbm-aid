"""CLAUDE.md ile AGENTS.md'nin BİREBİR AYNI kalmasını zorlayan lint testi.

NEDEN VAR (2026-08-28, gerçek bir olaydan doğdu):
------------------------------------------------
`CLAUDE.md` (Claude Code'un okuduğu kural dosyası) ve `AGENTS.md`
(Codex ajanlarının okuduğu kural dosyası) byte-birebir aynı olmak
zorundadır -- ikisi de projenin KRİTİK MİMARİ KURALLAR'ını taşır.

2026-08-28'de bir bulgu düzeltme turunda K14 ve kohort sayısı maddeleri
YALNIZ `CLAUDE.md`'ye işlendi; `AGENTS.md` eski (bayat) metni taşımaya
devam etti. Sonuç: Claude Code ile Codex FARKLI kural setiyle
çalışacaktı -- Codex'e göre K14 hâlâ "atlanmış", kohort hâlâ "771"di.
Hata kapanış taramasında yakalandı ve `cp` ile düzeltildi, ardından
Codex çapraz incelemesi bunu HIGH bulgu olarak işaretledi:

    "CLAUDE.md/AGENTS.md eşitliği elle `cp` ile sağlandı; otomatik
     senkron kontrolü (CI/pre-commit) yok -- gelecekte tekrar diverge
     edebilir."

Bu test o boşluğu kapatır. Projenin kalıcı dersi (K13):

    *Hatırlanması gereken bir kural, ihlal edilemeyen bir yapıdan
     daima zayıftır.*

NASIL DÜZELTİLİR (test kırmızıysa) -- TEK KOMUT (2026-09-13):
-------------------------------------------------------------
    python tools/sync_claude_agents.py --apply

`cp` ile ELLE kopyalama artık gerekmiyor. 2026-09-13'te Barış'ın onayıyla
üretici script (`tools/sync_claude_agents.py`) eklendi: kanonik kaynak
`CLAUDE.md`, üretilen kopya `AGENTS.md`. Bu test YAKALAR, o script DÜZELTİR.
Düzeltme AGENTS.md tarafına yazıldıysa (nadir):

    python tools/sync_claude_agents.py --apply --from-agents

Dosyalar proje KÖKÜNDEDİR (`gbm-aid mert/`'in bir üstü).

NEDEN SYMLINK/HARDLINK DEĞİL (2026-09-13'te ÖLÇÜLDÜ, varsayılmadı):
Claude Code'un Edit aracı symlink'e yazmayı REDDEDİYOR; hardlink ise TEK
bir Edit çağrısında SESSİZCE kopuyor (ölçüldü: LinkType boşaldı, içerikler
ayrıştı) -- yani "ayrışamaz" sanılan çözüm, ayrışmayı gizleyerek üretirdi.
Ayrıntı: `decisions/2026-09-13-claude-agents-senkron-kalici-cozum.md`

NOT: `Path(__file__).resolve()` yerine `.absolute()` kullanılıyor --
K12'de kayıtlı Türkçe-yol/`subst` tuzağı nedeniyle bu depoda `.resolve()`
kullanımı istenmiyor (bkz. `tests/test_no_resolve_path_regression.py`).

PROJE KÖKÜ ARTIK ARANARAK BULUNUYOR (2026-09-13, düzeltme 2)
------------------------------------------------------------
~~Eski hâli: `_PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent`~~
(üstü çizildi, wiki hard rule #3: çelişki silinmez, işaretlenir)

O sabit "üç seviye yukarı" sayımı, testin **hangi yoldan çağrıldığına**
bağlıydı ve `subst X:` üzerinden koşulduğunda YANLIŞ yere bakıyordu:

    C:'ten  ->  .../GBM-AID Prototip/CLAUDE.md            ✓ 3 passed
    X:'ten  ->  X:\\CLAUDE.md                              ✗ 3 failed
                ("Failed: CLAUDE.md bulunamadi: X:\\CLAUDE.md")

Sebep: `X:` = **kod** kökü (`gbm-aid mert`), dolayısıyla `X:\\tests`'ten üç
seviye yukarı çıkmak `X:\\`'te kalır -- `CLAUDE.md`/`AGENTS.md` ise bir üst
seviyede, **proje** kökündedir. Yani test, dosyalar senkron olduğu hâlde
kırmızıydı: **ters-yanlış-kırmızı** (yol hatası, drift değil).

Düzeltme: kök artık **yapısal çapa aranarak** bulunuyor
(`_ANCHOR_DIRS` = wiki'nin `decisions/` + `log/` + `takim/` klasörleri).
Çapa olarak BİLEREK `CLAUDE.md`/`AGENTS.md` KULLANILMADI -- onlar bu testin
KONUSU; biri silinirse arama başarısız olup kafa karıştırıcı bir hata
vermesin, testin kendi "bulunamadi" mesajı çıksın (amaç korunur).
"""

import sys
from pathlib import Path

import pytest

# Proje kökünü tanıyan yapısal çapalar (wiki şeması, bkz. CLAUDE.md).
# Bunlar kod kökünde (`gbm-aid mert/`) YOKTUR -- ayrım bu yüzden kesin.
_ANCHOR_DIRS = ("decisions", "log", "takim")


def _looks_like_project_root(candidate: Path) -> bool:
    return all((candidate / name).is_dir() for name in _ANCHOR_DIRS)


def _find_project_root() -> Path:
    """Proje kökünü yukarı doğru yürüyerek bul -- çalışma dizininden bağımsız.

    İki aşamalı, çünkü `subst` sürücüsünden lexical yürüyüş YETMEZ:

    1. `.absolute()` ile yürü (K12 tercihi: `subst` harfini korur). Normal
       (`C:`) çağrımda burada bulunur ve `.resolve()` HİÇ çağrılmaz.
    2. Bulunamazsa `.resolve()` ile yürü. `X:` bir `subst` eşlemesiyse
       `X:\\` = kod kökü olduğu için proje köküne lexical olarak ÇIKILAMAZ
       (`Path("X:/").parent == Path("X:/")`) -- eşlemeyi geri çözmek
       ZORUNLUDUR.
    """

    lexical = Path(__file__).absolute().parent
    for base in (lexical, *lexical.parents):
        if _looks_like_project_root(base):
            return base

    # lint-allow-resolve: subst sürücüsünden proje köküne çıkmanın tek yolu;
    # bu modül ITK'ya HİÇ dokunmaz (yalnız `read_bytes` ile .md okur), o
    # yüzden K12'nin yasakladığı "Türkçe yola NIfTI yazma" riski YOKTUR.
    # NOT: `tests/` zaten `test_no_resolve_path_regression.SCAN_DIRS`
    # ("tools", "pipeline") kapsamında değildir; işaret yalnız belgeleme
    # amaçlıdır ve muafiyet bütçesini ETKİLEMEZ.
    resolved = Path(__file__).resolve().parent
    for base in (resolved, *resolved.parents):
        if _looks_like_project_root(base):
            return base

    raise RuntimeError(
        "Proje kökü bulunamadı. Aranan çapalar: "
        f"{_ANCHOR_DIRS}\n"
        f"  lexical başlangıç : {lexical}\n"
        f"  resolved başlangıç: {resolved}\n"
        "CLAUDE.md/AGENTS.md senkron kontrolü yapılamaz."
    )


# 🔴 2026-09-16 (ilk CI kosusunun yakaladigi gercek sinir): bu test
# `CLAUDE.md` ile `AGENTS.md`'nin birebir ayni kalmasini denetler. O iki dosya
# ve arananan capalar (`decisions/`, `log/`, `takim/`) **KOD DEPOSUNUN DISINDA**,
# proje kokunde durur -- depo `gbm-aid mert/`'tir. Dolayisiyla CI runner'i
# yalniz depoyu checkout ettiginde bu denetim YAPILAMAZ ve modul seviyesinde
# `RuntimeError` firlatarak TUM TOPLAMAYI durduruyordu (2 collection error).
#
# Cozum SESSIZ ATLAMA DEGILDIR: pytest'in kendi skip mekanizmasi kullanilir,
# gerekce raporda GORUNUR ve testin iddiasi (senkron kurali) CURUTULMUS SAYILMAZ
# -- yalnizca bu ortamda calistirilamaz. Yerelde capalar bulundugu icin test
# AYNEN kosmaya devam eder.
try:
    _PROJECT_ROOT = _find_project_root()
except RuntimeError as _exc:  # pragma: no cover -- yalniz CI/checkout ortami
    pytest.skip(
        "CLAUDE.md/AGENTS.md senkron denetimi bu ortamda YAPILAMAZ: wiki capalari "
        "(decisions/, log/, takim/) kod deposunun DISINDA, proje kokunde durur ve "
        f"CI checkout'unda bulunmazlar. Ayrinti: {_exc}",
        allow_module_level=True,
    )

_TOOLS_DIR = Path(__file__).absolute().parent.parent / "tools"

if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import sync_claude_agents as scs  # noqa: E402  (sys.path ayarı sonrası)

CLAUDE_MD = _PROJECT_ROOT / "CLAUDE.md"
AGENTS_MD = _PROJECT_ROOT / "AGENTS.md"


def _read_bytes(path: Path) -> bytes:
    if not path.is_file():
        pytest.fail(
            f"{path.name} bulunamadi: {path}\n"
            "Bu iki dosya projenin kural otoritesidir; biri yoksa "
            "senkron kontrolu yapilamaz."
        )
    return path.read_bytes()


def test_claude_md_ve_agents_md_byte_birebir_ayni():
    """CLAUDE.md ve AGENTS.md tek bir byte bile farklı olamaz."""
    claude = _read_bytes(CLAUDE_MD)
    agents = _read_bytes(AGENTS_MD)

    if claude == agents:
        return

    # Kırmızıysa, NEREDE ayrıldıklarını göster -- "farklı" demek yetmez.
    claude_lines = claude.decode("utf-8", errors="replace").splitlines()
    agents_lines = agents.decode("utf-8", errors="replace").splitlines()

    ilk_fark = None
    for idx, (c_line, a_line) in enumerate(zip(claude_lines, agents_lines), start=1):
        if c_line != a_line:
            ilk_fark = (idx, c_line, a_line)
            break

    if ilk_fark is None:
        # Satırlar prefix olarak aynı; biri digerinden uzun.
        ilk_fark = (
            min(len(claude_lines), len(agents_lines)) + 1,
            "<dosya bitti>" if len(claude_lines) <= len(agents_lines) else claude_lines[len(agents_lines)],
            "<dosya bitti>" if len(agents_lines) <= len(claude_lines) else agents_lines[len(claude_lines)],
        )

    satir, c_metin, a_metin = ilk_fark
    pytest.fail(
        "CLAUDE.md ve AGENTS.md AYRIŞMIŞ -- ikisi byte-birebir ayni olmali.\n"
        f"  CLAUDE.md : {len(claude)} byte, {len(claude_lines)} satir\n"
        f"  AGENTS.md : {len(agents)} byte, {len(agents_lines)} satir\n"
        f"  Ilk farkli satir: {satir}\n"
        f"    CLAUDE.md[{satir}]: {c_metin[:200]!r}\n"
        f"    AGENTS.md[{satir}]: {a_metin[:200]!r}\n"
        "\n"
        "SEBEBI GENELLIKLE: biri guncellendi, digeri unutuldu.\n"
        "DUZELTME -- TEK KOMUT ('gbm-aid mert' dizininden):\n"
        f"  {scs.FIX_COMMAND}\n"
        "  (duzeltme AGENTS.md tarafina yazildiysa: yukaridakine "
        "--from-agents ekle)\n"
        "\n"
        "NEDEN ONEMLI: CLAUDE.md'yi Claude Code, AGENTS.md'yi Codex "
        "ajanlari okur. Ayrisirlarsa iki taraf FARKLI kilitli kurallarla "
        "calisir (2026-08-28'de tam bu yasandi: K14 ve kohort sayisi "
        "yalniz CLAUDE.md'ye islenmisti)."
    )


def test_iki_dosya_da_kritik_mimari_kurallar_bolumunu_tasiyor():
    """Senkron testi boş/bozuk iki dosyayla da 'geçebilir' -- bunu engelle.

    İkisi de birbirinin aynısı olup ikisi de içi boşalmış olabilir; o
    durumda yukarıdaki test yeşil kalır ama kurallar kaybolmuş olur.
    Bu, projede tekrar tekrar yakalanan 'sessizce iyi say' (fail-open)
    hata sınıfıdır -- burada kapalı tutuluyor.

    ÇAPALAR ARTIK `tools/sync_claude_agents.py`'den import edilir (2026-09-13)
    -- üretici script'in fail-closed kapısı ile bu testin kapısı AYNI listeyi
    kullansın, birbirinden kaymasın diye. Liste orada boşaltılırsa hem burası
    hem script sessizce "her şey yolunda" derdi; o yüzden listenin kendisi de
    aşağıda denetleniyor.
    """
    beklenen_capalar = scs.REQUIRED_ANCHORS

    assert len(beklenen_capalar) >= 5, (
        "sync_claude_agents.REQUIRED_ANCHORS boşaltılmış/kısaltılmış: "
        f"{beklenen_capalar}. Bu liste hem bu testin hem üretici script'in "
        "fail-closed kapısıdır; kısaltmak iki korumayı birden devre dışı "
        "bırakır."
    )

    for path in (CLAUDE_MD, AGENTS_MD):
        metin = _read_bytes(path).decode("utf-8", errors="replace")
        eksikler = [capa for capa in beklenen_capalar if capa not in metin]
        assert not eksikler, (
            f"{path.name} icinde beklenen capalar EKSIK: {eksikler}\n"
            "Bu dosya projenin kilitli kurallarini tasimali. Eksikse ya "
            "dosya bozulmus ya da bir kilitli deger sessizce silinmis "
            "demektir -- ikisi de CRITICAL'dir."
        )


def test_duzeltme_yolu_gercekten_bagli_ve_ayni_karari_veriyor():
    """'Tek komutla düzelt' yolu ÇALIŞIR durumda mı? (2026-09-13)

    Yukarıdaki iki test ayrışmayı YAKALAR ama DÜZELTMEZ. Düzeltme yolu
    `tools/sync_claude_agents.py`'dir ve hata mesajı kullanıcıyı oraya
    yönlendirir. Eğer o script silinir/bozulur/yeniden adlandırılırsa,
    mesaj çalışmayan bir komutu önerir -- test hâlâ yeşil, düzeltme yolu
    ölü. Bu test o sessiz çürümeyi engeller.

    İki şeyi ölçer (ikisi de GERÇEK çağrı, varsayım değil):
      1. `first_difference()` byte karşılaştırmasıyla AYNI kararı veriyor
         (uydurma bir tmp çifti üzerinde: hem eşit hem farklı durum).
      2. `apply()`'in fail-closed kapısı, çapasız bir kaynağı REDDEDİYOR.
    """
    # 1a. Aynı içerik -> fark YOK.
    assert scs.first_difference(b"ayni\nicerik\n", b"ayni\nicerik\n") is None

    # 1b. Farklı içerik -> doğru satır numarası.
    fark = scs.first_difference(b"a\nb\nc\n", b"a\nX\nc\n")
    assert fark is not None, "Gercek fark tespit EDILEMEDI -- fail-open."
    assert fark[0] == 2, f"Yanlis satir bildirildi: {fark}"

    # 1c. Biri digerinin prefix'i (2026-08-28'de yasanan 'bayat kuyruk' hali).
    fark_uzunluk = scs.first_difference(b"a\nb\n", b"a\n")
    assert fark_uzunluk is not None and fark_uzunluk[0] == 2, (
        f"Prefix/uzunluk farki yakalanmadi: {fark_uzunluk}"
    )

    # 2. Fail-closed kapisi: capasiz kaynak REDDEDILMELI.
    eksikler = scs.missing_anchors(b"# bos bir dosya\n")
    assert eksikler, (
        "missing_anchors() ici bos bir dosyayi KABUL etti -- uretici script "
        "bu dosyayi digerinin uzerine yazabilirdi (fail-open)."
    )

    # Gercek CLAUDE.md ise kapiyi GECMELI (kapi asiri sıkı olmasin).
    assert not scs.missing_anchors(_read_bytes(CLAUDE_MD)), (
        "Gercek CLAUDE.md fail-closed kapisini GECEMEDI -- ya dosya bozuldu "
        "ya da REQUIRED_ANCHORS artik dosyayla uyusmuyor."
    )

    # Script'in isaret ettigi dosya gercekten var mi?
    script = _PROJECT_ROOT / "gbm-aid mert" / "tools" / "sync_claude_agents.py"
    assert script.is_file(), (
        f"Duzeltme script'i bulunamadi: {script}\n"
        f"Hata mesajlari '{scs.FIX_COMMAND}' komutunu oneriyor; dosya yoksa "
        "bu oneri OLU demektir."
    )
    assert "--apply" in scs.FIX_COMMAND
