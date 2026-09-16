"""CLAUDE.md -> AGENTS.md ikizlemesinin TEK KOMUTLA duzeltme yolu.

NEDEN VAR (2026-09-13, Baris onayi: "bitirelim bunu da, ugrasilmasin sonra")
---------------------------------------------------------------------------
`CLAUDE.md` (Claude Code okur) ve `AGENTS.md` (Codex ajanlari okur) byte-
birebir ayni olmak zorundadir. `tests/test_claude_agents_sync.py` ayrismayi
YAKALIYORDU ama DUZELTMIYORDU: duzeltme elle `cp` ile yapiliyordu ve bu her
seferinde tekrarlayan bir insan hatasi kaynagiydi (2026-08-28'de yasandi,
2026-09-11'de yine ek is gerektirdi).

Bu script o bosluğu kapatir: kanonik kaynak `CLAUDE.md`, uretilen kopya
`AGENTS.md`. Tek komut:

    python tools/sync_claude_agents.py --apply

NEDEN SYMLINK / HARDLINK DEGIL (2026-09-13'te OLCULDU, varsayilmadi)
--------------------------------------------------------------------
* SYMLINK: Claude Code'un Edit araci sembolik baglara yazmayi REDDEDIYOR
  ("Refusing to write ...: it is a symbolic link"). AGENTS.md bir symlink
  olsaydi, ona yazmak isteyen her ajan sert hata alirdi.
* HARDLINK: daha kotusu -- TEK bir Edit cagrisi hardlink'i SESSIZCE kopardi
  (olcum: Edit sonrasi LinkType bosaldi, iki dosyanin icerigi ayristi).
  Yani "hic ayrisamaz" sanilan cozum, tam da ayrismayi gizleyerek uretirdi.
* PRE-COMMIT HOOK: bu proje git'e hic commit edilmedi (K8, `.git` YOK) ->
  hook fiilen hic calismaz.
Ayrinti: `decisions/2026-09-13-claude-agents-senkron-kalici-cozum.md`

KULLANIM
--------
    python tools/sync_claude_agents.py            # --check (varsayilan)
    python tools/sync_claude_agents.py --apply    # CLAUDE.md -> AGENTS.md
    python tools/sync_claude_agents.py --apply --from-agents
                                                  # AGENTS.md -> CLAUDE.md
                                                  # (NADIR: yalnizca duzeltme
                                                  #  AGENTS.md tarafina
                                                  #  yazildiysa)
    python tools/sync_claude_agents.py --hook     # Claude Code PostToolUse
                                                  # hook modu (stdin'den JSON
                                                  # okur, ASLA yazmaz)

Cikis kodlari: 0 = senkron / uygulandi, 1 = ayrisma var (--check),
2 = kullanim/guvenlik hatasi (capa kontrolu, dosya yok).
`--hook` HER ZAMAN 0 doner (oturumu bozmamak icin); ayrismayi stdout'a
JSON uyarisi olarak bildirir.

OTOMATIK TETIKLEYICI (2026-09-13, Baris onayi)
----------------------------------------------
`--check` ayrismayi yakaliyordu ama biri `pytest` kosana kadar SESSIZ
kaliyordu. `.claude/settings.local.json` icindeki PostToolUse hook'u
(matcher: Write|Edit|MultiEdit) her yazimdan sonra bu script'i `--hook`
ile cagirir; CLAUDE.md/AGENTS.md disindaki dosyalarda hemen sessizce
cikar, ayrisma varsa uyarir.
UYARI: hook BILEREK `--apply` CALISTIRMAZ. Sessizce uzerine yazan bir
otomasyon, 2026-09-13'te hardlink'te olculen "ayrismayi gizleme" hatasinin
aynisini uretirdi. Hook TESPIT eder, duzeltmeyi insan tek komutla yapar.

FAIL-CLOSED: kaynak dosya asagidaki CAPALARI (kilitli degerleri) tasimiyorsa
kopyalama YAPILMAZ. Boylece "ici bosalmis bir CLAUDE.md" digerinin uzerine
yazilarak kurallar iki dosyada birden kaybolamaz. Bu, projede tekrar tekrar
yakalanan "sessizce iyi say" (fail-open) hata sinifini kapali tutar.

NOT (K12): bu depoda `.resolve()` yerine `.absolute()` kullanilir --
Turkce-yol/`subst` tuzagi (bkz. `tests/test_no_resolve_path_regression.py`).
2026-09-14 (kalem 43) EKLEME: proje koku artik `find_project_root()` ile
YAPISAL CAPA aranarak bulunur; ONCE `.absolute()` denenir, yalniz o
basarisiz olursa (yani `subst` surucusunden kosuluyorsa) `.resolve()`
devreye girer. Gerekce fonksiyonun kendi docstring'indedir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Proje kokunu taniyan YAPISAL capalar (wiki semasi, bkz. CLAUDE.md).
# Bunlar kod kokunde ("gbm-aid mert/") YOKTUR -- ayrim bu yuzden kesindir.
# BILEREK `CLAUDE.md`/`AGENTS.md` capa olarak KULLANILMAZ: onlar bu
# script'in KONUSU; biri silinirse arama degil, _require_file()'in kendi
# "bulunamadi" mesaji cikmali (amac korunur).
# NOT: `tests/test_claude_agents_sync.py` ayni desene sahiptir; o dosya bu
# script'i import ettigi icin listeyi buradan alabilirdi, ama test kendi
# kokunu import'tan ONCE hesaplamak zorunda (sys.path'e tools/ eklemek icin)
# -> iki yerde ayri durmasi yapisal, drift riski _ANCHOR_DIRS testiyle degil
# her iki tarafin ayni sonucu vermesiyle (iki surucuden kosu) denetlenir.
ANCHOR_DIRS: tuple[str, ...] = ("decisions", "log", "takim")


def _looks_like_project_root(candidate: Path) -> bool:
    return all((candidate / name).is_dir() for name in ANCHOR_DIRS)


def find_project_root() -> Path:
    """Proje kokunu YUKARI DOGRU YURUYEREK bul -- calisma dizininden bagimsiz.

    NEDEN (2026-09-14, kalem 43 -- OLCULDU, varsayilmadi):
    ------------------------------------------------------
    ~~Eski hali: `PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent`~~
    (ustu cizildi, wiki hard rule #3: celiski silinmez, isaretlenir)

    O sabit "uc seviye yukari" sayimi `subst X:` uzerinden COKUYORDU:

        C:'ten  ->  .../GBM-AID Prototip/CLAUDE.md      ✓ SENKRON
        X:'ten  ->  X:\\CLAUDE.md                        ✗ exit 2
                    ("HATA: CLAUDE.md bulunamadi: X:\\CLAUDE.md")

    Sebep `.resolve()` DEGIL, surucu-koku KENETLENMESI: `X:` = **kod** koku
    ("gbm-aid mert"), yani `X:\\tools`'tan uc seviye yukari cikmak `X:\\`'te
    KALIR (`Path("X:/").parent == Path("X:/")`), `CLAUDE.md`/`AGENTS.md` ise
    bir ust seviyede, **proje** kokundedir. Test (`test_claude_agents_sync.py`)
    2026-09-13'te ayni sebeple duzeltilmisti; bu script geride kalmisti ->
    "test yesil ama DUZELTME KOMUTU X:'ten calismiyor" (kalem 43).

    Iki asamali, cunku `subst` surucusunden lexical yuruyus YETMEZ:
      1. `.absolute()` ile yuru (K12 tercihi: `subst` harfini korur). Normal
         (`C:`) cagrimda burada bulunur ve `.resolve()` HIC cagrilmaz.
      2. Bulunamazsa `.resolve()` ile yuru -- `subst` eslemesini geri cozmek
         proje kokune cikmanin TEK yoludur.
      3. Ikisi de basarisizsa ESKI davranisa don (uc seviye yukari). Boylece
         capalar bir gun yeniden adlandirilirsa script CRASH ETMEZ, yalnizca
         eski (bilinen) davranisina geriler -- import eden test de cokmez.
    """

    lexical = Path(__file__).absolute().parent
    for base in (lexical, *lexical.parents):
        if _looks_like_project_root(base):
            return base

    # lint-allow-resolve: subst surucusunden proje kokune cikmanin tek yolu.
    # Bu modul ITK'ya HIC dokunmaz (yalniz `read_bytes`/`write_bytes` ile .md),
    # o yuzden K12'nin yasakladigi "Turkce yola NIfTI yazma" riski YOKTUR.
    resolved = Path(__file__).resolve().parent
    for base in (resolved, *resolved.parents):
        if _looks_like_project_root(base):
            return base

    return lexical.parent.parent  # tools/ -> "gbm-aid mert"/ -> proje koku


PROJECT_ROOT = find_project_root()

CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"
AGENTS_MD = PROJECT_ROOT / "AGENTS.md"

# Kaynak dosyanin TASIMASI ZORUNLU oldugu kilitli capalar.
# `tests/test_claude_agents_sync.py` bu listeyi BURADAN import eder --
# iki yerde ayri ayri tanimlanip birbirinden kaymasin diye.
REQUIRED_ANCHORS: tuple[str, ...] = (
    "KRİTİK MİMARİ KURALLAR",
    "611",  # Cox egitim havuzu (hasta)
    "585",  # olum olayi
    "295",  # UCSF harici test (hasta)
    "169",  # UCSF harici test (olay)
)

FIX_COMMAND = "python tools/sync_claude_agents.py --apply"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def missing_anchors(data: bytes) -> list[str]:
    """Kilitli capalardan EKSIK olanlari dondur (fail-closed kapisi)."""

    text = data.decode("utf-8", errors="replace")
    return [anchor for anchor in REQUIRED_ANCHORS if anchor not in text]


def first_difference(claude: bytes, agents: bytes) -> tuple[int, str, str] | None:
    """Ilk farkli satiri (1-tabanli no, CLAUDE metni, AGENTS metni) dondur."""

    if claude == agents:
        return None

    claude_lines = claude.decode("utf-8", errors="replace").splitlines()
    agents_lines = agents.decode("utf-8", errors="replace").splitlines()

    for idx, (c_line, a_line) in enumerate(zip(claude_lines, agents_lines), start=1):
        if c_line != a_line:
            return (idx, c_line, a_line)

    # Satirlar prefix olarak ayni; biri digerinden uzun.
    ortak = min(len(claude_lines), len(agents_lines))
    return (
        ortak + 1,
        "<dosya bitti>" if len(claude_lines) <= ortak else claude_lines[ortak],
        "<dosya bitti>" if len(agents_lines) <= ortak else agents_lines[ortak],
    )


def _require_file(path: Path) -> bytes:
    if not path.is_file():
        print(f"HATA: {path.name} bulunamadi: {path}", file=sys.stderr)
        raise SystemExit(2)
    return path.read_bytes()


def check() -> int:
    claude = _require_file(CLAUDE_MD)
    agents = _require_file(AGENTS_MD)

    print(f"CLAUDE.md : {len(claude):>6} byte  sha256={sha256_of(CLAUDE_MD)}")
    print(f"AGENTS.md : {len(agents):>6} byte  sha256={sha256_of(AGENTS_MD)}")

    fark = first_difference(claude, agents)
    if fark is None:
        print("SONUC: SENKRON -- iki dosya byte-birebir ayni.")
        return 0

    satir, c_metin, a_metin = fark
    print("SONUC: AYRISMA VAR.", file=sys.stderr)
    print(f"  Ilk farkli satir: {satir}", file=sys.stderr)
    print(f"    CLAUDE.md[{satir}]: {c_metin[:200]!r}", file=sys.stderr)
    print(f"    AGENTS.md[{satir}]: {a_metin[:200]!r}", file=sys.stderr)
    print(f"\nDUZELTME (tek komut): {FIX_COMMAND}", file=sys.stderr)
    return 1


def apply(from_agents: bool) -> int:
    kaynak, hedef = (AGENTS_MD, CLAUDE_MD) if from_agents else (CLAUDE_MD, AGENTS_MD)

    kaynak_data = _require_file(kaynak)
    hedef_data = _require_file(hedef)

    # FAIL-CLOSED kapisi: ici bosalmis/bozulmus kaynagi yaymayi reddet.
    eksik = missing_anchors(kaynak_data)
    if eksik:
        print(
            f"HATA: kaynak {kaynak.name} beklenen capalari TASIMIYOR: {eksik}\n"
            "Kopyalama YAPILMADI. Bu dosya ya bozulmus ya da bir kilitli deger\n"
            "sessizce silinmis demektir -- ikisi de CRITICAL'dir.",
            file=sys.stderr,
        )
        return 2

    print(f"KAYNAK : {kaynak.name}  sha256={sha256_of(kaynak)}")
    print(f"HEDEF  : {hedef.name}  sha256={sha256_of(hedef)} (yazim oncesi)")

    if kaynak_data == hedef_data:
        print("DEGISIKLIK YOK -- zaten senkron, dosyaya dokunulmadi.")
        return 0

    fark = first_difference(kaynak_data, hedef_data)
    if fark is not None:
        satir, k_metin, h_metin = fark
        print(f"Ilk farkli satir {satir}:")
        print(f"  {kaynak.name}[{satir}]: {k_metin[:160]!r}")
        print(f"  {hedef.name}[{satir}]: {h_metin[:160]!r}")

    # Byte-birebir yazim (metin modu YOK -- satir sonu/BOM donusumu olmasin).
    hedef.write_bytes(kaynak_data)

    yeni = sha256_of(hedef)
    print(f"HEDEF  : {hedef.name}  sha256={yeni} (yazim sonrasi)")

    if yeni != sha256_of(kaynak):
        print("HATA: yazim sonrasi sha256 ESLESMEDI.", file=sys.stderr)
        return 2

    print(f"SONUC: {kaynak.name} -> {hedef.name} UYGULANDI, sha256 ESIT.")
    return 0


def _hook_target(raw: str) -> Path | None:
    """Hook girdisindeki dosya yolu CLAUDE.md/AGENTS.md ise onu dondur.

    Baska her durumda None -> hook sessizce cikar. Bozuk/bos girdi de None
    dondurur: hook ASLA oturumu bozmaz.
    """

    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    aday: object = None
    for kaynak, alan in (("tool_input", "file_path"), ("tool_response", "filePath")):
        blok = payload.get(kaynak)
        if isinstance(blok, dict) and isinstance(blok.get(alan), str):
            aday = blok[alan]
            break
    if not isinstance(aday, str) or not aday:
        return None

    try:
        yol = Path(aday).absolute()  # K12: .resolve() YOK (Turkce-yol/subst tuzagi)
    except (OSError, ValueError):
        return None

    for hedef in (CLAUDE_MD, AGENTS_MD):
        # Windows'ta yol karsilastirmasi buyuk/kucuk harf duyarsizdir.
        if str(yol).casefold() == str(hedef).casefold():
            return hedef
    return None


def hook(raw_stdin: str) -> int:
    """PostToolUse hook modu: TESPIT eder, ASLA yazmaz. Her zaman 0 doner."""

    duzenlenen = _hook_target(raw_stdin)
    if duzenlenen is None:
        return 0  # ilgisiz dosya -> sessiz

    if not CLAUDE_MD.is_file() or not AGENTS_MD.is_file():
        eksik = [p.name for p in (CLAUDE_MD, AGENTS_MD) if not p.is_file()]
        _hook_uyar(
            f"SENKRON HOOK: {', '.join(eksik)} bulunamadi -- ikizleme kontrolu "
            "YAPILAMADI.",
            f"{', '.join(eksik)} diskte yok. `{FIX_COMMAND}` calistirilamaz; "
            "dosyanin neden kayboldugunu once insan dogrulamali.",
        )
        return 0

    claude = CLAUDE_MD.read_bytes()
    agents = AGENTS_MD.read_bytes()
    if claude == agents:
        return 0  # senkron -> sessiz (gurultu yapma)

    fark = first_difference(claude, agents)
    satir = fark[0] if fark else "?"

    if duzenlenen is CLAUDE_MD:
        # Kanonik kaynak degisti: yon net.
        oneri = f"Duzeltme (tek komut): {FIX_COMMAND}"
    else:
        # AGENTS.md uretilen kopyadir; yonu SECMEK insanin isi.
        oneri = (
            "DIKKAT: duzenlenen dosya AGENTS.md, yani URETILEN kopya. Yonu "
            "hook SECMEZ:\n"
            f"  (a) degisiklik CLAUDE.md'de de olmali -> once CLAUDE.md'yi ayni "
            f"sekilde duzenle, sonra: {FIX_COMMAND}\n"
            "  (b) duzeltme yalnizca AGENTS.md tarafina yazildiysa (NADIR) -> "
            "python tools/sync_claude_agents.py --apply --from-agents\n"
            f"  (c) degisiklik istenmiyorsa -> {FIX_COMMAND} onu geri alir."
        )

    _hook_uyar(
        f"CLAUDE.md / AGENTS.md AYRISTI (ilk farkli satir: {satir}). "
        f"{duzenlenen.name} yazildi, ikizi guncellenmedi.",
        "CLAUDE.md ve AGENTS.md byte-birebir ayni olmak zorundadir "
        f"(ilk farkli satir: {satir}). Az once {duzenlenen.name} yazildi ve iki "
        f"dosya artik ayri.\n{oneri}\n"
        "Hook bilerek otomatik --apply YAPMAZ: sessiz uzerine yazma, "
        "ayrismayi gizleyen hata sinifidir (2026-09-13 hardlink olcumu).",
    )
    return 0


def _hook_uyar(sistem_mesaji: str, model_baglami: str) -> None:
    """Claude Code hook sozlesmesine uygun JSON'i stdout'a yaz."""

    print(
        json.dumps(
            {
                "systemMessage": sistem_mesaji,
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": model_baglami,
                },
            },
            ensure_ascii=False,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CLAUDE.md ve AGENTS.md'yi byte-birebir senkron tutar.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Kopyalamayi GERCEKTEN yap (varsayilan: yalnizca --check).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Yalnizca kontrol et (varsayilan davranis).",
    )
    parser.add_argument(
        "--from-agents",
        action="store_true",
        help="Yonu ters cevir: AGENTS.md -> CLAUDE.md (NADIR).",
    )
    parser.add_argument(
        "--hook",
        action="store_true",
        help="Claude Code PostToolUse hook modu (stdin'den JSON okur, yazmaz).",
    )
    args = parser.parse_args(argv)

    if args.apply and args.check:
        parser.error("--apply ve --check birlikte kullanilamaz.")
    if args.from_agents and not args.apply:
        parser.error("--from-agents yalnizca --apply ile anlamlidir.")
    if args.hook and (args.apply or args.check or args.from_agents):
        parser.error("--hook tek basina kullanilir (yazma YAPMAZ).")

    if args.hook:
        try:
            raw = sys.stdin.read()
        except (OSError, ValueError):  # pragma: no cover -- stdin yoksa
            raw = ""
        return hook(raw)
    if args.apply:
        return apply(from_agents=args.from_agents)
    return check()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover -- eski Python/ozel stream
        pass
    raise SystemExit(main())
