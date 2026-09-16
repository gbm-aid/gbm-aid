#!/usr/bin/env python3
"""GBM-AID site sunucusunu baslatir -- Windows ve Linux'ta AYNI script.

NEDEN VAR (2026-09-16, Baris karari):
    FAISS indeksi Turkce karakterli yolda acilamiyor (K12 tuzagi). Streamlit
    demosu bunu kendi icinde `setup_faiss_ascii_paths()` ile asiyordu, API
    sunucusu asmiyordu -- gercek uvicorn kosusunda `/patient/{id}/similar`
    **503 FAISS indeksi okunamadi** verdi (olculdu).

    Cozum secimi: kodu ASCII yola tasimak (K12-C) YERINE, baslaticinin
    ortam degiskenini kurmasi. Gerekce: **Linux VPS'te bu sorun HIC YOK**
    (yol zaten ASCII), yani K12-C yalnizca bu gelistirme makinesi icin
    gerekliydi -- urun icin degil. Bu script iki ortamda da ayni sekilde
    calisir, boylece ayni is iki kez yapilmaz:
      - Windows : 8.3 kisa yolu hesaplar, env degiskenlerini kurar
      - Linux   : hicbir sey yapmaz (yol zaten ASCII) -- no-op

    VPS'te systemd birimi dogrudan bu script'i cagirir.

ISITMA CAGRISI:
    Ilk `/predict` istegi SHAP arka planini Supabase'den cekiyor; olculen
    sure 19 sn ile 540 sn arasinda degisiyor (ag/DB'ye bagli). `--warmup`
    ile sunucu acildiktan sonra bir kez cagrilir, boylece ILK KULLANICI
    beklemez.

KULLANIM:
    python tools/run_site.py                    # 127.0.0.1:8000
    python tools/run_site.py --port 8011 --warmup UCSF-PDGM-167
    python tools/run_site.py --host 0.0.0.0 --port 8000 --warmup UCSF-PDGM-167
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

KOD_KOKU = Path(__file__).absolute().parents[1]  # .resolve() DEGIL -- K12


# ---------------------------------------------------------------------------
# FAISS yol kalkani
# ---------------------------------------------------------------------------
def _kisa_yol(yol: Path) -> str | None:
    """Windows 8.3 kisa yolunu dondurur; basarisizsa None."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    fn = ctypes.windll.kernel32.GetShortPathNameW
    fn.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    fn.restype = wintypes.DWORD
    tampon = ctypes.create_unicode_buffer(1024)
    n = fn(str(yol), tampon, 1024)
    if n == 0 or n >= 1024:
        return None
    kisa = tampon.value
    return kisa if kisa.isascii() else None


def faiss_yollarini_hazirla() -> dict:
    """FAISS indeks dizinlerini ASCII yola sabitler. Linux'ta no-op."""
    rapor: dict = {"platform": os.name, "uygulandi": False, "notlar": []}

    dizinler = {
        "GBMAID_FAISS_CLINICAL_RADIOMICS_INDEX_DIR":
            KOD_KOKU / "artifacts" / "week3" / "faiss_index",
        "GBMAID_FAISS_MOLECULAR_OMICS_INDEX_DIR":
            KOD_KOKU / "artifacts" / "week3" / "molecular_omics",
    }

    for degisken, dizin in dizinler.items():
        if os.environ.get(degisken):
            rapor["notlar"].append("%s zaten tanimli, DOKUNULMADI" % degisken)
            continue
        if not dizin.is_dir():
            rapor["notlar"].append("%s -> dizin YOK (%s), atlandi" % (degisken, dizin.name))
            continue
        if str(dizin).isascii():
            # Linux ve ASCII yollu Windows: sorun yok, degisken gereksiz.
            rapor["notlar"].append("%s -> yol zaten ASCII, gerek yok" % degisken)
            continue
        kisa = _kisa_yol(dizin)
        if kisa:
            os.environ[degisken] = kisa
            rapor["uygulandi"] = True
            rapor["notlar"].append("%s = %s (8.3 kisa yol)" % (degisken, kisa))
        else:
            rapor["notlar"].append(
                "🔴 %s -> kisa yol URETILEMEDI. FAISS 503 verecek. "
                "Cozum: `subst X: \"%s\"` ile ASCII surucu olustur." % (degisken, KOD_KOKU)
            )
    return rapor


# ---------------------------------------------------------------------------
# Isitma
# ---------------------------------------------------------------------------
def _isit(taban: str, hasta: str, zaman_asimi: int) -> None:
    """Sunucu acilinca bir kez `/predict` cagirir -- ilk kullanici beklemesin."""
    for _ in range(90):
        time.sleep(2)
        try:
            with urllib.request.urlopen(taban + "/health", timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:
            continue
    else:
        print("[isitma] saglik kontrolu gecilemedi, isitma ATLANDI", flush=True)
        return

    print("[isitma] %s icin SHAP arka plani onbellege aliniyor..." % hasta, flush=True)
    t0 = time.time()
    try:
        req = urllib.request.Request(
            taban + "/predict/" + hasta, data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=zaman_asimi) as r:
            r.read()
        print("[isitma] TAMAM -- %.1f sn. Sonraki istekler hizli." % (time.time() - t0), flush=True)
    except Exception as exc:
        print("[isitma] BASARISIZ (%s: %s) -- sunucu yine de calisiyor, "
              "ilk kullanici bekleyecek." % (type(exc).__name__, str(exc)[:120]), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="GBM-AID site sunucusu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--warmup", metavar="HASTA_ID", default=None,
                    help="sunucu acilinca bu hasta ile bir kez /predict cagir")
    ap.add_argument("--warmup-timeout", type=int, default=900)
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    rapor = faiss_yollarini_hazirla()
    print("=" * 66)
    print("GBM-AID site sunucusu")
    print("  kod koku : %s" % KOD_KOKU)
    print("  FAISS    : %s" % ("ASCII kalkani UYGULANDI" if rapor["uygulandi"]
                               else "kalkan gerekmedi / uygulanamadi"))
    for n in rapor["notlar"]:
        print("             - %s" % n)
    print("  adres    : http://%s:%d/app/" % (args.host, args.port))
    print("  API dok. : http://%s:%d/docs" % (args.host, args.port))
    print("=" * 66, flush=True)

    if args.warmup:
        threading.Thread(
            target=_isit,
            args=("http://127.0.0.1:%d" % args.port, args.warmup, args.warmup_timeout),
            daemon=True,
        ).start()

    # 🔴 TEK ISCI -- SHAP arka plan onbellegi surec-icidir; coklu isci hem
    #    bellegi hem soguk baslangici carpar (2026-09-16 olcumu).
    #
    # 2026-09-16 (klasor yapisi): `api/` ve `pipeline/` artik `backend/`
    # altinda. Import ADLARI DEGISMEDI (`api.main:app`, `from pipeline.x`),
    # bu yuzden uvicorn'un AYRI surecine `backend` arama yolu ORTAM
    # DEGISKENIYLE gecirilir -- `cwd` tek basina yetmez, cunku `-m uvicorn`
    # sys.path[0]'a uvicorn'un kendi dizinini koyar.
    ortam = dict(os.environ)
    mevcut = ortam.get("PYTHONPATH", "")
    backend_yolu = str(KOD_KOKU / "backend")
    if backend_yolu not in mevcut.split(os.pathsep):
        ortam["PYTHONPATH"] = (
            backend_yolu + (os.pathsep + mevcut if mevcut else "")
        )

    return subprocess.call(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", args.host, "--port", str(args.port),
         "--workers", "1", "--log-level", args.log_level],
        cwd=str(KOD_KOKU),
        env=ortam,
    )


if __name__ == "__main__":
    raise SystemExit(main())
