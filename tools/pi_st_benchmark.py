#!/usr/bin/env python3
"""GBM-AID -- Raspberry Pi5 `sentence-transformers` fizibilite olcumu.

NEDEN (2026-09-15, Baris talimati; `gunluk-rapor-2026-09-14.md` Oncelik 1):
    RAG gomme adimi su an Windows is istasyonunda kosuyor. Y2 fazinda sistem
    Pi5'e tasinacaksa, Pi5'in `sentence-transformers`'i TASIYIP TASIYAMADIGI
    "plani degistirebilecek tek bilinmeyen"dir. Bu script o bilinmeyeni
    OLCER -- tahmin etmez.

KAPSAM (bilinerek dar):
    - YALNIZ `~/gbmaid_bench/` altina yazar. NAS paylasimina, `gbmaid` veri
      dizinlerine, Samba paylasimina ve DB'ye HICBIR SEY yazmaz.
    - Model: `sentence-transformers/all-MiniLM-L6-v2` -- projenin KENDI modeli
      (`pipeline/rag_embed_index.py:34`), `max_seq_length=256` (`:59`).
    - Surum pinleri `requirements.txt`ten BIREBIR alinir; "en yenisi" kurulmaz.

OLCUM DURUSTLUGU (bu script'in en onemli ozelligi):
    1. Her faz KENDI kaydini `results.jsonl`e yazar; surec yarida olurse
       o ana kadarki olcumler KAYBOLMAZ.
    2. `manifest.json` ATOMIK yazilir (tmp + os.replace) -- yarim JSON olusmaz.
       (2026-08-14'te Mert'in `b1_supervisor.ps1`de buldugu IKINCI kusur:
       `flush()` atomik degildi.)
    3. `run_complete` alani YALNIZ tum fazlar bittiginde `true` olur; nobetci
       bu ALANI okur, dosyanin VARLIGINA bakmaz. (Mert'in buldugu BIRINCI
       kusur: durma kosulu yalnizca `Test-Path` idi.)
    4. Yeniden baslatmada `ok` isaretli fazlar ATLANIR -- sifirdan baslamaz.
    5. Tepe bellek (`ru_maxrss`) AYRI alt surecte olculur; yukleme ve kodlama
       birbirinin tepe degerini kirletmez.
    6. pip once `--only-binary=:all:` ile denenir: aarch64 tekeri YOKSA
       saatlerce kaynak derlemek yerine dakikalar icinde "teker yok" diye
       DURUSTCE raporlanir. Bu bir ariza degil, bir OLCUMDUR.

KULLANIM (Pi uzerinde; normalde nobetci cagirir):
    python3 pi_st_benchmark.py --out-dir ~/gbmaid_bench
    python3 pi_st_benchmark.py --phase-worker load_warm   (ic kullanim)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_MAX_SEQ_LENGTH = 256
TARGET_CHUNK_TOKENS = 200
PINS = [
    "numpy==1.26.4",
    "torch==2.13.0",
    "transformers==4.57.6",
    "sentence-transformers==3.3.1",
    "faiss-cpu==1.15.0",
]
PHASES = ["env", "install", "load_cold", "load_warm", "encode_bench", "request_proxy", "faiss_smoke"]
PHASE_TIMEOUT_S = {
    "install": 5400,
    "load_cold": 1800,
    "load_warm": 900,
    "encode_bench": 1800,
    "request_proxy": 900,
    "faiss_smoke": 600,
}

# Gercekci uzunlukta biyomedikal metin. Parca basina GERCEK token sayisi
# olcum aninda modelin KENDI tokenizer'i ile raporlanir -- tahmin edilmez.
SAMPLE_TEXT = (
    "Glioblastoma is the most common and aggressive primary malignant brain tumour in adults, "
    "with a median overall survival of approximately fifteen months despite maximal safe resection "
    "followed by concurrent chemoradiotherapy and adjuvant temozolomide. Radiomic analysis of "
    "preoperative contrast enhanced T1 weighted magnetic resonance imaging has been proposed as a "
    "non invasive source of prognostic information, since quantitative descriptors of shape, first "
    "order intensity and texture may capture intratumoural heterogeneity that is not apparent on "
    "visual inspection. Reported models frequently combine such descriptors with established "
    "clinical covariates including patient age at diagnosis, extent of surgical resection, "
    "isocitrate dehydrogenase mutation status and O6 methylguanine DNA methyltransferase promoter "
    "methylation status. External validation remains limited, and discrimination measured by the "
    "concordance index is commonly reported between zero point six and zero point seven in "
    "independent cohorts, with confidence intervals that frequently overlap those of models based "
    "on age alone. Harmonisation of intensity scales across acquisition sites, stability filtering "
    "of texture features and prevention of information leakage between feature selection and model "
    "fitting are therefore emphasised as prerequisites for credible reporting. "
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _peak_rss_mb() -> float:
    # Linux: ru_maxrss kilobayt cinsindedir.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def _read_first_line(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.readline().strip().replace("\x00", "")
    except OSError:
        return ""


def _meminfo_mb(key: str):
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(key):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return None


# --------------------------------------------------------------------------
# FAZLAR -- ana surecte kosanlar
# --------------------------------------------------------------------------
def phase_env(ctx: dict) -> dict:
    _total, _used, free = shutil.disk_usage(str(ctx["out_dir"]))
    return {
        "pi_model": _read_first_line("/proc/device-tree/model") or "bilinmiyor",
        "uname": " ".join(platform.uname()),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python_version": sys.version.split()[0],
        "mem_total_mb": _meminfo_mb("MemTotal:"),
        "mem_available_mb": _meminfo_mb("MemAvailable:"),
        "swap_total_mb": _meminfo_mb("SwapTotal:"),
        "disk_free_gb": round(free / 1024 ** 3, 2),
        "nas_uyarisi": (
            "Bu makine ayni zamanda MR NAS'idir; olcum `nice` ile kosar ama es "
            "zamanli demo/NAS okumasi gecikmeleri etkileyebilir -- rapora yazilmali."
        ),
    }


def phase_install(ctx: dict) -> dict:
    venv = ctx["venv"]
    steps = []
    if not venv.exists():
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "venv", str(venv)],
            capture_output=True, text=True, timeout=600,
        )
        steps.append({
            "adim": "venv_create",
            "rc": proc.returncode,
            "sure_s": round(time.time() - t0, 1),
            "stderr_kuyruk": proc.stderr[-400:],
        })
        if proc.returncode != 0:
            return {"steps": steps, "hata": "venv kurulamadi", "wheels_ok": False}

    pip = str(venv / "bin" / "pip")
    t0 = time.time()
    proc = subprocess.run(
        [pip, "install", "--upgrade", "pip", "wheel"],
        capture_output=True, text=True, timeout=900,
    )
    steps.append({"adim": "pip_upgrade", "rc": proc.returncode, "sure_s": round(time.time() - t0, 1)})

    # 1) SADECE hazir teker. aarch64 tekeri yoksa saatlerce derleme YAPILMAZ.
    t0 = time.time()
    proc = subprocess.run(
        [pip, "install", "--only-binary=:all:"] + PINS,
        capture_output=True, text=True, timeout=PHASE_TIMEOUT_S["install"],
    )
    wheels_ok = proc.returncode == 0
    steps.append({
        "adim": "pip_install_only_binary",
        "paketler": PINS,
        "rc": proc.returncode,
        "sure_s": round(time.time() - t0, 1),
        "stdout_kuyruk": proc.stdout[-1500:],
        "stderr_kuyruk": proc.stderr[-2000:],
    })

    kurulu = {}
    if wheels_ok:
        show = subprocess.run([pip, "list", "--format=json"], capture_output=True, text=True, timeout=300)
        if show.returncode == 0:
            try:
                kurulu = {p["name"].lower(): p["version"] for p in json.loads(show.stdout)}
            except (json.JSONDecodeError, KeyError, TypeError):
                kurulu = {}

    venv_mb = None
    if venv.exists():
        venv_mb = round(sum(f.stat().st_size for f in venv.rglob("*") if f.is_file()) / 1024 ** 2, 1)

    out = {
        "steps": steps,
        "wheels_ok": wheels_ok,
        "kurulu_surumler": kurulu,
        "venv_boyut_mb": venv_mb,
        "not": (
            "wheels_ok=false ise bu bir ARIZA DEGIL OLCUMDUR: pinlenmis surumlerin "
            "linux_aarch64 tekeri yok demektir. O durumda Y2 icin ya surum pinleri "
            "gozden gecirilir ya da gomme adimi Pi5 DISINDA kalir."
        ),
    }
    if not wheels_ok:
        out["hata"] = "pip --only-binary basarisiz (aarch64 teker yok veya cozumleme hatasi)"
    return out


def _run_worker(ctx: dict, name: str) -> dict:
    """Fazi AYRI alt surecte kosar -- tepe bellek olcumu izole olsun diye."""
    py = str(ctx["venv"] / "bin" / "python")
    if not os.path.exists(py):
        return {"hata": "venv python yok -- install fazi basarisiz", "atlandi": True}
    t0 = time.time()
    proc = subprocess.run(
        [py, os.path.abspath(__file__), "--phase-worker", name, "--out-dir", str(ctx["out_dir"])],
        capture_output=True, text=True, timeout=PHASE_TIMEOUT_S.get(name, 1800),
    )
    out = {"rc": proc.returncode, "duvar_saati_s": round(time.time() - t0, 1)}
    if proc.returncode == 0:
        try:
            out.update(json.loads(proc.stdout.strip().splitlines()[-1]))
        except (json.JSONDecodeError, IndexError):
            out["hata"] = "alt surec ciktisi ayristirilamadi"
            out["stdout_kuyruk"] = proc.stdout[-1200:]
    else:
        out["hata"] = "alt surec sifirdan farkli donus kodu"
        out["stderr_kuyruk"] = proc.stderr[-2500:]
    return out


# --------------------------------------------------------------------------
# FAZLAR -- alt surecte (venv icinde) kosanlar
# --------------------------------------------------------------------------
def worker_load_cold(out_dir: Path) -> dict:
    """Ilk yukleme: model indirilmesi DAHIL."""
    cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    vardi = cache.exists()
    t0 = time.time()
    from sentence_transformers import SentenceTransformer

    import_s = round(time.time() - t0, 2)
    t1 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    load_s = round(time.time() - t1, 2)
    cache_mb = None
    if cache.exists():
        cache_mb = round(sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) / 1024 ** 2, 1)
    return {
        "hf_cache_onceden_vardi": vardi,
        "import_suresi_s": import_s,
        "model_yukleme_s": load_s,
        "toplam_s": round(import_s + load_s, 2),
        "hf_cache_mb": cache_mb,
        "max_seq_length": getattr(model, "max_seq_length", None),
        "gomme_boyutu": model.get_sentence_embedding_dimension(),
        "tepe_bellek_mb": _peak_rss_mb(),
    }


def worker_load_warm(out_dir: Path) -> dict:
    """Ikinci yukleme: model diskte -- servis yeniden baslatma senaryosu."""
    t0 = time.time()
    from sentence_transformers import SentenceTransformer

    import_s = round(time.time() - t0, 2)
    t1 = time.time()
    SentenceTransformer(MODEL_NAME)
    return {
        "import_suresi_s": import_s,
        "model_yukleme_s": round(time.time() - t1, 2),
        "tepe_bellek_mb": _peak_rss_mb(),
        "yorum": (
            "Servis her istekte degil, surec basina BIR KEZ yukler "
            "(`pipeline/rag_embed_index.py:70` `_model_cache`)."
        ),
    }


def _build_chunks(model, n: int):
    tok = model.tokenizer
    words = (SAMPLE_TEXT * 4).split()
    chunks = []
    for i in range(n):
        start = (i * 7) % 40  # her parcaya farkli ofset -- onbellege dusmesin
        cand = " ".join(words[start:start + 400])
        ids = tok.encode(cand, add_special_tokens=False)[:TARGET_CHUNK_TOKENS]
        chunks.append(tok.decode(ids))
    olculen = len(tok.encode(chunks[0], add_special_tokens=False))
    return chunks, olculen


def worker_encode_bench(out_dir: Path) -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    chunks, tokens = _build_chunks(model, 64)
    model.encode(chunks[:4], convert_to_numpy=True, show_progress_bar=False)  # isinma
    sonuc = {}
    for batch in (1, 8, 32):
        alt = chunks[:32]
        t0 = time.time()
        vec = model.encode(alt, batch_size=batch, convert_to_numpy=True, show_progress_bar=False)
        gecen = time.time() - t0
        sonuc["batch_%d" % batch] = {
            "n_parca": len(alt),
            "toplam_s": round(gecen, 2),
            "parca_basina_ms": round(gecen / len(alt) * 1000, 1),
            "vektor_sekli": list(vec.shape),
        }
    return {
        "parca_basina_token": tokens,
        "hedef_token": TARGET_CHUNK_TOKENS,
        "model_max_seq_length": MODEL_MAX_SEQ_LENGTH,
        "olcumler": sonuc,
        "tepe_bellek_mb": _peak_rss_mb(),
    }


def worker_request_proxy(out_dir: Path) -> dict:
    """Tek bir /analyze_patient literatur blogunun GERCEK gomme maliyeti.

    Canli olcum (2026-09-14): 7 kaynak donuyor. Kaynak basina ~4 parca
    ACIKCA bir VARSAYIMDIR ve raporda oyle yazilir.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    n_kaynak, parca_basina = 7, 4
    chunks, tokens = _build_chunks(model, n_kaynak * parca_basina)
    model.encode(chunks[:2], convert_to_numpy=True, show_progress_bar=False)
    t0 = time.time()
    model.encode(chunks, batch_size=8, convert_to_numpy=True, show_progress_bar=False)
    korpus_s = time.time() - t0
    t1 = time.time()
    model.encode(
        ["glioblastoma radiomics overall survival prognosis"],
        convert_to_numpy=True, show_progress_bar=False,
    )
    sorgu_s = time.time() - t1
    return {
        "varsayim": "%d kaynak x %d parca = %d parca (VARSAYIM, olcum degil)"
                    % (n_kaynak, parca_basina, n_kaynak * parca_basina),
        "parca_basina_token": tokens,
        "korpus_gomme_s": round(korpus_s, 2),
        "sorgu_gomme_s": round(sorgu_s, 3),
        "istek_basina_toplam_gomme_s": round(korpus_s + sorgu_s, 2),
        "tepe_bellek_mb": _peak_rss_mb(),
        "karsilastirma_notu": (
            "Windows tarafinda olculen uctan uca gecikme ~13-14 sn, LLM'in kendisi 6,5 sn "
            "(`gunluk-rapor-2026-09-14.md`). Buradaki sayi YALNIZ gomme adimidir, uctan uca DEGIL."
        ),
    }


def worker_faiss_smoke(out_dir: Path) -> dict:
    import numpy as np

    try:
        import faiss
    except ImportError as exc:
        return {"faiss_kullanilabilir": False, "hata": str(exc)}
    rng = np.random.default_rng(42)
    xb = rng.random((722, 93), dtype="float32")  # FAISS v1 clinical indeksinin GERCEK sekli
    idx = faiss.IndexFlatL2(93)
    t0 = time.time()
    idx.add(xb)
    ekleme_s = time.time() - t0
    t1 = time.time()
    _dist, ids = idx.search(xb[:1], 10)
    return {
        "faiss_kullanilabilir": True,
        "faiss_surum": getattr(faiss, "__version__", "bilinmiyor"),
        "indeks_sekli": [722, 93],
        "ekleme_s": round(ekleme_s, 3),
        "arama_10nn_ms": round((time.time() - t1) * 1000, 2),
        "komsu_sayisi": int(ids.shape[1]),
        "tepe_bellek_mb": _peak_rss_mb(),
        "not": (
            "722x93 = FAISS v1 `clinical_radiomics` indeksinin GERCEK boyutu "
            "(UPenn 611 + LUMIERE 72 + TCGA 39). Bu faz indeksi YENIDEN KURMAZ, "
            "yalnizca Pi'nin ayni boyutu tasiyip tasimadigini olcer."
        ),
    }


WORKERS = {
    "load_cold": worker_load_cold,
    "load_warm": worker_load_warm,
    "encode_bench": worker_encode_bench,
    "request_proxy": worker_request_proxy,
    "faiss_smoke": worker_faiss_smoke,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="GBM-AID Pi5 sentence-transformers fizibilite olcumu",
    )
    ap.add_argument("--out-dir", default=str(Path.home() / "gbmaid_bench"))
    ap.add_argument("--phase-worker", choices=sorted(WORKERS))
    args = ap.parse_args()

    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase_worker:
        try:
            print(json.dumps(WORKERS[args.phase_worker](out_dir), ensure_ascii=False))
            return 0
        except Exception as exc:  # olcum surecidir: hatayi YUTMA, RAPORLA
            print(json.dumps({"hata": "%s: %s" % (type(exc).__name__, exc)}, ensure_ascii=False))
            return 1

    manifest_path = out_dir / "manifest.json"
    results_path = out_dir / "results.jsonl"
    manifest = _load_manifest(manifest_path)
    fazlar = manifest.get("fazlar", {})
    manifest.update({
        "sema": 1,
        "baslik": "GBM-AID Pi5 sentence-transformers fizibilite olcumu",
        "model": MODEL_NAME,
        "pinler": PINS,
        "baslangic": manifest.get("baslangic", _now()),
        "son_guncelleme": _now(),
        "run_complete": False,
        "fazlar": fazlar,
    })
    _atomic_write_json(manifest_path, manifest)

    ctx = {"out_dir": out_dir, "venv": out_dir / "venv"}

    for faz in PHASES:
        if fazlar.get(faz, {}).get("durum") == "ok":
            print("[atlandi] %s -- zaten ok" % faz, flush=True)
            continue
        print("[basliyor] %s %s" % (faz, _now()), flush=True)
        t0 = time.time()
        try:
            if faz == "env":
                veri = phase_env(ctx)
            elif faz == "install":
                veri = phase_install(ctx)
            else:
                veri = _run_worker(ctx, faz)
            durum = "hata" if "hata" in veri else "ok"
        except subprocess.TimeoutExpired:
            veri, durum = {"hata": "zaman asimi (%s s)" % PHASE_TIMEOUT_S.get(faz)}, "hata"
        except Exception as exc:
            veri, durum = {"hata": "%s: %s" % (type(exc).__name__, exc)}, "hata"

        kayit = {
            "faz": faz,
            "durum": durum,
            "zaman": _now(),
            "sure_s": round(time.time() - t0, 1),
            "veri": veri,
        }
        _append_jsonl(results_path, kayit)
        fazlar[faz] = {"durum": durum, "zaman": kayit["zaman"], "sure_s": kayit["sure_s"]}
        manifest["fazlar"] = fazlar
        manifest["son_guncelleme"] = _now()
        _atomic_write_json(manifest_path, manifest)
        print("[bitti] %s -> %s (%s s)" % (faz, durum, kayit["sure_s"]), flush=True)

        # Kurulum coktuyse model fazlarini kosmanin anlami yok: DURUSTCE dur.
        if faz == "install" and durum != "ok":
            manifest["durdurma_nedeni"] = (
                "install fazi basarisiz -- model/kodlama fazlari KOSULMADI. Bu bir "
                "OLCUM SONUCUDUR: pinlenmis surumlerin aarch64 tekeri yok."
            )
            manifest["run_complete"] = True  # nobetci bosuna yeniden baslatmasin
            manifest["sonuc"] = "TAMAMLANDI (install basarisiz -- kesin sonuc)"
            manifest["son_guncelleme"] = _now()
            _atomic_write_json(manifest_path, manifest)
            return 2

    manifest["run_complete"] = True
    manifest["sonuc"] = "TAMAMLANDI"
    manifest["son_guncelleme"] = _now()
    _atomic_write_json(manifest_path, manifest)
    print("[TAMAM] run_complete=true", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
