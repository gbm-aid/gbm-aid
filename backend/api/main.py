import concurrent.futures
import logging
import os
import secrets
import threading
import time
from collections import defaultdict
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import psycopg2.extras
from db_connection import get_connection, load_project_environment
from pipeline.harmonization import (
    ZScoreStatisticsNotFoundError,
    apply_n4_bias_correction,
    apply_zscore_normalization,
)
from pipeline.rag_cache import check_upstash_connectivity, load_upstash_config
from pipeline.segmentation import resolve_nas_path

# `POST /predict/{patient_id}` -- ayrı dosya (api/predict.py, backend-agent
# 2026-08-18). İskelet kapsamı/kısıtları o dosyanın docstring'inde.
from api.predict import router as predict_router

# `GET /patient/{patient_id}/similar` -- ayrı dosya (api/similar.py,
# backend-agent 2026-08-18, plan.txt:452-454). Kapsam/kısıtlar o dosyanın
# docstring'inde.
from api.similar import router as similar_router

# `POST /analyze_patient` -- ayrı dosya (api/analyze_patient.py,
# backend-agent 2026-09-11, Hafta 5 kapanışı, plan.txt:165-171/523-529).
# predict_router + similar_router'ın PUBLIC fonksiyonlarını zincirler,
# hiçbirini KOPYALAMAZ. Kapsam/kısıtlar o dosyanın docstring'inde.
from api.analyze_patient import router as analyze_patient_router

# `GET /patients` -- ayrı dosya (api/patients.py, backend-agent-W1,
# paralel-36, 2026-09-15/16, Barış onayı "SITE AŞAMA 1"). Kohort
# Gezgini'nin TEK besleyicisi. Kapsam/kısıtlar o dosyanın docstring'inde.
from api.patients import router as patients_router

# `GET /patient/{patient_id}/mr_slice` -- ayrı dosya (api/mr_slice.py,
# backend-agent-W1, paralel-36, 2026-09-15/16). Kapsam/kısıtlar o dosyanın
# docstring'inde.
from api.mr_slice import router as mr_slice_router

# `GET /patient/{patient_id}/volume_series` -- ayrı dosya
# (api/volume_series.py, backend-agent-X1, paralel-38, 2026-09-16, Barış
# onayı "SITE AŞAMA 3-A"). Hasta Geçmişi ekranının longitudinal hacim-zaman
# eğrisi. Kapsam/kısıtlar o dosyanın docstring'inde.
from api.volume_series import router as volume_series_router

# `GET /model_performance` -- ayrı dosya (api/model_performance.py,
# backend-agent-X1, paralel-38, 2026-09-16). Sistem Hakkında ekranındaki 8
# kollu Cox tablosu + XGBoost katmanı. Kapsam/kısıtlar o dosyanın
# docstring'inde.
from api.model_performance import router as model_performance_router

# `GET /model_curves` -- ayrı dosya (api/model_curves.py, backend-agent-Y1,
# paralel-40, 2026-09-16, Barış onayı). Sistem > Model Performansı
# ekranındaki kalibrasyon eğrisi + katsayı orman grafiği. Kapsam/kısıtlar o
# dosyanın docstring'inde.
from api.model_curves import router as model_curves_router

try:
    load_project_environment()
except FileNotFoundError:
    # Import ve birim testleri DB bağlantısı açmaz. Gerçek bağlantı çağrısı
    # eksik .env için merkezi modülde açık hata verecektir.
    pass

app = FastAPI(title="GBM-AID API")
app.include_router(predict_router)
app.include_router(similar_router)
app.include_router(analyze_patient_router)
app.include_router(patients_router)
app.include_router(mr_slice_router)
app.include_router(volume_series_router)
app.include_router(model_performance_router)
app.include_router(model_curves_router)

_auth_logger = logging.getLogger("api.auth")
_static_logger = logging.getLogger("api.static")

# =====================================================================
# STATİK SİTE SUNUMU -- `gbm-aid mert/web/` (paralel-37, AYRI bir ajan,
# bu görevde DOKUNULMADI). Görev talimatı: aynı origin'den sun ki CORS'a
# GEREK KALMASIN (`CORSMiddleware` BİLİNÇLİ OLARAK EKLENMEDİ).
#
# MOUNT YOLU KARARI -- `/app` (KÖK ("/") DEĞİL), ÖLÇÜLDÜ (varsayılmadı):
# Starlette `Mount`/`APIRoute` listesi KAYIT SIRASINA göre eşleşir. Bu
# mount ifadesi dosyanın ÜST kısmında (aşağıdaki `@app.get("/")`,
# `/health`, vb. tüm route'lardan ÖNCE) çalışıyor -- eğer KÖK'e ("/")
# mount edilseydi, bir `TestClient` deneyiyle DOĞRULANDI ki Starlette'in
# `Mount("/")`ı SONRA tanımlanan `/health` gibi TÜM route'ları
# GÖLGELERDİ (`/health` isteği health-check'e hiç ULAŞAMADAN mount'un
# kendi 404'üne düşüyordu) -- SPA fallback'inin 404'ü sessizce 200'e
# çevirmesi DEĞİL (bu iddia test edildi ve YANLIŞ çıktı, `html=True`
# gerçekten eksik dosyalar için düzgün 404 JSON döndürüyor), asıl risk
# TÜM SONRAKİ ROUTE'LARIN HİÇ ÇALIŞTIRILAMAMASIYDI. `/app` altına mount
# etmek bunu TAMAMEN önler: `/app` hiçbir mevcut/gelecekteki API
# route'uyla (`/patients`, `/patient/{id}/...`, `/health`, `/similar`,
# `/predict/{id}`, `/analyze_patient`) ÖNEK ÇAKIŞMASI yaşamaz. Frontend'in
# kendi `web/app/api.js::BASE_URL=''` varsayımı BUNU ETKİLEMEZ --
# `fetch('/patients')` gibi MUTLAK-KÖK çağrılar sitenin HANGİ alt yoldan
# (`/app/`) servis edildiğinden BAĞIMSIZDIR (aynı origin, farklı path).
#
# `web/` dizini henüz yok/boş OLABİLİR (paralel-37 orada eşzamanlı
# çalışıyor, AKTIF-GOREVLER.md) -- mount PATLAMAZ, zarifçe ATLANIR ve
# AÇIKÇA loglanır (sessiz atlama YOK).
_WEB_DIR = Path(__file__).resolve().parents[2] / "frontend"  # 2026-09-16: backend/ altina tasindi -> bir seviye daha yukari
if _WEB_DIR.is_dir():
    app.mount("/app", StaticFiles(directory=str(_WEB_DIR), html=True), name="web-app")
else:
    _static_logger.warning(
        "Statik site dizini bulunamadı, `/app` mount edilmedi: %s "
        "(paralel-37 henüz yazmamış olabilir -- bu bir HATA DEĞİL, "
        "siteye ihtiyaç duyulduğunda dizin oluşunca süreç yeniden "
        "başlatılmalı).",
        _WEB_DIR,
    )
_access_log_logger = logging.getLogger("api.access_log")


def _api_key_env() -> str | None:
    """`GBMAID_API_KEY` -- boş string de "tanımsız" sayılır (yanlışlıkla
    `GBMAID_API_KEY=` bırakılmış bir `.env` satırı auth'u sessizce
    devre dışı bırakmamalı gibi görünse de, CLAUDE.md ".env'e anahtar
    GÖMME" ilkesiyle tutarlı tek okuma noktası burasıdır -- gerçek
    değer HER ZAMAN ortam değişkeninden okunur, koda GÖMÜLMEZ)."""

    value = os.environ.get("GBMAID_API_KEY")
    return value if value else None


if _api_key_env() is None:
    # 2026-09-11 (G2 görev talimatı): auth devre dışı davranışı SESSİZCE
    # olmamalı -- modül yüklenirken (uygulama başlarken) AÇIKÇA loglanır.
    _auth_logger.warning(
        "GBMAID_API_KEY tanımlı değil -- Auth DEVRE DIŞI (geliştirme modu). "
        "TÜM endpoint'ler X-API-Key olmadan erişilebilir. Üretime "
        "geçmeden ÖNCE bu ortam değişkeni tanımlanmalı."
    )


def _is_authorized(provided_key: str | None) -> bool:
    """`GBMAID_API_KEY` tanımlı DEĞİLSE (geliştirme modu) HER ZAMAN True
    döner -- bu davranış modül yüklenirken YUKARIDA AYRICA loglanır.
    Tanımlıysa `provided_key` ile TAM (case-sensitive) eşleşme gerekir.

    Karşılaştırma `secrets.compare_digest` ile ZAMANLAMA-SALDIRISINA
    DAYANIKLI yapılır (Codex çapraz inceleme, 2026-09-11 HIGH bulgusu):
    düz `==` string karşılaştırması Python'da ilk uyuşmayan karakterde
    erken çıkar, bu da doğru anahtarın uzunluğunu/ön ekini byte-byte
    zamanlama farkıyla dışarı sızdırabilir. `provided_key` `None`
    olabilir (header hiç gönderilmemiş) -- `compare_digest` `None` ile
    çağrılırsa `TypeError` fırlatır, bu yüzden `None` için ÖNCE açık
    guard konur (istisna asla dışarı sızmaz)."""

    required = _api_key_env()
    if required is None:
        return True
    if provided_key is None:
        return False
    return secrets.compare_digest(provided_key.encode(), required.encode())


def _extract_patient_id_from_path(path: str) -> str | None:
    """`/predict/{id}`, `/patient/{id}/...` gibi yollardan `patient_id`'yi
    best-effort çıkarır. `/analyze_patient` (patient_id GÖVDEDE, yolda
    DEĞİL) için `None` döner -- İSTEK GÖVDESİ access_log'a YAZILMAZ (görev
    talimatı: "İSTEK GÖVDESİ/hasta verisi YAZILMAZ"), bu yüzden
    `analyze_patient` çağrılarında `patient_id` sütunu bilinçli olarak
    NULL kalır."""

    parts = [segment for segment in path.split("/") if segment]
    if not parts:
        return None
    if parts[0] in ("predict", "patient") and len(parts) >= 2:
        return parts[1]
    return None


def _write_access_log(*, action: str, patient_id: str | None) -> None:
    """`access_log` tablosuna TEKİL bir satır ekler (append-only --
    UPDATE/DELETE YOK, şema değişikliği YOK). Gerçek şema (2026-09-11
    canlı `information_schema.columns` ile doğrulandı): `log_id`
    (serial), `user_id`, `user_role`, `patient_id`, `action`,
    `accessed_at` (`DEFAULT now()`) -- ne `endpoint`/`status_code` ne de
    istek gövdesi için ayrı bir kolon VAR; bu yüzden `action` metni
    `"{method} {path} -> {status_code}"` biçiminde HEM endpoint'i HEM
    status_code'u taşır. `user_id`/`user_role` bu prototipte gerçek bir
    kimlik doğrulamadan (RBAC) gelmediği için bilinçli olarak `NULL`
    bırakılır -- bkz. modülün Auth notu (v45.txt Bölüm 4.4: "Prototip
    aşamasında minimal (tek kullanıcılı demo)").

    DB'ye erişilemezse İSTEĞİ DÜŞÜRMEZ (graceful) ama SESSİZ de
    değildir -- `_access_log_logger`'a warning basar (CLAUDE.md: sessiz
    başarısızlık yok)."""

    try:
        conn = get_connection(readonly=False)
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO access_log (user_id, user_role, patient_id, action) "
                "VALUES (%s, %s, %s, %s)",
                (None, None, patient_id, action),
            )
            conn.commit()
            cur.close()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 -- graceful, ama SESSİZ değil
        _access_log_logger.warning(
            "access_log satırı yazılamadı (istek DÜŞÜRÜLMEDİ, yalnız log "
            "kaybı): %s", exc,
        )


@app.middleware("http")
async def access_control_and_logging_middleware(request: Request, call_next):
    """G2 (Hafta 5 kapanışı): minimal tek-kullanıcılı Auth stub + HER
    isteğin `access_log`'a yazılması.

    Auth: `GBMAID_API_KEY` tanımlıysa `X-API-Key` header'ı ZORUNLU ve TAM
    eşleşmeli (401 aksi halde) -- tanımlı DEĞİLSE auth DEVRE DIŞI
    (modül-yükleme zamanında AÇIKÇA loglanmıştır, bkz. yukarısı).

    Loglama: hem BAŞARILI hem REDDEDİLMİŞ (401) istekler `access_log`'a
    yazılır (bir güvenlik denetim izinde reddedilen denemeler de
    GÖRÜNÜR olmalı) -- İSTEK GÖVDESİ/hasta verisi ASLA yazılmaz, sadece
    `{method} {path} -> {status_code}`.

    Codex çapraz inceleme (2026-09-11) MEDIUM bulgusu: `call_next`'ten
    (bir `HTTPException` DEĞİL, örn. beklenmeyen bir `RuntimeError` gibi)
    ham bir istisna fırlarsa eskiden bu satırın altındaki loglama koduna
    HİÇ ULAŞILMIYORDU -- 500'e (`ServerErrorMiddleware`'e) giden istekler
    güvenlik denetim izinde GÖRÜNMEZ kalıyordu. Düzeltme: `call_next`
    çağrısı try/finally ile sarılır -- istisna olsun olmasın log YAZILIR,
    istisna varsa (`action`'a "-> 500(unhandled)" işareti düşüldükten
    SONRA) AYNEN yeniden fırlatılır -- davranış (istisnanın yukarı
    yayılıp `ServerErrorMiddleware`'e ulaşması) DEĞİŞMEZ, sadece log
    eklenir."""

    provided_key = request.headers.get("X-API-Key")
    if not _is_authorized(provided_key):
        response = JSONResponse(
            status_code=401,
            content={"detail": "Eksik veya geçersiz X-API-Key header'ı."},
        )
        action = f"{request.method} {request.url.path} -> {response.status_code}"
        _write_access_log(
            action=action, patient_id=_extract_patient_id_from_path(request.url.path)
        )
        return response

    try:
        response = await call_next(request)
    except Exception:
        # HTTPException'lar FastAPI'nin kendi exception handler'ları
        # tarafından zaten bir Response'a çevrilip call_next'ten normal
        # DÖNER (buraya istisna olarak gelmez) -- bu blok yalnız
        # BEKLENMEYEN/ham istisnalar için çalışır.
        action = (
            f"{request.method} {request.url.path} -> 500(unhandled)"
        )
        _write_access_log(
            action=action, patient_id=_extract_patient_id_from_path(request.url.path)
        )
        raise

    action = f"{request.method} {request.url.path} -> {response.status_code}"
    _write_access_log(
        action=action, patient_id=_extract_patient_id_from_path(request.url.path)
    )
    return response


# NOT: eskiden burada `REDIS_URL = os.getenv("REDIS_URL")` vardı --
# `/health`'in artık kullanmadığı, yanıltıcı ölü bir değişkendi (bkz. G3
# düzeltmesi, `health_check()` içindeki not). `.env`'de bu isim HİÇ
# tanımlı değil, gerçek Redis (Upstash REST) yapılandırması
# `pipeline.rag_cache.load_upstash_config()` üzerinden okunuyor.

REQUIRED_MODALITIES = {"T1", "T1ce", "T2", "FLAIR"}

# `pipeline.segmentation.nas_root()`'un NAS kökü tamamen erişilemez
# olduğunda fırlattığı `FileNotFoundError` mesajının ayırt edici alt
# dizesi. `resolve_nas_path()` tek bir dosyanın bulunamadığı (bozuk/
# yanlış yol) durumda AYNI istisna sınıfını ama FARKLI bir mesajla
# ("... bulunamadı") fırlatır -- bu iki senaryoyu HTTP durum kodu
# seviyesinde ayırt etmek için mesaj eşleştirmesi kullanılır (NAS
# tamamen erişilemezse 503, tek dosya bozuk/eksikse 422).
_NAS_UNAVAILABLE_MARKER = "erişilebilir değil"


class HangingThreadCapacityExceededError(RuntimeError):
    """Süreç genelinde izin verilen 'asılı kalabilecek' thread sayısı aşıldı.

    Bu, thread-leak sorununu ÇÖZMEZ (Python thread'leri zorla
    öldürülemiyor, bkz. `_run_with_timeout` docstring'i) -- sadece en
    kötü senaryoyu (sınırsız zombi thread birikimi, dolaylı kaynak
    tükenmesi/kendi kendine DoS) sınırlı tutar. Bu istisna fırlatıldığında
    YENİ bir thread hiç YARATILMAZ.
    """


# Modül seviyesinde, SÜREÇ GENELİNDE (tek bir isteğin kendi taramaları
# değil, aynı anda gelen TÜM `/patient/{id}/harmonize` istekleri boyunca)
# "asılı" (timeout'a uğramış ama arka planda hâlâ çalışan) thread sayısını
# izler. `threading.Lock` ile korunur -- birden fazla istek eş zamanlı
# gelebilir (FastAPI/uvicorn thread havuzunda senkron endpoint'ler ayrı
# thread'lerde çalışır).
_hanging_thread_lock = threading.Lock()
_hanging_thread_count = 0


def _hanging_thread_limit() -> int:
    """Süreç genelinde izin verilen eş zamanlı 'asılı' thread üst sınırı."""

    return _positive_int_env("GBMAID_HARMONIZE_MAX_HANGING_THREADS", 5)


def _current_hanging_thread_count() -> int:
    with _hanging_thread_lock:
        return _hanging_thread_count


def _reserve_hanging_thread_slot() -> None:
    """Kapasite kontrolü ile rezervasyonu TEK atomik işlem olarak yapar.

    ÖNCEKİ HATA (2026-08-12 Codex review-gate CRITICAL bulgusu, Barış kod
    okuyarak doğruladı): kontrol (`current >= limit`) `_current_hanging_
    thread_count()` ile kilidi alıp okuyup HEMEN bırakıyordu, artırma ise
    SADECE timeout dalında, AYRI bir kilit bloğunda (`_increment_hanging_
    thread_count()`) yapılıyordu. İkisi arasında (thread submit + `future.
    result(timeout=...)` bekleme gibi) uzun bir pencere vardı -- iki
    eşzamanlı çağrı aynı anda `current=4/limit=5` görüp ikisi de `4 < 5`
    ile geçebiliyordu (klasik TOCTOU), sayaç limiti hiç görmeden 6'ya
    çıkabiliyordu. Bu yüzden decision dosyasındaki "üst sınırı sabitler"
    ifadesi YANLIŞTI -- gerçekte üst sınır DEĞİL, sadece "genelde düşük
    tutar" gibi bir şeydi.

    DÜZELTME: kontrol VE artırma artık AYNI `with _hanging_thread_lock`
    bloğu içinde, TEK atomik işlem -- ikisi arasına başka hiçbir thread
    giremez. Kritik bölüm bilinçli olarak MİNİMAL tutuluyor (sadece bir
    karşılaştırma + bir tamsayı artırma); thread submit etme/`future.
    result()` bekleme gibi uzun işlemler kilidin DIŞINDA kalıyor (deadlock/
    performans riski yok).

    SEMANTİK DEĞİŞİKLİĞİ (bilinçli, decision dosyasında belgelendi): sayaç
    artık "onaylanmış zombi thread sayısı" DEĞİL, "şu an rezerve edilmiş
    potansiyel kapasite" ifade ediyor. `_run_with_timeout()` artık submit
    ettiği HER thread için (timeout olsun olmasın) bir slot rezerve eder
    ve future TAMAMLANDIĞINDA (başarı/hata/timeout sonrası fark etmeksizin,
    er ya da geç) `add_done_callback` ile bu slotu serbest bırakır. Yani
    hızlı biten normal bir tarama da submit->done arasındaki KISA pencerede
    sayaca geçici olarak katkı yapar -- ama bu, limitin GERÇEKTEN bir üst
    sınır olması için gereken maliyettir.
    """

    global _hanging_thread_count
    limit = _hanging_thread_limit()
    with _hanging_thread_lock:
        if _hanging_thread_count >= limit:
            raise HangingThreadCapacityExceededError(
                f"Sürec genelinde asılı thread sınırı asildi: "
                f"{_hanging_thread_count}/{limit} "
                f"(GBMAID_HARMONIZE_MAX_HANGING_THREADS). Yeni istek "
                f"reddedildi, yeni bir zombi thread yaratilmadi."
            )
        _hanging_thread_count += 1


def _decrement_hanging_thread_count(_future: "concurrent.futures.Future") -> None:
    """`future.add_done_callback` için: rezerve edilen slotu serbest bırak.

    `_run_with_timeout()` artık submit ettiği HER thread için (timeout
    olsun olmasın) bir slot rezerve ediyor -- bu geri çağırma da HER
    submit'ten sonra kayıtlı (sadece timeout dalında DEĞİL). Future ister
    başarıyla bitsin, ister hata fırlatsın, ister timeout'tan SONRA geç de
    olsa doğal olarak bitsin -- fark etmeksizin bu çağrılır ve slotu geri
    azaltır. Bu, thread'i öldürmez -- sadece geç de olsa bittiğinde
    muhasebeyi doğru tutar, aksi halde sayaç asla azalmaz ve limit kalıcı
    olarak tıkanır.
    """

    global _hanging_thread_count
    with _hanging_thread_lock:
        _hanging_thread_count = max(0, _hanging_thread_count - 1)


# İKİNCİ, AYRI sayaç (2026-08-12 gözlemlenebilirlik netliği turu):
# `_hanging_thread_count` artık "şu an rezerve edilmiş aktif slot sayısı"
# anlamına geliyor -- HER submit edilen tarama (timeout olsun olmasın)
# buna kısa süreliğine katkı yapar (bkz. `_reserve_hanging_thread_slot`
# docstring'i, semantik değişikliği). Bu, "kaç tane GERÇEKTEN zombi
# (timeout'a uğramış, arka planda hâlâ çalışan, öldürülemeyen) thread
# var" sorusuna CEVAP VERMEZ -- normal işlenen 5 eşzamanlı tarama da
# aynı sayaca geçici katkı yapar. `_confirmed_hanging_thread_count` bu
# boşluğu kapatır: SADECE `future.result(timeout=...)` GERÇEKTEN
# `TimeoutError`/`concurrent.futures.TimeoutError` fırlattığında artar
# (bkz. `_run_with_timeout`), normal/hızlı biten bir tarama bunu ASLA
# etkilemez.
_confirmed_hanging_thread_count = 0


def _current_confirmed_hanging_thread_count() -> int:
    with _hanging_thread_lock:
        return _confirmed_hanging_thread_count


def _increment_confirmed_hanging_thread_count() -> None:
    global _confirmed_hanging_thread_count
    with _hanging_thread_lock:
        _confirmed_hanging_thread_count += 1


def _decrement_confirmed_hanging_thread_count(
    _future: "concurrent.futures.Future",
) -> None:
    """Gerçekten timeout'a uğramış bir future NİHAYET (doğal olarak,
    öldürülmeden) bittiğinde onaylanmış-zombi sayacını geri azalt."""

    global _confirmed_hanging_thread_count
    with _hanging_thread_lock:
        _confirmed_hanging_thread_count = max(0, _confirmed_hanging_thread_count - 1)


def _positive_int_env(name: str, default: int) -> int:
    """Bir ortam değişkenini pozitif tamsayı olarak oku; sessiz varsayılan YOK.

    Değişken tanımlı değilse `default` döner (bu açık bir varsayılandır,
    sessiz fallback değildir). Tanımlı ama geçersizse (negatif/sayı değil)
    açıkça `ValueError` fırlatılır.
    """

    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} pozitif bir tamsayı (saniye) olmalı, alınan: {raw_value!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"{name} pozitif olmalı, alınan: {value}")
    return value


def _harmonize_scan_timeout_seconds() -> int:
    """Tek bir taramanın NAS-çözümleme + N4 + Z-score zinciri için üst sınır."""

    return _positive_int_env("GBMAID_HARMONIZE_SCAN_TIMEOUT_SECONDS", 120)


def _harmonize_total_timeout_seconds() -> int:
    """Bir `/patient/{id}/harmonize` isteğinin TÜM taramalar için toplam üst sınırı."""

    return _positive_int_env("GBMAID_HARMONIZE_TOTAL_TIMEOUT_SECONDS", 600)


def _process_single_scan(scan: dict) -> str:
    """Bir taramanın NAS çözümleme + N4 + Z-score zincirini çalıştırır.

    Modül seviyesindeki `resolve_nas_path`/`apply_n4_bias_correction`/
    `apply_zscore_normalization` adlarını isim ile çağırır (import edilmiş
    referansı önceden yakalamaz) -- bu sayede testlerdeki
    `monkeypatch.setattr(api_main, "...", ...)` deseni, fonksiyon ayrı bir
    thread'de çalıştırılsa bile hâlâ çalışır.
    """

    image_path = resolve_nas_path(scan["file_path"])
    n4_path = apply_n4_bias_correction(str(image_path))
    return apply_zscore_normalization(n4_path, source=scan["source"])


def _run_with_timeout(func, *args, timeout: float, **kwargs):
    """`func`'ı ayrı, TEK KULLANIMLIK bir thread'de çalıştırıp süresini sınırla.

    Paylaşımlı/tekrar-kullanılan bir `ThreadPoolExecutor` KASITLI OLARAK
    kullanılmaz: NAS askıda kalırsa (yanıt vermeden bloke olursa) altta
    yatan Python thread'i zorla sonlandırılamaz (dilin bir kısıtı). Eğer
    tüm istekler tek bir paylaşımlı havuzu paylaşsaydı, bir askıda kalan
    tarama o havuzun TEK worker'ını sonsuza dek işgal eder ve sonraki HER
    isteği (farklı hastalar dahil) tıkardı. Bunun yerine her çağrı kendi
    tek-worker'lı, izole executor'ını alır; zaman aşımında executor
    `wait=False` ile hemen serbest bırakılır -- askıda kalan thread arka
    planda kendi başına çalışmaya devam edebilir ama BAŞKA hiçbir isteği
    bloke ETMEZ. Bilinen sınır: o tek taramaya ait iş parçacığı süreç
    ömrü boyunca arka planda yaşamaya devam edebilir (Python thread'leri
    kill edilemez); bu, timeout'un "isteği bekletmeme" garantisi için
    kabul edilen bir maliyettir.

    ÇÖZÜM DEĞİL, SADECE SINIRLAMA: yukarıdaki sınır kalıcıdır ve bu
    fonksiyon onu ORTADAN KALDIRMAZ. Ek olarak, YENİ bir thread
    yaratılmadan ÖNCE süreç genelindeki "asılı olabilecek" thread sayısı
    `GBMAID_HARMONIZE_MAX_HANGING_THREADS` sınırına karşı ATOMİK olarak
    kontrol edilir VE rezerve edilir (bkz. `_reserve_hanging_thread_slot`
    docstring'i, 2026-08-12 TOCTOU düzeltmesi) -- sınır aşılmışsa
    `HangingThreadCapacityExceededError` fırlatılır ve YENİ bir thread hiç
    açılmaz. Bu, tekrarlayan NAS kararsızlığında zombi thread sayısının
    sınırsız büyümesini (dolaylı kaynak tükenmesi/kendi kendine DoS)
    GERÇEKTEN önler (artık atomik); ama zaten var olan asılı thread'leri
    geriye dönük olarak durdurmaz.

    DAVRANIŞ DEĞİŞİKLİĞİ: rezervasyon submit'ten ÖNCE yapıldığı ve
    serbest bırakma HER submit için (timeout olsun olmasın)
    `add_done_callback` ile kayıtlı olduğu için, normal/hızlı biten bir
    tarama da submit->done arasındaki KISA pencerede sayaca geçici olarak
    katkı yapar. Bu kabul edilebilir bir maliyet -- limitin GERÇEKTEN
    atomik bir üst sınır olması için gerekli.

    GÖZLEMLENEBİLİRLİK NETLİĞİ (2026-08-12, reviewer HIGH bulgusu):
    yukarıdaki sayaç (`_hanging_thread_count`) artık meşru eşzamanlı
    işleri de saydığı için tek başına "kaç zombi var" sorusuna cevap
    vermiyor. Bu fonksiyon `future.result(timeout=...)` GERÇEKTEN
    `TimeoutError` fırlattığında AYRICA `_confirmed_hanging_thread_count`
    sayacını artırır (ve future nihayet doğal olarak bitince geri
    azaltır) -- bu, sadece GERÇEKTEN timeout'a uğramış, arka planda hâlâ
    çalışan thread'leri sayar, normal biten hiçbir çağrı bunu etkilemez.
    """

    _reserve_hanging_thread_slot()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(func, *args, **kwargs)
    except Exception:
        # Submit'in kendisi (beklenmedik biçimde) başarısız olursa rezerve
        # edilen slot sessizce sızmamalı -- future hiç oluşmadığı için
        # `add_done_callback` de kaydedilemez, bu yüzden burada elle
        # serbest bırakılır.
        _decrement_hanging_thread_count(None)
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    # Slot, future NE ŞEKİLDE biterse bitsin (başarı, hata, ya da
    # timeout'tan SONRA geç de olsa doğal biterse) serbest bırakılır --
    # SADECE timeout dalında DEĞİL, HER submit için kayıtlı.
    future.add_done_callback(_decrement_hanging_thread_count)
    try:
        return future.result(timeout=timeout)
    except (TimeoutError, concurrent.futures.TimeoutError):
        # GERÇEK zaman aşımı -- bu future'ın arka planda hâlâ çalıştığı
        # (öldürülemediği) ONAYLANMIŞ. Ayrı sayaç burada artırılır ve
        # future nihayet doğal olarak bittiğinde (kill edilmeden) geri
        # azaltılır -- `_hanging_thread_count`'un aksine, bu sayaca
        # normal/hızlı biten çağrılar HİÇBİR ZAMAN katkı yapmaz.
        _increment_confirmed_hanging_thread_count()
        future.add_done_callback(_decrement_confirmed_hanging_thread_count)
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def get_db_connection():
    """API sorguları için merkezi, salt-okunur DB bağlantısı."""

    return get_connection(readonly=True)


@app.get("/")
def read_root():
    return {"message": "GBM-AID API calisiyor"}


@app.get("/health")
def health_check():
    status = {"supabase": "error", "redis": "error"}

    try:
        conn = get_db_connection()
        conn.close()
        status["supabase"] = "ok"
    except Exception as e:
        status["supabase"] = f"error: {str(e)}"

    # G3 düzeltmesi (2026-09-11, 10+ gündür açık bulgu): eskiden burada
    # `redis.from_url(REDIS_URL)` çağrılıyordu -- `.env`'de `REDIS_URL`
    # (protokol-tabanlı `rediss://`) HİÇ YOK, yalnız Upstash'in REST
    # kimlik bilgileri (`UPSTASH_REDIS_REST_URL`/`_TOKEN`) var (bkz.
    # `pipeline/rag_cache.py` modül dokstring'i "ÖNEMLİ BULGU,
    # 2026-08-18"). `REDIS_URL=None` iken `redis.from_url(None)` HER
    # ZAMAN kırık bir "error" döndürüyordu -- bu üç durumu (yanlış
    # yapılandırılmış/gerçekten erişilemez/hiç yapılandırılmamış)
    # AYIRT ETMİYORDU. Artık `pipeline.rag_cache`'in ÇALIŞAN Upstash REST
    # deseni (aynı PING mantığı) kullanılıyor -- kod TEKRARLANMADI,
    # `check_upstash_connectivity()`/`load_upstash_config()` DOĞRUDAN
    # import edilip çağrıldı.
    if load_upstash_config() is None:
        status["redis"] = (
            "not_configured -- UPSTASH_REDIS_REST_URL/UPSTASH_REDIS_REST_TOKEN "
            ".env'de tanımlı değil (eski REDIS_URL/rediss:// deseni bu "
            "projede KULLANILMIYOR, bkz. pipeline/rag_cache.py)."
        )
    else:
        try:
            status["redis"] = "ok" if check_upstash_connectivity() else "error: PING PONG dönmedi"
        except Exception as e:  # noqa: BLE001 -- health-check hiçbir zaman çökmemeli
            status["redis"] = f"error: {str(e)}"

    # Thread-leak izlenebilirliği (bkz. decisions/2026-08-12-backend-
    # harmonize-endpoint-timeout-hardening.md "TAKİP"/"[2026-08-12]
    # NETLEŞTİRME" bölümleri). Bu bir ÇÖZÜM değil -- sadece görünürlük,
    # AMA İKİ AYRI SORUYA CEVAP VEREN İKİ AYRI ALAN OLARAK:
    #
    # - `active_harmonize_slot_count` / `active_harmonize_slot_limit`:
    #   "şu an rezerve edilmiş kapasite ne kadar dolu" (eski adı
    #   `hanging_thread_count`/`hanging_thread_limit` -- 2026-08-12'de
    #   yeniden adlandırıldı çünkü YANILTICIYDI). Bu sayaç HER işlenen
    #   tarama için (timeout olsun olmasın) submit->done arasında KISA
    #   süreliğine artar -- örn. varsayılan
    #   `GBMAID_HARMONIZE_MAX_HANGING_THREADS=5` iken, 5 hastanın
    #   harmonize isteği TAMAMEN NORMAL şekilde (hiç timeout olmadan)
    #   eşzamanlı işleniyorsa BİLE bu alan 5/5 gösterebilir ve 6. meşru
    #   eşzamanlı istek 503 alır -- bu "zombi thread" biriktiği anlamına
    #   GELMEZ, sadece kapasitenin o an dolu olduğu anlamına gelir.
    # - `confirmed_hanging_thread_count`: "kaç thread GERÇEKTEN
    #   timeout'a uğradı ve hâlâ arka planda (öldürülemeden) çalışıyor" --
    #   SADECE `future.result(timeout=...)` fiilen `TimeoutError`
    #   fırlattığında artar, normal biten hiçbir tarama bunu ETKİLEMEZ.
    #   Bir operatör "kaç zombi birikti" sorusunu buradan cevaplamalı,
    #   `active_harmonize_slot_count`'tan DEĞİL.
    #
    # `process_active_thread_count`, `threading.active_count()` --
    # SÜRECİN TÜMÜNDEKİ (yukarıdaki iki sayaçla izlenmeyenler dahil,
    # örn. uvicorn worker thread'leri) canlı thread sayısı, genel bir
    # sızıntı sinyali.
    status["active_harmonize_slot_count"] = _current_hanging_thread_count()
    status["active_harmonize_slot_limit"] = _hanging_thread_limit()
    status["confirmed_hanging_thread_count"] = (
        _current_confirmed_hanging_thread_count()
    )
    status["process_active_thread_count"] = threading.active_count()

    return status


@app.get("/patient/{patient_id}/status")
def get_patient_status(patient_id: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute(
            "SELECT patient_id, scan_date, modality, harmonization_status "
            "FROM mr_scans WHERE patient_id = %s ORDER BY scan_date",
            (patient_id,),
        )
        scans = cur.fetchall()
        cur.close()
        conn.close()

        if not scans:
            raise HTTPException(
                status_code=404,
                detail=f"Hasta bulunamadi veya tarama kaydi yok: {patient_id}",
            )

        return {"patient_id": patient_id, "scans": scans}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Veritabani hatasi: {str(e)}"
        )


@app.post("/patient/{patient_id}/harmonize")
def harmonize_patient(patient_id: str):
    """
    Goruntu-uzayi on islemesini bir hasta icin calistirir:
    N4 -> kaynak-bazli Z-score.

    ComBat bu endpoint'te goruntuye uygulanmaz. Hazir maske ile PyRadiomics
    cikarimindan sonra olusan ozellik vektorune Nisa'nin ayri adiminda
    uygulanir. Segmentasyon da ComBat sonrasi bir islem degildir.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT ms.scan_id, ms.scan_date, ms.timepoint_label, "
            "ms.modality, ms.file_path, ds.source_name AS source "
            "FROM mr_scans AS ms "
            "JOIN patients AS p ON p.patient_id = ms.patient_id "
            "JOIN dataset_sources AS ds ON ds.source_id = p.source_id "
            "WHERE ms.patient_id = %s",
            (patient_id,),
        )
        scans = cur.fetchall()
        cur.close()
        conn.close()

        if not scans:
            raise HTTPException(
                status_code=404,
                detail=f"Hasta bulunamadi: {patient_id}",
            )

        scans_by_timepoint = defaultdict(list)
        for scan in scans:
            timepoint = (
                scan.get("timepoint_label")
                or (
                    scan["scan_date"].isoformat()
                    if scan.get("scan_date") is not None
                    else "baseline"
                )
            )
            scans_by_timepoint[timepoint].append(scan)

        missing_by_timepoint = {}
        for timepoint, timepoint_scans in scans_by_timepoint.items():
            available = {scan["modality"] for scan in timepoint_scans}
            missing = REQUIRED_MODALITIES - available
            if missing:
                missing_by_timepoint[timepoint] = sorted(missing)
        if missing_by_timepoint:
            raise HTTPException(
                status_code=422,
                detail=f"Zaman noktası bazında eksik modalite: "
                f"{missing_by_timepoint}. "
                f"Gerekli: {sorted(REQUIRED_MODALITIES)}",
            )

        results = {}
        scan_timeout = _harmonize_scan_timeout_seconds()
        total_timeout = _harmonize_total_timeout_seconds()
        overall_deadline = time.monotonic() + total_timeout

        for scan in scans:
            result_key = str(scan["scan_id"])

            remaining_budget = overall_deadline - time.monotonic()
            if remaining_budget <= 0:
                raise HTTPException(
                    status_code=504,
                    detail=(
                        "Harmonizasyon toplam zaman asimi asildi "
                        f"(sinir: {total_timeout}s). scan_id={result_key} ve "
                        "sonrasi hic islenmedi -- kismi/sessiz sonuc DONULMEDI."
                    ),
                )
            effective_timeout = min(scan_timeout, remaining_budget)

            try:
                zscore_path = _run_with_timeout(
                    _process_single_scan,
                    scan,
                    timeout=effective_timeout,
                )
                results[result_key] = {
                    "status": "ok",
                    "modality": scan["modality"],
                    "timepoint": scan.get("timepoint_label"),
                    "path": zscore_path,
                }
            except HangingThreadCapacityExceededError as exc:
                # Sürec genelindeki asılı thread sınırı asıldı -- YENİ bir
                # thread hiç yaratılmadı (bkz. `_run_with_timeout`
                # docstring'i). Bu bir "çözüm" DEĞİL, sadece en kötü
                # senaryonun (sınırsız zombi thread birikimi) üst sınırını
                # koruma amaçlı bir tedbir -- zaten var olan asılı
                # thread'ler bundan etkilenmez/durmaz.
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Sunucu asili is yuku kapasitesi dolu, istek "
                        f"reddedildi (yeni thread yaratilmadi): {exc}"
                    ),
                )
            except NotImplementedError:
                results[result_key] = {
                    "status": "pending",
                    "detail": "Harmonizasyon fonksiyonu henuz implemente edilmedi",
                }
            except ZScoreStatisticsNotFoundError as exc:
                results[result_key] = {
                    "status": "blocked",
                    "stage": "zscore",
                    "detail": str(exc),
                }
            except KeyError as exc:
                # `pipeline.segmentation.nas_root()` GBMAID_NAS_ROOT tanimli
                # degilse KeyError firlatir -- bu bir konfigurasyon hatasidir,
                # NAS'in kendisiyle ilgili degil, 500 olarak siniflandirilir.
                raise HTTPException(
                    status_code=500,
                    detail=f"Sunucu konfigurasyon hatasi: {exc}",
                )
            except FileNotFoundError as exc:
                if _NAS_UNAVAILABLE_MARKER in str(exc):
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"NAS erisilemez, harmonizasyon calistirilamadi: {exc}"
                        ),
                    )
                raise HTTPException(
                    status_code=422,
                    detail=f"Bozuk veya bulunamayan dosya: {scan['file_path']} ({exc})",
                )
            except (TimeoutError, concurrent.futures.TimeoutError):
                raise HTTPException(
                    status_code=504,
                    detail=(
                        "Harmonizasyon zaman asimina ugradi: "
                        f"scan_id={result_key}, sinir={effective_timeout:.0f}s "
                        "(NAS yavas/erisilemez olabilir)."
                    ),
                )

        return {"patient_id": patient_id, "harmonization_results": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Beklenmeyen hata: {str(e)}"
        )
