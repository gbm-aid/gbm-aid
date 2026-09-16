from __future__ import annotations

import threading
import time
from typing import Any

import api.main as api_main
from pipeline.harmonization import ZScoreStatisticsNotFoundError


class FakeCursor:
    def __init__(self, scans: list[dict[str, Any]]) -> None:
        self.scans = scans
        self.query = ""

    def execute(self, query: str, _params: tuple[str]) -> None:
        self.query = query

    def fetchall(self) -> list[dict[str, Any]]:
        return self.scans

    def close(self) -> None:
        return None


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.fake_cursor = cursor

    def cursor(self, **_kwargs: Any) -> FakeCursor:
        return self.fake_cursor

    def close(self) -> None:
        return None


def _complete_tcga_scans() -> list[dict[str, str]]:
    return [
        {
            "scan_id": str(index),
            "scan_date": None,
            "timepoint_label": None,
            "modality": modality,
            "file_path": f"{modality}.nii.gz",
            "source": "TCGA-GBM",
        }
        for index, modality in enumerate(("T1", "T1ce", "T2", "FLAIR"), start=1)
    ]


def test_harmonize_endpoint_passes_database_source_to_zscore(monkeypatch) -> None:
    cursor = FakeCursor(_complete_tcga_scans())
    monkeypatch.setattr(
        api_main,
        "get_db_connection",
        lambda: FakeConnection(cursor),
    )
    monkeypatch.setattr(
        api_main,
        "apply_n4_bias_correction",
        lambda path: f"{path}.n4",
    )
    monkeypatch.setattr(api_main, "resolve_nas_path", lambda path: path)
    seen_sources: list[str] = []

    def fake_zscore(path: str, source: str) -> str:
        seen_sources.append(source)
        return f"{path}.zscore"

    monkeypatch.setattr(api_main, "apply_zscore_normalization", fake_zscore)
    result = api_main.harmonize_patient("TCGA-02-0003")

    assert "JOIN dataset_sources" in cursor.query
    assert seen_sources == ["TCGA-GBM"] * 4
    assert {
        item["status"]
        for item in result["harmonization_results"].values()
    } == {"ok"}


def test_harmonize_endpoint_reports_missing_zscore_stats_as_blocked(
    monkeypatch,
) -> None:
    cursor = FakeCursor(_complete_tcga_scans())
    monkeypatch.setattr(
        api_main,
        "get_db_connection",
        lambda: FakeConnection(cursor),
    )
    monkeypatch.setattr(
        api_main,
        "apply_n4_bias_correction",
        lambda path: f"{path}.n4",
    )
    monkeypatch.setattr(api_main, "resolve_nas_path", lambda path: path)

    def missing_stats(_path: str, source: str) -> str:
        assert source == "TCGA-GBM"
        raise ZScoreStatisticsNotFoundError("source stats missing")

    monkeypatch.setattr(api_main, "apply_zscore_normalization", missing_stats)
    result = api_main.harmonize_patient("TCGA-02-0003")

    assert {
        item["status"]
        for item in result["harmonization_results"].values()
    } == {"blocked"}
    assert {
        item["stage"]
        for item in result["harmonization_results"].values()
    } == {"zscore"}


def test_harmonize_endpoint_rejects_missing_modality_per_timepoint(
    monkeypatch,
) -> None:
    scans = _complete_tcga_scans()
    for scan in scans:
        scan["timepoint_label"] = "week-000"
    later_visit = [
        {
            **scan,
            "scan_id": str(int(scan["scan_id"]) + 10),
            "timepoint_label": "week-056",
        }
        for scan in scans
        if scan["modality"] != "T2"
    ]
    cursor = FakeCursor(scans + later_visit)
    monkeypatch.setattr(
        api_main,
        "get_db_connection",
        lambda: FakeConnection(cursor),
    )

    try:
        api_main.harmonize_patient("Patient-001")
    except api_main.HTTPException as exc:
        assert exc.status_code == 422
        assert "week-056" in str(exc.detail)
        assert "T2" in str(exc.detail)
    else:
        raise AssertionError("Eksik zaman-noktası modalitesi kabul edilmemeliydi.")


def test_harmonize_endpoint_returns_503_when_nas_root_unavailable(monkeypatch) -> None:
    """NAS'ın TAMAMI erişilemezse (nas_root() hatası) 503 dönmeli, 422 DEĞİL.

    Bugünkü NAS erişilemezlik olayının (bkz. AKTIF-GOREVLER.md,
    run_harmonization_cohort.py'nin "GORUNTU_YOK:NAS kökü erişilebilir
    değil" bulgusu) doğrudan gerekçelendirdiği regresyon testi.
    """

    cursor = FakeCursor(_complete_tcga_scans())
    monkeypatch.setattr(
        api_main,
        "get_db_connection",
        lambda: FakeConnection(cursor),
    )

    def nas_down(_path: str) -> str:
        raise FileNotFoundError(
            "NAS kökü erişilebilir değil: \\\\100.101.131.47\\gbmaid"
        )

    monkeypatch.setattr(api_main, "resolve_nas_path", nas_down)

    try:
        api_main.harmonize_patient("TCGA-02-0003")
    except api_main.HTTPException as exc:
        assert exc.status_code == 503
        assert "NAS erisilemez" in str(exc.detail) or "NAS" in str(exc.detail)
    else:
        raise AssertionError("NAS erişilemezliği sessizce/200 dönmemeliydi.")


def test_harmonize_endpoint_returns_422_for_single_missing_file(monkeypatch) -> None:
    """NAS erişilebilir ama TEK bir dosya bozuk/eksikse 422 dönmeli, 503 DEĞİL.

    503 (NAS erişilemez) ile 422 (bozuk/eksik tekil dosya) birbirine
    karıştırılmamalı -- ilki altyapı, ikincisi veri kalitesi sorunu.
    """

    cursor = FakeCursor(_complete_tcga_scans())
    monkeypatch.setattr(
        api_main,
        "get_db_connection",
        lambda: FakeConnection(cursor),
    )

    def file_missing(path: str) -> str:
        raise FileNotFoundError(f"NAS dosyası bulunamadı: {path}")

    monkeypatch.setattr(api_main, "resolve_nas_path", file_missing)

    try:
        api_main.harmonize_patient("TCGA-02-0003")
    except api_main.HTTPException as exc:
        assert exc.status_code == 422
    else:
        raise AssertionError("Bozuk/eksik tekil dosya sessizce kabul edilmemeliydi.")


def test_harmonize_endpoint_times_out_a_hanging_scan(monkeypatch) -> None:
    """Bir tarama işlemi asılı kalırsa (NAS yavaş/yanıtsız) 504 dönmeli.

    Bugün keşfedilen arka plan süreç ömrü sınırıyla aynı hata sınıfı --
    sessizce sonsuza dek beklemek YASAK, çağırana NET bir zaman aşımı
    hatası dönmeli.
    """

    cursor = FakeCursor(_complete_tcga_scans())
    monkeypatch.setattr(
        api_main,
        "get_db_connection",
        lambda: FakeConnection(cursor),
    )
    monkeypatch.setattr(api_main, "resolve_nas_path", lambda path: path)
    monkeypatch.setenv("GBMAID_HARMONIZE_SCAN_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("GBMAID_HARMONIZE_TOTAL_TIMEOUT_SECONDS", "600")

    # 5s DEĞİL 2s: gerçek zamanlanmış timeout'u (1s) tetiklemeye yeter,
    # ama arka planda gereksiz yere uzun süre "asılı" thread bırakıp
    # sonraki testleri (özellikle race-condition testinin baseline=0
    # varsayımını) kirletmez -- aşağıdaki cleanup bekleyişiyle birlikte.
    def hanging_n4(path: str) -> str:
        time.sleep(2)
        return f"{path}.n4"

    monkeypatch.setattr(api_main, "apply_n4_bias_correction", hanging_n4)

    started = time.monotonic()
    try:
        api_main.harmonize_patient("TCGA-02-0003")
    except api_main.HTTPException as exc:
        elapsed = time.monotonic() - started
        assert exc.status_code == 504
        # Gerçek 2s'lik uykuyu değil, ~1s'lik yapılandırılmış sınırı bekledi.
        assert elapsed < 4
    else:
        raise AssertionError("Asılı kalan tarama sessizce/200 dönmemeliydi.")

    # Test yalıtımı: arka plandaki asılı thread'in (2s uyku) doğal olarak
    # bitip rezerve ettiği slotu geri bırakmasını bekle -- aksi halde
    # sonraki testler (özellikle gerçek eşzamanlılık testi) sayacı sıfır
    # olmayan bir durumdan başlatıp yanlış sonuçlanabilir.
    cleanup_deadline = time.monotonic() + 4.0
    while (
        api_main._current_hanging_thread_count() > 0
        and time.monotonic() < cleanup_deadline
    ):
        time.sleep(0.05)
    assert api_main._current_hanging_thread_count() == 0


def test_harmonize_endpoint_rejects_when_hanging_thread_capacity_full(
    monkeypatch,
) -> None:
    """Sürec genelindeki asılı thread sınırı dolduğunda YENİ istek 503 ile
    reddedilmeli VE yeni bir thread hiç yaratılmamalı.

    Bugünkü review-gate'in bulduğu HIGH bulgunun (thread-leak, bkz.
    decisions/2026-08-12-backend-harmonize-endpoint-timeout-hardening.md
    "TAKİP" bölümü) somut takip görevi. Gerçek bir thread'i askıda
    bırakıp saniyelerce beklemek yerine, modül-seviyesindeki
    `_hanging_thread_count` sayacı DOĞRUDAN sınıra eşitlenerek
    deterministik/hızlı test edilir -- gerçek zamanlamaya dayanan bir
    test yarış koşulu riski taşırdı.

    NOT (2026-08-12 race-condition düzeltmesi sonrası): kontrol+artırma
    artık `_reserve_hanging_thread_slot()` içinde TEK atomik kritik
    bölümde, `_current_hanging_thread_count()` accessor'ı ÜZERİNDEN DEĞİL
    modül global'i (`_hanging_thread_count`) doğrudan okuyor -- bu yüzden
    bu test artık `_current_hanging_thread_count`'u monkeypatch ETMİYOR
    (bunu yapmak artık gerçek kontrol yolunu etkilemez), doğrudan
    `api_main._hanging_thread_count`'u ayarlayıp test sonunda eski haline
    geri yüklüyor.
    """

    cursor = FakeCursor(_complete_tcga_scans())
    monkeypatch.setattr(
        api_main,
        "get_db_connection",
        lambda: FakeConnection(cursor),
    )
    monkeypatch.setenv("GBMAID_HARMONIZE_MAX_HANGING_THREADS", "2")
    monkeypatch.setattr(api_main, "_hanging_thread_count", 2)

    submitted = {"count": 0}

    def fail_if_called(path: str) -> str:
        submitted["count"] += 1
        return path

    monkeypatch.setattr(api_main, "resolve_nas_path", fail_if_called)

    try:
        api_main.harmonize_patient("TCGA-02-0003")
    except api_main.HTTPException as exc:
        assert exc.status_code == 503
        assert "kapasite" in str(exc.detail) or "asili" in str(exc.detail)
    else:
        raise AssertionError(
            "Asılı thread kapasitesi dolduğunda istek sessizce/200 dönmemeliydi."
        )

    # En önemli iddia: kapasite dolu olduğunda YENİ bir thread/iş hiç
    # BAŞLATILMADI (zombi thread sayısı büyümedi).
    assert submitted["count"] == 0


def test_harmonize_endpoint_allows_request_when_under_hanging_thread_capacity(
    monkeypatch,
) -> None:
    """Asılı thread sayısı sınırın ALTINDAYSA istek normal şekilde işlenmeli
    (yeni eklenen kapasite kontrolü, mevcut başarı yolunu KIRMAMALI)."""

    cursor = FakeCursor(_complete_tcga_scans())
    monkeypatch.setattr(
        api_main,
        "get_db_connection",
        lambda: FakeConnection(cursor),
    )
    monkeypatch.setenv("GBMAID_HARMONIZE_MAX_HANGING_THREADS", "5")
    monkeypatch.setattr(api_main, "_hanging_thread_count", 1)
    monkeypatch.setattr(api_main, "resolve_nas_path", lambda path: path)
    monkeypatch.setattr(
        api_main, "apply_n4_bias_correction", lambda path: f"{path}.n4"
    )
    monkeypatch.setattr(
        api_main, "apply_zscore_normalization", lambda path, source: f"{path}.zscore"
    )

    result = api_main.harmonize_patient("TCGA-02-0003")

    assert {
        item["status"] for item in result["harmonization_results"].values()
    } == {"ok"}


def test_run_with_timeout_tracks_hanging_thread_count_across_timeout_lifecycle(
) -> None:
    """`_run_with_timeout` gerçek bir zaman aşımında sayacı artırmalı, arka
    plandaki thread NİHAYET bitince (done-callback ile) geri azaltmalı.

    Gerçek `time.sleep` kullanır (kısa, ~0.3s) çünkü burada test edilen
    şey tam olarak arka plan thread'inin GERÇEKTEN hayatta kaldığı ve
    bittiğinde muhasebenin doğru güncellendiği -- mock'lanamaz.
    """

    baseline = api_main._current_hanging_thread_count()

    def slow() -> str:
        time.sleep(0.3)
        return "done"

    try:
        api_main._run_with_timeout(slow, timeout=0.05)
    except api_main.concurrent.futures.TimeoutError:
        pass
    else:
        raise AssertionError("Bu çağrı gerçek bir timeout ile sonuçlanmalıydı.")

    # Timeout anında sayaç hemen artmış olmalı (thread hâlâ arka planda
    # çalışıyor, öldürülemiyor).
    assert api_main._current_hanging_thread_count() == baseline + 1

    # Arka plandaki thread'in gerçekten bitmesini bekle (0.3s uyku +
    # done-callback'in çalışması için makul bir pay), sonra sayacın geri
    # azaldığını doğrula.
    deadline = time.monotonic() + 2.0
    while (
        api_main._current_hanging_thread_count() > baseline
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)

    assert api_main._current_hanging_thread_count() == baseline


def test_health_check_reports_hanging_thread_metrics(monkeypatch) -> None:
    """`/health` yanıtı, thread-leak izlenebilirliği için DÖRT alanı
    içermeli: active_harmonize_slot_count, active_harmonize_slot_limit,
    confirmed_hanging_thread_count, process_active_thread_count.

    2026-08-12 netleştirme turu: eski `hanging_thread_count`/
    `hanging_thread_limit` alanları yeniden adlandırıldı (YANILTICIYDI --
    meşru eşzamanlı işleri de sayıyordu) ve GERÇEKTEN onaylanmış zombi
    thread sayısı için ayrı bir alan (`confirmed_hanging_thread_count`)
    eklendi."""

    monkeypatch.setattr(
        api_main,
        "get_db_connection",
        lambda: (_ for _ in ()).throw(RuntimeError("db kapali - test")),
    )
    # G3 düzeltmesi (2026-09-11): `/health` artık `redis.from_url()`
    # DEĞİL, `pipeline.rag_cache`'in Upstash REST desenini kullanıyor
    # (bkz. api/main.py::health_check() docstring notu). Bu test yalnız
    # thread-leak alanlarını doğruluyor, redis/upstash durumu KONU DIŞI --
    # "yapılandırılmamış" dalını tetiklemek yeterli.
    monkeypatch.setattr(api_main, "load_upstash_config", lambda: None)
    monkeypatch.setattr(api_main, "_current_hanging_thread_count", lambda: 3)
    monkeypatch.setattr(api_main, "_current_confirmed_hanging_thread_count", lambda: 1)
    monkeypatch.setenv("GBMAID_HARMONIZE_MAX_HANGING_THREADS", "5")

    result = api_main.health_check()

    assert result["active_harmonize_slot_count"] == 3
    assert result["active_harmonize_slot_limit"] == 5
    assert result["confirmed_hanging_thread_count"] == 1
    assert isinstance(result["process_active_thread_count"], int)
    assert result["process_active_thread_count"] >= 1
    # Eski, yanıltıcı alan adları artık YOK -- yanlışlıkla eski isme
    # bağımlı kalan bir çağıran (dashboard/script) sessizce yanlış veri
    # okumak yerine KeyError ile fark edecek.
    assert "hanging_thread_count" not in result
    assert "hanging_thread_limit" not in result


def test_confirmed_hanging_thread_count_only_increments_on_real_timeout() -> None:
    """`confirmed_hanging_thread_count` SADECE gerçek bir timeout'ta
    artmalı; normal/hızlı biten bir tarama bunu HİÇ etkilememeli -- sadece
    `active_harmonize_slot_count`'u geçici artırıp düşürmeli.

    Reviewer'ın HIGH bulgusunun (aktif=meşru işler ile onaylanmış-zombi
    ayrımının olmaması) doğrudan regresyon testi.
    """

    baseline_active = api_main._current_hanging_thread_count()
    baseline_confirmed = api_main._current_confirmed_hanging_thread_count()
    assert baseline_active == 0, (
        "Bu test önceki bir testten kalan asılı thread'le kirlenmiş "
        f"başlıyor (baseline_active={baseline_active})."
    )

    # 1) Normal/hızlı biten çağrı: confirmed sayaç HİÇ değişmemeli.
    result = api_main._run_with_timeout(lambda: "done", timeout=5)
    assert result == "done"
    assert api_main._current_hanging_thread_count() == baseline_active
    assert api_main._current_confirmed_hanging_thread_count() == baseline_confirmed

    # 2) Gerçek timeout: confirmed sayaç artmalı.
    def slow() -> str:
        time.sleep(0.3)
        return "done"

    try:
        api_main._run_with_timeout(slow, timeout=0.05)
    except api_main.concurrent.futures.TimeoutError:
        pass
    else:
        raise AssertionError("Bu çağrı gerçek bir timeout ile sonuçlanmalıydı.")

    assert (
        api_main._current_confirmed_hanging_thread_count()
        == baseline_confirmed + 1
    )

    # 3) Arka plandaki thread nihayet doğal olarak bitince confirmed
    # sayaç da geri azalmalı (geç de olsa muhasebe doğru kalır).
    deadline = time.monotonic() + 2.0
    while (
        api_main._current_confirmed_hanging_thread_count() > baseline_confirmed
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    assert api_main._current_confirmed_hanging_thread_count() == baseline_confirmed
    assert api_main._current_hanging_thread_count() == baseline_active


def test_harmonize_endpoint_enforces_total_request_budget(monkeypatch) -> None:
    """1. tarama kendi sınırı altında bitse bile, toplam bütçe önceki
    taramalarca tüketilmişse 2. taramaya HİÇ başlanmadan 504 dönmeli.

    Gerçek `time.sleep` yerine `time.monotonic()` deterministik bir
    sırayla sahteleniyor -- gerçek zamanlamaya dayanan testler bu ortamda
    (bkz. bugünkü ~25-45 dk arka plan süreç ölümü keşfi) kırılgan olabilir,
    bu yüzden saat sahtelenerek yarış koşulu riski TAMAMEN ortadan
    kaldırıldı.
    """

    cursor = FakeCursor(_complete_tcga_scans())
    monkeypatch.setattr(
        api_main,
        "get_db_connection",
        lambda: FakeConnection(cursor),
    )
    monkeypatch.setattr(api_main, "resolve_nas_path", lambda path: path)
    monkeypatch.setattr(
        api_main, "apply_n4_bias_correction", lambda path: f"{path}.n4"
    )
    monkeypatch.setattr(
        api_main, "apply_zscore_normalization", lambda path, source: f"{path}.zscore"
    )
    monkeypatch.setenv("GBMAID_HARMONIZE_SCAN_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("GBMAID_HARMONIZE_TOTAL_TIMEOUT_SECONDS", "5")

    # Sıra: [deadline hesabı, 1. tarama bütçe kontrolü, 2. tarama bütçe
    # kontrolü]. 1. tarama "0" anında (bütçe dolu) başlar ve biter (mock
    # fonksiyonlar time.monotonic çağırmaz); 2. tarama kontrolünde saat
    # deadline'ı çoktan geçmiş gibi ilerletiliyor.
    clock_values = iter([0.0, 0.0, 100.0])
    monkeypatch.setattr(api_main.time, "monotonic", lambda: next(clock_values))

    try:
        api_main.harmonize_patient("TCGA-02-0003")
    except api_main.HTTPException as exc:
        assert exc.status_code == 504
        assert "toplam" in str(exc.detail)
    else:
        raise AssertionError("Toplam bütçe aşımı sessizce/200 dönmemeliydi.")


def test_run_with_timeout_race_condition_never_exceeds_limit_under_real_concurrency(
    monkeypatch,
) -> None:
    """GERÇEK eşzamanlılık: limit GERÇEKTEN aşılmamalı (TOCTOU regresyonu).

    2026-08-12 Codex review-gate CRITICAL bulgusu: kontrol
    (`current >= limit`) ile artırma AYRI kritik bölümlerdeydi -- iki
    eşzamanlı çağrı aynı anda "current < limit" görüp ikisi de
    geçebiliyordu. Bu, o pencereyi mock'lanmış sahte zamanlama İLE DEĞİL,
    gerçek `threading.Thread` + `threading.Barrier` ile SENKRONİZE
    başlatılan 5 eşzamanlı "asılı kalan" çağrıyla (limit=2) zorlar --
    tam olarak yakalanması gereken race condition budur.
    """

    monkeypatch.setenv("GBMAID_HARMONIZE_MAX_HANGING_THREADS", "2")

    # Test yalıtımı: başka bir testten kalan asılı bir arka plan thread'i
    # varsa (done-callback henüz tetiklenmemiş olabilir) kısa bir süre
    # sayacın 0'a dönmesini bekle. Diğer tüm testler kendi asılı
    # thread'lerini bu noktaya kadar temizliyor (bkz. yukarıdaki testler),
    # bu yüzden normalde beklemeden geçer.
    settle_deadline = time.monotonic() + 2.0
    while (
        api_main._current_hanging_thread_count() > 0
        and time.monotonic() < settle_deadline
    ):
        time.sleep(0.05)
    baseline = api_main._current_hanging_thread_count()
    assert baseline == 0, (
        "Bu test önceki bir testten kalan asılı thread'le kirlenmiş "
        f"başlıyor (baseline={baseline}), sonuçlar güvenilir olmaz."
    )

    n_callers = 5
    limit = 2
    start_barrier = threading.Barrier(n_callers)
    hang_release = threading.Event()

    def hang() -> str:
        # Barrier, TÜM çağıranların `_run_with_timeout`'a neredeyse aynı
        # anda girmesini garanti eder -- eski (kırık) koddaki
        # kontrol-sonra-artır penceresini gerçek thread'lerle zorlar.
        # `hang_release` set edilene kadar bekler (kısa timeout ile
        # hepsi timeout'a uğrayacak şekilde tasarlandı).
        hang_release.wait(timeout=5)
        return "done"

    results: list[str] = []
    results_lock = threading.Lock()

    def caller() -> None:
        start_barrier.wait(timeout=5)
        try:
            api_main._run_with_timeout(hang, timeout=0.05)
        except api_main.concurrent.futures.TimeoutError:
            outcome = "timed_out"
        except api_main.HangingThreadCapacityExceededError:
            outcome = "rejected"
        else:
            outcome = "unexpected_success"
        with results_lock:
            results.append(outcome)

    threads = [threading.Thread(target=caller) for _ in range(n_callers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    # Asılı bırakılan thread'leri serbest bırak, sayacın gerçekten geri
    # azaldığını bekle -- sonraki testleri kirletmesin.
    hang_release.set()
    cleanup_deadline = time.monotonic() + 2.0
    while (
        api_main._current_hanging_thread_count() > 0
        and time.monotonic() < cleanup_deadline
    ):
        time.sleep(0.05)

    assert len(results) == n_callers
    assert results.count("unexpected_success") == 0
    # EN KRİTİK iddia: limit GERÇEKTEN aşılmadı -- rezervasyon atomik
    # olduğu için TAM OLARAK `limit` kadar çağrı slot alıp timeout'a
    # uğrayabildi, kalanı hiç thread açmadan 503-eşdeğeri istisna aldı.
    assert results.count("timed_out") == limit
    assert results.count("rejected") == n_callers - limit
    assert api_main._current_hanging_thread_count() == 0
