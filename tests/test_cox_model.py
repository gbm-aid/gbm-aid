from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import pipeline.cox_model as cox_model_module
from pipeline.cox_model import (
    COX_TRAINING_ALLOWED_SOURCES,
    EXCLUDED_FIRST_ORDER_ABSOLUTE_SCALE_FEATURES,
    STABLE_FEATURES_ICC60,
    ComBatIdentityAssumptionError,
    CoxTrainingLeakageError,
    RegionPivotError,
    _assemble_training_frame,
    _build_penalizer_argument,
    assert_combat_identity_for_training_pool,
    assert_training_pool_sources,
    build_training_frame,
    canonicalize_region_label,
    compute_shap_values,
    evaluate_external_test,
    pivot_radiomics_long_to_wide,
    run_nested_cv,
    run_stratified_cv,
    select_features_lasso,
    standardize_columns_fold_safe,
)

# NOT (2026-08-13, Codex HIGH-1 sertleştirmesi -- bkz. decisions/2026-08-12-
# cox-model-source-guard-canonicalization.md "[2026-08-13] TAKİP-3"):
# `_assemble_training_frame()` artık PRIVATE (eskiden `assemble_training_
# frame()` adıyla public'ti). Bu test dosyasındaki doğrudan
# `_assemble_training_frame()` çağrıları BİLİNÇLİ bir istisna --
# `tools/smoke_test_cox_model.py`'nin kendi docstring'indeki desenle AYNI
# gerekçe: guard'sız assemble mantığının KENDİSİNİ (dummy-encoding,
# event/duration türetme, kanonikleştirme) izole test etmek için, guard'ın
# (`assert_training_pool_sources()`/`build_training_frame()`) davranışından
# BAĞIMSIZ olarak. Üretim kodu ASLA bu deseni izlememeli.


def _make_synthetic_survival_frame(
    n: int, *, n_features: int = 6, informative_index: int = 0, seed: int = 7
) -> tuple[pd.DataFrame, list[str]]:
    """Bilinen bir prognostik özellikle sentetik Cox verisi üret.

    Exponential-baseline hazard + tek bilgilendirici özellik
    (`informative_index`) -- LASSO'nun bu özelliği gerçekten seçtiğini
    doğrulamak için standart bir simülasyon deseni.
    """

    rng = np.random.default_rng(seed)
    feature_columns = [f"feat_{i}" for i in range(n_features)]
    features = rng.normal(size=(n, n_features))

    true_beta = 1.5
    linear_predictor = true_beta * features[:, informative_index]
    baseline_hazard = 0.02
    u = rng.uniform(size=n)
    event_time = -np.log(u) / (baseline_hazard * np.exp(linear_predictor))

    censoring_time = rng.uniform(low=1.0, high=np.percentile(event_time, 90), size=n)
    duration = np.minimum(event_time, censoring_time)
    event = (event_time <= censoring_time).astype(int)

    frame = pd.DataFrame(features, columns=feature_columns)
    frame["survival_days"] = duration
    frame["event"] = event
    return frame, feature_columns


def test_assemble_training_frame_derives_event_and_drops_unrecognized_status() -> None:
    feature_frame = pd.DataFrame(
        {"feat_a": [1.0, 2.0, 3.0, 4.0], "feat_b": [0.5, 0.6, 0.7, 0.8]},
        index=["p1", "p2", "p3", "p4"],
    )
    patients_frame = pd.DataFrame(
        {
            "source": ["UPenn", "UPenn", "LUMIERE", "UPenn"],
            "survival_days": [100, 200, 300, None],
            "vital_status": ["DECEASED", "ALIVE", "DECEASED", "Lost to Follow-up"],
        },
        index=["p1", "p2", "p3", "p4"],
    )

    frame, report = _assemble_training_frame(feature_frame, patients_frame)

    assert report.n_input_rows == 4
    # p4: hem tanınmayan vital_status hem eksik survival_days -- vital_status
    # kontrolü önce çalışır, bu satır orada düşer.
    assert report.dropped_unrecognized_vital_status == {"Lost to Follow-up": 1}
    assert report.n_output_rows == 3
    assert list(frame["event"]) == [1, 0, 1]
    assert "source_LUMIERE" in frame.columns
    assert "source_UPenn" not in frame.columns  # referans seviye drop edilir


def test_assemble_training_frame_canonicalizes_db_style_source_aliases() -> None:
    """2026-08-13 FIX -- `_assemble_training_frame()`'in kapsam-dışı
    bırakılmış MEDIUM riski (bkz. decisions/2026-08-12-cox-model-
    source-guard-canonicalization.md "Çağıranlar kontrolü" bölümü):
    DB'nin GERÇEK `dataset_sources.source_name` değeri (`"UPenn-GBM"`,
    `"LUMIERE"`) doğrudan `source_col`'a beslenirse eskiden
    `pd.get_dummies()` `"source_UPenn-GBM"` kolonu üretiyordu,
    `source_reference="UPenn"` (varsayılan) ile HİÇ eşleşmiyordu ve
    `ValueError` fırlatılıyordu. Artık `merged[source_col]`
    `_canonical_source_series()` ile kanonikleştiriliyor -- dummy
    kolonlar her zaman kanonik (`source_UPenn`, `source_LUMIERE`)
    olmalı, ValueError FIRLAMAMALI.
    """

    feature_frame = pd.DataFrame(
        {"feat_a": [1.0, 2.0, 3.0, 4.0]},
        index=["p1", "p2", "p3", "p4"],
    )
    patients_frame = pd.DataFrame(
        {
            # DB-stili ham alias'lar -- düz "UPenn"/"LUMIERE" DEĞİL.
            "source": ["UPenn-GBM", "upenn-gbm", " UPenn-GBM ", "LUMIERE"],
            "survival_days": [100, 200, 300, 400],
            "vital_status": ["DECEASED", "ALIVE", "DECEASED", "ALIVE"],
        },
        index=["p1", "p2", "p3", "p4"],
    )

    frame, report = _assemble_training_frame(feature_frame, patients_frame)

    assert report.n_output_rows == 4
    assert "source_LUMIERE" in frame.columns
    # Referans seviye (kanonik "UPenn") drop edilir, ham "UPenn-GBM"
    # kolonu asla ÜRETİLMEZ (kanonikleştirme dummy encoding'den ÖNCE
    # yapılıyor).
    assert "source_UPenn" not in frame.columns
    assert "source_UPenn-GBM" not in frame.columns
    # report.sources da kanonik anahtarlarla raporlanmalı -- ham
    # "UPenn-GBM"/"upenn-gbm"/" UPenn-GBM " ayrı ayrı SAYILMAMALI.
    assert report.sources == {"UPenn": 3, "LUMIERE": 1}


def test_assemble_training_frame_rejects_unknown_source_loudly() -> None:
    """SOURCE_ALIASES'in tanımadığı bir kaynak adı SESSİZCE geçilmemeli
    -- proje kuralı "sessiz fallback yasak"."""

    feature_frame = pd.DataFrame({"feat_a": [1.0, 2.0]}, index=["p1", "p2"])
    patients_frame = pd.DataFrame(
        {
            "source": ["UPenn", "MysteryCohort"],
            "survival_days": [100, 200],
            "vital_status": ["DECEASED", "ALIVE"],
        },
        index=["p1", "p2"],
    )

    with pytest.raises(ValueError, match="tanınmayan"):
        _assemble_training_frame(feature_frame, patients_frame)


def test_assemble_training_frame_canonicalizes_raw_source_reference() -> None:
    """`source_reference` de (`merged[source_col]` gibi) ham bir alias
    olarak verilirse kanonikleştirilmeli -- aksi halde kanonikleştirilmiş
    dummy kolonlarla (`source_UPenn`) asla eşleşmezdi (`fit_combat_
    harmonization()`'ın FIX-5'inde `reference_batch` için düzeltilen
    AYNI hata sınıfı)."""

    feature_frame = pd.DataFrame(
        {"feat_a": [1.0, 2.0, 3.0]}, index=["p1", "p2", "p3"]
    )
    patients_frame = pd.DataFrame(
        {
            "source": ["UPenn-GBM", "UPenn-GBM", "LUMIERE"],
            "survival_days": [100, 200, 300],
            "vital_status": ["DECEASED", "ALIVE", "DECEASED"],
        },
        index=["p1", "p2", "p3"],
    )

    # source_reference ham bir alias olarak verildi -- "UPenn" DEĞİL.
    frame, report = _assemble_training_frame(
        feature_frame, patients_frame, source_reference="UPenn-GBM"
    )

    assert "source_LUMIERE" in frame.columns
    assert "source_UPenn" not in frame.columns
    assert report.n_output_rows == 3


def test_assert_training_pool_sources_rejects_tcga() -> None:
    covariates = pd.DataFrame({"source": ["UPenn", "LUMIERE", "TCGA-GBM"]})
    # 2026-08-12: guard artık `canonical_source()` ile normalize ediyor --
    # hata mesajı kanonikleştirilmiş "TCGA" değerini gösterir, ham
    # "TCGA-GBM" değerini DEĞİL (bkz. decisions/2026-08-12-cox-model-
    # source-guard-canonicalization.md). `fit_combat_harmonization()`'ın
    # FIX-3'ündeki AYNI desenle tutarlı.
    with pytest.raises(CoxTrainingLeakageError, match=r"\['TCGA'\]"):
        assert_training_pool_sources(covariates)

    # UPenn+LUMIERE tek başına sorun çıkarmamalı
    assert_training_pool_sources(pd.DataFrame({"source": ["UPenn", "LUMIERE"]}))
    assert COX_TRAINING_ALLOWED_SOURCES == {"UPenn", "LUMIERE"}


def test_assert_training_pool_sources_accepts_db_style_source_aliases() -> None:
    """2026-08-12 FIX -- aynı hata sınıfı `fit_combat_harmonization()`'ın
    FIX-3'ünde düzeltilenle BİREBİR AYNI (bkz. decisions/2026-08-12-cox-
    model-source-guard-canonicalization.md).

    `dataset_sources.source_name` DB kolonunun GERÇEK değeri literal
    "UPenn-GBM"/"LUMIERE" -- düz "UPenn" DEĞİL (bkz. bu modülün
    `verify_event_counts()` sorgusu). Guard artık bu DB-stili alias'ları
    (case-insensitive, boşluk toleranslı) kabul etmeli, yanlış-pozitif
    `CoxTrainingLeakageError` FIRLATMAMALI.
    """

    covariates = pd.DataFrame(
        {"source": ["UPenn-GBM", "upenn-gbm", " UPenn-GBM ", "LUMIERE", "lumiere"]}
    )
    # Hiçbir exception fırlatılmamalı.
    assert_training_pool_sources(covariates)


def test_assert_training_pool_sources_rejects_tcga_alias_variants() -> None:
    """TCGA'nın TÜM bilinen alias varyantları (SOURCE_ALIASES'te tanımlı)
    hâlâ reddedilmeli -- kanonikleştirme, TCGA'yı YAKALAMAMA riski
    getirmemeli (regresyon)."""

    for variant in ("TCGA", "TCGA-GBM", "tcga-gbm", " TCGA-GBM ", "Tcga"):
        covariates = pd.DataFrame({"source": ["UPenn", "LUMIERE", variant]})
        with pytest.raises(CoxTrainingLeakageError, match=r"\['TCGA'\]"):
            assert_training_pool_sources(covariates)


def test_assert_training_pool_sources_rejects_unknown_source_loudly() -> None:
    """Tanınmayan (SOURCE_ALIASES'te olmayan) bir kaynak adı SESSİZCE
    kabul edilmemeli -- whitelist mantığı gereği reddedilmeli (sessiz
    fallback yasak, CLAUDE.md kuralı)."""

    covariates = pd.DataFrame({"source": ["UPenn", "LUMIERE", "MysteryCohort"]})
    with pytest.raises(CoxTrainingLeakageError, match="tanımadığı"):
        assert_training_pool_sources(covariates)


def test_assert_training_pool_sources_rejects_widened_allowed_sources_with_tcga() -> None:
    """2026-08-13 -- Codex HIGH-2 bulgusu: `allowed_sources` bir SABİT
    değil bir PARAMETRE olduğu için, önceden bir çağıran
    `allowed_sources={"UPenn", "LUMIERE", "TCGA"}` geçirip whitelist'i
    GENİŞLETEBİLİR, guard TCGA'yı SESSİZCE kabul EDEBİLİRDİ (`covariates`
    içinde TCGA olsa bile hiçbir exception fırlamazdı). Artık
    `_validate_allowed_sources()` bunu ÖNCEDEN, `covariates`'e hiç
    bakmadan reddediyor -- `CoxTrainingLeakageError` `covariates`'te
    TCGA satırı OLMASA bile fırlar (genişletmenin kendisi hata, verinin
    içeriği değil)."""

    covariates = pd.DataFrame({"source": ["UPenn", "LUMIERE"]})  # TCGA YOK
    widened = {"UPenn", "LUMIERE", "TCGA"}

    with pytest.raises(CoxTrainingLeakageError, match="ALT KÜMESİ"):
        assert_training_pool_sources(covariates, allowed_sources=widened)


def test_assert_training_pool_sources_rejects_allowed_sources_with_unknown_entry() -> None:
    """Genişletme reddi sadece TCGA'ya özgü değil -- `COX_TRAINING_
    ALLOWED_SOURCES`'ta OLMAYAN herhangi bir eleman (hayali bir kaynak
    adı dahil) `allowed_sources`'a eklenirse de reddedilmeli."""

    covariates = pd.DataFrame({"source": ["UPenn", "LUMIERE"]})
    widened = {"UPenn", "LUMIERE", "MarsCohort"}

    with pytest.raises(CoxTrainingLeakageError, match="ALT KÜMESİ"):
        assert_training_pool_sources(covariates, allowed_sources=widened)


def test_assert_training_pool_sources_allows_narrowed_allowed_sources() -> None:
    """`allowed_sources` parametresinin var olma amacı korunmalı --
    whitelist'i DARALTMAK (örn. sadece UPenn-only bir alt-küme testi
    için) hâlâ serbest olmalı, sadece GENİŞLETME yasak."""

    upenn_only_covariates = pd.DataFrame({"source": ["UPenn", "UPenn"]})
    # Daraltma: {"UPenn"} ⊂ COX_TRAINING_ALLOWED_SOURCES -- kabul edilmeli.
    assert_training_pool_sources(upenn_only_covariates, allowed_sources={"UPenn"})

    # Daraltılmış whitelist'te olmayan (ama genel olarak izinli) bir
    # kaynak varsa yine reddedilmeli -- daraltma gerçekten uygulanıyor.
    mixed_covariates = pd.DataFrame({"source": ["UPenn", "LUMIERE"]})
    with pytest.raises(CoxTrainingLeakageError, match=r"\['LUMIERE'\]"):
        assert_training_pool_sources(mixed_covariates, allowed_sources={"UPenn"})


def test_build_training_frame_rejects_widened_allowed_sources_with_tcga() -> None:
    """Aynı HIGH-2 koruması `build_training_frame()` üzerinden de
    çalışmalı (guard'ı `assert_training_pool_sources()`'a delege ediyor,
    ayrı bir doğrulama YAZILMADI ama korumadan MUAF da değil)."""

    feature_frame = pd.DataFrame({"feat_a": [1.0, 2.0]}, index=["p1", "p2"])
    patients_frame = pd.DataFrame(
        {
            "source": ["UPenn", "LUMIERE"],  # TCGA YOK
            "survival_days": [100, 200],
            "vital_status": ["DECEASED", "ALIVE"],
        },
        index=["p1", "p2"],
    )

    with pytest.raises(CoxTrainingLeakageError, match="ALT KÜMESİ"):
        build_training_frame(
            feature_frame,
            patients_frame,
            allowed_sources={"UPenn", "LUMIERE", "TCGA"},
        )


def test_private_assemble_training_frame_not_exposed_under_old_public_name() -> None:
    """2026-08-13 -- Codex HIGH-1 bulgusu: eski public isim
    (`assemble_training_frame`, alt çizgisiz) modülde artık HİÇ mevcut
    olmamalı -- kazara `from pipeline.cox_model import
    assemble_training_frame` yapan biri `ImportError` almalı. Private
    isim (`_assemble_training_frame`) modülde bulunmaya devam etmeli
    (KALDIRILMADI, sadece adı değişti) -- `tools/smoke_test_cox_model.py`
    gibi bilinçli/belgeli çağıranlar hâlâ erişebilmeli."""

    assert not hasattr(cox_model_module, "assemble_training_frame")
    assert hasattr(cox_model_module, "_assemble_training_frame")

    with pytest.raises(ImportError):
        exec("from pipeline.cox_model import assemble_training_frame")


def test_build_training_frame_rejects_tcga_before_assembling_anything() -> None:
    """2026-08-13 -- `build_training_frame()`'in ana amacı: TCGA'yı
    (veya DB-stili herhangi bir TCGA alias'ını) `_assemble_training_
    frame()`'e HİÇ ulaşmadan, guard'ı atlamayı imkansız kılarak
    reddetmesi (bkz. decisions/2026-08-12-cox-model-source-guard-
    canonicalization.md "[2026-08-13] TAKİP-2").

    Not: `patients_frame`'deki diğer satırlar (UPenn/LUMIERE) kendi
    başına gayet geçerli/assemble edilebilir olsa bile -- guard,
    `_assemble_training_frame()`'in yaptığı `feature_frame` inner
    join'inden ÖNCE, tüm `patients_frame` üzerinde çalışır. Yani TCGA
    satırının `feature_frame`'de karşılığı olmasa bile (join onu zaten
    düşürecek olsa bile) reddediliyor -- "inner join zaten temizler"
    gibi örtük/sessiz bir varsayıma GÜVENİLMEDİĞİni kanıtlar.
    """

    feature_frame = pd.DataFrame(
        {"feat_a": [1.0, 2.0]}, index=["p1", "p2"]
    )
    patients_frame = pd.DataFrame(
        {
            "source": ["UPenn", "TCGA-GBM"],
            "survival_days": [100, 200],
            "vital_status": ["DECEASED", "ALIVE"],
        },
        index=["p1", "p3"],  # p3 -> feature_frame'de HİÇ YOK (join zaten düşürürdü)
    )

    with pytest.raises(CoxTrainingLeakageError, match=r"\['TCGA'\]"):
        build_training_frame(feature_frame, patients_frame)


def test_build_training_frame_rejects_tcga_alias_variants() -> None:
    """Regresyon -- TCGA'nın DB-stili tüm alias varyantları (`"TCGA"`,
    `"TCGA-GBM"`, küçük harf, boşluklu) `build_training_frame()`
    üzerinden de reddedilmeli, kanonikleştirme TCGA'yı KAÇIRMAMALI."""

    feature_frame = pd.DataFrame(
        {"feat_a": [1.0, 2.0, 3.0]}, index=["p1", "p2", "p3"]
    )
    for variant in ("TCGA", "TCGA-GBM", "tcga-gbm", " TCGA-GBM ", "Tcga"):
        patients_frame = pd.DataFrame(
            {
                "source": ["UPenn", "LUMIERE", variant],
                "survival_days": [100, 200, 300],
                "vital_status": ["DECEASED", "ALIVE", "DECEASED"],
            },
            index=["p1", "p2", "p3"],
        )
        with pytest.raises(CoxTrainingLeakageError, match=r"\['TCGA'\]"):
            build_training_frame(feature_frame, patients_frame)


def test_build_training_frame_processes_upenn_lumiere_correctly() -> None:
    """`build_training_frame()`'in UPenn+LUMIERE (ve DB-stili ham
    alias'ları) doğru işlediğini, çıktısının doğrudan
    `_assemble_training_frame()` çağırmakla BİREBİR AYNI olduğunu kanıtla --
    guard eklemek assemble mantığını DEĞİŞTİRMEMELİ.

    NOT (2026-08-14, C5 -- bkz. decisions/2026-08-13-combat-ozdeslik-
    bulgusu-ve-kapsami.md "ZORUNLU KORUMA"): bu test SADECE dummy-
    encoding/kanonikleştirme MEKANİĞİNİ kanıtlıyor, LUMIERE'in GERÇEK
    Cox eğitimine katılabileceğini İDDİA ETMİYOR (zaten katılamıyor,
    bkz. decisions/2026-08-06-lumiere-event-vital-status-eksik.md).
    `check_combat_identity=False` BİLİNÇLİ olarak geçiliyor -- aksi
    halde bu senaryo (LUMIERE + varsayılan `source_reference="UPenn"`)
    artık YENİ `assert_combat_identity_for_training_pool()` guard'ına
    (C5) takılırdı: LUMIERE whitelist'te İZİNLİ olsa bile ComBat'ın
    referans batch'i UPenn olduğu için LUMIERE'in havuzda bulunması
    ComBat'ı artık özdeşlik OLMAKTAN çıkarır. Bu YENİ guard'ın gerçek
    (varsayılan açık) davranışı ayrı bir testte kanıtlanıyor, bkz.
    `test_build_training_frame_rejects_lumiere_by_default_due_to_combat_identity_guard`.
    """

    feature_frame = pd.DataFrame(
        {"feat_a": [1.0, 2.0, 3.0, 4.0]},
        index=["p1", "p2", "p3", "p4"],
    )
    patients_frame = pd.DataFrame(
        {
            # DB-stili ham alias'lar -- assert_training_pool_sources()
            # VE _assemble_training_frame()'in İKİSİNİN de bunları doğru
            # kanonikleştirdiğini kanıtlar.
            "source": ["UPenn-GBM", "upenn-gbm", " UPenn-GBM ", "LUMIERE"],
            "survival_days": [100, 200, 300, 400],
            "vital_status": ["DECEASED", "ALIVE", "DECEASED", "ALIVE"],
        },
        index=["p1", "p2", "p3", "p4"],
    )

    frame, report = build_training_frame(
        feature_frame, patients_frame, check_combat_identity=False
    )
    direct_frame, direct_report = _assemble_training_frame(feature_frame, patients_frame)

    pd.testing.assert_frame_equal(frame, direct_frame)
    assert report == direct_report
    assert report.n_output_rows == 4
    assert "source_LUMIERE" in frame.columns
    assert "source_UPenn" not in frame.columns
    assert report.sources == {"UPenn": 3, "LUMIERE": 1}


def test_build_training_frame_rejects_lumiere_by_default_due_to_combat_identity_guard() -> None:
    """C5 (2026-08-14) -- `build_training_frame()`'in VARSAYILAN
    (`check_combat_identity=True`) davranışı: LUMIERE whitelist'te
    İZİNLİ olsa bile (TCGA gibi KESİN reddedilmez), ComBat'ın referans
    batch'i (`source_reference`, varsayılan "UPenn") DIŞINDA bir kaynak
    havuzda varsa `ComBatIdentityAssumptionError` fırlatılmalı --
    whitelist guard'ı (`assert_training_pool_sources()`) bunu
    YAKALAMAZ (LUMIERE zaten izinli), bu YÜZDEN ayrı bir guard
    gerekiyordu (bkz. decisions/2026-08-13-combat-ozdeslik-bulgusu-ve-
    kapsami.md "TESPİT EDİLEN MAYIN")."""

    feature_frame = pd.DataFrame(
        {"feat_a": [1.0, 2.0, 3.0]}, index=["p1", "p2", "p3"]
    )
    patients_frame = pd.DataFrame(
        {
            "source": ["UPenn", "UPenn", "LUMIERE"],
            "survival_days": [100, 200, 300],
            "vital_status": ["DECEASED", "ALIVE", "DECEASED"],
        },
        index=["p1", "p2", "p3"],
    )

    # whitelist guard'ı tek başına bunu ENGELLEMEZ (LUMIERE izinli).
    assert_training_pool_sources(patients_frame)

    # ama build_training_frame() varsayılan olarak C5 guard'ını da
    # çalıştırır ve burada durur.
    with pytest.raises(ComBatIdentityAssumptionError, match=r"\['LUMIERE'\]"):
        build_training_frame(feature_frame, patients_frame)


def test_build_training_frame_guard_cannot_be_bypassed_by_forgetting_it() -> None:
    """`build_training_frame()`'in tüm görevi: eskiden mümkün olan
    "sadece `_assemble_training_frame()`'i çağırıp guard'ı unutma"
    senaryosunu KODDA imkansız kılmak. Bu test, aynı (TCGA'lı)
    `patients_frame` ile (a) çıplak `_assemble_training_frame()`'in
    HİÇBİR itiraz etmeden `source_TCGA` dummy kolonu ürettiğini
    (kasıtlı kapsam sınırı, regresyon değil), (b) `build_training_
    frame()`'in AYNI veriyi guard'da yakalayıp durdurduğunu yan yana
    kanıtlar -- yani guard'ı "unutmak" artık `build_training_frame()`
    kullanan hiçbir çağıran için MÜMKÜN DEĞİL.
    """

    feature_frame = pd.DataFrame(
        {"feat_a": [1.0, 2.0, 3.0]}, index=["p1", "p2", "p3"]
    )
    patients_frame = pd.DataFrame(
        {
            "source": ["UPenn", "LUMIERE", "TCGA-GBM"],
            "survival_days": [100, 200, 300],
            "vital_status": ["DECEASED", "ALIVE", "DECEASED"],
        },
        index=["p1", "p2", "p3"],
    )

    # (a) Çıplak _assemble_training_frame() -- guard'sız -- TCGA'yı
    # sessizce kabul edip source_TCGA dummy kolonu üretir (kasıtlı
    # kapsam sınırı, bkz. fonksiyonun kendi docstring'i).
    bare_frame, _ = _assemble_training_frame(feature_frame, patients_frame)
    assert "source_TCGA" in bare_frame.columns

    # (b) build_training_frame() AYNI veriyle guard'da durur --
    # source_TCGA dummy kolonu ÜRETİLEMEZ, hiç assemble aşamasına
    # ulaşılmaz.
    with pytest.raises(CoxTrainingLeakageError, match=r"\['TCGA'\]"):
        build_training_frame(feature_frame, patients_frame)


def test_select_features_lasso_recovers_informative_feature() -> None:
    frame, feature_columns = _make_synthetic_survival_frame(n=300)

    # Not: lifelines'ın L1 penaltısı katsayıları TAM sıfıra indirmiyor
    # (deneyle doğrulandı) -- gürültü özellikleri 1e-6..1e-8 mertebesinde
    # kalıyor, sinyal 1e-2..1'de. Bu yüzden orta-güçlü bir penalizer
    # (0.5) + gevşek olmayan bir tolerans (1e-4, fonksiyonun varsayılanı)
    # gerekiyor -- penalizer=0.05 gibi zayıf bir değerle hiçbir şey
    # elenmiyor (bu da ayrıca doğrulandı, bkz. modeling-agent oturum notları).
    selected, cox, coefficients = select_features_lasso(
        frame, feature_columns, penalizer=0.5, l1_ratio=1.0
    )

    assert "feat_0" in selected  # bilgilendirici özellik (true_beta=1.5) seçilmeli
    assert len(selected) < len(feature_columns)  # LASSO gürültü özelliklerini elemeli
    assert coefficients["feat_0"] > 0  # işaret doğru yönde (yüksek risk)


def test_run_stratified_cv_produces_reasonable_c_index() -> None:
    frame, feature_columns = _make_synthetic_survival_frame(n=300)

    cv_results = run_stratified_cv(
        frame, feature_columns, n_splits=3, penalizer=0.05, l1_ratio=1.0
    )

    assert len(cv_results) == 3
    assert (cv_results["n_test"] > 0).all()
    assert (cv_results["c_index"] > 0.5).all()  # rastgeleden iyi olmalı (gerçek sinyal var)
    assert (cv_results["c_index"] <= 1.0).all()


def test_evaluate_external_test_returns_ci_bounds() -> None:
    train_frame, feature_columns = _make_synthetic_survival_frame(n=250, seed=1)
    external_frame, _ = _make_synthetic_survival_frame(n=50, seed=2)

    _, cox, _ = select_features_lasso(
        train_frame, feature_columns, penalizer=0.05, l1_ratio=1.0
    )

    result = evaluate_external_test(
        cox, external_frame, feature_columns, n_bootstrap=200, seed=3
    )

    assert result["n_patients"] == 50
    assert 0.0 <= result["c_index"] <= 1.0
    assert result["ci_lower"] <= result["c_index"] <= result["ci_upper"]
    assert result["n_bootstrap_valid"] > 0


def test_evaluate_external_test_requires_explicit_extra_columns() -> None:
    train_frame, feature_columns = _make_synthetic_survival_frame(n=200, seed=4)
    rng = np.random.default_rng(11)
    train_frame["source_LUMIERE"] = rng.integers(0, 2, size=len(train_frame))
    external_frame, _ = _make_synthetic_survival_frame(n=30, seed=5)

    _, cox, _ = select_features_lasso(
        train_frame,
        feature_columns,
        extra_columns=["source_LUMIERE"],
        penalizer=0.05,
        l1_ratio=1.0,
    )

    with pytest.raises(ValueError, match="source dummy"):
        evaluate_external_test(
            cox,
            external_frame,
            feature_columns,
            extra_columns=["source_LUMIERE"],
        )


def test_compute_shap_values_synthetic_only_shape_matches() -> None:
    """SHAP -- SADECE sentetik veriyle birim testi (kapsam notuna bkz.

    pipeline.cox_model.compute_shap_values docstring'i: bu fonksiyon
    gerçek üretim verisiyle bu Hafta 3 iskelet aşamasında
    ÇALIŞTIRILMAMALI, sadece plumbing'in çalıştığı burada, küçük
    sentetik veriyle kanıtlanıyor.
    """

    frame, feature_columns = _make_synthetic_survival_frame(n=40, seed=9)
    _, cox, _ = select_features_lasso(
        frame, feature_columns, penalizer=0.05, l1_ratio=1.0
    )

    shap_frame = compute_shap_values(
        cox, frame.head(10), feature_columns, n_background=15, seed=1
    )

    assert shap_frame.shape == (10, len(feature_columns))
    assert list(shap_frame.columns) == feature_columns


# =====================================================================
# C4 -- STABLE_FEATURES_ICC60 (93 özellik) prespesifikasyon testleri
# =====================================================================


def test_stable_features_icc60_has_expected_composition() -> None:
    """decisions/2026-08-13-bolge-stratejisi-ve-modelleme-protokolu.md
    Bölüm 4.1b: 93 = 14 shape + 4 first-order + 75 texture."""

    assert len(STABLE_FEATURES_ICC60) == 93
    assert len(set(STABLE_FEATURES_ICC60)) == 93  # tekrar YOK

    shape_features = [f for f in STABLE_FEATURES_ICC60 if f.startswith("original_shape_")]
    first_order_features = [
        f for f in STABLE_FEATURES_ICC60 if f.startswith("original_firstorder_")
    ]
    texture_features = [
        f
        for f in STABLE_FEATURES_ICC60
        if not f.startswith("original_shape_") and not f.startswith("original_firstorder_")
    ]

    assert len(shape_features) == 14
    assert len(first_order_features) == 4
    assert len(texture_features) == 75


def test_stable_features_icc60_first_order_subset_matches_decision_doc() -> None:
    """Karar dosyasının STABİL first-order dörtlüsü (Entropy/Uniformity/
    Skewness/Kurtosis) BİREBİR bu listede olmalı, başka first-order
    özelliği OLMAMALI."""

    expected = {
        "original_firstorder_Entropy",
        "original_firstorder_Kurtosis",
        "original_firstorder_Skewness",
        "original_firstorder_Uniformity",
    }
    actual = {f for f in STABLE_FEATURES_ICC60 if f.startswith("original_firstorder_")}
    assert actual == expected


def test_stable_features_icc60_excludes_absolute_scale_first_order_features() -> None:
    """Elenen 14 mutlak-ölçek first-order özelliği (Energy/TotalEnergy/
    RMS/Variance/Mean/... ) STABLE_FEATURES_ICC60'ta OLMAMALI -- karar
    dosyasının açıkça listelediği 14 özellik BİREBİR eşleşmeli."""

    assert len(EXCLUDED_FIRST_ORDER_ABSOLUTE_SCALE_FEATURES) == 14
    assert len(set(EXCLUDED_FIRST_ORDER_ABSOLUTE_SCALE_FEATURES)) == 14

    for excluded in EXCLUDED_FIRST_ORDER_ABSOLUTE_SCALE_FEATURES:
        assert excluded not in STABLE_FEATURES_ICC60

    # Toplam first-order evreni (stabil + elenen) tam 18 olmalı (karar
    # dosyası: first-order n=18).
    all_first_order = set(EXCLUDED_FIRST_ORDER_ABSOLUTE_SCALE_FEATURES) | {
        f for f in STABLE_FEATURES_ICC60 if f.startswith("original_firstorder_")
    }
    assert len(all_first_order) == 18


# =====================================================================
# C1 -- Bölge kanonikleştirme + uzun->geniş pivot testleri
# =====================================================================


def test_canonicalize_region_label_maps_all_known_source_region_pairs() -> None:
    assert canonicalize_region_label("UPenn", "NC") == "NC"
    assert canonicalize_region_label("UPenn-GBM", "ED") == "ED"  # ham alias kabul edilir
    assert canonicalize_region_label("UPenn", "WT_derived") == "WT"
    assert canonicalize_region_label("UPenn", "TC_derived") == "TC"

    assert canonicalize_region_label("LUMIERE", "Necrosis") == "NC"
    assert canonicalize_region_label("LUMIERE", "Edema") == "ED"
    assert canonicalize_region_label("LUMIERE", "Contrast-enhancing") == "ET"
    assert canonicalize_region_label("LUMIERE", "WT_derived") == "WT"
    assert canonicalize_region_label("LUMIERE", "TC_derived") == "TC"

    assert canonicalize_region_label("TCGA-GBM", "WT") == "WT"
    assert canonicalize_region_label("TCGA-GBM", "TC") == "TC"


def test_canonicalize_region_label_accepts_ucsf_source() -> None:
    """2026-09-12 regresyon testi -- canlı bug: `/predict` 8/8 UCSF
    hastasında "REGION_NAME_ALIASES kaynağı tanımıyor: 'UCSF'" ile 500
    patlıyordu (`pipeline/harmonization.py::SOURCE_ALIASES`'a 2026-08-18'de
    "ucsf-pdgm" -> "UCSF" eklenmişti ama bu dosyanın `REGION_NAME_ALIASES`
    sözlüğü hiç güncellenmemişti). Canlı DB'de doğrulandı (readonly
    SELECT, 2026-09-12): UCSF'in `tumor_region` değerleri UPenn'inkiyle
    BİREBİR AYNI (NC/ED/ET/WT_derived/TC_derived,
    segmentation_tool='UCSF-PDGM-PyRadiomics-107-C32'), bu yüzden eşleme
    UPenn'in bloğunun mekanik kopyası.
    """

    assert canonicalize_region_label("UCSF-PDGM", "NC") == "NC"
    assert canonicalize_region_label("UCSF-PDGM", "ED") == "ED"
    assert canonicalize_region_label("UCSF-PDGM", "ET") == "ET"
    assert canonicalize_region_label("UCSF-PDGM", "WT_derived") == "WT"
    assert canonicalize_region_label("UCSF-PDGM", "TC_derived") == "TC"
    # Ham alias da (SOURCE_ALIASES'in kabul ettiği "ucsf" kısa formu) kabul edilir.
    assert canonicalize_region_label("ucsf", "NC") == "NC"


def test_canonicalize_region_label_ucsf_addition_does_not_change_other_sources() -> None:
    """UCSF eklenmesi UPenn/LUMIERE/TCGA davranışını DEĞİŞTİRMEMELİ --
    additif değişiklik kanıtı (pozitif kontrol, mevcut üç kaynak testinin
    tekrarı, bu testin amacı sadece UCSF eklemesiyle YAN YANA aynı dosyada
    durup regresyon-farkındalığı sağlamak).
    """

    assert canonicalize_region_label("UPenn", "NC") == "NC"
    assert canonicalize_region_label("UPenn-GBM", "ED") == "ED"
    assert canonicalize_region_label("UPenn", "WT_derived") == "WT"
    assert canonicalize_region_label("UPenn", "TC_derived") == "TC"

    assert canonicalize_region_label("LUMIERE", "Necrosis") == "NC"
    assert canonicalize_region_label("LUMIERE", "Edema") == "ED"
    assert canonicalize_region_label("LUMIERE", "Contrast-enhancing") == "ET"
    assert canonicalize_region_label("LUMIERE", "WT_derived") == "WT"
    assert canonicalize_region_label("LUMIERE", "TC_derived") == "TC"

    assert canonicalize_region_label("TCGA-GBM", "WT") == "WT"
    assert canonicalize_region_label("TCGA-GBM", "TC") == "TC"


def test_canonicalize_region_label_rejects_unrecognized_region() -> None:
    with pytest.raises(ValueError, match="tanımıyor"):
        canonicalize_region_label("UPenn", "ET_but_typo")

    with pytest.raises(ValueError, match="tanımıyor"):
        # TCGA'da NC/ED/ET hiç üretilmez -- kilitli karar (bkz. CLAUDE.md).
        canonicalize_region_label("TCGA", "NC")


def test_canonicalize_region_label_rejects_unrecognized_source() -> None:
    with pytest.raises(ValueError):
        canonicalize_region_label("MysteryCohort", "WT")


def _make_region_long_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_pivot_radiomics_long_to_wide_wt_only_happy_path() -> None:
    long_frame = _make_region_long_frame(
        [
            {
                "patient_id": "p1",
                "source": "UPenn",
                "tumor_region": "WT_derived",
                "shape_features": {"original_shape_Sphericity": 0.5},
                "first_order_features": {"original_firstorder_Entropy": 1.2},
                "texture_features": {"original_glcm_Contrast": 3.3},
            },
            {
                "patient_id": "p2",
                "source": "TCGA-GBM",
                "tumor_region": "WT",
                "shape_features": {"original_shape_Sphericity": 0.6},
                "first_order_features": {"original_firstorder_Entropy": 1.4},
                "texture_features": {"original_glcm_Contrast": 3.9},
            },
        ]
    )

    wide, report = pivot_radiomics_long_to_wide(long_frame, ["WT"])

    assert report.n_candidate_patients == 2
    assert report.n_output_patients == 2
    assert report.dropped_patients_missing_region == {"WT": 0}
    assert list(wide.index) == ["p1", "p2"]
    assert set(wide.columns) == {
        "WT__original_shape_Sphericity",
        "WT__original_firstorder_Entropy",
        "WT__original_glcm_Contrast",
    }
    assert wide.loc["p1", "WT__original_shape_Sphericity"] == 0.5
    assert wide.loc["p2", "WT__original_shape_Sphericity"] == 0.6


def test_pivot_radiomics_long_to_wide_wt_tc_prefixes_avoid_collision() -> None:
    long_frame = _make_region_long_frame(
        [
            {
                "patient_id": "p1",
                "source": "TCGA-GBM",
                "tumor_region": "WT",
                "shape_features": {"original_shape_Sphericity": 0.5},
                "first_order_features": {},
                "texture_features": {},
            },
            {
                "patient_id": "p1",
                "source": "TCGA-GBM",
                "tumor_region": "TC",
                "shape_features": {"original_shape_Sphericity": 0.7},
                "first_order_features": {},
                "texture_features": {},
            },
        ]
    )

    wide, report = pivot_radiomics_long_to_wide(long_frame, ["WT", "TC"])

    assert report.n_output_patients == 1
    assert wide.loc["p1", "WT__original_shape_Sphericity"] == 0.5
    assert wide.loc["p1", "TC__original_shape_Sphericity"] == 0.7


def test_pivot_radiomics_long_to_wide_drops_patient_missing_region_by_default() -> None:
    long_frame = _make_region_long_frame(
        [
            {
                "patient_id": "p1",
                "source": "UPenn",
                "tumor_region": "NC",
                "shape_features": {"original_shape_Sphericity": 0.5},
                "first_order_features": {},
                "texture_features": {},
            },
            # p1'in ED/ET satırı YOK -- eksik bölge.
            {
                "patient_id": "p2",
                "source": "UPenn",
                "tumor_region": "NC",
                "shape_features": {"original_shape_Sphericity": 0.4},
                "first_order_features": {},
                "texture_features": {},
            },
            {
                "patient_id": "p2",
                "source": "UPenn",
                "tumor_region": "ED",
                "shape_features": {"original_shape_Sphericity": 0.3},
                "first_order_features": {},
                "texture_features": {},
            },
            {
                "patient_id": "p2",
                "source": "UPenn",
                "tumor_region": "ET",
                "shape_features": {"original_shape_Sphericity": 0.2},
                "first_order_features": {},
                "texture_features": {},
            },
        ]
    )

    wide, report = pivot_radiomics_long_to_wide(long_frame, ["NC", "ED", "ET"])

    assert report.n_candidate_patients == 2
    assert report.n_output_patients == 1
    assert list(wide.index) == ["p2"]
    # p1 ED VE ET'den eksik -- ikisi de sayılmalı (sessiz NaN doldurma YOK).
    assert report.dropped_patients_missing_region == {"NC": 0, "ED": 1, "ET": 1}


def test_pivot_radiomics_long_to_wide_raise_mode_stops_on_first_missing_patient() -> None:
    long_frame = _make_region_long_frame(
        [
            {
                "patient_id": "p1",
                "source": "UPenn",
                "tumor_region": "NC",
                "shape_features": {},
                "first_order_features": {},
                "texture_features": {},
            },
        ]
    )

    with pytest.raises(RegionPivotError, match="eksik"):
        pivot_radiomics_long_to_wide(
            long_frame, ["NC", "ED", "ET"], on_missing_region="raise"
        )


def test_pivot_radiomics_long_to_wide_raises_on_duplicate_patient_region() -> None:
    long_frame = _make_region_long_frame(
        [
            {
                "patient_id": "p1",
                "source": "UPenn",
                "tumor_region": "NC",
                "shape_features": {"a": 1.0},
                "first_order_features": {},
                "texture_features": {},
            },
            {
                # Aynı hasta+bölge için İKİNCİ satır (örn. iki farklı
                # segmentation_tool) -- pivot ÇAKIŞMASI.
                "patient_id": "p1",
                "source": "UPenn",
                "tumor_region": "NC",
                "shape_features": {"a": 2.0},
                "first_order_features": {},
                "texture_features": {},
            },
        ]
    )

    with pytest.raises(RegionPivotError, match="ÇAKIŞMASI"):
        pivot_radiomics_long_to_wide(long_frame, ["NC"])


def test_pivot_radiomics_long_to_wide_rejects_unrecognized_region_even_if_not_requested() -> None:
    """Fail-loud ilkesi: eşleme tablosunun tanımadığı bir (kaynak,
    ham-bölge) çifti, o satır sonradan filtrelenecek olsa bile
    SESSİZCE geçilmemeli."""

    long_frame = _make_region_long_frame(
        [
            {
                "patient_id": "p1",
                "source": "UPenn",
                "tumor_region": "SomeUnknownRegion",
                "shape_features": {},
                "first_order_features": {},
                "texture_features": {},
            },
        ]
    )

    with pytest.raises(ValueError, match="tanımıyor"):
        pivot_radiomics_long_to_wide(long_frame, ["NC"])


def test_pivot_radiomics_long_to_wide_rejects_empty_regions() -> None:
    with pytest.raises(ValueError, match="boş olamaz"):
        pivot_radiomics_long_to_wide(pd.DataFrame(), [])


# =====================================================================
# C5 -- assert_combat_identity_for_training_pool() doğrudan testleri
# =====================================================================


def test_assert_combat_identity_accepts_upenn_only_pool() -> None:
    patients_frame = pd.DataFrame({"source": ["UPenn", "UPenn-GBM", " upenn-gbm "]})
    # Hiçbir exception fırlatılmamalı -- havuz tamamen referans batch.
    assert_combat_identity_for_training_pool(patients_frame)


def test_assert_combat_identity_rejects_non_reference_source() -> None:
    patients_frame = pd.DataFrame({"source": ["UPenn", "LUMIERE"]})
    with pytest.raises(ComBatIdentityAssumptionError, match=r"\['LUMIERE'\]"):
        assert_combat_identity_for_training_pool(patients_frame)


def test_assert_combat_identity_respects_custom_reference_batch() -> None:
    """`reference_batch` parametresi ÇAĞIRANIN belirttiği ComBat
    referansıyla TUTARLI olmalı -- varsayılan "UPenn" değilse de guard
    doğru kaynağa göre çalışmalı."""

    patients_frame = pd.DataFrame({"source": ["LUMIERE", "LUMIERE"]})
    # reference_batch="LUMIERE" olsaydı bu havuz özdeşlik varsayımını
    # ihlal ETMEZDİ.
    assert_combat_identity_for_training_pool(patients_frame, reference_batch="LUMIERE")

    # ama varsayılan (UPenn) referansla aynı havuz REDDEDİLMELİ.
    with pytest.raises(ComBatIdentityAssumptionError):
        assert_combat_identity_for_training_pool(patients_frame)


def test_assert_combat_identity_rejects_unknown_source_loudly() -> None:
    patients_frame = pd.DataFrame({"source": ["UPenn", "MysteryCohort"]})
    with pytest.raises(ComBatIdentityAssumptionError, match="tanımadığı"):
        assert_combat_identity_for_training_pool(patients_frame)


# =====================================================================
# C2 + C3 -- run_nested_cv() testleri (sentetik veri)
# =====================================================================


def test_run_nested_cv_produces_one_row_per_outer_fold_with_reasonable_c_index() -> None:
    frame, feature_columns = _make_synthetic_survival_frame(n=200, n_features=8, seed=21)

    cv_results = run_nested_cv(
        frame,
        feature_columns,
        outer_splits=3,
        inner_splits=3,
        l1_ratio_grid=(0.5, 1.0),
        penalizer_grid=(0.1, 0.5),
        stability_selection=True,
        n_bootstrap_stability=30,
        seed=7,
    )

    assert len(cv_results) == 3
    assert (cv_results["n_test"] > 0).all()
    assert (cv_results["c_index"] >= 0.0).all()
    assert (cv_results["c_index"] <= 1.0).all()
    # her fold en az bir özellik seçmiş olmalı (boş model YASAK).
    assert (cv_results["n_selected_features_final"] > 0).all()
    # her fold'un kendi seçilen-özellik listesi raporlanmalı (görev
    # talimatı: "fold başına seçilen özellik listesi raporlanmalı").
    for selected in cv_results["selected_features"]:
        assert isinstance(selected, list)
        assert len(selected) > 0
    # taranan hiperparametreler gridin İÇİNDEN seçilmiş olmalı.
    assert set(cv_results["best_l1_ratio"]).issubset({0.5, 1.0})
    assert set(cv_results["best_penalizer"]).issubset({0.1, 0.5})


def test_run_nested_cv_without_stability_selection_still_uses_elastic_net_only() -> None:
    frame, feature_columns = _make_synthetic_survival_frame(n=150, n_features=6, seed=22)

    cv_results = run_nested_cv(
        frame,
        feature_columns,
        outer_splits=3,
        inner_splits=3,
        l1_ratio_grid=(1.0,),
        penalizer_grid=(0.1, 0.5),
        stability_selection=False,
        seed=11,
    )

    assert len(cv_results) == 3
    # stabilite kapalıyken final==elastic-net seçimiyle AYNI sayıda olmalı.
    assert (
        cv_results["n_selected_features_final"] == cv_results["n_selected_features_elastic_net"]
    ).all()


def test_run_nested_cv_rejects_bad_on_missing_region_type_via_pivot_is_unrelated() -> None:
    """Regresyon-koruması: `run_nested_cv()`'in kendisi region/pivot
    kavramından bağımsız (bölge-agnostik) çalışmaya devam etmeli --
    `feature_columns` STABLE_FEATURES_ICC60 gibi önekli/önereksiz HERHANGİ
    bir liste olabilir, fonksiyon bunu doğrulamaz (bu, çağıranın
    `pivot_radiomics_long_to_wide()` ile önceden ürettiği kolonlardır)."""

    frame, feature_columns = _make_synthetic_survival_frame(n=120, n_features=5, seed=23)
    renamed_columns = [f"WT__{c}" for c in feature_columns]
    frame = frame.rename(columns=dict(zip(feature_columns, renamed_columns)))

    cv_results = run_nested_cv(
        frame,
        renamed_columns,
        outer_splits=3,
        inner_splits=3,
        l1_ratio_grid=(0.5, 1.0),
        penalizer_grid=(0.05, 0.1),
        stability_selection=False,
        seed=5,
    )
    assert len(cv_results) == 3


# =====================================================================
# 2026-08-15 -- klinik kovaryat desteği (C7): standardize_columns_fold_safe,
# _build_penalizer_argument, extra_column_penalizer, passthrough_columns,
# boş feature_columns + extra_columns (clinical_base senaryosu)
# =====================================================================


def test_standardize_columns_fold_safe_uses_train_stats_only() -> None:
    train_frame = pd.DataFrame({"clinical_age": [10.0, 20.0, 30.0, 40.0]})
    test_frame = pd.DataFrame({"clinical_age": [100.0, 200.0]})

    train_out, test_out = standardize_columns_fold_safe(train_frame, test_frame, ["clinical_age"])

    train_mean = train_frame["clinical_age"].mean()
    train_std = train_frame["clinical_age"].std(ddof=0)
    # train_out kendi ortalaması/std'siyle standardize edilmiş olmalı.
    assert train_out["clinical_age"].mean() == pytest.approx(0.0, abs=1e-9)
    # test_out AYNI (train'den fit edilmiş) mean/std ile dönüştürülmeli --
    # test'in KENDİ istatistikleri (100/200) KULLANILMAMALI (sızıntı yok).
    expected_test = (test_frame["clinical_age"] - train_mean) / train_std
    pd.testing.assert_series_equal(test_out["clinical_age"], expected_test, check_names=False)


def test_standardize_columns_fold_safe_noop_when_columns_empty() -> None:
    train_frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    test_frame = pd.DataFrame({"a": [4.0, 5.0]})

    train_out, test_out = standardize_columns_fold_safe(train_frame, test_frame, [])

    pd.testing.assert_frame_equal(train_out, train_frame)
    pd.testing.assert_frame_equal(test_out, test_frame)


def test_standardize_columns_fold_safe_zero_std_column_does_not_raise() -> None:
    train_frame = pd.DataFrame({"const": [5.0, 5.0, 5.0]})

    train_out, test_out = standardize_columns_fold_safe(train_frame, None, ["const"])

    assert test_out is None
    # std sifirsa 1.0'a yuvarlanir -- (5-5)/1 = 0, NaN/inf URETILMEZ.
    assert (train_out["const"] == 0.0).all()
    assert train_out["const"].isna().sum() == 0


def test_build_penalizer_argument_returns_scalar_when_none() -> None:
    result = _build_penalizer_argument(
        ["f1", "f2"], ["extra1"], penalizer=0.3, extra_column_penalizer=None
    )
    assert result == 0.3
    assert not isinstance(result, np.ndarray)


def test_build_penalizer_argument_array_order_matches_lifelines_params_ordering() -> None:
    """`_build_penalizer_argument()`'ın döndürdüğü dizinin
    `CoxPHFitter`'a verilen DataFrame kolon SIRASIYLA hizalandığını
    ampirik olarak doğrular (train_cox_week3.py'nin bu fonksiyona
    dayandığı varsayım -- sırayla feature_columns sonra extra_columns)."""

    from lifelines import CoxPHFitter

    rng = np.random.default_rng(3)
    n = 200
    frame = pd.DataFrame(
        {
            "feat_a": rng.normal(size=n),
            "feat_b": rng.normal(size=n),
            "extra_x": rng.normal(size=n),
        }
    )
    beta_a, beta_b, beta_x = 1.2, -0.8, 0.6
    linear_predictor = beta_a * frame["feat_a"] + beta_b * frame["feat_b"] + beta_x * frame["extra_x"]
    baseline_hazard = 0.02
    u = rng.uniform(size=n)
    event_time = -np.log(u) / (baseline_hazard * np.exp(linear_predictor))
    censoring_time = rng.uniform(low=1.0, high=np.percentile(event_time, 90), size=n)
    frame["duration"] = np.minimum(event_time, censoring_time)
    frame["event"] = (event_time <= censoring_time).astype(int)

    feature_columns = ["feat_a", "feat_b"]
    extra_columns = ["extra_x"]
    penalizer_argument = _build_penalizer_argument(
        feature_columns, extra_columns, penalizer=0.5, extra_column_penalizer=0.0
    )
    assert isinstance(penalizer_argument, np.ndarray)
    np.testing.assert_array_equal(penalizer_argument, np.array([0.5, 0.5, 0.0]))

    model_columns = feature_columns + extra_columns
    cox = CoxPHFitter(penalizer=penalizer_argument, l1_ratio=1.0)
    cox.fit(frame[model_columns + ["duration", "event"]], duration_col="duration", event_col="event")
    # lifelines'in urettigi katsayi sirasi TAM OLARAK model_columns
    # sirasiyla ESLESMELI -- bu, _build_penalizer_argument()'in
    # dokumentasyonundaki hizalama varsayiminin KANITI.
    assert list(cox.params_.index) == model_columns


def test_select_features_lasso_extra_column_penalizer_zero_avoids_shrinkage() -> None:
    """`extra_column_penalizer=0.0` (klinik kovaryat -- "zorunlu, zayıf/
    sıfır penalize") ile `extra_columns`'ın katsayısı, AYNI (büyük)
    `penalizer`'la cezalandırılan bilgilendirici bir `feature_column`'a
    kıyasla ÇOK DAHA AZ küçülmeli (shrinkage farkı ölçülebilir)."""

    frame, feature_columns = _make_synthetic_survival_frame(n=400, n_features=1, seed=17)
    rng = np.random.default_rng(19)
    # extra_col da AYNI bilgilendiricilikte (feat_0 ile benzer buyuklukte
    # sinyal) ama extra_columns'a konuyor -- boylece iki farkli penalizasyon
    # rejiminin AYNI sinyal buyuklugundeki etkisini karsilastirabiliyoruz.
    frame["extra_col"] = frame["feat_0"] + rng.normal(scale=0.05, size=len(frame))

    strong_penalizer = 5.0  # feat_0'ı ağır şekilde küçültecek büyüklükte
    selected, cox, coefficients = select_features_lasso(
        frame,
        feature_columns,
        extra_columns=["extra_col"],
        penalizer=strong_penalizer,
        l1_ratio=1.0,
        extra_column_penalizer=0.0,
    )

    assert abs(coefficients["extra_col"]) > abs(coefficients["feat_0"])


def test_run_nested_cv_allows_empty_feature_columns_when_extra_columns_present() -> None:
    """`clinical_base` senaryosu (2026-08-15): `feature_columns=[]`
    (radyomik ADAY havuzu yok) ama `extra_columns` (klinik kovaryat)
    doluysa `run_nested_cv()` HATA VERMEMELİ -- boş final_features bir
    HATA değil, beklenen davranış (ne seçilecek radyomik var)."""

    frame, _ = _make_synthetic_survival_frame(n=150, n_features=3, seed=31)
    rng = np.random.default_rng(32)
    frame["clinical_extra"] = frame["feat_0"] + rng.normal(scale=0.1, size=len(frame))

    cv_results = run_nested_cv(
        frame,
        [],  # feature_columns BOS
        extra_columns=["clinical_extra"],
        outer_splits=3,
        inner_splits=2,
        l1_ratio_grid=(1.0,),
        penalizer_grid=(0.1,),
        stability_selection=True,
        n_bootstrap_stability=10,
        extra_column_penalizer=0.0,
        seed=41,
    )

    assert len(cv_results) == 3
    assert (cv_results["n_selected_features_final"] == 0).all()  # radyomik seçilecek yok
    assert (cv_results["c_index"] >= 0.0).all()
    assert (cv_results["c_index"] <= 1.0).all()


def test_run_nested_cv_still_rejects_totally_empty_model() -> None:
    """Regresyon koruması: `feature_columns=[]` VE `extra_columns=[]`
    (klinik kovaryat da yoksa) hâlâ HATA vermeli -- gevşetme SADECE
    `extra_columns` doluyken geçerli."""

    frame, _ = _make_synthetic_survival_frame(n=80, n_features=2, seed=33)

    with pytest.raises(RuntimeError, match="hiçbir özellik bırakmadı"):
        run_nested_cv(
            frame,
            [],
            extra_columns=[],
            outer_splits=3,
            inner_splits=2,
            l1_ratio_grid=(1.0,),
            penalizer_grid=(0.5,),
            stability_selection=True,
            n_bootstrap_stability=10,
            seed=1,
        )


def test_run_nested_cv_clinical_standardize_columns_uses_fold_safe_transform() -> None:
    """`clinical_standardize_columns` verildiğinde her dış fold'un
    train alt-kümesinden mean/std fit edilip test'e uygulandığını
    (`standardize_columns_fold_safe()`'in çağrıldığını) dolaylı olarak
    doğrular -- kaba büyüklükte (ham) bir "yaş" kolonu eklenip modelin
    hatasız çalıştığı + fold'lar arası ortalamanın (yaklaşık) 0'a
    yakınsadığı gözlemlenir (fold-safe standardizasyonun beklenen yan
    etkisi)."""

    frame, feature_columns = _make_synthetic_survival_frame(n=200, n_features=4, seed=51)
    rng = np.random.default_rng(52)
    frame["clinical_age"] = rng.uniform(low=20.0, high=85.0, size=len(frame))

    cv_results = run_nested_cv(
        frame,
        feature_columns,
        extra_columns=["clinical_age"],
        outer_splits=3,
        inner_splits=2,
        l1_ratio_grid=(1.0,),
        penalizer_grid=(0.3,),
        stability_selection=True,
        n_bootstrap_stability=10,
        extra_column_penalizer=0.0,
        clinical_standardize_columns=["clinical_age"],
        seed=61,
    )

    assert len(cv_results) == 3
    assert (cv_results["c_index"] >= 0.0).all()
    assert (cv_results["c_index"] <= 1.0).all()


def test_assemble_training_frame_passthrough_columns_carried_and_missing_dropped() -> None:
    feature_frame = pd.DataFrame(
        {"feat_a": [1.0, 2.0, 3.0, 4.0]}, index=["p1", "p2", "p3", "p4"]
    )
    patients_frame = pd.DataFrame(
        {
            "source": ["UPenn", "UPenn", "UPenn", "UPenn"],
            "survival_days": [100, 200, 300, 400],
            "vital_status": ["DECEASED", "ALIVE", "DECEASED", "DECEASED"],
            "clinical_age": [45.0, 60.0, np.nan, 70.0],  # p3 eksik
        },
        index=["p1", "p2", "p3", "p4"],
    )

    frame, report = _assemble_training_frame(
        feature_frame, patients_frame, passthrough_columns=["clinical_age"]
    )

    assert report.dropped_missing_passthrough == 1
    assert report.n_output_rows == 3
    assert "clinical_age" in frame.columns
    assert set(frame.index) == {"p1", "p2", "p4"}


def test_build_training_frame_forwards_passthrough_columns() -> None:
    feature_frame = pd.DataFrame(
        {"feat_a": [1.0, 2.0, 3.0]}, index=["p1", "p2", "p3"]
    )
    patients_frame = pd.DataFrame(
        {
            "source": ["UPenn", "UPenn", "UPenn"],
            "survival_days": [100, 200, 300],
            "vital_status": ["DECEASED", "ALIVE", "DECEASED"],
            "clinical_gender_male": [1.0, 0.0, 1.0],
        },
        index=["p1", "p2", "p3"],
    )

    frame, report = build_training_frame(
        feature_frame,
        patients_frame,
        check_combat_identity=False,
        passthrough_columns=["clinical_gender_male"],
    )

    assert "clinical_gender_male" in frame.columns
    assert report.n_output_rows == 3


# =====================================================================
# A2 (2026-09-13) -- run_nested_cv() SIZINTI PERTURBASYON TESTLERI
#
# NEDEN: 2026-09-13 review-gate kalem 4 olcumu: `grep -c perturb` ->
# test_cox_model.py = 0, test_xgboost_model.py = 48. Yani Cox tarafinda
# nested-CV'nin sizintisizligini KANITLAYAN tek bir test bile yoktu;
# `cox_model.py`'nin dis-fold dongusunde `train_frame` yerine `frame`
# yazan bir regresyon HICBIR testi kirmizi yapmiyordu (ölçüldü).
# Bu iki test, `test_xgboost_model.py:1152`'deki (aligned pipeline
# negatif kontrolu) desenin Cox esleniğidir.
#
# YONTEM (ve NEDEN bu yonde): bir dis fold'un TEST alt-kumesindeki bir
# hastanin `survival_days`'i bozulur. DOGRU kodda o hastanin verisi
# AYNI fold'un egitim tarafinda (ic-CV hiperparametre taramasi,
# elastic-net secimi, bootstrap stabilite secimi) HIC KULLANILMAZ --
# dolayisiyla O FOLD'un `selected_features` / `best_penalizer` /
# `best_l1_ratio` degerleri BIT-BIREBIR AYNI kalmalidir.
# ⚠️ DIGER fold'lar icin ayni sey BEKLENMEZ ve ASSERT EDILMEZ: o
# fold'larda ayni hasta EGITIM kumesindedir, secimin degismesi
# MESRUDUR (sizinti degil). Bu yuzden "o fold'un disindaki fold'lar
# ayni kalmali" diye bir assert YAZILMADI -- yazilsaydi dogru kodda
# bile kirmizi olurdu (olculdu: fold 1 ve 2'nin secimi gercekten
# degisiyor). Bunun yerine o fold'lardaki degisim PERTURBASYONUN
# ETKILI OLDUGUNUN kaniti olarak kullanilir (asagidaki "potens"
# assert'i) -- aksi halde test, perturbasyon hicbir seyi
# degistirmedigi icin bos yere yesil kalabilirdi.
# =====================================================================


def _outer_fold_splits_for_test(
    frame: pd.DataFrame, *, outer_splits: int, seed: int, event_col: str = "event"
) -> list[tuple[list[int], list[int]]]:
    """`run_nested_cv()`'nin dis-fold bolunmesini BIREBIR AYNI cagriyla
    (StratifiedKFold(shuffle=True, random_state=seed), `event_col`
    uzerinde) yeniden uretir -- test, hangi hastanin hangi fold'un TEST
    tarafinda oldugunu bilmek zorunda (run_nested_cv hasta kimliklerini
    dondurmuyor)."""

    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    return [
        (list(train_idx), list(test_idx))
        for train_idx, test_idx in splitter.split(frame, frame[event_col])
    ]


def _make_multi_signal_survival_frame(
    n: int, *, n_features: int, seed: int
) -> tuple[pd.DataFrame, list[str]]:
    """Perturbasyon testleri icin sentetik Cox verisi -- `_make_synthetic_
    survival_frame()`'den TEK farki: bilgilendirici ozellik SAYISI 3
    (1,5 / 0,8 / 0,4 katsayili). Tek bilgilendirici ozellikte secim
    neredeyse her zaman ayni kaliyor ve testin "potens" kontrolu
    (perturbasyonun gercekten secimi degistirebildigi) kurulamiyor."""

    rng = np.random.default_rng(seed)
    feature_columns = [f"feat_{i}" for i in range(n_features)]
    features = rng.normal(size=(n, n_features))
    linear_predictor = (
        1.5 * features[:, 0] + 0.8 * features[:, 1] + 0.4 * features[:, 2]
    )
    u = rng.uniform(size=n)
    event_time = -np.log(u) / (0.02 * np.exp(linear_predictor))
    censoring_time = rng.uniform(low=1.0, high=np.percentile(event_time, 90), size=n)
    duration = np.minimum(event_time, censoring_time)
    event = (event_time <= censoring_time).astype(int)

    frame = pd.DataFrame(features, columns=feature_columns)
    frame["survival_days"] = duration
    frame["event"] = event
    return frame, feature_columns


def _run_own_test_fold_perturbation_check(*, stability_selection: bool) -> None:
    """Iki testin ORTAK govdesi (stabilite secimi ACIK/KAPALI).

    ACIK  -> `cox_model.py`'nin hem elastic-net (satir ~1845) hem
             bootstrap stabilite (satir ~1857) cagrisini korur.
    KAPALI-> yalniz elastic-net dalini korur (fallback/stabilite yolu
             devre disi oldugunda regresyon yine yakalanir)."""

    outer_splits = 3
    seed = 5
    kwargs = dict(
        outer_splits=outer_splits,
        inner_splits=3,
        l1_ratio_grid=(1.0,),
        # 2 elemanli grid BILINCLI: ic-CV hiperparametre taramasi da bir
        # sizinti yuzeyidir (cox_model.py:1819-1830) -- tek elemanli
        # grid'de `best_penalizer` assert'i bos bir kontrol olurdu.
        penalizer_grid=(0.05, 0.3),
        stability_selection=stability_selection,
        n_bootstrap_stability=20,
        stability_frequency_threshold=0.6,
        seed=seed,
    )

    frame, feature_columns = _make_multi_signal_survival_frame(140, n_features=8, seed=31)
    splits = _outer_fold_splits_for_test(frame, outer_splits=outer_splits, seed=seed)
    target_fold = 0
    target_position = splits[target_fold][1][0]
    target_label = frame.index[target_position]

    # SADECE `survival_days` bozulur (x50 + 5000). `event` DEGISTIRILMEZ --
    # dis-fold bolunmesi `event` uzerinde stratified oldugu icin event'i
    # degistirmek fold YAPISINI degistirir ve karsilastirmayi anlamsiz
    # kilardi (test_xgboost_model.py:1185-1191'deki AYNI gerekce).
    perturbed = frame.copy()
    perturbed.loc[target_label, "survival_days"] = (
        float(perturbed.loc[target_label, "survival_days"]) * 50.0 + 5000.0
    )

    # (1) FOLD YAPISI DEGISMEDI -- bolunme yeniden uretilip karsilastirilir.
    perturbed_splits = _outer_fold_splits_for_test(
        perturbed, outer_splits=outer_splits, seed=seed
    )
    assert perturbed_splits == splits, (
        "Perturbasyon dis-fold bolunmesini DEGISTIRDI -- karsilastirma "
        "anlamsiz olurdu (event kolonuna dokunulmamali)."
    )
    assert target_position in splits[target_fold][1]
    assert target_position not in splits[target_fold][0]

    baseline = run_nested_cv(frame, feature_columns, **kwargs)
    perturbed_result = run_nested_cv(perturbed, feature_columns, **kwargs)

    base_row = baseline.loc[baseline["fold"] == target_fold].iloc[0]
    pert_row = perturbed_result.loc[perturbed_result["fold"] == target_fold].iloc[0]

    # (2) fold buyuklukleri/olay sayilari ayni (run_nested_cv'nin KENDI
    # raporladigi sayilar uzerinden ikinci bir dogrulama).
    for column in ("n_train", "n_test", "n_events_train", "n_events_test"):
        assert (baseline[column].tolist() == perturbed_result[column].tolist()), (
            f"Fold yapisi degisti ({column}) -- perturbasyon fold "
            "bolunmesini etkilemis olmamali."
        )

    # (3) ASIL SIZINTI ASSERT'I: perturbe edilen hastanin TEST tarafinda
    # oldugu fold'un EGITIMDEN TUREYEN her ciktisi BIT-BIREBIR ayni.
    assert list(pert_row["selected_features"]) == list(base_row["selected_features"]), (
        "SIZINTI REGRESYONU (cox_model.py dis-fold dongusu): bir dis "
        "fold'un TEST alt-kumesindeki hastanin `survival_days`'i "
        "bozulunca AYNI fold'un secilen ozellik listesi DEGISTI -- "
        "ozellik secimi (elastic-net ve/veya bootstrap stabilite) "
        "`train_frame` yerine TUM cerceveyi goruyor olabilir.\n"
        f"  baseline={list(base_row['selected_features'])}\n"
        f"  perturbe ={list(pert_row['selected_features'])}"
    )
    assert float(pert_row["best_penalizer"]) == float(base_row["best_penalizer"]), (
        "SIZINTI REGRESYONU: ic-CV hiperparametre taramasi dis-test "
        "hastasinin verisini gormus olabilir (best_penalizer degisti)."
    )
    assert float(pert_row["best_l1_ratio"]) == float(base_row["best_l1_ratio"]), (
        "SIZINTI REGRESYONU: ic-CV hiperparametre taramasi dis-test "
        "hastasinin verisini gormus olabilir (best_l1_ratio degisti)."
    )
    assert int(pert_row["n_selected_features_final"]) == int(
        base_row["n_selected_features_final"]
    )
    assert int(pert_row["n_selected_features_elastic_net"]) == int(
        base_row["n_selected_features_elastic_net"]
    )

    # (4) POTENS KONTROLU (testin bos yere yesil kalmadiginin kaniti):
    # perturbe edilen hasta DIGER fold'larin EGITIM kumesindedir, orada
    # secimin/skorun degismesi MESRUDUR -- ve gercekten degisiyor. Eger
    # hicbir sey degismeseydi, (3)'teki assert perturbasyonun etkisiz
    # olmasindan oturu de gecebilirdi.
    other_folds = [f for f in baseline["fold"].tolist() if f != target_fold]
    changed_somewhere = False
    for fold_index in other_folds:
        b = baseline.loc[baseline["fold"] == fold_index].iloc[0]
        p = perturbed_result.loc[perturbed_result["fold"] == fold_index].iloc[0]
        if list(b["selected_features"]) != list(p["selected_features"]):
            changed_somewhere = True
        if not np.isclose(float(b["c_index"]), float(p["c_index"]), atol=1e-12):
            changed_somewhere = True
    assert changed_somewhere, (
        "POTENS KONTROLU BASARISIZ: perturbasyon, hastanin EGITIM "
        "kumesinde oldugu fold'larda bile HICBIR seyi degistirmedi -- "
        "bu durumda yukaridaki sizinti assert'i bos bir kontroldur, "
        "perturbasyon buyutulmeli/veri degistirilmelidir."
    )


def test_run_nested_cv_selection_unaffected_by_own_test_fold_perturbation() -> None:
    """SIZINTI NEGATIF KONTROLU (stabilite secimi ACIK -- protokolun
    birincil yolu): bir dis-test hastasinin `survival_days`'ini bozmak,
    AYNI fold'un secilen ozellik listesini/hiperparametrelerini
    DEGISTIRMEMELIDIR.

    KIRMIZI/YESIL KANITI (2026-09-13, modeling-agent-H): `pipeline/
    cox_model.py`'de dis-fold dongusundeki `select_features_lasso(
    train_frame, ...)` ve `_bootstrap_stability_selection(train_frame,
    ...)` cagrilari GECICI olarak `frame` yapildiginda bu test KIRMIZI
    oldu; geri alininca YESIL. Detay gunluk kaydinda."""

    _run_own_test_fold_perturbation_check(stability_selection=True)


def test_run_nested_cv_elastic_net_only_selection_unaffected_by_own_test_fold_perturbation() -> None:
    """AYNI negatif kontrol, `stability_selection=False` ile -- stabilite
    dali devre disiyken elastic-net secim cagrisinin (cox_model.py
    ~1845) TEK BASINA fold-guvenli kaldigini kanitlar."""

    _run_own_test_fold_perturbation_check(stability_selection=False)


# =====================================================================
# A2-X (2026-09-14, modeling-agent-S3) -- X-TARAFLI PERTURBASYON:
# FOLD-GUVENLI STANDARDIZASYON (`standardize_columns_fold_safe`)
#
# NEDEN: yukaridaki A2 blogu YALNIZ `survival_days`'i (Y) bozuyor ->
# X-tarafli her sizinti kanalina yapisal olarak KOR. Somut bosluk:
# `test_run_nested_cv_clinical_standardize_columns_uses_fold_safe_
# transform()` (bu dosyada) bir SMOKE testidir -- assert'leri yalnizca
# `len(cv_results)==3` ve `0<=c_index<=1`; docstring'i "fold'lar arasi
# ortalamanin 0'a yakinsadigi GOZLEMLENIR" diyor ama BUNU ASSERT
# ETMIYOR. Dolayisiyla istatistik TUM cerceveden (fold disindan) fit
# edilse bile YESIL kalirdi. Uretimde (v3b) o liste 54 radyomik + yas
# iceriyor (`tools/train_cox_week3.py:4248-4255`, `standardize_radiomics=
# True`), yani v3b'nin EN BUYUK X-tarafli fold-guvenlik yuzeyi test
# edilmiyordu. Emsal: `tests/test_xgboost_model.py::test_reduce_
# collinearity_pool_bit_identical_when_perturbed_patient_outside_that_
# folds_train_set()` (spy + X-perturbasyon + bit-birebir karsilastirma) --
# desen ORADAN alindi, yeni desen icat EDILMEDI.
#
# OLCULEN YAPISAL KISIT (2026-09-14, modeling-agent-S3 olcumu --
# modeling-agent-R1'in "olcemedim" dedigi noktanin ACIKLAMASI):
# `run_nested_cv()`'nin DIS CIKTILARI (`selected_features`,
# `best_penalizer`, `best_l1_ratio`, `c_index`) uzerinden bu sizintiyi
# yakalamak MATEMATIKSEL OLARAK IMKANSIZDIR:
#   (1) `standardize_columns_fold_safe()` AYNI afin donusumu HEM train
#       HEM test cercevesine uygular (kaynak: `pipeline/cox_model.py`
#       satir 1506-1509 -- `means`/`stds` tek bir cift, ikisine de).
#   (2) lifelines `CoxPHFitter` (0.30.0) kovaryatlari fit'ten ONCE KENDI
#       ICINDE yeniden standardize eder ve beta'lari geri olcekler
#       (`lifelines/fitters/coxph_fitter.py` satir 1245 `self._norm_std =
#       X.std(0)`, satir 1249 `utils.normalize(...)`, satir 1399
#       `params_ = beta_ / self._norm_std`). Penalizer bu IC normalize
#       uzayda uygulanir -> disaridan uygulanan herhangi bir afin olcek
#       SOGURULUR.
#   (1)+(2) => disaridan uygulanan afin donusum, secim/hiperparametre/
#   concordance ciktilarinda OZDESLIK dogurur. AMPIRIK DOGRULAMA asagida
#   `test_run_nested_cv_outputs_are_invariant_to_clinical_standardize_
#   columns_documented_identity()` ile KAYIT ALTINA ALINDI: standardizasyon
#   ACIK vs KAPALI -> 3 fold'da da AYNI secim, max |delta c_index| = 0.0.
#   Bu yuzden A2'nin "bozup dis ciktiyi karsilastir" kalibi burada
#   CALISMAZ (R1'in 140x8 kurgusunda "secim DEGISMEDI" gozlemi ZAYIF
#   KURGU degil, bu OZDESLIGIN sonucudur).
#
# BU YUZDEN KORUMA, CAGRI NOKTASINA TASINDI (XGBoost'taki `_spy`
# deseninin aynisi): sizinti, fonksiyonun DONDURDUGU train cercevesi
# uzerinde dogrudan olculur. Ayirt edicilik iddiasi VARSAYILMADI,
# asagidaki `..._potency_leaky_implementation_is_detected()` testiyle
# KANITLANDI (sizdiran implementasyon enjekte edilince ayni assert'ler
# KIRMIZI oluyor).
# =====================================================================


_STD_PERTURBATION_SCALE = 500.0
_STD_PERTURBATION_SHIFT = 10_000.0


def _leaky_standardize_columns_fold_safe(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame | None,
    columns: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """SIZDIRAN implementasyon (SADECE test icin, uretim koduna ASLA
    girmez): mean/std istatistigi train+test HAVUZUNDAN fit edilir.
    `pipeline/cox_model.py:1502-1503`'un "fold disini goren" hatali
    varyanti -- asagidaki potens testinin enjekte ettigi bozma."""

    if not columns:
        return train_frame, test_frame

    pool = train_frame if test_frame is None else pd.concat([train_frame, test_frame])
    means = pool[columns].mean()
    stds = pool[columns].std(ddof=0)
    stds = stds.where(stds > 0, 1.0)

    train_frame = train_frame.copy()
    train_frame[columns] = (train_frame[columns] - means) / stds
    if test_frame is not None:
        test_frame = test_frame.copy()
        test_frame[columns] = (test_frame[columns] - means) / stds
    return train_frame, test_frame


def _make_standardization_perturbation_setup():
    """A2-X kurulumu: (baseline_frame, perturbed_frame, feature_columns,
    standardize_columns, target_fold, target_label, run_nested_cv kwargs).

    Perturbasyon X TARAFINDADIR (`feat_0`), Y'ye (survival_days/event)
    DOKUNULMAZ -- dis-fold bolunmesi `event` uzerinde stratified oldugu
    icin fold YAPISI degismez (A2'deki ayni gerekce). Bozulan hasta
    `target_fold`'un TEST tarafindadir: DOGRU kodda o hastanin ozellik
    degeri O FOLD'un standardizasyon istatistigine HIC girmemeli."""

    outer_splits = 3
    seed = 5
    frame, feature_columns = _make_multi_signal_survival_frame(140, n_features=8, seed=31)
    rng = np.random.default_rng(52)
    # Ham (standardize EDILMEMIS) yas kolonu -- uretimdeki
    # `clinical_standardize_columns = feature_columns + clinical_age_columns`
    # (v3b) deseninin sentetik esdegeri.
    frame["clinical_age"] = rng.uniform(low=20.0, high=85.0, size=len(frame))
    standardize_columns = feature_columns + ["clinical_age"]

    kwargs = dict(
        outer_splits=outer_splits,
        inner_splits=3,
        l1_ratio_grid=(1.0,),
        penalizer_grid=(0.05, 0.3),
        stability_selection=True,
        n_bootstrap_stability=20,
        stability_frequency_threshold=0.6,
        extra_columns=["clinical_age"],
        extra_column_penalizer=0.0,
        clinical_standardize_columns=standardize_columns,
        seed=seed,
    )

    splits = _outer_fold_splits_for_test(frame, outer_splits=outer_splits, seed=seed)
    target_fold = 0
    target_position = splits[target_fold][1][0]
    target_label = frame.index[target_position]
    assert target_position in splits[target_fold][1]
    assert target_position not in splits[target_fold][0]

    perturbed = frame.copy()
    perturbed.loc[target_label, "feat_0"] = (
        float(perturbed.loc[target_label, "feat_0"]) * _STD_PERTURBATION_SCALE
        + _STD_PERTURBATION_SHIFT
    )

    # Fold yapisi DEGISMEDI (X bozuldu, Y degil) -- karsilastirmanin on kosulu.
    assert (
        _outer_fold_splits_for_test(perturbed, outer_splits=outer_splits, seed=seed) == splits
    ), "X-perturbasyonu dis-fold bolunmesini degistirdi -- karsilastirma anlamsiz olurdu."

    return (
        frame,
        perturbed,
        feature_columns,
        standardize_columns,
        target_fold,
        target_label,
        kwargs,
    )


def _capture_standardize_calls(
    frame: pd.DataFrame,
    feature_columns: list[str],
    kwargs: dict,
    *,
    implementation=None,
) -> list[dict]:
    """`run_nested_cv()` icindeki HER `standardize_columns_fold_safe()`
    cagrisini casusla (XGBoost'taki `_spy` deseni) ve cagri baglamini +
    DONDURULEN cerceveleri kaydet.

    `implementation=None` -> gercek uretim fonksiyonu calisir (yalnizca
    gozlenir, davranis DEGISMEZ). `implementation` verilirse (potens
    testi) o bozuk implementasyon uretim fonksiyonunun YERINE gecer --
    `pipeline/cox_model.py` DISKTE DEGISTIRILMEZ."""

    real = cox_model_module.standardize_columns_fold_safe
    target = implementation if implementation is not None else real
    captured: list[dict] = []

    def _spy(train_frame, test_frame, columns):
        train_ids_in = frozenset(train_frame.index)
        test_ids_in = frozenset(test_frame.index) if test_frame is not None else frozenset()
        out_train, out_test = target(train_frame, test_frame, columns)
        column_list = list(columns or [])
        captured.append(
            {
                "columns": column_list,
                "train_ids": train_ids_in,
                "test_ids": test_ids_in,
                "out_train": out_train[column_list].copy(),
                "out_test": (out_test[column_list].copy() if out_test is not None else None),
            }
        )
        return out_train, out_test

    with patch.object(cox_model_module, "standardize_columns_fold_safe", side_effect=_spy):
        run_nested_cv(frame, feature_columns, **kwargs)
    return captured


def _max_abs_mean_and_std_deviation(record: dict) -> tuple[float, float]:
    """Bir cagrinin DONDURDUGU train cercevesinde, standardize edilen
    kolonlarin ortalamasinin 0'dan ve std'sinin (ddof=0) 1'den maksimum
    mutlak sapmasi. Istatistik SADECE train'den fit edildiyse ikisi de
    ~0 olmak ZORUNDADIR (cebirsel zorunluluk, gozlem degil); havuzdan
    (train+test) fit edildiyse ikisi de SIFIRDAN UZAKLASIR."""

    out_train = record["out_train"]
    return (
        float(out_train.mean().abs().max()),
        float((out_train.std(ddof=0) - 1.0).abs().max()),
    )


def test_run_nested_cv_standardization_statistics_are_fit_on_fold_train_only() -> None:
    """A2-X ASIL TEST (X-tarafli sizinti negatif kontrolu).

    `clinical_standardize_columns` DOLU (8 sentetik radyomik ozellik +
    yas) verilerek `standardize_columns_fold_safe()`'in no-op erken
    donusu (`pipeline/cox_model.py:1498`) BILINCLI olarak devre disi
    birakilir -- yani v3b'nin gercek yolu kosar. Dis fold'un TEST
    tarafindaki bir hastanin `feat_0` degeri sertce bozulur (X!) ve
    O FOLD'un standardizasyon ciktisinin BIT-BIREBIR ayni kalmasi
    beklenir.

    Yakaladigi regresyonlar:
      (a) istatistigin train+test havuzundan (veya tum cerceveden) fit
          edilmesi -> train ciktisinin mean'i artik 0, std'si artik 1
          OLMAZ + bozulan hastanin fold'unda cikti DEGISIR;
      (b) `run_nested_cv()`'nin bolunmeden ONCE standardize etmesi
          (cagri sayisi != outer_splits);
      (c) `columns` listesinin yolda bosalmasi (no-op'a dusme).

    Yakalamadigi: `run_nested_cv()`'nin DIS ciktilarinda bir fark --
    orada OZDESLIK var (blok basligindaki (1)+(2) argumani + asagidaki
    `..._documented_identity()` testi). Bu test bilincli olarak CAGRI
    NOKTASINDA olcer."""

    (
        frame,
        perturbed,
        feature_columns,
        standardize_columns,
        target_fold,
        target_label,
        kwargs,
    ) = _make_standardization_perturbation_setup()

    baseline_calls = _capture_standardize_calls(frame, feature_columns, kwargs)
    perturbed_calls = _capture_standardize_calls(perturbed, feature_columns, kwargs)

    # (b) + (c): fold BASINA bir cagri, ve `columns` GERCEKTEN dolu.
    assert len(baseline_calls) == kwargs["outer_splits"] == len(perturbed_calls), (
        "standardize_columns_fold_safe() dis-fold basina TAM BIR KEZ "
        f"cagrilmali (beklenen {kwargs['outer_splits']}, gercek "
        f"{len(baseline_calls)}) -- bolunmeden ONCE / fold disinda "
        "standardize ediliyor olabilir."
    )
    for record in baseline_calls:
        assert record["columns"] == standardize_columns, (
            "no-op erken donusu (cox_model.py:1498) tetiklenmis olabilir -- "
            "bu test `columns` DOLU iken anlamlidir."
        )
        assert not (record["train_ids"] & record["test_ids"]), (
            "SIZINTI: standardize_columns_fold_safe()'e verilen train ve "
            "test cerceveleri KESISIYOR."
        )

    # (a) CEBIRSEL INVARYANT: istatistik train'den fit edildiyse donen
    # train cercevesinin mean'i TAM 0, std'si (ddof=0) TAM 1 olmalidir.
    for fold_index, record in enumerate(baseline_calls):
        mean_dev, std_dev = _max_abs_mean_and_std_deviation(record)
        assert mean_dev < 1e-9 and std_dev < 1e-9, (
            f"SIZINTI REGRESYONU (fold {fold_index}): standardizasyondan "
            "SONRA train cercevesinin ortalamasi 0 / std'si 1 DEGIL "
            f"(max|mean|={mean_dev:.6g}, max|std-1|={std_dev:.6g}) -- "
            "istatistik bu fold'un KENDI train alt-kumesinden DEGIL, daha "
            "genis bir havuzdan (train+test veya tum cerceve) fit edilmis "
            "olmali."
        )

    # (a) X-PERTURBASYON ASSERT'I: bozulan hastanin TEST tarafinda oldugu
    # fold'un train ciktisi BIT-BIREBIR ayni.
    base_record = baseline_calls[target_fold]
    pert_record = perturbed_calls[target_fold]
    assert target_label not in base_record["train_ids"]
    assert target_label in base_record["test_ids"]
    assert base_record["train_ids"] == pert_record["train_ids"], (
        "Fold yapisi degisti -- X-perturbasyonu bolunmeyi etkilemis olmamali."
    )
    pd.testing.assert_frame_equal(
        base_record["out_train"],
        pert_record["out_train"],
        obj=(
            f"fold {target_fold} standardize-edilmis TRAIN cercevesi: dis "
            "TEST tarafindaki bir hastanin X degeri bozulunca DEGISTI -- "
            "standardizasyon istatistigi fold disini goruyor (SIZINTI)."
        ),
    )

    # POTENS 1 (perturbasyon GERCEKTEN ulasti mi?): ayni fold'un TEST
    # ciktisi DEGISMIS olmali; degismediyse yukaridaki assert bos bir
    # kontroldur.
    base_cell = float(base_record["out_test"].loc[target_label, "feat_0"])
    pert_cell = float(pert_record["out_test"].loc[target_label, "feat_0"])
    assert abs(pert_cell - base_cell) > 100.0, (
        "POTENS KONTROLU 1 BASARISIZ: X-perturbasyonu standardize edilmis "
        f"TEST cercevesine yansimadi (base={base_cell:.6g}, "
        f"perturbe={pert_cell:.6g}) -- perturbasyon fonksiyona hic "
        "ulasmamis olabilir, bu durumda sizinti assert'i hicbir sey "
        "kanitlamaz."
    )

    # POTENS 2 (perturbasyon fold DISINDA etkili mi?): bozulan hasta
    # DIGER fold'larin TRAIN tarafindadir -- orada standardizasyon
    # ciktisinin degismesi MESRUDUR ve gercekten degismelidir.
    other_folds = [i for i in range(len(baseline_calls)) if i != target_fold]
    assert any(
        target_label in baseline_calls[i]["train_ids"]
        and not baseline_calls[i]["out_train"].equals(perturbed_calls[i]["out_train"])
        for i in other_folds
    ), (
        "POTENS KONTROLU 2 BASARISIZ: bozulan hastanin EGITIM kumesinde "
        "oldugu fold'larda bile standardizasyon ciktisi DEGISMEDI -- "
        "perturbasyon etkisiz, test bos yere yesil olabilir."
    )


def test_run_nested_cv_standardization_leak_potency_leaky_implementation_is_detected() -> None:
    """A2-X POTENS/AYIRT EDICILIK KANITI (kalici KIRMIZI/YESIL kaydi).

    Yukaridaki testin assert'leri, sizdiran bir implementasyon altinda
    GERCEKTEN kirmizi oluyor mu? Burada `pipeline/cox_model.py` DISKTE
    DEGISTIRILMEDEN, bellekte `_leaky_standardize_columns_fold_safe()`
    (mean/std train+test HAVUZUNDAN fit edilir) enjekte edilir ve AYNI
    iki invaryantin IHLAL EDILDIGI assert edilir:
      (i) train ciktisinin mean'i 0 / std'si 1 OLMAZ;
      (ii) dis-TEST tarafindaki hastanin X'i bozulunca o fold'un train
           ciktisi DEGISIR.
    Bu test YESIL oldugu surece, asil testin yesil olmasi ANLAMLIDIR."""

    (
        frame,
        perturbed,
        feature_columns,
        _standardize_columns,
        target_fold,
        target_label,
        kwargs,
    ) = _make_standardization_perturbation_setup()

    leaky_baseline = _capture_standardize_calls(
        frame, feature_columns, kwargs, implementation=_leaky_standardize_columns_fold_safe
    )
    leaky_perturbed = _capture_standardize_calls(
        perturbed, feature_columns, kwargs, implementation=_leaky_standardize_columns_fold_safe
    )

    # (i) invaryant IHLAL EDILMELI -- ve marjin olculebilir buyuklukte olmali.
    worst_mean_dev = max(_max_abs_mean_and_std_deviation(r)[0] for r in leaky_baseline)
    worst_std_dev = max(_max_abs_mean_and_std_deviation(r)[1] for r in leaky_baseline)
    assert worst_mean_dev > 1e-3 or worst_std_dev > 1e-3, (
        "AYIRT EDICILIK KAYBI: havuzdan fit eden (sizdiran) implementasyon "
        f"bile mean/std invaryantini bozmadi (max|mean|={worst_mean_dev:.6g}, "
        f"max|std-1|={worst_std_dev:.6g}) -- asil testin (a) assert'i bos "
        "bir kontrol demektir, tolerans/n gozden gecirilmeli."
    )

    # (ii) X-perturbasyon assert'i de kirmizi olmali.
    base_record = leaky_baseline[target_fold]
    pert_record = leaky_perturbed[target_fold]
    assert target_label not in base_record["train_ids"]
    assert not base_record["out_train"].equals(pert_record["out_train"]), (
        "AYIRT EDICILIK KAYBI: sizdiran implementasyon altinda bile, dis-"
        "TEST hastasinin X'i bozuldugunda o fold'un train ciktisi "
        "DEGISMEDI -- asil testin X-perturbasyon assert'i bos bir kontrol "
        "demektir, perturbasyon buyutulmeli."
    )


def test_run_nested_cv_outputs_are_invariant_to_clinical_standardize_columns_documented_identity() -> None:
    """OLCULEN YAPISAL BULGU (2026-09-14) -- RAPORA GIRER, bir "iyilik"
    degil bir KISIT kaydidir.

    `clinical_standardize_columns` ACIK (8 ozellik + yas) ile KAPALI ([])
    kosullari `run_nested_cv()`'nin TUM dis ciktilarinda AYNI sonucu
    uretir: ayni `selected_features`, ayni `best_penalizer`/`best_l1_ratio`,
    ayni `c_index` (olculen max fark: 0.0). Sebep blok basliginda:
    (1) ayni afin donusum hem train hem test'e uygulanir, (2) lifelines
    CoxPHFitter kovaryatlari fit oncesi KENDI ICINDE standardize edip
    beta'lari geri olcekler -> penalizer olcek-bagimsiz uzayda uygulanir.

    SONUC (metodolojik): fold-guvenli standardizasyonun sizintisizligi
    `run_nested_cv()`'nin DIS ciktilarindan OLCULEMEZ; koruma cagri
    noktasinda olmak ZORUNDADIR (yukaridaki iki test). Bu test o
    ozdesligin regresyon kaydidir -- KIRILIRSA (orn. lifelines surumu
    ic normalizasyonu birakirsa, ya da penalizasyon disaridan olcege
    duyarli hale gelirse) yukaridaki argumanin varsayimi degismis
    demektir ve A2-X'in tasarimi YENIDEN degerlendirilmelidir."""

    frame, _perturbed, feature_columns, _cols, _fold, _label, kwargs = (
        _make_standardization_perturbation_setup()
    )

    with_standardization = run_nested_cv(frame, feature_columns, **kwargs)
    without_kwargs = dict(kwargs)
    without_kwargs["clinical_standardize_columns"] = []
    without_standardization = run_nested_cv(frame, feature_columns, **without_kwargs)

    for fold_index in with_standardization["fold"].tolist():
        a = with_standardization.loc[with_standardization["fold"] == fold_index].iloc[0]
        b = without_standardization.loc[without_standardization["fold"] == fold_index].iloc[0]
        assert list(a["selected_features"]) == list(b["selected_features"]), (
            f"fold {fold_index}: standardizasyon ACIK/KAPALI arasinda secim "
            "DEGISTI -- afin-degismezlik argumani (blok basligi (1)+(2)) "
            "artik gecerli olmayabilir; A2-X testlerinin gerekcesi gozden "
            "gecirilmeli."
        )
        assert float(a["best_penalizer"]) == float(b["best_penalizer"])
        assert float(a["best_l1_ratio"]) == float(b["best_l1_ratio"])
        assert np.isclose(float(a["c_index"]), float(b["c_index"]), atol=1e-8), (
            f"fold {fold_index}: c_index farki "
            f"{abs(float(a['c_index']) - float(b['c_index'])):.3g} -- "
            "ozdeslik bulgusu bozuldu."
        )
