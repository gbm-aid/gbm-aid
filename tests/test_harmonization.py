from __future__ import annotations

import copy
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from pipeline.harmonization import (
    ZSCORE_SCOPE_POOLED_ALL_MODALITIES,
    ZSCORE_SCOPE_T1CE_SOURCE,
    ComBatApplyLeakageError,
    ComBatFitLeakageError,
    ZScoreScopeMismatchError,
    ZScoreStatisticsNotFoundError,
    apply_combat_harmonization,
    apply_n4_bias_correction,
    apply_zscore_normalization,
    canonical_source,
    fit_combat_harmonization,
    fit_source_zscore_statistics,
    fit_source_zscore_statistics_if_needed,
    get_fitted_zscore_record,
    has_fitted_zscore_statistics,
    n4_output_path,
)


def _write_image(
    path: Path,
    array: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin((3.0, -2.0, 7.0))
    sitk.WriteImage(image, str(path), True)
    return path


def test_canonical_source_rejects_unknown_source() -> None:
    assert canonical_source("TCGA-GBM") == "TCGA"
    assert canonical_source("upenn") == "UPenn"
    assert canonical_source("LUMIERE") == "LUMIERE"
    with pytest.raises(ValueError, match="Desteklenmeyen kaynak"):
        canonical_source("unknown")


def test_canonical_source_recognizes_ucsf_alias() -> None:
    """2026-08-18 (Barış onayı) -- UCSF-PDGM tam kohort C32 çıkarımı.

    UCSF'in `canonical_source()`/N4/Z-score guard'larınca tanınması
    GEREKİR (aksi halde `apply_zscore_normalization(..., "UCSF")` her
    zaman `ValueError` fırlatır). AYRICA bu alias'ın Cox eğitim havuzu
    whitelist'ini (`COX_TRAINING_ALLOWED_SOURCES`) veya ComBat fit
    whitelist'ini (`COMBAT_FIT_ALLOWED_SOURCES`) GENİŞLETMEDİĞİNİ
    doğrula -- ikisi de SOURCE_ALIASES'ten BAĞIMSIZ, ayrı whitelist'ler.
    """

    from pipeline.cox_model import COX_TRAINING_ALLOWED_SOURCES
    from pipeline.harmonization import COMBAT_FIT_ALLOWED_SOURCES

    assert canonical_source("ucsf") == "UCSF"
    assert canonical_source("ucsf-pdgm") == "UCSF"
    assert canonical_source("UCSF-PDGM") == "UCSF"
    assert "UCSF" not in COX_TRAINING_ALLOWED_SOURCES
    assert "UCSF" not in COMBAT_FIT_ALLOWED_SOURCES


def test_n4_preserves_geometry_and_writes_new_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    z, y, x = np.indices((20, 24, 28))
    foreground = ((x - 14) ** 2 + (y - 12) ** 2 + (z - 10) ** 2) < 64
    bias = 0.7 + 0.6 * x / x.max()
    array = np.zeros(foreground.shape, dtype=np.float32)
    array[foreground] = 100.0 * bias[foreground]
    input_path = _write_image(
        tmp_path / "biased.nii.gz",
        array,
        spacing=(0.9, 1.1, 2.0),
    )

    monkeypatch.setenv("GBMAID_PROCESSED_ROOT", str(tmp_path / "processed"))
    monkeypatch.setenv("GBMAID_N4_ITERATIONS", "2,2")
    monkeypatch.setenv("GBMAID_N4_SHRINK_FACTOR", "2")

    output_path = Path(apply_n4_bias_correction(str(input_path)))
    original = sitk.ReadImage(str(input_path))
    corrected = sitk.ReadImage(str(output_path))

    assert output_path.is_file()
    assert output_path != input_path
    assert corrected.GetSize() == original.GetSize()
    assert corrected.GetSpacing() == original.GetSpacing()
    assert corrected.GetOrigin() == original.GetOrigin()
    assert corrected.GetDirection() == original.GetDirection()
    assert np.isfinite(sitk.GetArrayViewFromImage(corrected)).all()


def test_source_zscore_uses_fitted_cohort_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _write_image(
        tmp_path / "first_n4.nii.gz",
        np.arange(1, 65, dtype=np.float32).reshape(4, 4, 4),
    )
    second = _write_image(
        tmp_path / "second_n4.nii.gz",
        np.arange(101, 165, dtype=np.float32).reshape(4, 4, 4),
    )
    stats_path = tmp_path / "source_stats.json"
    monkeypatch.setenv("GBMAID_ZSCORE_STATS_PATH", str(stats_path))
    monkeypatch.setenv("GBMAID_PROCESSED_ROOT", str(tmp_path / "processed"))

    # `scope=` (2026-09-13, karar 28): `apply_zscore_normalization()` artık
    # stats dosyasında `zscore_scope` beyanı ZORUNLU kılıyor -- bu test
    # C32 runner'ının Pass 2 -> Pass 3 akışını taklit ettiği için beyanı
    # runner ile AYNI şekilde geçirir.
    stats = fit_source_zscore_statistics(
        [first, second],
        "TCGA",
        stats_path=stats_path,
        scope=ZSCORE_SCOPE_T1CE_SOURCE,
    )
    first_output = sitk.ReadImage(
        apply_zscore_normalization(str(first), "TCGA-GBM")
    )
    second_output = sitk.ReadImage(
        apply_zscore_normalization(str(second), "TCGA")
    )
    combined = np.concatenate(
        [
            sitk.GetArrayFromImage(first_output).ravel(),
            sitk.GetArrayFromImage(second_output).ravel(),
        ]
    )

    assert stats["image_count"] == 2
    assert stats["voxel_count"] == 128
    assert float(combined.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(combined.std()) == pytest.approx(1.0, abs=1e-6)


def test_zscore_refuses_missing_source_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = _write_image(
        tmp_path / "image_n4.nii.gz",
        np.ones((4, 4, 4), dtype=np.float32),
    )
    monkeypatch.setenv(
        "GBMAID_ZSCORE_STATS_PATH",
        str(tmp_path / "missing.json"),
    )
    with pytest.raises(ZScoreStatisticsNotFoundError):
        apply_zscore_normalization(str(image_path), "TCGA")


# ============================================================
# H4 (N4 cache) + B3 (Z-score refit-guard) -- 2026-08-13 review-gate
# düzeltmeleri (bkz. decisions/2026-08-13-pyradiomics-c32-bincount-
# karari.md "[2026-08-13] Review-gate BLOKE düzeltmeleri")
# ============================================================


@pytest.fixture()
def ascii_scratch_dir():
    """ASCII-only geçici klasör.

    pytest'in varsayılan `tmp_path`'i bu makinede kullanıcı profilinin
    Türkçe karakteri (`Barış`) yüzünden SimpleITK'nin NIfTI yazıcısıyla
    UYUŞMUYOR -- bu dosyadaki 3 pre-existing test (`test_n4_preserves_
    geometry_and_writes_new_file` vb.) tam bu yüzden HER ZAMAN
    başarısız (bilinen ortam sorunu, kod hatası DEĞİL). Aşağıdaki B3
    restart-guard testleri GERÇEKTEN NIfTI dosyası YAZMASI gerektiği
    için (sentetik "N4 çıktısı" üretmek üzere) `tmp_path` yerine bu
    ASCII-güvenli kökü kullanır -- aksi halde testler yanlış nedenle
    (ortam path sorunu, guard mantığı DEĞİL) başarısız görünürdü.
    """

    base = Path("C:/gbmaid_pytest_scratch") / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _write_ascii_image(path: Path, array: np.ndarray) -> Path:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path), True)
    return path


def test_n4_output_path_is_deterministic_and_matches_apply(
    ascii_scratch_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`n4_output_path()` gerçekten hesaplamadan `apply_n4_bias_correction()`
    ile AYNI yolu vermeli -- aksi halde H4 cache kontrolü çağıranlarda
    yanlış (asla cache-hit vermeyen ya da yanlış dosyayı "var" sayan)
    olurdu."""

    z, y, x = np.indices((12, 14, 16))
    foreground = ((x - 8) ** 2 + (y - 7) ** 2 + (z - 6) ** 2) < 25
    array = np.zeros(foreground.shape, dtype=np.float32)
    array[foreground] = 100.0
    input_path = _write_ascii_image(ascii_scratch_dir / "raw.nii.gz", array)

    monkeypatch.setenv("GBMAID_PROCESSED_ROOT", str(ascii_scratch_dir / "processed"))
    monkeypatch.setenv("GBMAID_N4_ITERATIONS", "2,2")
    monkeypatch.setenv("GBMAID_N4_SHRINK_FACTOR", "2")

    predicted_path = n4_output_path(str(input_path))
    assert not predicted_path.is_file()

    actual_path = Path(apply_n4_bias_correction(str(input_path)))
    assert actual_path == predicted_path
    assert predicted_path.is_file()


def test_fit_source_zscore_statistics_if_needed_skips_refit_when_already_fitted(
    ascii_scratch_dir: Path,
) -> None:
    """B3 KRİTİK regresyon testi -- "kısmen bitmiş + yeniden başlatılmış"
    senaryosunu GERÇEKTEN egzersiz eder (smoke testin kaçırdığı yol).

    Senaryo: Run 1 kohortun bir bölümünü (batch 1) işleyip Pass 2'sini
    tamamlıyor -- stats dosyasında TCGA için geçerli bir kayıt oluşuyor.
    Süreç kesiliyor, restart (Run 2) KALAN farklı bir alt-kümeyi (batch 2,
    KASITLI OLARAK batch 1'den FARKLI bir yoğunluk dağılımıyla) işliyor.
    Eski (koşulsuz `fit_source_zscore_statistics()`) davranışta restart'ın
    Pass 2'si SESSİZCE farklı bir (mean,std) ile üzerine yazardı --
    bu test tam olarak bunun ARTIK OLMADIĞINI kanıtlıyor: restart sonrası
    stats dosyası hâlâ Run 1'in (batch 1) değerini taşımalı.
    """

    stats_path = ascii_scratch_dir / "source_stats_t1ce_c32_test.json"

    # ---- Run 1: batch 1 (mean~100 civarında) ----
    batch1_paths = []
    for i in range(3):
        array = np.full((4, 4, 4), 100.0 + i, dtype=np.float32)
        batch1_paths.append(
            str(_write_ascii_image(ascii_scratch_dir / f"batch1_scan{i}.nii.gz", array))
        )

    # `scope=` (2026-09-13, karar 28): C32 runner'lari Pass 2'de bu
    # beyani geciriyor; test o akisi birebir taklit ediyor.
    record1 = fit_source_zscore_statistics_if_needed(
        batch1_paths,
        "TCGA",
        stats_path=stats_path,
        force=False,
        scope=ZSCORE_SCOPE_T1CE_SOURCE,
    )
    assert record1 is not None
    assert record1["mean"] == pytest.approx(101.0, abs=1e-6)
    assert has_fitted_zscore_statistics("TCGA", stats_path=stats_path)

    payload_after_run1 = json.loads(stats_path.read_text(encoding="utf-8"))
    mean_after_run1 = payload_after_run1["sources"]["TCGA"]["mean"]

    # ---- Run 2 (restart): batch 2, KASITLI OLARAK ÇOK FARKLI bir
    # yoğunluk dağılımıyla (mean~900) -- eski koşulsuz-fit davranışında
    # bu, stats dosyasındaki değeri SESSİZCE ~900'e kaydırırdı. ----
    batch2_paths = []
    for i in range(4):
        array = np.full((4, 4, 4), 900.0 + i, dtype=np.float32)
        batch2_paths.append(
            str(_write_ascii_image(ascii_scratch_dir / f"batch2_scan{i}.nii.gz", array))
        )

    record2 = fit_source_zscore_statistics_if_needed(
        batch2_paths,
        "TCGA-GBM",
        stats_path=stats_path,
        force=False,
        scope=ZSCORE_SCOPE_T1CE_SOURCE,
    )

    # KRİTİK doğrulama: refit ATLANDI (None döndü), dosyadaki değer
    # Run 1'in değeriyle BİREBİR AYNI kaldı -- restart'lar arası
    # TUTARLILIK garantisi kanıtlandı.
    assert record2 is None
    payload_after_run2 = json.loads(stats_path.read_text(encoding="utf-8"))
    mean_after_run2 = payload_after_run2["sources"]["TCGA"]["mean"]
    assert mean_after_run2 == mean_after_run1
    assert mean_after_run2 == pytest.approx(101.0, abs=1e-6)

    # apply_zscore_normalization() de -- Pass 3'ün gerçekte kullandığı
    # yol -- hâlâ Run 1'in istatistiğini kullanıyor mu, uçtan uca kanıtla.
    import os

    os.environ["GBMAID_ZSCORE_STATS_PATH"] = str(stats_path)
    os.environ["GBMAID_PROCESSED_ROOT"] = str(ascii_scratch_dir / "processed")
    try:
        normalized_path = apply_zscore_normalization(batch2_paths[0], "TCGA")
        normalized = sitk.GetArrayFromImage(sitk.ReadImage(normalized_path))
        # batch2 girdisi ~900, ama normalize eden istatistik hâlâ Run 1'in
        # (~101) -- yani sonuç büyük POZİTİF bir z-değeri olmalı (batch2
        # değeri Run 1 dağılımına göre aşırı uç), Run 2'nin KENDİ
        # (yakalanmamış) istatistiğiyle normalize edilseydi ~0 çıkardı.
        assert float(normalized.mean()) > 50.0
    finally:
        os.environ.pop("GBMAID_ZSCORE_STATS_PATH", None)
        os.environ.pop("GBMAID_PROCESSED_ROOT", None)


def test_fit_source_zscore_statistics_if_needed_force_true_overrides(
    ascii_scratch_dir: Path,
) -> None:
    """`force=True` (script'lerdeki `--fresh-fit`) AÇIKÇA istenirse refit
    ÇALIŞMALI -- guard'ın "asla refit etme" değil "istemeden refit etme"
    olduğunu kanıtlar."""

    stats_path = ascii_scratch_dir / "source_stats_t1ce_c32_force.json"

    batch1_array = np.full((4, 4, 4), 100.0, dtype=np.float32)
    batch1_array.flat[0] += 1.0  # sıfır-olmayan varyans için ufak sapma
    batch1_paths = [
        str(_write_ascii_image(ascii_scratch_dir / "f_batch1.nii.gz", batch1_array))
    ]
    fit_source_zscore_statistics_if_needed(batch1_paths, "UPenn", stats_path=stats_path)

    batch2_array = np.full((4, 4, 4), 500.0, dtype=np.float32)
    batch2_array.flat[0] += 1.0
    batch2_paths = [
        str(_write_ascii_image(ascii_scratch_dir / "f_batch2.nii.gz", batch2_array))
    ]
    record = fit_source_zscore_statistics_if_needed(
        batch2_paths, "UPenn", stats_path=stats_path, force=True
    )

    assert record is not None
    assert record["mean"] == pytest.approx(500.015625, abs=1e-3)
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert payload["sources"]["UPenn"]["mean"] == pytest.approx(500.015625, abs=1e-3)


def test_fit_source_zscore_statistics_if_needed_raises_when_nothing_to_fit(
    ascii_scratch_dir: Path,
) -> None:
    """Ne mevcut istatistik ne yeni N4 girdisi varsa SESSİZCE geçilmemeli --
    açık `ZScoreStatisticsNotFoundError` beklenir (sessiz fallback yasak)."""

    stats_path = ascii_scratch_dir / "source_stats_never_fitted.json"
    with pytest.raises(ZScoreStatisticsNotFoundError):
        fit_source_zscore_statistics_if_needed([], "LUMIERE", stats_path=stats_path)


def test_has_fitted_zscore_statistics_false_for_missing_file(
    ascii_scratch_dir: Path,
) -> None:
    stats_path = ascii_scratch_dir / "does_not_exist.json"
    assert has_fitted_zscore_statistics("TCGA", stats_path=stats_path) is False


# ============================================================
# A1.1 (2026-08-14) -- get_fitted_zscore_record() (image_count gate'in
# okuma katmanı, bkz. tools/run_pyradiomics_*.py'deki gerçek gate)
# ============================================================


def test_get_fitted_zscore_record_returns_none_for_missing_file(
    ascii_scratch_dir: Path,
) -> None:
    stats_path = ascii_scratch_dir / "does_not_exist.json"
    assert get_fitted_zscore_record("TCGA", stats_path=stats_path) is None


def test_get_fitted_zscore_record_returns_image_count_after_fit(
    ascii_scratch_dir: Path,
) -> None:
    stats_path = ascii_scratch_dir / "source_stats_record_test.json"
    paths = []
    for i in range(3):
        array = np.full((4, 4, 4), 100.0 + i, dtype=np.float32)
        paths.append(str(_write_ascii_image(ascii_scratch_dir / f"scan{i}.nii.gz", array)))

    fit_source_zscore_statistics(paths, "UPenn", stats_path=stats_path)

    record = get_fitted_zscore_record("UPenn-GBM", stats_path=stats_path)
    assert record is not None
    assert record["image_count"] == 3
    assert record["source"] == "UPenn"


def test_get_fitted_zscore_record_none_for_invalid_std(
    ascii_scratch_dir: Path,
) -> None:
    """Bozuk bir kayıt (std<=0) `has_fitted_zscore_statistics()` ile AYNI
    geçerlilik kriteriyle reddedilmeli -- sessizce geçersiz dict DÖNMEMELİ."""

    stats_path = ascii_scratch_dir / "source_stats_invalid.json"
    stats_path.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {
                    "TCGA": {
                        "source": "TCGA",
                        "mean": 100.0,
                        "std": 0.0,
                        "voxel_count": 10,
                        "image_count": 5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert get_fitted_zscore_record("TCGA", stats_path=stats_path) is None
    assert has_fitted_zscore_statistics("TCGA", stats_path=stats_path) is False


# ============================================================
# ComBat/neuroHarmonize sızıntı korumaları
# ============================================================


def test_fit_combat_harmonization_rejects_tcga_source() -> None:
    """`fit_combat_harmonization()` TCGA kaynağını fit'e ASLA almamalı."""

    feature_matrix = pd.DataFrame(
        {"f1": [0.1, 0.2, 0.15], "f2": [1.1, 1.3, 1.0]},
        index=["TCGA-0003", "TCGA-0006", "TCGA-0009"],
    )
    covariates = pd.DataFrame(
        {"SITE": ["TCGA", "TCGA", "TCGA"]}, index=feature_matrix.index
    )

    with pytest.raises(ComBatFitLeakageError, match="TCGA"):
        fit_combat_harmonization(
            feature_matrix, covariates, reference_batch="TCGA"
        )


def test_fit_combat_harmonization_accepts_db_style_source_aliases() -> None:
    """DB'nin gerçek `source_name` değerleri fit() guard'ından KABUL edilmeli.

    2026-08-12 FIX-3 regresyon testi (bkz. decisions/2026-08-11-apply-
    combat-harmonization-true-source-guard.md "[2026-08-12] FIX-3").
    Önceki guard ``covariates['SITE']``'ı HAM literal string olarak
    ``COMBAT_FIT_ALLOWED_SOURCES = {"UPenn", "LUMIERE"}`` ile
    karşılaştırıyordu -- `dataset_sources.source_name` DB kolonunun
    GERÇEK değeri literal ``"UPenn-GBM"`` olduğundan (bkz.
    `pipeline/cox_model.py` satır ~98), bu ham karşılaştırma MEŞRU
    UPenn verisini fail-safe ama üretimi kıran şekilde reddederdi.
    """

    rng = np.random.default_rng(7)
    upenn = pd.DataFrame(
        rng.normal(loc=0.0, scale=1.0, size=(6, 3)),
        columns=["f1", "f2", "f3"],
        index=[f"UPenn-{i}" for i in range(6)],
    )
    lumiere = pd.DataFrame(
        rng.normal(loc=5.0, scale=2.0, size=(6, 3)),
        columns=["f1", "f2", "f3"],
        index=[f"LUMIERE-{i}" for i in range(6)],
    )
    feature_matrix = pd.concat([upenn, lumiere])
    covariates = pd.DataFrame(
        {
            "SITE": ["UPenn-GBM", "upenn-gbm", " UPenn-GBM "]
            + ["UPenn-GBM"] * 3
            + ["LUMIERE"] * 6
        },
        index=feature_matrix.index,
    )

    model, harmonized, parameter_rows = fit_combat_harmonization(
        feature_matrix, covariates, reference_batch="UPenn"
    )

    assert set(str(label) for label in model["SITE_labels"]) == {"UPenn", "LUMIERE"}
    assert harmonized.shape == feature_matrix.shape
    assert {row["source_batch"] for row in parameter_rows} == {"UPenn", "LUMIERE"}


@pytest.mark.parametrize(
    "raw_tcga_source",
    ["TCGA", "TCGA-GBM", "tcga-gbm", " TCGA-GBM "],
)
def test_fit_combat_harmonization_rejects_tcga_gbm_alias(
    raw_tcga_source: str,
) -> None:
    """`"TCGA-GBM"` (ve varyantları) fit() whitelist guard'ından KAÇMAMALI.

    `apply_combat_harmonization()`'ın kardeş regresyon testiyle
    (`test_apply_combat_harmonization_rejects_tcga_gbm_alias`) aynı hata
    sınıfının fit() tarafındaki karşılığı -- guard artık kanonikleştirme
    kullandığı için "UPenn-GBM" kabul edilirken "TCGA-GBM" hâlâ
    reddedilmeli (whitelist mantığı: yalnızca UPenn/LUMIERE geçer).
    """

    feature_matrix = pd.DataFrame(
        {"f1": [0.1, 0.2, 0.15], "f2": [1.1, 1.3, 1.0]},
        index=["UPenn-0", "UPenn-1", "row-tcga"],
    )
    covariates = pd.DataFrame(
        {"SITE": ["UPenn", "UPenn", raw_tcga_source]}, index=feature_matrix.index
    )

    with pytest.raises(ComBatFitLeakageError, match="TCGA"):
        fit_combat_harmonization(
            feature_matrix, covariates, reference_batch="UPenn"
        )


def test_fit_combat_harmonization_rejects_unknown_source_loudly() -> None:
    """Tanınmayan bir kaynak adı fit()'te de SESSİZCE geçilmemeli.

    Whitelist mantığı gereği "tanınmıyorsa izin verme" zaten doğru
    davranıştır -- ama bu davranış açıkça kodlanmalı/test edilmeli,
    sessiz kabul YASAK (proje kuralı).
    """

    feature_matrix = pd.DataFrame(
        {"f1": [0.1, 0.2, 0.15], "f2": [1.1, 1.3, 1.0]},
        index=["UPenn-0", "UPenn-1", "row-mystery"],
    )
    covariates = pd.DataFrame(
        {"SITE": ["UPenn", "UPenn", "MysteryCohort"]}, index=feature_matrix.index
    )

    with pytest.raises(ComBatFitLeakageError, match="tanınmayan bir kaynak adı"):
        fit_combat_harmonization(
            feature_matrix, covariates, reference_batch="UPenn"
        )


def _fit_synthetic_upenn_lumiere_model() -> dict:
    """UPenn+LUMIERE'i taklit eden küçük, gerçek bir ComBat fit'i üret.

    Yalnızca test amaçlı sentetik veri -- gerçek radyomik özellik değil.
    """

    rng = np.random.default_rng(42)
    upenn = pd.DataFrame(
        rng.normal(loc=0.0, scale=1.0, size=(6, 3)),
        columns=["f1", "f2", "f3"],
        index=[f"UPenn-{i}" for i in range(6)],
    )
    lumiere = pd.DataFrame(
        rng.normal(loc=5.0, scale=2.0, size=(6, 3)),
        columns=["f1", "f2", "f3"],
        index=[f"LUMIERE-{i}" for i in range(6)],
    )
    feature_matrix = pd.concat([upenn, lumiere])
    covariates = pd.DataFrame(
        {"SITE": ["UPenn"] * 6 + ["LUMIERE"] * 6},
        index=feature_matrix.index,
    )
    model, _, _ = fit_combat_harmonization(
        feature_matrix, covariates, reference_batch="UPenn"
    )
    return model


def test_apply_combat_harmonization_rejects_tcga_even_when_site_spoofed() -> None:
    """`apply_combat_harmonization()` TCGA'yı `SITE` etiketinden BAĞIMSIZ reddetmeli.

    Bu, 2026-08-11'de bulunan açık kapıyı kapatır: `covariates['SITE']`
    referans-batch'e ('UPenn') spoof edilmiş olsa bile, gerçek kaynağı
    taşıyan `true_dataset_source='TCGA'` fonksiyonun sessizce çalışmasını
    engellemeli.
    """

    model = _fit_synthetic_upenn_lumiere_model()
    feature_matrix = pd.DataFrame(
        {"f1": [0.1], "f2": [0.2], "f3": [0.3]}, index=["TCGA-0003"]
    )
    # SITE kasten referans-batch'e ('UPenn') spoof edildi.
    covariates = pd.DataFrame({"SITE": ["UPenn"]}, index=feature_matrix.index)
    true_dataset_source = pd.Series(["TCGA"], index=feature_matrix.index)

    with pytest.raises(ComBatApplyLeakageError, match="TCGA"):
        apply_combat_harmonization(
            feature_matrix,
            covariates,
            model,
            true_dataset_source=true_dataset_source,
        )


def test_apply_combat_harmonization_rejects_tcga_case_insensitive() -> None:
    model = _fit_synthetic_upenn_lumiere_model()
    feature_matrix = pd.DataFrame(
        {"f1": [0.1], "f2": [0.2], "f3": [0.3]}, index=["row-0"]
    )
    covariates = pd.DataFrame({"SITE": ["UPenn"]}, index=feature_matrix.index)
    true_dataset_source = pd.Series([" tcga "], index=feature_matrix.index)

    with pytest.raises(ComBatApplyLeakageError):
        apply_combat_harmonization(
            feature_matrix,
            covariates,
            model,
            true_dataset_source=true_dataset_source,
        )


@pytest.mark.parametrize(
    "raw_source",
    ["TCGA-GBM", "tcga-gbm", " TCGA-GBM ", "Tcga-Gbm"],
)
def test_apply_combat_harmonization_rejects_tcga_gbm_alias(raw_source: str) -> None:
    """`"TCGA-GBM"` (ve varyantları) guard'dan KAÇMAMALI.

    2026-08-12 CRITICAL fix regresyon testi (Codex çapraz incelemesinde
    bulundu, bkz. decisions/2026-08-11-apply-combat-harmonization-
    true-source-guard.md "[2026-08-12] REVIEW-GATE"/"FIX-2"). Önceki
    guard yalnız düz `.strip().upper() == "TCGA"` karşılaştırması
    yapıyordu -- `"TCGA-GBM"` bunu YAKALAMIYORDU. `dataset_sources.
    source_name` DB kolonunun gerçek değeri literal `"TCGA-GBM"`
    (bkz. `pipeline/cox_model.py` satır ~98), düz `"TCGA"` DEĞİL --
    bu yüzden bu senaryo spekülatif değil, gerçek/canlı bir risk.
    """

    model = _fit_synthetic_upenn_lumiere_model()
    feature_matrix = pd.DataFrame(
        {"f1": [0.1], "f2": [0.2], "f3": [0.3]}, index=["row-0"]
    )
    # SITE kasten referans-batch'e ('UPenn') spoof edildi -- guard'ın
    # SITE'a değil true_dataset_source'a güvendiğini de doğrular.
    covariates = pd.DataFrame({"SITE": ["UPenn"]}, index=feature_matrix.index)
    true_dataset_source = pd.Series([raw_source], index=feature_matrix.index)

    with pytest.raises(ComBatApplyLeakageError, match="TCGA"):
        apply_combat_harmonization(
            feature_matrix,
            covariates,
            model,
            true_dataset_source=true_dataset_source,
        )


def test_apply_combat_harmonization_rejects_unknown_source_loudly() -> None:
    """Tanınmayan bir kaynak adı SESSİZCE geçilmemeli, `ValueError` fırlatılmalı.

    Guard'ın `canonical_source()` normalizasyonu, ne TCGA ne UPenn ne
    LUMIERE alias'ı olan bir değer gördüğünde (örn. yazım hatası)
    "büyük ihtimalle TCGA değildir, geç" diye sessiz varsayım YAPMAMALI
    -- proje kuralı "sessiz fallback yasak".
    """

    model = _fit_synthetic_upenn_lumiere_model()
    feature_matrix = pd.DataFrame(
        {"f1": [0.1], "f2": [0.2], "f3": [0.3]}, index=["row-0"]
    )
    covariates = pd.DataFrame({"SITE": ["UPenn"]}, index=feature_matrix.index)
    true_dataset_source = pd.Series(["MysteryCohort"], index=feature_matrix.index)

    with pytest.raises(ValueError, match="tanınmayan bir kaynak adı"):
        apply_combat_harmonization(
            feature_matrix,
            covariates,
            model,
            true_dataset_source=true_dataset_source,
        )


def test_apply_combat_harmonization_requires_true_dataset_source_series() -> None:
    """`true_dataset_source` zorunlu ve `pandas.Series` olmalı -- listeyle spoof edilemez."""

    model = _fit_synthetic_upenn_lumiere_model()
    feature_matrix = pd.DataFrame(
        {"f1": [0.1], "f2": [0.2], "f3": [0.3]}, index=["row-0"]
    )
    covariates = pd.DataFrame({"SITE": ["UPenn"]}, index=feature_matrix.index)

    with pytest.raises(TypeError, match="true_dataset_source"):
        apply_combat_harmonization(
            feature_matrix,
            covariates,
            model,
            true_dataset_source=["UPenn"],  # type: ignore[arg-type]
        )


def test_apply_combat_harmonization_requires_matching_index() -> None:
    model = _fit_synthetic_upenn_lumiere_model()
    feature_matrix = pd.DataFrame(
        {"f1": [0.1], "f2": [0.2], "f3": [0.3]}, index=["row-0"]
    )
    covariates = pd.DataFrame({"SITE": ["UPenn"]}, index=feature_matrix.index)
    mismatched_true_source = pd.Series(["UPenn"], index=["row-1"])

    with pytest.raises(ValueError, match="true_dataset_source"):
        apply_combat_harmonization(
            feature_matrix,
            covariates,
            model,
            true_dataset_source=mismatched_true_source,
        )


def test_apply_combat_harmonization_allows_legitimate_non_tcga_source() -> None:
    """Gerçek UPenn/LUMIERE satırı, doğru `true_dataset_source` ile normal çalışmalı.

    NOT: 2026-08-12'ye kadar bu test xfail idi (ref_level IndexError bug'ı
    yüzünden) -- bkz. decisions/2026-08-11-apply-combat-harmonization-
    true-source-guard.md "[2026-08-12] FIX" bölümü. Bug düzeltildi, xfail
    kaldırıldı.
    """

    model = _fit_synthetic_upenn_lumiere_model()
    feature_matrix = pd.DataFrame(
        {"f1": [0.1], "f2": [0.2], "f3": [0.3]}, index=["UPenn-new-0"]
    )
    covariates = pd.DataFrame({"SITE": ["UPenn"]}, index=feature_matrix.index)
    true_dataset_source = pd.Series(["UPenn"], index=feature_matrix.index)

    result = apply_combat_harmonization(
        feature_matrix,
        covariates,
        model,
        true_dataset_source=true_dataset_source,
    )

    assert list(result.columns) == ["f1", "f2", "f3"]
    assert result.index.tolist() == ["UPenn-new-0"]


def test_fit_then_apply_round_trip_with_db_style_source_aliases() -> None:
    """DB-stili alias'larla fit() SONRA apply() -- FIX-4 regresyon testi.

    2026-08-12 FIX-4 (bkz. decisions/2026-08-11-apply-combat-harmonization-
    true-source-guard.md "[2026-08-12] REVIEW-GATE FIX-3 SONUCU" ve
    "[2026-08-12] FIX-4"). FIX-3, `fit_combat_harmonization()`'ı DB-stili
    alias'ları ("UPenn-GBM") kabul edip KANONİK SITE_labels ile fit
    edecek şekilde düzeltti -- ama `apply_combat_harmonization()`'ın
    `known_site_labels` kontrolü hâlâ ham string bekliyordu, bu yüzden
    aynı DB-stili alias'larla apply() çağrıldığında (Hafta 3'te en doğal
    implementasyon: `true_dataset_source` VE `covariates['SITE']` aynı
    DB kolonundan dolduruluyor) ValueError fırlıyordu. reviewer bunu
    canlı çalıştırarak reprodüksiyon etti. Bu test o senaryoyu birebir
    tekrarlar: fit() DB-stili alias'larla, apply() de AYNI DB-stili
    alias'larla -- artık ValueError FIRLAMAMALI.
    """

    rng = np.random.default_rng(11)
    upenn = pd.DataFrame(
        rng.normal(loc=0.0, scale=1.0, size=(6, 3)),
        columns=["f1", "f2", "f3"],
        index=[f"UPenn-{i}" for i in range(6)],
    )
    lumiere = pd.DataFrame(
        rng.normal(loc=5.0, scale=2.0, size=(6, 3)),
        columns=["f1", "f2", "f3"],
        index=[f"LUMIERE-{i}" for i in range(6)],
    )
    feature_matrix = pd.concat([upenn, lumiere])
    fit_covariates = pd.DataFrame(
        {"SITE": ["UPenn-GBM"] * 6 + ["LUMIERE"] * 6},
        index=feature_matrix.index,
    )

    model, _, _ = fit_combat_harmonization(
        feature_matrix, fit_covariates, reference_batch="UPenn"
    )

    new_row = pd.DataFrame(
        {"f1": [0.12], "f2": [-0.34], "f3": [0.56]}, index=["UPenn-new-0"]
    )
    apply_covariates = pd.DataFrame(
        {"SITE": ["UPenn-GBM"]}, index=new_row.index
    )
    true_dataset_source = pd.Series(["UPenn-GBM"], index=new_row.index)

    result = apply_combat_harmonization(
        new_row,
        apply_covariates,
        model,
        true_dataset_source=true_dataset_source,
    )

    assert list(result.columns) == ["f1", "f2", "f3"]
    assert result.index.tolist() == ["UPenn-new-0"]
    assert np.isfinite(result.to_numpy()).all()

    # UPenn = reference_batch -- FIX ile aynı istatistiksel kural burada
    # da geçerli olmalı: referans-batch satırı için apply() ham veriyi
    # DEĞİŞTİRMEMELİ (bkz. yukarıdaki "identity" testinin gerekçesi).
    np.testing.assert_allclose(
        result.to_numpy(), new_row.to_numpy(), atol=1e-8
    )


def test_apply_combat_harmonization_reference_batch_row_is_identity() -> None:
    """UPenn (referans-batch) satırı apply'dan sonra HAM veriyle birebir aynı olmalı.

    ComBat'ın referans-batch semantiğinin matematiksel zorunlu sonucu
    (bkz. `apply_combat_harmonization()` docstring'indeki "KRİTİK
    İSTATİSTİKSEL NOT"). ref_level bug'ı düzeltilmeden önce bu satır
    IndexError ile çöküyordu (SITE_labels=['LUMIERE','UPenn'] alfabetik
    sıralamada ref_level=1 çıkıyor, tek-satır yerel one-hot'ta geçersiz
    indeks).
    """

    model = _fit_synthetic_upenn_lumiere_model()
    feature_matrix = pd.DataFrame(
        {"f1": [0.30471708], "f2": [-1.03998411], "f3": [0.7504512]},
        index=["UPenn-0"],
    )
    covariates = pd.DataFrame({"SITE": ["UPenn"]}, index=feature_matrix.index)
    true_dataset_source = pd.Series(["UPenn"], index=feature_matrix.index)

    result = apply_combat_harmonization(
        feature_matrix,
        covariates,
        model,
        true_dataset_source=true_dataset_source,
    )

    np.testing.assert_allclose(
        result.to_numpy(), feature_matrix.to_numpy(), atol=1e-8
    )


def test_apply_combat_harmonization_canonicalizes_raw_model_site_labels_no_silent_nan() -> (
    None
):
    """FIX-5 bulgu #2+#3 regresyon testi (reviewer'ın canlı reprodüksiyon deseni).

    Bkz. decisions/2026-08-11-apply-combat-harmonization-true-source-guard.md
    "[2026-08-12] REVIEW-GATE FIX-4 SONUCU". FIX-4'ün `known_site_labels`
    doğrulaması kanonikti ve GEÇİYORDU, ama kanonikleştirilmiş etiketler
    `model` dict'ine geri YAZILMIYORDU -- `_apply_model_one_fixed()` hâlâ
    ham (potansiyel eski-tip) `model['SITE_labels']` kullanıyordu, bu da
    `is_train_site` hesaplamasını hep `False` yapıp SESSİZCE NaN üretiyordu
    (hiçbir hata fırlamadan). Bu test taze fit edilmiş bir modeli
    deepcopy'leyip `SITE_labels`'ı elle ham forma ("UPenn" -> "UPenn-GBM")
    çevirerek FIX-3-öncesi bir "eski model" senaryosunu simüle eder --
    FIX-5'ten sonra sonuç artık NaN DEĞİL, referans-batch identity kuralına
    uyan sonlu bir değer olmalı.
    """

    model = _fit_synthetic_upenn_lumiere_model()
    # Taze fit edilmiş modelin SITE_labels'ı zaten kanonik (FIX-3 sayesinde).
    assert set(str(label) for label in model["SITE_labels"]) == {"LUMIERE", "UPenn"}

    old_style_model = copy.deepcopy(model)
    old_style_model["SITE_labels"] = np.array(
        [
            "UPenn-GBM" if str(label) == "UPenn" else str(label)
            for label in old_style_model["SITE_labels"]
        ],
        dtype=object,
    )
    assert set(str(label) for label in old_style_model["SITE_labels"]) == {
        "LUMIERE",
        "UPenn-GBM",
    }

    feature_matrix = pd.DataFrame(
        {"f1": [0.12], "f2": [-0.34], "f3": [0.56]}, index=["UPenn-new-0"]
    )
    covariates = pd.DataFrame({"SITE": ["UPenn-GBM"]}, index=feature_matrix.index)
    true_dataset_source = pd.Series(["UPenn-GBM"], index=feature_matrix.index)

    result = apply_combat_harmonization(
        feature_matrix,
        covariates,
        old_style_model,
        true_dataset_source=true_dataset_source,
    )

    assert np.isfinite(result.to_numpy()).all(), (
        "FIX-5 öncesi bu senaryo SESSİZCE NaN üretiyordu (reviewer canlı "
        "reprodüksiyon etti) -- guard geçiyordu ama _apply_model_one_fixed() "
        "hâlâ ham model['SITE_labels'] kullanıyordu."
    )
    # UPenn referans-batch olduğu için identity kısayolu geçerli olmalı --
    # sadece "NaN değil" yetmez, matematiksel olarak da doğru olmalı.
    np.testing.assert_allclose(
        result.to_numpy(), feature_matrix.to_numpy(), atol=1e-8
    )
    # Orijinal `old_style_model` (çağıranın elindeki) mutasyona uğramamalı --
    # `apply_combat_harmonization()` içeride SADECE `dict(model)` sığ
    # kopyasının SITE_labels'ını üzerine yazıyor, girdi model'i değiştirmiyor.
    assert set(str(label) for label in old_style_model["SITE_labels"]) == {
        "LUMIERE",
        "UPenn-GBM",
    }


def test_fit_combat_harmonization_canonicalizes_reference_batch_alias() -> None:
    """FIX-5 bulgu #1 regresyon testi: `reference_batch="UPenn-GBM"` artık KABUL edilmeli.

    Bkz. decisions/2026-08-11-apply-combat-harmonization-true-source-guard.md
    "[2026-08-12] REVIEW-GATE FIX-4 SONUCU" (Codex bulgusu). Önceden
    `reference_batch` parametresi `canonical_source()`'tan hiç geçirilmeden
    kanonikleştirilmiş `sources_present` seti ile karşılaştırılıyordu --
    biri açıkça DB-stili bir alias (`"UPenn-GBM"`) verirse `ValueError`
    yanlış-pozitif fırlardı. Artık `reference_batch` de kanonikleştirilip
    hem guard'da hem `nh.harmonizationLearn(ref_batch=...)` çağrısında
    kullanılıyor.
    """

    rng = np.random.default_rng(3)
    upenn = pd.DataFrame(
        rng.normal(loc=0.0, scale=1.0, size=(6, 3)),
        columns=["f1", "f2", "f3"],
        index=[f"UPenn-{i}" for i in range(6)],
    )
    lumiere = pd.DataFrame(
        rng.normal(loc=5.0, scale=2.0, size=(6, 3)),
        columns=["f1", "f2", "f3"],
        index=[f"LUMIERE-{i}" for i in range(6)],
    )
    feature_matrix = pd.concat([upenn, lumiere])
    covariates = pd.DataFrame(
        {"SITE": ["UPenn"] * 6 + ["LUMIERE"] * 6}, index=feature_matrix.index
    )

    model, harmonized, _ = fit_combat_harmonization(
        feature_matrix, covariates, reference_batch="UPenn-GBM"
    )

    assert model["ref_batch"] == "UPenn"
    # UPenn satırları referans-batch identity'si -- ham veriyle birebir
    # aynı olmalı (fit başarıyla "UPenn" referans-batch'iyle tamamlandı).
    np.testing.assert_allclose(
        harmonized.loc[upenn.index].to_numpy(), upenn.to_numpy(), atol=1e-8
    )


def test_apply_combat_harmonization_matches_in_sample_fit_for_non_reference_batch() -> (
    None
):
    """LUMIERE (referans-olmayan) satırının out-of-sample apply çıktısı,
    aynı satırın fit-zamanındaki in-sample harmonize edilmiş değeriyle
    (`fit_combat_harmonization()`'ın döndürdüğü ``harmonized_in_sample``)
    tutarlı olmalı -- ComBat parametreleri (gamma/delta) aynı olduğu için
    matematiksel olarak eşit çıkması beklenir, bu ref_level bug fix'inin
    doğruluğunu ampirik olarak doğrular (sadece IndexError atmaması
    yetmez, üretilen değerler de doğru olmalı).
    """

    rng = np.random.default_rng(42)
    upenn = pd.DataFrame(
        rng.normal(loc=0.0, scale=1.0, size=(6, 3)),
        columns=["f1", "f2", "f3"],
        index=[f"UPenn-{i}" for i in range(6)],
    )
    lumiere = pd.DataFrame(
        rng.normal(loc=5.0, scale=2.0, size=(6, 3)),
        columns=["f1", "f2", "f3"],
        index=[f"LUMIERE-{i}" for i in range(6)],
    )
    feature_matrix = pd.concat([upenn, lumiere])
    covariates = pd.DataFrame(
        {"SITE": ["UPenn"] * 6 + ["LUMIERE"] * 6}, index=feature_matrix.index
    )
    model, harmonized_in_sample, _ = fit_combat_harmonization(
        feature_matrix, covariates, reference_batch="UPenn"
    )

    lumiere_row = feature_matrix.loc[["LUMIERE-0"]]
    lumiere_cov = covariates.loc[["LUMIERE-0"]]
    true_dataset_source = pd.Series(["LUMIERE"], index=lumiere_row.index)

    result = apply_combat_harmonization(
        lumiere_row,
        lumiere_cov,
        model,
        true_dataset_source=true_dataset_source,
    )

    np.testing.assert_allclose(
        result.to_numpy(),
        harmonized_in_sample.loc[["LUMIERE-0"]].to_numpy(),
        atol=1e-8,
    )


# ============================================================
# 2026-09-13, KARAR 29 -- ComBat-apply guard'ı UCSF'i de reddeder
# ============================================================


@pytest.mark.parametrize("raw_source", ["UCSF", "ucsf", "UCSF-PDGM", " ucsf-pdgm "])
def test_apply_combat_harmonization_rejects_ucsf_even_when_site_spoofed(
    raw_source: str,
) -> None:
    """UCSF, TCGA ile AYNI muameleyi görmeli -- ComBat apply'a GİRMEZ.

    CLAUDE.md 2026-08-18 kuralı: *"UCSF ... feature-level ComBat düzeltmesi
    ALMAZ, ComBat fit'ine GİRMEZ"* (harici test setine eğitim-türevi bir
    dönüşüm uygulamak bağımsızlığını zedeler).

    2026-09-13 öncesi guard `canonical_true_source == "TCGA"` idi -- bir
    BLACKLIST. `SOURCE_ALIASES`'e 2026-08-18'de eklenen `"UCSF"`/
    `"ucsf-pdgm"` bu blacklist'ten SESSİZCE geçiyordu. Guard artık
    `COMBAT_FIT_ALLOWED_SOURCES` whitelist'inin TÜMLEYENİ.

    `covariates['SITE']` KASITLI olarak "UPenn" diye spoof edilmiştir:
    reddin SITE'a değil, spoof edilemez `true_dataset_source`'a bakarak
    yapıldığını kanıtlar.
    """

    model = _fit_synthetic_upenn_lumiere_model()
    feature_matrix = pd.DataFrame(
        {"f1": [0.1], "f2": [0.2], "f3": [0.3]}, index=["UCSF-999"]
    )
    covariates = pd.DataFrame({"SITE": ["UPenn"]}, index=feature_matrix.index)
    true_dataset_source = pd.Series([raw_source], index=feature_matrix.index)

    with pytest.raises(ComBatApplyLeakageError) as excinfo:
        apply_combat_harmonization(
            feature_matrix,
            covariates,
            model,
            true_dataset_source=true_dataset_source,
        )

    message = str(excinfo.value)
    assert "UCSF" in message
    assert "UCSF-999" in message


def test_apply_combat_harmonization_rejects_mixed_upenn_and_ucsf() -> None:
    """Karışık batch: tek bir UCSF satırı bile TÜM çağrıyı reddettirir."""

    model = _fit_synthetic_upenn_lumiere_model()
    feature_matrix = pd.DataFrame(
        {"f1": [0.1, 0.4], "f2": [0.2, 0.5], "f3": [0.3, 0.6]},
        index=["UPenn-new-0", "UCSF-001"],
    )
    covariates = pd.DataFrame(
        {"SITE": ["UPenn", "UPenn"]}, index=feature_matrix.index
    )
    true_dataset_source = pd.Series(
        ["UPenn-GBM", "UCSF-PDGM"], index=feature_matrix.index
    )

    with pytest.raises(ComBatApplyLeakageError) as excinfo:
        apply_combat_harmonization(
            feature_matrix,
            covariates,
            model,
            true_dataset_source=true_dataset_source,
        )
    assert "UCSF-001" in str(excinfo.value)
    # Meşru UPenn satırı suçlu olarak GÖSTERİLMEMELİ.
    assert "UPenn-new-0" not in str(excinfo.value)


def test_apply_combat_harmonization_guard_is_whitelist_complement() -> None:
    """Guard, SABİT bir {TCGA,UCSF} listesi DEĞİL, whitelist tümleyeni olmalı.

    Bu, "ileride `SOURCE_ALIASES`'e yeni bir kaynak eklenirse guard
    kendiliğinden korur mu" sorusunun çalıştırılabilir cevabı. UCSF'te
    tam olarak bu senaryo yaşandı (2026-08-18'de alias eklendi, guard
    güncellenmedi, sessizce açık kaldı).

    `SOURCE_ALIASES`'e geçici, sentetik bir kaynak enjekte edilir; guard
    kodunda o kaynağın adı HİÇ yazılı olmadığı halde reddedilmelidir.
    """

    from pipeline import harmonization as harmonization_module

    model = _fit_synthetic_upenn_lumiere_model()
    feature_matrix = pd.DataFrame(
        {"f1": [0.1], "f2": [0.2], "f3": [0.3]}, index=["FUTURE-0"]
    )
    covariates = pd.DataFrame({"SITE": ["UPenn"]}, index=feature_matrix.index)
    true_dataset_source = pd.Series(["future-cohort"], index=feature_matrix.index)

    original_aliases = dict(harmonization_module.SOURCE_ALIASES)
    harmonization_module.SOURCE_ALIASES["future-cohort"] = "FUTURECOHORT"
    try:
        with pytest.raises(ComBatApplyLeakageError) as excinfo:
            apply_combat_harmonization(
                feature_matrix,
                covariates,
                model,
                true_dataset_source=true_dataset_source,
            )
        assert "FUTURECOHORT" in str(excinfo.value)
    finally:
        harmonization_module.SOURCE_ALIASES.clear()
        harmonization_module.SOURCE_ALIASES.update(original_aliases)


def test_apply_combat_harmonization_still_rejects_tcga_after_widening() -> None:
    """Genişletme, ESKİ TCGA reddini bozmamış olmalı (regresyon)."""

    model = _fit_synthetic_upenn_lumiere_model()
    feature_matrix = pd.DataFrame(
        {"f1": [0.1], "f2": [0.2], "f3": [0.3]}, index=["TCGA-02-0003"]
    )
    covariates = pd.DataFrame({"SITE": ["UPenn"]}, index=feature_matrix.index)
    true_dataset_source = pd.Series(["TCGA-GBM"], index=feature_matrix.index)

    with pytest.raises(ComBatApplyLeakageError) as excinfo:
        apply_combat_harmonization(
            feature_matrix,
            covariates,
            model,
            true_dataset_source=true_dataset_source,
        )
    assert "TCGA" in str(excinfo.value)


# ============================================================
# 2026-09-13, KARAR 28 -- Z-score KAPSAM (scope) guard'ı
# ============================================================
#
# C32 sözleşmesi T1ce-ÖZEL kaynak Z-score'u ZORUNLU kılar; havuzlanmış
# (tüm modaliteler) Z-score YASAKTIR. `artifacts/week2/zscore/
# source_stats.json` havuzlanmış bir dosyadır ve diskte CANLI durur --
# C32 üretim dosyalarından YAPISAL OLARAK AYIRT EDİLEMEZ (aynı
# `version`/`fit_stage`/`sources` anahtarları). Hangisinin kullanılacağını
# yalnız `GBMAID_ZSCORE_STATS_PATH` belirler.


POOLED_STATS_PATH = (
    Path(__file__).absolute().parents[1]
    / "artifacts"
    / "week2"
    / "zscore"
    / "source_stats.json"
)


def _dummy_nifti(path: Path) -> Path:
    """`_require_nifti()`'yi geçecek kadar bir dosya -- ITK ile OKUNMAZ.

    Kapsam guard'ı, görüntü ITK ile okunmadan ÖNCE çalışır; bu yüzden
    KIRMIZI testler gerçek bir NIfTI'ye ihtiyaç duymaz (ve Türkçe-yol
    ITK sorunundan bağımsızdır).
    """

    path.write_bytes(b"not-a-real-nifti")
    return path


def test_apply_zscore_rejects_pooled_scope_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KIRMIZI: havuzlanmış kapsam BEYAN EDEN dosya REDDEDİLİR."""

    stats_path = tmp_path / "pooled_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "version": 1,
                "fit_stage": "post_n4",
                "zscore_scope": ZSCORE_SCOPE_POOLED_ALL_MODALITIES,
                "sources": {
                    "UPenn": {
                        "source": "UPenn",
                        "mean": 412.8116465211492,
                        "std": 368.12396133704146,
                        "voxel_count": 3998056517,
                        "image_count": 2684,
                        "foreground_rule": "finite_nonzero",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GBMAID_ZSCORE_STATS_PATH", str(stats_path))
    image_path = _dummy_nifti(tmp_path / "image_n4.nii.gz")

    with pytest.raises(ZScoreScopeMismatchError) as excinfo:
        apply_zscore_normalization(str(image_path), "UPenn")

    message = str(excinfo.value)
    assert ZSCORE_SCOPE_POOLED_ALL_MODALITIES in message
    assert ZSCORE_SCOPE_T1CE_SOURCE in message


def test_apply_zscore_rejects_the_real_pooled_week2_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KIRMIZI (CANLI DOSYA): gerçek `artifacts/week2/zscore/source_stats.json`
    guard tarafından reddedilir.

    Bu test sentetik değil: diskte duran GERÇEK havuzlanmış üretim-dışı
    dosyayı `GBMAID_ZSCORE_STATS_PATH` ile işaret eder. Dosya
    SİLİNMEDİ/TAŞINMADI (5 kod dosyası + karar/log dosyaları bu yolu
    anıyor; wiki hard rule #4 silmeyi yasaklıyor) -- yerinde bırakıldı ve
    içine `zscore_scope: source_pooled_all_modalities` + `_WARNING`
    alanları eklendi. Bu test o işaretlemenin GERÇEKTEN guard'ı
    ateşlediğini kanıtlar.
    """

    if not POOLED_STATS_PATH.is_file():
        pytest.skip(f"Havuzlanmış Hafta 2 dosyası diskte yok: {POOLED_STATS_PATH}")

    payload = json.loads(POOLED_STATS_PATH.read_text(encoding="utf-8"))
    # İşaretleme yerinde mi?
    assert payload.get("zscore_scope") == ZSCORE_SCOPE_POOLED_ALL_MODALITIES
    assert "YASAK" in payload.get("_WARNING", "")
    # Tarihsel içerik BOZULMAMIŞ olmalı (havuzlanmış image_count'lar).
    assert payload["sources"]["UPenn"]["image_count"] == 2684
    assert payload["sources"]["LUMIERE"]["image_count"] == 2152
    assert payload["sources"]["TCGA"]["image_count"] == 154

    monkeypatch.setenv("GBMAID_ZSCORE_STATS_PATH", str(POOLED_STATS_PATH))
    image_path = _dummy_nifti(tmp_path / "image_n4.nii.gz")

    for source in ("UPenn", "LUMIERE", "TCGA"):
        with pytest.raises(ZScoreScopeMismatchError):
            apply_zscore_normalization(str(image_path), source)


def test_apply_zscore_rejects_undeclared_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KIRMIZI: kapsam BEYAN ETMEYEN dosya da reddedilir (fail-closed).

    Kritik ayrım: beyansız dosya "muhtemelen T1ce'dir" diye SESSİZCE
    kabul EDİLMEZ. Havuzlanmış ve T1ce dosyaları yapısal olarak
    ayırt edilemez olduğu için, beyan yokluğunda doğru davranış
    DURMAKTIR (CLAUDE.md: sessiz fallback/varsayım yasak).
    """

    stats_path = tmp_path / "undeclared_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "version": 1,
                "fit_stage": "post_n4",
                "sources": {
                    "UCSF": {
                        "source": "UCSF",
                        "mean": 3880.6057028871514,
                        "std": 1842.093726334693,
                        "voxel_count": 454408128,
                        "image_count": 295,
                        "foreground_rule": "finite_nonzero",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GBMAID_ZSCORE_STATS_PATH", str(stats_path))
    image_path = _dummy_nifti(tmp_path / "image_n4.nii.gz")

    with pytest.raises(ZScoreScopeMismatchError) as excinfo:
        apply_zscore_normalization(str(image_path), "UCSF")
    assert "BEYAN ETM" in str(excinfo.value)


def test_apply_zscore_scope_guard_runs_before_reading_source_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard, (mean,std) OKUNMADAN önce çalışmalı.

    Aksi halde yanlış kapsamlı bir dosyadan istatistik okunur ve hata
    ancak sonradan (ya da hiç) fark edilir. Dosyada istenen kaynak HİÇ
    yok: eğer guard record okumadan ÖNCE çalışmazsa
    `ZScoreStatisticsNotFoundError` görürüz, `ZScoreScopeMismatchError`
    değil.
    """

    stats_path = tmp_path / "pooled_without_source.json"
    stats_path.write_text(
        json.dumps(
            {
                "version": 1,
                "fit_stage": "post_n4",
                "zscore_scope": ZSCORE_SCOPE_POOLED_ALL_MODALITIES,
                "sources": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GBMAID_ZSCORE_STATS_PATH", str(stats_path))
    image_path = _dummy_nifti(tmp_path / "image_n4.nii.gz")

    with pytest.raises(ZScoreScopeMismatchError):
        apply_zscore_normalization(str(image_path), "UPenn")


def test_apply_zscore_accepts_t1ce_scope_and_normalizes(
    ascii_scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """YEŞİL: `zscore_scope='t1ce_source'` beyan eden dosya GEÇER ve normalize eder.

    Uçtan uca: `fit_source_zscore_statistics(..., scope='t1ce_source')`
    ile fit -> beyan dosyaya yazılır -> `apply_zscore_normalization()`
    guard'dan GEÇER -> gerçek NIfTI çıktısı üretilir.

    (`ascii_scratch_dir`, bu dosyadaki 3 bilinen ITK/Türkçe-yol
    başarısızlığından kaçınmak için kullanılır -- bkz. o fixture'ın
    docstring'i.)
    """

    z, y, x = np.indices((10, 12, 14))
    foreground = ((x - 7) ** 2 + (y - 6) ** 2 + (z - 5) ** 2) < 20
    array = np.zeros(foreground.shape, dtype=np.float32)
    array[foreground] = 120.0 + 9.0 * x[foreground]
    n4_like = _write_ascii_image(ascii_scratch_dir / "upenn_n4.nii.gz", array)

    stats_path = ascii_scratch_dir / "source_stats_t1ce_c32_green.json"
    record = fit_source_zscore_statistics(
        [n4_like], "UPenn", stats_path=stats_path, scope=ZSCORE_SCOPE_T1CE_SOURCE
    )
    assert record["source"] == "UPenn"

    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert payload["zscore_scope"] == ZSCORE_SCOPE_T1CE_SOURCE

    monkeypatch.setenv("GBMAID_ZSCORE_STATS_PATH", str(stats_path))
    monkeypatch.setenv(
        "GBMAID_PROCESSED_ROOT", str(ascii_scratch_dir / "processed")
    )

    output_path = Path(apply_zscore_normalization(str(n4_like), "UPenn"))
    assert output_path.is_file()

    normalized = sitk.GetArrayFromImage(sitk.ReadImage(str(output_path)))
    values = normalized[foreground]
    assert abs(float(values.mean())) < 1e-4
    assert abs(float(values.std()) - 1.0) < 1e-3


def test_fit_source_zscore_statistics_omits_scope_when_not_given(
    ascii_scratch_dir: Path,
) -> None:
    """`scope=None` varsayılanı beyan YAZMAZ -- ve bu BİLİNÇLİ.

    `fit_source_zscore_statistics()` kendisine verilen görüntülerin
    modalitesini BİLEMEZ. Varsayılanı `'t1ce_source'` yapmak,
    tüm-modalite havuzlayan bir çağıranın (örn.
    `tools/run_harmonization_cohort.py`) çıktısını SESSİZCE ve YANLIŞ
    biçimde "T1ce" diye etiketler, guard'ı işlevsizleştirirdi.
    """

    array = np.zeros((8, 8, 8), dtype=np.float32)
    array[2:6, 2:6, 2:6] = np.arange(64, dtype=np.float32).reshape(4, 4, 4) + 50.0
    image = _write_ascii_image(ascii_scratch_dir / "noscope_n4.nii.gz", array)

    stats_path = ascii_scratch_dir / "source_stats_noscope.json"
    fit_source_zscore_statistics([image], "LUMIERE", stats_path=stats_path)

    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert "zscore_scope" not in payload


def test_fit_source_zscore_statistics_refuses_to_mix_scopes(
    ascii_scratch_dir: Path,
) -> None:
    """Aynı stats dosyasında kapsam KARIŞTIRMAK yasak (iki yönde de)."""

    array = np.zeros((8, 8, 8), dtype=np.float32)
    array[2:6, 2:6, 2:6] = np.arange(64, dtype=np.float32).reshape(4, 4, 4) + 50.0
    image = _write_ascii_image(ascii_scratch_dir / "mix_n4.nii.gz", array)

    stats_path = ascii_scratch_dir / "source_stats_mix.json"
    fit_source_zscore_statistics(
        [image], "UPenn", stats_path=stats_path, scope=ZSCORE_SCOPE_T1CE_SOURCE
    )

    # T1ce beyanlı dosyaya havuzlanmış record eklemek -> HATA
    with pytest.raises(ZScoreScopeMismatchError):
        fit_source_zscore_statistics(
            [image],
            "LUMIERE",
            stats_path=stats_path,
            scope=ZSCORE_SCOPE_POOLED_ALL_MODALITIES,
        )

    # Beyanlı dosyaya BEYANSIZ record eklemek -> HATA
    with pytest.raises(ZScoreScopeMismatchError):
        fit_source_zscore_statistics([image], "LUMIERE", stats_path=stats_path)


def test_fit_if_needed_scope_guard_fires_early_on_skipped_refit(
    ascii_scratch_dir: Path,
) -> None:
    """B3 restart-guard fit'i ATLARKEN de kapsamı doğrulamalı.

    Senaryo: operatör `--stats-path` olarak YANLIŞLIKLA havuzlanmış
    dosyayı verir. Kaynağın geçerli bir record'u vardır -> B3 guard fit'i
    ATLAR -> eskiden kapsam hiç kontrol edilmeden Pass 3'e geçilir ve hata
    ancak ilk `apply_zscore_normalization()` çağrısında çıkardı (tarama
    tarama). Artık Pass 2'de, koşu BAŞLAMADAN patlar.
    """

    stats_path = ascii_scratch_dir / "pooled_restart.json"
    stats_path.write_text(
        json.dumps(
            {
                "version": 1,
                "fit_stage": "post_n4",
                "zscore_scope": ZSCORE_SCOPE_POOLED_ALL_MODALITIES,
                "sources": {
                    "UPenn": {
                        "source": "UPenn",
                        "mean": 412.8116465211492,
                        "std": 368.12396133704146,
                        "voxel_count": 3998056517,
                        "image_count": 2684,
                        "foreground_rule": "finite_nonzero",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert has_fitted_zscore_statistics("UPenn", stats_path=stats_path) is True

    with pytest.raises(ZScoreScopeMismatchError):
        fit_source_zscore_statistics_if_needed(
            [], "UPenn", stats_path=stats_path, scope=ZSCORE_SCOPE_T1CE_SOURCE
        )
