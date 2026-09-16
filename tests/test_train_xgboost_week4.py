"""`tools/train_xgboost_week4.py` icin uctan uca sentetik testler.

KAPSAM: bu test dosyasi GERCEK DB'ye HIC baglanmaz -- `run_xgboost_week4_
pipeline()` (DB'den bagimsiz cekirdek) dogrudan sentetik bir Cox egitim
cercevesiyle cagrilir. `main()`/`_rebuild_variant_training_frame()` (DB'ye
bagimli CLI katmani) GERCEK DB ile test EDILMEZ -- nihai Cox varyanti
secilip gercek kosu yapilmadan test edilemez (bkz. script docstring'i).
⚠️ 2026-08-19 EKLENDİ (Codex 2. tur, task-mt04rec5-jjqc95, MEDIUM
duzeltmesi): `main()`'in DISPATCH mantigi (`--legacy-misaligned-pipeline`
bayraginin GERCEKTEN `run_xgboost_week4_pipeline()`'e gittigi, bayraksiz
calisirsa `run_xgboost_week4_pipeline_aligned()`'e gittigi) ARTIK test
EDILIYOR -- DB baglantisi + frame-kurma adimlari SAHTE (monkeypatch),
sadece dispatch mantigi dogrulaniyor (bkz. `test_main_dispatches_to_*`).

ASCII-safe desen: proje genelindeki konvansiyonla AYNI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import train_xgboost_week4 as week4  # noqa: E402
from pipeline.xgboost_model import (  # noqa: E402
    CoxFeatureCandidatePoolOverlapError,
    CoxFitFailureRecord,
    CoxPenalizerEscalationExhaustedError,
)


_SMALL_XGB_GRID = {
    "max_depth": [2],
    "learning_rate": [0.1],
    "n_estimators": [50],
    "min_child_weight": [1],
}


def _make_synthetic_cox_training_frame(
    n: int, *, n_radiomic_features: int = 6, seed: int = 4
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Radyomik + tek klinik (`clinical_age`) kovaryatli sentetik Cox
    egitim cercevesi -- `run_xgboost_week4_pipeline()`'in dogrudan girdisi
    (gercek `build_training_frame()` ciktisiyla AYNI sekil: patient_id
    index, feature_columns + extra_columns + survival_days + event)."""

    rng = np.random.default_rng(seed)
    feature_columns = [f"feat_{i}" for i in range(n_radiomic_features)]
    features = rng.normal(size=(n, n_radiomic_features))

    age = rng.normal(loc=55, scale=10, size=n)
    true_beta = 1.2
    age_beta = 0.02
    linear_predictor = true_beta * features[:, 0] + age_beta * (age - 55)
    baseline_hazard = 0.004
    u = rng.uniform(size=n)
    event_time = -np.log(u) / (baseline_hazard * np.exp(linear_predictor))
    censoring_time = rng.uniform(low=30.0, high=np.percentile(event_time, 85), size=n)
    duration = np.minimum(event_time, censoring_time)
    event = (event_time <= censoring_time).astype(int)

    frame = pd.DataFrame(features, columns=feature_columns)
    frame["clinical_age"] = age
    frame["survival_days"] = duration
    frame["event"] = event
    frame.index = pd.Index([f"P{i:04d}" for i in range(n)], name="patient_id")
    return frame, feature_columns, ["clinical_age"]


def test_run_xgboost_week4_pipeline_end_to_end_synthetic() -> None:
    frame, feature_columns, extra_columns = _make_synthetic_cox_training_frame(
        n=180, n_radiomic_features=6, seed=17
    )

    result = week4.run_xgboost_week4_pipeline(
        frame,
        feature_columns,
        extra_columns=extra_columns,
        clinical_standardize_columns=["clinical_age"],
        censoring_bias_continuous_columns=["clinical_age"],
        cox_outer_splits=3,
        cox_inner_splits=3,
        cox_seed=5,
        cox_n_bootstrap_stability=20,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.1, 0.3),
        xgb_outer_splits=3,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID,
        xgb_n_bootstrap_ci=100,
    )

    # 1) 12-ay hedef sayilari tutarli.
    report = result.twelve_month_report
    assert report.n_survived_yes + report.n_died_no + report.n_excluded_ambiguous_censoring == report.n_total
    assert result.n_patients_after_target_exclusion == report.n_total - report.n_excluded_ambiguous_censoring

    # 2) sansur-yanliligi ozeti uretildi (continuous_columns verildi).
    assert result.censoring_bias_summary is not None
    assert "clinical_age" in set(result.censoring_bias_summary["column"])

    # 3) Cox out-of-fold skoru BAGIMSIZ olarak da sizintisiz.
    from pipeline.xgboost_model import verify_out_of_fold_leakage_free

    verify_out_of_fold_leakage_free(result.cox_oof_result)

    # 4) XGBoost DOGRU (out-of-fold) sonucu gecerli AUC+CI uretti, tum
    #    hedef-etiketli hastalari (ambiguous DUSURULMUS) tam bir kez kapsar.
    correct = result.xgboost_result_correct
    assert 0.0 <= correct.auc_out_of_fold["auc"] <= 1.0
    assert correct.auc_out_of_fold["ci_lower"] <= correct.auc_out_of_fold["auc"] <= correct.auc_out_of_fold["ci_upper"]
    assert len(correct.out_of_fold_predictions) == result.n_patients_after_target_exclusion

    # 5) hem "yanlis" (sizintili-Cox-girdili) hem "tam in-sample" AUC de uretildi.
    assert "auc" in result.xgboost_result_leaky_cox_input.auc_out_of_fold
    assert "auc" in result.xgboost_auc_fully_in_sample

    # 6) donmus risk esikleri + risk sinifi -- her hasta low/medium/high.
    assert result.risk_thresholds.n_training_patients == len(correct.out_of_fold_predictions)
    assert set(result.risk_class_out_of_fold.unique()).issubset({"low", "medium", "high"})
    assert len(result.risk_class_out_of_fold) == len(correct.out_of_fold_predictions)


def test_run_xgboost_week4_pipeline_without_censoring_bias_columns_returns_none_summary() -> None:
    frame, feature_columns, extra_columns = _make_synthetic_cox_training_frame(
        n=150, n_radiomic_features=5, seed=23
    )

    result = week4.run_xgboost_week4_pipeline(
        frame,
        feature_columns,
        extra_columns=extra_columns,
        clinical_standardize_columns=["clinical_age"],
        cox_outer_splits=3,
        cox_inner_splits=3,
        cox_seed=5,
        cox_n_bootstrap_stability=20,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.1,),
        xgb_outer_splits=3,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID,
        xgb_n_bootstrap_ci=100,
    )
    assert result.censoring_bias_summary is None


# =====================================================================
# CLI arg-parse (DB'siz kisim)
# =====================================================================


def test_parse_args_requires_variant() -> None:
    with pytest.raises(SystemExit):
        week4.parse_args([])


def test_parse_args_rejects_unknown_variant() -> None:
    with pytest.raises(SystemExit):
        week4.parse_args(["--variant", "not_a_real_variant"])


def test_parse_args_accepts_known_variant_with_defaults() -> None:
    args = week4.parse_args(["--variant", "v1_referans"])
    assert args.variant == "v1_referans"
    assert args.cox_outer_splits == 5
    assert args.xgb_outer_splits == 5
    assert args.twelve_month_threshold_days == pytest.approx(365.0)
    assert args.expected_patient_count == 611
    assert args.expected_event_count == 585
    # ALIGNED (yeni birincil) yol VARSAYILAN -- --legacy-misaligned-pipeline
    # verilmedigi surece.
    assert args.legacy_misaligned_pipeline is False
    assert args.outer_splits == 5
    assert args.seed == 42
    # MEDIUM 5 (UCSF harici degerlendirme) -- varsayilan KAPALI, kilitli sayilar.
    assert args.evaluate_ucsf_external is False
    assert args.expected_ucsf_patient_count == 295
    assert args.expected_ucsf_event_count == 169


def test_parse_args_accepts_legacy_misaligned_pipeline_flag() -> None:
    args = week4.parse_args(["--variant", "v1_referans", "--legacy-misaligned-pipeline"])
    assert args.legacy_misaligned_pipeline is True


def test_parse_args_rejects_legacy_pipeline_with_reduce_collinearity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ş1 (Codex capraz inceleme, 2026-09-11): `--legacy-misaligned-pipeline`
    dali `reduce_collinearity`'i pipeline'a HIC ILETMEZ -- ikisi birlikte
    verilirse bayrak sessizce yok sayilirdi. Artik fail-loud argparse
    hatasi vermeli, hata mesaji NEDENINI acikca soylemeli."""

    with pytest.raises(SystemExit) as exc_info:
        week4.parse_args(
            [
                "--variant",
                "v1_referans",
                "--legacy-misaligned-pipeline",
                "--reduce-collinearity",
            ]
        )
    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "--legacy-misaligned-pipeline" in stderr
    assert "--reduce-collinearity" in stderr
    assert "sessizce" in stderr or "ILETMEZ" in stderr


def test_parse_args_allows_reduce_collinearity_without_legacy_flag() -> None:
    """Karsit durum: ALIGNED (varsayilan) yolda --reduce-collinearity TEK
    BASINA sorunsuz kabul edilmeli -- Ş1 duzeltmesi yalnizca ikisinin
    BIRLIKTE verilmesini engellemeli, --reduce-collinearity'i genel olarak
    KISITLAMAMALI."""

    args = week4.parse_args(["--variant", "v1_referans", "--reduce-collinearity"])
    assert args.reduce_collinearity is True
    assert args.legacy_misaligned_pipeline is False


# =====================================================================
# Ş2 / K16-a+b (Codex sartli onay, 2026-09-11) -- CLI: penalizer-
# eskalasyon grid'i + acik fold-atlama bayragi
# =====================================================================


def test_parse_args_defaults_for_escalation_grid_and_fold_skip() -> None:
    args = week4.parse_args(["--variant", "v1_referans"])
    assert args.cox_penalizer_escalation_grid is None
    assert args.allow_fold_skip is False


def test_parse_args_parses_escalation_grid_csv_into_float_tuple() -> None:
    args = week4.parse_args(
        [
            "--variant",
            "v1_referans",
            "--cox-penalizer-escalation-grid",
            "2.0,5.0,10.0",
        ]
    )
    assert args.cox_penalizer_escalation_grid == (2.0, 5.0, 10.0)


def test_parse_args_rejects_malformed_escalation_grid(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        week4.parse_args(
            [
                "--variant",
                "v1_referans",
                "--cox-penalizer-escalation-grid",
                "2.0,not-a-number",
            ]
        )
    assert exc_info.value.code == 2
    assert "cox-penalizer-escalation-grid" in capsys.readouterr().err


def test_parse_args_accepts_allow_fold_skip_flag() -> None:
    args = week4.parse_args(["--variant", "v1_referans", "--allow-fold-skip"])
    assert args.allow_fold_skip is True


def test_parse_args_rejects_legacy_pipeline_with_allow_fold_skip(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ş1'deki AYNI gerekce (K16-b icin): `--legacy-misaligned-pipeline`
    dali `allow_fold_skip`'i pipeline'a HIC ILETMEZ -- ikisi birlikte
    verilirse sessizce yok sayilirdi."""

    with pytest.raises(SystemExit) as exc_info:
        week4.parse_args(
            [
                "--variant",
                "v1_referans",
                "--legacy-misaligned-pipeline",
                "--allow-fold-skip",
            ]
        )
    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "--legacy-misaligned-pipeline" in stderr
    assert "--allow-fold-skip" in stderr


def test_main_threads_escalation_grid_and_fold_skip_to_aligned_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI dispatch KANITI (Ş1'deki `test_main_dispatches_to_*` desenleriyle
    AYNI): `--cox-penalizer-escalation-grid`/`--allow-fold-skip` GERCEKTEN
    `run_xgboost_week4_pipeline_aligned()`'e ULASIYOR mu?"""

    frame, feature_columns, extra_columns = _make_synthetic_cox_training_frame(
        n=30, n_radiomic_features=3, seed=1
    )
    legacy_frame = (frame, feature_columns, extra_columns, ["clinical_age"])
    aligned_frame = (frame, feature_columns, extra_columns, ["clinical_age"], None)

    _patch_main_db_and_frame_builders(monkeypatch, legacy_frame=legacy_frame, aligned_frame=aligned_frame)

    captured_kwargs: dict[str, Any] = {}

    def _fake_aligned(*args: Any, **kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        raise _StopMainEarly()

    monkeypatch.setattr(week4, "run_xgboost_week4_pipeline_aligned", _fake_aligned)

    with pytest.raises(_StopMainEarly):
        week4.main(
            [
                "--variant",
                "v1_referans",
                "--cox-penalizer-escalation-grid",
                "3.0,7.0",
                "--allow-fold-skip",
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert captured_kwargs["cox_penalizer_escalation_grid"] == (3.0, 7.0)
    assert captured_kwargs["allow_fold_skip"] is True


def test_main_threads_escalation_grid_default_none_and_fold_skip_false_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """VARSAYILAN-DAVRANIS KORUNUMU KANITI: bayraklar hic verilmezse
    `run_xgboost_week4_pipeline_aligned()` `cox_penalizer_escalation_
    grid=None`/`allow_fold_skip=False` alir (pipeline kendi VARSAYILAN
    grid'ini kullanir, eski davranış degismez)."""

    frame, feature_columns, extra_columns = _make_synthetic_cox_training_frame(
        n=30, n_radiomic_features=3, seed=1
    )
    legacy_frame = (frame, feature_columns, extra_columns, ["clinical_age"])
    aligned_frame = (frame, feature_columns, extra_columns, ["clinical_age"], None)

    _patch_main_db_and_frame_builders(monkeypatch, legacy_frame=legacy_frame, aligned_frame=aligned_frame)

    captured_kwargs: dict[str, Any] = {}

    def _fake_aligned(*args: Any, **kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        raise _StopMainEarly()

    monkeypatch.setattr(week4, "run_xgboost_week4_pipeline_aligned", _fake_aligned)

    with pytest.raises(_StopMainEarly):
        week4.main(["--variant", "v1_referans", "--output-dir", str(tmp_path)])

    assert captured_kwargs["cox_penalizer_escalation_grid"] is None
    assert captured_kwargs["allow_fold_skip"] is False


# =====================================================================
# Ş4 (Codex sartli onay, 2026-09-11): cross-fit vs dagitim Cox havuzu
# kolon-kesisim guard'i -- DB'siz, saf fonksiyon testi
# =====================================================================


def test_validate_ucsf_distribution_feature_pool_disjoint_passes_when_disjoint() -> None:
    week4._validate_ucsf_distribution_feature_pool_disjoint(
        ["feat_0", "feat_1"], ["clinical_age"]
    )  # raise ETMEMELI


def test_validate_ucsf_distribution_feature_pool_disjoint_raises_on_overlap() -> None:
    with pytest.raises(CoxFeatureCandidatePoolOverlapError, match="feat_0"):
        week4._validate_ucsf_distribution_feature_pool_disjoint(
            ["feat_0", "feat_1"], ["feat_0"]
        )


# =====================================================================
# Ş3 (Codex sartli onay, 2026-09-11): coksek bile diske dusen crash-
# manifest -- atomik yazim, HER basarisiz deneme kaydi
# =====================================================================


def _sample_failure_record(*, outer_fold: int = 0, penalizer: float = 0.05) -> CoxFitFailureRecord:
    return CoxFitFailureRecord(
        outer_fold=outer_fold,
        inner_fold=None,
        fit_type="outer_final",
        attempt_index=0,
        penalizer=penalizer,
        penalizer_escalation_grid=[2.0, 5.0, 10.0],
        candidate_feature_columns=["feat_0", "feat_1"],
        candidate_feature_columns_hash="deadbeefdeadbeef",
        error_class="ConvergenceError",
        error_message="Convergence halted due to matrix inversion problems.",
        timestamp_utc="2026-09-11T00:00:00+00:00",
    )


def test_write_crash_manifest_writes_valid_json_with_all_failure_record_fields(tmp_path: Path) -> None:
    args = week4.parse_args(["--variant", "v1_referans"])
    failure_log = [_sample_failure_record(outer_fold=0), _sample_failure_record(outer_fold=0, penalizer=2.0)]

    target = week4._write_crash_manifest(
        tmp_path,
        args,
        stage="aligned_pipeline",
        exc=CoxPenalizerEscalationExhaustedError("sentetik cokus (test)"),
        cox_fit_failure_logs={"aligned_pipeline": failure_log},
    )

    assert target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists(), (
        "atomik yazim sonrasi .tmp dosyasi KALMAMALI"
    )

    import json

    manifest = json.loads(target.read_text(encoding="utf-8"))
    assert manifest["variant"] == "v1_referans"
    assert manifest["stage"] == "aligned_pipeline"
    assert manifest["error_class"] == "CoxPenalizerEscalationExhaustedError"
    assert "sentetik cokus" in manifest["error_message"]
    assert manifest["total_failed_attempts"] == 2
    records = manifest["cox_fit_failure_logs"]["aligned_pipeline"]
    assert len(records) == 2
    first = records[0]
    assert first["outer_fold"] == 0
    assert first["fit_type"] == "outer_final"
    assert first["penalizer"] == 0.05
    assert first["penalizer_escalation_grid"] == [2.0, 5.0, 10.0]
    assert first["candidate_feature_columns"] == ["feat_0", "feat_1"]
    assert first["candidate_feature_columns_hash"] == "deadbeefdeadbeef"
    assert first["n_candidate_features"] == 2
    assert first["error_class"] == "ConvergenceError"
    assert "Convergence halted" in first["error_message"]
    assert first["timestamp_utc"] == "2026-09-11T00:00:00+00:00"


def test_write_crash_manifest_handles_empty_failure_log(tmp_path: Path) -> None:
    """Genel bir istisna (escalation ile ilgisiz) da manifest'e dusmeli --
    failure_log BOS olsa bile ("hangi asamada ne coktu" hala kayitli)."""

    args = week4.parse_args(["--variant", "v1_referans"])
    target = week4._write_crash_manifest(
        tmp_path,
        args,
        stage="ucsf_external_evaluation",
        exc=RuntimeError("baska bir hata"),
        cox_fit_failure_logs={"ucsf_external_evaluation": []},
    )

    import json

    manifest = json.loads(target.read_text(encoding="utf-8"))
    assert manifest["total_failed_attempts"] == 0
    assert manifest["cox_fit_failure_logs"]["ucsf_external_evaluation"] == []
    assert manifest["error_class"] == "RuntimeError"


def test_main_writes_crash_manifest_and_reraises_when_aligned_pipeline_crashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """UCTAN UCA KANIT -- `main()` `run_xgboost_week4_pipeline_aligned()`
    COKTUGUNDE (a) crash-manifest dosyasini GERCEKTEN yaziyor, (b)
    istisnayi YUTMADAN yeniden firlatiyor (sessizce basariya donmuyor)."""

    frame, feature_columns, extra_columns = _make_synthetic_cox_training_frame(
        n=30, n_radiomic_features=3, seed=1
    )
    legacy_frame = (frame, feature_columns, extra_columns, ["clinical_age"])
    aligned_frame = (frame, feature_columns, extra_columns, ["clinical_age"], None)

    _patch_main_db_and_frame_builders(monkeypatch, legacy_frame=legacy_frame, aligned_frame=aligned_frame)

    def _fake_aligned_crashes(*args: Any, **kwargs: Any) -> Any:
        # Gercekci senaryo: pipeline ic-fold'lardan bazi basarisiz
        # denemeler biriktirdikten SONRA coker -- bu, caller'in verdigi
        # `cox_fit_failure_log` listesine YAZAR (gercek kodun yaptigi gibi).
        failure_log = kwargs["cox_fit_failure_log"]
        failure_log.append(_sample_failure_record(outer_fold=1))
        raise CoxPenalizerEscalationExhaustedError("sentetik cokus (uctan uca test)")

    monkeypatch.setattr(week4, "run_xgboost_week4_pipeline_aligned", _fake_aligned_crashes)

    with pytest.raises(CoxPenalizerEscalationExhaustedError):
        week4.main(["--variant", "v1_referans", "--output-dir", str(tmp_path)])

    manifest_path = tmp_path / "week4_v1_referans_aligned_pipeline_crash_manifest.json"
    assert manifest_path.exists(), "main() coktukten sonra crash-manifest dosyasi OLUSMADI"

    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_failed_attempts"] == 1
    assert manifest["cox_fit_failure_logs"]["aligned_pipeline"][0]["outer_fold"] == 1


class _StopMainEarly(Exception):
    """Test-yardimcisi -- `main()`'in dogru dispatch fonksiyonuna ULASTIGINI
    kanitlamak icin, o fonksiyon GERCEKTEN cagrildigi anda `main()`'i erken
    durdurur (asagisindaki `_write_outputs()`/`_write_aligned_outputs()`
    adimlarini sahte bir sonuc nesnesiyle taklit etmeye GEREK KALMAZ)."""


def _patch_main_db_and_frame_builders(
    monkeypatch: pytest.MonkeyPatch,
    *,
    legacy_frame: tuple[Any, ...],
    aligned_frame: tuple[Any, ...],
) -> None:
    """`main()`'in DB baglantisi + frame-kurma adimlarini SAHTELER (gercek
    DB'ye HIC dokunulmaz) -- yalnizca DISPATCH (hangi orkestrasyon
    fonksiyonunun cagrildigi) test edilecek."""

    import db_connection

    class _FakeCursor:
        def close(self) -> None:
            return None

    class _FakeConnection:
        def cursor(self, cursor_factory: Any = None) -> _FakeCursor:
            return _FakeCursor()

        def close(self) -> None:
            return None

    monkeypatch.setattr(db_connection, "get_connection", lambda readonly=True: _FakeConnection())
    monkeypatch.setattr(
        week4,
        "_rebuild_variant_training_frame",
        lambda *, cursor, config, args: legacy_frame,
    )
    monkeypatch.setattr(
        week4,
        "_rebuild_variant_training_frame_for_aligned_pipeline",
        lambda *, cursor, config, args: aligned_frame,
    )


def test_main_dispatches_to_legacy_pipeline_when_flag_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MEDIUM (Codex 2. tur, task-mt04rec5-jjqc95) duzeltmesi: "CLI dispatch
    testi --legacy-misaligned-pipeline'in GERCEKTEN o yola gittigini
    dogrulasin" -- ONCEDEN `--legacy-misaligned-pipeline` SADECE `parse_
    args()` duzeyinde (`args.legacy_misaligned_pipeline is True`) test
    ediliyordu, `main()`'in bu bayrakla GERCEKTEN `run_xgboost_week4_
    pipeline()`'i (ALIGNED'i DEGIL) cagirdigi HICBIR testte DOGRULANMIYORDU.
    Bu test DB'siz (tum agir adimlar sahte) `main()`'i UCTAN UCA cagirir."""

    frame, feature_columns, extra_columns = _make_synthetic_cox_training_frame(
        n=30, n_radiomic_features=3, seed=1
    )
    legacy_frame = (frame, feature_columns, extra_columns, ["clinical_age"])
    aligned_frame = (frame, feature_columns, extra_columns, ["clinical_age"], None)

    _patch_main_db_and_frame_builders(monkeypatch, legacy_frame=legacy_frame, aligned_frame=aligned_frame)

    def _fake_legacy(*args: Any, **kwargs: Any) -> Any:
        raise _StopMainEarly()

    def _fake_aligned(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "run_xgboost_week4_pipeline_aligned() CAGRILDI -- "
            "--legacy-misaligned-pipeline verildiginde SADECE "
            "run_xgboost_week4_pipeline() (LEGACY) cagrilmali."
        )

    monkeypatch.setattr(week4, "run_xgboost_week4_pipeline", _fake_legacy)
    monkeypatch.setattr(week4, "run_xgboost_week4_pipeline_aligned", _fake_aligned)

    with pytest.raises(_StopMainEarly):
        week4.main(
            [
                "--variant",
                "v1_referans",
                "--legacy-misaligned-pipeline",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_main_dispatches_to_aligned_pipeline_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`test_main_dispatches_to_legacy_pipeline_when_flag_set()`'in AYNASI --
    bayrak VERİLMEDİĞİNDE `main()`'in `run_xgboost_week4_pipeline_aligned()`'i
    (LEGACY'yi DEGIL) cagirdigini dogrular."""

    frame, feature_columns, extra_columns = _make_synthetic_cox_training_frame(
        n=30, n_radiomic_features=3, seed=1
    )
    legacy_frame = (frame, feature_columns, extra_columns, ["clinical_age"])
    aligned_frame = (frame, feature_columns, extra_columns, ["clinical_age"], None)

    _patch_main_db_and_frame_builders(monkeypatch, legacy_frame=legacy_frame, aligned_frame=aligned_frame)

    def _fake_legacy(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "run_xgboost_week4_pipeline() (LEGACY) CAGRILDI -- "
            "--legacy-misaligned-pipeline VERILMEDIGINDE SADECE "
            "run_xgboost_week4_pipeline_aligned() cagrilmali."
        )

    def _fake_aligned(*args: Any, **kwargs: Any) -> Any:
        raise _StopMainEarly()

    monkeypatch.setattr(week4, "run_xgboost_week4_pipeline", _fake_legacy)
    monkeypatch.setattr(week4, "run_xgboost_week4_pipeline_aligned", _fake_aligned)

    with pytest.raises(_StopMainEarly):
        week4.main(["--variant", "v1_referans", "--output-dir", str(tmp_path)])


def test_run_xgboost_week4_pipeline_legacy_orchestrator_is_leaky_when_perturbed() -> None:
    """MEDIUM (Codex 2. tur, task-mt04rec5-jjqc95) duzeltmesi: "kirmizi test
    gercek legacy orkestratoru cagirmiyor" bulgusu -- `tests/test_xgboost_
    model.py::test_old_misaligned_pipeline_train_fold_cox_scores_change_
    when_a_patient_perturbed()` Cox/XGBoost adimlarini ELLE (inline) yeniden
    kuruyordu, GERCEK `run_xgboost_week4_pipeline()` orkestratorunu HIC
    CAGIRMIYORDU -- kod driftine karsi korumasizdi (inline kopya,
    `run_xgboost_week4_pipeline()`'in gercek govdesinden BAGIMSIZ yasar).

    Bu test AYNI kirmizi kaniti -- bir XGBoost dis-test hastasinin verisi
    bozulunca, XGBoost'un HICBIR sekilde test etmedigi bir fold'un EGITIM
    girdilerindeki (Cox out-of-fold skoru) hastalar da DEGISIYOR -- DOGRUDAN
    `run_xgboost_week4_pipeline()`'i cagirarak uretir."""

    frame, feature_columns, extra_columns = _make_synthetic_cox_training_frame(
        n=150, n_radiomic_features=5, seed=81
    )

    kwargs = dict(
        extra_columns=extra_columns,
        clinical_standardize_columns=["clinical_age"],
        cox_outer_splits=3,
        cox_inner_splits=3,
        cox_seed=5,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_n_bootstrap_stability=20,
        xgb_outer_splits=3,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID,
        xgb_n_bootstrap_ci=50,
    )

    baseline = week4.run_xgboost_week4_pipeline(frame, feature_columns, **kwargs)
    xgb_audit0 = baseline.xgboost_result_correct.fold_audit[0]
    xgb_train0 = frozenset(xgb_audit0["train_patient_ids"])
    xgb_test0 = frozenset(xgb_audit0["test_patient_ids"])
    baseline_scores = (
        baseline.cox_oof_result.scores.loc[sorted(xgb_train0), baseline.cox_oof_result.score_column]
        .sort_index()
    )

    # perturb edilecek hastayi, 12-ay hedefi (target_12mo_survival, bu
    # fonksiyonun ICINDE YENIDEN hesaplaniyor -- BOLUM 1 girdisi HAM
    # survival_days/event alir) survival_days x50+5000'den SONRA da AYNI
    # ("Yes"/survived) KALACAK sekilde SECIYORUZ (survival_days zaten
    # 400 gunun UZERINDE, sadece BUYUYOR) -- boylece fold yapisi (hem
    # Cox'un event-tabanli hem XGBoost'un target-tabanli disfold bolunmesi)
    # perturbasyondan ETKILENMEZ, karsilastirma anlamli kalir.
    candidates = [
        patient_id
        for patient_id in sorted(xgb_test0)
        if frame.loc[patient_id, "survival_days"] > 400.0
    ]
    assert candidates, (
        "test kurulumu: fold-0 XGBoost test kumesinde survival_days>400 "
        "hasta bulunamadi -- seed/n degistirilmeli."
    )
    perturb_id = candidates[0]

    perturbed_frame = frame.copy()
    perturbed_frame.loc[perturb_id, "survival_days"] = (
        perturbed_frame.loc[perturb_id, "survival_days"] * 50.0 + 5000.0
    )

    perturbed_result = week4.run_xgboost_week4_pipeline(perturbed_frame, feature_columns, **kwargs)
    xgb_audit0_2 = perturbed_result.xgboost_result_correct.fold_audit[0]
    xgb_train0_2 = frozenset(xgb_audit0_2["train_patient_ids"])
    xgb_test0_2 = frozenset(xgb_audit0_2["test_patient_ids"])

    # XGBoost'un KENDI fold yapisi degismedi (perturb_id secimi bunu
    # garanti eder -- target_12mo_survival AYNI kaldi).
    assert xgb_train0 == xgb_train0_2
    assert xgb_test0 == xgb_test0_2

    perturbed_scores = (
        perturbed_result.cox_oof_result.scores.loc[
            sorted(xgb_train0_2), perturbed_result.cox_oof_result.score_column
        ]
        .sort_index()
    )

    common_idx = baseline_scores.index.intersection(perturbed_scores.index)
    max_diff = (baseline_scores.loc[common_idx] - perturbed_scores.loc[common_idx]).abs().max()

    # KIRMIZI: eski/legacy orkestrator (GERCEK run_xgboost_week4_pipeline())
    # cagrisinda, XGBoost'un GORMEDIGI bir test hastasinin verisi bozulunca
    # XGBoost'un EGITIM kumesindeki Cox skorlari da DEGISIYOR -- Cox'un
    # KENDI dis-fold'u (run_nested_cv() icinde, event_col'a gore) XGBoost'un
    # KENDI dis-fold'undan (target_col'a gore) BAGIMSIZ oldugu icin.
    assert max_diff > 1e-6, (
        "GERCEK legacy orkestrator (run_xgboost_week4_pipeline()) beklenen "
        "KIRMIZI kaniti (siziniti) URETMEDI -- seed/perturb_id/n baska bir "
        "kombinasyonla denenmeli (bkz. tests/test_xgboost_model.py::"
        "test_old_misaligned_pipeline_train_fold_cox_scores_change_when_a_"
        "patient_perturbed(), AYNI kanitin ELLE-kurulmus referans versiyonu)."
    )


# =====================================================================
# CRITICAL 1 (Codex, task-mszuse9v-efm1gw) -- YENI birincil orkestrasyon:
# run_xgboost_week4_pipeline_aligned()
# =====================================================================


def test_run_xgboost_week4_pipeline_aligned_end_to_end_synthetic() -> None:
    frame, feature_columns, extra_columns = _make_synthetic_cox_training_frame(
        n=200, n_radiomic_features=6, seed=17
    )

    result = week4.run_xgboost_week4_pipeline_aligned(
        frame,
        feature_columns,
        extra_columns=extra_columns,
        clinical_standardize_columns=["clinical_age"],
        censoring_bias_continuous_columns=["clinical_age"],
        outer_splits=3,
        seed=5,
        cox_inner_splits=3,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.1, 0.3),
        cox_n_bootstrap_stability=20,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID,
        xgb_n_bootstrap_ci=100,
    )

    report = result.twelve_month_report
    assert report.n_survived_yes + report.n_died_no + report.n_excluded_ambiguous_censoring == report.n_total
    assert result.n_patients_after_target_exclusion == report.n_total - report.n_excluded_ambiguous_censoring

    assert result.censoring_bias_summary is not None
    assert "clinical_age" in set(result.censoring_bias_summary["column"])

    aligned = result.aligned_result
    assert 0.0 <= aligned.auc_out_of_fold["auc"] <= 1.0
    assert aligned.auc_out_of_fold["ci_lower"] <= aligned.auc_out_of_fold["auc"] <= aligned.auc_out_of_fold["ci_upper"]
    assert len(aligned.out_of_fold_predictions) == result.n_patients_after_target_exclusion

    # CRITICAL 1: her fold'un Cox fit hasta kumesi outer_train ile BIREBIR
    # AYNI, outer_test ile AYRIK -- run_xgboost_week4_pipeline_aligned()
    # verify_aligned_fold_leakage_free()'i KENDI ICINDE zaten cagirdi
    # (basarisiz olsaydi burada exception firlardi); burada AYRICA
    # fold_audit uzerinden dogrudan kontrol.
    for audit in aligned.fold_audit:
        assert audit.cox_fit_patient_ids == audit.outer_train_patient_ids
        assert audit.cox_fit_patient_ids.isdisjoint(audit.outer_test_patient_ids)

    assert result.risk_thresholds.n_training_patients == len(aligned.out_of_fold_predictions)
    assert set(result.risk_class_out_of_fold.unique()).issubset({"low", "medium", "high"})


def test_run_xgboost_week4_pipeline_aligned_with_fold_local_extra_column_builder() -> None:
    """HIGH 3 (Codex) -- `fold_local_extra_column_builder` verildiginde
    her outer-fold'a AYRI kolonlar EKLENDIGINI (fold-guvenli ozellik
    muhendisligi kancasinin gercekten CAGRILDIGINI) dogrular."""

    frame, feature_columns, extra_columns = _make_synthetic_cox_training_frame(
        n=180, n_radiomic_features=5, seed=23
    )

    calls: list[tuple[int, int]] = []
    noise_rng = np.random.default_rng(123)

    def _fake_builder(outer_train_raw, outer_test_raw):
        calls.append((len(outer_train_raw), len(outer_test_raw)))
        train_out = outer_train_raw.copy()
        test_out = outer_test_raw.copy()
        # gercek varyansi olan, MEVCUT radyomik ozelliklerle KOLLINEER
        # OLMAYAN bir kolon -- HIGH3'un asil ilgilendigi sey kancanin
        # CAGRILDIGI/kolonun EKLENDIGI, deger hesabinin kendisi degil
        # (bkz. ayri test_build_fold_safe_age_spline_columns_builder_
        # uses_only_train_ages()).
        train_out["fold_local_col"] = noise_rng.normal(size=len(train_out))
        test_out["fold_local_col"] = noise_rng.normal(size=len(test_out))
        return train_out, test_out, ["fold_local_col"]

    result = week4.run_xgboost_week4_pipeline_aligned(
        frame,
        feature_columns,
        extra_columns=[],
        clinical_standardize_columns=[],
        fold_local_extra_column_builder=_fake_builder,
        outer_splits=3,
        seed=5,
        cox_inner_splits=3,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_n_bootstrap_stability=20,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID,
        xgb_n_bootstrap_ci=50,
    )

    # 3. tur duzeltmesi (task-mt06mqzi-k0gb5j, HIGH -- "RCS duzenleri
    # yalniz dis-fold'dan; ic cross-fit'te yeniden fit edilmiyor"):
    # builder ARTIK HER outer-fold icin BIR KEZ (outer-level) + HER
    # ic-fold icin BIR KEZ (ic-fold-yerel, `pipeline.xgboost_model.
    # train_xgboost_with_fold_aligned_cox_scores()`'un ic cross-fit
    # dongusunde) CAGRILIYOR -- ONCEDEN (2. tur) SADECE outer-level'de
    # cagriliyordu (3 kez), ic-fold'lar outer-level'in SONUCUNU miras
    # aliyordu (sizinti). Beklenen: 3 outer-level + 3*3 ic-fold-yerel.
    assert len(calls) == 3 + 3 * 3
    for audit in result.aligned_result.fold_audit:
        assert audit.fold_local_extra_columns == ["fold_local_col"]


# =====================================================================
# HIGH 3 (Codex) -- fold-guvenli RCS yas dugumu insa kancasi
# =====================================================================


def test_build_fold_safe_age_spline_columns_builder_uses_only_train_ages() -> None:
    import train_cox_week3 as week3

    rng = np.random.default_rng(3)
    n = 60
    age = rng.normal(loc=55, scale=10, size=n)
    frame = pd.DataFrame(
        {week3.CLINICAL_AGE_COLUMN: age}, index=[f"P{i:04d}" for i in range(n)]
    )

    builder = week4._build_fold_safe_age_spline_columns_builder(raw_age_column=week3.CLINICAL_AGE_COLUMN)

    train_a, test_a = frame.iloc[:40], frame.iloc[40:]
    train_b, test_b = frame.iloc[10:50], frame.iloc[50:]

    out_train_a, out_test_a, cols_a = builder(train_a, test_a)
    out_train_b, out_test_b, cols_b = builder(train_b, test_b)

    assert cols_a == [week3.CLINICAL_AGE_RCS1_COLUMN, week3.CLINICAL_AGE_RCS2_COLUMN]
    # hasta kumesi DEGISMEDI -- SADECE kolon eklendi.
    assert set(out_train_a.index) == set(train_a.index)
    assert set(out_test_a.index) == set(test_a.index)

    # duzgunler (RCS2) FARKLI dugumlerden geldigi icin, PAYLASILAN
    # hastalarda FARKLI degerler URETMELI -- knot'lar SADECE o fold'un
    # kendi egitim yasindan geldigini KANITLAR (HIGH 3).
    shared = out_train_a.index.intersection(out_train_b.index)
    assert len(shared) > 0
    diff = (
        out_train_a.loc[shared, week3.CLINICAL_AGE_RCS2_COLUMN]
        - out_train_b.loc[shared, week3.CLINICAL_AGE_RCS2_COLUMN]
    ).abs().max()
    assert diff > 0.0

    # dogrulama: fold A'nin dugumleri GERCEKTEN train_a'nin (train_b'nin
    # DEGIL) yasindan hesaplanmis.
    knots_a = week3.compute_rcs_knots(train_a[week3.CLINICAL_AGE_COLUMN])
    expected_train_spline_a = week3.restricted_cubic_spline_basis(train_a[week3.CLINICAL_AGE_COLUMN], knots_a)
    pd.testing.assert_series_equal(
        out_train_a[week3.CLINICAL_AGE_RCS2_COLUMN],
        expected_train_spline_a[week3.CLINICAL_AGE_RCS2_COLUMN],
        check_names=False,
    )


def test_run_xgboost_week4_pipeline_aligned_rcs_knots_refit_per_inner_fold() -> None:
    """3. tur test #2/#4 (Codex, task-mt06mqzi-k0gb5j): "iç cross-fit'te
    RCS düğümleri yeniden fit edilmiyor" HIGH bulgusunun düzeltmesi --
    `_build_fold_safe_age_spline_columns_builder()`'in ürettiği kanca
    ARTIK sadece dış-fold'da değil, `pipeline.xgboost_model.train_
    xgboost_with_fold_aligned_cox_scores()`'un HER iç cross-fit
    fold'unda da (SADECE o iç-fold'un `inner_train`'inin yaşından)
    YENİDEN çağrılıyor mu -- `week3.compute_rcs_knots()`'u spy ile
    sararak, çağrı SAYISININ ve HER çağrıya giden yaş kümesinin
    beklenen dış/iç kademeyle örtüştüğünü (niyet değil, gerçek çağrı)
    doğrudan ölçer. v2b/v2c varyantları (`use_age_spline=True`) tam
    olarak bu kancayı kullanır -- bu test o üretim yolunun ENTEGRASYON
    kanıtıdır."""

    import train_cox_week3 as week3
    from unittest.mock import patch

    frame, feature_columns, _extra = _make_synthetic_cox_training_frame(
        n=180, n_radiomic_features=5, seed=31
    )

    outer_splits = 3
    cox_inner_splits = 3
    builder = week4._build_fold_safe_age_spline_columns_builder(raw_age_column="clinical_age")

    captured_age_indexes: list[frozenset] = []
    original_compute_rcs_knots = week3.compute_rcs_knots

    def _spy_compute_rcs_knots(age_series, *args, **kwargs):
        captured_age_indexes.append(frozenset(age_series.index))
        return original_compute_rcs_knots(age_series, *args, **kwargs)

    with patch.object(week3, "compute_rcs_knots", side_effect=_spy_compute_rcs_knots):
        week4.run_xgboost_week4_pipeline_aligned(
            frame,
            feature_columns,
            extra_columns=[],
            clinical_standardize_columns=[
                week3.CLINICAL_AGE_RCS1_COLUMN,
                week3.CLINICAL_AGE_RCS2_COLUMN,
            ],
            fold_local_extra_column_builder=builder,
            outer_splits=outer_splits,
            seed=5,
            cox_inner_splits=cox_inner_splits,
            cox_l1_ratio_grid=(1.0,),
            cox_penalizer_grid=(0.3,),
            cox_n_bootstrap_stability=20,
            xgb_inner_splits=3,
            xgb_seed=9,
            xgb_param_grid=_SMALL_XGB_GRID,
            xgb_n_bootstrap_ci=50,
        )

    # beklenen: 1 dis-fold-seviyesi (outer_train'in TAMAMI) + cox_inner_
    # splits ic-fold-yerel cagri, HER dis-fold icin.
    expected_calls = outer_splits * (1 + cox_inner_splits)
    assert len(captured_age_indexes) == expected_calls, (
        f"beklenen {expected_calls} compute_rcs_knots() cagrisi, GERCEKTE "
        f"{len(captured_age_indexes)} -- ic-fold-yerel RCS yeniden-fit "
        "kancasi CALISMIYOR olabilir."
    )

    calls_per_fold = 1 + cox_inner_splits
    for fold_index in range(outer_splits):
        fold_calls = captured_age_indexes[
            fold_index * calls_per_fold : (fold_index + 1) * calls_per_fold
        ]
        outer_call, inner_calls = fold_calls[0], fold_calls[1:]
        assert len(inner_calls) == cox_inner_splits
        for inner_call in inner_calls:
            # ic-fold-yerel cagri outer-level cagrinin KESIN bir ALT
            # KUMESI olmali (inner_train, outer_train'in STRICT alt
            # kumesi) -- ONCEDEN (duzeltme oncesi) ic-fold'lar HIC
            # compute_rcs_knots() CAGIRMIYORDU, outer-level'in SONUCUNU
            # (TUM outer_train'den) miras ALIYORDU -- bu da ic-test
            # hastasinin yasinin duzen tanimina SIZMASINA yol aciyordu.
            assert inner_call < outer_call, (
                "ic-fold-yerel compute_rcs_knots() cagrisi outer-level "
                "cagrinin KESIN bir alt kumesi DEGIL."
            )


# =====================================================================
# HIGH 5 (Codex 2. tur, task-mt04rec5-jjqc95) -- UCSF final/external
# yolu standardizasyon receteti tutarliligi
# =====================================================================


def test_prepare_full_pool_standardized_frames_uses_training_stats_only() -> None:
    """HIGH 5 duzeltmesi: `_prepare_full_pool_standardized_frames()`
    (1) `pipeline.cox_model.standardize_columns_fold_safe()` ile BIREBIR
    AYNI sayisal sonucu uretir (bagimsiz cift-hesaplama karsilastirmasi),
    (2) SADECE `training_frame`'in istatistigini kullanir --
    `external_frame`'in (UCSF) KENDI istatistigi hesaba KATILMAZ, (3)
    ogrenilen ortalama/std'yi (final artifact'e kaydedilmek uzere)
    dogru DONDURUR."""

    from pipeline.cox_model import standardize_columns_fold_safe

    training_frame = pd.DataFrame(
        {"clinical_age": [50.0, 60.0, 70.0, 40.0]}, index=["P1", "P2", "P3", "P4"]
    )
    # UCSF cercevesi KASITLI olarak COK FARKLI bir olcekte (ortalama
    # ~1500) -- eger fonksiyon yanlislikla external'in KENDI istatistigini
    # kullansaydi (ya da hic standardize etmeseydi) bu asagidaki
    # assertion'lar YAKALARDI.
    external_frame = pd.DataFrame({"clinical_age": [1000.0, 2000.0]}, index=["U1", "U2"])

    (
        full_pool_training_frame,
        full_pool_external_frame,
        means,
        stds,
    ) = week4._prepare_full_pool_standardized_frames(training_frame, external_frame, ["clinical_age"])

    expected_mean = training_frame["clinical_age"].mean()
    expected_std = training_frame["clinical_age"].std(ddof=0)
    assert means == pytest.approx({"clinical_age": expected_mean})
    assert stds == pytest.approx({"clinical_age": expected_std})

    reference_train, reference_external = standardize_columns_fold_safe(
        training_frame, external_frame, ["clinical_age"]
    )
    pd.testing.assert_frame_equal(full_pool_training_frame, reference_train)
    pd.testing.assert_frame_equal(full_pool_external_frame, reference_external)

    # UCSF'in KENDI olceginde standardize edilmis olsaydi degerler ~0
    # civarinda olurdu (ortalamalarina gore merkezlenmis) -- UPenn
    # istatistigiyle standardize edildigi icin BUYUK/UZAK kalmali.
    assert (full_pool_external_frame["clinical_age"].abs() > 10).all()


def test_prepare_full_pool_standardized_frames_noop_when_no_columns() -> None:
    """`clinical_standardize_columns` bos ise (no-op) frame'ler
    DEGISTIRILMEDEN doner, means/stds bos sozluk olur."""

    training_frame = pd.DataFrame({"clinical_age": [50.0, 60.0]}, index=["P1", "P2"])
    external_frame = pd.DataFrame({"clinical_age": [1000.0]}, index=["U1"])

    full_pool_training_frame, full_pool_external_frame, means, stds = (
        week4._prepare_full_pool_standardized_frames(training_frame, external_frame, [])
    )

    pd.testing.assert_frame_equal(full_pool_training_frame, training_frame)
    pd.testing.assert_frame_equal(full_pool_external_frame, external_frame)
    assert means == {}
    assert stds == {}
