"""`pipeline/xgboost_model.py` icin sentetik-veri testleri.

KAPSAM: GERCEK DB'ye HIC baglanmaz (nihai Cox varyanti henuz secilmedi,
gercek out-of-fold Cox skoru uretilemez). En kritik odak SIZINTI
GARANTISI -- CLAUDE.md: "Cox -> XGBoost arasi veri sizintisi: XGBoost'a
giden Cox skorlari out-of-fold/nested CV ile uretilmeli, asla in-sample
degil."

ASCII-safe desen: proje genelindeki (`tests/test_train_cox_week3.py`)
konvansiyonla AYNI -- sentetik string'ler ASCII.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from lifelines.exceptions import ConvergenceError

import pipeline.xgboost_model as xgboost_model_module
from pipeline.cox_model import run_nested_cv, select_features_lasso
from pipeline.reduce_collinearity import build_v3_candidate_pool
from pipeline.xgboost_model import (
    DEFAULT_XGB_PARAM_GRID,
    AlignedCoxXGBoostResult,
    AlignedXGBoostFoldAudit,
    CoxFeatureCandidatePoolOverlapError,
    CoxFitProvenanceRecord,
    FinalXGBoostPipelineResult,
    FrozenRiskThresholds,
    OutOfFoldCoxScoreResult,
    OutOfFoldFoldAudit,
    OutOfFoldLeakageError,
    OutOfFoldReconstructionMismatchError,
    XGBoostLeakageError,
    apply_twelve_month_target,
    assert_no_clinical_standardize_feature_overlap,
    assign_risk_class_column,
    build_xgboost_feature_frame,
    classify_risk,
    compute_frozen_risk_thresholds,
    define_twelve_month_survival_target,
    evaluate_xgboost_external_test,
    fit_final_cox_and_xgboost_pipeline_on_full_pool,
    generate_in_sample_cox_scores,
    generate_in_sample_xgboost_auc,
    generate_out_of_fold_cox_scores,
    summarize_excluded_censoring_bias,
    train_xgboost_nested_cv,
    train_xgboost_with_fold_aligned_cox_scores,
    verify_aligned_fold_leakage_free,
    verify_full_pool_pipeline_leakage_free,
    verify_out_of_fold_leakage_free,
)


# =====================================================================
# Sentetik veri yardimcilari
# =====================================================================


def _make_synthetic_survival_frame(
    n: int, *, n_features: int = 6, informative_index: int = 0, seed: int = 7
) -> tuple[pd.DataFrame, list[str]]:
    """`tests/test_cox_model.py::_make_synthetic_survival_frame()` ile AYNI
    desen -- exponential-baseline hazard + tek bilgilendirici ozellik.
    `patient_id` string index (gercek kullanimla tutarli)."""

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
    frame.index = pd.Index([f"P{i:04d}" for i in range(n)], name="patient_id")
    return frame, feature_columns


def _make_synthetic_classification_frame(
    n: int, *, n_features: int = 4, informative_index: int = 0, seed: int = 11
) -> tuple[pd.DataFrame, list[str]]:
    """Sentetik ikili siniflandirma verisi -- XGBoost testleri icin."""

    rng = np.random.default_rng(seed)
    feature_columns = [f"xfeat_{i}" for i in range(n_features)]
    features = rng.normal(size=(n, n_features))

    linear_predictor = 2.0 * features[:, informative_index]
    probability = 1.0 / (1.0 + np.exp(-linear_predictor))
    target = (rng.uniform(size=n) < probability).astype(float)

    frame = pd.DataFrame(features, columns=feature_columns)
    frame["target_12mo_survival"] = target
    frame.index = pd.Index([f"P{i:04d}" for i in range(n)], name="patient_id")
    return frame, feature_columns


# =====================================================================
# BOLUM 1 -- Out-of-fold Cox skoru: sizinti garantisi
# =====================================================================


def test_generate_out_of_fold_cox_scores_covers_every_patient_exactly_once() -> None:
    frame, feature_columns = _make_synthetic_survival_frame(n=150, n_features=6, seed=21)

    nested_cv_result = run_nested_cv(
        frame,
        feature_columns,
        outer_splits=3,
        inner_splits=3,
        l1_ratio_grid=(0.5, 1.0),
        penalizer_grid=(0.1, 0.5),
        stability_selection=True,
        n_bootstrap_stability=20,
        seed=7,
    )

    result = generate_out_of_fold_cox_scores(
        frame,
        feature_columns,
        nested_cv_result,
        outer_splits=3,
        seed=7,
        n_bootstrap_stability=20,
    )

    assert isinstance(result, OutOfFoldCoxScoreResult)
    assert len(result.scores) == len(frame)
    assert set(result.scores.index) == set(frame.index)
    assert not result.scores.index.duplicated().any()
    assert result.scores["fold"].nunique() == 3


def test_generate_out_of_fold_cox_scores_is_independently_leakage_free() -> None:
    frame, feature_columns = _make_synthetic_survival_frame(n=150, n_features=6, seed=22)

    nested_cv_result = run_nested_cv(
        frame,
        feature_columns,
        outer_splits=5,
        inner_splits=3,
        l1_ratio_grid=(1.0,),
        penalizer_grid=(0.1, 0.3),
        stability_selection=True,
        n_bootstrap_stability=20,
        seed=13,
    )

    result = generate_out_of_fold_cox_scores(
        frame,
        feature_columns,
        nested_cv_result,
        outer_splits=5,
        seed=13,
        n_bootstrap_stability=20,
    )

    # 1) fonksiyonun kendisi hicbir hata firlatmadan dondu -- kendi ic
    #    guard'lari zaten gecti.
    # 2) BAGIMSIZ dogrulama -- gorev talimatinin istedigi "fold
    #    atamalarini kaydet, her skorun hangi modelden geldigini izle,
    #    kesisim bos olmali" testi.
    verify_out_of_fold_leakage_free(result)

    for audit in result.fold_audit:
        assert audit.train_patient_ids.isdisjoint(audit.test_patient_ids)
        fold_scores = result.scores.loc[result.scores["fold"] == audit.fold]
        assert set(fold_scores.index) == audit.test_patient_ids
        # o fold'un skorunu ureten model, o fold'un train kumesinde
        # DEGIL, test kumesindeki hastalari tahmin etti.
        assert set(fold_scores.index).isdisjoint(audit.train_patient_ids)

    # her hastanin TAM BIR fold'da skorlandigi (birlesim = frame.index)
    all_test_ids: set = set()
    for audit in result.fold_audit:
        all_test_ids |= audit.test_patient_ids
    assert all_test_ids == set(frame.index)


def test_verify_out_of_fold_leakage_free_catches_injected_train_test_overlap() -> None:
    """`verify_out_of_fold_leakage_free()`'in KENDISI, generate_out_of_
    fold_cox_scores()'un urettigi guvenli sonuc DISINDA, KASITLI olarak
    bozulmus (train/test kesisimi olan) bir sonucu da yakalamali --
    boylece dogrulama fonksiyonunun kendisi de test edilmis olur."""

    bad_audit = OutOfFoldFoldAudit(
        fold=0,
        train_patient_ids=frozenset({"P0001", "P0002"}),
        test_patient_ids=frozenset({"P0002", "P0003"}),  # P0002 IKI KUMEDE de var
        n_train=2,
        n_test=2,
        best_penalizer=0.1,
        best_l1_ratio=1.0,
        final_features=["feat_0"],
    )
    scores = pd.DataFrame(
        {"fold": [0, 0], "cox_oof_score": [0.1, 0.2]}, index=["P0002", "P0003"]
    )
    bad_result = OutOfFoldCoxScoreResult(
        scores=scores, fold_audit=[bad_audit], outer_splits=1, seed=1, score_column="cox_oof_score"
    )

    with pytest.raises(OutOfFoldLeakageError):
        verify_out_of_fold_leakage_free(bad_result)


def test_generate_out_of_fold_cox_scores_rejects_mismatched_outer_splits() -> None:
    frame, feature_columns = _make_synthetic_survival_frame(n=120, n_features=5, seed=31)

    nested_cv_result = run_nested_cv(
        frame,
        feature_columns,
        outer_splits=3,
        inner_splits=3,
        l1_ratio_grid=(1.0,),
        penalizer_grid=(0.1,),
        stability_selection=False,
        seed=9,
    )

    # outer_splits kasitli olarak UYUSMUYOR (4 != 3) -- fold 3
    # nested_cv_result'ta hic bulunamaz.
    with pytest.raises(OutOfFoldReconstructionMismatchError):
        generate_out_of_fold_cox_scores(
            frame,
            feature_columns,
            nested_cv_result,
            outer_splits=4,
            seed=9,
            stability_selection=False,
        )


def test_generate_out_of_fold_cox_scores_rejects_corrupted_selected_features() -> None:
    frame, feature_columns = _make_synthetic_survival_frame(n=120, n_features=5, seed=32)

    nested_cv_result = run_nested_cv(
        frame,
        feature_columns,
        outer_splits=3,
        inner_splits=3,
        l1_ratio_grid=(1.0,),
        penalizer_grid=(0.1,),
        stability_selection=False,
        seed=17,
    )
    corrupted = nested_cv_result.copy()
    corrupted.loc[0, "selected_features"] = ["totally_wrong_feature_zzz"]

    with pytest.raises(OutOfFoldReconstructionMismatchError):
        generate_out_of_fold_cox_scores(
            frame,
            feature_columns,
            corrupted,
            outer_splits=3,
            seed=17,
            stability_selection=False,
        )


def test_generate_out_of_fold_cox_scores_accepts_semicolon_joined_selected_features() -> None:
    """`week3_*_fold_results.csv` diskten okunduğunda `selected_features`
    `";"`-ile-ayrilmis STRING olur (`write_variant_outputs()`), list
    DEGIL -- bu format da kabul edilmeli."""

    frame, feature_columns = _make_synthetic_survival_frame(n=120, n_features=5, seed=33)

    nested_cv_result = run_nested_cv(
        frame,
        feature_columns,
        outer_splits=3,
        inner_splits=3,
        l1_ratio_grid=(1.0,),
        penalizer_grid=(0.1,),
        stability_selection=False,
        seed=19,
    )
    csv_roundtrip = nested_cv_result.copy()
    csv_roundtrip["selected_features"] = csv_roundtrip["selected_features"].apply(
        lambda features: ";".join(sorted(features))
    )

    result = generate_out_of_fold_cox_scores(
        frame,
        feature_columns,
        csv_roundtrip,
        outer_splits=3,
        seed=19,
        stability_selection=False,
    )
    assert len(result.scores) == len(frame)


def test_generate_in_sample_cox_scores_is_optimistic_versus_out_of_fold() -> None:
    """Sizintinin GERCEK etkisini olcer -- in-sample (sizintili) Cox
    skorunun concordance'i, dogru out-of-fold nested-CV ortalamasindan
    (nested_cv_result'in kendi c_index'i) daha iyimser cikmasi BEKLENIR
    (in-sample skor kendi egitim verisini zaten gormus).

    Kucuk n (60) + cok sayida gurultu ozelligi (25) + zayif penalizer +
    stabilite seciMI KAPALI -- asiri uyumun (overfitting) GORUNUR olmasi
    icin BILEREK secildi (6 farkli seed'de ampirik olarak dogrulandi,
    fark her zaman pozitif: +0.058..+0.134). Guclu tek-sinyal/buyuk-n
    senaryolarinda (ilk denemede n=200/8-ozellik) bu fark CV gurultusu
    icinde kayboluyor -- KUCUK n/YUKSEK-boyut, sizintinin en gorunur
    oldugu, gercek dunyada da (EPV dusukken) en riskli rejim."""

    from lifelines.utils import concordance_index

    frame, feature_columns = _make_synthetic_survival_frame(n=60, n_features=25, seed=41)

    nested_cv_result = run_nested_cv(
        frame,
        feature_columns,
        outer_splits=5,
        inner_splits=3,
        l1_ratio_grid=(1.0,),
        penalizer_grid=(0.02, 0.05, 0.1),
        stability_selection=False,
        seed=5,
    )
    honest_c_index_mean = float(nested_cv_result["c_index"].mean())

    in_sample_score, _ = generate_in_sample_cox_scores(
        frame,
        feature_columns,
        penalizer=0.02,
        l1_ratio=1.0,
        stability_selection=False,
        seed=5,
    )
    in_sample_c_index = concordance_index(
        frame["survival_days"], -in_sample_score, frame["event"]
    )

    assert in_sample_c_index > honest_c_index_mean


# =====================================================================
# BOLUM 2 -- 12-ay sagkalim hedefi + sansur karari
# =====================================================================


def test_define_twelve_month_survival_target_classifies_by_censoring_rule() -> None:
    frame = pd.DataFrame(
        {
            "survival_days": [400.0, 100.0, 200.0, 900.0, 365.0],
            "event": [0, 1, 0, 1, 0],
        },
        index=["A", "B", "C", "D", "E"],
    )
    # A: 400>=365, event farketmez -> Yes (1)
    # B: 100<365, event=1 (oldu) -> No (0)
    # C: 200<365, event=0 (sansurlu, belirsiz) -> NaN, disaridan dusurulur
    # D: 900>=365, event=1 -> Yes (1) (12 aydan SONRA oldu -> yine "12 ayi gordu")
    # E: 365>=365 (esik dahil) -> Yes (1)

    target, report = define_twelve_month_survival_target(frame)

    assert target.loc["A"] == 1.0
    assert target.loc["B"] == 0.0
    assert np.isnan(target.loc["C"])
    assert target.loc["D"] == 1.0
    assert target.loc["E"] == 1.0

    assert report.n_total == 5
    assert report.n_survived_yes == 3
    assert report.n_died_no == 1
    assert report.n_excluded_ambiguous_censoring == 1
    assert report.excluded_patient_ids == ["C"]
    assert report.pct_excluded == pytest.approx(20.0)


def test_apply_twelve_month_target_drops_ambiguous_rows_only() -> None:
    frame = pd.DataFrame(
        {
            "survival_days": [400.0, 100.0, 200.0],
            "event": [0, 1, 0],
            "feat_0": [1.0, 2.0, 3.0],
        },
        index=["A", "B", "C"],
    )
    target, report = define_twelve_month_survival_target(frame)
    result = apply_twelve_month_target(frame, target)

    assert report.n_excluded_ambiguous_censoring == 1
    assert set(result.index) == {"A", "B"}
    assert result.loc["A", "target_12mo_survival"] == 1.0
    assert result.loc["B", "target_12mo_survival"] == 0.0


def test_summarize_excluded_censoring_bias_flags_significant_age_difference() -> None:
    rng = np.random.default_rng(3)
    n_retained = 100
    n_excluded = 40
    retained_age = rng.normal(loc=50, scale=5, size=n_retained)
    excluded_age = rng.normal(loc=70, scale=5, size=n_excluded)  # bariz farkli

    frame = pd.DataFrame(
        {
            "clinical_age": np.concatenate([retained_age, excluded_age]),
        }
    )
    excluded_mask = pd.Series(
        [False] * n_retained + [True] * n_excluded, index=frame.index
    )

    summary = summarize_excluded_censoring_bias(
        frame, excluded_mask, continuous_columns=["clinical_age"]
    )
    row = summary.loc[summary["column"] == "clinical_age"].iloc[0]
    assert row["p_value"] < 0.05
    assert row["excluded_mean"] > row["retained_mean"]


def test_summarize_excluded_censoring_bias_binary_column() -> None:
    frame = pd.DataFrame({"clinical_gender_male": [1, 1, 1, 1, 0, 0, 0, 0]})
    excluded_mask = pd.Series([True, True, False, False, True, True, False, False])
    summary = summarize_excluded_censoring_bias(
        frame, excluded_mask, binary_columns=["clinical_gender_male"]
    )
    row = summary.loc[summary["column"] == "clinical_gender_male"].iloc[0]
    assert row["type"] == "binary"
    assert 0.0 <= row["p_value"] <= 1.0


# =====================================================================
# BOLUM 3 -- XGBoost ozellik matrisi
# =====================================================================


def test_build_xgboost_feature_frame_merges_cox_score() -> None:
    feature_frame = pd.DataFrame(
        {"feat_0": [1.0, 2.0, 3.0]}, index=["A", "B", "C"]
    )
    cox_scores = pd.Series([0.1, 0.2, 0.3], index=["A", "B", "C"])

    combined = build_xgboost_feature_frame(cox_scores, feature_frame)
    assert list(combined.columns) == ["feat_0", "cox_oof_score"]
    assert combined.loc["B", "cox_oof_score"] == 0.2


def test_build_xgboost_feature_frame_raises_on_missing_patient_score() -> None:
    feature_frame = pd.DataFrame(
        {"feat_0": [1.0, 2.0, 3.0]}, index=["A", "B", "C"]
    )
    cox_scores = pd.Series([0.1, 0.2], index=["A", "B"])  # C eksik

    with pytest.raises(ValueError):
        build_xgboost_feature_frame(cox_scores, feature_frame)


# =====================================================================
# BOLUM 4 -- XGBoost nested-CV egitimi
# =====================================================================


_SMALL_XGB_GRID = {"max_depth": [2], "learning_rate": [0.1], "n_estimators": [50], "min_child_weight": [1]}


def test_train_xgboost_nested_cv_produces_valid_auc_with_ci() -> None:
    frame, feature_columns = _make_synthetic_classification_frame(n=150, n_features=4, seed=51)

    result = train_xgboost_nested_cv(
        frame,
        feature_columns,
        outer_splits=3,
        inner_splits=3,
        seed=5,
        param_grid=_SMALL_XGB_GRID,
        n_bootstrap_ci=200,
    )

    assert len(result.fold_results) == 3
    assert len(result.out_of_fold_predictions) == len(frame)
    assert 0.0 <= result.auc_out_of_fold["auc"] <= 1.0
    assert result.auc_out_of_fold["ci_lower"] <= result.auc_out_of_fold["auc"]
    assert result.auc_out_of_fold["auc"] <= result.auc_out_of_fold["ci_upper"]
    # informatif tek ozellik -- rastgele tahminden (0.5) belirgin iyi olmali
    assert result.auc_out_of_fold["auc"] > 0.6


def test_train_xgboost_nested_cv_is_leakage_free() -> None:
    frame, feature_columns = _make_synthetic_classification_frame(n=150, n_features=4, seed=52)

    result = train_xgboost_nested_cv(
        frame,
        feature_columns,
        outer_splits=5,
        inner_splits=3,
        seed=8,
        param_grid=_SMALL_XGB_GRID,
        n_bootstrap_ci=200,
    )

    all_test_ids: set = set()
    for audit in result.fold_audit:
        assert audit["train_patient_ids"].isdisjoint(audit["test_patient_ids"])
        all_test_ids |= audit["test_patient_ids"]
    assert all_test_ids == set(frame.index)
    assert set(result.out_of_fold_predictions.index) == set(frame.index)
    assert not result.out_of_fold_predictions.index.duplicated().any()


def test_generate_in_sample_xgboost_auc_is_more_optimistic_than_out_of_fold() -> None:
    """Gorev talimati (Ege icin): dogru (out-of-fold) AUC ile yanlis
    (in-sample) AUC arasindaki farki OLCULEBILIR kilmak. Kucuk n + zayif
    sinyal + regularizasyonsuz agaclar -- asiri uyumun (overfitting)
    gorunur olmasi icin kasitli."""

    frame, feature_columns = _make_synthetic_classification_frame(
        n=60, n_features=10, informative_index=0, seed=61
    )
    overfit_prone_params = {
        "max_depth": 6,
        "learning_rate": 0.3,
        "n_estimators": 300,
        "min_child_weight": 1,
    }

    out_of_fold = train_xgboost_nested_cv(
        frame,
        feature_columns,
        outer_splits=3,
        inner_splits=3,
        seed=3,
        param_grid={k: [v] for k, v in overfit_prone_params.items()},
        n_bootstrap_ci=200,
    )
    in_sample = generate_in_sample_xgboost_auc(
        frame,
        feature_columns,
        params=overfit_prone_params,
        seed=3,
        n_bootstrap_ci=200,
    )

    assert in_sample["auc"] > out_of_fold.auc_out_of_fold["auc"]


# =====================================================================
# BOLUM 5 -- Donmus risk sinifi esikleri
# =====================================================================


def test_compute_frozen_risk_thresholds_uses_tertile_percentiles() -> None:
    probabilities = pd.Series(np.linspace(0.0, 1.0, num=100))
    thresholds = compute_frozen_risk_thresholds(probabilities)

    assert isinstance(thresholds, FrozenRiskThresholds)
    assert thresholds.method == "tertile"
    assert thresholds.n_training_patients == 100
    assert thresholds.low_upper_bound == pytest.approx(np.percentile(probabilities, 33))
    assert thresholds.medium_upper_bound == pytest.approx(np.percentile(probabilities, 66))
    assert thresholds.low_upper_bound < thresholds.medium_upper_bound


def test_classify_risk_direction_low_probability_is_high_risk() -> None:
    thresholds = FrozenRiskThresholds(
        low_upper_bound=0.3,
        medium_upper_bound=0.6,
        method="tertile",
        n_training_patients=100,
        frozen_at="2026-08-18T00:00:00+00:00",
        source_score="xgboost_predicted_probability_out_of_fold",
    )
    assert classify_risk(0.1, thresholds) == "high"
    assert classify_risk(0.45, thresholds) == "medium"
    assert classify_risk(0.9, thresholds) == "low"
    # sinir degerleri (dahil) -- <=
    assert classify_risk(0.3, thresholds) == "high"
    assert classify_risk(0.6, thresholds) == "medium"


def test_assign_risk_class_column_matches_scalar_classify_risk() -> None:
    thresholds = FrozenRiskThresholds(
        low_upper_bound=0.3,
        medium_upper_bound=0.6,
        method="tertile",
        n_training_patients=100,
        frozen_at="2026-08-18T00:00:00+00:00",
        source_score="xgboost_predicted_probability_out_of_fold",
    )
    probabilities = pd.Series([0.1, 0.45, 0.9], index=["A", "B", "C"])
    classes = assign_risk_class_column(probabilities, thresholds)
    assert classes.loc["A"] == "high"
    assert classes.loc["B"] == "medium"
    assert classes.loc["C"] == "low"


# =====================================================================
# BOLUM 6 -- CRITICAL 1 duzeltmesi (Codex capraz inceleme, task-
# mszuse9v-efm1gw): Cox+XGBoost'un TEK, ORTAK dis-fold dongusu
# =====================================================================


def _make_synthetic_gentle_hazard_frame(
    n: int, *, n_features: int = 6, seed: int = 71
) -> tuple[pd.DataFrame, list[str]]:
    """`tests/test_train_xgboost_week4.py::_make_synthetic_cox_training_
    frame()` ile AYNI (yumusak) hazard deseni -- `_make_synthetic_
    survival_frame()`'in (bu dosyanin BOLUM 1 testleri icin kalibre
    edilmis, cok yuksek olay orani ureten) baseline_hazard'i BOLUM 6-8
    testlerinde 12-ay hedefinin HER IKI sinifini da (Yes/No) makul
    oranda uretmiyor -- bu yuzden AYRI, daha yumusak bir hazard ile
    uretiliyor. HAM (12-ay hedefi HENUZ eklenmemis) cerceve doner."""

    rng = np.random.default_rng(seed)
    feature_columns = [f"feat_{i}" for i in range(n_features)]
    features = rng.normal(size=(n, n_features))

    true_beta = 1.2
    linear_predictor = true_beta * features[:, 0]
    baseline_hazard = 0.004
    u = rng.uniform(size=n)
    event_time = -np.log(u) / (baseline_hazard * np.exp(linear_predictor))
    censoring_time = rng.uniform(low=30.0, high=np.percentile(event_time, 85), size=n)
    duration = np.minimum(event_time, censoring_time)
    event = (event_time <= censoring_time).astype(int)

    frame = pd.DataFrame(features, columns=feature_columns)
    frame["survival_days"] = duration
    frame["event"] = event
    frame.index = pd.Index([f"P{i:04d}" for i in range(n)], name="patient_id")
    return frame, feature_columns


def _make_synthetic_labeled_frame(
    n: int, *, n_features: int = 6, seed: int = 71
) -> tuple[pd.DataFrame, list[str]]:
    """`_make_synthetic_gentle_hazard_frame()` + 12-ay hedef etiketlemesi
    (`define_twelve_month_survival_target()` + `apply_twelve_month_
    target()`) -- `duration_col`/`event_col`/`target_col` AYNI ANDA
    icerir, `train_xgboost_with_fold_aligned_cox_scores()`'in dogrudan
    girdisi."""

    frame, feature_columns = _make_synthetic_gentle_hazard_frame(n, n_features=n_features, seed=seed)
    target, report = define_twelve_month_survival_target(frame)
    labeled = apply_twelve_month_target(frame, target, report=report)
    return labeled, feature_columns


def _make_synthetic_labeled_frame_with_collinear_radiomics(
    n: int, *, seed: int = 71
) -> tuple[pd.DataFrame, list[str]]:
    """BOLUM 6C (v3 recete-sadakati) icin -- `_make_synthetic_gentle_
    hazard_frame()`'in PROVEN sinif dengesini (BOLUM 6/6B'nin TUM
    testleri bunu kullanir, `train_xgboost_with_fold_aligned_cox_
    scores()`'un ic-CV grid taramasinda 'gecerli AUC yok' hatasi
    VERMEDIGI bilinen tek jeneratif desen) KORUR -- SADECE near-constant
    + tam-kolineer EK kolonlar ekler (survival_days/event/hedef
    etiketlemeyi ETKILEMEZ, cunku bu kolonlar hicbir yerde kullanilmaz).
    `build_v3_candidate_pool()`'un GERCEKTEN bir seyi elemesi + gercek
    sinyal tasiyan kolonlari (feat_0..feat_5) SECMESI icin."""

    frame, feature_columns = _make_synthetic_gentle_hazard_frame(n, n_features=6, seed=seed)
    frame = frame.copy()
    rng = np.random.default_rng(seed + 1000)
    for i in range(3):
        frame[f"feat_near_constant_{i}"] = np.full(n, 5.0 + i) + rng.normal(scale=1e-9, size=n)
        feature_columns.append(f"feat_near_constant_{i}")
    for i in range(3):
        base = rng.normal(size=n)
        frame[f"feat_dup_a_{i}"] = base
        frame[f"feat_dup_b_{i}"] = base.copy()
        feature_columns += [f"feat_dup_a_{i}", f"feat_dup_b_{i}"]
    target, report = define_twelve_month_survival_target(frame)
    labeled = apply_twelve_month_target(frame, target, report=report)
    return labeled, feature_columns


_SMALL_XGB_GRID_ALIGNED = {
    "max_depth": [2],
    "learning_rate": [0.1],
    "n_estimators": [50],
    "min_child_weight": [1],
}


def test_train_xgboost_with_fold_aligned_cox_scores_covers_every_patient_exactly_once() -> None:
    labeled, feature_columns = _make_synthetic_labeled_frame(n=180, n_features=6, seed=71)

    result = train_xgboost_with_fold_aligned_cox_scores(
        labeled,
        feature_columns,
        outer_splits=3,
        seed=5,
        cox_inner_splits=3,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_n_bootstrap_stability=20,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
        xgb_n_bootstrap_ci=50,
    )

    assert isinstance(result, AlignedCoxXGBoostResult)
    assert len(result.out_of_fold_predictions) == len(labeled)
    assert set(result.out_of_fold_predictions.index) == set(labeled.index)
    assert not result.out_of_fold_predictions.index.duplicated().any()
    assert 0.0 <= result.auc_out_of_fold["auc"] <= 1.0


def test_train_xgboost_with_fold_aligned_cox_scores_is_structurally_leakage_free() -> None:
    """CRITICAL 1 (Codex): "Hasta kimlikleri uzerinden Cox fit
    kumelerinin XGBoost test kumesiyle AYRIK oldugu YAPISAL olarak
    dogrulanmali" -- fold_audit uzerinden dogrudan kontrol."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=180, n_features=6, seed=72)

    result = train_xgboost_with_fold_aligned_cox_scores(
        labeled,
        feature_columns,
        outer_splits=4,
        seed=11,
        cox_inner_splits=3,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_n_bootstrap_stability=20,
        xgb_inner_splits=3,
        xgb_seed=13,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
        xgb_n_bootstrap_ci=50,
    )

    all_test_ids: set = set()
    for audit in result.fold_audit:
        assert isinstance(audit, AlignedXGBoostFoldAudit)
        # 1) outer train/test ayrik.
        assert audit.outer_train_patient_ids.isdisjoint(audit.outer_test_patient_ids)
        # 2) bu fold'daki HER Cox fit'inin (ic-CV + elastic-net + stabilite +
        #    ic cross-fit + final fit) girdisi outer_train ile BIREBIR AYNI --
        #    outer_test'ten HICBIR ZAMAN veri gormedi.
        assert audit.cox_fit_patient_ids == audit.outer_train_patient_ids
        assert audit.cox_fit_patient_ids.isdisjoint(audit.outer_test_patient_ids)
        all_test_ids |= audit.outer_test_patient_ids
    assert all_test_ids == set(labeled.index)

    # BAGIMSIZ ikinci dogrulama yolu (HIGH 2).
    verify_aligned_fold_leakage_free(result)


def test_verify_aligned_fold_leakage_free_catches_injected_test_patient_in_cox_fit_set() -> None:
    """`verify_aligned_fold_leakage_free()`'in KENDISI, KASITLI olarak
    bozulmus (bir XGBoost test hastasi Cox fit kumesinde de bulunan) bir
    sonucu yakalamali (HIGH 2)."""

    bad_audit = AlignedXGBoostFoldAudit(
        fold=0,
        outer_train_patient_ids=frozenset({"P0001", "P0002"}),
        outer_test_patient_ids=frozenset({"P0003", "P0004"}),
        cox_fit_patient_ids=frozenset({"P0001", "P0002", "P0003"}),  # P0003 test'te de SIZMIS
        best_cox_penalizer=0.3,
        best_cox_l1_ratio=1.0,
        cox_final_features=["feat_0"],
        n_cox_inner_splits=3,
        best_xgb_params={"max_depth": 2},
    )
    cox_score_audit = pd.DataFrame(
        {
            "fold": [0, 0, 0, 0, 0],
            "cox_oof_score": [0.1, 0.2, 0.3, 0.4, 0.5],
            "role": [
                "outer_train_cross_fit",
                "outer_train_cross_fit",
                "outer_train_cross_fit",
                "outer_test_honest",
                "outer_test_honest",
            ],
        },
        index=["P0001", "P0002", "P0003", "P0003", "P0004"],
    )
    out_of_fold_predictions = pd.DataFrame(
        {"fold": [0, 0], "predicted_probability": [0.3, 0.7], "true_label": [0.0, 1.0]},
        index=["P0003", "P0004"],
    )
    bad_result = AlignedCoxXGBoostResult(
        fold_results=pd.DataFrame([{"fold": 0}]),
        out_of_fold_predictions=out_of_fold_predictions,
        cox_score_audit=cox_score_audit,
        auc_out_of_fold={"auc": 0.5},
        fold_audit=[bad_audit],
    )

    with pytest.raises(OutOfFoldLeakageError):
        verify_aligned_fold_leakage_free(bad_result)


def test_verify_aligned_fold_leakage_free_catches_leak_visible_only_in_fit_provenance() -> None:
    """MEDIUM (Codex 2. tur, task-mt04rec5-jjqc95): "doğrulayıcı aynı
    sentetik audit'i okuyor" bulgusunun duzeltmesi -- `audit.cox_fit_
    patient_ids` KENDISI temiz (outer_train ile birebir ayni) olsa BILE,
    `fit_provenance`'taki (gercek `.fit()` cagrisinin gordugu index'i
    yansitan) KAYITLARDAN biri outer_test'ten bir hasta ICERIYORSA
    `verify_aligned_fold_leakage_free()` bunu YAKALAMALI -- bu, doğrulayıcının
    GERÇEKTEN `fit_provenance`'ı okudugunu (sadece `audit.cox_fit_patient_
    ids`'e guvenmedigini) kanitlar."""

    clean_audit = AlignedXGBoostFoldAudit(
        fold=0,
        outer_train_patient_ids=frozenset({"P0001", "P0002"}),
        outer_test_patient_ids=frozenset({"P0003", "P0004"}),
        cox_fit_patient_ids=frozenset({"P0001", "P0002"}),  # KENDISI TEMIZ
        best_cox_penalizer=0.3,
        best_cox_l1_ratio=1.0,
        cox_final_features=["feat_0"],
        n_cox_inner_splits=1,
        best_xgb_params={"max_depth": 2},
    )
    # AMA gercek fit_provenance kaydi (ic-fold) P0003'u (outer_test'ten)
    # de FIT KUMESINE almis -- SESSIZCE gecilmemesi gereken sizinti.
    leaked_fit_provenance = [
        CoxFitProvenanceRecord(
            outer_fold=0,
            fit_type="outer_final",
            inner_fold=None,
            patient_ids=frozenset({"P0001", "P0002"}),
            best_penalizer=0.3,
            best_l1_ratio=1.0,
            n_final_features=1,
        ),
        CoxFitProvenanceRecord(
            outer_fold=0,
            fit_type="cross_fit_final",
            inner_fold=0,
            patient_ids=frozenset({"P0001", "P0003"}),  # SIZINTI -- P0003
            best_penalizer=0.3,
            best_l1_ratio=1.0,
            n_final_features=1,
        ),
    ]
    cox_score_audit = pd.DataFrame(
        {
            "fold": [0, 0, 0, 0],
            "cox_oof_score": [0.1, 0.2, 0.3, 0.4],
            "role": [
                "outer_train_cross_fit",
                "outer_train_cross_fit",
                "outer_test_honest",
                "outer_test_honest",
            ],
        },
        index=["P0001", "P0002", "P0003", "P0004"],
    )
    out_of_fold_predictions = pd.DataFrame(
        {"fold": [0, 0], "predicted_probability": [0.3, 0.7], "true_label": [0.0, 1.0]},
        index=["P0003", "P0004"],
    )
    bad_result = AlignedCoxXGBoostResult(
        fold_results=pd.DataFrame([{"fold": 0}]),
        out_of_fold_predictions=out_of_fold_predictions,
        cox_score_audit=cox_score_audit,
        auc_out_of_fold={"auc": 0.5},
        fold_audit=[clean_audit],
        fit_provenance=leaked_fit_provenance,
    )

    with pytest.raises(OutOfFoldLeakageError):
        verify_aligned_fold_leakage_free(bad_result)


def test_verify_aligned_fold_leakage_free_rejects_missing_fit_provenance() -> None:
    """`fit_provenance` BOŞ (örn. eski/uyumsuz bir `AlignedCoxXGBoostResult`)
    ise `verify_aligned_fold_leakage_free()` SESSİZCE geçmemeli."""

    clean_audit = AlignedXGBoostFoldAudit(
        fold=0,
        outer_train_patient_ids=frozenset({"P0001", "P0002"}),
        outer_test_patient_ids=frozenset({"P0003", "P0004"}),
        cox_fit_patient_ids=frozenset({"P0001", "P0002"}),
        best_cox_penalizer=0.3,
        best_cox_l1_ratio=1.0,
        cox_final_features=["feat_0"],
        n_cox_inner_splits=1,
        best_xgb_params={"max_depth": 2},
    )
    cox_score_audit = pd.DataFrame(
        {
            "fold": [0, 0, 0, 0],
            "cox_oof_score": [0.1, 0.2, 0.3, 0.4],
            "role": [
                "outer_train_cross_fit",
                "outer_train_cross_fit",
                "outer_test_honest",
                "outer_test_honest",
            ],
        },
        index=["P0001", "P0002", "P0003", "P0004"],
    )
    out_of_fold_predictions = pd.DataFrame(
        {"fold": [0, 0], "predicted_probability": [0.3, 0.7], "true_label": [0.0, 1.0]},
        index=["P0003", "P0004"],
    )
    result_without_provenance = AlignedCoxXGBoostResult(
        fold_results=pd.DataFrame([{"fold": 0}]),
        out_of_fold_predictions=out_of_fold_predictions,
        cox_score_audit=cox_score_audit,
        auc_out_of_fold={"auc": 0.5},
        fold_audit=[clean_audit],
        fit_provenance=[],
    )

    with pytest.raises(OutOfFoldLeakageError):
        verify_aligned_fold_leakage_free(result_without_provenance)


def test_train_xgboost_with_fold_aligned_cox_scores_fit_provenance_matches_real_lifelines_fit_calls_via_spy() -> None:
    """HIGH 2 + MEDIUM (Codex 2. tur, task-mt04rec5-jjqc95): "testte Cox
    fitter spy/wrapper ile sarılıp gerçek .fit() index'leri yakalanarak
    bağımsızlık kanıtlansın" talebinin karşılığı -- `fit_provenance`'taki
    HER kaydın RAPORLADIĞI hasta kümesinin GERÇEKTEN lifelines
    `CoxPHFitter.fit()`'e giden dataframe'in index'iyle AYNI olduğunu,
    kodun KENDİ beyanından BAĞIMSIZ bir yoldan (gerçek `.fit()` çağrılarını
    spy ile yakalayarak) kanıtlar -- provenance kaydı "sentetik/niyet"
    DEĞİL, gerçek fit çağrısının bir YANSIMASIdır."""

    from lifelines import CoxPHFitter

    labeled, feature_columns = _make_synthetic_labeled_frame(n=150, n_features=5, seed=81)
    kwargs = dict(
        outer_splits=3,
        seed=5,
        cox_inner_splits=3,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_n_bootstrap_stability=20,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
        xgb_n_bootstrap_ci=50,
    )

    captured_fit_indexes: list[frozenset] = []
    original_fit = CoxPHFitter.fit

    def _spy_fit(self, df, *args, **kw):
        captured_fit_indexes.append(frozenset(df.index))
        return original_fit(self, df, *args, **kw)

    with patch.object(CoxPHFitter, "fit", _spy_fit):
        result = train_xgboost_with_fold_aligned_cox_scores(labeled, feature_columns, **kwargs)

    assert captured_fit_indexes, "spy hicbir CoxPHFitter.fit() cagrisi yakalamadi"
    assert result.fit_provenance, "sonuc.fit_provenance BOS -- HIGH 2 alani doldurulmamis"

    for record in result.fit_provenance:
        assert isinstance(record, CoxFitProvenanceRecord)
        assert record.patient_ids in captured_fit_indexes, (
            f"provenance kaydi (outer_fold={record.outer_fold}, "
            f"fit_type={record.fit_type!r}, inner_fold={record.inner_fold}) "
            "GERCEK bir lifelines .fit() cagrisiyla ESLESMIYOR -- provenance "
            "GUVENILIR DEGIL (kodun kendi beyanindan BAGIMSIZ dogrulanamadi)."
        )


def test_train_xgboost_with_fold_aligned_cox_scores_provenance_bijection_scope_documented() -> None:
    """3. tur test #1 (Codex, task-mt06mqzi-k0gb5j): "spy ile TÜM Cox
    .fit() çağrıları <-> provenance kayıtları İKİ YÖNLÜ bijeksiyon"
    talebinin, bu turun KAPSAM KISITI altında (görev talimatı
    `pipeline/cox_model.py`'de SADECE dar bir except-daraltmasına izin
    verdi -- grid-search/bootstrap-stabilite `.fit()` çağrılarına
    provenance enstrümantasyonu EKLENEMEDİ) ULAŞILABİLEN EN GÜÇLÜ hâli:

      1. HER provenance kaydı GERÇEKTEN spy'in yakaladığı bir `.fit()`
         çağrısıyla (index kümesi TAM olarak) eşleşir (var olan spy
         testiyle AYNI yön, burada TEKRAR-doğrulanır).
      2. Provenance kaydı sayısı TAM olarak `outer_splits + outer_splits
         * cox_inner_splits`'tir -- enstrümante EDİLEBİLEN kısmın
         (`outer_final` + `cross_fit_final`) KENDİ İÇİNDE eksiksiz/doğru
         olduğunun yapısal kanıtı.
      3. spy'in yakaladığı TOPLAM `.fit()` çağrı sayısı, provenance
         kayıt sayısından KESİN olarak FAZLADIR -- çünkü hiperparametre
         grid taraması (`cox_model._score_hyperparameters_via_inner_
         cv()`) VE bootstrap stabilite seçimi (`cox_model._bootstrap_
         stability_selection()`) KENDİ `.fit()` çağrılarını yapar, bunlar
         bu turun kapsam kısıtı gereği provenance'a EKLENMEDİ. Bu
         EŞİTSİZLİK (TAM bijeksiyon DEĞİL) BİLİNÇLİ ve BEKLENEN --
         sessizce varsayılmıyor, AÇIKÇA doğrulanıyor. TAM bijeksiyon
         SADECE `cox_model.py`'ye provenance enstrümantasyonu
         eklenirse mümkündür (bu turun kapsamı DIŞINDA -- görev
         talimatı 'pipeline/cox_model.py'de YALNIZ except daraltması'
         dedi)."""

    from lifelines import CoxPHFitter

    labeled, feature_columns = _make_synthetic_labeled_frame(n=150, n_features=5, seed=151)
    outer_splits = 3
    cox_inner_splits = 3
    kwargs = dict(
        outer_splits=outer_splits,
        seed=5,
        cox_inner_splits=cox_inner_splits,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_n_bootstrap_stability=20,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
        xgb_n_bootstrap_ci=50,
    )

    captured_fit_indexes: list[frozenset] = []
    original_fit = CoxPHFitter.fit

    def _spy_fit(self, df, *args, **kw):
        captured_fit_indexes.append(frozenset(df.index))
        return original_fit(self, df, *args, **kw)

    with patch.object(CoxPHFitter, "fit", _spy_fit):
        result = train_xgboost_with_fold_aligned_cox_scores(labeled, feature_columns, **kwargs)

    expected_provenance_count = outer_splits + outer_splits * cox_inner_splits
    assert len(result.fit_provenance) == expected_provenance_count

    for record in result.fit_provenance:
        assert record.patient_ids in captured_fit_indexes, (
            f"provenance kaydi (fit_type={record.fit_type!r}, "
            f"outer_fold={record.outer_fold}, inner_fold={record.inner_fold}) "
            "GERCEK bir spy yakalamasiyla ESLESMIYOR."
        )

    assert len(captured_fit_indexes) > len(result.fit_provenance), (
        "beklenmedik: toplam spy yakalamasi provenance sayisina esit/kucuk -- "
        "grid-search/bootstrap-stabilite .fit() cagrilarinin VAR OLDUGU "
        "varsayimi (KAPSAM DISI kalan kisim) burada DOGRULANAMADI."
    )


def test_aligned_pipeline_inner_cross_fit_selection_strictly_narrower_than_outer_train() -> None:
    """HIGH 1 (Codex 2. tur, task-mt04rec5-jjqc95) YAPISAL kanit: her
    ic-fold'un `fit_provenance` kaydindaki hasta kumesi outer_train'in
    TAMAMI DEGIL, o ic-fold'un GERCEK `inner_train` alt kumesi (KESIN/
    STRICT alt kume) olmali. Duzeltme ONCESI hiperparametre+ozellik
    SECIMI outer_train'in TAMAMINDA yapilip ic-fold'larda YENIDEN
    KULLANILIYORDU (fit'in KENDISI zaten inner_train'e daraliyordu, ama
    SECIM outer_train genisliginde kaliyordu) -- bu test dogrudan
    provenance'taki hasta kumesinin BUYUKLUGUNU/ICERIGINI kontrol ederek
    SECIMIN de artik inner_train'e daraldigini kanitlar."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=180, n_features=6, seed=71)

    result = train_xgboost_with_fold_aligned_cox_scores(
        labeled,
        feature_columns,
        outer_splits=3,
        seed=5,
        cox_inner_splits=3,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_n_bootstrap_stability=20,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
        xgb_n_bootstrap_ci=50,
    )

    for audit in result.fold_audit:
        fold_records = [record for record in result.fit_provenance if record.outer_fold == audit.fold]
        outer_final_records = [record for record in fold_records if record.fit_type == "outer_final"]
        inner_records = [record for record in fold_records if record.fit_type == "cross_fit_final"]

        assert len(outer_final_records) == 1
        assert outer_final_records[0].patient_ids == audit.outer_train_patient_ids
        assert len(inner_records) == audit.n_cox_inner_splits

        for inner_record in inner_records:
            # HER ic-fold'un fit/secim kumesi outer_train'in KESIN bir
            # ALT KUMESI (esit DEGIL) -- eger secim hala outer_train'in
            # TAMAMINI kullansaydi bu STRICT subset olmazdi.
            assert inner_record.patient_ids < audit.outer_train_patient_ids
            assert inner_record.patient_ids.isdisjoint(audit.outer_test_patient_ids)


def test_aligned_pipeline_inner_cross_fit_score_unaffected_by_own_survival_data_perturbation() -> None:
    """HIGH 1 (Codex 2. tur, task-mt04rec5-jjqc95) SAYISAL kirmizi/yesil
    kanit: bir ic-test hastasinin KENDI verisini (survival_days) SERTCE
    bozunca, O HASTANIN KENDI ic cross-fit Cox skoru DEGISMEMELI --
    kendi skorunu ureten model (o ic-fold'un inner_train'i) onu ZATEN
    hic gormuyordu (CRITICAL 1, 1. tur). DUZELTME ONCESI (HIGH 1) bu
    hastanin verisi outer_train_std'nin bir parcasi olarak hiperparametre
    aramasina + elastic-net'e + bootstrap stabilite secimine (TUMU
    outer_train_std uzerinde calisiyordu) GIRIYORDU -- bu da o secimle
    kurulan modelin (dolayisiyla bu hastanin skorunun) DOLAYLI olarak
    DEGISMESINE yol acabiliyordu (fit'in KENDISI hala inner_train'e
    dar olsa BILE, HANGI kolonlarla/hangi penalizer ile fit edildigi
    degisiyordu). DUZELTME SONRASI bu kanal TAMAMEN KAPALI: secim de
    SADECE inner_train'den yapiliyor, bu hastanin verisi HICBIR ADIMA
    (ne secime ne fit'e) GIRMIYOR -- skor BIT-BIREBIR AYNI KALMALI."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=150, n_features=5, seed=81)
    kwargs = dict(
        outer_splits=3,
        seed=5,
        cox_inner_splits=3,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_n_bootstrap_stability=20,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
        xgb_n_bootstrap_ci=50,
    )

    baseline = train_xgboost_with_fold_aligned_cox_scores(labeled, feature_columns, **kwargs)
    fold0_scores = baseline.cox_score_audit.loc[
        (baseline.cox_score_audit["fold"] == 0)
        & (baseline.cox_score_audit["role"] == "outer_train_cross_fit")
    ]
    target_patient = fold0_scores.index[0]
    baseline_own_score = float(fold0_scores.loc[target_patient, "cox_oof_score"])

    # `train_xgboost_with_fold_aligned_cox_scores()` `target_col`'u HAZIR
    # ALIR (KENDI ICINDE yeniden hesaplamaz) -- yani `survival_days`'i
    # degistirmek fold yapisini (outer VE ic-fold, ikisi de target_col'a
    # gore stratified) ETKILEMEZ, karsilastirma her zaman anlamli kalir.
    perturbed = labeled.copy()
    perturbed.loc[target_patient, "survival_days"] = (
        perturbed.loc[target_patient, "survival_days"] * 50.0 + 5000.0
    )

    result_perturbed = train_xgboost_with_fold_aligned_cox_scores(perturbed, feature_columns, **kwargs)
    fold0_scores_p = result_perturbed.cox_score_audit.loc[
        (result_perturbed.cox_score_audit["fold"] == 0)
        & (result_perturbed.cox_score_audit["role"] == "outer_train_cross_fit")
    ]
    assert target_patient in fold0_scores_p.index
    perturbed_own_score = float(fold0_scores_p.loc[target_patient, "cox_oof_score"])

    assert perturbed_own_score == pytest.approx(baseline_own_score, abs=1e-9), (
        "HIGH 1 REGRESYONU: bir ic-test hastasinin KENDI survival_days'i "
        "degistirilince KENDI ic cross-fit Cox skoru da DEGISTI -- bu "
        "hastanin verisi (dogrudan fit'e degil ama SECIM/hiperparametre "
        "araciligiyla) kendi skorunu ureten modele SIZMIS olabilir."
    )


def test_aligned_pipeline_train_fold_cox_scores_unaffected_by_test_fold_perturbation() -> None:
    """NEGATIF KONTROL (Codex HIGH 2): "bir dis-test hastasinin sagkalim
    verisini boz -> ayni fold'un egitim girdileri degismemeli". Bu,
    CRITICAL 1 duzeltmesinin (fold-hizali Cox+XGBoost) GERCEKTEN
    sizintisiz oldugunun -- bir XGBoost dis-test hastasinin verisinin
    AYNI fold'un HICBIR egitim-girdisini (Cox meta-skoru dahil)
    ETKILEMEDIGININ -- KANITIDIR."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=150, n_features=5, seed=81)

    kwargs = dict(
        outer_splits=3,
        seed=5,
        cox_inner_splits=3,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_n_bootstrap_stability=20,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
        xgb_n_bootstrap_ci=50,
    )

    baseline = train_xgboost_with_fold_aligned_cox_scores(labeled, feature_columns, **kwargs)
    fold0 = baseline.fold_audit[0]
    baseline_train_scores = (
        baseline.cox_score_audit.loc[
            (baseline.cox_score_audit["fold"] == 0)
            & (baseline.cox_score_audit["role"] == "outer_train_cross_fit")
        ]["cox_oof_score"]
        .sort_index()
    )

    # bir dis-test hastasini SERTCE boz (SADECE survival_days x50 + 5000 --
    # 2026-08-19 Codex 2. tur MEDIUM duzeltmesi: bu yorum ONCEDEN "event
    # flip + sure x50" diyordu ama kod HICBIR ZAMAN event'i degistirmiyor,
    # SADECE survival_days'i olceklendiriyor -- yorum kodla UYUSMUYORDU,
    # duzeltildi) -- ama 12-ay hedefini (target_12mo_survival) DEGISTIRMEYECEK
    # sekilde (fold bolunmesi target_col'a gore stratified, degisirse fold
    # yapisi da degisir ve karsilastirma anlamsizlasir).
    perturb_id = sorted(fold0.outer_test_patient_ids)[0]
    perturbed = labeled.copy()
    perturbed.loc[perturb_id, "survival_days"] = perturbed.loc[perturb_id, "survival_days"] * 50.0 + 5000.0

    result_perturbed = train_xgboost_with_fold_aligned_cox_scores(perturbed, feature_columns, **kwargs)
    fold0_perturbed = result_perturbed.fold_audit[0]

    # fold yapisi DEGISMEMELI (StratifiedKFold sadece target_col'a bakar).
    assert fold0_perturbed.outer_train_patient_ids == fold0.outer_train_patient_ids
    assert fold0_perturbed.outer_test_patient_ids == fold0.outer_test_patient_ids

    perturbed_train_scores = (
        result_perturbed.cox_score_audit.loc[
            (result_perturbed.cox_score_audit["fold"] == 0)
            & (result_perturbed.cox_score_audit["role"] == "outer_train_cross_fit")
        ]["cox_oof_score"]
        .sort_index()
    )

    # YESIL: fold-0'in EGITIM kumesindeki Cox meta-skorlari, fold-0'in
    # TEST kumesindeki bir hastanin verisi bozulduktan SONRA da AYNEN
    # (bit-birebir) kaliyor -- o test hastasi bu fold'daki HICBIR Cox
    # fit'inin girdisinde YOK.
    pd.testing.assert_series_equal(baseline_train_scores, perturbed_train_scores, check_names=False)


def test_old_misaligned_pipeline_train_fold_cox_scores_change_when_a_patient_perturbed() -> None:
    """KIRMIZI/referans kanit (Codex CRITICAL 1 bulgusunun KENDISI): ESKI
    yol (`generate_out_of_fold_cox_scores()` + `train_xgboost_nested_cv()`
    AYRI cagrilar, Cox'un KENDI dis-fold bolunmesi XGBoost'unkinden
    BAGIMSIZ) bu NEGATIF KONTROLU GECEMEZ -- bir hastanin verisini
    bozmak, XGBoost'un HICBIR sekilde o hastayi test etmedigi bir
    fold'un EGITIM girdilerini (Cox skorlari uzerinden) DEGISTIREBILIR.
    Bu test, YENI (aligned) yolun cozdugu somut sorunu KANITLAR --
    ileride biri ESKI yolu yanlislikla 'dogru' sanip kullanirsa bu test
    onu KIRMIZI olarak uyarmaya devam eder."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=150, n_features=5, seed=81)
    seed = 5

    def _old_pipeline_fold0_train_scores(frame: pd.DataFrame) -> tuple[pd.Series, frozenset, frozenset]:
        nested_cv_result = run_nested_cv(
            frame,
            feature_columns,
            outer_splits=3,
            inner_splits=3,
            l1_ratio_grid=(1.0,),
            penalizer_grid=(0.3,),
            stability_selection=True,
            n_bootstrap_stability=20,
            seed=seed,
        )
        cox_oof = generate_out_of_fold_cox_scores(
            frame,
            feature_columns,
            nested_cv_result,
            outer_splits=3,
            seed=seed,
            n_bootstrap_stability=20,
        )
        combined = build_xgboost_feature_frame(
            cox_oof.scores["cox_oof_score"], frame[feature_columns], score_column="cox_score"
        )
        combined["target_12mo_survival"] = frame["target_12mo_survival"]
        xgb_result = train_xgboost_nested_cv(
            combined,
            feature_columns + ["cox_score"],
            outer_splits=3,
            inner_splits=3,
            seed=9,
            param_grid=_SMALL_XGB_GRID_ALIGNED,
            n_bootstrap_ci=50,
        )
        xgb_train0 = xgb_result.fold_audit[0]["train_patient_ids"]
        xgb_test0 = xgb_result.fold_audit[0]["test_patient_ids"]
        return combined.loc[list(xgb_train0), "cox_score"].sort_index(), xgb_train0, xgb_test0

    baseline_scores, xgb_train0, xgb_test0 = _old_pipeline_fold0_train_scores(labeled)

    perturb_id = sorted(xgb_test0)[0]
    perturbed = labeled.copy()
    perturbed.loc[perturb_id, "survival_days"] = perturbed.loc[perturb_id, "survival_days"] * 50.0 + 5000.0

    perturbed_scores, xgb_train0_2, xgb_test0_2 = _old_pipeline_fold0_train_scores(perturbed)

    # XGBoost'un KENDI fold yapisi degismedi (StratifiedKFold sadece
    # target_col'a bakar, survival_days'e DEGIL).
    assert xgb_train0 == xgb_train0_2
    assert xgb_test0 == xgb_test0_2

    common_idx = baseline_scores.index.intersection(perturbed_scores.index)
    max_diff = (baseline_scores.loc[common_idx] - perturbed_scores.loc[common_idx]).abs().max()

    # KIRMIZI: eski yolda, XGBoost'un GORMEDIGI bir test hastasinin
    # verisi bozulunca, XGBoost'un EGITIM kumesindeki Cox skorlari da
    # DEGISIYOR -- Cox'un KENDI fold'u XGBoost'unkinden BAGIMSIZ oldugu
    # icin. Bu, CRITICAL 1'in duzelttigi somut sizinti kanalidir.
    assert max_diff > 1e-6, (
        "Bu test ESKI/hizali-olmayan yolun sizintili oldugunu KANITLAMASI "
        "beklenirken sizintisiz cikti -- ya sentetik veri/seed bu ornekte "
        "sizintiyi tetiklemedi (baska bir seed/perturb_id denenmeli) ya da "
        "run_nested_cv()/generate_out_of_fold_cox_scores() davranisi "
        "degisti."
    )


# =====================================================================
# BOLUM 6B -- B9: pipeline/reduce_collinearity.py'nin XGBoost hattina
# FOLD-YEREL baglanmasi (`reduce_collinearity` bayragi, varsayilan KAPALI).
#
# BAGLAM: 2026-08-19 uretim kosusu `artifacts/week4/xgboost_v1_aligned/
# production.log` satir 4607-4647'de coktu -- `pipeline/xgboost_model.py:
# 1698 train_xgboost_with_fold_aligned_cox_scores() -> _fit_cox_with_full_
# selection() (satir 1367) -> pipeline/cox_model.py:1260 select_features_
# lasso() -> lifelines CoxPHFitter.fit()` zincirinde `numpy.linalg.
# LinAlgError: Matrix is singular.` -> `lifelines.exceptions.
# ConvergenceError: Convergence halted due to matrix inversion problems.
# Suspicion is high collinearity.` (production.log satir 4607-4647). Cokme
# noktasi TAM OLARAK ic cross-fit cagrisiydi (fit_type="cross_fit_final"),
# 93 WT-only STABLE_FEATURES_ICC60 adayi HAM (standardize edilmemis, cunku
# `clinical_standardize_columns` SADECE yas kolonlarini icerir --
# `tools/train_xgboost_week4.py::_rebuild_variant_training_frame_for_
# aligned_pipeline()` satir 705/711) olarak `select_features_lasso()`'ya
# giriyordu.
# =====================================================================


def test_reduce_collinearity_default_off_leaves_candidate_pool_unchanged() -> None:
    """B9 -- varsayilan davranis DEGISMEMELI: `reduce_collinearity`
    parametresini HIC vermemek (implicit False) ile ACIKCA `False`
    vermek AYNI (bit-birebir) sonucu uretmeli, ve HER fold_audit kaydinda
    `cox_candidate_features_before_collinearity_filter == _after_ ==
    len(feature_columns)` olmali -- mevcut 7 Cox kolunun sonuclarinin
    bit-birebir yeniden uretilebilir kalmasinin KANITI."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=150, n_features=5, seed=81)
    kwargs = dict(
        outer_splits=3,
        seed=5,
        cox_inner_splits=3,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_n_bootstrap_stability=20,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
        xgb_n_bootstrap_ci=50,
    )

    result_implicit = train_xgboost_with_fold_aligned_cox_scores(labeled, feature_columns, **kwargs)
    result_explicit_false = train_xgboost_with_fold_aligned_cox_scores(
        labeled, feature_columns, reduce_collinearity=False, **kwargs
    )

    for audit in list(result_implicit.fold_audit) + list(result_explicit_false.fold_audit):
        assert audit.reduce_collinearity_enabled is False
        assert audit.cox_candidate_features_before_collinearity_filter == len(feature_columns)
        assert audit.cox_candidate_features_after_collinearity_filter == len(feature_columns)

    pd.testing.assert_frame_equal(
        result_implicit.out_of_fold_predictions.sort_index(),
        result_explicit_false.out_of_fold_predictions.sort_index(),
    )


def test_reduce_collinearity_recomputed_once_per_outer_and_inner_fold_call_count() -> None:
    """B9 KIRMIZI/YESIL KANIT 1 (fold-yerellik, cagri SAYISI) --
    `reduce_collinearity=True` iken `pipeline.reduce_collinearity.
    build_v3_candidate_pool()` HER dis-fold (outer-final) VE HER ic-fold
    (cross-fit) icin AYRI AYRI cagrilmali -- cagri sayisi TAM OLARAK
    `outer_splits + outer_splits * cox_inner_splits` olmali. Bu, gorev
    talimatinin YASAKLADIGI "havuzdan fold-disi, BIR KEZ" (v3'un
    `run_single_v3_variant()`'ta kullandigi -- ve Codex'in 2026-08-19'da
    XGBoost hatti icin BLOK ettigi- full-pool RCS sinifi) uygulamanin
    BURADA yapilmadiginin dogrudan kanitidir."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=180, n_features=6, seed=71)
    outer_splits = 3
    cox_inner_splits = 3

    real_build_pool = xgboost_model_module.build_v3_candidate_pool
    call_log: list[int] = []

    def _counting_wrapper(*args, **kwargs):
        call_log.append(1)
        return real_build_pool(*args, **kwargs)

    with patch.object(xgboost_model_module, "build_v3_candidate_pool", side_effect=_counting_wrapper):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = train_xgboost_with_fold_aligned_cox_scores(
                labeled,
                feature_columns,
                outer_splits=outer_splits,
                seed=5,
                cox_inner_splits=cox_inner_splits,
                cox_l1_ratio_grid=(1.0,),
                cox_penalizer_grid=(0.3,),
                cox_n_bootstrap_stability=20,
                xgb_inner_splits=3,
                xgb_seed=9,
                xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
                xgb_n_bootstrap_ci=50,
                reduce_collinearity=True,
            )

    expected_calls = outer_splits + outer_splits * cox_inner_splits
    assert len(call_log) == expected_calls, (
        f"beklenen {expected_calls} cagri (outer_splits + outer_splits*"
        f"cox_inner_splits), gercek={len(call_log)} -- fold-yerellik "
        "BOZULMUS olabilir (orn. havuz bir kez hesaplanip TUM fold'larda "
        "yeniden kullaniliyor -- v3'un full-pool desenine geri donmus "
        "olabilir)."
    )
    assert isinstance(result, AlignedCoxXGBoostResult)
    for audit in result.fold_audit:
        assert audit.reduce_collinearity_enabled is True


def test_reduce_collinearity_pool_bit_identical_when_perturbed_patient_outside_that_folds_train_set() -> None:
    """B9 KIRMIZI/YESIL KANIT 2 (fold-yerellik, DEGER duzeyinde) -- bir
    hastanin KENDI radyomik ozellik degerini SERTCE bozunca, O HASTAYI
    train_frame'inde ICERMEYEN her `build_v3_candidate_pool()` cagrisinin
    dondurdugu (kept-features) listesi BIT-BIREBIR AYNI kalmali -- havuz
    GERCEKTEN o fold'un KENDI egitim alt-kumesinden hesaplaniyor, disaridan
    (baska bir fold'un/hastanin verisinden) hicbir sekilde ETKILENMIYOR.
    `tests/test_xgboost_model.py`'deki mevcut perturbation-testleriyle
    (BOLUM 6) AYNI desen, YENI bir bilesen (collinearity havuzu) icin."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=180, n_features=6, seed=71)
    kwargs = dict(
        outer_splits=3,
        seed=5,
        cox_inner_splits=3,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_n_bootstrap_stability=20,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
        xgb_n_bootstrap_ci=50,
        reduce_collinearity=True,
    )

    def _run_and_capture_pool_calls(
        frame: pd.DataFrame,
    ) -> list[tuple[frozenset, tuple[str, ...]]]:
        captured: list[tuple[frozenset, tuple[str, ...]]] = []
        real_build_pool = xgboost_model_module.build_v3_candidate_pool

        def _spy(train_frame, feats, **kw):
            kept, report = real_build_pool(train_frame, feats, **kw)
            captured.append((frozenset(train_frame.index), tuple(sorted(kept))))
            return kept, report

        with patch.object(xgboost_model_module, "build_v3_candidate_pool", side_effect=_spy):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_xgboost_with_fold_aligned_cox_scores(frame, feature_columns, **kwargs)
        return captured

    baseline_calls = _run_and_capture_pool_calls(labeled)

    all_patient_ids = set(labeled.index)
    first_call_train_ids = baseline_calls[0][0]
    candidates_outside_first_call = sorted(all_patient_ids - first_call_train_ids)
    assert candidates_outside_first_call, "beklenmedik: ilk cagri TUM hastalari icerdi"
    perturb_id = candidates_outside_first_call[0]

    perturbed = labeled.copy()
    perturbed.loc[perturb_id, "feat_2"] = perturbed.loc[perturb_id, "feat_2"] * 5.0 + 20.0

    perturbed_calls = _run_and_capture_pool_calls(perturbed)

    assert len(baseline_calls) == len(perturbed_calls)
    n_checked_unaffected = 0
    for (base_train_ids, base_kept), (pert_train_ids, pert_kept) in zip(baseline_calls, perturbed_calls):
        # fold yapisi (hangi hasta hangi fold'da) StratifiedKFold'un
        # SADECE target_col'a bakmasi sayesinde DEGISMEMELI (ozellik
        # degeri, hedef DEGIL, degistirildi).
        assert base_train_ids == pert_train_ids
        if perturb_id not in base_train_ids:
            assert base_kept == pert_kept, (
                "B9 REGRESYONU: bu fold'un build_v3_candidate_pool() "
                "cagrisinin train_frame'i bozulan hastayi ICERMIYOR ama "
                "kept-features listesi DEGISTI -- havuz GERCEKTE "
                "fold-yerel hesaplanmiyor olabilir (baska bir fold'un/"
                "hastanin verisi sizmis olabilir)."
            )
            n_checked_unaffected += 1

    assert n_checked_unaffected > 0, (
        "hicbir cagri bozulan hastayi disarida birakmadi -- test kurulumu "
        "(seed/perturb_id secimi) gozden gecirilmeli, bu negatif kontrolun "
        "hicbir sey KANITLAMADIGI anlamina gelir."
    )


def _make_synthetic_singular_cox_candidate_frame(
    n: int = 150, *, seed: int = 123, n_dup_pairs: int = 8, n_near_constant: int = 6
) -> tuple[pd.DataFrame, list[str]]:
    """2026-08-19 production cokusunun (`Matrix is singular.` ->
    `ConvergenceError: ... Suspicion is high collinearity.`) SENTETIK
    yeniden-uretimi -- gercek 93-ozellikli UPenn WT matrisindeki IKI
    problem sinifini (bkz. `pipeline/reduce_collinearity.py` modul
    docstring'i, Bulgu 2/3) KUCUK olcekte taklit eder:
      - `feat_near_constant_i`: buyuk/sabit ortalama (1.0+i) + cok kucuk
        (1e-10) gurultu -- GERCEKTEN near-constant (dusuk CV, `glcm_Idmn`
        deseni).
      - `feat_dup_a_i`/`feat_dup_b_i`: BIREBIR ayni deger (r=1.0, TAM
        rank-eksikligi) -- `glszm_SizeZoneNonUniformityNormalized<->
        glszm_SmallAreaEmphasis` (|r|=0,9964, VIF~172) ciftinin ABARTILI/
        garanti-tetikleyen versiyonu.
      - `feat_signal`: GERCEK Cox iliskili tek ozellik (true_beta=0.6)."""

    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    true_beta = 0.6
    baseline_hazard = 0.02
    u = rng.uniform(size=n)
    event_time = -np.log(u) / (baseline_hazard * np.exp(true_beta * signal))
    censoring_time = rng.uniform(low=1.0, high=np.percentile(event_time, 85), size=n)
    duration = np.minimum(event_time, censoring_time)
    event = (event_time <= censoring_time).astype(int)

    columns: dict[str, np.ndarray] = {"feat_signal": signal}
    for i in range(n_near_constant):
        columns[f"feat_near_constant_{i}"] = np.full(n, 1.0 + i) + rng.normal(scale=1e-10, size=n)
    for i in range(n_dup_pairs):
        base = rng.normal(size=n)
        columns[f"feat_dup_a_{i}"] = base
        columns[f"feat_dup_b_{i}"] = base.copy()

    frame = pd.DataFrame(columns)
    frame["survival_days"] = duration
    frame["event"] = event
    frame.index = pd.Index([f"P{i:04d}" for i in range(n)], name="patient_id")
    candidate_columns = [c for c in frame.columns if c not in ("survival_days", "event")]
    return frame, candidate_columns


def test_reduce_collinearity_prevents_the_exact_production_singular_matrix_crash() -> None:
    """B9 KIRMIZI/YESIL KANIT 3 (singularity GERCEKTEN cozuldu mu?) --
    `pipeline/cox_model.py::select_features_lasso()` (production.log'un
    coktugu TAM fonksiyon, `_fit_cox_with_full_selection()` -> `select_
    features_lasso()` -> `cox.fit()` zincirinin en-dip halkasi) SENTETIK
    ama GERCEKCI (near-constant + tam-kolineer cift) bir aday havuzuyla,
    FILTRE UYGULANMADAN (`reduce_collinearity` KAPALI durumun esdegeri --
    tam `candidate_columns`), production.log'daki BIREBIR AYNI istisna
    SINIFINI (`lifelines.exceptions.ConvergenceError`, mesaj: "Convergence
    halted due to matrix inversion problems. Suspicion is high
    collinearity.") uretir (KIRMIZI). AYNI veri + AYNI penalizer,
    `build_v3_candidate_pool()`'dan gecirilmis (near-constant + |r|>=0.95
    dedup) havuzla TEMIZ fit eder (YESIL) -- B9'un iddia ettigi mekanizmanin
    (`pipeline.xgboost_model.train_xgboost_with_fold_aligned_cox_scores()`
    `reduce_collinearity=True` iken TAM OLARAK bu iki cagriyi yapar)
    dogrudan, izole KANITIDIR."""

    frame, candidate_columns = _make_synthetic_singular_cox_candidate_frame()
    fixed_penalizer = 0.05  # DEFAULT_PENALIZER_GRID'in (0.05,0.1,0.2,0.5,1.0) EN KUCUK ucu

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # KIRMIZI: filtre YOK -- TAM aday havuzu (near-constant + tam-
        # kolineer ciftler DAHIL) production.log'daki AYNI istisna
        # sinifini/mesajini uretiyor.
        with pytest.raises(ConvergenceError, match="Suspicion is high collinearity"):
            select_features_lasso(
                frame,
                candidate_columns,
                duration_col="survival_days",
                event_col="event",
                penalizer=fixed_penalizer,
                l1_ratio=1.0,
            )

        # YESIL: AYNI veri + AYNI penalizer, ama `build_v3_candidate_pool()`
        # ile daraltilmis havuz -- near-constant + kolineer cift-dedup
        # SONRASI temiz fit.
        reduced_columns, report = build_v3_candidate_pool(frame, candidate_columns)
        assert len(reduced_columns) < len(candidate_columns), (
            "beklenmedik: build_v3_candidate_pool() bu SENTETIK near-"
            "constant/tam-kolineer havuzdan HICBIR ozellik dusurmedi -- "
            "test kurulumu bozuk olabilir."
        )
        assert not (set(report.near_constant_dropped) & set(reduced_columns))
        assert "feat_signal" in reduced_columns

        selected, fitted_cox, coefficients = select_features_lasso(
            frame,
            reduced_columns,
            duration_col="survival_days",
            event_col="event",
            penalizer=fixed_penalizer,
            l1_ratio=1.0,
        )

    assert fitted_cox is not None
    assert "feat_signal" in coefficients.index


# =====================================================================
# BOLUM 6-B -- Ş2 / K16-a+b (Codex sartli onay, 2026-09-11): final Cox
# fit penalizer-eskalasyonu + acik fold-atlama bayragi
# =====================================================================


def test_fit_final_cox_model_with_penalizer_escalation_recovers_from_singular_matrix() -> None:
    """K16-a KIRMIZI/YESIL KANIT -- `_fit_final_cox_model_with_penalizer_
    escalation()`, `_make_synthetic_singular_cox_candidate_frame()`'in
    (production.log'un GERCEK cokus sinifi) ayni near-constant/tam-
    kolineer havuzuyla, `initial_penalizer=0.05` (KIRMIZI -- tek basina
    ConvergenceError verir) ile cagrildiginda, `DEFAULT_COX_PENALIZER_
    ESCALATION_GRID`'e eskalasyon YAPARAK basariyla fit eder (YESIL) --
    kullanilan penalizer'in GERCEKTEN eskalasyon grid'inden geldigini VE
    basarisiz ilk denemenin `failure_sink`'e (Ş3 crash-manifest girdisi)
    dogru sekilde kaydedildigini dogrudan kanitlar."""

    frame, candidate_columns = _make_synthetic_singular_cox_candidate_frame()
    fit_frame = frame[candidate_columns + ["survival_days", "event"]]
    failure_sink: list[xgboost_model_module.CoxFitFailureRecord] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, used_penalizer, escalated, attempts = (
            xgboost_model_module._fit_final_cox_model_with_penalizer_escalation(
                fit_frame,
                duration_col="survival_days",
                event_col="event",
                final_features=candidate_columns,
                extra_columns=[],
                initial_penalizer=0.05,
                l1_ratio=1.0,
                extra_column_penalizer=None,
                penalizer_escalation_grid=xgboost_model_module.DEFAULT_COX_PENALIZER_ESCALATION_GRID,
                error_context="test",
                outer_fold=0,
                inner_fold=None,
                fit_type="outer_final",
                candidate_feature_columns=candidate_columns,
                failure_sink=failure_sink,
            )
        )

    assert escalated is True
    assert used_penalizer in xgboost_model_module.DEFAULT_COX_PENALIZER_ESCALATION_GRID
    assert used_penalizer != 0.05
    assert model is not None
    assert attempts[0]["penalizer"] == 0.05
    assert attempts[0]["success"] is False
    assert attempts[0]["error_class"] == "ConvergenceError"
    assert attempts[-1]["success"] is True

    assert len(failure_sink) == 1, "SADECE ilk (basarisiz) deneme failure_sink'e girmeli"
    failure_record = failure_sink[0]
    assert failure_record.penalizer == 0.05
    assert failure_record.error_class == "ConvergenceError"
    assert failure_record.fit_type == "outer_final"
    assert failure_record.outer_fold == 0
    assert failure_record.candidate_feature_columns_hash == xgboost_model_module.hash_feature_columns(
        candidate_columns
    )


def test_fit_final_cox_model_with_penalizer_escalation_does_not_swallow_other_exceptions() -> None:
    """K16-a DAR-KAPSAM KANITI -- ConvergenceError/LinAlgError DISINDAKI
    bir istisna (orn. `l1_ratio` araligi disi -- lifelines'in KENDI
    ValueError'i) eskalasyon tarafindan YUTULMAZ, ILK denemede aynen
    yukari firlar (hicbir eskalasyon denemesi YAPILMAZ)."""

    frame, candidate_columns = _make_synthetic_singular_cox_candidate_frame(n=40, n_dup_pairs=0, n_near_constant=0)
    fit_frame = frame[candidate_columns + ["survival_days", "event"]]
    failure_sink: list[xgboost_model_module.CoxFitFailureRecord] = []

    with pytest.raises(ValueError):
        xgboost_model_module._fit_final_cox_model_with_penalizer_escalation(
            fit_frame,
            duration_col="survival_days",
            event_col="event",
            final_features=candidate_columns,
            extra_columns=[],
            initial_penalizer=0.3,
            l1_ratio=5.0,  # gecersiz -- l1_ratio [0,1] araliginda olmali
            extra_column_penalizer=None,
            penalizer_escalation_grid=xgboost_model_module.DEFAULT_COX_PENALIZER_ESCALATION_GRID,
            error_context="test",
            outer_fold=None,
            inner_fold=None,
            fit_type="outer_final",
            candidate_feature_columns=candidate_columns,
            failure_sink=failure_sink,
        )

    assert failure_sink == [], (
        "l1_ratio hatasi bir kod/girdi hatasi isaretidir -- eskalasyon "
        "DENENMEMELI, failure_sink'e HICBIR kayit girmemeli."
    )


def test_fit_final_cox_model_with_penalizer_escalation_exhausted_raises_with_all_attempts_logged() -> None:
    """K16-a -- eskalasyon grid'inin TAMAMI da basarisiz olursa
    `CoxPenalizerEscalationExhaustedError` firlar VE `failure_sink`'te
    (initial + TUM grid) kadar basarisiz kayit birikir -- SESSIZCE
    yarim birakilmaz."""

    frame, candidate_columns = _make_synthetic_singular_cox_candidate_frame()
    fit_frame = frame[candidate_columns + ["survival_days", "event"]]
    failure_sink: list[xgboost_model_module.CoxFitFailureRecord] = []

    tiny_escalation_grid = (0.06, 0.07)  # ikisi de 0.05 gibi cok kucuk -- HALA singular

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(xgboost_model_module.CoxPenalizerEscalationExhaustedError):
            xgboost_model_module._fit_final_cox_model_with_penalizer_escalation(
                fit_frame,
                duration_col="survival_days",
                event_col="event",
                final_features=candidate_columns,
                extra_columns=[],
                initial_penalizer=0.05,
                l1_ratio=1.0,
                extra_column_penalizer=None,
                penalizer_escalation_grid=tiny_escalation_grid,
                error_context="test-exhausted",
                outer_fold=2,
                inner_fold=None,
                fit_type="outer_final",
                candidate_feature_columns=candidate_columns,
                failure_sink=failure_sink,
            )

    assert len(failure_sink) == 3  # initial (0.05) + iki eskalasyon denemesi
    assert [record.penalizer for record in failure_sink] == [0.05, 0.06, 0.07]
    assert all(record.outer_fold == 2 for record in failure_sink)


def test_train_xgboost_with_fold_aligned_cox_scores_allow_fold_skip_false_reraises_by_default() -> None:
    """K16-b VARSAYILAN DAVRANIS -- `allow_fold_skip=False` (VARSAYILAN,
    hic verilmemis) iken, bir fold'un Cox fit'i eskalasyon TUKENDIGI icin
    basarisiz olursa pipeline ESKISI GIBI coker (fail-loud, sessizce
    atlama YOK)."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=150, n_features=5, seed=81)
    real_fit = xgboost_model_module._fit_cox_with_full_selection

    def _flaky_fit(*args, **kwargs):
        if kwargs.get("outer_fold") == 0 and kwargs.get("fit_type") == "outer_final":
            raise xgboost_model_module.CoxPenalizerEscalationExhaustedError("sentetik cokus (test)")
        return real_fit(*args, **kwargs)

    with patch.object(xgboost_model_module, "_fit_cox_with_full_selection", side_effect=_flaky_fit):
        with pytest.raises(xgboost_model_module.CoxPenalizerEscalationExhaustedError):
            train_xgboost_with_fold_aligned_cox_scores(
                labeled,
                feature_columns,
                outer_splits=3,
                seed=5,
                cox_inner_splits=3,
                cox_l1_ratio_grid=(1.0,),
                cox_penalizer_grid=(0.3,),
                cox_n_bootstrap_stability=20,
                xgb_inner_splits=3,
                xgb_seed=9,
                xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
                xgb_n_bootstrap_ci=50,
                # allow_fold_skip VERILMIYOR -- varsayilan False.
            )


def test_train_xgboost_with_fold_aligned_cox_scores_allow_fold_skip_true_skips_and_continues() -> None:
    """K16-b -- `allow_fold_skip=True` iken AYNI cokus, o TEK dis-fold'u
    ATLAR (coktermez): `result.skipped_outer_folds` fold 0'i kaydeder,
    `out_of_fold_predictions` fold 0'in test hastalarini ICERMEZ, kalan
    fold'lar NORMAL sekilde tamamlanir, `verify_aligned_fold_leakage_
    free()` skip-sonrasi sonucta da GECER."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=150, n_features=5, seed=81)
    real_fit = xgboost_model_module._fit_cox_with_full_selection

    def _flaky_fit(*args, **kwargs):
        if kwargs.get("outer_fold") == 0 and kwargs.get("fit_type") == "outer_final":
            raise xgboost_model_module.CoxPenalizerEscalationExhaustedError("sentetik cokus (test)")
        return real_fit(*args, **kwargs)

    with patch.object(xgboost_model_module, "_fit_cox_with_full_selection", side_effect=_flaky_fit):
        result = train_xgboost_with_fold_aligned_cox_scores(
            labeled,
            feature_columns,
            outer_splits=3,
            seed=5,
            cox_inner_splits=3,
            cox_l1_ratio_grid=(1.0,),
            cox_penalizer_grid=(0.3,),
            cox_n_bootstrap_stability=20,
            xgb_inner_splits=3,
            xgb_seed=9,
            xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
            xgb_n_bootstrap_ci=50,
            allow_fold_skip=True,
        )

    assert len(result.skipped_outer_folds) == 1
    assert result.skipped_outer_folds[0]["fold"] == 0
    assert len(result.fold_audit) == 2, "atlanan fold fold_audit'e EKLENMEMELI"
    assert all(audit.fold != 0 for audit in result.fold_audit)
    assert set(result.fold_results["fold"]) == {1, 2}

    verify_aligned_fold_leakage_free(result)

    assert len(result.out_of_fold_predictions) < len(labeled), (
        "atlanan fold'un test hastalari out_of_fold_predictions'ta HALA "
        "VARSA fold-atlama GERCEKTE calismiyor demektir."
    )


def test_train_xgboost_with_fold_aligned_cox_scores_default_escalation_grid_does_not_change_happy_path() -> None:
    """VARSAYILAN-DAVRANIS KORUNUMU -- eskalasyon mekanizmasi (Ş2)
    eklendikten SONRA bile, HICBIR ConvergenceError/LinAlgError
    tetiklenmeyen (mevcut 7 Cox kolunun tipik durumu) normal bir kosuda
    sonuc `cox_penalizer_escalation_grid` HIC verilmemis/verilmis
    (varsayilanla AYNI deger) arasinda BIT-BIREBIR AYNI kalir, VE hicbir
    `CoxFitProvenanceRecord`'da `penalizer_escalated=True` GORULMEZ."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=150, n_features=5, seed=81)
    kwargs = dict(
        outer_splits=3,
        seed=5,
        cox_inner_splits=3,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_n_bootstrap_stability=20,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
        xgb_n_bootstrap_ci=50,
    )

    result_implicit = train_xgboost_with_fold_aligned_cox_scores(labeled, feature_columns, **kwargs)
    result_explicit = train_xgboost_with_fold_aligned_cox_scores(
        labeled,
        feature_columns,
        cox_penalizer_escalation_grid=xgboost_model_module.DEFAULT_COX_PENALIZER_ESCALATION_GRID,
        allow_fold_skip=False,
        **kwargs,
    )

    pd.testing.assert_frame_equal(
        result_implicit.out_of_fold_predictions.sort_index(),
        result_explicit.out_of_fold_predictions.sort_index(),
    )
    assert result_implicit.skipped_outer_folds == []
    assert result_implicit.cox_fit_failure_log == []
    for record in result_implicit.fit_provenance:
        assert record.penalizer_escalated is False
        assert record.used_penalizer == record.best_penalizer


# =====================================================================
# BOLUM 6-C -- Ş4 (Codex sartli onay, 2026-09-11): cross-fit (HAM veri)
# vs dagitim (standardize veri) Cox filtresi kolon-kesisim guard'i
# =====================================================================


def test_assert_no_clinical_standardize_feature_overlap_passes_when_disjoint() -> None:
    assert_no_clinical_standardize_feature_overlap(
        ["feat_0", "feat_1"], ["clinical_age"], context="test"
    )  # raise ETMEMELI


def test_assert_no_clinical_standardize_feature_overlap_raises_on_overlap() -> None:
    with pytest.raises(CoxFeatureCandidatePoolOverlapError, match="feat_0"):
        assert_no_clinical_standardize_feature_overlap(
            ["feat_0", "feat_1"], ["feat_0", "clinical_age"], context="test-context"
        )


def test_train_xgboost_with_fold_aligned_cox_scores_raises_on_clinical_feature_overlap() -> None:
    """K16/Ş4 KABLOLAMA KANITI -- `train_xgboost_with_fold_aligned_cox_
    scores()`'un KENDISI de bu guard'i cagiriyor mu? (sadece izole
    fonksiyonun var olmasi yetmez, gercekten CAGRILDIGI dogrulanmali)."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=60, n_features=3, seed=81)
    overlapping_clinical_column = feature_columns[0]

    with pytest.raises(CoxFeatureCandidatePoolOverlapError, match=overlapping_clinical_column):
        train_xgboost_with_fold_aligned_cox_scores(
            labeled,
            feature_columns,
            clinical_standardize_columns=[overlapping_clinical_column],
            outer_splits=3,
            seed=5,
            cox_inner_splits=3,
            cox_l1_ratio_grid=(1.0,),
            cox_penalizer_grid=(0.3,),
            cox_n_bootstrap_stability=20,
            xgb_inner_splits=3,
            xgb_seed=9,
            xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
            xgb_n_bootstrap_ci=50,
        )


# =====================================================================
# BOLUM 6-D -- Ş5 (Codex sartli onay, 2026-09-11): fold-bazli collinearity
# filtre raporu -- dusurulen ozellikler + korelasyon kumeleri + esikler
# =====================================================================


def test_collinearity_filter_reports_empty_when_reduce_collinearity_off() -> None:
    """VARSAYILAN-DAVRANIS KORUNUMU: `reduce_collinearity=False` (varsayilan)
    iken `collinearity_filter_reports` HER ZAMAN BOS -- yeni alan eski
    7 kolun ciktisini ETKILEMEZ."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=150, n_features=5, seed=81)
    result = train_xgboost_with_fold_aligned_cox_scores(
        labeled,
        feature_columns,
        outer_splits=3,
        seed=5,
        cox_inner_splits=3,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_n_bootstrap_stability=20,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
        xgb_n_bootstrap_ci=50,
    )
    assert result.collinearity_filter_reports == []


def test_collinearity_filter_reports_populated_with_dropped_features_and_thresholds_when_on() -> None:
    """Ş5 ASIL KORUMA -- `reduce_collinearity=True` iken HER outer-final
    VE HER ic cross-fit cagrisi icin (outer_splits + outer_splits*cox_
    inner_splits KADAR kayit) dusurulen ozellik listeleri + korelasyon
    kumesi detaylari + esikler RAPORA GIRER (once SADECE toplam sayi
    vardi, artik HANGI ozelliklerin NEDEN dustugu izlenebilir)."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=180, n_features=6, seed=71)
    outer_splits = 3
    cox_inner_splits = 3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = train_xgboost_with_fold_aligned_cox_scores(
            labeled,
            feature_columns,
            outer_splits=outer_splits,
            seed=5,
            cox_inner_splits=cox_inner_splits,
            cox_l1_ratio_grid=(1.0,),
            cox_penalizer_grid=(0.3,),
            cox_n_bootstrap_stability=20,
            xgb_inner_splits=3,
            xgb_seed=9,
            xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
            xgb_n_bootstrap_ci=50,
            reduce_collinearity=True,
        )

    expected_count = outer_splits + outer_splits * cox_inner_splits
    assert len(result.collinearity_filter_reports) == expected_count

    scopes = {rep["scope"] for rep in result.collinearity_filter_reports}
    assert scopes == {"outer_final", "cross_fit_final"}

    for rep in result.collinearity_filter_reports:
        assert rep["cv_threshold"] == xgboost_model_module.DEFAULT_CV_THRESHOLD
        assert rep["corr_threshold"] == xgboost_model_module.DEFAULT_CORRELATION_CLUSTER_THRESHOLD
        assert rep["n_input_features"] == len(feature_columns)
        assert rep["n_kept_features"] == len(rep["kept_features"])
        assert rep["n_near_constant_dropped"] == len(rep["near_constant_dropped"])
        assert rep["n_correlation_dropped"] == len(rep["correlation_dropped"])
        assert isinstance(rep["kept_features"], list)
        # her bir korelasyon-dusum kaydi, kaynak V3CandidatePoolReport'un
        # kolonlarini (dropped_feature/kept_representative/cluster_id/
        # cluster_size) TASIR -- SESSIZCE ozetlenmedi.
        for dropped_row in rep["correlation_dropped"]:
            assert set(dropped_row.keys()) == {
                "dropped_feature",
                "kept_representative",
                "cluster_id",
                "cluster_size",
            }


def test_collinearity_filter_reports_populated_for_full_pool_pipeline() -> None:
    """Ş5 -- `fit_final_cox_and_xgboost_pipeline_on_full_pool()` icin de
    AYNI alan/mekanizma (`scope=\"full_pool_final\"`, split basina bir kayit)."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=160, n_features=5, seed=91)
    cross_fit_splits = 3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_final_cox_and_xgboost_pipeline_on_full_pool(
            labeled,
            feature_columns,
            cox_cross_fit_splits=cross_fit_splits,
            cox_inner_splits=3,
            cox_l1_ratio_grid=(1.0,),
            cox_penalizer_grid=(0.3,),
            cox_n_bootstrap_stability=20,
            seed=7,
            xgb_inner_splits=3,
            xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
            reduce_collinearity=True,
        )

    assert len(result.collinearity_filter_reports) == cross_fit_splits
    assert all(rep["scope"] == "full_pool_final" for rep in result.collinearity_filter_reports)


# =====================================================================
# BOLUM 6C -- v3 RECETE-SADAKATI (2026-09-12, Baris: "sapmayla kosmayalim,
# v3b ile XGBoost tam uyumluluk gostermesi lazim"). `tools/train_cox_
# week3.py::V3VariantConfig.standardize_radiomics=True` -- `build_v3_
# candidate_pool()`'un HAM veriden sectigi radyomikler (`clinical_
# standardize_columns = feature_columns + clinical_age_columns`, satir
# ~3806-3812) fold-yerel Z-score'a sokulur, PENALIZE EDILEN bir kovaryat
# oldugu icin bu Cox fit'ini/secimini GERCEKTEN degistirir (standardize_
# columns_fold_safe()'in docstring notu: penalize EDILMEYEN yas icin
# fark YOK, ama radyomikler icin VAR). Bu bolum, `train_xgboost_with_
# fold_aligned_cox_scores()`/`fit_final_cox_and_xgboost_pipeline_on_
# full_pool()`'un `reduce_collinearity=True` iken ARTIK bu standardizasyonu
# GERCEKTEN uyguladigini (ve `CoxFeatureCandidatePoolOverlapError`'in
# koruduğu sirayi -- filtre ONCE HAM veriden, standardizasyon SONRA --
# BOZMADIGINI) dogrudan, deger-seviyesinde kanitlar.
# =====================================================================


def test_reduce_collinearity_standardizes_selected_radiomics_fold_locally_in_aligned_pipeline() -> None:
    """KIRMIZI/YESIL KANIT -- `reduce_collinearity=True` iken HER outer-
    final VE HER ic cross-fit `standardize_columns_fold_safe()` cagrisi,
    o fold'un KENDI secilmis (post-filter) radyomik havuzunu da (yasla
    birlikte) standardize eder -- cikan egitim cercevesinde o kolonlarin
    ortalamasi ~0, std'si ~1 olur (ddof=0). `reduce_collinearity=False`
    iken (varsayilan) davranis DEGISMEZ -- SADECE `clinical_standardize_
    columns` (bu testte bos) standardize edilir, radyomikler HAM kalir."""

    labeled, feature_columns = _make_synthetic_labeled_frame_with_collinear_radiomics(n=150, seed=17)

    real_standardize = xgboost_model_module.standardize_columns_fold_safe
    captured_on: list[tuple[list[str], pd.DataFrame]] = []
    captured_off: list[tuple[list[str], pd.DataFrame]] = []

    def _make_spy(sink: list[tuple[list[str], pd.DataFrame]]):
        def _spy(train_frame, test_frame, columns):
            train_std, test_std = real_standardize(train_frame, test_frame, columns)
            sink.append((list(columns or []), train_std))
            return train_std, test_std

        return _spy

    kwargs = dict(
        outer_splits=3,
        seed=5,
        cox_inner_splits=3,
        # NOT: pure LASSO (l1_ratio=1.0) yerine 0.7 -- bu testin amaci
        # STANDARDIZASYONU dogrulamak, LASSO'nun tam-birebir uretilebilirligini
        # DEGIL; kucuk-n + near-constant/dup EK kolonlarla (filtrelense bile
        # bootstrap stabilite secimi ARA ADIMLARINDA -- escalation'in
        # KAPSAMADIGI bir katman, bkz. _fit_cox_with_full_selection()
        # docstring'i) pure-LASSO daha sik singular matrise dusuyor.
        cox_l1_ratio_grid=(0.7,),
        cox_penalizer_grid=(0.3,),
        cox_n_bootstrap_stability=20,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
        xgb_n_bootstrap_ci=50,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with patch.object(
            xgboost_model_module, "standardize_columns_fold_safe", side_effect=_make_spy(captured_on)
        ):
            train_xgboost_with_fold_aligned_cox_scores(
                labeled, feature_columns, reduce_collinearity=True, **kwargs
            )
        with patch.object(
            xgboost_model_module, "standardize_columns_fold_safe", side_effect=_make_spy(captured_off)
        ):
            train_xgboost_with_fold_aligned_cox_scores(
                labeled, feature_columns, reduce_collinearity=False, **kwargs
            )

    assert captured_on, "standardize_columns_fold_safe hic cagrilmadi (kurulum hatasi)"
    n_calls_with_radiomics_standardized = 0
    for columns, train_std in captured_on:
        radiomic_columns_in_call = [c for c in columns if c in feature_columns]
        if not radiomic_columns_in_call:
            continue
        n_calls_with_radiomics_standardized += 1
        for col in radiomic_columns_in_call:
            assert train_std[col].mean() == pytest.approx(0.0, abs=1e-9), (
                f"reduce_collinearity=True: secilmis radyomik '{col}' fold-yerel "
                "standardize edilmemis (v3 recete-sadakati BOZULMUS)."
            )
            assert train_std[col].std(ddof=0) == pytest.approx(1.0, abs=1e-6) or train_std[col].std(
                ddof=0
            ) == pytest.approx(0.0, abs=1e-9)
    assert n_calls_with_radiomics_standardized > 0, (
        "reduce_collinearity=True iken HICBIR cagrida secilmis radyomik "
        "standardize listesine girmedi -- test kurulumu (near-constant/"
        "korelasyon esikleri) gozden gecirilmeli."
    )

    # KAPALI durumda (varsayilan): radyomikler HICBIR cagrida standardize
    # listesine girmemeli -- davranis eskisiyle BIREBIR AYNI kalmali.
    for columns, _train_std in captured_off:
        radiomic_columns_in_call = [c for c in columns if c in feature_columns]
        assert not radiomic_columns_in_call, (
            "reduce_collinearity=False iken bir cagri radyomik icerdi -- "
            "varsayilan davranis DEGISMIS (regresyon)."
        )


def test_reduce_collinearity_standardizes_selected_radiomics_split_locally_in_full_pool_pipeline() -> None:
    """`fit_final_cox_and_xgboost_pipeline_on_full_pool()` icin AYNI kanit
    -- split-yerel standardizasyon `reduce_collinearity=True` iken
    GERCEKTEN uygulanir, KAPALIYKEN davranis DEGISMEZ."""

    labeled, feature_columns = _make_synthetic_labeled_frame_with_collinear_radiomics(n=160, seed=23)

    real_standardize = xgboost_model_module.standardize_columns_fold_safe
    captured_on: list[tuple[list[str], pd.DataFrame]] = []

    def _spy(train_frame, test_frame, columns):
        train_std, test_std = real_standardize(train_frame, test_frame, columns)
        captured_on.append((list(columns or []), train_std))
        return train_std, test_std

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with patch.object(xgboost_model_module, "standardize_columns_fold_safe", side_effect=_spy):
            fit_final_cox_and_xgboost_pipeline_on_full_pool(
                labeled,
                feature_columns,
                cox_cross_fit_splits=3,
                cox_inner_splits=3,
                cox_l1_ratio_grid=(1.0,),
                cox_penalizer_grid=(0.3,),
                cox_n_bootstrap_stability=20,
                seed=11,
                xgb_inner_splits=3,
                xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
                reduce_collinearity=True,
            )

    assert captured_on, "standardize_columns_fold_safe hic cagrilmadi (kurulum hatasi)"
    n_calls_with_radiomics_standardized = 0
    for columns, train_std in captured_on:
        radiomic_columns_in_call = [c for c in columns if c in feature_columns]
        if not radiomic_columns_in_call:
            continue
        n_calls_with_radiomics_standardized += 1
        for col in radiomic_columns_in_call:
            assert train_std[col].mean() == pytest.approx(0.0, abs=1e-9)
    assert n_calls_with_radiomics_standardized > 0, (
        "reduce_collinearity=True iken full-pool split'lerinin HICBIRINDE "
        "secilmis radyomik standardize edilmedi -- v3 recete-sadakati "
        "BOZULMUS olabilir."
    )


def test_reduce_collinearity_pool_still_computed_on_raw_data_even_with_new_standardization_step() -> None:
    """REGRESYON KORUMASI -- v3 recete-sadakati duzeltmesi (standardizasyonu
    EKLEMEK), `build_v3_candidate_pool()`'un HALA HAM veriden cagrildigini
    BOZMAMALI (aksi halde near-constant/CV filtresi Z-skorlanmis deger
    uzerinden hesaplanirdi -- `CoxFeatureCandidatePoolOverlapError`'in tam
    onledigi risk). `test_reduce_collinearity_pool_bit_identical_when_
    perturbed_patient_outside_that_folds_train_set()` (BOLUM 6B) ile AYNI
    desen, burada AYRICA `build_v3_candidate_pool()`'a giden `train_frame`
    argumaninin gercekten HAM (standardize edilmemis mean/std profiline
    sahip) oldugunu DEGER seviyesinde dogrular."""

    labeled, feature_columns = _make_synthetic_labeled_frame_with_collinear_radiomics(n=150, seed=29)

    real_build_pool = xgboost_model_module.build_v3_candidate_pool
    captured_pool_inputs: list[pd.DataFrame] = []

    def _spy(train_frame, feats, **kw):
        captured_pool_inputs.append(train_frame.copy())
        return real_build_pool(train_frame, feats, **kw)

    with patch.object(xgboost_model_module, "build_v3_candidate_pool", side_effect=_spy):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_xgboost_with_fold_aligned_cox_scores(
                labeled,
                feature_columns,
                outer_splits=3,
                seed=5,
                cox_inner_splits=3,
                cox_l1_ratio_grid=(1.0,),
                cox_penalizer_grid=(0.3,),
                cox_n_bootstrap_stability=20,
                xgb_inner_splits=3,
                xgb_seed=9,
                xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
                xgb_n_bootstrap_ci=50,
                reduce_collinearity=True,
            )

    assert captured_pool_inputs, "build_v3_candidate_pool hic cagrilmadi"
    # `feat_near_constant_i` HAM veride buyuk/sabit ortalamalar tasir
    # (5.0+i, bkz. _make_synthetic_labeled_frame_with_collinear_radiomics())
    # -- standardize edilmis olsaydi ortalamalari ~0 olurdu.
    for pool_input in captured_pool_inputs:
        assert pool_input["feat_near_constant_0"].mean() == pytest.approx(5.0, abs=0.05), (
            "build_v3_candidate_pool() standardize EDILMIS veri almis "
            "olabilir -- near-constant/CV filtresi Z-skorlanmis deger "
            "uzerinden hesaplanirdi (regresyon, guard'in korudugu risk)."
        )


# =====================================================================
# BOLUM 7 -- MEDIUM 4 (Codex): gecersiz sansur girdisi SESSIZCE
# dusurulmez, GURULTULU durur
# =====================================================================


def test_define_twelve_month_survival_target_rejects_negative_duration() -> None:
    frame = pd.DataFrame(
        {"survival_days": [400.0, -5.0, 200.0], "event": [0, 1, 0]}, index=["A", "B", "C"]
    )
    with pytest.raises(ValueError, match="gecersiz girdi"):
        define_twelve_month_survival_target(frame)


def test_define_twelve_month_survival_target_rejects_nan_duration() -> None:
    frame = pd.DataFrame(
        {"survival_days": [400.0, np.nan, 200.0], "event": [0, 1, 0]}, index=["A", "B", "C"]
    )
    with pytest.raises(ValueError, match="gecersiz girdi"):
        define_twelve_month_survival_target(frame)


def test_define_twelve_month_survival_target_rejects_event_outside_zero_one() -> None:
    frame = pd.DataFrame(
        {"survival_days": [400.0, 100.0, 200.0], "event": [0, 2, 0]}, index=["A", "B", "C"]
    )
    with pytest.raises(ValueError, match="gecersiz girdi"):
        define_twelve_month_survival_target(frame)


def test_apply_twelve_month_target_raises_on_report_mismatch() -> None:
    frame = pd.DataFrame(
        {"survival_days": [400.0, 100.0, 200.0], "event": [0, 1, 0]}, index=["A", "B", "C"]
    )
    target, report = define_twelve_month_survival_target(frame)

    # baska bir frame/report'tan gelen (uyusmayan) bir target -- C yerine
    # HICBIRI dusurulmesin diye NaN'lari sildik.
    mismatched_target = target.copy()
    mismatched_target.loc["C"] = 1.0  # artik NaN degil -- report ile UYUSMAZ

    with pytest.raises(ValueError, match="UYUSMUYOR"):
        apply_twelve_month_target(frame, mismatched_target, report=report)


def test_apply_twelve_month_target_accepts_matching_report() -> None:
    frame = pd.DataFrame(
        {"survival_days": [400.0, 100.0, 200.0], "event": [0, 1, 0]}, index=["A", "B", "C"]
    )
    target, report = define_twelve_month_survival_target(frame)
    result = apply_twelve_month_target(frame, target, report=report)
    assert set(result.index) == {"A", "B"}


# =====================================================================
# BOLUM 8 -- MEDIUM 5 (Codex): final full-pool fit + UCSF-tarzi harici
# degerlendirme yolu
# =====================================================================


def test_fit_final_cox_and_xgboost_pipeline_on_full_pool_and_evaluate_external() -> None:
    from lifelines import CoxPHFitter

    labeled, feature_columns = _make_synthetic_labeled_frame(n=160, n_features=5, seed=91)

    # Cox'un KENDI "final full-pool" adimi (tools/train_cox_week3.py::
    # fit_final_model_on_full_pool()'un BASITLESTIRILMIS -- SADECE tek
    # bir sabit (penalizer, l1_ratio) ile -- bu testte ONA dokunmuyoruz,
    # o public fonksiyon zaten tools/train_xgboost_week4.py::
    # run_ucsf_external_evaluation()'da CAGRILIYOR; burada SADECE ondan
    # SONRAKI XGBoost katmani + harici degerlendirme test ediliyor).
    cox_final_features = feature_columns[:1]
    cox = CoxPHFitter(penalizer=0.3, l1_ratio=1.0)
    cox.fit(
        labeled[cox_final_features + ["survival_days", "event"]],
        duration_col="survival_days",
        event_col="event",
    )

    # 3. tur CRITICAL duzeltmesi (task-mt06mqzi-k0gb5j): `cox_fitted_
    # model`/`cox_final_features`/`cox_best_penalizer`/`cox_best_l1_ratio`
    # ARTIK bu fonksiyona VERILMIYOR -- her cross-fit split KENDI secimini
    # yapar. `cox`/`cox_final_features` (yukarida) SADECE asagidaki
    # `evaluate_xgboost_external_test()` cagrisinda (gercek harici/uretim
    # skorlamasi icin) kullanilir.
    xgboost_final = fit_final_cox_and_xgboost_pipeline_on_full_pool(
        labeled,
        feature_columns,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_inner_splits=3,
        cox_n_bootstrap_stability=20,
        cox_cross_fit_splits=3,
        seed=5,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
    )
    verify_full_pool_pipeline_leakage_free(xgboost_final, labeled.index)

    assert len(xgboost_final.cross_fit_cox_scores) == len(labeled)
    assert set(xgboost_final.cross_fit_cox_scores.index) == set(labeled.index)
    assert xgboost_final.xgb_feature_columns == feature_columns + ["cox_oof_score"]

    # sentetik "harici" (UCSF-tarzi) cerceve -- AYNI (yumusak) dagilim,
    # FARKLI seed.
    external_frame, _ = _make_synthetic_gentle_hazard_frame(n=70, n_features=5, seed=999)

    evaluation = evaluate_xgboost_external_test(
        cox,
        cox_final_features,
        xgboost_final,
        external_frame,
        n_bootstrap_ci=100,
        seed=5,
    )

    assert 0.0 <= evaluation.auc["auc"] <= 1.0
    assert evaluation.auc["ci_lower"] <= evaluation.auc["auc"] <= evaluation.auc["ci_upper"]
    assert evaluation.n_patients_evaluated == len(external_frame) - evaluation.n_excluded_ambiguous_censoring
    assert evaluation.n_patients_evaluated > 0


def test_evaluate_xgboost_external_test_raises_on_missing_feature_columns() -> None:
    from lifelines import CoxPHFitter

    labeled, feature_columns = _make_synthetic_labeled_frame(n=120, n_features=4, seed=92)
    cox_final_features = feature_columns[:1]
    cox = CoxPHFitter(penalizer=0.3, l1_ratio=1.0)
    cox.fit(
        labeled[cox_final_features + ["survival_days", "event"]],
        duration_col="survival_days",
        event_col="event",
    )
    xgboost_final = fit_final_cox_and_xgboost_pipeline_on_full_pool(
        labeled,
        feature_columns,
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_inner_splits=3,
        cox_n_bootstrap_stability=20,
        cox_cross_fit_splits=3,
        seed=5,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
    )

    incomplete_external = pd.DataFrame(
        {
            "survival_days": [400.0, 100.0],
            "event": [0, 1],
            feature_columns[0]: [0.1, 0.2],
            # digger feature kolonlari KASITLI eksik
        },
        index=["Q0001", "Q0002"],
    )

    with pytest.raises(ValueError, match="icermiyor"):
        evaluate_xgboost_external_test(
            cox, cox_final_features, xgboost_final, incomplete_external, n_bootstrap_ci=50, seed=5
        )


# =====================================================================
# 3. TUR (task-mt06mqzi-k0gb5j) -- CRITICAL full-pool secim sizintisi
# duzeltmesinin testleri
# =====================================================================


def test_full_pool_cross_fit_score_unaffected_by_own_survival_data_perturbation() -> None:
    """3. tur test #3 (Codex, task-mt06mqzi-k0gb5j) kirmizi/yesil kanit:
    CRITICAL duzeltmesinden SONRA full-pool cross-fit meta-skorlarinda
    da (aligned pipeline'in ic cross-fit'iyle AYNI disiplin) bir HASTANIN
    KENDI verisini (survival_days) sertce bozunca, O HASTANIN KENDI
    cross-fit Cox skoru DEGISMEMELI -- artik SECIM de (fit'in kendisi
    ONCEDEN de fold-yerel'di) fold-yerel oldugu icin."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=160, n_features=5, seed=141)

    kwargs = dict(
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_inner_splits=3,
        cox_n_bootstrap_stability=20,
        cox_cross_fit_splits=4,
        seed=17,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
    )

    result_original = fit_final_cox_and_xgboost_pipeline_on_full_pool(labeled, feature_columns, **kwargs)

    target_patient = result_original.cross_fit_split_assignment.index[0]
    perturbed = labeled.copy()
    perturbed.loc[target_patient, "survival_days"] = float(perturbed["survival_days"].max()) * 5.0 + 999.0

    result_perturbed = fit_final_cox_and_xgboost_pipeline_on_full_pool(perturbed, feature_columns, **kwargs)

    # on-kosul: event_col degismedigi icin split ataması (StratifiedKFold
    # event_col'a gore stratifiye eder) IKI kosuda da AYNI olmali.
    assert (
        result_original.cross_fit_split_assignment[target_patient]
        == result_perturbed.cross_fit_split_assignment[target_patient]
    )
    assert result_original.cross_fit_cox_scores[target_patient] == pytest.approx(
        result_perturbed.cross_fit_cox_scores[target_patient], rel=1e-9, abs=1e-9
    )


def test_full_pool_reduce_collinearity_recomputed_once_per_cross_fit_split() -> None:
    """B9 -- `fit_final_cox_and_xgboost_pipeline_on_full_pool()` (nihai/
    dagitilabilir model icin full-pool cross-fit) `train_xgboost_with_
    fold_aligned_cox_scores()` ile AYNI kardes mekanizmayi tasir: `reduce_
    collinearity=True` iken `build_v3_candidate_pool()` HER cross-fit
    split'inde SADECE o split'in KENDI `inner_train`'inden YENIDEN
    cagrilir -- cagri sayisi TAM OLARAK `cox_cross_fit_splits` olmali."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=160, n_features=5, seed=141)
    cox_cross_fit_splits = 4

    real_build_pool = xgboost_model_module.build_v3_candidate_pool
    call_log: list[int] = []

    def _counting_wrapper(*args, **kwargs):
        call_log.append(1)
        return real_build_pool(*args, **kwargs)

    with patch.object(xgboost_model_module, "build_v3_candidate_pool", side_effect=_counting_wrapper):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_final_cox_and_xgboost_pipeline_on_full_pool(
                labeled,
                feature_columns,
                cox_l1_ratio_grid=(1.0,),
                cox_penalizer_grid=(0.3,),
                cox_inner_splits=3,
                cox_n_bootstrap_stability=20,
                cox_cross_fit_splits=cox_cross_fit_splits,
                seed=17,
                xgb_inner_splits=3,
                xgb_seed=9,
                xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
                reduce_collinearity=True,
            )

    assert len(call_log) == cox_cross_fit_splits, (
        f"beklenen {cox_cross_fit_splits} cagri (cox_cross_fit_splits), "
        f"gercek={len(call_log)} -- fold-yerellik BOZULMUS olabilir."
    )
    assert isinstance(result, FinalXGBoostPipelineResult)


def test_full_pool_reduce_collinearity_default_off_matches_legacy_behavior() -> None:
    """B9 -- `fit_final_cox_and_xgboost_pipeline_on_full_pool()` icin de
    varsayilan (`reduce_collinearity` verilmemis) davranis, ACIKCA `False`
    verilmesiyle bit-birebir AYNI olmali."""

    labeled, feature_columns = _make_synthetic_labeled_frame(n=160, n_features=5, seed=141)
    kwargs = dict(
        cox_l1_ratio_grid=(1.0,),
        cox_penalizer_grid=(0.3,),
        cox_inner_splits=3,
        cox_n_bootstrap_stability=20,
        cox_cross_fit_splits=4,
        seed=17,
        xgb_inner_splits=3,
        xgb_seed=9,
        xgb_param_grid=_SMALL_XGB_GRID_ALIGNED,
    )

    result_implicit = fit_final_cox_and_xgboost_pipeline_on_full_pool(labeled, feature_columns, **kwargs)
    result_explicit_false = fit_final_cox_and_xgboost_pipeline_on_full_pool(
        labeled, feature_columns, reduce_collinearity=False, **kwargs
    )

    pd.testing.assert_series_equal(
        result_implicit.cross_fit_cox_scores.sort_index(),
        result_explicit_false.cross_fit_cox_scores.sort_index(),
    )


def test_verify_full_pool_pipeline_leakage_free_catches_injected_leak() -> None:
    """`verify_full_pool_pipeline_leakage_free()`'in KENDISI, KASITLI
    olarak bozulmus (bir split'in fit kumesi, o split'in KENDI test/
    skorlanan hastasini iceren) bir sonucu yakalamali."""

    leaked_provenance = [
        CoxFitProvenanceRecord(
            outer_fold=None,
            fit_type="full_pool_final",
            inner_fold=0,
            # P0003 split 0'in KENDI test/skorlanan hastasi (asagida) --
            # AMA burada split 0'in FIT kumesine de SIZMIS.
            patient_ids=frozenset({"P0001", "P0002", "P0003"}),
            best_penalizer=0.3,
            best_l1_ratio=1.0,
            n_final_features=1,
        ),
        CoxFitProvenanceRecord(
            outer_fold=None,
            fit_type="full_pool_final",
            inner_fold=1,
            patient_ids=frozenset({"P0003"}),
            best_penalizer=0.3,
            best_l1_ratio=1.0,
            n_final_features=1,
        ),
    ]
    split_assignment = pd.Series({"P0001": 1, "P0002": 1, "P0003": 0, "P0004": 0})
    bad_result = FinalXGBoostPipelineResult(
        fitted_model=None,
        best_params={},
        xgb_feature_columns=[],
        cross_fit_cox_scores=pd.Series({"P0001": 0.1, "P0002": 0.2, "P0003": 0.3, "P0004": 0.4}),
        score_column="cox_oof_score",
        n_patients_used=4,
        n_cross_fit_splits=2,
        cross_fit_split_assignment=split_assignment,
        fit_provenance=leaked_provenance,
    )

    with pytest.raises(OutOfFoldLeakageError):
        verify_full_pool_pipeline_leakage_free(
            bad_result, pd.Index(["P0001", "P0002", "P0003", "P0004"])
        )


def test_verify_full_pool_pipeline_leakage_free_accepts_clean_result() -> None:
    """Temiz (kesisimsiz) bir `FinalXGBoostPipelineResult` icin
    `verify_full_pool_pipeline_leakage_free()` SESSIZCE gecmeli --
    yanlis-pozitif URETMEDIGININ kaniti."""

    # split_assignment: split 0 test={P0003,P0004}, split 1 test={P0001,P0002}
    # -- her split'in FIT (train) kumesi o split'in test kumesinin
    # TAMLAYANI olmali (temiz/sizintisiz cross-fit).
    clean_provenance = [
        CoxFitProvenanceRecord(
            outer_fold=None,
            fit_type="full_pool_final",
            inner_fold=0,
            patient_ids=frozenset({"P0001", "P0002"}),  # split0 test'inin TAMLAYANI
            best_penalizer=0.3,
            best_l1_ratio=1.0,
            n_final_features=1,
        ),
        CoxFitProvenanceRecord(
            outer_fold=None,
            fit_type="full_pool_final",
            inner_fold=1,
            patient_ids=frozenset({"P0003", "P0004"}),  # split1 test'inin TAMLAYANI
            best_penalizer=0.3,
            best_l1_ratio=1.0,
            n_final_features=1,
        ),
    ]
    split_assignment = pd.Series({"P0001": 1, "P0002": 1, "P0003": 0, "P0004": 0})
    clean_result = FinalXGBoostPipelineResult(
        fitted_model=None,
        best_params={},
        xgb_feature_columns=[],
        cross_fit_cox_scores=pd.Series({"P0001": 0.1, "P0002": 0.2, "P0003": 0.3, "P0004": 0.4}),
        score_column="cox_oof_score",
        n_patients_used=4,
        n_cross_fit_splits=2,
        cross_fit_split_assignment=split_assignment,
        fit_provenance=clean_provenance,
    )

    verify_full_pool_pipeline_leakage_free(
        clean_result, pd.Index(["P0001", "P0002", "P0003", "P0004"])
    )
