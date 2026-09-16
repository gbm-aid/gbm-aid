"""Hafta 4 -- XGBoost egitim orkestrasyonu (out-of-fold Cox skoru ->
XGBoost 12-ay sagkalim siniflandirmasi).

🔴 KRITIK ON-KOSUL -- NIHAI COX VARYANTI HENUZ SECILMEDI. Bu script
`tools/train_cox_week3.py --variants ...` (V2 dort-varyant kosusu, bu
yazi itibariyle CALISIYOR) BITTIKTEN ve Baris hangi varyantin (v1_referans/
v2a_mgmt/v2b_mgmt_spline/v2c_mgmt_spline_wttc) nihai model oldugunu
SECTIKTEN SONRA calistirilmalidir. Bu script su an DB'ye karsi
CALISTIRILMADI -- yalnizca kod + `tests/test_train_xgboost_week4.py`
(sentetik/fixture, DB'siz) hazir.

MIMARI -- iki katman:
  1. `run_xgboost_week4_pipeline()` -- DB'DEN BAGIMSIZ cekirdek. Girdi
     olarak ZATEN kurulmus bir Cox egitim cercevesi (`training_frame` +
     `feature_columns` + `extra_columns` + ...) alir, TUM Hafta 4
     zincirini (`pipeline.cox_model.run_nested_cv()` -> `pipeline.
     xgboost_model.generate_out_of_fold_cox_scores()` -> sizinti
     dogrulama -> 12-ay hedef -> XGBoost nested-CV -> donmus risk
     esikleri) calistirir. `tests/test_train_xgboost_week4.py` bu
     fonksiyonu SENTETIK verilerle dogrudan test eder (DB'siz) --
     `tools/train_cox_week3.py::run_pipeline()`'in DB-bagimsiz govde
     deseniyle AYNI.

     🔴 2026-08-19 GUNCELLEME (Codex capraz inceleme, task-mszuse9v-
     efm1gw, CRITICAL 1) -- BU FONKSIYON ARTIK URETIM/DOGRU YOL
     DEGILDIR. `run_nested_cv()`'nin Cox-KENDI dis-fold bolunmesi
     (`event_col`'a gore stratified) ile `train_xgboost_nested_cv()`'nin
     KENDI AYRI dis-fold bolunmesi (`target_col`'a gore stratified)
     HIZALI DEGIL -- bir XGBoost test hastasinin Cox skoru "durust"
     olsa BILE, o hastanin verisi BASKA bir hastanin Cox skorunu ureten
     BIR BASKA Cox-fold'un egitim kumesinde bulunabilir (iki fold yapisi
     BAGIMSIZ oldugu icin), bu da o baska hastanin skoru XGBoost'un
     EGITIM kumesindeyken DOLAYLI bir sizintiya yol acar. BU FONKSIYON
     SADECE Ege'nin sizinti-karsilastirma gorevi (plan.txt satir 508)
     icin KORUNUYOR -- YENI birincil/uretim yolu
     `run_xgboost_week4_pipeline_aligned()`'dir (asagida), tek, ORTAK
     dis-fold dongusu kullanir (`pipeline.xgboost_model.train_xgboost_
     with_fold_aligned_cox_scores()`).
  2. `run_xgboost_week4_pipeline_aligned()` -- DB'DEN BAGIMSIZ, YENI
     BIRINCIL cekirdek (CRITICAL 1 duzeltmesi). Cox VE XGBoost'un TEK,
     ORTAK dis-fold dongusunu (`pipeline.xgboost_model.train_xgboost_
     with_fold_aligned_cox_scores()`) kullanir -- bkz. o fonksiyonun
     docstring'i. HIGH 3 (RCS yas dugumlerinin fold-disi hesaplanmasi)
     duzeltmesini de bu yol tasir: `_build_fold_safe_age_spline_columns_
     builder()` ile yas dugumleri HER dis-fold'da SADECE o fold'un
     egitim yaslarindan hesaplanir.
  3. `main()`/CLI -- DB'den C32 verisini ceker (`tools/train_cox_week3.py`
     ile BIREBIR AYNI fonksiyonlari -- `fetch_c32_radiomics_long_frame`,
     `fetch_patients_frame`, `pivot_radiomics_long_to_wide`, klinik
     kovaryat insa fonksiyonlari -- import edip CAGIRIR, KOPYALAMAZ),
     secilen varyantin training_frame'ini `tools/train_cox_week3.py::
     run_single_variant()` ile AYNI recete ile yeniden kurar, sonra
     VARSAYILAN olarak `run_xgboost_week4_pipeline_aligned()`'i cagirir
     ve `artifacts/week4/xgboost_model/` altina yazar. `--legacy-
     misaligned-pipeline` bayragiyla eski (yukaridaki 1. maddedeki,
     ARTIK ONERILMEYEN) yol da hala calistirilabilir (Ege'nin
     karsilastirmasi icin).

`tools/train_cox_week3.py`/`pipeline/cox_model.py`'ye TEK SATIR bile
YAZILMADI -- SADECE import + cagri (salt-okunur kullanim, CLAUDE.md'nin
"YALNIZ OKU, DEGISTIRME" kuralina uygun).

SIZINTI GARANTISI -- bkz. `pipeline/xgboost_model.py` modul docstring'i
(BÖLÜM 1 + BÖLÜM 6). Bu script'in KENDI EKLEDIGI tek ek kontrol: hem
`run_xgboost_week4_pipeline()` hem `run_xgboost_week4_pipeline_aligned()`
"DOGRU" (out-of-fold/fold-hizali Cox skoru + XGBoost'un kendi durust
nested-CV'si) VE "YANLIS" (in-sample/sizintili Cox skoru XGBoost girdisi
olarak, Ege'nin gorevi icin) sonucu AYNI cagrida uretir -- boylece rapor
HER IKI sayiyi da YAN YANA gosterebilir (plan.txt satir 508).

MEDIUM 5 (Codex, task-mszuse9v-efm1gw) -- UCSF HARICI DEGERLENDIRME:
`run_ucsf_external_evaluation()` (DB'ye bagli, `--evaluate-ucsf-external`
bayragiyla tetiklenir) yalniz-UPenn'de fit edilen NIHAI Cox->XGBoost
zincirini (`pipeline.xgboost_model.fit_final_cox_and_xgboost_pipeline_
on_full_pool()`), YENIDEN FIT/ESIK SECIMI YAPMADAN, UCSF 295/169'da
(`pipeline.xgboost_model.evaluate_xgboost_external_test()`) degerlendirir.
Bu gorev talimati GEREGI URETIM KOSUSU YAPILMADI -- SADECE kod +
`tests/test_train_xgboost_week4.py`'deki sentetik/DB'siz testler hazir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from pipeline.cox_model import (  # noqa: E402
    STABLE_FEATURES_ICC60,
    build_training_frame,
    pivot_radiomics_long_to_wide,
    run_nested_cv,
    standardize_columns_fold_safe,
)
import pipeline.reduce_collinearity as reduce_collinearity_module  # noqa: E402
from pipeline.xgboost_model import (  # noqa: E402
    DEFAULT_COX_PENALIZER_ESCALATION_GRID,
    DEFAULT_TWELVE_MONTH_THRESHOLD_DAYS,
    DEFAULT_XGB_PARAM_GRID,
    AlignedCoxXGBoostResult,
    CoxFitFailureRecord,
    CoxPenalizerEscalationExhaustedError,
    ExternalXGBoostEvaluationResult,
    assert_no_clinical_standardize_feature_overlap,
    FinalXGBoostPipelineResult,
    FrozenRiskThresholds,
    OutOfFoldCoxScoreResult,
    TwelveMonthLabelReport,
    XGBoostNestedCVResult,
    apply_twelve_month_target,
    assign_risk_class_column,
    build_xgboost_feature_frame,
    compute_frozen_risk_thresholds,
    define_twelve_month_survival_target,
    evaluate_xgboost_external_test,
    fit_final_cox_and_xgboost_pipeline_on_full_pool,
    generate_in_sample_cox_scores,
    generate_in_sample_xgboost_auc,
    generate_out_of_fold_cox_scores,
    hash_feature_columns,
    summarize_excluded_censoring_bias,
    train_xgboost_nested_cv,
    train_xgboost_with_fold_aligned_cox_scores,
    verify_aligned_fold_leakage_free,
    verify_full_pool_pipeline_leakage_free,
    verify_out_of_fold_leakage_free,
)

import train_cox_week3 as week3  # noqa: E402  (sys.path ayari sonrasi -- SALT OKUNUR import)


# =====================================================================
# BÖLÜM 1 -- DB'den bağımsız çekirdek (sentetik testlerin doğrudan
# çağırdığı, gerçek üretim mantığının TAMAMI)
# =====================================================================


@dataclass
class Week4PipelineResult:
    cox_nested_cv_result: pd.DataFrame
    cox_oof_result: OutOfFoldCoxScoreResult
    cox_in_sample_score: pd.Series
    cox_in_sample_features: list[str]
    twelve_month_report: TwelveMonthLabelReport
    censoring_bias_summary: pd.DataFrame | None
    xgboost_result_correct: XGBoostNestedCVResult
    xgboost_result_leaky_cox_input: XGBoostNestedCVResult
    xgboost_auc_fully_in_sample: dict[str, Any]
    risk_thresholds: FrozenRiskThresholds
    risk_class_out_of_fold: pd.Series
    n_patients_before_target_exclusion: int
    n_patients_after_target_exclusion: int


def run_xgboost_week4_pipeline(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    duration_col: str = "survival_days",
    event_col: str = "event",
    extra_columns: list[str] | None = None,
    clinical_standardize_columns: list[str] | None = None,
    twelve_month_threshold_days: float = DEFAULT_TWELVE_MONTH_THRESHOLD_DAYS,
    censoring_bias_continuous_columns: list[str] | None = None,
    censoring_bias_binary_columns: list[str] | None = None,
    # Cox nested-CV parametreleri (run_nested_cv() ile AYNI varsayılanlar)
    cox_outer_splits: int = 5,
    cox_inner_splits: int = 5,
    cox_seed: int = 42,
    cox_l1_ratio_grid: tuple[float, ...] | None = None,
    cox_penalizer_grid: tuple[float, ...] | None = None,
    cox_n_bootstrap_stability: int = 200,
    cox_stability_frequency_threshold: float = 0.6,
    cox_extra_column_penalizer: float | None = None,
    # XGBoost nested-CV parametreleri
    xgb_outer_splits: int = 5,
    xgb_inner_splits: int = 5,
    xgb_seed: int = 42,
    xgb_param_grid: dict[str, list[Any]] | None = None,
    xgb_n_bootstrap_ci: int = 1000,
) -> Week4PipelineResult:
    """Hafta 4'ün TAM zinciri -- DB'den bağımsız, sentetik/gerçek her
    `training_frame` ile çalışır.

    ADIMLAR (sırayla):
      1. `run_nested_cv()` -- Cox'un KENDİ dış-fold sonucu (ÖZELLİK
         SEÇİMİ + hiperparametre + fold-başına C-index).
      2. `generate_out_of_fold_cox_scores()` -- (1)'in fold bölünmesiyle
         BİREBİR AYNı bölünmeyi reconstruct edip her hastanın "dürüst"
         Cox skorunu üretir + `verify_out_of_fold_leakage_free()` ile
         BAĞIMSIZ doğrulanır.
      3. `generate_in_sample_cox_scores()` -- ⚠️ SADECE karşılaştırma
         için, ASLA üretim girdisi değil.
      4. `define_twelve_month_survival_target()` + `apply_twelve_month_
         target()` -- 12 aydan önce sansürlenen (belirsiz) hastalar
         SAYILARAK düşürülür.
      5. `build_xgboost_feature_frame()` iki kez -- (a) doğru: out-of-
         fold Cox skoru, (b) yanlış/sızıntılı: in-sample Cox skoru.
      6. `train_xgboost_nested_cv()` -- (a) ve (b) çerçeveleri için AYRI
         AYRI (XGBoost'un KENDİ nested-CV'si HER İKİSİNDE de dürüst --
         bu, "jüri sorusu"nu (Cox skorunun in-sample/out-of-fold
         olmasının etkisini) İZOLE eder).
      7. `generate_in_sample_xgboost_auc()` -- (a) çerçevesi üzerinde,
         XGBoost'un KENDİSİ de tüm veriyi görüp tüm veriyi skorlarsa
         (en uç/en basit "yanlış AUC" -- plan.txt'nin literal istediği
         karşılaştırma).
      8. `compute_frozen_risk_thresholds()` -- (a)'nın out-of-fold
         tahmini olasılıklarından (in-sample DEĞİL) donmuş tertile.
    """

    extra_columns = list(extra_columns or [])
    clinical_standardize_columns = list(clinical_standardize_columns or [])
    l1_ratio_grid = cox_l1_ratio_grid  # None ise run_nested_cv kendi varsayılanını kullanır
    penalizer_grid = cox_penalizer_grid

    nested_cv_kwargs: dict[str, Any] = dict(
        duration_col=duration_col,
        event_col=event_col,
        extra_columns=extra_columns,
        outer_splits=cox_outer_splits,
        inner_splits=cox_inner_splits,
        seed=cox_seed,
        stability_selection=True,
        n_bootstrap_stability=cox_n_bootstrap_stability,
        stability_frequency_threshold=cox_stability_frequency_threshold,
        extra_column_penalizer=cox_extra_column_penalizer,
        clinical_standardize_columns=clinical_standardize_columns,
    )
    if l1_ratio_grid is not None:
        nested_cv_kwargs["l1_ratio_grid"] = l1_ratio_grid
    if penalizer_grid is not None:
        nested_cv_kwargs["penalizer_grid"] = penalizer_grid

    cox_nested_cv_result = run_nested_cv(training_frame, feature_columns, **nested_cv_kwargs)

    cox_oof_result = generate_out_of_fold_cox_scores(
        training_frame,
        feature_columns,
        cox_nested_cv_result,
        duration_col=duration_col,
        event_col=event_col,
        extra_columns=extra_columns,
        outer_splits=cox_outer_splits,
        seed=cox_seed,
        extra_column_penalizer=cox_extra_column_penalizer,
        clinical_standardize_columns=clinical_standardize_columns,
        stability_selection=True,
        n_bootstrap_stability=cox_n_bootstrap_stability,
        stability_frequency_threshold=cox_stability_frequency_threshold,
    )
    verify_out_of_fold_leakage_free(cox_oof_result)

    # ⚠️ SADECE karsilastirma -- en son fold'un best_penalizer/best_l1_ratio'su
    # kullanilir (tek bir sabit deger gerekiyor, in-sample fit tanimi geregi
    # tum havuzda TEK bir hiperparametre secimi yapmak zorunda -- gercek
    # uretimde bu deger `fit_final_model_on_full_pool()`'un kendi ic-CV'siyle
    # secilir, burada BASITLESTIRILDI cunku bu ciktinin KENDISI zaten
    # uretimde kullanilmayacak).
    last_fold = cox_nested_cv_result.iloc[-1]
    cox_in_sample_score, cox_in_sample_features = generate_in_sample_cox_scores(
        training_frame,
        feature_columns,
        duration_col=duration_col,
        event_col=event_col,
        extra_columns=extra_columns,
        penalizer=float(last_fold["best_penalizer"]),
        l1_ratio=float(last_fold["best_l1_ratio"]),
        extra_column_penalizer=cox_extra_column_penalizer,
        stability_selection=True,
        n_bootstrap_stability=cox_n_bootstrap_stability,
        stability_frequency_threshold=cox_stability_frequency_threshold,
        seed=cox_seed,
    )

    target, twelve_month_report = define_twelve_month_survival_target(
        training_frame,
        duration_col=duration_col,
        event_col=event_col,
        threshold_days=twelve_month_threshold_days,
    )

    censoring_bias_summary = None
    if censoring_bias_continuous_columns or censoring_bias_binary_columns:
        ambiguous_mask = target.isna()
        censoring_bias_summary = summarize_excluded_censoring_bias(
            training_frame,
            ambiguous_mask,
            continuous_columns=censoring_bias_continuous_columns,
            binary_columns=censoring_bias_binary_columns,
        )

    labeled_frame = apply_twelve_month_target(training_frame, target, report=twelve_month_report)

    correct_feature_frame = build_xgboost_feature_frame(
        cox_oof_result.scores[cox_oof_result.score_column],
        labeled_frame[feature_columns + extra_columns],
        score_column="cox_score",
    )
    correct_feature_frame["target_12mo_survival"] = labeled_frame["target_12mo_survival"]

    leaky_feature_frame = build_xgboost_feature_frame(
        cox_in_sample_score,
        labeled_frame[feature_columns + extra_columns],
        score_column="cox_score",
    )
    leaky_feature_frame["target_12mo_survival"] = labeled_frame["target_12mo_survival"]

    xgb_feature_columns = feature_columns + extra_columns + ["cox_score"]
    xgb_grid = xgb_param_grid or DEFAULT_XGB_PARAM_GRID

    xgboost_result_correct = train_xgboost_nested_cv(
        correct_feature_frame,
        xgb_feature_columns,
        target_col="target_12mo_survival",
        outer_splits=xgb_outer_splits,
        inner_splits=xgb_inner_splits,
        seed=xgb_seed,
        param_grid=xgb_grid,
        n_bootstrap_ci=xgb_n_bootstrap_ci,
    )
    xgboost_result_leaky_cox_input = train_xgboost_nested_cv(
        leaky_feature_frame,
        xgb_feature_columns,
        target_col="target_12mo_survival",
        outer_splits=xgb_outer_splits,
        inner_splits=xgb_inner_splits,
        seed=xgb_seed,
        param_grid=xgb_grid,
        n_bootstrap_ci=xgb_n_bootstrap_ci,
    )

    # en uc/en basit "yanlis AUC" -- XGBoost dogru (out-of-fold Cox skorlu)
    # cerceve UZERINDE hem egitilir hem SKORLANIR (tam in-sample).
    best_fold_params = xgboost_result_correct.fold_results.iloc[-1]["best_params"]
    xgboost_auc_fully_in_sample = generate_in_sample_xgboost_auc(
        correct_feature_frame,
        xgb_feature_columns,
        target_col="target_12mo_survival",
        params=best_fold_params,
        seed=xgb_seed,
        n_bootstrap_ci=xgb_n_bootstrap_ci,
    )

    risk_thresholds = compute_frozen_risk_thresholds(
        xgboost_result_correct.out_of_fold_predictions["predicted_probability"]
    )
    risk_class_out_of_fold = assign_risk_class_column(
        xgboost_result_correct.out_of_fold_predictions["predicted_probability"], risk_thresholds
    )

    return Week4PipelineResult(
        cox_nested_cv_result=cox_nested_cv_result,
        cox_oof_result=cox_oof_result,
        cox_in_sample_score=cox_in_sample_score,
        cox_in_sample_features=cox_in_sample_features,
        twelve_month_report=twelve_month_report,
        censoring_bias_summary=censoring_bias_summary,
        xgboost_result_correct=xgboost_result_correct,
        xgboost_result_leaky_cox_input=xgboost_result_leaky_cox_input,
        xgboost_auc_fully_in_sample=xgboost_auc_fully_in_sample,
        risk_thresholds=risk_thresholds,
        risk_class_out_of_fold=risk_class_out_of_fold,
        n_patients_before_target_exclusion=twelve_month_report.n_total,
        n_patients_after_target_exclusion=len(labeled_frame),
    )


# =====================================================================
# BÖLÜM 1.5 -- CRITICAL 1 düzeltmesi: Cox+XGBoost'un TEK, ORTAK dış-fold
# döngüsü (YENİ BİRİNCİL/ÜRETİM yolu, 2026-08-19 -- Codex çapraz inceleme
# task-mszuse9v-efm1gw)
# =====================================================================


@dataclass
class Week4AlignedPipelineResult:
    twelve_month_report: TwelveMonthLabelReport
    censoring_bias_summary: pd.DataFrame | None
    aligned_result: AlignedCoxXGBoostResult
    risk_thresholds: FrozenRiskThresholds
    risk_class_out_of_fold: pd.Series
    n_patients_before_target_exclusion: int
    n_patients_after_target_exclusion: int


def run_xgboost_week4_pipeline_aligned(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    duration_col: str = "survival_days",
    event_col: str = "event",
    extra_columns: list[str] | None = None,
    clinical_standardize_columns: list[str] | None = None,
    fold_local_extra_column_builder: Any = None,
    twelve_month_threshold_days: float = DEFAULT_TWELVE_MONTH_THRESHOLD_DAYS,
    censoring_bias_continuous_columns: list[str] | None = None,
    censoring_bias_binary_columns: list[str] | None = None,
    outer_splits: int = 5,
    seed: int = 42,
    cox_inner_splits: int = 5,
    cox_l1_ratio_grid: tuple[float, ...] | None = None,
    cox_penalizer_grid: tuple[float, ...] | None = None,
    cox_n_bootstrap_stability: int = 200,
    cox_stability_frequency_threshold: float = 0.6,
    cox_extra_column_penalizer: float | None = None,
    reduce_collinearity: bool = False,
    collinearity_cv_threshold: float = reduce_collinearity_module.DEFAULT_CV_THRESHOLD,
    collinearity_corr_threshold: float = reduce_collinearity_module.DEFAULT_CORRELATION_CLUSTER_THRESHOLD,
    xgb_inner_splits: int = 5,
    xgb_seed: int = 42,
    xgb_param_grid: dict[str, list[Any]] | None = None,
    xgb_n_bootstrap_ci: int = 1000,
    cox_penalizer_escalation_grid: tuple[float, ...] | None = None,
    allow_fold_skip: bool = False,
    cox_fit_failure_log: list[CoxFitFailureRecord] | None = None,
) -> Week4AlignedPipelineResult:
    """CRITICAL 1 düzeltmesinin ÜRETİM/birincil orkestrasyonu -- Cox VE
    XGBoost'un TEK, ORTAK dış-fold döngüsünü (`pipeline.xgboost_model.
    train_xgboost_with_fold_aligned_cox_scores()`) çağırır. `run_xgboost_
    week4_pipeline()` (yukarıda, ESKİ/HİZALI OLMAYAN yol -- SADECE Ege'nin
    sızıntı-karşılaştırması için korunuyor) ile temel fark: Cox'un KENDİ
    ayrı `run_nested_cv()` çağrısı YOK -- Cox VE XGBoost AYNI dış-fold
    bölünmesini (`target_col`'a göre stratified) paylaşır, bu yüzden bir
    XGBoost dış-test hastasının verisi, o fold'un XGBoost eğitim
    kümesindeki HİÇBİR hastanın Cox skorunu üreten hiçbir Cox fit'inde
    YER ALAMAZ (bkz. `pipeline/xgboost_model.py` BÖLÜM 6 -- yapısal
    kanıt `verify_aligned_fold_leakage_free()` ile burada da AYRICA
    doğrulanır).

    `fold_local_extra_column_builder` (HIGH 3 düzeltmesi -- ör. fold-
    güvenli RCS yaş düğümleri, bkz. `_build_fold_safe_age_spline_
    columns_builder()`): verilirse `train_xgboost_with_fold_aligned_
    cox_scores()`'a AYNEN iletilir.

    `reduce_collinearity`/`collinearity_cv_threshold`/`collinearity_
    corr_threshold` (B9, 2026-08-28 EKLENDİ -- VARSAYILAN `reduce_
    collinearity=False`, davranış değişikliği YOK): AYNEN `pipeline.
    xgboost_model.train_xgboost_with_fold_aligned_cox_scores()`'a
    iletilir -- bkz. o fonksiyonun docstring'i (2026-08-19 production
    çöküşünün -- `artifacts/week4/xgboost_v1_aligned/production.log`,
    `lifelines.exceptions.ConvergenceError: ... Matrix is singular.`
    -- kök nedenine bağlanan, HANGİ Cox varyantı/kolu seçilirse
    seçilsin model-agnostik bir opsiyonel filtre)."""

    extra_columns = list(extra_columns or [])
    clinical_standardize_columns = list(clinical_standardize_columns or [])

    target, twelve_month_report = define_twelve_month_survival_target(
        training_frame,
        duration_col=duration_col,
        event_col=event_col,
        threshold_days=twelve_month_threshold_days,
    )

    censoring_bias_summary = None
    if censoring_bias_continuous_columns or censoring_bias_binary_columns:
        ambiguous_mask = target.isna()
        censoring_bias_summary = summarize_excluded_censoring_bias(
            training_frame,
            ambiguous_mask,
            continuous_columns=censoring_bias_continuous_columns,
            binary_columns=censoring_bias_binary_columns,
        )

    labeled_frame = apply_twelve_month_target(training_frame, target, report=twelve_month_report)

    aligned_result = train_xgboost_with_fold_aligned_cox_scores(
        labeled_frame,
        feature_columns,
        duration_col=duration_col,
        event_col=event_col,
        target_col="target_12mo_survival",
        extra_columns=extra_columns,
        clinical_standardize_columns=clinical_standardize_columns,
        fold_local_extra_column_builder=fold_local_extra_column_builder,
        outer_splits=outer_splits,
        seed=seed,
        cox_inner_splits=cox_inner_splits,
        cox_l1_ratio_grid=cox_l1_ratio_grid,
        cox_penalizer_grid=cox_penalizer_grid,
        cox_n_bootstrap_stability=cox_n_bootstrap_stability,
        cox_stability_frequency_threshold=cox_stability_frequency_threshold,
        cox_extra_column_penalizer=cox_extra_column_penalizer,
        reduce_collinearity=reduce_collinearity,
        collinearity_cv_threshold=collinearity_cv_threshold,
        collinearity_corr_threshold=collinearity_corr_threshold,
        xgb_inner_splits=xgb_inner_splits,
        xgb_seed=xgb_seed,
        xgb_param_grid=xgb_param_grid,
        xgb_n_bootstrap_ci=xgb_n_bootstrap_ci,
        cox_penalizer_escalation_grid=cox_penalizer_escalation_grid,
        allow_fold_skip=allow_fold_skip,
        cox_fit_failure_log=cox_fit_failure_log,
    )
    verify_aligned_fold_leakage_free(aligned_result)

    risk_thresholds = compute_frozen_risk_thresholds(
        aligned_result.out_of_fold_predictions["predicted_probability"]
    )
    risk_class_out_of_fold = assign_risk_class_column(
        aligned_result.out_of_fold_predictions["predicted_probability"], risk_thresholds
    )

    return Week4AlignedPipelineResult(
        twelve_month_report=twelve_month_report,
        censoring_bias_summary=censoring_bias_summary,
        aligned_result=aligned_result,
        risk_thresholds=risk_thresholds,
        risk_class_out_of_fold=risk_class_out_of_fold,
        n_patients_before_target_exclusion=twelve_month_report.n_total,
        n_patients_after_target_exclusion=len(labeled_frame),
    )


def _build_fold_safe_age_spline_columns_builder(
    *, raw_age_column: str | None = None
) -> Any:
    """HIGH 3 düzeltmesi (Codex, task-mszuse9v-efm1gw): RCS yaş
    düğümleri artık `pipeline.xgboost_model.train_xgboost_with_fold_
    aligned_cox_scores()`'in HER dış-fold'unda, SADECE o fold'un
    `outer_train`'inin HAM yaşından hesaplanır (`week3.compute_rcs_
    knots()` -- salt-okunur import, `train_cox_week3.py`'ye TEK SATIR
    yazılmadı) -- `outer_test`'e AYNI (o fold'da ÖĞRENİLMİŞ) düğümler
    `week3.restricted_cubic_spline_basis()` ile uygulanır.

    Eski davranış (`_rebuild_variant_training_frame()`, TÜM UPenn'den
    TEK KEZ, nested-CV dışında -- `train_cox_week3.py` satır 735-747'de
    "AÇIK TASARIM KARARI (doğrulanmadı)" diye AÇIKÇA belgelenmiş) hâlâ
    MEVCUT/DEĞİŞMEDİ (Cox'un birincil zinciri hâlâ o dosyaya göre
    çalışıyor, bu görev `cox_model.py`/`train_cox_week3.py`'ye
    DOKUNMUYOR) -- bu SADECE `run_xgboost_week4_pipeline_aligned()`'in
    kullandığı YENİ, additive bir yoldur."""

    age_column = raw_age_column or week3.CLINICAL_AGE_COLUMN

    def _builder(
        outer_train_raw: pd.DataFrame, outer_test_raw: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        knots = week3.compute_rcs_knots(outer_train_raw[age_column].astype(float))
        train_spline = week3.restricted_cubic_spline_basis(outer_train_raw[age_column], knots)
        test_spline = week3.restricted_cubic_spline_basis(outer_test_raw[age_column], knots)

        outer_train_out = outer_train_raw.copy()
        outer_test_out = outer_test_raw.copy()
        outer_train_out[week3.CLINICAL_AGE_RCS1_COLUMN] = train_spline[week3.CLINICAL_AGE_RCS1_COLUMN]
        outer_train_out[week3.CLINICAL_AGE_RCS2_COLUMN] = train_spline[week3.CLINICAL_AGE_RCS2_COLUMN]
        outer_test_out[week3.CLINICAL_AGE_RCS1_COLUMN] = test_spline[week3.CLINICAL_AGE_RCS1_COLUMN]
        outer_test_out[week3.CLINICAL_AGE_RCS2_COLUMN] = test_spline[week3.CLINICAL_AGE_RCS2_COLUMN]
        return (
            outer_train_out,
            outer_test_out,
            [week3.CLINICAL_AGE_RCS1_COLUMN, week3.CLINICAL_AGE_RCS2_COLUMN],
        )

    return _builder


# =====================================================================
# BÖLÜM 2 -- DB'ye bağımlı CLI (tools/train_cox_week3.py'nin AYNI
# fonksiyonlarını salt-okunur olarak çağırır, hiçbir dosyaya yazmaz)
# =====================================================================


def _rebuild_variant_training_frame(
    *, cursor, config: "week3.VariantConfig", args: argparse.Namespace
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """`tools/train_cox_week3.py::run_single_variant()`'ın frame-kurma
    ÖN BÖLÜMÜYLE (pivot -> klinik kovaryat -> build_training_frame) AYNI
    reçeteyi uygular -- SADECE UPenn eğitim tarafı (UCSF harici test bu
    script'in kapsamında DEĞİL, bkz. modül docstring'i "açık risk").

    ⚠️ AÇIK RİSK: bu fonksiyon `run_single_variant()`'ın mantığını
    KOPYALAMIYOR, SADECE `train_cox_week3`'ün AYNI public fonksiyonlarını
    ÇAĞIRIYOR -- ama iki kod yolu bağımsız yaşadığı için `run_single_
    variant()`'ın gelecekte değişen bir adımı (ör. yeni bir klinik
    kovaryat) burada YANSIMAYABİLİR. Gerçek koşu öncesi bu fonksiyonun
    ürettiği `training_frame`'in `run_single_variant()`'ınkiyle (hasta
    sayısı/sütun kümesi) EŞLEŞTİĞİ AYRICA doğrulanmalı.
    """

    upenn_long = week3.fetch_c32_radiomics_long_frame(
        cursor, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
    )
    week3.raise_if_empty_c32(upenn_long, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32)
    upenn_patients = week3.fetch_patients_frame(cursor, source_name=week3.UPENN_SOURCE_NAME)

    upenn_wide, _ = pivot_radiomics_long_to_wide(upenn_long, regions=list(config.regions))

    feature_columns: list[str] = []
    for region in config.regions:
        feature_columns += week3.select_feature_columns(
            upenn_wide, STABLE_FEATURES_ICC60, region=region
        )

    extra_columns = week3.build_variant_clinical_extra_columns(
        include_mgmt=config.include_mgmt, use_age_spline=config.use_age_spline
    )
    age_knots = (
        week3.compute_rcs_knots(upenn_patients[week3.PATIENT_RAW_AGE_COLUMN].astype(float))
        if config.use_age_spline
        else None
    )
    upenn_clinical = week3.build_variant_clinical_frame(
        upenn_patients,
        include_mgmt=config.include_mgmt,
        use_age_spline=config.use_age_spline,
        age_knots=age_knots,
    )
    upenn_patients_c = upenn_patients.join(upenn_clinical)

    training_frame, training_report = build_training_frame(
        upenn_wide[feature_columns],
        upenn_patients_c,
        check_combat_identity=True,
        passthrough_columns=extra_columns,
    )
    n_train_events = int(training_frame["event"].sum())
    week3.check_training_pool_counts(
        training_report.n_output_rows,
        n_train_events,
        expected_patients=args.expected_patient_count,
        expected_events=args.expected_event_count,
        allow_mismatch=args.allow_unexpected_patient_count,
    )

    clinical_standardize_columns = (
        [week3.CLINICAL_AGE_RCS1_COLUMN, week3.CLINICAL_AGE_RCS2_COLUMN]
        if config.use_age_spline
        else [week3.CLINICAL_AGE_COLUMN]
    )

    return training_frame, feature_columns, extra_columns, clinical_standardize_columns


def _rebuild_variant_training_frame_for_aligned_pipeline(
    *, cursor, config: "week3.VariantConfig", args: argparse.Namespace
) -> tuple[pd.DataFrame, list[str], list[str], list[str], Any]:
    """`_rebuild_variant_training_frame()`'in HIGH 3 (Codex, task-
    mszuse9v-efm1gw) düzeltmesini taşıyan hâli -- `use_age_spline=True`
    olduğunda RCS düğümlerini TÜM UPenn'den TEK KEZ, nested-CV'nin
    DIŞINDA hesaplayıp `extra_columns`'a GÖMMEZ (eski fonksiyon böyle
    yapar, `train_cox_week3.py` satır 735-747'de AÇIK/doğrulanmamış bir
    tasarım kararı olarak belgelenmiştir). Bunun yerine HAM `clinical_age`
    kolonunu `training_frame`'de bırakır ve RCS dönüşümünü `pipeline.
    xgboost_model.train_xgboost_with_fold_aligned_cox_scores()`'in
    `fold_local_extra_column_builder`'ına ERTELER (bkz. `_build_fold_
    safe_age_spline_columns_builder()`) -- düğümler HER dış-fold'da
    SADECE o fold'un eğitim yaşından hesaplanır.

    `use_age_spline=False` varyantlarda davranış `_rebuild_variant_
    training_frame()` ile BİREBİR AYNI (RCS zaten yok, HAM yaşın fold-
    içi standardizasyonu zaten `standardize_columns_fold_safe()` ile
    güvenliydi -- HIGH 3 bu durumu ETKİLEMEZ).

    Döner: `(training_frame, feature_columns, extra_columns,
    clinical_standardize_columns, fold_local_extra_column_builder)` --
    `fold_local_extra_column_builder` `use_age_spline=False` iken
    `None`'dur (kanca gerekmez)."""

    upenn_long = week3.fetch_c32_radiomics_long_frame(
        cursor, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
    )
    week3.raise_if_empty_c32(upenn_long, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32)
    upenn_patients = week3.fetch_patients_frame(cursor, source_name=week3.UPENN_SOURCE_NAME)

    upenn_wide, _ = pivot_radiomics_long_to_wide(upenn_long, regions=list(config.regions))

    feature_columns: list[str] = []
    for region in config.regions:
        feature_columns += week3.select_feature_columns(
            upenn_wide, STABLE_FEATURES_ICC60, region=region
        )

    # HAM klinik cerceve -- use_age_spline NE OLURSA OLSUN False ile
    # cagrilir (RCS burada UYGULANMAZ, clinical_age HAM kalir).
    upenn_clinical = week3.build_variant_clinical_frame(
        upenn_patients,
        include_mgmt=config.include_mgmt,
        use_age_spline=False,
        age_knots=None,
    )
    upenn_patients_c = upenn_patients.join(upenn_clinical)

    # passthrough_columns (build_training_frame'e verilir) HER ZAMAN HAM
    # clinical_age + digerlerini icerir (RCS henuz yok).
    passthrough_columns = week3.build_variant_clinical_extra_columns(
        include_mgmt=config.include_mgmt, use_age_spline=False
    )

    training_frame, training_report = build_training_frame(
        upenn_wide[feature_columns],
        upenn_patients_c,
        check_combat_identity=True,
        passthrough_columns=passthrough_columns,
    )
    n_train_events = int(training_frame["event"].sum())
    week3.check_training_pool_counts(
        training_report.n_output_rows,
        n_train_events,
        expected_patients=args.expected_patient_count,
        expected_events=args.expected_event_count,
        allow_mismatch=args.allow_unexpected_patient_count,
    )

    if config.use_age_spline:
        # Modelleme-zamani extra_columns RCS ISIMLERINI kullanir (Cox/
        # XGBoost bunlari gorur) -- ama BU asamada training_frame'de
        # RCS kolonlari HENUZ YOK (fold_local_extra_column_builder her
        # fold'da EKLEYECEK). clinical_age (ham) extra_columns'TA
        # DEGIL -- sadece builder'in HAM YAS GIRDISI olarak
        # training_frame'de KALIYOR (Cox/XGBoost feature'i DEGIL).
        extra_columns = [
            column
            for column in week3.build_variant_clinical_extra_columns(
                include_mgmt=config.include_mgmt, use_age_spline=True
            )
            if column not in (week3.CLINICAL_AGE_RCS1_COLUMN, week3.CLINICAL_AGE_RCS2_COLUMN)
        ]
        clinical_standardize_columns = [week3.CLINICAL_AGE_RCS1_COLUMN, week3.CLINICAL_AGE_RCS2_COLUMN]
        fold_local_extra_column_builder = _build_fold_safe_age_spline_columns_builder(
            raw_age_column=week3.CLINICAL_AGE_COLUMN
        )
    else:
        extra_columns = list(passthrough_columns)
        clinical_standardize_columns = [week3.CLINICAL_AGE_COLUMN]
        fold_local_extra_column_builder = None

    return (
        training_frame,
        feature_columns,
        extra_columns,
        clinical_standardize_columns,
        fold_local_extra_column_builder,
    )


@dataclass
class UCSFExternalEvaluationBundle:
    """`run_ucsf_external_evaluation()`'ın çıktısı -- MEDIUM 5 (Codex)."""

    evaluation: ExternalXGBoostEvaluationResult
    cox_final_features: list[str]
    cox_best_penalizer: float
    cox_best_l1_ratio: float
    xgboost_best_params: dict[str, Any]
    n_upenn_training_patients: int
    n_upenn_training_events: int
    # HIGH 5 duzeltmesi (Codex 2. tur, task-mt04rec5-jjqc95): hangi
    # kolonlarin standardize edildigi + UPenn full-pool'dan OGRENILEN
    # ortalama/std -- final artifact'e kaydedilmek uzere (asagida
    # `_write_ucsf_external_outputs()` bunu JSON'a yazar).
    clinical_standardize_columns: list[str]
    clinical_standardize_means: dict[str, float]
    clinical_standardize_stds: dict[str, float]
    # Ş2/Ş3 (2026-09-11 EKLENDİ, varsayılanlı): `cox_final` (bu dosyanın
    # `_fit_final_model_on_full_pool_with_penalizer_escalation()` sarmalayıcısı
    # üzerinden) penalizer-eskalasyon zincirindeki HER BAŞARISIZ deneme.
    cox_final_fit_failure_log: list[CoxFitFailureRecord] = field(default_factory=list)
    # Ş4 (2026-09-11 EKLENDİ, varsayılanlı): iki Cox havuzu (cross-fit
    # `labeled_frame` -- XGBoost meta-skoru için -- vs dağıtım `training_
    # frame`/`full_pool_training_frame` -- `cox_final` için) AYNI reçete
    # DEĞİLDİR (bilinçli fark, bkz. `fit_final_cox_and_xgboost_pipeline_
    # on_full_pool()` docstring'i: "cox_final ARTIK bu fonksiyona
    # VERİLMİYOR"). Şeffaflık için hasta/olay sayıları AYRI raporlanır.
    n_cross_fit_training_patients: int = 0
    n_cross_fit_training_events: int = 0
    # v3 recete-sadakati (2026-09-12 EKLENDİ, şeffaflık): `cox_final`'in
    # GERÇEKTEN standardize ettiği tam kolon listesi (`reduce_collinearity
    # =True` iken `clinical_standardize_columns` + filtrelenmiş radyomik
    # havuz `cox_final_feature_columns` -- bkz. yukarıdaki tanımlandığı
    # blok) -- `clinical_standardize_columns` (üstteki alan) SADECE yaş/
    # RCS listesini taşır, bu YENİ alan olmadan hangi radyomiklerin de
    # standardize edildiği artifact'ten OKUNAMAZDI.
    cox_final_clinical_standardize_columns: list[str] = field(default_factory=list)


def _validate_ucsf_distribution_feature_pool_disjoint(
    feature_columns: list[str], clinical_standardize_columns: list[str]
) -> None:
    """Ş4 (Codex sartli onay, 2026-09-11) -- `run_ucsf_external_
    evaluation()`'un dagitim Cox havuzu (STANDARDIZE edilmis `full_pool_
    training_frame` uzerinden filtrelenir) ile aligned/cross-fit havuzu
    (HAM veri uzerinden filtrelenir, `pipeline.xgboost_model.train_
    xgboost_with_fold_aligned_cox_scores()`) arasindaki recete-farkinin
    SESSIZCE COGALMAMASI icin -- bkz. `pipeline.xgboost_model.
    CoxFeatureCandidatePoolOverlapError` docstring'i. DB'siz, saf
    fonksiyon -- ayri test edilebilir."""

    assert_no_clinical_standardize_feature_overlap(
        feature_columns,
        clinical_standardize_columns,
        context="run_ucsf_external_evaluation() (dagitim Cox havuzu)",
    )


def _fit_final_model_on_full_pool_with_penalizer_escalation(
    full_pool_training_frame: pd.DataFrame,
    cox_final_feature_columns: list[str],
    *,
    extra_columns: list[str],
    n_bootstrap_stability: int,
    stability_frequency_threshold: float,
    seed: int,
    extra_column_penalizer: float | None,
    penalizer_escalation_grid: tuple[float, ...],
    failure_sink: list[CoxFitFailureRecord],
) -> Any:
    """Ş2 / K16-a (Codex şartlı onay, 2026-09-11) -- `cox_final` yolu
    (UCSF harici/üretim skorlaması için tek, dağıtılabilir Cox modeli).

    ⚠️ KAPSAM FARKI (açıkça beyan edilir -- görev talimatı `tools/
    train_cox_week3.py`'ye DOKUNMAYI YASAKLADI): `pipeline.xgboost_model.
    _fit_cox_with_full_selection()` içindeki eskalasyon SADECE final
    `.fit()` çağrısını yeniden dener (hiperparametre grid-taraması +
    özellik seçimi TEKRARLANMAZ). `week3.fit_final_model_on_full_pool()`
    KENDİ İÇİNDE bu ayrımı yapan bir kanca SUNMUYOR (ve bu dosyaya
    dokunma yasağı var) -- bu yüzden burada eskalasyon, o fonksiyonun
    TAMAMINI (grid-taraması + elastic-net + stabilite seçimi + final fit)
    tek-elemanlı bir `penalizer_grid=(P,)` ile YENİDEN çağırarak yapılır.
    Bu, DAHA AĞIR bir retry'dır ama `train_cox_week3.py`'nin salt-okunur
    kalması için ZORUNLU bir mühendislik ödünüdür.

    Fold-skip (K16-b) YOK: bu TEK modelin başarısızlığı UCSF harici
    değerlendirmesinin TAMAMINI geçersiz kılar -- "atlamak" burada
    anlamsızdır. Eskalasyon tükenirse HER ZAMAN (bayraktan bağımsız)
    `CoxPenalizerEscalationExhaustedError` fırlatır."""

    from lifelines.exceptions import ConvergenceError

    candidates: list[float | None] = [None] + list(penalizer_escalation_grid)
    last_exc: BaseException | None = None
    for attempt_index, escalated_penalizer in enumerate(candidates):
        call_kwargs: dict[str, Any] = dict(
            extra_columns=extra_columns,
            n_bootstrap_stability=n_bootstrap_stability,
            stability_frequency_threshold=stability_frequency_threshold,
            seed=seed,
            extra_column_penalizer=extra_column_penalizer,
        )
        if escalated_penalizer is not None:
            call_kwargs["penalizer_grid"] = (escalated_penalizer,)
        try:
            return week3.fit_final_model_on_full_pool(
                full_pool_training_frame,
                cox_final_feature_columns,
                **call_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 -- daraltma hemen asagida
            if not isinstance(exc, (ConvergenceError, np.linalg.LinAlgError)):
                # Ş2: bu dar kapsamin DISINDAKI istisnalar (kod/girdi
                # hatasi isareti) burada ASLA yutulmaz.
                raise
            failure_sink.append(
                CoxFitFailureRecord(
                    outer_fold=None,
                    inner_fold=None,
                    fit_type="cox_final_full_pool",
                    attempt_index=attempt_index,
                    penalizer=escalated_penalizer if escalated_penalizer is not None else float("nan"),
                    penalizer_escalation_grid=list(penalizer_escalation_grid),
                    candidate_feature_columns=list(cox_final_feature_columns),
                    candidate_feature_columns_hash=hash_feature_columns(cox_final_feature_columns),
                    error_class=type(exc).__name__,
                    error_message=str(exc),
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
            )
            last_exc = exc
            continue

    raise CoxPenalizerEscalationExhaustedError(
        "cox_final (UCSF harici skorlama modeli, week3.fit_final_model_"
        "on_full_pool()): varsayilan penalizer_grid VE TUM eskalasyon "
        f"grid'i ({list(penalizer_escalation_grid)}) denendikten SONRA "
        f"HALA {type(last_exc).__name__ if last_exc else '?'} ile "
        f"basarisiz oldu -- son hata: {last_exc}."
    ) from last_exc


def _prepare_full_pool_standardized_frames(
    training_frame: pd.DataFrame,
    external_frame: pd.DataFrame,
    clinical_standardize_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, float]]:
    """HIGH 5 düzeltmesi (Codex 2. tur, task-mt04rec5-jjqc95): "UCSF final/
    external yolu standardizasyon listesini ATIYOR" bulgusunun düzeltmesi.

    ÖNCEDEN: `_rebuild_variant_training_frame()`'in ürettiği `clinical_
    standardize_columns` `run_ucsf_external_evaluation()` içinde `_`
    öneki ile ATILIYOR, final Cox (`week3.fit_final_model_on_full_pool`)
    HAM `training_frame` ile, external değerlendirme HAM `external_frame`
    ile çağrılıyordu -- oysa aligned CV yolu (`train_xgboost_with_fold_
    aligned_cox_scores()`, `pipeline/xgboost_model.py`) fold-yerel
    standardizasyon YAPIYOR (`standardize_columns_fold_safe()`). Bu iki
    yol FARKLI ölçeklerde çalıştığı için final/external sonucu (örn.
    2026-08-19 smoke'taki UCSF AUC 0,7137) TUTARSIZ bir reçeteden çıkmış
    ve GEÇERSİZ sayılmıştır.

    ARTIK: `tools/train_cox_week3.py::run_modeling_arm()`'daki KANONİK
    desenle (satır ~2164: `standardize_columns_fold_safe(training_frame,
    external_frame, clinical_standardize_columns)`) BİREBİR AYNI -- UPenn
    full-pool (`training_frame`) istatistikleriyle (SADECE `training_
    frame`'den fit edilir, `external_frame`'in KENDİ istatistikleri
    KULLANILMAZ -- CLAUDE.md: harici test setine eğitim-türevi olmayan
    kendi istatistiğini uygulamak yerine, TCGA/UCSF'nin diğer tüm
    adımlarıyla TUTARLI olarak eğitim havuzunun istatistiği uygulanır)
    hem `training_frame` hem `external_frame` AYNI ölçeğe taşınır.

    `pipeline.cox_model.standardize_columns_fold_safe()` kendisi
    öğrenilen ortalama/std'yi DÖNDÜRMEZ (sadece dönüştürülmüş frame'leri
    döner) -- bu fonksiyon `cox_model.py`'ye DOKUNMADAN (salt-okunur
    kullanım kuralı) AYNI mean/std=0->1 mantığını (ddof=0, std==0 ise
    1.0'a yuvarlama) BAĞIMSIZ olarak yeniden hesaplayıp final artifact'e
    kaydedilmek üzere döner -- `standardize_columns_fold_safe()`'in
    KENDİ İÇ hesaplamasıyla `tests/test_train_xgboost_week4.py`'de
    sayısal olarak KARŞILAŞTIRILARAK doğrulanır.

    `clinical_standardize_columns` boşsa (no-op) `training_frame`/
    `external_frame` DEĞİŞMEDEN döner, means/stds boş sözlük olur."""

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    if clinical_standardize_columns:
        means = {
            column: float(training_frame[column].mean())
            for column in clinical_standardize_columns
        }
        raw_stds = training_frame[clinical_standardize_columns].std(ddof=0)
        stds = {
            column: float(raw_stds[column]) if raw_stds[column] > 0 else 1.0
            for column in clinical_standardize_columns
        }

    full_pool_training_frame, full_pool_external_frame = standardize_columns_fold_safe(
        training_frame, external_frame, clinical_standardize_columns
    )
    return full_pool_training_frame, full_pool_external_frame, means, stds


def run_ucsf_external_evaluation(
    *,
    config: "week3.VariantConfig",
    args: argparse.Namespace,
    external_cox_final_fit_failure_log: list[CoxFitFailureRecord] | None = None,
) -> UCSFExternalEvaluationBundle:
    """MEDIUM 5 (Codex, task-mszuse9v-efm1gw) -- yalnız-UPenn'de fit
    edilen NİHAİ Cox->XGBoost zincirini, YENİDEN FİT/EŞİK SEÇİMİ
    YAPMADAN, UCSF 295/169'da değerlendirir. `week3.fit_final_model_on_
    full_pool()` + `week3.evaluate_external_test()`'in (Cox tarafı,
    `run_modeling_arm()` içinde ZATEN var olan) AYNI "tek final model +
    harici test" desenini XGBoost katmanına taşır.

    ⚠️ ESKİ (`_rebuild_variant_training_frame()`, global RCS düğümü)
    kullanılır -- burada CV/fold YOK, tüm UPenn havuzunda TEK bir final
    model üretiliyor; HIGH 3'ün fold-güvenlik sorunu bu adımda
    GEÇERSİZ -- Cox'un kendi `fit_final_model_on_full_pool()`'u zaten
    AYNI global-düğüm yaklaşımını kullanıyor (`run_modeling_arm()` ile
    TUTARLI, aksi hâlde eğitim/harici-test farklı düğüm kullanırdı).

    🔴 Bu görev talimatı gereği ÜRETİM KOŞUSU YAPILMADI -- bu fonksiyon
    SADECE `--evaluate-ucsf-external` bayrağıyla, gerçek DB bağlantısı
    olan bir ortamda çağrılabilir; `tests/test_train_xgboost_week4.py`
    bu fonksiyonun DB'siz alt-adımlarını (`fit_final_cox_and_xgboost_
    pipeline_on_full_pool`/`evaluate_xgboost_external_test`) sentetik
    veriyle AYRICA test eder."""

    from psycopg2.extras import RealDictCursor

    from db_connection import get_connection

    connection = get_connection(readonly=True)
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        training_frame, feature_columns, extra_columns, clinical_standardize_columns = (
            _rebuild_variant_training_frame(cursor=cursor, config=config, args=args)
        )
        upenn_patients = week3.fetch_patients_frame(cursor, source_name=week3.UPENN_SOURCE_NAME)
        age_knots = (
            week3.compute_rcs_knots(upenn_patients[week3.PATIENT_RAW_AGE_COLUMN].astype(float))
            if config.use_age_spline
            else None
        )

        ucsf_long = week3.fetch_c32_radiomics_long_frame(
            cursor, segmentation_tool=week3.SEGMENTATION_TOOL_UCSF_C32
        )
        week3.raise_if_empty_c32(ucsf_long, segmentation_tool=week3.SEGMENTATION_TOOL_UCSF_C32)
        ucsf_patients = week3.fetch_patients_frame(cursor, source_name=week3.UCSF_SOURCE_NAME)
        cursor.close()
    finally:
        connection.close()

    # Ş4 (Codex sartli onay, 2026-09-11) -- DB-siz, calisma-zamani guard.
    _validate_ucsf_distribution_feature_pool_disjoint(feature_columns, clinical_standardize_columns)

    ucsf_wide, _ = week3.pivot_ucsf_regions_long_to_wide(ucsf_long, regions=list(config.regions))
    ucsf_clinical = week3.build_variant_clinical_frame(
        ucsf_patients,
        include_mgmt=config.include_mgmt,
        use_age_spline=config.use_age_spline,
        age_knots=age_knots,
    )
    ucsf_patients_c = ucsf_patients.join(ucsf_clinical)

    external_frame, _external_report = week3.build_external_test_frame(
        ucsf_wide[feature_columns], ucsf_patients_c, passthrough_columns=extra_columns
    )
    week3.check_external_test_pool_counts(
        len(external_frame),
        int(external_frame["event"].sum()),
        expected_patients=args.expected_ucsf_patient_count,
        expected_events=args.expected_ucsf_event_count,
        allow_mismatch=args.allow_unexpected_ucsf_count,
        label=f"UCSF harici test ({config.name})",
    )

    # 🔴 B9-FIX2 (2026-08-28, Codex stop-time bulgusu): "`--reduce-
    # collinearity` egitim ve harici skorlamada FARKLI Cox receteleri
    # uretiyor."
    #
    # BULGU (dogrulandi): `cox_final` asagida SADECE `evaluate_xgboost_
    # external_test()`'e -- yani GERCEK UCSF/uretim skorlamasina --
    # veriliyor. Ama `feature_columns` (FILTRELENMEMIS tam havuz) ile fit
    # ediliyordu; oysa XGBoost'un meta-skorlarini ureten Cox fit'leri
    # `fit_final_cox_and_xgboost_pipeline_on_full_pool(reduce_
    # collinearity=...)` icinde FILTRELENMIS havuzdan geliyor.
    # Sonuc: `--reduce-collinearity` ACIKKEN XGBoost, FILTRELI Cox
    # skorlariyla EGITILIP FILTRESIZ Cox skorlariyla TEST ediliyordu --
    # klasik train/serve skew. Harici AUC bu yuzden modelin gercek
    # uretim davranisini olcmezdi.
    #
    # DUZELTME: `cox_final`'in aday havuzu da AYNI filtreden gecer.
    # SIZINTI YOK: `cox_final` tanimi geregi "TUM havuzda, CV'siz fit
    # edilmis final/dagitim modeli"dir; filtre yalniz UPenn egitim
    # cercevesinden hesaplanir, `full_pool_external_frame` (UCSF) HIC
    # GORULMEZ. Bu, cross-fit icindeki FOLD-YEREL filtreden bilincli
    # olarak FARKLIDIR ve oyle olmalidir: orada amac fold sizintisini
    # onlemek, burada amac dagitilacak modelin recetesini sabitlemek.
    #
    # 🔴 v3 RECETE-SADAKATI (2026-09-12, Baris: "sapmayla kosmayalim, v3b
    # ile XGBoost tam uyumluluk gostermesi lazim"): ONCEDEN bu filtre
    # `full_pool_training_frame` (ASAGIDA, standardize EDILMIS bir cerceve)
    # uzerinde hesaplaniyordu -- `clinical_standardize_columns` yalniz yasi
    # icerdigi icin BUGUNE KADAR bu radyomikleri etkilemiyordu (kazara
    # guvenli), AMA artik radyomikleri de standardize etmemiz gerektigi
    # icin (asagida) bu SIRA `build_v3_candidate_pool()`'un near-constant/
    # CV filtresini Z-SKORLANMIS deger uzerinden hesaplardi (bkz.
    # `CoxFeatureCandidatePoolOverlapError` docstring'i, `tools/train_cox_
    # week3.py::run_single_v3_variant()`'in `build_v3_radiomic_feature_
    # pool()`'u HER ZAMAN HAM `upenn_wide_for_pool`'dan cagirmasiyla AYNI
    # SIRA GEREKSINIMI). DUZELTME: filtre HAM `training_frame`'den
    # hesaplanir (asagida), TAM OLARAK week3'un sirasiyla.
    cox_final_feature_columns = feature_columns
    if args.reduce_collinearity:
        (
            cox_final_feature_columns,
            _cox_final_collinearity_report,
        ) = reduce_collinearity_module.build_v3_candidate_pool(
            training_frame,
            feature_columns,
            cv_threshold=args.collinearity_cv_threshold,
            corr_threshold=args.collinearity_corr_threshold,
        )
        if not cox_final_feature_columns:
            raise RuntimeError(
                "cox_final (harici skorlama modeli): reduce_collinearity "
                "havuzu SIFIR ozellik birakti -- cv_threshold/corr_threshold "
                "gozden gecirilmeli."
            )
        print(
            "[B9] cox_final (harici skorlama) aday havuzu filtrelendi: "
            f"{len(feature_columns)} -> {len(cox_final_feature_columns)} "
            "ozellik (XGBoost meta-skor recetesiyle AYNI filtre)."
        )

    # v3 recete-sadakati (devam): `tools/train_cox_week3.py::
    # V3VariantConfig.standardize_radiomics=True` ile AYNI recete --
    # `clinical_standardize_columns = feature_columns + clinical_age_
    # columns` (week3 satir ~3806-3812) -- filtre SONUCU (`cox_final_
    # feature_columns`, YUKARIDA HAM veriden hesaplandi) da yasla birlikte
    # standardize edilir. `reduce_collinearity=False` iken bu liste
    # `clinical_standardize_columns` ile AYNI (davranis DEGISMEZ).
    cox_final_clinical_standardize_columns = (
        clinical_standardize_columns + list(cox_final_feature_columns)
        if args.reduce_collinearity
        else clinical_standardize_columns
    )

    # HIGH 5 duzeltmesi (Codex 2. tur): ONCEDEN `clinical_standardize_
    # columns` `_` onekiyle ATILIYOR, final Cox HAM training_frame ile
    # fit ediliyor, external_frame HAM kaliyordu -- aligned CV yolunun
    # (fold-yerel standardizasyon) receteten TUTARSIZDI. ARTIK
    # `tools/train_cox_week3.py::run_modeling_arm()` ile AYNI desen:
    # UPenn full-pool istatistikleriyle (SADECE training_frame'den fit
    # edilir) hem training_frame hem external_frame (UCSF) AYNI olcege
    # tasinir; ogrenilen ortalama/std final artifact'e KAYDEDILIR.
    #
    # ⚠️ Bu frame CIFTI SADECE `cox_final` (dagitim Cox modeli) icindir --
    # `cox_final_clinical_standardize_columns` (reduce_collinearity=True
    # iken radyomikleri de icerir) kullanir. XGBoost'un KENDI cross-fit
    # meta-skor girdisi (asagida `xgb_meta_training_frame`) AYRI, yalniz-
    # yas-standardize bir cerceve kullanir -- `fit_final_cox_and_xgboost_
    # pipeline_on_full_pool()` artik radyomikleri KENDI split-yerel
    # filtresinden SONRA kendi icinde standardize ediyor (bkz. o
    # fonksiyonun 2026-09-12 guncellenen docstring'i); bu frame'e
    # onceden (globalden) standardize edilmis radyomik SOKMAK o split-
    # yerel filtreyi Z-skorlanmis veri uzerinden calistirirdi.
    (
        full_pool_training_frame,
        full_pool_external_frame,
        clinical_standardize_means,
        clinical_standardize_stds,
    ) = _prepare_full_pool_standardized_frames(
        training_frame, external_frame, cox_final_clinical_standardize_columns
    )
    xgb_meta_training_frame, xgb_meta_external_frame = standardize_columns_fold_safe(
        training_frame, external_frame, clinical_standardize_columns
    )

    # Ş2 / K16-a (Codex sartli onay, 2026-09-11): `cox_final`, UCSF harici/
    # uretim skorlamasina giden TEK dagitilabilir model -- eskalasyon
    # olmadan bir ConvergenceError/LinAlgError bu TUM degerlendirmeyi
    # cokertirdi. Bkz. `_fit_final_model_on_full_pool_with_penalizer_
    # escalation()` docstring'i (kapsam farki ACIKCA belgelenmis).
    # Ş3 (crash-manifest): caller (main()) verdiyse O listeyi KULLAN --
    # bu fonksiyon COKSE BILE caller'in elindeki liste guncel kalir.
    cox_final_fit_failure_log = (
        external_cox_final_fit_failure_log if external_cox_final_fit_failure_log is not None else []
    )
    cox_final_escalation_grid = (
        args.cox_penalizer_escalation_grid
        if args.cox_penalizer_escalation_grid is not None
        else DEFAULT_COX_PENALIZER_ESCALATION_GRID
    )
    cox_final = _fit_final_model_on_full_pool_with_penalizer_escalation(
        full_pool_training_frame,
        cox_final_feature_columns,
        extra_columns=extra_columns,
        n_bootstrap_stability=args.cox_n_bootstrap_stability,
        stability_frequency_threshold=args.cox_primary_threshold,
        seed=args.seed,
        extra_column_penalizer=args.cox_extra_column_penalizer,
        penalizer_escalation_grid=cox_final_escalation_grid,
        failure_sink=cox_final_fit_failure_log,
    )

    # v3 recete-sadakati (2026-09-12): `full_pool_training_frame` DEGIL --
    # `xgb_meta_training_frame` (yalniz-yas-standardize, radyomikler HAM)
    # kullanilir. Gerekce yukarida (frame ciftinin tanimlandigi blok).
    target, twelve_month_report = define_twelve_month_survival_target(
        xgb_meta_training_frame, threshold_days=args.twelve_month_threshold_days
    )
    labeled_frame = apply_twelve_month_target(
        xgb_meta_training_frame, target, report=twelve_month_report
    )

    # 🔴 CRITICAL duzeltmesi (3. tur, task-mt06mqzi-k0gb5j): ARTIK
    # `cox_final` (TUM havuzda, CV'siz fit edilmis final model) buraya
    # VERILMIYOR -- XGBoost'un KENDI cross-fit meta-skoru icin her split'te
    # YENIDEN secim yapilir (bkz. fonksiyonun guncellenmis docstring'i).
    # `cox_final` SADECE asagida `evaluate_xgboost_external_test()`'e
    # (gercek UCSF/uretim skorlamasi icin) verilir.
    xgboost_final = fit_final_cox_and_xgboost_pipeline_on_full_pool(
        labeled_frame,
        feature_columns,
        extra_columns=extra_columns,
        cox_extra_column_penalizer=args.cox_extra_column_penalizer,
        cox_cross_fit_splits=args.cox_inner_splits,
        cox_inner_splits=args.cox_inner_splits,
        cox_n_bootstrap_stability=args.cox_n_bootstrap_stability,
        cox_stability_frequency_threshold=args.cox_primary_threshold,
        seed=args.seed,
        reduce_collinearity=args.reduce_collinearity,
        collinearity_cv_threshold=args.collinearity_cv_threshold,
        collinearity_corr_threshold=args.collinearity_corr_threshold,
        xgb_inner_splits=args.xgb_inner_splits,
        xgb_seed=args.seed,
        cox_penalizer_escalation_grid=cox_final_escalation_grid,
        cox_fit_failure_log=cox_final_fit_failure_log,
    )
    verify_full_pool_pipeline_leakage_free(xgboost_final, labeled_frame.index)

    evaluation = evaluate_xgboost_external_test(
        cox_final.fitted_model,
        cox_final.final_features,
        xgboost_final,
        full_pool_external_frame,
        extra_columns=extra_columns,
        twelve_month_threshold_days=args.twelve_month_threshold_days,
        n_bootstrap_ci=args.xgb_n_bootstrap_ci,
        seed=args.seed,
        # v3 recete-sadakati (2026-09-12): XGBoost'un KENDI (ham-radyomik)
        # egitim olcegiyle AYNI harici cerceve -- bkz. yukaridaki frame
        # ciftinin tanimlandigi blok + evaluate_xgboost_external_test()'in
        # guncellenmis docstring'i.
        xgboost_external_frame=xgb_meta_external_frame,
    )

    return UCSFExternalEvaluationBundle(
        evaluation=evaluation,
        cox_final_features=cox_final.final_features,
        cox_best_penalizer=cox_final.best_penalizer,
        cox_best_l1_ratio=cox_final.best_l1_ratio,
        xgboost_best_params=xgboost_final.best_params,
        n_upenn_training_patients=len(training_frame),
        n_upenn_training_events=int(training_frame["event"].sum()),
        clinical_standardize_columns=clinical_standardize_columns,
        clinical_standardize_means=clinical_standardize_means,
        clinical_standardize_stds=clinical_standardize_stds,
        cox_final_fit_failure_log=cox_final_fit_failure_log,
        n_cross_fit_training_patients=len(labeled_frame),
        n_cross_fit_training_events=int(labeled_frame["event"].sum()),
        cox_final_clinical_standardize_columns=cox_final_clinical_standardize_columns,
    )


def _script_sha256() -> str:
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hafta 4 XGBoost egitimi -- out-of-fold Cox skoru + radyomik + "
            "klinik. NIHAI Cox varyanti secilmeden calistirilmamali "
            "(--variant ZORUNLU, hicbir varsayilan YOK)."
        )
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=list(week3.V2_VARIANT_NAMES),
        help="Hangi Cox varyantinin training_frame recetesiyle calisilacagi.",
    )
    parser.add_argument(
        "--legacy-misaligned-pipeline",
        action="store_true",
        help=(
            "ESKI/HIZALI OLMAYAN Cox->XGBoost yolu (run_xgboost_week4_pipeline()) "
            "-- SADECE Ege'nin sizinti-karsilastirma gorevi icin (CLAUDE.md "
            "kritik kural riski tasir, bkz. Codex capraz inceleme CRITICAL 1, "
            "task-mszuse9v-efm1gw). VARSAYILAN artik ALIGNED yoldur."
        ),
    )
    # --- ALIGNED (yeni birincil) yol icin PAYLASILAN dis-fold parametreleri ---
    parser.add_argument(
        "--outer-splits", type=int, default=5, help="ALIGNED yol -- Cox VE XGBoost'un PAYLASTIGI dis-fold sayisi."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="ALIGNED yol -- PAYLASILAN dis-fold bolunmesinin seed'i."
    )
    # --- LEGACY (--legacy-misaligned-pipeline) yol icin AYRI parametreler ---
    parser.add_argument("--cox-outer-splits", type=int, default=5)
    parser.add_argument("--cox-seed", type=int, default=42)
    parser.add_argument("--xgb-outer-splits", type=int, default=5)
    parser.add_argument("--xgb-seed", type=int, default=42)
    # --- HER IKI yolda ORTAK parametreler ---
    parser.add_argument("--cox-inner-splits", type=int, default=5)
    parser.add_argument("--cox-n-bootstrap-stability", type=int, default=200)
    parser.add_argument("--cox-primary-threshold", type=float, default=0.6)
    parser.add_argument("--cox-extra-column-penalizer", type=float, default=0.0)
    # --- B9 (2026-08-28): reduce_collinearity fold-yerel filtresi --
    # VARSAYILAN KAPALI, mevcut 7 kolun sonuclarini bit-birebir yeniden
    # uretilebilir tutmak icin (bkz. pipeline.xgboost_model.train_
    # xgboost_with_fold_aligned_cox_scores() docstring'i). Herhangi bir
    # Cox varyanti/kolu ile calisir -- belirli bir varyanta GOMULU DEGIL.
    parser.add_argument(
        "--reduce-collinearity",
        action="store_true",
        help=(
            "Cox aday havuzunu (STABLE_FEATURES_ICC60'tan gelen) HER dis/"
            "ic-fold'da SADECE o fold'un kendi egitim alt-kumesinden "
            "pipeline.reduce_collinearity.build_v3_candidate_pool() ile "
            "daraltir (near-constant + |r|>=esik dedup). VARSAYILAN "
            "KAPALI -- production.log'daki Matrix-is-singular/"
            "ConvergenceError cokusunu ONLEMEK icin B9 kapsaminda "
            "eklendi, hicbir Cox varyantina GOMULU DEGIL."
        ),
    )
    parser.add_argument(
        "--collinearity-cv-threshold",
        type=float,
        default=reduce_collinearity_module.DEFAULT_CV_THRESHOLD,
        help="--reduce-collinearity acikken near-constant esigi (CV=std/|mean|).",
    )
    parser.add_argument(
        "--collinearity-corr-threshold",
        type=float,
        default=reduce_collinearity_module.DEFAULT_CORRELATION_CLUSTER_THRESHOLD,
        help="--reduce-collinearity acikken korelasyon-kumeleme esigi (|r|).",
    )
    # --- Ş2 / K16-a+b (Codex sartli onay, 2026-09-11): final Cox fit
    # penalizer-eskalasyonu + acik fold-atlama bayragi. VARSAYILAN
    # davranis DEGISMEZ: eskalasyon SADECE gercek bir ConvergenceError/
    # LinAlgError anında devreye girer (zaten basarili olan hicbir fit'i
    # ETKILEMEZ), `--allow-fold-skip` VARSAYILAN KAPALIDIR (verilmezse
    # eskalasyon da tukenirse ESKISI GIBI fail-loud coker).
    parser.add_argument(
        "--cox-penalizer-escalation-grid",
        type=str,
        default=None,
        help=(
            "Final Cox fit ConvergenceError/LinAlgError ile basarisiz "
            "olursa denenecek, virgulle ayrilmis penalizer listesi (orn. "
            "'2.0,5.0,10.0,25.0,50.0,100.0'). Verilmezse pipeline.xgboost_"
            f"model.DEFAULT_COX_PENALIZER_ESCALATION_GRID kullanilir "
            f"({','.join(str(v) for v in DEFAULT_COX_PENALIZER_ESCALATION_GRID)})."
        ),
    )
    parser.add_argument(
        "--allow-fold-skip",
        action="store_true",
        help=(
            "ALIGNED yolda (--legacy-misaligned-pipeline ile GECERSIZ, bkz. "
            "asagidaki karsilikli-dislama) final Cox fit penalizer-"
            "eskalasyonu da TUKENIRSE, o TEK dis-fold'u ATLAR (kaydeder, "
            "coktermez) -- VARSAYILAN KAPALI: verilmezse tukenme ESKISI "
            "GIBI RuntimeError ile coker (fail-loud korunur)."
        ),
    )
    parser.add_argument("--xgb-inner-splits", type=int, default=5)
    parser.add_argument("--xgb-n-bootstrap-ci", type=int, default=1000)
    parser.add_argument(
        "--twelve-month-threshold-days", type=float, default=DEFAULT_TWELVE_MONTH_THRESHOLD_DAYS
    )
    parser.add_argument("--expected-patient-count", type=int, default=week3.EXPECTED_UPENN_TRAINING_PATIENTS)
    parser.add_argument("--expected-event-count", type=int, default=week3.EXPECTED_UPENN_TRAINING_EVENTS)
    parser.add_argument("--allow-unexpected-patient-count", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    # --- MEDIUM 5 (Codex) -- UCSF harici degerlendirme (opsiyonel, ek DB sorgusu) ---
    parser.add_argument(
        "--evaluate-ucsf-external",
        action="store_true",
        help=(
            "Nihai (tum-UPenn'de fit edilen) Cox->XGBoost zincirini, YENIDEN "
            "FIT/ESIK SECIMI YAPMADAN, UCSF 295/169'da degerlendirir "
            "(pipeline.xgboost_model.evaluate_xgboost_external_test())."
        ),
    )
    parser.add_argument(
        "--expected-ucsf-patient-count", type=int, default=week3.EXPECTED_UCSF_EXTERNAL_TEST_PATIENTS
    )
    parser.add_argument(
        "--expected-ucsf-event-count", type=int, default=week3.EXPECTED_UCSF_EXTERNAL_TEST_EVENTS
    )
    parser.add_argument("--allow-unexpected-ucsf-count", action="store_true")
    args = parser.parse_args(argv)
    # Ş1 (Codex capraz inceleme, 2026-09-11) -- `--legacy-misaligned-pipeline`
    # dali `run_xgboost_week4_pipeline()`'i cagirir ve bu fonksiyon
    # `reduce_collinearity` parametresini HIC ALMAZ/ILETMEZ (yalnizca ALIGNED
    # yol -- `run_xgboost_week4_pipeline_aligned()` -- bu bayragi gorur).
    # Ikisi birlikte verilirse kullanici filtrenin uygulandigini SANIR ama
    # sessizce YOK SAYILIR. Bunun yerine fail-loud: argparse hatasi.
    if args.legacy_misaligned_pipeline and args.reduce_collinearity:
        parser.error(
            "--legacy-misaligned-pipeline ile --reduce-collinearity birlikte "
            "kullanilamaz: legacy yol (run_xgboost_week4_pipeline()) "
            "reduce_collinearity parametresini pipeline'a HIC ILETMEZ, bu "
            "yuzden bayrak acik gorunse de sessizce etkisiz kalirdi (Codex "
            "capraz inceleme Ş1, 2026-09-11). Once hangi Cox->XGBoost "
            "yolunu (ALIGNED varsayilan, ya da --legacy-misaligned-pipeline) "
            "kullanacagini sec; --reduce-collinearity SADECE ALIGNED yolda "
            "gecerlidir."
        )
    # Ş2 / K16-b: ayni gerekce -- `--legacy-misaligned-pipeline` dali
    # `run_xgboost_week4_pipeline()`'i cagirir, bu fonksiyon `allow_fold_
    # skip` parametresini HIC ALMAZ. Birlikte verilirse sessizce yok
    # sayilirdi.
    if args.legacy_misaligned_pipeline and args.allow_fold_skip:
        parser.error(
            "--legacy-misaligned-pipeline ile --allow-fold-skip birlikte "
            "kullanilamaz: legacy yol (run_xgboost_week4_pipeline()) "
            "allow_fold_skip parametresini pipeline'a HIC ILETMEZ, bu "
            "yuzden bayrak acik gorunse de sessizce etkisiz kalirdi (Codex "
            "capraz inceleme Ş2/K16-b, 2026-09-11). --allow-fold-skip "
            "SADECE ALIGNED yolda gecerlidir."
        )
    # Ş2 / K16-a: `--cox-penalizer-escalation-grid` virgulle-ayrilmis
    # string'ini ERKEN (CLI parse zamaninda, calisma ortasinda DEGIL)
    # tuple[float,...]'a cevirir -- kotu girdi (orn. bos deger, sayi
    # olmayan token) hemen, acik bir argparse hatasiyla yakalanir.
    if args.cox_penalizer_escalation_grid is not None:
        raw_grid = args.cox_penalizer_escalation_grid
        try:
            parsed_grid = tuple(float(token.strip()) for token in raw_grid.split(",") if token.strip())
        except ValueError as exc:
            parser.error(
                f"--cox-penalizer-escalation-grid gecersiz: {raw_grid!r} -- "
                f"virgulle-ayrilmis sayi listesi olmali (orn. '2.0,5.0,10.0'). "
                f"Ayristirma hatasi: {exc}"
            )
        else:
            if not parsed_grid:
                parser.error(
                    "--cox-penalizer-escalation-grid BOS -- en az bir "
                    "penalizer degeri icermeli."
                )
            args.cox_penalizer_escalation_grid = parsed_grid
    return args


def _write_crash_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    *,
    stage: str,
    exc: BaseException,
    cox_fit_failure_logs: dict[str, list[CoxFitFailureRecord]],
) -> Path:
    """Ş3 (Codex şartlı onay, 2026-09-11) -- pipeline ÇÖKERSE (penalizer-
    eskalasyonu tükenip `--allow-fold-skip` KAPALI kaldığında, ya da BAŞKA
    bir beklenmeyen istisna) bu ana kadar biriken TÜM başarısız Cox fit
    denemelerini (fold/split kimliği, aday özellik listesi + hash'i,
    penalizer grid, denenen değerler, hata sınıfı/mesajı -- `CoxFitFailureRecord`
    alanlarının TAMAMI) ÇÖKÜŞ ANINDA diske yazar.

    ATOMİK YAZIM: önce `<hedef>.tmp` dosyasına yazılır, sonra `os.replace()`
    ile hedefe TAŞINIR -- yazım ortasında ikinci bir çökme (disk dolması
    vb.) yarım/bozuk bir JSON dosyası BIRAKMAZ.

    `cox_fit_failure_logs` -- stage'e göre birden fazla liste olabilir
    (örn. aligned pipeline VE (varsa) UCSF `cox_final`/`xgboost_final`
    -- ikisi de AYNI `main()` çağrısında biriktirilmiş olabilir); her
    biri kendi anahtarıyla (`"aligned"`, `"ucsf_external"` gibi) JSON'a
    yazılır, hangi listenin BOŞ/DOLU olduğu şeffaf kalır."""

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "stage": stage,
        "error_class": type(exc).__name__,
        "error_message": str(exc),
        "cli_args": {
            "reduce_collinearity": getattr(args, "reduce_collinearity", None),
            "allow_fold_skip": getattr(args, "allow_fold_skip", None),
            "cox_penalizer_escalation_grid": getattr(args, "cox_penalizer_escalation_grid", None),
            "legacy_misaligned_pipeline": getattr(args, "legacy_misaligned_pipeline", None),
            "evaluate_ucsf_external": getattr(args, "evaluate_ucsf_external", None),
        },
        "cox_fit_failure_logs": {
            log_name: [
                {
                    "outer_fold": record.outer_fold,
                    "inner_fold": record.inner_fold,
                    "fit_type": record.fit_type,
                    "attempt_index": record.attempt_index,
                    "penalizer": record.penalizer,
                    "penalizer_escalation_grid": record.penalizer_escalation_grid,
                    "candidate_feature_columns": record.candidate_feature_columns,
                    "candidate_feature_columns_hash": record.candidate_feature_columns_hash,
                    "n_candidate_features": len(record.candidate_feature_columns),
                    "error_class": record.error_class,
                    "error_message": record.error_message,
                    "timestamp_utc": record.timestamp_utc,
                }
                for record in records
            ]
            for log_name, records in cox_fit_failure_logs.items()
        },
        "total_failed_attempts": sum(len(records) for records in cox_fit_failure_logs.values()),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"week4_{args.variant}_{stage}_crash_manifest.json"
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    tmp_target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp_target.replace(target)
    return target


def main(argv: list[str] | None = None) -> int:
    from psycopg2.extras import RealDictCursor

    from db_connection import get_connection

    args = parse_args(argv)
    output_dir = args.output_dir or (PROJECT_ROOT / "artifacts" / "week4" / "xgboost_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = week3.select_v2_variant_configs([args.variant])[0]

    connection = get_connection(readonly=True)
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        if args.legacy_misaligned_pipeline:
            training_frame, feature_columns, extra_columns, clinical_standardize_columns = (
                _rebuild_variant_training_frame(cursor=cursor, config=config, args=args)
            )
            fold_local_extra_column_builder = None
        else:
            (
                training_frame,
                feature_columns,
                extra_columns,
                clinical_standardize_columns,
                fold_local_extra_column_builder,
            ) = _rebuild_variant_training_frame_for_aligned_pipeline(cursor=cursor, config=config, args=args)
        cursor.close()
    finally:
        connection.close()

    if args.legacy_misaligned_pipeline:
        result = run_xgboost_week4_pipeline(
            training_frame,
            feature_columns,
            extra_columns=extra_columns,
            clinical_standardize_columns=clinical_standardize_columns,
            twelve_month_threshold_days=args.twelve_month_threshold_days,
            censoring_bias_continuous_columns=[week3.CLINICAL_AGE_COLUMN]
            if week3.CLINICAL_AGE_COLUMN in extra_columns
            else None,
            cox_outer_splits=args.cox_outer_splits,
            cox_inner_splits=args.cox_inner_splits,
            cox_seed=args.cox_seed,
            cox_n_bootstrap_stability=args.cox_n_bootstrap_stability,
            cox_stability_frequency_threshold=args.cox_primary_threshold,
            cox_extra_column_penalizer=args.cox_extra_column_penalizer,
            xgb_outer_splits=args.xgb_outer_splits,
            xgb_inner_splits=args.xgb_inner_splits,
            xgb_seed=args.xgb_seed,
            xgb_n_bootstrap_ci=args.xgb_n_bootstrap_ci,
        )
        _write_outputs(output_dir, args, result)
        print(
            f"[xgboost/{args.variant}] (LEGACY/MISALIGNED, SADECE karsilastirma) "
            f"DOGRU auc_oof={result.xgboost_result_correct.auc_out_of_fold['auc']:.4f} "
            f"[{result.xgboost_result_correct.auc_out_of_fold['ci_lower']:.4f}-"
            f"{result.xgboost_result_correct.auc_out_of_fold['ci_upper']:.4f}] | "
            f"YANLIS(sizintili-cox-girdisi) auc_oof={result.xgboost_result_leaky_cox_input.auc_out_of_fold['auc']:.4f} | "
            f"YANLIS(tam-in-sample) auc={result.xgboost_auc_fully_in_sample['auc']:.4f}",
            flush=True,
        )
    else:
        # Ş3 (Codex şartlı onay, 2026-09-11): bu liste `run_xgboost_week4_
        # pipeline_aligned()`'e (ve onun içinden `train_xgboost_with_fold_
        # aligned_cox_scores()`'a) CANLI REFERANS olarak geçer -- pipeline
        # penalizer-eskalasyonu tükenip çökse BİLE (allow_fold_skip=False),
        # bu listede O ANA KADAR biriken TÜM başarısız deneme kayıtları KALIR.
        aligned_cox_fit_failure_log: list[CoxFitFailureRecord] = []
        try:
            result = run_xgboost_week4_pipeline_aligned(
                training_frame,
                feature_columns,
                extra_columns=extra_columns,
                clinical_standardize_columns=clinical_standardize_columns,
                fold_local_extra_column_builder=fold_local_extra_column_builder,
                twelve_month_threshold_days=args.twelve_month_threshold_days,
                censoring_bias_continuous_columns=[week3.CLINICAL_AGE_COLUMN]
                if week3.CLINICAL_AGE_COLUMN in training_frame.columns
                else None,
                outer_splits=args.outer_splits,
                seed=args.seed,
                cox_inner_splits=args.cox_inner_splits,
                cox_n_bootstrap_stability=args.cox_n_bootstrap_stability,
                cox_stability_frequency_threshold=args.cox_primary_threshold,
                cox_extra_column_penalizer=args.cox_extra_column_penalizer,
                reduce_collinearity=args.reduce_collinearity,
                collinearity_cv_threshold=args.collinearity_cv_threshold,
                collinearity_corr_threshold=args.collinearity_corr_threshold,
                xgb_inner_splits=args.xgb_inner_splits,
                xgb_seed=args.seed,
                xgb_n_bootstrap_ci=args.xgb_n_bootstrap_ci,
                cox_penalizer_escalation_grid=args.cox_penalizer_escalation_grid,
                allow_fold_skip=args.allow_fold_skip,
                cox_fit_failure_log=aligned_cox_fit_failure_log,
            )
        except Exception as exc:
            manifest_path = _write_crash_manifest(
                output_dir,
                args,
                stage="aligned_pipeline",
                exc=exc,
                cox_fit_failure_logs={"aligned_pipeline": aligned_cox_fit_failure_log},
            )
            print(
                f"[xgboost/{args.variant}] COKTU ({type(exc).__name__}) -- "
                f"crash-manifest: {manifest_path}",
                flush=True,
            )
            raise
        _write_aligned_outputs(output_dir, args, result)
        aof = result.aligned_result.auc_out_of_fold
        print(
            f"[xgboost/{args.variant}] (ALIGNED, CRITICAL-1-duzeltmesi) "
            f"auc_oof={aof['auc']:.4f} [{aof['ci_lower']:.4f}-{aof['ci_upper']:.4f}] "
            f"(n={aof['n_patients']}, n_positive={aof['n_positive']})",
            flush=True,
        )

    if args.evaluate_ucsf_external:
        ucsf_cox_final_fit_failure_log: list[CoxFitFailureRecord] = []
        try:
            ucsf_bundle = run_ucsf_external_evaluation(
                config=config,
                args=args,
                external_cox_final_fit_failure_log=ucsf_cox_final_fit_failure_log,
            )
        except Exception as exc:
            manifest_path = _write_crash_manifest(
                output_dir,
                args,
                stage="ucsf_external_evaluation",
                exc=exc,
                cox_fit_failure_logs={"ucsf_external_evaluation": ucsf_cox_final_fit_failure_log},
            )
            print(
                f"[xgboost/{args.variant}] UCSF HARICI DEGERLENDIRME COKTU "
                f"({type(exc).__name__}) -- crash-manifest: {manifest_path}",
                flush=True,
            )
            raise
        _write_ucsf_external_outputs(output_dir, args, ucsf_bundle)
        auc = ucsf_bundle.evaluation.auc
        print(
            f"[xgboost/{args.variant}] UCSF HARICI auc={auc['auc']:.4f} "
            f"[{auc['ci_lower']:.4f}-{auc['ci_upper']:.4f}] "
            f"(n={ucsf_bundle.evaluation.n_patients_evaluated}, "
            f"excluded_ambiguous={ucsf_bundle.evaluation.n_excluded_ambiguous_censoring})",
            flush=True,
        )
    return 0


def _write_outputs(output_dir: Path, args: argparse.Namespace, result: Week4PipelineResult) -> None:
    variant = args.variant

    fold_copy = result.xgboost_result_correct.fold_results.copy()
    fold_copy.to_csv(output_dir / f"week4_{variant}_xgboost_fold_results.csv", index=False)

    oof_copy = result.xgboost_result_correct.out_of_fold_predictions.copy()
    oof_copy["risk_class"] = result.risk_class_out_of_fold
    oof_copy.to_csv(output_dir / f"week4_{variant}_xgboost_out_of_fold_predictions.csv")

    if result.censoring_bias_summary is not None:
        result.censoring_bias_summary.to_csv(
            output_dir / f"week4_{variant}_censoring_bias_summary.csv", index=False
        )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _script_sha256(),
        "variant": variant,
        "cli_args": vars(args) | {"output_dir": str(args.output_dir) if args.output_dir else None},
        "twelve_month_label_report": {
            "n_total": result.twelve_month_report.n_total,
            "n_survived_yes": result.twelve_month_report.n_survived_yes,
            "n_died_no": result.twelve_month_report.n_died_no,
            "n_excluded_ambiguous_censoring": result.twelve_month_report.n_excluded_ambiguous_censoring,
            "pct_excluded": result.twelve_month_report.pct_excluded,
            "threshold_days": result.twelve_month_report.threshold_days,
        },
        "n_patients_before_target_exclusion": result.n_patients_before_target_exclusion,
        "n_patients_after_target_exclusion": result.n_patients_after_target_exclusion,
        "xgboost_auc_out_of_fold_correct": result.xgboost_result_correct.auc_out_of_fold,
        "xgboost_auc_out_of_fold_leaky_cox_input": result.xgboost_result_leaky_cox_input.auc_out_of_fold,
        "xgboost_auc_fully_in_sample": result.xgboost_auc_fully_in_sample,
        "leakage_effect_note": (
            "'auc_out_of_fold_correct' = dogru pipeline (out-of-fold Cox "
            "skoru girdi + XGBoost'un kendi durust nested-CV'si). "
            "'auc_out_of_fold_leaky_cox_input' = SADECE Cox skoru girdisi "
            "in-sample/sizintili (XGBoost'un KENDI CV'si hala durust) -- "
            "raw/mimari/v45.txt'nin 'juri sorusu'nu (Cox->XGBoost sizintisi) "
            "IZOLE eder. 'auc_fully_in_sample' = HEM Cox skoru HEM XGBoost'un "
            "kendisi tum veriyi gorup tum veriyi skorlar -- en asiri/en basit "
            "yanlis karsilastirma (plan.txt satir 508'in literal istegi)."
        ),
        "risk_thresholds": {
            "low_upper_bound": result.risk_thresholds.low_upper_bound,
            "medium_upper_bound": result.risk_thresholds.medium_upper_bound,
            "method": result.risk_thresholds.method,
            "n_training_patients": result.risk_thresholds.n_training_patients,
            "frozen_at": result.risk_thresholds.frozen_at,
            "source_score": result.risk_thresholds.source_score,
            "open_design_question": (
                "Tertile mi sabit olasilik esigi mi -- protokolde SAYISAL "
                "olarak KILITLENMEDI, Barisin onayi gerekiyor (bkz. "
                "pipeline/xgboost_model.py::compute_frozen_risk_thresholds())."
            ),
        },
        "reporting_rule_note": (
            "Tek-esik dili YASAK -- AUC/C-index icin daima nokta tahmini + "
            "%95 bootstrap CI birlikte raporlanir."
        ),
        "known_open_risks": [
            "Bu fonksiyon (_write_outputs/run_xgboost_week4_pipeline) ARTIK "
            "LEGACY/MISALIGNED yoldur (Codex CRITICAL 1, 2026-08-19) -- "
            "SADECE Ege'nin sizinti-karsilastirmasi icin calistirilmali. "
            "UCSF harici test AUC'si BU FONKSIYONDA hala YOK -- "
            "`--evaluate-ucsf-external` SADECE ALIGNED yolla (varsayilan) "
            "birlikte calisir, bkz. run_ucsf_external_evaluation()/"
            "_write_ucsf_external_outputs().",
            "_rebuild_variant_training_frame()'in run_single_variant() ile "
            "hasta sayisi/sutun kumesi ESLESTIGI GERCEK VERIYLE HENUZ "
            "DOGRULANMADI (bkz. o fonksiyonun docstring'i).",
            "twelve_month_threshold_days=365,0 (365,25 DEGIL) -- duyarlilik "
            "karsilastirmasi YAPILMADI.",
            "cox_in_sample_score icin TEK bir (penalizer,l1_ratio) cifti "
            "(son dis-fold'un secimi) kullanildi -- bu SADECE karsilastirma "
            "ciktisi oldugu icin kabul edilebilir, ama uretim kodu DEGIL.",
        ],
    }
    (output_dir / f"week4_{variant}_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _write_aligned_outputs(
    output_dir: Path, args: argparse.Namespace, result: "Week4AlignedPipelineResult"
) -> None:
    """`run_xgboost_week4_pipeline_aligned()`'in (YENİ birincil yol)
    çıktılarını yazar -- `_write_outputs()`'un ALIGNED karşılığı."""

    variant = args.variant
    aligned = result.aligned_result

    fold_copy = aligned.fold_results.copy()
    fold_copy.to_csv(output_dir / f"week4_{variant}_aligned_fold_results.csv", index=False)

    oof_copy = aligned.out_of_fold_predictions.copy()
    oof_copy["risk_class"] = result.risk_class_out_of_fold
    oof_copy.to_csv(output_dir / f"week4_{variant}_aligned_out_of_fold_predictions.csv")

    aligned.cox_score_audit.to_csv(output_dir / f"week4_{variant}_aligned_cox_score_audit.csv")

    if result.censoring_bias_summary is not None:
        result.censoring_bias_summary.to_csv(
            output_dir / f"week4_{variant}_aligned_censoring_bias_summary.csv", index=False
        )

    fold_audit_rows = [
        {
            "fold": audit.fold,
            "n_outer_train": len(audit.outer_train_patient_ids),
            "n_outer_test": len(audit.outer_test_patient_ids),
            "n_cox_fit_patients": len(audit.cox_fit_patient_ids),
            "cox_fit_equals_outer_train": audit.cox_fit_patient_ids == audit.outer_train_patient_ids,
            "cox_fit_disjoint_from_outer_test": not (audit.cox_fit_patient_ids & audit.outer_test_patient_ids),
            "best_cox_penalizer": audit.best_cox_penalizer,
            "best_cox_l1_ratio": audit.best_cox_l1_ratio,
            "cox_final_features": ";".join(sorted(audit.cox_final_features)),
            "best_xgb_params": audit.best_xgb_params,
            "fold_local_extra_columns": ";".join(audit.fold_local_extra_columns),
            "reduce_collinearity_enabled": audit.reduce_collinearity_enabled,
            "cox_candidate_features_before_collinearity_filter": (
                audit.cox_candidate_features_before_collinearity_filter
            ),
            "cox_candidate_features_after_collinearity_filter": (
                audit.cox_candidate_features_after_collinearity_filter
            ),
        }
        for audit in aligned.fold_audit
    ]
    pd.DataFrame(fold_audit_rows).to_csv(
        output_dir / f"week4_{variant}_aligned_fold_leakage_audit.csv", index=False
    )

    # Ş5 (Codex şartlı onay, 2026-09-11): `fold_audit_rows` yalnız DIŞ-final
    # önce/sonra aday SAYISINI taşıyordu (`cox_candidate_features_before/
    # after_collinearity_filter`) -- HANGİ özelliklerin/hangi gerekçeyle
    # (near-constant mi, hangi korelasyon kümesinin temsilcisi mi) elendiği
    # YOKTU. `aligned.collinearity_filter_reports` (HER outer/inner
    # `build_v3_candidate_pool()` çağrısı, `reduce_collinearity=False`
    # iken HER ZAMAN boş) burada iki dosyaya yazılır: (1) hızlı-taranabilir
    # özet CSV, (2) düşürülen özellik listeleri + korelasyon kümesi
    # detayları + eşiklerin TAMAMINI taşıyan JSON (mevcut audit deseniyle
    # tutarlı -- diğer JSON metadata dosyalarıyla AYNI `default=str` kalıbı).
    collinearity_summary_rows = [
        {
            "outer_fold": rep["outer_fold"],
            "inner_fold": rep["inner_fold"],
            "scope": rep["scope"],
            "cv_threshold": rep["cv_threshold"],
            "corr_threshold": rep["corr_threshold"],
            "n_input_features": rep["n_input_features"],
            "n_kept_features": rep["n_kept_features"],
            "n_dropped_total": rep["n_dropped_total"],
            "n_near_constant_dropped": rep["n_near_constant_dropped"],
            "n_correlation_dropped": rep["n_correlation_dropped"],
            "n_multi_member_clusters": rep["n_multi_member_clusters"],
        }
        for rep in aligned.collinearity_filter_reports
    ]
    pd.DataFrame(
        collinearity_summary_rows,
        columns=[
            "outer_fold",
            "inner_fold",
            "scope",
            "cv_threshold",
            "corr_threshold",
            "n_input_features",
            "n_kept_features",
            "n_dropped_total",
            "n_near_constant_dropped",
            "n_correlation_dropped",
            "n_multi_member_clusters",
        ],
    ).to_csv(output_dir / f"week4_{variant}_aligned_collinearity_filter_summary.csv", index=False)
    (output_dir / f"week4_{variant}_aligned_collinearity_filter_detail.json").write_text(
        json.dumps(aligned.collinearity_filter_reports, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _script_sha256(),
        "variant": variant,
        "pipeline": "aligned (CRITICAL 1 duzeltmesi, Codex task-mszuse9v-efm1gw)",
        "cli_args": vars(args) | {"output_dir": str(args.output_dir) if args.output_dir else None},
        "twelve_month_label_report": {
            "n_total": result.twelve_month_report.n_total,
            "n_survived_yes": result.twelve_month_report.n_survived_yes,
            "n_died_no": result.twelve_month_report.n_died_no,
            "n_excluded_ambiguous_censoring": result.twelve_month_report.n_excluded_ambiguous_censoring,
            "pct_excluded": result.twelve_month_report.pct_excluded,
            "threshold_days": result.twelve_month_report.threshold_days,
        },
        "n_patients_before_target_exclusion": result.n_patients_before_target_exclusion,
        "n_patients_after_target_exclusion": result.n_patients_after_target_exclusion,
        "xgboost_auc_out_of_fold": aligned.auc_out_of_fold,
        "structural_leakage_guarantee_note": (
            "Her outer-fold'da Cox fit hasta kumesi (ic-CV hiperparametre "
            "taramasi + elastic-net + stabilite + ic cross-fit + final fit) "
            "outer_train_patient_ids ile BIREBIR AYNI ve outer_test_patient_"
            "ids ile AYRIK -- bkz. week4_<variant>_aligned_fold_leakage_"
            "audit.csv + pipeline.xgboost_model.verify_aligned_fold_leakage_"
            "free() (bu script cagirir, basarisizsa RuntimeError ile durur)."
        ),
        "risk_thresholds": {
            "low_upper_bound": result.risk_thresholds.low_upper_bound,
            "medium_upper_bound": result.risk_thresholds.medium_upper_bound,
            "method": result.risk_thresholds.method,
            "n_training_patients": result.risk_thresholds.n_training_patients,
            "frozen_at": result.risk_thresholds.frozen_at,
            "source_score": result.risk_thresholds.source_score,
        },
        "reporting_rule_note": (
            "Tek-esik dili YASAK -- AUC/C-index icin daima nokta tahmini + "
            "%95 bootstrap CI birlikte raporlanir."
        ),
        "known_open_risks": [
            "twelve_month_threshold_days=365,0 (365,25 DEGIL) -- duyarlilik "
            "karsilastirmasi YAPILMADI.",
            "cox_penalizer_grid/cox_l1_ratio_grid HER outer-fold'da AYRI "
            "taraniyor (dogru/sizintisiz) ama bu maliyetlidir -- gercek "
            "611/585 veriyle calisma suresi GERCEK KOSUDA olculmedi.",
        ],
    }
    (output_dir / f"week4_{variant}_aligned_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _write_ucsf_external_outputs(
    output_dir: Path, args: argparse.Namespace, bundle: "UCSFExternalEvaluationBundle"
) -> None:
    """`run_ucsf_external_evaluation()`'ın (MEDIUM 5) çıktısını yazar."""

    variant = args.variant
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "note": (
            "Yalniz-UPenn'de fit edilen nihai Cox->XGBoost zinciri, YENIDEN "
            "FIT/ESIK SECIMI YAPMADAN, UCSF'te degerlendirildi. Eski "
            "'harici test TCGA' tarifi 2026-08-18'den beri GECERSIZ -- "
            "harici test UCSF'tir. HIGH 5 (Codex 2. tur, 2026-08-19): bu "
            "kosudan ONCEKI (2026-08-19 ilk smoke, ucsf_auc=0,7137) sonuc "
            "TUTARSIZ standardizasyon receteyle (final/external HAM olcek, "
            "aligned CV fold-yerel standardize) uretilmisti -- GECERSIZ "
            "sayilir. Bu kosu UPenn full-pool istatistikleriyle standardize "
            "edilmis training/external frame kullanir (bkz. "
            "clinical_standardize_columns/means/stds)."
        ),
        "n_upenn_training_patients": bundle.n_upenn_training_patients,
        "n_upenn_training_events": bundle.n_upenn_training_events,
        # Ş4 (Codex sartli onay, 2026-09-11): iki Cox havuzu -- "dagitim"
        # (cox_final, tam training_frame) vs "cross-fit" (XGBoost meta-
        # skoru, labeled_frame -- 12-ay hedefi ambiguous-sansurluleri
        # DUSURDUKTEN sonraki kume) -- BIREBIR AYNI RECETE DEGILDIR,
        # BILINCLI FARK. n_upenn_training_* = dagitim havuzu,
        # n_cross_fit_training_* = cross-fit havuzu.
        "cox_pool_recipe_note": (
            "n_upenn_training_* (dagitim/cox_final havuzu, tam training_frame) "
            "ile n_cross_fit_training_* (XGBoost meta-skoru icin labeled_frame, "
            "12-ay hedefi ambiguous-sansurluleri dusurulmus) BIREBIR AYNI "
            "receteye ait DEGILDIR -- bu bilincli bir fark (bkz. "
            "fit_final_cox_and_xgboost_pipeline_on_full_pool() docstring'i)."
        ),
        "n_cross_fit_training_patients": bundle.n_cross_fit_training_patients,
        "n_cross_fit_training_events": bundle.n_cross_fit_training_events,
        "cox_final_features": bundle.cox_final_features,
        "cox_best_penalizer": bundle.cox_best_penalizer,
        "cox_best_l1_ratio": bundle.cox_best_l1_ratio,
        "xgboost_best_params": bundle.xgboost_best_params,
        # HIGH 5 (Codex 2. tur): final Cox+XGBoost VE UCSF, UPenn full-pool
        # istatistikleriyle AYNI olcege tasindi -- hangi kolonlarin
        # standardize edildigi + ogrenilen ortalama/std burada kayitli
        # (aligned CV'nin fold-yerel standardizasyonuyla TUTARLI recete).
        "clinical_standardize_columns": bundle.clinical_standardize_columns,
        "clinical_standardize_means": bundle.clinical_standardize_means,
        "clinical_standardize_stds": bundle.clinical_standardize_stds,
        # v3 recete-sadakati (2026-09-12): `cox_final`'in GERCEKTEN
        # standardize ettigi TAM kolon listesi -- `reduce_collinearity`
        # ACIKKEN yukaridaki `clinical_standardize_columns` (yalniz yas)
        # ile AYNI DEGILDIR, filtrelenmis radyomik havuzu da icerir
        # (`tools/train_cox_week3.py::V3VariantConfig.standardize_
        # radiomics=True` ile AYNI recete). Sessiz filtreleme yasak
        # ilkesi geregi burada AYRICA raporlanir.
        "cox_final_clinical_standardize_columns": bundle.cox_final_clinical_standardize_columns,
        "ucsf_auc": bundle.evaluation.auc,
        "ucsf_twelve_month_report": {
            "n_total": bundle.evaluation.twelve_month_report.n_total,
            "n_survived_yes": bundle.evaluation.twelve_month_report.n_survived_yes,
            "n_died_no": bundle.evaluation.twelve_month_report.n_died_no,
            "n_excluded_ambiguous_censoring": bundle.evaluation.twelve_month_report.n_excluded_ambiguous_censoring,
            "pct_excluded": bundle.evaluation.twelve_month_report.pct_excluded,
        },
        "n_patients_evaluated": bundle.evaluation.n_patients_evaluated,
        "reporting_rule_note": (
            "Tek-esik dili YASAK -- AUC icin daima nokta tahmini + %95 "
            "bootstrap CI birlikte raporlanir."
        ),
    }
    (output_dir / f"week4_{variant}_ucsf_external_evaluation.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
