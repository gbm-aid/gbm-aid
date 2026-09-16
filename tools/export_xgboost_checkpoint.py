"""`v2a_mgmt + --reduce-collinearity` (XGBoost, 2026-09-12 uretim kosusu)
icin CANLI DEPLOY checkpoint'i uretir.

BAGLAM (2026-09-13, Baris talimati "XGBoost'un canli zincire baglanmasi,
ADIM 1"): 2026-09-12'de kosulan XGBoost uretim kosusu (bkz. `entities/
xgboost-kosu-katalogu.md`, `model_registry.model_id=10`,
`status='shadow'`) ic OOF AUC 0,7226 [0,6801-0,7605] (n=606/341) ve UCSF
harici AUC 0,7237 [0,6508-0,7903] (n=232) urettti, AMA `models/`
dizinine SERVIS EDILEBILIR bir .pkl BIRAKMADI -- `tools/train_xgboost_
week4.py::run_ucsf_external_evaluation()` (nihai Cox->XGBoost zincirini
UCSF'te DEGERLENDIREN fonksiyon) kendi urettigi `cox_final`/
`xgboost_final` fit nesnelerini DISKE YAZMADAN, SADECE ozet metrikleri
(`week4_v2a_mgmt_ucsf_external_evaluation.json`) dondurup atar.

YONTEM -- "yeniden calistir, KARSILASTIR" (`tools/export_v3b_deployment_
checkpoint.py`'nin AYNI deseni, Cox tarafinda 2026-09-12'de kullanildi):
bu script `run_ucsf_external_evaluation()`'in YAPTIGI HER ADIMI, o
fonksiyonun GERCEK ozel (alt-cizgili) yardimcilarini (`tools.train_
xgboost_week4._rebuild_variant_training_frame`, `._prepare_full_pool_
standardized_frames`, `._fit_final_model_on_full_pool_with_penalizer_
escalation`, `._validate_ucsf_distribution_feature_pool_disjoint`) VE
`pipeline.xgboost_model`/`pipeline.reduce_collinearity`/`tools.train_cox_
week3`'un GERCEK genel fonksiyonlarini BIREBIR AYNI sirada, AYNI CLI
argumanlariyla (`train_xgboost_week4.parse_args(["--variant", "v2a_mgmt",
"--reduce-collinearity", "--evaluate-ucsf-external"])` varsayilanlari)
CAGIRIR -- REIMPLEMENTE ETMEZ. `tools/train_xgboost_week4.py`/`tools/
train_cox_week3.py`/`pipeline/*.py`'ye TEK SATIR bile YAZILMADI.

TEK fark: `run_ucsf_external_evaluation()` `cox_final`/`xgboost_final`
nesnelerini yerel degisken olarak birakip fonksiyondan CIKARKEN atar --
bu script AYNI adimlari kendi govdesinde tekrarlayip bu nesneleri
SAKLAR (checkpoint'in ta kendisi budur).

DOGRULAMA (ZORUNLU, bu script ICINDE yapilir) -- IKI BAGIMSIZ KAYNAKLA:
  1. `artifacts/week4/xgboost_v2a_mgmt_reduce_collinearity_0912/week4_
     v2a_mgmt_ucsf_external_evaluation.json` -- 2026-09-12 uretim
     kosusunun (BAGIMSIZ bir process calistirmasi, AYNI seed=42) ozet
     ciktisi. `cox_final_features`/`cox_best_penalizer`/`cox_best_l1_
     ratio`/`xgboost_best_params`/`clinical_standardize_means`/`stds`/
     `ucsf_auc` (auc/ci_lower/ci_upper/n_patients/n_positive) BIREBIR
     (tol=1e-9 mutlak/1e-6 bagil) KARSILASTIRILIR.
  2. `models/cox_phm_v3b_lowvar_v2amgmt_2026-09-12.pkl` -- `cox_final`
     ile AYNI recete oldugu (katalog: "16/16 ozellik ayni, 24/24 katsayi
     farki 0,000000") BAGIMSIZ olarak, bu script'in KENDI ic hesabiyla
     yeniden dogrulanir (fitted_model.params_ Series karsilastirmasi).

Eslesmezse checkpoint YAZILMAZ, AssertionError firlar (fail-closed --
`tools/export_v3b_deployment_checkpoint.py` ile AYNI ilke).

⚠️ NE OLCULDU / NE OLCULEMEDI (uydurma-eslesme iddiasi KURULMAZ):
  - OLCULEN: bu script'in ürettigi `cox_final`/`xgboost_final` fit'inin,
    2026-09-12'deki BAGIMSIZ process calistirmasiyla (ayni seed/veri/
    kod) SAYISAL olarak (katsayi + hiperparametre + harici AUC
    duzeyinde) eslesmesi -- bu, "iki kez calistir, karsilastir"
    testinin GUCLU bir bicimidir (iki AYRI process invocation'i,
    AYNI recete).
  - OLCULEMEYEN: full-pool final fit'in nested-CV OOF AUC'siyle (0,7226)
    "ayni sayi" olmasi -- bunlar FARKLI seyler olcer (biri TUM havuzda
    TEK fit, digeri 5-disi-foldlu CV) ve KASITLI OLARAK karsilastirilmadi
    (gorev talimati). Full-pool fit'in KENDI ic-CV grid taramasi
    `min_child_weight=1` sectigi (fold-bazli taramalarda hep 5) --
    katalogda ONCEDEN not edilmis, aciklanmamis bir fark, bu script
    TEKRAR olcup DEGISTIRMEDI.
  - Bu script'in KENDI SURECINDE (ayni process, iki ardisik cagri) bir
    "ikinci kez calistir" testi YAPILMADI -- gerekce: tam adim zinciri
    (5 cross-fit Cox refit + XGBoost hiperparametre grid CV + UCSF
    bootstrap) DB'ye bagli ve ~15-30 dk suruyor (bkz. katalog "Bitis"
    zaman damgalari), COK PROSESLI ikinci bir tam kosuyu bu oturumda
    tekrarlamak orantisiz maliyetli. Yukaridaki (1) karsilastirmasi
    BUNUN YERINE GECER: 2026-09-12'deki kosu da AYNI seed'lerle,
    TAMAMEN AYRI bir process'te uretildi -- iki-ayri-calistirma testinin
    ozü zaten saglanmis durumda.

Cikti: `models/xgboost_v2a_mgmt_reduce_collinearity_2026-09-13.pkl`
(proje `models/` dizinine, iki mevcut Cox .pkl'ine DOKUNMADAN).

Checkpoint SOZLESMESI (bkz. modul-ici docstring `_build_checkpoint()`):
iki AYRI olcek tasir -- `cox_score` blogu (yas+54 filtreli radyomik
standardize edilmis) ve `xgboost` blogu (yas standardize, 93 radyomik
HAM) -- bunlarin KARISTIRILMAMASI backend-agent'in ADIM 2'sinin en
kritik riskidir (Cox tarafinda 2026-09-12'de ayni riskle karsilasildi,
bkz. bu dosyanin modul docstring'i).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import train_cox_week3 as week3  # noqa: E402  (salt-okunur import)
import train_xgboost_week4 as t4  # noqa: E402  (salt-okunur import)
import pipeline.reduce_collinearity as reduce_collinearity_module  # noqa: E402
from pipeline.cox_model import standardize_columns_fold_safe  # noqa: E402
from pipeline.xgboost_model import (  # noqa: E402
    DEFAULT_COX_PENALIZER_ESCALATION_GRID,
    apply_twelve_month_target,
    define_twelve_month_survival_target,
    evaluate_xgboost_external_test,
    fit_final_cox_and_xgboost_pipeline_on_full_pool,
    verify_full_pool_pipeline_leakage_free,
)
from db_connection import get_connection  # noqa: E402
from psycopg2.extras import RealDictCursor  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "backend" / "models"  # 2026-09-16: models/ backend/ altina tasindi
ARTIFACT_DIR = (
    PROJECT_ROOT / "artifacts" / "week4" / "xgboost_v2a_mgmt_reduce_collinearity_0912"
)
REFERENCE_UCSF_JSON = ARTIFACT_DIR / "week4_v2a_mgmt_ucsf_external_evaluation.json"
COX_V3B_CHECKPOINT_PATH = MODELS_DIR / "cox_phm_v3b_lowvar_v2amgmt_2026-09-12.pkl"
CHECKPOINT_OUT_PATH = MODELS_DIR / "xgboost_v2a_mgmt_reduce_collinearity_2026-09-13.pkl"

VARIANT_NAME = "v2a_mgmt"
ABS_TOL = 1e-9
REL_TOL = 1e-6


def _assert_close(label: str, got: float, want: float) -> float:
    diff = abs(float(got) - float(want))
    rel = diff / max(abs(float(want)), 1e-12)
    if diff > ABS_TOL and rel > REL_TOL:
        raise AssertionError(
            f"[UYUSMUYOR] {label}: yeniden-uretilen={got!r} referans={want!r} "
            f"abs_diff={diff!r} rel_diff={rel!r}. Checkpoint YAZILMADI."
        )
    return diff


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_OUT_PATH.exists():
        raise FileExistsError(
            f"{CHECKPOINT_OUT_PATH} ZATEN VAR -- uzerine YAZILMAZ (v3b "
            "export script'inin secilen yontemiyle AYNI: yeni dosya adi)."
        )
    if not REFERENCE_UCSF_JSON.is_file():
        raise FileNotFoundError(f"Referans JSON yok: {REFERENCE_UCSF_JSON}")
    if not COX_V3B_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Referans Cox checkpoint yok: {COX_V3B_CHECKPOINT_PATH}")

    reference = json.loads(REFERENCE_UCSF_JSON.read_text(encoding="utf-8"))
    with COX_V3B_CHECKPOINT_PATH.open("rb") as fh:
        v3b_checkpoint = pickle.load(fh)

    # ---- 1) args -- 2026-09-12 uretim komutuyla BIREBIR ayni (bkz.
    # entities/xgboost-kosu-katalogu.md SS0 "Komut") ----
    args = t4.parse_args(
        ["--variant", VARIANT_NAME, "--reduce-collinearity", "--evaluate-ucsf-external"]
    )
    assert args.seed == 42
    assert args.cox_inner_splits == 5
    assert args.cox_n_bootstrap_stability == 200
    assert args.cox_primary_threshold == 0.6
    assert args.cox_extra_column_penalizer == 0.0
    assert args.reduce_collinearity is True
    assert args.collinearity_cv_threshold == 0.02
    assert args.collinearity_corr_threshold == 0.95
    assert args.xgb_inner_splits == 5
    assert args.xgb_n_bootstrap_ci == 1000
    assert args.twelve_month_threshold_days == 365.0
    assert args.expected_patient_count == 611
    assert args.expected_event_count == 585
    assert args.expected_ucsf_patient_count == 295
    assert args.expected_ucsf_event_count == 169
    assert args.evaluate_ucsf_external is True

    config = week3.select_v2_variant_configs([VARIANT_NAME])[0]
    assert config.include_mgmt is True
    assert config.use_age_spline is False
    print(f"[dogrulama] config.regions={config.regions}", flush=True)

    # ---- 2) veri -- caching YOK (paylasilan cache dizinine dokunmamak
    # icin bilincli tercih -- 5 ajan paralel calisiyor, AKTIF-GOREVLER.md) ----
    connection = get_connection(readonly=True)
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        (
            training_frame,
            feature_columns,
            extra_columns,
            clinical_standardize_columns,
        ) = t4._rebuild_variant_training_frame(cursor=cursor, config=config, args=args)

        ucsf_long = week3.fetch_c32_radiomics_long_frame(
            cursor, segmentation_tool=week3.SEGMENTATION_TOOL_UCSF_C32
        )
        week3.raise_if_empty_c32(ucsf_long, segmentation_tool=week3.SEGMENTATION_TOOL_UCSF_C32)
        ucsf_patients = week3.fetch_patients_frame(cursor, source_name=week3.UCSF_SOURCE_NAME)
        cursor.close()
    finally:
        connection.close()

    print(
        f"[dogrulama] training_frame n={len(training_frame)} "
        f"events={int(training_frame['event'].sum())} (beklenen 611/585)",
        flush=True,
    )
    assert len(feature_columns) == 93, f"beklenen 93 aday radyomik, alinan {len(feature_columns)}"

    t4._validate_ucsf_distribution_feature_pool_disjoint(feature_columns, clinical_standardize_columns)

    ucsf_wide, _ = week3.pivot_ucsf_regions_long_to_wide(ucsf_long, regions=list(config.regions))
    ucsf_clinical = week3.build_variant_clinical_frame(
        ucsf_patients,
        include_mgmt=config.include_mgmt,
        use_age_spline=config.use_age_spline,
        age_knots=None,
    )
    ucsf_patients_c = ucsf_patients.join(ucsf_clinical)
    external_frame, _ = week3.build_external_test_frame(
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
    print(
        f"[dogrulama] external_frame n={len(external_frame)} "
        f"events={int(external_frame['event'].sum())} (beklenen 295/169)",
        flush=True,
    )

    # ---- 3) B9-FIX2 -- cox_final'in aday havuzu, XGBoost'un cross-fit
    # meta-skoru URETEN filtreyle AYNI (train_xgboost_week4.py satir
    # ~1067-1088) ----
    cox_final_feature_columns, _cox_final_collinearity_report = (
        reduce_collinearity_module.build_v3_candidate_pool(
            training_frame,
            feature_columns,
            cv_threshold=args.collinearity_cv_threshold,
            corr_threshold=args.collinearity_corr_threshold,
        )
    )
    if not cox_final_feature_columns:
        raise RuntimeError(
            "cox_final (harici/dagitim skorlama modeli): reduce_collinearity "
            "havuzu SIFIR ozellik birakti."
        )
    print(
        f"[dogrulama] cox_final aday havuzu: {len(feature_columns)} -> "
        f"{len(cox_final_feature_columns)} ozellik.",
        flush=True,
    )

    cox_final_clinical_standardize_columns = clinical_standardize_columns + list(
        cox_final_feature_columns
    )

    (
        full_pool_training_frame,
        full_pool_external_frame,
        clinical_standardize_means,
        clinical_standardize_stds,
    ) = t4._prepare_full_pool_standardized_frames(
        training_frame, external_frame, cox_final_clinical_standardize_columns
    )
    xgb_meta_training_frame, xgb_meta_external_frame = standardize_columns_fold_safe(
        training_frame, external_frame, clinical_standardize_columns
    )

    # ---- 4) cox_final -- TEK, dagitilabilir Cox modeli (UCSF/uretim
    # skorlamasi icin) ----
    cox_final_fit_failure_log: list[Any] = []
    cox_final = t4._fit_final_model_on_full_pool_with_penalizer_escalation(
        full_pool_training_frame,
        cox_final_feature_columns,
        extra_columns=extra_columns,
        n_bootstrap_stability=args.cox_n_bootstrap_stability,
        stability_frequency_threshold=args.cox_primary_threshold,
        seed=args.seed,
        extra_column_penalizer=args.cox_extra_column_penalizer,
        penalizer_escalation_grid=DEFAULT_COX_PENALIZER_ESCALATION_GRID,
        failure_sink=cox_final_fit_failure_log,
    )
    print(
        f"[dogrulama] cox_final.final_features n={len(cox_final.final_features)} "
        f"penalizer={cox_final.best_penalizer} l1_ratio={cox_final.best_l1_ratio}",
        flush=True,
    )

    # ---- 5) XGBoost'un KENDI cross-fit meta-skoru + full-pool fit'i ----
    target, twelve_month_report = define_twelve_month_survival_target(
        xgb_meta_training_frame, threshold_days=args.twelve_month_threshold_days
    )
    labeled_frame = apply_twelve_month_target(
        xgb_meta_training_frame, target, report=twelve_month_report
    )

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
        cox_penalizer_escalation_grid=DEFAULT_COX_PENALIZER_ESCALATION_GRID,
        cox_fit_failure_log=[],
    )
    verify_full_pool_pipeline_leakage_free(xgboost_final, labeled_frame.index)
    print(
        f"[dogrulama] xgboost_final.best_params={xgboost_final.best_params} "
        f"n_patients_used={xgboost_final.n_patients_used}",
        flush=True,
    )

    evaluation = evaluate_xgboost_external_test(
        cox_final.fitted_model,
        cox_final.final_features,
        xgboost_final,
        full_pool_external_frame,
        extra_columns=extra_columns,
        twelve_month_threshold_days=args.twelve_month_threshold_days,
        n_bootstrap_ci=args.xgb_n_bootstrap_ci,
        seed=args.seed,
        xgboost_external_frame=xgb_meta_external_frame,
    )
    print(
        f"[dogrulama] UCSF harici auc={evaluation.auc['auc']:.6f} "
        f"[{evaluation.auc['ci_lower']:.6f}-{evaluation.auc['ci_upper']:.6f}] "
        f"n={evaluation.n_patients_evaluated} "
        f"(referans/katalog: {reference['ucsf_auc']['auc']:.6f} "
        f"[{reference['ucsf_auc']['ci_lower']:.6f}-{reference['ucsf_auc']['ci_upper']:.6f}])",
        flush=True,
    )

    # =====================================================================
    # ---- 6) BIT-BIREBIR/SIKI-TOLERANSLI KANIT -- KAYNAK 1: 2026-09-12
    # BAGIMSIZ process-calistirmasinin JSON ozeti ----
    # =====================================================================
    if set(cox_final.final_features) != set(reference["cox_final_features"]):
        raise AssertionError(
            "cox_final.final_features referans JSON ile UYUSMUYOR -- "
            f"yeniden-uretilen={sorted(cox_final.final_features)} "
            f"referans={sorted(reference['cox_final_features'])}"
        )
    _assert_close("cox_best_penalizer", cox_final.best_penalizer, reference["cox_best_penalizer"])
    _assert_close("cox_best_l1_ratio", cox_final.best_l1_ratio, reference["cox_best_l1_ratio"])

    if dict(xgboost_final.best_params) != dict(reference["xgboost_best_params"]):
        raise AssertionError(
            f"xgboost_final.best_params referans ile UYUSMUYOR -- "
            f"yeniden-uretilen={xgboost_final.best_params} "
            f"referans={reference['xgboost_best_params']}"
        )

    if set(cox_final_clinical_standardize_columns) != set(
        reference["cox_final_clinical_standardize_columns"]
    ):
        raise AssertionError(
            "cox_final_clinical_standardize_columns referans ile UYUSMUYOR."
        )
    max_standardize_diff = 0.0
    for column in cox_final_clinical_standardize_columns:
        max_standardize_diff = max(
            max_standardize_diff,
            _assert_close(
                f"clinical_standardize_means[{column}]",
                clinical_standardize_means[column],
                reference["clinical_standardize_means"][column],
            ),
        )
        max_standardize_diff = max(
            max_standardize_diff,
            _assert_close(
                f"clinical_standardize_stds[{column}]",
                clinical_standardize_stds[column],
                reference["clinical_standardize_stds"][column],
            ),
        )

    ref_auc = reference["ucsf_auc"]
    assert evaluation.n_patients_evaluated == ref_auc["n_patients"], (
        f"n_patients_evaluated {evaluation.n_patients_evaluated} != referans {ref_auc['n_patients']}"
    )
    max_auc_diff = max(
        _assert_close("ucsf_auc.auc", evaluation.auc["auc"], ref_auc["auc"]),
        _assert_close("ucsf_auc.ci_lower", evaluation.auc["ci_lower"], ref_auc["ci_lower"]),
        _assert_close("ucsf_auc.ci_upper", evaluation.auc["ci_upper"], ref_auc["ci_upper"]),
    )
    print(
        f"[KANIT 1/2] 2026-09-12 referans JSON ile eslesti "
        f"(max standardizasyon farki={max_standardize_diff:.3e}, "
        f"max AUC farki={max_auc_diff:.3e}).",
        flush=True,
    )

    # =====================================================================
    # ---- 7) BIT-BIREBIR KANIT -- KAYNAK 2: v3b Cox checkpoint'inin
    # fitted_model.params_ ile dogrudan karsilastirma ----
    # =====================================================================
    v3b_fitted = v3b_checkpoint["fitted_model"]
    v3b_params = v3b_fitted.params_
    cox_final_params = cox_final.fitted_model.params_
    if set(cox_final_params.index) != set(v3b_params.index):
        raise AssertionError(
            "cox_final.fitted_model.params_ kolonlari v3b checkpoint'i ile "
            f"UYUSMUYOR -- yeniden-uretilen={sorted(cox_final_params.index)} "
            f"v3b={sorted(v3b_params.index)}"
        )
    max_coef_diff = 0.0
    for column in cox_final_params.index:
        max_coef_diff = max(
            max_coef_diff,
            _assert_close(
                f"cox_final vs v3b katsayi[{column}]",
                float(cox_final_params[column]),
                float(v3b_params[column]),
            ),
        )
    print(
        f"[KANIT 2/2] cox_final.fitted_model, models/cox_phm_v3b_lowvar_"
        f"v2amgmt_2026-09-12.pkl ile {len(cox_final_params)}/{len(cox_final_params)} "
        f"kolonda eslesti (max|fark|={max_coef_diff:.3e}).",
        flush=True,
    )

    # ---- 8) checkpoint sozlugu -- backend-agent icin IKI AYRI olcek ----
    cox_feature_standardization: dict[str, dict[str, float]] = {
        column: {
            "mean": float(clinical_standardize_means[column]),
            "std": float(clinical_standardize_stds[column]),
        }
        for column in cox_final_clinical_standardize_columns
    }
    age_column = week3.CLINICAL_AGE_COLUMN
    age_mean = float(clinical_standardize_means[age_column])
    age_std = float(clinical_standardize_stds[age_column])

    checkpoint = {
        "arm_name": "v2a_mgmt+reduce_collinearity-aligned-v3b_bitidentical",
        "model_registry_version_hint": "v2a_mgmt+reduce_collinearity-aligned-v3b_bitidentical_2026-09-12",
        "twelve_month_threshold_days": float(args.twelve_month_threshold_days),
        # ---- ADIM (a): cox_score -- XGBoost'un score_column girdisini
        # ureten TEK dagitilabilir Cox modeli. BU OLCEK: yas + (reduce_
        # collinearity filtresinden gecmis) 54 radyomik STANDARDIZE
        # edilmis (UPenn full-pool istatistiginden). `xgboost` blogundaki
        # HAM radyomik olcegiyle KARISTIRILMAMALI. ----
        "cox_score": {
            "fitted_model": cox_final.fitted_model,
            "final_features": list(cox_final.final_features),
            "clinical_extra_columns": list(extra_columns),
            "feature_standardization": cox_feature_standardization,
            "usage_note": (
                "cox_oof_score = fitted_model.predict_log_partial_hazard(frame[final_features + "
                "clinical_extra_columns]) -- frame icindeki final_features (radyomik, RAW degerden) "
                "VE 'clinical_age' feature_standardization[col] ile z-skorlanmis OLMALI; digger "
                "clinical_extra_columns (gender/gtr/idh/mgmt) 0/1 HAM kalir (feature_standardization'da "
                "yer almazlar)."
            ),
        },
        # ---- ADIM (b): xgboost -- final tahmin modeli. BU OLCEK: SADECE
        # yas standardize, 93 radyomik HAM (raw_radiomic_feature_columns). ----
        "xgboost": {
            "fitted_model": xgboost_final.fitted_model,
            "xgb_feature_columns": list(xgboost_final.xgb_feature_columns),
            "score_column": xgboost_final.score_column,
            "best_params": dict(xgboost_final.best_params),
            "raw_radiomic_feature_columns": list(feature_columns),
            "clinical_extra_columns": list(extra_columns),
            "age_standardization": {"column": age_column, "mean": age_mean, "std": age_std},
            "usage_note": (
                "predict_proba girdisi = frame[xgb_feature_columns] -- xgb_feature_columns = "
                "raw_radiomic_feature_columns (93, HAM) + clinical_extra_columns (8, 'clinical_age' "
                "age_standardization ile z-skorlanmis, digerleri HAM 0/1) + [score_column] (cox_score "
                "blogundan uretilen TEK deger). 'clinical_age' ICIN age_standardization BU BLOKTAKI "
                "ile cox_score blogundaki feature_standardization['clinical_age'] SAYISAL OLARAK AYNIDIR "
                "(ikisi de training_frame['clinical_age']'den, ayni formulle hesaplanir) -- yine de IKI "
                "AYRI ALANDA tutulur (bagimsiz okuma/hata izolasyonu icin), TEK bir age z-skoru hesaplayip "
                "HER IKI blokta da kullanmak yeterlidir."
            ),
        },
        "provenance": {
            "generated_by": "tools/export_xgboost_checkpoint.py",
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "reproduced_from": (
                "tools/train_xgboost_week4.py::run_ucsf_external_evaluation() "
                "(--variant v2a_mgmt --reduce-collinearity --evaluate-ucsf-external), canli DB (2026-09-13)"
            ),
            "n_train": int(len(training_frame)),
            "n_events_train": int(training_frame["event"].sum()),
            "n_ucsf_external": int(len(external_frame)),
            "n_events_ucsf_external": int(external_frame["event"].sum()),
            "verification": {
                "reference_json": str(REFERENCE_UCSF_JSON.relative_to(PROJECT_ROOT)),
                "reference_cox_checkpoint": str(COX_V3B_CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
                "max_standardize_abs_diff_vs_reference_json": max_standardize_diff,
                "max_ucsf_auc_abs_diff_vs_reference_json": max_auc_diff,
                "max_cox_coefficient_abs_diff_vs_v3b_checkpoint": max_coef_diff,
                "ucsf_external_auc_reproduced": {
                    "auc": evaluation.auc["auc"],
                    "ci_lower": evaluation.auc["ci_lower"],
                    "ci_upper": evaluation.auc["ci_upper"],
                    "n_patients": evaluation.n_patients_evaluated,
                },
                "not_verified": (
                    "Bu checkpoint'in full-pool fit'i, 2026-09-12 kosusunun ALIGNED 5-disi-fold "
                    "nested-CV OOF AUC'siyle (0,7226 [0,6801-0,7605]) KARSILASTIRILMADI -- bu ikisi "
                    "FARKLI seyler olcer (full-pool TEK fit vs 5-fold CV), 'ayni sayi' iddiasi "
                    "KURULMAZ. Ayrica bu script'in SURECI ICINDE bagimsiz bir 'ikinci kosu' "
                    "YAPILMADI (~15-30dk/kosu, DB-bagli); yerine 2026-09-12'nin BAGIMSIZ process "
                    "ciktisiyla (KAYNAK 1) ve v3b checkpoint'iyle (KAYNAK 2) karsilastirma kanit "
                    "olarak kullanildi."
                ),
            },
        },
    }

    with CHECKPOINT_OUT_PATH.open("wb") as fh:
        pickle.dump(checkpoint, fh)
    print(f"[YAZILDI] {CHECKPOINT_OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
