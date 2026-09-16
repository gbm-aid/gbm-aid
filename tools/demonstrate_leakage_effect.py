"""In-sample (YANLIS) vs out-of-fold (DOGRU) XGBoost AUC gosterimi -- GERCEK veriyle.

=====================================================================
NEDEN BU DOSYA VAR
=====================================================================
`raw/mimari/v45.txt` satir 1760-1761 ve `raw/plan/plan.txt` satir 508
ACIKCA sunu sart kosuyor:

    "Nested/out-of-fold Cox skorlariyla uretilmis XGBoost AUC'si ile
     in-sample (yanlis) hesaplanmis AUC arasindaki fark CI raporunda
     ayrica gosterilir."

Bu gosterim 2026-09-14'e kadar HIC URETILMEDI. Kok neden (2026-09-14'te
bulundu): gosterim kodu `tools/train_xgboost_week4.py::run_xgboost_week4_
pipeline()` (LEGACY / HIZALI-OLMAYAN yol) icinde yasiyor; 2026-09-12
uretim kosusu ise `run_xgboost_week4_pipeline_aligned()` (ALIGNED yol,
CRITICAL 1 duzeltmesi) ile kosuldu ve O yolda in-sample hesabi HIC YOK.
Kimse adimi atlamadi -- iki yol ayristiginda gosterim yanlis tarafta
kaldi. Bu script, uretim kosusunun ALIGNED cercevesini yeniden kurup
eksik sayilari ORADAN uretir.

=====================================================================
HANGI KAPILARDAN GECIYOR (ADIM 1 -- DOGRULAMA KAPISI)
=====================================================================
ADIM 2'nin (uc AUC) anlamli olmasi, cerceve rekonstruksiyonunun
uretim kosusuyla BIREBIR ayni olmasina baglidir. Yanlis kurulmus bir
cerceveden uretilen in-sample sayisi ANLAMSIZ ve ZARARLIDIR. Bu yuzden
ADIM 2'ye gecmeden once su kapilar CALISTIRILIR ve hepsi GECMEK
ZORUNDADIR (biri duserse script ADIM 2'yi KOSMADAN non-zero ile
cikar):

  G1  Egitim havuzu  : 611 hasta / 585 olum olayi (CLAUDE.md kilitli)
  G2  12-ay etiketli : 606 hasta (341 Yes / 265 No), 5 belirsiz dislandi
  G3  Ozellik uzayi  : 93 radyomik (WT, ICC>=0,60) + 8 klinik
  G4  Dis-fold bolunmesi, uretim kosusunun kaydettigi fold atamasiyla
      (week4_v2a_mgmt_aligned_out_of_fold_predictions.csv) BIREBIR ayni
  G5  Kolinearite filtresi (reduce_collinearity, CV<0,02 / |r|>=0,95)
      HER dis fold'da yeniden hesaplanip uretimin kaydettigi
      `week4_v2a_mgmt_aligned_collinearity_filter_detail.json`
      (scope=outer_final) ile BIREBIR ayni dusen-ozellik kumesini
      vermek zorunda
  G6  Her fold'un XGBoost tahmin olasiliklari, uretimin kaydettigi
      olasiliklarla <=1e-9 icinde ayni (uretimin kaydettigi
      out-of-fold Cox skorlari + kaydettigi en iyi XGB parametreleri
      + fold-yerel standardizasyon ile yeniden fit edilerek)
  G7  Yeniden hesaplanan out-of-fold AUC + %95 bootstrap CI, uretimin
      `week4_v2a_mgmt_aligned_run_metadata.json`'undaki degerle
      (AUC 0,7226470425496597 [0,6800539604349822-0,7605023576803478])
      <=1e-9 icinde ayni

=====================================================================
ADIM 2 -- URETILEN SAYILAR
=====================================================================
  (1) auc_out_of_fold_correct          DOGRU hat. Uretim kosusunun
      sayisi; bu script onu G6/G7 ile yeniden URETIR (kopyalamaz).
  (2) auc_fully_in_sample              !! YANLIS UC. XGBoost cercevenin
      TAMAMINI hem egitir hem skorlar (`pipeline.xgboost_model.
      generate_in_sample_xgboost_auc()`). plan.txt'nin literal istedigi
      en uc/en basit yanlis karsilastirma.
  (3) auc_out_of_fold_leaky_cox_input  !! YANLIS UC. Cox skoru girdisi
      in-sample (tum havuzda fit + tum havuzu skorla), XGBoost'un KENDI
      capraz-dogrulamasi HALA durust. v45'in "juri sorusunu"
      (Cox -> XGBoost sizintisi) IZOLE eder.
  (3k) auc_matched_control_honest_cox_input  (3)'un ESLESTIRILMIS
      KONTROLU: AYNI legacy XGBoost nested-CV makinesi, ama Cox skoru
      girdisi DURUST (uretimin out-of-fold skorlari). (3) ile yalniz
      ve yalniz Cox skorunun sizintili olup olmamasi bakimindan
      farklidir -- (3) ile karsilastirilmasi gereken sayi (1) DEGIL,
      BUDUR (cunku (1) ALIGNED makineden gelir).

!! DIL SINIRI: (2) ve (3) ASLA "sonuc" olarak sunulmaz -- bunlar
karsilastirmanin YANLIS ucudur. "Sizinti yoktur" gibi kesin bir cumle
de KURULMAZ; kurulabilecek cumle "sizintili ve sizintisiz hesaplamalar
arasindaki fark olculmus ve raporlanmistir" bicimindedir. Tek-esik dili
YASAK -- her AUC nokta tahmini + %95 bootstrap CI ile verilir.

=====================================================================
SINIRLAR
=====================================================================
* DB'ye YALNIZ SELECT yapar (`get_connection(readonly=True)`).
* Uretim artefakt dizinine (artifacts/week4/xgboost_v2a_mgmt_reduce_
  collinearity_0912/) HICBIR SEY YAZMAZ -- yalniz OKUR.
* `pipeline/xgboost_model.py` ve `tools/train_xgboost_week4.py`
  DEGISTIRILMEZ -- yalniz import edilir.
* (2) ve (3) global (fold-disi) bir tek cerceve gerektirir; uretimin
  ALIGNED yolunda boyle bir cerceve YOKTUR (standardizasyon ve Cox
  skoru fold-yereldir). Bu script global cerceveyi legacy yolun
  recetesiyle kurar ve bunu `notes` alaninda beyan eder.

Kullanim:
    python tools/demonstrate_leakage_effect.py            # tam kosu
    python tools/demonstrate_leakage_effect.py --gate-only
    python tools/demonstrate_leakage_effect.py --skip-leaky-cox-input
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).absolute().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(CODE_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "tools"))

from pipeline.cox_model import standardize_columns_fold_safe  # noqa: E402
from pipeline.reduce_collinearity import build_v3_candidate_pool  # noqa: E402
from pipeline.xgboost_model import (  # noqa: E402
    DEFAULT_XGB_PARAM_GRID,
    _bootstrap_auc_ci,
    apply_twelve_month_target,
    build_xgboost_feature_frame,
    define_twelve_month_survival_target,
    generate_in_sample_cox_scores,
    generate_in_sample_xgboost_auc,
    train_xgboost_nested_cv,
)

import train_cox_week3 as week3  # noqa: E402
import train_xgboost_week4 as week4  # noqa: E402


# --------------------------------------------------------------------
# Kilitli beklenen degerler -- CLAUDE.md + uretim kosusu metadata'si
# --------------------------------------------------------------------
EXPECTED_N_PATIENTS = 611
EXPECTED_N_EVENTS = 585
EXPECTED_N_LABELED = 606
EXPECTED_N_LABEL_YES = 341
EXPECTED_N_LABEL_NO = 265
EXPECTED_N_RADIOMIC_FEATURES = 93
EXPECTED_N_CLINICAL_EXTRA = 8
EXPECTED_AUC_OOF = 0.7226470425496597
EXPECTED_AUC_OOF_CI_LOWER = 0.6800539604349822
EXPECTED_AUC_OOF_CI_UPPER = 0.7605023576803478
GATE_TOLERANCE = 1e-9

DEFAULT_PRODUCTION_DIR = (
    CODE_ROOT / "artifacts" / "week4" / "xgboost_v2a_mgmt_reduce_collinearity_0912"
)
DEFAULT_OUTPUT_DIR = CODE_ROOT / "artifacts" / "week4" / "leakage_demonstration"
PRODUCTION_PREFIX = "week4_v2a_mgmt_aligned"


_T0 = time.time()


def log(message: str) -> None:
    """Ara ilerleme kaydi -- kosu kesilirse nereden devam edilecegi belli olsun."""

    elapsed = time.time() - _T0
    print(f"[{elapsed:8.1f}s] {message}", flush=True)


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).absolute().read_bytes()).hexdigest()


def _library_versions() -> dict[str, str]:
    import lifelines
    import sklearn
    import xgboost

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lifelines": lifelines.__version__,
    }


class GateFailure(RuntimeError):
    """ADIM 1 dogrulama kapisi DUSTU -- ADIM 2 KOSULMAZ."""


# --------------------------------------------------------------------
# Cerceve rekonstruksiyonu
# --------------------------------------------------------------------
def rebuild_training_frame(variant_name: str) -> dict[str, Any]:
    """Uretim kosusunun ALIGNED egitim cercevesini DB'den yeniden kurar.

    `tools/train_xgboost_week4.py::_rebuild_variant_training_frame_for_
    aligned_pipeline()` DOGRUDAN cagrilir -- cerceve "benzer sekilde"
    degil, URETIMLE AYNI KODLA kurulur."""

    from psycopg2.extras import RealDictCursor

    from db_connection import get_connection

    config = next(c for c in week3.V2_VARIANT_CONFIGS if c.name == variant_name)
    args = SimpleNamespace(
        expected_patient_count=EXPECTED_N_PATIENTS,
        expected_event_count=EXPECTED_N_EVENTS,
        allow_unexpected_patient_count=False,
    )

    connection = get_connection(readonly=True)
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        (
            training_frame,
            feature_columns,
            extra_columns,
            clinical_standardize_columns,
            fold_local_extra_column_builder,
        ) = week4._rebuild_variant_training_frame_for_aligned_pipeline(
            cursor=cursor, config=config, args=args
        )
    finally:
        connection.close()

    return {
        "training_frame": training_frame,
        "feature_columns": list(feature_columns),
        "extra_columns": list(extra_columns),
        "clinical_standardize_columns": list(clinical_standardize_columns),
        "fold_local_extra_column_builder": fold_local_extra_column_builder,
        "config": config,
    }


def load_production_artifacts(production_dir: Path) -> dict[str, Any]:
    """Uretim kosusunun kaydettigi artefaktlar -- SALT OKUNUR."""

    def _p(suffix: str) -> Path:
        return production_dir / f"{PRODUCTION_PREFIX}_{suffix}"

    metadata = json.loads(_p("run_metadata.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(_p("out_of_fold_predictions.csv")).set_index("patient_id")
    cox_audit = pd.read_csv(_p("cox_score_audit.csv")).set_index("patient_id")
    fold_results = pd.read_csv(_p("fold_results.csv"))
    collinearity_detail = json.loads(_p("collinearity_filter_detail.json").read_text(encoding="utf-8"))
    return {
        "metadata": metadata,
        "oof": oof,
        "cox_audit": cox_audit,
        "fold_results": fold_results,
        "collinearity_detail": collinearity_detail,
    }


# --------------------------------------------------------------------
# ADIM 1 -- DOGRULAMA KAPISI
# --------------------------------------------------------------------
def run_verification_gate(
    *,
    rebuilt: dict[str, Any],
    production: dict[str, Any],
    seed: int,
    n_bootstrap_ci: int,
) -> dict[str, Any]:
    """G1-G7. Hepsi gecerse ADIM 2 kosulabilir; biri duserse GateFailure."""

    from sklearn.model_selection import StratifiedKFold
    import xgboost as xgb

    gates: list[dict[str, Any]] = []

    def _gate(name: str, passed: bool, detail: Any) -> None:
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})
        log(f"  {'GECTI ' if passed else 'DUSTU '} {name}: {detail}")

    training_frame: pd.DataFrame = rebuilt["training_frame"]
    feature_columns: list[str] = rebuilt["feature_columns"]
    extra_columns: list[str] = rebuilt["extra_columns"]
    clinical_standardize_columns: list[str] = rebuilt["clinical_standardize_columns"]

    # ---- G1: egitim havuzu 611/585 ----
    n_patients = int(len(training_frame))
    n_events = int(training_frame["event"].sum())
    _gate(
        "G1_egitim_havuzu_611_585",
        n_patients == EXPECTED_N_PATIENTS and n_events == EXPECTED_N_EVENTS,
        {"n_patients": n_patients, "n_events": n_events},
    )

    # ---- G3: ozellik uzayi 93 + 8 ----
    _gate(
        "G3_ozellik_uzayi_93_radyomik_8_klinik",
        len(feature_columns) == EXPECTED_N_RADIOMIC_FEATURES
        and len(extra_columns) == EXPECTED_N_CLINICAL_EXTRA,
        {"n_radiomic": len(feature_columns), "n_clinical_extra": len(extra_columns)},
    )

    # ---- G2: 12-ay etiketi 606 (341/265) ----
    target, report = define_twelve_month_survival_target(
        training_frame,
        duration_col="survival_days",
        event_col="event",
        threshold_days=week4.DEFAULT_TWELVE_MONTH_THRESHOLD_DAYS,
    )
    labeled_frame = apply_twelve_month_target(training_frame, target, report=report)
    _gate(
        "G2_12ay_etiketli_606_341_265",
        len(labeled_frame) == EXPECTED_N_LABELED
        and report.n_survived_yes == EXPECTED_N_LABEL_YES
        and report.n_died_no == EXPECTED_N_LABEL_NO,
        {
            "n_labeled": int(len(labeled_frame)),
            "n_yes": int(report.n_survived_yes),
            "n_no": int(report.n_died_no),
            "n_excluded_ambiguous": int(report.n_excluded_ambiguous_censoring),
        },
    )

    # ---- G4: fold bolunmesi uretimle birebir ----
    outer_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(outer_splitter.split(labeled_frame, labeled_frame["target_12mo_survival"]))
    production_oof: pd.DataFrame = production["oof"]
    fold_mismatch: list[int] = []
    for fold_index, (_, test_idx) in enumerate(splits):
        reconstructed = set(labeled_frame.index[test_idx])
        produced = set(production_oof.index[production_oof["fold"] == fold_index])
        if reconstructed != produced:
            fold_mismatch.append(fold_index)
    _gate(
        "G4_dis_fold_bolunmesi_uretimle_ayni",
        not fold_mismatch,
        {"mismatched_folds": fold_mismatch, "n_folds": len(splits)},
    )

    # ---- G5 + G6: fold-yerel filtre + XGBoost tahminleri ----
    detail_by_fold = {
        entry["outer_fold"]: entry
        for entry in production["collinearity_detail"]
        if entry["scope"] == "outer_final"
    }
    cox_audit: pd.DataFrame = production["cox_audit"]
    fold_results: pd.DataFrame = production["fold_results"]

    collinearity_mismatch: list[dict[str, Any]] = []
    max_abs_probability_diff = 0.0
    reconstructed_rows: list[dict[str, Any]] = []
    honest_cox_score = pd.Series(index=labeled_frame.index, dtype=float)

    for fold_index, (train_idx, test_idx) in enumerate(splits):
        outer_train_raw = labeled_frame.iloc[train_idx]
        outer_test_raw = labeled_frame.iloc[test_idx]

        kept, report_c = build_v3_candidate_pool(
            outer_train_raw, feature_columns, cv_threshold=0.02, corr_threshold=0.95
        )
        produced_detail = detail_by_fold[fold_index]
        produced_dropped = set(produced_detail["near_constant_dropped"]) | {
            row["dropped_feature"] for row in produced_detail["correlation_dropped"]
        }
        reconstructed_dropped = set(feature_columns) - set(kept)
        if reconstructed_dropped != produced_dropped or len(kept) != produced_detail["n_kept_features"]:
            collinearity_mismatch.append(
                {
                    "fold": fold_index,
                    "n_kept_reconstructed": len(kept),
                    "n_kept_production": produced_detail["n_kept_features"],
                    "only_in_reconstructed": sorted(reconstructed_dropped - produced_dropped)[:5],
                    "only_in_production": sorted(produced_dropped - reconstructed_dropped)[:5],
                }
            )

        standardize_columns = clinical_standardize_columns + list(kept)
        outer_train_std, outer_test_std = standardize_columns_fold_safe(
            outer_train_raw, outer_test_raw, standardize_columns
        )

        fold_audit = cox_audit[cox_audit["fold"] == fold_index]
        train_scores = fold_audit[fold_audit["role"] == "outer_train_cross_fit"]["cox_oof_score"]
        test_scores = fold_audit[fold_audit["role"] == "outer_test_honest"]["cox_oof_score"]

        outer_train_with_score = outer_train_std.copy()
        outer_train_with_score["cox_score"] = train_scores.reindex(outer_train_with_score.index)
        outer_test_with_score = outer_test_std.copy()
        outer_test_with_score["cox_score"] = test_scores.reindex(outer_test_with_score.index)
        if outer_train_with_score["cox_score"].isna().any() or outer_test_with_score["cox_score"].isna().any():
            raise GateFailure(
                f"Fold {fold_index}: uretim Cox skor denetiminde eksik hasta var "
                "-- cerceve rekonstruksiyonu uretimle ortusmuyor."
            )

        honest_cox_score.loc[outer_test_with_score.index] = outer_test_with_score["cox_score"]

        xgb_feature_columns = feature_columns + extra_columns + ["cox_score"]
        best_params = ast.literal_eval(
            fold_results.loc[fold_results["fold"] == fold_index, "best_xgb_params"].iloc[0]
        )
        model = xgb.XGBClassifier(**best_params, eval_metric="logloss", random_state=seed)
        model.fit(outer_train_with_score[xgb_feature_columns], outer_train_with_score["target_12mo_survival"])
        predicted_probability = model.predict_proba(outer_test_with_score[xgb_feature_columns])[:, 1]

        produced_probability = production_oof.loc[outer_test_with_score.index, "predicted_probability"]
        diff = float(np.max(np.abs(predicted_probability - produced_probability.to_numpy())))
        max_abs_probability_diff = max(max_abs_probability_diff, diff)

        for patient_id, probability, true_label in zip(
            outer_test_with_score.index,
            predicted_probability,
            outer_test_with_score["target_12mo_survival"],
        ):
            reconstructed_rows.append(
                {
                    "patient_id": patient_id,
                    "fold": fold_index,
                    "predicted_probability": float(probability),
                    "true_label": float(true_label),
                }
            )
        log(f"    fold {fold_index}: kept={len(kept)} max|dP|={diff:.3e}")

    _gate(
        "G5_kolinearite_filtresi_uretimle_ayni",
        not collinearity_mismatch,
        {"mismatches": collinearity_mismatch},
    )
    _gate(
        "G6_fold_tahminleri_1e-9_icinde",
        max_abs_probability_diff <= GATE_TOLERANCE,
        {"max_abs_probability_diff": max_abs_probability_diff, "tolerance": GATE_TOLERANCE},
    )

    reconstructed_oof = pd.DataFrame(reconstructed_rows).set_index("patient_id")
    auc_reconstructed = _bootstrap_auc_ci(
        reconstructed_oof["true_label"],
        reconstructed_oof["predicted_probability"],
        n_bootstrap=n_bootstrap_ci,
        seed=seed,
        confidence_level=0.95,
    )
    auc_diff = abs(auc_reconstructed["auc"] - EXPECTED_AUC_OOF)
    ci_lower_diff = abs(auc_reconstructed["ci_lower"] - EXPECTED_AUC_OOF_CI_LOWER)
    ci_upper_diff = abs(auc_reconstructed["ci_upper"] - EXPECTED_AUC_OOF_CI_UPPER)
    _gate(
        "G7_out_of_fold_auc_ci_uretimle_1e-9_icinde",
        max(auc_diff, ci_lower_diff, ci_upper_diff) <= GATE_TOLERANCE,
        {
            "auc_reconstructed": auc_reconstructed["auc"],
            "auc_production": EXPECTED_AUC_OOF,
            "abs_diff_auc": auc_diff,
            "abs_diff_ci_lower": ci_lower_diff,
            "abs_diff_ci_upper": ci_upper_diff,
        },
    )

    all_passed = all(g["passed"] for g in gates)
    return {
        "all_passed": all_passed,
        "gates": gates,
        "labeled_frame": labeled_frame,
        "twelve_month_report": {
            "n_total": int(report.n_total),
            "n_survived_yes": int(report.n_survived_yes),
            "n_died_no": int(report.n_died_no),
            "n_excluded_ambiguous_censoring": int(report.n_excluded_ambiguous_censoring),
            "threshold_days": float(report.threshold_days),
        },
        "honest_cox_score": honest_cox_score,
        "auc_reconstructed": auc_reconstructed,
        "max_abs_probability_diff": max_abs_probability_diff,
    }


# --------------------------------------------------------------------
# ADIM 2
# --------------------------------------------------------------------
def build_global_frames(
    *,
    labeled_frame: pd.DataFrame,
    feature_columns: list[str],
    extra_columns: list[str],
    clinical_standardize_columns: list[str],
    honest_cox_score: pd.Series,
) -> dict[str, Any]:
    """ADIM 2 icin GLOBAL (fold-disi) cerceveler.

    !! Uretimin ALIGNED yolunda global bir cerceve YOKTUR (standardizasyon
    ve Cox skoru fold-yereldir). Burada legacy yolun recetesi kullanilir:
    tum havuzdan TEK bir kolinearite havuzu + TEK bir standardizasyon.
    Bu, karsilastirmanin HER IKI ucunda AYNI sekilde uygulandigi icin
    farki carpitmaz; yine de beyan edilir."""

    kept_global, _ = build_v3_candidate_pool(
        labeled_frame, feature_columns, cv_threshold=0.02, corr_threshold=0.95
    )
    standardize_columns = clinical_standardize_columns + list(kept_global)
    means = labeled_frame[standardize_columns].mean()
    stds = labeled_frame[standardize_columns].std(ddof=0).replace(0.0, 1.0)
    standardized = labeled_frame.copy()
    standardized[standardize_columns] = (
        labeled_frame[standardize_columns] - means
    ) / stds

    honest_frame = build_xgboost_feature_frame(
        honest_cox_score.reindex(standardized.index),
        standardized[feature_columns + extra_columns],
        score_column="cox_score",
    )
    honest_frame["target_12mo_survival"] = standardized["target_12mo_survival"]

    return {
        "standardized": standardized,
        "kept_global": list(kept_global),
        "honest_frame": honest_frame,
        "xgb_feature_columns": feature_columns + extra_columns + ["cox_score"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="v2a_mgmt")
    parser.add_argument("--production-dir", type=Path, default=DEFAULT_PRODUCTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap-ci", type=int, default=1000)
    parser.add_argument("--gate-only", action="store_true", help="ADIM 1'i kos, ADIM 2'yi ATLA")
    parser.add_argument(
        "--skip-leaky-cox-input",
        action="store_true",
        help="(3)/(3k) kolunu ATLA -- iki tam XGBoost nested-CV gerektirir (pahali)",
    )
    parser.add_argument("--frame-cache", type=Path, default=None, help="Cerceveyi pickle'a onbellekle (tekrar kosu icin)")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"BASLIYOR -- variant={args.variant} seed={args.seed}")
    log(f"uretim artefaktlari (SALT OKUNUR): {args.production_dir}")

    if args.frame_cache is not None and args.frame_cache.is_file():
        log(f"ADIM 0: cerceve onbellekten okunuyor -> {args.frame_cache}")
        rebuilt = pd.read_pickle(args.frame_cache)
    else:
        log("ADIM 0: egitim cercevesi DB'den yeniden kuruluyor (SELECT-only)...")
        rebuilt = rebuild_training_frame(args.variant)
        if args.frame_cache is not None:
            cacheable = {k: v for k, v in rebuilt.items() if k not in ("fold_local_extra_column_builder", "config")}
            pd.to_pickle(cacheable, args.frame_cache)
    log(
        f"ADIM 0 BITTI: n={len(rebuilt['training_frame'])} "
        f"radyomik={len(rebuilt['feature_columns'])} klinik={len(rebuilt['extra_columns'])}"
    )

    if rebuilt.get("fold_local_extra_column_builder") is not None:
        raise GateFailure(
            "Bu varyant fold-yerel ek kolon (RCS yas) kullaniyor -- bu script "
            "yalniz use_age_spline=False varyantlar (v2a_mgmt) icin dogrulanmistir."
        )

    production = load_production_artifacts(args.production_dir)
    log("ADIM 1: dogrulama kapisi kosuluyor (G1-G7)...")
    gate = run_verification_gate(
        rebuilt=rebuilt, production=production, seed=args.seed, n_bootstrap_ci=args.n_bootstrap_ci
    )

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).absolute()),
        "script_sha256": _script_sha256(),
        "library_versions": _library_versions(),
        "variant": args.variant,
        "production_run_dir": str(args.production_dir),
        "purpose": (
            "raw/mimari/v45.txt:1760-1761 + raw/plan/plan.txt:508 sarti -- "
            "out-of-fold (DOGRU) ve in-sample (YANLIS) XGBoost AUC'leri "
            "arasindaki farkin GERCEK veriyle sayisal gosterimi."
        ),
        "verification_gate": {
            "all_passed": gate["all_passed"],
            "gates": gate["gates"],
            "reconstructed_auc_out_of_fold": gate["auc_reconstructed"],
            "production_auc_out_of_fold": {
                "auc": EXPECTED_AUC_OOF,
                "ci_lower": EXPECTED_AUC_OOF_CI_LOWER,
                "ci_upper": EXPECTED_AUC_OOF_CI_UPPER,
            },
            "max_abs_fold_probability_diff": gate["max_abs_probability_diff"],
            "tolerance": GATE_TOLERANCE,
        },
        "cohort": {
            "n_training_patients": EXPECTED_N_PATIENTS,
            "n_training_events": EXPECTED_N_EVENTS,
            "twelve_month_label_report": gate["twelve_month_report"],
        },
    }

    summary_path = output_dir / "leakage_effect_summary.json"

    if not gate["all_passed"]:
        summary["status"] = "GATE_FAILED_ADIM2_KOSULMADI"
        summary["results"] = None
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log("!! DOGRULAMA KAPISI DUSTU -- ADIM 2 KOSULMADI. Ozet yazildi.")
        return 2

    log("ADIM 1 GECTI. ADIM 2 basliyor.")
    if args.gate_only:
        summary["status"] = "GATE_ONLY"
        summary["results"] = None
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log("--gate-only verildi, ADIM 2 atlandi.")
        return 0

    labeled_frame: pd.DataFrame = gate["labeled_frame"]
    globals_ = build_global_frames(
        labeled_frame=labeled_frame,
        feature_columns=rebuilt["feature_columns"],
        extra_columns=rebuilt["extra_columns"],
        clinical_standardize_columns=rebuilt["clinical_standardize_columns"],
        honest_cox_score=gate["honest_cox_score"],
    )
    log(f"ADIM 2: global cerceve kuruldu (kolinearite havuzu {len(globals_['kept_global'])}/93).")

    fold_results: pd.DataFrame = production["fold_results"]
    last_fold_params = ast.literal_eval(fold_results["best_xgb_params"].iloc[-1])
    log(f"ADIM 2 (2): auc_fully_in_sample -- tek fit, params={last_fold_params}")
    auc_fully_in_sample = generate_in_sample_xgboost_auc(
        globals_["honest_frame"],
        globals_["xgb_feature_columns"],
        target_col="target_12mo_survival",
        params=last_fold_params,
        seed=args.seed,
        n_bootstrap_ci=args.n_bootstrap_ci,
    )
    log(f"ADIM 2 (2) BITTI: auc={auc_fully_in_sample['auc']:.6f}")

    results: dict[str, Any] = {
        "auc_out_of_fold_correct": {
            **gate["auc_reconstructed"],
            "role": "DOGRU HAT -- raporlanabilir sonuc",
            "how": (
                "Uretim ALIGNED nested-CV'si: Cox VE XGBoost AYNI dis-fold "
                "bolunmesini paylasir; her hastanin Cox skoru onu HIC gormemis "
                "bir Cox fit'inden gelir. Bu script cerceveyi DB'den yeniden "
                "kurup fold tahminlerini yeniden uretti (G6/G7)."
            ),
        },
        "auc_fully_in_sample": {
            **auc_fully_in_sample,
            "role": "!! YANLIS UC -- ASLA sonuc olarak sunulmaz",
            "how": (
                "XGBoost global cercevenin TAMAMINI hem egitir hem skorlar "
                "(tek fit, capraz dogrulama YOK). Cox skoru girdisi DURUST "
                "(out-of-fold) oldugu halde AUC bu kadar iyimserlesir -- yani "
                "bu fark YALNIZ XGBoost katmaninin in-sample olmasindan gelir."
            ),
            "xgb_params": last_fold_params,
        },
    }

    if args.skip_leaky_cox_input:
        results["auc_out_of_fold_leaky_cox_input"] = {
            "status": "ATLANDI",
            "reason": "--skip-leaky-cox-input verildi (iki tam XGBoost nested-CV pahali).",
        }
    else:
        log("ADIM 2 (3k): eslestirilmis kontrol -- legacy XGBoost nested-CV, DURUST Cox girdisi...")
        matched_control = train_xgboost_nested_cv(
            globals_["honest_frame"],
            globals_["xgb_feature_columns"],
            target_col="target_12mo_survival",
            outer_splits=5,
            inner_splits=5,
            seed=args.seed,
            param_grid=DEFAULT_XGB_PARAM_GRID,
            n_bootstrap_ci=args.n_bootstrap_ci,
        )
        log(f"ADIM 2 (3k) BITTI: auc={matched_control.auc_out_of_fold['auc']:.6f}")

        log("ADIM 2 (3): SIZINTILI Cox girdisi uretiliyor (tum havuzda fit + tum havuzu skorla)...")
        best_penalizer = float(fold_results["best_cox_penalizer"].iloc[-1])
        best_l1_ratio = float(fold_results["best_cox_l1_ratio"].iloc[-1])
        leaky_cox_score, leaky_cox_features = generate_in_sample_cox_scores(
            globals_["standardized"],
            globals_["kept_global"],
            duration_col="survival_days",
            event_col="event",
            extra_columns=rebuilt["extra_columns"],
            penalizer=best_penalizer,
            l1_ratio=best_l1_ratio,
            extra_column_penalizer=0.0,
            stability_selection=True,
            n_bootstrap_stability=200,
            stability_frequency_threshold=0.6,
            seed=args.seed,
        )
        log(f"ADIM 2 (3): sizintili Cox skoru hazir ({len(leaky_cox_features)} ozellik). XGBoost nested-CV...")
        leaky_frame = build_xgboost_feature_frame(
            leaky_cox_score,
            globals_["standardized"][rebuilt["feature_columns"] + rebuilt["extra_columns"]],
            score_column="cox_score",
        )
        leaky_frame["target_12mo_survival"] = globals_["standardized"]["target_12mo_survival"]
        leaky_result = train_xgboost_nested_cv(
            leaky_frame,
            globals_["xgb_feature_columns"],
            target_col="target_12mo_survival",
            outer_splits=5,
            inner_splits=5,
            seed=args.seed,
            param_grid=DEFAULT_XGB_PARAM_GRID,
            n_bootstrap_ci=args.n_bootstrap_ci,
        )
        log(f"ADIM 2 (3) BITTI: auc={leaky_result.auc_out_of_fold['auc']:.6f}")

        results["auc_matched_control_honest_cox_input"] = {
            **matched_control.auc_out_of_fold,
            "role": "(3)'un ESLESTIRILMIS KONTROLU -- durust Cox girdisi, AYNI legacy XGBoost nested-CV",
            "how": (
                "Cox skoru girdisi uretimin out-of-fold skorlaridir (durust); "
                "XGBoost legacy nested-CV. (3) ile YALNIZ Cox skorunun sizintili "
                "olup olmamasi bakimindan farklidir -- (3) ile karsilastirilmasi "
                "gereken taban budur, auc_out_of_fold_correct DEGIL."
            ),
        }
        results["auc_out_of_fold_leaky_cox_input"] = {
            **leaky_result.auc_out_of_fold,
            "role": "!! YANLIS UC -- ASLA sonuc olarak sunulmaz",
            "how": (
                "Cox skoru TUM havuzda fit edilip TUM havuz skorlanarak uretildi "
                "(her hastanin skoru onu ZATEN GORMUS bir modelden geliyor); "
                "XGBoost'un KENDI nested-CV'si hala durust. Bu, v45'in 'juri "
                "sorusunu' (Cox -> XGBoost sizintisi) IZOLE eder."
            ),
            "cox_penalizer": best_penalizer,
            "cox_l1_ratio": best_l1_ratio,
            "n_cox_selected_features": len(leaky_cox_features),
        }
        results["delta_leaky_cox_input_minus_matched_control"] = (
            leaky_result.auc_out_of_fold["auc"] - matched_control.auc_out_of_fold["auc"]
        )

    results["delta_fully_in_sample_minus_out_of_fold_correct"] = (
        auc_fully_in_sample["auc"] - gate["auc_reconstructed"]["auc"]
    )

    summary["status"] = "OK"
    summary["results"] = results
    summary["global_frame_note"] = (
        "auc_fully_in_sample ve auc_out_of_fold_leaky_cox_input GLOBAL (fold-disi) "
        "tek bir cerceve gerektirir; uretimin ALIGNED yolunda boyle bir cerceve "
        "YOKTUR (standardizasyon ve kolinearite havuzu fold-yereldir). Global "
        "cerceve legacy yolun recetesiyle kuruldu: tum 606 hastadan TEK "
        f"kolinearite havuzu ({len(globals_['kept_global'])}/93) + TEK "
        "standardizasyon. Bu recete karsilastirmanin HER IKI ucunda AYNI "
        "uygulandi."
    )
    summary["language_rules"] = [
        "auc_fully_in_sample ve auc_out_of_fold_leaky_cox_input ASLA 'sonuc' degildir -- karsilastirmanin YANLIS ucudur.",
        "'Sizinti yoktur' gibi kesin cumle KURULMAZ; dogrusu: 'sizintili ve sizintisiz hesaplamalar arasindaki fark olculmus ve raporlanmistir'.",
        "Tek-esik dili YASAK -- her AUC nokta tahmini + %95 bootstrap CI ile verilir.",
    ]
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"BITTI -- ozet yazildi: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
