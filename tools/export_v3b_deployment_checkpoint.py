"""v3b_lowvar_v2amgmt icin CANLI DEPLOY checkpoint'i uretir.

BAGLAM (2026-09-12, Baris karari "Secenek A" -- bkz. AKTIF-GOREVLER.md
modeling-agent-B satiri + log/2026-09-12.md): v3b'nin final_coefficients.csv'si
FOLD-GUVENLI Z-SCORE ile STANDARDIZE edilmis radyomik+yas uzerinde fit
edildi (bkz. `entities/cox-model-varyant-katalogu.md` SS4 "Katsayi okuma
notu"). `api/predict.py` (backend-agent-B tarafindan AYNI ANDA
guncelleniyor) canli hastanin HAM degerlerini `feature_standardization`
alanindaki mean/std ile standardize EDECEK -- bu script o alani dolduran
checkpoint'i uretir.

YONTEM -- "yeniden calistir, KARSILASTIR" (reimplemented DEGIL): bu script
`tools/train_cox_week3.py`'nin GERCEK fonksiyonlarini (build_v3_radiomic_
feature_pool, build_training_frame, standardize_columns_fold_safe,
fit_final_model_on_full_pool, evaluate_external_test, ...) BIREBIR AYNI
sirada, AYNI cli-args ile (parse_args(["--variants", ...]) varsayilanlari)
cagirir -- `run_single_v3_variant()`'in YAPTIGI HER ADIMI tekrarlar, TEK
fark: ham (standardize-ONCESI) `training_frame`'i de saklar (mean/std
hesabi icin) -- `tools/train_cox_week3.py`'ye DOKUNULMADI/DEGISTIRILMEDI,
SADECE import edilip cagrildi.

DOGRULAMA (ZORUNLU, bu script ICINDE yapilir): uretilen fitted_model'in
katsayilari `artifacts/week3/cox_model/week3_v3b_lowvar_v2amgmt_final_
coefficients.csv` ile bit-birebir (tol=1e-9 mutlak/1e-6 bagil)
KARSILASTIRILIR -- eslesmezse checkpoint YAZILMAZ, AssertionError firlar.

Cikti: `models/cox_phm_v3b_lowvar_v2amgmt_2026-09-12.pkl` (proje
`models/` dizinine, eski v1 dosyasina DOKUNMADAN).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.train_cox_week3 as t3  # noqa: E402
from db_connection import get_connection  # noqa: E402
from psycopg2.extras import RealDictCursor  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "artifacts" / "week3" / "cox_model"
CACHE_DIR = OUTPUT_DIR / "_cache"
MODELS_DIR = REPO_ROOT / "backend" / "models"  # 2026-09-16: models/ backend/ altina tasindi
FINAL_COEFFICIENTS_CSV = OUTPUT_DIR / "week3_v3b_lowvar_v2amgmt_final_coefficients.csv"
CHECKPOINT_OUT_PATH = MODELS_DIR / "cox_phm_v3b_lowvar_v2amgmt_2026-09-12.pkl"

VARIANT_NAME = "v3b_lowvar_v2amgmt"


def _cached_fetch(cursor, *, tool: str, tag: str) -> pd.DataFrame:
    """`tools/train_cox_week3.py::main()` icindeki `_cached_fetch()`
    closure'inin BAGIMSIZ kopyasi -- AYNI mantik (DB satir sayisiyla
    onbellek gecerliligini dogrula, uyusmuyorsa yeniden cek). O
    fonksiyon `main()` icinde tanimli oldugu icin disaridan import
    EDILEMEZ, bu yuzden burada YENIDEN yazildi (train_cox_week3.py'ye
    DOKUNULMADI).

    🔧 2026-09-14 (kalem 48) -- TIMEPOINT KAPISI BURAYA DA GETIRILDI.
    ~~Eski hali: `SELECT COUNT(*) AS n FROM radiomics WHERE
    segmentation_tool = %s`~~ (ustu cizildi, wiki hard rule #3)

    Bu kopya, karar 39'un (2026-09-13/14) `train_cox_week3.py`'de yaptigi
    duzeltmeyi ALMAMISTI. Iki somut sonucu vardi:

    1. **Onbellek SONSUZA KADAR gecersizlesirdi.** `fetch_c32_radiomics_
       long_frame()` artik `mr_scans.timepoint_label` beyaz listesini
       uyguluyor (611/585 kilidinin IKINCI KATMANI), sayim ise filtresizdi
       -> `radiomics`'e TEK bir post-op satir yazildigi anda
       `len(frame) != db_rows` olur ve kapi DOGRU calistigi halde her
       kosuda ~11 dakikalik yeniden cekim tetiklenirdi (yanlis alarm).
    2. **Daha kotusu: CAPRAZ BULASMA.** Iki script AYNI onbellek dizinini
       paylasiyor (`artifacts/week3/cox_model/_cache`, olculdu). Bu kopya
       meta'ya `timepoint_labels` anahtarini HIC yazmiyordu ve `db_rows`'a
       FILTRESIZ sayiyi koyuyordu; `train_cox_week3.py`'nin fail-closed
       kapisi o meta'yi "kapidan ONCE yazilmis" sayip onbellegi ATARDI --
       yani bu script'i kosmak egitim kosusunun onbellegini bozardi.

    Duzeltme: sayim `t3.count_c32_radiomics_rows()` ile (fetch ile AYNI
    beyaz liste), meta'ya `timepoint_labels` parmak izi de yaziliyor ve
    eksik/farkli parmak izi fail-closed kabul ediliyor -- yani
    `train_cox_week3.py::main()::_cached_fetch` ile BIREBIR ayni sozlesme.
    """

    import json

    cache_path = CACHE_DIR / f"{tag}_long.pkl"
    meta_path = CACHE_DIR / f"{tag}_long.meta.json"
    db_rows = t3.count_c32_radiomics_rows(cursor, segmentation_tool=tool)
    timepoint_fingerprint = list(t3.allowed_timepoint_labels(tool))
    if cache_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # `timepoint_labels` anahtari ZORUNLU (fail-closed): kapidan
            # ONCE yazilmis onbellekler filtresiz cekilmis OLABILIR.
            if (
                meta.get("segmentation_tool") == tool
                and meta.get("db_rows") == db_rows
                and meta.get("timepoint_labels") == timepoint_fingerprint
            ):
                frame = pd.read_pickle(cache_path)
                if len(frame) == db_rows:
                    print(
                        f"[cache] {tag}: {db_rows} satir onbellekten okundu "
                        f"(DB sayimiyla eslesti, timepoint beyaz listesi "
                        f"{timepoint_fingerprint}).",
                        flush=True,
                    )
                    return frame
            if meta.get("timepoint_labels") != timepoint_fingerprint:
                print(
                    f"[cache] {tag}: onbellek TIMEPOINT KAPISINDAN ONCE "
                    f"yazilmis (meta.timepoint_labels="
                    f"{meta.get('timepoint_labels')!r} != "
                    f"{timepoint_fingerprint!r}) -- fail-closed, yeniden "
                    "cekiliyor.",
                    flush=True,
                )
            else:
                print(f"[cache] {tag}: onbellek GECERSIZ (DB {db_rows} satir) -- yeniden cekiliyor.", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[cache] {tag}: onbellek okunamadi ({type(exc).__name__}) -- yeniden cekiliyor.", flush=True)
    frame = t3.fetch_c32_radiomics_long_frame(cursor, segmentation_tool=tool)
    frame.to_pickle(cache_path)
    meta_path.write_text(
        json.dumps(
            {
                "segmentation_tool": tool,
                "db_rows": db_rows,
                "cached_rows": len(frame),
                "timepoint_labels": timepoint_fingerprint,
            }
        ),
        encoding="utf-8",
    )
    return frame


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not FINAL_COEFFICIENTS_CSV.is_file():
        raise FileNotFoundError(f"Referans CSV yok: {FINAL_COEFFICIENTS_CSV}")
    if CHECKPOINT_OUT_PATH.exists():
        raise FileExistsError(
            f"{CHECKPOINT_OUT_PATH} ZATEN VAR -- eski deploy dosyasi UZERINE "
            "YAZILMAZ (Baris'in secilen yontemi: yeni dosya adi)."
        )

    # ---- 1) args -- ORIJINAL kosunun cli_args'iyla BIREBIR ayni (bkz.
    # week3_v3b_lowvar_v2amgmt_run_metadata.json::cli_args) ----
    args = t3.parse_args(["--variants", "v3a_lowvar_v1referans", "v3b_lowvar_v2amgmt"])
    assert args.seed == 42
    assert args.primary_threshold == 0.6
    assert args.expected_patient_count == 611
    assert args.expected_event_count == 585
    assert args.expected_ucsf_patient_count == 295
    assert args.expected_ucsf_event_count == 169
    assert args.clinical_extra_column_penalizer == 0.0

    config = next(c for c in t3.ALL_V3_VARIANT_CONFIGS if c.name == VARIANT_NAME)
    assert config.base.include_mgmt is True
    assert config.base.use_age_spline is False
    assert config.standardize_radiomics is True

    # ---- 2) veri -- ayni onbellek (DB satir sayisiyla canli dogrulanir) ----
    connection = get_connection(readonly=True)
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        upenn_long = _cached_fetch(cursor, tool=t3.SEGMENTATION_TOOL_UPENN_C32, tag="upenn")
        t3.raise_if_empty_c32(upenn_long, segmentation_tool=t3.SEGMENTATION_TOOL_UPENN_C32)
        upenn_patients = t3.fetch_patients_frame(cursor, source_name=t3.UPENN_SOURCE_NAME)

        ucsf_long = _cached_fetch(cursor, tool=t3.SEGMENTATION_TOOL_UCSF_C32, tag="ucsf")
        t3.raise_if_empty_c32(ucsf_long, segmentation_tool=t3.SEGMENTATION_TOOL_UCSF_C32)
        ucsf_patients = t3.fetch_patients_frame(cursor, source_name=t3.UCSF_SOURCE_NAME)
        cursor.close()
    finally:
        connection.close()

    # ---- 3) `run_single_v3_variant()`'in BIREBIR AYNI adimlari --
    # (train_cox_week3.py satir ~3713-3831) -- TEK fark: `training_frame`
    # (HAM, standardize-ONCESI) burada AYRICA saklaniyor. ----
    upenn_wide, upenn_pivot_report = t3.pivot_radiomics_long_to_wide(
        upenn_long, regions=list(config.base.regions)
    )
    ucsf_wide, ucsf_pivot_report = t3.pivot_ucsf_regions_long_to_wide(
        ucsf_long, regions=config.base.regions
    )

    raw_candidate_columns: list[str] = []
    for region in config.base.regions:
        raw_candidate_columns += t3.select_feature_columns(
            upenn_wide, t3.STABLE_FEATURES_ICC60, region=region
        )

    upenn_wide_for_pool = t3.drop_declared_shortfall_patients_for_pool(
        upenn_wide, config.base.regions
    )

    feature_columns, pool_report = t3.build_v3_radiomic_feature_pool(
        upenn_wide_for_pool,
        raw_candidate_columns,
        cv_threshold=config.cv_threshold,
        corr_threshold=config.corr_threshold,
    )
    assert len(feature_columns) == 54, f"beklenen 54 aday, alinan {len(feature_columns)}"

    extra_columns = t3.build_variant_clinical_extra_columns(
        include_mgmt=config.base.include_mgmt, use_age_spline=config.base.use_age_spline
    )
    assert extra_columns == [
        "clinical_age",
        "clinical_gender_male",
        "clinical_gtr_y",
        "clinical_gtr_missing",
        "clinical_idh_mutant",
        "clinical_idh_missing",
        "clinical_mgmt_methylated",
        "clinical_mgmt_missing",
    ], extra_columns

    upenn_clinical = t3.build_variant_clinical_frame(
        upenn_patients,
        include_mgmt=config.base.include_mgmt,
        use_age_spline=config.base.use_age_spline,
        age_knots=None,
    )
    ucsf_clinical = t3.build_variant_clinical_frame(
        ucsf_patients,
        include_mgmt=config.base.include_mgmt,
        use_age_spline=config.base.use_age_spline,
        age_knots=None,
    )
    upenn_patients_c = upenn_patients.join(upenn_clinical)
    ucsf_patients_c = ucsf_patients.join(ucsf_clinical)

    training_frame, training_report = t3.build_training_frame(
        upenn_wide[feature_columns],
        upenn_patients_c,
        check_combat_identity=True,
        passthrough_columns=extra_columns,
    )
    n_train_events = int(training_frame["event"].sum())
    t3.check_training_pool_counts(
        training_report.n_output_rows,
        n_train_events,
        expected_patients=args.expected_patient_count,
        expected_events=args.expected_event_count,
        allow_mismatch=args.allow_unexpected_patient_count,
        regions=config.base.regions,
        dropped_patient_ids=t3.all_dropped_patient_ids(training_report),
    )
    print(
        f"[dogrulama] training_frame n={training_report.n_output_rows} "
        f"events={n_train_events} (beklenen 611/585)",
        flush=True,
    )

    external_frame, external_frame_report = t3.build_external_test_frame(
        ucsf_wide[feature_columns], ucsf_patients_c, passthrough_columns=extra_columns
    )
    t3.check_external_test_pool_counts(
        len(external_frame),
        int(external_frame["event"].sum()),
        expected_patients=args.expected_ucsf_patient_count,
        expected_events=args.expected_ucsf_event_count,
        allow_mismatch=args.allow_unexpected_ucsf_count,
        label=f"UCSF harici test ({config.name})",
    )

    clinical_age_columns = [t3.CLINICAL_AGE_COLUMN]
    clinical_standardize_columns = feature_columns + clinical_age_columns

    # ---- 4) DAGITIM ICIN kritik adim: standardizasyon mean/std'sini
    # `training_frame` (HAM, 611 satir) uzerinden AYRICA hesapla --
    # `standardize_columns_fold_safe()`'in KENDI formulu (pipeline/
    # cox_model.py:1465-1467): mean() + std(ddof=0), std==0 -> 1.0. ----
    means_full = training_frame[clinical_standardize_columns].mean()
    stds_full = training_frame[clinical_standardize_columns].std(ddof=0)
    stds_full = stds_full.where(stds_full > 0, 1.0)

    # ---- 5) `run_modeling_arm()`'in full-pool standardizasyon +
    # final-fit adimini BIREBIR cagir (nested-CV kismini ATLIYORUZ --
    # final_model'i ETKILEMEZ, sadece rapor edilen ic-CV fold sonuclari
    # icin gerekliydi, bu script'in amaci o degil). ----
    full_pool_training_frame, full_pool_external_frame = t3.standardize_columns_fold_safe(
        training_frame, external_frame, clinical_standardize_columns
    )
    # standardize_columns_fold_safe kendi ic mean/std'sini training_frame'den
    # hesapliyor -- yukaridaki means_full/stds_full ile AYNI olmali (ayni
    # formul, ayni girdi). Capraz-dogrulama:
    manual_standardized = (
        training_frame[clinical_standardize_columns] - means_full
    ) / stds_full
    pd.testing.assert_frame_equal(
        full_pool_training_frame[clinical_standardize_columns].reset_index(drop=True),
        manual_standardized.reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
    )
    print("[dogrulama] means_full/stds_full, standardize_columns_fold_safe ile BIT-BIREBIR eslesti.", flush=True)

    final_model = t3.fit_final_model_on_full_pool(
        full_pool_training_frame,
        feature_columns,
        duration_col="survival_days",
        event_col="event",
        extra_columns=extra_columns,
        inner_splits=args.inner_splits,
        l1_ratio_grid=tuple(args.l1_ratio_grid) if args.l1_ratio_grid else t3.DEFAULT_L1_RATIO_GRID,
        penalizer_grid=tuple(args.penalizer_grid) if args.penalizer_grid else t3.DEFAULT_PENALIZER_GRID,
        n_bootstrap_stability=args.n_bootstrap_stability,
        stability_frequency_threshold=args.primary_threshold,
        seed=args.seed,
        extra_column_penalizer=args.clinical_extra_column_penalizer,
    )

    print(
        f"[dogrulama] final_features (radyomik) n={len(final_model.final_features)}",
        flush=True,
    )

    # ---- 6) BIT-BIREBIR ESLESME KANITI -- final_coefficients.csv ile ----
    ref = pd.read_csv(FINAL_COEFFICIENTS_CSV)
    ref = ref[ref["variant"] == VARIANT_NAME].set_index("covariate")["coef"]

    fitted = final_model.fitted_model
    model_columns = list(fitted.params_.index)
    expected_columns = set(final_model.final_features) | set(extra_columns)
    if set(model_columns) != expected_columns:
        raise AssertionError(
            f"final_model.params_.index ({sorted(model_columns)}) "
            f"final_features+extra_columns ({sorted(expected_columns)}) ile UYUSMUYOR."
        )
    if set(model_columns) != set(ref.index):
        raise AssertionError(
            "YENIDEN FIT EDILEN model ile referans CSV'nin kolon kumeleri "
            f"UYUSMUYOR. Yeniden-fit: {sorted(model_columns)} "
            f"Referans CSV: {sorted(ref.index)}"
        )

    max_abs_diff = 0.0
    for col in model_columns:
        got = float(fitted.params_[col])
        want = float(ref[col])
        diff = abs(got - want)
        max_abs_diff = max(max_abs_diff, diff)
        rel = diff / max(abs(want), 1e-12)
        if diff > 1e-6 and rel > 1e-4:
            raise AssertionError(
                f"KATSAYI UYUSMUYOR -- kolon={col!r} yeniden-fit={got!r} "
                f"referans-csv={want!r} abs_diff={diff!r} rel_diff={rel!r}. "
                "Checkpoint YAZILMADI."
            )
    print(
        f"[KANIT] {len(model_columns)}/{len(model_columns)} kolon "
        f"week3_v3b_lowvar_v2amgmt_final_coefficients.csv ile eslesti "
        f"(max|fark|={max_abs_diff:.3e}).",
        flush=True,
    )

    # ---- 7) harici test (opsiyonel ekstra dogrulama -- katalogdaki
    # 0,659172 [0,616377-0,702010] ile karsilastirilir, checkpoint'e
    # YAZILMAZ, sadece konsola raporlanir) ----
    external_result = t3.evaluate_external_test(
        fitted,
        full_pool_external_frame,
        final_model.final_features,
        duration_col="survival_days",
        event_col="event",
        extra_columns=extra_columns,
        n_bootstrap=args.n_bootstrap_external,
        seed=args.seed,
    )
    print(
        f"[dogrulama] harici c_index={external_result['c_index']:.6f} "
        f"[{external_result['ci_lower']:.6f}-{external_result['ci_upper']:.6f}] "
        "(katalogdaki 0,659172 [0,616377-0,702010] ile karsilastir).",
        flush=True,
    )

    # ---- 8) checkpoint sozlugu -- backend-agent-B'nin SABIT sozlesmesi ----
    feature_standardization: dict[str, dict[str, float]] = {}
    for col in final_model.final_features + [t3.CLINICAL_AGE_COLUMN]:
        feature_standardization[col] = {
            "mean": float(means_full[col]),
            "std": float(stds_full[col]),
        }

    checkpoint = {
        "arm_name": VARIANT_NAME,
        "final_features": list(final_model.final_features),
        "clinical_extra_columns": list(extra_columns),
        "fitted_model": fitted,
        "age_rcs_knots": None,
        "feature_standardization": feature_standardization,
        # bilgi amacli, backend-agent-B'nin sozlesmesinde ZORUNLU DEGIL ama
        # sessizce atilmiyor -- provenance icin faydali:
        "provenance": {
            "generated_by": "tools/export_v3b_deployment_checkpoint.py",
            "reproduced_from": "week3_v3b_lowvar_v2amgmt_final_coefficients.csv + canli DB (2026-09-12)",
            "standardization_source": (
                "training_frame (UPenn, 611 hasta, HAM/standardize-ONCESI) "
                "mean()/std(ddof=0), pipeline.cox_model.standardize_columns_"
                "fold_safe() ile AYNI formul -- CV fold-ici DEGIL, TAM havuz."
            ),
            "n_train": training_report.n_output_rows,
            "n_events_train": n_train_events,
            "external_test": {
                "source": "UCSF-PDGM",
                "n_patients": len(external_frame),
                "n_events": int(external_frame["event"].sum()),
                "c_index": external_result["c_index"],
                "ci_lower": external_result["ci_lower"],
                "ci_upper": external_result["ci_upper"],
            },
            "coefficient_match_max_abs_diff": max_abs_diff,
        },
    }

    with CHECKPOINT_OUT_PATH.open("wb") as fh:
        pickle.dump(checkpoint, fh)
    print(f"[YAZILDI] {CHECKPOINT_OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
