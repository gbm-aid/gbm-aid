"""`probe_perturbation_stability.py`'nin uzun-format CSV ciktisini analiz
eder -- kaynak/extraction-varyanti/ozellik-sinifi bazinda within-image ICC
(ve CV%% yedek metrigi) hesaplar, C32 vs C64'u ROI-voksel-buyuklugune gore
kirar, bias-field'in A/B/C32/C64 uzerindeki ozel etkisini raporlar.

Bu script SALT-OKUNUR (yalniz CSV okur), DB'ye/uretim dosyalarina hicbir
seyle DOKUNMAZ, pandas/numpy disinda agir bagimlilik gerektirmez (ICC
manuel implementasyonla hesaplanir -- pingouin/statsmodels bagimliligina
gerek yok).

Metrikler
---------
ICC(2,1) -- iki-yonlu rastgele etki, mutlak uyum, tek olcum (McGraw&Wong
1996 formulasyonu): "subjects" = (patient_id,scan_id,region) uclusu,
"raters/conditions" = perturbasyon kosullari (scale0.4/scale1.0/scale2.5/
bias). 1.0'a yakin = kosullar arasinda deger neredeyse degismiyor (IYI,
kararli). 0'a yakin/negatif = kosuldan kosula deger neredeyse rastgele
sacilmis (KOTU, kararsiz).

CV%% -- her subject icin 4 kosul arasindaki std/|mean|*100, sonra
subject'ler/ozellikler uzerinden ortalanir -- ICC'nin yorumu zor oldugu
(orn. dusuk between-subject varyans nedeniyle ICC'nin yapay dusuk cikabildigi)
durumlar icin yedek/tamamlayici metrik.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).absolute().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "week3" / "pyradiomics"

ALL_PERTURBATIONS = ["scale0.4", "scale1.0", "scale2.5", "bias"]
EXTRACTION_VARIANTS = ["A", "B", "C32", "C64"]


def icc_2_1(matrix: np.ndarray) -> float:
    """ICC(2,1), iki-yonlu rastgele etki / mutlak-uyum / tek-olcum.

    `matrix`: shape (n_subjects, k_conditions), NaN/inf icermemeli.
    """

    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    grand_mean = matrix.mean()
    subject_means = matrix.mean(axis=1)
    rater_means = matrix.mean(axis=0)

    ss_total = float(np.sum((matrix - grand_mean) ** 2))
    ss_rows = float(k * np.sum((subject_means - grand_mean) ** 2))
    ss_cols = float(n * np.sum((rater_means - grand_mean) ** 2))
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else float("nan")

    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denominator == 0 or not np.isfinite(denominator):
        return float("nan")
    return float((ms_rows - ms_error) / denominator)


def _pivot_subject_condition(
    df: pd.DataFrame, feature_name: str, conditions: list[str]
) -> pd.DataFrame:
    subset = df[df["feature_name"] == feature_name]
    pivot = subset.pivot_table(
        index=["patient_id", "scan_id", "region"],
        columns="perturbation",
        values="value",
        aggfunc="first",
    )
    missing = [c for c in conditions if c not in pivot.columns]
    if missing:
        return pivot.iloc[0:0]
    pivot = pivot[conditions].dropna(axis=0, how="any")
    # sonsuz degerleri de disla (bazi kucuk ROI'lerde pyradiomics inf uretebilir)
    finite_mask = np.isfinite(pivot.to_numpy(dtype=np.float64)).all(axis=1)
    return pivot[finite_mask]


def compute_icc_table(
    df: pd.DataFrame, *, conditions: list[str], group_cols: list[str], min_subjects: int = 3
) -> pd.DataFrame:
    rows = []
    for group_key, group_df in df.groupby(group_cols):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        for feature_name in sorted(group_df["feature_name"].unique()):
            pivot = _pivot_subject_condition(group_df, feature_name, conditions)
            n_subjects = pivot.shape[0]
            if n_subjects < min_subjects:
                continue
            icc_value = icc_2_1(pivot.to_numpy(dtype=np.float64))
            cv_values = (
                pivot.std(axis=1, ddof=1) / pivot.mean(axis=1).abs().replace(0, np.nan)
            ) * 100.0
            row = dict(zip(group_cols, group_key))
            row.update(
                {
                    "feature_name": feature_name,
                    "n_subjects": n_subjects,
                    "icc_2_1": icc_value,
                    "mean_cv_pct": float(cv_values.mean(skipna=True)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(per_feature: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    if per_feature.empty:
        return pd.DataFrame()
    agg = (
        per_feature.groupby(group_cols)
        .agg(
            n_features=("feature_name", "count"),
            mean_icc=("icc_2_1", "mean"),
            median_icc=("icc_2_1", "median"),
            min_icc=("icc_2_1", "min"),
            mean_cv_pct=("mean_cv_pct", "mean"),
            median_cv_pct=("mean_cv_pct", "median"),
        )
        .reset_index()
    )
    return agg


def voxel_bucket(voxel_count: int) -> str:
    if voxel_count < 500:
        return "kucuk_lt500"
    if voxel_count < 5000:
        return "orta_500_5000"
    return "buyuk_gte5000"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="probe_perturbation_stability.py CSV ciktisini ICC/CV ile ozetler."
    )
    parser.add_argument(
        "--long-csv",
        type=Path,
        default=OUTPUT_ROOT / "probe_perturbation_stability_long.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_ROOT,
    )
    args = parser.parse_args()

    if not args.long_csv.is_file():
        raise FileNotFoundError(f"Girdi CSV bulunamadi: {args.long_csv}")

    df = pd.read_csv(args.long_csv)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["voxel_bucket"] = df["roi_voxel_count"].apply(voxel_bucket)

    print(f"Girdi: {args.long_csv} -- {len(df)} satir okundu.", flush=True)
    print(
        "Kaynak x extraction_variant x perturbation dagilimi (satir sayisi):",
        flush=True,
    )
    print(
        df.groupby(["source", "extraction_variant", "perturbation"]).size().to_string(),
        flush=True,
    )

    # ---- 1) Ana tablo: 4-kosul ICC, (source, extraction_variant, feature_class) ----
    per_feature_4cond = compute_icc_table(
        df,
        conditions=ALL_PERTURBATIONS,
        group_cols=["source", "extraction_variant", "feature_class"],
    )
    summary_4cond = summarize(
        per_feature_4cond, group_cols=["source", "extraction_variant", "feature_class"]
    )
    summary_4cond_overall = summarize(
        per_feature_4cond.assign(source="ALL"),
        group_cols=["source", "extraction_variant", "feature_class"],
    )

    per_feature_4cond.to_csv(args.out_dir / "probe_perturbation_icc_per_feature.csv", index=False)
    summary_4cond.to_csv(args.out_dir / "probe_perturbation_icc_summary.csv", index=False)
    summary_4cond_overall.to_csv(
        args.out_dir / "probe_perturbation_icc_summary_overall.csv", index=False
    )

    # ---- 2) Bias-only karsilastirma: scale1.0 vs bias (N4 gercekten duzeltiyor mu?) ----
    per_feature_bias = compute_icc_table(
        df,
        conditions=["scale1.0", "bias"],
        group_cols=["source", "extraction_variant", "feature_class"],
        min_subjects=3,
    )
    summary_bias = summarize(
        per_feature_bias, group_cols=["source", "extraction_variant", "feature_class"]
    )
    summary_bias_overall = summarize(
        per_feature_bias.assign(source="ALL"),
        group_cols=["source", "extraction_variant", "feature_class"],
    )
    per_feature_bias.to_csv(
        args.out_dir / "probe_perturbation_bias_only_icc_per_feature.csv", index=False
    )
    summary_bias.to_csv(args.out_dir / "probe_perturbation_bias_only_icc_summary.csv", index=False)
    summary_bias_overall.to_csv(
        args.out_dir / "probe_perturbation_bias_only_icc_summary_overall.csv", index=False
    )

    # ---- 3) C32 vs C64, ROI-voksel-buyuklugune gore kirilim (yalniz texture) ----
    texture_df = df[
        (df["feature_class"] == "texture") & (df["extraction_variant"].isin(["C32", "C64"]))
    ]
    per_feature_voxel = compute_icc_table(
        texture_df,
        conditions=ALL_PERTURBATIONS,
        group_cols=["extraction_variant", "voxel_bucket"],
        min_subjects=2,
    )
    summary_voxel = summarize(per_feature_voxel, group_cols=["extraction_variant", "voxel_bucket"])
    summary_voxel.to_csv(
        args.out_dir / "probe_perturbation_c32_vs_c64_by_voxel_bucket.csv", index=False
    )

    # ROI voksel sayisi dagilimi (region basina, tek satir/region -- kac tane var, hangi bucket'ta)
    voxel_dist = (
        df[["source", "patient_id", "scan_id", "region", "roi_voxel_count", "voxel_bucket"]]
        .drop_duplicates()
        .groupby(["source", "region", "voxel_bucket"])
        .agg(n=("roi_voxel_count", "count"), median_voxels=("roi_voxel_count", "median"))
        .reset_index()
    )
    voxel_dist.to_csv(args.out_dir / "probe_perturbation_roi_voxel_distribution.csv", index=False)

    print("\n=== OZET: 4-kosul ICC (source x extraction_variant x feature_class) ===", flush=True)
    if not summary_4cond.empty:
        print(summary_4cond.sort_values(["feature_class", "extraction_variant", "source"]).to_string(index=False), flush=True)

    print("\n=== OZET: 4-kosul ICC (TUM kaynaklar birlikte, extraction_variant x feature_class) ===", flush=True)
    if not summary_4cond_overall.empty:
        print(summary_4cond_overall.sort_values(["feature_class", "extraction_variant"]).to_string(index=False), flush=True)

    print("\n=== OZET: bias-only (scale1.0 vs bias) ICC, TUM kaynaklar ===", flush=True)
    if not summary_bias_overall.empty:
        print(summary_bias_overall.sort_values(["feature_class", "extraction_variant"]).to_string(index=False), flush=True)

    print("\n=== OZET: C32 vs C64, ROI-voksel-buyuklugune gore (yalniz texture) ===", flush=True)
    if not summary_voxel.empty:
        print(summary_voxel.sort_values(["voxel_bucket", "extraction_variant"]).to_string(index=False), flush=True)

    print(f"\nBitti. Ciktilar: {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
