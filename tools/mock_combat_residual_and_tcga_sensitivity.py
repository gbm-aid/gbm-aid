"""Dogrulama Adim B+C -- MOCK/kesif amacli, DB'ye YAZMAZ.

Bu script 2026-08-13 gorevinin ("modeling-agent Adim B+C") somut
uygulamasidir. TUM DB erisimi SELECT/readonly'dir (`get_connection
(readonly=True)`), hicbir tabloya INSERT/UPDATE yapilmaz, `combat_
parameters` tablosuna YAZILMAZ -- burada uretilen ComBat modeli
GECICI/tek-seferlik bir tanisal fit'tir, gercek/kalici model DEGILDIR.

Kapsam
------
Adim B (mekanik fit + residual site association):
    LUMIERE'in BUGUNKU DB yazimi (`segmentation_tool=
    'LUMIERE-PyRadiomics-107'`, Task #19) bu script calistirilirken
    HENUZ TAMAMLANMAMISTI (1521/1791 satir, aktif INSERT --
    AKTIF-GOREVLER.md satir 51 "CALISIYOR"). Protokol geregi bu satirlar
    KULLANILMADI. Bunun yerine, ComBat fit/apply MEKANIGININ dogru
    calistigini gostermek icin ESKI (Hafta 1'den kalma, binWidth/
    normalize parametreleri UPenn/TCGA'dan FARKLI, bkz. 2026-08-09
    Blokaj 2) LUMIERE precomputed verisi (`segmentation_tool=
    'DeepBraTumIA'`) kullanildi -- bu segmentation_tool degeri
    AKTIF YAZIMDAN TAMAMEN AYRI bir kaynak, hicbir kaynak catismasi
    YOK. SONUCLAR URETIM ComBat sayilari DEGILDIR, sadece "fit/apply
    kodu residual site sinyalini gercekten azaltiyor mu" sorusunun
    MEKANIK bir testidir -- acikca boyle raporlanmalidir.

Adim C (TCGA siralama duyarliligi):
    TCGA-ground-truth WT bolgesi (39 hasta, 1'i vital_status NULL
    oldugu icin 38 kullanilabilir) uzerinde basit (LASSO/univariate
    on-secim + kucuk-penalizer CoxPH) bir risk skoru kurulur, sonra
    Ege'nin olctugu T1ce Z-score sigma araligina ([0.89, 1.69])
    dayanan sentetik carpimsal gurultu (yalniz first-order+texture,
    shape SABIT -- sekil ozellikleri yogunluk olcegi degisiminden
    etkilenmez) uygulanir, siralama degisimi Spearman/Kendall +
    swap-rate ile olculur (Monte Carlo, N tekrar).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from db_connection import get_connection
from pipeline.harmonization import (
    ComBatApplyLeakageError,
    apply_combat_harmonization,
    fit_combat_harmonization,
)

RNG_SEED = 42
N_PER_SOURCE = 40  # kucuk alt-orneklem (hiz icin, ~2600 degil)
MC_DRAWS = 500

FEATURE_KEY_ORDER = None  # ilk satirdan turetilecek (deterministik sirali)


def _load_feature_rows(cursor, segmentation_tool: str, tumor_region: str, limit: int, seed: int) -> pd.DataFrame:
    """Belirli segmentation_tool+tumor_region icin non-null 107-ozellik satirlarini cek."""

    cursor.execute(
        """
        SELECT r.radiomics_id, r.scan_id, ms.patient_id,
               r.shape_features, r.first_order_features, r.texture_features
        FROM radiomics r
        JOIN mr_scans ms ON ms.scan_id = r.scan_id
        WHERE r.segmentation_tool = %s
          AND r.tumor_region = %s
        ORDER BY r.radiomics_id
        """,
        (segmentation_tool, tumor_region),
    )
    rows = cursor.fetchall()
    # Bos/NULL JSONB satirlarini (0-hacim, "bolge yok" satirlari) Python'da
    # ele -- WHERE'de jsonb != '{}'::jsonb kiyaslamasi pooler/DB'de asiri
    # yavas cikti (statement_timeout'a carpti), bu yuzden filtre buraya
    # tasindi (ayni sonuc, cok daha hizli).
    rows = [
        row for row in rows
        if row[3] and row[4] and row[5]
        and len(row[3]) > 0 and len(row[4]) > 0 and len(row[5]) > 0
    ]
    if not rows:
        raise RuntimeError(f"Hic satir bulunamadi: {segmentation_tool}/{tumor_region}")

    rng = np.random.default_rng(seed)
    if len(rows) > limit:
        idx = rng.choice(len(rows), size=limit, replace=False)
        rows = [rows[i] for i in sorted(idx)]

    records = []
    for radiomics_id, scan_id, patient_id, shape_f, fo_f, tex_f in rows:
        merged = {**shape_f, **fo_f, **tex_f}
        merged["_radiomics_id"] = radiomics_id
        merged["_scan_id"] = scan_id
        merged["_patient_id"] = patient_id
        records.append(merged)

    frame = pd.DataFrame(records).set_index("_radiomics_id")
    return frame


def _canonical_107_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(c for c in frame.columns if not c.startswith("_"))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_std <= 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def step_b_mechanical_fit_test(cursor) -> dict:
    print("=" * 70)
    print("ADIM B -- Mekanik ComBat fit + residual site association testi")
    print("=" * 70)
    print(
        "UYARI: LUMIERE-PyRadiomics-107 (yeni, Task #19) yazimi aktif "
        "(1521/1791, tamamlanmamis) -- KULLANILMADI. Bunun yerine ESKI "
        "DeepBraTumIA precomputed LUMIERE verisi (binWidth/normalize "
        "UPenn'den FARKLI, 2026-08-09 Blokaj 2) kullanildi. Bu SADECE "
        "fit/apply KODUNUN mekanigini test eder, URETIM ComBat sayisi "
        "DEGILDIR."
    )

    upenn = _load_feature_rows(cursor, "UPenn-PyRadiomics-107", "NC", N_PER_SOURCE, RNG_SEED)
    lumiere = _load_feature_rows(cursor, "DeepBraTumIA", "Necrosis", N_PER_SOURCE, RNG_SEED + 1)

    upenn_cols = set(_canonical_107_columns(upenn))
    lumiere_cols = set(_canonical_107_columns(lumiere))
    common_cols = sorted(upenn_cols & lumiere_cols)
    print(f"UPenn satir={len(upenn)}, LUMIERE(eski) satir={len(lumiere)}, ortak ozellik sayisi={len(common_cols)}")
    if len(common_cols) != 107:
        print(f"  DIKKAT: ortak kolon sayisi 107 degil ({len(common_cols)}) -- fark: "
              f"{sorted(upenn_cols ^ lumiere_cols)[:10]}")

    upenn_feat = upenn[common_cols].astype(np.float64)
    lumiere_feat = lumiere[common_cols].astype(np.float64)

    # NaN/inf satirlarini at (ComBat girdisi sonlu olmali)
    upenn_feat = upenn_feat.replace([np.inf, -np.inf], np.nan).dropna()
    lumiere_feat = lumiere_feat.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"NaN/inf temizligi sonrasi: UPenn={len(upenn_feat)}, LUMIERE={len(lumiere_feat)}")

    combined = pd.concat([upenn_feat, lumiere_feat], axis=0)
    combined.index = [f"row_{i}" for i in range(len(combined))]
    site = pd.Series(
        ["UPenn"] * len(upenn_feat) + ["LUMIERE"] * len(lumiere_feat),
        index=combined.index,
        name="SITE",
    )
    covariates = pd.DataFrame({"SITE": site})

    model, harmonized, parameter_rows = fit_combat_harmonization(
        combined, covariates, reference_batch="UPenn", seed=RNG_SEED
    )
    print(f"fit_combat_harmonization() basarili -- parameter_rows={len(parameter_rows)} satir "
          f"(combat_parameters'a YAZILMADI, sadece bellekte)")

    upenn_mask = site == "UPenn"
    lumiere_mask = site == "LUMIERE"

    pre_d = []
    post_d = []
    pre_p = []
    post_p = []
    for col in common_cols:
        a_pre = combined.loc[upenn_mask, col].to_numpy()
        b_pre = combined.loc[lumiere_mask, col].to_numpy()
        a_post = harmonized.loc[upenn_mask, col].to_numpy()
        b_post = harmonized.loc[lumiere_mask, col].to_numpy()

        pre_d.append(abs(_cohens_d(a_pre, b_pre)))
        post_d.append(abs(_cohens_d(a_post, b_post)))
        pre_p.append(stats.ttest_ind(a_pre, b_pre, equal_var=False).pvalue)
        post_p.append(stats.ttest_ind(a_post, b_post, equal_var=False).pvalue)

    pre_d = np.array(pre_d)
    post_d = np.array(post_d)
    pre_p = np.array(pre_p)
    post_p = np.array(post_p)

    n_sig_pre = int((pre_p < 0.05).sum())
    n_sig_post = int((post_p < 0.05).sum())

    print(f"\nResidual site association (UPenn vs LUMIERE(eski), 107 ozellik, region=NC/Necrosis):")
    print(f"  ComBat ONCESI: |Cohen d| ortalama={pre_d.mean():.3f}, medyan={np.median(pre_d):.3f}, "
          f"p<0.05 olan ozellik sayisi={n_sig_pre}/{len(common_cols)}")
    print(f"  ComBat SONRASI: |Cohen d| ortalama={post_d.mean():.3f}, medyan={np.median(post_d):.3f}, "
          f"p<0.05 olan ozellik sayisi={n_sig_post}/{len(common_cols)}")
    print(f"  (Referans-batch=UPenn oldugu icin UPenn satirlari matematiksel olarak degismez, "
          f"harmonized_in_sample'daki UPenn satirlari raw ile ayni -- bu beklenen/dokumante "
          f"edilmis davranis, bug degil.)")

    # apply_combat_harmonization() TCGA guard doğrulaması (guard calisiyor mu?)
    print("\n-- apply_combat_harmonization() TCGA guard canli dogrulama --")
    fake_tcga_features = upenn_feat.iloc[:3].copy()
    fake_tcga_features.index = [f"tcga_row_{i}" for i in range(3)]
    fake_covars = pd.DataFrame({"SITE": ["UPenn"] * 3}, index=fake_tcga_features.index)
    fake_true_source = pd.Series(["TCGA-GBM"] * 3, index=fake_tcga_features.index)
    try:
        apply_combat_harmonization(
            fake_tcga_features, fake_covars, model, true_dataset_source=fake_true_source
        )
        print("  BEKLENMEDIK: guard TCGA'yi REDDETMEDI -- bu bir REGRESYON, incelenmeli!")
    except ComBatApplyLeakageError:
        print("  DOGRULANDI: apply_combat_harmonization() true_dataset_source='TCGA-GBM' "
              "icin ComBatApplyLeakageError firlatti (guard calisiyor, kod-seviyesinde "
              "TCGA'nin ComBat'a hicbir sekilde giremeyecegi teyit edildi).")

    return {
        "pre_mean_abs_d": float(pre_d.mean()),
        "post_mean_abs_d": float(post_d.mean()),
        "n_sig_pre": n_sig_pre,
        "n_sig_post": n_sig_post,
        "n_features": len(common_cols),
    }


def step_b_raw_tcga_vs_upenn_context(cursor) -> None:
    print("\n" + "=" * 70)
    print("ADIM B (ek baglam) -- TCGA vs UPenn HAM ozellik farki (region-uyumsuz, tanisal)")
    print("=" * 70)
    print(
        "UYARI: TCGA sadece WT/TC uretir, UPenn/LUMIERE sadece NC/ED/ET -- "
        "hicbir ortak tumor_region YOK (2026-08-09 Blokaj 1, KAPANMADI). "
        "Asagidaki karsilastirma bolge-tanimi FARKLI iki maskeyi kiyaslar, "
        "bu yuzden olculen fark 'batch etkisi' + 'bolge tanimi farki'nin "
        "AYRISTIRILAMAZ TOPLAMIDIR -- salt bir kaynak-sinyali OLCUMU "
        "DEGILDIR, sadece buyukluk-mertebesi baglam icin verilir."
    )

    tcga_wt = _load_feature_rows(cursor, "TCGA-ground-truth", "WT", 39, RNG_SEED)
    upenn_et = _load_feature_rows(cursor, "UPenn-PyRadiomics-107", "ET", N_PER_SOURCE, RNG_SEED)

    common_cols = sorted(set(_canonical_107_columns(tcga_wt)) & set(_canonical_107_columns(upenn_et)))
    a = tcga_wt[common_cols].astype(np.float64).replace([np.inf, -np.inf], np.nan).dropna()
    b = upenn_et[common_cols].astype(np.float64).replace([np.inf, -np.inf], np.nan).dropna()

    ds = [abs(_cohens_d(a[c].to_numpy(), b[c].to_numpy())) for c in common_cols]
    ds = np.array(ds)
    print(f"TCGA/WT (n={len(a)}) vs UPenn/ET (n={len(b)}) ham |Cohen d|: "
          f"ortalama={ds.mean():.3f}, medyan={np.median(ds):.3f}, maks={ds.max():.3f}")
    print("(Karsilastirma icin Adim B'deki UPenn-vs-LUMIERE ComBat-ONCESI ortalama |d| ile "
          "AYNI mertebede mi/cok mu buyuk oldugunu Adim C yorumunda kullan.)")


def step_c_tcga_ranking_sensitivity(cursor) -> dict:
    print("\n" + "=" * 70)
    print("ADIM C -- TCGA siralama duyarliligi (sentetik Z-score sigma pertürbasyonu)")
    print("=" * 70)

    cursor.execute(
        """
        SELECT r.radiomics_id, ms.patient_id, p.survival_days, p.vital_status,
               r.shape_features, r.first_order_features, r.texture_features
        FROM radiomics r
        JOIN mr_scans ms ON ms.scan_id = r.scan_id
        JOIN patients p ON p.patient_id = ms.patient_id
        WHERE r.segmentation_tool = 'TCGA-ground-truth' AND r.tumor_region = 'WT'
        ORDER BY ms.patient_id
        """
    )
    rows = cursor.fetchall()
    records = []
    for radiomics_id, patient_id, survival_days, vital_status, shape_f, fo_f, tex_f in rows:
        if survival_days is None or vital_status not in ("ALIVE", "DECEASED"):
            continue
        merged = {**shape_f, **fo_f, **tex_f}
        merged["_patient_id"] = patient_id
        merged["survival_days"] = survival_days
        merged["event"] = 1 if vital_status == "DECEASED" else 0
        records.append(merged)

    frame = pd.DataFrame(records).set_index("_patient_id")
    print(f"Kullanilabilir TCGA/WT hasta sayisi: {len(frame)} "
          f"(39 toplamdan, survival_days/vital_status eksik olan(lar) atlandi)")

    feature_cols = sorted(c for c in frame.columns if c not in ("survival_days", "event"))
    shape_cols = [c for c in feature_cols if "_shape_" in c]
    intensity_cols = [c for c in feature_cols if c not in shape_cols]
    print(f"shape (sabit tutulacak, pertürbasyondan MUAF): {len(shape_cols)}, "
          f"first-order+texture (pertürbe edilecek): {len(intensity_cols)}")

    X = frame[feature_cols].astype(np.float64)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="any")  # NaN iceren kolonlari at (Cox girdisi sonlu olmali)
    feature_cols = list(X.columns)
    shape_cols = [c for c in feature_cols if "_shape_" in c]
    intensity_cols = [c for c in feature_cols if c not in shape_cols]

    duration = frame["survival_days"].astype(np.float64)
    event = frame["event"].astype(int)

    # Univariate on-secim: Spearman |rho| ile survival_days'e en cok korele 10 ozellik
    correlations = {}
    for c in feature_cols:
        rho, _ = stats.spearmanr(X[c], duration)
        correlations[c] = abs(rho) if np.isfinite(rho) else 0.0
    top_features = sorted(correlations, key=correlations.get, reverse=True)[:10]
    print(f"Univariate on-secimle secilen 10 ozellik: {top_features}")

    top_shape = [c for c in top_features if c in shape_cols]
    top_intensity = [c for c in top_features if c in intensity_cols]
    print(f"  -- bunlarin {len(top_shape)} tanesi shape (pertürbasyondan muaf), "
          f"{len(top_intensity)} tanesi first-order/texture (pertürbe edilecek)")

    means = X[top_features].mean()
    stds = X[top_features].std(ddof=1)
    Z = (X[top_features] - means) / stds

    from lifelines import CoxPHFitter

    cox_frame = pd.concat([Z, duration.rename("survival_days"), event.rename("event")], axis=1)
    cox = CoxPHFitter(penalizer=0.5)
    cox.fit(cox_frame, duration_col="survival_days", event_col="event")
    baseline_hazard = cox.predict_partial_hazard(Z)
    baseline_rank = baseline_hazard.rank()

    print(f"Baseline Cox katsayilari (n={len(frame)} hasta, penalizer=0.5):")
    print(cox.params_.round(4).to_string())

    # Ege'nin T1ce sigma araligindan turetilen carpimsal pertürbasyon:
    # sigma_i in [0.89, 1.69], varsayilan kohort-fit sigma ~ orta nokta 1.20
    # olcek-carpani = sigma_fit / sigma_i -> [1.20/1.69, 1.20/0.89] = [0.71, 1.35]
    sigma_range = (0.89, 1.69)
    assumed_fit_sigma = float(np.mean(sigma_range))
    scale_low = assumed_fit_sigma / sigma_range[1]
    scale_high = assumed_fit_sigma / sigma_range[0]
    print(f"\nPertürbasyon carpani araligi (Ege sigma={sigma_range} 'den turetildi, "
          f"varsayilan fit-sigma={assumed_fit_sigma:.2f}): [{scale_low:.3f}, {scale_high:.3f}]")

    rng = np.random.default_rng(RNG_SEED)
    n_patients = len(frame)
    spearman_results = []
    kendall_results = []
    swap_rates = []

    for _ in range(MC_DRAWS):
        factors = rng.uniform(scale_low, scale_high, size=n_patients)
        X_perturbed = X[top_features].copy()
        for c in top_intensity:
            X_perturbed[c] = X_perturbed[c] * factors
        # shape kolonlari (top_shape) DEGISTIRILMEZ (yogunluk olceginden bagimsiz)

        Z_perturbed = (X_perturbed - means) / stds
        perturbed_hazard = cox.predict_partial_hazard(Z_perturbed)

        rho, _ = stats.spearmanr(baseline_hazard, perturbed_hazard)
        tau, _ = stats.kendalltau(baseline_hazard, perturbed_hazard)
        spearman_results.append(rho)
        kendall_results.append(tau)

        # swap-rate: ikili siralarin ne kadari degisti
        base_order = baseline_hazard.to_numpy()
        pert_order = perturbed_hazard.to_numpy()
        n = len(base_order)
        n_pairs = n * (n - 1) // 2
        n_swaps = 0
        for i in range(n):
            for j in range(i + 1, n):
                base_sign = np.sign(base_order[i] - base_order[j])
                pert_sign = np.sign(pert_order[i] - pert_order[j])
                if base_sign != pert_sign and base_sign != 0:
                    n_swaps += 1
        swap_rates.append(n_swaps / n_pairs)

    spearman_results = np.array(spearman_results)
    kendall_results = np.array(kendall_results)
    swap_rates = np.array(swap_rates)

    print(f"\nMonte Carlo sonuclari ({MC_DRAWS} tekrar, n={n_patients} hasta, "
          f"{n_patients * (n_patients - 1) // 2} ikili):")
    print(f"  Spearman rho: ortalama={spearman_results.mean():.4f}, std={spearman_results.std():.4f}, "
          f"min={spearman_results.min():.4f}, maks={spearman_results.max():.4f}")
    print(f"  Kendall tau:  ortalama={kendall_results.mean():.4f}, std={kendall_results.std():.4f}, "
          f"min={kendall_results.min():.4f}")
    print(f"  Swap-rate:    ortalama={swap_rates.mean():.4f}, std={swap_rates.std():.4f}, "
          f"maks={swap_rates.max():.4f}")

    return {
        "n_patients": n_patients,
        "spearman_mean": float(spearman_results.mean()),
        "spearman_std": float(spearman_results.std()),
        "spearman_min": float(spearman_results.min()),
        "kendall_mean": float(kendall_results.mean()),
        "swap_rate_mean": float(swap_rates.mean()),
        "swap_rate_max": float(swap_rates.max()),
    }


def main() -> None:
    connection = get_connection(readonly=True)
    try:
        cursor = connection.cursor()
        try:
            step_b_results = step_b_mechanical_fit_test(cursor)
            step_b_raw_tcga_vs_upenn_context(cursor)
            step_c_results = step_c_tcga_ranking_sensitivity(cursor)
        finally:
            cursor.close()
    finally:
        connection.close()

    print("\n" + "=" * 70)
    print("OZET")
    print("=" * 70)
    print("Adim B (mekanik):", step_b_results)
    print("Adim C:", step_c_results)


if __name__ == "__main__":
    main()
