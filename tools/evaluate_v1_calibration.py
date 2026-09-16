# -*- coding: utf-8 -*-
"""v1_referans icin HARICI (UCSF) Brier skoru + kalibrasyon degerlendirmesi.

NEDEN (2026-08-19, iyilestirme plani #2'nin ikinci yarisi)
-----------------------------------------------------------
Ayrim gucu (C-index/AUC) SIRALAMAYI olcer; kalibrasyon BUYUKLUGU olcer:
model "1 yil sagkalim olasiligin %30" diyorsa boyle dedigi hastalarin
gercekten ~%30'u mu 1 yili goruyor? Klinik karar destek sistemi hasta
basi risk GOSTERECEKSE bu zorunludur -- mukemmel siralayan ama fena
yanilan bir model C-index'te kusursuz gorunur.

NEDEN YALNIZ v1: mutlak sagkalim olasiligi S(t|x) icin baseline hazard
gerekir; o da yalniz v1'in kayitli artefaktinda var
(`models/cox_phm_primary_wt93_clinical_full_ucsf_2026-08-18.pkl`,
`fitted_model.baseline_cumulative_hazard_`). Diger varyantlar icin
yeniden fit gerekirdi -- o ayri bir is, burada YAPILMIYOR ve bu kisit
raporda beyan edilir.

DOGRULAMA KAPISI (2026-08-19 REVIZE): katsayi kimligi -- pkl.params_
ile week3_final_model_coefficients.csv maks |fark| <= 1e-5. Ilk surum
"Harrell == 0.678064" bekliyordu; kapi DUSTU ve gorevini yapti: olcum,
pipeline'da IKI AYRI final fit oldugunu ortaya cikardi (degerlendirme
zinciri standardize-yas -> 0.678064; dagitim/pkl fit'i HAM-yas + ridge
-> olculen 0.676209). Detay: main() icindeki KAPI blogu yorumu.

EGITIM HAVUZU KAPISI (2026-09-14 EKLENDI, Baris'in 5 nolu karari):
`build_frames()` icinde `w3.check_training_pool_counts()` -- SERT
611/585 (WT-only), fail-closed. Bu scriptte Harrell bit-esitlik
kapisi YOK (yukaridaki revizyon: iki ayri fit) ve egitim cercevesi
sayisal olarak kullanilmiyor -- yani havuz kaymasina karsi BASKA
hicbir koruma yoktu.

URETILENLER (yalniz YENI dosyalar, DB'ye yazma YOK):
  week3_v1_brier_ipa.csv          Brier(t) + Brier_null(t) + IPA(t)
  week3_v1_calibration_365.csv    decile kalibrasyon tablosu (t=365)
  stdout                          kalibrasyon egimi + ozet

METRIK NOTLARI (rapora aynen tasinacak durustluk notlari):
- Brier tek basina okunmaz; kiyas KM-null modelidir (kovaryatsiz).
  IPA = 1 - Brier/Brier_null ("index of prediction accuracy"):
  0 = null modelden farksiz, negatif = null'dan KOTU.
- IPCW sansur dagilimi UCSF'in KENDISINDEN kestirildi (harici kohortun
  kendi sansur sureci) -- `evaluate_external_metrics.py` ile ayni tercih.
- Kalibrasyon egimi: harici veride tek kovaryat olarak dogrusal
  prediktorle Cox fit'i; egim 1.0 ideal, <1 asiri-iddiali (risk yayilimi
  fazla), >1 az-iddiali.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import train_cox_week3 as w3  # noqa: E402
from pipeline.cox_model import (  # noqa: E402
    build_training_frame,
    standardize_columns_fold_safe,
)
from db_connection import get_connection  # noqa: E402
from psycopg2.extras import RealDictCursor  # noqa: E402
from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402
from sksurv.metrics import brier_score  # noqa: E402
from sksurv.util import Surv  # noqa: E402

OUT_DIR = PROJECT_ROOT / "artifacts" / "week3" / "cox_model"
CACHE = OUT_DIR / "_cache"
PKL = PROJECT_ROOT / "models" / "cox_phm_primary_wt93_clinical_full_ucsf_2026-08-18.pkl"
EVAL_CHAIN_C = 0.6780636686756297  # degerlendirme zincirinin C'si
COEF_GATE_TOL = 1e-5
TIMES = [365, 548, 730]
N_CAL_BINS = 10


def build_frames():
    conn = get_connection(readonly=True)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    def cached(tag, tool):
        cur.execute(
            "SELECT COUNT(*) AS n FROM radiomics WHERE segmentation_tool=%s", (tool,)
        )
        n = int(cur.fetchone()["n"])
        f = pd.read_pickle(CACHE / f"{tag}_long.pkl")
        if len(f) != n:
            raise RuntimeError(f"{tag}: onbellek {len(f)} != DB {n}")
        return f

    upenn_long = cached("upenn", w3.SEGMENTATION_TOOL_UPENN_C32)
    ucsf_long = cached("ucsf", w3.SEGMENTATION_TOOL_UCSF_C32)
    upenn_pat = w3.fetch_patients_frame(cur, source_name=w3.UPENN_SOURCE_NAME)
    ucsf_pat = w3.fetch_patients_frame(cur, source_name=w3.UCSF_SOURCE_NAME)
    cur.close()
    conn.close()

    up_wide, _ = w3.pivot_radiomics_long_to_wide(upenn_long, regions=["WT"])
    uc_wide, _ = w3.pivot_ucsf_regions_long_to_wide(ucsf_long, regions=("WT",))
    cand = w3.select_feature_columns(up_wide, w3.STABLE_FEATURES_ICC60, region="WT")

    extra = w3.build_variant_clinical_extra_columns(include_mgmt=False, use_age_spline=False)
    up_c = upenn_pat.join(
        w3.build_variant_clinical_frame(
            upenn_pat, include_mgmt=False, use_age_spline=False, age_knots=None
        )
    )
    uc_c = ucsf_pat.join(
        w3.build_variant_clinical_frame(
            ucsf_pat, include_mgmt=False, use_age_spline=False, age_knots=None
        )
    )
    train, train_report = build_training_frame(
        up_wide[cand], up_c, check_combat_identity=True, passthrough_columns=extra
    )
    # 2026-09-14 ComBat OZDESLIK GUARD'I (modeling-agent-T1, Baris onayi):
    # `check_combat_identity` `False` -> `True`. `build_training_frame()`
    # docstring'i `False`'i yalniz "ComBat'a hic bagimli olmayan mekanik
    # test" icin mesru sayiyor; bu script rapora giren v1 kalibrasyon
    # sayilarini uretiyor. OLCULDU (canli DB, oncesi/sonrasi): 611/585,
    # dusen hasta 0, cerceve degerleri BIREBIR AYNI.
    ext, _ = w3.build_external_test_frame(
        uc_wide[cand], uc_c, passthrough_columns=extra
    )
    # DAGITIM modeli (pkl) HAM yasla fit edildi (coef 0.023/yil,
    # _norm_mean[age]=62.47 -- olculdu) -> standardizasyon UYGULANMAZ.
    #
    # 2026-09-14 EGITIM HAVUZU KAPISI (modeling-agent-S2, Baris'in 5 nolu
    # karari). Bu script egitim cercevesini SAYISAL olarak KULLANMIYOR
    # (yukaridaki not) -- burasi bir "frame-insa dogrulamasi". Ama o
    # dogrulama SAYISIZ yapilirsa hicbir sey dogrulamaz: kaynak whitelist'i
    # (`build_training_frame` icinde, TCGA/UCSF reddi) gecerken havuz
    # 611/585'ten sessizce kayabilir ve script bunu FARK ETMEZ. Kapi
    # `train_cox_week3.py:3003` ile AYNI desende (v1 = WT-only -> SERT
    # 611/585, beyan edilmis azalma YOK).
    w3.check_training_pool_counts(
        train_report.n_output_rows,
        int(train["event"].sum()),
        expected_patients=w3.EXPECTED_UPENN_TRAINING_PATIENTS,
        expected_events=w3.EXPECTED_UPENN_TRAINING_EVENTS,
        allow_mismatch=False,
        regions=("WT",),
        dropped_patient_ids=w3.all_dropped_patient_ids(train_report),
    )

    # --- HARICI VERI KIMLIK KAPILARI (2026-08-19 Codex bulgusu #2:
    # "harici veri yalniz satir sayisiyla dogrulaniyor; ayni sayida
    # farkli kayit sessizce gecer") ------------------------------------
    # (a) Kilitli havuz kapisi: 295/169, fail-loud (sayi).
    w3.check_external_test_pool_counts(
        len(ext), int(ext["event"].sum()),
        expected_patients=295, expected_events=169,
        allow_mismatch=False, label="UCSF (kalibrasyon)",
    )
    # (b) KIMLIK: frame insasinda SESSIZCE dusen hasta olmamali --
    # pivot'un 295 hastasi ile cikan cercevenin hastalari BIREBIR ayni.
    dropped = sorted(set(uc_wide.index) - set(ext.index))
    extra_ids = sorted(set(ext.index) - set(uc_wide.index))
    if dropped or extra_ids:
        raise RuntimeError(
            f"UCSF kimlik kapisi: dusen={dropped[:5]} fazla={extra_ids[:5]} "
            "-- sayi tutsa bile kayit kumesi degisti, SESSIZCE devam YOK."
        )
    # (c) DONDURULMUS KIMLIK BASELINE'I (2026-08-19 Codex bulgusu:
    # yazdirilan-ama-karsilastirilmayan parmak izi kapi DEGILDIR).
    # pid + sagkalim + WT-pivot OZELLIK DEGERLERI uc SHA-256 ile
    # baseline dosyasina karsi dogrulanir; uyusmazsa DURUR.
    # v2 (Codex 2. tur): IKI kohort + TUM-kolon hasta hash'i (klinik
    # kovaryat degerleri dahil) + WT pivot degerleri. UPenn de kilitli:
    # standardizasyon istatistikleri/knot'lar egitim verisinden turetiliyor.
    from external_identity_baseline import compute_full_identity, verify_or_fail
    identity = compute_full_identity(uc_wide, ucsf_pat, up_wide, upenn_pat)
    baseline = verify_or_fail(identity)
    _u = identity["cohorts"]["UCSF-PDGM"]
    print(f"KIMLIK KAPISI GECTI (UCSF+UPenn, sema v2): "
          f"ucsf_patients={_u['patients_sha256'][:12]}... "
          f"ucsf_wt_pivot={_u['wt_pivot_sha256'][:12]}... "
          f"(baseline: {baseline['frozen_at'][:19]})")
    return ext


def main() -> int:
    with open(PKL, "rb") as f:
        art = pickle.load(f)
    model: CoxPHFitter = art["fitted_model"]
    covs = list(model.params_.index)

    ext = build_frames()
    X = ext[covs]
    T = ext["survival_days"].to_numpy(dtype=float)
    E = ext["event"].to_numpy(dtype=int)

    # --- KAPI: KATSAYI KIMLIGI (2026-08-19 REVIZE) --------------------
    # Ilk surum Harrell==0.678064 bekliyordu; OLCUM bunun yanlis beklenti
    # oldugunu gosterdi. Pipeline'da IKI AYRI final fit var:
    #   (a) DEGERLENDIRME zinciri (standardize yas) -> raporlanan 0.678064
    #   (b) DAGITIM modeli (HAM yas, ridge penalizer=0.05/l1=0.0) -> pkl;
    #       katsayilari week3_final_model_coefficients.csv ile 4e-07 icinde
    #       ayni; HAM-yas cercevesinde olculen harici Harrell = 0.676209.
    # BULGU: pkl'nin `external_test.c_index` alani 0.678064 tasiyor --
    #    dagitim artefakti, degerlendirme zincirinin sayisini metadata'sinda
    #    tasiyor. /predict'in sundugu modelin GERCEK harici Harrell'i
    #    0.676209 (fark 0.0019 -- kucuk ama BELGELENMEMISTI).
    # Yeni kapi: katsayi kimligi (pkl == dagitim CSV'si); Harrell dayatilmaz,
    # iki deger de ciktida raporlanir. Kalibrasyon DAGITIM modeline aittir.
    ref = pd.read_csv(OUT_DIR / "week3_final_model_coefficients.csv")
    ref_cc = "covariate" if "covariate" in ref.columns else ref.columns[0]
    ref_coefcol = [c for c in ref.columns if "coef" in c.lower()][0]
    ref_coef = ref.set_index(ref_cc)[ref_coefcol]

    # 2026-08-19 Codex bulgusu #1 (FAIL-OPEN): pandas Series farki
    # INDEKS HIZALAMASIYLA calisir -- iki tarafta ortak olmayan kovaryat
    # NaN olur ve `.max()` NaN'i SESSIZCE atlar. Yani pkl'de fazladan ya
    # da eksik bir kovaryat kapidan gorunmez gecerdi. Once KUME ESITLIGI,
    # sonra SONLULUK, ancak ondan sonra fark:
    pkl_set, ref_set = set(model.params_.index), set(ref_coef.index)
    if pkl_set != ref_set:
        raise RuntimeError(
            "KAPI: kovaryat kumeleri FARKLI -- yalniz-pkl'de: "
            f"{sorted(pkl_set - ref_set)}; yalniz-CSV'de: {sorted(ref_set - pkl_set)}"
        )
    if not (np.isfinite(model.params_.to_numpy()).all()
            and np.isfinite(ref_coef.to_numpy()).all()):
        raise RuntimeError("KAPI: katsayilarda NaN/inf var -- devam YOK.")
    X_arr = X.to_numpy(dtype=float)
    if not np.isfinite(X_arr).all():
        bad = X.columns[~np.isfinite(X_arr).all(axis=0)].tolist()
        raise RuntimeError(f"KAPI: harici kovaryat matrisinde NaN/inf: {bad}")

    coef_diff = float((model.params_ - ref_coef.reindex(model.params_.index)).abs().max())
    assert np.isfinite(coef_diff)
    print(f"KAPI (katsayi kimligi): pkl vs week3_final_model_coefficients.csv "
          f"maks |fark| = {coef_diff:.2e}  (esik {COEF_GATE_TOL:.0e})")
    if coef_diff > COEF_GATE_TOL:
        print("KAPI BASARISIZ -- pkl beklenen dagitim fit'i DEGIL.", file=sys.stderr)
        return 2

    lp = model.predict_log_partial_hazard(X).to_numpy()
    harrell = concordance_index(T, -lp, E)
    print(f"KAPI GECTI. Dagitim modelinin olculen harici Harrell C = {harrell:.6f}")
    print(f"  (degerlendirme zincirinin raporlanan C'si = {EVAL_CHAIN_C:.6f}; "
          f"fark {abs(harrell - EVAL_CHAIN_C):.4f} -- iki AYRI fit, ustteki bulgu)")
    print()

    # --- S(t|x) tahminleri -------------------------------------------
    surv_fn = model.predict_survival_function(X, times=TIMES)  # satir=t, kolon=hasta
    surv = Surv.from_arrays(event=E.astype(bool), time=T)

    # --- Brier + null-Brier + IPA ------------------------------------
    km_all = KaplanMeierFitter().fit(T, E)
    rows = []
    for t in TIMES:
        pred = surv_fn.loc[t].to_numpy()
        _, (bs,) = brier_score(surv, surv, pred.reshape(-1, 1), [t])
        km_s = float(km_all.survival_function_at_times(t).iloc[0])
        _, (bs_null,) = brier_score(
            surv, surv, np.full((len(T), 1), km_s), [t]
        )
        ipa = 1.0 - bs / bs_null
        rows.append(
            {"t_gun": t, "brier": round(bs, 4), "brier_null_km": round(bs_null, 4),
             "IPA": round(ipa, 4), "km_pop_sagkalim": round(km_s, 4)}
        )
    brier_table = pd.DataFrame(rows)
    brier_table.to_csv(OUT_DIR / "week3_v1_brier_ipa.csv", index=False)
    print("BRIER / IPA (null = kovaryatsiz KM):")
    print(brier_table.to_string(index=False))

    # --- Kalibrasyon egimi -------------------------------------------
    cal_df = pd.DataFrame({"lp": lp, "T": T, "E": E})
    slope_model = CoxPHFitter().fit(cal_df, duration_col="T", event_col="E")
    slope = float(slope_model.params_["lp"])
    slope_lo, slope_hi = slope_model.confidence_intervals_.loc["lp"]
    print(f"\nKALIBRASYON EGIMI: {slope:.3f}  [%95 CI {slope_lo:.3f} - {slope_hi:.3f}]"
          "   (1.0 ideal; <1 asiri-iddiali, >1 az-iddiali)")

    # --- Decile kalibrasyonu (t=365) ---------------------------------
    pred365 = surv_fn.loc[365]
    dec = pd.qcut(pred365.rank(method="first"), N_CAL_BINS, labels=False)
    cal_rows = []
    for d in range(N_CAL_BINS):
        idx = dec[dec == d].index
        km = KaplanMeierFitter().fit(
            ext.loc[idx, "survival_days"], ext.loc[idx, "event"]
        )
        obs = float(km.survival_function_at_times(365).iloc[0])
        # KM std hata (Greenwood) -- gozlenenin belirsizligi
        try:
            ci = km.confidence_interval_survival_function_
            ts = ci.index[ci.index <= 365]
            lo, hi = (ci.loc[ts[-1]].tolist() if len(ts) else (np.nan, np.nan))
        except Exception:  # noqa: BLE001
            lo, hi = np.nan, np.nan
        cal_rows.append(
            {"decile": d + 1, "n": len(idx), "olay": int(ext.loc[idx, "event"].sum()),
             "tahmin_S365_ort": round(float(pred365.loc[idx].mean()), 4),
             "gozlenen_S365_KM": round(obs, 4),
             "KM_alt": round(float(lo), 4), "KM_ust": round(float(hi), 4)}
        )
    cal_table = pd.DataFrame(cal_rows)
    cal_table.to_csv(OUT_DIR / "week3_v1_calibration_365.csv", index=False)
    print("\nDECILE KALIBRASYONU (t=365 gun; decile 1 = EN DUSUK tahmini sagkalim):")
    print(cal_table.to_string(index=False))

    mae = float(np.mean(np.abs(cal_table["tahmin_S365_ort"] - cal_table["gozlenen_S365_KM"])))
    print(f"\nDecile ortalama mutlak kalibrasyon farki (t=365): {mae:.4f}")
    print("NOT: IPCW sansur dagilimi UCSF'in kendisinden; yalniz v1 (baseline "
          "hazard yalniz onun artefaktinda). Diger varyantlarin kalibrasyonu "
          "OLCULMEMISTIR -- rapor beyani zorunlu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
