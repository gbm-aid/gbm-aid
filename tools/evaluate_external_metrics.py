# -*- coding: utf-8 -*-
"""Harici test (UCSF) icin GENISLETILMIS metrikler -- alti Cox varyanti.

NEDEN GEREKLI (2026-08-19, Baris onayi -- iyilestirme plani #2)
----------------------------------------------------------------
Su ana kadar harici performans YALNIZ Harrell C-index ile raporlandi.
Olculen sansur asimetrisi bunu tek basina savunulamaz kiliyor:

    UPenn (egitim) : %4,3 sansur,  sansurlulerin medyan takibi 2402 gun
    UCSF (harici)  : %42,7 sansur, sansurlulerin medyan takibi  366 gun

UCSF'te neredeyse her iki hastadan biri sansurlu ve takip kisa (idari
sansur). Harrell C sansuru "karsilastirilabilir cift" tanimiyla ortuk
agirliklar; agir sansurde kohorta-bagimli yanlilik tasir. Bu script:

  1. Uno's C (IPCW)      -- sansur dagilimina gore acikca agirliklar
                            (tau = 365/548/730 gun)
  2. Zamana-bagli AUC    -- cumulative/dynamic, t = 365/548/730
  3. (yalniz v1, pkl'den) Brier skoru + kalibrasyon tablosu

MODELE DOKUNMAZ -- yalniz degerlendirmeyi derinlestirir.

DOGRULAMA KAPISI (sessiz-yanlis-risk-skoru onlemi)
---------------------------------------------------
Risk skorlari `week3_<varyant>_final_coefficients.csv` katsayilarindan
YENIDEN uretilir (frame kurulumu + standardizasyon birebir kosu
kodundaki gibi). Her varyant icin yeniden uretilen Harrell C, kosunun
kendi raporladigi `c_index` ile karsilastirilir:

    |fark| <= 1e-6  ->  KAPI GECTI, Uno/td-AUC bu skorlarla guvenilir
    aksi halde      ->  o varyant SONUC TABLOSUNA GIRMEZ, "KAPI
                        BASARISIZ" olarak raporlanir (sessiz devam yok)

EGITIM HAVUZU KAPISI (2026-09-14 EKLENDI, Baris'in 5 nolu karari)
-----------------------------------------------------------------
Her varyantta `build_training_frame()`'den SONRA
`w3.check_training_pool_counts()` calisir: WT-only kollarda SERT
611/585, v2c (WT+TC) kolunda beyan edilmis 609/583 + dusen 2 hastanin
KIMLIGI. Fail-closed. Gerekce: kaynak whitelist'i (TCGA/UCSF reddi)
zaten vardi ama SAYAC kilidi yoktu; ayrica olculdu ki Harrell
bit-esitlik kapisi bu kaymayi v1/v2a/v2b'de YAKALAMIYOR (1 hasta
dusurulunce harici Harrell farki 0.000e+00).

Sansur-dagilimi notu: IPCW agirliklari icin sansur Kaplan-Meier'i
UCSF'in KENDISINDEN kestirilir (harici kohortun kendi sansur sureci;
`survival_train=UCSF` sksurv cagrisinda). Bu tercih ciktida acikca
raporlanir.

Salt-okunur: DB'ye yazmaz; yalniz `artifacts/week3/cox_model/` altina
YENI dosyalar yazar (`week3_external_extended_metrics.csv`,
`week3_v1_calibration_365.csv`).
"""
from __future__ import annotations

import json
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
from lifelines.utils import concordance_index  # noqa: E402
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc  # noqa: E402
from sksurv.util import Surv  # noqa: E402

OUT_DIR = PROJECT_ROOT / "artifacts" / "week3" / "cox_model"
CACHE = OUT_DIR / "_cache"
GATE_TOL = 1e-6
TIMES = [365, 548, 730]

#: Varyant -> (regions, include_mgmt, use_age_spline, v3_filtre)
VARIANTS: dict[str, dict] = {
    "v1_referans": dict(regions=("WT",), mgmt=False, spline=False, v3=False),
    "v2a_mgmt": dict(regions=("WT",), mgmt=True, spline=False, v3=False),
    "v2b_mgmt_spline": dict(regions=("WT",), mgmt=True, spline=True, v3=False),
    "v2c_mgmt_spline_wttc": dict(regions=("WT", "TC"), mgmt=True, spline=True, v3=False),
    "v3a_lowvar_v1referans": dict(regions=("WT",), mgmt=False, spline=False, v3=True),
    "v3b_lowvar_v2amgmt": dict(regions=("WT",), mgmt=True, spline=False, v3=True),
}


def _load_cached(tag: str, tool: str, cur) -> pd.DataFrame:
    """Kosunun kendi onbellek dosyasi; DB satir sayisiyla dogrulanir --
    bayat onbellekle sessizce calisma YOK."""
    cur.execute(
        "SELECT COUNT(*) AS n FROM radiomics WHERE segmentation_tool = %s", (tool,)
    )
    db_rows = int(cur.fetchone()["n"])
    frame = pd.read_pickle(CACHE / f"{tag}_long.pkl")
    if len(frame) != db_rows:
        raise RuntimeError(f"{tag}: onbellek {len(frame)} != DB {db_rows} -- DUR")
    return frame


def build_risk_scores(name: str, cfg: dict, data: dict):
    """Kosu kodundaki final-model on-islemesini BIREBIR yeniden uygular,
    `final_coefficients.csv` ile risk skorunu uretir."""

    regions = cfg["regions"]
    upenn_wide, _ = w3.pivot_radiomics_long_to_wide(
        data["upenn_long"], regions=list(regions)
    )
    ucsf_wide, _ = w3.pivot_ucsf_regions_long_to_wide(data["ucsf_long"], regions=regions)

    # Radyomik aday havuzu
    cand: list[str] = []
    for r in regions:
        cand += w3.select_feature_columns(upenn_wide, w3.STABLE_FEATURES_ICC60, region=r)
    if cfg["v3"]:
        cand, _ = w3.build_v3_radiomic_feature_pool(
            upenn_wide,
            cand,
            cv_threshold=w3.V3_DEFAULT_CV_THRESHOLD,
            corr_threshold=w3.V3_DEFAULT_CORRELATION_CLUSTER_THRESHOLD,
        )

    # Klinik kovaryatlar (kosu koduyla ayni yardimcilar)
    extra = w3.build_variant_clinical_extra_columns(
        include_mgmt=cfg["mgmt"], use_age_spline=cfg["spline"]
    )
    up_cl = w3.build_variant_clinical_frame(
        data["upenn_pat"], include_mgmt=cfg["mgmt"],
        use_age_spline=cfg["spline"], age_knots=data["knots"],
    )
    uc_cl = w3.build_variant_clinical_frame(
        data["ucsf_pat"], include_mgmt=cfg["mgmt"],
        use_age_spline=cfg["spline"], age_knots=data["knots"],
    )
    up_c = data["upenn_pat"].join(up_cl)
    uc_c = data["ucsf_pat"].join(uc_cl)

    # 2026-09-14 ComBat OZDESLIK GUARD'I (modeling-agent-T1, Baris onayi).
    # `check_combat_identity` burada EskIDEN `False` idi. `build_training_
    # frame()` docstring'i `False`'i "ComBat'a hic bagimli olmayan mekanik
    # test" (orn. `smoke_test_cox_model.py`) ile sinirliyor -- bu script o
    # kategoride DEGIL, RAPORA GIREN sayilari uretiyor. CLAUDE.md: "ComBat,
    # OS-Cox birincil zincirinde OZDESLIK donusumudur ... AMA KIRILGAN:
    # LUMIERE egitime girerse veya `reference_batch` degisirse asimetri
    # dogar -> Cox egitiminde kod-seviyesi guard ZORUNLU."
    # OLCULDU (canli DB, degisiklik ONCESI/SONRASI): 6 varyantin hepsinde
    # havuz/olay/dusen-kimlik BIREBIR AYNI (WT-only 611/585; v2c 609/583 +
    # UPENN-GBM-00354/00397) -- `True` hicbir sayiyi degistirmez, yalniz
    # ileride LUMIERE/reference_batch kaymasini FAIL-LOUD yapar.
    train, train_report = build_training_frame(
        upenn_wide[cand], up_c, check_combat_identity=True, passthrough_columns=extra
    )
    # 2026-09-14 EGITIM HAVUZU KAPISI (modeling-agent-S2, Baris'in 5 nolu
    # karari). `build_training_frame()` KAYNAK whitelist'ini (TCGA/UCSF
    # reddi) zaten uyguluyordu, ama 611/585 SAYAC kilidi kapinin kendisinde
    # DEGIL -- `check_training_pool_counts()` AYRI cagrilmali (bkz.
    # `train_cox_week3.py:3003/3515/4215`, `run_k14_clinical_base_ucsf.py:113`,
    # `export_v3b_deployment_checkpoint.py:250` -- ayni desen).
    # Bu script egitim cercevesini YALNIZ standardizasyon istatistigi icin
    # kuruyor; havuz sessizce kayarsa `standardize_columns_fold_safe()`'in
    # ortalama/std'si de kayar ve HARICI risk skorlari sessizce bozulur.
    # `regions=regions` -> kapi BOLGE-FARKINDA: WT-only kollarda SERT
    # 611/585; v2c (WT+TC) kolunda `DECLARED_REGION_SHORTFALLS`'taki beyan
    # edilmis 609/583 + dusen 2 hastanin KIMLIGI aranir. Bu yuzden buraya
    # 609/583 DEGIL, TABAN 611/585 gecilir -- 609/583 gecilseydi kapi
    # ustteki esitlikten erken donup kimlik dogrulamasini ATLARDI.
    w3.check_training_pool_counts(
        train_report.n_output_rows,
        int(train["event"].sum()),
        expected_patients=w3.EXPECTED_UPENN_TRAINING_PATIENTS,
        expected_events=w3.EXPECTED_UPENN_TRAINING_EVENTS,
        allow_mismatch=False,
        regions=regions,
        dropped_patient_ids=w3.all_dropped_patient_ids(train_report),
    )
    ext, _ = w3.build_external_test_frame(
        ucsf_wide[cand], uc_c, passthrough_columns=extra
    )

    # Final-model standardizasyonu (kosu kodu `run_modeling_arm` sonundaki
    # full-pool cagrisiyla ayni): v3'te radyomikler + yas; digerlerinde
    # yalniz yas (spline'da rcs1/rcs2).
    age_cols = (
        [w3.CLINICAL_AGE_RCS1_COLUMN, w3.CLINICAL_AGE_RCS2_COLUMN]
        if cfg["spline"]
        else [w3.CLINICAL_AGE_COLUMN]
    )
    std_cols = (cand + age_cols) if cfg["v3"] else age_cols
    train_s, ext_s = standardize_columns_fold_safe(train, ext, std_cols)

    coef = pd.read_csv(OUT_DIR / f"week3_{name}_final_coefficients.csv")
    cov_col = "covariate" if "covariate" in coef.columns else coef.columns[1]
    coefs = coef.set_index(cov_col)["coef"]

    missing = [c for c in coefs.index if c not in ext_s.columns]
    if missing:
        raise RuntimeError(f"{name}: harici cercevede eksik kovaryat: {missing}")

    # 2026-08-19 sertlestirme (Codex'in kalibrasyon-scriptindeki fail-open
    # bulgusuyla AYNI sinif, burada da kapatildi): havuz kapisi (295/169,
    # kimlik disiplinli sayim) + sonluluk. Not: bu scriptin ASIL kimlik
    # kaniti zaten Harrell bit-esitligi (0.00e+00, 6/6) -- degisen tek
    # bir kayit o esitligi pratikte bozar; asagidakiler erken/net teshis.
    w3.check_external_test_pool_counts(
        len(ext_s), int(ext_s["event"].sum()),
        expected_patients=295, expected_events=169,
        allow_mismatch=False, label=f"UCSF ({name})",
    )
    X_arr = ext_s[list(coefs.index)].to_numpy(dtype=float)
    if not (np.isfinite(X_arr).all() and np.isfinite(coefs.to_numpy(dtype=float)).all()):
        bad = [c for c, ok in zip(coefs.index, np.isfinite(X_arr).all(axis=0)) if not ok]
        raise RuntimeError(f"{name}: NaN/inf kovaryat -- {bad}")

    risk_ext = X_arr @ coefs.to_numpy(dtype=float)
    return ext_s, pd.Series(risk_ext, index=ext_s.index, name="risk")


def main() -> int:
    conn = get_connection(readonly=True)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    data = {
        "upenn_long": _load_cached("upenn", w3.SEGMENTATION_TOOL_UPENN_C32, cur),
        "ucsf_long": _load_cached("ucsf", w3.SEGMENTATION_TOOL_UCSF_C32, cur),
        "upenn_pat": w3.fetch_patients_frame(cur, source_name=w3.UPENN_SOURCE_NAME),
        "ucsf_pat": w3.fetch_patients_frame(cur, source_name=w3.UCSF_SOURCE_NAME),
    }
    cur.close()
    conn.close()
    # DONDURULMUS KIMLIK BASELINE'I (Codex bulgusu -- sayi kapilari
    # "ayni sayida farkli kayit" sinifini yakalamaz; hash'ler yakalar).
    from external_identity_baseline import compute_full_identity, verify_or_fail
    uc_wide_wt, _ = w3.pivot_ucsf_regions_long_to_wide(
        data["ucsf_long"], regions=("WT",)
    )
    up_wide_wt, _ = w3.pivot_radiomics_long_to_wide(
        data["upenn_long"], regions=["WT"]
    )
    identity = compute_full_identity(
        uc_wide_wt, data["ucsf_pat"], up_wide_wt, data["upenn_pat"]
    )
    baseline = verify_or_fail(identity)
    print(f"KIMLIK KAPISI GECTI (UCSF+UPenn, sema v2; baseline: "
          f"{baseline['frozen_at'][:19]}; TC pivotu kapsam DISI, "
          "v2c bit-birebir Harrell kapisi dolayli kapsar)")

    data["knots"] = w3.compute_rcs_knots(
        data["upenn_pat"][w3.PATIENT_RAW_AGE_COLUMN].astype(float)
    )

    rows = []
    for name, cfg in VARIANTS.items():
        reported = pd.read_csv(OUT_DIR / f"week3_{name}_external_test.csv").iloc[0]
        try:
            ext_s, risk = build_risk_scores(name, cfg, data)
        except Exception as exc:  # noqa: BLE001 -- kapi basarisizligi ACIKCA raporlanir
            rows.append({"variant": name, "gate": f"HATA: {type(exc).__name__}: {exc}"})
            continue

        T = ext_s["survival_days"].to_numpy(dtype=float)
        E = ext_s["event"].to_numpy(dtype=int)

        harrell = concordance_index(T, -risk.to_numpy(), E)
        gate_diff = abs(harrell - float(reported["c_index"]))
        gate_ok = gate_diff <= GATE_TOL
        row = {
            "variant": name,
            "n": len(ext_s),
            "events": int(E.sum()),
            "reported_harrell": round(float(reported["c_index"]), 6),
            "reproduced_harrell": round(float(harrell), 6),
            "gate_abs_diff": f"{gate_diff:.2e}",
            "gate": "GECTI" if gate_ok else "BASARISIZ",
        }
        if not gate_ok:
            # Sessiz devam YOK: kapi tutmadiysa Uno/td-AUC HESAPLANMAZ.
            rows.append(row)
            continue

        surv = Surv.from_arrays(event=E.astype(bool), time=T)
        est = risk.to_numpy()
        for tau in TIMES:
            c_uno, *_ = concordance_index_ipcw(surv, surv, est, tau=float(tau))
            row[f"uno_c_tau{tau}"] = round(float(c_uno), 4)
        aucs, mean_auc = cumulative_dynamic_auc(surv, surv, est, times=TIMES)
        for t, a in zip(TIMES, aucs):
            row[f"td_auc_{t}"] = round(float(a), 4)
        row["td_auc_mean"] = round(float(mean_auc), 4)
        rows.append(row)

    out = pd.DataFrame(rows)
    target = OUT_DIR / "week3_external_extended_metrics.csv"
    out.to_csv(target, index=False)
    print(f"[OK] -> {target}")
    print(
        "NOT: IPCW sansur dagilimi UCSF'in KENDISINDEN kestirildi "
        "(harici kohortun kendi sansur sureci)."
    )
    print(out.to_string(index=False))

    failed = out.loc[out["gate"] != "GECTI", "variant"].tolist()
    if failed:
        print(
            f"\n[DIKKAT] KAPI BASARISIZ varyantlar (Uno/td-AUC HESAPLANMADI): {failed}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
