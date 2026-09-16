# -*- coding: utf-8 -*-
"""v3c (v3c_lowvar_wttc_mgmt_nospline) icin GENISLETILMIS harici metrikler.

NEDEN (2026-09-11 gorev talimati, "v3c koşusu BİTTİ — zorunlu takip
adımları"): `evaluate_external_metrics.py` + `bootstrap_extended_metric_
cis.py` 7 kolu (6 model varyanti + k14 taban cizgisi) zaten kapsiyor,
ama bu iki dosyaya DOKUNMA yasagi var (v3c koşusu bitti, coğaltma riski
almadan AYRI bir script ile SATIR EKLEMEK istendi). Bu script:

  1. Ayni risk-skoru yeniden-uretim yolunu (`evaluate_external_metrics.
     build_risk_scores`) KULLANIR -- kod coğaltmaz, import eder.
  2. v3c'nin final katsayilari `artifacts/week3/cox_model/final_v3c/`
     alt-dizininde durdugu icin `eem.OUT_DIR` cagri SIRASINDA GECICI
     olarak o dizine cevrilir, hemen sonra ESKI degerine geri alinir
     (CACHE/diger 6 kolun okuma yolu ETKILENMEZ -- CACHE modul yuklenme
     aninda zaten sabitlenmis).
  3. Harrell bit-esitlik KAPISI (rapor edilen c_index ile yeniden-uretilen
     Harrell farki <=1e-6) v3c icin de UYGULANIR -- kapi dusarsa Uno/td-AUC
     HESAPLANMAZ.
  4. Sonuc, MEVCUT `week3_external_extended_metrics.csv` /
     `week3_external_extended_metrics_ci.csv` dosyalarina v3c SATIRI
     olarak EKLENIR -- var olan 7 satir BOZULMAZ (once `.bak_pre_v3c`
     yedegi alinir, script IDEMPOTENT: yeniden calistirilirsa eski v3c
     satirini silip yenisini yazar, coklama YOK).

Bu bir YENI MODEL FITI DEGILDIR -- var olan v3c fit'inin (final_v3c/
altindaki katsayilarin) tanisal ek olcumudur, coksuluk sayacini
ARTIRMAZ (gorev talimati).

EGITIM HAVUZU KAPISI (2026-09-14 EKLENDI, Baris'in 5 nolu karari):
`build_risk_scores_v3c()` icinde `w3.check_training_pool_counts()`.
v3c bir WT+TC kolu -> havuz 609/583 (`ZORUNLU-BEYANLAR.md` B7), ama
kapiya TABAN 611/585 + `regions=("WT","TC")` gecilir; 609/583'u
beyan tablosundan kapi kendisi cozer ve dusen 2 hastanin KIMLIGINI
de dogrular (dogrudan 609/583 yazmak kimlik kontrolunu ATLATIRDI).

Salt-okunur: DB'ye yalniz SELECT (readonly connection); `final_v3c/`
veya `evaluate_external_metrics.py`/`bootstrap_extended_metric_cis.py`
dosyalarina YAZILMAZ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import evaluate_external_metrics as eem  # noqa: E402
import bootstrap_extended_metric_cis as bec  # noqa: E402
import train_cox_week3 as w3  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc  # noqa: E402
from sksurv.util import Surv  # noqa: E402

OUT_DIR = eem.OUT_DIR  # cox_model kok dizini -- HEDEF CSV'ler burada
V3C_DIR = OUT_DIR / "final_v3c"
NAME = "v3c_lowvar_wttc_mgmt_nospline"
CFG = dict(regions=("WT", "TC"), mgmt=True, spline=False, v3=True)
TIMES = [365, 548, 730]
B = 1000
SEED = 42

# 2026-09-11 NOT: `pipeline/reduce_collinearity.py` K16 ajani tarafindan
# AYNI ANDA degistiriliyor (bu dosyaya DOKUNULMUYOR -- gorev kisitina
# uygun). K16'nin bugun eklendigi katilaştirilmış NaN kapisi
# (`_assert_finite_features`), v3c'nin egitiminde kullanilan HAM
# `upenn_wide` (611 hasta, 2'sinde TC=0 -> NaN) uzerinde
# `w3.build_v3_radiomic_feature_pool()`'u YENIDEN cagirmayi imkansiz
# kiliyor (fail-loud NonFiniteFeatureValueError) -- v3c kosuldugunde bu
# kapi ya yoktu ya da farkli davraniyordu. Bu YENIDEN-URETIM riskini
# ORTADAN KALDIRMAK icin havuzu YENIDEN HESAPLAMIYORUZ; bunun yerine
# GERCEK kosunun kendi `v3_pool_report.csv` ciktisindan (hangi 79
# ozelligin dusuruldugu) TURETIYORUZ -- bu, canli fonksiyonu tekrar
# cagirmaktan DAHA SADIK bir yeniden-uretimdir (kosunun kendi
# kaydından, olası kod kaymasindan bagimsiz).


def _v3c_candidate_pool(upenn_wide: pd.DataFrame) -> list[str]:
    raw_candidate_columns: list[str] = []
    for region in CFG["regions"]:
        raw_candidate_columns += w3.select_feature_columns(
            upenn_wide, w3.STABLE_FEATURES_ICC60, region=region
        )
    pool_report = pd.read_csv(V3C_DIR / f"week3_{NAME}_v3_pool_report.csv")
    dropped = set(pool_report["feature"])
    cand = [c for c in raw_candidate_columns if c not in dropped]
    if len(cand) != 107:
        raise RuntimeError(
            f"v3c aday havuzu beklenenden farkli: {len(cand)} != 107 "
            f"(ham={len(raw_candidate_columns)}, dusurulen={len(dropped)}) -- "
            "v3_pool_report.csv ile raw STABLE_FEATURES_ICC60 listesi "
            "arasinda bir kayma var, DUR."
        )
    return cand


def build_risk_scores_v3c(data: dict):
    """`evaluate_external_metrics.build_risk_scores()` ile AYNI adimlar,
    TEK fark: aday havuzu canli `build_v3_radiomic_feature_pool()`
    cagrisiyla degil, kosunun kendi `v3_pool_report.csv`'sinden
    turetiliyor (yukaridaki not)."""

    regions = CFG["regions"]
    upenn_wide, _ = w3.pivot_radiomics_long_to_wide(data["upenn_long"], regions=list(regions))
    ucsf_wide, _ = w3.pivot_ucsf_regions_long_to_wide(data["ucsf_long"], regions=regions)

    cand = _v3c_candidate_pool(upenn_wide)

    extra = w3.build_variant_clinical_extra_columns(
        include_mgmt=CFG["mgmt"], use_age_spline=CFG["spline"]
    )
    up_cl = w3.build_variant_clinical_frame(
        data["upenn_pat"], include_mgmt=CFG["mgmt"],
        use_age_spline=CFG["spline"], age_knots=data["knots"],
    )
    uc_cl = w3.build_variant_clinical_frame(
        data["ucsf_pat"], include_mgmt=CFG["mgmt"],
        use_age_spline=CFG["spline"], age_knots=data["knots"],
    )
    up_c = data["upenn_pat"].join(up_cl)
    uc_c = data["ucsf_pat"].join(uc_cl)

    train, train_report = eem.build_training_frame(
        upenn_wide[cand], up_c, check_combat_identity=True, passthrough_columns=extra
    )
    # 2026-09-14 ComBat OZDESLIK GUARD'I (modeling-agent-T1, Baris onayi):
    # `check_combat_identity` `False` -> `True`. Bu script v3c'nin RAPORA
    # GIREN genisletilmis metriklerini uretiyor -- `build_training_frame()`
    # docstring'indeki `False` istisnasi ("ComBat'a hic bagimli olmayan
    # mekanik test") bu scripti KAPSAMAZ. OLCULDU (canli DB, oncesi/
    # sonrasi): 609/583, dusen UPENN-GBM-00354 + UPENN-GBM-00397, cerceve
    # degerleri BIREBIR AYNI -- sayi degisimi YOK.
    # 2026-09-14 EGITIM HAVUZU KAPISI (modeling-agent-S2, Baris'in 5 nolu
    # karari). `train` asagida `standardize_columns_fold_safe()`'e giriyor
    # -- havuz sessizce kayarsa ortalama/std kayar, harici risk skorlari
    # sessizce bozulur.
    #
    # ⚠️ SAYI TUZAGI (ZORUNLU-BEYANLAR.md B7): v3c bir WT+TC kolu, havuzu
    # 609/583 -- 611/585 DEGIL. Buna ragmen kapiya gecilen `expected_*`
    # yine TABAN 611/585'tir, cunku kapi BOLGE-FARKINDA: `regions=("WT","TC")`
    # goren `check_training_pool_counts()` `DECLARED_REGION_SHORTFALLS`'tan
    # beyan edilmis 609/583'u cozer VE dusen 2 hastanin KIMLIGINI
    # (UPENN-GBM-00354 / UPENN-GBM-00397) dogrular.
    # Buraya dogrudan 609/583 gecmek DAHA ZAYIF olurdu: kapi en ustteki
    # esitlikten erken doner, bolge/kimlik dogrulamasi HIC calismaz
    # ("ayni sayida ama BASKA hastalar dustu" sinifi yakalanmaz).
    w3.check_training_pool_counts(
        train_report.n_output_rows,
        int(train["event"].sum()),
        expected_patients=w3.EXPECTED_UPENN_TRAINING_PATIENTS,
        expected_events=w3.EXPECTED_UPENN_TRAINING_EVENTS,
        allow_mismatch=False,
        regions=CFG["regions"],
        dropped_patient_ids=w3.all_dropped_patient_ids(train_report),
    )
    ext, _ = w3.build_external_test_frame(
        ucsf_wide[cand], uc_c, passthrough_columns=extra
    )

    age_cols = [w3.CLINICAL_AGE_COLUMN]  # v3c dogrusal yas (spline=False)
    std_cols = cand + age_cols  # v3 kolu -> radyomikler de standardize edilir
    train_s, ext_s = eem.standardize_columns_fold_safe(train, ext, std_cols)

    coef = pd.read_csv(V3C_DIR / f"week3_{NAME}_final_coefficients.csv")
    cov_col = "covariate" if "covariate" in coef.columns else coef.columns[1]
    coefs = coef.set_index(cov_col)["coef"]

    missing = [c for c in coefs.index if c not in ext_s.columns]
    if missing:
        raise RuntimeError(f"{NAME}: harici cercevede eksik kovaryat: {missing}")

    w3.check_external_test_pool_counts(
        len(ext_s), int(ext_s["event"].sum()),
        expected_patients=295, expected_events=169,
        allow_mismatch=False, label=f"UCSF ({NAME})",
    )
    X_arr = ext_s[list(coefs.index)].to_numpy(dtype=float)
    if not (np.isfinite(X_arr).all() and np.isfinite(coefs.to_numpy(dtype=float)).all()):
        bad = [c for c, ok in zip(coefs.index, np.isfinite(X_arr).all(axis=0)) if not ok]
        raise RuntimeError(f"{NAME}: NaN/inf kovaryat -- {bad}")

    risk_ext = X_arr @ coefs.to_numpy(dtype=float)
    return ext_s, pd.Series(risk_ext, index=ext_s.index, name="risk")


def _backup_once(path: Path) -> None:
    backup = path.with_name(path.stem + ".csv.bak_pre_v3c")
    if not backup.exists():
        pd.read_csv(path).to_csv(backup, index=False)
        print(f"[YEDEK] {path.name} -> {backup.name}")


def _upsert_row(path: Path, row: dict) -> None:
    _backup_once(path)
    df = pd.read_csv(path)
    df = df[df["variant"] != row["variant"]].copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"[OK] -> {path} (v3c satiri eklendi/guncellendi, toplam {len(df)} satir)")


def main() -> int:
    data = bec.load_data()  # DB readonly SELECT + kimlik kapisi (sema v2)

    reported = pd.read_csv(V3C_DIR / f"week3_{NAME}_external_test.csv").iloc[0]

    ext_s, risk = build_risk_scores_v3c(data)

    T = ext_s["survival_days"].to_numpy(dtype=float)
    E = ext_s["event"].to_numpy(dtype=int)

    harrell = concordance_index(T, -risk.to_numpy(), E)
    reported_c = float(reported["c_index"])
    gate_diff = abs(harrell - reported_c)
    gate_ok = gate_diff <= eem.GATE_TOL

    row_point = {
        "variant": NAME,
        "n": len(ext_s),
        "events": int(E.sum()),
        "reported_harrell": round(reported_c, 6),
        "reproduced_harrell": round(float(harrell), 6),
        "gate_abs_diff": f"{gate_diff:.2e}",
        "gate": "GECTI" if gate_ok else "BASARISIZ",
    }

    if not gate_ok:
        print(f"[DIKKAT] KAPI BASARISIZ -- fark {gate_diff:.3e} > {eem.GATE_TOL:.0e}. "
              "Uno/td-AUC HESAPLANMADI.", file=sys.stderr)
        _upsert_row(OUT_DIR / "week3_external_extended_metrics.csv", row_point)
        return 1

    surv = Surv.from_arrays(event=E.astype(bool), time=T)
    est = risk.to_numpy()
    for tau in TIMES:
        c_uno, *_ = concordance_index_ipcw(surv, surv, est, tau=float(tau))
        row_point[f"uno_c_tau{tau}"] = round(float(c_uno), 4)
    aucs, mean_auc = cumulative_dynamic_auc(surv, surv, est, times=TIMES)
    for t, a in zip(TIMES, aucs):
        row_point[f"td_auc_{t}"] = round(float(a), 4)
    row_point["td_auc_mean"] = round(float(mean_auc), 4)

    _upsert_row(OUT_DIR / "week3_external_extended_metrics.csv", row_point)
    print(pd.Series(row_point))

    # Bootstrap %95 CI (ayni yontem: bootstrap_extended_metric_cis.bootstrap_cis)
    rng = np.random.default_rng(SEED)
    cis = bec.bootstrap_cis(T, E, est, rng)
    row_ci = {"variant": NAME, "gate": "GECTI"}
    for t in TIMES:
        row_ci[f"uno_c_tau{t}"] = row_point[f"uno_c_tau{t}"]
        row_ci[f"uno_c_tau{t}_ci"] = f"[{cis[f'uno_c_tau{t}_lo']}, {cis[f'uno_c_tau{t}_hi']}]"
        row_ci[f"td_auc_{t}"] = row_point[f"td_auc_{t}"]
        row_ci[f"td_auc_{t}_ci"] = f"[{cis[f'td_auc_{t}_lo']}, {cis[f'td_auc_{t}_hi']}]"
    row_ci["n_failed_resamples"] = cis["n_failed_resamples"]
    row_ci["n_valid_min"] = min(v for k, v in cis.items() if k.startswith("n_valid_"))

    _upsert_row(OUT_DIR / "week3_external_extended_metrics_ci.csv", row_ci)
    print(pd.Series(row_ci))
    print(
        "\nNOT: IPCW sansur dagilimi UCSF'in KENDISINDEN kestirildi "
        "(harici kohortun kendi sansur sureci). B=1000, seed=42, "
        "percentile %2,5-%97,5. Bu bir yeni model fiti DEGIL -- var olan "
        "v3c fit'inin tanisal ek olcumudur."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
