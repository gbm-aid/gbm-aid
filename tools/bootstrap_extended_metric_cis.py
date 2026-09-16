# -*- coding: utf-8 -*-
"""Genisletilmis harici metrikler (Uno C, td-AUC) icin BOOTSTRAP %95 CI.

NEDEN (2026-08-19, gunluk rapor "Riskler #3"): `evaluate_external_
metrics.py` nokta tahmini uretiyor; CLAUDE.md kural 3 "nokta tahmini +
%95 bootstrap CI zorunlu, tek-esik dili YASAK" diyor. AUC(365)=0,7515
gibi degerler CI'siz dolasima girerse "0,75 hedefini gectik" cumlesine
davetiye cikarir. Bu script 7 kol icin (6 varyant + k14 taban cizgisi)
hasta-seviyesi bootstrap (B=1000, percentile %2,5-%97,5) CI uretir.

Risk skorlari `evaluate_external_metrics.build_risk_scores()` ile AYNI
yoldan yeniden uretilir (kimlik baseline'i + Harrell bit-esitlik kapisi
DAHIL -- kapi gecmeyen kol tabloya girmez). k14 icin ayni desen elle
kurulur (o VARIANTS sozlugunde yok).

EGITIM HAVUZU KAPISI (2026-09-14 EKLENDI, Baris'in 5 nolu karari):
alti varyant kapiyi `eem.build_risk_scores()` uzerinden DEVRALIR;
k14 kolu burada ELLE kuruldugu icin `k14_risk()` kendi cagrisini
yapar (SERT 611/585, WT). Fail-closed.

Cikti: `week3_external_extended_metrics_ci.csv` (yalniz YENI dosya).
Basarisiz bootstrap ornekleri (tau oncesi olay yok vb.) SAYILIR ve
`n_valid_*` kolonlarinda raporlanir -- sessiz atlama yok.
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
import train_cox_week3 as w3  # noqa: E402
from pipeline.cox_model import build_training_frame, standardize_columns_fold_safe  # noqa: E402
from external_identity_baseline import compute_full_identity, verify_or_fail  # noqa: E402
from db_connection import get_connection  # noqa: E402
from psycopg2.extras import RealDictCursor  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc  # noqa: E402
from sksurv.util import Surv  # noqa: E402

OUT_DIR = eem.OUT_DIR
B = 1000
SEED = 42
TIMES = [365, 548, 730]


def load_data():
    conn = get_connection(readonly=True)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    data = {
        "upenn_long": eem._load_cached("upenn", w3.SEGMENTATION_TOOL_UPENN_C32, cur),
        "ucsf_long": eem._load_cached("ucsf", w3.SEGMENTATION_TOOL_UCSF_C32, cur),
        "upenn_pat": w3.fetch_patients_frame(cur, source_name=w3.UPENN_SOURCE_NAME),
        "ucsf_pat": w3.fetch_patients_frame(cur, source_name=w3.UCSF_SOURCE_NAME),
    }
    cur.close(); conn.close()
    data["knots"] = w3.compute_rcs_knots(
        data["upenn_pat"][w3.PATIENT_RAW_AGE_COLUMN].astype(float)
    )
    # Kimlik kapisi (sema v2)
    uc_wide = w3.pivot_ucsf_regions_long_to_wide(data["ucsf_long"], regions=("WT",))[0]
    up_wide = w3.pivot_radiomics_long_to_wide(data["upenn_long"], regions=["WT"])[0]
    verify_or_fail(compute_full_identity(uc_wide, data["ucsf_pat"], up_wide, data["upenn_pat"]))
    data["_wides"] = (uc_wide, up_wide)
    return data


def k14_risk(data):
    """k14 taban cizgisi icin risk skorlari (run_k14 scriptiyle ayni yol)."""
    uc_wide, up_wide = data["_wides"]
    extra = [w3.CLINICAL_AGE_COLUMN, w3.CLINICAL_GENDER_MALE_COLUMN]
    up_c = data["upenn_pat"].join(w3.build_clinical_covariate_frame(
        data["upenn_pat"], include_gender=True, include_gtr=False, include_idh=False))
    uc_c = data["ucsf_pat"].join(w3.build_clinical_covariate_frame(
        data["ucsf_pat"], include_gender=True, include_gtr=False, include_idh=False))
    train, train_report = build_training_frame(up_wide[[]], up_c,
                                               # 2026-09-14 ComBat OZDESLIK GUARD'I
                                               # (modeling-agent-T1, Baris onayi):
                                               # `False` -> `True`. Bu script rapora
                                               # giren bootstrap CI'lerini uretiyor;
                                               # docstring'deki `False` istisnasi
                                               # (mekanik smoke-test) burayi KAPSAMAZ.
                                               # OLCULDU (canli DB, oncesi/sonrasi):
                                               # 611/585, dusen hasta 0, BIREBIR AYNI.
                                               check_combat_identity=True,
                                               passthrough_columns=extra)
    # 2026-09-14 EGITIM HAVUZU KAPISI (modeling-agent-S2, Baris'in 5 nolu
    # karari). k14 kolu bu scriptte ELLE kuruluyor (eem.VARIANTS'ta yok) --
    # `run_k14_clinical_base_ucsf.py:113`'teki kapi burada YOKTU. `train`
    # asagida standardizasyon istatistigini (yas ort/std) veriyor; havuz
    # sessizce kayarsa k14 risk skorlari da kayar. k14 = WT kohortu (sifir
    # radyomik kolon) -> SERT 611/585, beyan edilmis azalma YOK.
    w3.check_training_pool_counts(
        train_report.n_output_rows,
        int(train["event"].sum()),
        expected_patients=w3.EXPECTED_UPENN_TRAINING_PATIENTS,
        expected_events=w3.EXPECTED_UPENN_TRAINING_EVENTS,
        allow_mismatch=False,
        regions=("WT",),
        dropped_patient_ids=w3.all_dropped_patient_ids(train_report),
    )
    ext, _ = w3.build_external_test_frame(uc_wide[[]], uc_c, passthrough_columns=extra)
    _, ext_s = standardize_columns_fold_safe(train, ext, [w3.CLINICAL_AGE_COLUMN])
    coef = pd.read_csv(OUT_DIR / "week3_k14_clinical_base_ucsf_final_coefficients.csv"
                       ).set_index("covariate")["coef"]
    assert set(coef.index) == set(extra)
    risk = ext_s[list(coef.index)].to_numpy(float) @ coef.to_numpy(float)
    return ext_s, pd.Series(risk, index=ext_s.index)


def bootstrap_cis(T, E, risk, rng):
    n = len(T)
    stats = {f"uno_c_tau{t}": [] for t in TIMES}
    stats.update({f"td_auc_{t}": [] for t in TIMES})
    fails = 0
    for _ in range(B):
        idx = rng.integers(0, n, n)
        Tb, Eb, rb = T[idx], E[idx], risk[idx]
        if Eb.sum() == 0:
            fails += 1
            continue
        try:
            surv = Surv.from_arrays(event=Eb.astype(bool), time=Tb)
            for t in TIMES:
                c, *_ = concordance_index_ipcw(surv, surv, rb, tau=float(t))
                stats[f"uno_c_tau{t}"].append(float(c))
            aucs, _ = cumulative_dynamic_auc(surv, surv, rb, times=TIMES)
            for t, a in zip(TIMES, aucs):
                stats[f"td_auc_{t}"].append(float(a))
        except Exception:  # noqa: BLE001 -- ornek-bazli, SAYILIYOR
            fails += 1
    out = {}
    for k, vals in stats.items():
        arr = np.array(vals)
        arr = arr[np.isfinite(arr)]
        out[f"{k}_lo"] = round(float(np.percentile(arr, 2.5)), 4)
        out[f"{k}_hi"] = round(float(np.percentile(arr, 97.5)), 4)
        out[f"n_valid_{k}"] = int(len(arr))
    out["n_failed_resamples"] = fails
    return out


def main() -> int:
    data = load_data()
    rng = np.random.default_rng(SEED)
    reported = pd.read_csv(OUT_DIR / "week3_external_extended_metrics.csv"
                           ).set_index("variant")

    rows = []
    jobs = list(eem.VARIANTS.items()) + [("k14_clinical_base_ucsf", None)]
    for name, cfg in jobs:
        if cfg is None:
            ext_s, risk = k14_risk(data)
        else:
            ext_s, risk = eem.build_risk_scores(name, cfg, data)
        T = ext_s["survival_days"].to_numpy(float)
        E = ext_s["event"].to_numpy(int)
        # Harrell bit-esitlik kapisi (rapor edilen degere karsi)
        harrell = concordance_index(T, -risk.to_numpy(), E)
        rep_c = float(reported.loc[name, "reported_harrell"])
        if abs(harrell - rep_c) > 1e-6:
            rows.append({"variant": name, "gate": f"BASARISIZ ({harrell:.6f} vs {rep_c:.6f})"})
            continue
        cis = bootstrap_cis(T, E, risk.to_numpy(), rng)
        # Nokta tahminleriyle birlestir (okunabilirlik)
        point = reported.loc[name].to_dict()
        row = {"variant": name, "gate": "GECTI"}
        for t in TIMES:
            row[f"uno_c_tau{t}"] = point.get(f"uno_c_tau{t}")
            row[f"uno_c_tau{t}_ci"] = f"[{cis[f'uno_c_tau{t}_lo']}, {cis[f'uno_c_tau{t}_hi']}]"
            row[f"td_auc_{t}"] = point.get(f"td_auc_{t}")
            row[f"td_auc_{t}_ci"] = f"[{cis[f'td_auc_{t}_lo']}, {cis[f'td_auc_{t}_hi']}]"
        row["n_failed_resamples"] = cis["n_failed_resamples"]
        row["n_valid_min"] = min(v for k, v in cis.items() if k.startswith("n_valid_"))
        rows.append(row)
        print(f"[OK] {name}: AUC365={row['td_auc_365']} {row['td_auc_365_ci']}  "
              f"fails={row['n_failed_resamples']}")

    out = pd.DataFrame(rows)
    target = OUT_DIR / "week3_external_extended_metrics_ci.csv"
    out.to_csv(target, index=False)
    print(f"\n[OK] B={B} percentile CI -> {target}")
    print(out.to_string(index=False))
    failed = out.loc[out["gate"] != "GECTI", "variant"].tolist()
    if failed:
        print(f"[DIKKAT] Kapi gecmeyen kollar (CI uretilmedi): {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
