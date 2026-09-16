# -*- coding: utf-8 -*-
"""v3b <-> v3c ESLESTIRILMIS FARK bootstrap'i (delta C-index, delta td-AUC(730)).

NEDEN (2026-09-11 gorev talimati + `MODEL-SECIM-ANALIZI-2026-09-11.md`
Sf4.3 zorunlu adimi): iki kolun HARICI CI'lari SEVIYE CI'sidir --
"v2c/v3c gec ufukta v3b'den iyi/kotu" cumlesi bunlarla KURULAMAZ
(ZORUNLU-BEYANLAR A8). Eslestirilmis fark bootstrap'i AYNI UCSF
hastalarini AYNI bootstrap indeksleriyle IKI modele de uygulayip
delta = metrik(v3c) - metrik(v3b) dagilimini uretir; delta'nin %95 CI'si
sifiri kapsiyorsa fark istatistiksel olarak SIFIRDAN AYIRT EDILEMEZ.

Kohort uyumu: v3b WT-only (295/169), v3c WT+TC (295/169) -- ikisi de
AYNI 295 UCSF hastasini kullaniyor (UCSF'te hicbir hastada TC bos
degil, bkz. katalog notu). Bu script ORTAK hasta kumesini index
hizasiyla DOGRULAR (`patient_id` join) -- farkli n olursa DUR.

Risk skorlari:
  - v3b: `evaluate_external_metrics.build_risk_scores()` (mevcut,
    dokunulmadi) ile Harrell bit-esitlik kapisindan GECEREK uretilir.
  - v3c: `evaluate_v3c_extended_metrics.build_risk_scores_v3c()` (bu
    oturumda yazilan, ayni bit-esitlik kapisi DAHIL) ile uretilir.

Metrikler: Harrell C (seviyesiz), Uno's C (tau=365/548/730), td-AUC
(t=365/548/730). Delta = v3c - v3b. B=1000, seed=42, ayni bootstrap
indeksleri (`rng.integers`) HER İKİ modele de uygulanir (paired).

Cikti: `artifacts/week3/cox_model/week3_paired_bootstrap_v3b_v3c.csv`
(YENI dosya, mevcut hicbir dosyaya YAZILMAZ).

Salt-okunur: DB'ye yalniz SELECT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import evaluate_external_metrics as eem  # noqa: E402
import evaluate_v3c_extended_metrics as ev3c  # noqa: E402
import bootstrap_extended_metric_cis as bec  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc  # noqa: E402
from sksurv.util import Surv  # noqa: E402

OUT_DIR = eem.OUT_DIR
TIMES = [365, 548, 730]
B = 1000
SEED = 42
V3B_NAME = "v3b_lowvar_v2amgmt"
V3B_CFG = eem.VARIANTS[V3B_NAME]
V3C_NAME = ev3c.NAME


def main() -> int:
    data = bec.load_data()  # DB readonly SELECT + kimlik kapisi

    # --- v3b risk skorlari (mevcut, dokunulmayan yoldan) ---
    reported_v3b = pd.read_csv(OUT_DIR / f"week3_{V3B_NAME}_external_test.csv").iloc[0]
    ext_b, risk_b = eem.build_risk_scores(V3B_NAME, V3B_CFG, data)
    harrell_b = concordance_index(
        ext_b["survival_days"].to_numpy(float), -risk_b.to_numpy(), ext_b["event"].to_numpy(int)
    )
    gate_b_diff = abs(harrell_b - float(reported_v3b["c_index"]))
    if gate_b_diff > eem.GATE_TOL:
        print(f"[DUR] v3b Harrell kapisi BASARISIZ: fark={gate_b_diff:.3e}", file=sys.stderr)
        return 1

    # --- v3c risk skorlari (bu oturumda yazilan yoldan) ---
    reported_v3c = pd.read_csv(ev3c.V3C_DIR / f"week3_{V3C_NAME}_external_test.csv").iloc[0]
    ext_c, risk_c = ev3c.build_risk_scores_v3c(data)
    harrell_c = concordance_index(
        ext_c["survival_days"].to_numpy(float), -risk_c.to_numpy(), ext_c["event"].to_numpy(int)
    )
    gate_c_diff = abs(harrell_c - float(reported_v3c["c_index"]))
    if gate_c_diff > eem.GATE_TOL:
        print(f"[DUR] v3c Harrell kapisi BASARISIZ: fark={gate_c_diff:.3e}", file=sys.stderr)
        return 1

    # --- ORTAK HASTA KUMESI DOGRULAMASI ---
    idx_b = set(ext_b.index)
    idx_c = set(ext_c.index)
    only_b = idx_b - idx_c
    only_c = idx_c - idx_b
    common = sorted(idx_b & idx_c)
    print(f"[BILGI] v3b n={len(idx_b)}, v3c n={len(idx_c)}, ortak={len(common)}, "
          f"yalniz-v3b={len(only_b)}, yalniz-v3c={len(only_c)}")
    if only_b or only_c:
        print(f"[DIKKAT] Hasta kumesi FARKLI -- yalniz-v3b: {sorted(only_b)[:10]}..., "
              f"yalniz-v3c: {sorted(only_c)[:10]}...", file=sys.stderr)
    if len(common) == 0:
        print("[DUR] Ortak hasta kumesi BOS.", file=sys.stderr)
        return 1

    # Aynı sirada hizala (patient_id/index bazli)
    ext_b_c = ext_b.loc[common]
    ext_c_c = ext_c.loc[common]
    T_b = ext_b_c["survival_days"].to_numpy(dtype=float)
    E_b = ext_b_c["event"].to_numpy(dtype=int)
    T_c = ext_c_c["survival_days"].to_numpy(dtype=float)
    E_c = ext_c_c["event"].to_numpy(dtype=int)
    if not (np.array_equal(T_b, T_c) and np.array_equal(E_b, E_c)):
        print("[DUR] Ortak hastalarda survival_days/event UYUŞMUYOR -- "
              "iki kolun harici cercevesi ayni hasta icin farkli deger tasiyor.",
              file=sys.stderr)
        return 1
    T, E = T_b, E_b
    n = len(common)
    risk_b_arr = risk_b.loc[common].to_numpy()
    risk_c_arr = risk_c.loc[common].to_numpy()

    # --- Nokta tahmini deltalar (seviyesiz Harrell dahil) ---
    harrell_b_pt = concordance_index(T, -risk_b_arr, E)
    harrell_c_pt = concordance_index(T, -risk_c_arr, E)
    point = {"harrell": (harrell_c_pt, harrell_b_pt)}

    surv = Surv.from_arrays(event=E.astype(bool), time=T)
    for tau in TIMES:
        cb, *_ = concordance_index_ipcw(surv, surv, risk_b_arr, tau=float(tau))
        cc, *_ = concordance_index_ipcw(surv, surv, risk_c_arr, tau=float(tau))
        point[f"uno_c_tau{tau}"] = (float(cc), float(cb))
    aucs_b, _ = cumulative_dynamic_auc(surv, surv, risk_b_arr, times=TIMES)
    aucs_c, _ = cumulative_dynamic_auc(surv, surv, risk_c_arr, times=TIMES)
    for t, ab, ac in zip(TIMES, aucs_b, aucs_c):
        point[f"td_auc_{t}"] = (float(ac), float(ab))

    # --- Eslestirilmis bootstrap ---
    rng = np.random.default_rng(SEED)
    deltas: dict[str, list[float]] = {k: [] for k in point}
    fails = 0
    for _ in range(B):
        idx = rng.integers(0, n, n)
        Tb, Eb = T[idx], E[idx]
        if Eb.sum() == 0:
            fails += 1
            continue
        rb, rc = risk_b_arr[idx], risk_c_arr[idx]
        try:
            surv_b = Surv.from_arrays(event=Eb.astype(bool), time=Tb)
            h_c = concordance_index(Tb, -rc, Eb)
            h_b = concordance_index(Tb, -rb, Eb)
            deltas["harrell"].append(h_c - h_b)
            for tau in TIMES:
                cb, *_ = concordance_index_ipcw(surv_b, surv_b, rb, tau=float(tau))
                cc, *_ = concordance_index_ipcw(surv_b, surv_b, rc, tau=float(tau))
                deltas[f"uno_c_tau{tau}"].append(cc - cb)
            aucs_bb, _ = cumulative_dynamic_auc(surv_b, surv_b, rb, times=TIMES)
            aucs_cc, _ = cumulative_dynamic_auc(surv_b, surv_b, rc, times=TIMES)
            for t, ab, ac in zip(TIMES, aucs_bb, aucs_cc):
                deltas[f"td_auc_{t}"].append(ac - ab)
        except Exception:  # noqa: BLE001 -- ornek-bazli, SAYILIYOR
            fails += 1

    rows = []
    for metric, (v_c, v_b) in point.items():
        arr = np.array(deltas[metric])
        arr = arr[np.isfinite(arr)]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        delta_pt = v_c - v_b
        ci_excludes_zero = (lo > 0) or (hi < 0)
        rows.append({
            "metric": metric,
            "v3c_point": round(v_c, 4),
            "v3b_point": round(v_b, 4),
            "delta_point_v3c_minus_v3b": round(delta_pt, 4),
            "delta_ci_lower": round(float(lo), 4),
            "delta_ci_upper": round(float(hi), 4),
            "ci_excludes_zero": ci_excludes_zero,
            "hukum": "AYIRT EDILEBILIR" if ci_excludes_zero else "AYIRT EDILEMEZ (CI sifiri kapsiyor)",
            "n_valid_resamples": len(arr),
            "n_failed_resamples": fails,
        })

    out = pd.DataFrame(rows)
    target = OUT_DIR / "week3_paired_bootstrap_v3b_v3c.csv"
    out.to_csv(target, index=False)
    print(f"\n[OK] n_common={n} (v3b n={len(idx_b)}, v3c n={len(idx_c)}) -> {target}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
