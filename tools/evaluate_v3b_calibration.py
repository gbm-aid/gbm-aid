# -*- coding: utf-8 -*-
"""v3b_lowvar_v2amgmt (NIHAI SECILEN MODEL, 2026-09-12 Baris karari) icin
HARICI (UCSF) Brier/IPA + kalibrasyon degerlendirmesi.

NEDEN (2026-09-12 gorev talimati -- juri metni eksik bulgusu)
--------------------------------------------------------------
`entities/cox-model-varyant-katalogu.md` SS2c acikca kayda geciyor:
"Diger 5 varyantin kalibrasyonu OLCULMEMISTIR" -- ve secilen nihai model
artik `v1_referans` DEGIL, `v3b_lowvar_v2amgmt`. Kalibrasyon (buyukluk
dogrulugu) HENUZ bilinmiyor. Bu script `tools/evaluate_v1_calibration.py`
ile AYNI YONTEMI (Brier/IPA + tek-kovaryatli kalibrasyon egimi + decile
tablosu) v3b icin uygular. MEVCUT HICBIR DOSYAYA DOKUNMAZ -- yalniz YENI
dosyalar yazar (`week3_v3b_brier_ipa.csv`, `week3_v3b_calibration_365.csv`).

Uno C / td-AUC CI'lari icin bu script GEREKMIYOR -- v3b `evaluate_
external_metrics.VARIANTS` sozlugunde ZATEN VAR ve
`week3_external_extended_metrics.csv` / `week3_external_extended_metrics_
ci.csv` dosyalarinda v3b satiri (Uno C tau=365/548/730 + td-AUC 365/548/
730, %95 CI'lariyla) ONCEDEN URETILMIS DURUMDA (2026-08-19 kosusundan --
v3b o zaman zaten 6 orijinal varyanttan biriydi). Bu script SADECE
kalibrasyon bosluguna odaklanir, coklama YAPMAZ.

DEGERLENDIRME ZINCIRI vs DAGITIM ARTEFAKTI (v1'deki 0,0019 farkin AYNISI
v3b'de VAR MI?) -- BU SCRIPT IKISINI DE OLCUP KARSILASTIRIR:
  (a) DEGERLENDIRME zinciri: `evaluate_external_metrics.build_risk_
      scores()` ile `week3_v3b_lowvar_v2amgmt_final_coefficients.csv`'den
      YENIDEN uretilen risk skoru (bit-birebir kapi: mevcut CSV satirinda
      GECTI, fark 0,0 -- bu script AYRICA tam-hassasiyetle dogrular).
  (b) DAGITIM artefakti: `models/cox_phm_v3b_lowvar_v2amgmt_2026-09-12.
      pkl` (provenance: "week3_..._final_coefficients.csv + canli DB
      (2026-09-12)"den REPRODUCE edildi, standardizasyon TAM havuzdan
      (fold-ici DEGIL) -- degerlendirme zinciriyle AYNI formul).
Ikisinin katsayi kimligi + risk-skoru bit-esitligi asagida AYRICA
dogrulanir -- v1'deki gibi bir sapma varsa ACIKCA raporlanir, YOKSA da
bu ACIKCA yazilir (v1'in "iki ayri fit" bulgusu burada TEKRARLANMAYABILIR,
ama bu bir VARSAYIM degil, OLCUM olmali).

⚠️ KALIBRASYON IDDIASI KURULMAZ (decisions/2026-08-18-...): UPenn
`survival_days` AMELIYATTAN, UCSF `OS` TANIDAN olculuyor. C-index
(siralama) bundan AZ etkilenir, ama KALIBRASYON (buyukluk -- "tahmin
edilen S(t) ile gozlenen S(t) ORTUSUYOR MU") sistematik olarak
KAYABILIR (UCSF hastalari "tani anindan" sayilirken model UPenn'in
"ameliyat anindan" baseline hazard'iyla egitildi -- zaman sifiri farkli
bir klinik olaya karsilik geliyor). Bu script kalibrasyon SAYISINI
URETIR (Brier/IPA/egim/decile) ama "model UCSF'te iyi kalibre" gibi bir
SONUC ciktisi VERMEZ -- bu beyan rapora aynen tasinir.

URETILENLER (yalniz YENI dosyalar, DB'ye yazma YOK, mevcut 7/8 satirlik
CSV'lere DOKUNULMAZ):
  week3_v3b_brier_ipa.csv          Brier(t) + Brier_null(t) + IPA(t)
  week3_v3b_calibration_365.csv    decile kalibrasyon tablosu (t=365)
  week3_v3b_calibration_summary.csv  kalibrasyon egimi [CI] + decile MAE
                                   (2026-09-13 EKLENDI -- bu iki sayi
                                   onceden SADECE stdout'taydi, makine-
                                   okunur bir evi yoktu)
  stdout                           kapi sonuclari + kalibrasyon egimi + ozet

Salt-okunur: DB'ye SADECE readonly SELECT (`bootstrap_extended_metric_
cis.load_data()` uzerinden, kimlik kapisi DAHIL).
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
from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402
from sksurv.metrics import brier_score  # noqa: E402
from sksurv.util import Surv  # noqa: E402

OUT_DIR = eem.OUT_DIR
NAME = "v3b_lowvar_v2amgmt"
PKL = PROJECT_ROOT / "models" / "cox_phm_v3b_lowvar_v2amgmt_2026-09-12.pkl"
TIMES = [365, 548, 730]
N_CAL_BINS = 10

# Karar dosyasinda/gorev talimatinda bildirilen tam-hassasiyet deger
# (`tools/export_v3b_deployment_checkpoint.py` provenance'inda da AYNI):
EXPECTED_HARRELL_FULL_PRECISION = 0.6591716890743317
HARRELL_GATE_TOL = 1e-9  # "bit-birebir" -- v3c'nin 0.00e+00 kapisiyla AYNI sertlik
COEF_GATE_TOL = 1e-9  # provenance'ta 1.11e-16 bildiriliyor -- bu kapi onu DOGRULAR


def main() -> int:
    import pickle

    with open(PKL, "rb") as f:
        art = pickle.load(f)
    model: CoxPHFitter = art["fitted_model"]
    print(f"[YUK] {PKL.name} -- arm_name={art['arm_name']}  n_train={art['provenance']['n_train']}"
          f"/{art['provenance']['n_events_train']}")

    # --- DB (readonly) + kimlik kapisi (sema v2) -- ayni yol butun 7/8 kol icin ---
    data = bec.load_data()

    reported = pd.read_csv(OUT_DIR / f"week3_{NAME}_external_test.csv").iloc[0]
    ext_s, risk = eem.build_risk_scores(NAME, eem.VARIANTS[NAME], data)

    T = ext_s["survival_days"].to_numpy(dtype=float)
    E = ext_s["event"].to_numpy(dtype=int)

    # --- KAPI 1: DEGERLENDIRME ZINCIRI bit-birebir (tam hassasiyet) --------
    harrell_eval_chain = concordance_index(T, -risk.to_numpy(), E)
    reported_c = float(reported["c_index"])
    diff_vs_reported = abs(harrell_eval_chain - reported_c)
    diff_vs_expected = abs(harrell_eval_chain - EXPECTED_HARRELL_FULL_PRECISION)
    print(f"\nKAPI 1 (degerlendirme zinciri, tam hassasiyet):")
    print(f"  yeniden uretilen Harrell C   = {harrell_eval_chain!r}")
    print(f"  kosunun raporladigi c_index  = {reported_c!r}")
    print(f"  gorev talimatindaki deger    = {EXPECTED_HARRELL_FULL_PRECISION!r}")
    print(f"  |fark| vs raporlanan         = {diff_vs_reported:.3e}")
    print(f"  |fark| vs gorev talimati     = {diff_vs_expected:.3e}")
    if diff_vs_expected > HARRELL_GATE_TOL:
        print(
            f"KAPI BASARISIZ -- fark {diff_vs_expected:.3e} > {HARRELL_GATE_TOL:.0e}. "
            "DUR -- risk skorlari beklenen kosuyla eslesmiyor.",
            file=sys.stderr,
        )
        return 2
    print("  KAPI 1 GECTI -- bit-birebir eslesme (fark ~makine epsilon).")

    # --- KAPI 2: pkl (dagitim artefakti) vs CSV (degerlendirme zinciri) ----
    # katsayi kimligi -- v1'de bu kapi 0,0019'luk gercek bir sapma bulmustu;
    # burada AYNI titizlikle olculur, VARSAYILMAZ.
    coef = pd.read_csv(OUT_DIR / f"week3_{NAME}_final_coefficients.csv")
    cov_col = "covariate" if "covariate" in coef.columns else coef.columns[1]
    csv_coef = coef.set_index(cov_col)["coef"]

    pkl_set, csv_set = set(model.params_.index), set(csv_coef.index)
    if pkl_set != csv_set:
        raise RuntimeError(
            "KAPI 2: kovaryat kumeleri FARKLI -- yalniz-pkl'de: "
            f"{sorted(pkl_set - csv_set)}; yalniz-CSV'de: {sorted(csv_set - pkl_set)}"
        )
    coef_diff = float((model.params_ - csv_coef.reindex(model.params_.index)).abs().max())
    print(f"\nKAPI 2 (katsayi kimligi, pkl vs CSV): maks |fark| = {coef_diff:.3e}"
          f"  (esik {COEF_GATE_TOL:.0e})")
    if coef_diff > COEF_GATE_TOL:
        print(
            "KAPI 2 BASARISIZ -- pkl beklenen degerlendirme-zinciri katsayilarini "
            "TASIMIYOR (v1'deki gibi bir 'iki ayri fit' sapmasi OLABILIR). "
            "DUR -- kalibrasyon dagitim artefaktindan mi degerlendirme zincirinden "
            "mi geldigi ACIKLIGA KAVUSTURULMADAN devam edilmez.",
            file=sys.stderr,
        )
        return 3
    print("  KAPI 2 GECTI -- pkl (dagitim artefakti) == degerlendirme zinciri "
          "(v1'deki 0,0019 sapmasi burada YOK; ayni fit).")

    # --- pkl'nin kendi Harrell'i (predict_log_partial_hazard) -- capraz-kontrol ---
    X = ext_s[list(model.params_.index)]
    X_arr = X.to_numpy(dtype=float)
    lp = model.predict_log_partial_hazard(X).to_numpy()
    harrell_pkl = concordance_index(T, -lp, E)
    print(f"\n[CAPRAZ-KONTROL] pkl.predict_log_partial_hazard ile Harrell C = "
          f"{harrell_pkl!r}  (siralama-degismez normalizasyon farki OLABILIR, "
          "concordance'i degistirmez -- asagida dogrulaniyor)")
    print(f"  |fark| (degerlendirme zinciri vs pkl-lp) = {abs(harrell_pkl - harrell_eval_chain):.3e}")

    # --- S(t|x) tahminleri (pkl'nin KENDI baseline hazard'i + standardize X) ---
    surv_fn = model.predict_survival_function(X, times=TIMES)  # satir=t, kolon=hasta
    surv = Surv.from_arrays(event=E.astype(bool), time=T)

    # --- Brier + null-Brier + IPA ------------------------------------------
    km_all = KaplanMeierFitter().fit(T, E)
    rows = []
    for t in TIMES:
        pred = surv_fn.loc[t].to_numpy()
        _, (bs,) = brier_score(surv, surv, pred.reshape(-1, 1), [t])
        km_s = float(km_all.survival_function_at_times(t).iloc[0])
        _, (bs_null,) = brier_score(surv, surv, np.full((len(T), 1), km_s), [t])
        ipa = 1.0 - bs / bs_null
        rows.append(
            {"t_gun": t, "brier": round(bs, 4), "brier_null_km": round(bs_null, 4),
             "IPA": round(ipa, 4), "km_pop_sagkalim": round(km_s, 4)}
        )
    brier_table = pd.DataFrame(rows)
    brier_out = OUT_DIR / f"week3_{NAME.split('_')[0]}_brier_ipa.csv"
    brier_table.to_csv(brier_out, index=False)
    print(f"\nBRIER / IPA (null = kovaryatsiz KM) -> {brier_out.name}:")
    print(brier_table.to_string(index=False))

    # --- Kalibrasyon egimi --------------------------------------------------
    cal_df = pd.DataFrame({"lp": lp, "T": T, "E": E})
    slope_model = CoxPHFitter().fit(cal_df, duration_col="T", event_col="E")
    slope = float(slope_model.params_["lp"])
    slope_lo, slope_hi = slope_model.confidence_intervals_.loc["lp"]
    print(f"\nKALIBRASYON EGIMI: {slope:.3f}  [%95 CI {slope_lo:.3f} - {slope_hi:.3f}]"
          "   (1.0 ideal; <1 asiri-iddiali, >1 az-iddiali)")

    # --- Decile kalibrasyonu (t=365) ---------------------------------------
    pred365 = surv_fn.loc[365]
    dec = pd.qcut(pred365.rank(method="first"), N_CAL_BINS, labels=False)
    cal_rows = []
    for d in range(N_CAL_BINS):
        idx = dec[dec == d].index
        km = KaplanMeierFitter().fit(ext_s.loc[idx, "survival_days"], ext_s.loc[idx, "event"])
        obs = float(km.survival_function_at_times(365).iloc[0])
        try:
            ci = km.confidence_interval_survival_function_
            ts = ci.index[ci.index <= 365]
            lo, hi = (ci.loc[ts[-1]].tolist() if len(ts) else (np.nan, np.nan))
        except Exception:  # noqa: BLE001
            lo, hi = np.nan, np.nan
        cal_rows.append(
            {"decile": d + 1, "n": len(idx), "olay": int(ext_s.loc[idx, "event"].sum()),
             "tahmin_S365_ort": round(float(pred365.loc[idx].mean()), 4),
             "gozlenen_S365_KM": round(obs, 4),
             "KM_alt": round(float(lo), 4), "KM_ust": round(float(hi), 4)}
        )
    cal_table = pd.DataFrame(cal_rows)
    cal_out = OUT_DIR / f"week3_{NAME.split('_')[0]}_calibration_365.csv"
    cal_table.to_csv(cal_out, index=False)
    print(f"\nDECILE KALIBRASYONU (t=365 gun; decile 1 = EN DUSUK tahmini sagkalim) -> {cal_out.name}:")
    print(cal_table.to_string(index=False))

    mae = float(np.mean(np.abs(cal_table["tahmin_S365_ort"] - cal_table["gozlenen_S365_KM"])))
    print(f"\nDecile ortalama mutlak kalibrasyon farki (t=365): {mae:.4f}")

    # --- Makine-okunur ozet (2026-09-13, belge-agent-F) ----------------------
    # GEREKCE: kalibrasyon EGIMI ve decile MAE bu script'te 2026-09-12'de
    # SADECE stdout'a yaziliyordu -- iki sayinin tek kalici evi
    # log/2026-09-12.md ve entities/cox-model-varyant-katalogu.md
    # §2c-v3b'nin DUZ METNIYDI. Rapor/tablo uretimi bu sayilari
    # makine-okunur bicimde okuyabilsin diye ucuncu bir cikti eklendi.
    # Mevcut iki CSV'ye DOKUNULMADI (ayri dosya, ayri satir formati).
    summary_out = OUT_DIR / f"week3_{NAME.split('_')[0]}_calibration_summary.csv"
    pd.DataFrame(
        [
            {
                "metrik": "kalibrasyon_egimi",
                "t_gun": "",
                "deger": slope,
                "ci_alt": float(slope_lo),
                "ci_ust": float(slope_hi),
                "kaynak": f"{Path(__file__).name} (yeniden kosuldu)",
            },
            {
                "metrik": "decile_mae",
                "t_gun": 365,
                "deger": mae,
                "ci_alt": "",
                "ci_ust": "",
                "kaynak": f"{Path(__file__).name} (yeniden kosuldu)",
            },
        ]
    ).to_csv(summary_out, index=False)
    print(f"\nOZET (makine-okunur) -> {summary_out.name}")

    print(
        "\n[UYARI] KALIBRASYON IDDIASI KURULMAZ: UPenn survival_days AMELIYATTAN, "
        "UCSF OS TANIDAN olculuyor (decisions/2026-08-18-...). C-index siralama "
        "olcutu oldugu icin az etkilenir; kalibrasyon (buyukluk) BOZULABILIR. "
        "Yukaridaki sayilar OLCUMDUR, 'model iyi/kotu kalibre' SONUCU degildir."
    )
    print(
        "\nNOT: hangi fit uzerinden olculdu -- pkl (DAGITIM artefakti, "
        "models/cox_phm_v3b_lowvar_v2amgmt_2026-09-12.pkl), KAPI 2 bu pkl'nin "
        "degerlendirme zinciriyle (week3_v3b_..._final_coefficients.csv) "
        "katsayi-kimligiyle AYNI oldugunu dogruladi -- v1'deki 'iki ayri fit' "
        "sapmasi (0,0019) burada olculmedi/YOK."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
