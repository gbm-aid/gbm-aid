# -*- coding: utf-8 -*-
"""K14 -- `clinical_base` (YALNIZ yas+cinsiyet) kolunun UCSF'e karsi kosusu.

KARAR ZINCIRI
--------------
- 2026-08-15: K14 acildi -- "clinical_base kosulmadan radyomigin katkisi
  IDDIA EDILEMEZ" (yas tek basina in-sample 0,6295 vs radyomik nested-CV
  0,5999).
- 2026-08-18: Baris zaman kisitiyla ATLADI (K14 "atlandi", beyan kosulu).
- 2026-08-19: Baris YENIDEN ACTI ("baslatalim kosuyu") -- maliyet dustu,
  ve iyilestirme yatiriminin PUSULASI: klinik-only harici skor, radyomigin
  gercek katki payini belirler.

NEDEN AYRI SCRIPT (--clinical-arms DEGIL)
------------------------------------------
`run_clinical_arms()` YALNIZ legacy `run_pipeline()` (TCGA n=38) icinden
cagrilir -- 2026-08-15'te harici test TCGA'ykan yazildi. Harici test
2026-08-18'den beri UCSF (295/169); K14'u TCGA'ya karsi kosmak hem yanlis
karsilastirici olur hem kilitli karara aykiridir. Bu script AYNI public
fonksiyonlari (frame kurulumu, `run_modeling_arm`, kapilar) kullanarak
kolu UCSF'e karsi kosar.

TANIM (K14 orijinali, DARALTILMADAN): kovaryatlar YALNIZ
`clinical_age` + `clinical_gender_male`. GTR/IDH YOK (onlar
"radiomics_clinical" ailesinin konusu; K14'un sorusu "yas+cinsiyet
taban cizgisi nerede?").

PROTOKOL (6 varyantla BIREBIR AYNI): nested CV 5x5, 200 bootstrap
stabilite (bos radyomik havuzda no-op), 1000 bootstrap harici CI,
seed 42, yas fold-guvenli standardize. Kapilar: kimlik baseline (sema
v2) + egitim 611/585 + harici 295/169.

CIKTILAR (yalniz YENI dosyalar): week3_k14_clinical_base_ucsf_
{external_test.csv, fold_results.csv, final_coefficients.csv,
run_metadata.json}. DB'ye YAZMA YOK.

RAPOR NOTU (simdiden): bu, UCSF'te degerlendirilen 7. koldur -- cokluk
dipnotu buna gore guncellenecek. Sonuc NE OLURSA OLSUN raporlanir.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import train_cox_week3 as w3  # noqa: E402
from pipeline.cox_model import build_training_frame  # noqa: E402
from external_identity_baseline import compute_full_identity, verify_or_fail  # noqa: E402
from db_connection import get_connection  # noqa: E402
from psycopg2.extras import RealDictCursor  # noqa: E402

OUT_DIR = PROJECT_ROOT / "artifacts" / "week3" / "cox_model"
CACHE = OUT_DIR / "_cache"
ARM_NAME = "k14_clinical_base_ucsf"


def main() -> int:
    conn = get_connection(readonly=True)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    def cached(tag, tool):
        cur.execute("SELECT COUNT(*) AS n FROM radiomics WHERE segmentation_tool=%s", (tool,))
        n = int(cur.fetchone()["n"])
        f = pd.read_pickle(CACHE / f"{tag}_long.pkl")
        if len(f) != n:
            raise RuntimeError(f"{tag}: onbellek {len(f)} != DB {n} -- DUR")
        return f

    upenn_long = cached("upenn", w3.SEGMENTATION_TOOL_UPENN_C32)
    ucsf_long = cached("ucsf", w3.SEGMENTATION_TOOL_UCSF_C32)
    upenn_pat = w3.fetch_patients_frame(cur, source_name=w3.UPENN_SOURCE_NAME)
    ucsf_pat = w3.fetch_patients_frame(cur, source_name=w3.UCSF_SOURCE_NAME)
    cur.close(); conn.close()

    # Kohort tanimi 6 varyantla AYNI: WT pivotu (radyomik SATIRI olan
    # hastalar) -- clinical_base radyomik KULLANMAZ ama AYNI 611/295
    # hastada kosulmali ki karsilastirma elmayla elma olsun (legacy
    # clinical_base da ayni nedenle upenn_wide kohortunu kullaniyordu).
    up_wide, _ = w3.pivot_radiomics_long_to_wide(upenn_long, regions=["WT"])
    uc_wide, _ = w3.pivot_ucsf_regions_long_to_wide(ucsf_long, regions=("WT",))

    # KIMLIK KAPISI (sema v2, dondurulmus baseline)
    identity = compute_full_identity(uc_wide, ucsf_pat, up_wide, upenn_pat)
    baseline = verify_or_fail(identity)
    print(f"KIMLIK KAPISI GECTI (baseline: {baseline['frozen_at'][:19]})")

    # YALNIZ yas + cinsiyet (K14 tanimi -- GTR/IDH BILINCLI OLARAK YOK)
    extra = [w3.CLINICAL_AGE_COLUMN, w3.CLINICAL_GENDER_MALE_COLUMN]
    up_cl = w3.build_clinical_covariate_frame(
        upenn_pat, include_gender=True, include_gtr=False, include_idh=False
    )
    uc_cl = w3.build_clinical_covariate_frame(
        ucsf_pat, include_gender=True, include_gtr=False, include_idh=False
    )
    up_c = upenn_pat.join(up_cl)
    uc_c = ucsf_pat.join(uc_cl)

    # feature_frame = SIFIR kolon (radyomik YOK) ama index = WT kohortu
    empty_up = up_wide[[]]
    empty_uc = uc_wide[[]]

    training_frame, training_report = build_training_frame(
        empty_up, up_c, check_combat_identity=True, passthrough_columns=extra
    )
    n_events = int(training_frame["event"].sum())
    w3.check_training_pool_counts(
        training_report.n_output_rows, n_events,
        expected_patients=611, expected_events=585,
        regions=("WT",),
        dropped_patient_ids=w3.all_dropped_patient_ids(training_report),
    )
    external_frame, external_report = w3.build_external_test_frame(
        empty_uc, uc_c, passthrough_columns=extra
    )
    w3.check_external_test_pool_counts(
        len(external_frame), int(external_frame["event"].sum()),
        expected_patients=295, expected_events=169,
        allow_mismatch=False, label="UCSF (K14 clinical_base)",
    )
    print(f"KAPILAR GECTI: egitim {training_report.n_output_rows}/{n_events}, "
          f"harici {len(external_frame)}/{int(external_frame['event'].sum())}")

    arm = w3.run_modeling_arm(
        ARM_NAME,
        training_frame,
        feature_columns=[],            # RADYOMIK YOK -- K14'un ozu
        external_frame=external_frame,
        stability_frequency_threshold=0.6,
        outer_splits=5,
        inner_splits=5,
        n_bootstrap_stability=200,
        n_bootstrap_external=1000,
        seed=42,
        extra_columns=extra,
        extra_column_penalizer=0.0,    # 6 varyantla ayni: klinik penalize edilmez
        clinical_standardize_columns=[w3.CLINICAL_AGE_COLUMN],
        run_external_test=True,
    )

    # --- ciktilar ---
    ext = dict(arm.external_test)
    pd.DataFrame([{**ext, "clinical_extra_columns": ";".join(extra)}]).to_csv(
        OUT_DIR / f"week3_{ARM_NAME}_external_test.csv", index=False
    )
    arm.nested_cv_fold_results.to_csv(
        OUT_DIR / f"week3_{ARM_NAME}_fold_results.csv", index=False
    )
    coefs = arm.final_model.fitted_model.params_
    coefs.rename("coef").to_frame().reset_index().rename(
        columns={"index": "covariate"}
    ).to_csv(OUT_DIR / f"week3_{ARM_NAME}_final_coefficients.csv", index=False)

    fr = arm.nested_cv_fold_results
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm_name": ARM_NAME,
        "k14_decision_chain": "2026-08-15 acildi / 2026-08-18 atlandi (Baris) / "
                              "2026-08-19 yeniden acildi ve kosuldu (Baris)",
        "covariates": extra,
        "n_radiomic_features": 0,
        "training_pool": f"{training_report.n_output_rows}/{n_events}",
        "external_pool": f"{len(external_frame)}/{int(external_frame['event'].sum())}",
        "external_test": {k: v for k, v in ext.items()
                          if isinstance(v, (int, float, str))},
        "inner_cv_mean": float(fr["c_index"].mean()),
        "inner_cv_std": float(fr["c_index"].std(ddof=1)),
        "identity_baseline_frozen_at": baseline["frozen_at"],
        "multiplicity_note": "UCSF'te degerlendirilen 7. kol -- cokluk "
                             "dipnotu guncellenmeli.",
    }
    (OUT_DIR / f"week3_{ARM_NAME}_run_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print(f"\n=== K14 SONUC ({ARM_NAME}) ===")
    print(f"ic CV: {[round(x,4) for x in fr['c_index']]} "
          f"ort={fr['c_index'].mean():.4f} std={fr['c_index'].std(ddof=1):.4f}")
    print(f"harici: C={ext.get('c_index'):.6f} "
          f"CI[{ext.get('ci_lower'):.6f}, {ext.get('ci_upper'):.6f}] "
          f"n={ext.get('n_patients')} olay={ext.get('n_events')}")
    print("katsayilar:")
    for k, v in coefs.items():
        print(f"  {k:<28} {v:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
