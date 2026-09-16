"""Kosu #7'nin (kombine model) fit edilmis nesnesini YENIDEN URETIR ve
kalici bir artefakt olarak `models/` altina kaydeder.

NEDEN GEREKLI (2026-08-18):
`run_primary_single_model()` sonuc CSV/JSON'larini yazdi ama fit edilmis
model nesnesini (.pkl) DISKE YAZMADI. Sonuc: `api/predict.py` hala
2026-08-15'in yalniz-radyomik checkpoint'ini yukluyor; bugunku kombine
model API'den kullanilamiyor (rag-agent'in 3-senaryo testinde tespit
edildi).

Model kaybolmadi -- yeniden uretilebilir:
  - secilen 8 radyomik ozellik: week3_external_test.csv
  - 6 klinik kovaryat: ayni dosya
  - hiperparametreler: week3_run_metadata.json
  - veri: canli DB (degismedi)
  - seed sabit, fit deterministik

DOGRULAMA: yeniden fit edilen modelin katsayilari
`week3_final_model_coefficients.csv` ile karsilastirilir; sapma varsa
script DURUR.

Cikti: `models/cox_phm_primary_wt93_clinical_full_ucsf_2026-08-18.pkl`
Icerik: SADE bir sozluk (fitted_model + final_features + extra_columns +
metadata) -- `tools.train_cox_week3`'un dataclass'larina BAGIMLI DEGIL
(backend-agent'in uyarisi: pickle `__main__` bagimliligi API'de patlar).

DB'ye yalniz SELECT. Hicbir yazma yok.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "tools")

import train_cox_week3 as w3  # noqa: E402
from db_connection import get_connection  # noqa: E402
from pipeline.cox_model import build_training_frame, pivot_radiomics_long_to_wide  # noqa: E402

COX_DIR = Path("artifacts/week3/cox_model")
MODELS_DIR = Path("models")
ARTIFACT = MODELS_DIR / "cox_phm_primary_wt93_clinical_full_ucsf_2026-08-18.pkl"


def main() -> int:
    ext = pd.read_csv(COX_DIR / "week3_external_test.csv")
    row = ext.iloc[0]
    feats = [f.strip() for f in str(row["final_features"]).split(";") if f.strip()]
    clin = [c.strip() for c in str(row["clinical_extra_columns"]).split(";") if c.strip()]
    print(f"kol            : {row['arm']}")
    print(f"radyomik       : {len(feats)}")
    print(f"klinik kovaryat: {len(clin)}")

    meta_path = COX_DIR / "week3_run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    long_frame = pd.read_pickle(COX_DIR / "_cache/upenn_long.pkl")
    SQL = """
    select p.patient_id, d.source_name as source, p.survival_days, p.vital_status,
           p.age, p.gender, p.gtr_over90percent, p.idh1_status, p.mgmt_status
    from patients p join dataset_sources d on d.source_id = p.source_id
    where d.source_name = 'UPenn-GBM'
    """
    con = get_connection()
    try:
        patients = pd.read_sql(SQL, con).set_index("patient_id")
    finally:
        con.close()

    wide, _ = pivot_radiomics_long_to_wide(long_frame, w3.PRIMARY_REGIONS)
    clin_frame = w3.build_clinical_covariate_frame(
        patients, include_gender=True, include_gtr=True, include_idh=True
    )
    merged = wide.join(clin_frame, how="inner")
    frame, rep = build_training_frame(
        merged[feats + clin], patients, check_combat_identity=False
    )
    print(f"egitim cercevesi: {rep.n_output_rows} hasta")
    if rep.n_output_rows != 611:
        print(f"HATA: beklenen 611, gelen {rep.n_output_rows}", file=sys.stderr)
        return 2

    from lifelines import CoxPHFitter

    cph = CoxPHFitter(penalizer=0.05, l1_ratio=0.0)
    cph.fit(frame, duration_col="survival_days", event_col="event")

    # --- DOGRULAMA: katsayilar eskiyle ayni mi ---
    ref_path = COX_DIR / "week3_final_model_coefficients.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path, index_col=0)
        new = cph.summary["coef"]
        common = [c for c in ref.index if c in new.index]
        diff = float(np.abs(ref.loc[common, "coef"].values - new.loc[common].values).max())
        print(f"katsayi dogrulamasi: {len(common)} kovaryat, maks fark {diff:.8f}")
        if diff > 1e-6:
            print("HATA: katsayilar eskiyle UYUSMUYOR -- durduruldu", file=sys.stderr)
            return 3
    else:
        print("UYARI: referans katsayi dosyasi yok, dogrulama atlandi")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "arm_name": str(row["arm"]),
        "fitted_model": cph,
        "final_features": feats,
        "clinical_extra_columns": clin,
        "penalizer": 0.05,
        "l1_ratio": 0.0,
        "n_train": int(rep.n_output_rows),
        "n_events_train": int(frame["event"].sum()),
        "external_test": {
            "source": "UCSF-PDGM",
            "n_patients": int(row["n_patients"]),
            "n_events": int(row["n_events"]),
            "c_index": float(row["c_index"]),
            "ci_lower": float(row["ci_lower"]),
            "ci_upper": float(row["ci_upper"]),
        },
        "provenance": {
            "rebuilt_from": "week3_external_test.csv + canli DB",
            "reason": "kosu #7 fit edilmis nesneyi diske yazmadi",
            "rebuilt_at": pd.Timestamp.utcnow().isoformat(),
            "script_sha256_of_run": meta.get("script_sha256"),
            "note": (
                "Bu artefakt kosunun kendisi tarafindan degil, sonradan ayni "
                "veri+parametrelerle yeniden fit edilerek uretildi. Katsayilar "
                "week3_final_model_coefficients.csv ile dogrulandi."
            ),
        },
    }
    with ARTIFACT.open("wb") as fh:
        pickle.dump(payload, fh)
    print(f"kaydedildi -> {ARTIFACT}  ({ARTIFACT.stat().st_size} bayt)")

    with ARTIFACT.open("rb") as fh:
        back = pickle.load(fh)
    print(f"geri okuma testi: {len(back['final_features'])} ozellik, "
          f"{len(back['clinical_extra_columns'])} klinik, "
          f"C-index {back['external_test']['c_index']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
