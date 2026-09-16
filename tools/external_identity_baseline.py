# -*- coding: utf-8 -*-
"""Cox degerlendirme verilerinin DONDURULMUS KIMLIK baseline'i (v2).

GECMIS (ayni gun iki revizyon -- ikisi de Codex cikis incelemesi bulgusu)
-------------------------------------------------------------------------
v0: parmak izi YAZDIRILIYOR ama karsilastirilmiyordu -> kapi degildi.
v1: yalniz UCSF, uc hash (pid / surv / wt_pivot). Codex iki fail-open
    daha buldu ve HAKLIYDI:
    (1) KLINIK KOVARYAT DEGERLERI hicbir hash'te yoktu -- DB'de bir
        hastanin yasi/GTR'si/IDH'si degisse pid+surv+pivot AYNI kalir,
        kapi sessizce gecer, kalibrasyon/risk X'i degismis olurdu.
    (2) UPENN (EGITIM) TARAFI hic kilitli degildi (yalniz satir sayisi)
        -- standardizasyon istatistikleri, v3 filtre havuzu ve spline
        knot'lari egitim verisinden turetiliyor; "ayni sayida farkli
        egitim kaydi" yalniz metrics-scriptinin Harrell bit-kapisina
        yaslaniyordu, kalibrasyon scriptinde o cipa yok.
v2 (BU SURUM): IKI kohort (UCSF + UPenn), kohort basina UC hash:
    pid_sha256       sirali hasta kimlikleri
    patients_sha256  hasta cercevesinin TUM kolonlari (yas, cinsiyet,
                     GTR, IDH, MGMT, KPS, sagkalim, vital_status...) --
                     satirlar pid'e, kolonlar ada gore sirali CSV
                     gosteriminin SHA-256'si. "Hangi kolon klinik
                     kovaryat olur" ongorusune BAGLI DEGIL: pipeline'in
                     okuyabilecegi HER deger kilitli.
    wt_pivot_sha256  WT radyomik pivot matrisi (satir/kolon sirali,
                     float64 bayt) -- ozellik degerleri dahil.
Ayrica baseline DOSYASININ kendisi `tests/test_external_identity_
baseline.py` icinde SABITLERLE pinlidir: dosya sessizce yeniden
uretilirse/oynanirsa pytest KIRMIZI yanar. Yenileme tek mesru yoldan:
`freeze_external_identity_baseline.py --refreeze --reason "..."` + test
sabitlerinin BILINCLI guncellenmesi.

KAPSAM DURUSTLUGU: TC pivotu (yalniz v2c) baseline'da DEGIL -- v2c'nin
bit-birebir Harrell kapisi dolayli kapsar; LUMIERE/TCGA bu baseline'in
konusu degil (Cox degerlendirme zincirinde degiller).

DONDURMA GEREKCESI baseline dosyasinin icinde yazilidir.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_PATH = (
    Path(__file__).absolute().parent.parent
    / "artifacts" / "week3" / "cox_model"
    / "week3_ucsf_external_identity_baseline.json"
)

_COHORT_KEYS = ("n_pivot_patients", "n_patient_rows", "pid_sha256",
                "patients_sha256", "wt_pivot_sha256", "wt_pivot_shape")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_cohort_identity(wide_wt: pd.DataFrame, patients: pd.DataFrame) -> dict:
    """Bir kohortun uc kimlik hash'i. Girdiler: WT pivotu ve HAM hasta
    cercevesi (ikisi de index=patient_id)."""

    pids = sorted(map(str, wide_wt.index))
    pid_sha = _sha("|".join(pids))

    pat = patients.copy()
    pat.index = pat.index.map(str)
    pat = pat.loc[sorted(pat.index), sorted(map(str, pat.columns))]
    # TUM kolonlar, deterministik CSV gosterimi -- klinik kovaryatlar
    # (yas/GTR/IDH/MGMT/KPS) DAHIL. NaN bos hucre olarak sabittir.
    patients_sha = _sha(pat.to_csv())

    wide = wide_wt.loc[pids, sorted(map(str, wide_wt.columns))]
    arr = np.ascontiguousarray(wide.to_numpy(dtype=np.float64))
    pivot_sha = hashlib.sha256(arr.tobytes()).hexdigest()

    return {
        "n_pivot_patients": len(pids),
        "n_patient_rows": len(pat),
        "pid_sha256": pid_sha,
        "patients_sha256": patients_sha,
        "wt_pivot_sha256": pivot_sha,
        "wt_pivot_shape": list(wide.shape),
    }


def compute_full_identity(
    ucsf_wide_wt: pd.DataFrame, ucsf_patients: pd.DataFrame,
    upenn_wide_wt: pd.DataFrame, upenn_patients: pd.DataFrame,
) -> dict:
    return {
        "schema_version": 2,
        "cohorts": {
            "UCSF-PDGM": compute_cohort_identity(ucsf_wide_wt, ucsf_patients),
            "UPenn-GBM": compute_cohort_identity(upenn_wide_wt, upenn_patients),
        },
    }


class ExternalIdentityMismatchError(RuntimeError):
    """Dondurulmus kimlik ile canli veri UYUSMUYOR -- fail-loud."""


def verify_or_fail(computed: dict, baseline_path: Path = BASELINE_PATH) -> dict:
    """Baseline dosyasi YOKSA, sema v2 DEGILSE, bir kohort EKSIKSE ya da
    herhangi bir alan uyusmuyorsa ExternalIdentityMismatchError."""

    if not baseline_path.is_file():
        raise ExternalIdentityMismatchError(
            f"Kimlik baseline dosyasi YOK: {baseline_path}. Bilincli dondurma "
            "icin tools/freeze_external_identity_baseline.py kosulmali -- bu "
            "dogrulayici onu ASLA kendisi uretmez."
        )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != 2:
        raise ExternalIdentityMismatchError(
            f"Baseline sema surumu {baseline.get('schema_version')!r} != 2 -- "
            "eski/uyumsuz baseline SESSIZCE kabul edilmez; bilerek "
            "--refreeze gerekir."
        )
    for cohort, comp in computed["cohorts"].items():
        base = (baseline.get("cohorts") or {}).get(cohort)
        if base is None:
            raise ExternalIdentityMismatchError(
                f"Baseline'da {cohort} kohortu YOK -- kapsam sessizce "
                "daraltilamaz."
            )
        for key in _COHORT_KEYS:
            if base.get(key) != comp.get(key):
                raise ExternalIdentityMismatchError(
                    f"[{cohort}] kimlik uyusmazligi [{key}]: baseline="
                    f"{str(base.get(key))[:20]} olculen="
                    f"{str(comp.get(key))[:20]}. Veri, baseline'in "
                    f"donduruldugu ({baseline.get('frozen_at')}) durumdan "
                    "SAPMIS -- once sapmanin mesru olup olmadigi "
                    "arastirilmali; mesruysa --refreeze --reason ile "
                    "YUKSEK SESLE yenilenir (test sabitleri de guncellenir)."
                )
    return baseline
