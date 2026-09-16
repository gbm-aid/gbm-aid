"""SHAP sarmalayici prototipi -- `POST /predict/{patient_id}` icin (plan
raw/plan/plan.txt:449-451: "risk_score + SHAP", `shap_values` JSONB).

KAPSAM (2026-08-18, backend-agent gorev talimati): bu BAGIMSIZ bir
PROTOTIP -- `pipeline/cox_model.py`/`tools/train_cox_week3.py`'ye
DOKUNMUYOR (modeling-agent'in alani, cakisma onleme kilidi). Var olan
`pipeline.cox_model.compute_shap_values()` (KernelExplainer + `predict_
partial_hazard`, YALNIZ sentetik veriyle test edilmis, GERCEK sayisal
dogruluk kontrolu HIC yapilmamis) BURADA DEGISTIRILMEDI -- bu script
ondan BAGIMSIZ, kendi kapali-form + KernelExplainer karsilastirmasini
yapar ve (a) hangi olcegin doğru oldugunu SAYISAL kanitla, (b) hiz
olc, (c) GERCEK `_checkpoints/arm_primary_wt93_icc60_th06.pkl` modeliyle
dene.

ANA BULGU (bu scriptin urettigi, asagida SAYISAL dogrulanir):
`lifelines.CoxPHFitter.predict_log_partial_hazard(X)` TAM OLARAK
`coef @ (X - norm_mean)` -- yani DOGRUSAL. Bu yuzden:
  1. Doğru olcek `predict_log_partial_hazard` (log-hazard) -- Cox
     modelinin kendisi bu uzayda dogrusal, additivite (SHAP toplami +
     taban = tahmin) BU uzayda TAM (float precision) tutuyor.
     `predict_partial_hazard` (= exp(log-hazard)) DOGRUSAL DEGIL --
     SHAP orada da hesaplanabilir ama katkilar carpimsal etkilesim
     tasir, "ozellik X, riski Y birim artirdi" gibi TEMIZ bir okuma
     VERMEZ (bu script ikisini de olcup karsilastiriyor, secim
     rastgele degil).
  2. `shap.LinearExplainer((coef, intercept), background,
     feature_perturbation="interventional")` KULLANILIRKEN intercept
     MUTLAKA `-(coef @ norm_mean)` olmali -- `intercept=0` verilirse
     (ilk denemede yapilan hata, asagida BILEREK gosteriliyor)
     toplam+taban `coef @ x`'e esitlenir, `predict_log_partial_hazard`
     ile SISTEMATIK bir sabit kadar (ornek: sentetik veri seed=0,
     n=200, 3 ozellik -> 0.0151) FARKLI cikar -- YANLIS ama SESSIZCE
     "calisiyor gibi gorunen" bir hata sinifi, bu script iki halini de
     gosterip ikinciyi (dogru intercept) DOGRULUYOR.

CALISTIRMA:
    python tools/shap_explainer_prototype.py            # sentetik + gercek model
    python tools/shap_explainer_prototype.py --skip-db  # yalniz sentetik + cache'ten gercek model (DB'ye hic baglanmaz)

DB ERISIMI: yalniz `readonly=True`, yalniz SELECT (bir hastanin C32
WT satirini cekmek icin, gercek-hasta demosu). INSERT/UPDATE/ALTER
YOK. `_checkpoints/*.pkl` YALNIZ OKUNUR (uzerine yazilmiyor/tasinmiyor).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SEED = 42
# `pipeline.cox_model.compute_shap_values()` ile AYNI varsayilan (50) --
# karsilastirilabilirlik icin bilincli, baska bir gerekce yok. Gercek
# model bolumunde n=93/107 sutunlu degil, 6 sutunlu (final_features)
# oldugu icin arka plan orneklem sayisi ozellik sayisindan cok daha
# buyuk -- KernelExplainer'in kararliligi icin yeterli (kucuk M'de M*10
# kurali literatur onerisi, 6 ozellik icin 50-100 fazlasiyla yeterli).
DEFAULT_N_BACKGROUND = 50

CHECKPOINT_PATH = (
    REPO_ROOT
    / "artifacts"
    / "week3"
    / "cox_model"
    / "_checkpoints"
    / "arm_primary_wt93_icc60_th06.pkl"
)
UPENN_LONG_CACHE_PATH = (
    REPO_ROOT / "artifacts" / "week3" / "cox_model" / "_cache" / "upenn_long.pkl"
)

# C32 kapali beyaz liste -- gorev talimatinda ACIKCA verilen 4 isim.
# `tools/train_cox_week3.py::ALLOWED_C32_SEGMENTATION_TOOLS` ile KASITLI
# olarak AYRI tutuluyor (o dosyaya DOKUNULMUYOR, modeling-agent'in
# egitim-havuzu beyaz listesi LUMIERE'i icermiyor -- egitimde LUMIERE
# olay verisi yok -- ama /predict TEK HASTA icin sadece radyomik
# gerektirir, LUMIERE bir hastanin da tahmin istemesi mesru olabilir,
# bu yuzden /predict'in kendi kapisi 4 ismi de kabul eder).
ALLOWED_C32_SEGMENTATION_TOOLS_FOR_PREDICT: frozenset[str] = frozenset(
    {
        "UPenn-PyRadiomics-107-C32",
        "LUMIERE-PyRadiomics-107-C32",
        "TCGA-ground-truth-C32",
        "UCSF-PDGM-PyRadiomics-107-C32",
    }
)


# =====================================================================
# 0) Gercek checkpoint'i GUVENLI (salt-okunur) yukleme
# =====================================================================


def load_arm_checkpoint(path: Path = CHECKPOINT_PATH) -> Any:
    """`_checkpoints/arm_*.pkl`'i YALNIZ OKUYARAK acar.

    KIRILGANLIK (gorev talimatinda onceden tespit edilmis, burada
    dogrulanip cozuluyor): pkl `tools.train_cox_week3` altinda
    `__main__` modulu varsayimiyla (script dogrudan `python tools/
    train_cox_week3.py ...` ile calistirildigi icin) pickle'lanmis --
    duz `pickle.load()` bu surecte `AttributeError: Can't get
    attribute 'ArmResult' on <module '__main__'>` ile patlar. Cozum:
    `tools.train_cox_week3`'u import edip `sys.modules['__main__']`'a
    GECICI olarak o modulu ata (yalniz bu fonksiyonun suresi boyunca,
    global durumu degistirmemek icin finally'de eski `__main__`'a geri
    donulur).
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint bulunamadi: {path}. Bu prototip GERCEK bir "
            "egitim kosusunun ciktisina bagimli -- yoksa `--skip-real-model` "
            "kullanin (yalniz sentetik demo calisir)."
        )

    import tools.train_cox_week3 as train_module

    original_main = sys.modules.get("__main__")
    sys.modules["__main__"] = train_module
    try:
        with path.open("rb") as fh:
            arm_result = pickle.load(fh)
    finally:
        if original_main is not None:
            sys.modules["__main__"] = original_main
        else:  # pragma: no cover -- normal calistirmada __main__ hep var
            del sys.modules["__main__"]
    return arm_result


# =====================================================================
# 1) Sentetik veri + model (hiz/dogruluk olcumu icin, gercek DB'ye
#    BAGLANMADAN calisir)
# =====================================================================


def build_synthetic_cox_dataset(
    n_patients: int, n_features: int, *, seed: int = DEFAULT_SEED
) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    feature_cols = [f"synthetic_feature_{i}" for i in range(n_features)]
    data = {col: rng.normal(size=n_patients) for col in feature_cols}
    frame = pd.DataFrame(data)
    frame["survival_days"] = rng.exponential(scale=500, size=n_patients) + 1.0
    frame["event"] = rng.binomial(1, 0.7, size=n_patients)
    return frame, feature_cols


def fit_synthetic_cox_model(frame: pd.DataFrame, feature_cols: list[str]):
    from lifelines import CoxPHFitter

    model = CoxPHFitter(penalizer=0.01)
    model.fit(
        frame[feature_cols + ["survival_days", "event"]],
        duration_col="survival_days",
        event_col="event",
    )
    return model


# =====================================================================
# 2) Kapali-form LinearExplainer sarmalayicisi (dogru olcek: log-hazard)
# =====================================================================


def explain_with_linear_explainer(
    fitted_model,
    feature_cols: list[str],
    background_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
    *,
    use_correct_intercept: bool,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """`shap.LinearExplainer` ile `predict_log_partial_hazard` skorunu acikla.

    `use_correct_intercept=False`: KASITLI HATALI cagri (`intercept=0`)
    -- karsilastirma icin, "sessizce yanlis calisir" hata sinifini
    somut gostermek uzere burada birakildi (varsayilan DEGIL, cagiran
    taraf acikca istemeli).
    `use_correct_intercept=True` (dogru/onerilen): `intercept = -(coef
    @ norm_mean)` -- `predict_log_partial_hazard` ile TAM (float
    precision) ayni sonucu SHAP toplam+taban olarak uretir (asagidaki
    `run_synthetic_benchmark`/`run_real_model_demo` sayisal olarak
    dogruluyor).

    Donen: (shap_frame, base_values, gecen_sure_saniye).
    """

    import shap

    coef = fitted_model.params_[feature_cols].to_numpy()
    if use_correct_intercept:
        norm_mean = fitted_model._norm_mean[feature_cols].to_numpy()
        intercept = -float(coef @ norm_mean)
    else:
        intercept = 0.0

    background_matrix = background_frame[feature_cols].to_numpy()
    explainer = shap.LinearExplainer(
        (coef, intercept), background_matrix, feature_perturbation="interventional"
    )

    start = time.perf_counter()
    explanation = explainer(query_frame[feature_cols].to_numpy())
    elapsed = time.perf_counter() - start

    shap_frame = pd.DataFrame(
        explanation.values, columns=feature_cols, index=query_frame.index
    )
    base_values = np.asarray(explanation.base_values, dtype=float)
    if base_values.ndim == 0:
        base_values = np.full(len(query_frame), float(base_values))
    return shap_frame, base_values, elapsed


# =====================================================================
# 3) KernelExplainer sarmalayicisi (model-agnostik, `pipeline.cox_model.
#    compute_shap_values()` ile AYNI aile -- ama burada olcek/hiz/dogruluk
#    ACIKCA olculuyor, oradaki stub bunu YAPMIYORDU)
# =====================================================================


def explain_with_kernel_explainer(
    fitted_model,
    feature_cols: list[str],
    background_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
    *,
    scale: str,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """`scale`: `"log_partial_hazard"` (dogrusal, onerilen) veya
    `"partial_hazard"` (hazard-ratio, DOGRUSAL DEGIL -- karsilastirma
    icin)."""

    import shap

    if scale == "log_partial_hazard":
        predict_fn = fitted_model.predict_log_partial_hazard
    elif scale == "partial_hazard":
        predict_fn = fitted_model.predict_partial_hazard
    else:
        raise ValueError(f"Bilinmeyen scale: {scale!r}")

    def _predict(data: np.ndarray) -> np.ndarray:
        as_frame = pd.DataFrame(data, columns=feature_cols)
        return predict_fn(as_frame).to_numpy()

    background_matrix = background_frame[feature_cols].to_numpy()
    explainer = shap.KernelExplainer(_predict, background_matrix, seed=seed)

    start = time.perf_counter()
    shap_values = explainer.shap_values(
        query_frame[feature_cols].to_numpy(), silent=True
    )
    elapsed = time.perf_counter() - start

    shap_frame = pd.DataFrame(shap_values, columns=feature_cols, index=query_frame.index)
    base_values = np.full(len(query_frame), float(explainer.expected_value))
    return shap_frame, base_values, elapsed


# =====================================================================
# 4) Dogruluk kontrolu -- SHAP toplami + taban == modelin GERCEK tahmini
# =====================================================================


def verify_additivity(
    shap_frame: pd.DataFrame,
    base_values: np.ndarray,
    true_prediction: np.ndarray,
    *,
    label: str,
    atol: float = 1e-6,
) -> bool:
    reconstructed = shap_frame.to_numpy().sum(axis=1) + base_values
    max_abs_diff = float(np.max(np.abs(reconstructed - true_prediction)))
    passed = max_abs_diff <= atol
    status = "GECTI" if passed else "BASARISIZ"
    print(
        f"  [additivite] {label}: max|SHAP_toplam+taban - gercek| = "
        f"{max_abs_diff:.3e} (esik={atol:.0e}) -> {status}"
    )
    if not passed:
        print(
            f"    UYARI: {label} icin SHAP toplami tahmine esit CIKMADI -- "
            "bu olcek/yontem 'dogrulandi' olarak RAPORLANAMAZ."
        )
    return passed


# =====================================================================
# 5) Sentetik uctan-uca benchmark
# =====================================================================


def run_synthetic_benchmark() -> None:
    print("=" * 70)
    print("SENTETIK VERI BENCHMARK (gercek model/DB kullanilmiyor)")
    print("=" * 70)

    n_patients = 200
    n_features = 6  # gercek primer modelin ozellik sayisiyla ESLESTIRILDI
    frame, feature_cols = build_synthetic_cox_dataset(
        n_patients, n_features, seed=DEFAULT_SEED
    )
    model = fit_synthetic_cox_model(frame, feature_cols)

    background = frame[feature_cols].sample(
        n=min(DEFAULT_N_BACKGROUND, len(frame)), random_state=DEFAULT_SEED
    )
    query = frame[feature_cols].iloc[:5]
    true_log_hazard = model.predict_log_partial_hazard(query).to_numpy()
    true_partial_hazard = model.predict_partial_hazard(query).to_numpy()

    print(f"\n-- LinearExplainer, intercept=0 (KASITLI HATALI cagri) --")
    shap_wrong, base_wrong, t_wrong = explain_with_linear_explainer(
        model, feature_cols, background, query, use_correct_intercept=False
    )
    verify_additivity(
        shap_wrong, base_wrong, true_log_hazard, label="Linear/intercept=0 (log-hazard)"
    )
    print(f"  sure: {t_wrong:.4f}s / {len(query)} hasta")

    print(f"\n-- LinearExplainer, intercept=-(coef@norm_mean) (DOGRU) --")
    shap_correct, base_correct, t_correct = explain_with_linear_explainer(
        model, feature_cols, background, query, use_correct_intercept=True
    )
    ok_linear = verify_additivity(
        shap_correct,
        base_correct,
        true_log_hazard,
        label="Linear/dogru-intercept (log-hazard)",
    )
    print(f"  sure: {t_correct:.4f}s / {len(query)} hasta")

    print(f"\n-- KernelExplainer, scale=log_partial_hazard --")
    shap_kernel_log, base_kernel_log, t_kernel_log = explain_with_kernel_explainer(
        model, feature_cols, background, query, scale="log_partial_hazard"
    )
    ok_kernel_log = verify_additivity(
        shap_kernel_log,
        base_kernel_log,
        true_log_hazard,
        label="Kernel (log-hazard)",
        atol=1e-3,  # KernelExplainer orneklem-tabanli, tam SIFIR beklenmez
    )
    print(f"  sure: {t_kernel_log:.4f}s / {len(query)} hasta")

    print(f"\n-- KernelExplainer, scale=partial_hazard (hazard-ratio, DOGRUSAL DEGIL) --")
    shap_kernel_hr, base_kernel_hr, t_kernel_hr = explain_with_kernel_explainer(
        model, feature_cols, background, query, scale="partial_hazard"
    )
    ok_kernel_hr = verify_additivity(
        shap_kernel_hr,
        base_kernel_hr,
        true_partial_hazard,
        label="Kernel (partial-hazard/HR)",
        atol=1e-3,
    )
    print(f"  sure: {t_kernel_hr:.4f}s / {len(query)} hasta")

    print("\n-- HIZ OZETI (hasta basina, background=%d) --" % DEFAULT_N_BACKGROUND)
    print(f"  LinearExplainer (kapali-form): {t_correct / len(query) * 1000:.3f} ms/hasta")
    print(f"  KernelExplainer (log-hazard):  {t_kernel_log / len(query) * 1000:.3f} ms/hasta")
    print(f"  KernelExplainer (hazard-ratio):{t_kernel_hr / len(query) * 1000:.3f} ms/hasta")

    all_ok = ok_linear and ok_kernel_log and ok_kernel_hr
    print(
        f"\nSONUC: tum dogruluk kontrolleri {'GECTI' if all_ok else 'BASARISIZ -- bkz. yukaridaki UYARI(lar)'}."
    )


# =====================================================================
# 6) GERCEK model demo (checkpoint + cache'ten arka plan, opsiyonel DB)
# =====================================================================


def run_real_model_demo(*, use_db: bool, patient_id: str | None) -> None:
    print("\n" + "=" * 70)
    print("GERCEK MODEL DEMO (_checkpoints/arm_primary_wt93_icc60_th06.pkl)")
    print("=" * 70)

    arm = load_arm_checkpoint()
    fitted_model = arm.final_model.fitted_model
    final_features = list(arm.final_model.final_features)
    extra_columns = list(getattr(arm, "extra_columns", None) or [])
    print(f"Kol: {arm.name!r}")
    print(f"final_features ({len(final_features)}): {sorted(final_features)}")
    print(f"extra_columns (klinik, varsa): {extra_columns}")
    print(
        f"Checkpoint'teki harici test c_index: "
        f"{arm.external_test.get('c_index')} (yalniz karsilastirma icin, "
        "bu script yeniden HESAPLAMIYOR)"
    )
    if extra_columns:
        print(
            "UYARI: bu checkpoint klinik kovaryat (extra_columns) TASIYOR -- "
            "asagidaki demo SADECE radyomik final_features'i kullanir, "
            "extra_columns'u ATLAR (bu bir kapsam sinirlamasidir, /predict "
            "iskeletiyle AYNI sinirlama, bkz. api/predict.py)."
        )

    from pipeline.cox_model import pivot_radiomics_long_to_wide

    if not UPENN_LONG_CACHE_PATH.is_file():
        raise FileNotFoundError(
            f"Egitim havuzu cache'i bulunamadi: {UPENN_LONG_CACHE_PATH}. "
            "Arka plan orneklemi icin GEREKLI -- tools/train_cox_week3.py "
            "en az bir kez calisip cache'i yazmis olmali."
        )
    upenn_long = pd.read_pickle(UPENN_LONG_CACHE_PATH)
    regions_needed = sorted({col.split("__", 1)[0] for col in final_features})
    if regions_needed != ["WT"]:
        raise RuntimeError(
            f"Bu prototip yalniz WT-only kollari icin yazildi, bulunan "
            f"bolgeler: {regions_needed}. Kol degistiyse (klinik/coklu-"
            "bolge) BURADA GENISLETILMELI, sessizce yanlis sonuc "
            "URETILMEMELI."
        )
    upenn_wide, pivot_report = pivot_radiomics_long_to_wide(upenn_long, ["WT"])
    missing_features = set(final_features) - set(upenn_wide.columns)
    if missing_features:
        raise RuntimeError(
            f"Egitim havuzunda final_features'in bazilari YOK: {missing_features}"
        )
    print(
        f"Arka plan havuzu: {pivot_report.n_output_patients} hasta "
        f"(UPenn C32 WT, cache'ten), {DEFAULT_N_BACKGROUND} orneklem alinacak."
    )

    background = upenn_wide[final_features].sample(
        n=min(DEFAULT_N_BACKGROUND, len(upenn_wide)), random_state=DEFAULT_SEED
    )

    if use_db:
        query_frame, source_label = _fetch_one_patient_wt_row(
            final_features, patient_id=patient_id
        )
    else:
        # DB'ye baglanmadan: cache'teki havuzdan RASTGELE bir hastayi
        # "sanki disaridan gelmis" gibi kullan (yalniz --skip-db demosu,
        # gercek /predict akisini TEMSIL ETMEZ -- yalniz SHAP mekanigini
        # gosterir).
        query_frame = upenn_wide[final_features].sample(n=1, random_state=7)
        source_label = f"cache (DB atlandi, --skip-db) -- hasta_id={query_frame.index[0]}"

    print(f"Sorgu hastasi kaynagi: {source_label}")

    true_log_hazard = fitted_model.predict_log_partial_hazard(
        query_frame[final_features]
    ).to_numpy()
    true_partial_hazard = fitted_model.predict_partial_hazard(
        query_frame[final_features]
    ).to_numpy()

    shap_frame, base_values, elapsed = explain_with_linear_explainer(
        fitted_model,
        final_features,
        background,
        query_frame,
        use_correct_intercept=True,
    )
    ok = verify_additivity(
        shap_frame, base_values, true_log_hazard, label="GERCEK model (log-hazard)"
    )
    print(f"Hiz: {elapsed:.4f}s / {len(query_frame)} hasta (LinearExplainer, kapali-form)")
    print(f"risk_score (log_partial_hazard): {true_log_hazard}")
    print(f"partial_hazard (HR, sadece bilgi icin, SHAP BUNU ACIKLAMIYOR): {true_partial_hazard}")
    print("shap_values (log-hazard skalasi):")
    print(shap_frame.T)
    print(f"base_value: {base_values}")

    # Ayni hastada KernelExplainer ile karsilastirmali hiz olcumu.
    shap_kernel, base_kernel, t_kernel = explain_with_kernel_explainer(
        fitted_model, final_features, background, query_frame, scale="log_partial_hazard"
    )
    verify_additivity(
        shap_kernel,
        base_kernel,
        true_log_hazard,
        label="GERCEK model, KernelExplainer (log-hazard)",
        atol=1e-3,
    )
    print(f"Hiz (KernelExplainer, karsilastirma): {t_kernel:.4f}s / {len(query_frame)} hasta")

    if not ok:
        print(
            "\nSONUC (gercek model): DOGRULANMADI -- SHAP toplami tahmine "
            "esit CIKMADI, /predict icin bu haliyle KULLANILMAMALI."
        )
    else:
        print("\nSONUC (gercek model): SHAP additivite TAM (float precision) DOGRULANDI.")


def _fetch_one_patient_wt_row(
    final_features: list[str], *, patient_id: str | None
) -> tuple[pd.DataFrame, str]:
    """DB'den (readonly, SELECT-only) bir hastanin C32 WT radyomik satirini
    ceker ve `final_features` sutunlarina pivotlar.

    `api/predict.py`'nin ayni sorgu/pivot desenini KULLANIR (bu fonksiyon
    orada da tekrar yazilmiyor -- ayni mantik burada PROTOTIP olarak
    once denendi, `api/predict.py` bunu kendi ic fonksiyonu olarak
    ayrica tasir; kod TEKRARI var ama modul BAGIMLILIGI YOK -- api/
    predict.py bu dosyayi import ETMEZ, ikisi bagimsiz kalir, ki
    prototip dosyasi GELECEKTE silinebilsin).
    """

    from db_connection import get_connection
    from pipeline.cox_model import pivot_radiomics_long_to_wide
    import psycopg2.extras

    conn = get_connection(readonly=True)
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if patient_id is None:
            # ONEMLI GERCEK-DB BULGUSU (2026-08-18, bu prototip calistirilirken
            # KESFEDILDI): varsayilan "ilk bulunan hasta" secimi ONCE
            # LUMIERE'e dustu (Patient-001) ve pivot `RegionPivotError` ile
            # PATLADI -- sebep bir yazim hatasi DEGIL, LUMIERE'in
            # LONGITUDINAL tasarimi (bir hastanin AYNI bolge icin BIRDEN
            # FAZLA scan_id'si/zaman-noktasi var, ornek: Patient-001 icin
            # WT_derived scan_id={2839,2843,2847,2851}). `pivot_radiomics_
            # long_to_wide()` HASTA-DUZEYINDE bir satir bekler -- bu varsayim
            # UPenn/TCGA/UCSF (tek-zaman-noktali C32 verisi) icin GECERLI,
            # LUMIERE icin GECERLI DEGIL. Bu yuzden demo/varsayilan secim
            # BILEREK UPenn'i ONCELIKLENDIRIR (egitim havuzuyla da tutarli) --
            # `api/predict.py`'nin GERCEK davranisi (bkz. o dosyanin
            # docstring'i) coklu-scan_id durumunu SESSIZCE ORTALAMA/SECIM
            # yapmadan ACIKCA 422 ile reddeder, bu script o mantigi TEKRAR
            # ETMEZ (yalniz burada calisabilir bir demo hastasi secer).
            cur.execute(
                """
                SELECT p.patient_id
                FROM radiomics r
                JOIN mr_scans ms ON ms.scan_id = r.scan_id
                JOIN patients p ON p.patient_id = ms.patient_id
                JOIN dataset_sources ds ON ds.source_id = p.source_id
                WHERE r.segmentation_tool = 'UPenn-PyRadiomics-107-C32'
                ORDER BY p.patient_id
                LIMIT 1
                """,
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(
                    "DB'de UPenn-PyRadiomics-107-C32 whitelist'inde HICBIR satir "
                    "bulunamadi (demo icin varsayilan kaynak)."
                )
            patient_id = row["patient_id"]

        cur.execute(
            """
            SELECT p.patient_id, ds.source_name AS source, r.tumor_region,
                   r.shape_features, r.first_order_features, r.texture_features,
                   r.segmentation_tool, r.scan_id
            FROM radiomics r
            JOIN mr_scans ms ON ms.scan_id = r.scan_id
            JOIN patients p ON p.patient_id = ms.patient_id
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE p.patient_id = %s AND r.segmentation_tool = ANY(%s)
            """,
            (patient_id, list(ALLOWED_C32_SEGMENTATION_TOOLS_FOR_PREDICT)),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    if not rows:
        raise RuntimeError(
            f"Hasta {patient_id!r} icin C32 whitelist'inde radyomik satiri YOK."
        )
    long_frame = pd.DataFrame(rows)
    n_distinct_scans = long_frame["scan_id"].nunique()
    if n_distinct_scans > 1:
        raise RuntimeError(
            f"Hasta {patient_id!r}: {n_distinct_scans} FARKLI scan_id C32 "
            "whitelist'inde bulundu -- bu hasta LONGITUDINAL (birden fazla "
            "zaman noktasi, ornegin LUMIERE) veya ayni bolge icin birden "
            "fazla segmentation_tool yazilmis olabilir. Bu prototip/`/predict` "
            "v1 tek-zaman-noktali hasta VARSAYAR, SESSIZCE bir scan_id "
            "SECMEZ -- kapsam disi (bkz. bu fonksiyonun docstring'i)."
        )
    long_frame = long_frame.drop(columns=["scan_id"])
    wide, report = pivot_radiomics_long_to_wide(long_frame, ["WT"])
    if wide.empty:
        raise RuntimeError(
            f"Hasta {patient_id!r}: WT bolgesi pivotlanamadi. Rapor: {report}"
        )
    missing = set(final_features) - set(wide.columns)
    if missing:
        raise RuntimeError(
            f"Hasta {patient_id!r}: final_features'in bazilari eksik: {missing}"
        )
    return wide[final_features], f"canli DB (readonly SELECT) -- hasta_id={patient_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Gercek model demosunda DB'ye baglanma, cache'ten rastgele bir hasta kullan.",
    )
    parser.add_argument(
        "--skip-real-model",
        action="store_true",
        help="Yalniz sentetik benchmark calistir, checkpoint/cache'e dokunma.",
    )
    parser.add_argument(
        "--patient-id",
        default=None,
        help="Gercek model demosunda kullanilacak spesifik hasta_id (verilmezse ilk bulunan).",
    )
    args = parser.parse_args()

    run_synthetic_benchmark()

    if not args.skip_real_model:
        run_real_model_demo(use_db=not args.skip_db, patient_id=args.patient_id)


if __name__ == "__main__":
    main()
