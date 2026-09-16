"""`tools/train_cox_week3.py` icin UCTAN UCA sentetik/fixture testleri.

KAPSAM: bu test dosyasi GERCEK DB'ye HIC baglanmaz -- tum testler
sentetik/fixture DataFrame'lerle `run_pipeline()`'i (DB'den bagimsiz
orkestrasyon govdesi) dogrudan cagirir. Gercek C32 verisi DB'de olmadigi
icin (bkz. CLAUDE.md "PyRadiomics C32" uyarisi) gercek egitim burada
YAPILMAZ -- yalniz kod zincirinin (pivot -> build_training_frame ->
run_nested_cv -> fallback/frekans audit'i -> full-pool final model ->
evaluate_external_test) uctan uca DOGRU CALISTIGI kanitlanir.

ASCII-safe desen: tum sentetik string'ler (hasta ID'leri, kaynak adlari)
ASCII -- repodaki ITK/Turkce-karakter path sorunuyla (bkz.
tests/test_radiomics_volume.py) hicbir ilgisi yok, ama proje genelindeki
ASCII-safe fixture konvansiyonu burada da izleniyor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import train_cox_week3 as week3  # noqa: E402  (sys.path ayari sonrasi import)
from pipeline.cox_model import STABLE_FEATURES_ICC60  # noqa: E402


# =====================================================================
# Sentetik veri uretim yardimcilari
# =====================================================================


def _bucket_feature(name: str) -> str:
    if "_shape_" in name:
        return "shape"
    if "_firstorder_" in name:
        return "first_order"
    return "texture"


def _make_synthetic_long_frame(
    patient_ids: list[str], *, source: str, region: str, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """`region`-etiketli bir bolge icin, `week3.ALL_107_FEATURES`'in
    HEPSINI iceren sentetik radyomik uzun-format satirlari uretir --
    gercek PyRadiomics anahtar adlariyla (STABLE_FEATURES_ICC60 +
    EXCLUDED_FIRST_ORDER_ABSOLUTE_SCALE_FEATURES kesisimi)."""

    feature_names = week3.ALL_107_FEATURES
    n = len(patient_ids)
    raw = {name: rng.normal(size=n) for name in feature_names}

    records = []
    for i, patient_id in enumerate(patient_ids):
        shape: dict[str, float] = {}
        first_order: dict[str, float] = {}
        texture: dict[str, float] = {}
        for name in feature_names:
            value = float(raw[name][i])
            bucket = _bucket_feature(name)
            if bucket == "shape":
                shape[name] = value
            elif bucket == "first_order":
                first_order[name] = value
            else:
                texture[name] = value
        records.append(
            {
                "patient_id": patient_id,
                "source": source,
                "tumor_region": region,
                "shape_features": shape,
                "first_order_features": first_order,
                "texture_features": texture,
            }
        )
    return pd.DataFrame(records), raw


def _simulate_survival(
    raw: dict[str, np.ndarray],
    informative_features: list[str],
    coefficients: list[float],
    *,
    rng: np.random.Generator,
    baseline_hazard: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """`test_cox_model.py::_make_synthetic_survival_frame` ile AYNI
    desen -- exponential-baseline hazard + bilinen bilgilendirici
    ozellikler, LASSO/elastic-net'in bunlari GERCEKTEN secebilecegi
    bir sinyal yerlestirir."""

    n = len(next(iter(raw.values())))
    linear_predictor = np.zeros(n)
    for name, beta in zip(informative_features, coefficients):
        linear_predictor += beta * raw[name]

    u = rng.uniform(size=n)
    event_time = -np.log(u) / (baseline_hazard * np.exp(linear_predictor))
    censoring_time = rng.uniform(low=1.0, high=np.percentile(event_time, 90), size=n)
    duration = np.minimum(event_time, censoring_time)
    event = (event_time <= censoring_time).astype(int)
    vital_status = np.where(event == 1, "DECEASED", "ALIVE")
    return duration, vital_status


def _make_patients_frame(
    patient_ids: list[str],
    source: str,
    duration: np.ndarray,
    vital_status: np.ndarray,
    *,
    clinical_raw: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    data = {
        "patient_id": patient_ids,
        "source": source,
        "survival_days": duration,
        "vital_status": vital_status,
    }
    if clinical_raw:
        data.update(clinical_raw)
    frame = pd.DataFrame(data).set_index("patient_id")
    return frame


def _make_synthetic_clinical_raw(
    n: int, *, rng: np.random.Generator, missing_gtr_idh: bool
) -> dict[str, np.ndarray]:
    """`PATIENTS_FRAME_CLINICAL_RAW_COLUMNS` ile AYNI ham kolonları
    (`age`/`gender`/`gtr_over90percent`/`idh1_status`) sentetik üretir.

    `missing_gtr_idh=True` -- TCGA'nın GERÇEK davranışını taklit eder
    (canlı DB, 2026-08-15 ölçümü: TCGA-GBM'de `gtr_over90percent`/
    `idh1_status` %100 NULL) -- `assert_external_test_feasible()`'ın
    GERÇEK veri koşulunu test edebilmesi için."""

    age = rng.integers(20, 86, size=n).astype(float)
    gender = np.where(rng.random(size=n) < 0.6, "Male", "Female")

    if missing_gtr_idh:
        gtr = np.array([None] * n, dtype=object)
        idh = np.array([None] * n, dtype=object)
    else:
        gtr = np.where(rng.random(size=n) < 0.85, np.where(rng.random(size=n) < 0.65, "Y", "N"), None)
        idh_roll = rng.random(size=n)
        idh = np.where(
            idh_roll < 0.80,
            "Wildtype",
            np.where(idh_roll < 0.83, "Mutated", "NOS/NEC"),
        )

    # UPenn'de yas/cinsiyet %100 dolu (canli DB ile dogrulandi); TCGA'da
    # ~%97 -- burada kucuk bir eksik oran taklit ediliyor (missing_gtr_idh
    # bayragiyla AYNI cagriyi TCGA icin kullaniyoruz).
    if missing_gtr_idh:
        age_missing_mask = rng.random(size=n) < 0.03
        age = age.astype(object)
        age[age_missing_mask] = None
        gender = gender.astype(object)
        gender_missing_mask = rng.random(size=n) < 0.03
        gender[gender_missing_mask] = None

    # 2026-08-18 EKLENDİ (Model v2 -- MGMT kovaryatı). UPenn'in GERÇEK
    # dağılımıyla (canlı DB, 2026-08-18): Methylated ~%18,3 / Unmethylated
    # ~%25,4 / Indeterminate ~%4,4 / NULL ~%51,9 (630 kayıtlı hasta).
    mgmt_roll = rng.random(size=n)
    mgmt = np.where(
        mgmt_roll < 0.183,
        "Methylated",
        np.where(
            mgmt_roll < 0.183 + 0.254,
            "Unmethylated",
            np.where(mgmt_roll < 0.183 + 0.254 + 0.044, "Indeterminate", None),
        ),
    )

    return {
        "age": age,
        "gender": gender,
        "gtr_over90percent": gtr,
        "idh1_status": idh,
        "mgmt_status": mgmt,
    }


INFORMATIVE_FEATURES = [
    "original_shape_Sphericity",
    "original_firstorder_Entropy",
    "original_glcm_Contrast",
]
INFORMATIVE_COEFFICIENTS = [0.9, -0.7, 0.5]


def _build_synthetic_cohort(
    *, n_upenn: int, n_tcga: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # NOT: UPenn'in RAW bolge adi "WT" DEGIL "WT_derived" (bkz.
    # pipeline.cox_model.REGION_NAME_ALIASES) -- canonicalize_region_
    # label() bunu kanonik "WT"'ye cevirir, TCGA'nin RAW "WT"'siyle
    # KARISTIRILMAMALI (TCGA native WT/TC uretir, UPenn turetir).
    upenn_ids = [f"SYN-UPENN-{i:04d}" for i in range(n_upenn)]
    upenn_long, upenn_raw = _make_synthetic_long_frame(
        upenn_ids, source="UPenn-GBM", region="WT_derived", rng=rng
    )
    upenn_duration, upenn_vital = _simulate_survival(
        upenn_raw, INFORMATIVE_FEATURES, INFORMATIVE_COEFFICIENTS, rng=rng
    )
    upenn_clinical_raw = _make_synthetic_clinical_raw(n_upenn, rng=rng, missing_gtr_idh=False)
    upenn_patients = _make_patients_frame(
        upenn_ids, "UPenn-GBM", upenn_duration, upenn_vital, clinical_raw=upenn_clinical_raw
    )

    tcga_ids = [f"SYN-TCGA-{i:04d}" for i in range(n_tcga)]
    tcga_long, tcga_raw = _make_synthetic_long_frame(
        tcga_ids, source="TCGA-GBM", region="WT", rng=rng
    )
    tcga_duration, tcga_vital = _simulate_survival(
        tcga_raw, INFORMATIVE_FEATURES, INFORMATIVE_COEFFICIENTS, rng=rng
    )
    # missing_gtr_idh=True -- TCGA-GBM'in GERCEK davranisini taklit eder
    # (GTR/IDH1 %100 NULL, canli DB ile dogrulandi).
    tcga_clinical_raw = _make_synthetic_clinical_raw(n_tcga, rng=rng, missing_gtr_idh=True)
    tcga_patients = _make_patients_frame(
        tcga_ids, "TCGA-GBM", tcga_duration, tcga_vital, clinical_raw=tcga_clinical_raw
    )

    return upenn_long, upenn_patients, tcga_long, tcga_patients


def _make_synthetic_ucsf_clinical_raw(n: int, *, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """2026-08-18 REVİZYON (Barış kararı (a)): UCSF `patients` satırları
    ARTIK HARMONİZE yazılır (db-agent'in `tools/onboard_ucsf_patients.py`
    işi) -- yani UPenn ile AYNI sözlüğü (`Male`/`Female`, `Y`/`N`,
    `Wildtype`/`Mutated`) taşır, UCSF'in HAM sözlüğünü (M/F, GTR/STR/
    biopsy, wildtype/mutasyon-alt-tipi) DEĞİL. `idh1_status`'ta `NOS/NEC`
    HİÇ üretilmiyor -- UCSF'te "test yapılmamış" kategorisi yok, yalnız
    Wildtype/Mutated (decisions/2026-08-18-tek-model-ucsf-harici-test-
    k15-kapanisi.md: kohort IDH filtresi olmadan dondurulur, ~%6,8'i
    Mutated çıkar)."""

    age = rng.integers(20, 86, size=n).astype(float)
    gender = np.where(rng.random(size=n) < 0.55, "Male", "Female")
    gtr = np.where(rng.random(size=n) < 0.637, "Y", "N")  # UPenn Y=%63,2 ile ayni ligde
    idh = np.where(rng.random(size=n) < 0.932, "Wildtype", "Mutated")  # ~%6,8 mutant
    # 2026-08-18 EKLENDİ (Model v2 -- MGMT kovaryatı). UCSF'in GERÇEK
    # dağılımıyla (canlı DB, 2026-08-18): Methylated ~%68,1 / Unmethylated
    # ~%27,8 / NULL ~%4,1 (295 hasta) -- Indeterminate UCSF'te HİÇ yok
    # (db-agent onboarding'de zaten NULL'a çevrilmiş, bkz. map_mgmt()).
    mgmt_roll = rng.random(size=n)
    mgmt = np.where(mgmt_roll < 0.681, "Methylated", np.where(mgmt_roll < 0.959, "Unmethylated", None))
    return {
        "age": age,
        "gender": gender,
        "gtr_over90percent": gtr,
        "idh1_status": idh,
        "mgmt_status": mgmt,
    }


def _build_synthetic_ucsf_cohort(*, n_ucsf: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`_build_synthetic_cohort()`'un UCSF karsiligi -- `pivot_ucsf_wt_
    long_to_wide()`'in `canonical_source()`'a BAGIMLI OLMADIGINI da
    dolayli olarak dogrular (region="WT_derived" -- UPenn konvansiyonu,
    yukaridaki UCSF_WT_RAW_REGION_LABELS varsayimiyla TUTARLI)."""

    rng = np.random.default_rng(seed)
    ucsf_ids = [f"SYN-UCSF-{i:04d}" for i in range(n_ucsf)]
    ucsf_long, ucsf_raw = _make_synthetic_long_frame(
        ucsf_ids, source="UCSF-PDGM", region="WT_derived", rng=rng
    )
    ucsf_duration, ucsf_vital = _simulate_survival(
        ucsf_raw, INFORMATIVE_FEATURES, INFORMATIVE_COEFFICIENTS, rng=rng
    )
    ucsf_clinical_raw = _make_synthetic_ucsf_clinical_raw(n_ucsf, rng=rng)
    ucsf_patients = _make_patients_frame(
        ucsf_ids, "UCSF-PDGM", ucsf_duration, ucsf_vital, clinical_raw=ucsf_clinical_raw
    )
    return ucsf_long, ucsf_patients


def _fast_args(output_dir: Path, **overrides) -> argparse.Namespace:
    """Testte hiz icin KUCULTULMUS grid/split/bootstrap degerleri --
    gercek kosuda protokolun kilitli varsayilanlari (DEFAULT_L1_RATIO_
    GRID/DEFAULT_PENALIZER_GRID, n_bootstrap_stability=200) kullanilir,
    burada YALNIZ test hizi icin degistiriliyor.

    `skip_clinical_arms=True` VARSAYILAN (2026-08-15 EKLENDİ): mevcut
    3-kol testleri klinik kovaryat kollarından ETKİLENMESİN diye --
    yeni klinik kovaryat testleri `skip_clinical_arms=False` ile AÇIKÇA
    override eder (bkz. aşağıdaki `test_run_pipeline_clinical_arms_*`)."""

    defaults = dict(
        sensitivity_threshold=0.7,
        primary_threshold=0.6,
        outer_splits=3,
        inner_splits=2,
        n_bootstrap_stability=12,
        n_bootstrap_external=40,
        seed=7,
        output_dir=output_dir,
        skip_full_107_arm=False,
        exploratory_cluster_stability=False,
        exploratory_cluster_corr_threshold=0.7,
        expected_patient_count=0,  # her testte gercek sayiyla override edilir
        expected_event_count=0,
        allow_unexpected_patient_count=False,
        l1_ratio_grid=[0.5, 1.0],
        penalizer_grid=[0.2, 0.5],
        skip_clinical_arms=True,
        clinical_arms=None,  # 2026-08-18: None -> TUM klinik kollar (mevcut davranis)
        clinical_extra_column_penalizer=0.0,
        # 2026-08-18 EKLENDİ (Barış kararı -- TEK MODEL + UCSF harici test):
        expected_ucsf_patient_count=0,  # her testte gercek sayiyla override edilir
        expected_ucsf_event_count=0,
        allow_unexpected_ucsf_count=False,
        legacy_multi_arm_tcga=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# =====================================================================
# Birim testleri -- dar kapsam
# =====================================================================


def test_raise_if_empty_c32_raises_on_zero_rows() -> None:
    empty = pd.DataFrame(columns=["patient_id", "source", "tumor_region"])
    with pytest.raises(week3.C32DataNotFoundError, match="0 satir"):
        week3.raise_if_empty_c32(empty, segmentation_tool="UPenn-PyRadiomics-107-C32")


def test_raise_if_empty_c32_does_not_raise_on_nonempty() -> None:
    non_empty = pd.DataFrame({"patient_id": ["p1"], "source": ["UPenn-GBM"], "tumor_region": ["WT"]})
    week3.raise_if_empty_c32(non_empty, segmentation_tool="UPenn-PyRadiomics-107-C32")


def test_all_107_features_matches_canonical_composition() -> None:
    assert len(week3.ALL_107_FEATURES) == 107
    assert set(STABLE_FEATURES_ICC60).issubset(set(week3.ALL_107_FEATURES))


def test_check_training_pool_counts_hard_fails_on_mismatch_without_override() -> None:
    with pytest.raises(week3.TrainingPoolCountMismatchError):
        week3.check_training_pool_counts(
            600, 580, expected_patients=611, expected_events=585, allow_mismatch=False
        )


def test_check_training_pool_counts_warns_and_continues_with_override(capsys) -> None:
    week3.check_training_pool_counts(
        600, 580, expected_patients=611, expected_events=585, allow_mismatch=True
    )
    captured = capsys.readouterr()
    assert "UYARI" in captured.err


def test_check_training_pool_counts_passes_on_exact_match() -> None:
    week3.check_training_pool_counts(
        611, 585, expected_patients=611, expected_events=585, allow_mismatch=False
    )


# ---------------------------------------------------------------------
# Bolge-farkinda egitim havuzu kapisi (2026-08-19, Barış talimati --
# bkz. gunluk-rapor-2026-08-18.md Riskler-1 ve `check_training_pool_
# counts()` docstring'i). Kurallar:
#   * WT-only          -> SERT 611/585
#   * WT+TC (beyanli)  -> 609/583 + AYNEN o 2 hasta ise GECER, loglanir
#   * beyansiz cok-bolge -> SERT durur
#   * sayi tutup KIMLIK tutmuyorsa -> SERT durur
# ---------------------------------------------------------------------

DECLARED_WT_TC = week3.DECLARED_REGION_SHORTFALLS[("WT", "TC")]


def test_declared_wt_tc_shortfall_matches_measured_live_values() -> None:
    """Beyan tablosu, 2026-08-19 canli olcumuyle BIREBIR ayni olmali.

    Olculen (salt-okunur, C32 onbellegi + canli DB hasta tablosu):
    WT-only 611/585, WT+TC 609/583, dusen UPENN-GBM-00354 / -00397.
    Bu test bir sabitin sessizce degistirilmesini yakalar.
    """

    assert DECLARED_WT_TC.n_patients == 609
    assert DECLARED_WT_TC.n_events == 583
    assert DECLARED_WT_TC.dropped_patient_ids == (
        "UPENN-GBM-00354",
        "UPENN-GBM-00397",
    )
    # Beyan edilen azalma, dusen hasta sayisiyla TUTARLI olmali
    # (611 - 609 == 2 dusen hasta).
    assert 611 - DECLARED_WT_TC.n_patients == len(DECLARED_WT_TC.dropped_patient_ids)


def test_check_training_pool_counts_wt_only_still_hard_fails_on_609() -> None:
    """WT-only'de 609/583 hala HATA -- bolge-farkindalik WT'yi gevsetmez."""

    with pytest.raises(week3.TrainingPoolCountMismatchError):
        week3.check_training_pool_counts(
            609,
            583,
            expected_patients=611,
            expected_events=585,
            regions=("WT",),
        )


def test_check_training_pool_counts_accepts_declared_wt_tc_shortfall(capsys) -> None:
    week3.check_training_pool_counts(
        609,
        583,
        expected_patients=611,
        expected_events=585,
        regions=("WT", "TC"),
        dropped_patient_ids=["UPENN-GBM-00397", "UPENN-GBM-00354"],
    )
    captured = capsys.readouterr()
    assert "BEYAN EDILMIS HAVUZ AZALMASI" in captured.err
    # Dusen hastalar ACIKCA loglanmali (Barış'in talimatinin ikinci yarisi).
    assert "UPENN-GBM-00354" in captured.err
    assert "UPENN-GBM-00397" in captured.err
    # Rapor sayisinin 611/585 DEGIL 609/583 oldugu hatirlatilmali.
    assert "609/583" in captured.err


def test_check_training_pool_counts_wt_tc_rejects_missing_identity_evidence() -> None:
    """FAIL-OPEN REGRESYONU (2026-08-19 Codex HIGH bulgusu #1).

    Beyan edilmis azalma dalinda KIMLIK KANITI verilmezse kapi
    GECIRMEMELI. Onceki surumde `ids_match = observed_dropped is None or
    ...` idi -> kanit yoksa kontrol OTOMATIK BASARILI sayiliyordu.
    Gercek senaryo: havuz 609/583'e `missing_features` DISINDA bir
    gerekcelerle duserse `.get("missing_features")` `None` doner ve
    kimlik hic dogrulanmadan gecerdi.
    """

    with pytest.raises(week3.TrainingPoolCountMismatchError) as excinfo:
        week3.check_training_pool_counts(
            609,
            583,
            expected_patients=611,
            expected_events=585,
            regions=("WT", "TC"),
            dropped_patient_ids=None,  # KANIT YOK
        )
    assert "KIMLIKLERINI SAGLAMADI" in str(excinfo.value)


def test_all_dropped_patient_ids_unions_every_reason() -> None:
    """`all_dropped_patient_ids()` TUM gerekceleri birlestirmeli --
    yalniz `missing_features`'i degil (Codex HIGH #1'in kok nedeni)."""

    class _Report:
        dropped_patient_ids = {
            "missing_features": ["B", "A"],
            "missing_duration": ["C"],
            "missing_passthrough": ["A"],  # mukerrer, tekillesmeli
        }

    assert week3.all_dropped_patient_ids(_Report()) == ["A", "B", "C"]

    class _Empty:
        dropped_patient_ids: dict = {}

    assert week3.all_dropped_patient_ids(_Empty()) == []


def test_check_training_pool_counts_rejects_shortfall_from_wrong_reason() -> None:
    """Sayi 609/583 tutuyor ama dusen hastalar BASKA bir gerekceden
    geliyorsa (kimlikler beyanla uyusmuyorsa) kapi SERT durmali."""

    with pytest.raises(week3.TrainingPoolCountMismatchError) as excinfo:
        week3.check_training_pool_counts(
            609,
            583,
            expected_patients=611,
            expected_events=585,
            regions=("WT", "TC"),
            # dogru SAYIDA ama BASKA hastalar (orn. survival_days NULL)
            dropped_patient_ids=["UPENN-GBM-00010", "UPENN-GBM-00020"],
        )
    assert "DUSEN HASTALAR FARKLI" in str(excinfo.value)


def test_check_training_pool_counts_accepts_generator_regions(capsys) -> None:
    """`regions` tek kullanimlik bir iterator ise log etiketi BOS
    CIKMAMALI (2026-08-19 Codex LOW bulgusu #5)."""

    week3.check_training_pool_counts(
        609,
        583,
        expected_patients=611,
        expected_events=585,
        regions=(r for r in ("WT", "TC")),  # generator
        dropped_patient_ids=["UPENN-GBM-00354", "UPENN-GBM-00397"],
    )
    captured = capsys.readouterr()
    assert "bolgeler       : WT+TC" in captured.err


def test_check_training_pool_counts_wt_tc_rejects_undeclared_count() -> None:
    """WT+TC'de beyan 609/583; 608/582 olculurse SESSIZCE gecmez."""

    with pytest.raises(week3.TrainingPoolCountMismatchError) as excinfo:
        week3.check_training_pool_counts(
            608,
            582,
            expected_patients=611,
            expected_events=585,
            regions=("WT", "TC"),
        )
    assert "BEYAN EDILEN havuz" in str(excinfo.value)


def test_check_training_pool_counts_wt_tc_rejects_same_count_different_patients() -> None:
    """Sayi tutuyor ama BASKA 2 hasta dustuyse kapi yine durur."""

    with pytest.raises(week3.TrainingPoolCountMismatchError) as excinfo:
        week3.check_training_pool_counts(
            609,
            583,
            expected_patients=611,
            expected_events=585,
            regions=("WT", "TC"),
            dropped_patient_ids=["UPENN-GBM-00001", "UPENN-GBM-00002"],
        )
    assert "DUSEN HASTALAR FARKLI" in str(excinfo.value)


def test_check_training_pool_counts_rejects_undeclared_region_set() -> None:
    """Beyani OLMAYAN cok-bolgeli bir kume (orn. WT+ET) sessizce gecemez."""

    assert ("WT", "ET") not in week3.DECLARED_REGION_SHORTFALLS
    with pytest.raises(week3.TrainingPoolCountMismatchError) as excinfo:
        week3.check_training_pool_counts(
            605,
            580,
            expected_patients=611,
            expected_events=585,
            regions=("WT", "ET"),
        )
    assert "BEYAN EDILMIS bir azalma YOK" in str(excinfo.value)


def test_check_training_pool_counts_exact_match_wins_over_region_logic() -> None:
    """Cok-bolgeli bir varyant tam 611/585 verirse (azalma olmazsa) gecer."""

    week3.check_training_pool_counts(
        611,
        585,
        expected_patients=611,
        expected_events=585,
        regions=("WT", "TC"),
    )


def test_resolve_declared_region_shortfall_returns_none_for_base_pool() -> None:
    assert week3.resolve_declared_region_shortfall(None) is None
    assert week3.resolve_declared_region_shortfall(("WT",)) is None
    assert week3.resolve_declared_region_shortfall(["WT"]) is None
    assert week3.resolve_declared_region_shortfall(("WT", "TC")) is DECLARED_WT_TC
    assert week3.resolve_declared_region_shortfall(("WT", "ET")) is None


def test_build_training_frame_records_dropped_patient_ids() -> None:
    """`TrainingFrameReport.dropped_patient_ids` gercekten KIMLIK tasiyor.

    Kapinin kimlik dogrulamasi bu alana dayaniyor -- alan sessizce bos
    kalirsa `ids_match` her zaman True olur ve koruma kaybolur.
    """

    from pipeline.cox_model import build_training_frame

    feature_frame = pd.DataFrame(
        {"WT__f": [1.0, 2.0, 3.0], "TC__f": [1.0, None, 3.0]},
        index=["p1", "p2", "p3"],
    )
    patients_frame = pd.DataFrame(
        {
            "source": ["UPenn-GBM"] * 3,
            "survival_days": [100, 200, 300],
            "vital_status": ["DECEASED", "DECEASED", "ALIVE"],
        },
        index=["p1", "p2", "p3"],
    )
    frame, report = build_training_frame(
        feature_frame, patients_frame, check_combat_identity=False
    )
    assert report.n_output_rows == 2
    assert report.dropped_missing_features == 1
    assert report.dropped_patient_ids["missing_features"] == ["p2"]
    # Dusme OLMAYAN gerekceler sozlukte BULUNMAZ (bos liste yazilmaz).
    assert "missing_duration" not in report.dropped_patient_ids
    assert "unrecognized_vital_status" not in report.dropped_patient_ids


def test_build_external_test_frame_derives_event_and_drops_unrecognized() -> None:
    feature_frame = pd.DataFrame(
        {"WT__feat_a": [1.0, 2.0, 3.0, 4.0]}, index=["t1", "t2", "t3", "t4"]
    )
    patients_frame = pd.DataFrame(
        {
            "survival_days": [100, 200, None, 400],
            "vital_status": ["DECEASED", "ALIVE", "DECEASED", "Lost to Follow-up"],
        },
        index=["t1", "t2", "t3", "t4"],
    )

    frame, report = week3.build_external_test_frame(feature_frame, patients_frame)

    assert report.n_input_rows == 4
    assert report.dropped_unrecognized_vital_status == {"Lost to Follow-up": 1}
    assert report.dropped_missing_duration == 1
    assert report.n_output_rows == 2
    assert list(frame["event"]) == [1, 0]


def test_compute_correlation_clusters_groups_correlated_features() -> None:
    rng = np.random.default_rng(1)
    n = 60
    base = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "a": base + rng.normal(scale=0.01, size=n),
            "b": base + rng.normal(scale=0.01, size=n),  # a ile neredeyse ozdes
            "c": rng.normal(size=n),  # bagimsiz
        }
    )
    clusters = week3.compute_correlation_clusters(frame, ["a", "b", "c"], corr_threshold=0.7)
    assert clusters["a"] == clusters["b"]
    assert clusters["c"] != clusters["a"]


def test_exploratory_cluster_selection_frequency_flags_any_selected() -> None:
    clusters = {"a": 0, "b": 0, "c": 1}
    nested_cv_result = pd.DataFrame(
        {
            "fold": [0, 1],
            "selected_features": [["a"], ["b"]],
        }
    )
    result = week3.exploratory_cluster_selection_frequency(clusters, nested_cv_result)
    cluster0 = result.loc[result["cluster_id"] == 0].iloc[0]
    cluster1 = result.loc[result["cluster_id"] == 1].iloc[0]
    assert cluster0["n_folds_with_any_selected"] == 2  # her fold'da a YA DA b secildi
    assert cluster1["n_folds_with_any_selected"] == 0


# =====================================================================
# Uctan uca smoke testi -- sentetik veriyle TUM zincir
# =====================================================================


def test_run_pipeline_end_to_end_smoke_synthetic(tmp_path: Path) -> None:
    upenn_long, upenn_patients, tcga_long, tcga_patients = _build_synthetic_cohort(
        n_upenn=140, n_tcga=45, seed=123
    )
    expected_events = int((upenn_patients["vital_status"] == "DECEASED").sum())

    args = _fast_args(
        tmp_path,
        expected_patient_count=len(upenn_patients),
        expected_event_count=expected_events,
        exploratory_cluster_stability=True,
    )

    exit_code = week3.run_pipeline(
        upenn_long=upenn_long,
        upenn_patients=upenn_patients,
        tcga_long=tcga_long,
        tcga_patients=tcga_patients,
        args=args,
        output_dir=tmp_path,
    )
    assert exit_code == 0

    expected_arms = {
        "primary_wt93_icc60_th06",
        "sensitivity_wt93_icc60_th07",
        "sensitivity_wt107_full_th06",
    }

    fold_results = pd.read_csv(tmp_path / "week3_fold_results.csv")
    assert set(fold_results["arm"]) == expected_arms
    assert "used_fallback" in fold_results.columns
    assert fold_results["fold"].notna().all()
    assert (fold_results.groupby("arm")["fold"].count() == args.outer_splits).all()

    freq_summary = pd.read_csv(tmp_path / "week3_selection_frequency.csv")
    assert not freq_summary.empty
    assert set(freq_summary["arm"]) == expected_arms
    assert freq_summary["mean_frequency"].between(0.0, 1.0).all()

    external = pd.read_csv(tmp_path / "week3_external_test.csv")
    assert len(external) == 3
    assert set(external["arm"]) == expected_arms
    finite_ci = external.dropna(subset=["ci_lower", "ci_upper"])
    assert (finite_ci["ci_lower"] <= finite_ci["c_index"] + 1e-9).all()
    assert (finite_ci["c_index"] <= finite_ci["ci_upper"] + 1e-9).all()
    assert (external["n_final_features"] > 0).all()

    metadata = json.loads((tmp_path / "week3_run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["segmentation_tool_upenn_c32"] == "UPenn-PyRadiomics-107-C32"
    assert metadata["segmentation_tool_tcga_c32"] == "TCGA-ground-truth-C32"
    assert metadata["method_name"] == "bootstrap selection-frequency filtering"
    assert metadata["primary_region"] == "WT"
    assert len(metadata["arms"]) == 3
    assert "combat_identity_note" in metadata

    exploratory = pd.read_csv(tmp_path / "week3_exploratory_cluster_stability.csv")
    assert not exploratory.empty
    assert exploratory["frequency_any_selected"].between(0.0, 1.0).all()


def test_run_pipeline_raises_on_training_pool_count_mismatch(tmp_path: Path) -> None:
    upenn_long, upenn_patients, tcga_long, tcga_patients = _build_synthetic_cohort(
        n_upenn=60, n_tcga=25, seed=99
    )
    args = _fast_args(
        tmp_path,
        expected_patient_count=999,  # bilincli yanlis -- gate tetiklenmeli
        expected_event_count=999,
        skip_full_107_arm=True,
    )

    with pytest.raises(week3.TrainingPoolCountMismatchError):
        week3.run_pipeline(
            upenn_long=upenn_long,
            upenn_patients=upenn_patients,
            tcga_long=tcga_long,
            tcga_patients=tcga_patients,
            args=args,
            output_dir=tmp_path,
        )


def test_run_pipeline_skip_full_107_arm_produces_two_arms(tmp_path: Path) -> None:
    upenn_long, upenn_patients, tcga_long, tcga_patients = _build_synthetic_cohort(
        n_upenn=70, n_tcga=30, seed=55
    )
    expected_events = int((upenn_patients["vital_status"] == "DECEASED").sum())
    args = _fast_args(
        tmp_path,
        expected_patient_count=len(upenn_patients),
        expected_event_count=expected_events,
        skip_full_107_arm=True,
    )

    exit_code = week3.run_pipeline(
        upenn_long=upenn_long,
        upenn_patients=upenn_patients,
        tcga_long=tcga_long,
        tcga_patients=tcga_patients,
        args=args,
        output_dir=tmp_path,
    )
    assert exit_code == 0

    external = pd.read_csv(tmp_path / "week3_external_test.csv")
    assert set(external["arm"]) == {"primary_wt93_icc60_th06", "sensitivity_wt93_icc60_th07"}
    assert not (tmp_path / "week3_exploratory_cluster_stability.csv").exists()


# =====================================================================
# 2026-08-15 -- klinik kovaryat destegi (yeni kollar: clinical_base /
# radiomics_clinical / radiomics_clinical_gtr / idh_sensitivity /
# who2021_wildtype)
# =====================================================================


def _make_raw_clinical_patients_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [45.0, 60.0, 70.0, np.nan],
            "gender": ["Male", "Female", "Male", None],
            "gtr_over90percent": ["Y", "N", None, "Y"],
            "idh1_status": ["Wildtype", "Mutated", "NOS/NEC", None],
            # 2026-08-18 EKLENDİ (Model v2 -- MGMT). p1=Methylated,
            # p2=Unmethylated, p3=Indeterminate (test yapıldı, eksik
            # sayılır), p4=ham NULL (eksik sayılır) -- IDH'nin p1..p4
            # desenindeki (Wildtype/Mutated/NOS-NEC/NULL) BİREBİR AYNI
            # yapıyı izler.
            "mgmt_status": ["Methylated", "Unmethylated", "Indeterminate", None],
        },
        index=["p1", "p2", "p3", "p4"],
    )


def test_build_clinical_covariate_frame_encodes_age_gender_linearly_and_binary() -> None:
    patients_frame = _make_raw_clinical_patients_frame()

    result = week3.build_clinical_covariate_frame(patients_frame, include_gender=True)

    assert list(result.loc[["p1", "p2", "p3"], week3.CLINICAL_AGE_COLUMN]) == [45.0, 60.0, 70.0]
    assert pd.isna(result.loc["p4", week3.CLINICAL_AGE_COLUMN])  # ham NULL, HAM olarak kalir
    assert result.loc["p1", week3.CLINICAL_GENDER_MALE_COLUMN] == 1.0
    assert result.loc["p2", week3.CLINICAL_GENDER_MALE_COLUMN] == 0.0
    # p4'un cinsiyeti NULL -- SESSIZCE 0'a dusurulmez, NaN kalir (gorev
    # talimati: "NULL SESSIZCE 0'a DUSURULMEZ").
    assert pd.isna(result.loc["p4", week3.CLINICAL_GENDER_MALE_COLUMN])


def test_build_clinical_covariate_frame_gtr_missing_indicator_correct() -> None:
    """(d) `gtr_missing` gostergesinin dogru uretildigi -- gorev
    talimatinin acikca istedigi test."""

    patients_frame = _make_raw_clinical_patients_frame()

    result = week3.build_clinical_covariate_frame(patients_frame, include_gtr=True)

    # p1=Y, p2=N, p3=eksik, p4=Y
    assert result.loc["p1", week3.CLINICAL_GTR_Y_COLUMN] == 1.0
    assert result.loc["p1", week3.CLINICAL_GTR_MISSING_COLUMN] == 0.0
    assert result.loc["p2", week3.CLINICAL_GTR_Y_COLUMN] == 0.0
    assert result.loc["p2", week3.CLINICAL_GTR_MISSING_COLUMN] == 0.0
    # eksik hasta DUSURULMEZ -- sabit-doldurma (GTR_Y=0) + ayri gosterge (GTR_MISSING=1).
    assert result.loc["p3", week3.CLINICAL_GTR_Y_COLUMN] == 0.0
    assert result.loc["p3", week3.CLINICAL_GTR_MISSING_COLUMN] == 1.0
    assert result.loc["p4", week3.CLINICAL_GTR_Y_COLUMN] == 1.0
    assert result.loc["p4", week3.CLINICAL_GTR_MISSING_COLUMN] == 0.0
    # HICBIR satir eksik/NaN kalmamali (GTR NaN URETMEMELI).
    assert result[week3.CLINICAL_GTR_Y_COLUMN].isna().sum() == 0
    assert result[week3.CLINICAL_GTR_MISSING_COLUMN].isna().sum() == 0


def test_build_clinical_covariate_frame_idh_missing_indicator_correct() -> None:
    """(K15 KAPANDI, 2026-08-18) `clinical_idh_missing` GTR'nin gosterge
    deseniyle BIREBIR ayni: NOS/NEC (test yapilmamis) VE ham NULL -> 1,
    Wildtype/Mutated -> 0. `clinical_idh_mutant` NOS/NEC'te (ve NULL'da)
    0'a (referans wildtype) FILLNA edilir -- gorev talimatinin acikca
    istedigi test."""

    patients_frame = _make_raw_clinical_patients_frame()

    result = week3.build_clinical_covariate_frame(patients_frame, include_idh=True)

    # p1=Wildtype, p2=Mutated, p3=NOS/NEC, p4=NULL
    assert result.loc["p1", week3.CLINICAL_IDH_MUTANT_COLUMN] == 0.0
    assert result.loc["p1", week3.CLINICAL_IDH_MISSING_COLUMN] == 0.0
    assert result.loc["p2", week3.CLINICAL_IDH_MUTANT_COLUMN] == 1.0
    assert result.loc["p2", week3.CLINICAL_IDH_MISSING_COLUMN] == 0.0
    # NOS/NEC -> mutant=0 (referans wildtype), missing=1 -- ARTIK NaN DEĞİL.
    assert result.loc["p3", week3.CLINICAL_IDH_MUTANT_COLUMN] == 0.0
    assert result.loc["p3", week3.CLINICAL_IDH_MISSING_COLUMN] == 1.0
    # ham NULL -- ayni sekilde eksik sayilir.
    assert result.loc["p4", week3.CLINICAL_IDH_MUTANT_COLUMN] == 0.0
    assert result.loc["p4", week3.CLINICAL_IDH_MISSING_COLUMN] == 1.0
    # HICBIR satir eksik/NaN kalmamali (GTR'deki "NaN URETMEMELI" ilkesiyle AYNI).
    assert result[week3.CLINICAL_IDH_MUTANT_COLUMN].isna().sum() == 0
    assert result[week3.CLINICAL_IDH_MISSING_COLUMN].isna().sum() == 0


def test_build_clinical_covariate_frame_raises_on_unrecognized_value() -> None:
    patients_frame = _make_raw_clinical_patients_frame()
    patients_frame.loc["p1", "gender"] = "Non-binary"  # semaya uymuyor

    with pytest.raises(week3.UnrecognizedClinicalValueError, match="gender"):
        week3.build_clinical_covariate_frame(patients_frame, include_gender=True)


def test_filter_to_idh_known_cohort_wildtype_or_mutated_drops_nos_nec() -> None:
    patients_frame = _make_raw_clinical_patients_frame()

    filtered = week3.filter_to_idh_known_cohort(patients_frame, keep="wildtype_or_mutated")

    assert set(filtered.index) == {"p1", "p2"}  # p3=NOS/NEC, p4=NULL disarida


def test_filter_to_idh_known_cohort_wildtype_only_keeps_wildtype_alone() -> None:
    patients_frame = _make_raw_clinical_patients_frame()

    filtered = week3.filter_to_idh_known_cohort(patients_frame, keep="wildtype_only")

    assert set(filtered.index) == {"p1"}


def test_assert_external_test_feasible_raises_when_coverage_zero() -> None:
    """(c) TCGA'da %100 eksik kovaryatlı kolun harici teste girmeye
    çalışınca net bir hata verdiği -- görev talimatının açıkça istediği
    test."""

    tcga_clinical_frame = pd.DataFrame(
        {week3.CLINICAL_GTR_Y_COLUMN: [np.nan, np.nan, np.nan]}, index=["t1", "t2", "t3"]
    )

    with pytest.raises(week3.ExternalTestNotSupportedError, match="TAMAMEN eksik"):
        week3.assert_external_test_feasible(
            "radiomics_clinical_gtr", tcga_clinical_frame, [week3.CLINICAL_GTR_Y_COLUMN]
        )


def test_assert_external_test_feasible_passes_when_coverage_positive() -> None:
    tcga_clinical_frame = pd.DataFrame(
        {week3.CLINICAL_AGE_COLUMN: [45.0, np.nan, 60.0]}, index=["t1", "t2", "t3"]
    )
    # hata firlatmamali
    week3.assert_external_test_feasible(
        "clinical_base", tcga_clinical_frame, [week3.CLINICAL_AGE_COLUMN]
    )


def test_assert_external_test_feasible_raises_when_column_missing_entirely() -> None:
    tcga_clinical_frame = pd.DataFrame({"other_col": [1.0, 2.0]}, index=["t1", "t2"])

    with pytest.raises(week3.ExternalTestNotSupportedError, match="YOK"):
        week3.assert_external_test_feasible(
            "idh_sensitivity", tcga_clinical_frame, [week3.CLINICAL_IDH_MUTANT_COLUMN]
        )


def test_run_modeling_arm_clinical_covariates_bypass_selection(tmp_path: Path) -> None:
    """(a) klinik kovaryatın elastic-net/stabilite seçimden GEÇMEDİĞİ --
    görev talimatının açıkça istediği test. Klinik kovaryat HER ZAMAN
    final modelde kalmalı (extra_columns'ta) ama `final_model.
    final_features` (radyomik seçim sonucu) İÇİNDE HİÇ görünmemeli."""

    upenn_long, upenn_patients, tcga_long, tcga_patients = _build_synthetic_cohort(
        n_upenn=90, n_tcga=30, seed=71
    )
    upenn_wide, _ = week3.pivot_radiomics_long_to_wide(upenn_long, regions=["WT"])
    tcga_wide, _ = week3.pivot_radiomics_long_to_wide(tcga_long, regions=["WT"])
    feature_columns = week3.select_feature_columns(upenn_wide, STABLE_FEATURES_ICC60, region="WT")

    upenn_clinical = week3.build_clinical_covariate_frame(upenn_patients, include_gender=True)
    tcga_clinical = week3.build_clinical_covariate_frame(tcga_patients, include_gender=True)
    upenn_patients_c = upenn_patients.join(upenn_clinical)
    tcga_patients_c = tcga_patients.join(tcga_clinical)

    extra_columns = [week3.CLINICAL_AGE_COLUMN, week3.CLINICAL_GENDER_MALE_COLUMN]
    training_frame, _ = week3.build_training_frame(
        upenn_wide[feature_columns],
        upenn_patients_c,
        check_combat_identity=True,
        passthrough_columns=extra_columns,
    )
    external_frame, _ = week3.build_external_test_frame(
        tcga_wide[feature_columns], tcga_patients_c, passthrough_columns=extra_columns
    )

    arm_result = week3.run_modeling_arm(
        "test_clinical_arm",
        training_frame,
        feature_columns,
        external_frame,
        stability_frequency_threshold=0.6,
        outer_splits=3,
        inner_splits=2,
        n_bootstrap_stability=10,
        n_bootstrap_external=30,
        seed=13,
        l1_ratio_grid=(1.0,),
        penalizer_grid=(0.3,),
        extra_columns=extra_columns,
        extra_column_penalizer=0.0,
        clinical_standardize_columns=[week3.CLINICAL_AGE_COLUMN],
        run_external_test=True,
    )

    # klinik kovaryatlar radyomik "final_features" seçim listesinde ASLA
    # gorunmemeli -- extra_columns mekanizmasi seciMden BAGIMSIZ.
    assert set(extra_columns).isdisjoint(set(arm_result.final_model.final_features))
    for fold_row in arm_result.nested_cv_fold_results.itertuples():
        assert set(extra_columns).isdisjoint(set(fold_row.selected_features))
    # AMA final fit edilmis modelde HER ZAMAN mevcut olmali (zorunlu kovaryat).
    fitted_params_index = set(arm_result.final_model.fitted_model.params_.index)
    assert set(extra_columns).issubset(fitted_params_index)
    assert arm_result.extra_columns == extra_columns
    assert arm_result.run_external_test is True


def test_run_pipeline_clinical_arms_end_to_end_smoke(tmp_path: Path) -> None:
    """5 yeni klinik kovaryat kolunun (clinical_base/radiomics_clinical/
    radiomics_clinical_gtr/idh_sensitivity/who2021_wildtype) uctan uca
    calistigini, dogru run_external_test bayraklarini ve GTR/IDH
    kollarinin harici testi BILINCLI olarak atladigini dogrular."""

    upenn_long, upenn_patients, tcga_long, tcga_patients = _build_synthetic_cohort(
        n_upenn=90, n_tcga=30, seed=81
    )
    expected_events = int((upenn_patients["vital_status"] == "DECEASED").sum())
    args = _fast_args(
        tmp_path,
        expected_patient_count=len(upenn_patients),
        expected_event_count=expected_events,
        skip_full_107_arm=True,
        skip_clinical_arms=False,  # bu testin AMACI -- klinik kollari AC
    )

    exit_code = week3.run_pipeline(
        upenn_long=upenn_long,
        upenn_patients=upenn_patients,
        tcga_long=tcga_long,
        tcga_patients=tcga_patients,
        args=args,
        output_dir=tmp_path,
    )
    assert exit_code == 0

    expected_clinical_arms = {
        "clinical_base",
        "radiomics_clinical",
        "radiomics_clinical_gtr",
        "idh_sensitivity",
        "who2021_wildtype",
    }
    external = pd.read_csv(tmp_path / "week3_external_test.csv")
    assert expected_clinical_arms.issubset(set(external["arm"]))

    external_indexed = external.set_index("arm")
    # gorev talimati tablosuyla BIREBIR: hangi kollar harici test ALIR/ALMAZ.
    assert bool(external_indexed.loc["clinical_base", "run_external_test"]) is True
    assert bool(external_indexed.loc["radiomics_clinical", "run_external_test"]) is True
    assert bool(external_indexed.loc["radiomics_clinical_gtr", "run_external_test"]) is False
    assert bool(external_indexed.loc["idh_sensitivity", "run_external_test"]) is False
    assert bool(external_indexed.loc["who2021_wildtype", "run_external_test"]) is False

    # run_external_test=False olan kollarda c_index NaN + skipped=True olmali
    # (SESSIZ degil, ACIKCA isaretli) -- gorev talimati: "sessizce NaN
    # uretmesin".
    for arm_name in ("radiomics_clinical_gtr", "idh_sensitivity", "who2021_wildtype"):
        row = external_indexed.loc[arm_name]
        assert pd.isna(row["c_index"])
        assert bool(row["skipped"]) is True

    # run_external_test=True olan kollarda GERCEK bir CI olmali.
    for arm_name in ("clinical_base", "radiomics_clinical"):
        row = external_indexed.loc[arm_name]
        assert not pd.isna(row["c_index"])
        assert row["ci_lower"] <= row["c_index"] + 1e-9
        assert row["c_index"] <= row["ci_upper"] + 1e-9

    # clinical_base radyomik ICERMEMELI (n_final_features radyomik-yalniz
    # sayimidir, 0 olmali -- klinik kovaryatlar ayri bir kolonda raporlanir).
    assert external_indexed.loc["clinical_base", "n_final_features"] == 0
    assert "clinical_age" in external_indexed.loc["clinical_base", "clinical_extra_columns"]

    metadata = json.loads((tmp_path / "week3_run_metadata.json").read_text(encoding="utf-8"))
    metadata_arm_names = {arm["name"] for arm in metadata["arms"]}
    assert expected_clinical_arms.issubset(metadata_arm_names)
    assert "clinical_covariate_note" in metadata


def test_run_pipeline_skip_clinical_arms_true_omits_new_arms(tmp_path: Path) -> None:
    """Regresyon koruması: `skip_clinical_arms=True` (varsayılan) iken
    5 yeni kol HİÇ üretilmemeli -- mevcut 3 kolun davranışı DEĞİŞMEMELİ."""

    upenn_long, upenn_patients, tcga_long, tcga_patients = _build_synthetic_cohort(
        n_upenn=60, n_tcga=25, seed=91
    )
    expected_events = int((upenn_patients["vital_status"] == "DECEASED").sum())
    args = _fast_args(
        tmp_path,
        expected_patient_count=len(upenn_patients),
        expected_event_count=expected_events,
        skip_full_107_arm=True,
        skip_clinical_arms=True,
    )

    exit_code = week3.run_pipeline(
        upenn_long=upenn_long,
        upenn_patients=upenn_patients,
        tcga_long=tcga_long,
        tcga_patients=tcga_patients,
        args=args,
        output_dir=tmp_path,
    )
    assert exit_code == 0

    external = pd.read_csv(tmp_path / "week3_external_test.csv")
    assert set(external["arm"]) == {"primary_wt93_icc60_th06", "sensitivity_wt93_icc60_th07"}


# =====================================================================
# `--clinical-arms` seçici bayrağı (2026-08-18 EKLENDİ, K15 gerekçesi)
# =====================================================================


def test_select_clinical_arm_configs_none_returns_all() -> None:
    """Bayrak verilmediğinde (None) mevcut davranış DEĞİŞMEZ: 5 kolun
    tamamı, kanonik sırayla döner."""

    assert week3.select_clinical_arm_configs(None) == week3.CLINICAL_ARM_CONFIGS


def test_select_clinical_arm_configs_filters_and_keeps_canonical_order() -> None:
    """Adı verilen kollar seçilir; sıra KULLANICININ yazdığı sıra değil,
    `CLINICAL_ARM_CONFIGS` içindeki kanonik sıradır."""

    selected = week3.select_clinical_arm_configs(
        ["radiomics_clinical", "clinical_base"]  # BİLEREK ters sırada
    )
    assert [config.name for config in selected] == ["clinical_base", "radiomics_clinical"]


def test_select_clinical_arm_configs_deduplicates() -> None:
    """Mükerrer ad iki kez koşuya sokmaz."""

    selected = week3.select_clinical_arm_configs(["clinical_base", "clinical_base"])
    assert [config.name for config in selected] == ["clinical_base"]


def test_select_clinical_arm_configs_rejects_unknown_name() -> None:
    """Bilinmeyen ad SESSİZCE atlanmaz -- patlar (argparse `choices`
    dışında ikinci savunma hattı)."""

    with pytest.raises(ValueError, match="bilinmeyen kol adı"):
        week3.select_clinical_arm_configs(["clinical_base", "yok_boyle_bir_kol"])


def test_select_clinical_arm_configs_excludes_idh_arms_for_k15() -> None:
    """K15 koruması: bugünkü koşuda kullanılan üçlü seçim, NOS/NEC'i
    dışlayan iki kolu (`idh_sensitivity`/`who2021_wildtype`) GERÇEKTEN
    dışarıda bırakmalı."""

    selected = week3.select_clinical_arm_configs(
        ["clinical_base", "radiomics_clinical", "radiomics_clinical_gtr"]
    )
    names = {config.name for config in selected}
    assert names == {"clinical_base", "radiomics_clinical", "radiomics_clinical_gtr"}
    assert "idh_sensitivity" not in names
    assert "who2021_wildtype" not in names
    # ve dışlananların gerçekten IDH kohort filtresi taşıdığını doğrula
    excluded = [c for c in week3.CLINICAL_ARM_CONFIGS if c.name not in names]
    assert all(config.idh_cohort_filter is not None for config in excluded)


def test_run_pipeline_clinical_arms_subset_runs_only_requested(tmp_path: Path) -> None:
    """Uçtan uca: `clinical_arms=['clinical_base']` iken çıktıda YALNIZ
    o klinik kol görünür -- diğer 4 kol HİÇ koşulmaz."""

    upenn_long, upenn_patients, tcga_long, tcga_patients = _build_synthetic_cohort(
        n_upenn=60, n_tcga=25, seed=91
    )
    expected_events = int((upenn_patients["vital_status"] == "DECEASED").sum())
    args = _fast_args(
        tmp_path,
        expected_patient_count=len(upenn_patients),
        expected_event_count=expected_events,
        skip_full_107_arm=True,
        skip_clinical_arms=False,
        clinical_arms=["clinical_base"],
    )

    exit_code = week3.run_pipeline(
        upenn_long=upenn_long,
        upenn_patients=upenn_patients,
        tcga_long=tcga_long,
        tcga_patients=tcga_patients,
        args=args,
        output_dir=tmp_path,
    )
    assert exit_code == 0

    folds = pd.read_csv(tmp_path / "week3_fold_results.csv")
    arms = set(folds["arm"])
    assert "clinical_base" in arms
    for not_requested in (
        "radiomics_clinical",
        "radiomics_clinical_gtr",
        "idh_sensitivity",
        "who2021_wildtype",
    ):
        assert not_requested not in arms


# =====================================================================
# 2026-08-18 EKLENDİ (Barış kararı) -- TEK MODEL + UCSF harici test
# =====================================================================


def test_c32_whitelist_accepts_ucsf_and_rejects_legacy_names() -> None:
    """C32 VERİ KAPISI: `UCSF-PDGM-PyRadiomics-107-C32` kabul edilir,
    eski nesil adlarının (C32 soneki OLMAYAN + kaynak-sağlayıcı
    precompute) HİÇBİRİ kabul edilmez -- görev talimatının açıkça
    istediği test."""

    for allowed in (
        week3.SEGMENTATION_TOOL_UPENN_C32,
        week3.SEGMENTATION_TOOL_TCGA_C32,
        week3.SEGMENTATION_TOOL_UCSF_C32,
    ):
        week3.assert_c32_segmentation_tool_allowed(allowed)  # hata firlatmamali

    forbidden_legacy_names = (
        "UPenn-PyRadiomics-107",
        "LUMIERE-PyRadiomics-107",
        "TCGA-ground-truth",
        "CaPTk-automatic",
        "CaPTk-corrected",
        "DeepBraTumIA",
        "HD-GLIO-AUTO",
        "UCSF-PDGM-PyRadiomics-107",  # -C32 soneksiz -- de RED edilmeli
        "UCSF-PDGM",
    )
    for forbidden in forbidden_legacy_names:
        with pytest.raises(week3.C32WhitelistViolationError):
            week3.assert_c32_segmentation_tool_allowed(forbidden)


def test_build_ucsf_clinical_covariate_frame_function_no_longer_exists() -> None:
    """2026-08-18 REVİZYON (Barış kararı (a) -- db-agent UCSF onboarding'i
    ARTIK harmonize değer yazacağı için): `build_ucsf_clinical_covariate_
    frame()` KALDIRILDI, UCSF için de `build_clinical_covariate_frame()`
    DOĞRUDAN kullanılıyor. Bu test o kaldırmanın regresyon koruması --
    fonksiyon YANLIŞLIKLA geri eklenirse (kod tekrarı/iki mapping kaynağı
    riski) bu test PATLAR."""

    assert not hasattr(week3, "build_ucsf_clinical_covariate_frame")
    assert not hasattr(week3, "UCSF_EOR_TO_GTR_LABEL")
    assert not hasattr(week3, "UCSF_SEX_MALE_LABEL")
    assert not hasattr(week3, "UCSF_IDH_MUTANT_LABELS")


def test_build_clinical_covariate_frame_handles_harmonized_ucsf_style_values() -> None:
    """UCSF `patients` satırları ARTIK HARMONİZE yazılıyor (db-agent'in
    `tools/onboard_ucsf_patients.py` işi) -- yani UPenn ile AYNI sözlüğü
    (`Male`/`Female`, `Y`/`N`, `Wildtype`/`Mutated`) taşıyor. Bu test,
    `build_clinical_covariate_frame()`'in (AYRI bir UCSF fonksiyonu
    OLMADAN) bu harmonize UCSF-tarzı veriyi doğru kodladığını gösterir --
    tek mapping kaynağı, görev talimatının açıkça istediği test."""

    patients_frame = pd.DataFrame(
        {
            "age": [40.0, 55.0, 62.0, 70.0],
            "gender": ["Male", "Female", "Male", "Female"],
            "gtr_over90percent": ["Y", "N", "N", "Y"],
            "idh1_status": ["Wildtype", "Mutated", "Mutated", "Wildtype"],
            # 2026-08-18 EKLENDİ: build_clinical_covariate_frame() TÜM
            # PATIENTS_FRAME_CLINICAL_RAW_COLUMNS'u (include_* bayrağından
            # BAĞIMSIZ) ister -- mgmt_status eklenince bu fixture'lar da
            # tamamlanmalı (include_mgmt burada False, kolon yalnız
            # varlık kontrolü için gerekli).
            "mgmt_status": ["Methylated", "Unmethylated", "Unmethylated", "Methylated"],
        },
        index=["u1", "u2", "u3", "u4"],
    )

    result = week3.build_clinical_covariate_frame(
        patients_frame, include_gender=True, include_gtr=True, include_idh=True
    )

    assert list(result[week3.CLINICAL_GENDER_MALE_COLUMN]) == [1.0, 0.0, 1.0, 0.0]
    assert list(result[week3.CLINICAL_GTR_Y_COLUMN]) == [1.0, 0.0, 0.0, 1.0]
    assert result[week3.CLINICAL_GTR_MISSING_COLUMN].sum() == 0.0
    assert list(result[week3.CLINICAL_IDH_MUTANT_COLUMN]) == [0.0, 1.0, 1.0, 0.0]
    # UCSF'te NOS/NEC hiç gelmiyor -- bu yüzden clinical_idh_missing
    # DOĞAL OLARAK (özel bir kod dalı OLMADAN) hep 0.
    assert (result[week3.CLINICAL_IDH_MISSING_COLUMN] == 0.0).all()


def test_build_clinical_covariate_frame_ucsf_idh_missing_always_zero_and_does_not_crash() -> None:
    """(d) Sabit sütun (`clinical_idh_missing` hep 0) harici testte
    PATLAMIYOR -- görev talimatının açıkça istediği test. UCSF'te "test
    yapılmamış" kategorisi olmadığı için (harmonize veri yalnız Wildtype/
    Mutated taşır) bu kolon SABİT 0 çıkar, ve bu sabit kolon
    `build_external_test_frame()`/downstream'e sorunsuz aktarılabilmeli."""

    patients_frame = pd.DataFrame(
        {
            "age": [40.0, 55.0, 62.0],
            "gender": ["Male", "Female", "Male"],
            "gtr_over90percent": ["Y", "N", "Y"],
            "idh1_status": ["Wildtype", "Mutated", "Wildtype"],
            "mgmt_status": ["Methylated", "Unmethylated", "Methylated"],
        },
        index=["u1", "u2", "u3"],
    )

    result = week3.build_clinical_covariate_frame(
        patients_frame, include_gender=True, include_gtr=True, include_idh=True
    )

    assert (result[week3.CLINICAL_IDH_MISSING_COLUMN] == 0.0).all()
    assert result[week3.CLINICAL_IDH_MISSING_COLUMN].isna().sum() == 0

    # Downstream: build_external_test_frame() sabit kolonla PATLAMAMALI.
    feature_frame = pd.DataFrame({"WT__dummy_feature": [1.0, 2.0, 3.0]}, index=patients_frame.index)
    full_patients = patients_frame.assign(
        survival_days=[100.0, 200.0, 300.0],
        vital_status=["DECEASED", "ALIVE", "DECEASED"],
    ).join(result)

    external_frame, report = week3.build_external_test_frame(
        feature_frame,
        full_patients,
        passthrough_columns=[week3.CLINICAL_IDH_MISSING_COLUMN, week3.CLINICAL_IDH_MUTANT_COLUMN],
    )
    assert report.n_output_rows == 3
    assert (external_frame[week3.CLINICAL_IDH_MISSING_COLUMN] == 0.0).all()


def test_pivot_ucsf_wt_long_to_wide_builds_wide_frame() -> None:
    ucsf_long, _ = _build_synthetic_ucsf_cohort(n_ucsf=8, seed=101)

    wide, report = week3.pivot_ucsf_wt_long_to_wide(ucsf_long)

    assert len(wide) == 8
    assert report.n_output_patients == 8
    assert report.dropped_patients_missing_region == {"WT": 0}
    # kolonlar "WT__" onekli olmali (pivot_radiomics_long_to_wide() ile TUTARLI).
    assert all(col.startswith("WT__") for col in wide.columns)
    assert f"WT__{week3.STABLE_FEATURES_ICC60[0]}" in wide.columns


def test_pivot_ucsf_wt_long_to_wide_drops_patient_missing_wt_row() -> None:
    ucsf_long, _ = _build_synthetic_ucsf_cohort(n_ucsf=5, seed=102)
    # Bir hastanin WT etiketini TANINMAYAN bir bolge adina cevir -- hasta
    # long_frame'de (aday olarak) KALIR ama WT satiri YOK -- eksik-bolge
    # hasta DUSMELI, SAYILARAK raporlanmali (sessiz filtreleme YASAK).
    # (Satiri TAMAMEN silmek yanlis olurdu -- o zaman hasta "aday" bile
    # sayilmaz, bkz. bu fonksiyonun `all_patients_full` mantigi.)
    mutated_id = ucsf_long["patient_id"].iloc[0]
    mutated = ucsf_long.copy()
    mutated.loc[mutated["patient_id"] == mutated_id, "tumor_region"] = "TC"

    wide, report = week3.pivot_ucsf_wt_long_to_wide(mutated)

    assert mutated_id not in wide.index
    assert len(wide) == 4
    assert report.n_candidate_patients == 5
    assert report.dropped_patients_missing_region == {"WT": 1}


def test_pivot_ucsf_wt_long_to_wide_raises_on_duplicate_patient_region() -> None:
    ucsf_long, _ = _build_synthetic_ucsf_cohort(n_ucsf=3, seed=103)
    duplicated = pd.concat([ucsf_long, ucsf_long.iloc[[0]]], ignore_index=True)

    with pytest.raises(week3.UCSFRegionPivotError):
        week3.pivot_ucsf_wt_long_to_wide(duplicated)


def test_check_external_test_pool_counts_hard_fails_on_mismatch_without_override() -> None:
    with pytest.raises(week3.ExternalTestCountMismatchError):
        week3.check_external_test_pool_counts(
            10, 5, expected_patients=295, expected_events=169, allow_mismatch=False
        )


def test_check_external_test_pool_counts_warns_and_continues_with_override(capsys) -> None:
    week3.check_external_test_pool_counts(
        10, 5, expected_patients=295, expected_events=169, allow_mismatch=True
    )
    captured = capsys.readouterr()
    assert "UYARI" in captured.err


def test_check_external_test_pool_counts_passes_on_exact_match() -> None:
    week3.check_external_test_pool_counts(
        295, 169, expected_patients=295, expected_events=169, allow_mismatch=False
    )  # hata firlatmamali


def test_run_primary_single_model_keeps_full_611_style_cohort_with_nos_nec_patients(
    tmp_path: Path,
) -> None:
    """611 tam kohort IDH'li modelde KORUNUYOR MU (515'e DÜŞMÜYOR) --
    görev talimatının açıkça istediği test. Sentetik kohortta bir kısım
    hasta NOS/NEC (IDH testi yapılmamış) -- yeni gösterge-değişkeni
    yaklaşımıyla bu hastalar DÜŞÜRÜLMEMELİ."""

    upenn_long, upenn_patients, _tcga_long, _tcga_patients = _build_synthetic_cohort(
        n_upenn=80, n_tcga=10, seed=111
    )
    # En az bir kismi NOS/NEC oldugunu garanti et (sentetik uretimde
    # rastgele geldigi icin -- olmazsa test anlamsizlasir).
    n_nos_nec = int((upenn_patients["idh1_status"] == "NOS/NEC").sum())
    assert n_nos_nec > 0, "sentetik kohortta NOS/NEC hasta yok -- seed degistir"

    ucsf_long, ucsf_patients = _build_synthetic_ucsf_cohort(n_ucsf=20, seed=112)

    expected_upenn_events = int((upenn_patients["vital_status"] == "DECEASED").sum())
    expected_upenn_patients = len(upenn_patients)  # NOS/NEC dahil, DUSMEMELI

    args = _fast_args(
        tmp_path,
        expected_patient_count=expected_upenn_patients,
        expected_event_count=expected_upenn_events,
        expected_ucsf_patient_count=0,  # asagida gercek sayiyla ayrica dogrulanacak
        expected_ucsf_event_count=0,
        allow_unexpected_ucsf_count=True,  # sentetik n kucuk, sabit 295/169 tutmaz
    )

    exit_code = week3.run_primary_single_model(
        upenn_long=upenn_long,
        upenn_patients=upenn_patients,
        ucsf_long=ucsf_long,
        ucsf_patients=ucsf_patients,
        args=args,
        output_dir=tmp_path,
    )
    assert exit_code == 0

    external = pd.read_csv(tmp_path / "week3_external_test.csv")
    assert len(external) == 1  # TEK MODEL -- kol karsilastirmasi YOK
    assert external.loc[0, "arm"] == week3.PRIMARY_SINGLE_MODEL_ARM_NAME
    assert external.loc[0, "external_test_source"] == "UCSF-PDGM"

    metadata = json.loads((tmp_path / "week3_run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "primary_single_model_ucsf_external_test"
    # NOS/NEC hastalar dahil TAM kohort korunuyor -- egitim havuzu
    # expected_upenn_patients'la (NOS/NEC HARIC TUTULMADAN olculen n) esit.
    assert metadata["training_frame_report"]["n_output_rows"] == expected_upenn_patients
    assert "idh_indicator_note" in metadata
    # 2026-08-18 REVİZYON: ayrı bir UCSF eşleme sözlüğü artık YOK (db-agent
    # onboarding'de harmonize yazıyor) -- bunun yerine tek-mapping-kaynağı
    # notu bekleniyor.
    assert "ucsf_harmonization_note" in metadata
    assert "onboard_ucsf_patients.py" in metadata["ucsf_harmonization_note"]
    # 2026-08-18 koordinatör düzeltmesi: post-hoc beyanı + ComBat muafiyeti
    # metadata'da AÇIKÇA yer almalı (rapordan çıkarılamaz, decisions/
    # 2026-08-18-tek-model-ucsf-harici-test-k15-kapanisi.md §0).
    assert "post_hoc_disclosure_note" in metadata
    assert "0,689" in metadata["post_hoc_disclosure_note"]
    assert "k14_component_contribution_note" in metadata
    assert "ucsf_combat_exemption_note" in metadata
    assert "ComBat" in metadata["ucsf_combat_exemption_note"]
    assert "ucsf_idh_wildtype_filter_note" in metadata
    assert "ucsf_expected_count_note" in metadata
    assert "ucsf_download_incomplete_selection_bias_note" in metadata


def test_run_primary_single_model_raises_on_ucsf_count_mismatch_without_override(
    tmp_path: Path,
) -> None:
    upenn_long, upenn_patients, _tcga_long, _tcga_patients = _build_synthetic_cohort(
        n_upenn=40, n_tcga=10, seed=121
    )
    ucsf_long, ucsf_patients = _build_synthetic_ucsf_cohort(n_ucsf=15, seed=122)

    expected_upenn_events = int((upenn_patients["vital_status"] == "DECEASED").sum())
    args = _fast_args(
        tmp_path,
        expected_patient_count=len(upenn_patients),
        expected_event_count=expected_upenn_events,
        expected_ucsf_patient_count=295,  # sentetik n=15 ile KASITLI uyusmuyor
        expected_ucsf_event_count=169,
        allow_unexpected_ucsf_count=False,
    )

    with pytest.raises(week3.ExternalTestCountMismatchError):
        week3.run_primary_single_model(
            upenn_long=upenn_long,
            upenn_patients=upenn_patients,
            ucsf_long=ucsf_long,
            ucsf_patients=ucsf_patients,
            args=args,
            output_dir=tmp_path,
        )


def test_run_primary_single_model_never_applies_combat_to_ucsf(
    tmp_path: Path, monkeypatch
) -> None:
    """2026-08-18 koordinatör kararı (decisions/2026-08-18-tek-model-
    ucsf-harici-test-k15-kapanisi.md §0 ÇELİŞKİ 3): UCSF, TCGA'nın
    bıraktığı harici test rolünü devraldığı için AYNI ComBat muamelesini
    görür -- feature-level ComBat düzeltmesi ALMAZ, ComBat fit'ine
    GİRMEZ. Bu test `pipeline.harmonization.apply_combat_harmonization()`
    / `fit_combat_harmonization()`'ı PATLAYACAK şekilde monkeypatch'ler --
    `run_primary_single_model()`'in UCSF yolu bunları GERÇEKTEN hiç
    çağırmıyorsa koşu sorunsuz tamamlanır (ÇAĞRILSAYDI test hemen
    patlardı, bu da lock'un ANLAMLI olduğunu kanıtlar)."""

    import pipeline.harmonization as harmonization_module

    def _boom(*_args, **_kwargs):
        raise AssertionError(
            "ComBat fonksiyonu (apply_combat_harmonization/"
            "fit_combat_harmonization) UCSF harici test yolunda "
            "ÇAĞRILDI -- bu YASAK (2026-08-18 koordinatör kararı, "
            "TCGA'nın ComBat muafiyetiyle AYNI)."
        )

    monkeypatch.setattr(harmonization_module, "apply_combat_harmonization", _boom)
    monkeypatch.setattr(harmonization_module, "fit_combat_harmonization", _boom)

    # NOT: n/seed BİLEREK `test_run_primary_single_model_keeps_full_611_
    # style_cohort_with_nos_nec_patients` ile AYNI (o testte GÜVENİLİR
    # şekilde yakınsadığı doğrulandı) -- bu testin AMACI ComBat guard'ı,
    # farklı bir seed'in küçük sentetik kohortta yakınsama flakiness'i
    # YARATMAMASI için.
    upenn_long, upenn_patients, _tcga_long, _tcga_patients = _build_synthetic_cohort(
        n_upenn=80, n_tcga=10, seed=111
    )
    ucsf_long, ucsf_patients = _build_synthetic_ucsf_cohort(n_ucsf=20, seed=112)
    expected_events = int((upenn_patients["vital_status"] == "DECEASED").sum())
    args = _fast_args(
        tmp_path,
        expected_patient_count=len(upenn_patients),
        expected_event_count=expected_events,
        allow_unexpected_ucsf_count=True,  # sentetik n kucuk, sabit 295/169 tutmaz
    )

    exit_code = week3.run_primary_single_model(
        upenn_long=upenn_long,
        upenn_patients=upenn_patients,
        ucsf_long=ucsf_long,
        ucsf_patients=ucsf_patients,
        args=args,
        output_dir=tmp_path,
    )
    assert exit_code == 0  # patlamadan tamamlandi -> ComBat fonksiyonlari HIC cagrilmadi

    metadata = json.loads((tmp_path / "week3_run_metadata.json").read_text(encoding="utf-8"))
    assert "ucsf_combat_exemption_note" in metadata
    assert "ComBat" in metadata["ucsf_combat_exemption_note"]


# =====================================================================
# 2026-08-18 -- Model v2 (MGMT kovaryatı + yaş spline + WT+TC bölgesi +
# dört varyant orkestrasyonu)
# =====================================================================


def _build_synthetic_multi_region_cohort(
    *, n_upenn: int, n_ucsf: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """`v2c_mgmt_spline_wttc` (WT+TC) DAHİL tüm Model-v2 varyantlarını
    tek bir sentetik kohortla test edebilmek için hem `WT_derived` HEM
    `TC_derived` satırlarını üretir -- WT-only varyantlar (v1/v2a/v2b)
    fazladan `TC_derived` satırlarını pivot aşamasında zaten görmezden
    gelir (`pivot_radiomics_long_to_wide(regions=["WT"])`/`pivot_ucsf_
    regions_long_to_wide(regions=("WT",))`), bu yüzden TEK kohort
    üreticisi dört varyanta da güvenle verilebilir."""

    rng = np.random.default_rng(seed)

    upenn_ids = [f"SYN-UPENN-{i:04d}" for i in range(n_upenn)]
    upenn_wt_long, upenn_wt_raw = _make_synthetic_long_frame(
        upenn_ids, source="UPenn-GBM", region="WT_derived", rng=rng
    )
    upenn_tc_long, _ = _make_synthetic_long_frame(
        upenn_ids, source="UPenn-GBM", region="TC_derived", rng=rng
    )
    upenn_long = pd.concat([upenn_wt_long, upenn_tc_long], ignore_index=True)
    upenn_duration, upenn_vital = _simulate_survival(
        upenn_wt_raw, INFORMATIVE_FEATURES, INFORMATIVE_COEFFICIENTS, rng=rng
    )
    upenn_clinical_raw = _make_synthetic_clinical_raw(n_upenn, rng=rng, missing_gtr_idh=False)
    upenn_patients = _make_patients_frame(
        upenn_ids, "UPenn-GBM", upenn_duration, upenn_vital, clinical_raw=upenn_clinical_raw
    )

    ucsf_ids = [f"SYN-UCSF-{i:04d}" for i in range(n_ucsf)]
    ucsf_wt_long, ucsf_wt_raw = _make_synthetic_long_frame(
        ucsf_ids, source="UCSF-PDGM", region="WT_derived", rng=rng
    )
    ucsf_tc_long, _ = _make_synthetic_long_frame(
        ucsf_ids, source="UCSF-PDGM", region="TC_derived", rng=rng
    )
    ucsf_long = pd.concat([ucsf_wt_long, ucsf_tc_long], ignore_index=True)
    ucsf_duration, ucsf_vital = _simulate_survival(
        ucsf_wt_raw, INFORMATIVE_FEATURES, INFORMATIVE_COEFFICIENTS, rng=rng
    )
    ucsf_clinical_raw = _make_synthetic_ucsf_clinical_raw(n_ucsf, rng=rng)
    ucsf_patients = _make_patients_frame(
        ucsf_ids, "UCSF-PDGM", ucsf_duration, ucsf_vital, clinical_raw=ucsf_clinical_raw
    )

    return upenn_long, upenn_patients, ucsf_long, ucsf_patients


# ---------------------------------------------------------------------
# A) MGMT kovaryatı
# ---------------------------------------------------------------------


def test_build_clinical_covariate_frame_mgmt_indeterminate_and_missing_indicator() -> None:
    """MGMT kodlaması -- GTR/IDH ile BİREBİR aynı gösterge-değişkeni
    deseni. Indeterminate (p3) VE ham NULL (p4) İKİSİ DE eksik sayılır
    (`clinical_mgmt_missing=1`, `clinical_mgmt_methylated=0` referans) --
    görev talimatının açıkça istediği test."""

    patients_frame = _make_raw_clinical_patients_frame()

    result = week3.build_clinical_covariate_frame(patients_frame, include_mgmt=True)

    # p1=Methylated, p2=Unmethylated, p3=Indeterminate, p4=NULL
    assert result.loc["p1", week3.CLINICAL_MGMT_METHYLATED_COLUMN] == 1.0
    assert result.loc["p1", week3.CLINICAL_MGMT_MISSING_COLUMN] == 0.0
    assert result.loc["p2", week3.CLINICAL_MGMT_METHYLATED_COLUMN] == 0.0
    assert result.loc["p2", week3.CLINICAL_MGMT_MISSING_COLUMN] == 0.0
    # Indeterminate -> methylated=0 (referans), missing=1.
    assert result.loc["p3", week3.CLINICAL_MGMT_METHYLATED_COLUMN] == 0.0
    assert result.loc["p3", week3.CLINICAL_MGMT_MISSING_COLUMN] == 1.0
    # ham NULL -- ayni sekilde eksik sayilir.
    assert result.loc["p4", week3.CLINICAL_MGMT_METHYLATED_COLUMN] == 0.0
    assert result.loc["p4", week3.CLINICAL_MGMT_MISSING_COLUMN] == 1.0
    # HICBIR satir NaN kalmamali (GTR/IDH'deki ilkeyle AYNI).
    assert result[week3.CLINICAL_MGMT_METHYLATED_COLUMN].isna().sum() == 0
    assert result[week3.CLINICAL_MGMT_MISSING_COLUMN].isna().sum() == 0


def test_build_clinical_covariate_frame_mgmt_not_built_when_include_mgmt_false() -> None:
    patients_frame = _make_raw_clinical_patients_frame()

    result = week3.build_clinical_covariate_frame(patients_frame, include_gender=True)

    assert week3.CLINICAL_MGMT_METHYLATED_COLUMN not in result.columns
    assert week3.CLINICAL_MGMT_MISSING_COLUMN not in result.columns


def test_build_clinical_covariate_frame_mgmt_raises_on_unrecognized_value() -> None:
    patients_frame = _make_raw_clinical_patients_frame()
    patients_frame.loc["p1", "mgmt_status"] = "Positive"  # semaya uymuyor

    with pytest.raises(week3.UnrecognizedClinicalValueError, match="mgmt_status"):
        week3.build_clinical_covariate_frame(patients_frame, include_mgmt=True)


def test_build_clinical_covariate_frame_mgmt_handles_ucsf_style_no_indeterminate() -> None:
    """UCSF'in harmonize sözlüğü SADECE Methylated/Unmethylated/NULL
    taşır (Indeterminate hiç görülmez) -- guard bunu SORUNSUZ kabul
    etmeli (tanınan küme fazlası zararsız)."""

    patients_frame = pd.DataFrame(
        {
            "age": [40.0, 55.0, 62.0],
            "gender": ["Male", "Female", "Male"],
            "gtr_over90percent": ["Y", "N", "Y"],
            "idh1_status": ["Wildtype", "Mutated", "Wildtype"],
            "mgmt_status": ["Methylated", "Unmethylated", None],
        },
        index=["u1", "u2", "u3"],
    )

    result = week3.build_clinical_covariate_frame(patients_frame, include_mgmt=True)

    assert list(result[week3.CLINICAL_MGMT_METHYLATED_COLUMN]) == [1.0, 0.0, 0.0]
    assert list(result[week3.CLINICAL_MGMT_MISSING_COLUMN]) == [0.0, 0.0, 1.0]


# ---------------------------------------------------------------------
# B) Yaş restricted cubic spline
# ---------------------------------------------------------------------


def test_compute_rcs_knots_returns_sorted_percentiles() -> None:
    ages = pd.Series(np.arange(20, 90, dtype=float))
    knots = week3.compute_rcs_knots(ages)
    assert len(knots) == 3
    assert knots[0] < knots[1] < knots[2]
    np.testing.assert_allclose(
        knots, np.percentile(ages.to_numpy(), list(week3.RCS_AGE_KNOT_PERCENTILES))
    )


def test_compute_rcs_knots_raises_on_degenerate_sample() -> None:
    ages = pd.Series([50.0] * 10)  # tum degerler ayni -- persentiller cakisir
    with pytest.raises(ValueError):
        week3.compute_rcs_knots(ages)


def test_restricted_cubic_spline_basis_is_exactly_zero_below_first_knot() -> None:
    """Restricted cubic spline'ın "restricted" özelliği: en düşük düğümün
    ALTINDA doğrusal-olmayan terim TAM SIFIR olmalı (formülün üç
    (x-t)+ teriminin de bu bölgede sıfır olması gerektiği matematiksel
    sonuç) -- bu görevin açıkça belirttiği doğrulama yöntemi."""

    knots = np.array([30.0, 50.0, 70.0])
    ages_below = pd.Series([10.0, 20.0, 29.999])
    basis = week3.restricted_cubic_spline_basis(ages_below, knots)

    np.testing.assert_allclose(basis[week3.CLINICAL_AGE_RCS2_COLUMN].to_numpy(), 0.0, atol=1e-9)
    assert list(basis[week3.CLINICAL_AGE_RCS1_COLUMN]) == [10.0, 20.0, 29.999]


def test_restricted_cubic_spline_basis_is_linear_beyond_last_knot() -> None:
    """Restricted cubic spline'ın diğer tanımlayıcı özelliği: en yüksek
    düğümün ÖTESİNDE de doğrusal (ikinci fark ~0) -- "restricted"/doğal
    sınır koşulu."""

    knots = np.array([30.0, 50.0, 70.0])
    x = pd.Series([80.0, 85.0, 90.0, 95.0, 100.0])
    basis = week3.restricted_cubic_spline_basis(x, knots)
    values = basis[week3.CLINICAL_AGE_RCS2_COLUMN].to_numpy()
    first_diff = np.diff(values)
    second_diff = np.diff(first_diff)
    np.testing.assert_allclose(second_diff, 0.0, atol=1e-6)


def test_restricted_cubic_spline_basis_requires_exactly_three_knots() -> None:
    with pytest.raises(ValueError):
        week3.restricted_cubic_spline_basis(pd.Series([40.0]), np.array([30.0, 50.0]))


def test_restricted_cubic_spline_basis_requires_strictly_increasing_knots() -> None:
    with pytest.raises(ValueError):
        week3.restricted_cubic_spline_basis(pd.Series([40.0]), np.array([30.0, 50.0, 50.0]))


def test_restricted_cubic_spline_basis_fixed_knots_no_leakage_when_applied_to_new_data() -> None:
    """SIZINTI TESTİ (görev talimatının açıkça istediği): düğümler
    (train'den önceden hesaplanmış) `test`/UCSF verisine AYNEN
    uygulanabilmeli -- fonksiyonun kendisi knot'ları YENİDEN
    HESAPLAMAZ (saf dönüşüm), yani aynı `knots` iki FARKLI yaş
    dağılımına (train/test) tutarlı şekilde uygulanabilir ve knot'lar
    ikinci çağrıda DEĞİŞMEZ."""

    train_ages = pd.Series(np.linspace(20, 85, 200))
    knots = week3.compute_rcs_knots(train_ages)

    test_ages = pd.Series([15.0, 33.0, 91.0])  # UCSF benzeri, egitimden FARKLI dagilim
    basis_test = week3.restricted_cubic_spline_basis(test_ages, knots)

    # Aynı knot'larla İKİNCİ bir çağrı (örn. train'in kendisi) knot
    # değerlerini DEĞİŞTİRMEMELİ -- knots parametresi saf girdi.
    basis_train_again = week3.restricted_cubic_spline_basis(train_ages, knots)
    np.testing.assert_array_equal(knots, week3.compute_rcs_knots(train_ages))
    assert len(basis_test) == 3
    assert len(basis_train_again) == 200


def test_build_variant_clinical_extra_columns_v1_matches_primary_constant() -> None:
    """`v1_referans` varyantının extra_columns'ı, mevcut `PRIMARY_
    CLINICAL_EXTRA_COLUMNS` ile BİREBİR AYNI olmalı (reprodüksiyon
    kontrolü -- görev talimatı: 'aynı sonucu vermeli')."""

    columns = week3.build_variant_clinical_extra_columns(include_mgmt=False, use_age_spline=False)
    assert columns == list(week3.PRIMARY_CLINICAL_EXTRA_COLUMNS)


def test_build_variant_clinical_extra_columns_mgmt_and_spline_composition() -> None:
    columns = week3.build_variant_clinical_extra_columns(include_mgmt=True, use_age_spline=True)
    assert columns[:2] == [week3.CLINICAL_AGE_RCS1_COLUMN, week3.CLINICAL_AGE_RCS2_COLUMN]
    assert columns[-2:] == [week3.CLINICAL_MGMT_METHYLATED_COLUMN, week3.CLINICAL_MGMT_MISSING_COLUMN]
    assert week3.CLINICAL_AGE_COLUMN not in columns


def test_build_variant_clinical_frame_replaces_age_with_spline_columns() -> None:
    patients_frame = _make_raw_clinical_patients_frame().dropna(subset=["age"])
    knots = week3.compute_rcs_knots(patients_frame["age"])

    result = week3.build_variant_clinical_frame(
        patients_frame, include_mgmt=False, use_age_spline=True, age_knots=knots
    )

    assert week3.CLINICAL_AGE_COLUMN not in result.columns
    assert week3.CLINICAL_AGE_RCS1_COLUMN in result.columns
    assert week3.CLINICAL_AGE_RCS2_COLUMN in result.columns


def test_build_variant_clinical_frame_raises_when_spline_requested_without_knots() -> None:
    patients_frame = _make_raw_clinical_patients_frame().dropna(subset=["age"])
    with pytest.raises(ValueError):
        week3.build_variant_clinical_frame(
            patients_frame, include_mgmt=False, use_age_spline=True, age_knots=None
        )


# ---------------------------------------------------------------------
# C) WT+TC bölge pivotu (UCSF)
# ---------------------------------------------------------------------


def test_pivot_ucsf_regions_long_to_wide_wt_only_matches_legacy_wrapper() -> None:
    ucsf_long, _ = _build_synthetic_ucsf_cohort(n_ucsf=6, seed=201)

    wide_generic, report_generic = week3.pivot_ucsf_regions_long_to_wide(ucsf_long, regions=("WT",))
    wide_legacy, report_legacy = week3.pivot_ucsf_wt_long_to_wide(ucsf_long)

    pd.testing.assert_frame_equal(wide_generic, wide_legacy)
    assert report_generic.dropped_patients_missing_region == report_legacy.dropped_patients_missing_region
    assert report_generic.regions_requested == report_legacy.regions_requested


def test_pivot_ucsf_regions_long_to_wide_wt_tc_builds_186_columns() -> None:
    upenn_long, upenn_patients, ucsf_long, ucsf_patients = _build_synthetic_multi_region_cohort(
        n_upenn=10, n_ucsf=6, seed=202
    )

    wide, report = week3.pivot_ucsf_regions_long_to_wide(ucsf_long, regions=("WT", "TC"))

    assert len(wide) == 6
    assert report.dropped_patients_missing_region == {"WT": 0, "TC": 0}
    wt_columns = [c for c in wide.columns if c.startswith("WT__")]
    tc_columns = [c for c in wide.columns if c.startswith("TC__")]
    assert len(wt_columns) == 107
    assert len(tc_columns) == 107


def test_pivot_ucsf_regions_long_to_wide_drops_patient_missing_any_requested_region() -> None:
    _upenn_long, _upenn_patients, ucsf_long, _ucsf_patients = _build_synthetic_multi_region_cohort(
        n_upenn=5, n_ucsf=5, seed=203
    )
    # Bir hastanin TC_derived satirini SIL -- WT+TC istenince o hasta
    # TUMDEN dusmeli (yalniz TC istense dusmez, WT istense dusmez).
    target_id = ucsf_long.loc[ucsf_long["tumor_region"] == "TC_derived", "patient_id"].iloc[0]
    mutated = ucsf_long.loc[
        ~((ucsf_long["patient_id"] == target_id) & (ucsf_long["tumor_region"] == "TC_derived"))
    ].copy()

    wide, report = week3.pivot_ucsf_regions_long_to_wide(mutated, regions=("WT", "TC"))

    assert target_id not in wide.index
    assert report.dropped_patients_missing_region["TC"] == 1
    assert report.dropped_patients_missing_region["WT"] == 0

    wide_wt_only, report_wt_only = week3.pivot_ucsf_regions_long_to_wide(mutated, regions=("WT",))
    assert target_id in wide_wt_only.index
    assert report_wt_only.dropped_patients_missing_region == {"WT": 0}


def test_pivot_ucsf_regions_long_to_wide_raises_on_unknown_region() -> None:
    ucsf_long, _ = _build_synthetic_ucsf_cohort(n_ucsf=4, seed=204)
    with pytest.raises(ValueError, match="tanimiyor"):
        week3.pivot_ucsf_regions_long_to_wide(ucsf_long, regions=("ET",))


def test_pivot_ucsf_regions_long_to_wide_raises_on_duplicate_patient_region() -> None:
    ucsf_long, _ = _build_synthetic_ucsf_cohort(n_ucsf=3, seed=205)
    duplicated = pd.concat([ucsf_long, ucsf_long.iloc[[0]]], ignore_index=True)
    with pytest.raises(week3.UCSFRegionPivotError):
        week3.pivot_ucsf_regions_long_to_wide(duplicated, regions=("WT",))


# ---------------------------------------------------------------------
# D) Dört-varyant orkestrasyonu -- uçtan uca
# ---------------------------------------------------------------------


def test_select_v2_variant_configs_default_returns_all_four_in_order() -> None:
    configs = week3.select_v2_variant_configs(None)
    assert [c.name for c in configs] == list(week3.V2_VARIANT_NAMES)
    assert list(week3.V2_VARIANT_NAMES) == [
        "v1_referans",
        "v2a_mgmt",
        "v2b_mgmt_spline",
        "v2c_mgmt_spline_wttc",
    ]


def test_select_v2_variant_configs_filters_by_name() -> None:
    configs = week3.select_v2_variant_configs(["v2a_mgmt"])
    assert [c.name for c in configs] == ["v2a_mgmt"]


def test_select_v2_variant_configs_raises_on_unknown_name() -> None:
    with pytest.raises(ValueError):
        week3.select_v2_variant_configs(["not_a_real_variant"])


# ---------------------------------------------------------------------
# D.1) 2026-09-11 FİNAL KOŞU eklentisi -- v2d/v3c (Barış onayı,
#      MODEL-SECIM-ANALIZI-2026-09-11.md §4.2). Bu bölüm SADECE
#      config-seviyesi/CLI-seviyesi testler içerir (uçtan uca model
#      fit'i D bölümündeki mevcut end-to-end testlerle AYNI ağır
#      fixture'ları GEREKTİRMEZ, bkz. görev talimatı "en az 2 yeni test").
# ---------------------------------------------------------------------


def test_v2_variant_configs_extra_v3c_base_has_expected_components() -> None:
    """`v2d_mgmt_wttc_nospline` (v3c'nin tabanı) -- bölge WT+TC, MGMT
    kovaryatları VAR, yaş DOĞRUSAL (spline kolonları YOK)."""

    base = week3.V2_VARIANT_CONFIGS_EXTRA[0]
    assert base.name == "v2d_mgmt_wttc_nospline"
    assert base.regions == ("WT", "TC")
    assert base.include_mgmt is True
    assert base.use_age_spline is False

    extra_columns = week3.build_variant_clinical_extra_columns(
        include_mgmt=base.include_mgmt, use_age_spline=base.use_age_spline
    )
    assert week3.CLINICAL_MGMT_METHYLATED_COLUMN in extra_columns
    assert week3.CLINICAL_MGMT_MISSING_COLUMN in extra_columns
    assert week3.CLINICAL_AGE_COLUMN in extra_columns
    assert week3.CLINICAL_AGE_RCS1_COLUMN not in extra_columns
    assert week3.CLINICAL_AGE_RCS2_COLUMN not in extra_columns


def test_v3_variant_configs_extra_v3c_wraps_expected_base_with_default_filter_grid() -> None:
    """`v3c_lowvar_wttc_mgmt_nospline` -- v3 filtresinin `v2d_mgmt_wttc_
    nospline` tabanını AYNEN referans aldığını (v2c'ye DEĞİL) ve cv/corr
    eşiklerinin v3a/v3b ile AYNI (elle sabitlenmemiş) varsayılan ızgarayı
    kullandığını doğrular -- görev talimatı: 'ızgara MEVCUT varyantlarla
    AYNI, elle sabitleme/yeni ızgara YOK'."""

    v3c = week3.V3_VARIANT_CONFIGS_EXTRA[0]
    assert v3c.name == "v3c_lowvar_wttc_mgmt_nospline"
    assert v3c.base is week3.V2_VARIANT_CONFIGS_EXTRA[0]
    assert v3c.base.regions == ("WT", "TC")
    assert v3c.base.include_mgmt is True
    assert v3c.base.use_age_spline is False
    assert v3c.cv_threshold == week3.V3_DEFAULT_CV_THRESHOLD
    assert v3c.corr_threshold == week3.V3_DEFAULT_CORRELATION_CLUSTER_THRESHOLD
    assert v3c.standardize_radiomics is True

    v3a = next(c for c in week3.V3_VARIANT_CONFIGS if c.name == "v3a_lowvar_v1referans")
    assert v3c.cv_threshold == v3a.cv_threshold
    assert v3c.corr_threshold == v3a.corr_threshold


def test_v2_and_v3_variant_names_extra_registered_in_cli_choices() -> None:
    """--variants CLI'sinin choices listesi yeni iki ismi de KAPSAMALI
    (aksi hâlde argparse SESSİZCE reddeder) -- eski isimler hâlâ kabul
    ediliyor (choices genişletildi, hiçbiri SİLİNMEDİ/DEĞİŞMEDİ)."""

    args_v3c = week3.parse_args(["--variants", "v3c_lowvar_wttc_mgmt_nospline"])
    assert args_v3c.variants == ["v3c_lowvar_wttc_mgmt_nospline"]

    args_v2d = week3.parse_args(["--variants", "v2d_mgmt_wttc_nospline"])
    assert args_v2d.variants == ["v2d_mgmt_wttc_nospline"]

    args_old = week3.parse_args(["--variants", *week3.V2_VARIANT_NAMES])
    assert args_old.variants == list(week3.V2_VARIANT_NAMES)

    with pytest.raises(SystemExit):
        week3.parse_args(["--variants", "not_a_real_variant"])


def test_existing_v2_and_v3_variant_configs_unchanged_snapshot() -> None:
    """Mevcut 4 v2 + 2 v3 varyant tanımının bit-birebir DEĞİŞMEDİĞİNİ
    dondurulmuş bir snapshot ile doğrular -- yeni v2d/v3c eklentisinin
    MEVCUT hiçbir satırı ETKİLEMEDİĞİNİN kanıtı (görev talimatı: 'mevcut
    varyant tanımlarının bit-birebir değişmediği')."""

    assert week3.V2_VARIANT_CONFIGS == (
        week3.VariantConfig(
            name="v1_referans", regions=("WT",), include_mgmt=False, use_age_spline=False
        ),
        week3.VariantConfig(
            name="v2a_mgmt", regions=("WT",), include_mgmt=True, use_age_spline=False
        ),
        week3.VariantConfig(
            name="v2b_mgmt_spline", regions=("WT",), include_mgmt=True, use_age_spline=True
        ),
        week3.VariantConfig(
            name="v2c_mgmt_spline_wttc",
            regions=("WT", "TC"),
            include_mgmt=True,
            use_age_spline=True,
        ),
    )
    assert week3.V3_VARIANT_CONFIGS == (
        week3.V3VariantConfig(name="v3a_lowvar_v1referans", base=week3.V2_VARIANT_CONFIGS[0]),
        week3.V3VariantConfig(name="v3b_lowvar_v2amgmt", base=week3.V2_VARIANT_CONFIGS[1]),
    )
    assert week3.ALL_V2_VARIANT_CONFIGS == week3.V2_VARIANT_CONFIGS + week3.V2_VARIANT_CONFIGS_EXTRA
    assert week3.ALL_V3_VARIANT_CONFIGS == week3.V3_VARIANT_CONFIGS + week3.V3_VARIANT_CONFIGS_EXTRA
    assert len(week3.ALL_V2_VARIANT_CONFIGS) == 5
    assert len(week3.ALL_V3_VARIANT_CONFIGS) == 3


def test_run_v2_variant_suite_end_to_end_all_four_variants(tmp_path: Path) -> None:
    """Dört varyantın da uçtan uca çalıştığını, HER birinin kendi dosya
    setini ürettiğini ve TEK bir karşılaştırma tablosunun HEPSİNİ
    içerdiğini doğrular -- görev talimatının açıkça istediği test."""

    upenn_long, upenn_patients, ucsf_long, ucsf_patients = _build_synthetic_multi_region_cohort(
        n_upenn=90, n_ucsf=20, seed=301
    )
    expected_events = int((upenn_patients["vital_status"] == "DECEASED").sum())
    args = _fast_args(
        tmp_path,
        expected_patient_count=len(upenn_patients),
        expected_event_count=expected_events,
        allow_unexpected_ucsf_count=True,
    )

    exit_code = week3.run_v2_variant_suite(
        upenn_long=upenn_long,
        upenn_patients=upenn_patients,
        ucsf_long=ucsf_long,
        ucsf_patients=ucsf_patients,
        args=args,
        output_dir=tmp_path,
    )
    assert exit_code == 0

    comparison = pd.read_csv(tmp_path / "week3_v2_variant_comparison.csv")
    assert list(comparison["variant"]) == list(week3.V2_VARIANT_NAMES)
    # HİÇBİR varyant dışarıda bırakılmadı -- şeffaflık kuralı.
    assert len(comparison) == 4
    assert comparison["external_c_index"].notna().all()
    assert comparison["external_ci_lower"].notna().all()
    assert comparison["external_ci_upper"].notna().all()
    # v2c WT+TC -- radyomik aday havuzu v1'in (WT-only) TAM 2 KATI olmali.
    v1_candidates = comparison.loc[comparison["variant"] == "v1_referans", "n_radiomic_candidates"].iloc[0]
    v2c_candidates = comparison.loc[
        comparison["variant"] == "v2c_mgmt_spline_wttc", "n_radiomic_candidates"
    ].iloc[0]
    assert v2c_candidates == 2 * v1_candidates == 186
    assert v1_candidates == 93

    for name in week3.V2_VARIANT_NAMES:
        assert (tmp_path / f"week3_{name}_fold_results.csv").is_file()
        assert (tmp_path / f"week3_{name}_selection_frequency.csv").is_file()
        assert (tmp_path / f"week3_{name}_external_test.csv").is_file()
        assert (tmp_path / f"week3_{name}_final_coefficients.csv").is_file()
        assert (tmp_path / f"week3_{name}_run_metadata.json").is_file()

        external = pd.read_csv(tmp_path / f"week3_{name}_external_test.csv")
        assert external.loc[0, "variant"] == name
        assert external.loc[0, "external_test_source"] == "UCSF-PDGM"

        coefficients = pd.read_csv(tmp_path / f"week3_{name}_final_coefficients.csv")
        assert not coefficients.empty
        assert {"covariate", "coef", "exp(coef)", "p"}.issubset(coefficients.columns)
        # klinik kovaryatlar HER ZAMAN final modelde olmali (extra_columns).
        metadata = json.loads((tmp_path / f"week3_{name}_run_metadata.json").read_text(encoding="utf-8"))
        expected_extra = set(metadata["clinical_extra_columns"])
        assert expected_extra.issubset(set(coefficients["covariate"]))

    # MGMT'siz/spline'siz v1_referans -- MGMT/spline kolonlari final
    # katsayi tablosunda GORUNMEMELI.
    v1_coefficients = pd.read_csv(tmp_path / "week3_v1_referans_final_coefficients.csv")
    assert week3.CLINICAL_MGMT_METHYLATED_COLUMN not in set(v1_coefficients["covariate"])
    assert week3.CLINICAL_AGE_RCS1_COLUMN not in set(v1_coefficients["covariate"])
    assert week3.CLINICAL_AGE_COLUMN in set(v1_coefficients["covariate"])

    # v2b/v2c spline kullanir -- final katsayi tablosunda dogrusal
    # clinical_age DEGIL, iki spline kolonu olmali.
    v2b_coefficients = pd.read_csv(tmp_path / "week3_v2b_mgmt_spline_final_coefficients.csv")
    assert week3.CLINICAL_AGE_COLUMN not in set(v2b_coefficients["covariate"])
    assert week3.CLINICAL_AGE_RCS1_COLUMN in set(v2b_coefficients["covariate"])
    assert week3.CLINICAL_AGE_RCS2_COLUMN in set(v2b_coefficients["covariate"])
    assert week3.CLINICAL_MGMT_METHYLATED_COLUMN in set(v2b_coefficients["covariate"])


def test_run_v2_variant_suite_respects_selected_variants_subset(tmp_path: Path) -> None:
    # n=200 (60/90'dan buyutuldu) -- MGMT'nin nadir "Indeterminate"
    # kategorisi (~%4,4) kucuk n'de outer/inner fold'lara boluncede
    # sifir-varyans/yakinsamama riski tasiyor (ampirik gozlem, bkz. bu
    # testin ilk versiyonlarindaki flaky RuntimeError). Daha buyuk n bu
    # riski azaltir.
    upenn_long, upenn_patients, ucsf_long, ucsf_patients = _build_synthetic_multi_region_cohort(
        n_upenn=200, n_ucsf=40, seed=302
    )
    expected_events = int((upenn_patients["vital_status"] == "DECEASED").sum())
    args = _fast_args(
        tmp_path,
        expected_patient_count=len(upenn_patients),
        expected_event_count=expected_events,
        allow_unexpected_ucsf_count=True,
    )
    selected = week3.select_v2_variant_configs(["v1_referans", "v2a_mgmt"])

    exit_code = week3.run_v2_variant_suite(
        upenn_long=upenn_long,
        upenn_patients=upenn_patients,
        ucsf_long=ucsf_long,
        ucsf_patients=ucsf_patients,
        args=args,
        output_dir=tmp_path,
        variants=selected,
    )
    assert exit_code == 0

    comparison = pd.read_csv(tmp_path / "week3_v2_variant_comparison.csv")
    assert set(comparison["variant"]) == {"v1_referans", "v2a_mgmt"}
    assert not (tmp_path / "week3_v2c_mgmt_spline_wttc_external_test.csv").exists()


def test_run_v2_variant_suite_v1_referans_reproduces_run_primary_single_model(tmp_path: Path) -> None:
    """v1_referans, `run_primary_single_model()`'in ürettiği modelle
    MATEMATİKSEL OLARAK AYNI olmalı -- aynı seed/argümanlarla aynı
    sonucu vermeli (görev talimatı: 'yeniden koşulması reprodüksiyon
    kontrolü olur')."""

    upenn_long, upenn_patients, ucsf_long, ucsf_patients = _build_synthetic_multi_region_cohort(
        n_upenn=200, n_ucsf=40, seed=303
    )
    expected_events = int((upenn_patients["vital_status"] == "DECEASED").sum())
    args = _fast_args(
        tmp_path,
        expected_patient_count=len(upenn_patients),
        expected_event_count=expected_events,
        allow_unexpected_ucsf_count=True,
    )

    v2_dir = tmp_path / "v2"
    v2_dir.mkdir()
    primary_dir = tmp_path / "primary"
    primary_dir.mkdir()

    week3.run_v2_variant_suite(
        upenn_long=upenn_long,
        upenn_patients=upenn_patients,
        ucsf_long=ucsf_long,
        ucsf_patients=ucsf_patients,
        args=args,
        output_dir=v2_dir,
        variants=week3.select_v2_variant_configs(["v1_referans"]),
    )
    week3.run_primary_single_model(
        upenn_long=upenn_long,
        upenn_patients=upenn_patients,
        ucsf_long=ucsf_long,
        ucsf_patients=ucsf_patients,
        args=args,
        output_dir=primary_dir,
    )

    v2_external = pd.read_csv(v2_dir / "week3_v1_referans_external_test.csv")
    primary_external = pd.read_csv(primary_dir / "week3_external_test.csv")

    assert v2_external.loc[0, "c_index"] == pytest.approx(primary_external.loc[0, "c_index"])
    assert v2_external.loc[0, "n_patients"] == primary_external.loc[0, "n_patients"]
    assert v2_external.loc[0, "n_events"] == primary_external.loc[0, "n_events"]


def test_main_variants_flag_routes_to_v2_variant_suite() -> None:
    """`--variants` CLI bayrağının argparse'ta doğru tanımlandığını
    (varsayılan None, geçerli seçenekler) doğrular -- gerçek DB'ye
    bağlanan `main()` uçtan uca burada test EDİLMEZ (kapsam dışı, bkz.
    dosyanın başlık notu -- bu dosya DB'ye hiç bağlanmaz)."""

    args = week3.parse_args([])
    assert args.variants is None

    args_v2a = week3.parse_args(["--variants", "v2a_mgmt"])
    assert args_v2a.variants == ["v2a_mgmt"]

    args_all = week3.parse_args(["--variants", *week3.V2_VARIANT_NAMES])
    assert args_all.variants == list(week3.V2_VARIANT_NAMES)


# =====================================================================
# 2026-09-11 -- reviewer v3c çapraz incelemesi HIGH bulgusu: v3 aday-
# havuz hesabı (`build_v3_radiomic_feature_pool()`) `run_single_v3_
# variant()` içinde `DECLARED_REGION_SHORTFALLS`'ta beyan edilmiş
# hastalar (TC=ET+NC=0 -> TC bloğu tamamen NaN) DÜŞÜRÜLMEDEN, `upenn_
# wide`'ın TAMAMI üzerinde çağrılıyordu. `pipeline.reduce_collinearity.
# build_v3_candidate_pool()`'un bugünkü katı `NonFiniteFeatureValueError`
# kapısı bu NaN'ları artık SESSİZCE geçirmiyor -- v3c ilk koşulduğunda
# (guard henüz yokken) pandas'ın örtük `skipna=True` davranışı aynı
# sonucu ürettiği için SONUÇ GEÇERLİ, ama kod artık kendi ürettiği
# artefaktı yeniden üretemiyordu. Düzeltme: `drop_declared_shortfall_
# patients_for_pool()` (yeni, `run_single_v3_variant()`'ın pool hesabına
# giden `upenn_wide` kopyasını hazırlar -- nihai `training_frame`
# ETKİLENMEZ).
# =====================================================================


def test_drop_declared_shortfall_patients_for_pool_wt_only_returns_input_unchanged() -> None:
    """WT-only'de (`resolve_declared_region_shortfall` `None` döner) bu
    adım devre dışı -- girdi AYNEN (hiçbir satır düşürülmeden) döner."""

    frame = pd.DataFrame(
        {"WT__f1": [1.0, 2.0, np.nan]}, index=["p1", "p2", "p3"]
    )
    frame.index.name = "patient_id"

    result = week3.drop_declared_shortfall_patients_for_pool(frame, ("WT",))

    pd.testing.assert_frame_equal(result, frame)


def test_drop_declared_shortfall_patients_for_pool_wt_tc_drops_only_declared_ids(capsys) -> None:
    """`("WT", "TC")` için beyan edilmiş `UPENN-GBM-00354`/`-00397`
    SADECE bu iki kimlik düşürülür, diğer hastalar/kolonlar dokunulmaz;
    düşme stderr'e AÇIKÇA loglanır (CLAUDE.md "sessiz filtreleme yasak")."""

    index = pd.Index(
        ["UPENN-GBM-00001", "UPENN-GBM-00354", "UPENN-GBM-00397", "UPENN-GBM-99999"],
        name="patient_id",
    )
    frame = pd.DataFrame(
        {
            "WT__f1": [1.0, 2.0, 3.0, 4.0],
            "TC__f1": [10.0, np.nan, np.nan, 40.0],
        },
        index=index,
    )

    result = week3.drop_declared_shortfall_patients_for_pool(frame, ("WT", "TC"))

    assert list(result.index) == ["UPENN-GBM-00001", "UPENN-GBM-99999"]
    assert not result.isna().any().any()

    captured = capsys.readouterr()
    assert "v3 ADAY HAVUZU HESABI" in captured.err
    assert "UPENN-GBM-00354" in captured.err
    assert "UPENN-GBM-00397" in captured.err
    assert "bölgeler=WT+TC" in captured.err


def test_drop_declared_shortfall_patients_for_pool_raises_if_declared_id_missing() -> None:
    """Beyan edilen bir kimlik `upenn_wide.index`'te YOKSA (veri/beyan
    uyuşmazlığı) SESSİZCE `errors="ignore"` ile geçilmez -- açık bir
    hata (fail-loud) fırlar."""

    index = pd.Index(["UPENN-GBM-00001", "UPENN-GBM-00397"], name="patient_id")
    frame = pd.DataFrame(
        {"WT__f1": [1.0, 2.0], "TC__f1": [10.0, np.nan]}, index=index
    )

    with pytest.raises(KeyError):
        week3.drop_declared_shortfall_patients_for_pool(frame, ("WT", "TC"))


def test_build_v3_radiomic_feature_pool_crashes_without_drop_but_succeeds_with_it() -> None:
    """Bu, HIGH bulgusunun TAM YENİDEN ÜRETİMİ: `DECLARED_REGION_
    SHORTFALLS`'taki 2 hastayı düşürmeden `build_v3_radiomic_feature_
    pool()`'a NaN'lı bir çerçeve verilirse `NonFiniteFeatureValueError`
    fırlar (guard'ın var oluş nedeni); `drop_declared_shortfall_patients_
    for_pool()` UYGULANDIKTAN SONRA aynı çağrı ÇÖKMEDEN döner."""

    ids = [f"UPENN-GBM-{i:05d}" for i in range(1, 21)]
    ids[3] = "UPENN-GBM-00354"
    ids[7] = "UPENN-GBM-00397"
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "WT__f1": rng.normal(size=len(ids)),
            "WT__f2": rng.normal(size=len(ids)),
            "TC__f1": rng.normal(size=len(ids)),
            "TC__f2": rng.normal(size=len(ids)),
        },
        index=pd.Index(ids, name="patient_id"),
    )
    # Uretim senaryosuyla AYNI: bu 2 hastanin TUM TC kolonlari NaN.
    frame.loc[["UPENN-GBM-00354", "UPENN-GBM-00397"], ["TC__f1", "TC__f2"]] = np.nan

    raw_candidate_columns = ["WT__f1", "WT__f2", "TC__f1", "TC__f2"]

    from pipeline.reduce_collinearity import NonFiniteFeatureValueError

    with pytest.raises(NonFiniteFeatureValueError):
        week3.build_v3_radiomic_feature_pool(
            frame, raw_candidate_columns, cv_threshold=0.02, corr_threshold=0.95
        )

    pool_input = week3.drop_declared_shortfall_patients_for_pool(frame, ("WT", "TC"))
    feature_columns, pool_report = week3.build_v3_radiomic_feature_pool(
        pool_input, raw_candidate_columns, cv_threshold=0.02, corr_threshold=0.95
    )
    assert feature_columns  # bos degil
    assert pool_report.n_input_features == 4


V3C_CACHE_UPENN_LONG_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "week3"
    / "cox_model"
    / "final_v3c"
    / "_cache"
    / "upenn_long.pkl"
)
V3C_HISTORICAL_POOL_REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "week3"
    / "cox_model"
    / "final_v3c"
    / "week3_v3c_lowvar_wttc_mgmt_nospline_v3_pool_report.csv"
)


@pytest.mark.skipif(
    not (V3C_CACHE_UPENN_LONG_PATH.is_file() and V3C_HISTORICAL_POOL_REPORT_PATH.is_file()),
    reason=(
        "final_v3c/_cache/upenn_long.pkl veya tarihsel v3_pool_report.csv "
        "bu makinede yok -- bu, DB'ye YENİDEN BAĞLANMADAN GERÇEK C32 "
        "verisiyle yapılan reproducibility kanıtı, dosya yoksa atlanır "
        "(CI/başka makine güvenliği için)."
    ),
)
def test_drop_declared_shortfall_patients_for_pool_reproduces_historical_v3c_pool_report() -> None:
    """REGRESYON KANITI (reviewer v3c çapraz incelemesi, 2026-09-11):
    gerçek `final_v3c/_cache/upenn_long.pkl` (611 satır, `UPenn-
    PyRadiomics-107-C32`) + YENİ `drop_declared_shortfall_patients_for_
    pool()` ile `build_v3_radiomic_feature_pool()` artık `NonFiniteFeature
    ValueError` FIRLATMADAN, `final_v3c/..._v3_pool_report.csv`'deki
    (guard'dan ÖNCE üretilmiş, pandas skipna'nın örtük eşdeğeri) near_
    constant (4) + correlation_cluster_duplicate (75) + final kept (107)
    kümeleriyle BİT-BİREBİR aynı sonucu üretir -- kod artık kendi
    ürettiği artefaktı ÇÖKMEDEN yeniden üretebiliyor."""

    from pipeline.cox_model import pivot_radiomics_long_to_wide

    upenn_long = pd.read_pickle(V3C_CACHE_UPENN_LONG_PATH)
    upenn_wide, pivot_report = pivot_radiomics_long_to_wide(upenn_long, regions=["WT", "TC"])
    assert upenn_wide.shape[0] == 611  # taban WT+TC pivot -- HENÜZ hiçbir hasta düşmedi

    raw_candidate_columns: list[str] = []
    for region in ("WT", "TC"):
        raw_candidate_columns += week3.select_feature_columns(
            upenn_wide, STABLE_FEATURES_ICC60, region=region
        )
    assert len(raw_candidate_columns) == 186  # 2 x 93 (WT + TC)

    pool_input = week3.drop_declared_shortfall_patients_for_pool(upenn_wide, ("WT", "TC"))
    assert pool_input.shape[0] == 609

    # Duzeltmeden ONCEki (guard'siz) davranisin AYNI sonucu urettigini de
    # dogrula -- yani bu drop adimi pandas'in skipna'siyla AYNI cevaba
    # varir, FARKLI bir cevaba DEGIL (sanity: coz drop islemi olculebilir
    # bir yan etki YARATMADI).
    import pipeline.reduce_collinearity as rc

    old_variability = rc.compute_feature_variability_summary(
        upenn_wide[raw_candidate_columns], raw_candidate_columns
    )
    old_near_constant = rc.select_near_constant_features(old_variability, cv_threshold=0.02)

    # NaN/±inf guard olmadan calisan alt adimlar (guard'i BILINCLI atla,
    # bu SADECE eski/skipna davranisiyla KARSILASTIRMA icin).
    old_survivors = [c for c in raw_candidate_columns if c not in set(old_near_constant)]
    old_clusters = rc.correlation_clusters_union_find(
        upenn_wide[raw_candidate_columns], old_survivors, corr_threshold=0.95
    )
    old_reps, old_dropped_report = rc.select_cluster_representatives(old_clusters, old_variability)

    feature_columns, pool_report = week3.build_v3_radiomic_feature_pool(
        pool_input, raw_candidate_columns, cv_threshold=0.02, corr_threshold=0.95
    )

    assert sorted(pool_report.near_constant_dropped) == sorted(old_near_constant)
    assert sorted(pool_report.correlation_dropped["dropped_feature"]) == sorted(
        old_dropped_report["dropped_feature"]
    )
    assert sorted(feature_columns) == sorted(old_reps)

    assert len(pool_report.near_constant_dropped) == 4
    assert len(pool_report.correlation_dropped) == 75
    assert len(feature_columns) == 107

    historical = pd.read_csv(V3C_HISTORICAL_POOL_REPORT_PATH)
    historical_near_constant = sorted(
        historical.loc[historical["reason"] == "near_constant_cv", "feature"]
    )
    historical_correlation_dropped = sorted(
        historical.loc[historical["reason"] == "correlation_cluster_duplicate", "feature"]
    )
    historical_kept = sorted(
        set(raw_candidate_columns) - set(historical_near_constant) - set(historical_correlation_dropped)
    )

    assert sorted(pool_report.near_constant_dropped) == historical_near_constant
    assert sorted(pool_report.correlation_dropped["dropped_feature"]) == historical_correlation_dropped
    assert sorted(feature_columns) == historical_kept


def test_run_single_v3_variant_wt_tc_declared_shortfall_end_to_end_no_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """TAM `run_single_v3_variant()` zincirini (pivot -> raw aday sütun
    -> `drop_declared_shortfall_patients_for_pool` -> `build_v3_radiomic_
    feature_pool` -> `build_training_frame` -> `check_training_pool_
    counts` -> nested-CV) sentetik WT+TC verisiyle, 2 hastanın TÜM TC
    kolonlarını NaN yaparak koşar -- `DECLARED_REGION_SHORTFALLS`
    sentetik ölçeğe UYARLANMIŞ (`monkeypatch`) sayılarla, ÇÖKMEDEN
    tamamlandığını ve nihai eğitim havuzunun beyan edilen 2 hastayı
    (SADECE onları) düşürdüğünü doğrular."""

    # n=200 (test_run_v2_variant_suite_respects_selected_variants_subset ile
    # AYNI olcek gerekcesi) -- MGMT'nin nadir "Indeterminate" kategorisi +
    # WT+TC'nin genis (186->107) aday havuzu kucuk n'de sifir-varyans/
    # yakinsamama riski tasiyor (ampirik gozlem, ilk versiyon n=60 ile
    # flaky RuntimeError verdi).
    upenn_long, upenn_patients, ucsf_long, ucsf_patients = _build_synthetic_multi_region_cohort(
        n_upenn=200, n_ucsf=40, seed=4242
    )

    shortfall_ids = list(upenn_patients.index[:2])
    tc_mask = upenn_long["patient_id"].isin(shortfall_ids) & (
        upenn_long["tumor_region"] == "TC_derived"
    )
    assert tc_mask.sum() == 2  # her iki hastanin da TC satiri VAR, sadece degerler NaN olacak

    def _all_nan(feature_dict: dict[str, float]) -> dict[str, float]:
        return {key: float("nan") for key in feature_dict}

    for col in ("shape_features", "first_order_features", "texture_features"):
        upenn_long.loc[tc_mask, col] = upenn_long.loc[tc_mask, col].apply(_all_nan)

    total_patients = len(upenn_patients)
    total_events = int((upenn_patients["vital_status"] == "DECEASED").sum())
    removed_events = int(
        (upenn_patients.loc[shortfall_ids, "vital_status"] == "DECEASED").sum()
    )
    expected_reduced_patients = total_patients - 2
    expected_reduced_events = total_events - removed_events

    synthetic_shortfall = week3.DeclaredRegionShortfall(
        n_patients=expected_reduced_patients,
        n_events=expected_reduced_events,
        dropped_patient_ids=tuple(shortfall_ids),
        reason="test-only sentetik beyan (HIGH duzeltmesi regresyon testi).",
        declared_on="2026-09-11",
    )
    monkeypatch.setattr(
        week3, "DECLARED_REGION_SHORTFALLS", {("WT", "TC"): synthetic_shortfall}
    )

    v3_config = week3.V3VariantConfig(
        name="v3_test_wttc_shortfall", base=week3.V2_VARIANT_CONFIGS_EXTRA[0]
    )
    args = _fast_args(
        tmp_path,
        expected_patient_count=total_patients,
        expected_event_count=total_events,
        allow_unexpected_ucsf_count=True,
    )

    result, pool_report = week3.run_single_v3_variant(
        config=v3_config,
        upenn_long=upenn_long,
        upenn_patients=upenn_patients,
        ucsf_long=ucsf_long,
        ucsf_patients=ucsf_patients,
        age_knots=None,
        args=args,
    )

    assert result.n_train_patients == expected_reduced_patients
    assert result.n_train_events == expected_reduced_events
    assert pool_report.kept_features  # bos degil

    captured = capsys.readouterr()
    assert "v3 ADAY HAVUZU HESABI" in captured.err
    assert "BEYAN EDILMIS HAVUZ AZALMASI" in captured.err or "BEYAN EDİLMİŞ HAVUZ AZALMASI" in captured.err


# =====================================================================
# A4 (2026-09-13) -- audit'in KUME (set) kontrolu
#
# NEDEN: `audit_nested_cv_fold_selection()` 2026-09-13'e kadar YALNIZ
# `int` vs `int` (ozellik SAYISI) karsilastiriyordu. AYNI BOYUTTA ama
# FARKLI uyelerden olusan bir kume sessizce geciyordu ve
# `week3_*_selection_frequency.csv` YANLIS bir fold'un frekanslariyla
# doluyordu. Codex'in 4 HIGH maddesinden TAMAMEN ACIK olani buydu.
# =====================================================================


def _make_audit_synthetic_frame(
    n: int, *, n_features: int, seed: int
) -> tuple[pd.DataFrame, list[str]]:
    """Audit testleri icin kucuk sentetik Cox cercevesi (DB YOK)."""

    rng = np.random.default_rng(seed)
    feature_columns = [f"feat_{i}" for i in range(n_features)]
    features = rng.normal(size=(n, n_features))
    linear_predictor = 1.4 * features[:, 0] + 0.7 * features[:, 1]
    u = rng.uniform(size=n)
    event_time = -np.log(u) / (0.02 * np.exp(linear_predictor))
    censoring_time = rng.uniform(low=1.0, high=np.percentile(event_time, 90), size=n)
    frame = pd.DataFrame(features, columns=feature_columns)
    frame["survival_days"] = np.minimum(event_time, censoring_time)
    frame["event"] = (event_time <= censoring_time).astype(int)
    return frame, feature_columns


_AUDIT_CV_KWARGS = dict(
    outer_splits=3,
    inner_splits=3,
    l1_ratio_grid=(1.0,),
    penalizer_grid=(0.1,),
    n_bootstrap_stability=12,
    stability_frequency_threshold=0.6,
    seed=13,
)


def _audit_nested_cv_pair() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    from pipeline.cox_model import run_nested_cv

    frame, feature_columns = _make_audit_synthetic_frame(120, n_features=6, seed=17)
    nested_cv_result = run_nested_cv(
        frame,
        feature_columns,
        stability_selection=True,
        **_AUDIT_CV_KWARGS,
    )
    return frame, feature_columns, nested_cv_result


def _call_audit(frame, feature_columns, nested_cv_result):
    return week3.audit_nested_cv_fold_selection(
        frame,
        feature_columns,
        nested_cv_result,
        outer_splits=_AUDIT_CV_KWARGS["outer_splits"],
        seed=_AUDIT_CV_KWARGS["seed"],
        n_bootstrap_stability=_AUDIT_CV_KWARGS["n_bootstrap_stability"],
        stability_frequency_threshold=_AUDIT_CV_KWARGS["stability_frequency_threshold"],
    )


def test_audit_nested_cv_fold_selection_accepts_faithful_result() -> None:
    """YESIL taban: audit, `run_nested_cv()`'nin GERCEK ciktisiyla
    cagrildiginda (hem sayi hem kume esit) hata vermez."""

    frame, feature_columns, nested_cv_result = _audit_nested_cv_pair()
    fallback_frame, frequency_frame = _call_audit(frame, feature_columns, nested_cv_result)

    assert sorted(fallback_frame["fold"].tolist()) == [0, 1, 2]
    assert not frequency_frame.empty
    assert set(frequency_frame["feature"]).issubset(set(feature_columns))


def test_audit_nested_cv_fold_selection_rejects_same_size_different_feature_set() -> None:
    """KIRMIZI kanit (Codex HIGH 4): AYNI BOYUTTA ama FARKLI uyeli bir
    kume artik SESSIZCE GECMEZ.

    Duzeltmeden ONCE bu senaryo hicbir hata uretmiyordu (`int` vs `int`
    karsilastirmasi gecerdi) ve `selection_frequency.csv` yanlis fold'u
    temsil ederdi."""

    frame, feature_columns, nested_cv_result = _audit_nested_cv_pair()

    fold0_mask = nested_cv_result["fold"] == 0
    selected = list(nested_cv_result.loc[fold0_mask, "selected_features"].iloc[0])
    assert selected, "test kurulumu: fold 0 en az bir ozellik secmis olmali"
    not_selected = [f for f in feature_columns if f not in selected]
    assert not_selected, (
        "test kurulumu: sahte kume icin SECILMEMIS en az bir gercek ozellik gerekli"
    )

    tampered_features = list(selected)
    swapped_out = tampered_features[0]
    swapped_in = not_selected[0]
    tampered_features[0] = swapped_in  # AYNI BOYUT, FARKLI KUME

    tampered = nested_cv_result.copy()
    tampered.at[tampered.index[fold0_mask][0], "selected_features"] = tampered_features
    # sayi kolonlari BILINCLI olarak DEGISTIRILMEDI -- eski (yalniz-sayi)
    # kontrol bu cerceveyi GECIRIRDI; testin anlami tam olarak budur.
    assert int(tampered.loc[fold0_mask, "n_selected_features_final"].iloc[0]) == len(
        tampered_features
    )

    with pytest.raises(week3.FallbackAuditMismatchError) as excinfo:
        _call_audit(frame, feature_columns, tampered)

    message = str(excinfo.value)
    assert "KUMESI" in message
    # hata mesaji HANGI ozelliklerin farkli oldugunu yazmali (sadece sayi degil)
    assert swapped_out in message
    assert swapped_in in message


def test_audit_nested_cv_fold_selection_keeps_count_check_as_well() -> None:
    """Mevcut SAYI kontrolu KALDIRILMADI: kume kontrolu EK bir katman."""

    frame, feature_columns, nested_cv_result = _audit_nested_cv_pair()
    tampered = nested_cv_result.copy()
    fold0_index = tampered.index[tampered["fold"] == 0][0]
    tampered.at[fold0_index, "n_selected_features_final"] = (
        int(tampered.at[fold0_index, "n_selected_features_final"]) + 3
    )

    with pytest.raises(week3.FallbackAuditMismatchError) as excinfo:
        _call_audit(frame, feature_columns, tampered)

    assert "n_selected_features_final" in str(excinfo.value)


def test_audit_nested_cv_fold_selection_fallback_branch_reconstructs_elastic_net_set() -> None:
    """FALLBACK DALI regresyonu (K14 `clinical_base` sekli): radyomik aday
    havuzu BOS (`feature_columns=[]`) oldugunda stabilite kumesi de bos
    kalir -> `used_fallback=True`. Kume kontrolu bu dalda elastic-net
    secimini `select_features_lasso()` ile YENIDEN URETIR; bu testin isi,
    o EK cagrinin gercek K14 seklinde CALISTIGINI (catlamadigini) ve
    audit'in yine de tutarli cikti verdigini kanitlamaktir.

    Gercek artefaktlarda `used_fallback=True` olan TEK kol K14'tur
    (olculdu: week3_k14_clinical_base_ucsf_fold_results.csv -- 5/5 fold
    fallback, n_selected_features_final=0)."""

    from pipeline.cox_model import run_nested_cv

    rng = np.random.default_rng(3)
    n = 120
    age = rng.normal(60.0, 10.0, n)
    u = rng.uniform(size=n)
    event_time = -np.log(u) / (0.02 * np.exp(0.04 * (age - 60.0)))
    censoring_time = rng.uniform(1.0, np.percentile(event_time, 90), n)
    frame = pd.DataFrame(
        {
            "clinical_age": age,
            "clinical_gender_male": rng.integers(0, 2, n).astype(float),
        }
    )
    frame["survival_days"] = np.minimum(event_time, censoring_time)
    frame["event"] = (event_time <= censoring_time).astype(int)
    extra_columns = ["clinical_age", "clinical_gender_male"]

    nested_cv_result = run_nested_cv(
        frame,
        [],
        extra_columns=extra_columns,
        stability_selection=True,
        extra_column_penalizer=0.0,
        **_AUDIT_CV_KWARGS,
    )
    assert (nested_cv_result["n_selected_features_final"] == 0).all()

    fallback_frame, frequency_frame = week3.audit_nested_cv_fold_selection(
        frame,
        [],
        nested_cv_result,
        extra_columns=extra_columns,
        outer_splits=_AUDIT_CV_KWARGS["outer_splits"],
        seed=_AUDIT_CV_KWARGS["seed"],
        n_bootstrap_stability=_AUDIT_CV_KWARGS["n_bootstrap_stability"],
        stability_frequency_threshold=_AUDIT_CV_KWARGS["stability_frequency_threshold"],
        extra_column_penalizer=0.0,
    )

    assert fallback_frame["used_fallback"].all()
    assert frequency_frame.empty  # radyomik aday YOK -> frekans satiri da YOK


def test_audit_nested_cv_fold_selection_rejects_non_list_selected_features() -> None:
    """CSV'den geri okunmus (string) bir `selected_features` kolonu kume
    kontrolunu SESSIZCE etkisizlestirmemeli -- audit bunu ACIKCA
    reddeder."""

    frame, feature_columns, nested_cv_result = _audit_nested_cv_pair()
    tampered = nested_cv_result.copy()
    tampered["selected_features"] = tampered["selected_features"].apply(str)

    with pytest.raises(week3.FallbackAuditMismatchError) as excinfo:
        _call_audit(frame, feature_columns, tampered)

    assert "liste degil" in str(excinfo.value)


# =====================================================================
# A3 (2026-09-13) -- protokol-disi stabilite esigi UYARISI
# (Baris: "10) icin de beyan arti uyari eklensin" -- DURDURMA DEGIL)
# =====================================================================


def test_off_protocol_primary_threshold_emits_loud_warning(capsys) -> None:
    emitted = week3.warn_if_off_protocol_primary_threshold(0.5)
    captured = capsys.readouterr()

    assert emitted is True
    assert "PROTOKOL DISI ESIK" in captured.err
    assert "0.5" in captured.err
    assert "0.6" in captured.err
    # uyari STDOUT'u kirletmemeli (artefakt/log ayrimi)
    assert "PROTOKOL DISI ESIK" not in captured.out


def test_protocol_primary_threshold_is_silent(capsys) -> None:
    emitted = week3.warn_if_off_protocol_primary_threshold(0.6)
    captured = capsys.readouterr()

    assert emitted is False
    assert captured.err == ""
    assert captured.out == ""


def test_parse_args_warns_on_off_protocol_primary_threshold(capsys) -> None:
    """BAGLANTI (wiring) testi: uyari yalniz yardimci fonksiyonda degil,
    CLI yolunda da tetikleniyor."""

    args = week3.parse_args(["--primary-threshold", "0.5"])
    captured = capsys.readouterr()

    assert args.primary_threshold == 0.5
    assert "PROTOKOL DISI ESIK" in captured.err
    assert "Kosu DURDURULMADI" in captured.err


def test_parse_args_default_primary_threshold_is_silent_and_locked(capsys) -> None:
    args = week3.parse_args([])
    captured = capsys.readouterr()

    assert args.primary_threshold == week3.PROTOCOL_PRIMARY_STABILITY_THRESHOLD == 0.6
    assert "PROTOKOL DISI ESIK" not in captured.err


# =====================================================================
# TIMEPOINT KAPISI -- 611/585 kilidinin IKINCI KATMANI
# (2026-09-13/14, karar 39; modeling-agent-I)
#
# Birinci katman: `tools/write_upenn_pyradiomics_to_db.py` GUARD 6
# (yazma tarafi). Buradaki testler OKUMA tarafini kilitler: `radiomics`
# tablosuna baska bir yolla post-op satir girse bile Cox egitim havuzuna
# GIRMEMELI ve dislama SESSIZ olmamali.
# =====================================================================

_TP_FEATURE_COLUMNS = ("shape_features", "first_order_features", "texture_features")


def _tp_row(
    *,
    scan_id: int,
    patient_id: str,
    timepoint_label: str | None,
    tool: str = "UPenn-PyRadiomics-107-C32",
    source: str = "UPenn-GBM",
    tumor_region: str = "WT_derived",
) -> dict:
    row = {
        "scan_id": scan_id,
        "patient_id": patient_id,
        "source": source,
        "tumor_region": tumor_region,
        "timepoint_label": timepoint_label,
        "segmentation_tool": tool,
    }
    for col in _TP_FEATURE_COLUMNS:
        row[col] = {f"{col}_dummy": 1.0}
    return row


class _FakeTimepointCursor:
    """`radiomics JOIN mr_scans` sorgularinin semantigini TAKLIT eden
    kursor -- DB'ye HIC baglanmaz.

    METIN-DUYARLI (bilincli): timepoint filtresini YALNIZCA sorgu
    metninde `ms.timepoint_label = ANY(%s)` GERCEKTEN varsa uygular.
    Boylece uretim SQL'indeki kosul ileride silinir/etkisizlestirilirse
    taklit kursor de filtrelemeyi BIRAKIR -> post-op satir cerceveye
    "sizar" -> `assert_frame_timepoints_allowed()` patlar (KIRMIZI).
    Sorgu metnine bakmayan bir taklit bu mutasyonu HIC goremezdi.

    `honor_timepoint_filter=False` ayrica ACIK bir MUTANTTIR: kosul
    metinde olsa bile yok sayilir (kapidan ONCEKI DB davranisi).
    """

    TIMEPOINT_SQL_CLAUSE = "ms.timepoint_label = ANY(%s)"

    def __init__(self, rows: list[dict], *, honor_timepoint_filter: bool = True) -> None:
        self._rows = [dict(r) for r in rows]
        self.honor_timepoint_filter = honor_timepoint_filter
        self._result: list[dict] = []
        self.queries: list[tuple[str, tuple]] = []

    def _passes(self, row: dict, allowed: set[str], query: str) -> bool:
        if not self.honor_timepoint_filter:  # MUTANT: filtre yok
            return True
        if self.TIMEPOINT_SQL_CLAUSE not in query:  # SQL metni bozulmus
            return True
        label = row["timepoint_label"]
        return label is not None and label in allowed

    def execute(self, query: str, params=None) -> None:
        params = tuple(params or ())
        self.queries.append((query, params))
        tool = params[0]
        rows = [r for r in self._rows if r["segmentation_tool"] == tool]
        allowed = (
            set(params[1])
            if len(params) > 1 and isinstance(params[1], (list, tuple, set, frozenset))
            else set()
        )

        if "ARRAY_AGG" in query:
            buckets: dict[str, list[dict]] = {}
            for row in rows:
                label = row["timepoint_label"]
                if label is not None and label in allowed:
                    continue
                buckets.setdefault("<NULL>" if label is None else label, []).append(row)
            self._result = [
                {
                    "timepoint_label": label,
                    "n_rows": len(group),
                    "n_patients": len({r["patient_id"] for r in group}),
                    "patient_ids": sorted({r["patient_id"] for r in group}),
                }
                for label, group in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            ]
            return

        if "COUNT(*) AS n" in query:
            self._result = [{"n": sum(1 for r in rows if self._passes(r, allowed, query))}]
            return

        selected = [r for r in rows if self._passes(r, allowed, query)]
        selected.sort(key=lambda r: (r["scan_id"], r["tumor_region"]))
        page_size, offset = params[-2], params[-1]
        keys = ("scan_id", "patient_id", "source", "tumor_region", "timepoint_label") + _TP_FEATURE_COLUMNS
        self._result = [{k: r[k] for k in keys} for r in selected[offset : offset + page_size]]

    def fetchall(self) -> list[dict]:
        return list(self._result)

    def fetchone(self):
        return self._result[0] if self._result else None


def _upenn_pool_rows() -> list[dict]:
    """4 pre-treatment hasta + 2 YALNIZ post-op hasta (19'un minyaturu)
    + 1 hastanin IKINCI (post-op) scan'i (41 coklama vakasinin minyaturu).
    """

    rows = [
        _tp_row(scan_id=i, patient_id=f"UPENN-{i:03d}", timepoint_label="pre-treatment")
        for i in (1, 2, 3, 4)
    ]
    rows += [
        _tp_row(scan_id=101, patient_id="UPENN-101", timepoint_label="post-op"),
        _tp_row(scan_id=102, patient_id="UPENN-102", timepoint_label="post-op"),
        # ayni hastanin (UPENN-001) post-op ikinci scan'i -> coklama riski
        _tp_row(scan_id=103, patient_id="UPENN-001", timepoint_label="post-op"),
    ]
    return rows


def test_allowed_timepoint_labels_are_tool_specific_not_a_single_literal() -> None:
    """KRITIK: UPenn 'pre-treatment', TCGA/UCSF 'baseline' (canli DB
    olcumu 2026-09-13: UPenn 3055/611 %100 pre-treatment; TCGA 78/39 ve
    UCSF 1475/295 %100 baseline). Tek literal 'pre-treatment' kosulu
    HARICI TEST SETINI (295/169) SIFIRLARDI."""

    assert week3.allowed_timepoint_labels(week3.SEGMENTATION_TOOL_UPENN_C32) == ("pre-treatment",)
    assert week3.allowed_timepoint_labels(week3.SEGMENTATION_TOOL_TCGA_C32) == ("baseline",)
    assert week3.allowed_timepoint_labels(week3.SEGMENTATION_TOOL_UCSF_C32) == ("baseline",)


def test_timepoint_maps_cover_exactly_the_c32_whitelist() -> None:
    assert set(week3.ALLOWED_TIMEPOINT_LABELS_BY_TOOL) == set(week3.ALLOWED_C32_SEGMENTATION_TOOLS)
    assert set(week3.KNOWN_EXCLUDED_TIMEPOINT_LABELS_BY_TOOL) == set(
        week3.ALLOWED_C32_SEGMENTATION_TOOLS
    )
    assert week3.KNOWN_EXCLUDED_TIMEPOINT_LABELS_BY_TOOL[
        week3.SEGMENTATION_TOOL_UPENN_C32
    ] == frozenset({"post-op"})


def test_allowed_timepoint_labels_fail_closed_for_undeclared_tool() -> None:
    """Beyan edilmemis arac -> filtresiz cekime DUSULMEZ, SERT durulur."""

    with pytest.raises(week3.TimepointWhitelistMissingError, match="TANIMLI DEGIL"):
        week3.allowed_timepoint_labels("LUMIERE-PyRadiomics-107-C32")


def test_fetch_excludes_post_op_rows_and_reports_them(capsys) -> None:
    cursor = _FakeTimepointCursor(_upenn_pool_rows())

    frame = week3.fetch_c32_radiomics_long_frame(
        cursor, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
    )
    captured = capsys.readouterr()

    # YESIL: yalniz 4 pre-treatment hasta; post-op'lar YOK
    assert len(frame) == 4
    assert sorted(frame["patient_id"]) == ["UPENN-001", "UPENN-002", "UPENN-003", "UPENN-004"]
    assert "UPENN-101" not in set(frame["patient_id"])
    assert "UPENN-102" not in set(frame["patient_id"])
    # ayni hastanin post-op ikinci satiri da elendi -> coklama yok
    assert list(frame["patient_id"]).count("UPENN-001") == 1
    # pivot sozlesmesi DEGISMEDI (yardimci kolonlar dusuruldu)
    assert "timepoint_label" not in frame.columns
    assert "scan_id" not in frame.columns

    # SESSIZ DEGIL: dislama ozeti stderr'e basildi, sebep + kimlik dahil
    assert "TIMEPOINT DISLAMA OZETI" in captured.err
    assert "post-op" in captured.err
    assert "UPENN-101" in captured.err and "UPENN-102" in captured.err
    assert "3 satir / 3 hasta" in captured.err
    assert "HAVUZA GIRMEDI" in captured.err


def test_fetch_keeps_every_baseline_row_for_ucsf_external_test(capsys) -> None:
    """Harici test kolunun SIFIRLANMADIGININ kaniti."""

    rows = [
        _tp_row(
            scan_id=900 + i,
            patient_id=f"UCSF-{i:03d}",
            timepoint_label="baseline",
            tool=week3.SEGMENTATION_TOOL_UCSF_C32,
            source="UCSF-PDGM",
        )
        for i in range(5)
    ]
    cursor = _FakeTimepointCursor(rows)

    frame = week3.fetch_c32_radiomics_long_frame(
        cursor, segmentation_tool=week3.SEGMENTATION_TOOL_UCSF_C32
    )
    captured = capsys.readouterr()

    assert len(frame) == 5
    assert sorted(frame["patient_id"]) == [f"UCSF-{i:03d}" for i in range(5)]
    # dislanan satir YOK -> gurultu de YOK
    assert "TIMEPOINT DISLAMA OZETI" not in captured.err


def test_fetch_paginates_over_filtered_rows() -> None:
    """Sayfalama filtrelenmis kume uzerinde dogru yuruyor (page_size=250,
    burada 300 izinli + 60 post-op satir)."""

    rows = [
        _tp_row(scan_id=i, patient_id=f"UPENN-{i:04d}", timepoint_label="pre-treatment")
        for i in range(300)
    ]
    rows += [
        _tp_row(scan_id=1000 + i, patient_id=f"POSTOP-{i:04d}", timepoint_label="post-op")
        for i in range(60)
    ]
    cursor = _FakeTimepointCursor(rows)

    frame = week3.fetch_c32_radiomics_long_frame(
        cursor, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
    )

    assert len(frame) == 300
    assert not any(str(pid).startswith("POSTOP") for pid in frame["patient_id"])


def test_fetch_sql_text_carries_the_timepoint_condition_and_params() -> None:
    """SQL SOZLESMESI kilidi: kosul GERCEKTEN sorgu metninde ve dogru
    parametreyle baglanmis olmali. (Taklit kursor semantigi tek basina
    yeterli kanit degildir -- uretim sorgusunun METNI de dogrulaniyor.)"""

    cursor = _FakeTimepointCursor(_upenn_pool_rows())
    week3.fetch_c32_radiomics_long_frame(
        cursor, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
    )

    page_queries = [(q, p) for q, p in cursor.queries if "ARRAY_AGG" not in q]
    assert page_queries, "sayfa sorgusu hic kosmadi"
    for query, params in page_queries:
        assert "JOIN mr_scans ms ON ms.scan_id = r.scan_id" in query
        assert "AND ms.timepoint_label = ANY(%s)" in query
        assert params[0] == week3.SEGMENTATION_TOOL_UPENN_C32
        assert params[1] == ["pre-treatment"]

    # dislama ozeti sorgusu da AYNI beyaz listeyi kullaniyor
    summary_queries = [(q, p) for q, p in cursor.queries if "ARRAY_AGG" in q]
    assert len(summary_queries) == 1
    assert summary_queries[0][1][1] == ["pre-treatment"]


def test_fetch_raises_if_sql_timepoint_condition_is_neutralized() -> None:
    """MUTANT: kursor `ms.timepoint_label = ANY(%s)` kosulunu YOK SAYAR
    (kapidan onceki davranis). Post-op satir SESSIZCE cerceveye girmek
    yerine `assert_frame_timepoints_allowed()` tarafindan yakalanmali."""

    cursor = _FakeTimepointCursor(_upenn_pool_rows(), honor_timepoint_filter=False)

    with pytest.raises(week3.UnexpectedTimepointLabelError, match="IZINLI OLMAYAN"):
        week3.fetch_c32_radiomics_long_frame(
            cursor, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
        )


def test_assert_frame_timepoints_allowed_flags_null_label() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "timepoint_label": ["pre-treatment", None],
        }
    )
    with pytest.raises(week3.UnexpectedTimepointLabelError, match="NULL"):
        week3.assert_frame_timepoints_allowed(
            frame, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
        )


def test_assert_frame_timepoints_allowed_passes_on_clean_frame() -> None:
    frame = pd.DataFrame({"patient_id": ["p1"], "timepoint_label": ["pre-treatment"]})
    week3.assert_frame_timepoints_allowed(
        frame, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
    )


def test_summarize_timepoint_exclusions_hard_fails_on_null_label(capsys) -> None:
    rows = _upenn_pool_rows() + [
        _tp_row(scan_id=201, patient_id="UPENN-201", timepoint_label=None)
    ]
    cursor = _FakeTimepointCursor(rows)

    with pytest.raises(week3.UnexpectedTimepointLabelError, match="ANALIZ EDILMEMIS"):
        week3.summarize_timepoint_exclusions(
            cursor, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
        )
    captured = capsys.readouterr()

    # dislama SESSIZ olmadi: once rapor, sonra istisna
    assert "<NULL>" in captured.err
    assert "ANALIZ EDILMEMIS" in captured.err
    assert "UPENN-201" in captured.err


def test_summarize_timepoint_exclusions_hard_fails_on_unknown_label(capsys) -> None:
    rows = [_tp_row(scan_id=1, patient_id="UPENN-001", timepoint_label="week-004")]
    cursor = _FakeTimepointCursor(rows)

    with pytest.raises(week3.UnexpectedTimepointLabelError, match="week-004"):
        week3.summarize_timepoint_exclusions(
            cursor, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
        )
    assert "week-004" in capsys.readouterr().err


def test_fetch_stops_before_paging_when_label_unanalyzed() -> None:
    """Analiz edilmemis etiket varsa 11 dakikalik cekim BASLAMADAN durur
    (yalniz ozet sorgusu kosmus olmali)."""

    cursor = _FakeTimepointCursor(
        [_tp_row(scan_id=1, patient_id="UPENN-001", timepoint_label=None)]
    )
    with pytest.raises(week3.UnexpectedTimepointLabelError):
        week3.fetch_c32_radiomics_long_frame(
            cursor, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
        )

    assert len(cursor.queries) == 1
    assert "ARRAY_AGG" in cursor.queries[0][0]


def test_summarize_timepoint_exclusions_returns_structured_counts() -> None:
    cursor = _FakeTimepointCursor(_upenn_pool_rows())
    excluded = week3.summarize_timepoint_exclusions(
        cursor, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
    )

    assert set(excluded) == {"post-op"}
    assert excluded["post-op"]["n_rows"] == 3
    assert excluded["post-op"]["n_patients"] == 3
    assert excluded["post-op"]["patient_ids"] == ["UPENN-001", "UPENN-101", "UPENN-102"]


def test_summarize_timepoint_exclusions_silent_when_nothing_excluded(capsys) -> None:
    """Bugunun canli durumu (UPenn C32 %100 pre-treatment): hicbir
    gurultu uretmemeli -- kapi mevcut veriyi DEGISTIRMIYOR."""

    rows = [
        _tp_row(scan_id=i, patient_id=f"UPENN-{i:03d}", timepoint_label="pre-treatment")
        for i in (1, 2, 3)
    ]
    cursor = _FakeTimepointCursor(rows)

    excluded = week3.summarize_timepoint_exclusions(
        cursor, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
    )
    captured = capsys.readouterr()

    assert excluded == {}
    assert captured.err == ""
    assert captured.out == ""


def test_count_c32_radiomics_rows_applies_same_timepoint_filter() -> None:
    cursor = _FakeTimepointCursor(_upenn_pool_rows())
    assert (
        week3.count_c32_radiomics_rows(
            cursor, segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
        )
        == 4
    )


def test_count_c32_radiomics_rows_matches_fetch_length() -> None:
    """Onbellek gecerlilik kontrolu `len(frame) == db_rows` karsilastirmasi
    yapiyor -- iki sorgu AYNI kumeyi gormezse kapi dogru calistigi halde
    onbellek sonsuza kadar gecersiz sayilirdi."""

    rows = _upenn_pool_rows()
    frame = week3.fetch_c32_radiomics_long_frame(
        _FakeTimepointCursor(rows), segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
    )
    counted = week3.count_c32_radiomics_rows(
        _FakeTimepointCursor(rows), segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
    )
    assert counted == len(frame)


def test_count_c32_radiomics_rows_rejects_non_c32_tool() -> None:
    cursor = _FakeTimepointCursor([])
    with pytest.raises(week3.C32WhitelistViolationError):
        week3.count_c32_radiomics_rows(cursor, segmentation_tool="UPenn-PyRadiomics-107")


def test_training_pool_gate_sees_only_pre_treatment_patients() -> None:
    """BAGLANTI testi: kapinin cikardigi hasta kumesi dogrudan
    `check_training_pool_counts()`'a besleniyor. Minyatur olcek: 4
    pre-treatment hasta (canlidaki 611'in yerine) + 3 post-op satir
    (19 hasta / 41 coklama vakasinin yerine)."""

    rows = _upenn_pool_rows()
    frame = week3.fetch_c32_radiomics_long_frame(
        _FakeTimepointCursor(rows), segmentation_tool=week3.SEGMENTATION_TOOL_UPENN_C32
    )
    n_patients = int(frame["patient_id"].nunique())

    # kapi ACIK -> beklenen sayi tutuyor (minyaturde 4/4)
    week3.check_training_pool_counts(
        n_patients, n_patients, expected_patients=4, expected_events=4, regions=("WT",)
    )

    # kapi KAPALI olsaydi havuz 6 HASTAYA cikardi; WT-only kapisi bu
    # durumda SERT durur (beyan edilmis bir azalma/artis YOK)
    mutant_patients = {r["patient_id"] for r in rows}
    assert len(mutant_patients) == 6
    with pytest.raises(week3.TrainingPoolCountMismatchError):
        week3.check_training_pool_counts(
            len(mutant_patients),
            len(mutant_patients),
            expected_patients=4,
            expected_events=4,
            regions=("WT",),
        )


def test_expected_upenn_pool_constants_still_locked() -> None:
    """Kilitli degerler bu degisiklikle OYNANMADI."""

    assert week3.EXPECTED_UPENN_TRAINING_PATIENTS == 611
    assert week3.EXPECTED_UPENN_TRAINING_EVENTS == 585



# =====================================================================
# 2026-09-14 (modeling-agent-S1) -- DAMGALAMA/HIJYEN testleri
# Baris'in sarti: "yanlis degerler gormek istemiyorum, herhangi bir
# zamanda." Uc bosluk kapatildi:
#   IS 1: kol adindaki `th..` etiketi SABIT DIZGIYDI, artik gercek
#         esikten turetiliyor.
#   IS 2: kol checkpoint'i parmak izi TASIMIYORDU, artik fail-closed.
#   IS 3: `*_fold_results.csv` esigi TASIMIYORDU, artik sutunu var.
# Bu testler SAYI URETMEZ -- model kosulmaz, yalniz damgalama denetlenir.
# =====================================================================


def test_format_threshold_tag_geriye_donuk_uyumlu() -> None:
    """PROTOKOL VARSAYILANLARI BUGUNKU ADI URETMELI.

    Diskteki artefaktlar ve onlara ATIF VEREN kod bu adlara bagli
    (`api/predict.py:397`, `demo/demo_helpers.py:151`,
    `tools/shap_explainer_prototype.py:76`,
    `tools/_dogrulama/inspect_checkpoint*.py`). Bu iki assert KIRILIRSA
    o tuketiciler de kirilir -- degistirmeden once oralara bak.
    """

    assert week3.format_threshold_tag(0.6) == "th06"
    assert week3.format_threshold_tag(0.7) == "th07"


def test_format_threshold_tag_protokol_disi_esikleri_ayirt_eder() -> None:
    """ASIL HATA: 0,5 ile kosulan cikti `th06` adiyla yaziliyordu."""

    assert week3.format_threshold_tag(0.5) == "th05"
    assert week3.format_threshold_tag(0.65) == "th065"
    assert week3.format_threshold_tag(0.55) == "th055"
    # farkli esikler farkli etiket uretmeli (ad -> esik ayirt edilebilir)
    tags = {week3.format_threshold_tag(v) for v in (0.5, 0.55, 0.6, 0.65, 0.7)}
    assert len(tags) == 5


def test_arm_configs_adlari_gercek_esikten_turetilir(tmp_path: Path) -> None:
    """`--primary-threshold 0.5` ile kosulan kol `th05` adini almali.

    KIRMIZI/YESIL: `arm_configs` sabit dizgiye geri alinirsa bu test
    `..._th06_...` gorur ve DUSER.
    """

    upenn_long, upenn_patients, tcga_long, tcga_patients = _build_synthetic_cohort(
        n_upenn=90, n_tcga=30, seed=404
    )
    expected_events = int((upenn_patients["vital_status"] == "DECEASED").sum())
    args = _fast_args(
        tmp_path,
        primary_threshold=0.5,
        sensitivity_threshold=0.9,
        expected_patient_count=len(upenn_patients),
        expected_event_count=expected_events,
        skip_full_107_arm=True,
    )

    exit_code = week3.run_pipeline(
        upenn_long=upenn_long,
        upenn_patients=upenn_patients,
        tcga_long=tcga_long,
        tcga_patients=tcga_patients,
        args=args,
        output_dir=tmp_path,
    )
    assert exit_code == 0

    folds = pd.read_csv(tmp_path / "week3_fold_results.csv")
    assert set(folds["arm"]) == {
        "primary_wt93_icc60_th05",
        "sensitivity_wt93_icc60_th09",
    }
    # IS 3: esik sutunu CSV'de ve kol adiyla TUTARLI
    esikler = folds.groupby("arm")["stability_frequency_threshold"].unique()
    assert list(esikler["primary_wt93_icc60_th05"]) == [0.5]
    assert list(esikler["sensitivity_wt93_icc60_th09"]) == [0.9]

    # checkpoint dosya adlari da gercek esigi tasimali
    ckpt_names = {p.name for p in (tmp_path / "_checkpoints").glob("arm_*.pkl")}
    assert ckpt_names == {
        "arm_primary_wt93_icc60_th05.pkl",
        "arm_sensitivity_wt93_icc60_th09.pkl",
    }


def test_arm_checkpoint_parmak_izi_esik_degisince_yeniden_hesaplatir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """IS 2 -- ASIL SENARYO: 0,5 ile yazildi, 0,6 ile okunmaya calisildi.

    Parmak izi OLMADAN `pd.read_pickle` sessizce 0,5'in sonucunu dondurur
    ve "0,6 birincil" diye raporlanirdi. Artik `None` donmeli.
    """

    ckpt = tmp_path / "arm_primary_wt93_icc60_th05.pkl"
    yazilan = week3.build_arm_checkpoint_fingerprint(
        threshold=0.5,
        seed=7,
        outer_splits=3,
        inner_splits=2,
        n_bootstrap_stability=12,
        n_bootstrap_external=40,
        feature_set="wt93_icc60",
        l1_ratio_grid=(0.5, 1.0),
        penalizer_grid=(0.2, 0.5),
    )
    week3.save_arm_checkpoint(ckpt, {"sonuc": "0.5 ile uretildi"}, yazilan)
    assert week3.arm_checkpoint_meta_path(ckpt).is_file()

    # AYNI parmak izi -> yuklenir
    assert (
        week3.load_arm_checkpoint(ckpt, yazilan, arm_name="x")["sonuc"]
        == "0.5 ile uretildi"
    )

    # ESIK degisti -> YUKLENMEZ
    istenen = dict(yazilan, threshold=0.6)
    assert week3.load_arm_checkpoint(ckpt, istenen, arm_name="x") is None
    hata = capsys.readouterr().err
    assert "PARMAK IZI UYUSMUYOR" in hata
    assert "threshold" in hata


@pytest.mark.parametrize(
    "alan,yeni_deger",
    [
        ("seed", 999),
        ("outer_splits", 5),
        ("inner_splits", 5),
        ("n_bootstrap_stability", 200),
        ("n_bootstrap_external", 1000),
        ("feature_set", "wt107_full"),
        ("l1_ratio_grid", [1.0]),
        ("penalizer_grid", [0.1]),
    ],
)
def test_arm_checkpoint_parmak_izi_her_alani_kontrol_eder(
    tmp_path: Path, alan: str, yeni_deger: object
) -> None:
    """Parmak izinin HER alani fail-closed olmali -- tek alan atlanirsa
    o alan uzerinden sessiz yanlis sonuc gecebilir."""

    ckpt = tmp_path / "arm_test.pkl"
    yazilan = week3.build_arm_checkpoint_fingerprint(
        threshold=0.6,
        seed=42,
        outer_splits=3,
        inner_splits=2,
        n_bootstrap_stability=12,
        n_bootstrap_external=40,
        feature_set="wt93_icc60",
        l1_ratio_grid=(0.5, 1.0),
        penalizer_grid=(0.2, 0.5),
    )
    week3.save_arm_checkpoint(ckpt, {"sonuc": 1}, yazilan)
    assert week3.load_arm_checkpoint(ckpt, yazilan, arm_name="x") is not None
    assert (
        week3.load_arm_checkpoint(
            ckpt, dict(yazilan, **{alan: yeni_deger}), arm_name="x"
        )
        is None
    )


def test_arm_checkpoint_meta_dosyasi_yoksa_fail_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Parmak izi KAPISINDAN ONCE yazilmis eski checkpoint'ler (meta YOK)
    sessizce KULLANILMAZ -- `_cached_fetch`'teki `timepoint_labels`
    anahtari eksikse fail-closed davranisiyla AYNI disiplin."""

    ckpt = tmp_path / "arm_eski.pkl"
    fingerprint = week3.build_arm_checkpoint_fingerprint(
        threshold=0.6,
        seed=42,
        outer_splits=3,
        inner_splits=2,
        n_bootstrap_stability=12,
        n_bootstrap_external=40,
        feature_set="wt93_icc60",
    )
    pd.to_pickle({"sonuc": "eski"}, ckpt)  # meta YAZILMADI (eski davranis)

    assert week3.load_arm_checkpoint(ckpt, fingerprint, arm_name="eski") is None
    assert "PARMAK IZI DOSYASI YOK" in capsys.readouterr().err


def test_arm_checkpoint_parmak_izi_json_roundtrip_sonrasi_esler(
    tmp_path: Path,
) -> None:
    """Tuple->list normalizasyonu YAPILMAZSA her karsilastirma
    "uyusmuyor" derdi (JSON tuple tanimaz) -- checkpoint HIC
    kullanilamaz hale gelirdi. Bu test o regresyonu yakalar."""

    ckpt = tmp_path / "arm_roundtrip.pkl"
    fingerprint = week3.build_arm_checkpoint_fingerprint(
        threshold=0.6,
        seed=42,
        outer_splits=3,
        inner_splits=2,
        n_bootstrap_stability=12,
        n_bootstrap_external=40,
        feature_set="wt93_icc60",
        l1_ratio_grid=(0.5, 1.0),  # TUPLE
        penalizer_grid=(0.2, 0.5),  # TUPLE
    )
    week3.save_arm_checkpoint(ckpt, {"sonuc": 1}, fingerprint)
    assert week3.load_arm_checkpoint(ckpt, fingerprint, arm_name="x") is not None


# =====================================================================
# K22 (2026-09-13) -- BAYAT PROVENANCE METNI REGRESYON KORUMASI
#
# Sorun: `--clinical-extra-column-penalizer` CLI yardim metni A6 karariyla
# (2026-09-13, Baris onayi) "KILITLENDI" olarak guncellenmisti, ama
# `run_metadata.json` YAZICISI guncellenmemisti -- uretilen provenance
# dosyalari KAPANMIS bir karari hala "acik soru / Barisin onayi
# bekleniyor" diye gosteriyordu. 8 artefakt JSON'u bu bayat metni
# tasiyordu (nihai model v3b dahil).
#
# Bu testler yazicinin metadata'ya KOYDUGU metni AST ile okur (kaynak
# dosyanin duz grep'ini DEGIL) -- boylece CLI yardimindaki bilerek
# saklanan "*(ESKI metin, tarihsel kayit -- GECERSIZ: ...)*" alintisi
# yanlis pozitif uretmez.
# =====================================================================

_STALE_OPEN_MARKERS = (
    "KILITLENMEDI",
    "KİLİTLENMEDİ",
    "onayi bekleniyor",
    "onayı bekleniyor",
)

# Eski metin PROVENANCE icin korunuyor, ama yalniz bu isaretten SONRA
# alintilanabilir -- isaretten ONCEKI kisim "guncel iddia"dir.
_HISTORICAL_QUOTE_MARKER = "(eski metin:"


def _writer_metadata_strings(keys: set[str]) -> list[str]:
    """`tools/train_cox_week3.py` kaynagindaki metadata sozluklerinde,
    verilen anahtarlarin ALTINA yazilan tum string literal'lerini AST ile
    toplar. Implicit string concatenation parser tarafindan zaten tek bir
    Constant'a katlanir, bu yuzden her liste elemani tek string gelir."""

    import ast

    tree = ast.parse((TOOLS_DIR / "train_cox_week3.py").read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and key.value in keys):
                continue
            for sub in ast.walk(value):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    found.append(sub.value)
    return found


def test_metadata_yazicisi_extra_column_penalizer_karari_KAPALI_gosterir() -> None:
    """A6 (2026-09-13) ile `extra_column_penalizer=0.0` KILITLENDI.
    Yazicinin urettigi hicbir metadata metni bunu 'acik/onay bekliyor'
    diye sunmamali. Bayat metin geri gelirse bu test KIRMIZI olur."""

    strings = _writer_metadata_strings({"known_open_questions", "clinical_covariate_note"})
    assert strings, "AST taramasi hicbir metadata string'i bulamadi -- test bozuk"

    ilgili = [s for s in strings if "extra_column_penalizer" in s]
    assert len(ilgili) == 3, (
        "extra_column_penalizer'dan bahseden metadata metni sayisi 3 bekleniyordu "
        f"(known_open_questions x2 + clinical_covariate_note x1), {len(ilgili)} bulundu. "
        "Yeni bir yazici sitesi eklendiyse bu testin de guncellenmesi gerekir."
    )

    for metin in ilgili:
        guncel_iddia = metin.split(_HISTORICAL_QUOTE_MARKER)[0]
        for marker in _STALE_OPEN_MARKERS:
            assert marker not in guncel_iddia, (
                f"BAYAT PROVENANCE METNI GERI GELDI: {marker!r} -- "
                "extra_column_penalizer=0.0 2026-09-13'te A6 karariyla "
                "KILITLENDI (decisions/2026-09-13-extra-column-penalizer-"
                f"sifir-onaylandi.md). Sorunlu metin: {metin!r}"
            )
        assert "KILITLENDI" in guncel_iddia, (
            f"Metin kararin KAPANDIGINI soylemiyor: {metin!r}"
        )
        assert "2026-09-13" in guncel_iddia, (
            f"Metin A6 karar tarihine atif yapmiyor: {metin!r}"
        )
        assert "2026-09-13-extra-column-penalizer-sifir-onaylandi" in metin, (
            f"Metin karar dosyasina atif yapmiyor (kaynaksiz iddia): {metin!r}"
        )


def test_yazilmis_run_metadata_jsonlari_extra_column_penalizeri_ACIK_gostermez() -> None:
    """Diskteki provenance JSON'lari (nihai model v3b dahil) ayni kurala
    uymali. Eski metin `(eski metin: ...)` alintisi icinde KORUNUYOR --
    silinmedi, yalniz 'acik soru' iddiasi kapatildi."""

    artifact_dir = PROJECT_ROOT / "artifacts" / "week3" / "cox_model"
    if not artifact_dir.is_dir():
        pytest.skip("artifacts/week3/cox_model yok")

    metadata_files = sorted(artifact_dir.rglob("*run_metadata.json"))
    assert metadata_files, "hic run_metadata.json bulunamadi"

    kontrol_edilen = 0
    for path in metadata_files:
        payload = json.loads(path.read_text(encoding="utf-8"))  # GECERLI JSON mi?
        for girdi in payload.get("known_open_questions", []):
            if "extra_column_penalizer" not in girdi:
                continue
            kontrol_edilen += 1
            guncel_iddia = girdi.split(_HISTORICAL_QUOTE_MARKER)[0]
            for marker in _STALE_OPEN_MARKERS:
                assert marker not in guncel_iddia, (
                    f"{path.name}: KAPANMIS karar hala 'acik' gosteriliyor -- {girdi!r}"
                )
            assert "KAPANDI 2026-09-13" in guncel_iddia, (
                f"{path.name}: kapanis isareti yok -- {girdi!r}"
            )
            # PROVENANCE: eski metin SILINMEMIS olmali.
            assert _HISTORICAL_QUOTE_MARKER in girdi, (
                f"{path.name}: eski metin SESSIZCE SILINMIS -- provenance kaybi: {girdi!r}"
            )

    assert kontrol_edilen == 8, (
        f"extra_column_penalizer girdisi tasiyan JSON sayisi 8 bekleniyordu, "
        f"{kontrol_edilen} bulundu."
    )
