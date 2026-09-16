from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.growth_simulation import (
    RANO_NON_RESPONSE_LABELS,
    RANO_RESPONSE_LABELS,
    SEGMENTATION_TOOL,
    TUMOR_REGION,
    UNKNOWN_RANO_LABEL,
    assign_patient_rano_group,
    classify_rano_label,
    compute_transition_growth_rates,
    fetch_lumiere_rano_labels,
    fetch_lumiere_wt_volume_series,
    fit_patient_growth_curves,
    gompertz,
    growth_fits_to_dataframe,
    growth_rate_by_rano_group,
    growth_rate_by_rano_group_and_model,
    log_ratio_growth_rate,
    logistic,
    normalize_rano_label,
    parse_timepoint_week,
    project_volume_range,
    simulate_boundary_growth,
    summarize_unknown_rano_labels,
    uninterpretable_group_names,
)


# --- parse_timepoint_week -------------------------------------------------


def test_parse_timepoint_week_no_suffix() -> None:
    assert parse_timepoint_week("week-044") == (44, 0)


def test_parse_timepoint_week_with_suffix() -> None:
    assert parse_timepoint_week("week-000-2") == (0, 2)


def test_parse_timepoint_week_zero_padding() -> None:
    assert parse_timepoint_week("week-000") == (0, 0)


def test_parse_timepoint_week_rejects_unknown_format() -> None:
    with pytest.raises(ValueError):
        parse_timepoint_week("month-3")


# --- RANO etiket normalizasyonu (Codex MEDIUM4) -----------------------------


def test_normalize_rano_label_strips_whitespace() -> None:
    assert normalize_rano_label("Post-Op ") == "Post-Op"
    assert normalize_rano_label(" PD") == "PD"


def test_normalize_rano_label_maps_none_variants_to_unknown() -> None:
    assert normalize_rano_label(None) == UNKNOWN_RANO_LABEL
    assert normalize_rano_label("None") == UNKNOWN_RANO_LABEL
    assert normalize_rano_label("") == UNKNOWN_RANO_LABEL
    assert normalize_rano_label("   ") == UNKNOWN_RANO_LABEL


def test_normalize_rano_label_preserves_compound_labels_as_is() -> None:
    """'Post-Op/PD' PD'ye SESSİZCE dönüştürülmemeli -- olduğu gibi kalmalı."""

    assert normalize_rano_label("Post-Op/PD") == "Post-Op/PD"


def test_classify_rano_label_categories() -> None:
    for label in RANO_RESPONSE_LABELS:
        assert classify_rano_label(label) == "response"
    for label in RANO_NON_RESPONSE_LABELS:
        assert classify_rano_label(label) == "non_response"
    assert classify_rano_label("Post-Op/PD") == "unknown"
    assert classify_rano_label(UNKNOWN_RANO_LABEL) == "unknown"


def test_summarize_unknown_rano_labels_flags_compound_and_missing() -> None:
    rano_df = pd.DataFrame(
        {
            "patient_id": ["Patient-A", "Patient-B", "Patient-C"],
            "visit_week": [10, 20, 30],
            "rano_label": ["PD", "Post-Op/PD", UNKNOWN_RANO_LABEL],
            "rano_label_raw": ["PD", "Post-Op/PD", "None"],
        }
    )
    unknown = summarize_unknown_rano_labels(rano_df)
    assert set(unknown["patient_id"]) == {"Patient-B", "Patient-C"}
    assert "Patient-A" not in unknown["patient_id"].values


# --- growth model shapes ---------------------------------------------------


def test_gompertz_monotonic_increasing_for_positive_r() -> None:
    t = np.linspace(0, 100, 20)
    v = gompertz(t, a=1000.0, b=2.0, r=0.05)
    assert np.all(np.diff(v) >= 0)
    assert v[-1] <= 1000.0 + 1e-6  # asimptota yaklaşır, aşmaz


def test_logistic_monotonic_increasing_for_positive_r() -> None:
    t = np.linspace(0, 100, 20)
    v = logistic(t, k=1000.0, r=0.1, t0=50.0)
    assert np.all(np.diff(v) >= 0)
    assert v[-1] <= 1000.0 + 1e-6


# --- fit_patient_growth_curves ---------------------------------------------


def _synthetic_series(patient_id: str, t: np.ndarray, v: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [patient_id] * len(t),
            "x_week": t,
            "tumor_volume_mm3": v,
        }
    )


def test_fit_patient_growth_curves_flags_insufficient_visits() -> None:
    series = _synthetic_series("Patient-X", np.array([0.0, 10.0]), np.array([100.0, 150.0]))
    fits = fit_patient_growth_curves(series, min_visits=3)
    assert len(fits) == 1
    assert fits[0].fit_status == "insufficient_visits"
    assert fits[0].n_visits == 2


def test_fit_patient_growth_curves_recovers_known_gompertz_r() -> None:
    rng = np.random.default_rng(42)
    t = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    true_r = 0.08
    v = gompertz(t, a=5000.0, b=3.0, r=true_r)
    v_noisy = v + rng.normal(0, v.max() * 0.01, size=v.shape)  # %1 gürültü

    series = _synthetic_series("Patient-Y", t, v_noisy)
    fits = fit_patient_growth_curves(series, min_visits=3)
    assert len(fits) == 1
    fit = fits[0]
    assert fit.fit_status == "ok"
    assert fit.gompertz.converged
    # Düşük gürültüde gerçek r'ye yakın fit bekleniyor (gevşek tolerans)
    assert fit.gompertz.r == pytest.approx(true_r, rel=0.5)
    assert fit.gompertz.r2 > 0.9


def test_fit_patient_growth_curves_handles_multiple_patients() -> None:
    t = np.array([0.0, 10.0, 20.0, 30.0])
    v1 = gompertz(t, a=3000.0, b=2.5, r=0.05)
    v2 = logistic(t, k=4000.0, r=0.1, t0=10.0)
    df = pd.concat(
        [
            _synthetic_series("Patient-A", t, v1),
            _synthetic_series("Patient-B", t, v2),
        ],
        ignore_index=True,
    )
    fits = fit_patient_growth_curves(df, min_visits=3)
    patient_ids = {f.patient_id for f in fits}
    assert patient_ids == {"Patient-A", "Patient-B"}


def test_growth_fits_to_dataframe_round_trip() -> None:
    t = np.array([0.0, 10.0, 20.0, 30.0])
    v = gompertz(t, a=3000.0, b=2.5, r=0.05)
    series = _synthetic_series("Patient-A", t, v)
    fits = fit_patient_growth_curves(series, min_visits=3)
    df = growth_fits_to_dataframe(fits)
    assert list(df["patient_id"]) == ["Patient-A"]
    assert df.loc[0, "fit_status"] == "ok"
    assert df.loc[0, "best_model"] in ("gompertz", "logistic")


# --- RANO group assignment + distribution -----------------------------------


def test_assign_patient_rano_group_uses_last_response_label() -> None:
    rano_df = pd.DataFrame(
        {
            "patient_id": ["Patient-A", "Patient-A", "Patient-A", "Patient-B"],
            "visit_week": [0, 20, 40, 0],
            "rano_label": ["Pre-Op", "SD", "PD", "Pre-Op"],
            "rano_rationale": [None, None, None, None],
        }
    )
    result = assign_patient_rano_group(rano_df)
    row_a = result[result["patient_id"] == "Patient-A"].iloc[0]
    assert row_a["rano_group"] == "PD"
    # Patient-B'nin hiç PD/SD/PR/CR etiketi yok -> gruba girmemeli
    assert "Patient-B" not in result["patient_id"].values


def test_growth_rate_by_rano_group_flags_small_n() -> None:
    fits_df = pd.DataFrame(
        {
            "patient_id": [f"Patient-{i}" for i in range(6)],
            "fit_status": ["ok"] * 6,
            "best_r": [0.05, 0.06, 0.07, 0.20, 0.21, 0.01],
            "best_model": ["gompertz"] * 6,
        }
    )
    rano_group_df = pd.DataFrame(
        {
            "patient_id": [f"Patient-{i}" for i in range(6)],
            "rano_group": ["SD", "SD", "SD", "PD", "PD", "CR"],
        }
    )
    result = growth_rate_by_rano_group(fits_df, rano_group_df, min_group_n=3)
    sd_row = result[result["rano_group"] == "SD"].iloc[0]
    pd_row = result[result["rano_group"] == "PD"].iloc[0]
    cr_row = result[result["rano_group"] == "CR"].iloc[0]
    assert sd_row["interpretable"]
    assert not pd_row["interpretable"]  # n=2 < min_group_n=3
    assert not cr_row["interpretable"]  # n=1
    assert sd_row["median_r"] == pytest.approx(0.06)


def test_growth_rate_by_rano_group_default_min_n_is_six_not_five() -> None:
    """2026-09-11 Barış kararı: eski varsayılan (n>=5) n>=6'ya yükseltildi.

    SD (n=5) gerçek verideki gibi burada da dar bir dağılımla (saçılım
    koşulunu TETİKLEMEYECEK şekilde) kuruluyor -- bu, `False` sonucunun
    YALNIZ n<6 kuralından geldiğini, IQR koşuluyla karışmadığını izole eder.
    """

    patients = [f"Patient-{i}" for i in range(5)]
    r_values = [0.050, 0.051, 0.052, 0.053, 0.054]  # dar IQR

    fits_df = pd.DataFrame(
        {
            "patient_id": patients,
            "fit_status": ["ok"] * 5,
            "best_r": r_values,
            "best_model": ["gompertz"] * 5,
        }
    )
    rano_group_df = pd.DataFrame({"patient_id": patients, "rano_group": ["SD"] * 5})

    result = growth_rate_by_rano_group(fits_df, rano_group_df)  # varsayılan eşikler
    sd_row = result[result["rano_group"] == "SD"].iloc[0]
    assert sd_row["n"] == 5
    assert not sd_row["interpretable"]  # yalnız n<6 nedeniyle -- dar IQR'a rağmen


def test_growth_rate_by_rano_group_marks_sd_false_pd_true_like_v2_finding() -> None:
    """2026-09-11 Barış kararı -- ana regresyon testi.

    `growth_rate_by_rano_group_v2.csv`'deki gerçek bulguyu taklit eder: SD
    (n=5, medyan~-1,05, geniş IQR) `interpretable=False`, PD-benzeri büyük/
    dar-dağılımlı bir grup (n=67) `interpretable=True` olmalı.
    """

    sd_patients = [f"SD-Patient-{i}" for i in range(5)]
    sd_r = np.array([-2.10, -1.80, -1.05, -0.40, -0.05])  # medyan~-1.05, geniş IQR

    n_pd = 67
    pd_patients = [f"PD-Patient-{i}" for i in range(n_pd)]
    pd_r = np.linspace(0.02, 0.08, n_pd)  # dar bant (küçük IQR)

    fits_df = pd.DataFrame(
        {
            "patient_id": sd_patients + pd_patients,
            "fit_status": ["ok"] * (len(sd_patients) + len(pd_patients)),
            "best_r": np.concatenate([sd_r, pd_r]),
            "best_model": ["gompertz"] * (len(sd_patients) + len(pd_patients)),
        }
    )
    rano_group_df = pd.DataFrame(
        {
            "patient_id": sd_patients + pd_patients,
            "rano_group": ["SD"] * len(sd_patients) + ["PD"] * len(pd_patients),
        }
    )

    result = growth_rate_by_rano_group(fits_df, rano_group_df)  # varsayılan eşikler
    sd_row = result[result["rano_group"] == "SD"].iloc[0]
    pd_row = result[result["rano_group"] == "PD"].iloc[0]

    assert sd_row["n"] == 5
    assert pd_row["n"] == 67
    # Gerçek veride SD/PD IQR oranı ~18x idi -- burada da referansın (PD,
    # en büyük n) IQR'ının çok üstünde olmalı.
    assert sd_row["iqr_width"] > 5.0 * pd_row["iqr_width"]
    assert not sd_row["interpretable"]
    assert pd_row["interpretable"]


def test_growth_rate_by_rano_group_dispersion_condition_works_independently_of_n() -> None:
    """(b) saçılım koşulu (a) n-eşiğinden BAĞIMSIZ çalışmalı: n>=6 olsa bile
    IQR referansın `max_iqr_ratio_to_reference` katını aşarsa `False` olmalı."""

    wide_patients = [f"Wide-Patient-{i}" for i in range(6)]
    wide_r = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])  # n=6 (eşiği GEÇER), çok dağınık

    n_ref = 67
    ref_patients = [f"Ref-Patient-{i}" for i in range(n_ref)]
    ref_r = np.linspace(0.02, 0.08, n_ref)

    fits_df = pd.DataFrame(
        {
            "patient_id": wide_patients + ref_patients,
            "fit_status": ["ok"] * (len(wide_patients) + len(ref_patients)),
            "best_r": np.concatenate([wide_r, ref_r]),
            "best_model": ["gompertz"] * (len(wide_patients) + len(ref_patients)),
        }
    )
    rano_group_df = pd.DataFrame(
        {
            "patient_id": wide_patients + ref_patients,
            "rano_group": ["WIDE"] * len(wide_patients) + ["REF"] * len(ref_patients),
        }
    )

    result = growth_rate_by_rano_group(fits_df, rano_group_df)
    wide_row = result[result["rano_group"] == "WIDE"].iloc[0]
    ref_row = result[result["rano_group"] == "REF"].iloc[0]

    assert wide_row["n"] == 6  # min_group_n eşiğini (a) GEÇER
    assert not wide_row["interpretable"]  # ama (b) saçılım koşulunu İHLAL EDER
    assert ref_row["interpretable"]


def test_growth_rate_by_rano_group_max_iqr_ratio_none_disables_dispersion_check() -> None:
    """`max_iqr_ratio_to_reference=None` verilirse (b) kapanır, yalnız n uygulanır."""

    wide_patients = [f"Wide-Patient-{i}" for i in range(6)]
    wide_r = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])

    n_ref = 67
    ref_patients = [f"Ref-Patient-{i}" for i in range(n_ref)]
    ref_r = np.linspace(0.02, 0.08, n_ref)

    fits_df = pd.DataFrame(
        {
            "patient_id": wide_patients + ref_patients,
            "fit_status": ["ok"] * (len(wide_patients) + len(ref_patients)),
            "best_r": np.concatenate([wide_r, ref_r]),
            "best_model": ["gompertz"] * (len(wide_patients) + len(ref_patients)),
        }
    )
    rano_group_df = pd.DataFrame(
        {
            "patient_id": wide_patients + ref_patients,
            "rano_group": ["WIDE"] * len(wide_patients) + ["REF"] * len(ref_patients),
        }
    )

    result = growth_rate_by_rano_group(
        fits_df, rano_group_df, max_iqr_ratio_to_reference=None
    )
    wide_row = result[result["rano_group"] == "WIDE"].iloc[0]
    assert wide_row["n"] == 6
    assert wide_row["interpretable"]  # (b) kapalı -- yalnız n>=6 kontrol edildi


def test_growth_rate_by_rano_group_and_model_applies_same_sd_pd_rule() -> None:
    """`growth_rate_by_rano_group_and_model`, aynı iki-koşullu kuralı
    (rano_group, best_model) çiftleri arasında da uygular."""

    sd_patients = [f"SD-Patient-{i}" for i in range(5)]
    sd_r = np.array([-2.10, -1.80, -1.05, -0.40, -0.05])

    n_pd = 67
    pd_patients = [f"PD-Patient-{i}" for i in range(n_pd)]
    pd_r = np.linspace(0.02, 0.08, n_pd)

    fits_df = pd.DataFrame(
        {
            "patient_id": sd_patients + pd_patients,
            "fit_status": ["ok"] * (len(sd_patients) + len(pd_patients)),
            "best_r": np.concatenate([sd_r, pd_r]),
            "best_model": ["gompertz"] * (len(sd_patients) + len(pd_patients)),
        }
    )
    rano_group_df = pd.DataFrame(
        {
            "patient_id": sd_patients + pd_patients,
            "rano_group": ["SD"] * len(sd_patients) + ["PD"] * len(pd_patients),
        }
    )

    result = growth_rate_by_rano_group_and_model(fits_df, rano_group_df)
    sd_row = result[(result["rano_group"] == "SD") & (result["best_model"] == "gompertz")].iloc[0]
    pd_row = result[(result["rano_group"] == "PD") & (result["best_model"] == "gompertz")].iloc[0]
    assert not sd_row["interpretable"]
    assert pd_row["interpretable"]


def test_uninterpretable_group_names_returns_flagged_groups_only() -> None:
    dist_df = pd.DataFrame(
        {
            "rano_group": ["PD", "PR", "CR"],
            "n": [61, 3, 1],
            "median_r": [0.01, 0.02, 0.03],
            "interpretable": [True, False, False],
        }
    )
    assert uninterpretable_group_names(dist_df) == ["PR", "CR"]


def test_uninterpretable_group_names_handles_empty_and_missing_column() -> None:
    assert uninterpretable_group_names(pd.DataFrame()) == []


def test_growth_rate_by_rano_group_and_model_stratifies_by_best_model() -> None:
    """Codex LOW: logistic/gompertz r'leri tek havuzda değil, model-tipine
    göre AYRI satırlar olarak raporlanmalı."""

    fits_df = pd.DataFrame(
        {
            "patient_id": [f"Patient-{i}" for i in range(6)],
            "fit_status": ["ok"] * 6,
            "best_r": [0.05, 0.06, 0.07, 0.20, 0.21, 0.01],
            "best_model": ["gompertz", "gompertz", "logistic", "gompertz", "logistic", "logistic"],
        }
    )
    rano_group_df = pd.DataFrame(
        {
            "patient_id": [f"Patient-{i}" for i in range(6)],
            "rano_group": ["SD", "SD", "SD", "PD", "PD", "CR"],
        }
    )
    result = growth_rate_by_rano_group_and_model(fits_df, rano_group_df, min_group_n=2)
    sd_gompertz = result[(result["rano_group"] == "SD") & (result["best_model"] == "gompertz")]
    sd_logistic = result[(result["rano_group"] == "SD") & (result["best_model"] == "logistic")]
    assert len(sd_gompertz) == 1 and sd_gompertz.iloc[0]["n"] == 2
    assert len(sd_logistic) == 1 and sd_logistic.iloc[0]["n"] == 1
    assert not sd_logistic.iloc[0]["interpretable"]  # n=1 < min_group_n=2


# --- geçiş-bazlı yerel büyüme hızı (Codex HIGH1) -----------------------------


def test_log_ratio_growth_rate_matches_manual_formula() -> None:
    rate = log_ratio_growth_rate(v_prev=1000.0, v_curr=1300.0, t_prev=0.0, t_curr=20.0)
    assert rate == pytest.approx(np.log(1.3) / 20.0)


def test_log_ratio_growth_rate_nan_on_non_positive_volume() -> None:
    assert np.isnan(log_ratio_growth_rate(0.0, 100.0, 0.0, 10.0))
    assert np.isnan(log_ratio_growth_rate(100.0, 0.0, 0.0, 10.0))


def test_log_ratio_growth_rate_nan_on_non_positive_time_delta() -> None:
    assert np.isnan(log_ratio_growth_rate(100.0, 200.0, 10.0, 10.0))
    assert np.isnan(log_ratio_growth_rate(100.0, 200.0, 10.0, 5.0))


def test_compute_transition_growth_rates_produces_one_row_per_consecutive_pair() -> None:
    series = pd.DataFrame(
        {
            "patient_id": ["Patient-A"] * 3 + ["Patient-B"] * 2,
            "scan_id": [1, 2, 3, 10, 11],
            "x_week": [0.0, 10.0, 20.0, 0.0, 5.0],
            "tumor_volume_mm3": [1000.0, 1100.0, 1210.0, 500.0, 250.0],
        }
    )
    transitions = compute_transition_growth_rates(series)
    assert len(transitions) == 3  # Patient-A: 2 geçiş, Patient-B: 1 geçiş
    a_first = transitions[
        (transitions["patient_id"] == "Patient-A") & (transitions["curr_week"] == 10.0)
    ].iloc[0]
    assert a_first["prev_volume_mm3"] == pytest.approx(1000.0)
    assert a_first["curr_volume_mm3"] == pytest.approx(1100.0)
    assert a_first["log_ratio_rate_per_week"] == pytest.approx(np.log(1.1) / 10.0)

    b_row = transitions[transitions["patient_id"] == "Patient-B"].iloc[0]
    # Patient-B küçülüyor (500 -> 250) -- log-oranı NEGATİF olmalı
    assert b_row["log_ratio_rate_per_week"] < 0


def test_compute_transition_growth_rates_handles_empty_series() -> None:
    empty = pd.DataFrame(columns=["patient_id", "scan_id", "x_week", "tumor_volume_mm3"])
    result = compute_transition_growth_rates(empty)
    assert result.empty
    assert "log_ratio_rate_per_week" in result.columns


# --- boundary-based morphological simulation --------------------------------


def _make_sphere_mask(shape: tuple[int, int, int], center, radius: float) -> np.ndarray:
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    dist = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    return dist <= radius


def test_simulate_boundary_growth_reaches_target_volume_within_tolerance() -> None:
    mask = _make_sphere_mask((40, 40, 40), (20, 20, 20), radius=8.0)
    spacing = (1.0, 1.0, 1.0)
    v0 = float(mask.sum())
    target = v0 * 1.5

    # Rank-bazlı seçim EXACT'tır (±1 voksel) -- bkz. fonksiyon docstring'i
    # ("neden bisection değil" bölümü).
    new_mask, achieved = simulate_boundary_growth(mask, spacing, target)
    assert achieved == pytest.approx(target, abs=1.0 * spacing[0] * spacing[1] * spacing[2])


def test_simulate_boundary_growth_grow_case_is_superset_of_original() -> None:
    """Sınır-temelli büyüme: yeni maske ESKİ maskeyi TAMAMEN içermeli.

    Bu, uniform scaling'den (merkezden ölçekleme) ayırt edici bir özelliktir
    -- merkezden ölçeklemede voksel konumları KAYAR, eski maske yeni
    maskenin alt kümesi OLMAK ZORUNDA DEĞİLDİR. Mesafe-dönüşümlü sınır
    genişlemesinde ise garanti EDİLİR.
    """

    mask = _make_sphere_mask((40, 40, 40), (20, 20, 20), radius=6.0)
    spacing = (1.0, 1.0, 1.0)
    v0 = float(mask.sum())
    target = v0 * 2.0

    new_mask, _ = simulate_boundary_growth(mask, spacing, target)
    assert np.all(new_mask[mask])  # eski maskenin TÜM voksel'leri yeni maskede de var


def test_simulate_boundary_growth_shrink_case_is_subset_of_original() -> None:
    mask = _make_sphere_mask((40, 40, 40), (20, 20, 20), radius=8.0)
    spacing = (1.0, 1.0, 1.0)
    v0 = float(mask.sum())
    target = v0 * 0.5
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    new_mask, achieved = simulate_boundary_growth(mask, spacing, target)
    assert achieved == pytest.approx(target, abs=1.0 * voxel_volume)
    assert np.all(mask[new_mask])  # yeni maskenin TÜM voksel'leri eski maskede de var


def test_simulate_boundary_growth_respects_anisotropic_spacing() -> None:
    mask = _make_sphere_mask((30, 30, 30), (15, 15, 15), radius=5.0)
    spacing = (1.0, 1.0, 3.0)  # anizotropik (LUMIERE'de gerçekçi bir durum)
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    v0 = float(mask.sum()) * voxel_volume
    target = v0 * 1.3

    new_mask, achieved = simulate_boundary_growth(mask, spacing, target)
    assert achieved == pytest.approx(target, abs=1.0 * voxel_volume)


def test_simulate_boundary_growth_rejects_empty_mask() -> None:
    mask = np.zeros((10, 10, 10), dtype=bool)
    with pytest.raises(ValueError):
        simulate_boundary_growth(mask, (1.0, 1.0, 1.0), 100.0)


# --- volume projection (IQR belirsizlik aralığı) ----------------------------


def test_project_volume_range_orders_low_median_high() -> None:
    projection = project_volume_range(
        v0=1000.0,
        model="gompertz",
        template_params=(5000.0, 2.5, 0.06),
        r_low=0.03,
        r_median=0.06,
        r_high=0.10,
        months=6.0,
    )
    assert projection.v_low <= projection.v_median <= projection.v_high
    assert projection.pct_low <= projection.pct_median <= projection.pct_high


def test_project_volume_range_default_from_week_preserves_legacy_horizon() -> None:
    """`from_week` VARSAYILANI (0.0) eski davranisi BIT-BIREBIR korur.

    Bu geriye-donuk uyumluluk testi `tools/run_growth_simulation_lumiere.py`
    icin gereklidir: o script `project_volume_range()`'i SENTETIK bir kure
    maskesi + sentetik sablon parametreleriyle cagirir, orada bir zaman
    serisi YOKTUR ve tek tanimli baslangic t=0'dir -- cikitisi DEGISMEMELI.
    """

    from pipeline.growth_simulation import _WEEKS_PER_MONTH

    projection = project_volume_range(
        v0=7153.0,
        model="gompertz",
        template_params=(7153.0 * 4.0, 2.5, 0.06),
        r_low=0.03,
        r_median=0.06,
        r_high=0.10,
        months=6.0,
    )
    assert projection.from_week == 0.0
    assert projection.t_target_week == pytest.approx(6.0 * _WEEKS_PER_MONTH)
    assert projection.t_target_week == pytest.approx(26.07)
    # 2026-09-13 duzeltmesinden ONCE olculen sentetik demo degerleri
    # (scratchpad/synthetic_unchanged.py, max|fark| = 0.0):
    assert projection.v_low == pytest.approx(9117.679111824129, rel=1e-12)
    assert projection.v_median == pytest.approx(16957.120497659787, rel=1e-12)
    assert projection.v_high == pytest.approx(23794.09673742653, rel=1e-12)


def test_project_volume_range_horizon_starts_at_from_week() -> None:
    """G1 (2026-09-13) -- ufuk `from_week`'ten ITIBAREN ileriye bakar.

    Eski davranista `t_target` kosulsuz `months*4,345` idi; `from_week`
    verilse bile yoksayilirdi -> asagidaki assert'ler KIRMIZI olurdu.
    """

    from pipeline.growth_simulation import _WEEKS_PER_MONTH

    kwargs = dict(
        v0=1000.0,
        model="gompertz",
        template_params=(5000.0, 2.5, 0.06),
        r_low=0.03,
        r_median=0.06,
        r_high=0.10,
        months=6.0,
    )
    at_zero = project_volume_range(**kwargs)
    at_38 = project_volume_range(**kwargs, from_week=38.0)

    assert at_38.from_week == 38.0
    assert at_38.t_target_week == pytest.approx(38.0 + 6.0 * _WEEKS_PER_MONTH)
    assert at_38.t_target_week == pytest.approx(64.07)
    # Ufuk baslangictan SONRA olmali (geriye donuk karsilastirma YASAK).
    assert at_38.t_target_week > at_38.from_week
    # Iki cagri ayni olamaz -- from_week gercekten etkili olmali.
    assert at_38.t_target_week != at_zero.t_target_week
    assert at_38.v_median > at_zero.v_median  # buyuyen egri, daha ileri t


def test_project_volume_range_patient_028_lower_bound_sign_g1() -> None:
    """G1'in klinik etkisi -- `Patient-028`'in GERCEK fit'i ile alt ucun
    ISARET DEGISTIRDIGININ saf-matematik regresyon testi.

    Girdiler canli DB'den olculdu (2026-09-13, readonly): ziyaretler
    0 / 0,05 / 15 / 20 / 21 / 25 / 30 / 35 / **38**; gompertz
    a=173999,65996683933 b=0,07209881404086578; v0 (hafta 38) =
    112.576 mm3; PD grubu r ucluvsu (-0,02823603947157014,
    0,0002388933400997105, 0,02737208714124538).

    ESKI (hatali, t=26,07): %+32,96 … %+43,87 … %+49,20  -> alt uc POZITIF
    DOGRU (t=38+26,07=64,07): %-0,47 … %+43,97 … %+52,64 -> alt uc NEGATIF
    Eski davranis geri gelirse `pct_low > 0` olur ve test KIRMIZI olur.
    """

    common = dict(
        v0=112576.0,
        model="gompertz",
        template_params=(173999.65996683933, 0.07209881404086578, -0.051261050902068124),
        r_low=-0.02823603947157014,
        r_median=0.0002388933400997105,
        r_high=0.02737208714124538,
        months=6.0,
    )

    legacy = project_volume_range(**common)  # from_week varsayilani 0.0
    assert legacy.t_target_week == pytest.approx(26.07)
    assert legacy.pct_low == pytest.approx(32.962439179702656, rel=1e-9)
    assert legacy.pct_median == pytest.approx(43.874846512828306, rel=1e-9)
    assert legacy.pct_high == pytest.approx(49.19811531712419, rel=1e-9)
    assert legacy.pct_low > 0  # eski hali: "kesin buyume" mesaji

    corrected = project_volume_range(**common, from_week=38.0)
    assert corrected.t_target_week == pytest.approx(64.07)
    assert corrected.pct_low == pytest.approx(-0.4716277361509173, rel=1e-9)
    assert corrected.pct_median == pytest.approx(43.96803595681777, rel=1e-9)
    assert corrected.pct_high == pytest.approx(52.64465863852963, rel=1e-9)
    # ASIL BULGU: alt uc ISARET DEGISTIRIYOR.
    assert corrected.pct_low < 0


def test_project_volume_range_rejects_unknown_model() -> None:
    with pytest.raises(ValueError):
        project_volume_range(
            v0=1000.0,
            model="exponential",
            template_params=(1.0, 1.0, 1.0),
            r_low=0.01,
            r_median=0.02,
            r_high=0.03,
        )


# --- Canlı DB (readonly) smoke testleri -------------------------------------


@pytest.fixture()
def db_conn():
    from db_connection import get_connection

    conn = get_connection(readonly=True)
    try:
        yield conn
    finally:
        conn.close()


def test_fetch_lumiere_wt_volume_series_live_db_uses_c32_only(db_conn) -> None:
    df = fetch_lumiere_wt_volume_series(db_conn)
    assert not df.empty
    # Sözleşme: yalnız WT_derived + C32 (fonksiyonun SQL'i zaten sabitliyor,
    # burada dönen veri üzerinden dolaylı doğrulama)
    assert SEGMENTATION_TOOL == "LUMIERE-PyRadiomics-107-C32"
    assert TUMOR_REGION == "WT_derived"
    assert df["patient_id"].str.startswith("Patient-").all()
    # 5/585 satır hacim=0 (kanıtlı, gerçek veri -- örn. tam yanıt/etiket
    # yokluğu, bkz. bu görevin final raporu) -- bu yüzden >=0, >0 DEĞİL.
    assert (df["tumor_volume_mm3"] >= 0).all()


def test_fetch_lumiere_rano_labels_live_db_matches_known_distribution(db_conn) -> None:
    df = fetch_lumiere_rano_labels(db_conn)
    assert len(df) == 616  # 2026-08-18'de canlı ölçülen sayı
    response_only = df[df["rano_label"].isin(RANO_RESPONSE_LABELS)]
    assert len(response_only) > 0
    assert df["followup_id"].notna().all()  # kaynak satır kimliği (MEDIUM5)


def test_fetch_lumiere_rano_labels_live_db_normalizes_known_dirty_values(db_conn) -> None:
    """Codex MEDIUM4: canlı DB'de bilinen 6 kirli satır (2026-08-19 doğrulandı:
    3x 'Post-Op ', 1x 'Post-Op/PD', 2x literal 'None') normalize edilmeli."""

    df = fetch_lumiere_rano_labels(db_conn)
    # Hiçbir normalize edilmiş etikette baştaki/sondaki boşluk KALMAMALI.
    assert (df["rano_label"] == df["rano_label"].str.strip()).all()
    # Ham sütun HALA kirli değerleri taşımalı (denetim izi korunmuş).
    assert (df["rano_label_raw"] == "Post-Op ").sum() == 3
    # 'Post-Op/PD' PD'ye SESSİZCE dönüştürülmemiş, olduğu gibi kalmış olmalı.
    compound = df[df["patient_id"] == "Patient-089"]
    assert (compound["rano_label"] == "Post-Op/PD").any()
    # Literal 'None' metni UNKNOWN'a normalize edilmiş olmalı.
    assert (df["rano_label"] == UNKNOWN_RANO_LABEL).sum() >= 2


def test_summarize_unknown_rano_labels_live_db_surfaces_known_rows(db_conn) -> None:
    df = fetch_lumiere_rano_labels(db_conn)
    unknown = summarize_unknown_rano_labels(df)
    # En az Patient-026 (literal 'None'), Patient-083 (literal 'None'),
    # Patient-089 ('Post-Op/PD') unknown kovasında olmalı.
    assert {"Patient-026", "Patient-089"} <= set(unknown["patient_id"])
    assert (unknown["rano_category"] == "unknown").all()
