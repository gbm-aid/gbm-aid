from __future__ import annotations

import pandas as pd
import pytest

from pipeline.followup_t3 import (
    _visit_match_report_from_frames,
    build_delta_rano_table,
    build_patient_visit_deltas,
    build_visit_match_report,
    compute_delta_metrics,
    pair_lumiere_visits_with_rano,
    summarize_excluded_transitions,
)


# --- compute_delta_metrics --------------------------------------------------


def _row(patient_id, x_week, volume, surface, sphericity, entropy, rano_label):
    return pd.Series(
        {
            "patient_id": patient_id,
            "x_week": x_week,
            "tumor_volume_mm3": volume,
            "surface_area": surface,
            "sphericity": sphericity,
            "glcm_joint_entropy": entropy,
            "rano_label": rano_label,
        }
    )


def test_compute_delta_metrics_percentage_math() -> None:
    prev = _row("Patient-001", 0, 1000.0, 500.0, 0.6, 2.0, "Post-Op")
    curr = _row("Patient-001", 20, 1300.0, 600.0, 0.5, 2.5, "PD")

    delta = compute_delta_metrics(prev, curr)

    assert delta.volume_pct == pytest.approx(30.0)
    assert delta.surface_pct == pytest.approx(20.0)
    assert delta.sphericity_delta_pct_points == pytest.approx(-10.0)
    assert delta.glcm_entropy_pct == pytest.approx(25.0)
    assert delta.curr_rano_label == "PD"
    # Codex HIGH1: geçiş-bazlı yerel büyüme hızı -- log(1300/1000)/20
    assert delta.log_ratio_rate_per_week == pytest.approx(0.013118, abs=1e-5)


def test_compute_delta_metrics_handles_zero_previous_value() -> None:
    prev = _row("Patient-001", 0, 0.0, 500.0, 0.6, 2.0, "Post-Op")
    curr = _row("Patient-001", 20, 100.0, 600.0, 0.5, 2.5, "PD")
    delta = compute_delta_metrics(prev, curr)
    assert pd.isna(delta.volume_pct)
    assert pd.isna(delta.log_ratio_rate_per_week)  # v_prev<=0 -> nan (Codex HIGH1)


# --- build_patient_visit_deltas ---------------------------------------------


def test_build_patient_visit_deltas_excludes_preop_postop_transitions() -> None:
    paired = pd.DataFrame(
        [
            {
                "patient_id": "Patient-001",
                "x_week": 0,
                "tumor_volume_mm3": 5000.0,
                "surface_area": 1000.0,
                "sphericity": 0.6,
                "glcm_joint_entropy": 3.0,
                "rano_label": "Pre-Op",
            },
            {
                "patient_id": "Patient-001",
                "x_week": 1,
                "tumor_volume_mm3": 500.0,
                "surface_area": 200.0,
                "sphericity": 0.5,
                "glcm_joint_entropy": 2.0,
                "rano_label": "Post-Op",
            },
            {
                "patient_id": "Patient-001",
                "x_week": 20,
                "tumor_volume_mm3": 700.0,
                "surface_area": 250.0,
                "sphericity": 0.55,
                "glcm_joint_entropy": 2.2,
                "rano_label": "SD",
            },
            {
                "patient_id": "Patient-001",
                "x_week": 40,
                "tumor_volume_mm3": 1400.0,
                "surface_area": 400.0,
                "sphericity": 0.4,
                "glcm_joint_entropy": 3.0,
                "rano_label": "PD",
            },
        ]
    )

    deltas = build_patient_visit_deltas(paired)

    # Pre-Op->Post-Op (rezeksiyon etkisi, RANO yanıtı DEĞİL) dahil EDİLMEMELİ
    assert "Pre-Op" not in deltas["curr_rano_label"].values
    assert "Post-Op" not in deltas["curr_rano_label"].values
    # Post-Op->SD ve SD->PD geçişleri KALMALI
    assert set(deltas["curr_rano_label"]) == {"SD", "PD"}
    assert len(deltas) == 2


def test_build_patient_visit_deltas_handles_multiple_patients_independently() -> None:
    paired = pd.DataFrame(
        [
            {
                "patient_id": "Patient-001",
                "x_week": 0,
                "tumor_volume_mm3": 1000.0,
                "surface_area": 200.0,
                "sphericity": 0.6,
                "glcm_joint_entropy": 2.0,
                "rano_label": "Post-Op",
            },
            {
                "patient_id": "Patient-001",
                "x_week": 20,
                "tumor_volume_mm3": 1100.0,
                "surface_area": 210.0,
                "sphericity": 0.58,
                "glcm_joint_entropy": 2.1,
                "rano_label": "SD",
            },
            {
                "patient_id": "Patient-002",
                "x_week": 0,
                "tumor_volume_mm3": 2000.0,
                "surface_area": 400.0,
                "sphericity": 0.5,
                "glcm_joint_entropy": 2.5,
                "rano_label": "Post-Op",
            },
            {
                "patient_id": "Patient-002",
                "x_week": 15,
                "tumor_volume_mm3": 500.0,
                "surface_area": 150.0,
                "sphericity": 0.7,
                "glcm_joint_entropy": 1.5,
                "rano_label": "CR",
            },
        ]
    )
    deltas = build_patient_visit_deltas(paired)
    assert set(deltas["patient_id"]) == {"Patient-001", "Patient-002"}
    assert len(deltas) == 2


# --- build_delta_rano_table --------------------------------------------------


def test_build_delta_rano_table_computes_median_iqr_per_group() -> None:
    deltas = pd.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(8)],
            "curr_rano_label": ["PD"] * 5 + ["CR"] * 3,
            "volume_pct": [30, 40, 25, 35, 45, -70, -80, -65],
            "surface_pct": [10, 15, 12, 14, 16, -30, -35, -32],
            "sphericity_delta_pct_points": [-5, -6, -4, -5, -7, 2, 3, 1],
            "glcm_entropy_pct": [26, 27, 28, 25, 29, -5, -6, -4],
            "log_ratio_rate_per_week": [0.01, 0.02, 0.015, 0.018, 0.022, -0.05, -0.06, -0.045],
        }
    )
    table = build_delta_rano_table(deltas, min_group_n=5)

    pd_row = table[table["rano_label"] == "PD"].iloc[0]
    cr_row = table[table["rano_label"] == "CR"].iloc[0]

    assert pd_row["n"] == 5
    assert pd_row["interpretable"]
    assert pd_row["volume_pct_median"] == pytest.approx(35.0)

    assert cr_row["n"] == 3
    assert not cr_row["interpretable"]  # n=3 < min_group_n=5


def test_build_delta_rano_table_orders_by_canonical_rano_sequence() -> None:
    deltas = pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3"],
            "curr_rano_label": ["CR", "PD", "SD"],
            "volume_pct": [-70, 30, 5],
            "surface_pct": [-30, 10, 2],
            "sphericity_delta_pct_points": [2, -5, 0],
            "glcm_entropy_pct": [-5, 26, 1],
            "log_ratio_rate_per_week": [-0.05, 0.02, 0.001],
        }
    )
    table = build_delta_rano_table(deltas, min_group_n=1)
    assert list(table["rano_label"]) == ["PD", "SD", "CR"]


def test_build_delta_rano_table_includes_log_ratio_rate_metric() -> None:
    """Codex HIGH1 düzeltmesi: transition-bazlı yerel büyüme hızı metriği
    `build_delta_rano_table`'ın döndürdüğü metrik kümesine dahil olmalı."""

    deltas = pd.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(5)],
            "curr_rano_label": ["PD"] * 5,
            "volume_pct": [30, 40, 25, 35, 45],
            "surface_pct": [10, 15, 12, 14, 16],
            "sphericity_delta_pct_points": [-5, -6, -4, -5, -7],
            "glcm_entropy_pct": [26, 27, 28, 25, 29],
            "log_ratio_rate_per_week": [0.01, 0.02, 0.015, 0.018, 0.022],
        }
    )
    table = build_delta_rano_table(deltas, min_group_n=5)
    assert "log_ratio_rate_per_week_median" in table.columns
    assert table.loc[0, "log_ratio_rate_per_week_median"] == pytest.approx(0.018)


# --- Canlı DB (readonly) smoke testi -----------------------------------------


@pytest.fixture()
def db_conn():
    from db_connection import get_connection

    conn = get_connection(readonly=True)
    try:
        yield conn
    finally:
        conn.close()


def test_pair_lumiere_visits_with_rano_live_db_produces_matches(db_conn) -> None:
    paired = pair_lumiere_visits_with_rano(db_conn)
    assert not paired.empty
    assert set(paired["match_type"].unique()) <= {"unique", "heuristic_paired"}
    assert (paired["tumor_volume_mm3"] >= 0).all()


# --- summarize_excluded_transitions (Codex MEDIUM4) -------------------------


def test_summarize_excluded_transitions_flags_surgical_and_unknown_labels() -> None:
    paired = pd.DataFrame(
        [
            {
                "patient_id": "Patient-001",
                "x_week": 0,
                "tumor_volume_mm3": 5000.0,
                "surface_area": 1000.0,
                "sphericity": 0.6,
                "glcm_joint_entropy": 3.0,
                "rano_label": "Pre-Op",
            },
            {
                "patient_id": "Patient-001",
                "x_week": 1,
                "tumor_volume_mm3": 500.0,
                "surface_area": 200.0,
                "sphericity": 0.5,
                "glcm_joint_entropy": 2.0,
                "rano_label": "Post-Op",
            },
            {
                "patient_id": "Patient-001",
                "x_week": 46,
                "tumor_volume_mm3": 700.0,
                "surface_area": 250.0,
                "sphericity": 0.55,
                "glcm_joint_entropy": 2.2,
                "rano_label": "Post-Op/PD",
            },
            {
                "patient_id": "Patient-001",
                "x_week": 60,
                "tumor_volume_mm3": 900.0,
                "surface_area": 260.0,
                "sphericity": 0.5,
                "glcm_joint_entropy": 2.4,
                "rano_label": "PD",
            },
        ]
    )
    excluded = summarize_excluded_transitions(paired)
    reasons = dict(zip(excluded["curr_rano_label"], excluded["exclusion_reason"]))
    assert reasons["Post-Op"] == "surgical_timepoint_not_rano_response"
    assert reasons["Post-Op/PD"] == "unknown_or_compound_label"
    # PD (yanıt kategorisi) HİÇ dışlanmamalı -- delta hesabına girer.
    assert "PD" not in excluded["curr_rano_label"].values


# --- build_visit_match_report / _visit_match_report_from_frames (MEDIUM5) ---


def test_visit_match_report_flags_unmatched_radiomics_and_rano_rows() -> None:
    radiomics_df = pd.DataFrame(
        {
            "patient_id": ["Patient-001", "Patient-001"],
            "week": [0, 5],
            "scan_id": [101, 102],
            "x_week": [0.0, 5.0],
            "suffix": [0, 0],
            "tumor_volume_mm3": [1000.0, 1100.0],
        }
    )
    rano_df = pd.DataFrame(
        {
            "patient_id": ["Patient-001", "Patient-001"],
            "visit_week": [0, 8],
            "rano_label": ["Pre-Op", "SD"],
            "followup_id": [901, 902],
        }
    )
    report = _visit_match_report_from_frames(radiomics_df, rano_df)

    matched = report[report["match_type"] != "unmatched"]
    assert len(matched) == 1
    assert matched.iloc[0]["scan_id"] == 101

    unmatched = report[report["match_type"] == "unmatched"]
    reasons = set(unmatched["unmatch_reason"])
    assert "no_rano_label_at_week" in reasons  # week=5 radyomik, RANO yok
    assert "no_radiomics_at_week" in reasons  # week=8 RANO, radyomik yok


def test_visit_match_report_flags_excess_rows_in_same_week_group() -> None:
    """2 radyomik satırı + 1 RANO satırı aynı haftada -- biri eşleşemez."""

    radiomics_df = pd.DataFrame(
        {
            "patient_id": ["Patient-001", "Patient-001"],
            "week": [0, 0],
            "scan_id": [101, 102],
            "x_week": [0.0, 0.05],
            "suffix": [1, 2],
            "tumor_volume_mm3": [1000.0, 1050.0],
        }
    )
    rano_df = pd.DataFrame(
        {
            "patient_id": ["Patient-001"],
            "visit_week": [0],
            "rano_label": ["Pre-Op"],
            "followup_id": [901],
        }
    )
    report = _visit_match_report_from_frames(radiomics_df, rano_df)
    unmatched = report[report["match_type"] == "unmatched"]
    assert "excess_radiomics_in_same_week_group" in set(unmatched["unmatch_reason"])


def test_visit_match_report_empty_inputs_returns_empty_frame() -> None:
    report = _visit_match_report_from_frames(pd.DataFrame(), pd.DataFrame())
    assert report.empty


def test_build_visit_match_report_live_db_produces_matched_and_unmatched(db_conn) -> None:
    report = build_visit_match_report(db_conn)
    assert not report.empty
    assert set(report["match_type"].unique()) <= {"unique", "heuristic_paired", "unmatched"}
    # Önceki davranışta bu satırlar sessizce kayboluyordu -- artık görünürler.
    assert (report["match_type"] == "unmatched").sum() > 0
