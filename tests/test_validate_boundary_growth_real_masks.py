from __future__ import annotations

import numpy as np
import pytest

from tools.validate_boundary_growth_real_masks import (
    _cavity_fraction,
    _select_representative_cases,
    _test_border_capacity_guard,
    _touches_border,
)


def test_touches_border_true_when_mask_reaches_face() -> None:
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[0, 5, 5] = True  # ilk z-dilimine değiyor
    assert _touches_border(mask)


def test_touches_border_false_for_interior_mask() -> None:
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[4:6, 4:6, 4:6] = True
    assert not _touches_border(mask)


def test_cavity_fraction_zero_for_solid_sphere() -> None:
    zz, yy, xx = np.meshgrid(np.arange(20), np.arange(20), np.arange(20), indexing="ij")
    dist = np.sqrt((zz - 10) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2)
    mask = dist <= 6
    assert _cavity_fraction(mask) == pytest.approx(0.0, abs=1e-9)


def test_cavity_fraction_positive_for_hollow_shell() -> None:
    zz, yy, xx = np.meshgrid(np.arange(20), np.arange(20), np.arange(20), indexing="ij")
    dist = np.sqrt((zz - 10) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2)
    shell = (dist <= 6) & (dist >= 4)  # içi boş küre kabuğu
    fraction = _cavity_fraction(shell)
    assert fraction > 0.1


def test_select_representative_cases_picks_distinct_extremes() -> None:
    descriptions = [
        {
            "folder": "small",
            "n_voxels": 10,
            "volume_mm3": 10.0,
            "n_components": 1,
            "touches_border": False,
            "cavity_fraction": 0.0,
        },
        {
            "folder": "medium",
            "n_voxels": 100,
            "volume_mm3": 100.0,
            "n_components": 2,
            "touches_border": False,
            "cavity_fraction": 0.05,
        },
        {
            "folder": "large_multi_cavity_border",
            "n_voxels": 1000,
            "volume_mm3": 1000.0,
            "n_components": 5,
            "touches_border": True,
            "cavity_fraction": 0.2,
        },
    ]
    selected = _select_representative_cases(descriptions)
    # "large_multi_cavity_border" hem en büyük hem en çok-parçalı hem en
    # boşluklu hem sınıra-dayanan -- dedup kuralı gereği yalnız İLK
    # eşleştiği etiket (insertion sırasına göre "buyuk") tutulur, aynı
    # klasör birden fazla satır olarak TEKRARLANMAZ.
    assert selected["kucuk"]["folder"] == "small"
    assert selected["buyuk"]["folder"] == "large_multi_cavity_border"
    assert selected["orta"]["folder"] == "medium"
    assert "cok_parcali" not in selected
    assert "ic_bosluklu" not in selected
    assert "sinira_dayanan" not in selected
    assert len(selected) == 3


def test_select_representative_cases_no_border_case_omits_key() -> None:
    descriptions = [
        {
            "folder": "a",
            "n_voxels": 10,
            "volume_mm3": 10.0,
            "n_components": 1,
            "touches_border": False,
            "cavity_fraction": 0.0,
        },
        {
            "folder": "b",
            "n_voxels": 20,
            "volume_mm3": 20.0,
            "n_components": 1,
            "touches_border": False,
            "cavity_fraction": 0.0,
        },
    ]
    selected = _select_representative_cases(descriptions)
    assert "sinira_dayanan" not in selected


def test_border_capacity_guard_raises_value_error_without_silent_clipping() -> None:
    result = _test_border_capacity_guard()
    assert result["raised"] is True
    assert "sığmıyor" in result["error"]
