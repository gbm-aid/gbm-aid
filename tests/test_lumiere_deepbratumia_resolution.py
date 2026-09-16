"""Hafta 2 kararı: LUMIERE ana kaynağı DeepBraTumIA, HD-GLIO duyarlılık analizi.

Bu dosya `tests/test_segmentation.py`'deki mevcut LUMIERE/HD-GLIO fallback
testini bozmadan (o test hâlâ değişmeden geçiyor), yeni DeepBraTumIA-öncelikli
davranışı ayrıca doğrular.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from pipeline.segmentation import resolve_ready_mask


def _write_nifti(
    path: Path,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    array: np.ndarray | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if array is None:
        array = np.ones((4, 5, 6), dtype=np.uint8)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path), True)
    return path


def test_lumiere_resolver_prefers_deepbratumia_native_over_hd_glio(
    tmp_path: Path,
) -> None:
    week_root = tmp_path / "lumiere" / "Imaging" / "Patient-001" / "week-000-1"

    # HD-GLIO'nun ön-işlenmiş görüntüsü (mr_scans.file_path'in gerçekte işaret ettiği dosya)
    hd_glio_image = _write_nifti(
        week_root / "HD-GLIO-AUTO-segmentation" / "registered" / "CT1_r2s_bet_reg.nii.gz",
        spacing=(0.36, 0.36, 6.0),
    )
    _write_nifti(
        week_root / "HD-GLIO-AUTO-segmentation" / "registered" / "segmentation.nii.gz",
        spacing=(0.36, 0.36, 6.0),
    )

    # Ham (root) görüntü + DeepBraTumIA native maskesi -- aynı geometride
    raw_image = _write_nifti(week_root / "CT1.nii.gz")
    dbtia_mask = _write_nifti(
        week_root
        / "DeepBraTumIA-segmentation"
        / "native"
        / "segmentation"
        / "ct1_seg_mask.nii.gz"
    )

    result = resolve_ready_mask(
        source="LUMIERE",
        image_path=hd_glio_image,
        patient_id="Patient-001",
        scan_id=2854,
    )

    assert result["mask_source"] == "lumiere_deepbratumia_native"
    assert result["mask_path"] == str(dbtia_mask)
    assert result["image_path"] == str(raw_image)
    assert not result["warnings"]


def test_lumiere_resolver_falls_back_to_hd_glio_when_deepbratumia_missing(
    tmp_path: Path,
) -> None:
    week_root = tmp_path / "lumiere" / "Imaging" / "Patient-099" / "week-010"
    hd_glio_image = _write_nifti(
        week_root / "HD-GLIO-AUTO-segmentation" / "registered" / "T1_r2s_bet_reg.nii.gz"
    )
    hd_glio_mask = _write_nifti(
        week_root / "HD-GLIO-AUTO-segmentation" / "registered" / "segmentation.nii.gz"
    )
    # DeepBraTumIA-segmentation klasörü hiç yok

    result = resolve_ready_mask(
        source="LUMIERE",
        image_path=hd_glio_image,
        patient_id="Patient-099",
        scan_id=1234,
    )

    assert result["mask_source"] == "lumiere_hd_glio_fallback"
    assert result["mask_path"] == str(hd_glio_mask)
    assert result["image_path"] == str(hd_glio_image)
    assert result["warnings"]
