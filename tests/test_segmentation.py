from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from pipeline.segmentation import (
    ImageMaskGeometryError,
    resolve_nas_path,
    resolve_ready_mask,
)


def _write_nifti(
    path: Path,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.ones((4, 5, 6), dtype=np.uint8))
    image.SetSpacing(spacing)
    image.SetOrigin((3.0, -2.0, 7.0))
    sitk.WriteImage(image, str(path), True)
    return path


def test_resolve_nas_path_converts_posix_relative_path(tmp_path: Path) -> None:
    expected = _write_nifti(tmp_path / "tcga-gbm" / "P1" / "T1c.nii.gz")
    resolved = resolve_nas_path(
        "tcga-gbm/P1/T1c.nii.gz",
        root=tmp_path,
    )
    assert resolved == expected

    with pytest.raises(ValueError, match="Güvensiz"):
        resolve_nas_path("../outside.nii.gz", root=tmp_path)


def test_tcga_resolver_prefers_whole_mask(tmp_path: Path) -> None:
    image = _write_nifti(tmp_path / "tcga-gbm" / "P1" / "T1c.nii.gz")
    whole = _write_nifti(image.parent / "whole.nii.gz")
    _write_nifti(image.parent / "core.nii.gz")

    result = resolve_ready_mask(
        source="TCGA-GBM",
        image_path=image,
        patient_id="P1",
        scan_id=1,
    )

    assert result["mask_path"] == str(whole)
    assert result["mask_source"] == "provided_whole_tumor"
    assert result["confidence_score"] is None
    assert result["geometry"]["geometry_match"] is True


def test_upenn_resolver_uses_automated_only_when_expert_missing(
    tmp_path: Path,
) -> None:
    nifti_root = tmp_path / "upenn-gbm" / "UPENN-GBM" / "NIfTI-files"
    scan_key = "UPENN-GBM-00001_11"
    image = _write_nifti(
        nifti_root
        / "images_structural"
        / scan_key
        / f"{scan_key}_T1GD.nii.gz"
    )
    automated = _write_nifti(
        nifti_root
        / "automated_segm"
        / f"{scan_key}_automated_approx_segm.nii.gz"
    )

    result = resolve_ready_mask(
        source="UPenn-GBM",
        image_path=image,
        patient_id="UPENN-GBM-00001",
        scan_id=155,
    )

    assert result["mask_path"] == str(automated)
    assert result["mask_source"] == "upenn_automated_approx"
    assert result["warnings"]


def test_lumiere_resolver_rejects_geometry_mismatch(tmp_path: Path) -> None:
    registered = (
        tmp_path
        / "lumiere"
        / "Imaging"
        / "Patient-001"
        / "week-056"
        / "HD-GLIO-AUTO-segmentation"
        / "registered"
    )
    image = _write_nifti(registered / "CT1_r2s_bet_reg.nii.gz")
    _write_nifti(
        registered / "segmentation.nii.gz",
        spacing=(2.0, 1.0, 1.0),
    )

    with pytest.raises(ImageMaskGeometryError, match="geometrisi eşleşmiyor"):
        resolve_ready_mask(
            source="LUMIERE",
            image_path=image,
            patient_id="Patient-001",
            scan_id=2854,
        )
