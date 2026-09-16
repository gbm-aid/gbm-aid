from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from pipeline.resampling import (
    resample_image_and_mask,
    validate_image_mask_geometry,
)


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    image_array = np.zeros((8, 12, 16), dtype=np.float32)
    image_array[2:7, 3:10, 4:13] = 50.0
    mask_array = np.zeros(image_array.shape, dtype=np.uint8)
    mask_array[3:6, 4:9, 5:12] = 1
    mask_array[4:5, 6:8, 7:10] = 2

    image = sitk.GetImageFromArray(image_array)
    mask = sitk.GetImageFromArray(mask_array)
    for item in (image, mask):
        item.SetSpacing((2.0, 1.5, 3.0))
        item.SetOrigin((10.0, -4.0, 2.0))

    image_path = tmp_path / "CT1.nii.gz"
    mask_path = tmp_path / "ct1_seg_mask.nii.gz"
    sitk.WriteImage(image, str(image_path), True)
    sitk.WriteImage(mask, str(mask_path), True)
    return image_path, mask_path


def test_resampling_produces_common_one_mm_grid(tmp_path: Path) -> None:
    image_path, mask_path = _write_pair(tmp_path)
    result = resample_image_and_mask(
        image_path,
        mask_path,
        output_dir=tmp_path / "resampled",
        max_volume_drift=0.25,
    )
    output_qc = validate_image_mask_geometry(
        result["image_output_path"],
        result["mask_output_path"],
    )

    assert result["status"] == "PASS"
    assert result["input_geometry_match"] is True
    assert result["output_geometry_match"] is True
    assert result["labels_preserved"] is True
    assert output_qc["geometry_match"] is True
    assert output_qc["image_geometry"]["spacing"] == pytest.approx(
        [1.0, 1.0, 1.0]
    )
    assert set(output_qc["mask"]["labels"]) == {0.0, 1.0, 2.0}
