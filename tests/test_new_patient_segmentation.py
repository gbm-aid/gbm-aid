from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from pipeline.new_patient_segmentation import (
    NewPatientDemoDisabledError,
    prepare_new_patient_case,
)


def _write_image(
    path: Path,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    image = sitk.GetImageFromArray(np.ones((4, 5, 6), dtype=np.float32))
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path), True)
    return path


def _modalities(tmp_path: Path) -> dict[str, Path]:
    return {
        modality: _write_image(tmp_path / f"{modality}.nii.gz")
        for modality in ("T1", "T1ce", "T2", "FLAIR")
    }


def test_new_patient_demo_rejects_cohort_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GBMAID_ENABLE_NEW_PATIENT_NNUNET", "1")
    with pytest.raises(ValueError, match="new_patient_demo"):
        prepare_new_patient_case(
            case_id="TCGA-02-0003",
            modalities=_modalities(tmp_path),
            destination=tmp_path / "input",
            scope="TCGA",
        )


def test_new_patient_demo_gate_defaults_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GBMAID_ENABLE_NEW_PATIENT_NNUNET", raising=False)
    with pytest.raises(NewPatientDemoDisabledError):
        prepare_new_patient_case(
            case_id="demo-001",
            modalities=_modalities(tmp_path),
            destination=tmp_path / "input",
            scope="new_patient_demo",
        )


def test_new_patient_demo_stages_brats_channel_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GBMAID_ENABLE_NEW_PATIENT_NNUNET", "1")
    destination = prepare_new_patient_case(
        case_id="demo-001",
        modalities=_modalities(tmp_path),
        destination=tmp_path / "input",
        scope="new_patient_demo",
    )
    assert sorted(path.name for path in destination.iterdir()) == [
        "demo-001_0000.nii.gz",
        "demo-001_0001.nii.gz",
        "demo-001_0002.nii.gz",
        "demo-001_0003.nii.gz",
    ]
    for staged in destination.iterdir():
        assert sitk.ReadImage(str(staged)).GetSize() == (6, 5, 4)


def test_new_patient_demo_rejects_unregistered_modalities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GBMAID_ENABLE_NEW_PATIENT_NNUNET", "1")
    modalities = _modalities(tmp_path)
    modalities["FLAIR"] = _write_image(
        tmp_path / "FLAIR_bad.nii.gz",
        spacing=(2.0, 1.0, 1.0),
    )
    with pytest.raises(ValueError, match="co-registered değil"):
        prepare_new_patient_case(
            case_id="demo-001",
            modalities=modalities,
            destination=tmp_path / "input",
            scope="new_patient_demo",
        )
