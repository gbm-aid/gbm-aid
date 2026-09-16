"""Yalnız yeni-hasta demosu için pretrained BraTS nnU-Net v2 çalıştırıcısı."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
from pipeline.lazy_sitk import sitk  # 2026-09-16: TEMBEL import (SAC blogu) -- bkz. pipeline/lazy_sitk.py

from pipeline.harmonization import _require_nifti, write_image_atomic
from pipeline.nnunet_runtime import find_nnunet_command
from pipeline.resampling import geometry_matches


DEMO_SCOPE = "new_patient_demo"
REQUIRED_MODALITIES = ("T1", "T1ce", "T2", "FLAIR")
BRATS_CHANNELS = {
    "T1": "0000",
    "T1ce": "0001",
    "T2": "0002",
    "FLAIR": "0003",
}
BRATS_VALID_OUTPUT_LABELS = {0, 1, 2, 4}


class NewPatientDemoDisabledError(RuntimeError):
    """Yeni-hasta nnU-Net demo kapısı açık değil."""


class PretrainedModelNotReadyError(RuntimeError):
    """Kurulu nnU-Net veya pretrained model eksik."""


def _require_demo_scope(scope: str) -> None:
    if scope != DEMO_SCOPE:
        raise ValueError(
            "nnU-Net yalnız scope='new_patient_demo' için kullanılabilir. "
            "TCGA/UPenn/LUMIERE hazır maskeleri resolver üzerinden alınmalıdır."
        )
    if os.environ.get("GBMAID_ENABLE_NEW_PATIENT_NNUNET") != "1":
        raise NewPatientDemoDisabledError(
            "Yeni-hasta nnU-Net demo kapısı kapalı. Doğrulama sonrası "
            "GBMAID_ENABLE_NEW_PATIENT_NNUNET=1 ayarlanmalıdır."
        )


def prepare_new_patient_case(
    *,
    case_id: str,
    modalities: dict[str, str | Path],
    destination: str | Path,
    scope: str,
) -> Path:
    """Dört co-registered modaliteyi BraTS kanal sırasıyla inference'a hazırla."""

    _require_demo_scope(scope)
    if not case_id or any(char in case_id for char in ("/", "\\", "..")):
        raise ValueError(f"Geçersiz case_id: {case_id!r}")

    missing = set(REQUIRED_MODALITIES) - set(modalities)
    extra = set(modalities) - set(REQUIRED_MODALITIES)
    if missing or extra:
        raise ValueError(
            f"Modalite sözleşmesi bozuk. Eksik={sorted(missing)}, "
            f"fazla={sorted(extra)}; gerekli={list(REQUIRED_MODALITIES)}"
        )

    images = {
        modality: sitk.ReadImage(
            str(_require_nifti(modalities[modality]))
        )
        for modality in REQUIRED_MODALITIES
    }
    reference = images["T1"]
    mismatched = [
        modality
        for modality, image in images.items()
        if not geometry_matches(reference, image)
    ]
    if mismatched:
        raise ValueError(
            "Yeni-hasta modaliteleri co-registered değil; nnU-Net öncesi "
            f"ortak fiziksel grid gerekli. Uyuşmayan: {mismatched}"
        )

    output_dir = Path(destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    for modality, channel in BRATS_CHANNELS.items():
        write_image_atomic(
            images[modality],
            output_dir / f"{case_id}_{channel}.nii.gz",
        )
    return output_dir


def run_new_patient_demo(
    *,
    case_id: str,
    modalities: dict[str, str | Path],
    work_dir: str | Path,
    output_dir: str | Path,
    scope: str,
) -> dict[str, object]:
    """Kurulu pretrained modeli inference için çağır; eğitim/fine-tune yapma."""

    _require_demo_scope(scope)
    dataset = os.environ.get(
        "GBMAID_NNUNET_DATASET",
        "Dataset002_BRATS19",
    )
    configuration = os.environ.get(
        "GBMAID_NNUNET_CONFIGURATION",
        "3d_fullres",
    )
    trainer = os.environ.get("GBMAID_NNUNET_TRAINER", "nnUNetTrainer")
    plans = os.environ.get("GBMAID_NNUNET_PLANS", "nnUNetPlans")
    executable = find_nnunet_command("nnUNetv2_predict")
    if executable is None:
        raise PretrainedModelNotReadyError(
            "nnUNetv2_predict bulunamadı; nnunetv2 ortamı kurulu değil."
        )

    results_root = os.environ.get("nnUNet_results")
    if not results_root:
        raise PretrainedModelNotReadyError("nnUNet_results tanımlı değil.")
    installed_dataset = Path(results_root) / dataset
    if not installed_dataset.is_dir():
        raise PretrainedModelNotReadyError(
            f"Pretrained model kurulu değil: {installed_dataset}"
        )

    input_dir = prepare_new_patient_case(
        case_id=case_id,
        modalities=modalities,
        destination=Path(work_dir) / "nnunet_input",
        scope=scope,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "-i",
        str(input_dir),
        "-o",
        str(destination),
        "-d",
        dataset,
        "-c",
        configuration,
        "-f",
        "all",
        "-tr",
        trainer,
        "-p",
        plans,
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "nnU-Net inference başarısız. "
            f"returncode={completed.returncode}; stderr={completed.stderr[-2000:]}"
        )

    segmentation_path = destination / f"{case_id}.nii.gz"
    if not segmentation_path.is_file():
        raise RuntimeError(
            f"nnU-Net çıktı maskesi bulunamadı: {segmentation_path}"
        )
    prediction = sitk.ReadImage(str(segmentation_path))
    labels = {
        int(value)
        for value in np.unique(sitk.GetArrayViewFromImage(prediction))
    }
    if not labels.issubset(BRATS_VALID_OUTPUT_LABELS):
        raise ValueError(
            "Beklenmeyen nnU-Net etiketleri. Model metadata'sındaki 3='empty' "
            "çıktıda tümör sınıfı olarak kabul edilmez; beklenen BraTS "
            f"etiketleri={sorted(BRATS_VALID_OUTPUT_LABELS)}, "
            f"görülen={sorted(labels)}"
        )

    return {
        "status": "ok",
        "scope": DEMO_SCOPE,
        "case_id": case_id,
        "segmentation_path": str(segmentation_path),
        "labels": sorted(labels),
        "model": {
            "dataset": dataset,
            "configuration": configuration,
            "trainer": trainer,
            "plans": plans,
            "fine_tuned": False,
        },
        "confidence_score": None,
    }
