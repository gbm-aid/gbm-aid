"""Yeni-hasta nnU-Net demosunun kurulum ve güvenlik kapılarını raporla."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path

from db_connection import load_project_environment
from pipeline.nnunet_runtime import find_nnunet_command


def main() -> int:
    load_project_environment()
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "scope": "new_patient_demo_only",
        "fine_tune_allowed": False,
        "dsc_acceptance_target": 0.85,
        "checks": {},
    }
    checks = report["checks"]

    try:
        import torch

        checks["torch"] = {
            "status": "PASS",
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        }
    except Exception as exc:
        checks["torch"] = {
            "status": "FAIL",
            "detail": f"{type(exc).__name__}: {exc}",
        }

    try:
        version = importlib.metadata.version("nnunetv2")
        checks["nnunetv2"] = {
            "status": "PASS",
            "version": version,
            "predict_executable": find_nnunet_command("nnUNetv2_predict"),
        }
    except importlib.metadata.PackageNotFoundError:
        checks["nnunetv2"] = {"status": "FAIL", "detail": "kurulu değil"}

    for variable in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        value = os.environ.get(variable)
        checks[variable] = {
            "status": "PASS" if value and Path(value).is_dir() else "FAIL",
            "configured": bool(value),
            "path_exists": bool(value and Path(value).is_dir()),
        }

    dataset = os.environ.get(
        "GBMAID_NNUNET_DATASET",
        "Dataset002_BRATS19",
    )
    results_root = os.environ.get("nnUNet_results")
    model_path = Path(results_root) / dataset if results_root else None
    metadata_files = (
        sorted(model_path.glob("**/dataset.json"))
        if model_path is not None and model_path.is_dir()
        else []
    )
    model_contract_ok = False
    model_channels = None
    model_labels = None
    fold_count = 0
    if metadata_files:
        metadata = json.loads(
            metadata_files[0].read_text(encoding="utf-8")
        )
        model_channels = {
            str(key): str(value).casefold()
            for key, value in (
                metadata.get("channel_names")
                or metadata.get("modality")
                or {}
            ).items()
        }
        model_labels = metadata.get("labels")
        model_contract_ok = model_channels == {
            "0": "t1",
            "1": "t1ce",
            "2": "t2",
            "3": "flair",
        }
        trainer_root = metadata_files[0].parent
        fold_count = len(
            [
                path
                for path in trainer_root.glob("fold_*")
                if path.is_dir()
            ]
        )
    checks["pretrained_model"] = {
        "status": (
            "PASS"
            if model_path is not None
            and model_path.is_dir()
            and model_contract_ok
            and fold_count == 5
            else "FAIL"
        ),
        "dataset": dataset,
        "installed": bool(model_path is not None and model_path.is_dir()),
        "channel_contract_ok": model_contract_ok,
        "channel_names": model_channels,
        "labels": model_labels,
        "fold_count": fold_count,
    }
    checks["demo_gate"] = {
        "status": (
            "PASS"
            if os.environ.get("GBMAID_ENABLE_NEW_PATIENT_NNUNET") == "1"
            else "BLOCKED"
        ),
        "enabled": os.environ.get("GBMAID_ENABLE_NEW_PATIENT_NNUNET") == "1",
    }

    required_checks = (
        "torch",
        "nnunetv2",
        "nnUNet_raw",
        "nnUNet_preprocessed",
        "nnUNet_results",
        "pretrained_model",
    )
    report["status"] = (
        "READY"
        if all(checks[name]["status"] == "PASS" for name in required_checks)
        else "NOT_READY"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
