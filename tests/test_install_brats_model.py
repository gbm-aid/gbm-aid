from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.install_brats_pretrained_model import _validate_installed_metadata


def _write_dataset_json(
    results_root: Path,
    channel_names: dict[str, str],
) -> None:
    metadata = (
        results_root
        / "Dataset002_BRATS19"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        / "dataset.json"
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps(
            {
                "channel_names": channel_names,
                "labels": {
                    "background": 0,
                    "whole tumor": [1, 2, 3],
                },
            }
        ),
        encoding="utf-8",
    )


def test_model_metadata_accepts_brats_channel_order(tmp_path: Path) -> None:
    _write_dataset_json(
        tmp_path,
        {"0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"},
    )
    metadata = _validate_installed_metadata(tmp_path)
    assert metadata["channel_names"]["1"] == "T1ce"


def test_model_metadata_rejects_wrong_channel_order(tmp_path: Path) -> None:
    _write_dataset_json(
        tmp_path,
        {"0": "FLAIR", "1": "T1ce", "2": "T2", "3": "T1"},
    )
    with pytest.raises(ValueError, match="kanal sırası"):
        _validate_installed_metadata(tmp_path)
