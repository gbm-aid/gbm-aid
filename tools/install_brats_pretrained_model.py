"""BraTS pretrained nnU-Net v2 modelini indir, doğrula ve kur."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from db_connection import load_project_environment
from pipeline.harmonization import PROJECT_ROOT
from pipeline.nnunet_runtime import find_nnunet_command


MODEL_FILE = "Dataset002_BRATS19.zip"
MODEL_URL = (
    "https://zenodo.org/records/11582627/files/"
    "Dataset002_BRATS19.zip?download=1"
)
EXPECTED_MD5 = "23a3f55dead4a6642271a08d1a503bbb"
EXPECTED_CHANNELS = {
    "0": "t1",
    "1": "t1ce",
    "2": "t2",
    "3": "flair",
}


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_with_resume(destination: Path) -> None:
    import certifi
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.is_file() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(
        MODEL_URL,
        headers=headers,
        stream=True,
        timeout=(20, 120),
        verify=certifi.where(),
    ) as response:
        status_code = response.status_code
        if existing and status_code == 200:
            existing = 0
            mode = "wb"
        elif status_code in (200, 206):
            mode = "ab" if existing else "wb"
        else:
            raise RuntimeError(
                f"Model indirme HTTP durumu beklenmiyor: {status_code}"
            )

        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    os.replace(partial, destination)


def _validate_installed_metadata(results_root: Path) -> dict[str, object]:
    dataset_root = results_root / "Dataset002_BRATS19"
    metadata_files = sorted(dataset_root.glob("**/dataset.json"))
    if not metadata_files:
        raise RuntimeError(
            f"Kurulan modelde dataset.json bulunamadı: {dataset_root}"
        )

    payload = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    raw_channels = payload.get("channel_names") or payload.get("modality")
    if not isinstance(raw_channels, dict):
        raise ValueError("Model dataset.json kanal sözleşmesi yok.")
    channels = {
        str(key): str(value).casefold()
        for key, value in raw_channels.items()
    }
    if channels != EXPECTED_CHANNELS:
        raise ValueError(
            f"BraTS kanal sırası beklenen sözleşmeyle uyuşmuyor: {channels}"
        )
    return {
        "dataset_root": str(dataset_root),
        "metadata_path": str(metadata_files[0]),
        "channel_names": raw_channels,
        "labels": payload.get("labels"),
    }


def main() -> int:
    load_project_environment()
    installer = find_nnunet_command(
        "nnUNetv2_install_pretrained_model_from_zip"
    )
    if installer is None:
        raise RuntimeError(
            "nnU-Net v2 kurulu değil. Önce requirements-nnunet.txt ortamını kur."
        )

    results_value = os.environ.get("nnUNet_results")
    if not results_value:
        raise KeyError("nnUNet_results .env içinde tanımlı değil.")
    results_root = Path(results_value)
    results_root.mkdir(parents=True, exist_ok=True)

    archive = (
        PROJECT_ROOT
        / "artifacts"
        / "nnunet"
        / "downloads"
        / MODEL_FILE
    )
    if not archive.is_file() or _md5(archive) != EXPECTED_MD5:
        _download_with_resume(archive)

    actual_md5 = _md5(archive)
    if actual_md5 != EXPECTED_MD5:
        raise ValueError(
            f"Model checksum uyuşmuyor: expected={EXPECTED_MD5}, "
            f"actual={actual_md5}"
        )

    completed = subprocess.run(
        [installer, str(archive)],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Pretrained model kurulamadı. "
            f"returncode={completed.returncode}; stderr={completed.stderr[-2000:]}"
        )

    metadata = _validate_installed_metadata(results_root)
    print(
        json.dumps(
            {
                "status": "INSTALLED",
                "archive_md5": actual_md5,
                "metadata": metadata,
                "demo_gate_enabled": (
                    os.environ.get("GBMAID_ENABLE_NEW_PATIENT_NNUNET") == "1"
                ),
                "note": (
                    "Model kurulumu demo kapısını açmaz; DSC >= 0.85 etiketli "
                    "yeni-hasta validasyonunda doğrulanmalıdır."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
