"""LUMIERE için görüntü-maske resampling ve geometri kalite kontrolü."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from pipeline.lazy_sitk import sitk  # 2026-09-16: TEMBEL import (SAC blogu) -- bkz. pipeline/lazy_sitk.py

from pipeline.harmonization import (
    PROJECT_ROOT,
    _nifti_stem,
    _require_nifti,
    write_image_atomic,
)


def _tuple_close(
    left: Sequence[float],
    right: Sequence[float],
    *,
    tolerance: float = 1e-5,
) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)
        for a, b in zip(left, right)
    )


def geometry_matches(
    image: sitk.Image,
    mask: sitk.Image,
    *,
    tolerance: float = 1e-5,
) -> bool:
    """Size, spacing, origin ve direction alanlarının tamamını karşılaştır."""

    return (
        image.GetSize() == mask.GetSize()
        and _tuple_close(image.GetSpacing(), mask.GetSpacing(), tolerance=tolerance)
        and _tuple_close(image.GetOrigin(), mask.GetOrigin(), tolerance=tolerance)
        and _tuple_close(
            image.GetDirection(),
            mask.GetDirection(),
            tolerance=tolerance,
        )
    )


def _geometry_dict(image: sitk.Image) -> dict[str, list[float] | list[int]]:
    return {
        "size": list(image.GetSize()),
        "spacing": [float(value) for value in image.GetSpacing()],
        "origin": [float(value) for value in image.GetOrigin()],
        "direction": [float(value) for value in image.GetDirection()],
    }


def _mask_metrics(mask: sitk.Image) -> dict[str, object]:
    array = sitk.GetArrayViewFromImage(mask)
    labels = sorted(
        float(value)
        for value in np.unique(array)
        if math.isfinite(float(value))
    )
    foreground_voxels = int(np.count_nonzero(array))
    voxel_volume = float(np.prod(mask.GetSpacing(), dtype=np.float64))
    return {
        "labels": labels,
        "foreground_voxels": foreground_voxels,
        "foreground_volume_mm3": foreground_voxels * voxel_volume,
    }


def validate_image_mask_geometry(
    image_path: str | Path,
    mask_path: str | Path,
) -> dict[str, object]:
    """Diskteki bir görüntü-maske çiftinin geometri ve hacim özetini döndür."""

    image_file = _require_nifti(image_path)
    mask_file = _require_nifti(mask_path)
    image = sitk.ReadImage(str(image_file))
    mask = sitk.ReadImage(str(mask_file))
    return {
        "image_path": str(image_file),
        "mask_path": str(mask_file),
        "geometry_match": geometry_matches(image, mask),
        "image_geometry": _geometry_dict(image),
        "mask_geometry": _geometry_dict(mask),
        "mask": _mask_metrics(mask),
    }


def _reference_grid(
    image: sitk.Image,
    output_spacing: Sequence[float],
) -> sitk.Image:
    if len(output_spacing) != image.GetDimension():
        raise ValueError("output_spacing görüntü boyutuyla aynı uzunlukta olmalı.")
    spacing = tuple(float(value) for value in output_spacing)
    if any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("output_spacing sonlu ve pozitif olmalı.")

    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    output_size = [
        max(
            1,
            int(
                round(
                    (input_size[index] - 1)
                    * input_spacing[index]
                    / spacing[index]
                )
            )
            + 1,
        )
        for index in range(image.GetDimension())
    ]

    reference = sitk.Image(output_size, sitk.sitkFloat32)
    reference.SetSpacing(spacing)
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    return reference


def _output_directory(
    image_path: Path,
    mask_path: Path,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        destination = Path(output_dir)
    else:
        configured = os.environ.get("GBMAID_RESAMPLED_ROOT")
        root = (
            Path(configured).expanduser()
            if configured
            else PROJECT_ROOT / "artifacts" / "resampled"
        )
        pair_key = (
            f"{image_path.absolute()}|{mask_path.absolute()}".casefold().encode("utf-8")
        )
        destination = root / hashlib.sha256(pair_key).hexdigest()[:12]
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def is_isotropic_spacing(
    image_or_path: sitk.Image | str | Path,
    *,
    target_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    tolerance: float = 0.02,
) -> bool:
    """Görüntünün spacing'i hedef izotropik değere (varsayılan 1mm) yakın mı?

    `geometry_matches`'in sıkı `1e-5` toleransından farklı olarak, burada
    "zaten standart mı" pratik kontrolü için gevşek bir tolerans (varsayılan
    0.02mm) kullanılır -- kayan nokta yuvarlama farkları (örn. 0.999998mm)
    yanlışlıkla resampling tetiklemesin diye.
    """

    image = (
        image_or_path
        if isinstance(image_or_path, sitk.Image)
        else sitk.ReadImage(str(_require_nifti(image_or_path)))
    )
    spacing = image.GetSpacing()
    if len(spacing) != len(target_spacing):
        raise ValueError("target_spacing görüntü boyutuyla aynı uzunlukta olmalı.")
    return all(
        abs(float(actual) - float(target)) <= tolerance
        for actual, target in zip(spacing, target_spacing)
    )


def _single_output_directory(image_path: Path, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        destination = Path(output_dir)
    else:
        configured = os.environ.get("GBMAID_RESAMPLED_ROOT")
        root = (
            Path(configured).expanduser()
            if configured
            else PROJECT_ROOT / "artifacts" / "resampled"
        )
        key = str(image_path.absolute()).casefold().encode("utf-8")
        destination = root / hashlib.sha256(key).hexdigest()[:12]
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def resample_image_only(
    image_path: str | Path,
    *,
    output_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Yalnız görüntüyü (maskesiz) fiziksel uzayda 1mm izotropik grid'e taşı.

    `resample_image_and_mask`'ten farkı: eşleşen bir maskeye ihtiyaç duymaz
    -- kohort-çapında N4/Z-score harmonizasyonu (Hafta 2) maske kullanmadan
    yalnız görüntü uzayında çalıştığı için gereklidir. Maske-eşleşmeli
    resampling (PyRadiomics/hacim QC öncesi) hâlâ `resample_image_and_mask`
    ile yapılmalıdır.
    """

    image_file = _require_nifti(image_path)
    image = sitk.Cast(sitk.ReadImage(str(image_file)), sitk.sitkFloat32)

    reference = _reference_grid(image, output_spacing)
    identity = sitk.Transform(image.GetDimension(), sitk.sitkIdentity)
    resampled_image = sitk.Resample(
        image,
        reference,
        identity,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    destination = _single_output_directory(image_file, output_dir)
    image_output = destination / f"{_nifti_stem(image_file)}_1mm.nii.gz"
    write_image_atomic(resampled_image, image_output)

    return {
        "status": "PASS",
        "image_output_path": str(image_output),
        "input_geometry": _geometry_dict(image),
        "output_geometry": _geometry_dict(resampled_image),
    }


def resample_image_and_mask(
    image_path: str | Path,
    mask_path: str | Path,
    *,
    output_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    output_dir: str | Path | None = None,
    max_volume_drift: float = 0.10,
) -> dict[str, object]:
    """Görüntü ve maskeyi görüntünün fiziksel uzayındaki ortak grid'e taşı.

    Görüntü için lineer, maske için nearest-neighbor interpolasyon kullanılır.
    Maskenin etiketleri ve fiziksel foreground hacmi QC raporuna yazılır.
    """

    if not 0 <= max_volume_drift <= 1:
        raise ValueError("max_volume_drift 0 ile 1 arasında olmalı.")

    image_file = _require_nifti(image_path)
    mask_file = _require_nifti(mask_path)
    image = sitk.Cast(sitk.ReadImage(str(image_file)), sitk.sitkFloat32)
    mask = sitk.ReadImage(str(mask_file))
    input_geometry_match = geometry_matches(image, mask)
    input_mask_metrics = _mask_metrics(mask)
    if int(input_mask_metrics["foreground_voxels"]) == 0:
        raise ValueError(f"Girdi maskesi boş: {mask_file}")

    reference = _reference_grid(image, output_spacing)
    identity = sitk.Transform(image.GetDimension(), sitk.sitkIdentity)
    resampled_image = sitk.Resample(
        image,
        reference,
        identity,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    resampled_mask = sitk.Resample(
        mask,
        reference,
        identity,
        sitk.sitkNearestNeighbor,
        0.0,
        mask.GetPixelID(),
    )

    destination = _output_directory(image_file, mask_file, output_dir)
    image_output = destination / f"{_nifti_stem(image_file)}_1mm.nii.gz"
    mask_output = destination / (
        f"{_nifti_stem(mask_file)}_on_{_nifti_stem(image_file)}_1mm.nii.gz"
    )
    write_image_atomic(resampled_image, image_output)
    write_image_atomic(resampled_mask, mask_output)

    output_mask_metrics = _mask_metrics(resampled_mask)
    input_volume = float(input_mask_metrics["foreground_volume_mm3"])
    output_volume = float(output_mask_metrics["foreground_volume_mm3"])
    volume_drift = abs(output_volume - input_volume) / input_volume
    input_labels = set(input_mask_metrics["labels"])
    output_labels = set(output_mask_metrics["labels"])
    labels_preserved = output_labels == input_labels and any(
        label != 0 for label in output_labels
    )
    output_geometry_match = geometry_matches(resampled_image, resampled_mask)

    return {
        "status": (
            "PASS"
            if output_geometry_match
            and labels_preserved
            and volume_drift <= max_volume_drift
            else "FAIL"
        ),
        "image_output_path": str(image_output),
        "mask_output_path": str(mask_output),
        "input_geometry_match": input_geometry_match,
        "output_geometry_match": output_geometry_match,
        "labels_preserved": labels_preserved,
        "volume_drift_fraction": volume_drift,
        "max_volume_drift": max_volume_drift,
        "input_image_geometry": _geometry_dict(image),
        "input_mask_geometry": _geometry_dict(mask),
        "output_geometry": _geometry_dict(resampled_image),
        "input_mask": input_mask_metrics,
        "output_mask": output_mask_metrics,
    }


def write_mask_overlay_preview(
    image_path: str | Path,
    mask_path: str | Path,
    destination: str | Path,
) -> int:
    """En çok tümör voxel'i içeren aksiyel kesiti RGB overlay olarak yaz."""

    image_file = _require_nifti(image_path)
    mask_file = _require_nifti(mask_path)
    image = sitk.ReadImage(str(image_file))
    mask = sitk.ReadImage(str(mask_file))
    if not geometry_matches(image, mask):
        raise ValueError("Overlay öncesi görüntü ve maske geometrisi eşleşmeli.")

    image_array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    mask_array = sitk.GetArrayFromImage(mask)
    slice_index = int(np.argmax(np.count_nonzero(mask_array, axis=(1, 2))))
    image_slice = image_array[slice_index]
    mask_slice = mask_array[slice_index]

    finite_foreground = image_array[np.isfinite(image_array) & (image_array != 0)]
    if finite_foreground.size == 0:
        raise ValueError("Overlay için görüntü foreground'u boş.")
    low, high = np.percentile(finite_foreground, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    gray = np.asarray(
        np.clip((image_slice - low) * 255.0 / (high - low), 0, 255),
        dtype=np.uint8,
    )
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)

    palette = {
        1: np.array([255.0, 64.0, 64.0], dtype=np.float32),
        2: np.array([64.0, 128.0, 255.0], dtype=np.float32),
        3: np.array([64.0, 255.0, 96.0], dtype=np.float32),
    }
    for raw_label in np.unique(mask_slice):
        label = int(raw_label)
        if label == 0:
            continue
        color = palette.get(
            label,
            np.array([255.0, 220.0, 64.0], dtype=np.float32),
        )
        selected = mask_slice == raw_label
        rgb[selected] = 0.45 * rgb[selected] + 0.55 * color

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = sitk.GetImageFromArray(np.asarray(rgb, dtype=np.uint8), isVector=True)
    sitk.WriteImage(preview, str(output))
    return slice_index
