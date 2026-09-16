"""Tek NIfTI üzerinde N4 çalıştır, sayısal QC ve orta-kesit PNG üret."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# `.resolve()` DEĞİL `.absolute()` -- `.resolve()` Windows'ta `subst X:`
# eşlemesini gerçek (Türkçe karakterli) yola geri çözer ve ITK'nın NIfTI
# yazımını bozar. Bkz. tests/test_no_resolve_path_regression.py
PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from pipeline.harmonization import apply_n4_bias_correction  # noqa: E402


def _summary(array: np.ndarray, foreground: np.ndarray) -> dict[str, float]:
    values = array[foreground].astype(np.float64, copy=False)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _write_preview(
    before: np.ndarray,
    after: np.ndarray,
    foreground: np.ndarray,
    destination: Path,
) -> int:
    slice_index = int(np.argmax(foreground.sum(axis=(1, 2))))
    values = np.concatenate(
        [before[foreground], after[foreground]]
    ).astype(np.float64, copy=False)
    low, high = np.percentile(values, [1.0, 99.0])
    if high <= low:
        high = low + 1.0

    def scale(slice_array: np.ndarray) -> np.ndarray:
        clipped = np.clip(slice_array, low, high)
        return np.asarray((clipped - low) * 255.0 / (high - low), dtype=np.uint8)

    before_slice = scale(before[slice_index])
    after_slice = scale(after[slice_index])
    separator = np.full((before_slice.shape[0], 8), 255, dtype=np.uint8)
    montage = np.concatenate([before_slice, separator, after_slice], axis=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(montage), str(destination))
    return slice_index


def main() -> int:
    parser = argparse.ArgumentParser(description="N4 sayısal ve görsel QC.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()

    output_path = Path(apply_n4_bias_correction(str(args.image)))
    original_image = sitk.ReadImage(str(args.image))
    corrected_image = sitk.ReadImage(str(output_path))
    before = sitk.GetArrayFromImage(original_image).astype(np.float32, copy=False)
    after = sitk.GetArrayFromImage(corrected_image).astype(np.float32, copy=False)
    foreground = np.isfinite(before) & (before != 0)
    slice_index = _write_preview(before, after, foreground, args.preview)

    result = {
        "input_path": str(args.image),
        "output_path": str(output_path),
        "preview_path": str(args.preview),
        "preview_layout": "left=before, right=after",
        "slice_index": slice_index,
        "geometry_preserved": (
            original_image.GetSize() == corrected_image.GetSize()
            and original_image.GetSpacing() == corrected_image.GetSpacing()
            and original_image.GetOrigin() == corrected_image.GetOrigin()
            and original_image.GetDirection() == corrected_image.GetDirection()
        ),
        "before": _summary(before, foreground),
        "after": _summary(after, foreground),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["geometry_preserved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
