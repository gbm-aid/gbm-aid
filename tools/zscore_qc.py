"""Fit edilmiş source Z-score istatistiğini tek görüntüde doğrula."""

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

from pipeline.harmonization import apply_zscore_normalization  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Source Z-score apply QC.")
    parser.add_argument("image", type=Path)
    parser.add_argument("source", choices=("TCGA", "UPenn", "LUMIERE"))
    args = parser.parse_args()

    input_image = sitk.ReadImage(str(args.image))
    output_path = Path(apply_zscore_normalization(str(args.image), args.source))
    output_image = sitk.ReadImage(str(output_path))
    input_array = sitk.GetArrayViewFromImage(input_image)
    output_array = sitk.GetArrayViewFromImage(output_image)
    foreground = np.isfinite(input_array) & (input_array != 0)
    values = output_array[foreground].astype(np.float64, copy=False)
    result = {
        "source": args.source,
        "input_path": str(args.image),
        "output_path": str(output_path),
        "foreground_mean": float(values.mean()),
        "foreground_std": float(values.std()),
        "finite": bool(np.isfinite(values).all()),
        "geometry_preserved": (
            input_image.GetSize() == output_image.GetSize()
            and input_image.GetSpacing() == output_image.GetSpacing()
            and input_image.GetOrigin() == output_image.GetOrigin()
            and input_image.GetDirection() == output_image.GetDirection()
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    passed = (
        result["finite"]
        and result["geometry_preserved"]
        and abs(result["foreground_mean"]) < 1e-5
        and abs(result["foreground_std"] - 1.0) < 1e-5
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
