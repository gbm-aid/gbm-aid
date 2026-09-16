"""Tek bir LUMIERE görüntü-maske çiftini ortak grid'e resample et."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# `.resolve()` DEĞİL `.absolute()` -- `.resolve()` Windows'ta `subst X:`
# eşlemesini gerçek (Türkçe karakterli) yola geri çözer ve ITK'nın NIfTI
# yazımını bozar. Bkz. tests/test_no_resolve_path_regression.py
PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.resampling import resample_image_and_mask  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LUMIERE görüntü+maskeyi 1 mm ortak grid'e resample et."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-volume-drift", type=float, default=0.10)
    args = parser.parse_args()

    result = resample_image_and_mask(
        args.image,
        args.mask,
        output_dir=args.output_dir,
        max_volume_drift=args.max_volume_drift,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
