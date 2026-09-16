"""Bir NIfTI görüntü-maske çiftinin geometrisini doğrula."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.resampling import validate_image_mask_geometry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Görüntü-maske geometri QC.")
    parser.add_argument("image", type=Path)
    parser.add_argument("mask", type=Path)
    args = parser.parse_args()

    result = validate_image_mask_geometry(args.image, args.mask)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["geometry_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
