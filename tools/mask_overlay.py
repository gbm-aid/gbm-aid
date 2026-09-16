"""Geometrisi doğrulanmış görüntü-maske çifti için PNG overlay üret."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# `.resolve()` DEĞİL `.absolute()` -- `.resolve()` Windows'ta `subst X:`
# eşlemesini gerçek (Türkçe karakterli) yola geri çözer ve bu modülün
# çağırdığı `write_mask_overlay_preview` (SimpleITK tabanlı) NIfTI/PNG
# yazımını bozar (K12, 2026-08-14 / 2026-09-12 düzeltmesi).
PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.resampling import write_mask_overlay_preview  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Görüntü+maske overlay PNG.")
    parser.add_argument("image", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    slice_index = write_mask_overlay_preview(args.image, args.mask, args.output)
    print(f"preview={args.output}")
    print(f"slice_index={slice_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
