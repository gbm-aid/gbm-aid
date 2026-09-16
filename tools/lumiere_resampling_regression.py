"""Patient-001'in dört zaman noktasında LUMIERE resampling regresyonu."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from db_connection import load_project_environment
from pipeline.harmonization import PROJECT_ROOT
from pipeline.resampling import (
    resample_image_and_mask,
    write_mask_overlay_preview,
)
from pipeline.segmentation import nas_root


TIMEPOINTS = ("week-000-1", "week-000-2", "week-044", "week-056")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LUMIERE Patient-001 dört-zaman-noktası 1 mm görüntü/maske "
            "resampling regresyonunu çalıştır."
        )
    )
    parser.add_argument("--patient-id", default="Patient-001")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "lumiere_resampling_regression",
    )
    parser.add_argument("--max-volume-drift", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_project_environment()
    patient_root = nas_root() / "lumiere" / "Imaging" / args.patient_id
    results: dict[str, object] = {
        "patient_id": args.patient_id,
        "output_spacing_mm": [1.0, 1.0, 1.0],
        "interpolation": {
            "image": "linear",
            "mask": "nearest_neighbor",
        },
        "timepoints": {},
    }

    for timepoint in TIMEPOINTS:
        visit_root = patient_root / timepoint
        image_path = visit_root / "CT1.nii.gz"
        mask_path = (
            visit_root
            / "DeepBraTumIA-segmentation"
            / "native"
            / "segmentation"
            / "ct1_seg_mask.nii.gz"
        )
        output_dir = args.output_root / args.patient_id / timepoint
        try:
            qc = resample_image_and_mask(
                image_path,
                mask_path,
                output_dir=output_dir,
                max_volume_drift=args.max_volume_drift,
            )
            slice_index = write_mask_overlay_preview(
                qc["image_output_path"],
                qc["mask_output_path"],
                output_dir / "overlay.png",
            )
            qc["overlay_slice_index"] = slice_index
            results["timepoints"][timepoint] = qc
        except Exception as exc:
            results["timepoints"][timepoint] = {
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "detail": str(exc),
            }

    statuses = [
        item["status"]
        for item in results["timepoints"].values()
    ]
    results["status"] = "PASS" if set(statuses) == {"PASS"} else "FAIL"
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / f"{args.patient_id}_qc.json"
    report_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": results["status"],
                "report_path": str(report_path),
                "timepoints": {
                    timepoint: {
                        "status": item["status"],
                        "input_geometry_match": item.get(
                            "input_geometry_match"
                        ),
                        "output_geometry_match": item.get(
                            "output_geometry_match"
                        ),
                        "volume_drift_fraction": item.get(
                            "volume_drift_fraction"
                        ),
                    }
                    for timepoint, item in results["timepoints"].items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
