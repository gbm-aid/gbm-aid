"""N4 çıktılarından kaynak-bazlı Z-score istatistik dosyası üret."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# `.resolve()` DEĞİL `.absolute()` -- `.resolve()` Windows'ta `subst X:`
# eşlemesini gerçek (Türkçe karakterli) yola geri çözer. Bu script kendisi
# NIfTI YAZMAZ (yalnız istatistik JSON'u), ama `sitk.ReadImage` ile görüntü
# OKUR -- K12 tutarlılığı için aynı kural uygulanır. Bkz.
# tests/test_no_resolve_path_regression.py (2026-09-12'de `IMAGE_WRITING_
# MARKERS`'a `fit_source_zscore_statistics` eklendi, bu dosya artık lint
# taramasına GİRİYOR -- Barış onayı, kanıt: log/2026-09-12.md).
PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.harmonization import fit_source_zscore_statistics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="N4 sonrası NIfTI dosyalarından source Z-score fit et."
    )
    parser.add_argument("source", choices=("TCGA", "UPenn", "LUMIERE"))
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--stats-path",
        type=Path,
        help="Varsayılan: artifacts/zscore/source_stats.json",
    )
    args = parser.parse_args()

    record = fit_source_zscore_statistics(
        args.images,
        args.source,
        stats_path=args.stats_path,
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
