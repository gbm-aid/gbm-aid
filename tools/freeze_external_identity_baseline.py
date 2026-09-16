# -*- coding: utf-8 -*-
"""UCSF harici-test kimlik baseline'ini BILINCLI olarak dondurur.

Kullanim:
  python tools/freeze_external_identity_baseline.py            # dosya yoksa yazar
  python tools/freeze_external_identity_baseline.py --refreeze --reason "..."
                                                               # mevcutun ustune,
                                                               # gerekce ZORUNLU

Dogrulayici scriptler (`evaluate_external_metrics.py`,
`evaluate_v1_calibration.py`) bu dosyayi ASLA kendileri uretmez/yenilemez.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import train_cox_week3 as w3  # noqa: E402
from external_identity_baseline import BASELINE_PATH, compute_full_identity  # noqa: E402
from db_connection import get_connection  # noqa: E402
from psycopg2.extras import RealDictCursor  # noqa: E402

DEFAULT_JUSTIFICATION = (
    "2026-08-19 (sema v2): evaluate_external_metrics.py alti varyantin harici "
    "Harrell C degerlerini kosularin rapor ettikleriyle 6/6 BIT-BIREBIR yeniden "
    "uretti -- cerceveler kosularin verisiyle ozdes; baseline bu cipaya "
    "dayanir. v2 kapsami: UCSF + UPenn, kohort basina pid + TUM-kolon hasta "
    "cercevesi (klinik kovaryatlar dahil) + WT pivot degerleri. TC pivotu "
    "haric (v2c bit-birebir Harrell kapisi dolayli kapsar)."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refreeze", action="store_true")
    ap.add_argument("--reason", default=None)
    args = ap.parse_args()

    if BASELINE_PATH.is_file() and not args.refreeze:
        print(f"Baseline ZATEN VAR: {BASELINE_PATH}\n--refreeze olmadan ustune yazilmaz.",
              file=sys.stderr)
        return 2
    if args.refreeze and not args.reason:
        print("--refreeze icin --reason ZORUNLU (gerekcesiz yenileme yok).",
              file=sys.stderr)
        return 2

    conn = get_connection(readonly=True)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    def load(tag, tool):
        cur.execute("SELECT COUNT(*) AS n FROM radiomics WHERE segmentation_tool=%s",
                    (tool,))
        n = int(cur.fetchone()["n"])
        f = pd.read_pickle(PROJECT_ROOT / f"artifacts/week3/cox_model/_cache/{tag}_long.pkl")
        if len(f) != n:
            raise SystemExit(f"{tag}: onbellek ({len(f)}) != DB ({n}) -- DONDURULMADI.")
        return f

    ucsf_long = load("ucsf", w3.SEGMENTATION_TOOL_UCSF_C32)
    upenn_long = load("upenn", w3.SEGMENTATION_TOOL_UPENN_C32)
    ucsf_pat = w3.fetch_patients_frame(cur, source_name=w3.UCSF_SOURCE_NAME)
    upenn_pat = w3.fetch_patients_frame(cur, source_name=w3.UPENN_SOURCE_NAME)
    cur.close(); conn.close()

    uc_wide, _ = w3.pivot_ucsf_regions_long_to_wide(ucsf_long, regions=("WT",))
    up_wide, _ = w3.pivot_radiomics_long_to_wide(upenn_long, regions=["WT"])
    identity = compute_full_identity(uc_wide, ucsf_pat, up_wide, upenn_pat)
    identity["frozen_at"] = datetime.now(timezone.utc).isoformat()
    identity["justification"] = args.reason or DEFAULT_JUSTIFICATION
    if args.refreeze and BASELINE_PATH.is_file():
        old = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        eski = {k: v for k, v in old.items() if "sha256" in k}
        if not eski and old.get("cohorts"):
            eski = {c: {kk: vv for kk, vv in cv.items() if "sha256" in kk}
                    for c, cv in old["cohorts"].items()}
        identity["refroze_over"] = {
            "frozen_at": old.get("frozen_at"),
            "schema_version": old.get("schema_version", 1),
            "eski_hashler": eski,
        }

    BASELINE_PATH.write_text(
        json.dumps(identity, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONDURULDU -> {BASELINE_PATH}")
    for cohort, cid in identity["cohorts"].items():
        print(f"  [{cohort}] n={cid['n_pivot_patients']} pid={cid['pid_sha256'][:16]} "
              f"patients={cid['patients_sha256'][:16]} wt_pivot={cid['wt_pivot_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
