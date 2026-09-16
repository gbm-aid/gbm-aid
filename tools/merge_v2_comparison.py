"""V2 varyant karsilastirma tablosunu HER varyantin kendi ciktisindan kurar.

NEDEN GEREKLI (2026-08-18, Codex cikis incelemesi bulgusu):
`train_cox_week3.py::write_variant_comparison_table()` tabloyu her cagrida
SIFIRDAN yaziyor ve yalniz O CAGRIDAKI varyantlari iceriyor. Dort varyant
AYRI AYRI kosuldugu icin (checkpoint olmadigindan, bkz. run_v2_variants.ps1)
her kosu bir oncekinin tablosunu EZIYOR -- sonunda yalniz son varyant kalir.

Bu, "kosulan HER varyantin sonucu raporda yer alir" seffaflik kuralinin
(CLAUDE.md, 2026-08-18) fiilen ihlalidir.

Bu script her varyantin KENDI `week3_<varyant>_run_metadata.json` /
`_external_test.csv` dosyalarindan tabloyu YENIDEN kurar ve EKSIK/BASARISIZ
varyantlari ACIKCA isaretler -- sessiz dusme yok.

Salt-okunur: yalniz `week3_v2_variant_comparison.csv` yazar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

VARIANTS = ["v1_referans", "v2a_mgmt", "v2b_mgmt_spline", "v2c_mgmt_spline_wttc"]


def collect(output_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for name in VARIANTS:
        row: dict = {"variant": name}
        ext = output_dir / f"week3_{name}_external_test.csv"
        meta = output_dir / f"week3_{name}_run_metadata.json"
        folds = output_dir / f"week3_{name}_fold_results.csv"

        if not ext.exists() and not meta.exists():
            row["durum"] = "KOSULMADI_VEYA_COKTU"
            rows.append(row)
            continue

        row["durum"] = "TAMAM"
        if ext.exists():
            try:
                e = pd.read_csv(ext)
                if len(e):
                    r0 = e.iloc[0]
                    for src, dst in (
                        ("c_index", "harici_c_index"),
                        ("ci_lower", "ci_alt"),
                        ("ci_upper", "ci_ust"),
                        ("n_patients", "harici_n"),
                        ("n_events", "harici_olay"),
                        ("n_final_features", "secilen_ozellik"),
                        ("clinical_extra_columns", "klinik_kovaryat"),
                    ):
                        if src in e.columns:
                            row[dst] = r0[src]
                    if {"ci_lower", "ci_upper"} <= set(e.columns):
                        row["ci_genisligi"] = round(
                            float(r0["ci_upper"]) - float(r0["ci_lower"]), 4
                        )
            except Exception as exc:  # noqa: BLE001
                row["durum"] = f"CIKTI_OKUNAMADI:{type(exc).__name__}"

        if folds.exists():
            try:
                f = pd.read_csv(folds)
                if "c_index" in f.columns and len(f):
                    row["ic_cv_ort"] = round(float(f["c_index"].mean()), 4)
                    row["ic_cv_std"] = round(float(f["c_index"].std()), 4)
                    row["n_fold"] = len(f)
                if "used_fallback" in f.columns:
                    row["fallback_fold"] = int(f["used_fallback"].sum())
            except Exception:  # noqa: BLE001
                pass

        if meta.exists():
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                arm = m.get("arm") or {}
                if isinstance(arm, dict):
                    row.setdefault("aday_ozellik", arm.get("n_feature_candidates"))
            except Exception:  # noqa: BLE001
                pass
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out_dir = Path(argv[0]) if argv else Path("artifacts/week3/cox_model")
    if not out_dir.exists():
        print(f"HATA: cikti dizini yok: {out_dir}", file=sys.stderr)
        return 2

    table = collect(out_dir)
    dest = out_dir / "week3_v2_variant_comparison.csv"
    table.to_csv(dest, index=False)

    print(f"{len(table)} varyant tarandi -> {dest}")
    print(table.to_string(index=False))

    eksik = table[table["durum"] != "TAMAM"]
    if len(eksik):
        print()
        print("UYARI -- tamamlanmayan varyant(lar):", file=sys.stderr)
        for _, r in eksik.iterrows():
            print(f"   {r['variant']}: {r['durum']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
