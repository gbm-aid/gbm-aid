"""UCSF-PDGM C32 PILOT (5 hasta) -- SALT-OKUNUR NAS/disk erisimi,
YALNIZ scratchpad'e yazim. artifacts/, DB, raw/ HICBIR YAZMA YOK.

C32 sozlesmesi (decisions/2026-08-13-pyradiomics-c32-bincount-karari.md):
  1) N4 bias-field duzeltmesi          -> UYGULANIR
     (production `pipeline.harmonization.apply_n4_bias_correction()`
     REUSE edilir -- bu fonksiyon kaynak-agnostik, canonical_source()
     cagirmiyor, guvenle kullanilabilir).
  2) T1ce-OZEL kaynak Z-score          -> UYGULANIR
     UCSF, `pipeline.harmonization.canonical_source()`'in SOURCE_ALIASES
     sozlugunde HENUZ TANIMLI DEGIL (yalniz tcga/upenn/lumiere var) --
     bu fonksiyonu UCSF icin cagirmak ValueError firlatir. Production
     dosyasi (paylasilan, ComBat guard'lariyla da bagli) BU PILOT ICIN
     DEGISTIRILMEDI. Bunun yerine fit/apply algoritmasi (Welford online
     mean/std, sadece sonlu+sifir-disi foreground voksel) BIREBIR AYNI
     matematikle burada YEREL olarak yeniden uygulandi.
  3) discretization binCount=32        -> UYGULANIR (binWidth KULLANILMADI)
  4) extractor parametreleri ACIKCA yazili, pyradiomics==3.0.1 (olcum
     asagida dogrulaniyor).

ONEMLI UYARI (pilot kapsam siniri): 5 hastalik Z-score istatistigi
GUVENILIR DEGIL (n=5, tam kohort n=275 olacak). Bu pilotun TEK amaci
"boru hatti calisiyor mu" sorusunu cevaplamak -- URETILEN SAYISAL
DEGERLER tam-kohort cikarimda KULLANILMAYACAK, yalniz pipeline
dogrulamasi icindir.
"""
from __future__ import annotations

# 2026-09-16: sabit kodlanmis mutlak yollar KALDIRILDI -- kullanici dizini
# adi halka acik depoya siziyordu ve bu script baska makinede calismiyordu.
# Yollar artik dosyanin kendi konumundan / ortam degiskeninden turetilir.
import os as _os
from pathlib import Path as _P
_BURASI   = _P(__file__).resolve().parent          # tools/_dogrulama
_KOK_KOD  = _P(__file__).resolve().parents[2]      # gbm-aid mert
_KOK_PROJ = _P(__file__).resolve().parents[3]      # GBM-AID Prototip
_VERI_KOKU = _P(_os.environ.get('GBMAID_VERI_KOKU', str(_KOK_PROJ)))

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = _KOK_KOD
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os

SCRATCH = Path(
    _BURASI / 'scratchpad'
)
os.environ["GBMAID_PROCESSED_ROOT"] = str(SCRATCH / "ucsf_pilot_n4")

from pipeline.harmonization import apply_n4_bias_correction  # noqa: E402
from pipeline.resampling import validate_image_mask_geometry  # noqa: E402

DATA_ROOT = Path(str(_VERI_KOKU / r'PKG - UCSF-PDGM Version 5\UCSF-PDGM-v5'))
PILOT_LIST = SCRATCH / "pilot_5_patients.txt"

REGION_LABELS_UCSF = {"NC": 1, "ED": 2, "ET": 4}  # UPenn ile BIREBIR ayni (olcumle dogrulandi)


def _require_nifti(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI bulunamadi: {path}")
    return path


def _foreground_values(image) -> np.ndarray:
    import SimpleITK as sitk

    array = sitk.GetArrayViewFromImage(image)
    foreground = np.isfinite(array) & (array != 0)
    if not np.any(foreground):
        raise ValueError("Sonlu, sifir-disi foreground voksel yok.")
    return np.asarray(array[foreground], dtype=np.float64)


def fit_ucsf_zscore_local(n4_paths: list[str]) -> dict:
    """`pipeline.harmonization.fit_source_zscore_statistics()` ile BIREBIR
    AYNI Welford online algoritmasi -- yalniz dosyaya YAZMAZ, sozluk doner
    (production stats JSON'larina hicbir sekilde DOKUNMAZ)."""
    import SimpleITK as sitk

    count = 0
    mean = 0.0
    m2 = 0.0
    image_count = 0
    for p in n4_paths:
        values = _foreground_values(sitk.ReadImage(str(p)))
        batch_count = int(values.size)
        batch_mean = float(values.mean(dtype=np.float64))
        batch_m2 = float(np.square(values - batch_mean, dtype=np.float64).sum(dtype=np.float64))
        combined = count + batch_count
        delta = batch_mean - mean
        mean += delta * batch_count / combined
        m2 += batch_m2 + delta * delta * count * batch_count / combined
        count = combined
        image_count += 1
    std = math.sqrt(m2 / count)
    return {"source": "UCSF-PDGM", "mean": mean, "std": std, "voxel_count": count, "image_count": image_count}


def apply_ucsf_zscore_local(n4_path: str, stats: dict, out_dir: Path) -> str:
    import SimpleITK as sitk

    image = sitk.Cast(sitk.ReadImage(str(n4_path)), sitk.sitkFloat32)
    mean, std = stats["mean"], stats["std"]
    normalized = sitk.Cast((image - mean) / std, sitk.sitkFloat32)
    normalized.CopyInformation(image)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(n4_path).stem.replace(".nii", "") + "_zscore.nii.gz")
    sitk.WriteImage(normalized, str(out_path), True)
    return str(out_path)


def make_extractor():
    from radiomics import featureextractor

    extractor = featureextractor.RadiomicsFeatureExtractor(binCount=32, normalize=False)
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()
    for cls in ("shape", "firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm"):
        extractor.enableFeatureClassByName(cls)
    return extractor


def build_wt_mask(seg_path: Path):
    import SimpleITK as sitk

    seg = sitk.ReadImage(str(seg_path))
    arr = sitk.GetArrayFromImage(seg)
    binary = np.isin(arr, [1, 2, 4]).astype(np.uint8)
    wt = sitk.GetImageFromArray(binary)
    wt.CopyInformation(seg)
    return wt


def main() -> int:
    import radiomics
    import SimpleITK as sitk

    print(f"pyradiomics version: {radiomics.__version__}")
    print(f"SimpleITK version: {sitk.Version()}")

    patients = [l.strip() for l in PILOT_LIST.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Pilot hasta sayisi: {len(patients)} -> {patients}")

    t0 = time.time()

    # ---- Pass 1: N4 ----
    resolved = []
    n4_paths = []
    for pid in patients:
        folder = DATA_ROOT / f"{pid}_nifti"
        t1c = _require_nifti(folder / f"{pid}_T1c.nii.gz")
        seg = _require_nifti(folder / f"{pid}_tumor_segmentation.nii.gz")

        geom = validate_image_mask_geometry(t1c, seg)
        if not geom["geometry_match"]:
            print(f"{pid}: GEOMETRI UYUSMUYOR -- ATLANDI")
            continue

        t_n4_start = time.time()
        n4_path = apply_n4_bias_correction(str(t1c))
        elapsed_n4 = time.time() - t_n4_start
        print(f"{pid}: N4 tamam ({elapsed_n4:.1f}s) -> {n4_path}")
        n4_paths.append(n4_path)
        resolved.append({"patient_id": pid, "n4_path": n4_path, "seg_path": seg})

    # ---- Pass 2: pilot-local T1ce-ozel Z-score fit (n=5, GUVENILIR DEGIL) ----
    stats = fit_ucsf_zscore_local(n4_paths)
    print(f"\n[PILOT Z-SCORE FIT -- n={stats['image_count']} GUVENILIR DEGIL] "
          f"mean={stats['mean']:.4f} std={stats['std']:.4f} voxel_count={stats['voxel_count']}")

    # ---- Pass 3: Z-score normalize + C32 extraction (WT bolgesi) ----
    extractor = make_extractor()
    print("\nExtractor settings:", json.dumps(extractor.settings, default=str))
    print("Enabled image types:", list(extractor.enabledImagetypes.keys()))
    print("Enabled feature classes:", list(extractor.enabledFeatures.keys()))

    rows = []
    zscore_dir = SCRATCH / "ucsf_pilot_zscore"
    for item in resolved:
        pid = item["patient_id"]
        zscore_path = apply_ucsf_zscore_local(item["n4_path"], stats, zscore_dir)

        wt_mask = build_wt_mask(item["seg_path"])
        try:
            result = extractor.execute(zscore_path, wt_mask, label=1)
        except Exception as exc:  # noqa: BLE001
            print(f"{pid}: HATA -- {type(exc).__name__}: {exc}")
            continue

        feature_keys = [k for k in result if k.startswith("original_")]
        shape_n = sum(1 for k in feature_keys if k.startswith("original_shape_"))
        fo_n = sum(1 for k in feature_keys if k.startswith("original_firstorder_"))
        tex_n = sum(
            1
            for k in feature_keys
            if k.startswith(("original_glcm_", "original_glrlm_", "original_glszm_", "original_ngtdm_", "original_gldm_"))
        )
        total = shape_n + fo_n + tex_n
        print(f"{pid}: shape={shape_n} firstorder={fo_n} texture={tex_n} TOPLAM={total} "
              f"(beklenen 14+18+75=107)")

        row = {"patient_id": pid, "region": "WT"}
        for k in feature_keys:
            row[k] = float(result[k])
        rows.append(row)

    elapsed_total = time.time() - t0
    print(f"\nToplam pilot süresi: {elapsed_total:.1f}s ({len(patients)} hasta)")
    print(f"Hasta basina ortalama: {elapsed_total/len(patients):.1f}s")

    # CSV yaz -- SADECE scratchpad
    if rows:
        import csv

        out_csv = SCRATCH / "ucsf_c32_pilot_features.csv"
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV yazildi (scratchpad): {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
