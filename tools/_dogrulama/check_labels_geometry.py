"""UCSF-PDGM etiket semasi + geometri dogrulama (SALT-OKUNUR).

Hicbir dosyaya yazmiyor (stdout haric). raw/ veya artifacts/'a DOKUNMAZ.
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

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(str(_VERI_KOKU / r'PKG - UCSF-PDGM Version 5\UCSF-PDGM-v5'))
LIST_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else None
N = int(sys.argv[2]) if len(sys.argv) > 2 else 25

with open(LIST_FILE, "r", encoding="utf-8") as fh:
    patients = [line.strip() for line in fh if line.strip()][:N]

print(f"{len(patients)} hasta incelenecek\n")

all_labels_seen = set()
geometry_mismatches = []
spacing_set = set()

for pid in patients:
    folder = ROOT / f"{pid}_nifti"
    t1c = folder / f"{pid}_T1c.nii.gz"
    seg = folder / f"{pid}_tumor_segmentation.nii.gz"
    if not t1c.is_file() or not seg.is_file():
        print(f"{pid}: EKSIK DOSYA (t1c={t1c.is_file()}, seg={seg.is_file()})")
        continue

    seg_img = sitk.ReadImage(str(seg))
    t1c_img = sitk.ReadImage(str(t1c))

    seg_arr = sitk.GetArrayFromImage(seg_img)
    labels, counts = np.unique(seg_arr, return_counts=True)
    label_counts = {int(l): int(c) for l, c in zip(labels, counts)}
    all_labels_seen.update(label_counts.keys())

    # geometri karsilastirma
    same_size = seg_img.GetSize() == t1c_img.GetSize()
    same_spacing = tuple(round(v, 4) for v in seg_img.GetSpacing()) == tuple(
        round(v, 4) for v in t1c_img.GetSpacing()
    )
    same_origin = tuple(round(v, 3) for v in seg_img.GetOrigin()) == tuple(
        round(v, 3) for v in t1c_img.GetOrigin()
    )
    same_direction = tuple(round(v, 4) for v in seg_img.GetDirection()) == tuple(
        round(v, 4) for v in t1c_img.GetDirection()
    )
    geometry_ok = same_size and same_spacing and same_origin and same_direction
    if not geometry_ok:
        geometry_mismatches.append(
            (pid, same_size, same_spacing, same_origin, same_direction)
        )

    spacing_set.add(tuple(round(v, 4) for v in t1c_img.GetSpacing()))

    print(
        f"{pid}: labels={label_counts} size={seg_img.GetSize()} "
        f"spacing={tuple(round(v,3) for v in t1c_img.GetSpacing())} "
        f"geom_match={geometry_ok}"
    )

print("\n==== OZET ====")
print(f"Goruelen TUM etiket degerleri (butun hastalar): {sorted(all_labels_seen)}")
print(f"Goruelen spacing degerleri: {spacing_set}")
print(f"Geometri UYUSMAYAN hasta sayisi: {len(geometry_mismatches)}")
for row in geometry_mismatches:
    print("  ", row)
