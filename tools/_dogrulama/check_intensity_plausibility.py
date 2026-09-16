"""Etiket-doku yon dogrulamasi: T1c yogunlugu ile biyolojik mantik kontrolu.

Mantik: label=4 (varsayilan ET/enhancing) T1c'de PARLAK (yuksek sinyal)
olmali; label=1 (varsayilan NCR/necrotic) T1c'de SONIK/DUSUK sinyal
olmali (enhancing rimin icinde nekrotik, kontrast tutmayan doku).
Eger bu ters cikarsa (label1 parlak, label4 sonuk) LUMIERE tipi bir
etiket-takasi supheden guclenir.

SALT-OKUNUR, hicbir sey yazmiyor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(r"Y:\PKG - UCSF-PDGM Version 5\UCSF-PDGM-v5")
LIST_FILE = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10

with open(LIST_FILE, "r", encoding="utf-8") as fh:
    patients = [line.strip() for line in fh if line.strip()][:N]

for pid in patients:
    folder = ROOT / f"{pid}_nifti"
    t1c = folder / f"{pid}_T1c.nii.gz"
    seg = folder / f"{pid}_tumor_segmentation.nii.gz"

    t1c_img = sitk.ReadImage(str(t1c))
    seg_img = sitk.ReadImage(str(seg))
    t1c_arr = sitk.GetArrayFromImage(t1c_img).astype(np.float64)
    seg_arr = sitk.GetArrayFromImage(seg_img)

    stats = {}
    for lbl in (1, 2, 4):
        mask = seg_arr == lbl
        if mask.sum() == 0:
            stats[lbl] = None
            continue
        vals = t1c_arr[mask]
        stats[lbl] = (float(vals.mean()), float(np.median(vals)), int(mask.sum()))

    # whole brain (label0 excluded) baseline for context
    brain_mask = seg_arr >= 0
    brain_mean = float(t1c_arr[t1c_arr > 0].mean())

    l1 = stats.get(1)
    l4 = stats.get(4)
    verdict = "N/A"
    if l1 and l4:
        verdict = "BEKLENEN (label4>label1)" if l4[0] > l1[0] else "TERS/SUPHELI (label1>=label4)"

    print(
        f"{pid}: label1(NCR?)mean={l1[0]:.1f} label2(ED?)mean={stats[2][0] if stats[2] else None:.1f} "
        f"label4(ET?)mean={l4[0]:.1f} brain_mean(pozitif vox)={brain_mean:.1f} -> {verdict}"
    )
