
# 2026-09-16: sabit kodlanmis mutlak yollar KALDIRILDI -- kullanici dizini
# adi halka acik depoya siziyordu ve bu script baska makinede calismiyordu.
# Yollar artik dosyanin kendi konumundan / ortam degiskeninden turetilir.
import os as _os
from pathlib import Path as _P
_BURASI   = _P(__file__).resolve().parent          # tools/_dogrulama
_KOK_KOD  = _P(__file__).resolve().parents[2]      # gbm-aid mert
_KOK_PROJ = _P(__file__).resolve().parents[3]      # GBM-AID Prototip
_VERI_KOKU = _P(_os.environ.get('GBMAID_VERI_KOKU', str(_KOK_PROJ)))
import SimpleITK as sitk
import numpy as np
from pathlib import Path

folder = Path(str(_VERI_KOKU / r'PKG - UCSF-PDGM Version 5\UCSF-PDGM-v5\UCSF-PDGM-0115_nifti'))
for name in ["UCSF-PDGM-0115_T1c.nii.gz", "UCSF-PDGM-0115_T1c_bias.nii.gz"]:
    p = folder / name
    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    fg = arr[arr > 0]
    print(name, "dtype=", img.GetPixelIDTypeAsString(), "size=", img.GetSize(),
          "min/max=", arr.min(), arr.max(), "fg_mean=%.2f" % fg.mean(), "fg_std=%.2f" % fg.std())
