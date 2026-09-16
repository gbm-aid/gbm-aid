import SimpleITK as sitk
import numpy as np
from pathlib import Path

folder = Path(r"Y:\PKG - UCSF-PDGM Version 5\UCSF-PDGM-v5\UCSF-PDGM-0115_nifti")
for name in ["UCSF-PDGM-0115_T1c.nii.gz", "UCSF-PDGM-0115_T1c_bias.nii.gz"]:
    p = folder / name
    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    fg = arr[arr > 0]
    print(name, "dtype=", img.GetPixelIDTypeAsString(), "size=", img.GetSize(),
          "min/max=", arr.min(), arr.max(), "fg_mean=%.2f" % fg.mean(), "fg_std=%.2f" % fg.std())
