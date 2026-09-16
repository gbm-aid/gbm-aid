# GBM-AID

## Ortam

Backend + görüntü entegrasyon ortamı Python 3.10.18 üzerinde çalışır.

```powershell
& "$env:USERPROFILE\miniforge3\shell\condabin\conda-hook.ps1"
conda activate gbm-aid-integration
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Değişmez görüntü akışı

```text
N4 -> kaynak-bazlı Z-score -> hazır maske -> PyRadiomics -> ComBat
```

- ComBat görüntüye değil radyomik özelliklere uygulanır.
- UPenn radyomikleri hazır CaPTk verisidir; UPenn'de PyRadiomics çalıştırılmaz.
- TCGA/LUMIERE hazır maskeleri kullanılır.
- nnU-Net yalnız yeni-hasta demo fallback'idir; fine-tune yapılmaz.
- LUMIERE görüntü ve maskeleri radyomik/hacim hesabından önce ortak 1 mm grid'e
  resample edilir.

## Temel doğrulamalar

```powershell
python -m tools.validate_mask_pair <image.nii.gz> <mask.nii.gz>
python -m tools.resample_lumiere_pair <image.nii.gz> <mask.nii.gz>
python -m tools.n4_qc <image.nii.gz> --preview artifacts/n4_preview.png
python -m tools.fit_source_zscore TCGA <n4-image-1> <n4-image-2>
python -m tools.zscore_qc <n4-image> TCGA
python -m tools.lumiere_resampling_regression
```

Hazır maske çözümleyici DB'deki göreli yolu `GBMAID_NAS_ROOT` altında açar:

- TCGA: `whole` hazır maskesi; yoksa `core` açık uyarıyla
- UPenn: expert hazır maske; yoksa veri setinin `automated_approx` maskesi
- LUMIERE: kayıtlı görüntünün kendi klasöründeki `segmentation`

Üç kaynakta da tam `size/spacing/origin/direction` eşleşmesi zorunludur.

## Yeni-hasta demo

nnU-Net yalnız yeni-hasta demosunda ve varsayılan olarak kapalıdır:

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 `
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install nnunetv2==2.8.1
python -m tools.install_brats_pretrained_model
python -m tools.nnunet_demo_preflight
```

Ayrıntılar: `docs/nnunet_new_patient_demo.md`.

Ayrıntılı durum: `docs/mert_week2_start_report.md`.
