# Yeni Hasta Demo — nnU-Net v2 Kapısı

Bu akış yalnız hazır maskesi olmayan yeni-hasta demosu içindir.
TCGA, UPenn ve LUMIERE taramalarında çalıştırılmaz; fine-tune yapılmaz.

## Sabit ortam

- Python: `3.10.18`
- GPU: NVIDIA RTX 4060 Laptop, 8 GB
- NVIDIA sürücüsü: `566.14`
- PyTorch: `2.6.0+cu126`
- torchvision: `0.21.0+cu126`
- nnU-Net v2: `2.8.1`

PyTorch, nnU-Net'ten önce donanıma uygun resmi wheel ile kurulmalıdır:

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 `
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install nnunetv2==2.8.1
python -m tools.install_brats_pretrained_model
python -m tools.nnunet_demo_preflight
```

## Model

BraTS 2021 ile eğitildiği belirtilen nnU-Net v2 modeli:

- Zenodo DOI: `10.5281/zenodo.11582627`
- Dosya: `Dataset002_BRATS19.zip`
- Boyut: yaklaşık 1.2 GB
- MD5: `23a3f55dead4a6642271a08d1a503bbb`

Kayıt adı `BRATS19`, açıklaması BraTS 2021'dir. Bu tutarsızlık model
`dataset.json` içinden kanal/etiket sözleşmesi doğrulanmadan gizlenmemelidir.
Model üçüncü tarafça yayımlanmıştır; MIC-DKFZ'nin resmi BraTS 2020 ağırlıkları
nnU-Net v1 formatındadır ve v2'ye doğrudan kurulamaz.

## Kanal ve güvenlik sözleşmesi

Girdi modaliteleri aynı fiziksel grid üzerinde olmalıdır:

1. `_0000`: T1
2. `_0001`: T1ce
3. `_0002`: T2
4. `_0003`: FLAIR

`pipeline/new_patient_segmentation.py` şu kapıları zorunlu tutar:

- scope tam olarak `new_patient_demo`
- `GBMAID_ENABLE_NEW_PATIENT_NNUNET=1`
- dört modalite ve tam geometri eşleşmesi
- kurulu pretrained model
- çıktı etiketleri yalnız BraTS `0,1,2,4`
- `fine_tuned=False`, uydurma confidence skoru yok

Kurulu modelin `dataset.json` dosyasında `3=empty`, `4=enhancing` yazmaktadır.
Bu nedenle `3` geçerli tümör çıktısı sayılmaz; görülürse koşucu işlemi durdurur.

DSC `>= 0.85` üretim kabul hedefidir. Etiketli yeni-hasta demo validasyon vakası
olmadan bu hedef sağlandı olarak raporlanamaz.
