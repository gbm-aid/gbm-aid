# GBM-AID

Glioblastoma hastaları için MR görüntülerinden çıkarılan radyomik özellikler ve
klinik değişkenlerle **sağkalım riski sıralayan** bir klinik karar destek
prototipi.

> ⚠️ **Araştırma prototipidir; klinik kullanım için değildir.** Model mutlak
> sağkalım olasılığı değil, hastalar arası bir **sıralama** verir. Ayrıntılı
> metodoloji, sınırlar ve zorunlu beyanlar proje raporundadır.

## Klasör yapısı

```text
backend/          Sunucu tarafının tamamı
  api/            FastAPI uç noktaları (predict, similar, analyze_patient, …)
  pipeline/       Cox PHM, XGBoost, harmonizasyon, segmentasyon, RAG modülleri
  db/             SQL taslakları (VIEW / migration)
  models/         Eğitilmiş model artefaktları (.pkl)
  db_connection.py
frontend/         Saf HTML/CSS/vanilla JS arayüz (derleme adımı YOK, CDN YOK)
tools/            Eğitim, çıkarım, doğrulama ve bakım script'leri
tests/            pytest paketi
demo/             Streamlit demosu
docs/             Teknik notlar
```

`api/` ve `pipeline/` **`backend/` altına taşınmıştır (2026-09-16)**, ancak
**import adları değişmemiştir** — kod içinde hâlâ `from pipeline.x import y` ve
`api.main:app` kullanılır. Bunun için `backend` dizini arama yoluna eklenir:
`pytest.ini` içindeki `pythonpath`, `tools/` script'lerindeki `sys.path` satırı
ve `tools/run_site.py`'nin `PYTHONPATH` ortam değişkeni bunu yapar.

`artifacts/` (ara çıktılar, ~22 GB) ve `.env` depoya **dâhil değildir**.

## Ortam

Backend + görüntü entegrasyon ortamı Python 3.10.18 üzerinde çalışır.

```powershell
& "$env:USERPROFILE\miniforge3\shell\condabin\conda-hook.ps1"
conda activate gbm-aid-integration
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

⚠️ `requirements-nnunet.txt` ve `requirements-pyradiomics.txt` **ayrı sanal
ortamlar içindir**, `requirements.txt` ile aynı ortama kurulmaz — `torch` pinleri
bilinçli olarak farklıdır (`requirements.txt` 2.13.0, `requirements-nnunet.txt`
2.6.0).

## Siteyi ve API'yi çalıştırma

```powershell
python tools/run_site.py --host 127.0.0.1 --port 8000 --warmup UCSF-PDGM-167
```

Arayüz `http://127.0.0.1:8000/app/` adresinde; `frontend/` dizini FastAPI
tarafından `/app` altına bağlanır (ayrı bir web sunucusu gerekmez, CORS yoktur).

## Değişmez görüntü akışı

```text
N4 -> kaynak-bazlı Z-score -> hazır maske -> PyRadiomics (C32)
```

- PyRadiomics çıkarımı **C32 sözleşmesiyle** yapılır: N4 uygulanır, T1ce-özel
  kaynak Z-score uygulanır, ayrıklaştırma `binCount=32` iledir (`binWidth`
  kullanılmaz) ve extractor parametreleri kodda açıkça yazılır.
- **ComBat üretim zincirinde uygulanmaz.** Kod tabanında `fit/apply`
  fonksiyonları ve sızıntı guard'ları bulunur, ancak servis yolunun hiçbir
  adımı bunları çağırmaz; `combat_parameters` tablosu bilinçli olarak boştur.
  *(Eski README bu satırı `… -> ComBat` diye bitiriyordu — geçersizdir.)*
- Dört kaynağın **dördünde de** PyRadiomics C32 çıkarımı koşulmuştur
  (`tools/run_pyradiomics_{upenn,lumiere,tcga,ucsf}.py`). *(Eski README
  "UPenn'de PyRadiomics çalıştırılmaz" diyordu — bu, C32 öncesi CaPTk
  verisinin kullanıldığı döneme aitti, geçersizdir.)*
- TCGA/LUMIERE hazır maskeleri kullanılır; nnU-Net yalnız yeni-hasta demo
  fallback'idir, fine-tune yapılmaz.
- LUMIERE görüntü ve maskeleri radyomik/hacim hesabından önce ortak 1 mm
  grid'e resample edilir.

## Temel doğrulamalar

```powershell
python -m tools.validate_mask_pair <image.nii.gz> <mask.nii.gz>
python -m tools.resample_lumiere_pair <image.nii.gz> <mask.nii.gz>
python -m tools.n4_qc <image.nii.gz> --preview artifacts/n4_preview.png
python -m tools.fit_source_zscore TCGA <n4-image-1> <n4-image-2>
python -m tools.zscore_qc <n4-image> TCGA
python -m tools.lumiere_resampling_regression
python tools/data_integrity_check.py --strict-expiry --verbose
```

Hazır maske çözümleyici DB'deki göreli yolu `GBMAID_NAS_ROOT` altında açar:

- TCGA: `whole` hazır maskesi; yoksa `core` açık uyarıyla
- UPenn: expert hazır maske; yoksa veri setinin `automated_approx` maskesi
- LUMIERE: kayıtlı görüntünün kendi klasöründeki `segmentation`

Üç kaynakta da tam `size/spacing/origin/direction` eşleşmesi zorunludur.

## Yeni-hasta demo

nnU-Net yalnız yeni-hasta demosunda ve varsayılan olarak kapalıdır
(`GBMAID_ENABLE_NEW_PATIENT_NNUNET=0`):

```powershell
python -m pip install -r requirements-nnunet.txt `
  --index-url https://download.pytorch.org/whl/cu126
python -m tools.install_brats_pretrained_model
python -m tools.nnunet_demo_preflight
```

Ayrıntılar: `docs/nnunet_new_patient_demo.md` ·
`docs/mert_week2_start_report.md`

## Yapılandırma

Bağlantı bilgileri ve anahtarlar **proje kökündeki `.env`** dosyasından okunur;
bu dosya depoya dâhil değildir. Gerekli değişkenlerin listesi için
`.env.example` dosyasına bakın.
