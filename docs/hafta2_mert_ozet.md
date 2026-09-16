# GBM-AID — Mert Hafta 1 + Hafta 2 Entegrasyon Raporu

Rapor tarihi: 25.07.2026  
Hedef okuyucu: Barış (Veri & DB Lead) ve GBM-AID teknik ekibi

## 1. Yönetici Özeti

Hafta 1'de NAS/Supabase erişimi, üç kaynak için veri envanteri, geometri QC'si ve 107 özellik sözleşmesi taslağı oluşturuldu. Hafta 2'de SimpleITK N4, kaynak-bazlı Z-score, LUMIERE 1 mm resampling, hazır maske çözümleme ve yalnız yeni-hasta demosuna açık nnU-Net v2 güvenlik kapısı kodlandı; son test koşusu 19/19 PASS verdi. Buna karşılık üretim PyRadiomics çalışması henüz başlamamalıdır: kanonik 75 doku özelliği listesi, UPenn 144→107 eşlemesi, LUMIERE ana maske seçimi ve üretim Z-score istatistikleri açık kapılardır.

## 2. Tamamlanan İşler (kod/veriden kanıtlanabilir olanlar)

### 2.1 Git geçmişi ve teslim durumu

`git log --oneline --all` içindeki tüm mevcut commitler Sultan'a aittir:

| Tarih (Europe/Istanbul) | Commit | Değişiklik |
|---|---|---|
| 16.07.2026 17:01 | `827b28d` | Proje dizinleri ve `.gitignore` |
| 24.07.2026 08:24 | `de8fb5d` | `.env.example` |
| 24.07.2026 09:10 | `bf045c5` | FastAPI iskeleti, health endpoint, requirements |
| 24.07.2026 09:26 | `2ad9f17` | Hasta durum endpoint'i ve ilk stub modüller |
| 24.07.2026 09:29 | `a4b7114` | Harmonizasyon endpoint'i ve hata yönetimi |
| 24.07.2026 09:33 | `c51e3e6` | PR #3 merge |
| 24.07.2026 12:15 | `1539017` | Endpoint docstring düzeltmesi |
| 24.07.2026 12:15 | `211aab9` | Feature branch merge |

Mert'in aşağıda raporlanan Hafta 1/2 değişiklikleri `codex/n4-zscore-resampling` çalışma ağacında henüz commit edilmemiştir. Dolayısıyla Git geçmişi bu değişikliklerin gün bazında kaynağını kanıtlamaz; 24–25 Temmuz tarihleri rapor ve artefakt zamanlarından gelir. Barış'ın entegrasyonu için önce kontrollü commit/PR gereklidir.

### 2.2 Hafta 1 altyapı ve veri envanteri

Kanıt: `C:\Users\merte\Documents\onkoloji\reports\week1\summary.json`, ilgili CSV/JSON raporları ve `docs/mert_week1_final_report.md`.

- Pi5/Tailscale/SMB erişimi PASS; gerçek bir TCGA NIfTI dosyası ağ üzerinden nibabel ve SimpleITK ile okundu.
- TCGA: 49 hasta klasörü; 39 T1c, 48 whole mask, 49 core mask, 36 tam dört-modalite vaka.
- UPenn: 630 hasta, 671 yapısal tarama, 611 pre-op ve 60 post-op tarama; 611 hazır automated ve 147 expert maske.
- UPenn Supabase: 10.032 `feature_source='precomputed'` satırı doğrulandı. Her satır mevcut DB yapısında 21 shape + 81 first-order + 42 texture = 144 özellik taşıyor.
- LUMIERE: 91 hasta, 638 benzersiz MR ziyareti ve 616 uzman değerlendirmeli ziyaret bulundu. 91 hastanın tamamında ziyaretler veya sekanslar arasında spacing değişimi görüldü.
- İlk geometri örneklemi: 10 görüntü/maske çifti kontrol edildi; 9 PASS, `Patient-001/week-056` FAIL.
- Notebook üretilmedi; Hafta 1/2 işi `.py`, `.md`, `.yaml`, `.csv` ve `.json` dosyalarıyla yürütüldü.

Hafta 1 ana araçları:

- `tools/week1_inventory.py`, `tools/week1_full_qc.py`
- `tools/inspect_upenn_radiomics_db.py`, `tools/check_upenn_radiomics.py`
- `tools/validate_radiomics_contract.py`
- `config/pyradiomics_107.yaml`
- `reports/week1/*.csv`, `reports/week1/*.json`

### 2.3 N4 Bias Field Correction

Uygulama: `pipeline/harmonization.py::apply_n4_bias_correction`.

Varsayılan parametreler:

| Parametre | Değer |
|---|---|
| Kütüphane/filtre | SimpleITK 2.3.1 / `N4BiasFieldCorrectionImageFilter` |
| Girdi tipi | `float32` |
| Foreground maskesi | Otsu; dış/ROI değerleri `0/1`, 200 histogram bin |
| Multi-resolution iterasyonları | `50,50,30,20` |
| Convergence threshold | `1e-7` |
| Shrink factor | `4` |
| Tam çözünürlük çıktısı | Shrunken grid'de fit edilen log bias field tam görüntüde uygulanıyor |
| Geometri | Size/spacing/origin/direction korunuyor |
| Yazım | Ham dosya değiştirilmiyor; türetilmiş çıktı atomik yazılıyor |

Gerçek veri üzerinde üç smoke vaka çalıştırıldı:

| Kaynak | Hasta/zaman | Modalite | N4 çıktı | Sonuç |
|---|---|---|---|---|
| TCGA | `TCGA-02-0003` | T1c | `artifacts/week2/n4/f20400188383/T1c_n4.nii.gz` | Geometri korundu, PASS |
| UPenn | `UPENN-GBM-00001_11` | T1GD | `artifacts/week2/n4/710cb4351006/UPENN-GBM-00001_11_T1GD_n4.nii.gz` | Geometri korundu, PASS |
| LUMIERE | `Patient-001/week-056` | CT1 | `artifacts/week2/n4/69c8a181c18e/CT1_n4.nii.gz` | Geometri korundu, PASS |

Bunlar tüm kohort batch sonucu değildir. N4 için ayrıca sentetik geometri-koruma birim testi vardır.

### 2.4 Kaynak-bazlı Z-score normalizasyonu

Uygulama:

- `fit_source_zscore_statistics`: N4 sonrası sonlu ve sıfır-dışı foreground voxel'lerinde streaming birleşik ortalama/varyans fit eder.
- `apply_zscore_normalization`: yalnız önceden fit edilmiş, kaynak anahtarlı JSON istatistiğini kullanır; eksik istatistikte işlemi durdurur.
- Kaynaklar ayrı anahtarlardır: `TCGA`, `UPenn`, `LUMIERE`.

Diskteki `artifacts/week2/zscore/smoke_source_stats.json` her kaynak için yalnız bir N4 görüntüsüyle üretilmiş smoke istatistiğidir:

| Kaynak | Fit görüntü sayısı | Foreground voxel | Fit μ | Fit σ | Normalize çıktı μ | Normalize çıktı σ |
|---|---:|---:|---:|---:|---:|---:|
| TCGA | 1 | 6.142.663 | 374,168365 | 335,896539 | 0,0000000024 | 0,9999999888 |
| UPenn | 1 | 1.439.414 | 495,150068 | 144,828728 | 0,0000000908 | 0,9999999639 |
| LUMIERE | 1 | 248.018 | 386,565696 | 74,049333 | -0,0000001092 | 1,0000000206 |

Bu sonuç kaynakların birbirine karıştırılmadığını ve uygulama kodunun `μ≈0, σ≈1` ürettiğini doğrular. Aynı tek görüntü hem fit hem apply için kullanıldığı için bu değerler üretim kohort istatistiği veya bağımsız doğrulama değildir.

### 2.5 LUMIERE resampling ve longitudinal regresyon

Uygulama: `pipeline/resampling.py`; görüntü lineer, maske nearest-neighbor ile görüntünün fiziksel uzayında oluşturulan 1 mm izotropik grid'e taşınıyor. Size/spacing/origin/direction, etiket kümesi ve fiziksel foreground hacmi kontrol ediliyor.

Kanıt: `artifacts/lumiere_resampling_regression/Patient-001_qc.json` ve `run.stdout.log`.

| Zaman noktası | Girdi geometri | Çıktı geometri | Hacim sapması | Sonuç |
|---|---:|---:|---:|---:|
| `week-000-1` | PASS | PASS | %0,0000 | PASS |
| `week-000-2` | PASS | PASS | %0,0068 | PASS |
| `week-044` | PASS | PASS | %0,4745 | PASS |
| `week-056` | FAIL | PASS | %0,0641 | PASS |

`week-056` çıktısı `230×230×139 @ 1×1×1 mm`; maske etiketleri `0,1,3` korundu ve fiziksel hacim `5661,63→5658,00 mm³` oldu. Bu regresyon, curve_fit/simülasyon öncesi zorunlu resampling mekanizmasının temsilci bir hastanın dört ziyaretinde çalıştığını gösterir; tüm LUMIERE kohortu henüz batch edilmedi.

### 2.6 TCGA-02-0003/0006 maske doğrulaması

Kanıt: `C:\Users\merte\Documents\onkoloji\reports\week1\geometry_qc.csv` ve `tcga_inventory.csv`.

| Hasta | T1c/WT size | Spacing | Orientation | Size/spacing/origin/direction | Sonuç |
|---|---|---|---|---|---|
| `TCGA-02-0003` | `200×200×162` | `1×1×1 mm` | LPS/LPS | Dördü de eşit | PASS |
| `TCGA-02-0006` | `200×200×150` | `1×1×1 mm` | LPS/LPS | Dördü de eşit | PASS |

`TCGA-02-0006` T2 içermiyor. Bu eksiklik T1c+WT radyomik geometrisini bozmaz; ancak dört-modalite isteyen API/yeni-hasta benzeri bir akışta vaka eksik modalite nedeniyle bloke edilmelidir.

### 2.7 Hazır maske çözümleme

Uygulama: `pipeline/segmentation.py` ve `pipeline/harmonization.py::get_segmentation_mask`.

- TCGA/UPenn/LUMIERE için nnU-Net fallback yok.
- Supabase'deki göreli NIfTI yolları `.env` içindeki NAS kökü altında çözülüyor; `..` path traversal reddediliyor.
- Görüntü/maske fiziksel geometrisi eşleşmezse PyRadiomics'e geçilmiyor.
- Canlı tekil resolver kontrolünde TCGA scan 2, UPenn scan 155 ve LUMIERE scan 2854 için geometri PASS görüldü; bu sonuç ayrı bir kalıcı JSON/log dosyasına yazılmadı.

### 2.8 nnU-Net v2 kurulum durumu

Kapsam yalnız `new_patient_demo`; fine-tune kapalıdır. TCGA/UPenn/LUMIERE hazır maskeleri yerine nnU-Net çalıştırılmaz.

25.07.2026 preflight sonucu:

| Bileşen | Durum |
|---|---|
| Python | 3.10.18 |
| PyTorch | `2.6.0+cu126` |
| torchvision | `0.21.0+cu126` |
| CUDA/GPU | CUDA 12.6, NVIDIA GeForce RTX 4060 Laptop GPU, PASS |
| nnU-Net v2 | `2.8.1`, predictor executable bulundu |
| BraTS arşivi | `Dataset002_BRATS19.zip`, 1.155.915.349 byte |
| MD5 | `23a3f55dead4a6642271a08d1a503bbb`, PASS |
| Model | `Dataset002_BRATS19`, `3d_fullres`, 5 fold/checkpoint |
| Kanal sırası | T1, T1ce, T2, FLAIR, PASS |
| İzinli çıktı etiketleri | `0,1,2,4`; metadata'daki `3=empty` reddediliyor |
| Demo gate | Varsayılan kapalı (`GBMAID_ENABLE_NEW_PATIENT_NNUNET=0`) |

Pretrained ağırlıklar indirilmiş ve kurulmuştur. Gerçek inference yapılmadı, etiketli yeni-hasta demo vakası yok ve DSC ölçülmedi; bu nedenle `DSC≥0,85` hedefi sağlandı denemez.

### 2.9 Testler ve paket sürümleri

25.07.2026 yeniden doğrulama:

- Python 3.10.18 entegrasyon ortamında `pytest`: **19/19 PASS**.
- Python 3.9.23 görüntü ve Python 3.10.18 entegrasyon ortamlarında `pip check`: PASS.
- Test kapsamı: N4 geometri, kaynak doğrulama, kaynak-bazlı fit/apply, eksik stats kapısı, LUMIERE resampling, API kaynak iletimi, hazır maske seçimi/geometri, path traversal, yeni-hasta kapsam/gate/kanal/geometri ve nnU-Net etiket sözleşmesi.

Paket durumu:

| Ortam | Kritik sürümler |
|---|---|
| Hafta 1 görüntü ortamı, Python 3.9.23 | NumPy 1.26.4; SciPy 1.11.4; pandas 2.2.2; SimpleITK 2.3.1; nibabel 5.2.1; PyRadiomics 3.1.0; PyYAML 6.0.2; psycopg2-binary 2.9.12; python-dotenv 1.0.1 |
| Hafta 2 entegrasyon ortamı, Python 3.10.18 | NumPy 1.26.4; SimpleITK 2.3.1; FastAPI 0.136.3; Uvicorn 0.41.0; PyYAML 6.0.3; psycopg2-binary 2.9.12; python-dotenv 1.0.1; pytest 8.2.2; torch 2.6.0+cu126; torchvision 0.21.0+cu126; nnunetv2 2.8.1 |

`requirements.txt`, `requirements-dev.txt` ve `requirements-nnunet.txt` entegrasyon ortamını pinliyor. PyRadiomics/nibabel/pandas/SciPy şu anda ana repo requirements zincirinde yoktur; PyRadiomics yalnız Hafta 1 görüntü ortamında doğrulanmıştır.

### 2.10 Üretilen kod dosyalarının entegrasyon envanteri

Hafta 2 repo dosyaları:

- DB/API: `db_connection.py`, `api/main.py`
- Pipeline: `pipeline/harmonization.py`, `pipeline/resampling.py`, `pipeline/segmentation.py`, `pipeline/new_patient_segmentation.py`, `pipeline/nnunet_runtime.py`
- CLI: `tools/n4_qc.py`, `tools/fit_source_zscore.py`, `tools/zscore_qc.py`, `tools/resample_lumiere_pair.py`, `tools/lumiere_resampling_regression.py`, `tools/validate_mask_pair.py`, `tools/mask_overlay.py`, `tools/install_brats_pretrained_model.py`, `tools/nnunet_demo_preflight.py`
- Test: `tests/test_api_harmonize.py`, `tests/test_harmonization.py`, `tests/test_resampling.py`, `tests/test_segmentation.py`, `tests/test_new_patient_segmentation.py`, `tests/test_nnunet_output_labels.py`, `tests/test_install_brats_model.py`
- Ortam/doküman: `.python-version`, `requirements*.txt`, `docs/mert_week2_start_report.md`, `docs/nnunet_new_patient_demo.md`, `README.md`

Hafta 1 çalışma alanı dosyaları:

- `tools/week1_inventory.py`, `tools/week1_full_qc.py`
- `tools/check_upenn_radiomics.py`, `tools/inspect_upenn_radiomics_db.py`
- `tools/validate_radiomics_contract.py`
- `config/pyradiomics_107.yaml`
- `db_connection.py`, `environment-mert.yml`
- `docs/environment_versions.md`, `docs/mert_week1_plan.md`, `docs/week1_findings.md`, `docs/radiomics_contract_v1.md`, `docs/mert_week1_final_report.md`

## 3. Elle Doldurulacak / Görsel Değerlendirme

### 3.1 Diskte bulunan görsel/log/CSV/JSON kanıtı

- N4 öncesi/sonrası montajları:
  - `artifacts/week2/qc/tcga_02_0003_t1c_n4.png`
  - `artifacts/week2/qc/upenn_00001_11_t1gd_n4.png`
  - `artifacts/week2/qc/lumiere_patient001_week056_ct1_n4.png`
- Maske overlay:
  - `artifacts/week2/qc/lumiere_patient001_week056_overlay.png`
  - `artifacts/lumiere_resampling_regression/Patient-001/*/overlay.png` — dört zaman noktası
- Sayısal sonuç:
  - `artifacts/week2/zscore/smoke_source_stats.json`
  - `artifacts/lumiere_resampling_regression/Patient-001_qc.json`
  - `artifacts/lumiere_resampling_regression/run.stdout.log`
  - `run.stderr.log` boş, yani kaydedilmiş stderr hatası yok.
- Hafta 1 CSV/JSON:
  - `C:\Users\merte\Documents\onkoloji\reports\week1\geometry_qc.csv`
  - `tcga_inventory.csv`, `upenn_inventory.csv`, `lumiere_visit_inventory.csv`
  - `upenn_radiomics_compatibility.json`, `summary.json`
- Beş `fold_*/progress.png` kurulu pretrained paketten gelir; Mert'in yerel eğitimi veya DSC kanıtı değildir.
- Repo artefaktları içinde CSV yoktur; Hafta 2 N4 ve canlı resolver çalışmaları için kalıcı makine-okunur QC raporu üretilmemiştir.

### 3.2 N4 öncesi/sonrası görsel karşılaştırma

Diskteki montajlarda TCGA ve UPenn vakalarında düzeltme görsel olarak sınırlı/subtil; LUMIERE `week-056` vakasında daha belirgin global yoğunluk değişimi görülüyor. PNG varlığı algoritmanın çalıştığını gösterir fakat klinik olarak kabul edilebilir doku/lez­yon kontrastı korunumu için nicel uniformity metriği veya bias-field görseli yoktur.

**[MERT DOLDURACAK]** Üç montaj için “kabul/red”, artefakt veya lezyon kontrast kaybı gözlemi ve gerekiyorsa N4 parametre değişikliği.

### 3.3 nnU-Net düşük güven vakası ve uyarı kuralı

Gerçek nnU-Net inference çalıştırılmadığı için düşük güvenli vaka kaydı yoktur. Kod `confidence_score=None` döndürüyor; olasılıktan türetilmiş güven eşiği ve düşük-güven etiketleme kuralı uygulanmış değildir. Mevcut uyarılar yalnız hazır maske seçiminde expert yerine automated maske seçilmesi gibi deterministik durumlardır.

**[MERT DOLDURACAK]** Etiketli yeni-hasta demo vakası seçildikten sonra DSC ölçümü, güven skorunun teknik tanımı/eşiği ve düşük güven uyarı etiketi.

### 3.4 UPenn/LUMIERE hazır maske — `mr_scans`/`followup_series` eşleşmesi

Disk kanıtında üç UPenn örnek görüntü/maske geometrisi PASS ve tek canlı resolver örneği PASS'tir; ancak kohort çapında kalıcı eşleşme raporu yoktur. Canlı Supabase salt-okunur kontrolünde:

- `mr_scans`: UPenn 2.684 satır/630 hasta; LUMIERE 2.396 satır/91 hasta.
- `followup_series`: LUMIERE 616 satırın **616'sında da `scan_id` NULL**.
- UPenn için `followup_series` satırı yok.
- LUMIERE `followup_series.segmentation_tool` alanı da 616/616 satırda NULL.

Bu nedenle UPenn/LUMIERE hazır maskelerinin `mr_scans` ve `followup_series` ile kohort çapında eşleştiği doğrulanamaz.

**[MERT DOLDURACAK]** Barış'ın kalıcı join/mapping'i sonrasında eşleşen, eşleşmeyen, çoklu eşleşen ve geometri FAIL sayıları CSV/JSON olarak kaydedilecek.

## 4. v4.5 Mimariden Sapmalar / Netleştirmeler

### 4.1 Kanonik 75 doku özelliği sözleşmesi — RED

- Yanlış/belirsiz: Hafta 1 `config/pyradiomics_107.yaml`, 75 doku özelliğini `GLCM 24 + GLRLM 16 + GLSZM 16 + NGTDM 5 + GLDM 14` olarak tanımlar.
- Çelişki: Güncel proje otoritesi 107 sayısını `14 shape + 18 first-order + 75 texture` olarak sabitliyor ve texture sınıflarını `GLCM+GLRLM+GLSZM` ile sınırlandırıyor. Mevcut YAML'da bu üç sınıf yalnız 56 eder.
- Somut düzeltme: PyRadiomics batch başlamadan Barış/Nisa/Mert tarafından sürümlü, tam 75 isimli allowlist imzalanmalı; YAML ve UPenn eşleme tablosu aynı listeye göre güncellenmeli.
- Etki: Çözülmeden üretilen TCGA/LUMIERE vektörleri UPenn, ComBat, Cox ve FAISS ile aynı özellik uzayında olmayacaktır.

### 4.2 LUMIERE ana maske seçimi — RED

- Yanlış/belirsiz: `pipeline/segmentation.py` kayıtlı `segmentation.nii.gz` dosyasını `lumiere_registered_hd_glio` olarak ana maske seçiyor.
- Çelişki: Hafta 1 kanonik sözleşmesi ana WT vektörü için DeepBraTumIA birleşimini, HD-GLIO'yu yalnız duyarlılık analizi için tanımlıyor.
- Somut düzeltme: DeepBraTumIA WT maskesinin işlenen görüntü grid'ine dönüşüm kuralı netleştirilmeli ve resolver buna göre değiştirilmelidir; karar değişecekse sözleşme sürümü artırılmalıdır.
- Etki: ROI, hacim eğrisi, 107 özellik, ComBat ve downstream model girdisi değişir.

### 4.3 TCGA WT→core fallback — RED

- Yanlış: Resolver WT yoksa core maskeyi uyarıyla seçiyor.
- Çelişki: Kanonik analiz birimi T1ce+WT'dir; başka ROI'ye fallback aynı özelliği temsil etmez.
- Somut düzeltme: Ana PyRadiomics akışında WT yoksa vaka `not_evaluable` olmalı. Core yalnız ayrı, açık etiketli analiz kolunda kullanılabilir.
- Etki: Özellikle `TCGA-02-0102` için sessiz/yarı-sessiz ROI değişimi ve harici test karşılaştırma hatası önlenir.

### 4.4 Z-score anahtarı netleştirmesi

Hafta 2 kodu güncel talimata uygun olarak yalnız kaynak-bazlı (`TCGA/UPenn/LUMIERE`) istatistik tutuyor. Hafta 1 çalışma sözleşmesinde “kaynak ve modalite bazlı” ifadesi var. Ana vektör yalnız T1ce olacaksa iki yaklaşım aynı pratik sonuca yaklaşır; birden fazla modalite normalize edilecekse istatistik anahtarı `(source, modality)` olmalıdır. Ekip bu ifadeyi tek bir sürümlü sözleşmede sabitlemelidir.

### 4.5 UPenn precomputed özellik uzayı

UPenn'de PyRadiomics çalıştırmama kuralına uyuluyor; ancak mevcut 10.032 satır 144 özellikli, kanonik isimlerle birebir eşleşmiyor ve WT satırı bulunmuyor. `feature_source='precomputed'` karşılaştırılabilirlik kanıtı değildir. Barış'ın ad-bazlı 144→107/ROI eşleme tablosu olmadan UPenn verisi ComBat/Cox/FAISS'e girmemelidir.

### 4.6 LUMIERE metadata/header farkı

`Patient-001/week-044` MRinfo slice thickness değeri 5 mm iken NIfTI header z-spacing yaklaşık 6 mm'dir. Hacim hesabında işlenen NIfTI header otorite alınmalı; fark QC raporuna yazılmalıdır.

### 4.7 Pretrained model adlandırma/provenans

Arşiv/dataset adı `BRATS19`, yayın açıklaması BraTS 2021'dir. Kanal/etiket sözleşmesi kurulu `dataset.json` üzerinden doğrulanmıştır; fakat bu modelin kendi beş `progress.png` dosyası Mert'in DSC kanıtı değildir.

## 5. Barış'a / Ekibe İletilecek Kritik Noktalar

### 5.1 TCGA-02-0003/0006: INSERT değil UPDATE

25.07.2026 canlı Supabase kontrolünde şu dört kayıt zaten vardır:

| Hasta | `radiomics_id` | `scan_id` | Tool | Bölge | Mevcut hacim (mm³) | Feature JSON sayıları |
|---|---:|---:|---|---|---:|---|
| TCGA-02-0003 | 26006 | 2 | `TCGA-ground-truth` | WT | 100.435 | 0/0/0 |
| TCGA-02-0003 | 26007 | 2 | `TCGA-ground-truth` | TC | 22.394 | 0/0/0 |
| TCGA-02-0006 | 26008 | 6 | `TCGA-ground-truth` | WT | 22.342 | 0/0/0 |
| TCGA-02-0006 | 26009 | 6 | `TCGA-ground-truth` | TC | 3.672 | 0/0/0 |

`radiomics` tablosunda `(scan_id, segmentation_tool, tumor_region)` üzerinde UNIQUE constraint vardır. Hafta 3 PyRadiomics çıktısı yeni `INSERT` olmamalı; ilgili `radiomics_id` veya bu üçlü üzerinden mevcut satırların `shape_features`, `first_order_features`, `texture_features` ve kararlaştırılmış türetilmiş alanları transaction içinde `UPDATE` edilmelidir. Mevcut hacimler korunmalı veya yeniden hesaplanacaksa eski/yeni fark QC kaydıyla açıklanmalıdır.

### 5.2 Kalan TCGA maske/görüntü durumu

Güncel Hafta 1 envanteri:

- Kayıtlı hasta: 49.
- T1c+WT radyomik adayı: 39.
- `TCGA-02-0003` ve `TCGA-02-0006` sonrası kalan aday: **37**; “~38” değil.
- Tamamen ham görüntüsüz: 9.
- Yalnız FLAIR bulunan: 1 (`TCGA-02-0046`).
- Böylece T1c+WT PyRadiomics çalıştırılamayan toplam: 10.
- Whole mask: 48; core mask: 49.
- `TCGA-02-0102`: whole mask yok, yalnız core; kanonik WT ana analizine alınmamalı.

Bu sayılar batch öncesi NAS envanteri yeniden çalıştırılarak dondurulmalıdır; dosya eklenmiş/silinmişse `evaluable_n` değişebilir.

### 5.3 Nisa `combat_parameters` bağımlılığı

Canlı DB'de `combat_parameters` tablosu mevcut fakat satır sayısı **0**. `pipeline/harmonization.py::apply_combat_harmonization` kasıtlı olarak `NotImplementedError` döndürüyor; ComBat Nisa'nın feature-space işidir.

Mert tarafından Nisa'nın tablo/algoritma geliştirmesini engelleyen bir kod kilidi yoktur. Ancak gerçek parametre fit'i için gereken ortak 107 özellik matrisi henüz teslim edilemez; kanonik 75 liste, UPenn eşleme/ROI, LUMIERE ana maske ve üretim Z-score/PyRadiomics batch kapıları açıktır. Bu gecikme Mert-Barış-Nisa arasında paylaşılan upstream veri sözleşmesi blokajıdır.

TCGA ComBat uygulamasında TCGA'ya kendi doğrudan hesaplanan γ/δ değerleri verilmemeli; UPenn referans-batch out-of-sample dönüşümü kullanılmalıdır.

### 5.4 Entegrasyon teslim biçimi

- Mert değişiklikleri commit/PR halinde değildir.
- `.env` doğru biçimde Git dışında tutuluyor; raporda kimlik bilgisi yoktur.
- `artifacts/` da `.gitignore` kapsamındadır. Barış yalnız branch'i çekerse N4, Z-score, resampling ve model artefaktlarını alamaz.
- Entegrasyon için kod commit/PR, küçük QC JSON/CSV'leri için kararlaştırılmış paylaşım alanı ve büyük NIfTI/model dosyaları için NAS yolu/manifest gereklidir.

## 6. Riskler / Açık Konular

| Öncelik | Açık konu | Kapanış ölçütü |
|---|---|---|
| RED | 75 doku özellik listesi güncel otoriteyle uyuşmuyor | Sürümlü tam 75 isimli allowlist ve üç kaynakta aynı kolon sırası |
| RED | UPenn 144 özellik/NC-ET-ED yapısı kanonik 107/WT ile uyumsuz | Barış'ın ad ve ROI bazlı, testli eşleme raporu |
| RED | LUMIERE resolver HD-GLIO seçiyor | Ana DeepBraTumIA WT dönüşümü veya onaylı sözleşme değişikliği |
| RED | Üretim Z-score istatistikleri yok | Kaynak başına dondurulmuş N4 manifesti; train/reference üzerinde fit; bağımsız QC |
| RED | `followup_series.scan_id` LUMIERE 616/616 NULL | Tekil ve doğrulanmış `followup_series→mr_scans` eşlemesi |
| RED | Mert kodu commit/PR değil | Temiz diff, test kanıtı, kontrollü commit ve PR |
| YELLOW | nnU-Net DSC ve güven etiketi yok | Etiketli yeni-hasta demo vakasında DSC≥0,85 ve tanımlı uyarı kuralı |
| YELLOW | Ana entegrasyon requirements'ında PyRadiomics yok | Python 3.10.18 ortamında PyRadiomics kurulumu, pin ve smoke test |
| YELLOW | N4 görsel kabul/nicel uniformity metriği yok | Mert görsel kabulü ve kaydedilmiş nicel QC |
| YELLOW | TCGA core fallback kanonik ROI'yi değiştiriyor | Ana akışta fallback'in kaldırılması ve `not_evaluable` durumu |
| YELLOW | Artefaktlar Git tarafından yok sayılıyor | NAS üzerinde sürümlü manifest, checksum ve yeniden üretim komutları |

PyRadiomics henüz TCGA veya LUMIERE batch üzerinde çalıştırılmadı. UPenn'de çalıştırılmaması bilinçli ve mimariye uygundur. Cox PHM/XGBoost eğitimi Mert akışında yapılmadı; TCGA hiçbir koşulda eğitim setine eklenmemelidir.
