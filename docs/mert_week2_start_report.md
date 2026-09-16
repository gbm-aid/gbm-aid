# GBM-AID - Mert Hafta 2 Teknik Raporu

Tarih: 25.07.2026

## Okunan ekip çıktıları

- Barış: `hafta1_final_raporu.pdf`
- Sultan: `Hafta 1 + Hafta 2 Özeti`
- Otorite: `GBM-AID Final Sistem Mimarisi` ve `GBM-AID Çalışma Planı`

## Düzeltilen sözleşme hataları

### 1. ComBat görüntüye uygulanamaz

- Yanlış: API açıklamasındaki `N4 -> Z-score -> ComBat -> segmentasyon`.
- Çelişki: Mimari Bölüm 5.1; ComBat radyomik feature matrisinde çalışır.
- Düzeltme: Görüntü endpoint'i yalnız `N4 -> kaynak-bazlı Z-score` çalıştırır.
  Hazır maske ile PyRadiomics sonrasında Nisa'nın ComBat adımı çağrılacaktır.
- Etki: `mr_scans.harmonization_status='combat'` ifadesi semantik olarak
  sorunludur. `combat` bir görüntü statüsü değildir; feature katmanında ayrı
  izlenmelidir. Bu DB kararı Barış/Nisa/Sultan ile netleştirilmelidir.

### 2. `source="unknown"` kaynak normalizasyonunu bozuyordu

- Yanlış: Sultan endpoint'i her taramaya `source="unknown"` gönderiyordu.
- Çelişki: TCGA/UPenn/LUMIERE ayrı scaler kullanmak zorundadır.
- Düzeltme: `mr_scans -> patients -> dataset_sources` join'i ile gerçek kaynak
  okunuyor ve kanonik `TCGA/UPenn/LUMIERE` değerine çevriliyor.
- Etki: Yanlış scaler kullanımı ve sessiz kurum karışması engellendi.

### 3. Per-image Z-score kaynak-bazlı Z-score değildir

- Yanlış yaklaşım: Her dosyanın kendi ortalama/sapmasını anlık hesaplamak.
- Düzeltme: Kaynak kohortuna ait istatistik önce N4 çıktılarından fit edilip
  sürümlü JSON artefaktına yazılıyor; apply fonksiyonu yalnız bu istatistiği
  kullanıyor. İstatistik yoksa işlem sessizce devam etmiyor.
- Etki: Test/inference sırasında yeniden fit ve veri sızıntısı önleniyor.

### 4. Python 3.14 ortak görüntü ortamı olarak kullanılamaz

- Sultan raporundaki backend venv Python 3.14'tür.
- Mert'in ilk doğrulanmış PyRadiomics/SimpleITK ortamı Python 3.9.23'tür; ancak
  Sultan'ın `annotated-types==0.8.0` pini Python 3.9'u desteklemez.
- Ortak uyum sürümü Python 3.10.18 olarak doğrulandı. Backend pinleri,
  NumPy 1.26.4 ve SimpleITK 2.3.1 aynı çözümlemede geçti; repo
  `.python-version` dosyası 3.10.18 olarak sabitlendi.
- Python 3.14 ortamı kullanılmayacak. Hafta 3 PyRadiomics kurulumu da bu
  entegrasyon ortamında ayrıca doğrulanmalıdır.

## Uygulanan kod

- `pipeline/harmonization.py`
  - SimpleITK N4 gerçek implementasyonu
  - kaynak adı doğrulaması
  - streaming cohort Z-score fit
  - fit edilmiş istatistikle Z-score apply
  - ham dosyayı değiştirmeyen atomik çıktı
- `pipeline/resampling.py`
  - görüntü: lineer interpolasyon
  - maske: nearest-neighbor
  - 1 mm ortak referans grid
  - size/spacing/origin/direction QC
  - etiket ve fiziksel hacim korunumu QC
- `api/main.py`
  - gerçek source join
  - yanlış ComBat/segmentasyon sırası düzeltmesi
  - Z-score istatistiği yoksa `blocked` durumu
- CLI araçları
  - `tools/n4_qc.py`
  - `tools/fit_source_zscore.py`
  - `tools/zscore_qc.py`
  - `tools/resample_lumiere_pair.py`
  - `tools/validate_mask_pair.py`
  - `tools/mask_overlay.py`

## Test sonuçları

### Birim testleri

`pytest`: 7/7 PASS.

- N4 geometri koruma
- bilinmeyen kaynak reddi
- fit edilmemiş kaynak istatistiğini reddetme
- birleşik source dağılımında `mu ~= 0`, `sigma ~= 1`
- 1 mm görüntü-maske resampling, etiket/hacim koruma
- API'nin DB kaynağını Z-score fonksiyonuna iletmesi
- source istatistiği yokken API'nin `blocked/zscore` dönmesi

### Gerçek N4 smoke testleri

| Kaynak | Vaka | Geometri | Sonuç |
|---|---|---|---|
| TCGA | TCGA-02-0003 / T1c | korundu | PASS |
| UPenn | UPENN-GBM-00001_11 / T1GD | korundu | PASS |
| LUMIERE | Patient-001 / week-056 / CT1 | korundu | PASS |

PNG yerleşimi: sol N4 öncesi, sağ N4 sonrası.

### Hazır maske doğrulaması

| Vaka | Görüntü/Maske | Sonuç |
|---|---|---|
| TCGA-02-0003 | T1c + whole | tam geometri eşleşmesi |
| TCGA-02-0006 | T1c + whole | tam geometri eşleşmesi |

TCGA-02-0006'nın T2 modalitesi eksiktir; bu durum T1c+WT maskesinin doğru
olmasını bozmaz. Ancak dört-modaliteli API endpoint'i bu hastaya 422 vermeye
devam etmelidir.

### Kritik LUMIERE regresyon vakası

`Patient-001/week-056`:

- Girdi geometri eşleşmesi: FAIL
- Çıktı grid: `230 x 230 x 139`, `1 x 1 x 1 mm`
- Çıktı geometri eşleşmesi: PASS
- Maske etiketleri: `0, 1, 3` korundu
- Girdi fiziksel foreground hacmi: `5661.63 mm3`
- Çıktı fiziksel foreground hacmi: `5658.00 mm3`
- Hacim sapması: `%0.064`
- Sonuç: PASS

## Hafta 2 devamında tamamlananlar

### Merkezi DB ve NAS yol çözümü

- `db_connection.py` bağlantı bilgisini yalnız proje `.env` dosyasından okur.
- API ve hazır maske sorguları aynı bağlantı modülünü kullanır.
- Supabase'deki göreli POSIX yollar `GBMAID_NAS_ROOT` altında güvenli biçimde
  UNC/yerele çevrilir; `..` geçişi reddedilir.
- `.env` git tarafından yok sayılıyor; `.env.example` yalnız placeholder içerir.

### Canlı hazır maske resolver testi

| Kaynak | Gerçek tarama | Seçilen hazır maske | Geometri |
|---|---|---|---|
| TCGA | TCGA-02-0003 / scan 2 | whole tumor | PASS |
| UPenn | UPENN-GBM-00001 / scan 155 | automated_approx; expert yok uyarısı | PASS |
| LUMIERE | Patient-001 / scan 2854 | registered HD-GLIO segmentation | PASS |

Hiçbir kohortta nnU-Net fallback çalıştırılmaz. Uydurma confidence skoru
üretilmez; `confidence_score=None` döner.

### LUMIERE dört-zaman-noktası regresyonu

| Zaman noktası | Girdi geometri | Çıktı geometri | Hacim sapması | Sonuç |
|---|---:|---:|---:|---:|
| week-000-1 | PASS | PASS | %0,000 | PASS |
| week-000-2 | PASS | PASS | %0,007 | PASS |
| week-044 | PASS | PASS | %0,475 | PASS |
| week-056 | FAIL | PASS | %0,064 | PASS |

Dört çıktının tamamı 1 mm izotropik grid üzerindedir; görüntü lineer, maske
nearest-neighbor ile resample edilmiştir ve etiketler korunmuştur.

### Yeni-hasta nnU-Net demo kapısı

- Python 3.10.18
- PyTorch `2.6.0+cu126`, torchvision `0.21.0+cu126`
- nnU-Net v2 `2.8.1`
- RTX 4060 Laptop GPU CUDA smoke testi: PASS
- BraTS model arşivi MD5: `23a3f55dead4a6642271a08d1a503bbb` — PASS
- Model: `Dataset002_BRATS19`, 5 fold
- Kanal sözleşmesi: T1, T1ce, T2, FLAIR — PASS
- Çıktı etiketi: `0,1,2,4`; model metadata'sındaki `3=empty` reddedilir
- Fine-tune: yok
- Varsayılan demo gate: kapalı

Zenodo kaydının dosya adı `BRATS19`, açıklaması BraTS 2021'dir. Bu isim
tutarsızlığı raporda korunmuştur; modelin kanal/etiket sözleşmesi doğrudan
kurulu `dataset.json` dosyasından doğrulanmıştır.

### Son doğrulama

- `pytest`: 19/19 PASS
- `pip check`: PASS
- Numpy `1.26.4` ve SimpleITK `2.3.1` pinleri korunuyor
- nnU-Net preflight: READY
- `GBMAID_ENABLE_NEW_PATIENT_NNUNET=0`: beklenen güvenlik durumu

## Açık üretim kapıları

1. Üç kaynak için mevcut Z-score değerleri tek-vaka smoke istatistiğidir;
   üretimde kullanılamaz. Uygun N4 kohort manifestleriyle yeniden fit edilmelidir.
2. Bu üretim istatistikleri olmadığı için API'nin gerçek hasta çağrısı Z-score
   aşamasında bilinçli olarak `blocked` döner; sahte/tek-vaka istatistiğiyle
   `ok` üretilmez.
3. nnU-Net kurulumu hazırdır fakat DSC `>= 0.85`, etiketli yeni-hasta demo
   validasyon vakası olmadan sağlandı olarak işaretlenemez. Bu nedenle gate
   kapalıdır. TCGA/UPenn/LUMIERE bu testi yapmak için segmentasyon girdisi
   olarak kullanılmaz.
4. UPenn'de PyRadiomics çalıştırılmayacak. Supabase'deki 10.032 CaPTk satırı
   kullanılacaktır; ortak özellik uzayı eşlemesi Hafta 3 kapısıdır.
5. ComBat Nisa'nın özellik-uzayı adımıdır. `mr_scans.harmonization_status`
   alanına `combat` yazmak semantik olarak yanlıştır; feature katmanında
   izlenmelidir.
