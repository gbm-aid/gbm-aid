# GBM-AID Hafta 6 — Streamlit Juri Demosu

Ilk calisir surum (2026-09-11). **Production-ready DEGILDIR** — review-gate
(fiziksel inceleme + Codex capraz inceleme + final rapor onayi) bekliyor.

## Kurulum

```powershell
pip install -r requirements.txt -r requirements-demo.txt
```

`.env` proje kokunde (`GBM-AID Prototip/.env`) olmali — DB baglantisi
oradan okunur (hicbir yere gomulmez).

## Baslatma

Depo kokunden (koyu tema bayragiyla):

```powershell
streamlit run demo/streamlit_app.py --theme.base dark
```

veya `demo/` icinden (tema `demo/.streamlit/config.toml`'dan gelir):

```powershell
cd demo
streamlit run streamlit_app.py
```

## Demo hastalari — UC BLOKLU kurgu (2026-09-13 guncellemesi)

`demo_helpers.py::DEMO_PATIENTS = ("UCSF-PDGM-167", "TCGA-06-5412",
"Patient-028")`. Sira DEMO AKISINA gore; ayrica serbest `patient_id`
girisi de var.

| Blok | Hasta | Rol |
|---|---|---|
| **A** | `UCSF-PDGM-167` | **Tam zincir.** Klinik kovaryatlar TAM (GTR=Y, IDH-Wildtype, MGMT-Methylated) → uctan uca canli akis. Omics paneli BOS kalir (omics yalniz 48 TCGA hastasinda var). |
| **B** | `TCGA-06-5412` | **Omics anlatisi + "sistem uydurmuyor" gosterimi.** GTR/IDH1 kaynakta NULL — sistemin eksik veriyle nasil davrandigini gosterir. ⚠️ Bu davranis 2026-09-13'te DEGISTI, asagidaki uyariya bkz. |
| **C** 🆕 | `Patient-028` | **Longitudinal buyume simulasyonu + YORUMLANABILIR hacim projeksiyon araligi** (9 ziyaret, gompertz, RANO **PD**; 6 aylik aralik **−%0,47 … +%52,64**, medyan **+%43,97**; hedef hafta 64,07). ~~Cox risk paneli **2026-09-13 itibariyla, Y1 uygulanmadan once** bu hastada **422** (cok-zaman-noktali) → B-1 duzeltmesinden sonra istek yine **200** doner ve buyume paneli DOLU gelir. ⚠️ Y1 (LUMIERE pre-op kanonik vizit) uygulandiktan sonra bu satirin 422 iddiasi YENIDEN OLCULMELIDIR.~~ ✅ **OLCULDU 2026-09-15: 422 YOK — istek 200, Cox risk paneli DOLU** (asagidaki olcum kutusuna bkz.). |

✅ **BLOK C OLCUM BORCU KAPANDI — 2026-09-15 (`belge-agent-Z4`).**
Yukaridaki satirin **422** iddiasi Y1 (LUMIERE pre-op kanonik vizit)
sonrasi **yeniden olculdu** ve **gecersiz cikti**. Koordinatorun canli
olcumu — `api.predict.predict_patient('Patient-028')`, canli DB, gercek
v3b checkpoint'i (`models/cox_phm_v3b_lowvar_v2amgmt_2026-09-12.pkl`):

| Alan | Deger |
|---|---|
| HTTP | **200** (422 DEGIL) |
| `risk_score_log_partial_hazard` | **-0.9120252240782928** |
| `hazard_ratio_partial_hazard` | **0.4017098472091532** |
| `model_arm` | **`v3b_lowvar_v2amgmt`** |
| `xgboost` | **`available: True`** — `probability_12_month_survival` **0.835877537727356**, `status: 'shadow'` |
| SHAP | **24 katki** (16 radyomik + 8 klinik) |

**Sebep:** Y1, cok-zaman-noktali LUMIERE hastalarinda kanonik viziti
`Rating == 'Pre-Op'` sozlesmesine bagladi; "hangi tarama kullanilacak"
belirsizligi kalktigi icin Cox 422'si artik dogmuyor. Ayni davranis
`Patient-002`'de de olculdu (`unavailable_blocks = []`).

🔴 **Iki ardisik duzeltme karistirilmasin:** (1) **B-1** — Cox 422'si
blok-fatal yapildi → istek **200** ama **`risk.available: false`**;
(2) **Y1** — risk skoru da doldu → **`risk.available: true`**. Ustteki
cizili satir (1) ile (2) ARASINDAKI ani donduruyordu, yanlis degildi —
**bayatti**.

⚠️ Bu olcum YALNIZ Cox/XGBoost panelini kapsar. **Panel 7'nin hacim
sayilari ve projeksiyon araligi DEGISMEDI**; asagidaki not aynen
gecerlidir.
📄 `ZORUNLU-BEYANLAR.md` **D7** · `BEKLEYEN-KARARLAR.md` (2026-09-14
kucuk borclar, kalem 3 — ayni gun kapatildi) · `log/2026-09-15.md`

🔴 **ARALIK 2026-09-13 AKSAMI DEGISTI (G1 projeksiyon ufku duzeltmesi).**
Yukaridaki tabloda yazan aralik GUNCEL olandir. **Eski/gecersiz hali
`+%32,96 … +%49,20` (bant 16,24 pp)** — o sayilar `project_volume_range()`
ufkunu `t = 6 × 4,345 = 26,07` hafta olarak hastanin **ILK** taramasindan
itibaren MUTLAK alan, ama `v0` olarak **SON** ziyaretin hacmini kullanan
HATALI bir hesaptan geliyordu; `Patient-028`'in son ziyareti hafta **38**
oldugu icin karsilastirma **geriye donuktu**. Duzeltmeden sonra hedef
hafta **38 + 26,07 = 64,07** ve **alt uc isaret degistirdi**
(`+%32,96` → `−%0,47`). Eski sayilar silinmedi, v1 CSV'de ve
`ZORUNLU-BEYANLAR.md` D8'de GECERSIZ isaretiyle duruyor.
Detay: `decisions/2026-09-13-projeksiyon-ufku-duzeltmesi.md`.

🆕 **Blok C hastasi 2026-09-13'te `Patient-002` DEGIL `Patient-028`.**
Sebep OLCULDU (`artifacts/week6/demo_blok_c_projection_scan_2026-09-13.csv`,
90 LUMIERE hastasinin tamami canli tarandi): `Patient-002`'nin 6 aylik
araligi **−%99,99999999 … +%3.305.185,28** (dejenere, ust uc ~1,9 milyar
mm³) — ekranda gosterilemez. *(Eski gerekce: `Patient-028` 67 PD hastasi
arasinda EN DAR banda sahipti — 16,24 yuzde puani. **Bu gerekce
2026-09-13 aksami GECERSIZLESTI**, asagiya bkz.)*

🔄 **SECIM OLCUTU 2026-09-13 aksami ACIKCA DEGISTIRILDI** (sessizce
degil). Yalniz "en dar bant" yetersiz cikti: duzeltilmis taramada 67 PD
hastasinin **7'sinde** projeksiyonun **ust ucu bile negatif**
(`Patient-009`, `-037`, `-040`, `-043`, `-081`, `-085`, `-090`) — yani
"progresif hastalik grubundaki hasta, tumoru **kesin** kuculuyor". En dar
uc bant tam bu gruptadir, jurriye savunulamaz. Yeni olcut **dort
basamakli**: (1) uc kosullu yorumlanabilirlik, (2) RANO **PD** ise
projeksiyonun **ust ucu pozitif** olmali (klinik tutarlilik), (3) makul
seri uzunlugu / son ziyaret haftasi (`Patient-081`: 3 ziyaret, son
ziyaret hafta 6 → elenir), (4) bunlari gecen **5** aday arasinda en dar
bant. Sonuc: hasta **`Patient-028` olarak KALDI** (bant 53,12 pp;
digerleri `Patient-022` 62,77 · `Patient-065` 117,97 · `Patient-064`
121,49 · `Patient-012` 128,23), ama **gerekcesi degisti** — artik "en dar
bant" DEGIL, "olcutu gecenler arasinda en dar bant". Tum
yorumlanabilirler arasinda en dar olan `Patient-009`'dur (47,39 pp,
aralik −%64,25 … −%16,86).

⚠️ Bu bir **secimdir** ve panelde **kapsam orani beyan edilir**
(67 PD hastasinin **12'sinde** aralik yorumlanabilir — eski/gecersiz sayi
**14**) — bu yuzden cherry-picking degildir. Beyan:
`ZORUNLU-BEYANLAR.md` **D8**.

Blok A yedekleri (buton listesinde DEGIL, "Serbest giris" ile elle):
`UCSF-PDGM-137` (53/E, 799 gun, vefat), `UCSF-PDGM-421` (80/E, 518 gun,
yasiyor).

*Eski (2026-09-11) tek-hastali kurgu `TCGA-06-5412` (birincil) +
`TCGA-76-4928` (yedek) idi — bu README o kurguyu anlatiyordu, 2026-09-13'te
duzeltildi.*

🔴 **CANLI DAVRANIS PROVA ONCESI TEYIT EDILMELIDIR.** Her iki blok icin de
"su hasta 200 alir / su hasta 422 alir" iddiasi bu dosyada KURULMAZ:
servis tarafindaki eksik-deger politikasi 2026-09-12'den beri aktif olarak
degisti — K19 ile `mgmt_status`, 2026-09-13'te ayrica `gtr_over90percent`
ve `idh1_status` NULL'lari egitimle HIZALANDI (`CLINICAL_COVARIATE_NULL_
POLICY` → `TRAIN_ALIGNED`; `age`/`gender` STRICT kaldi).
⚠️ **Blok B'nin "sistem eksik veride REDDEDER" anlatisi bu degisiklikle
GECERSIZLESTI** (kaynak: `log/2026-09-13.md` [backend-agent-E], gercek DB
olcumu). Anlatinin yeni bicimi BELGE tarafinda ayrica karara baglanacak
(`syntheses/juri-demo-metinleri-2026-09-11.md` §2B) — bu README o karari
ONCELEMEZ. Provadan once her iki hasta da canli calistirilip
gozlenmelidir.

⚠️ **Risk sayilari bu dosyada VERILMEZ** (Baris karari 2026-09-12): legacy
`arm_primary_wt93_icc60_th06.pkl` fallback kolundan gelen risk/HR degerleri
nihai modelden (`v3b_lowvar_v2amgmt`) DEGILDIR ve juriyi yanıltir. Sayilar
SILINMEDI — tarihsel kayit `ZORUNLU-BEYANLAR.md` D6 ve
`syntheses/juri-demo-metinleri-2026-09-11.md` §0.3'te duruyor.
Kurulacak cumle: *"Model goreli risk siralamasi uretir; 'bu hasta X ay
yasar' cumlesi KURULMAZ."*

## Dogrulama durumu (2026-09-11, canli veri)

⚠️ **Bu bolum TARIHSEL kayittir** — o gunku **tek-hastali** kurguya
(`TCGA-06-5412`) aittir ve Blok A (`UCSF-PDGM-167`) icin AYNI 7/7 panel
dogrulamasi HENUZ TEKRARLANMADI.

7/7 panel GERCEK DB + gercek artefaktlarla kosturuldu. Yontem:
`streamlit.testing.v1.AppTest` ile TAM script kosusu (**exception 0**),
panel-basina bagimsiz harness, ve headless `streamlit run` (HTTP 200 +
`/_stcore/health` 200).

| Panel | Kanit (TCGA-06-5412, 2026-09-11) |
|---|---|
| 1 Hasta ozeti | 78 / Female / TCGA-GBM / DECEASED 138 gun |
| 2 Cox risk | panel calisti; sayisal deger 2026-08-18 referansiyla birebir esitti (deger burada YAZILMIYOR — yukaridaki karara bkz.) |
| 3 SHAP | 6 katki; additivite tam (katkilar toplami + taban = risk skoru) |
| 4 FAISS | clinical index_size **722**, omics **48** — ikisi de acildi |
| 5 Omics | TMZ 39,92 sensitive · agresiflik 53,97 · DNA-onarim 36,45 impaired · parp_inhibitor_candidate |
| 6 Literatur | 5/5 PubMed parcasi (Upstash onbellegi), LLM kapali + gerekcesi ekranda |
| 7 Rol tablosu | 611/585 · 295/169 · 91/72 · 39 · 48 · 722 (sabit) |

Eski yedek hasta `TCGA-76-4928` de o gun dogrulandi (risk/HR degerleri
kayitli ama burada YAZILMIYOR); bu hasta artik demo listesinde DEGIL.

### 🆕 2026-09-13 dogrulamasi (demo-agent-B, Panel 7 eklendikten sonra)

`streamlit.testing.v1.AppTest` ile TAM script kosusu, **UC hasta icin**
(literatur paneli kapali — ag erisimi demo disi tutuldu):

| Hasta | Exception | Panel 7 durumu |
|---|---|---|
| `UCSF-PDGM-167` | **0** | beyanli BOS (LUMIERE serisinde 0 satir) |
| `TCGA-06-5412` | **0** | beyanli BOS (ayni sebep) |
| `Patient-028` | **0** | DOLU: aralik **+%32,96 … +%49,20** + kapsam beyani (67/14) + semantik notu ekranda |

Ayrica `pytest tests/test_api_analyze_patient.py tests/test_growth_simulation.py`
→ **60 passed** (regresyon yok; `api/` ve `pipeline/` dosyalarina
DOKUNULMADI).

### 🔴 2026-09-13 (aksam) — G1 duzeltmesinden SONRA yeniden dogrulama (backend-agent-H)

Yukaridaki tablonun `Patient-028` satirindaki aralik **GECERSIZDIR**
(hatali ufuk). Duzeltme sonrasi AYNI `streamlit.testing.v1.AppTest`
kosusu (literatur paneli kapali):

| Hasta | Exception | Panel 7 durumu |
|---|---|---|
| `UCSF-PDGM-167` | **0** | beyanli BOS (LUMIERE serisinde 0 satir) |
| `TCGA-06-5412` | **0** | beyanli BOS (ayni sebep) |
| `Patient-028` | **0** | DOLU: aralik **−%0,47 … +%52,64** (medyan **+%43,97**) · ufuk metni *"son ziyaret hafta 38.00 → hedef hafta 64.07"* · kapsam beyani **67/12** · semantik notu ekranda |

Kontroller (hepsi geçti): yeni aralik ve yeni ufuk metni **yalniz**
`Patient-028` panelinde görünüyor; **eski** aralik (`%+32.96`) ve **eski**
kapsam sayisi (`67 PD hastasinin 14'unde`) hiçbir hastanin ekraninda
GÖRÜNMÜYOR.

`pytest tests/test_api_analyze_patient.py tests/test_growth_simulation.py`
→ **64 passed** (60 → 64; **4 yeni G1 regresyon testi**, hiçbir test
silinmedi). Bu sefer `pipeline/growth_simulation.py` ve
`api/analyze_patient.py` **DEGISTIRILDI** (ufuk duzeltmesi) —
`api/predict.py`'ye DOKUNULMADI.

⚠️ **`subst X:` GEREKMEDI** — 8.3 kisa yol override'i K12'yi tek basina
asti (canli kanit: her iki indeks de acildi, `api/similar.py`
DEGISTIRILMEDEN).

⚠️ `st.altair_chart` streamlit 1.49.1'de `width=` KABUL ETMIYOR; o cagri
bilincli olarak `use_container_width` ile birakildi (AppTest yakaladi).

Review-gate'in 3 adimi HENUZ YAPILMADI.

## Bilinen sinirlar (bilincli)

- **K12 / Turkce-karakterli yol:** faiss, `C:\Users\Barış\...` yolunu
  acamiyor. Demo, `GBMAID_FAISS_*_INDEX_DIR` override'ina Windows 8.3
  KISA YOLUNU (ASCII) vererek bunu asar (api dosyasina dokunmadan).
  Kisa yol uretilemezse panel zarifce duser ve `subst X:` cozumunu soyler.
- **Eksik klinik kovaryat → fallback kolu.** TCGA hastalarinda
  `gtr_over90percent`/`idh1_status` kaynakta NULL; klinik-kovaryatli Cox
  artefakti bu durumda tahmin URETMEZ ve "Otomatik" modda demo yalniz-
  radyomik birincil kola (`arm_primary_wt93_icc60_th06`) SEFFAF bir
  uyariyla duser. ⚠️ **Bu davranis 2026-09-13'te DEGISTI** — servis
  tarafindaki NULL politikasi egitimle hizalandi (K19 MGMT + ayni gun
  GTR/IDH). Fallback kolunun ne zaman devreye girecegi bu yuzden ARTIK
  ESKI KAYITLARDAN OKUNAMAZ. Prova oncesi canli teyit SART; bu README'de
  yanit kodu (200/422) iddiasi KURULMUYOR.
  ⚠️ Fallback kolundan gelen risk/HR sayilari **sunumda kullanilmaz**
  (nihai model `v3b_lowvar_v2amgmt` degil, legacy kol).
- **LLM ozeti KAPALI** (bilincli — saglayici anahtari yok, yeni bulut
  servisi acilmiyor). Literatur paneli PubMed sorgusu + FAISS siralamasini
  gosterir, degrade mesaji verir.
- **XGBoost paneli YOK** (bilincli): uretim kosusu 2026-09-12'de YAPILDI
  (`entities/xgboost-kosu-katalogu.md`) ama demoya panel olarak
  BAGLANMADI; `model_registry` statusu `shadow`, `POST /analyze_patient`
  yanitinda da `available:false` doner.
  *(Eski hali: "XGBoost VE buyume-simulasyonu panelleri YOK" —
  2026-09-13'te buyume paneli EKLENDI, asagiya bkz.)*
- 🆕 **Buyume simulasyonu paneli EKLENDI (Panel 7, 2026-09-13).**
  `api/analyze_patient.py::_build_growth_simulation_block()` process-ici
  cagrilir (o dosyaya YAZILMADI, yalniz import). Rol tablosu paneli
  **7 → 8** olarak yeniden numaralandi. Panel yalniz LUMIERE
  (`Patient-XXX`) hastalarinda dolar; Blok A/B hastalarinda **beyanli
  bos** kalir (pipeline'in kilitli sorgu sozlesmesi `patient_id ILIKE
  'Patient-%'`).
  ⚠️ **Projeksiyonun iki ciddi okuma tuzagi panelde YAZILI:** (1) aralik
  hastanin kendi r'sinin ileriye tasinmasi DEGIL, hastanin egri
  sablonuna RANO grubunun r-IQR'sinin konmasidir; (2) ufuk hastanin
  KENDI zaman ekseninde hesaplanir.
  🔴 **(2) 2026-09-13 aksami DUZELTILDI (G1).** *(Eski/hatali hali: "ufuk
  t=26,07 hafta HASTANIN KENDI zaman ekseninde hesaplanir — son
  ziyaretin USTUNE eklenen 6 ay DEGILDIR". Bu bir **hataydi**, bilincli
  bir tasarim degil: `v0` son ziyaretin hacmiydi, ufuk ise ilk taramadan
  itibaren mutlak 26,07 hafta → son ziyaret 26,07'den sonraysa
  karsilastirma GERIYE DONUK oluyordu.)* Guncel hali:
  `t = son_ziyaret_haftasi + 6 × 4,345`; `Patient-028` icin
  **38 + 26,07 = 64,07**. Panel artik "son ziyaret hafta X → hedef hafta
  Y" olarak ikisini de yazar (sabit 4,345 arayuzde yeniden
  hesaplanmiyor, `projection.t_target_week` alanindan okunuyor).
  Bu duzeltmeye ragmen *"6 ay sonra tumor %44 buyur"* cumlesi KURULMAZ.
- **Hacim projeksiyonu kohortun cogunda YORUMLANAMIYOR (olculdu).**
  90 LUMIERE hastasinin 76'sinda egri fit'i `ok`, projeksiyon yalniz
  **67 PD** hastasinda uretilebiliyor (CR n=1 / PR n=3 / SD n=5
  `interpretable=False`), ve bu 67'nin **yalniz 12'sinde** aralik
  yorumlanabilir *(eski/gecersiz sayi: 14 — hatali ufukla olculmustu)*.
  Panel bu orani ekranda beyan eder.
  🔴 **Ek olculmus sinir:** 67'nin **7'sinde** projeksiyonun **ust ucu
  bile negatif** — PD (progresif) grubundaki hasta icin "tumor kesin
  kuculuyor". RANO etiketi ile volumetrik yorunge birbirini
  desteklemiyor; bu hastalar demoda GOSTERILMEZ ve secim olcutunun 2.
  basamagiyle elenir (bkz. yukaridaki secim olcutu bolumu · C4:
  volumetrik, resmi RANO degil).

## 🆕 Panel 9 — MR kesiti + segmentasyon overlay'i (2026-09-14, "ŞIK A")

Barış'ın kararı: **YALNIZ GÖSTERİM.** Yeni hasta yükleme YOK, görüntüden
**canlı radyomik çıkarım YOK** (Şık B/C reddedildi). Analiz mevcut haliyle
DB'den koşmaya devam eder; bu panel yalnız NAS/diskteki görüntüyü ve
**hazır** maskeyi ekrana getirir. Rol tablosu **Panel 8** olarak yerinde
kaldı, panel sırası değişmedi — yeni panel **9** numarasıyla sona eklendi.

**Ekranda görünen ZORUNLU beyan (her koşulda, veri olmasa da):**
> Görüntü NAS'tan salt-okunur okunmaktadır. Radyomik özellikler bu
> görüntüden **canlı çıkarılmamaktadır**; C32 sözleşmesiyle önceden
> çıkarılmış ve veri tabanında saklanan değerlerdir. Segmentasyon maskesi
> **hazır** maskedir (DeepBraTumIA), bu oturumda üretilmemiştir.

**Görüntü/maske çifti nasıl bulunur:** kendi yol mantığımız YAZILMADI —
`pipeline/segmentation.py::resolve_nas_path()` + `resolve_ready_mask()`
doğrudan çağrılır. Böylece ekrandaki maske, C32 radyomik çıkarımının
kullandığı maskenin **ta kendisidir**.

### Canlı ölçüm (2026-09-14, üç demo hastasının tamamı)

| Hasta | Kaynak | mask_source | Hacim (voksel) | En geniş kesit z | O kesitte voksel | Tümör aralığı z | Toplam voksel |
|---|---|---|---|---|---|---|---|
| `Patient-028` (week-038) | LUMIERE / NAS | `lumiere_deepbratumia_native` | 160×256×256 @1 mm | **178** | **3.640** | 157–220 | **112.576** |
| `TCGA-06-5412` | TCGA / NAS | `provided_whole_tumor` | 240×240×190 @1 mm | 97 | 3.107 | 79–152 | 127.503 |
| `UCSF-PDGM-167` | UCSF / **yerel disk** | `ucsf_native` | 240×240×155 @1 mm | 120 | 2.713 | 81–148 | 104.795 |

`Patient-028`/week-038'in **112.576 mm³**'ü, Panel 7'deki projeksiyonun
`v0` değeriyle **birebir aynıdır** (1 mm izotropik → voksel sayısı = mm³).
İki panel aynı maskeyi gösteriyor.

### Ölçülen ve düzeltilen üç tuzak

1. 🔴 **Görev talimatındaki "atlas uzayı" premisi bu dal için GEÇERSİZ.**
   Talimat, maskenin `atlas/segmentation/seg_mask.nii.gz` (182×218×182) ve
   ham `CT1.nii.gz`'in native (256×256×192) olduğunu, bu yüzden
   `atlas/skull_strip/ct1_skull_strip.nii.gz` kullanılması gerektiğini
   söylüyordu. **Projenin kullandığı dal o değil:**
   `pipeline/segmentation.py::_lumiere_mask()` ana kaynak olarak
   `DeepBraTumIA-segmentation/**native**/segmentation/ct1_seg_mask.nii.gz`
   seçer ve onu ham `week-XXX/CT1.nii.gz` ile eşleştirir. Canlı ölçüm
   (Patient-028/week-038): ikisi de **(256,256,160)**, `geometry_match=True`.
   Atlas dalına hiç girilmiyor; girilseydi radyomikle **tutarsız** bir
   maske gösterilirdi.
2. 🔴 **NIfTI eksen düzeni kohortlar arası FARKLI** (talimatta yoktu):
   LUMIERE `PIR`, TCGA ve UCSF `LPS`. Sabit "axis=2 eksenel kesittir"
   varsayımı LUMIERE'de **koronal/sagital** kesit üretirdi. Çözüm:
   `nibabel.as_closest_canonical()` (yalnız eksen permütasyonu/çevirme —
   interpolasyon YOK) ile her iki hacim RAS'a getirilir.
3. 🔴 **SimpleITK Türkçe karakterli yolları AÇAMIYOR** (K12'nin aynı
   sınıfı; UCSF görüntüleri Türkçe karakterli yerel bir yolda ve
   `sitk.ReadImage` "Unable to open" fırlatıyor). Bu panel bu yüzden
   **yalnız `nibabel`** kullanır; `pipeline/resampling.py::
   validate_image_mask_geometry()` (SimpleITK tabanlı) burada
   ÇAĞRILMAZ, yerine gösterim seviyesinde shape + affine eşitliği
   kontrol edilir.

### Yön (orientation) sözleşmesi — ölçülerek doğrulandı

Eksenel kesit, **radyolojik** yön: görüntünün SOLU = hastanın SAĞI,
yukarı = anterior. Maske ağırlık merkezinin RAS x koordinatı ile ekran
sütunu karşılaştırılarak üç hastada da doğrulandı:
`UCSF-167` x=−152,9 (hasta solu) → ekranda sağ yarı; `TCGA-06-5412`
x=+29,7 → ekranda sol yarı; `Patient-028` x=+12,0 → ekranda sol yarı.

### Fail-closed davranış (B-1 deseni, gerçekten tetiklendi)

`GBMAID_NAS_ROOT` geçici olarak bozuk bir yola çevrilip **temiz bir
process**te AppTest koşuldu (`.env`'e hiçbir şey yazılmadı):
`exception=0`, `error=0`, üretilen görüntü **0**, panel sayısı **9**
(hepsi render edildi), diğer panellerin metrikleri çalışmaya devam etti
(10 metric), ZORUNLU beyan ekranda kaldı ve Panel 9 şunu yazdı:
> ⚪ MR görüntüsü şu an erişilemiyor: NAS bağlantısı yok veya dosya
> çözümlenemedi (FileNotFoundError: NAS kökü erişilebilir değil: …)

Aynı şekilde DB kopukluğu (`psycopg2.OperationalError`) ve
**geometri uyuşmazlığı** da tetiklendi. Geometri uyuşmazlığı **sessiz
geçilmez**: `geometry_mismatch: True` döner ve panel `st.error` ile
görünür hata verir — yanlış hizalanmış bir overlay jüriye tümörü yanlış
yerde gösterirdi.

### Açık borç / bilinen sınır

- ⚠️ **UCSF dalı `pipeline/`'da YOK.** `resolve_ready_mask()` UCSF'i
  tanımıyor (*"Hazır maske sözleşmesi tanımsız kaynak: UCSF"*), çünkü
  UCSF pipeline'a 2026-08-18'de ayrı bir araçla
  (`tools/run_pyradiomics_ucsf.py`) girdi. Görev kısıtı gereği
  `pipeline/`'a dokunulmadığı için UCSF maskesi (`<ID>_tumor_
  segmentation.nii.gz`, `mask_source="ucsf_native"`) **demo seviyesinde**
  çözülüyor — o aracın sözleşmesi birebir taklit edilerek. Kalıcı çözüm
  `pipeline/segmentation.py`'ye bir `_ucsf_mask()` eklemektir
  (imaging-agent/backend-agent işi).
- ⚠️ UCSF görüntüleri **NAS'ta değil**, yerel diskte (`mr_scans.file_path`
  mutlak yol). Bu yüzden UCSF paneli NAS kapalıyken de çalışır — demo
  provasında "NAS'tan okuyoruz" cümlesi UCSF için **yanlış** olur.
- ⚠️ Panel yalnız **T1ce** modalitesini gösterir (`mr_scans.modality =
  'T1ce'`), çünkü C32 çıkarımı da T1ce üzerinden yapılır.
- ⚠️ Görüntü, C32 zincirinin **N4 + T1ce-özel Z-score** adımlarından
  GEÇMEMİŞ ham görüntüdür — gösterim için pencereleme (%1–%99 persentil)
  uygulanır. Radyomik değerler bundan türetilmediği için bu bir tutarsızlık
  değildir, ama "ekranda gördüğünüz piksel değerleri modele giren
  değerlerdir" DENMEZ.
- ⚠️ **Review-gate YAPILMADI** (ZORUNLU-BEYANLAR.md E3) — bu panel
  "production-ready" değildir.
