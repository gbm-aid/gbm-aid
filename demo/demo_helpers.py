"""GBM-AID Hafta 6 juri demosu -- veri erisim yardimcilari.

GOREV SINIRLARI (2026-09-11, AKTIF-GOREVLER.md kaydi):
- DB'ye YALNIZ readonly SELECT (dogrudan sorgular `get_connection(
  readonly=True)` ile; `api/predict.py` / `api/similar.py` /
  `pipeline/rag_pipeline.py` fonksiyonlari zaten yalniz SELECT yapar).
- `api/`, `pipeline/`, `tools/` dosyalari DEGISTIRILMEZ -- yalniz
  import edilip process-ici cagrilir (HTTP sunucusuna bagimlilik YOK;
  API ayakta olmasa da demo calisir).

BILINEN ISLETIMSEL RISKLER (bu modulun actigi degil, yonettigi):
- K12 / Turkce-karakterli yol: faiss'in C++ dosya okuyucusu
  "C:\\Users\\Baris..." benzeri yollari ACAMIYOR (canli dogrulandi,
  2026-09-11). `api/similar.py` bunun icin `GBMAID_FAISS_*_INDEX_DIR`
  env override'i sunuyor; burada Windows 8.3 KISA YOLU (ASCII)
  uretilip o override'a veriliyor -- api dosyasina DOKUNULMADAN.
  Kisa-yol uretimi basarisiz olursa (8.3 adlari kapali bir volume vb.)
  panel zarifce duser ve `subst X:` cozumunu onerir.
- Varsayilan Cox artefakti (GUNCEL, 2026-09-13'ten beri):
  `models/cox_phm_v3b_lowvar_v2amgmt_2026-09-12.pkl` -- NIHAI MODEL
  `v3b_lowvar_v2amgmt` (Baris karari 2026-09-12), 16 radyomik + 8
  klinik = 24 ozellik. `api/predict.py::DEFAULT_CHECKPOINT_PATH` bunu
  gosterir; bu modul artefakt adini SABIT YAZMAZ, `_call_with_
  checkpoint(None)` ile api/predict.py'nin kendi varsayilanini kullanir.
  ~~Eski/GECERSIZ metin: "Varsayilan Cox artefakti (`models/cox_phm_
  primary_wt93_clinical_full_ucsf_2026-08-18.pkl`, v1 = 93 radyomik +
  6 klinik kovaryat)"~~ (uzeri cizildi, SILINMEDI -- wiki hard rule #3:
  celiski silinmez, isaretlenir).
- Fallback kolu: klinik kovaryat eksikligi 422'ye yol acarsa demo ESKI
  yalniz-radyomik kola (`artifacts/week3/cox_model/_checkpoints/
  arm_primary_wt93_icc60_th06.pkl`, 2026-08-15 kosusu, harici C-index
  0,689 [0,546-0,824], TCGA n=38) SEFFAF bir notla otomatik duser.
  UYARI: bu kol NIHAI MODEL DEGILDIR; buradan gelen risk/HR sayilari
  sunumda nihai model sonucu olarak KULLANILMAZ (demo/README.md).
  UYARI: 2026-09-13'ten beri `gtr_over90percent`/`idh1_status`/
  `mgmt_status` NULL'lari 422 TETIKLEMEZ (`api/predict.py::CLINICAL_
  COVARIATE_NULL_POLICY` = TRAIN_ALIGNED); yalniz `age`/`gender` NULL'u
  (STRICT) ve coklu-tarama durumu 422 verir -- yani fallback ARTIK cok
  nadir devreye girer. ~~Eski/GECERSIZ: "TCGA hastalarinda
  gtr_over90percent/idh1_status NULL oldugu icin 422 verir"~~.
  2026-08-18 referans olcumu (TCGA-06-5412: risk 0,2359 / HR 1,2661)
  bu fallback kolla birebir yeniden uretildi (2026-09-11 dogrulamasi).
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# Depo kokunu sys.path'e ekle -- `streamlit run demo/streamlit_app.py`
# hangi CWD'den kosulursa kosulsun `api.*` / `pipeline.*` importlari
# calissin. `.absolute()` bilincli (`.resolve()` DEGIL) -- subst/kisa-yol
# senaryolarinda api/similar.py'nin kendi desenine sadik (bkz.
# tests/test_no_resolve_path_regression.py).
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).absolute().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Demo hastalari -- UC BLOKLU demo kurgusu. Sira DEMO AKISINA gore:
#   1) UCSF-PDGM-167 -- BLOK A, tam zincir, canli /analyze_patient -> 200
#      (GTR=Y, IDH-Wildtype, MGMT-Methylated; klinik kovaryatlar TAM).
#      Omics paneli BOS kalir (omics yalniz 48 TCGA hastasinda).
#   2) TCGA-06-5412  -- BLOK B, omics anlatisi + "sistem eksik veriyi
#      beyanla modele veriyor" gosterimi (2026-09-13'ten sonra bu hasta
#      canli 200 aliyor; GTR/IDH/MGMT NULL'lari TRAIN_ALIGNED).
#   3) Patient-028   -- BLOK C (LUMIERE), longitudinal buyume simulasyonu
#      + YORUMLANABILIR 6 aylik hacim projeksiyon araligi.
#
# BLOK C HASTASI 2026-09-13'te DEGISTI: Patient-002 -> Patient-028
# (demo-agent-B, B-2 gorevi). GEREKCE OLCULDU, varsayim DEGIL --
# `artifacts/week6/demo_blok_c_projection_scan_2026-09-13.csv`
# (LUMIERE'in longitudinal C32 WT serisindeki 90 hastasinin TAMAMI,
# `api/analyze_patient.py::_build_growth_simulation_block()` CANLI
# cagrilarak tarandi):
#   * `Patient-002` 6 aylik aralik: -%99,99999999 .. +%3.305.185,28
#     (ust uc ~1,9 milyar mm3 = kafa-ici hacmin ~1360 kati) -> ekranda
#     gosterilemez, sayi hatasi gibi gorunur.
#   * `Patient-028` 6 aylik aralik: +%32,96 .. +%43,87 .. +%49,20
#     (v0=112.576 mm3, ust uc 167.961 mm3) -> 67 PD hastasi arasinda EN
#     DAR bant (16,24 yuzde puani) ve tek yonlu (buyuyen tumor).
#
# 🔴 GECERSIZ (2026-09-13 aksami, G1) -- YUKARIDAKI `Patient-028`
# ARALIGI VE "16,24 pp EN DAR BANT" GEREKCESI ARTIK KULLANILMAZ.
# Sebep: `project_volume_range()` ufku `t = 6 x 4,345 = 26,07` haftayi
# hastanin ILK taramasindan itibaren MUTLAK aliyordu, ama `v0` SON
# ziyaretin hacmiydi -> `Patient-028`'in son ziyareti hafta 38 oldugu
# icin karsilastirma GERIYE DONUKTU. Duzeltildi (`from_week` =
# son ziyaret haftasi, `t_target = 38 + 26,07 = 64,07`).
# YENI (olculdu, `...-v2-duzeltilmis.csv`, 90/90 hasta yeniden tarandi):
#   * `Patient-028` aralik: **-%0,47 .. +%43,97 .. +%52,64**
#     (v0=112.576 mm3 -> 112.045 / 162.073 / 171.841 mm3), bant
#     **53,12 yuzde puani**. ALT UC ISARET DEGISTIRDI (+%32,96 -> -%0,47):
#     "kesin buyume" -> "buyume olasi, kucul(me de mumkun)".
#   * Yorumlanabilir sayisi 14 -> **12** (uc kosullu olcut).
#   * `Patient-028` artik EN DAR bant DEGIL (12 arasinda 4.); en dar
#     `Patient-009` (47,39 pp).
# ⚠️ SECIM OLCUTU DEGISTI (AÇIKÇA, sessizce degil): yalniz "en dar
# bant" YETERSIZ cikti -- 67 PD hastasinin **7'sinde** projeksiyonun
# UST UCU BILE NEGATIF, yani "progresif hastalik grubundaki hasta,
# tumoru KESIN kuculuyor" diyor; en dar uc bant (Patient-009 / -081 /
# -043) tam bu gruptadir ve juriye savunulamaz. Yeni olcut DORT
# basamakli: (1) uc kosullu yorumlanabilirlik, (2) klinik tutarlilik --
# RANO grubu PD ise ust uc POZITIF olmali, (3) makul seri uzunlugu /
# son ziyaret haftasi (`Patient-081`'in son ziyareti hafta 6, 3
# ziyaret -> elenir), (4) bunlari gecenler arasinda EN DAR bant.
# Bu olcutle 5 aday kaliyor ve en dari YINE `Patient-028`
# (53,12 pp; digerleri Patient-022 62,77 · Patient-065 117,97 ·
# Patient-064 121,49 · Patient-012 128,23) -- hasta DEGISMEDI ama
# GEREKCESI ve ARALIGI DEGISTI.
# ⚠️ Ek durustluk notu: `Patient-028`'in bandinin dar olmasinin sebebi
# biyolojik kesinlik DEGIL, Gompertz sablonunun asimptota yakin
# olmasidir (a=173.999,7 · b=0,0721 -> ucu senaryo da platoya yakin
# hacimler veriyor). Hastanin KENDI fit edilmis r'si NEGATIF
# (-0,051261) oldugu halde grup r-IQR'siyle pozitif medyan cikmasi da
# bu sablon etkisinin sonucudur.
# Detay: `decisions/2026-09-13-projeksiyon-ufku-duzeltmesi.md`
# ⚠️ Bu bir SECIMDIR ve cherry-picking DEGILDIR -- cunku yaninda kapsam
# orani BEYAN EDILIR (asagidaki GROWTH_PROJECTION_COVERAGE_NOTE, panelde
# projeksiyonun YANINDA gorunur). Beyan: ZORUNLU-BEYANLAR.md D8.
# Blok A yedekleri (buton listesinde DEGIL, elle "Serbest giris" ile):
#   UCSF-PDGM-137 (53/E, ayni profil, 799 gun, vefat),
#   UCSF-PDGM-421 (80/E, ayni profil, 518 gun, yasiyor).
# Blok C yedekleri -- 🔴 ESKI/GECERSIZ liste (hatali ufukla olculdu):
#   Patient-065 (-%42,09..-%18,23), Patient-040 (-%62,71..-%37,34),
#   Patient-012 (-%34,11..+%15,82 -- sifiri kapsayan bant).
# Blok C yedekleri -- GUNCEL (2026-09-13 v2-duzeltilmis tarama, dort
# basamakli olcutu gecen 5 adayin `Patient-028` disindaki 4'u):
#   Patient-022 (bant 62,77 pp · -%47,47 .. -%15,36 .. +%15,30 · 6
#     ziyaret · son ziyaret hafta 53),
#   Patient-065 (117,97 pp · -%89,71 .. -%29,60 .. +%28,26 · 7 ziyaret),
#   Patient-064 (121,49 pp · -%45,63 .. +%16,45 .. +%75,86 · 6 ziyaret),
#   Patient-012 (128,23 pp · -%74,25 .. -%8,83 .. +%53,98 · 7 ziyaret).
# Eski (2026-09-11, tek-hastali kurgu): ("TCGA-06-5412", "TCGA-76-4928")
# Eski (2026-09-12, iki-hastali kurgu): ("UCSF-PDGM-167", "TCGA-06-5412")
DEMO_PATIENTS: tuple[str, ...] = ("UCSF-PDGM-167", "TCGA-06-5412", "Patient-028")

# Yalniz-radyomik birincil kol artefakti (fallback icin). Bicim B
# (ArmResult pickle) -- api.predict._normalize_arm bunu taniyor.
RADIOMICS_ONLY_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "week3"
    / "cox_model"
    / "_checkpoints"
    / "arm_primary_wt93_icc60_th06.pkl"
)

# ---------------------------------------------------------------------
# Sayi/rol tablosu -- KILITLI degerler (CLAUDE.md "KRITIK MIMARI
# KURALLAR" + "Sayi/isim tutarliligi"). TEK toplam sayi KULLANILMAZ
# ("~721" / "771 toplam" ifadeleri YASAK); her kaynak KENDI ROLUYLE
# verilir. Buradaki degerler SABITTIR, DB'den turetilmez.
# ---------------------------------------------------------------------
ROLE_TABLE_ROWS: tuple[dict[str, str], ...] = (
    {
        "Kaynak": "UPenn-GBM",
        "Rol": "Cox/XGBoost EGITIM havuzu (tek egitim kaynagi)",
        "Sayi": "611 hasta / 585 olum olayi (kayitli 630; 19'u yalniz post-op, disarida)",
    },
    {
        "Kaynak": "UCSF-PDGM",
        "Rol": "HARICI TEST seti (2026-08-18 revizyonu; TCGA'dan devraldi)",
        "Sayi": "295 hasta / 169 olay",
    },
    {
        "Kaynak": "LUMIERE",
        "Rol": "Longitudinal buyume simulasyonu + T3 RANO kalibrasyonu + FAISS kohortu (EGITIME GIRMEZ)",
        "Sayi": "91 hasta / FAISS'e giren 72",
    },
    {
        "Kaynak": "TCGA-GBM",
        "Rol": "FAISS v1 uyesi (egitime ve harici teste GIRMEZ)",
        "Sayi": "39 hasta (radyomik satiri olan)",
    },
    {
        "Kaynak": "TCGA-Omics",
        "Rol": "Opsiyonel omics modulu (Cox'un zorunlu girdisi DEGIL)",
        "Sayi": "48 hasta",
    },
    {
        "Kaynak": "FAISS v1 indeksi",
        "Rol": "Benzer-hasta havuzu = UPenn 611 + LUMIERE 72 + TCGA 39",
        "Sayi": "722 hasta (FIILI)",
    },
)

ROLE_TABLE_FOOTNOTE = (
    "Kilitli sayilar (CLAUDE.md): tek bir 'toplam kohort' sayisi "
    "KULLANILMAZ -- her kaynak kendi roluyle raporlanir. UCSF FAISS "
    "havuzunda YOKTUR ve hicbir toplama eklenmez. Eski/yanlis ifadeler "
    "('~721 ile egitildi', '771 toplam', 'harici test TCGA') gecersizdir."
)

RELATIVE_RISK_NOTE = (
    "Bu skor bir OLASILIK degildir; egitim kohortu (UPenn 611) "
    "ortalamasina gore GORELI risk siralamasidir. HR = exp(risk skoru): "
    "1'in ustu kohort ortalamasindan yuksek, alti dusuk goreli risk. "
    "Bireysel sagkalim suresi/olasiligi TAHMIN ETMEZ -- nihai modelin "
    "(v3b) kalibrasyon egimi 0,685 [0,470-0,900] olarak OLCULDU, yani "
    "mutlak sagkalim olasiligi iddiasi KURULMAZ; yalniz SIRALAMA "
    "yorumlanir (ZORUNLU-BEYANLAR.md A9/A10)."
)

# ---------------------------------------------------------------------
# NIHAI MODEL KIMLIGI -- ekranda gorunen TEK dogruluk kaynagi.
# Bu metni degistirirken CLAUDE.md "NIHAI COX MODELI SECILDI" maddesiyle
# ve decisions/2026-09-12-nihai-cox-model-secimi-v3b.md ile birebir
# tutarli kal. Sayilar KILITLI.
# UYARI: "en iyi model" / "en yuksek performansli" DENMEZ -- 8 kolun
# harici C guven araliklari TAMAMEN ortusuyor, secim ikincil olcutlerle
# yapildi.
# UYARI: tek-esik dili (ornegin "0,72 hedefini gectik") YASAK -- her
# metrik nokta tahmini + %95 CI ile verilir.
# UYARI: model ADI != checkpoint DOSYA ADI (ZORUNLU-BEYANLAR.md D1/D2).
# ---------------------------------------------------------------------
FINAL_MODEL_NOTE = (
    "Nihai Cox PHM modeli: v3b_lowvar_v2amgmt (Baris karari, 2026-09-12). "
    "WT-only; ICC>=0,60 ile 93 kararli ozellik -> v3 dusuk-varyans/"
    "kolinearite filtresi -> 54 radyomik aday -> elastic net + stabilite "
    "secimi -> NIHAI UZAY 16 radyomik + 8 klinik = 24 ozellik. "
    "Egitim: UPenn 611 hasta / 585 olum olayi (EPV 9,435; aday havuzu "
    "585/62). Harici test: UCSF-PDGM 295 hasta / 169 olay -- Harrell "
    "C = 0,6592 [0,6164-0,7020]; ic CV 0,6671 +/- 0,0308. "
    "v3b 'en iyi model' DEGILDIR: 8 kolun harici C guven araliklari "
    "TAMAMEN ortusuyor (istatistiksel olarak ayirt edilemez); secim "
    "ikincil olcutlerle yapildi (EPV, katsayi patlamasi yok, sayisal "
    "kararlilik, yalinlik). Model ADI ile servis edilen CHECKPOINT "
    "DOSYASI (models/cox_phm_v3b_lowvar_v2amgmt_2026-09-12.pkl) AYNI SEY "
    "DEGILDIR; degerlendirme modeli / dagitim modeli ayrimi icin bkz. "
    "ZORUNLU-BEYANLAR.md D1/D2. DB kaydi: model_registry'de "
    "cox_phm/v3b_lowvar_v2amgmt satiri VARDIR, ama status='shadow' -- "
    "hicbir Cox/XGBoost satiri 'production' DEGILDIR (canliya alma ayri "
    "bir fazdir), bu yuzden D2'nin ozu gecerlidir."
)
# ~~Eski/GECERSIZ metin (2026-09-14'te duzeltildi -- belge-agent-Y2):
#   "... degerlendirme modeli / dagitim modeli ayrimi ve
#    model_registry'nin hala BOS oldugu icin bkz.
#    ZORUNLU-BEYANLAR.md D1/D2."~~
# Bu cumle CANLI DB ile CELISIYORDU ve juriye GORUNUYOR
# (streamlit_app.py:183 -- Cox model kolu secicisinin help metni).
# Canli olcum 2026-09-14 (readonly SELECT):
#   SELECT model_id, model_name, version, status FROM model_registry;
#   -> 4 satir: 7 cox_phm/v3b_lowvar_v2amgmt = shadow
#               8 nnunet                     = shadow
#               9 omics_scorer/v1.0          = production
#              10 xgboost/...bitidentical... = shadow
# Tablo 2026-09-12/13'te dolduruldu; dogru beyan "kayit VAR ama
# shadow"dur. Eski metin wiki hard rule #3 geregi SILINMEDI, burada
# isaretli olarak korunuyor.

# ---------------------------------------------------------------------
# Blok C (buyume simulasyonu) -- ZORUNLU KAPSAM BEYANI
# Sayilar OLCULDU (2026-09-13, canli DB + canli kod, salt-okunur):
# `artifacts/week6/demo_blok_c_projection_scan_2026-09-13.{csv,json}`.
# LUMIERE longitudinal C32 WT serisi: 90 hasta -> egri fit'i "ok" olan
# 76 -> projeksiyon URETILEBILEN 67 (yalniz PD grubu; CR n=1 / PR n=3 /
# SD n=5 `interpretable=False` oldugu icin projeksiyon almaz).
# "Yorumlanabilir" TANIMI (bu demoda kullanilan, UC kosul birlikte):
#   (i)  ust uc FIZIKSEL olarak mumkun: v_high <= 1.400.000 mm3
#        (yetiskin kafa-ici hacmi ~1,4 L -- tumor kafatasini asamaz),
#   (ii) alt uc sayisal yok-olma DEGIL: pct_low > -%99,9,
#   (iii) bant genisligi <= 200 yuzde puani.
# SONUC: 67 PD hastasinin 14'u. Esik keyfi DEGIL: olculen bant
# genisligi dagiliminda 173,90 pp ile 286,04 pp arasinda GERCEK bir
# bosluk var, yani (174, 286] araligindaki HER esik AYNI 14 hastayi
# verir. Duyarlilik (ayni tarama): 50 pp -> 9, 100 pp -> 12, 200 pp ->
# 14, 300 pp -> 15, 500/1000 pp -> 16; yalniz (i)+(ii) -> 16.
# ⚠️ Yalniz-genislik (<=200 pp) olcutu 15 verir (belge-agent-G'nin
# 2026-09-13 sayisi); fark TEK hastadir -- `Patient-052`, alt ucu
# -%100,00 oldugu icin (ii)'yi gecemiyor. Iki sayi da dogru, olcutu
# YAZMADAN kullanilmaz.
#
# 🔴 YUKARIDAKI BLOK 2026-09-13 AKSAMI GECERSIZLESTI (G1 duzeltmesi).
# Tarama YENIDEN kosuldu (`...-v2-duzeltilmis.{csv,json}`, 90/90, 2282 s,
# ayni canli yolla). DEGISMEYENLER: seride 90 hasta, fit `ok` 76,
# projeksiyon uretilen 67 (PD; CR 1 / PR 3 / SD 5), ust ucu kafa-ici
# hacmi asan 45/67. DEGISENLER:
#   * uc kosullu yorumlanabilir: 14 -> **12**
#   * alt ucu <= -%99,9 olan: 27 -> **29** (Patient-059 ve Patient-086
#     bu yuzden yorumlanabilir kumeden cikti; yeni giren YOK)
#   * yalniz (i)+(ii): 16 -> **14**
#   * yalniz bant-genisligi (<=200 pp): 15 -> **15** (DEGISMEDI, ama
#     UYELIK degisti)
# Esik duyarliligi (yalniz bant, YENI): 25 pp -> 0, 50 -> 2, 100 -> 9,
#   200 -> 15, 300 -> 15, 500 -> 17, 1000 -> 19.
# Esik duyarliligi (uc kosul, YENI): 25 -> 0, 50 -> 2, 100 -> 9,
#   200 -> 12, 300 -> 12, 500 -> 13, 1000 -> 14.
# Bant dagilimindaki GERCEK BOSLUK HALA VAR ama yeri kaydi:
#   eski 173,90 -> 286,04 (112,13 pp) · YENI **173,90 -> 302,76
#   (128,86 pp)** -> (174, 302] araligindaki HER esik AYNI 15 hastayi
#   verir, 200 pp esigi savunulabilir KALIYOR.
# 🔴 YENI BULGU (eski taramada beyan edilmemisti): 67 PD hastasinin
#   **7'sinde** projeksiyonun UST UCU BILE NEGATIF (Patient-009, -037,
#   -040, -043, -081, -085, -090) -- yani "progresif hastalik
#   grubundaki hasta, tumoru KESIN kuculuyor". Bu, RANO etiketi ile
#   volumetrik yorunge arasindaki uyusmazligin dogrudan olcumu (bkz.
#   ZORUNLU-BEYANLAR C4: volumetrik, resmi RANO degil) ve secim
#   olcutune 2. basamagin (klinik tutarlilik) eklenme sebebidir.
# ---------------------------------------------------------------------

GROWTH_PROJECTION_COVERAGE_NOTE = (
    "KAPSAM BEYANI (olculdu, 2026-09-13 v2 -- projeksiyon ufku "
    "duzeltmesinden SONRA): Projeksiyon araligi 67 PD hastasinin "
    "12'sinde yorumlanabilir; geri kalan 55'inde model uyumu araligi "
    "fiziksel olarak anlamsiz kiliyor (67 hastanin 45'inde ust uc "
    "yetiskin kafa-ici hacmini -- 1,4 L -- asiyor, 29'unda alt uc "
    "-%99,9'un altina dusuyor). AYRICA 67'nin 7'sinde projeksiyonun UST "
    "UCU BILE NEGATIF, yani PD (progresif) grubundaki hasta icin "
    "'tumor kesin kuculuyor' diyor -- bu hastalar demo icin "
    "SAVUNULAMAZ ve secim olcutunun 2. basamagiyle (klinik tutarlilik) "
    "elenir. Bu hasta 12 arasindan SECILDI; olcut DORT basamakli: "
    "(1) uc kosullu yorumlanabilirlik, (2) RANO PD ise ust ucun POZITIF "
    "olmasi, (3) makul seri uzunlugu/son ziyaret haftasi, (4) bunlari "
    "gecenler (5 hasta) arasinda en dar bant. UYARI: bu hasta TUM "
    "yorumlanabilirler arasinda en dar bant DEGILDIR (en dar "
    "`Patient-009`, 47,39 yuzde puani; bu hasta 53,12 pp ile 4.) -- "
    "olcut 2026-09-13'te ACIKCA degistirildi, 'en dar bant' ifadesi "
    "artik tek basina kullanilmaz. Yalniz bant-genisligi olcutuyle "
    "(<=200 yuzde puani) sayi 15/67'dir. Olcum: artifacts/week6/"
    "demo_blok_c_projection_scan_2026-09-13-v2-duzeltilmis.csv (90 "
    "LUMIERE hastasinin TAMAMI yeniden tarandi). Eski/gecersiz sayilar "
    "(14/67, 'en dar bant 16,24 pp') ayni klasordeki v1 CSV'de "
    "tarihsel kayit olarak DURUYOR. Beyan: ZORUNLU-BEYANLAR.md D8."
)

GROWTH_PROJECTION_SEMANTICS_NOTE = (
    "ARALIGIN ANLAMI -- yanlis okunmamasi icin: bu aralik hastanin KENDI "
    "buyume katsayisinin ileriye tasinmasi DEGILDIR. pipeline/"
    "growth_simulation.py::project_volume_range() hastanin fit edilmis "
    "egri sablonunu (Gompertz A,B / lojistik K,t0) SABIT tutup yalniz "
    "buyume katsayisi r'yi hastanin RANO grubunun (PD, n=67) r-IQR'sinin "
    "alt/medyan/ust ceyregiyle DEGISTIRIR; aralik bu uc senaryonun "
    "hacimleridir. Ufuk 2026-09-13'te DUZELTILDI: artik "
    "t = son_ziyaret_haftasi + 6 x 4,345 olarak hesaplanir; bu hasta "
    "icin son ziyaret hafta 38, hedef hafta 38 + 26,07 = 64,07. Yani "
    "karsilastirma 'hafta 38 hacmi' ile 'egrinin hafta 64,07'deki "
    "degeri' arasindadir -- ILERIYE donuk. (ESKI/HATALI hali: ufuk "
    "kosulsuz 26,07 haftaydi, hastanin ILK taramasindan itibaren MUTLAK; "
    "son ziyaret 38. haftada oldugu icin karsilastirma GERIYE DONUKTU ve "
    "alt uc +%32,96 cikiyordu -- duzeltilmis hali -%0,47, ISARET "
    "DEGISIYOR.) Buna ragmen 'bu tumor 6 ay sonra %44 buyur' cumlesi "
    "KURULMAZ; kurulabilecek cumle: 'bu hastanin egri sablonu, kendi "
    "RANO grubunun buyume-hizi araliginda calistirildiginda hacim "
    "-%0,5 ile +%53 bandinda cikiyor, medyan senaryo +%44'. Bandin dar "
    "olmasi biyolojik kesinlik DEGILDIR: Gompertz sablonu (a=173.999,7 "
    "b=0,0721) asimptota yakindir, bu yuzden uc senaryo da platoya "
    "yakin hacimler verir. Hastanin KENDI fit edilmis r'si ayrica "
    "NEGATIFTIR (-0,051261) -- pozitif medyan grup r-IQR'sinden gelir, "
    "hastanin kendi yorungesinden DEGIL. Bu bir SIMULASYONDUR, tahmin "
    "DEGILDIR."
)

BLOCK_C_SELECTION_NOTE = (
    "Blok C hastasi 2026-09-13'te `Patient-002`'den `Patient-028`'e "
    "CEVRILDI (demo-agent-B). Sebep olculdu: `Patient-002`'nin ayni "
    "hesapla cikan 6 aylik araligi -%99,99999999 .. +%3.305.185,28 "
    "(ust uc ~1,9 milyar mm3) -- dejenere, ekranda gosterilemez. Ufku "
    "kisaltmak COZMUYOR (3 ay +%40.960, 1 ay +%709 -- olculdu). "
    "`Patient-002` verisi SILINMEDI, tarihsel kayit olarak "
    "ZORUNLU-BEYANLAR.md D7/D8 ve ayni CSV'de duruyor. "
    "GUNCELLEME (2026-09-13 aksami, G1): projeksiyon ufku hatasi "
    "duzeltildikten sonra 90 hastalik tarama YENIDEN kosuldu ve secim "
    "olcutu dort basamakli hale getirildi (yorumlanabilirlik + RANO-PD "
    "ile yon tutarliligi + makul seri uzunlugu + en dar bant). Hasta "
    "`Patient-028` olarak KALDI -- ama artik 'en dar bant' oldugu icin "
    "degil, olcutu gecen 5 aday arasinda en dar oldugu icin. Araligi "
    "-%0,47 .. +%43,97 .. +%52,64 (eski/gecersiz: +%32,96 .. +%49,20). "
    "Detay: decisions/2026-09-13-projeksiyon-ufku-duzeltmesi.md."
)


# ---------------------------------------------------------------------
# K12: FAISS indeks yollari icin ASCII (8.3 kisa yol) override'i
# ---------------------------------------------------------------------


def _win_short_path(path: Path) -> str | None:
    """Windows 8.3 kisa yolunu (ASCII) dondurur; uretilemezse None."""

    if os.name != "nt":
        return None
    try:
        buf = ctypes.create_unicode_buffer(1024)
        n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 1024)
    except Exception:
        return None
    if not n:
        return None
    short = buf.value
    return short if short.isascii() else None


def setup_faiss_ascii_paths() -> dict[str, Any]:
    """`GBMAID_FAISS_*_INDEX_DIR` env override'larini ASCII kisa yolla
    doldurur (yalniz yol ASCII DEGILSE ve override zaten yoksa).

    api/similar.py DEGISTIRILMEDEN K12 Turkce-yol sorununu asar.
    Donen sozluk paneller icin durum notu tasir.
    """

    targets = {
        "GBMAID_FAISS_CLINICAL_RADIOMICS_INDEX_DIR": (
            REPO_ROOT / "artifacts" / "week3" / "faiss_index"
        ),
        "GBMAID_FAISS_MOLECULAR_OMICS_INDEX_DIR": (
            REPO_ROOT / "artifacts" / "week3" / "molecular_omics"
        ),
    }
    report: dict[str, Any] = {"ok": True, "notes": []}
    for env_name, default_dir in targets.items():
        if os.environ.get(env_name):
            report["notes"].append(f"{env_name} zaten disaridan ayarli, dokunulmadi.")
            continue
        if str(default_dir).isascii():
            report["notes"].append(f"{env_name}: varsayilan yol zaten ASCII, override gereksiz.")
            continue
        short = _win_short_path(default_dir)
        if short is None:
            report["ok"] = False
            report["notes"].append(
                f"{env_name}: 8.3 kisa yol uretilemedi -- FAISS paneli "
                "Turkce-yol nedeniyle dusebilir. Cozum: depoyu `subst X: ...` "
                "ile ASCII bir surucuye bagla ve demoyu oradan baslat (K12)."
            )
            continue
        os.environ[env_name] = short
        report["notes"].append(f"{env_name} = {short} (8.3 kisa yol, K12 cozumu).")
    return report


# ---------------------------------------------------------------------
# Panel 1 -- hasta ozeti (patients + dataset_sources, readonly SELECT)
# ---------------------------------------------------------------------


def fetch_patient_summary(patient_id: str) -> dict[str, Any] | None:
    """`patients` satirini kaynak adiyla dondurur; hasta yoksa None."""

    from db_connection import get_connection

    conn = get_connection(readonly=True)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.patient_id, ds.source_name, p.age, p.gender,
                   p.kps_score, p.diagnosis_type, p.vital_status,
                   p.survival_days, p.mgmt_status, p.idh1_status,
                   p.gtr_over90percent, p.has_mr, p.has_omics
            FROM patients p
            LEFT JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE p.patient_id = %s
            """,
            (patient_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))
    finally:
        conn.close()


# ---------------------------------------------------------------------
# Panel 2+3 -- Cox risk + SHAP (api/predict.py process-ici)
# ---------------------------------------------------------------------

# os.environ mutasyonu process-genelinde gorunur; Streamlit oturumlari
# ayni process'te thread olarak kosabilir. Fallback cagrisi kisa surdugu
# ve demo tek-kullanicili oldugu icin kilitle korumak yeterli.
_predict_env_lock = threading.Lock()


def _classify_http_error(exc: Exception) -> tuple[int | None, str]:
    status = getattr(exc, "status_code", None)
    detail = getattr(exc, "detail", None) or str(exc)
    return status, str(detail)


def predict_with_fallback(patient_id: str, mode: str = "auto") -> dict[str, Any]:
    """`api.predict.predict_patient`'i process-ici cagirir.

    mode:
      "auto"      -- once varsayilan artefakt (asagidaki NOT'a bak); klinik
                     kovaryat eksikligi (422) halinde yalniz-radyomik
                     birincil kola SEFFAF fallback.
      "default"   -- yalniz varsayilan artefakt (fallback yok).
      "radiomics" -- dogrudan yalniz-radyomik kol.

    NOT -- VARSAYILAN ARTEFAKT v3b'DIR, v1 DEGIL (2026-09-13, karar 33)
    ------------------------------------------------------------------
    ~~Eski dokstring: "once varsayilan artefakt (v1 klinik_full)"~~
    (ustu cizildi, SILINMEDI -- wiki hard rule #3: celiski silinmez,
    isaretlenir.)

    Bu satir BAYATTI. Karar 33 ile `api/predict.py`'deki
    `DEFAULT_CHECKPOINT_PATH` artik
    `models/cox_phm_v3b_lowvar_v2amgmt_2026-09-12.pkl`'dir
    (`api/predict.py:377`). Dogrulandi (2026-09-13): env override
    (`GBMAID_COX_CHECKPOINT_PATH`) VERILMEDEN v3b yuklenir;
    `UCSF-PDGM-167` icin risk skoru -0.7096787591025546.

    ISLEVSEL ETKI YOKTU: bu fonksiyon artefakt adini hicbir yerde sabit
    yazmaz, `_call_with_checkpoint(None)` ile `api/predict.py`'nin kendi
    varsayilanini kullanir -- yani kod v3b'yi zaten yukluyordu, yalniz
    dokstring v1 diyordu.

    Donen sozluk: {"status": "ok"|"error", "result": ..., "arm_source":
    "default"|"radiomics_fallback"|"radiomics_forced", "fallback_note",
    "http_status", "error_detail"}.
    """

    from api.predict import predict_patient

    def _call_with_checkpoint(checkpoint: Path | None) -> dict[str, Any]:
        with _predict_env_lock:
            previous = os.environ.get("GBMAID_COX_CHECKPOINT_PATH")
            try:
                if checkpoint is not None:
                    os.environ["GBMAID_COX_CHECKPOINT_PATH"] = str(checkpoint)
                return predict_patient(patient_id)
            finally:
                if checkpoint is not None:
                    if previous is None:
                        os.environ.pop("GBMAID_COX_CHECKPOINT_PATH", None)
                    else:
                        os.environ["GBMAID_COX_CHECKPOINT_PATH"] = previous

    if mode == "radiomics":
        try:
            result = _call_with_checkpoint(RADIOMICS_ONLY_CHECKPOINT)
            return {
                "status": "ok",
                "result": result,
                "arm_source": "radiomics_forced",
                "fallback_note": None,
            }
        except Exception as exc:
            status, detail = _classify_http_error(exc)
            return {"status": "error", "http_status": status, "error_detail": detail}

    try:
        result = _call_with_checkpoint(None)
        return {
            "status": "ok",
            "result": result,
            "arm_source": "default",
            "fallback_note": None,
        }
    except Exception as exc:
        status, detail = _classify_http_error(exc)
        clinical_gap = status == 422 and "klinik kovaryat" in detail
        if mode == "auto" and clinical_gap and RADIOMICS_ONLY_CHECKPOINT.is_file():
            try:
                result = _call_with_checkpoint(RADIOMICS_ONLY_CHECKPOINT)
                return {
                    "status": "ok",
                    "result": result,
                    "arm_source": "radiomics_fallback",
                    "fallback_note": (
                        "Nihai model v3b_lowvar_v2amgmt (16 radyomik + 8 "
                        "klinik = 24 ozellik) bu hastada kullanilamadi -- "
                        f"api/predict.py 422 dondu: {detail} Bunun yerine "
                        "ESKI YALNIZ-RADYOMIK kol "
                        "(primary_wt93_icc60_th06, 2026-08-15 kosusu, harici "
                        "C-index 0,689 [0,546-0,824], TCGA n=38) kullanildi. "
                        "Bu SEFFAF bir fallback'tir; sessiz varsayilan atama "
                        "YAPILMADI. DIKKAT: bu kol NIHAI MODEL DEGILDIR -- "
                        "buradaki risk/HR degerleri sunumda nihai model "
                        "sonucu olarak KULLANILMAZ."
                    ),
                }
            except Exception as exc2:
                status2, detail2 = _classify_http_error(exc2)
                return {
                    "status": "error",
                    "http_status": status2,
                    "error_detail": (
                        f"Varsayilan model 422 verdi ({detail}); yalniz-"
                        f"radyomik fallback de basarisiz: {detail2}"
                    ),
                }
        return {"status": "error", "http_status": status, "error_detail": detail}


# ---------------------------------------------------------------------
# Panel 4 -- benzer hastalar (api/similar.py process-ici, iki FAISS katmani)
# ---------------------------------------------------------------------


def similar_patients(patient_id: str, k: int = 5) -> dict[str, Any]:
    """`api.similar.get_similar_patients`'i process-ici cagirir.

    NOT: fonksiyon imzasindaki varsayilanlar FastAPI `Query` nesneleri
    oldugu icin TUM parametreler ACIKCA verilir.
    """

    faiss_env = setup_faiss_ascii_paths()

    from api.similar import get_similar_patients

    try:
        result = get_similar_patients(
            patient_id, k=k, idh1_status=None, mgmt_status=None
        )
        return {"status": "ok", "result": result, "faiss_env": faiss_env}
    except Exception as exc:
        status, detail = _classify_http_error(exc)
        return {
            "status": "error",
            "http_status": status,
            "error_detail": detail,
            "faiss_env": faiss_env,
        }


# ---------------------------------------------------------------------
# Panel 5 -- omics (api/predict.py::fetch_omics_interpretation)
# ---------------------------------------------------------------------


def omics_interpretation(patient_id: str) -> dict[str, Any]:
    from api.predict import fetch_omics_interpretation

    try:
        block = fetch_omics_interpretation(patient_id)
    except Exception as exc:
        return {"status": "error", "error_detail": str(exc)}
    if block is None:
        return {"status": "none"}
    return {"status": "ok", "result": block}


# ---------------------------------------------------------------------
# Panel 7 -- buyume simulasyonu (Blok C, LUMIERE longitudinal)
# ---------------------------------------------------------------------


def growth_simulation(patient_id: str) -> dict[str, Any]:
    """`api/analyze_patient.py::_build_growth_simulation_block()`'i
    process-ici cagirir -- mantik BURADA YENIDEN YAZILMADI, o dosyaya da
    (db-agent/backend-agent kilidi) YAZILMADI, yalniz import edildi.

    Neden `analyze_patient()` (tam orkestrasyon) DEGIL: tam orkestrasyon
    ayni istekte Cox + FAISS + RAG'i de kosar; bu paneller demoda ZATEN
    ayri ayri cagriliyor, tekrar kosmak gereksiz. Buyume blogu, uctan uca
    `POST /analyze_patient` yanitinda da AYNI fonksiyonla uretilir
    (2026-09-13 B-1 duzeltmesinden sonra Cox 422 alsa bile istek 200
    doner ve bu blok dolu gelir -- canli dogrulandi).

    Donen: {"status": "ok", "block": {...}} veya {"status": "error", ...}.
    """

    from api.analyze_patient import _build_growth_simulation_block

    try:
        block = _build_growth_simulation_block(patient_id)
    except Exception as exc:
        return {"status": "error", "error_detail": f"{type(exc).__name__}: {exc}"}
    return {"status": "ok", "block": block}


# ---------------------------------------------------------------------
# Panel 6 -- literatur (pipeline/rag_pipeline.py, LLM bilincli KAPALI)
# ---------------------------------------------------------------------


def literature_summary(
    patient_id: str, retmax: int = 10, top_k: int = 5
) -> dict[str, Any]:
    """`generate_literature_summary` sonucunu cache'lenebilir sade bir
    sozluge indirger (dataclass -> dict; Streamlit cache uyumu)."""

    from pipeline.rag_pipeline import generate_literature_summary

    try:
        r = generate_literature_summary(patient_id, retmax=retmax, top_k=top_k)
    except Exception as exc:
        return {"status": "error", "error_detail": repr(exc)}

    out: dict[str, Any] = {
        "status": "ok",
        "available": r.available,
        "reason": r.reason,
        "used_cache": r.used_cache,
        "sanitization_violation": r.sanitization_violation,
        "llm_summary_available": r.llm_summary_available,
        "llm_reason": r.llm_reason,
        "notes": list(r.notes),
        "retmax": retmax,
        "top_k": top_k,
        "boolean_query": None,
        "semantic_query": None,
        "chunks": [],
    }
    if r.query is not None:
        out["boolean_query"] = r.query.pubmed_boolean_query
        out["semantic_query"] = r.query.semantic_query_text
    for c in r.retrieved_chunks:
        out["chunks"].append(
            {
                "pmid": c.pmid,
                "title": c.title,
                "snippet": (c.chunk_text[:280] + "...")
                if len(c.chunk_text) > 280
                else c.chunk_text,
            }
        )
    return out


# ---------------------------------------------------------------------
# Panel 9 -- MR kesiti + hazir segmentasyon overlay'i (2026-09-14, SIK A)
# ---------------------------------------------------------------------
#
# KAPSAM (Baris karari 2026-09-14, "SIK A"): YALNIZ GOSTERIM.
#   * Yeni hasta yukleme YOK.
#   * Goruntuden CANLI radyomik cikarim YOK -- panelde gosterilen risk/
#     SHAP sayilari DB'deki C32 satirlarindan gelir, bu goruntuden
#     TUREMEZ (Sik B/C reddedildi).
#   * NAS'a ve yerel diske YALNIZ OKUMA.
#
# GORUNTU/MASKE CIFTI NASIL BULUNUR -- kendi mantigimiz YAZILMADI:
#   `pipeline/segmentation.py::resolve_nas_path()` + `resolve_ready_mask()`
#   DOGRUDAN cagrilir (LUMIERE/UPenn/TCGA). Boylece ekranda gorulen maske,
#   C32 radyomik cikariminin kullandigi maskenin TA KENDISIDIR.
#
# KOORDINATOR PREMISI DUZELTILDI (olculdu 2026-09-14, canli NAS):
#   Gorev talimati "maske ATLAS uzayinda (182x218x182), ham CT1 NATIVE
#   uzayda (256x256x192) -- overlay icin `atlas/skull_strip/ct1_skull_
#   strip.nii.gz` kullanilmali" diyordu. Bu, LUMIERE'in ATLAS dali icin
#   dogru ama PROJENIN KULLANDIGI DAL O DEGIL: `_lumiere_mask()` ANA
#   kaynak olarak `DeepBraTumIA-segmentation/native/segmentation/
#   ct1_seg_mask.nii.gz`'i secer ve onu ham `week-XXX/CT1.nii.gz` ile
#   eslestirir. Canli olcum (Patient-028/week-038): her ikisi de
#   (256, 256, 160), 1,0 mm izotropik, `geometry_match=True`. Yani
#   atlas/skull_strip'e GEREK YOK ve kullanilmasi radyomikle
#   TUTARSIZ bir maske gosterirdi. Atlas tuzagina hic girilmiyor.
#
# IKINCI TUZAK (talimatda YOK, burada olculdu): NIfTI eksen duzeni
#   kohortlar arasi FARKLI -- LUMIERE ('P','I','R'), TCGA ve UCSF
#   ('L','P','S'). Sabit "axis=2 eksenel kesittir" varsayimi LUMIERE'de
#   KORONAL/SAGITAL kesit uretirdi. Cozum: `nib.as_closest_canonical()`
#   (yalniz eksen permutasyonu/cevirme -- interpolasyon YOK, voksel
#   degeri DEGISMEZ) ile her iki hacim de RAS'a getirilir; RAS'ta axis=2
#   her zaman inferior->superior, yani eksenel kesit eksenidir.
#
# UCUNCU TUZAK: SimpleITK, Turkce karakterli yollari ACAMIYOR (K12'nin
#   ayni sinifi; canli dogrulandi -- UCSF goruntuleri Turkce karakterli
#   bir yerel yol altinda ve `sitk.ReadImage` "Unable to open" firlatiyor).
#   Bu yuzden bu panel SADECE `nibabel` kullanir (Python seviyesinde
#   `open()`), `pipeline/resampling.py::validate_image_mask_geometry()`
#   (SimpleITK tabanli) burada CAGRILMAZ; yerine gosterim-seviyesi
#   shape+affine esitligi kontrolu yapilir (`MrOverlayGeometryError`).
#
# UCSF ISTISNASI: `resolve_ready_mask()` UCSF'i TANIMIYOR ("Hazir
#   maske sozlesmesi tanimsiz kaynak: UCSF") cunku UCSF pipeline'a
#   2026-08-18'de ayri bir arac (`tools/run_pyradiomics_ucsf.py`) ile
#   girdi. `pipeline/`'a DOKUNULMADIGI icin (gorev kisiti) UCSF dali
#   BURADA, o aracin sozlesmesi BIREBIR taklit edilerek cozulur:
#   maske = goruntuyle AYNI klasorde `<ID>_tumor_segmentation.nii.gz`
#   (bkz. tools/run_pyradiomics_ucsf.py:479 `mask_path = disk_base /
#   folder / seg_file`, MASK_SOURCE = "ucsf_native").
#   ACIK BORC: bu, `pipeline/segmentation.py`'de olmasi gereken bir
#   dalin demo-seviyesinde kopyasidir. Kalici cozum o dosyaya bir
#   `_ucsf_mask()` eklemektir (imaging-agent/backend-agent isi).

MR_VIEWER_DISCLAIMER = (
    "Görüntü NAS'tan salt-okunur okunmaktadır. Radyomik özellikler bu "
    "görüntüden **canlı çıkarılmamaktadır**; C32 sözleşmesiyle önceden "
    "çıkarılmış ve veri tabanında saklanan değerlerdir. Segmentasyon "
    "maskesi **hazır** maskedir (DeepBraTumIA), bu oturumda "
    "üretilmemiştir."
)

# Gosterim yonu: RADYOLOJIK (goruntunun solu = hastanin SAGI).
MR_VIEWER_ORIENTATION_NOTE = (
    "Eksenel kesit, radyolojik yön: görüntünün SOLU = hastanın SAĞI; "
    "yukarı = anterior. Hacimler `nibabel.as_closest_canonical()` ile "
    "RAS'a çevrilir (yalnız eksen permütasyonu/çevirme — interpolasyon "
    "YOK, voksel değeri değişmez)."
)

# Bolge -> renk (dataviz koyu-tema paleti). Anahtarlar `pipeline/
# radiomics_volume.py::REGION_LABELS_BY_MASK_SOURCE`'un bolge ADLARIDIR
# -- etiket SAYILARI orada tanimli, burada TEKRAR YAZILMAZ.
MR_REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "Necrosis": (230, 103, 103),
    "NC": (230, 103, 103),
    "Contrast-enhancing": (245, 199, 68),
    "ET": (245, 199, 68),
    "Edema": (57, 135, 229),
    "ED": (57, 135, 229),
    "Non-enhancing": (156, 122, 214),
    "WT": (230, 103, 103),
    "TC": (245, 199, 68),
}
_MR_FALLBACK_COLOR = (200, 200, 200)


class MrOverlayGeometryError(RuntimeError):
    """Goruntu ve maske ayni uzayda DEGIL -- overlay uretilemez.

    Bilincli olarak SESSIZ GECILMEZ: yanlis hizalanmis bir overlay,
    juriye tumoru YANLIS YERDE gosterir. Panel bu hatayi gorunur
    kilar (bkz. `load_mr_overlay_volume()`'un `available: False` +
    `geometry_mismatch: True` blogu).
    """


def _mr_visit_label(file_path: str) -> str:
    """Yol icinden insana okunur ziyaret etiketi cikar (LUMIERE: week-XXX)."""

    parts = [p for p in str(file_path).replace("\\", "/").split("/") if p]
    for part in parts:
        if part.lower().startswith("week-"):
            return part
    return "tek zaman noktasi"


def list_mr_t1ce_scans(patient_id: str) -> dict[str, Any]:
    """Hastanin T1ce taramalarini (readonly SELECT) ziyaret sirasiyla dondurur.

    `api/analyze_patient.py`'nin graceful-degradation desenini taklit eder:
    HICBIR kosulda exception firlatmaz, `available: False` + gorunur
    `not_available_reason` doner.
    """

    try:
        from db_connection import get_connection

        conn = get_connection(readonly=True)
    except Exception as exc:  # savunmaci -- DB erisilemezse panel DURMAZ
        return {
            "available": False,
            "not_available_reason": (
                f"DB baglantisi kurulamadi: {type(exc).__name__}: {exc}"
            ),
        }

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT s.scan_id, s.file_path, ds.source_name
            FROM mr_scans s
            LEFT JOIN patients p ON p.patient_id = s.patient_id
            LEFT JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE s.patient_id = %s AND s.modality = 'T1ce'
            ORDER BY s.scan_id
            """,
            (patient_id,),
        )
        rows = cur.fetchall()
    except Exception as exc:  # savunmaci
        return {
            "available": False,
            "not_available_reason": (
                f"mr_scans okunamadi: {type(exc).__name__}: {exc}"
            ),
        }
    finally:
        conn.close()

    if not rows:
        return {
            "available": False,
            "not_available_reason": (
                f"'{patient_id}' icin `mr_scans` tablosunda T1ce satiri YOK "
                "-- bu hastanin MR goruntusu sistemde kayitli degil."
            ),
        }

    scans = [
        {
            "scan_id": int(r[0]),
            "file_path": str(r[1]),
            "source_name": str(r[2]) if r[2] else "",
            "visit_label": _mr_visit_label(str(r[1])),
        }
        for r in rows
    ]
    return {"available": True, "scans": scans}


def _resolve_ucsf_display_pair(image_path: Path) -> tuple[Path, str]:
    """UCSF icin (maske, mask_source). `tools/run_pyradiomics_ucsf.py`
    sozlesmesinin BIREBIR kopyasi -- bkz. bu bolumun basindaki
    "UCSF ISTISNASI" notu."""

    from pipeline.harmonization import _nifti_stem

    stem = _nifti_stem(image_path)
    if not stem.endswith("_T1c"):
        raise FileNotFoundError(
            f"UCSF T1c dosya adi beklenen desende degil ('*_T1c'): {image_path}"
        )
    base = stem[: -len("_T1c")]
    for suffix in (".nii.gz", ".nii"):
        candidate = image_path.parent / f"{base}_tumor_segmentation{suffix}"
        if candidate.is_file():
            return candidate, "ucsf_native"
    raise FileNotFoundError(
        "UCSF hazir tumor maskesi bulunamadi: "
        f"{image_path.parent / (base + '_tumor_segmentation.nii.gz')}"
    )


def _resolve_display_pair(
    *, source_name: str, file_path: str, patient_id: str, scan_id: int
) -> dict[str, Any]:
    """(goruntu, maske, mask_source, uyarilar) -- kaynak sozlesmesine gore."""

    from pipeline.harmonization import canonical_source
    from pipeline.segmentation import resolve_nas_path, resolve_ready_mask

    canonical = canonical_source(source_name)
    image_path = resolve_nas_path(file_path)

    if canonical == "UCSF":
        mask_path, mask_source = _resolve_ucsf_display_pair(image_path)
        return {
            "image_path": image_path,
            "mask_path": mask_path,
            "mask_source": mask_source,
            "source": canonical,
            "warnings": [],
        }

    info = resolve_ready_mask(
        source=source_name,
        image_path=image_path,
        patient_id=patient_id,
        scan_id=scan_id,
        # Gosterim-seviyesi kontrol asagida nibabel ile yapilir; buradaki
        # SimpleITK tabanli dogrulayiciya Turkce-yol riski nedeniyle
        # GUVENILMEZ (bkz. "UCUNCU TUZAK").
        raise_on_geometry_mismatch=False,
    )
    return {
        "image_path": Path(str(info["image_path"])),
        "mask_path": Path(str(info["mask_path"])),
        "mask_source": str(info["mask_source"]),
        "source": canonical,
        "warnings": list(info["warnings"]),
    }


def load_mr_overlay_volume(patient_id: str, scan_id: int) -> dict[str, Any]:
    """Bir taramanin T1ce hacmini + hazir maskesini yukler, eksenel kesit
    icin hazirlar. HICBIR kosulda exception firlatmaz (fail-closed).

    Donen sozluk `available: True` ise su alanlari tasir:
      `img_u8`/`seg_u8` (RAS'a cevrilmis, shape=(X,Y,Z)), `slice_counts`
      (z basina tumor voksel sayisi), `best_slice` (EN GENIS kesit),
      `tumor_voxels`, `tumor_volume_mm3`, `region_labels`, `mask_source`.
    """

    import numpy as np

    scans = list_mr_t1ce_scans(patient_id)
    if not scans["available"]:
        return scans
    match = next((s for s in scans["scans"] if s["scan_id"] == int(scan_id)), None)
    if match is None:
        return {
            "available": False,
            "not_available_reason": (
                f"scan_id={scan_id} bu hastanin T1ce taramalari arasinda yok."
            ),
        }

    try:
        pair = _resolve_display_pair(
            source_name=match["source_name"],
            file_path=match["file_path"],
            patient_id=patient_id,
            scan_id=int(scan_id),
        )
    except Exception as exc:  # savunmaci -- NAS kapali/dosya yok/kaynak tanimsiz
        return {
            "available": False,
            "not_available_reason": (
                "MR görüntüsü şu an erişilemiyor: NAS bağlantısı yok veya "
                f"dosya çözümlenemedi ({type(exc).__name__}: {exc})"
            ),
        }

    try:
        import nibabel as nib

        raw_img = nib.load(str(pair["image_path"]))
        raw_seg = nib.load(str(pair["mask_path"]))
        original_axcodes = "".join(nib.aff2axcodes(raw_img.affine))
        can_img = nib.as_closest_canonical(raw_img)
        can_seg = nib.as_closest_canonical(raw_seg)
        arr = np.asanyarray(can_img.dataobj, dtype=np.float32)
        seg = np.asanyarray(can_seg.dataobj)
    except Exception as exc:  # savunmaci -- okuma/ag hatasi paneli COKERTMEZ
        return {
            "available": False,
            "not_available_reason": (
                "MR görüntüsü şu an erişilemiyor: NAS bağlantısı yok veya "
                f"dosya okunamadı ({type(exc).__name__}: {exc})"
            ),
        }

    # ZORUNLU GEOMETRI KONTROLU -- sessiz devam YASAK.
    if arr.shape != seg.shape or not np.allclose(
        can_img.affine, can_seg.affine, atol=1e-4
    ):
        err = MrOverlayGeometryError(
            f"Goruntu ve maske AYNI uzayda degil -- goruntu {arr.shape} "
            f"{pair['image_path']}, maske {seg.shape} {pair['mask_path']}. "
            "Overlay URETILMEZ (yanlis hizalanmis bir kesit juriye tumoru "
            "yanlis yerde gosterirdi)."
        )
        return {
            "available": False,
            "geometry_mismatch": True,
            "not_available_reason": str(err),
        }

    seg = seg.astype(np.int16, copy=False)
    tumor = seg > 0
    slice_counts = tumor.sum(axis=(0, 1)).astype(int)
    tumor_voxels = int(tumor.sum())
    if tumor_voxels == 0:
        return {
            "available": False,
            "not_available_reason": (
                f"Hazir maske ({pair['mask_source']}) bu taramada HIC tumor "
                "vokseli icermiyor -- gosterilecek lezyon yok."
            ),
        }
    best_slice = int(np.argmax(slice_counts))

    # Pencereleme: beyin vokselleri (>0) uzerinden %1-%99 persentil.
    brain = arr[arr > 0]
    lo, hi = (
        (float(np.percentile(brain, 1)), float(np.percentile(brain, 99)))
        if brain.size
        else (float(arr.min()), float(arr.max()))
    )
    if hi <= lo:
        lo, hi = float(arr.min()), float(max(arr.max(), arr.min() + 1.0))
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    img_u8 = (scaled * 255.0).round().astype(np.uint8)

    zooms = tuple(float(z) for z in can_img.header.get_zooms()[:3])
    voxel_mm3 = zooms[0] * zooms[1] * zooms[2]

    from pipeline.radiomics_volume import REGION_LABELS_BY_MASK_SOURCE

    region_labels = dict(REGION_LABELS_BY_MASK_SOURCE.get(pair["mask_source"], {}))
    present = {int(v) for v in np.unique(seg) if int(v) != 0}
    region_labels = {k: v for k, v in region_labels.items() if int(v) in present}

    return {
        "available": True,
        "patient_id": patient_id,
        "scan_id": int(scan_id),
        "source": pair["source"],
        "visit_label": match["visit_label"],
        "image_path": str(pair["image_path"]),
        "mask_path": str(pair["mask_path"]),
        "mask_source": pair["mask_source"],
        "warnings": list(pair["warnings"]),
        "shape": tuple(int(x) for x in arr.shape),
        "zooms_mm": zooms,
        "original_axcodes": original_axcodes,
        "img_u8": img_u8,
        "seg_u8": seg.astype(np.uint8, copy=False),
        "slice_counts": [int(c) for c in slice_counts],
        "best_slice": best_slice,
        "best_slice_voxels": int(slice_counts[best_slice]),
        "tumor_voxels": tumor_voxels,
        "voxel_volume_mm3": voxel_mm3,
        "tumor_volume_mm3": tumor_voxels * voxel_mm3,
        "region_labels": region_labels,
    }


def _mr_axial_slice(volume_array, slice_index: int):
    """RAS hacminden eksenel kesit -- RADYOLOJIK yon (sol = hastanin SAGI)."""

    import numpy as np

    plane = volume_array[:, :, int(slice_index)]
    # rot90: satirlar y'nin TERSI (anterior yukari), sutunlar x (L->R,
    # yani hastanin solu ekranin solunda = NOROLOJIK). `[:, ::-1]`
    # sutunlari cevirip RADYOLOJIK yone getirir.
    return np.ascontiguousarray(np.rot90(plane)[:, ::-1])


def render_mr_slice_png(
    volume: dict[str, Any],
    slice_index: int,
    *,
    with_overlay: bool = True,
    alpha: float = 0.45,
) -> bytes:
    """Tek eksenel kesiti PNG bayt dizisi olarak uretir (PIL; matplotlib YOK)."""

    import io

    import numpy as np
    from PIL import Image

    gray = _mr_axial_slice(volume["img_u8"], slice_index)
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)

    if with_overlay:
        seg_plane = _mr_axial_slice(volume["seg_u8"], slice_index)
        for region_name, label_value in volume["region_labels"].items():
            region_mask = seg_plane == int(label_value)
            if not region_mask.any():
                continue
            color = np.asarray(
                MR_REGION_COLORS.get(region_name, _MR_FALLBACK_COLOR),
                dtype=np.float32,
            )
            rgb[region_mask] = (1.0 - alpha) * rgb[region_mask] + alpha * color

    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
