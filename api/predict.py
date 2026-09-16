"""``POST /predict/{patient_id}`` -- ISKELET (v1 kapsam sinirlariyla).

Plan referansi: raw/plan/plan.txt:449-451 ("/predict `risk_score + SHAP`
donmeli, `shap_values` JSONB olarak saklanmali"). Bu dosya YALNIZ
endpoint'i ve tahmin/SHAP hesabini icerir -- `shap_values` DB'ye
YAZILMAZ (JSONB kolonu), tahmin SADECE API yaniti olarak uretilir.

~~GEREKCE (eski, 2026-08-18): "db-agent'in `model_registry` semasi henuz
onaylanmadi, gorev talimati acikca 'model_registry kaydi ATMA' diyor."~~
⚠️ 2026-09-13 ATIF DUZELTMESI (backend-agent-K): yukaridaki gerekce
BAYATLADI, uzeri cizildi -- `model_registry` semasi ONAYLANDI ve tablo
CANLI (XGBoost shadow modelinin `model_id=10` satiri orada, bkz. asagida
"XGBOOST IKINCI KATMAN" md.1). DOGRU/GUNCEL gerekce **REVIZYON 13**'tur
(`concepts/mimari-revizyonlari.md:519-543`): plan.txt:565'in adlandirdigi
**`predictions` tablosu YOK ve ACILMAYACAK** (Baris onayi 2026-09-13);
canli olcum `public` semada 13 BASE TABLE + 1 VIEW (`radiomics_c32`),
`predict` iceren hicbir tablo/view YOK. Kalici tahmin persistansi 6
haftalik kapsam DISIDIR -- yani `shap_values`'in yazilmamasi artik bir
"bekleyen sema onayi" DEGIL, KARARLASMIS bir kapsam sinirdir. Detay:
`decisions/2026-09-11-predictions-tablosu-karari.md` (durum:
KARARLASTI 2026-09-13; 2026-09-14'te dosya adindan "-TASLAK" eki
KALDIRILDI -- eski ad `...-predictions-tablosu-karari-TASLAK.md` artik
YOK, bkz. K23).

OMICS YORUMU (Adim 7, plan.txt:145-151 + v45.txt SS 6.4, satir 1105-1210)
-- 2026-08-18'de eklendi:
  - `raw/mimari/v45.txt:1109` -- "Bu modul Cox modelinin ZORUNLU GIRDISI
    DEGILDIR." `fetch_omics_interpretation()` Cox risk skoru/SHAP
    HESAPLANDIKTAN SONRA cagrilir, `compute_risk_score_and_shap()`'e
    HICBIR GIRDI VERMEZ -- omics'li/omics'siz iki hasta AYNI kod yolundan
    (`compute_risk_score_and_shap`) gecer, bkz. tests/test_api_predict.py
    `test_omics_presence_does_not_affect_cox_risk_score`.
  - Kapi: `patients.has_omics=TRUE` DEGILSE yanit "omics" alani OLMADAN
    doner -- hata YOK, log YOK (plan.txt:503 "sessizce atla" birebir).
  - `patients.has_omics=TRUE` AMA `molecular_scores`'ta satir YOKSA bu,
    yukaridaki "sessizce atla" senaryosu DEGIL (o senaryo has_omics=
    FALSE ile temsil ediliyor) -- bu bir VERI TUTARSIZLIGI, gorunur ama
    Cox sonucunu BOZMAYAN `{"has_omics": True, "available": False,
    "note": ...}` blogu doner (2026-08-18 canli DB'de 0 ornegi var,
    savunma amacli).
  - ACIK BULGU (bu dosyanin yazildigi gunku canli olcum, gorev
    talimatinda YOKTU): `molecular_scores.egfr_amp_flag`,
    `pten_del_flag`, `cdkn2a_del_flag`, `mgmt_interpretation` 48/48
    satirda NULL (yalniz `tmz_class_relative` degil). Kod bunlari `None`
    olarak dondurur, YUTMAZ -- bkz. asagida `_build_omics_block()`.
  - `aggressiveness_score`/`dna_repair_score` KOHORT-ICI GORECELI (v45.txt
    satir 459-460, z-score icerir) VE `score_thresholds` tablosu su an
    0 satir (siniflar hicbir yerde DONMUS degil) -- yanitta
    `interpretation_note` bunu acikca belirtir, MUTLAK olcek iddiasi
    YAPILMAZ.
  - `tmz_resistance_score` yonu: DUSUK = TMZ'ye DUYARLI (v45.txt satir
    453) -- yanitta `tmz_resistance_direction` alaniyla acikca yazilir,
    ters yorum riski kapatilir.

KAPSAM SINIRLARI (gorev talimati, ACIKCA belirtiliyor):
  1. v1 SADECE DB'de C32 radyomigi ZATEN VAR olan hastalari destekler.
     Canli goruntu isleme / yeni-hasta nnU-Net segmentasyonu YOK --
     `GBMAID_ENABLE_NEW_PATIENT_NNUNET` varsayilan KAPALI + bilinen K12
     ITK Turkce-karakter yol bug'i (bkz. tests/test_no_resolve_path_
     regression.py) -- bu ENDPOINT o yolu HIC CAGIRMIYOR.
  2. Model artefakti (varsayilan: `models/cox_phm_v3b_lowvar_v2amgmt_
     2026-09-12.pkl` -- **2026-09-13 REVIZYONU, karar 33**; ~~eskiden
     `models/cox_phm_primary_wt93_clinical_full_ucsf_2026-08-18.pkl`
     (v1, 2026-08-18 -- 2026-09-13)~~, ~~ondan once `_checkpoints/
     arm_primary_wt93_icc60_th06.pkl`, yalniz-radyomik~~; gerekce ve
     kirmizi/yesil olcum icin bkz. asagida `DEFAULT_CHECKPOINT_PATH`
     tanimindaki blok) DOGRUDAN diskten okunur -- `final_features`/`clinical_extra_columns` HER ZAMAN
     artefaktten okunur, ASLA sabit kodlanmaz (CLAUDE.md ilkesi).
     `GBMAID_COX_CHECKPOINT_PATH` env degiskeni HALA GECERLI -- eski
     yalniz-radyomik checkpoint'e (veya baska bir artefakta) geriye donuk
     isaret edebilir, `load_cox_arm_result()` HER IKI pickle bicimini de
     (eski `ArmResult` dataclass'i VE yeni sade sozluk) normalize eder.
  3. Artefakt klinik kovaryat (`clinical_extra_columns`) TASIYORSA
     (2026-09-13 varsayilani v3b SEKIZ kolon TASIYOR -- canli olcum:
     `clinical_age`, `clinical_gender_male`, `clinical_gtr_y`,
     `clinical_gtr_missing`, `clinical_idh_mutant`, `clinical_idh_missing`,
     `clinical_mgmt_methylated`, `clinical_mgmt_missing`; ~~eski varsayilan
     v1 MGMT'siz ALTI kolon tasiyordu~~) bu endpoint hastanin klinik
     verisini `patients` tablosundan CEKER ve modele VERIR. Ic kural
     (2026-08-18, gorev talimati -- "sessiz sifir/varsayilan ATAMA YOK",
     2026-09-12 K19 karariyla KISMEN REVIZE EDILDI, bkz. asagida):
       - Artefaktin istedigi klinik kovaryat kumesi bilinen isimlerden
         FARKLI (desteklenmeyen) bir sey iceriyorsa -> 501 (kapsam disi).
       - Desteklenen kume ama HASTANIN ham degeri taninmayan (yazim
         hatasi/sema disi kategori) -> HER ZAMAN 422, ACIK hata
         mesajiyla hangi alan(lar) eksik/tanimsiz oldugu SOYLENIR.
       - `NOS/NEC` (IDH icin) ve `Indeterminate` (MGMT icin) -- "test
         yapildi, sonuc kullanilamaz" -- ISTISNA, egitimde de ayni
         sekilde "eksik-gosterge" ile MODELLENDIGI icin (K15 karari +
         MGMT_INDETERMINATE_LABEL notu) gecerli/kabul edilen bir deger
         sayilir, 422 TETIKLEMEZ.
       - **2026-09-13 EKLENDI (Baris onayi -- LUMIERE normalizasyon
         gorevi):** "taninmayan deger" kontrolunden ONCE ham deger
         `IDH1_LABEL_NORMALIZATION_MAP`/`MGMT_LABEL_NORMALIZATION_MAP`'ten
         (bkz. asagida "HAM ETIKET NORMALIZASYONU") GECER -- LUMIERE'nin
         `"WT"`/`"wt"`/`"R132H mut"`/`"methylated"`/`"not methylated"`
         gibi kanonik-olmayan (ama ANLAMI acik) ham etiketleri artik 422
         ATMAZ, kanonik karsiliklarina cevrilip NORMAL yoldan kodlanir.
         `"IDH1 neg, Sequencing required"` (10 LUMIERE hastasi) BILINCLI
         olarak `Wildtype`'a DEGIL `NOS/NEC`'e (yani `idh_missing=1`)
         eslenir -- "test yapildi ama sonuc KESINLESMEMIS" epistemik
         durumu Wildtype ile AYNI SAYILAMAZ. DB'nin kendisi DEGISMEZ,
         normalizasyon SADECE bu okuma katmanindadir. Detay: decisions/
         2026-09-13-lumiere-idh-mgmt-etiket-normalizasyonu.md.
       - **HAM DEGER TAMAMEN NULL ise davranis KOVARYAT BAZINDA FARKLI**
         (K19 karari, Baris 2026-09-12, Secenek 2 -- bkz. `CLINICAL_
         COVARIATE_NULL_POLICY`): `age`/`gender` icin HALA 422 (`STRICT`)
         -- egitimde bunlara karsilik gelen bir `*_missing` gosterge
         kolonu YOK, yapisal sebep. `gtr_over90percent`/`idh1_status`/
         `mgmt_status` icin ARTIK 422 YOK (`TRAIN_ALIGNED`) -- NULL,
         `Indeterminate`/`NOS/NEC` ile AYNI sekilde `clinical_*_missing=1`
         olarak kodlanir. Gerekce: `tools/train_cox_week3.py` (satir
         663-687) ucunu de BIREBIR AYNI desenle kodluyor (isna()|belirsiz
         -> *_missing=1); v3b_lowvar_v2amgmt'nin harici dogrulamasi
         (UCSF n=295) NULL-MGMT hastalarini zaten boyle skorlamisti --
         servis bu davranisla HIZALANDI (dogrulanan populasyon == hizmet
         edilen populasyon). MGMT icin bkz. decisions/2026-09-12-k19-mgmt-
         null-politikasi.md. GTR/IDH1 icin 2026-09-13'te AYNI mantikla
         (D6'nin BILINCLI STRICT'inden -- ayni gun -- TRAIN_ALIGNED'e)
         HIZALANDI, tutarlilik gerekcesiyle (bkz. log/2026-09-13.md).
         ⚠️ Bu, TCGA-06-5412 gibi GTR/IDH/MGMT'si NULL hastalarin ARTIK
         200 alabilecegi anlamina gelir -- jury demo Blok B'nin "422
         alir" anlatisi GEÇERSİZLEŞTİ, belge tarafi ayrica ele alinmali.
  4. Kalici model artefakti `models/` altinda VAR (2026-08-18'den itibaren)
     ama `model_registry` DB kaydi bu dosyanin kapsaminda DEGIL (ayri bir
     db-agent gorevi) -- her istek artefakt dosyasini (mtime-anahtarli
     in-process cache ile) okur. Artefakt dosyasi degisirse (yeniden
     egitim) SUREC YENIDEN BASLATILMADAN yeni sonuc alinir.
  5. Bir hastanin C32 whitelist'inde BIRDEN FAZLA `scan_id`'si varsa
     (longitudinal/coklu-zaman-noktali) 422 doner -- hangi taramanin
     kullanilacagina SESSIZCE karar VERILMEZ (gercek DB ile bu prototipte
     KESFEDILEN bir durum, bkz. tools/shap_explainer_prototype.py
     docstring'i "ONEMLI GERCEK-DB BULGUSU").
     **2026-09-13 (Y1 karari) TEK ISTISNA -- YALNIZ LUMIERE:** bkz.
     asagida "LUMIERE KANONIK VIZIT". UPenn/UCSF/TCGA'da davranis
     DEGISMEDI, coklu-tarama HALA 422'dir.

=======================================================================
LUMIERE KANONIK VIZIT (2026-09-13, Y1 -- Baris: "y1 i onaylıyorum
lumiere 72 hastaya çıksın")
=======================================================================
Karar: `decisions/2026-09-13-lumiere-risk-skoru-preop-kanonik-vizit.md`.
LUMIERE hastalari (`patient_id` oneki `Patient-`) icin K2 KILITLI kurali
(2026-08-18) uygulanir: kanonik vizit = LUMIERE ExpertRating CSV'sinde
`Rating == 'Pre-Op'` olan vizit. O vizitin C32 WT radyomigi kullanilir.

KURAL BU DOSYADA YENIDEN YAZILMADI -- `pipeline/lumiere_canonical_visit.py`
TEK KAYNAKTIR ve `tools/rebuild_faiss_indexes.py` (FAISS v1 indeksi) de
AYNI modulu kullanir. Kanit (canli, 2026-09-13): tasima oncesi/sonrasi
kohort secimi BIREBIR ayni (72 hasta / 360 satir, rapor alanlari ozdes) ve
tek-hasta yolu kohort yoluyla 72/72 hastada AYNI `scan_id`'yi seciyor.

🔴 `timepoint_label` SIRALAMASI OLCUT DEGILDIR ("en erken taramayi al"
kurali olcumle CURUTULDU): `week-000*` etiketli 135 satirin 44'u
`Post-Op`; `Patient-020`'de `week-000`->`Post-Op`, `week-000-1`->`Pre-Op`;
`Patient-060`'in bir `Pre-Op` viziti `week-069`'da. TEK olcut `Rating`dir.

GATE "coklu-tarama" DEGIL "LUMIERE"DIR (bilincli): TEK taramasi olan
LUMIERE hastasi da bu kurala TABIDIR. Gerekce -- canli olcum (2026-09-13):
bugune kadar servis edilebilen 4 LUMIERE hastasinin (`Patient-026`/`-044`/
`-053`/`-076`) TEK C32 taramasi `Rating=='Post-Op'`tur; "tek tarama varsa
onu kullan" davranisi bu 4 hastaya AMELIYAT-SONRASI goruntuden risk skoru
basiyordu (sessiz yanlis sonuc). Artik bu 4 hasta da ACIK 422 alir. Yani
4 -> 72 gecisi bir GENISLEME DEGIL, KUME DEGISIMIDIR (eski 4 disarida).

C32'si olmayan `Pre-Op`: 19 hasta -> ACIK 422 (post-op fallback
REDDEDILDI, Baris karari). Sessiz bos/varsayilan DONDURULMEZ.
Yanitta `lumiere_canonical_visit` blogu hangi vizitin kullanildigini
(`timepoint_label` + `rating` + `scan_id`) ACIKCA beyan eder.

SHAP OLCEGI (sayisal olarak tools/shap_explainer_prototype.py'de
DOGRULANDI -- oraya BAKINIZ, bu dosya o kanitlamayi TEKRARLAMAZ):
`lifelines.CoxPHFitter.predict_log_partial_hazard(X) == coef @ (X -
norm_mean)` -- TAM dogrusal. `shap.LinearExplainer((coef, intercept=
-(coef @ norm_mean)), background, feature_perturbation=
"interventional")` bu skoru KAPALI FORMDA (orneklem/yaklastirma YOK)
acikliyor, additivite (SHAP toplami + taban == risk_score) FLOAT
PRECISION'DA tutuyor. `predict_partial_hazard` (hazard ratio, exp
olcegi) ayrica `hazard_ratio_partial_hazard` alaninda bilgi amacli
donuyor ama SHAP DEGERLERI O OLCEGI ACIKLAMIYOR (dogrusal degil).

ARKA PLAN (SHAP referans dagilimi) -- ACIK RISK: su an SABIT olarak
`UPenn-PyRadiomics-107-C32` egitim havuzundan orneklenir (`n=50,
seed=42`, deterministik). Model TCGA yerine UCSF'i harici test olarak
kullanacak sekilde degisince VEYA coklu-kaynakli bir egitim havuzuna
gecilince bu SABIT gozden gecirilmeli -- background artik "gercek
egitim havuzu" ile eslesmeyebilir (bkz. asagida BACKGROUND_
SEGMENTATION_TOOL).

lifelines PRIVATE API KULLANIMI (bilinen kirilganlik): `fitted_model.
_norm_mean` alt-cizgili (private) bir alan -- `lifelines==0.30.0`
`requirements.txt`'de PINLENMIS oldugu icin kisa vadede risk dusuk,
ama surum degisirse bu alanin var olmama/davranis degistirme riski
VAR, ayrica dogrulanmali.

=======================================================================
XGBOOST IKINCI KATMAN -- "shadow" (2026-09-13, ADIM 2; Baris onayi)
=======================================================================
Checkpoint: `models/xgboost_v2a_mgmt_reduce_collinearity_2026-09-13.pkl`
(`tools/export_xgboost_checkpoint.py`, modeling-agent-G, ADIM 1 -- o
dosyaya/`models/`e BURADAN YAZILMAZ, yalniz OKUNUR).

1) STATU = `shadow`, DEGISMEDI. `model_registry` tablosunda bu modelin
   satiri (`model_id=10`, `status='shadow'`) BU DOSYADAN GORULMEZ/
   YAZILMAZ -- 'shadow'->'production' gecisi AYRI bir politika kararidir.
   Yanitta `xgboost.status="shadow"` + `status_note` ile ACIKCA beyan
   edilir: bu blok BIRINCIL KARAR ARACI DEGILDIR. Gerekce: (a) review-gate
   (fiziksel inceleme + codex capraz inceleme + resmi gozden gecirme)
   TAMAMLANMADI, (b) harici AUC 0,723729 [0,650801-0,790298] YALNIZ
   n=232 UCSF hastasinda olculdu (295'in 63'u 12-ay hedefi belirsiz-
   sansurlu oldugu icin DUSURULDU) -- servis edilen populasyon (UPenn/
   UCSF/TCGA/LUMIERE, tek-tarama C32'si olan HERKES) bundan GENISTIR.

2) IKI AYRI OLCEK -- bu dosyanin en kritik riski (2026-09-12'de Cox
   tarafinda AYNI sinifta bir hata YASANDI: API modele ham deger besledi,
   model z-score'lanmis veriyle fit edilmisti; yas katsayisi 0,3532 vs
   0,023029/yil farki ortaya cikti). Checkpoint bu yuzden olcegi IKI
   AYRI blokta TASIR ve bu modul ikisini BIRBIRINE KARISTIRMAZ:
     - `cox_score` blogu (XGBoost'un `cox_oof_score` girdisini ureten TEK
       Cox modeli): 16 radyomik final ozellik + `clinical_age`
       STANDARDIZE (`feature_standardization`, 55 kolonluk sozlukten
       YALNIZ bu modelin kullandigi 17'si), diger 7 klinik bayrak HAM 0/1.
     - `xgboost` blogu: `clinical_age` STANDARDIZE (`age_standardization`,
       `cox_score` blogundaki yas istatistigiyle SAYISAL OLARAK AYNI --
       kod bunu FAIL-LOUD dogrular), **93 WT radyomik HAM (raw)**, 7
       klinik bayrak HAM, + `cox_oof_score`. `xgb_feature_columns` = 102
       kolon, SIRASI ONEMLI.
   Kod-seviyesi savunma: `_assert_xgboost_raw_scale_preserved()` --
   XGBoost'a giden 93 radyomik degerin, ~~DB'den TAZE cekilen~~ ham
   degerle BIREBIR (==) ayni oldugu her istekte dogrulanir. Bir gun biri
   bu 93'u yanlislikla standardizasyondan gecirirse istek SESSIZCE yanlis
   olasilik DONDURMEZ, blok ACIK bir hatayla dusar.
   ⚠️ 2026-09-13 KAPSAM DUZELTMESI (backend-agent-K, kod okunarak
   olculdu): yukaridaki "DB'den TAZE cekilen" ifadesi guard'in gucunu
   ABARTIYORDU, uzeri cizildi. Guard DB'ye IKINCI bir sorgu ATMAZ --
   `build_xgboost_shadow_block()` icinde `fetch_c32_radiomic_row()`
   BIR KEZ cagrilir ve AYNI bellek-ici `radiomic_row` nesnesi hem
   `build_xgboost_feature_frame()`'e hem guard'a verilir. Yani bu bir
   **BUILD-ADIMI REGRESYON GUARD'IDIR**: "cerceve kurulurken/sonrasinda
   bu 93 deger degistirildi mi" sorusunu yanitlar. KAPSAM DISI: eger
   `fetch_c32_radiomic_row()`'un KENDISI bir gun standardize/donusturulmus
   veri dondurmeye baslarsa (ornek: `radiomics_c32` view'i degisirse,
   veya DB'ye standardize deger yazilirsa) guard bunu GOREMEZ -- iki
   taraf da AYNI bozuk kaynaktan gelir, esitlik TUTAR ve istek SESSIZCE
   yanlis olasilik dondurur. O senaryonun savunmasi bu guard degil,
   veri-katmani denetimidir (`tools/data_integrity_check.py`).
   Kolon sirasi savunmasi: `_assert_xgboost_feature_frame_contract()` --
   kolon kumesi VE SIRASI `xgb_feature_columns` ile, ayrica (varsa)
   `fitted_model.feature_names_in_` ile BIREBIR karsilastirilir.

3) HATA FELSEFESI -- `api/analyze_patient.py::_risk_block_unavailable()`
   (B-1, 2026-09-13) deseniyle AYNI: XGBoost blogu uretilemezse bu
   **BLOK-FATAL**'dir, `{"available": False, "error_type", "error_detail",
   "not_available_reason"}` doner; Cox risk skorunu/SHAP'i ve istegin
   BUTUNUNU OLDURMEZ. UYDURMA bir olasilik/skor ASLA dondurulmez.
   ⚠️ B-1'DEN BILINCLI TEK SAPMA: B-1 `psycopg2.OperationalError`'i
   ISTEK-FATAL sayar (DB'ye hic ulasilamazken kismi 200 yaniltici olur).
   BURADA o hata da BLOK-FATAL'dir -- gerekce: bu satira gelinebilmis
   olmasi, AYNI DB'den Cox radyomigi+klinigi+SHAP arka planinin ZATEN
   BASARIYLA okundugu anlamina gelir; son sorguda olusan bir DB
   kesintisinin, ZATEN URETILMIS gecerli bir Cox tahminini cope atmasi
   regresyon olurdu. Sistemik sinyal GIZLENMEZ: `error_type` (=
   `OperationalError`) ve `systemic_hint=True` yanitta GORUNUR.

4) YON (ters yorum riski): hedef `target_12mo_survival` -- 1 = hasta
   12 ayi GORDU (`survival_days >= 365`), 0 = 12 aydan once OLDU.
   `predict_proba(...)[:, 1]` bu yuzden **12-AYLIK SAGKALIM olasiligidir**
   -- YUKSEK deger IYI prognoz demektir, Cox `risk_score`'un TERSI yonde.
   Yanitta `probability_12_month_survival` + `direction_note` ile acikca
   yazilir.
   ~~⚠️ `api/analyze_patient.py` bu modeli `xgboost_recurrence`
   ("nuks/progresyon") olarak adlandiriyor -- egitilen hedef NUKS DEGIL,
   12-ay SAGKALIMdir; bu adlandirma tutarsizligi o dosyanin sahibine
   BULGU olarak bildirildi (bu gorevde o dosyaya DOKUNULMADI).~~
   ✅ 2026-09-13 KAPANDI (backend-agent-K, kod okunarak olculdu): yukaridaki
   BULGU giderildi, uzeri cizildi. `api/analyze_patient.py` artik
   `XGBOOST_BLOCK_KEY = "xgboost_12mo_survival"` (satir 228) kullaniyor;
   eski/yanlis ad YALNIZ `XGBOOST_PREVIOUS_BLOCK_KEY = "xgboost_recurrence"`
   (satir 233) sabitinde, yanitin `previous_block_name` alaninda TARIHSEL
   kayit olarak duruyor ve ust duzey anahtar olarak DONMUYOR (regresyon
   testi: `tests/test_api_analyze_patient.py::test_analyze_patient_old_
   xgboost_recurrence_key_is_gone`). Bu gorevde o dosyaya YINE DOKUNULMADI,
   yalniz buradaki bayat atif duzeltildi.

5) `GBMAID_ENABLE_XGBOOST_SHADOW=0` ile blok KAPATILABILIR (varsayilan
   ACIK). Kapatildiginda blok `available=False` + sebep ("acikca devre
   disi") doner -- Cox yolu ETKILENMEZ.
"""

from __future__ import annotations

import logging
import pickle
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2.extras
from fastapi import APIRouter, HTTPException

from db_connection import get_connection, load_project_environment
from pipeline.cox_model import RegionPivotError, pivot_radiomics_long_to_wide

# K2/Y1 KANONIK VIZIT -- TEK KAYNAK. Bu dosya kurali YENIDEN YAZMAZ;
# `tools/rebuild_faiss_indexes.py` de AYNI modulu kullanir (bkz. modul
# dokstring'i "LUMIERE KANONIK VIZIT"). `api/` bir `tools/` script'ini
# import ETMEZ -- ortak mantik `pipeline/` altindadir.
from pipeline.lumiere_canonical_visit import (
    DEFAULT_LUMIERE_EXPERT_RATING_CSV,
    LUMIERE_PATIENT_ID_PREFIX,
    LUMIERE_PREOP_RATING_LABEL,
    LUMIERE_SOURCE_NAME,
    CanonicalVisitSelection,
    ExpertRatingFileNotFoundError,
    LumierePreopVisitNotAvailableError,
    is_lumiere_patient_id,
    select_canonical_preop_visit_for_patient,
)

router = APIRouter()

# ⚠️ `.resolve()` BURADA BILINCLI ve ZORUNLU -- `.absolute()`'a
# CEVIRILMEMELI (2026-09-13, backend-agent-K OLCTU ve DEGISIKLIGI GERI
# ALDI). K12 deseni (`api/similar.py:154`, `tests/test_no_resolve_path_
# regression.py`) `.absolute()` ister, AMA o kural GORUNTU YAZAN modulleri
# (ITK/SimpleITK; lint'in `SCAN_DIRS`'i yalniz `tools/` + `pipeline/`)
# kapsar -- bu dosya NIfTI YAZMAZ, yalniz pickle/DB okur (Python'un kendi
# G/C'si Turkce yollarda sorunsuz).
# OLCUM (X: subst surucusunden, 2026-09-13):
#   Path('api/predict.py').resolve().parent.parent  -> C:\...\gbm-aid mert ✓
#   Path('api/predict.py').absolute().parent.parent -> X:\                 ✗
# `.absolute()` ile `REPO_ROOT` = `X:\` olur ve `REPO_ROOT.parent` de
# `X:\` kalir (surucu kokunun ebeveyni YOKTUR) -> proje kokundeki KANONIK
# `.env`'e ULASILAMAZ. Bu yola GERCEKTEN bagimli kod var: `tests/test_api_
# predict.py::test_checkpoint_path_picks_up_env_override_in_fresh_process_
# without_db_touch` `REPO_ROOT.parent / ".env"` uzerinden gidiyor ve
# `.absolute()`'a cevrildiginde dosyayi bulamayip SESSIZCE skip'e dusuyor
# (olculdu: 123 passed -> 122 passed + 1 skipped, yani kritik env-override
# korumasi inert kalirdi). `db_connection.py:18` de AYNI sebeple
# `.resolve()` kullaniyor (`PROJECT_ROOT` -> `ENV_FILE`).
REPO_ROOT = Path(__file__).resolve().parent.parent  # lint-allow-resolve: proje kokundeki .env'e ulasmak icin gerekli, bu modul goruntu YAZMAZ

# =====================================================================
# VARSAYILAN COX CHECKPOINT'I -- 2026-09-13 DUZELTMESI (backend-agent-K)
# =====================================================================
# ~~ESKI (2026-08-18 -- 2026-09-13): `cox_phm_primary_wt93_clinical_full_
# ucsf_2026-08-18.pkl` (v1)~~ -- SILINMEDI, uzeri cizildi (wiki hard rule
# #3). Asagidaki `V1_BASELINE_CHECKPOINT_PATH` sabitinde ISMIYLE duruyor,
# `.pkl` dosyasi da diskte KALDI (pinlenmis baseline testi ona baglidir:
# tests/test_api_predict.py::test_predict_patient_real_db_zero_regression_
# pinned_baseline).
#
# NEDEN DEGISTI (karar 33): nihai model secimi `v3b_lowvar_v2amgmt`
# (decisions/2026-09-12-nihai-cox-model-secimi-v3b.md) -- o kararin "Dalga
# etkisi" maddesi 6'si ACIKCA `/predict`'in VARSAYILAN checkpoint'inin bu
# model olmasini sart kosuyordu, madde 2026-09-13'e kadar KARSILANMAMISTI:
# v3b YALNIZ proje kokundeki `.env`'in `GBMAID_COX_CHECKPOINT_PATH` satiri
# sayesinde geliyordu. O satirin OLMADIGI her ortam (CI, temiz klon, baska
# bir takim uyesinin makinesi, dagitilmis sunucu) SESSIZCE v1'i servis
# ediyordu -- hata YOK, uyari YOK.
#
# KIRMIZI/YESIL OLCUM (canli DB, 2026-09-13, backend-agent-K):
#   env override NOTR iken, DEGISIKLIK ONCESI:
#     yuklenen kol = `primary_wt93_clinical_full_ucsf` (8 radyomik + 6
#     klinik, `feature_standardization=None` -> HAM olcek),
#     UCSF-PDGM-167 risk = -0.19639713520164725   ✗ (YANLIS MODEL)
#   env override NOTR iken, DEGISIKLIK SONRASI:
#     yuklenen kol = `v3b_lowvar_v2amgmt` (16 radyomik + 8 klinik,
#     `feature_standardization` 17 kolon),
#     UCSF-PDGM-167 risk = -0.7096787591025546    ✓ (v3b pini)
#
# `GBMAID_COX_CHECKPOINT_PATH` env override'i KORUNDU (bkz.
# `_checkpoint_path()`) -- geriye donuk olarak v1'e/baska bir artefakta
# isaret etmek HALA mumkun. Bu sabitin DEGERI bir REGRESYON TESTIYLE
# PINLENDI (`test_default_checkpoint_path_is_pinned_to_v3b`): biri ileride
# sessizce baska bir dosyaya cevirirse test KIRMIZI olur.
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "models" / "cox_phm_v3b_lowvar_v2amgmt_2026-09-12.pkl"

# v1 (2026-08-18 -- 2026-09-13 arasi varsayilan): kod icinde ARTIK
# KULLANILMIYOR ama dosya diskte DURUYOR ve `tests/test_api_predict.py`'nin
# pinlenmis sifir-regresyon baseline'i ONU acikca yukler. Silinmez/tasinmaz.
V1_BASELINE_CHECKPOINT_PATH = (
    REPO_ROOT
    / "models"
    / "cox_phm_primary_wt93_clinical_full_ucsf_2026-08-18.pkl"
)

# ESKI (2026-08-15, yalniz-radyomik) checkpoint yolu -- kod icinde
# KULLANILMIYOR, yalniz belge/gerekce amacli tutuluyor. Geriye donuk
# erisim `GBMAID_COX_CHECKPOINT_PATH` env degiskeniyle saglanir.
LEGACY_RADIOMICS_ONLY_CHECKPOINT_PATH = (
    REPO_ROOT
    / "artifacts"
    / "week3"
    / "cox_model"
    / "_checkpoints"
    / "arm_primary_wt93_icc60_th06.pkl"
)

# 2026-08-18 EKLENDI, 2026-08-18 (GEC SAAT) GENISLETILDI -- MODEL-AGNOSTIK
# KARAR (Baris): V2 dort-varyant kosusu (v1_referans/v2a_mgmt/v2b_mgmt_
# spline/v2c_mgmt_spline_wttc) HENUZ SURUYOR, hangisi nihai model olacak
# BELLI DEGIL. Bu yuzden bu liste TEK bir varyantin kolon kumesine
# SABITLENMEDI -- tools/train_cox_week3.py::build_variant_clinical_extra_
# columns()'un URETEBILECEGI TUM bilinen kolon adlarini kapsar (o dosyaya
# DOKUNULMUYOR/IMPORT EDILMIYOR, burada BAGIMSIZ/kendi kopyasi tutuluyor).
# Bir artefakt bu kumenin ALT KUMESINI (ornek: v1_referans -- MGMT/spline
# YOK) veya bu kumenin TAMAMINI (v2c) tasiyabilir -- kod HANGISI gelirse
# GELSIN artefaktin KENDI `clinical_extra_columns` listesine gore davranir,
# hicbir isim burada "asil model" olarak VARSAYILMAZ. Bu kumenin TAMAMEN
# DISINDA bir isim gelirse (bilinmeyen bir 3. kovaryat semasi) endpoint
# 501 doner -- SESSIZCE yanlis kodlama uygulanmaz.
CLINICAL_AGE_COLUMN = "clinical_age"
CLINICAL_GENDER_MALE_COLUMN = "clinical_gender_male"
CLINICAL_GTR_Y_COLUMN = "clinical_gtr_y"
CLINICAL_GTR_MISSING_COLUMN = "clinical_gtr_missing"
CLINICAL_IDH_MUTANT_COLUMN = "clinical_idh_mutant"
CLINICAL_IDH_MISSING_COLUMN = "clinical_idh_missing"
# v2a_mgmt/v2b/v2c EKLER (v1_referans TASIMAZ) -- ham kaynak: patients.mgmt_status.
CLINICAL_MGMT_METHYLATED_COLUMN = "clinical_mgmt_methylated"
CLINICAL_MGMT_MISSING_COLUMN = "clinical_mgmt_missing"
# v2b_mgmt_spline/v2c_mgmt_spline_wttc, `clinical_age`'in YERINE (AYNI
# ANDA DEGIL) bu ikisini kullanir -- bkz. asagida "AGE RCS SPLINE (AÇIK
# RISK)" notu, knot'lar YALNIZ EGITIM ANINDA (UPenn'den) sabitlenir.
CLINICAL_AGE_RCS1_COLUMN = "clinical_age_rcs1"
CLINICAL_AGE_RCS2_COLUMN = "clinical_age_rcs2"

SUPPORTED_CLINICAL_EXTRA_COLUMNS: frozenset[str] = frozenset(
    {
        CLINICAL_AGE_COLUMN,
        CLINICAL_GENDER_MALE_COLUMN,
        CLINICAL_GTR_Y_COLUMN,
        CLINICAL_GTR_MISSING_COLUMN,
        CLINICAL_IDH_MUTANT_COLUMN,
        CLINICAL_IDH_MISSING_COLUMN,
        CLINICAL_MGMT_METHYLATED_COLUMN,
        CLINICAL_MGMT_MISSING_COLUMN,
        CLINICAL_AGE_RCS1_COLUMN,
        CLINICAL_AGE_RCS2_COLUMN,
    }
)

# AGE RCS SPLINE (AÇIK RISK, 2026-08-18): `clinical_age_rcs1`/`_rcs2`
# tools/train_cox_week3.py::restricted_cubic_spline_basis()'in UPenn'den
# BIR KEZ hesapladigi 3 dugum noktasina (t1<t2<t3) BAGIMLIDIR -- bu
# dugumler TEK bir hastanin ham `age`'inden YENIDEN TURETILEMEZ (egitim
# kohortunun dagilimina bagli). Bu yuzden bu iki kolon SADECE artefakt
# kendi `age_rcs_knots` (3 elemanli) alanini TASIYORSA hesaplanabilir --
# TASIMIYORSA (bugunku tek artefakt DAHIL, hicbiri henuz tasimiyor)
# endpoint 501 doner (`ClinicalCovariatesNotSupportedError`), SESSIZCE
# yanlis/uydurma bir dugum KULLANILMAZ. v2b/v2c nihai model olarak
# secilirse artefakt kaydetme adiminin knot'lari da SAKLAMASI GEREKIR --
# bu ACIK bir bagimlilik, backend-agent'in kapsaminda cozulmedi.

# Ham `patients` kolonlarinin tanidigi kategorik degerler -- tools/
# train_cox_week3.py::build_clinical_covariate_frame ile AYNI sozluk
# (2026-08-18 db-agent/modeling-agent kararlariyla dondurulmus, bkz. o
# dosyadaki GENDER_MALE_LABEL/GTR_YES_LABEL/IDH1_*_LABEL/MGMT_*_LABEL
# sabitleri).
GENDER_MALE_LABEL = "Male"
GENDER_FEMALE_LABEL = "Female"
GTR_YES_LABEL = "Y"
GTR_NO_LABEL = "N"
IDH1_WILDTYPE_LABEL = "Wildtype"
IDH1_MUTATED_LABEL = "Mutated"
IDH1_NOS_LABEL = "NOS/NEC"  # "test yapilmadi" -- 422 TETIKLEMEZ (K15 karari)
MGMT_METHYLATED_LABEL = "Methylated"
MGMT_UNMETHYLATED_LABEL = "Unmethylated"
MGMT_INDETERMINATE_LABEL = "Indeterminate"  # IDH'nin NOS/NEC'iyle AYNI
# epistemik durum (test yapildi, sonuc kullanilamaz) -- 422 TETIKLEMEZ.

# =====================================================================
# HAM ETIKET NORMALIZASYONU (okuma-zamaninda, DB'YE DOKUNMADAN) --
# 2026-09-13, Baris'in onayi ("LUMIERE'nin ~90 hastasi bir normalizasyon
# adimiyla kurtarilabilir ... baslatalim isi, 2 tane hasta az olmaz zira").
#
# TASARIM KARARI (B): kod-katmaninda okuma-zamani normalizasyon, DB'de
# toplu UPDATE'e (A) TERCIH EDILDI. Gerekce (tam detay: decisions/
# 2026-09-13-lumiere-idh-mgmt-etiket-normalizasyonu.md):
#   1. Provenance korunur -- LUMIERE'nin ORIJINAL ham etiketleri
#      (`patients.idh1_status`/`mgmt_status`) DB'de HIC DEGISMEZ. Ileride
#      "bu deger kimin sozlugunden geldi" sorusu sorulursa ham deger
#      halen orada durur.
#   2. Geri alinabilir/denetlenebilir -- bir eslesme HATALI cikarsa
#      (ornek: yanlis bir esdegerlik) duzeltme TEK satirlik bir kod
#      degisikligi, DB'ye ikinci bir duzeltici UPDATE GEREKMEZ.
#   3. Toplu UPDATE CLAUDE.md'nin KESIN SINIRLAR #4'unun ("semа
#      degisikligi/toplu UPDATE onay ALMADAN calistirilmaz") tam
#      kapsamina girer -- Baris'in bugunku onayi GENEL bir onaydi, DB'ye
#      YAZMADAN cozum bu riski BASTAN ELER.
#   4. TEK OKUMA YOLU riski dusuk: bu ozel durumda (LUMIERE egitime hic
#      girmiyor -- vital_status 0/91, `tools/train_cox_week3.py` LUMIERE'i
#      havuza ALMIYOR) normalizasyonun uygulanmasi gereken tum canli-
#      servis okuma yollari `_fetch_patients_clinical_raw()` FONKSIYONUNA
#      indirgeniyor -- bu dosyada BUNU cagiran TEK IKI nokta (tek-hasta
#      `build_patient_clinical_covariate_row()` VE SHAP arka-plan orneklemi)
#      OTOMATIK olarak normalize edilmis degeri gorur, ayri bir yerde
#      TEKRAR uygulanmasi GEREKMEZ.
#
# ⚠️⚠️ 2026-09-13 GUNCELLEME (backend-agent-K, kod okunarak OLCULDU) --
# ASAGIDAKI "BILINCLI KAPSAM DISI" PARAGRAFI BAYATLADI, uzeri cizildi:
# `api/similar.py` ARTIK bu iki sozlugu KENDISI KOPYALAMIYOR/yok saymiyor,
# DOGRUDAN BURADAN IMPORT EDIYOR (`api/similar.py:145-148`, yorumu: "🔒 TEK
# KAYNAK -- drift YASAK"). Yani asagida tarif edilen `Wildtype`/`WT`
# asimetrisi `/similar` tarafinda KAPANDI ve iki endpoint ARTIK BIRLIKTE
# degisir. Bu sozluklerin icerigini degistiren herkes `/similar`'i da
# ETKILEDIGINI bilmelidir. (Eski metin tarihsel kayit olarak birakildi.)
#
# ~~⚠️ BILINCLI KAPSAM DISI (bu gorevde DOKUNULMADI, BULGU olarak bildirilir):~~
# `api/similar.py` (FAISS benzer-hasta OPSIYONEL filtresi) KENDI docstring'inde
# ("WT"=="Wildtype" gibi bir kaynaklar-arasi esleme kararinin BILINCLI
# olarak VERILMEDIGINI) ONCEDEN belgeliyor ve tam-metin eslesmesi YAPIYOR --
# bu normalizasyonu KULLANMAZ, o dosyaya bu gorevde DOKUNULMADI. Sonuc:
# `idh1_status=WT` ile filtrelenen bir FAISS sorgusu LUMIERE'in "WT"
# etiketli hastalarini YAKALAR (tam-metin eslesme) ama `idh1_status=Wildtype`
# ile filtrelenen bir sorgu YAKALAMAZ -- bu asimetri BU GOREVDEN ONCE de
# vardi (similar.py'nin kendi docstring'i ile ONCEDEN belgelenmisti),
# bu gorev onu DEGISTIRMEDI/BUYUTMEDI, sadece MEVCUT oldugunu tekrar
# bildirir.
#
# ⚠️ KRITIK KARAR -- `'IDH1 neg, Sequencing required'` (10 hasta) BILINCLI
# olarak `NOS/NEC`'e ESLENIR, `Wildtype`'a ASLA DEGIL. Bu ham etiket
# "immunohistokimyada NEGATIF ama dizileme sonucu HENUZ KESINLESMEMIS"
# anlamina gelir -- yani test YAPILMIS ama sonuc HENUZ KULLANILABILIR
# DEGIL. Bu, K15 kararindaki `NOS/NEC` semantigiyle ("test yapildi, sonuc
# kullanilamaz" -> `idh_missing=1`) BIREBIR ortusuyor. Bunu `Wildtype`
# saymak, HENUZ DOGRULANMAMIS/KESINLESMEMIS bir sonucu KESINLESMIS gibi
# kodlamak olurdu -- bilmedigimiz bir seyi biliyormus gibi davranmak,
# ZORUNLU-BEYANLAR.md'nin ruhuyla DOGRUDAN CELISIR. `IDH1_NOS_LABEL`'e
# eslenen bu hastalar `clinical_idh_missing=1`/`clinical_idh_mutant=0`
# olarak kodlanir (asagida `_encode_clinical_frame()`), ASLA `mutant=0`
# ("bilinen wildtype") ANLAMINDA YORUMLANMAZ -- gosterge kolonu bu ikisini
# AYRISTIRIR.
#
# Guvenli/duz eslemeler (buyuk/kucuk harf veya kisaltma FARKI, anlam
# DEGISMEDI):
#   IDH1 "WT"/"wt"        -> Wildtype  (kisaltma, UPenn/UCSF'in kendi
#                            "Wildtype" ile AYNI klinik anlam)
#   IDH1 "R132H mut"      -> Mutated   (R132H, IDH1'in EN YAYGIN nokta
#                            mutasyonu -- klasik/tartismasiz Mutated)
#   MGMT "methylated"     -> Methylated
#   MGMT "not methylated" -> Unmethylated
#
# Haritada OLMAYAN bir deger (BILINEN kanonik etiketler DAHIL)
# DEGISTIRILMEDEN kalir -- `pd.Series.replace()` sadece haritadaki
# anahtarlari degistirir, kalanini OLDUGU GIBI birakir. Asagidaki mevcut
# "taninmayan deger" kontrolleri (422) BU normalizasyondan SONRA calisir --
# yani harita KAPSAMADIGI HICBIR yazim-hatasi/sema-disi degeri SESSIZCE
# gecistirmez, sadece BURADA ACIKCA listelenen 4 esdegerligi cozer.
#
# 🔒 DRIFT GUARD (2026-09-13 EKLENDI, karar 35 -- backend-agent-K):
# `tools/data_integrity_check.py` bu iki sozlugun ELLE senkronize edilen
# BAGIMSIZ bir AYNASINI tutar (`KNOWN_NORMALIZED_IDH1_RAW_VALUES` /
# `KNOWN_NORMALIZED_MGMT_RAW_VALUES`) -- o dosya `api/` paketini (FastAPI/
# shap) BILINCLI olarak import ETMEZ. Esitligi artik bir test KORUYOR:
# `tests/test_normalization_map_mirror_sync.py` (iki dosyayi `ast.parse`
# ile okur, import ETMEZ). ⚠️ Bu sozluklerden birini degistiren KISI
# AYNAYI DA guncellemek zorundadir, yoksa o test KIRMIZI olur. Ayrica
# `api/similar.py` bu sozlukleri DOGRUDAN buradan IMPORT ediyor -- icerik
# degisikligi `/similar`'in filtre davranisini da ANINDA degistirir.
IDH1_LABEL_NORMALIZATION_MAP: dict[str, str] = {
    "WT": IDH1_WILDTYPE_LABEL,
    "wt": IDH1_WILDTYPE_LABEL,
    "R132H mut": IDH1_MUTATED_LABEL,
    "IDH1 neg, Sequencing required": IDH1_NOS_LABEL,
}
MGMT_LABEL_NORMALIZATION_MAP: dict[str, str] = {
    "methylated": MGMT_METHYLATED_LABEL,
    "not methylated": MGMT_UNMETHYLATED_LABEL,
}


def _normalize_clinical_categorical_raw_values(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """`idh1_status`/`mgmt_status` ham kolonlarindaki BILINEN-kanonik-olmayan
    (bkz. yukaridaki `IDH1_LABEL_NORMALIZATION_MAP`/`MGMT_LABEL_
    NORMALIZATION_MAP`, 2026-09-13 karari) etiketleri kanonik forma cevirir.

    DB'ye HICBIR SEKILDE YAZMAZ -- sadece bu surecteki bellek-ici DataFrame
    kopyasini degistirir (`raw_frame.copy()`), cagiran tarafin elindeki
    orijinal nesneyi MUTASYONA UGRATMAZ. `_fetch_patients_clinical_raw()`
    TARAFINDAN (tek okuma noktasi) OTOMATIK cagirilir -- bu fonksiyonu
    ayrica DOGRUDAN cagirmak GEREKMEZ."""

    result = raw_frame.copy()
    if "idh1_status" in result.columns:
        result["idh1_status"] = result["idh1_status"].replace(IDH1_LABEL_NORMALIZATION_MAP)
    if "mgmt_status" in result.columns:
        result["mgmt_status"] = result["mgmt_status"].replace(MGMT_LABEL_NORMALIZATION_MAP)
    return result


# =====================================================================
# KOVARYAT-BAZLI NULL POLITIKASI (K19 karari, Baris 2026-09-12, Secenek
# 2) -- ham `patients` kolonu NULL geldiginde endpoint'in davranisi
# BURADAN, TEK bir yerden okunur. Yeni bir kovaryati (ornek: ileride
# GTR/IDH) hizalamak bu sozlukte TEK SATIR degistirmektir -- kod
# (`build_patient_clinical_covariate_row`) politikayi okur, kovaryat
# adini SABIT KODLAMAZ.
#
#   STRICT        -> ham deger NULL ise `ClinicalCovariateMissingError`
#                     (422) -- sessiz varsayilan/eksik-gosterge ATANMAZ.
#   TRAIN_ALIGNED -> ham deger NULL ise 422 ATILMAZ; `_encode_clinical_
#                     frame()`'in ZATEN tolerant (fillna+eksik-gosterge)
#                     kodlamasindan gecer -- egitimdeki kodlamayla
#                     BIREBIR AYNI (`*_missing=1`).
#
# ⚠️ Bu politika SADECE ham degerin NULL/None/NaN olma durumunu kapsar.
# NULL-DISI ama taninmayan bir deger (ornek: yazim hatasi/sema disi
# kategori) HER IKI politikada da HER ZAMAN 422 atar -- bu "bilinen
# eksiklik" degil, veri bozuklugudur, politikadan BAGIMSIZDIR (bkz.
# `_clinical_raw_value_is_rejected()`).
#
# 2026-09-12 durumu: `mgmt_status` TRAIN_ALIGNED yapildi (v3b_lowvar_
# v2amgmt'nin harici dogrulamasi NULL-MGMT hastalarini zaten mgmt_
# missing=1 ile skorladi -- bkz. decisions/2026-09-12-k19-mgmt-null-
# politikasi.md). `gtr_over90percent`/`idh1_status` O GUN BILINCLI olarak
# STRICT KALMISTI (D6 karari, ayni gun).
#
# 2026-09-13 GUNCELLEME (Baris onayi): GTR/IDH1 de MGMT gibi
# TRAIN_ALIGNED yapildi -- gerekce TUTARLILIK: `tools/train_cox_week3.py`
# (satir 663-687) GTR/IDH/MGMT'nin ucunu de BIREBIR AYNI desenle kodluyor
# (isna()|belirsiz -> *_missing=1). Asimetri SADECE serviste vardi ve
# "dogrulanan populasyon != hizmet edilen populasyon" farkini
# SURDURUYORDU (K19'un MGMT icin cozdugu sorunun GTR/IDH ayagi acikti).
# `age`/`gender` HALA STRICT -- egitimde bunlara karsilik gelen bir
# `*_missing` gosterge kolonu YOK (yapisal sebep, K19 notunda da
# belirtilmisti), politika-parametrik hizalamanin bir karsiligi olmazdi.
CLINICAL_NULL_POLICY_STRICT = "STRICT"
CLINICAL_NULL_POLICY_TRAIN_ALIGNED = "TRAIN_ALIGNED"

CLINICAL_COVARIATE_NULL_POLICY: dict[str, str] = {
    "age": CLINICAL_NULL_POLICY_STRICT,
    "gender": CLINICAL_NULL_POLICY_STRICT,
    "gtr_over90percent": CLINICAL_NULL_POLICY_TRAIN_ALIGNED,  # 2026-09-13, Baris
    "idh1_status": CLINICAL_NULL_POLICY_TRAIN_ALIGNED,  # 2026-09-13, Baris
    "mgmt_status": CLINICAL_NULL_POLICY_TRAIN_ALIGNED,  # K19, Baris 2026-09-12
}


def _clinical_raw_value_is_rejected(
    raw_value: Any,
    recognized_labels: frozenset[str],
    *,
    raw_column: str,
) -> bool:
    """Tek bir ham klinik kolon degerinin `build_patient_clinical_
    covariate_row()`'un sikilik on-kontrolunde REDDEDILIP reddedilmeyecegini
    dondurur (`True` -> reddet/422'ye ekle).

    Kural:
      1. Deger `recognized_labels` icindeyse (ornek: MGMT icin
         `Indeterminate`, IDH1 icin `NOS/NEC` -- ikisi de K15/MGMT_
         INDETERMINATE_LABEL notuyla ONCEDEN "bilinen eksiklik" sayilan
         sentinel'ler) HICBIR ZAMAN reddedilmez -- bu politikadan
         BAGIMSIZDIR.
      2. Deger NULL/None/NaN ise: SADECE `CLINICAL_COVARIATE_NULL_
         POLICY[raw_column]` `STRICT` ise reddedilir; `TRAIN_ALIGNED`
         ise reddedilmez (asagida `_encode_clinical_frame()` bunu zaten
         tolerant kodlayacaktir).
      3. Deger NULL DEGIL ama `recognized_labels` DISINDA (ornek: yazim
         hatasi/sema disi kategori) ise HER ZAMAN reddedilir -- bu bir
         "bilinen eksiklik" degil veri bozuklugu, politika ONU KAPSAMAZ.
    """

    if raw_value in recognized_labels:
        return False
    if pd.isna(raw_value):
        return CLINICAL_COVARIATE_NULL_POLICY[raw_column] == CLINICAL_NULL_POLICY_STRICT
    return True


# C32 KAPALI BEYAZ LISTE -- gorev talimatinda ACIKCA verilen 4 isim.
# `tools/train_cox_week3.py::ALLOWED_C32_SEGMENTATION_TOOLS` ile KASITLI
# AYRI tutuluyor: o liste egitim-havuzu icin LUMIERE'i HARIC tutuyor
# (LUMIERE'de olay/vital_status verisi yok, egitime giremez) -- ama TEK
# HASTA tahmini icin LUMIERE'den bir hasta istemek mesru olabilir (yalniz
# radyomik gerekli, sağkalim etiketi gerekmiyor), bu yuzden burasi 4
# ismin TUMUNU kabul eder.
ALLOWED_C32_SEGMENTATION_TOOLS: frozenset[str] = frozenset(
    {
        "UPenn-PyRadiomics-107-C32",
        "LUMIERE-PyRadiomics-107-C32",
        "TCGA-ground-truth-C32",
        "UCSF-PDGM-PyRadiomics-107-C32",
    }
)

# LUMIERE ExpertRating CSV'si (K2'nin TEK girdisi) -- `raw/` YALNIZ OKUNUR.
# Modul seviyesinde tutulur ki testler monkeypatch edebilsin; degeri
# `pipeline/lumiere_canonical_visit.py`'nin KENDI sabitinden gelir, burada
# YENIDEN KURULMAZ (ikinci bir yol tanimi = drift riski).
LUMIERE_EXPERT_RATING_CSV_PATH = DEFAULT_LUMIERE_EXPERT_RATING_CSV

# =====================================================================
# XGBOOST IKINCI KATMAN -- sabitler (2026-09-13, ADIM 2). Bkz. modul
# dokstring'i "XGBOOST IKINCI KATMAN -- shadow".
# =====================================================================
DEFAULT_XGBOOST_CHECKPOINT_PATH = (
    REPO_ROOT / "models" / "xgboost_v2a_mgmt_reduce_collinearity_2026-09-13.pkl"
)
XGBOOST_CHECKPOINT_ENV_VAR = "GBMAID_XGBOOST_CHECKPOINT_PATH"
XGBOOST_ENABLE_ENV_VAR = "GBMAID_ENABLE_XGBOOST_SHADOW"

# `model_registry.model_id=10` satirindaki statu -- BU DOSYADAN
# DEGISTIRILMEZ/DB'YE YAZILMAZ, yalniz yanitta BEYAN edilir.
XGBOOST_MODEL_STATUS = "shadow"

XGBOOST_STATUS_NOTE = (
    "Bu blok GOLGE (shadow) modundadir: BIRINCIL KARAR ARACI DEGILDIR, "
    "klinik/juri karari Cox risk skoru + SHAP uzerinden verilir. Gerekce "
    "(1) review-gate (ilgili takim uyesinin fiziksel incelemesi + codex "
    "capraz incelemesi + final rapor oncesi resmi gozden gecirme) HENUZ "
    "TAMAMLANMADI; (2) harici gecerleme YALNIZ n=232 UCSF hastasinda "
    "yapildi (AUC 0,723729 [0,650801-0,790298]) -- UCSF'in 295 hastasindan "
    "63'u 12-ay hedefi belirsiz-sansurlu oldugu icin DUSURULDU; bu "
    "endpoint'in servis ettigi populasyon (tek-tarama C32 radyomigi olan "
    "TUM hastalar: UPenn/UCSF/TCGA/LUMIERE) o 232'den DAHA GENISTIR, yani "
    "raporlanan AUC bu hastaya BIREBIR devredilemez. `model_registry` "
    "tablosundaki status='shadow' satiri DEGISTIRILMEDI. "
    # 2026-09-13 EKLENDI (backend-agent-K): asagidaki cumle bir KURAL
    # IHLALI beyani DEGIL, bir YORUM RISKI uyarisidir -- bkz. `build_
    # xgboost_shadow_block()` icindeki `cox_score_input.source_note`.
    "(4) ADLANDIRMA UYARISI: `cox_score_input.score_column` adi "
    "`cox_oof_score`'dur ve EGITIMDE gercekten out-of-fold'du (Cox->XGBoost "
    "sizinti kurali boyle saglandi); SERVIS aninda ise ayni isimli deger "
    "hastanin KENDI kovaryatlarindan, egitilmis Cox katsayilariyla "
    "hesaplanir -- egitim havuzundaki bir UPenn hastasi icin bu deger "
    "artik IN-SAMPLE'dir (o hasta Cox'un fit'ine girmisti). Bu, egitim "
    "protokolunun ihlali DEGILDIR (out-of-fold sarti EGITIM zamani icindir; "
    "serviste her hasta icin tek bir uretim modeli vardir) ama UPenn "
    "hastalarinda dondurulen olasilik bu yuzden IYIMSER olabilir ve bir "
    "GECERLEME sonucu gibi okunmamalidir. Harici (UCSF/TCGA/LUMIERE) "
    "hastalar icin boyle bir in-sample durumu YOKTUR."
)

XGBOOST_TARGET_NOTE = (
    "Hedef `target_12mo_survival`: 1 = hasta 12 ayi GORDU (survival_days "
    ">= 365,0 -- 12. aydan SONRA olduyse de 1), 0 = 12 aydan ONCE oldu. "
    "Egitimde 12 aydan once TAKIBI KESILEN (sansurlu) hastalar UYDURMA "
    "etiket verilmesin diye DUSURULDU (IPCW kapsam disi, bkz. pipeline/"
    "xgboost_model.py::define_twelve_month_survival_target)."
)

XGBOOST_DIRECTION_NOTE = (
    "`probability_12_month_survival` = predict_proba(...)[:, 1] = 12 ayi "
    "GORME olasiligi -- YUKSEK deger IYI prognozdur. DIKKAT: Cox "
    "`risk_score_log_partial_hazard` ile TERS yonludur (orada YUKSEK = "
    "KOTU). Ikisini ayni yonde okumak ciddi bir yorum hatasi olur."
)

XGBOOST_SCALE_CONTRACT_NOTE = (
    "IKI AYRI OLCEK: (a) `cox_oof_score` uretilirken 16 radyomik final "
    "ozellik + clinical_age checkpoint'in `cox_score.feature_"
    "standardization` istatistigiyle Z-SKORLANIR, diger klinik bayraklar "
    "HAM 0/1 kalir; (b) XGBoost girdisinde 93 WT radyomik HAM (raw) "
    "kalir, YALNIZ clinical_age Z-skorlanir (`xgboost.age_"
    "standardization`), 7 klinik bayrak HAM. Her istekte iki kod-seviyesi "
    "guard calisir: ham-olcek korunmasi (93 deger DB'den gelen ham degere "
    "BIREBIR esit mi) ve kolon kumesi/SIRASI (xgb_feature_columns + "
    "fitted_model.feature_names_in_)."
)

# `age_standardization` (xgboost blogu) ile `feature_standardization
# ['clinical_age']` (cox_score blogu) SAYISAL OLARAK AYNI olmali --
# checkpoint'i ureten script ikisini de AYNI training_frame'den, AYNI
# formulle hesapliyor (bkz. tools/export_xgboost_checkpoint.py, ayrica
# canli olcum 2026-09-13: fark 0,0). Ayrisirlarsa bu, "hangi yas olcegi
# dogru" sorusunu SESSIZCE cozulemez hale getirir -> FAIL-LOUD.
XGBOOST_AGE_STANDARDIZATION_TOLERANCE = 1e-12

# SHAP arka plani icin SABIT kaynak -- bkz. modul dokstring'i "ACIK RISK".
BACKGROUND_SEGMENTATION_TOOL = "UPenn-PyRadiomics-107-C32"
DEFAULT_N_SHAP_BACKGROUND = 50
DEFAULT_SHAP_SEED = 42
SHAP_ADDITIVITY_TOLERANCE = 1e-6


class CheckpointNotFoundError(RuntimeError):
    """`_checkpoints/arm_*.pkl` diskte yok."""


class ClinicalCovariatesNotSupportedError(RuntimeError):
    """Artefaktin istedigi `clinical_extra_columns` kumesi `SUPPORTED_
    CLINICAL_EXTRA_COLUMNS` DISINDA bir isim iceriyor -- bu endpoint o
    kodlamayi TANIMIYOR, SESSIZCE yanlis/varsayilan bir sey UYGULAMAZ."""


class ClinicalCovariateMissingError(RuntimeError):
    """Hastanin klinik kovaryat icin gereken ham `patients` kolonu (yas/
    cinsiyet/GTR/IDH1/MGMT) NULL (SADECE `STRICT` politikali kovaryatlar
    icin -- bkz. `CLINICAL_COVARIATE_NULL_POLICY`) veya HER ZAMAN
    taninmayan bir deger tasiyor -- 2026-08-18 gorev talimati: sessiz
    sifir/varsayilan ATAMA YOK, endpoint 422 doner ve HANGI alan(lar)in
    eksik/tanimsiz oldugunu ACIKCA soyler.

    2026-09-12 GUNCELLEME (K19 karari, Baris, Secenek 2): `mgmt_status`
    ARTIK bu istisnayi NULL icin FIRLATMAZ -- politikasi `TRAIN_ALIGNED`,
    NULL `clinical_mgmt_missing=1` olarak sessizce (ama egitimle BIREBIR
    AYNI sekilde) kodlanir.

    2026-09-13 GUNCELLEME (Baris onayi, tutarlilik gerekcesiyle):
    `gtr_over90percent`/`idh1_status` DA `TRAIN_ALIGNED`'e HIZALANDI --
    NULL artik bu ikisi icin de bu istisnayi FIRLATMAZ, `clinical_gtr_
    missing=1`/`clinical_idh_missing=1` olarak kodlanir. Eski hali
    (2026-09-12 D6 karari, ayni gun) BILINCLI bir asimetriydi; simdi
    `age`/`gender` DISINDA (yapisal sebep -- egitimde karsilik gelen bir
    `*_missing` kolonu yok) hicbir kovaryat STRICT DEGIL."""


class PatientNotFoundError(RuntimeError):
    """`patients` tablosunda bu `patient_id` yok."""


class C32RadiomicsNotFoundError(RuntimeError):
    """Hasta var ama C32 beyaz listesinde radyomik satiri yok."""


class MultiScanNotSupportedError(RuntimeError):
    """Hastanin C32 whitelist'inde BIRDEN FAZLA `scan_id`'si var
    (longitudinal/coklu-zaman-noktali) -- v1 bunu desteklemiyor.

    2026-09-13 (Y1): bu istisna **YALNIZ LUMIERE-DISI** kaynaklar (UPenn/
    UCSF/TCGA) icin firlatilmaya DEVAM EDER. LUMIERE'de K2 kanonik-vizit
    kurali vardir ve coklu-tarama ORADA cozulur; diger kaynaklarda boyle
    bir kural YOKTUR, gevsetmek sessiz-hata kapisi acardi (Y1 sart 2)."""


class LumiereCanonicalVisitDataError(RuntimeError):
    """LUMIERE kanonik-vizit kurali UYGULANAMADI cunku gereken veri
    YAPISAL olarak eksik/celiskili: `mr_scans.timepoint_label` kolonu
    sorgu sonucunda yok, ya da `patient_id` oneki LUMIERE desenindeyken
    (`Patient-`) satirlarin `dataset_sources.source_name`'i `LUMIERE`
    DEGIL (veya tersi).

    SESSIZCE baska bir kurala DUSULMEZ -- kanonik vizit secilemiyorsa
    hicbir risk skoru URETILMEZ. (`tools/data_integrity_check.py::
    PATIENT_ID_PREFIX_RULES` bu iki alanin canli DB'de tutarli oldugunu
    DENETLER; bu istisna o denetim bir gun kirilirsa API'nin SESSIZCE
    yanlis hasta/vizit servis etmemesi icin vardir.)"""


class RequiredRegionMissingError(RuntimeError):
    """Modelin gerektirdigi bolge (ornek: WT) bu hastada yok/pivotlanamadi."""


class RegionCanonicalizationError(RuntimeError):
    """`pipeline.cox_model.canonicalize_region_label()` (dolayisiyla
    `pivot_radiomics_long_to_wide()`) taninmayan bir (kaynak, ham-bolge)
    cifti gordu -- ornek: 2026-09-12 canli bulgusu, UCSF icin
    `REGION_NAME_ALIASES` henuz tanimsizdi (`ValueError: REGION_NAME_ALIASES
    kaynagi tanimiyor: 'UCSF'`). O fonksiyon bare `ValueError` firlatiyor;
    bu sinif onu burada YAKALANABILIR/isimlendirilmis bir hataya cevirir --
    SESSIZCE yutulmaz, yakalanmamis 500 olarak sizmaz. Kok neden (alias
    eksikligi) `pipeline/cox_model.py`'de duzeltilir/duzeltiliyor olabilir --
    bu sinifin gorevi API katmaninin AYNI hata SINIFINA (baska bir
    kaynak/bolge icin tekrar olusabilecek) karsi dayanikli olmasidir."""


class FeatureStandardizationConfigError(RuntimeError):
    """Checkpoint'in `feature_standardization` sozlugu bozuk: std==0/None,
    mean/std eksik, ya da listelenen bir kolon modelin
    final_features+clinical_extra_columns kumesinde yok. Sessizce
    atlanmaz/varsayilan uygulanmaz -- 2026-09-12 Baris karari (Secenek A)
    sozlesmesinin 4. maddesi."""


class FeatureStandardizationColumnMissingError(RuntimeError):
    """`feature_standardization`'da listelenen bir kolon, standardize
    edilecek veri setinde (hasta satiri veya SHAP arka plani) yok."""


class XGBoostCheckpointNotFoundError(RuntimeError):
    """XGBoost (ikinci katman, shadow) checkpoint'i diskte yok."""


class XGBoostCheckpointContractError(RuntimeError):
    """XGBoost checkpoint'i `tools/export_xgboost_checkpoint.py`'nin
    sozlesmesine UYMUYOR: beklenen blok/alan eksik, `xgb_feature_columns`
    yapisal kurulusu (`raw_radiomic + clinical_extra + [score_column]`)
    ile uyusmuyor, `cox_score` blogunun standardizasyon sozlugu modelin
    kullandigi bir kolonu KAPSAMIYOR (-> sessiz ham-besleme riski), ya da
    iki blogun yas istatistigi AYRISMIS. SESSIZCE duzeltilmez/atlanmaz."""


class XGBoostFeatureFrameError(RuntimeError):
    """XGBoost'a verilecek cerceve sozlesmeyi ihlal ediyor: kolon eksik/
    fazla, kolon SIRASI `xgb_feature_columns` (veya `fitted_model.
    feature_names_in_`) ile ayni DEGIL, ya da HAM kalmasi gereken 93
    radyomik deger DB'den okunan ham degerle AYNI DEGIL (olcek karismasi).
    Bu hatada olasilik URETILMEZ -- sessizce yanlis bir olasilik
    dondurmekten sonsuz kez iyidir."""


# =====================================================================
# 1) Checkpoint yukleme -- mtime-anahtarli in-process cache
# =====================================================================

_checkpoint_cache_lock = threading.Lock()
_checkpoint_cache: dict[str, dict[str, Any]] = {}


_dotenv_loaded_lock = threading.Lock()
_dotenv_loaded = False


def _ensure_project_env_loaded() -> None:
    """`.env` HENÜZ yüklenmemişken `_checkpoint_path()` çağrılırsa
    `GBMAID_COX_CHECKPOINT_PATH` override'ı SESSİZCE kaçırılabiliyordu
    (2026-09-12 canlı bulgusu -- eskiden `.env` YALNIZ `db_connection.
    get_connection()` içinde LAZY yükleniyordu; `predict_patient()`'in
    İLK satırı `load_cox_arm_result()` -> `_checkpoint_path()`'tir, yani
    herhangi bir DB çağrısından ÖNCE calisir -- taze bir process'te DB'ye
    hiç dokunulmadan gelen ilk istek override'i KACIRIP sessizce eski/
    varsayilan checkpoint'i yüklerdi. Koordinator: "bugünün tüm deploy
    işini geçersiz kılabilecek sınıfta bir sorun" -- v3b deploy edilmiş
    olsa BILE taze bir uvicorn süreci ilk istekte v1'e düşebilirdi).

    Çözüm: `db_connection.load_project_environment()`'i (AYNI, TEK
    kanonik `.env` yolu -- proje kökü, `db_connection.ENV_FILE`; CLAUDE.md
    ".env kuralı") burada da İDEMPOTENT olarak çağırıyoruz --
    `get_connection()`'ın kendi lazy yüklemesi BOZULMUYOR (`load_dotenv
    (..., override=False)` iki kez çağrılsa da zararsız, `db_connection.
    py`'ye DOKUNULMADI), yalniz checkpoint-seçiminin bu yüklemeye
    BAĞIMLI KALMASI ortadan kaldırılıyor. Process içinde bir kez
    çalışır (`_dotenv_loaded` bayrağı) -- her istekte diski tekrar
    OKUMAZ."""

    global _dotenv_loaded
    if _dotenv_loaded:
        return
    with _dotenv_loaded_lock:
        if _dotenv_loaded:
            return
        load_project_environment()
        _dotenv_loaded = True


# `api/analyze_patient.py` / `api/main.py` ile AYNI desen: kutuphane kodu
# global log yapilandirmasini ELE GECIRMEZ -- `logging.basicConfig()` BURADA
# CAGRILMAZ, yalniz adlandirilmis bir logger alinir. Handler'i uvicorn/pytest
# (caplog) kurar.
_logger = logging.getLogger("api.predict")


def _checkpoint_override_points_elsewhere(resolved: Path) -> bool:
    """Cozulen checkpoint yolu `DEFAULT_CHECKPOINT_PATH` ile AYNI DOSYAYI mi
    gosteriyor? (`True` => FARKLI dosya.)

    Neden duz string karsilastirmasi DEGIL: proje kokundeki `.env` bu makinede
    override'i ZATEN v3b'nin KENDISINE isaret ettiriyor (olculdu, 2026-09-14) --
    ve ayni dosyaya subst edilmis bir surucu harfiyle (X:) veya goreli bir
    yolla da isaret edilebilir. Bu durumlarda uyari basmak SAF GURULTU
    olurdu (gorev sarti:
    "varsayilan yolla kosuldugunda hicbir sey basilmamali"). `os.path.samefile`
    subst/symlink/goreli farklarini eler; dosya diskte YOKSA (bozuk override)
    `OSError` atar -> normcase+abspath'e duseriz, ki o durumda uyari basmak
    ZATEN dogrudur."""

    import os

    try:
        return not os.path.samefile(str(resolved), str(DEFAULT_CHECKPOINT_PATH))
    except (OSError, ValueError):
        return os.path.normcase(os.path.abspath(str(resolved))) != os.path.normcase(
            os.path.abspath(str(DEFAULT_CHECKPOINT_PATH))
        )


def _checkpoint_path() -> Path:
    """Servis edilecek Cox checkpoint'inin yolunu cozer.

    DAVRANIS 2026-09-14'te DEGISMEDI -- override varsa o, yoksa
    `DEFAULT_CHECKPOINT_PATH`. EKLENEN TEK SEY GORUNURLUK (Baris'in 2 numarali
    karari): override BASKA bir dosyayi gosteriyorsa tek satirlik bir
    `logging.warning` basilir. Bu bir HATA DEGILDIR (override geriye donuk
    erisim icin BILINCLI olarak korunuyor, bkz. `DEFAULT_CHECKPOINT_PATH`
    blogu) -- `assert`/`raise` YOK, 503/yukleme mantigi/hata siniflari
    DOKUNULMADI. Bulgu (backend-agent-R2, 2026-09-14): modulde HIC `logging`
    yoktu, yani `.env`'in o satirini elle v1'e cevirmek API'de hicbir iz
    birakmiyordu; tek isaret yanittaki `model_arm` alaniydi.

    Uyari HER cagrida (yani her istekte) basilir -- bilincli: amac "gurultulu"
    olmasi. Varsayilanla kosuldugunda hicbir sey basilmaz."""

    import os

    _ensure_project_env_loaded()
    override = os.environ.get("GBMAID_COX_CHECKPOINT_PATH")
    if not override:
        return DEFAULT_CHECKPOINT_PATH

    resolved = Path(override)
    if _checkpoint_override_points_elsewhere(resolved):
        _logger.warning(
            "Cox checkpoint OVERRIDE aktif: GBMAID_COX_CHECKPOINT_PATH -> %s "
            "(beklenen varsayilan: %s). Servis edilen model nihai varsayilan "
            "model DEGIL -- bu kosunun ciktisi BIRINCIL SONUC OLARAK "
            "RAPORLANAMAZ.",
            resolved.name,
            DEFAULT_CHECKPOINT_PATH.name,
        )
    return resolved


def _validate_feature_standardization(
    feature_standardization: dict[str, dict[str, float]] | None,
    model_features: list[str],
) -> None:
    """`feature_standardization` (2026-09-12 EKLENDI, Baris karari Secenek
    A) sozlugunun sozlesmesini dogrular -- FAIL-LOUD, sessizce
    atlanmaz/duzeltilmez:
      - alan YOK/`None` -> hicbir kontrol yapilmaz (mevcut ham-olcek
        davranisi BIREBIR korunur, sifir-regresyon sarti).
      - listelenen bir kolon `model_features` (final_features +
        clinical_extra_columns) DISINDA -> `FeatureStandardizationConfigError`.
      - `mean`/`std` eksik, veya `std` None/0 (sifira bolme) ->
        `FeatureStandardizationConfigError`.
    """

    if not feature_standardization:
        return
    model_features_set = set(model_features)
    for column, stats in feature_standardization.items():
        if column not in model_features_set:
            raise FeatureStandardizationConfigError(
                f"'feature_standardization' kolonu {column!r} modelin "
                f"final_features+clinical_extra_columns kumesinde "
                f"({sorted(model_features_set)}) yok -- bilinmeyen/kapsam "
                "disi bir kolon icin olcekleme tanimlanamaz."
            )
        if not isinstance(stats, dict) or "mean" not in stats:
            raise FeatureStandardizationConfigError(
                f"'feature_standardization[{column!r}]' icin 'mean' eksik "
                f"(alinan: {stats!r})."
            )
        if "std" not in stats:
            raise FeatureStandardizationConfigError(
                f"'feature_standardization[{column!r}]' icin 'std' eksik "
                f"(alinan: {stats!r})."
            )
        mean_value = stats["mean"]
        std_value = stats["std"]
        if mean_value is None:
            raise FeatureStandardizationConfigError(
                f"'feature_standardization[{column!r}]' icin mean=None."
            )
        if std_value is None or float(std_value) == 0.0:
            raise FeatureStandardizationConfigError(
                f"'feature_standardization[{column!r}]' icin std=0 (veya "
                "None) -- sifira bolme, olcekleme UYGULANAMAZ."
            )


def _apply_feature_standardization(
    frame: pd.DataFrame,
    feature_standardization: dict[str, dict[str, float]] | None,
) -> pd.DataFrame:
    """`feature_standardization`'da listelenen HER kolon icin
    `(deger - mean) / std` donusumu uygular; listelenmeyen kolonlar
    DEGISTIRILMEZ. `frame` KOPYALANIR (girdi mutate edilmez). Alan
    YOK/`None` ise `frame` DEGISMEDEN doner -- v1 (ham-olcek) checkpoint'i
    icin BIREBIR ayni sonuc, sifir-regresyon sarti.

    Cagiran taraf `_validate_feature_standardization()`'i ONCEDEN
    cagirmis olmalidir (config-duzeyi fail-loud kontrolleri burada
    TEKRARLANMAZ) -- bu fonksiyon SADECE veri-duzeyinde eksik kolon
    kontrolu yapar: listelenen bir kolon `frame.columns`'ta yoksa
    `FeatureStandardizationColumnMissingError` firlatir (sessizce
    atlanmaz)."""

    if not feature_standardization:
        return frame
    result = frame.copy()
    for column, stats in feature_standardization.items():
        if column not in result.columns:
            raise FeatureStandardizationColumnMissingError(
                f"'feature_standardization' kolonu {column!r} standardize "
                f"edilecek veri setinde ({sorted(result.columns)}) yok."
            )
        mean = float(stats["mean"])
        std = float(stats["std"])
        result[column] = (result[column].astype(float) - mean) / std
    return result


def _normalize_arm(loaded: Any) -> dict[str, Any]:
    """Iki farkli pickle bicimini AYNI kucuk sozluge indirger:
    `{"name", "final_features", "extra_columns", "fitted_model",
    "age_rcs_knots", "feature_standardization"}`.

    Bicim A (2026-08-18 EKLENDI, varsayilan artefakt) -- sade `dict`,
    `tools.train_cox_week3` dataclass'larina BAGIMLI DEGIL: anahtarlar
    `arm_name`/`final_features`/`clinical_extra_columns`/`fitted_model`
    (+opsiyonel `age_rcs_knots`, bkz. modul dokstring'i "AGE RCS SPLINE
    AÇIK RISK" -- bugunku tek artefakt bunu TASIMIYOR, `None` doner;
    +opsiyonel `feature_standardization`, 2026-09-12 Baris karari Secenek
    A -- `{kolon: {"mean": float, "std": float}}`, YOK/`None` ise ham-olcek
    davranisi BIREBIR korunur).

    Bicim B (ESKI, 2026-08-15 checkpoint'i + `GBMAID_COX_CHECKPOINT_PATH`
    ile isaret edilebilecek benzer eski dosyalar) -- `tools.train_cox_
    week3.ArmResult` dataclass'i: `.name`, `.final_model.final_features`,
    `.final_model.fitted_model`, `.extra_columns` (+opsiyonel
    `.feature_standardization`, ayni sozlesme).

    Taninmayan bir ucuncu bicim gelirse (ne dict ne de bu ozniteliklere
    sahip obje) `TypeError` firlatilir -- SESSIZCE bos/varsayilan bir
    sonuc UYDURULMAZ. `feature_standardization` bozuksa (std==0, mean/std
    eksik, bilinmeyen kolon) `FeatureStandardizationConfigError` firlatilir
    (bkz. `_validate_feature_standardization`) -- bu da SESSIZCE
    yutulmaz/duzeltilmez.
    """

    if isinstance(loaded, dict):
        try:
            final_features = list(loaded["final_features"])
            extra_columns = list(loaded.get("clinical_extra_columns") or [])
            feature_standardization = loaded.get("feature_standardization")
            _validate_feature_standardization(
                feature_standardization, final_features + extra_columns
            )
            return {
                "name": loaded["arm_name"],
                "final_features": final_features,
                "extra_columns": extra_columns,
                "fitted_model": loaded["fitted_model"],
                "age_rcs_knots": loaded.get("age_rcs_knots"),
                "feature_standardization": feature_standardization,
            }
        except KeyError as exc:
            raise TypeError(
                f"Artefakt sozlugunde beklenen anahtar yok: {exc}. Beklenen: "
                "arm_name/final_features/fitted_model (+opsiyonel "
                "clinical_extra_columns/age_rcs_knots/feature_standardization)."
            ) from exc

    if hasattr(loaded, "final_model") and hasattr(loaded, "name"):
        final_features = list(loaded.final_model.final_features)
        extra_columns = list(getattr(loaded, "extra_columns", None) or [])
        feature_standardization = getattr(loaded, "feature_standardization", None)
        _validate_feature_standardization(
            feature_standardization, final_features + extra_columns
        )
        return {
            "name": loaded.name,
            "final_features": final_features,
            "extra_columns": extra_columns,
            "fitted_model": loaded.final_model.fitted_model,
            "age_rcs_knots": getattr(loaded, "age_rcs_knots", None),
            "feature_standardization": feature_standardization,
        }

    raise TypeError(
        f"Taninmayan artefakt bicimi: {type(loaded)!r}. Ne 'yeni sozluk' "
        "(arm_name/final_features/fitted_model) ne de 'eski ArmResult' "
        "(name/final_model/extra_columns) sozlesmesine uyuyor."
    )


def load_cox_arm_result(path: Path | None = None) -> dict[str, Any]:
    """Model artefaktini (varsayilan: `models/cox_phm_..._2026-08-18.pkl`,
    env override: `GBMAID_COX_CHECKPOINT_PATH`) YALNIZ OKUYARAK acar ve
    `_normalize_arm()` ile ORTAK bir sozluge indirger. mtime degismedigi
    surece process icinde tekrar unpickle ETMEZ (~0.5s/cagri olculdu --
    tools/shap_explainer_prototype.py ile ayni dosyada, bkz. o dosyanin
    calisma ciktisi).

    Artefakt dosyasi ELLE degistirilmiyor/silinmiyor/tasinmiyor --
    yalniz OKUNUYOR. Dosya mtime'i degisirse (ornek: yeni egitim kosusu
    ayni dosya adina yeniden yazdi) cache OTOMATIK gecersiz sayilir,
    process YENIDEN BASLATILMASI GEREKMEZ.
    """

    target = path or _checkpoint_path()
    if not target.is_file():
        raise CheckpointNotFoundError(
            f"Model artefakti bulunamadi: {target}. Egitim kosusu (tools/"
            "train_cox_week3.py) veya kaydetme adimi henuz calismamis "
            "olabilir, ya da GBMAID_COX_CHECKPOINT_PATH yanlis bir yolu "
            "isaret ediyor."
        )

    mtime = target.stat().st_mtime
    cache_key = str(target)
    with _checkpoint_cache_lock:
        cached = _checkpoint_cache.get(cache_key)
        if cached is not None and cached["mtime"] == mtime:
            return cached["arm"]

        # KIRILGANLIK (ESKI ArmResult bicimi icin GECERLI, YENI sade-sozluk
        # artefaktinda GEREKMEZ ama zararsizdir): eski pkl `tools.train_
        # cox_week3` altinda `__main__` varsayimiyla pickle'landi (script
        # `python tools/train_cox_week3.py ...` ile CALISTIRILDIGI icin) --
        # duz `pickle.load()` `AttributeError` verir. Cozum: import edip
        # `sys.modules['__main__']`'a GECICI ata, `finally`'de geri al
        # (lock ICINDE -- eszamanli istekler `sys.modules['__main__']`'i
        # BIRBIRINE KARISTIRMASIN diye).
        import tools.train_cox_week3 as train_module

        original_main = sys.modules.get("__main__")
        sys.modules["__main__"] = train_module
        try:
            with target.open("rb") as fh:
                loaded = pickle.load(fh)
        finally:
            if original_main is not None:
                sys.modules["__main__"] = original_main
            else:  # pragma: no cover -- normal calistirmada __main__ hep var
                del sys.modules["__main__"]

        arm = _normalize_arm(loaded)
        _checkpoint_cache[cache_key] = {"mtime": mtime, "arm": arm}
        return arm


# =====================================================================
# 2) DB'den hasta radyomigi -- yalniz SELECT, C32 beyaz liste ZORUNLU
# =====================================================================


def _get_db_connection():
    return get_connection(readonly=True)


def fetch_c32_radiomic_row(patient_id: str, feature_columns: list[str]) -> pd.DataFrame:
    """Bir hastanin C32 whitelist'indeki radyomik satir(lar)ini ceker ve
    `feature_columns` sutunlarina pivotlar -- degerler DB'de ne ise O
    (HAM), hicbir olcekleme UYGULANMAZ. Hata sozlesmesi (endpoint bu
    istisnalari HTTP durum koduna cevirir):
      - hasta `patients` tablosunda yok -> `PatientNotFoundError`
      - hasta var ama C32 whitelist'inde satir yok -> `C32RadiomicsNotFoundError`
      - hastanin BIRDEN FAZLA `scan_id`'si var -> `MultiScanNotSupportedError`
      - gerekli bolge (WT vb.) pivotlanamiyor -> `RequiredRegionMissingError`

    2026-09-13 (XGBoost ADIM 2) -- BU FONKSIYON `fetch_c32_wt_row()`'un
    GOVDESIDIR, tek degisiklik parametre ADI (`final_features` ->
    `feature_columns`): XGBoost ikinci katmani Cox'un SECTIGI 16 ozellik
    DEGIL, 93 ozelliklik HAM aday havuzunun TAMAMINI istiyor; ayni
    govdeyi iki farkli kolon listesiyle cagirabilmek icin isim
    genellestirildi. `fetch_c32_wt_row()` AYNEN korunuyor (imza/davranis/
    istisnalar BIREBIR ayni, monkeypatch eden mevcut testler BOZULMADI) --
    bu fonksiyona DELEGE eder.

    2026-09-13 (Y1, LUMIERE kanonik vizit) -- IKI YENI DAL:
      - hasta LUMIERE ise (`patient_id` oneki `Patient-`) tarama sayisi
        KAC OLURSA OLSUN K2 kanonik `Rating=='Pre-Op'` viziti secilir
        (`LumierePreopVisitNotAvailableError` -> 422 aday yoksa);
      - hasta LUMIERE DEGILSE davranis DEGISMEDI: coklu tarama
        `MultiScanNotSupportedError` (422).
    """

    conn = _get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("SELECT 1 FROM patients WHERE patient_id = %s", (patient_id,))
        if cur.fetchone() is None:
            raise PatientNotFoundError(patient_id)

        cur.execute(
            """
            SELECT p.patient_id, ds.source_name AS source, r.tumor_region,
                   r.shape_features, r.first_order_features, r.texture_features,
                   r.scan_id, ms.timepoint_label
            FROM radiomics r
            JOIN mr_scans ms ON ms.scan_id = r.scan_id
            JOIN patients p ON p.patient_id = ms.patient_id
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE p.patient_id = %s AND r.segmentation_tool = ANY(%s)
            """,
            (patient_id, list(ALLOWED_C32_SEGMENTATION_TOOLS)),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    if not rows:
        raise C32RadiomicsNotFoundError(
            f"Hasta {patient_id!r} icin C32 whitelist'inde "
            f"({sorted(ALLOWED_C32_SEGMENTATION_TOOLS)}) radyomik satiri yok."
        )

    long_frame = pd.DataFrame(rows)
    n_distinct_scans = long_frame["scan_id"].nunique()
    if is_lumiere_patient_id(patient_id):
        # K2/Y1 -- kural `pipeline/lumiere_canonical_visit.py`'de, BURADA
        # DEGIL. Tek taramali LUMIERE hastasi da BU yoldan gecer (bkz.
        # modul dokstring'i: eski 4 hastanin tek taramasi `Post-Op`tu).
        selection = select_lumiere_canonical_visit_from_long_frame(patient_id, long_frame)
        long_frame = long_frame[long_frame["scan_id"] == selection.scan_id].copy()
    elif n_distinct_scans > 1:
        source_names = sorted({str(value) for value in long_frame["source"].dropna().unique()})
        raise MultiScanNotSupportedError(
            f"Hasta {patient_id!r}: {n_distinct_scans} farkli scan_id C32 "
            "whitelist'inde bulundu (longitudinal/coklu-zaman-noktali) -- v1 "
            "tek-zaman-noktali hasta varsayar, hangi taramanin kullanilacagina "
            f"SESSIZCE karar VERILMEDI. Kaynak: {source_names}. LUMIERE'nin "
            "K2 kanonik-vizit kurali (Rating=='Pre-Op', 2026-09-13/Y1) BU "
            "kaynak icin TANIMLI DEGILDIR -- bu kaynaklarda hangi taramanin "
            "'ameliyat oncesi' oldugunu soyleyen bir ExpertRating tablosu YOK, "
            "kurali gevsetmek sessizce yanlis tarama secmek olurdu."
        )
    long_frame = long_frame.drop(columns=["scan_id", "timepoint_label"], errors="ignore")

    regions_needed = sorted({col.split("__", 1)[0] for col in feature_columns})
    try:
        wide, report = pivot_radiomics_long_to_wide(
            long_frame, regions_needed, on_missing_region="raise"
        )
    except RegionPivotError as exc:
        raise RequiredRegionMissingError(str(exc)) from exc
    except ValueError as exc:
        # 2026-09-12 rag-agent bulgusu (canli UCSF kaniti): `pivot_radiomics_
        # long_to_wide()` -> `canonicalize_region_label()` taninmayan bir
        # (kaynak, ham-bolge) cifti icin bare `ValueError` firlatiyor (kok
        # neden `pipeline/cox_model.py::REGION_NAME_ALIASES`'te -- o dosyaya
        # BURADAN DOKUNULMUYOR). Yakalanmamis birakilirsa endpoint'te
        # aciklanmamis 500 olarak sizar -- burada isimlendirilmis bir hataya
        # cevrilir (SESSIZCE yutulmaz).
        raise RegionCanonicalizationError(
            f"Hasta {patient_id!r}: radyomik satir(lar)i bolge/kaynak "
            f"kanoniklestirmesinden gecemedi (REGION_NAME_ALIASES bu "
            f"kaynagi/bolgeyi henuz tanimiyor olabilir): {exc}"
        ) from exc

    missing = set(feature_columns) - set(wide.columns)
    if missing:
        raise RequiredRegionMissingError(
            f"Hasta {patient_id!r}: radyomik satiri var ama beklenen "
            f"ozellik(ler) eksik: {sorted(missing)} (rapor: {report})."
        )
    return wide[feature_columns]


def fetch_c32_wt_row(patient_id: str, final_features: list[str]) -> pd.DataFrame:
    """Cox yolunun (2026-08-18'den beri degismeyen) girisi -- `fetch_c32_
    radiomic_row()`'a DELEGE eder, davranisi/istisnalari BIREBIR AYNIDIR.
    Ayri bir isim olarak KORUNUYOR cunku (a) cagiran kod/testler bu ismi
    monkeypatch ediyor, (b) "Cox'un final_features'ini cek" niyeti
    (XGBoost'un "93 ham aday havuzunu cek" niyetinden farkli) cagri
    yerinde okunur kalsin."""

    return fetch_c32_radiomic_row(patient_id, final_features)


# =====================================================================
# 2z) LUMIERE KANONIK VIZIT (K2 kilitli kural, Y1 -- 2026-09-13)
#     Kural `pipeline/lumiere_canonical_visit.py`'de TEK KAYNAK olarak
#     durur; buradaki fonksiyonlar YALNIZ (a) DB'den vizit anahtarlarini
#     cekmek, (b) secim sonucunu yanit blogu hâline getirmekle
#     ilgilenir. HICBIRI kurali YENIDEN UYGULAMAZ.
# =====================================================================

LUMIERE_CANONICAL_VISIT_RULE_NOTE = (
    "LUMIERE hastalari icin risk skoru YALNIZ K2 kanonik vizitinden "
    "(LUMIERE ExpertRating CSV'sinde `Rating == 'Pre-Op'`) uretilir -- Cox "
    "modeli TEK bir ameliyat-ONCESI T1ce uzerine kuruldu. `timepoint_label` "
    "siralamasi ('en erken taramayi al') OLCUT DEGILDIR: `week-000` etiketli "
    "135 satirin 44'u `Post-Op`, `Patient-020`'de `week-000` Post-Op ama "
    "`week-000-1` Pre-Op, `Patient-060`'in bir Pre-Op viziti `week-069`'da. "
    "Ayni hastada birden fazla Pre-Op+C32 viziti varsa EN ERKEN olan "
    "secilir (yalniz tie-break). Kural: decisions/2026-09-13-lumiere-risk-"
    "skoru-preop-kanonik-vizit.md + K2 (2026-08-18). LUMIERE EGITIME "
    "GIRMEMISTIR (olay verisi 0/91) -- bu bir DIS DOGRULAMA DEGIL, modelin "
    "uygulanmasidir."
)


def select_lumiere_canonical_visit_from_long_frame(
    patient_id: str, long_frame: pd.DataFrame
) -> CanonicalVisitSelection:
    """Bir LUMIERE hastasinin C32 satirlarindan (uzun format, `scan_id` +
    `timepoint_label` kolonlari ZORUNLU) kanonik `Pre-Op` vizitini secer.

    Secimi YAPAN kod `pipeline.lumiere_canonical_visit.select_canonical_
    preop_visit_for_patient()`tir -- burada kural YOK, yalniz veri
    hazirligi ve YAPISAL tutarlilik guard'lari var."""

    if not is_lumiere_patient_id(patient_id):
        raise LumiereCanonicalVisitDataError(
            f"{patient_id!r} LUMIERE deseninde ({LUMIERE_PATIENT_ID_PREFIX!r} "
            "oneki) DEGIL -- kanonik vizit kurali bu hastaya UYGULANMAZ."
        )
    if "timepoint_label" not in long_frame.columns:
        raise LumiereCanonicalVisitDataError(
            f"Hasta {patient_id!r}: `mr_scans.timepoint_label` kolonu sorgu "
            "sonucunda YOK -- K2 kanonik-vizit kurali bu kolon olmadan "
            "UYGULANAMAZ, sessizce rastgele bir tarama SECILMEDI."
        )
    if "source" in long_frame.columns:
        sources = sorted({str(value) for value in long_frame["source"].dropna().unique()})
        if sources and sources != [LUMIERE_SOURCE_NAME]:
            raise LumiereCanonicalVisitDataError(
                f"Hasta {patient_id!r}: `patient_id` oneki LUMIERE deseninde "
                f"ama `dataset_sources.source_name` = {sources} (beklenen "
                f"[{LUMIERE_SOURCE_NAME!r}]) -- kimlik/kaynak celiskisi, "
                "kanonik vizit kurali SESSIZCE uygulanmadi (bkz. tools/"
                "data_integrity_check.py::PATIENT_ID_PREFIX_RULES)."
            )

    visit_keys = [
        (str(label), scan_id)
        for label, scan_id in long_frame[["timepoint_label", "scan_id"]]
        .drop_duplicates()
        .itertuples(index=False)
    ]
    return select_canonical_preop_visit_for_patient(
        patient_id, visit_keys, csv_path=LUMIERE_EXPERT_RATING_CSV_PATH
    )


def fetch_lumiere_canonical_visit_selection(patient_id: str) -> CanonicalVisitSelection:
    """LUMIERE hastasinin kanonik vizitini DB'den (yalniz SELECT, JSONB
    ozellik kolonlari CEKILMEZ -- hafif sorgu) belirler.

    Endpoint bunu YANIT SEFFAFLIGI icin cagirir (`fetch_c32_wt_row` ayni
    kurali kendi icinde ZATEN uygulamistir; ikisi AYNI fonksiyona
    (`select_canonical_preop_visit_for_patient`) gider, kural TEK
    yerdedir)."""

    conn = _get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT DISTINCT ms.timepoint_label, r.scan_id,
                   ds.source_name AS source, p.patient_id
            FROM radiomics r
            JOIN mr_scans ms ON ms.scan_id = r.scan_id
            JOIN patients p ON p.patient_id = ms.patient_id
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE p.patient_id = %s AND r.segmentation_tool = ANY(%s)
            """,
            (patient_id, list(ALLOWED_C32_SEGMENTATION_TOOLS)),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    if not rows:
        raise C32RadiomicsNotFoundError(
            f"Hasta {patient_id!r} icin C32 whitelist'inde radyomik satiri yok "
            "-- kanonik vizit belirlenemez."
        )
    return select_lumiere_canonical_visit_from_long_frame(patient_id, pd.DataFrame(rows))


def build_lumiere_canonical_visit_block(selection: CanonicalVisitSelection) -> dict[str, Any]:
    """Yanitin `lumiere_canonical_visit` blogu -- HANGI vizitin kullanildigi
    ACIKCA beyan edilir (Y1 gorevi: seffaflik sarti)."""

    return {
        "applied": True,
        "rule": "K2: Rating == 'Pre-Op' (LUMIERE ExpertRating)",
        "timepoint_label": selection.timepoint_label,
        "rating": selection.rating,
        "scan_id": selection.scan_id,
        "n_preop_visits_in_expert_rating": selection.n_preop_visits_in_csv,
        "n_candidate_visits_with_c32": selection.n_candidate_visits_with_c32,
        "tie_broken_earliest_week": selection.tie_broken,
        "candidate_visits_with_c32": list(selection.candidate_visits),
        "expert_rating_source": Path(LUMIERE_EXPERT_RATING_CSV_PATH).name,
        "rule_note": LUMIERE_CANONICAL_VISIT_RULE_NOTE,
    }


# =====================================================================
# 2a) Klinik kovaryatlar (2026-08-18 EKLENDI) -- `patients` tablosundan
#     ham deger cekilir, tools/train_cox_week3.py::build_clinical_
#     covariate_frame'in KODLAMA DESENI burada BAGIMSIZ olarak yeniden
#     uygulanir (o dosyaya DOKUNULMUYOR/IMPORT EDILMIYOR -- gorev
#     talimati, ayrica o dosya bugun baska bir aktif kosunun/agent'in
#     calisma alani). Egitimdeki (tolerant, fillna+eksik-gosterge)
#     kodlama ile matematiksel olarak AYNI -- fark, TEK HASTA tahmininde
#     bu kodlamaya girmeden ONCE ham degerin NULL/taninmayan olup
#     olmadigi kontrol edilir (bkz. asagida "strict").
#
#     2026-09-12 GUNCELLEME (K19 karari, Baris, Secenek 2 -- bkz.
#     `CLINICAL_COVARIATE_NULL_POLICY`): bu kontrol ARTIK TUM kovaryatlar
#     icin AYNI/universal-SIKI DEGIL, kovaryat basina tanimli bir
#     politikaya gore davranir:
#       - `STRICT` (age/gender): ham deger NULL ise 422 -- sessiz
#         varsayilan/eksik-gosterge ATANMAZ (yapisal sebep: egitimde
#         bunlara karsilik gelen bir `*_missing` kolonu YOK).
#       - `TRAIN_ALIGNED` (gtr_over90percent/idh1_status/mgmt_status):
#         ham deger NULL ise 422 ATILMAZ, egitimdeki TOLERANT kodlamayla
#         AYNI sekilde `clinical_*_missing=1` olarak KODLANIR. mgmt_status
#         gerekcesi: v3b_lowvar_v2amgmt'nin harici dogrulamasi (UCSF
#         n=295) NULL-MGMT hastalarini zaten `mgmt_missing=1` ile
#         skorladi -- servisin bu hastalari reddetmesi "dogrulanan
#         populasyon" ile "hizmet edilen populasyon"u ayristiriyordu
#         (bkz. decisions/2026-09-12-k19-mgmt-null-politikasi.md).
#         2026-09-13 GUNCELLEME (Baris onayi): GTR/IDH1 de AYNI mantikla
#         TRAIN_ALIGNED'e HIZALANDI -- eski hali (D6 karari, 2026-09-12,
#         ayni gun) BILINCLI bir asimetriydi, `tools/train_cox_week3.py`
#         (satir 663-687) ucunu de BIREBIR AYNI desenle kodladigi icin
#         asimetri servis-ozgu ve gereksizdi. Taninmayan (NULL-DISI,
#         ornek yazim hatasi) bir deger HER IKI politikada da HER ZAMAN
#         422 ATAR (politikadan bagimsiz -- bu "bilinen eksiklik" degil,
#         veri bozuklugu).
# =====================================================================


_CLINICAL_RAW_COLUMNS: tuple[str, ...] = (
    "age",
    "gender",
    "gtr_over90percent",
    "idh1_status",
    "mgmt_status",
)


def _fetch_patients_clinical_raw(patient_ids: list[str]) -> pd.DataFrame:
    """Birden fazla `patient_id` icin ham klinik kolonlari (`age`,
    `gender`, `gtr_over90percent`, `idh1_status`, `mgmt_status`)
    DataFrame olarak doner (index=patient_id sirasiyla DEGIL, `patient_id`
    kolonu ile).

    2026-09-13 GUNCELLEME (Baris onayi -- bkz. yukaridaki "HAM ETIKET
    NORMALIZASYONU" bloğu): DB'den okunan ham deger, dondurulmeden ONCE
    `_normalize_clinical_categorical_raw_values()`'tan GECER -- bu, bu
    modulun `idh1_status`/`mgmt_status` OKUYAN TEK noktasi oldugu icin
    (tek-hasta `build_patient_clinical_covariate_row()` ve SHAP arka-plan
    orneklemi, ikisi de BU fonksiyonu cagirir) normalizasyonun UYGULANDIGI
    TEK yer burasidir -- baska bir yerde TEKRARLANMASI gerekmez. DB'nin
    KENDISI DEGISMEZ, sadece bu surecteki DataFrame normalize edilir."""

    if not patient_ids:
        return pd.DataFrame(columns=["patient_id", *_CLINICAL_RAW_COLUMNS])
    conn = _get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            f"""
            SELECT patient_id, {", ".join(_CLINICAL_RAW_COLUMNS)}
            FROM patients
            WHERE patient_id = ANY(%s)
            """,
            (patient_ids,),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()
    return _normalize_clinical_categorical_raw_values(pd.DataFrame(rows))


def _needs_age_raw(extra_columns: list[str]) -> bool:
    """Yas ham degeri (`age`), HEM duz `clinical_age` HEM de spline
    (`clinical_age_rcs1`/`_rcs2`) kodlamasi icin gereklidir."""

    return bool(
        {CLINICAL_AGE_COLUMN, CLINICAL_AGE_RCS1_COLUMN, CLINICAL_AGE_RCS2_COLUMN}
        & set(extra_columns)
    )


def _restricted_cubic_spline_basis(age: pd.Series, knots: tuple[float, float, float]) -> pd.DataFrame:
    """`tools/train_cox_week3.py::restricted_cubic_spline_basis()`'in
    3-dugumlu (t1<t2<t3) formulunun BAGIMSIZ kopyasi (o dosyaya
    DOKUNULMUYOR/IMPORT EDILMIYOR, matematik AYNI -- regresyon testiyle
    dogrulanmali, bkz. tests/test_api_predict.py). Dugumler ARTEFAKTIN
    KENDI `age_rcs_knots` alanindan gelir, burada YENIDEN HESAPLANMAZ."""

    t1, t2, t3 = (float(k) for k in knots)
    if not (t1 < t2 < t3):
        raise ValueError(f"Dugumler KESIN ARTAN olmali (t1<t2<t3), alinan: {(t1, t2, t3)}")

    x = age.astype(float)

    def _pos_cubed(v: pd.Series) -> pd.Series:
        return v.clip(lower=0.0) ** 3

    denom = (t3 - t1) ** 2
    nonlinear_term = (
        _pos_cubed(x - t1)
        - _pos_cubed(x - t2) * (t3 - t1) / (t3 - t2)
        + _pos_cubed(x - t3) * (t2 - t1) / (t3 - t2)
    ) / denom

    return pd.DataFrame(
        {CLINICAL_AGE_RCS1_COLUMN: x, CLINICAL_AGE_RCS2_COLUMN: nonlinear_term}, index=age.index
    )


def _encode_clinical_frame(
    raw_frame: pd.DataFrame,
    extra_columns: list[str],
    *,
    age_rcs_knots: tuple[float, float, float] | None = None,
) -> pd.DataFrame:
    """`raw_frame` (kolonlar: `_CLINICAL_RAW_COLUMNS`, index korunur)
    icindeki ham degerleri `extra_columns`'un istedigi sayisal kolonlara
    cevirir -- HANGI kolonlarin uretilecegi TAMAMEN `extra_columns`'tan
    (yani ARTEFAKTTEN) TURETILIR, model/varyant adi burada SABIT
    KODLANMAZ (2026-08-18 model-agnostik karari, Baris).

    TOLERANT (fillna+eksik-gosterge) -- egitimdeki desenle AYNI.
    Taninmayan bir kategorik deger (yazim hatasi/sema disi)
    `ClinicalCovariateMissingError` firlatir (SESSIZCE gecilmez); bu
    fonksiyon TEK BASINA "NULL -> 422" kuralini UYGULAMAZ (o kural
    cagiran tarafta, hasta-ozel `strict` on-kontrolde) -- arka plan/SHAP
    orneklemi icin fillna davranisi BILINCLI/istenen.

    `clinical_age_rcs1`/`clinical_age_rcs2` istenirse `age_rcs_knots`
    ZORUNLUDUR (yoksa `ClinicalCovariatesNotSupportedError` -- 501,
    bkz. modul dokstring'i "AGE RCS SPLINE AÇIK RISK")."""

    needs_gender = CLINICAL_GENDER_MALE_COLUMN in extra_columns
    needs_gtr = CLINICAL_GTR_Y_COLUMN in extra_columns or CLINICAL_GTR_MISSING_COLUMN in extra_columns
    needs_idh = CLINICAL_IDH_MUTANT_COLUMN in extra_columns or CLINICAL_IDH_MISSING_COLUMN in extra_columns
    needs_mgmt = (
        CLINICAL_MGMT_METHYLATED_COLUMN in extra_columns or CLINICAL_MGMT_MISSING_COLUMN in extra_columns
    )
    needs_age_spline = bool({CLINICAL_AGE_RCS1_COLUMN, CLINICAL_AGE_RCS2_COLUMN} & set(extra_columns))

    result = pd.DataFrame(index=raw_frame.index)
    if CLINICAL_AGE_COLUMN in extra_columns:
        result[CLINICAL_AGE_COLUMN] = raw_frame["age"].astype(float)

    if needs_age_spline:
        if age_rcs_knots is None:
            raise ClinicalCovariatesNotSupportedError(
                "Artefakt yas-spline kovaryati istiyor "
                f"({sorted({CLINICAL_AGE_RCS1_COLUMN, CLINICAL_AGE_RCS2_COLUMN} & set(extra_columns))}) "
                "ama 'age_rcs_knots' TASIMIYOR -- dugum noktalari egitim "
                "kohortundan (UPenn) BIR KEZ hesaplanmasi gereken bir "
                "artefakt-duzeyi degerdir, tek hastadan YENIDEN "
                "TURETILEMEZ. Kapsam disi (bkz. modul dokstring'i)."
            )
        spline = _restricted_cubic_spline_basis(raw_frame["age"].astype(float), age_rcs_knots)
        result[CLINICAL_AGE_RCS1_COLUMN] = spline[CLINICAL_AGE_RCS1_COLUMN]
        result[CLINICAL_AGE_RCS2_COLUMN] = spline[CLINICAL_AGE_RCS2_COLUMN]

    if needs_gender:
        gender_raw = raw_frame["gender"]
        unrecognized = gender_raw.dropna().loc[
            ~gender_raw.dropna().isin({GENDER_MALE_LABEL, GENDER_FEMALE_LABEL})
        ]
        if not unrecognized.empty:
            raise ClinicalCovariateMissingError(
                f"'gender' kolonunda taninmayan deger(ler): {sorted(unrecognized.unique())}"
            )
        result[CLINICAL_GENDER_MALE_COLUMN] = gender_raw.map(
            {GENDER_MALE_LABEL: 1.0, GENDER_FEMALE_LABEL: 0.0}
        )

    if needs_gtr:
        gtr_raw = raw_frame["gtr_over90percent"]
        unrecognized = gtr_raw.dropna().loc[
            ~gtr_raw.dropna().isin({GTR_YES_LABEL, GTR_NO_LABEL})
        ]
        if not unrecognized.empty:
            raise ClinicalCovariateMissingError(
                f"'gtr_over90percent' kolonunda taninmayan deger(ler): {sorted(unrecognized.unique())}"
            )
        result[CLINICAL_GTR_MISSING_COLUMN] = gtr_raw.isna().astype(float)
        result[CLINICAL_GTR_Y_COLUMN] = gtr_raw.map(
            {GTR_YES_LABEL: 1.0, GTR_NO_LABEL: 0.0}
        ).fillna(0.0)

    if needs_idh:
        idh_raw = raw_frame["idh1_status"]
        unrecognized = idh_raw.dropna().loc[
            ~idh_raw.dropna().isin({IDH1_WILDTYPE_LABEL, IDH1_MUTATED_LABEL, IDH1_NOS_LABEL})
        ]
        if not unrecognized.empty:
            raise ClinicalCovariateMissingError(
                f"'idh1_status' kolonunda taninmayan deger(ler): {sorted(unrecognized.unique())}"
            )
        idh_missing_mask = idh_raw.isna() | (idh_raw == IDH1_NOS_LABEL)
        result[CLINICAL_IDH_MISSING_COLUMN] = idh_missing_mask.astype(float)
        result[CLINICAL_IDH_MUTANT_COLUMN] = idh_raw.map(
            {IDH1_MUTATED_LABEL: 1.0, IDH1_WILDTYPE_LABEL: 0.0}
        ).fillna(0.0)

    if needs_mgmt:
        mgmt_raw = raw_frame["mgmt_status"]
        unrecognized = mgmt_raw.dropna().loc[
            ~mgmt_raw.dropna().isin(
                {MGMT_METHYLATED_LABEL, MGMT_UNMETHYLATED_LABEL, MGMT_INDETERMINATE_LABEL}
            )
        ]
        if not unrecognized.empty:
            raise ClinicalCovariateMissingError(
                f"'mgmt_status' kolonunda taninmayan deger(ler): {sorted(unrecognized.unique())}"
            )
        mgmt_missing_mask = mgmt_raw.isna() | (mgmt_raw == MGMT_INDETERMINATE_LABEL)
        result[CLINICAL_MGMT_MISSING_COLUMN] = mgmt_missing_mask.astype(float)
        result[CLINICAL_MGMT_METHYLATED_COLUMN] = mgmt_raw.map(
            {MGMT_METHYLATED_LABEL: 1.0, MGMT_UNMETHYLATED_LABEL: 0.0}
        ).fillna(0.0)

    missing_cols = set(extra_columns) - set(result.columns)
    if missing_cols:
        raise ClinicalCovariateMissingError(
            f"Kodlama sonrasi beklenen kolon(lar) uretilemedi: {sorted(missing_cols)}."
        )
    return result[extra_columns]


def build_patient_clinical_covariate_row(
    patient_id: str,
    extra_columns: list[str],
    *,
    age_rcs_knots: tuple[float, float, float] | None = None,
) -> pd.DataFrame:
    """TEK hasta icin klinik kovaryat satirini uretir: ham `age`/
    `gender`/`gtr_over90percent`/`idh1_status`/`mgmt_status`
    kolonlarindan modelin ihtiyac duydugu HERHANGI biri taninmayan bir
    deger tasiyorsa (veya NULL ise VE o kovaryatin politikasi `STRICT`
    ise) `ClinicalCovariateMissingError` firlatilir (endpoint bunu
    422'ye cevirir).

    2026-09-12 GUNCELLEME (K19 karari, Baris, Secenek 2 -- bkz.
    `CLINICAL_COVARIATE_NULL_POLICY`): NULL icin davranis ARTIK
    UNIVERSAL-SIKI DEGIL, kovaryat basina tanimli:
      - `age`/`gender`: `STRICT` -- NULL 422 ATAR, eskisi gibi (yapisal
        sebep: egitimde bunlara karsilik gelen bir `*_missing` gosterge
        kolonu YOK).
      - `gtr_over90percent`/`idh1_status`/`mgmt_status`: `TRAIN_ALIGNED`
        -- NULL 422 ATMAZ, egitimdeki TOLERANT kodlamayla
        (`_encode_clinical_frame`) AYNI sekilde `clinical_*_missing=1`
        olarak KODLANIR. `mgmt_status` gerekcesi: v3b_lowvar_v2amgmt'nin
        harici dogrulamasi (UCSF n=295) zaten bu sekilde (NULL ->
        mgmt_missing=1) skorlanmisti -- "dogrulanan populasyon" ile
        "hizmet edilen populasyon"un ayni olmasi icin servis egitimle
        hizalandi (bkz. decisions/2026-09-12-k19-mgmt-null-politikasi.md).

    2026-09-13 GUNCELLEME (Baris onayi): `gtr_over90percent`/
    `idh1_status` DA MGMT gibi `TRAIN_ALIGNED`'e HIZALANDI (eski hali,
    D6 karari 2026-09-12 ayni gun, BILINCLI bir asimetriydi -- gerekce
    TUTARLILIK: `tools/train_cox_week3.py` satir 663-687 ucunu de
    BIREBIR AYNI desenle kodluyor, asimetri sadece serviste vardi).
    ⚠️ Bu, GTR/IDH/MGMT'si NULL bir hastanin (ornek: TCGA-06-5412) ARTIK
    422 DEGIL 200 alabilecegi anlamina gelir -- jury demo anlatisi buna
    gore GUNCELLENMELI (kod tarafinda cozulmedi, sadece bildirilir).

    HER IKI politikada da NULL-DISI ama taninmayan bir deger HER ZAMAN
    reddedilir -- bu bir "bilinen eksiklik" degil veri bozuklugudur.
    IDH1 icin `NOS/NEC` ve MGMT icin `Indeterminate` ("test yapildi,
    sonuc kullanilamaz") ise HER ZAMAN "bilinen eksiklik" sentinel'i
    sayilir (egitimde de aynen boyle modellendigi icin -- K15 karari +
    MGMT_INDETERMINATE_LABEL notu) -- 422 TETIKLEMEZ, ilgili
    `*_missing=1` olarak KODLANIR (bu davranis politikadan BAGIMSIZ,
    degismedi).

    `extra_columns` HANGI alanlarin gerekli oldugunu BELIRLER -- bu
    fonksiyon hicbir varyanti (v1/v2a/v2b/v2c) VARSAYMAZ, SADECE
    artefaktin istedigi kolonlara BAKAR (model-agnostik).
    """

    raw_frame = _fetch_patients_clinical_raw([patient_id])
    if raw_frame.empty:
        raise ClinicalCovariateMissingError(
            f"Hasta {patient_id!r} icin patients tablosunda klinik satir bulunamadi."
        )
    raw = raw_frame.iloc[0]

    # NOT: `age`/`gender` HER ZAMAN STRICT kalir (yapisal sebep -- egitim
    # kodlamasinda bu ikisi icin bir "*_missing" gosterge kolonu YOK,
    # `CLINICAL_COVARIATE_NULL_POLICY`'de TRAIN_ALIGNED'e cevirmenin bir
    # karsiligi olmazdi). GTR/IDH1/MGMT ucu de (2026-09-13'ten itibaren)
    # TRAIN_ALIGNED -- ucunun de egitimde bir eslesen `*_missing` kolonu
    # vardir, bkz. `_clinical_raw_value_is_rejected()`.
    missing_fields: list[str] = []
    if _needs_age_raw(extra_columns) and pd.isna(raw["age"]):
        missing_fields.append("age (NULL)")
    if CLINICAL_GENDER_MALE_COLUMN in extra_columns and raw["gender"] not in (
        GENDER_MALE_LABEL,
        GENDER_FEMALE_LABEL,
    ):
        missing_fields.append(f"gender (deger={raw['gender']!r})")
    needs_gtr = CLINICAL_GTR_Y_COLUMN in extra_columns or CLINICAL_GTR_MISSING_COLUMN in extra_columns
    if needs_gtr and _clinical_raw_value_is_rejected(
        raw["gtr_over90percent"],
        frozenset({GTR_YES_LABEL, GTR_NO_LABEL}),
        raw_column="gtr_over90percent",
    ):
        missing_fields.append(f"gtr_over90percent (deger={raw['gtr_over90percent']!r})")
    needs_idh = CLINICAL_IDH_MUTANT_COLUMN in extra_columns or CLINICAL_IDH_MISSING_COLUMN in extra_columns
    if needs_idh and _clinical_raw_value_is_rejected(
        raw["idh1_status"],
        frozenset({IDH1_WILDTYPE_LABEL, IDH1_MUTATED_LABEL, IDH1_NOS_LABEL}),
        raw_column="idh1_status",
    ):
        missing_fields.append(f"idh1_status (deger={raw['idh1_status']!r})")
    needs_mgmt = (
        CLINICAL_MGMT_METHYLATED_COLUMN in extra_columns or CLINICAL_MGMT_MISSING_COLUMN in extra_columns
    )
    if needs_mgmt and _clinical_raw_value_is_rejected(
        raw["mgmt_status"],
        frozenset({MGMT_METHYLATED_LABEL, MGMT_UNMETHYLATED_LABEL, MGMT_INDETERMINATE_LABEL}),
        raw_column="mgmt_status",
    ):
        missing_fields.append(f"mgmt_status (deger={raw['mgmt_status']!r})")

    if missing_fields:
        raise ClinicalCovariateMissingError(
            f"Hasta {patient_id!r}: model klinik kovaryat gerektiriyor ama "
            f"su alan(lar) eksik/NULL/taninmayan: {missing_fields}. Sessiz "
            "varsayilan/sifir ATANMADI -- bu hastaya bu modelle tahmin "
            "URETILEMEZ."
        )

    encoded = _encode_clinical_frame(
        raw_frame.set_index("patient_id"), extra_columns, age_rcs_knots=age_rcs_knots
    )
    encoded = encoded.reset_index(drop=True)
    return encoded


# =====================================================================
# 2b) Omics yorumu (Adim 7, OPSIYONEL) -- plan.txt:145-151, v45.txt SS 6.4
#     Cox'un ZORUNLU girdisi DEGIL: bu bolumdeki hicbir fonksiyon
#     `compute_risk_score_and_shap()`'e girdi VERMEZ, endpoint bu
#     fonksiyonlari Cox/SHAP sonucu HESAPLANDIKTAN SONRA cagirir.
# =====================================================================

# molecular_scores'tan cekilen ham kolonlar -- skor FORMULLERI burada
# YENIDEN HESAPLANMAZ, yalniz DB'deki mevcut degerler sunulur (gorev
# talimati siniri).
_MOLECULAR_SCORE_COLUMNS: tuple[str, ...] = (
    "tmz_resistance_score",
    "tmz_class",
    "tmz_class_relative",
    "aggressiveness_score",
    "aggr_class",
    "dna_repair_score",
    "repair_class",
    "molecular_subtype",
    "egfr_amp_flag",
    "pten_del_flag",
    "cdkn2a_del_flag",
    "mgmt_interpretation",
    "score_version",
)


def _to_float_or_none(value: Any) -> float | None:
    """`molecular_scores`'un `numeric` kolonlari psycopg2'den `Decimal`
    olarak gelir -- JSON serilestirme icin `float`'a cevrilir, `NULL` ise
    `None` olarak KORUNUR (sessizce 0.0'a duzeltilmez)."""

    return float(value) if value is not None else None


def _check_tmz_class_consistency(
    score_row: dict[str, Any], thresholds: dict[str, dict[str, Any]]
) -> str:
    """DB'nin sakladigi `tmz_class`'in, ham `tmz_resistance_score` +
    canli `score_thresholds` esikleriyle YENIDEN hesaplanan siniftan
    FARKLI olup olmadigini kontrol eder.

    Codex review-gate bulgusu (2026-08-18): "API, DB'den gelen
    tmz_resistance_score ile tmz_class'i CAPRAZLAMADAN aynen donduruyor
    -- bozuk/eski bir DB satiri varsa celiskili bilgi sizabilir." Bu
    fonksiyon o capraz kontrolu EKLER, ama DB'deki `tmz_class` degerini
    SESSIZCE DUZELTMEZ -- yalniz bir uyari METNI doner, cagiran taraf
    (klinisyen arayuzu) bunu gormeli.

    Esik veya ham skor YOKSA (`score_thresholds` bos, veya `tmz_class`
    NULL) kontrol YAPILAMAZ -- bu durum "tutarli" ile KARISTIRILMAZ,
    ayri bir "cannot_verify" metniyle donuk kalir (Codex'in 5. maddesi:
    "tablo boş dönerse sessizce varsayılan eşik kullanma").
    """

    threshold = thresholds.get("tmz_resistance_score")
    raw_score = score_row.get("tmz_resistance_score")
    stored_class = score_row.get("tmz_class")
    if threshold is None or raw_score is None or stored_class is None:
        return (
            "cannot_verify -- score_thresholds'ta 'tmz_resistance_score' "
            "esigi yok veya ham skor/sinif NULL, capraz kontrol YAPILAMADI."
        )

    score = float(raw_score)
    p33 = float(threshold["p33_cutoff"])
    p66 = float(threshold["p66_cutoff"])
    expected_class = "sensitive" if score <= p33 else ("intermediate" if score <= p66 else "resistant")
    if expected_class != stored_class:
        return (
            f"TUTARSIZLIK: DB'deki tmz_class={stored_class!r} ama ham "
            f"tmz_resistance_score={score} + score_thresholds (p33={p33}, "
            f"p66={p66}) ile YENIDEN hesaplanan sinif={expected_class!r}. "
            "Bu satir gozden gecirilmeli (db-agent'e bildirin) -- Cox risk "
            "skorunu ETKILEMEZ, yalniz omics yorum blogu icin gecerli."
        )
    return f"consistent -- tmz_class={stored_class!r} esiklerle (p33={p33}, p66={p66}) uyusuyor."


def _build_omics_block(
    score_row: dict[str, Any], thresholds: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """`molecular_scores` satirindan yanit blogunu kurar.

    ACIK BULGU (2026-08-18 canli DB olcumu, gorev talimatinda sadece
    `tmz_class_relative` icin belirtilmisti): `egfr_amp_flag`,
    `pten_del_flag`, `cdkn2a_del_flag`, `mgmt_interpretation` DE 48/48
    satirda NULL. Bu fonksiyon bunlari `None` olarak DONDURUR, sessizce
    baska bir degere (ornek: False) DUZELTMEZ -- cagiran taraf (klinisyen
    arayuzu) `None`'u "bilinmiyor/yuklenmedi" olarak yorumlamali, "negatif"
    DEGIL.

    2026-08-18 GUNCELLENDI (Ege'nin bulgusu, backend-agent gorev talimati):
    eskiden SADECE `tmz_class_relative`'in acikca DOKUMENTE edilmis bir notu
    vardi, diger 4 NULL kolonun (`egfr_amp_flag`/`pten_del_flag`/
    `cdkn2a_del_flag`/`mgmt_interpretation`) yoktu -- bu tutarsizlik burada
    KAPATILDI. Kanonik metin (db-agent'in kaynak-sozlugu arastirmasindan,
    log/2026-08-18.md): "Kaynak veri sozlugunde (genom_verileri.xlsx, 11
    sayfa) bu alanin uretim kurali tanimli degil; 48/48 NULL, uretilmedi."
    NOT -- bu 4 alan `tmz_class_relative`'den DAHA koklu bir eksiklik tasir:
    `tmz_class_relative` en azindan turetilebilecegi bir ham skor/formul
    ICERIR (deprecated/duplicate sunum sorunu), bu 4'unun ise DB semasinda
    TUKETICI olarak tanimlanmis olmasi disinda hicbir URETICI formul/esik/
    kaynak kolonu YOK (tersine muhendislik UYGULANAMAZ, sifir ground-truth).
    """

    UNPRODUCED_FIELD_NOTE = (
        "Kaynak veri sozlugunde (genom_verileri.xlsx, 11 sayfa) bu alanin "
        "uretim kurali tanimli degil; 48/48 NULL, uretilmedi."
    )

    return {
        "has_omics": True,
        "available": True,
        "score_version": score_row["score_version"],
        "tmz_resistance_score": _to_float_or_none(score_row["tmz_resistance_score"]),
        "tmz_resistance_direction": (
            "DUSUK skor = temozolomide DUYARLI; YUKSEK skor = TMZ direnc "
            "ihtimali artmis (raw/mimari/v45.txt satir 453). Ana Cox "
            "karar skoruna DAHIL DEGIL -- yalniz biyolojik yorum."
        ),
        "tmz_class": score_row["tmz_class"],
        "tmz_class_consistency_check": _check_tmz_class_consistency(score_row, thresholds),
        "tmz_class_relative": score_row["tmz_class_relative"],
        "tmz_class_relative_note": (
            "molecular_scores.tmz_class_relative DB'de 48/48 satirda NULL "
            "(v4.1 mimarisinin tanimladigi 48-hasta donmus tertil siniri "
            "henuz yuklenmedi -- db-agent arastiriyor, 2026-08-18 itibariyle "
            "acik). `null` donmesi KOD HATASI DEGIL, bilinen veri "
            "eksikligidir."
        ),
        "aggressiveness_score": _to_float_or_none(score_row["aggressiveness_score"]),
        "aggr_class": score_row["aggr_class"],
        "dna_repair_score": _to_float_or_none(score_row["dna_repair_score"]),
        "repair_class": score_row["repair_class"],
        "molecular_subtype": score_row["molecular_subtype"],
        "egfr_amp_flag": score_row["egfr_amp_flag"],
        "egfr_amp_flag_note": UNPRODUCED_FIELD_NOTE,
        "pten_del_flag": score_row["pten_del_flag"],
        "pten_del_flag_note": UNPRODUCED_FIELD_NOTE,
        "cdkn2a_del_flag": score_row["cdkn2a_del_flag"],
        "cdkn2a_del_flag_note": UNPRODUCED_FIELD_NOTE,
        "mgmt_interpretation": score_row["mgmt_interpretation"],
        "mgmt_interpretation_note": UNPRODUCED_FIELD_NOTE,
        "score_thresholds": thresholds,
        "interpretation_note": (
            "aggressiveness_score ve dna_repair_score 48 hastalik omics "
            "kohortuna GORECELIDIR (kohort-ici z-score/normalizasyon icerir, "
            "raw/mimari/v45.txt satir 459-460); yeni bir hasta icin mutlak "
            "bir olcek IDDIA ETMEZ. "
            + (
                "`score_thresholds` tablosu su an 0 satir -- tmz_class/"
                "aggr_class/repair_class sinir degerleri hicbir yerde "
                "DONMUS degil, siniflandirma DOGRULANAMADI (canli DB "
                "olcumu -- eger bu notu goruyorsan tablo yeniden BOSALMIS "
                "olabilir, db-agent'e sorun)."
                if not thresholds
                else (
                    f"`score_thresholds`'ta DONMUS esikler VAR (surum "
                    f"{sorted({v['score_version'] for v in thresholds.values()})}, "
                    f"kohort n={sorted({v['cohort_snapshot_n'] for v in thresholds.values()})}) "
                    "-- yalniz tmz_resistance_score/dna_repair_score icin "
                    "(aggressiveness_score mimari geregi sabit mutlak esikli, "
                    "bu tabloya GIRMEZ, v45.txt satir 583-586)."
                )
            )
        ),
    }


_SCORE_THRESHOLDS_NAMES: tuple[str, ...] = ("tmz_resistance_score", "dna_repair_score")


def _fetch_latest_score_thresholds(
    score_names: tuple[str, ...] = _SCORE_THRESHOLDS_NAMES,
) -> dict[str, dict[str, Any]]:
    """`score_thresholds` tablosundan HER `score_name` icin EN GUNCEL
    (`frozen_at` en buyuk) satiri ceker.

    2026-08-18 EKLENDI -- Codex review-gate bulgusu: `interpretation_note`
    eskiden SABIT bir metinle "score_thresholds 0 satir" diyordu, ama
    db-agent tabloyu ayni gun DOLDURDU (bkz. tools/write_score_
    thresholds.py) -- API kendi DB'siyle CELISEN bir metin donduruyordu.
    Bu fonksiyon canli tabloyu OKUR, hicbir esik deger SABIT KODLANMAZ.

    Tablo BOSSA (veya istenen `score_name` hic yoksa) o isim sozlukte
    YER ALMAZ -- cagiran taraf ("esik yok") SESSIZCE bir varsayilan esik
    UYDURMAZ (Codex'in 5. maddesi)."""

    conn = _get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT DISTINCT ON (score_name)
                   score_name, score_version, p33_cutoff, p66_cutoff,
                   cohort_snapshot_n, frozen_at
            FROM score_thresholds
            WHERE score_name = ANY(%s)
            ORDER BY score_name, frozen_at DESC
            """,
            (list(score_names),),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()
    return {row["score_name"]: dict(row) for row in rows}


def fetch_omics_interpretation(patient_id: str) -> dict[str, Any] | None:
    """`patients.has_omics=TRUE` ise `molecular_scores` satirindan omics
    yorum blogunu doner, DEGILSE `None` doner (cagiran taraf bunu yanittan
    TAMAMEN CIKARIR -- plan.txt:503 "sessizce atla").

    `has_omics=TRUE` AMA `molecular_scores`'ta satir YOKSA (2026-08-18
    canli DB'de 0 ornegi -- 48/48 tutarli) bu "sessizce atla" senaryosu
    DEGILDIR: gorunur ama Cox sonucunu ETKILEMEYEN bir anomali blogu
    doner (`available: False`), boylece veri tutarsizligi SESSIZCE
    YUTULMAZ.
    """

    conn = _get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute(
            "SELECT has_omics FROM patients WHERE patient_id = %s", (patient_id,)
        )
        patient_row = cur.fetchone()
        if patient_row is None or not patient_row.get("has_omics"):
            cur.close()
            return None

        cur.execute(
            f"""
            SELECT {", ".join(_MOLECULAR_SCORE_COLUMNS)}
            FROM molecular_scores
            WHERE patient_id = %s
            """,
            (patient_id,),
        )
        score_row = cur.fetchone()
        cur.close()
    finally:
        conn.close()

    if score_row is None:
        return {
            "has_omics": True,
            "available": False,
            "note": (
                f"patients.has_omics=TRUE ama molecular_scores tablosunda "
                f"{patient_id!r} icin satir yok -- veri tutarsizligi, "
                "db-agent'e bildirin. Cox risk skoru bu durumdan "
                "ETKILENMEDI."
            ),
        }

    thresholds = _fetch_latest_score_thresholds()
    return _build_omics_block(score_row, thresholds)


# =====================================================================
# 3) SHAP arka plani -- egitim havuzundan orneklem, mtime-anahtarli cache
# =====================================================================

_background_cache_lock = threading.Lock()
_background_cache: dict[tuple, pd.DataFrame] = {}


def _fetch_background_long_frame(segmentation_tool: str) -> pd.DataFrame:
    """SAYFALI cekim (`tools/train_cox_week3.py::fetch_c32_radiomics_
    long_frame` ile AYNI desen -- BURADA KOPYALANDI, IMPORT EDILMEDI,
    cunku o dosya modeling-agent'in aktif calisma alani ve `segmentation_
    tool`'u parametre olarak DISARIYA ACMAMASI BILINCLI bir tasarim
    karari; burada AYRI/kucuk bir kopya tutmak, o dosyadaki gelecek
    degisikliklere bu endpoint'i KIRILGAN hale getirmemek icin tercih
    edildi). Buyuk JSONB kolonlarda tek-sorgu timeout riski 2026-08-14'te
    GERCEK kosuda gozlemlenmisti (bkz. o dosyanin yorum satiri) -- ayni
    onlem burada da uygulanıyor.
    """

    conn = _get_db_connection()
    page_size = 250
    offset = 0
    rows: list = []
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        while True:
            cur.execute(
                """
                SELECT p.patient_id, ds.source_name AS source, r.tumor_region,
                       r.shape_features, r.first_order_features, r.texture_features
                FROM radiomics r
                JOIN mr_scans ms ON ms.scan_id = r.scan_id
                JOIN patients p ON p.patient_id = ms.patient_id
                JOIN dataset_sources ds ON ds.source_id = p.source_id
                WHERE r.segmentation_tool = %s
                ORDER BY r.scan_id, r.tumor_region
                LIMIT %s OFFSET %s
                """,
                (segmentation_tool, page_size, offset),
            )
            page = cur.fetchall()
            if not page:
                break
            rows.extend(page)
            offset += page_size
        cur.close()
    finally:
        conn.close()
    return pd.DataFrame(rows)


def get_shap_background(
    final_features: list[str],
    extra_columns: list[str],
    cache_generation: float,
    *,
    age_rcs_knots: tuple[float, float, float] | None = None,
) -> pd.DataFrame:
    """`final_features`'in gerektirdigi bolge(ler) -- WT/TC/... HANGISI
    olursa olsun, kolon ONEKLERINDEN turetilir, TEK bir bolgeye SABIT
    KODLANMAZ (2026-08-18 model-agnostik karari, v2c_mgmt_spline_wttc
    WT+TC istiyor olabilir) -- icin egitim havuzundan (`BACKGROUND_
    SEGMENTATION_TOOL`, SABIT -- bkz. modul dokstring'i "ACIK RISK")
    deterministik bir radyomik orneklem (`n=DEFAULT_N_SHAP_BACKGROUND,
    seed=DEFAULT_SHAP_SEED`) doner. `extra_columns` BOS DEGILSE
    (2026-08-18 EKLENDI) ayni orneklem hastalari icin klinik kovaryatlar
    da `patients` tablosundan cekilip TOLERANT (fillna+eksik-gosterge,
    egitimdeki desenle AYNI) kodlanir ve radyomik sutunlarla
    BIRLESTIRILIR -- SHAP arka plani, hedef hastayla AYNI ozellik
    uzayinda (radyomik+klinik) olmali, aksi halde `shap.LinearExplainer`
    boyut uyusmazligiyla HATA verir. `cache_generation` (genelde artefakt
    mtime'i) degisirse cache gecersiz sayilir -- model/ozellik listesi
    degisince arka plan da yeniden hesaplanir.
    """

    regions = tuple(sorted({c.split("__", 1)[0] for c in final_features}))
    cache_key = (regions, tuple(extra_columns), cache_generation)
    with _background_cache_lock:
        cached = _background_cache.get(cache_key)
        if cached is not None:
            return cached

    long_frame = _fetch_background_long_frame(BACKGROUND_SEGMENTATION_TOOL)
    if long_frame.empty:
        raise C32RadiomicsNotFoundError(
            f"SHAP arka plani icin egitim havuzunda "
            f"({BACKGROUND_SEGMENTATION_TOOL}) 0 satir bulundu."
        )
    try:
        wide, report = pivot_radiomics_long_to_wide(long_frame, list(regions))
    except ValueError as exc:
        # 2026-09-12 -- ayni dayaniksizlik SHAP arka plani icin de gecerli
        # olabilir (bkz. fetch_c32_wt_row'daki ayni blok/gerekce).
        raise RegionCanonicalizationError(
            f"SHAP arka plani icin bolge/kaynak kanoniklestirmesi basarisiz "
            f"(egitim havuzu={BACKGROUND_SEGMENTATION_TOOL!r}): {exc}"
        ) from exc
    missing = set(final_features) - set(wide.columns)
    if missing:
        raise RequiredRegionMissingError(
            f"Egitim havuzunda final_features'in bazilari eksik: "
            f"{sorted(missing)} (rapor: {report})."
        )
    radiomics_background = wide[final_features].sample(
        n=min(DEFAULT_N_SHAP_BACKGROUND, len(wide)), random_state=DEFAULT_SHAP_SEED
    )

    if extra_columns:
        raw_frame = _fetch_patients_clinical_raw(list(radiomics_background.index))
        raw_frame = raw_frame.set_index("patient_id").reindex(radiomics_background.index)
        fully_missing = raw_frame.isna().all(axis=1)
        if fully_missing.any():
            raise ClinicalCovariateMissingError(
                "SHAP arka plan orneklemindeki bazi hastalar icin patients "
                f"satiri bulunamadi: {raw_frame.index[fully_missing].tolist()}"
            )
        clinical_background = _encode_clinical_frame(
            raw_frame, extra_columns, age_rcs_knots=age_rcs_knots
        )
        background = pd.concat([radiomics_background, clinical_background], axis=1)
    else:
        background = radiomics_background

    with _background_cache_lock:
        _background_cache[cache_key] = background
    return background


# =====================================================================
# 4) Risk skoru + SHAP -- kapali-form LinearExplainer (bkz. modul
#    dokstring'i, sayisal kanit tools/shap_explainer_prototype.py'de)
# =====================================================================


def compute_risk_score_and_shap(
    fitted_model,
    model_features: list[str],
    patient_row: pd.DataFrame,
    background: pd.DataFrame,
    feature_standardization: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """`model_features` = `final_features` (radyomik) + `extra_columns`
    (klinik, VARSA) -- `fitted_model.params_.index` ile AYNI SIRADA
    olmalidir (cagiran taraf bunu garanti eder, bkz. endpoint). Eski adi
    `final_features` idi; klinik kovaryat destegi (2026-08-18) eklenince
    isim genellestirildi, MATEMATIK DEGISMEDI (bkz. modul dokstring'i
    "SHAP OLCEGI").

    `feature_standardization` (2026-09-12 EKLENDI, Baris karari Secenek A)
    -- checkpoint fold-guvenli Z-score ile STANDARDIZE edilmis bir uzayda
    fit edildiyse (ornek: v3b_lowvar_v2amgmt -- bkz.
    entities/cox-model-varyant-katalogu.md §4 "Katsayi okuma notu"),
    `fitted_model.params_`/`fitted_model._norm_mean` da O olcekte
    ogrenilmistir. Bu parametre YOK/`None` ise (v1 checkpoint'i, HAM
    olcek) davranis BIREBIR eskisiyle AYNI kalir -- `_apply_feature_
    standardization()` no-op'tur (sifir-regresyon sarti). VARSA, HEM
    `patient_row` HEM `background` (SHAP referans dagilimi) AYNI
    donusumden gecirilir -- aksi halde SHAP additivitesi (toplam+taban
    == risk_score) modelin GERCEKTE kullandigi olcekle tutarsiz olur."""

    import shap

    _validate_feature_standardization(feature_standardization, model_features)
    patient_row = _apply_feature_standardization(patient_row, feature_standardization)
    background = _apply_feature_standardization(background, feature_standardization)

    coef = fitted_model.params_[model_features].to_numpy()
    norm_mean = fitted_model._norm_mean[model_features].to_numpy()
    intercept = -float(coef @ norm_mean)

    explainer = shap.LinearExplainer(
        (coef, intercept),
        background[model_features].to_numpy(),
        feature_perturbation="interventional",
    )
    explanation = explainer(patient_row[model_features].to_numpy())
    shap_values = np.asarray(explanation.values[0], dtype=float)
    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])

    risk_score = float(
        fitted_model.predict_log_partial_hazard(patient_row[model_features]).iloc[0]
    )
    hazard_ratio = float(
        fitted_model.predict_partial_hazard(patient_row[model_features]).iloc[0]
    )

    reconstructed = float(shap_values.sum() + base_value)
    additivity_abs_diff = abs(reconstructed - risk_score)

    return {
        "risk_score_log_partial_hazard": risk_score,
        "hazard_ratio_partial_hazard": hazard_ratio,
        "shap_base_value": base_value,
        "shap_values": {
            feature: float(value) for feature, value in zip(model_features, shap_values)
        },
        "shap_additivity_check_abs_diff": additivity_abs_diff,
    }


# =====================================================================
# 4a) XGBOOST IKINCI KATMAN (shadow) -- 2026-09-13, ADIM 2
#     Bkz. modul dokstring'i "XGBOOST IKINCI KATMAN -- shadow".
#     TEMEL ILKE: bu bolumdeki HICBIR fonksiyon Cox risk skorunu/SHAP'i
#     HESAPLAYAN kod yoluna GIRMEZ -- `compute_risk_score_and_shap()`
#     BU BLOKTAN BAGIMSIZ calisir ve ONCE calisir (omics blogunun
#     2026-08-18'de kurulan ayni ilkesi). Bu blogun basarisiz olmasi
#     Cox sonucunu DEGISTIRMEZ/SILMEZ.
# =====================================================================

_xgboost_checkpoint_cache_lock = threading.Lock()
_xgboost_checkpoint_cache: dict[str, dict[str, Any]] = {}


def _xgboost_checkpoint_path() -> Path:
    import os

    _ensure_project_env_loaded()
    override = os.environ.get(XGBOOST_CHECKPOINT_ENV_VAR)
    return Path(override) if override else DEFAULT_XGBOOST_CHECKPOINT_PATH


def xgboost_shadow_enabled() -> bool:
    """`GBMAID_ENABLE_XGBOOST_SHADOW` -- VARSAYILAN ACIK (Baris onayi,
    2026-09-13: "baglanir ama shadow etiketi KALIR"). `0`/`false`/`no`/
    `off` (buyuk-kucuk harf duyarsiz) ile KAPATILIR; kapatildiginda blok
    `available=False` + acik sebep doner, Cox yolu ETKILENMEZ."""

    import os

    _ensure_project_env_loaded()
    raw = os.environ.get(XGBOOST_ENABLE_ENV_VAR)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _require_mapping_keys(block: Any, keys: tuple[str, ...], *, label: str) -> None:
    if not isinstance(block, dict):
        raise XGBoostCheckpointContractError(
            f"XGBoost checkpoint'inde {label!r} bir sozluk DEGIL (alinan: "
            f"{type(block)!r})."
        )
    missing = [key for key in keys if key not in block]
    if missing:
        raise XGBoostCheckpointContractError(
            f"XGBoost checkpoint'inde {label!r} icin beklenen alan(lar) yok: "
            f"{missing}. Beklenen sozlesme: tools/export_xgboost_checkpoint.py"
            "::_build_checkpoint (arm_name/cox_score/xgboost)."
        )


def _validate_xgboost_checkpoint(loaded: Any) -> dict[str, Any]:
    """`tools/export_xgboost_checkpoint.py`'nin yazdigi checkpoint'i
    DOGRULAR ve DUZ (flat) bir sozluge indirger. HICBIR kontrol
    "sessizce duzelt/atla" yapmaz -- hepsi `XGBoostCheckpointContract
    Error` firlatir.

    Dogrulanan sozlesme (tamami GERCEK dosyadan OKUNARAK teyit edildi,
    2026-09-13 canli olcum):
      1. `cox_score` blogu: fitted_model + final_features(16) +
         clinical_extra_columns(8) + feature_standardization(55).
      2. `xgboost` blogu: fitted_model + xgb_feature_columns(102) +
         score_column('cox_oof_score') + raw_radiomic_feature_columns(93)
         + clinical_extra_columns(8) + age_standardization.
      3. YAPISAL KURULUS: `xgb_feature_columns` ==
         `raw_radiomic_feature_columns + clinical_extra_columns +
         [score_column]` -- SIRA DAHIL birebir (egitimdeki kurulus,
         pipeline/xgboost_model.py satir ~2925). Kolon tekrari YOK.
      4. `fitted_model.feature_names_in_` (xgboost>=1.6 DataFrame ile
         fit edildiginde tasinir; GERCEK checkpoint TASIYOR) ==
         `xgb_feature_columns`, SIRA DAHIL. `n_features_in_` == 102.
      5. Cox alt-modeli: `fitted_model.params_.index` kumesi ==
         final_features + clinical_extra_columns.
      6. **OLCEK GUARD'I (2026-09-12 hatasinin tekrarini onleyen asil
         savunma):** `cox_score.feature_standardization`, Cox alt-modelinin
         KULLANDIGI HER radyomik ozelligi VE yas kolonunu KAPSAMAK
         ZORUNDA -- kapsamiyorsa standardize bir uzayda fit edilmis
         modele HAM deger beslenir ve sonuc SESSIZCE yanlis olur.
         Tersi de kontrol edilir: ikili klinik bayraklar (gender/gtr/
         idh/mgmt) standardizasyon sozlugunde YER ALMAMALI (egitimde
         HAM 0/1 girdiler).
      7. Iki blogun YAS istatistigi (`age_standardization` vs
         `feature_standardization[age]`) SAYISAL OLARAK AYNI olmali
         (tol=1e-12) -- ayrisirsa "hangi yas olcegi dogru" sorusu
         sessizce cozulemez hale gelir.
      8. `clinical_extra_columns` iki blokta AYNI (sira dahil) --
         farkliysa iki ayri klinik kodlama gerekir, KAPSAM DISI.
      9. Klinik kovaryatlar `SUPPORTED_CLINICAL_EXTRA_COLUMNS` icinde VE
         yas-spline kolonlari (`clinical_age_rcs1/2`) DEGIL -- spline
         dugumleri bu checkpoint'te TASINMIYOR (Cox tarafiyla AYNI
         gerekce, bkz. modul dokstring'i "AGE RCS SPLINE ACIK RISK").
    """

    if not isinstance(loaded, dict):
        raise XGBoostCheckpointContractError(
            f"XGBoost checkpoint'i bir sozluk DEGIL (alinan: {type(loaded)!r})."
        )
    _require_mapping_keys(loaded, ("arm_name", "cox_score", "xgboost"), label="checkpoint")

    cox_block = loaded["cox_score"]
    _require_mapping_keys(
        cox_block,
        ("fitted_model", "final_features", "clinical_extra_columns", "feature_standardization"),
        label="cox_score",
    )
    xgb_block = loaded["xgboost"]
    _require_mapping_keys(
        xgb_block,
        (
            "fitted_model",
            "xgb_feature_columns",
            "score_column",
            "raw_radiomic_feature_columns",
            "clinical_extra_columns",
            "age_standardization",
        ),
        label="xgboost",
    )

    cox_final_features = list(cox_block["final_features"])
    cox_extra_columns = list(cox_block["clinical_extra_columns"])
    cox_feature_standardization = cox_block["feature_standardization"]
    xgb_feature_columns = list(xgb_block["xgb_feature_columns"])
    raw_radiomic_feature_columns = list(xgb_block["raw_radiomic_feature_columns"])
    xgb_extra_columns = list(xgb_block["clinical_extra_columns"])
    score_column = xgb_block["score_column"]
    age_standardization = xgb_block["age_standardization"]

    if not cox_final_features:
        raise XGBoostCheckpointContractError("cox_score.final_features BOS.")
    if not raw_radiomic_feature_columns:
        raise XGBoostCheckpointContractError("xgboost.raw_radiomic_feature_columns BOS.")
    if not isinstance(score_column, str) or not score_column:
        raise XGBoostCheckpointContractError(
            f"xgboost.score_column bir metin DEGIL/bos: {score_column!r}."
        )
    if not isinstance(cox_feature_standardization, dict) or not cox_feature_standardization:
        raise XGBoostCheckpointContractError(
            "cox_score.feature_standardization YOK/BOS -- bu checkpoint'in Cox "
            "alt-modeli STANDARDIZE bir uzayda fit edildi, olcek bilgisi "
            "olmadan HAM deger beslenirse sonuc SESSIZCE yanlis olur "
            "(2026-09-12 hatasinin tam olarak kendisi)."
        )

    # (3) yapisal kurulus + tekrar kontrolu
    expected_columns = raw_radiomic_feature_columns + xgb_extra_columns + [score_column]
    if xgb_feature_columns != expected_columns:
        raise XGBoostCheckpointContractError(
            "xgboost.xgb_feature_columns, egitimdeki yapisal kurulusla "
            "(raw_radiomic_feature_columns + clinical_extra_columns + "
            "[score_column], SIRA DAHIL) UYUSMUYOR -- "
            f"beklenen n={len(expected_columns)}, alinan n={len(xgb_feature_columns)}; "
            f"ilk fark: {next((i for i, (a, b) in enumerate(zip(xgb_feature_columns, expected_columns)) if a != b), 'uzunluk')}."
        )
    if len(set(xgb_feature_columns)) != len(xgb_feature_columns):
        duplicated = sorted({c for c in xgb_feature_columns if xgb_feature_columns.count(c) > 1})
        raise XGBoostCheckpointContractError(
            f"xgboost.xgb_feature_columns'ta TEKRARLAYAN kolon(lar): {duplicated}."
        )
    if score_column in raw_radiomic_feature_columns or score_column in xgb_extra_columns:
        raise XGBoostCheckpointContractError(
            f"score_column {score_column!r} ayni zamanda radyomik/klinik kolon "
            "listesinde -- sozlesme ihlali."
        )

    # (8) iki blogun klinik kolonlari AYNI olmali
    if cox_extra_columns != xgb_extra_columns:
        raise XGBoostCheckpointContractError(
            "cox_score.clinical_extra_columns ile xgboost.clinical_extra_columns "
            f"AYNI DEGIL ({cox_extra_columns} vs {xgb_extra_columns}) -- iki ayri "
            "klinik kodlama gerektirir, KAPSAM DISI (sessizce biri digerine "
            "uygulanmaz)."
        )

    # (9) desteklenen klinik kovaryatlar + spline yasagi
    unsupported = sorted(set(xgb_extra_columns) - SUPPORTED_CLINICAL_EXTRA_COLUMNS)
    if unsupported:
        raise XGBoostCheckpointContractError(
            f"XGBoost checkpoint'i desteklenmeyen klinik kovaryat(lar) istiyor: "
            f"{unsupported} -- bu endpoint yalniz "
            f"{sorted(SUPPORTED_CLINICAL_EXTRA_COLUMNS)} kodlamasini biliyor."
        )
    spline_requested = sorted({CLINICAL_AGE_RCS1_COLUMN, CLINICAL_AGE_RCS2_COLUMN} & set(xgb_extra_columns))
    if spline_requested:
        raise XGBoostCheckpointContractError(
            f"XGBoost checkpoint'i yas-spline kovaryati istiyor ({spline_requested}) "
            "ama spline dugumleri (`age_rcs_knots`) bu checkpoint sozlesmesinde "
            "TASINMIYOR -- tek hastadan YENIDEN TURETILEMEZ, KAPSAM DISI."
        )

    # Cox alt-modeli: params_ kumesi
    cox_fitted_model = cox_block["fitted_model"]
    try:
        cox_model_columns = list(cox_fitted_model.params_.index)
    except AttributeError as exc:
        raise XGBoostCheckpointContractError(
            f"cox_score.fitted_model'in `params_` alani yok ({exc}) -- lifelines "
            "CoxPHFitter beklenir."
        ) from exc
    if set(cox_model_columns) != set(cox_final_features) | set(cox_extra_columns):
        raise XGBoostCheckpointContractError(
            "cox_score.fitted_model.params_.index, final_features+clinical_extra_"
            f"columns ile UYUSMUYOR ({sorted(cox_model_columns)} vs "
            f"{sorted(set(cox_final_features) | set(cox_extra_columns))})."
        )

    # (6) OLCEK GUARD'I -- standardizasyon kapsami
    age_column = CLINICAL_AGE_COLUMN
    columns_that_must_be_standardized = set(cox_final_features)
    if age_column in cox_extra_columns:
        columns_that_must_be_standardized.add(age_column)
    uncovered = sorted(columns_that_must_be_standardized - set(cox_feature_standardization))
    if uncovered:
        raise XGBoostCheckpointContractError(
            "cox_score.feature_standardization, Cox alt-modelinin KULLANDIGI su "
            f"kolon(lar)i KAPSAMIYOR: {uncovered}. Bu kolonlari HAM beslemek "
            "standardize uzayda ogrenilmis katsayilarla SESSIZCE yanlis bir "
            "skor uretirdi (2026-09-12'de Cox tarafinda YASANAN hata) -- blok "
            "URETILMEZ."
        )
    binary_flag_columns = set(cox_extra_columns) - {age_column}
    wrongly_standardized = sorted(binary_flag_columns & set(cox_feature_standardization))
    if wrongly_standardized:
        raise XGBoostCheckpointContractError(
            f"cox_score.feature_standardization ikili klinik bayrak(lar) iceriyor: "
            f"{wrongly_standardized} -- egitimde bunlar HAM 0/1 girdi (bkz. "
            "tools/train_cox_week3.py). Sozlesme beklenmedik bicimde degismis, "
            "sessizce uygulanmaz."
        )
    # `_validate_feature_standardization()` (mean/std varligi, std!=0) AYNI
    # sozlesmeyi zaten dogruluyor -- SADECE modelin kullandigi alt kume icin
    # cagirilir (checkpoint 55 kolon tasir, model 17'sini kullanir; 55'in
    # tamamini `model_features` ile karsilastirmak YANLIS POZITIF verirdi).
    _validate_feature_standardization(
        {
            column: stats
            for column, stats in cox_feature_standardization.items()
            if column in set(cox_model_columns)
        },
        cox_model_columns,
    )

    # (7) iki blogun yas istatistigi
    _require_mapping_keys(age_standardization, ("column", "mean", "std"), label="xgboost.age_standardization")
    if age_standardization["column"] != age_column:
        raise XGBoostCheckpointContractError(
            f"xgboost.age_standardization['column'] beklenmeyen: "
            f"{age_standardization['column']!r} (beklenen {age_column!r})."
        )
    age_mean = float(age_standardization["mean"])
    age_std = float(age_standardization["std"])
    if age_std == 0.0:
        raise XGBoostCheckpointContractError(
            "xgboost.age_standardization['std'] == 0 -- sifira bolme."
        )
    if age_column in cox_feature_standardization:
        cox_age_stats = cox_feature_standardization[age_column]
        mean_diff = abs(float(cox_age_stats["mean"]) - age_mean)
        std_diff = abs(float(cox_age_stats["std"]) - age_std)
        if max(mean_diff, std_diff) > XGBOOST_AGE_STANDARDIZATION_TOLERANCE:
            raise XGBoostCheckpointContractError(
                "Iki blogun YAS istatistigi AYRISMIS -- cox_score."
                f"feature_standardization['{age_column}']={cox_age_stats} vs "
                f"xgboost.age_standardization={age_standardization} "
                f"(|dmean|={mean_diff:.3e}, |dstd|={std_diff:.3e} > "
                f"tol={XGBOOST_AGE_STANDARDIZATION_TOLERANCE:.0e}). Hangi olcegin "
                "dogru oldugu SESSIZCE secilemez."
            )
    if age_column in xgb_extra_columns and age_column not in cox_feature_standardization:
        # Yukaridaki (6) guard'i bunu zaten yakalar; burada AYRICA aciklikla
        # birakiyoruz -- sozlesme degisirse ikinci bir savunma.
        raise XGBoostCheckpointContractError(
            f"{age_column!r} klinik kovaryat olarak isteniyor ama cox_score."
            "feature_standardization'da yok."
        )

    # (4) modelin KENDI kolon sozlesmesi
    xgb_fitted_model = xgb_block["fitted_model"]
    model_feature_names = getattr(xgb_fitted_model, "feature_names_in_", None)
    if model_feature_names is not None:
        model_feature_names = [str(name) for name in list(model_feature_names)]
        if model_feature_names != xgb_feature_columns:
            first_diff = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(model_feature_names, xgb_feature_columns))
                    if a != b
                ),
                "uzunluk",
            )
            raise XGBoostCheckpointContractError(
                "xgboost.fitted_model.feature_names_in_ ile xgb_feature_columns "
                f"AYNI DEGIL (ilk fark indeksi: {first_diff}, "
                f"n_model={len(model_feature_names)} n_liste={len(xgb_feature_columns)}) "
                "-- kolon SIRASI karisirsa model SESSIZCE yanlis olasilik doner."
            )
    n_features_in = getattr(xgb_fitted_model, "n_features_in_", None)
    if n_features_in is not None and int(n_features_in) != len(xgb_feature_columns):
        raise XGBoostCheckpointContractError(
            f"xgboost.fitted_model.n_features_in_={int(n_features_in)} != "
            f"len(xgb_feature_columns)={len(xgb_feature_columns)}."
        )
    if not hasattr(xgb_fitted_model, "predict_proba"):
        raise XGBoostCheckpointContractError(
            f"xgboost.fitted_model'in `predict_proba` metodu yok "
            f"({type(xgb_fitted_model)!r})."
        )

    provenance = loaded.get("provenance") or {}
    verification = provenance.get("verification") or {} if isinstance(provenance, dict) else {}
    external_validation = verification.get("ucsf_external_auc_reproduced")

    return {
        "arm_name": loaded["arm_name"],
        "model_registry_version_hint": loaded.get("model_registry_version_hint"),
        "twelve_month_threshold_days": loaded.get("twelve_month_threshold_days"),
        "cox_fitted_model": cox_fitted_model,
        "cox_model_columns": cox_model_columns,
        "cox_final_features": cox_final_features,
        "cox_extra_columns": cox_extra_columns,
        "cox_feature_standardization": dict(cox_feature_standardization),
        "xgb_fitted_model": xgb_fitted_model,
        "xgb_feature_columns": xgb_feature_columns,
        "raw_radiomic_feature_columns": raw_radiomic_feature_columns,
        "xgb_extra_columns": xgb_extra_columns,
        "score_column": score_column,
        "age_column": age_column,
        "age_mean": age_mean,
        "age_std": age_std,
        "best_params": dict(xgb_block.get("best_params") or {}),
        "external_validation": dict(external_validation) if external_validation else None,
        "provenance": provenance if isinstance(provenance, dict) else {},
    }


def load_xgboost_checkpoint(path: Path | None = None) -> dict[str, Any]:
    """XGBoost (shadow) checkpoint'ini YALNIZ OKUYARAK acar, dogrular
    (`_validate_xgboost_checkpoint`) ve mtime-anahtarli in-process cache'e
    koyar -- Cox tarafiyla AYNI desen (`load_cox_arm_result`). `models/`
    dizinine YAZMAZ."""

    target = path or _xgboost_checkpoint_path()
    if not target.is_file():
        raise XGBoostCheckpointNotFoundError(
            f"XGBoost (shadow) checkpoint'i bulunamadi: {target}. "
            "tools/export_xgboost_checkpoint.py (2026-09-13, ADIM 1) henuz "
            f"calismamis olabilir, ya da {XGBOOST_CHECKPOINT_ENV_VAR} yanlis "
            "bir yolu isaret ediyor."
        )

    mtime = target.stat().st_mtime
    cache_key = str(target)
    with _xgboost_checkpoint_cache_lock:
        cached = _xgboost_checkpoint_cache.get(cache_key)
        if cached is not None and cached["mtime"] == mtime:
            return cached["checkpoint"]

        with target.open("rb") as fh:
            loaded = pickle.load(fh)
        checkpoint = _validate_xgboost_checkpoint(loaded)
        checkpoint["checkpoint_file_name"] = target.name
        _xgboost_checkpoint_cache[cache_key] = {"mtime": mtime, "checkpoint": checkpoint}
        return checkpoint


def compute_cox_oof_score_for_xgboost(
    checkpoint: dict[str, Any],
    radiomic_row: pd.DataFrame,
    clinical_row: pd.DataFrame,
) -> float:
    """XGBoost'un `score_column` girdisini (`cox_oof_score`) uretir.

    OLCEK: checkpoint'in `cox_score.feature_standardization` sozlugunden
    YALNIZ bu modelin kullandigi kolonlar (16 radyomik + `clinical_age`)
    Z-SKORLANIR; ikili klinik bayraklar HAM 0/1 kalir. Bu, `pipeline/
    xgboost_model.py::evaluate_xgboost_external_test()`'in UCSF'te
    yaptigi ADIMIN AYNISIDIR (orada `full_pool_external_frame` AYNI
    istatistiklerle standardize edilmis olarak gelir).

    ⚠️ `radiomic_row` HAM gelmelidir (`fetch_c32_radiomic_row` ne
    donduruyorsa) -- standardizasyon BURADA, YALNIZ Cox'un kopyasina
    uygulanir; XGBoost'a giden 93 kolon BU FONKSIYONDAN ETKILENMEZ
    (girdi frame'leri KOPYALANIR, mutate EDILMEZ)."""

    cox_model_columns = checkpoint["cox_model_columns"]
    combined = pd.concat(
        [radiomic_row.reset_index(drop=True), clinical_row.reset_index(drop=True)], axis=1
    )
    missing = [column for column in cox_model_columns if column not in combined.columns]
    if missing:
        raise XGBoostFeatureFrameError(
            f"Cox alt-modeli icin gereken kolon(lar) hasta satirinda yok: {missing}."
        )
    cox_frame = combined[cox_model_columns].copy()
    standardization_for_model = {
        column: stats
        for column, stats in checkpoint["cox_feature_standardization"].items()
        if column in set(cox_model_columns)
    }
    standardized = _apply_feature_standardization(cox_frame, standardization_for_model)
    return float(
        checkpoint["cox_fitted_model"]
        .predict_log_partial_hazard(standardized[cox_model_columns])
        .iloc[0]
    )


def build_xgboost_feature_frame(
    checkpoint: dict[str, Any],
    radiomic_row: pd.DataFrame,
    clinical_row: pd.DataFrame,
    cox_oof_score: float,
) -> pd.DataFrame:
    """XGBoost'a verilecek TEK satirli cerceveyi, `xgb_feature_columns`
    SIRASINDA kurar: 93 radyomik (HAM, DB'den geldigi gibi) -> 8 klinik
    (YALNIZ `clinical_age` z-skorlu, 7 bayrak HAM 0/1) -> `cox_oof_score`.

    Kolonlar TEK TEK ve SOZLESME SIRASINDA eklenir; yine de cagiran taraf
    `_assert_xgboost_feature_frame_contract()` ile SIRAYI dogrulamak
    ZORUNDADIR (savunma katmani, bkz. modul dokstring'i madde 2)."""

    raw_columns = checkpoint["raw_radiomic_feature_columns"]
    extra_columns = checkpoint["xgb_extra_columns"]
    radiomic = radiomic_row.reset_index(drop=True)
    clinical = clinical_row.reset_index(drop=True)

    missing_radiomic = [column for column in raw_columns if column not in radiomic.columns]
    if missing_radiomic:
        raise XGBoostFeatureFrameError(
            f"Hasta radyomik satirinda su ham kolon(lar) yok: {missing_radiomic} "
            f"(beklenen n={len(raw_columns)})."
        )
    missing_clinical = [column for column in extra_columns if column not in clinical.columns]
    if missing_clinical:
        raise XGBoostFeatureFrameError(
            f"Hasta klinik satirinda su kolon(lar) yok: {missing_clinical}."
        )

    # Kolonlar SOZLESME SIRASINDA bir sozluge yazilir, sonra TEK seferde
    # DataFrame'e cevrilir (`dict` ekleme sirasini korur) -- tek tek
    # `frame[col] = ...` atamasi 102 kolonda pandas'in "highly fragmented"
    # uyarisini uretiyordu.
    values: dict[str, float] = {}
    for column in raw_columns:
        values[column] = float(radiomic.iloc[0][column])  # HAM -- olcekleme YOK
    for column in extra_columns:
        values[column] = float(clinical.iloc[0][column])
    values[checkpoint["score_column"]] = float(cox_oof_score)
    frame = pd.DataFrame([values], index=[0])

    # YALNIZ yas z-skorlanir (`_apply_feature_standardization` listelenmeyen
    # kolonlara DOKUNMAZ, kolon SIRASINI korur -- mevcut/dogrulanmis yardimci
    # yeniden kullaniliyor, ikinci bir olcekleme kodu YAZILMADI).
    if checkpoint["age_column"] in extra_columns:
        frame = _apply_feature_standardization(
            frame,
            {
                checkpoint["age_column"]: {
                    "mean": checkpoint["age_mean"],
                    "std": checkpoint["age_std"],
                }
            },
        )
    return frame


def _values_equal_allowing_nan(left: float, right: float) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return float(left) == float(right)


def _assert_xgboost_raw_scale_preserved(
    frame: pd.DataFrame, radiomic_row: pd.DataFrame, raw_columns: list[str]
) -> None:
    """**OLCEK-KARISMASI GUARD'I** -- XGBoost'a gidecek cercevedeki 93
    radyomik degerin, DB'den okunan HAM degerle BIREBIR (==, NaN==NaN
    dahil) ayni oldugunu dogrular.

    NEDEN: 2026-09-12'de Cox tarafinda tam tersi yon yasandi (model
    standardize, API ham besledi). Bu checkpoint'te risk TERS yondedir --
    XGBoost HAM radyomikle egitildi, biri yanlislikla `cox_score`
    blogunun ~~54-kolonluk~~ **55-kolonluk** standardizasyon sozlugunu bu
    cerceveye uygularsa model SESSIZCE yanlis bir olasilik dondururdu. Bu
    guard o senaryoda `XGBoostFeatureFrameError` firlatir.

    ⚠️ 2026-09-13 SAYI DUZELTMESI (backend-agent-K, canli olcum --
    `models/xgboost_v2a_mgmt_reduce_collinearity_2026-09-13.pkl` acilarak):
    `cox_score.feature_standardization` sozlugu **55 anahtar** tasir =
    **54 radyomik + `clinical_age`**. "54-kolonluk" yaziyordu, uzeri
    cizildi -- 54 sayisi sozlugun DEGIL, sozluk icindeki RADYOMIK alt
    kumesinin buyuklugudur (bkz. modul dokstring'i satir ~202 ve
    `_validate_xgboost_checkpoint()` md.1, ikisi de 55 diyor -- ORADAKI
    sayi DOGRUYDU, buradaki YANLISTI).

    KAPSAM (bkz. modul dokstring'i md.2'deki kapsam duzeltmesi): bu guard
    `radiomic_row`'u DB'den YENIDEN OKUMAZ, cagiranin elindeki AYNI
    bellek-ici satirla karsilastirir -- build-adimi regresyon guard'idir."""

    source = radiomic_row.reset_index(drop=True)
    offending: list[str] = []
    for column in raw_columns:
        if not _values_equal_allowing_nan(frame.iloc[0][column], source.iloc[0][column]):
            offending.append(column)
    if offending:
        first = offending[0]
        raise XGBoostFeatureFrameError(
            "OLCEK KARISMASI: XGBoost cercevesindeki radyomik deger(ler) DB'den "
            f"okunan HAM degerle AYNI DEGIL -- {len(offending)} kolon, ornek "
            f"{first!r}: cerceve={frame.iloc[0][first]!r} ham={source.iloc[0][first]!r}. "
            "XGBoost 93 radyomigi HAM gordu (bkz. checkpoint 'xgboost' blogu); "
            "standardize deger beslemek SESSIZCE yanlis olasilik uretirdi -- "
            "olasilik URETILMEDI."
        )


def _assert_xgboost_feature_frame_contract(
    frame: pd.DataFrame, checkpoint: dict[str, Any]
) -> None:
    """**KOLON-SIRASI GUARD'I** -- cercevenin kolon kumesi VE SIRASI
    `xgb_feature_columns` ile (ve varsa `fitted_model.feature_names_in_`
    ile) BIREBIR ayni olmali. Sira karisirsa XGBoost sessizce yanlis bir
    olasilik dondurur (pozisyonel besleme)."""

    expected = checkpoint["xgb_feature_columns"]
    actual = list(frame.columns)
    if actual == expected:
        model_names = getattr(checkpoint["xgb_fitted_model"], "feature_names_in_", None)
        if model_names is not None and [str(n) for n in list(model_names)] != expected:
            raise XGBoostFeatureFrameError(
                "fitted_model.feature_names_in_ ile xgb_feature_columns AYNI "
                "DEGIL -- checkpoint dogrulamasindan SONRA degismis olabilir."
            )
        return

    missing = [column for column in expected if column not in frame.columns]
    unexpected = [column for column in actual if column not in set(expected)]
    if missing or unexpected:
        raise XGBoostFeatureFrameError(
            f"XGBoost cercevesinin kolonlari sozlesmeye UYMUYOR -- eksik: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''} (n={len(missing)}), "
            f"fazla: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''} "
            f"(n={len(unexpected)}); beklenen n={len(expected)}, alinan n={len(actual)}."
        )
    first_diff = next(i for i, (a, b) in enumerate(zip(actual, expected)) if a != b)
    raise XGBoostFeatureFrameError(
        "XGBoost cercevesinin kolon SIRASI sozlesmeyle AYNI DEGIL (kume ayni, "
        f"sira farkli) -- ilk fark indeks {first_diff}: cerceve={actual[first_diff]!r} "
        f"beklenen={expected[first_diff]!r}. Sira karismis bir cerceve SESSIZCE "
        "yanlis olasilik uretirdi -- olasilik URETILMEDI."
    )


def compute_xgboost_shadow_prediction(
    checkpoint: dict[str, Any],
    radiomic_row: pd.DataFrame,
    clinical_row: pd.DataFrame,
) -> dict[str, Any]:
    """Cox alt-skoru -> XGBoost cercevesi -> IKI GUARD -> `predict_proba`.

    Dondurdugu `probability_12_month_survival` = `predict_proba(...)[:, 1]`
    = 12 ayi GORME olasiligi (YUKSEK = IYI prognoz; Cox risk skorunun
    TERSI yonu -- bkz. `XGBOOST_DIRECTION_NOTE`)."""

    cox_oof_score = compute_cox_oof_score_for_xgboost(checkpoint, radiomic_row, clinical_row)
    frame = build_xgboost_feature_frame(checkpoint, radiomic_row, clinical_row, cox_oof_score)
    _assert_xgboost_feature_frame_contract(frame, checkpoint)
    _assert_xgboost_raw_scale_preserved(
        frame, radiomic_row, checkpoint["raw_radiomic_feature_columns"]
    )

    probabilities = checkpoint["xgb_fitted_model"].predict_proba(frame)
    probability = float(np.asarray(probabilities)[0, 1])
    if not np.isfinite(probability):
        raise XGBoostFeatureFrameError(
            f"predict_proba sonlu bir olasilik dondurmedi ({probability!r}) -- "
            "uydurma bir deger DONDURULMEZ."
        )
    return {
        "probability_12_month_survival": probability,
        "cox_oof_score": cox_oof_score,
    }


def _xgboost_block_unavailable(
    *, error_type: str | None, error_detail: str, systemic_hint: bool = False
) -> dict[str, Any]:
    """`api/analyze_patient.py::_risk_block_unavailable()` (B-1) deseniyle
    AYNI govde -- BLOK-FATAL, istek-fatal DEGIL. UYDURMA olasilik/skor
    ALANI YOKTUR (bilincli: `probability_12_month_survival` anahtari HIC
    eklenmez, `None` olarak da eklenmez -- tuketici tarafta "0.0 mi
    None mi" belirsizligi dogmasin)."""

    return {
        "available": False,
        "status": XGBOOST_MODEL_STATUS,
        "is_primary_decision_model": False,
        "error_type": error_type,
        "error_detail": error_detail,
        "systemic_hint": systemic_hint,
        "not_available_reason": (
            f"XGBoost (shadow) ikinci katman olasiligi URETILEMEDI: {error_detail} "
            "Bu BLOK-FATAL'dir -- Cox risk skoru/SHAP bundan ETKILENMEDI ve "
            "yanitta AYNEN duruyor. Uydurma bir olasilik DONDURULMEDI."
        ),
        "status_note": XGBOOST_STATUS_NOTE,
    }


def build_xgboost_shadow_block(
    patient_id: str, *, primary_cox_risk_score: float | None = None
) -> dict[str, Any]:
    """`/predict` yanitinin `xgboost` blogunu kurar. HICBIR istisna
    YUKARI SIZMAZ (BLOK-FATAL felsefesi, bkz. modul dokstring'i madde 3)
    -- `psycopg2.OperationalError` DAHIL (B-1'den BILINCLI sapma, gerekce
    dokstring'te).

    `primary_cox_risk_score`: endpoint'in ZATEN hesapladigi Cox risk
    skoru. YALNIZ SEFFAFLIK icin kullanilir (fark raporlanir); XGBoost'un
    girdisi HER ZAMAN checkpoint'in KENDI `cox_score` blogundan yeniden
    hesaplanir -- servis edilen Cox artefakti farkli bir varyant olsa
    bile XGBoost'un gordugu skor egitimdekiyle AYNI recete kalir."""

    if not xgboost_shadow_enabled():
        return _xgboost_block_unavailable(
            error_type="Disabled",
            error_detail=(
                f"{XGBOOST_ENABLE_ENV_VAR} ortam degiskeniyle ACIKCA devre disi "
                "birakildi (varsayilan: ACIK)."
            ),
        )

    try:
        checkpoint = load_xgboost_checkpoint()
        radiomic_row = fetch_c32_radiomic_row(
            patient_id, checkpoint["raw_radiomic_feature_columns"]
        )
        clinical_row = build_patient_clinical_covariate_row(
            patient_id, checkpoint["xgb_extra_columns"]
        )
        prediction = compute_xgboost_shadow_prediction(checkpoint, radiomic_row, clinical_row)
    except Exception as exc:  # BLOK-FATAL -- istek OLMEZ
        # `psycopg2` modul seviyesinde ZATEN import edilmis (`import
        # psycopg2.extras`) -- burada TEKRAR import edilmez (yerel bir isim
        # olusturup modul seviyesindekini golgelemesin).
        return _xgboost_block_unavailable(
            error_type=type(exc).__name__,
            error_detail=str(exc),
            systemic_hint=isinstance(exc, psycopg2.OperationalError),
        )

    cox_oof_score = prediction["cox_oof_score"]
    block: dict[str, Any] = {
        "available": True,
        "status": XGBOOST_MODEL_STATUS,
        "is_primary_decision_model": False,
        "status_note": XGBOOST_STATUS_NOTE,
        "model_arm": checkpoint["arm_name"],
        "model_registry_version_hint": checkpoint["model_registry_version_hint"],
        "checkpoint_file_name": checkpoint.get("checkpoint_file_name"),
        "probability_12_month_survival": prediction["probability_12_month_survival"],
        "direction_note": XGBOOST_DIRECTION_NOTE,
        "target_definition_note": XGBOOST_TARGET_NOTE,
        "twelve_month_threshold_days": checkpoint["twelve_month_threshold_days"],
        "scale_contract_note": XGBOOST_SCALE_CONTRACT_NOTE,
        "cox_score_input": {
            "score_column": checkpoint["score_column"],
            "value": cox_oof_score,
            "source_note": (
                "Checkpoint'in KENDI `cox_score` blogundan hesaplandi (v3b ile "
                "bit-birebir ayni katsayilar, modeling-agent-G'nin 2026-09-13 "
                "olcumu: 24/24 kolonda max|fark|=0,0) -- servis edilen Cox "
                "artefaktinin ciktisi DEGIL, ondan BAGIMSIZ hesaplanir."
            ),
        },
        "n_xgb_feature_columns": len(checkpoint["xgb_feature_columns"]),
        "n_raw_radiomic_features": len(checkpoint["raw_radiomic_feature_columns"]),
        "clinical_extra_columns": list(checkpoint["xgb_extra_columns"]),
        "best_params": checkpoint["best_params"],
        "guards_passed": [
            "xgb_feature_columns kolon kumesi+SIRASI (fitted_model.feature_names_in_ dahil)",
            "93 radyomik HAM olcek korunmasi (DB degeriyle birebir ==)",
            "cox_score standardizasyon kapsami (16 radyomik + clinical_age)",
            "iki blogun yas istatistigi ayni (tol=1e-12)",
        ],
    }

    external = checkpoint["external_validation"]
    if external:
        block["external_validation"] = {
            **external,
            "note": (
                "Checkpoint'in provenance'indan OKUNDU (bu istekte YENIDEN "
                "HESAPLANMADI): UCSF harici test, 12-ay hedefi belirsiz-sansurlu "
                "hastalar DUSURULDUKTEN SONRA kalan hasta sayisi. Tek-esik dili "
                "YASAK -- nokta tahmini %95 bootstrap CI ile birlikte okunur."
            ),
        }
    else:
        block["external_validation"] = None
        block["external_validation_note"] = (
            "Checkpoint'in provenance'inda `verification.ucsf_external_auc_"
            "reproduced` alani YOK -- harici AUC bu yanitta BEYAN EDILMEZ "
            "(uydurulmaz)."
        )

    if primary_cox_risk_score is not None:
        difference = abs(float(primary_cox_risk_score) - float(cox_oof_score))
        block["cox_score_input"]["primary_cox_risk_score_abs_diff"] = difference
        block["cox_score_input"]["primary_cox_risk_score_note"] = (
            "Servis edilen Cox artefakti v3b_lowvar_v2amgmt ISE bu fark ~0 "
            "olmalidir (iki model bit-birebir ayni). BASKA bir varyant servis "
            "ediliyorsa fark BEKLENIR ve HATA DEGILDIR -- XGBoost her durumda "
            "kendi egitim recetesindeki Cox skorunu kullanir. Bu alan yalniz "
            "SEFFAFLIK icindir, hicbir esikle karsilastirilip istek "
            "DUSURULMEZ."
        )
    return block


# =====================================================================
# 5) Endpoint
# =====================================================================


@router.post("/predict/{patient_id}")
def predict_patient(patient_id: str):
    try:
        arm = load_cox_arm_result()
    except CheckpointNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    extra_columns = arm["extra_columns"]
    age_rcs_knots = arm.get("age_rcs_knots")
    if extra_columns:
        unsupported = sorted(set(extra_columns) - SUPPORTED_CLINICAL_EXTRA_COLUMNS)
        if unsupported:
            raise HTTPException(
                status_code=501,
                detail=(
                    f"Model {arm['name']!r} desteklenmeyen klinik kovaryat(lar) "
                    f"iceriyor: {unsupported}. Bu endpoint su an yalniz "
                    f"{sorted(SUPPORTED_CLINICAL_EXTRA_COLUMNS)} kodlamasini "
                    "biliyor (kapsam disi -- bilinmeyen bir 3. kovaryat semasi "
                    "eklenince bu liste genisletilmeli)."
                ),
            )
        needs_age_spline = bool({CLINICAL_AGE_RCS1_COLUMN, CLINICAL_AGE_RCS2_COLUMN} & set(extra_columns))
        if needs_age_spline and age_rcs_knots is None:
            raise HTTPException(
                status_code=501,
                detail=(
                    f"Model {arm['name']!r} yas-spline kovaryati istiyor ama "
                    "artefakt 'age_rcs_knots' TASIMIYOR -- kapsam disi (bkz. "
                    "modul dokstring'i 'AGE RCS SPLINE AÇIK RISK')."
                ),
            )

    final_features = arm["final_features"]
    if not final_features:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model {arm['name']!r} icin final_features BOS -- artefakt "
                "bozuk/beklenmedik olabilir."
            ),
        )
    fitted_model = arm["fitted_model"]
    model_features = list(fitted_model.params_.index)
    if set(model_features) != set(final_features) | set(extra_columns):
        raise HTTPException(
            status_code=500,
            detail=(
                "Artefakt tutarsiz: fitted_model.params_.index "
                f"({sorted(model_features)}) final_features+extra_columns "
                f"({sorted(set(final_features) | set(extra_columns))}) ile "
                "UYUSMUYOR."
            ),
        )

    try:
        patient_row = fetch_c32_wt_row(patient_id, final_features)
    except PatientNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Hasta bulunamadi: {patient_id}"
        )
    except C32RadiomicsNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except MultiScanNotSupportedError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except LumierePreopVisitNotAvailableError as exc:
        # Y1 sart 4 -- `Pre-Op` viziti olup C32 radyomigi OLMAYAN 19 LUMIERE
        # hastasi: ACIK hata, SESSIZ bos/varsayilan DEGIL.
        raise HTTPException(status_code=422, detail=str(exc))
    except LumiereCanonicalVisitDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ExpertRatingFileNotFoundError as exc:
        # Altyapi/konfigurasyon sorunu (raw/ CSV'si erisilemez), hastaya
        # ait bir veri sorunu DEGIL -> 503. Baska bir kurala DUSULMEZ.
        raise HTTPException(status_code=503, detail=str(exc))
    except RequiredRegionMissingError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RegionCanonicalizationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # SEFFAFLIK (Y1): LUMIERE'de HANGI vizitin kullanildigi yanitta ACIKCA
    # yer alir. LUMIERE DISI hastalarda EK SORGU YAPILMAZ (`is_lumiere_
    # patient_id` yalniz `patient_id` onekine bakar, DB'ye gitmez).
    lumiere_canonical_visit_block: dict[str, Any] | None = None
    if is_lumiere_patient_id(patient_id):
        try:
            lumiere_canonical_visit_block = build_lumiere_canonical_visit_block(
                fetch_lumiere_canonical_visit_selection(patient_id)
            )
        except (
            C32RadiomicsNotFoundError,
            LumierePreopVisitNotAvailableError,
            LumiereCanonicalVisitDataError,
        ) as exc:
            # Buraya gelmek, `fetch_c32_wt_row` AYNI kurali basariyla
            # uygulamisken ikinci okumada BASARISIZ olmak demektir (veri
            # istek ortasinda degismis olabilir). Sessizce "vizit bilinmiyor"
            # DENMEZ -- risk skorunu HANGI vizitten urettigimizi
            # soyleyemiyorsak o skoru SERVIS ETMEYIZ.
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Hasta {patient_id!r}: radyomik satir SECILDI ama kanonik "
                    f"vizit beyani URETILEMEDI ({type(exc).__name__}: {exc}). "
                    "Hangi vizitin kullanildigi beyan edilemeyen bir LUMIERE "
                    "risk skoru DONDURULMEZ."
                ),
            )
        except ExpertRatingFileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    if extra_columns:
        try:
            clinical_row = build_patient_clinical_covariate_row(
                patient_id, extra_columns, age_rcs_knots=age_rcs_knots
            )
        except ClinicalCovariateMissingError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except ClinicalCovariatesNotSupportedError as exc:
            raise HTTPException(status_code=501, detail=str(exc))
        patient_row = pd.concat(
            [patient_row.reset_index(drop=True), clinical_row.reset_index(drop=True)],
            axis=1,
        )

    checkpoint_mtime = _checkpoint_path().stat().st_mtime
    try:
        background = get_shap_background(
            final_features, extra_columns, checkpoint_mtime, age_rcs_knots=age_rcs_knots
        )
    except (C32RadiomicsNotFoundError, RequiredRegionMissingError) as exc:
        raise HTTPException(
            status_code=500, detail=f"SHAP arka plani olusturulamadi: {exc}"
        )
    except RegionCanonicalizationError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP arka plani icin bolge kanoniklestirmesi basarisiz: {exc}",
        )
    except ClinicalCovariateMissingError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP arka plani icin klinik kovaryat olusturulamadi: {exc}",
        )
    except ClinicalCovariatesNotSupportedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))

    result = compute_risk_score_and_shap(
        fitted_model,
        model_features,
        patient_row,
        background,
        feature_standardization=arm.get("feature_standardization"),
    )

    if result["shap_additivity_check_abs_diff"] > SHAP_ADDITIVITY_TOLERANCE:
        raise HTTPException(
            status_code=500,
            detail=(
                "SHAP dogruluk kontrolu BASARISIZ (SHAP_toplami + taban != "
                f"risk_score, fark={result['shap_additivity_check_abs_diff']:.3e}"
                f" > esik={SHAP_ADDITIVITY_TOLERANCE:.0e}) -- sonuc GUVENILIR "
                "sayilmadi, DONDURULMEDI."
            ),
        )

    # Adim 7 -- Omics yorumu (OPSIYONEL). Cox risk_score/SHAP YUKARIDA
    # ZATEN hesaplandi -- bu cagri o hesaba HICBIR SEKILDE girmez, sadece
    # yanita EK bir yorum katmani ekler ya da (has_omics != TRUE ise)
    # hicbir sey eklemez (bkz. modul dokstring'i "OMICS YORUMU").
    response: dict[str, Any] = {
        "patient_id": patient_id,
        "model_arm": arm["name"],
        "final_features": final_features,
        "clinical_extra_columns": extra_columns,
        **result,
    }

    # Y1 -- LUMIERE'de kullanilan kanonik vizit ACIKCA beyan edilir. LUMIERE
    # DISI hastalarda bu anahtar HIC EKLENMEZ (`None` olarak da degil), ki
    # tuketici taraf "vizit secimi yapildi mi" sorusunu karistirmasin.
    if lumiere_canonical_visit_block is not None:
        response["lumiere_canonical_visit"] = lumiere_canonical_visit_block

    # ADIM 2 (2026-09-13) -- XGBoost ikinci katmani, SHADOW. Cox risk skoru
    # ve SHAP YUKARIDA ZATEN hesaplandi ve additivite kontrolunden GECTI;
    # bu cagri o hesaba HICBIR girdi VERMEZ (omics blogunun 2026-08-18'de
    # kurulan ayni ilkesi) ve HICBIR istisna SIZDIRMAZ -- basarisiz olursa
    # `available=False` + sebep doner, istek Cox sonucuyla 200 DEVAM EDER.
    response["xgboost"] = build_xgboost_shadow_block(
        patient_id, primary_cox_risk_score=result["risk_score_log_partial_hazard"]
    )

    omics = fetch_omics_interpretation(patient_id)
    if omics is not None:
        response["omics"] = omics
    return response
