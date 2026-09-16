"""Hafta 3 Cox PHM eğitim orkestrasyon script'i (GBM-AID, 2026-08-14, Nisa).

🔴 GÜNCELLEME (2026-08-18, Barış kararı -- kilitli, sorgulanmadı): VARSAYILAN
çalışma modu artık TEK MODEL, kol karşılaştırması YAPILMAZ. Model: `WT` ×
93 radyomik (ICC≥0,60) + yaş + cinsiyet + GTR(+missing) + IDH1(mutant+
missing), TAM 611 hasta / 585 ölüm olayı kohortu (`idh_cohort_filter`
UYGULANMAZ -- K15 kapandı, GTR ile birebir aynı gösterge-değişkeni deseni).
Harici test artık TCGA DEĞİL **UCSF-PDGM** (TCGA'da GTR/IDH1 %0 dolu,
model bu iki kovaryatı zorunlu içerdiği için TCGA'ya uygulanamaz). Yeni
giriş noktası: `run_primary_single_model()` (bkz. o fonksiyonun docstring'i)
-- `main()` VARSAYILAN olarak bunu çağırır. Aşağıdaki (2026-08-14 tarihli)
"ZİNCİR"/"SAYILAR" bölümleri hâlâ DOĞRU ama artık yalnız `--legacy-multi-
arm-tcga` bayrağıyla erişilen ESKİ 8-kollu `run_pipeline()` yolunu
tanımlıyor -- o yol/testleri (84 test) SİLİNMEDİ, duyarlılık amaçlı
korunuyor, ama üretim koşusunun VARSAYILANI DEĞİL.

KAPSAM (bugünkü görev talimatı, Barış onayı 2026-08-14): uçtan uca eğitim
zincirini KURAR ve sentetik/fixture veriyle test eder -- bkz.
`tests/test_train_cox_week3.py`. **GERÇEK VERİYLE ÇALIŞTIRILMAZ**: bu
dosyanın yazıldığı tarihte `radiomics` tablosunda C32 satırı **0** (canlı
DB, 2026-08-14 sabah ölçümü) -- eski A-yöntemi veriyle (`UPenn-PyRadiomics-
107`, `TCGA-ground-truth`) eğitim KESİNLİKLE YASAK (CLAUDE.md "PyRadiomics
C32" bölümü). Bu script C32 satır sayısı 0 olduğu sürece kendi kendine
`C32DataNotFoundError` ile DURUR -- sessizce eski veriye düşmek YOK.

ZİNCİR (görev talimatındaki sıra):
  1. `fetch_c32_radiomics_long_frame()` -- `radiomics` tablosundan YALNIZ
     `SEGMENTATION_TOOL_UPENN_C32`/`SEGMENTATION_TOOL_TCGA_C32` (sabit,
     PARAMETRE DEĞİL -- operatör bunu CLI'dan değiştiremez, kazara eski
     veriye dönme riski kod seviyesinde kapatılmıştır).
  2. `pipeline.cox_model.pivot_radiomics_long_to_wide()` -- birincil
     `regions=["WT"]` (`PRIMARY_REGIONS`). Düşen hasta sayısı
     `RegionPivotReport` ile raporlanır.
  3. Özellik filtresi: `STABLE_FEATURES_ICC60` (93, birincil) + tam 107
     (duyarlılık kolu).
  4. `pipeline.cox_model.build_training_frame()` -- TEK resmi giriş
     noktası (`check_combat_identity=True` varsayılanıyla).
     `_assemble_training_frame()` bu script'ten DOĞRUDAN ÇAĞRILMAZ.
  5. `pipeline.cox_model.run_nested_cv()` -- fold başına C-index + seçilen
     özellik listesi. **Fallback sayısı** ve **seçilme frekansı**
     `run_nested_cv()`'nin döndürdüğü `NestedCVFoldResult`'ta YOK (Codex
     2026-08-14 MEDIUM bulgusu) -- bu script `audit_nested_cv_fold_
     selection()` ile bunu, `pipeline/cox_model.py`'yi DEĞİŞTİRMEDEN,
     `run_nested_cv()`'nin KULLANDIĞI aynı private yardımcı fonksiyonları
     (`_bootstrap_stability_selection`) aynı tohum/parametrelerle
     yeniden çalıştırıp ÖZ-DOĞRULAMALI (self-consistency check) şekilde
     üretir -- bkz. o fonksiyonun docstring'i.
  6. Stabilite eşiği: birincil 0,6 (`DEFAULT_STABILITY_FREQUENCY_
     THRESHOLD`), `--sensitivity-threshold 0.7` ile ikinci koşu (bkz.
     decisions/2026-08-14-stabilite-secimi-esigi.md).
  7. `evaluate_external_test()` ile TCGA (`SEGMENTATION_TOOL_TCGA_C32`,
     WT) -- bootstrap %95 CI ZORUNLU. TCGA'ya `survival_days` ile YENİ
     özellik seçimi bu script'te YAPILMAZ (yalnız prespesifiye
     `STABLE_FEATURES_ICC60`/tam-107 kullanılır).
  8. Çıktılar `artifacts/week3/cox_model/` altına yazılır (bkz. alttaki
     "ÇIKTI DOSYALARI" bölümü).

YÖNTEM ADI (raporlama dili, Codex 2026-08-14 HIGH bulgusu -- decisions/
2026-08-14-stabilite-secimi-esigi.md): bu script'in ürettiği stabilite
seçimi klasik Meinshausen-Bühlmann "stability selection" DEĞİLDİR (yarım-
örneklem subsampling yerine n-out-of-n bootstrap, PFER kalibrasyonu yok).
Doğru ad: **"bootstrap selection-frequency filtering"**. Raporlarda/
jüride bu ad kullanılmalı, "stability selection" resmi hata garantilerini
ima eder -- bizde yok.

SAYILAR (CLAUDE.md 2026-08-14 düzeltmesi -- 630/603 DEĞİL): fiili UPenn Cox
eğitim havuzu **611 hasta / 585 ölüm olayı** (585/93 = EPV 6,29, WT-only
ICC>=0,60 birincil model). Bu script varsayılan olarak ölçülen hasta/olay
sayısını bu rakamlarla KARŞILAŞTIRIR (`--expected-patient-count 611
--expected-event-count 585`) ve uyuşmazlıkta (override verilmedikçe) SERT
DURUR -- CLAUDE.md/AKTIF-GOREVLER.md'nin A1.1 `image_count` gate
desenindeki AYNI disiplin (sessiz sapma yerine loud gate).

COMBAT NOTU (metadata'ya da yazılır): eğitim havuzu YALNIZ UPenn (LUMIERE
`vital_status` 0/91, event sağlayamıyor) ve ComBat'ın referans batch'i de
UPenn -- yani bu birincil OS-Cox zincirinde ComBat cebirsel olarak
ÖZDEŞLİK dönüşümüdür (bkz. decisions/2026-08-13-combat-ozdeslik-bulgusu-
ve-kapsami.md). `build_training_frame(check_combat_identity=True)`
(varsayılan) bu varsayımı HER koşuda kod seviyesinde doğrular -- bozulursa
`ComBatIdentityAssumptionError` ile SERT durur.

KAPSAM SINIRI (kendiliğinden genişletilmedi -- CLAUDE.md davranış kuralı
#5): bu script yalnız Cox aşamasını kapsar. XGBoost'a giden out-of-fold
Cox skorlarının üretimi, WT+TC/TC-only bölge duyarlılık kolları
(decisions/.../Bölüm 4.5) ve NC/ED/ET keşifsel katmanı (Bölüm 4.6) bu
script'in kapsamı DIŞINDA -- ayrı, sonraki bir görev.
⚠️ [2026-09-15 DÜZELTME, K23/BEKLEYEN-KARARLAR.md K5, modeling-agent-Z5]:
yukarıdaki çerçeve bu SCRIPT'in kendisi için hâlâ doğru, ama proje
genelinde üç alt madde ARTIK BAŞKA YOLLARDA TAMAMLANDI -- bu metin
"hiç yapılmadı" izlenimi vermemeli: (1) Cox->XGBoost out-of-fold aktarımı
`pipeline/xgboost_model.py`+`tools/train_xgboost_week4.py`'de YAPILDI ve
denetlendi (`verify_aligned_fold_leakage_free`, 5/5 fold sızıntısız);
(2) **WT+TC** duyarlılık kolu bu script İÇİNDE `--variants` mekanizmasıyla
(`v2d_mgmt_wttc_nospline`/`v3c_lowvar_wttc_mgmt_nospline`) iki kez KOŞULDU
(eğitim havuzu 609/583, 611/585 DEĞİL); **TC-only** ise HİÇ KOŞULMADI,
kapsam dışı kalır. Hâlâ gerçekten kapsam dışı olan tek madde: NC/ED/ET
keşifsel katmanı (K18 kararıyla BİLİNÇLİ olarak dışarıda bırakıldı,
"unutuldu" değil). Detay: `decisions/2026-09-13-k4-k5-k6-k12c-stabilite-
sabitleri-kapanisi.md` §2.

ÇIKTI DOSYALARI (`artifacts/week3/cox_model/`, `--output-dir` ile
değiştirilebilir):
  - `week3_fold_results.csv`       -- her arm x her dış fold (fallback
                                       bayrağı + 2026-09-14'ten itibaren
                                       `stability_frequency_threshold`
                                       sütunu dahil; dosya TEK BAŞINA
                                       alındığında hangi eşikle
                                       üretildiği görünsün diye. ESKİ
                                       dosyalarda bu sütun YOKTUR --
                                       okuyucular OPSİYONEL saymalı.)
  - `week3_selection_frequency.csv` -- her arm x her özellik, fold'lar
                                       arası ortalama/min/max bootstrap
                                       seçilme frekansı
  - `week3_external_test.csv`      -- her arm için TCGA C-index + %95 CI
  - `week3_run_metadata.json`      -- provenance (script sha256, sabit
                                       segmentation_tool adları, hasta/
                                       olay sayıları, ComBat notu, tarih)
  - `week3_exploratory_cluster_stability.csv` (yalnız
    `--exploratory-cluster-stability` ile) -- korelasyon kümesi başına
    "kümeden en az bir özellik seçildi" frekansı (keşifsel, birincil
    sonucun YERİNİ ALMAZ)

DB ERİŞİMİ: yalnız `readonly=True` bağlantı, yalnız SELECT. Bu script
DB'ye HİÇBİR YAZMA yapmaz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from psycopg2.extras import RealDictCursor

from db_connection import get_connection
from pipeline.cox_model import (
    DEFAULT_L1_RATIO_GRID,
    DEFAULT_PENALIZER_GRID,
    DEFAULT_STABILITY_FREQUENCY_THRESHOLD,
    EXCLUDED_FIRST_ORDER_ABSOLUTE_SCALE_FEATURES,
    PRIMARY_REGIONS,
    STABLE_FEATURES_ICC60,
    RegionPivotReport,
    TrainingFrameReport,
    build_training_frame,
    evaluate_external_test,
    pivot_radiomics_long_to_wide,
    run_nested_cv,
    select_features_lasso,
    standardize_columns_fold_safe,
)
# NOT (bilinçli, belgeli istisna -- `tools/smoke_test_cox_model.py`'nin
# `_assemble_training_frame` erişimiyle AYNI desen): bu üç fonksiyon
# `run_nested_cv()`'nin KENDİSİNİN kullandığı private yardımcılardır --
# audit + full-pool final model adımları için bu script'ten TEKRAR
# çağrılıyor (2026-08-14'ten beri).
#
# GÜNCELLEME (2026-08-15, klinik kovaryat desteği): bugünkü görev
# talimatı (önceki 2026-08-14 görevinden FARKLI olarak) `pipeline/
# cox_model.py`'ye dokunmayı YASAKLAMIYOR -- klinik kovaryatların (a)
# elastic-net/stabilite seçimine hiç GİRMEMESİ (`extra_columns` --
# zaten var olan mekanizma) VE (b) radyomik bloktan FARKLI/zayıf
# penalize edilmesi ("Codex önerisi") ve (c) fold-içi standardizasyon
# gereksinimleri, seçim/penalizasyon/CV döngülerinin YAŞADIĞI TEK yer
# olan `cox_model.py`'nin bu fonksiyonlarına dokunmadan mümkün DEĞİLDİ.
# Yapılan değişiklikler TAMAMEN GERİYE UYUMLU (yeni parametreler hepsi
# `None`/eski-varsayılan ile ESKİ davranışı BİREBİR korur) --
# `pytest tests/test_cox_model.py` (43 test) DEĞİŞİKLİK SONRASI da
# aynen geçiyor (regresyon yok, bkz. bu görevin final raporu). Bu HALA
# review-gate'ten GEÇMEDİ (fiziksel inceleme/Codex çapraz inceleme/
# resmi gözden geçirme YAPILMADI) -- "production-ready" değil.
from pipeline.cox_model import (  # noqa: E402  (bilinçli ayrı import bloğu)
    _bootstrap_stability_selection,
    _build_penalizer_argument,
    _score_hyperparameters_via_inner_cv,
)
# 2026-08-18 EKLENDİ (v3 hazırlığı, koordinatör görevi -- "düşük varyans
# filtresi"): `pipeline/reduce_collinearity.py` YENİ, BAĞIMSIZ bir modül
# -- `pipeline/cox_model.py`'ye TEK SATIR dokunulmadı, bu import ONA
# değil YENİ dosyaya. V2_VARIANT_CONFIGS/run_single_variant/mevcut HİÇBİR
# fonksiyon bu importu KULLANMAZ -- yalnız aşağıdaki "8.6) v3" bölümü.
from pipeline.reduce_collinearity import (  # noqa: E402
    DEFAULT_CORRELATION_CLUSTER_THRESHOLD as V3_DEFAULT_CORRELATION_CLUSTER_THRESHOLD,
    DEFAULT_CV_THRESHOLD as V3_DEFAULT_CV_THRESHOLD,
    V3CandidatePoolReport,
    build_v3_candidate_pool,
)

METHOD_NAME = "bootstrap selection-frequency filtering"
COMBAT_IDENTITY_NOTE = (
    "ComBat, bu birincil OS-Cox zincirinde egitim havuzu (yalniz UPenn) == "
    "referans batch (UPenn) oldugu icin cebirsel olarak OZDESLIK "
    "donusumudur (max fark 0.0, decisions/2026-08-13-combat-ozdeslik-"
    "bulgusu-ve-kapsami.md). build_training_frame(check_combat_identity="
    "True) bu varsayimi HER kosuda dogrular."
)

SEGMENTATION_TOOL_UPENN_C32 = "UPenn-PyRadiomics-107-C32"
SEGMENTATION_TOOL_TCGA_C32 = "TCGA-ground-truth-C32"
# 2026-08-18 EKLENDİ (Barış kararı -- UCSF-PDGM harici test desteği).
# ⚠️ UCSF verisi bu satırın yazıldığı tarihte DB'ye HENÜZ YAZILMADI
# (imaging-agent dry-run'da) -- bu sabit imaging-agent'in DONDURDUĞU
# `segmentation_tool` adı, canlı DB'de HENÜZ DOĞRULANMADI.
SEGMENTATION_TOOL_UCSF_C32 = "UCSF-PDGM-PyRadiomics-107-C32"
UPENN_SOURCE_NAME = "UPenn-GBM"
TCGA_SOURCE_NAME = "TCGA-GBM"
# DOĞRULANDI (2026-09-11 güncellemesi -- eski "UNVERIFIED (2026-08-18)"
# notu artık BAYAT): bu sabit ile `SEGMENTATION_TOOL_UCSF_C32` en az 8
# ayrı üretim koşusunda (v1_referans, v2a_mgmt, v2b_mgmt_spline,
# v2c_mgmt_spline_wttc, v3a_lowvar_v1referans, v3b_lowvar_v2amgmt,
# v3c_lowvar_wttc_mgmt_nospline, k14_clinical_base_ucsf) sorunsuz UCSF
# 295 hasta / 169 olay çekti -- her birinin `week3_<variant>_external_
# test.csv` dosyasında `n_patients=295`/`n_events=169` birebir doğrulandı
# (`artifacts/week3/cox_model/` + `final_v3c/`). `dataset_sources.
# source_name`'in canlı DB'deki gerçek literal değeri bu incelemede
# AYRICA sorgulanmadı -- doğrulanan şey bu sabitin PRATİKTE (fetch
# sorgusu + pivot) beklenen 295/169 kohortunu üretmeye devam ettiğidir,
# DB şemasındaki ham metin DEĞİL.
UCSF_SOURCE_NAME = "UCSF-PDGM"

# C32 VERİ KAPISI -- beyaz liste (CLAUDE.md "ADIM 0", 2026-08-18 UCSF
# eklendi). `fetch_c32_radiomics_long_frame()`'in TEK meşru girdi kümesi.
# Eski nesil adları (CaPTk-*, DeepBraTumIA, HD-GLIO-AUTO, *-PyRadiomics-107
# [C32 soneki OLMAYAN], TCGA-ground-truth [C32 soneki OLMAYAN]) burada YOK
# -- kasıtlı, bu fonksiyona segmentation_tool olarak asla verilmemeli.
ALLOWED_C32_SEGMENTATION_TOOLS: frozenset[str] = frozenset(
    {SEGMENTATION_TOOL_UPENN_C32, SEGMENTATION_TOOL_TCGA_C32, SEGMENTATION_TOOL_UCSF_C32}
)


class C32WhitelistViolationError(RuntimeError):
    """`segmentation_tool` C32 beyaz listesinde (`ALLOWED_C32_SEGMENTATION_
    TOOLS`) DEĞİL -- CLAUDE.md "ADIM 0 -- C32 VERİ KAPISI": eski nesil
    veriye (binWidth=25, N4/Z-score atlanmış) veya kaynak-sağlayıcı
    precompute'a (CaPTk/DeepBraTumIA/HD-GLIO-AUTO -- 'şekil olarak AYIRT
    EDİLEMEZ') SESSİZCE düşülmesin diye burada SERT durulur."""


def assert_c32_segmentation_tool_allowed(segmentation_tool: str) -> None:
    if segmentation_tool not in ALLOWED_C32_SEGMENTATION_TOOLS:
        raise C32WhitelistViolationError(
            f"segmentation_tool={segmentation_tool!r} C32 beyaz listesinde "
            f"DEĞİL. İZİN VERİLEN (yalnız bu üçü): {sorted(ALLOWED_C32_SEGMENTATION_TOOLS)}. "
            "Eski nesil adları (ör. 'UPenn-PyRadiomics-107' [-C32 soneki "
            "YOK], 'CaPTk-automatic', 'DeepBraTumIA', 'HD-GLIO-AUTO') bu "
            "fonksiyona ASLA verilmemeli -- CLAUDE.md 'ADIM 0' kapısı."
        )


# =====================================================================
# TIMEPOINT KAPISI -- 611/585 KILIDININ IKINCI YAPISAL KATMANI
# (2026-09-13/14, Baris onayi: "39) ikinci katman eklensin bence de iyi
# olur")
#
# BIRINCI KATMAN nerede: `tools/write_upenn_pyradiomics_to_db.py`
# GUARD 6 (2026-09-13, db-agent-G) -- CSV'deki `timepoint_used` degeri
# "pre-treatment" olmayan satiri INSERT etmeden reddeder (exit 3).
# O guard TEK yapisal savunmaydi: `radiomics` tablosuna baska bir yolla
# (elle INSERT, yeni bir script, baska bir yazici) post-op satir girerse
# bu script'in havuz sorgusu YALNIZ `segmentation_tool`'a baktigi icin
# onlari SESSIZCE alirdi -> kilitli 611/585 bozulurdu.
#
# 🔴 KOORDINATORUN TARIFI BURADA DUZELTILDI (olculdu, canli DB
# 2026-09-13 gecesi, readonly SELECT): gorev tarifi "sorguya
# `mr_scans.timepoint_label = 'pre-treatment'` kosulu ekle" diyordu.
# Bu TEK LITERAL, ayni sorgunun TCGA ve UCSF icin de kullanildigi
# gerceginden dolayi HARICI TEST SETINI SIFIRLARDI:
#   UPenn-PyRadiomics-107-C32     -> timepoint_label = 'pre-treatment'
#                                    (3055 satir / 611 hasta, %100)
#   TCGA-ground-truth-C32         -> timepoint_label = 'baseline'
#                                    (78 satir / 39 hasta, %100)
#   UCSF-PDGM-PyRadiomics-107-C32 -> timepoint_label = 'baseline'
#                                    (1475 satir / 295 hasta, %100)
# Bu yuzden kosul TEK LITERAL DEGIL, ARAC BASINA BEYAZ LISTE olarak
# kuruldu. UCSF'e 'pre-treatment' sartini uygulamak 295/169 harici test
# kohortunu 0'a dusururdu (CLAUDE.md kilitli degeri).
#
# ÖLÇÜLEN TEHDIT BOYUTU (ayni gece, canli): UPenn'in `post-op` etiketli
# 240 scan'i ve bu scan'lerin ait oldugu 60 HASTA var (db-agent-G'nin
# raporundaki 19, "YALNIZ post-op'u olan" alt kumesidir). Bugun bu 240
# scan'in HICBIRININ radiomics satiri YOK. Maskeler NAS'a eklenirse:
#   * yalniz post-op'u olan 19 hasta havuza YENI hasta olarak girer
#     (611 -> 630 riski),
#   * hem pre hem post-op'u olan 41 hasta ICIN AYNI hastanin IKINCI
#     scan'i gelir -> pivot'ta hasta-bazli COKLAMA riski.
# Ikisi de bu kapiyla SQL seviyesinde kesilir.
# =====================================================================

ALLOWED_TIMEPOINT_LABELS_BY_TOOL: dict[str, frozenset[str]] = {
    SEGMENTATION_TOOL_UPENN_C32: frozenset({"pre-treatment"}),
    SEGMENTATION_TOOL_TCGA_C32: frozenset({"baseline"}),
    SEGMENTATION_TOOL_UCSF_C32: frozenset({"baseline"}),
}

# BEKLENEN (analiz edilmis) DISLAMALAR -- bu etiketler disarida kalir
# ama kosuyu DURDURMAZ, yalniz "TIMEPOINT DISLAMA OZETI" ile GURULTULU
# raporlanir. UPenn'in `post-op`'u tam olarak 611/585 kilidinin dislamak
# istedigi seydir (decisions/2026-08-13-bolge-stratejisi-ve-modelleme-
# protokolu.md "SAYI DUZELTMESI (2026-08-14)"): 19 hastanin yalniz `_21`
# post-op T1ce'si var, hazir maskesi YOK.
# Bu kumede OLMAYAN bir etiket (veya NULL) gorulurse -> analiz edilmemis
# veri kaymasi demektir -> `UnexpectedTimepointLabelError` ile SERT
# durulur (fail-closed; sessiz dislama da sessiz dahil etme kadar
# tehlikelidir).
KNOWN_EXCLUDED_TIMEPOINT_LABELS_BY_TOOL: dict[str, frozenset[str]] = {
    SEGMENTATION_TOOL_UPENN_C32: frozenset({"post-op"}),
    SEGMENTATION_TOOL_TCGA_C32: frozenset(),
    SEGMENTATION_TOOL_UCSF_C32: frozenset(),
}

# Import anindaki senkron kontrolu (F3 deseni, db-agent-G 2026-09-13):
# beyaz liste ile timepoint haritalari BIRLIKTE buyumek ZORUNDA. Yeni
# bir C32 aracı `ALLOWED_C32_SEGMENTATION_TOOLS`'a eklenip timepoint
# beyani UNUTULURSA modul IMPORT EDILEMEZ -- sessizce filtresiz
# calismaz.
if set(ALLOWED_TIMEPOINT_LABELS_BY_TOOL) != set(ALLOWED_C32_SEGMENTATION_TOOLS):
    raise RuntimeError(  # pragma: no cover -- import-time tutarlilik kapisi
        "ALLOWED_TIMEPOINT_LABELS_BY_TOOL anahtarlari "
        "ALLOWED_C32_SEGMENTATION_TOOLS ile AYNI olmali. Fark: "
        f"{sorted(set(ALLOWED_TIMEPOINT_LABELS_BY_TOOL) ^ set(ALLOWED_C32_SEGMENTATION_TOOLS))}"
    )
if set(KNOWN_EXCLUDED_TIMEPOINT_LABELS_BY_TOOL) != set(ALLOWED_C32_SEGMENTATION_TOOLS):
    raise RuntimeError(  # pragma: no cover -- import-time tutarlilik kapisi
        "KNOWN_EXCLUDED_TIMEPOINT_LABELS_BY_TOOL anahtarlari "
        "ALLOWED_C32_SEGMENTATION_TOOLS ile AYNI olmali. Fark: "
        f"{sorted(set(KNOWN_EXCLUDED_TIMEPOINT_LABELS_BY_TOOL) ^ set(ALLOWED_C32_SEGMENTATION_TOOLS))}"
    )
for _tool, _allowed in ALLOWED_TIMEPOINT_LABELS_BY_TOOL.items():
    if not _allowed:
        raise RuntimeError(  # pragma: no cover -- bos beyaz liste = filtresiz cekim
            f"ALLOWED_TIMEPOINT_LABELS_BY_TOOL[{_tool!r}] BOS -- bos beyaz "
            "liste tum satirlari dislardi/filtresiz birakirdi, kasitli olamaz."
        )
    if _allowed & KNOWN_EXCLUDED_TIMEPOINT_LABELS_BY_TOOL[_tool]:
        raise RuntimeError(  # pragma: no cover -- celisik beyan
            f"{_tool!r} icin ayni timepoint etiketi hem IZINLI hem BEKLENEN-"
            f"DISLAMA listesinde: {sorted(_allowed & KNOWN_EXCLUDED_TIMEPOINT_LABELS_BY_TOOL[_tool])}"
        )
del _tool, _allowed

EXPECTED_UPENN_TRAINING_PATIENTS = 611
EXPECTED_UPENN_TRAINING_EVENTS = 585
# 2026-08-18 GÜNCELLENDİ -- ARTIK ÇÖZÜLDÜ (önceki sürümde 295/169 vs
# 367/223 "çelişki" olarak işaretlenmişti): decisions/2026-08-18-tek-
# model-ucsf-harici-test-k15-kapanisi.md "UCSF KOHORT SAYISI" bölümü
# bunun bir çelişki DEĞİL, tanım evrimi olduğunu netleştirdi --
# 367/223 (2026-08-15, IDH-wildtype filtreli + "tam indirilecek"
# varsayımı) ve 275/165 (2026-08-18 10:15, IDH-wildtype filtreli) ikisi
# de GEÇERSİZ; **295/169** (grade 4 + IDH filtresi YOK + diskte tam
# olan T1c+seg) GEÇERLİ ve kilitli sayı. IDH-wildtype filtresi bilinçli
# KALDIRILDI -- model IDH1'i kovaryat olarak içerdiği için kohort
# yalnız wildtype'tan oluşursa IDH varyansı sıfır olur, kovaryat harici
# testte hiçbir şey ölçemez (20 IDH-mutant hasta [%6,8] kohorta girer,
# WHO 2021'e göre bunlar teknik olarak "IDH-mutant astrositom, grade 4"
# -- saf GBM değil, bu raporda BEYAN edilmeli). Otorite yine de
# imaging-agent'in çıkarım anında dondurduğu liste -- ölçülen sayı
# 295/169 ile UYUŞMAZSA (override verilmedikçe) SERT durur.
EXPECTED_UCSF_EXTERNAL_TEST_PATIENTS = 295
EXPECTED_UCSF_EXTERNAL_TEST_EVENTS = 169

# 2026-08-18 EKLENDİ (koordinatör kararı, aynı gün -- CLAUDE.md/decisions/
# 2026-08-18-tek-model-ucsf-harici-test-k15-kapanisi.md §0 ÇELİŞKİ 3'ün
# doldurulması): UCSF, TCGA'nın bıraktığı harici test rolünü devraldığı
# için AYNI ComBat muamelesini görür --
#   ✅ görüntü-seviyesi N4 + T1ce-özel kaynak Z-score (kendi kohortundan
#      fit edilir, imaging-agent'in işi -- bu script'in kapsamı DIŞINDA)
#   ❌ feature-level ComBat düzeltmesi ALMAZ
#   ❌ ComBat fit'ine GİRMEZ (fit yalnız UPenn+LUMIERE'i görür, CLAUDE.md
#      kilitli kuralı DEĞİŞMEDİ)
# Gerekçe TCGA'dakiyle birebir aynı: harici test setine eğitim-türevi bir
# dönüşüm uygulamak bağımsızlığını zedeler; ayrıca ComBat birincil Cox
# zincirinde zaten özdeşlik dönüşümüdür (decisions/2026-08-13-combat-
# ozdeslik-bulgusu-ve-kapsami.md). BU SCRIPT'TEKİ FİİLİ DURUM: `pivot_
# ucsf_wt_long_to_wide()`/`build_clinical_covariate_frame()`/
# `run_primary_single_model()` HİÇBİR NOKTADA `pipeline.harmonization.
# apply_combat_harmonization()`/`fit_combat_harmonization()`'ı ÇAĞIRMAZ
# -- UCSF radyomik özellikleri C32 çıkarımından (imaging-agent'in
# yazacağı, bu script'e HAM olarak gelen) DOĞRUDAN kullanılır. Bu,
# `tests/test_train_cox_week3.py::test_run_primary_single_model_never_
# applies_combat_to_ucsf` ile monkeypatch tabanlı bir regresyon testiyle
# KİLİTLENDİ (fonksiyonlar çağrılırsa test PATLAR).
UCSF_NO_COMBAT_NOTE = (
    "UCSF, TCGA'nin biraktigi harici test rolunu devraldigi icin AYNI "
    "muameleyi gorur: goruntu-seviyesi N4 + T1ce-ozel kaynak Z-score "
    "ALIR, feature-level ComBat duzeltmesi ALMAZ, ComBat fit'ine "
    "GIRMEZ (fit yalniz UPenn+LUMIERE'i gorur). Gerekce TCGA'dakiyle "
    "ayni (2026-08-09 karari) + ComBat birincil Cox zincirinde zaten "
    "ozdeslik donusumudur (decisions/2026-08-13-combat-ozdeslik-"
    "bulgusu-ve-kapsami.md). decisions/2026-08-18-tek-model-ucsf-"
    "harici-test-k15-kapanisi.md SS0 CELISKI 3."
)

ALL_107_FEATURES: tuple[str, ...] = tuple(
    sorted(set(STABLE_FEATURES_ICC60) | set(EXCLUDED_FIRST_ORDER_ABSOLUTE_SCALE_FEATURES))
)
if len(ALL_107_FEATURES) != 107:  # pragma: no cover -- prespesifikasyon koruması
    raise RuntimeError(
        f"ALL_107_FEATURES 107 ozellik icermeli, {len(ALL_107_FEATURES)} bulundu -- "
        "STABLE_FEATURES_ICC60/EXCLUDED_FIRST_ORDER_ABSOLUTE_SCALE_FEATURES "
        "kesisimi bozulmus olabilir. DURDUR ve Baris'a raporla."
    )


# =====================================================================
# İstisnalar -- hepsi "sessiz düşme yerine loud dur" ilkesine göre
# =====================================================================


class C32DataNotFoundError(RuntimeError):
    """`radiomics` tablosunda beklenen `segmentation_tool` icin 0 satir.

    CLAUDE.md: "C32 ile yeniden cikarim tamamlanmadan bu adimlara gecen
    olursa DURDUR ve uyar." Bu istisna TAM OLARAK bunu yapar -- eski
    A-yontemi veriye (`UPenn-PyRadiomics-107`, `TCGA-ground-truth`)
    SESSIZCE DUSULMEZ.
    """


class TimepointWhitelistMissingError(RuntimeError):
    """`segmentation_tool` icin timepoint beyaz listesi TANIMLI DEGIL.

    Fail-closed: beyan yoksa filtresiz cekime DUSMEK yerine durulur --
    aksi halde 611/585 kilidinin ikinci katmani sessizce devre disi
    kalirdi.
    """


class UnexpectedTimepointLabelError(RuntimeError):
    """`mr_scans.timepoint_label` NULL veya ANALIZ EDILMEMIS bir deger.

    Bu satirlar HICBIR kosulda havuza DAHIL EDILMEZ (SQL beyaz listesi
    onlari zaten dislar); bu istisna "sessiz dislama" olmasin diye
    firlatilir -- bilinmeyen bir etiketin nereye ait oldugu (pre? post?
    baska bir zaman noktasi?) INSAN karari gerektirir.
    """


class TrainingPoolCountMismatchError(RuntimeError):
    """Olculen hasta/olay sayisi beklenen (611/585) ile uyusmuyor.

    `--allow-unexpected-patient-count` ile bilincli olarak gecilebilir
    (ornegin C32 cikarimi tamamlanip yeni hastalar eklendiginde beklenen
    sayi henuz guncellenmemisse).
    """


class FallbackAuditMismatchError(RuntimeError):
    """Fallback/frekans audit'i `run_nested_cv()`'nin kendi sonucuyla
    UYUSMUYOR -- audit run_nested_cv() ile senkron degil demektir,
    sessizce yanlis bir fallback/frekans raporu URETILMEZ.
    """


# =====================================================================
# PROTOKOL-DISI STABILITE ESIGI UYARISI (2026-09-13, Baris'in karari:
# "10) icin de beyan arti uyari eklensin")
#
# Protokol birincil stabilite esigini 0,6'ya KILITLER (CLAUDE.md
# "Cox modelleme protokolu KILITLIDIR" + decisions/2026-08-14-stabilite-
# secimi-esigi.md). Ancak `--primary-threshold` bir CLI argumani --
# `--primary-threshold 0.5` yazan bir kosuyu kod DURDURMUYORDU ve
# uretilen artefaktlar (isim/metadata disinda) protokol kosusundan
# AYIRT EDILEMIYORDU.
#
# NEDEN `assert` DEGIL UYARI: Baris'in talimati acikca "uyari eklensin"
# -- bayrak zorunlulugu/durdurma ISTENMEDI. Duyarlilik kollari
# (ornegin 0,7) mesru sekilde farkli esikle kosulur; kosuyu durdurmak
# mesru islerin onunu keserdi. Sert `assert` deseni deploy
# artefaktlarinda korunur (tools/export_v3b_deployment_checkpoint.py
# `assert args.primary_threshold == 0.6`) -- orada dogru yer orasi,
# cunku ORADA tek bir kilitli kosu dogrulaniyor.
# =====================================================================

PROTOCOL_PRIMARY_STABILITY_THRESHOLD: float = 0.6


def warn_if_off_protocol_primary_threshold(
    threshold: float, *, stream: Any = None
) -> bool:
    """`--primary-threshold` protokol degeri (0,6) DISINDA ise stderr'e
    GURULTULU bir uyari basar. Kosuyu DURDURMAZ.

    Dondurur: uyari basildiysa True (test edilebilirlik icin).
    """

    target = stream if stream is not None else sys.stderr
    if float(threshold) == PROTOCOL_PRIMARY_STABILITY_THRESHOLD:
        return False

    banner = "!" * 72
    print(
        f"\n{banner}\n"
        f"!!! PROTOKOL DISI ESIK -- BIRINCIL STABILITE FREKANS ESIGI "
        f"{threshold} KULLANILIYOR\n"
        f"!!! Protokolun KILITLI birincil degeri: "
        f"{PROTOCOL_PRIMARY_STABILITY_THRESHOLD}\n"
        "!!! Referans: CLAUDE.md 'Cox modelleme protokolu KILITLIDIR' + "
        "decisions/2026-08-14-stabilite-secimi-esigi.md\n"
        "!!! Bu kosunun ciktilari BIRINCIL SONUC olarak raporlanamaz; "
        "kullanilan esik raporda\n"
        "!!! ACIKCA beyan edilmelidir (seffaflik kurali, CLAUDE.md "
        "'MODEL GELISTIRME SERBESTTIR').\n"
        "!!! Kosu DURDURULMADI (uyari amaclidir).\n"
        f"{banner}\n",
        file=target,
        flush=True,
    )
    return True


# =====================================================================
# KOL ADI ETIKETI + KOL CHECKPOINT PARMAK IZI
# (2026-09-14, modeling-agent-S1 -- Baris'in "yanlis degerler gormek
#  istemiyorum, herhangi bir zamanda" sarti)
#
# NEDEN (1): `arm_configs` icindeki kol adlari SABIT DIZGIYDI
# (`primary_wt93_icc60_th06`), esik ise CLI'dan (`--primary-threshold`)
# geliyordu -- `--primary-threshold 0.5` ile kosulan bir cikti
# `..._th06_...` adiyla diske yaziliyordu. Bu, damgasizliktan KOTUDUR:
# aktif olarak YANLIS bilgi verir. Ad artik GERCEK esikten turetilir.
#
# NEDEN (2): kol checkpoint'i (`_checkpoints/arm_*.pkl`) yalnizca DOSYA
# VAR MI diye bakiyordu -- esik/seed/grid HIC karsilastirilmiyordu.
# Senaryo: 0,5 ile kosuldu -> yazildi; ertesi gun 0,6 ile kosuldu ->
# dosya var -> yeniden hesaplanmadi -> 0,5'in sonucu "0,6 birincil"
# olarak raporlandi. Cozum: `main()`'deki VERI onbelleginin
# (`_cached_fetch`) parmak izi desenini BIREBIR taklit etmek -- yan
# dosyada meta, anahtar EKSIK veya deger UYUSMUYORSA fail-closed.
# =====================================================================


def format_threshold_tag(threshold: float) -> str:
    """Kol adindaki `th..` etiketini GERCEK esikten turetir.

    GERIYE DONUK UYUM ZORUNLU: protokol varsayilanlari BUGUNKU etiketi
    uretmeli -- 0,6 -> `"th06"`, 0,7 -> `"th07"`. Diskte bu adlarla
    yazilmis artefaktlar ve onlara ATIF VEREN kod var
    (`api/predict.py`, `demo/demo_helpers.py`,
    `tools/shap_explainer_prototype.py`,
    `tools/_dogrulama/inspect_checkpoint*.py`,
    `tests/test_train_cox_week3.py`) -- etiket degisirse bunlar kirilir.

    Kural: ondalik gosterimden noktayi at, sondaki sifirlari kirp.
      0.6  -> "0.6"  -> "th06"     (bugunku ad, DEGISMEDI)
      0.7  -> "0.7"  -> "th07"     (bugunku ad, DEGISMEDI)
      0.5  -> "0.5"  -> "th05"     (ONCEDEN yanlis "th06" yaziliyordu)
      0.65 -> "0.65" -> "th065"
    """

    text = f"{float(threshold):.6f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return "th" + text.replace(".", "").replace("-", "m")


def build_arm_checkpoint_fingerprint(
    *,
    threshold: float,
    seed: int,
    outer_splits: int,
    inner_splits: int,
    n_bootstrap_stability: int,
    n_bootstrap_external: int,
    feature_set: str,
    l1_ratio_grid: Any = None,
    penalizer_grid: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Kol checkpoint'inin GECERLILIK parmak izi.

    JSON'a yazilip geri okundugunda AYNI degerleri uretmeli -- bu yuzden
    tuple'lar listeye, sayilar int/float'a NORMALIZE edilir (JSON tuple
    tanimaz; normalize edilmezse her karsilastirma "uyusmuyor" derdi).
    """

    fingerprint: dict[str, Any] = {
        "threshold": float(threshold),
        "seed": int(seed),
        "outer_splits": int(outer_splits),
        "inner_splits": int(inner_splits),
        "n_bootstrap_stability": int(n_bootstrap_stability),
        "n_bootstrap_external": int(n_bootstrap_external),
        "feature_set": str(feature_set),
        "l1_ratio_grid": [float(value) for value in (l1_ratio_grid or ())],
        "penalizer_grid": [float(value) for value in (penalizer_grid or ())],
    }
    if extra:
        for key, value in extra.items():
            if isinstance(value, (list, tuple, set)):
                fingerprint[key] = sorted(str(item) for item in value)
            elif isinstance(value, bool):
                fingerprint[key] = bool(value)
            elif isinstance(value, (int, float)):
                fingerprint[key] = value
            else:
                fingerprint[key] = None if value is None else str(value)
    return fingerprint


def arm_checkpoint_meta_path(ckpt_path: Path) -> Path:
    """`.../arm_X.pkl` -> `.../arm_X.meta.json` (yan dosya deseni --
    `_cached_fetch`'teki `*.meta.json` ile AYNI)."""

    return ckpt_path.with_suffix(".meta.json")


def load_arm_checkpoint(
    ckpt_path: Path,
    fingerprint: dict[str, Any],
    *,
    arm_name: str,
    stream: Any = None,
) -> Any | None:
    """Kaydedilmis kol sonucunu YALNIZ parmak izi eslesiyorsa dondurur.

    FAIL-CLOSED (`_cached_fetch` ile AYNI disiplin): meta dosyasi YOKSA
    (parmak izi kapisindan ONCE yazilmis eski checkpoint), okunamazsa
    veya HERHANGI bir alan uyusmuyorsa `None` doner -> cagiran yeniden
    hesaplar. Yeniden hesaplama GEREKCESI stderr'e yazilir (sessizce
    dusup yanlis sayi raporlanmasin).
    """

    target = stream if stream is not None else sys.stderr
    if not ckpt_path.is_file():
        return None

    meta_path = arm_checkpoint_meta_path(ckpt_path)
    if not meta_path.is_file():
        print(
            f"[checkpoint] {arm_name}: PARMAK IZI DOSYASI YOK "
            f"({meta_path.name}) -- kaydin hangi esik/seed/grid ile "
            "uretildigi DOGRULANAMIYOR, fail-closed: yeniden hesaplaniyor.",
            file=target,
            flush=True,
        )
        return None

    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(
            f"[checkpoint] {arm_name}: parmak izi okunamadi "
            f"({type(exc).__name__}) -- yeniden hesaplaniyor.",
            file=target,
            flush=True,
        )
        return None

    if not isinstance(stored, dict):
        print(
            f"[checkpoint] {arm_name}: parmak izi BEKLENEN BICIMDE DEGIL "
            "-- yeniden hesaplaniyor.",
            file=target,
            flush=True,
        )
        return None

    differences = [
        f"{key}: kayitli={stored.get(key)!r} != istenen={value!r}"
        for key, value in fingerprint.items()
        if stored.get(key) != value
    ]
    if differences:
        print(
            f"[checkpoint] {arm_name}: PARMAK IZI UYUSMUYOR -- kaydedilmis "
            "sonuc KULLANILMADI, yeniden hesaplaniyor. Farklar: "
            + "; ".join(differences),
            file=target,
            flush=True,
        )
        return None

    try:
        arm_result = pd.read_pickle(ckpt_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[checkpoint] {arm_name}: kayit okunamadi "
            f"({type(exc).__name__}) -- yeniden hesaplanacak.",
            file=target,
            flush=True,
        )
        return None

    print(
        f"[checkpoint] {arm_name}: kaydedilmis sonuc yuklendi "
        "(parmak izi ESLESTI), yeniden hesaplanmadi.",
        flush=True,
    )
    return arm_result


def save_arm_checkpoint(
    ckpt_path: Path, arm_result: Any, fingerprint: dict[str, Any]
) -> None:
    """Kol sonucunu + parmak izini diske yazar.

    SIRA ONEMLI: once `.pkl`, sonra `.meta.json`. Arada cokerse meta YOK
    kalir -> bir sonraki kosu fail-closed davranir (yarim kaydi
    "gecerli" sanmaz).
    """

    pd.to_pickle(arm_result, ckpt_path)
    arm_checkpoint_meta_path(ckpt_path).write_text(
        json.dumps(fingerprint, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


# =====================================================================
# 1) DB'den okuma -- yalniz SELECT, segmentation_tool SABIT
#    + timepoint beyaz listesi (611/585 kilidinin ikinci katmani)
# =====================================================================


def allowed_timepoint_labels(segmentation_tool: str) -> tuple[str, ...]:
    """`segmentation_tool` icin IZINLI `mr_scans.timepoint_label` kumesi.

    Fail-closed: beyan YOKSA `TimepointWhitelistMissingError` -- filtresiz
    cekime DUSULMEZ.
    """

    try:
        allowed = ALLOWED_TIMEPOINT_LABELS_BY_TOOL[segmentation_tool]
    except KeyError as exc:
        raise TimepointWhitelistMissingError(
            f"segmentation_tool={segmentation_tool!r} icin timepoint beyaz "
            "listesi TANIMLI DEGIL (ALLOWED_TIMEPOINT_LABELS_BY_TOOL). "
            "Filtresiz cekim YAPILMAZ -- 611/585 kilidinin ikinci katmani "
            "sessizce devre disi kalamaz. Yeni bir kohort ekleniyorsa "
            "izinli timepoint etiketini ACIKCA beyan et."
        ) from exc
    return tuple(sorted(allowed))


def summarize_timepoint_exclusions(
    cursor,
    *,
    segmentation_tool: str,
    stream: Any = None,
) -> dict[str, dict[str, Any]]:
    """Timepoint kapisinin DISLADIGI satirlari SAY, RAPORLA, gerekirse DUR.

    Neden var: "sessiz dislama da sessiz dahil etme kadar tehlikelidir."
    GUARD 6'nin (write_upenn_pyradiomics_to_db.py) "POST-OP FALLBACK
    OZETI" deseninin okuma tarafindaki karsiligidir.

    Davranis:
      * BEKLENEN dislama (`KNOWN_EXCLUDED_TIMEPOINT_LABELS_BY_TOOL`,
        orn. UPenn `post-op`) -> stderr'e GURULTULU ozet, kosu DEVAM.
      * NULL veya ANALIZ EDILMEMIS etiket -> ayni ozet + SERT
        `UnexpectedTimepointLabelError`.
      * Dislanan satir YOK -> hicbir sey basilmaz (bugunun durumu:
        UPenn/TCGA/UCSF C32 satirlarinin %100'u izinli etikette).

    Donus: `{etiket: {"n_rows", "n_patients", "patient_ids"}}`.
    `cursor` RealDictCursor (bu modulun her yerinde oldugu gibi) varsayilir.
    """

    target = stream if stream is not None else sys.stderr
    allowed = list(allowed_timepoint_labels(segmentation_tool))
    known_excluded = KNOWN_EXCLUDED_TIMEPOINT_LABELS_BY_TOOL.get(
        segmentation_tool, frozenset()
    )

    cursor.execute(
        """
        SELECT COALESCE(ms.timepoint_label, '<NULL>') AS timepoint_label,
               COUNT(*) AS n_rows,
               COUNT(DISTINCT ms.patient_id) AS n_patients,
               ARRAY_AGG(DISTINCT ms.patient_id) AS patient_ids
        FROM radiomics r
        JOIN mr_scans ms ON ms.scan_id = r.scan_id
        WHERE r.segmentation_tool = %s
          AND (ms.timepoint_label IS NULL OR NOT (ms.timepoint_label = ANY(%s)))
        GROUP BY 1
        ORDER BY 2 DESC, 1
        """,
        (segmentation_tool, allowed),
    )
    excluded: dict[str, dict[str, Any]] = {}
    for row in cursor.fetchall():
        label = str(row["timepoint_label"])
        excluded[label] = {
            "n_rows": int(row["n_rows"]),
            "n_patients": int(row["n_patients"]),
            "patient_ids": sorted(str(pid) for pid in (row["patient_ids"] or [])),
        }

    if not excluded:
        return excluded

    unanalyzed = sorted(set(excluded) - set(known_excluded))
    banner = "=" * 72
    lines = [
        banner,
        f"TIMEPOINT DISLAMA OZETI -- {segmentation_tool}",
        f"  izinli etiketler : {allowed}",
        f"  beyan edilmis dislamalar: {sorted(known_excluded)}",
    ]
    for label, info in excluded.items():
        flag = "BEKLENEN" if label in known_excluded else "ANALIZ EDILMEMIS"
        ids = info["patient_ids"]
        shown = ", ".join(ids[:20]) + (f" ... (+{len(ids) - 20})" if len(ids) > 20 else "")
        lines.append(
            f"  [{flag}] {label!r}: {info['n_rows']} satir / "
            f"{info['n_patients']} hasta -> {shown}"
        )
    lines.append(
        "  BU SATIRLAR HAVUZA GIRMEDI (SQL beyaz listesi disladi). "
        "Sessiz atlama yok -- kilitli 611/585 havuzunun ikinci katmani."
    )
    lines.append(banner)
    print("\n".join(lines), file=target, flush=True)

    if unanalyzed:
        raise UnexpectedTimepointLabelError(
            f"segmentation_tool={segmentation_tool!r} icin ANALIZ EDILMEMIS "
            f"timepoint etiketi/etiketleri var: {unanalyzed} "
            f"(izinli: {allowed}, beyan edilmis dislama: {sorted(known_excluded)}). "
            "Bu satirlar havuza DAHIL EDILMEDI, ama sessizce de gecilmez -- "
            "'<NULL>' veya yeni bir etiket, analiz edilmemis veri kaymasi "
            "demektir ve hangi zaman noktasina karsilik geldigi INSAN karari "
            "gerektirir. DURDUR ve Baris'a raporla."
        )
    return excluded


def assert_frame_timepoints_allowed(
    frame: pd.DataFrame, *, segmentation_tool: str
) -> None:
    """Cekilen cercevede IZINLI OLMAYAN timepoint etiketi var mi -- SQL
    beyaz listesinin PYTHON tarafindaki ikizi (defense in depth).

    Amaci: sorgu metni ileride yanlislikla degistirilirse (kosul silinir/
    parametre yanlis baglanir) bozukluk SESSIZ kalmasin. Bugun bu
    fonksiyon HICBIR seyi elemez -- yalniz dogrular.
    """

    if frame.empty or "timepoint_label" not in frame.columns:
        return
    allowed = set(allowed_timepoint_labels(segmentation_tool))
    observed = frame["timepoint_label"].astype("object")
    bad_mask = ~observed.isin(allowed) | observed.isna()
    if not bool(bad_mask.any()):
        return
    bad_labels = sorted(
        {("<NULL>" if pd.isna(v) else str(v)) for v in observed[bad_mask].tolist()}
    )
    raise UnexpectedTimepointLabelError(
        f"Cekilen cercevede IZINLI OLMAYAN timepoint etiketi bulundu: "
        f"{bad_labels} (izinli: {sorted(allowed)}, segmentation_tool="
        f"{segmentation_tool!r}). SQL beyaz listesi ({len(frame)} satir cekildi) "
        "bunlari elemis OLMALIYDI -- sorgu metni bozulmus olabilir. "
        "Kosu DURDURULDU; bu satirlar havuza GIRMEDI."
    )


def count_c32_radiomics_rows(cursor, *, segmentation_tool: str) -> int:
    """`radiomics` satir sayisi -- `fetch_c32_radiomics_long_frame()` ile
    AYNI timepoint kosulunu uygular.

    Neden ayri fonksiyon: `main()`'in onbellek gecerlilik kontrolu satir
    sayisini cekilen cerceve ile KARSILASTIRIYOR. Sayimda timepoint
    kosulu olmazsa (eskiden oyleydi) ilk post-op satir geldigi anda
    `len(frame) != db_rows` olur ve onbellek SONSUZA KADAR gecersiz
    sayilir -- kapinin dogru calistigi durumda yanlis alarm.
    """

    assert_c32_segmentation_tool_allowed(segmentation_tool)
    allowed = list(allowed_timepoint_labels(segmentation_tool))
    cursor.execute(
        """
        SELECT COUNT(*) AS n
        FROM radiomics r
        JOIN mr_scans ms ON ms.scan_id = r.scan_id
        WHERE r.segmentation_tool = %s
          AND ms.timepoint_label = ANY(%s)
        """,
        (segmentation_tool, allowed),
    )
    return int(cursor.fetchone()["n"])


def fetch_c32_radiomics_long_frame(
    cursor, *, segmentation_tool: str, report_stream: Any = None
) -> pd.DataFrame:
    """`radiomics` tablosundan UZUN format cek -- `pivot_radiomics_long_
    to_wide()`'in dogrudan girdisi.

    `segmentation_tool` bu fonksiyonun cagirani (bu modulun `main()`'i)
    TARAFINDAN HER ZAMAN `SEGMENTATION_TOOL_UPENN_C32`/`SEGMENTATION_
    TOOL_TCGA_C32` sabitleriyle cagrilir -- CLI'dan operator tarafindan
    DEGISTIRILEMEZ (gorev kisiti: "segmentation_tool adi parametre
    DEGIL, sabit"). Fonksiyonun kendisi bu ikisini kabul eder (DRY --
    UPenn/TCGA ayni sorgu seklini paylasir), ama disariya PARAMETRE
    olarak ACILMAZ.

    TIMEPOINT KAPISI (2026-09-13/14, karar 39 -- 611/585 kilidinin IKINCI
    KATMANI): sorgu artik `mr_scans.timepoint_label`'i ARAC BASINA BEYAZ
    LISTEYLE (`ALLOWED_TIMEPOINT_LABELS_BY_TOOL`) kisitlar. `= ANY(%s)`
    NULL ile ASLA eslesmedigi icin NULL/beklenmeyen etiketli satirlar
    OTOMATIK olarak DISARIDA kalir (fail-closed), ve dislanan her satir
    `summarize_timepoint_exclusions()` tarafindan cekimden ONCE
    raporlanir (analiz edilmemis etiket varsa cekime BASLAMADAN durur --
    11 dakikalik bosa cekim olmaz).
    """

    assert_c32_segmentation_tool_allowed(segmentation_tool)
    allowed_timepoints = list(allowed_timepoint_labels(segmentation_tool))

    # Dislama raporu + fail-closed drift kapisi CEKIMDEN ONCE.
    summarize_timepoint_exclusions(
        cursor, segmentation_tool=segmentation_tool, stream=report_stream
    )

    # SAYFALI CEKIM (2026-08-14, koordinator duzeltmesi -- KOSU ENGELI):
    # Tek seferde cekim UPenn'de (3055 satir x 3 buyuk JSONB kolon) GERCEK
    # kosuda IKI KEZ dustu: once `QueryCanceled: statement timeout`, sonra
    # `OperationalError: SSL connection has been closed unexpectedly`.
    # Sorun veri/mantik degil, tek sorgunun tasidigi JSONB hacmi.
    #
    # Cozum: `scan_id` sirasina gore SAYFALI cekim. Determinizm icin siralama
    # ZORUNLU (aksi halde sayfalar arasinda satir atlanabilir/tekrarlanabilir --
    # ayni hata sinifi 2026-08-14'te `run_pyradiomics_upenn.py`'de de duzeltildi).
    # Veri icerigi DEGISMEZ: sayfalarin birlesimi tek-sorgu sonucuyla ozdestir.
    page_size = 250
    offset = 0
    rows: list = []
    while True:
        cursor.execute(
            """
            SELECT r.scan_id, p.patient_id, ds.source_name AS source, r.tumor_region,
                   ms.timepoint_label,
                   r.shape_features, r.first_order_features, r.texture_features
            FROM radiomics r
            JOIN mr_scans ms ON ms.scan_id = r.scan_id
            JOIN patients p ON p.patient_id = ms.patient_id
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE r.segmentation_tool = %s
              AND ms.timepoint_label = ANY(%s)
            ORDER BY r.scan_id, r.tumor_region
            LIMIT %s OFFSET %s
            """,
            (segmentation_tool, allowed_timepoints, page_size, offset),
        )
        page = cursor.fetchall()
        if not page:
            break
        rows.extend(page)
        offset += page_size

    frame = pd.DataFrame(rows)
    if not frame.empty:
        # Python tarafi ikiz kontrol -- SQL kosulu ileride bozulursa
        # sessiz kalmasin (bugun hicbir satir elemez, yalniz dogrular).
        assert_frame_timepoints_allowed(frame, segmentation_tool=segmentation_tool)
        # `scan_id` yalniz sayfalama/determinizm icin cekildi, `timepoint_
        # label` yalniz yukaridaki dogrulama icin; asagi akista ikisi de
        # kullanilmiyor -- pivot sozlesmesi DEGISMESIN diye dusuruluyor.
        frame = frame.drop(columns=["scan_id", "timepoint_label"])
    return frame


def raise_if_empty_c32(long_frame: pd.DataFrame, *, segmentation_tool: str) -> None:
    if long_frame.empty:
        raise C32DataNotFoundError(
            f"radiomics tablosunda segmentation_tool={segmentation_tool!r} "
            "icin 0 satir. C32 cikarimi/yuklemesi henuz DB'ye YAZILMAMIS "
            "(CLAUDE.md: 'C32 ile yeniden cikarim tamamlanmadan bu "
            "adimlara gecen olursa DURDUR'). Eski A-yontemi veriye "
            "(UPenn-PyRadiomics-107 / TCGA-ground-truth) SESSIZCE "
            "DUSULMEDI -- bu kasitli, kod seviyesinde zorlanan bir "
            "durma noktasidir."
        )


PATIENTS_FRAME_CLINICAL_RAW_COLUMNS: tuple[str, ...] = (
    "age",
    "gender",
    "gtr_over90percent",
    "idh1_status",
    # 2026-08-18 EKLENDİ (Model v2, GÖREV: MGMT kovaryatı -- CLAUDE.md
    # 2026-08-18 güncellemesi: "MODEL GELİŞTİRME SERBEST, tek koşul
    # ŞEFFAFLIK"). Harici test TCGA'dan UCSF'e geçtiği için MGMT artık
    # kullanılabilir (TCGA'da MGMT %0 doluydu, UCSF'te %95,9 -- canlı DB,
    # 2026-08-18 SELECT ile doğrulandı). Ham değerler (canlı DB):
    #   UPenn (630 kayıtlı): Methylated 115 / Unmethylated 160 /
    #     Indeterminate 28 / NULL 327.
    #   UCSF (295): Methylated 201 / Unmethylated 82 / NULL 12
    #     (Indeterminate/"unknown" `tools/onboard_ucsf_patients.py`'nin
    #     `map_mgmt()`'i tarafından onboarding anında zaten NULL'a
    #     çevrilmiş -- UCSF'te ham "Indeterminate" değeri HİÇ görünmez).
    "mgmt_status",
)


def fetch_patients_frame(cursor, *, source_name: str) -> pd.DataFrame:
    """`patients` x `dataset_sources` -- `source`/`survival_days`/
    `vital_status` + HAM klinik kolonlar (`age`/`gender`/
    `gtr_over90percent`/`idh1_status`), `patient_id` index'li.

    2026-08-15 GENİŞLETİLDİ (klinik kovaryat desteği): eskiden yalnız
    `source`/`survival_days`/`vital_status` çekiyordu. Dört ham klinik
    kolon EKLENDİ -- bu fonksiyonun İMZASI (parametreler) DEĞİŞMEDİ,
    yalnız döndürülen DataFrame'in kolonları genişledi. Mevcut
    çağıranlar (bkz. `main()`) ETKİLENMEZ: `build_training_frame()`/
    `_assemble_training_frame()`/`build_external_test_frame()` zaten
    yalnız İHTİYAÇ DUYDUKLARI kolonları `merged[...]` ile AÇIKÇA seçip
    çıktı çerçevesine taşıyor (bkz. o fonksiyonların kaynak kodu) --
    fazladan kolonlar SESSİZCE hiçbir yere sızmaz, yalnız BİLİNÇLİ
    olarak `passthrough_columns` ile geçirilirse Cox çerçevesine girer.

    Ham kolonların DB'deki gerçek değerleri (2026-08-15, canlı DB,
    readonly SELECT ile ölçüldü -- UPenn-GBM 630 kayıtlı hasta):
      - `gender`: `"Male"`/`"Female"`/`NULL` (UPenn'de NULL YOK, 630/630 dolu).
      - `idh1_status`: `"Wildtype"`/`"Mutated"`/`"NOS/NEC"`/`NULL`.
      - `gtr_over90percent`: `"Y"`/`"N"`/`NULL`.
      - `age`: tamsayı, UPenn'de NULL YOK.
    Bu fonksiyon KODLAMA/dönüşüm YAPMAZ (ham değerleri olduğu gibi
    döndürür) -- bkz. `build_clinical_covariate_frame()` (kodlama TEK
    yerde, orada yapılır).
    """

    columns_sql = ", ".join(f"p.{col}" for col in PATIENTS_FRAME_CLINICAL_RAW_COLUMNS)
    cursor.execute(
        f"""
        SELECT p.patient_id, ds.source_name AS source,
               p.survival_days, p.vital_status, {columns_sql}
        FROM patients p
        JOIN dataset_sources ds ON ds.source_id = p.source_id
        WHERE ds.source_name = %s
        """,
        (source_name,),
    )
    rows = cursor.fetchall()
    expected_columns = [
        "patient_id",
        "source",
        "survival_days",
        "vital_status",
        *PATIENTS_FRAME_CLINICAL_RAW_COLUMNS,
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=expected_columns)
    frame = frame.set_index("patient_id")
    return frame


# =====================================================================
# 1.5) Klinik kovaryat kodlama (2026-08-15, klinik kovaryat destegi)
# =====================================================================
#
# KODLAMA KURALI (gorev talimati, ONCEDEN DONDURULMUS -- sonuca bakip
# DEGISTIRILMEZ):
#   - yas: dogrusal/surekli (olculen HR 1,032/yil temiz log-lineer).
#     Spline/kategorilestirme ARANMAZ.
#   - cinsiyet: ikili.
#   - GTR: Y=1/N=0 + AYRI bir gtr_missing gosterge degiskeni (eksik
#     hasta DUSURULMEZ).
#   - IDH (2026-08-18 GÜNCELLENDİ -- K15 KAPANDI, Barış kararı):
#     GTR ile BİREBİR AYNI "gösterge değişkeni" deseni. NOS/NEC (IDH testi
#     yapılmamış, 96 hasta) ARTIK KOHORTTAN DÜŞÜRÜLMEZ -- 2026-08-15
#     ölçümü NOS/NEC'in yaşa-düzeltilmiş HR 1,410 [1,128-1,764] p=0,0026
#     ile daha kötü prognozlu olduğunu gösterdi (rastgele eksik DEĞİL,
#     düşürmek kohortu yanlılaştırır). Kodlama: Mutant=1/Wildtype=0/
#     NOS-NEC=0 (referans wildtype) + AYRI `clinical_idh_missing`
#     (NOS/NEC veya ham NULL -> 1, aksi -> 0). Eski "515'lik alt kohort,
#     NOS/NEC NaN bırakılır" davranışı ARTIK SADECE `filter_to_idh_known_
#     cohort()` ile AÇIKÇA istenen duyarlılık kollarında kullanılır (bu
#     mekanizma SİLİNMEDİ, bkz. o fonksiyon) -- birincil/tek model artık
#     tam 611 kohortu kullanır.

CLINICAL_AGE_COLUMN = "clinical_age"
CLINICAL_GENDER_MALE_COLUMN = "clinical_gender_male"
CLINICAL_GTR_Y_COLUMN = "clinical_gtr_y"
CLINICAL_GTR_MISSING_COLUMN = "clinical_gtr_missing"
CLINICAL_IDH_MUTANT_COLUMN = "clinical_idh_mutant"
CLINICAL_IDH_MISSING_COLUMN = "clinical_idh_missing"
# 2026-08-18 EKLENDİ (Model v2 -- MGMT kovaryatı, GTR/IDH ile BİREBİR
# AYNI "gösterge değişkeni" deseni, bkz. `build_clinical_covariate_frame()`
# `include_mgmt` bölümü).
CLINICAL_MGMT_METHYLATED_COLUMN = "clinical_mgmt_methylated"
CLINICAL_MGMT_MISSING_COLUMN = "clinical_mgmt_missing"

GENDER_MALE_LABEL = "Male"
GENDER_FEMALE_LABEL = "Female"
GTR_YES_LABEL = "Y"
GTR_NO_LABEL = "N"
IDH1_WILDTYPE_LABEL = "Wildtype"
IDH1_MUTATED_LABEL = "Mutated"
IDH1_NOS_LABEL = "NOS/NEC"
MGMT_METHYLATED_LABEL = "Methylated"
MGMT_UNMETHYLATED_LABEL = "Unmethylated"
# "Indeterminate" (UPenn'de -- test yapıldı ama sonuç belirsiz) SADECE
# UPenn'de görülür (canlı DB, 2026-08-18: UCSF'te HİÇ yok -- db-agent'in
# `map_mgmt()`'i onboarding anında Indeterminate/"unknown"'ı zaten NULL'a
# çevirmiş). KARAR (görev talimatının açıkça istediği, gerekçeli):
# Indeterminate "eksik" SAYILIR (MGMT_MISSING=1) -- gerekçe: PCR/pyroseq
# testi YAPILDI ama SONUÇ istatistiksel olarak güvenilir/yorumlanabilir
# değil (kit-bağımlı eşik belirsizliği) -- yani hastanın GERÇEK MGMT
# durumu hakkında KLİNİK OLARAK KULLANILABİLİR bilgi YOK, GTR/IDH'deki
# "test yapılmadı" (NOS/NEC) ile AYNI epistemik durumda: "bilgi yok".
# Ayrı bir üçüncü kategori (örn. mgmt_indeterminate=1 ayrı sütun) DEĞİL
# -- GTR/IDH'nin ikili (bilinen-değer + missing-gösterge) deseniyle
# TUTARLI tutuldu, modele yeni bir serbestlik derecesi eklemedi.
MGMT_INDETERMINATE_LABEL = "Indeterminate"


class UnrecognizedClinicalValueError(RuntimeError):
    """Ham klinik kolonda (`gender`/`gtr_over90percent`/`idh1_status`)
    beklenmeyen bir deger bulundu -- sessizce yutulmaz/varsayilan bir
    kategoriye DUSURULMEZ (CLAUDE.md "sessiz fallback yasak" ilkesi)."""


def _assert_recognized_categorical(
    raw: pd.Series, *, column_name: str, recognized: set[str]
) -> None:
    non_null = raw.dropna()
    unrecognized = non_null.loc[~non_null.isin(recognized)]
    if not unrecognized.empty:
        raise UnrecognizedClinicalValueError(
            f"patients_frame[{column_name!r}] icinde taninmayan "
            f"deger(ler): {sorted(unrecognized.unique())} -- beklenen: "
            f"{sorted(recognized)} (veya NULL)."
        )


def build_clinical_covariate_frame(
    patients_frame: pd.DataFrame,
    *,
    include_gender: bool = True,
    include_gtr: bool = False,
    include_idh: bool = False,
    include_mgmt: bool = False,
) -> pd.DataFrame:
    """Ham klinik kolonlari (bkz. `PATIENTS_FRAME_CLINICAL_RAW_COLUMNS`)
    modele hazir sayisal kolonlara cevirir -- KODLAMA burada TEK YERDE
    yapilir (yukaridaki "KODLAMA KURALI" notuna gore).

    - yas (`CLINICAL_AGE_COLUMN`): HAM float olarak doner --
      standardizasyon burada YAPILMAZ, fold-ici (pipeline.cox_model.
      standardize_columns_fold_safe) ayrica uygulanir (sizinti onleme,
      gorev talimati).
    - cinsiyet (`CLINICAL_GENDER_MALE_COLUMN`): Male=1/Female=0. NULL
      SESSIZCE 0'a DUSURULMEZ -- NaN kalir, `build_training_frame`/
      `build_external_test_frame`'in MEVCUT "eksik ozellik -> dusur +
      say" mekanizmasi (passthrough_columns / missing_feature_mask) bunu
      yakalar.
    - GTR (`CLINICAL_GTR_Y_COLUMN` + `CLINICAL_GTR_MISSING_COLUMN`):
      Y=1/N=0, eksikse GTR_Y=0 + GTR_MISSING=1 (sabit-doldurma + ayri
      gosterge -- eksik hasta DUSURULMEZ, gorev talimati).
    - IDH (`CLINICAL_IDH_MUTANT_COLUMN` + `CLINICAL_IDH_MISSING_COLUMN`,
      yalniz `include_idh=True`, 2026-08-18 GÜNCELLENDİ -- K15 KAPANDI):
      GTR ile BİREBİR AYNI "gösterge değişkeni" deseni. `CLINICAL_IDH_
      MISSING_COLUMN` = 1 eğer NOS/NEC (test yapılmamış) VEYA ham NULL,
      aksi halde 0. `CLINICAL_IDH_MUTANT_COLUMN` = Mutated->1, Wildtype
      VEYA NOS/NEC VEYA NULL -> 0 (referans wildtype -- eksiklik AYRI
      sütunda taşınıyor, bu kolonda NaN KALMAZ). Eski davranış (NOS/NEC
      için NaN bırakıp kohort filtresiyle 515'e daraltmak) hâlâ
      `filter_to_idh_known_cohort()` ile İSTEĞE BAĞLI olarak uygulanabilir
      (duyarlılık kolları için, bu fonksiyon kohort filtrelemesi YAPMAZ,
      yalnız kodlama yapar) -- ama BİRİNCİL/tek model bu filtreyi
      KULLANMAZ, tam 611 kohortu geçirir.
    - MGMT (`CLINICAL_MGMT_METHYLATED_COLUMN` + `CLINICAL_MGMT_MISSING_
      COLUMN`, yalnız `include_mgmt=True`, 2026-08-18 EKLENDİ -- Model
      v2): GTR/IDH ile BİREBİR AYNI "gösterge değişkeni" deseni.
      `CLINICAL_MGMT_MISSING_COLUMN` = 1 eğer `Indeterminate` VEYA ham
      NULL, aksi halde 0. `CLINICAL_MGMT_METHYLATED_COLUMN` = Methylated
      -> 1, Unmethylated VEYA Indeterminate VEYA NULL -> 0 (referans
      unmethylated/bilinmiyor -- eksiklik AYRI sütunda taşınır, bu
      kolonda NaN KALMAZ). KARAR GEREKÇESİ (Indeterminate = eksik
      sayılır): bkz. `MGMT_INDETERMINATE_LABEL` sabitinin yanındaki not
      -- test yapıldı ama klinik olarak kullanılabilir bir MGMT durumu
      ÜRETMEDİ, GTR/IDH'nin "test yapılmadı" kategorisiyle AYNI epistemik
      durum. `recognized` kümesi hem UPenn'in (Methylated/Unmethylated/
      Indeterminate) hem UCSF'in (Methylated/Unmethylated -- Indeterminate
      hiç görünmez, db-agent onboarding'de zaten NULL'a çevrilmiş) ham
      sözlüğünü KAPSAR -- `_assert_recognized_categorical()` UCSF'te
      hiç görülmeyen "Indeterminate"i reddetmez (tanınan küme fazlası
      zararsız), ama iki kaynağın DA sözlük DIŞI bir değer (yazım hatası)
      üretmesini AYNI guard'la yakalar.

    Taninmayan bir ham deger (yazim hatasi/sema disi kategori) SESSIZCE
    gecilmez -- `UnrecognizedClinicalValueError` firlatilir.
    """

    missing_raw = set(PATIENTS_FRAME_CLINICAL_RAW_COLUMNS) - set(patients_frame.columns)
    if missing_raw:
        raise ValueError(f"patients_frame eksik ham klinik kolon(lar): {sorted(missing_raw)}")

    result = pd.DataFrame(index=patients_frame.index)
    result[CLINICAL_AGE_COLUMN] = patients_frame["age"].astype(float)

    if include_gender:
        gender_raw = patients_frame["gender"]
        _assert_recognized_categorical(
            gender_raw, column_name="gender", recognized={GENDER_MALE_LABEL, GENDER_FEMALE_LABEL}
        )
        result[CLINICAL_GENDER_MALE_COLUMN] = gender_raw.map(
            {GENDER_MALE_LABEL: 1.0, GENDER_FEMALE_LABEL: 0.0}
        )

    if include_gtr:
        gtr_raw = patients_frame["gtr_over90percent"]
        _assert_recognized_categorical(
            gtr_raw, column_name="gtr_over90percent", recognized={GTR_YES_LABEL, GTR_NO_LABEL}
        )
        result[CLINICAL_GTR_MISSING_COLUMN] = gtr_raw.isna().astype(float)
        result[CLINICAL_GTR_Y_COLUMN] = (
            gtr_raw.map({GTR_YES_LABEL: 1.0, GTR_NO_LABEL: 0.0}).fillna(0.0)
        )

    if include_idh:
        idh_raw = patients_frame["idh1_status"]
        _assert_recognized_categorical(
            idh_raw,
            column_name="idh1_status",
            recognized={IDH1_WILDTYPE_LABEL, IDH1_MUTATED_LABEL, IDH1_NOS_LABEL},
        )
        # 2026-08-18 GÜNCELLENDİ (K15 KAPANDI, Barış kararı) -- GTR ile
        # BİREBİR AYNI "gösterge değişkeni" deseni: eksiklik (NOS/NEC VEYA
        # ham NULL) AYRI bir sütunda taşınır, `clinical_idh_mutant`'ta
        # ARTIK NaN KALMAZ (eskiden NOS/NEC -> NaN idi, kohort filtresi
        # ZORUNLUYDU -- artık DEĞİL, `.fillna(0.0)` referans wildtype'a
        # düşürür, eksiklik bilgisi `clinical_idh_missing`'te kaybolmaz).
        idh_missing_mask = idh_raw.isna() | (idh_raw == IDH1_NOS_LABEL)
        result[CLINICAL_IDH_MISSING_COLUMN] = idh_missing_mask.astype(float)
        result[CLINICAL_IDH_MUTANT_COLUMN] = idh_raw.map(
            {IDH1_MUTATED_LABEL: 1.0, IDH1_WILDTYPE_LABEL: 0.0}
        ).fillna(0.0)  # NOS/NEC + ham NULL -> 0 (referans wildtype)

    if include_mgmt:
        mgmt_raw = patients_frame["mgmt_status"]
        _assert_recognized_categorical(
            mgmt_raw,
            column_name="mgmt_status",
            recognized={
                MGMT_METHYLATED_LABEL,
                MGMT_UNMETHYLATED_LABEL,
                MGMT_INDETERMINATE_LABEL,
            },
        )
        # Indeterminate VEYA ham NULL -> eksik (bkz. MGMT_INDETERMINATE_
        # LABEL sabitinin yanindaki karar gerekcesi -- GTR/IDH ile ayni
        # gosterge-degiskeni deseni).
        mgmt_missing_mask = mgmt_raw.isna() | (mgmt_raw == MGMT_INDETERMINATE_LABEL)
        result[CLINICAL_MGMT_MISSING_COLUMN] = mgmt_missing_mask.astype(float)
        result[CLINICAL_MGMT_METHYLATED_COLUMN] = mgmt_raw.map(
            {MGMT_METHYLATED_LABEL: 1.0, MGMT_UNMETHYLATED_LABEL: 0.0}
        ).fillna(0.0)  # Indeterminate + ham NULL -> 0 (referans unmethylated/bilinmiyor)

    return result


def filter_to_idh_known_cohort(
    patients_frame: pd.DataFrame, *, keep: str = "wildtype_or_mutated"
) -> pd.DataFrame:
    """IDH duyarlilik kollarinin KOHORT filtresi (gorev talimati):
      - `idh_sensitivity` kolu: yalniz Wildtype/Mutated (`keep=
        "wildtype_or_mutated"`) -- NOS/NEC hastalar bu kolun egitim
        havuzundan TAMAMEN cikarilir (yalnizca IDH kolonunda NaN
        BIRAKILMAZ -- kohortun kendisi degisir).
      - `who2021_wildtype` kolu: yalniz Wildtype (`keep="wildtype_
        only"`) -- Mutated + NOS/NEC ikisi de cikar.

    `patients_frame`'i (henuz radyomik ozelliklerle join'lenmemis)
    index bazinda FILTRELER -- cagiran taraf sonucu `build_training_
    frame()`'e gecirmeden ONCE kullanmali, boylece disланan hastalar
    hem nested-CV hem full-pool final modele HIC girmez.
    """

    if keep not in {"wildtype_or_mutated", "wildtype_only"}:
        raise ValueError(f"keep gecersiz: {keep!r} (beklenen: 'wildtype_or_mutated'/'wildtype_only')")

    idh_raw = patients_frame["idh1_status"]
    if keep == "wildtype_or_mutated":
        mask = idh_raw.isin({IDH1_WILDTYPE_LABEL, IDH1_MUTATED_LABEL})
    else:
        mask = idh_raw == IDH1_WILDTYPE_LABEL
    return patients_frame.loc[mask].copy()


# =====================================================================
# 1.55) Yaş restricted cubic spline -- 3 düğüm (2026-08-18, Model v2
#       "yaş spline" bölümü B)
# =====================================================================
#
# GEREKÇE (görev talimatı): yaş en güçlü kovaryat (final modelde HR
# 1,023/yıl, p<0,0001) ama doğrusal giriyor. GBM'de yaş-risk ilişkisi
# 65 üstünde keskinleşir -- literatür 3 düğümlü restricted cubic spline
# (RCS) öneriyor (Harrell, "Regression Modeling Strategies", 2015,
# eş. 2.24 -- 3 düğüm literatür varsayılanı).
#
# UYGULAMA (patsy DEĞİL -- görev talimatı "patsy veya elle bazis
# üretimi" ikisine de izin veriyor; patsy `requirements*.txt`'de
# DEĞİL/örtük bir bağımlılık, elle üretim tercih edildi -- yeni harici
# bağımlılık YOK, formül belgeli/test edilebilir).
#
# k=3 düğüm için Harrell'in RCS bazisi TOPLAM 2 kolon üretir: X0 = x
# (doğrusal terim, ham yaş) + X1 (TEK doğrusal-olmayan terim, k-2=1).
# `CLINICAL_AGE_COLUMN` (1 kolon) bu 2 kolonla DEĞİŞTİRİLİR -- görev
# talimatındaki "Maliyet: +2 parametre" ifadesi tam olarak bunu
# tanımlıyor (yaş artık 1 değil 2 model sütunu işgal ediyor).
#
# SIZINTI ÖNLEME (görev talimatı, kritik): düğüm yerleri (10/50/90
# persentil) SADECE UPenn eğitim havuzunun ham yaş dağılımından BİR KEZ
# hesaplanır (`compute_rcs_knots()`) ve UCSF'e (harici test) AYNEN
# uygulanır -- UCSF'in kendi yaş dağılımından YENİDEN hesaplanmaz.
# AÇIK TASARIM KARARI (dürüstçe belirtiliyor, doğrulanmadı): düğüm
# yerleri nested-CV'nin dış fold'ları İÇİNDE de YENİDEN hesaplanmaz --
# `STABLE_FEATURES_ICC60` (aday özellik listesinin kendisi) gibi
# PRE-SPESİFİYE bir yapısal seçim olarak ele alındı, tam UPenn havuzunda
# (611 hasta) bir kez sabitlendi. Bu, `standardize_columns_fold_safe()`
# ile aynı disiplin DEĞİL (o, ortalama/std'yi HER dış fold içinde
# yeniden fit eder) -- düğüm yerlerinin fold-içi yeniden hesaplanması
# İSTATİSTİKSEL OLARAK daha sıkı bir nested-CV olurdu ama bu görevin
# kapsamında YAPILMADI, bir sonraki incelemede sorgulanmalı.
RCS_AGE_KNOT_PERCENTILES: tuple[float, float, float] = (10.0, 50.0, 90.0)
CLINICAL_AGE_RCS1_COLUMN = "clinical_age_rcs1"
CLINICAL_AGE_RCS2_COLUMN = "clinical_age_rcs2"
PATIENT_RAW_AGE_COLUMN = "age"


def compute_rcs_knots(
    age_series: pd.Series, *, percentiles: tuple[float, float, float] = RCS_AGE_KNOT_PERCENTILES
) -> np.ndarray:
    """UPenn eğitim havuzunun HAM yaş dağılımından 3 düğüm (10/50/90
    persentil, varsayılan) hesaplar -- `restricted_cubic_spline_basis()`'e
    girdi. `age_series`'teki NaN'lar (varsa) `dropna()` ile persentil
    hesabından ÖNCE atılır (UPenn'de yaş NULL yok, canlı DB ile
    doğrulandı -- ama bu fonksiyon bunu VARSAYMAZ, savunmacı davranır).

    Düğümler KESİN ARTAN olmalı (t1 < t2 < t3) -- aksi halde
    `restricted_cubic_spline_basis()`'teki bölme işlemleri sıfıra
    bölünür/dejenere olur; bu fonksiyon bunu ValueError ile ÖNceden
    yakalar (küçük/dejenere bir örneklemde persentiller çakışabilir).
    """

    values = age_series.dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("age_series bos (tum degerler NaN) -- dugum hesaplanamaz.")
    knots = np.percentile(values, list(percentiles))
    if len(set(np.round(knots, 8).tolist())) != len(knots):
        raise ValueError(
            f"Hesaplanan dugumler KESIN ARTAN degil (cakisma var): {knots.tolist()} "
            "-- ornekleme cok kucuk/dejenere olabilir."
        )
    return knots


def restricted_cubic_spline_basis(age: pd.Series, knots: np.ndarray) -> pd.DataFrame:
    """Harrell (2015) 3-düğümlü restricted cubic spline bazisi.

    `knots` `compute_rcs_knots()`'un (ya da eşdeğer, ÖNCEDEN eğitim
    havuzundan hesaplanmış) çıktısı olmalı -- bu fonksiyonun KENDİSİ
    düğüm HESAPLAMAZ, salt bir DÖNÜŞÜMDÜR (sızıntı önleme: aynı `knots`
    hem eğitim hem UCSF'e AYNEN uygulanabilsin diye ayrıştırıldı).

    Döndürür: 2 kolonlu DataFrame (`CLINICAL_AGE_RCS1_COLUMN` = ham yaş
    [doğrusal terim], `CLINICAL_AGE_RCS2_COLUMN` = tek doğrusal-olmayan
    terim) -- `age.index` ile hizalı.

    Formül (k=3, j=1..k-2=1 -- yalnız j=t1 için):
      X1(x) = [(x-t1)+^3 - (x-t2)+^3*(t3-t1)/(t3-t2)
                        + (x-t3)+^3*(t2-t1)/(t3-t2)] / (t3-t1)^2
    (Harrell eş. 2.24, k-1'inci/k'ıncı düğüm t_{k-1}/t_k = t2/t3 ile
    normalize edilir; (t3-t1)^2 ile ölçek-kararlılığı için bölünür --
    R'nin `Hmisc::rcspline.eval(..., norm=2)` konvansiyonuyla TUTARLI.)

    x <= t1 için X1(x) == 0 (tüm pozitif-kısım terimleri sıfır) --
    yani spline t1'in ALTINDA TAM DOĞRUSAL (bu özellik testte
    doğrulanır). x > t3 için X1 doğrusal (restricted/natural sınır
    koşulu -- ikinci türev dış düğümlerin ÖTESİNDE sıfır).
    """

    if len(knots) != 3:
        raise ValueError(
            f"restricted_cubic_spline_basis SADECE 3 dugumlu RCS icin yazildi "
            f"(k-2=1 dogrusal-olmayan terim), alinan dugum sayisi: {len(knots)}."
        )
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
        {CLINICAL_AGE_RCS1_COLUMN: x, CLINICAL_AGE_RCS2_COLUMN: nonlinear_term},
        index=age.index,
    )


def build_variant_clinical_extra_columns(
    *, include_mgmt: bool, use_age_spline: bool
) -> list[str]:
    """Bir Model-v2 varyantının `extra_columns` (zorunlu/forced-in klinik
    kovaryat) LİSTESİNİ kurar -- radyomik elastic-net/stabilite seçimine
    HİÇ girmeyen kovaryatlar. `v1_referans` için bu, mevcut
    `PRIMARY_CLINICAL_EXTRA_COLUMNS` ile BİREBİR AYNI sırayı/kümeyi
    üretir (regresyon testiyle KİLİTLENDİ, bkz. tests)."""

    age_columns = (
        [CLINICAL_AGE_RCS1_COLUMN, CLINICAL_AGE_RCS2_COLUMN]
        if use_age_spline
        else [CLINICAL_AGE_COLUMN]
    )
    extra_columns = list(age_columns) + [
        CLINICAL_GENDER_MALE_COLUMN,
        CLINICAL_GTR_Y_COLUMN,
        CLINICAL_GTR_MISSING_COLUMN,
        CLINICAL_IDH_MUTANT_COLUMN,
        CLINICAL_IDH_MISSING_COLUMN,
    ]
    if include_mgmt:
        extra_columns += [CLINICAL_MGMT_METHYLATED_COLUMN, CLINICAL_MGMT_MISSING_COLUMN]
    return extra_columns


def build_variant_clinical_frame(
    patients_frame: pd.DataFrame,
    *,
    include_mgmt: bool,
    use_age_spline: bool,
    age_knots: np.ndarray | None,
) -> pd.DataFrame:
    """Bir Model-v2 varyantının TAM klinik kovaryat çerçevesini kurar --
    `build_clinical_covariate_frame()`'i (cinsiyet+GTR+IDH+[MGMT]) çağırır,
    `use_age_spline=True` ise `CLINICAL_AGE_COLUMN`'u (doğrusal) `age_knots`
    ile üretilmiş 2 spline kolonuyla DEĞİŞTİRİR.

    `age_knots` SADECE `use_age_spline=True` iken kullanılır ve ZORUNLUDUR
    -- burada YENİDEN hesaplanmaz (çağıran taraf `compute_rcs_knots()`'u
    UPenn eğitim havuzunda BİR KEZ çalıştırıp AYNI değeri hem UPenn hem
    UCSF çağrısına vermeli, bkz. `run_single_variant()`)."""

    clinical = build_clinical_covariate_frame(
        patients_frame,
        include_gender=True,
        include_gtr=True,
        include_idh=True,
        include_mgmt=include_mgmt,
    )
    if use_age_spline:
        if age_knots is None:
            raise ValueError(
                "use_age_spline=True icin age_knots ZORUNLU (UPenn'den "
                "onceden hesaplanmis olmali -- bkz. compute_rcs_knots())."
            )
        spline_basis = restricted_cubic_spline_basis(clinical[CLINICAL_AGE_COLUMN], age_knots)
        clinical = clinical.drop(columns=[CLINICAL_AGE_COLUMN]).join(spline_basis)
    return clinical


# =====================================================================
# 1.6) UCSF-PDGM klinik kovaryatları -- AYRI FONKSİYON YOK (2026-08-18
#      REVİZYON, Barış kararı (a))
# =====================================================================
#
# GEÇMİŞ (bu satırların İLK sürümü, aynı gün): burada bir `build_ucsf_
# clinical_covariate_frame()` fonksiyonu + UCSF'in HAM değer sözlüğünü
# (M/F, GTR/STR/biopsy, wildtype/mutasyon-alt-tipi) UPenn'in harmonize
# sözlüğüne çeviren `UCSF_EOR_TO_GTR_LABEL` gibi sabitler vardı.
#
# 🔴 REVİZYON GEREKÇESİ: `db-agent`'in UCSF onboarding tasarımı, `patients`
# tablosuna UCSF satırlarını **HAM DEĞİL, ZATEN HARMONİZE** yazacak şekilde
# kesinleşti (Barış'ın onayladığı (a) şıkkı) --
#   `gender`            : 'Male' / 'Female'      (UCSF ham 'M'/'F' DEĞİL)
#   `gtr_over90percent`  : 'Y' / 'N'              (UCSF ham 'GTR'/'STR'/
#                                                   'biopsy' DEĞİL)
#   `idh1_status`        : 'Wildtype' / 'Mutated' (UCSF ham 'wildtype'/
#                                                   'IDH1 p.R132H'/... DEĞİL)
#   `vital_status`        : 'DECEASED' / 'ALIVE'
# Eşleme artık TEK yerde -- DB yazım katmanında (`tools/onboard_ucsf_
# patients.py`, db-agent'in işi, bu görevin kapsamı DIŞINDA) -- yapılıyor.
# Gerekçe: mevcut 3 kaynak (UPenn/LUMIERE/TCGA) zaten harmonize; UCSF'i ham
# sözlükle yazmak, kaynak-farkında olmayan bir `WHERE gtr_over90percent=
# 'Y'` sorgusunun UCSF'i SESSİZCE atlamasına yol açardı (2026-08-14'ün
# `CENSORED` sessiz-düşme hatasıyla AYNI hata sınıfı).
#
# SONUÇ: `build_clinical_covariate_frame()` UCSF için de DOĞRUDAN
# kullanılır (aşağıda `run_primary_single_model()`'de) -- ayrı bir UCSF
# fonksiyonuna/sözlüğüne GEREK KALMADI, TEK mapping kaynağı (yukarıdaki
# fonksiyon) korunuyor, kod tekrarı yok. `_assert_recognized_categorical()`
# guard'ı DEĞİŞMEDEN çalışmaya devam ediyor: db-agent'in onboarding'i
# BOZULURSA (örn. ham UCSF değeri sızarsa -- 'M'/'GTR'/'wildtype' gibi)
# bu, `build_clinical_covariate_frame()`'in recognized-set kontrolünde
# SESSİZCE yanlış sonuç ÜRETMEZ, `UnrecognizedClinicalValueError` ile
# AÇIKÇA patlar.


class ExternalTestNotSupportedError(RuntimeError):
    """Bir kol harici test (TCGA) istiyor ama gereken kovaryat TCGA'da
    TAMAMEN eksik -- `evaluate_external_test()` sessizce NaN/anlamsiz
    bir sonuc URETMESIN diye burada DURDURULUR (gorev talimati: "net
    bir hata mesajiyla dursun; sessizce NaN uretmesin")."""


def assert_external_test_feasible(
    arm_name: str, tcga_clinical_frame: pd.DataFrame, required_columns: list[str]
) -> None:
    for column in required_columns:
        if column not in tcga_clinical_frame.columns:
            raise ExternalTestNotSupportedError(
                f"Kol {arm_name!r}: harici test icin gereken kolon "
                f"{column!r} TCGA klinik cercevesinde YOK."
            )
        coverage = float(tcga_clinical_frame[column].notna().mean())
        if coverage <= 0.0:
            raise ExternalTestNotSupportedError(
                f"Kol {arm_name!r}: harici test istendi ama TCGA'da "
                f"{column!r} kolonu TAMAMEN eksik (kapsam %0) -- "
                "CLAUDE.md kilitli olcumu: GTR/IDH1/MGMT TCGA'da hic "
                "yok (alan YOK). evaluate_external_test() sessizce "
                "NaN/anlamsiz bir sonuc uretmesin diye burada "
                "DURDURULDU -- bu kol icin run_external_test=False "
                "olmali (gorev talimati: 'yalniz ic CV')."
            )


# =====================================================================
# 2) Ozellik kolon listeleri (bolge-onekli)
# =====================================================================


def region_prefixed_columns(feature_names: Iterable[str], *, region: str) -> list[str]:
    return [f"{region}__{name}" for name in feature_names]


def select_feature_columns(
    wide_frame: pd.DataFrame, feature_names: Iterable[str], *, region: str
) -> list[str]:
    """`wide_frame`'de beklenen bolge-onekli kolonlarin HEPSININ var
    oldugunu dogrular -- eksikse sessizce alt-kume ALINMAZ, ValueError
    firlatilir (fail-loud ilkesi)."""

    expected = region_prefixed_columns(feature_names, region=region)
    missing = [col for col in expected if col not in wide_frame.columns]
    if missing:
        raise ValueError(
            f"wide_frame'de beklenen {len(missing)} kolon eksik (ornek: "
            f"{missing[:5]}) -- pivot cikisi beklenen 107-ozellik "
            "sozlesmesiyle UYUSMUYOR olabilir."
        )
    return expected


# =====================================================================
# 2.5) UCSF WT-only pivot -- pipeline.cox_model.pivot_radiomics_long_to_
#      wide()'DAN BAĞIMSIZ (2026-08-18, Barış kararı)
# =====================================================================
#
# NEDEN AYRI/BAĞIMSIZ (bu görevin KESİN SINIRI): `pivot_radiomics_long_
# to_wide()` her satırı `canonicalize_region_label()` üzerinden
# `pipeline.harmonization.canonical_source()`'a geçirir -- o fonksiyonun
# `SOURCE_ALIASES` sözlüğü şu an SADECE tcga/upenn/lumiere tanıyor,
# "UCSF"/"UCSF-PDGM" YOK. Bu görev talimatı `pipeline/harmonization.py`'ye
# DOKUNMAYI YASAKLIYOR (imaging-agent'in alanı) -- yani UCSF'i o ortak
# kanonikleştirme yoluyla geçirmek YAPISAL OLARAK MÜMKÜN DEĞİL bu görev
# kapsamında. Bu fonksiyon `pivot_radiomics_long_to_wide()`'ın WT-only
# alt kümesini `canonical_source()`'a HİÇ bağımlı olmadan yeniden uygular
# -- `TCGA harici test build_external_test_frame()`'in `_assemble_
# training_frame()`'den BAĞIMSIZ olması ile AYNI mimari desen (bkz. o
# fonksiyonun docstring'i).
#
# ⚠️ AÇIK RİSK (doğrulanmadı -- UCSF henüz DB'de yok): UCSF'in radiomics
# tablosundaki ham `tumor_region` değeri "WT_derived" mi (UPenn'in
# türetme konvansiyonuyla AYNI, decisions/2026-08-15-.../"7. Radyomik
# özellikler": "WT türetmesi UPenn'dekiyle aynı voksel-birleşimi
# kuralıyla yapılır") yoksa native "WT" mi (TCGA'nın konvansiyonu)
# BİLİNMİYOR. Varsayılan `UCSF_WT_RAW_REGION_LABELS` HER İKİSİNİ de kabul
# eder (birleşimlerinin BOŞ olmayacağını garanti eder) -- ama gerçek
# değer bunlardan FARKLIYSA (ör. "WT_UCSF") bu fonksiyon SESSİZCE 0 satır
# DÖNDÜRMEZ, aşağı akıştaki `check_external_test_pool_counts()` (0 hasta
# beklenen 295/169 ile uyuşmaz) bunu SERT durarak yakalar.
#
# DOĞRULANDI (2026-08-18, canlı DB SELECT, Model v2 görevi): UCSF C32
# verisi ARTIK DB'de (yukarıdaki "AÇIK RİSK" notu yazıldığı andan
# SONRA imaging-agent tarafından yüklendi) -- ham `tumor_region` gerçek
# değerleri `ED`/`ET`/`NC`/`TC_derived`/`WT_derived` (295 satır her biri),
# yani `WT_derived` varsayımı DOĞRU çıktı. `TC_derived` de aynı şekilde
# doğrulandı (UPenn 611 / UCSF 295 / LUMIERE 584 satır).
UCSF_WT_RAW_REGION_LABELS: tuple[str, ...] = ("WT_derived", "WT")
# 2026-08-18 EKLENDİ (Model v2, v2c_mgmt_spline_wttc varyantı -- WT+TC
# bölgesi). UPenn'deki TC türetme konvansiyonuyla TUTARLI (`TC_derived`).
TC_RAW_REGION_LABELS: tuple[str, ...] = ("TC_derived", "TC")

#: `pivot_ucsf_regions_long_to_wide()`'ın varsayılan kanonik-bölge ->
#: ham-etiket eşlemesi -- yalnız `regions` parametresinde İSTENEN
#: bölgeler bu sözlükten okunur (bilinmeyen bir bölge istenirse
#: ValueError, aşağıya bkz.).
UCSF_REGION_RAW_LABELS: dict[str, tuple[str, ...]] = {
    "WT": UCSF_WT_RAW_REGION_LABELS,
    "TC": TC_RAW_REGION_LABELS,
}


class UCSFRegionPivotError(RuntimeError):
    """UCSF bölge pivotunda çakışma (aynı hasta+bölge için birden fazla
    satır) -- `pivot_radiomics_long_to_wide()`'ın `RegionPivotError`'ıyla
    AYNI ilke, bağımsız bir istisna sınıfı (bu fonksiyon `cox_model`'in
    fonksiyonunu ÇAĞIRMIYOR, o yüzden onun istisnasını da paylaşmıyor)."""


def pivot_ucsf_regions_long_to_wide(
    long_frame: pd.DataFrame,
    *,
    regions: Iterable[str] = ("WT",),
    patient_col: str = "patient_id",
    region_col: str = "tumor_region",
    feature_dict_cols: tuple[str, str, str] = (
        "shape_features",
        "first_order_features",
        "texture_features",
    ),
    region_raw_labels: dict[str, tuple[str, ...]] = UCSF_REGION_RAW_LABELS,
) -> tuple[pd.DataFrame, RegionPivotReport]:
    """`pivot_ucsf_wt_long_to_wide()`'ın ÇOK-BÖLGELİ genellemesi (2026-08-18,
    Model v2 -- `v2c_mgmt_spline_wttc` varyantı WT+TC'nin İKİSİNİ birden
    gerektiriyor). `pivot_ucsf_wt_long_to_wide()` artık bu fonksiyonun
    `regions=("WT",)` ile DAR bir sarmalayıcısı -- eski davranış BİREBİR
    korunur, mevcut testler ETKİLENMEZ.

    `canonical_source()`'a BAĞIMLI DEĞİL (yukarıdaki bölüm başlığı
    notuna bkz. -- bu görev `pipeline/harmonization.py`'ye dokunmayı
    YASAKLIYOR). Çıktı kolon adları `pivot_radiomics_long_to_wide()` ile
    TUTARLI (`"{bolge}__<ozellik_adi>"`, örn. `"TC__original_shape_..."`).

    Bir hastanın istenen `regions`'DAN biri EKSİKSE hasta TÜMDEN
    çıktıdan düşürülür (WT+TC ikisi de gerekliyse, yalnız WT'si olan
    hasta da düşer) -- `pivot_radiomics_long_to_wide()`'ın
    `on_missing_region="drop"` davranışıyla TUTARLI, kaç hastanın hangi
    bölge eksikliğinden düştüğü `RegionPivotReport.dropped_patients_
    missing_region`'a bölge bazında SAYILIR (sessizce filtrelenmez).

    Bir hastanın aynı kanonik bölge için BİRDEN FAZLA ham-etiketli satırı
    varsa (`patient_col`, kanonik bölge bazında duplicate) -- ortalama
    alma/rastgele seçim YAPILMAZ, `UCSFRegionPivotError` fırlatılır.
    """

    regions = list(regions)
    if not regions:
        raise ValueError("regions bos olamaz.")
    unknown_regions = set(regions) - set(region_raw_labels)
    if unknown_regions:
        raise ValueError(
            f"region_raw_labels bu bolge(ler)i tanimiyor: {sorted(unknown_regions)} "
            f"-- bilinen: {sorted(region_raw_labels)}"
        )

    required_cols = {patient_col, region_col, *feature_dict_cols}
    missing_cols = required_cols - set(long_frame.columns)
    if missing_cols:
        raise ValueError(f"long_frame eksik kolon(lar): {sorted(missing_cols)}")

    n_input_rows = len(long_frame)
    all_patients_full = sorted(long_frame[patient_col].unique())
    n_candidate_patients = len(all_patients_full)

    # Ham etiketten kanonik bolgeye eslemeyi TERSINE cevir -- birden
    # fazla ham etiket (orn. "WT_derived" VE "WT") ayni kanonik bolgeye
    # dusebilir. Iki farkli kanonik bolge AYNI ham etiketi paylasamaz
    # (cakisma -- asagida acikca reddediliyor).
    raw_to_canonical: dict[str, str] = {}
    for canonical_region in regions:
        for raw_label in region_raw_labels[canonical_region]:
            existing = raw_to_canonical.get(raw_label)
            if existing is not None and existing != canonical_region:
                raise ValueError(
                    f"region_raw_labels CAKISMASI: ham etiket {raw_label!r} "
                    f"hem {existing!r} hem {canonical_region!r} kanonik "
                    "bolgesi icin tanimli -- iki bolge ayni ham etiketi "
                    "PAYLASAMAZ."
                )
            raw_to_canonical[raw_label] = canonical_region

    working = long_frame.copy()
    working["_canonical_region"] = working[region_col].map(raw_to_canonical)
    filtered = working.loc[working["_canonical_region"].notna()].copy()

    duplicate_mask = filtered.duplicated(subset=[patient_col, "_canonical_region"], keep=False)
    if duplicate_mask.any():
        duplicate_counts = (
            filtered.loc[duplicate_mask, "_canonical_region"].value_counts().to_dict()
        )
        raise UCSFRegionPivotError(
            "UCSF long_frame'de ayni (hasta, kanonik bolge) cifti icin "
            "birden fazla satir var -- pivot CAKISMASI, ortalama alma/"
            "rastgele secim YAPILMAZ. Cagiran taraf onceden "
            "tekillestirmeli (muhtemel neden: birden fazla "
            "segmentation_tool ayni hasta+bolge icin satir uretmis). "
            f"Bolge kirilimi: {duplicate_counts}"
        )

    groups = {pid: rows for pid, rows in filtered.groupby(patient_col)}
    wide_records: dict[str, dict[str, float]] = {}
    dropped_missing_region: dict[str, int] = {region: 0 for region in regions}

    for patient_id in all_patients_full:
        patient_rows = groups.get(patient_id)
        present_regions = (
            set(patient_rows["_canonical_region"]) if patient_rows is not None else set()
        )
        missing = [region for region in regions if region not in present_regions]
        if missing:
            for region in missing:
                dropped_missing_region[region] += 1
            continue

        patient_features: dict[str, float] = {}
        for _, row in patient_rows.iterrows():
            prefix = f"{row['_canonical_region']}__"
            for dict_col in feature_dict_cols:
                feature_dict = row[dict_col] or {}
                for feature_name, value in feature_dict.items():
                    patient_features[f"{prefix}{feature_name}"] = value
        wide_records[patient_id] = patient_features

    wide_frame = pd.DataFrame.from_dict(wide_records, orient="index")
    wide_frame.index.name = patient_col
    wide_frame = wide_frame.sort_index()

    report = RegionPivotReport(
        regions_requested=regions,
        n_input_rows=n_input_rows,
        n_candidate_patients=n_candidate_patients,
        n_output_patients=len(wide_records),
        dropped_patients_missing_region=dropped_missing_region,
    )
    return wide_frame, report


def pivot_ucsf_wt_long_to_wide(
    long_frame: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    region_col: str = "tumor_region",
    feature_dict_cols: tuple[str, str, str] = (
        "shape_features",
        "first_order_features",
        "texture_features",
    ),
    raw_region_labels: tuple[str, ...] = UCSF_WT_RAW_REGION_LABELS,
) -> tuple[pd.DataFrame, RegionPivotReport]:
    """UCSF `radiomics` uzun-formatını WT-only geniş matrise çevirir --
    `pivot_ucsf_regions_long_to_wide(regions=("WT",))`'ın DAR bir
    sarmalayıcısı (2026-08-18'de genellemeye REFAKTÖR edildi -- bu
    fonksiyonun İMZASI/DAVRANIŞI DEĞİŞMEDİ, mevcut çağıranlar/testler
    ETKİLENMEZ). Çıktı kolon adları `pivot_radiomics_long_to_wide()` ile
    TUTARLI (`"WT__<ozellik_adi>"`), `select_feature_columns()`/`STABLE_
    FEATURES_ICC60` doğrudan kullanılabilir.

    Bir hastanın `raw_region_labels`'tan BİRDEN FAZLA satırı varsa
    (`patient_col` bazında duplicate) -- ortalama alma/rastgele seçim
    YAPILMAZ, `UCSFRegionPivotError` fırlatılır (fail-loud, `pivot_
    radiomics_long_to_wide()` ile AYNI ilke).

    WT satırı olmayan hasta çıktıdan düşürülür, sayısı `RegionPivotReport.
    dropped_patients_missing_region["WT"]`'ye SAYILIR (sessizce
    filtrelenmez).
    """

    return pivot_ucsf_regions_long_to_wide(
        long_frame,
        regions=("WT",),
        patient_col=patient_col,
        region_col=region_col,
        feature_dict_cols=feature_dict_cols,
        region_raw_labels={"WT": raw_region_labels},
    )


# =====================================================================
# 3) TCGA harici test cercevesi -- `_assemble_training_frame()` KULLANILMAZ
# =====================================================================


@dataclass
class ExternalFrameReport:
    """`build_external_test_frame()` ciktisinin seffaf denetim izi."""

    n_input_rows: int
    n_output_rows: int
    dropped_missing_duration: int
    dropped_missing_features: int
    dropped_unrecognized_vital_status: dict[str, int]


def build_external_test_frame(
    feature_frame: pd.DataFrame,
    patients_frame: pd.DataFrame,
    *,
    duration_col: str = "survival_days",
    event_col: str = "event",
    vital_status_col: str = "vital_status",
    deceased_labels: Iterable[str] = ("DECEASED",),
    # "CENSORED" 2026-08-14'te EKLENDİ -- `pipeline/cox_model.py`'deki AYNI
    # düzeltmenin harici-test tarafındaki karşılığı (iki taraf tutarlı olmalı,
    # aksi halde eğitim ve test farklı sansür konvansiyonu kullanırdı).
    censored_labels: Iterable[str] = ("ALIVE", "CENSORED"),
    # 2026-08-15 EKLENDİ (klinik kovaryat desteği): `patients_frame`'den
    # (dummy-encoding YAPILMADAN) OLDUĞU GİBİ taşınacak kolonlar (örn.
    # `clinical_age`/`clinical_gender_male` -- `build_clinical_covariate_
    # frame()`'in çıktısı, çağıran tarafça `patients_frame`'e ÖNCEDEN
    # join'lenmiş olmalı). `pipeline.cox_model._assemble_training_frame()`'in
    # AYNI görev için eklenen `passthrough_columns` parametresiyle TUTARLI
    # isimlendirme -- eğitim/harici-test tarafında AYNI kavram. Varsayılan
    # `None` -- eski davranış BİREBİR korunur.
    passthrough_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, ExternalFrameReport]:
    """TCGA harici test cercevesini kurar.

    BILINCLI BAGIMSIZ UYGULAMA -- `pipeline.cox_model._assemble_training_
    frame()`'i CAGIRMAZ. Gorev talimati `build_training_frame()`'in Cox
    egitim cercevesi kurmak icin TEK resmi giris noktasi oldugunu ve
    `_assemble_training_frame()`'in bu script'ten DOGRUDAN CAGRILMAYACAGINI
    sart kosuyor. Ama `build_training_frame()` zaten TCGA'yi HER KOSULDA
    whitelist guard'inda REDDEDER (CLAUDE.md: "Cox PHM + XGBoost egitimi
    SADECE UPenn-GBM + LUMIERE ile yapilir") -- yani o fonksiyon TCGA
    harici test cercevesi icin ZATEN KULLANILAMAZ, bu bir gozden kacirma
    degil yapisal bir gercek. Bu fonksiyon, `_assemble_training_frame()`'in
    event/duration turetme mantiginin (source dummy encoding HARIC --
    egitim havuzu tek-kaynakli [UPenn] oldugu icin zaten hic source dummy
    kolonu uretilmiyor, bkz. bu modulun basindaki "COMBAT NOTU") gerekli
    alt kumesini BAGIMSIZ olarak yeniden uygular. Whitelist/TCGA-reddi
    burada KASITLI OLARAK YOK -- bu fonksiyonun TEK amaci TCGA'yi (izin
    verilen tek harici test kaynagi) `evaluate_external_test()`'e
    hazirlamaktir.

    TCGA'nin `vital_status` etiketleri de "DECEASED"/"ALIVE" (canli DB
    ile dogrulandi, decisions/2026-07-19-tcga-mgmt-idh1-survival-days-
    acik.md: 260 hastanin 259'u DECEASED(213)/ALIVE(47) enum'u) -- UPenn
    ile AYNI varsayilan etiketler kullanilabilir.
    """

    required = {duration_col, vital_status_col}
    missing_cols = required - set(patients_frame.columns)
    if missing_cols:
        raise ValueError(f"patients_frame eksik kolon(lar): {sorted(missing_cols)}")

    merged = feature_frame.join(patients_frame, how="inner")
    n_input_rows = len(merged)

    deceased_set = set(deceased_labels)
    censored_set = set(censored_labels)
    recognized = deceased_set | censored_set
    unrecognized_mask = ~merged[vital_status_col].isin(recognized)
    unrecognized_counts = (
        merged.loc[unrecognized_mask, vital_status_col]
        .fillna("<NULL>")
        .value_counts()
        .to_dict()
    )
    merged = merged.loc[~unrecognized_mask].copy()
    merged[event_col] = merged[vital_status_col].isin(deceased_set).astype(int)

    missing_duration_mask = merged[duration_col].isna()
    dropped_missing_duration = int(missing_duration_mask.sum())
    merged = merged.loc[~missing_duration_mask].copy()

    passthrough_columns = list(passthrough_columns or [])
    missing_passthrough_columns = set(passthrough_columns) - set(merged.columns)
    if missing_passthrough_columns:
        raise ValueError(
            f"patients_frame passthrough_columns icin eksik kolon(lar): "
            f"{sorted(missing_passthrough_columns)}"
        )

    feature_columns = list(feature_frame.columns)
    check_columns = feature_columns + passthrough_columns
    missing_feature_mask = merged[check_columns].isna().any(axis=1)
    dropped_missing_features = int(missing_feature_mask.sum())
    merged = merged.loc[~missing_feature_mask].copy()

    frame_parts = [merged[feature_columns]]
    if passthrough_columns:
        frame_parts.append(merged[passthrough_columns])
    frame_parts.append(merged[[duration_col, event_col]])
    frame = pd.concat(frame_parts, axis=1)

    report = ExternalFrameReport(
        n_input_rows=n_input_rows,
        n_output_rows=len(frame),
        dropped_missing_duration=dropped_missing_duration,
        dropped_missing_features=dropped_missing_features,
        dropped_unrecognized_vital_status=unrecognized_counts,
    )
    return frame, report


# =====================================================================
# 4) Egitim havuzu buyukluk kapisi (611/585 -- CLAUDE.md 2026-08-14)
# =====================================================================


@dataclass(frozen=True)
class DeclaredRegionShortfall:
    """WT-only (611/585) taban havuzundan, BELIRLI bir cok-bolgeli
    varyant icin ONCEDEN BEYAN EDILMIS ve GEREKCESI YAZILMIS azalma.

    Bu bir "kapiyi gevsetme" mekanizmasi DEGIL -- kapiya IKINCI bir
    beklenti tablosu eklemektir: beyan edilen dusus AYNEN olculmezse
    (sayi ya da KIMLIK farkliysa) kapi yine sert durur.
    """

    n_patients: int
    n_events: int
    dropped_patient_ids: tuple[str, ...]
    reason: str
    declared_on: str


#: Bolge kumesi -> beyan edilmis havuz azalmasi.
#:
#: NEDEN VAR (2026-08-18 gercek kosusu, gunluk-rapor-2026-08-18.md
#: bulgu 16 + Riskler-1): `v2c_mgmt_spline_wttc` varyanti WT VE TC'nin
#: IKISINI birden istiyor. UPenn'in 611 hastasinin 2'sinde `ET=0` **ve**
#: `NC=0`, yani `TC = ET + NC = 0` -- tumorleri saf odem, cekirdek yok.
#: Bu hastalarin 107 TC kolonunun HEPSI NaN olur ve
#: `build_training_frame()` onlari (dogru sekilde) dusurur -> havuz
#: 609/583 olur. Bu GERCEK VERI, hata DEGIL.
#:
#: Eski (bolge-kor) kapi bu mesru azalmayi da "beklenmedik degisim"
#: sayip kosuyu durduruyordu. Barış karari (2026-08-19): kapi
#: bolge-farkinda olsun -- WT-only'de 611/585 SERT beklensin,
#: cok-bolgeli varyantlarda BEYAN EDILMIS azalmaya izin verilsin ve
#: dusen hastalar ACIKCA loglansin.
#:
#: BU TABLO BIR "KAPIYI KAPATMA" ARACI DEGIL: burada bir bolge kumesi
#: YOKSA kapi yine sert durur -- yeni bir cok-bolgeli varyant, azalmasi
#: olculup gerekcesi yazilmadan sessizce gecemez.
DECLARED_REGION_SHORTFALLS: dict[tuple[str, ...], DeclaredRegionShortfall] = {
    ("WT", "TC"): DeclaredRegionShortfall(
        n_patients=609,
        n_events=583,
        dropped_patient_ids=("UPENN-GBM-00354", "UPENN-GBM-00397"),
        reason=(
            "TC = ET + NC; bu iki hastada canli DB'de ET=0 VE NC=0 "
            "(UPENN-GBM-00354: ED=12472, UPENN-GBM-00397: ED=27095, "
            "ikisinde de TC_derived=0) -> 107 TC kolonunun hepsi NaN. "
            "Tumor saf odem, cekirdek yok -- gercek veri bulgusu, "
            "cikarim/kod hatasi DEGIL."
        ),
        declared_on="2026-08-19",
    ),
}


def resolve_declared_region_shortfall(
    regions: Iterable[str] | None,
) -> DeclaredRegionShortfall | None:
    """`regions` icin beyan edilmis azalmayi dondurur.

    `None` / `("WT",)` -> `None` (taban havuz, azalma beyan EDILMEMIS,
    sert 611/585 beklenir). Bilinmeyen bir cok-bolgeli kume icin de
    `None` doner -- cagiran taraf bunu "beyan yok" olarak yorumlar ve
    kapi sert calisir (sessiz gecis YOK).
    """

    if regions is None:
        return None
    # `regions` bir generator ise `tuple()` onu TUKETIR; cagiran taraf
    # ayni degeri tekrar okumaya calisirsa BOS gorur (2026-08-19 Codex
    # LOW bulgusu #5). Bu yuzden `check_training_pool_counts()` donusumu
    # BIR KEZ yapip bu fonksiyona hazir tuple gecer; burada da savunmali
    # davraniliyor.
    key = tuple(regions)
    if key == ("WT",):
        return None
    return DECLARED_REGION_SHORTFALLS.get(key)


def all_dropped_patient_ids(report: Any) -> list[str]:
    """`TrainingFrameReport.dropped_patient_ids`'in TUM gerekcelerindeki
    hasta kimliklerinin sirali BIRLESIMI.

    2026-08-19 (Codex HIGH bulgusu #1): kapiya yalnizca
    `"missing_features"` gerekcesi gecmek YETMEZ. Havuz baska bir
    gerekcelerle (`missing_duration` / `missing_passthrough` /
    `unrecognized_vital_status`) de kuculebilir; o durumda
    `"missing_features"` anahtari HIC OLUSMAZ ve kapi kimlik kanitini
    `None` gorup dogrulamayi atlardi. Birlesimi gecmek iki sey saglar:
    (1) kanit HER ZAMAN vardir (havuz kuculduyse en az bir kimlik olur),
    (2) BEKLENMEDIK bir gerekce ile dusen hasta, beyan edilen kimlik
        listesiyle eslesmeyecegi icin kapi SERT durur.
    """

    mapping = getattr(report, "dropped_patient_ids", None) or {}
    merged: set[str] = set()
    for ids in mapping.values():
        merged.update(str(pid) for pid in ids)
    return sorted(merged)


def check_training_pool_counts(
    n_patients: int,
    n_events: int,
    *,
    expected_patients: int,
    expected_events: int,
    allow_mismatch: bool = False,
    regions: Iterable[str] | None = None,
    dropped_patient_ids: Iterable[str] | None = None,
) -> None:
    """Egitim havuzu buyukluk kapisi -- 2026-08-19'da BOLGE-FARKINDA yapildi.

    `regions=None` (varsayilan) -> davranis BIREBIR ESKISI GIBI: sert
    `expected_patients`/`expected_events` esitligi. Mevcut cagiranlar ve
    testler ETKILENMEZ.

    `regions` verilirse:
      * `("WT",)`  -> yine SERT 611/585 (taban havuz, azalma beklenmiyor).
      * cok-bolgeli VE `DECLARED_REGION_SHORTFALLS`'ta beyani VAR ->
        beyan edilen sayi/kimlikle BIREBIR eslesirse gecer (dusen
        hastalar stderr'e ACIKCA loglanir), eslesmezse SERT durur.
      * cok-bolgeli VE beyani YOK -> SERT durur.

    `dropped_patient_ids` (`TrainingFrameReport.dropped_patient_ids`'in
    `"missing_features"` girdisi) verilirse KIMLIK de dogrulanir: sayi
    tutup kimlik tutmuyorsa -- yani BASKA hastalar dustuyse -- bu
    sessizce gecmez. Sayi esitligi tek basina yeterli bir kanit degildir;
    kapinin amaci "beklenmedik SESSIZ degisimi yakalamak".
    """

    # `regions` BIR KEZ tuple'a cevrilir -- generator gecilirse ikinci
    # okuma tukenmis olurdu (2026-08-19 Codex LOW bulgusu #5: log
    # etiketi bos cikiyordu).
    regions_tuple = tuple(regions) if regions is not None else None

    observed_dropped = (
        tuple(sorted(str(pid) for pid in dropped_patient_ids))
        if dropped_patient_ids is not None
        else None
    )

    if n_patients == expected_patients and n_events == expected_events:
        return

    shortfall = resolve_declared_region_shortfall(regions_tuple)
    if shortfall is not None:
        region_label = "+".join(regions_tuple)  # type: ignore[arg-type]
        counts_match = n_patients == shortfall.n_patients and n_events == shortfall.n_events
        declared_ids = tuple(sorted(shortfall.dropped_patient_ids))

        # 2026-08-19 DUZELTMESI (Codex capraz incelemesi, HIGH #1 --
        # FAIL-OPEN). Onceki surum `ids_match = observed_dropped is None
        # or observed_dropped == declared_ids` diyordu. Yani KIMLIK
        # KANITI VERILMEZSE kontrol OTOMATIK BASARILI sayiliyordu.
        #
        # Bu teorik bir acik degildi: uc uretim cagrisi da
        # `report.dropped_patient_ids.get("missing_features")` geciyor ve
        # `.get()` anahtar YOKSA `None` doner. Havuz 609/583'e BASKA bir
        # gerekcelerle (orn. `missing_duration` / `missing_passthrough`)
        # duserse "missing_features" anahtari HIC OLUSMAZ -> `None` ->
        # kimlik hic dogrulanmadan kapi GECER. Kapinin en yeni korumasi
        # tam da o senaryoda kaybolurdu.
        #
        # Artik beyan edilmis azalma dalinda KIMLIK KANITI ZORUNLU.
        # KAPI SIRASI (bilincli): once SAYI, sonra KANIT VARLIGI, sonra
        # KIMLIK. Sayi zaten tutmuyorken "kimlik kaniti yok" demek daha
        # az bilgilendirici olurdu -- cagiran once sayi farkini gorsun.
        if not counts_match:
            raise TrainingPoolCountMismatchError(
                f"Bolge kumesi {region_label} icin BEYAN EDILEN havuz "
                f"({shortfall.n_patients}/{shortfall.n_events}, beyan tarihi "
                f"{shortfall.declared_on}) ile OLCULEN ({n_patients}/{n_events}) "
                "UYUSMUYOR. Beyan edilmis azalma disinda bir degisim var -- "
                f"SESSIZCE devam EDILMEMELI. Beyanin gerekcesi: {shortfall.reason}"
            )

        if observed_dropped is None:
            raise TrainingPoolCountMismatchError(
                f"Bolge kumesi {region_label} icin beyan edilmis bir azalma "
                f"({shortfall.n_patients}/{shortfall.n_events}) VAR ve sayilar "
                "TUTUYOR, ama cagiran taraf DUSEN HASTA KIMLIKLERINI "
                "SAGLAMADI. Sayi esitligi tek basina yeterli KANIT DEGILDIR "
                "(ayni sayida ama BASKA hastalar dusmus olabilir) -- kapi bu "
                "durumda GECIRMEZ. Cagiran, `TrainingFrameReport."
                "dropped_patient_ids` icindeki TUM gerekcelerin birlesimini "
                "gecmelidir (bkz. `all_dropped_patient_ids()`)."
            )

        if observed_dropped == declared_ids:
            dropped_line = ", ".join(declared_ids)
            print(
                "BEYAN EDILMIS HAVUZ AZALMASI (bolge-farkinda kapi, beyan "
                f"tarihi {shortfall.declared_on}):"
                f"\n  bolgeler       : {region_label}"
                f"\n  taban havuz    : {expected_patients} hasta / "
                f"{expected_events} olay (WT-only)"
                f"\n  olculen havuz  : {n_patients} hasta / {n_events} olay"
                f"\n  dusen hasta ({len(declared_ids)}): {dropped_line}"
                f"\n  gerekce        : {shortfall.reason}"
                f"\n  NOT: bu varyantin raporunda egitim havuzu "
                f"{n_patients}/{n_events} olarak BEYAN EDILMELI, "
                f"{expected_patients}/{expected_events} DEGIL.",
                file=sys.stderr,
                flush=True,
            )
            return

        raise TrainingPoolCountMismatchError(
            f"Bolge kumesi {region_label} icin beyan edilen azalma "
            f"({shortfall.n_patients}/{shortfall.n_events}) SAYICA tutuyor "
            f"ama DUSEN HASTALAR FARKLI -- beyan: {list(declared_ids)}, "
            f"olculen: {list(observed_dropped)}. Ayni sayida ama BASKA "
            "hastalarin dusmesi, beyanin artik gecerli olmadigi anlamina "
            "gelir (veri degismis olabilir) -- SESSIZCE devam EDILMEMELI."
        )

    message = (
        f"Olculen egitim havuzu ({n_patients} hasta / {n_events} olay) "
        f"beklenen ({expected_patients}/{expected_events}, CLAUDE.md "
        "2026-08-14 duzeltmesi) ile UYUSMUYOR. Bu ya (a) C32 cikarimi "
        "beklenenden farkli sayida hasta uretti, ya da (b) beklenen "
        "sabit artik gecersiz -- her iki durumda da SESSIZCE devam "
        "EDILMEMELI. Gercek fark C32 cikarimi tamamlaninca dogrulanmali."
    )
    if regions_tuple is not None and regions_tuple != ("WT",):
        message += (
            f" Bolge kumesi {'+'.join(regions_tuple)} icin "
            "DECLARED_REGION_SHORTFALLS'ta BEYAN EDILMIS bir azalma YOK -- "
            "cok-bolgeli bir varyantin havuz azalmasi once olculup "
            "gerekcesiyle beyan edilmeli, ancak ondan sonra gecebilir."
        )
    if observed_dropped:
        message += f" Dusen hastalar (missing_features): {list(observed_dropped)}."
    if allow_mismatch:
        print(f"UYARI (--allow-unexpected-patient-count ile gecildi): {message}", file=sys.stderr)
        return
    raise TrainingPoolCountMismatchError(message)


class ExternalTestCountMismatchError(RuntimeError):
    """Olculen harici test (UCSF/TCGA) hasta/olay sayisi beklenenle
    UYUSMUYOR. `check_training_pool_counts()`'un harici-test karsiligi --
    ayri istisna sinifi, cagiranin egitim havuzu mu yoksa harici test mi
    diye kaynagi ayirt edebilmesi icin (fail-loud, sessiz devam yok)."""


def check_external_test_pool_counts(
    n_patients: int,
    n_events: int,
    *,
    expected_patients: int,
    expected_events: int,
    allow_mismatch: bool = False,
    label: str = "harici test",
) -> None:
    """`check_training_pool_counts()` ile AYNI "loud gate" deseni, harici
    test havuzu icin genellestirildi (2026-08-18, UCSF destegi -- gorev
    talimati: "imaging-agent'in dondurdugu liste otorite, sayi tutmazsa
    DUR ve bildir; sessizce devam ETME")."""

    if n_patients == expected_patients and n_events == expected_events:
        return
    message = (
        f"Olculen {label} havuzu ({n_patients} hasta / {n_events} olay) "
        f"beklenen ({expected_patients}/{expected_events}) ile UYUSMUYOR. "
        "Bu ya (a) C32 cikarimi/kohort filtresi beklenenden farkli sayida "
        "hasta uretti, ya da (b) beklenen sabit (295/169, decisions/"
        "2026-08-18-tek-model-ucsf-harici-test-k15-kapanisi.md ile "
        "kilitli) artik gecersiz -- her iki durumda da SESSIZCE devam "
        "EDILMEMELI."
    )
    if allow_mismatch:
        print(f"UYARI (--allow-unexpected-ucsf-count ile gecildi): {message}", file=sys.stderr)
        return
    raise ExternalTestCountMismatchError(message)


# =====================================================================
# 5) Full-pool final model -- TCGA'ya karsi degerlendirilecek TEK model
# =====================================================================


@dataclass
class FinalModelResult:
    """Tum UPenn egitim havuzunda (dis test bolmesi YOK) fit edilen TEK
    final modelin ciktisi -- harici (TCGA) degerlendirme icin."""

    best_penalizer: float
    best_l1_ratio: float
    inner_mean_c_index: float
    n_selected_elastic_net: int
    final_features: list[str]
    used_fallback: bool
    selection_frequency: pd.Series
    fitted_model: Any


def fit_final_model_on_full_pool(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    duration_col: str = "survival_days",
    event_col: str = "event",
    extra_columns: list[str] | None = None,
    inner_splits: int = 5,
    l1_ratio_grid: tuple[float, ...] = DEFAULT_L1_RATIO_GRID,
    penalizer_grid: tuple[float, ...] = DEFAULT_PENALIZER_GRID,
    n_bootstrap_stability: int = 200,
    stability_frequency_threshold: float = DEFAULT_STABILITY_FREQUENCY_THRESHOLD,
    seed: int = 42,
    extra_column_penalizer: float | None = None,
) -> FinalModelResult:
    """`run_nested_cv()`'nin dis-fold dongusunun (ic-CV grid taramasi ->
    elastic-net fit -> bootstrap stabilite secimi -> final fit) AYNI
    mantigini TUM egitim havuzunda (dis test bolmesi OLMADAN) bir kez
    calistirir -- TEK bir dagitilabilir model uretir.

    NEDEN GEREKLI: `run_nested_cv()` yalniz durust bir IC performans
    TAHMINI verir (5 ayri dis-fold modeli, hicbiri tum havuzda fit
    edilmedi) -- TCGA'ya karsi degerlendirilecek TEK bir final model
    URETMEZ. Bu fonksiyon, `run_nested_cv()`'nin KULLANDIGI AYNI private
    yardimcilari (`_score_hyperparameters_via_inner_cv`,
    `select_features_lasso`, `_bootstrap_stability_selection`) TAM
    havuz uzerinde tekrar kullanir -- `pipeline/cox_model.py`
    DEGISTIRILMEDI.

    TCGA'YA DOKUNULMAZ: bu fonksiyon SADECE `training_frame` (UPenn)
    kullanir -- kilitli kural ("TCGA'ya survival_days ile YENI ozellik
    secimi YASAK") burada ihlal EDILMIYOR, TCGA bu adimin hicbir
    yerinde gorulmuyor.

    `extra_column_penalizer` (2026-08-15 EKLENDİ, klinik kovaryat
    desteği): `None` (varsayılan) -- eski davranış BİREBİR korunur.
    Verilirse `extra_columns` (klinik kovaryatlar) `feature_columns`'DAN
    FARKLI (genelde zayıf/sıfır) ceza ile fit edilir -- bkz.
    `pipeline.cox_model._build_penalizer_argument()`.

    BOŞ RADYOMİK HAVUZ (2026-08-15 EKLENDİ, `clinical_base` kolu için):
    `feature_columns=[]` verilebilir (yalnız klinik kovaryatlardan
    oluşan, radyomik adayı OLMAYAN bir model) -- bu durumda
    `final_features` HER ZAMAN boş olur (radyomik seçilecek bir şey
    yok, bu bir HATA değil, beklenen davranış), "boş model" kontrolü
    `final_features`'IN DEĞİL, modelin TAMAMININ (final_features VEYA
    extra_columns) boş olup olmadığına bakar.
    """

    extra_columns = extra_columns or []

    best_score = float("-inf")
    best_penalizer = penalizer_grid[0]
    best_l1_ratio = l1_ratio_grid[0]
    any_valid = False
    for penalizer in penalizer_grid:
        for l1_ratio in l1_ratio_grid:
            score = _score_hyperparameters_via_inner_cv(
                training_frame,
                feature_columns,
                duration_col=duration_col,
                event_col=event_col,
                extra_columns=extra_columns,
                penalizer=penalizer,
                l1_ratio=l1_ratio,
                inner_splits=inner_splits,
                seed=seed,
                extra_column_penalizer=extra_column_penalizer,
            )
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_penalizer = penalizer
                best_l1_ratio = l1_ratio
                any_valid = True

    if not any_valid:
        raise RuntimeError(
            "Full-pool final model: ic CV grid taramasindaki HICBIR "
            "(penalizer, l1_ratio) kombinasyonu gecerli bir skor uretmedi."
        )

    en_selected, _, _ = select_features_lasso(
        training_frame,
        feature_columns,
        duration_col=duration_col,
        event_col=event_col,
        extra_columns=extra_columns,
        penalizer=best_penalizer,
        l1_ratio=best_l1_ratio,
        extra_column_penalizer=extra_column_penalizer,
    )

    stable_features, frequency = _bootstrap_stability_selection(
        training_frame,
        feature_columns,
        duration_col=duration_col,
        event_col=event_col,
        extra_columns=extra_columns,
        penalizer=best_penalizer,
        l1_ratio=best_l1_ratio,
        n_bootstrap=n_bootstrap_stability,
        frequency_threshold=stability_frequency_threshold,
        seed=seed,
        extra_column_penalizer=extra_column_penalizer,
    )
    used_fallback = not stable_features
    final_features = stable_features if stable_features else en_selected

    # 2026-08-15 GEVŞETİLDİ (klinik kovaryat desteği, `clinical_base`
    # kolu -- feature_columns=[] olabilir, bu durumda final_features
    # HER ZAMAN boş, HATA DEĞİL): eskiden yalnız `final_features` boşsa
    # hata veriyordu -- şimdi model TAMAMEN boşsa (ne radyomik seçim NE
    # klinik kovaryat) hata verir. `extra_columns=[]` olan TÜM eski
    # çağıranlarda (mevcut 3 birincil/duyarlılık kolu) davranış AYNI --
    # `final_features` boşsa `extra_columns` de boş olduğu için hata
    # yine fırlar.
    if not final_features and not extra_columns:
        raise RuntimeError(
            "Full-pool final model: elastic-net/stabilite secimi hicbir "
            "ozellik birakmadi (ve extra_columns/klinik kovaryat da yok)."
        )

    from lifelines import CoxPHFitter

    final_model_columns = final_features + extra_columns
    final_penalizer_argument = _build_penalizer_argument(
        final_features,
        extra_columns,
        penalizer=best_penalizer,
        extra_column_penalizer=extra_column_penalizer,
    )
    cox_final = CoxPHFitter(penalizer=final_penalizer_argument, l1_ratio=best_l1_ratio)
    cox_final.fit(
        training_frame[final_model_columns + [duration_col, event_col]],
        duration_col=duration_col,
        event_col=event_col,
    )

    return FinalModelResult(
        best_penalizer=best_penalizer,
        best_l1_ratio=best_l1_ratio,
        inner_mean_c_index=best_score,
        n_selected_elastic_net=len(en_selected),
        final_features=final_features,
        used_fallback=used_fallback,
        selection_frequency=frequency,
        fitted_model=cox_final,
    )


# =====================================================================
# 6) Nested-CV fallback + secilme-frekansi audit'i (Codex MEDIUM bulgusu)
# =====================================================================


def audit_nested_cv_fold_selection(
    frame: pd.DataFrame,
    feature_columns: list[str],
    nested_cv_result: pd.DataFrame,
    *,
    duration_col: str = "survival_days",
    event_col: str = "event",
    extra_columns: list[str] | None = None,
    outer_splits: int = 5,
    seed: int = 42,
    n_bootstrap_stability: int = 200,
    stability_frequency_threshold: float = DEFAULT_STABILITY_FREQUENCY_THRESHOLD,
    extra_column_penalizer: float | None = None,
    clinical_standardize_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`run_nested_cv()`'nin GORUNMEYEN iki bilgisini (fallback bayragi +
    ozellik-basina bootstrap secilme frekansi) uretir.

    NEDEN GEREKLI: `NestedCVFoldResult` bir fold'da stabilite kumesinin
    BOS kalip elastic-net'e mi dustugunu (fallback) YA DA her ozelligin
    bootstrap secilme FREKANSINI (yalniz esik-sonrasi son liste)
    raporlamiyor -- Codex'in 2026-08-14 MEDIUM bulgusu (decisions/
    2026-08-14-stabilite-secimi-esigi.md: "stabilite kumesi bos kalirsa
    pipeline elastic-net'e dusuyor -- yuksek esigin basarisizligini
    MASKELEYEBILIR. Hafta 3'te her esik icin fallback sayisi ayrica
    raporlanmali").

    YONTEM: `run_nested_cv()`'nin KENDI dis-fold bolunmesini (`StratifiedKFold
    (n_splits=outer_splits, shuffle=True, random_state=seed)` -- BIREBIR
    AYNI cagri) yeniden uretir ve `run_nested_cv()`'nin ZATEN HESAPLAYIP
    dondurdugu `best_penalizer`/`best_l1_ratio`'yu (YENIDEN TARANMAZ,
    `nested_cv_result`'tan okunur) kullanarak `_bootstrap_stability_
    selection()`'i `seed=seed+fold_index` ile (run_nested_cv()'nin ic
    cagrisiyla BIREBIR AYNI imza) tekrar calistirir.

    OZ-DOGRULAMA: her fold icin reconstruct edilen final kume buyuklugu
    `nested_cv_result`'in `n_selected_features_final` sutunuyla
    KARSILASTIRILIR. Uyusmazsa `FallbackAuditMismatchError` -- audit'in
    run_nested_cv() ile senkron OLMADIGININ kaniti, boyle bir durumda
    sessizce yanlis bir fallback/frekans raporu URETILMEZ.

    OZ-DOGRULAMA 2 (2026-09-13 EKLENDI -- Codex HIGH 4'un acik kalan
    maddesi): sayi karsilastirmasi TEK BASINA YETMEZ -- AYNI BOYUTTA
    ama FARKLI UYELERDEN olusan bir kume sessizce gecerdi. Bu yuzden
    `nested_cv_result`'in `selected_features` sutunuyla KUME (uyelik)
    karsilastirmasi da yapilir; uyusmazsa hata mesaji hangi ozelliklerin
    farkli oldugunu (iki yonlu fark) YAZAR. Fallback dalinda (stabilite
    kumesi bos -> elastic-net listesi) `nested_cv_result` yalniz SAYIYI
    tasidigindan, audit elastic-net secimini `select_features_lasso()`
    ile (run_nested_cv()'nin cagrisiyla BIREBIR AYNI imza, deterministik)
    yeniden uretip kumeyi oyle karsilastirir. Mevcut sayi kontrolu
    KALDIRILMADI -- kume kontrolu EK bir katmandir.

    SINIR (durustce belirtiliyor): bu, run_nested_cv()'nin zaten
    calistirdigi bootstrap stabilite secimini AUDIT icin TEKRAR
    calistirir -- pahali ama outer_splits kadar (varsayilan 5) tekrar,
    kabul edilebilir maliyet.

    `extra_column_penalizer`/`clinical_standardize_columns` (2026-08-15
    EKLENDİ, klinik kovaryat desteği): `run_nested_cv()`'ye VERİLEN
    AYNI değerlerle çağrılmalı -- aksi halde bu audit'in reconstruct
    ettiği `train_frame` (özellikle `clinical_standardize_columns`
    standardizasyonundan SONRA) `run_nested_cv()`'nin dış-fold içinde
    GERÇEKTEN kullandığı `train_frame`'den FARKLI olur ve öz-doğrulama
    (`FallbackAuditMismatchError`) YANLIŞ bir uyuşmazlık raporlayabilir
    (ya da -- daha kötüsü -- tesadüfen aynı sayıda özellik seçilirse
    SESSİZCE yanlış bir frekans tablosu üretilebilir). Standardizasyon
    `run_nested_cv()` ile BİREBİR AYNI noktada (dış-fold train
    alt-kümesi ayrıldıktan HEMEN SONRA, test alt-kümesi olmadan --
    burada ayrı bir test kümesi kavramı yok) uygulanır.
    """

    from sklearn.model_selection import StratifiedKFold

    extra_columns = extra_columns or []
    clinical_standardize_columns = clinical_standardize_columns or []
    splitter = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    fallback_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []

    for fold_index, (train_idx, _test_idx) in enumerate(
        splitter.split(frame, frame[event_col])
    ):
        train_frame = frame.iloc[train_idx]
        train_frame, _ = standardize_columns_fold_safe(
            train_frame, None, clinical_standardize_columns
        )
        matches = nested_cv_result.loc[nested_cv_result["fold"] == fold_index]
        if matches.empty:
            raise FallbackAuditMismatchError(
                f"Fold {fold_index}: nested_cv_result icinde bulunamadi -- "
                "outer_splits/seed audit ile run_nested_cv() cagrisi "
                "arasinda UYUSMUYOR olabilir."
            )
        fold_row = matches.iloc[0]
        best_penalizer = float(fold_row["best_penalizer"])
        best_l1_ratio = float(fold_row["best_l1_ratio"])

        stable_features, frequency = _bootstrap_stability_selection(
            train_frame,
            feature_columns,
            duration_col=duration_col,
            event_col=event_col,
            extra_columns=extra_columns,
            penalizer=best_penalizer,
            l1_ratio=best_l1_ratio,
            n_bootstrap=n_bootstrap_stability,
            frequency_threshold=stability_frequency_threshold,
            seed=seed + fold_index,
            extra_column_penalizer=extra_column_penalizer,
        )
        used_fallback = len(stable_features) == 0
        reconstructed_final_count = (
            int(fold_row["n_selected_features_elastic_net"])
            if used_fallback
            else len(stable_features)
        )
        reported_final_count = int(fold_row["n_selected_features_final"])
        if reconstructed_final_count != reported_final_count:
            raise FallbackAuditMismatchError(
                f"Fold {fold_index}: fallback audit reconstruct edilen final "
                f"ozellik sayisi ({reconstructed_final_count}) run_nested_cv()'nin "
                f"raporladigi n_selected_features_final ({reported_final_count}) "
                "ile UYUSMUYOR -- audit run_nested_cv() ile senkron degil, "
                "sessizce yanlis fallback/frekans raporu URETILMEDI."
            )

        # === KUME (SET) KONTROLU -- 2026-09-13, Codex HIGH 4'un kapanisi ===
        # NEDEN: yukaridaki kontrol YALNIZ `int` vs `int` karsilastiriyordu.
        # AYNI BOYUTTA ama FARKLI ozellikler iceren bir kume SESSIZCE
        # geciyordu ve `week3_*_selection_frequency.csv` YANLIS bir fold'un
        # frekanslariyla dolduruluyordu (audit, run_nested_cv()'nin fold'unu
        # reconstruct ettigini SANIYOR ama baska bir kume uretmis oluyor).
        # `selected_features` listesi `nested_cv_result`'ta ZATEN var
        # (`pipeline/cox_model.py` NestedCVFoldResult.selected_features),
        # dolayisiyla karsilastirma uye-duzeyinde YAPILABILIR.
        # SIRA MUHIM DEGIL (secim sirasi bir protokol ciktisi degil), bu
        # yuzden `set` karsilastirmasi -- ama tekrar/duplikasyon da
        # anlamsiz oldugundan uzunluk zaten yukarida kontrol ediliyor.
        if "selected_features" not in nested_cv_result.columns:
            raise FallbackAuditMismatchError(
                f"Fold {fold_index}: `nested_cv_result` icinde "
                "`selected_features` kolonu YOK -- kume-duzeyi audit "
                "yapilamaz. Bu kolon `run_nested_cv()`'nin standart "
                "ciktisidir; eksikse audit'e verilen cerceve "
                "run_nested_cv() ciktisi DEGILDIR (veya CSV'den geri "
                "okunmustur -- audit in-memory cerceve ile cagrilmali)."
            )
        reported_features = fold_row["selected_features"]
        if not isinstance(reported_features, (list, tuple, set, pd.Series, np.ndarray)):
            raise FallbackAuditMismatchError(
                f"Fold {fold_index}: `selected_features` bir liste degil "
                f"({type(reported_features).__name__}) -- CSV'den geri okunmus "
                "bir cerceve (string) olabilir; audit in-memory "
                "`run_nested_cv()` ciktisi ile cagrilmali."
            )
        reported_feature_set = set(reported_features)

        if used_fallback:
            # FALLBACK DALI: stabilite kumesi bos kaldi -> run_nested_cv()
            # elastic-net listesine dustu. `nested_cv_result` bu listenin
            # yalniz SAYISINI (`n_selected_features_elastic_net`) tasidigi
            # icin audit, elastic-net secimini KENDISI yeniden uretmek
            # zorunda -- `run_nested_cv()`'nin cagrisiyla BIREBIR AYNI
            # imza (deterministik, seed gerektirmez).
            reconstructed_en_features, _, _ = select_features_lasso(
                train_frame,
                feature_columns,
                duration_col=duration_col,
                event_col=event_col,
                extra_columns=extra_columns,
                penalizer=best_penalizer,
                l1_ratio=best_l1_ratio,
                extra_column_penalizer=extra_column_penalizer,
            )
            reconstructed_feature_set = set(reconstructed_en_features)
            branch_label = "fallback (elastic-net listesi)"
        else:
            reconstructed_feature_set = set(stable_features)
            branch_label = "stabilite kumesi"

        if reconstructed_feature_set != reported_feature_set:
            only_reconstructed = sorted(reconstructed_feature_set - reported_feature_set)
            only_reported = sorted(reported_feature_set - reconstructed_feature_set)
            raise FallbackAuditMismatchError(
                f"Fold {fold_index}: fallback audit reconstruct edilen final "
                f"ozellik KUMESI ({branch_label}) run_nested_cv()'nin "
                "raporladigi `selected_features` kumesiyle UYUSMUYOR -- "
                "sayilar esit olsa bile UYELER FARKLI, bu yuzden uretilecek "
                "secim-frekansi tablosu YANLIS fold'u temsil ederdi "
                "(sessizce uretilmedi).\n"
                f"  yalniz audit'te : {only_reconstructed}\n"
                f"  yalniz raporda  : {only_reported}\n"
                f"  audit kume buyuklugu={len(reconstructed_feature_set)} "
                f"rapor kume buyuklugu={len(reported_feature_set)}"
            )

        fallback_rows.append({"fold": fold_index, "used_fallback": used_fallback})
        for feature_name, freq_value in frequency.items():
            frequency_rows.append(
                {
                    "fold": fold_index,
                    "feature": feature_name,
                    "selection_frequency": float(freq_value),
                }
            )

    fallback_frame = pd.DataFrame(fallback_rows)
    frequency_frame = pd.DataFrame(frequency_rows)
    return fallback_frame, frequency_frame


def summarize_selection_frequency(frequency_frame: pd.DataFrame) -> pd.DataFrame:
    """Fold'lar arasi ozellik-basina bootstrap secilme frekansi ozeti.

    RAPORLAMA KURALI (decisions/2026-08-14-stabilite-secimi-esigi.md +
    concepts/istatistik-kavramlari-sozlugu.md Bolum 5): TEK bir "secilen
    ozellik listesi" SUNULMAZ -- bu tablo secilme frekansini TAM SAYIYLA
    raporlar (ornek: "Sphericity bootstrap'larin %78'inde secildi").
    """

    if frequency_frame.empty:
        return pd.DataFrame(
            columns=["feature", "mean_frequency", "min_frequency", "max_frequency", "n_folds"]
        )
    summary = (
        frequency_frame.groupby("feature")["selection_frequency"]
        .agg(mean_frequency="mean", min_frequency="min", max_frequency="max", n_folds="count")
        .reset_index()
        .sort_values("mean_frequency", ascending=False)
        .reset_index(drop=True)
    )
    return summary


# =====================================================================
# 7) Kesifsel ek -- korelasyon kumesi duzeyinde secim frekansi
#    (Codex onerisi, varsayilan KAPALI -- --exploratory-cluster-stability)
# =====================================================================


def compute_correlation_clusters(
    training_frame: pd.DataFrame, feature_columns: list[str], *, corr_threshold: float = 0.7
) -> dict[str, int]:
    """Egitim verisi ICINDE (yalniz egitim havuzu, TCGA'ya HIC dokunulmaz)
    93x93 (ya da 107x107) korelasyon matrisinden basit union-find ile
    korelasyon kumeleri cikarir -- |r| >= corr_threshold olan her ciftin
    ayni kumeye dustugu bir grafik baglantililigi.

    KESIFSEL, birincil secimin YERINI ALMAZ (Codex'in onerisi, kabul
    edildi -- decisions/2026-08-14-stabilite-secimi-esigi.md "Codex'in
    onerdigi UCUNCU secenek"). `corr_threshold=0.7` bu script'in
    yazarinin KOYDUGU bir varsayilan -- protokolde KILITLENMEMIS, bkz.
    bu dosyanin "Baris'a soru" notu (rapor).
    """

    corr = training_frame[feature_columns].corr().abs()
    n = len(feature_columns)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if corr.iloc[i, j] >= corr_threshold:
                union(i, j)

    roots = [find(i) for i in range(n)]
    unique_roots = sorted(set(roots))
    remap = {root: cluster_id for cluster_id, root in enumerate(unique_roots)}
    return {feature_columns[i]: remap[roots[i]] for i in range(n)}


def exploratory_cluster_selection_frequency(
    clusters: dict[str, int], nested_cv_result: pd.DataFrame
) -> pd.DataFrame:
    """Her korelasyon kumesi icin "kumeden en az bir ozellik secildi mi"
    frekansi -- oy bolusmesini GORUNUR kilar (ayri ayri iki neredeyse-
    ozdes ozellik frekansi ~%50'de kalsa bile kumeleri birlikte
    bakildiginda oy carpimi ortaya cikar)."""

    cluster_features: dict[int, list[str]] = defaultdict(list)
    for feature, cluster_id in clusters.items():
        cluster_features[cluster_id].append(feature)

    n_folds = len(nested_cv_result)
    rows = []
    for cluster_id, features in cluster_features.items():
        feature_set = set(features)
        n_with_any = 0
        for _, fold_row in nested_cv_result.iterrows():
            selected = set(fold_row["selected_features"])
            if selected & feature_set:
                n_with_any += 1
        rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(features),
                "cluster_features": ";".join(sorted(features)),
                "n_folds_with_any_selected": n_with_any,
                "frequency_any_selected": (n_with_any / n_folds) if n_folds else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("frequency_any_selected", ascending=False).reset_index(
        drop=True
    )


# =====================================================================
# 8) Arm orkestrasyonu -- bir (ozellik kumesi, esik) kombinasyonu
# =====================================================================


#: `evaluate_external_test()` ile AYNI anahtar kümesi -- `run_external_
#: test=False` olan kollarda (GTR/IDH/who2021_wildtype) bu placeholder
#: kullanılır, `write_outputs()`'un `**arm.external_test` yayılımı
#: KIRILMASIN diye (görev talimatı: "sessizce NaN üretmesin" -- burada
#: NaN'lar SESSİZ değil, `skipped=True` + `skip_reason` ile AÇIKÇA
#: işaretli).
def _external_test_skipped_placeholder(reason: str) -> dict[str, Any]:
    return {
        "n_patients": 0,
        "n_events": 0,
        "c_index": float("nan"),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "confidence_level": float("nan"),
        "n_bootstrap_valid": 0,
        "n_bootstrap_requested": 0,
        "skipped": True,
        "skip_reason": reason,
    }


@dataclass
class ArmResult:
    name: str
    feature_columns: list[str]
    stability_frequency_threshold: float
    nested_cv_fold_results: pd.DataFrame  # fallback kolonu eklenmis
    n_fallback_folds: int
    selection_frequency_summary: pd.DataFrame
    final_model: FinalModelResult
    external_test: dict[str, Any]
    # 2026-08-15 EKLENDİ (klinik kovaryat desteği). VARSAYILAN DEĞERLER
    # VAR (`extra_columns=[]`/`run_external_test=True`) -- ama bu DAHA
    # ÖNCE (bugün, run #5) diskte YAZILMIŞ `_checkpoints/arm_*.pkl`
    # dosyalarını GERİYE DÖNÜK uyumlu YAPMAZ: pickle, dataclass
    # varsayılanlarını değil kaydedilmiş `__dict__`'i geri yükler --
    # eski bir checkpoint bu iki alanı HİÇ İÇERMEZ. `write_outputs()`
    # bu yüzden bu iki alana `getattr(..., varsayılan)` ile erişir
    # (bkz. orada bırakılan not) -- doğrudan `arm.extra_columns` erişimi
    # eski checkpoint'lerde `AttributeError` verirdi.
    extra_columns: list[str] = field(default_factory=list)
    run_external_test: bool = True


def run_modeling_arm(
    name: str,
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    external_frame: pd.DataFrame | None,
    *,
    stability_frequency_threshold: float,
    outer_splits: int = 5,
    inner_splits: int = 5,
    n_bootstrap_stability: int = 200,
    n_bootstrap_external: int = 1000,
    seed: int = 42,
    l1_ratio_grid: tuple[float, ...] = DEFAULT_L1_RATIO_GRID,
    penalizer_grid: tuple[float, ...] = DEFAULT_PENALIZER_GRID,
    # 2026-08-15 EKLENDİ (klinik kovaryat desteği) -- hepsi varsayılan
    # DEĞERLERİYLE eski davranışı BİREBİR korur (extra_columns=[] +
    # run_external_test=True + diğer ikisi None => önceki 3 kol
    # ETKİLENMEZ).
    extra_columns: list[str] | None = None,
    extra_column_penalizer: float | None = None,
    clinical_standardize_columns: list[str] | None = None,
    run_external_test: bool = True,
) -> ArmResult:
    """Bir arm'in tam zinciri: run_nested_cv() -> fallback/frekans audit'i
    -> full-pool final model -> (run_external_test=True ise) TCGA'ya
    karsi evaluate_external_test().

    `extra_columns` (2026-08-15 EKLENDİ): ZORUNLU/mandatory kovaryatlar
    (örn. klinik `clinical_age`/`clinical_gender_male`/...) -- elastic-
    net/stabilite seçimine GİRMEZ, `run_nested_cv()`/`fit_final_model_
    on_full_pool()`'un KENDİ `extra_columns` mekanizmasına (var olan,
    şimdiye kadar yalnız `source_*` dummy'ler için kullanılan) devredilir.
    Varsayılan `None`/`[]` -- eski 3 kol (radyomik-yalnız) ETKİLENMEZ.

    `run_external_test=False` (2026-08-15 EKLENDİ, görev talimatı: "GTR'li
    bir kol için evaluate_external_test ÇAĞIRMA"): `evaluate_external_
    test()` HİÇ ÇAĞRILMAZ -- `external_frame` `None` olabilir.
    `ArmResult.external_test` `_external_test_skipped_placeholder()` ile
    doldurulur (SESSİZ NaN değil, `skipped=True` + nedeni AÇIKÇA).
    """

    duration_col = "survival_days"
    event_col = "event"
    extra_columns = list(extra_columns or [])
    clinical_standardize_columns = list(clinical_standardize_columns or [])

    nested_cv_result = run_nested_cv(
        training_frame,
        feature_columns,
        duration_col=duration_col,
        event_col=event_col,
        extra_columns=extra_columns,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        seed=seed,
        l1_ratio_grid=l1_ratio_grid,
        penalizer_grid=penalizer_grid,
        stability_selection=True,
        n_bootstrap_stability=n_bootstrap_stability,
        stability_frequency_threshold=stability_frequency_threshold,
        extra_column_penalizer=extra_column_penalizer,
        clinical_standardize_columns=clinical_standardize_columns,
    )

    fallback_frame, frequency_frame = audit_nested_cv_fold_selection(
        training_frame,
        feature_columns,
        nested_cv_result,
        duration_col=duration_col,
        event_col=event_col,
        extra_columns=extra_columns,
        outer_splits=outer_splits,
        seed=seed,
        n_bootstrap_stability=n_bootstrap_stability,
        stability_frequency_threshold=stability_frequency_threshold,
        extra_column_penalizer=extra_column_penalizer,
        clinical_standardize_columns=clinical_standardize_columns,
    )
    fold_results = nested_cv_result.merge(fallback_frame, on="fold", how="left")
    n_fallback_folds = int(fallback_frame["used_fallback"].sum()) if not fallback_frame.empty else 0
    selection_frequency_summary = summarize_selection_frequency(frequency_frame)

    # 2026-08-15 EKLENDİ (klinik kovaryat desteği -- görev talimatı:
    # "Standardizasyon (scaler) da fold içinde fit edilmeli"): TAM
    # havuz (`training_frame`, dağıtılacak final model) düzeyinde BİR
    # KEZ, `training_frame`'in KENDİ istatistikleriyle fit edilir; AYNI
    # dönüşüm (verilmişse) `external_frame`'e (TCGA) uygulanır --
    # TCGA'nın kendi istatistikleri KULLANILMAZ (sızıntı/asimetri
    # riski, ComBat'ın TCGA-referans-apply tartışmasıyla AYNI SINIF
    # dikkat). `run_nested_cv()`'nin KENDİ İÇ fold-safe standardizasyonundan
    # BAĞIMSIZ -- orası dış-fold train alt-kümesiyle ayrıca fit eder
    # (bkz. `standardize_columns_fold_safe()` docstring'i: penalize
    # EDİLMEYEN bir kovaryat için bu iki ayrı fit C-index'i DEĞİŞTİRMEZ).
    full_pool_training_frame, full_pool_external_frame = standardize_columns_fold_safe(
        training_frame, external_frame, clinical_standardize_columns
    )

    final_model = fit_final_model_on_full_pool(
        full_pool_training_frame,
        feature_columns,
        duration_col=duration_col,
        event_col=event_col,
        extra_columns=extra_columns,
        inner_splits=inner_splits,
        l1_ratio_grid=l1_ratio_grid,
        penalizer_grid=penalizer_grid,
        n_bootstrap_stability=n_bootstrap_stability,
        stability_frequency_threshold=stability_frequency_threshold,
        seed=seed,
        extra_column_penalizer=extra_column_penalizer,
    )

    if run_external_test:
        if full_pool_external_frame is None:
            raise ValueError(
                f"Kol {name!r}: run_external_test=True ama external_frame "
                "verilmedi (None)."
            )
        external_result = evaluate_external_test(
            final_model.fitted_model,
            full_pool_external_frame,
            final_model.final_features,
            duration_col=duration_col,
            event_col=event_col,
            extra_columns=extra_columns,
            n_bootstrap=n_bootstrap_external,
            seed=seed,
        )
    else:
        external_result = _external_test_skipped_placeholder(
            f"Kol {name!r}: görev talimatı gereği harici test BİLİNÇLİ "
            "olarak atlandı (bu kolun klinik kovaryatı TCGA'da "
            "eksik/mevcut değil -- bkz. run_pipeline() arm config'i)."
        )

    return ArmResult(
        name=name,
        feature_columns=feature_columns,
        stability_frequency_threshold=stability_frequency_threshold,
        nested_cv_fold_results=fold_results,
        n_fallback_folds=n_fallback_folds,
        selection_frequency_summary=selection_frequency_summary,
        final_model=final_model,
        external_test=external_result,
        extra_columns=extra_columns,
        run_external_test=run_external_test,
    )


# =====================================================================
# 8.5) Klinik kovaryat kolları -- önceden ilan edilmiş 5 EK kol
# =====================================================================
#
# Görev talimatı tablosu (dondurulmuş, sonuç görülmeden):
#   clinical_base           | yaş+cinsiyet (radyomik YOK)              | ✅ TCGA
#   radiomics_clinical      | WT93 (seçimli) + yaş+cinsiyet (zorunlu)  | ✅ TCGA
#   radiomics_clinical_gtr  | yukarısı + GTR + gtr_missing             | ❌ yalnız iç CV
#   idh_sensitivity         | 515 alt kohort, radyomik+yaş+cinsiyet+IDH| ❌ yalnız iç CV
#   who2021_wildtype        | yalnız 499 IDH-wildtype, radyomik+yaş+cinsiyet | ❌ yalnız iç CV
#
# BİRİNCİL MODEL (`primary_wt93_icc60_th06`) DEĞİŞMEZ -- bu bölüm SADECE
# EK kollar üretir.


@dataclass
class ClinicalArmConfig:
    name: str
    feature_set: str  # "wt93" | "none" ("none" -> radyomik YOK, clinical_base)
    include_gtr: bool
    include_idh: bool
    idh_cohort_filter: str | None  # None / "wildtype_or_mutated" / "wildtype_only"
    run_external_test: bool


CLINICAL_ARM_CONFIGS: tuple[ClinicalArmConfig, ...] = (
    ClinicalArmConfig("clinical_base", "none", False, False, None, True),
    ClinicalArmConfig("radiomics_clinical", "wt93", False, False, None, True),
    ClinicalArmConfig("radiomics_clinical_gtr", "wt93", True, False, None, False),
    ClinicalArmConfig("idh_sensitivity", "wt93", False, True, "wildtype_or_mutated", False),
    ClinicalArmConfig("who2021_wildtype", "wt93", False, False, "wildtype_only", False),
)


def select_clinical_arm_configs(
    names: list[str] | None,
    arm_configs: tuple[ClinicalArmConfig, ...] = CLINICAL_ARM_CONFIGS,
) -> tuple[ClinicalArmConfig, ...]:
    """`--clinical-arms` ile verilen adları `ClinicalArmConfig`'lere çevirir.

    `names` None ise (bayrak verilmedi) TÜM kollar döner -- yani mevcut
    davranış DEĞİŞMEZ. Aksi halde YALNIZ adı geçen kollar, `arm_configs`
    içindeki KANONİK SIRAYLA döner (kullanıcının yazdığı sıra DEĞİL --
    kol sırası checkpoint/rapor düzenini etkilemesin diye).

    Mükerrer adlar sessizce tekilleştirilir. Bilinmeyen ad `ValueError`
    ile patlar (argparse `choices` zaten yakalar; bu, fonksiyon doğrudan
    çağrıldığında -- örn. testlerde -- ikinci savunma hattıdır).
    """

    if names is None:
        return arm_configs

    known = {config.name: config for config in arm_configs}
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise ValueError(
            f"select_clinical_arm_configs: bilinmeyen kol adı {unknown} -- "
            f"geçerli adlar: {sorted(known)}"
        )

    requested = set(names)
    return tuple(config for config in arm_configs if config.name in requested)


def clinical_arm_extra_columns(config: ClinicalArmConfig) -> list[str]:
    """Bir `ClinicalArmConfig`'in `extra_columns` (ZORUNLU kovaryat)
    listesini üretir -- yaş+cinsiyet HER ZAMAN, GTR/IDH yalnız
    `config.include_gtr`/`config.include_idh` iken eklenir."""

    columns = [CLINICAL_AGE_COLUMN, CLINICAL_GENDER_MALE_COLUMN]
    if config.include_gtr:
        columns += [CLINICAL_GTR_Y_COLUMN, CLINICAL_GTR_MISSING_COLUMN]
    if config.include_idh:
        columns += [CLINICAL_IDH_MUTANT_COLUMN]
    return columns


def run_clinical_arms(
    *,
    upenn_wide: pd.DataFrame,
    upenn_patients: pd.DataFrame,
    tcga_wide: pd.DataFrame,
    tcga_patients: pd.DataFrame,
    feature_columns_wt93: list[str],
    region: str,
    args: argparse.Namespace,
    l1_ratio_grid: tuple[float, ...],
    penalizer_grid: tuple[float, ...],
    ckpt_dir: Path,
    arm_results: list[ArmResult],
    training_reports: dict[str, Any],
    external_reports: dict[str, Any],
    arm_configs: tuple[ClinicalArmConfig, ...] = CLINICAL_ARM_CONFIGS,
) -> None:
    """`CLINICAL_ARM_CONFIGS`'teki (varsayılan) 5 EK kolu sırayla çalıştırır.

    `region` şu an SADECE `PRIMARY_REGIONS[0]` ("WT") ile çağrılır --
    parametre olarak alınması `upenn_wide`/`tcga_wide`'ın hangi bölgeyle
    pivot edildiğini AÇIKÇA belgelemek için (görev kapsamı WT-only,
    bkz. `run_pipeline()`'in `assert PRIMARY_REGIONS == ("WT",)` satırı).

    `arm_results`/`training_reports`/`external_reports` YERİNDE (in
    place) DEĞİŞTİRİLİR (çağıranın -- `run_pipeline()` -- zaten sahip
    olduğu listeler/sözlükler) -- bu üç kap birincil 3 kolun ÇIKTISINI
    da İÇERİR, bu fonksiyon SADECE EKLER, ÜZERİNE YAZMAZ (kol adları
    birbirinden FARKLI olduğu için çakışma riski yok, ayrıca aşağıda
    AÇIKÇA doğrulanıyor).
    """

    collision = set(config.name for config in arm_configs) & set(training_reports)
    if collision:
        raise ValueError(
            f"run_clinical_arms: kol adı çakışması -- {sorted(collision)} "
            "zaten training_reports'ta var (birincil kollarla AYNI isim "
            "kullanılmış olabilir)."
        )

    upenn_clinical = build_clinical_covariate_frame(
        upenn_patients, include_gender=True, include_gtr=True, include_idh=True
    )
    tcga_clinical = build_clinical_covariate_frame(
        tcga_patients, include_gender=True, include_gtr=True, include_idh=True
    )
    upenn_patients_with_clinical = upenn_patients.join(upenn_clinical)
    tcga_patients_with_clinical = tcga_patients.join(tcga_clinical)

    for config in arm_configs:
        extra_columns = clinical_arm_extra_columns(config)
        feature_columns = list(feature_columns_wt93) if config.feature_set == "wt93" else []

        cohort_patients = upenn_patients_with_clinical
        if config.idh_cohort_filter is not None:
            cohort_patients = filter_to_idh_known_cohort(
                cohort_patients, keep=config.idh_cohort_filter
            )

        if config.feature_set == "wt93":
            feature_frame = upenn_wide[feature_columns]
        else:
            # clinical_base: radyomik YOK -- kohort YİNE DE `upenn_wide`
            # (radyomik satırı olan 611 hasta) ile hizalanır -- diğer
            # arm'larla AYNI kohort üzerinde karşılaştırılabilir olsun
            # diye (görev talimatı bunu açıkça belirtmiyor ama farklı
            # kohortla "birincil model"le karşılaştırma anlamsız olurdu
            # -- bu BİLİNÇLİ bir tasarım kararı, Barış'a raporda AÇIKÇA
            # belirtilmeli).
            feature_frame = pd.DataFrame(index=upenn_wide.index)

        training_frame, training_report = build_training_frame(
            feature_frame,
            cohort_patients,
            check_combat_identity=True,
            passthrough_columns=extra_columns,
        )
        training_reports[config.name] = training_report

        if config.run_external_test:
            assert_external_test_feasible(
                config.name, tcga_patients_with_clinical, extra_columns
            )
            if config.feature_set == "wt93":
                tcga_feature_frame = tcga_wide[feature_columns]
            else:
                tcga_feature_frame = pd.DataFrame(index=tcga_wide.index)
            external_frame, external_frame_report = build_external_test_frame(
                tcga_feature_frame,
                tcga_patients_with_clinical,
                passthrough_columns=extra_columns,
            )
            external_reports[config.name] = external_frame_report
        else:
            external_frame = None

        clinical_standardize_columns = [CLINICAL_AGE_COLUMN]

        # 2026-09-14 (modeling-agent-S1): klinik kollar da AYNI parmak izi
        # kapisindan gecer -- yukaridaki `run_pipeline()` kolunda anlatilan
        # "esik/seed/grid karsilastirilmiyordu" problemi burada da vardi.
        # EK alanlar (`extra_columns`, `extra_column_penalizer`,
        # `run_external_test`) bu kollarin sonucunu DEGISTIRDIGI icin
        # parmak izine dahil edildi.
        arm_ckpt = ckpt_dir / f"arm_{config.name}.pkl"
        arm_fingerprint = build_arm_checkpoint_fingerprint(
            threshold=args.primary_threshold,
            seed=args.seed,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            n_bootstrap_stability=args.n_bootstrap_stability,
            n_bootstrap_external=args.n_bootstrap_external,
            feature_set=config.feature_set,
            l1_ratio_grid=l1_ratio_grid,
            penalizer_grid=penalizer_grid,
            extra={
                "extra_columns": extra_columns,
                "extra_column_penalizer": float(args.clinical_extra_column_penalizer),
                "run_external_test": bool(config.run_external_test),
                "clinical_standardize_columns": clinical_standardize_columns,
            },
        )
        arm_result = None
        if not getattr(args, "ignore_arm_checkpoints", False):
            arm_result = load_arm_checkpoint(
                arm_ckpt, arm_fingerprint, arm_name=config.name
            )

        if arm_result is None:
            arm_result = run_modeling_arm(
                config.name,
                training_frame,
                feature_columns,
                external_frame,
                stability_frequency_threshold=args.primary_threshold,
                outer_splits=args.outer_splits,
                inner_splits=args.inner_splits,
                n_bootstrap_stability=args.n_bootstrap_stability,
                n_bootstrap_external=args.n_bootstrap_external,
                seed=args.seed,
                l1_ratio_grid=l1_ratio_grid,
                penalizer_grid=penalizer_grid,
                extra_columns=extra_columns,
                extra_column_penalizer=args.clinical_extra_column_penalizer,
                clinical_standardize_columns=clinical_standardize_columns,
                run_external_test=config.run_external_test,
            )
            save_arm_checkpoint(arm_ckpt, arm_result, arm_fingerprint)
            print(f"[checkpoint] {config.name}: sonuç diske yazıldı ({arm_ckpt.name}).", flush=True)

        arm_results.append(arm_result)
        if config.run_external_test:
            external_summary = (
                f"{arm_result.external_test['c_index']:.4f} "
                f"[{arm_result.external_test['ci_lower']:.4f}-"
                f"{arm_result.external_test['ci_upper']:.4f}]"
            )
        else:
            external_summary = "ATLANDI (yalniz ic CV -- gorev talimati)"
        print(
            f"[{config.name}] n_train={training_report.n_output_rows} "
            f"fold C-index ort={arm_result.nested_cv_fold_results['c_index'].mean():.4f} "
            f"fallback_folds={arm_result.n_fallback_folds}/{args.outer_splits} "
            f"final_ozellik(radyomik)={len(arm_result.final_model.final_features)} "
            f"klinik_kovaryat={sorted(extra_columns)} "
            f"harici_c_index={external_summary}"
        )


# =====================================================================
# 8.6) BİRİNCİL TEK MODEL -- 2026-08-18 Barış kararı (kol karşılaştırması
#      YOK). Model: WT x 93 (ICC>=0,60) + yaş + cinsiyet + GTR(+missing)
#      + IDH1(mutant+missing), TAM 611 kohort, harici test UCSF-PDGM
#      (TCGA harici testten ÇIKARILDI -- TCGA'da GTR/IDH1 %0 dolu, model
#      bu iki kovaryatı ZORUNLU içerdiği için TCGA'ya uygulanamaz).
# =====================================================================

PRIMARY_SINGLE_MODEL_ARM_NAME = "primary_wt93_clinical_full_ucsf"

#: Görev talimatındaki tanım: "WT × 93 radyomik + yaş + cinsiyet + GTR +
#: IDH1". Sıra `clinical_arm_extra_columns()`'daki desenle TUTARLI.
PRIMARY_CLINICAL_EXTRA_COLUMNS: tuple[str, ...] = (
    CLINICAL_AGE_COLUMN,
    CLINICAL_GENDER_MALE_COLUMN,
    CLINICAL_GTR_Y_COLUMN,
    CLINICAL_GTR_MISSING_COLUMN,
    CLINICAL_IDH_MUTANT_COLUMN,
    CLINICAL_IDH_MISSING_COLUMN,
)


def run_primary_single_model(
    *,
    upenn_long: pd.DataFrame,
    upenn_patients: pd.DataFrame,
    ucsf_long: pd.DataFrame,
    ucsf_patients: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> int:
    """2026-08-18 Barış kararı: TEK MODEL, kol karşılaştırması YAPILMAZ.

    `run_pipeline()`'DAN (eski 3+5=8 kollu karşılaştırma, TCGA harici
    test) BİLİNÇLİ olarak AYRI bir fonksiyon -- `run_pipeline()` ve onun
    84 testi HİÇ DEĞİŞTİRİLMEDİ (duyarlılık kolları ileride gerekirse
    hâlâ kullanılabilir olsun diye, görev talimatı: "idh_cohort_filter
    mekanizmasını SİLME"). `main()` artık VARSAYILAN olarak bu fonksiyonu
    çağırır -- `run_pipeline()` yalnız testlerden/bilinçli legacy
    kullanımdan doğrudan çağrılabilir kalır.

    Model tanımı (kilitli, sorgulanmadı): `WT` bölgesi × `STABLE_
    FEATURES_ICC60` (93) + `PRIMARY_CLINICAL_EXTRA_COLUMNS` (yaş/
    cinsiyet/GTR+missing/IDH+missing) -- `idh_cohort_filter` UYGULANMAZ,
    tam 611 hasta/585 olay kohortu kullanılır (K15 kapandı, gösterge
    değişkeni deseni -- bkz. `build_clinical_covariate_frame()`).

    Harici test: UCSF-PDGM (`SEGMENTATION_TOOL_UCSF_C32`) -- TCGA'ya BU
    FONKSİYONDA hiç dokunulmaz (parametre olarak bile alınmıyor).

    ComBat MUAFİYETİ (2026-08-18, koordinatör kararı -- `UCSF_NO_COMBAT_
    NOTE`): UCSF, TCGA'nın bıraktığı harici test rolünü devraldığı için
    AYNI muameleyi görür -- feature-level ComBat düzeltmesi ALMAZ, ComBat
    fit'ine GİRMEZ. Bu fonksiyon `pipeline.harmonization.apply_combat_
    harmonization()`/`fit_combat_harmonization()`'ı HİÇ ÇAĞIRMAZ (UCSF
    radyomikleri `pivot_ucsf_wt_long_to_wide()`'dan HAM/C32 olarak gelir)
    -- bu davranış `tests/test_train_cox_week3.py::test_run_primary_
    single_model_never_applies_combat_to_ucsf` ile monkeypatch tabanlı
    bir regresyon testiyle KİLİTLENDİ.
    """

    region = PRIMARY_REGIONS[0]
    assert PRIMARY_REGIONS == ("WT",), "Birincil bolge WT olmali (kilitli karar)."

    upenn_wide, upenn_pivot_report = pivot_radiomics_long_to_wide(
        upenn_long, regions=list(PRIMARY_REGIONS)
    )
    ucsf_wide, ucsf_pivot_report = pivot_ucsf_wt_long_to_wide(ucsf_long)

    feature_columns = select_feature_columns(upenn_wide, STABLE_FEATURES_ICC60, region=region)
    extra_columns = list(PRIMARY_CLINICAL_EXTRA_COLUMNS)

    # 2026-08-18 REVİZYON (Barış kararı (a)): UCSF de `patients` tablosuna
    # ZATEN HARMONİZE yazılır (db-agent'in `tools/onboard_ucsf_patients.py`
    # işi) -- ayrı bir `build_ucsf_clinical_covariate_frame()` fonksiyonuna
    # gerek yok, UPenn ile AYNI fonksiyon/AYNI sözlük kullanılır (bkz.
    # yukarıdaki "1.6) UCSF-PDGM klinik kovaryatları" bölüm notu).
    upenn_clinical = build_clinical_covariate_frame(
        upenn_patients, include_gender=True, include_gtr=True, include_idh=True
    )
    ucsf_clinical = build_clinical_covariate_frame(
        ucsf_patients, include_gender=True, include_gtr=True, include_idh=True
    )
    upenn_patients_c = upenn_patients.join(upenn_clinical)
    ucsf_patients_c = ucsf_patients.join(ucsf_clinical)

    # idh_cohort_filter BİLİNÇLİ OLARAK UYGULANMIYOR -- tam 611 kohort
    # (görev talimatı: "Kohort daraltma OLMADAN 611'in tamamı kullanılabilmeli").
    training_frame, training_report = build_training_frame(
        upenn_wide[feature_columns],
        upenn_patients_c,
        check_combat_identity=True,
        passthrough_columns=extra_columns,
    )

    n_events = int(training_frame["event"].sum())
    # `regions=PRIMARY_REGIONS` (= ("WT",)) -- bolge-farkinda kapinin
    # TABAN kolu: WT-only'de azalma beyan EDILMEMISTIR, 611/585 SERT
    # beklenir (2026-08-19, bkz. `check_training_pool_counts()`).
    check_training_pool_counts(
        training_report.n_output_rows,
        n_events,
        expected_patients=args.expected_patient_count,
        expected_events=args.expected_event_count,
        allow_mismatch=args.allow_unexpected_patient_count,
        regions=PRIMARY_REGIONS,
        dropped_patient_ids=all_dropped_patient_ids(training_report),
    )

    external_frame, external_frame_report = build_external_test_frame(
        ucsf_wide[feature_columns], ucsf_patients_c, passthrough_columns=extra_columns
    )
    check_external_test_pool_counts(
        len(external_frame),
        int(external_frame["event"].sum()),
        expected_patients=args.expected_ucsf_patient_count,
        expected_events=args.expected_ucsf_event_count,
        allow_mismatch=args.allow_unexpected_ucsf_count,
        label="UCSF harici test",
    )

    l1_ratio_grid = (
        tuple(args.l1_ratio_grid) if getattr(args, "l1_ratio_grid", None) else DEFAULT_L1_RATIO_GRID
    )
    penalizer_grid = (
        tuple(args.penalizer_grid) if getattr(args, "penalizer_grid", None) else DEFAULT_PENALIZER_GRID
    )

    arm_result = run_modeling_arm(
        PRIMARY_SINGLE_MODEL_ARM_NAME,
        training_frame,
        feature_columns,
        external_frame,
        stability_frequency_threshold=args.primary_threshold,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        n_bootstrap_stability=args.n_bootstrap_stability,
        n_bootstrap_external=args.n_bootstrap_external,
        seed=args.seed,
        l1_ratio_grid=l1_ratio_grid,
        penalizer_grid=penalizer_grid,
        extra_columns=extra_columns,
        extra_column_penalizer=args.clinical_extra_column_penalizer,
        clinical_standardize_columns=[CLINICAL_AGE_COLUMN],
        run_external_test=True,
    )

    print(
        f"[{PRIMARY_SINGLE_MODEL_ARM_NAME}] n_train={training_report.n_output_rows} "
        f"n_events={n_events} fold C-index ort="
        f"{arm_result.nested_cv_fold_results['c_index'].mean():.4f} "
        f"fallback_folds={arm_result.n_fallback_folds}/{args.outer_splits} "
        f"final_ozellik(radyomik)={len(arm_result.final_model.final_features)} "
        f"klinik_kovaryat={sorted(extra_columns)} "
        f"UCSF n={external_frame_report.n_output_rows} "
        f"harici_c_index={arm_result.external_test['c_index']:.4f} "
        f"[{arm_result.external_test['ci_lower']:.4f}-{arm_result.external_test['ci_upper']:.4f}]"
    )

    write_primary_single_model_outputs(
        output_dir=output_dir,
        arm_result=arm_result,
        upenn_pivot_report=upenn_pivot_report,
        ucsf_pivot_report=ucsf_pivot_report,
        training_report=training_report,
        external_report=external_frame_report,
        args=args,
    )
    return 0


def write_primary_single_model_outputs(
    *,
    output_dir: Path,
    arm_result: ArmResult,
    upenn_pivot_report: RegionPivotReport,
    ucsf_pivot_report: RegionPivotReport,
    training_report: TrainingFrameReport,
    external_report: "ExternalFrameReport",
    args: argparse.Namespace,
) -> None:
    """`write_outputs()`'un TEK-model/UCSF karşılığı -- `write_outputs()`
    KASITLI OLARAK yeniden kullanılmadı: o fonksiyonun metadata çıktısı
    `"tcga_pivot_report"`/`"segmentation_tool_tcga_c32"` anahtarlarını
    HER ZAMAN yazar (legacy 8-kollu karşılaştırmanın TCGA'ya sabit
    tasarımı) -- UCSF verisini "tcga_pivot_report" etiketiyle yazmak
    YANILTICI olurdu (CLAUDE.md davranış kuralı #3: sayı/isim tutarlılığı).
    """

    fold_copy = arm_result.nested_cv_fold_results.copy()
    fold_copy.insert(0, "arm", arm_result.name)
    fold_copy["selected_features"] = fold_copy["selected_features"].apply(
        lambda features: ";".join(sorted(features))
    )
    # 2026-09-14 (modeling-agent-S1): fold CSV'si TEK BASINA alindiginda
    # hangi stabilite esigiyle uretildigini SOYLEMIYORDU (deger yalnizca
    # run_metadata.json'da vardi). Sutun SONA ekleniyor -> mevcut
    # tuketicilerin kolon SIRASI degismiyor.
    fold_copy["stability_frequency_threshold"] = arm_result.stability_frequency_threshold
    fold_copy.to_csv(output_dir / "week3_fold_results.csv", index=False)

    freq_copy = arm_result.selection_frequency_summary.copy()
    freq_copy.insert(0, "arm", arm_result.name)
    freq_copy.to_csv(output_dir / "week3_selection_frequency.csv", index=False)

    external_row = {
        "arm": arm_result.name,
        "stability_frequency_threshold": arm_result.stability_frequency_threshold,
        "n_final_features": len(arm_result.final_model.final_features),
        "final_features": ";".join(sorted(arm_result.final_model.final_features)),
        "used_fallback_full_pool": arm_result.final_model.used_fallback,
        "n_fallback_folds": arm_result.n_fallback_folds,
        "n_outer_folds": len(arm_result.nested_cv_fold_results),
        "clinical_extra_columns": ";".join(sorted(arm_result.extra_columns)),
        "run_external_test": arm_result.run_external_test,
        "external_test_source": "UCSF-PDGM",
        **arm_result.external_test,
    }
    pd.DataFrame([external_row]).to_csv(output_dir / "week3_external_test.csv", index=False)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _script_sha256(),
        "method_name": METHOD_NAME,
        "mode": "primary_single_model_ucsf_external_test",
        "primary_model_note": (
            "2026-08-18 Baris karari (kilitli, sorgulanmadi): TEK MODEL, "
            "kol karsilastirmasi YAPILMADI. Model: WT x 93 radyomik "
            "(ICC>=0,60) + yas + cinsiyet + GTR(+missing) + "
            "IDH1(mutant+missing), TAM 611 hasta / 585 olum olayi "
            "kohortu (idh_cohort_filter UYGULANMADI). Harici test TCGA "
            "DEGIL UCSF-PDGM (TCGA'da GTR/IDH1 %0 dolu, model bu iki "
            "kovaryati zorunlu icerdigi icin TCGA'ya UYGULANAMAZ)."
        ),
        # 🔴 ZORUNLU BEYAN (rapordan CIKARILAMAZ -- decisions/2026-08-18-
        # tek-model-ucsf-harici-test-k15-kapanisi.md SS0 CELISKI 1):
        # CLAUDE.md'nin kilitli birincil model tanimi (2026-08-13) YALNIZ
        # radyomiktir (WT x 93, klinik kovaryat YOK). Bu script'in TEK
        # modeli (radyomik+yas+cinsiyet+GTR+IDH1) o tanimin DISINDADIR ve
        # POST-HOC'tur -- tetikleyicisi 2026-08-15'te GORULEN bir sonuc
        # (yas tek basina C-index 0,6295, 93 radyomigin TAMAMINDAN iyi;
        # yalniz-radyomik modelin harici sonucu 0,689 [0,546-0,824]).
        # CLAUDE.md/AGENTS.md bu karari yansitacak sekilde GUNCELLENMEDEN
        # bu script "kilitli protokolun BIREBIR uygulanmasi" olarak
        # SUNULAMAZ -- raporlarda asagidaki cumle KULLANILMALI.
        "post_hoc_disclosure_note": (
            "Birincil analiz plani 2026-08-13'te yalniz-radyomik (WT x93) "
            "model olarak dondurulmus ve 2026-08-15'te kosulmustur "
            "(harici C-index 0,689 [0,546-0,824], n=38, TCGA). Klinik "
            "kovaryatli model (bu script'in uretttigi), bu sonuc VE "
            "yasin tek basina 0,6295 verdigi olcum GORULDUKTEN SONRA "
            "tanimlanmistir; dolayisiyla onceden belirlenmis (pre-"
            "specified) bir analiz DEGIL, kesifsel/post-hoc bir "
            "analizdir. Yalniz-radyomik sonuc raporda YER ALMALIDIR -- "
            "yeni model onun yerini ALMAZ, yanina eklenir."
        ),
        "k14_component_contribution_note": (
            "K14 KAPANDI (2026-08-19) -- KOSULDU, sonuc OLUMSUZ. "
            "'clinical_base' (yalniz yas+cinsiyet, radyomik 0) kolu ayni "
            "nested-CV protokolüyle KOSULDU "
            "(tools/run_k14_clinical_base_ucsf.py, UCSF 295/169): harici "
            "C-index 0,666454 [0,621209-0,712068], ic CV 0,6250+-0,0504; "
            "cinsiyet katsayisi -0,0031 (fiilen YAS TEK BASINA). "
            "v1_referans 0,678064 -> radyomik+GTR+IDH blogunun artimi "
            "yalniz +0,0116 ve %95 GA'lar ORTUSUK. Bu nedenle 'radyomik "
            "yasin ustune ANLAMLI katki sagliyor' cumlesi HARICI VERIYLE "
            "DESTEKLENEMEDI -- kurulamaz. Katkinin tek izi GEC ufuktaki "
            "td-AUC'dedir: AUC(730) taban 0,6834 / v1 0,7191 / v2c 0,7534 "
            "(nokta tahminleri; eslestirilmis fark bootstrap'i YAPILMADI). "
            "DIL UYARISI: 'bilesen katki analizi YAPILMADI' YAZILMAZ -- "
            "yapildi; dogrusu 'yapildi, artim istatistiksel olarak "
            "ayrismadi'. Detay: "
            "decisions/2026-08-19-k14-clinical-base-sonucu.md. "
            "(Bu metin 2026-08-28'de duzeltildi; eski hali 'K14 ATLANDI "
            "-- KAPANMADI ... kolunu KOSMUYOR' idi ve 2026-08-19 "
            "kosusuyla gecersizlesmisti.)"
        ),
        "idh_indicator_note": (
            "K15 KAPANDI (2026-08-18): IDH eksikligi (UPenn NOS/NEC, 96 "
            "hasta) GTR'yle BIREBIR ayni gosterge-degiskeni desenle ele "
            "alindi (clinical_idh_missing + clinical_idh_mutant=0 "
            "referans). Kohort 515'e DUSMEDI, tam 611 kullanildi. "
            "Gerekce: 2026-08-15 olcumu NOS/NEC HR 1,410 [1,128-1,764] "
            "p=0,0026 -- rastgele eksik DEGIL, dusurmek yanlilik yaratir."
        ),
        "ucsf_idh_missing_constant_note": (
            "UCSF'te IDH 'test yapilmamis' kategorisi YOK -- "
            "clinical_idh_missing UCSF harici test cercevesinde HER "
            "ZAMAN 0,0 (sabit kolon, kasitli, build_clinical_covariate_"
            "frame()'in dogal sonucu -- UCSF sadece Wildtype/Mutated "
            "tasir). Model calisir (terim sifirla carpilir) ama bu "
            "katsayi harici olarak TEST EDILEMEZ -- beyan edilecek "
            "sinirlilik."
        ),
        "ucsf_idh_wildtype_filter_note": (
            "UCSF kohortu (295/169) IDH-wildtype ile SINIRLI DEGIL -- "
            "filtre bilincli KALDIRILDI (model IDH1'i kovaryat olarak "
            "icerdigi icin yalniz-wildtype kohortta IDH varyansi sifir "
            "olurdu). ~20 hasta (%6,8) WHO 2021'e gore teknik olarak "
            "'IDH-mutant astrositom, grade 4' -- saf glioblastom DEGIL, "
            "raporda BEYAN edilmelidir."
        ),
        "ucsf_combat_exemption_note": UCSF_NO_COMBAT_NOTE,
        "ucsf_harmonization_note": (
            "2026-08-18 REVİZYON (Barış kararı (a)): UCSF `patients` "
            "tablosuna ZATEN HARMONİZE yazılır (Male/Female, Y/N, "
            "Wildtype/Mutated, DECEASED/ALIVE) -- eşleme tek yerde, DB "
            "yazım katmanında (tools/onboard_ucsf_patients.py, db-agent) "
            "yapılır. Bu script UCSF için UPenn ile AYNI "
            "build_clinical_covariate_frame() fonksiyonunu kullanır, "
            "ayrı bir UCSF-özel eşleme sözlüğü/fonksiyonu YOK (tek "
            "mapping kaynağı, kod tekrarı yok). GTR/EOR ve IDH ham-"
            "değer eşleme ayrıntıları (UPenn hacimsel >%90 vs UCSF "
            "ameliyat notu, vb.) artık db-agent'in onboarding kodunda "
            "denetlenmeli -- decisions/2026-08-15-upenn-ucsf-kovaryat-"
            "esleme-kurallari.md 'EN RISKLI ESLEME' hala gecerli "
            "tarihsel gerekce."
        ),
        "segmentation_tool_upenn_c32": SEGMENTATION_TOOL_UPENN_C32,
        "segmentation_tool_ucsf_c32": SEGMENTATION_TOOL_UCSF_C32,
        "ucsf_source_name_unverified": UCSF_SOURCE_NAME,
        "primary_region": PRIMARY_REGIONS[0],
        "combat_identity_note": COMBAT_IDENTITY_NOTE,
        "expected_training_patient_count": args.expected_patient_count,
        "expected_training_event_count": args.expected_event_count,
        "allow_unexpected_patient_count": args.allow_unexpected_patient_count,
        "expected_ucsf_patient_count": args.expected_ucsf_patient_count,
        "expected_ucsf_event_count": args.expected_ucsf_event_count,
        "allow_unexpected_ucsf_count": args.allow_unexpected_ucsf_count,
        "ucsf_expected_count_note": (
            "295/169 KILITLI (decisions/2026-08-18-tek-model-ucsf-"
            "harici-test-k15-kapanisi.md) -- 367/223 (2026-08-15, IDH-"
            "wildtype filtreli + 'tam indirilecek' varsayimi) ve "
            "275/165 (2026-08-18 10:15, IDH-wildtype filtreli) ikisi de "
            "GECERSIZ kilindi (tanim evrimi, celiski DEGIL). Otorite "
            "hala imaging-agent'in cikarim aninda dondurdugu liste; "
            "sayi tutmazsa bu script SERT durur (--allow-unexpected-"
            "ucsf-count OLMADIKCA)."
        ),
        "ucsf_download_incomplete_selection_bias_note": (
            "UCSF harici test kohortu, veri indirme kesintisi nedeniyle "
            "koleksiyonun ID 118-541 dilimidir; ID 4-116 araligindaki 94 "
            "hasta indirilemedi (IBM Aspera, ~142 GB, %80'de/113,34 "
            "GB'da kalici olarak durdu). Bu alt kume rezeksiyon "
            "genisligi dagiliminda tam kohorttan anlamli olarak "
            "FARKLIDIR (GTR %65 vs %39, p<0,001); yas, cinsiyet, "
            "sagkalim ve olay oraninda fark YOKTUR (p=0,358/0,517/"
            "0,836/0,735). Bu script'in kapsami DISINDA olculdu "
            "(decisions/2026-08-18-tek-model-ucsf-harici-test-k15-"
            "kapanisi.md), rapora AYNEN tasinmali."
        ),
        "cli_args": vars(args) | {"output_dir": str(args.output_dir) if args.output_dir else None},
        "upenn_pivot_report": {
            "regions_requested": upenn_pivot_report.regions_requested,
            "n_input_rows": upenn_pivot_report.n_input_rows,
            "n_candidate_patients": upenn_pivot_report.n_candidate_patients,
            "n_output_patients": upenn_pivot_report.n_output_patients,
            "dropped_patients_missing_region": upenn_pivot_report.dropped_patients_missing_region,
        },
        "ucsf_pivot_report": {
            "regions_requested": ucsf_pivot_report.regions_requested,
            "n_input_rows": ucsf_pivot_report.n_input_rows,
            "n_candidate_patients": ucsf_pivot_report.n_candidate_patients,
            "n_output_patients": ucsf_pivot_report.n_output_patients,
            "dropped_patients_missing_region": ucsf_pivot_report.dropped_patients_missing_region,
        },
        "training_frame_report": {
            "n_input_rows": training_report.n_input_rows,
            "n_output_rows": training_report.n_output_rows,
            "dropped_missing_duration": training_report.dropped_missing_duration,
            "dropped_missing_features": training_report.dropped_missing_features,
            "dropped_missing_passthrough": training_report.dropped_missing_passthrough,
            "dropped_unrecognized_vital_status": training_report.dropped_unrecognized_vital_status,
            "sources": training_report.sources,
        },
        "ucsf_external_frame_report": {
            "n_input_rows": external_report.n_input_rows,
            "n_output_rows": external_report.n_output_rows,
            "dropped_missing_duration": external_report.dropped_missing_duration,
            "dropped_missing_features": external_report.dropped_missing_features,
            "dropped_unrecognized_vital_status": external_report.dropped_unrecognized_vital_status,
        },
        "arm": {
            "name": arm_result.name,
            "n_feature_candidates": len(arm_result.feature_columns),
            "stability_frequency_threshold": arm_result.stability_frequency_threshold,
            "n_fallback_folds": arm_result.n_fallback_folds,
            "n_final_features": len(arm_result.final_model.final_features),
            "final_model_used_fallback": arm_result.final_model.used_fallback,
            "external_test": arm_result.external_test,
            "clinical_extra_columns": arm_result.extra_columns,
            "run_external_test": arm_result.run_external_test,
        },
        "reporting_rule_note": (
            "Tek-esik dili YASAK -- 'C-index >= X'i gectik' gibi cumle "
            "KURULMAZ, daima nokta tahmini + %95 bootstrap CI birlikte "
            "raporlanir (external_test.c_index/ci_lower/ci_upper)."
        ),
        "known_open_questions": [
            "UCSF verisi bu kosunun yazildigi tarihte DB'ye HENUZ "
            "YAZILMADI -- bu script SENTETIK veriyle test edildi, "
            "gercek DB kosusu imaging-agent'in dry-run'i tamamlanip "
            "Baris onay verdikten SONRA yapilmali.",
            "UCSF_SOURCE_NAME/dataset_sources.source_name gercek "
            "degeri DOGRULANMADI.",
            "UCSF'in ham tumor_region etiketi ('WT_derived' mi 'WT' mi) "
            "DOGRULANMADI -- bkz. UCSF_WT_RAW_REGION_LABELS yorumu.",
            "UCSF beklenen sayi 295/169 olarak KILITLENDI (decisions/"
            "2026-08-18-tek-model-ucsf-harici-test-k15-kapanisi.md) -- "
            "gercek C32 cikarimi/kohort dondurmasi TAMAMLANANA kadar bu "
            "sayinin GERCEKTEN tuttugu dogrulanmadi.",
            "Birincil model tanimi (radyomik+yas+cinsiyet+GTR+IDH1) "
            "CLAUDE.md'nin 2026-08-13 kilitli 'yalniz-radyomik WT x93' "
            "tanimindan POST-HOC bir sapmadir (bkz. post_hoc_disclosure_"
            "note). CLAUDE.md/AGENTS.md 2026-08-18'de bu karari "
            "yansitacak sekilde REVIZE EDILDI (yalniz-radyomik modelin "
            "birincil oldugu, 0,689 [0,546-0,824] sonucunun degismezligi "
            "ve kombine modelin post-hoc statusu artik oraya yazili) -- "
            "bu ARTIK acik bir risk DEGIL, yalniz raporlama disiplini "
            "olarak HATIRLATMA amacli burada tutuluyor.",
            # 2026-09-14 DUZELTILDI (K22): bu girdi KAPANMIS bir karari
            # "acik" gosteriyordu. CLI yardim metni A6 ile guncellenmisti,
            # metadata yazicisi guncellenmemisti. Eski metin, provenance
            # bozulmasin diye AYNEN korunuyor (parantez icinde).
            "[KAPANDI 2026-09-13, karar A6] extra_column_penalizer=0.0 "
            "sayisal degeri KILITLENDI (Baris onayi); klinik kovaryatlar "
            "forced-in tutulur (glmnet penalty.factor=0 pratigi). Detay: "
            "decisions/2026-09-13-extra-column-penalizer-sifir-onaylandi.md. "
            "(eski metin: \"extra_column_penalizer=0.0 sayisal degeri "
            "protokolde KILITLENMEDI (Barisin onayi bekleniyor).\")",
            "Cox skorlarinin XGBoost'a out-of-fold aktarimi bu scriptin "
            "kapsaminda DEGIL -- ayri gorev.",
        ],
    }
    (output_dir / "week3_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


# =====================================================================
# 8.7) Model v2 -- dört varyant orkestrasyonu (2026-08-18, Barış görevi)
# =====================================================================
#
# KURAL DEĞİŞİKLİĞİ (2026-08-18, Barış -- CLAUDE.md güncellendi):
# "MODEL GELİŞTİRME SERBESTTİR -- tek koşul ŞEFFAFLIK." Varyant denemek
# artık post-hoc diye ENGELLENMİYOR; yasak olan SAKLAMAK/GİZLEMEK, TEK
# eşik dili kullanmak veya koşulan bir varyantı rapordan ÇIKARMAK.
# `decisions/2026-08-13-...` Bölüm 4.7'nin (TCGA'ya survival_days ile
# YENİ özellik seçimi yasağı) TCGA'YA ÖZGÜ olduğu netleşti -- harici
# test artık UCSF, o kısıt UCSF'i BAĞLAMAZ. Bu bölümdeki dört varyant
# (v1_referans/v2a_mgmt/v2b_mgmt_spline/v2c_mgmt_spline_wttc) bu yeni
# kural altında ÖNCEDEN TANIMLANMIŞ -- TCGA'ya hiçbir noktada dokunmaz
# (yalnız UPenn eğitim + UCSF harici test, `run_primary_single_model()`
# ile AYNI iki kaynak).
#
# v1_referans NOT: `run_primary_single_model()` ile matematiksel olarak
# AYNI modeli üretir (WT x93 + yaş[doğrusal] + cinsiyet + GTR + IDH,
# MGMT YOK, tek bölge WT) -- bu bölüm o fonksiyonu DEĞİŞTİRMEDEN, AYNI
# alt-fonksiyonları (`build_training_frame`/`run_modeling_arm`/...)
# yeniden kullanarak bir "varyant" olarak yeniden üretir; aynı
# seed/argümanlarla koşulduğunda AYNI sonucu vermesi beklenir
# (reprodüksiyon kontrolü, görev talimatı).


@dataclass
class VariantConfig:
    """Bir Model-v2 varyantının TAM tanımı -- görev talimatındaki tablo
    (Kovaryatlar/Bölge/Yaş formu) burada kod haline gelir."""

    name: str
    regions: tuple[str, ...]
    include_mgmt: bool
    use_age_spline: bool


#: Görev talimatındaki DÖRT varyant, BİREBİR (kovaryat/bölge/yaş formu
#: tablosuyla TUTARLI). Sıra rapor/CLI sırasıyla AYNI.
V2_VARIANT_CONFIGS: tuple[VariantConfig, ...] = (
    VariantConfig(name="v1_referans", regions=("WT",), include_mgmt=False, use_age_spline=False),
    VariantConfig(name="v2a_mgmt", regions=("WT",), include_mgmt=True, use_age_spline=False),
    VariantConfig(name="v2b_mgmt_spline", regions=("WT",), include_mgmt=True, use_age_spline=True),
    VariantConfig(
        name="v2c_mgmt_spline_wttc",
        regions=("WT", "TC"),
        include_mgmt=True,
        use_age_spline=True,
    ),
)
V2_VARIANT_NAMES: tuple[str, ...] = tuple(config.name for config in V2_VARIANT_CONFIGS)

#: 2026-09-11 EKLENDİ (Barış onayı, FİNAL KOŞU -- MODEL-SECIM-ANALIZI-
#: 2026-09-11.md §3/§4.2). `v3c_lowvar_wttc_mgmt_nospline` duyarlılık
#: kolunun TABAN'ı (v3 filtresi bunun ÜZERİNE sarılır, aşağıdaki "8.6) v3"
#: bölümüne bkz.). BİLİNÇLİ OLARAK `v2c_mgmt_spline_wttc`'YE
#: DAYANMAZ (yaş formu FARKLI -- doğrusal, spline YOK) -- yeni, BAĞIMSIZ
#: bir taban. `V2_VARIANT_CONFIGS`/`V2_VARIANT_NAMES` (yukarıdaki 4
#: varyant) SATIR SATIR DEĞİŞMEDEN KORUNUR -- bu AYRI, EK bir tuple
#: (regresyon kilidi: `test_select_v2_variant_configs_default_returns_
#: all_four_in_order`, `select_v2_variant_configs(None)`'ın HÂLÂ yalnız
#: bu dört adı döndürdüğünü doğrular).
V2_VARIANT_CONFIGS_EXTRA: tuple[VariantConfig, ...] = (
    VariantConfig(
        name="v2d_mgmt_wttc_nospline",
        regions=("WT", "TC"),
        include_mgmt=True,
        use_age_spline=False,
    ),
)
V2_VARIANT_NAMES_EXTRA: tuple[str, ...] = tuple(config.name for config in V2_VARIANT_CONFIGS_EXTRA)
#: `select_v2_variant_configs()`/CLI dispatch'in kullandığı GENİŞLETİLMİŞ
#: liste (4 orijinal + 1 ek) -- `V2_VARIANT_CONFIGS`'İN KENDİSİ hâlâ
#: yalnız 4 eleman, bu yeni bir birleşik görünüm.
ALL_V2_VARIANT_CONFIGS: tuple[VariantConfig, ...] = V2_VARIANT_CONFIGS + V2_VARIANT_CONFIGS_EXTRA


def select_v2_variant_configs(
    names: list[str] | None, *, configs: tuple[VariantConfig, ...] = V2_VARIANT_CONFIGS
) -> tuple[VariantConfig, ...]:
    """`--variants` CLI degerini `VariantConfig` tuple'ina cevirir.
    `names=None` (bayrak verilmedi) -> TUM varyantlar (varsayilan sira).
    Bilinmeyen bir ad SESSIZCE yoksayilmaz -- ValueError firlatilir."""

    if names is None:
        return configs
    lookup = {config.name: config for config in configs}
    unknown = [name for name in names if name not in lookup]
    if unknown:
        raise ValueError(
            f"Bilinmeyen Model-v2 varyant adi(lar): {unknown} -- gecerli "
            f"adlar: {list(lookup)}"
        )
    return tuple(lookup[name] for name in names)


@dataclass
class VariantRunResult:
    """Bir varyantın tam çalıştırma çıktısı -- `write_variant_outputs()`/
    `write_variant_comparison_table()`'ın girdisi."""

    config: VariantConfig
    arm_result: ArmResult
    upenn_pivot_report: RegionPivotReport
    ucsf_pivot_report: RegionPivotReport
    training_report: TrainingFrameReport
    external_report: "ExternalFrameReport"
    feature_columns: list[str]
    extra_columns: list[str]
    n_train_patients: int
    n_train_events: int
    age_knots: list[float] | None


def run_single_variant(
    *,
    config: VariantConfig,
    upenn_long: pd.DataFrame,
    upenn_patients: pd.DataFrame,
    ucsf_long: pd.DataFrame,
    ucsf_patients: pd.DataFrame,
    age_knots: np.ndarray | None,
    args: argparse.Namespace,
) -> VariantRunResult:
    """Bir `VariantConfig`'in TAM zincirini çalıştırır: pivot (UPenn +
    UCSF, `config.regions`) -> klinik kovaryat inşası (`config.
    include_mgmt`/`config.use_age_spline`) -> `build_training_frame`/
    `build_external_test_frame` -> `run_modeling_arm` (nested-CV +
    fallback/frekans audit'i + full-pool final model + UCSF harici test).

    TCGA'ya bu fonksiyonda HİÇ dokunulmaz (parametre olarak bile
    alınmıyor) -- `run_primary_single_model()` ile AYNI mimari desen.
    ComBat MUAFİYETİ de AYNI (`UCSF_NO_COMBAT_NOTE`): bu fonksiyon
    `pipeline.harmonization.apply_combat_harmonization()`/
    `fit_combat_harmonization()`'ı hiç çağırmaz -- UCSF radyomikleri
    `pivot_ucsf_regions_long_to_wide()`'dan HAM/C32 olarak gelir.
    """

    upenn_wide, upenn_pivot_report = pivot_radiomics_long_to_wide(
        upenn_long, regions=list(config.regions)
    )
    ucsf_wide, ucsf_pivot_report = pivot_ucsf_regions_long_to_wide(
        ucsf_long, regions=config.regions
    )

    feature_columns: list[str] = []
    for region in config.regions:
        feature_columns += select_feature_columns(upenn_wide, STABLE_FEATURES_ICC60, region=region)

    extra_columns = build_variant_clinical_extra_columns(
        include_mgmt=config.include_mgmt, use_age_spline=config.use_age_spline
    )

    upenn_clinical = build_variant_clinical_frame(
        upenn_patients,
        include_mgmt=config.include_mgmt,
        use_age_spline=config.use_age_spline,
        age_knots=age_knots,
    )
    ucsf_clinical = build_variant_clinical_frame(
        ucsf_patients,
        include_mgmt=config.include_mgmt,
        use_age_spline=config.use_age_spline,
        age_knots=age_knots,
    )
    upenn_patients_c = upenn_patients.join(upenn_clinical)
    ucsf_patients_c = ucsf_patients.join(ucsf_clinical)

    training_frame, training_report = build_training_frame(
        upenn_wide[feature_columns],
        upenn_patients_c,
        check_combat_identity=True,
        passthrough_columns=extra_columns,
    )
    n_train_events = int(training_frame["event"].sum())
    # 2026-08-19: kapi artik BOLGE-FARKINDA. WT-only varyantlarda
    # (v1/v2a/v2b) davranis DEGISMEDI -- sert 611/585. `v2c` (WT+TC)
    # icin `DECLARED_REGION_SHORTFALLS`'ta beyan edilmis 609/583
    # aranir ve dusen 2 hasta stderr'e loglanir.
    #
    # UCSF harici test kapisi BILINCLI OLARAK bolge-kor birakildi:
    # canli olcumle dogrulandi (2026-08-19) -- UCSF'in 295 hastasinin
    # HICBIRINDE TC bos degil, WT+TC pivotunda da havuz 295/169
    # KALIYOR. Yani harici tarafta beyan edilecek bir azalma YOK;
    # varsa gelecekte, o kapi da ayrica sert durur (istenen davranis).
    check_training_pool_counts(
        training_report.n_output_rows,
        n_train_events,
        expected_patients=args.expected_patient_count,
        expected_events=args.expected_event_count,
        allow_mismatch=args.allow_unexpected_patient_count,
        regions=config.regions,
        dropped_patient_ids=all_dropped_patient_ids(training_report),
    )

    external_frame, external_frame_report = build_external_test_frame(
        ucsf_wide[feature_columns], ucsf_patients_c, passthrough_columns=extra_columns
    )
    check_external_test_pool_counts(
        len(external_frame),
        int(external_frame["event"].sum()),
        expected_patients=args.expected_ucsf_patient_count,
        expected_events=args.expected_ucsf_event_count,
        allow_mismatch=args.allow_unexpected_ucsf_count,
        label=f"UCSF harici test ({config.name})",
    )

    l1_ratio_grid = (
        tuple(args.l1_ratio_grid) if getattr(args, "l1_ratio_grid", None) else DEFAULT_L1_RATIO_GRID
    )
    penalizer_grid = (
        tuple(args.penalizer_grid) if getattr(args, "penalizer_grid", None) else DEFAULT_PENALIZER_GRID
    )
    clinical_standardize_columns = (
        [CLINICAL_AGE_RCS1_COLUMN, CLINICAL_AGE_RCS2_COLUMN]
        if config.use_age_spline
        else [CLINICAL_AGE_COLUMN]
    )

    arm_result = run_modeling_arm(
        config.name,
        training_frame,
        feature_columns,
        external_frame,
        stability_frequency_threshold=args.primary_threshold,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        n_bootstrap_stability=args.n_bootstrap_stability,
        n_bootstrap_external=args.n_bootstrap_external,
        seed=args.seed,
        l1_ratio_grid=l1_ratio_grid,
        penalizer_grid=penalizer_grid,
        extra_columns=extra_columns,
        extra_column_penalizer=args.clinical_extra_column_penalizer,
        clinical_standardize_columns=clinical_standardize_columns,
        run_external_test=True,
    )

    return VariantRunResult(
        config=config,
        arm_result=arm_result,
        upenn_pivot_report=upenn_pivot_report,
        ucsf_pivot_report=ucsf_pivot_report,
        training_report=training_report,
        external_report=external_frame_report,
        feature_columns=feature_columns,
        extra_columns=extra_columns,
        n_train_patients=training_report.n_output_rows,
        n_train_events=n_train_events,
        age_knots=None if age_knots is None else [float(k) for k in age_knots],
    )


def extract_final_model_coefficients(fitted_model) -> pd.DataFrame:
    """lifelines `CoxPHFitter.summary`'den (coef, HR=exp(coef), %95 CI,
    p) Barış'ın istediği "final model KATSAYILARI" raporunu üretir.

    ⚠️ v1 koşusunda bu KAYDEDİLMEMİŞTİ (koordinatör sonradan yeniden fit
    ederek çıkarmak zorunda kalmıştı) -- bu görev bunu, koşunun KENDİSİ
    yazacak şekilde telafi eder (görev talimatı)."""

    summary = fitted_model.summary.reset_index()
    keep_columns = [
        column
        for column in (
            "covariate",
            "coef",
            "exp(coef)",
            "se(coef)",
            "coef lower 95%",
            "coef upper 95%",
            "exp(coef) lower 95%",
            "exp(coef) upper 95%",
            "z",
            "p",
        )
        if column in summary.columns
    ]
    return summary[keep_columns]


def write_variant_outputs(
    *, output_dir: Path, result: VariantRunResult, args: argparse.Namespace
) -> dict[str, Any]:
    """Bir varyantın TAM çıktı setini (`week3_<variant>_*.csv` +
    `week3_<variant>_run_metadata.json`) yazar -- dosya adları varyant
    adıyla ayrışır, birbirinin ÜZERİNE YAZMAZ (görev talimatı).
    `write_variant_comparison_table()`'ın kullanacağı ÖZET satırı da
    döner (dosyaya YAZMAZ -- karşılaştırma tablosu tüm varyantlar
    toplandıktan SONRA bir kez yazılır)."""

    name = result.config.name
    arm = result.arm_result

    fold_copy = arm.nested_cv_fold_results.copy()
    fold_copy.insert(0, "variant", name)
    fold_copy["selected_features"] = fold_copy["selected_features"].apply(
        lambda features: ";".join(sorted(features))
    )
    # 2026-09-14 (modeling-agent-S1): esik damgasi -- bkz. write_outputs()
    fold_copy["stability_frequency_threshold"] = arm.stability_frequency_threshold
    fold_copy.to_csv(output_dir / f"week3_{name}_fold_results.csv", index=False)

    freq_copy = arm.selection_frequency_summary.copy()
    freq_copy.insert(0, "variant", name)
    freq_copy.to_csv(output_dir / f"week3_{name}_selection_frequency.csv", index=False)

    external_row = {
        "variant": name,
        "regions": ";".join(result.config.regions),
        "include_mgmt": result.config.include_mgmt,
        "use_age_spline": result.config.use_age_spline,
        "stability_frequency_threshold": arm.stability_frequency_threshold,
        "n_final_features_radiomic": len(arm.final_model.final_features),
        "final_features_radiomic": ";".join(sorted(arm.final_model.final_features)),
        "used_fallback_full_pool": arm.final_model.used_fallback,
        "n_fallback_folds": arm.n_fallback_folds,
        "n_outer_folds": len(arm.nested_cv_fold_results),
        "clinical_extra_columns": ";".join(result.extra_columns),
        "run_external_test": arm.run_external_test,
        "external_test_source": "UCSF-PDGM",
        **arm.external_test,
    }
    pd.DataFrame([external_row]).to_csv(output_dir / f"week3_{name}_external_test.csv", index=False)

    coefficients = extract_final_model_coefficients(arm.final_model.fitted_model)
    coefficients.insert(0, "variant", name)
    coefficients.to_csv(output_dir / f"week3_{name}_final_coefficients.csv", index=False)

    n_radiomic_candidates = len(result.feature_columns)
    n_clinical_covariates = len(result.extra_columns)
    n_candidate_total = n_radiomic_candidates + n_clinical_covariates
    epv_candidate_pool = (
        result.n_train_events / n_candidate_total if n_candidate_total > 0 else float("nan")
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _script_sha256(),
        "method_name": METHOD_NAME,
        "mode": "model_v2_variant",
        "variant": {
            "name": name,
            "regions": list(result.config.regions),
            "include_mgmt": result.config.include_mgmt,
            "use_age_spline": result.config.use_age_spline,
        },
        "model_development_freedom_note": (
            "2026-08-18 Baris karari (CLAUDE.md guncellendi): 'MODEL "
            "GELISTIRME SERBESTTIR -- tek kosul SEFFAFLIK.' Bu varyant "
            "post-hoc bir deneme OLABILIR (v1_referans DISINDAKI ucu) -- "
            "bu ARTIK yasak degil. Yasak olan: bu sonucu SAKLAMAK/"
            "GIZLEMEK ya da tek-esik diliyle raporlamak. Kosulan HER "
            "varyant week3_v2_variant_comparison.csv'ye GIRER."
        ),
        "mgmt_indeterminate_missing_rationale": (
            "Indeterminate (UPenn'de test yapildi ama sonuc belirsiz) "
            "'eksik' sayildi (clinical_mgmt_missing=1) -- GTR/IDH'nin "
            "'test yapilmadi' kategorisiyle AYNI epistemik durum: "
            "hastanin GERCEK MGMT durumu hakkinda klinik olarak "
            "kullanilabilir bilgi YOK. UCSF'te Indeterminate HIC "
            "gorulmuyor (db-agent onboarding'de zaten NULL'a cevrilmis)."
        ),
        "age_spline_note": (
            "Restricted cubic spline, 3 dugum (persentil 10/50/90, "
            f"RCS_AGE_KNOT_PERCENTILES={list(RCS_AGE_KNOT_PERCENTILES)}), "
            "dugum yerleri SADECE UPenn egitim havuzunun ham yas "
            "dagilimindan BIR KEZ hesaplandi, UCSF'e AYNEN uygulandi "
            "(YENIDEN hesaplanmadi -- sizinti onleme). AÇIK TASARIM "
            "KARARI: dugum yerleri nested-CV'nin dis fold'lari icinde "
            "de YENIDEN hesaplanmadi (STABLE_FEATURES_ICC60 gibi "
            "pre-spesifiye bir yapisal secim olarak ele alindi) -- bu "
            "doğrulanmadi, sonraki incelemede sorgulanmali."
            if result.config.use_age_spline
            else "Bu varyant yas splinini KULLANMIYOR (dogrusal yas)."
        ),
        "age_knots": result.age_knots,
        "ucsf_combat_exemption_note": UCSF_NO_COMBAT_NOTE,
        "segmentation_tool_upenn_c32": SEGMENTATION_TOOL_UPENN_C32,
        "segmentation_tool_ucsf_c32": SEGMENTATION_TOOL_UCSF_C32,
        "combat_identity_note": COMBAT_IDENTITY_NOTE,
        "reporting_rule_note": (
            "Tek-esik dili YASAK -- 'C-index >= X'i gectik' gibi cumle "
            "KURULMAZ, daima nokta tahmini + %95 bootstrap CI birlikte "
            "raporlanir (external_test.c_index/ci_lower/ci_upper)."
        ),
        "expected_training_patient_count": args.expected_patient_count,
        "expected_training_event_count": args.expected_event_count,
        "allow_unexpected_patient_count": args.allow_unexpected_patient_count,
        "expected_ucsf_patient_count": args.expected_ucsf_patient_count,
        "expected_ucsf_event_count": args.expected_ucsf_event_count,
        "allow_unexpected_ucsf_count": args.allow_unexpected_ucsf_count,
        "cli_args": vars(args) | {"output_dir": str(args.output_dir) if args.output_dir else None},
        "upenn_pivot_report": {
            "regions_requested": result.upenn_pivot_report.regions_requested,
            "n_input_rows": result.upenn_pivot_report.n_input_rows,
            "n_candidate_patients": result.upenn_pivot_report.n_candidate_patients,
            "n_output_patients": result.upenn_pivot_report.n_output_patients,
            "dropped_patients_missing_region": result.upenn_pivot_report.dropped_patients_missing_region,
        },
        "ucsf_pivot_report": {
            "regions_requested": result.ucsf_pivot_report.regions_requested,
            "n_input_rows": result.ucsf_pivot_report.n_input_rows,
            "n_candidate_patients": result.ucsf_pivot_report.n_candidate_patients,
            "n_output_patients": result.ucsf_pivot_report.n_output_patients,
            "dropped_patients_missing_region": result.ucsf_pivot_report.dropped_patients_missing_region,
        },
        "training_frame_report": {
            "n_input_rows": result.training_report.n_input_rows,
            "n_output_rows": result.training_report.n_output_rows,
            "dropped_missing_duration": result.training_report.dropped_missing_duration,
            "dropped_missing_features": result.training_report.dropped_missing_features,
            "dropped_missing_passthrough": result.training_report.dropped_missing_passthrough,
            "dropped_unrecognized_vital_status": result.training_report.dropped_unrecognized_vital_status,
            "sources": result.training_report.sources,
        },
        "ucsf_external_frame_report": {
            "n_input_rows": result.external_report.n_input_rows,
            "n_output_rows": result.external_report.n_output_rows,
            "dropped_missing_duration": result.external_report.dropped_missing_duration,
            "dropped_missing_features": result.external_report.dropped_missing_features,
            "dropped_unrecognized_vital_status": result.external_report.dropped_unrecognized_vital_status,
        },
        "n_radiomic_candidates": n_radiomic_candidates,
        "n_clinical_covariates": n_clinical_covariates,
        "epv_candidate_pool": epv_candidate_pool,
        "clinical_extra_columns": result.extra_columns,
        "arm": {
            "name": arm.name,
            "n_feature_candidates": len(arm.feature_columns),
            "stability_frequency_threshold": arm.stability_frequency_threshold,
            "n_fallback_folds": arm.n_fallback_folds,
            "n_final_features": len(arm.final_model.final_features),
            "final_model_used_fallback": arm.final_model.used_fallback,
            "external_test": arm.external_test,
            "clinical_extra_columns": arm.extra_columns,
            "run_external_test": arm.run_external_test,
        },
        "known_open_questions": [
            "Yas spline dugum yerleri nested-CV dis fold'lari icinde "
            "YENIDEN hesaplanmadi (yukaridaki age_spline_note) -- bu "
            "varyant spline kullaniyorsa acik risk.",
            "WT+TC kolinearite (|r|=0,87, decisions/2026-08-13-...) "
            "elastic net tarafindan yonetiliyor ama secim kararsizligini "
            "artirabilir -- fold-basina secilen ozellik sayisi "
            "week3_<variant>_fold_results.csv'de raporlaniyor.",
            # 2026-09-14 DUZELTILDI (K22): bu girdi KAPANMIS bir karari
            # "acik" gosteriyordu. CLI yardim metni A6 ile guncellenmisti,
            # metadata yazicisi guncellenmemisti. Eski metin, provenance
            # bozulmasin diye AYNEN korunuyor (parantez icinde).
            "[KAPANDI 2026-09-13, karar A6] extra_column_penalizer=0.0 "
            "sayisal degeri KILITLENDI (Baris onayi); klinik kovaryatlar "
            "forced-in tutulur (glmnet penalty.factor=0 pratigi). Detay: "
            "decisions/2026-09-13-extra-column-penalizer-sifir-onaylandi.md. "
            "(eski metin: \"extra_column_penalizer=0.0 sayisal degeri "
            "protokolde KILITLENMEDI (Barisin onayi bekleniyor).\")",
            "Cox skorlarinin XGBoost'a out-of-fold aktarimi bu scriptin "
            "kapsaminda DEGIL -- ayri gorev.",
        ]
        + (
            [
                "v3 ailesinin radyomik aday-havuz filtresi (near-constant + "
                "kolinearite kümeleme), ICC>=0,60 filtresiyle AYNI mimari "
                "kademede, eğitim havuzunun TAMAMINDAN nested-CV dış-fold "
                "döngüsünden ÖNCE ve BİR KEZ türetilir (X-only, "
                "survival_days/event kullanılmaz; UCSF bu hesaba GİRMEZ). "
                "Tam fold-yerel bir versiyon bu koşunun kapsamında DEĞİLDİR "
                "-- bkz. MODEL-SECIM-ANALIZI-2026-09-11.md §3 'a-risk'."
                if result.config.name.startswith("v3")
                else None
            ]
        ),
    }
    metadata["known_open_questions"] = [
        item for item in metadata["known_open_questions"] if item is not None
    ]
    (output_dir / f"week3_{name}_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    return {
        "variant": name,
        "regions": ";".join(result.config.regions),
        "include_mgmt": result.config.include_mgmt,
        "use_age_spline": result.config.use_age_spline,
        "n_radiomic_candidates": n_radiomic_candidates,
        "n_clinical_covariates": n_clinical_covariates,
        "n_train_patients": result.n_train_patients,
        "n_train_events": result.n_train_events,
        "epv_candidate_pool": epv_candidate_pool,
        "inner_cv_c_index_mean": float(arm.nested_cv_fold_results["c_index"].mean()),
        "inner_cv_c_index_std": float(arm.nested_cv_fold_results["c_index"].std()),
        "n_outer_folds": len(arm.nested_cv_fold_results),
        "n_fallback_folds": arm.n_fallback_folds,
        "n_final_features_radiomic": len(arm.final_model.final_features),
        "final_model_used_fallback": arm.final_model.used_fallback,
        "n_external_patients": arm.external_test["n_patients"],
        "n_external_events": arm.external_test["n_events"],
        "external_c_index": arm.external_test["c_index"],
        "external_ci_lower": arm.external_test["ci_lower"],
        "external_ci_upper": arm.external_test["ci_upper"],
        "external_confidence_level": arm.external_test["confidence_level"],
    }


def write_variant_comparison_table(
    output_dir: Path, summary_rows: list[dict[str, Any]]
) -> pd.DataFrame:
    """TEK karşılaştırma tablosu -- her satır bir varyant (görev
    talimatı: "Barış her varyantın tam künyesini görecek ve final
    raporda hepsini tablolayacak"). `write_variant_outputs()`'un
    döndürdüğü özet satırlarından kurulur -- HİÇBİR varyant dışarıda
    bırakılmaz (şeffaflık kuralı, bkz. bu bölümün başlık notu)."""

    comparison = pd.DataFrame(summary_rows)
    comparison.to_csv(output_dir / "week3_v2_variant_comparison.csv", index=False)
    return comparison


def run_v2_variant_suite(
    *,
    upenn_long: pd.DataFrame,
    upenn_patients: pd.DataFrame,
    ucsf_long: pd.DataFrame,
    ucsf_patients: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    variants: tuple[VariantConfig, ...] = V2_VARIANT_CONFIGS,
) -> int:
    """Model v2 -- SEÇİLEN (varsayılan: DÖRT) varyantı SIRAYLA koşar.
    Her varyant kendi `week3_<variant>_*.csv` dosya setini üretir +
    hepsi bittiğinde TEK bir `week3_v2_variant_comparison.csv` yazılır.

    ŞEFFAFLIK KURALI (görev talimatı, bkz. bu bölümün başlık notu):
    `variants` parametresi/`--variants` CLI bayrağı yalnız HANGİ
    varyantların koşulacağını SEÇER -- koşulan HER varyant çıktı
    setine/karşılaştırma tablosuna GİRER, SEÇİCİ raporlama YAPILMAZ.

    Yaş spline düğümleri (`compute_rcs_knots()`) -- EN AZ bir varyant
    `use_age_spline=True` ise -- UPenn'in HAM `age` kolonundan (tüm
    kayıtlı hastalar, radyomik eşleşmesinden ÖNCE) BİR KEZ hesaplanır ve
    TÜM spline-kullanan varyantlara (hem UPenn eğitim hem UCSF harici
    test tarafında) AYNEN uygulanır -- sızıntı önleme (görev talimatı).
    """

    needs_spline = any(config.use_age_spline for config in variants)
    age_knots = (
        compute_rcs_knots(upenn_patients[PATIENT_RAW_AGE_COLUMN].astype(float))
        if needs_spline
        else None
    )

    summary_rows: list[dict[str, Any]] = []
    for config in variants:
        print(
            f"[v2/{config.name}] basliyor -- bolgeler={config.regions} "
            f"mgmt={config.include_mgmt} spline={config.use_age_spline}",
            flush=True,
        )
        result = run_single_variant(
            config=config,
            upenn_long=upenn_long,
            upenn_patients=upenn_patients,
            ucsf_long=ucsf_long,
            ucsf_patients=ucsf_patients,
            age_knots=age_knots,
            args=args,
        )
        summary_row = write_variant_outputs(output_dir=output_dir, result=result, args=args)
        summary_rows.append(summary_row)
        arm = result.arm_result
        print(
            f"[v2/{config.name}] n_train={result.n_train_patients} "
            f"n_events={result.n_train_events} ic_CV="
            f"{arm.nested_cv_fold_results['c_index'].mean():.4f} "
            f"fallback_folds={arm.n_fallback_folds}/{args.outer_splits} "
            f"final_ozellik(radyomik)={len(arm.final_model.final_features)} "
            f"UCSF n={result.external_report.n_output_rows} "
            f"harici_c_index={arm.external_test['c_index']:.4f} "
            f"[{arm.external_test['ci_lower']:.4f}-{arm.external_test['ci_upper']:.4f}]",
            flush=True,
        )

    write_variant_comparison_table(output_dir, summary_rows)
    return 0


# =====================================================================
# 8.6) v3 -- Düşük-varyans + kolinearite filtresi (koordinatör görevi,
# 2026-08-18: "düşük varyans filtresini de v3'e ekle, hazırlığını yap")
# =====================================================================
#
# BU BÖLÜM TAMAMEN EKLEMELİDİR (additive) -- yukarıdaki V2_VARIANT_
# CONFIGS, VariantConfig, run_single_variant(), run_v2_variant_suite(),
# select_v2_variant_configs(), write_variant_outputs(), write_variant_
# comparison_table() BİREBİR KORUNUR, TEK SATIR DEĞİŞTİRİLMEDİ. Mevcut
# 4 varyantın (v1_referans/v2a_mgmt/v2b_mgmt_spline/v2c_mgmt_spline_
# wttc) davranışı bu bölümden HİÇ ETKİLENMEZ (regresyon kanıtı:
# `tests/test_train_cox_week3.py`, TÜM eski testler değişmeden geçiyor).
#
# GEREKÇE (gerçek UPenn 611/585 verisiyle bu görevi yazan ajan
# tarafından ÖLÇÜLDÜ, bkz. `pipeline/reduce_collinearity.py` modül
# docstring'i -- koordinatörün raporuyla BAĞIMSIZ olarak doğrulandı):
# `week3_v2a_mgmt_final_coefficients.csv`'deki `WT__original_gldm_
# SmallDependenceLowGrayLevelEmphasis` (coef=98,6, se=106,8,
# ConvergenceWarning) İKİ AYRI sorunu BİRDEN yansıtıyordu:
#   (a) ÖLÇEK/SAYISAL KIRILGANLIK -- standardizasyon ConvergenceWarning'i
#       TAMAMEN gideriyor (0'a düşüyor) ama se/|coef| ORANINI (istatistiksel
#       belirsizlik) DEĞİŞTİRMİYOR (matematiksel olarak beklenen -- bkz.
#       `pipeline.cox_model.standardize_columns_fold_safe()` docstring'i).
#       Bu özelliğin KENDİ VIF'i (v2a final modelinde) sadece 2,43 --
#       kolinearite bu ÖZEL katsayının asıl sebebi DEĞİL, basitçe ZAYIF
#       bir sinyal taşıyor.
#   (b) AYRI ve GERÇEK bir kolinearite sorunu AYNI modelde VAR: `glszm_
#       SizeZoneNonUniformityNormalized<->glszm_SmallAreaEmphasis`
#       |r|=0,9964 (VIF≈172) -- tüm tasarım matrisini kötü-koşullandırıyor.
#   (c) GERÇEKTEN near-constant bir özellik de (`glcm_Idmn`, CV=0,0032)
#       aday havuzda (STABLE_FEATURES_ICC60 içinde, ICC filtresi bunu
#       YAKALAMIYOR) -- ama bu HAM std ile DEĞİL CV ile doğru tespit
#       ediliyor (ham std'ye göre `ngtdm_Coarseness` de küçük ama CV'si
#       93 özellik içinde neredeyse EN YÜKSEK -- yani GERÇEKTEN değişken,
#       yanlışlıkla elenmemeli).
#
# v3 HER ÜÇÜNÜ de HEDEFLER:
#   1. Radyomik özellikler `run_nested_cv()`'ye `clinical_standardize_
#      columns` üzerinden FOLD-GÜVENLİ standardize edilir -- bu parametre
#      ZATEN JENERİK (hangi kolon "klinik" hangisi "radyomik" ayrımı
#      YAPMIYOR, bkz. `pipeline.cox_model.run_nested_cv()` imzası), bu
#      yüzden (a) `pipeline/cox_model.py`'ye TEK SATIR dokunmadan çözülür.
#   2. Aday havuz (`STABLE_FEATURES_ICC60`) nested-CV'den ÖNCE, TEK
#      SEFER (ICC filtresiyle AYNI mimari kademe -- X-ONLY, çıktıya HİÇ
#      bakmaz, bkz. `pipeline.reduce_collinearity.build_v3_candidate_
#      pool()` docstring'indeki kapsam notu) `pipeline.reduce_
#      collinearity.build_v3_candidate_pool()` ile daraltılır -- (b) ve
#      (c) bu adımda elenir.
#
# Koordinatörün açık sorusuna yanıt ("v1_referans mı v2a_mgmt mı? ...
# ya da ikisinin de v3 versiyonunu tanımla" -- İKİNCİ seçenek
# uygulandı): v2a_mgmt somut problemin GÖZLEMLENDİĞİ varyant (iç CV
# 0,646->0,667 ARTMIŞ, harici 0,678->0,662 DÜŞMÜŞ -- klasik eğitim-
# üzerinde-iyileşme/harici-testte-kötüleşme deseni, coef=98,6
# bulgusunun kaynağı) -- filtrenin BURADA işe yarayıp yaramadığı EN
# İLGİLİ soru. v1_referans ise filtrenin, sorunun GÖRÜLMEDİĞİ bir
# varyantta ZARAR VERİP VERMEDİĞİNİ (sağlıklı bir modelde gereksiz
# özellik kaybı/harici performans düşüşü var mı) gösteren bir SANITY-
# CHECK kontrol koludur.


@dataclass
class V3VariantConfig:
    """v1/v2a/v2b/v2c'nin (`VariantConfig`) kovaryat/bölge/yaş-formu
    tanımını `base` üzerinden REFERANS ALIR (KOPYALAMAZ, sadece OKUR) --
    v3'ün TEK farkı radyomik aday havuzunun düşük-varyans+kolinearite
    filtresinden geçmiş olması + radyomiklerin fold-güvenli standardize
    edilmesidir."""

    name: str
    base: VariantConfig
    cv_threshold: float = V3_DEFAULT_CV_THRESHOLD
    corr_threshold: float = V3_DEFAULT_CORRELATION_CLUSTER_THRESHOLD
    standardize_radiomics: bool = True


V3_VARIANT_CONFIGS: tuple[V3VariantConfig, ...] = (
    V3VariantConfig(name="v3a_lowvar_v1referans", base=V2_VARIANT_CONFIGS[0]),
    V3VariantConfig(name="v3b_lowvar_v2amgmt", base=V2_VARIANT_CONFIGS[1]),
)
V3_VARIANT_NAMES: tuple[str, ...] = tuple(config.name for config in V3_VARIANT_CONFIGS)

#: 2026-09-11 EKLENDİ (Barış onayı, FİNAL KOŞU -- MODEL-SECIM-ANALIZI-
#: 2026-09-11.md §3/§4.2/§4.3). `v3b_lowvar_v2amgmt`'in AYNI v3
#: filtresini (near-constant CV<0,02 + kolinearite |r|>=0,95 kümeleme +
#: fold-güvenli radyomik standardizasyon), v2c'nin geç-ufuk sinyalinin
#: (AUC730 0,753) EPV 2,99/katsayı-patlaması KİRLİLİĞİNDEN arındırılmış
#: hâlde test etmek için, `V2_VARIANT_CONFIGS_EXTRA`'daki YENİ tabana
#: (WT+TC + MGMT + doğrusal yaş -- v2c'den FARKLI: spline YOK) uygular.
#: `V3_VARIANT_CONFIGS`/`V3_VARIANT_NAMES` (v3a/v3b) SATIR SATIR
#: DEĞİŞMEDEN KORUNUR -- bu AYRI, EK bir tuple. `cv_threshold`/
#: `corr_threshold`/`standardize_radiomics` KASITLI OLARAK varsayılan
#: (v3a/v3b ile AYNI ızgara) -- görev talimatı "penalizer/l1-ratio
#: ızgarası MEVCUT varyantlarla AYNI" diyor, burada elle sabitleme YOK.
V3_VARIANT_CONFIGS_EXTRA: tuple[V3VariantConfig, ...] = (
    V3VariantConfig(name="v3c_lowvar_wttc_mgmt_nospline", base=V2_VARIANT_CONFIGS_EXTRA[0]),
)
V3_VARIANT_NAMES_EXTRA: tuple[str, ...] = tuple(config.name for config in V3_VARIANT_CONFIGS_EXTRA)
#: `select_v3_variant_configs()`/CLI dispatch'in kullandığı GENİŞLETİLMİŞ
#: liste (2 orijinal + 1 ek) -- `V3_VARIANT_CONFIGS`'İN KENDİSİ hâlâ
#: yalnız 2 eleman.
ALL_V3_VARIANT_CONFIGS: tuple[V3VariantConfig, ...] = V3_VARIANT_CONFIGS + V3_VARIANT_CONFIGS_EXTRA


def select_v3_variant_configs(
    names: list[str] | None, *, configs: tuple[V3VariantConfig, ...] = V3_VARIANT_CONFIGS
) -> tuple[V3VariantConfig, ...]:
    """`select_v2_variant_configs()` ile AYNI desen (isim doğrulama,
    bilinmeyen ad SESSİZCE yoksayılmaz)."""

    if names is None:
        return configs
    lookup = {config.name: config for config in configs}
    unknown = [name for name in names if name not in lookup]
    if unknown:
        raise ValueError(
            f"Bilinmeyen v3 varyant adi(lar): {unknown} -- gecerli "
            f"adlar: {list(lookup)}"
        )
    return tuple(lookup[name] for name in names)


def build_v3_radiomic_feature_pool(
    upenn_wide: pd.DataFrame,
    raw_candidate_columns: list[str],
    *,
    cv_threshold: float,
    corr_threshold: float,
) -> tuple[list[str], V3CandidatePoolReport]:
    """`pipeline.reduce_collinearity.build_v3_candidate_pool()`'un ince
    sarmalayıcısı -- `upenn_wide[raw_candidate_columns]` (SADECE X,
    `survival_days`/`event` bu noktada henüz frame'e HİÇ JOIN EDİLMEDİ)
    üzerinde çalışır. `raw_candidate_columns` tipik olarak `select_
    feature_columns(upenn_wide, STABLE_FEATURES_ICC60, region=...)`
    çıktısıdır -- ICC filtresi ZATEN uygulanmış bir havuzu GİRDİ alır,
    bu fonksiyon ICC'nin YERİNE GEÇMEZ, ONU DARALTIR."""

    return build_v3_candidate_pool(
        upenn_wide[raw_candidate_columns],
        raw_candidate_columns,
        cv_threshold=cv_threshold,
        corr_threshold=corr_threshold,
    )


def drop_declared_shortfall_patients_for_pool(
    upenn_wide: pd.DataFrame, regions: tuple[str, ...]
) -> pd.DataFrame:
    """`run_single_v3_variant()`'ın v3 aday-havuz ADIMINA giden `upenn_
    wide`'ı hazırlar -- `regions` için `DECLARED_REGION_SHORTFALLS`'ta
    beyan edilmiş bir azalma VARSA, o hastaları (SADECE bu havuz
    hesabından, `upenn_wide`'ın kendisi/`training_frame` ETKİLENMEZ)
    açıkça düşürür; YOKSA `upenn_wide`'ı OLDUĞU GİBİ döndürür.

    2026-09-11 DÜZELTMESİ (reviewer v3c çapraz incelemesi, HIGH bulgusu)
    -- ayrı bir fonksiyon olarak çıkarılmasının nedeni doğrudan birim
    testi: `tests/test_train_cox_week3.py` bu fonksiyonu hem sentetik
    hem gerçek (`final_v3c/_cache/upenn_long.pkl`) veriyle çağırıp
    `pipeline.reduce_collinearity.build_v3_candidate_pool()`'un yeni
    `NonFiniteFeatureValueError` kapısını ARTIK tetiklemediğini kanıtlar.

    NEDEN GÜVENLİ: `DECLARED_REGION_SHORTFALLS`'taki her giriş, o hasta
    kümesinin bu bölge kombinasyonunda TÜM özellik kolonlarının NaN
    olduğunu (ör. TC = ET+NC = 0 -> TC bloğunun 93 kolonu NaN) belgeler
    -- yani bu hastalar zaten `build_training_frame()` tarafından NİHAİ
    eğitim çerçevesinden KENDİ BAŞINA düşürülecekler (`check_training_
    pool_counts()` bunu ayrıca doğrular). Burada ERKEN düşürülmeleri
    yalnız aday-havuz istatistiklerinin (CV/korelasyon) NaN'sız
    hesaplanabilmesi içindir -- pandas'ın eskiden ÖRTÜK skipna/pairwise-
    complete-obs davranışının YERİNE geçen, açık/izlenebilir bir eşdeğer
    (`tests/test_train_cox_week3.py::
    test_drop_declared_shortfall_patients_for_pool_reproduces_historical_v3c_pool_report`
    gerçek C32 verisiyle bunun HAVUZ SONUCUNU DEĞİŞTİRMEDİĞİNİ -- near_
    constant 4, correlation_cluster_duplicate 75, final 107 -- bit-
    birebir doğruluyor).

    `errors="raise"` bilinçli: beyan edilen bir kimlik `upenn_wide.
    index`'te YOKSA bu, beyanın artık veriyi DOĞRU tanımlamadığı
    anlamına gelir -- sessizce `errors="ignore"` ile geçmek, bu
    fonksiyonun önlemeye çalıştığı "beyan/veri uyuşmazlığını sessizce
    yutma" hatasını tekrar eder; erken/açık bir `KeyError` teşhis-
    dostudur (aşağıdaki `check_training_pool_counts()` da SERT durur)."""

    shortfall = resolve_declared_region_shortfall(regions)
    if shortfall is None:
        return upenn_wide

    dropped_ids = list(shortfall.dropped_patient_ids)
    print(
        "v3 ADAY HAVUZU HESABI -- beyan edilmiş hastalar SADECE bu "
        "hesaptan (near-constant/korelasyon istatistikleri) düşürüldü, "
        f"nihai `training_frame` etkilenmedi: {dropped_ids} "
        f"(bölgeler={'+'.join(regions)}, beyan tarihi={shortfall.declared_on}).",
        file=sys.stderr,
        flush=True,
    )
    return upenn_wide.drop(index=dropped_ids, errors="raise")


def run_single_v3_variant(
    *,
    config: V3VariantConfig,
    upenn_long: pd.DataFrame,
    upenn_patients: pd.DataFrame,
    ucsf_long: pd.DataFrame,
    ucsf_patients: pd.DataFrame,
    age_knots: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[VariantRunResult, V3CandidatePoolReport]:
    """`run_single_variant()` ile BİREBİR AYNI iskelet -- TEK fark:
    (1) radyomik `feature_columns` doğrudan `select_feature_columns()`
    çıktısı DEĞİL, `build_v3_radiomic_feature_pool()`'dan geçmiş
    (daraltılmış) hâli; (2) `config.standardize_radiomics=True` ise bu
    daraltılmış radyomik kolonlar da `clinical_standardize_columns`'a
    EKLENİR (fold-güvenli z-score); (3) döndürülen `VariantRunResult.
    config` KASITLI OLARAK `config.base`'in KENDİSİ DEĞİL, `name=
    config.name` (v3 adı) ile kurulmuş YENİ bir `VariantConfig` --
    aksi hâlde `write_variant_outputs()` `result.config.name`'i (örn.
    "v1_referans") dosya adı olarak kullanır ve V2 koşusunun GERÇEK
    `week3_v1_referans_*.csv` çıktısının ÜZERİNE YAZARDI (kritik
    çakışma önleme düzeltmesi).

    `run_single_variant()` DEĞİŞTİRİLMEDİ, ondan bağımsız YENİ bir
    fonksiyondur -- kod tekrarı BİLİNÇLİ (hot dosyanın ÇALIŞAN kısmına
    dokunmama önceliği, bkz. bu bölümün başlık notu).

    ⚠️ 2026-09-11 DÜZELTMESİ (reviewer v3c çapraz incelemesi, HIGH):
    `build_v3_radiomic_feature_pool()` eskiden `upenn_wide`'ın TAMAMI
    (611 satır) üzerinde çağrılıyordu -- çok-bölgeli (örn. WT+TC)
    varyantlarda `DECLARED_REGION_SHORTFALLS`'ta beyan edilmiş 2 hasta
    (TC=ET+NC=0 -> TC bloğunun 93 kolonu NaN) HÂLÂ frame'deydi. v3c ilk
    koşulduğunda `pipeline.reduce_collinearity.build_v3_candidate_pool()`
    'un bugünkü katı NaN/±inf kapısı (`_assert_finite_features`) henüz
    yoktu; pandas'ın örtük `skipna=True`/pairwise-complete-obs davranışı
    bu 2 satırı SESSİZCE dışladığı için sonuç GEÇERLİ çıktı. Guard
    eklendikten sonra AYNI çağrı `NonFiniteFeatureValueError` ile
    çökmeye başladı -- kod kendi ürettiği artefaktı yeniden üretemez
    hâle gelmişti. Düzeltme artık `drop_declared_shortfall_patients_
    for_pool()`'da (bkz. o fonksiyonun docstring'i + `tests/test_train_
    cox_week3.py::test_drop_declared_shortfall_patients_for_pool_
    reproduces_historical_v3c_pool_report`) -- pool hesabına giden
    `upenn_wide` kopyasından (SADECE bu hesap için -- aşağıdaki
    `training_frame` hâlâ ORİJİNAL `upenn_wide`'dan kuruluyor,
    `build_training_frame()` bu hastaları KENDİ BAŞINA ayrıca düşürüp
    609/583'e iniyor) beyan edilmiş hastalar açıkça düşürülür. WT-only
    kollarda davranış DEĞİŞMEZ."""

    upenn_wide, upenn_pivot_report = pivot_radiomics_long_to_wide(
        upenn_long, regions=list(config.base.regions)
    )
    ucsf_wide, ucsf_pivot_report = pivot_ucsf_regions_long_to_wide(
        ucsf_long, regions=config.base.regions
    )

    raw_candidate_columns: list[str] = []
    for region in config.base.regions:
        raw_candidate_columns += select_feature_columns(
            upenn_wide, STABLE_FEATURES_ICC60, region=region
        )

    upenn_wide_for_pool = drop_declared_shortfall_patients_for_pool(
        upenn_wide, config.base.regions
    )

    feature_columns, pool_report = build_v3_radiomic_feature_pool(
        upenn_wide_for_pool,
        raw_candidate_columns,
        cv_threshold=config.cv_threshold,
        corr_threshold=config.corr_threshold,
    )
    if not feature_columns:
        raise RuntimeError(
            f"v3 kolu {config.name!r}: filtre TÜM radyomik aday havuzunu "
            "elemiş -- cv_threshold/corr_threshold gözden geçirilmeli, "
            "sessizce boş bir havuzla devam EDİLMEDİ."
        )

    extra_columns = build_variant_clinical_extra_columns(
        include_mgmt=config.base.include_mgmt, use_age_spline=config.base.use_age_spline
    )

    upenn_clinical = build_variant_clinical_frame(
        upenn_patients,
        include_mgmt=config.base.include_mgmt,
        use_age_spline=config.base.use_age_spline,
        age_knots=age_knots,
    )
    ucsf_clinical = build_variant_clinical_frame(
        ucsf_patients,
        include_mgmt=config.base.include_mgmt,
        use_age_spline=config.base.use_age_spline,
        age_knots=age_knots,
    )
    upenn_patients_c = upenn_patients.join(upenn_clinical)
    ucsf_patients_c = ucsf_patients.join(ucsf_clinical)

    training_frame, training_report = build_training_frame(
        upenn_wide[feature_columns],
        upenn_patients_c,
        check_combat_identity=True,
        passthrough_columns=extra_columns,
    )
    n_train_events = int(training_frame["event"].sum())
    # 2026-08-19: bolge-farkinda kapi (bkz. `run_single_variant()`
    # icindeki ayni cagri). v3 kollari `config.base.regions`'i AYNEN
    # devralir -- v3a/v3b su an WT-only, ama v2c tabanli bir v3 kolu
    # eklenirse kapi kendiliginden dogru davranir.
    check_training_pool_counts(
        training_report.n_output_rows,
        n_train_events,
        expected_patients=args.expected_patient_count,
        expected_events=args.expected_event_count,
        allow_mismatch=args.allow_unexpected_patient_count,
        regions=config.base.regions,
        dropped_patient_ids=all_dropped_patient_ids(training_report),
    )

    external_frame, external_frame_report = build_external_test_frame(
        ucsf_wide[feature_columns], ucsf_patients_c, passthrough_columns=extra_columns
    )
    check_external_test_pool_counts(
        len(external_frame),
        int(external_frame["event"].sum()),
        expected_patients=args.expected_ucsf_patient_count,
        expected_events=args.expected_ucsf_event_count,
        allow_mismatch=args.allow_unexpected_ucsf_count,
        label=f"UCSF harici test ({config.name})",
    )

    l1_ratio_grid = (
        tuple(args.l1_ratio_grid) if getattr(args, "l1_ratio_grid", None) else DEFAULT_L1_RATIO_GRID
    )
    penalizer_grid = (
        tuple(args.penalizer_grid) if getattr(args, "penalizer_grid", None) else DEFAULT_PENALIZER_GRID
    )
    clinical_age_columns = (
        [CLINICAL_AGE_RCS1_COLUMN, CLINICAL_AGE_RCS2_COLUMN]
        if config.base.use_age_spline
        else [CLINICAL_AGE_COLUMN]
    )
    clinical_standardize_columns = list(clinical_age_columns)
    if config.standardize_radiomics:
        # (a) ölçek/sayısal kırılganlık düzeltmesi -- bkz. bölüm
        # başlığındaki gerekçe notu. `run_nested_cv()`/
        # `standardize_columns_fold_safe()` bu listeyi JENERİK olarak
        # ele alır, "klinik"/"radyomik" ayrımı YAPMAZ.
        clinical_standardize_columns = feature_columns + clinical_age_columns

    arm_result = run_modeling_arm(
        config.name,
        training_frame,
        feature_columns,
        external_frame,
        stability_frequency_threshold=args.primary_threshold,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        n_bootstrap_stability=args.n_bootstrap_stability,
        n_bootstrap_external=args.n_bootstrap_external,
        seed=args.seed,
        l1_ratio_grid=l1_ratio_grid,
        penalizer_grid=penalizer_grid,
        extra_columns=extra_columns,
        extra_column_penalizer=args.clinical_extra_column_penalizer,
        clinical_standardize_columns=clinical_standardize_columns,
        run_external_test=True,
    )

    result = VariantRunResult(
        config=VariantConfig(
            name=config.name,
            regions=config.base.regions,
            include_mgmt=config.base.include_mgmt,
            use_age_spline=config.base.use_age_spline,
        ),
        arm_result=arm_result,
        upenn_pivot_report=upenn_pivot_report,
        ucsf_pivot_report=ucsf_pivot_report,
        training_report=training_report,
        external_report=external_frame_report,
        feature_columns=feature_columns,
        extra_columns=extra_columns,
        n_train_patients=training_report.n_output_rows,
        n_train_events=n_train_events,
        age_knots=None if age_knots is None else [float(k) for k in age_knots],
    )
    return result, pool_report


def write_v3_pool_report(
    output_dir: Path, config_name: str, pool_report: V3CandidatePoolReport
) -> pd.DataFrame:
    """v3'e ÖZEL ek çıktı -- hangi özelliğin hangi GEREKÇEYLE (near-
    constant/CV, ya da korelasyon-kümesi temsilcisi DEĞİL) düşürüldüğü
    (CLAUDE.md "sessiz filtreleme yasak" ilkesi). `write_variant_
    outputs()`'un zaten yazdığı `week3_<variant>_*` dosyalarına
    DOKUNMAZ, EK bir `week3_<variant>_v3_pool_report.csv` yazar."""

    rows: list[dict[str, Any]] = [
        {
            "reason": "near_constant_cv",
            "feature": feature,
            "kept_representative": None,
            "cluster_id": None,
        }
        for feature in pool_report.near_constant_dropped
    ]
    for _, row in pool_report.correlation_dropped.iterrows():
        rows.append(
            {
                "reason": "correlation_cluster_duplicate",
                "feature": row["dropped_feature"],
                "kept_representative": row["kept_representative"],
                "cluster_id": row["cluster_id"],
            }
        )
    report_frame = pd.DataFrame(
        rows, columns=["reason", "feature", "kept_representative", "cluster_id"]
    )
    report_frame.to_csv(output_dir / f"week3_{config_name}_v3_pool_report.csv", index=False)
    return report_frame


def write_v3_variant_comparison_table(
    output_dir: Path, summary_rows: list[dict[str, Any]]
) -> pd.DataFrame:
    """`write_variant_comparison_table()`'ın v3 karşılığı -- BİLİNÇLİ
    OLARAK o fonksiyon YENİDEN KULLANILMADI, çünkü dosya adı orada
    SABİT KODLANMIŞ (`week3_v2_variant_comparison.csv`) -- v3 onu
    çağırsaydı V2 koşusunun (şu an SÜREN) karşılaştırma tablosunun
    ÜZERİNE YAZARDI. Bu fonksiyon AYRI bir dosyaya
    (`week3_v3_variant_comparison.csv`) yazar."""

    comparison = pd.DataFrame(summary_rows)
    comparison.to_csv(output_dir / "week3_v3_variant_comparison.csv", index=False)
    return comparison


def run_v3_variant_suite(
    *,
    upenn_long: pd.DataFrame,
    upenn_patients: pd.DataFrame,
    ucsf_long: pd.DataFrame,
    ucsf_patients: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    variants: tuple[V3VariantConfig, ...] = V3_VARIANT_CONFIGS,
) -> int:
    """`run_v2_variant_suite()` ile AYNI orkestrasyon deseni -- SEÇİLEN
    v3 varyant(lar)ı SIRAYLA koşar, `write_variant_outputs()`'u (VAR
    OLAN, DEĞİŞTİRİLMEMİŞ fonksiyon) yeniden kullanır + `write_v3_pool_
    report()` ile ek filtre-provenance dosyası, `write_v3_variant_
    comparison_table()` ile AYRI (v2'yle ÇAKIŞMAYAN) bir karşılaştırma
    tablosu yazar."""

    needs_spline = any(config.base.use_age_spline for config in variants)
    age_knots = (
        compute_rcs_knots(upenn_patients[PATIENT_RAW_AGE_COLUMN].astype(float))
        if needs_spline
        else None
    )

    summary_rows: list[dict[str, Any]] = []
    for config in variants:
        print(
            f"[v3/{config.name}] basliyor -- base={config.base.name} "
            f"cv_threshold={config.cv_threshold} corr_threshold={config.corr_threshold}",
            flush=True,
        )
        result, pool_report = run_single_v3_variant(
            config=config,
            upenn_long=upenn_long,
            upenn_patients=upenn_patients,
            ucsf_long=ucsf_long,
            ucsf_patients=ucsf_patients,
            age_knots=age_knots,
            args=args,
        )
        summary_row = write_variant_outputs(output_dir=output_dir, result=result, args=args)
        write_v3_pool_report(output_dir, config.name, pool_report)
        summary_row["v3_n_radiomic_input"] = pool_report.n_input_features
        summary_row["v3_n_radiomic_kept"] = pool_report.n_kept_features
        summary_row["v3_n_near_constant_dropped"] = len(pool_report.near_constant_dropped)
        summary_row["v3_n_correlation_dropped"] = len(pool_report.correlation_dropped)
        summary_row["v3_base_variant"] = config.base.name
        summary_rows.append(summary_row)
        arm = result.arm_result
        print(
            f"[v3/{config.name}] n_train={result.n_train_patients} "
            f"n_events={result.n_train_events} "
            f"radyomik_havuz={pool_report.n_input_features}->{pool_report.n_kept_features} "
            f"ic_CV={arm.nested_cv_fold_results['c_index'].mean():.4f} "
            f"fallback_folds={arm.n_fallback_folds}/{args.outer_splits} "
            f"final_ozellik(radyomik)={len(arm.final_model.final_features)} "
            f"UCSF n={result.external_report.n_output_rows} "
            f"harici_c_index={arm.external_test['c_index']:.4f} "
            f"[{arm.external_test['ci_lower']:.4f}-{arm.external_test['ci_upper']:.4f}]",
            flush=True,
        )

    write_v3_variant_comparison_table(output_dir, summary_rows)
    return 0


# =====================================================================
# 9) CLI + main()
# =====================================================================


def _script_sha256() -> str:
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hafta 3 Cox PHM egitim orkestrasyonu. VARSAYILAN MOD "
            "(2026-08-18 Baris karari, kilitli): TEK MODEL -- WT x 93 "
            "radyomik (ICC>=0,60) + yas + cinsiyet + GTR + IDH1, tam 611 "
            "hasta / 585 olum olayi kohortu, harici test UCSF-PDGM "
            "(TCGA harici testten CIKARILDI). Kol karsilastirmasi "
            "YAPILMAZ. Eski 8-kollu TCGA-harici-testli karsilastirma "
            "`--legacy-multi-arm-tcga` ile hala erisilebilir (duyarlilik "
            "amacli, varsayilan DEGIL). GERCEK VERIYLE calistirilamaz -- "
            "C32 verisi DB'de 0 satir oldugu surece C32DataNotFoundError "
            "ile durur."
        )
    )
    parser.add_argument(
        "--sensitivity-threshold",
        type=float,
        default=0.7,
        help="Duyarlilik kolu stabilite frekans esigi (varsayilan 0,7 -- "
        "decisions/2026-08-14-stabilite-secimi-esigi.md).",
    )
    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=DEFAULT_STABILITY_FREQUENCY_THRESHOLD,
        help="Birincil stabilite frekans esigi (varsayilan 0,6, kilitli).",
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--n-bootstrap-stability", type=int, default=200)
    parser.add_argument("--n-bootstrap-external", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--l1-ratio-grid",
        type=float,
        nargs="+",
        default=None,
        help="Varsayilan: pipeline.cox_model.DEFAULT_L1_RATIO_GRID "
        "(0,3/0,5/0,7/1,0). Yalniz testte/hiz icin kucultmede kullan -- "
        "gercek kosuda protokolun kilitli grid'i degistirilmemeli.",
    )
    parser.add_argument(
        "--penalizer-grid",
        type=float,
        nargs="+",
        default=None,
        help="Varsayilan: pipeline.cox_model.DEFAULT_PENALIZER_GRID.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Varsayilan: <gbm-aid mert>/artifacts/week3/cox_model",
    )
    parser.add_argument(
        "--skip-full-107-arm",
        action="store_true",
        help="Tam-107 duyarlilik kolunu ATLA (varsayilan: kosulur).",
    )
    parser.add_argument(
        "--exploratory-cluster-stability",
        action="store_true",
        help="Kesifsel kume-duzeyi secim frekansi analizini de calistir "
        "(varsayilan KAPALI, Codex onerisi -- birincil sonucun yerini ALMAZ).",
    )
    parser.add_argument("--exploratory-cluster-corr-threshold", type=float, default=0.7)
    parser.add_argument("--expected-patient-count", type=int, default=EXPECTED_UPENN_TRAINING_PATIENTS)
    parser.add_argument("--expected-event-count", type=int, default=EXPECTED_UPENN_TRAINING_EVENTS)
    parser.add_argument(
        "--allow-unexpected-patient-count",
        action="store_true",
        help="Olculen hasta/olay sayisi 611/585 ile uyusmazsa SERT durmak "
        "yerine UYARIYLA devam et.",
    )
    parser.add_argument(
        "--expected-ucsf-patient-count",
        type=int,
        default=EXPECTED_UCSF_EXTERNAL_TEST_PATIENTS,
        help="2026-08-18 EKLENDİ: beklenen UCSF harici test hasta sayısı. "
        "Varsayılan 295 -- decisions/2026-08-18-tek-model-ucsf-harici-"
        "test-k15-kapanisi.md ile KİLİTLENDİ (grade 4 + IDH filtresi "
        "YOK + diskte T1c+seg tam olan; 367/223 ve 275/165 GEÇERSİZ "
        "kılındı -- tanım evrimi, çelişki değil). Otorite hâlâ imaging-"
        "agent'in çıkarım anında dondurduğu liste; sayı tutmazsa DUR.",
    )
    parser.add_argument(
        "--expected-ucsf-event-count",
        type=int,
        default=EXPECTED_UCSF_EXTERNAL_TEST_EVENTS,
        help="2026-08-18 EKLENDİ: beklenen UCSF harici test olay sayısı "
        "(varsayılan 169, bkz. --expected-ucsf-patient-count notu).",
    )
    parser.add_argument(
        "--allow-unexpected-ucsf-count",
        action="store_true",
        help="Ölçülen UCSF hasta/olay sayısı beklenenle uyuşmazsa SERT "
        "durmak yerine UYARIYLA devam et.",
    )
    parser.add_argument(
        "--legacy-multi-arm-tcga",
        action="store_true",
        help="2026-08-18 EKLENDİ: VARSAYILAN DEĞİL. Eski 8-kollu (3 "
        "radyomik-yalnız + 5 klinik) karşılaştırma + TCGA harici test "
        "yolunu (`run_pipeline()`) çalıştırır -- bugünkü Barış kararı "
        "('TEK MODEL, kol karşılaştırması YAPILMAYACAK', 'TCGA harici "
        "test setinden ÇIKARILDI') bu bayrak OLMADAN geçerlidir. Yalnız "
        "gelecekte duyarlılık amaçlı gerekirse kullanılmalı.",
    )
    parser.add_argument(
        "--ignore-arm-checkpoints",
        action="store_true",
        help="Kaydedilmis kol sonuclarini YOK SAY ve hepsini yeniden hesapla. "
        "Kod/veri degistiginde kullanilir -- aksi halde eski sonuc yuklenir.",
    )
    parser.add_argument(
        "--skip-clinical-arms",
        action="store_true",
        help="2026-08-15 EKLENDİ: klinik kovaryat kollarının (clinical_base/"
        "radiomics_clinical/radiomics_clinical_gtr/idh_sensitivity/"
        "who2021_wildtype) TAMAMINI ATLA (varsayılan: hepsi koşulur). "
        "Hız/geliştirme amaçlı -- gerçek koşuda kullanılmamalı.",
    )
    parser.add_argument(
        "--clinical-arms",
        nargs="+",
        default=None,
        choices=[config.name for config in CLINICAL_ARM_CONFIGS],
        help="2026-08-18 EKLENDİ: klinik kovaryat kollarından YALNIZ "
        "adı verilenleri koş (varsayılan: hepsi). `--skip-clinical-arms` "
        "ya hep ya hiç olduğu için eklendi. NOT (K15 KAPANDI, 2026-08-18): "
        "`idh_sensitivity` (515) ve `who2021_wildtype` (499) kolları "
        "NOS/NEC'in 96 hastasını DIŞLAYARAK çalışır -- K15'in '(b) "
        "eksik say ve düşür' seçeneğinin bir kalıntısı. K15 artık "
        "GÖSTERGE DEĞİŞKENİ deseniyle kapatıldı (bkz. `run_primary_"
        "single_model()`, tam 611 kohort kullanır) -- bu iki legacy kol "
        "SADECE eski 8-kollu `--legacy-multi-arm-tcga` yolunda, isteğe "
        "bağlı duyarlılık amaçlı hâlâ mevcut, birincil/tek modelin BİR "
        "PARÇASI DEĞİL.",
    )
    parser.add_argument(
        "--clinical-extra-column-penalizer",
        type=float,
        default=0.0,
        help="2026-08-15 EKLENDİ: klinik kovaryatların (yaş/cinsiyet/GTR/IDH) "
        "aldığı ceza (varsayılan 0,0 -- PENALİZE EDİLMEZ, glmnet'in "
        "penalty.factor=0 ile 'zorunlu/forced-in' kovaryat pratiğiyle AYNI). "
        "KİLİTLENDİ 2026-09-13 (Barış onayı, karar A6): değer 0,0'dır. "
        "Gerekçe: kayıtlı TÜM kollar bu değerle koşuldu; geriye dönük "
        "değiştirmek her kolu yeniden koşmak demektir ve bunu gerektiren "
        "metodolojik bir bulgu YOKTUR. Detay: "
        "decisions/2026-09-13-extra-column-penalizer-sifir-onaylandi.md. "
        "*(ESKİ metin, tarihsel kayıt -- GEÇERSİZ: \"AÇIK TASARIM SORUSU "
        "(Barış'a soru, henüz doğrulanmadı): 'zayıf-penalize' (Codex "
        "önerisi) tam olarak 0,0 mı yoksa küçük pozitif bir değer mi olmalı "
        "-- protokolde SAYISAL olarak KİLİTLENMEDİ.\")*",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        # 2026-08-18 EKLENDİ (v3 hazırlığı): V3_VARIANT_NAMES SADECE
        # EKLENDİ -- V2_VARIANT_NAMES'İN kendisi/sırası/içeriği DEĞİŞMEDİ,
        # eski 4 isimle çağrılan HİÇBİR komutun davranışı etkilenmez.
        # 2026-09-11 EKLENDİ (final koşu): V2_VARIANT_NAMES_EXTRA (yeni
        # taban `v2d_mgmt_wttc_nospline`) + V3_VARIANT_NAMES_EXTRA (yeni
        # duyarlılık kolu `v3c_lowvar_wttc_mgmt_nospline`) SADECE
        # EKLENDİ -- aynı disiplin, eski isimlerin DAVRANIŞI DEĞİŞMEDİ.
        choices=list(V2_VARIANT_NAMES)
        + list(V3_VARIANT_NAMES)
        + list(V2_VARIANT_NAMES_EXTRA)
        + list(V3_VARIANT_NAMES_EXTRA),
        help=(
            "2026-08-18 EKLENDİ (Model v2): koşulacak varyant(lar) -- "
            f"{list(V2_VARIANT_NAMES)}'dan biri/birkaçı/hepsi. Verilirse "
            "`run_v2_variant_suite()` çalışır (`run_primary_single_model()` "
            "YERİNE) -- her varyant kendi week3_<variant>_*.csv setini "
            "üretir + TEK bir week3_v2_variant_comparison.csv. ŞEFFAFLIK "
            "KURALI: koşulan HER varyantın sonucu yazılır, hiçbiri "
            "gizlenmez (Barış, 2026-08-18 -- CLAUDE.md güncellendi, "
            "post-hoc varyant denemesi artık YASAK DEĞİL). Varsayılan "
            "None -- eski davranış (run_primary_single_model, TEK model, "
            "v1_referans'la matematiksel olarak AYNI) DEĞİŞMEZ. "
            f"2026-08-18 EK (v3): {list(V3_VARIANT_NAMES)} verilirse "
            "`run_v3_variant_suite()` çalışır -- düşük-varyans+kolinearite "
            "filtresi + radyomik fold-güvenli standardizasyon uygular, "
            "bkz. '8.6) v3' bölümü. v2 ve v3 isimleri AYNI komutta "
            "KARIŞTIRILAMAZ. 2026-09-11 EK (FİNAL KOŞU, Barış onayı, "
            "MODEL-SECIM-ANALIZI-2026-09-11.md): "
            f"{list(V2_VARIANT_NAMES_EXTRA)} (yeni taban, WT+TC + MGMT + "
            "doğrusal yaş) `run_v2_variant_suite()` yoluna, "
            f"{list(V3_VARIANT_NAMES_EXTRA)} (o tabanın v3-filtreli "
            "duyarlılık kolu) `run_v3_variant_suite()` yoluna girer -- "
            "ikisi de yukarıdaki karışık-liste kısıtına tabidir."
        ),
    )
    args = parser.parse_args(argv)
    # 2026-09-13: protokol-disi birincil esik (0,6 disinda) GURULTULU
    # uyari uretir -- kosu DURDURULMAZ (bkz. `warn_if_off_protocol_
    # primary_threshold` docstring'i). Uyari parse_args()'in SONUNDA
    # basilir, cunku bu modulun TUM giris yollari (CLI `main()` ve
    # `t3.parse_args([...])` ile arguman kuran export araclari) buradan
    # gecer -- tek bir yere koymak yeterli.
    warn_if_off_protocol_primary_threshold(args.primary_threshold)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or (PROJECT_ROOT / "artifacts" / "week3" / "cox_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    # === CHECKPOINT KATMANI 1: VERİ ÖNBELLEĞİ =========================
    # (2026-08-14, koordinatör) DB'den çekim UPenn'de ~11 dakika sürüyor
    # (3055 satır × 3 büyük JSONB, ağ-bağımlı). Koşu her yeniden
    # başlatıldığında bunu tekrarlamak saf kayıp. Önbellek diskte tutulur;
    # DB'deki satır sayısı DEĞİŞMİŞSE önbellek GEÇERSİZ sayılır (sessizce
    # eski veriyle çalışma riski yok).
    cache_dir = output_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _cached_fetch(cursor, *, tool: str, tag: str) -> pd.DataFrame:
        cache_path = cache_dir / f"{tag}_long.pkl"
        meta_path = cache_dir / f"{tag}_long.meta.json"
        # 2026-09-13/14 (karar 39): sayim artik `count_c32_radiomics_rows()`
        # ile YAPILIYOR -- fetch ile AYNI timepoint beyaz listesini uygular.
        # Eski hali (`SELECT COUNT(*) ... WHERE segmentation_tool = %s`,
        # timepoint kosulu YOK) bir post-op satir geldigi anda `len(frame)
        # != db_rows` uretip onbellegi SONSUZA KADAR gecersiz kilardi.
        db_rows = count_c32_radiomics_rows(cursor, segmentation_tool=tool)
        timepoint_fingerprint = list(allowed_timepoint_labels(tool))
        if cache_path.is_file() and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                # `timepoint_labels` anahtari ZORUNLU: kapidan ONCE yazilmis
                # onbellekler (anahtar YOK) filtresiz cekilmis olabilir ->
                # fail-closed, yeniden cekilir. Bugunun verisinde icerik
                # ozdes (UPenn C32 %100 pre-treatment) ama provenance
                # kanitini varsayima birakmiyoruz.
                if (
                    meta.get("segmentation_tool") == tool
                    and meta.get("db_rows") == db_rows
                    and meta.get("timepoint_labels") == timepoint_fingerprint
                ):
                    frame = pd.read_pickle(cache_path)
                    if len(frame) == db_rows:
                        print(f"[cache] {tag}: {db_rows} satır önbellekten okundu (DB sayımıyla eşleşti, timepoint beyaz listesi {timepoint_fingerprint}).", flush=True)
                        return frame
                if meta.get("timepoint_labels") != timepoint_fingerprint:
                    print(f"[cache] {tag}: önbellek TIMEPOINT KAPISINDAN ÖNCE yazılmış (meta.timepoint_labels={meta.get('timepoint_labels')!r} != {timepoint_fingerprint!r}) -- fail-closed, yeniden çekiliyor.", flush=True)
                else:
                    print(f"[cache] {tag}: önbellek GEÇERSİZ (DB {db_rows} satır) -- yeniden çekiliyor.", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[cache] {tag}: önbellek okunamadı ({type(exc).__name__}) -- yeniden çekiliyor.", flush=True)
        frame = fetch_c32_radiomics_long_frame(cursor, segmentation_tool=tool)
        frame.to_pickle(cache_path)
        meta_path.write_text(
            json.dumps(
                {
                    "segmentation_tool": tool,
                    "db_rows": db_rows,
                    "cached_rows": len(frame),
                    "timepoint_labels": timepoint_fingerprint,
                }
            ),
            encoding="utf-8",
        )
        return frame

    connection = get_connection(readonly=True)
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)

        upenn_long = _cached_fetch(cursor, tool=SEGMENTATION_TOOL_UPENN_C32, tag="upenn")
        raise_if_empty_c32(upenn_long, segmentation_tool=SEGMENTATION_TOOL_UPENN_C32)
        upenn_patients = fetch_patients_frame(cursor, source_name=UPENN_SOURCE_NAME)

        # 2026-08-18 Barış kararı: VARSAYILAN harici test kaynağı artık
        # TCGA DEĞİL UCSF-PDGM (TCGA'da GTR/IDH1 %0 dolu, model bu iki
        # kovaryatı zorunlu içerdiği için TCGA'ya uygulanamaz).
        # `--legacy-multi-arm-tcga` ile eski TCGA-harici-testli 8-kollu
        # yol (duyarlılık amaçlı) hâlâ çalıştırılabilir.
        if args.legacy_multi_arm_tcga:
            tcga_long = _cached_fetch(cursor, tool=SEGMENTATION_TOOL_TCGA_C32, tag="tcga")
            raise_if_empty_c32(tcga_long, segmentation_tool=SEGMENTATION_TOOL_TCGA_C32)
            tcga_patients = fetch_patients_frame(cursor, source_name=TCGA_SOURCE_NAME)
            cursor.close()
        else:
            ucsf_long = _cached_fetch(cursor, tool=SEGMENTATION_TOOL_UCSF_C32, tag="ucsf")
            raise_if_empty_c32(ucsf_long, segmentation_tool=SEGMENTATION_TOOL_UCSF_C32)
            ucsf_patients = fetch_patients_frame(cursor, source_name=UCSF_SOURCE_NAME)
            cursor.close()
    finally:
        connection.close()

    if args.legacy_multi_arm_tcga:
        return run_pipeline(
            upenn_long=upenn_long,
            upenn_patients=upenn_patients,
            tcga_long=tcga_long,
            tcga_patients=tcga_patients,
            args=args,
            output_dir=output_dir,
        )

    # 2026-08-18 EKLENDİ (Model v2): `--variants` verildiyse dört-varyant
    # orkestrasyonuna geç -- verilmediyse (varsayılan None) eski
    # davranış (TEK model, run_primary_single_model) DEĞİŞMEZ.
    #
    # 2026-08-18 EK (v3 hazırlığı, koordinatör görevi): `--variants`
    # ile bir v3 adı (`V3_VARIANT_NAMES`) verilirse `run_v3_variant_
    # suite()`'e yönlendirilir -- BU DAL v2 isimleri verildiğinde HİÇ
    # ÇALIŞMAZ, yukarıdaki `select_v2_variant_configs`/`run_v2_variant_
    # suite` çağrısı v2 isimleriyle BİREBİR AYNI KALDI (bit-birebir
    # regresyon). v2+v3 isimlerinin AYNI komutta KARIŞTIRILMASI
    # BİLİNÇLİ OLARAK desteklenmiyor (basitlik -- üretim zaten "dört
    # AYRI komut" deseniyle çalışıyor, v3 de kendi ayrı komutuyla
    # çalışacak) -- karışık liste verilirse AÇIKÇA hata verir, sessizce
    # bir alt-kümeyi yoksaymaz.
    if args.variants is not None:
        # 2026-09-11 EKLENDİ (final koşu): v3/v2 üyeliği kontrolü artık
        # ...NAMES_EXTRA'yı da KAPSAR (aksi hâlde yeni isimler burada
        # "ne v2 ne v3" görünüp aşağıdaki select_v2_variant_configs()
        # varsayılan (yalnız 4 orijinal) listesinde ValueError ile patlardı).
        is_v3_request = any(
            name in V3_VARIANT_NAMES or name in V3_VARIANT_NAMES_EXTRA for name in args.variants
        )
        is_v2_request = any(
            name in V2_VARIANT_NAMES or name in V2_VARIANT_NAMES_EXTRA for name in args.variants
        )
        if is_v3_request and is_v2_request:
            raise ValueError(
                f"--variants icinde HEM v2 ({list(V2_VARIANT_NAMES) + list(V2_VARIANT_NAMES_EXTRA)}) "
                f"HEM v3 ({list(V3_VARIANT_NAMES) + list(V3_VARIANT_NAMES_EXTRA)}) adlari karisik "
                f"verildi: {args.variants} -- desteklenmiyor, ayri komutlarla kosun."
            )
        if is_v3_request:
            # `configs=ALL_V3_VARIANT_CONFIGS` AÇIKÇA geçiliyor -- fonksiyonun
            # KENDİ varsayılanı (`V3_VARIANT_CONFIGS`, yalnız v3a/v3b) hâlâ
            # `select_v3_variant_configs(None)`/testler için DEĞİŞMEDEN duruyor.
            selected_v3_variants = select_v3_variant_configs(
                args.variants, configs=ALL_V3_VARIANT_CONFIGS
            )
            return run_v3_variant_suite(
                upenn_long=upenn_long,
                upenn_patients=upenn_patients,
                ucsf_long=ucsf_long,
                ucsf_patients=ucsf_patients,
                args=args,
                output_dir=output_dir,
                variants=selected_v3_variants,
            )
        # `configs=ALL_V2_VARIANT_CONFIGS` AÇIKÇA geçiliyor -- aynı gerekçe.
        selected_variants = select_v2_variant_configs(args.variants, configs=ALL_V2_VARIANT_CONFIGS)
        return run_v2_variant_suite(
            upenn_long=upenn_long,
            upenn_patients=upenn_patients,
            ucsf_long=ucsf_long,
            ucsf_patients=ucsf_patients,
            args=args,
            output_dir=output_dir,
            variants=selected_variants,
        )

    return run_primary_single_model(
        upenn_long=upenn_long,
        upenn_patients=upenn_patients,
        ucsf_long=ucsf_long,
        ucsf_patients=ucsf_patients,
        args=args,
        output_dir=output_dir,
    )


def run_pipeline(
    *,
    upenn_long: pd.DataFrame,
    upenn_patients: pd.DataFrame,
    tcga_long: pd.DataFrame,
    tcga_patients: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> int:
    """DB'den bagimsiz, tamamen DataFrame girdili orkestrasyon govdesi --
    `main()` DB'den ceker bu fonksiyona geciler, testler sentetik/fixture
    DataFrame'lerle DOGRUDAN bu fonksiyonu cagirir (bkz.
    tests/test_train_cox_week3.py)."""

    region = PRIMARY_REGIONS[0]
    assert PRIMARY_REGIONS == ("WT",), "Birincil bolge WT olmali (kilitli karar)."

    upenn_wide, upenn_pivot_report = pivot_radiomics_long_to_wide(
        upenn_long, regions=list(PRIMARY_REGIONS)
    )
    tcga_wide, tcga_pivot_report = pivot_radiomics_long_to_wide(
        tcga_long, regions=list(PRIMARY_REGIONS)
    )

    feature_sets: dict[str, list[str]] = {
        "wt93_icc60": select_feature_columns(upenn_wide, STABLE_FEATURES_ICC60, region=region),
    }
    if not args.skip_full_107_arm:
        feature_sets["wt107_full"] = select_feature_columns(
            upenn_wide, ALL_107_FEATURES, region=region
        )

    # 2026-09-14 (modeling-agent-S1): kol adindaki `th..` etiketi ARTIK
    # SABIT DIZGI DEGIL -- GERCEK esikten turetiliyor. Onceden ad sabit
    # `th06`/`th07` idi ama deger CLI'dan geliyordu; `--primary-threshold
    # 0.5` ile kosulan cikti `..._th06_...` adiyla yaziliyordu (aktif
    # YANLIS bilgi). `format_threshold_tag()` protokol varsayilanlarinda
    # (0,6/0,7) AYNI adi uretir -> diskteki artefaktlar ve onlara atif
    # veren kod (api/predict.py, demo/, shap_explainer_prototype.py)
    # KIRILMAZ.
    primary_tag = format_threshold_tag(args.primary_threshold)
    sensitivity_tag = format_threshold_tag(args.sensitivity_threshold)

    arm_configs: list[tuple[str, str, float]] = [
        (f"primary_wt93_icc60_{primary_tag}", "wt93_icc60", args.primary_threshold),
        (f"sensitivity_wt93_icc60_{sensitivity_tag}", "wt93_icc60", args.sensitivity_threshold),
    ]
    if not args.skip_full_107_arm:
        arm_configs.append(
            (f"sensitivity_wt107_full_{primary_tag}", "wt107_full", args.primary_threshold)
        )

    l1_ratio_grid = (
        tuple(args.l1_ratio_grid) if getattr(args, "l1_ratio_grid", None) else DEFAULT_L1_RATIO_GRID
    )
    penalizer_grid = (
        tuple(args.penalizer_grid) if getattr(args, "penalizer_grid", None) else DEFAULT_PENALIZER_GRID
    )

    # Kol-bazlı checkpoint dizini (bkz. aşağıdaki "CHECKPOINT KATMANI 2").
    # `output_dir` bu fonksiyona parametre olarak geliyor; `main()`'deki
    # tanımı burada GÖRÜNMEZ, o yüzden burada ayrıca kuruluyor.
    ckpt_dir = output_dir / "_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    arm_results: list[ArmResult] = []
    training_reports: dict[str, Any] = {}
    external_reports: dict[str, Any] = {}
    gate_checked = False

    for arm_name, feature_set_key, threshold in arm_configs:
        feature_columns = feature_sets[feature_set_key]

        training_frame, training_report = build_training_frame(
            upenn_wide[feature_columns],
            upenn_patients,
            check_combat_identity=True,
        )
        training_reports[arm_name] = training_report

        if not gate_checked:
            # Kapi kontrolu bir kez (ilk arm'in havuzuyla) yeterli -- ayni
            # patients_frame/feature-varlik filtresi tum arm'larda ayni
            # hasta kumesini uretmeli (93 vs 107 ozellik sadece kolon
            # sayisini degistirir, NaN paterni farkli DEGILSE ayni n
            # cikar). Farkli cikarsa asagida ayrica raporlanir.
            n_events = int(training_frame["event"].sum())
            check_training_pool_counts(
                training_report.n_output_rows,
                n_events,
                expected_patients=args.expected_patient_count,
                expected_events=args.expected_event_count,
                allow_mismatch=args.allow_unexpected_patient_count,
            )
            gate_checked = True

        tcga_feature_frame = tcga_wide[feature_columns]
        external_frame, external_frame_report = build_external_test_frame(
            tcga_feature_frame, tcga_patients
        )
        external_reports[arm_name] = external_frame_report

        # === CHECKPOINT KATMANI 2: KOL-BAZLI KAYIT ====================
        # (2026-08-14, koordinatör) Üretim ayarlarıyla bir kol ~1-2 saat
        # sürüyor (5 dış fold × [20 ızgara × 5 iç fold + 200 bootstrap]).
        # Bu ortamda arka plan süreçleri ölebiliyor (2026-08-14'te 4 kez
        # vurdu). Checkpoint YOKSA her ölüm = SIFIRDAN başlama.
        # Kol biter bitmez diske yazılır; yeniden başlatmada YÜKLENİR,
        # yeniden hesaplanmaz.
        #
        # DETERMİNİZM: `seed` sabit ve kol adına bağlı olmadığı için,
        # yüklenen kol ile yeniden hesaplanan kol AYNI sonucu verir --
        # checkpoint sayıyı DEĞİŞTİRMEZ, yalnız tekrar hesabı önler.
        #
        # 2026-09-14 (modeling-agent-S1) PARMAK IZI EKLENDI: onceden
        # YALNIZ "dosya var mi" bakiliyordu -- esik/seed/grid HIC
        # karsilastirilmiyordu. 0,5 ile kosulup yazilan bir kayit,
        # ertesi gun 0,6 ile kosuldugunda sessizce yuklenip "0,6
        # birincil" olarak raporlanabilirdi. Artik `main()`'deki VERI
        # onbelleginin fail-closed deseni (yan `*.meta.json`) BIREBIR
        # uygulaniyor.
        arm_ckpt = ckpt_dir / f"arm_{arm_name}.pkl"
        arm_fingerprint = build_arm_checkpoint_fingerprint(
            threshold=threshold,
            seed=args.seed,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            n_bootstrap_stability=args.n_bootstrap_stability,
            n_bootstrap_external=args.n_bootstrap_external,
            feature_set=feature_set_key,
            l1_ratio_grid=l1_ratio_grid,
            penalizer_grid=penalizer_grid,
        )
        arm_result = None
        # `getattr` KASITLI: `args` bu fonksiyona disaridan (testler dahil)
        # kuruluyor; eski cagrilarda bu bayrak YOK -- onceki kod
        # `and` kisa-devresi sayesinde kazara calisiyordu.
        if not getattr(args, "ignore_arm_checkpoints", False):
            arm_result = load_arm_checkpoint(
                arm_ckpt, arm_fingerprint, arm_name=arm_name
            )

        if arm_result is None:
            arm_result = run_modeling_arm(
                arm_name,
                training_frame,
                feature_columns,
                external_frame,
                stability_frequency_threshold=threshold,
                outer_splits=args.outer_splits,
                inner_splits=args.inner_splits,
                n_bootstrap_stability=args.n_bootstrap_stability,
                n_bootstrap_external=args.n_bootstrap_external,
                seed=args.seed,
                l1_ratio_grid=l1_ratio_grid,
                penalizer_grid=penalizer_grid,
            )
            save_arm_checkpoint(arm_ckpt, arm_result, arm_fingerprint)
            print(f"[checkpoint] {arm_name}: sonuç diske yazıldı ({arm_ckpt.name}).", flush=True)

        arm_results.append(arm_result)
        print(
            f"[{arm_name}] fold C-index ort={arm_result.nested_cv_fold_results['c_index'].mean():.4f} "
            f"fallback_folds={arm_result.n_fallback_folds}/{args.outer_splits} "
            f"final_ozellik={len(arm_result.final_model.final_features)} "
            f"harici_c_index={arm_result.external_test['c_index']:.4f} "
            f"[{arm_result.external_test['ci_lower']:.4f}-{arm_result.external_test['ci_upper']:.4f}]"
        )

    # =====================================================================
    # 2026-08-15 EKLENDİ -- klinik kovaryat kolları (görev talimatı,
    # önceden ilan edilmiş kollar): clinical_base / radiomics_clinical /
    # radiomics_clinical_gtr / idh_sensitivity / who2021_wildtype.
    # BİRİNCİL MODEL (yukarıdaki 3 kol) DEĞİŞMEDİ -- bu bölüm SADECE EK
    # kollar çalıştırır, yukarıdaki mantığa DOKUNMAZ.
    # =====================================================================
    if not args.skip_clinical_arms:
        selected_clinical_configs = select_clinical_arm_configs(args.clinical_arms)
        if selected_clinical_configs:
            print(
                "[klinik kollar] koşulacak: "
                + ", ".join(config.name for config in selected_clinical_configs),
                flush=True,
            )
            run_clinical_arms(
                upenn_wide=upenn_wide,
                upenn_patients=upenn_patients,
                tcga_wide=tcga_wide,
                tcga_patients=tcga_patients,
                feature_columns_wt93=feature_sets["wt93_icc60"],
                region=region,
                args=args,
                l1_ratio_grid=l1_ratio_grid,
                penalizer_grid=penalizer_grid,
                ckpt_dir=ckpt_dir,
                arm_results=arm_results,
                training_reports=training_reports,
                external_reports=external_reports,
                arm_configs=selected_clinical_configs,
            )

    exploratory_frame: pd.DataFrame | None = None
    if args.exploratory_cluster_stability:
        primary_arm = arm_results[0]
        primary_feature_columns = feature_sets["wt93_icc60"]
        primary_training_frame, _ = build_training_frame(
            upenn_wide[primary_feature_columns], upenn_patients, check_combat_identity=True
        )
        clusters = compute_correlation_clusters(
            primary_training_frame,
            primary_feature_columns,
            corr_threshold=args.exploratory_cluster_corr_threshold,
        )
        exploratory_frame = exploratory_cluster_selection_frequency(
            clusters, primary_arm.nested_cv_fold_results
        )

    write_outputs(
        output_dir=output_dir,
        arm_results=arm_results,
        upenn_pivot_report=upenn_pivot_report,
        tcga_pivot_report=tcga_pivot_report,
        training_reports=training_reports,
        external_reports=external_reports,
        exploratory_frame=exploratory_frame,
        args=args,
    )
    return 0


def write_outputs(
    *,
    output_dir: Path,
    arm_results: list[ArmResult],
    upenn_pivot_report: RegionPivotReport,
    tcga_pivot_report: RegionPivotReport,
    training_reports: dict[str, Any],
    external_reports: dict[str, Any],
    exploratory_frame: pd.DataFrame | None,
    args: argparse.Namespace,
) -> None:
    fold_frames = []
    frequency_frames = []
    external_rows = []

    for arm in arm_results:
        fold_copy = arm.nested_cv_fold_results.copy()
        fold_copy.insert(0, "arm", arm.name)
        fold_copy["selected_features"] = fold_copy["selected_features"].apply(
            lambda features: ";".join(sorted(features))
        )
        # 2026-09-14 (modeling-agent-S1): esik damgasi. Bu CSV COK KOLLU --
        # kollarin esikleri FARKLI (0,6 vs 0,7), yani sutun burada ayrica
        # gereklidir: dosyayi tek basina okuyan hangi satirin hangi esikten
        # geldigini kol adina GUVENMEDEN gorur.
        fold_copy["stability_frequency_threshold"] = arm.stability_frequency_threshold
        fold_frames.append(fold_copy)

        freq_copy = arm.selection_frequency_summary.copy()
        freq_copy.insert(0, "arm", arm.name)
        frequency_frames.append(freq_copy)

        # 2026-08-15 EKLENDİ (klinik kovaryat desteği): `getattr(..., varsayılan)`
        # KASITLI -- bugün (run #5) `_checkpoints/arm_*.pkl` olarak diske
        # yazılmış ESKİ `ArmResult` nesneleri bu iki alanı HİÇ İÇERMİYOR
        # (pickle dataclass varsayılanlarını değil kaydedilmiş `__dict__`'i
        # geri yükler) -- doğrudan `arm.extra_columns` erişimi bu eski
        # checkpoint'ler yüklendiğinde `AttributeError` fırlatırdı.
        arm_extra_columns = getattr(arm, "extra_columns", [])
        arm_run_external_test = getattr(arm, "run_external_test", True)
        external_rows.append(
            {
                "arm": arm.name,
                "stability_frequency_threshold": arm.stability_frequency_threshold,
                "n_final_features": len(arm.final_model.final_features),
                "final_features": ";".join(sorted(arm.final_model.final_features)),
                "used_fallback_full_pool": arm.final_model.used_fallback,
                "n_fallback_folds": arm.n_fallback_folds,
                "n_outer_folds": len(arm.nested_cv_fold_results),
                "clinical_extra_columns": ";".join(sorted(arm_extra_columns)),
                "run_external_test": arm_run_external_test,
                **arm.external_test,
            }
        )

    pd.concat(fold_frames, ignore_index=True).to_csv(
        output_dir / "week3_fold_results.csv", index=False
    )
    pd.concat(frequency_frames, ignore_index=True).to_csv(
        output_dir / "week3_selection_frequency.csv", index=False
    )
    pd.DataFrame(external_rows).to_csv(output_dir / "week3_external_test.csv", index=False)

    if exploratory_frame is not None:
        exploratory_frame.to_csv(
            output_dir / "week3_exploratory_cluster_stability.csv", index=False
        )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _script_sha256(),
        "method_name": METHOD_NAME,
        "segmentation_tool_upenn_c32": SEGMENTATION_TOOL_UPENN_C32,
        "segmentation_tool_tcga_c32": SEGMENTATION_TOOL_TCGA_C32,
        "primary_region": PRIMARY_REGIONS[0],
        "combat_identity_note": COMBAT_IDENTITY_NOTE,
        "expected_training_patient_count": args.expected_patient_count,
        "expected_training_event_count": args.expected_event_count,
        "allow_unexpected_patient_count": args.allow_unexpected_patient_count,
        "cli_args": vars(args) | {"output_dir": str(args.output_dir) if args.output_dir else None},
        "upenn_pivot_report": {
            "regions_requested": upenn_pivot_report.regions_requested,
            "n_input_rows": upenn_pivot_report.n_input_rows,
            "n_candidate_patients": upenn_pivot_report.n_candidate_patients,
            "n_output_patients": upenn_pivot_report.n_output_patients,
            "dropped_patients_missing_region": upenn_pivot_report.dropped_patients_missing_region,
        },
        "tcga_pivot_report": {
            "regions_requested": tcga_pivot_report.regions_requested,
            "n_input_rows": tcga_pivot_report.n_input_rows,
            "n_candidate_patients": tcga_pivot_report.n_candidate_patients,
            "n_output_patients": tcga_pivot_report.n_output_patients,
            "dropped_patients_missing_region": tcga_pivot_report.dropped_patients_missing_region,
        },
        "training_frame_reports": {
            arm_name: {
                "n_input_rows": report.n_input_rows,
                "n_output_rows": report.n_output_rows,
                "dropped_missing_duration": report.dropped_missing_duration,
                "dropped_missing_features": report.dropped_missing_features,
                "dropped_unrecognized_vital_status": report.dropped_unrecognized_vital_status,
                "sources": report.sources,
            }
            for arm_name, report in training_reports.items()
        },
        "external_frame_reports": {
            arm_name: {
                "n_input_rows": report.n_input_rows,
                "n_output_rows": report.n_output_rows,
                "dropped_missing_duration": report.dropped_missing_duration,
                "dropped_missing_features": report.dropped_missing_features,
                "dropped_unrecognized_vital_status": report.dropped_unrecognized_vital_status,
            }
            for arm_name, report in external_reports.items()
        },
        "arms": [
            {
                "name": arm.name,
                "n_feature_candidates": len(arm.feature_columns),
                "stability_frequency_threshold": arm.stability_frequency_threshold,
                "n_fallback_folds": arm.n_fallback_folds,
                "n_final_features": len(arm.final_model.final_features),
                "final_model_used_fallback": arm.final_model.used_fallback,
                "external_test": arm.external_test,
                # `getattr` NOTU icin bkz. yukarida external_rows -- ayni
                # geriye-donuk uyumluluk gerekcesi (eski checkpoint'ler).
                "clinical_extra_columns": getattr(arm, "extra_columns", []),
                "run_external_test": getattr(arm, "run_external_test", True),
            }
            for arm in arm_results
        ],
        "clinical_covariate_note": (
            "2026-08-15 EKLENDI: klinik kovaryatlar (yas/cinsiyet/GTR/IDH) "
            "extra_columns mekanizmasiyla ZORUNLU olarak modelde tutulur -- "
            "elastic-net/stabilite secimine hic GIRMEZ. Varsayilan "
            "extra_column_penalizer=0.0 (PENALIZE EDILMEZ) -- bu SAYISAL "
            "deger 2026-09-13'te A6 karariyla KILITLENDI (Baris onayi), "
            "klinik kovaryatlar forced-in. Detay: decisions/2026-09-13-"
            "extra-column-penalizer-sifir-onaylandi.md "
            "(bkz. --clinical-extra-column-penalizer yardimi). "
            "(eski metin: \"bu SAYISAL deger protokolde KILITLENMEDI, "
            "Barisin onayi bekleniyor.\")"
        ),
        "known_open_questions": [
            # 2026-09-15 DUZELTILDI (K23/BEKLEYEN-KARARLAR.md K5, Z5): bu
            # 4 girdiden 3'u KAPANMIS bir karari "acik/kapsam disi"
            # gosteriyordu (CLI/docstring guncellenmisti, bu metadata
            # yazicisi guncellenmemisti -- K22'deki AYNI desen). Eski
            # metin, provenance bozulmasin diye AYNEN korunuyor (parantez
            # icinde). NC/ED/ET girdisi DOKUNULMADI -- o hala gercekten
            # kapsam disi (K18 karariyla BILINCLI).
            "[KAPANDI 2026-09-13, karar K6] exploratory_cluster_corr_"
            "threshold=0.7 icin |r|>=0,95 esigiyle korelasyon kumeleri "
            "OLCULDU (93 aday -> 54, v3 havuzu). Detay: decisions/"
            "2026-09-13-k4-k5-k6-k12c-stabilite-sabitleri-kapanisi.md "
            "Bolum 3. (eski metin: \"exploratory_cluster_corr_threshold="
            "0.7 bu scriptin yazarinin koydugu bir varsayilan, protokolde "
            "kilitlenmedi.\")",
            "[KISMEN KAPANDI 2026-09-13, karar K5] WT+TC duyarlilik kolu "
            "bu script icinde iki kez KOSULDU (v2d_mgmt_wttc_nospline, "
            "v3c_lowvar_wttc_mgmt_nospline; egitim havuzu 609/583, "
            "611/585 DEGIL); TC-only ise HIC KOSULMADI, kapsam disi "
            "KALIR. Detay: decisions/2026-09-13-k4-k5-k6-k12c-stabilite-"
            "sabitleri-kapanisi.md Bolum 2. (eski metin: \"WT+TC / "
            "TC-only bolge duyarlilik kollari (decisions/.../Bolum 4.5) "
            "bu scriptin kapsaminda DEGIL -- ayri gorev.\")",
            "NC/ED/ET kesifsel katmani (Bolum 4.6) bu scriptin kapsaminda DEGIL.",
            "[KAPANDI, YAPILDI] Cox skorlarinin XGBoost'a out-of-fold "
            "aktarimi pipeline/xgboost_model.py "
            "(verify_aligned_fold_leakage_free) + tools/"
            "train_xgboost_week4.py'de UYGULANDI ve denetlendi (5/5 fold "
            "cox_fit_equals_outer_train=True VE "
            "cox_fit_disjoint_from_outer_test=True, week4_v2a_mgmt_"
            "aligned_fold_leakage_audit.csv). Bu script'in KENDISI hala "
            "yalniz Cox asamasini kapsar, XGBoost kodu bu dosyada DEGIL. "
            "(eski metin: \"Cox skorlarinin XGBoost'a out-of-fold "
            "aktarimi bu scriptin kapsaminda DEGIL -- ayri gorev.\")",
        ],
    }
    (output_dir / "week3_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        C32DataNotFoundError,
        TrainingPoolCountMismatchError,
        FallbackAuditMismatchError,
        C32WhitelistViolationError,
        ExternalTestCountMismatchError,
        UCSFRegionPivotError,
    ) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
