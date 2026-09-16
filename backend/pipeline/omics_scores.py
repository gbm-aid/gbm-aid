"""Omics skor motoru — ``tmz_resistance_score``, ``aggressiveness_score``,
``dna_repair_score`` ve bunların dondurulmuş sınıf eşikleri.

Kaynaklar:
  - `raw/mimari/v45.txt` satır 449-460 (skor formülleri), 471-492
    (donmuş tertil eşikleri, "Kritik Bulgu — Sınıflandırma Sınırlarının
    Donması Gerekiyor"), 1135-1160 (skor tanımları + backend clean enum).
  - `raw/veri/genom_verileri.xlsx`, "📖 Data Dictionary" sayfası (formül
    metni + kolon sözlüğü — v45.txt ile birebir).
  - `decisions/2026-08-18-omics-skor-formulleri-ve-esikler.md` — KARAR 1
    (score_thresholds eşikleri), KARAR 2 (tmz_class_relative), KARAR 3
    (dna_repair_score'un tersine mühendislikle çıkarılması).

⚠️ Bu dosya `pipeline/cox_model.py` VE `tools/train_cox_week3.py`'ye
DOKUNMAZ, oralardan hiçbir şey import ETMEZ ve oralardan hiçbir şey bu
dosyayı import ETMEZ — modeling-agent 2026-08-18'de o iki dosya üzerinde
paralel çalışıyordu (AKTIF-GOREVLER.md çakışma uyarısı).

Girdi sözleşmesi: ``omics_profiles`` tablosunun ham
``[gen]_methylation`` / ``[gen]_expression`` / ``[gen]_cnv`` kolonları
(küçük harfli, DB şemasıyla birebir — örn. ``mgmt_methylation``).
Çıktı: ``molecular_scores`` tablosunun ilgili skor + sınıf kolonlarına
birebir karşılık gelir; tüm formüller canlı DB'deki 48/48 satıra karşı
db-agent tarafından bağımsız olarak doğrulanmıştır (2026-08-18).

Not (hassasiyet): tüm ``compute_*`` fonksiyonları sonucu 2 ondalığa
yuvarlar (``molecular_scores``'ta skorlar 2 ondalıklı saklanıyor, örn.
"48.39"). ``classify_*`` fonksiyonlarının HER BİRİ de girdiyi kendi
içinde AYRICA 2 ondalığa yuvarlar (savunmacı programlama, 2026-08-18
Codex çapraz incelemesi BUG 1 düzeltmesi) — ham (yuvarlanmamış) float
ile p66=48.3900 gibi bir sınırda kayan-nokta gürültüsü tek bir
hastanın sınıfını değiştirebilirdi (canlı ölçüldü: TCGA-26-5134, ham
hesap 48.3905 vs DB'nin 48.39 — yuvarlamadan önce "resistant",
yuvarlamadan sonra doğru sınıf "intermediate"). Bu artık iki yerde de
(``compute_*`` VE ``classify_*``) korunuyor — `classify_*` DOĞRUDAN,
yuvarlanmamış bir skorla çağrılsa BİLE kural bozulmaz.
"""

from __future__ import annotations

from typing import Mapping

Number = float


def _clip100(x: float) -> float:
    return max(0.0, min(100.0, x))


# =============================================================================
# 1) tmz_resistance_score  (v45.txt satır 453 + Data Dictionary formül bloğu)
# =============================================================================


def compute_tmz_resistance_score(row: Mapping[str, Number]) -> float:
    """TMZ direnç skoru (0-100). Düşük = duyarlı, yüksek = dirençli.

    Formül (v45.txt satır 453, Excel Data Dictionary satır "tmz_resistance_score"):
        50 - (MGMT_met x 25) + (MSH6_met x 10) + (MLH1_met x 10)
           + (PMS2_met x 10) + max(0, 5 - MSH6_rna) x 3
        clip[0, 100]

    Doğrulama: `omics_profiles` JOIN `molecular_scores`, 48/48 satırda
    yuvarlama sonrası TAM eşleşme (db-agent, 2026-08-18).
    """
    mgmt_met = float(row["mgmt_methylation"])
    msh6_met = float(row["msh6_methylation"])
    mlh1_met = float(row["mlh1_methylation"])
    pms2_met = float(row["pms2_methylation"])
    msh6_rna = float(row["msh6_expression"])

    raw = (
        50.0
        - mgmt_met * 25.0
        + msh6_met * 10.0
        + mlh1_met * 10.0
        + pms2_met * 10.0
        + max(0.0, 5.0 - msh6_rna) * 3.0
    )
    return round(_clip100(raw), 2)


# Donmuş v1.0 tertil eşikleri (48 hastalık kohort, score_thresholds
# tablosuna 2026-08-18'de yazıldı — bkz. decisions/2026-08-18-omics-skor-
# formulleri-ve-esikler.md KARAR 1). Bu sabitler score_thresholds'taki
# satırlarla BİREBİR aynı olmalı; DB'yi otorite kabul eden bir çağıran
# kod bunun yerine tabloyu canlı okuyabilir, bu modül yalnız v1.0
# varsayılanını sağlıyor (STABLE_FEATURES_ICC60 emsaliyle aynı desen,
# bkz. pipeline/cox_model.py).
TMZ_RESISTANCE_P33_V1_0 = 46.5009
TMZ_RESISTANCE_P66_V1_0 = 48.3900


def classify_tmz_resistance(
    score: float,
    *,
    p33: float = TMZ_RESISTANCE_P33_V1_0,
    p66: float = TMZ_RESISTANCE_P66_V1_0,
) -> str:
    """`tmz_class` (backend clean enum: sensitive/intermediate/resistant).

    Donmuş tertil kuralı (v45.txt satır 471-492 "Kritik Bulgu"): sınırlar
    48 hastalık v1.0 referans kohortundan hesaplanmış ve DONDURULMUŞTUR;
    yeni hasta eklendikçe kohort YENİDEN HESAPLANMAZ (aksi halde eski
    hastaların sınıfı kayar — klinik karar destek sisteminde kabul
    edilemez).

    ⚠️ SAVUNMACI YUVARLAMA (2026-08-18, Codex çapraz incelemesi BUG 1):
    `score` burada YENİDEN 2 ondalığa yuvarlanır — `compute_tmz_resistance_
    score()` zaten yuvarlıyor olsa da, bu fonksiyon DOĞRUDAN ham/yuvarlanmamış
    bir float ile çağrılırsa (örn. başka bir kaynaktan gelen skor) sınır
    değere (örn. p66=48.3900) çok yakın kayan-nokta gürültüsü YANLIŞ sınıfa
    yol açabiliyordu — canlı kanıt: `classify_tmz_resistance(48.3905)` YUVARLAMA
    OLMADAN "resistant" dönerdi, DB'nin gerçek karşılığı (48.39, TCGA-26-5134
    ailesi) "intermediate". Artık HANGİ yoldan çağrılırsa çağrılsın kural
    bozulmaz.
    """
    score = round(score, 2)
    if score <= p33:
        return "sensitive"
    if score <= p66:
        return "intermediate"
    return "resistant"


# =============================================================================
# 2) aggressiveness_score  (v45.txt satır 456-458 + 486, 1143-1149)
# =============================================================================


def compute_aggressiveness_score(row: Mapping[str, Number]) -> float:
    """Agresiflik skoru (0-100, exploratory — ana karara dahil değil).

    Formül (v45.txt satır 456-458):
        30 + min(EGFR_rna, 12) x 2.5 + max(0, EGFR_cnv) x 8
           + PTEN_met x 10 + max(0, -CDKN2A_cnv) x 12 - TP53_met x 5
        clip[0, 100]

    Doğrulama: 48/48 satırda yuvarlama sonrası TAM eşleşme (db-agent,
    2026-08-18).
    """
    egfr_rna = float(row["egfr_expression"])
    egfr_cnv = float(row["egfr_cnv"])
    pten_met = float(row["pten_methylation"])
    cdkn2a_cnv = float(row["cdkn2a_cnv"])
    tp53_met = float(row["tp53_methylation"])

    raw = (
        30.0
        + min(egfr_rna, 12.0) * 2.5
        + max(0.0, egfr_cnv) * 8.0
        + pten_met * 10.0
        + max(0.0, -cdkn2a_cnv) * 12.0
        - tp53_met * 5.0
    )
    return round(_clip100(raw), 2)


def classify_aggressiveness(score: float) -> str:
    """`aggr_class` — SABİT mutlak eşik (<45/65/80), kod sabiti.

    v45.txt satır 486 + 1148-1149: bu skor için dondurma GEREKMEZ,
    eşikler mutlak/sabit.

    ŞEMA UYUMSUZLUĞU DÜZELTİLDİ (2026-08-18, Barış onayı: "aggr_class
    CHECK constraint düzeltmesini UYGULA"): `raw/veri/genom_verileri.xlsx`'in
    Data Dictionary'si ve bu fonksiyon "low/intermediate/high/very_high"
    (4 kategori) tanımlıyordu, ama DB'nin `molecular_scores_aggr_class_check`
    CHECK constraint'i başlangıçta yalnız {'intermediate','high','very_high'}
    kabul ediyordu — 'low' DB şemasında YOKTU (48 hastalık kohortta hiçbir
    skor 45'in altına inmediği için bu hiç gözlemlenmemişti/test
    edilmemişti). Constraint genişletildi: `ALTER TABLE molecular_scores
    DROP CONSTRAINT molecular_scores_aggr_class_check` + `ADD CONSTRAINT
    ... CHECK (aggr_class = ANY (ARRAY['low','intermediate','high',
    'very_high']))` — tek transaction, önce 48/48 satırın yeni constraint'i
    geçtiği doğrulandı (daraltma değil, saf genişletme), sonra AYRI
    bağlantıyla canlı doğrulandı, sonra savepoint-izole fonksiyonel testle
    ('low' kabul + geçersiz değer hâlâ ret) kanıtlandı. Bu düzeltmeden
    önce bu fonksiyonun döndürdüğü 'low' değeri `molecular_scores`'a
    INSERT edilirken CHECK ihlaliyle reddediliyordu — artık REDDEDİLMİYOR.
    Rollback (kayıtta, uygulanmadı): `ALTER TABLE molecular_scores DROP
    CONSTRAINT molecular_scores_aggr_class_check; ALTER TABLE
    molecular_scores ADD CONSTRAINT molecular_scores_aggr_class_check
    CHECK (aggr_class = ANY (ARRAY['intermediate','high','very_high']));`

    ⚠️ SAVUNMACI YUVARLAMA (2026-08-18, Codex çapraz incelemesi BUG 1):
    bkz. `classify_tmz_resistance()` docstring'i — aynı gerekçe, aynı
    düzeltme. Canlı kanıt: `classify_aggressiveness(44.996)` YUVARLAMA
    OLMADAN "low" dönerdi, doğrusu (2 ondalığa yuvarlanmış 45.0 ile)
    "intermediate".
    """
    score = round(score, 2)
    if score < 45.0:
        return "low"
    if score < 65.0:
        return "intermediate"
    if score < 80.0:
        return "high"
    return "very_high"


# =============================================================================
# 3) dna_repair_score — TERSİNE MÜHENDİSLİKLE ÇIKARILDI
# =============================================================================
#
# Orijinal formül metni (v45.txt satır 459-460, Excel Data Dictionary):
#   "Ham: (-met×w + min(rna,10)×w) → z-score normalize → ×15+50 → clip[0,100]"
# HİÇBİR yerde hangi genlerin Σ'ya girdiği ve `w` ağırlıkları YAZMIYOR.
# Barış (kaynağı hazırlayan): "elimizdeki her şeyi oraya aktardım" — formülü
# üreten orijinal script kayıp. Koordinatör 48 hastanın mevcut skorlarından
# en küçük kareler regresyonuyla katsayıları geri çıkardı
# (R²=0,99999997, maks. sapma 0,0066); db-agent BAĞIMSIZ olarak aynı
# katsayılarla 48/48 hastayı ±0,01 içinde yeniden üretti (2026-08-18).
#
# 🔴 İKİ UYARI (orijinal formül metni EKSİK/YANILTICI, karar dosyasından):
#   1. Metin "ortak w" ima ediyor (aynı w hem met hem rna için) ama
#      gerçekte met/rna AYRI ağırlık alıyor — ortak-w varsayımı R²'yi
#      0,9995'e düşürüyor.
#   2. CHEK2'nin metilasyon terimi HİÇ kullanılmıyor (katsayı ≈ 0).

_DNA_REPAIR_GENES = ("msh2", "mlh1", "brca1", "brca2", "chek2")

_DNA_REPAIR_W_MET: dict[str, float] = {
    "msh2": 6.853,
    "mlh1": 6.991,
    "brca1": 8.752,
    "brca2": 8.766,
    "chek2": 0.000,  # metilasyon terimi kullanılmıyor (bkz. uyarı #2)
}
_DNA_REPAIR_W_RNA: dict[str, float] = {
    "msh2": 5.838,
    "mlh1": 5.837,
    "brca1": 7.000,
    "brca2": 7.001,
    "chek2": 4.667,
}
_DNA_REPAIR_CONST = -210.278

# FROZEN v1.0 referans kohort istatistikleri — HAM (normalize-öncesi)
# skorun 48 hastalık kohorttaki ortalama/std'si (ddof=0/nüfus std).
# db-agent tarafından yukarıdaki katsayılarla 48 hastanın ham skoru
# hesaplanıp bağımsız olarak türetildi (2026-08-18).
#
# ⚠️ MİMARİ BOŞLUĞU (bulundu, öneri — uygulanmadı): bu iki sabit,
# `score_thresholds`'taki P33/P66 veya `combat_parameters`'taki
# gamma/delta ile AYNI sınıftan bir "dondurulması gereken referans
# kohort istatistiği"dir ama şu an bunun için DB'de ayrı bir satır/tablo
# YOK — yalnız bu Python modülünde sabit kodlu. Yeni bir hasta için
# `dna_repair_score` hesaplanırken bu iki değer YENİDEN kohorttan
# hesaplanırsa (örn. gelecekte kohort büyüdükçe) TÜM eski hastaların
# skoru KAYAR — score_thresholds'un çözdüğü sorunun BİREBİR AYNISI.
# Bu modül bilinçli olarak bu iki değeri SABİT KODLUYOR (yeniden
# hesaplamıyor) ki bu riske düşülmesin; DB'de resmi bir dondurma
# mekanizması (yeni tablo/kolon) Barış'a ÖNERİLDİ, uygulanmadı.
DNA_REPAIR_RAW_MEAN_V1_0 = 50.007812
DNA_REPAIR_RAW_STD_V1_0 = 15.000797  # ddof=0 (nüfus std)


def _dna_repair_raw(row: Mapping[str, Number]) -> float:
    total = _DNA_REPAIR_CONST
    for gene in _DNA_REPAIR_GENES:
        met = float(row[f"{gene}_methylation"])
        rna = float(row[f"{gene}_expression"])
        total += -met * _DNA_REPAIR_W_MET[gene] + min(rna, 10.0) * _DNA_REPAIR_W_RNA[gene]
    return total


def compute_dna_repair_score(
    row: Mapping[str, Number],
    *,
    reference_mean: float = DNA_REPAIR_RAW_MEAN_V1_0,
    reference_std: float = DNA_REPAIR_RAW_STD_V1_0,
) -> float:
    """DNA onarım aktivite skoru (0-100, EXPLORATORY — ana karara dahil
    DEĞİL, v45.txt satır 1146-1147/1547-1549).

    ⚠️ ZORUNLU UYARI (karar dosyasından birebir, her kullanımda/raporda
    birlikte taşınmalı):
    "Bu katsayılar, orijinal formül tanımı (gen listesi ve `w`
    ağırlıkları) hiçbir kaynakta bulunamadığı için 48 hastanın mevcut
    skorlarından tersine mühendislikle çıkarılmıştır (R²=0,99999997,
    maksimum sapma 0,007 puan). Orijinal ağırlıklarla birebir aynı
    olduğu garanti edilemez."

    `reference_mean`/`reference_std`: v1.0 referans kohortunun (48
    hasta) HAM skorunun dondurulmuş ortalama/std'si. Bilinçli olarak
    modül sabiti (yeniden hesaplanmaz) — yukarıdaki "MİMARİ BOŞLUĞU"
    notuna bakın.
    """
    raw = _dna_repair_raw(row)
    z = (raw - reference_mean) / reference_std
    return round(_clip100(z * 15.0 + 50.0), 2)


DNA_REPAIR_P33_V1_0 = 41.9847
DNA_REPAIR_P66_V1_0 = 59.0642


def classify_dna_repair(
    score: float,
    *,
    p33: float = DNA_REPAIR_P33_V1_0,
    p66: float = DNA_REPAIR_P66_V1_0,
) -> str:
    """`repair_class` (backend clean enum: impaired/intermediate/intact).

    Donmuş tertil kuralı — `classify_tmz_resistance` ile aynı prensip
    (v45.txt satır 471-492).

    ⚠️ SAVUNMACI YUVARLAMA (2026-08-18, Codex çapraz incelemesi BUG 1):
    bkz. `classify_tmz_resistance()` docstring'i — aynı gerekçe, aynı
    düzeltme.
    """
    score = round(score, 2)
    if score <= p33:
        return "impaired"
    if score <= p66:
        return "intermediate"
    return "intact"


# =============================================================================
# tmz_class_relative — KASITLI OLARAK HESAPLANMAZ
# =============================================================================
#
# Karar (decisions/2026-08-18-omics-skor-formulleri-ve-esikler.md KARAR 2,
# Barış onayı "üçünü de onaylıyorum, uygula"):
#
#   "`tmz_class_relative`, `tmz_class`'ın emoji sunum formatıdır. Aynı
#   tertil sınıflandırması. Backend `tmz_class` kullanır."
#
# `molecular_scores.tmz_class_relative` kolonu bilinçli olarak NULL
# bırakılır — DÜŞÜRÜLMEZ (şema değişikliği, KESİN SINIRLAR #4) ama bu
# modül de onu DOLDURMAZ (backend'de emoji kullanılmaz, v45.txt satır
# 1156-1160). İleride bir sunum katmanı (Streamlit) emoji formatı
# isterse, `classify_tmz_resistance()`'ın döndürdüğü clean enum'dan
# (`sensitive`/`intermediate`/`resistant`) emoji'ye eşleme AYRI bir
# sunum-katmanı fonksiyonu olmalı, bu modülün sorumluluğu DEĞİL.
