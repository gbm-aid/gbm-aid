"""`stability_frequency_threshold` (bootstrap stabilite seçimi eşiği)
duyarlılık analizi -- SADECE SENTETİK VERİ, gerçek Cox eğitimi DEĞİL.

GÖREV BAĞLAMI (2026-08-14, koordinatör talimatı, Barış onayı):
`pipeline/cox_model.py::run_nested_cv()` içindeki bootstrap stabilite
seçimi `stability_frequency_threshold` parametresini kullanır, varsayılanı
**0,6**. Bu değer literatür-standardı (Meinshausen & Bühlmann 2010,
tipik aralık 0,6-0,9) olarak seçildi ama decisions/2026-08-13-bolge-
stratejisi-ve-modelleme-protokolu.md'de SAYISAL olarak KİLİTLENMEDİ ve
GERÇEK veriyle hiç sınanmadı (bu C1-C6 modülünü yazan ajanın kendi
"Barış'a soru" notu, AKTIF-GOREVLER.md 2026-08-14 girdisi).

KRİTİK KISIT (canlı doğrulandı, 2026-08-14 sabah): `radiomics` tablosunda
C32 sözleşmesiyle üretilmiş **0 satır** var. Eski `UPenn-PyRadiomics-107`/
`LUMIERE-PyRadiomics-107`/`TCGA-ground-truth` satırları A-yöntemiyle
(ham görüntü + binWidth=25, N4/Z-score atlanmış) üretildi ve GEÇERSİZ --
bunlarla model eğitimi/karar üretimi YASAK (decisions/2026-08-13-
pyradiomics-c32-bincount-karari.md kilitli karar).

**Bu yüzden bu script GERÇEK bir eşik SEÇEMEZ/KİLİTLEYEMEZ.** Yalnızca
bilinen sinyal/gürültü yapısına sahip SENTETİK veri ile
`stability_frequency_threshold`'un DAVRANIŞINI karakterize eder ve bir
TAVSİYE üretir. Nihai eşik, C32 verisi DB'ye yazıldıktan sonra GERÇEK
UPenn verisiyle yeniden ölçülmelidir -- bu script'in sonucu o ölçümün
YERİNE GEÇMEZ.

DB'YE HİÇ BAĞLANMAZ (`db_connection`'a hiç dokunulmuyor, `get_connection()`
hiç çağrılmıyor). `pipeline/cox_model.py` DEĞİŞTİRİLMEDİ -- yalnız
salt-okunur import edilip GERÇEK üretim fonksiyonları
(`select_features_lasso`, `_bootstrap_stability_selection`,
`_score_hyperparameters_via_inner_cv`, `evaluate_external_test`,
`STABLE_FEATURES_ICC60`, `run_nested_cv`) sentetik veriyle ÇAĞRILIYOR.

NEDEN `run_nested_cv()`'NİN DOĞRUDAN, THRESHOLD-BAŞINA TEKRAR TEKRAR
ÇAĞRILMADIĞI (performans notu, dürüstlük için açıkça yazılıyor):
Gerçek `n=630, p=93` ölçeğinde tek bir `CoxPHFitter.fit()` çağrısı
ölçüldü: ~1,1-1,5 saniye (aşağıdaki `select_features_lasso` zamanlaması,
bu script'in geliştirilmesi sırasında canlı ölçüldü). `run_nested_cv()`
her fold'da ~(grid taraması + 1 elastic-net fit + `n_bootstrap_stability`
bootstrap fit + 1 final fit) Cox fit'i yapıyor -- `n_bootstrap_stability`
(varsayılan 200) BASKIN maliyet bileşeni. `stability_frequency_threshold`
parametresi yalnızca ZATEN HESAPLANMIŞ bir `frequency` (özellik başına
bootstrap seçilme oranı) Series'ini FİLTRELİYOR -- bootstrap örneklemesinin
KENDİSİNİ etkilemiyor (bkz. `_bootstrap_stability_selection`'ın kodu:
`seed` parametresi threshold'dan bağımsız, `frequency` hesabı threshold'dan
ÖNCE tamamlanıyor). Bu yüzden AYNI (fold, seed) için 3 farklı threshold'la
`run_nested_cv()`'yi 3 kez çağırmak, AYNI bootstrap örneklerini 3 KEZ
gereksiz yere yeniden hesaplardı -- ölçülen tam-ölçek maliyetle bu,
makul bir duyarlılık analizini pratik olarak imkânsız kılacak kadar yavaş
(tek bir nested-CV çağrısı 120 saniyeden uzun sürdü, arka plana düştü).

Bu script bunun yerine `_bootstrap_stability_selection()`'ı HER (replikasyon,
dış-fold) çifti için BİR KEZ (frequency_threshold=0,0 ile, yani "hiçbir şeyi
filtreleme, tüm frekansları döndür") çağırıp dönen `frequency` Series'ini
HER threshold için AYRI AYRI filtreliyor -- geri kalan tüm adımlar
(grid taraması, elastic-net seçimi, final fit, dış-test C-index) `run_
nested_cv()`'nin KENDİSİYLE BİREBİR AYNI mantığı, AYNI production
fonksiyonlarını (`_score_hyperparameters_via_inner_cv`,
`select_features_lasso`, `lifelines.CoxPHFitter`,
`lifelines.utils.concordance_index`) kullanarak, AYNI seed şemasıyla
(`seed=replicate_seed`, bootstrap `seed=replicate_seed+fold_index`)
inline tekrar eder. Sonuç, AYNI (fold, seed) için `run_nested_cv()`'yi
gerçekten 3 kez çağırmakla SAYISAL OLARAK ÖZDEŞTİR (bootstrap örnekleme
threshold'dan bağımsız olduğu için) -- bu denklik `--verify-equivalence`
bayrağıyla KÜÇÜK ÖLÇEKLİ (n=80, 12 özellik) bir senaryoda GERÇEKTEN
`run_nested_cv()`'ye karşı test edilip doğrulanabilir (aşağıya bkz.).

SENTETİK VERİ ÜRETİM MODELİ (aşağıda parametreleriyle birlikte
belgelidir -- KEYFİ VARSAYIMLARDIR, gerçek radyomik kovaryans yapısının
KANITLANMIŞ bir ölçümü DEĞİLDİR):
  1. 93 özellik `STABLE_FEATURES_ICC60`'tan (gerçek şema, GERÇEK isimler)
     alınır, PyRadiomics aile önekine göre 7 aileye ayrılır (firstorder,
     shape, glcm, gldm, glrlm, glszm, ngtdm).
  2. Bir faktör modeliyle blok-korelasyonlu kovaryans üretilir: her
     özellik = 0,30×global_faktör + 0,70×aile_faktörü + kalan_gürültü
     (birim varyansa normalize) -> aile-içi korelasyon ~0,58,
     aileler-arası ~0,09. Bu, karar dosyasındaki GERÇEK ÖLÇÜLEN WT/TC-
     ARASI korelasyonla (|r|~0,80-0,87, FARKLI bir karşılaştırma --
     kompozit-bölge örtüşmesinden kaynaklanıyor) KARIŞTIRILMAMALI; burada
     amaçlanan TEK-bölge (WT-only) İÇİ orta düzey aile-korelasyonu --
     KEYFİ bir varsayım, gerçek UPenn WT-içi kovaryansın ölçümü DEĞİL.
  3. 12 "gerçek" (true) öngörücü özellik BİR KEZ, sabit bir tohumla
     (GROUND_TRUTH_SEED) seçilir ve dondurulur -- tüm replikasyonlarda
     AYNI kalır (yalnız örnekleme gürültüsü replikasyondan replikasyona
     değişir). Katsayılar (log-hazard) ±[0,15, 0,35] aralığından.
  4. Sağkalım süresi Cox-uyumlu üstel taban tehlike ile üretilir
     (Bender ​et al. 2005 yöntemi): T = -ln(U) / (λ0·exp(η)).
  5. Sansür, hedef olay oranına (UPenn: 603/630 ≈ %95,7,
     `hafta2_durum_ozeti.md`/AKTIF-GOREVLER.md'de doğrulanan GERÇEK
     sayı) tek seferlik kalibre edilen üstel bir sansür süresiyle üretilir.
  6. "TCGA benzeri" harici sentetik test seti (n=39, karar dosyasındaki
     Bölüm 4.4 CI hesabındaki n) AYNI gerçek-özellik/katsayı kümesiyle
     ama hafif kovaryat kaymasıyla (shift_mean=0,10, shift_scale=1,15 --
     KEYFİ, gerçek ComBat-öncesi TCGA batch farkının bir ölçümü DEĞİL,
     yalnızca "iç CV havuzuyla özdeş DEĞİL" niteliğini simüle etmek için)
     üretilir.

Regresyon: bu script `tests/test_cox_model.py`'yi ETKİLEMEZ (yeni dosya,
mevcut hiçbir fonksiyon değişmedi).
"""

from __future__ import annotations

import argparse
import ast
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Konsol kodlaması güvencesi (2026-09-11, gerçek hata düzeltmesi) ------
# Windows konsolu bu makinede cp1254 ile açılıyor. Modül docstring'i
# argparse'a `description=__doc__` olarak veriliyor ve içinde cp1254'e
# kodlanamayan karakterler var (U+200B, λ, η, ≈; ayrıca dosyanın başka
# yerlerinde ⚠ ⛔ →). Bu yüzden `--help` UnicodeEncodeError ile ÇÖKÜYORDU
# (2026-09-11'de ölçüldü) ve detached koşuda log dosyası da aynı riski
# taşıyor. Çıktı akışlarını UTF-8'e sabitliyoruz; SAYISAL hiçbir şeyi
# etkilemez (yalnız konsol/log kodlaması), CSV yazımı zaten pandas'ın
# kendi kodlamasını kullanıyor.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # yeniden yönlendirilmiş/eski akış
        pass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold

from pipeline.cox_model import (
    STABLE_FEATURES_ICC60,
    _bootstrap_stability_selection,
    _score_hyperparameters_via_inner_cv,
    evaluate_external_test,
    run_nested_cv,
    select_features_lasso,
)

# =====================================================================
# Sabit "gerçek dünya" bağlam parametreleri (görev talimatından)
# =====================================================================
# ⚠️ İKİ KOŞU VAR — VARSAYILAN SABİTLER BİLEREK ESKİ DEĞERLERDE
# (işaret: 2026-08-28; parametrizasyon + ek koşu: 2026-09-11, Barış onayı)
# --------------------------------------------------------------------
# Aşağıdaki modül sabitleri, simülasyonun İLK koşulduğu tarihteki
# (2026-08-14) değerlerdir ve VARSAYILAN olarak öyle KALIR:
#
#   Varsayılan sabit        Bugünkü kilitli gerçek
#   ---------------------   ---------------------------------------
#   N_TRAIN_PATIENTS 630    611  (19 hastada yalnız post-op T1ce var
#                                 ve hiçbirinde maske yok -> radyomik
#                                 satırı üretilemiyor)
#   N_TRAIN_EVENTS   603    585  (`censored_labels` hatası nedeniyle
#                                 9 CENSORED hasta sessizce düşüyordu)
#   N_EXTERNAL        39     harici test 2026-08-18'de UCSF-PDGM
#                                 295 hasta / 169 olay oldu
#
# NEDEN VARSAYILANLAR ESKİ KALDI: 2026-08-14 koşusunun çıktıları
# (`stability_threshold_sensitivity_synthetic_{summary,folds,external}.csv`)
# stabilite eşiğinin 0,6'da bırakılması kararının ORİJİNAL KANITIDIR ve
# bit-birebir yeniden üretilebilir kalmalıdır: script'i argümansız
# koşmak eski davranışı AYNEN verir ("hata düzeltmek post-hoc değildir"
# kuralı — eski sonuç silinmez/ezilmez, yenisi YANINA eklenir).
#
# İKİ KOŞU (İKİSİ DE RAPORLANIR — 2026-09-11, Barış onayı):
#   1. ESKİ (2026-08-14): 630/603 eğitim, 39 harici (TCGA analoğu),
#      6 replikasyon × eşik {0,5, 0,6, 0,7, 0,8}. 0,6 kararının orijinal
#      destekleyici kanıtı. Çıktılar: suffix'siz *_synthetic_*.csv —
#      DOKUNULMAZ. Fiili CLI (2026-09-11'de bit-birebir yeniden üretimle
#      GERİYE DÖNÜK TESPİT edildi; log/2026-08-14.md komutu kaydetmemişti):
#      `--thresholds 0.5 0.6 0.7 0.8 --n-replicates 6
#       --n-bootstrap-external-ci 500` (gerisi varsayılan). Kanıt:
#      replikasyon-0 fold satırları varsayılanlarla, harici CI'lar ise
#      YALNIZ n_bootstrap_external_ci=500 ile 1e-12 içinde eşleşti
#      (200/300/1000 eşleşmedi).
#   2. YENİ (2026-09-11): 611/585 eğitim, 295/169 harici (UCSF analoğu,
#      --n-external-events ile ayrı sansür kalibrasyonu), diğer her şey
#      (seed şeması, replikasyon sayısı, eşik ızgarası, grid'ler)
#      ORİJİNALLE AYNI. Çıktılar: *_synthetic_611585_*.csv.
#   Raporda İKİSİ DE yer alır; yeni koşu güncel kilitli sayılarla
#   ÖLÇÜLMÜŞ duyarlılıktır, eski koşu tarihsel kanıt olarak korunur.
#
# ⛔ Rapor kohort sayısı olarak 630/603 DEĞİL 611/585 yazar; eski
#    koşunun eski sabitlerle koşulduğu beyan edilir.
# =====================================================================
N_TRAIN_PATIENTS = 630          # varsayılan = ESKİ koşu; yeni koşu için --n-train-patients 611
N_TRAIN_EVENTS_TARGET = 603     # varsayılan = ESKİ koşu; yeni koşu için --n-train-events 585
TARGET_EVENT_RATE = N_TRAIN_EVENTS_TARGET / N_TRAIN_PATIENTS  # ~0.9571 (585/611 ile ~0.9574)
N_EXTERNAL_PATIENTS = 39        # varsayılan = ESKİ koşu (TCGA analoğu); yeni koşu --n-external-patients 295 --n-external-events 169
N_FEATURES_EXPECTED = 93        # ICC>=0.60 birincil küme

# =====================================================================
# Sentetik veri üretim modeli parametreleri (KEYFİ, belgelenmiş)
# =====================================================================
GROUND_TRUTH_SEED = 20260814
N_TRUE_FEATURES = 12
GLOBAL_FACTOR_LOADING = 0.30
FAMILY_FACTOR_LOADING = 0.70
BASELINE_LAMBDA = 0.01
EXTERNAL_SHIFT_MEAN = 0.10
EXTERNAL_SHIFT_SCALE = 1.15
CENSORING_CALIBRATION_SEED = 999
CENSORING_CALIBRATION_N = 8000
CENSORING_TOLERANCE = 0.005

DURATION_COL = "survival_days"
EVENT_COL = "event"


def classify_family(feature_name: str) -> str:
    """PyRadiomics `original_<sinif>_<ozellik>` adından aile çıkarır."""

    if "firstorder" in feature_name:
        return "firstorder"
    if "shape" in feature_name:
        return "shape"
    for fam in ("glcm", "gldm", "glrlm", "glszm", "ngtdm"):
        if fam in feature_name:
            return fam
    raise ValueError(f"Bilinmeyen PyRadiomics ailesi: {feature_name!r}")


@dataclass(frozen=True)
class GroundTruth:
    feature_columns: tuple
    families: dict
    true_features: tuple
    beta: dict  # feature_name -> log-hazard katsayısı (yalnız true_features icin dolu)
    beta_vector: np.ndarray  # feature_columns sirasiyla, noise=0.0


def build_ground_truth(feature_columns=None) -> GroundTruth:
    """`feature_columns=None` -> gerçek `STABLE_FEATURES_ICC60` (93) kullanılır.

    `--verify-equivalence` küçük-ölçekli denklik testi için AZALTILMIŞ bir
    özellik listesi geçirebilir (aşağıya bkz.) -- bu yüzden parametrik.
    """

    columns = tuple(feature_columns) if feature_columns is not None else tuple(STABLE_FEATURES_ICC60)
    if feature_columns is None and len(columns) != N_FEATURES_EXPECTED:
        raise RuntimeError(
            f"STABLE_FEATURES_ICC60 {N_FEATURES_EXPECTED} ozellik beklerken "
            f"{len(columns)} bulundu -- pipeline.cox_model degismis olabilir, "
            "bu script'in varsayimlari artik gecersiz. DURDUR."
        )
    families = {f: classify_family(f) for f in columns}

    rng = np.random.default_rng(GROUND_TRUTH_SEED)
    n_true = min(N_TRUE_FEATURES, len(columns))
    true_features = tuple(
        sorted(rng.choice(np.array(columns), size=n_true, replace=False).tolist())
    )
    magnitudes = rng.uniform(0.15, 0.35, size=n_true)
    signs = rng.choice([-1.0, 1.0], size=n_true)
    beta = {feat: float(mag * sign) for feat, mag, sign in zip(true_features, magnitudes, signs)}

    beta_vector = np.array([beta.get(f, 0.0) for f in columns])
    return GroundTruth(
        feature_columns=columns,
        families=families,
        true_features=true_features,
        beta=beta,
        beta_vector=beta_vector,
    )


def _simulate_features_and_eta(
    n: int,
    ground_truth: GroundTruth,
    rng: np.random.Generator,
    shift_mean: float = 0.0,
    shift_scale: float = 1.0,
) -> tuple:
    """Blok-korelasyonlu sentetik özellikler + Cox lineer prediktörü."""

    unique_families = sorted(set(ground_truth.families.values()))
    family_factors = {fam: rng.standard_normal(n) for fam in unique_families}
    global_factor = rng.standard_normal(n)
    residual_scale_sq = 1.0 - GLOBAL_FACTOR_LOADING**2 - FAMILY_FACTOR_LOADING**2
    if residual_scale_sq <= 0:
        raise ValueError("GLOBAL_FACTOR_LOADING/FAMILY_FACTOR_LOADING varyansi asiyor.")
    residual_scale = np.sqrt(residual_scale_sq)

    n_features = len(ground_truth.feature_columns)
    X = np.empty((n, n_features), dtype=float)
    for j, feat in enumerate(ground_truth.feature_columns):
        fam = ground_truth.families[feat]
        resid = rng.standard_normal(n)
        X[:, j] = (
            GLOBAL_FACTOR_LOADING * global_factor
            + FAMILY_FACTOR_LOADING * family_factors[fam]
            + residual_scale * resid
        )

    X = X * shift_scale + shift_mean
    eta = X @ ground_truth.beta_vector
    return X, eta


def _calibrate_censoring_mean(
    ground_truth: GroundTruth,
    target_event_rate: float = TARGET_EVENT_RATE,
    shift_mean: float = 0.0,
    shift_scale: float = 1.0,
) -> float:
    """Hedef olay oranina ulasan sansur-suresi ortalamasini BIR KEZ
    kalibre eder (geometrik bisection, sabit tohum -- deterministik).

    Geriye donuk uyumluluk: varsayilan parametrelerle (target_event_rate=
    TARGET_EVENT_RATE=603/630, shift yok) 2026-08-14 kosusuyla BIREBIR ayni
    yol izlenir. `shift_mean`/`shift_scale`, HARICI kohort icin ayri sansur
    kalibrasyonu gerektiginde (2026-09-11 ek kosusu: UCSF analogu 295/169,
    olay orani ~0,573) harici kovaryat kaymasi altinda kalibrasyon yapmak
    icindir."""

    rng = np.random.default_rng(CENSORING_CALIBRATION_SEED)
    X, eta = _simulate_features_and_eta(
        CENSORING_CALIBRATION_N, ground_truth, rng,
        shift_mean=shift_mean, shift_scale=shift_scale,
    )
    u = rng.uniform(1e-9, 1.0, size=CENSORING_CALIBRATION_N)
    T = -np.log(u) / (BASELINE_LAMBDA * np.exp(eta))

    lo, hi = 1.0, 1e8
    mid = float(np.sqrt(lo * hi))
    for _ in range(80):
        mid = float(np.sqrt(lo * hi))
        rng_c = np.random.default_rng(CENSORING_CALIBRATION_SEED + 1)
        C = rng_c.exponential(scale=mid, size=CENSORING_CALIBRATION_N)
        event_rate = float(np.mean(T <= C))
        if abs(event_rate - target_event_rate) <= CENSORING_TOLERANCE:
            break
        if event_rate < target_event_rate:
            lo = mid  # sansur cok fazla -> ortalamayi buyut (sansuru seyreklestir)
        else:
            hi = mid
    return mid


def simulate_cohort(
    n_patients: int,
    ground_truth: GroundTruth,
    censoring_mean: float,
    seed: int,
    shift_mean: float = 0.0,
    shift_scale: float = 1.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X, eta = _simulate_features_and_eta(
        n_patients, ground_truth, rng, shift_mean=shift_mean, shift_scale=shift_scale
    )
    u = rng.uniform(1e-9, 1.0, size=n_patients)
    T = -np.log(u) / (BASELINE_LAMBDA * np.exp(eta))
    C = rng.exponential(scale=censoring_mean, size=n_patients)

    duration = np.minimum(T, C)
    event = (T <= C).astype(int)

    # Gunlere olcekleme (yalnizca okunabilirlik icin, C-index/Cox siralamayi ETKILEMEZ
    # -- monoton bir olcek donusumu).
    duration_days = duration * 100.0

    frame = pd.DataFrame(X, columns=list(ground_truth.feature_columns))
    frame[DURATION_COL] = duration_days
    frame[EVENT_COL] = event
    frame.index = [f"SYN-{seed}-{i:04d}" for i in range(n_patients)]
    frame.index.name = "patient_id"
    return frame


def _pairwise_jaccard_mean(feature_sets) -> float:
    sets = [set(s) for s in feature_sets if s]
    if len(sets) < 2:
        return float("nan")
    scores = []
    for a, b in itertools.combinations(sets, 2):
        union = a | b
        if not union:
            continue
        scores.append(len(a & b) / len(union))
    return float(np.mean(scores)) if scores else float("nan")


def _tpr_fpr(selected, true_features, all_features) -> tuple:
    selected_set = set(selected)
    true_set = set(true_features)
    noise_set = set(all_features) - true_set
    tpr = len(selected_set & true_set) / len(true_set) if true_set else float("nan")
    fpr = len(selected_set & noise_set) / len(noise_set) if noise_set else float("nan")
    return tpr, fpr


def _grid_search_best_hyperparams(
    train_sub: pd.DataFrame,
    feature_columns: list,
    *,
    l1_ratio_grid,
    penalizer_grid,
    inner_splits: int,
    seed: int,
) -> tuple:
    """`run_nested_cv()`'nin dış-fold başına yaptığı grid taramasının BİREBİR
    AYNISI -- aynı `_score_hyperparameters_via_inner_cv()` çağrılıyor."""

    best_score = float("-inf")
    best_penalizer = penalizer_grid[0]
    best_l1_ratio = l1_ratio_grid[0]
    any_valid = False
    for penalizer in penalizer_grid:
        for l1_ratio in l1_ratio_grid:
            score = _score_hyperparameters_via_inner_cv(
                train_sub,
                feature_columns,
                duration_col=DURATION_COL,
                event_col=EVENT_COL,
                extra_columns=[],
                penalizer=penalizer,
                l1_ratio=l1_ratio,
                inner_splits=inner_splits,
                seed=seed,
            )
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_penalizer = penalizer
                best_l1_ratio = l1_ratio
                any_valid = True
    if not any_valid:
        raise RuntimeError(
            "Grid taramasında hiçbir (penalizer, l1_ratio) kombinasyonu geçerli "
            "bir skor üretmedi -- grid/inner_splits/n gözden geçirilmeli."
        )
    return best_penalizer, best_l1_ratio, best_score


def run_efficient_threshold_sweep(
    thresholds: list,
    n_replicates: int,
    outer_splits: int,
    inner_splits: int,
    l1_ratio_grid: tuple,
    penalizer_grid: tuple,
    n_bootstrap_stability: int,
    n_bootstrap_external_ci: int,
    base_seed: int,
    ground_truth: GroundTruth = None,
    verbose: bool = True,
    train_frame_override: pd.DataFrame = None,
    n_train_patients: int = N_TRAIN_PATIENTS,
    target_event_rate: float = TARGET_EVENT_RATE,
    n_external_patients: int = N_EXTERNAL_PATIENTS,
    external_target_event_rate: float = None,
    replicate_start: int = 0,
) -> tuple:
    """Threshold-başına maliyeti minimize eden, `run_nested_cv()` ile
    SAYISAL OLARAK ÖZDEŞ (bkz. modül docstring'i + `verify_equivalence()`)
    ama bootstrap'ı fold başına BİR KEZ hesaplayan sarmalayıcı.

    `train_frame_override` verilirse (yalnız `verify_equivalence()`'ın
    kullandığı bir test-kancası) `n_replicates` 1 OLMALI ve bu DataFrame
    `simulate_cohort()` yerine DOĞRUDAN kullanılır -- gerçek `run_nested_cv()`
    ile BİREBİR AYNI girdi üzerinde çalıştığını garanti eder.

    Geriye dönük uyumluluk (2026-09-11 parametrizasyonu):
    - Varsayılanlarla (630/603, harici 39, `external_target_event_rate=None`,
      `replicate_start=0`) davranış 2026-08-14 koşusuyla BİREBİR aynıdır:
      harici kohort eğitim sansür kalibrasyonunu paylaşır, replikasyon
      tohumları `base_seed + replicate*1000`.
    - `external_target_event_rate` verilirse harici kohort için AYRI bir
      sansür ortalaması, harici kovaryat kayması (EXTERNAL_SHIFT_*)
      altında kalibre edilir (UCSF analoğu 295/169 → oran ~0,573).
    - `replicate_start`, uzun koşuyu replikasyon bazında PARÇALARA bölmek
      içindir: replikasyon indeksi ve tohumu (`base_seed + replicate*1000`)
      monolitik koşuyla aynı kalır, çıktılar sonra birleştirilir
      (`--merge-fold-csvs`)."""

    ground_truth = ground_truth or build_ground_truth()
    censoring_mean = _calibrate_censoring_mean(ground_truth, target_event_rate=target_event_rate)
    if external_target_event_rate is None:
        external_censoring_mean = censoring_mean  # eski (2026-08-14) davranış
    else:
        external_censoring_mean = _calibrate_censoring_mean(
            ground_truth,
            target_event_rate=external_target_event_rate,
            shift_mean=EXTERNAL_SHIFT_MEAN,
            shift_scale=EXTERNAL_SHIFT_SCALE,
        )
    feature_columns = list(ground_truth.feature_columns)

    if train_frame_override is not None and n_replicates != 1:
        raise ValueError("train_frame_override yalnız n_replicates=1 ile kullanılabilir.")

    fold_records = []
    external_records = []

    for replicate in range(replicate_start, replicate_start + n_replicates):
        replicate_seed = base_seed + replicate * 1000
        if train_frame_override is not None:
            train_frame = train_frame_override
        else:
            train_frame = simulate_cohort(
                n_train_patients,
                ground_truth,
                censoring_mean,
                seed=replicate_seed,
            )
        external_frame = simulate_cohort(
            n_external_patients,
            ground_truth,
            external_censoring_mean,
            seed=replicate_seed + 500,
            shift_mean=EXTERNAL_SHIFT_MEAN,
            shift_scale=EXTERNAL_SHIFT_SCALE,
        )
        if verbose:
            print(
                f"[replicate {replicate}] train n={len(train_frame)} "
                f"events={int(train_frame[EVENT_COL].sum())} "
                f"(oran={train_frame[EVENT_COL].mean():.4f}, hedef={target_event_rate:.4f}) | "
                f"external n={len(external_frame)} events={int(external_frame[EVENT_COL].sum())}"
            )

        # ---- Nested-CV analoğu: outer split BİR KEZ, her fold'da grid+EN+bootstrap BİR KEZ ----
        outer_splitter = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=replicate_seed)
        for fold_index, (train_idx, test_idx) in enumerate(
            outer_splitter.split(train_frame, train_frame[EVENT_COL])
        ):
            t0 = time.time()
            train_sub = train_frame.iloc[train_idx]
            test_sub = train_frame.iloc[test_idx]

            best_penalizer, best_l1_ratio, inner_score = _grid_search_best_hyperparams(
                train_sub,
                feature_columns,
                l1_ratio_grid=l1_ratio_grid,
                penalizer_grid=penalizer_grid,
                inner_splits=inner_splits,
                seed=replicate_seed,
            )
            en_selected, _, _ = select_features_lasso(
                train_sub,
                feature_columns,
                duration_col=DURATION_COL,
                event_col=EVENT_COL,
                extra_columns=[],
                penalizer=best_penalizer,
                l1_ratio=best_l1_ratio,
            )
            # threshold=0.0 -> HİÇBİR ŞEY filtrelenmez, TÜM frekanslar dönsün.
            _, frequency = _bootstrap_stability_selection(
                train_sub,
                feature_columns,
                duration_col=DURATION_COL,
                event_col=EVENT_COL,
                extra_columns=[],
                penalizer=best_penalizer,
                l1_ratio=best_l1_ratio,
                n_bootstrap=n_bootstrap_stability,
                frequency_threshold=0.0,
                seed=replicate_seed + fold_index,
            )
            elapsed_shared = time.time() - t0

            for threshold in thresholds:
                stable_features = sorted(frequency[frequency >= threshold].index.tolist())
                final_features = stable_features if stable_features else en_selected

                cox_final = CoxPHFitter(penalizer=best_penalizer, l1_ratio=best_l1_ratio)
                cox_final.fit(
                    train_sub[final_features + [DURATION_COL, EVENT_COL]],
                    duration_col=DURATION_COL,
                    event_col=EVENT_COL,
                )
                test_hazard = cox_final.predict_partial_hazard(test_sub[final_features])
                c_index = concordance_index(test_sub[DURATION_COL], -test_hazard, test_sub[EVENT_COL])

                tpr, fpr = _tpr_fpr(final_features, ground_truth.true_features, ground_truth.feature_columns)
                fold_records.append(
                    {
                        "replicate": replicate,
                        "threshold": threshold,
                        "fold": fold_index,
                        "n_train": len(train_sub),
                        "n_test": len(test_sub),
                        "n_events_train": int(train_sub[EVENT_COL].sum()),
                        "n_events_test": int(test_sub[EVENT_COL].sum()),
                        "best_penalizer": best_penalizer,
                        "best_l1_ratio": best_l1_ratio,
                        "n_selected_features_elastic_net": len(en_selected),
                        "n_selected_features_final": len(final_features),
                        "tpr": tpr,
                        "fpr": fpr,
                        "c_index": c_index,
                        "selected_features": final_features,
                        "shared_grid_and_bootstrap_seconds": elapsed_shared,
                    }
                )

        if verbose:
            for threshold in thresholds:
                thr_rows = [r for r in fold_records if r["replicate"] == replicate and r["threshold"] == threshold]
                mean_c = np.mean([r["c_index"] for r in thr_rows])
                mean_n = np.mean([r["n_selected_features_final"] for r in thr_rows])
                print(f"  threshold={threshold:.2f} -> nested-CV C-index (fold ort.)={mean_c:.4f}, ort. n_features={mean_n:.1f}")

        # ---- Harici sentetik test analoğu (eski: TCGA-benzeri n=39; yeni: UCSF-benzeri 295/169) ----
        # TÜM eğitim havuzunda (n_train_patients) grid+EN+bootstrap BİR KEZ, sonra HER threshold
        # için ayrı final model + evaluate_external_test().
        t0 = time.time()
        best_penalizer_full, best_l1_ratio_full, _ = _grid_search_best_hyperparams(
            train_frame,
            feature_columns,
            l1_ratio_grid=l1_ratio_grid,
            penalizer_grid=penalizer_grid,
            inner_splits=inner_splits,
            seed=replicate_seed,
        )
        en_selected_full, _, _ = select_features_lasso(
            train_frame,
            feature_columns,
            duration_col=DURATION_COL,
            event_col=EVENT_COL,
            extra_columns=[],
            penalizer=best_penalizer_full,
            l1_ratio=best_l1_ratio_full,
        )
        _, frequency_full = _bootstrap_stability_selection(
            train_frame,
            feature_columns,
            duration_col=DURATION_COL,
            event_col=EVENT_COL,
            extra_columns=[],
            penalizer=best_penalizer_full,
            l1_ratio=best_l1_ratio_full,
            n_bootstrap=n_bootstrap_stability,
            frequency_threshold=0.0,
            seed=replicate_seed + 9000,
        )
        elapsed_external_shared = time.time() - t0

        for threshold in thresholds:
            stable_features_full = sorted(frequency_full[frequency_full >= threshold].index.tolist())
            final_features_full = stable_features_full if stable_features_full else en_selected_full

            final_model = CoxPHFitter(penalizer=best_penalizer_full, l1_ratio=best_l1_ratio_full)
            final_model.fit(
                train_frame[final_features_full + [DURATION_COL, EVENT_COL]],
                duration_col=DURATION_COL,
                event_col=EVENT_COL,
            )
            ext_result = evaluate_external_test(
                final_model,
                external_frame,
                final_features_full,
                duration_col=DURATION_COL,
                event_col=EVENT_COL,
                extra_columns=None,
                n_bootstrap=n_bootstrap_external_ci,
                seed=replicate_seed + 999,
            )
            ext_tpr, ext_fpr = _tpr_fpr(final_features_full, ground_truth.true_features, ground_truth.feature_columns)
            external_records.append(
                {
                    "replicate": replicate,
                    "threshold": threshold,
                    "n_selected_features": len(final_features_full),
                    "tpr": ext_tpr,
                    "fpr": ext_fpr,
                    "external_c_index": ext_result["c_index"],
                    "external_ci_lower": ext_result["ci_lower"],
                    "external_ci_upper": ext_result["ci_upper"],
                    "external_n_events": ext_result["n_events"],
                    "shared_grid_and_bootstrap_seconds": elapsed_external_shared,
                }
            )
            if verbose:
                print(
                    f"  [external] threshold={threshold:.2f} -> C-index={ext_result['c_index']:.4f} "
                    f"[{ext_result['ci_lower']:.4f}, {ext_result['ci_upper']:.4f}] n_features={len(final_features_full)}"
                )

    fold_df = pd.DataFrame(fold_records)
    return fold_df, pd.DataFrame(external_records)


def verify_equivalence(verbose: bool = True) -> bool:
    """KÜÇÜK ÖLÇEKLİ (n=80, 12 özellik) senaryoda, bu script'in verimli
    (`run_efficient_threshold_sweep`) yolunun GERÇEK `run_nested_cv()`'yi
    threshold başına ayrı ayrı çağırmakla SAYISAL OLARAK ÖZDEŞ sonuç
    verdiğini kanıtlar (fold C-index'leri + seçilen özellik kümeleri
    birebir eşleşmeli). Bu, modül docstring'indeki "denklik" iddiasının
    KANIT AYAĞI -- iddia edilip doğrulanmadan bırakılmıyor."""

    small_features = list(STABLE_FEATURES_ICC60[:12])
    ground_truth = build_ground_truth(feature_columns=small_features)
    censoring_mean = _calibrate_censoring_mean(ground_truth)
    frame = simulate_cohort(80, ground_truth, censoring_mean, seed=7)

    thresholds = [0.5, 0.6, 0.7]
    l1_grid = (0.5, 0.8)
    pen_grid = (0.1, 0.3)

    efficient_df, _ = run_efficient_threshold_sweep(
        thresholds=thresholds,
        n_replicates=1,
        outer_splits=3,
        inner_splits=2,
        l1_ratio_grid=l1_grid,
        penalizer_grid=pen_grid,
        n_bootstrap_stability=20,
        n_bootstrap_external_ci=20,
        base_seed=7,
        ground_truth=ground_truth,
        verbose=False,
        train_frame_override=frame,
    )
    # replicate 0 -> replicate_seed = 7 (base_seed + 0*1000)
    replicate_seed = 7

    all_ok = True
    for threshold in thresholds:
        real_df = run_nested_cv(
            frame,
            small_features,
            duration_col=DURATION_COL,
            event_col=EVENT_COL,
            outer_splits=3,
            inner_splits=2,
            seed=replicate_seed,
            l1_ratio_grid=l1_grid,
            penalizer_grid=pen_grid,
            stability_selection=True,
            n_bootstrap_stability=20,
            stability_frequency_threshold=threshold,
        )
        eff_rows = efficient_df[efficient_df["threshold"] == threshold].sort_values("fold")
        real_rows = real_df.sort_values("fold")

        c_index_match = np.allclose(
            eff_rows["c_index"].to_numpy(), real_rows["c_index"].to_numpy(), atol=1e-9
        )
        n_features_match = (
            eff_rows["n_selected_features_final"].to_numpy() == real_rows["n_selected_features_final"].to_numpy()
        ).all()
        ok = c_index_match and n_features_match
        all_ok = all_ok and ok
        if verbose:
            print(
                f"threshold={threshold}: c_index eşleşti={c_index_match} "
                f"(verimli={eff_rows['c_index'].to_numpy()}, gerçek={real_rows['c_index'].to_numpy()}), "
                f"n_features eşleşti={n_features_match} -> {'OK' if ok else 'UYUŞMUYOR'}"
            )

    return all_ok


def summarize(fold_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    """Not (performans): fold-arası Jaccard stabilitesi artık AYRI bir ikinci
    geçiş (`_collect_fold_feature_sets`, kaldırıldı) İLE DEĞİL, ana taramanın
    zaten sakladığı `fold_df["selected_features"]` kolonundan hesaplanıyor --
    ekstra Cox fit'i GEREKMİYOR (~2x süre tasarrufu)."""

    rng = np.random.default_rng(12345)
    rows = []
    for threshold, group in fold_df.groupby("threshold"):
        pooled_c_index = group["c_index"].dropna().to_numpy()
        if len(pooled_c_index) > 0:
            boot_means = [
                np.mean(rng.choice(pooled_c_index, size=len(pooled_c_index), replace=True))
                for _ in range(2000)
            ]
            ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
        else:
            ci_lower = ci_upper = float("nan")

        ext_group = external_df[external_df["threshold"] == threshold]
        stability_values = [
            _pairwise_jaccard_mean(list(rep_group["selected_features"]))
            for _replicate, rep_group in group.groupby("replicate")
        ]
        rows.append(
            {
                "threshold": threshold,
                "n_folds_pooled": len(group),
                "mean_n_selected_final": group["n_selected_features_final"].mean(),
                "sd_n_selected_final": group["n_selected_features_final"].std(),
                "mean_tpr": group["tpr"].mean(),
                "mean_fpr": group["fpr"].mean(),
                "mean_fold_to_fold_jaccard_stability": float(np.nanmean(stability_values)) if stability_values else float("nan"),
                "nested_cv_c_index_mean": pooled_c_index.mean() if len(pooled_c_index) else float("nan"),
                "nested_cv_c_index_median": np.median(pooled_c_index) if len(pooled_c_index) else float("nan"),
                "nested_cv_c_index_bootstrap_ci_lower": ci_lower,
                "nested_cv_c_index_bootstrap_ci_upper": ci_upper,
                "external_c_index_mean": ext_group["external_c_index"].mean(),
                "external_c_index_mean_ci_lower": ext_group["external_ci_lower"].mean(),
                "external_c_index_mean_ci_upper": ext_group["external_ci_upper"].mean(),
                "external_mean_n_selected": ext_group["n_selected_features"].mean(),
                "external_mean_tpr": ext_group["tpr"].mean(),
                "external_mean_fpr": ext_group["fpr"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def _output_paths(out_dir: Path, suffix: str) -> tuple:
    """Suffix'li çıktı yolları. suffix="" -> 2026-08-14 koşusunun ORİJİNAL
    adları (geriye dönük uyumluluk); örn. suffix="_611585" -> yeni koşu."""

    fold_csv = out_dir / f"stability_threshold_sensitivity_synthetic{suffix}_folds.csv"
    external_csv = out_dir / f"stability_threshold_sensitivity_synthetic{suffix}_external.csv"
    summary_csv = out_dir / f"stability_threshold_sensitivity_synthetic{suffix}_summary.csv"
    return fold_csv, external_csv, summary_csv


def _read_fold_csv(path: Path) -> pd.DataFrame:
    """Parça (chunk) fold CSV'sini okur; `selected_features` kolonu CSV'de
    stringleşmiş liste -- Jaccard hesabı için GERÇEK listeye geri çevrilir
    (aksi hâlde `set(str)` karakter kümesi olurdu, sessiz yanlış sonuç)."""

    frame = pd.read_csv(path, float_precision="round_trip")
    frame["selected_features"] = frame["selected_features"].map(ast.literal_eval)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7])
    parser.add_argument("--n-replicates", type=int, default=4)
    parser.add_argument("--outer-splits", type=int, default=4)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--l1-ratio-grid", type=float, nargs="+", default=[0.5, 0.8])
    parser.add_argument("--penalizer-grid", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--n-bootstrap-stability", type=int, default=50)
    parser.add_argument("--n-bootstrap-external-ci", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "week3" / "cox_model"),
    )
    # --- 2026-09-11 parametrizasyonu (varsayılanlar = 2026-08-14 koşusu) ---
    parser.add_argument("--n-train-patients", type=int, default=N_TRAIN_PATIENTS,
                        help="Sentetik egitim havuzu boyutu (varsayilan 630 = ESKI kosu; guncel kilitli deger 611).")
    parser.add_argument("--n-train-events", type=int, default=N_TRAIN_EVENTS_TARGET,
                        help="Egitim hedef olay sayisi -- yalniz ORAN icin kullanilir (varsayilan 603; guncel 585).")
    parser.add_argument("--n-external-patients", type=int, default=N_EXTERNAL_PATIENTS,
                        help="Sentetik harici test boyutu (varsayilan 39 = TCGA analogu; guncel UCSF 295).")
    parser.add_argument("--n-external-events", type=int, default=None,
                        help="Harici hedef olay sayisi. VERILMEZSE eski davranis: harici kohort egitim "
                             "sansur kalibrasyonunu paylasir (~%%95,7 olay). Verilirse harici icin AYRI "
                             "kalibrasyon yapilir (UCSF analogu: 169/295 ~ %%57,3).")
    parser.add_argument("--output-suffix", type=str, default="",
                        help='Cikti dosya adlarina eklenecek suffix (orn. "_611585"). Bos = ORIJINAL adlar.')
    parser.add_argument("--replicate-start", type=int, default=0,
                        help="Ilk replikasyon indeksi -- uzun kosuyu parcalara bolmek icin. Tohum semasi "
                             "monolitik kosuyla ayni kalir (base_seed + replicate*1000).")
    parser.add_argument("--force-overwrite", action="store_true",
                        help="Mevcut cikti dosyalarinin uzerine yazmaya IZIN VER. Varsayilan: mevcut dosya "
                             "varsa HATA (2026-08-14 orijinal kanit dosyalarini kazara ezmemek icin).")
    parser.add_argument("--merge-fold-csvs", type=str, nargs="+", default=None,
                        help="MERGE MODU: parca kosularin *_folds.csv dosyalarini birlestirip ozet uret. "
                             "--merge-external-csvs ile birlikte kullanilir; simulasyon KOSULMAZ.")
    parser.add_argument("--merge-external-csvs", type=str, nargs="+", default=None,
                        help="MERGE MODU: parca kosularin *_external.csv dosyalari (siralama fold ile ayni olmali).")
    parser.add_argument("--verify-equivalence", action="store_true", help="Kucuk-olcekli denklik testini calistirip cik.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.verify_equivalence:
        ok = verify_equivalence(verbose=True)
        print(f"\nDENKLIK TESTI SONUCU: {'GECTI' if ok else 'BASARISIZ'}")
        sys.exit(0 if ok else 1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_csv, external_csv, summary_csv = _output_paths(out_dir, args.output_suffix)

    existing = [p for p in (fold_csv, external_csv, summary_csv) if p.exists()]
    if existing and not args.force_overwrite:
        parser.error(
            "Cikti dosyalari zaten var (uzerine yazma KAPALI -- 2026-08-14 orijinal kanit "
            "dosyalarini korumak icin): "
            + ", ".join(str(p) for p in existing)
            + " | Farkli bir --output-suffix verin ya da bilincli olarak --force-overwrite kullanin."
        )

    # ------------------------- MERGE MODU -------------------------
    if args.merge_fold_csvs is not None or args.merge_external_csvs is not None:
        if not (args.merge_fold_csvs and args.merge_external_csvs):
            parser.error("--merge-fold-csvs ve --merge-external-csvs BIRLIKTE verilmeli.")
        fold_df = pd.concat([_read_fold_csv(Path(p)) for p in args.merge_fold_csvs], ignore_index=True)
        external_df = pd.concat(
            [pd.read_csv(p, float_precision="round_trip") for p in args.merge_external_csvs],
            ignore_index=True,
        )
        fold_df = fold_df.sort_values(["replicate", "fold", "threshold"], kind="stable").reset_index(drop=True)
        external_df = external_df.sort_values(["replicate", "threshold"], kind="stable").reset_index(drop=True)
        dup = fold_df.duplicated(subset=["replicate", "fold", "threshold"])
        if dup.any():
            parser.error("Merge girdilerinde tekrarlanan (replicate, fold, threshold) satirlari var -- ayni parca iki kez verilmis olabilir.")
        summary_df = summarize(fold_df, external_df)
        fold_df.to_csv(fold_csv, index=False)
        external_df.to_csv(external_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        print(f"MERGE tamam: {len(args.merge_fold_csvs)} parca, {fold_df['replicate'].nunique()} replikasyon, "
              f"{len(fold_df)} fold satiri, {len(external_df)} harici satir.")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(summary_df.to_string(index=False))
        print(f"\nCSV yazıldı:\n  {fold_csv}\n  {external_csv}\n  {summary_csv}")
        return

    target_event_rate = args.n_train_events / args.n_train_patients
    external_target_event_rate = (
        args.n_external_events / args.n_external_patients
        if args.n_external_events is not None
        else None
    )

    print(
        "=== stability_frequency_threshold duyarlilik analizi (SENTETIK VERI) ===\n"
        f"thresholds={args.thresholds} n_replicates={args.n_replicates} "
        f"replicate_start={args.replicate_start} "
        f"outer_splits={args.outer_splits} inner_splits={args.inner_splits}\n"
        f"l1_ratio_grid={args.l1_ratio_grid} penalizer_grid={args.penalizer_grid} "
        f"n_bootstrap_stability={args.n_bootstrap_stability}\n"
        f"n_train={args.n_train_patients} hedef_olay={args.n_train_events} "
        f"(oran={target_event_rate:.4f}) | n_external={args.n_external_patients}"
        + (f" hedef_harici_olay={args.n_external_events} (oran={external_target_event_rate:.4f})"
           if args.n_external_events is not None else " (harici oran=egitimle ayni, eski davranis)")
        + "\n"
    )

    ground_truth = build_ground_truth()
    print(f"Gercek (true) ozellik sayisi: {len(ground_truth.true_features)} / {len(ground_truth.feature_columns)}")
    print(f"True features: {ground_truth.true_features}\n")

    t_start = time.time()
    fold_df, external_df = run_efficient_threshold_sweep(
        thresholds=args.thresholds,
        n_replicates=args.n_replicates,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        l1_ratio_grid=tuple(args.l1_ratio_grid),
        penalizer_grid=tuple(args.penalizer_grid),
        n_bootstrap_stability=args.n_bootstrap_stability,
        n_bootstrap_external_ci=args.n_bootstrap_external_ci,
        base_seed=args.base_seed,
        ground_truth=ground_truth,
        verbose=not args.quiet,
        n_train_patients=args.n_train_patients,
        target_event_rate=target_event_rate,
        n_external_patients=args.n_external_patients,
        external_target_event_rate=external_target_event_rate,
        replicate_start=args.replicate_start,
    )

    total_elapsed = time.time() - t_start
    summary_df = summarize(fold_df, external_df)

    fold_df.to_csv(fold_csv, index=False)
    external_df.to_csv(external_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nToplam sure: {total_elapsed:.1f}s\n")
    print("=== ÖZET (threshold başına) ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary_df.to_string(index=False))

    print(f"\nCSV yazıldı:\n  {fold_csv}\n  {external_csv}\n  {summary_csv}")


if __name__ == "__main__":
    main()
