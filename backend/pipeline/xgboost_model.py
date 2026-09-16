"""Hafta 4 -- XGBoost ikincil sınıflandırma motoru (12-ay sağkalım Yes/No
+ Low/Medium/High risk sınıfı).

BAĞLAM
------
`raw/plan/plan.txt` satır 480-490 (Nisa görevleri) + `raw/mimari/v45.txt`
Bölüm 6.2 ("Secondary Classification -- 12 Aylık Sağkalım Tahmini").

🔴 EN KRİTİK KURAL (CLAUDE.md, kilitli):
    "Cox -> XGBoost arası veri sızıntısı: XGBoost'a giden Cox skorları
    out-of-fold/nested CV ile üretilmeli, asla in-sample değil."

Bu modül bu kuralı İKİ KATMANDA zorlar:
  1. YAPISAL -- `generate_out_of_fold_cox_scores()` bir hastanın skorunu
     SADECE o hastayı hiç görmemiş bir Cox modelinden üretebilir; fonksiyon
     kendi içinde (a) `nested_cv_result` ile öz-doğrulama (reconstruct
     edilen final özellik kümesi `nested_cv_result`'ın raporladığıyla
     birebir uyuşmuyorsa `OutOfFoldReconstructionMismatchError`), (b)
     fold-bazlı train/test kesişim kontrolü, (c) skorlanan hasta kümesinin
     `frame` ile tam örtüştüğü kontrolü YAPAR -- sızıntı varsa RuntimeError
     ile SESSİZCE değil GÜRÜLTÜLÜ durur.
  2. TEST -- `tests/test_xgboost_model.py`'deki
     `test_generate_out_of_fold_cox_scores_is_leakage_free` bu garantiyi
     BAĞIMSIZ olarak (fonksiyonun kendi guard'larından ayrı bir ikinci
     kanıt yoluyla) doğrular.

Ayrıca `generate_in_sample_cox_scores()` / `generate_in_sample_xgboost_auc()`
BİLİNÇLİ OLARAK sızıntılı fonksiyonlardır -- SADECE Ege'nin doğrulama
görevi için (plan.txt satır 508: "XGBoost AUC'sini in-sample (yanlış)
hesaplanmış AUC ile karşılaştır -- sızıntı etkisini göster"). Bu
fonksiyonların ürettiği skor/AUC ASLA üretim/rapor sonucu olarak
KULLANILMAZ -- docstring'lerinde açıkça uyarılıyor.

YÖNTEM (out-of-fold Cox skoru) -- `pipeline/cox_model.py`'nin KENDİ
`run_nested_cv()`'siyle BİREBİR AYNI dış-fold bölünmesini (`StratifiedKFold
(n_splits=outer_splits, shuffle=True, random_state=seed)`) yeniden üretir
-- `tools/train_cox_week3.py::audit_nested_cv_fold_selection()`'ın
kullandığı AYNI "reconstruct + öz-doğrula" deseni (bkz. o fonksiyonun
docstring'i). `pipeline/cox_model.py`'ye TEK SATIR bile YAZILMADI --
sadece onun public + private (tek alt çizgili, "doğrudan çağırma"
sözleşmesiyle) yardımcı fonksiyonları import edilip ÇAĞRILIYOR.

RİSK SKORU ADLANDIRMASI -- `predict_log_partial_hazard()` kullanılıyor,
`api/predict.py::compute_risk_score_and_shap()`'in üretim `risk_score_
log_partial_hazard` alanıyla AYNI ölçek/yorum (bkz. o dosyanın satır
1169-1180) -- XGBoost'a giden Cox skoru, /predict'in yeni bir hastaya
üreteceği skorla AYNI tanıma sahip.

SANSÜR KARARI (12-ay hedef) -- bkz. `define_twelve_month_survival_target()`
docstring'i: 12 aydan ÖNCE sansürlenen (ne yaşadığı ne öldüğü net olmayan)
hastalar DÜŞÜRÜLÜR, sessizce değil SAYILARAK. IPCW gibi bir sansür-
ağırlıklandırma YAPILMADI -- 6 haftalık plan kapsamı dışı, açık risk
olarak bildiriliyor.

RİSK SINIFI EŞİKLERİ -- `compute_frozen_risk_thresholds()`: Low/Medium/
High sınırları EĞİTİM kohortunun OUT-OF-FOLD tahmini olasılıklarından
(in-sample DEĞİL) hesaplanır ve DONDURULUR -- `score_thresholds`
tablosunun omics'teki (tertile, donmuş) deseniyle AYNI disiplin.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from pipeline.cox_model import (
    DEFAULT_L1_RATIO_GRID,
    DEFAULT_PENALIZER_GRID,
    _bootstrap_stability_selection,
    _build_penalizer_argument,
    _score_hyperparameters_via_inner_cv,
    select_features_lasso,
    standardize_columns_fold_safe,
)
from pipeline.reduce_collinearity import (
    DEFAULT_CORRELATION_CLUSTER_THRESHOLD,
    DEFAULT_CV_THRESHOLD,
    build_v3_candidate_pool,
)

__all__ = [
    "OutOfFoldReconstructionMismatchError",
    "OutOfFoldLeakageError",
    "XGBoostLeakageError",
    "OutOfFoldFoldAudit",
    "OutOfFoldCoxScoreResult",
    "generate_out_of_fold_cox_scores",
    "verify_out_of_fold_leakage_free",
    "generate_in_sample_cox_scores",
    "TwelveMonthLabelReport",
    "define_twelve_month_survival_target",
    "apply_twelve_month_target",
    "summarize_excluded_censoring_bias",
    "build_xgboost_feature_frame",
    "DEFAULT_XGB_PARAM_GRID",
    "XGBoostNestedCVResult",
    "train_xgboost_nested_cv",
    "generate_in_sample_xgboost_auc",
    "FrozenRiskThresholds",
    "compute_frozen_risk_thresholds",
    "classify_risk",
    "assign_risk_class_column",
    "DEFAULT_TWELVE_MONTH_THRESHOLD_DAYS",
    "DEFAULT_COX_OOF_SCORE_COLUMN",
    "AlignedXGBoostFoldAudit",
    "CoxFitProvenanceRecord",
    "AlignedCoxXGBoostResult",
    "train_xgboost_with_fold_aligned_cox_scores",
    "verify_aligned_fold_leakage_free",
    "FinalXGBoostPipelineResult",
    "fit_final_cox_and_xgboost_pipeline_on_full_pool",
    "verify_full_pool_pipeline_leakage_free",
    "ExternalXGBoostEvaluationResult",
    "evaluate_xgboost_external_test",
    "DEFAULT_COX_PENALIZER_ESCALATION_GRID",
    "CoxPenalizerEscalationExhaustedError",
    "CoxFitFailureRecord",
    "hash_feature_columns",
    "CoxFeatureCandidatePoolOverlapError",
    "assert_no_clinical_standardize_feature_overlap",
]


DEFAULT_TWELVE_MONTH_THRESHOLD_DAYS = 365.0
DEFAULT_COX_OOF_SCORE_COLUMN = "cox_oof_score"

# Ş2 / K16-a (Codex sartli onay, 2026-09-11): `_fit_cox_with_full_
# selection()`'un FINAL `.fit()` cagrisi -- hiperparametre grid taramasi
# + elastic-net + stabilite secimi BITTIKTEN SONRAKI tek fit -- `lifelines.
# exceptions.ConvergenceError` (ya da onun sardigi `numpy.linalg.
# LinAlgError` tekillik durumu) ile basarisiz olursa denenecek eskalasyon
# grid'i. `DEFAULT_PENALIZER_GRID` (0.05, 0.1, 0.2, 0.5, 1.0) TAMAMEN
# denenmis ve grid-taramasi SONUCUNDA "en iyi" secilmis bir penalizer'in
# YINE final fit'te tekillige dusmesi durumunda devreye girer -- bu yuzden
# grid'in UST UCUNDAN (1.0) baslar. Ampirik olarak dogrulandi (bkz.
# tests/test_xgboost_model.py'deki singular-matrix reprodüksiyonu): ayni
# near-constant/tam-kolineer sentetik veri penalizer=0.05'te ConvergenceError
# veriyor, penalizer>=0.1'de (l1_ratio=1.0 ile bile) TEMIZ fit ediyor.
DEFAULT_COX_PENALIZER_ESCALATION_GRID: tuple[float, ...] = (2.0, 5.0, 10.0, 25.0, 50.0, 100.0)


class CoxPenalizerEscalationExhaustedError(RuntimeError):
    """Ş2 / K16-a: final Cox `.fit()` cagrisi, hem grid-taramasinin
    sectigi baslangic penalizer'i HEM `DEFAULT_COX_PENALIZER_ESCALATION_
    GRID`'deki (ya da CLI'dan verilmis) TUM eskalasyon degerleriyle
    `lifelines.exceptions.ConvergenceError`/`numpy.linalg.LinAlgError`
    ile basarisiz oldugunda firlatilir. K16-b (`--allow-fold-skip`)
    SADECE bu istisna sinifini yakalayip o dis-fold'u/final-fit'i
    ATLAMAK icin kullanilabilir -- BASKA hicbir istisna sinifi burada
    "atlanabilir" sayilmaz."""


@dataclass
class CoxFitFailureRecord:
    """Ş3 (crash-manifest) -- final Cox fit penalizer-eskalasyon
    zincirindeki HER BASARISIZ deneme icin bir kayit (basarili son deneme
    BURAYA EKLENMEZ, sadece `CoxFitProvenanceRecord`'a). `tools/
    train_xgboost_week4.py::main()` bu kayitlarin BIRIKTIRILMIS listesini,
    pipeline COKERSE (escalation da tukenip `--allow-fold-skip` KAPALIYSA)
    ANINDA diske (`*_crash_manifest.json`) yazar -- bkz. o dosyadaki
    crash-manifest yazicisi."""

    outer_fold: int | None
    inner_fold: int | None
    fit_type: str
    attempt_index: int
    penalizer: float
    penalizer_escalation_grid: list[float]
    candidate_feature_columns: list[str]
    candidate_feature_columns_hash: str
    error_class: str
    error_message: str
    timestamp_utc: str


def hash_feature_columns(columns: Iterable[str]) -> str:
    """Ş3/Ş5 -- bir aday-ozellik listesinin DETERMINISTIK (sirasiz --
    once sort edilir) sha256 kisa-hash'i. Crash-manifest'te VE fold-bazli
    filtre raporunda (Ş5) ayni listenin farkli kayitlar arasinda ayni
    hash'i uretmesi icin kullanilir."""

    digest = hashlib.sha256()
    for column in sorted(columns):
        digest.update(column.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def _collinearity_report_to_dict(
    report: "V3CandidatePoolReport",
    *,
    outer_fold: int | None,
    inner_fold: int | None,
    scope: str,
) -> dict[str, Any]:
    """Ş5 (Codex şartlı onay, 2026-09-11) -- `pipeline.reduce_collinearity.
    build_v3_candidate_pool()`'un döndürdüğü `V3CandidatePoolReport`
    (ÖNCEDEN her outer/inner fold çağrısında `_`-önekli/kullanılmayan
    değişkene atanıp SESSİZCE ATILIYORDU) burada JSON/CSV-yazılabilir bir
    sözlüğe çevrilir -- düşürülen özellik listeleri (near-constant VE
    korelasyon-kümesi bazında, hangi temsilcinin yerine geçtiği dahil) +
    kullanılan eşikler + girdi/çıktı özellik sayıları TAMAMEN korunur,
    yalnız aggregate önce/sonra SAYISI (eski `AlignedXGBoostFoldAudit`
    alanları) DEĞİL."""

    correlation_dropped_records = report.correlation_dropped.to_dict(orient="records")
    return {
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "scope": scope,  # "outer_final" | "cross_fit_final" | "full_pool_final"
        "cv_threshold": report.cv_threshold,
        "corr_threshold": report.corr_threshold,
        "n_input_features": report.n_input_features,
        "n_kept_features": report.n_kept_features,
        "n_dropped_total": report.n_dropped_total,
        "near_constant_dropped": list(report.near_constant_dropped),
        "n_near_constant_dropped": len(report.near_constant_dropped),
        "correlation_dropped": correlation_dropped_records,
        "n_correlation_dropped": len(correlation_dropped_records),
        "n_multi_member_clusters": report.n_multi_member_clusters,
        "kept_features": list(report.kept_features),
    }


class CoxFeatureCandidatePoolOverlapError(RuntimeError):
    """Ş4 (Codex sartli onay, 2026-09-11): `clinical_standardize_columns`
    (Z-score/RCS gibi standardize edilen klinik kovaryatlar) ile Cox aday
    radyomik havuzu (`cox_feature_columns`/`feature_columns`) arasinda
    KESISIM varsa firlatilir.

    NEDEN ONEMLI: cross-fit filtresi (`pipeline.xgboost_model.
    train_xgboost_with_fold_aligned_cox_scores()`/`fit_final_cox_and_
    xgboost_pipeline_on_full_pool()`) `build_v3_candidate_pool()`'u HAM
    (standardize EDILMEMIS) `outer_train_raw`/`inner_train_raw` uzerinde
    cagirir; UCSF dagitim Cox filtresi (`tools/train_xgboost_week4.py::
    run_ucsf_external_evaluation()`) AYNI fonksiyonu STANDARDIZE EDILMIS
    `full_pool_training_frame` uzerinde cagirir. Bu ikisi FARKLI reçeteler
    (bilincli -- bkz. ilgili docstring'ler), AMA `clinical_standardize_
    columns ∩ feature_columns` BOS DEGILSE, o ORTAK kolon(lar) icin
    near-constant/korelasyon istatistigi HAM'a karsi STANDARDIZE-EDILMIS
    deger uzerinden hesaplanir -- iki yol FARKLI aday havuzu uretebilir,
    sessizce. Bu guard bu SESSIZ riski calisma ANINDA (import-zamani
    degil, cagri-zamani) yakalar.

    ⚠️ 2026-09-12 EKLENDİ (v3b reçete-sadakati düzeltmesi, Barış: "sapmayla
    koşmayalım, v3b ile XGBoost tam uyumluluk göstermesi lazım"): bu guard
    SADECE ÇAĞIRANIN (dışarıdan) verdiği `clinical_standardize_columns`'ı
    (tipik olarak yalnız yaş/RCS) `cox_feature_columns` (FİLTRE ÖNCESİ,
    HAM aday havuzu) ile karşılaştırır -- bu invariant DEĞİŞMEDİ, hâlâ
    TAM olarak eskisi gibi korunuyor. `train_xgboost_with_fold_aligned_
    cox_scores()`/`fit_final_cox_and_xgboost_pipeline_on_full_pool()`
    içinde `reduce_collinearity=True` iken artık AYRICA, `build_v3_
    candidate_pool()` HAM veriden (`outer_train_raw`/`inner_train_raw`/
    `inner_train`) sonucunu ÜRETTİKTEN SONRA, o SEÇİLMİŞ (post-filter)
    alt kümeyi (`outer_cox_candidate_columns`/`inner_cox_candidate_
    columns`/`split_cox_candidate_columns`) fold/split-yerel olarak
    standardize eden YENİ, İÇSEL bir adım var (`tools/train_cox_week3.py::
    V3VariantConfig.standardize_radiomics=True` ile AYNI reçete --
    `clinical_standardize_columns = feature_columns + clinical_age_
    columns`, ORADA da filtre SONRASI uygulanır). Bu YENİ adım bu guard'ı
    ASLA çağırmaz/tetiklemez ve invariantı ZAYIFLATMAZ -- çünkü sırası
    yapısal olarak GARANTİLİDİR: standardizasyon HER ZAMAN `build_v3_
    candidate_pool()`'un HAM veriyi tükettiği çağrıdan SONRA yapılır,
    yani near-constant/korelasyon istatistiği ASLA standardize edilmiş
    değer üzerinden hesaplanmaz (bu class'ın önlediği TAM riskin
    KENDİSİ). Dış (guard'lı) kanal ile iç (post-filter) kanal KASITLI
    olarak AYRI tutulur -- karıştırılmamalı."""


def assert_no_clinical_standardize_feature_overlap(
    feature_columns: Iterable[str],
    clinical_standardize_columns: Iterable[str],
    *,
    context: str,
) -> None:
    """Ş4 -- bkz. `CoxFeatureCandidatePoolOverlapError` docstring'i.
    Kesisim BOS DEGILSE fail-loud durur, mesaj kesisen kolonlari VE
    NEDEN sorun oldugunu ACIKCA soyler."""

    overlap = sorted(set(feature_columns) & set(clinical_standardize_columns))
    if overlap:
        raise CoxFeatureCandidatePoolOverlapError(
            f"{context}: `clinical_standardize_columns` ile Cox aday "
            f"radyomik havuzu ARASINDA KESISIM VAR -- {overlap[:10]}"
            f"{'...' if len(overlap) > 10 else ''}. Bu kolon(lar) icin "
            "cross-fit filtresi HAM degerle, dagitim filtresi STANDARDIZE-"
            "EDILMIS degerle calisir -- iki yol SESSIZCE FARKLI aday "
            "havuzu uretebilir (bkz. CoxFeatureCandidatePoolOverlapError "
            "docstring'i). Klinik standardizasyon listesi ile radyomik "
            "ozellik listesi AYRIK olmalidir."
        )


def _is_cox_singularity_exception(exc: BaseException) -> bool:
    """Ş2 / K16-a: SADECE `lifelines.exceptions.ConvergenceError` (ve
    bazi surum/yollarda lifelines'a ulasmadan once dogrudan firlayabilen
    `numpy.linalg.LinAlgError`) DAR kapsaminda taninir -- baska HICBIR
    istisna sinifi burada "eskalasyon ile kurtarilabilir" sayilmaz, aynen
    yukari firlatilir (fail-loud korunur)."""

    from lifelines.exceptions import ConvergenceError

    return isinstance(exc, (ConvergenceError, np.linalg.LinAlgError))


def _fit_final_cox_model_with_penalizer_escalation(
    fit_frame: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    final_features: list[str],
    extra_columns: list[str],
    initial_penalizer: float,
    l1_ratio: float,
    extra_column_penalizer: float | None,
    penalizer_escalation_grid: tuple[float, ...],
    error_context: str,
    outer_fold: int | None,
    inner_fold: int | None,
    fit_type: str,
    candidate_feature_columns: list[str],
    failure_sink: list[CoxFitFailureRecord] | None,
) -> tuple[Any, float, bool, list[dict[str, Any]]]:
    """Ş2 / K16-a (Codex sartli onay, 2026-09-11): final Cox `.fit()`
    cagrisini ONCE `initial_penalizer` (hiperparametre grid-taramasinin
    sectigi deger) ile, BASARISIZ olursa SIRAYLA `penalizer_escalation_
    grid`'deki degerlerle dener. SADECE `_is_cox_singularity_exception()`
    (ConvergenceError/LinAlgError) yakalanir -- BASKA HICBIR istisna
    sinifi burada YUTULMAZ, aynen yukari firlatilir.

    Basarili olursa: `(fitted_model, kullanilan_penalizer, escalated_mi,
    attempt_log)`. `attempt_log` -- HER denemenin (basarisiz olanlar
    DAHIL) `{attempt, penalizer, success, error_class, error_message}`
    kaydini tutar (provenance/crash-manifest'in girdisi).

    TUM grid tukenirse `CoxPenalizerEscalationExhaustedError` firlatir --
    `failure_sink` verilmisse HER basarisiz denemeyi (crash-manifest icin)
    zaten BIRIKTIRMIS olur (fonksiyon coksa BILE bu kayitlar cagiranin
    elindeki listede KALICI kalir, ayrica sarma gerekmez)."""

    from lifelines import CoxPHFitter

    attempts: list[dict[str, Any]] = []
    tried_penalizers = [initial_penalizer] + [
        p for p in penalizer_escalation_grid if p != initial_penalizer
    ]
    last_exc: BaseException | None = None
    for attempt_index, penalizer_value in enumerate(tried_penalizers):
        penalizer_argument = _build_penalizer_argument(
            final_features,
            extra_columns,
            penalizer=penalizer_value,
            extra_column_penalizer=extra_column_penalizer,
        )
        model = CoxPHFitter(penalizer=penalizer_argument, l1_ratio=l1_ratio)
        try:
            model.fit(fit_frame, duration_col=duration_col, event_col=event_col)
        except Exception as exc:  # noqa: BLE001 -- daraltma hemen asagida
            if not _is_cox_singularity_exception(exc):
                # Ş2: bu dar kapsamin DISINDAKI istisnalar (orn. l1_ratio
                # araligi/kolon hatasi -- kod/girdi hatasi isareti) burada
                # ASLA yutulmaz.
                raise
            attempt_record = {
                "attempt": attempt_index,
                "penalizer": penalizer_value,
                "success": False,
                "error_class": type(exc).__name__,
                "error_message": str(exc),
                "escalated": attempt_index > 0,
            }
            attempts.append(attempt_record)
            if failure_sink is not None:
                failure_sink.append(
                    CoxFitFailureRecord(
                        outer_fold=outer_fold,
                        inner_fold=inner_fold,
                        fit_type=fit_type,
                        attempt_index=attempt_index,
                        penalizer=penalizer_value,
                        penalizer_escalation_grid=list(penalizer_escalation_grid),
                        candidate_feature_columns=list(candidate_feature_columns),
                        candidate_feature_columns_hash=hash_feature_columns(candidate_feature_columns),
                        error_class=type(exc).__name__,
                        error_message=str(exc),
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    )
                )
            last_exc = exc
            continue
        attempts.append(
            {
                "attempt": attempt_index,
                "penalizer": penalizer_value,
                "success": True,
                "error_class": None,
                "error_message": None,
                "escalated": attempt_index > 0,
            }
        )
        return model, penalizer_value, attempt_index > 0, attempts

    raise CoxPenalizerEscalationExhaustedError(
        f"{error_context}: final Cox fit, baslangic penalizer'i "
        f"({initial_penalizer}) VE TUM eskalasyon grid'i "
        f"({list(penalizer_escalation_grid)}) denendikten SONRA HALA "
        f"{type(last_exc).__name__ if last_exc else '?'} ile basarisiz "
        f"oldu -- son hata: {last_exc}."
    ) from last_exc


# =====================================================================
# BÖLÜM 1 -- Out-of-fold Cox skor üretimi (sızıntısız, yapısal garantili)
# =====================================================================


class OutOfFoldReconstructionMismatchError(RuntimeError):
    """`generate_out_of_fold_cox_scores()`'un reconstruct ettiği dış-fold
    özellik seçimi, verilen `nested_cv_result`'ın raporladığıyla
    UYUŞMUYOR -- ya `frame`/`feature_columns`/`outer_splits`/`seed`
    parametreleri `nested_cv_result`'ı üreten `run_nested_cv()` çağrısıyla
    AYNI değil, ya da ikisi arasında bir kod-senkron sorunu var. Sessizce
    yanlış bir out-of-fold skor seti ÜRETİLMEZ."""


class OutOfFoldLeakageError(RuntimeError):
    """YAPISAL sızıntı kontrolü başarısız oldu. Bu, `StratifiedKFold`'un
    kendi garantisinin (train/test kesişimi boş) bir İKİNCİ savunma
    katmanıdır -- normal koşulda ASLA tetiklenmemesi beklenir; tetiklenirse
    bir kodlama hatasına işaret eder ve sessizce geçilmez."""


class XGBoostLeakageError(RuntimeError):
    """XGBoost'un KENDİ nested-CV döngüsünde yapısal sızıntı kontrolü
    başarısız oldu (train/test kesişimi boş değil ya da bir hasta birden
    fazla dış fold'da skorlanmış) -- `OutOfFoldLeakageError` ile AYNI
    sınıf garanti, XGBoost'un kendi fold döngüsü için."""


@dataclass
class OutOfFoldFoldAudit:
    """Bir dış-fold'un tam denetim izi -- hangi hastaların EĞİTİMDE,
    hangilerinin SKORLANDIĞI (test) ve hangi hiperparametre/özellik
    kümesiyle üretildiği. `verify_out_of_fold_leakage_free()`'nin girdisi."""

    fold: int
    train_patient_ids: frozenset
    test_patient_ids: frozenset
    n_train: int
    n_test: int
    best_penalizer: float
    best_l1_ratio: float
    final_features: list[str]


@dataclass
class OutOfFoldCoxScoreResult:
    scores: pd.DataFrame  # index=patient_id, kolonlar=[fold, <score_column>]
    fold_audit: list[OutOfFoldFoldAudit]
    outer_splits: int
    seed: int
    score_column: str


def generate_out_of_fold_cox_scores(
    frame: pd.DataFrame,
    feature_columns: list[str],
    nested_cv_result: pd.DataFrame,
    *,
    duration_col: str = "survival_days",
    event_col: str = "event",
    extra_columns: list[str] | None = None,
    outer_splits: int = 5,
    seed: int = 42,
    extra_column_penalizer: float | None = None,
    clinical_standardize_columns: list[str] | None = None,
    stability_selection: bool = True,
    n_bootstrap_stability: int = 200,
    stability_frequency_threshold: float = 0.6,
    score_column: str = DEFAULT_COX_OOF_SCORE_COLUMN,
) -> OutOfFoldCoxScoreResult:
    """Her hasta için, O HASTAYI HİÇ GÖRMEMİŞ bir Cox modelinden üretilmiş
    "dürüst" risk skoru döndürür (`predict_log_partial_hazard`, /predict'in
    üretim `risk_score_log_partial_hazard` alanıyla AYNI ölçek).

    ZORUNLU TUTARLILIK -- `nested_cv_result`, BU çağrıdakiyle BİREBİR AYNI
    `frame`/`feature_columns`/`extra_columns`/`outer_splits`/`seed`/
    `stability_frequency_threshold`/`extra_column_penalizer`/
    `clinical_standardize_columns` değerleriyle `pipeline.cox_model.
    run_nested_cv()` çağrılarak üretilmiş OLMALI (aynı arm'ın nihai Cox
    sonucu). Bu fonksiyon `run_nested_cv()`'yi TEKRAR ÇAĞIRMAZ (pahalı
    hiperparametre grid taramasını tekrarlamaz) -- SADECE her fold için
    `best_penalizer`/`best_l1_ratio`'yu `nested_cv_result`'tan okur, final
    özellik kümesini (elastic-net + stabilite, `run_nested_cv()` ile
    BİREBİR AYNI mantık) yeniden üretir ve reconstruct edilen kümeyi
    `nested_cv_result["selected_features"]` ile KARŞILAŞTIRIR --
    uyuşmazsa `OutOfFoldReconstructionMismatchError` (sessizce yanlış bir
    out-of-fold skor seti üretmek yerine GÜRÜLTÜLÜ durur).

    `nested_cv_result["selected_features"]` hem `list[str]` (bellek-içi,
    `run_nested_cv()`'nin döndürdüğü ham DataFrame) hem `";"`-ile-ayrılmış
    `str` (diskten okunmuş `week3_*_fold_results.csv`, bkz.
    `tools/train_cox_week3.py::write_variant_outputs()`) biçimini kabul
    eder.

    Sonra final Cox modeli SADECE dış-fold'un EĞİTİM alt-kümesinde
    (`train_frame`) fit edilir ve SADECE dış-fold'un TEST alt-kümesi
    (`test_frame` -- eğitimde HİÇ kullanılmayan hastalar) skorlanır.

    YAPISAL SIZINTI GARANTİSİ (RuntimeError ile zorlanır, opsiyonel
    değil):
      1. Her fold için train/test hasta kümesi kesişimi BOŞ olmalı.
      2. Skorlanan hasta kümesi o fold'un test kümesiyle BİREBİR aynı
         olmalı (train kümesiyle kesişmemeli).
      3. Tüm fold'lardaki skorlanan hastaların BİRLEŞİMİ `frame.index`
         ile BİREBİR aynı olmalı (her hasta TAM BİR KEZ skorlanmış).
    Bu kontroller `OutOfFoldLeakageError` fırlatır -- normal koşulda
    `StratifiedKFold`'un kendi garantisi zaten bunu sağlar, burada
    İKİNCİ bir savunma katmanı olarak AYRICA doğrulanıyor.
    """

    from lifelines import CoxPHFitter
    from sklearn.model_selection import StratifiedKFold

    extra_columns = list(extra_columns or [])
    clinical_standardize_columns = list(clinical_standardize_columns or [])

    outer_splitter = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    fold_audit: list[OutOfFoldFoldAudit] = []
    score_rows: list[pd.DataFrame] = []

    for fold_index, (train_idx, test_idx) in enumerate(
        outer_splitter.split(frame, frame[event_col])
    ):
        train_frame = frame.iloc[train_idx]
        test_frame = frame.iloc[test_idx]
        train_frame, test_frame = standardize_columns_fold_safe(
            train_frame, test_frame, clinical_standardize_columns
        )

        matches = nested_cv_result.loc[nested_cv_result["fold"] == fold_index]
        if matches.empty:
            raise OutOfFoldReconstructionMismatchError(
                f"Fold {fold_index}: nested_cv_result icinde bulunamadi -- "
                "outer_splits/seed/frame bu cagriyla nested_cv_result'i "
                "ureten run_nested_cv() cagrisi arasinda UYUSMUYOR olabilir."
            )
        fold_row = matches.iloc[0]
        best_penalizer = float(fold_row["best_penalizer"])
        best_l1_ratio = float(fold_row["best_l1_ratio"])
        reported_selected = _coerce_selected_features(fold_row["selected_features"])

        en_selected, _, _ = select_features_lasso(
            train_frame,
            feature_columns,
            duration_col=duration_col,
            event_col=event_col,
            extra_columns=extra_columns,
            penalizer=best_penalizer,
            l1_ratio=best_l1_ratio,
            extra_column_penalizer=extra_column_penalizer,
        )

        if stability_selection:
            stable_features, _ = _bootstrap_stability_selection(
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
            final_features = stable_features if stable_features else en_selected
        else:
            final_features = en_selected

        if sorted(final_features) != sorted(reported_selected):
            raise OutOfFoldReconstructionMismatchError(
                f"Fold {fold_index}: reconstruct edilen final ozellik "
                f"kumesi ({sorted(final_features)}) nested_cv_result'in "
                f"raporladigi selected_features ile ({sorted(reported_selected)}) "
                "UYUSMUYOR -- parametreler run_nested_cv() cagrisiyla AYNI "
                "olmayabilir. Sessizce yanlis bir out-of-fold skor seti "
                "URETILMEDI."
            )

        final_model_columns = final_features + extra_columns
        penalizer_argument = _build_penalizer_argument(
            final_features,
            extra_columns,
            penalizer=best_penalizer,
            extra_column_penalizer=extra_column_penalizer,
        )
        cox_final = CoxPHFitter(penalizer=penalizer_argument, l1_ratio=best_l1_ratio)
        cox_final.fit(
            train_frame[final_model_columns + [duration_col, event_col]],
            duration_col=duration_col,
            event_col=event_col,
        )
        oof_score = cox_final.predict_log_partial_hazard(test_frame[final_model_columns])

        train_ids = frozenset(train_frame.index)
        test_ids = frozenset(test_frame.index)
        if train_ids & test_ids:
            raise OutOfFoldLeakageError(
                f"Fold {fold_index}: train/test hasta kumesi kesisimi BOS "
                f"DEGIL -- {sorted(train_ids & test_ids)[:5]}..."
            )

        score_rows.append(
            pd.DataFrame({"fold": fold_index, score_column: oof_score.astype(float)})
        )
        fold_audit.append(
            OutOfFoldFoldAudit(
                fold=fold_index,
                train_patient_ids=train_ids,
                test_patient_ids=test_ids,
                n_train=len(train_frame),
                n_test=len(test_frame),
                best_penalizer=best_penalizer,
                best_l1_ratio=best_l1_ratio,
                final_features=list(final_features),
            )
        )

    scores = pd.concat(score_rows).sort_index()

    if scores.index.duplicated().any():
        dupes = sorted(set(scores.index[scores.index.duplicated()]))
        raise OutOfFoldLeakageError(
            f"Bir/birden fazla hasta BIRDEN FAZLA fold'da skorlanmis -- "
            f"{dupes[:5]}... -- StratifiedKFold'un garantisi ihlal edilmis."
        )
    if set(scores.index) != set(frame.index):
        missing = set(frame.index) - set(scores.index)
        extra = set(scores.index) - set(frame.index)
        raise OutOfFoldLeakageError(
            "Skorlanan hasta kumesi frame.index ile BIREBIR ORTUSMUYOR -- "
            f"eksik={sorted(missing)[:5]}... fazla={sorted(extra)[:5]}..."
        )

    return OutOfFoldCoxScoreResult(
        scores=scores,
        fold_audit=fold_audit,
        outer_splits=outer_splits,
        seed=seed,
        score_column=score_column,
    )


def _coerce_selected_features(value: Any) -> list[str]:
    """`nested_cv_result["selected_features"]` hucresini `list[str]`'e
    cevirir -- hem bellek-ici (`run_nested_cv()`'nin ham cikti listesi)
    hem diskten okunmus `";"`-ile-ayrilmis string (`week3_*_fold_results.
    csv`) formati kabul edilir. Bos string -> bos liste (BOS ozellik
    kumesi gecerli bir durum, `run_nested_cv()`'nin `extra_columns`
    varken bunu urettigi durum var)."""

    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        if value == "":
            return []
        return value.split(";")
    raise TypeError(
        f"selected_features hucresi beklenmeyen tipte: {type(value)!r} -- "
        "list[str] ya da ';'-ile-ayrilmis str olmali."
    )


def verify_out_of_fold_leakage_free(result: OutOfFoldCoxScoreResult) -> None:
    """`generate_out_of_fold_cox_scores()`'un KENDİ İÇİNDE zaten yaptığı
    sızıntı kontrollerini, BAĞIMSIZ bir ikinci yoldan (sadece dönen
    `OutOfFoldCoxScoreResult`'a bakarak, üretim mantığını TEKRARLAMADAN)
    yeniden doğrular. Görev talimatındaki "fold atamalarını kaydet, her
    skorun hangi modelden geldiğini izle, kesişim boş olmalı" önerisinin
    TEST-KATMANI karşılığıdır -- `tests/test_xgboost_model.py` bunu
    çağırır. Başarısızsa `OutOfFoldLeakageError`."""

    all_scored_patients: set[Any] = set()
    for audit in result.fold_audit:
        overlap = audit.train_patient_ids & audit.test_patient_ids
        if overlap:
            raise OutOfFoldLeakageError(
                f"Fold {audit.fold}: train/test kesisimi BOS DEGIL: "
                f"{sorted(overlap)[:5]}..."
            )
        fold_scores = result.scores.loc[result.scores["fold"] == audit.fold]
        scored_patients = set(fold_scores.index)
        if scored_patients != audit.test_patient_ids:
            raise OutOfFoldLeakageError(
                f"Fold {audit.fold}: skorlanan hasta kumesi ({sorted(scored_patients)[:5]}...) "
                f"fold audit'in test kumesiyle ({sorted(audit.test_patient_ids)[:5]}...) UYUSMUYOR."
            )
        if scored_patients & audit.train_patient_ids:
            raise OutOfFoldLeakageError(
                f"Fold {audit.fold}: skorlanan hastalarin bir kismi AYNI "
                "fold'un EGITIM kumesinde de var -- SIZINTI."
            )
        all_scored_patients |= scored_patients

    if all_scored_patients != set(result.scores.index):
        raise OutOfFoldLeakageError(
            "Fold audit'lerin birlesimi, result.scores.index ile UYUSMUYOR."
        )


def generate_in_sample_cox_scores(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    duration_col: str = "survival_days",
    event_col: str = "event",
    extra_columns: list[str] | None = None,
    penalizer: float,
    l1_ratio: float,
    extra_column_penalizer: float | None = None,
    stability_selection: bool = True,
    n_bootstrap_stability: int = 200,
    stability_frequency_threshold: float = 0.6,
    seed: int = 42,
) -> tuple[pd.Series, list[str]]:
    """⚠️ SADECE Ege'nin sızıntı-karşılaştırma testi için (plan.txt satır
    508: "XGBoost AUC'sini in-sample (yanlış) hesaplanmış AUC ile
    karşılaştır"). Cox modeli `frame`'in TAMAMINI hem EĞİTİR hem SKORLAR
    -- yani her hastanın skoru, O HASTAYI ZATEN GÖRMÜŞ bir modelden gelir.

    🔴 BU FONKSİYONUN ÇIKTISI XGBoost'un GERÇEK eğitim girdisi olarak
    ASLA KULLANILMAZ -- kullanılırsa CLAUDE.md'nin "Cox -> XGBoost arası
    veri sızıntısı" kuralı doğrudan İHLAL EDİLMİŞ olur. Bu fonksiyon
    yalnızca "sızıntı olsaydı sonuç ne kadar iyimser görünürdü" sorusunu
    ölçülebilir kılmak için var.
    """

    from lifelines import CoxPHFitter

    extra_columns = list(extra_columns or [])

    en_selected, _, _ = select_features_lasso(
        frame,
        feature_columns,
        duration_col=duration_col,
        event_col=event_col,
        extra_columns=extra_columns,
        penalizer=penalizer,
        l1_ratio=l1_ratio,
        extra_column_penalizer=extra_column_penalizer,
    )
    if stability_selection:
        stable_features, _ = _bootstrap_stability_selection(
            frame,
            feature_columns,
            duration_col=duration_col,
            event_col=event_col,
            extra_columns=extra_columns,
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            n_bootstrap=n_bootstrap_stability,
            frequency_threshold=stability_frequency_threshold,
            seed=seed,
            extra_column_penalizer=extra_column_penalizer,
        )
        final_features = stable_features if stable_features else en_selected
    else:
        final_features = en_selected

    final_model_columns = final_features + extra_columns
    penalizer_argument = _build_penalizer_argument(
        final_features, extra_columns, penalizer=penalizer, extra_column_penalizer=extra_column_penalizer
    )
    cox_final = CoxPHFitter(penalizer=penalizer_argument, l1_ratio=l1_ratio)
    cox_final.fit(
        frame[final_model_columns + [duration_col, event_col]],
        duration_col=duration_col,
        event_col=event_col,
    )
    in_sample_score = cox_final.predict_log_partial_hazard(frame[final_model_columns])
    return in_sample_score.astype(float), list(final_features)


# =====================================================================
# BÖLÜM 2 -- 12-ay sağkalım hedef değişkeni (sansür ele alışı AÇIKÇA)
# =====================================================================


@dataclass
class TwelveMonthLabelReport:
    n_total: int
    n_survived_yes: int
    n_died_no: int
    n_excluded_ambiguous_censoring: int
    excluded_patient_ids: list
    threshold_days: float
    pct_excluded: float


def define_twelve_month_survival_target(
    frame: pd.DataFrame,
    *,
    duration_col: str = "survival_days",
    event_col: str = "event",
    threshold_days: float = DEFAULT_TWELVE_MONTH_THRESHOLD_DAYS,
) -> tuple[pd.Series, TwelveMonthLabelReport]:
    """12-ay sağkalım hedefini (1=Yes/hayatta, 0=No/öldü, NaN=belirsiz)
    türetir.

    KARAR (görev talimatının açıkça gerekçelendirilmesini istediği sansür
    sorunu):
      - `duration >= threshold_days`                    -> target=1 (Yes).
        Olay durumu (öldü/sansürlü) FARK ETMEZ -- ikisi de "hasta 12 ayı
        GÖRDÜ" bilgisini taşır (12 aydan SONRA öldüyse de "12 ay hayatta
        kaldı" doğrudur).
      - `duration <  threshold_days` VE `event==1` (öldü)   -> target=0
        (No) -- kesin, sansüre bağlı değil.
      - `duration <  threshold_days` VE `event==0` (sansürlü) -> target=
        NaN, hasta target-bağımlı adımlarda (XGBoost eğitimi)
        DÜŞÜRÜLÜR. GEREKÇE: bu hastanın 12. ayda gerçekte ne durumda
        olacağı BİLİNMİYOR (takip 12 aydan önce kesildi) -- "hayatta
        sayma" ya da "öldü sayma" ikisi de UYDURMA olur. Bu SESSİZCE
        yapılmaz: `TwelveMonthLabelReport` düşürülen hasta sayısını VE
        kimliklerini taşır, `summarize_excluded_censoring_bias()` ile
        yanlılık AYRICA ölçülebilir.
      - REDDEDİLEN ALTERNATİF: IPCW (inverse probability of censoring
        weighting) ile bu hastaları AĞIRLIKLANDIRARAK dahil etmek --
        istatistiksel olarak daha doğru ama 6 haftalık plan kapsamı
        DIŞINDA (plan.txt'de bu yöntem hiç anılmıyor); AÇIK RİSK olarak
        bildiriliyor, sessizce atlanmadı.
      - `threshold_days=365,0` (365,25 DEĞİL) -- protokolde SAYISAL
        olarak KİLİTLENMEDİ, bu modülün varsayılanı. Gerçek veriyle
        duyarlılık kontrolü (365 vs 365,25) ÖNERİLİR, YAPILMADI.

    2026-08-19 EKLENDİ (Codex çapraz inceleme MEDIUM 4 -- task-
    mszuse9v-efm1gw): `duration_col`/`event_col` GİRDİ DOĞRULAMASI
    EKLENDİ -- eskiden geçersiz bir satır (örn. `survival_days` NaN/
    negatif/sonsuz, `event` {0,1} dışında bir değer) `>=`/`==`
    karşılaştırmalarında SESSİZCE `False` üretip hem `survived` hem
    `died_before` hem `ambiguous` maskelerinde `False` kalıyordu --
    yani hasta SESSİZCE ne "dahil" ne "hariç tutulan-belirsiz" listesine
    giriyordu (üçüncü, GÖRÜNMEZ bir kategori). Şimdi böyle bir satır
    varsa `ValueError` ile GÜRÜLTÜLÜ durur, hasta kimlikleriyle."""

    duration = frame[duration_col]
    event = frame[event_col]

    invalid_duration_mask = duration.isna() | ~np.isfinite(duration.astype(float)) | (duration < 0)
    invalid_event_mask = ~event.isin([0, 1, 0.0, 1.0]) | event.isna()
    invalid_mask = invalid_duration_mask | invalid_event_mask
    if invalid_mask.any():
        invalid_ids = list(frame.index[invalid_mask])
        raise ValueError(
            f"{len(invalid_ids)} hasta icin gecersiz girdi -- "
            f"'{duration_col}' sonlu VE negatif-olmayan, '{event_col}' TAM "
            f"{{0,1}} OLMALI (sessizce dusurulmedi, GORULMEZ ucuncu bir "
            f"kategoriye SIZMASIN diye once burada durduruldu). Ilk "
            f"kimlikler: {sorted(str(pid) for pid in invalid_ids)[:10]}..."
        )

    survived = duration >= threshold_days
    died_before = (duration < threshold_days) & (event == 1)
    ambiguous = (duration < threshold_days) & (event == 0)

    target = pd.Series(np.nan, index=frame.index, dtype=float)
    target.loc[survived] = 1.0
    target.loc[died_before] = 0.0

    n_total = len(frame)
    n_excluded = int(ambiguous.sum())
    report = TwelveMonthLabelReport(
        n_total=n_total,
        n_survived_yes=int(survived.sum()),
        n_died_no=int(died_before.sum()),
        n_excluded_ambiguous_censoring=n_excluded,
        excluded_patient_ids=list(frame.index[ambiguous]),
        threshold_days=threshold_days,
        pct_excluded=(n_excluded / n_total * 100.0) if n_total else float("nan"),
    )
    return target, report


def apply_twelve_month_target(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    target_col: str = "target_12mo_survival",
    report: TwelveMonthLabelReport | None = None,
) -> pd.DataFrame:
    """`target`'ı `frame`'e ekler, NaN (belirsiz-sansürlü, bkz.
    `define_twelve_month_survival_target()`) olan satırları DÜŞÜRÜR.
    Sessizce değil -- çağıran taraf düşürülen sayıyı zaten
    `TwelveMonthLabelReport`'tan biliyor olmalı.

    2026-08-19 EKLENDİ (Codex çapraz inceleme MEDIUM 4): `report`
    verilirse (`define_twelve_month_survival_target()`'ın AYNI `target`'ı
    ürettiği çağrının raporu), bu fonksiyonun FİİLEN düşürdüğü hasta
    kimlik kümesi `report.excluded_patient_ids` ile BİREBİR eşleşmeli --
    eşleşmezse (örn. çağıran taraf farklı bir `target`/`report` çiftini
    yanlışlıkla eşleştirdiyse) `ValueError` ile durur, rapor ile fiili
    düşme SESSİZCE AYRIŞMAZ."""

    result = frame.copy()
    result[target_col] = target
    dropped_mask = target.isna()
    dropped_ids = set(frame.index[dropped_mask])

    if report is not None:
        reported_ids = set(report.excluded_patient_ids)
        if dropped_ids != reported_ids:
            only_dropped = sorted(str(pid) for pid in (dropped_ids - reported_ids))[:10]
            only_reported = sorted(str(pid) for pid in (reported_ids - dropped_ids))[:10]
            raise ValueError(
                "apply_twelve_month_target(): FIILEN dusurulen hasta kumesi "
                "report.excluded_patient_ids ile UYUSMUYOR -- 'target'/'report' "
                "farkli cagrilardan gelmis olabilir. Sadece fiilen "
                f"dusurulende olanlar (ilk 10): {only_dropped}... Sadece "
                f"raporda olanlar (ilk 10): {only_reported}..."
            )

    return result.dropna(subset=[target_col])


def summarize_excluded_censoring_bias(
    frame: pd.DataFrame,
    excluded_mask: pd.Series,
    *,
    continuous_columns: list[str] | None = None,
    binary_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Dışlanan (12 aydan önce sansürlenmiş, belirsiz-durum) hastaların,
    kalan kohorttan istatistiksel olarak farklı olup olmadığını ölçer --
    "sessizce düşürme" riskini görünür kılmak için (UCSF seçim-yanlılığı
    raporunun -- `decisions/2026-08-18-tek-model-ucsf-harici-test-k15-
    kapanisi.md` -- AYNI disipliniyle: p<0,05 fark varsa AÇIKÇA
    raporlanmalı, sonuç gizlenmez).

    `continuous_columns` (örn. `clinical_age`) için Mann-Whitney U,
    `binary_columns` (örn. `clinical_gender_male`) için ki-kare (2x2
    beklenen frekans yeterliyse) ya da Fisher'in kesin testi kullanılır
    (`scipy.stats` -- proje zaten `scikit-learn` üzerinden zorunlu
    bağımlılık olarak taşıyor, YENİ bir `requirements.txt` girdisi
    GEREKMEZ).
    """

    from scipy import stats

    continuous_columns = continuous_columns or []
    binary_columns = binary_columns or []

    rows: list[dict[str, Any]] = []
    excluded = frame.loc[excluded_mask]
    retained = frame.loc[~excluded_mask]

    for column in continuous_columns:
        excluded_values = excluded[column].dropna()
        retained_values = retained[column].dropna()
        if len(excluded_values) == 0 or len(retained_values) == 0:
            p_value = float("nan")
        else:
            _, p_value = stats.mannwhitneyu(
                excluded_values, retained_values, alternative="two-sided"
            )
        rows.append(
            {
                "column": column,
                "type": "continuous",
                "excluded_mean": float(excluded_values.mean()) if len(excluded_values) else float("nan"),
                "retained_mean": float(retained_values.mean()) if len(retained_values) else float("nan"),
                "n_excluded": int(len(excluded_values)),
                "n_retained": int(len(retained_values)),
                "p_value": float(p_value),
            }
        )

    for column in binary_columns:
        excluded_values = excluded[column].dropna()
        retained_values = retained[column].dropna()
        table = [
            [int(excluded_values.sum()), int(len(excluded_values) - excluded_values.sum())],
            [int(retained_values.sum()), int(len(retained_values) - retained_values.sum())],
        ]
        try:
            _, p_value = stats.fisher_exact(table)
        except ValueError:
            p_value = float("nan")
        rows.append(
            {
                "column": column,
                "type": "binary",
                "excluded_mean": float(excluded_values.mean()) if len(excluded_values) else float("nan"),
                "retained_mean": float(retained_values.mean()) if len(retained_values) else float("nan"),
                "n_excluded": int(len(excluded_values)),
                "n_retained": int(len(retained_values)),
                "p_value": float(p_value),
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# BÖLÜM 3 -- XGBoost özellik matrisi (Cox oof skoru + radyomik + klinik)
# =====================================================================


def build_xgboost_feature_frame(
    cox_scores: pd.Series,
    feature_frame: pd.DataFrame,
    *,
    score_column: str = DEFAULT_COX_OOF_SCORE_COLUMN,
) -> pd.DataFrame:
    """Cox skorunu (out-of-fold ya da -- SADECE karşılaştırma amaçlı --
    in-sample) radyomik+klinik matrisiyle `patient_id` (index) üzerinden
    birleştirir. Cox skoru olmayan hasta SESSİZCE düşürülmez -- `feature_
    frame`'deki HER hastanın bir skoru olmalı (aksi hâlde bu, out-of-fold
    üretiminde bir hastanın atlandığı anlamına gelir, ki bu başlı başına
    bir sızıntı/veri-bütünlüğü sorunu belirtisidir)."""

    missing = set(feature_frame.index) - set(cox_scores.index)
    if missing:
        raise ValueError(
            f"{len(missing)} hasta icin Cox skoru YOK: {sorted(missing)[:5]}... "
            "-- out-of-fold uretiminde bu hastalar atlanmis olabilir."
        )
    combined = feature_frame.copy()
    combined[score_column] = cox_scores.reindex(combined.index).astype(float)
    return combined


# =====================================================================
# BÖLÜM 4 -- XGBoost nested-CV eğitimi (out-of-fold AUC + in-sample AUC)
# =====================================================================


DEFAULT_XGB_PARAM_GRID: dict[str, list[Any]] = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 200],
    "min_child_weight": [1, 5],
}


def _iter_param_grid(grid: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    keys = list(grid.keys())
    for combo in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, combo))


def _score_xgb_params_via_inner_cv(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_col: str,
    params: dict[str, Any],
    inner_splits: int,
    seed: int,
) -> float:
    """Bir XGBoost hiperparametre kombinasyonu için İÇ CV ortalama AUC'si
    -- `pipeline.cox_model._score_hyperparameters_via_inner_cv()` ile AYNI
    tasarım deseni (dış fold'un HİPERPARAMETRE seçimi de kendi içinde
    sızıntısız olmalı)."""

    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    scores: list[float] = []
    for inner_train_idx, inner_test_idx in splitter.split(train_frame, train_frame[target_col]):
        inner_train = train_frame.iloc[inner_train_idx]
        inner_test = train_frame.iloc[inner_test_idx]
        if inner_train[target_col].nunique() < 2 or inner_test[target_col].nunique() < 2:
            continue
        model = xgb.XGBClassifier(**params, eval_metric="logloss", random_state=seed)
        model.fit(inner_train[feature_columns], inner_train[target_col])
        predicted_probability = model.predict_proba(inner_test[feature_columns])[:, 1]
        scores.append(roc_auc_score(inner_test[target_col], predicted_probability))

    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _bootstrap_auc_ci(
    true_label: Iterable[float],
    predicted_probability: Iterable[float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """`pipeline.cox_model.evaluate_external_test()`'in bootstrap-CI
    deseniyle AYNI (percentile bootstrap) -- AUC için. Tek-eşik dili
    YASAK kuralı gereği HER ZAMAN nokta tahmini + CI birlikte döner."""

    from sklearn.metrics import roc_auc_score

    y = np.asarray(list(true_label), dtype=float)
    p = np.asarray(list(predicted_probability), dtype=float)
    n = len(y)
    if n == 0 or len(np.unique(y)) < 2:
        return {
            "auc": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "confidence_level": confidence_level,
            "n_bootstrap_valid": 0,
            "n_bootstrap_requested": n_bootstrap,
            "n_patients": n,
            "n_positive": int(y.sum()) if n else 0,
        }

    point_estimate = float(roc_auc_score(y, p))

    rng = np.random.default_rng(seed)
    bootstrap_scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue  # bootstrap orneginde tek sinif -- AUC tanimsiz, atla
        bootstrap_scores.append(roc_auc_score(y[idx], p[idx]))

    alpha = 1 - confidence_level
    if bootstrap_scores:
        lower = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))
    else:
        lower = upper = float("nan")

    return {
        "auc": point_estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence_level": confidence_level,
        "n_bootstrap_valid": len(bootstrap_scores),
        "n_bootstrap_requested": n_bootstrap,
        "n_patients": n,
        "n_positive": int(y.sum()),
    }


@dataclass
class XGBoostNestedCVResult:
    fold_results: pd.DataFrame
    out_of_fold_predictions: pd.DataFrame  # index=patient_id: predicted_probability/true_label/fold
    auc_out_of_fold: dict[str, Any]
    fold_audit: list[dict[str, Any]] = field(default_factory=list)


def train_xgboost_nested_cv(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_col: str = "target_12mo_survival",
    outer_splits: int = 5,
    inner_splits: int = 5,
    seed: int = 42,
    param_grid: dict[str, list[Any]] | None = None,
    n_bootstrap_ci: int = 1000,
    confidence_level: float = 0.95,
) -> XGBoostNestedCVResult:
    """XGBoost'un KENDİ nested-CV döngüsü -- hiperparametre araması İÇ
    CV'de (dış-eğitim İÇİNDE), final fit dış-eğitimin TAMAMINDA, tahmin
    SADECE dış-test'te (`predict_proba`, eğitimde hiç görülmemiş hastalar)
    -- Cox tarafındaki `run_nested_cv()` ile AYNI iskelet.

    `feature_columns` tipik olarak `[cox_oof_score_kolonu] + radyomik +
    klinik` listesidir -- bu fonksiyon hangi kolonların "Cox skoru" olduğu
    konusunda bir VARSAYIM YAPMAZ, çağıran taraf `build_xgboost_feature_
    frame()`'in ürettiği kolonu `feature_columns`'a dahil eder.

    YAPISAL SIZINTI GARANTİSİ: her dış fold için train/test kesişimi BOŞ
    olmalı, tüm dış-test kümelerinin birleşimi `frame.index` ile BİREBİR
    örtüşmeli, hiçbir hasta birden fazla fold'da tahmin edilmemeli --
    ihlalde `XGBoostLeakageError`.
    """

    from sklearn.model_selection import StratifiedKFold

    param_grid = param_grid or DEFAULT_XGB_PARAM_GRID
    outer_splitter = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    fold_audit: list[dict[str, Any]] = []

    import xgboost as xgb

    for fold_index, (train_idx, test_idx) in enumerate(
        outer_splitter.split(frame, frame[target_col])
    ):
        train_frame = frame.iloc[train_idx]
        test_frame = frame.iloc[test_idx]

        best_score = float("-inf")
        best_params: dict[str, Any] | None = None
        for params in _iter_param_grid(param_grid):
            score = _score_xgb_params_via_inner_cv(
                train_frame,
                feature_columns,
                target_col=target_col,
                params=params,
                inner_splits=inner_splits,
                seed=seed,
            )
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_params = params

        if best_params is None:
            raise RuntimeError(
                f"Dis fold {fold_index}: XGBoost ic-CV grid taramasindaki "
                "HICBIR parametre kombinasyonu gecerli bir AUC uretmedi -- "
                "grid/inner_splits/n gozden gecirilmeli."
            )

        model = xgb.XGBClassifier(**best_params, eval_metric="logloss", random_state=seed)
        model.fit(train_frame[feature_columns], train_frame[target_col])
        predicted_probability = model.predict_proba(test_frame[feature_columns])[:, 1]

        train_ids = frozenset(train_frame.index)
        test_ids = frozenset(test_frame.index)
        if train_ids & test_ids:
            raise XGBoostLeakageError(
                f"Fold {fold_index}: train/test hasta kumesi kesisimi BOS "
                f"DEGIL -- {sorted(train_ids & test_ids)[:5]}..."
            )

        fold_rows.append(
            {
                "fold": fold_index,
                "n_train": len(train_frame),
                "n_test": len(test_frame),
                "n_positive_train": int(train_frame[target_col].sum()),
                "n_positive_test": int(test_frame[target_col].sum()),
                "inner_mean_auc": best_score,
                "best_params": best_params,
            }
        )
        for patient_id, probability, true_label in zip(
            test_frame.index, predicted_probability, test_frame[target_col]
        ):
            oof_rows.append(
                {
                    "patient_id": patient_id,
                    "fold": fold_index,
                    "predicted_probability": float(probability),
                    "true_label": float(true_label),
                }
            )
        fold_audit.append(
            {"fold": fold_index, "train_patient_ids": train_ids, "test_patient_ids": test_ids}
        )

    oof_frame = pd.DataFrame(oof_rows).set_index("patient_id")

    if oof_frame.index.duplicated().any():
        dupes = sorted(set(oof_frame.index[oof_frame.index.duplicated()]))
        raise XGBoostLeakageError(
            f"Bir/birden fazla hasta BIRDEN FAZLA dis fold'da tahmin edilmis "
            f"-- {dupes[:5]}..."
        )
    if set(oof_frame.index) != set(frame.index):
        missing = set(frame.index) - set(oof_frame.index)
        extra = set(oof_frame.index) - set(frame.index)
        raise XGBoostLeakageError(
            "out-of-fold tahmin kumesi frame.index ile BIREBIR ORTUSMUYOR -- "
            f"eksik={sorted(missing)[:5]}... fazla={sorted(extra)[:5]}..."
        )

    auc_out_of_fold = _bootstrap_auc_ci(
        oof_frame["true_label"],
        oof_frame["predicted_probability"],
        n_bootstrap=n_bootstrap_ci,
        seed=seed,
        confidence_level=confidence_level,
    )

    return XGBoostNestedCVResult(
        fold_results=pd.DataFrame(fold_rows),
        out_of_fold_predictions=oof_frame,
        auc_out_of_fold=auc_out_of_fold,
        fold_audit=fold_audit,
    )


def generate_in_sample_xgboost_auc(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_col: str = "target_12mo_survival",
    params: dict[str, Any],
    seed: int = 42,
    n_bootstrap_ci: int = 1000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """⚠️ SADECE Ege'nin sızıntı-karşılaştırma testi için (plan.txt satır
    508). XGBoost `frame`'in TAMAMINI hem EĞİTİR hem SKORLAR -- AUC bu
    yüzden İYİMSER/YANLIŞ. `raw/mimari/v45.txt` satır 1760-1761:
    "Nested/out-of-fold Cox skorlarıyla üretilmiş XGBoost AUC'si ile
    in-sample (yanlış) hesaplanmış AUC arasındaki fark CI raporunda
    ayrıca gösterilir." -- bu fonksiyon o karşılaştırmanın "yanlış" ucunu
    üretir, ASLA raporun GERÇEK sonucu olarak KULLANILMAZ."""

    import xgboost as xgb

    model = xgb.XGBClassifier(**params, eval_metric="logloss", random_state=seed)
    model.fit(frame[feature_columns], frame[target_col])
    predicted_probability = model.predict_proba(frame[feature_columns])[:, 1]
    return _bootstrap_auc_ci(
        frame[target_col],
        pd.Series(predicted_probability, index=frame.index),
        n_bootstrap=n_bootstrap_ci,
        seed=seed,
        confidence_level=confidence_level,
    )


# =====================================================================
# BÖLÜM 5 -- Donmuş risk sınıfı eşikleri (Low/Medium/High)
# =====================================================================


@dataclass
class FrozenRiskThresholds:
    """Low/Medium/High sınırları -- EĞİTİM kohortunun OUT-OF-FOLD tahmini
    olasılıklarından hesaplanır ve DONDURULUR (in-sample DEĞİL -- bkz.
    modül docstring'i). `score_thresholds` tablosunun omics'teki
    (donmuş tertile) deseniyle AYNI disiplin, ama BU tablo XGBoost için
    DEĞİL -- `model_registry`/ayrı bir alan gerekebilir, bu ARTEFAKT
    şimdilik SADECE bellek-içi/JSON dosyası olarak üretiliyor."""

    low_upper_bound: float  # P(12-ay hayatta) <= bu -> risk YUKSEK
    medium_upper_bound: float  # bu < P <= medium_upper_bound -> risk ORTA
    method: str
    n_training_patients: int
    frozen_at: str
    source_score: str


def compute_frozen_risk_thresholds(
    out_of_fold_predicted_probability: pd.Series,
    *,
    method: str = "tertile",
    source_score: str = "xgboost_predicted_probability_out_of_fold",
) -> FrozenRiskThresholds:
    """Tertile (P33/P66) sınır noktalarını EĞİTİM kohortunun OUT-OF-FOLD
    tahmini olasılıklarından hesaplar.

    🔴 AÇIK TASARIM SORUSU (Barış'a soru, KİLİTLENMEDİ): tertile mi sabit
    olasılık eşiği mi (örn. <%40/%40-70/>%70) kullanılmalı -- protokol
    (plan.txt/v45.txt) bunu SAYISAL olarak belirlemiyor. Bu fonksiyonun
    yazarı (modeling-agent) `score_thresholds`'un omics tarafındaki
    (db-agent, 2026-08-18) tertile deseniyle TUTARLILIK için tertile'ı
    seçti -- gerçek eğitim verisiyle duyarlılık karşılaştırması
    (tertile vs sabit eşik) YAPILMADI.

    NEDEN out-of-fold (in-sample DEĞİL): `dna_repair_score`'un kohort-
    bağımlılığı bulgusuyla (AKTIF-GOREVLER.md db-agent girdisi,
    2026-08-18) AYNI SINIF risk -- in-sample olasılıklardan tertile
    hesaplanırsa sınırlar modelin GERÇEKTE üreteceğinden daha keskin/
    iyimser ayrışır, üretimde sınıf kayması olur (omics'in `score_
    thresholds` boşken yaşadığı sorunla AYNI)."""

    values = out_of_fold_predicted_probability.dropna()
    if method != "tertile":
        raise NotImplementedError(f"Desteklenmeyen yontem: {method!r} -- yalniz 'tertile' var.")
    p33 = float(np.percentile(values, 33))
    p66 = float(np.percentile(values, 66))
    return FrozenRiskThresholds(
        low_upper_bound=p33,
        medium_upper_bound=p66,
        method=method,
        n_training_patients=int(len(values)),
        frozen_at=datetime.now(timezone.utc).isoformat(),
        source_score=source_score,
    )


def classify_risk(predicted_probability: float, thresholds: FrozenRiskThresholds) -> str:
    """`predicted_probability` = XGBoost'un ürettiği P(12-ay hayatta).

    YÖN (bilinçli, ters DEĞİL): DÜŞÜK P(hayatta) = YÜKSEK risk. Bu yüzden
    en alt tertil (`<= low_upper_bound`) "high", en üst tertil
    (`> medium_upper_bound`) "low" olarak etiketlenir."""

    if predicted_probability <= thresholds.low_upper_bound:
        return "high"
    if predicted_probability <= thresholds.medium_upper_bound:
        return "medium"
    return "low"


def assign_risk_class_column(
    predicted_probability: pd.Series, thresholds: FrozenRiskThresholds
) -> pd.Series:
    """`classify_risk()`'in vektörleştirilmiş hâli -- bir kolonun
    tamamına uygulanır."""

    return predicted_probability.apply(lambda value: classify_risk(value, thresholds))


# =====================================================================
# BÖLÜM 6 -- Fold-hizali (aligned) Cox + XGBoost zinciri
# =====================================================================
#
# 2026-08-19 EKLENDİ -- Codex çapraz inceleme (task-mszuse9v-efm1gw)
# CRITICAL 1 düzeltmesi.
#
# SORUN (Codex bulgusu): `generate_out_of_fold_cox_scores()` (BÖLÜM 1)
# `pipeline.cox_model.run_nested_cv()`'nin KENDİ dış-fold bölünmesini
# (event_col'a göre stratified) yeniden üretir -- bu bölünme XGBoost'un
# `train_xgboost_nested_cv()` (BÖLÜM 4) içinde KENDİ AYRI dış-fold
# bölünmesinden (target_col'a göre stratified, FARKLI bir StratifiedKFold
# çağrısı) TAMAMEN BAĞIMSIZDIR. Bir hasta P'nin kendi Cox skoru
# "dürüst"tür (P'yi hiç görmemiş bir Cox modelinden gelir) AMA P,
# XGBoost'un test-fold'unda olsa bile, P'nin verisi BAŞKA bir Cox-fold'un
# (Q hastasının skorunu üreten) EĞİTİM kümesinde bulunabilir -- çünkü iki
# fold yapısı hizalı değildir. Bu, P'nin bilgisinin Q'nun Cox skoru
# ÜZERİNDEN, Q XGBoost'un eğitim kümesindeyken, DOLAYLI olarak sızmasına
# yol açar -- P kendi test-fold'unda değerlendirilirken XGBoost zaten
# P'yle "ilişkili" bir sinyali (Q'nun skoru aracılığıyla) eğitimde görmüş
# olabilir.
#
# DÜZELTME: outer döngü ARTIK XGBoost'un kendi fold'udur (`target_col`'a
# göre stratified `StratifiedKFold`). Her outer-fold'da:
#   1. Cox hiperparametre araması (`_score_hyperparameters_via_inner_cv`)
#      + elastic-net seçimi + bootstrap stabilite seçimi SADECE
#      `outer_train` üzerinde çalışır -- `outer_test` bu adımların
#      HİÇBİRİNDE görünmez.
#   2. `outer_train`'in KENDİ İÇİNDE bir K-fold cross-fit ile (Codex:
#      "dış-eğitimin meta-girdileri için içeride cross-fitting")
#      `outer_train`'in HER hastasına, O HASTAYI GÖRMEMİŞ (ama
#      `outer_test`'i de HİÇBİR ZAMAN görmemiş) bir Cox modelinden
#      "dürüst" bir meta-skor üretilir -- bu, XGBoost'un `outer_train`
#      eğitim girdisi olur.
#   3. `outer_train`'in TAMAMINDA final bir Cox modeli fit edilir,
#      SADECE `outer_test`'i skorlamak için kullanılır (bu hastalar
#      hiçbir Cox fit'inin girdisinde YOKTUR).
#   4. XGBoost'un kendi hiperparametre araması + final fit + tahmini
#      AYNI outer_train/outer_test bölünmesini kullanır.
#
# SONUÇ: bu fold'daki HER Cox fit'inin (hiperparametre taraması +
# elastic-net + stabilite + iç cross-fit + final fit) girdisi
# `outer_train_ids` ile SINIRLIDIR -- `outer_test_ids` bu adımların
# HİÇBİRİNDE kullanılmaz. Bu, kod-yapısı gereği zaten böyledir; AYRICA
# hasta-kimliği tabanlı olarak (`AlignedXGBoostFoldAudit`) doğrulanır
# (`XGBoostLeakageError`/`OutOfFoldLeakageError` -- sessizce geçilmez).
#
# ESKİ FONKSİYONLAR (BÖLÜM 1 + BÖLÜM 4) SİLİNMEDİ -- `generate_in_sample_
# cox_scores()`/`generate_in_sample_xgboost_auc()` gibi BİLİNÇLİ sızıntılı
# karşılaştırma fonksiyonları hâlâ geçerli (Ege'nin görevi). AMA
# `generate_out_of_fold_cox_scores()` + `train_xgboost_nested_cv()`'nin
# İKİ AYRI ÇAĞRI olarak zincirlenmesi (eski `tools/train_xgboost_week4.py::
# run_xgboost_week4_pipeline()`'in yaptığı gibi) ARTIK "DOĞRU/üretim"
# yolu DEĞİLDİR -- bkz. `tools/train_xgboost_week4.py`'deki
# `run_xgboost_week4_pipeline_aligned()` (yeni birincil yol) ve eski
# `run_xgboost_week4_pipeline()`'in docstring'ine eklenen uyarı.


class _AlignedPipelineInternalError(RuntimeError):
    """Bu modülün kendi kodlama hatasına işaret eder (fold_local_extra_
    column_builder sözleşmesini ihlal ederse) -- kullanıcı girdisi
    hatası DEĞİL."""


@dataclass
class AlignedXGBoostFoldAudit:
    """Bir dış-fold'un TAM denetim izi -- CRITICAL 1 düzeltmesinin
    yapısal kanıtı. `cox_fit_patient_ids`, bu fold'daki HER Cox fit'inin
    (iç-CV hiperparametre taraması + özellik seçimi + iç cross-fit +
    final fit) SADECE `outer_train_patient_ids` alt kümesinden veri
    gördüğünü -- yani `outer_test_patient_ids` ile HİÇBİR KESİŞİMİ
    olmadığını -- dışarı taşır (Codex bulgusu: "Hasta kimlikleri
    üzerinden Cox fit kümelerinin XGBoost test kümesiyle AYRIK olduğu
    YAPISAL olarak doğrulanmalı").

    ⚠️ HIGH 2 düzeltmesi (Codex 2. tur, 2026-08-19): `cox_fit_patient_ids`
    ARTIK bir NİYET ataması (`= outer_train_patient_ids`) DEĞİL -- bu
    fold'daki `fit_provenance` kayıtlarının (her GERÇEK `.fit()`
    çağrısının GÖRDÜĞÜ dataframe index'i) BİRLEŞİMİNDEN hesaplanır. Bu
    birleşimin yapısal olarak `outer_train_patient_ids`'e EŞİT olması
    beklenir (aşağıda `train_xgboost_with_fold_aligned_cox_scores()`
    içinde AYRICA doğrulanır) -- ama artık bu bir VARSAYIM değil,
    çalışma-zamanı ÖLÇÜMÜdür: kodda bir hata olsa (örn. bir iç-fold
    yanlışlıkla outer_test'ten bir satır görse) bu KAÇMAZ.

    B9 EKLENDİ (`reduce_collinearity` bayrağı, varsayılan KAPALI):
    `cox_candidate_features_before_collinearity_filter`/`_after_` bu
    fold'un DIŞ-final Cox seçimine giren aday havuzunun (`pipeline.
    reduce_collinearity.build_v3_candidate_pool()`'dan geçmeden önce/
    sonra) boyutunu şeffaf şekilde taşır -- bayrak KAPALIYSA ikisi
    EŞİTTİR (filtre uygulanmadı, sessiz davranış değişikliği YOK).
    Bu alanlar SADECE dış-final çağrısını yansıtır; her iç-fold'un
    KENDİ (potansiyel olarak farklı) daraltılmış havuzu `fit_provenance`
    kayıtlarında YOKTUR -- kapsam dışı, `CoxFitProvenanceRecord`'un
    KENDİ kapsam notundaki AYNI "final .fit() dışındakiler enstrümante
    edilmedi" sınırlaması burada da geçerli."""

    fold: int
    outer_train_patient_ids: frozenset
    outer_test_patient_ids: frozenset
    cox_fit_patient_ids: frozenset
    best_cox_penalizer: float
    best_cox_l1_ratio: float
    cox_final_features: list[str]
    n_cox_inner_splits: int
    best_xgb_params: dict[str, Any]
    fold_local_extra_columns: list[str] = field(default_factory=list)
    reduce_collinearity_enabled: bool = False
    cox_candidate_features_before_collinearity_filter: int = 0
    cox_candidate_features_after_collinearity_filter: int = 0


@dataclass
class CoxFitProvenanceRecord:
    """HIGH 2 düzeltmesi (Codex 2. tur), 3. tur (task-mt06mqzi-k0gb5j)
    KAPSAM DÜZELTMESİ ile: `_fit_cox_with_full_selection()`'ın kendi
    İÇİNDE, GERÇEK `lifelines.CoxPHFitter.fit()` çağrısından HEMEN ÖNCE
    o çağrıya giden dataframe'in GERÇEK index'ini yakalar (ÖNCEDEN --
    2. tur -- bu kayıt `_fit_cox_with_full_selection()` DÖNDÜKTEN SONRA
    çağıran tarafça "niyet" olarak üretiliyordu; Codex 3. tur bunu
    DOĞRULAMADI ve düzeltme istedi). `AlignedXGBoostFoldAudit.cox_fit_
    patient_ids` bu kayıtların birleşiminden TÜRETİLİR. `verify_aligned_
    fold_leakage_free()` bu kayıtları BAĞIMSIZ ikinci bir yoldan (üretim
    mantığını tekrarlamadan) doğrular; `tests/test_xgboost_model.py`'deki
    spy/wrapper testi bu kayıtların GERÇEKTEN lifelines `.fit()`'e giden
    index'le eşleştiğini (kodun kendi beyanından bağımsız) kanıtlar.

    ⚠️ KAPSAM (3. tur netleştirmesi, AÇIKÇA beyan): bu kayıt SADECE
    `_fit_cox_with_full_selection()`'ın SON/final `.fit()` çağrısını
    (elastic-net + stabilite seçimi BİTTİKTEN SONraki final model fit'i)
    kapsar. O fonksiyonun İÇİNDEKİ hiperparametre grid taramasının
    (`cox_model._score_hyperparameters_via_inner_cv()`, HER (penalizer,
    l1_ratio, iç-fold) kombinasyonu için AYRI bir `.fit()`) ve bootstrap
    stabilite seçiminin (`cox_model._bootstrap_stability_selection()`,
    HER bootstrap örneği için AYRI bir `.fit()`) İÇ `.fit()` çağırıları
    BU KAYITLARDA YER ALMAZ -- bu iki fonksiyon `pipeline/cox_model.py`
    içinde tanımlı ve bu görev talimatı o dosyada SADECE dar bir except-
    daraltmasına izin verdi, provenance enstrümantasyonu EKLEMEDİ.
    `fit_type` alanı bu nedenle FİİLEN sadece {"outer_final",
    "cross_fit_final", "full_pool_final"} değerlerini ALIR --
    "grid_selection"/"stability_bootstrap" (Codex'in istediği 5'li
    taksonominin kalan ikisi) bu turda ENSTRÜMANTE EDİLEMEDİ, kapsam
    dışı olarak raporlanmıştır (`cox_model.py`'ye dokunma kısıtı)."""

    outer_fold: int | None
    fit_type: str  # "outer_final" | "cross_fit_final" | "full_pool_final"
    inner_fold: int | None
    patient_ids: frozenset
    best_penalizer: float
    best_l1_ratio: float
    n_final_features: int
    success: bool = True
    error_message: str | None = None
    # Ş2 / K16-a (2026-09-11 EKLENDİ, varsayılanlı -- mevcut hiçbir çağıran
    # ETKİLENMEZ): `best_penalizer` HER ZAMAN hiperparametre grid-taramasının
    # SEÇTİĞİ (eskalasyon ÖNCESİ) değeri taşımaya devam eder (geriye dönük
    # uyumluluk) -- `used_penalizer`/`penalizer_escalated` başarılı fit'in
    # GERÇEKTE hangi penalizer'la üretildiğini AYRICA, şeffaf şekilde taşır.
    used_penalizer: float | None = None
    penalizer_escalated: bool = False
    penalizer_escalation_attempts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AlignedCoxXGBoostResult:
    fold_results: pd.DataFrame
    out_of_fold_predictions: pd.DataFrame  # index=patient_id: predicted_probability/true_label/fold
    cox_score_audit: pd.DataFrame  # index=patient_id: fold/<score_column>/role/inner_fold
    auc_out_of_fold: dict[str, Any]
    fold_audit: list[AlignedXGBoostFoldAudit] = field(default_factory=list)
    fit_provenance: list[CoxFitProvenanceRecord] = field(default_factory=list)
    # Ş2 / K16-b (2026-09-11 EKLENDİ, varsayılanlı -- `allow_fold_skip`
    # VARSAYILAN KAPALIYKEN HER ZAMAN boş kalır, davranış değişikliği YOK):
    # `--allow-fold-skip` açıkken, penalizer-eskalasyonu TÜKENDİĞİ için
    # ATLANAN dış-fold'ların kaydı (fold index + hangi fit_type'ta/hangi
    # iç-fold'da tükendiği + son hata mesajı).
    skipped_outer_folds: list[dict[str, Any]] = field(default_factory=list)
    # Ş3 (crash-manifest) -- final Cox fit penalizer-eskalasyon zincirindeki
    # HER BAŞARISIZ deneme (başarılı son deneme HARİÇ, o `fit_provenance`'ta).
    cox_fit_failure_log: list[CoxFitFailureRecord] = field(default_factory=list)
    # Ş5 (2026-09-11 EKLENDİ, varsayılanlı -- `reduce_collinearity=False`
    # KEN HER ZAMAN boş kalır): HER outer/inner `build_v3_candidate_pool()`
    # çağrısının TAM raporu (düşürülen özellikler + korelasyon kümeleri +
    # eşikler) -- bkz. `_collinearity_report_to_dict()`.
    collinearity_filter_reports: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _CoxSelectionFitResult:
    """`_fit_cox_with_full_selection()`'un dönüş değeri."""

    fitted_model: Any  # lifelines.CoxPHFitter
    final_model_columns: list[str]
    final_features: list[str]
    best_penalizer: float
    best_l1_ratio: float
    # Ş2 / K16-a: `best_penalizer` grid-taramasinin SECTIGI (eskalasyon
    # ONCESI) deger olarak KALIR (geriye-donuk uyumluluk); asagidaki ikisi
    # GERCEKTE kullanilan (escalasyon sonrasi olabilir) degeri taşır.
    used_penalizer: float = 0.0
    penalizer_escalated: bool = False
    penalizer_escalation_attempts: list[dict[str, Any]] = field(default_factory=list)


def _fit_cox_with_full_selection(
    train_frame: pd.DataFrame,
    cox_feature_columns: list[str],
    *,
    duration_col: str,
    event_col: str,
    extra_columns: list[str],
    penalizer_grid: tuple[float, ...],
    l1_ratio_grid: tuple[float, ...],
    inner_splits: int,
    hyperparameter_search_seed: int,
    stability_selection: bool,
    n_bootstrap_stability: int,
    stability_frequency_threshold: float,
    stability_seed: int,
    extra_column_penalizer: float | None,
    error_context: str,
    fit_type: str,
    outer_fold: int | None = None,
    inner_fold: int | None = None,
    provenance_sink: list[CoxFitProvenanceRecord] | None = None,
    penalizer_escalation_grid: tuple[float, ...] | None = None,
    failure_sink: list[CoxFitFailureRecord] | None = None,
) -> _CoxSelectionFitResult:
    """HIGH 1 düzeltmesi (Codex 2. tur, task-mt04rec5-jjqc95): TAM Cox
    model-seçim hattı -- hiperparametre grid taraması (iç-CV) + elastic-
    net özellik seçimi + (opsiyonel) bootstrap stabilite seçimi + final
    fit -- HEPSİ SADECE `train_frame` üzerinde. `train_xgboost_with_
    fold_aligned_cox_scores()`'un HEM dış-fold'un TEK final modeli
    (`outer_train_std` üzerinde, `outer_test`'i skorlamak için) HEM her
    iç-fold'un KENDİ bağımsız modeli (`inner_train_std` üzerinde, o iç-
    fold'un iç-test hastalarını skorlamak için) için ÇAĞRILIR.

    ÖNCEDEN (düzeltme öncesi): hiperparametre araması + özellik seçimi
    outer_train'in TAMAMINDA BİR KEZ yapılıp, iç cross-fit'in HER
    ic-fold'unda (sadece FİT `inner_train`'e daraltılarak) YENİDEN
    KULLANILIYORDU -- yani bir iç-test hastasının KENDİ verisi (
    outer_train'in bir parçası olarak) o hastayı SKORLAYACAK modelin
    HANGİ özellikleri/hiperparametreleri kullanacağını ETKİLEYEBİLİYORDU
    (klasik "kendi seçimini gördü" sızıntısı, fit'in kendisi sızıntısız
    olsa bile). ARTIK her çağrı -- outer-level de, HER iç-fold da --
    kendi `train_frame`'inde TAM VE BAĞIMSIZ bir arama+seçim+fit yapıyor;
    iç-fold çağrılarında `train_frame` HER ZAMAN o iç-fold'un `inner_
    train` alt kümesidir, iç-test hastası bu çağrının HİÇBİR adımına
    (ne hiperparametre taramasına, ne elastic-net'e, ne stabilite
    bootstrap'ına, ne final fit'e) GİRMEZ.

    3. TUR EKLENDİ (task-mt06mqzi-k0gb5j, HIGH -- "provenance kaydı
    .fit()'ten HEMEN ÖNCE değil, fonksiyon döndükten SONRA çağıran
    tarafından üretiliyor" bulgusu): `provenance_sink` verilirse, bu
    fonksiyonun SONUNDAKİ final `.fit()` çağrısından HEMEN ÖNCE o
    çağrıya giden dataframe'in GERÇEK index'i yakalanır ve bir
    `CoxFitProvenanceRecord` (`fit_type`/`outer_fold`/`inner_fold` ile
    etiketlenmiş) `provenance_sink`'e eklenir -- `.fit()` başarısız
    olursa `success=False`/`error_message` ile AYRICA eklenip istisna
    YENİDEN FIRLATILIR (sessizce yutulmaz). ⚠️ Bu SADECE bu fonksiyonun
    KENDİ final fit'ini kapsar -- yukarıdaki hiperparametre grid
    taraması VE bootstrap stabilite seçimi `pipeline/cox_model.py`
    içindeki AYRI fonksiyonlarda kendi `.fit()` çağrılarını yapar, bu
    görev talimatı o dosyaya provenance enstrümantasyonu eklemeye izin
    VERMEDİ (bkz. `CoxFitProvenanceRecord` docstring'indeki KAPSAM notu)."""

    best_score = float("-inf")
    best_penalizer: float | None = None
    best_l1_ratio: float | None = None
    for penalizer in penalizer_grid:
        for l1_ratio in l1_ratio_grid:
            score = _score_hyperparameters_via_inner_cv(
                train_frame,
                cox_feature_columns,
                duration_col=duration_col,
                event_col=event_col,
                extra_columns=extra_columns,
                penalizer=penalizer,
                l1_ratio=l1_ratio,
                inner_splits=inner_splits,
                seed=hyperparameter_search_seed,
                extra_column_penalizer=extra_column_penalizer,
            )
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_penalizer = penalizer
                best_l1_ratio = l1_ratio

    if best_penalizer is None or best_l1_ratio is None:
        raise RuntimeError(
            f"{error_context}: Cox ic-CV grid taramasindaki HICBIR "
            "(penalizer, l1_ratio) kombinasyonu gecerli bir skor uretmedi "
            "-- grid/inner_splits/n gozden gecirilmeli."
        )

    en_selected, _, _ = select_features_lasso(
        train_frame,
        cox_feature_columns,
        duration_col=duration_col,
        event_col=event_col,
        extra_columns=extra_columns,
        penalizer=best_penalizer,
        l1_ratio=best_l1_ratio,
        extra_column_penalizer=extra_column_penalizer,
    )
    if stability_selection:
        stable_features, _ = _bootstrap_stability_selection(
            train_frame,
            cox_feature_columns,
            duration_col=duration_col,
            event_col=event_col,
            extra_columns=extra_columns,
            penalizer=best_penalizer,
            l1_ratio=best_l1_ratio,
            n_bootstrap=n_bootstrap_stability,
            frequency_threshold=stability_frequency_threshold,
            seed=stability_seed,
            extra_column_penalizer=extra_column_penalizer,
        )
        final_features = stable_features if stable_features else en_selected
    else:
        final_features = en_selected

    if not final_features and not extra_columns:
        raise RuntimeError(
            f"{error_context}: Cox elastic-net/stabilite secimi hicbir "
            "ozellik birakmadi (extra_columns de yok) -- grid/esik "
            "gozden gecirilmeli."
        )

    final_model_columns = final_features + extra_columns
    # 3. tur duzeltmesi -- provenance index'i .fit()'e giden GERCEK
    # dataframe'den, cagridan HEMEN ONCE yakalanir (fonksiyon dondukten
    # SONRA cagiran tarafca degil).
    final_fit_frame = train_frame[final_model_columns + [duration_col, event_col]]
    final_fit_patient_ids = frozenset(final_fit_frame.index)

    # Ş2 / K16-a (Codex sartli onay, 2026-09-11): final `.fit()` cagrisi
    # ARTIK cıplak degil -- SADECE ConvergenceError/LinAlgError (tekillik)
    # icin dar kapsamli bir penalizer-eskalasyonu dener. Grid-taramasinin
    # SECTIGI `best_penalizer` HALA ILK denemedir -- eskalasyon SADECE o
    # basarisiz olursa devreye girer.
    escalation_grid = (
        tuple(penalizer_escalation_grid)
        if penalizer_escalation_grid is not None
        else DEFAULT_COX_PENALIZER_ESCALATION_GRID
    )
    try:
        fitted_model, used_penalizer, penalizer_escalated, escalation_attempts = (
            _fit_final_cox_model_with_penalizer_escalation(
                final_fit_frame,
                duration_col=duration_col,
                event_col=event_col,
                final_features=final_features,
                extra_columns=extra_columns,
                initial_penalizer=best_penalizer,
                l1_ratio=best_l1_ratio,
                extra_column_penalizer=extra_column_penalizer,
                penalizer_escalation_grid=escalation_grid,
                error_context=error_context,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                fit_type=fit_type,
                candidate_feature_columns=cox_feature_columns,
                failure_sink=failure_sink,
            )
        )
    except CoxPenalizerEscalationExhaustedError as exc:
        if provenance_sink is not None:
            provenance_sink.append(
                CoxFitProvenanceRecord(
                    outer_fold=outer_fold,
                    fit_type=fit_type,
                    inner_fold=inner_fold,
                    patient_ids=final_fit_patient_ids,
                    best_penalizer=best_penalizer,
                    best_l1_ratio=best_l1_ratio,
                    n_final_features=len(final_features),
                    success=False,
                    error_message=str(exc),
                )
            )
        raise
    except Exception as exc:
        # Ş2: bu dar kapsamin DISINDAKI istisnalar (_is_cox_singularity_
        # exception() FALSE dondugu icin escalasyon helper'in kendisi
        # AYNEN yeniden firlattigi durumlar) da AYNI sekilde provenance'a
        # basarisiz olarak islenir, sessizce yutulmaz.
        if provenance_sink is not None:
            provenance_sink.append(
                CoxFitProvenanceRecord(
                    outer_fold=outer_fold,
                    fit_type=fit_type,
                    inner_fold=inner_fold,
                    patient_ids=final_fit_patient_ids,
                    best_penalizer=best_penalizer,
                    best_l1_ratio=best_l1_ratio,
                    n_final_features=len(final_features),
                    success=False,
                    error_message=str(exc),
                )
            )
        raise

    if provenance_sink is not None:
        provenance_sink.append(
            CoxFitProvenanceRecord(
                outer_fold=outer_fold,
                fit_type=fit_type,
                inner_fold=inner_fold,
                patient_ids=final_fit_patient_ids,
                best_penalizer=best_penalizer,
                best_l1_ratio=best_l1_ratio,
                n_final_features=len(final_features),
                used_penalizer=used_penalizer,
                penalizer_escalated=penalizer_escalated,
                penalizer_escalation_attempts=escalation_attempts,
            )
        )

    return _CoxSelectionFitResult(
        fitted_model=fitted_model,
        final_model_columns=final_model_columns,
        final_features=final_features,
        best_penalizer=best_penalizer,
        best_l1_ratio=best_l1_ratio,
        used_penalizer=used_penalizer,
        penalizer_escalated=penalizer_escalated,
        penalizer_escalation_attempts=escalation_attempts,
    )


def train_xgboost_with_fold_aligned_cox_scores(
    frame: pd.DataFrame,
    cox_feature_columns: list[str],
    *,
    duration_col: str = "survival_days",
    event_col: str = "event",
    target_col: str = "target_12mo_survival",
    extra_columns: list[str] | None = None,
    clinical_standardize_columns: list[str] | None = None,
    fold_local_extra_column_builder: (
        Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame, list[str]]] | None
    ) = None,
    outer_splits: int = 5,
    seed: int = 42,
    cox_inner_splits: int = 5,
    cox_l1_ratio_grid: tuple[float, ...] | None = None,
    cox_penalizer_grid: tuple[float, ...] | None = None,
    cox_stability_selection: bool = True,
    cox_n_bootstrap_stability: int = 200,
    cox_stability_frequency_threshold: float = 0.6,
    cox_extra_column_penalizer: float | None = None,
    reduce_collinearity: bool = False,
    collinearity_cv_threshold: float = DEFAULT_CV_THRESHOLD,
    collinearity_corr_threshold: float = DEFAULT_CORRELATION_CLUSTER_THRESHOLD,
    xgb_inner_splits: int = 5,
    xgb_seed: int = 42,
    xgb_param_grid: dict[str, list[Any]] | None = None,
    xgb_n_bootstrap_ci: int = 1000,
    confidence_level: float = 0.95,
    score_column: str = DEFAULT_COX_OOF_SCORE_COLUMN,
    cox_penalizer_escalation_grid: tuple[float, ...] | None = None,
    allow_fold_skip: bool = False,
    cox_fit_failure_log: list[CoxFitFailureRecord] | None = None,
) -> AlignedCoxXGBoostResult:
    """Cox VE XGBoost'un TEK, ORTAK dış-fold döngüsü -- CRITICAL 1
    düzeltmesi (bkz. bu bölümün başındaki modül-seviyesi açıklama).

    `cox_fit_failure_log` (Ş3, crash-manifest, 2026-09-11 EKLENDİ):
    verilirse, İÇERİDE oluşturulan yeni bir liste yerine BU liste
    kullanılır -- ÇAĞIRAN taraf pipeline ÇÖKSE BİLE (escalation tükenip
    `allow_fold_skip=False` olduğunda) kendi elindeki bu liste referansı
    üzerinden O ANA KADAR biriken TÜM başarısız deneme kayıtlarına
    erişebilir (`tools/train_xgboost_week4.py::main()`'in crash-manifest
    yazıcısının dayandığı mekanizma). Verilmezse (varsayılan `None`)
    davranış eskisiyle AYNI -- fonksiyon kendi yerel listesini kurar.

    `cox_penalizer_escalation_grid`/`allow_fold_skip` (Ş2 / K16-a+b,
    Codex şartlı onay 2026-09-11): her outer-final VE her iç cross-fit
    Cox fit'i, `_fit_cox_with_full_selection()` üzerinden `DEFAULT_COX_
    PENALIZER_ESCALATION_GRID`'i (ya da burada verilen özel grid'i)
    KULLANARAK ConvergenceError/LinAlgError'a karşı dar-kapsamlı bir
    penalizer-eskalaması dener. Eskalasyon da TÜKENIRSE: `allow_fold_
    skip=False` (VARSAYILAN) iken davranış BİREBİR ESKİSİ GİBİDİR --
    `CoxPenalizerEscalationExhaustedError` (bir `RuntimeError` alt sınıfı)
    yukarı fırlar, pipeline çöker (fail-loud, sessizce YOK). `allow_fold_
    skip=True` verilirse, o TEK outer-fold'un TAMAMI (hem outer-final hem
    içindeki TÜM cross-fit'ler dahil edilmiş sayılır) `result.skipped_
    outer_folds`'a kaydedilip ATLANIR -- o fold'un test hastaları `out_of_
    fold_predictions`'a HİÇ GİRMEZ (AUC hesaplaması kalan fold'ların
    hastalarıyla yapılır). ⚠️ Bu iki parametre `reduce_collinearity` gibi
    VARSAYILAN KAPALI/eskisiyle-birebir-aynı ilkesine uyar: `allow_fold_
    skip` verilmezse (`False`) mevcut 7 Cox kolunun sonuçları etkilenmez
    -- eskalasyon YALNIZ gerçek bir ConvergenceError/LinAlgError anında
    devreye girer, `DEFAULT_PENALIZER_GRID` ile ZATEN başarılı olan hiçbir
    fit'in çıktısını DEĞİŞTİRMEZ (escalasyon zinciri `[initial_penalizer,
    *grid]` sırasında dener, `initial_penalizer` başarılıysa TEK bir
    deneme yapılır, sonuç eskisiyle bit-birebir aynıdır).

    `frame` `target_col`'u (12-ay hedefi, ambiguous-sansürlüler ZATEN
    düşürülmüş -- bkz. `apply_twelve_month_target()`) VE Cox'un ihtiyaç
    duyduğu `duration_col`/`event_col`'u AYNI ANDA içermelidir (bu
    fonksiyon `define_twelve_month_survival_target()`'ı ÇAĞIRMAZ --
    çağıran taraf hedefi ÖNCEDEN kurup ambiguous hastaları düşürmüş
    olmalı, BÖLÜM 2 fonksiyonlarıyla).

    `fold_local_extra_column_builder` (HIGH 3 düzeltmesi -- Codex: "RCS
    knot'ları tüm UPenn'den, CV dışında" -- fold-güvenli hâle getirildi):
    verilirse, HER outer-fold'da `(outer_train_raw, outer_test_raw)` ile
    çağrılır, `(outer_train_with_cols, outer_test_with_cols,
    new_column_names)` döner -- örn. RCS yaş düğümlerini SADECE
    `outer_train`'in ham yaşından hesaplayıp `restricted_cubic_spline_
    basis()`'i HEM `outer_train` HEM `outer_test`'e (öğrenilen düğümlerle)
    uygulamak için kullanılır (bkz. `tools/train_xgboost_week4.py::
    _build_fold_safe_age_spline_columns()`). Döndürülen `new_column_names`
    bu fold için `extra_columns`'a EKLENİR (Cox VE XGBoost girdisi olur).
    ⚠️ `clinical_standardize_columns`'a bu isimleri eklemek ÇAĞIRANIN
    SORUMLULUĞUDUR (bu fonksiyon otomatik eklemez) -- callback'in
    üreteceği kolon adları çağıran tarafça zaten BİLİNİYOR olmalı.
    Callback SADECE kolon EKLEYEBİLİR, hasta kümesini DEĞİŞTİREMEZ --
    değiştirirse `XGBoostLeakageError`.

    YAPISAL SIZINTI GARANTİSİ -- her outer-fold için:
      1. `outer_train_ids & outer_test_ids == set()`.
      2. Cox iç cross-fit, `outer_train`'in HER hastasını TAM BİR KEZ
         skorlar (`cross_fit_scored_ids == outer_train_ids`).
      3. `cox_fit_patient_ids == outer_train_ids` (bu fold'daki HİÇBİR
         Cox fit'i `outer_test`'ten veri görmedi).
      4. Tüm `outer_test` kümelerinin birleşimi `frame.index` ile
         BİREBİR örtüşür, hiçbir hasta birden fazla fold'da tahmin
         edilmez.
    İhlalde `OutOfFoldLeakageError`/`XGBoostLeakageError` -- sessizce
    geçilmez. `verify_aligned_fold_leakage_free()` bu garantiyi
    BAĞIMSIZ bir ikinci yoldan yeniden doğrular (HIGH 2).

    `reduce_collinearity` (B9, 2026-08-28 EKLENDİ -- VARSAYILAN KAPALI,
    davranış değişikliği YOK): `True` verilirse, `_fit_cox_with_full_
    selection()`'a giden aday radyomik havuzu (`cox_feature_columns`)
    `pipeline.reduce_collinearity.build_v3_candidate_pool()` ile
    daraltılır -- HEM dış-final çağrısı için `outer_train_raw`'dan HEM
    HER iç cross-fit çağrısı için o iç-fold'un KENDİ `inner_train_raw`'
    ından AYRI AYRI, YENİDEN hesaplanır (asla tüm havuzdan bir kez değil
    -- `tools/train_cox_week3.py::run_single_v3_variant()`'ın v3 kolunda
    kullandığı FULL-POOL/nested-CV-dışı uygulamadan BİLİNÇLİ olarak
    FARKLI: 2026-08-19'da Codex tam bu sınıfı -- tüm havuzdan fold-dışı
    RCS -- XGBoost hattı için BLOCK etmişti). CV (`near_constant`)
    hesaplaması HAM/standardize-edilmemiş değerler ister (bkz.
    `compute_feature_variability_summary()` docstring'i) -- bu yüzden
    havuz `outer_train_raw`/`inner_train_raw` üzerinden, `standardize_
    columns_fold_safe()` çağrılmadan ÖNCE kurulur (korelasyon zaten
    afin dönüşüme değişmez, ama modülün kendi tasarım sözleşmesiyle
    tutarlılık için ikisi de HAM veriden hesaplanır).

    ⚠️ KAPSAM: bu daraltma SADECE Cox'un KENDİ aday havuzunu etkiler --
    `xgb_feature_columns` (XGBoost'a giden nihai özellik listesi) HER
    ZAMAN `cox_feature_columns`'ın TAMAMINI kullanır (aşağıda
    DEĞİŞTİRİLMEDİ). Gerekçe: gradient-boosted ağaçlar, `lifelines`
    `CoxPHFitter`'ın Newton-Raphson Hessian tersine çevirmesinin aksine,
    neredeyse-tam kolineer sütunlara karşı sayısal olarak KIRILGAN
    DEĞİLDİR -- bu görevin gerçek amacı (production.log'daki `Matrix is
    singular`/`ConvergenceError`) SADECE Cox fit'inde gözlemlendi.

    🔴 2026-09-12 EKLENDİ (v3b reçete-sadakati, Barış: "sapmayla kosmayalim,
    v3b ile XGBoost tam uyumluluk gostermesi lazim"): `reduce_collinearity
    =True` iken artık YUKARIDAKİ HAM-veriden-filtre adımından SONRA, o
    filtrenin SEÇTİĞİ (`outer_cox_candidate_columns`/`inner_cox_candidate_
    columns`) radyomikler de `clinical_standardize_columns` ile birlikte
    fold-yerel Z-score'a sokulur -- `tools/train_cox_week3.py::
    V3VariantConfig.standardize_radiomics=True` (`clinical_standardize_
    columns = feature_columns + clinical_age_columns`, week3 satır ~3806-
    3812) ile AYNI reçete. Bu, penalize EDİLEN bir kovaryat için Cox
    fit'ini/seçimini GERÇEKTEN değiştirir (`standardize_columns_fold_
    safe()`'in penalize-EDİLMEYEN-yaş-için-fark-YOK notunun aksine).
    `reduce_collinearity=False` iken davranış DEĞİŞMEZ (bu adım no-op).
    Sıra GÜVENLİDİR: `CoxFeatureCandidatePoolOverlapError`'in koruduğu
    yapısal garanti (near-constant/CV filtresi ASLA standardize edilmiş
    değer üzerinden hesaplanmaz) bozulmaz, çünkü standardizasyon HER
    ZAMAN filtrenin HAM veriyi tükettiği çağrıdan SONRA gelir (bkz. o
    class'ın docstring'i).
    """

    from sklearn.model_selection import StratifiedKFold

    import xgboost as xgb

    extra_columns = list(extra_columns or [])
    clinical_standardize_columns = list(clinical_standardize_columns or [])
    # Ş4 (Codex sartli onay, 2026-09-11) -- bkz. CoxFeatureCandidatePoolOverlapError
    # docstring'i: standardize edilen klinik kolonlar Cox aday radyomik
    # havuzuyla KESISMEMELI, aksi halde cross-fit (HAM veri) filtresi ile
    # dagitim (standardize veri) filtresi SESSIZCE farkli aday havuzu
    # uretebilir.
    assert_no_clinical_standardize_feature_overlap(
        cox_feature_columns,
        clinical_standardize_columns,
        context="train_xgboost_with_fold_aligned_cox_scores()",
    )
    xgb_grid = xgb_param_grid or DEFAULT_XGB_PARAM_GRID
    l1_ratio_grid = tuple(cox_l1_ratio_grid) if cox_l1_ratio_grid is not None else DEFAULT_L1_RATIO_GRID
    penalizer_grid = tuple(cox_penalizer_grid) if cox_penalizer_grid is not None else DEFAULT_PENALIZER_GRID

    outer_splitter = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    cox_score_rows: list[dict[str, Any]] = []
    fold_audit: list[AlignedXGBoostFoldAudit] = []
    fit_provenance: list[CoxFitProvenanceRecord] = []
    # Ş2 / K16-a+b (Codex sartli onay, 2026-09-11):
    escalation_grid = (
        tuple(cox_penalizer_escalation_grid)
        if cox_penalizer_escalation_grid is not None
        else DEFAULT_COX_PENALIZER_ESCALATION_GRID
    )
    # Ş3 (crash-manifest): caller verdiyse O listeyi KULLAN (coksek bile
    # caller'in elinde kalsin) -- vermediyse yerel/varsayilan.
    if cox_fit_failure_log is None:
        cox_fit_failure_log = []
    skipped_outer_folds: list[dict[str, Any]] = []
    skipped_patient_ids: set[Any] = set()
    # Ş5 (Codex sartli onay, 2026-09-11): `reduce_collinearity=True` iken
    # HER outer/inner filtre cagrisinin TAM raporu (dusurulen ozellikler +
    # korelasyon kumeleri + esikler) -- KAPALIYSA (varsayilan) HER ZAMAN
    # BOS kalir, davranis degismez.
    collinearity_filter_reports: list[dict[str, Any]] = []

    for fold_index, (outer_train_idx, outer_test_idx) in enumerate(
        outer_splitter.split(frame, frame[target_col])
    ):
      try:
        outer_train_raw = frame.iloc[outer_train_idx]
        outer_test_raw = frame.iloc[outer_test_idx]

        outer_train_ids = frozenset(outer_train_raw.index)
        outer_test_ids = frozenset(outer_test_raw.index)
        if outer_train_ids & outer_test_ids:
            raise XGBoostLeakageError(
                f"Fold {fold_index}: outer train/test hasta kumesi kesisimi "
                f"BOS DEGIL -- {sorted(outer_train_ids & outer_test_ids)[:5]}..."
            )

        fold_local_extra_columns: list[str] = []
        if fold_local_extra_column_builder is not None:
            outer_train_raw, outer_test_raw, fold_local_extra_columns = fold_local_extra_column_builder(
                outer_train_raw, outer_test_raw
            )
            if set(outer_train_raw.index) != outer_train_ids or set(outer_test_raw.index) != outer_test_ids:
                raise _AlignedPipelineInternalError(
                    f"Fold {fold_index}: fold_local_extra_column_builder hasta "
                    "kumesini DEGISTIRDI -- sadece kolon EKLEMESI beklenir."
                )

        fold_extra_columns = extra_columns + fold_local_extra_columns

        # B9: dis-final Cox adayi -- reduce_collinearity=True ise SADECE
        # bu fold'un outer_train_raw'indan (HAM, standardize edilmeden
        # ONCE -- bkz. fonksiyon docstring'i) YENIDEN hesaplanir. KAPALI
        # (varsayilan) iken outer_cox_candidate_columns == cox_feature_
        # columns (davranis DEGISMEZ).
        if reduce_collinearity:
            outer_cox_candidate_columns, outer_collinearity_report = build_v3_candidate_pool(
                outer_train_raw,
                cox_feature_columns,
                cv_threshold=collinearity_cv_threshold,
                corr_threshold=collinearity_corr_threshold,
            )
            if not outer_cox_candidate_columns:
                raise RuntimeError(
                    f"Dis fold {fold_index}: reduce_collinearity havuzu "
                    "SIFIR ozellik birakti -- cv_threshold/corr_threshold "
                    "gozden gecirilmeli."
                )
            # Ş5 (Codex sartli onay, 2026-09-11): rapor ARTIK atilmiyor --
            # dusurulen ozellik listeleri + korelasyon kumeleri + esikler
            # `collinearity_filter_reports`'a EKLENIYOR (bkz. sonuc
            # dataclass'inin dokumantasyonu).
            collinearity_filter_reports.append(
                _collinearity_report_to_dict(
                    outer_collinearity_report, outer_fold=fold_index, inner_fold=None, scope="outer_final"
                )
            )
        else:
            outer_cox_candidate_columns = cox_feature_columns

        # v3 recete-sadakati (2026-09-12, Baris: "sapmayla kosmayalim, v3b
        # ile XGBoost tam uyumluluk gostermesi lazim"): `outer_cox_
        # candidate_columns` HAM `outer_train_raw`'dan (yukarida) ZATEN
        # hesaplandi -- bu satirdan SONRA o secilmis radyomikleri de
        # fold-yerel standardize etmek `CoxFeatureCandidatePoolOverlapError`
        # riskini YENIDEN URETMEZ (bkz. o class'in guncellenmis docstring'i):
        # near-constant/korelasyon istatistigi HALA yalniz HAM veriden
        # hesaplandi, standardizasyon SIRADAN SONRA gelir. `reduce_
        # collinearity=False` iken bu liste `clinical_standardize_columns`
        # ile AYNI (davranis DEGISMEZ). `tools/train_cox_week3.py::
        # V3VariantConfig.standardize_radiomics=True` ile AYNI recete --
        # `clinical_standardize_columns = feature_columns + clinical_age_
        # columns` (orada da filtre SONRASI uygulanir).
        outer_fold_standardize_columns = (
            clinical_standardize_columns + list(outer_cox_candidate_columns)
            if reduce_collinearity
            else clinical_standardize_columns
        )
        outer_train_std, outer_test_std = standardize_columns_fold_safe(
            outer_train_raw, outer_test_raw, outer_fold_standardize_columns
        )

        # ---- Cox: dis-fold'un TEK final modeli -- hiperparametre aramasi +
        # ozellik secimi + final fit, SADECE outer_train_std uzerinde
        # (outer_test bu adimlarin HICBIRINDE YOK). Bu model outer_test'i
        # skorlamak icin kullanilir -- outer_test hicbir zaman secime/fit'e
        # GIRMEDIGI icin bu adim zaten sizintisiz (klasik nested-CV). ----
        outer_selection = _fit_cox_with_full_selection(
            outer_train_std,
            outer_cox_candidate_columns,
            duration_col=duration_col,
            event_col=event_col,
            extra_columns=fold_extra_columns,
            penalizer_grid=penalizer_grid,
            l1_ratio_grid=l1_ratio_grid,
            inner_splits=cox_inner_splits,
            hyperparameter_search_seed=seed,
            stability_selection=cox_stability_selection,
            n_bootstrap_stability=cox_n_bootstrap_stability,
            stability_frequency_threshold=cox_stability_frequency_threshold,
            stability_seed=seed + fold_index,
            extra_column_penalizer=cox_extra_column_penalizer,
            error_context=f"Dis fold {fold_index} (outer final model)",
            fit_type="outer_final",
            outer_fold=fold_index,
            inner_fold=None,
            provenance_sink=fit_provenance,
            penalizer_escalation_grid=escalation_grid,
            failure_sink=cox_fit_failure_log,
        )
        best_penalizer = outer_selection.best_penalizer
        best_l1_ratio = outer_selection.best_l1_ratio
        final_features = outer_selection.final_features
        final_model_columns = outer_selection.final_model_columns
        cox_final = outer_selection.fitted_model

        outer_final_fit_ids = frozenset(outer_train_std.index)
        if outer_final_fit_ids & outer_test_ids:
            raise OutOfFoldLeakageError(
                f"Fold {fold_index}: outer-final Cox fit hasta kumesi "
                "outer_test ile KESISIYOR -- YAPISAL SIZINTI."
            )
        # NOT: fit_provenance kaydi ARTIK burada degil -- yukaridaki
        # `_fit_cox_with_full_selection(..., provenance_sink=fit_provenance)`
        # cagrisinin ICINDE, GERCEK `.fit()` cagrisindan HEMEN ONCE
        # eklendi (3. tur duzeltmesi, task-mt06mqzi-k0gb5j).

        test_scores = cox_final.predict_log_partial_hazard(
            outer_test_std[final_model_columns]
        ).astype(float)

        # ---- Cox: outer_train'in KENDI ICINDE TAM (cross-fit) -- HIGH 1
        # duzeltmesi (Codex 2. tur): HER ic-fold KENDI hiperparametre
        # aramasini + ozellik secimini + fit'ini SADECE o ic-fold'un
        # egitim alt-kumesinden (`inner_train`) yapar -- ic-test hastasinin
        # KENDI verisi (outer_train_std'nin bir parcasi olarak) ARTIK bu
        # secime GIRMIYOR (onceden outer-level'de SECILEN final_model_
        # columns/best_penalizer/best_l1_ratio buraya YENIDEN KULLANILIYORDU
        # -- ic-test hastasi kendi skorunu ureten modelin SECIMINI dolayli
        # etkileyebiliyordu). Standardizasyon da HER ic-fold'da SADECE
        # `inner_train`'den (outer_train_raw'in bir alt kumesi -- HENUZ
        # outer_train_std DEGIL) yeniden fit edilir. ----
        inner_splitter = StratifiedKFold(
            n_splits=cox_inner_splits, shuffle=True, random_state=seed + fold_index
        )
        train_score_parts: list[pd.Series] = []
        cross_fit_scored_ids: set[Any] = set()
        inner_fold_of_patient: dict[Any, int] = {}
        for inner_fold_index, (inner_train_idx, inner_test_idx) in enumerate(
            inner_splitter.split(outer_train_raw, outer_train_raw[event_col])
        ):
            inner_train_raw = outer_train_raw.iloc[inner_train_idx]
            inner_test_raw = outer_train_raw.iloc[inner_test_idx]

            # HIGH duzeltmesi (3. tur, task-mt06mqzi-k0gb5j -- "RCS
            # duzenleri yalniz dis-fold'dan; ic cross-fit'te yeniden fit
            # edilmiyor"): `outer_train_raw` bu noktada (eger
            # `fold_local_extra_column_builder` verilmisse -- orn. RCS yas
            # duzenleri) ZATEN outer_train'in TAMAMINDAN fit edilmis ek
            # kolonlar tasiyor -- `inner_train_raw`/`inner_test_raw` bu
            # kolonlari OLDUGU GIBI miras alirdi, yani ic-test hastasinin
            # yasi (outer_train'in bir parcasi olarak) duzen TANIMINI
            # dolayli etkilerdi. AYNI builder'i BURADA, SADECE
            # `inner_train_raw`'dan yeniden cagirarak, bu iki kolonu ic-
            # fold-yerel bir versiyonla UZERINE YAZIYORUZ (ham yas kolonu
            # hala mevcut oldugu icin builder'i ikinci kez cagirmak
            # guvenli) -- disaridaki (outer-final icin kullanilan) deger
            # DEGISMEDEN kalir, sadece bu ic-fold'un KENDI kopyasi guncellenir.
            inner_train_ids_pre_builder = frozenset(inner_train_raw.index)
            inner_test_ids_pre_builder = frozenset(inner_test_raw.index)
            if fold_local_extra_column_builder is not None:
                inner_train_raw, inner_test_raw, inner_local_extra_columns = (
                    fold_local_extra_column_builder(inner_train_raw, inner_test_raw)
                )
                if (
                    set(inner_train_raw.index) != inner_train_ids_pre_builder
                    or set(inner_test_raw.index) != inner_test_ids_pre_builder
                ):
                    raise _AlignedPipelineInternalError(
                        f"Fold {fold_index} ic-fold {inner_fold_index}: "
                        "fold_local_extra_column_builder ic-fold'da hasta "
                        "kumesini DEGISTIRDI -- sadece kolon EKLEMESI beklenir."
                    )
                if sorted(inner_local_extra_columns) != sorted(fold_local_extra_columns):
                    raise _AlignedPipelineInternalError(
                        f"Fold {fold_index} ic-fold {inner_fold_index}: "
                        "fold_local_extra_column_builder ic-fold'da disaridaki "
                        f"({sorted(fold_local_extra_columns)}) ile FARKLI kolon "
                        f"adlari uretti ({sorted(inner_local_extra_columns)})."
                    )

            # B9: ic cross-fit Cox adayi -- reduce_collinearity=True ise
            # SADECE bu ic-fold'un KENDI inner_train_raw'indan (HAM,
            # standardize edilmeden ONCE) YENIDEN hesaplanir -- outer-level
            # icin hesaplanan `outer_cox_candidate_columns` BURADA ASLA
            # yeniden kullanilmaz (o, outer_train'in TAMAMINDAN turetildi,
            # bu ic-test hastasinin verisini -- outer_train'in bir parcasi
            # olarak -- dolayli GOREBILIRDI; tam olarak `_fit_cox_with_
            # full_selection()`'in kendi modul-ustu docstring'inin
            # UYARDIGI "secimin outer_train'in TAMAMINDAN bir kez yapilip
            # ic-fold'a YENIDEN KULLANILMASI" sizinti sinifi). KAPALI
            # (varsayilan) iken inner_cox_candidate_columns == cox_feature_
            # columns (davranis DEGISMEZ).
            if reduce_collinearity:
                inner_cox_candidate_columns, inner_collinearity_report = build_v3_candidate_pool(
                    inner_train_raw,
                    cox_feature_columns,
                    cv_threshold=collinearity_cv_threshold,
                    corr_threshold=collinearity_corr_threshold,
                )
                if not inner_cox_candidate_columns:
                    raise RuntimeError(
                        f"Dis fold {fold_index} ic-fold {inner_fold_index}: "
                        "reduce_collinearity havuzu SIFIR ozellik birakti -- "
                        "cv_threshold/corr_threshold gozden gecirilmeli."
                    )
                # Ş5: bkz. outer-final'daki AYNI desen.
                collinearity_filter_reports.append(
                    _collinearity_report_to_dict(
                        inner_collinearity_report,
                        outer_fold=fold_index,
                        inner_fold=inner_fold_index,
                        scope="cross_fit_final",
                    )
                )
            else:
                inner_cox_candidate_columns = cox_feature_columns

            # v3 recete-sadakati (2026-09-12) -- outer-final'daki AYNI
            # desen: `inner_cox_candidate_columns` HAM `inner_train_raw`'dan
            # (yukarida) ZATEN hesaplandi, bu SIRADAN SONRA fold-yerel
            # standardize etmek guard'in korudugu sirayi BOZMAZ.
            inner_fold_standardize_columns = (
                clinical_standardize_columns + list(inner_cox_candidate_columns)
                if reduce_collinearity
                else clinical_standardize_columns
            )
            inner_train_std, inner_test_std = standardize_columns_fold_safe(
                inner_train_raw, inner_test_raw, inner_fold_standardize_columns
            )
            inner_train_ids = frozenset(inner_train_std.index)

            # Seed'ler outer-level'in seed'inden (hiperparametre aramasi
            # icin `seed`, stabilite icin `seed+fold_index`) BILINCLI olarak
            # AYRISTIRILIR -- hem ic-fold'lar arasi hem outer-level'den
            # farkli deterministik akislar, cakisma riski dusuk (kesin
            # istatistiksel bagimsizlik iddiasi YOK, sadece determinizm).
            inner_seed_base = seed + (fold_index + 1) * 97 + (inner_fold_index + 1) * 131
            inner_selection = _fit_cox_with_full_selection(
                inner_train_std,
                inner_cox_candidate_columns,
                duration_col=duration_col,
                event_col=event_col,
                extra_columns=fold_extra_columns,
                penalizer_grid=penalizer_grid,
                l1_ratio_grid=l1_ratio_grid,
                inner_splits=cox_inner_splits,
                hyperparameter_search_seed=inner_seed_base,
                stability_selection=cox_stability_selection,
                n_bootstrap_stability=cox_n_bootstrap_stability,
                stability_frequency_threshold=cox_stability_frequency_threshold,
                stability_seed=inner_seed_base + 1,
                extra_column_penalizer=cox_extra_column_penalizer,
                error_context=f"Dis fold {fold_index} ic-fold {inner_fold_index}",
                fit_type="cross_fit_final",
                outer_fold=fold_index,
                inner_fold=inner_fold_index,
                provenance_sink=fit_provenance,
                penalizer_escalation_grid=escalation_grid,
                failure_sink=cox_fit_failure_log,
            )

            if inner_train_ids & outer_test_ids:
                raise OutOfFoldLeakageError(
                    f"Fold {fold_index} ic-fold {inner_fold_index}: ic-fold "
                    "Cox fit hasta kumesi outer_test ile KESISIYOR -- "
                    "YAPISAL SIZINTI."
                )

            inner_scores = inner_selection.fitted_model.predict_log_partial_hazard(
                inner_test_std[inner_selection.final_model_columns]
            )
            train_score_parts.append(pd.Series(inner_scores.astype(float), index=inner_test_std.index))
            cross_fit_scored_ids |= set(inner_test_std.index)
            for patient_id in inner_test_std.index:
                inner_fold_of_patient[patient_id] = inner_fold_index
            # NOT: fit_provenance kaydi ARTIK burada degil -- yukaridaki
            # `_fit_cox_with_full_selection(..., fit_type="cross_fit_final",
            # provenance_sink=fit_provenance)` cagrisinin ICINDE, GERCEK
            # `.fit()` cagrisindan HEMEN ONCE eklendi (3. tur duzeltmesi).

        if cross_fit_scored_ids != set(outer_train_ids):
            raise OutOfFoldLeakageError(
                f"Fold {fold_index}: Cox ic cross-fit, outer_train hastalarinin "
                "TAMAMINI TAM BIR KEZ kapsamadi -- eksik="
                f"{sorted(outer_train_ids - cross_fit_scored_ids)[:5]}..."
            )
        train_scores = pd.concat(train_score_parts).sort_index()
        if train_scores.index.duplicated().any():
            raise OutOfFoldLeakageError(
                f"Fold {fold_index}: Cox ic cross-fit'te bir/birden fazla "
                "outer_train hastasi BIRDEN FAZLA ic-fold'da skorlanmis."
            )

        # ---- YAPISAL SIZINTI GARANTISI -- bu fold'daki TUM fit_provenance
        # kayitlarinin (outer-final + HER ic-fold) BIRLESIMI outer_train_ids
        # ile BIREBIR AYNI olmali VE outer_test_ids ile HICBIR KESISIMI
        # olmamali (Codex HIGH 2: "gercek .fit() index'lerinden hesapla,
        # niyet ataması degil"). ----
        this_fold_fit_ids: frozenset = frozenset().union(
            *(
                record.patient_ids
                for record in fit_provenance
                if record.outer_fold == fold_index
            )
        )
        if this_fold_fit_ids & outer_test_ids:
            raise OutOfFoldLeakageError(
                f"Fold {fold_index}: fit_provenance birlesimi outer_test ile "
                "KESISIYOR -- YAPISAL SIZINTI."
            )
        if this_fold_fit_ids != outer_train_ids:
            raise OutOfFoldLeakageError(
                f"Fold {fold_index}: fit_provenance birlesimi outer_train_ids "
                "ile BIREBIR AYNI DEGIL -- yapisal tutarsizlik "
                f"(eksik={sorted(outer_train_ids - this_fold_fit_ids)[:5]}..., "
                f"fazla={sorted(this_fold_fit_ids - outer_train_ids)[:5]}...)."
            )
        cox_fit_patient_ids = this_fold_fit_ids

        for patient_id, score in train_scores.items():
            cox_score_rows.append(
                {
                    "patient_id": patient_id,
                    "fold": fold_index,
                    score_column: float(score),
                    "role": "outer_train_cross_fit",
                    "inner_fold": inner_fold_of_patient[patient_id],
                }
            )
        for patient_id, score in test_scores.items():
            cox_score_rows.append(
                {
                    "patient_id": patient_id,
                    "fold": fold_index,
                    score_column: float(score),
                    "role": "outer_test_honest",
                    "inner_fold": None,
                }
            )

        outer_train_with_score = outer_train_std.copy()
        outer_train_with_score[score_column] = train_scores.reindex(outer_train_with_score.index)
        outer_test_with_score = outer_test_std.copy()
        outer_test_with_score[score_column] = test_scores.reindex(outer_test_with_score.index)

        # B9 KAPSAM NOTU: `xgb_feature_columns` KASITLI OLARAK `cox_feature_
        # columns`'IN TAMAMINI kullanir -- `reduce_collinearity` SADECE Cox
        # fit'inin aday havuzunu daraltir, XGBoost'un KENDI girdisini DEGIL
        # (bkz. fonksiyonun docstring'indeki "KAPSAM" notu).
        xgb_feature_columns = cox_feature_columns + fold_extra_columns + [score_column]

        # ---- XGBoost: hiperparametre aramasi -- SADECE outer_train icinde
        # (target_col'a gore StratifiedKFold) ----
        best_xgb_score = float("-inf")
        best_xgb_params: dict[str, Any] | None = None
        for params in _iter_param_grid(xgb_grid):
            score = _score_xgb_params_via_inner_cv(
                outer_train_with_score,
                xgb_feature_columns,
                target_col=target_col,
                params=params,
                inner_splits=xgb_inner_splits,
                seed=xgb_seed,
            )
            if not np.isnan(score) and score > best_xgb_score:
                best_xgb_score = score
                best_xgb_params = params

        if best_xgb_params is None:
            raise RuntimeError(
                f"Dis fold {fold_index}: XGBoost ic-CV grid taramasindaki "
                "HICBIR parametre kombinasyonu gecerli bir AUC uretmedi."
            )

        model = xgb.XGBClassifier(**best_xgb_params, eval_metric="logloss", random_state=xgb_seed)
        model.fit(outer_train_with_score[xgb_feature_columns], outer_train_with_score[target_col])
        predicted_probability = model.predict_proba(outer_test_with_score[xgb_feature_columns])[:, 1]

        train_ids_check = frozenset(outer_train_with_score.index)
        test_ids_check = frozenset(outer_test_with_score.index)
        if train_ids_check & test_ids_check:
            raise XGBoostLeakageError(
                f"Fold {fold_index}: XGBoost train/test hasta kumesi kesisimi "
                f"BOS DEGIL -- {sorted(train_ids_check & test_ids_check)[:5]}..."
            )

        for patient_id, probability, true_label in zip(
            outer_test_with_score.index, predicted_probability, outer_test_with_score[target_col]
        ):
            oof_rows.append(
                {
                    "patient_id": patient_id,
                    "fold": fold_index,
                    "predicted_probability": float(probability),
                    "true_label": float(true_label),
                }
            )

        fold_rows.append(
            {
                "fold": fold_index,
                "n_outer_train": len(outer_train_with_score),
                "n_outer_test": len(outer_test_with_score),
                "n_positive_outer_train": int(outer_train_with_score[target_col].sum()),
                "n_positive_outer_test": int(outer_test_with_score[target_col].sum()),
                "best_cox_penalizer": best_penalizer,
                "best_cox_l1_ratio": best_l1_ratio,
                "cox_final_features": ";".join(sorted(final_features)),
                "n_cox_final_features": len(final_features),
                "best_xgb_inner_auc": best_xgb_score,
                "best_xgb_params": best_xgb_params,
            }
        )
        fold_audit.append(
            AlignedXGBoostFoldAudit(
                fold=fold_index,
                outer_train_patient_ids=outer_train_ids,
                outer_test_patient_ids=outer_test_ids,
                cox_fit_patient_ids=cox_fit_patient_ids,
                best_cox_penalizer=best_penalizer,
                best_cox_l1_ratio=best_l1_ratio,
                cox_final_features=list(final_features),
                n_cox_inner_splits=cox_inner_splits,
                best_xgb_params=best_xgb_params,
                fold_local_extra_columns=fold_local_extra_columns,
                reduce_collinearity_enabled=reduce_collinearity,
                cox_candidate_features_before_collinearity_filter=len(cox_feature_columns),
                cox_candidate_features_after_collinearity_filter=len(outer_cox_candidate_columns),
            )
        )
      except CoxPenalizerEscalationExhaustedError as exc:
        # Ş2 / K16-b: eskalasyon TUKENDI. `allow_fold_skip=False`
        # (VARSAYILAN) iken davranış BİREBİR ESKİSİ GİBİ -- aynen yukarı
        # fırlatılır, pipeline çöker (sessizce YOK). `allow_fold_skip=True`
        # iken bu TEK outer-fold ATLANIR -- fold'un test hastaları
        # out_of_fold_predictions'a GİRMEZ.
        if not allow_fold_skip:
            raise
        skipped_outer_folds.append(
            {
                "fold": fold_index,
                "reason": str(exc),
                "n_outer_test": int(len(outer_test_idx)),
            }
        )
        skipped_patient_ids |= set(frame.iloc[outer_test_idx].index)
        continue

    # Ş2 / K16-b: `allow_fold_skip=True` iken -- VE SADECE O ZAMAN --
    # skip edilen fold'ların test hastaları asagidaki "frame.index ile
    # BIREBIR ORTUSUR" garantisinin DISINDA tutulur (kasitli, kayitli
    # istisna). `allow_fold_skip=False` (VARSAYILAN) iken `skipped_
    # patient_ids` HER ZAMAN bos kalir -- guard ESKISIYLE BIREBIR AYNI
    # davranir.
    if skipped_outer_folds and len(skipped_patient_ids) == len(frame.index):
        raise CoxPenalizerEscalationExhaustedError(
            "train_xgboost_with_fold_aligned_cox_scores(): TUM dis-fold'lar "
            f"atlandi ({len(skipped_outer_folds)}/{outer_splits}) -- "
            "hicbir hasta out-of-fold skorlanamadi. allow_fold_skip=True "
            "olsa bile bu durum SESSIZCE gecilmez."
        )
    expected_scored_ids = set(frame.index) - skipped_patient_ids
    oof_frame = pd.DataFrame(oof_rows, columns=["patient_id", "fold", "predicted_probability", "true_label"]).set_index(
        "patient_id"
    )
    if oof_frame.index.duplicated().any():
        dupes = sorted(set(oof_frame.index[oof_frame.index.duplicated()]))
        raise XGBoostLeakageError(
            f"Bir/birden fazla hasta BIRDEN FAZLA dis fold'da tahmin edilmis "
            f"-- {dupes[:5]}..."
        )
    if set(oof_frame.index) != expected_scored_ids:
        missing = expected_scored_ids - set(oof_frame.index)
        extra = set(oof_frame.index) - expected_scored_ids
        raise XGBoostLeakageError(
            "out-of-fold tahmin kumesi beklenen kumeyle (frame.index EKSI "
            "atlanan fold hastalari) BIREBIR ORTUSMUYOR -- "
            f"eksik={sorted(missing)[:5]}... fazla={sorted(extra)[:5]}..."
        )

    cox_score_audit = pd.DataFrame(cox_score_rows).set_index("patient_id")

    auc_out_of_fold = _bootstrap_auc_ci(
        oof_frame["true_label"],
        oof_frame["predicted_probability"],
        n_bootstrap=xgb_n_bootstrap_ci,
        seed=xgb_seed,
        confidence_level=confidence_level,
    )

    return AlignedCoxXGBoostResult(
        fold_results=pd.DataFrame(fold_rows),
        out_of_fold_predictions=oof_frame,
        cox_score_audit=cox_score_audit,
        auc_out_of_fold=auc_out_of_fold,
        fold_audit=fold_audit,
        fit_provenance=fit_provenance,
        skipped_outer_folds=skipped_outer_folds,
        cox_fit_failure_log=cox_fit_failure_log,
        collinearity_filter_reports=collinearity_filter_reports,
    )


def verify_aligned_fold_leakage_free(result: AlignedCoxXGBoostResult) -> None:
    """`train_xgboost_with_fold_aligned_cox_scores()`'un iç guard'larından
    BAĞIMSIZ, ikinci bir doğrulama -- sadece dönen `AlignedCoxXGBoostResult`'a
    bakarak (üretim mantığını TEKRARLAMADAN). HIGH 2 (Codex): "her XGBoost
    dış fold'u için o fold'un test hastalarının HİÇBİR Cox fit kümesinde
    olmadığını doğrula".

    MEDIUM düzeltmesi (Codex 2. tur, task-mt04rec5-jjqc95): ÖNCEDEN bu
    fonksiyon SADECE `audit.cox_fit_patient_ids`'e (kendisi de üretim
    kodunun bir NİYET beyanıydı) bakıyordu -- "aynı sentetik audit'i
    okuyan doğrulayıcı" eleştirisi. ARTIK `result.fit_provenance`'taki
    HER kaydı (outer-final + HER iç-fold, gerçek `.fit()` çağrısının
    GÖRDÜĞÜ index) DOĞRUDAN, `audit.cox_fit_patient_ids`'ten BAĞIMSIZ bir
    yoldan denetler -- `audit.cox_fit_patient_ids` alanı sadece bir
    ÇAPRAZ-TUTARLILIK kontrolü olarak kalır (aşağıda ikisi de doğrulanır,
    biri diğerinin YERİNE geçmez)."""

    all_test_ids: set[Any] = set()
    for audit in result.fold_audit:
        if audit.outer_train_patient_ids & audit.outer_test_patient_ids:
            raise OutOfFoldLeakageError(
                f"Fold {audit.fold}: outer train/test kesisimi BOS DEGIL."
            )
        if audit.cox_fit_patient_ids & audit.outer_test_patient_ids:
            raise OutOfFoldLeakageError(
                f"Fold {audit.fold}: Cox fit hasta kumesi outer_test ile "
                "KESISIYOR -- test hastalari BIR Cox fit kumesinde bulunmus."
            )
        if audit.cox_fit_patient_ids != audit.outer_train_patient_ids:
            raise OutOfFoldLeakageError(
                f"Fold {audit.fold}: Cox fit hasta kumesi outer_train ile "
                "BIREBIR AYNI DEGIL -- yapisal tutarsizlik."
            )

        # ---- fit_provenance-tabanli BAGIMSIZ dogrulama (MEDIUM, Codex
        # 2. tur) -- audit.cox_fit_patient_ids'e HIC BAKMADAN, dogrudan
        # bu fold'un GERCEK fit kayitlarindan yeniden hesaplanir. ----
        fold_provenance = [
            record for record in result.fit_provenance if record.outer_fold == audit.fold
        ]
        if not fold_provenance:
            raise OutOfFoldLeakageError(
                f"Fold {audit.fold}: fit_provenance'ta HICBIR kayit yok -- "
                "bu fold'un HANGI hastalarla fit edildigi dogrulanamiyor "
                "(eski/uyumsuz bir AlignedCoxXGBoostResult mi verildi?)."
            )
        if not any(record.fit_type == "outer_final" for record in fold_provenance):
            raise OutOfFoldLeakageError(
                f"Fold {audit.fold}: fit_provenance'ta 'outer_final' turunde "
                "HICBIR kayit yok."
            )
        provenance_fit_ids: frozenset = frozenset().union(
            *(record.patient_ids for record in fold_provenance)
        )
        if provenance_fit_ids & audit.outer_test_patient_ids:
            raise OutOfFoldLeakageError(
                f"Fold {audit.fold}: fit_provenance kayitlarinin BIRLESIMI "
                "outer_test ile KESISIYOR -- GERCEK bir .fit() cagrisi "
                "outer_test hastasi GORMUS."
            )
        if provenance_fit_ids != audit.outer_train_patient_ids:
            raise OutOfFoldLeakageError(
                f"Fold {audit.fold}: fit_provenance kayitlarinin BIRLESIMI "
                "outer_train_patient_ids ile BIREBIR AYNI DEGIL -- yapisal "
                "tutarsizlik."
            )
        if provenance_fit_ids != audit.cox_fit_patient_ids:
            raise OutOfFoldLeakageError(
                f"Fold {audit.fold}: fit_provenance birlesimi ile "
                "audit.cox_fit_patient_ids UYUSMUYOR -- audit alani "
                "fit_provenance'i DOGRU YANSITMIYOR."
            )

        fold_cox_scores = result.cox_score_audit.loc[result.cox_score_audit["fold"] == audit.fold]
        train_role_ids = set(fold_cox_scores.loc[fold_cox_scores["role"] == "outer_train_cross_fit"].index)
        test_role_ids = set(fold_cox_scores.loc[fold_cox_scores["role"] == "outer_test_honest"].index)
        if train_role_ids != set(audit.outer_train_patient_ids):
            raise OutOfFoldLeakageError(
                f"Fold {audit.fold}: cox_score_audit'teki 'outer_train_cross_fit' "
                "kumesi audit.outer_train_patient_ids ile UYUSMUYOR."
            )
        if test_role_ids != set(audit.outer_test_patient_ids):
            raise OutOfFoldLeakageError(
                f"Fold {audit.fold}: cox_score_audit'teki 'outer_test_honest' "
                "kumesi audit.outer_test_patient_ids ile UYUSMUYOR."
            )

        fold_predictions = result.out_of_fold_predictions.loc[
            result.out_of_fold_predictions["fold"] == audit.fold
        ]
        if set(fold_predictions.index) != set(audit.outer_test_patient_ids):
            raise OutOfFoldLeakageError(
                f"Fold {audit.fold}: XGBoost tahmin kumesi outer_test ile "
                "UYUSMUYOR."
            )
        all_test_ids |= set(audit.outer_test_patient_ids)

    if all_test_ids != set(result.out_of_fold_predictions.index):
        raise OutOfFoldLeakageError(
            "Fold audit'lerin test-kumesi birlesimi, out_of_fold_predictions.index "
            "ile UYUSMUYOR."
        )


# =====================================================================
# BÖLÜM 7 -- Final full-pool fit (üretim modeli) + UCSF harici test
# =====================================================================
#
# 2026-08-19 EKLENDİ -- Codex çapraz inceleme MEDIUM 5: "UCSF harici
# değerlendirme yolu YOK". Bu bölüm `tools/train_cox_week3.py::
# fit_final_model_on_full_pool()`'un XGBoost karşılığıdır -- final Cox
# modeli o fonksiyonla (SALT-OKUNUR import, `train_cox_week3.py`'ye TEK
# SATIR yazılmadı) ÜRETİLİR, bu modül SADECE onun ÜZERİNE final bir
# XGBoost katmanı ekler. Yeniden fit/eşik seçimi UCSF'te YAPILMAZ --
# UCSF sadece `predict_proba` ile DEĞERLENDİRİLİR (CLAUDE.md: harici
# test setine eğitim-türevi bir dönüşüm/yeniden-seçim uygulanmaz).


@dataclass
class FinalXGBoostPipelineResult:
    """`fit_final_cox_and_xgboost_pipeline_on_full_pool()`'un çıktısı --
    TÜM UPenn eğitim havuzunda (dış test bölmesi YOK) fit edilmiş TEK
    dağıtılabilir XGBoost modeli (harici -- UCSF -- değerlendirme için)."""

    fitted_model: Any  # xgboost.XGBClassifier
    best_params: dict[str, Any]
    xgb_feature_columns: list[str]
    cross_fit_cox_scores: pd.Series  # index=patient_id, egitim havuzunun TAMAMI icin durust meta-skor
    score_column: str
    n_patients_used: int
    n_cross_fit_splits: int
    # 3. tur EKLENDİ (CRITICAL duzeltmesi, task-mt06mqzi-k0gb5j): hangi
    # hastanin hangi cross-fit split'inde "test/skorlanan" rolunde oldugu
    # -- `verify_full_pool_pipeline_leakage_free()`'nin BAGIMSIZ ikinci
    # dogrulama yolunun girdisi (index=patient_id, deger=split index).
    cross_fit_split_assignment: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    fit_provenance: list[CoxFitProvenanceRecord] = field(default_factory=list)
    # Ş2/Ş3 (2026-09-11 EKLENDİ, varsayılanlı): final Cox fit penalizer-
    # eskalasyon zincirindeki HER BAŞARISIZ deneme (crash-manifest girdisi).
    cox_fit_failure_log: list[CoxFitFailureRecord] = field(default_factory=list)
    # Ş5 (2026-09-11 EKLENDİ, varsayılanlı): bkz. `AlignedCoxXGBoostResult.
    # collinearity_filter_reports` -- AYNI alan, full-pool cross-fit
    # split'leri için.
    collinearity_filter_reports: list[dict[str, Any]] = field(default_factory=list)


def fit_final_cox_and_xgboost_pipeline_on_full_pool(
    training_frame: pd.DataFrame,
    cox_feature_columns: list[str],
    *,
    duration_col: str = "survival_days",
    event_col: str = "event",
    target_col: str = "target_12mo_survival",
    extra_columns: list[str] | None = None,
    cox_extra_column_penalizer: float | None = None,
    cox_cross_fit_splits: int = 5,
    cox_inner_splits: int = 5,
    cox_l1_ratio_grid: tuple[float, ...] | None = None,
    cox_penalizer_grid: tuple[float, ...] | None = None,
    cox_stability_selection: bool = True,
    cox_n_bootstrap_stability: int = 200,
    cox_stability_frequency_threshold: float = 0.6,
    seed: int = 42,
    reduce_collinearity: bool = False,
    collinearity_cv_threshold: float = DEFAULT_CV_THRESHOLD,
    collinearity_corr_threshold: float = DEFAULT_CORRELATION_CLUSTER_THRESHOLD,
    xgb_inner_splits: int = 5,
    xgb_seed: int = 42,
    xgb_param_grid: dict[str, list[Any]] | None = None,
    score_column: str = DEFAULT_COX_OOF_SCORE_COLUMN,
    cox_penalizer_escalation_grid: tuple[float, ...] | None = None,
    cox_fit_failure_log: list[CoxFitFailureRecord] | None = None,
) -> FinalXGBoostPipelineResult:
    """Tüm eğitim havuzunda (dış test bölmesi YOK) TEK bir dağıtılabilir
    XGBoost modeli üretir.

    `cox_penalizer_escalation_grid` (Ş2 / K16-a, 2026-09-11 EKLENDİ):
    `_fit_cox_with_full_selection()`'a AYNEN iletilir -- HER cross-fit
    split'inin final fit'i ConvergenceError/LinAlgError ile başarısız
    olursa dar-kapsamlı penalizer-eskalaması dener (bkz. `train_xgboost_
    with_fold_aligned_cox_scores()`'daki AYNI mekanizma). ⚠️ KAPSAM
    BİLİNÇLİ FARKI: burada `--allow-fold-skip` (K16-b) YOK -- bu fonksiyon
    TEK bir dağıtılabilir modelin XGBoost eğitim meta-skorlarını üretir,
    bir cross-fit split'ini "atlamak" o split'in hastalarını meta-skorsuz
    bırakır (XGBoost eğitim girdisini bozar) -- bu, bir DEĞERLENDİRME
    fold'unu atlamaktan (aligned yol) KATEGORİK OLARAK FARKLIDIR. Eskalasyon
    tükenirse bu fonksiyon HER ZAMAN (bayraktan bağımsız) `CoxPenalizerEscalationExhaustedError`
    ile çöker -- fail-loud, sessizce YOK.

    🔴 CRITICAL DÜZELTMESİ (3. tur, task-mt06mqzi-k0gb5j -- Codex: "full-
    pool Cox meta-skorlarında SEÇİM sızıntısı"): ÖNCEDEN bu fonksiyon,
    TÜM UPenn havuzunda (`cox_final_features`/`cox_best_penalizer`/
    `cox_best_l1_ratio` -- dışarıdan, `training_frame`'in TAMAMI görülerek
    SEÇİLMİŞ) sabit bir özellik/hiperparametre kümesini alıp SADECE fit'i
    her cross-fit fold'una daraltıyordu -- yani SEÇİM (hangi özellikler,
    hangi penalizer/l1_ratio) her hastanın KENDİ meta-skorunu üreten
    modele hâlâ dolaylı sızıyordu (fit fold-yerel olsa bile seçim
    değildi). Kilitli kural ("XGBoost'a giden Cox skorları OOF/nested
    olmalı") bunu ihlal ediyordu.

    ARTIK: dışarıdan seçilmiş bir Cox modeli KABUL EDİLMİYOR -- her
    cross-fit fold'unda `_fit_cox_with_full_selection()` (hiperparametre
    grid taraması + elastic-net + stabilite seçimi + final fit, HEPSİ
    SADECE o fold'un `inner_train`'inde) YENİDEN çağrılır --
    `train_xgboost_with_fold_aligned_cox_scores()`'un iç cross-fit
    adımıyla (fit_type="cross_fit_final" orada, burada
    fit_type="full_pool_final") BİREBİR AYNI mantık, TEK FARK: burada dış
    bir test bölmesi YOK (final/üretim modeli), tüm havuz "eğitim"
    rolünde.

    ⚠️ Tüm havuzda (CV'siz, TEK) fit edilmiş final Cox modeli (örn.
    `tools/train_cox_week3.py::fit_final_model_on_full_pool()`'un çıktısı)
    ARTIK bu fonksiyona GİRMEZ -- CLAUDE.md/görev talimatı: "Tüm havuzda
    fit edilmiş final Cox YALNIZ UCSF/üretim skorlaması için korunur."
    O model `evaluate_xgboost_external_test()`'e AYRI (`cox_fitted_model`
    parametresiyle) verilir, XGBoost'un KENDİ eğitim meta-skorunu ÜRETMEZ.

    XGBoost'un eğitim girdisi için Cox skoru IN-SAMPLE KULLANILMAZ --
    `training_frame`'in TAMAMI `cox_cross_fit_splits` K-fold ile cross-fit
    edilir (her hasta O HASTAYI HİÇ GÖRMEMİŞ -- ne SEÇİMİNDE ne FİT'İNDE
    -- bir Cox modelinden skorlanır).

    `reduce_collinearity` (B9, 2026-08-28 EKLENDİ -- VARSAYILAN KAPALI):
    `train_xgboost_with_fold_aligned_cox_scores()`'daki AYNI mekanizma
    -- HER cross-fit split'inde `_fit_cox_with_full_selection()`'a giden
    aday havuzu, SADECE o split'in KENDİ `inner_train`'inden (HAM,
    `pipeline.reduce_collinearity.build_v3_candidate_pool()` ile)
    YENİDEN daraltılır. Bu fonksiyonda ayrı bir standardizasyon adımı
    YOK (çağıran taraf `training_frame`'i zaten hazırlamış olmalı), bu
    yüzden havuz doğrudan `inner_train`'den kurulur. `xgb_feature_
    columns` burada da HER ZAMAN `cox_feature_columns`'ın TAMAMINI
    kullanır (yukarıdaki kardeş fonksiyonla AYNI kapsam kararı)."""

    from sklearn.model_selection import StratifiedKFold

    import xgboost as xgb

    extra_columns = list(extra_columns or [])
    xgb_grid = xgb_param_grid or DEFAULT_XGB_PARAM_GRID
    l1_ratio_grid = tuple(cox_l1_ratio_grid) if cox_l1_ratio_grid is not None else DEFAULT_L1_RATIO_GRID
    penalizer_grid = tuple(cox_penalizer_grid) if cox_penalizer_grid is not None else DEFAULT_PENALIZER_GRID

    splitter = StratifiedKFold(n_splits=cox_cross_fit_splits, shuffle=True, random_state=seed)
    score_parts: list[pd.Series] = []
    scored_ids: set[Any] = set()
    split_assignment_parts: list[pd.Series] = []
    fit_provenance: list[CoxFitProvenanceRecord] = []
    # Ş2 / K16-a: bkz. `train_xgboost_with_fold_aligned_cox_scores()`'daki
    # AYNI mekanizma -- fold-skip (K16-b) BURADA YOK (docstring'deki
    # KAPSAM notuna bkz.).
    escalation_grid = (
        tuple(cox_penalizer_escalation_grid)
        if cox_penalizer_escalation_grid is not None
        else DEFAULT_COX_PENALIZER_ESCALATION_GRID
    )
    # Ş3 (crash-manifest): caller verdiyse O listeyi KULLAN.
    if cox_fit_failure_log is None:
        cox_fit_failure_log = []
    # Ş5: bkz. train_xgboost_with_fold_aligned_cox_scores()'daki AYNI alan.
    collinearity_filter_reports: list[dict[str, Any]] = []
    for split_index, (train_idx, test_idx) in enumerate(
        splitter.split(training_frame, training_frame[event_col])
    ):
        inner_train = training_frame.iloc[train_idx]
        inner_test = training_frame.iloc[test_idx]

        # B9: bkz. `train_xgboost_with_fold_aligned_cox_scores()`'daki AYNI
        # desen -- reduce_collinearity=True ise SADECE bu split'in KENDI
        # inner_train'inden YENIDEN hesaplanir, KAPALI iken davranis DEGISMEZ.
        if reduce_collinearity:
            split_cox_candidate_columns, split_collinearity_report = build_v3_candidate_pool(
                inner_train,
                cox_feature_columns,
                cv_threshold=collinearity_cv_threshold,
                corr_threshold=collinearity_corr_threshold,
            )
            if not split_cox_candidate_columns:
                raise RuntimeError(
                    f"Full-pool cross-fit split {split_index}: reduce_collinearity "
                    "havuzu SIFIR ozellik birakti -- cv_threshold/corr_threshold "
                    "gozden gecirilmeli."
                )
            # Ş5: bkz. train_xgboost_with_fold_aligned_cox_scores()'daki AYNI desen.
            collinearity_filter_reports.append(
                _collinearity_report_to_dict(
                    split_collinearity_report, outer_fold=None, inner_fold=split_index, scope="full_pool_final"
                )
            )
        else:
            split_cox_candidate_columns = cox_feature_columns

        # v3 recete-sadakati (2026-09-12, Baris: "sapmayla kosmayalim, v3b
        # ile XGBoost tam uyumluluk gostermesi lazim") -- `train_xgboost_
        # with_fold_aligned_cox_scores()`'daki AYNI desen: `split_cox_
        # candidate_columns` HAM `inner_train`'den (yukarida) ZATEN
        # hesaplandi -- bu SIRADAN SONRA o secilmis radyomikleri SPLIT-
        # YEREL standardize etmek `CoxFeatureCandidatePoolOverlapError`'un
        # koruduğu sirayi BOZMAZ (bkz. o class'in docstring'i). `reduce_
        # collinearity=False` iken bu no-op'tur (davranis DEGISMEZ) --
        # `inner_train`/`inner_test` bu fonksiyonun BASKA HICBIR YERINDE
        # (leakage kontrolleri SADECE `.index` okur) HAM haliyle
        # GEREKMEZ, bu yuzden BURADA guvenle UZERINE YAZILABILIR.
        split_standardize_columns = list(split_cox_candidate_columns) if reduce_collinearity else []
        inner_train, inner_test = standardize_columns_fold_safe(
            inner_train, inner_test, split_standardize_columns
        )

        split_seed_base = seed + (split_index + 1) * 131
        inner_selection = _fit_cox_with_full_selection(
            inner_train,
            split_cox_candidate_columns,
            duration_col=duration_col,
            event_col=event_col,
            extra_columns=extra_columns,
            penalizer_grid=penalizer_grid,
            l1_ratio_grid=l1_ratio_grid,
            inner_splits=cox_inner_splits,
            hyperparameter_search_seed=split_seed_base,
            stability_selection=cox_stability_selection,
            n_bootstrap_stability=cox_n_bootstrap_stability,
            stability_frequency_threshold=cox_stability_frequency_threshold,
            stability_seed=split_seed_base + 1,
            extra_column_penalizer=cox_extra_column_penalizer,
            error_context=f"Full-pool cross-fit split {split_index}",
            fit_type="full_pool_final",
            outer_fold=None,
            inner_fold=split_index,
            provenance_sink=fit_provenance,
            penalizer_escalation_grid=escalation_grid,
            failure_sink=cox_fit_failure_log,
        )

        inner_train_ids = frozenset(inner_train.index)
        inner_test_ids = frozenset(inner_test.index)
        if inner_train_ids & inner_test_ids:
            raise OutOfFoldLeakageError(
                f"fit_final_cox_and_xgboost_pipeline_on_full_pool(): split "
                f"{split_index} icin train/test kesisimi BOS DEGIL -- "
                "YAPISAL SIZINTI."
            )

        inner_scores = inner_selection.fitted_model.predict_log_partial_hazard(
            inner_test[inner_selection.final_model_columns]
        )
        score_parts.append(pd.Series(inner_scores.astype(float), index=inner_test.index))
        scored_ids |= set(inner_test.index)
        split_assignment_parts.append(pd.Series(split_index, index=inner_test.index, dtype=int))

    if scored_ids != set(training_frame.index):
        raise OutOfFoldLeakageError(
            "fit_final_cox_and_xgboost_pipeline_on_full_pool(): cross-fit "
            "training_frame'in TAMAMINI TAM BIR KEZ kapsamadi -- eksik="
            f"{sorted(set(training_frame.index) - scored_ids)[:5]}..."
        )
    cross_fit_scores = pd.concat(score_parts).sort_index()
    if cross_fit_scores.index.duplicated().any():
        raise OutOfFoldLeakageError(
            "fit_final_cox_and_xgboost_pipeline_on_full_pool(): bir/birden "
            "fazla hasta BIRDEN FAZLA cross-fit fold'unda skorlanmis."
        )
    cross_fit_split_assignment = pd.concat(split_assignment_parts).sort_index()

    provenance_fit_ids: frozenset = frozenset().union(
        *(record.patient_ids for record in fit_provenance)
    ) if fit_provenance else frozenset()
    if provenance_fit_ids & frozenset(training_frame.index) != provenance_fit_ids:
        raise OutOfFoldLeakageError(
            "fit_final_cox_and_xgboost_pipeline_on_full_pool(): fit_provenance "
            "kayitlarindan biri training_frame disindan bir hasta iceriyor."
        )

    frame_with_score = training_frame.copy()
    frame_with_score[score_column] = cross_fit_scores.reindex(frame_with_score.index)
    xgb_feature_columns = cox_feature_columns + extra_columns + [score_column]

    best_score = float("-inf")
    best_params: dict[str, Any] | None = None
    for params in _iter_param_grid(xgb_grid):
        score = _score_xgb_params_via_inner_cv(
            frame_with_score,
            xgb_feature_columns,
            target_col=target_col,
            params=params,
            inner_splits=xgb_inner_splits,
            seed=xgb_seed,
        )
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_params = params

    if best_params is None:
        raise RuntimeError(
            "fit_final_cox_and_xgboost_pipeline_on_full_pool(): XGBoost "
            "ic-CV grid taramasindaki HICBIR kombinasyon gecerli bir AUC "
            "uretmedi."
        )

    model = xgb.XGBClassifier(**best_params, eval_metric="logloss", random_state=xgb_seed)
    model.fit(frame_with_score[xgb_feature_columns], frame_with_score[target_col])

    return FinalXGBoostPipelineResult(
        fitted_model=model,
        best_params=best_params,
        xgb_feature_columns=xgb_feature_columns,
        cross_fit_cox_scores=cross_fit_scores,
        score_column=score_column,
        n_patients_used=len(frame_with_score),
        n_cross_fit_splits=cox_cross_fit_splits,
        cross_fit_split_assignment=cross_fit_split_assignment,
        fit_provenance=fit_provenance,
        cox_fit_failure_log=cox_fit_failure_log,
        collinearity_filter_reports=collinearity_filter_reports,
    )


def verify_full_pool_pipeline_leakage_free(
    result: FinalXGBoostPipelineResult, training_frame_index: pd.Index
) -> None:
    """`fit_final_cox_and_xgboost_pipeline_on_full_pool()`'un İÇ
    guard'larından BAĞIMSIZ, ikinci bir doğrulama -- SADECE dönen
    `FinalXGBoostPipelineResult`'a bakarak (üretim mantığını
    TEKRARLAMADAN). `verify_aligned_fold_leakage_free()`'in full-pool
    karşılığı (HIGH -- Codex 3. tur: "verify_aligned_fold_leakage_free()
    genişletilsin ... full-pool yolu kapsansın" -- yapısal olarak AYNI
    dataclass olmadığı için AYRI bir fonksiyon olarak eklendi, YOKSA
    `AlignedCoxXGBoostResult`'a özgü `fold_audit` alanına bağımlı kod
    burada anlamsız olurdu).

    Kontroller:
      1. `fit_provenance` boş değil, TAMAMI `fit_type == "full_pool_final"`.
      2. `inner_fold` değerleri `0..n_cross_fit_splits-1` kümesiyle
         BİREBİR aynı, hiçbiri yinelenmiyor/eksik değil.
      3. HER split için: o split'in fit kümesi (`patient_ids`) ile o
         split'te "test/skorlanan" rolündeki hastalar (`cross_fit_split_
         assignment`'tan bağımsız olarak okunur) KESİŞMİYOR.
      4. `cross_fit_split_assignment`'ın kapsadığı hasta kümesi
         `training_frame_index` ile BİREBİR aynı, yineleme yok."""

    if not result.fit_provenance:
        raise OutOfFoldLeakageError(
            "verify_full_pool_pipeline_leakage_free(): fit_provenance BOS -- "
            "full-pool cross-fit'in HANGI hastalarla fit edildigi "
            "dogrulanamiyor (eski/uyumsuz bir FinalXGBoostPipelineResult mi "
            "verildi?)."
        )
    if any(record.fit_type != "full_pool_final" for record in result.fit_provenance):
        unexpected = sorted({record.fit_type for record in result.fit_provenance} - {"full_pool_final"})
        raise OutOfFoldLeakageError(
            "verify_full_pool_pipeline_leakage_free(): fit_provenance'ta "
            f"'full_pool_final' DISINDA fit_type(lar) var: {unexpected}."
        )

    expected_splits = set(range(result.n_cross_fit_splits))
    seen_splits = [record.inner_fold for record in result.fit_provenance]
    if sorted(seen_splits) != sorted(expected_splits):
        raise OutOfFoldLeakageError(
            "verify_full_pool_pipeline_leakage_free(): fit_provenance'taki "
            f"split numaralari ({sorted(seen_splits)}) beklenen "
            f"{sorted(expected_splits)} ile BIREBIR AYNI DEGIL (yinelenen/"
            "eksik split)."
        )

    training_ids = frozenset(training_frame_index)
    assignment_ids = frozenset(result.cross_fit_split_assignment.index)
    if result.cross_fit_split_assignment.index.duplicated().any():
        raise OutOfFoldLeakageError(
            "verify_full_pool_pipeline_leakage_free(): cross_fit_split_"
            "assignment'ta bir/birden fazla hasta BIRDEN FAZLA kez var."
        )
    if assignment_ids != training_ids:
        raise OutOfFoldLeakageError(
            "verify_full_pool_pipeline_leakage_free(): cross_fit_split_"
            "assignment kumesi training_frame_index ile BIREBIR ORTUSMUYOR "
            f"-- eksik={sorted(training_ids - assignment_ids)[:5]}..., "
            f"fazla={sorted(assignment_ids - training_ids)[:5]}..."
        )

    for record in result.fit_provenance:
        split_test_ids = frozenset(
            result.cross_fit_split_assignment.index[
                result.cross_fit_split_assignment == record.inner_fold
            ]
        )
        if record.patient_ids & split_test_ids:
            raise OutOfFoldLeakageError(
                f"verify_full_pool_pipeline_leakage_free(): split "
                f"{record.inner_fold} icin fit kumesi, o split'in test/"
                "skorlanan hastalariyla KESISIYOR -- YAPISAL SIZINTI."
            )
        if record.patient_ids - training_ids:
            raise OutOfFoldLeakageError(
                f"verify_full_pool_pipeline_leakage_free(): split "
                f"{record.inner_fold} fit kumesinde training_frame_index "
                "DISINDA hasta(lar) var."
            )


@dataclass
class ExternalXGBoostEvaluationResult:
    auc: dict[str, Any]
    twelve_month_report: TwelveMonthLabelReport
    n_patients_evaluated: int
    n_excluded_ambiguous_censoring: int


def evaluate_xgboost_external_test(
    cox_fitted_model: Any,
    cox_final_features: list[str],
    xgboost_final_result: FinalXGBoostPipelineResult,
    external_frame: pd.DataFrame,
    *,
    duration_col: str = "survival_days",
    event_col: str = "event",
    extra_columns: list[str] | None = None,
    twelve_month_threshold_days: float = DEFAULT_TWELVE_MONTH_THRESHOLD_DAYS,
    n_bootstrap_ci: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
    xgboost_external_frame: pd.DataFrame | None = None,
) -> ExternalXGBoostEvaluationResult:
    """Yalnız-UPenn'de fit edilen nihai Cox->XGBoost zincirini, YENİDEN
    FİT/EŞİK SEÇİMİ YAPMADAN, `external_frame`'de (üretimde UCSF 295/169
    -- `tools/train_xgboost_week4.py::run_ucsf_external_evaluation()`
    bu fonksiyonu DB'den çekilen gerçek UCSF çerçevesiyle çağırır)
    değerlendirir (MEDIUM 5 düzeltmesi -- Codex).

    ADIMLAR: (1) `cox_fitted_model.predict_log_partial_hazard()` ile
    `external_frame`'in Cox skoru (in-sample/out-of-fold AYRIMI YOK --
    `cox_fitted_model` `external_frame`'i HİÇ görmedi, tek bir dışarıdan
    skorlama, TCGA/UCSF'nin Cox tarafındaki `evaluate_external_test()`
    ile AYNI mantık). (2) 12-ay hedefi (`define_twelve_month_survival_
    target()`) + ambiguous-sansürlü hastaların düşürülmesi (AYNI kural,
    UPenn eğitimindeki ile TUTARLI). (3) `xgboost_final_result.fitted_
    model.predict_proba()` -- YENİDEN eğitim/eşik seçimi YOK. (4)
    bootstrap AUC + %95 CI (tek-eşik dili YASAK kuralı gereği).

    `xgboost_external_frame` (2026-09-12 EKLENDİ, v3 reçete-sadakati):
    `cox_fitted_model` (yani `cox_final`) ile `xgboost_final_result`
    (`fit_final_cox_and_xgboost_pipeline_on_full_pool()`'un çıktısı)
    ARTIK -- `reduce_collinearity=True` iken -- FARKLI ölçeklerde eğitilmiş
    olabilir (`cox_final` radyomikleri standardize edilmiş görür, XGBoost
    kendi eğitiminde HAM radyomik görür -- bkz. `tools/train_xgboost_
    week4.py::run_ucsf_external_evaluation()`'daki `full_pool_*` vs
    `xgb_meta_*` frame ayrımı). Bu yüzden Cox skoru İÇİN `external_frame`
    (cox_final'ın eğitim ölçeğiyle AYNI), XGBoost'un KENDİ ham/extra
    özellik girdisi İÇİN `xgboost_external_frame` (XGBoost eğitiminin
    ölçeğiyle AYNI) AYRI ayrı okunur -- `None` verilirse (varsayılan,
    TÜM eski çağrılar/testler) `external_frame` ile AYNI kabul edilir,
    davranış BİREBİR eskisi gibi kalır (tek-frame, tek-ölçek). İki frame
    verildiğinde `survival_days`/`event`/index'in AYNI hastaları temsil
    ettiği (yalnız bazı öznitelik kolonlarının ÖLÇEĞİNİN farklı olduğu)
    ÇAĞIRANIN sorumluluğudur -- burada indeks eşitliği ile doğrulanır."""

    extra_columns = list(extra_columns or [])
    if xgboost_external_frame is None:
        xgboost_external_frame = external_frame
    elif not xgboost_external_frame.index.equals(external_frame.index):
        raise ValueError(
            "evaluate_xgboost_external_test(): `external_frame` (Cox ölçeği) "
            "ile `xgboost_external_frame` (XGBoost ölçeği) AYNI hasta "
            "kümesini/sırasını temsil etmiyor -- indeksler BİREBİR eşleşmeli "
            "(sadece bazı kolonların ÖLÇEĞİ farklı olmalı, hasta kümesi DEĞİL)."
        )
    cox_model_columns = cox_final_features + extra_columns
    # `xgb_feature_columns` = TÜM radyomik aday havuzu (`cox_feature_
    # columns`, Cox'un SEÇTİĞİ `cox_final_features` ALT KÜMESİ DEĞİL --
    # bkz. `fit_final_cox_and_xgboost_pipeline_on_full_pool()`/
    # `train_xgboost_with_fold_aligned_cox_scores()`'daki AYNI kural)
    # + `extra_columns` + skor kolonu -- Cox'un skoru için gereken
    # `cox_model_columns` bunun HER ZAMAN bir ALT KÜMESİDİR.
    raw_feature_columns = [
        column
        for column in xgboost_final_result.xgb_feature_columns
        if column != xgboost_final_result.score_column
    ]
    missing = set(raw_feature_columns) - set(xgboost_external_frame.columns)
    if missing:
        raise ValueError(
            f"xgboost_external_frame, final XGBoost modelinin bekledigi "
            f"kolon(lar)i icermiyor: {sorted(missing)} -- kurulusu "
            "egitimdekiyle (feature_columns/extra_columns) UYUSMUYOR olabilir."
        )

    cox_score = cox_fitted_model.predict_log_partial_hazard(
        external_frame[cox_model_columns]
    ).astype(float)

    target, twelve_month_report = define_twelve_month_survival_target(
        xgboost_external_frame,
        duration_col=duration_col,
        event_col=event_col,
        threshold_days=twelve_month_threshold_days,
    )
    labeled_frame = apply_twelve_month_target(xgboost_external_frame, target, report=twelve_month_report)

    feature_frame = labeled_frame[raw_feature_columns].copy()
    feature_frame[xgboost_final_result.score_column] = cox_score.reindex(feature_frame.index)
    feature_frame["target_12mo_survival"] = labeled_frame["target_12mo_survival"]

    predicted_probability = xgboost_final_result.fitted_model.predict_proba(
        feature_frame[xgboost_final_result.xgb_feature_columns]
    )[:, 1]

    auc = _bootstrap_auc_ci(
        feature_frame["target_12mo_survival"],
        pd.Series(predicted_probability, index=feature_frame.index),
        n_bootstrap=n_bootstrap_ci,
        seed=seed,
        confidence_level=confidence_level,
    )

    return ExternalXGBoostEvaluationResult(
        auc=auc,
        twelve_month_report=twelve_month_report,
        n_patients_evaluated=len(feature_frame),
        n_excluded_ambiguous_censoring=twelve_month_report.n_excluded_ambiguous_censoring,
    )
