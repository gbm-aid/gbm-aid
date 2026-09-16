"""`pipeline/reduce_collinearity.py` icin testler -- v3 hazirligi
(koordinator gorevi, 2026-08-18: "dusuk varyans filtresi").

GERCEK VERI DOGRULAMASI (bu dosyanin disinda, rapor icin ayrica
kosuldu): gercek UPenn WT x93 matrisi uzerinde bu modulun ciktisi,
coordinatorun raporladigi Bulgu 1/2/3 sayilarini BIREBIR/yakin
dogruladi -- bkz. modul docstring'i. Bu test dosyasi SENTETIK veriyle
fonksiyonlarin DAVRANISINI (dogru ozelligi eliyor mu, dogru olani
tutuyor mu) izole test eder, DB'ye baglanmaz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.reduce_collinearity import (
    V3CandidatePoolReport,
    build_v3_candidate_pool,
    compute_feature_variability_summary,
    correlation_clusters_union_find,
    select_cluster_representatives,
    select_near_constant_features,
)


def _make_synthetic_feature_matrix(n: int = 300, seed: int = 5) -> pd.DataFrame:
    """Bulgu 1/2/3'un HER UCUNU de sentetik olarak yeniden uretir:
      - `tiny_scale_high_relative_variability`: kucuk MUTLAK olcek AMA
        SAGLIKLI goreli degiskenlik (Coarseness benzeri) -- CV filtresi
        BUNU DUSURMEMELI.
      - `near_constant_small_scale`: kucuk MUTLAK olcek VE dusuk goreli
        degiskenlik (gercek near-constant, Idmn benzeri) -- CV filtresi
        BUNU DUSURMELI.
      - `normal_a`/`normal_b`: buyuk olcekli, birbirinden BAGIMSIZ (dusuk
        korelasyon) iki ozellik -- ikisi de KALMALI.
      - `dup_a`/`dup_b`: birbirinin neredeyse BIREBIR kopyasi (|r|>0.99)
        -- sadece BIRI kalmali.
    """

    rng = np.random.default_rng(seed)

    tiny_scale_high_cv = rng.normal(loc=0.0005, scale=0.0006, size=n)  # CV yuksek, std kucuk
    near_constant = rng.normal(loc=1.0, scale=0.0001, size=n)  # CV cok dusuk, std kucuk
    normal_a = rng.normal(loc=50.0, scale=10.0, size=n)
    normal_b = rng.normal(loc=-20.0, scale=5.0, size=n)
    dup_a = rng.normal(loc=30.0, scale=8.0, size=n)
    dup_b = dup_a * 1.0001 + rng.normal(loc=0.0, scale=0.01, size=n)  # neredeyse birebir kopya

    return pd.DataFrame(
        {
            "tiny_scale_high_relative_variability": tiny_scale_high_cv,
            "near_constant_small_scale": near_constant,
            "normal_a": normal_a,
            "normal_b": normal_b,
            "dup_a": dup_a,
            "dup_b": dup_b,
        }
    )


def test_compute_feature_variability_summary_ranks_by_cv_not_raw_std() -> None:
    frame = _make_synthetic_feature_matrix()
    feature_columns = list(frame.columns)
    summary = compute_feature_variability_summary(frame, feature_columns)

    # near_constant_small_scale: std kucuk AMA CV COK daha kucuk (Idmn deseni)
    assert summary.loc["near_constant_small_scale", "cv"] < 0.01
    # tiny_scale_high_relative_variability: std de kucuk ama CV YUKSEK (Coarseness deseni)
    assert summary.loc["tiny_scale_high_relative_variability", "cv"] > 0.5
    # HAM std'ye gore ikisi de "en kucuk 2" olmali (Bulgu 1 -- numerik kirilganlik ayni)
    smallest_std = summary.sort_values("std").index[:2].tolist()
    assert set(smallest_std) == {"tiny_scale_high_relative_variability", "near_constant_small_scale"}
    # ama CV sirasi TAMAMEN FARKLI -- biri en dusuk CV, digeri en yuksek CV'lerden biri
    assert summary["cv"].idxmin() == "near_constant_small_scale"


def test_select_near_constant_features_uses_cv_not_raw_std() -> None:
    """Bulgu 2'nin cekirdegi: CV esigi dogru ozelligi (near-constant)
    eler, kucuk-olcekli-ama-saglikli-degisken ozelligi ELEMEZ."""

    frame = _make_synthetic_feature_matrix()
    feature_columns = list(frame.columns)
    summary = compute_feature_variability_summary(frame, feature_columns)

    dropped = select_near_constant_features(summary, cv_threshold=0.02)

    assert "near_constant_small_scale" in dropped
    assert "tiny_scale_high_relative_variability" not in dropped
    assert "normal_a" not in dropped
    assert "normal_b" not in dropped


def test_correlation_clusters_union_find_groups_near_duplicates() -> None:
    frame = _make_synthetic_feature_matrix()
    feature_columns = ["normal_a", "normal_b", "dup_a", "dup_b"]
    clusters = correlation_clusters_union_find(frame, feature_columns, corr_threshold=0.95)

    assert clusters["dup_a"] == clusters["dup_b"]
    assert clusters["normal_a"] != clusters["dup_a"]
    assert clusters["normal_b"] != clusters["dup_a"]
    assert clusters["normal_a"] != clusters["normal_b"]


def test_correlation_clusters_union_find_matches_train_cox_week3_implementation() -> None:
    """`tools/train_cox_week3.py::compute_correlation_clusters()` ile
    davranissal ESDEGERLIK -- bu modul o fonksiyonu import ETMIYOR
    (bkz. `correlation_clusters_union_find()` docstring'i, hot-dosya
    riskinden kaciniliyor), bu yuzden AYRISMA riskini bu test YAKALAR."""

    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    tools_dir = project_root / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import train_cox_week3 as week3

    frame = _make_synthetic_feature_matrix(n=200, seed=99)
    feature_columns = list(frame.columns)

    ours = correlation_clusters_union_find(frame, feature_columns, corr_threshold=0.9)
    theirs = week3.compute_correlation_clusters(frame, feature_columns, corr_threshold=0.9)

    # cluster ID'leri farkli olabilir (remap sirasi), ama PARTISYON (hangi
    # ozelliklerin AYNI kumede oldugu) BIREBIR AYNI olmali.
    def to_partition(clusters: dict[str, int]) -> set[frozenset[str]]:
        groups: dict[int, set[str]] = {}
        for feature, cluster_id in clusters.items():
            groups.setdefault(cluster_id, set()).add(feature)
        return {frozenset(members) for members in groups.values()}

    assert to_partition(ours) == to_partition(theirs)


def test_select_cluster_representatives_keeps_largest_std_member() -> None:
    frame = _make_synthetic_feature_matrix()
    feature_columns = ["normal_a", "normal_b", "dup_a", "dup_b"]
    summary = compute_feature_variability_summary(frame, feature_columns)
    clusters = correlation_clusters_union_find(frame, feature_columns, corr_threshold=0.95)

    representatives, dropped_report = select_cluster_representatives(clusters, summary)

    assert "normal_a" in representatives
    assert "normal_b" in representatives
    # dup_a/dup_b'den sadece biri kalmali -- std'si buyuk olan
    kept_dup = [f for f in representatives if f in {"dup_a", "dup_b"}]
    assert len(kept_dup) == 1
    expected_kept = summary.loc[["dup_a", "dup_b"], "std"].idxmax()
    assert kept_dup[0] == expected_kept

    assert len(dropped_report) == 1
    assert dropped_report.iloc[0]["kept_representative"] == expected_kept
    dropped_name = "dup_a" if expected_kept == "dup_b" else "dup_b"
    assert dropped_report.iloc[0]["dropped_feature"] == dropped_name


def test_select_cluster_representatives_singleton_clusters_all_survive() -> None:
    frame = _make_synthetic_feature_matrix()
    feature_columns = ["normal_a", "normal_b"]
    summary = compute_feature_variability_summary(frame, feature_columns)
    clusters = correlation_clusters_union_find(frame, feature_columns, corr_threshold=0.95)

    representatives, dropped_report = select_cluster_representatives(clusters, summary)
    assert set(representatives) == {"normal_a", "normal_b"}
    assert dropped_report.empty


def test_build_v3_candidate_pool_end_to_end_drops_both_problem_classes() -> None:
    frame = _make_synthetic_feature_matrix()
    feature_columns = list(frame.columns)

    kept, report = build_v3_candidate_pool(
        frame, feature_columns, cv_threshold=0.02, corr_threshold=0.95
    )

    assert isinstance(report, V3CandidatePoolReport)
    assert report.n_input_features == 6

    # near-constant DUSTU, tiny-scale-high-CV KALDI (Bulgu 2 ayrimi)
    assert "near_constant_small_scale" in report.near_constant_dropped
    assert "near_constant_small_scale" not in kept
    assert "tiny_scale_high_relative_variability" in kept

    # dup ciftinden biri dustu (kolinearite dedup)
    dropped_dup = set(report.correlation_dropped["dropped_feature"])
    assert len(dropped_dup & {"dup_a", "dup_b"}) == 1

    # bagimsiz normal ozellikler HER ZAMAN kalir
    assert "normal_a" in kept
    assert "normal_b" in kept

    assert report.n_kept_features == len(kept)
    assert report.n_dropped_total == report.n_input_features - len(kept)
    assert report.n_multi_member_clusters == 1  # sadece dup_a/dup_b kumesi


def test_build_v3_candidate_pool_never_uses_survival_columns() -> None:
    """X-ONLY kapsam garantisi -- `survival_days`/`event` frame'de OLSA
    bile bu fonksiyonlar onlari feature_columns'a dahil etmedigi surece
    hicbir hesaplamaya karismaz (fonksiyonlar sadece `feature_columns`
    ile calisir, frame'in DIGER kolonlarini gormez)."""

    frame = _make_synthetic_feature_matrix()
    frame["survival_days"] = np.linspace(1, 1000, len(frame))
    frame["event"] = (np.arange(len(frame)) % 2).astype(int)
    feature_columns = [
        "tiny_scale_high_relative_variability",
        "near_constant_small_scale",
        "normal_a",
        "normal_b",
    ]

    kept, report = build_v3_candidate_pool(frame, feature_columns)
    assert "survival_days" not in kept
    assert "event" not in kept
    assert "survival_days" not in report.variability_summary.index
    assert "event" not in report.variability_summary.index


def test_build_v3_candidate_pool_is_deterministic() -> None:
    frame = _make_synthetic_feature_matrix()
    feature_columns = list(frame.columns)

    kept1, _ = build_v3_candidate_pool(frame, feature_columns)
    kept2, _ = build_v3_candidate_pool(frame, feature_columns)
    assert kept1 == kept2


# =====================================================================
# BÖLÜM: SIFIR-ORTALAMA / TEKİLLİK KORUMASI
# (2026-08-28, Codex stop-time incelemesinin bulgusu)
#
# BULGU: `cv = std / |mean|` ölçütü mean==0 olduğunda TANIMSIZDIR.
#   (a) std==0 & mean==0 (tümü 0 olan kolon) -> cv = 0/0 = NaN ->
#       `NaN < esik` False -> kolon ELENMİYORDU. Oysa sıfır varyanslı
#       kolon TEKİLLİĞİN EN BARİZ KAYNAĞIDIR ve bu filtrenin XGBoost
#       hattına bağlanma sebebi (B9) tam olarak `Matrix is singular`
#       hatasını önlemekti -- yani filtre asıl işini yapmıyordu.
#   (b) std>0 & mean==0 -> cv = inf -> `.rank().astype(int)` çağrısı
#       `IntCastingNaNError` fırlatıp TÜM ÖZET FONKSİYONUNU çökertiyordu.
#
# GERÇEKLEŞEBİLİRLİĞİ VARSAYIMSAL DEĞİL: WT+TC kolunda bir fold'un
# tamamında TC hacmi 0 olan hastalar bulunabilir (UPENN-GBM-00354 ve
# UPENN-GBM-00397 saf ödem, TC=0 -- 609/583 kararının sebebi budur).
# =====================================================================


def _zero_mean_edge_case_frame(n: int = 60) -> pd.DataFrame:
    """Sıfır-ortalama uç durumlarının hepsini tek çerçevede toplar."""
    rng = np.random.default_rng(42)
    half = n // 2
    return pd.DataFrame(
        {
            # normal: elenmemeli
            "normal_feat": rng.normal(10.0, 2.0, n),
            # std=0 & mean=0 -> cv NaN (ESKİDEN SESSİZCE KALIYORDU)
            "zero_constant": np.zeros(n),
            # std=0 & mean!=0 -> cv=0, zaten eleniyordu (kontrol)
            "nonzero_constant": np.full(n, 7.0),
            # std>0 & mean=0 -> cv=inf (ESKİDEN ÇÖKERTİYORDU); GERÇEKTEN
            # değişken olduğu için ELENMEMELİ
            "zero_mean_but_variable": np.r_[np.full(half, -3.0), np.full(n - half, 3.0)],
        }
    )


def test_variability_summary_zero_mean_kolonla_cokmez() -> None:
    """(b) REGRESYON: mean==0 olan kolon varken özet fonksiyonu ÇÖKMEMELİ.

    Eskiden `IntCastingNaNError: Cannot convert non-finite values` ile
    patlıyordu -- bu, fold-yerel koşuda TÜM eğitimi düşürürdü.
    """
    frame = _zero_mean_edge_case_frame()
    summary = compute_feature_variability_summary(frame, list(frame.columns))

    assert len(summary) == len(frame.columns)
    # Tanımsızlık bastırılmadı, VERİ olarak taşınıyor:
    assert bool(summary.loc["zero_constant", "is_constant"]) is True
    assert bool(summary.loc["zero_constant", "cv_is_undefined"]) is True
    assert bool(summary.loc["zero_mean_but_variable", "cv_is_undefined"]) is True
    assert bool(summary.loc["zero_mean_but_variable", "is_constant"]) is False
    assert bool(summary.loc["normal_feat", "cv_is_undefined"]) is False


def test_sifir_ortalamali_sabit_kolon_daima_elenir() -> None:
    """(a) ASIL BULGU: std==0 olan kolon CV'ye BAKILMAKSIZIN elenmeli."""
    frame = _zero_mean_edge_case_frame()
    summary = compute_feature_variability_summary(frame, list(frame.columns))
    dropped = set(select_near_constant_features(summary))

    # cv = NaN olmasına rağmen elenmeli (eski davranışta KALIYORDU)
    assert "zero_constant" in dropped
    # mean!=0 sabit kolon da elenmeye devam etmeli (regresyon yok)
    assert "nonzero_constant" in dropped
    # GERÇEKTEN değişken olan kolon, mean==0 diye ELENMEMELİ
    assert "zero_mean_but_variable" not in dropped
    assert "normal_feat" not in dropped


def test_havuz_sabit_kolonlari_atinca_tasarim_matrisi_tam_rank_olur() -> None:
    """Filtrenin var oluş amacının doğrudan testi: TEKİLLİK çözülüyor mu?

    Bu test `Matrix is singular` hatasının kök nedenini ölçer: sabit
    kolonlar tasarım matrisinin rank'ını düşürür. Filtre öncesi matris
    rank-eksik OLMALI, sonrası TAM RANK olmalı.
    """
    frame = _zero_mean_edge_case_frame()
    feature_columns = list(frame.columns)

    def _rank_ve_sutun(columns: list[str]) -> tuple[int, int]:
        design = np.c_[np.ones(len(frame)), frame[columns].to_numpy(float)]
        return int(np.linalg.matrix_rank(design)), design.shape[1]

    rank_before, ncol_before = _rank_ve_sutun(feature_columns)
    assert rank_before < ncol_before, (
        "Test kurgusu bozuk: filtre ÖNCESİ matris zaten tam rank -- "
        "bu testin ölçtüğü tekillik senaryosu yeniden üretilemiyor."
    )

    kept, report = build_v3_candidate_pool(frame, feature_columns)
    rank_after, ncol_after = _rank_ve_sutun(list(kept))
    assert rank_after == ncol_after, (
        "Filtre sonrası tasarım matrisi HÂLÂ tekil -- "
        f"rank={rank_after} < sutun={ncol_after}. Kalanlar: {sorted(kept)}"
    )

    assert "zero_constant" in report.near_constant_dropped


# =====================================================================
# BÖLÜM: NaN/±inf GİRDİ DOĞRULAMASI
# (Codex şartlı onay eki, 2026-09-11 -- DÜŞÜK öncelik kalemi)
#
# BULGU: bir özellik sütunu ±inf/NaN içeriyorsa `std`/`mean` de bozulur
# -> `cv = inf/inf = NaN` -> near-constant filtresi bunu YAKALAMAZ (ne
# `cv < esik`, ne `is_constant`/std==0 -- ±inf std==0 DEĞİLDİR). Sonraki
# adımda `.corr()` bu sütun için TÜM eşleşmelerde NaN döner, hiçbir
# zaman `>= corr_threshold` olmaz -> sütun KENDİ tek-üyeli kümesinde
# "temsilci" olarak SESSİZCE hayatta kalır. `build_v3_candidate_pool()`
# artık bunu ÇAĞRI ANINDA `NonFiniteFeatureValueError` ile yakalar.
# =====================================================================


def _finite_guard_test_frame(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "clean_a": rng.normal(loc=10.0, scale=2.0, size=n),
            "clean_b": rng.normal(loc=-5.0, scale=1.0, size=n),
        }
    )


def test_build_v3_candidate_pool_raises_on_infinite_feature_value() -> None:
    from pipeline.reduce_collinearity import NonFiniteFeatureValueError

    frame = _finite_guard_test_frame()
    frame.loc[frame.index[0], "clean_a"] = np.inf

    with pytest.raises(NonFiniteFeatureValueError, match="clean_a"):
        build_v3_candidate_pool(frame, ["clean_a", "clean_b"])


def test_build_v3_candidate_pool_raises_on_negative_infinite_feature_value() -> None:
    from pipeline.reduce_collinearity import NonFiniteFeatureValueError

    frame = _finite_guard_test_frame()
    frame.loc[frame.index[3], "clean_b"] = -np.inf

    with pytest.raises(NonFiniteFeatureValueError, match="clean_b"):
        build_v3_candidate_pool(frame, ["clean_a", "clean_b"])


def test_build_v3_candidate_pool_raises_on_nan_feature_value() -> None:
    from pipeline.reduce_collinearity import NonFiniteFeatureValueError

    frame = _finite_guard_test_frame()
    frame.loc[frame.index[7], "clean_a"] = np.nan

    with pytest.raises(NonFiniteFeatureValueError, match="clean_a"):
        build_v3_candidate_pool(frame, ["clean_a", "clean_b"])


def test_build_v3_candidate_pool_finite_guard_does_not_flag_clean_columns() -> None:
    """Regresyon-yok kontrolü: tamamen sonlu bir matriste guard hiçbir
    şeyi tetiklememeli, mevcut 331 satırlık davranış AYNEN korunmalı."""

    frame = _make_synthetic_feature_matrix()
    feature_columns = list(frame.columns)

    # Guard hicbir istisna firlatmadan calisir -- fonksiyon normal
    # sekilde donmeli.
    kept, report = build_v3_candidate_pool(frame, feature_columns)
    assert isinstance(report, V3CandidatePoolReport)
    assert kept
    # Bulgu 2 davranışı AYNEN korunmalı (bkz. yukarıdaki end-to-end test):
    # gerçek near-constant kolon hâlâ elenir, guard'ın kendisi FALSE
    # POSITIVE üretmez.
    assert "near_constant_small_scale" in report.near_constant_dropped
