"""`tools/rebuild_faiss_indexes.py` icin FAISS self-query + guard testleri.

KAPSAM (RAG agent Hafta 3 kalemi -- "FAISS self-query testi"): bu dosyanın
ÇOĞU testi GERÇEK DB'ye BAĞLANMAZ -- sentetik/fixture DataFrame'lerle
çalışır (aynı desen: `tests/test_train_cox_week3.py`). K1/K2 KİLİTLENDİ
(2026-08-18, Barış) -- gerçek DB'de `tools/rebuild_faiss_indexes.py
--no-combat --apply` ile GERÇEK v1 indeksi kuruldu (722 hasta, bkz. görev
raporu) ve self-query o PERSISTED indekste ayrıca (bu dosyanın dışında,
görev raporunda) doğrulandı. Burada test edilen, kodun KENDİ mantığı:
self-query mesafesi ~0, C32 beyaz liste guard'ı (UCSF gate'te ama v1
havuzunda değil), K2'nin KİLİTLİ LUMIERE Pre-Op seçim kuralı (fixture
CSV'siyle, kirli değerler dahil), determinizm.

⚠️ `tests/test_cox_model.py`/`tests/test_train_cox_week3.py`'ye
DOKUNULMADI (modeling-agent'in dosyaları) -- bu TAMAMEN AYRI, YENİ bir
test dosyası.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

faiss = pytest.importorskip("faiss", reason="faiss-cpu kurulu değil (requirements.txt'e bakın)")

import rebuild_faiss_indexes as rfi  # noqa: E402  (sys.path ayarı sonrası)
from pipeline.cox_model import STABLE_FEATURES_ICC60  # noqa: E402


# =====================================================================
# 0) faiss kurulumu -- kütüphane seviyesi doğrulama
# =====================================================================


def test_faiss_importable_and_versioned():
    assert hasattr(faiss, "__version__")
    assert faiss.__version__  # boş string değil


def test_index_flat_l2_constructible():
    index = faiss.IndexFlatL2(93)
    assert index.is_trained is True  # Flat index eğitim gerektirmez
    assert index.d == 93
    assert index.ntotal == 0


# =====================================================================
# 1) Self-query sanity testi (RAG agent süreç kuralı madde 2 / K1 taslağı
#    §1 ön-koşul madde 2: "indeksteki bir vektörün AYNI vektörle
#    sorgulanması mesafe ≈ 0 vermeli")
# =====================================================================


def test_self_query_distance_is_zero_synthetic():
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(50, 93)).astype("float32")

    index = faiss.IndexFlatL2(93)
    index.add(vectors)

    distances, indices = index.search(vectors, 1)

    assert index.ntotal == 50
    assert np.allclose(distances.ravel(), 0.0, atol=1e-4)
    assert (indices.ravel() == np.arange(50)).all()


def test_self_query_distance_is_zero_via_frozen_scaler_pipeline():
    """`FrozenScaler.transform()` çıktısı da self-query'de mesafe ~0 vermeli
    -- yalnız ham vektörle değil, gerçek üretim yolunun (ölçekleme dahil)
    ürettiği vektörle de sanity testi."""

    rng = np.random.default_rng(7)
    columns = [f"WT__{name}" for name in STABLE_FEATURES_ICC60[:10]]
    matrix = pd.DataFrame(
        rng.normal(loc=100.0, scale=50.0, size=(20, len(columns))),
        columns=columns,
        index=[f"PATIENT-{i:03d}" for i in range(20)],
    )
    source_series = pd.Series("UPenn-GBM", index=matrix.index)

    scaler = rfi.fit_frozen_scaler(matrix, source_series=source_series)
    vectors = scaler.transform(matrix)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    distances, indices = index.search(vectors, 1)

    assert np.allclose(distances.ravel(), 0.0, atol=1e-4)
    assert (indices.ravel() == np.arange(len(matrix))).all()


def test_self_query_nonzero_for_distinct_vectors():
    """Sanity testinin kendisinin anlamlı olduğunu kanıtla -- FARKLI
    vektörler sorgulanırsa mesafe 0 OLMAMALI (yoksa test her zaman
    'geçer', ayırt edici değildir)."""

    rng = np.random.default_rng(1)
    index_vectors = rng.normal(size=(30, 93)).astype("float32")
    query_vectors = rng.normal(size=(5, 93)).astype("float32") + 50.0  # uzak

    index = faiss.IndexFlatL2(93)
    index.add(index_vectors)
    distances, _ = index.search(query_vectors, 1)

    assert (distances.ravel() > 1.0).all()


# =====================================================================
# 2) Edge case -- boş FAISS sonucu (CLAUDE.md: sessizce boş dönme YASAK)
# =====================================================================


def test_empty_index_pool_raises_not_silent():
    empty_matrix = pd.DataFrame(columns=[f"WT__{n}" for n in STABLE_FEATURES_ICC60])
    with pytest.raises(rfi.EmptyIndexPoolError):
        rfi.fit_frozen_scaler(empty_matrix, source_series=pd.Series(dtype=object))


def test_search_on_empty_index_returns_no_valid_neighbors():
    """0 vektörlü bir `IndexFlatL2`'ye sorgu atıldığında FAISS -1/sonsuz
    döndürür -- çağıran kod bunu 'benzer hasta bulunamadı' olarak AÇIKÇA
    yorumlamalı, sessizce 0 komşuyla devam ETMEMELİ. Bu test, bu davranışın
    farkında olunduğunu (sürpriz olmadığını) kayda geçirir."""

    index = faiss.IndexFlatL2(93)
    query = np.zeros((1, 93), dtype="float32")
    distances, indices = index.search(query, 10)
    assert index.ntotal == 0
    assert (indices == -1).all()  # FAISS'in "komşu yok" sözleşmesi


# =====================================================================
# 3) C32 veri kapısı -- kapalı beyaz liste guard'ı
# =====================================================================


@pytest.mark.parametrize(
    "forbidden_tool",
    [
        "UPenn-PyRadiomics-107",  # -C32 soneki YOK
        "LUMIERE-PyRadiomics-107",
        "TCGA-ground-truth",
        "CaPTk-automatic",
        "CaPTk-corrected",
        "DeepBraTumIA",
        "HD-GLIO-AUTO",
        "UPenn-PyRadiomics-107-C32-v2",  # K13 uyarısı -- LIKE ile kaçabilirdi
    ],
)
def test_c32_whitelist_rejects_non_c32_tools(forbidden_tool: str):
    with pytest.raises(rfi.C32WhitelistViolationError):
        rfi.assert_c32_segmentation_tool_allowed(forbidden_tool)


@pytest.mark.parametrize(
    "allowed_tool",
    [
        rfi.SEGMENTATION_TOOL_UPENN_C32,
        rfi.SEGMENTATION_TOOL_LUMIERE_C32,
        rfi.SEGMENTATION_TOOL_TCGA_C32,
        rfi.SEGMENTATION_TOOL_UCSF_C32,
    ],
)
def test_c32_whitelist_accepts_exact_four(allowed_tool: str):
    rfi.assert_c32_segmentation_tool_allowed(allowed_tool)  # exception atmamalı


def test_c32_whitelist_is_closed_set_of_exactly_four():
    """2026-08-18 Barış kararı: UCSF C32 GATE'e eklendi (radyomiği DB'de
    hazır), ama v1 vektör havuzuna GİRMEZ -- bkz. `test_ucsf_in_gate_but_
    not_in_v1_index_pool`."""

    assert rfi.FAISS_ALLOWED_C32_SEGMENTATION_TOOLS == frozenset(
        {
            "UPenn-PyRadiomics-107-C32",
            "LUMIERE-PyRadiomics-107-C32",
            "TCGA-ground-truth-C32",
            "UCSF-PDGM-PyRadiomics-107-C32",
        }
    )


def test_ucsf_in_gate_but_not_in_v1_index_pool():
    """UCSF beyaz listede (gate geçer) AMA v1 vektör havuzunda DEĞİL --
    K1 taslağının sorgu seti/sınıflandırıcısı 3 sınıfla (TCGA/UPenn/
    LUMIERE) donduruldu, UCSF eklemek AYRI bir Barış onayı gerektirir."""

    assert rfi.SEGMENTATION_TOOL_UCSF_C32 in rfi.FAISS_ALLOWED_C32_SEGMENTATION_TOOLS
    assert rfi.SEGMENTATION_TOOL_UCSF_C32 not in rfi.FAISS_V1_INDEX_SEGMENTATION_TOOLS
    assert rfi.FAISS_V1_INDEX_SEGMENTATION_TOOLS == frozenset(
        {
            "UPenn-PyRadiomics-107-C32",
            "LUMIERE-PyRadiomics-107-C32",
            "TCGA-ground-truth-C32",
        }
    )


def test_main_rejects_invalid_argv_combinations():
    with pytest.raises(SystemExit):
        rfi.main([])  # ne --combat ne --no-combat -- argparse zorunlu grup hatası


# =====================================================================
# 5) K2 KİLİTLİ -- LUMIERE kanonik vizit = Rating == 'Pre-Op'
# =====================================================================


def _make_lumiere_preop_long_frame() -> pd.DataFrame:
    """3 hasta -- biri (LUM-060) İKİ Pre-Op+C32 vizitine sahip (Patient-060
    örneğiyle analojik, tie-break gerekir), biri Pre-Op'ta C32'siz (dışlanır,
    CSV fixture'ında hiç Pre-Op satırı olmayacak şekilde kurulur)."""

    feature_dicts = {
        "shape_features": {"original_shape_Sphericity": 0.5},
        "first_order_features": {"original_firstorder_Entropy": 1.0},
        "texture_features": {"original_glcm_Contrast": 2.0},
    }
    rows = [
        # LUM-001: yalnız tek Pre-Op+C32 vizit (week-000)
        {"scan_id": 10, "timepoint_label": "week-000",
         "patient_id": "LUM-001", "source": "LUMIERE", "tumor_region": "WT", **feature_dicts},
        {"scan_id": 11, "timepoint_label": "week-010",  # Post-Op, CSV'de Pre-Op DEĞİL
         "patient_id": "LUM-001", "source": "LUMIERE", "tumor_region": "WT", **feature_dicts},
        # LUM-060: İKİ Pre-Op+C32 vizit -- week-000 kazanmalı (en erken)
        {"scan_id": 60, "timepoint_label": "week-000",
         "patient_id": "LUM-060", "source": "LUMIERE", "tumor_region": "WT", **feature_dicts},
        {"scan_id": 61, "timepoint_label": "week-069",
         "patient_id": "LUM-060", "source": "LUMIERE", "tumor_region": "WT", **feature_dicts},
        # LUM-999: Pre-Op etiketli AMA bu long_frame'de (C32) HİÇ YOK --
        # "Pre-Op'ta C32 yok" durumunu simüle eder, matched'e hiç girmez.
    ]
    return pd.DataFrame(rows)


def _write_fixture_expert_rating_csv(tmp_path) -> Path:
    import csv as csv_module

    path = tmp_path / "fixture-expert-rating.csv"
    rating_col = (
        "Rating (according to RANO, PD: Progressive disease, SD: Stable disease, "
        "PR: Partial response, CR: Complete response, Pre-Op: Pre-Operative, "
        "Post-Op: Post-Operative)"
    )
    fieldnames = ["Patient", "Date", "LessThan3Months", "NonMeasurableLesions", rating_col]
    rows = [
        {"Patient": "LUM-001", "Date": "week-000", "LessThan3Months": "", "NonMeasurableLesions": "", rating_col: "Pre-Op"},
        {"Patient": "LUM-001", "Date": "week-010", "LessThan3Months": "", "NonMeasurableLesions": "", rating_col: "Post-Op "},  # kirli deger (sondan bosluklu)
        {"Patient": "LUM-060", "Date": "week-000", "LessThan3Months": "", "NonMeasurableLesions": "", rating_col: "Pre-Op"},
        {"Patient": "LUM-060", "Date": "week-069", "LessThan3Months": "", "NonMeasurableLesions": "", rating_col: "Pre-Op"},
        {"Patient": "LUM-999", "Date": "week-000", "LessThan3Months": "", "NonMeasurableLesions": "", rating_col: "Pre-Op"},  # C32'de karsiligi YOK
        {"Patient": "LUM-777", "Date": "week-005", "LessThan3Months": "", "NonMeasurableLesions": "", rating_col: "Post-Op/PD"},  # kirli deger, Pre-Op DEGIL
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_lumiere_timepoint_sort_key_orders_by_week_number():
    keys = sorted(
        ["week-069", "week-000", "week-010"], key=rfi._lumiere_timepoint_sort_key
    )
    assert keys == ["week-000", "week-010", "week-069"]


def test_fetch_lumiere_preop_canonical_visits_locked_rule(tmp_path, monkeypatch):
    """K2 KİLİTLİ: Rating=='Pre-Op' (tam eşleşme, `.strip()` ile), kirli
    değerler (`'Post-Op '`, `'Post-Op/PD'`) YANLIŞLIKLA eşleşmemeli, aynı
    hastanın birden fazla Pre-Op+C32 vizitinde EN ERKEN kazanmalı, ve
    beklenen sayı guard'ı (bu fixture'da 2, gerçek DB'de 72) çalışmalı."""

    csv_path = _write_fixture_expert_rating_csv(tmp_path)
    long_frame = _make_lumiere_preop_long_frame()

    monkeypatch.setattr(rfi, "EXPECTED_LUMIERE_PREOP_PATIENTS", 2)
    selected, report = rfi.fetch_lumiere_preop_canonical_visits(long_frame, csv_path=csv_path)

    assert report.n_preop_rated_rows == 4  # LUM-001/week-000, LUM-060x2, LUM-999
    assert report.n_preop_rated_patients == 3  # LUM-001, LUM-060, LUM-999
    assert report.n_output_patients == 2  # LUM-001, LUM-060 (LUM-999 C32'siz)
    assert report.n_preop_patients_without_c32 == 1
    assert report.preop_patients_without_c32 == ["LUM-999"]
    assert report.n_patients_with_tie_broken_duplicate == 1  # LUM-060

    scan_by_patient = selected.set_index("patient_id")["scan_id"].to_dict()
    assert scan_by_patient["LUM-001"] == 10
    assert scan_by_patient["LUM-060"] == 60  # week-000 (en erken), week-069 DEĞİL
    assert "LUM-999" not in scan_by_patient

    # Kirli değerler ('Post-Op ', 'Post-Op/PD') hiçbir hastayı YANLIŞLIKLA
    # Pre-Op olarak işaretlemedi (LUM-001'in week-010'u ve LUM-777 dışlandı).
    assert selected.groupby("patient_id")["scan_id"].nunique().eq(1).all()


def test_fetch_lumiere_preop_canonical_visits_raises_on_count_mismatch(tmp_path):
    csv_path = _write_fixture_expert_rating_csv(tmp_path)
    long_frame = _make_lumiere_preop_long_frame()

    # EXPECTED_LUMIERE_PREOP_PATIENTS gerçek (72) iken fixture 2 üretiyor --
    # "sayı tutmazsa DUR" guard'ı tetiklenmeli.
    with pytest.raises(rfi.LumierePreopCohortMismatchError):
        rfi.fetch_lumiere_preop_canonical_visits(long_frame, csv_path=csv_path)


def test_fetch_lumiere_preop_canonical_visits_missing_csv_raises():
    long_frame = _make_lumiere_preop_long_frame()
    with pytest.raises(rfi.ExpertRatingFileNotFoundError):
        rfi.fetch_lumiere_preop_canonical_visits(
            long_frame, csv_path=Path("does/not/exist.csv")
        )


def test_default_lumiere_expert_rating_csv_exists_and_matches_locked_count():
    """Gerçek `raw/veri/` dosyasına karşı -- K2'nin kilitli sayısını (72)
    canlı DB olmadan, yalnız CSV + önceden bilinen C32 (patient,timepoint)
    çiftleriyle DOĞRULAMAZ (DB'ye bağlanmadan bu mümkün değil) -- yalnız
    dosyanın VAR olduğunu ve beklenen `Rating` kolonunu taşıdığını
    doğrular (asıl 72 sayısı canlı DB testinde/görev raporunda doğrulandı,
    bu test yalnız CSV okuma altyapısının bozulmadığını garanti eder)."""

    assert rfi.DEFAULT_LUMIERE_EXPERT_RATING_CSV.is_file()
    preop_pairs, n_rows, n_patients = rfi._load_lumiere_preop_visit_keys(
        rfi.DEFAULT_LUMIERE_EXPERT_RATING_CSV
    )
    assert n_rows == 92
    assert n_patients == 91


# =====================================================================
# 6) Determinizm -- aynı girdiden aynı çıktı
# =====================================================================


def test_fit_frozen_scaler_is_deterministic():
    rng = np.random.default_rng(123)
    columns = [f"WT__{name}" for name in STABLE_FEATURES_ICC60[:15]]
    matrix = pd.DataFrame(
        rng.normal(size=(25, len(columns))),
        columns=columns,
        index=[f"P-{i:03d}" for i in range(25)],
    )
    source_series = pd.Series("UPenn-GBM", index=matrix.index)

    scaler_1 = rfi.fit_frozen_scaler(matrix, source_series=source_series)
    scaler_2 = rfi.fit_frozen_scaler(matrix, source_series=source_series)

    assert np.array_equal(scaler_1.mean_, scaler_2.mean_)
    assert np.array_equal(scaler_1.scale_, scaler_2.scale_)
    assert scaler_1.feature_order == scaler_2.feature_order

    vectors_1 = scaler_1.transform(matrix)
    vectors_2 = scaler_2.transform(matrix)
    assert np.array_equal(vectors_1, vectors_2)


def test_pivot_wt93_output_is_sorted_by_patient_id():
    """`pivot_radiomics_long_to_wide()` çıktısı `patient_id`'ye göre
    sıralı -- indeks sırası da (dolayısıyla FAISS satır sırası da)
    girdi satır sırasından BAĞIMSIZ, deterministik olmalı."""

    feature_dicts = {
        name: 1.0
        for name in STABLE_FEATURES_ICC60
    }
    shape = {n: v for n, v in feature_dicts.items() if "_shape_" in n}
    first_order = {n: v for n, v in feature_dicts.items() if "_firstorder_" in n}
    texture = {n: v for n, v in feature_dicts.items() if n not in shape and n not in first_order}

    rows = []
    for pid in ["ZZZ-003", "AAA-001", "MMM-002"]:
        rows.append(
            {
                "patient_id": pid,
                "source": "UPenn-GBM",
                "tumor_region": "WT_derived",  # UPenn'in ham bölge adı -- REGION_NAME_ALIASES['UPenn']['WT_derived']='WT'
                "shape_features": shape,
                "first_order_features": first_order,
                "texture_features": texture,
            }
        )
    long_frame = pd.DataFrame(rows)

    wide_frame, _report = rfi.pivot_wt93(long_frame)
    assert list(wide_frame.index) == ["AAA-001", "MMM-002", "ZZZ-003"]
    # Pivot çıktısının kolon SIRASI (shape/first_order/texture dict sırası)
    # WT_FEATURE_COLUMNS'ınkiyle aynı olmak ZORUNDA değil -- yalnız KÜME
    # olarak eşleşmeli (build_design_a/b zaten `combined[list(WT_FEATURE_
    # COLUMNS)]` ile AÇIKÇA yeniden sıralıyor, bkz. rebuild_faiss_indexes.py).
    assert set(wide_frame.columns) == set(rfi.WT_FEATURE_COLUMNS)
    reordered = wide_frame[list(rfi.WT_FEATURE_COLUMNS)]
    assert list(reordered.columns) == list(rfi.WT_FEATURE_COLUMNS)
