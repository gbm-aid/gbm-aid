"""``GET /patient/{patient_id}/similar`` -- birim + gerçek DB/indeks
duman (smoke) testleri (api/similar.py).

⚠️ FAISS DİSK G/Ç TUZAĞI (bu makineye özgü, bkz. api/similar.py modül
dokstring'i "BİLİNEN İŞLETİMSEL RİSK"): `faiss.write_index()`/`read_index()`
Windows'ta `C:\\Users\\Barış\\...` (Türkçe karakter) yolunu ÇÖZEMİYOR.
pytest'in KENDİ `tmp_path` fixture'ı da bu makinede AYNI Türkçe yolu
üretiyor (canlı ölçüldü, `tempfile.gettempdir()`'ın 8.3 kısa-ad
çözümlemesinin AKSİNE) -- bu yüzden birim testleri `tmp_path`'e HİÇ
`faiss.write_index()` YAPMAZ. Bunun yerine `_make_bundle()` yardımcı
fonksiyonu TAMAMEN BELLEK-İÇİ bir `faiss.IndexFlatL2` kurar (disk hiç
kullanılmaz) ve `api.similar._load_index_bundle` doğrudan monkeypatch
edilir -- bu, disk G/Ç'sini test kapsamı DIŞINA alır (zaten `tools/
rebuild_faiss_indexes.py --apply`'ın kendi görev raporunda GERÇEK
diske yazma AYRICA doğrulandı).

Gerçek DB + gerçek `artifacts/week3/faiss_index/` duman testleri BU
YÜZDEN yalnız `X:\\` (ASCII-safe subst) üzerinden çalıştırılan bir
süreçte GEÇER -- başka bir yoldan çalıştırılırsa `SimilarIndexArtifacts
Error`/`RuntimeError` ile pytest.skip'e düşerler (sessizce "geçti"
YAZILMAZ, neden AÇIKÇA mesajda görünür).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

faiss = pytest.importorskip("faiss", reason="faiss-cpu kurulu değil (requirements.txt'e bakın)")

import api.similar as similar_module  # noqa: E402


# =====================================================================
# Yardımcılar -- BELLEK-İÇİ FAISS bundle (disk YOK, bkz. modül dokstring'i)
# =====================================================================


def _make_bundle(
    vectors: dict[str, list[float]], sources: dict[str, str] | None = None
) -> similar_module.IndexBundle:
    sources = sources or {}
    patient_ids = list(vectors.keys())
    dim = len(next(iter(vectors.values())))
    matrix = np.array([vectors[pid] for pid in patient_ids], dtype="float32")

    index = faiss.IndexFlatL2(dim)
    index.add(matrix)

    order = pd.DataFrame(
        {
            "row_index": range(len(patient_ids)),
            "source": [sources.get(pid, "UPenn-GBM") for pid in patient_ids],
        },
        index=pd.Index(patient_ids, name="patient_id"),
    )
    return similar_module.IndexBundle(index=index, order=order, mtime_key=0.0)


def _metadata_frame(rows: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["source", *similar_module._CLINICAL_METADATA_RAW_COLUMNS]
        )
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "patient_id"
    return frame


# =====================================================================
# 1) `_query_all_neighbors_sorted` -- self-exclusion + sıralama
# =====================================================================


def test_query_all_neighbors_sorted_excludes_self_and_sorts_ascending():
    bundle = _make_bundle({"A": [0.0, 0.0], "B": [1.0, 0.0], "C": [5.0, 0.0], "D": [2.0, 0.0]})

    result = similar_module._query_all_neighbors_sorted(bundle, "A")

    assert list(result["patient_id"]) == ["B", "D", "C"]
    assert "A" not in list(result["patient_id"])
    assert result["l2_distance"].is_monotonic_increasing


def test_query_all_neighbors_sorted_raises_for_patient_not_in_index():
    bundle = _make_bundle({"A": [0.0], "B": [1.0]})
    with pytest.raises(similar_module.PatientNotInIndexError):
        similar_module._query_all_neighbors_sorted(bundle, "ZZZ-999")


def test_query_all_neighbors_sorted_empty_when_only_self_in_index():
    bundle = _make_bundle({"A": [0.0]})
    result = similar_module._query_all_neighbors_sorted(bundle, "A")
    assert result.empty
    assert list(result.columns) == ["patient_id", "l2_distance"]


# =====================================================================
# 2) `_load_index_bundle` -- eksik/tutarsız artefakt guard'ı
# =====================================================================


def test_load_index_bundle_raises_when_files_missing(tmp_path):
    # NOT: burada disk yazma YOK (write_index çağrılmıyor) -- yalnız
    # OLMAYAN bir dizin okunmaya çalışılıyor, Türkçe-karakter tuzağı
    # devreye girmiyor (dosya zaten hiç yok, faiss'e ulaşılmıyor).
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(similar_module.SimilarIndexArtifactsError):
        similar_module._load_index_bundle(missing_dir, "clinical_radiomics.index")


# =====================================================================
# 3) `_passes_optional_filter` -- "alan varsa filtrele, yoksa filtreleme"
# =====================================================================


@pytest.mark.parametrize(
    "value,filter_value,expected",
    [
        ("Wildtype", None, True),  # filtre yok -> her zaman gecer
        (None, "Wildtype", True),  # deger eksik -> ELENMEZ
        (float("nan"), "Wildtype", True),  # NaN -> ELENMEZ
        ("Wildtype", "wildtype", True),  # case-insensitive tam eslesme
        (" Wildtype ", "wildtype", True),  # bosluk toleransi
        ("Mutated", "Wildtype", False),  # dolu VE uyusmuyor -> ELENIR
    ],
)
def test_passes_optional_filter(value, filter_value, expected):
    assert similar_module._passes_optional_filter(value, filter_value) is expected


# =====================================================================
# 4) `_build_similar_block` -- k>available uyarısı + filtre zarifçe bozulma
# =====================================================================


def test_build_similar_block_warns_when_k_exceeds_available(monkeypatch):
    bundle = _make_bundle({"A": [0.0], "B": [1.0], "C": [2.0]})
    monkeypatch.setattr(
        similar_module,
        "_fetch_clinical_metadata_frame",
        lambda ids: _metadata_frame({pid: {"source": "UPenn-GBM"} for pid in ids}),
    )

    block = similar_module._build_similar_block(
        bundle,
        "A",
        10,
        idh1_status=None,
        mgmt_status=None,
        result_columns=similar_module._RESULT_METADATA_COLUMNS,
    )

    assert block["k_requested"] == 10
    assert block["k_returned"] == 2
    assert any("yalnızca 2 komşu" in w for w in block["warnings"])


def test_build_similar_block_filter_keeps_missing_excludes_mismatched(monkeypatch):
    bundle = _make_bundle({"A": [0.0], "B": [1.0], "C": [2.0], "D": [3.0]})
    meta = {
        "B": {"source": "UPenn-GBM", "idh1_status": "Wildtype"},
        "C": {"source": "UPenn-GBM", "idh1_status": "Mutated"},
        "D": {"source": "UPenn-GBM", "idh1_status": None},
    }
    monkeypatch.setattr(
        similar_module,
        "_fetch_clinical_metadata_frame",
        lambda ids: _metadata_frame({pid: meta[pid] for pid in ids}),
    )

    block = similar_module._build_similar_block(
        bundle,
        "A",
        10,
        idh1_status="Wildtype",
        mgmt_status=None,
        result_columns=similar_module._RESULT_METADATA_COLUMNS,
    )

    returned_ids = [r["patient_id"] for r in block["results"]]
    assert "C" not in returned_ids  # dolu VE uyumsuz -- elendi
    assert "B" in returned_ids  # dolu VE uyumlu -- kaldı
    assert "D" in returned_ids  # eksik deger -- SESSİZCE ELENMEDİ
    assert block["filters_applied"] == {"idh1_status": "Wildtype"}


# =====================================================================
# 5) Endpoint -- hata sözleşmesi (404/422/503) + iki katmanlı akış
# =====================================================================


def _patch_metadata(monkeypatch, extra: dict[str, dict[str, Any]] | None = None) -> None:
    extra = extra or {}

    def _fake(ids: list[str]) -> pd.DataFrame:
        rows = {pid: extra.get(pid, {"source": "UPenn-GBM"}) for pid in ids}
        return _metadata_frame(rows)

    monkeypatch.setattr(similar_module, "_fetch_clinical_metadata_frame", _fake)


def test_get_similar_patients_404_when_patient_not_in_db(monkeypatch):
    bundle = _make_bundle({"Q": [0.0]})
    monkeypatch.setattr(similar_module, "_load_index_bundle", lambda *_a, **_k: bundle)
    monkeypatch.setattr(
        similar_module, "_fetch_patient_existence_and_omics",
        lambda pid: {"exists": False, "has_omics": False},
    )

    with pytest.raises(similar_module.HTTPException) as exc_info:
        similar_module.get_similar_patients("NOPE-999", k=10, idh1_status=None, mgmt_status=None)
    assert exc_info.value.status_code == 404


def test_get_similar_patients_422_when_patient_not_in_index(monkeypatch):
    bundle = _make_bundle({"A": [0.0], "B": [1.0]})
    monkeypatch.setattr(similar_module, "_load_index_bundle", lambda *_a, **_k: bundle)
    monkeypatch.setattr(
        similar_module, "_fetch_patient_existence_and_omics",
        lambda pid: {"exists": True, "has_omics": False},
    )

    with pytest.raises(similar_module.HTTPException) as exc_info:
        similar_module.get_similar_patients("UCSF-PDGM-190", k=10, idh1_status=None, mgmt_status=None)
    assert exc_info.value.status_code == 422
    assert "UCSF" in str(exc_info.value.detail) or "indeksinde YOK" in str(exc_info.value.detail)


def test_get_similar_patients_503_when_index_artifacts_missing(monkeypatch):
    def _raise(*_a, **_k):
        raise similar_module.SimilarIndexArtifactsError("dosya bulunamadı")

    monkeypatch.setattr(similar_module, "_load_index_bundle", _raise)

    with pytest.raises(similar_module.HTTPException) as exc_info:
        similar_module.get_similar_patients("ANY-ID", k=10, idh1_status=None, mgmt_status=None)
    assert exc_info.value.status_code == 503


def test_get_similar_patients_success_no_omics_excludes_self(monkeypatch):
    bundle = _make_bundle({"Q": [0.0], "N1": [1.0], "N2": [2.0]}, {"N1": "UPenn-GBM", "N2": "TCGA-GBM"})
    monkeypatch.setattr(similar_module, "_load_index_bundle", lambda *_a, **_k: bundle)
    monkeypatch.setattr(
        similar_module, "_fetch_patient_existence_and_omics",
        lambda pid: {"exists": True, "has_omics": False},
    )
    _patch_metadata(monkeypatch)

    result = similar_module.get_similar_patients("Q", k=10, idh1_status=None, mgmt_status=None)

    assert result["patient_id"] == "Q"
    assert "molecular_omics_faiss" not in result
    returned = [r["patient_id"] for r in result["clinical_radiomics_faiss"]["results"]]
    assert "Q" not in returned
    assert set(returned) == {"N1", "N2"}
    assert result["clinical_radiomics_faiss"]["index_size"] == 3


def test_get_similar_patients_with_available_omics_layer(monkeypatch):
    clinical_bundle = _make_bundle({"Q": [0.0] * 3, "N1": [1.0] * 3})
    omics_bundle = _make_bundle({"Q": [0.0] * 11, "M1": [0.5] * 11, "M2": [5.0] * 11})

    def _fake_load(index_dir, index_filename):
        if index_filename == similar_module.CLINICAL_RADIOMICS_INDEX_FILENAME:
            return clinical_bundle
        return omics_bundle

    monkeypatch.setattr(similar_module, "_load_index_bundle", _fake_load)
    monkeypatch.setattr(
        similar_module, "_fetch_patient_existence_and_omics",
        lambda pid: {"exists": True, "has_omics": True},
    )
    _patch_metadata(monkeypatch)

    result = similar_module.get_similar_patients("Q", k=10, idh1_status=None, mgmt_status=None)

    assert "molecular_omics_faiss" in result
    assert result["molecular_omics_faiss"]["has_omics"] is True
    assert result["molecular_omics_faiss"]["available"] is True
    omics_ids = [r["patient_id"] for r in result["molecular_omics_faiss"]["results"]]
    assert set(omics_ids) == {"M1", "M2"}


def test_get_similar_patients_omics_index_missing_reports_available_false_without_breaking_main(monkeypatch):
    clinical_bundle = _make_bundle({"Q": [0.0], "N1": [1.0]})

    def _fake_load(index_dir, index_filename):
        if index_filename == similar_module.CLINICAL_RADIOMICS_INDEX_FILENAME:
            return clinical_bundle
        raise similar_module.SimilarIndexArtifactsError("omics indeksi eksik")

    monkeypatch.setattr(similar_module, "_load_index_bundle", _fake_load)
    monkeypatch.setattr(
        similar_module, "_fetch_patient_existence_and_omics",
        lambda pid: {"exists": True, "has_omics": True},
    )
    _patch_metadata(monkeypatch)

    result = similar_module.get_similar_patients("Q", k=10, idh1_status=None, mgmt_status=None)

    assert result["molecular_omics_faiss"]["available"] is False
    assert "eksik" in result["molecular_omics_faiss"]["note"]
    # Ana radyomik sonuc omics katmanindan ETKILENMEDI.
    assert result["clinical_radiomics_faiss"]["k_returned"] == 1


def test_get_similar_patients_omics_patient_missing_from_omics_index_is_visible_anomaly(monkeypatch):
    clinical_bundle = _make_bundle({"Q": [0.0], "N1": [1.0]})
    omics_bundle = _make_bundle({"OTHER-1": [0.0], "OTHER-2": [1.0]})  # Q icinde YOK

    def _fake_load(index_dir, index_filename):
        if index_filename == similar_module.CLINICAL_RADIOMICS_INDEX_FILENAME:
            return clinical_bundle
        return omics_bundle

    monkeypatch.setattr(similar_module, "_load_index_bundle", _fake_load)
    monkeypatch.setattr(
        similar_module, "_fetch_patient_existence_and_omics",
        lambda pid: {"exists": True, "has_omics": True},
    )
    _patch_metadata(monkeypatch)

    result = similar_module.get_similar_patients("Q", k=10, idh1_status=None, mgmt_status=None)

    assert result["molecular_omics_faiss"]["available"] is False
    assert "tutarsız" in result["molecular_omics_faiss"]["note"]


def test_get_similar_patients_has_omics_false_omits_layer_silently(monkeypatch):
    bundle = _make_bundle({"Q": [0.0], "N1": [1.0]})
    monkeypatch.setattr(similar_module, "_load_index_bundle", lambda *_a, **_k: bundle)
    monkeypatch.setattr(
        similar_module, "_fetch_patient_existence_and_omics",
        lambda pid: {"exists": True, "has_omics": False},
    )
    _patch_metadata(monkeypatch)

    result = similar_module.get_similar_patients("Q", k=10, idh1_status=None, mgmt_status=None)
    assert "molecular_omics_faiss" not in result


# =====================================================================
# 6) Gerçek DB + gerçek indeks -- duman testleri (bkz. modül dokstring'i)
# =====================================================================


def test_get_similar_patients_real_db_smoke_tcga_02_0003():
    """Ege'nin `tools/faiss_similar_patients_query_demo.py` ile doğruladığı
    sayı: top-1 = TCGA-02-0011, L2 ≈ 51.3 (bu görevde bağımsız olarak
    51.268925 ölçüldü, X:\\ üzerinden çalıştırılarak)."""

    try:
        result = similar_module.get_similar_patients(
            "TCGA-02-0003", k=10, idh1_status=None, mgmt_status=None
        )
    except Exception as exc:  # pragma: no cover -- yalnız DB/indekse erişilemezse
        pytest.skip(f"Gerçek DB/indekse erişilemedi: {exc}")

    block = result["clinical_radiomics_faiss"]
    assert block["index_size"] == 722
    results = block["results"]
    assert results, "Sonuç boş dönmemeli"
    top1 = results[0]
    assert top1["patient_id"] == "TCGA-02-0011"
    assert top1["l2_distance"] == pytest.approx(51.268925, abs=0.05)


def test_get_similar_patients_real_db_smoke_tcga_02_0006():
    """Ege'nin doğruladığı sayı: top-1 = TCGA-02-0027, L2 ≈ 35.9 (bu
    görevde bağımsız olarak 35.872101 ölçüldü)."""

    try:
        result = similar_module.get_similar_patients(
            "TCGA-02-0006", k=10, idh1_status=None, mgmt_status=None
        )
    except Exception as exc:  # pragma: no cover -- yalnız DB/indekse erişilemezse
        pytest.skip(f"Gerçek DB/indekse erişilemedi: {exc}")

    results = result["clinical_radiomics_faiss"]["results"]
    assert results
    top1 = results[0]
    assert top1["patient_id"] == "TCGA-02-0027"
    assert top1["l2_distance"] == pytest.approx(35.872101, abs=0.05)


def test_get_similar_patients_real_db_smoke_ucsf_patient_returns_422():
    """UCSF K1 kararı gereği v1 indeksine HİÇ dahil değil (`UCSF-PDGM-190`
    canlı DB'de gerçek bir hasta, 2026-08-18 doğrulandı) -- sessiz boş
    sonuç DEĞİL, açık 422.

    🔧 2026-09-14 (kalem 45) -- BU TEST MODÜL DOKSTRİNG'İNİN KENDİ SKIP
    SÖZLEŞMESİNİ İHLAL EDİYORDU (ölçülerek bulundu, varsayılmadı):
    Modül dokstring'i *"gerçek DB/indeks duman testleri yalnız `X:\\`
    üzerinden GEÇER, başka yoldan çalıştırılırsa pytest.skip'e düşerler"*
    diyor. Kardeş iki test (`..._tcga_02_0003`, `..._tcga_02_0006`) bunu
    `except Exception -> skip` ile sağlıyor. Bu test ise `HTTPException`'ı
    ÖNCE yakaladığı için altyapı hatası olan **503**'ü de o dala
    düşürüyordu ve `assert == 422` KIRILIYORDU -- yani sözleşmeye göre
    SKIP olması gereken durum FAIL oluyordu.

    📊 ÖLÇÜM (2026-09-14, aynı kod, aynı DB, yalnız çalışma dizini farklı):
        X:\\ (ASCII subst) -> HTTPException **422** (beklenen sözleşme) ✅
        C:\\Users\\Barış\\... -> HTTPException **503**
            "FAISS indeksi okunamadi: ... RuntimeError: FileIOReader ...
             could not open ... No such file or directory"
    Yani kırmızılık ne bayat testten ne de kod hatasından geliyordu:
    **K12'nin Türkçe-yol yanlış-kırmızısıydı.** `faiss.read_index()` adım
    1'de patlıyor, endpoint hastaya HİÇ bakamadan 503 dönüyor.

    🔴 BEKLENTİ GEVŞETİLMEDİ: 503 artık skip'e gidiyor ama **başka her
    durum hâlâ 422 olmak zorunda**. Örneğin hasta `patients` tablosundan
    silinseydi 404 dönerdi ve test KIRMIZI olurdu (canlı doğrulandı:
    `UCSF-PDGM-190` DB'de var, `patients` içinde 295 UCSF satırı var;
    `patient_order.csv` 722 satır = UPenn 611 + LUMIERE 72 + TCGA 39,
    UCSF satırı 0).
    """

    try:
        similar_module.get_similar_patients(
            "UCSF-PDGM-190", k=10, idh1_status=None, mgmt_status=None
        )
    except similar_module.HTTPException as exc:
        # 503 = altyapı/artefakt erişilemedi (FAISS indeksi okunamadı).
        # Bu, 422 sözleşmesi hakkında HİÇBİR ŞEY söylemez -- kardeş
        # testlerle aynı şekilde skip'e gider, sessizce "geçti" YAZILMAZ.
        if exc.status_code == 503:
            pytest.skip(
                "Gerçek FAISS indeksine erişilemedi (503), 422 sözleşmesi "
                "ÖLÇÜLEMEDİ. Bu makinede en olası sebep K12: "
                "`faiss.read_index()` Türkçe karakterli yolu açamıyor -- "
                "testi ASCII bir çalışma dizininden (`subst X:`) koşun. "
                f"Sunucu detayı: {str(exc.detail)[:300]}"
            )
        assert exc.status_code == 422, (
            f"UCSF-PDGM-190 icin 422 beklenirdi, {exc.status_code} geldi. "
            "404 ise hasta `patients` tablosundan kaybolmus demektir; "
            f"detay: {str(exc.detail)[:300]}"
        )
        return
    except Exception as exc:  # pragma: no cover -- yalnız DB/indekse erişilemezse
        pytest.skip(f"Gerçek DB/indekse erişilemedi: {exc}")

    pytest.fail(
        "UCSF-PDGM-190 icin 422 beklenirdi (v1 indeksinde yok) -- BASARIYLA "
        "dondu. UCSF v1 havuzuna eklenmis olabilir, K1 kararini kontrol edin."
    )


# =====================================================================
# 7) KARAR 34 -- filtre etiketi KANONİKLEŞTİRME + TEK KAYNAK guard'ı
# =====================================================================


def test_normalization_maps_are_the_same_objects_as_predict_module():
    """🔒 ANTI-DRIFT (karar 34'ün ASIL koruması): sözlükler BU DOSYADA
    yeniden tanımlanmaz, `api/predict.py`'den İMPORT edilir. Bu test
    NESNE KİMLİĞİNİ (`is`) doğrular -- birisi ileride similar.py'ye bir
    KOPYA sözlük koyarsa (eşit içerikli olsa bile) test KIRILIR ve iki
    endpoint'in sessizce ayrışması ENGELLENİR."""

    import api.predict as predict_module

    assert (
        similar_module.IDH1_LABEL_NORMALIZATION_MAP
        is predict_module.IDH1_LABEL_NORMALIZATION_MAP
    )
    assert (
        similar_module.MGMT_LABEL_NORMALIZATION_MAP
        is predict_module.MGMT_LABEL_NORMALIZATION_MAP
    )


@pytest.mark.parametrize(
    "value,filter_value,expected,why",
    [
        # 🔴 KARAR 34'ÜN ÇEKİRDEĞİ -- LUMIERE "WT" hastası artık
        # `Wildtype` filtresiyle ELENMEZ (eskiden ELENİYORDU).
        ("WT", "Wildtype", True, "LUMIERE WT <-> UPenn Wildtype"),
        ("wt", "Wildtype", True, "kucuk harf varyanti"),
        # SİMETRİ -- ters yön de tutmalı.
        ("Wildtype", "WT", True, "UPenn Wildtype <-> LUMIERE WT sorgusu"),
        ("R132H mut", "Mutated", True, "LUMIERE R132H mut -> Mutated"),
        # K15 SEMANTİĞİ -- "dizileme gerekli" Wildtype SAYILMAZ.
        (
            "IDH1 neg, Sequencing required",
            "Wildtype",
            False,
            "kesinlesmemis sonuc Wildtype DEGIL (db-agent-F kilitli karari)",
        ),
        (
            "IDH1 neg, Sequencing required",
            "NOS/NEC",
            True,
            "dogru kategori NOS/NEC",
        ),
        # Gerçek uyumsuzluk hâlâ elenmeli.
        ("WT", "Mutated", False, "kanonikten sonra da uyusmuyor"),
        # Eksik değer korumasi normalizasyondan SONRA da gecerli.
        (None, "Wildtype", True, "eksik deger ELENMEZ (zarifce bozulma)"),
        (float("nan"), "Wildtype", True, "NaN ELENMEZ"),
        # Haritada olmayan deger UYDURULMAZ, oldugu gibi karsilastirilir.
        ("Wildtype", "Wildtype", True, "kanonik deger kimlik esleme"),
        ("beklenmeyen-etiket", "Wildtype", False, "haritasiz deger UYDURULMAZ"),
    ],
)
def test_passes_optional_filter_canonicalizes_idh1_labels(
    value, filter_value, expected, why
):
    assert (
        similar_module._passes_optional_filter(
            value,
            filter_value,
            label_lookup=similar_module._IDH1_FILTER_LABEL_LOOKUP,
        )
        is expected
    ), why


@pytest.mark.parametrize(
    "value,filter_value,expected",
    [
        ("methylated", "Methylated", True),
        ("not methylated", "Unmethylated", True),
        ("Methylated", "methylated", True),
        ("not methylated", "Methylated", False),
        ("methylated", "Unmethylated", False),
        (None, "Methylated", True),
    ],
)
def test_passes_optional_filter_canonicalizes_mgmt_labels(
    value, filter_value, expected
):
    assert (
        similar_module._passes_optional_filter(
            value,
            filter_value,
            label_lookup=similar_module._MGMT_FILTER_LABEL_LOOKUP,
        )
        is expected
    )


def test_passes_optional_filter_without_lookup_keeps_legacy_raw_behaviour():
    """`label_lookup` verilmezse ESKİ ham davranış korunur -- karar 34
    yalnız sözlüğü OLAN alanları etkiler, fonksiyonu geneliyle
    değiştirmez."""

    assert similar_module._passes_optional_filter("WT", "Wildtype") is False
    assert similar_module._passes_optional_filter("WT", "wt") is True


def test_passes_optional_filter_non_string_filter_still_raises_loudly():
    """REGRESYON (sessiz başarısızlık YOK): `str()` ile zorla çevirme
    YAPILMAZ. Endpoint fonksiyonu FastAPI olmadan DOĞRUDAN çağrılırsa
    `mgmt_status` varsayılanı bir `Query` NESNESİ olur -- bu nesnenin bir
    etiket gibi sessizce karşılaştırmaya girmesi ENGELLENİR."""

    class _NotAString:
        pass

    with pytest.raises(AttributeError):
        similar_module._passes_optional_filter(
            "Wildtype",
            _NotAString(),  # type: ignore[arg-type]
            label_lookup=similar_module._IDH1_FILTER_LABEL_LOOKUP,
        )


def test_build_case_insensitive_lookup_rejects_conflicting_keys():
    """`api/predict.py`'deki sözlüğe ileride harf-duyarsız ÇAKIŞAN bir
    anahtar eklenirse SESSİZCE biri seçilmez -- gürültülü hata."""

    with pytest.raises(RuntimeError, match="ÇAKIŞMA"):
        similar_module._build_case_insensitive_label_lookup(
            {"WT": "Wildtype", "wt": "Mutated"}, "idh1_status"
        )

    # Aynı kanonik değere düşen harf varyantları MEŞRU -- hata VERMEZ.
    lookup = similar_module._build_case_insensitive_label_lookup(
        {"WT": "Wildtype", "wt": "Wildtype"}, "idh1_status"
    )
    assert lookup == {"wt": "Wildtype"}


def test_build_similar_block_lumiere_wt_survives_wildtype_filter(monkeypatch):
    """UÇTAN UCA (karar 34): LUMIERE `"WT"` etiketli komşu, `Wildtype`
    filtresiyle ARTIK dönüyor; `'IDH1 neg, Sequencing required'` komşusu
    DOĞRU ŞEKİLDE elenmeye devam ediyor."""

    bundle = _make_bundle(
        {"A": [0.0], "LUM_WT": [1.0], "UPENN_WT": [2.0], "LUM_SEQ": [3.0], "MUT": [4.0]}
    )
    meta = {
        "LUM_WT": {"source": "LUMIERE", "idh1_status": "WT"},
        "UPENN_WT": {"source": "UPenn-GBM", "idh1_status": "Wildtype"},
        "LUM_SEQ": {"source": "LUMIERE", "idh1_status": "IDH1 neg, Sequencing required"},
        "MUT": {"source": "UPenn-GBM", "idh1_status": "Mutated"},
    }
    monkeypatch.setattr(
        similar_module,
        "_fetch_clinical_metadata_frame",
        lambda ids: _metadata_frame({pid: meta[pid] for pid in ids}),
    )

    block = similar_module._build_similar_block(
        bundle,
        "A",
        10,
        idh1_status="Wildtype",
        mgmt_status=None,
        result_columns=similar_module._RESULT_METADATA_COLUMNS,
    )
    returned = [r["patient_id"] for r in block["results"]]

    assert "LUM_WT" in returned, "KARAR 34: LUMIERE 'WT' ARTIK elenmemeli"
    assert "UPENN_WT" in returned
    assert "LUM_SEQ" not in returned, "K15: kesinlesmemis sonuc Wildtype DEGIL"
    assert "MUT" not in returned


# =====================================================================
# 8) KARAR 36 -- `row_index` hiza guard'ı + konumsal varsayımın kaldırılması
# =====================================================================


def _order_frame(pairs: list[tuple[str, int]], source: str = "UPenn-GBM") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": [ri for _pid, ri in pairs],
            "source": [source] * len(pairs),
        },
        index=pd.Index([pid for pid, _ri in pairs], name="patient_id"),
    )


def test_build_row_index_to_patient_id_happy_path():
    mapping = similar_module._build_row_index_to_patient_id(
        _order_frame([("A", 0), ("B", 1), ("C", 2)])
    )
    assert mapping == {0: "A", 1: "B", 2: "C"}


def test_build_row_index_to_patient_id_ignores_csv_row_order():
    """KARAR 36'nın ÖZÜ: harita KOLONDAN kurulur -- CSV satır sırası
    `row_index` ile AYNI OLMASA BİLE doğru haritayı verir."""

    mapping = similar_module._build_row_index_to_patient_id(
        _order_frame([("C", 2), ("A", 0), ("B", 1)])
    )
    assert mapping == {0: "A", 1: "B", 2: "C"}


@pytest.mark.parametrize(
    "pairs,expected_message",
    [
        ([("A", 0), ("B", 0), ("C", 2)], "TEKRARLIYOR"),
        ([("A", 0), ("B", 1), ("C", 99)], "BİREBİR örtmüyor"),
        ([("A", 1), ("B", 2), ("C", 3)], "BİREBİR örtmüyor"),
    ],
)
def test_build_row_index_to_patient_id_guards_fire(pairs, expected_message):
    with pytest.raises(
        similar_module.SimilarIndexArtifactsError, match=expected_message
    ):
        similar_module._build_row_index_to_patient_id(_order_frame(pairs))


def test_build_row_index_to_patient_id_guard_missing_column():
    frame = _order_frame([("A", 0), ("B", 1)]).drop(columns=["row_index"])
    with pytest.raises(
        similar_module.SimilarIndexArtifactsError, match="`row_index` kolonu YOK"
    ):
        similar_module._build_row_index_to_patient_id(frame)


def test_build_row_index_to_patient_id_guard_non_integer_column():
    frame = _order_frame([("A", 0), ("B", 1)])
    frame["row_index"] = ["sifir", "bir"]
    with pytest.raises(
        similar_module.SimilarIndexArtifactsError, match="TAMSAYI değil"
    ):
        similar_module._build_row_index_to_patient_id(frame)


def test_query_all_neighbors_sorted_correct_when_csv_order_differs_from_row_index():
    """🔴 KARAR 36 KIRMIZI/YEŞİL REGRESYONU -- 2026-09-13 öncesi kod bu
    durumda SESSİZCE YANLIŞ komşu kimlikleri döndürüyordu (mesafeler
    doğru, isimler kaymış). Vektörler: A=0, B=1, C=5, D=2 -> A'nın doğru
    komşu sırası B, D, C.

    `order` burada BİLİNÇLİ olarak TERS satır sırasıyla kurulur, ama
    `row_index` kolonu her hastayı KENDİ FAISS satırına doğru bağlar."""

    index = faiss.IndexFlatL2(1)
    index.add(np.array([[0.0], [1.0], [5.0], [2.0]], dtype="float32"))
    # CSV satır sırası: D, C, B, A (FAISS sırası: A, B, C, D)
    order = _order_frame([("D", 3), ("C", 2), ("B", 1), ("A", 0)])
    bundle = similar_module.IndexBundle(index=index, order=order, mtime_key=0.0)

    result = similar_module._query_all_neighbors_sorted(bundle, "A")

    assert list(result["patient_id"]) == ["B", "D", "C"]
    assert "A" not in list(result["patient_id"]), "self-exclusion da BOZULMAMALI"
    assert result["l2_distance"].is_monotonic_increasing


def test_query_all_neighbors_sorted_guard_fires_on_broken_row_index():
    """Doğrudan kurulan (birim test) bundle'lar da korunur -- guard
    yalnız `_load_index_bundle()` yolunda DEĞİL."""

    index = faiss.IndexFlatL2(1)
    index.add(np.array([[0.0], [1.0]], dtype="float32"))
    bundle = similar_module.IndexBundle(
        index=index, order=_order_frame([("A", 0), ("B", 0)]), mtime_key=0.0
    )

    with pytest.raises(
        similar_module.SimilarIndexArtifactsError, match="TEKRARLIYOR"
    ):
        similar_module._query_all_neighbors_sorted(bundle, "A")


def test_real_index_bundle_row_index_is_aligned_with_faiss_ids():
    """Gerçek artefakt duman testi (X:\\ gerekir): 722 satır, `row_index`
    0..721 ile BİREBİR, kaynak dağılımı K1 kilitli değerleriyle uyumlu."""

    try:
        bundle = similar_module._load_index_bundle(
            similar_module._clinical_radiomics_index_dir(),
            similar_module.CLINICAL_RADIOMICS_INDEX_FILENAME,
        )
    except Exception as exc:  # pragma: no cover -- yalnız indekse erişilemezse
        pytest.skip(f"Gerçek FAISS indeksine erişilemedi: {exc}")

    assert bundle.index.ntotal == 722
    assert bundle.index.d == 93
    assert bundle.row_index_to_patient_id is not None
    assert set(bundle.row_index_to_patient_id) == set(range(722))
    assert bundle.order["source"].value_counts().to_dict() == {
        "UPenn-GBM": 611,
        "LUMIERE": 72,
        "TCGA-GBM": 39,
    }


def test_real_db_lumiere_wt_patient_survives_wildtype_filter():
    """🔴 KARAR 34 GERÇEK-DB KIRMIZI/YEŞİL KANITI (X:\\ + canlı DB gerekir).

    `Patient-001` canlı DB'de ham `idh1_status='WT'` taşıyor ve v1
    indeksinde VAR. 2026-09-13 öncesinde `?idh1_status=Wildtype` sorgusu
    onu (ve ölçülen 48 kardeşini) SESSİZCE eliyordu."""

    try:
        block = similar_module.get_similar_patients(
            "Patient-002", k=722, idh1_status="Wildtype", mgmt_status=None
        )["clinical_radiomics_faiss"]
    except Exception as exc:  # pragma: no cover -- DB/indeks yoksa
        pytest.skip(f"Gerçek DB/indekse erişilemedi: {exc}")

    returned = {r["patient_id"] for r in block["results"]}
    raw_wt = [r for r in block["results"] if r["idh1_status"] == "WT"]
    seq_required = [
        r
        for r in block["results"]
        if r["idh1_status"] == "IDH1 neg, Sequencing required"
    ]

    assert "Patient-001" in returned, "LUMIERE ham 'WT' hastasi ELENMEMELI"
    assert len(raw_wt) == 49, f"v1 indeksinde 49 ham-WT bekleniyordu, {len(raw_wt)}"
    assert not seq_required, "K15: 'Sequencing required' Wildtype SAYILMAZ"

    # SİMETRİ: 'WT' sorgusu da AYNI kümeyi vermeli.
    symmetric = similar_module.get_similar_patients(
        "Patient-002", k=722, idh1_status="WT", mgmt_status=None
    )["clinical_radiomics_faiss"]
    assert {r["patient_id"] for r in symmetric["results"]} == returned
