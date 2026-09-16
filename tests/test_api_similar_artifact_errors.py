"""`api/similar.py` -- ARTEFAKT HATA YOLU testleri (503 sozlesmesi).

NEDEN AYRI BIR DOSYA (2026-08-19)
----------------------------------
`tests/test_api_similar.py` modul basinda
`pytest.importorskip("faiss", ...)` cagiriyor -- yani GERCEK faiss
kurulu degilse (veya bu makinedeki gibi Windows Defender Application
Control `_swigfaiss` DLL'ini engelliyorsa) O DOSYANIN TAMAMI atlaniyor.

Ama buradaki testler GERCEK faiss'e IHTIYAC DUYMUYOR: hepsi
`sys.modules["faiss"]`'i monkeypatch ediyor (`api/similar.py` faiss'i
MODUL SEVIYESINDE degil, `_load_index_bundle()` ICINDE import ediyor --
bu dogrulandi, `api/similar.py:174`). Onlari ayri bir dosyaya almak,
503 sozlesmesinin faiss'siz ortamlarda da (CI, baska makineler) test
EDILEBILIR olmasini saglar -- ki bu sozlesmenin en cok onem tasidigi
yer tam olarak orasi.

NE TEST EDILIYOR
-----------------
reviewer capraz dogrulamasinin MEDIUM bulgusu (2026-08-19):
`api/similar.py` modul dokstring'i *"surec yanlis path'ten baslatilirsa
bu endpoint HER istekte 503 doner"* diye VAAT EDIYORDU, ama kod bunu
GARANTI ETMIYORDU. `path.is_file()` kontrolu yalniz dosyanin VAR OLUP
OLMADIGINA bakiyor; asil Turkce-karakter/izin/bozuk-dosya hatasi
`faiss.read_index()` veya `pd.read_csv()` cagrisinda olusuyordu ve o
iki cagri try/except'siz idi. Firlatilan `RuntimeError`/`OSError`
`SimilarIndexArtifactsError` OLMADIGI icin endpoint'in
`except SimilarIndexArtifactsError` blogu tetiklenmiyor, istemciye
bilgilendirici 503 yerine jenerik 500 doniyordu.

Mevcut test (`test_get_similar_patients_503_when_index_artifacts_
missing`) yalniz "dosya YOK" senaryosunu kapsiyordu -- "dosya VAR ama
OKUNAMIYOR" senaryosu test EDILMEMISTI. Bu dosya o boslugu kapatir.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import api.similar as similar_module


class _FakeIndex:
    """`faiss.read_index()`'in dondurdugu nesnenin bu kod yolunda
    kullanilan TEK alani `ntotal`."""

    def __init__(self, ntotal: int = 1) -> None:
        self.ntotal = ntotal


def _write_present_but_invalid_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    """Iki dosya da VAR (`is_file()` gecer) ama icerikleri gecersiz --
    testin amaci tam olarak bu ayrimi zorlamak."""

    index_path = tmp_path / "clinical_radiomics.index"
    order_path = tmp_path / "patient_order.csv"
    index_path.write_bytes(b"bu gecerli bir FAISS indeksi DEGIL")
    order_path.write_text("patient_id" + chr(10) + "P1" + chr(10), encoding="utf-8")
    return index_path, order_path


@pytest.fixture(autouse=True)
def _clear_bundle_cache():
    """`_load_index_bundle()` process-ici onbellek tutuyor; testler
    birbirinin onbellegini gormesin."""

    similar_module._index_bundle_cache.clear()
    yield
    similar_module._index_bundle_cache.clear()


def test_load_index_bundle_raises_artifacts_error_when_index_unreadable(tmp_path, monkeypatch):
    """Dosya VAR ama `faiss.read_index()` acamiyor -> 503 sozlesmesi."""

    _write_present_but_invalid_artifacts(tmp_path)

    class _FakeFaiss:
        @staticmethod
        def read_index(_path):
            raise RuntimeError("Error in ... could not open file for reading")

    monkeypatch.setitem(sys.modules, "faiss", _FakeFaiss)

    with pytest.raises(similar_module.SimilarIndexArtifactsError) as exc_info:
        similar_module._load_index_bundle(tmp_path, "clinical_radiomics.index")

    assert "FAISS indeksi okunamadi" in str(exc_info.value)
    # Orijinal hata YUTULMAMALI -- teshis icin zincirlenmis olmali.
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_load_index_bundle_raises_artifacts_error_when_order_csv_unparsable(tmp_path, monkeypatch):
    """Dosya VAR ama `pd.read_csv()` parse edemiyor -> 503 sozlesmesi."""

    _index_path, order_path = _write_present_but_invalid_artifacts(tmp_path)
    order_path.write_bytes(b"")  # 0 baytlik -> pandas EmptyDataError

    class _FakeFaiss:
        @staticmethod
        def read_index(_path):
            return _FakeIndex(ntotal=1)

    monkeypatch.setitem(sys.modules, "faiss", _FakeFaiss)

    with pytest.raises(similar_module.SimilarIndexArtifactsError) as exc_info:
        similar_module._load_index_bundle(tmp_path, "clinical_radiomics.index")

    assert "patient_order.csv okunamadi" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None


def test_load_index_bundle_still_raises_when_files_missing(tmp_path, monkeypatch):
    """REGRESYON: eski davranis (dosya YOK) BOZULMAMIS olmali."""

    class _FakeFaiss:
        @staticmethod
        def read_index(_path):  # pragma: no cover -- buraya hic gelinmemeli
            raise AssertionError("dosya yokken read_index CAGRILMAMALI")

    monkeypatch.setitem(sys.modules, "faiss", _FakeFaiss)

    with pytest.raises(similar_module.SimilarIndexArtifactsError) as exc_info:
        similar_module._load_index_bundle(tmp_path, "clinical_radiomics.index")

    assert "Beklenen FAISS artefaktı bulunamadı" in str(exc_info.value)


def test_load_index_bundle_still_raises_on_row_count_mismatch(tmp_path, monkeypatch):
    """REGRESYON: satir sayisi uyusmazligi kapisi BOZULMAMIS olmali."""

    _write_present_but_invalid_artifacts(tmp_path)  # order'da 1 hasta var

    class _FakeFaiss:
        @staticmethod
        def read_index(_path):
            return _FakeIndex(ntotal=99)  # 99 != 1

    monkeypatch.setitem(sys.modules, "faiss", _FakeFaiss)

    with pytest.raises(similar_module.SimilarIndexArtifactsError) as exc_info:
        similar_module._load_index_bundle(tmp_path, "clinical_radiomics.index")

    assert "UYUŞMUYOR" in str(exc_info.value)


def test_get_similar_patients_returns_503_not_500_when_artifacts_unreadable(monkeypatch):
    """Uctan uca: okunamayan artefakt istemciye 500 DEGIL 503 dondurmeli.

    Bu, reviewer bulgusunun ASIL sonucudur -- dokumantasyonun vaat
    ettigi sozlesme ile kodun davranisinin eslesmesi.
    """

    def _raise(*_args, **_kwargs):
        raise similar_module.SimilarIndexArtifactsError(
            "FAISS indeksi okunamadi: ... Dosya VAR ama ACILAMIYOR"
        )

    monkeypatch.setattr(similar_module, "_load_index_bundle", _raise)

    with pytest.raises(similar_module.HTTPException) as exc_info:
        similar_module.get_similar_patients(
            "ANY-ID", k=10, idh1_status=None, mgmt_status=None
        )
    assert exc_info.value.status_code == 503
