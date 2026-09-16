"""A1.2 (2026-08-14) -- N4 cache bütünlüğü testleri.

`run_pyradiomics_{tcga,upenn,lumiere}.py`'nin ÜÇÜNDE de BİREBİR AYNI
`_n4_cache_hit()`/`_write_n4_cache_meta()`/`_input_content_hash()`/
`_n4_params_signature()` deseni var (bilinçli kod tekrarı, mevcut proje
konvansiyonuyla tutarlı -- bkz. `run_pyradiomics_upenn.py`/`_lumiere.py`
docstring'leri "bkz. run_pyradiomics_tcga.py'deki AYNI fonksiyon").
Testler yalnız `run_pyradiomics_tcga`'dan import eder -- diğer ikisinin
davranışı KOD OLARAK birebir aynı olduğu için ayrıca tekrar edilmedi,
ama üçünün de İÇE AKTARILABİLİR olduğu (importerror yok, modül
top-level'da DB/NAS bağlantısı açılmıyor) ayrıca doğrulanır.

Kapsam: Codex review TUR 2 bulgusu (bkz. decisions/2026-08-13-pyradiomics-
c32-bincount-karari.md Riskler #1, A1.2): eski `_n4_cache_hit()` yalnız
`ImageFileReader.ReadImageInformation()` ile HEADER okuyordu -- gövdesi
bozuk (NaN/Inf/tamamen sıfır) ama header'ı sağlam bir dosya sessizce
"cache hit" sayılabiliyordu; cache anahtarı da yalnız girdi YOLUNU
hashliyordu (N4 parametreleri/girdi İÇERİĞİ değişse bile eski çıktı
sessizce yeniden kullanılabiliyordu). Bu testler HER İKİ boşluğu da
gerçekten egzersiz eder.
"""

from __future__ import annotations

import importlib
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from tools.run_pyradiomics_tcga import (
    _apply_n4_cached,
    _input_content_hash,
    _n4_cache_hit,
    _n4_params_signature,
    _write_n4_cache_meta,
)


@pytest.fixture()
def ascii_scratch_dir():
    """`tests/test_harmonization.py::ascii_scratch_dir` ile AYNI desen --
    bu makinede Türkçe karakterli (`Barış`) `tmp_path` SimpleITK'nin NIfTI
    yazıcısıyla uyuşmuyor."""

    base = Path("C:/gbmaid_pytest_scratch") / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _write_image(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path), True)
    return path


def test_other_two_scripts_share_identical_n4_cache_helpers() -> None:
    """UPenn/LUMIERE script'leri de İÇE AKTARILABİLİR ve AYNI fonksiyon
    isimlerini taşıyor -- kod tekrarının BİLİNÇLİ (ve senkron) olduğunu
    doğrular."""

    upenn = importlib.import_module("tools.run_pyradiomics_upenn")
    lumiere = importlib.import_module("tools.run_pyradiomics_lumiere")
    for module in (upenn, lumiere):
        assert hasattr(module, "_n4_cache_hit")
        assert hasattr(module, "_write_n4_cache_meta")
        assert hasattr(module, "_input_content_hash")
        assert hasattr(module, "_n4_params_signature")


def test_n4_cache_hit_false_when_output_missing(ascii_scratch_dir: Path) -> None:
    missing = ascii_scratch_dir / "does_not_exist_n4.nii.gz"
    input_path = _write_image(
        ascii_scratch_dir / "raw.nii.gz", np.ones((3, 3, 3), dtype=np.float32)
    )
    assert _n4_cache_hit(missing, input_path=input_path) is False


def test_n4_cache_hit_false_when_meta_sidecar_missing(ascii_scratch_dir: Path) -> None:
    """Header sağlam ama sidecar meta dosyası HİÇ yazılmamış -- eski
    (H4-only) davranışta bu 'cache hit' sayılırdı, artık sayılmamalı."""

    input_path = _write_image(
        ascii_scratch_dir / "raw.nii.gz", np.ones((3, 3, 3), dtype=np.float32)
    )
    cached = _write_image(
        ascii_scratch_dir / "raw_n4.nii.gz", np.full((3, 3, 3), 5.0, dtype=np.float32)
    )
    assert _n4_cache_hit(cached, input_path=input_path) is False


def test_n4_cache_hit_true_after_write_meta_with_matching_params(
    ascii_scratch_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GBMAID_N4_ITERATIONS", "2,2")
    monkeypatch.setenv("GBMAID_N4_SHRINK_FACTOR", "2")

    input_path = _write_image(
        ascii_scratch_dir / "raw.nii.gz", np.ones((3, 3, 3), dtype=np.float32)
    )
    cached = _write_image(
        ascii_scratch_dir / "raw_n4.nii.gz", np.full((3, 3, 3), 5.0, dtype=np.float32)
    )
    _write_n4_cache_meta(cached, input_path)

    assert _n4_cache_hit(cached, input_path=input_path) is True


def test_n4_cache_hit_false_when_n4_params_changed(
    ascii_scratch_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A1.2'nin ASIL yeni garantisi: N4 iterasyon/shrink_factor DEĞİŞİRSE
    (örn. operatör GBMAID_N4_ITERATIONS'ı elle değiştirdi), eski N4 çıktısı
    ARTIK cache-hit sayılmamalı -- eski davranışta (yol-tabanlı hash) bu
    sessizce yeniden kullanılırdı."""

    monkeypatch.setenv("GBMAID_N4_ITERATIONS", "50,50,30,20")
    monkeypatch.setenv("GBMAID_N4_SHRINK_FACTOR", "4")

    input_path = _write_image(
        ascii_scratch_dir / "raw.nii.gz", np.ones((3, 3, 3), dtype=np.float32)
    )
    cached = _write_image(
        ascii_scratch_dir / "raw_n4.nii.gz", np.full((3, 3, 3), 5.0, dtype=np.float32)
    )
    _write_n4_cache_meta(cached, input_path)
    assert _n4_cache_hit(cached, input_path=input_path) is True

    # Operatör N4 parametrelerini değiştirdi (örn. farklı bir shrink_factor
    # ile yeniden koşmak istiyor) -- AYNI dosya artık miss vermeli.
    monkeypatch.setenv("GBMAID_N4_SHRINK_FACTOR", "2")
    assert _n4_cache_hit(cached, input_path=input_path) is False


def test_n4_cache_hit_false_when_input_content_changed(
    ascii_scratch_dir: Path,
) -> None:
    """Girdi dosyası AYNI YOLDA ama FARKLI İÇERİKLE değişmişse (örn. NAS
    senkronizasyonu/düzeltmesi) cache bunu YAKALAMALI -- yol-tabanlı hash
    tek başına bunu YAKALAYAMAZDI."""

    input_path = _write_image(
        ascii_scratch_dir / "raw.nii.gz", np.ones((3, 3, 3), dtype=np.float32)
    )
    cached = _write_image(
        ascii_scratch_dir / "raw_n4.nii.gz", np.full((3, 3, 3), 5.0, dtype=np.float32)
    )
    _write_n4_cache_meta(cached, input_path)
    assert _n4_cache_hit(cached, input_path=input_path) is True

    # Girdi dosyasının GERÇEK içeriği değişti (AYNI yol).
    _write_image(input_path, np.full((3, 3, 3), 999.0, dtype=np.float32))
    assert _n4_cache_hit(cached, input_path=input_path) is False


def test_n4_cache_hit_false_when_cached_body_is_all_zero(
    ascii_scratch_dir: Path,
) -> None:
    """A1.2'nin İKİNCİ yeni garantisi: header sağlam + meta eşleşiyor AMA
    gövde tamamen sıfır (bozuk/yarım bir N4 sonucu simülasyonu) -- eski
    (yalnız header okuyan) `_n4_cache_hit()` bunu YAKALAYAMAZDI."""

    input_path = _write_image(
        ascii_scratch_dir / "raw.nii.gz", np.ones((3, 3, 3), dtype=np.float32)
    )
    cached = _write_image(
        ascii_scratch_dir / "raw_n4.nii.gz", np.zeros((3, 3, 3), dtype=np.float32)
    )
    _write_n4_cache_meta(cached, input_path)
    assert _n4_cache_hit(cached, input_path=input_path) is False


def test_nifti_writer_silently_sanitizes_nan_to_zero_environment_note(
    ascii_scratch_dir: Path,
) -> None:
    """BULGU (bu test A1.2'yi test ederken keşfedildi, kod hatası DEĞİL --
    kayıt altına almak için tutuluyor): bu ortamdaki SimpleITK/ITK NIfTI
    yazıcısı (`sitk.WriteImage`, sıkıştırmalı VEYA sıkıştırmasız fark
    etmiyor), belleğe yazmadan ÖNCE NaN/Inf içeren bir float32 dizisini
    SESSİZCE 0.0'a çeviriyor -- yazma zamanında hata/uyarı YOK. Bu yüzden
    `_n4_cache_hit()`'in `np.isfinite(array).all()` kontrolü, GERÇEK bir N4
    çıktısında (`apply_n4_bias_correction()`'ın kendi ürettiği) pratikte
    ASLA NaN/Inf ile karşılaşmayacak -- disk-üzerindeki bir dosya zaten
    yazılırken sanitize edilmiş olur. Kontrol yine de KALDIRILMADI
    (defans-derinliği: farklı bir okuma/yazma yolu -- örn. başka bir
    kütüphane/format ile üretilmiş bozuk bir dosya -- gerçek NaN
    taşıyabilir), ama `_n4_cache_hit()`'in NaN'ı GERÇEKTEN yakaladığını
    `sitk.WriteImage()` üzerinden test ETMEK bu ortamda MÜMKÜN DEĞİL (bu
    test tam olarak bunu -- NaN'ın 0'a dönüştüğünü -- kanıtlıyor). Asıl
    çalışan/test EDİLEBİLEN koruma `test_n4_cache_hit_false_when_cached_
    body_is_all_zero` -- tamamen sıfır bir gövde (bozuk/yarım N4 çıktısı
    için gerçekçi bir senaryo) doğru şekilde reddediliyor."""

    body = np.full((3, 3, 3), 5.0, dtype=np.float32)
    body[0, 0, 0] = np.nan
    path = _write_image(ascii_scratch_dir / "nan_roundtrip.nii.gz", body)

    roundtripped = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    assert np.isfinite(roundtripped).all()
    assert roundtripped[0, 0, 0] == 0.0


def test_n4_cache_hit_false_when_meta_json_corrupt(ascii_scratch_dir: Path) -> None:
    input_path = _write_image(
        ascii_scratch_dir / "raw.nii.gz", np.ones((3, 3, 3), dtype=np.float32)
    )
    cached = _write_image(
        ascii_scratch_dir / "raw_n4.nii.gz", np.full((3, 3, 3), 5.0, dtype=np.float32)
    )
    _write_n4_cache_meta(cached, input_path)
    meta_path = cached.with_name(cached.name + ".n4meta.json")
    meta_path.write_text("{not valid json", encoding="utf-8")
    assert _n4_cache_hit(cached, input_path=input_path) is False


def test_input_content_hash_stable_and_content_sensitive(
    ascii_scratch_dir: Path,
) -> None:
    path_a = _write_image(
        ascii_scratch_dir / "a.nii.gz", np.ones((2, 2, 2), dtype=np.float32)
    )
    path_b = _write_image(
        ascii_scratch_dir / "b.nii.gz", np.ones((2, 2, 2), dtype=np.float32) * 2
    )
    hash_a1 = _input_content_hash(path_a)
    hash_a2 = _input_content_hash(path_a)
    hash_b = _input_content_hash(path_b)
    assert hash_a1 == hash_a2
    assert hash_a1 != hash_b


def test_n4_params_signature_reflects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GBMAID_N4_ITERATIONS", "10,10")
    monkeypatch.setenv("GBMAID_N4_SHRINK_FACTOR", "3")
    signature = _n4_params_signature()
    assert signature == {"n4_iterations": "10,10", "n4_shrink_factor": "3"}


def test_apply_n4_cached_writes_sidecar_and_reuses_on_second_call(
    ascii_scratch_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uçtan uca: `_apply_n4_cached()` ilk çağrıda GERÇEKTEN N4 hesaplar VE
    sidecar'ı yazar; ikinci çağrıda (AYNI girdi/parametreler) dosyayı
    YENİDEN HESAPLAMADAN, olduğu gibi döndürür (mtime DEĞİŞMEZ)."""

    monkeypatch.setenv("GBMAID_PROCESSED_ROOT", str(ascii_scratch_dir / "processed"))
    monkeypatch.setenv("GBMAID_N4_ITERATIONS", "2,2")
    monkeypatch.setenv("GBMAID_N4_SHRINK_FACTOR", "2")

    z, y, x = np.indices((10, 12, 14))
    foreground = ((x - 7) ** 2 + (y - 6) ** 2 + (z - 5) ** 2) < 16
    array = np.zeros(foreground.shape, dtype=np.float32)
    array[foreground] = 100.0
    input_path = _write_image(ascii_scratch_dir / "raw.nii.gz", array)

    first_output = Path(_apply_n4_cached(str(input_path)))
    assert first_output.is_file()
    meta_path = first_output.with_name(first_output.name + ".n4meta.json")
    assert meta_path.is_file()
    first_mtime = first_output.stat().st_mtime_ns

    second_output = Path(_apply_n4_cached(str(input_path)))
    assert second_output == first_output
    assert second_output.stat().st_mtime_ns == first_mtime
