"""HIGH regresyon testi (2026-08-14, A3 review-gate/Codex bulgusu) --
LUMIERE'in ikinci-seviye resample-kurtarmasının SADECE İLK bölgeyi
kurtarması bug'ı.

BULGU (Codex, `tools/run_pyradiomics_lumiere.py` Pass 3): resample başarılı
olduğunda yalnız `retry_zscore_path`/`retry_mask_path` YEREL
değişkenleri kullanılıyordu -- döngünün ÜST kapsamındaki `zscore_path`/
`mask_path` (bir sonraki `region` iterasyonunun İLK denemesinin
kullandığı değişkenler) GÜNCELLENMİYORDU. Sonuç: `scan_resampled` artık
`True` olduğu için (`if scan_resampled or "geometry mismatch" not in
str(exc).lower():`) sonraki bölgeler yeniden kurtarma denenmeden
doğrudan `HATA`ya düşüyordu -- 5 bölgeden yalnız DÖNGÜDE İLK hata veren
bölge kurtuluyordu, WT_derived/TC_derived dahil kalanlar eksik kalıyordu.

DÜZELTME: resample başarılı olduğunda `zscore_path`/`mask_path` artık
TARAMA SEVİYESİNE yükseltiliyor -- kalan TÜM bölgeler (native + türetilmiş)
düzeltilmiş geometriyle doğrudan İLK denemede çalışır.

Bu test `resample_image_and_mask()`'i MOCK'lar (koordinatörün talimatı) ve
gerçek DB/NAS'a hiç dokunmadan (tüm DB/NAS çağrıları da mock'lanmış), sahte
ama GERÇEKÇİ bir "geometri kontrolü yapan" fake extractor ile şu senaryoyu
KANITLANABİLİR şekilde egzersiz eder: 5 bölgeli (Necrosis/Contrast-
enhancing/Edema/WT_derived/TC_derived) TEK bir LUMIERE taraması, İLK
bölgede (Necrosis) "geometry mismatch" hatası alır, resample-kurtarma
BAŞARILI olur -- test, kalan 4 bölgenin de (retry sonrası düzeltilmiş
geometriyle) `OK`/`LABEL_YOK_0_VOXEL` döndüğünü, HİÇBİRİNİN `HATA` ile
başarısız OLMADIĞINI doğrular.

MANUEL DOĞRULAMA (bu test dosyasının YAZILMA amacı, rapora da yazıldı):
düzeltme (`zscore_path = str(retry_zscore_path)` / `mask_path =
retry_mask_path` satırları) GERİ ALINDIĞINDA bu test GERÇEKTEN KIRMIZI
oluyor -- 2026-08-14'te elle doğrulandı (bkz. log/2026-08-14.md).
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

import tools.run_pyradiomics_lumiere as lumiere_module


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


def _write_image(
    path: Path, array: np.ndarray, *, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path), True)
    return path


class _FakeExtractor:
    """Gerçek `RadiomicsFeatureExtractor`'ın YERİNE geçen, HIZLI ve
    DETERMİNİSTİK bir sahte -- yalnız GERÇEK PyRadiomics'in de yaptığı
    TEK şeyi taklit eder: image/mask spacing'i eşleşmiyorsa "geometry
    mismatch" içeren bir hata fırlatır. `settings`/`enabledImagetypes`/
    `enabledFeatures`, script'in provenance-manifest adımının (`_write_
    provenance_manifest()`) C32 doğrulamasından GEÇMESİ için gerçek
    extractor'ın ürettiği değerlerle AYNI şekilde dolduruldu.
    """

    def __init__(self) -> None:
        self.settings = {"binCount": 32, "normalize": False}
        self.enabledImagetypes = {"Original": {}}
        self.enabledFeatures = {
            "shape": [],
            "firstorder": [],
            "glcm": [],
            "glrlm": [],
            "glszm": [],
            "ngtdm": [],
            "gldm": [],
        }
        self.calls: list[tuple[str, bool]] = []

    def execute(self, image_arg, mask_arg, label=None):
        image = sitk.ReadImage(str(image_arg)) if isinstance(image_arg, (str, Path)) else image_arg
        mask = mask_arg if isinstance(mask_arg, sitk.Image) else sitk.ReadImage(str(mask_arg))

        mask_path_repr = mask_arg if isinstance(mask_arg, (str, Path)) else "<derived-in-memory>"
        matched = image.GetSpacing() == mask.GetSpacing()
        self.calls.append((str(mask_path_repr), matched))

        if not matched:
            raise RuntimeError(
                "Image/Mask geometry mismatch. Potential fix: resample "
                "both image and mask to the same reference space."
            )

        mask_array = sitk.GetArrayViewFromImage(mask)
        voxel_count = int(np.sum(mask_array == label)) if label is not None else int(np.sum(mask_array != 0))
        if voxel_count == 0:
            raise RuntimeError("Label not present in mask")

        return {
            "original_shape_VoxelVolume": float(voxel_count),
            "original_shape_SurfaceArea": 1.0,
            "original_firstorder_Entropy": 1.0,
            "original_glcm_Contrast": 1.0,
        }


class _DummyConnection:
    def close(self) -> None:
        return None


def _run_lumiere_resample_recovery_scenario(
    ascii_scratch_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, list[dict[str, object]]]:
    """Senaryoyu kur ve `main()`'i çalıştır -- (exit_code, csv_rows) döner.

    Bu fonksiyon HER ZAMAN üretim kodundaki GÜNCEL (düzeltilmiş) davranışı
    çalıştırır. Testin GERÇEKTEN bir şeyi koruduğunu kanıtlamak için ayrı
    bir "kırmızı" otomatik test YAZILMADI -- bunun yerine üretim
    dosyasındaki 2 satırlık yükseltme (`zscore_path = str(retry_zscore_
    path)` / `mask_path = retry_mask_path`) GEÇİCİ OLARAK geri alınıp bu
    test elle çalıştırıldı, KIRMIZI olduğu doğrulandı, sonra düzeltme
    GERİ KONULUP tekrar YEŞİL olduğu doğrulandı (sonuçlar rapora/log'a
    yazıldı) -- bkz. bu dosyanın modül docstring'i.
    """

    monkeypatch.setenv("GBMAID_N4_ITERATIONS", "2,2")
    monkeypatch.setenv("GBMAID_N4_SHRINK_FACTOR", "2")

    # ---- Sahte kohort: TEK LUMIERE taraması ----
    patient_id = "Patient-TEST"
    scan_id = 999001
    raw_image_path = _write_image(
        ascii_scratch_dir / "raw" / "CT1.nii.gz",
        (np.indices((14, 16, 18)).sum(axis=0) % 7).astype(np.float32) + 50.0,
        spacing=(1.0, 1.0, 1.0),
    )

    # ORİJİNAL (Pass 1'in gördüğü) çok-etiketli maske -- BİLEREK Z-score
    # görüntüsüyle (spacing 1,1,1) UYUŞMAYAN bir spacing'de (2,2,2).
    # Necrosis=1, Contrast-enhancing=2, Edema=3 -- hepsi nonzero voksel.
    original_mask_array = np.zeros((6, 6, 6), dtype=np.uint8)
    original_mask_array[0:2, 0:2, 0:2] = 1  # Necrosis
    original_mask_array[2:4, 2:4, 2:4] = 2  # Contrast-enhancing
    original_mask_array[4:6, 4:6, 4:6] = 3  # Edema
    original_mask_path = _write_image(
        ascii_scratch_dir / "raw" / "ct1_seg_mask.nii.gz",
        original_mask_array,
        spacing=(2.0, 2.0, 2.0),
    )

    # RETRY (resample-kurtarma SONRASI) maske -- Z-score görüntüsüyle
    # AYNI spacing'de (1,1,1), AYNI 3 etiketi nonzero taşıyor.
    retry_mask_array = np.zeros((14, 16, 18), dtype=np.uint8)
    retry_mask_array[0:3, 0:3, 0:3] = 1
    retry_mask_array[4:7, 4:7, 4:7] = 2
    retry_mask_array[8:11, 8:11, 8:11] = 3
    retry_mask_path = _write_image(
        ascii_scratch_dir / "retry" / "ct1_seg_mask_on_ct1_1mm.nii.gz",
        retry_mask_array,
        spacing=(1.0, 1.0, 1.0),
    )

    resample_calls: list[str] = []

    def fake_resample_image_and_mask(image_path, mask_path, *, output_spacing=(1.0, 1.0, 1.0), output_dir=None, max_volume_drift=0.10):
        resample_calls.append(str(image_path))
        return {
            "status": "PASS",
            "image_output_path": str(image_path),
            "mask_output_path": str(retry_mask_path),
            "output_mask": {"labels": [1.0, 2.0, 3.0]},
            "volume_drift_fraction": 0.0,
            "output_geometry_match": True,
            "labels_preserved": True,
        }

    def fake_resolve_ready_mask(*, source, image_path, patient_id, scan_id, raise_on_geometry_mismatch=False):
        return {
            "patient_id": patient_id,
            "scan_id": scan_id,
            "source": "LUMIERE",
            "image_path": str(raw_image_path),
            "mask_path": str(original_mask_path),
            "mask_source": "lumiere_deepbratumia_native",
            "confidence_score": None,
            # Pass-1 seviyesinde GEOMETRİ EŞLEŞİYOR "sayılıyor" -- bu test
            # yalnız Pass 3'ün İKİNCİ-seviye (PyRadiomics'in kendi dahili
            # kontrolü tetiklediği) kurtarma yolunu izole ediyor.
            "geometry": {"geometry_match": True, "mask": {"labels": [1.0, 2.0, 3.0]}},
            "warnings": [],
        }

    def fake_fetch_lumiere_t1ce_scans(connection, modality):
        return [
            {
                "scan_id": scan_id,
                "patient_id": patient_id,
                "file_path": str(raw_image_path),
                "timepoint_label": "week-000",
            }
        ]

    def fake_fetch_existing_volumes(connection):
        return {}

    monkeypatch.setattr(lumiere_module, "resample_image_and_mask", fake_resample_image_and_mask)
    monkeypatch.setattr(lumiere_module, "resolve_ready_mask", fake_resolve_ready_mask)
    monkeypatch.setattr(lumiere_module, "resolve_nas_path", lambda stored_path, **kwargs: Path(stored_path))
    monkeypatch.setattr(lumiere_module, "_fetch_lumiere_t1ce_scans", fake_fetch_lumiere_t1ce_scans)
    monkeypatch.setattr(lumiere_module, "_fetch_existing_volumes", fake_fetch_existing_volumes)
    monkeypatch.setattr(lumiere_module, "get_connection", lambda readonly=True: _DummyConnection())

    fake_extractor = _FakeExtractor()
    monkeypatch.setattr(lumiere_module, "_make_extractor", lambda: fake_extractor)

    report_path = ascii_scratch_dir / "lumiere_c32.csv"
    stats_path = ascii_scratch_dir / "stats_lumiere.json"
    processed_root = ascii_scratch_dir / "processed"
    resample_root = ascii_scratch_dir / "resample_out"

    argv = [
        "run_pyradiomics_lumiere.py",
        "--report",
        str(report_path),
        "--stats-path",
        str(stats_path),
        "--processed-root",
        str(processed_root),
        "--resample-output-root",
        str(resample_root),
        "--skip-image-count-gate",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    exit_code = lumiere_module.main()

    import csv

    with report_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    return exit_code, rows


def test_resample_recovery_fixes_all_regions_not_just_first(
    ascii_scratch_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DÜZELTME SONRASI (mevcut üretim kodu): 5 bölgenin TAMAMI (Necrosis/
    Contrast-enhancing/Edema/WT_derived/TC_derived) `OK` veya
    `LABEL_YOK_0_VOXEL` dönmeli -- HİÇBİRİ `HATA` ile başarısız OLMAMALI."""

    exit_code, rows = _run_lumiere_resample_recovery_scenario(ascii_scratch_dir, monkeypatch)

    assert exit_code == 0
    assert len(rows) == 5
    regions_seen = {row["region"] for row in rows}
    assert regions_seen == {
        "Necrosis",
        "Contrast-enhancing",
        "Edema",
        "WT_derived",
        "TC_derived",
    }

    statuses = {row["region"]: row["status"] for row in rows}
    for region, status in statuses.items():
        assert status in ("OK", "LABEL_YOK_0_VOXEL"), (
            f"{region} bölgesi HATA ile başarısız oldu (HIGH bug'ı geri "
            f"gelmiş olabilir): status={status!r}"
        )
    # Bu senaryoda 3 native etiketin hepsi retry maskesinde nonzero --
    # LABEL_YOK_0_VOXEL hiç beklenmiyor, tam 5 OK bekleniyor.
    assert all(status == "OK" for status in statuses.values())

    # Tüm satırlar resampled=True olarak işaretlenmeli (kurtarma yolu).
    assert all(row["resampled"] == "True" for row in rows)

    # WT_derived = Necrosis+Contrast-enhancing+Edema hacim toplamı (union,
    # örtüşme yok) -- türetilmiş bölgenin GERÇEKTEN retry (düzeltilmiş)
    # geometrisinden üretildiğinin kanıtı.
    voxel = {row["region"]: float(row["voxel_volume"]) for row in rows}
    assert voxel["WT_derived"] == pytest.approx(
        voxel["Necrosis"] + voxel["Contrast-enhancing"] + voxel["Edema"]
    )
    assert voxel["TC_derived"] == pytest.approx(voxel["Necrosis"] + voxel["Contrast-enhancing"])
