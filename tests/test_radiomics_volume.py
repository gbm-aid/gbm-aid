from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from pipeline.radiomics_volume import (
    UnknownMaskSourceError,
    UnsupportedWTTCDerivationError,
    build_derived_region_masks,
    compute_region_volumes_mm3,
    compute_volumes_for_mask_source,
    derived_region_has_voxels,
)


@pytest.fixture()
def ascii_scratch_dir():
    """ASCII-only geçici klasör.

    pytest'in varsayılan `tmp_path`'i bu makinede kullanıcı profilinin
    Türkçe karakteri (`Barış`) yüzünden SimpleITK'nin NIfTI yazıcısıyla
    UYUŞMUYOR (bkz. `tests/test_harmonization.py::ascii_scratch_dir` --
    AYNI desen, bu dosyada da ayrıca tanımlanıyor çünkü pytest fixture'ları
    modüller arası paylaşılmıyor). Bu dosyanın TÜM testleri (yeni A2
    testleri dahil, mevcut olanlar da) NIfTI dosyası yazdığı için bu kökü
    kullanır -- aksi halde hepsi yanlış nedenle (ortam path sorunu, kod
    mantığı DEĞİL) başarısız görünür (2026-08-14'te canlı ölçüldü).
    """

    base = Path("C:/gbmaid_pytest_scratch") / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _write_label_mask(
    path: Path,
    labels: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(labels.astype(np.uint8))
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path), True)
    return path


def test_binary_mask_volume_matches_voxel_count_times_spacing(
    ascii_scratch_dir: Path,
) -> None:
    # 4x5x6 hacimde 10 voksel etiketli, spacing=1mm izotropik -> 10 mm3 beklenir
    labels = np.zeros((4, 5, 6), dtype=np.uint8)
    flat = labels.reshape(-1)
    flat[:10] = 1
    labels = flat.reshape((4, 5, 6))
    mask_path = _write_label_mask(ascii_scratch_dir / "whole.nii.gz", labels)

    volumes = compute_region_volumes_mm3(mask_path, {"WT": 1})
    assert volumes == {"WT": 10.0}


def test_multilabel_mask_respects_non_isotropic_spacing(
    ascii_scratch_dir: Path,
) -> None:
    labels = np.zeros((2, 3, 3), dtype=np.uint8)
    labels[0, 0, 0] = 1  # NC
    labels[0, 0, 1] = 2  # ED
    labels[0, 0, 2] = 4  # ET
    mask_path = _write_label_mask(
        ascii_scratch_dir / "segm.nii.gz", labels, spacing=(2.0, 1.0, 0.5)
    )

    voxel_volume = 2.0 * 1.0 * 0.5
    volumes = compute_region_volumes_mm3(
        mask_path, {"NC": 1, "ED": 2, "ET": 4}
    )
    assert volumes == {
        "NC": voxel_volume,
        "ED": voxel_volume,
        "ET": voxel_volume,
    }


def test_compute_volumes_for_mask_source_uses_known_contract(
    ascii_scratch_dir: Path,
) -> None:
    labels = np.ones((2, 2, 2), dtype=np.uint8)
    mask_path = _write_label_mask(ascii_scratch_dir / "core.nii.gz", labels)

    volumes = compute_volumes_for_mask_source(mask_path, "provided_tumor_core")
    assert volumes == {"TC": 8.0}


def test_compute_volumes_for_mask_source_rejects_unknown_source(
    ascii_scratch_dir: Path,
) -> None:
    labels = np.ones((2, 2, 2), dtype=np.uint8)
    mask_path = _write_label_mask(ascii_scratch_dir / "core.nii.gz", labels)

    with pytest.raises(UnknownMaskSourceError):
        compute_volumes_for_mask_source(mask_path, "not_a_real_mask_source")


# ============================================================
# A2 (2026-08-14) -- WT_derived/TC_derived türetmesi
# ============================================================


def test_build_derived_region_masks_upenn_union_is_correct(
    ascii_scratch_dir: Path,
) -> None:
    # 1x1x6 hacim: NC=1, ED=2, ET=4, arta kalan 0 (background).
    labels = np.array([[[1, 2, 4, 0, 1, 2]]], dtype=np.uint8)
    mask_path = _write_label_mask(ascii_scratch_dir / "segm.nii.gz", labels)

    derived = build_derived_region_masks(mask_path, "upenn_expert")
    assert set(derived) == {"WT_derived", "TC_derived"}

    wt = sitk.GetArrayFromImage(derived["WT_derived"])
    tc = sitk.GetArrayFromImage(derived["TC_derived"])

    # WT = NC∪ED∪ET -> yalnız background (index 3) sıfır kalmalı.
    assert wt.tolist() == [[[1, 1, 1, 0, 1, 1]]]
    # TC = NC∪ET -> ED (index 1, 5) ve background (index 3) sıfır kalmalı.
    assert tc.tolist() == [[[1, 0, 1, 0, 1, 0]]]

    # Geometri (spacing/origin/direction) mask ile aynı olmalı.
    original = sitk.ReadImage(str(mask_path))
    assert derived["WT_derived"].GetSpacing() == original.GetSpacing()
    assert derived["WT_derived"].GetOrigin() == original.GetOrigin()


def test_build_derived_region_masks_lumiere_role_alias_mapping(
    ascii_scratch_dir: Path,
) -> None:
    # LUMIERE DeepBraTumIA: Necrosis=1, Contrast-enhancing=2, Edema=3.
    labels = np.array([[[1, 2, 3, 0]]], dtype=np.uint8)
    mask_path = _write_label_mask(ascii_scratch_dir / "seg_mask.nii.gz", labels)

    derived = build_derived_region_masks(mask_path, "lumiere_deepbratumia_native")
    wt = sitk.GetArrayFromImage(derived["WT_derived"])
    tc = sitk.GetArrayFromImage(derived["TC_derived"])

    assert wt.tolist() == [[[1, 1, 1, 0]]]
    # TC = NC(1)∪ET(Contrast-enhancing=2) -> Edema(3) hariç.
    assert tc.tolist() == [[[1, 1, 0, 0]]]


def test_build_derived_region_masks_rejects_hd_glio_fallback(
    ascii_scratch_dir: Path,
) -> None:
    """2026-08-14 kararı: HD-GLIO fallback (ödem YOK) için türetme YASAK --
    jenerik 'tüm etiketlerin birleşimi=WT' varsayımı sessizce üretilmemeli."""

    labels = np.array([[[1, 2, 0]]], dtype=np.uint8)
    mask_path = _write_label_mask(ascii_scratch_dir / "segmentation.nii.gz", labels)

    with pytest.raises(UnsupportedWTTCDerivationError, match="lumiere_hd_glio_fallback"):
        build_derived_region_masks(mask_path, "lumiere_hd_glio_fallback")


def test_build_derived_region_masks_rejects_tcga_mask_sources(
    ascii_scratch_dir: Path,
) -> None:
    """TCGA zaten native WT/TC üretiyor -- türetme fonksiyonu TCGA'nın
    provided_whole_tumor/provided_tumor_core mask_source'ları için de
    tanımlı DEĞİL (birincil modelde TCGA maskesine dokunulmuyor kararı)."""

    labels = np.ones((1, 1, 2), dtype=np.uint8)
    mask_path = _write_label_mask(ascii_scratch_dir / "whole.nii.gz", labels)

    with pytest.raises(UnsupportedWTTCDerivationError):
        build_derived_region_masks(mask_path, "provided_whole_tumor")


def test_compute_volumes_for_mask_source_ucsf_native_matches_upenn_contract(
    ascii_scratch_dir: Path,
) -> None:
    """2026-08-18 (Barış onayı) -- UCSF NC=1/ED=2/ET=4, UPenn ile BİREBİR AYNI."""

    labels = np.array([[[1, 2, 4, 0]]], dtype=np.uint8)
    mask_path = _write_label_mask(ascii_scratch_dir / "tumor_segmentation.nii.gz", labels)

    volumes = compute_volumes_for_mask_source(mask_path, "ucsf_native")
    assert volumes == {"NC": 1.0, "ED": 1.0, "ET": 1.0}


def test_build_derived_region_masks_ucsf_union_matches_upenn(
    ascii_scratch_dir: Path,
) -> None:
    """UCSF'in WT_derived/TC_derived türetmesi UPenn'inkiyle AYNI union
    kuralını (WT=NC∪ED∪ET, TC=NC∪ET) uygulamalı -- ayrı bir yol/mantık DEĞİL."""

    labels = np.array([[[1, 2, 4, 0, 1, 2]]], dtype=np.uint8)
    upenn = build_derived_region_masks(
        _write_label_mask(ascii_scratch_dir / "upenn_segm.nii.gz", labels), "upenn_expert"
    )
    ucsf = build_derived_region_masks(
        _write_label_mask(ascii_scratch_dir / "ucsf_seg.nii.gz", labels), "ucsf_native"
    )

    assert sitk.GetArrayFromImage(ucsf["WT_derived"]).tolist() == sitk.GetArrayFromImage(
        upenn["WT_derived"]
    ).tolist()
    assert sitk.GetArrayFromImage(ucsf["TC_derived"]).tolist() == sitk.GetArrayFromImage(
        upenn["TC_derived"]
    ).tolist()


def test_derived_region_has_voxels_detects_empty_mask() -> None:
    empty = sitk.GetImageFromArray(np.zeros((2, 2, 2), dtype=np.uint8))
    nonempty = sitk.GetImageFromArray(
        np.array([[[1, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=np.uint8)
    )

    assert derived_region_has_voxels(empty) is False
    assert derived_region_has_voxels(nonempty) is True
