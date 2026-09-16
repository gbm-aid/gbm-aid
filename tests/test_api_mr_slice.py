"""``GET /patient/{patient_id}/mr_slice`` (api/mr_slice.py) -- birim +
gerçek DB/NAS duman testleri.

Görev talimatı: paralel-36 (backend-agent-W1, 2026-09-15/16). Diğer test
dosyalarının deseniyle AYNI -- endpoint doğrudan Python fonksiyonu olarak
çağrılır (`Query(...)` varsayılanları HER çağrıda AÇIKÇA geçirilir, bkz.
`tests/test_api_patients.py` modül dokstring'i).

Gerçek DB/NAS gerektiren testler (`_real_db_smoke_` sonekli) NAS'a/DB'ye
erişilemezse `pytest.skip` ile atlanır (mevcut desen).
"""

from __future__ import annotations

import struct
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi import HTTPException

import api.mr_slice as mr_slice_module


# =====================================================================
# 1) Birim testleri -- DB/NAS YOK
# =====================================================================


def test_axial_slice_applies_rot90_and_horizontal_flip():
    # 2x2 taban düzlem, tek z dilimi.
    volume = np.array([[[10], [20]], [[30], [40]]], dtype=np.uint8)
    plane = mr_slice_module._axial_slice(volume, 0)
    # plane girişi [[10,20],[30,40]] -- rot90 sonrası [[20,40],[10,30]],
    # sonra sütunlar ters çevrilir -> [[40,20],[30,10]].
    expected = np.rot90(np.array([[10, 20], [30, 40]]))[:, ::-1]
    assert np.array_equal(plane, expected)


def test_render_png_returns_valid_png_bytes():
    img = np.zeros((4, 4, 1), dtype=np.uint8)
    seg = np.zeros((4, 4, 1), dtype=np.uint8)
    seg[1:3, 1:3, 0] = 1
    volume = {
        "img_u8": img,
        "seg_u8": seg,
        "region_labels": {"WT": 1},
    }
    png_bytes = mr_slice_module._render_png(volume, 0, with_overlay=True)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    png_no_overlay = mr_slice_module._render_png(volume, 0, with_overlay=False)
    assert png_no_overlay[:8] == b"\x89PNG\r\n\x1a\n"
    assert png_no_overlay != png_bytes


def test_resolve_ucsf_mask_finds_matching_segmentation(tmp_path):
    image_path = tmp_path / "UCSF-PDGM-999_T1c.nii.gz"
    image_path.write_bytes(b"")
    mask_path = tmp_path / "UCSF-PDGM-999_tumor_segmentation.nii.gz"
    mask_path.write_bytes(b"")

    resolved_mask, mask_source = mr_slice_module._resolve_ucsf_mask(image_path)
    assert resolved_mask == mask_path
    assert mask_source == "ucsf_native"


def test_resolve_ucsf_mask_rejects_non_t1c_stem(tmp_path):
    image_path = tmp_path / "UCSF-PDGM-999_T1.nii.gz"
    image_path.write_bytes(b"")
    with pytest.raises(mr_slice_module.UnsupportedMaskSourceError):
        mr_slice_module._resolve_ucsf_mask(image_path)


def test_resolve_ucsf_mask_missing_segmentation_raises_file_not_found(tmp_path):
    image_path = tmp_path / "UCSF-PDGM-999_T1c.nii.gz"
    image_path.write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        mr_slice_module._resolve_ucsf_mask(image_path)


def test_select_scan_raises_when_no_t1ce_scans(monkeypatch):
    monkeypatch.setattr(mr_slice_module, "_fetch_t1ce_scans", lambda conn, pid: [])
    monkeypatch.setattr(mr_slice_module, "is_lumiere_patient_id", lambda pid: False)
    with pytest.raises(mr_slice_module.NoT1ceScanError):
        mr_slice_module._select_scan(None, "UPENN-GBM-00001", None)


def test_select_scan_single_scan_non_lumiere_no_scan_id_needed(monkeypatch):
    scans = [{"scan_id": 42, "file_path": "a/b/c.nii.gz", "source": "UPenn-GBM"}]
    monkeypatch.setattr(mr_slice_module, "_fetch_t1ce_scans", lambda conn, pid: scans)
    monkeypatch.setattr(mr_slice_module, "is_lumiere_patient_id", lambda pid: False)

    result = mr_slice_module._select_scan(None, "UPENN-GBM-00001", None)
    assert result["scan_id"] == 42
    assert result["source"] == "UPenn-GBM"
    assert "lumiere_canonical_visit" not in result


def test_select_scan_multi_scan_non_lumiere_without_scan_id_is_ambiguous(monkeypatch):
    scans = [
        {"scan_id": 1, "file_path": "a.nii.gz", "source": "TCGA-GBM"},
        {"scan_id": 2, "file_path": "b.nii.gz", "source": "TCGA-GBM"},
    ]
    monkeypatch.setattr(mr_slice_module, "_fetch_t1ce_scans", lambda conn, pid: scans)
    monkeypatch.setattr(mr_slice_module, "is_lumiere_patient_id", lambda pid: False)

    with pytest.raises(mr_slice_module.MultiScanAmbiguousError):
        mr_slice_module._select_scan(None, "TCGA-06-XXXX", None)


def test_select_scan_multi_scan_non_lumiere_with_valid_scan_id(monkeypatch):
    scans = [
        {"scan_id": 1, "file_path": "a.nii.gz", "source": "TCGA-GBM"},
        {"scan_id": 2, "file_path": "b.nii.gz", "source": "TCGA-GBM"},
    ]
    monkeypatch.setattr(mr_slice_module, "_fetch_t1ce_scans", lambda conn, pid: scans)
    monkeypatch.setattr(mr_slice_module, "is_lumiere_patient_id", lambda pid: False)

    result = mr_slice_module._select_scan(None, "TCGA-06-XXXX", 2)
    assert result["scan_id"] == 2
    assert result["file_path"] == "b.nii.gz"


def test_select_scan_non_lumiere_invalid_scan_id_raises(monkeypatch):
    scans = [{"scan_id": 1, "file_path": "a.nii.gz", "source": "TCGA-GBM"}]
    monkeypatch.setattr(mr_slice_module, "_fetch_t1ce_scans", lambda conn, pid: scans)
    monkeypatch.setattr(mr_slice_module, "is_lumiere_patient_id", lambda pid: False)

    with pytest.raises(mr_slice_module.ScanIdNotFoundError):
        mr_slice_module._select_scan(None, "TCGA-06-XXXX", 999)


def test_select_scan_lumiere_uses_canonical_visit_regardless_of_scan_count(monkeypatch):
    scans = [
        {"scan_id": 5, "file_path": "week-000.nii.gz", "source": "LUMIERE"},
        {"scan_id": 9, "file_path": "week-010.nii.gz", "source": "LUMIERE"},
    ]
    monkeypatch.setattr(mr_slice_module, "_fetch_t1ce_scans", lambda conn, pid: scans)
    monkeypatch.setattr(mr_slice_module, "is_lumiere_patient_id", lambda pid: True)
    selection = SimpleNamespace(scan_id=9, timepoint_label="week-010")
    monkeypatch.setattr(
        mr_slice_module, "fetch_lumiere_canonical_visit_selection", lambda pid: selection
    )

    result = mr_slice_module._select_scan(None, "Patient-028", None)
    assert result["scan_id"] == 9
    assert result["lumiere_canonical_visit"]["timepoint_label"] == "week-010"


def test_select_scan_lumiere_rejects_mismatched_requested_scan_id(monkeypatch):
    scans = [{"scan_id": 5, "file_path": "week-000.nii.gz", "source": "LUMIERE"}]
    monkeypatch.setattr(mr_slice_module, "_fetch_t1ce_scans", lambda conn, pid: scans)
    monkeypatch.setattr(mr_slice_module, "is_lumiere_patient_id", lambda pid: True)
    selection = SimpleNamespace(scan_id=5, timepoint_label="week-000")
    monkeypatch.setattr(
        mr_slice_module, "fetch_lumiere_canonical_visit_selection", lambda pid: selection
    )

    with pytest.raises(mr_slice_module.ScanIdNotFoundError):
        mr_slice_module._select_scan(None, "Patient-028", 999)


def test_select_scan_lumiere_canonical_scan_id_missing_from_t1ce_list_is_data_bug(monkeypatch):
    scans = [{"scan_id": 5, "file_path": "week-000.nii.gz", "source": "LUMIERE"}]
    monkeypatch.setattr(mr_slice_module, "_fetch_t1ce_scans", lambda conn, pid: scans)
    monkeypatch.setattr(mr_slice_module, "is_lumiere_patient_id", lambda pid: True)
    selection = SimpleNamespace(scan_id=999, timepoint_label="week-999")
    monkeypatch.setattr(
        mr_slice_module, "fetch_lumiere_canonical_visit_selection", lambda pid: selection
    )

    with pytest.raises(mr_slice_module.NoT1ceScanError):
        mr_slice_module._select_scan(None, "Patient-028", None)


def test_get_mr_slice_patient_not_found_raises_404(monkeypatch):
    monkeypatch.setattr(mr_slice_module, "_fetch_patient_exists", lambda conn, pid: False)

    class _FakeConn:
        def close(self):
            pass

    monkeypatch.setattr(mr_slice_module, "_get_db_connection", lambda: _FakeConn())

    with pytest.raises(HTTPException) as excinfo:
        mr_slice_module.get_mr_slice("NOPE-999", z=None, overlay=True, scan_id=None)
    assert excinfo.value.status_code == 404


# =====================================================================
# 2) Gerçek DB + NAS duman testleri -- erişilemezse ATLANIR.
# =====================================================================


def _call_get_mr_slice(patient_id: str, **overrides: Any):
    params: dict[str, Any] = {"z": None, "overlay": True, "scan_id": None}
    params.update(overrides)
    return mr_slice_module.get_mr_slice(patient_id, **params)


def test_get_mr_slice_real_db_nas_smoke_ucsf_single_scan_returns_png():
    try:
        response = _call_get_mr_slice("UCSF-PDGM-167")
    except HTTPException as exc:
        if exc.status_code in (503, 500):
            pytest.skip(f"NAS/DB erisilemedi veya konfigure degil: {exc.detail}")
        raise
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gercek DB/NAS'a erisilemedi, smoke test atlandi: {exc}")

    assert response.media_type == "image/png"
    assert response.body[:8] == b"\x89PNG\r\n\x1a\n"
    assert response.headers["X-GBMAID-Source"] == "UCSF"


def test_get_mr_slice_real_db_nas_smoke_lumiere_multiscan_uses_canonical_visit():
    try:
        response = _call_get_mr_slice("Patient-028")
    except HTTPException as exc:
        if exc.status_code in (503, 500):
            pytest.skip(f"NAS/DB erisilemedi veya konfigure degil: {exc.detail}")
        raise
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gercek DB/NAS'a erisilemedi, smoke test atlandi: {exc}")

    assert response.media_type == "image/png"
    assert response.body[:8] == b"\x89PNG\r\n\x1a\n"
    assert "X-GBMAID-Lumiere-Timepoint" in response.headers


def test_get_mr_slice_real_db_smoke_lumiere_wrong_scan_id_rejected():
    try:
        with pytest.raises(HTTPException) as excinfo:
            _call_get_mr_slice("Patient-028", scan_id=999999)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gercek DB'ye erisilemedi, smoke test atlandi: {exc}")
        return
    assert excinfo.value.status_code == 422


def test_get_mr_slice_real_db_nas_smoke_z_out_of_range_returns_422():
    try:
        with pytest.raises(HTTPException) as excinfo:
            _call_get_mr_slice("UCSF-PDGM-167", z=10_000_000)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gercek DB/NAS'a erisilemedi, smoke test atlandi: {exc}")
        return
    assert excinfo.value.status_code == 422
