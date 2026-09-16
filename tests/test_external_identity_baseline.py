# -*- coding: utf-8 -*-
"""Dondurulmus kimlik baseline'inin PIN testi (DB gerektirmez).

NEDEN (2026-08-19, Codex 2. tur bulgusu): baseline JSON'i dogrulayicilarin
tek referansiydi ama DOSYANIN KENDISI korumasizdi -- sessizce yeniden
uretilse/oynansa hicbir sey kirmizi yanmazdi. Bu test, alti hash'i KOD
ICINDE SABIT olarak pinler (`tests/test_omics_scores.py`'nin
FIXTURE_SHA256 deseniyle ayni): dosya degisirse pytest KIRMIZI yanar.

Mesru yenileme yolu: `freeze_external_identity_baseline.py --refreeze
--reason "..."` + BU sabitlerin BILINCLI guncellenmesi (iki ayri, gorunur
adim -- sessiz yol yok).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).absolute().parent.parent / "tools"))
from external_identity_baseline import (  # noqa: E402
    BASELINE_PATH,
    ExternalIdentityMismatchError,
    verify_or_fail,
)

#: 2026-08-19 dondurmasi (sema v2). Cipa: 6/6 bit-birebir Harrell
#: reproduksiyonu. Degistirmek = bilincli --refreeze + bu tablonun
#: guncellenmesi.
PINNED = {
    "UCSF-PDGM": {
        "pid_sha256": "0c28bbfe2124fec59bb784fd820870d55e0e8568416c0d58a294f11a55867a8c",
        "patients_sha256": "de9341288d6c53669c0546bcc9cffabed7c14c3cdd7fa09bb9bbf36dee9a2ee8",
        "wt_pivot_sha256": "0b6f12cdbe45064aec2a32bff33bb8087d8c4f5c17477b9a1d99bb5a10cab610",
        "n_pivot_patients": 295,
    },
    "UPenn-GBM": {
        "pid_sha256": "1b1091efa1c5a384c114c663759e90724555eff4d9ee82de8f913a03e5b7ab53",
        "patients_sha256": "c3b76296cdd597bc3701db781a62ad65937c1bdfb136abde86c574fd127508f9",
        "wt_pivot_sha256": "d2a8728f49182e2e4dbe6f121f503ea6fbca4cc3ad72c6013faaafa4b32ad9c1",
        "n_pivot_patients": 611,
    },
}


def _load_baseline() -> dict:
    assert BASELINE_PATH.is_file(), (
        f"Baseline dosyasi YOK: {BASELINE_PATH} -- dogrulayici scriptler "
        "calisamaz; freeze_external_identity_baseline.py kosulmali."
    )
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def test_baseline_file_matches_pinned_hashes() -> None:
    """Dosyadaki alti hash + iki n, koddaki sabitlerle BIREBIR ayni olmali."""
    baseline = _load_baseline()
    assert baseline.get("schema_version") == 2
    for cohort, expected in PINNED.items():
        actual = baseline["cohorts"][cohort]
        for key, val in expected.items():
            assert actual[key] == val, (
                f"[{cohort}] {key} PIN'den sapti -- baseline dosyasi "
                "degistirilmis. Mesruysa: --refreeze --reason + bu test "
                "sabitlerinin bilincli guncellenmesi."
            )


def test_baseline_carries_justification_and_freeze_time() -> None:
    baseline = _load_baseline()
    assert baseline.get("frozen_at"), "frozen_at bos olamaz"
    assert len(baseline.get("justification", "")) > 40, (
        "justification bos/kisa -- gerekcesiz baseline kabul edilmez"
    )


def test_verify_or_fail_rejects_tampered_copy(tmp_path) -> None:
    """KIRMIZI: tek hash degistirilmis bir baseline kopyasi REDDEDILMELI."""
    baseline = _load_baseline()
    tampered = json.loads(json.dumps(baseline))
    tampered["cohorts"]["UCSF-PDGM"]["wt_pivot_sha256"] = "0" * 64
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(tampered), encoding="utf-8")
    computed = {"cohorts": baseline["cohorts"]}
    with pytest.raises(ExternalIdentityMismatchError):
        verify_or_fail(computed, baseline_path=bad)


def test_verify_or_fail_rejects_missing_cohort(tmp_path) -> None:
    """KIRMIZI: kapsami sessizce daraltilmis (UPenn'siz) baseline REDDEDILMELI."""
    baseline = _load_baseline()
    narrowed = json.loads(json.dumps(baseline))
    del narrowed["cohorts"]["UPenn-GBM"]
    bad = tmp_path / "narrowed.json"
    bad.write_text(json.dumps(narrowed), encoding="utf-8")
    with pytest.raises(ExternalIdentityMismatchError):
        verify_or_fail({"cohorts": baseline["cohorts"]}, baseline_path=bad)


def test_verify_or_fail_rejects_v1_schema(tmp_path) -> None:
    """KIRMIZI: eski (v1, tek-kohort) sema sessizce kabul EDILMEMELI."""
    old_style = {"pid_sha256": "x", "frozen_at": "2026-08-19"}
    bad = tmp_path / "v1.json"
    bad.write_text(json.dumps(old_style), encoding="utf-8")
    baseline = _load_baseline()
    with pytest.raises(ExternalIdentityMismatchError):
        verify_or_fail({"cohorts": baseline["cohorts"]}, baseline_path=bad)
