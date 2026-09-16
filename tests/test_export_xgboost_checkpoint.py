"""`tools/export_xgboost_checkpoint.py` icin DB'siz, saf birim testleri.

KAPSAM BILINCLI SINIRLI: bu script'in asil govdesi (`main()`) canli DB'ye
bagli, ~15-20dk suren bir uretim kosusudur (bkz. modul docstring'i) --
`tools/export_v3b_deployment_checkpoint.py` (Cox tarafi, 2026-09-12) icin
de ayri bir pytest dosyasi YAZILMAMISTI, aynı ilke burada da gecerli:
gercek dogrulama script'in KENDI ICINDE (canli kosu sirasinda, referans
JSON + v3b checkpoint'iyle bit-birebir karsilastirma) yapilir, fail-closed
(AssertionError -> checkpoint YAZILMAZ). Bu dosya SADECE saf/DB-siz
yardimci fonksiyonu (`_assert_close`) test eder."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from export_xgboost_checkpoint import _assert_close  # noqa: E402


def test_assert_close_identical_values_returns_zero_diff() -> None:
    assert _assert_close("label", 0.723729, 0.723729) == 0.0


def test_assert_close_within_absolute_tolerance_passes() -> None:
    # ABS_TOL = 1e-9 -- bunun altindaki bir fark istisna FIRLATMAMALI.
    diff = _assert_close("label", 1.0000000001, 1.0)
    assert diff < 1e-9 or diff == pytest.approx(1e-10, abs=1e-9)


def test_assert_close_within_relative_tolerance_passes_for_large_values() -> None:
    # REL_TOL = 1e-6 -- buyuk mutlak degerlerde (orn. GLSZM Energy gibi
    # olcekler) bagil tolerans devreye girmeli, mutlak tolerans yetersiz
    # kalsa bile istisna FIRLAMAMALI.
    got = 10_234_612.48280388
    want = got * (1 + 1e-7)
    diff = _assert_close("label", got, want)
    assert diff > 0.0  # gercekten bir fark var, ama tolerans icinde


def test_assert_close_raises_when_both_tolerances_exceeded() -> None:
    with pytest.raises(AssertionError, match=r"UYUSMUYOR"):
        _assert_close("cox_best_penalizer", 0.20, 0.05)


def test_assert_close_raises_for_small_values_beyond_abs_tolerance() -> None:
    # Kucuk mutlak degerlerde bagil tolerans (1e-6) tek basina bir 1e-8
    # farki KURTARMAZ (rel=1e-8/1e-8=1.0 > 1e-6 GORUNMEZ cunku deger
    # kucukse rel kucuk kalabilir) -- burada bilerek ABS_TOL'un COK
    # ustunde bir mutlak fark (1e-3) ve REL_TOL'un da ustunde bir bagil
    # fark (0.5) veriyoruz, ikisi de asilir.
    with pytest.raises(AssertionError):
        _assert_close("tiny", 0.003, 0.002)
