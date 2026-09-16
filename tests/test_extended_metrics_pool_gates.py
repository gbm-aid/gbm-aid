# -*- coding: utf-8 -*-
"""Egitim havuzu (611/585) kapisi -- DORT degerlendirme aracinin sozlesmesi.

NEDEN VAR (2026-09-14, modeling-agent-S2, Baris'in 5 nolu karari)
=================================================================
`modeling-agent-R1` bulgusu: `build_training_frame()` egitim cercevesinin
tek giris noktasi ve KAYNAK whitelist'i (TCGA/UCSF reddi) her yoldan
geciyor -- AMA 611/585 SAYAC kilidi kapinin kendisinde DEGIL, ayri bir
`check_training_pool_counts()` cagrisi gerektiriyor. Rapora giren
sayilari ureten su dort arac o cagriyi HIC yapmiyordu:

    tools/evaluate_external_metrics.py      (Uno's C, td-AUC -- 6 varyant)
    tools/evaluate_v1_calibration.py        (Brier/IPA/kalibrasyon)
    tools/evaluate_v3c_extended_metrics.py  (v3c satiri)
    tools/bootstrap_extended_metric_cis.py  (genisletilmis bootstrap CI)

Dordu de egitim cercevesini standardizasyon istatistigi icin kuruyor
(v1 kalibrasyonunda "frame-insa dogrulamasi" olarak); havuz sessizce
kayarsa o istatistik de kayar ve HARICI skorlar sessizce bozulur.

BU TESTIN KAPSAMI (bilincli olarak DB'siz)
------------------------------------------
1. YAPISAL: her dort dosya kapiyi `regions=` + `dropped_patient_ids=`
   ile cagiriyor mu (fail-open eden eksik-argumanli bir cagri yok mu).
2. ANLAMSAL: her aracin GECTIGI argumanlarla kapi dogru davraniyor mu --
   ozellikle "609/583 tuzagi" (bkz. `test_v3c_*`).
Canli DB uzerindeki kirmizi/yesil kanit ayrica calistirilip rapora
gecirilmistir (6 varyant + k14 + v3c + kalibrasyon, hepsi YESIL; sahte
havuzla hepsi KIRMIZI) -- burada tekrarlanmaz, DB'ye baglidir.
"""
from __future__ import annotations


import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import train_cox_week3 as w3  # noqa: E402

TOOLS = PROJECT_ROOT / "tools"
GATED_TOOLS = [
    "evaluate_external_metrics.py",
    "evaluate_v1_calibration.py",
    "evaluate_v3c_extended_metrics.py",
    "bootstrap_extended_metric_cis.py",
]

#: `DECLARED_REGION_SHORTFALLS[("WT","TC")]`'in kimlik kaniti
WTTC_DROPPED = ["UPENN-GBM-00354", "UPENN-GBM-00397"]


def _extract_call(src: str, marker: str) -> str | None:
    """`marker` ile baslayan cagrinin PARANTEZ DENGELI govdesini dondurur.

    Duz regex yetmiyor: cagri ic ice parantez tasiyor
    (`w3.all_dropped_patient_ids(report)`, `int(train["event"].sum())`).
    """
    pos = 0
    while True:
        start = src.find(marker, pos)
        if start < 0:
            return None
        pos = start + 1
        depth = 0
        for j in range(start + len(marker) - 1, len(src)):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    body = src[start:j + 1]
                    # Yorum icindeki `check_training_pool_counts()`
                    # (bos argumanli anma) GERCEK cagri DEGILDIR -- atla.
                    if body[len(marker):-1].strip():
                        return body
                    break
        else:
            return None


# --------------------------------------------------------------------------
# 1. YAPISAL: kapi gercekten cagriliyor mu, fail-open argumanlarla degil mi
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tool", GATED_TOOLS)
def test_tool_calls_training_pool_gate(tool: str) -> None:
    """Dort aracin her biri `check_training_pool_counts()` cagirmali.

    Bu bir "kod var mi" testi degil bir SOZLESME testidir: bu cagri
    silinirse/unutulursa havuz kaymasi sessizlesir ve test kirmizi olur.
    """
    src = (TOOLS / tool).read_text(encoding="utf-8")
    assert "check_training_pool_counts(" in src, (
        f"{tool}: egitim havuzu kapisi YOK -- `build_training_frame()` "
        "yalniz KAYNAK whitelist'ini uygular, 611/585 sayacini DEGIL."
    )


@pytest.mark.parametrize("tool", GATED_TOOLS)
def test_gate_call_passes_regions_and_identity(tool: str) -> None:
    """Kapi `regions=` VE `dropped_patient_ids=` ile cagrilmali.

    `regions` olmadan kapi bolge-kor olur: WT+TC kollari (v2c/v3c, havuz
    609/583) 611/585 beklentisiyle YANLISLIKLA kirmizi olurdu.
    `dropped_patient_ids` olmadan beyan-edilmis-azalma dalinda KIMLIK
    dogrulanmaz; `check_training_pool_counts()` bunu zaten sert reddeder
    (2026-08-19 Codex HIGH #1 fail-open duzeltmesi), ama cagri bicimini
    burada da kilitliyoruz.
    """
    src = (TOOLS / tool).read_text(encoding="utf-8")
    body = _extract_call(src, "check_training_pool_counts(")
    assert body is not None, f"{tool}: cagri ayristirilamadi"
    assert "regions=" in body, f"{tool}: kapi `regions=` almiyor (bolge-kor)"
    assert "dropped_patient_ids=" in body, (
        f"{tool}: kapi `dropped_patient_ids=` almiyor (kimlik kaniti yok)"
    )
    assert "allow_mismatch=True" not in body, (
        f"{tool}: kapi fail-OPEN -- `allow_mismatch=True` kabul edilemez"
    )


# --------------------------------------------------------------------------
# 2. ANLAMSAL: WT-only kollar (v1 kalibrasyonu, k14, v1/v2a/v2b/v3a/v3b)
# --------------------------------------------------------------------------
def test_wt_only_611_585_passes() -> None:
    w3.check_training_pool_counts(
        611, 585, expected_patients=611, expected_events=585,
        allow_mismatch=False, regions=("WT",), dropped_patient_ids=[],
    )


@pytest.mark.parametrize("n,e", [(610, 584), (609, 583), (611, 584), (630, 603)])
def test_wt_only_rejects_any_drift(n: int, e: int) -> None:
    """WT-only'de beyan edilmis azalma YOK -- 609/583 dahil her sapma DURUR.

    Ozellikle 609/583: o sayi WT+TC kollarinin BEYAN EDILMIS havuzudur
    (`ZORUNLU-BEYANLAR.md` B7); WT-only bir kolda gorulmesi bir HATADIR
    ve sessizce gecmemelidir.
    """
    with pytest.raises(w3.TrainingPoolCountMismatchError):
        w3.check_training_pool_counts(
            n, e, expected_patients=611, expected_events=585,
            allow_mismatch=False, regions=("WT",), dropped_patient_ids=[],
        )


# --------------------------------------------------------------------------
# 3. ANLAMSAL: WT+TC kollari (v2c -- eem icinde; v3c -- kendi scripti)
# --------------------------------------------------------------------------
def test_v3c_wttc_609_583_passes_with_base_expectation() -> None:
    """609/583, kapiya TABAN 611/585 gecilerek gecer (bolge-farkinda dal).

    v3c/v2c icin `expected_*`'a 609/583 YAZMAK yanlistir (asagidaki
    teste bak) -- dogru bicim TABAN 611/585 + `regions=("WT","TC")`.
    """
    w3.check_training_pool_counts(
        609, 583, expected_patients=611, expected_events=585,
        allow_mismatch=False, regions=("WT", "TC"),
        dropped_patient_ids=WTTC_DROPPED,
    )


def test_v3c_wttc_rejects_wrong_dropped_identity() -> None:
    """Sayi 609/583 tutsa bile BASKA hastalar dustuyse kapi DURUR."""
    with pytest.raises(w3.TrainingPoolCountMismatchError):
        w3.check_training_pool_counts(
            609, 583, expected_patients=611, expected_events=585,
            allow_mismatch=False, regions=("WT", "TC"),
            dropped_patient_ids=["UPENN-GBM-00354", "UPENN-GBM-99999"],
        )


def test_v3c_hardcoded_609_583_would_skip_identity_check() -> None:
    """`expected_*`'a DOGRUDAN 609/583 yazmanin neden DAHA ZAYIF oldugu.

    Bu test bir REGRESYON testi degil, bir GEREKCE testidir: dogrudan
    609/583 beklenirse kapi en ustteki esitlikten ERKEN doner ve
    "ayni sayida ama BASKA hastalar dustu" sinifi HIC kontrol edilmez --
    YANLIS kimlikle bile GECER. Bu yuzden dort aracin dordu de TABAN
    611/585 + `regions=` gecirir.
    """
    w3.check_training_pool_counts(  # hatali kimlige ragmen GECIYOR
        609, 583, expected_patients=609, expected_events=583,
        allow_mismatch=False, regions=("WT", "TC"),
        dropped_patient_ids=["UPENN-GBM-99999", "UPENN-GBM-88888"],
    )


def test_wttc_rejects_undeclared_shortfall() -> None:
    """WT+TC'de beyan edilenden BASKA bir azalma (608/582) DURUR."""
    with pytest.raises(w3.TrainingPoolCountMismatchError):
        w3.check_training_pool_counts(
            608, 582, expected_patients=611, expected_events=585,
            allow_mismatch=False, regions=("WT", "TC"),
            dropped_patient_ids=WTTC_DROPPED + ["UPENN-GBM-00001"],
        )


def test_declared_shortfall_table_is_the_locked_609_583() -> None:
    """`ZORUNLU-BEYANLAR.md` B7 ile kod BIREBIR tutuyor mu."""
    shortfall = w3.DECLARED_REGION_SHORTFALLS[("WT", "TC")]
    assert (shortfall.n_patients, shortfall.n_events) == (609, 583)
    assert sorted(shortfall.dropped_patient_ids) == WTTC_DROPPED


# --------------------------------------------------------------------------
# 4. ComBat OZDESLIK GUARD'I -- `check_combat_identity=True` PIN'i
#    (2026-09-14, modeling-agent-T1, Baris onayi: "yapalim zaten maliyeti yok")
# --------------------------------------------------------------------------
# NEDEN AYRI BIR PIN: `build_training_frame()` docstring'i
# `check_combat_identity=False`'i AÇIKCA sinirliyor -- "SADECE ComBat'a hic
# bagimli olmayan/onu bilincli bypass eden (orn. `tools/smoke_test_cox_model.py`
# tarzi mekanik test) senaryolar icin var". Bu dort arac o kategoride DEGIL:
# rapora giren sayilari (Uno's C, td-AUC, Brier/IPA, bootstrap CI) uretiyorlar.
# CLAUDE.md: "ComBat, OS-Cox birincil zincirinde OZDESLIK donusumudur ... AMA
# KIRILGAN: LUMIERE egitime girerse veya `reference_batch` degisirse asimetri
# dogar -> Cox egitiminde kod-seviyesi guard ZORUNLU."
#
# SAYI ETKISI YOK (canli DB ile OLCULDU, 2026-09-14, degisiklik oncesi VE
# sonrasi, 9 cagri noktasi): False/True ikisinde de AYNI havuz, AYNI olay
# sayisi, AYNI dusen kimlikler ve BIREBIR ayni cerceve degerleri --
# WT-only 611/585 (dusen 0), WT+TC 609/583 (dusen UPENN-GBM-00354/00397).
# `assert_combat_identity_for_training_pool()` geciyor cunku egitim
# havuzunun `source` kolonu TEK degerli: "UPenn-GBM".
#
# Bu testler DB GEREKTIRMEZ -- kaynak kodu AST ile ayristirirlar.
import ast  # noqa: E402


def _build_training_frame_calls(tool: str) -> list[ast.Call]:
    """`build_training_frame(...)` cagrilarini AST ile bul (yorum/docstring HARIC).

    Metin aramasi yetmez: ayni ad yorumlarda ve docstring'lerde de geciyor.
    `eem.build_training_frame(...)` (v3c) ve duz `build_training_frame(...)`
    (digerleri) bicimlerinin IKISI de yakalanir.
    """
    tree = ast.parse((TOOLS / tool).read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == "build_training_frame":
            calls.append(node)
    return calls


@pytest.mark.parametrize("tool", GATED_TOOLS)
def test_tool_has_exactly_one_build_training_frame_call(tool: str) -> None:
    """Her arac egitim cercevesini TEK yerde kuruyor (pin'in kapsam kaniti).

    Ikinci bir cagri eklenirse bu test kirmizi olur ve asagidaki pin
    testinin o cagriyi da kapsadigini gozden gecirmeye zorlar.
    """
    calls = _build_training_frame_calls(tool)
    assert len(calls) == 1, (
        f"{tool}: {len(calls)} adet `build_training_frame(...)` cagrisi var "
        "(beklenen 1) -- ComBat ozdeslik pin'inin kapsamini gozden gecir."
    )


@pytest.mark.parametrize("tool", GATED_TOOLS)
def test_build_training_frame_pins_combat_identity_true(tool: str) -> None:
    """`check_combat_identity` ACIKCA `True` gecilmeli -- `False` YASAK.

    Biri ileride bu bayragi sessizce `False`'a cevirirse (veya argumani
    tamamen silerse) bu test KIRMIZI yanar. Argumanin silinmesi de
    reddediliyor: varsayilan zaten `True` ama bu dort arac icin niyetin
    ACIK olmasi isteniyor -- "varsayilan degisirse sessizce kayar" sinifi
    kapatiliyor.
    """
    for call in _build_training_frame_calls(tool):
        kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        assert "check_combat_identity" in kwargs, (
            f"{tool}: `build_training_frame(...)` cagrisinda "
            "`check_combat_identity` ACIKCA gecilmemis -- varsayilana "
            "guvenilmez, `True` yazilmali."
        )
        value = kwargs["check_combat_identity"]
        assert isinstance(value, ast.Constant) and value.value is True, (
            f"{tool}: `check_combat_identity` sabit `True` DEGIL "
            f"(bulunan: {ast.dump(value)}). `False` yalniz ComBat'a hic "
            "bagimli olmayan mekanik testler icindir (bkz. "
            "`build_training_frame()` docstring'i); bu arac rapora giren "
            "sayilari uretiyor."
        )


@pytest.mark.parametrize("tool", GATED_TOOLS)
def test_no_literal_check_combat_identity_false_in_source(tool: str) -> None:
    """Metin duzeyinde ikinci kemer: `check_combat_identity=False` HIC gecmemeli.

    AST testi cagri-bazli; bu test dosyanin TAMAMINI (ornek kod parcasi,
    yorum icindeki kopyala-yapistir oneri vb. dahil) tarar -- boylece
    "yoruma yazip sonra acan" yol da gorunur olur.
    """
    src = (TOOLS / tool).read_text(encoding="utf-8")
    normalized = src.replace(" ", "")
    assert "check_combat_identity=False" not in normalized, (
        f"{tool}: kaynakta `check_combat_identity=False` gecti -- ComBat "
        "ozdeslik guard'i kapatilamaz (2026-09-14 pin'i)."
    )
