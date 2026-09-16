"""`--reduce-collinearity` REÇETE TUTARLILIĞI lint testi (AST tabanlı).

NEDEN VAR (2026-08-28, Codex stop-time incelemesinin bulgusu):
--------------------------------------------------------------
Bulgu birebir: *"`--reduce-collinearity` eğitim ve harici skorlamada
farklı Cox reçeteleri üretiyor."* Doğrulandı ve gerçekti.

`tools/train_xgboost_week4.py::run_ucsf_external_evaluation()` içinde
İKİ ayrı Cox yolu var:

  1. `fit_final_cox_and_xgboost_pipeline_on_full_pool(..., reduce_
     collinearity=args.reduce_collinearity, ...)`
     -> XGBoost'un meta-skorlarını üreten cross-fit Cox fit'leri.
        Bayrak açıkken aday havuzu FİLTRELENİR.

  2. `cox_final = week3.fit_final_model_on_full_pool(...)`
     -> SADECE `evaluate_xgboost_external_test()`'e, yani GERÇEK
        UCSF/üretim skorlamasına verilen dağıtım modeli.

Yol 2, bayrağı hiç görmüyordu ve FİLTRELENMEMİŞ `feature_columns` ile
fit ediliyordu. Sonuç: `--reduce-collinearity` açıkken XGBoost
**filtreli** Cox skorlarıyla EĞİTİLİP **filtresiz** Cox skorlarıyla
TEST ediliyordu -- klasik train/serve skew. Harici AUC bu durumda
modelin gerçek üretim davranışını ÖLÇMEZ.

Bu test, düzeltmenin sessizce geri alınmasını engeller. Kaynak kodu
AST ile denetler; DB veya uzun koşu GEREKTİRMEZ.

NASIL DÜZELTİLİR (test kırmızıysa):
------------------------------------
`cox_final = week3.fit_final_model_on_full_pool(...)` çağrısı ham
`feature_columns` DEĞİL, bayrağa göre filtrelenmiş
`cox_final_feature_columns` almalıdır; filtre `build_v3_candidate_pool()`
ile AYNI eşiklerden (`args.collinearity_cv_threshold`,
`args.collinearity_corr_threshold`) hesaplanmalıdır.

⚠️ 2026-09-11 GÜNCELLEME (Ş2 / K16-a, Codex şartlı onay): `run_ucsf_
external_evaluation()` artık `week3.fit_final_model_on_full_pool()`'u
DOĞRUDAN çağırmıyor -- final Cox fit'i ConvergenceError/LinAlgError'a
karşı penalizer-eskalaması yapan `_fit_final_model_on_full_pool_with_
penalizer_escalation()` (AYNI dosyada, `train_cox_week3.py`'ye TEK SATIR
yazılmadı) sarmalayıcısı ARAYA GİRDİ. Bu test artık İKİ ZINCIRLI ADIMI
denetler: (1) `run_ucsf_external_evaluation()` sarmalayıcıyı FİLTRELİ
`cox_final_feature_columns` ile çağırıyor mu, (2) sarmalayıcının KENDİSİ
o parametreyi DEĞİŞTİRMEDEN `week3.fit_final_model_on_full_pool()`'a
iletiyor mu -- ikisi birlikte ESKİ (tek adımlı) korumanın AYNI garantisini
verir.

NOT: `Path(__file__).resolve()` yerine `.absolute()` -- K12'de kayıtlı
Türkçe-yol/`subst` tuzağı (bkz. `tests/test_no_resolve_path_regression.py`).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).absolute().parent.parent
_TARGET = _REPO_ROOT / "tools" / "train_xgboost_week4.py"

_FONKSIYON = "run_ucsf_external_evaluation"
_SARMALAYICI_FONKSIYON = "_fit_final_model_on_full_pool_with_penalizer_escalation"
_FILTRELI_AD = "cox_final_feature_columns"
_HAM_AD = "feature_columns"


def _module_ast() -> ast.Module:
    if not _TARGET.is_file():
        pytest.fail(f"Denetlenecek dosya bulunamadi: {_TARGET}")
    return ast.parse(_TARGET.read_text(encoding="utf-8"))


def _fonksiyon_bul(tree: ast.Module, isim: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == isim:
            return node
    pytest.fail(
        f"`{isim}()` bulunamadi -- fonksiyon yeniden adlandirildiysa "
        "bu testin hedefi de guncellenmelidir (testi SILME, hedefi duzelt)."
    )


def _hedef_fonksiyon() -> ast.FunctionDef:
    return _fonksiyon_bul(_module_ast(), _FONKSIYON)


def _cagri_bul(fn: ast.FunctionDef, attr_adi: str) -> ast.Call:
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == attr_adi:
            return node
        if isinstance(f, ast.Name) and f.id == attr_adi:
            return node
    pytest.fail(
        f"`{fn.name}()` icinde `{attr_adi}(...)` cagrisi bulunamadi. Harici "
        "skorlama modeli baska bir yoldan uretiliyorsa recete tutarliligi "
        "YENIDEN degerlendirilmelidir."
    )


def _fit_final_cagrisi(fn: ast.FunctionDef) -> ast.Call:
    return _cagri_bul(fn, "fit_final_model_on_full_pool")


def test_harici_skorlama_cox_modeli_filtrelenmis_havuz_aliyor() -> None:
    """ASIL KORUMA (2026-09-11 GÜNCELLENDİ -- iki-zincirli): dağıtım Cox'u
    NE `run_ucsf_external_evaluation()` seviyesinde NE de eskalasyon
    sarmalayıcısının İÇİNDE ham `feature_columns` ALMAMALI."""

    tree = _module_ast()
    ucsf_fn = _fonksiyon_bul(tree, _FONKSIYON)
    wrapper_fn = _fonksiyon_bul(tree, _SARMALAYICI_FONKSIYON)

    # ADIM 1: run_ucsf_external_evaluation() sarmalayiciyi FILTRELI
    # `cox_final_feature_columns` ile cagiriyor mu?
    outer_call = _cagri_bul(ucsf_fn, _SARMALAYICI_FONKSIYON)
    assert len(outer_call.args) >= 2, (
        f"`{_SARMALAYICI_FONKSIYON}()` cagrisi beklenen pozisyonel "
        "argumanlari tasimiyor; recete tutarliligi dogrulanamadi."
    )
    outer_aday_arg = outer_call.args[1]
    assert isinstance(outer_aday_arg, ast.Name), (
        "Aday havuzu argumani basit bir degisken adi degil; bu test "
        "guncellenmeli (sessizce gecmesine izin verme)."
    )
    assert outer_aday_arg.id != _HAM_AD, (
        "🔴 RECETE TUTARSIZLIGI GERI GELDI.\n"
        f"`{_FONKSIYON}()` icindeki dagitim Cox modeli ham `{_HAM_AD}` "
        "ile cagriliyor; oysa XGBoost'un meta-skorlari "
        "`reduce_collinearity` FILTRELI havuzdan uretiliyor.\n"
        f"DUZELTME: `{_FILTRELI_AD}` gec (bayraga gore filtrelenmis liste)."
    )
    assert outer_aday_arg.id == _FILTRELI_AD, (
        f"Beklenen `{_FILTRELI_AD}`, bulunan `{outer_aday_arg.id}` -- "
        "adlandirma degistiyse bu testin sabiti de bilincli olarak "
        "guncellenmelidir."
    )

    # ADIM 2: sarmalayicinin KENDISI o parametreyi DEGISTIRMEDEN
    # `week3.fit_final_model_on_full_pool()`'a iletiyor mu? Sarmalayicinin
    # KENDI parametre adi da `cox_final_feature_columns` OLMALI (outer
    # cagridan gelen degerin AYNI isimle -- yeniden-turetme/filtreleme
    # OLMADAN -- ictekine ulastigini AD ile kanitlar).
    wrapper_arg_names = [arg.arg for arg in wrapper_fn.args.args]
    assert _FILTRELI_AD in wrapper_arg_names, (
        f"`{_SARMALAYICI_FONKSIYON}()` parametre listesinde `{_FILTRELI_AD}` "
        "YOK -- ikinci pozisyonel argumanin GERCEKTEN filtreli havuz "
        "oldugu adlandirma yoluyla dogrulanamiyor."
    )
    inner_call = _fit_final_cagrisi(wrapper_fn)
    assert len(inner_call.args) >= 2, (
        "`fit_final_model_on_full_pool()` ic cagrisi beklenen pozisyonel "
        "argumanlari tasimiyor; recete tutarliligi dogrulanamadi."
    )
    inner_aday_arg = inner_call.args[1]
    assert isinstance(inner_aday_arg, ast.Name) and inner_aday_arg.id == _FILTRELI_AD, (
        "🔴 RECETE TUTARSIZLIGI (sarmalayici katmaninda): "
        f"`{_SARMALAYICI_FONKSIYON}()` icindeki `fit_final_model_on_full_"
        f"pool()` cagrisi `{_FILTRELI_AD}` parametresini DEGISTIRMEDEN "
        f"iletmiyor (bulunan: {getattr(inner_aday_arg, 'id', ast.dump(inner_aday_arg))})."
    )


def test_filtre_ayni_esiklerle_ve_bayraga_bagli_hesaplaniyor() -> None:
    """Filtre gerçekten `args.reduce_collinearity`'ye bağlı mı ve
    cross-fit yoluyla AYNI eşikleri mi kullanıyor?

    Aksi hâlde 'filtreli' isim altında BAŞKA bir reçete üretilebilir --
    isim tutarlılığı tek başına yetmez.
    """
    fn = _hedef_fonksiyon()
    kaynak = ast.unparse(fn)

    assert "args.reduce_collinearity" in kaynak, (
        "Dagitim Cox'unun aday havuzu `args.reduce_collinearity` bayragina "
        "BAGLI DEGIL -- filtre kosulsuz uygulaniyorsa varsayilan davranis "
        "degismis demektir (7 kolun bit-birebir yeniden uretilebilirligi "
        "bozulur)."
    )
    assert "build_v3_candidate_pool" in kaynak, (
        "Filtre `build_v3_candidate_pool()` ile hesaplanmiyor -- cross-fit "
        "yoluyla AYNI fonksiyon kullanilmalidir."
    )
    for esik in ("args.collinearity_cv_threshold", "args.collinearity_corr_threshold"):
        assert esik in kaynak, (
            f"`{esik}` dagitim Cox'unun filtresine gecirilmiyor -- iki yol "
            "FARKLI esiklerle farkli recete uretebilir."
        )


def test_bos_havuz_sessizce_gecilmiyor() -> None:
    """Filtre her şeyi elerse fail-loud olmalı (sessiz bos havuz yasak)."""
    kaynak = ast.unparse(_hedef_fonksiyon())
    assert "RuntimeError" in kaynak, (
        "Filtre sonrasi havuz bosalirsa `RuntimeError` ile durulmali; "
        "sessizce bos/ham havuza dusmek fail-open'dir."
    )
