"""DRIFT GUARD'I: `api/predict.py` <-> `tools/data_integrity_check.py`
IDH1/MGMT ham-etiket normalizasyon haritalari BIREBIR AYNI kalmali.

## Neden bu test var (2026-09-13, karar 35 -- backend-agent-K)

2026-09-13'te LUMIERE'nin kanonik-olmayan ham etiketlerini (`"WT"`,
`"wt"`, `"R132H mut"`, `"IDH1 neg, Sequencing required"`, `"methylated"`,
`"not methylated"`) okuma-zamaninda kanonik forma ceviren iki sozluk
eklendi (`decisions/2026-09-13-lumiere-idh-mgmt-etiket-normalizasyonu.md`):

  * OTORITE (gercek servis davranisini belirleyen):
    `api/predict.py::IDH1_LABEL_NORMALIZATION_MAP` /
    `MGMT_LABEL_NORMALIZATION_MAP`
  * AYNA (yalniz denetim/rapor baglami, hicbir veri DONUSTURMEZ):
    `tools/data_integrity_check.py::KNOWN_NORMALIZED_IDH1_RAW_VALUES` /
    `KNOWN_NORMALIZED_MGMT_RAW_VALUES`

Ayna, `api/` paketini (FastAPI/shap/lifelines) denetim aracina CEKMEMEK
icin **BILINCLI olarak** elle senkronize edilen bir KOPYADIR -- bu tasarim
karari `data_integrity_check.py`'nin kendi yorumunda (satir ~757-769)
yazili ve bu test onu DEGISTIRMEZ. Iki sozluk 2026-09-13'te birebir ayni
olcuKldu, AMA esitligi koruyan HICBIR test YOKTU: biri guncellenip digeri
unutulursa `check_domain_values()` "bilinen/normalize-edilen" ile
"GERCEKTEN yeni/kapsam-disi" degerleri ayirt edemez ve LUMIERE'nin ham
etiketleri denetim raporunda SESSIZCE "yeni kapsam-disi deger" gibi
gorunur (ya da tersi: gercek bir bozuk deger "bilinen" sayilip yutulur).
Bu test o sessiz drift'i imkansiz kilar.

## NASIL denetler -- AST, IMPORT DEGIL

`tools/data_integrity_check.py` **IMPORT EDILMEZ** (bilincli: o dosyanin
`api/` bagimliligi OLMAMASI bir tasarim karari; import etmek testi de
FastAPI/shap'a baglardi ve ayna dosyasinin bagimsizligini kagit uzerinde
birakirdi). Iki dosya da `ast.parse()` ile KAYNAK OLARAK okunur, modul
seviyesindeki sozluk literalleri cikarilir ve karsilastirilir.

`api/predict.py` tarafinda sozluk DEGERLERI sabit ISIMLERDIR
(`IDH1_WILDTYPE_LABEL` gibi) -- bu yuzden once modul seviyesindeki
`ISIM = "duz string"` atamalari toplanir ve degerler o tablodan cozulur.
Cozulemeyen bir deger SESSIZCE atlanmaz, `AssertionError` firlatir.

## Bu test VAKUM DEGIL

`test_mirror_drift_guard_is_red_when_a_single_value_drifts` ve
`test_mirror_drift_guard_is_red_when_a_key_is_removed` ayni cikarim
fonksiyonunu, gecici bir dizine yazilmis DEGISTIRILMIS kaynak uzerinde
kosar ve karsilastirmanin GERCEKTEN kirmizi oldugunu kanitlar (tautoloji
kontrolu -- gorev talimatinin ZORUNLU sarti). GERCEK dosyalar bu testler
sirasinda DEGISTIRILMEZ; kopya `tmp_path` altinda olusturulur.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# `.resolve()` -- proje/repo kokune cikmak icin (bkz. api/predict.py'deki
# ayni gerekce: bu dosya goruntu YAZMAZ, K12 ITK kisiti gecerli degil).
REPO_ROOT = Path(__file__).resolve().parent.parent  # "gbm-aid mert"

PREDICT_SOURCE_PATH = REPO_ROOT / "api" / "predict.py"
INTEGRITY_SOURCE_PATH = REPO_ROOT / "tools" / "data_integrity_check.py"

PREDICT_IDH1_MAP_NAME = "IDH1_LABEL_NORMALIZATION_MAP"
PREDICT_MGMT_MAP_NAME = "MGMT_LABEL_NORMALIZATION_MAP"
MIRROR_IDH1_MAP_NAME = "KNOWN_NORMALIZED_IDH1_RAW_VALUES"
MIRROR_MGMT_MAP_NAME = "KNOWN_NORMALIZED_MGMT_RAW_VALUES"

# 2026-09-13 canli olcum (her iki dosyada AYNI) -- sozluklerin BOS/kucuk
# olmadigini kanitlayan bagimsiz bir cipa. Icerik bilincli olarak
# degisirse (yeni bir ham etiket kurtarilirsa) ONCE karar dosyasi
# guncellenir, SONRA bu sayilar ve iki sozluk BIRLIKTE degisir.
EXPECTED_IDH1_MAP_SIZE = 4
EXPECTED_MGMT_MAP_SIZE = 2


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    """Modul seviyesindeki `ISIM = "duz string"` atamalarini toplar."""

    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = node.value.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str) and isinstance(node.target, ast.Name):
                constants[node.target.id] = node.value.value
    return constants


def _extract_str_dict(source_path: Path, dict_name: str) -> dict[str, str]:
    """`source_path`'in AST'sinden `dict_name` adli modul-seviyesi
    sozluk literalini `dict[str, str]` olarak cikarir.

    Sozluk DEGERLERI ya duz string literali ya da AYNI modulde tanimli bir
    string sabitinin ADI olabilir (`api/predict.py` deseni). Bunlarin
    DISINDA bir sey (f-string, cagri, baska bir modulden gelen isim...)
    gorulurse SESSIZCE atlanmaz -- `AssertionError` firlatilir. Hedef
    sozluk hic bulunamazsa da HATA (drift'in en sinsi hali: sozlugun adi
    degistirilmis/silinmis olabilir).
    """

    assert source_path.is_file(), f"Kaynak dosya yok: {source_path}"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    constants = _module_string_constants(tree)

    found: ast.Dict | None = None
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == dict_name:
                assert isinstance(node.value, ast.Dict), (
                    f"{source_path.name}::{dict_name} bir SOZLUK LITERALI degil "
                    f"({type(node.value).__name__}) -- bu test onu AST ile "
                    "okuyamaz, drift guard'i SESSIZCE ise yaramaz hale gelirdi."
                )
                found = node.value

    assert found is not None, (
        f"{source_path.name} icinde modul seviyesinde `{dict_name}` sozlugu "
        "BULUNAMADI. Ad degistiyse/silindiyse bu bir DRIFT'tir: otorite "
        "(api/predict.py) ile ayna (tools/data_integrity_check.py) arasindaki "
        "esitlik ARTIK denetlenemez -- degisikligi yapan her iki tarafi da "
        "guncellemek zorundadir."
    )

    extracted: dict[str, str] = {}
    for key_node, value_node in zip(found.keys, found.values):
        assert isinstance(key_node, ast.Constant) and isinstance(key_node.value, str), (
            f"{source_path.name}::{dict_name} icinde string OLMAYAN bir anahtar var: "
            f"{ast.dump(key_node) if key_node is not None else None}"
        )
        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
            extracted[key_node.value] = value_node.value
        elif isinstance(value_node, ast.Name):
            assert value_node.id in constants, (
                f"{source_path.name}::{dict_name}[{key_node.value!r}] degeri "
                f"`{value_node.id}` adli bir isim ama o isim AYNI modulde bir "
                "duz-string sabiti olarak BULUNAMADI -- deger cozulemedi, "
                "sessizce atlanmiyor."
            )
            extracted[key_node.value] = constants[value_node.id]
        else:  # pragma: no cover -- bugun bu desenlerin ikisi de yeterli
            raise AssertionError(
                f"{source_path.name}::{dict_name}[{key_node.value!r}] degeri "
                f"beklenmeyen bir AST dugumu ({type(value_node).__name__}) -- "
                "cozulemedi, bu testin cikarimi guncellenmelidir."
            )

    assert len(extracted) == len(found.keys), (
        f"{source_path.name}::{dict_name} icinde TEKRARLANAN anahtar var "
        f"({len(found.keys)} giris -> {len(extracted)} tekil anahtar)."
    )
    return extracted


def _mirror_pairs_from_sources(
    predict_path: Path, integrity_path: Path
) -> list[tuple[str, dict[str, str], dict[str, str]]]:
    """Karsilastirilacak (etiket, otorite, ayna) uclulerini dondurur."""

    return [
        (
            "IDH1",
            _extract_str_dict(predict_path, PREDICT_IDH1_MAP_NAME),
            _extract_str_dict(integrity_path, MIRROR_IDH1_MAP_NAME),
        ),
        (
            "MGMT",
            _extract_str_dict(predict_path, PREDICT_MGMT_MAP_NAME),
            _extract_str_dict(integrity_path, MIRROR_MGMT_MAP_NAME),
        ),
    ]


def _drift_report(pairs: list[tuple[str, dict[str, str], dict[str, str]]]) -> list[str]:
    """Esit OLMAYAN her cift icin insan-okunur bir fark satiri dondurur.
    Bos liste == DRIFT YOK."""

    problems: list[str] = []
    for label, authority, mirror in pairs:
        if authority == mirror:
            continue
        only_authority = {k: v for k, v in authority.items() if mirror.get(k) != v}
        only_mirror = {k: v for k, v in mirror.items() if authority.get(k) != v}
        problems.append(
            f"{label}: api/predict.py'de olup aynada AYNI OLMAYAN {only_authority}; "
            f"aynada olup otoritede AYNI OLMAYAN {only_mirror}"
        )
    return problems


# =====================================================================
# 1) ASIL ESITLIK TESTI (gercek dosyalar)
# =====================================================================


def test_normalization_maps_are_identical_in_authority_and_mirror() -> None:
    pairs = _mirror_pairs_from_sources(PREDICT_SOURCE_PATH, INTEGRITY_SOURCE_PATH)
    problems = _drift_report(pairs)
    assert not problems, (
        "NORMALIZASYON HARITASI DRIFT'I -- `api/predict.py` (OTORITE) ile "
        "`tools/data_integrity_check.py` (AYNA) ARTIK AYNI DEGIL:\n  "
        + "\n  ".join(problems)
        + "\nIkisi ELLE senkronize edilen bilincli kopyalardir (ayna `api/`yi "
        "import ETMEZ, bu bir tasarim karari) -- birini degistiren DIGERINI de "
        "degistirmek zorundadir. Bkz. decisions/2026-09-13-lumiere-idh-mgmt-"
        "etiket-normalizasyonu.md"
    )


def test_normalization_maps_are_non_empty_and_expected_size() -> None:
    """Esitlik testinin "bos == bos" ile GECMESINI imkansiz kilar."""

    idh1 = _extract_str_dict(PREDICT_SOURCE_PATH, PREDICT_IDH1_MAP_NAME)
    mgmt = _extract_str_dict(PREDICT_SOURCE_PATH, PREDICT_MGMT_MAP_NAME)
    assert len(idh1) == EXPECTED_IDH1_MAP_SIZE, idh1
    assert len(mgmt) == EXPECTED_MGMT_MAP_SIZE, mgmt
    # Karar dosyasinin ACIKCA vurguladigi esleme: "sequencing required"
    # `Wildtype`'a DEGIL `NOS/NEC`'e gider (epistemik durum korunur).
    assert idh1["IDH1 neg, Sequencing required"] == "NOS/NEC"
    assert idh1["WT"] == "Wildtype"
    assert mgmt["not methylated"] == "Unmethylated"


def test_ast_extraction_matches_runtime_dicts() -> None:
    """AST cikariminin DOGRU seyi okudugunu kanitlar: `api/predict.py`
    IMPORT edilip calisma-zamani sozlukleriyle karsilastirilir (otorite
    tarafini import etmek serbest -- yasak olan `tools/`u import etmekti).
    Bu, cikarim fonksiyonu bozulup her yerde bos/yanlis sozluk dondurse
    esitlik testinin sessizce yesil kalmasini ONLER."""

    import api.predict as predict_module

    assert _extract_str_dict(PREDICT_SOURCE_PATH, PREDICT_IDH1_MAP_NAME) == dict(
        predict_module.IDH1_LABEL_NORMALIZATION_MAP
    )
    assert _extract_str_dict(PREDICT_SOURCE_PATH, PREDICT_MGMT_MAP_NAME) == dict(
        predict_module.MGMT_LABEL_NORMALIZATION_MAP
    )


# =====================================================================
# 2) TAUTOLOJI KONTROLU -- guard'in KIRMIZI olabildigini kanitla
# =====================================================================


def _write_mutated_copy(tmp_path: Path, source_path: Path, old: str, new: str) -> Path:
    text = source_path.read_text(encoding="utf-8")
    assert old in text, f"Beklenen metin bulunamadi (test bakimi gerekli): {old!r}"
    target = tmp_path / source_path.name
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    return target


def test_mirror_drift_guard_is_red_when_a_single_value_drifts(tmp_path: Path) -> None:
    """**ZORUNLU KIRMIZI TEST**: aynadaki TEK bir eslemenin degeri
    degisirse karsilastirma KIRMIZI olmali. GERCEK dosya DEGISTIRILMEZ --
    `tmp_path` altina degistirilmis bir KOPYA yazilir."""

    mutated = _write_mutated_copy(
        tmp_path,
        INTEGRITY_SOURCE_PATH,
        '"IDH1 neg, Sequencing required": "NOS/NEC"',
        '"IDH1 neg, Sequencing required": "Wildtype"',  # karar dosyasinin YASAKLADIGI esleme
    )
    problems = _drift_report(_mirror_pairs_from_sources(PREDICT_SOURCE_PATH, mutated))
    assert problems, (
        "Guard VAKUM: ayna sozlugunun bir DEGERI degistirildigi halde fark "
        "raporlanmadi -- bu test yanlis seyi olcuyor demektir."
    )
    assert "IDH1" in problems[0]
    assert "Wildtype" in problems[0] or "NOS/NEC" in problems[0]


def test_mirror_drift_guard_is_red_when_a_key_is_removed(tmp_path: Path) -> None:
    """Ikinci kirmizi senaryo: aynadan bir ANAHTAR dusurulurse (ornek:
    MGMT `"not methylated"` unutulursa) guard yine KIRMIZI olmali."""

    mutated = _write_mutated_copy(
        tmp_path,
        INTEGRITY_SOURCE_PATH,
        '"methylated": "Methylated", "not methylated": "Unmethylated",',
        '"methylated": "Methylated",',
    )
    problems = _drift_report(_mirror_pairs_from_sources(PREDICT_SOURCE_PATH, mutated))
    assert problems, "Guard VAKUM: aynadan anahtar dusuruldu ama fark raporlanmadi."
    assert "MGMT" in problems[0]


def test_extractor_is_red_when_dict_is_renamed_or_deleted(tmp_path: Path) -> None:
    """Drift'in en sinsi hali: sozluk YENIDEN ADLANDIRILIRSA cikarim
    SESSIZCE bos donmemeli, ACIK hata vermeli."""

    mutated = _write_mutated_copy(
        tmp_path,
        INTEGRITY_SOURCE_PATH,
        "KNOWN_NORMALIZED_IDH1_RAW_VALUES = {",
        "RENAMED_IDH1_RAW_VALUES = {",
    )
    with pytest.raises(AssertionError, match="BULUNAMADI"):
        _extract_str_dict(mutated, MIRROR_IDH1_MAP_NAME)
