"""``GET /model_curves`` -- Sistem > Model Performansi ekranindaki IKI grafik
(kalibrasyon egrisi + katsayi orman grafigi) icin veri kaynagi
(backend-agent-Y1, paralel-40, 2026-09-16, Baris onayi).

TEK DOGRULUK KAYNAGI ARTEFAKTTIR -- bu modul hicbir sayiyi SABIT
GOMMEZ/YUVARLAMAZ, hepsi asagidaki 3 dosyadan HAM olarak okunur
(`api/model_performance.py`'nin desenini takip eder):

  1. `artifacts/week3/cox_model/week3_v3b_calibration_365.csv`
     (10 desil: n/olay/tahmin/gozlenen KM + KM %95 GA)
  2. `artifacts/week3/cox_model/week3_v3b_calibration_summary.csv`
     (kalibrasyon egimi + decile_mae(365), ikisi de %95 GA/kaynak notuyla)
  3. `artifacts/week3/cox_model/week3_v3b_lowvar_v2amgmt_final_coefficients.csv`
     (nihai model v3b_lowvar_v2amgmt'nin 24 kovaryati: 16 radyomik [`WT__`
     onekli] + 8 klinik -- coef/exp(coef)/se + %95 GA + z/p)

Bu dosyalarin HICBIRI bu modulde YENIDEN URETILMEZ (tools/**, artifacts/**
SALT OKUNUR, gorev kisiti) -- yalniz OKUNUR ve JSON'a serilestirilir.

⚠️ Gercek SHAP beeswarm'i BILEREK YAPILMIYOR (611 hasta icin SHAP hesabi
yeni bir kosu ister, saatler surer) -- bunun yerine zaten olculmus ve
kilitli katsayilar (coef + %95 GA + hazard orani) kullaniliyor; Cox modeli
icin bu YENI hesap gerektirmeden bilgilendiricidir.

HATA SOZLESMESI (CLAUDE.md: sessiz basarisizlik YOK):
  - Yukaridaki 3 dosyadan HERHANGI BIRI eksik/bozuk/parse edilemiyor ->
    503, hangi dosyanin sorunlu oldugu ACIKCA govde/mesajda belirtilir.
  - Katsayi CSV'sinde radyomik+klinik toplami 24 (16+8) DEGILSE -> bu
    SESSIZCE GECILMEZ, gurultulu bir hata firlatilir (503) -- CSV'nin
    kendisi bozulmus/degismis demektir, yanlis bir grafik SUNULMAZ.

🔴 ZORUNLU BEYANLAR (jüriye gösterilecek grafikler icin, ZORUNLU-BEYANLAR.md
A9/A10'dan) -- `notes` alaninda AYNEN tasinir, sadelestirilmez:
  - Kalibrasyon egimi 0,685 [0,470-0,900]; ust ucu 1'e ULASMIYOR -> model
    uclarda asiri-guvenli. MUTLAK sagkalim olasiligi iddiasi KURULMAZ,
    yalniz SIRALAMA. "Iyi/kotu kalibre" hukmu YASAK.
  - Kalibrasyon yalniz IKI kolda (v1 + v3b) olculdu; kalan 6 kolda HIC
    olculmedi.
  - Katsayilar nihai modelin TAM-HAVUZ fit'ine aittir; nested-CV sonucu
    DEGILDIR -- karistirilmamali.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()

# `.absolute()` -- `.resolve()` DEGIL, ayni gerekce `api/model_performance.py`
# ve `api/similar.py`'deki gibi (Turkce-karakterli yol riski).
REPO_ROOT = Path(__file__).absolute().parent.parent

COX_CALIBRATION_365_CSV = (
    REPO_ROOT / "artifacts" / "week3" / "cox_model" / "week3_v3b_calibration_365.csv"
)
COX_CALIBRATION_SUMMARY_CSV = (
    REPO_ROOT / "artifacts" / "week3" / "cox_model" / "week3_v3b_calibration_summary.csv"
)
COX_V3B_COEFFICIENTS_CSV = (
    REPO_ROOT
    / "artifacts"
    / "week3"
    / "cox_model"
    / "week3_v3b_lowvar_v2amgmt_final_coefficients.csv"
)

VARIANT_NAME = "v3b_lowvar_v2amgmt"
EXPECTED_N_RADIOMIC_COVARIATES = 16
EXPECTED_N_CLINICAL_COVARIATES = 8
EXPECTED_N_COVARIATES = EXPECTED_N_RADIOMIC_COVARIATES + EXPECTED_N_CLINICAL_COVARIATES


class ModelCurvesArtifactError(RuntimeError):
    """Kaynak artefakt (CSV) eksik/bozuk VEYA icerik beklenen sozlesmeyle
    (24 = 16 radyomik + 8 klinik) uyusmuyor -- endpoint bunu 503'e cevirir,
    sessizce eksik/yanlis bir grafik DONMEZ."""


def _typed_cell(value: str) -> Any:
    """Bir CSV hucresini HAM tutarak sayisal olabiliyorsa float'a cevirir.

    Bos hucre -> `None` (eksik degeri sessizce 0/"" YAPMAZ). Sayisal
    OLMAYAN her sey (ornegin serbest metin kaynak notu) STRING olarak
    OLDUGU GIBI kalir -- `api/model_performance.py::_typed_cell` ile
    AYNI desen (kasitli tekrar, o modul DOKUNULMAZ oldugu icin import
    yerine kucuk bir kopya tercih edildi)."""

    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _as_whole_int(value: Any, *, field_name: str) -> int:
    """Bir sayisal alani (decile/n/olay gibi TAM SAYI olmasi gereken bir
    sayac) int'e cevirir. `_typed_cell` zaten float'a cevirmisti (ornegin
    `1` -> `1.0`) -- burada YUVARLAMA yok, yalniz tam-sayi oldugu
    DOGRULANIP int'e donusturuluyor (365.0 -> 365, veri kaybi yok).
    Tam sayi DEGILSE (beklenmedik durum) hata firlatir -- sessizce
    yuvarlama YAPILMAZ."""

    if not isinstance(value, float):
        raise ModelCurvesArtifactError(
            f"'{field_name}' alani sayisal degil (beklenen tam sayi): {value!r}"
        )
    if not value.is_integer():
        raise ModelCurvesArtifactError(
            f"'{field_name}' alani tam sayi degil (beklenmedik kesirli deger): {value!r}"
        )
    return int(value)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ModelCurvesArtifactError(
            f"Beklenen artefakt bulunamadi: {path} -- model_curves URETILEMEDI "
            "(sessizce kismi/bos veri DONULMEDI)."
        )
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = [
                {key: _typed_cell(value) for key, value in row.items()}
                for row in reader
            ]
    except (OSError, csv.Error) as exc:
        raise ModelCurvesArtifactError(
            f"Artefakt okunamadi/parse edilemedi: {path} ({type(exc).__name__}: {exc})"
        ) from exc
    if not rows:
        raise ModelCurvesArtifactError(f"Artefakt bos (0 satir): {path}")
    return rows


def _relative_path(path: Path) -> str:
    """Yanit govdesinde `REPO_ROOT`'a GORE yol dondurur -- Turkce-karakterli
    MUTLAK yolu istemciye sizdirmadan seffaflik saglar (`api/model_
    performance.py::_relative_path` ile AYNI desen)."""

    try:
        return str(path.relative_to(REPO_ROOT).as_posix())
    except ValueError:  # pragma: no cover -- REPO_ROOT altinda olmayan bir yol asla beklenmez
        return str(path)


def _build_calibration_block() -> dict[str, Any]:
    decile_rows = _read_csv_rows(COX_CALIBRATION_365_CSV)
    summary_rows = _read_csv_rows(COX_CALIBRATION_SUMMARY_CSV)

    summary_by_metrik = {row.get("metrik"): row for row in summary_rows}
    slope_row = summary_by_metrik.get("kalibrasyon_egimi")
    mae_row = summary_by_metrik.get("decile_mae")
    if slope_row is None:
        raise ModelCurvesArtifactError(
            f"'kalibrasyon_egimi' satiri bulunamadi: {COX_CALIBRATION_SUMMARY_CSV}"
        )
    if mae_row is None:
        raise ModelCurvesArtifactError(
            f"'decile_mae' satiri bulunamadi: {COX_CALIBRATION_SUMMARY_CSV}"
        )

    t_days = _as_whole_int(mae_row.get("t_gun"), field_name="decile_mae.t_gun")

    deciles: list[dict[str, Any]] = []
    for row in decile_rows:
        deciles.append(
            {
                "decile": _as_whole_int(row.get("decile"), field_name="decile"),
                "n": _as_whole_int(row.get("n"), field_name="n"),
                "events": _as_whole_int(row.get("olay"), field_name="olay"),
                "predicted_s365": row.get("tahmin_S365_ort"),
                "observed_km": row.get("gozlenen_S365_KM"),
                "km_lower": row.get("KM_alt"),
                "km_upper": row.get("KM_ust"),
            }
        )
    deciles.sort(key=lambda d: d["decile"])

    return {
        "variant": VARIANT_NAME,
        "t_days": t_days,
        "slope": {
            "value": slope_row.get("deger"),
            "ci_lower": slope_row.get("ci_alt"),
            "ci_upper": slope_row.get("ci_ust"),
            "source_note": slope_row.get("kaynak"),
        },
        "decile_mae": mae_row.get("deger"),
        "deciles": deciles,
    }


def _build_coefficients_block() -> dict[str, Any]:
    coef_rows = _read_csv_rows(COX_V3B_COEFFICIENTS_CSV)

    rows: list[dict[str, Any]] = []
    n_radiomic = 0
    n_clinical = 0
    for row in coef_rows:
        variant = row.get("variant")
        if variant != VARIANT_NAME:
            raise ModelCurvesArtifactError(
                f"Beklenmedik varyant adi katsayi CSV'sinde: {variant!r} "
                f"(beklenen: {VARIANT_NAME!r}) -- {COX_V3B_COEFFICIENTS_CSV}"
            )
        covariate = row.get("covariate")
        if not isinstance(covariate, str) or covariate == "":
            raise ModelCurvesArtifactError(
                f"Bos/gecersiz kovaryat adi -- {COX_V3B_COEFFICIENTS_CSV}"
            )
        kind = "radiomic" if covariate.startswith("WT__") else "clinical"
        if kind == "radiomic":
            n_radiomic += 1
        else:
            n_clinical += 1

        rows.append(
            {
                "covariate": covariate,
                "kind": kind,
                "coef": row.get("coef"),
                "se": row.get("se(coef)"),
                "coef_ci_lower": row.get("coef lower 95%"),
                "coef_ci_upper": row.get("coef upper 95%"),
                "hr": row.get("exp(coef)"),
                "hr_ci_lower": row.get("exp(coef) lower 95%"),
                "hr_ci_upper": row.get("exp(coef) upper 95%"),
                "z": row.get("z"),
                "p": row.get("p"),
            }
        )

    n_total = n_radiomic + n_clinical
    if (
        n_radiomic != EXPECTED_N_RADIOMIC_COVARIATES
        or n_clinical != EXPECTED_N_CLINICAL_COVARIATES
        or n_total != EXPECTED_N_COVARIATES
    ):
        raise ModelCurvesArtifactError(
            "Katsayi CSV'sindeki kovaryat sayisi beklenen sozlesmeyle "
            f"UYUSMUYOR (beklenen {EXPECTED_N_RADIOMIC_COVARIATES} radyomik + "
            f"{EXPECTED_N_CLINICAL_COVARIATES} klinik = {EXPECTED_N_COVARIATES}; "
            f"bulunan {n_radiomic} radyomik + {n_clinical} klinik = {n_total}) -- "
            f"CSV degismis/bozulmus olabilir, grafik SESSIZCE SUNULMADI: "
            f"{COX_V3B_COEFFICIENTS_CSV}"
        )

    return {
        "variant": VARIANT_NAME,
        "n_covariates": n_total,
        "rows": rows,
    }


def build_model_curves_block() -> dict[str, Any]:
    """Ana mantik -- endpoint'ten AYRI test edilebilsin diye fonksiyona
    cikarildi (mevcut desen: `api/model_performance.py::build_model_
    performance_block`).

    Basarisizlikta `ModelCurvesArtifactError` FIRLATIR -- sessizce
    kismi/yanlis veri DONMEZ, cagiran (endpoint) bunu 503'e cevirir."""

    calibration = _build_calibration_block()
    coefficients = _build_coefficients_block()

    return {
        "calibration": calibration,
        "coefficients": coefficients,
        "source_artifacts": {
            "calibration_csv": _relative_path(COX_CALIBRATION_365_CSV),
            "calibration_summary_csv": _relative_path(COX_CALIBRATION_SUMMARY_CSV),
            "coefficients_csv": _relative_path(COX_V3B_COEFFICIENTS_CSV),
        },
        "notes": [
            "Kalibrasyon egimi 0,685 [0,470-0,900] -- ust ucu 1'e ULASMIYOR, "
            "yani model uclarda asiri-guvenlidir. Bu grafikten MUTLAK sagkalim "
            "olasiligi iddiasi KURULMAZ, yalniz goreli SIRALAMA okunur; "
            "\"iyi/kotu kalibre\" hukmu YASAKTIR (CLAUDE.md A9/A10).",
            "Kalibrasyon yalniz IKI kolda (v1 ve nihai model v3b_lowvar_"
            "v2amgmt) olculmustur; kalan 6 Cox kolunda HIC olculmedi.",
            "Katsayi orman grafigindeki degerler nihai modelin TAM-HAVUZ "
            "(tum 611/585 egitim setiyle) fit'ine aittir -- nested-CV "
            "sonucu DEGILDIR, karistirilmamalidir.",
        ],
    }


@router.get("/model_curves")
def get_model_curves() -> dict[str, Any]:
    try:
        return build_model_curves_block()
    except ModelCurvesArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
