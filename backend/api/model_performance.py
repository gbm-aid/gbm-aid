"""``GET /model_performance`` -- Sistem Hakkinda ekranindaki 8 kollu Cox
model tablosu + XGBoost ikinci katmani (backend-agent-X1, paralel-38,
2026-09-16, Baris onayi).

TEK DOGRULUK KAYNAGI ARTEFAKTTIR -- bu modul hicbir sayiyi SABIT
GOMMEZ/YUVARLAMAZ, hepsi asagidaki 5 dosyadan HAM olarak okunur:

  1. `artifacts/week3/cox_model/week3_all_variants_comparison.csv`
     (8 Cox kolu -- variant/rol/secim_durumu/egitim_havuzu/harici_c_index/
     ci_alt/ci_ust/ic_cv_ort/ic_cv_std/aday_ozellik/harici_n/harici_olay/...)
  2. `artifacts/week3/cox_model/week3_external_extended_metrics_ci.csv`
     (Uno-C + td-AUC uclusu (365/548/730 gun) + %95 bootstrap CI, ayni 8 kol)
  3. `artifacts/week3/cox_model/week3_v3b_calibration_summary.csv`
     (v3b kalibrasyon egimi + decile MAE)
  4. `artifacts/week4/xgboost_v2a_mgmt_reduce_collinearity_0912/
     week4_v2a_mgmt_aligned_run_metadata.json` (OOF AUC + %95 CI)
  5. `artifacts/week4/xgboost_v2a_mgmt_reduce_collinearity_0912/
     week4_v2a_mgmt_ucsf_external_evaluation.json` (UCSF harici AUC + %95 CI)

Bu dosyalarin HICBIRI bu modulde YENIDEN URETILMEZ (tools/**, artifacts/**
SALT OKUNUR, gorev kisiti) -- yalniz OKUNUR ve JSON'a serilestirilir.

⚠️ CLAUDE.md kilitli sayilariyla TUTARLILIK (okuyucu icin capraz referans,
degistirilmez): v2c/v3c egitim havuzu 609/583, digerleri 611/585 -- bu
CSV'de ZATEN boyle yaziyor, burada DUZELTILMEYE calisilmadi (gorev
talimati). Nihai secilen model `v3b_lowvar_v2amgmt`
(`secim_durumu` sutununda "SECILEN_BIRINCIL_MODEL" alt dizesiyle isaretli).

HATA SOZLESMESI (CLAUDE.md: sessiz basarisizlik YOK):
  - Yukaridaki 5 dosyadan HERHANGI BIRI eksik/bozuk/parse edilemiyor ->
    503, hangi dosyanin sorunlu oldugu ACIKCA govde/mesajda belirtilir
    (sessizce kismi/eksik bir tablo DONULMEZ -- ya TAM ya HIC).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()

# `.absolute()` -- `.resolve()` DEGIL, ayni gerekce `api/similar.py`'deki
# gibi (Turkce-karakterli yol riski -- bu modul faiss/ITK okumuyor ama
# tutarlilik icin ayni desen korunuyor).
REPO_ROOT = Path(__file__).absolute().parents[2]  # 2026-09-16: backend/ altina tasindi -> bir seviye daha yukari

COX_VARIANTS_CSV = REPO_ROOT / "artifacts" / "week3" / "cox_model" / "week3_all_variants_comparison.csv"
COX_EXTERNAL_METRICS_CSV = (
    REPO_ROOT / "artifacts" / "week3" / "cox_model" / "week3_external_extended_metrics_ci.csv"
)
COX_CALIBRATION_CSV = (
    REPO_ROOT / "artifacts" / "week3" / "cox_model" / "week3_v3b_calibration_summary.csv"
)
_XGBOOST_ARTIFACT_DIR = (
    REPO_ROOT / "artifacts" / "week4" / "xgboost_v2a_mgmt_reduce_collinearity_0912"
)
XGBOOST_RUN_METADATA_JSON = _XGBOOST_ARTIFACT_DIR / "week4_v2a_mgmt_aligned_run_metadata.json"
XGBOOST_UCSF_EXTERNAL_JSON = _XGBOOST_ARTIFACT_DIR / "week4_v2a_mgmt_ucsf_external_evaluation.json"

_CI_BRACKET_RE = re.compile(r"\[\s*([\-0-9.eE]+)\s*,\s*([\-0-9.eE]+)\s*\]")


class ModelPerformanceArtifactError(RuntimeError):
    """Kaynak artefakt (CSV/JSON) eksik veya parse edilemiyor -- endpoint
    bunu 503'e cevirir, sessizce kismi tablo DONMEZ."""


def _typed_cell(value: str) -> Any:
    """Bir CSV hucresini HAM tutarak sayisal olabiliyorsa float'a cevirir.

    Bos hucre -> `None` (eksik degeri sessizce 0/"" YAPMAZ). Sayisal
    OLMAYAN her sey (ornegin "[0.65, 0.72]" gibi CI aralik metni, ya da
    "611/585" gibi egitim havuzu metni) STRING olarak OLDUGU GIBI kalir --
    bu deger ayrica `_CI_BRACKET_RE` ile ayristirilir (bkz.
    `_parse_ci_bracket`)."""

    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _parse_ci_bracket(value: Any) -> list[float] | None:
    """`"[0.6529, 0.7599]"` bicimli bir CI metnini `[alt, ust]` float
    listesine cevirir. Ayristirilamazsa (bicim degistiyse) `None` doner --
    sessizce yanlis bir aralik UYDURULMAZ, cagiran ham `str` alani zaten
    ayrica tasiyor."""

    if not isinstance(value, str):
        return None
    match = _CI_BRACKET_RE.search(value)
    if not match:
        return None
    return [float(match.group(1)), float(match.group(2))]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ModelPerformanceArtifactError(
            f"Beklenen artefakt bulunamadi: {path} -- Cox model performans "
            "tablosu URETILEMEDI (sessizce kismi/bos tablo DONULMEDI)."
        )
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = [
                {key: _typed_cell(value) for key, value in row.items()}
                for row in reader
            ]
    except (OSError, csv.Error) as exc:
        raise ModelPerformanceArtifactError(
            f"Artefakt okunamadi/parse edilemedi: {path} ({type(exc).__name__}: {exc})"
        ) from exc
    if not rows:
        raise ModelPerformanceArtifactError(f"Artefakt bos (0 satir): {path}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelPerformanceArtifactError(
            f"Beklenen artefakt bulunamadi: {path} -- XGBoost performans "
            "blogu URETILEMEDI (sessizce kismi/bos blok DONULMEDI)."
        )
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelPerformanceArtifactError(
            f"Artefakt okunamadi/parse edilemedi: {path} ({type(exc).__name__}: {exc})"
        ) from exc


def _relative_path(path: Path) -> str:
    """Yanit govdesinde `REPO_ROOT`'a GORE yol dondurur -- Turkce-karakterli
    MUTLAK yolu (`C:/Users/Barış/...`) istemciye sizdirmadan seffaflik
    (`ZORUNLU`: kaynak artefakt yollari yanitta gorunur olmali) saglar."""

    try:
        return str(path.relative_to(REPO_ROOT).as_posix())
    except ValueError:  # pragma: no cover -- REPO_ROOT altinda olmayan bir yol asla beklenmez
        return str(path)


def _build_td_auc_block(row: dict[str, Any]) -> dict[str, Any]:
    """`week3_external_extended_metrics_ci.csv`'nin bir satirindan
    Uno-C + td-AUC uclusunu (365/548/730 gun) `{deger, ci:[alt,ust]}`
    formatinda cikarir. Ham `..._ci` string sutunu da AYRICA saklanir
    (seffaflik -- format degisirse ayristirma sessizce yanlis bir sey
    UYDURMAZ, ham metin her zaman erisilebilir kalir)."""

    horizons = (365, 548, 730)
    block: dict[str, Any] = {}
    for tau in horizons:
        uno_key = f"uno_c_tau{tau}"
        uno_ci_key = f"{uno_key}_ci"
        auc_key = f"td_auc_{tau}"
        auc_ci_key = f"{auc_key}_ci"
        block[f"tau{tau}"] = {
            "uno_c": row.get(uno_key),
            "uno_c_ci": _parse_ci_bracket(row.get(uno_ci_key)),
            "uno_c_ci_raw": row.get(uno_ci_key),
            "td_auc": row.get(auc_key),
            "td_auc_ci": _parse_ci_bracket(row.get(auc_ci_key)),
            "td_auc_ci_raw": row.get(auc_ci_key),
        }
    block["n_failed_resamples"] = row.get("n_failed_resamples")
    block["n_valid_min"] = row.get("n_valid_min")
    block["gate"] = row.get("gate")
    return block


def build_model_performance_block() -> dict[str, Any]:
    """Ana mantik -- endpoint'ten AYRI test edilebilsin diye fonksiyona
    cikarildi (mevcut desen: `api/volume_series.py::build_volume_series_
    block`, `api/analyze_patient.py::_build_growth_simulation_block`).

    Basarisizlikta `ModelPerformanceArtifactError` FIRLATIR -- sessizce
    kismi tablo DONMEZ, cagiran (endpoint) bunu 503'e cevirir."""

    variant_rows = _read_csv_rows(COX_VARIANTS_CSV)
    external_metric_rows = _read_csv_rows(COX_EXTERNAL_METRICS_CSV)
    calibration_rows = _read_csv_rows(COX_CALIBRATION_CSV)

    external_metrics_by_variant = {
        row.get("variant"): row for row in external_metric_rows
    }

    cox_variants: list[dict[str, Any]] = []
    for row in variant_rows:
        variant_name = row.get("variant")
        secim_durumu = str(row.get("secim_durumu") or "")
        merged = dict(row)
        merged["secilen_mi"] = "SECILEN" in secim_durumu.upper()

        extended_row = external_metrics_by_variant.get(variant_name)
        if extended_row is not None:
            merged["td_auc"] = _build_td_auc_block(extended_row)
        else:
            # Sessizce atlanmaz -- eslesme bulunamadiginin kendisi ACIK
            # bir alanla tasinir (CSV'ler arasinda variant adi driftı
            # olursa bu GORUNUR kalir, sessizce eksik td_auc olmaz).
            merged["td_auc"] = None
            merged["td_auc_not_available_reason"] = (
                f"'{variant_name}' external_extended_metrics_ci.csv'de "
                "bulunamadi -- iki artefakt arasinda variant adi "
                "uyusmazligi olabilir."
            )
        cox_variants.append(merged)

    xgboost_run_metadata = _read_json(XGBOOST_RUN_METADATA_JSON)
    xgboost_ucsf_external = _read_json(XGBOOST_UCSF_EXTERNAL_JSON)

    xgboost_block = {
        "variant": xgboost_ucsf_external.get("variant"),
        "role_note": (
            "XGBoost bir Cox kolu DEGILDIR -- Cox meta-skorunu girdi alan "
            "IKINCI katmandir (bkz. CLAUDE.md 2026-09-14 duzeltmesi: UCSF'e "
            "toplam bakis = 8 Cox kolu + 1 XGBoost harici degerlendirmesi = 9)."
        ),
        "n_upenn_training_patients": xgboost_ucsf_external.get("n_upenn_training_patients"),
        "n_upenn_training_events": xgboost_ucsf_external.get("n_upenn_training_events"),
        "n_cross_fit_training_patients": xgboost_ucsf_external.get("n_cross_fit_training_patients"),
        "n_cross_fit_training_events": xgboost_ucsf_external.get("n_cross_fit_training_events"),
        "oof_auc": xgboost_run_metadata.get("xgboost_auc_out_of_fold"),
        "ucsf_external_auc": xgboost_ucsf_external.get("ucsf_auc"),
        "ucsf_twelve_month_report": xgboost_ucsf_external.get("ucsf_twelve_month_report"),
        "reporting_rule_note": xgboost_ucsf_external.get("reporting_rule_note"),
    }

    return {
        "cox_variants": cox_variants,
        "calibration": calibration_rows,
        "xgboost": xgboost_block,
        "source_artifacts": {
            "cox_variants_csv": _relative_path(COX_VARIANTS_CSV),
            "cox_external_metrics_ci_csv": _relative_path(COX_EXTERNAL_METRICS_CSV),
            "cox_calibration_csv": _relative_path(COX_CALIBRATION_CSV),
            "xgboost_run_metadata_json": _relative_path(XGBOOST_RUN_METADATA_JSON),
            "xgboost_ucsf_external_evaluation_json": _relative_path(XGBOOST_UCSF_EXTERNAL_JSON),
        },
        "notes": [
            "Hicbir sayi yuvarlanarak/gomulerek uretilmedi -- hepsi yukaridaki "
            "artefaktlardan HAM okundu (bicimlendirme frontend'e birakildi).",
            "v2c_mgmt_spline_wttc ve v3c_lowvar_wttc_mgmt_nospline kollarinin "
            "egitim havuzu 609/583'tur, digerleri 611/585 -- CSV'de boyle "
            "kayitli, DUZELTILMEDI (CLAUDE.md kilitli ayrimi).",
            "Nihai secilen model v3b_lowvar_v2amgmt (`secilen_mi=true`); "
            "diger 7 kol duyarlilik/taban-cizgisi olarak SILINMEDEN listelenir "
            "(seffaflik kurali, CLAUDE.md '2026-08-18, Barış'ın netleştirmesi').",
        ],
    }


@router.get("/model_performance")
def get_model_performance() -> dict[str, Any]:
    try:
        return build_model_performance_block()
    except ModelPerformanceArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
