"""``GET /patient/{patient_id}/volume_series`` -- longitudinal WT hacim-zaman
serisi (Hasta Gecmisi ekrani, backend-agent-X1, paralel-38, 2026-09-16,
Baris onayi).

BU DOSYA VERI URETMEZ -- `pipeline/growth_simulation.py`'nin KENDI kilitli
sorgu sozlesmesini (`SEGMENTATION_TOOL='LUMIERE-PyRadiomics-107-C32'`,
`TUMOR_REGION='WT_derived'`, `patient_id ILIKE 'Patient-%'`) DOGRUDAN cagirir,
YENIDEN YAZMAZ (gorev talimati). RANO etiketi icin de `pipeline/followup_t3.
py::pair_lumiere_visits_with_rano()` KULLANILIR -- o fonksiyon zaten
(patient_id, week) uzerinden hacim serisini RANO etiketleriyle eslestiriyor
(sezgisel/heuristic oldugu KENDI docstring'inde beyan edilmis) -- burada
AYNI eslesmeyi tekrar YAZMAK yerine DOGRUDAN cagrilir.

`api/analyze_patient.py`'ye DOKUNULMADI (gorev kisiti) -- o dosyadaki
`_build_growth_simulation_block()` hala yalniz `n_visits` doner, bu YENI/
AYRI uc nokta seriyi disari verir.

=======================================================================
SOZLESME (koordinator karari, GORUNTU harici -- degistirilmedi)
=======================================================================
{
  "available": bool,
  "patient_id": str,
  "n_visits": int,
  "visits": [ {"week": float, "volume_mm3": float, "rano": str|null} ],
  "not_available_reason": str|null
}

`visits[].week` -- `pipeline.growth_simulation.fetch_lumiere_wt_volume_
series()`'in urettigi `x_week` degeridir (hafta + varsa ayni-hafta-icinde
sira icin kucuk epsilon ofseti, DOGRULANMAMIS bir yaklastirma -- bkz. o
modulun docstring'i "Bilinmeyen/varsayilan noktalar"). Gercek tarama
TARIHI LUMIERE icin DB'de hic dolu degil, bu yuzden GERCEK zaman araligi
DEGIL, YALNIZ sira/yaklasik hafta ekseni temsil eder -- rapor/UI bunu
mutlak bir zaman cizelgesi gibi SUNMAMALIDIR.

`visits[].rano` -- `pair_lumiere_visits_with_rano()`'nin (patient_id, week)
uzerinden esledigi normalize RANO etiketidir (PD/SD/PR/CR/Pre-Op/Post-Op/
UNKNOWN -- bkz. `growth_simulation.normalize_rano_label`). Eslesme
bulunamayan (RANO kaydi yok VEYA ayni-hafta kovasinda fazlalik/eslenemedi)
ziyaretler icin `null` doner -- UYDURULMUS bir etiket ASLA konulmaz.
Seffaflik icin ek bir alan (`rano_match_type`: "unique"|"heuristic_paired"|
null) de eklenmistir -- sozlesmeyi BOZMAZ (yalniz EK alan), ama ayni-hafta
kovasinda birden fazla aday oldugunda eslesmenin sezgisel oldugunu ACIKCA
tasir (gizlenmez).

=======================================================================
HATA SOZLESMESI (CLAUDE.md: sessiz basarisizlik YOK, mevcut desenlerle
hizali -- bkz. `api/mr_slice.py`/`api/similar.py`)
=======================================================================
  - Hasta `patients` tablosunda YOK                      -> 404
  - DB'ye hic erisilemiyor (baglanti/sorgu hatasi)        -> 503
  - Hasta VAR ama LUMIERE longitudinal C32 WT serisinde
    kaydi YOK (UCSF/TCGA/UPenn -- tek-zaman-noktali
    kohortlar, veya LUMIERE'de radyomik satiri olmayan
    bir hasta)                                            -> 200,
    `available:false` + acik `not_available_reason` (SESSIZ
    bos liste veya uydurulmus tek nokta DONDURULMEZ).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from db_connection import get_connection
import pipeline.growth_simulation as growth_simulation_module
from pipeline.followup_t3 import pair_lumiere_visits_with_rano

router = APIRouter()
_logger = logging.getLogger("api.volume_series")


def _get_db_connection():
    return get_connection(readonly=True)


def _fetch_patient_exists(conn, patient_id: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM patients WHERE patient_id = %s", (patient_id,))
    found = cur.fetchone() is not None
    cur.close()
    return found


class VolumeSeriesDbError(RuntimeError):
    """DB baglantisi/sorgusu basarisiz oldu -- endpoint bunu 503'e cevirir."""


def build_volume_series_block(patient_id: str) -> dict[str, Any]:
    """Cagiran (endpoint) icin ana mantik -- hasta `patients` tablosunda VAR
    kabul edilerek cagrilir (404 kontrolu endpoint'te, DB baglantisi bu
    fonksiyondan ONCE acilip patient-existence icin kullanilir; burada
    AYRI bir baglanti acilir cunku growth_simulation fonksiyonlari kendi
    connection parametrelerini alir).

    Hicbir zaman istisna FIRLATMAZ (savunmaci) -- HER basarisizlik acik
    `not_available_reason` ile 200 doner; `VolumeSeriesDbError` YALNIZ
    endpoint'in 404 kontrolu icin acilan ilk baglanti basarisiz olursa
    endpoint tarafindan dogrudan yakalanir (bu fonksiyonun DISINDA)."""

    try:
        conn = _get_db_connection()
    except Exception as exc:  # noqa: BLE001 -- savunmaci, DB erisilemezse cokme
        _logger.warning(
            "volume_series DB baglantisi kuramadi (patient_id=%s): %s",
            patient_id, exc,
        )
        return {
            "available": False,
            "patient_id": patient_id,
            "n_visits": 0,
            "visits": [],
            "not_available_reason": f"DB baglantisi kurulamadi: {type(exc).__name__}: {exc}",
        }

    try:
        series_df = growth_simulation_module.fetch_lumiere_wt_volume_series(conn)
    except Exception as exc:  # noqa: BLE001 -- savunmaci
        _logger.warning(
            "volume_series hacim serisi okuma hatasi (patient_id=%s): %s",
            patient_id, exc,
        )
        return {
            "available": False,
            "patient_id": patient_id,
            "n_visits": 0,
            "visits": [],
            "not_available_reason": f"Beklenmeyen hata (hacim serisi okuma): {type(exc).__name__}: {exc}",
        }

    patient_series = series_df[series_df["patient_id"] == patient_id]
    if patient_series.empty:
        conn.close()
        return {
            "available": False,
            "patient_id": patient_id,
            "n_visits": 0,
            "visits": [],
            "not_available_reason": (
                f"'{patient_id}' icin LUMIERE (Patient-XXX) longitudinal C32 "
                "WT hacim serisinde kayit bulunamadi -- hacim-zaman serisi su "
                "an SADECE LUMIERE kohortunun (segmentation_tool="
                f"{growth_simulation_module.SEGMENTATION_TOOL!r}, tumor_region="
                f"{growth_simulation_module.TUMOR_REGION!r}) longitudinal C32 "
                "verisiyle uretilebiliyor -- pipeline/growth_simulation.py'nin "
                "kendi kilitli sorgu sozlesmesi (`patient_id ILIKE 'Patient-"
                "%'`). Tek-zaman-noktali hastalar (UCSF/TCGA/UPenn) icin "
                "uretilebilecek bir seri YOK -- uydurulmus bir nokta "
                "DONDURULMEDI."
            ),
        }

    # RANO eslestirmesi -- `pipeline.followup_t3.pair_lumiere_visits_with_
    # rano()` TUM LUMIERE hastalarini doner, `scan_id` ile filtrelenir.
    # Bu fonksiyon (patient_id, week) uzerinden sezgisel/heuristic eslesme
    # yapar (KENDI docstring'inde beyan edilmis) -- burada TEKRAR YAZILMAZ.
    try:
        rano_matched_df = pair_lumiere_visits_with_rano(conn)
    except Exception as exc:  # noqa: BLE001 -- RANO eslestirmesi basarisizsa
        # hacim serisi YINE DE donmeli (RANO opsiyonel zenginlestirme) --
        # ama sessizce yutulmaz, warning loglanir.
        _logger.warning(
            "volume_series RANO eslestirmesi basarisiz oldu (patient_id=%s), "
            "hacim serisi RANO'suz donuyor: %s", patient_id, exc,
        )
        rano_matched_df = None
    finally:
        conn.close()

    rano_by_scan_id: dict[Any, tuple[str, str]] = {}
    if rano_matched_df is not None and not rano_matched_df.empty:
        patient_rano_rows = rano_matched_df[rano_matched_df["patient_id"] == patient_id]
        for _, row in patient_rano_rows.iterrows():
            scan_id = row.get("scan_id")
            if scan_id is None:
                continue
            rano_by_scan_id[scan_id] = (row.get("rano_label"), row.get("match_type"))

    patient_series = patient_series.sort_values("x_week")
    visits: list[dict[str, Any]] = []
    for _, row in patient_series.iterrows():
        rano_label, match_type = rano_by_scan_id.get(row.get("scan_id"), (None, None))
        visits.append(
            {
                "week": float(row["x_week"]),
                "volume_mm3": float(row["tumor_volume_mm3"]),
                "rano": rano_label,
                "rano_match_type": match_type,
            }
        )

    return {
        "available": True,
        "patient_id": patient_id,
        "n_visits": len(visits),
        "visits": visits,
        "not_available_reason": None,
    }


@router.get("/patient/{patient_id}/volume_series")
def get_volume_series(patient_id: str) -> dict[str, Any]:
    try:
        conn = _get_db_connection()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"DB'ye erisilemedi: {exc}")

    try:
        exists = _fetch_patient_exists(conn, patient_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"DB'ye erisilemedi: {exc}")
    finally:
        conn.close()

    if not exists:
        raise HTTPException(status_code=404, detail=f"Hasta bulunamadi: {patient_id}")

    return build_volume_series_block(patient_id)
