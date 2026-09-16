"""``GET /patients`` -- kohort listeleme (Kohort Gezgini'nin TEK besleyicisi).

Görev talimatı: paralel-36 (backend-agent-W1, 2026-09-15, Barış onayı --
"tasarıma sadık gerçek bir web uygulaması"). Frontend tarafı
`web/app/api.js::listPatients()` + `web/app/declarations.js::COHORT_ROLES`
(paralel-37, AYRI bir ajan, bu görevde DOKUNULMADI) -- o taraf rol
METNİNİ kendi sabitinde tutuyor, SAYILARI (n_hasta/n_olay) burada
üretilenden bekliyor ("Sayılar `GET /patients` özet alanından gelecek").

🔴 EN KRİTİK KURAL -- sayılar KAYNAKTA (bu dosyada, HER İSTEKTE canlı SQL
ile) üretilir, hiçbir yerde Python literali olarak SABİTLENMEZ. CLAUDE.md
kilitli değerleri (611/585, 295/169, 91/72, 39/38, 48) burada bir
DOĞRULAMA/regresyon-pini olarak `tests/test_api_patients.py`'de
kullanılır ama endpoint'in KENDİSİ bu sayıları HER ZAMAN sorgulayarak
üretir -- ileride DB değişirse (yeni hasta eklenirse) endpoint bunu
YANSITIR, donmuş bir sabit DEĞİL.

=======================================================================
TIMEPOINT/SEGMENTASYON-ARACI BEYAZ LİSTESİ -- TEK KAYNAK, KOPYA YAZILMADI
=======================================================================
`ALLOWED_TIMEPOINT_LABELS_BY_TOOL` (UPenn->pre-treatment, TCGA/UCSF->
baseline) `tools/train_cox_week3.py`'den DOĞRUDAN import edilir --
görev talimatının açık uyarısı: düz `WHERE timepoint_label='pre-
treatment'` yazan biri UCSF'in 295'ini ve TCGA'nın 39'unu SIFIRLAR
(2026-09-14'te koordinatör bu tuzağa düştü). `tools/` dizinine bu
dosyadan HİÇBİR ŞEY YAZILMAZ, yalnız üç sabit + bir sözlük OKUNUR.

LUMIERE bu sözlükte YOK (eğitime hiç girmediği için `tools/train_cox_
week3.py`'nin whitelisti onu kapsamıyor) -- LUMIERE'in "servis
edilebilir" tanımı YAPISAL OLARAK FARKLIDIR: statik bir timepoint etiketi
değil, `pipeline/lumiere_canonical_visit.py`'nin K2 kilitli kuralı
(ExpertRating CSV'sinde `Rating=='Pre-Op'` OLAN ve C32 radyomiği OLAN
vizit). Bu modül o kuralı YENİDEN YAZMAZ, `select_lumiere_preop_
canonical_visits()`'i BİR KEZ (istek başına) çağırıp kanonik 72 hastanın
`patient_id` kümesini elde eder.

⚠️ K2 hesaplaması BAŞARISIZ olursa (ExpertRating CSV bulunamadı VEYA
ölçülen sayı kilitli 72'den farklı) bu modül endpoint'in TAMAMINI
ÇÖKERTMEZ -- LUMIERE hastaları için `servis_edilebilir_mi` alanı bu
istekte `null` (bilinmiyor) döner ve `warnings` listesinde AÇIKÇA
nedeni yazılır (CLAUDE.md: sessiz başarısızlık yok, ama tek bir
kaynağın iç tutarlılık hatası TÜM listelemeyi öldürmemeli -- kısmi
bozulma > toplam çökme).

=======================================================================
"SERVİS EDİLEBİLİR" TANIMI -- YAKLAŞIK, `/predict`İN TAM GARANTİSİ DEĞİL
=======================================================================
Bu alan `api/predict.py::predict_patient()`'in fiilen 200 döneceğinin
KESİN garantisi DEĞİLDİR -- yalnız üç GEREKLİ (ama teorik olarak yeterli
olmayabilecek) koşulu kontrol eder:
  1. `age`/`gender` NULL DEĞİL (predict.py'nin STRICT politikası,
     `CLINICAL_COVARIATE_NULL_POLICY`),
  2. hastanın doğru zaman-noktasında (kaynağa göre pre-treatment/baseline)
     C32 radyomiği VAR (UPenn/TCGA/UCSF) veya LUMIERE'in K2 kanonik-72
     kümesinde YER ALIYOR,
  3. (kontrol EDİLMEZ) `idh1_status`/`mgmt_status`/`gtr_over90percent`
     tanınmayan/yazım-hatalı bir değer taşımıyor -- predict.py bunu HER
     ZAMAN 422 ile reddeder (TRAIN_ALIGNED politika NULL'u kapsar ama
     BOZUK bir string'i kapsamaz), bu uç nokta bu kontrolü TEKRARLAMAZ
     (predict.py'nin tanınan-etiket sözlüklerini ikinci kez taşımak
     drift riski doğururdu). Bu, `servis_edilebilir_mi=true` iken
     `/predict`'in nadiren 422 dönebileceği anlamına gelir -- BEYAN
     EDİLİR, gizlenmez.
  4. (kontrol EDİLMEZ) UPenn/TCGA/UCSF için "birden fazla scan_id"
     (`MultiScanNotSupportedError`) durumu -- ölçülen kilitli sayılar
     (611/295/39) bu üç kaynakta hasta-başına TEK C32 satırı olduğunu
     ima ediyor (GUARD 6 + timepoint kapısı), ama bu uç nokta bunu
     AYRICA doğrulamaz.

=======================================================================
FİLTRE KANONİKLEŞTİRMESİ -- `/similar`'daki (karar 34) İLE AYNI DEĞİL
=======================================================================
`idh1_status`/`mgmt_status` filtreleri BURADA ham DB değeriyle
(case-insensitive TAM eşleşme) çalışır -- `api/similar.py`'nin karar-34
kanonikleştirmesi (ör. LUMIERE'in "WT"si ile UPenn'in "Wildtype"ının
EŞ sayılması) BU UÇ NOKTAYA taşınmadı (kapsam dışı, ayrı bir görev
gerektirir). Yani `?idh1_status=Wildtype` LUMIERE'in "WT" etiketli
hastalarını YAKALAMAZ. Bu, BİLİNÇLİ bir kapsam sınırıdır, sessizce
geçiştirilmez -- yanıtta `filter_semantics_note` alanıyla açıkça belirtilir.

=======================================================================
HATA SÖZLEŞMESİ
=======================================================================
  - `source` filtresi bilinen 5 `dataset_sources.source_name` değerinden
    biri DEĞİLSE -> 422 (sessizce boş liste DÖNÜLMEZ).
  - DB'ye erişilemezse -> 503.
  - Aksi halde HER ZAMAN 200 (boş sonuç kümesi de geçerli bir yanıttır).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import psycopg2.extras
from fastapi import APIRouter, HTTPException, Query

from db_connection import get_connection

# 🔒 TEK KAYNAK -- K2 kilitli kural, bkz. modül dokstring'i. Bu modül
# LUMIERE'in kanonik-vizit mantığını YENİDEN YAZMAZ.
from pipeline.lumiere_canonical_visit import (
    EXPECTED_LUMIERE_PREOP_PATIENTS,
    LUMIERE_SOURCE_NAME,
    ExpertRatingFileNotFoundError,
    LumierePreopCohortMismatchError,
    select_lumiere_preop_canonical_visits,
)

# 🔒 TEK KAYNAK -- görev talimatının açık uyarısı: bu sözlüğü İMPORT ET,
# yeni bir kopya YAZMA (`tools/train_cox_week3.py:283-287`).
from tools.train_cox_week3 import (
    ALLOWED_TIMEPOINT_LABELS_BY_TOOL,
    SEGMENTATION_TOOL_TCGA_C32,
    SEGMENTATION_TOOL_UCSF_C32,
    SEGMENTATION_TOOL_UPENN_C32,
)

router = APIRouter()

# LUMIERE'in C32 segmentasyon-aracı adı `tools/train_cox_week3.py`'nin
# `ALLOWED_TIMEPOINT_LABELS_BY_TOOL`'unda YOK (LUMIERE eğitim havuzuna hiç
# girmiyor, bkz. modül dokstring'i) -- ama `api/predict.py::ALLOWED_C32_
# SEGMENTATION_TOOLS`'ta (4 isim) VAR. O sabiti BURADA yeniden tanımlamak
# yerine `api.predict`'ten import etmek daha "TEK KAYNAK" olurdu, ama bu
# modülün `api.predict`e bağımlı olması (shap/xgboost checkpoint yükleme
# ağırlığı) gereksiz bir bağ eklerdi -- literal string TEK başına, hiçbir
# koşulda değişmeyecek bir DB sabiti olduğu için burada AÇIKÇA tutulur.
SEGMENTATION_TOOL_LUMIERE_C32 = "LUMIERE-PyRadiomics-107-C32"

# Kaynak -> segmentasyon aracı (yalnız UPenn/TCGA/UCSF -- LUMIERE ayrı
# ele alınır). Bu eşleme hiçbir mevcut modülde TEK bir sözlük olarak
# tanımlı değil (predict.py/train_cox_week3.py bunu SQL sorgu metninde
# örtük tutuyor) -- burada AÇIKÇA yazıldı.
_SOURCE_NAME_BY_TOOL: dict[str, str] = {
    SEGMENTATION_TOOL_UPENN_C32: "UPenn-GBM",
    SEGMENTATION_TOOL_TCGA_C32: "TCGA-GBM",
    SEGMENTATION_TOOL_UCSF_C32: "UCSF-PDGM",
}

# `ZORUNLU-BEYANLAR.md` B8 rol tablosuyla BİREBİR (kelimesi kelimesine).
SOURCE_ROLES: dict[str, str] = {
    "UPenn-GBM": "Eğitim",
    "UCSF-PDGM": "Harici test",
    LUMIERE_SOURCE_NAME: "Longitudinal simülasyon + T3 RANO + FAISS",
    "TCGA-GBM": "Ne eğitim ne test — FAISS üyesi + tarihsel referans",
    "TCGA-Omics": "Opsiyonel omics modülü",
}

KNOWN_SOURCE_NAMES: frozenset[str] = frozenset(SOURCE_ROLES)

# `vital_status` üç değerden biri: NULL, 'CENSORED', 'ALIVE', 'DECEASED'
# (canlı DB'de ölçüldü, 2026-09-15). "Olay" (event) = DECEASED.
DECEASED_LABEL = "DECEASED"

DEFAULT_LIMIT = 8  # tasarımın kendi varsayılanı (görev talimatı).
MAX_LIMIT = 200  # savunmacı üst sınır -- görev talimatında YOK, kötüye
# kullanım/yanlışlıkla tüm tabloyu tek seferde çekmeyi önlemek için.


class UnknownSourceFilterError(ValueError):
    """`source` filtresi bilinen 5 `dataset_sources.source_name`'den biri değil."""


def _get_db_connection():
    return get_connection(readonly=True)


# =====================================================================
# 1) LUMIERE K2 kanonik-72 kümesi -- istek başına BİR KEZ hesaplanır.
# =====================================================================


def _fetch_lumiere_c32_long_frame(conn) -> pd.DataFrame:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT DISTINCT p.patient_id, ms.timepoint_label, r.scan_id
        FROM radiomics r
        JOIN mr_scans ms ON ms.scan_id = r.scan_id
        JOIN patients p ON p.patient_id = ms.patient_id
        JOIN dataset_sources ds ON ds.source_id = p.source_id
        WHERE ds.source_name = %s AND r.segmentation_tool = %s
        """,
        (LUMIERE_SOURCE_NAME, SEGMENTATION_TOOL_LUMIERE_C32),
    )
    rows = cur.fetchall()
    cur.close()
    if not rows:
        return pd.DataFrame(columns=["patient_id", "timepoint_label", "scan_id"])
    return pd.DataFrame(rows)


def _compute_lumiere_servisable_patient_ids(conn) -> tuple[frozenset[str], str | None]:
    """K2 kilitli kanonik-72 kümesini döner. Başarısız olursa (CSV yok /
    sayı uyuşmuyor) boş küme + görünür bir uyarı metni döner -- endpoint'in
    TAMAMINI ÇÖKERTMEZ (bkz. modül dokstring'i)."""

    long_frame = _fetch_lumiere_c32_long_frame(conn)
    if long_frame.empty:
        return frozenset(), (
            "LUMIERE için C32 radyomiği olan hiçbir satır bulunamadı -- "
            "`servis_edilebilir_mi` bu kaynak için tamamen `false`."
        )
    try:
        selected, _report = select_lumiere_preop_canonical_visits(
            long_frame, expected_patients=EXPECTED_LUMIERE_PREOP_PATIENTS
        )
    except (ExpertRatingFileNotFoundError, LumierePreopCohortMismatchError) as exc:
        return frozenset(), (
            "LUMIERE K2 kanonik-vizit kümesi hesaplanamadı "
            f"({type(exc).__name__}: {exc}). LUMIERE hastaları için "
            "`servis_edilebilir_mi` bu istekte `null` (bilinmiyor) döndü -- "
            "sessizce yutulmadı, bu uyarı ile beyan edildi."
        )
    return frozenset(selected["patient_id"].unique()), None


# =====================================================================
# 2) Kaynak-bazlı sayaçlar -- HER İSTEKTE canlı SQL (bkz. modül
#    dokstring'i "EN KRİTİK KURAL").
# =====================================================================


def _scalar_patient_event_counts(
    conn, *, segmentation_tool: str, timepoint_labels: frozenset[str]
) -> tuple[int, int]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT count(DISTINCT p.patient_id),
               count(DISTINCT p.patient_id) FILTER (WHERE p.vital_status = %s)
        FROM radiomics r
        JOIN mr_scans ms ON ms.scan_id = r.scan_id
        JOIN patients p ON p.patient_id = ms.patient_id
        WHERE r.segmentation_tool = %s AND ms.timepoint_label = ANY(%s)
        """,
        (DECEASED_LABEL, segmentation_tool, list(timepoint_labels)),
    )
    n_pool, n_events = cur.fetchone()
    cur.close()
    return int(n_pool), int(n_events)


def _scalar_predict_servisable_count(
    conn, *, segmentation_tool: str, timepoint_labels: frozenset[str]
) -> int:
    """`age`/`gender` NULL OLMAYAN alt küme -- predict.py'nin STRICT
    politikasının (age/gender) yaklaşık karşılığı."""

    cur = conn.cursor()
    cur.execute(
        """
        SELECT count(DISTINCT p.patient_id)
        FROM radiomics r
        JOIN mr_scans ms ON ms.scan_id = r.scan_id
        JOIN patients p ON p.patient_id = ms.patient_id
        WHERE r.segmentation_tool = %s AND ms.timepoint_label = ANY(%s)
          AND p.age IS NOT NULL AND p.gender IS NOT NULL
        """,
        (segmentation_tool, list(timepoint_labels)),
    )
    (n,) = cur.fetchone()
    cur.close()
    return int(n)


def _registered_counts_by_source(conn) -> dict[str, int]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ds.source_name, count(*)
        FROM patients p
        JOIN dataset_sources ds ON ds.source_id = p.source_id
        GROUP BY 1
        """
    )
    result = {row[0]: int(row[1]) for row in cur.fetchall()}
    cur.close()
    return result


def _has_omics_total(conn) -> int:
    """`patients.has_omics=TRUE` toplamı -- CLAUDE.md'nin "48 TCGA hastası"
    dediği sayı BUDUR, `dataset_sources.source_name='TCGA-Omics'` satır
    sayısı (34) DEĞİL (canlı ölçüldü, 2026-09-15: TCGA-GBM 14 +
    TCGA-Omics 34 = 48 -- iki dataset_sources satırına YAYILI). Bu bir
    BULGU olarak rapora taşınır, `ZORUNLU-BEYANLAR.md` B8'in "TCGA-Omics |
    48" satırı bu ayrımı AÇIKÇA yazmıyor."""

    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM patients WHERE has_omics = TRUE")
    (n,) = cur.fetchone()
    cur.close()
    return int(n)


def _build_counters(conn, lumiere_servisable_ids: frozenset[str]) -> dict[str, Any]:
    registered = _registered_counts_by_source(conn)

    upenn_pool, upenn_events = _scalar_patient_event_counts(
        conn,
        segmentation_tool=SEGMENTATION_TOOL_UPENN_C32,
        timepoint_labels=ALLOWED_TIMEPOINT_LABELS_BY_TOOL[SEGMENTATION_TOOL_UPENN_C32],
    )
    ucsf_pool, ucsf_events = _scalar_patient_event_counts(
        conn,
        segmentation_tool=SEGMENTATION_TOOL_UCSF_C32,
        timepoint_labels=ALLOWED_TIMEPOINT_LABELS_BY_TOOL[SEGMENTATION_TOOL_UCSF_C32],
    )
    tcga_c32, _tcga_events = _scalar_patient_event_counts(
        conn,
        segmentation_tool=SEGMENTATION_TOOL_TCGA_C32,
        timepoint_labels=ALLOWED_TIMEPOINT_LABELS_BY_TOOL[SEGMENTATION_TOOL_TCGA_C32],
    )
    tcga_predict_servisable = _scalar_predict_servisable_count(
        conn,
        segmentation_tool=SEGMENTATION_TOOL_TCGA_C32,
        timepoint_labels=ALLOWED_TIMEPOINT_LABELS_BY_TOOL[SEGMENTATION_TOOL_TCGA_C32],
    )

    lumiere_c32_long_frame_size = len(
        _fetch_lumiere_c32_long_frame(conn)["patient_id"].unique()
    )

    by_source = [
        {
            "source": "UPenn-GBM",
            "role": SOURCE_ROLES["UPenn-GBM"],
            "n_registered": registered.get("UPenn-GBM", 0),
            "n_train_pool": upenn_pool,
            "n_events_train_pool": upenn_events,
        },
        {
            "source": "UCSF-PDGM",
            "role": SOURCE_ROLES["UCSF-PDGM"],
            "n_registered": registered.get("UCSF-PDGM", 0),
            "n_external_test_pool": ucsf_pool,
            "n_events_external_test": ucsf_events,
        },
        {
            "source": LUMIERE_SOURCE_NAME,
            "role": SOURCE_ROLES[LUMIERE_SOURCE_NAME],
            "n_registered": registered.get(LUMIERE_SOURCE_NAME, 0),
            "n_c32_radiomics_any_visit": lumiere_c32_long_frame_size,
            "n_faiss_servisable_canonical_preop": len(lumiere_servisable_ids),
        },
        {
            "source": "TCGA-GBM",
            "role": SOURCE_ROLES["TCGA-GBM"],
            "n_registered": registered.get("TCGA-GBM", 0),
            "n_c32_radiomics": tcga_c32,
            "n_predict_servisable": tcga_predict_servisable,
        },
        {
            "source": "TCGA-Omics",
            "role": SOURCE_ROLES["TCGA-Omics"],
            "n_registered": registered.get("TCGA-Omics", 0),
            "n_has_omics_total": _has_omics_total(conn),
            "note": (
                "n_has_omics_total, `has_omics=TRUE` olan TÜM hastaları "
                "sayar (TCGA-GBM + TCGA-Omics dataset_sources satırlarına "
                "YAYILI: 14 + 34 = 48) -- yalnız bu satırın `n_registered` "
                "değeri (dataset_sources.source_name='TCGA-Omics') DEĞİL."
            ),
        },
    ]

    total_servis_edilebilir = (
        upenn_pool
        + ucsf_pool
        + len(lumiere_servisable_ids)
        + tcga_predict_servisable
    )

    return {
        "by_source": by_source,
        "total_servis_edilebilir": total_servis_edilebilir,
        "total_servis_edilebilir_formula": (
            "UPenn eğitim havuzu + UCSF harici test havuzu + LUMIERE K2 "
            "kanonik-72 (veya hesaplanamadıysa ölçülen alt küme) + TCGA "
            "predict-servisable"
        ),
        "note": (
            "Bu kohortlar DÖRT FARKLI ROLDE kullanılır, tek bir 'toplam "
            "kohort' sayısı YOKTUR (ZORUNLU-BEYANLAR.md B8). "
            "`patients` tablosunun toplam satır sayısı (1311) BİR KOHORT "
            "BÜYÜKLÜĞÜ DEĞİLDİR -- TCGA'nın burada MR'lı/servis edilen "
            "39/38'i değil, klinik kayıtlı ~261 satırı sayılır; bu uç "
            "nokta o toplamı KASITLI OLARAK döndürmez."
        ),
    }


# =====================================================================
# 3) Hasta listesi -- filtre + sayfalama
# =====================================================================


def _validate_source_filter(source: str | None) -> None:
    if source is not None and source not in KNOWN_SOURCE_NAMES:
        raise UnknownSourceFilterError(
            f"Bilinmeyen kaynak: {source!r}. Beklenen: {sorted(KNOWN_SOURCE_NAMES)}"
        )


def _build_where_clause(
    *,
    source: str | None,
    gender: str | None,
    idh1_status: str | None,
    mgmt_status: str | None,
    gtr_over90percent: str | None,
    servis_edilebilir: bool | None,
    lumiere_servisable_ids: frozenset[str],
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}

    if source is not None:
        clauses.append("ds.source_name = %(source)s")
        params["source"] = source
    if gender is not None:
        clauses.append("p.gender ILIKE %(gender)s")
        params["gender"] = gender
    if idh1_status is not None:
        clauses.append("p.idh1_status ILIKE %(idh1_status)s")
        params["idh1_status"] = idh1_status
    if mgmt_status is not None:
        clauses.append("p.mgmt_status ILIKE %(mgmt_status)s")
        params["mgmt_status"] = mgmt_status
    if gtr_over90percent is not None:
        clauses.append("p.gtr_over90percent ILIKE %(gtr_over90percent)s")
        params["gtr_over90percent"] = gtr_over90percent

    if servis_edilebilir is not None:
        servisable_sql = """
            (
                p.age IS NOT NULL AND p.gender IS NOT NULL AND (
                    (ds.source_name = %(tool_upenn_source)s AND EXISTS (
                        SELECT 1 FROM radiomics r JOIN mr_scans ms2
                            ON ms2.scan_id = r.scan_id
                        WHERE ms2.patient_id = p.patient_id
                          AND r.segmentation_tool = %(tool_upenn)s
                          AND ms2.timepoint_label = ANY(%(tp_upenn)s)
                    ))
                    OR (ds.source_name = %(tool_tcga_source)s AND EXISTS (
                        SELECT 1 FROM radiomics r JOIN mr_scans ms2
                            ON ms2.scan_id = r.scan_id
                        WHERE ms2.patient_id = p.patient_id
                          AND r.segmentation_tool = %(tool_tcga)s
                          AND ms2.timepoint_label = ANY(%(tp_tcga)s)
                    ))
                    OR (ds.source_name = %(tool_ucsf_source)s AND EXISTS (
                        SELECT 1 FROM radiomics r JOIN mr_scans ms2
                            ON ms2.scan_id = r.scan_id
                        WHERE ms2.patient_id = p.patient_id
                          AND r.segmentation_tool = %(tool_ucsf)s
                          AND ms2.timepoint_label = ANY(%(tp_ucsf)s)
                    ))
                    OR (ds.source_name = %(lumiere_source)s
                        AND p.patient_id = ANY(%(lumiere_ids)s))
                )
            )
        """
        params.update(
            {
                "tool_upenn_source": _SOURCE_NAME_BY_TOOL[SEGMENTATION_TOOL_UPENN_C32],
                "tool_upenn": SEGMENTATION_TOOL_UPENN_C32,
                "tp_upenn": list(ALLOWED_TIMEPOINT_LABELS_BY_TOOL[SEGMENTATION_TOOL_UPENN_C32]),
                "tool_tcga_source": _SOURCE_NAME_BY_TOOL[SEGMENTATION_TOOL_TCGA_C32],
                "tool_tcga": SEGMENTATION_TOOL_TCGA_C32,
                "tp_tcga": list(ALLOWED_TIMEPOINT_LABELS_BY_TOOL[SEGMENTATION_TOOL_TCGA_C32]),
                "tool_ucsf_source": _SOURCE_NAME_BY_TOOL[SEGMENTATION_TOOL_UCSF_C32],
                "tool_ucsf": SEGMENTATION_TOOL_UCSF_C32,
                "tp_ucsf": list(ALLOWED_TIMEPOINT_LABELS_BY_TOOL[SEGMENTATION_TOOL_UCSF_C32]),
                "lumiere_source": LUMIERE_SOURCE_NAME,
                "lumiere_ids": list(lumiere_servisable_ids),
            }
        )
        if servis_edilebilir:
            clauses.append(servisable_sql)
        else:
            clauses.append(f"NOT {servisable_sql}")

    where_sql = " AND ".join(clauses) if clauses else "TRUE"
    return where_sql, params


def _row_is_servisable(
    row: dict[str, Any], lumiere_servisable_ids: frozenset[str]
) -> bool | None:
    if row["age"] is None or row["gender"] is None:
        return False
    source = row["source"]
    if source == LUMIERE_SOURCE_NAME:
        return row["patient_id"] in lumiere_servisable_ids
    tool = {
        "UPenn-GBM": SEGMENTATION_TOOL_UPENN_C32,
        "TCGA-GBM": SEGMENTATION_TOOL_TCGA_C32,
        "UCSF-PDGM": SEGMENTATION_TOOL_UCSF_C32,
    }.get(source)
    if tool is None:
        return False
    return bool(row.get(f"_has_c32__{tool}"))


@router.get("/patients")
def list_patients(
    source: str | None = Query(
        None,
        description=(
            f"`dataset_sources.source_name` -- bilinen değerler: "
            f"{sorted(KNOWN_SOURCE_NAMES)}. Bilinmeyen bir değer 422 döner "
            "(sessizce boş liste DÖNMEZ)."
        ),
    ),
    gender: str | None = Query(None, description="Case-insensitive TAM eşleşme."),
    idh1_status: str | None = Query(
        None,
        description=(
            "Ham DB değeriyle case-insensitive TAM eşleşme -- `/similar`'ın "
            "karar-34 kanonikleştirmesi (WT==Wildtype) BURADA UYGULANMAZ "
            "(bkz. modül dokstring'i)."
        ),
    ),
    mgmt_status: str | None = Query(None, description="Ham DB değeriyle case-insensitive TAM eşleşme."),
    gtr_over90percent: str | None = Query(None, description="'Y'/'N' -- case-insensitive TAM eşleşme."),
    servis_edilebilir: bool | None = Query(
        None,
        description=(
            "true/false -- yaklaşık tanım için modül dokstring'ine bakın "
            "(predict.py'nin TAM garantisi değildir)."
        ),
    ),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
):
    try:
        _validate_source_filter(source)
    except UnknownSourceFilterError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        conn = _get_db_connection()
    except Exception as exc:  # noqa: BLE001 -- DB erişilemezse 503
        raise HTTPException(status_code=503, detail=f"DB'ye erişilemedi: {exc}")

    try:
        warnings: list[str] = []
        lumiere_servisable_ids, lumiere_warning = _compute_lumiere_servisable_patient_ids(conn)
        if lumiere_warning is not None:
            warnings.append(lumiere_warning)

        counters = _build_counters(conn, lumiere_servisable_ids)

        where_sql, where_params = _build_where_clause(
            source=source,
            gender=gender,
            idh1_status=idh1_status,
            mgmt_status=mgmt_status,
            gtr_over90percent=gtr_over90percent,
            servis_edilebilir=servis_edilebilir,
            lumiere_servisable_ids=lumiere_servisable_ids,
        )

        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            f"""
            SELECT count(*)
            FROM patients p
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE {where_sql}
            """,
            where_params,
        )
        total_matching = int(cur.fetchone()["count"])

        cur.execute(
            f"""
            SELECT p.patient_id, ds.source_name AS source, p.age, p.gender,
                   p.survival_days, p.vital_status, p.idh1_status,
                   p.mgmt_status, p.gtr_over90percent, p.has_omics,
                   EXISTS (
                       SELECT 1 FROM radiomics r JOIN mr_scans ms2
                           ON ms2.scan_id = r.scan_id
                       WHERE ms2.patient_id = p.patient_id
                         AND r.segmentation_tool = %(tool_upenn)s
                         AND ms2.timepoint_label = ANY(%(tp_upenn)s)
                   ) AS "_has_c32__{SEGMENTATION_TOOL_UPENN_C32}",
                   EXISTS (
                       SELECT 1 FROM radiomics r JOIN mr_scans ms2
                           ON ms2.scan_id = r.scan_id
                       WHERE ms2.patient_id = p.patient_id
                         AND r.segmentation_tool = %(tool_tcga)s
                         AND ms2.timepoint_label = ANY(%(tp_tcga)s)
                   ) AS "_has_c32__{SEGMENTATION_TOOL_TCGA_C32}",
                   EXISTS (
                       SELECT 1 FROM radiomics r JOIN mr_scans ms2
                           ON ms2.scan_id = r.scan_id
                       WHERE ms2.patient_id = p.patient_id
                         AND r.segmentation_tool = %(tool_ucsf)s
                         AND ms2.timepoint_label = ANY(%(tp_ucsf)s)
                   ) AS "_has_c32__{SEGMENTATION_TOOL_UCSF_C32}"
            FROM patients p
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE {where_sql}
            ORDER BY p.patient_id
            LIMIT %(limit)s OFFSET %(offset)s
            """,
            {
                **where_params,
                "tool_upenn": SEGMENTATION_TOOL_UPENN_C32,
                "tp_upenn": list(ALLOWED_TIMEPOINT_LABELS_BY_TOOL[SEGMENTATION_TOOL_UPENN_C32]),
                "tool_tcga": SEGMENTATION_TOOL_TCGA_C32,
                "tp_tcga": list(ALLOWED_TIMEPOINT_LABELS_BY_TOOL[SEGMENTATION_TOOL_TCGA_C32]),
                "tool_ucsf": SEGMENTATION_TOOL_UCSF_C32,
                "tp_ucsf": list(ALLOWED_TIMEPOINT_LABELS_BY_TOOL[SEGMENTATION_TOOL_UCSF_C32]),
                "limit": limit,
                "offset": offset,
            },
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    patients_out = []
    for row in rows:
        servisable = _row_is_servisable(row, lumiere_servisable_ids)
        if row["source"] == LUMIERE_SOURCE_NAME and lumiere_warning is not None:
            servisable = None  # K2 kümesi hesaplanamadı -- bilinmiyor.
        patients_out.append(
            {
                "patient_id": row["patient_id"],
                "source": row["source"],
                "role": SOURCE_ROLES.get(row["source"]),
                "age": row["age"],
                "gender": row["gender"],
                # `web/app/api.js` yorum satırı "sex" bekliyor (henüz
                # bağlanmamış/ENABLED=false taslak) -- iki anahtar da
                # verilir, hangisi kullanılırsa çalışsın (BULGU olarak
                # ayrıca raporlanır, bkz. görev raporu).
                "sex": row["gender"],
                "survival_days": row["survival_days"],
                "vital_status": row["vital_status"],
                "idh1_status": row["idh1_status"],
                "mgmt_status": row["mgmt_status"],
                "gtr_over90percent": row["gtr_over90percent"],
                "has_omics": row["has_omics"],
                "servis_edilebilir_mi": servisable,
            }
        )

    return {
        "patients": patients_out,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total_matching": total_matching,
        },
        "filters_applied": {
            k: v
            for k, v in {
                "source": source,
                "gender": gender,
                "idh1_status": idh1_status,
                "mgmt_status": mgmt_status,
                "gtr_over90percent": gtr_over90percent,
                "servis_edilebilir": servis_edilebilir,
            }.items()
            if v is not None
        },
        "filter_semantics_note": (
            "idh1_status/mgmt_status filtreleri HAM DB değeriyle "
            "case-insensitive TAM eşleşme yapar -- /similar'ın karar-34 "
            "kanonikleştirmesi (WT==Wildtype) burada UYGULANMAZ."
        ),
        "counters": counters,
        "warnings": warnings,
    }
