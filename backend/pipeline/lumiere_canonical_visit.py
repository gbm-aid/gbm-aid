"""K2 KILITLI kural -- LUMIERE kanonik vizit secimi (`Rating == 'Pre-Op'`).

=====================================================================
TEK KAYNAK (single source of truth) -- 2026-09-13, Y1 gorevi
=====================================================================
Bu modul, K2 kilitli kuralinin (2026-08-18, Baris karari) TEK
uygulamasidir. ONCEDEN ayni mantik YALNIZ `tools/rebuild_faiss_indexes.py`
icinde yasiyordu; Y1 karari (`decisions/2026-09-13-lumiere-risk-skoru-
preop-kanonik-vizit.md`) ayni kurali `api/predict.py`nin LUMIERE yoluna da
bagladigi icin kural BURAYA TASINDI ve iki cagiran da BURADAN kullanir:

  - `tools/rebuild_faiss_indexes.py::fetch_lumiere_preop_canonical_visits()`
    -> bu modulun `select_lumiere_preop_canonical_visits()`'ine DELEGE eder
    (govde TASINDI, TEK SATIR mantik degisikligi YOK -- kohort-seviyesi
    beklenen-sayi guard'i cagiran tarafta kaldi ki o dosyadaki
    `EXPECTED_LUMIERE_PREOP_PATIENTS` monkeypatch'lenebilir kalsin).
  - `api/predict.py` -> `select_canonical_preop_visit_for_patient()`
    (TEK hasta, ayni ilkel yapi taslari: `load_lumiere_preop_visit_keys()`
    + `lumiere_timepoint_sort_key()`).

❌ Bu kural UCUNCU bir yere KOPYALANMAZ. (2026-09-13'te `tools/data_
integrity_check.py`'de bir "eslesme aynasi" yuzunden drift riski dogdu ve
belgelendi -- ayni hata tekrarlanmayacak. Yeni bir tuketici olursa BU
modulu import eder.)

=====================================================================
KURAL
=====================================================================
Kanonik vizit = `raw/veri/LUMIERE-ExpertRating-v202211 database
girecek.csv` icinde `Rating (according to RANO...)` kolonu `.strip()`
sonrasi TAM OLARAK `'Pre-Op'` olan vizit. `Patient` kolonu
`patients.patient_id`, `Date` kolonu `mr_scans.timepoint_label` ile
BIREBIR eslesir (canli DB'ye karsi dogrulandi, 2026-08-18).

🔴 `timepoint_label` SIRALAMASI ASLA OLCUT DEGILDIR -- "en erken taramayi
al" kurali ÖLÇÜMLE ÇÜRÜTÜLDÜ (2026-09-13):
  - `week-000*` etiketli 135 satirin **44'u `Post-Op`** (yalniz 91'i Pre-Op).
  - **`Patient-020`: `week-000` -> `Post-Op`, `week-000-1` -> `Pre-Op`.**
    Etikete gore siralayan bir kural TAM BURADA ameliyat-SONRASI taramayi
    secer ve ameliyat-oncesi veriyle egitilmis modele yanlis goruntuyu
    verir (SESSIZCE yanlis risk skoru).
  - **`Patient-060`**: bir `Pre-Op` viziti **`week-069`**'da -- "hafta 0"
    varsayimi da coker.
`lumiere_timepoint_sort_key()` YALNIZ ayni hastanin BIRDEN FAZLA
Pre-Op-VE-C32 vizitini tekillestirmek icin (tie-break) kullanilir --
kuralin KENDISI her zaman `Rating` kolonudur.

`Rating` kolonunda kirli degerler var (`'Post-Op '` sondan bosluklu,
`'Post-Op/PD'`) -- bu yuzden `.strip()` + TAM esleme (`== 'Pre-Op'`)
kullanilir, `startswith`/`in` ASLA (reviewer uyarisi, 2026-08-18).

=====================================================================
OLCUM (canli DB + CSV, 2026-09-13'te backend-agent-I tarafindan TEKRAR
dogrulandi -- 2026-08-18 K2 olcumuyle BIREBIR AYNI)
=====================================================================
  Pre-Op etiketli (hasta, vizit) cifti : 92  (91 ayri hasta;
                                        `Patient-060`'ta 2 -- week-000
                                        VE week-069)
  Pre-Op VE C32 radyomigi olan         : 73 vizit / **72 hasta**  <- KANONIK
  Pre-Op vizitinde C32'si OLMAYAN      : 19 hasta -> ACIK HATA (fallback
                                        REDDEDILDI, Baris karari: post-op
                                        goruntu yapisal olarak farklidir
                                        [rezeksiyon boslugu, degisen
                                        kontrast tutulumu])

⚠️ `raw/` DEGISMEZ -- bu modul CSV'yi YALNIZ OKUR.

PATH NOTU (K12): `Path(__file__).resolve()` KULLANILMAZ -- Windows'ta
`subst X:` eslemesini Turkce karakterli gercek yola geri cozer (bkz.
tests/test_no_resolve_path_regression.py). `.absolute()` kullanilir.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# `.absolute()` -- `.resolve()` DEGIL (K12, bkz. modul dokstring'i).
PIPELINE_DIR = Path(__file__).absolute().parent
CODE_ROOT = PIPELINE_DIR.parents[1]  # "gbm-aid mert" (backend/pipeline -> kok)

_EXPERT_RATING_RELATIVE_PATH = Path("raw") / "veri" / (
    "LUMIERE-ExpertRating-v202211 database girecek.csv"
)

#: `.env`/ortam degiskeniyle acik override (varsayilan YOK). Yol
#: bulunamazsa ACIK hata verilir, sessizce baska bir kurala DUSULMEZ.
EXPERT_RATING_CSV_ENV_VAR = "GBMAID_LUMIERE_EXPERT_RATING_CSV"


def _candidate_expert_rating_csv_paths() -> list[Path]:
    """K12 ILE UYUMLU proje-koku bulma (ACIK RISK, belgelenmis).

    `raw/` PROJE KOKUNDE, yani `gbm-aid mert`in BIR UST dizinindedir --
    ama K12 geregi surec `X:` (subst) surucusunden baslatilir ve **`X:`
    dogrudan `gbm-aid mert`e eslenir**, dolayisiyla `X:\\..` PROJE KOKU
    DEGILDIR (canli olculdu, 2026-09-13: `X:\\raw\\veri\\...` YOK).
    `Path.resolve()` subst eslemesini gercek (Turkce karakterli) yola geri
    cozdugu icin DOGRU koku verir -- `tools/rebuild_faiss_indexes.py` de
    tam bu sebeple `.resolve()` kullaniyordu.

    Cozum: iki adayi SIRAYLA dener, VAR OLANI kullanir. `.resolve()` bu
    modulde SADECE CSV OKUMAK icin, TEK bu fonksiyonda gecer -- bu modul
    HICBIR goruntu/NIfTI YAZMAZ (K12'nin ITK yasagi goruntu YAZAN
    modullere ozgudur, bkz. tests/test_no_resolve_path_regression.py
    docstring'i "CSV/DB-only araclar kapsam DISIDIR").
    """

    import os

    candidates: list[Path] = []
    override = os.environ.get(EXPERT_RATING_CSV_ENV_VAR)
    if override:
        candidates.append(Path(override))
    # 1) Normal calistirma (subst YOK): .absolute() yeterli.
    candidates.append(CODE_ROOT.parent / _EXPERT_RATING_RELATIVE_PATH)
    # 2) `X:` subst altinda: gercek yola geri coz.
    # lint-allow-resolve: yalniz CSV OKUMA yolu; bu modul goruntu YAZMAZ,
    # subst kokunun ustune (proje koku) cikmanin baska yolu yok.
    candidates.append(Path(__file__).resolve().parents[3] / _EXPERT_RATING_RELATIVE_PATH)
    return candidates


def _locate_expert_rating_csv() -> Path:
    candidates = _candidate_expert_rating_csv_paths()
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Hicbiri yoksa ILK adayi dondur -- `load_lumiere_preop_visit_keys()`
    # ACIK `ExpertRatingFileNotFoundError` firlatsin (sessiz fallback YOK).
    return candidates[0]


DEFAULT_LUMIERE_EXPERT_RATING_CSV = _locate_expert_rating_csv()

#: K2'nin TEK olcutu. `.strip()` sonrasi TAM esleme aranir.
LUMIERE_PREOP_RATING_LABEL = "Pre-Op"

#: K2 kilitli beklenen kanonik hasta sayisi (Pre-Op VE C32'si olan).
EXPECTED_LUMIERE_PREOP_PATIENTS = 72

#: LUMIERE hasta kimligi oneki -- `patients.patient_id` deseni
#: (`tools/data_integrity_check.py::PATIENT_ID_PREFIX_RULES` bunu canli
#: DB'de DENETLER: "LUMIERE" -> "Patient-%"; `pipeline/growth_simulation.py`
#: da ayni sozlesmeyi SQL tarafinda kullanir).
LUMIERE_PATIENT_ID_PREFIX = "Patient-"

LUMIERE_SOURCE_NAME = "LUMIERE"

_LUMIERE_WEEK_NUMBER_PATTERN = re.compile(r"week-(\d+)")


class ExpertRatingFileNotFoundError(RuntimeError):
    """`raw/veri/LUMIERE-ExpertRating-v202211 database girecek.csv` bulunamadi.

    `raw/` immutable oldugu icin bu dosya asla YAZILMAZ, yalniz okunur --
    yoksa K2'nin kilitli kurali UYGULANAMAZ, sessizce baska bir kurala
    (or. "en erken tarama") DUSULMEZ.
    """


class LumierePreopCohortMismatchError(RuntimeError):
    """K2 kilitli beklenen sayi (72 hasta) olcuenle UYUSMUYOR.

    Baris'in acik talimati: "sayi tutmazsa DUR."
    """


class LumierePreopVisitNotAvailableError(RuntimeError):
    """TEK hasta icin kanonik (`Rating=='Pre-Op'`) vizit SECILEMEDI.

    Iki ayri sebep olabilir, ikisi de mesaja ACIKCA yazilir:
      1. Hastanin CSV'de hic `Pre-Op` satiri YOK (bugunku canli veride
         91/91 hastada VAR -- bu dal olusursa CSV degismis demektir).
      2. `Pre-Op` viziti VAR ama O VIZITTE C32 radyomigi YOK (bugun 19
         hasta). Baska bir vizite (post-op) DUSULMEZ -- Baris karari.

    Sessiz bos/varsayilan DONDURULMEZ (Y1 karari, sart 4).
    """

    def __init__(
        self,
        message: str,
        *,
        patient_id: str,
        preop_labels_in_csv: list[str],
        available_labels: list[str],
    ) -> None:
        super().__init__(message)
        self.patient_id = patient_id
        self.preop_labels_in_csv = preop_labels_in_csv
        self.available_labels = available_labels


def is_lumiere_patient_id(patient_id: str) -> bool:
    """`patient_id` LUMIERE deseninde mi (`Patient-%`)?

    Bu, K2'nin kanonik-vizit kuralinin UYGULANACAGI kohortun kapisidir --
    diger kaynaklarda (UPenn/UCSF/TCGA) kanonik vizit kurali YOKTUR ve
    coklu-tarama durumu ACIK HATA olarak kalmalidir (Y1 karari, sart 2).
    """

    return patient_id.startswith(LUMIERE_PATIENT_ID_PREFIX)


def lumiere_timepoint_sort_key(timepoint_label: str) -> tuple[int, str]:
    """`Date`/`timepoint_label` degerini (or. 'week-000', 'week-000-1')
    "en erken once" sirasina koyacak bir anahtara cevirir -- YALNIZ AYNI
    hastanin BIRDEN FAZLA Pre-Op-VE-C32 vizitini tekillestirmek icin
    (`Patient-060` ornegi) kullanilir, K2'nin KENDISINI DEGISTIRMEZ."""

    match = _LUMIERE_WEEK_NUMBER_PATTERN.match(timepoint_label)
    week_number = int(match.group(1)) if match else 10**9
    return (week_number, timepoint_label)


def load_lumiere_preop_visit_keys(
    csv_path: Path = DEFAULT_LUMIERE_EXPERT_RATING_CSV,
) -> tuple[set[tuple[str, str]], int, int]:
    """CSV'den `Rating.strip() == 'Pre-Op'` olan (patient_id, timepoint)
    ciftlerini dondurur. `.strip()` + TAM esleme kullanilir -- CSV'de
    kirli degerler (`'Post-Op '`, `'Post-Op/PD'`) VAR, bunlar `startswith`/
    `in` ile YANLISLIKLA eslesebilirdi (reviewer uyarisi, 2026-08-18)."""

    if not csv_path.is_file():
        raise ExpertRatingFileNotFoundError(
            f"LUMIERE ExpertRating CSV'si bulunamadi: {csv_path}. K2'nin "
            "kilitli kurali (Rating=='Pre-Op') bu dosya olmadan "
            "UYGULANAMAZ -- sessizce baska bir kurala DUSULMEDI."
        )

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rating_col_candidates = [
            c for c in reader.fieldnames or [] if c.startswith("Rating (according to RANO")
        ]
        if len(rating_col_candidates) != 1:
            raise ExpertRatingFileNotFoundError(
                "LUMIERE ExpertRating CSV'sinde beklenen 'Rating (according "
                f"to RANO...' kolonu bulunamadi/belirsiz. Kolonlar: {reader.fieldnames}"
            )
        rating_col = rating_col_candidates[0]
        rows = list(reader)

    preop_pairs: set[tuple[str, str]] = set()
    for row in rows:
        if row[rating_col].strip() == LUMIERE_PREOP_RATING_LABEL:
            preop_pairs.add((row["Patient"].strip(), row["Date"].strip()))

    n_preop_rated_rows = len(preop_pairs)
    n_preop_rated_patients = len({pid for pid, _date in preop_pairs})
    return preop_pairs, n_preop_rated_rows, n_preop_rated_patients


# =====================================================================
# KOHORT SEVIYESI (FAISS indeks insasi) -- govde `tools/rebuild_faiss_
# indexes.py`'den AYNEN tasindi, mantik DEGISMEDI.
# =====================================================================


@dataclass
class LumierePreopSelectionReport:
    csv_path: str
    n_preop_rated_rows: int
    n_preop_rated_patients: int
    n_matched_preop_c32_pairs: int
    n_output_patients: int
    n_preop_patients_without_c32: int
    preop_patients_without_c32: list[str]
    n_patients_with_tie_broken_duplicate: int


def select_lumiere_preop_canonical_visits(
    long_frame: pd.DataFrame,
    *,
    csv_path: Path = DEFAULT_LUMIERE_EXPERT_RATING_CSV,
    expected_patients: int = EXPECTED_LUMIERE_PREOP_PATIENTS,
) -> tuple[pd.DataFrame, LumierePreopSelectionReport]:
    """K2 KILITLI kural -- hasta basina TEK `scan_id` secer (Rating=='Pre-Op'
    VE C32 radyomigi olan vizit). `long_frame`, `fetch_c32_radiomics_long_
    frame(segmentation_tool=SEGMENTATION_TOOL_LUMIERE_C32)`'in CIKTISI olmali
    (`scan_id`/`timepoint_label` kolonlari tasimali).

    Beklenen sayi (`expected_patients`, kilitli varsayilan 72) tutmazsa
    `LumierePreopCohortMismatchError` ile SERT durur -- "sayi tutmazsa DUR"
    (Baris talimati).

    2026-09-13 NOT: bu govde `tools/rebuild_faiss_indexes.py::fetch_lumiere_
    preop_canonical_visits()`'ten BURAYA TASINDI (Y1 gorevi, tek-kaynak);
    o fonksiyon artik buraya DELEGE eder. Beklenen-sayi TEK degisiklik
    olarak parametre haline getirildi (cagiran taraf kendi modul
    sabitini gecirir -- boylece o dosyadaki mevcut monkeypatch'li testler
    BIREBIR ayni davranisi gorur).
    """

    preop_pairs, n_preop_rated_rows, n_preop_rated_patients = load_lumiere_preop_visit_keys(
        csv_path
    )

    c32_pairs = (
        long_frame[["patient_id", "timepoint_label", "scan_id"]]
        .drop_duplicates()
    )
    c32_pairs["_key"] = list(zip(c32_pairs["patient_id"], c32_pairs["timepoint_label"]))
    matched = c32_pairs[c32_pairs["_key"].isin(preop_pairs)].copy()

    n_matched_pairs = len(matched.drop_duplicates(subset=["patient_id", "timepoint_label"]))

    # Ayni hastanin BIRDEN FAZLA Pre-Op+C32 vizitini tekillestir (Patient-060
    # ornegi) -- en erken vizit kazanir (proje genelinde "birden fazla aday
    # varsa en erken" ilkesiyle tutarli, UPenn'in A1.3 tie-break'iyle analojik).
    matched["_sort_key"] = matched["timepoint_label"].map(lumiere_timepoint_sort_key)
    dedup_counts = matched.groupby("patient_id")["scan_id"].nunique()
    n_tie_broken = int((dedup_counts > 1).sum())

    canonical = (
        matched.sort_values(["patient_id", "_sort_key"])
        .groupby("patient_id", as_index=False)
        .first()
    )
    canonical_scan_ids = set(canonical["scan_id"])

    n_output_patients = canonical["patient_id"].nunique()
    if n_output_patients != expected_patients:
        raise LumierePreopCohortMismatchError(
            f"K2 kilitli beklenen LUMIERE hasta sayisi "
            f"{expected_patients}, olculen {n_output_patients}. "
            "CSV veya C32 verisi degismis olabilir -- Baris talimati geregi "
            "SESSIZCE devam EDILMEDI, DURULDU."
        )

    preop_patient_ids = {pid for pid, _date in preop_pairs}
    matched_patient_ids = set(canonical["patient_id"])
    without_c32 = sorted(preop_patient_ids - matched_patient_ids)

    selected = long_frame[long_frame["scan_id"].isin(canonical_scan_ids)].copy()

    report = LumierePreopSelectionReport(
        csv_path=str(csv_path),
        n_preop_rated_rows=n_preop_rated_rows,
        n_preop_rated_patients=n_preop_rated_patients,
        n_matched_preop_c32_pairs=n_matched_pairs,
        n_output_patients=n_output_patients,
        n_preop_patients_without_c32=len(without_c32),
        preop_patients_without_c32=without_c32,
        n_patients_with_tie_broken_duplicate=n_tie_broken,
    )
    return selected, report


# =====================================================================
# TEK HASTA (canli servis -- `api/predict.py`) -- AYNI ilkel yapi
# taslarini kullanir: `load_lumiere_preop_visit_keys()` + `lumiere_
# timepoint_sort_key()`. Kural IKINCI KEZ YAZILMADI.
# =====================================================================


@dataclass(frozen=True)
class CanonicalVisitSelection:
    """Tek hasta icin secilen kanonik vizit -- SEFFAFLIK icin yanitta
    AYNEN beyan edilir (Y1 gorevi: "hangi vizitin kullanildigi acikca yer
    alsin")."""

    patient_id: str
    timepoint_label: str
    scan_id: Any
    rating: str
    n_preop_visits_in_csv: int
    n_candidate_visits_with_c32: int
    tie_broken: bool
    candidate_visits: tuple[str, ...]


def select_canonical_preop_visit_for_patient(
    patient_id: str,
    visit_keys: Iterable[tuple[str, Any]],
    *,
    csv_path: Path = DEFAULT_LUMIERE_EXPERT_RATING_CSV,
) -> CanonicalVisitSelection:
    """TEK LUMIERE hastasi icin kanonik `Rating=='Pre-Op'` vizitini secer.

    `visit_keys`: o hastanin C32 radyomigi OLAN vizitleri --
    `(timepoint_label, scan_id)` ciftleri. Hangi `segmentation_tool`
    beyaz listesinden geldigi CAGIRAN tarafin sorumlulugudur (bu modul
    DB'ye HIC dokunmaz).

    Secim `select_lumiere_preop_canonical_visits()` ile MATEMATIKSEL
    OLARAK AYNIDIR:
      1. aday kume = CSV'de `Rating=='Pre-Op'` olan (patient_id, label)
         ciftleriyle kesisim,
      2. birden fazla aday varsa `lumiere_timepoint_sort_key` ile EN ERKEN
         olan (tie-break; kohort yolunda `sort_values(_sort_key)` +
         `groupby.first()` AYNI sonucu verir).

    Aday YOKSA `LumierePreopVisitNotAvailableError` -- sessiz bos/
    varsayilan/post-op fallback YOK (Y1 karari, sart 4).
    """

    preop_pairs, _rows, _patients = load_lumiere_preop_visit_keys(csv_path)
    preop_labels_in_csv = sorted(
        {label for pid, label in preop_pairs if pid == patient_id},
        key=lumiere_timepoint_sort_key,
    )

    unique_visits = sorted({(str(label), scan_id) for label, scan_id in visit_keys})
    candidates = [
        (label, scan_id)
        for label, scan_id in unique_visits
        if (patient_id, label) in preop_pairs
    ]

    if not candidates:
        available_labels = sorted({label for label, _scan in unique_visits}, key=lumiere_timepoint_sort_key)
        if not preop_labels_in_csv:
            reason = (
                f"Hasta {patient_id!r} icin LUMIERE ExpertRating CSV'sinde "
                f"HIC `Rating=='{LUMIERE_PREOP_RATING_LABEL}'` satiri YOK "
                f"({csv_path.name}). Bugunku canli veride 91/91 hastada VAR -- "
                "bu dal olustuysa CSV/kohort degismis demektir."
            )
        else:
            reason = (
                f"Hasta {patient_id!r}: kanonik ameliyat-oncesi vizit(ler)i "
                f"{preop_labels_in_csv} CSV'de TANIMLI ama O VIZIT(LER)DE C32 "
                f"radyomigi YOK (C32'si olan vizitler: {available_labels}). "
                "Post-op/baska bir vizite SESSIZCE DUSULMEDI -- post-op "
                "goruntu yapisal olarak farklidir (rezeksiyon boslugu, degisen "
                "kontrast tutulumu) ve model ameliyat-oncesi T1ce ile "
                "egitilmistir (Baris karari, K2/Y1). Bugun bu durumda 19 "
                "LUMIERE hastasi var."
            )
        raise LumierePreopVisitNotAvailableError(
            reason,
            patient_id=patient_id,
            preop_labels_in_csv=preop_labels_in_csv,
            available_labels=available_labels,
        )

    candidates.sort(key=lambda item: lumiere_timepoint_sort_key(item[0]))
    label, scan_id = candidates[0]
    return CanonicalVisitSelection(
        patient_id=patient_id,
        timepoint_label=label,
        scan_id=scan_id,
        rating=LUMIERE_PREOP_RATING_LABEL,
        n_preop_visits_in_csv=len(preop_labels_in_csv),
        n_candidate_visits_with_c32=len(candidates),
        tie_broken=len(candidates) > 1,
        candidate_visits=tuple(lbl for lbl, _scan in candidates),
    )
