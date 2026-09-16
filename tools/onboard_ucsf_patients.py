"""UCSF-PDGM'i `dataset_sources` + `patients` + `mr_scans`'a onboard et.

Kapsam: SADECE altyapı (`radiomics` tablosuna DOKUNMAZ -- imaging-agent'in
`write_ucsf_pyradiomics_to_db.py`'si UCSF radyomiğini bu altyapı yazıldıktan
SONRA, AYRI bir onaylı görevde yazacak). Bu script, `write_ucsf_pyradiomics_
to_db.py`'nin GUARD 5'te tespit ettiği yapısal blokajı (UCSF için scan_id
eşlemesi YOK) çözer.

Kaynak: dondurulmuş kohort (OTORİTE, `decisions/2026-08-18-tek-model-ucsf-
harici-test-k15-kapanisi.md` "Kohort dondurma kuralı"):
    artifacts/week3/ucsf_cohort/ucsf_frozen_cohort_2026-08-18_295.csv
    (SHA256 7f3be47a74c93f8d420ad0bacfb5fab874095f32f6cbd0dccf79f514170fa968,
    manifest'te doğrulanır, mismatch varsa DUR)
    295 hasta / 169 olay. MGMT status bu dosyada YOK -- raw metadata'dan
    (`raw/veri/ucsf-pdgm/metadata/UCSF-PDGM-metadata_v5.csv`, salt-okunur,
    ID ile join) tamamlanır; bu, `raw/` immutability kuralını ihlal ETMEZ
    (yalnızca okunuyor, hangi hastaların dahil olduğu değişmiyor).

Varsayılan: DRY-RUN (`get_connection(readonly=True)`, DB'ye hiçbir yazma
yapılmaz). `--apply` ile gerçek yazım (2026-08-18, BARIŞ ONAYI VERİLDİ --
bkz. aşağıdaki "ÇAKIŞMA ÇÖZÜLDÜ" notu). Yazım TEK transaction içinde
atomiktir: `get_connection(readonly=False)` autocommit=False ile açılır
(db_connection.py:59-60 -- yalnız readonly=True dalı `autocommit=True`
set ediyor, apply dalı psycopg2'nin varsayılan davranışıyla implicit
transaction açıyor), dataset_sources + patients + mr_scans satırlarının
TÜMÜ tek `commit()` ile yazılır; herhangi bir satırda hata olursa TÜMÜ
`rollback()` edilir (kısmi yazım YOK).

✅ ÇAKIŞMA ÇÖZÜLDÜ (2026-08-18, Barış onayı): `tools/train_cox_week3.py::
build_ucsf_clinical_covariate_frame()`'in HAM UCSF sözlüğü (M/F, GTR/STR/
biopsy, wildtype/mutasyon-string) beklemesi ile bu script'in HARMONİZE
değer (Male/Female, Y/N, Wildtype/Mutated) yazması arasındaki çakışma
Barış tarafından (a) şıkkıyla çözüldü: **DB HARMONİZE DEĞER TAŞIR.**
Gerekçe: mevcut 3 kaynak (UPenn/TCGA/LUMIERE) zaten harmonize
(`gender`='Male'/'Female' üçünde de); UCSF'i ham sözlükle yazmak
kaynak-farkında olmayan bir sorguyu (`WHERE gtr_over90percent='Y'`)
UCSF'i SESSİZCE atlayacak şekilde bozar -- 2026-08-14'ün CENSORED
hatasıyla AYNI sınıf. `build_ucsf_clinical_covariate_frame()`'in
kaldırılması modeling-agent'e AYRICA iletildi, bu script'in kapsamı
DIŞINDA.

⚠️ BİLİNÇLİ NULL bırakılan alanlar (2026-08-18, Barış talimatı --
"tahminle doldurma, NULL bırak, sonradan doğrulanıp UPDATE edilebilir"):
`dataset_sources.institution` / `license_type` / `access_url` -- bu ajanın
web erişimi yok, TCIA sayfası doğrulanamadı. Üçü de nullable (information_
schema ile doğrulandı). `n_patients`=295 bu üçünden FARKLI -- tahmin değil,
kendi dry-run/apply ölçümümüz (onboard edilen dondurulmuş kohort sayısı),
LUMIERE (91=91) ve UPenn-GBM (630=630) emsaliyle TUTARLI.

Kabul edilen sapmalar (Barış onayı, engel DEĞİL -- raporda beyan edilir):
- `mr_scans.file_path` MUTLAK yerel disk yolu (diğer kaynaklar NAS-göreli)
  -- UCSF NAS'ta değil, `resolve_nas_path()`'in `direct.is_absolute()`
  dalı kod değişikliği gerektirmeden bunu çözer (pipeline/segmentation.py
  okunarak doğrulandı). SimpleITK'nın Türkçe karakter kısıtı nedeniyle bu
  kaydı okuyacak her script kendi ASCII-safe (subst) çözümünü uygulamalı.
- `vital_status` 0 -> 'ALIVE' (UCSF'te CENSORED ayrımı yok) --
  `cox_model.py`'de ALIVE/CENSORED zaten aynı event=0 muamelesini görüyor.
- `harmonization_status`='zscore' -- TCGA-GBM emsaliyle (154/154) aynı:
  N4 + T1ce-özel Z-score alır, ComBat almaz.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

# `.resolve()` DEĞİL `.absolute()` -- `.resolve()` Windows'ta `subst X:`
# eşlemesini gerçek (Türkçe karakterli) yola geri çözer (K12, 2026-08-14 /
# 2026-09-12 düzeltmesi). Bu script kendisi NIfTI YAZMAZ (yalnız CSV/DB),
# ama `UCSF_DISK_ROOT` gibi mutlak dosya yolları `REPO_ROOT`'tan türetiliyor
# ve bu yollar ileride görüntü okuyan/yazan araçlara (mask_overlay.py vb.)
# geçirilebiliyor -- tutarlılık için aynı kural uygulanır.
PROJECT_ROOT = Path(__file__).absolute().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from db_connection import get_connection  # noqa: E402

SOURCE_NAME = "UCSF-PDGM"
# dataset_sources planlanan diğer kolonlar -- n_patients ONBOARD EDİLEN
# (dondurulmuş kohort=295) sayıyı taşır, TCIA'nın yayınladığı tam koleksiyon
# büyüklüğünü (501) DEĞİL -- LUMIERE (91=91) ve UPenn-GBM (630=630)
# emsaliyle TUTARLI. institution/license_type/access_url BİLİNÇLİ OLARAK
# NULL (2026-08-18 Barış talimatı -- doğrulanmamış tahminle doldurma).
PLANNED_DATASET_SOURCE_ROW = {
    "source_name": SOURCE_NAME,
    "institution": None,  # DOĞRULANMADI -- Barış talimatıyla NULL bırakıldı
    "license_type": None,  # DOĞRULANMADI -- Barış talimatıyla NULL bırakıldı
    "n_patients": 295,
    "access_url": None,  # DOĞRULANMADI -- Barış talimatıyla NULL bırakıldı
}
DEFAULT_FROZEN_CSV = (
    PROJECT_ROOT / "artifacts" / "week3" / "ucsf_cohort" / "ucsf_frozen_cohort_2026-08-18_295.csv"
)
EXPECTED_FROZEN_SHA256 = "7f3be47a74c93f8d420ad0bacfb5fab874095f32f6cbd0dccf79f514170fa968"
DEFAULT_RAW_METADATA_CSV = (
    REPO_ROOT / "raw" / "veri" / "ucsf-pdgm" / "metadata" / "UCSF-PDGM-metadata_v5.csv"
)
# Yerel disk kökü -- NAS DEĞİL (2026-08-18 görev talimatı, Barış onayladı).
# `subst` harfleri (X:/Y:) SESSİONA ÖZGÜ olduğu için DB'ye GÖMÜLMEZ; gerçek
# proje-köküne göre MUTLAK yol yazılır.
UCSF_DISK_ROOT = REPO_ROOT / "PKG - UCSF-PDGM Version 5" / "UCSF-PDGM-v5"

EXPECTED_N_PATIENTS = 295
EXPECTED_N_EVENTS = 169
EXPECTED_GTR = {"GTR": 188, "STR": 76, "biopsy": 31}
EXPECTED_IDH_WILDTYPE = 275
EXPECTED_IDH_MUTANT = 20

# ---------------------------------------------------------------------
# Eşleme sözlükleri (TEK YERDE, açık -- görev talimatı, Barış (a) şıkkını
# onayladı: DB HARMONİZE değer taşır)
# ---------------------------------------------------------------------

# Cinsiyet: UPenn/TCGA/LUMIERE'in HER ÜÇÜ de 'Male'/'Female' kullanıyor
# (2026-08-18 canlı SELECT ile doğrulandı) -- UCSF bu konvansiyona uyar.
SEX_MAP = {"M": "Male", "F": "Female"}

# Sağkalım statüsü: UPenn'in 3-kategorili şeması (DECEASED/ALIVE/CENSORED)
# ile UYUMLU (pipeline/cox_model.py:780 -- deceased_labels=("DECEASED",),
# censored_labels=("ALIVE","CENSORED"), ikisi de event=0). UCSF'in kendi
# CSV'si "1-dead 0-alive" İKİLİ -- CENSORED/ALIVE ayrımı YOK, bu yüzden
# 0 -> 'ALIVE' (alan adının kendisi de "alive" diyor; Cox eğitiminde
# ALIVE/CENSORED zaten AYNI event=0 muamelesini görüyor, bu seçim
# sayısal sonucu ETKİLEMEZ, Barış onayladı).
VITAL_STATUS_MAP = {"1": "DECEASED", "0": "ALIVE"}

# Rezeksiyon genişliği -- 🔴 EN RİSKLİ EŞLEME, VARSAYIM (ölçüm değil).
# Kilitli (decisions/2026-08-18-tek-model-ucsf-harici-test-k15-kapanisi.md
# + decisions/2026-08-15-upenn-ucsf-kovaryat-esleme-kurallari.md "3."):
GTR_MAP = {"GTR": "Y", "STR": "N", "biopsy": "N"}

# IDH -- UPenn'in 3-kategorili şeması (Wildtype/Mutated/NOS-NEC) ile
# UYUMLU. UCSF'te "test yapılmamış" (NOS/NEC) kategorisi YOK (K15
# kararı, decisions/2026-08-18-.../"KARAR 3") -- bu yüzden UCSF
# idh1_status'u yalnız Wildtype/Mutated değeri alır, NOS/NEC HİÇ üretilmez.
IDH_WILDTYPE_RAW = "wildtype"
IDH_MUTANT_RAW = frozenset(
    {
        "IDH1 p.R132H",
        "IDH1 p.R132C",
        "IDH1 p.R132G",
        "IDH1 p.R132S",
        "mutated (NOS)",
        "IDH2 p.R172K",
        "IDH2 p.Arg172Trp",
    }
)

# MGMT -- decisions/2026-08-15-.../"4. MGMT": Methylated<->positive,
# Unmethylated<->negative, Indeterminate/unknown<->eksik (NULL).
MGMT_MAP = {"positive": "Methylated", "negative": "Unmethylated"}
MGMT_NULL_RAW = frozenset({"indeterminate", "unknown"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_cohort(csv_path: Path) -> list[dict[str, str]]:
    actual_sha = _sha256(csv_path)
    if actual_sha != EXPECTED_FROZEN_SHA256:
        raise SystemExit(
            f"HATA: dondurulmuş kohort SHA256 uyuşmuyor.\n"
            f"  beklenen: {EXPECTED_FROZEN_SHA256}\n"
            f"  bulunan : {actual_sha}\n"
            f"  dosya   : {csv_path}\n"
            "  Bu, dondurulmuş-kohort bütünlüğü ihlalidir -- DURDUR."
        )
    manifest_path = csv_path.with_name(csv_path.name + ".manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded = manifest.get("frozen_cohort", {}).get("output_csv_sha256")
        if recorded and recorded != actual_sha:
            raise SystemExit(
                f"HATA: manifest'teki sha256 ({recorded}) dosyanınkiyle "
                f"({actual_sha}) uyuşmuyor."
            )
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def load_mgmt_by_id(csv_path: Path, wanted_ids: set[str]) -> dict[str, str]:
    """Raw UCSF metadata'dan (salt-okunur) MGMT status'u ID ile eşle.

    `raw/` içeriği DEĞİŞTİRİLMİYOR -- yalnız okunuyor. Frozen kohortun
    HANGİ hastaları içerdiği bu fonksiyonla değişmez, sadece o hastaların
    zaten dondurulmuş kohort CSV'sinde OLMAYAN bir alanı (MGMT) tamamlar.
    """
    if not csv_path.is_file():
        print(f"UYARI: raw metadata bulunamadı ({csv_path}) -- mgmt_status hep NULL kalacak.", file=sys.stderr)
        return {}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, str] = {}
    for row in rows:
        pid = row.get("ID", "").strip()
        if pid not in wanted_ids:
            continue
        raw = (row.get("MGMT status") or "").strip()
        result[pid] = raw
    missing = wanted_ids - set(result)
    if missing:
        print(
            f"UYARI: {len(missing)} hasta raw metadata'da MGMT status için bulunamadı "
            f"(mgmt_status NULL kalacak): {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}",
            file=sys.stderr,
        )
    return result


def map_mgmt(raw: str, *, patient_id: str, unmapped: Counter) -> str | None:
    raw_norm = raw.strip()
    if not raw_norm:
        unmapped["MGMT_BOS"] += 1
        return None
    if raw_norm in MGMT_MAP:
        return MGMT_MAP[raw_norm]
    if raw_norm.lower() in MGMT_NULL_RAW:
        return None
    # 2026-08-14 dersi -- tanınmayan değeri SESSİZCE NULL'a çevirme,
    # AÇIKÇA say ve raporla.
    unmapped[f"MGMT_TANINMAYAN:{raw_norm}"] += 1
    return None


def map_idh(raw: str, *, patient_id: str, unmapped: Counter) -> str:
    raw_norm = raw.strip()
    if raw_norm == IDH_WILDTYPE_RAW:
        return "Wildtype"
    if raw_norm in IDH_MUTANT_RAW:
        return "Mutated"
    unmapped[f"IDH_TANINMAYAN:{raw_norm}"] += 1
    raise ValueError(f"{patient_id}: tanınmayan IDH değeri {raw_norm!r} -- kohort giriş kriteri ihlali")


def map_gtr(raw: str, *, patient_id: str, unmapped: Counter) -> str:
    raw_norm = raw.strip()
    if raw_norm in GTR_MAP:
        return GTR_MAP[raw_norm]
    unmapped[f"EOR_TANINMAYAN:{raw_norm}"] += 1
    raise ValueError(f"{patient_id}: tanınmayan EOR değeri {raw_norm!r}")


def map_sex(raw: str, *, patient_id: str, unmapped: Counter) -> str:
    raw_norm = raw.strip()
    if raw_norm in SEX_MAP:
        return SEX_MAP[raw_norm]
    unmapped[f"SEX_TANINMAYAN:{raw_norm}"] += 1
    raise ValueError(f"{patient_id}: tanınmayan Sex değeri {raw_norm!r}")


def map_vital_status(raw: str, *, patient_id: str, unmapped: Counter) -> str:
    raw_norm = raw.strip()
    if raw_norm in VITAL_STATUS_MAP:
        return VITAL_STATUS_MAP[raw_norm]
    unmapped[f"VITAL_TANINMAYAN:{raw_norm}"] += 1
    raise ValueError(f"{patient_id}: tanınmayan '1-dead 0-alive' değeri {raw_norm!r}")


def build_planned_rows(
    frozen_rows: list[dict[str, str]],
    mgmt_by_id: dict[str, str],
) -> tuple[list[dict[str, object]], Counter]:
    unmapped: Counter = Counter()
    planned: list[dict[str, object]] = []
    for row in frozen_rows:
        patient_id = row["ID"].strip()
        folder = row["folder"].strip()
        t1c_file = row["t1c_file"].strip()

        age = int(row["Age at MRI"])
        gender = map_sex(row["Sex"], patient_id=patient_id, unmapped=unmapped)
        vital_status = map_vital_status(row["1-dead 0-alive"], patient_id=patient_id, unmapped=unmapped)
        survival_days = int(row["OS"])
        idh1_status = map_idh(row["IDH"], patient_id=patient_id, unmapped=unmapped)
        gtr_over90percent = map_gtr(row["EOR"], patient_id=patient_id, unmapped=unmapped)
        mgmt_raw = mgmt_by_id.get(patient_id, "")
        mgmt_status = map_mgmt(mgmt_raw, patient_id=patient_id, unmapped=unmapped)

        file_path = str((UCSF_DISK_ROOT / folder / t1c_file)).replace("\\", "/")

        planned.append(
            {
                "patient_id": patient_id,
                "age": age,
                "gender": gender,
                "kps_score": None,
                "diagnosis_type": None,
                "vital_status": vital_status,
                "survival_days": survival_days,
                "mgmt_status": mgmt_status,
                "mgmt_status_raw": mgmt_raw or None,
                "idh1_status": idh1_status,
                "idh_raw": row["IDH"].strip(),
                "gtr_over90percent": gtr_over90percent,
                "eor_raw": row["EOR"].strip(),
                "has_mr": True,
                "has_omics": False,
                "file_path": file_path,
                "modality": "T1ce",
                "harmonization_status": "zscore",
                "timepoint_label": "baseline",
                "source_format": "nifti",
                "has_perfusion": False,
                "has_diffusion": False,
            }
        )
    return planned, unmapped


def fetch_existing_source_id(connection) -> int | None:
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT source_id FROM dataset_sources WHERE source_name = %s", (SOURCE_NAME,))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        cursor.close()


def fetch_existing_patient_ids(connection, patient_ids: list[str]) -> set[str]:
    if not patient_ids:
        return set()
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT patient_id FROM patients WHERE patient_id = ANY(%s)",
            (patient_ids,),
        )
        return {row[0] for row in cursor.fetchall()}
    finally:
        cursor.close()


def fetch_existing_scan_keys(connection, patient_ids: list[str]) -> set[tuple[str, str, str]]:
    if not patient_ids:
        return set()
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT patient_id, modality, timepoint_label
            FROM mr_scans
            WHERE patient_id = ANY(%s)
            """,
            (patient_ids,),
        )
        return {(row[0], row[1], row[2]) for row in cursor.fetchall()}
    finally:
        cursor.close()


def compute_plan(connection, planned: list[dict[str, object]]) -> tuple[str, list[dict[str, object]]]:
    """DB'de olan/olmayanı ayır -- hem dry-run raporu hem --apply için TEK yer."""

    existing_source_id = fetch_existing_source_id(connection)
    source_action = "ZATEN_VAR" if existing_source_id is not None else "EKLENECEK"

    planned_ids = [p["patient_id"] for p in planned]
    existing_patient_ids = fetch_existing_patient_ids(connection, planned_ids)
    existing_scan_keys = fetch_existing_scan_keys(connection, planned_ids)

    rows_out: list[dict[str, object]] = []
    for p in planned:
        pid = p["patient_id"]
        patient_action = "ZATEN_VAR" if pid in existing_patient_ids else "EKLENECEK"
        scan_key = (pid, p["modality"], p["timepoint_label"])
        scan_action = "ZATEN_VAR" if scan_key in existing_scan_keys else "EKLENECEK"
        rows_out.append(
            {
                **p,
                "patient_db_action": patient_action,
                "scan_db_action": scan_action,
            }
        )
    return source_action, rows_out


def write_report(report_path: Path, rows_out: list[dict[str, object]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id", "age", "gender", "vital_status", "survival_days",
        "idh1_status", "idh_raw", "gtr_over90percent", "eor_raw",
        "mgmt_status", "mgmt_status_raw", "file_path", "harmonization_status",
        "timepoint_label", "patient_db_action", "scan_db_action",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_out:
            out = dict(row)
            out["mgmt_status"] = out.get("mgmt_status") or ""
            out["mgmt_status_raw"] = out.get("mgmt_status_raw") or ""
            writer.writerow(out)


def print_summary(source_action: str, rows_out: list[dict[str, object]], *, mode: str, report_path: Path) -> None:
    print(f"\n=== {mode} ÖZETİ ===")
    print(f"dataset_sources: source_name='{SOURCE_NAME}' -> {source_action}")
    print(f"  Planlanan satır (mevcutsa YAZILMAZ): {PLANNED_DATASET_SOURCE_ROW}")
    print(f"patients: {Counter(r['patient_db_action'] for r in rows_out)}")
    print(f"mr_scans: {Counter(r['scan_db_action'] for r in rows_out)}")
    print(f"\ngender dağılımı: {Counter(r['gender'] for r in rows_out)}")
    print(f"vital_status dağılımı: {Counter(r['vital_status'] for r in rows_out)}")
    print(f"idh1_status dağılımı: {Counter(r['idh1_status'] for r in rows_out)}")
    print(f"gtr_over90percent dağılımı: {Counter(r['gtr_over90percent'] for r in rows_out)}")
    print(f"mgmt_status dağılımı: {Counter(r['mgmt_status'] or 'NULL' for r in rows_out)}")
    print(f"harmonization_status: {Counter(r['harmonization_status'] for r in rows_out)}")
    print(f"\nRapor: {report_path}")


def apply_writes(connection, source_action: str, rows_out: list[dict[str, object]]) -> dict[str, int]:
    """TEK transaction içinde dataset_sources + patients + mr_scans yaz.

    Hata olursa (herhangi bir INSERT) exception yukarı fırlatılır --
    çağıran `main()` bunu yakalayıp `connection.rollback()` çağırır.
    Kısmi yazım YOK (atomik).
    """
    cursor = connection.cursor()
    counts = {"dataset_sources_inserted": 0, "patients_inserted": 0, "mr_scans_inserted": 0}
    try:
        if source_action == "EKLENECEK":
            cursor.execute(
                """
                INSERT INTO dataset_sources (source_name, institution, license_type, n_patients, access_url)
                VALUES (%(source_name)s, %(institution)s, %(license_type)s, %(n_patients)s, %(access_url)s)
                """,
                PLANNED_DATASET_SOURCE_ROW,
            )
            counts["dataset_sources_inserted"] = 1

        cursor.execute("SELECT source_id FROM dataset_sources WHERE source_name = %s", (SOURCE_NAME,))
        source_id = cursor.fetchone()[0]

        for row in rows_out:
            if row["patient_db_action"] == "EKLENECEK":
                cursor.execute(
                    """
                    INSERT INTO patients (
                        patient_id, source_id, age, gender, kps_score, diagnosis_type,
                        vital_status, survival_days, mgmt_status, idh1_status,
                        has_mr, has_omics, gtr_over90percent
                    ) VALUES (
                        %(patient_id)s, %(source_id)s, %(age)s, %(gender)s, %(kps_score)s, %(diagnosis_type)s,
                        %(vital_status)s, %(survival_days)s, %(mgmt_status)s, %(idh1_status)s,
                        %(has_mr)s, %(has_omics)s, %(gtr_over90percent)s
                    )
                    """,
                    {**row, "source_id": source_id},
                )
                counts["patients_inserted"] += 1

            if row["scan_db_action"] == "EKLENECEK":
                cursor.execute(
                    """
                    INSERT INTO mr_scans (
                        patient_id, scan_date, modality, file_path, harmonization_status,
                        has_perfusion, has_diffusion, timepoint_label, source_format
                    ) VALUES (
                        %(patient_id)s, NULL, %(modality)s, %(file_path)s, %(harmonization_status)s,
                        %(has_perfusion)s, %(has_diffusion)s, %(timepoint_label)s, %(source_format)s
                    )
                    """,
                    row,
                )
                counts["mr_scans_inserted"] += 1
    finally:
        cursor.close()
    return counts


def verify_live(report_path: Path) -> None:
    """Ayrı, TAZE bir bağlantıyla (commit'ten SONRA) canlı doğrulama."""

    connection = get_connection(readonly=True)
    try:
        cursor = connection.cursor()
        print("\n=== CANLI DOĞRULAMA (ayrı bağlantı, commit sonrası) ===")

        cursor.execute("SELECT count(*) FROM dataset_sources WHERE source_name = %s", (SOURCE_NAME,))
        print(f"dataset_sources UCSF-PDGM satır sayısı: {cursor.fetchone()[0]} (beklenen 1)")

        cursor.execute(
            """
            SELECT count(*) FROM patients p JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE ds.source_name = %s
            """,
            (SOURCE_NAME,),
        )
        print(f"patients UCSF satır sayısı: {cursor.fetchone()[0]} (beklenen 295)")

        cursor.execute(
            """
            SELECT count(*) FROM mr_scans ms
            JOIN patients p ON p.patient_id = ms.patient_id
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE ds.source_name = %s
            """,
            (SOURCE_NAME,),
        )
        print(f"mr_scans UCSF satır sayısı: {cursor.fetchone()[0]} (beklenen 295)")

        for col in ("gender", "vital_status", "idh1_status", "gtr_over90percent", "mgmt_status"):
            cursor.execute(
                f"""
                SELECT p.{col}, count(*) FROM patients p JOIN dataset_sources ds ON ds.source_id = p.source_id
                WHERE ds.source_name = %s GROUP BY 1 ORDER BY 1
                """,
                (SOURCE_NAME,),
            )
            print(f"  {col}: {cursor.fetchall()}")

        cursor.execute(
            """
            SELECT ms.harmonization_status, count(*) FROM mr_scans ms
            JOIN patients p ON p.patient_id = ms.patient_id
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE ds.source_name = %s GROUP BY 1
            """,
            (SOURCE_NAME,),
        )
        print(f"  harmonization_status: {cursor.fetchall()}")

        cursor.execute("SELECT count(*) FROM patients")
        print(f"patients TOPLAM satır sayısı: {cursor.fetchone()[0]} (beklenen 1311, önceki 1016)")

        cursor.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'"
        )
        print(f"toplam tablo sayısı: {cursor.fetchone()[0]} (beklenen 13)")

        cursor.close()
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "UCSF-PDGM'i dataset_sources/patients/mr_scans'a onboard et "
            "(295 hasta, 2026-08-18 dondurulmuş kohort). Varsayılan DRY-RUN."
        )
    )
    parser.add_argument("--apply", action="store_true", help="Gerçek yazım (2026-08-18 Barış onayı ile serbest).")
    parser.add_argument("--frozen-csv", type=Path, default=DEFAULT_FROZEN_CSV)
    parser.add_argument("--raw-metadata-csv", type=Path, default=DEFAULT_RAW_METADATA_CSV)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Varsayılan: dry-run için ucsf_onboarding_dryrun.csv, apply için ucsf_onboarding_applied.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    report_path = args.report or (
        PROJECT_ROOT / "artifacts" / "week3" / "ucsf_cohort"
        / ("ucsf_onboarding_applied.csv" if args.apply else "ucsf_onboarding_dryrun.csv")
    )

    frozen_rows = load_frozen_cohort(args.frozen_csv)
    if len(frozen_rows) != EXPECTED_N_PATIENTS:
        print(
            f"HATA: dondurulmuş kohort {len(frozen_rows)} satır -- beklenen {EXPECTED_N_PATIENTS}. DUR.",
            file=sys.stderr,
        )
        return 2
    n_events = sum(1 for r in frozen_rows if r["1-dead 0-alive"].strip() == "1")
    if n_events != EXPECTED_N_EVENTS:
        print(
            f"HATA: olay sayısı {n_events} -- beklenen {EXPECTED_N_EVENTS}. DUR.",
            file=sys.stderr,
        )
        return 2
    eor_counts = Counter(r["EOR"].strip() for r in frozen_rows)
    if dict(eor_counts) != EXPECTED_GTR:
        print(
            f"HATA: EOR dağılımı {dict(eor_counts)} -- beklenen {EXPECTED_GTR}. DUR.",
            file=sys.stderr,
        )
        return 2
    idh_wildtype_n = sum(1 for r in frozen_rows if r["IDH"].strip() == IDH_WILDTYPE_RAW)
    idh_mutant_n = sum(1 for r in frozen_rows if r["IDH"].strip() in IDH_MUTANT_RAW)
    if idh_wildtype_n != EXPECTED_IDH_WILDTYPE or idh_mutant_n != EXPECTED_IDH_MUTANT:
        print(
            f"HATA: IDH dağılımı wildtype={idh_wildtype_n}/mutant={idh_mutant_n} -- "
            f"beklenen wildtype={EXPECTED_IDH_WILDTYPE}/mutant={EXPECTED_IDH_MUTANT}. DUR.",
            file=sys.stderr,
        )
        return 2

    print(f"SAYI DOĞRULAMASI TUTTU: {len(frozen_rows)} hasta / {n_events} olay, EOR={dict(eor_counts)}, "
          f"IDH wildtype={idh_wildtype_n}/mutant={idh_mutant_n}.")

    if args.limit is not None:
        frozen_rows = frozen_rows[: args.limit]

    wanted_ids = {r["ID"].strip() for r in frozen_rows}
    mgmt_by_id = load_mgmt_by_id(args.raw_metadata_csv, wanted_ids)

    try:
        planned, unmapped = build_planned_rows(frozen_rows, mgmt_by_id)
    except ValueError as exc:
        print(f"HATA: eşleme sırasında tanınmayan değer -- {exc}", file=sys.stderr)
        return 2

    if unmapped:
        print(f"UYARI: eşlemede kaybolan/tanınmayan değerler: {dict(unmapped)}", file=sys.stderr)

    connection = get_connection(readonly=not args.apply)
    try:
        source_action, rows_out = compute_plan(connection, planned)
        write_report(report_path, rows_out)
        print_summary(
            source_action, rows_out,
            mode="APPLY (yazım öncesi plan)" if args.apply else "DRY-RUN (--apply ÇALIŞTIRILMADI)",
            report_path=report_path,
        )

        if not args.apply:
            print(
                "\nBu script SADECE dry-run modunda çalıştı. --apply İÇİN Barış onayı VAR "
                "(2026-08-18) -- gerçek yazım için --apply bayrağıyla tekrar çalıştırılmalı."
            )
            return 0

        n_new_patients = sum(1 for r in rows_out if r["patient_db_action"] == "EKLENECEK")
        n_new_scans = sum(1 for r in rows_out if r["scan_db_action"] == "EKLENECEK")
        print(
            f"\n=== --apply: TEK TRANSACTION İÇİNDE YAZIM BAŞLIYOR "
            f"(yeni dataset_sources={0 if source_action == 'ZATEN_VAR' else 1}, "
            f"yeni patients={n_new_patients}, yeni mr_scans={n_new_scans}) ==="
        )
        try:
            counts = apply_writes(connection, source_action, rows_out)
            connection.commit()
            print(f"COMMIT TAMAMLANDI: {counts}")
        except Exception:
            connection.rollback()
            print("HATA: yazım sırasında istisna oluştu, TÜM transaction ROLLBACK edildi.", file=sys.stderr)
            raise
    finally:
        connection.close()

    if args.apply:
        verify_live(report_path)
        print(
            "\nRollback (uygulanmadı, kayıtta):\n"
            "  DELETE FROM mr_scans WHERE patient_id IN "
            "(SELECT patient_id FROM patients p JOIN dataset_sources ds ON ds.source_id=p.source_id "
            "WHERE ds.source_name='UCSF-PDGM');\n"
            "  DELETE FROM patients WHERE source_id = "
            "(SELECT source_id FROM dataset_sources WHERE source_name='UCSF-PDGM');\n"
            "  DELETE FROM dataset_sources WHERE source_name='UCSF-PDGM';"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
