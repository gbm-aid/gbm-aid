"""UCSF-PDGM harici test kohortunu DONDUR (2026-08-18, Barış onayı).

Kural (görev talimatı, Barış kararı): "UCSF harici test kohortu = C32
çıkarımının başladığı an diskte `_T1c` + `_tumor_segmentation` dosyası
TAM olan tüm uygun hastalar."

Uygunluk kriteri (2026-08-18 -- DİKKAT, decisions/2026-08-15-upenn-ucsf-
kovaryat-esleme-kurallari.md'deki 367/223 DONDURULMUŞ kararından FARKLI,
o belge henüz Barış onayı BEKLİYORDU ve IDH-wildtype filtresi
içeriyordu):
  - `_FU` sonekli takip taramaları HARİÇ (6 kayıt, 501 -> 495 baseline)
  - `WHO CNS Grade` == 4
  - **IDH-wildtype filtresi UYGULANMAZ** (model IDH1'i kovaryat olarak
    içeriyor, varyans gerekli -- 2026-08-15 belgesindeki filtre bu
    görevde BİLİNÇLİ OLARAK gevşetildi)
  - Şu alanlar dolu: `OS`, `1-dead 0-alive`, `Age at MRI`, `Sex`, `EOR`,
    `IDH` (IDH burada kohort-giriş filtresi DEĞİL, yalnız doluluk şartı)
  - `_T1c.nii.gz` VE `_tumor_segmentation.nii.gz` ikisi de diskte TAM
    (ne `.partial` ne `.aspera-ckpt` kalıntısı yok)

Metadata ID formatı ("UCSF-PDGM-004", sıfır doldurmasız) ile disk klasör
adı formatı ("UCSF-PDGM-0004_nifti", 4 haneli) FARKLI -- `normalize_id()`
sayısal kısmı `int()`'ten geçirip ikisini birleştirir. Bu script bunu
YAPMADAN önce (2026-08-18 ilk taramada) doğrudan string eşlemesi 0
eşleşme veriyordu -- BULUNUP DÜZELTİLDİ, bkz. log/2026-08-18.md.

DB'ye YAZMAZ, `raw/`'a YAZMAZ (2026-08-18'de bir immutability ihlali
zaten oldu, bu script tekrarlamaz) -- yalnız `gbm-aid mert/artifacts/
week3/ucsf_cohort/` altına CSV + sidecar SHA256 manifest yazar.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[2]
DISK_BASE = PROJECT_ROOT / "PKG - UCSF-PDGM Version 5" / "UCSF-PDGM-v5"
META_PATH = (
    PROJECT_ROOT
    / "raw"
    / "veri"
    / "ucsf-pdgm"
    / "metadata"
    / "UCSF-PDGM-metadata_v5.csv"
)
OUTPUT_DIR = Path(__file__).absolute().parents[1] / "artifacts" / "week3" / "ucsf_cohort"

REQUIRED_FIELDS = ["OS", "1-dead 0-alive", "Age at MRI", "Sex", "EOR", "IDH"]


def normalize_id(raw_id: str) -> str:
    match = re.match(r"(UCSF-PDGM-)(\d+)(_FU\S*)?$", raw_id)
    if not match:
        return raw_id
    prefix, num, fu = match.groups()
    return f"{prefix}{int(num)}{fu or ''}"


def scan_disk(base: Path) -> tuple[dict[str, dict[str, str]], list[tuple], list[str]]:
    complete: dict[str, dict[str, str]] = {}
    incomplete: list[tuple] = []
    empty: list[str] = []

    for entry in sorted(os.listdir(base)):
        directory = base / entry
        files = os.listdir(directory)
        match = re.match(r"(UCSF-PDGM-\d+)(_FU\S*)?_nifti", entry)
        raw_id = match.group(1) if match else entry
        fu_suffix = match.group(2) if (match and match.group(2)) else None
        key = normalize_id(raw_id + (fu_suffix or ""))

        if not files:
            empty.append(entry)
            continue

        partial_stems: set[str] = set()
        ckpt_stems: set[str] = set()
        real_files: set[str] = set()
        for filename in files:
            if filename.endswith(".aspera-ckpt"):
                ckpt_stems.add(filename[: -len(".aspera-ckpt")])
            elif filename.endswith(".partial"):
                partial_stems.add(filename[: -len(".partial")])
            else:
                real_files.add(filename)

        t1c_name = next((f for f in real_files if f.endswith("_T1c.nii.gz")), None)
        seg_name = next(
            (f for f in real_files if f.endswith("_tumor_segmentation.nii.gz")), None
        )
        t1c_complete = t1c_name is not None and t1c_name not in partial_stems and t1c_name not in ckpt_stems
        seg_complete = seg_name is not None and seg_name not in partial_stems and seg_name not in ckpt_stems

        if t1c_complete and seg_complete:
            complete[key] = {"folder": entry, "t1c_file": t1c_name, "seg_file": seg_name}
        else:
            incomplete.append((key, entry, t1c_complete, seg_complete, len(files)))

    return complete, incomplete, empty


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def eligible_metadata_rows(rows: list[dict[str, str]]) -> tuple[list[dict], dict[str, list[str]]]:
    baseline = [row for row in rows if "_FU" not in row["ID"]]
    grade4 = [row for row in baseline if row["WHO CNS Grade"].strip() == "4"]

    def is_filled(row: dict[str, str]) -> bool:
        return all(str(row.get(field, "")).strip() for field in REQUIRED_FIELDS)

    eligible = [row for row in grade4 if is_filled(row)]
    dropped = {
        row["ID"]: [f for f in REQUIRED_FIELDS if not str(row.get(f, "")).strip()]
        for row in grade4
        if not is_filled(row)
    }
    return eligible, dropped


def main() -> int:
    complete, incomplete, empty = scan_disk(DISK_BASE)
    meta_rows = load_metadata(META_PATH)
    eligible_rows, dropped = eligible_metadata_rows(meta_rows)
    eligible_by_id = {normalize_id(row["ID"]): row for row in eligible_rows}

    frozen_ids = sorted(set(eligible_by_id) & set(complete))
    events = sum(
        1 for pid in frozen_ids if str(eligible_by_id[pid]["1-dead 0-alive"]).strip() == "1"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_csv = OUTPUT_DIR / f"ucsf_frozen_cohort_2026-08-18_{len(frozen_ids)}.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "folder", "t1c_file", "seg_file"] + REQUIRED_FIELDS)
        for pid in frozen_ids:
            row = eligible_by_id[pid]
            disk = complete[pid]
            writer.writerow(
                [pid, disk["folder"], disk["t1c_file"], disk["seg_file"]]
                + [row[field] for field in REQUIRED_FIELDS]
            )

    sha256 = hashlib.sha256(out_csv.read_bytes()).hexdigest()

    manifest = {
        "frozen_at_utc": timestamp,
        "rule": (
            "diskte _T1c + _tumor_segmentation TAM olan VE metadata "
            "kriterlerini (WHO CNS Grade==4, non-FU, OS/vital/age/sex/"
            "EOR/IDH dolu, IDH-wildtype filtresi YOK) karşılayan tüm "
            "UCSF-PDGM hastaları"
        ),
        "disk_scan": {
            "total_patient_dirs": len(complete) + len(incomplete) + len(empty),
            "complete_t1c_and_seg": len(complete),
            "incomplete_partial_download": len(incomplete),
            "empty_not_started": len(empty),
        },
        "metadata_scan": {
            "total_rows": len(meta_rows),
            "baseline_non_fu": len(meta_rows) - sum(1 for r in meta_rows if "_FU" in r["ID"]),
            "grade4_baseline": len(eligible_rows) + len(dropped),
            "eligible_after_required_fields": len(eligible_rows),
            "dropped_for_missing_fields": dropped,
        },
        "frozen_cohort": {
            "n_patients": len(frozen_ids),
            "n_events": events,
            "output_csv": str(out_csv),
            "output_csv_sha256": sha256,
        },
        "known_caveat": (
            "UCSF-PDGM indirmesi bu dondurma anında TAMAMLANMAMIŞTI "
            f"({len(incomplete)} kısmi + {len(empty)} başlamamış klasör). "
            "Kural gereği bu SNAPSHOT dondurulmuş kohorttur -- indirme "
            "ilerledikçe DAHA FAZLA hasta uygun hale gelebilir ama bu "
            "kohort GERİYE DÖNÜK genişletilmez (kural: extraction'ın "
            "BAŞLADIĞI an)."
        ),
    }
    manifest_path = out_csv.with_suffix(out_csv.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Dondurulan kohort: {len(frozen_ids)} hasta / {events} olay")
    print(f"CSV: {out_csv}")
    print(f"SHA256: {sha256}")
    print(f"Manifest: {manifest_path}")
    print(
        f"Disk: complete={len(complete)} incomplete={len(incomplete)} empty={len(empty)} "
        f"(toplam {len(complete) + len(incomplete) + len(empty)})"
    )
    print(
        f"Metadata: grade4_baseline={len(eligible_rows) + len(dropped)} "
        f"eligible_after_fields={len(eligible_rows)} dropped={len(dropped)} ({list(dropped)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
