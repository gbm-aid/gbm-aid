"""Bagimsiz kohort dogrulamasi (decisions/2026-08-15-upenn-ucsf-kovaryat-
esleme-kurallari.md kurallarina gore) + indirilen 401 hastayla kesisim.

SALT-OKUNUR, hicbir dosyaya yazmiyor, kohort DONDURMUYOR.
"""
from __future__ import annotations

import csv
from pathlib import Path

META = Path(r"Y:\raw\veri\ucsf-pdgm\metadata\UCSF-PDGM-metadata_v5.csv")
AVAILABLE = Path(
    r"C:\Users\BAR~1\AppData\Local\Temp\claude\C--Users-Bar---Desktop-GBM-AID-Prototip"
    r"\0147f9fb-cb88-4c3a-ade8-9bd39dd90392\scratchpad\available_patients.txt"
)

with open(AVAILABLE, "r", encoding="utf-8") as fh:
    available = {line.strip() for line in fh if line.strip()}


def to_disk_id(raw_id: str) -> str:
    prefix, num = raw_id.rsplit("-", 1)
    return f"{prefix}-{int(num):04d}"


with open(META, "r", encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))

print(f"metadata toplam satir: {len(rows)}")

fu_excluded = [r for r in rows if "_FU" in r["ID"]]
baseline = [r for r in rows if "_FU" not in r["ID"]]
print(f"_FU haric: {len(fu_excluded)}, baseline: {len(baseline)}")

def eligible(r):
    if r["WHO CNS Grade"] != "4":
        return False
    if r["IDH"].strip().lower() != "wildtype":
        return False
    for field in ("OS", "1-dead 0-alive", "Age at MRI", "Sex", "EOR"):
        if not r[field] or not r[field].strip():
            return False
    return True

cohort = [r for r in baseline if eligible(r)]
events = sum(1 for r in cohort if r["1-dead 0-alive"].strip() == "1")
print(f"uygun kohort (grade4+IDHwt+OS/olay/yas/cinsiyet/EOR dolu): {len(cohort)} hasta / {events} olay")

cohort_disk_ids = {to_disk_id(r["ID"]) for r in cohort}
intersect = cohort_disk_ids & available
missing_from_disk = cohort_disk_ids - available
events_intersect = sum(
    1 for r in cohort if to_disk_id(r["ID"]) in available and r["1-dead 0-alive"].strip() == "1"
)
print(f"indirilen 401 ile kesisim: {len(intersect)} hasta / {events_intersect} olay")
print(f"uygun kohortta olup DISKTE OLMAYAN: {len(missing_from_disk)}")

pilot_ids = sorted(intersect)[:5]
print("PILOT_5:", pilot_ids)
out = Path(
    r"C:\Users\BAR~1\AppData\Local\Temp\claude\C--Users-Bar---Desktop-GBM-AID-Prototip"
    r"\0147f9fb-cb88-4c3a-ade8-9bd39dd90392\scratchpad\pilot_5_patients.txt"
)
out.write_text("\n".join(pilot_ids) + "\n", encoding="utf-8")

