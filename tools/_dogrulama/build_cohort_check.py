"""Bagimsiz kohort dogrulamasi (decisions/2026-08-15-upenn-ucsf-kovaryat-
esleme-kurallari.md kurallarina gore) + indirilen 401 hastayla kesisim.

SALT-OKUNUR, hicbir dosyaya yazmiyor, kohort DONDURMUYOR.
"""
from __future__ import annotations

# 2026-09-16: sabit kodlanmis mutlak yollar KALDIRILDI -- kullanici dizini
# adi halka acik depoya siziyordu ve bu script baska makinede calismiyordu.
# Yollar artik dosyanin kendi konumundan / ortam degiskeninden turetilir.
import os as _os
from pathlib import Path as _P
_BURASI   = _P(__file__).resolve().parent          # tools/_dogrulama
_KOK_KOD  = _P(__file__).resolve().parents[2]      # gbm-aid mert
_KOK_PROJ = _P(__file__).resolve().parents[3]      # GBM-AID Prototip
_VERI_KOKU = _P(_os.environ.get('GBMAID_VERI_KOKU', str(_KOK_PROJ)))

import csv
from pathlib import Path

META = Path(str(_VERI_KOKU / r'raw\veri\ucsf-pdgm\metadata\UCSF-PDGM-metadata_v5.csv'))
AVAILABLE = Path(
    _BURASI / 'available_patients.txt'
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
    _BURASI / 'pilot_5_patients.txt'
)
out.write_text("\n".join(pilot_ids) + "\n", encoding="utf-8")

