import os, re, csv, json, hashlib

ROOT = r"C:\Users\Barış\Desktop\GBM-AID Prototip"
BASE = os.path.join(ROOT, "PKG - UCSF-PDGM Version 5", "UCSF-PDGM-v5")
META = os.path.join(ROOT, "raw", "veri", "ucsf-pdgm", "metadata", "UCSF-PDGM-metadata_v5.csv")

def normalize_id(raw_id):
    """UCSF-PDGM-0107 and UCSF-PDGM-107 (metadata's non-zero-padded form)
    must resolve to the same key. Normalize numeric part via int() to
    drop leading zeros; keep any _FU suffix attached verbatim."""
    m = re.match(r'(UCSF-PDGM-)(\d+)(_FU\S*)?$', raw_id)
    if not m:
        return raw_id
    prefix, num, fu = m.groups()
    return f"{prefix}{int(num)}{fu or ''}"

def scan_disk():
    patients = sorted(os.listdir(BASE))
    complete = {}
    incomplete = []
    empty = []
    for p in patients:
        d = os.path.join(BASE, p)
        files = os.listdir(d)
        m = re.match(r'(UCSF-PDGM-\d+)(_FU\S*)?_nifti', p)
        pid_raw = m.group(1) if m else p
        fu_suffix = m.group(2) if (m and m.group(2)) else None
        key = normalize_id(pid_raw + (fu_suffix if fu_suffix else ""))
        if not files:
            empty.append(p)
            continue
        partial_stems = set()
        ckpt_stems = set()
        real_files = set()
        for f in files:
            if f.endswith('.aspera-ckpt'):
                ckpt_stems.add(f[:-len('.aspera-ckpt')])
            elif f.endswith('.partial'):
                partial_stems.add(f[:-len('.partial')])
            else:
                real_files.add(f)
        t1c_name = None
        seg_name = None
        for f in real_files:
            if f.endswith('_T1c.nii.gz'):
                t1c_name = f
            if f.endswith('_tumor_segmentation.nii.gz'):
                seg_name = f
        t1c_complete = t1c_name is not None and t1c_name not in partial_stems and t1c_name not in ckpt_stems
        seg_complete = seg_name is not None and seg_name not in partial_stems and seg_name not in ckpt_stems
        if t1c_complete and seg_complete:
            complete[key] = {"folder": p, "t1c": t1c_name, "seg": seg_name}
        else:
            incomplete.append((key, p, t1c_complete, seg_complete, len(files)))
    return complete, incomplete, empty

def load_meta():
    rows = []
    with open(META, encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def main():
    complete, incomplete, empty = scan_disk()
    meta = load_meta()
    print("=== DISK SCAN ===")
    print("total patient dirs:", len(complete) + len(incomplete) + len(empty))
    print("complete (T1c+tumor_seg fully present):", len(complete))
    print("incomplete (partially downloaded):", len(incomplete))
    print("empty (not started):", len(empty))
    print()

    # metadata rows
    print("=== METADATA ===")
    print("total metadata rows:", len(meta))
    fu_rows = [row for row in meta if '_FU' in row['ID']]
    print("rows with _FU suffix (excluded, follow-up):", len(fu_rows))
    baseline_rows = [row for row in meta if '_FU' not in row['ID']]
    print("baseline rows (non-FU):", len(baseline_rows))

    grade4 = [row for row in baseline_rows if row['WHO CNS Grade'].strip() == '4']
    print("baseline + WHO CNS Grade==4:", len(grade4))

    required_fields = ['OS', '1-dead 0-alive', 'Age at MRI', 'Sex', 'EOR', 'IDH']
    def filled(row):
        for fld in required_fields:
            v = row.get(fld, '')
            if v is None or str(v).strip() == '':
                return False
        return True

    eligible_meta = [row for row in grade4 if filled(row)]
    print(f"+ required fields filled {required_fields}:", len(eligible_meta))

    missing_detail = {}
    for row in grade4:
        if not filled(row):
            missing = [fld for fld in required_fields if not str(row.get(fld,'')).strip()]
            missing_detail[row['ID']] = missing
    print("dropped for missing fields (from grade4 set):", len(missing_detail))
    for k, v in list(missing_detail.items())[:20]:
        print(" ", k, v)

    # Now cross with disk-complete
    eligible_ids = {normalize_id(row['ID']): row for row in eligible_meta}
    disk_complete_ids = set(complete.keys())

    on_disk_and_eligible = [pid for pid in eligible_ids if pid in disk_complete_ids]
    eligible_not_on_disk = [pid for pid in eligible_ids if pid not in disk_complete_ids]
    on_disk_not_eligible_grade4reqfields = [pid for pid in disk_complete_ids if pid not in eligible_ids]

    print()
    print("=== CROSS: metadata-eligible (grade4+fields) AND disk-complete ===")
    print("count:", len(on_disk_and_eligible))
    print("eligible by metadata but NOT complete on disk:", len(eligible_not_on_disk))
    print("disk-complete but not metadata-eligible (grade!=4 or missing field or FU):", len(on_disk_not_eligible_grade4reqfields))

    # events
    events = sum(1 for pid in on_disk_and_eligible if str(eligible_ids[pid]['1-dead 0-alive']).strip() == '1')
    print("events (1-dead 0-alive==1) among on_disk_and_eligible:", events)

    # IDH breakdown for this cohort (since IDH filter removed, check variance)
    from collections import Counter
    idh_counts = Counter(eligible_ids[pid]['IDH'].strip() for pid in on_disk_and_eligible)
    print("IDH breakdown:", dict(idh_counts))

    # save frozen list
    out_dir = os.path.join(ROOT, "gbm-aid mert", "artifacts", "week3", "ucsf_cohort")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "ucsf_frozen_cohort_2026-08-18_v2.csv")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'folder', 't1c_file', 'seg_file'] + required_fields)
        for pid in sorted(on_disk_and_eligible):
            row = eligible_ids[pid]
            w.writerow([pid, complete[pid]['folder'], complete[pid]['t1c'], complete[pid]['seg']] +
                        [row[fld] for fld in required_fields])
    print()
    print("frozen list written:", out_csv, "rows:", len(on_disk_and_eligible))

    # sha256 of the file
    h = hashlib.sha256()
    with open(out_csv, 'rb') as f:
        h.update(f.read())
    print("sha256:", h.hexdigest())

    # also dump incomplete/empty pid lists for reference
    print()
    print("incomplete keys sample:", [x[0] for x in incomplete][:20])

if __name__ == "__main__":
    main()
