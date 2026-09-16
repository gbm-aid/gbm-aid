"""T3 Takip Modülü — delta metriklerinin RANO-benzeri kalibrasyonu (Hafta 4, Görev 4).

`pipeline/followup_t3.py`'deki fonksiyonları GERÇEK DB verisiyle çalıştırır.
DB'ye **HİÇBİR YAZMA YAPMAZ** (yalnız `readonly=True` SELECT). Çıktı, iki
ardışık LUMIERE ziyareti arasındaki hacim/yüzey/şekil/texture delta'larının
RANO grubu bazında medyan ± IQR eşik tablosudur.

⚠️ Bu modülün ürettiği sınıflandırma **"RANO-benzeri volumetrik yaklaşım"**dır
— resmi RANO'nun birebir otomasyonu DEĞİLDİR (bkz. `pipeline/followup_t3.py`
modül docstring'i).

Çıktı `artifacts/week4/followup_t3_v2/`'ye yazılır (v1
`artifacts/week4/followup_t3/`'ın ÜZERİNE YAZILMAZ — Codex
task-mszut9wy-qx8sal düzeltmeleri, 2026-08-19).

Kullanım:
    python tools/run_followup_t3_calibration.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from db_connection import get_connection  # noqa: E402
from pipeline.followup_t3 import (  # noqa: E402
    build_delta_rano_table,
    build_patient_visit_deltas,
    build_visit_match_report,
    pair_lumiere_visits_with_rano,
    summarize_excluded_transitions,
)
from pipeline.growth_simulation import uninterpretable_group_names  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "week4" / "followup_t3_v2"

# CLAUDE.md ruhuna uygun beyan -- HIGH3, `pipeline/followup_t3.py` modül
# docstring'indeki "CR-WT hacim uyumsuzluğu" bölümüyle BİREBİR aynı metin.
CR_WT_MISMATCH_DISCLAIMER = (
    "LUMIERE üzerinde uzman RANO etiketleri ile WT hacim değişimi arasında "
    "güçlü ve monoton bir ilişki gözlenmemiştir. 26 CR etiketli geçişte WT "
    "delta medyanı -%1,2; dağılım hem belirgin küçülme hem büyüme "
    "içermiştir. Bu, resmi RANO'nun WT hacmi dışında kalan bileşenleriyle "
    "(ölçülebilir kontrast tutan lezyon, 2B çap, yeni lezyon, steroid, "
    "klinik durum) uyumludur."
)

# v45.txt Bölüm 7'nin ÖRNEK eşikleri -- doğrulanmamış, mimari dokümandaki
# ilüstratif değerler. Bu script'in ürettiği tablo bunları DOĞRULAMAK için
# gerçek veriden yeniden ölçer -- örnek eşikler girdi olarak KULLANILMAZ.
ARCHITECTURE_EXAMPLE_THRESHOLDS = {
    "CR": {"volume_pct": "< -65%"},
    "PR": {"surface_pct": "< -30%"},
    "SD": {"sphericity_delta_pct_points": "±5 puan"},
    "PD": {"glcm_entropy_pct": "> +25%"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-group-n", type=int, default=5)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_connection(readonly=True)
    try:
        print("LUMIERE ziyaret <-> RANO eşleştirmesi okunuyor (SADECE SELECT)...")
        paired_df = pair_lumiere_visits_with_rano(conn)
        # --- MEDIUM5: eşleşme denetim izi (matched + unmatched) ARTIFACT'ı --
        match_report_df = build_visit_match_report(conn)
    finally:
        conn.close()

    n_unique = int((paired_df["match_type"] == "unique").sum())
    n_heuristic = int((paired_df["match_type"] == "heuristic_paired").sum())
    print(
        f"  {len(paired_df)} eşleşme ({n_unique} kesin/'unique', "
        f"{n_heuristic} sezgisel/'heuristic_paired' -- bkz. modül docstring'i "
        "'DOĞRULANMAMIŞ sezgisel' uyarısı)."
    )

    n_matched = int((match_report_df["match_type"] != "unmatched").sum())
    n_unmatched = int((match_report_df["match_type"] == "unmatched").sum())
    print(
        f"\n  Eşleştirme denetim izi: {len(match_report_df)} satır "
        f"({n_matched} eşleşti, {n_unmatched} eşleşemedi -- bkz. "
        "'unmatch_reason' sütunu)."
    )
    if n_unmatched:
        print(
            match_report_df.loc[
                match_report_df["match_type"] == "unmatched", "unmatch_reason"
            ]
            .value_counts()
            .to_string()
        )
    match_report_csv = OUTPUT_DIR / "visit_rano_match_report_v2.csv"
    match_report_df.to_csv(match_report_csv, index=False)
    print(f"  Yazıldı: {match_report_csv}")

    deltas_df = build_patient_visit_deltas(paired_df)
    print(f"\n  {len(deltas_df)} ardışık-ziyaret delta'sı (Pre-Op/Post-Op geçişleri HARİÇ).")
    if not deltas_df.empty:
        print(deltas_df["curr_rano_label"].value_counts().to_string())

    deltas_csv = OUTPUT_DIR / "patient_visit_deltas_v2.csv"
    deltas_df.to_csv(deltas_csv, index=False)
    print(f"  Yazıldı: {deltas_csv}")

    # --- MEDIUM4: SESSİZCE atlanan geçişler AÇIKÇA raporlanır -------------
    excluded_df = summarize_excluded_transitions(paired_df)
    if not excluded_df.empty:
        print(f"\n  {len(excluded_df)} geçiş delta hesabına dahil edilmedi:")
        print(excluded_df["exclusion_reason"].value_counts().to_string())
        unknown_excluded = excluded_df[excluded_df["exclusion_reason"] == "unknown_or_compound_label"]
        if not unknown_excluded.empty:
            print(
                "  UYARI -- bilinmeyen/bileşik RANO etiketi (PD'ye ya da başka bir "
                "kategoriye SESSİZCE dönüştürülmedi):"
            )
            print(unknown_excluded.to_string(index=False))
    excluded_csv = OUTPUT_DIR / "excluded_transitions_v2.csv"
    excluded_df.to_csv(excluded_csv, index=False)
    print(f"  Yazıldı: {excluded_csv}")

    print("\n=== Görev 4: RANO-benzeri delta eşik tablosu (medyan ± IQR) ===")
    table = build_delta_rano_table(deltas_df, min_group_n=args.min_group_n)
    print(table.to_string(index=False) if not table.empty else "  (boş)")
    bad_groups = uninterpretable_group_names(table, group_col="rano_label")
    if bad_groups:
        print(
            f"  UYARI -- YORUMLANAMAZ (n < min_group_n={args.min_group_n}): "
            f"{bad_groups}. Bu gruplarla (veya bunları içeren HERHANGİ bir "
            "ikili karşılaştırmayla) yön/karşılaştırma iddiası KURULAMAZ "
            "(Codex HIGH2)."
        )

    table_csv = OUTPUT_DIR / "delta_rano_threshold_table_v2.csv"
    table.to_csv(table_csv, index=False)
    print(f"  Yazıldı: {table_csv}")

    if not table.empty and "CR" in table["rano_label"].astype(str).values:
        print("\n=== HIGH3 — CR-WT hacim uyumsuzluğu (gerçek veri bulgusu) ===")
        print(f"  {CR_WT_MISMATCH_DISCLAIMER}")
        print(
            "  UYARI -- WT-tabanlı eşikler 'resmi RANO otomasyonu' olarak SUNULMAZ; "
            "yalnızca 'RANO-benzeri volumetrik yaklaşımın yöntem sınırı' "
            "olarak etiketlenir."
        )

    print(
        "\nKarşılaştırma (mimari örnek eşikleri, v45.txt Bölüm 7 -- "
        "DOĞRULAMA amaçlı, girdi olarak KULLANILMADI, ÜRETİM EŞİĞİ OLARAK "
        "TÜRETİLMEDİ):"
    )
    for label, hints in ARCHITECTURE_EXAMPLE_THRESHOLDS.items():
        print(f"  {label}: {hints}")

    print(
        "\nUYARI: Bu tablo 'RANO-benzeri volumetrik yaklaşım'dır -- resmi RANO'nun "
        "hacim-dışı bileşenlerini (yeni lezyon, steroid dozu, klinik durum) "
        "İÇERMEZ (v45.txt satır 1257-1265)."
    )
    print(f"\nÇıktı klasörü: {OUTPUT_DIR}")
    print("DB'ye HİÇBİR YAZMA yapılmadı (readonly=True, yalnız SELECT).")
    print(
        "v1 çıktıları (artifacts/week4/followup_t3/) DEĞİŞTİRİLMEDİ -- bu, "
        "Codex task-mszut9wy-qx8sal düzeltmelerini içeren v2 koşusudur."
    )


if __name__ == "__main__":
    main()
