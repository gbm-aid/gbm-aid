"""pipeline.cox_model için SMOKE TEST -- gerçek Hafta 3 eğitimi DEĞİL.

Amaç: `_assemble_training_frame`/`select_features_lasso`/`run_stratified_cv`/
`evaluate_external_test`'in GERÇEK Supabase verisiyle (sentetik değil)
uçtan uca çalıştığını kanıtlamak.

NEDEN TCGA KULLANILIYOR (ve bu bir mimari ihlal DEĞİL):
Bu script'in yazıldığı tarihte (2026-08-09) gerçek UPenn+LUMIERE eğitim
havuzu henüz KULLANILAMAZ:
  - LUMIERE'in `patients.vital_status` alanı TAMAMEN NULL (91/91,
    canlı SELECT ile doğrulandı) -- event bilgisi yok.
  - UPenn'de 27 hastanın `survival_days`'i NULL, 10 hastanın
    `vital_status`'ü normalize edilmemiş (bkz.
    decisions/2026-07-19-tcga-mgmt-idh1-survival-days-acik.md) --
    db-agent bu satırı bu script yazıldığı sırada düzeltiyordu
    (AKTIF-GOREVLER.md, ÇALIŞIYOR).
  - `combat_parameters` hâlâ 0 satır (Blokaj 2 açık, LUMIERE'in
    PyRadiomics'i yeniden çıkarılıyor).

TCGA'nın KENDİSİ ise gerçek `vital_status`/`survival_days`'e sahip
(260 hastanın 259'u kullanılabilir) ve zaten mimari olarak EĞİTİME
GİRMEYECEK bir kaynak -- yani TCGA verisiyle burada yapılan
train/test/CV bölmesi SADECE KOD MEKANİĞİNİ (DB->frame->LASSO->CV->
external-eval->SHAP) test eder, üretilen hiçbir model/parametre
GERÇEK Cox modeli olarak KULLANILMAZ, kaydedilmez, `model_registry`'ye
YAZILMAZ. Gerçek eğitim UPenn+LUMIERE hazır olunca ayrı bir script ile
yapılacak.

NEDEN `pipeline.cox_model.build_training_frame()` DEĞİL, DOĞRUDAN
(ARTIK PRIVATE) `_assemble_training_frame()` KULLANILIYOR (2026-08-13,
bilinçli/belgeli istisna -- bkz. `pipeline.cox_model.build_training_
frame()` ve `_assemble_training_frame()`'in kendi docstring'lerindeki
"ÖNEMLİ İSTİSNA" notu): `build_training_frame()` artık `assert_
training_pool_sources()`'ı (whitelist guard'ı) ZORUNLU olarak önce
çalıştırıyor ve TCGA'yı HER koşulda reddediyor -- bu script'in yukarıda
açıklanan BİLİNÇLİ TCGA-ile-mekanik-test amacını da engellerdi. Bu
yüzden bu script `build_training_frame()`'i DEĞİL, guard'sız
`_assemble_training_frame()`'i doğrudan çağırmaya devam ediyor -- bu
bir gözden kaçırma DEĞİL, kayıtlı bir istisna. Gerçek Hafta 3 eğitim
script'i (UPenn+LUMIERE ile) bu script'in AKSİNE HER ZAMAN
`build_training_frame()` kullanmalı.

NEDEN PRIVATE (alt çizgi ön-ekli) BİR FONKSİYONA DOĞRUDAN ERİŞİLİYOR
(2026-08-13, Codex HIGH-1 sertleştirmesi -- bkz. decisions/2026-08-12-
cox-model-source-guard-canonicalization.md "[2026-08-13] TAKİP-3"):
`_assemble_training_frame()` eskiden (`assemble_training_frame()`
adıyla) public'ti, bu görev onu private yaptı -- bu script BUNU BİLE
BİLE, yukarıda açıklanan gerekçeyle (TCGA'yı `build_training_frame()`
reddettiği için) doğrudan private fonksiyona erişmeye devam ediyor.
Bu, "private isim demek Python'da erişilemez demek DEĞİL" gerçeğine
dayanan, kayıtlı/belgeli TEK bilinçli istisna -- gerçek Hafta 3 eğitim
script'i bunu YAPMAMALI.

Çalıştırma: yalnızca readonly SELECT, hiçbir DB yazımı yapmaz.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import pandas as pd
from psycopg2.extras import RealDictCursor

from db_connection import get_connection
from pipeline.cox_model import (
    _assemble_training_frame,
    compute_shap_values,
    evaluate_external_test,
    run_stratified_cv,
    select_features_lasso,
    verify_event_counts,
)

CANONICAL_KEY_COUNTS = {"shape": 14, "first_order": 18, "texture": 75}


def _fetch_tcga_wt_frame(cursor) -> tuple[pd.DataFrame, pd.DataFrame]:
    cursor.execute(
        """
        SELECT p.patient_id, p.vital_status, p.survival_days,
               r.shape_features, r.first_order_features, r.texture_features
        FROM radiomics r
        JOIN mr_scans ms ON ms.scan_id = r.scan_id
        JOIN patients p ON p.patient_id = ms.patient_id
        JOIN dataset_sources ds ON ds.source_id = p.source_id
        WHERE ds.source_name = 'TCGA-GBM'
          AND r.segmentation_tool = 'TCGA-ground-truth'
          AND r.tumor_region = 'WT'
        """
    )
    rows = cursor.fetchall()

    feature_records: dict[str, dict[str, float]] = {}
    patient_records: dict[str, dict[str, object]] = {}
    for row in rows:
        shape = row["shape_features"] or {}
        first_order = row["first_order_features"] or {}
        texture = row["texture_features"] or {}
        if (
            len(shape) != CANONICAL_KEY_COUNTS["shape"]
            or len(first_order) != CANONICAL_KEY_COUNTS["first_order"]
            or len(texture) != CANONICAL_KEY_COUNTS["texture"]
        ):
            continue
        merged = {**shape, **first_order, **texture}
        if any(v is None for v in merged.values()):
            continue
        pid = row["patient_id"]
        feature_records.setdefault(pid, merged)
        patient_records.setdefault(
            pid,
            {
                "source": "TCGA-GBM",
                "vital_status": row["vital_status"],
                "survival_days": row["survival_days"],
            },
        )

    feature_frame = pd.DataFrame.from_dict(feature_records, orient="index")
    feature_frame.index.name = "patient_id"
    patients_frame = pd.DataFrame.from_dict(patient_records, orient="index")
    patients_frame.index.name = "patient_id"
    return feature_frame.sort_index(), patients_frame.sort_index()


def main() -> int:
    connection = get_connection(readonly=True)
    try:
        print("=== verify_event_counts() -- gerçek DB, plan.txt madde 436 ===")
        event_counts = verify_event_counts(connection)
        print(event_counts.to_string())

        cursor = connection.cursor(cursor_factory=RealDictCursor)
        feature_frame, patients_frame = _fetch_tcga_wt_frame(cursor)
        cursor.close()
    finally:
        connection.close()

    print(f"\nTCGA/WT kanonik hasta sayısı: {len(feature_frame)}")
    if feature_frame.empty:
        print("HATA: TCGA kanonik verisi bulunamadı, smoke test yürütülemiyor.")
        return 1

    # _assemble_training_frame() TCGA-GBM'i "source_reference" olarak
    # kullanıyor -- bu SADECE bu smoke test'e özgü (TCGA tek kaynak
    # olduğu için dummy kolonu bile üretilmiyor), gerçek üretimde
    # source_reference her zaman "UPenn" olacak.
    frame, report = _assemble_training_frame(
        feature_frame, patients_frame, source_reference="TCGA-GBM"
    )
    print("\n=== _assemble_training_frame() raporu ===")
    print(f"n_input_rows={report.n_input_rows}, n_output_rows={report.n_output_rows}")
    print(f"dropped_missing_duration={report.dropped_missing_duration}")
    print(f"dropped_missing_features={report.dropped_missing_features}")
    print(f"dropped_unrecognized_vital_status={report.dropped_unrecognized_vital_status}")
    print(f"sources={report.sources}")

    feature_columns = list(feature_frame.columns)
    print(f"\nÖzellik sayısı: {len(feature_columns)} (beklenen 107)")

    # Basit mekanik train/test bölmesi (TCGA'yı TCGA ile test ediyoruz --
    # GERÇEK harici test DEĞİL, sadece kod yolu kanıtlanıyor).
    rng_cutoff = int(len(frame) * 0.8)
    shuffled = frame.sample(frac=1.0, random_state=42)
    train_frame = shuffled.iloc[:rng_cutoff]
    holdout_frame = shuffled.iloc[rng_cutoff:]

    print(
        f"\nMekanik train/holdout bölmesi: train={len(train_frame)}, "
        f"holdout={len(holdout_frame)} (GERÇEK harici test DEĞİL)"
    )

    print("\n=== select_features_lasso() ===")
    selected, cox, coefficients = select_features_lasso(
        train_frame, feature_columns, penalizer=0.5, l1_ratio=1.0
    )
    print(f"Seçilen özellik sayısı: {len(selected)} / {len(feature_columns)}")
    print("Seçilenler:", selected[:10], "..." if len(selected) > 10 else "")

    print("\n=== run_stratified_cv() (train_frame içinde, 5-fold) ===")
    cv_results = run_stratified_cv(
        train_frame, feature_columns, n_splits=5, penalizer=0.5, l1_ratio=1.0
    )
    print(cv_results.to_string())
    print(f"Ortalama C-index: {cv_results['c_index'].mean():.4f}")

    print("\n=== evaluate_external_test() (holdout_frame'e karşı, MEKANİK) ===")
    external_result = evaluate_external_test(
        cox, holdout_frame, feature_columns, n_bootstrap=500, seed=42
    )
    for key, value in external_result.items():
        print(f"  {key}: {value}")

    print(
        "\n=== compute_shap_values() -- KÜÇÜK bir alt-kümeyle (5 hasta), "
        "yalnız plumbing kanıtı ==="
    )
    shap_frame = compute_shap_values(
        cox, holdout_frame.head(5), feature_columns, n_background=20, seed=42
    )
    print(f"shap_frame shape: {shap_frame.shape} (beklenen (5, {len(feature_columns)}))")

    print(
        "\nSMOKE TEST TAMAMLANDI -- hiçbir DB yazımı yapılmadı, hiçbir model "
        "kaydedilmedi, bu GERÇEK bir Hafta 3 eğitimi DEĞİLDİR."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
