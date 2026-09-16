"""`model_registry`'ye XGBoost nüks/progresyon modelinin 4. satırını YAZAR.

Kaynak (SABİT KODLANMADI, tamamı dosyadan OKUNUR):
  artifacts/week4/xgboost_v2a_mgmt_reduce_collinearity_0912/
    week4_v2a_mgmt_aligned_run_metadata.json
    week4_v2a_mgmt_ucsf_external_evaluation.json
    week4_v2a_mgmt_aligned_fold_results.csv
    week4_v2a_mgmt_aligned_fold_leakage_audit.csv
    week4_v2a_mgmt_aligned_collinearity_filter_summary.csv

Bu script `write_model_registry.py` (cox_phm) ile AYNI dry-run/apply/
idempotent/canlı-doğrulama desenini izler. `write_model_registry.py`nin
JSON-artifact-okuma yaklaşımından FARKI: cox_phm scripti week3 CSV
şemasına göre parametrik `_load_metrics()` kullanıyordu; burada tek bir
sabit varyant/koşu var (2026-09-12 16:50'de bitti), o yüzden yol
parametrik BIRAKILMADI ama HİÇBİR SAYI elle yazılmadı -- hepsi
yukarıdaki JSON/CSV dosyalarından okunuyor.

Sadakat kanıtı (v2a_mgmt+reduce_collinearity -> v3b_lowvar_v2amgmt
reçetesiyle bit-birebir): bu koşunun kendi çıktı dosyalarında SAYISAL
OLARAK bulunmuyor (o kanıt ayrı bir tek-kullanımlık script'in çıktısıydı,
bkz. log/2026-09-12.md modeling-agent-A girişi, "satır 32" -- 16/16
özellik aynı, harici Harrell C farkı 0.000000, 24/24 katsayı farkı
0.000000). Bu yüzden bu değerler burada METIN olarak (kaynak
belirtilerek) taşınıyor, sayısal bir dosyadan tekrar OKUNMUYOR --
metrics.cox_recipe_fidelity_to_v3b.source alanında açıkça işaretli.

DAVRANIŞ:
  - Varsayılan: DRY-RUN -- hiçbir yazma yapılmaz.
  - `--apply`: gerçek INSERT (görev talimatı zaten "xgboost satırını yaz"
    diyor -- bu görevin kendisi onaydır, AKTIF-GOREVLER.md satırı
    db-agent-C için ÇALIŞIYOR olarak açık).
  - Idempotent: (model_name, version) zaten varsa INSERT ATLANIR.

KİLİTLİ (görev talimatı + CLAUDE.md):
  model_name='xgboost'. status='shadow' -- `api/analyze_patient.py`
  XGBoost'u canlı serve ETMİYOR (`available: False` sabit, satır
  ~123/133/157/197/200-201), bu yüzden 'production' YANLIŞ olur.

Kullanım:
  python tools/write_model_registry_xgboost.py
  python tools/write_model_registry_xgboost.py --apply   # yazım
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # gbm-aid mert/tools -> gbm-aid mert
sys.path.insert(0, str(PROJECT_ROOT))

from db_connection import get_connection  # noqa: E402

ARTIFACT_DIR = (
    PROJECT_ROOT / "artifacts" / "week4" / "xgboost_v2a_mgmt_reduce_collinearity_0912"
)
RUN_METADATA_PATH = ARTIFACT_DIR / "week4_v2a_mgmt_aligned_run_metadata.json"
UCSF_EVAL_PATH = ARTIFACT_DIR / "week4_v2a_mgmt_ucsf_external_evaluation.json"
FOLD_RESULTS_PATH = ARTIFACT_DIR / "week4_v2a_mgmt_aligned_fold_results.csv"
FOLD_LEAKAGE_AUDIT_PATH = ARTIFACT_DIR / "week4_v2a_mgmt_aligned_fold_leakage_audit.csv"
COLLINEARITY_SUMMARY_PATH = (
    ARTIFACT_DIR / "week4_v2a_mgmt_aligned_collinearity_filter_summary.csv"
)

MODEL_NAME = "xgboost"
# api/analyze_patient.py XGBoost'u bilinçli olarak bağlamıyor
# (available:False sabit) -- 'production' bu kanıtla KURULAMAZ.
STATUS = "shadow"


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_leakage_audit() -> dict:
    rows = _read_csv_rows(FOLD_LEAKAGE_AUDIT_PATH)
    all_true_equals = all(r["cox_fit_equals_outer_train"] == "True" for r in rows)
    all_true_disjoint = all(r["cox_fit_disjoint_from_outer_test"] == "True" for r in rows)
    return {
        "n_folds_audited": len(rows),
        "all_folds_cox_fit_equals_outer_train": all_true_equals,
        "all_folds_cox_fit_disjoint_from_outer_test": all_true_disjoint,
        "per_fold": [
            {
                "fold": int(r["fold"]),
                "n_outer_train": int(r["n_outer_train"]),
                "n_outer_test": int(r["n_outer_test"]),
                "cox_fit_equals_outer_train": r["cox_fit_equals_outer_train"] == "True",
                "cox_fit_disjoint_from_outer_test": r["cox_fit_disjoint_from_outer_test"] == "True",
                "cox_candidate_features_before_collinearity_filter": int(
                    r["cox_candidate_features_before_collinearity_filter"]
                ),
                "cox_candidate_features_after_collinearity_filter": int(
                    r["cox_candidate_features_after_collinearity_filter"]
                ),
            }
            for r in rows
        ],
        "verified_by": (
            "pipeline.xgboost_model.verify_aligned_fold_leakage_free() -- script "
            "kendisi çağırır, başarısız olsa RuntimeError ile durur (bkz. "
            "run_metadata.json::structural_leakage_guarantee_note)."
        ),
    }


def _load_fold_results() -> dict:
    rows = _read_csv_rows(FOLD_RESULTS_PATH)
    n_final_features = [int(r["n_cox_final_features"]) for r in rows]
    return {
        "n_outer_folds": len(rows),
        "cox_final_feature_count_per_fold": n_final_features,
        "cox_final_feature_count_note": (
            "Fold-başına seçilen Cox özellik sayısı OYNAK (nested-CV özellik "
            "seçimi her fold içinde bağımsız yapıldığı için beklenen davranış, "
            "sızıntı belirtisi DEĞİL) -- bkz. fold_leakage_audit."
        ),
        "best_xgb_inner_auc_per_fold": [float(r["best_xgb_inner_auc"]) for r in rows],
        "best_cox_penalizer_per_fold": [float(r["best_cox_penalizer"]) for r in rows],
        "best_cox_l1_ratio_per_fold": [float(r["best_cox_l1_ratio"]) for r in rows],
        "best_xgb_params_per_fold": [r["best_xgb_params"] for r in rows],
    }


def _load_collinearity_summary() -> dict:
    rows = _read_csv_rows(COLLINEARITY_SUMMARY_PATH)
    outer_final_rows = [r for r in rows if r["scope"] == "outer_final"]
    return {
        "cv_threshold": float(outer_final_rows[0]["cv_threshold"]),
        "corr_threshold": float(outer_final_rows[0]["corr_threshold"]),
        "outer_final_n_input_features": sorted(
            {int(r["n_input_features"]) for r in outer_final_rows}
        ),
        "outer_final_n_kept_features": sorted(
            {int(r["n_kept_features"]) for r in outer_final_rows}
        ),
        "note": (
            "93 aday radyomik özellik (WT-only birincil havuz) her outer-fold'da "
            "near-constant + korelasyon filtresiyle 54'e indirildi (5/5 outer_final "
            "satırında 93->54) -- bu sayı v3b_lowvar_v2amgmt Cox reçetesiyle AYNI "
            "filtre eşikleri (cv=0.02, corr=0.95). 93 ile PyRadiomics 107/bölge veya "
            "Cox eğitim havuzu 611/585 KARIŞTIRILMAMALI -- bu üçü FARKLI sayılardır."
        ),
    }


def _build_metrics(run_meta: dict, ucsf_eval: dict) -> dict:
    label_report = run_meta["twelve_month_label_report"]
    oof = run_meta["xgboost_auc_out_of_fold"]
    ucsf_auc = ucsf_eval["ucsf_auc"]
    ucsf_label_report = ucsf_eval["ucsf_twelve_month_report"]

    return {
        "variant": run_meta["variant"],
        "pipeline": run_meta["pipeline"],
        "reduce_collinearity": bool(run_meta["cli_args"]["reduce_collinearity"]),
        "collinearity_cv_threshold": run_meta["cli_args"]["collinearity_cv_threshold"],
        "collinearity_corr_threshold": run_meta["cli_args"]["collinearity_corr_threshold"],
        "collinearity_filter_summary": _load_collinearity_summary(),
        "cox_recipe_fidelity_to_v3b": {
            "claim": (
                "v2a_mgmt + --reduce-collinearity (aligned pipeline) reçetesiyle "
                "üretilen Cox meta-skoru, resmi v3b_lowvar_v2amgmt fit'iyle "
                "BİT-BİREBİR eşleşti: 16/16 radyomik özellik AYNI, harici Harrell "
                "C farkı 0.000000 (0.6591716890743317), 24/24 ortak katsayı "
                "(16 radyomik + 8 klinik) farkı 0.000000."
            ),
            "source": (
                "log/2026-09-12.md, [modeling-agent-A] girişi ('XGBoost v3b "
                "reçete-sadakati DÜZELTİLDİ (bit-birebir kanıtlı)') -- bu sayısal "
                "kanıt AYRI, tek-kullanımlık bir sadakat script'inin çıktısıdır "
                "(scratchpad'te, kalıcı dosya değil); bu koşunun kendi "
                "run_metadata.json/fold_results.csv dosyalarında TEKRAR "
                "ÜRETİLMEMİŞTİR -- bu alan metin olarak kaynak-atıflı taşınıyor, "
                "sayısal bir dosyadan otomatik OKUNMADI."
            ),
            "independent_cross_checks": (
                "Aynı 0.659172 harici Harrell-C rakamı modeling-agent-B'nin "
                "TAMAMEN BAĞIMSIZ export_v3b_deployment_checkpoint.py'siyle VE "
                "modeling-agent-D'nin evaluate_v3b_calibration.py doğrulama "
                "kapısıyla (fark 0.000e+00, katsayı kimliği maks fark 1.110e-16) "
                "ayrı ayrı yeniden üretildi -- üç bağımsız doğrulama."
            ),
        },
        "twelve_month_label_definition": {
            "threshold_days": label_report["threshold_days"],
            "threshold_days_note": (
                "365.0 kullanıldı (365.25 DEĞİL) -- duyarlılık karşılaştırması "
                "YAPILMADI, run_metadata.json::known_open_risks[0]'da açık risk "
                "olarak kayıtlı."
            ),
        },
        "internal_oof_auc": {
            "auc": oof["auc"],
            "ci_lower": oof["ci_lower"],
            "ci_upper": oof["ci_upper"],
            "confidence_level": oof["confidence_level"],
            "n_patients": oof["n_patients"],
            "n_positive": oof["n_positive"],
            "n_bootstrap_valid": oof["n_bootstrap_valid"],
            "n_bootstrap_requested": oof["n_bootstrap_requested"],
            "method_note": (
                "Out-of-fold (nested-CV dış fold) AUC -- in-sample DEĞİL. "
                "n_patients=606 (611 UPenn eğitim havuzundan 5 hasta 12-ay "
                "ufkunda belirsiz sansür nedeniyle 12-ay etiketinden "
                "DIŞLANDI, bkz. twelve_month_label_report)."
            ),
        },
        "twelve_month_label_report_upenn_training": {
            "n_total": label_report["n_total"],
            "n_survived_yes": label_report["n_survived_yes"],
            "n_died_no": label_report["n_died_no"],
            "n_excluded_ambiguous_censoring": label_report["n_excluded_ambiguous_censoring"],
            "pct_excluded": label_report["pct_excluded"],
        },
        "ucsf_external_auc": {
            "auc": ucsf_auc["auc"],
            "ci_lower": ucsf_auc["ci_lower"],
            "ci_upper": ucsf_auc["ci_upper"],
            "confidence_level": ucsf_auc["confidence_level"],
            "n_patients_evaluated": ucsf_auc["n_patients"],
            "n_positive": ucsf_auc["n_positive"],
            "n_bootstrap_valid": ucsf_auc["n_bootstrap_valid"],
            "n_bootstrap_requested": ucsf_auc["n_bootstrap_requested"],
            "n_excluded_ambiguous_censoring": ucsf_label_report["n_excluded_ambiguous_censoring"],
            "pct_excluded": ucsf_label_report["pct_excluded"],
            "n_ucsf_cohort_total": ucsf_label_report["n_total"],
            "coverage_note": (
                "⚠️ Bu AUC UCSF'in TAMAMI (295 hasta) üzerinde DEĞİL -- 63 hasta "
                "12-ay ufkunda belirsiz sansür nedeniyle dışlandı (%21,36), "
                "kalan n=232 üzerinde ölçüldü. 'UCSF harici AUC 0,7237' derken "
                "bu kapsam daralması HER ZAMAN birlikte belirtilmelidir, "
                "gizlenmez (CLAUDE.md tek-eşik/şeffaflık kuralı)."
            ),
            "reweighting_note": (
                "Yeniden-ağırlıklandırma/IPCW UYGULANMADI -- ambiguous-censoring "
                "satırları basitçe DIŞLANDI, dışlama mekanizması (sağkalım süresi "
                "12 aydan az VE hayatta) rastgele DEĞİL, bu potansiyel bir "
                "yanlılık kaynağıdır ama nicel olarak ÖLÇÜLMEDİ."
            ),
        },
        "structural_leakage_guarantee": _load_leakage_audit(),
        "fold_results": _load_fold_results(),
        "risk_thresholds_frozen_at_training": run_meta["risk_thresholds"],
        "reporting_rule_note": run_meta["reporting_rule_note"],
        "known_open_risks": run_meta["known_open_risks"],
        "service_status_note": (
            "XGBoost bu kayıt anında CANLI API'de SERVİS EDİLMİYOR -- "
            "api/analyze_patient.py XGBoost nüks/progresyon bloğunu bilinçli "
            "olarak bağlamıyor, `available: False` + `not_available_reason` "
            "HER ZAMAN sabit dönüyor (satır ~123/133/157/197/200-201). Bu "
            "yüzden status='production' KURULAMAZ; 'shadow' seçildi (mimari "
            "v4.5 §6.5 ile aynı desen: yeni model doğrudan production'a geçmez, "
            "ayrıca burada zaten hiç bağlanmamış)."
        ),
        "external_test_source": "UCSF-PDGM (K15/2026-08-18 revizyonu, TCGA DEĞİL)",
        "cox_meta_score_upstream": (
            "XGBoost'un girdisi olan Cox meta-skoru HER outer-fold'da (nested-CV) "
            "yalnız o fold'un eğitim hastalarıyla fit edilir -- resmi v3b "
            "checkpoint'i (models/cox_phm_v3b_lowvar_v2amgmt_2026-09-12.pkl) "
            "DOĞRUDAN kullanılmaz, onunla reçete-bit-birebir bir Cox fit'i her "
            "fold içinde YENİDEN üretilir (sızıntı önleme)."
        ),
        "run_metadata_generated_at_utc": run_meta["generated_at_utc"],
        "ucsf_eval_generated_at_utc": ucsf_eval["generated_at_utc"],
        "script_path": run_meta["script_path"],
        "script_sha256": run_meta["script_sha256"],
        "cli_args": run_meta["cli_args"],
        "source_files": [
            str(RUN_METADATA_PATH),
            str(UCSF_EVAL_PATH),
            str(FOLD_RESULTS_PATH),
            str(FOLD_LEAKAGE_AUDIT_PATH),
            str(COLLINEARITY_SUMMARY_PATH),
        ],
        "compiled_by": (
            "db-agent-C, 2026-09-12 -- tools/write_model_registry_xgboost.py, "
            "tamamı yukarıdaki JSON/CSV artifact dosyalarından OTOMATİK okundu "
            "(cox_recipe_fidelity_to_v3b hariç -- o alan kaynak-atıflı METİN, "
            "gerekçesi kendi alanında yazılı)."
        ),
    }


def _build_insert_row() -> dict:
    if not RUN_METADATA_PATH.is_file():
        raise FileNotFoundError(f"beklenen kaynak dosya yok: {RUN_METADATA_PATH}")
    if not UCSF_EVAL_PATH.is_file():
        raise FileNotFoundError(f"beklenen kaynak dosya yok: {UCSF_EVAL_PATH}")

    run_meta = json.loads(RUN_METADATA_PATH.read_text(encoding="utf-8"))
    ucsf_eval = json.loads(UCSF_EVAL_PATH.read_text(encoding="utf-8"))

    metrics = _build_metrics(run_meta, ucsf_eval)
    training_date = run_meta["generated_at_utc"].split("T", 1)[0]

    version = "v2a_mgmt+reduce_collinearity-aligned-v3b_bitidentical_2026-09-12"

    training_cohort_snapshot = (
        "XGBoost eğitim/değerlendirme havuzu: UPenn-GBM 611 hasta / 585 ölüm "
        "olayı (Cox tarafının kilitli havuzu, kayıtlı 630/603 DEĞİL) -- bunun "
        "606/341'i (5 hasta 12-ay ufkunda belirsiz sansür nedeniyle dışlandı) "
        "12-ay ikili nüks/progresyon hedefiyle nested 5-fold dış CV'de "
        "kullanıldı, n_positive(12-ay hayatta)=341. Her outer-fold'un Cox "
        "meta-skoru YALNIZ o fold'un eğitim hastalarıyla fit edildi (bkz. "
        "metrics.structural_leakage_guarantee, 5/5 fold doğrulandı). Harici "
        "test: UCSF-PDGM 295 hasta / 169 ölüm olayının 232'si (63 hasta "
        "belirsiz sansür nedeniyle dışlandı, bkz. metrics.ucsf_external_auc). "
        "Radyomik aday havuzu: WT-only 93 özellik (ICC>=0.60 stabilite "
        "filtresi), her fold içinde near-constant+korelasyon filtresiyle "
        "54'e indirildi (v3b Cox reçetesiyle aynı eşikler). Cox meta-skor "
        "reçetesi v3b_lowvar_v2amgmt ile bit-birebir sadakati kanıtlanmış "
        "(bkz. metrics.cox_recipe_fidelity_to_v3b) -- bu XGBoost'un HANGİ "
        "Cox modelinin skorlarıyla eğitildiği sorusunun cevabıdır."
    )

    return {
        "model_name": MODEL_NAME,
        "version": version,
        "training_date": training_date,
        "training_cohort_snapshot": training_cohort_snapshot,
        "metrics": metrics,
        "status": STATUS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply", action="store_true",
        help="gerçek INSERT çalıştır (varsayılan: dry-run)",
    )
    args = parser.parse_args()

    row = _build_insert_row()

    print("=" * 70)
    print(f"model_name={row['model_name']!r} version={row['version']!r} status={row['status']!r}")
    print(f"training_date={row['training_date']!r}")
    print("-" * 70)
    print("training_cohort_snapshot:")
    print(row["training_cohort_snapshot"])
    print("-" * 70)
    print("metrics (JSONB):")
    print(json.dumps(row["metrics"], indent=2, ensure_ascii=False))
    print("=" * 70)

    conn = get_connection(readonly=not args.apply)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM model_registry WHERE model_name = %s AND version = %s",
            (row["model_name"], row["version"]),
        )
        already_exists = cur.fetchone() is not None
        cur.close()
    finally:
        conn.close()

    if already_exists:
        print(
            f"\n[BILGI] ({row['model_name']!r}, {row['version']!r}) zaten "
            "model_registry'de VAR -- INSERT ATLANACAK (idempotent)."
        )

    if not args.apply:
        print("\n[DRY-RUN] --apply verilmedi, hiçbir yazma yapılmadı.")
        return 0

    if already_exists:
        print("[APPLY] Yazılacak yeni satır yok (zaten var), işlem YOK.")
        return 0

    conn = get_connection(readonly=False)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO model_registry
                (model_name, version, training_date, training_cohort_snapshot, metrics, status)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING model_id
            """,
            (
                row["model_name"],
                row["version"],
                row["training_date"],
                row["training_cohort_snapshot"],
                json.dumps(row["metrics"], ensure_ascii=False),
                row["status"],
            ),
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        print(f"\n[APPLY] COMMIT edildi. model_id={new_id} yazıldı.")
    except Exception:
        conn.rollback()
        print("\n[APPLY] HATA -- ROLLBACK edildi.", file=sys.stderr)
        raise
    finally:
        cur.close()
        conn.close()

    # Canlı doğrulama -- AYRI, salt-okunur bağlantı
    verify_conn = get_connection(readonly=True)
    try:
        verify_cur = verify_conn.cursor()
        verify_cur.execute(
            "SELECT model_id, model_name, version, status, training_date "
            "FROM model_registry WHERE model_name = %s AND version = %s",
            (row["model_name"], row["version"]),
        )
        print(f"[DOGRULAMA] canlı SELECT: {verify_cur.fetchone()}")
        verify_cur.close()
    finally:
        verify_conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
