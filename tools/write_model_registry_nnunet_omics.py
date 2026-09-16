"""`model_registry`'ye `nnunet` ve `omics_scorer` kayitlarini HAZIRLAR/YAZAR.

`cox_phm` icin AYRI bir script var (`tools/write_model_registry.py`, artifact
CSV/JSON'lardan metrik okuyor) -- bu iki model o desene UYMUYOR: `nnunet`
uculcu parti bir pretrained model (bizim tarafimizdan EGITILMEDI), `omics_
scorer` ise kod-icindeki sabit formul/esiklerden olusuyor (egitim/CV
metrigi YOK). Bu yuzden metrikler burada, KAYNAK DOSYALARINA (docs/
nnunet_new_patient_demo.md, docs/hafta2_mert_ozet.md, pipeline/omics_scores.py,
decisions/2026-08-18-omics-skor-formulleri-ve-esikler.md) DAYANARAK, db-agent
tarafindan 2026-09-12'de elle DERLENDI -- HICBIR SAYI otomatik dosyadan
okunmuyor (cox_phm script'inin yaptigi gibi), bu yuzden her deger yaninda
KAYNAK belirtilir.

DAVRANIS (write_model_registry.py ile AYNI desen):
  - Varsayilan: DRY-RUN -- hicbir yazma yapilmaz.
  - `--apply`: gercek INSERT (Baris'in acik onayi/gorev talimati var --
    2026-09-12 gorev metni, "3 satir yazilacak" ZATEN talimat).
  - Idempotent: (model_name, version) zaten varsa INSERT ATLANIR.

Kullanim:
  python tools/write_model_registry_nnunet_omics.py --model nnunet
  python tools/write_model_registry_nnunet_omics.py --model omics_scorer --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # gbm-aid mert/tools -> gbm-aid mert
sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from db_connection import get_connection  # noqa: E402


def _build_nnunet_row() -> dict:
    """Kaynaklar: docs/nnunet_new_patient_demo.md, docs/hafta2_mert_ozet.md SS2.8/3.3,
    pipeline/new_patient_segmentation.py, pipeline/nnunet_runtime.py,
    tools/install_brats_pretrained_model.py."""
    metrics = {
        "source": (
            "BraTS 2021 ile egitildigi yayinci tarafindan belirtilen ucuncu "
            "parti pretrained nnU-Net v2 modeli -- BIZIM TARAFIMIZDAN "
            "EGITILMEDI/FINE-TUNE EDILMEDI (CLAUDE.md kurali: 'hazir "
            "maskeler + pretrained nnU-Net kullanilir, fine-tune YAPILMAZ')."
        ),
        "zenodo_doi": "10.5281/zenodo.11582627",
        "archive_file": "Dataset002_BRATS19.zip",
        "archive_size_bytes": 1155915349,
        "md5": "23a3f55dead4a6642271a08d1a503bbb",
        "nnunet_dataset_name": "Dataset002_BRATS19",
        "nnunet_version": "2.8.1",
        "configuration": "3d_fullres",
        "n_folds_checkpoints": 5,
        "channel_order": ["T1", "T1ce", "T2", "FLAIR"],
        "channel_suffix_map": {"_0000": "T1", "_0001": "T1ce", "_0002": "T2", "_0003": "FLAIR"},
        "allowed_output_labels": [0, 1, 2, 4],
        "label_3_note": (
            "Kurulu dataset.json'da 3=empty (gecerli tumor ciktisi SAYILMAZ) "
            "-- koşucu bu etiketi gorurse ACIKCA durur (bkz. pipeline/"
            "new_patient_segmentation.py)."
        ),
        "fine_tuned": False,
        "scope": (
            "SADECE new_patient_demo -- TCGA/UPenn/LUMIERE/UCSF-PDGM icin "
            "hazir segmentasyon maskeleri kullanilir (pipeline/segmentation.py), "
            "bu model o kohortlarda HICBIR ZAMAN calistirilmaz/calistirilmadi."
        ),
        "demo_gate_env_var": "GBMAID_ENABLE_NEW_PATIENT_NNUNET",
        "demo_gate_default": "0 (kapali)",
        "runtime_environment": {
            "python": "3.10.18",
            "pytorch": "2.6.0+cu126",
            "torchvision": "0.21.0+cu126",
            "nnunetv2": "2.8.1",
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB",
            "nvidia_driver": "566.14",
            "cuda": "12.6",
        },
        "preflight_validated_at": "2026-07-25",
        "preflight_result": (
            "PASS -- MD5 dogrulandi, kanal sirasi T1/T1ce/T2/FLAIR PASS, "
            "izinli etiketler 0/1/2/4 PASS, predictor executable bulundu "
            "(bkz. docs/hafta2_mert_ozet.md SS2.8, tools/nnunet_demo_preflight.py)."
        ),
        "real_inference_run": False,
        "dsc_measured": False,
        "dsc_production_target": 0.85,
        "dsc_note": (
            "Gercek nnU-Net inference'i HENUZ hicbir etiketli yeni-hasta "
            "demo vakasinda CALISTIRILMADI (2026-09-12 itibariyle, "
            "docs/hafta2_mert_ozet.md SS3.3 + Eylul loglarinda guncelleme "
            "bulunamadi) -- confidence_score=None donuyor, uydurma guven "
            "skoru YOK. DSC>=0.85 bir UeRETIM KABUL HEDEFIDIR, "
            "'ULASILDI' OLARAK RAPORLANAMAZ."
        ),
        "naming_inconsistency_note": (
            "Zenodo kayit adi 'BRATS19', yayin aciklamasi BraTS 2021'dir -- "
            "bu tutarsizlik dataset.json uzerinden kanal/etiket sozlesmesi "
            "BAGIMSIZ dogrulanarak ele alindi, GIZLENMEDI (docs/"
            "nnunet_new_patient_demo.md)."
        ),
        "source_files": [
            "pipeline/new_patient_segmentation.py",
            "pipeline/nnunet_runtime.py",
            "tools/install_brats_pretrained_model.py",
            "tools/nnunet_demo_preflight.py",
            "docs/nnunet_new_patient_demo.md",
        ],
        "compiled_by": "db-agent, 2026-09-12 -- elle derlendi, HICBIR SAYI "
                        "otomatik dosyadan okunmadi (yukaridaki kaynak "
                        "dosyalar elle okunup aktarildi).",
    }
    training_cohort_snapshot = (
        "YOK / UYGULANAMAZ -- bu model bizim tarafimizdan EGITILMEDI. "
        "Egitim verisi (BraTS 2021) ucuncu partiye ait, bizim kohortumuzdan "
        "(UPenn-GBM/LUMIERE/TCGA-GBM/UCSF-PDGM) HICBIRI bu modelin "
        "egitimine girmedi ve bu model o kohortlarda hic calistirilmadi -- "
        "TCGA/UPenn/LUMIERE/UCSF icin hazir maskeler kullanilir (pipeline/"
        "segmentation.py), fine-tune YAPILMAZ (CLAUDE.md kilitli kurali). "
        "Kapsam SADECE ayri 'yeni-hasta demo' akisidir (pipeline/"
        "new_patient_segmentation.py), varsayilan olarak KAPALI "
        "(GBMAID_ENABLE_NEW_PATIENT_NNUNET=0)."
    )
    return {
        "model_name": "nnunet",
        "version": "Dataset002_BRATS19-zenodo-11582627",
        # Bizim EGITTIGIMIZ bir tarih YOK -- pretrained modelin kendi
        # egitim tarihi bilinmiyor (yayinci beyani "BraTS 2021" YIL
        # seviyesinde, gun/ay yok). training_date SADECE bizim
        # kurulum/dogrulama tarihimizi (preflight) tasir, bu bir EGITIM
        # tarihi DEGILDIR -- metrics.preflight_validated_at ile
        # AYNI/tutarli tutuldu, karisiklik olmasin diye asagida ayrica
        # not edildi.
        "training_date": "2026-07-25",
        "training_date_note": (
            "Bu tarih modelin EGITIM tarihi DEGILDIR (ucuncu parti "
            "pretrained, bizim egitimimiz yok) -- bizim KURULUM+PREFLIGHT-"
            "DOGRULAMA tarihimizdir (metrics.preflight_validated_at ile "
            "birebir ayni)."
        ),
        "training_cohort_snapshot": training_cohort_snapshot,
        "metrics": metrics,
        # 'shadow' -- gate varsayilan KAPALI, gercek inference/DSC olcumu
        # yok; 'production' iddiasi bu kanitla KURULAMAZ.
        "status": "shadow",
    }


def _build_omics_scorer_row() -> dict:
    """Kaynaklar: decisions/2026-08-18-omics-skor-formulleri-ve-esikler.md,
    pipeline/omics_scores.py, canli DB (score_thresholds/molecular_scores/
    omics_profiles, db-agent 2026-09-12 SELECT)."""
    metrics = {
        "score_version": "v1.0",
        "cohort_snapshot_n": 48,
        "frozen_at_utc": "2026-08-18T13:38:47.989685+00:00",
        "live_verification_2026_09_12": {
            "omics_profiles_rows": 48,
            "omics_profiles_distinct_patients": 48,
            "molecular_scores_rows": 48,
            "molecular_scores_distinct_patients": 48,
            "score_thresholds_rows": 2,
        },
        "formulas": {
            "tmz_resistance_score": (
                "clip[0,100](50 - mgmt_met*25 + msh6_met*10 + mlh1_met*10 "
                "+ pms2_met*10 + max(0, 5-msh6_rna)*3)"
            ),
            "aggressiveness_score": (
                "clip[0,100](30 + min(egfr_rna,12)*2.5 + max(0,egfr_cnv)*8 "
                "+ pten_met*10 + max(0,-cdkn2a_cnv)*12 - tp53_met*5)"
            ),
            "dna_repair_score": (
                "z=((-MSH2_met*6.853 + min(MSH2_rna,10)*5.838) + "
                "(-MLH1_met*6.991 + min(MLH1_rna,10)*5.837) + "
                "(-BRCA1_met*8.752 + min(BRCA1_rna,10)*7.000) + "
                "(-BRCA2_met*8.766 + min(BRCA2_rna,10)*7.001) + "
                "(-CHEK2_met*0.000 + min(CHEK2_rna,10)*4.667) - 210.278 "
                f"- {50.007812}) / {15.000797}; score = z*15+50, clip[0,100]"
            ),
        },
        "dna_repair_score_note": (
            "Bu katsayilar, orijinal formul tanimi (gen listesi ve 'w' "
            "agirliklari) hicbir kaynakta bulunamadigi icin 48 hastanin "
            "mevcut skorlarindan TERSINE MUHENDISLIKLE cikarilmistir "
            "(R^2=0.99999997, maksimum sapma 0.0066/0.007 puan -- H2 "
            "hipotezi: met ve rna AYRI agirlik). Orijinal agirliklarla "
            "birebir ayni oldugu GARANTI EDILEMEZ. dna_repair_score "
            "EXPLORATORY statusundedir, ana karara (Cox/XGBoost) DAHIL "
            "EDILMEZ."
        ),
        "score_thresholds": {
            "tmz_resistance_score": {"p33": 46.5009, "p66": 48.3900},
            "dna_repair_score": {"p33": 41.9847, "p66": 59.0642},
        },
        "score_thresholds_note": (
            "aggressiveness_score BU TABLOYA GIRMEZ -- mimari (v45.txt "
            "satir 583-586) onu SABIT MUTLAK esikli (<45/65/80) tanimliyor, "
            "goreceli/tertil DEGIL."
        ),
        "aggr_class_categories": ["low", "intermediate", "high", "very_high"],
        "aggr_class_constraint_fix": (
            "molecular_scores_aggr_class_check DB constraint'i 'low' "
            "kategorisini EKSIK tanimliyordu (mimari 4 kategori diyordu, "
            "DB 3 kategoriyle constraint kuruyordu) -- 2026-08-18 karar 4 "
            "ile genisletildi (daraltma degil, veri kaybi riski yok). "
            "2026-09-12 canli dogrulama: constraint artik "
            "['low','intermediate','high','very_high'] iceriyor -- OK."
        ),
        "deprecated_columns": {
            "tmz_class_relative": (
                "DUSURULMEDI (sema degisikligi olurdu) -- 'deprecated' "
                "olarak belgelendi. tmz_class'in emoji sunum formati, AYNI "
                "tertil siniflandirmasi. Backend tmz_class kullanir, bu "
                "kolon HICBIR yerde okunmuyor."
            ),
        },
        "unproducible_null_columns": {
            "columns": ["egfr_amp_flag", "pten_del_flag", "cdkn2a_del_flag", "mgmt_interpretation"],
            "status": "48/48 NULL, kalici -- 'kaynakta uretim kurali yok' olarak kapatildi (Karar 6).",
            "canonical_statement": (
                "Kaynak veri sozlugunde (genom_verileri.xlsx, 11 sayfa) bu "
                "alanin uretim kurali tanimli degil; 48/48 NULL, uretilmedi."
            ),
        },
        "extra_column_penalizer": 0.0,
        "extra_column_penalizer_note": (
            "Karar 7 -- klinik kovaryatlar (yas/cinsiyet/GTR/IDH/MGMT) "
            "ceza almadan modelde kalir (glmnet penalty.factor=0 pratigi). "
            "Bu Cox modeline ait bir not, omics_scorer'in KENDI "
            "skorlamasindan degil -- burada tasindi cunku ayni karar "
            "dosyasinda (2026-08-18) birlikte kayitli."
        ),
        "review_gate_status": (
            "Adim 2 (Codex capraz inceleme) TAMAMLANDI 2026-08-18 -- 1 "
            "gercek bug (classify_* yuvarlama) + 2 bosluk (test skip, "
            "threshold dogrulama) bulundu, UCU DE DUZELTILDI, 28/28+53/53 "
            "testle yeniden dogrulandi. Adim 1 (fiziksel inceleme) ve "
            "adim 3 (final rapor oncesi resmi gozden gecirme) HENUZ "
            "yapilmadi -- final rapor hazirliginda toplu yapilacak "
            "(review-gate TAM KAPANMADI, bu kayit 'production-ready' "
            "ILAN ETMEZ)."
        ),
        "consumers": ["api/predict.py", "api/similar.py"],
        "not_a_required_cox_input": (
            "CLAUDE.md: omics modulu (48 TCGA hastasi) opsiyoneldir, "
            "Cox'un ZORUNLU girdisi DEGILDIR -- MR kohortundan bagimsiz, "
            "ayri bir populasyon."
        ),
        "compiled_by": "db-agent, 2026-09-12 -- decisions/2026-08-18-omics-"
                        "skor-formulleri-ve-esikler.md + pipeline/omics_"
                        "scores.py + canli DB SELECT'ten derlendi.",
    }
    training_cohort_snapshot = (
        "48 TCGA-Omics hastasi (omics_profiles/molecular_scores, canli "
        "2026-09-12 SELECT ile 48/48 dogrulandi) -- Cox/XGBoost MR "
        "kohortundan (UPenn/LUMIERE/UCSF/TCGA-GBM) BAGIMSIZ, ayri bir "
        "populasyon (CLAUDE.md kurali). score_thresholds bu 48 hastanin "
        "p33/p66 tertillerinden DONDURULMUS (frozen_at 2026-08-18 "
        "13:38:47 UTC) -- yeni hasta eklendiginde eski hastalarin sinifi "
        "KAYMAZ (mimari v45.txt:471-492 kritik bulgusunun duzeltmesi)."
    )
    return {
        "model_name": "omics_scorer",
        "version": "v1.0",
        "training_date": "2026-08-18",
        "training_cohort_snapshot": training_cohort_snapshot,
        "metrics": metrics,
        # 'production' -- score_thresholds DONMUS, molecular_scores 48/48
        # CANLI DOLU, api/predict.py + api/similar.py bu skorlari GERCEKTEN
        # OKUYOR (grep ile dogrulandi) -- golgede/paralel test edilen bir
        # alternatif YOK, tek/fiili skorlama motoru budur.
        "status": "production",
    }


BUILDERS = {
    "nnunet": _build_nnunet_row,
    "omics_scorer": _build_omics_scorer_row,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(BUILDERS.keys()))
    parser.add_argument("--apply", action="store_true", help="gercek INSERT calistir (varsayilan: dry-run)")
    args = parser.parse_args()

    row = BUILDERS[args.model]()

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
        print("\n[DRY-RUN] --apply verilmedi, hicbir yazma yapilmadi.")
        return 0

    if already_exists:
        print("[APPLY] Yazilacak yeni satir yok (zaten var), islem YOK.")
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
        print(f"\n[APPLY] COMMIT edildi. model_id={new_id} yazildi.")
    except Exception:
        conn.rollback()
        print("\n[APPLY] HATA -- ROLLBACK edildi.", file=sys.stderr)
        raise
    finally:
        cur.close()
        conn.close()

    verify_conn = get_connection(readonly=True)
    try:
        verify_cur = verify_conn.cursor()
        verify_cur.execute(
            "SELECT model_id, model_name, version, status, training_date "
            "FROM model_registry WHERE model_name = %s AND version = %s",
            (row["model_name"], row["version"]),
        )
        print(f"[DOGRULAMA] canli SELECT: {verify_cur.fetchone()}")
        verify_cur.close()
    finally:
        verify_conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
