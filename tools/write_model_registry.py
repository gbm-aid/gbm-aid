"""`model_registry`'ye bir Cox PH varyantinin kaydini HAZIRLAR (dry-run).

2026-08-18 KAPSAM DUZELTMESI (Baris): V2 dort-varyant kosusu (v1_referans/
v2a_mgmt/v2b_mgmt_spline/v2c_mgmt_spline_wttc) bu script yazildigi anda
HALA SURUYOR -- hangisinin NIHAI model olacagi Baris'in kosu-sonrasi
kararina birakildi. Bu yuzden bu script HICBIR varyanti/versiyonu
VARSAYMAZ: `--variant` VE `--version` ZORUNLU CLI argumanlaridir, ikisi de
BURADA SABIT KODLANMADI.

Kaynak (variant'a gore parametrik): `artifacts/week3/cox_model/
week3_<variant>_{external_test.csv,fold_results.csv,run_metadata.json}`
(modeling-agent'in V2 suite ciktisi, bkz. AKTIF-GOREVLER.md).

METRIK DEGERLERI ELLE YAZILMADI -- hepsi yukaridaki CSV/JSON dosyalarindan
OKUNUR (bkz. _load_metrics()). Bu script metrik HESAPLAMAZ, yalniz
zaten-uretilmis sonuclari DB'ye tasimaya HAZIRLANIR.

DAVRANIS:
  - Varsayilan (VE su an TEK desteklenen mod): DRY-RUN -- hicbir yazma
    yapilmaz, INSERT metni + metrics JSON yazdirilir, ORADA DURULUR.
  - `--apply` kod olarak MEVCUT (write_score_thresholds.py ile AYNI
    desen, ileride nihai model secilince yeniden yazmaya gerek kalmasin
    diye) AMA Baris'in acik onayi olmadan ASLA calistirilmamalidir --
    2026-08-18 itibariyle nihai model HENUZ SECILMEDI, bu script'in
    kapsami "hazirlik", "yazim" DEGIL.
  - Idempotent: (model_name, version) zaten varsa (UNIQUE constraint)
    INSERT ATLANIR, hata firlatilmaz -- mukerrer yazim onlenir.

Kullanim:
  python tools/write_model_registry.py --variant v1_referans --version v1.0
  python tools/write_model_registry.py --variant v2a_mgmt --version v1.0 --apply   # SADECE Baris onayiyla

KILITLI (degistirilemez, gorev talimati + mimari v4.5 SS6.5):
  model_name='cox_phm', status='shadow' (yeni versiyon DOGRUDAN
  production'a gecmez).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # gbm-aid mert/tools -> gbm-aid mert
sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from db_connection import get_connection  # noqa: E402

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "week3" / "cox_model"
MODELS_DIR = PROJECT_ROOT / "backend" / "models"  # 2026-09-16: models/ backend/ altina tasindi

MODEL_NAME = "cox_phm"
STATUS = "shadow"  # mimari v4.5 SS6.5 -- yeni versiyon DOGRUDAN production'a gecmez, KILITLI

KNOWN_VARIANTS = (
    "v1_referans", "v2a_mgmt", "v2b_mgmt_spline", "v2c_mgmt_spline_wttc",
    # 2026-09-12 EKLENDI (db-agent, Baris'in nihai model karari --
    # decisions/2026-09-12-nihai-cox-model-secimi-v3b.md): v3 kolinearite
    # filtresiyle 93->54 radyomik aday havuzundan secilen NIHAI Cox PHM
    # varyanti. Diger v3 ailesi (v3a/v3c) bu script'in KAPSAMINA
    # BILEREK EKLENMEDI -- gorev talimati SADECE v3b'nin kaydini istiyor.
    "v3b_lowvar_v2amgmt",
)

# 2026-09-12 DUZELTMESI (db-agent): eski kod TRAINING_DATE'i SABIT
# "2026-08-18" olarak kodluyordu -- bu, dort-varyant V2 kosusuna (o gun
# calisti) DOGRUYDU ama v3b GERCEKTE 2026-08-19'da kostu (bkz.
# week3_v3b_lowvar_v2amgmt_run_metadata.json::generated_at_utc). Sabit
# kodlanmis bir tarih SESSIZCE YANLIS olurdu -- artik her varyant icin
# run_metadata.json'un KENDI `generated_at_utc` alanindan OKUNUYOR, hicbir
# tarih elle yazilmiyor.
def _training_date_from_run_metadata(run_meta: dict) -> str:
    generated_at = run_meta.get("generated_at_utc")
    if not generated_at:
        raise KeyError(
            "run_metadata.json'da 'generated_at_utc' YOK -- training_date "
            "TAHMIN EDILEMEZ, elle dogrulanmali."
        )
    return generated_at.split("T", 1)[0]


def _variant_paths(variant: str) -> dict[str, Path]:
    return {
        "external_test": ARTIFACT_DIR / f"week3_{variant}_external_test.csv",
        "fold_results": ARTIFACT_DIR / f"week3_{variant}_fold_results.csv",
        "run_metadata": ARTIFACT_DIR / f"week3_{variant}_run_metadata.json",
    }


def _first(row: Any, *names: str) -> Any:
    """Bir pandas Series/dict'ten ilk MEVCUT anahtari doner -- CSV sema
    adlari eski (Kosu #7) ve yeni (V2 suite) arasinda DEGISTIGI icin
    (ornek: `final_features` -> `final_features_radiomic`) BURADA
    esneklik saglanir, hicbir sayi UYDURULMAZ (isim yoksa KeyError)."""

    for name in names:
        try:
            value = row[name]
        except (KeyError, IndexError):
            continue
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            return value
    raise KeyError(f"Hicbiri bulunamadi: {names}")


def _load_metrics(variant: str) -> dict:
    """Metrikleri artifacts/week3/cox_model/week3_<variant>_*.csv/.json
    dosyalarindan OKUR -- hicbir sayi ELLE yazilmadi. JSON'un `arm`
    alt-sozlugu (eski VE yeni sema arasinda STABIL) ONCELIKLI kaynak,
    CSV yalniz JSON'da olmayan tam ozellik LISTELERI icin kullanilir."""

    paths = _variant_paths(variant)
    external_row = pd.read_csv(paths["external_test"]).iloc[0]
    folds = pd.read_csv(paths["fold_results"])
    run_meta = json.loads(paths["run_metadata"].read_text(encoding="utf-8"))
    arm_meta = run_meta.get("arm") or {}
    ext_meta = arm_meta.get("external_test") or {}

    final_features = str(_first(external_row, "final_features_radiomic", "final_features")).split(";")
    clinical_extra_columns = list(
        arm_meta.get("clinical_extra_columns")
        or str(_first(external_row, "clinical_extra_columns")).split(";")
    )
    n_final_features = int(arm_meta.get("n_final_features") or _first(external_row, "n_final_features_radiomic", "n_final_features"))
    n_clinical_covariates = len(clinical_extra_columns)
    n_model_params = n_final_features + n_clinical_covariates
    n_events_train = int(run_meta["expected_training_event_count"])
    regions = sorted({f.split("__", 1)[0] for f in final_features})

    c_index_external = float(ext_meta.get("c_index", _first(external_row, "c_index")))
    ci_lower = float(ext_meta.get("ci_lower", _first(external_row, "ci_lower")))
    ci_upper = float(ext_meta.get("ci_upper", _first(external_row, "ci_upper")))

    return {
        "variant": variant,
        "run_metadata_generated_at_utc": run_meta.get("generated_at_utc"),
        "arm_name": arm_meta.get("name", variant),
        "regions": regions,
        "c_index_external": c_index_external,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": float(ext_meta.get("confidence_level", _first(external_row, "confidence_level"))),
        "c_index_internal_cv": float(folds["c_index"].mean()),
        "c_index_internal_cv_per_fold": [float(x) for x in folds["c_index"].tolist()],
        "c_index_internal_cv_std": float(folds["c_index"].std(ddof=1)),
        "n_outer_folds": int(folds["fold"].nunique()),
        "n_patients_train": int(run_meta["expected_training_patient_count"]),
        "n_events_train": n_events_train,
        "n_patients_external": int(ext_meta.get("n_patients", _first(external_row, "n_patients"))),
        "n_events_external": int(ext_meta.get("n_events", _first(external_row, "n_events"))),
        "n_final_features": n_final_features,
        "final_features": final_features,
        "clinical_covariates": clinical_extra_columns,
        "n_clinical_covariates": n_clinical_covariates,
        "n_model_parameters": n_model_params,
        "stability_frequency_threshold": float(
            arm_meta.get("stability_frequency_threshold", _first(external_row, "stability_frequency_threshold"))
        ),
        "epv": round(n_events_train / n_model_params, 4),
        "epv_note": (
            f"{n_events_train} olay / {n_model_params} parametre "
            f"({n_final_features} radyomik + {n_clinical_covariates} klinik) = "
            f"{n_events_train / n_model_params:.2f} -- CLAUDE.md'nin genel "
            "WT-only x93 EPV=6,3 rakamindan FARKLI (bu, sadece secili "
            "ozellik + klinik kovaryatlarin FINAL model EPV'si)."
        ),
        "bootstrap_n": int(ext_meta.get("n_bootstrap_valid", _first(external_row, "n_bootstrap_valid"))),
        "bootstrap_n_requested": int(
            ext_meta.get("n_bootstrap_requested", _first(external_row, "n_bootstrap_requested"))
        ),
        "fallback_folds": int(arm_meta.get("n_fallback_folds", _first(external_row, "n_fallback_folds"))),
        "used_fallback_full_pool": bool(
            arm_meta.get("final_model_used_fallback", _first(external_row, "used_fallback_full_pool"))
        ),
        "external_test_source": str(_first(external_row, "external_test_source")),
        "age_knots": run_meta.get("age_knots"),
        # -- ZORUNLU BEYAN ALANLARI (gorev talimati, CLAUDE.md kurali) --
        "post_hoc_note": (
            "Klinik kovaryatli (radyomik+klinik) model ailesi (v1_referans/"
            "v2a_mgmt/v2b_mgmt_spline/v2c_mgmt_spline_wttc), yalniz-radyomik "
            "sonuc (WTx93, harici C-index 0,689 [0,546-0,824], TCGA n=38, "
            "2026-08-15) GORULDUKTEN SONRA tanimlanmistir -- pre-specified "
            "DEGIL, kesifsel/post-hoc bir analizdir. Yalniz-radyomik sonuc "
            "birincil rapordan CIKARILMAZ, bu model onun YERINE gecmez, "
            "yanina EKLENIR."
        ),
        "k14_note": (
            "K14 KAPANDI (2026-08-19) -- 'clinical_base' (yalniz "
            "yas+cinsiyet, radyomiksiz) kolu KOSULDU, sonuc OLUMSUZ. "
            "UCSF 295/169'da harici C-index 0,666454 [0,621209-0,712068] "
            "(ic CV 0,6250+-0,0504; cinsiyet katsayisi -0,0031, fiilen yas "
            "tek basina). v1_referans 0,678064 -> radyomik+GTR+IDH artimi "
            "+0,0116, %95 GA'lar ORTUSUK. Bu yuzden 'radyomik ozellikler "
            "yasin/cinsiyetin UZERINE ANLAMLI prognostik katki sagliyor' "
            "cumlesi HARICI VERIYLE DESTEKLENEMEDI -- kurulamaz. "
            "DIL UYARISI: 'olculmedi/yapilmadi' DEGIL, 'olculdu, artim "
            "ayrismadi'. Detay: "
            "decisions/2026-08-19-k14-clinical-base-sonucu.md. "
            "(2026-08-28'de duzeltildi; eski hali 'K14 KAPANMADI ... kolu "
            "HIC KOSULMADI' idi.)"
        ),
        "ucsf_selection_bias_note": (
            "UCSF harici test kohortu (295/169), veri indirme kesintisi "
            "nedeniyle koleksiyonun ID 118-541 dilimidir (ID 4-116 "
            "araligindaki 94 hasta indirilemedi, IBM Aspera ~142 GB, %80'de "
            "kalici olarak durdu). Bu alt kume rezeksiyon genisligi "
            "dagiliminda TAM kohorttan anlamli olarak FARKLIDIR (GTR %65 vs "
            "%39, p<0,001); yas/cinsiyet/sagkalim/olay oraninda fark YOKTUR "
            "(p=0,358/0,517/0,836/0,735)."
        ),
        "calibration_note": (
            "UPenn sagkalim suresi AMELIYATTAN, UCSF ise TANIDAN baslar "
            "(farkli index-date tanimlari) -- iki kohort arasinda KALIBRASYON "
            "iddiasi (ornek: '5 yillik sagkalim olasiligi X%') KURULAMAZ, "
            "yalniz DISKRIMINASYON (C-index/siralamada dogruluk) raporlanir."
        ),
        "reporting_rule_note": (
            "Tek-esik dili YASAK -- 'C-index >= X'i gectik' gibi cumle "
            "KURULMAZ, daima nokta tahmini + %95 bootstrap CI birlikte "
            "raporlanir."
        ),
        "variant_selection_note": (
            f"Bu kayit {variant!r} varyantina aittir -- 2026-08-18 itibariyle "
            "V2 dort-varyant kosusunun (v1_referans/v2a_mgmt/v2b_mgmt_spline/"
            "v2c_mgmt_spline_wttc) HANGISININ nihai/production adayi olacagi "
            "Baris'in kosu-sonrasi kararina birakilmistir -- bu kaydin "
            "kendisi bir 'nihai model secildi' beyani DEGILDIR."
            + (
                " GUNCELLEME (2026-09-12): bu belirsizlik bu varyant "
                "ICIN artik COZULDU -- bkz. 'final_selection_note' alani "
                "(ayni metrics JSONB icinde). Bu cumle burada SILINMEDI, "
                "cunku yazildigi anda dogruydu ve karar tarihcesinin bir "
                "parcasidir (CLAUDE.md hard rule #3, 'celiski silinmez')."
                if variant in FINAL_SELECTION_NOTES else ""
            )
        ),
    }


def _load_artifact_provenance(model_artifact_path: Path | None) -> dict[str, Any] | str:
    """Model .pkl artefaktinin KENDI `provenance` alanini (varsa) okur --
    hicbir provenance iddiasi burada UYDURULMAZ/SABIT KODLANMAZ. Yol
    verilmemis veya dosya yoksa acik bir UYARI metni doner (SESSIZCE
    atlanmaz)."""

    if model_artifact_path is None:
        return (
            "--model-artifact-path VERILMEDI -- bu kayit metrikleri "
            "artifacts/week3/cox_model/*.csv'den okudu ama HANGI .pkl "
            "dosyasinin bu sonuclara karsilik geldigini DOGRULAMADI. Nihai "
            "model secilip .pkl'e fit edildiginde bu kayit --model-artifact-"
            "path ile GUNCELLENMELI (yeni bir satir/versiyon olarak)."
        )
    if not model_artifact_path.is_file():
        return f"belirtilen --model-artifact-path YOK: {model_artifact_path}"

    import pickle

    try:
        with model_artifact_path.open("rb") as fh:
            loaded = pickle.load(fh)
    except Exception as exc:  # pragma: no cover -- yalniz bozuk/uyumsuz pkl
        return f"artefakt OKUNAMADI ({model_artifact_path}): {exc}"

    if isinstance(loaded, dict) and "provenance" in loaded:
        return {"path": str(model_artifact_path), **loaded["provenance"]}
    return {
        "path": str(model_artifact_path),
        "note": "artefaktin kendi 'provenance' alani YOK (eski/farkli bicim olabilir).",
    }


FINAL_SELECTION_NOTES: dict[str, str] = {
    # 2026-09-12 EKLENDI (db-agent) -- SADECE nihai model olarak SECILEN
    # varyanta iliskilendirilir, digerlerine (v1/v2a/v2b/v2c/v3a/v3c)
    # BOS/YOK kalir -- hicbiri "secildi" gibi gorunmemeli (CLAUDE.md
    # "MODEL GELISTIRME SERBESTTIR" seffaflik kurali).
    "v3b_lowvar_v2amgmt": (
        "NIHAI MODEL SECIMI (Baris, 2026-09-12, birebir): 'evet uzun "
        "ugraslar sonucu nihai model belli oldu v3b yi kullaniyoruz onu "
        "seciyorum'. Karar dayanagi: 7/8 kolun harici Harrell-C %95 GA'lari "
        "TAMAMEN ORTUSUYOR (istatistiksel olarak ayirt edilemez) -- secim "
        "Occam ilkesiyle ikincil kriterlere (EPV=9,435, katsayi patlamasi "
        "YOK [|coef|<0,15], WT-only bolge sadeligi, MGMT dahil) dayanir, "
        "nokta-tahmin ustunlugu DEGIL. Diger 7 kol (v1/v2a/v2b/v2c/v3a/v3c/"
        "clinical_base-k14) raporda SILINMEZ, Tablo X'te v3b 'secilen' "
        "olarak isaretlenir. Cokluk beyani: harici test seti 8 kol uzerinde "
        "degerlendirildi -- sunulan skor bu nedenle bir miktar iyimser "
        "olabilir. Detay: decisions/2026-09-12-nihai-cox-model-secimi-"
        "v3b.md."
    ),
}


def _build_insert_row(variant: str, version: str, model_artifact_path: Path | None) -> dict:
    metrics = _load_metrics(variant)
    metrics["artifact_provenance"] = _load_artifact_provenance(model_artifact_path)
    if variant in FINAL_SELECTION_NOTES:
        metrics["final_selection_note"] = FINAL_SELECTION_NOTES[variant]
    training_date = _training_date_from_run_metadata(
        {"generated_at_utc": metrics["run_metadata_generated_at_utc"]}
    )

    n_final_features = metrics["n_final_features"]
    n_clinical_covariates = metrics["n_clinical_covariates"]
    regions_str = "+".join(metrics["regions"])
    training_cohort_snapshot = (
        f"Egitim: UPenn-GBM 611 hasta / 585 olum olayi (630/603 kayitli DEGIL -- "
        "19 hasta hazir segmentasyon maskesi olmadigi icin radyomik satiri "
        "uretilemedi, bkz. decisions/2026-08-13-bolge-stratejisi-ve-modelleme-"
        "protokolu.md 'SAYI DUZELTMESI 2026-08-14'). Harici test: UCSF-PDGM 295 "
        "hasta / 169 olum olayi (ID 118-541 dilimi, indirme kesintisi nedeniyle "
        "-- bkz. metrics.ucsf_selection_bias_note). "
        f"Model (varyant={variant!r}): {regions_str}-bolge {n_final_features} secili "
        f"radyomik ozellik + {n_clinical_covariates} klinik kovaryat "
        f"({', '.join(metrics['clinical_covariates'])}). Radyomik sozlesmesi: C32 "
        "(N4 + T1ce-ozel kaynak Z-score + binCount=32, bkz. decisions/2026-08-13-"
        "pyradiomics-c32-bincount-karari.md). ComBat: UYGULANMADI (egitim havuzu"
        "==referans batch UPenn oldugu icin cebirsel olarak OZDESLIK donusumu, "
        "bkz. decisions/2026-08-13-combat-ozdeslik-bulgusu-ve-kapsami.md -- maks "
        "fark 0,0). TCGA-GBM bu egitime KESINLIKLE GIRMEDI (tam bagimsiz harici "
        "test seti konumu korunuyor)."
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
        "--variant", required=True, choices=KNOWN_VARIANTS,
        help="artifacts/week3/cox_model/week3_<variant>_*.{csv,json} dosyalarini secer -- ZORUNLU, VARSAYILAN YOK",
    )
    parser.add_argument(
        "--version", required=True,
        help="model_registry.version (ornek: v1.0) -- ZORUNLU, VARSAYILAN YOK (nihai model kararindan sonra belirlenir)",
    )
    parser.add_argument(
        "--model-artifact-path", type=Path, default=None,
        help="bu varyanta karsilik gelen .pkl dosyasi (varsa, provenance icin okunur) -- OPSIYONEL",
    )
    parser.add_argument("--apply", action="store_true", help="gercek INSERT calistir (varsayilan: dry-run) -- Baris onayi olmadan KULLANMA")
    args = parser.parse_args()

    paths = _variant_paths(args.variant)
    for path in paths.values():
        if not path.is_file():
            print(f"HATA: beklenen kaynak dosya yok: {path}", file=sys.stderr)
            return 1

    row = _build_insert_row(args.variant, args.version, args.model_artifact_path)

    print("=" * 70)
    print(f"model_name={row['model_name']!r} version={row['version']!r} status={row['status']!r}")
    print(f"variant={args.variant!r} training_date={row['training_date']!r}")
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

    # Canli dogrulama -- AYRI, salt-okunur baglanti
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
