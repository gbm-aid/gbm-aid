"""TUM varyantlarin (v1/v2a/v2b/v2c + v3a/v3b/v3c + k14 taban cizgisi) TEK
birlesik karsilastirma tablosunu kurar.

2026-09-12 GUNCELLEMESI (modeling-agent-E): liste 6 kosudan 8'e cikarildi
-- `v3c_lowvar_wttc_mgmt_nospline` (8. kol, 2026-09-11, ciktilari
`final_v3c/` ALT-DIZININDE) ve `k14_clinical_base_ucsf` (taban cizgisi,
varyant DEGIL, radyomik 0) eklendi. Ayni gunde `v3b_lowvar_v2amgmt`
NIHAI BIRINCIL MODEL olarak SECILDI (`decisions/2026-09-12-nihai-cox-
model-secimi-v3b.md`) -- tabloya bunu isaretleyen bir kolon eklendi.
Diger 6 kolun satirlari/mantigi DEGISTIRILMEDI.

NEDEN GEREKLI (2026-08-19, koordinator bulgusu)
------------------------------------------------
CLAUDE.md'nin "MODEL GELISTIRME SERBESTTIR" kurali (2026-08-18, Baris'in
netlestirmesi) sunu soyluyor:

    "TEK ZORUNLULUK: kosulan HER varyantin sonucu raporda yer alir.
     Final raporda tum varyantlarin ciktilari BIR TABLODA verilir."
    "Yasak olan tek sey: varyantlari gizleyip yalniz en iyisini
     'tek kosulan model' gibi sunmak."

Ama kod IKI AYRI tablo yaziyor:

    week3_v2_variant_comparison.csv   ->  v1, v2a, v2b, v2c
    week3_v3_variant_comparison.csv   ->  v3a, v3b

ve `tools/merge_v2_comparison.py`'nin `VARIANTS` listesi YALNIZ dort v2
adini iceriyor -- v3 adlari orada YOK. Rapor yazan biri "iste varyant
karsilastirma tablosu" deyip v2 tablosunu koyarsa v3a/v3b SESSIZCE
kaybolur. Kimse kasten gizlemez; script oyle yazilmis diye olur. Sonuc
yine de seffaflik kuralinin fiilen ihlalidir.

Bu script, `merge_v2_comparison.py` ile AYNI deseni izler ama HER IKI
neslin varyantlarini tek tabloda toplar ve eksik/coken varyantlari
ACIKCA isaretler -- sessiz dusme yok.

`merge_v2_comparison.py` SILINMEDI/DEGISTIRILMEDI: o, v2 tablosunun
kanonik yeniden kurucusu olarak kaliyor (dort v2 kosusu birbirinin
tablosunu eziyor, o sorunu o cozuyor). Bu script onun USTUNE, rapor
icin tek bir gorunum uretir.

Salt-okunur: yalniz `week3_all_variants_comparison.csv` yazar. DB'ye
DOKUNMAZ.

TAMAMLANMA OLCUTU (2026-08-19 DUZELTMESI -- Codex cikis incelemesi
bulgusu: "Birlesik varyant tablosu EKSIK ciktiyi BASARI sayiyor")
-------------------------------------------------------------------
Bu script'in ILK surumu `not ext AND not meta` olcutunu kullaniyordu --
yani UC ciktidan BIRI bile varsa satir "TAMAM" isaretleniyor, metrik
kolonlari NaN kaliyordu. Yarim kalmis bir kosu (orn. `run_metadata.json`
yazildi ama surec `_external_test.csv` yazilmadan oldu) tabloda
BASARILI gorunurdu. Bu, tam da bu script'in onlemek icin yazildigi
seffaflik hatasinin kendisidir.

Gecerli olcut, `tools/variant_run_watchdog.ps1`'inkiyle BIREBIR AYNI
(projede "bitti"nin TEK bir tanimi olsun diye):

    UC cikti da var olmali:
        week3_<varyant>_external_test.csv
        week3_<varyant>_run_metadata.json
        week3_<varyant>_fold_results.csv
    VE run_metadata.json GECERLI JSON olarak parse edilebilmeli
    VE iki CSV de UC KAPIDAN gecmeli (ikisi icin de AYNI kapilar):
         BOZUK        -> parse edilemedi (0 baytlik / yarim yazilmis)
         BOS          -> baslik satiri var, VERI satiri YOK
         KOLON_EKSIK  -> dolu ama asil metrik kolonu (`c_index`) yok
         DEGER_YOK    -> kolon var ama degeri BOS/NaN
         DEGER_EKSIK  -> (yalniz fold_results) bazi fold'lar NaN --
                         `mean()` onlari SESSIZCE atlardi
         CI_YOK       -> (yalniz external_test) C-index var ama %95 CI
                         yok/NaN. CLAUDE.md davranis kurali 3 geregi
                         nokta tahmini TEK BASINA raporlanamaz.

Uc durum ayirt edilir ve HICBIRI digerine yuvarlanmaz:
    TAMAM                 -> uc cikti tam, metadata gecerli
    EKSIK_CIKTI           -> bazi ciktilar var, bazilari yok/bozuk
                             (`eksik_dosyalar` kolonu neyin eksik
                             oldugunu ACIKCA yazar)
    KOSULMADI_VEYA_COKTU  -> hicbir cikti yok
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).absolute().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "week3" / "cox_model"

#: (varyant_adi, nesil, tanim, rol) -- rapor sirasi bu siradir.
#: Yeni bir varyant kosulursa BURAYA EKLENMELI; eklenmezse tabloda
#: gorunmez ve seffaflik kurali ihlal edilir.
#: `rol`: "varyant" (radyomik+klinik kombinasyonu dener) veya
#: "taban_cizgisi" (radyomik YOK, karsilastirma cipasi -- ornegin k14).
#: Taban cizgisi bir "varyant" DEGILDIR, tabloda ayirt edilmesi CLAUDE.md
#: K14 maddesinin geregidir (bkz. asagidaki `_secim_durumu`).
VARIANT_CATALOG: list[tuple[str, str, str, str]] = [
    ("v1_referans", "v2", "WT x93 + yas(dogrusal) + cinsiyet + GTR + IDH", "varyant"),
    ("v2a_mgmt", "v2", "v1 + MGMT", "varyant"),
    ("v2b_mgmt_spline", "v2", "v2a + yas restricted cubic spline", "varyant"),
    ("v2c_mgmt_spline_wttc", "v2", "v2b + TC bolgesi (WT+TC, 186 aday)", "varyant"),
    ("v3a_lowvar_v1referans", "v3", "v1 + kolinearite filtresi + radyomik standardizasyon", "varyant"),
    ("v3b_lowvar_v2amgmt", "v3", "v2a + kolinearite filtresi + radyomik standardizasyon", "varyant"),
    (
        "v3c_lowvar_wttc_mgmt_nospline",
        "v3",
        "v2a + kolinearite filtresi + radyomik standardizasyon + TC bolgesi (WT+TC, spline yok) -- 8. kol, 2026-09-11",
        "varyant",
    ),
    (
        "k14_clinical_base_ucsf",
        "baseline",
        "SADECE yas + cinsiyet (radyomik 0) -- karsilastirma cipasi, varyant DEGIL",
        "taban_cizgisi",
    ),
]

#: WT-only disindaki bolge kumeleri icin BEYAN EDILMIS egitim havuzu
#: (bkz. `train_cox_week3.py::DECLARED_REGION_SHORTFALLS` ve
#: decisions/2026-08-19-bolge-farkinda-egitim-havuzu-kapisi.md).
#: Tabloda "egitim_havuzu" kolonu bu sayiyi tasir -- v2c/v3c icin
#: 611/585 YAZMAK YANLIS olurdu (2 hastada TC=0, bkz. katalog SS "v2c/v3c'nin
#: ozel durumu").
DECLARED_TRAINING_POOL: dict[str, str] = {
    "v2c_mgmt_spline_wttc": "609/583",
    "v3c_lowvar_wttc_mgmt_nospline": "609/583",
}
DEFAULT_TRAINING_POOL = "611/585"

#: `v3c`'nin ciktilari diger 7 kolun aksine KOK dizinde degil, alt-dizinde
#: durur (`entities/cox-model-varyant-katalogu.md` SS5 "Cikti dizini
#: istisnasi"). Bu sozluk BOS ise davranis eskisiyle AYNIdir (kok dizin).
VARIANT_OUTPUT_SUBDIR: dict[str, str] = {
    "v3c_lowvar_wttc_mgmt_nospline": "final_v3c",
}

#: 2026-09-12 Baris karari -- nihai birincil Cox PHM modeli.
#: `decisions/2026-09-12-nihai-cox-model-secimi-v3b.md`. Diger 7 kol
#: (6 varyant + k14 taban cizgisi) SILINMEZ, duyarlilik/taban-cizgisi
#: kollari olarak tabloda kalir (CLAUDE.md "MODEL GELISTIRME SERBESTTIR").
SELECTED_PRIMARY_VARIANT = "v3b_lowvar_v2amgmt"
SELECTED_PRIMARY_VARIANT_DECISION_REF = "decisions/2026-09-12-nihai-cox-model-secimi-v3b.md"

STATUS_OK = "TAMAM"
STATUS_PARTIAL = "EKSIK_CIKTI"
STATUS_ABSENT = "KOSULMADI_VEYA_COKTU"


def _secim_durumu(name: str, role: str) -> str:
    """Bir satirin nihai model secimindeki konumunu aciklayan metin.

    CLAUDE.md 2026-09-12: `v3b_lowvar_v2amgmt` birincil model SECILDI;
    diger 6 varyant duyarlilik kolu, `k14` ise taban cizgisi (varyant
    DEGIL) olarak kalir -- hicbiri tablodan SILINMEZ (seffaflik kurali).
    """

    if name == SELECTED_PRIMARY_VARIANT:
        return f"SECILEN_BIRINCIL_MODEL ({SELECTED_PRIMARY_VARIANT_DECISION_REF})"
    if role == "taban_cizgisi":
        return "taban_cizgisi (varyant DEGIL, karsilastirma cipasi)"
    return "duyarlilik_kolu"


def _read_json(path: Path) -> dict | None:
    """Gecerli JSON ise sozluk, degilse `None`. Bozuk dosya SESSIZCE
    'yok' sayilmaz -- cagiran taraf `None`'i ayri bir eksiklik olarak
    isaretler."""

    if not path.is_file():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"[UYARI] {path.name} okunamadi/gecersiz ({type(exc).__name__}).",
            file=sys.stderr,
        )
        return None
    # 2026-08-19 (Codex MEDIUM bulgusu #4): sozdizimi gecerli ama KOK TIPI
    # yanlis bir JSON iki ayri sekilde zarar veriyordu:
    #   `[]`      -> falsy, `meta is None` DEGIL -> bozuk SAYILMIYOR,
    #                satir sessizce TAMAM olabiliyordu;
    #   `["x"]`   -> truthy -> sonraki `meta[key]` erisimi `TypeError`
    #                firlatip TUM birlestirmeyi cokertiyordu.
    # Kok `dict` degilse artik acikca bozuk sayilir.
    if not isinstance(parsed, dict):
        print(
            f"[UYARI] {path.name} gecerli JSON ama kok tipi 'dict' DEGIL "
            f"({type(parsed).__name__}) -- bozuk sayiliyor.",
            file=sys.stderr,
        )
        return None
    return parsed


def _read_csv(path: Path) -> pd.DataFrame | None:
    """Gecerli CSV ise DataFrame, degilse `None`.

    2026-08-19 DUZELTMESI (Codex cikis incelemesi bulgusu #2: "Eksik/yarim
    cikti kapisi BOS CSV dosyasinda hala cokuyor"). `pd.read_csv()` 0
    baytlik bir dosyada `EmptyDataError`, bozuk bir dosyada `ParserError`
    firlatir. Bu istisnalar YAKALANMAYINCA script'in TAMAMI cokuyordu --
    ve bu, tam da kapinin yakalamasi gereken senaryo: surec CSV yazimi
    ORTASINDA olurse geriye 0 baytlik/yarim bir dosya kalir. Yani kapi,
    korumasi gereken durumda devre disi kaliyordu.

    `None` donmesi cagiran tarafta `*_BOZUK` eksikligi olarak isaretlenir
    -- sessizce "dosya yok" sayilmaz, cunku dosyanin VAR ama OKUNAMAZ
    olmasi farkli (ve daha suphe verici) bir durumdur.
    """

    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, UnicodeDecodeError) as exc:
        print(
            f"[UYARI] {path.name} okunamadi/gecersiz ({type(exc).__name__}).",
            file=sys.stderr,
        )
        return None


def collect(output_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for name, generation, description, role in VARIANT_CATALOG:
        row: dict = {
            "variant": name,
            "nesil": generation,
            "tanim": description,
            "rol": role,
            "secim_durumu": _secim_durumu(name, role),
            "egitim_havuzu": DECLARED_TRAINING_POOL.get(name, DEFAULT_TRAINING_POOL),
        }
        # `v3c`'nin ciktilari `final_v3c/` alt-dizininde; digerleri kok
        # dizinde (VARIANT_OUTPUT_SUBDIR bos donerse eski davranis AYNI).
        variant_dir = output_dir / VARIANT_OUTPUT_SUBDIR[name] if name in VARIANT_OUTPUT_SUBDIR else output_dir
        ext = variant_dir / f"week3_{name}_external_test.csv"
        meta_path = variant_dir / f"week3_{name}_run_metadata.json"
        folds = variant_dir / f"week3_{name}_fold_results.csv"

        present = {
            "external_test": ext.is_file(),
            "run_metadata": meta_path.is_file(),
            "fold_results": folds.is_file(),
        }

        if not any(present.values()):
            row["durum"] = STATUS_ABSENT
            rows.append(row)
            continue

        eksik = sorted(key for key, ok in present.items() if not ok)

        meta = _read_json(meta_path) if present["run_metadata"] else None
        if present["run_metadata"] and meta is None:
            # Dosya var ama GECERSIZ JSON ya da kok tipi dict DEGIL.
            eksik.append("run_metadata_BOZUK")

        # Metrikleri her durumda topla (varsa) -- ama `durum` bunlara
        # BAKMAZ: eksik bir cikti, dolu bir metrik kolonuyla TAMAM'a
        # yuvarlanamaz.
        if present["external_test"]:
            frame = _read_csv(ext)
            if frame is None:
                eksik.append("external_test_BOZUK")
            elif not len(frame):
                # Baslik satiri var, VERI satiri yok -- yarim yazim.
                eksik.append("external_test_BOS")
            elif "c_index" not in frame.columns:
                # Dosya dolu ama ASIL metrik kolonu yok -- semasi bozuk.
                eksik.append("external_test_KOLON_EKSIK")
            elif pd.isna(frame.iloc[0]["c_index"]):
                # 2026-08-19 DUZELTMESI (Codex bulgusu #4): kolon VAR ama
                # DEGERI BOS/NaN. Onceki surumde bu satir TAMAM sayilip
                # tabloya NaN bir C-index ile giriyordu -- "kosuldu ama
                # sonuc uretemedi" durumu "basarili" gorunuyordu.
                eksik.append("external_test_DEGER_YOK")
            else:
                first = frame.iloc[0]
                for src, dst in (
                    ("c_index", "harici_c_index"),
                    ("ci_lower", "ci_alt"),
                    ("ci_upper", "ci_ust"),
                    ("n_patients", "harici_n"),
                    ("n_events", "harici_olay"),
                    ("n_final_features", "secilen_ozellik"),
                    ("clinical_extra_columns", "klinik_kovaryat"),
                ):
                    if src in frame.columns:
                        row[dst] = first[src]
                # CLAUDE.md DAVRANIS KURALI 3: "nokta tahmini + %95
                # bootstrap CI ZORUNLU, tek-esik dili YASAK". Bir varyant
                # C-index uretip CI uretmemisse o satir raporlanabilir
                # DEGILDIR -- TAMAM sayilmaz.
                if {"ci_lower", "ci_upper"} <= set(frame.columns) and not (
                    pd.isna(first["ci_lower"]) or pd.isna(first["ci_upper"])
                ):
                    row["ci_genisligi"] = round(
                        float(first["ci_upper"]) - float(first["ci_lower"]), 4
                    )
                else:
                    eksik.append("external_test_CI_YOK")

        # 2026-08-19 DUZELTMESI (Codex cikis incelemesi bulgusu #3:
        # "Baslik satiri olan fakat VERI satiri bulunmayan
        # fold_results.csv hala TAMAM sayiliyor"). Onceki surumde
        # `if "c_index" in cols and len(frame):` KOSULU SAGLANMAZSA
        # hicbir eksiklik isaretlenmiyor, sessizce TAMAM'a dusuluyordu --
        # `external_test` tarafinda ayni kontrol VARDI (`external_test_BOS`),
        # fold tarafinda YOKTU. Asimetri kapatildi; iki cikti da AYNI
        # uc kapidan geciyor: BOZUK (parse edilemedi) / BOS (veri satiri
        # yok) / KOLON_EKSIK (asil metrik kolonu yok).
        if present["fold_results"]:
            fold_frame = _read_csv(folds)
            if fold_frame is None:
                eksik.append("fold_results_BOZUK")
            elif not len(fold_frame):
                eksik.append("fold_results_BOS")
            elif "c_index" not in fold_frame.columns:
                eksik.append("fold_results_KOLON_EKSIK")
            elif int(fold_frame["c_index"].isna().sum()) == len(fold_frame):
                # Kolon var, TUM degerler NaN.
                eksik.append("fold_results_DEGER_YOK")
            elif int(fold_frame["c_index"].isna().sum()) > 0:
                # KISMEN NaN: `mean()` NaN fold'lari SESSIZCE atlar ve
                # kalan fold'larin ortalamasini "ic CV" diye raporlardi.
                # NaN bir fold = COKEN bir fold; gizlenmemeli.
                n_nan = int(fold_frame["c_index"].isna().sum())
                eksik.append(f"fold_results_DEGER_EKSIK({n_nan}/{len(fold_frame)})")
            else:
                row["ic_cv_ort"] = round(float(fold_frame["c_index"].mean()), 4)
                # ddof=1 (pandas varsayilani) -- `merge_v2_comparison.py`
                # ve 2026-08-18 raporundaki KILITLI sayilarla (v1 0,0415 /
                # v2a 0,0302 / v2b 0,0333) tutarli olmak icin. ddof=0
                # kullanilsa ayni kosu icin farkli bir std raporlanirdi.
                row["ic_cv_std"] = round(float(fold_frame["c_index"].std(ddof=1)), 4)
                row["n_fold"] = len(fold_frame)
                if "used_fallback" in fold_frame.columns:
                    row["fallback_fold"] = int(fold_frame["used_fallback"].sum())

        if meta:
            # Aday ozellik sayisi: v3'te filtre SONRASI havuz, v2'de 93/186.
            # `n_radiomic_features` (2026-09-12 eklendi): k14'un metadata'si
            # bu anahtari kullaniyor (deger 0 -- radyomik yok, taban cizgisi).
            for key in (
                "n_candidate_features",
                "n_radiomic_candidates",
                "aday_ozellik",
                "n_radiomic_features",
            ):
                if key in meta:
                    row["aday_ozellik"] = meta[key]
                    break
            pool = meta.get("v3_pool_report")
            if isinstance(pool, dict):
                row["v3_filtre_girdi"] = pool.get("n_input")
                row["v3_filtre_kalan"] = pool.get("n_kept")

        if eksik:
            row["durum"] = STATUS_PARTIAL
            row["eksik_dosyalar"] = ";".join(sorted(set(eksik)))
        else:
            row["durum"] = STATUS_OK

        rows.append(row)

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    output_dir = Path(argv[0]).absolute() if argv else DEFAULT_OUTPUT_DIR
    if not output_dir.is_dir():
        print(f"HATA: cikti dizini yok: {output_dir}", file=sys.stderr)
        return 2

    table = collect(output_dir)
    target = output_dir / "week3_all_variants_comparison.csv"
    table.to_csv(target, index=False)

    n_ok = int((table["durum"] == STATUS_OK).sum())
    print(f"[OK] {len(table)} varyant -> {target}  (TAMAM: {n_ok}/{len(table)})")
    print(table.to_string(index=False))

    incomplete = table.loc[table["durum"] != STATUS_OK]
    if len(incomplete):
        print(
            "\n[DIKKAT] Su varyantlar TAMAMLANMADI -- tabloda oyle "
            "isaretlendi, raporda SESSIZCE atlanamaz:",
            file=sys.stderr,
        )
        for _, item in incomplete.iterrows():
            detail = item.get("eksik_dosyalar")
            suffix = f" (eksik: {detail})" if isinstance(detail, str) and detail else ""
            print(f"   {item['variant']}: {item['durum']}{suffix}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
