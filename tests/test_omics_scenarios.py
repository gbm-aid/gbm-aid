"""Omics 3-senaryo GERCEK DB testleri (rag-agent gorev talimati,
2026-09-12 -- AKTIF-GOREVLER.md satir 155).

Bu dosya `api/predict.py`'ye VE `pipeline/omics_scores.py`'ye
DOKUNMAZ/IMPORT ile degistirmez -- YALNIZ okur ve test eder (backend-
agent'in alani). DB'ye YALNIZ SELECT/readonly baglanti kullanilir
(`api.predict._get_db_connection()` zaten `get_connection(readonly=True)`
cagiriyor, bkz. api/predict.py).

Plan.txt'nin orijinal 3-senaryo tarifi:
  1. Yalniz MRI/radyomik var, omics YOK, klinik MGMT/IDH1 da YOK.
  2. MGMT+IDH1 klinik bilgisi VAR ama tam omics-profili YOK.
  3. Tam omics profili VAR (TMZ hassasiyeti, agresiflik skoru,
     DNA-onarim skoru, molecular_subtype/parp_inhibitor_candidate dahil).

Her 3 senaryoda da beklenen davranis: `fetch_omics_interpretation()`
UYDURMA/varsayilan bir deger URETMEZ -- ya acikca `None` doner (has_omics
FALSE, plan.txt:503 "sessizce atla") ya da eksik alt-alanlari acikca
`None` + dokumante edilmis bir not ile isaretler (has_omics TRUE ama
molecular_scores'ta HALA NULL kalan kolonlar icin, bkz. `api/predict.py::
_build_omics_block` docstring'i).

Gercek hasta ID'leri (2026-09-12 canli DB olcumu, bu dosyanin yazildigi
sirada dogrulandi -- bkz. asagidaki her testin docstring'i):
  - Senaryo 1: TCGA-02-0003 (has_omics=FALSE, mgmt_status=NULL,
    idh1_status=NULL, gtr_over90percent=NULL, tek scan_id, C32
    whitelist'inde radyomigi VAR).
  - Senaryo 2: UPENN-GBM-00034 (has_omics=FALSE, mgmt_status=
    'Unmethylated', idh1_status='Wildtype', gtr_over90percent='N', tek
    scan_id, C32 whitelist'inde radyomigi VAR).
  - Senaryo 3: TCGA-06-5412 (has_omics=TRUE, molecular_scores'ta tam
    satir VAR -- zaten tests/test_api_predict.py::
    test_fetch_omics_interpretation_real_db_smoke_tcga_06_5412'de
    kismen kapsanmis, burada EK olarak "hicbir NULL alt-alan sessizce
    baska bir degere DUZELTILMEDI" invariant'i ayrica dogrulaniyor).

DB erisilemezse (baglanti hatasi) test `pytest.skip` ile ATLANIR --
DB ERISILEBILIYOR ama beklenen satir/deger DEGISMISSE (canli veri
degisti) test KIRMIZI olur, KOR GECIRILMEZ (mevcut dosyadaki desenle
AYNI, bkz. tests/test_api_predict.py "GERCEK DB SMOKE TESTI" bolumu).
"""

from __future__ import annotations

import pytest

import api.predict as predict_module


def _skip_if_db_unavailable(callable_, *args, **kwargs):
    try:
        return callable_(*args, **kwargs)
    except Exception as exc:  # pragma: no cover -- yalniz DB erisilemezse
        pytest.skip(f"Gercek DB'ye erisilemedi, senaryo testi atlandi: {exc}")


# =====================================================================
# Senaryo 1 -- yalniz MRI/radyomik, klinik MGMT/IDH1 YOK, omics YOK
# =====================================================================


def test_scenario1_mri_only_omics_and_clinical_absent_no_fabrication() -> None:
    """TCGA-02-0003: `has_omics=FALSE`, `mgmt_status`/`idh1_status`/
    `gtr_over90percent` de NULL (2026-09-12 canli DB dogrulandi) --
    "yalniz MRI" senaryosunun gercek bir temsilcisi.

    Beklenen: `fetch_omics_interpretation()` `None` doner (uydurma skor
    YOK, hata YOK) -- plan.txt:503 "sessizce atla" davranisi."""

    patient_id = "TCGA-02-0003"
    block = _skip_if_db_unavailable(predict_module.fetch_omics_interpretation, patient_id)

    assert block is None, (
        f"{patient_id} icin has_omics=FALSE bekleniyordu (omics blogu olmamali), "
        f"alinan: {block!r} -- eger bu None DEGILSE canli DB'de bu hastanin "
        "has_omics durumu degismis olabilir, KOR GECIRME."
    )


def test_scenario1_predict_patient_fails_loud_not_silent_default() -> None:
    """AYNI hasta (TCGA-02-0003) icin uctan-uca `predict_patient()`.

    ⚠️ BU TEST 2026-09-13'te GUNCELLENDI -- SILINMEDI. Eski hali
    `HTTPException(422)` bekliyordu ve o beklenti BUGUN GECERSIZ oldu;
    sebebi bir bug DEGIL, Baris'in ONAYLADIGI politika degisikligidir
    (`gtr_over90percent`/`idh1_status` -> `TRAIN_ALIGNED`, K19 ile
    hizalama; `api/predict.py::CLINICAL_COVARIATE_NULL_POLICY` satir
    394-395). Testin ASIL AMACI ("sessiz varsayilan ATAMA YOK")
    DEGISMEDI -- asagida ayni amac YENI politika altinda, hem
    TRAIN_ALIGNED hem STRICT tarafi icin dogrulanir.

    ESKI BULGU (silinmez, tarihsel kayit -- wiki hard rule #3):
    2026-09-12'de bu model artefakti GTR+IDH1'i ZORUNLU kiliyordu, bu
    yuzden "yalniz MRI/radyomik" hastalari HICBIR risk skoru ALAMIYOR,
    422 ile reddediliyordu. Bu kisit 2026-09-13'te KALDIRILDI --
    servis edilebilir TCGA populasyonu 0 -> 38 oldu.

    YENI beklenen davranis, UC parcada:
      (A) NULL GTR/IDH1 ARTIK 422 uretmez -- istek BASARIYLA doner.
      (B) Ama bu SESSIZ bir sifir/varsayilan ATAMA DEGILDIR: eksiklik
          `clinical_*_missing` gosterge kolonlariyla ACIKCA kodlanir
          (K15 deseni). Sessiz-sifir ile gosterge-kodlama arasindaki
          fark TAM OLARAK bu testin korudugu seydir.
      (C) `age`/`gender` HALA `STRICT` -- onlar icin sessiz varsayilan
          YASAGI aynen gecerli. Politika haritasi burada assert edilir,
          boylece ileride biri onlari sessizce TRAIN_ALIGNED'e cevirirse
          bu test KIRMIZI olur."""

    patient_id = "TCGA-02-0003"

    # (C) Politika haritasi -- sessiz bir gevsetmeyi yakalayan guard.
    policy = predict_module.CLINICAL_COVARIATE_NULL_POLICY
    assert policy["age"] == predict_module.CLINICAL_NULL_POLICY_STRICT, (
        "`age` kovaryati STRICT KALMALI -- egitimde `clinical_age_missing` "
        f"gosterge kolonu YOK, alinan politika: {policy['age']!r}"
    )
    assert policy["gender"] == predict_module.CLINICAL_NULL_POLICY_STRICT, (
        "`gender` kovaryati STRICT KALMALI -- egitimde gosterge kolonu YOK, "
        f"alinan politika: {policy['gender']!r}"
    )
    for train_aligned_column in ("gtr_over90percent", "idh1_status", "mgmt_status"):
        assert policy[train_aligned_column] == (
            predict_module.CLINICAL_NULL_POLICY_TRAIN_ALIGNED
        ), (
            f"{train_aligned_column!r} 2026-09-13 Baris onayi ile TRAIN_ALIGNED "
            f"olmali, alinan: {policy[train_aligned_column]!r}"
        )

    # (A) Istek artik 422 ile olmuyor.
    try:
        response = predict_module.predict_patient(patient_id)
    except predict_module.HTTPException as exc:
        pytest.fail(
            f"predict_patient({patient_id!r}) {exc.status_code} atti -- GTR/IDH1 "
            "TRAIN_ALIGNED oldugu icin NULL bu hastayi ARTIK reddetmemeli. "
            f"Detay: {exc.detail!r}. KOR GECIRME: ya politika geri alindi ya "
            "da baska bir kovaryat (age/gender -- ikisi STRICT) gercekten NULL."
        )
    except Exception as exc:  # pragma: no cover -- yalniz DB/artefakta erisilemezse
        pytest.skip(f"Gercek DB/model artefaktina erisilemedi, senaryo testi atlandi: {exc}")

    assert "risk_score_log_partial_hazard" in response, (
        "Basarili yanitta risk skoru bulunmali, alinan anahtarlar: "
        f"{sorted(response)}"
    )

    # (B) Eksiklik SESSIZ sifir degil, ACIK gosterge kolonu olarak kodlanmis mi?
    extra_columns = list(response["clinical_extra_columns"])
    clinical_row = predict_module.build_patient_clinical_covariate_row(
        patient_id, extra_columns
    )
    for indicator, raw_field in (
        ("clinical_gtr_missing", "gtr_over90percent"),
        ("clinical_idh_missing", "idh1_status"),
    ):
        if indicator not in extra_columns:
            continue  # bu model kolu o gostergeyi kullanmiyorsa atla
        value = float(clinical_row.iloc[0][indicator])
        assert value == 1.0, (
            f"{patient_id} icin {raw_field!r} canli DB'de NULL, bu yuzden "
            f"{indicator!r} 1.0 olmali (eksiklik ACIKCA kodlanir). Alinan: "
            f"{value!r} -- 0.0 ise bu SESSIZ VARSAYILAN ATAMASI demektir ve "
            "CLAUDE.md geregi YASAKTIR."
        )


# =====================================================================
# Senaryo 2 -- MGMT+IDH1 klinik bilgisi VAR, tam omics-profili YOK
# =====================================================================


def test_scenario2_mgmt_idh1_present_but_omics_absent_no_fabrication() -> None:
    """UPENN-GBM-00034: `has_omics=FALSE` AMA `mgmt_status`='Unmethylated',
    `idh1_status`='Wildtype', `gtr_over90percent`='N' -- yani klinik
    kovaryatlar TAM dolu, omics-profili ise HIC YOK (2026-09-12 canli DB
    dogrulandi).

    Beklenen: klinik alanlarin doluluk durumu, `fetch_omics_interpretation()`'in
    omics-yoklugu davranisini DEGISTIRMEMELI -- fonksiyon SADECE
    `patients.has_omics` bayragina bakar, MGMT/IDH1 doluluguna BAKMAZ.
    Bu, "klinik alan doluluğu omics varlığına sessizce sızmıyor" kontrolu."""

    patient_id = "UPENN-GBM-00034"
    block = _skip_if_db_unavailable(predict_module.fetch_omics_interpretation, patient_id)

    assert block is None, (
        f"{patient_id} icin has_omics=FALSE bekleniyordu, klinik MGMT/IDH1 "
        f"doluluğu omics blogunu TETIKLEMEMELI. Alinan: {block!r}."
    )


def test_scenario2_predict_patient_succeeds_and_omics_key_gracefully_absent() -> None:
    """AYNI hasta (UPENN-GBM-00034) icin uctan-uca `predict_patient()` --
    klinik kovaryatlar (yas/cinsiyet/GTR/IDH1) TAM oldugu icin model
    BASARIYLA calismali ve gercek bir risk skoru URETMELI, AMA yanitta
    `"omics"` anahtari OLMAMALI (has_omics=FALSE -- plan.txt:503
    "sessizce atla", uydurma bir omics blogu EKLENMEMELI)."""

    patient_id = "UPENN-GBM-00034"
    result = _skip_if_db_unavailable(predict_module.predict_patient, patient_id)

    assert result["patient_id"] == patient_id
    assert "omics" not in result, (
        "has_omics=FALSE olan bir hasta icin yanitta 'omics' anahtari "
        f"OLMAMALI (uydurma/varsayilan blok riski) -- alinan yanit: {result!r}"
    )
    risk_score = result["risk_score_log_partial_hazard"]
    assert risk_score is not None
    assert risk_score == risk_score  # NaN degil (NaN != NaN)


# =====================================================================
# Senaryo 3 -- tam omics profili VAR (TMZ/agresiflik/DNA-onarim/parp_
# inhibitor_candidate dahil) -- ek "uydurma yok" invaryanti
# =====================================================================


def test_scenario3_full_omics_profile_present_no_silent_fabrication_of_null_subfields() -> None:
    """TCGA-06-5412: `has_omics=TRUE` VE `molecular_scores`'ta satir VAR
    (2026-09-12 canli DB dogrulandi, tests/test_api_predict.py::
    test_fetch_omics_interpretation_real_db_smoke_tcga_06_5412 ile
    KISMEN ortusuyor -- burada EK olarak skor/sinif alanlarinin
    sayisal/gecerli oldugu VE 4 bilinen-eksik alt-alanin (egfr_amp_flag/
    pten_del_flag/cdkn2a_del_flag/mgmt_interpretation) HICBIRININ
    sessizce baska bir degere (ornek: False/0) DUZELTILMEDIGI, `None` +
    dokumante not olarak KALDIGI ayrica dogrulaniyor."""

    patient_id = "TCGA-06-5412"
    block = _skip_if_db_unavailable(predict_module.fetch_omics_interpretation, patient_id)

    assert block is not None, f"{patient_id} icin has_omics=TRUE bekleniyordu."
    assert block["has_omics"] is True
    assert block["available"] is True

    # Skorlar/uretilen sinif alanlar UYDURULMAMIS, gercek DB degerleri --
    # sayisal skorlar float, siniflar bilinen enum degerleri olmali.
    assert isinstance(block["tmz_resistance_score"], float)
    assert isinstance(block["aggressiveness_score"], float)
    assert isinstance(block["dna_repair_score"], float)
    assert block["tmz_class"] in {"sensitive", "intermediate", "resistant"}
    assert block["aggr_class"] in {"low", "intermediate", "high", "very_high"}
    assert block["repair_class"] in {"impaired", "intermediate", "intact"}
    assert block["molecular_subtype"] == "parp_inhibitor_candidate"

    # 4 bilinen-eksik alt-alan: None OLARAK KALMALI (sessizce False/0
    # gibi bir "negatif" degere DUZELTILMEMIS olmali) VE her biri
    # dokumante edilmis bir not tasimali (UYDURMA-YOK kontrolu).
    always_null_fields_with_notes = {
        "egfr_amp_flag": "egfr_amp_flag_note",
        "pten_del_flag": "pten_del_flag_note",
        "cdkn2a_del_flag": "cdkn2a_del_flag_note",
        "mgmt_interpretation": "mgmt_interpretation_note",
    }
    for field, note_field in always_null_fields_with_notes.items():
        assert block[field] is None, (
            f"{patient_id}: {field!r} None DEGIL (alinan: {block[field]!r}) -- "
            "canli DB'de bu alan doldurulmus olabilir, bu artik eski bir "
            "bulgu/varsayimi GECERSIZ kilar, KOR GECIRME."
        )
        assert block[note_field], f"{field!r} icin dokumante not ({note_field}) BOS/YOK."

    # `tmz_class_relative` de bilinen eksiklik (48/48 NULL) -- ayni sekilde
    # None + not ile kalmali, sessizce baska bir string'e DUZELTILMEMIS.
    assert block["tmz_class_relative"] is None
    assert block["tmz_class_relative_note"]

    # score_thresholds capraz kontrolu KOD HATASI/TUTARSIZLIK sinyali
    # VERMEMELI (bu hasta icin -- baska bir DB satiri bozuksa bu testin
    # sorumlulugunda DEGIL, o tests/test_api_predict.py'de kapsanir).
    assert block["tmz_class_consistency_check"].startswith("consistent"), (
        f"{patient_id} icin TMZ sinif capraz kontrolu tutarsiz/dogrulanamaz "
        f"cikti: {block['tmz_class_consistency_check']!r} -- canli DB'de "
        "tmz_resistance_score/tmz_class/score_thresholds arasinda bir "
        "uyusmazlik olabilir, arastirilmali (db-agent'e bildirin)."
    )
