"""``POST /analyze_patient`` -- Hafta 5 uctan uca orkestrasyon (plan.txt:165-171,
523-529, 547-549; v45.txt Bolum 9 "uctan uca akis").

KAPSAM (v1, gorev talimati -- api/predict.py ile AYNI sinir): SADECE DB'de
C32 radyomigi ZATEN VAR olan hastalar desteklenir. Harmonizasyon/
segmentasyon/radyomik-cikarim adimlari (plan.txt'nin "HER HASTADA TEKRAR"
zincirinin ilk 3 adimi) bu endpoint'te CANLI CALISTIRILMAZ -- Mert'in
harmonizasyon/segmentasyon islevleri, `api/predict.py::fetch_c32_wt_row`
zaten diskteki/DB'deki HAZIR radyomigi okuyor, bu davranis DEGISTIRILMEDI.

BU DOSYA HICBIR PIPELINE FONKSIYONUNU YENIDEN YAZMAZ -- SADECE zincirler:
  1) Cox risk skoru + SHAP + (varsa) omics yorumu:
     `api.predict.predict_patient()` DOGRUDAN cagrilir (kopyalanmadi).
     Bu fonksiyon zaten kendi ICINDE `fetch_omics_interpretation()`'i
     cagirip sonuca `omics` alanini EKLIYOR (bkz. api/predict.py Adim 7
     yorumu) -- burada AYRICA cagrilmiyor (cift-cagri/DRY ihlali olurdu).

     ⚠️ 2026-09-13 DUZELTME (Baris onayi, B-1 mimari kusuru): BU ADIM
     ARTIK "HEPSI-YA-HIC" DEGIL. Onceki hali (yukarida, tarihsel not
     olarak birakildi) `predict_patient()`'in firlattigi HER
     HTTPException'i AYNEN yukari yayiyordu -- bu, Cox 422/503
     dondugunde FAISS/RAG/buyume-simulasyonu bloklarinin HICBIRININ
     uretilememesine yol aciyordu (canli kanit: `Patient-002` icin
     `predict_patient()` coklu-tarama nedeniyle 422 donuyor, bu da
     TUM `/analyze_patient` istegini 422'ye dusuruyordu -- buyume
     simulasyonu paneli bu hastada HICBIR ZAMAN gorunmuyordu, oysa
     LUMIERE kohortu icin tam da bu blok URETILEBILIR durumdaydi).

     ISTEK-FATAL vs BLOK-FATAL AYRIMI (`_build_risk_and_xgboost_blocks()`
     icinde uygulanir -- 2026-09-13'e kadar bu fonksiyonun adi
     `_build_risk_block()` idi; TEK bir jenerik mekanizma, XGBoost bloğu
     da AYNI mekanizmayi kullanir [madde 5], ozel-durum kodu
     COGALTILMADI):
       a) `HTTPException(status_code=404, ...)` (`PatientNotFoundError`)
          -> ISTEK-FATAL, AYNEN yukari yayilir. Gerekce: hasta `patients`
          tablosunda HIC YOK -- bu durumda FAISS/RAG/buyume-simulasyonu
          bloklarinin da (kendi ic mantiklariyla) URETEBILECEGI bir sey
          YOK, "hasta X icin kismi sonuc" ifadesi ANLAMSIZ (hasta X
          mevcut degil). Diger bloklar da hasta kimligine bagli oldugu
          icin bu TEK istek-fatal durumdur.
       b) `psycopg2.OperationalError` (DB'ye HIC ULASILAMIYOR -- baglanti
          timeout/reddedildi) -> ISTEK-FATAL, `HTTPException(503, ...)`'e
          CEVRILIR ve yukari yayilir. Gerekce: FAISS/buyume-simulasyonu
          bloklari da AYNI DB'yi okur -- DB tamamen erisilemezken "200 +
          hepsi available:false" donmek, SISTEMIK bir kesintiyi sanki
          "bu hasta icin kismi bir eksiklik" gibi GIZLER; 503 ile ACIKCA
          "sistem su an erisilemez, tekrar dene" mesaji verilir.
       c) BUNLARIN DISINDAKI HER SEY -- `HTTPException` (422 radyomik/
          klinik kovaryat/coklu-tarama/bolge sorunu, 501 desteklenmeyen
          kovaryat semasi, 500 artefakt/SHAP tutarlilik hatasi, 503
          CHECKPOINT dosyasi yok [DB'den BAGIMSIZ, disk dosyasi eksik])
          VE beklenmedik herhangi bir `Exception` (ornek: artefakt
          pickle bicimi taninmiyor) -- BLOK-FATAL: `risk` bloğu
          `available: False` + `error_status_code`/`error_detail` ile
          doner, istek 200 devam eder, DIGER bloklar (FAISS/RAG/buyume-
          simulasyonu) BAGIMSIZ olarak URETILMEYE DEVAM EDER. Gerekce:
          bunlarin HICBIRI "hasta/istek gecersiz" anlamina GELMEZ, sadece
          "Cox modeli SU AN bu hasta/artefakt icin bir sayi URETEMIYOR"
          anlamina gelir.
     SHAP/omics Cox'a DOGRUDAN BAGIMLIDIR (AYNI `predict_patient()`
     cagrisinda hesaplanir) -- risk bloğu `available:False` oldugunda
     `shap_available`/`omics_available` de ACIKCA `False` doner, uydurma
     bir SHAP/omics degeri URETILMEZ (bagimlilik zinciri gorunur kalir).
  2) FAISS benzer-hasta: `api.similar.get_similar_patients()` DOGRUDAN
     cagrilir. Bu adim OPSIYONEL/ZENGINLESTIRME kabul edilir -- fonksiyonun
     firlattigi `HTTPException` (404/422/503) YAKALANIR ve `available:
     False` + orijinal `status_code`/`detail` tasiyan gorunur bir blok
     dondurulur (pipeline DURMAZ). UCSF gibi v1 FAISS indeksine hic
     girmemis bir hasta icin `similar.py`'nin 422 sozlesmesi boylece
     KORUNUR (bilgi kaybolmaz, sadece HTTP-seviyesinde degil blok-
     seviyesinde tasinir) -- gorev talimatinin "UCSF hastasi icin 422
     davranisi korunur" maddesi budur.
  3) RAG literatur ozeti: `pipeline.rag_pipeline.generate_literature_
     summary()` cagrilir -- bu fonksiyon KENDI SOZLESMESI geregi HICBIR
     ZAMAN exception firlatmaz (bkz. o modulun dokstring'i), ama burada
     YINE DE savunmaci bir try/except ile sarilir (CLAUDE.md ilkesi:
     bir sozlesmeye korkorune guvenip tek nokta-of-failure yaratma) --
     beklenmedik bir hata gelirse pipeline yine DURMAZ, `available: False`
     + hata metniyle devam eder.
  4) Buyume simulasyonu (Hafta 4, `pipeline/growth_simulation.py`) --
     2026-09-13 Baris onayiyla ("XGBoost tarafi kesin okeyse, v3b ile tam
     uyumluluk gosterdiyse" -- bu sart 2026-09-12'de karsilandi, bkz.
     AKTIF-GOREVLER.md) CANLI ZINCIRE BAGLANDI. `_build_growth_simulation_
     block()` `pipeline/growth_simulation.py`'nin var olan fonksiyonlarini
     (fetch_lumiere_wt_volume_series/fit_patient_growth_curves/
     fetch_lumiere_rano_labels/assign_patient_rano_group/growth_rate_by_
     rano_group/project_volume_range) DOGRUDAN cagirir -- o modulun
     mantigi BURADA TEKRAR YAZILMADI, sadece zincirlendi.

     TASARIM KARARI (Secenek A -- GERCEK ZAMANLI hesap, Secenek B'nin
     "artifacts/week4/growth_simulation_v3/ CSV'lerini oku" YERINE):
     canli olcum (2026-09-13) CSV'lerle BIREBIR AYNI sonucu uretiyor
     (ayni deterministik parametreler: min_visits=3, min_group_n=6,
     max_iqr_ratio_to_reference=5.0 -- `tools/run_growth_simulation_
     lumiere.py`'nin varsayilanlariyla hizali), bu yuzden B'nin
     "donuk/incelenmis sayi" avantaji burada YOK; A'nin DB buyudukce
     otomatik guncel kalma avantaji VAR. Maliyet: her istekte TUM LUMIERE
     kohortu (90 hasta/585 tarama) icin curve_fit YENIDEN calisir (cache
     YOK, canli olculdu <1s, AÇIK RISK olarak beyan edilir -- DB
     buyurse/yeni LUMIERE ziyareti eklenirse yeniden degerlendirilmeli).

     ⚠️ KAPSAM KISITI (pipeline'in KENDI kilitli sozlesmesi, BURADA
     EKLENMEDI): `fetch_lumiere_wt_volume_series()` SADECE `patient_id
     ILIKE 'Patient-%'` (LUMIERE) satirlarini doner -- UCSF/TCGA/UPenn
     gibi tek-zaman-noktali hastalar icin seri HER ZAMAN BOS'tur. Demo
     hastalari `UCSF-PDGM-167` ve `TCGA-06-5412` LUMIERE kohortunda
     DEGIL -- ikisi icin de bu blok `available:false` doner (canli DB'de
     2026-09-13'te dogrulandi: ikisi de 0 satir). Bu SESSIZCE degil,
     `not_available_reason` alaninda ACIKCA soylenir.

  5) XGBoost ikinci katman (SHADOW) -- 2026-09-13 (backend-agent-J,
     Baris onayi "karar 7"). `api.predict.predict_patient()` yanitinin
     `xgboost` blogu `/analyze_patient` ciktisina AYNEN TASINIR
     (`_propagate_xgboost_block()`), YENIDEN HESAPLANMAZ: bu blok
     `api/predict.py::build_xgboost_shadow_block()` tarafindan, `predict_
     patient()` cagrisinin ICINDE zaten uretiliyor (bkz. o dosyanin
     "XGBOOST IKINCI KATMAN -- shadow" bolumu). Burada TEK hesap noktasi
     ilkesi korunur -- olasilik bu dosyada HICBIR SEKILDE hesaplanmaz.

     ⚠️ ALAN ADI DUZELTILDI: `xgboost_recurrence` -> `xgboost_12mo_
     survival`. ESKI AD YANLISTI: "recurrence" nuks/progresyon ima
     ediyordu, oysa egitilen hedef `target_12mo_survival` (12 ayi GORME
     sinifi, `pipeline/xgboost_model.py::define_twelve_month_survival_
     target`). Eski ad GECIS ALANI OLARAK DA TUTULMADI (temiz kesme):
     (a) alanin TEK tuketicisi bu dosyanin kendi testiydi -- canli kod
     tabaninda baska hicbir yerde okunmuyordu (`demo/`, `api/main.py`,
     diger testler tarandi: 0 isabet); (b) arayuz ekibine verilen
     sartname "bu alana gore kod yazmayin" uyarisini TASIYOR; (c) yanlis
     adi yanitta tutmak, duzeltmenin amacini (etiket = gercek) bozardi --
     iki ad ayni anda servis edilirse hangisinin dogru oldugu yine
     belirsiz kalirdi. Bunun YERINE blogun ICINE `previous_block_name` +
     `previous_block_name_note` konur: eski adi arayan biri sessiz bir
     `KeyError` yerine ACIK bir iz bulur.
     ⚠️ BILINEN BAYAT ATIF (bu gorevde DUZELTILEMEDI -- `api/predict.py`
     kapsam disiydi): `api/predict.py` modul dokstring'i satir ~237 hala
     "`api/analyze_patient.py` bu modeli `xgboost_recurrence` ... diye
     adlandiriyor" diyor. Yorum satiridir, davranisi ETKILEMEZ.

     HATA FELSEFESI (B-1 deseniyle AYNI -- BLOK-FATAL): XGBoost blogu
     uretilemezse `available: false` + sebep doner, istek 200 DEVAM
     EDER, `summary.unavailable_blocks`'a DUSER. UYDURMA bir olasilik
     ASLA dondurulmez (`probability_12_month_survival` anahtari hata
     durumunda HIC eklenmez, `None` olarak da degil -- `api/predict.py::
     _xgboost_block_unavailable()` ile ayni sozlesme). Iki ek blok-fatal
     dal BU DOSYADA tanimlidir: (i) Cox blogu BLOK-FATAL olarak
     dusmusse `predict_patient()` yaniti HIC YOKTUR -> tasinacak blok da
     yoktur (bagimlilik zinciri ACIKCA yazilir; blok burada YENIDEN
     HESAPLANMAZ, cunku tek hesap noktasi `predict.py`dir); (ii)
     `predict_patient()` 200 donduyse ama yanit sozlesmesi bozuksa
     (`xgboost` anahtari yok / `available` yok / `available:true` iken
     olasilik-yon-statu alanlari eksik) -> SESSIZCE yarim blok servis
     etmek YERINE `available:false` + sozlesme-ihlali sebebi doner.
     Ozellikle `direction_note` (yuksek olasilik = IYI prognoz, Cox
     riskinin TERSI) ve `status="shadow"` / `is_primary_decision_model:
     false` KAYBOLAMAZ -- bunlar olmadan olasilik servis EDILMEZ.

     (Eski hali, tarihsel not: "XGBoost nuks/progresyon modeli bu
     orkestrasyona HALA BAGLANMADI ... Yanitta bu alan HER ZAMAN
     `available: false` + `not_available_reason` ile doner" -- 2026-09-13
     ADIM 2'de `/predict` zincirine baglandiktan sonra bu ifade
     GECERSIZLESTI: sabit bir `available:false` dondurmek, canli
     calisan bir modeli "yok" gostermek anlamina geliyordu.)

RISK SKORU YORUMU (Nisa'nin Hafta 5 kontrol maddesi): `risk_score_log_
partial_hazard` bir OLASILIK/sagkalim yuzdesi DEGILDIR -- Cox orantisal-
hazard modelinin GORECELI risk skorudur (egitim kohortunun ortalama
kovaryat profiline gore log-hazard-orani). Bu not yanitta HER ZAMAN
`risk.risk_score_interpretation_note` alaninda acikca tasinir.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import psycopg2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import api.predict as predict_module
import api.similar as similar_module
import db_connection as db_connection_module
import pipeline.growth_simulation as growth_simulation_module
import pipeline.rag_pipeline as rag_pipeline_module

router = APIRouter()

_logger = logging.getLogger("api.analyze_patient")

RISK_SCORE_INTERPRETATION_NOTE = (
    "risk_score_log_partial_hazard bir OLASILIK/sagkalim yuzdesi DEGILDIR -- "
    "lifelines.CoxPHFitter.predict_log_partial_hazard() ciktisidir, yani "
    "egitim kohortunun ortalama kovaryat profiline GORE goreceli bir "
    "log-hazard-orani skorudur. hazard_ratio_partial_hazard (exp olcegi) "
    "bilgi amaclidir; SHAP degerleri bu exp olcegini ACIKLAMAZ, yalniz "
    "log-partial-hazard olcegini aciklar (bkz. api/predict.py modul "
    "dokstring'i 'SHAP OLCEGI')."
)

RISK_BLOCK_DEPENDENCY_NOTE = (
    "shap_available/omics_available Cox risk skoruna DOGRUDAN BAGIMLIDIR -- "
    "SHAP degerleri VE omics yorumu AYNI `api.predict.predict_patient()` "
    "cagrisinda hesaplanir (bkz. api/predict.py Adim 7). Cox uretilemedigi "
    "icin bu ikisi de uretilemedi -- uydurma bir SHAP/omics degeri "
    "DONDURULMEDI, bagimlilik zinciri burada ACIKCA belirtilir."
)

# ISTEK-FATAL vs BLOK-FATAL AYRIMI -- bkz. modul dokstring'i madde 1.
# `predict_patient()`'in firlattigi bu TEK status_code istek-fatal kabul
# edilir (hasta HIC YOK) -- gerisi (422/500/501/503-checkpoint) VE
# beklenmedik Exception'lar BLOK-FATAL'dir.
_RISK_BLOCK_REQUEST_FATAL_HTTP_STATUS = 404

_PROJECTION_MONTHS_DEFAULT = 6.0

# =====================================================================
# XGBoost ikinci katman (SHADOW) -- 2026-09-13, backend-agent-J.
# Bkz. modul dokstring'i madde 5.
# =====================================================================

# `/analyze_patient` yanitindaki UST-SEVIYE blok adi. ESKI ADI
# (`xgboost_recurrence`) YANLISTI -- egitilen hedef nuks/progresyon
# DEGIL, 12-ay sagkalim sinifidir. Yeni ad hem model ailesini
# (`xgboost`, `/predict`'teki blok adi + `model_registry` satiriyla ayni
# kelime) hem HEDEFI (`12mo_survival`) tasir; boylece ayni "etiket !=
# gercek" hatasinin tekrarlanmasi icin acikca yalan yazmak gerekir.
XGBOOST_BLOCK_KEY = "xgboost_12mo_survival"

# Eski/yanlis ad -- yanitta UST SEVIYEDE ARTIK YOK (temiz kesme,
# gerekcesi modul dokstring'i madde 5'te), yalniz blogun ICINDE bir iz
# olarak tasinir.
XGBOOST_PREVIOUS_BLOCK_KEY = "xgboost_recurrence"

# `api.predict.predict_patient()` yanitindaki kaynak anahtar.
_PREDICT_XGBOOST_RESPONSE_KEY = "xgboost"

XGBOOST_PREVIOUS_BLOCK_NAME_NOTE = (
    f"Bu blok 2026-09-13'e kadar {XGBOOST_PREVIOUS_BLOCK_KEY!r} adiyla ve "
    "HER ZAMAN `available:false` sabitiyle donuyordu. Ad YANLISTI: "
    "'recurrence' nuks/progresyon ima ediyor, oysa egitilen hedef "
    "`target_12mo_survival` (12 ayi GORME sinifi). Eski ad geri UYUMLULUK "
    "icin DE tutulmadi -- iki ad ayni anda servis edilirse hangisinin "
    "dogru oldugu belirsiz kalirdi."
)

XGBOOST_PROPAGATION_NOTE = (
    "Bu blok api.predict.predict_patient() yanitinin 'xgboost' alanindan "
    "AYNEN TASINDI (api/predict.py::build_xgboost_shadow_block). "
    "/analyze_patient bu degerlerin HICBIRINI yeniden hesaplamaz -- tek "
    "hesap noktasi api/predict.py'dir."
)

XGBOOST_BLOCK_DEPENDENCY_NOTE = (
    "XGBoost (shadow) blogu api.predict.predict_patient() yanitinin ICINDE "
    "uretilir -- Cox blogu uretilemedigi icin o yanit HIC OLUSMADI, "
    "dolayisiyla tasinacak bir XGBoost blogu da YOK. Bu dosyada YENIDEN "
    "HESAPLANMAZ (tek hesap noktasi ilkesi). UYDURMA bir olasilik "
    "DONDURULMEDI."
)

# `available: true` bir XGBoost blogunun TASIMAK ZORUNDA oldugu alanlar.
# `direction_note` burada KRITIKTIR: yuksek olasilik = IYI prognoz, yani
# Cox risk skorunun TERSI yonu. Bu not kaybolursa olasilik ters okunabilir
# -- o yuzden eksikse olasilik SERVIS EDILMEZ (fail-closed).
_REQUIRED_AVAILABLE_XGBOOST_KEYS = (
    "probability_12_month_survival",
    "direction_note",
    "status",
    "is_primary_decision_model",
)


class AnalyzePatientRequest(BaseModel):
    """`raw/plan/plan.txt:169` ornegiyle AYNI govde sozlesmesi:
    `curl -X POST /analyze_patient -d '{"patient_id": "TCGA-02-0003"}'`."""

    patient_id: str


def _risk_block_unavailable(
    *, error_status_code: int | None, error_detail: str
) -> dict[str, Any]:
    """`risk` bloğu icin BLOK-FATAL (istek DEGIL) durumlarda donen ortak
    govde -- bkz. modul dokstring'i madde 1(c). `shap_available`/
    `omics_available` HER ZAMAN `False` (bagimlilik zinciri, uydurma YOK)."""

    return {
        "available": False,
        "error_status_code": error_status_code,
        "error_detail": error_detail,
        "shap_available": False,
        "omics_available": False,
        "not_available_reason": (
            f"Cox risk skoru uretilemedi: {error_detail} {RISK_BLOCK_DEPENDENCY_NOTE}"
        ),
    }


def _xgboost_block_unavailable(*, error_detail: str) -> dict[str, Any]:
    """XGBoost (shadow) blogu icin BLOK-FATAL govdesi -- `api/predict.py::
    _xgboost_block_unavailable()` ve `_risk_block_unavailable()` (B-1) ile
    AYNI sozlesme: `probability_12_month_survival` anahtari HIC EKLENMEZ
    (`None` olarak da degil), yani UYDURMA/belirsiz bir olasilik
    DONDURULMEZ.

    `status`/`is_primary_decision_model` burada da GORUNUR kalir -- blok
    uretilememis olsa bile modelin golge statusu beyan edilir."""

    return {
        "available": False,
        "status": predict_module.XGBOOST_MODEL_STATUS,
        "is_primary_decision_model": False,
        "error_detail": error_detail,
        "not_available_reason": error_detail,
        "previous_block_name": XGBOOST_PREVIOUS_BLOCK_KEY,
        "previous_block_name_note": XGBOOST_PREVIOUS_BLOCK_NAME_NOTE,
    }


def _propagate_xgboost_block(cox_and_omics: dict[str, Any]) -> dict[str, Any]:
    """`predict_patient()` yanitinin `xgboost` blogunu `/analyze_patient`
    ciktisina TASIR -- hicbir degeri YENIDEN HESAPLAMAZ (bkz. modul
    dokstring'i madde 5).

    Kaynak sozluk MUTASYONA UGRATILMAZ (sig kopya alinir). Eklenen
    orkestrasyon-metadata anahtarlari, kaynakta ZATEN varsa EZILMEZ --
    `api/predict.py`'nin kendi degeri her zaman kazanir.

    Sozlesme ihlalleri SESSIZ GECILMEZ: `xgboost` anahtari yoksa, sozluk
    degilse, `available` tasimiyorsa veya `available:true` iken
    olasilik/yon/statu alanlarindan biri eksikse -> BLOK-FATAL
    (`available:false` + acik sebep). Ozellikle `direction_note` eksikken
    olasilik servis EDILMEZ: o not olmadan "yuksek olasilik = IYI prognoz"
    bilgisi kaybolur ve deger Cox riskiyle AYNI yonde okunabilir."""

    source = cox_and_omics.get(_PREDICT_XGBOOST_RESPONSE_KEY)
    if not isinstance(source, dict):
        return _xgboost_block_unavailable(
            error_detail=(
                "api.predict.predict_patient() yaniti bir "
                f"{_PREDICT_XGBOOST_RESPONSE_KEY!r} SOZLUGU tasimiyor "
                f"(gelen tip: {type(source).__name__}) -- sozlesme ihlali. "
                "Bu dosyada olasilik YENIDEN HESAPLANMAZ (tek hesap noktasi "
                "api/predict.py), uydurma bir deger DONDURULMEZ."
            )
        )

    if "available" not in source:
        return _xgboost_block_unavailable(
            error_detail=(
                f"api.predict.predict_patient()['{_PREDICT_XGBOOST_RESPONSE_KEY}'] "
                "blogunda `available` alani YOK -- sozlesme ihlali; blogun "
                "uretilip uretilmedigi belirsiz oldugu icin AVAILABLE "
                "SAYILMADI."
            )
        )

    block = dict(source)  # sig kopya -- kaynak sozluk DEGISTIRILMEZ

    if block.get("available") is True:
        missing = [key for key in _REQUIRED_AVAILABLE_XGBOOST_KEYS if key not in block]
        if missing:
            return _xgboost_block_unavailable(
                error_detail=(
                    "api.predict.predict_patient() `available:true` bir XGBoost "
                    f"blogu dondurdu ama ZORUNLU alanlar EKSIK: {missing}. "
                    "Yarim bir blok servis etmek yerine BLOK-FATAL kabul "
                    "edildi -- `direction_note` olmadan olasilik (yuksek = IYI "
                    "prognoz) Cox riskiyle TERS yonde okunabilirdi."
                )
            )

    for key, value in (
        ("propagated_from", f"api.predict.predict_patient()['{_PREDICT_XGBOOST_RESPONSE_KEY}']"),
        ("propagation_note", XGBOOST_PROPAGATION_NOTE),
        ("previous_block_name", XGBOOST_PREVIOUS_BLOCK_KEY),
        ("previous_block_name_note", XGBOOST_PREVIOUS_BLOCK_NAME_NOTE),
    ):
        if key not in block:  # api/predict.py'nin KENDI degeri EZILMEZ
            block[key] = value
    return block


def _build_risk_and_xgboost_blocks(
    patient_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cox risk skoru + SHAP + (varsa) omics katmani **VE** XGBoost
    (shadow) ikinci katman blogu -- IKISI DE TEK bir `api.predict.
    predict_patient()` cagrisindan gelir (o fonksiyon `xgboost` blogunu
    kendi ICINDE uretir), bu yuzden tek yerde kurulurlar. Cagri
    KOPYALANMADI/YENIDEN YAZILMADI; bkz. modul dokstring'i madde 1
    "ISTEK-FATAL vs BLOK-FATAL AYRIMI" ve madde 5.

    (2026-09-13'e kadar bu fonksiyonun adi `_build_risk_block()` idi ve
    YALNIZ `risk` blogunu donuyordu; `predict_patient()` yanitindan sadece
    belirli anahtarlar kopyalandigi icin ADIM 2'de eklenen `xgboost` blogu
    `/analyze_patient`'a PROPAGE ETMIYORDU -- ad, artik IKI blok
    dondurdugunu yansitacak sekilde degistirildi.)

    Sadece IKI durum ISTEK-FATAL kabul edilir (yukari yeniden firlatilir):
      1) `HTTPException(status_code=404)` -- hasta `patients` tablosunda
         HIC YOK.
      2) `psycopg2.OperationalError` -- DB'ye HIC ULASILAMIYOR, `HTTPException
         (503, ...)`'e cevrilir.
    Bunlarin DISINDAKI HER seyin (baska HTTPException status_code'lari VEYA
    beklenmedik herhangi bir Exception) BLOK-FATAL oldugu kabul edilir --
    `_risk_block_unavailable()` + `_xgboost_block_unavailable()` doner,
    pipeline DURMAZ.
    """

    try:
        cox_and_omics = predict_module.predict_patient(patient_id)
    except HTTPException as exc:
        if exc.status_code == _RISK_BLOCK_REQUEST_FATAL_HTTP_STATUS:
            raise  # istek-fatal (hasta HIC YOK) -- AYNEN yukari yayilir
        detail = str(exc.detail)
        return (
            _risk_block_unavailable(
                error_status_code=exc.status_code, error_detail=detail
            ),
            _xgboost_block_unavailable(
                error_detail=(
                    f"Cox/predict adimi HTTP {exc.status_code} ile basarisiz "
                    f"oldu: {detail} {XGBOOST_BLOCK_DEPENDENCY_NOTE}"
                )
            ),
        )
    except psycopg2.OperationalError as exc:
        # istek-fatal -- FAISS/RAG/buyume-simulasyonu bloklari da AYNI DB'yi
        # okur, DB tamamen erisilemezken kismi bir "200" yaniltici olurdu.
        raise HTTPException(
            status_code=503,
            detail=(
                f"DB'ye ulasilamadi (psycopg2.OperationalError): {exc}. Bu "
                "istek-fatal kabul edilir -- FAISS/RAG/buyume-simulasyonu "
                "bloklari da AYNI DB'yi okur, DB tamamen erisilemezken "
                "kismi bir '200' yanit SISTEMIK kesintiyi gizlerdi."
            ),
        ) from exc
    except Exception as exc:  # savunmaci -- beklenmedik hata istegi DURDURMAZ
        _logger.warning(
            "risk (cox) blogu beklenmedik hatayla basarisiz oldu "
            "(patient_id=%s): %s", patient_id, exc,
        )
        detail = f"Beklenmeyen hata: {type(exc).__name__}: {exc}"
        return (
            _risk_block_unavailable(error_status_code=None, error_detail=detail),
            _xgboost_block_unavailable(
                error_detail=(
                    "Cox/predict adimi beklenmedik bir hatayla basarisiz oldu: "
                    f"{detail} {XGBOOST_BLOCK_DEPENDENCY_NOTE}"
                )
            ),
        )

    risk_block: dict[str, Any] = {
        "available": True,
        "model_arm": cox_and_omics["model_arm"],
        "final_features": cox_and_omics["final_features"],
        "clinical_extra_columns": cox_and_omics["clinical_extra_columns"],
        "risk_score_log_partial_hazard": cox_and_omics["risk_score_log_partial_hazard"],
        "risk_score_interpretation_note": RISK_SCORE_INTERPRETATION_NOTE,
        "hazard_ratio_partial_hazard": cox_and_omics["hazard_ratio_partial_hazard"],
        "shap_available": True,
        "shap_base_value": cox_and_omics["shap_base_value"],
        "shap_values": cox_and_omics["shap_values"],
        "shap_additivity_check_abs_diff": cox_and_omics["shap_additivity_check_abs_diff"],
        "omics_available": "omics" in cox_and_omics,
    }
    if "omics" in cox_and_omics:
        risk_block["omics"] = cox_and_omics["omics"]
    return risk_block, _propagate_xgboost_block(cox_and_omics)


def _block_not_available_reason(block: dict[str, Any]) -> str:
    """Ozet alani (`summary.unavailable_blocks`) icin bir blogun okunabilir
    sebebini cikartir -- her blok ayni anahtar adini KULLANMIYOR
    (`not_available_reason` / `error_detail` / `reason`), bu fonksiyon
    UCUNU de destekler, hicbirini bulamazsa "belirtilmemis" doner (sessizce
    bos birakilmaz)."""

    for key in ("not_available_reason", "error_detail", "reason"):
        value = block.get(key)
        if value:
            return str(value)
    return "belirtilmemis"


def _build_summary(blocks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Ust-seviye ozet -- istegin KISMEN basarili oldugunu (hangi bloklarin
    uretildigini/uretilemedigini ve NEDEN) TEK bir yerden okunabilir kilar
    (gorev talimati madde 3 -- "sistem neyi uretebildigini/uretemedigini
    soyluyor")."""

    unavailable = [
        {"block": name, "reason": _block_not_available_reason(block)}
        for name, block in blocks.items()
        if not block.get("available", False)
    ]
    return {
        "all_blocks_available": len(unavailable) == 0,
        "unavailable_blocks": unavailable,
    }


def _build_similar_patients_block(patient_id: str) -> dict[str, Any]:
    """FAISS benzer-hasta katmani -- OPSIYONEL/zenginlestirme. `api.similar.
    get_similar_patients()`'in firlattigi HTTPException (404/422/503)
    YAKALANIR, pipeline DURMAZ -- ama sozlesme (status_code/detail) GORUNUR
    sekilde blok icine tasinir (bkz. modul dokstring'i madde 2)."""

    try:
        block = similar_module.get_similar_patients(
            patient_id,
            k=similar_module.DEFAULT_K,
            idh1_status=None,
            mgmt_status=None,
        )
    except HTTPException as exc:
        return {
            "available": False,
            "error_status_code": exc.status_code,
            "error_detail": exc.detail,
        }
    except Exception as exc:  # savunmaci -- beklenmedik hata pipeline'i DURDURMAZ
        _logger.warning(
            "similar_patients blogu beklenmedik hatayla basarisiz oldu "
            "(patient_id=%s): %s", patient_id, exc,
        )
        return {
            "available": False,
            "error_status_code": None,
            "error_detail": f"Beklenmeyen hata: {type(exc).__name__}: {exc}",
        }

    block["available"] = True
    return block


def _build_literature_block(patient_id: str) -> dict[str, Any]:
    """RAG literatur katmani. `generate_literature_summary()` KENDI
    sozlesmesi geregi exception FIRLATMAZ (bkz. pipeline/rag_pipeline.py) --
    yine de savunmaci try/except ile sarilir (bkz. modul dokstring'i madde
    3). Sonuc bir dataclass agaci oldugu icin `dataclasses.asdict()` ile
    JSON-serilestirilebilir bir sozluge cevrilir."""

    try:
        result = rag_pipeline_module.generate_literature_summary(patient_id)
    except Exception as exc:  # sozlesme "asla firlatmaz" dese de savunmaci
        _logger.warning(
            "literature blogu beklenmedik hatayla basarisiz oldu "
            "(patient_id=%s): %s", patient_id, exc,
        )
        return {
            "available": False,
            "reason": f"Beklenmeyen hata: {type(exc).__name__}: {exc}",
        }

    return dataclasses.asdict(result)


def _build_growth_simulation_block(patient_id: str) -> dict[str, Any]:
    """Buyume simulasyonu katmani -- OPSIYONEL/zenginlestirme, GERCEK
    ZAMANLI hesaplanir (bkz. modul dokstring'i madde 4 "TASARIM KARARI").

    `pipeline/growth_simulation.py`'nin fonksiyonlarini DOGRUDAN cagirir,
    hicbirini yeniden yazmaz. Her basarisizlik/uygunsuzluk durumu ACIKCA
    `not_available_reason` ile doner -- sessiz bos sonuc veya uydurma
    projeksiyon YOK.
    """

    try:
        conn = db_connection_module.get_connection(readonly=True)
    except Exception as exc:  # savunmaci -- DB erisilemezse pipeline DURMAZ
        _logger.warning(
            "growth_simulation blogu DB baglantisi kuramadi (patient_id=%s): %s",
            patient_id, exc,
        )
        return {
            "available": False,
            "not_available_reason": f"DB baglantisi kurulamadi: {type(exc).__name__}: {exc}",
        }

    try:
        series_df = growth_simulation_module.fetch_lumiere_wt_volume_series(conn)
        rano_df = growth_simulation_module.fetch_lumiere_rano_labels(conn)
    except Exception as exc:  # savunmaci -- beklenmedik hata pipeline'i DURDURMAZ
        _logger.warning(
            "growth_simulation blogu beklenmedik hatayla basarisiz oldu "
            "(patient_id=%s): %s", patient_id, exc,
        )
        return {
            "available": False,
            "not_available_reason": f"Beklenmeyen hata (DB okuma): {type(exc).__name__}: {exc}",
        }
    finally:
        conn.close()

    patient_series = series_df[series_df["patient_id"] == patient_id]
    if patient_series.empty:
        return {
            "available": False,
            "not_available_reason": (
                f"'{patient_id}' icin LUMIERE (Patient-XXX) longitudinal C32 "
                "WT hacim serisinde kayit bulunamadi -- buyume simulasyonu su "
                "an SADECE LUMIERE kohortunun (segmentation_tool="
                f"{growth_simulation_module.SEGMENTATION_TOOL!r}, tumor_region="
                f"{growth_simulation_module.TUMOR_REGION!r}) longitudinal C32 "
                "verisiyle calisiyor -- pipeline/growth_simulation.py'nin "
                "kendi kilitli sorgu sozlesmesi (`patient_id ILIKE 'Patient-"
                "%'`). Tek-zaman-noktali hastalar (UCSF/TCGA/UPenn) icin "
                "uretilebilecek bir seri YOK -- uydurulmus bir projeksiyon "
                "DONDURULMEDI."
            ),
        }

    try:
        fits = growth_simulation_module.fit_patient_growth_curves(series_df)
    except Exception as exc:  # savunmaci
        _logger.warning(
            "growth_simulation fit blogu beklenmedik hatayla basarisiz oldu "
            "(patient_id=%s): %s", patient_id, exc,
        )
        return {
            "available": False,
            "not_available_reason": f"Beklenmeyen hata (egri fit): {type(exc).__name__}: {exc}",
        }

    patient_fit = next((f for f in fits if f.patient_id == patient_id), None)
    if patient_fit is None:  # pragma: no cover -- seri bulunduysa fit de bulunmali
        return {
            "available": False,
            "not_available_reason": (
                f"'{patient_id}' hacim serisinde bulundu ama fit sonuc "
                "listesinde yok -- beklenmeyen bir ic tutarsizlik."
            ),
        }

    if patient_fit.fit_status != "ok":
        return {
            "available": False,
            "n_visits": patient_fit.n_visits,
            "fit_status": patient_fit.fit_status,
            "not_available_reason": (
                f"'{patient_id}' icin buyume egrisi fit edilemedi -- "
                f"fit_status={patient_fit.fit_status!r}, sebep: "
                f"{patient_fit.fail_reason!r}."
            ),
        }

    fit_block: dict[str, Any] = {
        "n_visits": patient_fit.n_visits,
        "fit_status": patient_fit.fit_status,
        "best_model": patient_fit.best_model,
        "best_r": patient_fit.best_r,
        "simulation_interpretation_note": (
            "BU BIR SIMULASYONDUR, GERCEK TAHMIN DEGILDIR -- mevcut hacim "
            "serisine fit edilmis Gompertz/lojistik egrinin buyume "
            "katsayisidir (bkz. pipeline/growth_simulation.py "
            "simulate_boundary_growth() docstring'i)."
        ),
    }

    rano_group_df = growth_simulation_module.assign_patient_rano_group(rano_df)
    patient_group_row = rano_group_df[rano_group_df["patient_id"] == patient_id]
    if patient_group_row.empty:
        return {
            "available": True,
            **fit_block,
            "rano_group": None,
            "projection": None,
            "projection_not_available_reason": (
                f"'{patient_id}' icin gecerli bir RANO yanit etiketi (PD/SD/"
                "PR/CR) yok (yalniz Pre-Op/Post-Op veya bilinmeyen kayitli) -- "
                "6 aylik hacim projeksiyon araligi RANO-grubu buyume-"
                "katsayisi IQR'sine dayandigi icin hesaplanamadi. Nokta-"
                "tahmini (best_r) yukarida mevcut."
            ),
        }

    rano_group = str(patient_group_row.iloc[0]["rano_group"])
    fits_df = growth_simulation_module.growth_fits_to_dataframe(fits)
    group_dist_df = growth_simulation_module.growth_rate_by_rano_group(fits_df, rano_group_df)
    group_row = group_dist_df[group_dist_df["rano_group"] == rano_group]

    if group_row.empty or not bool(group_row.iloc[0]["interpretable"]):
        group_n = int(group_row.iloc[0]["n"]) if not group_row.empty else 0
        return {
            "available": True,
            **fit_block,
            "rano_group": rano_group,
            "rano_group_n": group_n,
            "rano_group_interpretable": False,
            "projection": None,
            "projection_not_available_reason": (
                f"'{rano_group}' RANO grubunun buyume-katsayisi dagilimi "
                "istatistiksel olarak guvenilmez isaretli (interpretable="
                "False -- n<6 VE/VEYA grup IQR'i referans grubun IQR'inin "
                "5 katini asiyor, bkz. CLAUDE.md 2026-09-11 Baris karari) -- "
                "6 aylik projeksiyon araligi bu grup icin URETILMEDI. Nokta-"
                "tahmini (best_r) yukarida mevcut."
            ),
        }

    # 2026-09-13 DUZELTME (G1): `v0` SON ziyaretin hacmi -- dolayisiyla
    # projeksiyonun baslangic noktasi da SON ziyaretin HAFTASI olmalidir.
    # Onceki halde `project_volume_range()` hedef haftayi kosulsuz
    # `months*4,345 = 26,07` (ILK taramadan itibaren MUTLAK) aliyordu; son
    # ziyaret hafta 26,07'den sonraysa (LUMIERE'de tipik) yuzde degisim
    # GERIYE DONUK bir karsilastirmaya donusuyordu. Olculen etki
    # (`Patient-028`, son ziyaret hafta 38): alt uc %+32,96 -> %-0,47,
    # yani ISARET DEGISIYOR. Bkz. decisions/2026-09-13-projeksiyon-ufku-
    # duzeltmesi.md ve `project_volume_range()` dokstring'i.
    last_visit = patient_series.sort_values("x_week").iloc[-1]
    v0 = float(last_visit["tumor_volume_mm3"])
    last_visit_week = float(last_visit["x_week"])
    template_params = (
        patient_fit.gompertz.params
        if patient_fit.best_model == "gompertz"
        else patient_fit.logistic.params
    )
    group_stats = group_row.iloc[0]

    try:
        projection = growth_simulation_module.project_volume_range(
            v0=v0,
            model=patient_fit.best_model,
            template_params=template_params,
            r_low=float(group_stats["iqr_low"]),
            r_median=float(group_stats["median_r"]),
            r_high=float(group_stats["iqr_high"]),
            months=_PROJECTION_MONTHS_DEFAULT,
            from_week=last_visit_week,
        )
    except Exception as exc:  # savunmaci
        _logger.warning(
            "growth_simulation projeksiyon blogu beklenmedik hatayla "
            "basarisiz oldu (patient_id=%s): %s", patient_id, exc,
        )
        return {
            "available": True,
            **fit_block,
            "rano_group": rano_group,
            "rano_group_n": int(group_stats["n"]),
            "rano_group_interpretable": True,
            "projection": None,
            "projection_not_available_reason": (
                f"Beklenmeyen hata (projeksiyon): {type(exc).__name__}: {exc}"
            ),
        }

    return {
        "available": True,
        **fit_block,
        "rano_group": rano_group,
        "rano_group_n": int(group_stats["n"]),
        "rano_group_interpretable": True,
        "projection": dataclasses.asdict(projection),
    }


@router.post("/analyze_patient")
def analyze_patient(request: AnalyzePatientRequest) -> dict[str, Any]:
    patient_id = request.patient_id

    # 1) Cox risk skoru + SHAP + (varsa) omics -- 2026-09-13 DUZELTME (B-1):
    #    ARTIK graceful degradation UYGULANIR -- SADECE hasta HIC YOK (404)
    #    veya DB'ye HIC ULASILAMIYOR (psycopg2.OperationalError -> 503) ise
    #    istek-fatal olarak yukari yayilir (bkz. `_build_risk_and_xgboost_
    #    blocks()` ve
    #    modul dokstring'i madde 1 "ISTEK-FATAL vs BLOK-FATAL AYRIMI").
    #    Digger tum durumlarda (422/500/501/503-checkpoint, beklenmedik
    #    Exception) `risk_block["available"]=False` doner, istek 200 DEVAM
    #    EDER -- FAISS/RAG/buyume-simulasyonu BAGIMSIZ olarak uretilir.
    #    2026-09-13 (backend-agent-J): AYNI cagri XGBoost (shadow) blogunu
    #    da tasir -- `predict_patient()` onu kendi icinde uretiyor, burada
    #    YENIDEN HESAPLANMAZ (bkz. modul dokstring'i madde 5).
    risk_block, xgboost_block = _build_risk_and_xgboost_blocks(patient_id)

    # 2) FAISS benzer-hasta -- OPSIYONEL, graceful degradation.
    similar_block = _build_similar_patients_block(patient_id)

    # 3) RAG literatur ozeti -- OPSIYONEL, graceful degradation.
    literature_block = _build_literature_block(patient_id)

    # 4) Buyume simulasyonu -- CANLI ZINCIRE BAGLANDI (2026-09-13), OPSIYONEL/
    #    graceful degradation.
    growth_simulation_block = _build_growth_simulation_block(patient_id)

    # 5) XGBoost (shadow) ikinci katman -- YUKARIDA (1) ile AYNI
    #    `predict_patient()` cagrisindan TASINDI. Blok adi
    #    `xgboost_recurrence` DEGIL (yanlis etiket, bkz. modul dokstring'i
    #    madde 5) -- `XGBOOST_BLOCK_KEY`.
    blocks = {
        "risk": risk_block,
        "similar_patients": similar_block,
        "literature": literature_block,
        "growth_simulation": growth_simulation_block,
        XGBOOST_BLOCK_KEY: xgboost_block,
    }

    return {
        "patient_id": patient_id,
        # Ust-seviye "kismi basari" isareti -- gorev talimati madde 3.
        "summary": _build_summary(blocks),
        **blocks,
    }
