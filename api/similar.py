"""``GET /patient/{patient_id}/similar`` -- çift-katmanlı FAISS benzer-hasta sorgusu.

Plan referansı: `raw/plan/plan.txt:452-454` ("Sultan (Risk API) -- GET
/patient/{id}/similar endpoint'i -- iki FAISS indeksini sorgular"), akış:
`raw/plan/plan.txt:142-145` / `raw/mimari/v45.txt:1074-1095` ("Her yeni
hasta için önce geniş `clinical_radiomics_faiss` sorgulanır. Hasta
`omics_profiles`'da mevcutsa **ek olarak** `molecular_omics_faiss` de
sorgulanır.").

BU DOSYA NE OKUR (yeniden İNŞA ETMEZ -- `tools/rebuild_faiss_indexes.py`
Ege'nin işi, buradan İMPORT EDİLMEZ, dosyalar DEĞİŞTİRİLMEZ):
  - `artifacts/week3/faiss_index/{clinical_radiomics.index,patient_order.csv}`
    -- 722 hasta (UPenn 611 + LUMIERE 72 + TCGA 39), K1 Tasarım A (ham C32,
    ComBat'sız), K2 LUMIERE Pre-Op kuralı.
  - `artifacts/week3/molecular_omics/{molecular_omics.index,patient_order.csv}`
    -- 48 hasta, TCGA-Omics, 11-boyut.
`tools/faiss_similar_patients_query_demo.py`'nin sorgu MANTIĞI (self-query
+ k+1 çekip kendini çıkarma) buraya TAŞINDI -- o script'in kendisi
çağrılmıyor, kopyala-yapıştır da değil: burada bir HTTP endpoint'e uygun
hata sözleşmesi (404/422/503) + iki katmanlı akış + isteğe bağlı filtre
eklenerek YENİDEN yazıldı.

🔒 KİLİTLİ KURAL (`decisions/2026-08-14-faiss-v1-vektor-icerigi.md`):
vektör mesafesi YALNIZ `WT × STABLE_FEATURES_ICC60` (93 özellik,
`tools/rebuild_faiss_indexes.py` tarafından dondurulmuş `scaler.json` ile
üretildi). Yaş/KPS/MGMT/IDH1/GTR/tedavi bu dosyada da mesafeye HİÇ
GİRMEZ -- yalnız isteğe bağlı FİLTRE (`idh1_status`/`mgmt_status` query
parametreleri) ve GÖRÜNTÜLEME (`results[].{age,gender,...}`) için
kullanılır. Filtre "alan varsa filtrele, yoksa filtreleme" ilkesiyle
ZARİFÇE bozulur (`_passes_optional_filter`): eksik (`NULL`) değerli bir
komşu filtre nedeniyle SESSİZCE ELENMEZ, yalnız DEĞERİ olup filtreyle
UYUŞMAYAN komşular çıkarılır.

✅ KAPANDI (2026-09-13, karar 34 -- Barış onayı) -- filtre etiketleri artık
KANONİKLEŞTİRİLİYOR. Eski hâli (bu dosyanın 2026-09-13 ÖNCESİ davranışı,
tarihsel kayıt olarak korunur): *"Bu dosya bu etiketleri BİRLEŞTİRMEZ/
UYDURMAZ -- filtre karşılaştırması yalnız `strip().lower()` ile normalize
edilmiş TAM eşleşmedir; kaynaklar arası bir eşleme kararı (ör. "WT" ==
"Wildtype") VERİLMEDİ."* O davranışın ÖLÇÜLEN sonucu bir SESSİZ ELEME idi:
canlı DB'de `idh1_status` UPenn/UCSF'te `"Wildtype"/"Mutated"/"NOS/NEC"`,
LUMIERE'de `"WT"/"wt"/"R132H mut"/"IDH1 neg, Sequencing required"` ile
geliyor (aynı şekilde `mgmt_status`: UPenn `"Methylated"/"Unmethylated"`,
LUMIERE `"methylated"/"not methylated"`) -> `?idh1_status=Wildtype`
sorgusu, v1 indeksindeki 49 LUMIERE `"WT"` hastasının HEPSİNİ eliyordu
(değeri DOLU olduğu için "zarifçe bozulma" korumasına da girmiyorlardı).

🔒 TEK KAYNAK (drift YASAK): eşleme sözlüğü BU DOSYADA TANIMLANMAZ,
`api/predict.py`'den İMPORT EDİLİR (`IDH1_LABEL_NORMALIZATION_MAP` /
`MGMT_LABEL_NORMALIZATION_MAP`, `decisions/2026-09-13-lumiere-idh-mgmt-
etiket-normalizasyonu.md`, db-agent-F). Gerekçe: `/predict` ile `/similar`
AYNI hastayı AYNI kategoride görmek ZORUNDA; sözlüğü kopyalamak iki
endpoint'in sessizce ayrışmasına açık kapı bırakırdı. `api/main.py` ZATEN
`api.predict`'i (satır 22) `api.similar`'dan (satır 27) ÖNCE import ettiği
için bu import sunucu sürecine EK BAĞIMLILIK AĞIRLIĞI GETİRMEZ; `api.
predict` modül seviyesinde checkpoint YÜKLEMEZ (`shap` da fonksiyon içi),
ölçülen tek başına import maliyeti 0,44 s. Döngüsel import YOK --
`api/predict.py` `api/similar.py`'yi import ETMEZ (yalnız yorumda anar).

⚠️ KALAN/BEYAN EDİLEN SAPMA (sessizce geçiştirilmiyor): sözlüğün İÇERİĞİ
tek kaynaktır ama UYGULAMA BİÇİMİ iki tarafta farklıdır -- `api/predict.py`
`pd.Series.replace()` ile TAM (harf-duyarlı) eşleşme yapar; burada
karşılaştırma ZATEN (2026-09-13 öncesinden beri) `strip().lower()`
tabanlı olduğu için eşleme de harf-duyarsız bir anahtarla kurulur
(`_build_case_insensitive_label_lookup`). Pratik fark YOK -- gerçek DB'deki
tüm varyantlar (`"WT"` VE `"wt"`) sözlükte AÇIKÇA listeli. Teorik fark:
sözlükte OLMAYAN bir harf-varyantı (ör. `"Wt"`) `/similar` tarafında
`Wildtype` sayılır, `/predict` tarafında 422 atar (sessiz DEĞİL, gürültülü
red). Bu asimetri BEYAN EDİLİR; kapatılması `api/predict.py`'ye dokunmayı
gerektirir ve bu görevin kapsamı DIŞINDADIR.

⚠️ `'IDH1 neg, Sequencing required'` (v1 indeksinde 7 hasta) `NOS/NEC`'e
eşlenir, `Wildtype`'a ASLA DEĞİL (db-agent-F'in kilitli kararı, K15
semantiği: "test yapıldı, sonuç KESİNLEŞMEMİŞ" != "bilinen wildtype").
Yani `?idh1_status=Wildtype` bu 7 hastayı DOĞRU ŞEKİLDE elemeye devam eder.

METADATA KAYNAĞI -- CANLI DB (bilinçli seçim, `clinical_metadata.csv`
DEĞİL): iki gerekçe -- (1) tazelik: `vital_status`/`survival_days` zamanla
değişebilir, dondurulmuş CSV inşa anındaki (2026-08-18 10:11 UTC) bir
STATİK görüntü; (2) tamlık: görev talimatının istediği `gtr_over90percent`
alanı `tools/rebuild_faiss_indexes.py::fetch_patients_metadata_frame()`'in
ürettiği `clinical_metadata.csv`'de HİÇ YOK (yalnız age/gender/kps_score/
mgmt_status/idh1_status/vital_status/survival_days var) -- CSV'den okunsa
bile GTR için AYRICA canlı DB'ye gitmek gerekirdi. Buna karşılık indeksin
KENDİSİ (hangi 722/48 hasta içeride, vektör değerleri) HER ZAMAN dondurulmuş
artefaktten okunur -- metadata "tazelik" tercihi vektör uzayını/mesafeyi
ETKİLEMEZ, yalnız görüntüleme/filtre katmanını etkiler.

🔴 BİLİNEN İŞLETİMSEL RİSK -- Windows Türkçe-karakter yol hatası (ITK için
zaten belgelenmiş AYNI hata sınıfı, bkz. `tests/test_no_resolve_path_
regression.py`, `tools/rebuild_faiss_indexes.py::write_index_artifacts()`
docstring'i "Windows'ta Türkçe-karakter path bug'ı"): `faiss.read_index()`
C++ `fopen()` kullanır, bu makinedeki `C:/Users/Barış/...` yolunu ÇÖZEMEZ.
`Path(__file__).resolve()` (BİLİNÇLİ OLARAK KULLANILMADI, `.absolute()`
kullanıldı) `subst` eşlemesini (`X:` sürücüsü) GERÇEK yola geri çözer ve
hata GERİ GELİR -- `.absolute()` sürücü harfini KORUR. **Ama bu, sunucu
SÜRECİNİN KENDİSİ `X:` (veya başka ASCII-safe bir subst) üzerinden
başlatılmasını GEREKTİRİR** -- süreç doğrudan `C:/Users/Barış/...` yolundan başlatılırsa
(ör. IDE "Run" düğmesi, gerçek yoldan `uvicorn api.main:app`) bu endpoint
HER istekte 503 döner. Ortam değişkenleri
(`GBMAID_FAISS_CLINICAL_RADIOMICS_INDEX_DIR`,
`GBMAID_FAISS_MOLECULAR_OMICS_INDEX_DIR`) operatöre açık bir ASCII-safe
yol vermek için EKLENDİ -- sessiz bir "otomatik düzeltme" YAPILMADI,
sorun 503 mesajında AÇIKÇA görünür.

HATA SÖZLEŞMESİ (CLAUDE.md: sessiz başarısızlık YOK):
  - İndeks dosyaları (index/patient_order.csv) eksik/tutarsız -> 503
  - Hasta `patients` tablosunda yok -> 404
  - Hasta var ama v1 `clinical_radiomics_faiss` indeksinde yok (ör. UCSF,
    K1 kararı gereği v1'e HİÇ dahil edilmedi) -> 422, sessiz boş liste YOK
  - `k` indeks boyutunu (kendisi hariç mevcut komşu sayısını) aşıyorsa ->
    mevcut kadarı döner, `warnings` alanına AÇIK not düşülür
  - Sorgulanan hastanın kendisi (mesafe ~0) top-k'dan HER ZAMAN çıkarılır
    (K1 taslağı §3 kuralı, `neighbor_patient_id == patient_id` eşleşmesiyle
    -- mesafe==0 eşiğiyle DEĞİL, iki farklı hastanın teorik olarak özdeş
    vektöre sahip olabileceği durumu YANLIŞLIKLA elemesin diye)
  - Omics katmanı: `patients.has_omics != TRUE` ise `molecular_omics_faiss`
    yanıttan TAMAMEN ÇIKARILIR (plan.txt:503 "sessizce atla" deseni,
    `api/predict.py::fetch_omics_interpretation` ile AYNI ilke). AMA
    `has_omics=TRUE` olup omics indeksinde satırı YOKSA (veri tutarsızlığı)
    veya omics indeks dosyaları eksikse bu SESSİZCE YUTULMAZ -- görünür
    `{"available": false, "note": ...}` bloğu döner, ana radyomik sonucu
    ETKİLEMEZ (predict.py'nin `available: False` anomalisi deseniyle aynı).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2.extras
from fastapi import APIRouter, HTTPException, Query

from db_connection import get_connection

# 🔒 TEK KAYNAK -- bkz. modül dokstring'i "TEK KAYNAK (drift YASAK)".
# Bu iki sözlük BURADA YENİDEN TANIMLANMAZ/KOPYALANMAZ; `api/predict.py`
# (db-agent-F, `decisions/2026-09-13-lumiere-idh-mgmt-etiket-
# normalizasyonu.md`) onların TEK otoriter sahibidir. İçeriği değişirse
# `/predict` ve `/similar` OTOMATİK olarak birlikte değişir.
from api.predict import (
    IDH1_LABEL_NORMALIZATION_MAP,
    MGMT_LABEL_NORMALIZATION_MAP,
)

router = APIRouter()

# `.absolute()` -- `.resolve()` DEĞİL (bkz. modül dokstring'i "BİLİNEN
# İŞLETİMSEL RİSK" + `tests/test_no_resolve_path_regression.py`).
REPO_ROOT = Path(__file__).absolute().parent.parent

DEFAULT_CLINICAL_RADIOMICS_INDEX_DIR = REPO_ROOT / "artifacts" / "week3" / "faiss_index"
DEFAULT_MOLECULAR_OMICS_INDEX_DIR = REPO_ROOT / "artifacts" / "week3" / "molecular_omics"
CLINICAL_RADIOMICS_INDEX_FILENAME = "clinical_radiomics.index"
MOLECULAR_OMICS_INDEX_FILENAME = "molecular_omics.index"

DEFAULT_K = 10


# =====================================================================
# 0) Etiket kanonikleştirme (karar 34) -- sözlük İMPORT edilir, BURADA
#    tanımlanmaz; burada yalnız FİLTRE karşılaştırmasına uygun bir
#    ERİŞİM biçimi (harf-duyarsız anahtar) türetilir.
# =====================================================================


def _build_case_insensitive_label_lookup(
    canonical_map: dict[str, str], field_name: str
) -> dict[str, str]:
    """`api/predict.py`'den import edilen TAM-eşleşmeli sözlükten,
    filtre karşılaştırmasına uygun `strip().lower()`-anahtarlı bir arama
    tablosu türetir.

    İki farklı ham etiket AYNI harf-duyarsız anahtara düşüp FARKLI kanonik
    değere işaret ediyorsa SESSİZCE biri seçilmez -- import anında
    `RuntimeError` fırlatılır. Böylece `api/predict.py`'deki sözlüğe
    ileride çelişkili bir anahtar eklenirse hata GÜRÜLTÜLÜ olur
    (CLAUDE.md: sessiz başarısızlık YOK). Aynı kanonik değere düşen
    harf-varyantları (ör. `"WT"` ve `"wt"` -> `Wildtype`) MEŞRUDUR ve
    çakışma SAYILMAZ."""

    lookup: dict[str, str] = {}
    for raw_label, canonical_label in canonical_map.items():
        key = str(raw_label).strip().lower()
        previous = lookup.get(key)
        if previous is not None and previous != canonical_label:
            raise RuntimeError(
                f"{field_name} etiket sözlüğünde harf-duyarsız ÇAKIŞMA: "
                f"{key!r} anahtarı hem {previous!r} hem {canonical_label!r} "
                "kanonik değerine işaret ediyor. `api/predict.py`'deki "
                f"sözlük ({field_name}) tutarsız -- /similar filtresi "
                "sessizce birini seçmek yerine DURDU."
            )
        lookup[key] = canonical_label
    return lookup


_IDH1_FILTER_LABEL_LOOKUP = _build_case_insensitive_label_lookup(
    IDH1_LABEL_NORMALIZATION_MAP, "idh1_status"
)
_MGMT_FILTER_LABEL_LOOKUP = _build_case_insensitive_label_lookup(
    MGMT_LABEL_NORMALIZATION_MAP, "mgmt_status"
)


def _canonicalize_label(value: Any, lookup: dict[str, str]) -> Any:
    """Ham etiketi kanonik forma çevirir. Sözlükte OLMAYAN bir değer
    DEĞİŞTİRİLMEDEN döner (`api/predict.py`'nin `pd.Series.replace()`
    davranışıyla AYNI ilke: harita kapsamadığı hiçbir şeyi UYDURMAZ).
    `None`/`NaN` DOKUNULMADAN geri döner -- "eksik değer ELENMEZ"
    korumasını (`_passes_optional_filter`) bozmamak için kritik."""

    if value is None:
        return value
    try:
        if pd.isna(value):
            return value
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return lookup.get(text.lower(), text)


def _clinical_radiomics_index_dir() -> Path:
    override = os.environ.get("GBMAID_FAISS_CLINICAL_RADIOMICS_INDEX_DIR")
    return Path(override) if override else DEFAULT_CLINICAL_RADIOMICS_INDEX_DIR


def _molecular_omics_index_dir() -> Path:
    override = os.environ.get("GBMAID_FAISS_MOLECULAR_OMICS_INDEX_DIR")
    return Path(override) if override else DEFAULT_MOLECULAR_OMICS_INDEX_DIR


class SimilarIndexArtifactsError(RuntimeError):
    """FAISS indeks dosyaları (`*.index`/`patient_order.csv`) eksik veya
    tutarsız (satır sayıları uyuşmuyor) -- sessizce boş/kısmi bir indeksle
    devam EDİLMEZ (CLAUDE.md: mock/sahte benzerlik sonucu üretme)."""


class PatientNotInIndexError(RuntimeError):
    """Hasta `patients` tablosunda var ama bu FAISS indeksinde yok."""


# =====================================================================
# 1) İndeks artifact'leri -- mtime-anahtarlı in-process cache (predict.py
#    ::load_cox_arm_result() deseniyle AYNI mantık, BAĞIMSIZ küçük kopya)
# =====================================================================


@dataclass
class IndexBundle:
    index: Any
    order: pd.DataFrame  # index=patient_id, kolonlar=[row_index, source]
    # 2026-08-19 (Codex HIGH bulgusu #2): eskiden `float` idi ve degeri
    # `max(index_mtime, order_mtime)` ile hesaplaniyordu. O yontem IKI
    # artefakti AYIRT ETMIYOR: indeks daha yeniyken `patient_order.csv`
    # degisir ama yeni mtime'i hala indeksinkinin ALTINDA kalirsa
    # `max()` DEGISMEZ -> onbellek gecerli sayilir, YENI CSV OKUNMAZ ve
    # satir-sayisi tutarlilik kontrolu YENIDEN CALISMAZ. Artik her iki
    # dosyanin (mtime_ns, size) ikilisi AYRI AYRI anahtarin parcasi.
    mtime_key: tuple[int, int, int, int]
    # karar 36 (2026-09-13): FAISS'in kendi iç kimlikleri (`search()`/
    # `reconstruct()`) HER ZAMAN 0..ntotal-1 KONUMSAL tamsayılardır.
    # Komşu kimliğini çözmek için `patient_order.csv`'nin SATIR SIRASINA
    # güvenmek yerine `row_index` KOLONUNDAN kurulmuş açık ters harita
    # kullanılır. `None` ise (ör. doğrudan kurulan birim-test bundle'ı)
    # ihtiyaç anında `_build_row_index_to_patient_id()` ile hesaplanır.
    row_index_to_patient_id: dict[int, str] | None = None


def _build_row_index_to_patient_id(order: pd.DataFrame) -> dict[int, str]:
    """`row_index` -> `patient_id` ters haritasını KOLONDAN kurar ve
    haritanın `range(len(order))` üzerine BİREBİR-ÖRTEN olduğunu doğrular.

    🔴 NEDEN (karar 36, 2026-09-13): bu fonksiyondan ÖNCE
    `_query_all_neighbors_sorted()` İKİ AYRI KONVANSİYONU karıştırıyordu --
    sorgu vektörünü `order.loc[pid, "row_index"]` (KOLON) ile, komşu
    kimliğini ise `order.reset_index().loc[int(idx)]` (KONUMSAL) ile
    çözüyordu. Bu, *"CSV satır sırası == `row_index`"* varsayımına
    dayanıyordu ve bu varsayım kodda HİÇBİR YERDE doğrulanmıyordu.
    Bugün doğru (ölçüldü: 722 satır, `row_index == range(722)`), ama CSV
    bir gün `patient_id`'ye göre SIRALANARAK yazılsaydı endpoint SESSİZCE
    YANLIŞ komşu kimlikleri döndürürdü -- mesafeler doğru, isimler yanlış:
    tespit edilmesi en zor hata sınıfı. Artık sıra ÖNEMSİZ (harita
    kolondan kurulur) ve kolon bozuksa hata GÜRÜLTÜLÜ (503)."""

    if "row_index" not in order.columns:
        raise SimilarIndexArtifactsError(
            "patient_order.csv'de `row_index` kolonu YOK -- komşu "
            "kimlikleri çözülemez, CSV satır sırasına SESSİZCE "
            "güvenilmedi (karar 36)."
        )

    try:
        row_indices = order["row_index"].astype(int)
    except (TypeError, ValueError) as exc:
        raise SimilarIndexArtifactsError(
            f"patient_order.csv'deki `row_index` kolonu TAMSAYI değil: {exc}"
        ) from exc

    mapping: dict[int, str] = {}
    for patient_id, row_index in zip(order.index, row_indices):
        if row_index in mapping:
            raise SimilarIndexArtifactsError(
                f"patient_order.csv'de `row_index` TEKRARLIYOR: {row_index} "
                f"hem {mapping[row_index]!r} hem {patient_id!r} için "
                "kayıtlı -- artefakt bozuk, sessizce devam EDİLMEDİ."
            )
        mapping[int(row_index)] = str(patient_id)

    expected = set(range(len(order)))
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))[:5]
        unexpected = sorted(set(mapping) - expected)[:5]
        raise SimilarIndexArtifactsError(
            f"patient_order.csv'deki `row_index` kolonu 0..{len(order) - 1} "
            "aralığını BİREBİR örtmüyor (FAISS iç kimlikleri her zaman bu "
            f"aralıktadır) -- eksik (ilk 5): {missing}, beklenmeyen (ilk 5): "
            f"{unexpected}. Artefakt bozuk, SESSİZCE yanlış komşu kimliği "
            "döndürmek yerine DURULDU (karar 36)."
        )

    return mapping


_index_bundle_cache_lock = threading.Lock()
_index_bundle_cache: dict[str, IndexBundle] = {}


def _load_index_bundle(index_dir: Path, index_filename: str) -> IndexBundle:
    """`index_dir`'deki FAISS indeksini + `patient_order.csv`'yi okur.

    Dosya mtime'i değişmediği sürece process içinde tekrar okumaz.

    `SimilarIndexArtifactsError` fırlatılan BEŞ durum (endpoint hepsini
    503'e çevirir -- 2026-08-19'da genişletildi, öncesinde yalnız ilk ve
    dördüncü kapsanıyordu; 5. madde 2026-09-13/karar 36'da eklendi):
      1. dosya YOK,
      2. FAISS indeksi VAR ama `faiss.read_index()` açamıyor,
      3. `patient_order.csv` VAR ama `pd.read_csv()` parse edemiyor,
      4. indeks ile `patient_order.csv` satır sayıları UYUŞMUYOR,
      5. satır sayısı DOĞRU ama `row_index` kolonu 0..n-1 aralığını
         birebir örtmüyor (yok/tekrarlı/aralık dışı/tamsayı değil) --
         bkz. `_build_row_index_to_patient_id()`.
    2 ve 3, Türkçe-karakterli yolun tipik belirtisidir (bkz. modül
    dokstring'i "BİLİNEN İŞLETİMSEL RİSK")."""

    import faiss

    index_path = index_dir / index_filename
    order_path = index_dir / "patient_order.csv"
    for path in (index_path, order_path):
        if not path.is_file():
            raise SimilarIndexArtifactsError(
                f"Beklenen FAISS artefaktı bulunamadı: {path}. "
                "`tools/rebuild_faiss_indexes.py --no-combat --apply` henüz "
                "çalışmamış olabilir, ya da GBMAID_FAISS_*_INDEX_DIR yanlış "
                "bir yolu işaret ediyor (bkz. modül dokstring'i 'BİLİNEN "
                "İŞLETİMSEL RİSK' -- Türkçe-karakter yol hatası da bu hatayı "
                "üretebilir)."
            )

    # 2026-08-19 (Codex MEDIUM bulgusu #3): `stat()` cagrilari YENI
    # try/except bloklarindan ONCE calisiyordu. Varlik kontrolu ile
    # `stat()` arasinda dosya silinirse/IO hatasi olursa `OSError`
    # DOGRUDAN yayilir, `SimilarIndexArtifactsError` yakalamaz ->
    # istemciye 503 yerine 500 doner. Artik ikisi de korumada.
    try:
        index_stat = index_path.stat()
        order_stat = order_path.stat()
    except OSError as exc:
        raise SimilarIndexArtifactsError(
            f"FAISS artefaktlarinin dosya bilgisi okunamadi ({index_dir}): "
            f"{type(exc).__name__}: {exc}. Artefaktlar es zamanli olarak "
            "yenileniyor ya da silinmis olabilir."
        ) from exc

    mtime_key = (
        index_stat.st_mtime_ns,
        index_stat.st_size,
        order_stat.st_mtime_ns,
        order_stat.st_size,
    )
    cache_key = str(index_path)
    with _index_bundle_cache_lock:
        cached = _index_bundle_cache.get(cache_key)
        if cached is not None and cached.mtime_key == mtime_key:
            return cached

        # 2026-08-19 DUZELTMESI (reviewer capraz dogrulamasi, MEDIUM
        # bulgu): bu iki cagri ONCEDEN try/except'siz idi. Yukaridaki
        # `path.is_file()` kontrolu yalniz dosyanin VAR OLUP OLMADIGINI
        # bakiyor; asil Turkce-karakter/izin/bozuk-dosya hatasi TAM
        # BURADA olusuyor. O durumda firlatilan RuntimeError/OSError
        # `SimilarIndexArtifactsError` OLMADIGI icin endpoint'in
        # `except SimilarIndexArtifactsError` blogu tetiklenmiyor ve
        # istemciye modul dokstring'inin VAAT ETTIGI bilgilendirici 503
        # yerine jenerik 500 doniyordu -- dokumantasyon/kod tutarsizligi.
        # Artik iki cagri da sarmalanip ayni istisnaya cevriliyor;
        # orijinal hata `from exc` ile ZINCIRLENIYOR (yutulmuyor).
        try:
            index = faiss.read_index(str(index_path))
        except Exception as exc:  # noqa: BLE001 -- faiss C++ katmani
            raise SimilarIndexArtifactsError(
                f"FAISS indeksi okunamadi: {index_path} "
                f"({type(exc).__name__}: {exc}). Dosya VAR ama ACILAMIYOR "
                "-- en olasi neden Turkce-karakterli yol (bkz. modul "
                "dokstring'i 'BILINEN ISLETIMSEL RISK': surec `X:` subst "
                "surucusu uzerinden baslatilmali)."
            ) from exc

        try:
            order = pd.read_csv(order_path, index_col="patient_id")
        except Exception as exc:  # noqa: BLE001 -- pandas parse/IO katmani
            raise SimilarIndexArtifactsError(
                f"patient_order.csv okunamadi: {order_path} "
                f"({type(exc).__name__}: {exc}). Dosya VAR ama "
                "PARSE EDILEMIYOR -- bos/yarim yazilmis olabilir, ya da "
                "Turkce-karakterli yol sorunu (bkz. modul dokstring'i)."
            ) from exc

        if index.ntotal != len(order):
            raise SimilarIndexArtifactsError(
                f"İndeks ({index.ntotal}) ve patient_order.csv ({len(order)}) "
                f"satır sayıları UYUŞMUYOR ({index_path}) -- artefaktler "
                "tutarsız olabilir, sessizce devam EDİLMEDİ."
            )

        # karar 36 -- satır SAYISI doğru olsa bile `row_index` kolonunun
        # FAISS'in 0..ntotal-1 iç kimlikleriyle HİZALI olduğu AYRICA
        # doğrulanır (satır sayısı guard'ı bunu YAKALAMAZ). Bilinçli
        # olarak satır-sayısı kontrolünden SONRA: uyuşmazlık durumunda
        # istemci daha spesifik olan "satır sayıları UYUŞMUYOR" mesajını
        # görmeye devam eder.
        row_index_to_patient_id = _build_row_index_to_patient_id(order)

        bundle = IndexBundle(
            index=index,
            order=order,
            mtime_key=mtime_key,
            row_index_to_patient_id=row_index_to_patient_id,
        )
        _index_bundle_cache[cache_key] = bundle
        return bundle


def _query_all_neighbors_sorted(bundle: IndexBundle, patient_id: str) -> pd.DataFrame:
    """`patient_id` için indeksteki TÜM komşuları (kendisi HARİÇ), L2
    mesafeye göre artan sırada döner (`tools/faiss_similar_patients_query_
    demo.py::query_similar_patients()` ile aynı self-query + hariç tutma
    mantığı -- ama `k` ile SINIRLAMAZ, tam liste döner; filtre/limit
    işlemi çağıran tarafta yapılır, çünkü filtre eksik hastaları elemez
    ve bu yüzden top-k'ya ULAŞMAK için filtre-öncesi TAM sıralı listeye
    ihtiyaç var).

    `patient_id` indekste yoksa `PatientNotInIndexError` fırlatır."""

    if patient_id not in bundle.order.index:
        raise PatientNotInIndexError(patient_id)

    row_index = int(bundle.order.loc[patient_id, "row_index"])
    query_vector = bundle.index.reconstruct(row_index).reshape(1, -1).astype(np.float32)

    n_total = bundle.index.ntotal
    distances, indices = bundle.index.search(query_vector, n_total)

    # karar 36: ters harita `row_index` KOLONUNDAN kurulur -- CSV'nin
    # SATIR SIRASI artık ÖNEMSİZ (eski `reset_index().loc[idx]` konumsal
    # erişimi, satır sırası `row_index`'ten farklı olduğunda sessizce
    # YANLIŞ hasta kimliği döndürürdü). `_load_index_bundle()` bunu zaten
    # doğrulayıp bundle'a koyar; doğrudan kurulan (birim test) bundle'lar
    # için burada hesaplanır -- doğrulama İKİ yolda da çalışır.
    row_index_to_patient_id = bundle.row_index_to_patient_id
    if row_index_to_patient_id is None:
        row_index_to_patient_id = _build_row_index_to_patient_id(bundle.order)

    rows: list[dict[str, Any]] = []
    for dist, idx in zip(distances.ravel(), indices.ravel()):
        if idx == -1:
            continue
        neighbor_id = row_index_to_patient_id[int(idx)]
        if neighbor_id == patient_id:
            continue  # kendisi -- HER ZAMAN hariç (bkz. modül dokstring'i)
        rows.append({"patient_id": neighbor_id, "l2_distance": float(dist)})

    return pd.DataFrame(rows, columns=["patient_id", "l2_distance"])


# =====================================================================
# 2) DB -- yalnız SELECT (hasta varlığı/has_omics + klinik metadata)
# =====================================================================


def _get_db_connection():
    return get_connection(readonly=True)


def _fetch_patient_existence_and_omics(patient_id: str) -> dict[str, bool]:
    conn = _get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT has_omics FROM patients WHERE patient_id = %s", (patient_id,)
        )
        row = cur.fetchone()
        cur.close()
    finally:
        conn.close()
    if row is None:
        return {"exists": False, "has_omics": False}
    return {"exists": True, "has_omics": bool(row.get("has_omics"))}


# 🔒 Bu kolonlar `decisions/2026-08-14-faiss-v1-vektor-icerigi.md` kararı
# gereği MESAFEYE GİRMEZ -- yalnız görüntüleme/isteğe bağlı filtre için.
_CLINICAL_METADATA_RAW_COLUMNS: tuple[str, ...] = (
    "age",
    "gender",
    "kps_score",
    "mgmt_status",
    "idh1_status",
    "gtr_over90percent",
    "vital_status",
    "survival_days",
)


def _fetch_clinical_metadata_frame(patient_ids: list[str]) -> pd.DataFrame:
    """Birden fazla `patient_id` için CANLI klinik metadata (bkz. modül
    dokstring'i "METADATA KAYNAĞI"). Index=`patient_id`."""

    columns = ["patient_id", "source", *_CLINICAL_METADATA_RAW_COLUMNS]
    if not patient_ids:
        return pd.DataFrame(columns=columns).set_index("patient_id")

    conn = _get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        columns_sql = ", ".join(f"p.{c}" for c in _CLINICAL_METADATA_RAW_COLUMNS)
        cur.execute(
            f"""
            SELECT p.patient_id, ds.source_name AS source, {columns_sql}
            FROM patients p
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE p.patient_id = ANY(%s)
            """,
            (patient_ids,),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    return frame.set_index("patient_id")


# =====================================================================
# 3) İsteğe bağlı filtre -- "alan varsa filtrele, yoksa filtreleme"
# =====================================================================


def _passes_optional_filter(
    value: Any,
    filter_value: str | None,
    *,
    label_lookup: dict[str, str] | None = None,
) -> bool:
    """`filter_value` verilmemişse HER ZAMAN True. Verilmişse: `value`
    eksik (`None`/`NaN`) olan bir komşu ASLA elenmez (zarifçe bozulma) --
    yalnız `value` DOLU ve filtreyle UYUŞMAYAN komşular elenir.

    `label_lookup` verilirse (karar 34) karşılaştırma HAM değerler
    üzerinde DEĞİL, İKİ TARAFI DA aynı sözlükten geçirilmiş KANONİK
    değerler üzerinde yapılır -- yani `value="WT"` (LUMIERE) ile
    `filter_value="Wildtype"` (UPenn sözlüğü) artık EŞLEŞİR. Simetriktir:
    `?idh1_status=WT` sorgusu da UPenn'in `"Wildtype"` hastalarını
    yakalar. `label_lookup=None` (varsayılan) ESKİ ham davranıştır ve
    yalnız sözlüğü olmayan alanlar/birim testleri için bırakılmıştır."""

    if filter_value is None:
        return True
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    if label_lookup is not None:
        value = _canonicalize_label(value, label_lookup)
        # `filter_value` SADECE gerçek bir `str` ise kanonikleştirilir.
        # `str(...)` ile ZORLA çevirmek, endpoint fonksiyonu DOĞRUDAN
        # (FastAPI olmadan) çağrıldığında varsayılan olarak gelen `Query`
        # NESNESİNİ sessizce bir etiket gibi karşılaştırmaya sokardı.
        # Bu satır o SESSİZ yolu bilinçli olarak KAPALI tutar -- aşağıdaki
        # `filter_value.strip()` böyle bir durumda (2026-09-13 öncesinde
        # olduğu gibi) GÜRÜLTÜLÜ `AttributeError` atmaya devam eder.
        if isinstance(filter_value, str):
            filter_value = _canonicalize_label(filter_value, label_lookup)

    return str(value).strip().lower() == filter_value.strip().lower()


def _num_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def _val_or_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


# =====================================================================
# 4) Ortak: sıralı komşu listesi -> metadata birleştir -> filtre -> top-k
# =====================================================================


def _build_similar_block(
    bundle: IndexBundle,
    patient_id: str,
    k: int,
    *,
    idh1_status: str | None,
    mgmt_status: str | None,
    result_columns: tuple[str, ...],
) -> dict[str, Any]:
    all_neighbors = _query_all_neighbors_sorted(bundle, patient_id)

    warnings: list[str] = []
    n_available = len(all_neighbors)
    if k > n_available:
        warnings.append(
            f"k={k} istendi ama indekste (kendisi hariç) yalnızca "
            f"{n_available} komşu var -- mevcut kadarı döndürüldü."
        )

    metadata = _fetch_clinical_metadata_frame(list(all_neighbors["patient_id"]))
    merged = all_neighbors.set_index("patient_id").join(metadata, how="left")

    filters_applied: dict[str, str] = {}
    if idh1_status is not None:
        filters_applied["idh1_status"] = idh1_status
    if mgmt_status is not None:
        filters_applied["mgmt_status"] = mgmt_status

    if filters_applied and not merged.empty:
        # karar 34: her iki alan da KANONİK etiketle karşılaştırılır --
        # sözlük `api/predict.py`'den gelir (TEK KAYNAK).
        mask = merged.apply(
            lambda row: (
                _passes_optional_filter(
                    row.get("idh1_status"),
                    idh1_status,
                    label_lookup=_IDH1_FILTER_LABEL_LOOKUP,
                )
                and _passes_optional_filter(
                    row.get("mgmt_status"),
                    mgmt_status,
                    label_lookup=_MGMT_FILTER_LABEL_LOOKUP,
                )
            ),
            axis=1,
        )
        n_before_filter = len(merged)
        merged = merged[mask]
        if len(merged) < n_before_filter:
            warnings.append(
                f"Filtre uygulandı ({filters_applied}) -- eksik değerli "
                "komşular ELENMEDİ (zarifçe bozulma), yalnızca değeri olup "
                f"filtreyle uyuşmayanlar çıkarıldı. Havuz {n_before_filter} "
                f"-> {len(merged)} komşuya düştü."
            )

    top_k = merged.iloc[:k]
    # k>n_available uyarısı zaten yukarıda verildi -- filtre YÜZÜNDEN
    # (n_available'dan daha fazla) kısaldıysa AYRI/ek bir uyarı ver (aynı
    # mesajın tekrarı değil).
    if filters_applied and len(top_k) < min(k, n_available):
        warnings.append(
            f"Filtre/indeks sınırları sonrası yalnızca {len(top_k)} komşu "
            f"döndürülebildi (istenen k={k})."
        )

    results = []
    for pid, row in top_k.iterrows():
        entry: dict[str, Any] = {
            "patient_id": pid,
            "l2_distance": float(row["l2_distance"]),
        }
        for col in result_columns:
            value = row.get(col)
            entry[col] = _num_or_none(value) if col in ("age", "kps_score", "survival_days") else _val_or_none(value)
        results.append(entry)

    return {
        "index_size": int(bundle.index.ntotal),
        "k_requested": k,
        "k_returned": len(results),
        "filters_applied": filters_applied,
        "warnings": warnings,
        "results": results,
    }


_RESULT_METADATA_COLUMNS: tuple[str, ...] = (
    "source",
    "age",
    "gender",
    "kps_score",
    "mgmt_status",
    "idh1_status",
    "gtr_over90percent",
    "vital_status",
    "survival_days",
)


def _build_omics_similar_block(
    patient_id: str, k: int, *, idh1_status: str | None, mgmt_status: str | None
) -> dict[str, Any]:
    """`molecular_omics_faiss` katmanı -- yalnız `has_omics=TRUE` ise
    çağrılır. Omics indeksi eksik/bozuksa veya hasta orada yoksa (veri
    tutarsızlığı) SESSİZCE atlamaz -- `available: False` + görünür `note`
    döner, ana radyomik sonucu ETKİLEMEZ (bkz. modül dokstring'i)."""

    try:
        bundle = _load_index_bundle(
            _molecular_omics_index_dir(), MOLECULAR_OMICS_INDEX_FILENAME
        )
    except SimilarIndexArtifactsError as exc:
        return {
            "has_omics": True,
            "available": False,
            "note": f"molecular_omics_faiss indeks dosyaları eksik/bozuk: {exc}",
        }

    try:
        block = _build_similar_block(
            bundle,
            patient_id,
            k,
            idh1_status=idh1_status,
            mgmt_status=mgmt_status,
            result_columns=_RESULT_METADATA_COLUMNS,
        )
    except PatientNotInIndexError:
        return {
            "has_omics": True,
            "available": False,
            "note": (
                f"{patient_id!r}: patients.has_omics=TRUE ama molecular_"
                "omics_faiss indeksinde satırı YOK -- veri tutarsızlığı, "
                "db-agent'e bildirin. Radyomik benzer-hasta sonucu bu "
                "durumdan ETKİLENMEDİ."
            ),
        }

    block["has_omics"] = True
    block["available"] = True
    return block


# =====================================================================
# 5) Endpoint
# =====================================================================


@router.get("/patient/{patient_id}/similar")
def get_similar_patients(
    patient_id: str,
    k: int = Query(DEFAULT_K, ge=1, description="En yakın kaç hasta dönsün (varsayılan 10)."),
    idh1_status: str | None = Query(
        None,
        description=(
            "İsteğe bağlı filtre -- 'Wildtype' | 'Mutated' | 'NOS/NEC'. "
            "Kaynak sözlükleri KANONİKLEŞTİRİLİR (karar 34): 'Wildtype' "
            "sorgusu LUMIERE'in 'WT'/'wt' hastalarını da YAKALAR, 'WT' "
            "sorgusu UPenn'in 'Wildtype' hastalarını da yakalar (simetrik). "
            "'IDH1 neg, Sequencing required' -> 'NOS/NEC' sayılır, "
            "'Wildtype' DEĞİL. Mesafeye GİRMEZ, yalnız sonuç kümesini "
            "daraltır. Eksik (NULL) değerli komşular filtre nedeniyle "
            "ELENMEZ."
        ),
    ),
    mgmt_status: str | None = Query(
        None,
        description=(
            "İsteğe bağlı filtre -- 'Methylated' | 'Unmethylated'. Kaynak "
            "sözlükleri KANONİKLEŞTİRİLİR (karar 34): LUMIERE'in "
            "'methylated'/'not methylated' etiketleri UPenn'in "
            "'Methylated'/'Unmethylated' ile EŞ sayılır. Mesafeye GİRMEZ. "
            "Eksik değerli komşular ELENMEZ."
        ),
    ),
):
    # 1) Radyomik indeksi yükle -- infra/artefakt hazırlığı ÖNCE kontrol
    #    edilir (api/predict.py::predict_patient()'in checkpoint-önce
    #    deseniyle tutarlı).
    try:
        bundle = _load_index_bundle(
            _clinical_radiomics_index_dir(), CLINICAL_RADIOMICS_INDEX_FILENAME
        )
    except SimilarIndexArtifactsError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # 2) Hasta `patients` tablosunda var mı (404)
    try:
        status = _fetch_patient_existence_and_omics(patient_id)
    except Exception as exc:  # pragma: no cover -- yalnız DB erişilemezse
        raise HTTPException(status_code=503, detail=f"DB'ye erişilemedi: {exc}")

    if not status["exists"]:
        raise HTTPException(status_code=404, detail=f"Hasta bulunamadı: {patient_id}")

    # 3) Hasta v1 clinical_radiomics_faiss indeksinde var mı (422)
    try:
        clinical_block = _build_similar_block(
            bundle,
            patient_id,
            k,
            idh1_status=idh1_status,
            mgmt_status=mgmt_status,
            result_columns=_RESULT_METADATA_COLUMNS,
        )
    except PatientNotInIndexError:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{patient_id!r} clinical_radiomics_faiss v1 indeksinde YOK "
                "(K1 kararı: yalnız UPenn 611 + LUMIERE 72 + TCGA 39 = 722 "
                "hasta indekste; UCSF DAHİL DEĞİL). Açık durum: bu hasta "
                "için benzer-hasta indeksi yok -- sessizce boş sonuç "
                "DÖNÜLMEDİ."
            ),
        )

    response: dict[str, Any] = {
        "patient_id": patient_id,
        "clinical_radiomics_faiss": clinical_block,
    }

    # 4) Omics katmanı -- OPSİYONEL, has_omics != TRUE ise yanıttan
    #    TAMAMEN ÇIKARILIR (plan.txt:503 "sessizce atla" deseni).
    if status["has_omics"]:
        response["molecular_omics_faiss"] = _build_omics_similar_block(
            patient_id, k, idh1_status=idh1_status, mgmt_status=mgmt_status
        )

    return response
