"""FAISS `clinical_radiomics_faiss` / `molecular_omics_faiss` indeks İNŞA script'i.

🔒 K1/K2 KİLİTLENDİ (2026-08-18, Barış kararı -- bu dosyanın SON revizyonu):

**K1 -- Tasarım A (ham C32, ComBat'sız) SEÇİLDİ.** Gerekçe yalnız K1
taslağının (`decisions/2026-08-18-k1-faiss-secim-kurali-TASLAK.md`) §7'sindeki
üç mühendislik maddesi (yeni-hasta sorgusu asimetrisi yok, CLAUDE.md'nin
"ComBat...TCGA HARİÇ" dar/geniş okuma belirsizliğine hiç girmiyor, şema/onay
bağımlılığı yok) -- havuzlanmış "1017 vs 683" gerekçesi reviewer tarafından
GERİ ÇEKİLDİ (K1 taslağı §3.2'nin "havuzlanmış sayı doğrudan karşılaştırılamaz"
uyarısını ihlal ediyordu). `--combat` (Tasarım B) kodu SİLİNMEDİ -- olası bir
duyarlılık kolu için korunuyor, ama v1 ÜRETİM indeksi YALNIZ `--no-combat`
ile kurulur.

**K2 -- LUMIERE kanonik vizit = `Rating == 'Pre-Op'` (72 hasta) SEÇİLDİ.**
Kaynak: `raw/veri/LUMIERE-ExpertRating-v202211 database girecek.csv`, `Date`
kolonu `mr_scans.timepoint_label` ile birebir eşleşiyor (canlı doğrulandı).
Ölçüm: 91/91 hastada Pre-Op etiketi var (Patient-060'ta 2 satır, week-000 +
week-069), bunlardan yalnız 73 vizit/**72 hasta**da o vizitte C32 radyomiği
de var -- 19 hasta Pre-Op vizitinde C32'siz (fallback REDDEDİLDİ, Barış
kararı: post-op görüntü yapısal olarak farklı [rezeksiyon boşluğu, değişen
kontrast tutulumu], aynı indekste karıştırmak mesafeyi anlamsızlaştırır).
Provizyonel `earliest_scan_date`/`--acknowledge-k2-open` KALDIRILDI --
`fetch_lumiere_preop_canonical_visits()` bu KİLİTLİ kuralı UYGULAR, opt-in
gerekmez. `Rating` kolonunda kirli değerler var (`'Post-Op '` sondan
boşluklu, `'Post-Op/PD'`) -- guard `.strip()` + tam eşleşme `== 'Pre-Op'`
kullanır, `startswith`/`in` KULLANILMAZ (reviewer uyarısı).

**v1 kapsamı -- UCSF DAHİL DEĞİL.** K1 taslağının sorgu seti/sınıflandırıcı/
eşikleri UCSF hiç yokken (3 sınıf: TCGA/UPenn/LUMIERE) donduruldu -- UCSF
eklemek taslağın 4-sınıflı revizyonunu (AYRI Barış onayı) gerektirir.
`UCSF-PDGM-PyRadiomics-107-C32` (bugün DB'ye yazıldı, 1475 satır/295 tarama)
C32 BEYAZ LİSTESİNE eklendi (gate hazır) ama `build_design_a_ham_c32()`/
`build_design_b_combat()` bu adı ASLA sorgulamaz -- yalnız gate seviyesinde
tanınır, v1 vektör havuzuna girmez.

**v1 = UPenn 611 + LUMIERE 72 + TCGA 39 = 722 hasta** (mimarinin "~771,
TCGA dahil" tarifine en yakın rakam, `--no-combat` ile).

🔴 C32 VERİ KAPISI (CLAUDE.md "ADIM 0", atlanamaz): bu script'in TEK meşru
girdisi `FAISS_ALLOWED_C32_SEGMENTATION_TOOLS` -- kapalı (`IN (...)`)
beyaz liste, `LIKE` YOK (K1 taslağı §1 madde 3'ün "-C32-v2 gibi bir ad
sessizce kaçar" uyarısıyla aynı gerekçe). Eski nesil adları
(`UPenn-PyRadiomics-107`, `LUMIERE-PyRadiomics-107`, `TCGA-ground-truth`,
`CaPTk-*`, `DeepBraTumIA`, `HD-GLIO-AUTO`) bu fonksiyonlara ASLA verilmez
-- `DeepBraTumIA`/`HD-GLIO-AUTO` da tam 107 özellik taşıdığı için özellik
SAYISI bunları AYIRT ETMEZ, yalnız `segmentation_tool` ADI ayırt eder.

🔒 VEKTÖR İÇERİĞİ KİLİTLİ (`decisions/2026-08-14-faiss-v1-vektor-icerigi.md`):
yalnız `WT` bölgesi × `STABLE_FEATURES_ICC60` (93 özellik, `pipeline.cox_
model`'den AYNEN yeniden kullanılır) + tek dondurulmuş `StandardScaler`.
Yaş/KPS/MGMT/IDH1/tedavi mesafeye GİRMEZ -- yalnız metadata/filtre için ayrı
bir CSV'ye yazılır (bkz. `fetch_patients_metadata_frame()`).

DB ERİŞİMİ: yalnız `readonly=True` bağlantı, yalnız SELECT. Bu script DB'ye
HİÇBİR YAZMA yapmaz (CLAUDE.md KESİN SINIRLAR #4 gereği zaten hiçbir şema
değişikliği/riskli DB işlemi bu dosyada YOK). `raw/veri/LUMIERE-ExpertRating-
v202211 database girecek.csv` yalnız OKUNUR, `raw/` DEĞİŞTİRİLMEZ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import numpy as np
import pandas as pd
from psycopg2.extras import RealDictCursor
from sklearn.preprocessing import StandardScaler

from db_connection import get_connection
from pipeline.cox_model import (
    PRIMARY_REGIONS,
    STABLE_FEATURES_ICC60,
    RegionPivotError,
    pivot_radiomics_long_to_wide,
)
from pipeline.harmonization import (
    COMBAT_FIT_ALLOWED_SOURCES,
    ComBatFitLeakageError,
    canonical_source,
    fit_combat_harmonization,
)

# =====================================================================
# K2 KİLİTLİ kural -- TEK KAYNAK `pipeline/lumiere_canonical_visit.py`
# (2026-09-13, Y1 görevi). Aşağıdaki isimler BU DOSYADA YENİDEN
# TANIMLANMAZ, yalnız YENİDEN İHRAÇ edilir -- `api/predict.py` de AYNI
# modülü kullanır, kural İKİNCİ BİR YERE YAZILMADI. Mantık DEĞİŞMEDİ:
# gövde buradan o modüle TAŞINDI (bkz. o modülün docstring'i).
# `_load_lumiere_preop_visit_keys`/`_load_...sort_key` alt-çizgili adları
# mevcut testler (tests/test_faiss_index.py) bu adlarla çağırdığı için
# KORUNUYOR.
# =====================================================================
from pipeline.lumiere_canonical_visit import (  # noqa: E402
    DEFAULT_LUMIERE_EXPERT_RATING_CSV,
    EXPECTED_LUMIERE_PREOP_PATIENTS,
    ExpertRatingFileNotFoundError,
    LumierePreopCohortMismatchError,
    LumierePreopSelectionReport,
)
from pipeline.lumiere_canonical_visit import (  # noqa: E402
    load_lumiere_preop_visit_keys as _load_lumiere_preop_visit_keys,
)
from pipeline.lumiere_canonical_visit import (  # noqa: E402
    lumiere_timepoint_sort_key as _lumiere_timepoint_sort_key,
)
from pipeline.lumiere_canonical_visit import (  # noqa: E402
    select_lumiere_preop_canonical_visits as _select_lumiere_preop_canonical_visits,
)

# =====================================================================
# C32 veri kapısı -- kapalı beyaz liste (CLAUDE.md ADIM 0)
# =====================================================================

SEGMENTATION_TOOL_UPENN_C32 = "UPenn-PyRadiomics-107-C32"
SEGMENTATION_TOOL_LUMIERE_C32 = "LUMIERE-PyRadiomics-107-C32"
SEGMENTATION_TOOL_TCGA_C32 = "TCGA-ground-truth-C32"
# 2026-08-18 EKLENDİ (Barış kararı) -- UCSF'in C32 radyomiği bugün DB'ye
# yazıldı (canlı SELECT ile doğrulandı: 1475 satır / 295 tarama). Yalnız
# GATE'e (beyaz listeye) eklenir -- v1 FAISS vektör havuzuna GİRMEZ (bkz.
# modül docstring'i "v1 kapsamı -- UCSF DAHİL DEĞİL"). `build_design_a_
# ham_c32()`/`build_design_b_combat()` bu sabiti ASLA fetch çağrısında
# kullanmaz -- yalnız `assert_c32_segmentation_tool_allowed()` bunu tanır.
SEGMENTATION_TOOL_UCSF_C32 = "UCSF-PDGM-PyRadiomics-107-C32"

FAISS_ALLOWED_C32_SEGMENTATION_TOOLS: frozenset[str] = frozenset(
    {
        SEGMENTATION_TOOL_UPENN_C32,
        SEGMENTATION_TOOL_LUMIERE_C32,
        SEGMENTATION_TOOL_TCGA_C32,
        SEGMENTATION_TOOL_UCSF_C32,
    }
)
# v1 vektör havuzuna girecek kaynaklar -- UCSF BİLİNÇLİ OLARAK YOK (K1
# taslağının sorgu seti/sınıflandırıcı/eşikleri 3 sınıfla [TCGA/UPenn/
# LUMIERE] donduruldu, UCSF eklemek AYRI bir Barış onayı/taslak revizyonu
# gerektirir -- bkz. modül docstring'i).
FAISS_V1_INDEX_SEGMENTATION_TOOLS: frozenset[str] = frozenset(
    {
        SEGMENTATION_TOOL_UPENN_C32,
        SEGMENTATION_TOOL_LUMIERE_C32,
        SEGMENTATION_TOOL_TCGA_C32,
    }
)

UPENN_SOURCE_NAME = "UPenn-GBM"
LUMIERE_SOURCE_NAME = "LUMIERE"
TCGA_SOURCE_NAME = "TCGA-GBM"
UCSF_SOURCE_NAME = "UCSF-PDGM"

DEFAULT_SEED = 42


class C32WhitelistViolationError(RuntimeError):
    """`segmentation_tool` FAISS'in C32 beyaz listesinde DEĞİL.

    CLAUDE.md "ADIM 0 -- C32 VERİ KAPISI": eski nesil veriye (binWidth=25,
    N4/Z-score atlanmış) veya kaynak-sağlayıcı precompute'a (CaPTk/
    DeepBraTumIA/HD-GLIO-AUTO -- şekil/özellik sayısıyla AYIRT EDİLEMEZ)
    SESSİZCE düşülmesin diye burada SERT durulur.
    """


def assert_c32_segmentation_tool_allowed(segmentation_tool: str) -> None:
    if segmentation_tool not in FAISS_ALLOWED_C32_SEGMENTATION_TOOLS:
        raise C32WhitelistViolationError(
            f"segmentation_tool={segmentation_tool!r} FAISS C32 beyaz "
            f"listesinde DEĞİL. İZİN VERİLEN (yalnız bu üçü): "
            f"{sorted(FAISS_ALLOWED_C32_SEGMENTATION_TOOLS)}. Eski nesil "
            "adları (ör. 'UPenn-PyRadiomics-107' [-C32 soneki YOK], "
            "'CaPTk-automatic', 'DeepBraTumIA', 'HD-GLIO-AUTO') bu "
            "fonksiyona ASLA verilmemeli -- CLAUDE.md 'ADIM 0' kapısı."
        )


class C32DataNotFoundError(RuntimeError):
    """`radiomics` tablosunda beklenen `segmentation_tool` icin 0 satir."""


def raise_if_empty_c32(long_frame: pd.DataFrame, *, segmentation_tool: str) -> None:
    if long_frame.empty:
        raise C32DataNotFoundError(
            f"radiomics tablosunda segmentation_tool={segmentation_tool!r} "
            "icin 0 satir. C32 cikarimi/yuklemesi henuz DB'ye YAZILMAMIS "
            "olabilir (CLAUDE.md: 'C32 ile yeniden cikarim tamamlanmadan bu "
            "adimlara gecen olursa DURDUR'). Eski A-yontemi veriye SESSIZCE "
            "DUSULMEDI -- bu kasitli, kod seviyesinde zorlanan bir durma "
            "noktasidir."
        )


# NOT (2026-09-13, Y1): `LumierePreopCohortMismatchError` ve
# `ExpertRatingFileNotFoundError` ARTIK BURADA TANIMLANMIYOR -- ikisi de
# `pipeline/lumiere_canonical_visit.py`'den YENİDEN İHRAÇ EDİLİYOR (bkz.
# yukarıdaki import bloğu). Burada yeniden tanımlanmaları, import edilen
# sınıfları GÖLGELEYİP `api/predict.py` ile AYRI istisna kimlikleri
# doğurur ve `except` blokları sessizce eşleşmez hâle gelirdi.


class UnknownRegionColumnError(RuntimeError):
    """Beklenen `WT__<STABLE_FEATURES_ICC60>` kolonlarından biri pivot
    çıktısında yok -- STABLE_FEATURES_ICC60/pivot sözleşmesi bozulmuş
    olabilir, sessizce eksik özellik ile devam EDİLMEZ."""


class EmptyIndexPoolError(RuntimeError):
    """Seçilen tasarım/parametrelerle indekse girecek 0 hasta kaldı.

    CLAUDE.md: 'FAISS sonucu boşsa sessizce boş dönme, açık bir benzer
    hasta bulunamadı durumu döndür' ilkesinin İNŞA-ZAMANI karşılığı --
    boş bir indeks dosyası SESSİZCE ÜRETİLMEZ.
    """


# =====================================================================
# 1) DB'den okuma -- yalniz SELECT
# =====================================================================


def fetch_c32_radiomics_long_frame(cursor, *, segmentation_tool: str) -> pd.DataFrame:
    """`radiomics` x `mr_scans` x `patients` x `dataset_sources` -- UZUN format.

    `train_cox_week3.py::fetch_c32_radiomics_long_frame()` ile AYNI desen
    (sayfalı çekim -- UPenn'in büyük JSONB hacminde gerçek koşuda
    `QueryCanceled`/`SSL connection closed` görüldüğü için, bkz. o dosyanın
    2026-08-14 notu) -- ama BURADA `scan_id`/`scan_date`/`timepoint_label`
    DÜŞÜRÜLMEZ (Cox'un fonksiyonundan farkı budur): LUMIERE'in hasta-başı
    kanonik vizit seçimi (K2) bu kolonlara ihtiyaç duyar.
    """

    assert_c32_segmentation_tool_allowed(segmentation_tool)

    page_size = 250
    offset = 0
    rows: list = []
    real_cursor = cursor.connection.cursor(cursor_factory=RealDictCursor)
    try:
        while True:
            real_cursor.execute(
                """
                SELECT r.scan_id, ms.scan_date, ms.timepoint_label,
                       p.patient_id, ds.source_name AS source, r.tumor_region,
                       r.shape_features, r.first_order_features, r.texture_features
                FROM radiomics r
                JOIN mr_scans ms ON ms.scan_id = r.scan_id
                JOIN patients p ON p.patient_id = ms.patient_id
                JOIN dataset_sources ds ON ds.source_id = p.source_id
                WHERE r.segmentation_tool = %s
                ORDER BY r.scan_id, r.tumor_region
                LIMIT %s OFFSET %s
                """,
                (segmentation_tool, page_size, offset),
            )
            page = real_cursor.fetchall()
            if not page:
                break
            rows.extend(page)
            offset += page_size
    finally:
        real_cursor.close()

    frame = pd.DataFrame(rows)
    return frame


# =====================================================================
# K2 KİLİTLİ -- LUMIERE kanonik vizit = Rating == 'Pre-Op' (72 hasta)
# =====================================================================
#
# Kaynak (SADECE OKUNUR, raw/ immutable): `LUMIERE-ExpertRating-v202211
# database girecek.csv`. `Patient` kolonu `patients.patient_id` ile,
# `Date` kolonu `mr_scans.timepoint_label` ile BİREBİR eşleşiyor (canlı
# DB'ye karşı doğrulandı -- 2026-08-18).
#
# ÖLÇÜM (canlı, bu dosyayı yazan görevde tekrar doğrulandı):
#   Pre-Op etiketli satır          : 92 (91 hasta, Patient-060'ta 2 satır
#                                     -- week-000 VE week-069, ikisinde de
#                                     C32 radyomiği VAR, tie-break gerekir)
#   Pre-Op VE C32'si olan (eşleşen): 73 vizit / 72 hasta  <- KANONİK
#   Pre-Op'ta C32'si OLMAYAN       : 19 hasta (fallback REDDEDİLDİ, bkz.
#                                     modül docstring'i)
# 2026-09-13 (Y1 görevi) -- KURALIN GÖVDESİ BURADAN
# `pipeline/lumiere_canonical_visit.py`'ye TAŞINDI (TEK KAYNAK). Bu dosya
# artık yalnız DELEGE eder; `api/predict.py` de AYNI modülü kullanır.
# Sabitler/yardımcılar yukarıdaki import bloğunda YENİDEN İHRAÇ edilmiştir
# (`DEFAULT_LUMIERE_EXPERT_RATING_CSV`, `EXPECTED_LUMIERE_PREOP_PATIENTS`,
# `_lumiere_timepoint_sort_key`, `_load_lumiere_preop_visit_keys`,
# `LumierePreopSelectionReport`) -- MANTIK DEĞİŞMEDİ, tek fark
# beklenen-hasta-sayısının parametre olarak geçirilmesi (aşağıda), ki bu
# dosyadaki `EXPECTED_LUMIERE_PREOP_PATIENTS` monkeypatch'lenebilir kalsın.


def fetch_lumiere_preop_canonical_visits(
    long_frame: pd.DataFrame, *, csv_path: Path = DEFAULT_LUMIERE_EXPERT_RATING_CSV
) -> tuple[pd.DataFrame, LumierePreopSelectionReport]:
    """K2 KİLİTLİ kural -- hasta başına TEK `scan_id` seçer (Rating=='Pre-Op'
    VE C32 radyomiği olan vizit).

    Gövde `pipeline.lumiere_canonical_visit.select_lumiere_preop_canonical_
    visits()`'tedir (TEK KAYNAK, 2026-09-13/Y1) -- burada YENİDEN
    YAZILMAZ/KOPYALANMAZ. Beklenen hasta sayısı bu modülün
    `EXPECTED_LUMIERE_PREOP_PATIENTS` sabitinden ÇAĞRI ANINDA okunur.
    """

    return _select_lumiere_preop_canonical_visits(
        long_frame,
        csv_path=csv_path,
        expected_patients=EXPECTED_LUMIERE_PREOP_PATIENTS,
    )


PATIENTS_METADATA_RAW_COLUMNS: tuple[str, ...] = (
    "age",
    "gender",
    "kps_score",
    "mgmt_status",
    "idh1_status",
    # 2026-08-18 EKLENDİ (TCGA-02-0003/0006 benzer-hasta doğrulaması için):
    # `vital_status`/`survival_days` de klinik SONUÇ (mesafeye girmeyen,
    # yalnız görüntüleme/tutarlılık amaçlı) -- bkz. `fetch_lumiere_
    # preop_canonical_visits()`'in aksine, bunlar hiçbir zaman FİLTRE
    # olarak da kullanılmaz, yalnız "klinik seyirle mantıksal tutarlılık"
    # gözle doğrulaması için görüntülenir (`tools/faiss_similar_patients_
    # query_demo.py`).
    "vital_status",
    "survival_days",
)


def fetch_patients_metadata_frame(cursor, *, source_name: str) -> pd.DataFrame:
    """`patients` x `dataset_sources` -- yaş/cinsiyet/KPS/MGMT/IDH1/sağkalım (HAM).

    🔒 Bu kolonlar `decisions/2026-08-14-faiss-v1-vektor-icerigi.md`
    kararı gereği vektöre/mesafeye HİÇ GİRMEZ -- yalnız isteğe bağlı
    filtre/rerank/görüntüleme için ayrı bir metadata CSV'sine yazılır
    (bkz. `write_index_artifacts()`).

    AÇIK İŞ (bu fonksiyonun kapsamı DIŞI, çözülmedi): "tedavi" alanının
    tek-kategorik türetilmesi (`treatments` bire-çok tablo) --
    ~~`concepts/faiss-vektor-spesifikasyonu-taslak.md`~~ §1'in kendi
    notuna göre bu artık ENGELLEYİCİ değil (mesafeye girmediği için),
    filtre özelliği gerçekten eklenmek istenirse AYRI ele alınmalı.
    📍 YOL DÜZELTMESİ (2026-09-14, kalem 5 -- diskten doğrulandı): o
    taslak `archive/faiss-vektor-spesifikasyonu-taslak.md`'ye TAŞINDI
    (wiki hard rule #4; 2026-09-13, `belge-agent-I`). `concepts/`
    altında ARTIK YOK -- atıf kırıktı. Arşiv otorite DEĞİLDİR; bu
    konudaki güncel karar
    `decisions/2026-09-13-k7-faiss-v1-vektor-icerigi-onaylandi.md`'dir.
    """

    real_cursor = cursor.connection.cursor(cursor_factory=RealDictCursor)
    try:
        columns_sql = ", ".join(f"p.{col}" for col in PATIENTS_METADATA_RAW_COLUMNS)
        real_cursor.execute(
            f"""
            SELECT p.patient_id, ds.source_name AS source, {columns_sql}
            FROM patients p
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE ds.source_name = %s
            """,
            (source_name,),
        )
        rows = real_cursor.fetchall()
    finally:
        real_cursor.close()
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["patient_id", "source", *PATIENTS_METADATA_RAW_COLUMNS])
    return frame.set_index("patient_id")


# =====================================================================
# 2) WT x 93 pivot -- kilitli vektör içeriği
# =====================================================================

WT_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"WT__{feature}" for feature in STABLE_FEATURES_ICC60
)


def _prepare_long_frame_for_pivot(long_frame: pd.DataFrame) -> pd.DataFrame:
    """`pivot_radiomics_long_to_wide()`'in beklediği kolonlara indirger."""

    required = ["patient_id", "source", "tumor_region", "shape_features",
                "first_order_features", "texture_features"]
    return long_frame[required].copy()


def pivot_wt93(long_frame: pd.DataFrame, *, on_missing_region: str = "drop"):
    prepared = _prepare_long_frame_for_pivot(long_frame)
    wide_frame, report = pivot_radiomics_long_to_wide(
        prepared, regions=list(PRIMARY_REGIONS), on_missing_region=on_missing_region
    )
    missing_cols = [c for c in WT_FEATURE_COLUMNS if c not in wide_frame.columns]
    if missing_cols and not wide_frame.empty:
        raise UnknownRegionColumnError(
            f"Pivot çıktısında {len(missing_cols)} beklenen WT__ özellik "
            f"kolonu yok (ör. {missing_cols[:3]}) -- STABLE_FEATURES_ICC60 "
            "sözleşmesi bozulmuş olabilir. Sessizce eksik özellikle devam "
            "EDİLMEDİ."
        )
    return wide_frame, report


# =====================================================================
# 3) Dondurulmuş scaler
# =====================================================================


@dataclass
class FrozenScaler:
    mean_: np.ndarray
    scale_: np.ndarray
    feature_order: list[str]
    fit_pool_sources: dict[str, int]
    fit_date: str
    seed: int = DEFAULT_SEED

    def transform(self, matrix: pd.DataFrame) -> np.ndarray:
        ordered = matrix[self.feature_order].to_numpy(dtype=np.float64)
        return ((ordered - self.mean_) / self.scale_).astype(np.float32)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "mean_": self.mean_.tolist(),
            "scale_": self.scale_.tolist(),
            "feature_order": self.feature_order,
            "fit_pool_sources": self.fit_pool_sources,
            "fit_date": self.fit_date,
            "seed": self.seed,
        }


def fit_frozen_scaler(
    matrix: pd.DataFrame, *, source_series: pd.Series
) -> FrozenScaler:
    """StandardScaler'ı `matrix` üzerinde fit eder, DONDURULMUŞ döndürür.

    NOT (görev talimatı, açık iş -- bkz. ~~`concepts/faiss-vektor-
    spesifikasyonu-taslak.md`~~ -> **`archive/faiss-vektor-
    spesifikasyonu-taslak.md`** [2026-09-14 kalem 5 yol düzeltmesi:
    dosya `archive/`'a taşındı, `concepts/` altında YOK; arşiv otorite
    DEĞİL, güncel karar `decisions/2026-09-13-k7-faiss-v1-vektor-
    icerigi-onaylandi.md`] "Barış'ın karar vermesi gereken noktalar"
    madde 2): scaler'ın fit edileceği TAM kohort (yalnız UPenn mi,
    UPenn+LUMIERE mi, yoksa TÜM indeks havuzu mu) henüz kilitlenmedi --
    bu fonksiyon çağıranın verdiği `matrix`'i OLDUĞU GİBİ fit eder,
    kohort seçimi `main()`'in `--scaler-fit-pool` parametresinde.
    """

    if matrix.empty:
        raise EmptyIndexPoolError("fit_frozen_scaler(): fit matrisi boş.")
    scaler = StandardScaler()
    scaler.fit(matrix[list(matrix.columns)].to_numpy(dtype=np.float64))
    return FrozenScaler(
        mean_=scaler.mean_,
        scale_=scaler.scale_,
        feature_order=list(matrix.columns),
        fit_pool_sources=source_series.value_counts().to_dict(),
        fit_date=datetime.now(timezone.utc).isoformat(),
    )


# =====================================================================
# 4) Tasarım A / B -- ComBat'sız / ComBat'lı vektör montajı
# =====================================================================


@dataclass
class FaissBuildReport:
    design: str
    combat: bool
    n_vectors: int
    dim: int
    source_counts: dict[str, int]
    lumiere_visit_rule: str | None
    lumiere_visit_report: dict[str, Any] | None
    pivot_reports: dict[str, dict[str, Any]]
    scaler_fit_pool: str
    scaler_fit_pool_sources: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    excluded_sources: list[str] = field(default_factory=list)


LUMIERE_VISIT_RULE_NAME = "preop_expert_rating"


def _fetch_source_wt93(
    cursor,
    *,
    segmentation_tool: str,
) -> tuple[pd.DataFrame, dict[str, Any], LumierePreopSelectionReport | None]:
    """Tek bir kaynağın (UPenn/LUMIERE/TCGA) WT×93 geniş matrisini üretir.

    LUMIERE için K2 KİLİTLİ kural (`fetch_lumiere_preop_canonical_visits()`)
    HER ZAMAN uygulanır -- artık opt-in/provizyonel DEĞİL.
    """

    long_frame = fetch_c32_radiomics_long_frame(cursor, segmentation_tool=segmentation_tool)
    raise_if_empty_c32(long_frame, segmentation_tool=segmentation_tool)

    visit_report = None
    if segmentation_tool == SEGMENTATION_TOOL_LUMIERE_C32:
        long_frame, visit_report = fetch_lumiere_preop_canonical_visits(long_frame)

    wide_frame, pivot_report = pivot_wt93(long_frame)
    pivot_report_dict = {
        "regions_requested": pivot_report.regions_requested,
        "n_input_rows": pivot_report.n_input_rows,
        "n_candidate_patients": pivot_report.n_candidate_patients,
        "n_output_patients": pivot_report.n_output_patients,
        "dropped_patients_missing_region": pivot_report.dropped_patients_missing_region,
    }
    return wide_frame, pivot_report_dict, visit_report


def build_design_a_ham_c32(
    cursor, *, scaler_fit_pool: str, include_ucsf: bool = False
) -> tuple[np.ndarray, pd.DataFrame, FaissBuildReport, "FrozenScaler"]:
    """Tasarım A -- ham C32, ComBat'sız TEK ortak uzay.

    🔒 K1 KİLİTLİ SEÇİM (2026-08-18, Barış) -- v1 ÜRETİM indeksi bu
    fonksiyonla kurulur. LUMIERE her zaman K2'nin kilitli kuralıyla
    (`fetch_lumiere_preop_canonical_visits()`, 72 hasta) dahil edilir.

    `include_ucsf` (2026-09-17, Barış onayı) -- VARSAYILAN `False`, yani
    v1 davranışı DEĞİŞMEDİ. `True` iken UCSF-PDGM de havuza girer
    (ölçüldü: UPenn 611 + LUMIERE 72 + TCGA 39 + UCSF 295 = **1017**).
    Bu, K1 taslağının 3-sınıflı kapsamının BİLİNÇLİ bir revizyonudur ve
    ayrı bir artefakt dizinine yazılmalıdır -- v1 (722) raporlanan
    artefakt olarak YERİNDE KALIR.

    🔴 `include_ucsf=True` iken `scaler_fit_pool="index_pool"` KULLANILMAZ;
    çağıran taraf (CLI) bunu sert biçimde reddeder. Gerekçe
    `_select_scaler_fit_pool()` içindeki `upenn_lumiere_tcga` notunda.
    """

    warnings: list[str] = []
    excluded_sources: list[str] = []
    upenn_wide, upenn_pivot, _ = _fetch_source_wt93(
        cursor, segmentation_tool=SEGMENTATION_TOOL_UPENN_C32
    )
    lumiere_wide, lumiere_pivot, lumiere_visit_report = _fetch_source_wt93(
        cursor, segmentation_tool=SEGMENTATION_TOOL_LUMIERE_C32
    )
    tcga_wide, tcga_pivot, _ = _fetch_source_wt93(
        cursor, segmentation_tool=SEGMENTATION_TOOL_TCGA_C32
    )
    if include_ucsf:
        ucsf_wide, ucsf_pivot, _ = _fetch_source_wt93(
            cursor, segmentation_tool=SEGMENTATION_TOOL_UCSF_C32
        )
    else:
        ucsf_wide = pd.DataFrame()

    candidate_frames = (upenn_wide, lumiere_wide, tcga_wide, ucsf_wide)
    non_empty_frames = [f for f in candidate_frames if not f.empty]
    if not non_empty_frames:
        raise EmptyIndexPoolError(
            "Tasarım A: birleşik havuz boş -- hiçbir kaynaktan hasta indekse girmedi."
        )
    combined = pd.concat(non_empty_frames, axis=0, sort=False)
    if combined.empty:
        raise EmptyIndexPoolError(
            "Tasarım A: birleşik havuz boş -- hiçbir kaynaktan hasta indekse girmedi."
        )
    combined = combined.sort_index()

    source_map = {}
    for pid in upenn_wide.index:
        source_map[pid] = UPENN_SOURCE_NAME
    for pid in lumiere_wide.index:
        source_map[pid] = LUMIERE_SOURCE_NAME
    for pid in tcga_wide.index:
        source_map[pid] = TCGA_SOURCE_NAME
    for pid in ucsf_wide.index:
        source_map[pid] = UCSF_SOURCE_NAME
    source_series = pd.Series(source_map).reindex(combined.index)

    combined_wt93 = combined[list(WT_FEATURE_COLUMNS)]
    fit_pool = _select_scaler_fit_pool(
        combined_wt93, source_series, scaler_fit_pool=scaler_fit_pool
    )
    scaler = fit_frozen_scaler(fit_pool, source_series=source_series.loc[fit_pool.index])
    vectors = scaler.transform(combined_wt93)

    report = FaissBuildReport(
        design="A_ham_c32_no_combat",
        combat=False,
        n_vectors=len(combined),
        dim=vectors.shape[1],
        source_counts=source_series.value_counts().to_dict(),
        lumiere_visit_rule=LUMIERE_VISIT_RULE_NAME,
        lumiere_visit_report=(
            lumiere_visit_report.__dict__ if lumiere_visit_report is not None else None
        ),
        pivot_reports={"UPenn": upenn_pivot, "LUMIERE": lumiere_pivot, "TCGA": tcga_pivot},
        scaler_fit_pool=scaler_fit_pool,
        scaler_fit_pool_sources=scaler.fit_pool_sources,
        warnings=warnings,
        excluded_sources=excluded_sources,
    )

    metadata = pd.DataFrame({"source": source_series})
    return vectors, metadata, report, scaler


def build_design_b_combat(
    cursor, *, scaler_fit_pool: str
) -> tuple[np.ndarray, pd.DataFrame, FaissBuildReport, FrozenScaler]:
    """Tasarım B -- UPenn-referanslı ComBat homojen indeks (B1: TCGA HARİÇ).

    TCGA'nın dışlanması bu fonksiyonun kendi tercihi DEĞİL --
    `pipeline.harmonization.fit_combat_harmonization()`'ın (bu görevde
    DEĞİŞTİRİLMEYEN, önceden var olan) `COMBAT_FIT_ALLOWED_SOURCES =
    {"UPenn","LUMIERE"}` guard'ının doğal sonucu (`ComBatFitLeakageError`
    fırlatır). B2 (TCGA'yı üçüncü batch olarak dahil eden ayrı fit) bu
    fonksiyonun kapsamı DIŞINDA -- bkz. modül docstring'i.
    """

    warnings: list[str] = []
    excluded_sources = [TCGA_SOURCE_NAME]
    upenn_wide, upenn_pivot, _ = _fetch_source_wt93(
        cursor, segmentation_tool=SEGMENTATION_TOOL_UPENN_C32
    )
    lumiere_wide, lumiere_pivot, lumiere_visit_report = _fetch_source_wt93(
        cursor, segmentation_tool=SEGMENTATION_TOOL_LUMIERE_C32
    )

    combined_raw = pd.concat([upenn_wide, lumiere_wide], axis=0, sort=False)
    if combined_raw.empty:
        raise EmptyIndexPoolError(
            "Tasarım B: birleşik UPenn+LUMIERE havuzu boş."
        )
    combined_raw = combined_raw.sort_index()

    source_map: dict[str, str] = {pid: UPENN_SOURCE_NAME for pid in upenn_wide.index}
    source_map.update({pid: LUMIERE_SOURCE_NAME for pid in lumiere_wide.index})
    source_series = pd.Series(source_map).reindex(combined_raw.index)

    covariates = pd.DataFrame(
        {"SITE": source_series.map(canonical_source)}, index=combined_raw.index
    )

    try:
        _model, harmonized_in_sample, _parameter_rows = fit_combat_harmonization(
            combined_raw[list(WT_FEATURE_COLUMNS)],
            covariates,
            reference_batch="UPenn",
            seed=DEFAULT_SEED,
        )
    except ComBatFitLeakageError:
        raise

    fit_pool = _select_scaler_fit_pool(
        harmonized_in_sample, source_series, scaler_fit_pool=scaler_fit_pool
    )
    scaler = fit_frozen_scaler(fit_pool, source_series=source_series.loc[fit_pool.index])
    vectors = scaler.transform(harmonized_in_sample)

    report = FaissBuildReport(
        design="B_upenn_referans_combat_b1_tcga_haric",
        combat=True,
        n_vectors=len(combined_raw),
        dim=vectors.shape[1],
        source_counts=source_series.value_counts().to_dict(),
        lumiere_visit_rule=LUMIERE_VISIT_RULE_NAME,
        lumiere_visit_report=(
            lumiere_visit_report.__dict__ if lumiere_visit_report is not None else None
        ),
        pivot_reports={"UPenn": upenn_pivot, "LUMIERE": lumiere_pivot},
        scaler_fit_pool=scaler_fit_pool,
        scaler_fit_pool_sources=scaler.fit_pool_sources,
        warnings=warnings,
        excluded_sources=excluded_sources,
    )

    metadata = pd.DataFrame({"source": source_series})
    return vectors, metadata, report, scaler


def _select_scaler_fit_pool(
    matrix: pd.DataFrame, source_series: pd.Series, *, scaler_fit_pool: str
) -> pd.DataFrame:
    if scaler_fit_pool == "index_pool":
        return matrix
    if scaler_fit_pool == "upenn_only":
        keep = source_series[source_series == UPENN_SOURCE_NAME].index
        return matrix.loc[matrix.index.intersection(keep)]
    if scaler_fit_pool == "upenn_lumiere":
        keep = source_series[source_series.isin([UPENN_SOURCE_NAME, LUMIERE_SOURCE_NAME])].index
        return matrix.loc[matrix.index.intersection(keep)]
    if scaler_fit_pool == "upenn_lumiere_tcga":
        # 2026-09-17 (Barış onayı) -- v2'de UCSF indekse GIRER ama scaler
        # ORIJINAL v1 havuzuna (UPenn+LUMIERE+TCGA = 722) SABITLENIR.
        #
        # 🔴 NEDEN ZORUNLU: varsayilan `index_pool` ile UCSF eklenirse
        # scaler 1017 hasta uzerinde YENIDEN fit edilir ve MEVCUT 722
        # vektorun HEPSI degisir -- v1 yeniden uretilemez hale gelir ve
        # raporlanmis kom&scedil;uluk kanitlari (2026-09-16: `TCGA-06-5412`
        # -> `UPENN-GBM-00540`, L2 22,36) sessizce kayar.
        #
        # Ayrica bu secim, harici test setine EGITIM-TUREVI bir donusum
        # uygulama tartismasini da acik tutar: UCSF vektorleri v1'in
        # dondurulmus istatistikleriyle DONUSTURULUR (out-of-sample),
        # v1'in istatistiklerini DEGISTIRMEZ. FAISS modelin parcasi
        # DEGILDIR (C-index/EPV gibi hicbir raporlanan sayiyi etkilemez),
        # ama bu asimetri raporda BEYAN EDILMELIDIR.
        keep = source_series[
            source_series.isin(
                [UPENN_SOURCE_NAME, LUMIERE_SOURCE_NAME, TCGA_SOURCE_NAME]
            )
        ].index
        return matrix.loc[matrix.index.intersection(keep)]
    raise ValueError(f"Bilinmeyen scaler_fit_pool={scaler_fit_pool!r}")


# =====================================================================
# 5) Omics -- 🔒 11 BOYUT KİLİTLİ (2026-08-18, Barış) -- 48 hasta
# =====================================================================
#
# `raw/plan/plan.txt` satır 444-445 tam listeyi veriyor (BELİRSİZLİK YOKMUŞ
# -- önceki revizyonda "..." ifadesini yanlışlıkla belirsiz saymıştım, bu
# DÜZELTİLDİ): `[MGMT_m/e/c, EGFR_m/e/c, PTEN_m/e/c, tmz_score, aggr_score]`
# = 3 gen × 3 platform (methylation/expression/cnv) + 2 skor = **11 boyut**.
# Canlı DB'de doğrulandı: bu 11 alanın 48/48 satırda %100 dolu olduğu
# (`mgmt_*`/`egfr_*`/`pten_*` + `tmz_resistance_score`/`aggressiveness_
# score`) -- ek veri/doldurma GEREKMİYOR.
#
# NOT -- `omics_profiles` tablosunda **12 gen × 3 platform = 36 alan** var
# (MGMT/MSH2/MSH6/MLH1/PMS2/EGFR/PTEN/TP53/CDKN2A/BRCA1/BRCA2/CHEK2) --
# 11 boyut bunun küçük bir alt kümesi, geri kalan 9 gen (27 alan) vektöre
# GİRMEZ.
#
# ⚠️ AÇIK NOKTA (uydurulmadı, Barış'a sorulacak): `dna_repair_score`
# (`molecular_scores` tablosunda, 48/48 dolu, canlı SELECT ile doğrulandı)
# neden 11 boyutun DIŞINDA bırakılmış -- hiçbir belgede gerekçe yok. Bu
# fonksiyon `dna_repair_score`'u DAHİL ETMEZ (plan.txt'in literal listesine
# sadık kalınıyor), ama bunun bilinçli bir tasarım kararı mı yoksa
# plan.txt'in kendi eksikliği mi olduğu NETLEŞTİRİLMEDİ.
OMICS_NUMERIC_COLUMNS: tuple[str, ...] = (
    "mgmt_methylation", "mgmt_expression", "mgmt_cnv",
    "egfr_methylation", "egfr_expression", "egfr_cnv",
    "pten_methylation", "pten_expression", "pten_cnv",
)
MOLECULAR_SCORE_COLUMNS: tuple[str, ...] = ("tmz_resistance_score", "aggressiveness_score")

if len(OMICS_NUMERIC_COLUMNS) + len(MOLECULAR_SCORE_COLUMNS) != 11:  # pragma: no cover
    raise RuntimeError(
        "Omics vektörü 11 boyut olmalı (plan.txt:444-445, Barış kararı "
        f"2026-08-18) -- {len(OMICS_NUMERIC_COLUMNS) + len(MOLECULAR_SCORE_COLUMNS)} bulundu."
    )


def build_omics_index(cursor) -> tuple[np.ndarray, pd.DataFrame, FaissBuildReport, FrozenScaler]:
    """`molecular_omics_faiss` -- 48 hasta, TCGA-Omics, 🔒 11-boyut KİLİTLİ."""

    real_cursor = cursor.connection.cursor(cursor_factory=RealDictCursor)
    try:
        omics_cols_sql = ", ".join(f"op.{c}" for c in OMICS_NUMERIC_COLUMNS)
        score_cols_sql = ", ".join(f"ms.{c}" for c in MOLECULAR_SCORE_COLUMNS)
        real_cursor.execute(
            f"""
            SELECT op.patient_id, {omics_cols_sql}, {score_cols_sql}
            FROM omics_profiles op
            JOIN molecular_scores ms ON ms.patient_id = op.patient_id
            """
        )
        rows = real_cursor.fetchall()
    finally:
        real_cursor.close()

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise EmptyIndexPoolError("omics_profiles x molecular_scores: 0 satır.")
    frame = frame.set_index("patient_id").sort_index()

    feature_cols = list(OMICS_NUMERIC_COLUMNS) + list(MOLECULAR_SCORE_COLUMNS)
    numeric = frame[feature_cols].astype(float)

    n_before = len(numeric)
    numeric = numeric.dropna(axis=0, how="any")
    n_dropped = n_before - len(numeric)
    warnings = []
    if n_dropped:
        warnings.append(
            f"{n_dropped} hasta NaN içeren omics/skor satırı yüzünden "
            "dışlandı (sessizce doldurulmadı -- kaynak kod bunu "
            "raporluyor, DOLDURMUYOR)."
        )

    source_series = pd.Series("TCGA-Omics", index=numeric.index)
    scaler = fit_frozen_scaler(numeric, source_series=source_series)
    vectors = scaler.transform(numeric)

    report = FaissBuildReport(
        design="omics_11dim_mgmt_egfr_pten_tmz_aggr",
        combat=False,
        n_vectors=len(numeric),
        dim=vectors.shape[1],
        source_counts={"TCGA-Omics": len(numeric)},
        lumiere_visit_rule=None,
        lumiere_visit_report=None,
        pivot_reports={},
        scaler_fit_pool="index_pool",
        scaler_fit_pool_sources=scaler.fit_pool_sources,
        warnings=warnings,
    )
    metadata = pd.DataFrame({"source": source_series})
    return vectors, metadata, report, scaler


# =====================================================================
# 6) Determinizm yardımcıları + artifact yazımı (--apply)
# =====================================================================


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_index_artifacts(
    *,
    vectors: np.ndarray,
    metadata: pd.DataFrame,
    clinical_metadata: pd.DataFrame,
    report: FaissBuildReport,
    scaler: FrozenScaler,
    output_dir: Path,
    index_filename: str = "clinical_radiomics.index",
) -> dict[str, str]:
    """Gerçek FAISS indeks dosyasını + eşlik eden artifact'leri yazar.

    ⚠️ Windows'ta Türkçe-karakter path bug'ı (`faiss.write_index()`'in C++
    `fopen()` çağrısı proje kökündeki 'Barış' klasör adını çözemiyor --
    projede ITK için de belgelenen AYNI hata sınıfı) -- `output_dir` niyeti
    ASCII olmayan bir yol içeriyorsa bu fonksiyon `subst` (`X:`/`Y:`) gibi
    bir ASCII-safe yeniden-eşlemeden geçirilmiş bir yoldan ÇAĞRILMALI (bu
    görevde uygulanan çözüm), aksi halde `RuntimeError` fırlatır.
    """

    import faiss  # local import -- yalnız --apply modunda gerekli

    output_dir.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    index_path = output_dir / index_filename
    faiss.write_index(index, str(index_path))

    order_frame = metadata.copy()
    order_frame.insert(0, "row_index", range(len(order_frame)))
    order_path = output_dir / "patient_order.csv"
    order_frame.to_csv(order_path)

    scaler_path = output_dir / "scaler.json"
    scaler_path.write_text(json.dumps(scaler.to_json_dict(), indent=2), encoding="utf-8")

    clinical_path = output_dir / "clinical_metadata.csv"
    clinical_metadata.to_csv(clinical_path)

    written = {
        "index": str(index_path),
        "patient_order": str(order_path),
        "scaler": str(scaler_path),
        "clinical_metadata": str(clinical_path),
    }

    manifest = {
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": _sha256_file(Path(__file__)),
        "report": report.__dict__,
        "files_sha256": {name: _sha256_file(Path(p)) for name, p in written.items()},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    written["manifest"] = str(manifest_path)
    return written


# =====================================================================
# 7) CLI
# =====================================================================


def _print_report(report: FaissBuildReport) -> None:
    print(f"Tasarım           : {report.design}")
    print(f"ComBat            : {report.combat}")
    print(f"Vektör sayısı     : {report.n_vectors}")
    print(f"Boyut             : {report.dim}")
    print(f"Kaynak dağılımı   : {report.source_counts}")
    print(f"Dışlanan kaynaklar: {report.excluded_sources}")
    print(f"LUMIERE kural     : {report.lumiere_visit_rule}")
    if report.lumiere_visit_report:
        print(f"LUMIERE vizit raporu: {report.lumiere_visit_report}")
    print(f"Scaler fit havuzu : {report.scaler_fit_pool} -> {report.scaler_fit_pool_sources}")
    for key, val in report.pivot_reports.items():
        print(f"Pivot raporu [{key}]: {val}")
    for warning in report.warnings:
        print(f"UYARI: {warning}")


def main(argv: list[str] | None = None) -> int:
    """K1/K2 KİLİTLİ -- gerçek üretim komutu (bu artık ÇALIŞTIRILABİLİR):

    Tasarım A (ham C32, ComBat'sız) -- v1 ÜRETİM indeksi, dry-run:
        python tools/rebuild_faiss_indexes.py --no-combat

    Tasarım A, gerçek dosya üretimi (K1'in seçtiği tasarım):
        python tools/rebuild_faiss_indexes.py --no-combat --apply

    Tasarım B (ComBat'lı, B1 -- TCGA hariç) -- K1 SEÇMEDİ, yalnız olası bir
    duyarlılık kolu için korunuyor:
        python tools/rebuild_faiss_indexes.py --combat --apply \\
            --output-dir artifacts/week3/faiss_index/design_b_combat_sensitivity

    Omics (opsiyonel, ayrı, 11-boyut kilitli):
        python tools/rebuild_faiss_indexes.py --no-combat --build-omics
    """

    parser = argparse.ArgumentParser(description=__doc__)
    combat_group = parser.add_mutually_exclusive_group(required=True)
    combat_group.add_argument("--combat", action="store_true", help="Tasarım B (UPenn-referanslı ComBat, B1 -- TCGA hariç). K1 SEÇMEDİ, yalnız duyarlılık kolu.")
    combat_group.add_argument("--no-combat", action="store_true", help="Tasarım A (ham C32, ComBat'sız) -- K1'in SEÇTİĞİ v1 üretim tasarımı.")
    parser.add_argument(
        "--scaler-fit-pool",
        choices=["index_pool", "upenn_only", "upenn_lumiere", "upenn_lumiere_tcga"],
        default="index_pool",
        # 2026-09-14 (kalem 5): yol düzeltildi -- taslak `concepts/` değil
        # `archive/` altında (wiki hard rule #4 taşıması, diskten doğrulandı).
        help="Dondurulmuş StandardScaler'ın fit edileceği kohort (uygulama detayı, K1'den bağımsız AÇIK -- bkz. archive/faiss-vektor-spesifikasyonu-taslak.md; güncel karar: decisions/2026-09-13-k7-faiss-v1-vektor-icerigi-onaylandi.md).",
    )
    parser.add_argument(
        "--include-ucsf",
        action="store_true",
        help=(
            "Tasarım A havuzuna UCSF-PDGM'yi de kat (v2, 1017 hasta). "
            "VARSAYILAN KAPALI -- v1 (722) davranışı değişmez. "
            "`--scaler-fit-pool upenn_lumiere_tcga` ZORUNLUDUR ve ayrı bir "
            "`--output-dir` verilmelidir."
        ),
    )
    parser.add_argument("--build-omics", action="store_true", help="molecular_omics_faiss'i de üret (opsiyonel, 11-boyut kilitli).")
    parser.add_argument("--apply", action="store_true", help="Belirtilmezse dry-run (dosya YAZILMAZ).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "week3" / "faiss_index",
        help="--apply modunda indeks dosyalarının yazılacağı dizin.",
    )
    args = parser.parse_args(argv)

    # 🔴 SESSIZ SAPMA KALKANI (2026-09-17): UCSF havuza girerken varsayilan
    # `index_pool` birakilirsa scaler 1017 hasta uzerinde YENIDEN fit edilir
    # ve MEVCUT 722 vektorun HEPSI degisir. Bu, v1'i yeniden uretilemez
    # yapar ve raporlanmis komsuluk kanitlarini kaydirir. Bu yuzden burada
    # SERT DURULUR -- otomatik "dogrusunu secme" YAPILMAZ, cunku o da sessiz
    # bir karar olurdu.
    if args.include_ucsf and args.scaler_fit_pool == "index_pool":
        parser.error(
            "--include-ucsf ile --scaler-fit-pool index_pool BIRLIKTE "
            "KULLANILAMAZ: scaler 1017 hasta uzerinde yeniden fit edilir ve "
            "v1'in 722 vektorunun HEPSI degisir. Dogru kullanim: "
            "--include-ucsf --scaler-fit-pool upenn_lumiere_tcga "
            "--output-dir <v1'DEN FARKLI bir dizin>"
        )
    if args.include_ucsf and args.combat:
        parser.error(
            "--include-ucsf yalniz Tasarim A (--no-combat) icin tanimlidir; "
            "UCSF ComBat fit'ine GIRMEZ (CLAUDE.md kritik kural)."
        )

    connection = get_connection(readonly=True)
    try:
        cursor = connection.cursor()
        try:
            if args.combat:
                vectors, source_metadata, report, scaler = build_design_b_combat(
                    cursor, scaler_fit_pool=args.scaler_fit_pool
                )
                clinical_frames = []
                if UPENN_SOURCE_NAME in report.source_counts:
                    clinical_frames.append(
                        fetch_patients_metadata_frame(cursor, source_name=UPENN_SOURCE_NAME)
                    )
                if LUMIERE_SOURCE_NAME in report.source_counts:
                    clinical_frames.append(
                        fetch_patients_metadata_frame(cursor, source_name=LUMIERE_SOURCE_NAME)
                    )
            else:
                vectors, source_metadata, report, scaler = build_design_a_ham_c32(
                    cursor,
                    scaler_fit_pool=args.scaler_fit_pool,
                    include_ucsf=args.include_ucsf,
                )
                clinical_frames = [
                    fetch_patients_metadata_frame(cursor, source_name=UPENN_SOURCE_NAME)
                ]
                if LUMIERE_SOURCE_NAME in report.source_counts:
                    clinical_frames.append(
                        fetch_patients_metadata_frame(cursor, source_name=LUMIERE_SOURCE_NAME)
                    )
                clinical_frames.append(
                    fetch_patients_metadata_frame(cursor, source_name=TCGA_SOURCE_NAME)
                )
                if UCSF_SOURCE_NAME in report.source_counts:
                    clinical_frames.append(
                        fetch_patients_metadata_frame(cursor, source_name=UCSF_SOURCE_NAME)
                    )
            clinical_frames = [f for f in clinical_frames if not f.empty]
            clinical_metadata = pd.concat(clinical_frames, axis=0, sort=False)
            clinical_metadata = clinical_metadata.reindex(source_metadata.index)

            print("=" * 70)
            print("clinical_radiomics_faiss")
            print("=" * 70)
            _print_report(report)

            if args.apply:
                written = write_index_artifacts(
                    vectors=vectors,
                    metadata=source_metadata,
                    clinical_metadata=clinical_metadata,
                    report=report,
                    scaler=scaler,
                    output_dir=args.output_dir,
                )
                print(f"YAZILDI: {written}")
            else:
                print("DRY-RUN -- hiçbir dosya yazılmadı (--apply verilmedi).")

            if args.build_omics:
                omics_vectors, omics_metadata, omics_report, omics_scaler = build_omics_index(cursor)
                print("=" * 70)
                print("molecular_omics_faiss")
                print("=" * 70)
                _print_report(omics_report)
                if args.apply:
                    omics_dir = args.output_dir.parent / "molecular_omics"
                    written_omics = write_index_artifacts(
                        vectors=omics_vectors,
                        metadata=omics_metadata,
                        clinical_metadata=omics_metadata,
                        report=omics_report,
                        scaler=omics_scaler,
                        output_dir=omics_dir,
                        index_filename="molecular_omics.index",
                    )
                    print(f"YAZILDI (omics): {written_omics}")
                else:
                    print("DRY-RUN (omics) -- hiçbir dosya yazılmadı.")
        finally:
            cursor.close()
    finally:
        connection.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
