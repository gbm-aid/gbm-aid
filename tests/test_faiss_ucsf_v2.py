"""FAISS v2 -- UCSF'in benzer-hasta indeksine EKLENMESİ (2026-09-17, Barış onayı).

NEDEN VAR
=========
v1 indeksi K1 kararıyla 3 kaynakla donduruldu: UPenn 611 + LUMIERE 72 +
TCGA 39 = **722**; UCSF-PDGM DAHİL DEĞİLDİ. Sonuç olarak canlı sitede bir
UCSF hastası açıldığında "Benzer Hastalar" bloğu 422 veriyordu (sessiz boş
sonuç değil -- doğru davranış, ama jüriye eksik görünüyor).

Barış 2026-09-17'de UCSF'in eklenmesini istedi. Ölçülen kapsam:
UPenn 611 + LUMIERE 72 + TCGA 39 + **UCSF 295** = **1017** (dördü de
`_fetch_source_wt93` ile canlı DB'den sayıldı, hepsi 107 özellik).

🔴 BU DOSYANIN ASIL İŞİ: SESSİZ SAPMAYI ÖNLEMEK
===============================================
`--scaler-fit-pool` VARSAYILANI `index_pool`'dur. UCSF havuza eklenip bu
varsayılan bırakılırsa dondurulmuş `StandardScaler` **1017 hasta üzerinde
yeniden fit edilir** ve mevcut **722 vektörün HEPSİ değişir** -- yani v1
yeniden üretilemez hale gelir ve raporlanmış komşuluk kanıtları
(2026-09-16: `TCGA-06-5412` -> `UPENN-GBM-00540`, L2 22,36) sessizce kayar.

Bu yüzden `--include-ucsf` ile `--scaler-fit-pool index_pool` birlikte
kullanılamaz; CLI SERT durur. Otomatik "doğrusunu seçme" BİLİNÇLİ olarak
yapılmaz -- o da sessiz bir karar olurdu.

⚠️ DEPO VARSAYILANI DEĞİŞMEDİ: `FAISS_V1_INDEX_SEGMENTATION_TOOLS` hâlâ 3
kaynaktır ve `--include-ucsf` verilmedikçe davranış v1'dir. v2 AYRI bir
`--output-dir`e yazılır; hangi indeksin SERVİS edildiği
`GBMAID_FAISS_CLINICAL_RADIOMICS_INDEX_DIR` ile belirlenir.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

faiss = pytest.importorskip("faiss", reason="faiss-cpu kurulu değil")

import rebuild_faiss_indexes as rfi  # noqa: E402
import api.similar as similar_module  # noqa: E402
from test_api_similar import _make_bundle  # type: ignore[import-not-found]  # noqa: E402


# =====================================================================
# 1) CLI KALKANLARI -- sessiz sapma önleme
# =====================================================================


def test_include_ucsf_ile_index_pool_REDDEDILIR():
    """Asıl koruma. Bu kalkan olmasaydı scaler 1017'de yeniden fit edilir
    ve v1'in 722 vektörü sessizce kayardı."""

    with pytest.raises(SystemExit):
        rfi.main(["--no-combat", "--include-ucsf"])  # varsayılan = index_pool

    with pytest.raises(SystemExit):
        rfi.main(["--no-combat", "--include-ucsf", "--scaler-fit-pool", "index_pool"])


def test_include_ucsf_ile_combat_REDDEDILIR():
    """UCSF ComBat fit'ine GİRMEZ (CLAUDE.md kritik kuralı) -- bu yüzden
    `--include-ucsf` yalnız Tasarım A için tanımlıdır."""

    with pytest.raises(SystemExit):
        rfi.main(
            [
                "--combat",
                "--include-ucsf",
                "--scaler-fit-pool",
                "upenn_lumiere_tcga",
            ]
        )


# =====================================================================
# 2) SCALER FİT HAVUZU -- UCSF dışarıda kalır
# =====================================================================


def _kucuk_matris(patient_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        np.arange(len(patient_ids) * 2, dtype=float).reshape(len(patient_ids), 2),
        index=pd.Index(patient_ids, name="patient_id"),
        columns=["a", "b"],
    )


def test_upenn_lumiere_tcga_havuzu_UCSF_i_DISLAR():
    ids = ["UPENN-1", "Patient-1", "TCGA-1", "UCSF-PDGM-1"]
    matrix = _kucuk_matris(ids)
    sources = pd.Series(
        {
            "UPENN-1": rfi.UPENN_SOURCE_NAME,
            "Patient-1": rfi.LUMIERE_SOURCE_NAME,
            "TCGA-1": rfi.TCGA_SOURCE_NAME,
            "UCSF-PDGM-1": rfi.UCSF_SOURCE_NAME,
        }
    )

    pool = rfi._select_scaler_fit_pool(
        matrix, sources, scaler_fit_pool="upenn_lumiere_tcga"
    )

    assert list(pool.index) == ["UPENN-1", "Patient-1", "TCGA-1"]
    assert "UCSF-PDGM-1" not in pool.index


def test_index_pool_havuzu_UCSF_i_DAHIL_EDER_bu_yuzden_yasakli():
    """Karşıt kanıt: `index_pool` gerçekten UCSF'i içeri alır -- kalkanın
    neden gerekli olduğunu gösterir."""

    ids = ["UPENN-1", "UCSF-PDGM-1"]
    matrix = _kucuk_matris(ids)
    sources = pd.Series(
        {"UPENN-1": rfi.UPENN_SOURCE_NAME, "UCSF-PDGM-1": rfi.UCSF_SOURCE_NAME}
    )

    pool = rfi._select_scaler_fit_pool(matrix, sources, scaler_fit_pool="index_pool")

    assert "UCSF-PDGM-1" in pool.index


def test_bilinmeyen_havuz_adi_SESSIZCE_gecmez():
    ids = ["UPENN-1"]
    with pytest.raises(ValueError):
        rfi._select_scaler_fit_pool(
            _kucuk_matris(ids),
            pd.Series({"UPENN-1": rfi.UPENN_SOURCE_NAME}),
            scaler_fit_pool="ne_oldugu_belirsiz",
        )


# =====================================================================
# 3) build_design_a_ham_c32 -- include_ucsf kablolaması
# =====================================================================


def _sahte_wide(patient_ids: list[str]) -> pd.DataFrame:
    """WT_FEATURE_COLUMNS'un tamamını taşıyan küçük bir kare."""

    data = {
        col: np.linspace(1.0, 2.0, len(patient_ids))
        for col in rfi.WT_FEATURE_COLUMNS
    }
    return pd.DataFrame(data, index=pd.Index(patient_ids, name="patient_id"))


@pytest.fixture()
def sahte_kaynaklar(monkeypatch):
    """`_fetch_source_wt93`'ü DB'siz taklit eder ve HANGİ araçların
    sorgulandığını kaydeder."""

    istenen: list[str] = []
    kaynak_haritasi = {
        rfi.SEGMENTATION_TOOL_UPENN_C32: ["UPENN-1", "UPENN-2"],
        rfi.SEGMENTATION_TOOL_LUMIERE_C32: ["Patient-1"],
        rfi.SEGMENTATION_TOOL_TCGA_C32: ["TCGA-1"],
        rfi.SEGMENTATION_TOOL_UCSF_C32: ["UCSF-PDGM-1", "UCSF-PDGM-2"],
    }

    def _sahte(cursor, *, segmentation_tool):
        istenen.append(segmentation_tool)
        ids = kaynak_haritasi[segmentation_tool]
        # 3. dönüş LUMIERE vizit raporudur ve üretim kodu `__dict__`ine
        # bakar -- düz `dict` DEĞİL, nitelikli bir nesne olmalı.
        return _sahte_wide(ids), pd.DataFrame(), SimpleNamespace(n_output_patients=len(ids))

    monkeypatch.setattr(rfi, "_fetch_source_wt93", _sahte)
    return istenen


def test_varsayilan_UCSF_i_SORGULAMAZ_bile(sahte_kaynaklar):
    """v1 davranışı KORUNDU: `include_ucsf` verilmezse UCSF aracı hiç
    sorgulanmaz (yalnızca havuzdan düşürülmüş de değil -- DB'ye o sorgu
    HİÇ gitmez)."""

    _, _, report, _ = rfi.build_design_a_ham_c32(
        cursor=None, scaler_fit_pool="index_pool"
    )

    assert rfi.SEGMENTATION_TOOL_UCSF_C32 not in sahte_kaynaklar
    assert rfi.UCSF_SOURCE_NAME not in report.source_counts
    assert set(report.source_counts) == {
        rfi.UPENN_SOURCE_NAME,
        rfi.LUMIERE_SOURCE_NAME,
        rfi.TCGA_SOURCE_NAME,
    }


def test_include_ucsf_havuza_UCSF_i_KATAR(sahte_kaynaklar):
    vectors, _, report, scaler = rfi.build_design_a_ham_c32(
        cursor=None,
        scaler_fit_pool="upenn_lumiere_tcga",
        include_ucsf=True,
    )

    assert rfi.SEGMENTATION_TOOL_UCSF_C32 in sahte_kaynaklar
    assert report.source_counts[rfi.UCSF_SOURCE_NAME] == 2
    # 2 UPenn + 1 LUMIERE + 1 TCGA + 2 UCSF = 6 vektör
    assert vectors.shape[0] == 6
    assert vectors.shape[1] == len(rfi.WT_FEATURE_COLUMNS)
    # 🔴 Scaler UCSF'i GÖRMEDİ -- fit havuzu yalnız 4 hasta.
    assert rfi.UCSF_SOURCE_NAME not in scaler.fit_pool_sources
    assert sum(scaler.fit_pool_sources.values()) == 4


# =====================================================================
# 4) API -- 422 mesajı artık SABİT 722 demiyor
# =====================================================================


def test_PatientNotInIndexError_indeks_boyutunu_TASIR():
    """422 metni sabit yazılıydı ("722 hasta ... UCSF DAHİL DEĞİL"); v2
    devreye alınınca o cümle SESSİZCE yanlış olurdu. Artık sayı yüklü
    artefakttan gelir."""

    bundle = _make_bundle({"UPENN-1": [1.0, 2.0], "UPENN-2": [3.0, 4.0]})

    with pytest.raises(similar_module.PatientNotInIndexError) as exc_info:
        similar_module._query_all_neighbors_sorted(bundle, "INDEKSTE-YOK-1")

    assert exc_info.value.index_size == bundle.index.ntotal == 2
    assert exc_info.value.patient_id == "INDEKSTE-YOK-1"


def test_index_size_yoksa_mesaj_yine_de_URETILEBILIR():
    """Geriye dönük uyumluluk: `index_size` verilmeden de fırlatılabilir
    (eski çağrı biçimi kırılmaz)."""

    hata = similar_module.PatientNotInIndexError("X-1")
    assert hata.index_size is None
    assert hata.patient_id == "X-1"
