"""``GET /patient/{patient_id}/similar`` -- yaş/cinsiyet/IDH/MGMT
zenginleştirme alanları (görev talimatı: paralel-38, backend-agent-X1,
2026-09-16, "SITE AŞAMA 3-A").

BULGU (bu dosya bunu REGRESYON PİNİ olarak sabitler): kod OKUNDUĞUNDA
`api/similar.py::_RESULT_METADATA_COLUMNS` ZATEN `age`/`gender`/
`idh1_status`/`mgmt_status`/`gtr_over90percent`/`source`/`kps_score`
alanlarını taşıyordu (`_build_similar_block()`'un `result_columns` ile
her komşu satırına bu kolonları EKLEDİĞİ` doğrulandı) -- görev talimatının
"şu an yalnız l2_distance/survival_days/vital_status dönüyor" önermesi bu
kod tabanı için YANLIŞ ÇIKTI (bkz. backend-agent-X1'in final raporu).
Bu dosya, o zenginleştirmenin GERÇEKTEN var olduğunu ve `api/similar.py`'ye
DOKUNULMADAN (görev kısıtı: "yalnız yanıt alanı zenginleştirme") hem
`clinical_radiomics_faiss` hem `molecular_omics_faiss` katmanlarında
göründüğünü kilitler -- ileride biri bu alanları YANLIŞLIKLA kaldırırsa
bu test KIRMIZI olur.

`tests/test_api_similar.py`'nin AYNI yardımcı desenini (`_make_bundle`,
`_metadata_frame`) kullanır -- bir üçüncü kopya YAZILMAZ, doğrudan
`similar_module`'den import edilen fonksiyonlar/sabitler kullanılır.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

faiss = pytest.importorskip("faiss", reason="faiss-cpu kurulu değil (requirements.txt'e bakın)")

import api.similar as similar_module  # noqa: E402


def _make_bundle(
    vectors: dict[str, list[float]], sources: dict[str, str] | None = None
) -> similar_module.IndexBundle:
    sources = sources or {}
    patient_ids = list(vectors.keys())
    dim = len(next(iter(vectors.values())))
    matrix = np.array([vectors[pid] for pid in patient_ids], dtype="float32")

    index = faiss.IndexFlatL2(dim)
    index.add(matrix)

    order = pd.DataFrame(
        {
            "row_index": range(len(patient_ids)),
            "source": [sources.get(pid, "UPenn-GBM") for pid in patient_ids],
        },
        index=pd.Index(patient_ids, name="patient_id"),
    )
    return similar_module.IndexBundle(index=index, order=order, mtime_key=0.0)


def _metadata_frame(rows: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["source", *similar_module._CLINICAL_METADATA_RAW_COLUMNS]
        )
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "patient_id"
    return frame


def test_result_metadata_columns_include_age_gender_idh_mgmt():
    """Sözleşme sabiti: bu 4 alan (+ tasarımın istediği diğerleri)
    `_RESULT_METADATA_COLUMNS`'ta OLMAK ZORUNDA -- eksik bulunan bir isim
    varsa tasarımın istediği zenginleştirme kod tabanından KAYBOLMUŞ demektir."""

    required = {"age", "gender", "idh1_status", "mgmt_status"}
    assert required <= set(similar_module._RESULT_METADATA_COLUMNS)


def test_clinical_radiomics_neighbor_entries_carry_demographic_and_molecular_fields(
    monkeypatch,
):
    bundle = _make_bundle({"A": [0.0], "B": [1.0], "C": [2.0]})
    meta = {
        "B": {
            "source": "UPenn-GBM",
            "age": 61.0,
            "gender": "Male",
            "idh1_status": "Wildtype",
            "mgmt_status": "Methylated",
            "gtr_over90percent": "Y",
            "vital_status": "Dead",
            "survival_days": 420.0,
        },
        "C": {
            "source": "LUMIERE",
            "age": None,  # eksik deger -- ELENMEZ, None olarak GEÇER
            "gender": "Female",
            "idh1_status": "WT",
            "mgmt_status": None,
            "gtr_over90percent": None,
            "vital_status": None,
            "survival_days": None,
        },
    }
    monkeypatch.setattr(
        similar_module,
        "_fetch_clinical_metadata_frame",
        lambda ids: _metadata_frame({pid: meta[pid] for pid in ids if pid in meta}),
    )

    block = similar_module._build_similar_block(
        bundle,
        "A",
        10,
        idh1_status=None,
        mgmt_status=None,
        result_columns=similar_module._RESULT_METADATA_COLUMNS,
    )

    by_id = {r["patient_id"]: r for r in block["results"]}

    assert by_id["B"]["age"] == pytest.approx(61.0)
    assert by_id["B"]["gender"] == "Male"
    assert by_id["B"]["idh1_status"] == "Wildtype"
    assert by_id["B"]["mgmt_status"] == "Methylated"
    # Mevcut alanlar (l2_distance/survival_days/vital_status) DEĞİŞMEMİŞ.
    assert by_id["B"]["survival_days"] == pytest.approx(420.0)
    assert by_id["B"]["vital_status"] == "Dead"
    assert "l2_distance" in by_id["B"]

    # Eksik deger SESSİZCE elenmez -- None olarak GÖRÜNÜR kalır (zarifçe
    # bozulma ilkesi, `_num_or_none`/`_val_or_none`).
    assert by_id["C"]["age"] is None
    assert by_id["C"]["mgmt_status"] is None
    assert by_id["C"]["gender"] == "Female"


def test_omics_layer_neighbor_entries_also_carry_demographic_fields(monkeypatch):
    """`molecular_omics_faiss` katmanı da AYNI `_RESULT_METADATA_COLUMNS`
    ile `_build_similar_block()`'u çağırıyor (`api/similar.py::
    _build_omics_similar_block`) -- omics komşularında da bu alanlar
    bulunmalı, ayrı bir kopya/eksik sözleşme OLUŞMAMALI."""

    bundle = _make_bundle({"X": [0.0], "Y": [1.0]})
    monkeypatch.setattr(
        similar_module,
        "_load_index_bundle",
        lambda index_dir, filename: bundle,
    )
    monkeypatch.setattr(
        similar_module,
        "_fetch_clinical_metadata_frame",
        lambda ids: _metadata_frame(
            {
                pid: {
                    "source": "TCGA-Omics",
                    "age": 55.0,
                    "gender": "Female",
                    "idh1_status": "Mutated",
                    "mgmt_status": "Unmethylated",
                }
                for pid in ids
            }
        ),
    )

    block = similar_module._build_omics_similar_block(
        "X", 10, idh1_status=None, mgmt_status=None
    )

    assert block["available"] is True
    entry = block["results"][0]
    assert entry["age"] == pytest.approx(55.0)
    assert entry["gender"] == "Female"
    assert entry["idh1_status"] == "Mutated"
    assert entry["mgmt_status"] == "Unmethylated"
