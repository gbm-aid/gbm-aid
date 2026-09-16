"""ComBat/neuroHarmonize fit+apply için SMOKE TEST -- gerçek üretim fit'i DEĞİL.

Amaç: `pipeline.harmonization.fit_combat_harmonization()` ve
`apply_combat_harmonization()`'ın gerçek Supabase verisiyle uçtan uca
çalıştığını, `combat_parameters` şemasına uyan çıktı ürettiğini ve TCGA
sızıntı korumasının gerçekten hata fırlattığını kanıtlamak.

ÖNEMLİ -- bu GERÇEK UPenn+LUMIERE fit'i DEĞİLDİR:
UPenn'in kanonik 107-özellik verisi bu script'in yazıldığı tarihte
(2026-08-09) henüz DB'de yok (bkz. decisions/2026-08-07-upenn-144-vs-107-uyumsuzlugu.md).
Bu yüzden fit() burada LUMIERE'in KENDİ İÇİNDEKİ iki segmentasyon aracını
(DeepBraTumIA vs HD-GLIO-AUTO) yapay/geçici "batch" gibi kullanır -- ikisi
de gerçek LUMIERE verisi, TCGA HİÇBİR ZAMAN fit()'e girmez (mimari kural).
apply() ise gerçek TCGA canonical verisiyle out-of-sample projeksiyonu
test eder -- bu kısmı GERÇEK apply senaryosuna birebir denk (TCGA zaten
apply()'dan başka bir yere giremez).

UPenn'in kanonik verisi hazır olduğunda bu script YERİNE gerçek
UPenn+LUMIERE fit'i çalıştıran ayrı bir üretim script'i yazılmalı (bu
script SADECE plumbing'i doğrulamak için var, sonucu combat_parameters'a
YAZMAZ).

Çalıştırma: readonly SELECT dışında hiçbir DB yazımı yapmaz.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from psycopg2.extras import RealDictCursor

import neuroHarmonize as nh
from db_connection import get_connection
from pipeline.harmonization import (
    ComBatApplyLeakageError,
    ComBatFitLeakageError,
    _model_to_combat_parameter_rows,
    apply_combat_harmonization,
    fit_combat_harmonization,
)

CANONICAL_KEY_COUNTS = {"shape": 14, "first_order": 18, "texture": 75}


def _fetch_canonical_feature_matrix(
    cursor, *, source_name: str, segmentation_tool: str, tumor_region: str
) -> pd.DataFrame:
    """Kanonik (14+18+75=107) satırları çek, hasta_id -> 107 sütunluk vektör."""

    cursor.execute(
        """
        SELECT p.patient_id, r.shape_features, r.first_order_features,
               r.texture_features
        FROM radiomics r
        JOIN mr_scans ms ON ms.scan_id = r.scan_id
        JOIN patients p ON p.patient_id = ms.patient_id
        JOIN dataset_sources ds ON ds.source_id = p.source_id
        WHERE ds.source_name = %s
          AND r.segmentation_tool = %s
          AND r.tumor_region = %s
        """,
        (source_name, segmentation_tool, tumor_region),
    )
    rows = cursor.fetchall()

    records: dict[str, dict[str, float]] = {}
    for row in rows:
        shape = row["shape_features"] or {}
        first_order = row["first_order_features"] or {}
        texture = row["texture_features"] or {}
        if (
            len(shape) != CANONICAL_KEY_COUNTS["shape"]
            or len(first_order) != CANONICAL_KEY_COUNTS["first_order"]
            or len(texture) != CANONICAL_KEY_COUNTS["texture"]
        ):
            continue  # kanonik olmayan (eski CaPTk/kısmi) satır, atla
        merged = {**shape, **first_order, **texture}
        if any(v is None for v in merged.values()):
            continue
        # aynı hasta için birden fazla satır varsa (tekrar tarama) ilkini al
        records.setdefault(row["patient_id"], merged)

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame.from_dict(records, orient="index")
    frame.index.name = "patient_id"
    return frame.sort_index()


def main() -> int:
    connection = get_connection(readonly=True)
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)

        lumiere_deep = _fetch_canonical_feature_matrix(
            cursor,
            source_name="LUMIERE",
            segmentation_tool="DeepBraTumIA",
            tumor_region="Contrast-enhancing",
        )
        lumiere_hdglio = _fetch_canonical_feature_matrix(
            cursor,
            source_name="LUMIERE",
            segmentation_tool="HD-GLIO-AUTO",
            tumor_region="Contrast-enhancing",
        )
        tcga_wt = _fetch_canonical_feature_matrix(
            cursor,
            source_name="TCGA-GBM",
            segmentation_tool="TCGA-ground-truth",
            tumor_region="WT",
        )
        cursor.close()
    finally:
        connection.close()

    print(f"LUMIERE/DeepBraTumIA kanonik hasta sayısı: {len(lumiere_deep)}")
    print(f"LUMIERE/HD-GLIO-AUTO kanonik hasta sayısı: {len(lumiere_hdglio)}")
    print(f"TCGA/WT kanonik hasta sayısı: {len(tcga_wt)}")

    if lumiere_deep.empty or lumiere_hdglio.empty:
        print("HATA: LUMIERE kanonik verisi bulunamadı, smoke test yürütülemiyor.")
        return 1
    if tcga_wt.empty:
        print("HATA: TCGA kanonik verisi bulunamadı, apply() smoke test yürütülemiyor.")
        return 1

    common_features = sorted(
        set(lumiere_deep.columns) & set(lumiere_hdglio.columns) & set(tcga_wt.columns)
    )
    print(f"Ortak özellik sayısı (isim eşleşmesi): {len(common_features)}")
    if len(common_features) != 107:
        print(
            "UYARI: 107 değil -- kanonik sözleşme kolonları arasında beklenmedik "
            "fark var, dry-run devam ediyor ama gerçek fit için araştırılmalı."
        )

    lumiere_deep = lumiere_deep[common_features]
    lumiere_hdglio = lumiere_hdglio[common_features]
    tcga_wt = tcga_wt[common_features]

    # --- SMOKE TEST fit(): iki gerçek-LUMIERE alt-batch (asla TCGA yok) ---
    #
    # NOT: pipeline.harmonization.fit_combat_harmonization()'ın kendi
    # sızıntı-koruması SADECE covariates['SITE'] içinde tam olarak
    # {'UPenn','LUMIERE'} bekler ('LUMIERE-DeepBraTumIA' gibi türetilmiş
    # bir etiket bile reddedilir -- bu guard'ın DOĞRU çalıştığının kanıtı,
    # aşağıda ayrı bir testte gösteriliyor). UPenn henüz hazır olmadığı
    # için gerçek 2-source (UPenn+LUMIERE) fit testi burada YAPILAMAZ; bu
    # yüzden mekanik/şema testi neuroHarmonize'ı DOĞRUDAN çağırarak (wrapper
    # bypass edilerek) LUMIERE'in kendi 2 segmentasyon-aracı alt-kümesiyle
    # yürütülüyor, ardından `_model_to_combat_parameter_rows()` (wrapper'ın
    # şema-eşleme fonksiyonu) ayrıca gerçek veriyle doğrulanıyor.
    fit_matrix = pd.concat([lumiere_deep, lumiere_hdglio], axis=0)
    renamed_index = [f"{idx}__{i}" for i, idx in enumerate(fit_matrix.index)]
    hdglio_renamed_index = renamed_index[len(lumiere_deep):]
    fit_matrix.index = renamed_index
    covariates = pd.DataFrame(
        {
            "SITE": ["LUMIERE-DeepBraTumIA"] * len(lumiere_deep)
            + ["LUMIERE-HDGLIO"] * len(lumiere_hdglio),
        },
        index=fit_matrix.index,
    )

    data = fit_matrix.to_numpy(dtype=np.float64)
    model, bayes_data = nh.harmonizationLearn(
        data, covariates, ref_batch="LUMIERE-DeepBraTumIA", seed=42
    )
    harmonized_in_sample = pd.DataFrame(
        bayes_data, index=fit_matrix.index, columns=fit_matrix.columns
    )
    parameter_rows = _model_to_combat_parameter_rows(
        model=model,
        feature_names=list(fit_matrix.columns),
        library_tag=f"neuroHarmonize=={nh.__version__ if hasattr(nh, '__version__') else '2.5.1'}",
    )
    print(
        "neuroHarmonize.harmonizationLearn() gercek LUMIERE verisiyle "
        f"(n={len(fit_matrix)}, 2 alt-batch) calisti."
    )
    print(f"_model_to_combat_parameter_rows() satir sayisi: {len(parameter_rows)}")
    expected_rows = len(common_features) * 2  # 2 pseudo-batch
    print(f"Beklenen (107 x 2 batch = {expected_rows}): {'UYUSUYOR' if len(parameter_rows) == expected_rows else 'UYUSMUYOR'}")
    print("Örnek satır:", parameter_rows[0])

    schema_columns = {
        "source_batch", "feature_name", "gamma", "delta", "grand_mean",
        "pooled_variance", "design_matrix_ref", "fit_date",
        "reference_batch", "harmonization_library",
    }
    row_columns = set(parameter_rows[0].keys())
    print(
        "combat_parameters şema kolon eşleşmesi:",
        "TAM UYUMLU" if row_columns == schema_columns else f"UYUMSUZ: {row_columns.symmetric_difference(schema_columns)}",
    )

    # --- TCGA sizinti korumasi testi ---
    bad_covariates = pd.DataFrame(
        {"SITE": ["TCGA-GBM"] * len(tcga_wt)}, index=tcga_wt.index
    )
    try:
        fit_combat_harmonization(tcga_wt, bad_covariates, reference_batch="TCGA-GBM")
        print("HATA: TCGA fit()'e girebildi -- SIZINTI KORUMASI CALISMIYOR!")
        return 1
    except ComBatFitLeakageError as exc:
        print(f"TCGA sizinti korumasi calisti (beklenen hata): {exc}")

    # --- apply() sizinti korumasi testi (2026-08-11, TCGA -- SITE spoof
    # edilmis olsa bile, 2026-08-09 karari geregi TCGA apply()'a HICBIR
    # ZAMAN giremez artik) ---
    # (burada referans-batch = 'LUMIERE-DeepBraTumIA', gercek uretimde
    # referans 'UPenn' olacak -- ayni mekanizma, farkli etiket. SITE
    # kasten referans-batch'e spoof edildi, guard SITE'a degil
    # true_dataset_source'a bakip reddetmeli.)
    tcga_covariates = pd.DataFrame(
        {"SITE": [model["ref_batch"]] * len(tcga_wt)}, index=tcga_wt.index
    )
    tcga_true_source = pd.Series(
        ["TCGA"] * len(tcga_wt), index=tcga_wt.index, name="true_dataset_source"
    )
    try:
        apply_combat_harmonization(
            tcga_wt,
            tcga_covariates,
            model,
            true_dataset_source=tcga_true_source,
        )
        print(
            "HATA: TCGA apply()'a girebildi (SITE spoof edilmisti) -- "
            "APPLY SIZINTI KORUMASI CALISMIYOR!"
        )
        return 1
    except ComBatApplyLeakageError as exc:
        print(f"TCGA apply() sizinti korumasi calisti (beklenen hata): {exc}")

    # --- apply() smoke test: gercekten fit edilmis batch'e yeni satir ---
    hdglio_for_apply = lumiere_hdglio.copy()
    hdglio_for_apply.index = hdglio_renamed_index
    real_apply_covariates = pd.DataFrame(
        {"SITE": ["LUMIERE-HDGLIO"] * len(hdglio_for_apply)}, index=hdglio_for_apply.index
    )
    real_apply_true_source = pd.Series(
        ["LUMIERE"] * len(hdglio_for_apply),
        index=hdglio_for_apply.index,
        name="true_dataset_source",
    )
    real_apply_result = apply_combat_harmonization(
        hdglio_for_apply,
        real_apply_covariates,
        model,
        true_dataset_source=real_apply_true_source,
    )
    matches_in_sample = np.allclose(
        real_apply_result.to_numpy(dtype=np.float64),
        harmonized_in_sample.loc[hdglio_renamed_index].to_numpy(dtype=np.float64),
        atol=1e-6,
    )
    print(
        "Gercek (non-ref) batch out-of-sample apply() == fit()'in in-sample "
        f"ciktisiyla ayni mi (beklenen: EVET): {matches_in_sample}"
    )

    print("\nSMOKE TEST TAMAMLANDI -- hicbir DB yazimi yapilmadi (readonly).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
