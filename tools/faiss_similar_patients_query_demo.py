"""`clinical_radiomics_faiss` benzer-hasta sorgusu -- doğrulama demosu.

GÖREV BAĞLAMI (2026-08-18, Barış kararı, K1/K2 kilitlendikten sonra):
Hafta 3'ün rag-agent'a ait açık kalemi -- "TCGA-02-0003 ve TCGA-02-0006
için benzer-hasta sorgusu... klinik seyirle mantıksal tutarlılığı
kontrol et." Bu script `tools/rebuild_faiss_indexes.py --no-combat --apply`
ile ÜRETİLMİŞ, DİSKTEKİ v1 indeksini (`artifacts/week3/faiss_index/`)
okur -- kendi başına DB'ye BAĞLANMAZ, hiçbir yeni sorgu/skor üretmez,
YALNIZ mevcut artifact'leri okuyup top-k komşuları yazdırır.

BU BİR PRODÜKSİYON API DEĞİL -- Sultan'ın `GET /patient/{id}/similar`
endpoint'i (plan.txt satır 452) AYRI, bu görevin kapsamı DIŞINDA. Bu
script yalnız DOĞRULAMA/GÖZLE-KONTROL amaçlı, tek seferlik bir demo.

Kullanım:
    python tools/faiss_similar_patients_query_demo.py \\
        --index-dir artifacts/week3/faiss_index \\
        --patient-id TCGA-02-0003 --patient-id TCGA-02-0006 --k 10

Klinik tutarlılık notu: vektör mesafesi SALT radyomik (WT×93,
`decisions/2026-08-14-faiss-v1-vektor-icerigi.md` kilit kararı) --
yaş/KPS/MGMT/IDH1/sağkalım hiçbir zaman mesafeye girmiyor. Bu script
onları yalnız GÖRÜNTÜLEME için `clinical_metadata.csv`'den okur --
"komşular klinik olarak da mantıklı mı" sorusu bir GÖZLE-KONTROL, bir
kabul KRİTERİ değil (K1 taslağı §3.4 "klinik tutarlılık, GATE DEĞİL,
yalnız yüz-geçerliliği" ilkesiyle aynı).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


class SimilarPatientQueryError(RuntimeError):
    """İndeks artifact'leri okunamadı/tutarsız -- sessizce boş sonuç
    ÜRETİLMEZ (CLAUDE.md: 'FAISS sonucu boşsa sessizce boş dönme')."""


def load_index_artifacts(index_dir: Path, *, index_filename: str = "clinical_radiomics.index"):
    import faiss

    index_path = index_dir / index_filename
    order_path = index_dir / "patient_order.csv"
    clinical_path = index_dir / "clinical_metadata.csv"

    for path in (index_path, order_path, clinical_path):
        if not path.is_file():
            raise SimilarPatientQueryError(f"Beklenen artifact bulunamadı: {path}")

    index = faiss.read_index(str(index_path))
    order = pd.read_csv(order_path, index_col="patient_id")
    clinical = pd.read_csv(clinical_path, index_col="patient_id")

    if index.ntotal != len(order):
        raise SimilarPatientQueryError(
            f"İndeks ({index.ntotal}) ve patient_order.csv ({len(order)}) "
            "satır sayıları UYUŞMUYOR -- artifact'ler tutarsız olabilir."
        )
    return index, order, clinical


def query_similar_patients(
    *, index, order: pd.DataFrame, clinical: pd.DataFrame, patient_id: str, k: int = 10
) -> pd.DataFrame:
    """`patient_id` için top-`k` benzer hastayı (KENDİSİ HARİÇ) döndürür.

    Boş/eksik durumda sessizce boş DataFrame DÖNMEZ -- `patient_id`
    indekste yoksa AÇIK bir `SimilarPatientQueryError` fırlatır (CLAUDE.md
    "benzer hasta bulunamadı" kuralının bu demo'daki karşılığı).
    """

    if patient_id not in order.index:
        raise SimilarPatientQueryError(
            f"{patient_id!r} v1 indeksinde YOK (K1: yalnız UPenn 611 + "
            "LUMIERE 72 + TCGA 39 = 722 hasta indekste; UCSF DAHİL "
            "DEĞİL) -- açık durum: 'bu hasta için benzer-hasta indeksi "
            "yok', sessizce boş sonuç DÖNÜLMEDİ."
        )

    row_index = int(order.loc[patient_id, "row_index"])
    query_vector = index.reconstruct(row_index).reshape(1, -1).astype(np.float32)

    # k+1 çek -- ilk sonuç kendisi olacak (mesafe 0), sonra hariç tutulur.
    distances, indices = index.search(query_vector, k + 1)

    order_by_pos = order.reset_index()  # row_index -> patient_id eşlemesi için
    rows = []
    for dist, idx in zip(distances.ravel(), indices.ravel()):
        if idx == -1:
            continue
        neighbor_id = order_by_pos.loc[idx, "patient_id"]
        if neighbor_id == patient_id:
            continue  # kendisi -- top-k'dan HARİÇ (proje konvansiyonu)
        rows.append({"neighbor_patient_id": neighbor_id, "l2_distance": float(dist)})
        if len(rows) == k:
            break

    if not rows:
        raise SimilarPatientQueryError(
            f"{patient_id!r} için indekste kendisi hariç HİÇBİR komşu "
            "bulunamadı -- açık 'benzer hasta bulunamadı' durumu "
            "(sessiz boş dönme YASAK)."
        )

    result = pd.DataFrame(rows).set_index("neighbor_patient_id")
    result = result.join(clinical, how="left")
    return result


def _print_query_result(patient_id: str, clinical: pd.DataFrame, result: pd.DataFrame) -> None:
    print("=" * 78)
    print(f"SORGU HASTASI: {patient_id}")
    if patient_id in clinical.index:
        query_row = clinical.loc[patient_id]
        print(
            f"  kaynak={query_row.get('source')} yaş={query_row.get('age')} "
            f"cinsiyet={query_row.get('gender')} KPS={query_row.get('kps_score')} "
            f"MGMT={query_row.get('mgmt_status')} IDH1={query_row.get('idh1_status')} "
            f"vital_status={query_row.get('vital_status')} "
            f"survival_days={query_row.get('survival_days')}"
        )
    print("-" * 78)
    print(f"TOP-{len(result)} BENZER HASTA (kendisi hariç, L2 mesafeye göre artan):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(result.to_string())
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "week3" / "faiss_index",
        help="`tools/rebuild_faiss_indexes.py --apply` çıktı dizini.",
    )
    parser.add_argument(
        "--patient-id",
        action="append",
        required=True,
        dest="patient_ids",
        help="Sorgulanacak hasta ID'si (birden fazla kez verilebilir).",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k komşu sayısı (varsayılan 10).")
    args = parser.parse_args(argv)

    index, order, clinical = load_index_artifacts(args.index_dir)
    print(f"İndeks yüklendi: {index.ntotal} hasta, {index.d} boyut, dizin={args.index_dir}")

    for patient_id in args.patient_ids:
        result = query_similar_patients(
            index=index, order=order, clinical=clinical, patient_id=patient_id, k=args.k
        )
        _print_query_result(patient_id, clinical, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
