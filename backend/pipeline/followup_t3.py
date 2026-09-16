"""T3 Takip Modülü — delta metriklerinin RANO-benzeri kalibrasyonu (Hafta 4, Görev 4).

`raw/mimari/v45.txt` Bölüm 7'nin tarif ettiği "RANO-benzeri volumetrik
yaklaşım"ı uygular: iki ardışık LUMIERE ziyareti arasındaki hacim/yüzey/
şekil/texture delta'larını gerçek RANO etiketleriyle kalibre eder.

## ZORUNLU ETİKET (v45.txt satır 1257-1265, CLAUDE.md'nin ruhuna uygun)

Bu modülün ürettiği sınıflandırma **resmi RANO'nun birebir otomasyonu
DEĞİLDİR** — yalnızca hacimsel/radyomik değişime dayanır. Resmi RANO;
yeni lezyon varlığı, steroid dozu ve klinik durum gibi hacim-dışı unsurları
da içerir, bunlar bu modülde YOKTUR. Çıktı ve raporlarda **"RANO-benzeri
volumetrik yaklaşım"** ifadesi kullanılır, "RANO" tek başına KULLANILMAZ.

## Önemli ayrım (v45.txt satır 1252-1255)

Bu modül SİMÜLASYON çıktısını değil, GERÇEK takip MR'larından elde edilen
radyomik değerleri değerlendirir (`pipeline/growth_simulation.py`'nin
ürettiği projeksiyonlardan TAMAMEN AYRI bir veri kaynağı).

## C32 kapısı

Bu modül `pipeline.growth_simulation.fetch_lumiere_wt_volume_series` ve
`fetch_lumiere_rano_labels`'i kullanır — ikisi de zaten
`LUMIERE-PyRadiomics-107-C32` / `WT_derived` sözleşmesine kilitlidir.

## Ziyaret eşleştirme — DOĞRULANMAMIŞ sezgisel (heuristic) uyarısı

`followup_series.scan_id` LUMIERE'de HER SATIRDA NULL'dır (canlı DB'de
616/616 doğrulandı, 2026-08-18) — yani RANO etiketleri hiçbir taramaya
DOĞRUDAN bağlı değil. Bu modül etiketleri `(patient_id, visit_week)` ile
`mr_scans.timepoint_label`'dan türetilen `(patient_id, week)`'e eşler.
Aynı nominal haftada BİRDEN FAZLA tarama/etiket varsa (haftaların ~%7'si
canlı ölçüldü) eşleştirme sıra-bazlı bir SEZGİSELDİR (bkz.
`_pair_same_week_groups` docstring'i) — DOĞRULANMAMIŞTIR, ölçülen
belirsizlik `PairedVisit.match_type` alanında görünür kalır. Eşleşemeyen
(radyomik-var-RANO-yok / RANO-var-radyomik-yok / aynı-hafta kovasında
fazlalık) satırlar `build_visit_match_report`'ta AYRI bir denetim-izi
artifact'ı olarak raporlanır (Codex MEDIUM5, 2026-08-19) — bu fonksiyon
öncesinde bu satırlar sessizce düşüyordu.

## Etiket normalizasyonu — TEK NOKTA (Codex MEDIUM4, 2026-08-19)

`rano_label` normalizasyonu (boşluk kırpma, `None`/`'None'` -> `UNKNOWN`)
YALNIZ `pipeline.growth_simulation.fetch_lumiere_rano_labels` içinde
yapılır — bu modül bunu TEKRARLAMAZ, ham/kirli etiketle KARŞILAŞTIRMA
YAPMAZ. Bileşik/bilinmeyen etiketler (örn. `'Post-Op/PD'`, Patient-089)
`RANO_RESPONSE_LABELS`'a girmediği için delta hesaplarına GİRMEZ ve
`summarize_excluded_transitions` ile AÇIKÇA raporlanır — PD'ye ya da
başka bir kategoriye SESSİZCE dönüştürülmez.

## CR-WT hacim uyumsuzluğu — GERÇEK VERİ BULGUSU (Codex HIGH3, 2026-08-19)

LUMIERE üzerinde uzman RANO etiketleri ile WT hacim değişimi arasında
güçlü ve monoton bir ilişki GÖZLENMEMİŞTİR. 26 CR etiketli geçişte WT
delta medyanı **-%1,2**; dağılım hem belirgin küçülme hem büyüme
içermiştir (IQR [-%20,1, +%30,4]; Patient-015: -%96,7 sonra +%356,6, iki
ayrı geçiş). Bu, resmi RANO'nun WT hacmi dışında kalan bileşenleriyle
(ölçülebilir kontrast tutan lezyon, 2B çap, yeni lezyon, steroid, klinik
durum) UYUMLUDUR — kod hatası DEĞİLDİR.

⚠️ **Sonuç:** WT-tabanlı eşikler "resmi RANO otomasyonu" olarak
SUNULMAZ; yalnızca "RANO-benzeri volumetrik yaklaşımın yöntem sınırı"
olarak etiketlenir. Mimarinin örnek `<-%65 CR` eşiği (v45.txt Bölüm 7)
bu modülde WT hacminden TÜRETİLMEZ ve TÜRETİLMEMELİDİR — yalnız
KARŞILAŞTIRMA amacıyla `tools/run_followup_t3_calibration.py`'de
`ARCHITECTURE_EXAMPLE_THRESHOLDS` sabiti olarak yan yana yazdırılır,
hiçbir kod yolu bu örneği gerçek bir üretim eşiği olarak KULLANMAZ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from psycopg2.extensions import connection as _PgConnection

from pipeline.growth_simulation import (
    RANO_RESPONSE_LABELS,
    classify_rano_label,
    fetch_lumiere_rano_labels,
    fetch_lumiere_wt_volume_series,
    log_ratio_growth_rate,
)

_PRIORITY = {"Pre-Op": 0, "Post-Op": 1}


def _label_sort_key(label: str, row_order: int) -> tuple[int, int]:
    # NOT: burada .strip() YAPILMAZ -- normalizasyon TEK noktada
    # (`growth_simulation.fetch_lumiere_rano_labels`) zaten yapılmıştır
    # (Codex MEDIUM4, bkz. modül docstring'i "Etiket normalizasyonu").
    return _PRIORITY.get(label, 2), row_order


def _pair_same_week_groups(
    radiomics_group: pd.DataFrame, rano_group: pd.DataFrame
) -> list[tuple[pd.Series, pd.Series, str]]:
    """Aynı (hasta, hafta) kovasındaki N radyomik satırı ile M RANO satırını eşler.

    N==M==1 ise "unique" (kesin). Aksi halde konum-bazlı bir sezgisel
    kullanılır: radyomik satırlar `suffix`'e göre, RANO satırları
    Pre-Op < Post-Op < diğer (orijinal satır sırası) önceliğine göre
    sıralanır ve pozisyonel olarak eşlenir — fazlalık taraf EŞLEŞMEDEN
    KALIR (rapora "eşlenemedi" olarak yansır).
    """

    radiomics_sorted = radiomics_group.sort_values("suffix").reset_index(drop=True)
    rano_sorted = rano_group.reset_index(drop=True)
    rano_sorted["_order"] = range(len(rano_sorted))
    rano_sorted = rano_sorted.sort_values(
        by="_order",
        key=lambda s: s.map(lambda i: _label_sort_key(rano_sorted.loc[i, "rano_label"], i)),
    ).reset_index(drop=True)

    match_type = "unique" if len(radiomics_sorted) == 1 and len(rano_sorted) == 1 else (
        "heuristic_paired"
    )

    n = min(len(radiomics_sorted), len(rano_sorted))
    pairs = []
    for i in range(n):
        pairs.append((radiomics_sorted.iloc[i], rano_sorted.iloc[i], match_type))
    return pairs


def pair_lumiere_visits_with_rano(conn: _PgConnection) -> pd.DataFrame:
    """LUMIERE radyomik ziyaretlerini RANO etiketleriyle eşler (SADECE SELECT).

    Dönen her satır bir (radyomik ölçümü + RANO etiketi) eşleşmesidir.
    `match_type` sütunu "unique" (kesin, tek-tek eşleşme) veya
    "heuristic_paired" (aynı hafta kovasında birden fazla aday olduğu için
    sıra-bazlı sezgiselle eşleştirildi) değerini taşır.
    """

    radiomics_df = fetch_lumiere_wt_volume_series(conn)
    rano_df = fetch_lumiere_rano_labels(conn)

    if radiomics_df.empty or rano_df.empty:
        return pd.DataFrame()

    matched_rows: list[dict] = []
    for (patient_id, week), radiomics_group in radiomics_df.groupby(["patient_id", "week"]):
        rano_group = rano_df[
            (rano_df["patient_id"] == patient_id) & (rano_df["visit_week"] == week)
        ]
        if rano_group.empty:
            continue
        for radiomics_row, rano_row, match_type in _pair_same_week_groups(
            radiomics_group, rano_group
        ):
            matched_rows.append(
                {
                    "patient_id": patient_id,
                    "week": week,
                    "x_week": radiomics_row["x_week"],
                    "suffix": radiomics_row["suffix"],
                    "scan_id": radiomics_row.get("scan_id"),
                    "followup_id": rano_row.get("followup_id"),
                    "tumor_volume_mm3": radiomics_row["tumor_volume_mm3"],
                    "surface_area": radiomics_row["surface_area"],
                    "sphericity": radiomics_row["sphericity"],
                    "entropy": radiomics_row["entropy"],
                    "glcm_joint_entropy": radiomics_row["glcm_joint_entropy"],
                    "rano_label": rano_row["rano_label"],
                    "rano_rationale": rano_row["rano_rationale"],
                    "match_type": match_type,
                }
            )

    result = pd.DataFrame(matched_rows)
    if not result.empty:
        result = result.sort_values(["patient_id", "x_week"]).reset_index(drop=True)
    return result


@dataclass
class DeltaMetrics:
    patient_id: str
    prev_week: float
    curr_week: float
    curr_rano_label: str
    volume_pct: float
    surface_pct: float
    sphericity_delta_pct_points: float
    glcm_entropy_pct: float
    log_ratio_rate_per_week: float


def compute_delta_metrics(prev_row: pd.Series, curr_row: pd.Series) -> DeltaMetrics:
    """İki ardışık ziyaret arasındaki delta metriklerini hesaplar.

    - `volume_pct`/`surface_pct`/`glcm_entropy_pct`: (curr-prev)/prev * 100
    - `sphericity_delta_pct_points`: sferisite [0,1] birimsizdir, YÜZDE
      DEĞİŞİM yerine YÜZDE-PUAN farkı raporlanır (v45.txt Tablo 7'deki
      "Sferisity değişimi ±5%" ifadesiyle birim tutarlılığı için).
    - `log_ratio_rate_per_week`: `pipeline.growth_simulation.log_ratio_growth_rate`
      ile aynı formül (`log(V_curr/V_prev)/(t_curr-t_prev)`) — bu geçişin
      KENDİ yerel büyüme hızı (Codex HIGH1 düzeltmesi, model-fit
      VARSAYMAZ). `volume_pct`'ten FARKLIDIR: zaman aralığını normalize
      eder, bu yüzden farklı uzunluktaki geçişler karşılaştırılabilir hâle
      gelir.
    """

    def _pct(prev: float, curr: float) -> float:
        if prev == 0:
            return float("nan")
        return (curr - prev) / prev * 100.0

    return DeltaMetrics(
        patient_id=curr_row["patient_id"],
        prev_week=float(prev_row["x_week"]),
        curr_week=float(curr_row["x_week"]),
        curr_rano_label=curr_row["rano_label"],
        volume_pct=_pct(prev_row["tumor_volume_mm3"], curr_row["tumor_volume_mm3"]),
        surface_pct=_pct(prev_row["surface_area"], curr_row["surface_area"]),
        sphericity_delta_pct_points=(curr_row["sphericity"] - prev_row["sphericity"]) * 100.0,
        glcm_entropy_pct=_pct(prev_row["glcm_joint_entropy"], curr_row["glcm_joint_entropy"]),
        log_ratio_rate_per_week=log_ratio_growth_rate(
            float(prev_row["tumor_volume_mm3"]),
            float(curr_row["tumor_volume_mm3"]),
            float(prev_row["x_week"]),
            float(curr_row["x_week"]),
        ),
    )


def build_patient_visit_deltas(paired_df: pd.DataFrame) -> pd.DataFrame:
    """Her hasta için ardışık ziyaretler arasındaki delta'ları çıkarır.

    Yalnız CURRENT (ikinci) ziyaretin RANO etiketi PD/SD/PR/CR (gerçek
    yanıt kategorisi) olan geçişler tutulur — Pre-Op/Post-Op birer cerrahi
    zaman noktasıdır, RANO yanıt sınıfı DEĞİLDİR (bkz. modül docstring'i).
    """

    rows = []
    for patient_id, group in paired_df.groupby("patient_id"):
        group = group.sort_values("x_week").reset_index(drop=True)
        for i in range(1, len(group)):
            prev_row = group.iloc[i - 1]
            curr_row = group.iloc[i]
            if curr_row["rano_label"] not in RANO_RESPONSE_LABELS:
                continue
            delta = compute_delta_metrics(prev_row, curr_row)
            rows.append(delta.__dict__)
    return pd.DataFrame(rows)


def summarize_excluded_transitions(paired_df: pd.DataFrame) -> pd.DataFrame:
    """`build_patient_visit_deltas`'ın SESSİZCE atladığı geçişleri raporlar (Codex MEDIUM4).

    İki kategori:
    - `"surgical_timepoint_not_rano_response"`: `curr_rano_label` Pre-Op/
      Post-Op (cerrahi zaman noktası, RANO yanıtı DEĞİL) — BEKLENEN ve
      istenen dışlama, bilgi amaçlı raporlanır.
    - `"unknown_or_compound_label"`: `curr_rano_label` ne yanıt ne cerrahi
      kategorisine giriyor (örn. `'Post-Op/PD'` gibi bileşik/bilinmeyen bir
      etiket, ya da `UNKNOWN_RANO_LABEL`) — bu satırlar PD'ye ya da başka
      bir kategoriye SESSİZCE dönüştürülMEZ, burada AÇIKÇA işaretlenir.
    """

    rows = []
    for patient_id, group in paired_df.groupby("patient_id"):
        group = group.sort_values("x_week").reset_index(drop=True)
        for i in range(1, len(group)):
            curr_row = group.iloc[i]
            label = curr_row["rano_label"]
            if label in RANO_RESPONSE_LABELS:
                continue
            category = classify_rano_label(label)
            reason = (
                "surgical_timepoint_not_rano_response"
                if category == "non_response"
                else "unknown_or_compound_label"
            )
            rows.append(
                {
                    "patient_id": patient_id,
                    "prev_week": float(group.iloc[i - 1]["x_week"]),
                    "curr_week": float(curr_row["x_week"]),
                    "curr_rano_label": label,
                    "exclusion_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def build_delta_rano_table(deltas_df: pd.DataFrame, *, min_group_n: int = 5) -> pd.DataFrame:
    """RANO grubu bazında delta metriklerinin medyan ± IQR eşik tablosunu çıkarır.

    v45.txt Tablo 7'nin örnek eşiklerini (hacim <-65% CR, yüzey <-30% PR,
    sferisite ±5% SD, GLCM entropi >+25% PD) DOĞRULAMAK için üretilir —
    örnek eşikler VARSAYILMAZ, gerçek LUMIERE verisinden YENİDEN ÖLÇÜLÜR.
    """

    metrics = [
        "volume_pct",
        "surface_pct",
        "sphericity_delta_pct_points",
        "glcm_entropy_pct",
        "log_ratio_rate_per_week",
    ]
    rows = []
    for rano_label, group in deltas_df.groupby("curr_rano_label"):
        n = len(group)
        row = {"rano_label": rano_label, "n": n, "interpretable": n >= min_group_n}
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                row[f"{metric}_median"] = float("nan")
                row[f"{metric}_iqr_low"] = float("nan")
                row[f"{metric}_iqr_high"] = float("nan")
                continue
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_iqr_low"] = float(np.percentile(values, 25))
            row[f"{metric}_iqr_high"] = float(np.percentile(values, 75))
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        order = [lbl for lbl in RANO_RESPONSE_LABELS if lbl in result["rano_label"].values]
        result["rano_label"] = pd.Categorical(result["rano_label"], categories=order, ordered=True)
        result = result.sort_values("rano_label").reset_index(drop=True)
    return result


# --- Eşleştirme denetim izi (matched + unmatched), Codex MEDIUM5 -----------

_MATCH_REPORT_COLUMNS = [
    "patient_id",
    "week",
    "scan_id",
    "followup_id",
    "rano_label",
    "match_type",
    "unmatch_reason",
]


def _visit_match_report_from_frames(
    radiomics_df: pd.DataFrame, rano_df: pd.DataFrame
) -> pd.DataFrame:
    """`build_visit_match_report`'ın SAF (DB'siz, test edilebilir) çekirdeği.

    `pair_lumiere_visits_with_rano`'nun aksine yalnız BAŞARILI eşleşmeleri
    değil, eşleşemeyen radyomik/RANO satırlarını da nedenleriyle döner —
    önceki davranışta bu satırlar `rano_group.empty` kontrolünde ya da
    `_pair_same_week_groups`'ın `n = min(...)` kırpmasında SESSİZCE
    düşüyordu.

    `unmatch_reason` değerleri:
    - `"no_rano_label_at_week"`: o (hasta, hafta)'da radyomik VAR, RANO
      etiketi YOK.
    - `"no_radiomics_at_week"`: o (hasta, hafta)'da RANO etiketi VAR,
      radyomik satırı YOK.
    - `"excess_radiomics_in_same_week_group"` / `"excess_rano_in_same_week_group"`:
      aynı-hafta kovasında N!=M olduğu için `_pair_same_week_groups`'ın
      pozisyonel eşleştirmesinin dışında kalan fazlalık taraf.
    """

    if radiomics_df.empty and rano_df.empty:
        return pd.DataFrame(columns=_MATCH_REPORT_COLUMNS)

    rows: list[dict] = []
    radiomics_keys = (
        set(zip(radiomics_df["patient_id"], radiomics_df["week"]))
        if not radiomics_df.empty
        else set()
    )
    rano_keys = (
        set(zip(rano_df["patient_id"], rano_df["visit_week"])) if not rano_df.empty else set()
    )

    for patient_id, week in sorted(radiomics_keys | rano_keys):
        radiomics_group = (
            radiomics_df[
                (radiomics_df["patient_id"] == patient_id) & (radiomics_df["week"] == week)
            ]
            if not radiomics_df.empty
            else radiomics_df
        )
        rano_group = (
            rano_df[(rano_df["patient_id"] == patient_id) & (rano_df["visit_week"] == week)]
            if not rano_df.empty
            else rano_df
        )

        if radiomics_group.empty:
            for _, rano_row in rano_group.iterrows():
                rows.append(
                    {
                        "patient_id": patient_id,
                        "week": week,
                        "scan_id": None,
                        "followup_id": rano_row.get("followup_id"),
                        "rano_label": rano_row["rano_label"],
                        "match_type": "unmatched",
                        "unmatch_reason": "no_radiomics_at_week",
                    }
                )
            continue
        if rano_group.empty:
            for _, radiomics_row in radiomics_group.iterrows():
                rows.append(
                    {
                        "patient_id": patient_id,
                        "week": week,
                        "scan_id": radiomics_row.get("scan_id"),
                        "followup_id": None,
                        "rano_label": None,
                        "match_type": "unmatched",
                        "unmatch_reason": "no_rano_label_at_week",
                    }
                )
            continue

        pairs = _pair_same_week_groups(radiomics_group, rano_group)
        matched_scan_ids = set()
        matched_followup_ids = set()
        for radiomics_row, rano_row, match_type in pairs:
            rows.append(
                {
                    "patient_id": patient_id,
                    "week": week,
                    "scan_id": radiomics_row.get("scan_id"),
                    "followup_id": rano_row.get("followup_id"),
                    "rano_label": rano_row["rano_label"],
                    "match_type": match_type,
                    "unmatch_reason": None,
                }
            )
            matched_scan_ids.add(radiomics_row.get("scan_id"))
            matched_followup_ids.add(rano_row.get("followup_id"))

        for _, radiomics_row in radiomics_group.iterrows():
            if radiomics_row.get("scan_id") not in matched_scan_ids:
                rows.append(
                    {
                        "patient_id": patient_id,
                        "week": week,
                        "scan_id": radiomics_row.get("scan_id"),
                        "followup_id": None,
                        "rano_label": None,
                        "match_type": "unmatched",
                        "unmatch_reason": "excess_radiomics_in_same_week_group",
                    }
                )
        for _, rano_row in rano_group.iterrows():
            if rano_row.get("followup_id") not in matched_followup_ids:
                rows.append(
                    {
                        "patient_id": patient_id,
                        "week": week,
                        "scan_id": None,
                        "followup_id": rano_row.get("followup_id"),
                        "rano_label": rano_row["rano_label"],
                        "match_type": "unmatched",
                        "unmatch_reason": "excess_rano_in_same_week_group",
                    }
                )

    result = pd.DataFrame(rows, columns=_MATCH_REPORT_COLUMNS)
    if not result.empty:
        result = result.sort_values(["patient_id", "week"]).reset_index(drop=True)
    return result


def build_visit_match_report(conn: _PgConnection) -> pd.DataFrame:
    """`pair_lumiere_visits_with_rano`'nun TAM denetim izi (SADECE SELECT).

    Yalnız başarılı eşleşmeleri değil, eşleşemeyen HER radyomik/RANO
    satırını nedeniyle birlikte döner (Codex MEDIUM5). `scan_id` +
    `followup_id` (kaynak satır kimlikleri) + `match_type` +
    `unmatch_reason` sütunlarını taşır — `heuristic_paired` sonuçlar bu
    raporda `match_type` sütunundan filtrelenerek ayrı bir duyarlılık kolu
    olarak incelenebilir.
    """

    radiomics_df = fetch_lumiere_wt_volume_series(conn)
    rano_df = fetch_lumiere_rano_labels(conn)
    return _visit_match_report_from_frames(radiomics_df, rano_df)
