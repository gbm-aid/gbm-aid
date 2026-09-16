"""LUMIERE-kalibrasyonlu tümör büyümesi simülasyonu (Hafta 4, Görev 1-3).

Bu modül `raw/mimari/v45.txt` Bölüm 5.4'ün tarif ettiği akışı uygular:

1. LUMIERE'nin gerçek longitudinal WT (whole-tumor) hacim ölçümlerinden
   Gompertz VE lojistik büyüme eğrileri fit edilir (`scipy.optimize.curve_fit`).
2. Fit edilen büyüme katsayısı (r), RANO grubuna göre (PD/SD/PR/CR) medyan
   ± IQR olarak özetlenir.
3. Sınır-temelli (boundary-based) morfolojik büyüme/küçülme fonksiyonu —
   **uniform scaling DEĞİL** — mesafe-dönüşümü (distance transform)
   kullanarak maskenin yüzeyini normal yönünde ilerletir/geriletir.

## KİLİTLİ KURALLAR (CLAUDE.md, 2026-08-13 C32 kararı)

- Radyomik girdi **YALNIZ** `segmentation_tool = 'LUMIERE-PyRadiomics-107-C32'`
  olmalı. Eski nesil adlar (`LUMIERE-PyRadiomics-107`, `DeepBraTumIA`,
  `HD-GLIO-AUTO`) bu modülde KULLANILMAZ — `SEGMENTATION_TOOL` sabiti tek
  kaynak, başka bir yerde tekrar yazılmamalı.
- `tumor_region = 'WT_derived'` (whole-tumor) kullanılır — DeepBraTumIA
  native ana kaynak kararına uygun (bkz.
  `decisions/2026-08-07-lumiere-ana-maske-karari.md`).
- Kullanılan özellikler (tumor_volume_mm3, surface_area, entropy [first-
  order], sphericity [shape]) ICC>=0,60 stabilite filtresinden GEÇEN
  kümededir (bkz. `decisions/2026-08-13-pyradiomics-c32-bincount-karari.md`
  "Açık riskler" §1) — first-order mutlak-ölçek özellikleri (Energy, Mean,
  vb.) BİLİNÇLİ OLARAK kullanılmıyor.

## Bilinmeyen/varsayılan noktalar (raporda açıkça beyan edilir)

- `mr_scans.timepoint_label` biçimi `week-<NNN>` veya `week-<NNN>-<K>`
  (aynı nominal haftada birden fazla tarama — örn. Patient-001'in
  week-000-1/week-000-2'si). Gerçek tarama TARİHİ (`mr_scans.scan_date`)
  LUMIERE için DB'de HİÇ dolu değil (canlı ölçüldü, 2026-08-18) — bu yüzden
  `-K` soneki **varsayılan olarak kronolojik sıra** kabul edilir (dataset'in
  kendi klasör adlandırması), bu DOĞRULANMAMIŞ bir varsayımdır. Aynı
  nominal haftadaki yinelenen taramalar arasında sıralamayı korumak için
  küçük bir epsilon ofseti eklenir (`_SAME_WEEK_EPSILON`); bu, büyüme
  eğrisi fit'inde haftalık ölçeği anlamlı şekilde BOZMAZ.

## Codex çapraz-inceleme düzeltmeleri (task-mszut9wy-qx8sal, 2026-08-19)

**HIGH1 — büyüme eğrisi TEK RANO etiketiyle ilişkilendirme sorunu:**
`fit_patient_growth_curves` TÜM zaman serisine TEK bir `r` (Gompertz/
lojistik) fit eder; `assign_patient_rano_group` ise hastaya yalnız EN SON
kaydedilen yanıt-kategorisi etiketini atar. `growth_rate_by_rano_group`
bu ikisini `patient_id` üzerinden birleştirdiğinde, uzun süre SD/PR seyreden
ama son vizitte PD'ye dönen bir hastanın TÜM geçmişi (haftalarca süren
SD/PR dönemi dahil) PD grubuna yazılır — bu YANLIŞ zaman kapsamı eşlemesidir
ve olduğu gibi (aşağıdaki fonksiyonlarla BİRLİKTE, ONLARIN YERİNE değil)
KORUNUR, çünkü "hastanın nihai/karakteristik büyüme hızı" sorusuna cevap
vermeye çalışır (tek-hasta-tek-değer). Bunun YANINDA, HER RANO etiketini
KENDİ geçişine bağlayan bir YEREL ölçü eklendi:
`compute_transition_growth_rates` — haftaya normalize log-oran
`log(V_curr/V_prev)/(t_curr-t_prev)`, model varsayımı YAPMAZ (curve_fit
yok), yalnız iki ardışık ölçüm arasındaki geçişi tanımlar. RANO-grubu
bazında bu YEREL ölçünün dağılımı `pipeline.followup_t3.build_delta_rano_table`
çıktısındaki `log_ratio_rate_per_week_*` sütunlarında raporlanır (o modül
zaten her geçişi KENDİ `curr_rano_label`'ına bağlıyor — bkz. o modülün
docstring'i). **Rapor kullanıcısı için kural:** RANO-grubu karakteristik
büyüme hızı sorusu için `growth_rate_by_rano_group` (tüm-seri fit, TEK
son-etiket varsayımı, BİLİNEN SINIRLAMA) yerine `followup_t3`'ün geçiş-
bazlı `log_ratio_rate_per_week` tablosu TERCİH EDİLİR; ilki şeffaflık için
SİLİNMEDİ, ikincisiyle YAN YANA raporlanır.

**2026-09-11 Barış kararı — `interpretable` kuralı n<5'ten n<6'ya VE bir
saçılım koşuluna genişletildi:** `growth_rate_by_rano_group_v2.csv`'de CR
(n=1) ve PR (n=3) `interpretable=False` işaretliyken SD (n=5) eski eşikle
(`n >= 5`) tam sınırda `True` kalıyordu — oysa SD'nin `best_r` dağılımı
(medyan≈-1,05, IQR≈0,98) PD'nin (n=67, IQR≈0,056) dağılımının **~18 katı**
saçılımlıydı; "yorumlanabilir" etiketi n eşiğini geçmiş ama istatistiksel
olarak GÜVENİLMEZ bir grubu örtüyordu. Barış kararı: *"yanlış değerler ve
işlemler istemiyorum"* — SD `interpretable=False` olmalı. İki koşul birden
uygulanır (`_DEFAULT_MIN_GROUP_N=6`, `_DEFAULT_MAX_IQR_RATIO_TO_REFERENCE=5.0`,
bkz. bu modülün "RANO grubu bazında büyüme katsayısı dağılımı" bölümü):
(a) küçük-n eşiği bir birim yükseltildi (`n < 6` -> yorumlanamaz) — SD (n=5)
bu koşulla TEK BAŞINA zaten dışarı düşer; (b) EK OLARAK bir saçılım koşulu
eklendi (grup IQR'ı, EN BÜYÜK n'li grubun -- "referans" -- IQR'ının
`max_iqr_ratio_to_reference` katını (varsayılan 5x) aşarsa `interpretable
=False`) — bu, n eşiğini geçebilecek ama aşırı dağınık kalan bir grubu da
yakalar (SD örneğinde 18x/5x oranı fazlasıyla geçiyor). (b) opsiyoneldir
(`max_iqr_ratio_to_reference=None` ile kapatılabilir), varsayılan AÇIK.
Etkilenen çıktı: yalnız `growth_rate_by_rano_group[_and_model]` fonksiyon
çıktıları ve bunları tüketen `tools/run_growth_simulation_lumiere*.py`
CSV'leri. **`pipeline/followup_t3.py::build_delta_rano_table`'ın KENDİ
(bağımsız) `min_group_n=5` eşiği ve SD n=87 delta tablosu bu değişiklikten
ETKİLENMEZ** — o modül ayrı bir veri kaynağı (geçiş-bazlı deltalar) ve ayrı
bir örneklem büyüklüğü kullanıyor, bilinçli olarak dokunulmadı.

**MEDIUM4 — etiket normalizasyonu merkezileştirildi:** `fetch_lumiere_rano_labels`
artık `rano_label`'ı TEK noktada `normalize_rano_label` ile temizler
(baştaki/sondaki boşluk kırpılır, `None`/`'None'` -> `UNKNOWN_RANO_LABEL`);
ham değer `rano_label_raw` sütununda denetim izi olarak KORUNUR. Bilinen
kirli değerler (`'Post-Op '`, `'Post-Op/PD'`, literal `'None'` metni) canlı
DB'de doğrulandı (2026-08-19, 6 satır: Patient-026/028/083/089/091).
`'Post-Op/PD'` gibi bileşik/bilinmeyen etiketler PD'ye ya da başka bir
kategoriye SESSİZCE DÖNÜŞTÜRÜLMEZ — `classify_rano_label` bunları ayrı bir
`"unknown"` kovasına düşürür, `summarize_unknown_rano_labels` bu satırları
AÇIKÇA raporlar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from psycopg2.extensions import connection as _PgConnection
from scipy import ndimage
from scipy.optimize import curve_fit

# --- Kilitli sözleşme sabitleri (C32) ----------------------------------

SEGMENTATION_TOOL: str = "LUMIERE-PyRadiomics-107-C32"
TUMOR_REGION: str = "WT_derived"

# RANO yanıt kategorileri (Pre-Op/Post-Op birer cerrahi zaman noktasıdır,
# RANO yanıt sınıfı DEĞİLDİR — büyüme-katsayısı/RANO grup analizinde hariç
# tutulur, bkz. `growth_rate_by_rano_group`).
RANO_RESPONSE_LABELS: tuple[str, ...] = ("PD", "SD", "PR", "CR")
RANO_NON_RESPONSE_LABELS: tuple[str, ...] = ("Pre-Op", "Post-Op")

# Kayıp/tanınmayan RANO etiketi için kanonik yer tutucu (bkz. `normalize_rano_label`).
UNKNOWN_RANO_LABEL: str = "UNKNOWN"

# --- Yorumlanabilirlik eşikleri (2026-09-11 Barış kararı) ------------------
#
# Eskiden tek koşul vardı: n >= 5. SD grubu (n=5) bu eşiği tam sınırda
# geçip `interpretable=True` oluyordu, ama dağılımı (IQR≈0,98) PD grubunun
# (n=67, IQR≈0,056) ~18 katı saçılımlıydı -- "yorumlanabilir" etiketi
# yanıltıcıydı. Artık İKİ koşul birden aranır (ikisi de sağlanmalı):
#   (a) n >= _DEFAULT_MIN_GROUP_N (5 -> 6 yükseltildi, SD'yi tek başına dışarı alır)
#   (b) grup IQR'ı, o çağrıdaki EN BÜYÜK n'li grubun ("referans") IQR'ının
#       _DEFAULT_MAX_IQR_RATIO_TO_REFERENCE katını AŞMAMALI
# (b) bilinçli olarak gevşek (5x) seçildi: SD/PD oranı (~18x) bunu fazlasıyla
# geçiyor, ama iki katına çıkan sıradan bir örnekleme varyansını cezalandırmaz.
# Ayrıntı: modül docstring'i "2026-09-11 Barış kararı" bölümü,
# `decisions/2026-09-11-sd-n5-interpretable-false.md`.
_DEFAULT_MIN_GROUP_N: int = 6
_DEFAULT_MAX_IQR_RATIO_TO_REFERENCE: float = 5.0

_WEEK_LABEL_RE = re.compile(r"^week-(\d+)(?:-(\d+))?$")
_SAME_WEEK_EPSILON = 0.05  # hafta biriminde küçük ofset, yalnız sıralama için


# --- RANO etiket normalizasyonu (TEK nokta, Codex MEDIUM4) ----------------


def normalize_rano_label(raw_label: str | None) -> str:
    """RANO etiketini TEK noktada normalize eder.

    - `None` (gerçek SQL NULL) veya literal `"None"` metni (canlı DB'de
      2 satırda bulunan bir veri-girişi hatası, 2026-08-19 doğrulandı) ->
      `UNKNOWN_RANO_LABEL`.
    - Baştaki/sondaki boşluk KIRPILIR (`'Post-Op '` -> `'Post-Op'`, canlı
      DB'de 3 satır: Patient-028/083/091).
    - Bileşik/bilinmeyen değerler (örn. `'Post-Op/PD'`, Patient-089)
      OLDUĞU GİBİ bırakılır — PD'ye ya da başka bir kategoriye SESSİZCE
      DÖNÜŞTÜRÜLMEZ. `classify_rano_label` bu değeri ayrı bir "unknown"
      kovasına düşürür.

    Bu fonksiyon `fetch_lumiere_rano_labels` içinde DB okumasından hemen
    sonra ÇAĞRILIR — başka hiçbir yerde tekrar `.strip()` YAPILMAMALIDIR
    (tek-kaynak-doğruluk ilkesi).
    """

    if raw_label is None:
        return UNKNOWN_RANO_LABEL
    cleaned = raw_label.strip()
    if cleaned == "" or cleaned.lower() == "none":
        return UNKNOWN_RANO_LABEL
    return cleaned


def classify_rano_label(label: str) -> Literal["response", "non_response", "unknown"]:
    """Normalize edilmiş bir RANO etiketini üç kategoriden birine ayırır.

    - `"response"`: PD/SD/PR/CR (gerçek RANO yanıt kategorisi).
    - `"non_response"`: Pre-Op/Post-Op (cerrahi zaman noktası, yanıt DEĞİL).
    - `"unknown"`: ne yanıt ne cerrahi kategorisine giren HER ŞEY (boş,
      `UNKNOWN_RANO_LABEL`, bileşik `'Post-Op/PD'` gibi) — bu bir HATA
      kovası değil, AÇIKÇA raporlanması gereken bir kategoridir.
    """

    if label in RANO_RESPONSE_LABELS:
        return "response"
    if label in RANO_NON_RESPONSE_LABELS:
        return "non_response"
    return "unknown"


# --- Zaman etiketi ayrıştırma --------------------------------------------


def parse_timepoint_week(label: str) -> tuple[int, int]:
    """``mr_scans.timepoint_label``'ı (hafta, sonek) çiftine ayrıştırır.

    Örnekler: ``"week-044"`` -> ``(44, 0)``; ``"week-000-2"`` -> ``(0, 2)``.
    Sonek yoksa ``0`` döner (tekil tarama).

    Biçim tanınmazsa ``ValueError`` fırlatır — sessiz fallback YOK
    (CLAUDE.md kısıt #2'nin ruhuna uygun: geometri/veri sözleşmesi
    uyuşmazlığında açık hata).
    """

    match = _WEEK_LABEL_RE.match(label.strip())
    if not match:
        raise ValueError(
            f"Tanınmayan timepoint_label biçimi: {label!r} "
            "(beklenen 'week-NNN' veya 'week-NNN-K')"
        )
    week = int(match.group(1))
    suffix = int(match.group(2)) if match.group(2) else 0
    return week, suffix


def _x_week(week: int, suffix: int) -> float:
    """Aynı nominal haftadaki yinelenen taramalar için sıralı x-değeri.

    Gerçek tarih olmadığı için bu bir YAKLAŞIKLIKTIR (bkz. modül
    docstring'i) — yalnız aynı haftadaki noktaları ayırt etmek/sıralamak
    için kullanılır, gerçek zaman aralığını temsil ETMEZ.
    """

    if suffix <= 1:
        return float(week)
    return float(week) + (suffix - 1) * _SAME_WEEK_EPSILON


# --- DB'den veri çekme (SALT-OKUNUR) -------------------------------------


def fetch_lumiere_wt_volume_series(conn: _PgConnection) -> pd.DataFrame:
    """LUMIERE WT hacim serisini (C32) DB'den okur.

    Dönen sütunlar: patient_id, scan_id, timepoint_label, week, suffix,
    x_week, tumor_volume_mm3, surface_area, sphericity, entropy,
    glcm_joint_entropy.

    SADECE ``SELECT`` — hiçbir yazma yapmaz.
    """

    query = """
        SELECT
            m.patient_id,
            r.scan_id,
            m.timepoint_label,
            r.tumor_volume_mm3,
            r.surface_area,
            r.entropy,
            r.shape_features -> 'original_shape_Sphericity' AS sphericity,
            r.texture_features -> 'original_glcm_JointEntropy' AS glcm_joint_entropy
        FROM radiomics r
        JOIN mr_scans m ON m.scan_id = r.scan_id
        WHERE r.segmentation_tool = %s
          AND r.tumor_region = %s
          AND m.patient_id ILIKE 'Patient-%%'
        ORDER BY m.patient_id, m.timepoint_label
    """
    with conn.cursor() as cur:
        cur.execute(query, (SEGMENTATION_TOOL, TUMOR_REGION))
        columns = [desc.name for desc in cur.description]
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df

    parsed = df["timepoint_label"].apply(parse_timepoint_week)
    df["week"] = parsed.apply(lambda t: t[0])
    df["suffix"] = parsed.apply(lambda t: t[1])
    df["x_week"] = df.apply(lambda row: _x_week(row["week"], row["suffix"]), axis=1)
    for col in ("tumor_volume_mm3", "surface_area", "entropy", "sphericity", "glcm_joint_entropy"):
        df[col] = df[col].astype(float)
    df = df.sort_values(["patient_id", "x_week"]).reset_index(drop=True)
    return df


def fetch_lumiere_rano_labels(conn: _PgConnection) -> pd.DataFrame:
    """LUMIERE RANO etiketlerini `followup_series`'ten okur (SADECE SELECT).

    Dönen sütunlar: patient_id, visit_week, rano_label (NORMALİZE EDİLMİŞ —
    bkz. `normalize_rano_label`), rano_label_raw (DB'deki HAM değer, denetim
    izi), rano_rationale, followup_id (kaynak satır kimliği, PK).
    """

    query = """
        SELECT
            f.followup_id,
            f.patient_id,
            f.visit_week,
            f.rano_label,
            f.rano_rationale
        FROM followup_series f
        JOIN dataset_sources ds ON ds.source_id = f.source_id
        WHERE ds.source_name = 'LUMIERE'
        ORDER BY f.patient_id, f.visit_week
    """
    with conn.cursor() as cur:
        cur.execute(query)
        columns = [desc.name for desc in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        df["rano_label_raw"] = df.get("rano_label")
        return df

    df["rano_label_raw"] = df["rano_label"]
    df["rano_label"] = df["rano_label"].apply(normalize_rano_label)
    return df


def summarize_unknown_rano_labels(rano_df: pd.DataFrame) -> pd.DataFrame:
    """`classify_rano_label`'a göre `"unknown"` kategorisine düşen satırları döner.

    Bu satırlar hiçbir alt-sistem tarafından SESSİZCE atılmamalı; bu
    fonksiyon çağıranın (tool script/rapor) onları AÇIKÇA yüzeye
    çıkarmasını sağlar (Codex MEDIUM4 — 'Post-Op/PD' örneği). Girdi
    `rano_df`, `fetch_lumiere_rano_labels`'in döndürdüğü (normalize
    edilmiş `rano_label` + ham `rano_label_raw` sütunlu) DataFrame'dir.
    """

    if rano_df.empty:
        return rano_df.iloc[0:0]
    working = rano_df.copy()
    working["rano_category"] = working["rano_label"].apply(classify_rano_label)
    unknown = working[working["rano_category"] == "unknown"]
    return unknown.reset_index(drop=True)


# --- Büyüme modelleri -----------------------------------------------------


def gompertz(t: np.ndarray, a: float, b: float, r: float) -> np.ndarray:
    """Gompertz büyüme modeli: V(t) = A * exp(-B * exp(-r*t)).

    A: asimptotik ölçek, B: gecikme/konum parametresi, r: büyüme katsayısı.
    """

    return a * np.exp(-b * np.exp(-r * t))


def logistic(t: np.ndarray, k: float, r: float, t0: float) -> np.ndarray:
    """Lojistik büyüme modeli: V(t) = K / (1 + exp(-r*(t - t0))).

    K: taşıma kapasitesi (asimptot), r: büyüme katsayısı, t0: eğim noktası.
    """

    return k / (1.0 + np.exp(-r * (t - t0)))


@dataclass
class ModelFitResult:
    model: str
    params: tuple[float, ...] | None
    r: float | None
    r2: float | None
    aic: float | None
    bic: float | None
    converged: bool
    fail_reason: str | None = None


def _aic_bic(n: int, k: int, rss: float) -> tuple[float, float]:
    if n <= k or rss <= 0:
        return float("nan"), float("nan")
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return float(aic), float(bic)


def _r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def fit_gompertz_model(t: np.ndarray, v: np.ndarray) -> ModelFitResult:
    """Gompertz modelini `curve_fit` ile fit eder."""

    v_max_guess = max(float(np.max(v)) * 1.5, float(np.max(v)) + 1.0)
    p0 = (v_max_guess, 1.0, 0.05)
    bounds = ([1e-6, 1e-6, -5.0], [np.inf, 50.0, 5.0])
    try:
        popt, _ = curve_fit(gompertz, t, v, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        return ModelFitResult("gompertz", None, None, None, None, None, False, str(exc))

    y_hat = gompertz(t, *popt)
    if not np.all(np.isfinite(y_hat)):
        return ModelFitResult(
            "gompertz", None, None, None, None, None, False, "non-finite fit output"
        )
    rss = float(np.sum((v - y_hat) ** 2))
    aic, bic = _aic_bic(len(t), 3, rss)
    return ModelFitResult(
        "gompertz",
        tuple(popt),
        float(popt[2]),
        _r_squared(v, y_hat),
        aic,
        bic,
        True,
    )


def fit_logistic_model(t: np.ndarray, v: np.ndarray) -> ModelFitResult:
    """Lojistik modeli `curve_fit` ile fit eder."""

    k_guess = max(float(np.max(v)) * 1.5, float(np.max(v)) + 1.0)
    t0_guess = float(np.median(t))
    p0 = (k_guess, 0.05, t0_guess)
    bounds = ([1e-6, -5.0, -1000.0], [np.inf, 5.0, 1000.0])
    try:
        popt, _ = curve_fit(logistic, t, v, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        return ModelFitResult("logistic", None, None, None, None, None, False, str(exc))

    y_hat = logistic(t, *popt)
    if not np.all(np.isfinite(y_hat)):
        return ModelFitResult(
            "logistic", None, None, None, None, None, False, "non-finite fit output"
        )
    rss = float(np.sum((v - y_hat) ** 2))
    aic, bic = _aic_bic(len(t), 3, rss)
    return ModelFitResult(
        "logistic",
        tuple(popt),
        float(popt[1]),
        _r_squared(v, y_hat),
        aic,
        bic,
        True,
    )


@dataclass
class PatientGrowthFit:
    patient_id: str
    n_visits: int
    fit_status: Literal["ok", "insufficient_visits", "both_failed"]
    gompertz: ModelFitResult | None = None
    logistic: ModelFitResult | None = None
    best_model: str | None = None
    best_r: float | None = None
    fail_reason: str | None = None


def fit_patient_growth_curves(
    series_df: pd.DataFrame, *, min_visits: int = 3
) -> list[PatientGrowthFit]:
    """Her LUMIERE hastası için Gompertz VE lojistik eğrileri fit eder.

    Model seçimi AIC'ye göre yapılır (düşük AIC = daha iyi fit); AIC
    hesaplanamazsa (n çok küçükse) R²'ye düşer. Varsayım YAPILMAZ — her iki
    model de fit edilip ÖLÇÜLÜR.

    En az `min_visits` (varsayılan 3) gerçek ziyareti olmayan hastalar
    ``insufficient_visits`` olarak işaretlenir, fit DENENMEZ.
    """

    results: list[PatientGrowthFit] = []
    for patient_id, group in series_df.groupby("patient_id"):
        group = group.sort_values("x_week")
        n = len(group)
        if n < min_visits:
            results.append(
                PatientGrowthFit(
                    patient_id=patient_id,
                    n_visits=n,
                    fit_status="insufficient_visits",
                    fail_reason=f"{n} < min_visits={min_visits}",
                )
            )
            continue

        t = group["x_week"].to_numpy(dtype=float)
        v = group["tumor_volume_mm3"].to_numpy(dtype=float)

        gomp = fit_gompertz_model(t, v)
        logi = fit_logistic_model(t, v)

        if not gomp.converged and not logi.converged:
            results.append(
                PatientGrowthFit(
                    patient_id=patient_id,
                    n_visits=n,
                    fit_status="both_failed",
                    gompertz=gomp,
                    logistic=logi,
                    fail_reason=f"gompertz={gomp.fail_reason}; logistic={logi.fail_reason}",
                )
            )
            continue

        candidates = [m for m in (gomp, logi) if m.converged]
        # AIC her ikisinde de hesaplanabiliyorsa AIC'ye göre seç; değilse R²'ye düş.
        if all(np.isfinite(c.aic) for c in candidates if c.aic is not None) and all(
            c.aic is not None for c in candidates
        ):
            best = min(candidates, key=lambda c: c.aic)
        else:
            best = max(candidates, key=lambda c: (c.r2 if c.r2 is not None else -np.inf))

        results.append(
            PatientGrowthFit(
                patient_id=patient_id,
                n_visits=n,
                fit_status="ok",
                gompertz=gomp,
                logistic=logi,
                best_model=best.model,
                best_r=best.r,
            )
        )

    return results


def growth_fits_to_dataframe(fits: list[PatientGrowthFit]) -> pd.DataFrame:
    """`fit_patient_growth_curves` çıktısını düz bir DataFrame'e çevirir."""

    rows = []
    for f in fits:
        rows.append(
            {
                "patient_id": f.patient_id,
                "n_visits": f.n_visits,
                "fit_status": f.fit_status,
                "fail_reason": f.fail_reason,
                "gompertz_converged": f.gompertz.converged if f.gompertz else None,
                "gompertz_r": f.gompertz.r if f.gompertz else None,
                "gompertz_r2": f.gompertz.r2 if f.gompertz else None,
                "gompertz_aic": f.gompertz.aic if f.gompertz else None,
                "gompertz_bic": f.gompertz.bic if f.gompertz else None,
                "logistic_converged": f.logistic.converged if f.logistic else None,
                "logistic_r": f.logistic.r if f.logistic else None,
                "logistic_r2": f.logistic.r2 if f.logistic else None,
                "logistic_aic": f.logistic.aic if f.logistic else None,
                "logistic_bic": f.logistic.bic if f.logistic else None,
                "best_model": f.best_model,
                "best_r": f.best_r,
            }
        )
    return pd.DataFrame(rows)


# --- Geçiş-bazlı (transition-based) yerel büyüme ölçüsü (Codex HIGH1) -----


def log_ratio_growth_rate(v_prev: float, v_curr: float, t_prev: float, t_curr: float) -> float:
    """Haftaya normalize log-oran büyüme hızı: ``log(V_curr/V_prev)/(t_curr-t_prev)``.

    `fit_gompertz_model`/`fit_logistic_model`'ın TÜM-SERİ fit'inden
    FARKLI olarak model varsayımı YAPMAZ — yalnız İKİ ardışık ölçüm
    arasındaki YEREL büyüme hızını verir (birim: 1/hafta). Bu, HER RANO
    etiketini KENDİ geçişine bağlamak için kullanılır (bkz. modül
    docstring'i "Codex çapraz-inceleme düzeltmeleri" HIGH1 bölümü).

    `t_curr <= t_prev` veya `v_prev <= 0` veya `v_curr <= 0` ise ``nan``
    döner (sessizce 0'a/başka bir değere YUVARLANMAZ — çağıran bu satırı
    raporunda görünür tutmalı, filtrelemek istiyorsa `dropna()` kendisi
    yapmalı).
    """

    if t_curr <= t_prev or v_prev <= 0 or v_curr <= 0:
        return float("nan")
    return float(np.log(v_curr / v_prev) / (t_curr - t_prev))


def compute_transition_growth_rates(series_df: pd.DataFrame) -> pd.DataFrame:
    """Her hastanın TÜM ardışık ziyaret çiftleri için yerel büyüme hızını hesaplar.

    `fetch_lumiere_wt_volume_series`'in döndürdüğü ham (RANO'dan bağımsız)
    hacim serisi üzerinde çalışır — her satır bir (prev, curr) geçişidir.
    RANO etiketiyle eşleme burada YAPILMAZ (bu modülün RANO bağımlılığı
    YOKTUR, döngüsel import'tan kaçınmak için); RANO-etiketli geçiş
    tablosu için `pipeline.followup_t3.build_patient_visit_deltas` +
    `build_delta_rano_table`'daki `log_ratio_rate_per_week_*` sütunlarına
    bakın (aynı formülü kullanır, `log_ratio_growth_rate` ile).

    Dönen sütunlar: patient_id, prev_scan_id, curr_scan_id, prev_week,
    curr_week, delta_weeks, prev_volume_mm3, curr_volume_mm3,
    log_ratio_rate_per_week.
    """

    if series_df.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "prev_scan_id",
                "curr_scan_id",
                "prev_week",
                "curr_week",
                "delta_weeks",
                "prev_volume_mm3",
                "curr_volume_mm3",
                "log_ratio_rate_per_week",
            ]
        )

    rows = []
    for patient_id, group in series_df.groupby("patient_id"):
        group = group.sort_values("x_week").reset_index(drop=True)
        for i in range(1, len(group)):
            prev_row = group.iloc[i - 1]
            curr_row = group.iloc[i]
            t_prev = float(prev_row["x_week"])
            t_curr = float(curr_row["x_week"])
            v_prev = float(prev_row["tumor_volume_mm3"])
            v_curr = float(curr_row["tumor_volume_mm3"])
            rows.append(
                {
                    "patient_id": patient_id,
                    "prev_scan_id": prev_row.get("scan_id"),
                    "curr_scan_id": curr_row.get("scan_id"),
                    "prev_week": t_prev,
                    "curr_week": t_curr,
                    "delta_weeks": t_curr - t_prev,
                    "prev_volume_mm3": v_prev,
                    "curr_volume_mm3": v_curr,
                    "log_ratio_rate_per_week": log_ratio_growth_rate(
                        v_prev, v_curr, t_prev, t_curr
                    ),
                }
            )
    return pd.DataFrame(rows)


# --- RANO grubu bazında büyüme katsayısı dağılımı --------------------------


def _assign_interpretable_flags(
    dist_df: pd.DataFrame,
    *,
    min_group_n: int,
    max_iqr_ratio_to_reference: float | None,
) -> pd.DataFrame:
    """`n`/`iqr_width` sütunlu bir dağılım tablosuna `interpretable` bayrağı ekler.

    2026-09-11 Barış kararı (bkz. modül docstring'i): İKİ koşul birden
    aranır (ikisi de sağlanmalı, yoksa `False`):
    (a) ``n >= min_group_n``
    (b) ``max_iqr_ratio_to_reference is None`` DEĞİLSE: grubun `iqr_width`'i,
        bu çağrıdaki EN BÜYÜK `n`'li satırın ("referans") `iqr_width`'inin
        `max_iqr_ratio_to_reference` katını AŞMAMALI. Referans satırın
        `iqr_width`'i 0 ise (tekil değer, saçılım tanımsız) bu koşul
        SESSİZCE atlanır (sıfıra bölme/anlamsız oran üretmemek için) --
        yalnız (a) uygulanır.

    Girdi `dist_df` DEĞİŞTİRİLMEZ, kopyası döner. Boş girdi için boş
    (ama `interpretable` sütunlu) bir DataFrame döner.
    """

    if dist_df.empty:
        result = dist_df.copy()
        result["interpretable"] = pd.Series(dtype=bool)
        return result

    result = dist_df.copy()
    reference_iqr = float(result.loc[result["n"].idxmax(), "iqr_width"])

    def _check(row: pd.Series) -> bool:
        if row["n"] < min_group_n:
            return False
        if max_iqr_ratio_to_reference is not None and reference_iqr > 0:
            if float(row["iqr_width"]) > max_iqr_ratio_to_reference * reference_iqr:
                return False
        return True

    result["interpretable"] = result.apply(_check, axis=1)
    return result


def assign_patient_rano_group(rano_df: pd.DataFrame) -> pd.DataFrame:
    """Her hastaya TEK bir RANO grubu atar (PD/SD/PR/CR) — son gerçek yanıt.

    **Varsayım (doğrulanmadı, açıkça beyan edilir):** bir hastanın büyüme
    katsayısı tek bir sayıdır ama RANO etiketi zaman içinde DEĞİŞİR (örn.
    SD -> PD). Bu fonksiyon hastayı, ``visit_week``'e göre EN SON kaydedilen
    yanıt-kategorisi etiketiyle (PD/SD/PR/CR; Pre-Op/Post-Op hariç)
    gruplar — hastalığın nihai seyrini büyüme hızıyla ilişkilendirmenin en
    savunulabilir tek-etiket yaklaşımı budur, ama TEK olası yaklaşım
    değildir (alternatif: en sık görülen etiket / ilk PD anı).

    ⚠️ **BİLİNEN SINIRLAMA (Codex HIGH1, 2026-08-19):** bu fonksiyonun
    çıktısı `growth_rate_by_rano_group` ile birleştirildiğinde, hastanın
    TÜM zaman serisine fit edilmiş TEK bir `r` değeri yalnızca EN SON
    etikete atanır — uzun süre SD/PR seyredip son vizitte PD'ye dönen bir
    hasta TAMAMEN PD grubuna yazılır. Geçiş-bazlı (her etiket kendi
    dönemine bağlı) alternatif için `compute_transition_growth_rates` +
    `pipeline.followup_t3.build_delta_rano_table` kullanın.
    """

    response_only = rano_df[rano_df["rano_label"].isin(RANO_RESPONSE_LABELS)]
    if response_only.empty:
        return pd.DataFrame(columns=["patient_id", "rano_group"])

    idx = response_only.groupby("patient_id")["visit_week"].idxmax()
    last = response_only.loc[idx, ["patient_id", "rano_label"]].rename(
        columns={"rano_label": "rano_group"}
    )
    return last.reset_index(drop=True)


def growth_rate_by_rano_group(
    fits_df: pd.DataFrame,
    rano_group_df: pd.DataFrame,
    *,
    min_group_n: int = _DEFAULT_MIN_GROUP_N,
    max_iqr_ratio_to_reference: float | None = _DEFAULT_MAX_IQR_RATIO_TO_REFERENCE,
) -> pd.DataFrame:
    """RANO grubu bazında `best_r` medyan ± IQR dağılımını çıkarır.

    `interpretable` bayrağı (2026-09-11 Barış kararı, bkz. modül docstring'i
    ve `_assign_interpretable_flags`) İKİ koşulu birden gerektirir:
    (a) ``n >= min_group_n`` (varsayılan 6 -- eski 5 kuralı SD'yi (n=5) tam
    sınırda `True` bırakıyordu); (b) grup IQR'ı bu çağrıdaki en büyük n'li
    grubun IQR'ının `max_iqr_ratio_to_reference` katını (varsayılan 5x)
    aşmamalı -- `None` verilirse bu koşul kapanır, yalnız (a) uygulanır.
    """

    ok = fits_df[fits_df["fit_status"] == "ok"][["patient_id", "best_r", "best_model"]]
    merged = ok.merge(rano_group_df, on="patient_id", how="inner")

    rows = []
    for group_name, group in merged.groupby("rano_group"):
        r_values = group["best_r"].to_numpy(dtype=float)
        n = len(r_values)
        if n == 0:
            continue
        median = float(np.median(r_values))
        q1 = float(np.percentile(r_values, 25))
        q3 = float(np.percentile(r_values, 75))
        rows.append(
            {
                "rano_group": group_name,
                "n": n,
                "median_r": median,
                "iqr_low": q1,
                "iqr_high": q3,
                "iqr_width": q3 - q1,
            }
        )
    result = pd.DataFrame(rows)
    result = _assign_interpretable_flags(
        result,
        min_group_n=min_group_n,
        max_iqr_ratio_to_reference=max_iqr_ratio_to_reference,
    )
    if not result.empty:
        result = result.sort_values("rano_group").reset_index(drop=True)
    return result


def growth_rate_by_rano_group_and_model(
    fits_df: pd.DataFrame,
    rano_group_df: pd.DataFrame,
    *,
    min_group_n: int = _DEFAULT_MIN_GROUP_N,
    max_iqr_ratio_to_reference: float | None = _DEFAULT_MAX_IQR_RATIO_TO_REFERENCE,
) -> pd.DataFrame:
    """`growth_rate_by_rano_group`'un (rano_group, best_model) ile TABAKALI hâli.

    Codex LOW bulgusu: `growth_rate_by_rano_group` logistic/Gompertz `r`
    değerlerini TEK bir havuzda karıştırıyor (iki modelin `r`'si aynı
    büyüme hızını FARKLI parametrizasyonlarla temsil eder, doğrudan
    karşılaştırılabilir olduğu VARSAYILMAMALI). Bu fonksiyon aynı hesabı
    `best_model`'e göre de tabakalayarak üretir — orijinal (havuzlanmış)
    fonksiyon SİLİNMEDİ, bu YANINDA bir ek görünümdür.

    `interpretable` bayrağı `growth_rate_by_rano_group` ile AYNI iki koşulu
    kullanır (2026-09-11 Barış kararı) — referans grup burada (rano_group,
    best_model) çiftleri arasında en büyük n'e sahip satırdır.
    """

    ok = fits_df[fits_df["fit_status"] == "ok"][["patient_id", "best_r", "best_model"]]
    merged = ok.merge(rano_group_df, on="patient_id", how="inner")

    rows = []
    for (group_name, model_name), group in merged.groupby(["rano_group", "best_model"]):
        r_values = group["best_r"].to_numpy(dtype=float)
        n = len(r_values)
        if n == 0:
            continue
        median = float(np.median(r_values))
        q1 = float(np.percentile(r_values, 25))
        q3 = float(np.percentile(r_values, 75))
        rows.append(
            {
                "rano_group": group_name,
                "best_model": model_name,
                "n": n,
                "median_r": median,
                "iqr_low": q1,
                "iqr_high": q3,
                "iqr_width": q3 - q1,
            }
        )
    result = pd.DataFrame(rows)
    result = _assign_interpretable_flags(
        result,
        min_group_n=min_group_n,
        max_iqr_ratio_to_reference=max_iqr_ratio_to_reference,
    )
    if not result.empty:
        result = result.sort_values(["rano_group", "best_model"]).reset_index(drop=True)
    return result


def uninterpretable_group_names(dist_df: pd.DataFrame, *, group_col: str = "rano_group") -> list:
    """`interpretable=False` işaretli grupların adlarını döner (Codex HIGH2).

    `growth_rate_by_rano_group`/`growth_rate_by_rano_group_and_model` VE
    `pipeline.followup_t3.build_delta_rano_table` çıktılarıyla çalışır
    (yalnız `group_col` adı farklıdır: sırasıyla `"rano_group"`/`"rano_label"`).
    Amaç: `interpretable` bayrağının CSV'nin içine gömülü kalıp üst-düzey
    konsol/rapor çıktısına TAŞINMAMASI riskini gidermek — çağıran script bu
    listeyi AÇIKÇA yazdırmalı ve bu gruplarla YÖN/KARŞILAŞTIRMA iddiası
    KURMAMALIDIR.
    """

    if dist_df.empty or "interpretable" not in dist_df.columns:
        return []
    return dist_df.loc[~dist_df["interpretable"], group_col].tolist()


# --- Sınır-temelli morfolojik büyüme/küçülme simülasyonu -------------------


def simulate_boundary_growth(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    target_volume_mm3: float,
) -> tuple[np.ndarray, float]:
    """Maskeyi, SINIRDAN (yüzey normali yönünde) hedef hacme genişletir/küçültür.

    **BU BİR SİMÜLASYONDUR, GERÇEK TAHMİN DEĞİLDİR.** Çıktı, mevcut
    segmentasyon maskesinin şeklini KORUYARAK yüzeyini içe/dışa iten
    matematiksel bir projeksiyondur — hastanın gerçek gelecekteki tümör
    şeklinin KESİN bir tahmini değildir.

    **Yöntem — uniform scaling DEĞİL:** merkezden ölçekleme (tüm
    koordinatları sabit bir katsayıyla çarpma) yerine, Öklid mesafe
    dönüşümü (`scipy.ndimage.distance_transform_edt`, voksel-anizotropisi
    `spacing` ile hesaba katılır) kullanılarak sınıra EN YAKIN artık/dışta-
    kalan vokseller SIRAYLA eklenir/çıkarılır (rank-bazlı seçim). Bu,
    tümörün mevcut düzensiz (non-konveks) şeklini korur; uniform scaling
    bunu KORUMAZ (küçük çıkıntılar orantısız büyür/küçülür).

    **Neden sabit-eşik (`dist <= d`) bisection DEĞİL, rank-bazlı seçim:**
    İlk uygulama sürekli bir mesafe eşiği `d` üzerinde ikili arama
    yapıyordu (`new_mask = mask | (dist_outside <= d)`). Izgara-tabanlı
    Öklid mesafe dönüşümünde BİRÇOK voksel TAM OLARAK aynı mesafe değerini
    paylaşır (örn. köşegen komşular hep √2) — bu, `d` o değeri geçtiği anda
    hacmin BÜYÜK bir sıçrama yapmasına yol açar (canlı ölçüldü: yarıçap-8
    küre maskesinde tek bir √2 eşiğinde hacim 2823→3287'ye sıçradı, hedefin
    %10'undan fazla hata). Rank-bazlı seçim (mesafeye göre sırala, tam
    olarak N voksel al) bu sıçramayı YOK EDER — sonuç ±1 vokselin hacmi
    kadar (tipik olarak <%1) hassasiyetle KESİNDİR.

    Döner: (yeni_maske, ulaşılan_hacim_mm3).

    Raises:
        ValueError: maske boşsa, `target_volume_mm3<=0` ise, veya hedef
            hacim maskenin sınırlayıcı voksel matrisine (`mask.size`)
            sığmayacak kadar büyükse (sessiz kırpma YAPILMAZ — açık hata).
    """

    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        raise ValueError("Boş maskeden simülasyon yapılamaz (0 voksel).")
    if target_volume_mm3 <= 0:
        raise ValueError("target_volume_mm3 > 0 olmalı.")

    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
    v0_voxels = int(mask.sum())
    target_voxels = int(round(target_volume_mm3 / voxel_volume))

    if target_voxels > mask.size:
        raise ValueError(
            f"Hedef hacim ({target_volume_mm3:.1f} mm3 -> {target_voxels} voksel) "
            f"maskenin sınırlayıcı voksel matrisine ({mask.size} voksel) sığmıyor -- "
            "daha büyük bir görüntü matrisi/ROI kırpması gerekir. Sessizce kırpılmadı."
        )
    if target_voxels < 0:
        raise ValueError("target_volume_mm3 negatif voksel sayısına karşılık geliyor.")

    if target_voxels == v0_voxels:
        return mask.copy(), float(v0_voxels) * voxel_volume

    new_mask = mask.copy()
    if target_voxels > v0_voxels:
        n_needed = target_voxels - v0_voxels
        dist_outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
        bg_flat_idx = np.flatnonzero(~mask)
        bg_dist = dist_outside.reshape(-1)[bg_flat_idx]
        order = np.argpartition(bg_dist, n_needed - 1)[:n_needed]
        chosen = bg_flat_idx[order]
        new_mask.reshape(-1)[chosen] = True
    else:
        n_remove = v0_voxels - target_voxels
        dist_inside = ndimage.distance_transform_edt(mask, sampling=spacing)
        fg_flat_idx = np.flatnonzero(mask)
        fg_dist = dist_inside.reshape(-1)[fg_flat_idx]
        # sınıra EN YAKIN (mesafesi en küçük) vokseller ÖNCE çıkarılır
        order = np.argpartition(fg_dist, n_remove - 1)[:n_remove]
        chosen = fg_flat_idx[order]
        new_mask.reshape(-1)[chosen] = False

    achieved_volume = float(new_mask.sum()) * voxel_volume
    return new_mask, achieved_volume


@dataclass
class VolumeProjection:
    """6 aylık (veya `months`) hacim projeksiyonu — IQR bazlı belirsizlik aralığı.

    Bu ÜÇ NOKTA TAHMİNİ DEĞİL, bir ARALIKTIR: `v_low`/`v_high` RANO
    grubunun büyüme katsayısı IQR'sinin alt/üst çeyreğine karşılık gelir.

    **`from_week` / `t_target_week` (2026-09-13 DÜZELTME, G1):** yüzde
    değişim `v0`'a göre hesaplanır, ama `v0` çağıranın verdiği bir
    REFERANS hacimdir ve eğrinin t=0 değeri OLMAK ZORUNDA DEĞİLDİR
    (`api/analyze_patient.py` SON ziyaretin hacmini verir). Bu yüzden
    projeksiyonun hangi haftadan BAŞLADIĞI (`from_week`) ve hangi haftada
    DEĞERLENDİRİLDİĞİ (`t_target_week = from_week + months*4,345`) artık
    çıktıda AÇIKÇA taşınır — tüketici (demo/rapor) bu sabiti kendi
    tarafında yeniden hesaplamamalıdır.
    """

    v0: float
    months: float
    model: str
    r_low: float
    r_median: float
    r_high: float
    v_low: float
    v_median: float
    v_high: float
    pct_low: float
    pct_median: float
    pct_high: float
    from_week: float = 0.0
    t_target_week: float = 0.0


_WEEKS_PER_MONTH = 4.345  # 52.14/12 haftanın ay eşdeğeri (yaklaşık, belgelenir)


def project_volume_range(
    v0: float,
    model: str,
    template_params: tuple[float, ...],
    r_low: float,
    r_median: float,
    r_high: float,
    *,
    months: float = 6.0,
    from_week: float = 0.0,
) -> VolumeProjection:
    """Bir hastanın hacim projeksiyonunu, RANO-grubu r IQR'siyle üç senaryoda hesaplar.

    `template_params`, hastanın (veya popülasyon medyanının) fit edilmiş
    diğer model parametrelerini taşır (Gompertz: A,B; lojistik: K,t0) — yalnız
    `r` değişir, diğerleri sabit tutularak IQR'in HACME etkisi izole edilir.

    **Not:** hafta -> ay dönüşümü yaklaşık bir sabitle (`_WEEKS_PER_MONTH`)
    yapılır; LUMIERE zaten hafta biriminde kalibre edildiği için ay ekseni
    yalnız RAPORLAMA amaçlıdır.

    `from_week`: projeksiyonun BAŞLANGIÇ noktası, hastanın KENDİ zaman
    ekseninde (hafta). Hedef hafta `t_target = from_week + months *
    _WEEKS_PER_MONTH` olarak hesaplanır. `v0` ile AYNI ana karşılık
    gelmelidir — yani çağıran, yüzde değişimin referansı olan `v0`'ın
    ölçüldüğü haftayı vermelidir.

    ⚠️ **DÜZELTME 2026-09-13 (G1) — eski (HATALI) davranış SİLİNMEDİ, burada
    kayıtlıdır (wiki hard rule #3: çelişki silinmez, işaretlenir):**
    Bu parametre eklenmeden ÖNCE hedef hafta KOŞULSUZ `t_target = months *
    _WEEKS_PER_MONTH` (6 ay için 26,07 hafta) idi, yani hastanın İLK
    taramasından itibaren MUTLAK bir haftaydı. `api/analyze_patient.py`
    ise `v0` olarak **SON** ziyaretin hacmini veriyordu. Son ziyaret hafta
    26,07'den SONRAYSA (LUMIERE'de tipik) yüzde değişim fiilen
    *"eğrinin GEÇMİŞTEKİ bir haftadaki değeri vs BUGÜNKÜ gözlem"*
    karşılaştırmasıydı — ileriye dönük bir projeksiyon DEĞİL.
    Ölçülen etki (`Patient-028`, son ziyaret hafta 38, v0=112.576 mm³):
    eski t=26,07 ile aralık %+32,96 … %+43,87 … %+49,20;
    düzeltilmiş t=38+26,07=64,07 ile %−0,47 … %+43,97 … %+52,64 —
    **alt uç İŞARET DEĞİŞTİRİYOR** (klinik mesaj ters).
    `from_week` varsayılanı **0.0**'dır, yani eski çağrılar (örn.
    `tools/run_growth_simulation_lumiere.py`'nin SENTETİK küre demosu —
    orada zaman serisi YOKTUR, tek tanımlı başlangıç t=0'dır) bit-birebir
    aynı sonucu üretmeye DEVAM EDER. Gerçek bir hasta serisiyle çağıran
    her yeni kod `from_week`'i AÇIKÇA vermelidir.
    """

    if model not in ("gompertz", "logistic"):
        raise ValueError(f"Bilinmeyen model: {model!r}")

    t_target = from_week + months * _WEEKS_PER_MONTH

    def _v_at(r: float) -> float:
        if model == "gompertz":
            a, b, _ = template_params
            return float(gompertz(np.array([t_target]), a, b, r)[0])
        else:
            k, _, t0 = template_params
            return float(logistic(np.array([t_target]), k, r, t0)[0])

    v_low, v_median, v_high = sorted((_v_at(r_low), _v_at(r_median), _v_at(r_high)))

    def _pct(v: float) -> float:
        return (v - v0) / v0 * 100.0 if v0 else float("nan")

    return VolumeProjection(
        v0=v0,
        months=months,
        model=model,
        r_low=r_low,
        r_median=r_median,
        r_high=r_high,
        v_low=v_low,
        v_median=v_median,
        v_high=v_high,
        pct_low=_pct(v_low),
        pct_median=_pct(v_median),
        pct_high=_pct(v_high),
        from_week=float(from_week),
        t_target_week=float(t_target),
    )
