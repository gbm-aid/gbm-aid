"""v3 hazırlığı -- iki AYRI radyomik özellik sorununu ayırıp AYRI AYRI
çözen filtre katmanı (koordinatör görevi, 2026-08-18: "düşük varyans
filtresini de v3'e ekle, hazırlığını yap").

🔴 GERÇEK VERİYLE ÖLÇÜLDÜ (bu modülü yazan ajan, `week3_v2a_mgmt_final_
coefficients.csv`'deki `WT__original_gldm_SmallDependenceLowGrayLevel
Emphasis` (coef=98,6, se=106,8) bulgusunu gerçek UPenn 611/585 WT×93
matrisiyle BAĞIMSIZ olarak yeniden ölçtü, sonuçlar koordinatörün
raporuyla BİREBİR eşleşti):

BULGU 1 (doğrulandı) -- `ConvergenceWarning` üreten 6 özellik, HAM
(standardize edilmemiş) mutlak std'de tam ilk 6 sırada:
    ngtdm_Coarseness(std=3.03e-4) < gldm_SmallDependenceLowGrayLevel
    Emphasis(5.88e-4) < glcm_Idmn(3.19e-3) < glszm_SmallAreaLowGrayLevel
    Emphasis(4.66e-3) < glrlm_ShortRunLowGrayLevelEmphasis(6.63e-3) <
    glszm_LowGrayLevelZoneEmphasis(7.39e-3).

BULGU 2 (doğrulandı, TAM sayısal eşleşme) -- 🔴 VARYASYON KATSAYISI (CV
= std/|mean|) YANLIŞ ÖLÇÜT: aynı 6 özelliğin CV-artan sıralamasındaki
yerleri {1, 47, 51, 53, 54, 89}/93 -- `glcm_Idmn` mutlak std'de 3.
sırada (numerik olarak kırılgan) ama CV'de TÜM 93 özellik içinde EN
DÜŞÜK (rank 1, "en istikrarlı" görünüyor) çünkü değerleri her hastada
~1.0'a çok yakın (mean=0.994, std=0.0032) -- CV bunu YANLIŞLIKLA "en
güvenli özellik" ilan eder. `ngtdm_Coarseness` ise CV'de neredeyse EN
YÜKSEK (rank 89/93) -- ham ölçekte en küçük olmasına rağmen GERÇEKTEN
DEĞİŞKEN bir özellik, yanlışlıkla "düşük varyans" diye atılmamalı.
SONUÇ: ham-ölçek numerik kırılganlığı ile GERÇEK bilgi-yoksunluğu
(near-constant) AYNI ŞEY DEĞİL -- ilki ÖLÇEKLEME sorunu (standardizasyon
çözer), ikincisi GERÇEK bir düşük-bilgi durumu (CV ile yakalanır, ham
std ile YAKALANMAZ).

BULGU 3 (doğrulandı + GENİŞLETİLDİ) -- kolinearite AYRI bir sorun:
`glcm_Idmn<->glcm_Contrast` |r|=0.999, `glszm_LowGrayLevelZoneEmphasis
<->glszm_SmallAreaLowGrayLevelEmphasis` |r|=0.987,
`glrlm_ShortRunLowGrayLevelEmphasis<->glrlm_LowGrayLevelRunEmphasis`
|r|=0.984, `ngtdm_Coarseness<->ngtdm_Strength` |r|=0.829 -- hepsi
doğrulandı. EK BULGU (koordinatörün listesinde YOKTU, bu modülü yazan
ajan buldu): v2a'nın GERÇEK final 12-radyomik-özellik setinde
`glszm_SizeZoneNonUniformityNormalized<->glszm_SmallAreaEmphasis`
|r|=0.9964 (VIF≈172, R²=0.994) -- bu, coef=98,6 olan özelliğin
KENDİSİYLE değil ama AYNI modeldeki BAŞKA bir çiftle, tüm tasarım
matrisini kötü-koşullandıran (ill-conditioned) GERÇEK bir ikinci
kolinearite kaynağı.

TEŞHİS DENEYİ (gerçek UPenn verisiyle, bu modülü yazan ajan tarafından
koşuldu -- görev talimatının istediği "somut test"):
    A) HAM (standardize edilmemiş) fit  -> 1 ConvergenceWarning,
       gldm_SmallDependenceLowGrayLevelEmphasis: coef=21.73, se=100.19,
       se/|coef|=4.61
    B) TÜM 12 radyomik özellik z-score STANDARDIZE edilip AYNI fit ->
       0 ConvergenceWarning, coef=0.0128, se=0.0589, se/|coef|=4.61
    SONUÇ: standardizasyon ConvergenceWarning'i (SAYISAL belirti)
    TAMAMEN ORTADAN KALDIRIYOR -- bu ÜCRETSİZ ve DOĞRU bir düzeltme,
    HER ZAMAN uygulanmalı. AMA se/|coef| ORANI (istatistiksel belirsizlik)
    DEĞİŞMİYOR (matematiksel olarak beklenen -- doğrusal yeniden
    ölçekleme z-istatistiğini değiştirmez, bkz. `pipeline.cox_model.
    standardize_columns_fold_safe()` docstring'indeki AYNI kanıt).
    Yani standardizasyon TEK BAŞINA sorunu "ÇÖZMÜYOR" -- SAYISAL
    belirtiyi giderir ama katsayının GERÇEKTEN yorumlanamaz/belirsiz
    kaldığı gerçeğini DEĞİŞTİRMEZ.
    C/D) gldm_SmallDependenceLowGrayLevelEmphasis'in v2a-final-modelindeki
    KENDİ VIF'i sadece 2,43 (dikkat çekici değil) -- modeldeki iki
    korelasyon-partnerini (SizeZoneNonUniformityNormalized /
    SmallAreaEmphasis) TEK TEK modelden ÇIKARMAK bu özelliğin kendi
    se/|coef| oranını İYİLEŞTİRMEDİ (4,61 -> 4,89 / 8,60, yani KÖTÜLEŞTİ).
    SONUÇ: bu ÖZEL katsayının belirsizliği o çiftle İLİŞKİLİ DEĞİL --
    bu özelliğin kendisi basitçe ZAYIF/GERÇEK bir sinyal taşımıyor
    (standardize edilince coef≈0,013, p≈0,36-0,90 civarı, her
    senaryoda). Ayrı ve GERÇEK bir kolinearite sorunu (VIF≈172 çift)
    AYNI modelde AYRICA var, ama bu spesifik katsayıyı DEĞİL, modelin
    GENEL kararlılığını tehdit ediyor.

SONUÇ (iki sorun kesin olarak AYRIŞTIRILDI):
  (a) ÖLÇEK/SAYISAL KIRILGANLIK -- HER ZAMAN standardizasyonla düzeltilir
      (`v3`'te `clinical_standardize_columns`'a radyomik kolonlar da
      EKLENİR -- bkz. `tools/train_cox_week3.py::run_single_v3_variant()`,
      `pipeline/cox_model.py`'ye TEK SATIR dokunmadan, ÇÜNKÜ o parametre
      zaten "hangi kolonlar" konusunda JENERİK).
  (b) NEREDEYSE-MÜKEMMEL KOLİNEARİTE -- standardizasyonla ÇÖZÜLMEZ, bu
      modülün `select_cluster_representatives()`'i ile HAVUZ SEVİYESİNDE
      (nested-CV'den ÖNCE, `STABLE_FEATURES_ICC60`'ın KENDİSİNİN
      türetildiği "aday havuz tanımlama" adımıyla AYNI mimari kademede,
      bkz. modül-altı not) dedup edilir.
  (c) GERÇEKTEN NEAR-CONSTANT (düşük GÖRECELİ bilgi, örn. `glcm_Idmn`) --
      CV ile (HAM std İLE DEĞİL) tespit edilip havuzdan ÇIKARILIR.

⚠️ ICC>=0,60 filtresi (`pipeline.cox_model.STABLE_FEATURES_ICC60`) BU
SORUNU YAKALAMAZ -- ICC kaynaklar-arası (UPenn/LUMIERE/TCGA C32 çıkarım
tekrarları) TEKRARLANABİLİRLİĞİ ölçer, ÖLÇEK ya da KOLİNEARİTE değil.
Bir özellik HEM tekrarlanabilir (yüksek ICC) HEM neredeyse sabit/neredeyse
kopya (düşük CV / yüksek |r|) OLABİLİR -- `glcm_Idmn` ZATEN
`STABLE_FEATURES_ICC60`'IN İÇİNDE (ICC filtresini geçmiş) ama CV=0,0032
ile neredeyse sabit. Bu modül ICC filtresinin YERİNE GEÇMEZ, ONA EK bir
İKİNCİ, BAĞIMSIZ eksende (X-ONLY, çıktı/survival'a HİÇ bakmadan) çalışan
bir süzgeçtir.

KAPSAM/YÖNTEM NOTU (dürüstçe belirtilir, gizlenmez): bu filtreler
`STABLE_FEATURES_ICC60`'ın KENDİSİ gibi TÜM eğitim havuzundan (fold-dışı,
BİR KEZ) türetilir -- nested-CV'nin dış-fold döngüsü İÇİNDE YENİDEN
hesaplanmaz. Bu, ICC filtresiyle AYNI mimari kademe/emsal (X-ONLY, çıktı
DEĞİŞKENİNE hiç bakmıyor -- `survival_days`/`event` bu fonksiyonların
hiçbirine PARAMETRE OLARAK bile VERİLMİYOR, bu yüzden CLAUDE.md'nin asıl
endişe ettiği "çıktıyı görüp öğrenme" sızıntı sınıfından YAPISAL olarak
FARKLI ve daha hafif) ama SIFIR-RİSK değildir: test-fold hastalarının
X-değerleri de havuz istatistiklerine (std/mean/korelasyon) katkı verir.
Bu AÇIK RİSK olarak beyan edilir, `STABLE_FEATURES_ICC60` için de ZATEN
kabul edilmiş olan AYNI ödünleşimdir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_CV_THRESHOLD",
    "DEFAULT_CORRELATION_CLUSTER_THRESHOLD",
    "compute_feature_variability_summary",
    "select_near_constant_features",
    "correlation_clusters_union_find",
    "select_cluster_representatives",
    "V3CandidatePoolReport",
    "build_v3_candidate_pool",
    "NonFiniteFeatureValueError",
]


#: Bulgu 2'nin CV-artan sıralamasındaki DOĞAL kopma noktası (bu modülü
#: yazan ajanın gerçek UPenn WT×93 verisiyle ÖLÇTÜĞÜ, tahmin edilmedi):
#: CV değerleri log10-ölçekte sıralandığında ilk iki sıçrama
#: (`glcm_Idmn`->`glcm_Idn` arası log10-gap=0,593, `glcm_Idn`->
#: `glszm_ZoneEntropy` arası log10-gap=0,447) bulk'un GERİ KALANINDAKİ
#: (rank>=3, ~0,05-0,09 aralığında, SÜREKLİ/kademeli) gap'lerden 5-13
#: kat büyük -- yalnız BU İKİ özellik (`glcm_Idmn` CV=0,0032, `glcm_Idn`
#: CV=0,0126) geri kalan 91 özellikten NİTELİKSEL olarak AYRIŞIYOR.
#: Eşik bu iki gap'in ARASINA (0,0126 ile 0,0352 arası) konursa TAM
#: bu iki özellik yakalanır. 0,02 bu aralığın ortasıdır.
DEFAULT_CV_THRESHOLD = 0.02

#: Korelasyon-kümeleme (union-find, `correlation_clusters_union_find()`)
#: için eşik -- GERÇEK UPenn WT×93 verisiyle eşik-taraması yapıldı
#: (0,60'tan 0,99'a, `tools/train_cox_week3.py::compute_correlation_
#: clusters()` -- AYNI algoritma -- ile). BULGU: eşik 0,925'in ALTINA
#: düşünce union-find'in TRANSİTİF ZİNCİRLEME patolojisi devreye giriyor
#: -- en büyük küme boyutu 8 (eşik=0,95) -> 24 (eşik=0,925) -> 30
#: (eşik=0,90) -> 45 (eşik=0,85) -> 57 (eşik=0,80) -> 84 (eşik=0,70,
#: `compute_correlation_clusters()`'ın KENDİ varsayılanı!) şeklinde
#: SIÇRIYOR -- yani mevcut kod tabanındaki 0,7 varsayılanı (kesifsel
#: küme-frekansı analizi için MAKUL, orada birincil seçimin YERİNİ
#: ALMIYOR) bir DEDUPLİKASYON/temsilci-seçme filtresi için KULLANILAMAZ
#: (93 özellikten 84'ü TEK bir dev kümeye düşer, "bir temsilci seç"
#: anlamsızlaşır). 0,95 eşiği hem "neredeyse-tam-kopya" tanımına sadık
#: kalır (küme büyüklüğü <=8) hem zincirleme patolojisinden ÖNCEki son
#: durak. `week3_v2a_mgmt_final_coefficients.csv`'deki GERÇEK VIF≈172
#: çift (`glszm_SizeZoneNonUniformityNormalized`<->`glszm_
#: SmallAreaEmphasis`, |r|=0,9964) bu eşikte doğru şekilde yakalanıyor
#: (bkz. modül docstring'i).
DEFAULT_CORRELATION_CLUSTER_THRESHOLD = 0.95


class NonFiniteFeatureValueError(ValueError):
    """DÜŞÜK öncelik (Codex şartlı onay eki, 2026-09-11) -- `frame[feature_
    columns]` içinde NaN/±inf değer taşıyan sütun(lar) varsa `build_v3_
    candidate_pool()` tarafından fırlatılır.

    NEDEN ÖNEMLİ: `compute_feature_variability_summary()`'nin CV filtresi
    yalnızca "tüm satırlar aynı sabit değer" (std==0) sınıfını yakalar
    (bkz. o fonksiyondaki 2026-08-28 SIFIR-ORTALAMA/TEKİLLİK korumaları).
    Bir sütun ±inf İÇERİYORSA `std`/`mean` de ±inf/NaN olur -> `cv =
    inf/inf = NaN` -> `select_near_constant_features()`'ın `NaN < esik ->
    False` + `is_constant` (std==0 DEĞİL, ±inf'te std sonlu değildir)
    mantığı bu sütunu ELEMEZ. Sıradaki adımda `correlation_clusters_
    union_find()` içindeki `.corr()` bu sütun için TÜM eşleşmelerde NaN
    döner (asla `>= corr_threshold` olmaz) -- yani sütun HİÇBİR kümeye
    katılmaz, KENDİ tek-üyeli kümesinde "temsilci" olarak SESSİZCE hayatta
    kalır. Bozuk/dejenere bir sütun filtre zincirinin HER İKİ aşamasını
    (near-constant VE korelasyon) da sessizce atlatmış olur.

    Bu guard bunu ÇAĞRI ANINDA (fail-loud) yakalar -- NaN/inf değerlerin
    DÜŞÜRÜLMESİ/doldurulması bu modülün SORUMLULUĞU DEĞİLDİR, çağıran
    taraf veri hazırlığında bu kararı ÖNCEDEN vermiş olmalıdır."""


def _assert_finite_features(frame: pd.DataFrame, feature_columns: list[str]) -> None:
    """`build_v3_candidate_pool()`'un ilk adımı -- bkz. `NonFiniteFeature
    ValueError` docstring'i. `feature_columns`'daki HER sütunu NaN/±inf
    için tarar, ihlal varsa sütun adlarını VE her sütun için kaç
    satırın etkilendiğini içeren bir hata fırlatır (sessizce YOK SAYMAZ,
    kısmi doldurma/düşürme YAPMAZ)."""

    subset = frame[feature_columns]
    non_finite_counts: dict[str, int] = {}
    for column in feature_columns:
        values = pd.to_numeric(subset[column], errors="coerce").to_numpy(dtype=float)
        n_non_finite = int((~np.isfinite(values)).sum())
        if n_non_finite:
            non_finite_counts[column] = n_non_finite

    if non_finite_counts:
        offending = sorted(non_finite_counts)
        detail = ", ".join(f"{col}={non_finite_counts[col]}" for col in offending[:10])
        raise NonFiniteFeatureValueError(
            "build_v3_candidate_pool(): asagidaki ozellik sutun(lar)i NaN/"
            f"±inf deger iceriyor (sutun=etkilenen_satir_sayisi): {detail}"
            f"{'...' if len(offending) > 10 else ''}. Bu deger CV/korelasyon "
            "hesaplarinda dejenere bir sutunu sessizce filtre disi "
            "birakabilir (bkz. NonFiniteFeatureValueError docstring'i) -- "
            "cagiran taraf bu sutun(lar)i once temizlemeli/dusurmelidir."
        )


def compute_feature_variability_summary(
    frame: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Her özellik için HAM (standardize edilmemiş) std/mean/CV + rank'lar.

    `survival_days`/`event` bu fonksiyona hiç PARAMETRE OLARAK verilmiyor
    -- X-only, çıktı-bağımsız (bkz. modül docstring'indeki kapsam notu).
    CV = std/|mean| -- Bulgu 2'nin gösterdiği gibi HAM std, farklı fiziksel
    birimlere sahip özellikler arasında KIYASLANAMAZ; CV özelliğin KENDİ
    ölçeğine göre göreli değişkenliği ölçer.
    """

    std = frame[feature_columns].std(ddof=1)
    mean = frame[feature_columns].mean()

    # --- SIFIR-ORTALAMA KORUMASI (2026-08-28, Codex stop-time bulgusu) ---
    # CV = std/|mean| ölçütü mean==0 olduğunda TANIMSIZDIR ve iki ayrı
    # hataya yol açıyordu:
    #   (a) std==0 & mean==0 (TAM SABİT, ör. tümü 0 olan kolon)
    #       -> cv = 0/0 = NaN -> `NaN < esik` False -> kolon ELENMİYORDU.
    #          Oysa sıfır-varyanslı kolon TEKİLLİĞİN EN BARİZ KAYNAĞIDIR;
    #          bu filtrenin var oluş sebebi tam olarak onu elemekti.
    #   (b) std>0 & mean==0 -> cv = inf; ardından `.rank().astype(int)`
    #       `IntCastingNaNError` ile TÜM FONKSİYONU ÇÖKERTİYORDU.
    # Bu, XGBoost hattında GERÇEKLEŞEBİLİR bir senaryodur: WT+TC kolunda
    # bir fold'un tamamında TC hacmi 0 olan hastalar bulunabilir (nitekim
    # UPENN-GBM-00354 ve 00397 saf ödem, TC=0 -- 609/583 kararının sebebi).
    # Ölçek-bağımsız olduğu için tam sıfır varyans testi kullanılıyor:
    # std==0 matematiksel olarak tekillik demektir, birime bakılmaz.
    is_constant = std == 0.0
    cv_is_undefined = mean.abs() == 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        cv = std / mean.abs()

    summary = pd.DataFrame(
        {
            "std": std,
            "mean": mean,
            "cv": cv,
            "is_constant": is_constant,
            "cv_is_undefined": cv_is_undefined,
        }
    )
    # NaN/inf rank'ı int'e çevrilemez -> rank'lar nullable Int64, sıralamada
    # tanımsızlar EN SONA konur. Fonksiyon artık ÇÖKMEZ; tanımsızlık veri
    # olarak taşınır ve `select_near_constant_features()` onu ele alır.
    summary["std_rank_ascending"] = (
        summary["std"].rank(method="min", na_option="bottom").astype("Int64")
    )
    summary["cv_rank_ascending"] = (
        summary["cv"].replace([np.inf, -np.inf], np.nan)
        .rank(method="min", na_option="bottom")
        .astype("Int64")
    )
    return summary.sort_values("cv", na_position="last")


def select_near_constant_features(
    variability_summary: pd.DataFrame, *, cv_threshold: float = DEFAULT_CV_THRESHOLD
) -> list[str]:
    """`cv < cv_threshold` olan (GERÇEKTEN göreli-değişkenliği düşük,
    near-constant) özellikleri döndürür -- HAM std ile DEĞİL, CV ile
    (Bulgu 2, bkz. modül docstring'i: ham std ile filtrelemek `ngtdm_
    Coarseness` gibi GERÇEKTEN değişken ama küçük-birimli özellikleri
    yanlışlıkla elerdi, `glcm_Idmn` gibi GERÇEKTEN near-constant bir
    özelliği ise HAM std sıralamasında -tesadüfen doğru yakalasa bile-
    YANLIŞ GEREKÇEYLE yakalardı)."""

    cv_mask = variability_summary["cv"] < cv_threshold

    # --- TEKİLLİK KORUMASI (2026-08-28, Codex stop-time bulgusu) ---
    # `std == 0` olan kolon HER DURUMDA elenir -- CV'ye BAKILMAZ.
    # Gerekçe: sıfır varyanslı kolon Cox tasarım matrisinde doğrudan
    # tekillik üretir (`Matrix is singular`), ki bu filtrenin bağlanma
    # sebebi tam olarak oydu. mean==0 iken CV = 0/0 = NaN olduğu ve
    # `NaN < esik` False döndüğü için, bu kolonlar ESKİDEN sessizce
    # havuzda KALIYORDU -- klasik fail-open.
    # `.fillna(False)` da ayrıca gerekli: cv NaN/inf ise karşılaştırma
    # NA üretebilir; boolean maskede NA'yı "eleme" saymıyoruz, sabitlik
    # kararını `is_constant` veriyor.
    if "is_constant" in variability_summary.columns:
        constant_mask = variability_summary["is_constant"].astype(bool)
    else:  # geriye dönük uyum: eski özet çerçeveleri bu kolonu taşımaz
        constant_mask = variability_summary["std"] == 0.0

    mask = cv_mask.fillna(False).astype(bool) | constant_mask
    return sorted(variability_summary.index[mask].tolist())


def correlation_clusters_union_find(
    frame: pd.DataFrame, feature_columns: list[str], *, corr_threshold: float
) -> dict[str, int]:
    """`tools/train_cox_week3.py::compute_correlation_clusters()` ile
    BİREBİR AYNI algoritma (basit union-find, |r|>=corr_threshold olan
    her çift aynı kümeye düşer) -- BİLİNÇLİ OLARAK KOPYALANDI, İTHAL
    EDİLMEDİ.

    NEDEN KOPYA (import DEĞİL): `tools/train_cox_week3.py` bu görev
    sırasında (2026-08-18) AYRI bir üretim koşusu tarafından ÇALIŞIYOR
    durumda (`--variants v2b_mgmt_spline` sürüyor, `v2c_mgmt_spline_
    wttc` sırada) -- bu modülün ondan import etmesi, bu iki dosya
    arasında YENİ bir bağımlılık yönü kurar ve gelecekte o dosyada
    yapılacak bir değişikliğin bu modülü SESSİZCE bozma riskini
    doğurur. `pipeline/reduce_collinearity.py`'nin `pipeline/cox_
    model.py`/`tools/train_cox_week3.py`'den TAMAMEN BAĞIMSIZ
    çalışabilmesi bilinçli bir tasarım tercihidir. Davranışsal
    eşdeğerlik `tests/test_reduce_collinearity.py::
    test_correlation_clusters_union_find_matches_train_cox_week3_
    implementation` ile AYNI sentetik veri üzerinde doğrulanır --
    ikisi ayrışırsa bu test YAKALAR. Gelecekte `tools/train_cox_
    week3.py` sakin bir döneme girince bu iki fonksiyonun TEK bir
    ortak yardımcıya (`pipeline/cox_model.py` ya da yeni bir ortak
    modül) BİRLEŞTİRİLMESİ ÖNERİLİR -- şu an YAPILMADI (risk/fayda
    dengesi -- hot dosyayı bu görev sırasında minimum değiştirmek
    tercih edildi).
    """

    corr = frame[feature_columns].corr().abs()
    n = len(feature_columns)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if corr.iloc[i, j] >= corr_threshold:
                union(i, j)

    roots = [find(i) for i in range(n)]
    unique_roots = sorted(set(roots))
    remap = {root: cluster_id for cluster_id, root in enumerate(unique_roots)}
    return {feature_columns[i]: remap[roots[i]] for i in range(n)}


def select_cluster_representatives(
    clusters: dict[str, int], variability_summary: pd.DataFrame
) -> tuple[list[str], pd.DataFrame]:
    """Her korelasyon kümesinden TEK bir temsilci seçer -- tercih kuralı:
    kümedeki en BÜYÜK HAM std'ye sahip üye (basit, deterministik,
    tekrarlanabilir; bkz. AÇIK TASARIM NOTU aşağıda).

    🔴 AÇIK TASARIM NOTU (Barış'a soru, KİLİTLENMEDİ): "en büyük std"
    kuralı bu modülün yazarının seçimidir -- tek doğru seçenek DEĞİL.
    Alternatifler: (a) önceki bir arm'ın (v1/v2a) bootstrap seçilme-
    frekansına göre seçmek (proje zaten "bootstrap selection-frequency
    filtering" metodolojisini kullanıyor) -- REDDEDİLDİ, çünkü bu HANGİ
    arm'ın frekansının kullanılacağına bağlı POST-HOC bir bağımlılık
    kurar ve havuz tanımı dolaylı olarak bir önceki modelin SONUCUNA
    bağımlı hale gelir (CLAUDE.md 2026-08-18 revizyonu post-hoc
    değişikliği YASAKLAMIYOR ama bu modül X-ONLY/sonuç-bağımsız kalmayı
    tercih etti, daha temiz ve tekrarlanabilir); (b) alfabetik/ilk-
    görülen -- keyfi, gerekçesiz, REDDEDİLDİ. "En büyük std" kuralı
    HALA ölçek-bağımlı bir kıstas ama SADECE aynı-kümedeki (zaten
    |r|>=eşik ile "aynı bilgiyi taşıyan") üyeler arasında bir TIE-BREAK
    olarak kullanılıyor, nihai model FİTİ İÇİN değil.

    Döndürür
    --------
    (representatives, dropped_report)
        `representatives`: alfabetik sıralı, kümelerden seçilmiş nihai
        özellik listesi (tekil kümeler + çoklu-üyeli kümelerin
        temsilcileri).
        `dropped_report`: düşürülen HER özellik için hangi temsilcinin
        yerine geçtiği, hangi kümede olduğu -- SESSİZCE atılmaz, tam
        izlenebilir (CLAUDE.md "sessiz filtreleme yasak" ilkesi).
    """

    cluster_members: dict[int, list[str]] = {}
    for feature, cluster_id in clusters.items():
        cluster_members.setdefault(cluster_id, []).append(feature)

    representatives: list[str] = []
    dropped_rows: list[dict[str, object]] = []
    for cluster_id, members in cluster_members.items():
        if len(members) == 1:
            representatives.append(members[0])
            continue
        best = max(members, key=lambda feature: variability_summary.loc[feature, "std"])
        representatives.append(best)
        for member in members:
            if member != best:
                dropped_rows.append(
                    {
                        "dropped_feature": member,
                        "kept_representative": best,
                        "cluster_id": cluster_id,
                        "cluster_size": len(members),
                    }
                )

    dropped_report = pd.DataFrame(
        dropped_rows, columns=["dropped_feature", "kept_representative", "cluster_id", "cluster_size"]
    )
    return sorted(representatives), dropped_report


@dataclass
class V3CandidatePoolReport:
    """`build_v3_candidate_pool()`'un tam denetim izi -- kaç özellik
    girdi olarak geldi, kaçı hangi GEREKÇEYLE düşürüldü, kaçı kaldı."""

    n_input_features: int
    cv_threshold: float
    corr_threshold: float
    near_constant_dropped: list[str]
    correlation_dropped: pd.DataFrame  # dropped_feature/kept_representative/cluster_id/cluster_size
    n_multi_member_clusters: int
    kept_features: list[str]
    variability_summary: pd.DataFrame

    @property
    def n_kept_features(self) -> int:
        return len(self.kept_features)

    @property
    def n_dropped_total(self) -> int:
        return self.n_input_features - self.n_kept_features


def build_v3_candidate_pool(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    cv_threshold: float = DEFAULT_CV_THRESHOLD,
    corr_threshold: float = DEFAULT_CORRELATION_CLUSTER_THRESHOLD,
) -> tuple[list[str], V3CandidatePoolReport]:
    """v3'ün radyomik ADAY HAVUZUNU üretir -- İKİ AŞAMALI, SIRALI filtre:

      1. `select_near_constant_features()` -- CV < `cv_threshold` olan
         (gerçekten near-constant) özellikler ÇIKARILIR.
      2. `correlation_clusters_union_find()` + `select_cluster_
         representatives()` -- KALAN özellikler |r| >= `corr_threshold`
         ile kümelenir, her kümeden TEK temsilci kalır.

    `frame` SADECE özellik matrisidir (X) -- `survival_days`/`event`
    bu fonksiyona hiç verilmez (bkz. modül docstring'indeki kapsam notu:
    X-only, ICC filtresiyle AYNI mimari kademe).

    ⚠️ Bu fonksiyon `STABLE_FEATURES_ICC60`'IN YERİNE GEÇMEZ -- ICC
    filtresi ZATEN uygulanmış bir havuzu (örn. WT×93) GİRDİ olarak alır,
    ONU DARALTIR. `pipeline.cox_model.STABLE_FEATURES_ICC60` bu
    fonksiyona `feature_columns` olarak VERİLMELİDİR, burada YENİDEN
    hesaplanmaz.

    ⚠️ GİRDİ DOĞRULAMASI (Codex şartlı onay eki, 2026-09-11): `frame[
    feature_columns]` NaN/±inf içeriyorsa `NonFiniteFeatureValueError`
    ile fail-loud durur -- bkz. o istisnanın docstring'i (dejenere bir
    sütunun CV/korelasyon filtresinin HER İKİ aşamasını da sessizce
    atlatabileceği risk).
    """

    _assert_finite_features(frame, feature_columns)
    variability_summary = compute_feature_variability_summary(frame, feature_columns)
    near_constant = select_near_constant_features(variability_summary, cv_threshold=cv_threshold)

    survivors = [column for column in feature_columns if column not in set(near_constant)]

    clusters = correlation_clusters_union_find(frame, survivors, corr_threshold=corr_threshold)
    representatives, correlation_dropped = select_cluster_representatives(
        clusters, variability_summary
    )

    cluster_sizes = pd.Series(clusters).value_counts()
    n_multi_member_clusters = int((cluster_sizes > 1).sum())

    report = V3CandidatePoolReport(
        n_input_features=len(feature_columns),
        cv_threshold=cv_threshold,
        corr_threshold=corr_threshold,
        near_constant_dropped=near_constant,
        correlation_dropped=correlation_dropped,
        n_multi_member_clusters=n_multi_member_clusters,
        kept_features=representatives,
        variability_summary=variability_summary,
    )
    return representatives, report
