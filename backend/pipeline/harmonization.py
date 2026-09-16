"""GBM-AID görüntü harmonizasyonu için ortak sözleşme.

Değişmez işlem sırası:

1. N4 ve kaynak-bazlı Z-score görüntü uzayında çalışır.
2. Hazır segmentasyon maskesiyle PyRadiomics özellikleri çıkarılır.
3. ComBat/neuroHarmonize yalnız radyomik özellik uzayında çalışır.

UPenn radyomikleri hazırdır; UPenn görüntülerinde PyRadiomics çalıştırılmaz.
nnU-Net yalnızca hazır maskesi olmayan yeni-hasta demo akışının fallback'idir.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pipeline.lazy_sitk import sitk  # 2026-09-16: TEMBEL import (SAC blogu) -- bkz. pipeline/lazy_sitk.py


PROJECT_ROOT = Path(__file__).absolute().parents[2]  # 2026-09-16: backend/ altina tasindi -> bir seviye daha yukari
# ------------------------------------------------------------------
# Kaynak kanonikleştirme -- 2026-09-14'te `pipeline/source_canonical.py`'ye
# TAŞINDI (B1, Barış onayı: "b1 i uygula, sac a dokunma").
# ------------------------------------------------------------------
# `SOURCE_ALIASES`, `canonical_source()` ve `_canonical_source_series()`
# eskiden BU dosyada tanımlıydı. Üçü de saf string/pandas işi yapar,
# görüntüye hiç dokunmaz -- ama bu modül yukarıda `import SimpleITK as
# sitk` yaptığı için, onları almak isteyen her modül (özellikle
# `pipeline/cox_model.py` → `api/predict.py` → tüm servis yolu) 104 MB'lık
# bir görüntü kütüphanesine bağımlı hâle geliyordu. 2026-09-14'te Windows
# Smart App Control `_SimpleITK.cp310-win_amd64.pyd`'yi engelleyince bu
# gizli bağımlılık TÜM tahmin yolunu düşürdü.
#
# Gerekçenin tamamı + ölçümler: `pipeline/source_canonical.py` docstring'i.
#
# ⚠️ AŞAĞIDAKİ YENİDEN İHRAÇ KALDIRILMAMALI: `pipeline/segmentation.py`,
# `tools/*.py` ve testler hâlâ `from pipeline.harmonization import
# canonical_source` yazıyor ve bu ÇALIŞMAYA DEVAM ETMELİ.
# ⚠️ `SOURCE_ALIASES` AYNI SÖZLÜK NESNESİDİR (kopya değil) --
# `tests/test_harmonization.py` onu yerinde mutasyona uğratıp geri alıyor
# (sentetik "future-cohort" enjeksiyonu, ~satır 1084). Nesne kimliği
# korunmasaydı o test SESSİZCE anlamsızlaşırdı.
from pipeline.source_canonical import (  # noqa: F401  (yeniden ihraç)
    SOURCE_ALIASES,
    _canonical_source_series,
    canonical_source,
)


class ZScoreStatisticsNotFoundError(RuntimeError):
    """İstenen kaynak için fit edilmiş Z-score istatistiği yok."""


# ------------------------------------------------------------------
# Z-score KAPSAM (scope) sözleşmesi -- 2026-09-13, karar 28
# ------------------------------------------------------------------
# C32 sözleşmesi (CLAUDE.md, 2026-08-13 kilitli kararı) Z-score'un
# **T1ce-ÖZEL (modalite filtreli) kaynak** istatistiğinden hesaplanmasını
# ZORUNLU kılar; **kaynak-havuzlanmış (tüm modaliteler birlikte)** Z-score
# AÇIKÇA YASAKTIR.
#
# SOMUT RİSK (ölçüldü, 2026-09-13): `artifacts/week2/zscore/
# source_stats.json` (12 Ağustos) HAVUZLANMIŞ bir istatistik taşıyor
# (`image_count`: TCGA 154 / UPenn 2684 / LUMIERE 2152 -- tüm modaliteler)
# ve diskte CANLI duruyor. C32 üretim dosyaları
# (`artifacts/week3/pyradiomics/source_stats_t1ce_c32_*.json`,
# `image_count`: TCGA 39 / UPenn 611 / LUMIERE 580 / UCSF 295) ile
# YAPISAL OLARAK AYIRT EDİLEMEZ: ikisinde de aynı `version`/`fit_stage`/
# `sources` anahtarları ve aynı record alanları var. Hangi dosyanın
# kullanılacağını YALNIZ `GBMAID_ZSCORE_STATS_PATH` belirliyor -- yani
# bir yeni-hasta çıkarımı, modelin eğitildiği uzaydan FARKLI bir Z-score
# uzayında normalize edilebilir ve bu SESSİZCE olur.
#
# Bu yüzden kapsam artık **içerikten POZİTİF olarak BEYAN EDİLMEK
# ZORUNDA**: stats payload'ının kökünde `"zscore_scope"` alanı olmalı ve
# beklenen kapsama eşit olmalı. Beyan YOKSA da hata verilir (fail-closed)
# -- "beyan edilmemiş = muhtemelen doğrudur" varsayımı YASAK (CLAUDE.md
# "sessiz fallback yasak"). Sezgisel (heuristic) tahmin YAPILMAZ:
# `image_count`/kaynak sayısına bakıp "bu T1ce olsa gerek" demek tam
# olarak sessiz varsayımdır.
ZSCORE_SCOPE_KEY = "zscore_scope"
# C32 sözleşmesinin TEK meşru kapsamı.
ZSCORE_SCOPE_T1CE_SOURCE = "t1ce_source"
# `artifacts/week2/zscore/source_stats.json`'ın (Hafta 2 tarihsel kaydı)
# taşıdığı YASAK kapsam etiketi.
ZSCORE_SCOPE_POOLED_ALL_MODALITIES = "source_pooled_all_modalities"


class ZScoreScopeMismatchError(RuntimeError):
    """Yüklenen Z-score istatistik dosyasının kapsamı C32 ile uyuşmuyor.

    İki ayrı durumu kapsar (ikisi de FAIL-CLOSED):
    1. Dosya bir kapsam BEYAN EDİYOR ama beklenenden farklı (örn.
       havuzlanmış `source_pooled_all_modalities`).
    2. Dosya HİÇ kapsam beyan etmiyor -- doğrulanamayan bir istatistikle
       normalize etmek yasak.
    """


def _require_zscore_scope(
    payload: Any, stats_path: Path, expected_scope: str
) -> str:
    """Stats payload'ının beyan ettiği kapsamı doğrula, yoksa/uyuşmuyorsa PATLA.

    Döndürür: doğrulanmış (beklenene eşit) kapsam etiketi.
    """

    declared = payload.get(ZSCORE_SCOPE_KEY) if isinstance(payload, dict) else None

    if declared is None:
        raise ZScoreScopeMismatchError(
            f"Z-score istatistik dosyası kapsam BEYAN ETMİYOR: {stats_path}. "
            f"C32 sözleşmesi kökte '{ZSCORE_SCOPE_KEY}': "
            f"'{expected_scope}' alanını ZORUNLU kılar (CLAUDE.md 2026-08-13 "
            "kilitli kararı: T1ce-ÖZEL kaynak Z-score; kaynak-havuzlanmış "
            "Z-score YASAK). Beyan olmadan bu dosyanın hangi modalite "
            "kümesinden fit edildiği DOĞRULANAMAZ -- sessizce 'doğrudur' "
            "varsayılmaz, normalizasyon DURDURULDU.\n"
            "DÜZELTME: (a) doğru dosyayı gösterin "
            "(`GBMAID_ZSCORE_STATS_PATH`), veya (b) dosyayı üreten koşunun "
            "GERÇEKTEN yalnız T1ce görüntülerinden fit edildiğini insan "
            f"gözüyle doğrulayıp payload köküne '{ZSCORE_SCOPE_KEY}': "
            f"'{expected_scope}' alanını ekleyin, veya (c) "
            f"`fit_source_zscore_statistics(..., scope='{expected_scope}')` "
            "ile yeniden fit edin."
        )

    if declared != expected_scope:
        pooled_hint = ""
        if declared == ZSCORE_SCOPE_POOLED_ALL_MODALITIES:
            pooled_hint = (
                " Bu dosya TÜM MODALİTELER havuzlanarak fit edilmiş "
                "(Hafta 2, 2026-08-12) tarihsel bir kayıttır ve C32'de "
                "KULLANILMASI YASAKTIR -- C32 üretim dosyaları "
                "`artifacts/week3/pyradiomics/source_stats_t1ce_c32_*.json`."
            )
        raise ZScoreScopeMismatchError(
            f"Z-score istatistik kapsamı uyuşmuyor: {stats_path} "
            f"'{ZSCORE_SCOPE_KEY}': {declared!r} beyan ediyor, beklenen "
            f"{expected_scope!r} (C32 sözleşmesi)." + pooled_hint
        )

    return declared


# NOT (2026-09-14, B1): `canonical_source()` ve `_canonical_source_series()`
# gövdeleri BU DOSYADAN KALDIRILDI -- tanımları artık
# `pipeline/source_canonical.py`'de. Yukarıdaki yeniden ihraç sayesinde
# `from pipeline.harmonization import canonical_source` ÇALIŞMAYA DEVAM EDER.
#
# Gövdeler bilinçli olarak KOPYA BIRAKILMADI: kodda ölü ikiz tanım
# tutmak, birinin yanlış kopyayı düzenlemesine yol açan bir drift
# riskidir (hard rule #3 wiki metinleri içindir, çalışan kod için değil).
# Değişiklik öncesi dosyanın birebir hâli arşivde:
#   archive/2026-09-14-b1-simpleitk-ayristirma-yedek/harmonization.py.oncesi
#   (sha256 e79029267abb7ef0a2a6c7fec139e19903548c6183e3eb4a66276d3d4003edfb)


def _require_nifti(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI dosyası bulunamadı: {path}")
    lower_name = path.name.lower()
    if not (lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
        raise ValueError(f"Yalnızca NIfTI girdisi desteklenir: {path}")
    return path


def _nifti_stem(path: Path) -> str:
    if path.name.lower().endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def _processed_root() -> Path:
    configured = os.environ.get("GBMAID_PROCESSED_ROOT")
    if configured:
        return Path(configured).expanduser()
    return PROJECT_ROOT / "artifacts" / "harmonized"


def _derived_output_path(input_path: Path, stage: str) -> Path:
    path_key = str(input_path.absolute()).casefold().encode("utf-8")
    input_id = hashlib.sha256(path_key).hexdigest()[:12]
    output_dir = _processed_root() / input_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{_nifti_stem(input_path)}_{stage}.nii.gz"


def write_image_atomic(image: sitk.Image, output_path: str | Path) -> Path:
    """NIfTI çıktısını yarım dosya bırakmadan atomik biçimde yaz."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{_nifti_stem(output)}.{uuid.uuid4().hex}.nii.gz"
    )
    try:
        sitk.WriteImage(image, str(temporary), True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _n4_iterations() -> list[int]:
    raw_value = os.environ.get("GBMAID_N4_ITERATIONS", "50,50,30,20")
    try:
        iterations = [int(item.strip()) for item in raw_value.split(",")]
    except ValueError as exc:
        raise ValueError(
            "GBMAID_N4_ITERATIONS virgülle ayrılmış pozitif tamsayılar olmalı."
        ) from exc
    if not iterations or any(value <= 0 for value in iterations):
        raise ValueError("N4 iterasyon değerleri pozitif olmalı.")
    return iterations


def _foreground_values(image: sitk.Image) -> np.ndarray:
    array = sitk.GetArrayViewFromImage(image)
    foreground = np.isfinite(array) & (array != 0)
    if not np.any(foreground):
        raise ValueError("Görüntüde sonlu, sıfır-dışı foreground voxel bulunamadı.")
    return np.asarray(array[foreground], dtype=np.float64)


def _zscore_stats_path() -> Path:
    configured = os.environ.get("GBMAID_ZSCORE_STATS_PATH")
    if configured:
        return Path(configured).expanduser()
    return PROJECT_ROOT / "artifacts" / "zscore" / "source_stats.json"


def apply_n4_bias_correction(image_path: str) -> str:
    """SimpleITK N4 bias field correction uygula ve çıktı yolunu döndür.

    Ham görüntü değiştirilmez. Çıktı ``GBMAID_PROCESSED_ROOT`` altında,
    girdi yolundan türetilmiş deterministik bir klasöre atomik olarak yazılır.
    """

    input_path = _require_nifti(image_path)
    image = sitk.Cast(sitk.ReadImage(str(input_path)), sitk.sitkFloat32)

    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    if int(sitk.GetArrayViewFromImage(mask).sum()) == 0:
        raise ValueError(f"N4 foreground maskesi boş: {input_path}")

    try:
        shrink_factor = int(os.environ.get("GBMAID_N4_SHRINK_FACTOR", "4"))
    except ValueError as exc:
        raise ValueError("GBMAID_N4_SHRINK_FACTOR pozitif tamsayı olmalı.") from exc
    if shrink_factor <= 0:
        raise ValueError("GBMAID_N4_SHRINK_FACTOR pozitif olmalı.")

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(_n4_iterations())
    corrector.SetConvergenceThreshold(1e-7)

    if shrink_factor > 1:
        factors = [shrink_factor] * image.GetDimension()
        fit_image = sitk.Shrink(image, factors)
        fit_mask = sitk.Shrink(mask, factors)
    else:
        fit_image = image
        fit_mask = mask

    corrector.Execute(fit_image, sitk.Cast(fit_mask, sitk.sitkUInt8))
    log_bias_field = corrector.GetLogBiasFieldAsImage(image)
    corrected = sitk.Cast(image / sitk.Exp(log_bias_field), sitk.sitkFloat32)
    corrected.CopyInformation(image)

    output_path = _derived_output_path(input_path, "n4")
    return str(write_image_atomic(corrected, output_path))


def n4_output_path(image_path: str | Path) -> Path:
    """`apply_n4_bias_correction()`'ın üreteceği N4 çıktı yolunu, GERÇEKTEN
    HESAPLAMADAN (hiçbir I/O yapmadan) döndür.

    2026-08-13 H4 düzeltmesi: çağıranların (örn. `tools/run_pyradiomics_*.py`)
    pahalı N4 hesaplamasını tekrar çalıştırmadan ÖNCE "bu çıktı zaten diskte
    var mı" diye sorabilmesi için eklendi -- restart/resume senaryosunda
    Pass 1'in tüm kohort üzerinde yeniden N4 hesaplaması yapmasını önler
    (bkz. decisions/2026-08-13-pyradiomics-c32-bincount-karari.md "[2026-08-13]
    Review-gate BLOKE düzeltmeleri" bölümü, bulgu B1/H4).

    `apply_n4_bias_correction()` ile AYNI `_derived_output_path()`
    mantığını kullanır (kod tekrarı/drift riski YOK, tek bir kaynak) --
    `apply_n4_bias_correction()`'ın kendi davranışı bu fonksiyonla
    DEĞİŞTİRİLMEDİ, bu SALT-EKLEME (purely additive) bir yardımcıdır.
    Dosyanın GERÇEKTEN var olup olmadığını veya geçerli olup olmadığını
    KONTROL ETMEZ -- yalnız yolu hesaplar, çağıran taraf `Path.is_file()`
    (ve varsa kendi geçerlilik kontrolünü) ayrıca yapmalı.
    """

    input_path = _require_nifti(image_path)
    return _derived_output_path(input_path, "n4")


def has_fitted_zscore_statistics(
    source: str, *, stats_path: str | Path | None = None
) -> bool:
    """Verilen kaynak için stats dosyasında ZATEN geçerli bir Z-score
    istatistiği kayıtlı mı kontrol et -- yalnız OKUR, hiçbir şey YAZMAZ.

    2026-08-13 B3 düzeltmesi: `fit_source_zscore_statistics_if_needed()`'in
    "zaten var mı" kontrolü bunun üzerine kurulu. Bozuk/eksik/sonsuz
    değerli bir kayıt "geçerli DEĞİL" (False) sayılır -- sessizce
    güvenilmeyip yeniden fit'e izin vermek için.
    """

    canonical = canonical_source(source)
    destination = Path(stats_path) if stats_path else _zscore_stats_path()
    if not destination.is_file():
        return False
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
        record = payload["sources"][canonical]
        mean = float(record["mean"])
        standard_deviation = float(record["std"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return math.isfinite(mean) and math.isfinite(standard_deviation) and standard_deviation > 1e-12


def get_fitted_zscore_record(
    source: str, *, stats_path: str | Path | None = None
) -> dict[str, float | int | str] | None:
    """Kaynağa ait, stats dosyasında kayıtlı fit RECORD'unu (varsa) döndür.

    2026-08-14 A1.1 düzeltmesi (Codex review TUR 2 -- "image_count gate"
    bulgusu, bkz. decisions/2026-08-13-pyradiomics-c32-bincount-karari.md
    Riskler #1): `has_fitted_zscore_statistics()` yalnız bir bool
    döndürüyor (mean/std sonlu mu) -- ama parçalı koşularda (`--offset`/
    `--limit`) B3 guard'ının dondurduğu istatistiğin HANGİ örneklemden
    geldiğini (`image_count`) çağıranın görebilmesi gerekiyor: donan
    istatistik yalnızca 50 taramalık bir alt-kümeden gelmişse, tam
    kohort (örn. 611/599/154) için sessizce kullanılmamalı -- çağıran
    taraf (`tools/run_pyradiomics_*.py`) bunu bir "beklenen tam kohort
    image_count" değeriyle karşılaştırıp uyuşmuyorsa Pass 3'e GEÇMEMELİ.

    Salt-okur, hiçbir şey YAZMAZ. Bozuk/eksik/geçersiz bir kayıt
    (`has_fitted_zscore_statistics()` ile AYNI geçerlilik kriteri) için
    `None` döner -- sessizce eski/yanlış bir dict UYDURMAZ.
    """

    canonical = canonical_source(source)
    destination = Path(stats_path) if stats_path else _zscore_stats_path()
    if not destination.is_file():
        return None
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
        record = payload["sources"][canonical]
        mean = float(record["mean"])
        standard_deviation = float(record["std"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not (
        math.isfinite(mean)
        and math.isfinite(standard_deviation)
        and standard_deviation > 1e-12
    ):
        return None
    return record


def fit_source_zscore_statistics_if_needed(
    image_paths: Iterable[str | Path],
    source: str,
    *,
    stats_path: str | Path | None = None,
    force: bool = False,
    scope: str | None = None,
) -> dict[str, float | int | str] | None:
    """B3 restart-guard: SADECE gerekiyorsa fit et, aksi halde MEVCUT
    istatistiği REUSE et (yeniden fit ETMEZ).

    KRİTİK BULGU (2026-08-13 review-gate, bkz. decisions/2026-08-13-
    pyradiomics-c32-bincount-karari.md): `tools/run_pyradiomics_*.py`
    script'lerinin ÜÇÜ de Pass 2'de KOŞULSUZ `fit_source_zscore_statistics()`
    çağırıyordu. Bir koşu kısmen bitip (örn. kohortun 300/611'i işlenip)
    yeniden başlatıldığında, restart'ın Pass 1'i yalnız KALAN (henüz
    `--report`'ta OK/YAZILDI olmayan) taramaları işler -- bu yüzden
    restart'ın Pass 2'si yalnız o KALAN alt-kümenin N4 çıktılarından YENİ
    bir `(mean,std)` fit edip `payload.setdefault("sources", {})[canonical]
    = record` deseniyle kaynağın TEK kanonik istatistiğinin ÜZERİNE
    SESSİZCE yazıyordu. Sonuç: aynı kaynağın hastaları HANGİ partide
    işlendiğine göre FARKLI Z-score istatistiğiyle normalize ediliyordu --
    "kaynak-bazlı tek istatistik, tüm kohorta uniform" önermesi (C32
    kararının temeli) sessizce ihlal ediliyordu. Hata fırlamıyordu, log'a
    düşmüyordu.

    Düzeltme (bu fonksiyon): `force=False` (varsayılan) iken, stats
    dosyasında `source` için ZATEN geçerli bir kayıt varsa fit ATLANIR,
    mevcut değer -- ve dolayısıyla `apply_zscore_normalization()`'ın
    kullanacağı `(mean,std)` -- DEĞİŞMEDEN kalır. İlk (kesintisiz) koşuda
    stats dosyası henüz yoktur, bu yüzden Pass 1'in TOPLADIĞI TÜM N4
    çıktılarından (o koşunun gördüğü tüm kohort) TEK seferlik bir fit
    yapılır -- sonraki HER restart bu DONMUŞ değeri paylaşır. Bilinçli
    ödünleşim: eğer ilk koşu da kesintiye uğrarsa (örn. sadece 300/611
    işlenip kesilirse VE o kesintiden önce hiç Pass 2 çalışmadıysa),
    donan istatistik TAM kohort yerine bu ilk kısmi alt-kümeden gelir --
    ama bu YİNE DE "kaynağın TÜM hastaları AYNI istatistikle normalize
    edilir" tutarlılık garantisini KORUR (istatistiğin hangi ÖRNEKLEMDEN
    geldiği ayrı bir istatistiksel-kalite sorusu, `fit_record`'un
    `image_count`/`voxel_count` alanlarına bakılarak operatör tarafından
    denetlenebilir). Operatör kasıtlı olarak DAHA GÜNCEL/DAHA BÜYÜK bir
    örneklemle yeniden fit etmek isterse `force=True` (script'lerdeki
    `--fresh-fit` bayrağı) AÇIKÇA verilmeli -- örtük/otomatik refit YOK.

    Döndürür
    --------
    dict veya None
        Fit GERÇEKTEN çalıştırıldıysa `fit_source_zscore_statistics()`'in
        döndürdüğü record; `force=False` VE zaten geçerli bir kayıt
        VARSA fit ATLANDI demektir, `None` döner (çağıran bunu "mevcut
        istatistik kullanılmaya devam ediyor" olarak yorumlamalı).

    Hata
    ----
    ZScoreStatisticsNotFoundError
        Ne mevcut geçerli bir istatistik VAR ne de fit edilecek `image_paths`
        VERİLMİŞ (`force=False` durumunda) -- Pass 3 zaten çalışamayacağı
        için burada erken ve açık hata verilir (sessiz "hiçbir şey yapma"
        YASAK).
    ValueError
        `force=True` verildi ama `image_paths` boş -- zorla fit edilecek
        veri yok.
    """

    canonical = canonical_source(source)
    if not force and has_fitted_zscore_statistics(canonical, stats_path=stats_path):
        # Kapsam guard'ını BURADA da uygula (2026-09-13, karar 28):
        # restart senaryosunda fit ATLANDIĞI için dosyanın kapsamı hiç
        # kontrol edilmeden Pass 3'e geçilirdi ve hata ancak ilk
        # `apply_zscore_normalization()` çağrısında (tarama tarama)
        # çıkardı. Burada erken patlamak, yanlış dosyayla koşuya
        # BAŞLAMAYI engeller.
        if scope is not None:
            destination = Path(stats_path) if stats_path else _zscore_stats_path()
            payload = json.loads(destination.read_text(encoding="utf-8"))
            _require_zscore_scope(payload, destination, scope)
        return None

    image_paths = list(image_paths)
    if not image_paths:
        if force:
            raise ValueError(
                f"{canonical} için force=True (fresh-fit) istendi ama fit "
                "edilecek hiç N4 çıktısı verilmedi."
            )
        raise ZScoreStatisticsNotFoundError(
            f"{canonical} için ne mevcut geçerli bir Z-score istatistiği "
            "var ne de fit edilecek yeni N4 çıktısı -- Pass 3 "
            "çalıştırılamaz (stats_path="
            f"{Path(stats_path) if stats_path else _zscore_stats_path()!s})."
        )
    return fit_source_zscore_statistics(
        image_paths, canonical, stats_path=stats_path, scope=scope
    )


def fit_source_zscore_statistics(
    image_paths: Iterable[str | Path],
    source: str,
    *,
    stats_path: str | Path | None = None,
    scope: str | None = None,
) -> dict[str, float | int | str]:
    """N4 çıktılarından bir kaynağa ait cohort Z-score istatistiğini fit et.

    İstatistik yalnız sonlu ve sıfır-dışı foreground voxel'lerden, görüntüler
    sırayla okunarak hesaplanır. Bu fonksiyonun girdileri N4 çıktıları olmalıdır.

    `scope` (2026-09-13, karar 28)
    ------------------------------
    Verilirse payload köküne `"zscore_scope": <scope>` olarak YAZILIR ve
    `apply_zscore_normalization()`'ın kapsam guard'ı bunu okur. C32
    çıkarımında `scope="t1ce_source"` (`ZSCORE_SCOPE_T1CE_SOURCE`)
    GEÇİLMELİDİR -- `tools/run_pyradiomics_*.py` bunu kendi `ZSCORE_SCOPE`
    sabitiyle yapar.

    ⚠️ VARSAYILAN BİLİNÇLİ OLARAK `None`'dır, `"t1ce_source"` DEĞİL. Bu
    fonksiyon kendisine verilen görüntülerin HANGİ MODALİTEDEN olduğunu
    BİLEMEZ (yalnız bir yol listesi alır). Varsayılanı `"t1ce_source"`
    yapmak, tüm modaliteleri havuzlayan bir çağıranın (örn.
    `tools/run_harmonization_cohort.py`, 2026-08-12'deki havuzlanmış
    `artifacts/week2/zscore/source_stats.json`'ı üreten script) çıktısını
    SESSİZCE ve YANLIŞ biçimde "T1ce" diye etiketlemesine yol açardı --
    guard'ı tamamen işlevsizleştirirdi. `scope=None` ile yazılan dosya
    kapsam beyan etmez, `apply_zscore_normalization()` onu REDDEDER
    (fail-closed) -- doğru davranış budur.

    Aynı dosyada KAPSAM KARIŞTIRMA yasaktır: dosyada zaten farklı bir
    `zscore_scope` beyanı varsa `ZScoreScopeMismatchError` fırlatılır
    (havuzlanmış bir dosyaya T1ce record'u eklemek ya da tersi
    engellenir).
    """

    canonical = canonical_source(source)
    count = 0
    mean = 0.0
    m2 = 0.0
    image_count = 0

    for image_path in image_paths:
        path = _require_nifti(image_path)
        values = _foreground_values(sitk.ReadImage(str(path)))
        batch_count = int(values.size)
        batch_mean = float(values.mean(dtype=np.float64))
        batch_m2 = float(
            np.square(values - batch_mean, dtype=np.float64).sum(dtype=np.float64)
        )

        combined_count = count + batch_count
        delta = batch_mean - mean
        mean += delta * batch_count / combined_count
        m2 += batch_m2 + delta * delta * count * batch_count / combined_count
        count = combined_count
        image_count += 1

    if image_count == 0 or count < 2:
        raise ValueError("Z-score fit için en az bir dolu görüntü ve iki voxel gerekir.")

    standard_deviation = math.sqrt(m2 / count)
    if not math.isfinite(standard_deviation) or standard_deviation <= 1e-12:
        raise ValueError(
            f"{canonical} için hesaplanan standart sapma geçersiz: "
            f"{standard_deviation}"
        )

    record: dict[str, float | int | str] = {
        "source": canonical,
        "mean": mean,
        "std": standard_deviation,
        "voxel_count": count,
        "image_count": image_count,
        "foreground_rule": "finite_nonzero",
    }

    destination = Path(stats_path) if stats_path else _zscore_stats_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        payload = json.loads(destination.read_text(encoding="utf-8"))
    else:
        payload = {
            "version": 1,
            "fit_stage": "post_n4",
            "sources": {},
        }

    # Kapsam karıştırma engeli (2026-09-13, karar 28): aynı dosyada
    # farklı kapsamlı record'lar birikirse guard'ın anlamı kalmaz.
    existing_scope = payload.get(ZSCORE_SCOPE_KEY)
    if scope is not None:
        if existing_scope is not None and existing_scope != scope:
            raise ZScoreScopeMismatchError(
                f"{destination} dosyası '{ZSCORE_SCOPE_KEY}': "
                f"{existing_scope!r} beyan ediyor ama bu fit "
                f"{scope!r} kapsamıyla yazmak istiyor -- aynı stats "
                "dosyasında kapsam karıştırmak YASAK (C32 sözleşmesi). "
                "Ayrı bir `--stats-path` kullanın."
            )
        payload[ZSCORE_SCOPE_KEY] = scope
    elif existing_scope is not None:
        raise ZScoreScopeMismatchError(
            f"{destination} dosyası '{ZSCORE_SCOPE_KEY}': "
            f"{existing_scope!r} beyan ediyor ama bu fit HİÇ kapsam "
            "vermedi (scope=None) -- beyanlı bir dosyaya beyansız record "
            "eklemek kapsam garantisini bozar. Çağırana açıkça "
            f"`scope={existing_scope!r}` geçin veya ayrı bir "
            "`--stats-path` kullanın."
        )

    payload.setdefault("sources", {})[canonical] = record

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def apply_zscore_normalization(
    image_path: str,
    source: str,
    *,
    expected_scope: str = ZSCORE_SCOPE_T1CE_SOURCE,
) -> str:
    """Fit edilmiş kaynak istatistiğiyle N4 görüntüsünü Z-score normalize et.

    KAPSAM GUARD'I (2026-09-13, karar 28 -- bkz. `ZScoreScopeMismatchError`
    ve yukarıdaki "Z-score KAPSAM sözleşmesi" bloğu): yüklenen stats
    dosyası kökünde `"zscore_scope": "t1ce_source"` BEYAN ETMEK
    ZORUNDADIR. Beyan yoksa ya da farklıysa (örn. havuzlanmış
    `source_pooled_all_modalities`) `ZScoreScopeMismatchError` fırlatılır
    -- SESSİZ DEVAM/fallback YOK, fail-closed. Guard, kaynak record'u
    okunmadan ÖNCE çalışır: yanlış kapsamlı bir dosyadan hiçbir
    (mean,std) çifti alınmaz.

    `expected_scope` parametresi yalnız test/gelecekteki bir sözleşme
    değişikliği için vardır; üretimde ASLA geçilmez (varsayılan
    `ZSCORE_SCOPE_T1CE_SOURCE`). Guard'ı gevşetmek için kullanılması
    C32 sözleşmesinin ihlalidir.
    """

    input_path = _require_nifti(image_path)
    canonical = canonical_source(source)
    stats_path = _zscore_stats_path()
    if not stats_path.is_file():
        raise ZScoreStatisticsNotFoundError(
            f"Kaynak Z-score istatistik dosyası bulunamadı: {stats_path}. "
            "Önce fit_source_zscore_statistics() çalıştırılmalı."
        )

    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    _require_zscore_scope(payload, stats_path, expected_scope)
    try:
        stats = payload["sources"][canonical]
        mean = float(stats["mean"])
        standard_deviation = float(stats["std"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ZScoreStatisticsNotFoundError(
            f"{canonical} için geçerli Z-score istatistiği yok: {stats_path}"
        ) from exc

    if not math.isfinite(mean) or not math.isfinite(standard_deviation):
        raise ValueError(f"{canonical} Z-score istatistikleri sonlu değil.")
    if standard_deviation <= 1e-12:
        raise ValueError(f"{canonical} Z-score standart sapması geçersiz.")

    image = sitk.Cast(sitk.ReadImage(str(input_path)), sitk.sitkFloat32)
    array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    foreground = np.isfinite(array) & (array != 0)
    if not np.any(foreground):
        raise ValueError(f"Z-score foreground maskesi boş: {input_path}")

    normalized = np.zeros(array.shape, dtype=np.float32)
    normalized[foreground] = (
        array[foreground] - np.float32(mean)
    ) / np.float32(standard_deviation)

    output_image = sitk.GetImageFromArray(normalized)
    output_image.CopyInformation(image)
    output_path = _derived_output_path(input_path, f"zscore_{canonical.lower()}")
    return str(write_image_atomic(output_image, output_path))


# ============================================================
# ComBat/neuroHarmonize -- özellik-uzayı harmonizasyonu
# ============================================================
#
# MİMARİ KURAL (CLAUDE.md + 2026-08-09 revizyonu, ihlal edilirse
# istatistiksel sızıntı olur):
# `fit_combat_harmonization()` SADECE UPenn+LUMIERE ile çağrılabilir.
# `apply_combat_harmonization()` da SADECE UPenn+LUMIERE için çağrılabilir.
# TCGA **ve UCSF** ComBat kapsamının TAMAMEN DIŞINDADIR -- ne fit'e ne
# apply'a girerler; ikisi de yalnızca görüntü-seviyesi N4+T1ce Z-score
# harmonizasyonu alır (TCGA: decisions/2026-08-09-combat-fit-iki-blokaj-
# region-ve-pyradiomics-parametre.md; UCSF: CLAUDE.md 2026-08-18 kuralı
# -- harici test setine eğitim-türevi dönüşüm uygulanmaz).
# `apply_combat_harmonization()` bunu, spoof edilebilen
# `covariates['SITE']`'a değil, zorunlu `true_dataset_source` parametresine
# bakarak kod-seviyesinde zorlar (bkz. aşağıdaki fonksiyonun docstring'i).
# Guard, `COMBAT_FIT_ALLOWED_SOURCES` whitelist'inin TÜMLEYENİ üzerinden
# çalışır (2026-09-13 karar 29) -- sabit bir blacklist DEĞİL.

REQUIRED_COVARIATE_COLUMN = "SITE"
COMBAT_FIT_ALLOWED_SOURCES = {"UPenn", "LUMIERE"}


class ComBatFitLeakageError(RuntimeError):
    """`fit_combat_harmonization()`'a yasak bir kaynak (örn. TCGA) sızdı."""


class ComBatApplyLeakageError(RuntimeError):
    """`apply_combat_harmonization()`'a ComBat fit'inde GÖRÜLMEMİŞ veri sızdı.

    2026-09-13 (karar 29) itibarıyla kapsam TCGA'dan `COMBAT_FIT_ALLOWED_
    SOURCES` (= {"UPenn","LUMIERE"}) TÜMLEYENİNE genişletildi: TCGA **ve
    UCSF** (ve `SOURCE_ALIASES`'e ileride eklenecek her yeni kaynak)
    reddedilir. UCSF gerekçesi: CLAUDE.md 2026-08-18 kuralı -- harici
    test seti, TCGA ile AYNI muamele, feature-level ComBat ALMAZ.

    `covariates['SITE']` spoof edilebilir (örn. TCGA satırları açıkça
    ``SITE="UPenn"`` etiketiyle çağrılabilir ve `apply_combat_harmonization()`
    bunu ayırt edemez) -- bu yüzden bu hata SITE kolonuna DEĞİL, ayrı ve
    spoof edilemez bir `true_dataset_source` parametresine bakarak fırlatılır.
    Bkz. 2026-08-09 kararı:
    decisions/2026-08-09-combat-fit-iki-blokaj-region-ve-pyradiomics-parametre.md
    (TCGA ComBat kapsamının tamamen dışında bırakılması).
    """


def _neuroharmonize_version() -> str:
    """neuroHarmonize paket sürümünü döndür.

    Not: `neuroHarmonize` modülü kendi `__version__` özniteliğini
    tanımlamıyor (2.5.1'de doğrulandı) -- bu yüzden paket metadata'sından
    okunur, doğrudan `nh.__version__` KULLANILMAZ (AttributeError verir).
    """

    import importlib.metadata

    try:
        return importlib.metadata.version("neuroHarmonize")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _validate_feature_matrix(feature_matrix: pd.DataFrame) -> None:
    if not isinstance(feature_matrix, pd.DataFrame):
        raise TypeError(
            "feature_matrix pandas.DataFrame olmalı "
            "(satır=hasta/scan, kolon=kanonik 107 özellik adı)."
        )
    if feature_matrix.empty:
        raise ValueError("feature_matrix boş olamaz.")
    numeric_values = feature_matrix.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "feature_matrix NaN/sonsuz değer içeriyor -- ComBat'tan önce "
            "eksik değer politikası netleşmeli, burada sessizce doldurulmaz."
        )


def _validate_covariates(
    feature_matrix: pd.DataFrame, covariates: pd.DataFrame
) -> None:
    if not isinstance(covariates, pd.DataFrame):
        raise TypeError("covariates pandas.DataFrame olmalı.")
    if REQUIRED_COVARIATE_COLUMN not in covariates.columns:
        raise ValueError(
            f"covariates '{REQUIRED_COVARIATE_COLUMN}' kolonu içermeli "
            "(ComBat batch/kaynak etiketi -- neuroHarmonize sözleşmesi)."
        )
    if not feature_matrix.index.equals(covariates.index):
        raise ValueError(
            "feature_matrix ve covariates aynı index'te (aynı hasta/scan "
            "sırasında) olmalı -- hizalama garantisi yok, sessizce "
            "hizalanmaz."
        )


def fit_combat_harmonization(
    feature_matrix: pd.DataFrame,
    covariates: pd.DataFrame,
    *,
    reference_batch: str = "UPenn",
    seed: int | None = 42,
    library_tag: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    """UPenn+LUMIERE üzerinde `harmonizationLearn()` ile ComBat fit et.

    Parametreler
    ------------
    feature_matrix : pandas.DataFrame
        Satır = hasta/scan, kolon = kanonik PyRadiomics özellik adı
        (örn. ``original_shape_Elongation``). Kaynaklar arası SAYISAL
        karşılaştırılabilirlik ÇAĞIRAN TARAFIN sorumluluğundadır -- bu
        fonksiyon isim eşleşmesini doğrular ama CaPTk/PyRadiomics gibi
        farklı çıkarım motorlarının aynı isimli-ama-farklı-anlamlı
        özellik ürettiği durumu YAKALAYAMAZ (bkz.
        [[2026-08-07-upenn-144-vs-107-uyumsuzlugu]]).
    covariates : pandas.DataFrame
        `feature_matrix` ile aynı index'te, en az bir ``SITE`` kolonu
        (kaynak etiketi: ``"UPenn"``/``"LUMIERE"``) içermeli. Ek
        kovaryatlar (örn. ``AGE``) sayısal olmalı.
    reference_batch : str
        Referans-batch adı, varsayılan ``"UPenn"`` (mimari kural: UPenn
        referans-batch).

    Döndürür
    --------
    (model, harmonized_in_sample, parameter_rows)
        ``model``: `apply_combat_harmonization()`'a geçirilecek ham
        neuroHarmonize model sözlüğü.
        ``harmonized_in_sample``: UPenn+LUMIERE için in-sample
        harmonize edilmiş özellik matrisi (XGBoost'a bu DEĞİL,
        out-of-fold Cox skorları gider -- bu sadece ComBat'ın kendi
        çıktısı, ayrı bir kavram, CLAUDE.md'nin Cox->XGBoost sızıntı
        kuralıyla karıştırılmamalı).
        ``parameter_rows``: `combat_parameters` tablosu şemasına
        (source_batch, feature_name, gamma, delta, grand_mean,
        pooled_variance, design_matrix_ref, fit_date, reference_batch,
        harmonization_library) birebir uyan, INSERT'e hazır dict listesi.

    Hata
    ----
    ComBatFitLeakageError
        ``covariates['SITE']`` içinde UPenn/LUMIERE dışında bir kaynak
        (örn. TCGA, TCGA'nın alias'ları veya `SOURCE_ALIASES`'te hiç
        tanımlı olmayan bilinmeyen bir ad) görülürse -- mimari sızıntı
        önleme kontrolü, WHITELIST mantığıyla çalışır: yalnızca kanonik
        ``"UPenn"``/``"LUMIERE"`` KABUL edilir, geri kalan HER ŞEY
        (TCGA dahil, tanınmayan yazım hataları dahil) reddedilir.

    Not (2026-08-12 FIX-3, bkz. decisions/2026-08-11-apply-combat-
    harmonization-true-source-guard.md "[2026-08-12] FIX-3"): bu guard
    önceden ``covariates['SITE']``'ı HAM literal string olarak
    ``COMBAT_FIT_ALLOWED_SOURCES = {"UPenn", "LUMIERE"}`` ile
    karşılaştırıyordu -- `dataset_sources.source_name` DB kolonunun
    gerçek değeri literal ``"UPenn-GBM"``/``"TCGA-GBM"`` olduğundan
    (bkz. `pipeline/cox_model.py` satır ~98), bu ham karşılaştırma
    MEŞRU UPenn verisini de yanlışlıkla reddederdi (fail-safe ama
    üretimi kıran bir hata). Guard artık `_canonical_source_series()`
    (aynı `SOURCE_ALIASES`/`canonical_source()` normalizasyonunu
    `apply_combat_harmonization()`'ın TCGA-reddiyle PAYLAŞIR) ile
    normalize edilmiş değerler üzerinden çalışır: ``"UPenn-GBM"``,
    ``"upenn-gbm"``, ``" UPenn-GBM "`` gibi varyantlar kanonik
    ``"UPenn"``'e eşlenip KABUL edilir; ``"TCGA"``, ``"TCGA-GBM"`` gibi
    tüm TCGA varyantları hâlâ REDDEDİLİR. `_canonical_source_series()`
    tanımadığı bir ad için `ValueError` fırlatır -- bu, `fit()`'in
    whitelist semantiğine (`apply()`'ın TCGA-özel blacklist'inden
    FARKLI) uydurulmak için burada YAKALANIP `ComBatFitLeakageError`'a
    çevrilir: whitelist mantığında "tanınmıyorsa izin verme" zaten
    doğru davranıştır, sessiz kabul YOK.
    """

    import neuroHarmonize as nh

    _validate_feature_matrix(feature_matrix)
    _validate_covariates(feature_matrix, covariates)

    try:
        canonical_site_series = _canonical_source_series(
            covariates[REQUIRED_COVARIATE_COLUMN]
        )
    except ValueError as exc:
        raise ComBatFitLeakageError(
            "fit_combat_harmonization() SADECE "
            f"{sorted(COMBAT_FIT_ALLOWED_SOURCES)} kaynaklarını görebilir "
            "(whitelist). covariates['SITE'] içinde `SOURCE_ALIASES`'in "
            "tanımadığı bir kaynak adı bulundu -- whitelist mantığı "
            "gereği tanınmayan her şey reddedilir (sessiz kabul yasak). "
            f"Orijinal hata: {exc}"
        ) from exc

    sources_present = set(canonical_site_series.unique())
    forbidden = sources_present - COMBAT_FIT_ALLOWED_SOURCES
    if forbidden:
        raise ComBatFitLeakageError(
            "fit_combat_harmonization() SADECE "
            f"{sorted(COMBAT_FIT_ALLOWED_SOURCES)} kaynaklarını görebilir. "
            "Yasak/beklenmeyen kaynak(lar) covariates['SITE']'da "
            f"(kanonikleştirilmiş) bulundu: {sorted(forbidden)}. TCGA "
            "(veya başka bir kaynak) fit()'e ASLA girmemeli -- CLAUDE.md "
            "kritik mimari kural."
        )
    # 2026-08-12 FIX-5, bulgu #1 (Codex): `reference_batch` parametresi
    # (varsayılan "UPenn", zaten kanonik) önceden kanonikleştirilmeden
    # `sources_present` (kanonik) ile karşılaştırılıyordu -- biri açıkça
    # `reference_batch="UPenn-GBM"` diye çağırırsa `sources_present`
    # içindeki kanonik "UPenn" ile asla eşleşmez, ValueError yanlış-pozitif
    # fırlatırdı. Ayrıca kanonikleştirilmemiş ham değer aşağıda
    # `nh.harmonizationLearn(ref_batch=...)`'e geçirilirse, `covars_for_fit
    # ['SITE']` (kanonik) içindeki hiçbir satırla eşleşmeyen bir ref_batch
    # ile fit edilmiş olurdu (referans-batch identity kısayolu sessizce
    # bozulurdu). Şimdi reference_batch de aynı `canonical_source()` ile
    # normalize edilip hem guard'da hem nh.harmonizationLearn() çağrısında
    # kullanılıyor.
    try:
        canonical_reference_batch = canonical_source(reference_batch)
    except ValueError as exc:
        raise ValueError(
            f"reference_batch={reference_batch!r} `canonical_source()` "
            f"tarafından tanınmıyor. Orijinal hata: {exc}"
        ) from exc

    if canonical_reference_batch not in sources_present:
        raise ValueError(
            f"reference_batch={reference_batch!r} (kanonik: "
            f"{canonical_reference_batch!r}) covariates['SITE'] içinde "
            f"(kanonikleştirilmiş) bulunamadı: {sorted(sources_present)}"
        )

    covars_for_fit = covariates.copy()
    covars_for_fit[REQUIRED_COVARIATE_COLUMN] = canonical_site_series.astype(str)

    data = feature_matrix.to_numpy(dtype=np.float64)
    model, bayes_data = nh.harmonizationLearn(
        data, covars_for_fit, ref_batch=canonical_reference_batch, seed=seed
    )

    parameter_rows = _model_to_combat_parameter_rows(
        model=model,
        feature_names=list(feature_matrix.columns),
        library_tag=library_tag or f"neuroHarmonize=={_neuroharmonize_version()}",
    )

    harmonized_in_sample = pd.DataFrame(
        bayes_data, index=feature_matrix.index, columns=feature_matrix.columns
    )
    return model, harmonized_in_sample, parameter_rows


def _model_to_combat_parameter_rows(
    model: dict[str, Any], feature_names: list[str], library_tag: str
) -> list[dict[str, Any]]:
    """neuroHarmonize model sözlüğünü `combat_parameters` satırlarına çevir.

    Not: `model['stand_mean']`'in her kolonu birbirinin AYNISI (kovaryat
    etkisinden bağımsız, saf batch-öncesi ortalama) -- bu ``B_hat[0, :]``
    ile birebir eşittir (deneyle doğrulandı) ve kanonik "grand_mean"
    değeri budur. `var_pooled` tüm batch'lerde ortak (havuzlanmış
    varyans), her batch satırında aynı değer tekrar edilir -- şema bunu
    batch-bazlı sakladığı için (`combat_parameters.pooled_variance`),
    burada bilinçli bir tekrar var, hata değil.
    """

    site_labels = list(model["SITE_labels"])
    gamma = np.asarray(model["gamma_star"], dtype=np.float64)
    delta = np.asarray(model["delta_star"], dtype=np.float64)
    grand_mean = np.asarray(model["B_hat"], dtype=np.float64)[0, :]
    pooled_variance = np.asarray(model["var_pooled"], dtype=np.float64).reshape(-1)
    reference_batch = model["ref_batch"]
    fit_date = date.today().isoformat()

    expected_shape = (len(site_labels), len(feature_names))
    if gamma.shape != expected_shape or delta.shape != expected_shape:
        raise RuntimeError(
            "neuroHarmonize model boyutları beklenenden farklı "
            f"(gamma={gamma.shape}, beklenen={expected_shape}) -- "
            "şema eşlemesi bozulmuş olabilir, INSERT'e devam etme."
        )
    if grand_mean.shape[0] != len(feature_names) or pooled_variance.shape[0] != len(
        feature_names
    ):
        raise RuntimeError(
            "grand_mean/pooled_variance uzunluğu feature_names ile uyuşmuyor."
        )

    design_matrix_ref = json.dumps(
        {
            "covariates": list(model["Covariates"]),
            "reference_batch": reference_batch,
            "site_labels": site_labels,
            "n_sample": int(model["info_dict"]["n_sample"]),
            "sample_per_batch": {
                site_labels[i]: int(n)
                for i, n in enumerate(model["info_dict"]["sample_per_batch"])
            },
            "eb": bool(model.get("eb", True)),
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    rows: list[dict[str, Any]] = []
    for batch_idx, batch_label in enumerate(site_labels):
        for feature_idx, feature_name in enumerate(feature_names):
            rows.append(
                {
                    "source_batch": batch_label,
                    "feature_name": feature_name,
                    "gamma": float(gamma[batch_idx, feature_idx]),
                    "delta": float(delta[batch_idx, feature_idx]),
                    "grand_mean": float(grand_mean[feature_idx]),
                    "pooled_variance": float(pooled_variance[feature_idx]),
                    "design_matrix_ref": design_matrix_ref,
                    "fit_date": fit_date,
                    "reference_batch": batch_label == reference_batch,
                    "harmonization_library": library_tag,
                }
            )
    return rows


def apply_combat_harmonization(
    feature_matrix: pd.DataFrame,
    covariates: pd.DataFrame,
    model: dict[str, Any],
    *,
    true_dataset_source: pd.Series,
) -> pd.DataFrame:
    """Fitted ComBat modelini `harmonizationApply()` ile out-of-sample uygula.

    MİMARİ KURAL (2026-08-09 revizyonu, bkz.
    decisions/2026-08-09-combat-fit-iki-blokaj-region-ve-pyradiomics-parametre.md;
    2026-08-18 UCSF kuralı, CLAUDE.md):
    **TCGA ve UCSF, ComBat kapsamının TAMAMEN DIŞINDADIR.** Bu fonksiyon
    onlar için ASLA çağrılmamalı -- yalnızca N4 + kaynak-bazlı Z-score
    (görüntü seviyesi) harmonizasyonu alır
    (`apply_zscore_normalization`), feature-level ComBat düzeltmesi
    ALMAZ. Gerekçe: (a) TCGA'nın WT/TC (kompozit) maskeleri UPenn/
    LUMIERE'in NC/ED/ET alt-bölgeleriyle yapısal olarak uyuşmuyor
    (TCGA'da ET hiçbir zaman üretilmeyecek, kilitli karar), (b)
    referans-batch'e out-of-sample "apply" zaten cebirsel olarak
    özdeşlik dönüşümüydü (aşağıdaki matematiksel not), pratik faydası
    yoktu.

    Bu kuralı zorlamak için çağıran taraf **zorunlu** ``true_dataset_source``
    parametresini vermelidir: `feature_matrix` ile aynı index'e sahip,
    her satırın GERÇEK veri kaynağını taşıyan bir `pandas.Series`.
    Bu değer ``covariates['SITE']``'DAN BAĞIMSIZDIR ve ona GÜVENİLMEZ --
    ``SITE`` spoof edilebilir (örn. TCGA satırları
    ``SITE="UPenn"`` etiketiyle çağrılabilir), bu yüzden TCGA
    reddi ayrı, spoof edilemez bir kanaldan yapılır. Her satır
    ``canonical_source()``/``SOURCE_ALIASES`` ile normalize edilir
    (2026-08-12 CRITICAL fix -- önceki ad-hoc ``.strip().upper()``
    karşılaştırması yalnız düz ``"TCGA"``'yı yakalıyordu, ``"TCGA-GBM"``
    gibi alias'ları KAÇIRIYORDU; `dataset_sources.source_name` DB
    kolonunun gerçek değeri literal ``"TCGA-GBM"``). Normalize edilmiş
    değer ``"TCGA"`` ise fonksiyon `ComBatApplyLeakageError` fırlatır --
    SITE değeri ne olursa olsun. Tanınmayan bir kaynak adı (ne TCGA ne
    UPenn ne LUMIERE alias'ı) SESSİZCE geçirilmez, `ValueError` fırlatılır.

    Meşru kullanım şekli (yalnız bu):
    UPenn/LUMIERE'den GERÇEKTEN yeni bir hasta/satır (fit'te görülmedi
    ama kaynağı fit'te vardı) -- ``covariates['SITE']`` o hastanın
    gerçek kaynağı (``"UPenn"`` veya ``"LUMIERE"``) ve
    ``true_dataset_source`` de aynı gerçek kaynağı taşımalı, modelin
    o batch için fit ettiği gamma/delta kullanılır.

    KRİTİK İSTATİSTİKSEL NOT -- referans-batch projeksiyonu özdeşlik
    dönüşümüdür (cebirsel + deneysel doğrulandı, bkz. modeling-agent
    2026-08-09 oturum notları, YUKARIDAKİ TCGA-dışlama kararının
    gerekçelerinden biri): neuroHarmonize'ın referans-batch
    tasarımında, referans batch için ``gamma_star=0`` ve
    ``delta_star=1`` (ComBat'ın referans-batch tanımının doğrudan
    sonucu). Standardize/de-standardize adımları (``stand_mean``,
    ``mod_mean``, ``var_pooled``) referans batch için matematiksel
    olarak TAM olarak birbirini iptal eder: ``bayes_data == X``
    (girdiyle BİREBİR AYNI, ondalık hassasiyete kadar). Yani bir
    satırı referans-batch'e projekte etmek o satırın ham özelliklerini
    FİİLEN DEĞİŞTİRMEZ -- pratikte "ComBat hiç çalıştırılmamış" ile
    aynı sonucu verir. Bu bir kütüphane hatası DEĞİL, ComBat'ın
    referans-batch semantiğinin matematiksel zorunlu sonucu (referans
    batch = hedef uzay, kendi kendine düzeltme sıfır).

    Kütüphane sınırlaması (deneyle doğrulandı, bkz. yukarıdaki not):
    neuroHarmonize 2.5.1'in ``harmonizationApply()``'ı, apply-veri
    kümesindeki ``covariates['SITE']`` TEK bir batch etiketi
    içerdiğinde (üretimdeki her gerçek senaryo budur -- ya tüm-yeni-
    kohort tek seferde ya da tek-yeni-hasta) iç tasarım matrisi boyut
    uyuşmazlığı hatası veriyor (`ValueError: shapes ... not aligned`).
    Bu yüzden bu fonksiyon `harmonizationApply()` yerine satır satır
    çalışır -- daha yavaş ama doğru.

    KÜTÜPHANE HATASI -- ref_level indeks uyuşmazlığı (2026-08-12'de
    bulundu ve DÜZELTİLDİ, bkz. decisions/2026-08-11-apply-combat-
    harmonization-true-source-guard.md "[2026-08-12] FIX" bölümü):
    neuroHarmonize'ın kendi ``applyModelOne()``'ı, satır-bazlı tasarım
    matrisini kurarken `make_design_matrix(..., ref_level)`'i
    fit-zamanındaki GLOBAL `ref_level` indeksiyle (SITE_labels'ın
    ALFABETİK sırasındaki referans-batch index'i) çağırıyor, ama
    tek-satırlık `covars`'ın YEREL one-hot matrisi HER ZAMAN 1 sütun
    (tek örnek = tek kategori). `ref_level>0` olduğunda (üretim
    konfigürasyonunda öyle: `reference_batch="UPenn"`,
    `SITE_labels=["LUMIERE","UPenn"]` çünkü 'LUMIERE'<'UPenn'
    alfabetik, `ref_level=1`), `batch_onehot[:, 1] = ...` 1 sütunluk
    matriste `IndexError` fırlatıyor -- hem referans hem
    referans-olmayan satırlar aynı şekilde çöküyor. Bu yerel one-hot
    sütunu zaten hemen ardından `design_i_batch` (fit-zamanındaki TAM
    SITE_labels boyutunda, doğru kurulan) ile DEĞİŞTİRİLİYOR -- yani
    orijinal koddaki bu ilk çağrıya `ref_level` geçmenin nihai sonuca
    hiçbir katkısı yok, sadece çöküyor. Bu fonksiyon artık
    `applyModelOne()`'ın yerel (bu modüldeki `_apply_model_one_fixed`)
    bir kopyasını kullanır: TEK FARK, o ilk `make_design_matrix`
    çağrısına `ref_level` yerine `None` geçirilir (çökmeyi önler,
    nihai `design_i`'yi hiç ETKİLEMEZ -- design_i_batch zaten üzerine
    yazıyor); geri kalan matematik (referans-batch identity kısayolu
    dahil, `ref_level`'in GERÇEK değeriyle) birebir aynıdır. Ampirik
    doğrulama: `tests/test_harmonization.py::
    test_apply_combat_harmonization_reference_batch_row_is_identity`
    ve `::test_apply_combat_harmonization_matches_in_sample_fit_for_non_reference_batch`.

    Hata
    ----
    ComBatApplyLeakageError
        ``true_dataset_source`` içinde herhangi bir satır (`canonical_source()`
        normalizasyonundan sonra) ``COMBAT_FIT_ALLOWED_SOURCES``
        (= ``{"UPenn","LUMIERE"}``) DIŞINDA ise -- SITE etiketinden
        bağımsız, mimari sızıntı önleme kontrolü. 2026-09-13 (karar 29)
        öncesinde bu guard yalnız ``"TCGA"``'ya bakıyordu ve
        2026-08-18'de `SOURCE_ALIASES`'e eklenen ``"UCSF"`` sessizce
        geçiyordu; artık whitelist tümleyeni kullanılır. ``"TCGA"``,
        ``"tcga-gbm"``, ``" TCGA-GBM "``, ``"UCSF"``, ``"ucsf-pdgm"``
        gibi tüm bilinen alias varyantlarını yakalar.
    ValueError
        ``true_dataset_source`` içinde `canonical_source()`'ın
        tanımadığı bir kaynak adı varsa (sessiz fallback yasak).
    """

    _validate_feature_matrix(feature_matrix)
    _validate_covariates(feature_matrix, covariates)

    if not isinstance(true_dataset_source, pd.Series):
        raise TypeError(
            "true_dataset_source pandas.Series olmalı -- her satırın "
            "GERÇEK veri kaynağını taşımalı, covariates['SITE']'dan "
            "BAĞIMSIZ (SITE spoof edilebilir, bu parametre edilemez)."
        )
    if not feature_matrix.index.equals(true_dataset_source.index):
        raise ValueError(
            "true_dataset_source, feature_matrix ile aynı index'te "
            "(aynı hasta/scan sırasında) olmalı -- hizalama garantisi "
            "yok, sessizce hizalanmaz."
        )

    canonical_true_source = _canonical_source_series(true_dataset_source)
    # 2026-09-13 (karar 29): guard önceden SADECE `== "TCGA"` bakıyordu --
    # BLACKLIST. CLAUDE.md'nin 2026-08-18 kuralı ("UCSF ... feature-level
    # ComBat düzeltmesi ALMAZ, ComBat fit'ine GİRMEZ") UCSF'e TCGA ile
    # AYNI muameleyi şart koşuyor, ama `SOURCE_ALIASES`'e 2026-08-18'de
    # eklenen "UCSF" bu blacklist'ten SESSİZCE geçiyordu.
    #
    # Sabit `{"TCGA", "UCSF"}` listesi YERİNE `COMBAT_FIT_ALLOWED_SOURCES`
    # (= {"UPenn", "LUMIERE"}) WHITELIST'inin TÜMLEYENİ seçildi. Gerekçe:
    # (a) `fit_combat_harmonization()` zaten AYNI whitelist'i kullanıyor
    #     -- fit/apply artık simetrik: bir kaynak fit'e giremiyorsa
    #     apply'a da giremez. İki guard'ın ayrı listelerle sürüklenmesi
    #     (drift) riski ORTADAN KALKAR.
    # (b) Matematiksel olarak doğru olan da bu: apply, batch'e özgü
    #     gamma_star/delta_star gerektirir ve bunlar YALNIZ fit'te
    #     görülmüş batch'ler için vardır.
    # (c) İleride `SOURCE_ALIASES`'e yeni bir kaynak eklenirse (UCSF'te
    #     tam olarak bu oldu) guard KENDİLİĞİNDEN korur; blacklist ise
    #     her yeni kaynakta elle güncellenmeyi gerektirir ve
    #     unutulduğunda SESSİZCE açık kalır -- fail-open.
    forbidden_mask = ~canonical_true_source.isin(COMBAT_FIT_ALLOWED_SOURCES)
    if bool(forbidden_mask.any()):
        offending_index = list(true_dataset_source.index[forbidden_mask][:10])
        offending_sources = sorted(set(canonical_true_source[forbidden_mask]))
        raise ComBatApplyLeakageError(
            "apply_combat_harmonization() YALNIZ ComBat fit'inde GÖRÜLMÜŞ "
            f"kaynaklarla ({sorted(COMBAT_FIT_ALLOWED_SOURCES)}) "
            "çağrılabilir. true_dataset_source içinde fit'te görülmemiş "
            f"kaynak(lar) bulundu: {offending_sources} (örn. index: "
            f"{offending_index}).\n"
            "- TCGA: 2026-08-09 kararı, ComBat kapsamının tamamen "
            "dışında (bkz. decisions/2026-08-09-combat-fit-iki-blokaj-"
            "region-ve-pyradiomics-parametre.md).\n"
            "- UCSF: 2026-08-18 kararı (CLAUDE.md), harici test seti "
            "olarak TCGA ile AYNI muamele -- görüntü-seviyesi N4+T1ce "
            "Z-score ALIR, feature-level ComBat düzeltmesi ALMAZ, ComBat "
            "fit'ine GİRMEZ. Harici test setine eğitim-türevi bir "
            "dönüşüm uygulamak bağımsızlığını zedeler.\n"
            "Bu kontrol covariates['SITE'] değerine GÜVENMEZ -- satırlar "
            "'UPenn' gibi başka bir kaynak etiketiyle spoof edilmiş olsa "
            "bile reddedilir. Normalizasyon `canonical_source()`/"
            "`SOURCE_ALIASES` üzerinden yapılır: 'TCGA', 'tcga-gbm', "
            "' TCGA-GBM ', 'UCSF', 'ucsf-pdgm' gibi tüm bilinen "
            "varyantlar yakalanır (2026-08-12 CRITICAL fix + 2026-09-13 "
            "whitelist genişletmesi)."
        )

    # 2026-08-12 FIX-4 (bkz. decisions/2026-08-11-apply-combat-harmonization-
    # true-source-guard.md "[2026-08-12] FIX-4"): FIX-3, fit_combat_
    # harmonization()'ı artık covars_for_fit['SITE']'ı kanonik değerlerle
    # ("UPenn"/"LUMIERE") doldurup fit ediyor -- yani `model["SITE_labels"]`
    # de artık KANONİK. Ama bu blok önceden `covariates['SITE']`'ı SADECE
    # `.astype(str)` yapıp ham haliyle model["SITE_labels"]'e karşı
    # karşılaştırıyordu (canonical_source()'tan HİÇ geçirmiyordu) -- bu
    # yüzden gerçek DB source_name değerleriyle ("UPenn-GBM") fit()
    # başarılı oluyor ama HEMEN ARDINDAN apply() aynı değerlerle
    # ValueError ile çöküyordu (reviewer canlı reprodüksiyon etti).
    #
    # Düzeltme: HEM `model["SITE_labels"]` HEM `covariates['SITE']`
    # `canonical_source()`/`_canonical_source_series()` üzerinden
    # kanonikleştirilip öyle karşılaştırılıyor -- karşılaştırma artık
    # kanonik-kanonik.
    #
    # GERİYE DÖNÜK UYUMLULUK VARSAYIMI (açıkça belirtiliyor, sessizce
    # yapılmıyor): Bu proje şu an `fit_combat_harmonization()`'ın
    # döndürdüğü `model` sözlüğünü hiçbir yerde disk/DB'ye serialize
    # etmiyor (grep ile doğrulandı: repo genelinde bu modül dışında
    # `model["SITE_labels"]`/pickle/joblib kullanan bir kalıcılaştırma
    # noktası yok) -- yani PRATİKTE her `apply_combat_harmonization()`
    # çağrısı aynı süreç içinde az önce fit edilmiş (dolayısıyla her
    # zaman kanonik SITE_labels taşıyan) bir modelle çalışıyor. Yine de
    # bu varsayıma KÖRÜ KÖRÜNE güvenmiyoruz: `canonical_source()` HEM
    # eski-tip ham etiketler (örn. "UPenn-GBM") HEM zaten-kanonik
    # etiketler (örn. "UPenn") için İDEMPOTENTTİR (`canonical_source(
    # "UPenn") == "UPenn"`, deneyle doğrulandı -- `SOURCE_ALIASES`
    # lower-case anahtarları kanonik değerlerin lower-case haline de
    # eşleniyor). Bu yüzden aşağıdaki kanonikleştirme, model ister
    # FIX-3-öncesi ham etiketlerle ister FIX-3-sonrası kanonik
    # etiketlerle fit edilmiş olsun DOĞRU GEÇER (`unresolved` boş
    # kalır) -- "model her zaman taze fit edilir" varsayımına muhtaç
    # DEĞİLDİR. Tek istisna: model SITE_labels'ı `SOURCE_ALIASES`'in
    # hiç tanımadığı bir değer taşıyorsa (örn. bambaşka bir eski
    # format), bu artık sessizce `str()`'a düşmez, `canonical_source()`
    # `ValueError` fırlatır -- bu BİLİNÇLİ bir davranış (sessiz fallback
    # yasak), böyle bir model zaten güvenilir biçimde yorumlanamaz.
    #
    # DÜZELTME (2026-08-12 FIX-4 SONRASI, FIX-5'te düzeltildi -- bkz.
    # aşağıdaki blok): YUKARIDAKİ "DOĞRU ÇALIŞIR" iddiası eksikti. Bu
    # `known_site_labels` kontrolü SADECE bu guard'ın kendisinin GEÇMESİNİ
    # sağlıyordu -- kanonikleştirilmiş etiketler `model` dict'ine geri
    # YAZILMIYORDU, bu yüzden aşağıdaki `_apply_model_one_fixed()` çağrısı
    # hâlâ ORİJİNAL (potansiyel olarak ham) `model["SITE_labels"]`'ı
    # kullanıyor, ham-etiketli bir "eski model" senaryosunda guard'dan
    # geçmesine rağmen içeride SESSİZCE NaN üretiyordu (reviewer canlı
    # reprodüksiyon etti). Bu artık FIX-5'te `canonical_model` kopyasıyla
    # düzeltildi.
    try:
        known_site_labels = {
            canonical_source(str(label)) for label in model["SITE_labels"]
        }
    except ValueError as exc:
        raise ValueError(
            "model['SITE_labels'] içinde `canonical_source()`'ın "
            "tanımadığı bir etiket var -- bu modelin kaynağı/formatı "
            "güvenilir biçimde yorumlanamıyor, sessizce str()'a "
            f"düşülmüyor. Orijinal hata: {exc}"
        ) from exc

    covars = covariates.copy()
    canonical_covar_site = _canonical_source_series(
        covars[REQUIRED_COVARIATE_COLUMN]
    )
    covars[REQUIRED_COVARIATE_COLUMN] = canonical_covar_site.astype(str)
    unresolved = set(canonical_covar_site.unique()) - known_site_labels
    if unresolved:
        raise ValueError(
            f"covariates['SITE'] içinde modelin tanımadığı etiket(ler) var "
            f"(kanonikleştirilmiş): {sorted(unresolved)}. Modelin bildiği "
            f"kaynaklar (kanonikleştirilmiş): {sorted(known_site_labels)}. "
            "Fit'te hiç görülmemiş bir kaynağı (örn. TCGA) out-of-sample "
            f"uygularken SITE değerini AÇIKÇA model['ref_batch']="
            f"{model['ref_batch']!r} olarak ayarlayın -- bu fonksiyon "
            "örtük bir varsayılan uygulamaz."
        )

    # 2026-08-12 FIX-5, bulgu #2+#3 (Codex + reviewer, reviewer CANLI
    # REPRODÜKSİYON ETTİ -- bkz. decisions/2026-08-11-apply-combat-
    # harmonization-true-source-guard.md "[2026-08-12] REVIEW-GATE FIX-4
    # SONUCU"): yukarıdaki `known_site_labels` kontrolü doğrulamayı
    # KANONİK yapıyor ve doğru geçiyor, ama bu kanonikleştirilmiş
    # etiketler önceden `model` dict'ine geri YAZILMIYORDU --
    # `_apply_model_one_fixed()` hâlâ ORİJİNAL (potansiyel olarak ham,
    # örn. FIX-3-öncesi "UPenn-GBM" formatında) `model["SITE_labels"]`'ı
    # alıyordu. İçeride `covars["SITE"].isin(model["SITE_labels"])` ham
    # liste ile kanonik covars'ı karşılaştırdığı için (kanonik covars'ta
    # "UPenn-GBM" hiç yok, sadece "UPenn" var) `is_train_site` HER ZAMAN
    # False çıkıyor ve kod `if int(is_train_site.sum()) == 0: bayes_data
    # = np.full(..., np.nan)` yoluna SESSİZCE düşüyordu -- guard'ın
    # kendisi geçtiği için hiçbir hata fırlamıyor, sonuç NaN oluyordu.
    # Bu, "sessiz fallback yasak" kuralına doğrudan aykırıydı.
    #
    # Düzeltme: `model`'in SIĞ bir kopyasını al, SADECE `SITE_labels`'ı
    # üzerine kanonik değerlerle yaz, `_apply_model_one_fixed()`'e bu
    # kopyayı geçir. `model["ref_batch"]`, `model["gamma_star"]`,
    # `model["delta_star"]`, `model["B_hat"]`, `model["var_pooled"]`,
    # `model["info_dict"]` gibi diğer alanlar `dict(model)` sığ kopyada
    # AYNI referansı paylaşır (mutasyona uğramaz, sadece okunur) -- bu
    # doğru ve güvenlidir, çünkü `_apply_model_one_fixed()` bu alanları
    # HİÇBİRİNİ üzerine yazmaz (yalnız okur). Orijinal `model` dict'i
    # (çağıranın elindeki) HİÇ mutasyona uğratılmaz.
    canonical_model = dict(model)
    canonical_model["SITE_labels"] = np.array(
        [canonical_source(str(label)) for label in model["SITE_labels"]],
        dtype=object,
    )

    data = feature_matrix.to_numpy(dtype=np.float64)
    n_samples, n_features = data.shape
    output = np.empty((n_samples, n_features), dtype=np.float64)
    for row_index in range(n_samples):
        row_data = data[row_index : row_index + 1, :]
        row_covars = covars.iloc[row_index : row_index + 1]
        harmonized_row = _apply_model_one_fixed(row_data, row_covars, canonical_model)
        output[row_index, :] = harmonized_row[0, :]

    return pd.DataFrame(
        output, index=feature_matrix.index, columns=feature_matrix.columns
    )


def _apply_model_one_fixed(
    data: np.ndarray, covars: pd.DataFrame, model: dict[str, Any]
) -> np.ndarray:
    """neuroHarmonize 2.5.1'in ``applyModelOne()``'ının ref_level-güvenli kopyası.

    Bkz. `apply_combat_harmonization()` docstring'indeki "KÜTÜPHANE HATASI"
    notu -- burada TEK değişiklik, ilk `make_design_matrix()` çağrısına
    `ref_level` yerine `None` geçirilmesi (o çağrının ürettiği batch
    one-hot sütunu zaten hemen ardından `design_i_batch` ile
    değiştiriliyor, `ref_level`'in oraya geçirilmesinin nihai sonuca
    hiçbir katkısı yok, sadece tek-satır/tek-kategori boyut
    uyuşmazlığıyla çöküyor). Kalan mantık (gamma_star/delta_star
    uygulaması, referans-batch identity kısayolu) neuroHarmonize'ın
    orijinal `applyModelOne()`'ıyla birebir aynıdır.
    """

    from neuroCombat.neuroCombat import make_design_matrix

    if data.shape[0] > 1:
        raise ValueError("Argument `data` contains more than one sample!")
    if covars.shape[0] > 1:
        raise ValueError("Argument `covars` contains more than one sample!")
    if model["smooth_model"]["perform_smoothing"]:
        raise NotImplementedError(
            "[neuroHarmonize] applyModelOne does not support models trained "
            "with smooth_terms. Use flattenNIFTIs + harmonizationApply "
            "instead."
        )

    X = data.T
    batch_labels = np.asarray(model["SITE_labels"])
    batch_i = covars["SITE"].to_numpy()[0]
    is_train_site = covars["SITE"].isin(model["SITE_labels"]).to_numpy()

    if batch_i not in batch_labels:
        batch_level_i = np.array([0])
    else:
        batch_level_i = np.argwhere(batch_i == batch_labels)[0]

    ref_level = model["info_dict"].get("ref_level")

    batch_col = covars.columns.get_loc("SITE")
    cat_cols: list[int] = []
    num_cols = [covars.columns.get_loc(c) for c in covars.columns if c != "SITE"]
    covars_array = np.array(covars, dtype="object")

    covars_array[:, batch_col] = np.unique(
        covars_array[:, batch_col], return_inverse=True
    )[-1]

    # DÜZELTME: bu ilk (yerel, her zaman 1 satır -> 1 kategori) tasarım
    # matrisi çağrısına orijinal koddaki gibi global `ref_level` yerine
    # `None` geçiriyoruz -- ürettiği batch one-hot sütunu aşağıda zaten
    # `design_i_batch` ile değiştiriliyor, bu yüzden `ref_level`'in bu
    # çağrıya katkısı yok, sadece 1 sütunluk matriste geçersiz indeks
    # (IndexError) üretiyordu.
    design_i = make_design_matrix(covars_array, batch_col, cat_cols, num_cols, None)

    # batch'leri fit-zamanındaki TAM SITE_labels boyutuna göre kodla --
    # bu satır her zaman geçerlidir, `ref_level` global indeksinin
    # sınırları içindedir.
    design_i_batch = np.zeros((1, len(batch_labels)))
    design_i_batch[:, batch_level_i] = 1
    design_i = np.concatenate((design_i_batch, design_i[:, 1:]), axis=1)

    design_i[~is_train_site, 0 : len(model["SITE_labels"])] = np.nan

    n_sample = 1
    sample_per_batch = 1
    design_matrix = design_i
    n_batch = len(batch_labels)
    batch_index = batch_level_i[0]

    b_hat = model["B_hat"]
    stand_mean = model["stand_mean"][:, [0]]
    var_pooled = model["var_pooled"]

    mod_mean = 0
    if design_i is not None:
        tmp = design_i.copy()
        tmp[:, range(0, n_batch)] = 0
        mod_mean = np.transpose(np.dot(tmp, b_hat))

    s_data = (X - stand_mean - mod_mean) / np.dot(
        np.sqrt(var_pooled), np.ones((1, n_sample))
    )

    if int(is_train_site.sum()) == 0:
        bayes_data = np.full(s_data.shape, np.nan)
    else:
        batch_design = design_matrix[:, :n_batch]

        bayes_data = s_data
        gamma_star = np.array(model["gamma_star"])
        delta_star = np.array(model["delta_star"])

        dsq = np.sqrt(delta_star[batch_index, :])
        dsq = dsq.reshape((len(dsq), 1))
        denom = np.dot(dsq, np.ones((1, sample_per_batch)))
        numer = np.array(bayes_data - np.dot(batch_design, gamma_star).T)

        bayes_data = numer / denom

        vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
        bayes_data = (
            bayes_data * np.dot(vpsq, np.ones((1, n_sample))) + stand_mean + mod_mean
        )

        # referans batch için orijinal veriyi koru (adjust_data_final ile
        # aynı davranış) -- ref_level artık her zaman geçerli bir global
        # indeks, çünkü design_i_batch fit-zamanındaki TAM SITE_labels
        # boyutuna göre kuruldu.
        if ref_level is not None and batch_index == ref_level:
            bayes_data = X

    return bayes_data.T


def get_segmentation_mask(patient_id: str, scan_id: str) -> dict:
    """DB kaydından kohorta ait hazır maskeyi çözümle.

    Bu fonksiyon TCGA/UPenn/LUMIERE için nnU-Net fallback uygulamaz. Geometri
    eşleşmiyorsa PyRadiomics'e geçmek yerine açık hata verir.
    """

    from psycopg2.extras import RealDictCursor

    from db_connection import get_connection
    from pipeline.segmentation import resolve_nas_path, resolve_ready_mask

    connection = get_connection(readonly=True)
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(
                "SELECT ms.scan_id, ms.patient_id, ms.modality, ms.file_path, "
                "ds.source_name AS source "
                "FROM mr_scans AS ms "
                "JOIN patients AS p ON p.patient_id = ms.patient_id "
                "JOIN dataset_sources AS ds ON ds.source_id = p.source_id "
                "WHERE ms.scan_id = %s AND ms.patient_id = %s",
                (scan_id, patient_id),
            )
            scan = cursor.fetchone()
        finally:
            cursor.close()
    finally:
        connection.close()

    if scan is None:
        raise LookupError(
            f"Tarama bulunamadı veya hastayla eşleşmiyor: "
            f"patient_id={patient_id}, scan_id={scan_id}"
        )

    image_path = resolve_nas_path(scan["file_path"])
    return resolve_ready_mask(
        source=scan["source"],
        image_path=image_path,
        patient_id=scan["patient_id"],
        scan_id=scan["scan_id"],
    )
