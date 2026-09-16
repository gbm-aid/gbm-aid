"""Harmonizasyon öncesi/sonrası dağılım karşılaştırma DOĞRULAMA scripti.

Görev kaynağı: `raw/plan/plan.txt` satır 386-387 (Ege / Test & Destek, Hafta 2):
"Harmonizasyon öncesi/sonrası dağılım karşılaştırmasını (μ=0, σ=1 kontrolü)
doğrulama scripti yaz."

Bu script BAĞIMSIZ bir doğrulamadır -- `pipeline/harmonization.py`'nin
`fit_source_zscore_statistics()`/`apply_zscore_normalization()`
fonksiyonlarını ÇAĞIRMAZ (bu fonksiyonları çağırmak, üretim
`artifacts/week2/zscore/source_stats.json` dosyasını YENİDEN YAZAR --
mimari kural: bu script DB'ye/dosyaya hiçbir YAZMA yapmaz). Onun yerine:

  1. DB'den (salt-okunur) her kaynaktan (TCGA-GBM/UPenn-GBM/LUMIERE)
     `harmonization_status='zscore'` olan gerçek `scan_id`'lerden küçük
     bir örneklem seçer (varsayılan 6/kaynak, `--n-per-source` ile
     ayarlanabilir; `--all` ile TÜM kohort da taranabilir).
  2. Her örneklem taraması için GERÇEK NAS dosyasını okur (ham/raw) ve
     `run_harmonization_cohort.py`'nin ürettiği, diskte ZATEN VAR OLAN
     N4 ve Z-score ARA ÇIKTILARINI (aynı deterministik hash-yol şemasıyla,
     bkz. `pipeline/harmonization.py::_derived_output_path`) okur --
     hiçbir görüntü YENİDEN İŞLENMEZ/YAZILMAZ (varsayılan davranış;
     `--allow-recompute` açıkça verilirse eksik ara çıktı üretim
     fonksiyonlarıyla üretilip diske yazılabilir, bu KAPALI varsayılan).
  3. Her aşamada (raw / n4 / zscore) foreground voksellerinin (kural:
     sonlu VE sıfır-dışı -- `source_stats.json`'daki
     `foreground_rule: finite_nonzero` ile birebir aynı) μ/σ'sini kendi
     numpy koduyla, harmonization.py'nin fonksiyonlarını çağırmadan
     bağımsızca hesaplar.
  4. İKİ AYRI kontrol raporlar:
     a) SIKI/cebirsel kontrol (asıl doğrulama): her taramanın kendi N4-
        foreground μ/σ'sinden + `source_stats.json`'daki fit edilmiş
        (kaynak-bazlı) μ/σ'den Z-score formülüyle BEKLENEN μ/σ'yi
        hesaplar, diskteki GERÇEK Z-score çıktısında GÖZLENEN μ/σ ile
        karşılaştırır. Fark ~0 olmalı (float32 hassasiyeti içinde) --
        değilse pipeline'da gerçek bir hata var demektir (yanlış
        kaynağın istatistiği uygulanmış, foreground kuralı tutmamış vb.)
     b) GEVŞEK/örneklem-havuzlanmış kontrol: örneklemdeki taramaların
        TÜM foreground vokselleri havuzlanıp kaynak-bazlı tek bir μ/σ
        hesaplanır, μ≈0/σ≈1 hedefiyle karşılaştırılır. KAYNAK-BAZINDA
        değerlendirilir (TÜM kohort birlikte DEĞİL -- her kaynak kendi
        μ=0/σ=1'ine ayrı ayrı normalize ediliyor, `harmonization.py`
        kaynak-bazlı Z-score mantığıyla tutarlı). Küçük örneklem
        nedeniyle tam 0/1 BEKLENMEZ (sapma toleransı gevşek), bu kontrol
        bilgilendiricidir -- asıl doğrulama (a)'dır.

Çıktı: konsola okunabilir rapor + (`--csv` verilirse) tarama-bazlı CSV.
DB'ye hiçbir INSERT/UPDATE YAPILMAZ (salt-okunur bağlantı). Üretim
dosyalarına (artifacts/week2/n4, /zscore, /zscore/source_stats.json)
varsayılan modda hiçbir YAZMA yapılmaz.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


# NOT: `.resolve()` KULLANMA -- bu makinede `subst X:` ile ASCII sürücü
# haritalandı (kullanıcı adı Türkçe karakter içeriyor). `.absolute()`
# sürücü harfini korur, `run_harmonization_cohort.py` ile aynı kural.
PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from psycopg2.extras import RealDictCursor  # noqa: E402

from db_connection import get_connection  # noqa: E402
from pipeline.harmonization import _nifti_stem, canonical_source  # noqa: E402
from pipeline.resampling import is_isotropic_spacing, resample_image_only  # noqa: E402
from pipeline.segmentation import resolve_nas_path  # noqa: E402

# `run_harmonization_cohort.py`'nin ürettiği GERÇEK üretim yolları --
# tek kaynak burada TEKRARLANIYOR (import yerine), çünkü o modülü import
# etmek DB bağlantısı + tüm pipeline modüllerini erken yükler; sabitler
# değişmez/statik olduğu için kopya riski düşük, ama DEĞİŞİRSE burası da
# güncellenmeli (bkz. tools/run_harmonization_cohort.py WEEK2_ROOT/N4_ROOT/
# ZSCORE_ROOT/LUMIERE_RESAMPLED_ROOT/STATS_PATH sabitleri).
WEEK2_ROOT = PROJECT_ROOT / "artifacts" / "week2"
N4_ROOT = WEEK2_ROOT / "n4"
ZSCORE_ROOT = WEEK2_ROOT / "zscore"
LUMIERE_RESAMPLED_ROOT = WEEK2_ROOT / "lumiere_resampled_1mm"
STATS_PATH = ZSCORE_ROOT / "source_stats.json"

SOURCES = ("TCGA-GBM", "UPenn-GBM", "LUMIERE")

# Sıkı (cebirsel) kontrol toleransı: taramanın kendi N4-öncesi μ/σ'sinden
# + fit istatistiğinden BEKLENEN Z-score μ/σ'si ile diskteki GERÇEK
# çıktıdan GÖZLENEN μ/σ arasındaki fark. float32 yazım + çift hassasiyet
# hesap farkını tolere eder, gerçek bir mantık hatasını (örn. yanlış
# kaynağın istatistiği) YAKALAR.
STRICT_TOLERANCE = 5e-3

# Gevşek (örneklem-havuzlanmış) kontrol toleransı -- küçük örneklem
# nedeniyle tam 0/1 beklenmiyor, bilgilendirici bir eşik.
LOOSE_MEAN_TOLERANCE = 0.15
LOOSE_STD_TOLERANCE = 0.15


class ScanProcessingError(RuntimeError):
    """Bir taramanın raw/N4/Z-score aşamalarından biri okunamadı."""


def _derived_path(root: Path, input_path: Path, stage: str) -> Path:
    """`pipeline.harmonization._derived_output_path()` ile BİREBİR AYNI hash şeması.

    Global ortam değişkeni (`GBMAID_PROCESSED_ROOT`) mutasyona uğratmadan,
    parametrik `root` ile deterministik yolu yeniden üretir -- üç kaynağı
    art arda işlerken env-var yan etkisi riskini önler.
    """

    path_key = str(input_path.absolute()).casefold().encode("utf-8")
    input_id = hashlib.sha256(path_key).hexdigest()[:12]
    return root / input_id / f"{_nifti_stem(input_path)}_{stage}.nii.gz"


def _resampled_path(root: Path, image_path: Path) -> Path:
    """`pipeline.resampling._single_output_directory()` ile BİREBİR AYNI hash şeması."""

    key = str(image_path.absolute()).casefold().encode("utf-8")
    folder = root / hashlib.sha256(key).hexdigest()[:12]
    return folder / f"{_nifti_stem(image_path)}_1mm.nii.gz"


def _foreground_stats(array: np.ndarray) -> tuple[float, float, int]:
    """`pipeline.harmonization._foreground_values()` kuralıyla (sonlu VE sıfır-dışı)
    bağımsızca μ/σ/voksel-sayısı hesapla -- harmonization.py'nin kendi
    fonksiyonu ÇAĞRILMIYOR, üretim koduna paralel ama ayrı bir hesap yolu.
    """

    foreground = np.isfinite(array) & (array != 0)
    if not np.any(foreground):
        raise ScanProcessingError("Foreground (sonlu, sıfır-dışı) voksel bulunamadı.")
    values = array[foreground].astype(np.float64, copy=False)
    return float(values.mean()), float(values.std()), int(values.size)


def _read_stats(path: Path) -> tuple[float, float, int]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayViewFromImage(image)
    return _foreground_stats(array)


def _fetch_scans(connection: Any) -> dict[str, list[dict]]:
    cursor = connection.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(
            """
            SELECT ms.scan_id, ms.patient_id, ms.modality, ms.file_path,
                   ds.source_name AS source
            FROM mr_scans ms
            JOIN patients p ON p.patient_id = ms.patient_id
            JOIN dataset_sources ds ON ds.source_id = p.source_id
            WHERE ds.source_name IN ('TCGA-GBM', 'UPenn-GBM', 'LUMIERE')
              AND ms.harmonization_status = 'zscore'
            ORDER BY ds.source_name, ms.patient_id, ms.scan_id
            """
        )
        rows = list(cursor.fetchall())
    finally:
        cursor.close()

    by_source: dict[str, list[dict]] = {source: [] for source in SOURCES}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    return by_source


def _resolve_n4_output(
    scan: dict, raw_path: Path, source_canonical: str, *, allow_recompute: bool
) -> tuple[Path, str]:
    """N4 çıktısının yolunu döndür; ('cached'|'computed', path) durumuyla birlikte."""

    n4_input_path = raw_path
    if source_canonical == "LUMIERE" and not is_isotropic_spacing(raw_path):
        resampled = _resampled_path(LUMIERE_RESAMPLED_ROOT, raw_path)
        if resampled.is_file():
            n4_input_path = resampled
        elif allow_recompute:
            result = resample_image_only(raw_path, output_dir=None)
            n4_input_path = Path(result["image_output_path"])
        else:
            raise ScanProcessingError(
                f"LUMIERE 1mm resample ara çıktısı önbellekte yok: {resampled} "
                "(--allow-recompute verilmedi, üretilmedi)."
            )

    n4_path = _derived_path(N4_ROOT, n4_input_path, "n4")
    if n4_path.is_file():
        return n4_path, "cached"
    if not allow_recompute:
        raise ScanProcessingError(f"N4 önbellek çıktısı yok: {n4_path}")

    from pipeline.harmonization import apply_n4_bias_correction

    import os

    os.environ["GBMAID_PROCESSED_ROOT"] = str(N4_ROOT)
    computed = Path(apply_n4_bias_correction(str(n4_input_path)))
    return computed, "computed"


def _resolve_zscore_output(
    n4_path: Path, source_canonical: str, *, allow_recompute: bool
) -> tuple[Path, str]:
    zscore_path = _derived_path(ZSCORE_ROOT, n4_path, f"zscore_{source_canonical.lower()}")
    if zscore_path.is_file():
        return zscore_path, "cached"
    if not allow_recompute:
        raise ScanProcessingError(f"Z-score önbellek çıktısı yok: {zscore_path}")

    from pipeline.harmonization import apply_zscore_normalization

    import os

    os.environ["GBMAID_PROCESSED_ROOT"] = str(ZSCORE_ROOT)
    os.environ["GBMAID_ZSCORE_STATS_PATH"] = str(STATS_PATH)
    computed = Path(
        apply_zscore_normalization(str(n4_path), source_canonical)
    )
    return computed, "computed"


def _process_scan(
    scan: dict, fit_stats: dict[str, dict[str, float]], *, allow_recompute: bool
) -> dict[str, Any]:
    source_canonical = canonical_source(scan["source"])
    record: dict[str, Any] = {
        "scan_id": scan["scan_id"],
        "patient_id": scan["patient_id"],
        "source": scan["source"],
        "source_canonical": source_canonical,
        "modality": scan["modality"],
        "error": "",
    }

    try:
        raw_path = resolve_nas_path(scan["file_path"])
    except FileNotFoundError as exc:
        record["error"] = f"NAS_DOSYASI_YOK: {exc}"
        return record

    try:
        raw_mean, raw_std, raw_n = _read_stats(raw_path)
        record.update(
            raw_mean=raw_mean, raw_std=raw_std, raw_voxel_count=raw_n,
            raw_path=str(raw_path),
        )
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"RAW_OKUMA_HATASI: {type(exc).__name__}: {exc}"
        return record

    try:
        n4_path, n4_source = _resolve_n4_output(
            scan, raw_path, source_canonical, allow_recompute=allow_recompute
        )
        n4_mean, n4_std, n4_n = _read_stats(n4_path)
        record.update(
            n4_mean=n4_mean, n4_std=n4_std, n4_voxel_count=n4_n,
            n4_path=str(n4_path), n4_source=n4_source,
        )
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"N4_OKUMA_HATASI: {type(exc).__name__}: {exc}"
        return record

    try:
        zscore_path, zscore_source = _resolve_zscore_output(
            n4_path, source_canonical, allow_recompute=allow_recompute
        )
        zscore_mean, zscore_std, zscore_n = _read_stats(zscore_path)
        record.update(
            zscore_mean=zscore_mean, zscore_std=zscore_std,
            zscore_voxel_count=zscore_n, zscore_path=str(zscore_path),
            zscore_source=zscore_source,
        )
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"ZSCORE_OKUMA_HATASI: {type(exc).__name__}: {exc}"
        return record

    fit = fit_stats.get(source_canonical)
    if fit is None:
        record["error"] = f"FIT_ISTATISTIGI_YOK: {source_canonical} source_stats.json'da yok"
        return record

    global_mean = float(fit["mean"])
    global_std = float(fit["std"])
    predicted_mean = (n4_mean - global_mean) / global_std
    predicted_std = n4_std / global_std
    record.update(
        fit_global_mean=global_mean,
        fit_global_std=global_std,
        predicted_zscore_mean=predicted_mean,
        predicted_zscore_std=predicted_std,
        mean_diff=zscore_mean - predicted_mean,
        std_diff=zscore_std - predicted_std,
    )
    record["strict_pass"] = (
        abs(record["mean_diff"]) < STRICT_TOLERANCE
        and abs(record["std_diff"]) < STRICT_TOLERANCE
    )
    return record


def _pooled_combine(
    running: tuple[int, float, float], batch_mean: float, batch_std: float, batch_n: int
) -> tuple[int, float, float]:
    """Chan's paralel algoritması -- `fit_source_zscore_statistics()`'teki
    AYNI matematik, BAĞIMSIZ bir kopya (üretim fonksiyonu çağrılmıyor,
    stats dosyasına yazma riski yok).
    """

    count, mean, m2 = running
    batch_m2 = (batch_std ** 2) * batch_n
    combined_count = count + batch_n
    delta = batch_mean - mean
    mean += delta * batch_n / combined_count
    m2 += batch_m2 + delta * delta * count * batch_n / combined_count
    return combined_count, mean, m2


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Harmonizasyon öncesi/sonrası (raw/N4/Z-score) dağılım "
            "karşılaştırma doğrulama scripti (Ege, Hafta 2 -- salt-okunur)."
        )
    )
    parser.add_argument(
        "--n-per-source", type=int, default=6,
        help="Kaynak başına örneklem büyüklüğü (varsayılan 6, öneri 5-10).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Örneklem seed'i (tekrarlanabilirlik).")
    parser.add_argument(
        "--all", action="store_true",
        help="Örneklem yerine harmonization_status='zscore' TÜM taramaları işle.",
    )
    parser.add_argument(
        "--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES),
        help="Yalnız belirtilen kaynak(lar)ı işle.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Tarama-bazlı CSV rapor yolu (opsiyonel).")
    parser.add_argument(
        "--allow-recompute", action="store_true",
        help=(
            "Önbellekte eksik N4/Z-score ara çıktısını üretim fonksiyonlarıyla "
            "üret (DİSKE YAZAR). Varsayılan KAPALI -- script salt-okunur kalır."
        ),
    )
    args = parser.parse_args()

    if not STATS_PATH.is_file():
        print(f"HATA: fit istatistik dosyası bulunamadı: {STATS_PATH}", file=sys.stderr)
        return 2
    fit_stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))["sources"]

    connection = get_connection(readonly=True)
    try:
        by_source = _fetch_scans(connection)
    finally:
        connection.close()

    rng = random.Random(args.seed)
    all_records: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []

    for source in args.sources:
        pool = by_source.get(source, [])
        if not pool:
            print(f"UYARI: {source} için harmonization_status='zscore' tarama bulunamadı.")
            continue

        if args.all:
            sample = pool
        else:
            k = min(args.n_per_source, len(pool))
            sample = rng.sample(pool, k)

        print(f"\n=== Kaynak: {source} (örneklem={len(sample)} / havuz={len(pool)}) ===")

        raw_running = (0, 0.0, 0.0)
        n4_running = (0, 0.0, 0.0)
        zscore_running = (0, 0.0, 0.0)
        n_ok = 0
        n_error = 0
        max_abs_mean_diff = 0.0
        max_abs_std_diff = 0.0

        for scan in sample:
            record = _process_scan(scan, fit_stats, allow_recompute=args.allow_recompute)
            all_records.append(record)
            if record["error"]:
                n_error += 1
                print(f"  [HATA] scan_id={record['scan_id']} patient={record['patient_id']}: {record['error']}")
                continue

            n_ok += 1
            raw_running = _pooled_combine(raw_running, record["raw_mean"], record["raw_std"], record["raw_voxel_count"])
            n4_running = _pooled_combine(n4_running, record["n4_mean"], record["n4_std"], record["n4_voxel_count"])
            zscore_running = _pooled_combine(
                zscore_running, record["zscore_mean"], record["zscore_std"], record["zscore_voxel_count"]
            )
            max_abs_mean_diff = max(max_abs_mean_diff, abs(record["mean_diff"]))
            max_abs_std_diff = max(max_abs_std_diff, abs(record["std_diff"]))

            flag = "OK" if record["strict_pass"] else "SIKI-KONTROL-BASARISIZ"
            print(
                f"  scan_id={record['scan_id']:>6} patient={record['patient_id']:<16} "
                f"modality={record['modality']:<8} n4_source={record['n4_source']:<8} "
                f"zscore_source={record['zscore_source']:<8} "
                f"raw(mu={record['raw_mean']:.2f},sd={record['raw_std']:.2f}) -> "
                f"n4(mu={record['n4_mean']:.2f},sd={record['n4_std']:.2f}) -> "
                f"zscore(mu={record['zscore_mean']:.4f},sd={record['zscore_std']:.4f}) "
                f"[beklenen mu={record['predicted_zscore_mean']:.4f},sd={record['predicted_zscore_std']:.4f}] "
                f"fark(mu={record['mean_diff']:.2e},sd={record['std_diff']:.2e}) [{flag}]"
            )

        def _finalize(running: tuple[int, float, float]) -> tuple[float, float, int]:
            count, mean, m2 = running
            if count < 2:
                return float("nan"), float("nan"), count
            return mean, float(np.sqrt(m2 / count)), count

        raw_mean, raw_std, raw_n = _finalize(raw_running)
        n4_mean, n4_std, n4_n = _finalize(n4_running)
        zscore_mean, zscore_std, zscore_n = _finalize(zscore_running)

        loose_pass = (
            n_ok > 0
            and abs(zscore_mean) < LOOSE_MEAN_TOLERANCE
            and abs(zscore_std - 1.0) < LOOSE_STD_TOLERANCE
        )
        strict_pass = n_ok > 0 and max_abs_mean_diff < STRICT_TOLERANCE and max_abs_std_diff < STRICT_TOLERANCE

        summary = {
            "source": source,
            "n_sampled": len(sample),
            "n_ok": n_ok,
            "n_error": n_error,
            "raw_pooled_mean": raw_mean, "raw_pooled_std": raw_std, "raw_pooled_voxels": raw_n,
            "n4_pooled_mean": n4_mean, "n4_pooled_std": n4_std, "n4_pooled_voxels": n4_n,
            "zscore_pooled_mean": zscore_mean, "zscore_pooled_std": zscore_std, "zscore_pooled_voxels": zscore_n,
            "fit_global_mean": fit_stats.get(canonical_source(source), {}).get("mean"),
            "fit_global_std": fit_stats.get(canonical_source(source), {}).get("std"),
            "fit_cohort_image_count": fit_stats.get(canonical_source(source), {}).get("image_count"),
            "max_abs_strict_mean_diff": max_abs_mean_diff,
            "max_abs_strict_std_diff": max_abs_std_diff,
            "strict_pass": strict_pass,
            "loose_pass": loose_pass,
        }
        source_summaries.append(summary)

        print(
            f"\n  -- {source} ÖZET (havuzlanmış örneklem, n_ok={n_ok}/{len(sample)}) --\n"
            f"     RAW    : mu={raw_mean:10.4f}  sd={raw_std:10.4f}  (n_voxel={raw_n:,})\n"
            f"     N4     : mu={n4_mean:10.4f}  sd={n4_std:10.4f}  (n_voxel={n4_n:,})\n"
            f"     ZSCORE : mu={zscore_mean:10.4f}  sd={zscore_std:10.4f}  (n_voxel={zscore_n:,})  "
            f"[hedef: mu~=0, sd~=1]\n"
            f"     fit istatistiği (source_stats.json, tam kohort n_img={summary['fit_cohort_image_count']}): "
            f"mu={summary['fit_global_mean']:.4f} sd={summary['fit_global_std']:.4f}\n"
            f"     SIKI kontrol (her tarama: beklenen vs gözlenen, tolerans={STRICT_TOLERANCE:.0e}): "
            f"max|fark_mu|={max_abs_mean_diff:.2e} max|fark_sd|={max_abs_std_diff:.2e} -> "
            f"{'GECTI' if strict_pass else 'BASARISIZ'}\n"
            f"     GEVSEK kontrol (havuzlanmis ornek mu~=0/sd~=1, tolerans mu/sd={LOOSE_MEAN_TOLERANCE}/{LOOSE_STD_TOLERANCE}): "
            f"{'GECTI' if loose_pass else 'SAPMA VAR (kucuk orneklem nedeniyle beklenebilir, SIKI kontrol asil olcuttur)'}"
        )

    print("\n" + "=" * 78)
    print("GENEL ÖZET")
    print("=" * 78)
    overall_strict_pass = True
    for summary in source_summaries:
        overall_strict_pass = overall_strict_pass and summary["strict_pass"]
        print(
            f"{summary['source']:<12} n_ok={summary['n_ok']}/{summary['n_sampled']:<3} "
            f"zscore(mu={summary['zscore_pooled_mean']:.4f}, sd={summary['zscore_pooled_std']:.4f}) "
            f"sıkı-kontrol={'GECTI' if summary['strict_pass'] else 'BASARISIZ'} "
            f"gevşek-kontrol={'GECTI' if summary['loose_pass'] else 'SAPMA'}"
        )
    print(
        "\nNOT: 'sıkı kontrol' asıl doğrulamadır (her taramanın kendi N4 "
        "istatistiğinden + source_stats.json'daki fit değerinden BEKLENEN "
        "Z-score çıktısı, diskteki GERÇEK Z-score çıktısıyla cebirsel "
        "olarak eşleşiyor mu). 'gevşek kontrol' (havuzlanmış örneklem "
        "mu~=0/sd~=1) küçük örneklem nedeniyle bilgilendiricidir, tek "
        "başına FAIL kriteri değildir."
    )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for record in all_records for key in record})
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\nCSV rapor yazıldı: {args.csv}")

    return 0 if overall_strict_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
