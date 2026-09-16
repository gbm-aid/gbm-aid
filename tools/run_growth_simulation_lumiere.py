"""LUMIERE-kalibrasyonlu büyüme eğrisi fit'i + RANO grup dağılımı + sınır-
temelli simülasyon demo'su (Hafta 4, Görev 1-3).

`pipeline/growth_simulation.py`'deki fonksiyonları GERÇEK DB verisiyle
çalıştırır. DB'ye **HİÇBİR YAZMA YAPMAZ** (yalnız `readonly=True` SELECT).
Çıktı CSV'leri `artifacts/week4/growth_simulation_v2/`'ye yazılır (v1
`artifacts/week4/growth_simulation/`'ın ÜZERİNE YAZILMAZ — Codex
task-mszut9wy-qx8sal düzeltmeleri, 2026-08-19, bkz. bu script'in ve
`pipeline/growth_simulation.py`'nin "Codex çapraz-inceleme düzeltmeleri"
bölümleri). Dry-run rapor niteliğindedir — `followup_series.tumor_volume_mm3`
doldurulması AYRI, onaylı bir görevdir, bu script'in kapsamı DIŞINDA.

Kullanım:
    python tools/run_growth_simulation_lumiere.py
    python tools/run_growth_simulation_lumiere.py --min-visits 3 --min-group-n 6
    python tools/run_growth_simulation_lumiere.py --output-suffix v3

2026-09-11 Barış kararı (BEKLEYEN-KARARLAR #5, `SD n=5 interpretable=True`
tutarsızlığı): `pipeline.growth_simulation.growth_rate_by_rano_group[_and_model]`
artık İKİ koşulu birden uygular -- (a) `n >= min_group_n` (eşik 5 -> 6
yükseltildi), (b) grup IQR'ı referans (en büyük n'li) grubun IQR'ının
`--max-iqr-ratio-to-reference` katını (varsayılan 5x) aşmasın. Bu script'in
`--min-group-n` CLI varsayılanı da BURADA 5 -> **6** olarak güncellendi
(pipeline'ın kendi varsayılanıyla hizalı) -- ayrıntı ve gerekçe:
`pipeline/growth_simulation.py` docstring'i "2026-09-11 Barış kararı"
bölümü ve `decisions/2026-09-11-sd-n5-interpretable-false.md`.
`--output-suffix` (varsayılan `v2`, ESKİ davranışı KORUR) yeni bir çıktı
klasörüne (`artifacts/week4/growth_simulation_<suffix>/`) yazmayı sağlar
-- v2 dosyalarının ÜZERİNE YAZILMAMASI için `v3` gibi bir değer verilmeli.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db_connection import get_connection  # noqa: E402
from pipeline.growth_simulation import (  # noqa: E402
    assign_patient_rano_group,
    compute_transition_growth_rates,
    fetch_lumiere_rano_labels,
    fetch_lumiere_wt_volume_series,
    fit_patient_growth_curves,
    growth_fits_to_dataframe,
    growth_rate_by_rano_group,
    growth_rate_by_rano_group_and_model,
    project_volume_range,
    simulate_boundary_growth,
    summarize_unknown_rano_labels,
    uninterpretable_group_names,
)

DEFAULT_OUTPUT_SUFFIX = "v2"


def _make_sphere_mask(shape, center, radius) -> np.ndarray:
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    dist = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    return dist <= radius


def _run_boundary_simulation_demo() -> None:
    """Sınır-temelli simülasyonu SENTETİK bir küre maskesiyle gösterir.

    **NEDEN GERÇEK BİR LUMIERE MASKESİ DEĞİL:** bu script'in çalıştığı
    oturumda NAS'a (`\\\\100.101.131.47\\gbmaid`, Tailscale üzerinden)
    erişim YOKTU (`Test-Path` False döndü, 2026-08-18) — bu DOĞRULANMIŞ bir
    kısıt, varsayım DEĞİL. Fonksiyonun kendisi herhangi bir 3D boolean
    numpy maskesiyle çalışır; gerçek bir LUMIERE maskesiyle canlı
    doğrulama AYRI bir NAS-erişimli oturumda yapılmalıdır.
    """

    print("\n=== Sınır-temelli morfolojik simülasyon (SENTETİK demo) ===")
    print(
        "UYARI: NAS bu oturumda erişilemez durumdaydı, bu yüzden gerçek bir "
        "LUMIERE maskesi YERİNE sentetik bir küre maskesi kullanılıyor. "
        "Fonksiyonun kendisi gerçek maskelerle de ÇALIŞIR (bkz. "
        "tests/test_growth_simulation.py::test_simulate_boundary_growth_* "
        "-- anizotropik spacing dahil test edildi), yalnız bu demo'nun "
        "GİRDİSİ sentetiktir."
    )

    mask = _make_sphere_mask((60, 60, 60), (30, 30, 30), radius=12.0)
    spacing = (1.0, 1.0, 1.0)
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    v0 = float(mask.sum()) * voxel_volume

    # RANO grubu bazında ölçülen r-IQR'sinden örnek bir üçlü (rapor
    # çalıştırıldıktan sonra gerçek sayılarla değiştirilecek, aşağıdaki
    # `main()` bu fonksiyonu gerçek IQR ile de çağırıyor)
    projection = project_volume_range(
        v0=v0,
        model="gompertz",
        template_params=(v0 * 4.0, 2.5, 0.06),
        r_low=0.03,
        r_median=0.06,
        r_high=0.10,
        months=6.0,
    )
    print(
        f"V0={v0:.0f} mm3 -> 6 ay projeksiyon: "
        f"%{projection.pct_low:.1f} - %{projection.pct_high:.1f} arası "
        f"(medyan %{projection.pct_median:.1f})"
    )

    for label, target_v in (
        ("düşük (r_low)", projection.v_low),
        ("medyan (r_median)", projection.v_median),
        ("yüksek (r_high)", projection.v_high),
    ):
        new_mask, achieved = simulate_boundary_growth(mask, spacing, target_v)
        contains_original = bool(np.all(new_mask[mask])) if target_v >= v0 else bool(
            np.all(mask[new_mask])
        )
        print(
            f"  {label}: hedef={target_v:.0f}mm3 ulaşılan={achieved:.0f}mm3 "
            f"(orijinali_içeriyor={contains_original})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-visits", type=int, default=3)
    parser.add_argument(
        "--min-group-n",
        type=int,
        default=6,
        help=(
            "2026-09-11 Barış kararı ile 5 -> 6 yükseltildi -- "
            "pipeline/growth_simulation.py'nin kendi varsayılanıyla hizalı."
        ),
    )
    parser.add_argument(
        "--max-iqr-ratio-to-reference",
        type=float,
        default=5.0,
        help=(
            "Grup IQR'ı, en büyük n'li ('referans') grubun IQR'ının bu katını "
            "aşarsa interpretable=False (2026-09-11 Barış kararı). "
            "0 veya negatif verilirse (b) koşulu KAPATILIR (yalnız n uygulanır)."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=DEFAULT_OUTPUT_SUFFIX,
        help=(
            "Çıktı klasörü `artifacts/week4/growth_simulation_<suffix>/` olur. "
            "Varsayılan 'v2' ESKİ davranışı korur -- v2 dosyalarının ÜZERİNE "
            "YAZILMAMASI için (örn. 2026-09-11 SD-interpretable düzeltmesi "
            "sonrası yeniden üretimde) 'v3' gibi bir değer VERİLMELİDİR."
        ),
    )
    parser.add_argument("--skip-boundary-demo", action="store_true")
    args = parser.parse_args()

    max_iqr_ratio = (
        args.max_iqr_ratio_to_reference if args.max_iqr_ratio_to_reference > 0 else None
    )
    output_dir = PROJECT_ROOT / "artifacts" / "week4" / f"growth_simulation_{args.output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = get_connection(readonly=True)
    try:
        print("LUMIERE WT hacim serisi okunuyor (C32, SADECE SELECT)...")
        series_df = fetch_lumiere_wt_volume_series(conn)
        n_patients_total = series_df["patient_id"].nunique()
        n_scans_total = len(series_df)
        print(f"  {n_scans_total} tarama, {n_patients_total} hasta.")

        rano_df = fetch_lumiere_rano_labels(conn)
        print(f"  {len(rano_df)} RANO kaydı (followup_series, SADECE SELECT).")
    finally:
        conn.close()

    print("\n=== Görev 1: Gompertz/lojistik büyüme eğrisi fit'i ===")
    fits = fit_patient_growth_curves(series_df, min_visits=args.min_visits)
    fits_df = growth_fits_to_dataframe(fits)

    status_counts = fits_df["fit_status"].value_counts().to_dict()
    print(f"  Durum dağılımı: {status_counts}")

    ok_df = fits_df[fits_df["fit_status"] == "ok"]
    if not ok_df.empty:
        model_counts = ok_df["best_model"].value_counts().to_dict()
        print(f"  Başarılı fit'lerde en-iyi-model dağılımı (AIC/R²): {model_counts}")
        print(
            f"  best_r özet: medyan={ok_df['best_r'].median():.4f} "
            f"IQR=[{ok_df['best_r'].quantile(0.25):.4f}, "
            f"{ok_df['best_r'].quantile(0.75):.4f}]"
        )

    fits_csv = output_dir / "patient_growth_fits.csv"
    fits_df.to_csv(fits_csv, index=False)
    print(f"  Yazıldı: {fits_csv}")

    # --- MEDIUM4: bilinmeyen/bileşik RANO etiketleri AÇIKÇA raporlanır ----
    unknown_df = summarize_unknown_rano_labels(rano_df)
    print(
        f"\n  RANO etiket kalitesi: {len(rano_df)} toplam satır, "
        f"{len(unknown_df)} 'unknown' kategoride (bileşik/boş/tanınmayan -- "
        "SESSİZCE PD/başka bir kategoriye dönüştürülmedi)."
    )
    if not unknown_df.empty:
        print(unknown_df[["patient_id", "visit_week", "rano_label_raw"]].to_string(index=False))
    unknown_csv = output_dir / "unknown_rano_labels.csv"
    unknown_df.to_csv(unknown_csv, index=False)
    print(f"  Yazıldı: {unknown_csv}")

    print("\n=== Görev 2: RANO grubu bazında büyüme katsayısı (r) dağılımı ===")
    print(
        "  UYARI -- BİLİNEN SINIRLAMA (Codex HIGH1): aşağıdaki `best_r`, hastanın "
        "TÜM zaman serisine fit edilmiş TEK bir değerdir ve yalnız EN SON "
        "RANO etiketiyle eşlenir -- uzun süre SD/PR seyredip son vizitte "
        "PD'ye dönen bir hastanın TÜM geçmişi PD grubuna yazılır. Geçiş-"
        "bazlı (her etiket KENDİ dönemine bağlı) tercih edilen alternatif "
        "için aşağıdaki 'Görev 2b' ve `artifacts/week4/followup_t3_v2/"
        "delta_rano_threshold_table_v2.csv`'deki `log_ratio_rate_per_week_*` "
        "sütunlarına bakın."
    )
    rano_group_df = assign_patient_rano_group(rano_df)
    print(
        f"  {len(rano_group_df)}/{n_patients_total} hastaya bir RANO-yanıt grubu "
        "atanabildi (Pre-Op/Post-Op-only hastalar hariç)."
    )
    group_dist_df = growth_rate_by_rano_group(
        fits_df,
        rano_group_df,
        min_group_n=args.min_group_n,
        max_iqr_ratio_to_reference=max_iqr_ratio,
    )
    print(group_dist_df.to_string(index=False) if not group_dist_df.empty else "  (boş)")
    bad_groups = uninterpretable_group_names(group_dist_df, group_col="rano_group")
    if bad_groups:
        print(
            f"  UYARI -- YORUMLANAMAZ (n < min_group_n={args.min_group_n} VEYA "
            f"IQR > {args.max_iqr_ratio_to_reference}x referans-IQR, 2026-09-11 "
            f"Barış kararı): {bad_groups}. Bu gruplarla (veya bunları içeren "
            "HERHANGİ bir ikili karşılaştırmayla, örn. 'PD vs PR') yön/"
            "karşılaştırma iddiası KURULAMAZ (Codex HIGH2)."
        )
    group_csv = output_dir / f"growth_rate_by_rano_group_{args.output_suffix}.csv"
    group_dist_df.to_csv(group_csv, index=False)
    print(f"  Yazıldı: {group_csv}")

    print("\n=== Görev 2 (LOW) — model-tipine göre tabakalı r dağılımı ===")
    group_model_df = growth_rate_by_rano_group_and_model(
        fits_df,
        rano_group_df,
        min_group_n=args.min_group_n,
        max_iqr_ratio_to_reference=max_iqr_ratio,
    )
    print(group_model_df.to_string(index=False) if not group_model_df.empty else "  (boş)")
    group_model_csv = output_dir / f"growth_rate_by_rano_group_and_model_{args.output_suffix}.csv"
    group_model_df.to_csv(group_model_csv, index=False)
    print(f"  Yazıldı: {group_model_csv}")

    print("\n=== Görev 2b (Codex HIGH1) — geçiş-bazlı yerel büyüme hızı (TÜM seri) ===")
    transitions_df = compute_transition_growth_rates(series_df)
    print(
        f"  {len(transitions_df)} geçiş (RANO'dan BAĞIMSIZ -- yalnız hacim "
        "serisi üzerinden). RANO-etiketli, RANO-grubu bazında özetlenmiş "
        "hâli için `followup_t3_v2/delta_rano_threshold_table_v2.csv` "
        "(aynı formül, `log_ratio_growth_rate`) kullanılmalıdır."
    )
    transitions_csv = output_dir / f"patient_transition_growth_rates_{args.output_suffix}.csv"
    transitions_df.to_csv(transitions_csv, index=False)
    print(f"  Yazıldı: {transitions_csv}")

    if not args.skip_boundary_demo:
        _run_boundary_simulation_demo()

    print("\n=== Özet ===")
    print(f"Çıktı klasörü: {output_dir}")
    print("DB'ye HİÇBİR YAZMA yapılmadı (readonly=True, yalnız SELECT).")
    print(
        "v1 çıktıları (artifacts/week4/growth_simulation/) ve v2 çıktıları "
        "(artifacts/week4/growth_simulation_v2/, --output-suffix v2 varsayılanı "
        "KULLANILMADIYSA) DEĞİŞTİRİLMEDİ."
    )


if __name__ == "__main__":
    main()
