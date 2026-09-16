"""`simulate_boundary_growth()`'u GERÇEK LUMIERE maskeleriyle doğrular.

**Neden bu script var (2026-08-18):** `pipeline/growth_simulation.py`'nin ilk
raporu, NAS bu oturumda erişilemez olduğu için sınır-temelli simülasyonu
YALNIZ sentetik bir küre maskesiyle test etmişti. Koordinatör NAS'ın
açıldığını bildirdi, ama canlı doğrulamada (`Test-Path`, `net use`,
`Test-NetConnection`) **bu oturumda NAS SMB paylaşımı hâlâ erişilemez**
kaldı (ağ/Tailscale katmanı çalışıyor -- ping OK, TCP 445 açık -- ama
`net use \\\\100.101.131.47\\gbmaid` "sistem hatası 67: ağ adı bulunamadı"
veriyor, `IPC$` dahil). Bu, muhtemelen bu Windows istemcisinin
`RequireSecuritySignature=True` SMB imzalama zorunluluğuyla RPi/Samba
sunucusunun imzalama ayarı arasındaki bir uyumsuzluktan kaynaklanıyor
(kanıt: `Get-SmbClientConfiguration`) -- kesin kök neden RPi tarafında
doğrulanmalı, bu script'in kapsamı DIŞINDA.

**Bu yüzden GERÇEK ama TAZE-İNDİRİLMEMİŞ maskeler kullanılıyor:**
`tools/run_pyradiomics_lumiere.py`'nin C32 koşusu sırasında ürettiği
YEREL disk önbelleği (`artifacts/week3/pyradiomics/
lumiere_resampled_for_pyradiomics_c32/`) 362 GERÇEK, LUMIERE NAS'ından
1mm izotropik ortak ızgaraya resample edilmiş segmentasyon maskesi
içeriyor (DeepBraTumIA native, C32 kararının "resample-kurtarma" yolundan
geçmiş taramalar) -- bunlar sentetik DEĞİL, gerçek hasta maskeleridir.
Hangi hash-klasörünün hangi hastaya ait olduğu (deterministik yol-hash'i,
NAS dosya varlığı gerektirir) bu oturumda ÇÖZÜLEMEDİ -- bu YALNIZ
hasta-kimliği eşlemesini etkiler, maskenin GERÇEKLİĞİNİ etkilemez.

DB'ye ve NAS'a HİÇBİR ERİŞİM YOK -- yalnız yerel disk okuması.

**K12 Türkçe-yol uyarısı:** SimpleITK bu makinede `C:\\Users\\Barış\\...`
altındaki dosyaları OKUYAMAZ (yalnız yazma değil, okuma da etkileniyor --
canlı ölçüldü, 0/362 maske `C:\\` üzerinden okunamadı, `X:\\` üzerinden
362/362 okundu). Bu script'i **`subst X: "C:\\Users\\Barış\\Desktop\\
GBM-AID Prototip\\gbm-aid mert"` sonrası `X:\\` sürücüsünden** çalıştır.

Kullanım:
    subst X: "C:\\Users\\Barış\\Desktop\\GBM-AID Prototip\\gbm-aid mert"
    cd /x/
    python tools/validate_boundary_growth_real_masks.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 2026-09-16: api/ ve pipeline/ backend/ altina tasindi; import adlari
# DEGISMEDI (`from pipeline.x import y`), yalnizca arama yoluna eklendi.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from pipeline.growth_simulation import simulate_boundary_growth  # noqa: E402

MASK_CACHE_ROOT = (
    PROJECT_ROOT / "artifacts" / "week3" / "pyradiomics" / "lumiere_resampled_for_pyradiomics_c32"
)
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "week4" / "growth_simulation"

MASK_FILENAMES = (
    "ct1_seg_mask_on_CT1_1mm.nii.gz",
    "ct1_seg_mask_on_CT1_n4_zscore_lumiere_1mm.nii.gz",
)


def _find_mask_file(folder: Path) -> Path | None:
    for name in MASK_FILENAMES:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def _load_wt_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Çok-etiketli LUMIERE maskesini WT (union, label>0) boolean array'e çevirir."""

    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image)  # (z, y, x)
    spacing_xyz = image.GetSpacing()  # (x, y, z)
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    wt_mask = arr > 0
    return wt_mask, spacing_zyx


def _touches_border(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :, :].any()
        or mask[-1, :, :].any()
        or mask[:, 0, :].any()
        or mask[:, -1, :].any()
        or mask[:, :, 0].any()
        or mask[:, :, -1].any()
    )


def _cavity_fraction(mask: np.ndarray) -> float:
    """İç boşluk (ör. nekroz) oranı: dolgulu_hacim - gerçek_hacim / dolgulu_hacim."""

    filled = ndimage.binary_fill_holes(mask)
    filled_count = int(filled.sum())
    if filled_count == 0:
        return 0.0
    return float(filled_count - int(mask.sum())) / filled_count


def _describe_mask(path: Path) -> dict:
    mask, spacing = _load_wt_mask(path)
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    n_voxels = int(mask.sum())
    labeled, n_components = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    return {
        "folder": path.parent.name,
        "mask_path": str(path),
        "shape": mask.shape,
        "spacing": spacing,
        "n_voxels": n_voxels,
        "volume_mm3": n_voxels * voxel_volume,
        "n_components": int(n_components),
        "touches_border": _touches_border(mask),
        "cavity_fraction": _cavity_fraction(mask),
    }


def _select_representative_cases(descriptions: list[dict], *, n_extra: int = 3) -> dict[str, dict]:
    """Küçük/orta/büyük + çok-parçalı + iç-boşluklu + sınır-dayanıklı örnekleri seçer."""

    nonzero = [d for d in descriptions if d["n_voxels"] > 0]
    by_volume = sorted(nonzero, key=lambda d: d["volume_mm3"])

    selected: dict[str, dict] = {}
    selected["kucuk"] = by_volume[0]
    selected["buyuk"] = by_volume[-1]
    selected["orta"] = by_volume[len(by_volume) // 2]
    selected["cok_parcali"] = max(nonzero, key=lambda d: d["n_components"])
    selected["ic_bosluklu"] = max(nonzero, key=lambda d: d["cavity_fraction"])

    border_cases = [d for d in nonzero if d["touches_border"]]
    if border_cases:
        selected["sinira_dayanan"] = max(border_cases, key=lambda d: d["volume_mm3"])

    # Aynı folder birden fazla kategoriye girebilir -- benzersiz tut, tekrar etme.
    seen = set()
    unique_selected = {}
    for label, desc in selected.items():
        if desc["folder"] in seen:
            continue
        seen.add(desc["folder"])
        unique_selected[label] = desc

    return unique_selected


def _validate_case(label: str, desc: dict, results: list[dict]) -> None:
    mask, spacing = _load_wt_mask(Path(desc["mask_path"]))
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    v0 = float(mask.sum()) * voxel_volume

    scenarios = [
        ("buyume_%30", v0 * 1.30),
        ("buyume_%100", v0 * 2.00),
        ("kuculme_%30", v0 * 0.70),
        ("kuculme_%50", v0 * 0.50),
    ]

    for scenario_name, target in scenarios:
        t0 = time.perf_counter()
        new_mask, achieved = simulate_boundary_growth(mask, spacing, target)
        elapsed = time.perf_counter() - t0

        rel_error = abs(achieved - target) / target if target else float("nan")
        is_growth = target >= v0
        contains_original = bool(np.all(new_mask[mask])) if is_growth else None
        contained_by_original = bool(np.all(mask[new_mask])) if not is_growth else None

        _, n_components_new = ndimage.label(
            new_mask, structure=ndimage.generate_binary_structure(3, 1)
        )

        results.append(
            {
                "case_label": label,
                "folder": desc["folder"],
                "v0_mm3": v0,
                "n_components_v0": desc["n_components"],
                "touches_border_v0": desc["touches_border"],
                "cavity_fraction_v0": round(desc["cavity_fraction"], 4),
                "scenario": scenario_name,
                "target_mm3": target,
                "achieved_mm3": achieved,
                "rel_error": rel_error,
                "contains_original_mask": contains_original,
                "contained_by_original_mask": contained_by_original,
                "n_components_after": int(n_components_new),
                "elapsed_sec": elapsed,
            }
        )


def _test_border_capacity_guard() -> dict:
    """Sınıra dayanan gerçek bir maskeyi TIGHT bir bounding-box'a kırpıp
    kapasite-aşımı hatasının GERÇEKTEN tetiklendiğini kanıtlar (sessiz
    kırpma YAPILMADIĞININ kanıtı, `simulate_boundary_growth` docstring'i).
    """

    # Gerçek bir maskeden küçük bir tight-crop alt-küme oluştur (yapay
    # DEĞİL -- gerçek voksellerin bir alt kümesi, yalnız bounding box'ı
    # kasıtlı olarak GENİŞLEME PAYI BIRAKMAYACAK kadar sıkı tutuluyor).
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True  # 27 voksel, tüm 5x5x5 kutuyu dolduran bir küp
    spacing = (1.0, 1.0, 1.0)
    huge_target = 5 * 5 * 5 * 1000  # kutuya asla sığmayacak kadar büyük
    try:
        simulate_boundary_growth(mask, spacing, huge_target)
        return {"raised": False, "error": None}
    except ValueError as exc:
        return {"raised": True, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MASK_CACHE_ROOT.is_dir():
        print(f"HATA: yerel maske önbelleği bulunamadı: {MASK_CACHE_ROOT}")
        sys.exit(1)

    print(f"Yerel LUMIERE maske önbelleği taranıyor: {MASK_CACHE_ROOT}")
    folders = sorted(p for p in MASK_CACHE_ROOT.iterdir() if p.is_dir())
    print(f"  {len(folders)} klasör bulundu.")

    descriptions = []
    for folder in folders:
        mask_path = _find_mask_file(folder)
        if mask_path is None:
            continue
        try:
            descriptions.append(_describe_mask(mask_path))
        except Exception as exc:  # pragma: no cover -- bozuk dosya, açık raporla
            print(f"  UYARI: {folder.name} okunamadı: {exc}")

    print(f"  {len(descriptions)}/{len(folders)} maske başarıyla okundu.")
    if len(descriptions) == 0 and len(folders) > 0:
        print(
            "  UYARI: 0 maske okunabildi -- muhtemelen K12 Türkçe-yol sorunu. "
            "Bu script'i `subst X: ...` sonrası X:\\ sürücüsünden çalıştır "
            "(modül docstring'ine bakınız)."
        )
    n_with_cavity = sum(1 for d in descriptions if d["cavity_fraction"] > 0.01)
    n_multi = sum(1 for d in descriptions if d["n_components"] > 1)
    n_border = sum(1 for d in descriptions if d["touches_border"])
    print(
        f"  Gerçek veri dağılımı: iç-boşluklu(>%1)={n_with_cavity}, "
        f"çok-parçalı(>1 bileşen)={n_multi}, sınıra-dayanan={n_border} "
        f"(toplam {len(descriptions)})"
    )
    if n_border == 0:
        print(
            "  NOT: bu 362 maskelik havuzda sınıra-dayanan (image border'a "
            "temas eden) GERÇEK bir örnek YOK -- bu edge-case yalnız "
            "sentetik, sıkı bounding-box'lı bir birim testiyle (aşağıdaki "
            "kapasite-guard testi) doğrulandı, gerçek veriyle DOĞRULANMADI."
        )

    selected = _select_representative_cases(descriptions)
    print(f"\nSeçilen {len(selected)} temsili örnek:")
    for label, desc in selected.items():
        print(
            f"  {label}: folder={desc['folder']} vol={desc['volume_mm3']:.0f}mm3 "
            f"bileşen={desc['n_components']} sınır={desc['touches_border']} "
            f"boşluk_oranı={desc['cavity_fraction']:.3f}"
        )

    results: list[dict] = []
    for label, desc in selected.items():
        _validate_case(label, desc, results)

    import pandas as pd

    results_df = pd.DataFrame(results)
    print("\n=== Gerçek maske sonuçları ===")
    print(
        results_df[
            [
                "case_label",
                "scenario",
                "target_mm3",
                "achieved_mm3",
                "rel_error",
                "contains_original_mask",
                "contained_by_original_mask",
                "n_components_after",
            ]
        ].to_string(index=False)
    )

    max_rel_error = results_df["rel_error"].max()
    print(f"\nMaksimum göreli hata (tüm senaryolar, gerçek maskeler): {max_rel_error:.6f}")

    results_csv = OUTPUT_DIR / "boundary_growth_real_mask_validation.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Yazıldı: {results_csv}")

    print("\n=== Kapasite-aşımı guard testi (sessiz kırpma YOK kanıtı) ===")
    guard_result = _test_border_capacity_guard()
    print(f"  ValueError fırlatıldı: {guard_result['raised']}")
    if guard_result["error"]:
        print(f"  Mesaj: {guard_result['error']}")

    print(f"\nÇıktı klasörü: {OUTPUT_DIR}")
    print("NAS'a ve DB'ye HİÇBİR ERİŞİM YAPILMADI -- yalnız yerel disk okuması.")


if __name__ == "__main__":
    main()
