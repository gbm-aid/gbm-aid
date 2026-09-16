"""Kohortların hazır segmentasyon maskelerini güvenli biçimde çözümle.

TCGA, UPenn ve LUMIERE için bu modül hiçbir koşulda nnU-Net çalıştırmaz.
BraTS pretrained nnU-Net yalnızca ayrı yeni-hasta demo akışına aittir.

Hafta 2 kararı (Barış, NAS'ta canlı doğrulandı): LUMIERE ana WT/TC/ET kaynağı
DeepBraTumIA'dır, HD-GLIO-AUTO yalnız duyarlılık analizi içindir (bkz.
hafta2_durum_ozeti.md). Gerekçe: DeepBraTumIA 3 bölge verir (Necrosis dahil,
UPenn'in NC/ED/ET tanımına daha yakın) ve native maskesi mr_scans'te saklanan
HD-GLIO-önişlenmiş görüntü yerine LUMIERE'in ham (root) görüntüsüyle eşleşir
-- böylece N4/resample/Z-score zinciri üçüncü parti (HD-GLIO'nun kendi bet/
reg) önişlemesi yerine projenin kendi kontrollü pipeline'ından geçer.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from pipeline.harmonization import _nifti_stem, canonical_source
from pipeline.resampling import validate_image_mask_geometry


class ReadyMaskNotFoundError(FileNotFoundError):
    """Bir kohort taraması için beklenen hazır maske bulunamadı."""


class ImageMaskGeometryError(ValueError):
    """Hazır maske ile görüntünün fiziksel geometrisi eşleşmiyor."""


def nas_root() -> Path:
    """NAS kökünü ortamdan oku; ağ adresini koda gömme."""

    configured = os.environ.get("GBMAID_NAS_ROOT")
    if not configured:
        raise KeyError(
            "GBMAID_NAS_ROOT tanımlı değil. NAS kökünü proje .env dosyasına ekle."
        )
    root = Path(configured).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"NAS kökü erişilebilir değil: {root}")
    return root


#: UCSF hasta klasorleri her zaman `UCSF-PDGM-<no>_nifti` desenindedir; baska
#: hicbir kaynakta bu desen YOKTUR, bu yuzden yolun KENDISINDEN hangi koke ait
#: oldugu anlasilir ve `resolve_nas_path()`'e ayri bir `source` parametresi
#: eklemek GEREKMEZ (o fonksiyon ~10 yerden cagriliyor).
_UCSF_DIZIN_ONEKI = "UCSF-PDGM-"


def ucsf_root() -> Path:
    """UCSF-PDGM goruntu kokunu ortamdan oku.

    🔴 **UCSF NAS'ta DEGILDIR** (2026-09-15 olcumu) -- ayri bir diskte/dizinde
    durur, bu yuzden kendi koku vardir. `GBMAID_UCSF_ROOT` tanimli degilse
    `GBMAID_NAS_ROOT`'a duser; bu bir "sessiz fallback" DEGIL, belgelenmis
    varsayilandir: dagitim sunucusunda UCSF de NAS koku altina kopyalanacagi
    icin (karar: 6,47 GB secici kopya) orada tek kok yeterlidir. Gelistirme
    makinesinde UCSF yerel diskte durdugu icin degisken ACIKCA tanimlanir.

    Karar: decisions/2026-09-15-ucsf-mr-goruntuleri-canlida-gosterilecek.md
    """

    configured = os.environ.get("GBMAID_UCSF_ROOT")
    if not configured:
        return nas_root()
    root = Path(configured).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"UCSF kökü erişilebilir değil: {root}")
    return root


def resolve_nas_path(
    stored_path: str | Path,
    *,
    root: str | Path | None = None,
    require_file: bool = True,
) -> Path:
    """DB'deki POSIX göreli yolu yerel/UNC NAS yoluna dönüştür."""

    raw = str(stored_path).strip()
    if not raw:
        raise ValueError("Boş dosya yolu çözümlenemez.")

    direct = Path(raw).expanduser()
    if direct.is_absolute():
        resolved = direct
    else:
        relative = PurePosixPath(raw.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Güvensiz NAS göreli yolu: {stored_path}")
        if root is not None:
            base = Path(root).expanduser()
        elif relative.parts and relative.parts[0].startswith(_UCSF_DIZIN_ONEKI):
            # 2026-09-16: UCSF kendi kokunde durur -- bkz. `ucsf_root()`.
            base = ucsf_root()
        else:
            base = nas_root()
        resolved = base.joinpath(*relative.parts)

    if require_file and not resolved.is_file():
        raise FileNotFoundError(f"NAS dosyası bulunamadı: {resolved}")
    return resolved


def _find_nifti(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _tcga_mask(image_path: Path) -> tuple[Path, str, list[str]]:
    whole = _find_nifti(image_path.parent, ("whole.nii.gz", "whole.nii"))
    if whole is not None:
        return whole, "provided_whole_tumor", []

    core = _find_nifti(image_path.parent, ("core.nii.gz", "core.nii"))
    if core is not None:
        return (
            core,
            "provided_tumor_core",
            [
                "TCGA whole-tumor maskesi bulunamadı; core maskesi seçildi. "
                "ROI tanımı farklı olduğu için radyomik karşılaştırmada açıkça "
                "etiketlenmelidir."
            ],
        )
    raise ReadyMaskNotFoundError(
        f"TCGA hazır whole/core maskesi bulunamadı: {image_path.parent}"
    )


def _upenn_mask(image_path: Path) -> tuple[Path, str, list[str]]:
    parts = list(image_path.parts)
    try:
        nifti_index = next(
            index
            for index, part in enumerate(parts)
            if part.casefold() == "nifti-files"
        )
    except StopIteration as exc:
        raise ValueError(
            f"UPenn yolu NIfTI-files dizinini içermiyor: {image_path}"
        ) from exc

    scan_key = image_path.parent.name
    nifti_root = Path(*parts[: nifti_index + 1])
    expert = _find_nifti(
        nifti_root / "images_segm",
        (f"{scan_key}_segm.nii.gz", f"{scan_key}_segm.nii"),
    )
    if expert is not None:
        return expert, "upenn_expert", []

    automated = _find_nifti(
        nifti_root / "automated_segm",
        (
            f"{scan_key}_automated_approx_segm.nii.gz",
            f"{scan_key}_automated_approx_segm.nii",
        ),
    )
    if automated is not None:
        return (
            automated,
            "upenn_automated_approx",
            [
                "UPenn expert maskesi yok; veri setinin hazır automated_approx "
                "maskesi seçildi. Bu seçim nnU-Net fallback değildir."
            ],
        )
    raise ReadyMaskNotFoundError(
        f"UPenn hazır expert/automated maske bulunamadı: {scan_key}"
    )


def _lumiere_week_root(image_path: Path) -> Path:
    """`.../week-XXX/<ARAÇ>-segmentation/registered/<dosya>` -> `.../week-XXX`."""

    return image_path.parents[2]


def _lumiere_modality_prefix(image_path: Path) -> str:
    """Dosya adından modalite önekini çıkar: `CT1_r2s_bet_reg.nii.gz` -> `CT1`."""

    return _nifti_stem(image_path).split("_")[0]


def _lumiere_mask(image_path: Path) -> tuple[Path, Path, str, list[str]]:
    """LUMIERE için ana kaynak DeepBraTumIA'dır; bulunamazsa HD-GLIO'ya (duyarlılık
    analizi aracı) düşülür ve bu açıkça uyarı olarak işaretlenir.

    Döner: (mask_path, radyomik için kullanılacak_görüntü_path, mask_source, warnings)
    """

    week_root = _lumiere_week_root(image_path)
    modality = _lumiere_modality_prefix(image_path)

    deepbratumia_mask = _find_nifti(
        week_root / "DeepBraTumIA-segmentation" / "native" / "segmentation",
        (
            f"{modality.lower()}_seg_mask.nii.gz",
            f"{modality.lower()}_seg_mask.nii",
        ),
    )
    if deepbratumia_mask is not None:
        raw_image = _find_nifti(week_root, (f"{modality}.nii.gz", f"{modality}.nii"))
        if raw_image is None:
            raise ReadyMaskNotFoundError(
                "DeepBraTumIA native maskesi var ama eşleşen ham LUMIERE "
                f"görüntüsü bulunamadı: {week_root}/{modality}.nii.gz"
            )
        return deepbratumia_mask, raw_image, "lumiere_deepbratumia_native", []

    hd_glio_mask = _find_nifti(
        image_path.parent,
        ("segmentation.nii.gz", "segmentation.nii"),
    )
    if hd_glio_mask is not None:
        return (
            hd_glio_mask,
            image_path,
            "lumiere_hd_glio_fallback",
            [
                "DeepBraTumIA native maskesi bulunamadı; ana kaynak yerine "
                "duyarlılık analizi aracı olan HD-GLIO-AUTO geçici olarak "
                "kullanıldı. Bu vaka DeepBraTumIA verisi tamamlanınca yeniden "
                "işlenmelidir."
            ],
        )

    raise ReadyMaskNotFoundError(
        f"LUMIERE hazır DeepBraTumIA/HD-GLIO maskesi bulunamadı: {week_root}"
    )


def _ucsf_mask(image_path: Path) -> tuple[Path, str, list[str]]:
    """UCSF-PDGM hazır tümör maskesini çöz: `<ad>_T1c` -> `<ad>_tumor_segmentation`.

    🔴 2026-09-16'da `demo/demo_helpers.py::_resolve_ucsf_display_pair()`'den
    ÜRETİME TAŞINDI (kopyalanmadı). Gerekçe `decisions/2026-09-15-ucsf-mr-
    goruntuleri-canlida-gosterilecek.md` Adım 1: kural demo katmanında
    çalışıyordu ama `resolve_ready_mask()` UCSF'i HİÇ tanımıyordu, dolayısıyla
    üretim yolunda UCSF maskesi çözülemiyordu. Demo artık bu fonksiyonu ÇAĞIRIR
    -- ikinci nüsha YOKTUR (`CLAUDE.md` kalem-41: *"aynı kuralın üçüncü kopyası
    kaçınılmaz olarak sapar"*).

    Sözleşme kaynağı: `tools/run_pyradiomics_ucsf.py`. UCSF'te hasta başına tek
    zaman noktası ve tek maske vardır; TCGA'daki `whole`/`core` geri düşüşü veya
    UPenn'deki `automated_approx` geri düşüşü UCSF için TANIMLI DEĞİLDİR --
    maske yoksa sessizce başka bir şeye düşmek YASAK, hata yükseltilir.
    """

    stem = _nifti_stem(image_path)
    if not stem.endswith("_T1c"):
        raise ReadyMaskNotFoundError(
            f"UCSF T1c dosya adı beklenen desende değil ('*_T1c'): {image_path}"
        )
    base = stem[: -len("_T1c")]
    for suffix in (".nii.gz", ".nii"):
        candidate = image_path.parent / f"{base}_tumor_segmentation{suffix}"
        if candidate.is_file():
            return candidate, "ucsf_native", []
    raise ReadyMaskNotFoundError(
        "UCSF hazır tümör maskesi bulunamadı: "
        f"{image_path.parent / (base + '_tumor_segmentation.nii.gz')}"
    )


def resolve_ready_mask(
    *,
    source: str,
    image_path: str | Path,
    patient_id: str,
    scan_id: str | int,
    raise_on_geometry_mismatch: bool = True,
) -> dict[str, object]:
    """Kaynağa uygun hazır maskeyi seç ve tam geometri eşleşmesini zorunlu kıl.

    `raise_on_geometry_mismatch=False` yalnız çağıranın kendi açık resampling
    kurtarma adımını (bkz. `pipeline/resampling.py::resample_image_and_mask`,
    decisions/2026-08-09-lumiere-resampling-entegrasyonu-onaylandi.md)
    tetikleyebilmesi için görüntü/maske yollarını geometri eşleşmese bile
    döndürür -- bu ham çıktı PyRadiomics'e ASLA doğrudan geçirilmemelidir;
    çağıran taraf `geometry["geometry_match"]` alanını kontrol edip önce
    resample etmeli, ardından yeniden geometri doğrulaması yapmalıdır
    (mimari kural: sessiz fallback yasak). Varsayılan (`True`) mevcut sıkı
    davranışı korur.
    """

    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(f"MR görüntüsü bulunamadı: {image}")

    canonical = canonical_source(source)
    if canonical == "TCGA":
        mask, mask_source, warnings = _tcga_mask(image)
        resolved_image = image
    elif canonical == "UPenn":
        mask, mask_source, warnings = _upenn_mask(image)
        resolved_image = image
    elif canonical == "LUMIERE":
        mask, resolved_image, mask_source, warnings = _lumiere_mask(image)
    elif canonical == "UCSF":
        # 2026-09-16: UCSF kolu EKLENDI -- daha once bu dal YOKTU ve UCSF
        # "tanimsiz kaynak" diye reddediliyordu (bkz. `_ucsf_mask` dokstring'i).
        mask, mask_source, warnings = _ucsf_mask(image)
        resolved_image = image
    else:
        raise ValueError(f"Hazır maske sözleşmesi tanımsız kaynak: {canonical}")

    geometry = validate_image_mask_geometry(resolved_image, mask)
    if not geometry["geometry_match"] and raise_on_geometry_mismatch:
        raise ImageMaskGeometryError(
            f"{canonical} görüntü/maske geometrisi eşleşmiyor; "
            "PyRadiomics öncesi kaynak sözleşmesine uygun resampling gerekli. "
            f"Görüntü: {resolved_image}; maske: {mask}"
        )

    return {
        "patient_id": patient_id,
        "scan_id": scan_id,
        "source": canonical,
        "image_path": str(resolved_image),
        "mask_path": str(mask),
        "mask_source": mask_source,
        "confidence_score": None,
        "geometry": geometry,
        "warnings": warnings,
    }
