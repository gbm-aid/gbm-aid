"""``GET /patient/{patient_id}/mr_slice`` -- MR kesiti (PNG, opsiyonel
hazır-maske overlay'i).

Görev talimatı: paralel-36 (backend-agent-W1, 2026-09-15). Tasarımda
"Blok MRI" şu an her hastada `available:false` -- bu uç nokta gerçek
kesiti üretir.

=======================================================================
KAYNAK -- `demo/demo_helpers.py`'DEN REFERANS ALINDI, KOPYALANMADI/
IMPORT EDİLMEDİ
=======================================================================
Görev talimatı: "Mantığı YENİDEN YAZMA -- demo/demo_helpers.py'deki
çalışan çözümü referans al ... O kodu oku ve aynı sözleşmeyi uygula;
demo dosyasını DEĞİŞTİRME." Bu modül `demo/demo_helpers.py`'yi
`import` ETMEZ -- iki gerekçe: (1) `demo/` bu görevde DOKUNULMAZ/salt-
referans olarak işaretli, `api/`nin `demo/`ye bağımlı olması (ters
bağımlılık yönü: demo zaten `api/`yi import ediyor) mimari olarak
tersine döner; (2) `demo/demo_helpers.py` bir "jüri demosu" modülüdür
(Streamlit'e özgü sys.path/8.3-kısa-yol/ctypes yardımcıları da taşır),
üretim endpoint'inin ona bağımlı olması istenmeyen bir bağ eklerdi.

Bunun yerine AŞAĞIDAKİ fonksiyonlar demo'nun sözleşmesini (aynı adım
sırası, aynı geometri/eksen kararları) BAĞIMSIZ olarak yeniden uygular,
ortak alt-katman (`pipeline.segmentation.resolve_ready_mask`,
`pipeline.harmonization.canonical_source`/`_nifti_stem`,
`pipeline.radiomics_volume.REGION_LABELS_BY_MASK_SOURCE`) ÜZERİNDEN --
bu üçü zaten TEK KAYNAK'tır, demo da BUNLARI kullanıyordu.
⚠️ BİLİNEN BAKIM RİSKİ (açıkça beyan edilir, gizlenmez): geometri
doğrulama + eksenel-kesit + pencereleme + overlay-boyama kodu (~120
satır) `demo/demo_helpers.py:956-1122` ile BİREBİR aynı algoritmayı
uygular ama fiziksel olarak AYRI bir kopyadır -- ikisi arasında drift
riski VAR. Bu risk, `pipeline.source_canonical`'ın 2026-09-14'te
ayrıştırılmasına benzer bir refactor (ortak mantığı `pipeline/`e
taşımak) ile kapatılabilir; bu görevin kapsamı dışındadır, Barış'a
BULGU olarak raporlanır.

=======================================================================
LUMIERE TUZAĞI -- NATIVE ÇİFT, ATLAS DEĞİL
=======================================================================
`pipeline.segmentation.resolve_ready_mask()` LUMIERE için ZATEN native
DeepBraTumIA çiftini seçer (`CT1.nii.gz` + `.../native/segmentation/
ct1_seg_mask.nii.gz`, `_lumiere_mask()`, Hafta 2 kararı) -- bu modül bu
kuralı YENİDEN YAZMAZ, doğrudan çağırır. Radyomik de AYNI native
maskeden hesaplandı (bkz. `pipeline/segmentation.py` modül dokstring'i).

=======================================================================
EKSEN DÜZENİ -- `as_closest_canonical()` ÖNCE, `axis=2` SONRA
=======================================================================
LUMIERE `PIR`, TCGA/UCSF `LPS` ham eksen kodlu -- `nibabel.
as_closest_canonical()` HER İKİ hacmi de (görüntü + maske) RAS'a çevirir,
YALNIZ SONRASINDA eksen 2 "eksenel/S-I" eksenidir. Sabit `axis=2`
DOĞRUDAN ham veriye uygulanırsa LUMIERE'de koronal kesit üretilirdi --
bu modül HER ZAMAN önce kanonikleştirir.

=======================================================================
SimpleITK KULLANILMAZ (piksel okuma için) -- yalnız `nibabel`
=======================================================================
Görüntü/maske piksel verisi YALNIZ `nibabel` ile okunur (Türkçe-
karakterli yol riski: ITK'nin C++ `fopen()`'ı bu tür yolları
açamayabiliyor, bkz. `api/similar.py` modül dokstring'i "BİLİNEN
İŞLETİMSEL RİSK", `tests/test_no_resolve_path_regression.py`).
⚠️ `pipeline.segmentation.resolve_ready_mask()` kendi İÇİNDE
(`pipeline.resampling.validate_image_mask_geometry`) SimpleITK
kullanır -- bu modülün DIŞARIDAN doğrudan `import SimpleITK` YAPMAMASI
anlamındadır, alt katmanın kendi iç geometriksel ön-kontrolü hâlâ
SimpleITK'dır (demo/demo_helpers.py da AYNI şekilde `resolve_ready_mask`
çağırıyor, `raise_on_geometry_mismatch=False` ile). Bu modülün KENDİ
geometri kararı (render edilecek/edilmeyecek) HER ZAMAN nibabel'in
`affine`/`shape` karşılaştırmasına dayanır.

=======================================================================
UCSF -- `resolve_ready_mask()` KAPSAMI DIŞINDA, AYRI ÇÖZÜMLEME
=======================================================================
`pipeline.segmentation.resolve_ready_mask()` yalnız TCGA/UPenn/LUMIERE
için tanımlıdır (`canonical == "UCSF"` durumunda `ValueError` fırlatır).
UCSF için maske `<ID>_tumor_segmentation.nii.gz` adıyla T1c'nin YANINDA
aranır (`tools/run_pyradiomics_ucsf.py` sözleşmesi, demo'nun
`_resolve_ucsf_display_pair()`'iyle BİREBİR aynı desen). UCSF'in
`mr_scans.file_path` alanı bu makinede MUTLAK Windows yolu taşıyor
(`decisions/2026-09-15-ucsf-mr-goruntuleri-canlida-gosterilecek.md`) --
başka bir sunucuda dosya BULUNAMAZ, bu GÜRÜLTÜLÜ 422 ile bildirilir
(sessiz boş PNG ASLA dönmez).

=======================================================================
HATA SÖZLEŞMESİ -- mevcut `api/predict.py`/`api/main.py` felsefesiyle
HİZALI (görev talimatı: "mevcut hata sınıflandırmasına uy")
=======================================================================
  - Hasta `patients` tablosunda YOK                          -> 404
  - Hasta var ama T1ce taraması YOK                           -> 422
  - `scan_id` istendi ama o hastanın T1ce taramaları arasında
    YOK                                                       -> 422
  - Birden fazla T1ce taraması var, `scan_id` verilmedi
    (LUMIERE DIŞI kaynaklar)                                  -> 422
  - LUMIERE: C32 radyomiği hiç yok / kanonik Pre-Op vizit
    seçilemiyor                                                -> 422
  - LUMIERE ExpertRating CSV'si erişilemez (altyapı)            -> 503
  - NAS kökü tamamen erişilemez (`GBMAID_NAS_ROOT` dizini yok)  -> 503
  - `GBMAID_NAS_ROOT` hiç tanımlı değil (konfigürasyon)         -> 500
  - Belirli bir dosya (görüntü/maske) NAS'ta erişilebilir kökte
    bulunamadı (main.py'nin `harmonize` ile AYNI kuralı:
    main.py:722-733 "bozuk veya bulunamayan dosya")            -> 422
  - Görüntü/maske geometrisi (shape/affine) uyuşmuyor           -> 422
  - Hazır maske hiç tümör vokseli içermiyor                     -> 422
  - `z` kesit indeksi sınır dışı                                -> 422
  - Tanımsız/bilinmeyen kaynak (config/whitelist tutarsızlığı)  -> 500
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import psycopg2.extras
from fastapi import APIRouter, HTTPException, Query, Response

from db_connection import get_connection
from pipeline.harmonization import _nifti_stem, canonical_source
from pipeline.lumiere_canonical_visit import is_lumiere_patient_id
from pipeline.radiomics_volume import REGION_LABELS_BY_MASK_SOURCE
from pipeline.segmentation import (
    ImageMaskGeometryError,
    ReadyMaskNotFoundError,
    resolve_nas_path,
    resolve_ready_mask,
)

# 🔒 TEK KAYNAK -- K2 kilitli kanonik-vizit seçimi `api/predict.py`'de
# ZATEN implemente edilmiş durumda (`fetch_lumiere_canonical_visit_
# selection`, Y1 görevi). Burada YENİDEN YAZILMAZ -- `/predict`in risk
# skorunu ürettiği TAM AYNI LUMIERE taraması gösterilsin diye DOĞRUDAN
# import edilir. `api/main.py` zaten `api.predict`i (satır 22)
# `api.mr_slice`den önce yüklüyor olacak (router kayıt sırası), yani bu
# import sürece EK ağırlık GETİRMEZ (bkz. `api/similar.py`'nin AYNI
# gerekçesi).
from api.predict import (
    C32RadiomicsNotFoundError,
    LumiereCanonicalVisitDataError,
    fetch_lumiere_canonical_visit_selection,
)
from pipeline.lumiere_canonical_visit import (
    ExpertRatingFileNotFoundError,
    LumierePreopVisitNotAvailableError,
)

router = APIRouter()

REQUIRED_MODALITY = "T1ce"

# `pipeline.segmentation.nas_root()`'un NAS kökü tamamen erişilemez
# olduğunda fırlattığı mesajın ayırt edici alt dizesi -- `api/main.py`
# ile AYNI desen (bkz. main.py satır 214-221).
_NAS_UNAVAILABLE_MARKER = "erişilebilir değil"

# Bölge adı -> RGB renk (dataviz koyu-tema paleti). `demo/demo_helpers.
# py::MR_REGION_COLORS` ile AYNI renk seçimleri -- sunum tercihi, "TEK
# KAYNAK" gerektiren bir iş kuralı DEĞİL, bu yüzden bağımsız kopyalanması
# kabul edilebilir (drift riski yalnız görsel tutarlılığı etkiler,
# hiçbir sayısal/klinik sonucu etkilemez).
REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "Necrosis": (230, 103, 103),
    "NC": (230, 103, 103),
    "Contrast-enhancing": (245, 199, 68),
    "ET": (245, 199, 68),
    "Edema": (57, 135, 229),
    "ED": (57, 135, 229),
    "Non-enhancing": (156, 122, 214),
    "WT": (230, 103, 103),
    "TC": (245, 199, 68),
}
_FALLBACK_COLOR = (200, 200, 200)


def _get_db_connection():
    return get_connection(readonly=True)


class MultiScanAmbiguousError(RuntimeError):
    """LUMIERE-dışı bir hastanın birden fazla T1ce taraması var, `scan_id`
    ile disambiguation YAPILMADI -- main.py/predict.py'nin "sessizce karar
    verme" karşıtı felsefesiyle hizalı."""


class ScanIdNotFoundError(RuntimeError):
    """İstenen `scan_id` bu hastanın T1ce taramaları arasında yok."""


class NoT1ceScanError(RuntimeError):
    """Hasta var ama hiç T1ce taraması yok."""


class UnsupportedMaskSourceError(RuntimeError):
    """UCSF T1c dosya adı beklenen `_T1c` desenine uymuyor (config/veri
    tutarsızlığı, whitelist'e girmiş olmaması gerekirdi)."""


# =====================================================================
# 1) Hastanın T1ce taramaları + kanonik/istenen tarama seçimi
# =====================================================================


def _fetch_patient_exists(conn, patient_id: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM patients WHERE patient_id = %s", (patient_id,))
    found = cur.fetchone() is not None
    cur.close()
    return found


def _fetch_t1ce_scans(conn, patient_id: str) -> list[dict[str, Any]]:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT ms.scan_id, ms.file_path, ds.source_name AS source
        FROM mr_scans ms
        JOIN patients p ON p.patient_id = ms.patient_id
        JOIN dataset_sources ds ON ds.source_id = p.source_id
        WHERE ms.patient_id = %s AND ms.modality = %s
        ORDER BY ms.scan_id
        """,
        (patient_id, REQUIRED_MODALITY),
    )
    rows = cur.fetchall()
    cur.close()
    return list(rows)


def _select_scan(
    conn, patient_id: str, requested_scan_id: int | None
) -> dict[str, Any]:
    """Gösterilecek taramayı seçer. Dönen sözlük: `scan_id`, `file_path`,
    `source`, ve (LUMIERE ise) `lumiere_canonical_visit` beyan bloğu."""

    scans = _fetch_t1ce_scans(conn, patient_id)
    if not scans:
        raise NoT1ceScanError(
            f"Hasta {patient_id!r} için `mr_scans` tablosunda T1ce satırı "
            "YOK -- bu hastanın MR görüntüsü sistemde kayıtlı değil."
        )
    by_scan_id = {int(row["scan_id"]): row for row in scans}

    if is_lumiere_patient_id(patient_id):
        # K2 kilitli kural -- risk skorunun ürettiği TAM AYNI tarama
        # gösterilsin diye `requested_scan_id` YOK SAYILMAZ ama kanonik
        # olanla UYUŞMUYORSA GÜRÜLTÜLÜ reddedilir (LUMIERE'de "hangi
        # tarama gösterilsin" kararı hastaya/isteğe değil K2 kuralına
        # aittir -- görev talimatının "LUMIERE tuzağı" uyarısı BUNU
        # gerekçelendiriyor: post-op görüntü yapısal olarak farklıdır).
        selection = fetch_lumiere_canonical_visit_selection(patient_id)
        canonical_scan_id = int(selection.scan_id)
        if requested_scan_id is not None and requested_scan_id != canonical_scan_id:
            raise ScanIdNotFoundError(
                f"LUMIERE hastası {patient_id!r} için yalnız K2 kanonik "
                f"Pre-Op vizit ({canonical_scan_id}) gösterilebilir -- "
                f"istenen scan_id={requested_scan_id} REDDEDİLDİ (sessizce "
                "değiştirilmedi). Post-op görüntü radyomik/risk skorunun "
                "kullandığı görüntüyle AYNI değildir."
            )
        if canonical_scan_id not in by_scan_id:
            raise NoT1ceScanError(
                f"Veri tutarsızlığı: LUMIERE kanonik scan_id={canonical_scan_id} "
                "radyomik whitelist'inde var ama `mr_scans`'te T1ce olarak "
                "bulunamadı -- db-agent'e bildirin."
            )
        row = by_scan_id[canonical_scan_id]
        return {
            "scan_id": canonical_scan_id,
            "file_path": row["file_path"],
            "source": row["source"],
            "lumiere_canonical_visit": {
                "applied": True,
                "rule": "K2: Rating == 'Pre-Op' (LUMIERE ExpertRating)",
                "timepoint_label": selection.timepoint_label,
                "scan_id": canonical_scan_id,
            },
        }

    if requested_scan_id is not None:
        if requested_scan_id not in by_scan_id:
            raise ScanIdNotFoundError(
                f"scan_id={requested_scan_id} hasta {patient_id!r}'in T1ce "
                f"taramaları arasında yok. Mevcut: {sorted(by_scan_id)}."
            )
        row = by_scan_id[requested_scan_id]
        return {"scan_id": requested_scan_id, "file_path": row["file_path"], "source": row["source"]}

    if len(scans) > 1:
        raise MultiScanAmbiguousError(
            f"Hasta {patient_id!r}: {len(scans)} farklı T1ce scan_id var "
            f"({sorted(by_scan_id)}) -- hangisinin gösterileceğine SESSİZCE "
            "karar VERİLMEDİ. `scan_id` query parametresiyle belirtin."
        )

    only = scans[0]
    return {"scan_id": int(only["scan_id"]), "file_path": only["file_path"], "source": only["source"]}


# =====================================================================
# 2) Görüntü/maske çifti çözümleme (kaynağa göre)
# =====================================================================


def _resolve_ucsf_mask(image_path: Path) -> tuple[Path, str]:
    """`tools/run_pyradiomics_ucsf.py` sözleşmesinin BİREBİR karşılığı --
    `resolve_ready_mask()` UCSF'i desteklemiyor (yalnız TCGA/UPenn/LUMIERE),
    bu yüzden UCSF için AYRI, basit bir dosya-yanı arama yapılır."""

    stem = _nifti_stem(image_path)
    if not stem.endswith("_T1c"):
        raise UnsupportedMaskSourceError(
            f"UCSF T1c dosya adı beklenen desende değil ('*_T1c'): {image_path}"
        )
    base = stem[: -len("_T1c")]
    for suffix in (".nii.gz", ".nii"):
        candidate = image_path.parent / f"{base}_tumor_segmentation{suffix}"
        if candidate.is_file():
            return candidate, "ucsf_native"
    raise FileNotFoundError(
        "UCSF hazır tümör maskesi bulunamadı: "
        f"{image_path.parent / (base + '_tumor_segmentation.nii.gz')}"
    )


def _resolve_display_pair(
    *, source_name: str, file_path: str, patient_id: str, scan_id: int
) -> dict[str, Any]:
    canonical = canonical_source(source_name)
    image_path = resolve_nas_path(file_path)

    if canonical == "UCSF":
        mask_path, mask_source = _resolve_ucsf_mask(image_path)
        return {
            "image_path": image_path,
            "mask_path": mask_path,
            "mask_source": mask_source,
            "source": canonical,
            "warnings": [],
        }

    info = resolve_ready_mask(
        source=source_name,
        image_path=image_path,
        patient_id=patient_id,
        scan_id=scan_id,
        # Görüntüleme-seviyesi geometri kontrolü AŞAĞIDA nibabel ile
        # YAPILIR (bkz. modül dokstring'i) -- burada SimpleITK'nın kendi
        # ön-kontrolüne FATAL olarak GÜVENİLMEZ.
        raise_on_geometry_mismatch=False,
    )
    return {
        "image_path": Path(str(info["image_path"])),
        "mask_path": Path(str(info["mask_path"])),
        "mask_source": str(info["mask_source"]),
        "source": canonical,
        "warnings": list(info["warnings"]),
    }


# =====================================================================
# 3) nibabel ile hacim yükleme + kesit/overlay üretimi
# =====================================================================


def _load_overlay_volume(pair: dict[str, Any]) -> dict[str, Any]:
    import nibabel as nib

    raw_img = nib.load(str(pair["image_path"]))
    raw_seg = nib.load(str(pair["mask_path"]))
    can_img = nib.as_closest_canonical(raw_img)
    can_seg = nib.as_closest_canonical(raw_seg)
    arr = np.asanyarray(can_img.dataobj, dtype=np.float32)
    seg = np.asanyarray(can_seg.dataobj)

    if arr.shape != seg.shape or not np.allclose(can_img.affine, can_seg.affine, atol=1e-4):
        raise ImageMaskGeometryError(
            f"Görüntü ve maske AYNI uzayda değil -- görüntü {arr.shape} "
            f"{pair['image_path']}, maske {seg.shape} {pair['mask_path']}. "
            "Overlay ÜRETİLMEZ (yanlış hizalanmış bir kesit tümörü yanlış "
            "yerde gösterirdi)."
        )

    seg = seg.astype(np.int16, copy=False)
    tumor = seg > 0
    slice_counts = tumor.sum(axis=(0, 1)).astype(int)
    tumor_voxels = int(tumor.sum())
    if tumor_voxels == 0:
        raise ValueError(
            f"Hazır maske ({pair['mask_source']}) bu taramada HİÇ tümör "
            "vokseli içermiyor -- gösterilecek lezyon yok."
        )
    best_slice = int(np.argmax(slice_counts))

    brain = arr[arr > 0]
    if brain.size:
        lo, hi = float(np.percentile(brain, 1)), float(np.percentile(brain, 99))
    else:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        lo, hi = float(arr.min()), float(max(arr.max(), arr.min() + 1.0))
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    img_u8 = (scaled * 255.0).round().astype(np.uint8)

    region_labels = dict(REGION_LABELS_BY_MASK_SOURCE.get(pair["mask_source"], {}))
    present = {int(v) for v in np.unique(seg) if int(v) != 0}
    region_labels = {k: v for k, v in region_labels.items() if int(v) in present}

    return {
        "img_u8": img_u8,
        "seg_u8": seg.astype(np.uint8, copy=False),
        "n_slices": int(arr.shape[2]),
        "best_slice": best_slice,
        "region_labels": region_labels,
        "tumor_voxels": tumor_voxels,
    }


def _axial_slice(volume_array: np.ndarray, slice_index: int) -> np.ndarray:
    """RAS hacminden eksenel kesit -- RADYOLOJİK yön (sol = hastanın SAĞI),
    `demo/demo_helpers.py::_mr_axial_slice()` ile AYNI dönüşüm."""

    plane = volume_array[:, :, int(slice_index)]
    return np.ascontiguousarray(np.rot90(plane)[:, ::-1])


def _render_png(volume: dict[str, Any], slice_index: int, *, with_overlay: bool, alpha: float = 0.45) -> bytes:
    from PIL import Image

    gray = _axial_slice(volume["img_u8"], slice_index)
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)

    if with_overlay:
        seg_plane = _axial_slice(volume["seg_u8"], slice_index)
        for region_name, label_value in volume["region_labels"].items():
            region_mask = seg_plane == int(label_value)
            if not region_mask.any():
                continue
            color = np.asarray(REGION_COLORS.get(region_name, _FALLBACK_COLOR), dtype=np.float32)
            rgb[region_mask] = (1.0 - alpha) * rgb[region_mask] + alpha * color

    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


# =====================================================================
# 4) Endpoint
# =====================================================================


@router.get("/patient/{patient_id}/mr_slice")
def get_mr_slice(
    patient_id: str,
    z: int | None = Query(
        None, ge=0, description="Eksenel kesit indeksi -- verilmezse EN GENİŞ tümör kesiti kullanılır."
    ),
    overlay: bool = Query(True, description="True ise hazır maske renkli katman olarak bindirilir."),
    scan_id: int | None = Query(
        None,
        description=(
            "Hastanın birden fazla T1ce taraması varsa hangisinin "
            "gösterileceğini belirtir. LUMIERE'de YOK SAYILMAZ ama K2 "
            "kanonik vizitle UYUŞMUYORSA 422 döner (bkz. modül dokstring'i)."
        ),
    ),
):
    try:
        conn = _get_db_connection()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"DB'ye erişilemedi: {exc}")

    try:
        if not _fetch_patient_exists(conn, patient_id):
            raise HTTPException(status_code=404, detail=f"Hasta bulunamadı: {patient_id}")

        try:
            selection = _select_scan(conn, patient_id, scan_id)
        except NoT1ceScanError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except MultiScanAmbiguousError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except ScanIdNotFoundError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except C32RadiomicsNotFoundError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except LumierePreopVisitNotAvailableError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except LumiereCanonicalVisitDataError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except ExpertRatingFileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
    finally:
        conn.close()

    try:
        pair = _resolve_display_pair(
            source_name=selection["source"],
            file_path=selection["file_path"],
            patient_id=patient_id,
            scan_id=selection["scan_id"],
        )
    except KeyError as exc:
        # `pipeline.segmentation.nas_root()` -- `GBMAID_NAS_ROOT` tanımsız.
        raise HTTPException(status_code=500, detail=f"Sunucu konfigürasyon hatası: {exc}")
    except FileNotFoundError as exc:
        if _NAS_UNAVAILABLE_MARKER in str(exc):
            raise HTTPException(status_code=503, detail=f"NAS erişilemez: {exc}")
        raise HTTPException(
            status_code=422, detail=f"Bozuk veya bulunamayan dosya: {exc}"
        )
    except ReadyMaskNotFoundError as exc:
        raise HTTPException(status_code=422, detail=f"Hazır maske bulunamadı: {exc}")
    except UnsupportedMaskSourceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except ValueError as exc:
        # `canonical_source()`/`resolve_ready_mask()`'ın tanımadığı kaynak
        # -- whitelist/config tutarsızlığı, hastanın hatası DEĞİL.
        raise HTTPException(status_code=500, detail=f"Sunucu konfigürasyon hatası: {exc}")
    except Exception as exc:  # noqa: BLE001 -- bkz. modül dokstring'i "SimpleITK KULLANILMAZ"
        # `resolve_ready_mask()` KENDİ İÇİNDE (`validate_image_mask_
        # geometry`) SimpleITK kullanır -- Türkçe-karakterli bir yolda
        # `sitk.ReadImage()` sınıflandırılmamış bir RuntimeError/OSError
        # fırlatabilir (görev talimatının açıkça uyardığı risk). Bu
        # backstop, o durumda İSTEMCİYE sınıflandırılmamış bir 500 yerine
        # AÇIKÇA "geçici erişilemezlik" diyen bir 503 döner -- sessiz
        # YUTULMAZ, orijinal hata mesajı korunur.
        raise HTTPException(
            status_code=503,
            detail=(
                "MR görüntüsü/maskesi şu an erişilemiyor (alt katmanın "
                f"geometri doğrulaması başarısız oldu: {type(exc).__name__}: {exc})"
            ),
        )

    try:
        volume = _load_overlay_volume(pair)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=f"Bozuk veya bulunamayan dosya: {exc}")
    except ImageMaskGeometryError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        # "hiç tümör vokseli yok" -- bkz. `_load_overlay_volume`.
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 -- nibabel okuma/ağ hatası
        raise HTTPException(
            status_code=503,
            detail=(
                "MR görüntüsü şu an erişilemiyor: NAS bağlantısı yok veya "
                f"dosya okunamadı ({type(exc).__name__}: {exc})"
            ),
        )

    slice_index = z if z is not None else volume["best_slice"]
    if slice_index >= volume["n_slices"]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"z={slice_index} sınır dışı -- bu taramada toplam "
                f"{volume['n_slices']} eksenel kesit var (0..{volume['n_slices'] - 1})."
            ),
        )

    png_bytes = _render_png(volume, slice_index, with_overlay=overlay)

    headers = {
        "X-GBMAID-Source": pair["source"],
        "X-GBMAID-Scan-Id": str(selection["scan_id"]),
        "X-GBMAID-Slice-Index": str(slice_index),
        "X-GBMAID-Best-Slice": str(volume["best_slice"]),
        "X-GBMAID-N-Slices": str(volume["n_slices"]),
        "X-GBMAID-Mask-Source": pair["mask_source"],
        "X-GBMAID-Overlay-Applied": "true" if overlay else "false",
    }
    if "lumiere_canonical_visit" in selection:
        headers["X-GBMAID-Lumiere-Timepoint"] = str(
            selection["lumiere_canonical_visit"]["timepoint_label"]
        )
    return Response(content=png_bytes, media_type="image/png", headers=headers)
