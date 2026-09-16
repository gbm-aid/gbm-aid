"""UCSF-PDGM hazir maske cozumleme testleri.

🔴 AYRI DOSYADA olmasinin sebebi OLCULDU: `tests/test_segmentation.py` modul
basinda `import SimpleITK as sitk` yapar; Windows Smart App Control bu makinede
SimpleITK'nin native uzantisini engelledigi icin O DOSYA TOPLANAMIYOR. UCSF
maske kurali SimpleITK'ya ihtiyac duymaz (yalniz dosya adi eslemesi), bu yuzden
testleri ayri tutmak onlari bu makinede de KOSULABILIR kilar.

Karar: decisions/2026-09-15-ucsf-mr-goruntuleri-canlida-gosterilecek.md Adim 1
"""

from __future__ import annotations

from pathlib import Path

import pytest

def _ucsf_ciftini_kur(tmp_path, *, maske_uzantisi=".nii.gz", maske_yaz=True):
    """`<ad>_T1c` + `<ad>_tumor_segmentation` ciftini diske kurar."""
    hasta = tmp_path / "UCSF-PDGM-0115_nifti"
    hasta.mkdir()
    goruntu = hasta / "UCSF-PDGM-0115_T1c.nii.gz"
    goruntu.write_bytes(b"")
    if maske_yaz:
        (hasta / f"UCSF-PDGM-0115_tumor_segmentation{maske_uzantisi}").write_bytes(b"")
    return goruntu


def test_ucsf_mask_t1c_den_tumor_segmentation_bulur(tmp_path):
    from pipeline.segmentation import _ucsf_mask

    goruntu = _ucsf_ciftini_kur(tmp_path)
    maske, kaynak, uyarilar = _ucsf_mask(goruntu)

    assert maske.name == "UCSF-PDGM-0115_tumor_segmentation.nii.gz"
    assert kaynak == "ucsf_native"
    assert uyarilar == []


def test_ucsf_mask_sikistirilmamis_nii_de_kabul_eder(tmp_path):
    from pipeline.segmentation import _ucsf_mask

    goruntu = _ucsf_ciftini_kur(tmp_path, maske_uzantisi=".nii")
    maske, _kaynak, _uyarilar = _ucsf_mask(goruntu)
    assert maske.suffix == ".nii"


def test_ucsf_mask_maske_yoksa_SESSIZCE_baska_seye_dusmez(tmp_path):
    """UCSF'te TCGA'daki whole/core veya UPenn'deki automated_approx geri
    dusus TANIMLI DEGILDIR -- maske yoksa hata yukselir."""
    from pipeline.segmentation import ReadyMaskNotFoundError, _ucsf_mask

    goruntu = _ucsf_ciftini_kur(tmp_path, maske_yaz=False)
    with pytest.raises(ReadyMaskNotFoundError, match="tümör maskesi bulunamadı"):
        _ucsf_mask(goruntu)


def test_ucsf_mask_T1c_olmayan_dosya_adini_reddeder(tmp_path):
    from pipeline.segmentation import ReadyMaskNotFoundError, _ucsf_mask

    hasta = tmp_path / "UCSF-PDGM-0115_nifti"
    hasta.mkdir()
    yanlis = hasta / "UCSF-PDGM-0115_FLAIR.nii.gz"
    yanlis.write_bytes(b"")
    with pytest.raises(ReadyMaskNotFoundError, match=r"_T1c"):
        _ucsf_mask(yanlis)


def test_resolve_ready_mask_UCSF_kolunu_tanir(monkeypatch, tmp_path):
    """2026-09-16 oncesinde bu dal YOKTU ve UCSF 'tanimsiz kaynak' diye
    ValueError ile reddediliyordu."""
    from pipeline import segmentation as seg

    goruntu = _ucsf_ciftini_kur(tmp_path)
    monkeypatch.setattr(
        seg, "validate_image_mask_geometry", lambda *a, **k: {"geometry_match": True}
    )
    sonuc = seg.resolve_ready_mask(
        source="UCSF-PDGM", image_path=goruntu, patient_id="UCSF-PDGM-0115", scan_id=1
    )
    assert sonuc["mask_source"] == "ucsf_native"


def test_demo_sarmalayicisi_URETIMDEKI_fonksiyonu_cagirir(tmp_path, monkeypatch):
    """Kural IKI YERDE TUTULMAZ: demo, uretimdeki `_ucsf_mask`'i cagirmali.
    Uretimdeki fonksiyon degistirilirse demo'nun ciktisi da DEGISMELI."""
    import sys

    from pipeline import segmentation as seg

    kok = Path(seg.__file__).resolve().parents[2]
    if str(kok) not in sys.path:
        sys.path.insert(0, str(kok))
    from demo.demo_helpers import _resolve_ucsf_display_pair

    goruntu = _ucsf_ciftini_kur(tmp_path)
    isaret = (Path(tmp_path) / "SAHTE_MASKE.nii.gz", "sahte_kaynak", [])
    monkeypatch.setattr(seg, "_ucsf_mask", lambda p: isaret)

    maske, kaynak = _resolve_ucsf_display_pair(goruntu)
    assert kaynak == "sahte_kaynak", "demo hâlâ KENDI kopyasini kullaniyor"
    assert maske.name == "SAHTE_MASKE.nii.gz"
