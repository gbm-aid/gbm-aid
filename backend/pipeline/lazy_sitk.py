"""SimpleITK için TEMBEL (lazy) import vekili — B1'in ikinci turu (2026-09-16).

**Neden var:** Windows Smart App Control, imzasız
`_SimpleITK.cp310-win_amd64.pyd`'yi yüklerken engelliyor (CodeIntegrity
Event 3077/3118, Policy ID `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`).
`import SimpleITK` MODÜL BAŞINDA yapıldığı için, SimpleITK'ya hiç ihtiyaç
duymayan kod yolları da (site, `/patients`, `/predict`, `/mr_slice`,
`/model_curves` …) import zincirinde çöküyordu:

    api/main.py:14 -> pipeline.harmonization:26 -> import SimpleITK -> ImportError

**Bu, 2026-09-14'teki B1 ile AYNI hata sınıfıdır** (`BEKLEYEN-KARARLAR.md`
K21): ağır bir bağımlılığın arkasında sıkışan hafif kod yolu. B1 saf
string yardımcılarını `pipeline/source_canonical.py`'ye taşıyarak
`cox_model`/`predict`/`similar` zincirini kurtarmıştı; bu modül aynı işi
kalan dört görüntü modülü için, **sembol taşımadan** yapar.

**Nasıl çalışır:** `sitk` artık bir modül değil, modüle vekillik eden tek
bir nesnedir. `sitk.ReadImage(...)` ilk kez çağrıldığında SimpleITK
GERÇEKTEN import edilir ve sonuç önbelleğe alınır. Yani:

  * SimpleITK'ya dokunmayan bir kod yolu onu HİÇ import etmez.
  * SimpleITK'ya dokunan bir kod yolu, tam olarak eskisi gibi çalışır.
  * SimpleITK gerçekten gerekliyse ve bloklanmışsa, hata **çağrı anında**
    ve ORİJİNAL `ImportError` metniyle yükselir -- yutulmaz, sessiz
    fallback YOKTUR (`assert_no_identifiers_leaked` desenindeki
    fail-loud kuralıyla aynı çizgide).

🔴 **BU BİR DÜZELTME DEĞİL, ERTELEMEDİR.** SimpleITK'yı GERÇEKTEN kullanan
yollar bu makinede SAC bloğu sürdükçe ÇALIŞMAZ: `POST /patient/{id}/harmonize`
(N4 + Z-score), PyRadiomics çıkarım araçları, `tools/rebuild_faiss_indexes.py`
(ComBat import'u -- K21 ile ayrıştırılmama kararı var). Kalıcı çözüm Linux
ortamıdır (SAC Windows'a özgüdür).

⚠️ Çağrı yerleri DEĞİŞMEDİ: dört modüldeki 61 `sitk.` kullanımının hiçbirine
dokunulmadı, yalnız `import SimpleITK as sitk` satırları bu modülden
`from pipeline.lazy_sitk import sitk` ile değiştirildi.

📄 `BEKLEYEN-KARARLAR.md` K21 · `log/2026-09-14.md` (11:00 SAC bulgusu,
12:05 B1) · `log/2026-09-16.md` · `pipeline/source_canonical.py` (B1)
"""

from __future__ import annotations

from typing import Any

# Gerçek modül ilk erişimde buraya yerleşir; ikinci erişimde import
# tekrarlanmaz. (`sys.modules` zaten önbellekler, bu yalnız attribute
# arama maliyetini de kaldırır.)
_real_module: Any = None


class _LazySimpleITK:
    """`SimpleITK` modülüne vekillik eden tek nesne.

    `__slots__ = ()` bilinçlidir: örnek sözlüğü olmadığı için HER
    attribute erişimi `__getattr__`'a düşer, yani vekil hiçbir adı
    sessizce gölgeleyemez.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        global _real_module
        if _real_module is None:
            import SimpleITK as _imported  # noqa: PLC0415 -- tembel: modülün tek amacı bu

            _real_module = _imported
        return getattr(_real_module, name)

    def __repr__(self) -> str:  # pragma: no cover -- yalnız hata ayıklama kolaylığı
        state = "yüklenmedi" if _real_module is None else "yüklendi"
        return f"<tembel SimpleITK vekili ({state})>"


sitk = _LazySimpleITK()


def is_loaded() -> bool:
    """SimpleITK'nın GERÇEKTEN import edilip edilmediğini söyler.

    Test/teşhis içindir; üretim kod yolları bunu çağırmaz.
    """
    return _real_module is not None
