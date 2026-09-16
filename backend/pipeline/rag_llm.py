"""LLM çağrı arayüzü — GERÇEK sağlayıcı çağrısı YAPILIR (2026-09-14'ten beri).

🔴 **GÜNCEL CÜMLE ÜSTTE (2026-09-16, koordinatör ölçümü; Barış talimatı).**
Aşağıdaki *"Hafta 5 — İSKELET, gerçek çağrı YAPILMAZ"* açılışı ve
*"anahtar TANIMLI DEĞİL"* paragrafı **BAYATTIR, esas alınmaz.** Eski metin
SİLİNMEDİ (hard rule #3), yalnız işaretlendi.

**ÖLÇÜM (2026-09-16, canlı):**
  * `_invoke_llm_provider` (bu dosyada) `pipeline/llm_provider.py`
    üzerinden GERÇEK bir sağlayıcı çağrısı yapar. Koşul:
    `GBMAID_LLM_ENABLED` açık VE yapılandırma tam.
  * Proje kökündeki `.env`'de `GBMAID_LLM_PROVIDER`, `GBMAID_LLM_MODEL`,
    `OPENAI_API_KEY`, `GBMAID_LLM_ENABLED` dahil 7 LLM değişkeni
    TANIMLIDIR -- yalnız değişken ADLARI okundu, DEĞER OKUNMADI/YAZILMADI.
  * Canlı kanıt: 2026-09-15 uçtan uca doğrulamada 4 hastada model
    `gpt-5.4-2026-03-05`, `citation_grounding_violation=False`;
    2026-09-16'da `POST /analyze_patient` 200, literatür 7 kaynak.
  * **VARSAYILAN HÂLÂ KAPALIDIR:** `GBMAID_LLM_ENABLED` yoksa davranış
    2026-09-13'ün BİREBİR AYNISI (`LLMNotConfiguredError`).

⚠️ **NEDEN İŞARETLENDİ:** bayat açılış 2026-09-16'da iki ayrı okuyucuyu
*"RAG canlı değil, LLM sağlayıcı kararı verilmedi"* sonucuna götürdü. Bu,
`ZORUNLU-BEYANLAR.md`'de C8 / A8 / D4-e'de yakalanan hata sınıfının
aynısıdır: yaptığımız bir işi "yapmadık" diye beyan etmek.
📄 `ZORUNLU-BEYANLAR.md` D3 · `log/2026-09-15.md` · `log/2026-09-16.md`

⬇️ **AŞAĞISI TARİHSEL (2026-09-13 durumu) -- esas alınmaz:**

    Hafta 5 — LLM çağrı arayüzü (İSKELET, gerçek çağrı YAPILMAZ).

    Görev talimatı: "Çağrı arayüzünü yaz (sanitize edilmiş girdi ->
    PMID'li özet), ama gerçek çağrı YAPMA." Bu dosyadaki HİÇBİR fonksiyon
    dış ağa istek ATMAZ.

    Anahtar durumu (bu görevde ÖLÇÜLDÜ, kimlik bilgisi YAZILMADI): proje
    kökündeki `.env`'de `OPENAI_API_KEY` (veya benzeri bir LLM sağlayıcı
    anahtarı) TANIMLI DEĞİL -- yalnız değişken adı kontrol edildi, değer
    YAZILMADI/OKUNMADI. CLAUDE.md "yeni bulut hesabı/servis AÇMA" kısıtı +
    anahtar YOKLUĞU birlikte bu görevde gerçek bir LLM çağrısını zaten
    İMKANSIZ kılıyor -- bu Barış'a AÇIK bir soru: hangi LLM sağlayıcısı/
    anahtarı kullanılacak (v45.txt Bölüm 8.3: "OpenAI API / Mistral /
    LLaMA, prototip için API tabanlı; üretim için lokal açık kaynak
    modele geçiş planlanmaktadır").

⬆️ **TARİHSEL BÖLÜM BİTTİ -- aşağısı GÜNCELDİR.**

Yapısal zorunluluk (görev talimatı + `pipeline/rag_sanitize.py`
dokstring'i): bu modüldeki TEK LLM-çağrı fonksiyonu (`call_llm_summarize`)
parametre tipi olarak SADECE `pipeline.rag_sanitize.SanitizedLLMContext`
kabul eder -- ham string/dict alan bir "kestirme" fonksiyon BİLİNÇLİ
OLARAK YAZILMADI. Bu, "sanitizasyon katmanını atlayan bir kod yolu
olmasın" gereksinimini Python tip sistemiyle yapısal olarak destekler
(çağıran taraf `SanitizedLLMContext` inşa etmeden bu fonksiyonu
çağıramaz -- inşa etmenin TEK yolu `rag_sanitize.build_sanitized_context`
'tir).

⚠️ 2026-09-13 (rag-agent-C, Barış onayı / karar 6) — PMID TEMELLENDİRME
(grounding) KAPISI EKLENDİ.

Sorun: dil modeli, bizim GETİRMEDİĞİMİZ bir PMID uydurabilir
(halüsinasyon). `LLMSummaryResult.cited_pmids` doğrudan kullanıcıya/
jüriye gösterilecek; jüri *"bu kaynağı nereden buldunuz"* diye
sorduğunda cevabımız olmalı. Bu, SAĞLAYICIDAN BAĞIMSIZ bir doğruluk
şartıdır -- hangi modeli seçersek seçelim gerekir.

Çözümün yapısı (`assert_no_identifiers_leaked` desenine BİREBİR uyumlu:
fail-loud, SESSİZ DÜZELTME YOK):
  * Gerçek sağlayıcı çağrısı artık PRIVATE bir fonksiyondadır
    (`_invoke_llm_provider`). Dışarıya açık TEK yol `call_llm_summarize`
    ve o yol guard'ı KOŞULSUZ uygular -- guard'ı atlayan bir
    konfigürasyon bayrağı, parametre veya "skip" yolu YOKTUR
    (bilinçli: `assert_citations_are_grounded` çağrısı `if`siz, tek
    satırdır ve public fonksiyonun `return`'ünden ÖNCEdir).
  * Küme dışı bir PMID -> `FabricatedPmidError`; özet REDDEDİLİR,
    sessizce kırpılmaz, sessizce kabul edilmez.
  * Hiç PMID alıntılanmamışsa -> `UncitedSummaryError` (gerekçe
    `assert_citations_are_grounded` dokstring'inde).

⚠️ DALGA ETKİSİ / AÇIK İŞ (bu görevin kapsamı DIŞINDA, kasıtlı
dokunulmadı): `pipeline/rag_pipeline.py::generate_literature_summary`
şu an `call_llm_summarize` çevresinde YALNIZ `LLMNotConfiguredError`
yakalıyor. Gerçek sağlayıcı entegrasyonu yapıldığı GÜN o `except`
bloğuna `CitationGroundingError` de EKLENMELİDİR, aksi halde bir
halüsinasyon pipeline'ı ÇÖKERTİR (graceful-degradation sözleşmesi
bozulur). Bugün bu bir sorun DEĞİL, çünkü `_invoke_llm_provider` her
zaman `LLMNotConfiguredError` fırlatıyor ve guard'a hiç sıra gelmiyor.
✅ **2026-09-14 (rag-agent-P5): YUKARIDAKİ AÇIK İŞ KAPANDI.** Hem gerçek
sağlayıcı çağrısı yazıldı (`pipeline/llm_provider.py`) hem de
`rag_pipeline` tarafındaki `CitationGroundingError` yakalaması eklendi.
Yukarıdaki paragraf TARİHSEL kayıt olarak korunuyor.

⚠️ 2026-09-14 (rag-agent-P5) — GERÇEK SAĞLAYICI ÇAĞRISI EKLENDİ.

`_invoke_llm_provider` artık koşullu: `GBMAID_LLM_ENABLED` açık ve
yapılandırma tamsa `pipeline/llm_provider.py` üzerinden GERÇEK bir çağrı
yapar; aksi halde **2026-09-13'teki davranışın BİREBİR AYNISI** olan
`LLMNotConfiguredError`'ı fırlatır. Varsayılan KAPALI'dır.

İki yapısal kapı (`assert_no_identifiers_leaked` + `assert_citations_
are_grounded`) DEĞİŞMEDİ, KOŞULSUZ ve ATLANAMAZ kaldı -- sağlayıcı
eklemek onları zayıflatmadı, aralarına girdi.

G3 (Barış kararı, 2026-09-14): model **TÜRKÇE** özet üretir; kaynakların
**orijinal (İngilizce) abstract'ları** `pipeline/rag_pipeline.py`
tarafından ayrıca taşınır -- jüri ikisini yan yana karşılaştırabilsin.
⚠️ Ayrım önemli: **LLM'e giden metin HER ZAMAN sanitize edilmiş
parçalardır**; orijinal abstract YALNIZ kendi kullanıcımıza/jüriye
dönen yanıtta bulunur, hiçbir zaman prompt'a girmez.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from pipeline.llm_provider import (
    LLMConfigError,
    LLMDisabledError,
    LLMProviderError,  # BİLİNÇLİ yeniden dışa aktarım: `call_llm_summarize`
    invoke_provider,   # çağıranları hata tiplerini TEK modülden alabilsin.
    load_llm_config,
)

__all__ = [
    "CitationGroundingError",
    "FabricatedPmidError",
    "LLMNotConfiguredError",
    "LLMProviderError",
    "LLMResponseFormatError",
    "LLMSummaryResult",
    "UncitedSummaryError",
    "allowed_pmids",
    "assert_citations_are_grounded",
    "build_llm_prompt",
    "build_llm_system_prompt",
    "call_llm_summarize",
    "extract_cited_pmids",
    "parse_llm_json_payload",
]
from pipeline.rag_sanitize import SanitizedLLMContext, assert_no_identifiers_leaked


class LLMNotConfiguredError(RuntimeError):
    """Gerçek bir LLM sağlayıcı anahtarı/uç noktası TANIMLI DEĞİL veya bu
    görev kapsamında gerçek çağrı BİLİNÇLİ OLARAK devre dışı bırakıldı."""


class CitationGroundingError(RuntimeError):
    """LLM özeti, KENDİSİNE VERİLEN literatür kümesiyle
    TEMELLENDİRİLEMEDİ -- özet REDDEDİLİR. `SanitizationViolationError`
    ile aynı felsefe: sessizce düzeltme/kırpma YOK, fail-loud."""


class FabricatedPmidError(CitationGroundingError):
    """Özette, bize verilen retrieved chunk kümesinde BULUNMAYAN (ya da
    hiç PMID'ye çözülemeyen/biçimsiz) bir PMID alıntılandı."""


class UncitedSummaryError(CitationGroundingError):
    """Sözleşme "PMID-referanslı özet" diyor ama özet HİÇ PMID
    alıntılamadı (ya da alıntılanacak literatür hiç YOKTU)."""


class LLMResponseFormatError(RuntimeError):
    """Sağlayıcı yanıtı sözleşmeli JSON şemasına ÇÖZÜLEMEDİ.

    ⚠️ BİLİNÇLİ OLARAK `CitationGroundingError`'ın ALT TİPİ DEĞİLDİR:
    biçim hatası bir "uydurma alıntı" değildir ve `rag_pipeline` ikisini
    FARKLI ele alır (biçim hatası = geçici/teknik arıza, literatür bloğu
    ayakta kalır; uydurma alıntı = doğruluk ihlali, blok düşer). İkisini
    tek tipte birleştirmek bu ayrımı kaybettirirdi."""


@dataclass(frozen=True)
class LLMSummaryResult:
    """LLM yanıt şeması.

    `summary_text` **TÜRKÇE** özettir (G3, Barış kararı 2026-09-14);
    `pipeline/rag_pipeline.py` bunu yanıta `summary_tr` adıyla koyar --
    alan adının dili açıkça söylemesi istenmişti."""

    summary_text: str
    cited_pmids: tuple[str, ...]
    model_name: str
    disclaimer: str = (
        "Bu özet bir Klinik Karar Destek Aracı çıktısıdır; bağımsız tanı "
        "veya tedavi kararı vermez. Nihai klinik karar yetkisi ve "
        "sorumluluğu lisanslı sağlık profesyoneline aittir."
    )
    # Maliyet/performans olcumu (gorev talimati: ilk gercek kosuda
    # sure/token/maliyet raporlanacak). Sahte saglayicili testlerde None.
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_seconds: float | None = None


#: Modele verilen rol/biçim talimatı. HASTA VERİSİ İÇERMEZ -- bu yüzden
#: sanitizasyon kapısının konusu değildir; kapı, veri taşıyan KULLANICI
#: prompt'una uygulanır (`build_llm_prompt`).
_SYSTEM_PROMPT = (
    "Sen bir klinik karar destek asistanisin. Sana verilen hasta "
    "ozniteliklerini ve PMID etiketli literatur parcalarini kullanarak "
    "TURKCE bir ozet yazarsin. Kurallar: "
    "(a) YALNIZCA sana verilen literatur parcalarina dayan; "
    "(b) sana VERILMEYEN hicbir PMID'yi alintilama -- kendi hafizandan "
    "kaynak UYDURMA; "
    "(c) en az bir PMID alintila; "
    "(d) bagimsiz tani veya tedavi onerisi VERME, yalnizca literaturu "
    "ozetle; "
    "(e) ozet en fazla 200 kelime olsun. "
    "Yanitini SADECE su semada bir JSON nesnesi olarak ver, baska hicbir "
    "metin ekleme: "
    '{"summary_tr": "<turkce ozet metni>", "cited_pmids": ["<pmid>"]}'
)


def build_llm_system_prompt() -> str:
    """Rol/biçim talimatı (sabit, hasta verisi TAŞIMAZ)."""

    return _SYSTEM_PROMPT


def build_llm_prompt(context: SanitizedLLMContext) -> str:
    """`SanitizedLLMContext`'ten deterministik bir prompt METNİ üretir
    (henüz hiçbir API'ye GÖNDERİLMEZ). Fonksiyon imzası SADECE bu tipi
    kabul eder -- bu, sanitizasyondan geçmemiş bir string'in buraya
    yanlışlıkla verilmesini Python seviyesinde ENGELLEMEZ (duck typing),
    ama kod-inceleme/tip-kontrolcüsü (mypy vb.) seviyesinde YAKALANIR;
    asıl yapısal garanti `call_llm_summarize`'daki ikinci kapıdır (aşağı
    bakın)."""

    lines = [
        "Sen bir klinik karar destek asistanısın. Aşağıdaki hasta "
        "özniteliklerine ve PMID kaynaklı literatür parçalarına dayanarak "
        "PMID referanslı, kısa (en fazla 200 kelime) bir klinik özet üret. "
        "Hiçbir bağımsız tanı/tedavi önerisi verme, yalnızca literatürü "
        "özetle.",
        "",
        f"Hasta öznitelikleri: {context.patient_context_text}",
        "",
        "Literatür parçaları:",
    ]
    for snippet in context.literature_snippets:
        lines.append(f"[PMID:{snippet.pmid}] {snippet.title} -- {snippet.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PMID TEMELLENDİRME (grounding) KAPISI -- 2026-09-13, rag-agent-C.
# ---------------------------------------------------------------------------

# Modelin metin İÇİNDE geçirdiği PMID referansları ("PMID: 12345678",
# "[PMID 12345678]", "pmid#12345678"). AÇIKÇA "PMID" ile ETİKETLENMİŞ
# olması ZORUNLU -- çıplak bir sayı (örneklem büyüklüğü, yıl, doz, hasta
# sayısı) PMID SAYILMAZ, aksi halde her "n=295" yanlış pozitif olurdu.
_PMID_IN_TEXT_PATTERN = re.compile(r"(?i)\bPMID\s*[:#]?\s*(\d{1,9})\b")

# "PMID:12345678" / " 12345678 " gibi biçimlerden çekirdek numarayı ayıklar.
_PMID_PREFIX_PATTERN = re.compile(r"(?i)^\s*pmid\s*[:#]?\s*")


def _normalize_pmid(raw: str) -> str:
    """Bir PMID dizesini karşılaştırma biçimine indirger: baş/son boşluk
    ve isteğe bağlı "PMID:"/"PMID#" öneki atılır.

    ⚠️ BİLİNÇLİ OLARAK "hoşgörülü" DEĞİL: baştaki sıfırlar KIRPILMAZ,
    rakam-dışı içerik TEMİZLENMEZ. Sebep -- her "toparlama" sessiz bir
    düzeltmedir; biçimsiz bir alıntı, doğrulanamayan bir alıntıdır ve
    fail-loud reddedilmelidir. AYNI fonksiyon HEM modelin alıntılarına
    HEM retrieved chunk PMID'lerine uygulanır, yani iki taraf arasında
    normalizasyon ASİMETRİSİ YOKTUR (2026-08-18'in "iki katman aynı
    zaafı paylaşmasın" dersinin buradaki karşılığı: burada ASİMETRİ
    tehlikeli olurdu, çünkü asimetri sahte bir 'eşleşmedi' veya sahte
    bir 'eşleşti' üretebilirdi)."""

    return _PMID_PREFIX_PATTERN.sub("", str(raw)).strip()


def allowed_pmids(context: SanitizedLLMContext) -> frozenset[str]:
    """Modelin alıntılamasına İZİN VERİLEN PMID kümesi = bize verilen
    `SanitizedLLMContext` içindeki retrieved chunk'ların PMID'leri.
    Başka hiçbir kaynak (modelin parametrik hafızası dahil) meşru
    değildir."""

    return frozenset(_normalize_pmid(s.pmid) for s in context.literature_snippets)


def extract_cited_pmids(result: LLMSummaryResult) -> tuple[list[str], list[str]]:
    """Modelin ürettiği TÜM PMID'leri iki kaynaktan toplar:
      (1) `cited_pmids` alanı (yapılandırılmış alan),
      (2) `summary_text` içinde AÇIKÇA "PMID" etiketiyle geçen numaralar.

    (2) neden dahil: `summary_text` de jüriye/kullanıcıya GÖSTERİLİYOR.
    Model, `cited_pmids` listesine koymadığı bir PMID'yi özet metnine
    yazarsa, yalnız (1)'i denetleyen bir guard bu uydurmayı KAÇIRIRDI --
    yani guard, gösterilen yüzeyin tamamını değil bir kısmını korurdu.

    ⚠️ DÜRÜSTLÜK NOTU (kapsam sınırı): "PMID" etiketi OLMADAN, düz bir
    sayı olarak yazılmış bir kaynak referansı bu tarama tarafından
    YAKALANMAZ -- yakalamaya çalışmak her örneklem büyüklüğünü/yılı
    reddetmek olurdu (fail-closed guard'da yanlış pozitif = özetin
    tamamen engellenmesi). Bu bilinen ve kabul edilmiş ödünleşimdir.

    Döner: (alan_kaynaklı, metin_kaynaklı) -- ikisi de normalize edilmiş."""

    from_field = [_normalize_pmid(p) for p in (result.cited_pmids or ())]
    from_text = [_normalize_pmid(m.group(1)) for m in _PMID_IN_TEXT_PATTERN.finditer(result.summary_text or "")]
    return from_field, from_text


def assert_citations_are_grounded(
    result: LLMSummaryResult, context: SanitizedLLMContext
) -> None:
    """Modelin ürettiği HER PMID'nin, bize verilen retrieved chunk
    kümesinde BULUNDUĞUNU doğrular. `assert_no_identifiers_leaked` ile
    AYNI sözleşme: ihlalde EXCEPTION, sessiz kırpma/düzeltme YOK.

    Üç ret durumu:

    1) **Küme dışı PMID** -> `FabricatedPmidError`. Mesaj, hangi
       PMID'nin uydurulduğunu VE beklenen kümeyi açıkça söyler (jüri
       sorusuna cevap verebilmek için).

    2) **Biçimsiz/çözülemeyen PMID** (rakam dışı içerik, boş dize) ->
       `FabricatedPmidError`. Doğrulanamayan alıntı = temellendirilmemiş
       alıntı.

    3) **HİÇ PMID alıntılanmadı** -> `UncitedSummaryError`.
       GEREKÇE (görev talimatı "karar ver ve gerekçelendir"): sözleşme
       `LLMSummaryResult`'ta ve `build_llm_prompt`'ta açıkça "PMID
       REFERANSLI özet" diyor. Sıfır alıntı KABUL EDİLEMEZ, çünkü:
         (a) Sıfır alıntı, tam olarak guard'ın engellemeye çalıştığı
             tehlikeli davranışın işaretidir -- model, verilen
             literatürü kullanmak yerine PARAMETRİK HAFIZASINDAN
             konuşmuş olabilir. Uydurma PMID'yi reddedip "hiç kaynak
             göstermeyen" özeti kabul etmek, halüsinasyonu YASAKLAMAK
             yerine yalnız İZİNİ SİLMEK olurdu.
         (b) Jüri "bu kaynağı nereden buldunuz" diye sorduğunda
             cevabımız olmalı; kaynaksız bir özet bu soruya
             yapısal olarak cevap veremez.
         (c) Fail-loud ucuz: bu durumda `rag_pipeline`'ın zaten var olan
             graceful-degradation yolu devreye girer (literatür özeti
             ÜRETİLEMEDİ), yani kullanıcı yanlış bir şey görmez -- hiçbir
             şey görmez. Yanlış görmek, görmemekten kötüdür.
       ⚠️ Bu kararın bilinen maliyeti: gerçekten alakasız literatür
       gelmişse ve model dürüstçe "ilgili kaynak yok" demek isterse, o
       dürüst cevap da REDDEDİLİR. Bu bilinçli seçimdir -- böyle bir
       "boş özet" zaten kullanıcıya gösterilmemelidir.

    4) **Bağlamda hiç literatür parçası YOK** -> `UncitedSummaryError`
       (özel mesaj). Bu durumda temellendirilebilir bir özet üretmek
       MANTIKEN imkânsızdır; guard bunu sessizce "0 alıntı normaldir"
       diye geçemez."""

    allowed = allowed_pmids(context)
    from_field, from_text = extract_cited_pmids(result)
    cited = from_field + from_text

    if not allowed:
        raise UncitedSummaryError(
            "PMID temellendirme kapisi: verilen SanitizedLLMContext'te HIC "
            "literatur parcasi YOK (retrieved chunk sayisi 0) -- "
            "'PMID-referansli ozet' sozlesmesi MANTIKEN saglanamaz, ozet "
            f"REDDEDILDI. Modelin alintiladigi PMID sayisi: {len(cited)}."
        )

    if not cited:
        raise UncitedSummaryError(
            "PMID temellendirme kapisi: model HIC PMID alintilamadi "
            "(cited_pmids bos VE summary_text'te 'PMID' etiketli referans "
            "yok). Sozlesme 'PMID-referansli ozet' oldugu icin sifir alinti "
            "KABUL EDILMEZ -- ozet REDDEDILDI. Beklenen (izin verilen) "
            f"PMID kumesi: {sorted(allowed)}."
        )

    malformed = sorted({p for p in cited if not p.isdigit()})
    if malformed:
        raise FabricatedPmidError(
            "PMID temellendirme kapisi: DOGRULANAMAYAN/bicimsiz PMID "
            f"alintisi: {malformed}. Bir alinti rakamlara cozulemiyorsa "
            "retrieved kumeyle karsilastirilamaz; sessizce temizlenmez, "
            f"ozet REDDEDILDI. Beklenen PMID kumesi: {sorted(allowed)}."
        )

    fabricated = sorted(set(cited) - allowed)
    if fabricated:
        raise FabricatedPmidError(
            "PMID temellendirme kapisi: model, kendisine VERILMEYEN "
            f"PMID('ler)i alintiladi: {fabricated}. Beklenen (retrieved "
            f"chunk'lardan gelen) PMID kumesi: {sorted(allowed)}. "
            "Ozet REDDEDILDI -- uydurulan PMID sessizce KIRPILMAZ ve "
            "kalan metin KABUL EDILMEZ (bkz. assert_no_identifiers_leaked "
            "ile ayni fail-loud sozlesmesi)."
        )


# ---------------------------------------------------------------------------
# Sağlayıcı yanıtının JSON zarfını çözme.
# ---------------------------------------------------------------------------

#: Bazı modeller JSON'u ``` çitleri içinde döndürür. Çiti soymak bir
#: "sessiz düzeltme" DEĞİLDİR -- içeriği değiştirmez, yalnız zarfı açar.
_JSON_FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_llm_json_payload(text: str) -> tuple[str, tuple[str, ...]]:
    """Sağlayıcı metnini `(summary_tr, cited_pmids)` ikilisine çözer.

    KATI: şema dışı her şey `LLMResponseFormatError`. Gerekçe
    `_normalize_pmid` ile aynı: "toparlamak" sessiz düzeltmedir ve
    doğrulanamayan bir çıktıyı doğrulanmış gibi gösterir.

    ⚠️ `cited_pmids` boşsa BURADA hata verilmez -- o karar
    `assert_citations_are_grounded`'ın (`UncitedSummaryError`) işidir.
    İki yerde ayrı ayrı reddetmek, hangi kapının reddettiğini
    belirsizleştirirdi."""

    if not isinstance(text, str) or not text.strip():
        raise LLMResponseFormatError("Saglayici bos metin dondurdu.")

    candidate = text.strip()
    fenced = _JSON_FENCE_PATTERN.match(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        payload = json.loads(candidate)
    except ValueError as exc:
        raise LLMResponseFormatError(
            f"Saglayici yaniti JSON olarak cozulemedi: {exc}. Sozlesme: "
            '{"summary_tr": "...", "cited_pmids": [...]}'
        ) from exc

    if not isinstance(payload, dict):
        raise LLMResponseFormatError(
            f"Saglayici yaniti JSON NESNESI degil (tip: {type(payload).__name__})."
        )

    summary = payload.get("summary_tr")
    if not isinstance(summary, str) or not summary.strip():
        raise LLMResponseFormatError(
            "Saglayici yanitinda 'summary_tr' alani YOK veya bos "
            f"(anahtarlar: {sorted(payload)})."
        )

    raw_pmids = payload.get("cited_pmids", [])
    if not isinstance(raw_pmids, list) or any(not isinstance(p, (str, int)) for p in raw_pmids):
        raise LLMResponseFormatError(
            "Saglayici yanitinda 'cited_pmids' bir dize listesi DEGIL "
            f"(okunan tip: {type(raw_pmids).__name__})."
        )

    return summary.strip(), tuple(str(p) for p in raw_pmids)


def _invoke_llm_provider(prompt: str) -> LLMSummaryResult:
    """GERÇEK sağlayıcı çağrısı (2026-09-14, rag-agent-P5).

    ⚠️ **VARSAYILAN DAVRANIŞ DEĞİŞMEDİ.** `GBMAID_LLM_ENABLED` tanımsız
    veya `false` iken bu fonksiyon -- 2026-09-13'teki hâliyle BİREBİR
    aynı şekilde -- `LLMNotConfiguredError` fırlatır. Yapılandırma eksik
    (sağlayıcı/model/anahtar yok, model adı hareketli alias) ise yine
    `LLMNotConfiguredError` fırlatır, ama mesaj NEDENİ söyler.

    Bu fonksiyon PRIVATE'tır ve BİLİNÇLİ OLARAK öyle kalmalıdır: dışarıya
    açık tek yol `call_llm_summarize`'dır ve o yol PMID temellendirme
    kapısını KOŞULSUZ uygular. Bu fonksiyonu public yapmak, guard'ı
    atlayan bir kod yolu AÇMAK demektir.

    (Testlerde bu fonksiyon monkeypatch'lenerek sahte bir LLM yanıtı
    üretilir -- gerçek API çağrısı YAPILMAZ.)

    Fırlatır:
      * `LLMNotConfiguredError` -- kapalı veya yapılandırma eksik/geçersiz.
      * `LLMProviderError`      -- ağ/kota/HTTP/SDK-yok (teknik arıza).
      * `LLMResponseFormatError`-- yanıt sözleşmeli JSON'a çözülemedi."""

    try:
        config = load_llm_config()
    except LLMDisabledError as exc:
        raise LLMNotConfiguredError(str(exc)) from exc
    except LLMConfigError as exc:
        raise LLMNotConfiguredError(
            f"LLM yapilandirmasi eksik/gecersiz: {exc}"
        ) from exc

    started = time.monotonic()
    response = invoke_provider(
        config,
        system_prompt=build_llm_system_prompt(),
        user_prompt=prompt,
    )
    elapsed = time.monotonic() - started

    summary_tr, cited = parse_llm_json_payload(response.text)
    return LLMSummaryResult(
        summary_text=summary_tr,
        cited_pmids=cited,
        # Saglayicinin BILDIRDIGI model adi tercih edilir (istenen ile
        # servis edilen ayrilabilir); yoksa yapilandirmadaki ad.
        model_name=response.model_name or config.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_seconds=elapsed,
    )


def call_llm_summarize(context: SanitizedLLMContext) -> LLMSummaryResult:
    """GERÇEK ÇAĞRI YAPILMAZ -- bu görevin kapsamı SADECE arayüzü yazmak
    (görev talimatı, madde 6). `_invoke_llm_provider` şu an her zaman
    `LLMNotConfiguredError` fırlattığı için bu fonksiyon da fırlatır.

    `pipeline/rag_pipeline.py::generate_literature_summary` bu hatayı
    YAKALAR ve "literatür özeti üretilemedi (LLM yapılandırılmadı)"
    notuyla graceful-degradation sözleşmesine uygun devam eder -- pipeline
    ÇÖKMEZ.

    İKİ YAPISAL KAPI (ikisi de KOŞULSUZ, bayrakla atlanamaz):

      1) ÇAĞRIDAN ÖNCE -- `assert_no_identifiers_leaked(prompt)`
         (savunma katmanı 2'nin son tekrarı): bu fonksiyon gerçek bir HTTP
         çağrısına dönüştürülse bile bu satır SİLİNMEDEN kimlik sızıntılı
         bir prompt asla gönderilemez.

      2) ÇAĞRIDAN SONRA -- `assert_citations_are_grounded(result, context)`
         (2026-09-13): modelin ürettiği HER PMID, bize verilen retrieved
         chunk kümesiyle karşılaştırılır. Uydurma PMID varsa özet
         REDDEDİLİR. Bu kontrol sağlayıcıdan BAĞIMSIZDIR ve `return`'den
         ÖNCE, `if`siz/parametresiz çalışır -- guard'ı devre dışı bırakan
         bir konfigürasyon bayrağı YOKTUR ve EKLENMEMELİDİR."""

    prompt = build_llm_prompt(context)
    assert_no_identifiers_leaked(prompt)

    result = _invoke_llm_provider(prompt)

    # KOŞULSUZ: sağlayıcı ne dönerse dönsün, guard HER ZAMAN çalışır.
    assert_citations_are_grounded(result, context)

    return result
