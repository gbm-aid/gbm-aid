"""Hafta 5 — Prompt Sanitizasyon Katmanı (`raw/mimari/v45.txt` Bölüm 8.2).

Amaç: hasta profili LLM bağlamına eklenmeden ÖNCE kimlik/tarih/kurum
bilgisi kaldırılır, yalnız öznitelik-düzeyi bilgi geçer. Örnek (v45.txt
satır 1310-1311): LLM'e "TCGA-06-5412, 2009 tarihli, UCSF" DEĞİL,
"62 yaşında, IDH-wildtype, MGMT-metile" gitmeli.

⚠️ 2026-08-18 ÇAPRAZ DOĞRULAMA DÜZELTMESİ (reviewer CRITICAL 1): İLK
sürüm regex'leri yalnız TEK bir ayraç biçimini ("-") tanıyordu ve
`assert_no_identifiers_leaked` AYNI desen listesini paylaşıyordu --
format varyasyonu (alt çizgi, ayraçsız, noktalı/kompakt tarih, kurum
adından KOPUK bare-ID) hem redaksiyonu HEM son-kontrolü AYNI ANDA
atlatıyordu. Somut kanıtlanmış bypass'lar ve düzeltmeleri:

  'TCGA_06_5412 patient'      -> DÜZELTİLDİ (ayraç-esnek desen: [-_ ]?)
  'TCGA065412 patient'        -> DÜZELTİLDİ (ayraçsız da eşleşir)
  '20090315 scan'             -> DÜZELTİLDİ (kompakt ISO tarih deseni eklendi)
  '06-5412 from TCGA cohort'  -> DÜZELTİLDİ (bağlamsal geçiş: kurum adına
                                  YAKIN [±80 karakter] bare-ID-şekilli sayı
                                  öbekleri de redakte edilir -- bkz.
                                  `_find_context_sensitive_spans`)

Bu düzeltmeler `sanitize_free_text`'i regex-`subn`-zinciri yerine
POZİSYON-TABANLI (span-based) bir algoritmaya çevirdi -- tüm eşleşme
adayları ÖNCE ORİJİNAL metin üzerinde toplanır, sonra birleştirilip TEK
geçişte redakte edilir; bu hem "kurum tokenı ID'den önce silinirse geri
kalan sayı açıkta kalır" sırası sorununu ORTADAN KALDIRIR (artık sıra
önemli değil, hepsi orijinal koordinatlarla eşleşiyor) hem de
kontamünasyonu önler.

⚠️ 2026-09-11 GÜVENLİK KAPANIŞI (Ege alanı, G1) -- 2026-08-19 reviewer
turunun belgelediği (xfail-strict ile nöbette tutulan) 4 residüel bypass
KAPATILDI. Kök nedenler ve yapısal (regex-yaması DEĞİL) düzeltmeler:

  1) 'TCGA--06--5412' (çift ayraç) -- `_SEP` TEK karakterlik ayracı
     ({0,1}) kabul ediyordu. DÜZELTME: `_SEP` artık 0-3 ayraç karakteri
     kabul eder (`_SEP_CLASS{0,3}`) -- ayraç TEKRARINA karşı genel.
  2) '5412-06-TCGA' (ters sıra) -- `_PATIENT_ID_PATTERNS` yalnız ileri
     yönde (KURUM-ayraç-ayraç-sayı) token sırasını tanıyordu. DÜZELTME:
     desenler artık TEK BİR token-listesinden (`_PATIENT_ID_TOKEN_SPECS`)
     hem ileri hem TERS sırada otomatik üretilir (`_build_id_patterns`)
     -- yeni bir kaynak eklenirse iki ayrı regex elle yazmak GEREKMEZ.
  3) '06-5412 from' + 90+ karakter + 'TCGA cohort' (pencere kenarı) --
     sabit `_CONTEXT_WINDOW=80` karakter, aynı noktalamasız cümle içinde
     bile bu sınırı aşan boşluklarda KÖRDÜ. DÜZELTME: yakınlık artık
     SADECE karakter sayısına değil, CÜMLE sınırına da bakar (`_sentence_
     spans`) -- bare-ID-şekilli öbek, kurum adıyla AYNI cümledeyse (nokta/
     ünlem/soru/satır sonu ile ayrılmamışsa) mesafe SIFIR kabul edilir.
     Sabit karakter penceresi (80) DIŞARI ATILMADI, "VEYA" ile korunuyor
     -- kısaltma noktaları ("Dr.") gibi durumlarda cümle bölünmesi
     hatalı olabileceği için modest bir karakter-penceresi yedek olarak
     kalıyor. Gerekçe: sabit pencere büyütmek (örn. 80->200) yalnız eşiği
     ötelemekti, "bir sonraki testte tekrar aşılır" sorununu ÇÖZMEZ;
     cümle sınırı dilbilimsel/doğal bir sınır olduğu için karakter
     sayısına göre OYNANMASI daha zor (bkz. modül altındaki yeni testler
     -- 120 karakter mesafeli karşıt-örnek de kapanıyor).
  4) 'TCGA－06－5412' (Unicode tam-genişlik tire U+FF0D) -- ayraç sınıfı
     yalnız ASCII (`-_ `) tanıyordu. DÜZELTME: ayraç karakter sınıfı
     (`_SEP_CLASS`) Unicode tire/dash kategorisine genişletildi (U+2010-
     U+2015 [hyphen..horizontal bar], U+2212 [minus sign], U+FF0D [fullwidth
     hyphen-minus]). AYNI genişletme hem redaksiyon (`_SEP_CLASS`, bare-
     residual) HEM bağımsız katı kapı (`_DIGIT_CLUSTER_PATTERN`) tarafından
     PAYLAŞILIYOR (aşağıdaki ÇİFT KATMAN notuna bkz.) -- 2026-08-18'in
     dersi ("iki katman aynı regex'i paylaşıp birlikte kör olmasın") bu
     kez "aynı genişletmeyi AYRI AYRI ama TUTARLI şekilde al" olarak
     uygulandı: `assert_patient_context_has_no_digit_clusters` ayrı bir
     desen kullanmaya devam ediyor (bağımsızlık kilit tasarım kararı,
     aşağıda), ama aynı `_SEP_CLASS` sabitini import ederek Unicode
     kapsamını PAYLAŞIYOR -- iki katman birbirinden regex-KODU olarak
     bağımsız kalıyor, ama karakter-KAPSAMI olarak senkron.

  ⚠️ DÜRÜSTLÜK NOTU: bu dört BİLİNEN formu + aşağıdaki yeni karşıt-
  örnekleri kapatır. "Her yeni/görülmemiş kimlik şekline karşı korur"
  iddiası KURULMAZ -- enumere edilmiş desen listesi hâlâ sonlu bir
  listedir (bağımsız/geniş koruma yalnız `assert_patient_context_has_
  no_digit_clusters`'ın kapsadığı `patient_context_text` yolu içindir).

⚠️ 2026-09-13 G2 -- İKİNCİ NESİL KAPANIŞ (`rag-agent-B`).

BULGU: G1'in (2026-09-11) dört bypass'ı KAPATTIĞI canlı doğrulandı; ama
G1 bunu AYRAÇ LİSTESİNİ GENİŞLETEREK yapmıştı (8 Unicode tire eklemek,
tekrar sınırını {0,1}->{0,3} yapmak). Bu, eşiği ÖTELEDİ, kök nedeni
çözmedi. 2026-09-13 ölçümü aynı sınıfın bir adım ötesinde SEKİZ yeni
sızıntı buldu (hepsi canlı, `sanitize_free_text` rakam gövdesini metinde
BIRAKIYORDU):

  'TCGA----06----5412'        {0,3} tekrar sınırı aşıldı
  'TCGA.06.5412'              '.' ayraç listesinde yok
  'TCGA<U+00A0>06...'         NBSP listede yok
  'TCGA<U+00AD>06...'         yumuşak tire listede yok
  'TCGA<U+200B>06...'         sıfır-genişlik boşluk listede yok
  'TCGA-06-54<U+200B>12'      rakam GÖVDESİNE görünmez karakter
  'ＴＣＧＡ-06-5412'            tam-genişlik HARF -> kurum tokenı bile eşleşmiyor
  'UCSF<U+00A0>PDGM<U+00A0>190'

YAPISAL DÜZELTME (liste genişletmek DEĞİL): girdi artık TÜM tespit
yüzeylerinde önce `normalize_for_detection()`'dan geçer -- NFKC +
Unicode KATEGORİ katlaması (`Cf` düşür, `Pd`/U+2212 -> "-", `Zs` -> " ").
Eşleşmeler normalize metinde aranır, İNDEKS HARİTASIYLA orijinal
koordinatlara geri yazılır; böylece kullanıcıya dönen metin normalize
EDİLMEZ (meşru Unicode -- Türkçe karakter, '≥', em-dash -- korunur).
Artık "listeye hangi karakter eklendi" sorusu YOKTUR. Ayraç tekrar
sınırı da kaldırıldı (`{0,3}` -> `*`).

BAĞIMSIZLIK (2026-08-18 dersinin literatür-metni yolundaki karşılığı):
`assert_no_identifiers_leaked` artık İKİ AYRI MEKANİZMA kullanır --
enumere desenler VE `identifier_skeleton()` (alfanümerik olmayan HER
karakteri sıyırıp şekil arama). Ayraç/tekrar/span mantığındaki bir hata
ikincisini KÖR EDEMEZ. Detay: `_SKELETON_SPEC_NAMES` üstündeki blok.

⚠️ BU İŞ SIRASINDA KENDİ KODUMDA YAKALANAN HATA (kayda geçer): iskelet
desenleri `"".join(tokens).upper()` ile üretiliyordu; bu regex
parçalarını da büyütüp `\d` -> `\D` yapıyordu (RAKAM -> RAKAM OLMAYAN).
Sözde "bağımsız kapı" hem gerçek ID'yi kaçırıyor hem meşru metni
engelliyordu -- yani sessizce İŞE YARAMAZ olacaktı. Yalnızca ÖLÇÜMLE
fark edildi; `_upper_literal_only()` + `test_skeleton_patterns_are_not_
inverted_by_uppercasing` bunu kalıcı nöbete aldı.

⚠️ HÂLÂ AÇIK / DEVREDİLEN: enumere katmanın iki geniş spec'i
(`LUMIERE-Patient`, `generic-source-numeric`) meşru literatür dilinde
HAM metne uygulanırsa yanlış pozitif verir ('patient 12 months',
'recurrent GBM 2019'). G2'nin getirdiği gerileme DEĞİL -- G1 desenleriyle
de eşleşiyordu (ölçüldü). Üretim yolunda etkisi 0/41 (kapı redaksiyondan
SONRA uygulanıyor). Bkz. `test_known_preexisting_enumerated_layer_
false_positive`.

⚠️ DÜRÜSTLÜK NOTU (docstring'in ESKİ, YANLIŞ TEMELLENDİRİLMİŞ gerekçesi
DÜZELTİLDİ): önceki sürüm "TCGA-NN-NNNN gibi iç ID'lerimizle örtüşmesi
neredeyse imkânsız" diyordu -- bu YANLIŞTI, bizim `patient_id`'lerimiz
ZATEN literal TCGA barkodu, aynı ID uzayını paylaşıyoruz. Gerçek koruma
"format örtüşmezliği" DEĞİL, "yayınlanmış bir abstract'ta BİZİM
kohortumuzdaki TEKİL bir hasta barkodunun kelimesi kelimesine geçme
OLASILIĞININ düşüklüğü" -- bu bir garanti değil, olasılıksal bir
savunma katmanıdır. Asıl GARANTİ katman 1'dedir (aşağıda).

İKİ KATMANLI SAVUNMA (görev talimatı: "Bu katmanı atlayan bir kod yolu
OLMAMALI — yapısal olarak zorunlu kıl") + reviewer'ın istediği BAĞIMSIZLIK
düzeltmesi:

  1) TİP-DÜZEYİ ALLOWLIST (BİRİNCİL/GERÇEK GARANTİ): `PatientAttributeProfile`
     dataclass'ı YAPISAL OLARAK `patient_id`/tarih/kurum alanı TAŞIMAZ —
     bu alanları taşımak için Python tip sistemini atlatmak gerekir,
     "unutma" ile mümkün değildir. LLM'e giden hasta-özniteliği metni
     SADECE `render_patient_context_text()` ile, SADECE bu tipten üretilir.
     Bu katman regex'e HİÇ bağımlı DEĞİLDİR -- format varyasyonu onu
     etkilemez (hasta ID'si zaten hiçbir zaman bu tipe GİRMEZ).

  2) REGEX KAPISI (İKİNCİL, olasılıksal savunma -- literatür metni gibi
     serbest/üçüncü-taraf metinler için): `SanitizedLLMContext` yalnızca
     `build_sanitized_context()` üzerinden inşa edilebilir.
     `assert_no_identifiers_leaked()` artık `sanitize_free_text`'in
     kullandığı desenlerle SINIRLI DEĞİL -- reviewer'ın istediği
     BAĞIMSIZLIK için hasta bağlamı metnine (`patient_context_text`)
     AYRICA `assert_patient_context_has_no_digit_clusters()` uygulanır:
     bu, ENUMERE edilmiş bir desen listesi DEĞİL, "5+ karakterlik herhangi
     bir rakam/ayraç öbeği" gibi çok daha geniş/bağımsız bir kural --
     `render_patient_context_text()`'in sabit şablonu ZATEN böyle bir öbek
     ÜRETMEDİĞİ için burada YANLIŞ POZİTİF riski YOK (bkz. fonksiyon
     dokstring'i). AYNI geniş kural literatür metnine (serbest, p-değeri/
     doz/örneklem büyüklüğü gibi meşru rakam öbekleri İÇEREBİLİR)
     UYGULANMAZ -- orada hâlâ enumere edilmiş `assert_no_identifiers_
     leaked()` + bağlamsal bare-ID kontrolü kullanılır (aşağıda gerekçeli).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Sequence


class SanitizationViolationError(RuntimeError):
    """Sanitize edilmiş bağlamda hâlâ bir kimlik/tarih deseni bulundu --
    LLM çağrısına ASLA izin verilmez, sessizce temizlenip devam edilmez."""


# ---------------------------------------------------------------------------
# 1) Bilinen hasta ID formatları (canlı DB'de doğrulanmış -- 2026-08-18):
#    TCGA-NN-NNNN (295), UPENN-GBM-NNNNN (630), UCSF-PDGM-NNN (295),
#    Patient-NNN (91, LUMIERE). Her biri AYRI test edilir (görev talimatı).
#    AYRAÇ-ESNEK ("-", "_", " ", ASCII/Unicode tire, TEKRARLI, veya YOK) --
#    reviewer CRITICAL 1 (2026-08-18) + residüel bypass kapanışı (2026-09-11).
# ---------------------------------------------------------------------------
#
# Unicode tire/dash kapsamı: U+2010-U+2015 (hyphen .. horizontal bar),
# U+2212 (minus sign), U+FF0D (fullwidth hyphen-minus). Bu sabit hem
# ayraç sınıfı (`_SEP_CLASS`) hem bağımsız katı kapı
# (`_DIGIT_CLUSTER_PATTERN`) tarafından PAYLAŞILIR -- 2026-08-18'in dersi
# ("iki katman aynı regex'i paylaşıp birlikte kör olmasın") burada "aynı
# KARAKTER KAPSAMINI paylaş ama aynı REGEX KODUNU paylaşma" şeklinde
# uygulanıyor: iki katman hâlâ ayrı desenler kullanıyor (bağımsızlık
# korunuyor), ama ikisi de aynı Unicode tire kümesini tanıyor.
_UNICODE_DASH_CHARS = "‐‑‒–—―−－"

# ---------------------------------------------------------------------------
# 2026-09-13 (G2) — NORMALİZASYON KATMANI: karakter ENUMERASYONU yerine
# KARAKTER SINIFI. Bkz. modül dokstring'i "2026-09-13 G2" bölümü.
#
# Tüm tespit yüzeyleri (redaksiyon, enumere kapı, katı rakam-öbeği kapısı,
# iskelet kapısı) girdiyi ÖNCE bu fonksiyondan geçirir. Böylece "hangi
# görünmez/benzer karakter listeye eklendi" sorusu ORTADAN KALKAR.
# ---------------------------------------------------------------------------


def _is_ignorable(ch: str) -> bool:
    """Görünmez/biçimlendirme karakteri mi? (sıfır-genişlik boşluk, yumuşak
    tire, BOM, yön-işaretleri ...) — Unicode `Cf` KATEGORİSİNİN tamamı,
    tek tek karakter listesi DEĞİL."""

    return unicodedata.category(ch) == "Cf"


def _fold_char(ch: str) -> str:
    """Ayraç-benzeri karakterleri KANONİK ASCII karşılığına indirger:
    her tire (`Pd` kategorisi + `Sm` sınıfındaki U+2212 MINUS SIGN) -> "-",
    her boşluk (`Zs`/`Zl`/`Zp`) -> " "."""

    if ch in _UNICODE_DASH_CHARS or unicodedata.category(ch) == "Pd":
        return "-"
    if unicodedata.category(ch) in ("Zs", "Zl", "Zp"):
        return " "
    return ch


def normalize_for_detection(text: str) -> tuple[str, list[int]]:
    """Metni tespit için normalize eder VE her normalize karakterin
    ORİJİNAL metindeki indeksini tutan haritayı döndürür.

    Üç işlem (hepsi KATEGORİ tabanlı, karakter listesi tabanlı DEĞİL):
      1) `Cf` (görünmez/biçim) karakterleri DÜŞÜRÜLÜR — 'TCGA<ZWSP>06...'
         gibi araya görünmez karakter serpme saldırısı çöker.
      2) Karakter-başına NFKC — tam-genişlik harf/rakam (ＴＣＧＡ, ０６)
         ASCII'ye iner, NBSP normal boşluğa döner.
      3) Tire/boşluk KATEGORİLERİ kanonik "-"/" " karakterine indirgenir.

    İndeks haritası sayesinde eşleşmeler NORMALİZE metinde aranıp
    ORİJİNAL koordinatlara geri yazılabilir — yani redaksiyon kullanıcıya
    dönen metni normalize etmek ZORUNDA DEĞİLDİR (meşru Unicode içerik,
    örn. Türkçe karakterler veya '≥', bozulmadan korunur).

    ⚠️ NFKC karakter BAŞINA uygulanır (tüm dizeye değil): bu, birleşik
    (combining) dizilerin kompozisyonunu bilinçli olarak YAPMAZ, ama
    indeks haritasını bire bir korur. Amacımız ASCII-benzeri kimlik
    şekillerini yakalamak olduğu için bu ödünleşim kabul edilmiştir.
    """

    out: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        if _is_ignorable(ch):
            continue
        for c in unicodedata.normalize("NFKC", ch):
            if _is_ignorable(c):
                continue
            out.append(_fold_char(c))
            index_map.append(i)
    return "".join(out), index_map


def _map_span(index_map: list[int], start: int, end: int) -> tuple[int, int]:
    """Normalize metindeki [start, end) span'ını ORİJİNAL metin
    koordinatlarına çevirir. Aradaki düşürülmüş (görünmez) karakterler de
    kapsanır, çünkü orijinal uç noktalar arası TÜM aralık redakte edilir."""

    return index_map[start], index_map[end - 1] + 1


# Normalizasyondan SONRA ayraç sınıfı sadeleşir: her tire zaten "-",
# her boşluk zaten " ". Geriye "hangi noktalama ayraç sayılır" kalır.
# Nokta/iki nokta/eğik çizgi EKLENDİ ('TCGA.06.5412' bypass'ı).
_SEP_CLASS = r"[-_ ./:]"
# TEKRAR SINIRI YOK (`*`): 2026-09-11'in `{0,3}` sınırı yalnız eşiği 1'den
# 3'e ötelemişti; 4 ayraç ('TCGA----06----5412') yine sızıyordu (2026-09-13
# ölçümü). Ayraç sınıfı harf/rakam İÇERMEDİĞİ için sınırsız tekrar, ID
# token'ları arasına yalnız noktalama/boşluk koyan saldırıları kapsar.
_SEP = rf"{_SEP_CLASS}*"

# Her kaynak formatı TEK BİR token dizisi olarak tanımlanır; ileri VE ters
# sıradaki regex'ler bu TEK tanımdan otomatik üretilir (`_build_id_patterns`)
# -- 'TCGA-06-5412' vs '5412-06-TCGA' (reviewer'ın ters-sıra bulgusu) için
# iki ayrı elle-yazılmış regex GEREKMEZ, kaynak eklenirse tek satır yeter.
_PATIENT_ID_TOKEN_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("TCGA", ("TCGA", r"\d{2}", r"\d{4}")),
    ("UPENN-GBM", ("UPENN", "GBM", r"\d{4,6}")),
    ("UCSF-PDGM", ("UCSF", "PDGM", r"\d{2,4}")),
    ("LUMIERE-Patient", ("Patient", r"\d{2,4}")),
    # Genel/gelecekte eklenebilecek kaynaklar için ihtiyatlı yakalama:
    # "<KAYNAK>-GBM-<sayı>" kalıbı.
    ("generic-source-numeric", (r"[A-Z]{3,10}", "GBM", r"\d{3,6}")),
)


def _build_id_patterns(
    specs: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Her token-listesinden İLERİ ve (token sayısı >= 2 ise) TERS sıradaki
    regex'i üretir. Tek bir yerde tanımlanan ayraç genişlemesi (`_SEP`)
    otomatik olarak HER iki yöne de uygulanır -- elle senkronize edilecek
    iki kopya regex YOK."""

    built: list[tuple[str, re.Pattern[str]]] = []
    for name, tokens in specs:
        forward_body = _SEP.join(tokens)
        built.append(
            (name, re.compile(rf"\b{forward_body}\b", re.IGNORECASE))
        )
        if len(tokens) >= 2:
            reverse_body = _SEP.join(reversed(tokens))
            built.append(
                (f"{name}-reversed", re.compile(rf"\b{reverse_body}\b", re.IGNORECASE))
            )
    return tuple(built)


_PATIENT_ID_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = _build_id_patterns(
    _PATIENT_ID_TOKEN_SPECS
)

# Kurum/kaynak adları -- serbest metinde tek başına geçse bile kaldırılır
# (örn. "UCSF" hastanın kaynağını ifşa eder, ID'nin parçası olmasa dahi).
_INSTITUTION_TOKENS: tuple[str, ...] = (
    "TCGA",
    "UPENN-GBM",
    "UPenn",
    "University of Pennsylvania",
    "UCSF-PDGM",
    "UCSF",
    "LUMIERE",
    "The Cancer Genome Atlas",
)
_INSTITUTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(tok) for tok in _INSTITUTION_TOKENS) + r")\b",
    re.IGNORECASE,
)

# Tarih desenleri: ISO (2009-03-15), ABD (03/15/2009), noktalı (15.03.2009),
# kompakt ISO (20090315), "March 2009", "15 March 2009" ve çıplak 4 haneli
# yıl. Noktalı + kompakt ISO reviewer CRITICAL 1 düzeltmesiyle EKLENDİ.
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b"),  # ISO
    re.compile(r"\b(19|20)\d{6}\b"),  # kompakt ISO (YYYYMMDD, örn. 20090315)
    re.compile(r"\b\d{1,2}/\d{1,2}/(19|20)\d{2}\b"),  # US/EU sayısal (slash)
    re.compile(r"\b\d{1,2}\.\d{1,2}\.(19|20)\d{2}\b"),  # noktalı (Avrupa tarzı)
    re.compile(rf"\b(?:{_MONTHS})\s+\d{{1,2}},?\s+(19|20)\d{{2}}\b", re.IGNORECASE),
    re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTHS})\s+(19|20)\d{{2}}\b", re.IGNORECASE),
    re.compile(rf"\b(?:{_MONTHS})\s+(19|20)\d{{2}}\b", re.IGNORECASE),
    re.compile(r"\b(19|20)\d{2}\b"),  # çıplak yıl (en geniş, en son uygulanır)
)

# Kurum tokenına YAKIN ayraçlı bare-ID-şekilli sayı öbekleri -- reviewer'ın
# '06-5412 from TCGA cohort' örneği (kurum adı ile ID'nin sözcüksel olarak
# KOPUK olduğu durum). Ayraç ZORUNLU (1-3 tekrar, ASCII/Unicode tire/alt
# çizgi/boşluk -- `_SEP_CLASS`, 2026-09-11 genişletmesi) -- ayraçsız çıplak
# sayılar (örn. yıl, örneklem büyüklüğü) burada KASITLI OLARAK dışarıda
# bırakılıyor, aksi halde meşru istatistiksel içerik (örn. "100-200 mg"
# doz aralığı) kurum adı aynı cümlede geçtiğinde gereksiz yere silinirdi.
# ⚠️ Bilinen kalıntı risk (fix kapsamı DIŞI, aşağıda testle belgelendi):
# bir doz/örneklem ARALIĞI ("100-200") kurum adıyla AYNI cümlede geçerse
# bu bare-residual kuralı yine de eşleşir -- bu, ayracı ZORUNLU tutmanın
# (çıplak sayıları hariç tutarken) kabul ettiği bilinen ödünleşimdir,
# yeni bir bulgu DEĞİL (orijinal 2026-08-18 tasarımının doğal sonucu).
#
# ⚠️ 2026-09-13 (G2) ASİMETRİ — BİLİNÇLİ: bu kuralın ayraç sınıfı ID
# desenlerininkinden DAR tutulur (nokta/iki nokta/eğik çizgi YOK). Gerekçe
# ölçüldü: nokta eklenmesi 'p<0.0001' (0 + "." + 0001) ve '14.6-18.2' gibi
# MEŞRU istatistikleri, kurum adı aynı cümlede geçtiği anda redakte
# ederdi. ID desenleri kurum TOKEN'ıyla çapalandığı için orada nokta
# güvenli; bu kural ise çapasız (yalnız yakınlık) olduğu için dar kalmalı.
_BARE_SEP_CLASS = r"[-_ ]"
_BARE_RESIDUAL_PATTERN = re.compile(rf"\b\d{{1,4}}{_BARE_SEP_CLASS}{{1,6}}\d{{3,6}}\b")

# 2026-09-11: sabit karakter penceresi TEK BAŞINA "pencere kenarı" bypass'ına
# (reviewer 2026-08-19) karşı kırılgandı -- her büyütme yalnız eşiği ötelerdi.
# Yakınlık artık İKİ kuralın "VEYA"sıdır:
#   (a) modest karakter penceresi (yedek -- kısaltma noktaları vb. yüzünden
#       cümle bölünmesi hatalı olabileceği durumlar için tutuluyor)
#   (b) AYNI CÜMLE (nokta/ünlem/soru/satır sonu ile bölünmemiş metin bloğu)
# Cümle sınırı karakter sayısından bağımsız olduğu için "bir sonraki testte
# yine aşılır" sorununu yapısal olarak ortadan kaldırır.
_CONTEXT_WINDOW = 80
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"[.!?\n]+")

REDACTED_TOKEN = "[REDACTED]"


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """`text`'i cümle sınırlarına (`.`, `!`, `?`, satır sonu) göre
    (start, end) span listesine böler -- bare-ID/kurum yakınlığını sabit
    karakter penceresinden BAĞIMSIZ bir dilbilimsel sınırla da ölçmek
    için (bkz. `_CONTEXT_WINDOW` üstündeki not)."""

    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENTENCE_BOUNDARY_PATTERN.finditer(text):
        spans.append((start, m.end()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def _same_sentence(pos_a: int, pos_b: int, sentence_spans: list[tuple[int, int]]) -> bool:
    for s, e in sentence_spans:
        if s <= pos_a < e and s <= pos_b < e:
            return True
    return False


def _collect_spans(text: str) -> list[tuple[int, int, str]]:
    """`text` üzerinde TÜM aday eşleşmeleri (pozisyon + kategori) toplar --
    hiçbir redaksiyon henüz UYGULANMAZ (sıra bağımsızlığı için, reviewer
    CRITICAL 1 madde 3).

    2026-09-13 (G2): eşleşmeler NORMALİZE metinde aranır, span'lar
    ORİJİNAL koordinatlara geri çevrilir (`normalize_for_detection` +
    `_map_span`). Döndürülen span'lar HER ZAMAN orijinal `text`'in
    koordinatlarındadır -- çağıran taraf (`sanitize_free_text`) orijinal
    metni keser, yani kullanıcıya dönen çıktı normalize EDİLMEZ."""

    norm, index_map = normalize_for_detection(text)
    spans: list[tuple[int, int, str]] = []
    if not norm:
        return spans

    def add(ns: int, ne: int, category: str) -> None:
        if ne <= ns:
            return
        s, e = _map_span(index_map, ns, ne)
        spans.append((s, e, category))

    for name, pattern in _PATIENT_ID_PATTERNS:
        for m in pattern.finditer(norm):
            add(m.start(), m.end(), f"patient_id:{name}")

    institution_spans = [(m.start(), m.end()) for m in _INSTITUTION_PATTERN.finditer(norm)]
    for s, e in institution_spans:
        add(s, e, "institution")

    # Bağlamsal geçiş: kurum tokenına yakın bare-ID-şekilli sayı öbekleri.
    # Yakınlık = karakter penceresi İÇİNDE OLMAK VEYA AYNI CÜMLEDE OLMAK.
    # (Yakınlık hesabı NORMALİZE koordinatlarda yapılır -- tutarlı olması
    # için kurum span'ları da normalize koordinatlardan alınmıştır.)
    if institution_spans:
        sentence_spans = _sentence_spans(norm)
        for m in _BARE_RESIDUAL_PATTERN.finditer(norm):
            ms, me = m.start(), m.end()
            near = any(
                (
                    (inst_s - _CONTEXT_WINDOW) <= me and ms <= (inst_e + _CONTEXT_WINDOW)
                )
                or _same_sentence(ms, inst_s, sentence_spans)
                for inst_s, inst_e in institution_spans
            )
            if near:
                add(ms, me, "patient_id:bare_residual_near_institution")

    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(norm):
            add(m.start(), m.end(), "date")

    return spans


def _merge_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Örtüşen/bitişik span'ları tek bir redaksiyon bloğuna birleştirir
    (örn. 'TCGA-06-5412' hem `patient_id:TCGA` hem `institution` olarak
    eşleşebilir -- iki ayrı `[REDACTED]` yerine TEK blok istenir)."""

    spans = sorted(spans, key=lambda t: (t[0], -t[1]))
    merged: list[tuple[int, int, str]] = []
    for s, e, cat in spans:
        if merged and s <= merged[-1][1]:
            prev_s, prev_e, prev_cat = merged[-1]
            merged[-1] = (prev_s, max(prev_e, e), prev_cat)
        else:
            merged.append((s, e, cat))
    return merged


def sanitize_free_text(text: str) -> tuple[str, dict[str, int]]:
    """Serbest bir metinden bilinen hasta-ID/kurum/tarih desenlerini
    kaldırır. Bu fonksiyon SAVUNMA KATMANI 2'dir (birincil savunma tip-
    düzeyi allowlist'tir, `PatientAttributeProfile`).

    POZİSYON-TABANLI (span-based) çalışır -- tüm adaylar ÖNCE orijinal
    metin üzerinde toplanır (`_collect_spans`), SONRA birleştirilip
    (`_merge_spans`) TEK geçişte redakte edilir. Bu, önceki `subn`-zinciri
    sürümünün "kurum tokenı ID'den önce silinirse geri kalan rakam açıkta
    kalır" sıra-bağımlılığı hatasını ORTADAN KALDIRIR.

    Döner: (temizlenmiş_metin, {desen_adı: kaldırılan_eşleşme_sayısı}).
    Kaldırılan DEĞERLER loglanmaz/döndürülmez (yalnız SAYI) -- aksi halde
    "temizleme raporu" kendisi bir sızıntı kanalı olurdu.
    """

    spans = _merge_spans(_collect_spans(text))

    counts: dict[str, int] = {}
    parts: list[str] = []
    last = 0
    for s, e, cat in spans:
        parts.append(text[last:s])
        parts.append(REDACTED_TOKEN)
        counts[cat] = counts.get(cat, 0) + 1
        last = e
    parts.append(text[last:])

    return "".join(parts), counts


# ---------------------------------------------------------------------------
# 2026-09-13 (G2) — BAĞIMSIZ İSKELET KAPISI.
#
# 2026-08-18'in dersi: kapı ile redaksiyon AYNI regex'e dayandığı için
# regex zayıfladığında İKİ savunma AYNI ANDA kör oldu. 2026-09-11 bunu
# yalnız `patient_context_text` yolu için çözmüştü (`assert_patient_
# context_has_no_digit_clusters`); LİTERATÜR metni yolunda kapı hâlâ
# redaksiyonla aynı desen setini paylaşıyordu.
#
# Bu kapı FARKLI BİR TESPİT MEKANİZMASI kullanır: metni normalize edip
# alfanümerik OLMAYAN HER ŞEYİ sıyırır ("iskelet") ve bu iskelette basit
# bir şekil arar. Ayraç sınıfı, tekrar sınırı, span mantığı veya cümle
# sınırı hesabındaki BİR HATA bu kapıyı KÖR EDEMEZ -- çünkü ayraçların
# hepsi zaten atılmıştır.
#
# ⚠️ DÜRÜSTLÜK: bu kapı token ENVANTERİNİ (TCGA/UPENN-GBM/UCSF-PDGM)
# `_PATIENT_ID_TOKEN_SPECS` ile PAYLAŞIR -- tam bağımsız değildir,
# EŞLEŞME MANTIĞI bağımsızdır. Yeni bir KAYNAK eklenirse iki katman da
# aynı anda güncellenir; ama bir AYRAÇ/FORMAT hatası yalnız birini vurur.
#
# ⚠️ KAPSAM DIŞI BIRAKILAN 2 SPEC (ölçülmüş yanlış-pozitif gerekçesiyle,
# 2026-09-13): `LUMIERE-Patient` ve `generic-source-numeric`. Ayraçlar
# sıyrıldığı için bunlar meşru metinde patlıyordu:
#   "in each patient 12 months"  -> iskelet "...PATIENT12MONTHS" -> eşleşir
#   "recurrent GBM 2019 cohort"  -> iskelet "...GBM2019..."      -> eşleşir
# Bu kapı ihlalde EXCEPTION fırlattığı (fail-closed) için yanlış pozitif
# = özetin tamamen ENGELLENMESİ demektir. İki spec, ayraç-çapalı enumere
# katmanda KAPSANMAYA DEVAM EDER; yalnız bu ek kapının dışındadır.
# ---------------------------------------------------------------------------
_SKELETON_SPEC_NAMES: frozenset[str] = frozenset({"TCGA", "UPENN-GBM", "UCSF-PDGM"})


def _upper_literal_only(token: str) -> str:
    """Token'ı SADECE düz bir literal ise büyük harfe çevirir.

    ⚠️ 2026-09-13 yakalanan HATA: `"".join(tokens).upper()` regex
    PARÇALARINI da büyütüyordu -- `\\d{2}` -> `\\D{2}`, yani "rakam"
    sınıfı "RAKAM OLMAYAN" sınıfına dönüyordu. Sonuç iki yönlü felaketti:
    `TCGA\\D{2}\\D{4}` HEM gerçek ID'yi ('TCGA065412') KAÇIRIYOR HEM de
    meşru metni ('TCGA cohort study') eşleştirip fail-closed kapıyı
    boşuna tetikliyordu. Yani "bağımsız kapı" sessizce İŞE YARAMAZ
    olacaktı -- 2026-08-18'in hatasının birebir tekrarı. Aşağıdaki
    `test_skeleton_patterns_are_not_inverted_by_uppercasing` bu hatayı
    kalıcı olarak nöbette tutar."""

    return token.upper() if token.isalnum() else token


def _build_skeleton_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    built: list[tuple[str, re.Pattern[str]]] = []
    for name, tokens in _PATIENT_ID_TOKEN_SPECS:
        if name not in _SKELETON_SPEC_NAMES:
            continue
        forward = "".join(_upper_literal_only(t) for t in tokens)
        reverse = "".join(_upper_literal_only(t) for t in reversed(tokens))
        # Sondaki `(?!\d)` / baştaki `(?<!\d)`: iskelette kelimeler bitişik
        # olduğu için "TCGA 2008, 206 hasta" -> "TCGA2008206..." gibi
        # diziler RAKAM SAYISI tutmadığında eşleşmemeli.
        built.append((name, re.compile(rf"{forward}(?!\d)")))
        built.append((f"{name}-reversed", re.compile(rf"(?<!\d){reverse}")))
    return tuple(built)


_SKELETON_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = _build_skeleton_patterns()
_NON_ALNUM_PATTERN = re.compile(r"[^0-9A-Za-z]+")


def identifier_skeleton(text: str) -> str:
    """Normalize edilmiş metinden alfanümerik olmayan HER karakteri
    sıyırıp büyük harfe çevirir. 'TCGA.06.5412', 'TCGA--06--5412',
    'TCGA​06​5412', 'ＴＣＧＡ－０６－５４１２' -> hepsi
    'TCGA065412'."""

    norm, _ = normalize_for_detection(text)
    return _NON_ALNUM_PATTERN.sub("", norm).upper()


def assert_no_identifiers_leaked(text: str) -> None:
    """Son kontrol kapısı -- metinde HÂLÂ bilinen bir ID/kurum deseni
    varsa `SanitizationViolationError` fırlatır. `pipeline/rag_llm.py`'nin
    TEK LLM-çağrı fonksiyonu bunu ATLAYAMAZ (bkz. modül dokstring'i).

    İKİ AYRI MEKANİZMA kullanır (2026-09-13 / G2):
      (a) enumere desenler (`_PATIENT_ID_PATTERNS`, `_INSTITUTION_PATTERN`)
          -- normalize metin üzerinde; redaksiyonla AYNI desenler.
      (b) İSKELET araması (`_SKELETON_PATTERNS`) -- ayraçların TAMAMI
          sıyrılmış metinde şekil araması; (a)'daki ayraç/tekrar/span
          mantığından BAĞIMSIZ. 2026-08-18'de "tek zayıflık iki savunmayı
          birden düşürdü" hatasının literatür-metni yolundaki karşılığı
          budur ve burada kapatılmıştır.

    ⚠️ (a) ve (b) token envanterini paylaşır (bkz. `_SKELETON_SPEC_NAMES`
    üstündeki dürüstlük notu): HİÇ GÖRÜLMEMİŞ bir KAYNAK adı için ikisi de
    kördür. O senaryoya karşı koruma yalnız `patient_context_text` yolunda
    (`assert_patient_context_has_no_digit_clusters`) vardır."""

    norm, _ = normalize_for_detection(text)

    for name, pattern in _PATIENT_ID_PATTERNS:
        if pattern.search(norm):
            raise SanitizationViolationError(
                f"Sanitize edilmiş bağlamda hâlâ bir hasta-ID deseni bulundu "
                f"({name}) -- LLM çağrısı ENGELLENDİ."
            )

    skeleton = identifier_skeleton(text)
    for name, pattern in _SKELETON_PATTERNS:
        if pattern.search(skeleton):
            raise SanitizationViolationError(
                f"Sanitize edilmiş bağlamda, ayraçlardan arındırılmış "
                f"iskelet aramasında bir hasta-ID şekli bulundu ({name}) -- "
                f"LLM çağrısı ENGELLENDİ."
            )

    if _INSTITUTION_PATTERN.search(norm):
        raise SanitizationViolationError(
            "Sanitize edilmiş bağlamda hâlâ bir kurum/kaynak adı bulundu -- "
            "LLM çağrısı ENGELLENDİ."
        )


# Reviewer'ın istediği BAĞIMSIZ/daha katı kapı: enumere edilmiş bir desen
# listesi DEĞİL -- "5+ karakter uzunluğunda, en az bir ayraçla bölünmüş
# rakam öbeği" gibi çok daha geniş bir kural. Bilerek SADECE hasta bağlamı
# metnine (`patient_context_text`) uygulanır, literatür metnine DEĞİL --
# gerekçe: `render_patient_context_text()` SABİT ŞABLONLUDUR (yalnız
# "NN yaşında", "KPS NN", "MGMT <bucket>", "IDH1 <bucket>" gibi TEK
# rakamlı/ayraçsız ifadeler üretir), bu yüzden burada YANLIŞ POZİTİF
# riski YOKTUR. Literatür metni ise SERBEST/üçüncü-taraf bilimsel metin
# olduğu için aynı geniş kural orada meşru içeriği (p-değerleri, doz
# aralıkları, örneklem büyüklükleri, %95 GA aralıkları) sistematik olarak
# yok ederdi -- bu YANLIŞ POZİTİF riski somut örnekle test edilip
# `tests/test_rag_sanitize.py`'de belgelenmiştir.
#
# 2026-09-11: ayraç karakter sınıfı `_SEP_CLASS`'taki (Unicode tire dahil)
# KAPSAMI paylaşır -- reviewer'ın 2026-08-19 bulgusu ("TCGA－06－5412"
# tam-genişlik tire yüzünden rakam dizisi "06" + "5412" diye bölünüyor,
# kalan "5412" tek başına min-5-karakter kuralını sağlamıyor) burada
# KAPANIR. ⚠️ Desen KODU hâlâ `sanitize_free_text`'in enumere listesinden
# BAĞIMSIZDIR (yalnız karakter kapsamı ortak) -- bağımsızlık tasarım
# kararı korunuyor, bkz. fonksiyon dokstring'i.
#
# 2026-09-13 (G2): karakter ENUMERASYONU tamamen BIRAKILDI. Girdi artık
# `normalize_for_detection`'dan geçtiği için bu desenin Unicode tire/
# görünmez karakter listesi TUTMASINA gerek yok -- normalizasyon onları
# zaten "-"/" " yapıyor veya düşürüyor. 2026-09-13 ölçümünde bu kapının
# NBSP (U+00A0), nokta, yumuşak tire (U+00AD) ve sıfır-genişlik boşluk
# (U+200B) ayraçlı formları KAÇIRDIĞI canlı gösterilmişti; kök neden
# "listeye hangi karakter eklendi" idi, normalizasyonla ORTADAN KALKTI.
# Nokta ayraç sınıfına EKLENDİ: bu kapı YALNIZ `patient_context_text`'e
# uygulandığı ve o metin sabit şablonlu (tam sayı) olduğu için burada
# yanlış pozitif riski yok (literatür metnine UYGULANMAZ -- aşağıdaki
# gerekçe ve `build_sanitized_context`).
_DIGIT_CLUSTER_PATTERN = re.compile(r"\d[\d\-_ .]{3,}\d")


def assert_patient_context_has_no_digit_clusters(text: str) -> None:
    """Hasta bağlamı metni için BAĞIMSIZ/daha katı son kontrol -- `sanitize_
    free_text`'in enumere ettiği format listesine BAĞLI DEĞİLDİR, bu yüzden
    henüz görülmemiş/gelecekte eklenecek bir ID formatını da (aynı zaafı
    PAYLAŞMADAN) yakalayabilir. `render_patient_context_text()`'in çıktısı
    hiçbir zaman böyle bir öbek İÇERMEMELİDİR -- içeriyorsa bu, birincil
    savunmanın (tip-düzeyi allowlist) BEKLENMEDİK şekilde ihlal edildiğinin
    kanıtıdır (örn. `PatientAttributeProfile`'a gelecekte eklenecek serbest
    bir alan).

    2026-09-13 (G2): kontrol NORMALİZE metin üzerinde yapılır -- görünmez
    karakter serpiştirerek ('06<ZWSP>5412') bu kapıyı atlatma yolu
    KAPATILDI."""

    norm, _ = normalize_for_detection(text)
    if _DIGIT_CLUSTER_PATTERN.search(norm):
        raise SanitizationViolationError(
            "Hasta bağlamı metninde beklenmeyen bir rakam öbeği bulundu "
            "(bağımsız/katı kapı) -- LLM çağrısı ENGELLENDİ. Bu normalde "
            "HİÇ tetiklenmemeli; tetiklendiyse PatientAttributeProfile/"
            "render_patient_context_text() gözden geçirilmeli."
        )


# ---------------------------------------------------------------------------
# 2) Tip-düzeyi allowlist (BİRİNCİL savunma)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatientAttributeProfile:
    """LLM bağlamına girmesine İZİN VERİLEN alanların TAMAMI budur.

    ⚠️ Bilinçli olarak `patient_id`, `scan_date`, `source_dataset`,
    `center` alanları YOKTUR -- bu tip bunları TAŞIYAMAZ (Python'da bir
    dataclass'a olmayan bir alanı atamak `TypeError`/`AttributeError`
    ile başarısız olur, "unutma" ile bu kural çiğnenemez).
    """

    age_years: int | None
    gender: str | None  # "Male" | "Female" | None
    kps_score: int | None
    mgmt_status_bucket: str  # "methylated" | "unmethylated" | "unknown"
    idh1_status_bucket: str  # "wildtype" | "mutant" | "unknown"
    risk_category: str | None  # örn. "orta" -- Cox risk skorundan türetilmiş kategori, HAM skor değil
    has_omics: bool = False
    molecular_subtype: str | None = None  # omics_profiles'tan, kimlik taşımaz


def render_patient_context_text(attrs: PatientAttributeProfile) -> str:
    """`PatientAttributeProfile`'dan LLM bağlamına eklenecek SABİT-ŞABLONLU
    metni üretir. Fonksiyon imzası SADECE bu tipi kabul eder -- serbest
    bir `**kwargs`/`dict` alacak şekilde GENİŞLETİLMEMELİDİR (aksi halde
    çağıran taraf yanlışlıkla `patient_id` sızdırabilir)."""

    parts: list[str] = []
    if attrs.age_years is not None:
        parts.append(f"{attrs.age_years} yaşında")
    if attrs.gender:
        parts.append({"Male": "erkek", "Female": "kadın"}.get(attrs.gender, attrs.gender))
    if attrs.kps_score is not None:
        parts.append(f"KPS {attrs.kps_score}")
    parts.append(f"MGMT {attrs.mgmt_status_bucket}")
    parts.append(f"IDH1 {attrs.idh1_status_bucket}")
    if attrs.risk_category:
        parts.append(f"risk kategorisi: {attrs.risk_category}")
    if attrs.has_omics and attrs.molecular_subtype:
        parts.append(f"moleküler alt tip: {attrs.molecular_subtype}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# 3) Sanitize edilmiş LLM bağlamı -- yalnız `build_sanitized_context()`
#    ile üretilir, yapıcı doğrudan "güvenli" varsayılmaz.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SanitizedSnippet:
    pmid: str
    title: str
    text: str


@dataclass(frozen=True)
class SanitizedLLMContext:
    """`pipeline/rag_llm.py`'nin TEK LLM çağrı fonksiyonunun kabul ettiği
    TEK parametre tipi. Serbest string/dict bu fonksiyona VERİLEMEZ --
    yapısal zorunluluk budur."""

    patient_context_text: str
    literature_snippets: tuple[SanitizedSnippet, ...] = field(default_factory=tuple)


def build_sanitized_context(
    attrs: PatientAttributeProfile,
    literature_chunks: Sequence[tuple[str, str, str]] = (),
) -> SanitizedLLMContext:
    """TEK giriş noktası: hasta öznitelik profili + (pmid, title, text)
    üçlülerinden `SanitizedLLMContext` üretir.

    Hasta bağlamı metni: `render_patient_context_text` (katman 1, gerçek
    garanti) + `assert_no_identifiers_leaked` + `assert_patient_context_
    has_no_digit_clusters` (katman 2, BAĞIMSIZ/katı kapı -- yalnız bu
    metin için, gerekçe yukarıda).

    Literatür parçaları: `sanitize_free_text` (enumere edilmiş, span-
    tabanlı redaksiyon) + `assert_no_identifiers_leaked` -- burada geniş
    rakam-öbeği kapısı KASITLI OLARAK UYGULANMAZ (meşru bilimsel içeriği
    yok etme riski, yukarıda belgelendi)."""

    patient_text = render_patient_context_text(attrs)
    assert_no_identifiers_leaked(patient_text)
    assert_patient_context_has_no_digit_clusters(patient_text)

    snippets: list[SanitizedSnippet] = []
    for pmid, title, text in literature_chunks:
        clean_title, _ = sanitize_free_text(title)
        clean_text, _ = sanitize_free_text(text)
        assert_no_identifiers_leaked(clean_title)
        assert_no_identifiers_leaked(clean_text)
        snippets.append(SanitizedSnippet(pmid=pmid, title=clean_title, text=clean_text))

    return SanitizedLLMContext(
        patient_context_text=patient_text,
        literature_snippets=tuple(snippets),
    )
