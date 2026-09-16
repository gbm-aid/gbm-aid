-- =============================================================================
-- ✅ UYGULANDI (2026-09-13, Barış onayı, db-agent-D). Barış'ın şartı ("ADIM
-- 1/2/3'ün üç ayrı BEGIN...ROLLBACK bloğu değil, TEK transaction + guard-
-- assert zinciriyle çalışması") bir Python orkestrasyon script'iyle
-- karşılandı — ALTER + UPDATE1 + UPDATE2 + doğrulama TEK psycopg2
-- transaction'ında, her adımda rowcount/agregasyon ASSERT edilerek,
-- hepsi geçerse tek COMMIT ile yazıldı (herhangi biri başarısız olsaydı
-- TÜMÜ ROLLBACK olurdu — kısmi/elle-parça-parça çalıştırma riski KAPANDI).
-- Aşağıdaki ADIM 1/2/3 blokları artık yalnız TARİHSEL/açıklayıcı metindir
-- (orijinal taslak, dry-run→onay sürecinde yazıldı) — gerçek çalıştırma bu
-- üç bloğu TEK transaction'da birleştirdi, ayrı ayrı ELLE çalıştırılmadı.
-- Canlı doğrulama (2026-09-13, bağımsız readonly bağlantı, apply SONRASI):
--   select harmonization_status, count(*) from mr_scans group by 1 order by 1;
--     raw=191, zscore_pooled_legacy=3808, zscore_t1ce_c32=1530 (zscore/n4/combat=0)
--   select count(*) from mr_scans;  -> 5529 (değişmedi)
--   CHECK constraint canlı tanımı: ARRAY['raw','n4','zscore','combat',
--     'zscore_pooled_legacy','zscore_t1ce_c32'] -- beklenenle birebir.
-- Companion güncelleme: tools/data_integrity_check.py::DOMAIN_ALLOWED
-- (satır ~559-568) AYNI işte güncellendi. KNOWN_STALE_HARMONIZATION_GAPS
-- (K3 registry) BİLEREK DOKUNULMADI — bkz. decisions/2026-09-11-
-- harmonization-status-kalici-cozum.md DURUM NOTU (2026-09-14'te dosya
-- adından "-TASLAK" eki kaldırıldı, eski ad artık YOK, bkz. K23; bu
-- migrasyon registry'yi BEKLENDİĞİ ÜZERE bayatlattı; registry'nin
-- kapatılması AYRI bir onaylı iş, çünkü 567 satırlık dedike bir test
-- dosyasına bağlı).
-- 53 scan_id'lik "raw idi, şimdi zscore_t1ce_c32" listesi (rollback için
-- gereken yedek, orijinal taslağın rollback notunda istenen) canlı DB'den
-- migrasyon ÖNCESİ okunup KNOWN_STALE_HARMONIZATION_GAPS[0].expected_scan_ids
-- ile BİREBİR (53/53) çapraz doğrulandı — log/2026-09-13.md'de kayıtlı.
-- Detay: log/2026-09-13.md.
-- =============================================================================

-- =============================================================================
-- (ORİJİNAL TASLAK METNİ, 2026-09-11 — artık "UYGULANMADI" ifadesi GEÇERSİZ,
-- yukarıdaki güncel not otoritedir)
-- TASLAK — UYGULANMADI. Barış onayı ZORUNLU (şema değişikliği + toplu UPDATE,
-- CLAUDE.md KESİN SINIRLAR #4). Bu dosyada hiçbir DDL/UPDATE çalıştırılmadı.
-- Karar dosyası: decisions/2026-09-11-harmonization-status-kalici-cozum-TASLAK.md
-- [2026-09-15 DÜZELTME, K23]: bu dosya 2026-09-14'te
-- decisions/2026-09-11-harmonization-status-kalici-cozum.md olarak
-- yeniden adlandırıldı ("-TASLAK" eki kaldırıldı) -- eski ad artık YOK.
-- Yukarıdaki satır orijinal taslak metni olduğu için AYNEN bırakıldı,
-- bu atıf notu yalnızca DÜZELTME olarak eklendi.
--
-- Kaynak karar: decisions/2026-08-14-harmonization-status-bayragi-c32-ile-
-- birlikte-guncellenecek.md — "(c) C32 koşusuyla birlikte" seçildi, B2'de
-- şu şartlarla uygulanacaktı: (1) değer ayrımı zorunlu — önerilen
-- `zscore_pooled_legacy` (eski) / `zscore_t1ce_c32` (yeni), (2) şema
-- kontrolü önce (CHECK constraint varsa önce genişletilmeli), (3) dry-run→
-- onay→apply, (4) yazımdan sonra bağımsız SELECT doğrulaması.
-- B2 (2026-08-14/15) bu bayrağı GÜNCELLEMEDİ — o günden bu yana UCSF de
-- eklendi (2026-08-18/19 civarı) ve AYNI ambiguous 'zscore' değeriyle
-- yazıldı. Bu görev o eksik adımı 2026-09-11'de KAPATIYOR (taslak olarak).
--
-- CANLI ÖLÇÜM (2026-09-11, readonly SELECT, db-agent):
--   select ds.source_name, ms.harmonization_status, count(*)
--   from mr_scans ms join patients p on p.patient_id=ms.patient_id
--   join dataset_sources ds on ds.source_id=p.source_id
--   group by 1,2 order by 1,2;
--
--     LUMIERE     raw     244
--     LUMIERE     zscore 2152
--     TCGA-GBM    zscore  154
--     UCSF-PDGM   zscore  295
--     UPenn-GBM   zscore 2684
--   TOPLAM mr_scans: 5529
--
--   Crosstab (harmonization_status x "bu scan_id'nin C32 sözleşmeli
--   [UPenn/LUMIERE/UCSF/TCGA -C32] bir radiomics satırı var mı"):
--     raw    + C32-radyomik-YOK   -> 191 satır
--     raw    + C32-radyomik-VAR   ->  53 satır  (== K3 known-stale-gap
--                                                 registry'nin LUMIERE-
--                                                 PyRadiomics-107-C32 için
--                                                 beklediği expected_n_scans=53,
--                                                 data_integrity_check.py
--                                                 satır 918-930 ile BİREBİR
--                                                 eşleşiyor — bağımsız
--                                                 doğrulama)
--     zscore + C32-radyomik-YOK   -> 3808 satır
--     zscore + C32-radyomik-VAR   -> 1477 satır
--   TOPLAM: 191+53+3808+1477 = 5529 (mr_scans toplam sayısıyla TAM eşleşiyor
--   — partition tüketici/kapsayıcı, kayıp/çakışma yok).
--
--   C32-radyomik-VAR toplamı (53+1477=1530) LOCKED_RADIOMICS_C32'nin
--   dört aracının distinct scan_id toplamıyla (611+585+295+39=1530)
--   BİREBİR eşleşiyor (data_integrity_check.py satır 123-128).
--
-- 2026-08-15 GÜNLÜK RAPORDA GEÇEN "1182/53" SAYISI GÜNCEL DEĞİL — o tarihte
-- UCSF henüz DB'de yoktu ve B1/B2'nin yalnız TCGA+UPenn ayağı bitmişti;
-- bugünkü tam ölçüm (UCSF dahil, tüm kohort) yukarıdaki 191/53/3808/1477
-- dörtlüsüdür. Eski "1182/53" rakamı raporlarda KULLANILMAMALI.
--
-- YORUM — 53 'raw'+C32 satırı NEDEN 'zscore_t1ce_c32' OLMALI (yalnız 'raw'
-- KALMAMALI):
-- Bu 53 scan_id, `decisions/2026-08-09-lumiere-resampling-entegrasyonu-
-- onaylandi.md` kararıyla devreye giren resampling-kurtarma yoluyla işlendi
-- — orijinal toplu harmonizasyon işi (`run_harmonization_cohort.py`,
-- 2026-08-12) bunları "skipped_not_evaluable" (raw) bıraktı, AMA C32
-- PyRadiomics çıkarım script'i (`run_pyradiomics_lumiere.py`) kendi N4+
-- T1ce-Z-score ön işlemesini TÜM 580 LUMIERE T1ce görüntüsüne (bu 53 dahil)
-- UYGULADIKTAN sonra çalıştı (B1 görevinde "Aşama 1: 580 N4 dosyası
-- cache'te hazır" doğrulandı) — resampling burada YALNIZ maske-görüntü
-- geometri uyuşmazlığını (voxel-spacing) çözdü, N4/Z-score'un KENDİSİNİ
-- değil. Yani bu 53 scan'in ALTINDAKİ GÖRÜNTÜ C32 sözleşmesiyle (N4+T1ce-
-- Z-score) FİİLEN harmonize edildi; `mr_scans.harmonization_status='raw'`
-- kalması SADECE eski toplu-iş bayrağının GÜNCELLENMEMİŞ olmasından
-- kaynaklanıyor, görüntünün gerçek durumundan değil.
-- ⚠️ AÇIK VARSAYIM (Barış'a bulgu, bu taslakta doğrulanmadı): yukarıdaki
-- akıl yürütme "resampling sadece maskeyi hizalar, görüntünün N4/Z-score
-- durumunu değiştirmez" iddiasına dayanıyor — bu `pipeline/resampling.py`
-- kod okumasıyla TEYİT EDİLMEDİ (kapsam dışı bırakıldı, zaman kısıtı).
-- Onay öncesi bu varsayımın kod okumasıyla doğrulanması ÖNERİLİR.
--
-- =============================================================================
-- ADIM 0: ŞEMA KONTROLÜ (mevcut CHECK constraint) -- SALT-OKUNUR, DOĞRULANDI
-- =============================================================================
-- Canlı ölçüm: mr_scans_harmonization_status_check ->
--   CHECK ((harmonization_status = ANY (ARRAY['raw','n4','zscore','combat'])))
-- Yeni değerler bu listede YOK -> ADIM 1 (ALTER) UPDATE'lerden ÖNCE ZORUNLU.

-- =============================================================================
-- ADIM 1: ŞEMA DEĞİŞİKLİĞİ (ONAY ZORUNLU -- CLAUDE.md KESİN SINIRLAR #4)
-- Saf GENİŞLETME (superset) -- eski 4 değer KALIYOR (ileride 'n4'/'combat'
-- yazan başka bir pipeline kırılmasın diye), yalnız 2 yeni değer EKLENİYOR.
-- =============================================================================
BEGIN;

ALTER TABLE mr_scans DROP CONSTRAINT mr_scans_harmonization_status_check;
ALTER TABLE mr_scans ADD CONSTRAINT mr_scans_harmonization_status_check
    CHECK (harmonization_status = ANY (ARRAY[
        'raw'::text, 'n4'::text, 'zscore'::text, 'combat'::text,
        'zscore_pooled_legacy'::text, 'zscore_t1ce_c32'::text
    ]));

-- Dry-run doğrulama (aynı transaction içinde, henüz COMMIT edilmedi):
-- select conname, pg_get_constraintdef(oid) from pg_constraint
-- where conrelid = 'mr_scans'::regclass and conname = 'mr_scans_harmonization_status_check';

ROLLBACK;  -- <-- dry-run modu. Onay sonrası bu satır silinip COMMIT; yapılır.

-- =============================================================================
-- ADIM 2: UPDATE 1 -- C32 sözleşmesiyle FİİLEN işlenmiş scan'ler
-- Guard'lı WHERE: yalnız (raw veya zscore) durumundaki, C32 radiomics'i
-- olan scan_id'ler. Beklenen etkilenen satır: 1530 (53 raw + 1477 zscore).
-- =============================================================================
BEGIN;

WITH c32_scans AS (
    SELECT DISTINCT scan_id FROM radiomics
    WHERE segmentation_tool IN (
        'UPenn-PyRadiomics-107-C32', 'LUMIERE-PyRadiomics-107-C32',
        'UCSF-PDGM-PyRadiomics-107-C32', 'TCGA-ground-truth-C32'
    )
)
UPDATE mr_scans
SET harmonization_status = 'zscore_t1ce_c32'
WHERE scan_id IN (SELECT scan_id FROM c32_scans)
  AND harmonization_status IN ('raw', 'zscore');

-- GUARD: beklenen rowcount == 1530. Uygulama script'i bunu ASSERT etmeli,
-- eşleşmezse COMMIT ETMEDEN dur (bkz. sql-dogrulama-protokolu deseni).

ROLLBACK;  -- <-- dry-run modu.

-- =============================================================================
-- ADIM 3: UPDATE 2 -- eski (havuzlanmış, T1ce-özel OLMAYAN) Z-score kalanları
-- ADIM 2'den SONRA çalıştırılmalı (yalnız hâlâ 'zscore' olan satırları hedefler
-- -- ADIM 2 C32'li olanları zaten 'zscore_t1ce_c32'ye çevirdiği için buradaki
-- WHERE 'zscore' otomatik olarak "C32'siz kalanlar" anlamına gelir, ayrı bir
-- NOT IN alt sorgusuna GEREK YOK).
-- Beklenen etkilenen satır: 3808.
-- =============================================================================
BEGIN;

UPDATE mr_scans
SET harmonization_status = 'zscore_pooled_legacy'
WHERE harmonization_status = 'zscore';

-- GUARD: beklenen rowcount == 3808.

ROLLBACK;  -- <-- dry-run modu.

-- =============================================================================
-- DOKUNULMAYAN: 191 satır ('raw' + C32-radyomik-YOK) -- LUMIERE'in kalıcı/
-- bilinçli veri eksikliği (2026-08-12 kararı kapsamında, bu görevin
-- kapsamı DIŞINDA), 'raw' olarak KALIR. Değişmez.
-- =============================================================================

-- =============================================================================
-- DOĞRULAMA (her iki UPDATE --apply edildikten SONRA, AYRI bir readonly
-- bağlantıdan çalıştırılacak) -----------------------------------------------
-- select harmonization_status, count(*) from mr_scans group by 1 order by 1;
--   beklenen: raw=191, zscore_t1ce_c32=1530, zscore_pooled_legacy=3808,
--             (zscore=0, n4=0, combat=0)
-- select count(*) from mr_scans;  -- beklenen: 5529 (değişmedi)
-- =============================================================================

-- =============================================================================
-- ROLLBACK (geri alma, apply SONRASI gerekirse) -----------------------------
-- BEGIN;
-- UPDATE mr_scans SET harmonization_status = 'zscore'
--   WHERE harmonization_status IN ('zscore_t1ce_c32', 'zscore_pooled_legacy');
-- ALTER TABLE mr_scans DROP CONSTRAINT mr_scans_harmonization_status_check;
-- ALTER TABLE mr_scans ADD CONSTRAINT mr_scans_harmonization_status_check
--     CHECK (harmonization_status = ANY (ARRAY['raw','n4','zscore','combat']));
-- COMMIT;
-- (Not: bu rollback 'raw' <-> 'zscore_t1ce_c32' ayrımını geri getirmez --
-- 53 satırın hangisinin orijinalde 'raw' olduğu bilgisini KAYBEDER. Onay
-- öncesi bu 53 scan_id'nin listesi ayrıca (ör. bu SQL dosyasının bir
-- yorum satırında veya scratchpad'de) yedeklenmeli.)
-- =============================================================================

-- =============================================================================
-- BAĞIMLI KOD DEĞİŞİKLİĞİ (bu SQL taslağının kapsamı DIŞINDA, AYRI onaylı
-- bir db-agent/imaging-agent görevi gerektirir -- BU GÖREVDE DOKUNULMADI):
-- `gbm-aid mert/tools/data_integrity_check.py`:
--   1. `DOMAIN_ALLOWED["mr_scans.harmonization_status"]` (satır 512) şu an
--      {"raw","n4","zscore","combat"} -- migrasyon SONRASI bu set
--      'zscore_pooled_legacy'/'zscore_t1ce_c32' içermezse
--      `check_domain_values()` TÜM 5338 migrasyona uğramış satırı YANLIŞ
--      pozitif CRITICAL olarak işaretler. Migrasyonla AYNI PR/onayda
--      güncellenmeli.
--   2. `check_c32_harmonization_status_gap()` (satır 1306-1338) VE
--      `KNOWN_STALE_HARMONIZATION_GAPS` registry'si (satır ~900+) YALNIZ
--      `m.harmonization_status = 'raw' AND segmentation_tool IN C32-listesi`
--      anomalisini arıyor. Migrasyon sonrası bu WHERE koşulu artık HİÇBİR
--      satırla eşleşmeyecek (53 satır 'zscore_t1ce_c32'ye taşındığı için) --
--      yani bu kontrol fonksiyonel olarak KENDİLİĞİNDEN SADELEŞİR (K3
--      istisnasının varlık nedeni ortadan kalkar, "kural yerine yapı" --
--      K13 dersiyle AYNI desen). Registry'nin TAMAMEN silinip
--      silinmeyeceği (yoksa "geçmiş kayıt" olarak mı bırakılacağı) AYRI
--      bir karar -- bu taslak SİLMİYOR, yalnız etkisinin sıfırlanacağını
--      not ediyor.
-- =============================================================================
