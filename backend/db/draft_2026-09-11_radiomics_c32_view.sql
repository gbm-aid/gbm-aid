-- =============================================================================
-- ✅ UYGULANDI (2026-09-13, Barış onayı, db-agent-D). CREATE VIEW + COMMENT
-- gerçekten çalıştırıldı (dry-run→apply→canlı SELECT doğrulaması ile).
-- Canlı doğrulama (2026-09-13, bağımsız readonly bağlantı):
--   select count(*) from radiomics_c32;               -> 7528 (beklenenle birebir)
--   select count(distinct scan_id) from radiomics_c32; -> 1530 (beklenenle birebir)
--   4 araç grup-sayımı da dry-run ile BİREBİR eşleşti (aşağıdaki eski not).
-- Detay: log/2026-09-13.md, decisions/2026-09-11-radiomics-c32-view.md
-- (DURUM NOTU eklendi; 2026-09-14'te dosya adından "-TASLAK" eki
-- kaldırıldı, eski ad `...-radiomics-c32-view-TASLAK.md` artık YOK,
-- bkz. K23). Aşağıdaki metin, uygulama ÖNCESİ yazılan orijinal
-- taslak olarak DEĞİŞTİRİLMEDEN bırakıldı (tarihsel doğruluk) --
-- İSTİSNA: içindeki bu dosya adı atıfları K23 kapsamında 2026-09-15'te
-- güncel dosya adına düzeltildi (bkz. aşağıdaki [2026-09-15] notları),
-- kalan metin/karar gövdesi DOKUNULMADI.
-- =============================================================================

-- =============================================================================
-- (ORİJİNAL TASLAK METNİ, 2026-09-11 — artık "UYGULANMADI" ifadesi GEÇERSİZ,
-- yukarıdaki güncel not otoritedir)
-- TASLAK — UYGULANMADI. Barış onayı ZORUNLU. DRY-RUN sonuçları bu dosyanın
-- başında canlı ölçüm olarak verilmiştir; hiçbir CREATE/DROP burada
-- çalıştırılmamıştır (bu bir metin taslağıdır).
--
-- K13 — BEKLEYEN-KARARLAR.md: `radiomics_c32` VIEW'ı — yanlış nesli yapısal
-- olarak imkânsızlaştır. Barış kaydetme talimatı: 2026-08-15.
-- Karar taslağı: decisions/2026-09-11-radiomics-c32-view-TASLAK.md
-- [2026-09-15 DÜZELTME, K23]: bu dosya 2026-09-14'te
-- decisions/2026-09-11-radiomics-c32-view.md olarak yeniden adlandırıldı
-- ("-TASLAK" eki kaldırıldı) -- eski ad artık YOK. Yukarıdaki satır
-- orijinal taslak metni olduğu için AYNEN bırakıldı, bu atıf notu
-- yalnızca DÜZELTME olarak eklendi.
--
-- CANLI ÖLÇÜM (2026-09-11, readonly SELECT, db-agent):
--   select segmentation_tool, count(*), count(distinct scan_id)
--   from radiomics group by 1 order by 1;
--
--   Nesil 1 (kaynak-sağlayıcı precompute):
--     CaPTk-automatic                7268 satır / 2444 scan
--     CaPTk-corrected                2764 satır /  928 scan
--     DeepBraTumIA                   7188 satır / 2396 scan
--     HD-GLIO-AUTO                   4792 satır / 2396 scan
--   Nesil 2 (A-yöntemi, binWidth=25, ESKİ/baseline — SİLİNMEDİ):
--     UPenn-PyRadiomics-107           1833 satır /  611 scan
--     LUMIERE-PyRadiomics-107         1751 satır /  585 scan
--     TCGA-ground-truth                 78 satır /   39 scan
--   Nesil 3 (C32 — GEÇERLİ, bu VIEW'ın kapsamı):
--     UPenn-PyRadiomics-107-C32       3055 satır /  611 scan
--     LUMIERE-PyRadiomics-107-C32     2920 satır /  585 scan
--     UCSF-PDGM-PyRadiomics-107-C32   1475 satır /  295 scan
--     TCGA-ground-truth-C32             78 satır /   39 scan
--   TOPLAM (Nesil 3): 7528 satır / 1530 benzersiz scan_id
--
-- UCSF DAHİL Mİ? — K13 yazıldığında (2026-08-15) UCSF-PDGM aracı DB'de
-- YOKTU (UCSF indirmesi/yazımı sonradan tamamlandı). Bugünkü ölçüm UCSF'in
-- ZATEN `gbm-aid mert/tools/data_integrity_check.py`'nin `LOCKED_RADIOMICS_C32`
-- kapalı listesinde (satır 123-128) olduğunu gösteriyor:
--     "UCSF-PDGM-PyRadiomics-107-C32": (1475, 295)
-- yani K13'ün DIŞINDA ama sonradan yazılan bağımsız bir kilitli-sayı
-- registry'si UCSF'i ZATEN nesil-3 üyesi kabul etmiş durumda. Bu VIEW'ı
-- UCSF'siz kapatmak, mevcut registry ile YENİ bir tutarsızlık yaratır.
-- Ayrıca CLAUDE.md K13/C32 sözleşmesi ("N4 + T1ce-özel Z-score + binCount=32
-- + tüm parametreler açık") UCSF'e AYNEN uygulanıyor (2026-08-18 kararı,
-- yalnız ComBat feature-level düzeltmesi ve ComBat fit'i UCSF'i dışlıyor —
-- bu VIEW ComBat'la ilgili değil, yalnız "hangi radyomik satırı C32
-- sözleşmesiyle üretildi" sorusuna cevap veriyor). SONUÇ: UCSF DAHİL
-- EDİLMELİ — dört aracın hepsi aynı çıkarım sözleşmesini (C32) paylaşıyor,
-- ComBat/Cox eğitim kapsamı ayrı bir sorudur ve bu VIEW'ın sorumluluğunda
-- değildir (WT-only/ComBat/eğitim-havuzu filtreleri tüketici tarafında,
-- ör. Cox eğitim sorgusunda, ayrıca uygulanır).
--
-- ÖNERİ: kapalı `IN (...)` listesi (LIKE DEĞİL — K13/K3-M1 dersi: bir
-- `-C32-v2` gibi yeniden adlandırılmış varyant LIKE ile sessizce kaçar).
-- Bu liste `data_integrity_check.py`'deki `LOCKED_RADIOMICS_C32` anahtar
-- kümesiyle BİREBİR aynı tutulmalı (elle senkron — bu VIEW SQL dosyası bir
-- Python sabiti okuyamaz; ileride ikisi diverge ederse bu bir bulgu olarak
-- işaretlenmeli, ör. bir test/CI kontrolü ile).
-- =============================================================================

-- --- ADIM 1: VIEW oluştur (Barış onayından SONRA çalıştırılacak) -----------

CREATE VIEW radiomics_c32 AS
SELECT
    radiomics_id,
    scan_id,
    segmentation_tool,
    tumor_region,
    tumor_volume_mm3,
    surface_area,
    entropy,
    contrast,
    shape_features,
    first_order_features,
    texture_features,
    feature_source
FROM radiomics
WHERE segmentation_tool IN (
    'UPenn-PyRadiomics-107-C32',
    'LUMIERE-PyRadiomics-107-C32',
    'UCSF-PDGM-PyRadiomics-107-C32',
    'TCGA-ground-truth-C32'
);

-- ⚠️ [2026-09-15 NOT, K23, modeling-agent-Z5]: bu COMMENT ON VIEW
-- metni 2026-09-13'te GERÇEKTEN çalıştırıldı (yukarıdaki "✅ UYGULANDI"
-- notuna bkz.) ve canlı `radiomics_c32` VIEW'ının COMMENT'i hâlâ eski
-- dosya adını (`...-radiomics-c32-view-TASLAK.md`) taşıyor. Aşağıdaki
-- metin YALNIZ bu dosyadaki atfı günceller -- canlı DB'ye HİÇBİR
-- DDL/UPDATE ÇALIŞTIRILMADI (Barış'ın bilinçli kararı, K23'ün COMMENT
-- ayağı BEKLETİLDİ). Yani şu an dosya ile canlı DB'deki COMMENT metni
-- BİRBİRİNDEN FARKLI -- canlı COMMENT güncellenmek istenirse bu metin
-- AYRI bir onaylı DDL işiyle yeniden çalıştırılmalı.
COMMENT ON VIEW radiomics_c32 IS
    'K13 (BEKLEYEN-KARARLAR.md) — yalnız binCount=32/N4/T1ce-Z-score '
    'sözleşmesiyle (2026-08-13 kararı) üretilmiş radiomics satırlarını '
    'gösterir. Kapalı IN(...) listesi data_integrity_check.py '
    'LOCKED_RADIOMICS_C32 ile elle senkron tutulmalıdır. '
    'Kaynak: decisions/2026-09-11-radiomics-c32-view.md';

-- --- DOĞRULAMA (VIEW oluşturulduktan SONRA çalıştırılacak, readonly) -------
-- select count(*) from radiomics_c32;                          -- beklenen: 7528
-- select count(distinct scan_id) from radiomics_c32;           -- beklenen: 1530
-- select segmentation_tool, count(*) from radiomics_c32 group by 1 order by 1;

-- --- ROLLBACK (geri alma, tek komut, veri kaybı YOK — VIEW taban tabloya
--     dokunmaz) -----------------------------------------------------------
-- DROP VIEW IF EXISTS radiomics_c32;

-- --- HANGİ KOD/AJANLAR BU VIEW'A GEÇMELİ (öneri, uygulama AYRI onaylı iş) --
-- 1. gbm-aid mert/pipeline/growth_simulation.py:fetch_lumiere_wt_volume_series()
--    — şu an `FROM radiomics r WHERE r.segmentation_tool = %s` (parametrik,
--    tek aracı seçiyor); VIEW'a geçerse `FROM radiomics_c32 r` olur ve
--    segmentation_tool filtresi GEREKSİZ hâle gelir YA DA ek güvence olarak
--    kalabilir (tercih Mert'e/koordinatöre bırakılmalı — davranış değişmez,
--    yalnız yanlış-nesil satır sızma riski yapısal olarak kapanır).
-- 2. gbm-aid mert/tools/rebuild_faiss_indexes.py — FAISS vektör inşası C32
--    satırlarını okuyorsa (kod incelenmeli, bu taslağın kapsamı DIŞINDA)
--    aynı mantıkla VIEW'a geçebilir.
-- 3. gbm-aid mert/tools/train_cox_week3.py / pipeline/cox_model.py — Cox
--    eğitim sorguları zaten `LOCKED_RADIOMICS_C32`/`IN(...)` desenini
--    kod-seviyesinde taşıyor olabilir (kontrol edilmeli, DOKUNULMADI —
--    modeling-agent aynı anda bu dosyalarda ÇALIŞIYOR, bkz.
--    AKTIF-GOREVLER.md).
-- 4. gbm-aid mert/tools/data_integrity_check.py — VIEW'a geçmemeli/geçmesi
--    ZORUNLU DEĞİL: bu script'in tüm amacı nesiller ARASI tutarsızlığı
--    (ör. eski `UPenn-PyRadiomics-107` satırlarının hâlâ orada durup
--    durmadığını) denetlemek, bu yüzden TABAN TABLOYU (`radiomics`) okumaya
--    devam etmeli — VIEW'a geçerse kendi denetleme amacını kısmen kaybeder.
-- Not: yukarıdaki liste TAM DEĞİL — `select ... from radiomics` deseni için
-- ayrı bir kod taraması (grep) UYGULAMA ADIMINDA yapılmalı, bu taslak
-- yalnız bilinen/öne çıkan tüketicileri listeler.
