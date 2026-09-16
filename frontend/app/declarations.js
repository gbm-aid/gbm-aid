/* ===========================================================================
   GBM.decl — ZORUNLU BEYANLAR ve yapısal sabitler.

   İKİ TÜR İÇERİK VARDIR, KARIŞTIRMA:

   (A) `TEXTS` — `ZORUNLU-BEYANLAR.md` içindeki **"kullanıma hazır cümle"**
       bloklarından BİREBİR alınmış beyan metinleri. Her girdinin `src`
       alanı hangi maddeden geldiğini söyler. Değeri `null` olan girdi
       ekranda görünür bir `[[BEYAN: anahtar]]` yer tutucusu olarak çizilir
       — metin UYDURULMAZ.

       🔴 ALIM KURALI (2026-09-16): `ZORUNLU-BEYANLAR.md` içinde
       `⬇️ AŞAĞISI TARİHSEL — rapora girmez` işaretli bloklar ASLA
       alınmamıştır. A1/A3/A9/D2/D3/D4/D6/D7 maddelerinde gövdenin İLK
       alıntı bloğu bayattır; bu dosyadaki metinler her seferinde
       `✅ KULLANIMA HAZIR GÜNCEL CÜMLE` kutusundan alınmıştır.

   (B) `LOCKED`  — CLAUDE.md "KRİTİK MİMARİ KURALLAR" bölümünde KİLİTLİ olan
       yapısal sabitler (kol adları, eğitim havuzu büyüklükleri, rol
       tanımları). Bunlar API'den GELMEZ, hasta verisi DEĞİLDİR; projenin
       kayıtlı kararlarıdır. Model performans SAYILARI (C-index, CI, iç CV)
       bu listede BİLİNÇLİ OLARAK YOKTUR — onlar `GET /model_performance`
       ucundan gelir.
   =========================================================================== */
(function (global) {
  'use strict';

  var GBM = global.GBM = global.GBM || {};

  /* ---------------------------------------------------------------- (A) */
  /*
     Değer biçimi:
       { src: 'A9 · ZORUNLU-BEYANLAR.md', p: ['paragraf', 'paragraf', ...] }
     `p` dizisi null ise metin HENÜZ YOK demektir → yer tutucu çizilir.
  */
  var TEXTS = {

    /* --- Kalıcı üst şerit (her sayfada görünür) ------------------------- */
    'arastirma-seridi': {
      src: 'A10 + D2 + D4 (shadow kutusu)',
      p: [
        'Bu bir araştırma prototipidir; klinik karar için kullanılamaz. Sistem tanı koymaz. ' +
        'Model bir göreli risk SIRALAMASI üretir; olasılık veya beklenen sağkalım süresi üretmez. ' +
        'Nihai Cox modeli model kaydında (model_registry) "shadow" statüsündedir — "shadow" burada ' +
        '"hesaplanmıyor" değil, "klinik karara esas alınmaz" demektir.'
      ]
    },

    /* --- A10 — footer ve Sınırlar sekmesi ------------------------------ */
    'klinik-kullanim-yok': {
      src: 'A10 · ZORUNLU-BEYANLAR.md',
      p: [
        'Model bir göreli risk sıralaması üretir; olasılık veya beklenen sağkalım süresi üretmez. ' +
        '"Bu hasta 14 ay yaşar" biçiminde bir cümle kurulamaz; kurulabilecek cümle ' +
        '"A hastası B hastasından daha yüksek riskli" biçimindedir.',
        'Yapısal nedeni: UPenn-GBM\'de survival_days ameliyattan, UCSF-PDGM\'de OS tanıdan başlamaktadır. ' +
        'Zaman sıfırları farklı olduğu için mutlak kalibrasyon iddiası kurulamaz.'
      ]
    },

    /* --- B3 — harici test kohortunda seçim yanlılığı -------------------- */
    'secim-yanliligi': {
      src: 'B3 (+ 2026-09-14 ekleme kutusu) · ZORUNLU-BEYANLAR.md',
      p: [
        'UCSF-PDGM kohortunun tamamı indirilememiştir; indirme %80\'de kesilmiş ve elimizdeki dilim ' +
        'koleksiyonun ID 118–541 aralığıdır (ID 4–116 arası 94 hasta indirilemedi). Bu alt küme, ' +
        'rezeksiyon genişliği dağılımında tam kohorttan anlamlı biçimde farklıdır ' +
        '(GTR %64,8 vs %39,4, p<0,001). Yaş, cinsiyet, sağkalım ve olay oranında fark saptanmamıştır.',
        'UCSF\'te İKİ ayrı seçim ekseni vardır ve birbirinin yerine geçmez: (1) indirme kesintisi — ' +
        'elimizde hiç olmayan 94 hasta, Cox\'un harici testini etkiler; (2) 12-ay hedefinden dışlama — ' +
        'elimizde olan ama ikinci katman XGBoost hedefine girmeyen 63 hasta.',
        '12-ay hedefinden dışlanan 63 hasta ile değerlendirmeye giren 232 hasta arasındaki yanlılık ' +
        'ölçülmüştür (2026-09-14). Yaş, cinsiyet, rezeksiyon genişliği (GTR) ve MGMT durumunda fark ' +
        'saptanmamıştır. Tek sinyal IDH ekseninde ve dışlananların prognostik olarak lehinedir ' +
        '(IDH-mutant oranı dışlananlarda %12,7, kalanlarda %5,2); ham p=0,0469 iken Holm düzeltmesi ' +
        'sonrası p=0,2346, yani yedi testlik ailede ayrışmamaktadır. Takip süresi ve olay oranındaki ' +
        'fark mekaniktir (dışlama kuralının kendisi "erken sansürlenmiş" hastaları seçer, bu bir bulgu ' +
        'değildir). Örneklem n=63 olduğu için testin gücü düşüktür; bu ölçüm bir "yanlılık yoktur" ' +
        'sonucu DEĞİLDİR.'
      ]
    },

    /* --- A9 + A10 — kalibrasyon -------------------------------------- */
    'kalibrasyon-egimi': {
      src: 'A9 (güncel cümle) + A10 · ZORUNLU-BEYANLAR.md',
      p: [
        'Brier skoru, IPA ve kalibrasyon eğimi iki kol için hesaplanmıştır: v1 dağıtım modeli ve nihai ' +
        'model v3b_lowvar_v2amgmt. Nihai model için kalibrasyon eğimi 0,685 (%95 GA 0,470–0,900)\'dir; ' +
        'aralığın üst sınırı 1,0\'ın altında kaldığından model, tahmin yayılımını gözlenenden geniş ' +
        'vermektedir. Brier skorları t=365/548/730 için sırasıyla 0,1924 / 0,2092 / 0,1989, IPA ' +
        'değerleri 0,124 / 0,163 / 0,140 ve decile MAE(365) 0,0693\'tür. Kalan altı kol için ' +
        'kalibrasyon ölçülmemiştir.',
        'Bu sayılar YALNIZ ölçüm olarak raporlanır; "iyi kalibre" veya "kötü kalibre" hükmü ' +
        'kurulmamaktadır. Mutlak sağkalım olasılığı iddiası da kurulmaz — model yalnız sıralama verir.'
      ]
    },

    /* --- A3 — çokluk dipnotu ------------------------------------------ */
    'cokluk-dipnotu': {
      src: 'A3 (güncel cümle, 2026-09-13 kutusu) · ZORUNLU-BEYANLAR.md',
      p: [
        'Harici test kohortu (UCSF-PDGM) toplamda dokuz değerlendirmede kullanılmıştır: sekiz Cox kolu ' +
        '(v1, v2a, v2b, v2c, v3a, v3b, v3c, k14_clinical_base — tamamı 295 hasta / 169 olay üzerinde, ' +
        'Harrell C) ve bunlara ek olarak ikinci katman XGBoost sınıflandırıcısının bir harici ' +
        'değerlendirmesi (12-ay sağkalım sınıflandırması, AUC 0,7237 [%95 GA 0,6508–0,7903], n=232; ' +
        '12-ay ufkunda durumu belirsiz 63 hasta dışlandı, IPCW uygulanmadı). Sunulan sonuçlar bu ' +
        'nedenle bir miktar iyimser olabilir. Sekiz Cox kolunun tamamı Tablo X\'te, XGBoost ' +
        'değerlendirmesi Tablo Y\'de verilmiştir; hiçbiri gizlenmemiştir.',
        'Sayım kuralı: "8 kol" yalnız Cox ailesidir; "9 bakış" UCSF\'e yapılan toplam değerlendirmedir. ' +
        'XGBoost bir Cox kolu DEĞİLDİR — Cox meta-skorunu girdi alan ikinci katmandır.'
      ]
    },

    /* --- C8 — IPCW yok, n=232 ----------------------------------------- */
    'ipcw-yok': {
      src: 'C8 · ZORUNLU-BEYANLAR.md',
      p: [
        'İkinci katman XGBoost sınıflandırıcısı 12-ay sağkalım hedefiyle eğitilmiştir. Etiket kuralı ' +
        'şudur: takip süresi 365 günü aşan hasta Yes, 365 günden önce ölen hasta No, 365 günden önce ' +
        'sansürlenen hasta ise 12. ayda durumu bilinemediği için hedef dışı bırakılır. Bu dışlama, ' +
        'uydurma etiket üretmemek için bilinçli olarak seçilmiştir, ancak ters-sansür-olasılığı ' +
        'ağırlıklandırması (IPCW) uygulanmamıştır ve dışlanma oranı iki kohortta çarpıcı biçimde ' +
        'farklıdır: eğitim kohortunda (UPenn) 5/611 hasta (%0,82), harici test kohortunda (UCSF-PDGM) ' +
        '63/295 hasta (%21,4) dışlanmıştır — yaklaşık 26 kat fark. Bu nedenle XGBoost\'un iç ' +
        'out-of-fold AUC\'si n=606, harici AUC\'si ise 295\'in tamamı değil n=232 hasta üzerinde ' +
        'ölçülmüştür. Yani harici test popülasyonunun yaklaşık beşte biri ikinci katman ' +
        'değerlendirmesinin dışındadır ve Cox katmanının harici popülasyonu (295) ile XGBoost ' +
        'katmanının harici popülasyonu (232) birebir aynı değildir. IPCW yeni bir yöntem katmanı ' +
        'gerektirdiği için bu çalışmanın kapsamı dışında bırakılmıştır; eksikliği bir sınır olarak ' +
        'beyan edilmektedir.'
      ]
    },

    /* --- D2 — model_registry shadow ------------------------------------ */
    'model-registry-shadow': {
      src: 'D2 (güncel cümle, 2026-09-14 kutusu) · ZORUNLU-BEYANLAR.md',
      p: [
        'model_registry tablosunda dört model kaydı bulunmaktadır; nihai Cox modeli (cox_phm, sürüm ' +
        'v3b_lowvar_v2amgmt) "shadow" statüsüyle kayıtlıdır — mimari v4.5 §6.5 gereği yeni bir sürüm ' +
        'doğrudan production\'a geçmez (tools/write_model_registry.py:56, sabit STATUS = "shadow", ' +
        'kodda KİLİTLİ). Kaydın sınırı açıkça beyan edilir: metrikler eğitim/değerlendirme ' +
        'CSV\'lerinden okunmuştur ve hangi .pkl artefaktının bu metriklere karşılık geldiği kayıt ' +
        'tarafından doğrulanmamıştır (artifact_provenance alanı bunu kendi metniyle söyler). Yani ' +
        'model kimliği kayıt altına alınmıştır, ancak artefakt–metrik bağı henüz otoriteyle ' +
        'bağlanmamıştır.'
      ]
    },

    /* --- B1 + B2 + B8 + C8 — hangi popülasyon? -------------------------- */
    'hangi-populasyonda-dogrulandi': {
      src: 'B1 · B2 · B8 · C8 · ZORUNLU-BEYANLAR.md',
      p: [
        'Model UPenn-GBM\'in 611 hastası / 585 ölüm olayı ile eğitilmiştir. Kayıtlı 630 hastanın ' +
        '19\'unda yalnız post-operatif T1ce bulunmakta ve bu 19 hastanın hiçbirinde hazır segmentasyon ' +
        'maskesi olmadığı için radyomik satırı üretilememiştir. Dışarıda kalan 19 hasta 18 ölüm olayı ' +
        'taşımaktadır (olayların %3,0\'ı).',
        'Harici doğrulama YALNIZ UCSF-PDGM üzerinde (295 hasta / 169 olay) yapılmıştır. LUMIERE ' +
        'kohortu OS-Cox eğitimine dahil edilmemiştir; kaynağında olay/vital_status verisi ' +
        'bulunmamaktadır (0/91). LUMIERE\'nin rolü longitudinal büyüme simülasyonu, T3 RANO ' +
        'kalibrasyonu ve FAISS benzer-hasta kohortudur. TCGA-GBM ne eğitim ne harici test setidir; ' +
        'FAISS benzer-hasta havuzunun üyesidir.',
        'Dört veri kaynağı dört farklı rolde kullanıldığı için tek bir "toplam kohort" sayısı ' +
        'verilmemektedir. İkinci katman XGBoost\'un harici popülasyonu (n=232) Cox katmanınınkiyle ' +
        '(n=295) birebir aynı değildir.'
      ]
    },

    /* --- D8-v2 — büyüme projeksiyonunun yorumlanabilirliği -------------- */
    'buyume-yorumlanabilirlik': {
      src: 'D8-v2 (a-v2) · ZORUNLU-BEYANLAR.md',
      p: [
        'Büyüme projeksiyonunun yorumlanabilirlik oranı ÖLÇÜLMÜŞTÜR ve düşüktür. LUMIERE\'in ' +
        'longitudinal C32 WT serisindeki 90 hastanın tamamı canlı kodla iki kez değerlendirilmiştir: ' +
        'büyüme eğrisi fit\'i 76 hastada "ok", projeksiyon 67 hastada (yalnız RANO PD) üretilebilmiş, ' +
        've bu 67\'nin yalnız 12\'sinde aralık yorumlanabilir çıkmıştır. Geri kalan 55 hastada model ' +
        'uyumu aralığı fiziksel olarak anlamsız kılmaktadır: 67 hastanın 45\'inde üst uç yetişkin ' +
        'kafa-içi hacmini (≈1,4 L) aşmakta, 29\'unda alt uç −%99,9\'un altına düşmektedir.',
        'Ayrıca 67 PD hastasının 7\'sinde projeksiyonun üst ucu bile negatiftir; RANO\'da progresif ' +
        'etiketli hasta için model "tümör kesin küçülüyor" demektedir — RANO etiketi ile volumetrik ' +
        'yörünge birbirini desteklememektedir. Volumetrik değerlendirme resmî RANO DEĞİLDİR.'
      ]
    },

    /* --- A2 — radyomiğin yaş üstüne katkısı ---------------------------- */
    'radyomik-katki-beyani': {
      src: 'A2 · ZORUNLU-BEYANLAR.md',
      p: [
        'Yalnız yaş ve cinsiyet içeren taban kol (clinical_base) aynı nested-CV protokolüyle koşulmuş, ' +
        'UCSF harici testinde C-index 0,666 [0,621–0,712] vermiştir. 93 radyomik özellik + GTR + IDH1 ' +
        'eklendiğinde harici C-index 0,678 [0,634–0,723]\'e çıkmış, artım +0,012 olmuş ve iki kolun ' +
        'güven aralıkları tamamen örtüşmüştür. Bu nedenle radyomik bloğun yaşın üstüne anlamlı ' +
        'prognostik katkı sağladığı bu veriyle gösterilememiştir. Katkının tek izi geç zaman ufkundaki ' +
        'zamana-bağlı AUC\'dedir (730 gün: taban 0,683 → v1 0,719 → v2c 0,753); bunlar nokta ' +
        'tahminleridir ve eşleştirilmiş fark testi yapılmamıştır.',
        'Dil uyarısı: "bileşen katkı analizi yapılmadı" YANLIŞTIR — yapıldı; doğrusu "yapıldı, artım ' +
        'istatistiksel olarak ayrışmadı"dır.'
      ]
    },

    /* --- D13 (+ B6/B9) — SHAP yorum sınırı ----------------------------- */
    'shap-yorum-siniri': {
      src: 'D13 + B6(a) + B9 · ZORUNLU-BEYANLAR.md',
      p: [
        'Açıklanabilirlik katmanı (SHAP) her istekte yeniden hesaplanır ve API yanıtında canlı olarak ' +
        'döner; nihai model v3b_lowvar_v2amgmt için yanıt 24 katkı değeri taşır (16 radyomik + 8 ' +
        'klinik kovaryat) ve additivite artığı ölçülmüştür (~1,11e-16, makine hassasiyeti seviyesinde). ' +
        'Bu değerler kalıcı olarak veritabanına YAZILMAMAKTADIR; hangi hastaya hangi risk skorunun ve ' +
        'hangi SHAP profilinin döndüğü geriye dönük olarak SQL ile sorgulanamaz.',
        'SHAP katkıları modelin kendi çıktısının ayrıştırılmasıdır; nedensellik iddiası değildir. ' +
        'Özellikle clinical_idh_missing ve clinical_mgmt_missing katsayıları biyoloji değil SÜREÇ ' +
        'yansıtır — testin kime yapılabildiğini. "IDH testi yaptırmamak kötü prognoz sebebidir" ' +
        'biçiminde bir cümle kurulamaz.'
      ]
    },

    /* --- B1 + B5 + D13 + D15 — bilinen veri eksiklikleri ---------------- */
    'bilinen-veri-eksiklikleri': {
      src: 'B1 · B5 (+2026-09-14 eki) · D13 · D15 · ZORUNLU-BEYANLAR.md',
      p: [
        'UPenn\'in kayıtlı 630 hastasından 19\'u (18 ölüm olayı) radyomik satırı üretilemediği için ' +
        'eğitim havuzunun dışındadır; fiili havuz 611 hasta / 585 olaydır.',
        'UPenn\'in hacimsel ">%90 rezeksiyon" göstergesi ile UCSF\'in cerrahi "extent of resection" ' +
        'notu eşleştirilmiştir. Bu bir VARSAYIMDIR; iki ölçütün aynı şeyi ölçtüğü doğrulanmamıştır. ' +
        'Varsayım review-gate Adım 3\'te (2026-09-14) ekip tarafından incelenmiş ve varsayım olarak ' +
        'korunmasına karar verilmiştir; iki ölçütün aynı şeyi ölçtüğünü gösterecek bir doğrulama ' +
        'çalışması yapılmamıştır. "İncelendi" demek "doğrulandı" demek DEĞİLDİR. GTR kovaryatı ' +
        'olmadan yeniden fit (duyarlılık analizi) koşulmamıştır.',
        'IDH testi yapılmamış hastalar düşürülmemiş, gösterge değişkeniyle (clinical_idh_missing) ele ' +
        'alınmıştır; UCSF\'te "test yapılmamış" kategorisi olmadığı için bu gösterge harici testte ' +
        'sabit 0\'dır ve o katsayı harici olarak doğrulanamaz.',
        'Tahmin ve SHAP çıktıları veritabanına kalıcı olarak yazılmamaktadır (bilinçli kapsam kararı, ' +
        '2026-09-11 predictions tablosu kararı).'
      ]
    },

    /* --- Metni HENÜZ OLMAYAN beyan: uydurulmaz, yer tutucu kalır -------- */
    'veri-kullanim-sartlari': {
      src: null,
      p: null   // ZORUNLU-BEYANLAR.md'de karşılığı olan "kullanıma hazır cümle" YOK.
    }
  };

  /**
   * Beyan metnini HTML olarak döndürür. Metin yoksa görünür yer tutucu.
   * Kullanım: `GBM.decl.html('secim-yanliligi')`
   */
  function html(key) {
    var e = Object.prototype.hasOwnProperty.call(TEXTS, key) ? TEXTS[key] : undefined;
    if (!e || !e.p || !e.p.length) {
      return '<span class="decl-pending">[[BEYAN: ' + GBM.util.esc(key) + ']] — ' +
             'metin ZORUNLU-BEYANLAR.md\'de henüz "kullanıma hazır cümle" olarak yok; uydurulmaz.</span>';
    }
    return e.p.map(function (para) {
      return '<p class="decl-p">' + GBM.util.esc(para) + '</p>';
    }).join('') +
    (e.src ? '<div class="decl-src">Kaynak: ' + GBM.util.esc(e.src) + '</div>' : '');
  }

  /** Tek paragraflık, kaynak etiketi OLMAYAN kısa sürüm (üst şerit için). */
  function inline(key) {
    var e = Object.prototype.hasOwnProperty.call(TEXTS, key) ? TEXTS[key] : undefined;
    if (!e || !e.p || !e.p.length) {
      return '<span class="decl-pending">[[BEYAN: ' + GBM.util.esc(key) + ']]</span>';
    }
    return GBM.util.esc(e.p[0]);
  }

  /** `data-decl="anahtar"` taşıyan tüm düğümleri doldurur (index.html için). */
  function hydrate(root) {
    GBM.util.$$('[data-decl]', root || document).forEach(function (el) {
      el.innerHTML = el.hasAttribute('data-decl-inline')
        ? inline(el.getAttribute('data-decl'))
        : html(el.getAttribute('data-decl'));
    });
  }

  /* ---------------------------------------------------------------- (B) */

  /**
   * 8 Cox kolu — CLAUDE.md kilitli listesi.
   * `pool` = eğitim havuzu. WT+TC kolları (v2c, v3c) 609/583'tür; diğerleri
   * 611/585. Bu ayrım ZORUNLU-BEYANLAR.md B7 gereği tabloda görünür kalır.
   * Performans hücreleri (iç CV, harici C + CI) BU DOSYADA YOKTUR —
   * `GET /model_performance` ucundan gelir.
   */
  /*
     ⚠️ `api` alanı, `GET /model_performance -> cox_variants[].variant`
     içindeki GERÇEK adı taşır. CLAUDE.md'deki kısa adlar (v1, v2a, …) ile
     artefakt CSV'sindeki adlar (v1_referans, v2a_mgmt, …) AYNI DEĞİLDİR;
     çapraz kontrol bu alan üzerinden yapılır. Tablo API satırlarından
     çizilir — burası yalnız "8 kolun hepsi geldi mi" denetimi içindir.
  */
  var COX_ARMS = [
    { name: 'v1',                            api: 'v1_referans',                    space: '93 rad + 6 klinik',   pool: '611/585', role: 'varyant',    flag: 'Değerlendirildi' },
    { name: 'v2a',                           api: 'v2a_mgmt',                       space: '—',                   pool: '611/585', role: 'varyant',    flag: 'Değerlendirildi' },
    { name: 'v2b',                           api: 'v2b_mgmt_spline',                space: '—',                   pool: '611/585', role: 'varyant',    flag: 'Değerlendirildi' },
    { name: 'v2c',                           api: 'v2c_mgmt_spline_wttc',           space: '—',                   pool: '609/583', role: 'duyarlilik', flag: 'Değerlendirildi' },
    { name: 'v3a',                           api: 'v3a_lowvar_v1referans',          space: '—',                   pool: '611/585', role: 'varyant',    flag: 'Değerlendirildi' },
    { name: 'v3b_lowvar_v2amgmt',            api: 'v3b_lowvar_v2amgmt',             space: '16 rad + 8 klinik',   pool: '611/585', role: 'birincil',   flag: 'SEÇİLDİ' },
    { name: 'v3c_lowvar_wttc_mgmt_nospline', api: 'v3c_lowvar_wttc_mgmt_nospline',  space: '—',                   pool: '609/583', role: 'duyarlilik', flag: 'Duyarlılık kolu' },
    { name: 'k14_clinical_base',             api: 'k14_clinical_base_ucsf',         space: 'yalnız yaş+cinsiyet', pool: '611/585', role: 'taban',      flag: 'Taban çizgisi' }
  ];

  /**
   * Kohort rolleri — tek bir "toplam hasta sayısı" VERİLMEZ (B8).
   * Buradaki `role` metni YALNIZ yedektir: API (`GET /patients`) kendi rol
   * etiketini döndürüyorsa EKRANDA ONUN metni kullanılır.
   */
  var COHORT_ROLES = [
    { source: 'UPenn-GBM', role: 'Eğitim kohortu' },
    { source: 'UCSF-PDGM', role: 'Harici test' },
    { source: 'LUMIERE',   role: 'Büyüme simülasyonu · RANO · FAISS kohortu' },
    { source: 'TCGA-GBM',  role: 'FAISS benzer-hasta havuzu üyesi' }
  ];

  /** Sınırlar sekmesi başlık listesi. */
  var LIMIT_TOPICS = [
    { title: 'Klinik kullanım için değildir',                          decl: 'klinik-kullanim-yok' },
    { title: 'Hangi popülasyonda doğrulandı, hangisinde doğrulanmadı', decl: 'hangi-populasyonda-dogrulandi' },
    { title: 'Seçim yanlılığı — harici test kohortu',                  decl: 'secim-yanliligi' },
    { title: 'Kalibrasyon eğimi — mutlak olasılık iddiası kurulmaz',   decl: 'kalibrasyon-egimi' },
    { title: 'Çoklu değerlendirme dipnotu',                            decl: 'cokluk-dipnotu' },
    { title: '12-ay hedefinde IPCW yok — harici n=232',                decl: 'ipcw-yok' },
    { title: 'Model kaydı "shadow" statüsündedir',                     decl: 'model-registry-shadow' },
    { title: 'Büyüme projeksiyonunun yorumlanabilirlik oranı',         decl: 'buyume-yorumlanabilirlik' },
    { title: 'Radyomik bloğun yaşın üstüne artımı',                    decl: 'radyomik-katki-beyani' },
    { title: 'SHAP katkılarının yorumlanma sınırı',                    decl: 'shap-yorum-siniri' },
    { title: 'Bilinen veri eksiklikleri',                              decl: 'bilinen-veri-eksiklikleri' },
    { title: 'Veri kullanım şartları ve atıflar',                      decl: 'veri-kullanim-sartlari' }
  ];

  /**
   * İnsan-okunur SHAP etiketleri.
   * 🔴 KURAL: bu harita YALNIZ bir OKUMA YARDIMIDIR. Ham özellik adı her
   * zaman `title`/tooltip olarak erişilebilir kalır ve haritada olmayan ad
   * OLDUĞU GİBİ gösterilir — uydurma ad ÜRETİLMEZ.
   */
  var FEATURE_LABELS = {
    'clinical_age':             'Yaş',
    'clinical_sex_male':        'Cinsiyet (erkek)',
    /* 🔴 2026-09-16 ÖLÇÜLDÜ: nihai modelin 24 kovaryatından 23'ünün etiketi
       vardı, BİRİ eksikti — çünkü harita `clinical_sex_male` diyor ama
       `week3_v3b_lowvar_v2amgmt_final_coefficients.csv` içindeki gerçek ad
       `clinical_gender_male`. Eski anahtar SİLİNMEDİ (başka bir kolda o ad
       kullanılmış olabilir), doğrusu EKLENDİ. */
    'clinical_gender_male':     'Cinsiyet (erkek)',
    'clinical_gtr_y':           'Rezeksiyon >%90 (GTR)',
    'clinical_gtr_missing':     'GTR bilgisi yok (süreç göstergesi)',
    'clinical_idh_mutant':      'IDH1 mutant',
    'clinical_idh_missing':     'IDH testi yapılmamış (süreç göstergesi)',
    'clinical_mgmt_methylated': 'MGMT metile',
    'clinical_mgmt_missing':    'MGMT testi yapılmamış (süreç göstergesi)'
  };

  /**
   * Radyomik ad çözümleme: `WT__original_glcm_InverseVariance` gibi adları
   * "WT · GLCM · InverseVariance" biçiminde okunur hâle getirir.
   * Ad kalıba uymuyorsa HAM AD döner (uydurma yok).
   */
  function featureLabel(raw) {
    if (!raw) return '';
    if (Object.prototype.hasOwnProperty.call(FEATURE_LABELS, raw)) return FEATURE_LABELS[raw];
    var m = /^([A-Za-z]+)__([A-Za-z0-9-]+)_([A-Za-z0-9]+)_(.+)$/.exec(String(raw));
    if (m) {
      // m[1]=bölge (WT/TC/NC/ED/ET), m[2]=görüntü tipi (original/wavelet-...),
      // m[3]=özellik ailesi (glcm/glrlm/shape/firstorder), m[4]=özellik adı
      var family = m[3].toUpperCase();
      var img = m[2] === 'original' ? '' : ' · ' + m[2];
      return m[1] + img + ' · ' + family + ' · ' + m[4];
    }
    /* 2026-09-16 (Barış kararı): ham `snake_case` ad ekranda GÖSTERİLMEZ.
       Kalıba uymayan bir ad gelirse alt çizgiler boşluğa çevrilip ilk harf
       büyütülür — uydurma bir ad ÜRETİLMEZ, yalnız biçim insanîleştirilir. */
    var t = String(raw).replace(/_/g, ' ').trim();
    return t ? t.charAt(0).toUpperCase() + t.slice(1) : '';
  }

  GBM.decl = {
    html: html, inline: inline, hydrate: hydrate, texts: TEXTS,
    COX_ARMS: COX_ARMS, COHORT_ROLES: COHORT_ROLES, LIMIT_TOPICS: LIMIT_TOPICS,
    FEATURE_LABELS: FEATURE_LABELS, featureLabel: featureLabel
  };
})(window);
