/* ===========================================================================
   GBM.views — beş ekranın HTML üretimi (AŞAMA 3-B: gerçek API'ye bağlı).

   Dönüşüm kuralları (tasarım -> vanilla):
     <sc-for list="{{x}}" as="i">  ->  x.map(i => `...`).join('')
     <sc-if value="{{c}}">         ->  c ? `...` : ''
     {{ x }}                       ->  ${esc(x)}
     onClick="{{ fn }}"            ->  data-act="fn" + olay delegasyonu

   🔴 Kaynak tasarımdaki `synthesizePatient` / `upennRawToPatient` /
   `lumiereRawToPatient` / `tcgaRawToPatient` / `buildSimilar` fonksiyonları
   risk skorunu ve SHAP katkılarını hasta kimliğinin HASH'inden üretiyordu.
   HİÇBİRİ BURAYA TAŞINMAMIŞTIR. Tüm sayılar API'den gelir; gelmeyen sayı
   '—' kalır veya görünür bir "yok" kutusu çizilir.
   =========================================================================== */
(function (global) {
  'use strict';

  var GBM = global.GBM = global.GBM || {};
  var U = GBM.util;
  var esc = U.esc;

  /* =======================================================================
     ORTAK PARÇALAR
     ======================================================================= */

  var NAV = [
    { key: 'home',     hash: '#/',              label: 'Ana Sayfa' },
    { key: 'analysis', hash: '#/hasta',         label: 'Hasta Analizi' },
    { key: 'history',  hash: '#/gecmis',        label: 'Hasta Geçmişi' },
    { key: 'cohort',   hash: '#/kohort',        label: 'Kohort Gezgini' },
    { key: 'system',   hash: '#/sistem/sistem', label: 'Sistem Hakkında' }
  ];

  function navHTML(active, patientId) {
    return NAV.map(function (n) {
      var hash = n.hash;
      if (n.key === 'history' && patientId) hash = '#/gecmis/' + encodeURIComponent(patientId);
      return '<button class="nav-link" data-act="nav" data-hash="' + esc(hash) + '"' +
             (n.key === active ? ' aria-current="page"' : '') + '>' + esc(n.label) + '</button>';
    }).join('');
  }

  /** Kartın hangi uçtan beslendiğini görünür kılar (şeffaflık). */
  /* 2026-09-16 (Barış kararı): "kaynak: GET /..." etiketleri siteden kaldırıldı.
     Hangi uç noktanın hangi bloğu beslediği bir geliştirici ayrıntısıdır; jüri
     vitrininde yeri yoktur. Fonksiyon imzası KORUNDU (27 çağrı yeri var), yalnız
     boş dönüyor — böylece çağrı yerlerine dokunmaya gerek kalmadı. */
  function hookTag(hook) {
    return '';
  }

  /**
   * Bir bloğun gövdesini durumuna göre çizer.
   * @param {object} o {status, block, reason, renderFn, skelLines, what, hook}
   *   status: idle | loading | ready | error | missing
   */
  function blockBody(o) {
    var status = o.status;

    if (status === 'idle') {
      return '<div class="block-unavailable">Önce bir hasta seçin.</div>';
    }
    if (status === 'loading') {
      return U.skeleton(o.skelLines || 3) +
             '<div class="block-wait" data-ticker="' + esc(o.what || 'Sonuç bekleniyor') + '"></div>';
    }
    if (status === 'missing') {
      return '<div class="block-unavailable"><b>Bu bölüm şu an gösterilemiyor.</b></div>';
    }
    // 'na' = sunucu 422 ile "bu hasta bu bloğa TANIM GEREĞİ girmiyor" dedi.
    // (Örn. UCSF-PDGM hastaları FAISS v1 indeksinde yoktur.) Arıza DEĞİLDİR,
    // ama SESSİZ de değildir — sunucunun gerekçesi tam olarak yazılır.
    if (status === 'na') {
      return '<div class="block-unavailable">' +
        '<div class="block-unavailable__head">Bu bölüm bu hasta için üretilmiyor.</div></div>';
    }
    if (status === 'error') {
      return '<div class="block-error"><b>Bu bölüm şu an gösterilemiyor.</b></div>';
    }
    // ready
    var b = o.block;
    if (!b) {
      return '<div class="block-unavailable">Bu blok yanıtta yok.' + hookTag(o.hook || '') + '</div>';
    }
    if (b.available === false) {
      /* 2026-09-16 (Barış kararı): gerekçe artık `humanizeReason()`'dan geçer —
         kod ayrıntısı taşıyan parçalar atılır, anlamlı kısım korunur. HTTP
         durum kodu satırı kaldırıldı. */
      var why = humanizeReason(b.not_available_reason || b.error_detail || b.reason || '');
      return '<div class="block-unavailable">' +
        '<div class="block-unavailable__head">Bu bölüm bu hasta için üretilemedi.</div>' +
        (why ? '<div class="block-unavailable__why">' + esc(why) + '</div>' : '') +
      '</div>';
    }
    return o.renderFn(b);
  }

  /** "0,6592 [0,6164–0,7020]" biçimi. Eksik değer '—'. */
  function fmtCI(point, lo, hi, digits) {
    var d = digits === undefined ? 4 : digits;
    var p = U.num(point, d);
    if (p === '—') return '—';
    var l = U.num(lo, d), h = U.num(hi, d);
    if (l === '—' || h === '—') return p;
    return p + ' [' + l + '–' + h + ']';
  }

  /* =======================================================================
     EKRAN 1 — ANA SAYFA
     ======================================================================= */

  /** `counters.by_source[]` -> kaynak adına göre sözlük. */
  function countersBySource(s) {
    var out = {};
    var c = s.directory && s.directory.counters;
    if (!c && s.cohort && s.cohort.counters) c = s.cohort.counters;
    if (!c || !Array.isArray(c.by_source)) return out;
    c.by_source.forEach(function (row) { out[row.source] = row; });
    return out;
  }

  /** Sayaç satırındaki sayısal alanların insan-okunur etiketleri. */
  var COUNTER_LABELS = {
    n_registered: 'kayıtlı',
    n_train_pool: 'eğitim havuzu',
    n_events_train_pool: 'eğitim ölüm olayı',
    n_external_test_pool: 'harici test havuzu',
    n_events_external_test: 'harici test olayı',
    n_c32_radiomics_any_visit: 'C32 radyomikli (herhangi ziyaret)',
    n_faiss_servisable_canonical_preop: 'FAISS servis edilebilir (kanonik pre-op)',
    n_c32_radiomics: 'C32 radyomikli',
    n_predict_servisable: '/predict servis edilebilir',
    n_has_omics_total: 'omics verisi olan'
  };

  function counterMetrics(row) {
    return Object.keys(row).filter(function (k) {
      return k !== 'source' && k !== 'role' && k !== 'note' && typeof row[k] === 'number';
    }).map(function (k) {
      return '<div class="counter__metric"><span class="counter__metric-k">' +
             esc(COUNTER_LABELS[k] || k) + '</span><span class="counter__metric-v">' +
             esc(U.num(row[k])) + '</span></div>';
    }).join('');
  }

  function home(s) {
    var cards = [
      { title: 'Hasta Analizi',   desc: 'Risk skoru, SHAP katkıları, benzer hastalar ve büyüme projeksiyonu.', bg: '#f3eefc', color: '#6d28d9', shape: '50%',           hash: '#/hasta' },
      { title: 'Hasta Geçmişi',   desc: 'Longitudinal tümör hacmi takibi ve RANO durum tablosu.',              bg: '#eef2f6', color: '#475569', shape: '4px',           hash: '#/gecmis' },
      { title: 'Kohort Gezgini',  desc: 'Tüm veri setlerini filtreleyip karşılaştırabileceğiniz hasta tablosu.', bg: '#fdf6ec', color: '#b45309', shape: '4px',         hash: '#/kohort' },
      { title: 'Sistem Hakkında', desc: 'Model performansı, veri setleri ve bilinen sınırlar.',                bg: '#eef2f6', color: '#0f172a', shape: '50% 50% 50% 0', hash: '#/sistem/sistem' }
    ];

    var by = countersBySource(s);
    var stats = GBM.decl.COHORT_ROLES.map(function (c) {
      var row = by[c.source];
      var count = row && typeof row.n_registered === 'number' ? U.num(row.n_registered) : '—';
      return '<div class="stat"><div class="stat__value">' + esc(count) + '</div>' +
             '<div class="stat__label">' + esc(c.source) + '</div>' +
             '<div class="stat__role">' + esc((row && row.role) || c.role) + '</div></div>';
    }).join('');

    return '' +
    '<section class="hero">' +
      '<div class="hero__art"><img src="assets/logo.png" alt=""></div>' +
      '<div class="hero__inner">' +
        '<h1 class="hero__title">Glioblastoma için<br>klinik karar desteği</h1>' +
        '<p class="hero__text">MR görüntüleri, klinik veriler ve moleküler profilleri birleştirerek risk sıralaması, benzer hasta eşleştirmesi ve tümör büyüme projeksiyonu sunar. Bileşenler bağımsız çalışır — biri kullanılamadığında diğerleri devam eder.</p>' +
        '<div class="hero__cta">' +
          '<button class="btn btn--primary" data-act="nav" data-hash="#/hasta">Hasta Analizine Git &rarr;</button>' +
          '<button class="btn btn--ghost" data-act="nav" data-hash="#/sistem/sistem">Sistem Nasıl Çalışır?</button>' +
        '</div>' +
      '</div>' +
    '</section>' +

    '<section class="section">' +
      '<div class="section__kicker">Sistemi Keşfedin</div>' +
      '<div class="feature-grid">' +
        cards.map(function (f) {
          return '<button class="feature-card" data-act="nav" data-hash="' + esc(f.hash) + '">' +
                   '<span class="feature-card__icon" style="background:' + esc(f.bg) + ';">' +
                     '<span class="feature-card__dot" style="background:' + esc(f.color) + ';border-radius:' + esc(f.shape) + ';"></span>' +
                   '</span>' +
                   '<span class="feature-card__title">' + esc(f.title) + '</span>' +
                   '<span class="feature-card__desc">' + esc(f.desc) + '</span>' +
                 '</button>';
        }).join('') +
      '</div>' +
    '</section>' +

    '<section class="section section--last">' +
      '<div class="stat-strip">' + stats + '</div>' +
      (directoryNote(s) ? '<p class="fine" style="margin-top:10px;">' + directoryNote(s) + '</p>' : '') +
    '</section>';
  }

  /**
   * Dizin yüklemesinin durumu — sessiz beklemeyi engeller.
   * 2026-09-16 (Barış kararı): başarılı durumda HİÇBİR ŞEY yazmaz. "N satır
   * yüklendi" ve sayıların nereden geldiği bir geliştirici ayrıntısıdır.
   * Yalnız kullanıcının gerçekten bilmesi gereken iki durum konuşur:
   * yükleniyor ve alınamadı.
   */
  function directoryNote(s) {
    if (s.directoryStatus === 'loading') return 'Hasta dizini yükleniyor&hellip;';
    if (s.directoryStatus === 'error' || s.directoryStatus === 'missing') {
      return '<span class="inline-error">Hasta dizini alınamadı.</span>';
    }
    return '';
  }

  /* =======================================================================
     EKRAN 2 — HASTA ANALİZİ
     ======================================================================= */

  function directoryRows(s) {
    return (s.directory && s.directory.rows) || [];
  }

  function suggestionsHTML(s) {
    var q = (s.searchInput || '').trim().toLowerCase();
    if (q.length < 2) return '';
    var pool = directoryRows(s).filter(function (p) {
      return String(p.patient_id).toLowerCase().indexOf(q) >= 0;
    }).slice(0, 8);
    if (!pool.length) return '';
    return '<div class="suggestions">' + pool.map(function (p) {
      return '<button class="suggestion" data-act="pick-patient" data-id="' + esc(p.patient_id) + '">' +
               '<span class="suggestion__id">' + esc(p.patient_id) + '</span>' +
               '<span class="suggestion__src">' + esc(p.source || '') + '</span>' +
             '</button>';
    }).join('') + '</div>';
  }

  function searchCard(s) {
    var rows = directoryRows(s);
    var opts = '<option value="">Hasta seçin</option>';
    if (rows.length) {
      opts += rows.slice(0, 1500).map(function (p) {
        return '<option value="' + esc(p.patient_id) + '"' +
               (p.patient_id === s.route.patientId ? ' selected' : '') + '>' +
               esc(p.patient_id) + ' · ' + esc(p.source || '') + '</option>';
      }).join('');
    } else {
      opts += '<option value="" disabled>' +
              (s.directoryStatus === 'loading' ? 'Liste yükleniyor…' : 'Liste yok') +
              '</option>';
    }

    return '' +
    '<div class="search-card">' +
      '<div class="search-card__title">Hasta Paneli</div>' +
      '<label class="label-kicker" for="patient-select">Hasta Seç</label>' +
      '<select id="patient-select" data-act="select-patient">' + opts + '</select>' +
      '<label class="label-kicker" for="patient-input">veya hasta kimliğini manuel gir</label>' +
      '<div class="search-row">' +
        '<input id="patient-input" type="text" autocomplete="off" placeholder="örn. UCSF-PDGM-167" value="' + esc(s.searchInput) + '" data-act="search-input">' +
        '<button class="btn btn--primary btn--sm" data-act="search-submit">Analiz Et</button>' +
        '<div id="suggestions-box">' + suggestionsHTML(s) + '</div>' +
      '</div>' +
      (s.searchError ? '<div class="form-error">' + esc(s.searchError) + '</div>' : '') +
      '<div class="fine" style="margin-top:8px;">' + directoryNote(s) + '</div>' +
    '</div>';
  }

  /**
   * Hasta kimlik şeridi.
   * 🔴 `/analyze_patient` ve `/predict` demografi DÖNDÜRMEZ (sözleşmede
   * üst düzey `patient` bloğu YOKTUR). Demografi `GET /patients`
   * satırından gelir. Dizin yüklenmediyse alanlar '—' kalır — UYDURULMAZ.
   */
  function identityCard(s) {
    var id = s.route.patientId;
    var row = GBM.state.directoryRow(id);
    var facts = [
      ['Yaş', row && row.age],
      ['Cinsiyet', row && (row.gender || row.sex)],
      ['MGMT', row && row.mgmt_status],
      ['IDH1', row && row.idh1_status],
      ['GTR >%90', row && row.gtr_over90percent],
      ['Vital', row && row.vital_status],
      ['Sağkalım (gün)', row && row.survival_days]
    ];
    var srcLine = row
      ? (row.source || '—') + (row.role ? ' · ' + row.role : '')
      : (s.directoryStatus === 'loading' ? 'Kaynak bilgisi yükleniyor…'
                                         : 'Bu kimlik hasta dizininde bulunamadı');
    return '' +
    '<div class="card">' +
      '<div class="identity">' +
        '<div>' +
          '<div class="identity__id mono">' + esc(id) + '</div>' +
          '<div class="identity__src">' + esc(srcLine) + '</div>' +
        '</div>' +
        '<div class="identity__facts">' +
          facts.map(function (f) {
            return '<div><div class="fact__k">' + esc(f[0]) + '</div>' +
                   '<div class="fact__v">' + esc(U.orDash(f[1])) + '</div></div>';
          }).join('') +
        '</div>' +
      '</div>' +
      (row && row.servis_edilebilir_mi === false
        ? '<div class="block-unavailable" style="margin-top:12px;">Bu hasta için analiz üretilemeyebilir.</div>'
        : '') +
      hookTag('GET /patients (demografi) · POST /predict/{id} ve POST /analyze_patient demografi döndürmez') +
    '</div>';
  }

  /**
   * `summary.unavailable_blocks[]` şeridi.
   * 🔴 SESSİZ BOŞLUK YOK: her öğenin `reason` metni TAM olarak yazılır.
   * UCSF'te FAISS ve büyüme, TCGA'da büyüme NORMALDE boştur — bu arıza
   * değil, tanım gereğidir ve ekranda öyle görünür.
   */
  function unavailableStrip(s) {
    if (s.analysisStatus !== 'ready') return '';
    var a = s.analysis;
    if (!a || !a.summary) return '';
    var list = a.summary.unavailable_blocks || [];
    if (a.summary.all_blocks_available || !list.length) {
      return '';
    }
    /* 2026-09-16 (Barış kararı): ham sunucu gerekçesi EKRANA YAZILMAZ. O metin
       dosya yolu, fonksiyon adı ve DLL hatası içeriyordu — jüri okumaz, okusa
       anlamaz. Yerine blok başına tek cümlelik insan dili konur; ayrıntı
       sunucu yanıtında ve loglarda zaten duruyor. */
    return '<div class="unavail-strip">' +
      list.map(function (u) {
        return '<div class="unavail-item">' +
                 '<div class="unavail-item__reason">' + esc(blockUnavailableText(u.block)) + '</div>' +
               '</div>';
      }).join('') +
    '</div>';
  }

  /** Kullanılamayan blok için kısa, insan dilinde tek cümle. */
  function blockUnavailableText(block) {
    var m = {
      risk:                  'Risk skoru bu hasta için üretilemedi.',
      xgboost_12mo_survival: 'İkinci katman model çıktısı bu hasta için üretilemedi.',
      similar_patients:      'Bu hasta için benzer hasta karşılaştırması yapılamadı.',
      growth_simulation:     'Bu hasta için tümör büyüme simülasyonu üretilemedi.',
      literature:            'İlgili literatür özeti üretilemedi.'
    };
    return m[block] || 'Bu bölüm şu an gösterilemiyor.';
  }

  /* --------------------------------------------------------------- SHAP */

  /**
   * SHAP katkıları — sözlüğün TAMAMINI alır, en büyük N'i çizer.
   * Nihai model v3b için 24 değer gelir (16 radyomik + 8 klinik).
   * 🔴 `shap_base_value` GERÇEK değerden gelir; iskeletteki sabit −0,05 SİLİNDİ.
   */
  function shapHTML(risk, topN) {
    var dict = risk.shap_values;
    if (!dict || typeof dict !== 'object' || !Object.keys(dict).length) {
      return '<div class="block-unavailable">Bu hasta için katkı grafiği üretilemedi.</div>';
    }
    var all = Object.keys(dict).map(function (k) { return { feature: k, value: Number(dict[k]) }; })
      .filter(function (r) { return !isNaN(r.value); });
    var total = all.length;
    all.sort(function (a, b) { return Math.abs(b.value) - Math.abs(a.value); });

    var n = (topN >= total) ? total : topN;
    var shown = all.slice(0, n);
    var maxAbs = Math.max.apply(null, all.map(function (r) { return Math.abs(r.value); }).concat([1e-9]));

    var rows = shown.map(function (r) {
      var w = Math.abs(r.value) / maxAbs * 45;     // görsel ölçek (yarım eksen = %45)
      var pos = r.value >= 0;
      /* 2026-09-16 (Barış kararı): ham özellik adı (`WT__original_glcm_…`)
         ne satırda ne de `title` ipucunda GÖSTERİLMEZ — yalnız insan-okunur
         etiket kalır. */
      var label = GBM.decl.featureLabel(r.feature);
      return '<div class="shap-row">' +
               '<div class="shap-row__name">' + esc(label) + '</div>' +
               '<div class="shap-row__track">' +
                 '<div class="shap-row__bar ' + (pos ? 'shap-row__bar--pos' : 'shap-row__bar--neg') + '"' +
                 ' style="left:' + (pos ? 50 : 50 - w) + '%;width:' + w + '%;"></div>' +
               '</div>' +
               '<div class="shap-row__val ' + (pos ? 'shap-row__val--pos' : 'shap-row__val--neg') + '">' +
                 esc(U.signed(r.value, 4)) + '</div>' +
             '</div>';
    }).join('');

    var add = risk.shap_additivity_check_abs_diff;

    return rows +
      '<div class="shap-meta">' +
        '<div class="fine">' + esc(shown.length) + '/' + esc(total) +
          ' katkı gösteriliyor (mutlak değere göre en büyükler).</div>' +
        '<button class="linkish" data-act="shap-toggle">' +
          (n >= total ? 'Yalnız en büyük 8\'i göster' : 'Tümünü göster (' + esc(total) + ')') +
        '</button>' +
      '</div>' +
      /* Taban değeri ve toplanabilirlik denetimi satırı 2026-09-16'da kaldırıldı. */
      '';
  }

  /** 1,11e-16 gibi çok küçük sayıları bilimsel gösterimle yazar. */
  function formatTiny(v) {
    var n = Number(v);
    if (isNaN(n)) return '—';
    if (n === 0) return '0';
    if (Math.abs(n) < 1e-4) return n.toExponential(3).replace('.', ',');
    return U.num(n, 8);
  }

  /* ---------------------------------------------------------- RİSK KARTI */

  /**
   * Risk bloğunun kaynağı: ÖNCE `/predict` (hızlı, üst düzey alanlar),
   * o olmazsa `/analyze_patient -> risk`.
   */
  function riskStream(s) {
    if (s.predictStatus === 'ready' && s.predict) {
      return { status: 'ready', block: s.predict, reason: '', hook: 'POST /predict/{id}' };
    }
    if (s.predictStatus === 'loading') {
      return { status: 'loading', block: null, reason: '', hook: 'POST /predict/{id}' };
    }
    // /predict başarısız: tam zincirin risk bloğuna düş.
    if (s.analysisStatus === 'ready' && s.analysis) {
      return { status: 'ready', block: s.analysis.risk, reason: '',
               hook: 'POST /analyze_patient -> risk (yedek: /predict başarısız oldu)' };
    }
    if (s.analysisStatus === 'loading' && s.predictStatus !== 'idle') {
      return { status: 'loading', block: null, reason: '', hook: 'POST /analyze_patient -> risk' };
    }
    return { status: s.predictStatus, block: null, reason: s.predictReason, hook: 'POST /predict/{id}' };
  }

  function riskCard(s) {
    var st = riskStream(s);
    var arm = (st.block && st.block.model_arm) || '';
    return '' +
    '<div class="card">' +
      '<div class="card__head">' +
        '<div class="h2" style="margin:0;">Risk Skoru</div>' +
        '<div class="badge badge--neutral">' + esc(arm || 'model kolu bekleniyor') + '</div>' +
      '</div>' +
      blockBody({
        status: st.status, block: st.block, reason: st.reason, skelLines: 6,
        hook: st.hook, what: 'Cox risk skoru ve SHAP hesaplanıyor',
        renderFn: function (r) {
          var score = Number(r.risk_score_log_partial_hazard);
          var pct = U.clamp(50 + (isNaN(score) ? 0 : score) * 22, 4, 96);
          var nFeat = (r.final_features || []).length;
          var nClin = (r.clinical_extra_columns || []).length;
          return '' +
          '<div class="bigrow">' +
            '<div><div class="big__k">Log Kısmi Hazard</div>' +
              '<div class="big__v ' + (score >= 0 ? 'big__v--risk-high' : 'big__v--risk-low') + '">' +
              esc(U.signed(score, 4)) + '</div></div>' +
            '<div><div class="big__k">Kısmi Hazard Oranı</div>' +
              '<div class="big__v">' + esc(U.num(r.hazard_ratio_partial_hazard, 4)) + '&times;</div></div>' +
          '</div>' +
          '<div class="riskaxis">' +
            '<div class="riskaxis__track"></div><div class="riskaxis__mid"></div>' +
            '<div class="riskaxis__knob" style="left:' + pct + '%;border-color:' + (score >= 0 ? '#dc2626' : '#6d28d9') + ';"></div>' +
          '</div>' +
          '<div class="riskaxis__legend"><span>düşük risk</span><span>ortalama</span><span>yüksek risk</span></div>' +
          /* 2026-09-16 (Barış kararı): risk yorum notu ve "Model uzayı: ...
             (final_features + clinical_extra_columns)" satırı kaldırıldı. */
          '<div class="label-kicker" style="margin-top:18px;">SHAP Katkı Grafiği</div>' +
          shapHTML(r, s.shapTopN) +
          hookTag(st.hook);
        }
      }) +
    '</div>';
  }

  /* ------------------------------------------------------ BENZER HASTALAR */

  function similarCard(s) {
    var kOpts = ['5', '10', '20'];
    var filters = '<div class="filters">' +
      '<select data-act="sim-k" aria-label="Komşu sayısı">' +
        kOpts.map(function (k) {
          return '<option value="' + k + '"' + (s.simK === k ? ' selected' : '') + '>k = ' + k + '</option>';
        }).join('') +
      '</select>' +
      selectHTML('sim-idh', 'IDH1 — tümü', ['Wildtype', 'Mutated', 'NOS/NEC'], s.simIdh, 'IDH1 filtresi') +
      selectHTML('sim-mgmt', 'MGMT — tümü', ['Methylated', 'Unmethylated'], s.simMgmt, 'MGMT filtresi') +
    '</div>';

    return '' +
    '<div class="card">' +
      '<div class="h2">Benzer Hastalar (FAISS)</div>' +
      filters +
      blockBody({
        status: s.similarStatus, block: s.similar, reason: s.similarReason, skelLines: 5,
        hook: 'GET /patient/{id}/similar',
        what: 'FAISS benzer-hasta araması çalışıyor',
        renderFn: function (b) {
          var faiss = b.clinical_radiomics_faiss || {};
          var rows = faiss.results || [];
          var head =
            '<div class="fine" style="margin-bottom:10px;">' +
              'İndeks büyüklüğü: ' + esc(U.orDash(faiss.index_size)) +
              ' · istenen k: ' + esc(U.orDash(faiss.k_requested)) +
              ' · dönen k: ' + esc(U.orDash(faiss.k_returned)) +
              ' · L2 mesafe: küçük olan daha benzerdir.' +
            '</div>' +
            ((faiss.warnings && faiss.warnings.length)
              ? '<div class="unavail-strip">' + faiss.warnings.map(function (w) {
                  return '<div>&#9888; ' + esc(w) + '</div>';
                }).join('') + '</div>'
              : '');

          if (!rows.length) {
            return head + '<div class="block-unavailable">Bu filtrelerle komşu döndürülmedi.</div>' +
                   omicsFaissHTML(b) + hookTag('GET /patient/{id}/similar');
          }

          return head + rows.map(function (n) {
            return '<button class="simrow" data-act="pick-patient" data-id="' + esc(n.patient_id) + '">' +
                     '<span class="simrow__left">' +
                       '<span class="simrow__id mono">' + esc(n.patient_id) + '</span>' +
                       '<span class="simrow__meta">' + esc(U.orDash(n.source)) + '</span>' +
                       '<span class="simrow__meta">Yaş ' + esc(U.orDash(n.age)) + '</span>' +
                       '<span class="simrow__meta">' + esc(U.orDash(n.gender)) + '</span>' +
                       '<span class="simrow__meta">IDH1: ' + esc(U.orDash(n.idh1_status)) + '</span>' +
                       '<span class="simrow__meta">MGMT: ' + esc(U.orDash(n.mgmt_status)) + '</span>' +
                       '<span class="simrow__meta">GTR: ' + esc(U.orDash(n.gtr_over90percent)) + '</span>' +
                     '</span>' +
                     '<span class="simrow__right">' +
                       '<span class="simrow__meta">' + esc(U.orDash(n.vital_status)) + '</span>' +
                       '<span class="simrow__meta">Sağkalım: ' + esc(U.orDash(n.survival_days)) + ' gün</span>' +
                       '<span class="fine mono">L2 ' + esc(U.num(n.l2_distance, 4)) + '</span>' +
                     '</span>' +
                   '</button>';
          }).join('') +
          omicsFaissHTML(b) +
          hookTag('GET /patient/{id}/similar');
        }
      }) +
    '</div>';
  }

  function omicsFaissHTML(b) {
    var om = b.molecular_omics_faiss;
    if (!om) return '';
    if (om.available === false) {
      return '<div class="block-unavailable" style="margin-top:12px;">Moleküler (omics) FAISS indeksi bu hasta için ' +
             'kullanılamıyor — ' + esc(om.note || 'gerekçe bildirilmedi') + '</div>';
    }
    var rows = om.results || [];
    return '<div class="label-kicker" style="margin-top:16px;">Moleküler (omics) FAISS — indeks ' +
           esc(U.orDash(om.index_size)) + '</div>' +
           (rows.length
             ? rows.map(function (n) {
                 return '<div class="simrow simrow--static">' +
                   '<span class="simrow__id mono">' + esc(n.patient_id) + '</span>' +
                   '<span class="fine mono">L2 ' + esc(U.num(n.l2_distance, 4)) + '</span></div>';
               }).join('')
             : '<div class="fine">Sonuç döndürülmedi.</div>');
  }

  /* ------------------------------------------------------------- BÜYÜME */

  function projectionTiles(p) {
    if (!p) return '';
    return '<div class="tiles" style="margin-top:14px;">' +
      '<div class="tile"><div class="tile__k">En Az</div><div class="tile__v tile__v--low">' + esc(U.num(p.pct_low, 1)) + '%</div>' +
        '<div class="tile__sub mono">' + esc(U.num(p.v_low, 0)) + ' mm³</div></div>' +
      '<div class="tile"><div class="tile__k">Medyan</div><div class="tile__v tile__v--mid">' + esc(U.num(p.pct_median, 1)) + '%</div>' +
        '<div class="tile__sub mono">' + esc(U.num(p.v_median, 0)) + ' mm³</div></div>' +
      '<div class="tile"><div class="tile__k">En Fazla</div><div class="tile__v tile__v--high">' + esc(U.num(p.pct_high, 1)) + '%</div>' +
        '<div class="tile__sub mono">' + esc(U.num(p.v_high, 0)) + ' mm³</div></div>' +
    '</div>' +
    '<div class="fine" style="margin-top:6px;">Referans hacim: ' + esc(U.num(p.v0, 0)) + ' mm³ · ' +
      esc(U.num(p.months, 1)) + ' aylık projeksiyon</div>';
  }

  function growthBodyHTML(b) {
    var p = b.projection || null;
    var head =
      '<div class="fine" style="margin-bottom:8px;">' +
        'Ziyaret: ' + esc(U.orDash(b.n_visits)) +
        ' · eğri: ' + esc(U.orDash(b.best_model)) +
        ' (r=' + esc(U.num(b.best_r, 4)) + ')' +
        ' · uyum: ' + esc(U.orDash(b.fit_status)) +
        ' · RANO grubu: ' + esc(U.orDash(b.rano_group)) +
        (b.rano_group_n !== undefined ? ' (n=' + esc(U.orDash(b.rano_group_n)) + ')' : '') +
        (b.rano_group_interpretable !== undefined
          ? ' · grup yorumlanabilir: ' + (b.rano_group_interpretable ? 'EVET' : 'HAYIR')
          : '') +
      '</div>';

    /* 2026-09-16 (Barış kararı): ham projeksiyon gerekçesi (interpretable=False,
       IQR kuralı, CLAUDE.md atfı) ve "BU BIR SIMULASYONDUR ... docstring'i"
       notu ekrandan kaldırıldı. Yerine tek cümle. */
    var proj = p
      ? projectionTiles(p)
      : '<div class="block-unavailable">Bu hasta için projeksiyon aralığı üretilemedi.</div>';

    return head + proj;
  }

  function growthSummaryCard(s) {
    var a = s.analysis;
    var g = a && a.growth_simulation;
    var id = s.route.patientId;
    return '' +
    '<div class="card">' +
      '<div class="card__head">' +
        '<div class="h2" style="margin:0;">Tümör Nasıl İlerleyecek? — Projeksiyon</div>' +
        '<button class="linkish" data-act="nav" data-hash="#/gecmis/' + esc(encodeURIComponent(id || '')) + '">Hasta Geçmişini Görüntüle &rarr;</button>' +
      '</div>' +
      blockBody({
        status: s.analysisStatus, block: g, reason: s.analysisReason, skelLines: 4,
        hook: 'POST /analyze_patient -> growth_simulation',
        what: 'Büyüme simülasyonu çalışıyor (tam zincir)',
        renderFn: function (b) { return growthBodyHTML(b) + hookTag('POST /analyze_patient -> growth_simulation'); }
      }) +
    '</div>';
  }

  /* ------------------------------------------------------------ MR KESİTİ */

  function mriCard(s) {
    var mr = s.mr;
    var meta = mr.meta || {};
    var maxZ = (meta.nSlices !== null && meta.nSlices !== undefined) ? (meta.nSlices - 1) : null;
    var curZ = (mr.z !== null && mr.z !== undefined) ? mr.z
             : (meta.sliceIndex !== null && meta.sliceIndex !== undefined ? meta.sliceIndex : '');

    var controls =
      '<div class="mri-controls">' +
        '<label class="mri-ctl">' +
          '<span>Kesit (z)' + (maxZ !== null ? ' · 0–' + esc(maxZ) : '') + '</span>' +
          '<input type="range" id="mr-z" data-act="mr-z" min="0" ' +
            'max="' + esc(maxZ === null ? 200 : maxZ) + '" value="' + esc(curZ === '' ? 0 : curZ) + '"' +
            (maxZ === null ? ' disabled' : '') + '>' +
          '<span class="mono mri-ctl__val">' + esc(curZ === '' ? '—' : curZ) + '</span>' +
        '</label>' +
        '<label class="mri-ctl mri-ctl--row">' +
          '<input type="checkbox" data-act="mr-overlay"' + (mr.overlay ? ' checked' : '') + '>' +
          '<span>Segmentasyon maskesini bindirme (<code>overlay</code>)</span>' +
        '</label>' +
        '<button class="btn btn--ghost btn--sm" data-act="mr-reset">En geniş tümör kesitine dön</button>' +
        '<button class="btn btn--ghost btn--sm" data-act="mr-reload">Yeniden yükle</button>' +
      '</div>';

    var body;
    if (mr.status === 'idle') {
      body = '<div class="block-unavailable">MR kesiti henüz istenmedi.' +
             '<div style="margin-top:8px;"><button class="btn btn--primary btn--sm" data-act="mr-reload">MR kesitini yükle</button></div></div>';
    } else if (mr.status === 'loading') {
      body = '<div class="mri-thumb mri-thumb--loading"></div>' +
             '<div class="block-wait" data-ticker="MR kesiti NAS üzerinden okunuyor"></div>';
    } else if (mr.status === 'missing' || mr.status === 'error') {
      // 🔴 GÜRÜLTÜLÜ hata — sessiz boş kutu YOK.
      /* 2026-09-16 (Barış kararı): ham sunucu gerekçesi (NAS yolu, HTTP kodları,
         olası-nedenler listesi) ekrandan kaldırıldı — tek cümle yeterli. */
      body = '<div class="block-error"><b>MR kesiti gösterilemiyor.</b></div>';
    } else {
      body = '<div class="mri-grid">' +
          '<div>' +
            '<div class="mri-thumb"><img src="' + esc(mr.objectUrl) + '" alt="' +
              esc(s.route.patientId) + ' T1ce eksenel kesit ' + esc(U.orDash(meta.sliceIndex)) + '"></div>' +
            '<div class="mri-label">T1ce · kesit ' + esc(U.orDash(meta.sliceIndex)) +
              ' / ' + esc(U.orDash(meta.nSlices)) + '</div>' +
          '</div>' +
        '</div>' +
        '<div class="fine" style="margin-top:8px;">' +
          /* `scan_id` 2026-09-16'da kaldırıldı — ham veritabanı kimliği. */
          'Kaynak: ' + esc(U.orDash(meta.source)) +
          ' · en geniş tümör kesiti: ' + esc(U.orDash(meta.bestSlice)) +
          ' · maske: ' + esc(U.orDash(meta.maskSource)) +
          ' · bindirme uygulandı: ' + esc(U.orDash(meta.overlay)) +
          (meta.lumiereTp ? ' · LUMIERE zaman noktası: ' + esc(meta.lumiereTp) : '') +
        '</div>';
    }

    return '' +
    '<div class="card">' +
      '<div class="h2">MR Görüntüleme</div>' +
      controls + body +
      hookTag('GET /patient/{id}/mr_slice?z=..&overlay=..') +
    '</div>';
  }

  /* ---------------------------------------------------------- LİTERATÜR */

  function pubmedLink(pmid) {
    var p = String(pmid || '').replace(/[^0-9]/g, '');
    if (!p) return esc(pmid);
    return '<a class="pmid-link mono" href="https://pubmed.ncbi.nlm.nih.gov/' + esc(p) + '/" ' +
           'target="_blank" rel="noopener noreferrer">PMID ' + esc(p) + ' &#8599;</a>';
  }

  function literatureBodyHTML(b) {
    var srcs = b.sources || [];
    var violation = !!b.citation_grounding_violation;
    var sanitizeViolation = !!b.sanitization_violation;

    var warn = '';
    if (violation) {
      warn += '<div class="block-error" style="margin-bottom:12px;">' +
        '<b>Türkçe özet gösterilmiyor.</b> Özet, aşağıdaki kaynaklarda bulunmayan bir atıf ' +
        'içerdiği için gizlendi; PubMed kayıtları olduğu gibi verilmiştir.</div>';
    }
    if (sanitizeViolation) {
      warn += '<div class="block-error" style="margin-bottom:12px;">' +
        '<b>Bu bölümde bir güvenlik denetimi uyarısı oluştu.</b></div>';
    }

    var summary = '';
    if (!violation && b.summary_tr) {
      summary = '<div class="note-box" style="margin-top:12px;">' + esc(b.summary_tr) + '</div>';
    } else if (!violation && !b.summary_tr) {
      summary = '<div class="block-unavailable" style="margin-top:12px;">Türkçe özet üretilmedi.</div>';
    }

    var cards = srcs.length
      ? srcs.map(function (src) {
          var abs = src.abstract_original ? String(src.abstract_original) : '';
          return '<div class="pmid-card">' +
            '<div class="pmid-card__head">' + pubmedLink(src.pmid) +
              (src.retrieved_chunk_count !== undefined
                ? '<span class="fine">alınan parça: ' + esc(src.retrieved_chunk_count) + '</span>' : '') +
            '</div>' +
            '<div class="pmid-card__title">' + esc(src.title_original || '(başlık yok)') + '</div>' +
            (abs
              ? '<details class="pmid-card__abs"><summary>Özgün özeti (abstract) göster</summary>' +
                '<div class="pmid-card__abs-body">' + esc(abs) + '</div></details>'
              : '<div class="fine">Bu kayıt için özgün abstract döndürülmedi.</div>') +
          '</div>';
        }).join('')
      : '<div class="block-unavailable">PubMed kaynağı döndürülmedi.</div>';

    /* 2026-09-16 (Barış kararı): literatür meta bloğu KOMPLE kaldırıldı — model
       adı, gecikme, önbellek durumu, atıf verilen PMID listesi, sorumluluk reddi
       ve omics notları (`egfr_amplification: ... degerlendirilemiyor`) ekranda
       sergilenecek şeyler değil. PMID kartları ve Türkçe özet KALDI. */
    return warn + cards + summary;
  }

  function literatureCard(s) {
    var a = s.analysis;
    var lit = a && a.literature;
    return '' +
    '<div class="card" id="lit-chat-block">' +
      '<div class="h2">Literatür</div>' +
      blockBody({
        status: s.analysisStatus, block: lit, reason: s.analysisReason, skelLines: 4,
        hook: 'POST /analyze_patient -> literature',
        what: 'PubMed araması ve özet üretimi çalışıyor',
        renderFn: function (b) { return literatureBodyHTML(b) + hookTag('POST /analyze_patient -> literature'); }
      }) +

      '<div class="label-kicker" style="margin-top:18px;">Literatür Asistanına Soru Sor</div>' +
      '<div class="chatwrap">' +
        '<div class="chatlog" id="lit-chatlog" aria-live="polite"></div>' +
        '<div class="preset-list" id="lit-presets"></div>' +
      '</div>' +
    '</div>';
  }

  /* -------------------------------------------------------------- XGBOOST */

  function xgbStream(s) {
    if (s.predictStatus === 'ready' && s.predict && s.predict.xgboost) {
      return { status: 'ready', block: s.predict.xgboost, reason: '', hook: 'POST /predict/{id} -> xgboost' };
    }
    if (s.predictStatus === 'loading') {
      return { status: 'loading', block: null, reason: '', hook: 'POST /predict/{id} -> xgboost' };
    }
    if (s.analysisStatus === 'ready' && s.analysis) {
      return { status: 'ready', block: s.analysis.xgboost_12mo_survival, reason: '',
               hook: 'POST /analyze_patient -> xgboost_12mo_survival' };
    }
    return { status: s.predictStatus, block: null, reason: s.predictReason,
             hook: 'POST /predict/{id} -> xgboost' };
  }

  function xgbCard(s) {
    var st = xgbStream(s);   // 🔴 blok anahtarı `xgboost_12mo_survival`, `xgboost_recurrence` DEĞİL
    return '' +
    '<div class="card">' +
      '<div class="card__head">' +
        '<div class="h2" style="margin:0;">İkinci Katman (XGBoost)</div>' +
        /* "shadow — klinik karara esas alınmaz" rozetinin İKİNCİ kopyası
           2026-09-16'da kaldırıldı (Barış kararı). */
      '</div>' +
      blockBody({
        status: st.status, block: st.block, reason: st.reason, skelLines: 3,
        hook: st.hook, what: 'XGBoost gölge modeli çalışıyor',
        renderFn: function (b) {
          var p = b.probability_12_month_survival;
          var pctText = (p === undefined || p === null || isNaN(Number(p)))
            ? '—' : U.num(Number(p) * 100, 1) + '%';
          return '' +
          '<div class="big__k">12 Aylık Sağkalım Sınıflandırıcı Çıktısı</div>' +
          '<div class="big__v big__v--accent" style="margin-bottom:6px;">' + esc(pctText) + '</div>' +
          '<div class="fine" style="margin-bottom:14px;">Kalibrasyonu ÖLÇÜLMEMİŞ bir sınıflandırıcı çıktısıdır; ' +
            'klinik bir olasılık gibi okunmamalıdır.</div>' +
          (b.direction_note ? '<div class="note-box" style="margin-bottom:8px;">' + esc(b.direction_note) + '</div>' : '') +
          (b.target_definition_note ? '<div class="note-box" style="margin-bottom:8px;">' + esc(b.target_definition_note) + '</div>' : '') +
          (b.status_note ? '<div class="note-box" style="margin-bottom:8px;">' + esc(b.status_note) + '</div>' : '') +
          '<div class="fine">Model statüsü: ' + esc(U.orDash(b.status)) +
            ' · birincil karar modeli: ' + (b.is_primary_decision_model ? 'EVET' : 'HAYIR') +
            ' · kol: ' + esc(U.orDash(b.model_arm)) +
            ' · 12-ay eşiği: ' + esc(U.orDash(b.twelve_month_threshold_days)) + ' gün</div>' +
          (b.cox_score_input
            ? '<div class="fine">Girdi Cox skoru (<code>' + esc(U.orDash(b.cox_score_input.score_column)) + '</code>): ' +
              esc(U.signed(b.cox_score_input.value, 4)) + '</div>'
            : '') +
          hookTag(st.hook);
        }
      }) +
    '</div>';
  }

  /* ------------------------------------------------------------- OMICS */

  function omicsCard(s) {
    var om = (s.predictStatus === 'ready' && s.predict && s.predict.omics) ||
             (s.analysisStatus === 'ready' && s.analysis && s.analysis.risk && s.analysis.risk.omics);
    if (!om) return '';
    if (om.available === false) {
      return '<div class="card"><div class="h2">Moleküler (Omics) Modül</div>' +
             '<div class="block-unavailable">Bu hastada omics kaydı işaretli ama skor satırı yok — ' +
             esc(om.note || 'gerekçe bildirilmedi') + '</div></div>';
    }
    var fields = [
      ['Skor sürümü', om.score_version],
      ['TMZ direnç skoru', om.tmz_resistance_score],
      ['TMZ sınıfı', om.tmz_class],
      ['Agresiflik skoru', om.aggressiveness_score],
      ['Agresiflik sınıfı', om.aggr_class],
      ['DNA onarım skoru', om.dna_repair_score],
      ['Onarım sınıfı', om.repair_class],
      ['Moleküler alt tip', om.molecular_subtype]
    ];
    return '<div class="card">' +
      '<div class="card__head"><div class="h2" style="margin:0;">Moleküler (Omics) Modül</div>' +
      '<div class="badge badge--neutral">opsiyonel · Cox\'un zorunlu girdisi değil</div></div>' +
      '<div class="kv-grid">' + fields.map(function (f) {
        return '<div><div class="kv__k">' + esc(f[0]) + '</div>' + esc(U.orDash(f[1])) + '</div>';
      }).join('') + '</div>' +
      (om.interpretation_note ? '<div class="note-box" style="margin-top:12px;">' + esc(om.interpretation_note) + '</div>' : '') +
      hookTag('POST /predict/{id} -> omics') +
    '</div>';
  }

  function analysis(s) {
    var id = s.route.patientId;
    if (!id) {
      return '<div class="page page--wide">' + searchCard(s) +
             '<div class="card"><div class="block-unavailable">Analiz için yukarıdan bir hasta seçin veya kimlik girin.</div></div></div>';
    }
    return '<div class="page page--wide">' +
      searchCard(s) +
      identityCard(s) +
      unavailableStrip(s) +
      riskCard(s) +
      similarCard(s) +
      growthSummaryCard(s) +
      mriCard(s) +
      literatureCard(s) +
      xgbCard(s) +
      omicsCard(s) +
      '<p style="text-align:center;margin-top:28px;font-size:13px;">' +
        '<button class="linkish" data-act="nav" data-hash="#/sistem/performans">Bu sonuçlar nasıl üretildi? &rarr; Sistem Hakkında</button>' +
      '</p>' +
    '</div>';
  }

  /* =======================================================================
     EKRAN 3 — HASTA GEÇMİŞİ
     ======================================================================= */

  /**
   * Longitudinal hacim eğrisi.
   * @param {Array} visits [{week, volume_mm3, rano}]  — GET /patient/{id}/volume_series
   */
  function growthChartSVG(visits) {
    var pts = visits.filter(function (v) {
      return !isNaN(Number(v.volume_mm3)) && !isNaN(Number(v.week));
    });
    if (pts.length < 2) return '';

    var vols = pts.map(function (v) { return Number(v.volume_mm3); });
    var weeks = pts.map(function (v) { return Number(v.week); });
    var minV = Math.min.apply(null, vols), maxV = Math.max.apply(null, vols);
    var pad = (maxV - minV) * 0.12 || Math.max(1, maxV * 0.12);
    minV -= pad; maxV += pad;
    var minW = Math.min.apply(null, weeks), maxW = Math.max.apply(null, weeks);
    var spanW = (maxW - minW) || 1;

    var xOf = function (w) { return 48 + ((Number(w) - minW) / spanW) * 512; };
    var yOf = function (v) { return 180 - ((Number(v) - minV) / (maxV - minV || 1)) * 150; };

    var colors = { 'Baseline': '#3b82f6', 'CR': '#16a34a', 'PR': '#16a34a', 'SD': '#f59e0b', 'PD': '#dc2626' };
    var line = pts.map(function (v) { return xOf(v.week) + ',' + yOf(v.volume_mm3); }).join(' ');

    return '<svg class="growth-chart" viewBox="0 0 600 220" role="img" aria-label="Longitudinal tümör hacmi eğrisi">' +
      '<line x1="48" y1="180" x2="580" y2="180" stroke="#e2e8f0" stroke-width="1"></line>' +
      '<line x1="48" y1="20" x2="48" y2="180" stroke="#e2e8f0" stroke-width="1"></line>' +
      '<polyline points="' + esc(line) + '" fill="none" stroke="#6d28d9" stroke-width="2.5"></polyline>' +
      pts.map(function (v) {
        var c = colors[String(v.rano || '').split(' ')[0]] || '#6d28d9';
        return '<circle cx="' + xOf(v.week) + '" cy="' + yOf(v.volume_mm3) + '" r="5.5" fill="' + esc(c) + '"></circle>' +
               '<text x="' + xOf(v.week) + '" y="' + (yOf(v.volume_mm3) - 11) + '" font-size="10" fill="#475569" text-anchor="middle" font-family="monospace">' +
                 esc(U.num(v.volume_mm3)) + '</text>' +
               '<text x="' + xOf(v.week) + '" y="196" font-size="10" fill="#94a3b8" text-anchor="middle">h' + esc(v.week) + '</text>';
      }).join('') +
    '</svg>';
  }

  function volumeSeriesBody(s) {
    return blockBody({
      status: s.volumeStatus, block: s.volume, reason: s.volumeReason, skelLines: 6,
      hook: 'GET /patient/{id}/volume_series',
      what: 'Longitudinal hacim serisi çekiliyor',
      renderFn: function (b) {
        var visits = b.visits || [];
        var head = '<div class="fine" style="margin-bottom:12px;">Ziyaret sayısı: ' +
          esc(U.orDash(b.n_visits !== undefined ? b.n_visits : visits.length)) + '</div>';

        if (!visits.length) {
          return head + '<div class="block-unavailable">Ziyaret listesi boş döndü.' +
                 hookTag('GET /patient/{id}/volume_series') + '</div>';
        }
        return head + growthChartSVG(visits) +
          '<div class="growth-legend">' +
            '<span><span style="color:#3b82f6;">&#9679;</span> Baseline</span>' +
            '<span><span style="color:#16a34a;">&#9679;</span> CR / PR</span>' +
            '<span><span style="color:#f59e0b;">&#9679;</span> SD</span>' +
            '<span><span style="color:#dc2626;">&#9679;</span> PD</span>' +
          '</div>' +
          '<div class="trend-table" style="margin-top:18px;">' +
            '<div class="trend-head"><div>Hafta</div><div>Tümör Hacmi</div><div>RANO Durumu</div></div>' +
            visits.map(function (v) {
              return '<div class="trend-row"><div>' + esc(U.orDash(v.week)) + '</div>' +
                     '<div class="mono">' + esc(U.num(v.volume_mm3)) + ' mm&sup3;</div>' +
                     '<div>' + esc(U.orDash(v.rano)) + '</div></div>';
            }).join('') +
          '</div>' +
          '<div class="fine" style="margin-top:8px;">RANO etiketi VOLUMETRİK bir türetmedir; resmî RANO değerlendirmesi DEĞİLDİR.</div>' +
          hookTag('GET /patient/{id}/volume_series');
      }
    });
  }

  function history(s) {
    var id = s.route.patientId;
    var a = s.analysis;
    var g = a && a.growth_simulation;

    if (!id) {
      return '<div class="page page--narrow">' +
        '<h1 class="h1">Hasta Geçmişi</h1>' +
        '<div class="card"><div class="block-unavailable">Önce Hasta Analizi ekranından bir hasta seçin.</div></div></div>';
    }

    return '<div class="page page--narrow">' +
      '<h1 class="h1">Hasta Geçmişi</h1>' +
      '<div class="mono" style="font-size:13px;color:#6d28d9;margin-bottom:24px;">' + esc(id) + '</div>' +

      '<div class="card">' +
        '<div class="h2">Longitudinal Tümör Hacmi Takibi</div>' +
        volumeSeriesBody(s) +
      '</div>' +

      '<div class="card">' +
        '<div class="h2">Büyüme Modeli ve Projeksiyon</div>' +
        blockBody({
          status: s.analysisStatus, block: g, reason: s.analysisReason, skelLines: 4,
          hook: 'POST /analyze_patient -> growth_simulation',
          what: 'Büyüme simülasyonu çalışıyor (tam zincir)',
          renderFn: function (b) { return growthBodyHTML(b) + hookTag('POST /analyze_patient -> growth_simulation'); }
        }) +
      '</div>' +
    '</div>';
  }

  /* =======================================================================
     EKRAN 4 — KOHORT GEZGİNİ
     ======================================================================= */

  /**
   * Rol-farkındalıklı sayaçlar.
   * 🔴 SABİT SAYI YOKTUR — hepsi `GET /patients -> counters` alanından gelir.
   * Rol etiketleri de API'nin döndürdüğü metindir.
   */
  function cohortCounters(s) {
    var c = (s.cohort && s.cohort.counters) || (s.directory && s.directory.counters) || null;
    if (!c || !Array.isArray(c.by_source)) {
      var msg = (s.cohortStatus === 'loading' || s.directoryStatus === 'loading')
        ? 'Sayaçlar yükleniyor&hellip;'
        : 'Sayaçlar alınamadı.';
      return '<div class="counter-grid"><div class="counter"><div class="counter__name">' + msg + '</div></div></div>';
    }
    return '<div class="counter-grid">' + c.by_source.map(function (row) {
      return '<div class="counter">' +
               '<div class="counter__name">' + esc(row.source) + '</div>' +
               '<div class="counter__role">' + esc(row.role || '') + '</div>' +
               '<div class="counter__metrics">' + counterMetrics(row) + '</div>' +
             '</div>';   /* row.note (n_has_omics_total açıklaması) 2026-09-16'da kaldırıldı */
    }).join('') + '</div>' +
    /* 2026-09-16 (Barış kararı): "Servis edilebilir toplam + formül", sunucudan
       gelen `note` (B8 / n_has_omics_total açıklaması) ve rol paragrafı
       kaldırıldı. Sayaç kartları KALDI. */
    '';
  }

  function cohort(s) {
    var f = s.cohortFilters;

    var filterBar = '<div class="filters">' +
      selectHTML('cohort-source', 'Kaynak — tümü',
                 ['UPenn-GBM', 'UCSF-PDGM', 'LUMIERE', 'TCGA-GBM', 'TCGA-Omics'], f.source, 'Kaynak filtresi') +
      selectHTML('cohort-mgmt', 'MGMT — tümü', ['Methylated', 'Unmethylated'], f.mgmt, 'MGMT filtresi') +
      selectHTML('cohort-idh', 'IDH1 — tümü', ['Wildtype', 'Mutated', 'NOS/NEC'], f.idh, 'IDH1 filtresi') +
      selectHTML('cohort-gender', 'Cinsiyet — tümü', ['F', 'M'], f.gender, 'Cinsiyet filtresi') +
      selectHTML('cohort-gtr', 'GTR >%90 — tümü', ['Y', 'N'], f.gtr, 'GTR filtresi') +
    '</div>';   /* filtre açıklama paragrafı 2026-09-16'da kaldırıldı (Barış kararı) */

    var body;
    if (s.cohortStatus === 'loading') {
      body = '<div class="card">' + U.skeleton(8) +
             '<div class="block-wait" data-ticker="Hasta listesi çekiliyor"></div></div>';
    } else if (s.cohortStatus === 'missing') {
      body = '<div class="card"><div class="block-unavailable"><b>Hasta listesi gösterilemiyor.</b></div></div>';
    } else if (s.cohortStatus === 'error') {
      body = '<div class="card"><div class="block-error"><b>Hasta listesi alınamadı.</b></div></div>';
    } else if (!s.cohort) {
      body = '<div class="card"><div class="block-unavailable">Liste henüz istenmedi.</div></div>';
    } else {
      var rows = s.cohort.patients || [];
      var pg = s.cohort.pagination || { limit: s.cohortPageSize, offset: 0, total_matching: rows.length };
      var totalPages = Math.max(1, Math.ceil((pg.total_matching || 0) / (pg.limit || s.cohortPageSize)));
      var page = Math.floor((pg.offset || 0) / (pg.limit || s.cohortPageSize));

      body =
        ((s.cohort.warnings && s.cohort.warnings.length)
          ? '<div class="unavail-strip">' + s.cohort.warnings.map(function (w) {
              return '<div>&#9888; ' + esc(w) + '</div>';
            }).join('') + '</div>'
          : '') +
        '<div class="cohort-table">' +
          '<div class="cohort-head"><div>Hasta ID</div><div>Kaynak</div><div>Rol</div><div>Yaş</div>' +
          '<div>Cinsiyet</div><div>MGMT</div><div>IDH1</div><div>GTR</div><div>Vital</div></div>' +
          rows.map(function (r) {
            return '<button class="cohort-row" data-act="pick-patient" data-id="' + esc(r.patient_id) + '">' +
              '<span class="cohort-row__id mono">' + esc(r.patient_id) + '</span>' +
              '<span>' + esc(U.orDash(r.source)) + '</span>' +
              '<span class="cohort-row__role">' + esc(U.orDash(r.role)) + '</span>' +
              '<span>' + esc(U.orDash(r.age)) + '</span>' +
              '<span>' + esc(U.orDash(r.gender || r.sex)) + '</span>' +
              '<span>' + esc(U.orDash(r.mgmt_status)) + '</span>' +
              '<span>' + esc(U.orDash(r.idh1_status)) + '</span>' +
              '<span>' + esc(U.orDash(r.gtr_over90percent)) + '</span>' +
              '<span>' + esc(U.orDash(r.vital_status)) + '</span>' +
            '</button>';
          }).join('') +
        '</div>' +
        '<div class="cohort-cards">' +
          rows.map(function (r) {
            return '<button class="cohort-card" data-act="pick-patient" data-id="' + esc(r.patient_id) + '">' +
              '<span class="cohort-card__top"><b class="mono" style="color:#6d28d9;">' + esc(r.patient_id) + '</b>' +
              '<span class="fine">' + esc(U.orDash(r.source)) + '</span></span>' +
              '<span class="cohort-row__role">' + esc(U.orDash(r.role)) + '</span>' +
              '<span class="cohort-card__grid">' +
                '<span>Yaş: ' + esc(U.orDash(r.age)) + '</span><span>Cinsiyet: ' + esc(U.orDash(r.gender || r.sex)) + '</span>' +
                '<span>MGMT: ' + esc(U.orDash(r.mgmt_status)) + '</span><span>IDH1: ' + esc(U.orDash(r.idh1_status)) + '</span>' +
                '<span>GTR: ' + esc(U.orDash(r.gtr_over90percent)) + '</span><span>Vital: ' + esc(U.orDash(r.vital_status)) + '</span>' +
              '</span>' +
            '</button>';
          }).join('') +
        '</div>' +
        (rows.length ? '' : '<div class="card"><div class="block-unavailable">Filtrelerle eşleşen satır yok.</div></div>') +
        '<div class="pager">' +
          '<button class="linkish" data-act="cohort-prev"' + (page <= 0 ? ' disabled' : '') + '>&larr; Önceki</button>' +
          '<span class="pager__label">Sayfa ' + (page + 1) + ' / ' + totalPages + ' · ' +
            esc(U.num(pg.total_matching)) + ' eşleşen satır</span>' +
          '<button class="linkish" data-act="cohort-next"' + (page + 1 >= totalPages ? ' disabled' : '') + '>Sonraki &rarr;</button>' +
        '</div>' +
        /* `filter_semantics_note` ("idh1_status/mgmt_status filtreleri HAM DB
           değeriyle case-insensitive TAM eşleşme yapar -- /similar'ın karar-34
           kanonikleştirmesi…") 2026-09-16'da ekrandan kaldırıldı (Barış kararı).
           API alanı DEĞİŞMEDİ, yalnız çizilmiyor. */
        '';
    }

    return '<div class="page page--xwide">' +
      '<h1 class="h1">Kohort Gezgini</h1>' +
      '<p class="lede">Sistemin tek bir hasta üzerinde değil, ölçekte çalıştığını gösteren kohort görünümü.</p>' +
      cohortCounters(s) + filterBar + body +
    '</div>';
  }

  function selectHTML(act, placeholder, options, value, label) {
    return '<select data-act="' + esc(act) + '" aria-label="' + esc(label) + '">' +
      '<option value="">' + esc(placeholder) + '</option>' +
      options.map(function (o) {
        return '<option value="' + esc(o) + '"' + (value === o ? ' selected' : '') + '>' + esc(o) + '</option>';
      }).join('') +
    '</select>';
  }

  /* =======================================================================
     EKRAN 5 — SİSTEM HAKKINDA
     ======================================================================= */

  function system(s) {
    var tab = s.route.tab;
    var tabs = [
      ['sistem', 'Sistem'], ['performans', 'Model Performansı'],
      ['siniflar', 'Sınıflar / Veri Setleri']
    ];

    var head = '<div class="subtabs" role="tablist">' + tabs.map(function (t) {
      return '<button class="subtab" role="tab" aria-selected="' + (tab === t[0]) + '"' +
             ' data-act="nav" data-hash="#/sistem/' + t[0] + '">' + esc(t[1]) + '</button>';
    }).join('') + '</div>';

    var body = '';
    if (tab === 'performans') body = systemPerformance(s);
    else if (tab === 'siniflar') body = cohortCounters(s);
    else body = systemAbout();

    return '<div class="page page--mid">' +
      '<h1 class="h1">Sistem Hakkında</h1>' +
      '<p class="lede">GBM-AID\'nin nasıl çalıştığı, hangi yöntemleri kullandığı, model performansı ve veri setleri.</p>' +
      head + body +
    '</div>';
  }

  function systemAbout() {
    return '<div class="h2">Sistem</div>' +
      '<p class="prose">GBM-AID, glioblastoma hastaları için MR görüntülerinden çıkarılan radyomik özellikler ve klinik değişkenlerle sağkalım riski sıralayan bir karar destek prototipidir. Kullanıcı bir hasta kimliği girer; sistem beş blok döndürür: risk skoru ve bu skoru hangi özelliklerin oluşturduğu (SHAP), benzer hastalar, tümör büyüme simülasyonu, ilgili literatür, ve ikinci katman model çıktısı.</p>' +
      '<p class="prose">Bu bileşenler bağımsız bloklardır: biri kullanılamadığında diğerleri çalışmaya devam eder. Kullanılamayan her blok, gerekçesiyle birlikte ekranda gösterilir.</p>';
  }

  /* =====================================================================
     /model_performance — GERÇEK sözleşme (api/model_performance.py):
       { cox_variants:[ ...ham CSV sütunları... , secilen_mi, td_auc ],
         calibration:[ {metrik,t_gun,deger,ci_alt,ci_ust,kaynak} ],
         xgboost:{ variant, role_note, oof_auc:{auc,ci_lower,ci_upper,
                   n_patients,n_positive}, ucsf_external_auc:{...},
                   ucsf_twelve_month_report:{n_total,...,pct_excluded},
                   reporting_rule_note, n_upenn_training_* },
         source_artifacts:{...}, notes:[...] }
     `cox_variants` satırları CSV'den geldiği için DEĞERLER METİNDİR;
     `U.num()` sayıya çevirir, çeviremezse '—' verir (uydurmaz).
     ⚠️ Kol adları CLAUDE.md'deki kısa adlarla AYNI DEĞİLDİR
     (`v1` ↔ `v1_referans`); tablo API satırlarından çizilir, kilitli liste
     yalnız "8 kolun hepsi geldi mi" çapraz denetimi için kullanılır.
     ===================================================================== */

  function normalizePerformance(data) {
    if (!data) return { variants: [], calibration: [], xgb: null, artifacts: null, notes: [], raw: null };
    return {
      variants: data.cox_variants || [],
      calibration: data.calibration || [],
      xgb: data.xgboost || null,
      artifacts: data.source_artifacts || null,
      notes: data.notes || [],
      raw: data
    };
  }

  /** {auc, ci_lower, ci_upper} -> "0,7237 [0,6508–0,7903]" */
  function aucCI(o, digits) {
    if (!o) return '—';
    return fmtCI(o.auc, o.ci_lower, o.ci_upper, digits === undefined ? 4 : digits);
  }

  /** Kilitli 8 kolun hepsi API yanıtında var mı? Eksik varsa GÖRÜNÜR uyarı. */
  function armCrossCheck(variants) {
    var seen = {};
    variants.forEach(function (v) { if (v && v.variant) seen[v.variant] = true; });
    var missing = GBM.decl.COX_ARMS.filter(function (a) { return !seen[a.api]; });
    var extra = variants.filter(function (v) {
      return v && v.variant && !GBM.decl.COX_ARMS.some(function (a) { return a.api === v.variant; });
    }).map(function (v) { return v.variant; });

    /* 2026-09-16 (Barış kararı): bu bir GELİŞTİRİCİ denetimidir — beklenen kol
       listesiyle API yanıtı uyuşmazsa uyarır. Metni jüri ekranında göstermek
       ("CLAUDE.md'de kilitli olan şu kol(lar)…") sitenin amacına aykırı.
       Denetim KALDIRILMADI, yalnız hedefi konsola alındı: geliştirici görür,
       ziyaretçi görmez. */
    if (!missing.length && !extra.length) return '';
    if (typeof console !== 'undefined' && console.warn) {
      if (missing.length) {
        console.warn('[GBM] /model_performance yanıtında eksik kol(lar):',
                     missing.map(function (a) { return a.api; }).join(', '));
      }
      if (extra.length) {
        console.warn('[GBM] /model_performance beklenmeyen kol(lar) döndürdü:', extra.join(', '));
      }
    }
    return '';
  }

  function tdAucTable(variants) {
    var any = variants.some(function (v) { return v && v.td_auc; });
    if (!any) return '';
    var taus = ['tau365', 'tau548', 'tau730'];
    return '<div class="h2" style="margin-top:22px;">Zamana Bağlı AUC (Uno) — 365 / 548 / 730 gün</div>' +
      '<div class="variant-table">' +
        '<div class="tdauc-head"><div>Kol Adı</div><div>td-AUC 365</div><div>td-AUC 548</div><div>td-AUC 730</div><div>Kapı</div></div>' +
        variants.map(function (v) {
          if (!v.td_auc) {
            return '<div class="tdauc-row"><div class="mono">' + esc(v.variant) + '</div>' +
                   '<div style="grid-column:2 / span 4;">' +
                   esc(v.td_auc_not_available_reason || 'td-AUC bu kol için yok.') + '</div></div>';
          }
          var cells = taus.map(function (t) {
            var b = v.td_auc[t] || {};
            var ci = b.td_auc_ci;
            return '<div title="' + esc(b.td_auc_ci_raw || '') + '">' +
                   esc(fmtCI(b.td_auc, ci && ci[0], ci && ci[1], 4)) + '</div>';
          }).join('');
          return '<div class="tdauc-row"><div class="mono">' + esc(v.variant) + '</div>' + cells +
                 '<div>' + esc(U.orDash(v.td_auc.gate)) + '</div></div>';
        }).join('') +
      '</div>' +
      '<p class="fine">Nokta tahminleridir; kollar arasında eşleştirilmiş fark testi YAPILMAMIŞTIR.</p>';
  }

  function calibrationTable(rows) {
    if (!rows || !rows.length) return '';
    return '<div class="h2" style="margin-top:22px;">Kalibrasyon ve Ek Metrikler (nihai model v3b)</div>' +
      '<div class="variant-table">' +
        /* 2026-09-16 (Barış kararı): "Kaynak" sütunu KALDIRILDI — içeriği
           log dosyası adı, script adı ve biçim açıklaması taşıyordu. */
        '<div class="calib-head calib-head--3"><div>Metrik</div><div>t (gün)</div><div>Değer [%95 GA]</div></div>' +
        rows.map(function (r) {
          return '<div class="calib-row calib-row--3">' +
            '<div class="mono">' + esc(U.orDash(r.metrik)) + '</div>' +
            '<div>' + esc(U.orDash(r.t_gun)) + '</div>' +
            '<div>' + esc(fmtCI(r.deger, r.ci_alt, r.ci_ust, 4)) + '</div>' +
          '</div>';
        }).join('') +
      '</div>';
  }

  /**
   * Sunucudan gelen serbest metinlerden dosya-yolu atıflarını temizler.
   * 2026-09-16 (Barış kararı): varyant tablosundaki "seçildi
   * (decisions/2026-09-12-....md)" gibi parantezler ekrana çıkmamalı.
   */
  function stripRefs(txt) {
    if (txt === null || txt === undefined) return txt;
    return String(txt)
      .replace(/\s*\((?:bkz\.?\s*)?[^()]*(?:\.md|\.py|\.csv|\.json|\/)[^()]*\)/gi, '')
      .replace(/\s*\bbkz\.[^.;]*[.;]?/gi, '')
      .replace(/\s{2,}/g, ' ')
      .trim();
  }

  /**
   * Sunucu gerekçe metnini ziyaretçiye göstermeden önce insanîleştirir.
   * 2026-09-16 (Barış kararı).
   *
   * Sunucu gerekçeleri " -- " ile ayrılmış parçalardan oluşur ve bazı
   * parçalar SAF KOD AYRINTISIDIR (dosya yolu, sütun adı, SQL kalıbı,
   * "UYDURULMADI" gibi iç gerekçelendirme). Kodu içeren parçalar ATILIR,
   * anlamlı parçalar KORUNUR. Örnek:
   *   girdi : "'X' icin ... kayit bulunamadi -- ... (segmentation_tool='..')
   *            -- pipeline/growth_simulation.py'nin ... (`patient_id ILIKE ..`)
   *            -- Tek-zaman-noktali hastalar ... YOK -- ... DONDURULMEDI."
   *   çıktı : "'X' icin ... kayit bulunamadi. Tek-zaman-noktali hastalar ... YOK."
   */
  var CODEISH = new RegExp([
    '`',                             // kod aralığı
    '\\.(py|md|json|csv|dll)\\b',    // dosya uzantısı
    '\\b(pipeline|api|tools|db)\\/', // repo yolu
    '\\b[a-z][a-z0-9]*_[a-z0-9_]+\\b', // snake_case tanımlayıcı (shap_available, not_available_reason…)
    '[A-Za-z_]{3,}\\s*=',            // atama / kwarg
    '\\d\\s*[+=]\\s*\\d',            // "611 + 72 + 39 = 722" gibi aritmetik
    '\\bILIKE\\b|\\bSELECT\\b',      // SQL
    'DONDURULMEDI|UYDURUL',          // iç gerekçelendirme
    'OSError|shared object|Traceback',
    '\\bHTTP\\b'
  ].join('|'), 'i');

  /** Parantezi dengesiz kalmış (kırpma artığı) parçayı eler. */
  function balanced(t) {
    var o = (t.match(/\(/g) || []).length, c = (t.match(/\)/g) || []).length;
    return o === c;
  }

  function humanizeReason(txt) {
    if (txt === null || txt === undefined) return '';
    /* Önce " -- " ile, SONRA cümle sınırıyla böl: kod ayrıntısı ile anlamlı
       cümle çoğu zaman AYNI parçanın içinde, nokta ile ayrılmış hâlde gelir
       (ölçüldü: growth_simulation gerekçesi). İki düzeyde bölmezsek anlamlı
       cümle kod parçasıyla birlikte atılır. */
    var parts = [];
    String(txt).split(/\s+--\s+|\s+—\s+/).forEach(function (chunk) {
      chunk.split(/(?<=\.)\s+/).forEach(function (sent) { parts.push(sent); });
    });
    var keep = parts
      .filter(function (p) { return p.trim() && !CODEISH.test(p); })
      .map(function (p) { return stripRefs(p).replace(/[.\s]+$/, ''); })
      .filter(function (p) { return p && balanced(p); });
    if (!keep.length) return '';
    return keep.join('. ') + '.';
  }

  function systemPerformance(s) {
    var perf = normalizePerformance(s.performance);
    var status = s.performanceStatus;

    var banner = '';
    if (status === 'loading') {
      banner = '<div class="block-wait" data-ticker="Model performans tablosu çekiliyor"></div>';
    } else if (status === 'missing') {
      banner = '<div class="block-unavailable"><b>Model performans tablosu gösterilemiyor.</b></div>';
    } else if (status === 'error') {
      banner = '<div class="block-error"><b>Model performans tablosu alınamadı.</b></div>';
    }

    var variants = perf.variants;
    var sel = variants.filter(function (v) { return v && v.secilen_mi; })[0] || null;

    var rows = variants.length
      ? variants.map(function (v) {
          var isSel = !!v.secilen_mi;
          var inner = (v.ic_cv_ort !== undefined && v.ic_cv_ort !== null && v.ic_cv_ort !== '')
            ? U.num(v.ic_cv_ort, 4) + (v.ic_cv_std ? ' ± ' + U.num(v.ic_cv_std, 4) : '')
            : '—';
          return '<div class="variant-row' + (isSel ? ' variant-row--selected' : '') + '">' +
            '<div class="mono">' + esc(U.orDash(v.variant)) + '</div>' +
            '<div class="variant-row__desc">' + esc(U.orDash(v.tanim)) + '</div>' +
            '<div>' + esc(U.orDash(v.rol)) + '</div>' +
            '<div>' + esc(inner) + '</div>' +
            '<div>' + esc(fmtCI(v.harici_c_index, v.ci_alt, v.ci_ust, 4)) + '</div>' +
            '<div>' + esc(U.orDash(v.harici_n)) + ' / ' + esc(U.orDash(v.harici_olay)) + '</div>' +
            '<div>' + esc(U.orDash(v.egitim_havuzu)) + '</div>' +
            '<div class="variant-row__flag ' + (isSel ? 'variant-row__flag--selected' : '') + '">' +
              esc(stripRefs(U.orDash(v.secim_durumu))) + '</div>' +
          '</div>';
        }).join('')
      : '<div class="variant-row"><div style="grid-column:1 / -1;">Kol satırı yok — ' +
        esc(status === 'ready' ? 'sonuç gelmedi.' : 'veri henüz gelmedi.') + '</div></div>';

    var xgb = perf.xgb;
    var rep = xgb && xgb.ucsf_twelve_month_report;

    return '' +
    '<div class="h2">Model Performansı</div>' +
    banner +

    '<div class="selected-model">' +
      '<div class="selected-model__kicker">Seçilen model (birincil kol)</div>' +
      '<div class="kv-grid">' +
        '<div><div class="kv__k">Kol</div><span class="mono">' +
          esc(sel ? U.orDash(sel.variant) : '—') + '</span></div>' +
        '<div><div class="kv__k">Harici test C + %95 CI</div>' +
          esc(sel ? fmtCI(sel.harici_c_index, sel.ci_alt, sel.ci_ust, 4) : '—') + '</div>' +
        '<div><div class="kv__k">İç CV</div>' +
          esc(sel && sel.ic_cv_ort ? U.num(sel.ic_cv_ort, 4) + ' ± ' + U.num(sel.ic_cv_std, 4) : '—') + '</div>' +
        '<div><div class="kv__k">Eğitim havuzu</div>' + esc(sel ? U.orDash(sel.egitim_havuzu) : '—') + '</div>' +
        '<div><div class="kv__k">Harici test (n / olay)</div>' +
          esc(sel ? U.orDash(sel.harici_n) + ' / ' + U.orDash(sel.harici_olay) : '—') + '</div>' +
        '<div><div class="kv__k">Aday özellik sayısı</div>' + esc(sel ? U.orDash(sel.aday_ozellik) : '—') + '</div>' +
      '</div>' +
      (sel && sel.tanim ? '<p class="fine" style="margin-top:10px;">Tanım: ' + esc(sel.tanim) + '</p>' : '') +
      /* EPV/seçim gerekçesi paragrafı 2026-09-16'da kaldırıldı (Barış kararı) —
         rapora ve sunuma aittir. */
    '</div>' +

    '<div class="h2">Varyant Tablosu — koşulan TÜM Cox kolları</div>' +
    armCrossCheck(variants) +
    '<div class="variant-table">' +
      '<div class="variant-head"><div>Kol Adı</div><div>Tanım</div><div>Rol</div><div>İç CV</div>' +
      '<div>Harici C + %95 CI</div><div>Harici n / olay</div><div>Eğitim havuzu</div><div>Seçim Durumu</div></div>' +
      rows +
    '</div>' +
    '<p class="fine">Koşulan HER kol bu tabloda kalır. WT+TC kolları (v2c, v3c) 609/583 havuzuyla eğitilmiştir; ' +
      'diğerleri 611/585. Nokta tahmini yanında %95 bootstrap güven aralığı olmadan hiçbir skor yorumlanmaz.</p>' +

    tdAucTable(variants) +
    calibrationTable(perf.calibration) +

    '<div class="card" style="margin-top:22px;">' +
      '<div class="card__head"><div class="label-kicker" style="margin:0;">İkinci Katman — XGBoost' +
        (xgb && xgb.variant ? ' (' + esc(xgb.variant) + ')' : '') + '</div>' +
      '</div>' +   /* "shadow — klinik karara esas alınmaz" rozeti 2026-09-16'da kaldırıldı */
      '<div class="kv-grid">' +
        '<div><div class="kv__k">Harici AUC + %95 GA (UCSF)</div>' + esc(aucCI(xgb && xgb.ucsf_external_auc)) + '</div>' +
        '<div><div class="kv__k">n (UCSF harici)</div>' +
          esc(xgb && xgb.ucsf_external_auc ? U.orDash(xgb.ucsf_external_auc.n_patients) : '—') + '</div>' +
        '<div><div class="kv__k">İç OOF AUC + %95 GA</div>' + esc(aucCI(xgb && xgb.oof_auc)) + '</div>' +
        '<div><div class="kv__k">n (iç OOF)</div>' +
          esc(xgb && xgb.oof_auc ? U.orDash(xgb.oof_auc.n_patients) : '—') + '</div>' +
      '</div>' +
    '</div>' +
    /* "12-ay hedefi: toplam N · Yes · No · dışlanan" satırı ve komple
       "Kaynak Artefaktlar" listesi (CSV/JSON dosya yolları) 2026-09-16'da
       kaldırıldı — Barış kararı. */

    // ⬇️ İki SVG grafik (paralel-41, 2026-09-16). Veri kaynağı GET /model_curves.
    // Çizim kodu `app/charts.js` içindedir; burada yalnız GÖMÜLÜR.
    // Eski "statik görsel henüz üretilmedi" yer tutucuları KALDIRILDI çünkü
    // grafikler artık canlı veriden çiziliyor. Hasta bazlı SHAP AYRI bir
    // bölümdür (Hasta Analizi ekranı) — buradaki katsayı grafiği SHAP DEĞİLDİR.
    GBM.charts.section(s);
  }

  /* `systemLimits()` ve "Sınırlar" sekmesi 2026-09-16'da KALDIRILDI (Barış
     kararı): site bir vitrindir; zorunlu beyanlar rapora, sunuma ve basılı
     ekte jüriye verilecek belgeye aittir. `GBM.decl` modülü SİLİNMEDİ —
     `featureLabel()`, `COX_ARMS` ve `COHORT_ROLES` işlevsel olarak kullanılıyor. */

  /* =================================================================== */

  GBM.views = {
    NAV: NAV, navHTML: navHTML, suggestionsHTML: suggestionsHTML,
    home: home, analysis: analysis, history: history, cohort: cohort, system: system,
    normalizePerformance: normalizePerformance, fmtCI: fmtCI, blockBody: blockBody
  };
})(window);
