/* ===========================================================================
   app.js — bağlama noktası: yönlendirme, render döngüsü, olay delegasyonu,
   ve KADEMELİ VERİ ÇEKME (AŞAMA 3-B, 2026-09-16).

   Kademe sırası (ölçülmüş gecikme yüzünden):
     1) POST /predict/{id}            — risk + 24 SHAP + XGBoost, HIZLI yol
        GET  /patient/{id}/similar    — aynı anda, bağımsız
     2) /predict yanıtlandığında:
        POST /analyze_patient         — büyüme + literatür blokları (YAVAŞ)
        GET  /patient/{id}/mr_slice   — NAS'tan PNG
     3) Hasta Geçmişi ekranına girildiğinde:
        GET  /patient/{id}/volume_series
     4) Sistem > Model Performansı sekmesinde:
        GET  /model_performance
        GET  /model_curves             — kalibrasyon eğrisi + katsayı grafiği

   Her blok kendi iskelet yükleyicisini ve "ne bekleniyor" sayacını gösterir.
   Klasik script (ES module DEĞİL).
   =========================================================================== */
(function (global) {
  'use strict';

  var GBM = global.GBM;
  var U = GBM.util;
  var S = GBM.state;
  var API = GBM.api;

  var viewEl = U.$('#view');
  var stopTickers = [];

  /* ---------------------------------------------------------------- RENDER */

  function render() {
    var s = S.data;

    U.setHTML(U.$('#nav-desktop'), GBM.views.navHTML(s.route.page, s.route.patientId));
    U.setHTML(U.$('#nav-mobile'), GBM.views.navHTML(s.route.page, s.route.patientId));

    var html;
    switch (s.route.page) {
      case 'analysis': html = GBM.views.analysis(s); break;
      case 'history':  html = GBM.views.history(s);  break;
      case 'cohort':   html = GBM.views.cohort(s);   break;
      case 'system':   html = GBM.views.system(s);   break;
      default:         html = GBM.views.home(s);
    }
    U.setHTML(viewEl, html);

    // Beyan metinlerini doldur (index.html'deki sabit düğümler dahil).
    GBM.decl.hydrate(document);

    // Bekleme sayaçlarını bağla.
    stopTickers.forEach(function (fn) { fn(); });
    stopTickers = U.$$('[data-ticker]', document).map(function (el) {
      return U.waitTicker(el, el.getAttribute('data-ticker'));
    });

    GBM.chat.mountLitChat();
  }

  S.subscribe(render);

  /* ============================================================ VERİ ÇEKME */

  /** Hasta dizini — açılır liste, öneri kutusu ve kimlik şeridi için. */
  function loadDirectory() {
    if (S.data.directoryStatus === 'loading' || S.data.directoryStatus === 'ready') return;
    S.set({ directoryStatus: 'loading', directoryReason: '', directoryLoaded: 0 });

    API.loadDirectory(function (n) { S.setQuiet({ directoryLoaded: n }); }).then(function (r) {
      if (!r.ok) {
        S.set({ directory: null,
                directoryStatus: r.missing ? 'missing' : 'error',
                directoryReason: r.reason || '' });
        return;
      }
      var byId = {};
      (r.data.rows || []).forEach(function (row) { byId[row.patient_id] = row; });
      S.set({
        directory: { rows: r.data.rows || [], byId: byId, counters: r.data.counters,
                     total: r.data.total, truncated: !!r.data.truncated },
        directoryStatus: 'ready',
        directoryReason: '',
        directoryLoaded: (r.data.rows || []).length
      });
    });
  }

  /** Kohort Gezgini — sunucu tarafı filtre + sayfalama. */
  function loadCohort() {
    var s = S.data;
    var f = s.cohortFilters;
    S.set({ cohortStatus: 'loading', cohortReason: '' });
    API.listPatients({
      source: f.source || undefined,
      mgmt_status: f.mgmt || undefined,
      idh1_status: f.idh || undefined,
      gender: f.gender || undefined,
      gtr_over90percent: f.gtr || undefined,
      limit: s.cohortPageSize,
      offset: s.cohortPage * s.cohortPageSize
    }).then(function (r) {
      S.absorb('cohort', r);
    });
  }

  /** 1. kademe — hızlı yol. */
  function loadPredict(patientId) {
    if (!patientId) return Promise.resolve();
    if (S.data.predictStatus !== 'idle') return Promise.resolve();
    S.set({ predictStatus: 'loading', predictReason: '' });
    return API.predictPatient(patientId).then(function (r) {
      if (S.data.route.patientId !== patientId) return;  // kullanıcı hasta değiştirdi
      S.absorb('predict', r);
    });
  }

  /** 1. kademe — paralel. */
  function loadSimilar(patientId, force) {
    if (!patientId) return;
    if (!force && S.data.similarStatus !== 'idle') return;
    var s = S.data;
    S.set({ similarStatus: 'loading', similarReason: '' });
    API.similarPatients(patientId, { k: s.simK, idh: s.simIdh, mgmt: s.simMgmt }).then(function (r) {
      if (S.data.route.patientId !== patientId) return;
      // 🔴 422 = "hasta FAISS v1 indeksinde yok" (UCSF-PDGM'nin NORMAL hâli).
      // Bu bir arıza değil, tanım gereğidir — kırmızı hata yerine 'na'.
      if (!r.ok && r.status === 422) {
        S.set({ similar: null, similarStatus: 'na', similarReason: r.reason || '' });
        return;
      }
      S.absorb('similar', r);
    });
  }

  /** 2. kademe — yavaş tam zincir (büyüme + literatür). */
  function loadAnalysis(patientId) {
    if (!patientId) return;
    if (S.data.analysisStatus !== 'idle') return;
    S.set({ analysisStatus: 'loading', analysisReason: '' });
    API.analyzePatient(patientId).then(function (r) {
      if (S.data.route.patientId !== patientId) return;
      S.absorb('analysis', r);
    });
  }

  /** 3. kademe — Hasta Geçmişi eğrisi. */
  function loadVolume(patientId) {
    if (!patientId) return;
    if (S.data.volumeStatus !== 'idle') return;
    S.set({ volumeStatus: 'loading', volumeReason: '' });
    API.volumeSeries(patientId).then(function (r) {
      if (S.data.route.patientId !== patientId) return;
      S.absorb('volume', r);
    });
  }

  /** Model performans tablosu. */
  function loadPerformance() {
    if (S.data.performanceStatus === 'loading' || S.data.performanceStatus === 'ready') return;
    S.set({ performanceStatus: 'loading', performanceReason: '' });
    API.modelPerformance().then(function (r) { S.absorb('performance', r); });
  }

  /**
   * Model eğrileri (kalibrasyon + katsayı orman grafiği).
   * `/model_performance` ile AYNI sekmede yüklenir ama AYRI akıştır: biri
   * 404 verirse diğeri yine de çizilir.
   */
  function loadCurves() {
    if (S.data.curvesStatus === 'loading' || S.data.curvesStatus === 'ready') return;
    S.set({ curvesStatus: 'loading', curvesReason: '' });
    API.modelCurves().then(function (r) { S.absorb('curves', r); });
  }

  /** MR kesiti — PNG baytları + başlıklardaki meta veri. */
  function loadMrSlice(patientId) {
    if (!patientId) return;
    var prev = S.data.mr.objectUrl;
    S.setMr({ status: 'loading', reason: '' });
    API.mrSlice(patientId, { z: S.data.mr.z, overlay: S.data.mr.overlay }).then(function (r) {
      if (S.data.route.patientId !== patientId) {
        if (r.ok) API.revokeObjectUrl(r.objectUrl);
        return;
      }
      if (prev) API.revokeObjectUrl(prev);
      if (!r.ok) {
        S.setMr({ status: r.missing ? 'missing' : 'error', reason: r.reason || '',
                  objectUrl: null, meta: null });
        return;
      }
      S.setMr({ status: 'ready', reason: '', objectUrl: r.objectUrl, meta: r.meta,
                z: (r.meta && r.meta.sliceIndex !== null) ? r.meta.sliceIndex : S.data.mr.z });
    });
  }

  /** Bir hasta seçildiğinde kademeli zinciri başlatır. */
  function startPatientChain(patientId) {
    loadSimilar(patientId);
    loadPredict(patientId).then(function () {
      if (S.data.route.patientId !== patientId) return;
      loadAnalysis(patientId);
      if (S.data.route.page === 'analysis' && S.data.mr.status === 'idle') loadMrSlice(patientId);
    });
  }

  /* --------------------------------------------------------------- OLAYLAR */

  var handlers = {
    /* --- gezinme --- */
    'nav': function (el) {
      U.$('#nav-mobile').hidden = true;
      U.$('#nav-burger').setAttribute('aria-expanded', 'false');
      S.go(el.getAttribute('data-hash'));
    },

    /* --- hasta seçimi --- */
    'pick-patient': function (el) {
      S.go('#/hasta/' + encodeURIComponent(el.getAttribute('data-id')));
    },
    'select-patient': function (el) {
      if (el.value) S.go('#/hasta/' + encodeURIComponent(el.value));
    },
    // Yazarken TAM render yapılmaz (odak kaybolur); yalnız öneri kutusu güncellenir.
    'search-input': function (el) {
      S.setQuiet({ searchInput: el.value, searchError: '' });
      var box = U.$('#suggestions-box');
      if (box) box.innerHTML = GBM.views.suggestionsHTML(S.data);
    },
    'search-submit': function () {
      var v = (S.data.searchInput || '').trim();
      if (!v) return;
      if (!S.isValidPatientId(v)) {
        S.set({ searchError: 'Geçersiz hasta kimliği. Yalnız harf, rakam, tire ve alt çizgi (3-64 karakter).' });
        return;
      }
      S.setQuiet({ searchInput: '', searchError: '' });
      S.go('#/hasta/' + encodeURIComponent(v));
    },

    /* --- SHAP --- */
    'shap-toggle': function () {
      var s = S.data;
      var risk = (s.predict && s.predict.shap_values) ? s.predict
               : (s.analysis && s.analysis.risk ? s.analysis.risk : null);
      var total = (risk && risk.shap_values) ? Object.keys(risk.shap_values).length : 0;
      S.set({ shapTopN: s.shapTopN >= total ? 8 : total });
    },

    /* --- benzer hasta filtreleri: değişince YENİDEN ÇEK --- */
    'sim-k':    function (el) { S.setQuiet({ simK: el.value });    loadSimilar(S.data.route.patientId, true); },
    'sim-idh':  function (el) { S.setQuiet({ simIdh: el.value });  loadSimilar(S.data.route.patientId, true); },
    'sim-mgmt': function (el) { S.setQuiet({ simMgmt: el.value }); loadSimilar(S.data.route.patientId, true); },

    /* --- MR kesiti --- */
    'mr-z': function (el, ev) {
      var v = parseInt(el.value, 10);
      var lbl = el.parentNode && el.parentNode.querySelector('.mri-ctl__val');
      if (lbl) lbl.textContent = isNaN(v) ? '—' : String(v);
      // Sürüklerken her piksel için istek atılmaz; yalnız bırakıldığında çekilir.
      if (!ev || ev.type !== 'change') return;
      S.setQuiet({ mr: Object.assign({}, S.data.mr, { z: isNaN(v) ? null : v }) });
      loadMrSlice(S.data.route.patientId);
    },
    'mr-overlay': function (el) {
      S.setQuiet({ mr: Object.assign({}, S.data.mr, { overlay: !!el.checked }) });
      loadMrSlice(S.data.route.patientId);
    },
    'mr-reset': function () {
      S.setQuiet({ mr: Object.assign({}, S.data.mr, { z: null }) });
      loadMrSlice(S.data.route.patientId);
    },
    'mr-reload': function () { loadMrSlice(S.data.route.patientId); },

    /* --- kohort --- */
    'cohort-source': function (el) { setCohortFilter('source', el.value); },
    'cohort-mgmt':   function (el) { setCohortFilter('mgmt', el.value); },
    'cohort-idh':    function (el) { setCohortFilter('idh', el.value); },
    'cohort-gender': function (el) { setCohortFilter('gender', el.value); },
    'cohort-gtr':    function (el) { setCohortFilter('gtr', el.value); },
    'cohort-prev':   function () { S.setQuiet({ cohortPage: Math.max(0, S.data.cohortPage - 1) }); loadCohort(); },
    'cohort-next':   function () { S.setQuiet({ cohortPage: S.data.cohortPage + 1 }); loadCohort(); },

    /* --- sohbetler --- */
    'assistant-preset': function (el) { GBM.chat.onAssistantPreset(el.getAttribute('data-preset')); },
    'lit-preset':       function (el) { GBM.chat.onLitPreset(el.getAttribute('data-preset')); }
  };

  function setCohortFilter(key, value) {
    var f = Object.assign({}, S.data.cohortFilters);
    f[key] = value;
    S.setQuiet({ cohortFilters: f, cohortPage: 0 });
    loadCohort();
  }

  // Tek dinleyici, tüm uygulama. Form denetimleri (select/input) YALNIZ
  // change/input ile tetiklenir — aksi halde bir <select>'e tıklamak da
  // eylemi çalıştırırdı.
  var clickHandlers = {};
  Object.keys(handlers).forEach(function (k) {
    clickHandlers[k] = function (el, ev) {
      var tag = el.tagName;
      if (tag === 'SELECT' || tag === 'INPUT') return;
      handlers[k](el, ev);
    };
  });
  U.delegate(document, 'click', clickHandlers);
  U.delegate(document, 'change', handlers);
  U.delegate(document, 'input', handlers);

  // Enter ile arama.
  document.addEventListener('keydown', function (ev) {
    if (ev.key === 'Enter' && ev.target && ev.target.id === 'patient-input') {
      ev.preventDefault();
      handlers['search-submit']();
    }
    if (ev.key === 'Escape' && !U.$('#chat-panel').hidden) { GBM.chat.toggleAssistant(false); }
  });

  // Mobil menü
  U.$('#nav-burger').addEventListener('click', function () {
    var m = U.$('#nav-mobile');
    m.hidden = !m.hidden;
    this.setAttribute('aria-expanded', String(!m.hidden));
  });

  // Asistan balonu
  U.$('#chat-launcher').addEventListener('click', function () { GBM.chat.toggleAssistant(true); });
  U.$('#chat-close').addEventListener('click', function () { GBM.chat.toggleAssistant(false); });

  /* ------------------------------------------------------------ BAŞLANGIÇ */

  function onRoute() {
    S.syncRoute();
    var s = S.data;
    var page = s.route.page;

    if (page === 'home' || page === 'cohort' || page === 'analysis' || page === 'history') {
      loadDirectory();
    }
    if (page === 'cohort' && s.cohortStatus === 'idle') loadCohort();
    if (page === 'system' && s.route.tab === 'performans') { loadPerformance(); loadCurves(); }

    if ((page === 'analysis' || page === 'history') && s.route.patientId) {
      if (s.predictStatus === 'idle') startPatientChain(s.route.patientId);
      if (page === 'history' && s.volumeStatus === 'idle') loadVolume(s.route.patientId);
      if (page === 'analysis' && s.predictStatus !== 'idle' && s.mr.status === 'idle') {
        loadMrSlice(s.route.patientId);
      }
    }
  }

  global.addEventListener('hashchange', onRoute);
  onRoute();

  // Test/otomasyon için dışa aç (DOM stub ile render denemesi yapılabilsin).
  GBM.app = {
    render: render,
    loadDirectory: loadDirectory,
    loadCohort: loadCohort,
    loadPredict: loadPredict,
    loadSimilar: loadSimilar,
    loadAnalysis: loadAnalysis,
    loadVolume: loadVolume,
    loadPerformance: loadPerformance,
    loadCurves: loadCurves,
    loadMrSlice: loadMrSlice,
    handlers: handlers
  };
})(window);
