/* ===========================================================================
   GBM.state — uygulama durumu + hash tabanlı yönlendirme.

   AŞAMA 3-B (2026-09-16): her veri akışının KENDİ durumu vardır.
   Sebep: `/predict` 32-86 sn, `/analyze_patient` daha uzun sürer. Tek bir
   "analysisStatus" ile beklenirse hızlı gelen risk bloğu da geç boyanırdı.
   Akışlar:
     directory   GET  /patients            (tüm satırlar, dizin)
     cohort      GET  /patients            (filtreli sayfa, Kohort Gezgini)
     predict     POST /predict/{id}        (risk + 24 SHAP + XGBoost — HIZLI)
     similar     GET  /patient/{id}/similar
     analysis    POST /analyze_patient     (büyüme + literatür — YAVAŞ)
     volume      GET  /patient/{id}/volume_series
     performance GET  /model_performance
     curves      GET  /model_curves        (kalibrasyon eğrisi + katsayılar)
     mr          GET  /patient/{id}/mr_slice

   Durum değerleri: idle | loading | ready | error | missing | na
     `missing` = uç nokta ya da kaynak 404/405 verdi (henüz yazılmamış
     olabilir). Arıza gibi DEĞİL, "yok" diye çizilir — ama SESSİZ DEĞİL.
     `na`      = sunucu 422 ile "bu hasta bu bloğa TANIM GEREĞİ girmiyor"
     dedi (UCSF-PDGM + FAISS v1 gibi). Kırmızı hata DEĞİL, ama gerekçesi
     tam metniyle gösterilir.

   Rotalar:
     #/                     Ana Sayfa
     #/hasta                Hasta Analizi (hasta seçilmemiş)
     #/hasta/<id>           Hasta Analizi (hasta seçili)
     #/gecmis/<id>          Hasta Geçmişi
     #/kohort               Kohort Gezgini
     #/sistem/<sekme>       Sistem Hakkında (sistem|performans|siniflar|sinirlar)
   =========================================================================== */
(function (global) {
  'use strict';

  var GBM = global.GBM = global.GBM || {};

  var state = {
    route: { page: 'home', patientId: null, tab: 'sistem' },

    /* --- Hasta dizini (GET /patients, tüm satırlar) --- */
    directory: null,            // { rows:[], byId:{}, counters, total, truncated }
    directoryStatus: 'idle',
    directoryReason: '',
    directoryLoaded: 0,         // ön-yükleme ilerlemesi

    /* --- Kohort Gezgini (GET /patients, sunucu tarafı filtre+sayfa) --- */
    cohort: null,               // { patients:[], pagination:{}, counters:{}, filters_applied:{}, warnings:[] }
    cohortStatus: 'idle',
    cohortReason: '',
    cohortFilters: { source: '', mgmt: '', idh: '', gender: '', gtr: '' },
    cohortPage: 0,
    cohortPageSize: 8,

    /* --- Hızlı yol: POST /predict/{id} --- */
    predict: null,
    predictStatus: 'idle',
    predictReason: '',

    /* --- Benzer hastalar: GET /patient/{id}/similar --- */
    similar: null,
    similarStatus: 'idle',
    similarReason: '',

    /* --- Tam zincir: POST /analyze_patient --- */
    analysis: null,
    analysisStatus: 'idle',
    analysisReason: '',

    /* --- Longitudinal hacim: GET /patient/{id}/volume_series --- */
    volume: null,
    volumeStatus: 'idle',
    volumeReason: '',

    /* --- Model performansı: GET /model_performance --- */
    performance: null,
    performanceStatus: 'idle',
    performanceReason: '',

    /* --- Model eğrileri: GET /model_curves (kalibrasyon + katsayılar) --- */
    curves: null,
    curvesStatus: 'idle',
    curvesReason: '',

    /* --- MR kesiti: GET /patient/{id}/mr_slice --- */
    mr: {
      status: 'idle',           // idle | loading | ready | error | missing
      reason: '',
      objectUrl: null,
      meta: null,
      z: null,                  // null = sunucu en geniş tümör kesitini seçsin
      overlay: true
    },

    /* --- Benzer hasta filtreleri --- */
    simK: '5', simIdh: '', simMgmt: '',

    /* --- SHAP görünümü: en büyük N katkı. N/24 bilgisi her zaman yazılır. --- */
    shapTopN: 8,

    /* --- Arama kutusu --- */
    searchInput: '',
    searchError: '',

    /* --- Sohbetler (yalnız hazır seçenek yanıtları) --- */
    chatOpen: false,
    chatMessages: [],
    litChatMessages: []
  };

  var listeners = [];
  function subscribe(fn) { listeners.push(fn); }
  function emit() { listeners.forEach(function (fn) { fn(state); }); }

  /** Durumu birleştirip abonelere haber verir. */
  function set(patch) {
    Object.keys(patch).forEach(function (k) { state[k] = patch[k]; });
    emit();
  }

  /** Abonelere haber VERMEDEN günceller (render döngüsünü tetiklemeden). */
  function setQuiet(patch) {
    Object.keys(patch).forEach(function (k) { state[k] = patch[k]; });
  }

  /** `state.mr` alt nesnesini birleştirir. */
  function setMr(patch) {
    var next = Object.assign({}, state.mr);
    Object.keys(patch).forEach(function (k) { next[k] = patch[k]; });
    state.mr = next;
    emit();
  }

  /**
   * API zarfını (`{ok, missing, reason, data}`) durum üçlüsüne çevirir.
   * Tek yerde toplandı ki her ekran aynı sözlüğü kullansın.
   */
  function absorb(prefix, r, extract) {
    var patch = {};
    if (r.ok) {
      patch[prefix] = extract ? extract(r.data) : r.data;
      patch[prefix + 'Status'] = 'ready';
      patch[prefix + 'Reason'] = '';
    } else {
      patch[prefix] = null;
      patch[prefix + 'Status'] = r.missing ? 'missing' : 'error';
      patch[prefix + 'Reason'] = r.reason || '';
    }
    set(patch);
  }

  /* ------------------------------------------------------------- ROUTING */

  var TABS = ['sistem', 'performans', 'siniflar', 'sinirlar'];

  function parseHash() {
    var h = (global.location.hash || '#/').replace(/^#/, '');
    var parts = h.split('/').filter(function (p) { return p !== ''; });
    if (parts.length === 0) return { page: 'home', patientId: null, tab: 'sistem' };

    var head = parts[0];
    var arg = parts[1] ? decodeURIComponent(parts[1]) : null;

    if (head === 'hasta')  return { page: 'analysis', patientId: arg, tab: 'sistem' };
    if (head === 'gecmis') return { page: 'history',  patientId: arg, tab: 'sistem' };
    if (head === 'kohort') return { page: 'cohort',   patientId: null, tab: 'sistem' };
    if (head === 'sistem') return { page: 'system',   patientId: null, tab: TABS.indexOf(arg) >= 0 ? arg : 'sistem' };
    return { page: 'home', patientId: null, tab: 'sistem' };
  }

  function go(hash) {
    if (global.location.hash === hash) { syncRoute(); return; }
    global.location.hash = hash;
  }

  /** Hasta değiştiğinde TÜM hastaya-özel akışlar sıfırlanır. */
  function resetPatientStreams() {
    if (state.mr.objectUrl) GBM.api.revokeObjectUrl(state.mr.objectUrl);
    state.predict = null;  state.predictStatus = 'idle';  state.predictReason = '';
    state.similar = null;  state.similarStatus = 'idle';  state.similarReason = '';
    state.analysis = null; state.analysisStatus = 'idle'; state.analysisReason = '';
    state.volume = null;   state.volumeStatus = 'idle';   state.volumeReason = '';
    state.mr = { status: 'idle', reason: '', objectUrl: null, meta: null, z: null, overlay: true };
    state.shapTopN = 8;
    state.litChatMessages = [];
  }

  function syncRoute() {
    var r = parseHash();
    var changedPatient = r.patientId !== state.route.patientId;
    state.route = r;
    if (changedPatient) resetPatientStreams();
    emit();
  }

  /**
   * Hasta kimliği biçim doğrulaması. Serbest metin kutusunun tek savunması
   * budur: yalnız harf/rakam/tire/alt çizgi, 3-64 karakter.
   */
  function isValidPatientId(id) {
    return typeof id === 'string' && /^[A-Za-z0-9_-]{3,64}$/.test(id.trim());
  }

  /** Dizinden bir hastanın satırı (kimlik şeridi için). Yoksa null. */
  function directoryRow(patientId) {
    if (!patientId || !state.directory || !state.directory.byId) return null;
    return state.directory.byId[patientId] || null;
  }

  GBM.state = {
    data: state,
    set: set,
    setQuiet: setQuiet,
    setMr: setMr,
    absorb: absorb,
    subscribe: subscribe,
    emit: emit,
    go: go,
    syncRoute: syncRoute,
    parseHash: parseHash,
    isValidPatientId: isValidPatientId,
    directoryRow: directoryRow,
    resetPatientStreams: resetPatientStreams,
    TABS: TABS
  };
})(window);
