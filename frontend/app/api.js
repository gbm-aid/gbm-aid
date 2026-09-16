/* ===========================================================================
   GBM.api — backend çağrıları (AŞAMA 3-B: CANLIYA BAĞLANDI, 2026-09-16).

   Site `api/main.py` tarafından AYNI ORIGIN'den `/app` altına mount edilir
   (`app.mount("/app", StaticFiles(...))`). Bu yüzden:
     · CORS YOKTUR,
     · MUTLAK URL YAZILMAZ — uç noktalar kök-göreli ('/patients') çağrılır.

   GECİKME: ölçülen tam zincir 32-86 sn; süreç yeni açıldıysa dakikalar.
   Bu yüzden zaman aşımı 180 sn (AbortController) ve blok-blok doldurma.

   🔴 UYDURMA DEĞER ÜRETİLMEZ. Bir uç nokta yoksa/hata verirse arayüz bunu
   GÖRÜNÜR biçimde söyler; sayı tahmin edilmez.

   ⚠️ AUTH: `api/main.py` ara katmanı, ortamda `GBMAID_API_KEY` TANIMLIYSA
   her isteğe `X-API-Key` başlığı bekler. Tarayıcıya gömülecek bir anahtar
   OLMADIĞI için burada başlık GÖNDERİLMEZ; anahtar tanımlıysa tüm çağrılar
   401 döner ve arayüz bunu hata olarak gösterir (sessizce gizlemez).
   Kararı Barış verecek (paralel-36 bulgusu).
   =========================================================================== */
(function (global) {
  'use strict';

  var GBM = global.GBM = global.GBM || {};

  /** Backend kökü. Aynı origin'den servis edildiği için BOŞ. */
  var BASE_URL = '';

  /** Zaman aşımı — en az 180 sn olmak ZORUNDA (ölçülen gecikme gereği). */
  var TIMEOUT_MS = 180000;

  /** `/patients` uç noktasının izin verdiği en büyük sayfa boyutu. */
  var PATIENTS_MAX_LIMIT = 200;

  /** Dizin ön-yüklemesinde çekilecek en fazla sayfa (200 x 12 = 2400 satır). */
  var DIRECTORY_MAX_PAGES = 12;

  function qs(params) {
    var q = [];
    Object.keys(params || {}).forEach(function (k) {
      var v = params[k];
      if (v === undefined || v === null || v === '') return;
      q.push(encodeURIComponent(k) + '=' + encodeURIComponent(v));
    });
    return q.length ? '?' + q.join('&') : '';
  }

  /**
   * Zaman aşımlı fetch. JSON döner.
   * Dönen zarf: {ok, status, data} | {ok:false, status, reason, timeout?, missing?}
   * `missing:true` -> 404/405: uç nokta ya da kaynak YOK (henüz yazılmamış
   * olabilir). Arayüz bunu "arıza" yerine "bu uç henüz yok" diye çizer.
   */
  function request(path, opts) {
    opts = opts || {};
    var ctrl = new AbortController();
    var timer = setTimeout(function () { ctrl.abort(); }, TIMEOUT_MS);

    var init = {
      method: opts.method || 'GET',
      headers: { 'Accept': 'application/json' },
      signal: ctrl.signal
    };
    if (opts.body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(opts.body);
    }

    return fetch(BASE_URL + path, init)
      .then(function (res) {
        return res.json().catch(function () { return null; }).then(function (data) {
          if (!res.ok) {
            return {
              ok: false,
              status: res.status,
              missing: (res.status === 404 || res.status === 405),
              reason: detailText(data) || ('HTTP ' + res.status),
              path: path
            };
          }
          return { ok: true, status: res.status, data: data, path: path };
        });
      })
      .catch(function (err) {
        if (err && err.name === 'AbortError') {
          return { ok: false, timeout: true, path: path,
                   reason: 'İstek ' + (TIMEOUT_MS / 1000) + ' sn içinde tamamlanmadı (' + path + ').' };
        }
        return { ok: false, path: path,
                 reason: 'Ağa ulaşılamadı (' + path + '): ' + (err && err.message ? err.message : String(err)) };
      })
      .then(function (r) { clearTimeout(timer); return r; });
  }

  /** FastAPI hata gövdesi: {detail: "..."} ya da {detail:[{msg,...}]} */
  function detailText(data) {
    if (!data) return '';
    var d = data.detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) {
      return d.map(function (x) {
        return (x && (x.msg || x.message)) ? String(x.msg || x.message) : JSON.stringify(x);
      }).join(' · ');
    }
    if (d) return JSON.stringify(d);
    return '';
  }

  /* =======================================================================
     1) HASTA LİSTESİ — GET /patients
     ======================================================================= */

  /**
   * Tek sayfa. Sunucu tarafı filtre + sayfalama + rol-farkındalıklı sayaçlar.
   * Yanıt: { patients:[], pagination:{limit,offset,total_matching},
   *          filters_applied:{}, counters:{by_source:[], ...}, warnings:[] }
   * ⚠️ `limit` üst sınırı 200'dür (api/patients.py) — aşılırsa 422 döner.
   */
  function listPatients(params) {
    var p = Object.assign({}, params || {});
    if (p.limit === undefined || p.limit === null) p.limit = 8;
    p.limit = Math.min(Number(p.limit) || 8, PATIENTS_MAX_LIMIT);
    return request('/patients' + qs(p));
  }

  /**
   * Hasta DİZİNİ: açılır liste / öneri kutusu / kimlik şeridi için tüm
   * satırların ön-yüklenmesi. Uç nokta sayfa başına en fazla 200 satır
   * verdiği için ardışık sayfalar çekilir.
   * @param {function} onProgress (yüklenen, toplam)
   */
  function loadDirectory(onProgress) {
    var rows = [];
    var counters = null;
    var total = null;

    function page(offset, pageIndex) {
      return listPatients({ limit: PATIENTS_MAX_LIMIT, offset: offset }).then(function (r) {
        if (!r.ok) return r;
        var d = r.data || {};
        if (!counters && d.counters) counters = d.counters;
        if (total === null && d.pagination) total = d.pagination.total_matching;
        rows = rows.concat(d.patients || []);
        if (typeof onProgress === 'function') onProgress(rows.length, total);

        var got = (d.patients || []).length;
        var more = got === PATIENTS_MAX_LIMIT &&
                   (total === null || rows.length < total) &&
                   (pageIndex + 1) < DIRECTORY_MAX_PAGES;
        if (!more) {
          return { ok: true, status: 200, data: {
            rows: rows, counters: counters, total: total,
            truncated: (total !== null && rows.length < total)
          } };
        }
        return page(offset + PATIENTS_MAX_LIMIT, pageIndex + 1);
      });
    }
    return page(0, 0);
  }

  /* =======================================================================
     2) HIZLI YOL — POST /predict/{patient_id}
     Cox risk + 24 SHAP + XGBoost. `/analyze_patient`'in alt kümesi, HIZLI.
     🔴 patient_id YOL parametresidir, gövde YOKTUR.
     Yanıt alanları ÜST DÜZEYDEDİR (risk bloğu sarmalayıcısı YOK).
     ======================================================================= */
  function predictPatient(patientId) {
    return request('/predict/' + encodeURIComponent(patientId), { method: 'POST' });
  }

  /* =======================================================================
     3) TAM ZİNCİR — POST /analyze_patient
     Üst anahtarlar: patient_id, summary, risk, similar_patients,
     literature, growth_simulation, xgboost_12mo_survival
     ======================================================================= */
  function analyzePatient(patientId) {
    return request('/analyze_patient', { method: 'POST', body: { patient_id: patientId } });
  }

  /* =======================================================================
     4) BENZER HASTALAR — GET /patient/{id}/similar
     Yanıt: { patient_id, clinical_radiomics_faiss:{index_size, k_requested,
              k_returned, filters_applied, warnings, results:[...]},
              molecular_omics_faiss? }
     results[] : patient_id, l2_distance, source, age, gender, kps_score,
                 mgmt_status, idh1_status, gtr_over90percent, vital_status,
                 survival_days
     ⚠️ Alan adı `gender`'dır (`sex` DEĞİL).
     ======================================================================= */
  function similarPatients(patientId, opts) {
    opts = opts || {};
    return request('/patient/' + encodeURIComponent(patientId) + '/similar' + qs({
      k: opts.k || undefined,
      idh1_status: opts.idh || undefined,
      mgmt_status: opts.mgmt || undefined
    }));
  }

  /* =======================================================================
     5) LONGİTUDİNAL HACİM — GET /patient/{id}/volume_series
     Beklenen: { available, patient_id, n_visits,
                 visits:[{week, volume_mm3, rano}], not_available_reason }
     ⚠️ 2026-09-16 saat itibarıyla uç nokta HENÜZ YAZILIYOR (paralel-38).
     404 gelirse arayüz "uç nokta henüz yok" der; UYDURMA EĞRİ ÇİZİLMEZ.
     ======================================================================= */
  function volumeSeries(patientId) {
    return request('/patient/' + encodeURIComponent(patientId) + '/volume_series');
  }

  /* =======================================================================
     6) MODEL PERFORMANSI — GET /model_performance
     Beklenen: 8 Cox kolu + XGBoost satırı, artefakt kaynaklı HAM değerler.
     ⚠️ Uç nokta HENÜZ YAZILIYOR (paralel-38). Yoksa hücreler '—' kalır.
     ======================================================================= */
  function modelPerformance() {
    return request('/model_performance');
  }

  /* =======================================================================
     7) MODEL EĞRİLERİ — GET /model_curves
     Beklenen: { calibration:{variant,t_days,slope:{value,ci_lower,ci_upper,
                 source_note}, decile_mae, deciles:[10 desil]},
                 coefficients:{variant,n_covariates,rows:[24 satır]},
                 source_artifacts:{}, notes:[] }
     ⚠️ Uç nokta HENÜZ YAZILIYOR (paralel-40). 404/503 gelirse grafik alanı
     GÖRÜNÜR bir "henüz yüklenemedi" kutusu gösterir; TEMSİLÎ EĞRİ ÇİZİLMEZ.
     ======================================================================= */
  function modelCurves() {
    return request('/model_curves');
  }

  /* =======================================================================
     8) MR KESİTİ — GET /patient/{id}/mr_slice?z=..&overlay=..
     🔴 Parametreler `z` ve `overlay`'dir (`modality`/`slice` DEĞİL).
     Yanıt: image/png BAYTLARI. Meta veri YALNIZ başlıklarda:
       X-GBMAID-Source, X-GBMAID-Scan-Id, X-GBMAID-Slice-Index,
       X-GBMAID-Best-Slice, X-GBMAID-N-Slices, X-GBMAID-Mask-Source,
       X-GBMAID-Overlay-Applied, X-GBMAID-Lumiere-Timepoint
     Bu yüzden <img src> ile DEĞİL, fetch + blob ile alınır: başlıkları
     okuyabilmek ve 404/503 gövdesindeki `detail` metnini GÖSTEREBİLMEK için.
     ======================================================================= */
  function mrSlicePath(patientId, params) {
    return '/patient/' + encodeURIComponent(patientId) + '/mr_slice' + qs({
      z: (params && params.z !== undefined && params.z !== null && params.z !== '') ? params.z : undefined,
      overlay: (params && params.overlay !== undefined) ? (params.overlay ? 'true' : 'false') : undefined,
      scan_id: (params && params.scanId) ? params.scanId : undefined
    });
  }

  function mrSlice(patientId, params) {
    var path = mrSlicePath(patientId, params);
    var ctrl = new AbortController();
    var timer = setTimeout(function () { ctrl.abort(); }, TIMEOUT_MS);

    return fetch(BASE_URL + path, { headers: { 'Accept': 'image/png' }, signal: ctrl.signal })
      .then(function (res) {
        if (!res.ok) {
          return res.json().catch(function () { return null; }).then(function (data) {
            return {
              ok: false, status: res.status, path: path,
              missing: res.status === 404,
              reason: detailText(data) || ('HTTP ' + res.status)
            };
          });
        }
        var h = res.headers;
        var meta = {
          source:      h.get('X-GBMAID-Source'),
          scanId:      h.get('X-GBMAID-Scan-Id'),
          sliceIndex:  toIntOrNull(h.get('X-GBMAID-Slice-Index')),
          bestSlice:   toIntOrNull(h.get('X-GBMAID-Best-Slice')),
          nSlices:     toIntOrNull(h.get('X-GBMAID-N-Slices')),
          maskSource:  h.get('X-GBMAID-Mask-Source'),
          overlay:     h.get('X-GBMAID-Overlay-Applied'),
          lumiereTp:   h.get('X-GBMAID-Lumiere-Timepoint')
        };
        return res.blob().then(function (blob) {
          return { ok: true, status: res.status, path: path, meta: meta,
                   objectUrl: URL.createObjectURL(blob) };
        });
      })
      .catch(function (err) {
        if (err && err.name === 'AbortError') {
          return { ok: false, timeout: true, path: path,
                   reason: 'MR kesiti ' + (TIMEOUT_MS / 1000) + ' sn içinde gelmedi.' };
        }
        return { ok: false, path: path,
                 reason: 'MR kesiti alınamadı: ' + (err && err.message ? err.message : String(err)) };
      })
      .then(function (r) { clearTimeout(timer); return r; });
  }

  function toIntOrNull(v) {
    if (v === null || v === undefined || v === '') return null;
    var n = parseInt(v, 10);
    return isNaN(n) ? null : n;
  }

  function revokeObjectUrl(url) {
    if (url && typeof URL !== 'undefined' && URL.revokeObjectURL) {
      try { URL.revokeObjectURL(url); } catch (e) { /* yoksay */ }
    }
  }

  GBM.api = {
    BASE_URL: BASE_URL,
    TIMEOUT_MS: TIMEOUT_MS,
    PATIENTS_MAX_LIMIT: PATIENTS_MAX_LIMIT,
    request: request,
    detailText: detailText,
    listPatients: listPatients,
    loadDirectory: loadDirectory,
    predictPatient: predictPatient,
    analyzePatient: analyzePatient,
    similarPatients: similarPatients,
    volumeSeries: volumeSeries,
    modelPerformance: modelPerformance,
    modelCurves: modelCurves,
    mrSlice: mrSlice,
    mrSlicePath: mrSlicePath,
    revokeObjectUrl: revokeObjectUrl
  };
})(window);
