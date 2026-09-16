/* ===========================================================================
   GBM.charts — SAF SVG grafikler (2026-09-16, paralel-41).

   İKİ GRAFİK:
     1) Kalibrasyon eğrisi   — desil bazlı tahmin vs. gözlenen (Kaplan-Meier)
     2) Model katsayıları    — 24 kovaryatın hazard oranı orman grafiği
   Kaynak: `GET /model_curves` (paralel-40 tarafından yazılıyor).

   🔴 KURALLAR
     · Grafik kütüphanesi YOK, CDN YOK, derleme adımı YOK — elle <svg>.
     · UYDURMA DEĞER YOK: uç nokta 404/503 verirse GÖRÜNÜR bir "henüz
       yüklenemedi" kutusu çizilir, örnek/temsili eğri ÇİZİLMEZ.
     · Her grafiğin SAYISAL KARŞILIĞI ayrıca bir tabloda verilir
       (ekran okuyucu + "sayıyı göster" denildiğinde).
     · Renkler `styles.css`'teki mevcut paletten gelir; SVG düğümleri sınıf
       alır, renk değerleri CSS'te tutulur (yeni renk sistemi icat edilmedi).
     · Çizim yaklaşımı `views.js -> growthChartSVG()` ile aynıdır: viewBox'lu
       tek <svg>, string birleştirme, `esc()`'ten geçen her dinamik değer.

   ⚠️ ORMAN GRAFİĞİ ≠ SHAP. Bu grafik nihai modelin TAM HAVUZ fit'indeki
   katsayılarıdır (popülasyon düzeyi, tek bir sayı seti). Hasta bazlı SHAP
   Hasta Analizi ekranındaki AYRI bölümdür ve gerçek SHAP odur. Karışmasın
   diye bu grafiğin adında "SHAP" kelimesi GEÇMEZ.
   =========================================================================== */
(function (global) {
  'use strict';

  var GBM = global.GBM = global.GBM || {};
  var U = GBM.util;
  var esc = U.esc;

  /* ------------------------------------------------------------ yardımcı */

  function isNum(v) {
    return v !== null && v !== undefined && v !== '' && !isNaN(Number(v));
  }
  function nz(v) { return Number(v); }

  /**
   * "0,6850 [0,4700–0,9000]" biçimi.
   * ⚠️ `views.js` içinde aynı isimli bir yardımcı var; burada KOPYASI
   * tutuluyor çünkü charts.js views.js'ten ÖNCE yüklenir ve tek başına
   * (DOM stub testinde) de çalışabilmelidir.
   */
  function fmtCI(point, lo, hi, digits) {
    var d = digits === undefined ? 4 : digits;
    var p = U.num(point, d);
    if (p === '—') return '—';
    var l = U.num(lo, d), h = U.num(hi, d);
    if (l === '—' || h === '—') return p;
    // Katsayılar NEGATİF olabilir: "[-0,2386–-0,0567]" okunmuyor. Sınırlardan
    // biri negatifse ayraç boşluklu yazılır; her ikisi de pozitifse projenin
    // kanonik sıkı biçimi ("0,685 [0,470–0,900]") KORUNUR.
    var sep = (Number(lo) < 0 || Number(hi) < 0) ? ' – ' : '–';
    return p + ' [' + l + sep + h + ']';
  }

  /** Çok küçük p değerleri için: 0,0000 yerine "<0,0001". */
  function pFmt(v) {
    if (!isNum(v)) return '—';
    var x = nz(v);
    if (x > 0 && x < 0.0001) return '<0,0001';
    return U.num(x, 4);
  }

  /** Kartın hangi uçtan beslendiğini görünür kılar (views.js ile aynı desen). */
  /* 2026-09-16 (Barış kararı): "kaynak: GET /..." etiketi kaldırıldı — bkz.
     app/views.js'teki ikizi. İmza korundu, boş dönüyor. */
  function hookTag(hook) {
    return '';
  }

  /** Uzun radyomik adlarını kırpar — HAM ad her zaman `title`'da tam kalır. */
  function trimLabel(txt, max) {
    var t = String(txt === null || txt === undefined ? '' : txt);
    var m = max || 34;
    return t.length <= m ? t : t.slice(0, m - 1) + '…';
  }

  /* =======================================================================
     1) KALİBRASYON EĞRİSİ
     X: predicted_s365 · Y: observed_km · dikey çubuk: km_lower–km_upper
     Kesikli 45° çizgi = mükemmel kalibrasyon referansı.
     Eksenler AYNI alan adını ve KARE çizim alanını paylaşır; aksi hâlde
     "45°" görsel olarak yalan söylerdi.
     ======================================================================= */

  var CAL = { W: 520, H: 500, X0: 64, Y0: 20, SIDE: 416 };

  function calibrationSVG(cal) {
    var rows = (cal && cal.deciles ? cal.deciles : []).filter(function (d) {
      return isNum(d.predicted_s365) && isNum(d.observed_km);
    });
    if (!rows.length) return '';

    // --- alan adı: tüm noktalar + güven aralığı uçları, [0,1] içine kırpılı
    var vals = [];
    rows.forEach(function (d) {
      vals.push(nz(d.predicted_s365));
      vals.push(nz(d.observed_km));
      if (isNum(d.km_lower)) vals.push(nz(d.km_lower));
      if (isNum(d.km_upper)) vals.push(nz(d.km_upper));
    });
    var lo = Math.max(0, Math.min.apply(null, vals));
    var hi = Math.min(1, Math.max.apply(null, vals));
    var pad = (hi - lo) * 0.08 || 0.05;
    lo = Math.max(0, lo - pad);
    hi = Math.min(1, hi + pad);
    if (hi - lo < 0.1) { hi = Math.min(1, lo + 0.1); lo = Math.max(0, hi - 0.1); }
    var span = (hi - lo) || 1;

    var X0 = CAL.X0, Y0 = CAL.Y0, SIDE = CAL.SIDE;
    var X1 = X0 + SIDE, Y1 = Y0 + SIDE;
    function xOf(v) { return X0 + ((nz(v) - lo) / span) * SIDE; }
    function yOf(v) { return Y1 - ((nz(v) - lo) / span) * SIDE; }

    // --- nokta yarıçapı desilin n'ine göre (alan ~ n)
    var ns = rows.map(function (d) { return isNum(d.n) ? nz(d.n) : 0; });
    var maxN = Math.max.apply(null, ns.concat([1]));
    function rOf(n) {
      if (!isNum(n) || maxN <= 0) return 5;
      return U.clamp(3.5 + 5 * Math.sqrt(nz(n) / maxN), 3.5, 9);
    }

    // --- 5 ızgara/etiket çizgisi
    var ticks = [];
    for (var i = 0; i <= 4; i++) ticks.push(lo + (span * i) / 4);

    var grid = ticks.map(function (t) {
      var x = xOf(t), y = yOf(t);
      return '<line class="gc__grid" x1="' + x.toFixed(1) + '" y1="' + Y0 + '" x2="' + x.toFixed(1) + '" y2="' + Y1 + '"></line>' +
             '<line class="gc__grid" x1="' + X0 + '" y1="' + y.toFixed(1) + '" x2="' + X1 + '" y2="' + y.toFixed(1) + '"></line>' +
             '<text class="gc__tick" x="' + x.toFixed(1) + '" y="' + (Y1 + 18) + '" text-anchor="middle">' + esc(U.num(t, 2)) + '</text>' +
             '<text class="gc__tick" x="' + (X0 - 8) + '" y="' + (y + 3.5).toFixed(1) + '" text-anchor="end">' + esc(U.num(t, 2)) + '</text>';
    }).join('');

    var points = rows.map(function (d) {
      var x = xOf(d.predicted_s365), y = yOf(d.observed_km);
      var bar = '';
      if (isNum(d.km_lower) && isNum(d.km_upper)) {
        var yl = yOf(Math.max(lo, Math.min(hi, nz(d.km_lower))));
        var yu = yOf(Math.max(lo, Math.min(hi, nz(d.km_upper))));
        bar = '<line class="gc__ci" x1="' + x.toFixed(1) + '" y1="' + yl.toFixed(1) + '" x2="' + x.toFixed(1) + '" y2="' + yu.toFixed(1) + '"></line>' +
              '<line class="gc__ci" x1="' + (x - 4).toFixed(1) + '" y1="' + yl.toFixed(1) + '" x2="' + (x + 4).toFixed(1) + '" y2="' + yl.toFixed(1) + '"></line>' +
              '<line class="gc__ci" x1="' + (x - 4).toFixed(1) + '" y1="' + yu.toFixed(1) + '" x2="' + (x + 4).toFixed(1) + '" y2="' + yu.toFixed(1) + '"></line>';
      }
      var tip = 'Desil ' + U.orDash(d.decile) +
                ' · n=' + U.orDash(d.n) +
                ' · olay=' + U.orDash(d.events) +
                ' · tahmin=' + U.num(d.predicted_s365, 4) +
                ' · gözlenen=' + U.num(d.observed_km, 4) +
                ' · GA ' + (isNum(d.km_lower) && isNum(d.km_upper)
                             ? U.num(d.km_lower, 4) + '–' + U.num(d.km_upper, 4)
                             : 'yok');
      return '<g class="gc__pt"><title>' + esc(tip) + '</title>' + bar +
             '<circle class="gc__dot" cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) +
             '" r="' + rOf(d.n).toFixed(1) + '"></circle></g>';
    }).join('');

    var desc = 'Her nokta bir desildir (toplam ' + rows.length + '). Yatay eksen modelin tahmin ettiği ' +
               '365 günlük sağkalım olasılığı, dikey eksen Kaplan-Meier ile gözlenen orandır; dikey çubuk ' +
               'gözlenen oranın %95 güven aralığıdır. Kesikli çizgi 45° referansıdır. Sayısal karşılığı ' +
               'aşağıdaki tabloda verilmiştir.';

    return '<div class="chart-scroll">' +
      '<svg class="chart chart--calib" viewBox="0 0 ' + CAL.W + ' ' + CAL.H + '" role="img" ' +
           'aria-labelledby="calib-svg-title calib-svg-desc">' +
        '<title id="calib-svg-title">Kalibrasyon eğrisi — tahmin edilen ve gözlenen 365 günlük sağkalım</title>' +
        '<desc id="calib-svg-desc">' + esc(desc) + '</desc>' +
        grid +
        '<line class="gc__ref" x1="' + X0 + '" y1="' + Y1 + '" x2="' + X1 + '" y2="' + Y0 + '"></line>' +
        '<line class="gc__axis" x1="' + X0 + '" y1="' + Y0 + '" x2="' + X0 + '" y2="' + Y1 + '"></line>' +
        '<line class="gc__axis" x1="' + X0 + '" y1="' + Y1 + '" x2="' + X1 + '" y2="' + Y1 + '"></line>' +
        points +
        '<text class="gc__axlabel" x="' + (X0 + SIDE / 2) + '" y="' + (Y1 + 44) + '" text-anchor="middle">' +
          'Tahmin edilen 365 günlük sağkalım (model)</text>' +
        '<text class="gc__axlabel" transform="translate(18,' + (Y0 + SIDE / 2) + ') rotate(-90)" text-anchor="middle">' +
          'Gözlenen sağkalım (Kaplan-Meier)</text>' +
      '</svg>' +
    '</div>' +
    '<div class="chart-legend">' +
      '<span><span class="chart-legend__dot"></span> Desil (nokta büyüklüğü desildeki hasta sayısıyla orantılı)</span>' +
      '<span><span class="chart-legend__bar"></span> Gözlenen oranın %95 GA\'sı</span>' +
      '<span><span class="chart-legend__dash"></span> 45° referans (tahmin = gözlenen)</span>' +
    '</div>';
  }

  /** Kalibrasyon grafiğinin SAYISAL karşılığı. */
  function calibrationTableHTML(cal) {
    var rows = (cal && cal.deciles) ? cal.deciles : [];
    if (!rows.length) return '';
    return '<div class="variant-table" style="margin-top:12px;">' +
      '<div class="calibpt-head"><div>Desil</div><div>n</div><div>Olay</div>' +
      '<div>Tahmin (S365)</div><div>Gözlenen (KM)</div><div>Gözlenen %95 GA</div></div>' +
      rows.map(function (d) {
        return '<div class="calibpt-row">' +
          '<div>' + esc(U.orDash(d.decile)) + '</div>' +
          '<div>' + esc(U.orDash(d.n)) + '</div>' +
          '<div>' + esc(U.orDash(d.events)) + '</div>' +
          '<div>' + esc(U.num(d.predicted_s365, 4)) + '</div>' +
          '<div>' + esc(U.num(d.observed_km, 4)) + '</div>' +
          '<div>' + esc(isNum(d.km_lower) && isNum(d.km_upper)
                          ? U.num(d.km_lower, 4) + '–' + U.num(d.km_upper, 4) : '—') + '</div>' +
        '</div>';
      }).join('') +
    '</div>';
  }

  /* =======================================================================
     2) MODEL KATSAYILARI (HAZARD ORANI) — ORMAN GRAFİĞİ
     X ekseni LOGARİTMİKTİR (HR çarpımsal bir ölçektir; doğrusal eksende
     0,5 ile 2,0 aynı uzaklıkta görünmez). HR=1'de dikey referans çizgisi.

     SIRALAMA GEREKÇESİ: satırlar önce `kind`'a göre gruplanır (görev
     gereği radyomik/klinik ayrımı görünür olmalı), grup İÇİNDE ise HR'ye
     göre ARTAN sıralanır. Alfabetik sıralama `WT__original_glcm_*` ön
     ekleri yüzünden fiilen "özellik ailesine göre" sıralama olurdu ve
     okuyucuya hiçbir bilgi vermezdi; HR sıralaması ise koruyucu yöndeki
     (HR<1) ve riski artıran yöndeki (HR>1) katsayıları iki uçta toplar.
     HR'si olmayan satırlar grubun SONUNA konur (kaybolmasınlar diye).
     ======================================================================= */

  var FOR = { W: 760, LABEL_W: 246, X0: 300, X1: 740, ROW_H: 22, GROUP_H: 28, TOP: 40, BOTTOM: 34 };

  var KIND_LABEL = { radiomic: 'Radyomik', clinical: 'Klinik' };

  function groupRows(rows) {
    var order = ['radiomic', 'clinical'];
    var buckets = {};
    rows.forEach(function (r) {
      var k = (r && r.kind) ? String(r.kind) : 'diger';
      (buckets[k] = buckets[k] || []).push(r);
    });
    var keys = order.filter(function (k) { return buckets[k]; })
      .concat(Object.keys(buckets).filter(function (k) { return order.indexOf(k) < 0; }).sort());
    return keys.map(function (k) {
      var list = buckets[k].slice().sort(function (a, b) {
        var ha = isNum(a.hr) ? nz(a.hr) : Infinity;
        var hb = isNum(b.hr) ? nz(b.hr) : Infinity;
        return ha - hb;
      });
      return { kind: k, label: KIND_LABEL[k] || k, rows: list };
    });
  }

  /** Log eksende gösterilecek "yuvarlak" HR değerleri. 1 her zaman vardır. */
  function logTicks(lo, hi) {
    var cand = [0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 2, 3, 4, 5, 10];
    var out = cand.filter(function (t) { return t >= lo && t <= hi; });
    if (out.indexOf(1) < 0 && lo <= 1 && hi >= 1) out.push(1);
    out.sort(function (a, b) { return a - b; });
    // Çok sıkışırsa seyreltme: en fazla 8 etiket.
    while (out.length > 8) {
      out = out.filter(function (t, i) { return t === 1 || i % 2 === 0; });
    }
    return out;
  }

  function forestSVG(coef) {
    var all = (coef && coef.rows) ? coef.rows : [];
    if (!all.length) return '';
    var groups = groupRows(all);

    // --- log alan adı: HR ve GA uçlarının tümü (pozitif olanlar)
    var vals = [];
    all.forEach(function (r) {
      [r.hr, r.hr_ci_lower, r.hr_ci_upper].forEach(function (v) {
        if (isNum(v) && nz(v) > 0) vals.push(nz(v));
      });
    });
    if (!vals.length) return '';
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    lo = Math.min(lo, 1); hi = Math.max(hi, 1);
    var k = Math.pow(hi / lo, 0.06) || 1.05;   // log uzayında %6 pay
    lo = lo / k; hi = hi * k;
    var lLo = Math.log(lo), lSpan = (Math.log(hi) - lLo) || 1;

    var X0 = FOR.X0, X1 = FOR.X1, PW = X1 - X0;
    function xOf(v) { return X0 + ((Math.log(nz(v)) - lLo) / lSpan) * PW; }

    var nRows = all.length;
    var H = FOR.TOP + groups.length * FOR.GROUP_H + nRows * FOR.ROW_H + FOR.BOTTOM;

    var ticks = logTicks(lo, hi);
    var axis = ticks.map(function (t) {
      var x = xOf(t);
      return '<line class="fp__grid' + (t === 1 ? ' fp__grid--one' : '') + '" x1="' + x.toFixed(1) +
             '" y1="' + (FOR.TOP - 12) + '" x2="' + x.toFixed(1) + '" y2="' + (H - FOR.BOTTOM + 4) + '"></line>' +
             '<text class="gc__tick" x="' + x.toFixed(1) + '" y="' + (FOR.TOP - 18) + '" text-anchor="middle">' +
               esc(U.num(t, t < 1 ? 2 : (t % 1 === 0 ? 0 : 2))) + '</text>';
    }).join('');

    var y = FOR.TOP;
    var body = '';
    groups.forEach(function (g) {
      body += '<text class="fp__group" x="12" y="' + (y + 14) + '">' +
                esc(g.label + ' (' + g.rows.length + ')') + '</text>';
      y += FOR.GROUP_H;

      g.rows.forEach(function (r) {
        var cy = y + FOR.ROW_H / 2;
        var spansOne = isNum(r.hr_ci_lower) && isNum(r.hr_ci_upper) &&
                       nz(r.hr_ci_lower) <= 1 && nz(r.hr_ci_upper) >= 1;
        // GA'sı yok sayılan satırlar da SOLUK çizilir (iddia kurulmasın diye).
        var cls = (!isNum(r.hr_ci_lower) || !isNum(r.hr_ci_upper) || spansOne) ? ' fp--pale' : '';

        var raw = String(r.covariate === null || r.covariate === undefined ? '' : r.covariate);
        var human = (GBM.decl && GBM.decl.featureLabel) ? GBM.decl.featureLabel(raw) : raw;
        /* 2026-09-16 (Baris karari): ipucunda da HAM ad degil insan etiketi. */
        var tip = human +
                  ' · HR ' + fmtCI(r.hr, r.hr_ci_lower, r.hr_ci_upper, 4) +
                  ' · katsayı ' + fmtCI(r.coef, r.coef_ci_lower, r.coef_ci_upper, 4) +
                  (isNum(r.se) ? ' · SE ' + U.num(r.se, 4) : '') +
                  (isNum(r.z) ? ' · z ' + U.num(r.z, 3) : '') +
                  (isNum(r.p) ? ' · p ' + pFmt(r.p) : '');

        var bar = '';
        if (isNum(r.hr_ci_lower) && isNum(r.hr_ci_upper) && nz(r.hr_ci_lower) > 0 && nz(r.hr_ci_upper) > 0) {
          var xl = xOf(r.hr_ci_lower), xu = xOf(r.hr_ci_upper);
          bar = '<line class="fp__ci' + cls + '" x1="' + xl.toFixed(1) + '" y1="' + cy + '" x2="' + xu.toFixed(1) + '" y2="' + cy + '"></line>' +
                '<line class="fp__ci' + cls + '" x1="' + xl.toFixed(1) + '" y1="' + (cy - 4) + '" x2="' + xl.toFixed(1) + '" y2="' + (cy + 4) + '"></line>' +
                '<line class="fp__ci' + cls + '" x1="' + xu.toFixed(1) + '" y1="' + (cy - 4) + '" x2="' + xu.toFixed(1) + '" y2="' + (cy + 4) + '"></line>';
        }
        var dot = isNum(r.hr) && nz(r.hr) > 0
          ? '<circle class="fp__dot' + cls + '" cx="' + xOf(r.hr).toFixed(1) + '" cy="' + cy + '" r="3.6"></circle>'
          : '';

        body += '<g class="fp__row"><title>' + esc(tip) + '</title>' +
                  '<text class="fp__label' + cls + '" x="' + (12 + 12) + '" y="' + (cy + 3.5) + '">' +
                    esc(trimLabel(human, 32)) + '</text>' +
                  bar + dot +
                '</g>';
        y += FOR.ROW_H;
      });
      y += 4;
    });

    var desc = 'Nihai Cox modelindeki ' + all.length + ' kovaryatın hazard oranı ve %95 güven aralığı. ' +
               'Yatay eksen logaritmiktir; dikey çizgi HR=1\'dir. Güven aralığı 1\'i kapsayan satırlar soluk ' +
               'çizilmiştir. Sayısal karşılığı aşağıdaki tabloda verilmiştir.';

    return '<div class="chart-scroll">' +
      '<svg class="chart chart--forest" viewBox="0 0 ' + FOR.W + ' ' + H + '" role="img" ' +
           'aria-labelledby="forest-svg-title forest-svg-desc" style="min-width:' + FOR.W + 'px;">' +
        '<title id="forest-svg-title">Model katsayıları (hazard oranı) — orman grafiği</title>' +
        '<desc id="forest-svg-desc">' + esc(desc) + '</desc>' +
        axis + body +
        '<text class="gc__axlabel" x="' + ((X0 + X1) / 2) + '" y="' + (H - 8) + '" text-anchor="middle">' +
          'Hazard oranı (log ölçek)</text>' +
      '</svg>' +
    '</div>' +
    '<div class="chart-legend">' +
      '<span><span class="chart-legend__dot"></span> Hazard oranı noktası</span>' +
      '<span><span class="chart-legend__bar chart-legend__bar--h"></span> %95 güven aralığı</span>' +
      '<span><span class="chart-legend__pale"></span> Güven aralığı 1\'i kapsıyor (soluk çizilir)</span>' +
    '</div>';
  }

  /** Orman grafiğinin SAYISAL karşılığı. */
  function forestTableHTML(coef) {
    var rows = (coef && coef.rows) ? coef.rows : [];
    if (!rows.length) return '';
    var groups = groupRows(rows);
    return '<div class="variant-table" style="margin-top:12px;">' +
      '<div class="coef-head"><div>Kovaryat</div><div>Tür</div><div>Katsayı [%95 GA]</div>' +
      '<div>SE</div><div>HR [%95 GA]</div><div>z</div><div>p</div></div>' +
      groups.map(function (g) {
        return '<div class="coef-group">' + esc(g.label + ' — ' + g.rows.length + ' kovaryat') + '</div>' +
          g.rows.map(function (r) {
            var raw = String(r.covariate === null || r.covariate === undefined ? '' : r.covariate);
            /* 2026-09-16 (Baris karari): tabloda da HAM ad gosterilmez. */
            var human = (GBM.decl && GBM.decl.featureLabel) ? GBM.decl.featureLabel(raw) : raw;
            return '<div class="coef-row">' +
              '<div class="coef-row__name">' + esc(human) + '</div>' +
              '<div>' + esc(U.orDash(KIND_LABEL[r.kind] || r.kind)) + '</div>' +
              '<div>' + esc(fmtCI(r.coef, r.coef_ci_lower, r.coef_ci_upper, 4)) + '</div>' +
              '<div>' + esc(U.num(r.se, 4)) + '</div>' +
              '<div>' + esc(fmtCI(r.hr, r.hr_ci_lower, r.hr_ci_upper, 4)) + '</div>' +
              '<div>' + esc(U.num(r.z, 3)) + '</div>' +
              '<div>' + esc(pFmt(r.p)) + '</div>' +
            '</div>';
          }).join('');
      }).join('') +
    '</div>';
  }

  /* =======================================================================
     BÖLÜM — durum makinesi (idle | loading | ready | error | missing)
     ======================================================================= */

  var HOOK = 'GET /model_curves';

  function emptyBox(status, reason) {
    if (status === 'loading' || status === 'idle') {
      return U.skeleton(6) +
             '<div class="block-wait" data-ticker="Kalibrasyon ve katsayı eğrileri çekiliyor"></div>';
    }
    if (status === 'missing') {
      return '<div class="block-unavailable">' +
        '<b>Grafikler şu an gösterilemiyor.</b></div>';
    }
    return '<div class="block-error">' +
      '<b>Grafikler şu an gösterilemiyor.</b></div>';
  }

  /* `calibrationNotice()` 2026-09-16'da KALDIRILDI (Barış kararı): grafik altı
     beyan paragrafları siteden çıkarıldı, rapora ve sunuma aittir. Uç noktanın
     `notes` alanı DEĞİŞMEDİ — API onu döndürmeye devam ediyor, site çizmiyor. */

  function slopeLine(cal) {
    if (!cal) return '';
    var sl = cal.slope || {};
    var parts = '<div class="kv-grid" style="margin-top:12px;">' +
      '<div><div class="kv__k">Kalibrasyon eğimi + %95 GA</div>' +
        esc(fmtCI(sl.value, sl.ci_lower, sl.ci_upper, 3)) + '</div>' +
      '<div><div class="kv__k">Desil MAE</div>' + esc(U.num(cal.decile_mae, 5)) + '</div>' +
      '<div><div class="kv__k">Ufuk (t)</div>' + esc(U.orDash(cal.t_days)) + ' gün</div>' +
      '<div><div class="kv__k">Kol</div><span class="mono">' + esc(U.orDash(cal.variant)) + '</span></div>' +
    '</div>';
    /* 2026-09-16 (Barış kararı): `sl.source_note` ("TASINDI (yeniden
       hesaplanmadi) -- log/... ureten kosu: tools/... .3f formatiyla basmisti")
       ekrana YAZILMIYOR. Alan API yanıtında duruyor, site çizmiyor. */
    return parts;
  }

  /**
   * Sistem Hakkında > Model Performansı ekranına gömülen bölüm.
   * @param {object} s  GBM.state.data
   */
  function section(s) {
    var status = s.curvesStatus || 'idle';
    var c = s.curves || null;

    var head = '<div class="h2" style="margin-top:26px;">Kalibrasyon Eğrisi</div>';
    var head2 = '<div class="h2" style="margin-top:26px;">Model katsayıları (hazard oranı)</div>';

    if (status !== 'ready') {
      return head + emptyBox(status, s.curvesReason) +
             head2 + emptyBox(status, s.curvesReason);
    }

    var cal = c ? c.calibration : null;
    var coef = c ? c.coefficients : null;

    var calibBlock = '';
    var calSvg = calibrationSVG(cal);
    calibBlock += calSvg
      ? calSvg + slopeLine(cal) + calibrationTableHTML(cal) + hookTag(HOOK)
      : '<div class="block-unavailable">Yanıtta kalibrasyon desilleri yok — grafik çizilmedi ' +
        '</div>';

    var forSvg = forestSVG(coef);
    var forestBlock = forSvg
      /* "Kovaryat sayısı: 24 · kol: v3b_lowvar_v2amgmt" üst satırı
         2026-09-16'da kaldırıldı (Barış kararı). */
      ? forSvg + forestTableHTML(coef)
      : '<div class="block-unavailable">Yanıtta katsayı satırı yok — grafik çizilmedi ' +
        '</div>';

    /* Orman grafiği altındaki 4 beyan paragrafı 2026-09-16'da KALDIRILDI
       (Barış kararı) — rapora ve sunuma aittir. */

    return head + calibBlock + head2 + forestBlock;
  }

  GBM.charts = {
    section: section,
    calibrationSVG: calibrationSVG,
    calibrationTableHTML: calibrationTableHTML,
    forestSVG: forestSVG,
    forestTableHTML: forestTableHTML,
    groupRows: groupRows,
    logTicks: logTicks,
    HOOK: HOOK
  };
})(window);
