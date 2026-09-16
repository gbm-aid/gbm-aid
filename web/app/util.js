/* ===========================================================================
   GBM.util — küçük yardımcılar (DOM, kaçış, biçimleme).
   Klasik script; global tek isim alanı: `window.GBM`.
   =========================================================================== */
(function (global) {
  'use strict';

  var GBM = global.GBM = global.GBM || {};

  /** HTML kaçışı. Template literal ile kurulan HER dinamik değer bundan geçer. */
  function esc(v) {
    if (v === null || v === undefined) return '';
    return String(v)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  /** Attribute içine giden değer (esc + boşluk/tırnak güvenli). */
  function attr(v) { return esc(v); }

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  /** `el.innerHTML = html` + içindeki [data-on-click] düğmelerini bağlama. */
  function setHTML(el, html) { if (el) el.innerHTML = html; }

  /**
   * Olay delegasyonu: `root` üzerinde tek dinleyici; `data-act` taşıyan en
   * yakın ata bulunur ve `handlers[act](el, event)` çağrılır.
   * onClick="{{ fn }}" tasarım deseninin vanilla karşılığı budur.
   */
  function delegate(root, type, handlers) {
    root.addEventListener(type, function (ev) {
      var el = ev.target.closest('[data-act]');
      if (!el || !root.contains(el)) return;
      var fn = handlers[el.getAttribute('data-act')];
      if (typeof fn === 'function') { fn(el, ev); }
    });
  }

  /** Sayıyı tr-TR ile biçimler; sayı değilse '—'. */
  function num(v, digits) {
    if (v === null || v === undefined || v === '' || isNaN(Number(v))) return '—';
    var n = Number(v);
    return n.toLocaleString('tr-TR', {
      minimumFractionDigits: digits === undefined ? 0 : digits,
      maximumFractionDigits: digits === undefined ? 0 : digits
    });
  }

  /** İşaretli ondalık: +0,42 / −0,71 (Türkçe ondalık ayracı). */
  function signed(v, digits) {
    if (v === null || v === undefined || isNaN(Number(v))) return '—';
    var n = Number(v);
    var body = Math.abs(n).toFixed(digits === undefined ? 2 : digits).replace('.', ',');
    return (n >= 0 ? '+' : '−') + body;
  }

  /** Boş/None değerler için görünür yer tutucu. */
  function orDash(v) {
    if (v === null || v === undefined || v === '') return '—';
    return v;
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  /** İskelet yükleyici blokları (satır sayısı verilir). */
  function skeleton(lines) {
    var n = lines || 3;
    var out = '<span class="skel skel--title"></span>';
    for (var i = 0; i < n; i++) {
      out += '<span class="skel skel--line ' + (i % 3 === 2 ? 'skel--w50' : i % 3 === 1 ? 'skel--w70' : '') + '"></span>';
    }
    return out;
  }

  /**
   * Uzun bekleme metni: geçen süreyi sayar ve ne beklendiğini yazar.
   * Ölçülen gerçek gecikme: tam zincir 32-86 sn, süreç yeni açıldıysa dakikalar.
   */
  function waitTicker(el, what) {
    var t0 = Date.now();
    function tick() {
      if (!el.isConnected) { clearInterval(id); return; }
      var s = Math.round((Date.now() - t0) / 1000);
      var extra = s > 90 ? ' Sunucu süreci yeni açılmış olabilir; ilk çağrı dakikalar sürebilir.' : '';
      el.textContent = what + ' — geçen süre: ' + s + ' sn.' +
        ' Tam zincir tipik olarak 32-86 sn sürer.' + extra;
    }
    tick();
    var id = setInterval(tick, 1000);
    return function stop() { clearInterval(id); };
  }

  GBM.util = {
    esc: esc, attr: attr, $: $, $$: $$, setHTML: setHTML, delegate: delegate,
    num: num, signed: signed, orDash: orDash, clamp: clamp,
    skeleton: skeleton, waitTicker: waitTicker
  };
})(window);
