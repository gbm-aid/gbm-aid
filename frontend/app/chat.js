/* ===========================================================================
   GBM.chat — Asistan balonu + Literatür sohbeti.

   🔴 SERBEST METİN KUTUSU YOKTUR (Barış kararı, 2026-09-15).
   Gerekçe: prompt temizleme katmanında kapatılmamış 4 residuel açık var;
   serbest metin enjeksiyon yüzeyi açar. Tasarımdaki iki `<input type="text">`
   (asistan ve literatür sohbeti) BİLİNÇLİ OLARAK KALDIRILMIŞTIR.

   🔴 YANIT KAYNAĞI (2026-09-16): Backend'de `/assistant/preset` DİYE BİR UÇ
   NOKTA YOKTUR (api/ taraması: `assistant`/`preset` sıfır isabet). Bu yüzden
   hazır soruların yanıtları:
     (a) `GBM.decl` içindeki ZORUNLU BEYAN metinlerinden, ve
     (b) O AN EKRANDA OLAN hastanın gerçek API yanıtından
   okunur. Bir dil modeline serbest soru SORULMAZ, cevap UYDURULMAZ.
   Kaynağı olmayan soruya "bu soruya dayanaklı cevap veremiyorum" denir.
   =========================================================================== */
(function (global) {
  'use strict';

  var GBM = global.GBM = global.GBM || {};
  var U = GBM.util;
  var esc = U.esc;

  /* Asistan: 7 hazır seçenek — 6'sı gezinme, 1'i literatür bloğuna gider. */
  var ASSISTANT_PRESETS = [
    { id: 'go-analysis', label: 'Hasta Analizine git',          hash: '#/hasta' },
    { id: 'go-history',  label: 'Hasta Geçmişine git',          hash: '#/gecmis' },
    { id: 'go-cohort',   label: 'Kohort Gezginine git',         hash: '#/kohort' },
    { id: 'how-works',   label: 'Sistem nasıl çalışıyor?',      hash: '#/sistem/sistem' },
    { id: 'performance', label: 'Model performansını gör',      hash: '#/sistem/performans' },
    { id: 'literature',  label: 'Literatür hakkında soru sor',  scrollTo: 'lit-chat-block' }
  ];

  /* Literatür sohbeti: hazır sorular (serbest metin YOK). */
  var LIT_PRESETS = [
    { id: 'lit-sources', label: 'Bu hasta için hangi kaynaklar getirildi?' },
    { id: 'lit-mgmt',    label: 'MGMT durumu modelde nasıl kullanılıyor?' },
    { id: 'lit-idh',     label: 'IDH durumu modelde nasıl kullanılıyor?' },
    { id: 'lit-age',     label: 'Yaş bu risk skorunda ne kadar ağırlıklı?' },
    { id: 'lit-limit',   label: 'Bu literatür özetinin sınırları neler?' }
  ];

  function bubble(m) {
    return '<div class="bubble ' + (m.from === 'user' ? 'bubble--user' : 'bubble--bot') + '">' +
           esc(m.text) + '</div>';
  }

  /** Bir beyan maddesinin düz metnini döndürür (baloncuk için). */
  function declText(key) {
    var e = GBM.decl.texts[key];
    if (!e || !e.p || !e.p.length) {
      return 'Bu soruya şu an cevap veremiyorum.';
    }
    /* 2026-09-16 (Barış kararı): "(Kaynak: A10 · ZORUNLU-BEYANLAR.md)" eki kaldırıldı. */
    return e.p.join(' ');
  }

  /* ----------------------------------------------------------- ASİSTAN  */

  function renderAssistant() {
    var s = GBM.state.data;
    var body = U.$('#chat-body');
    if (!body) return;

    body.innerHTML =
      '<div class="bubble bubble--bot">Merhaba. Aşağıdan bir konu seçebilirsiniz.</div>' +
      s.chatMessages.map(bubble).join('') +
      '<div class="preset-list" style="border:0;padding:4px 0;">' +
        ASSISTANT_PRESETS.map(function (p) {
          return '<button class="preset" data-act="assistant-preset" data-preset="' + esc(p.id) + '">' +
                 esc(p.label) + '</button>';
        }).join('') +
      '</div>';
    body.scrollTop = body.scrollHeight;
  }

  function onAssistantPreset(id) {
    var p = ASSISTANT_PRESETS.filter(function (x) { return x.id === id; })[0];
    if (!p) return;

    if (p.hash) {
      GBM.state.setQuiet({ chatOpen: false });
      toggleAssistant(false);
      GBM.state.go(p.hash);
      return;
    }
    if (p.scrollTo) {
      GBM.state.setQuiet({ chatOpen: false });
      toggleAssistant(false);
      if (GBM.state.data.route.page !== 'analysis') GBM.state.go('#/hasta');
      setTimeout(function () {
        var el = document.getElementById(p.scrollTo);
        if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.pageYOffset - 90, behavior: 'smooth' });
      }, 180);
      return;
    }

    GBM.state.data.chatMessages.push({ from: 'user', text: p.label });
    /* 2026-09-16 (Barış kararı): asistan açılışı artık beyan metni DEĞİL. */
    GBM.state.data.chatMessages.push({ from: 'bot', text:
      'GBM-AID hasta analizi, benzer hastalar, tümör büyüme simülasyonu ve ilgili ' +
      'literatürü tek ekranda toplar. Yukarıdaki seçeneklerden biriyle başlayabilirsiniz.' });
    renderAssistant();
  }

  function toggleAssistant(open) {
    var panel = U.$('#chat-panel');
    var launcher = U.$('#chat-launcher');
    if (!panel || !launcher) return;
    var next = (open === undefined) ? panel.hidden : open;
    panel.hidden = !next;
    launcher.hidden = next;
    launcher.setAttribute('aria-expanded', String(next));
    GBM.state.data.chatOpen = next;
    if (next) { renderAssistant(); panel.querySelector('.chat-panel__close').focus(); }
    else { launcher.focus(); }
  }

  /* ------------------------------------------------------ LİTERATÜR SOHBETİ */

  function mountLitChat() {
    var log = U.$('#lit-chatlog');
    var presets = U.$('#lit-presets');
    if (!log || !presets) return;

    var s = GBM.state.data;
    var lit = s.analysis && s.analysis.literature;
    var intro;
    /* 2026-09-16 (Barış kararı): karşılama metinlerinden "kayıtlı beyanlar",
       "serbest metin sorulamaz" ve ham sunucu gerekçesi çıkarıldı. */
    if (s.analysisStatus === 'loading') {
      intro = 'Literatür sonuçları hâlâ geliyor. Aşağıdaki sorular şimdi de yanıtlanır.';
    } else if (lit && lit.available === false) {
      intro = 'Bu hasta için literatür sonucu üretilemedi. Aşağıdaki sorular yine de yanıtlanabilir.';
    } else {
      intro = 'Bu hastanın literatür sonuçlarına dayanan hazır sorular.';
    }

    log.innerHTML = '<div class="bubble bubble--bot">' + esc(intro) + '</div>' +
                    s.litChatMessages.map(bubble).join('');
    presets.innerHTML = LIT_PRESETS.map(function (p) {
      return '<button class="preset" data-act="lit-preset" data-preset="' + esc(p.id) + '">' + esc(p.label) + '</button>';
    }).join('');
    log.scrollTop = log.scrollHeight;
  }

  /** Hazır sorunun yanıtını GERÇEK veriden + beyanlardan kurar. */
  function litAnswer(id) {
    var s = GBM.state.data;
    var lit = (s.analysis && s.analysis.literature) || null;
    var row = GBM.state.directoryRow(s.route.patientId);

    if (id === 'lit-sources') {
      if (!lit) {
        return 'Literatür sonuçları henüz gelmedi.';
      }
      if (lit.available === false) return 'Bu hasta için literatür sonucu üretilemedi.';
      var srcs = lit.sources || [];
      if (!srcs.length) return 'Bu hasta için kaynak bulunamadı.';
      var head = srcs.length + ' PubMed kaydı getirildi: ' +
                 srcs.map(function (x) { return 'PMID ' + x.pmid; }).join(', ') + '.';
      if (lit.citation_grounding_violation || !lit.summary_tr) {
        head += ' Türkçe özet bu kayıt için gösterilmiyor.';
      }
      return head;
    }

    if (id === 'lit-mgmt') {
      var mgmt = row && row.mgmt_status ? row.mgmt_status : 'bilinmiyor / dizinde yok';
      return 'Bu hastanın kayıtlı MGMT durumu: ' + mgmt + '. ' +
             'MGMT modelin girdilerinden biridir; bilgisi olmayan hastalar da değerlendirilebilir.';
    }

    if (id === 'lit-idh') {
      var idh = row && row.idh1_status ? row.idh1_status : 'bilinmiyor / dizinde yok';
      return 'Bu hastanın kayıtlı IDH1 durumu: ' + idh + '. ' +
             'IDH bilgisi olmayan hastalar kohorttan çıkarılmaz, model onları da değerlendirir.';
    }

    if (id === 'lit-age') {
      return 'Yaş, modelin en güçlü klinik girdilerinden biridir. Radyomik özelliklerin ' +
             'yaşın üzerine ne kadar katkı verdiği ayrıca ölçülmüştür; ayrıntısı raporda yer alır.';
    }

    if (id === 'lit-limit') {
      return 'Literatür bölümü bir kanıt derlemesi değildir: PubMed\'den getirilen kayıtların ' +
             'özgün başlık ve özetleri gösterilir, Türkçe özet yalnız bu kayıtlara dayandığında ' +
             'gösterilir. Bu bölüm tedavi önerisi üretmez.';
    }

    return 'Bu soruya şu an cevap veremiyorum.';
  }

  function onLitPreset(id) {
    var p = LIT_PRESETS.filter(function (x) { return x.id === id; })[0];
    if (!p) return;
    GBM.state.data.litChatMessages.push({ from: 'user', text: p.label });
    GBM.state.data.litChatMessages.push({ from: 'bot', text: litAnswer(id) });
    mountLitChat();
  }

  GBM.chat = {
    ASSISTANT_PRESETS: ASSISTANT_PRESETS,
    LIT_PRESETS: LIT_PRESETS,
    renderAssistant: renderAssistant,
    onAssistantPreset: onAssistantPreset,
    toggleAssistant: toggleAssistant,
    mountLitChat: mountLitChat,
    onLitPreset: onLitPreset,
    litAnswer: litAnswer
  };
})(window);
