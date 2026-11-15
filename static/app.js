(() => {
  const $ = (s, r = document) => r.querySelector(s);

  /* ---------- mobile menu ---------- */
  const menuButton = $('[data-menu-toggle]'), nav = $('[data-main-nav]');
  if (menuButton && nav) {
    menuButton.addEventListener('click', () => {
      const open = nav.classList.toggle('open');
      menuButton.setAttribute('aria-expanded', String(open));
    });
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && nav.classList.contains('open')) { nav.classList.remove('open'); menuButton.setAttribute('aria-expanded', 'false'); menuButton.focus(); } });
  }

  /* ---------- tiny safe markdown (escape first) ---------- */
  const esc = s => s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const inline = s => esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  function markdown(text) {
    const out = []; let list = false;
    for (const line of String(text).split('\n')) {
      const li = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
      if (li) { if (!list) { out.push('<ul>'); list = true; } out.push(`<li>${inline(li[1])}</li>`); continue; }
      if (list) { out.push('</ul>'); list = false; }
      const h = line.match(/^#{1,6}\s+(.*)$/);
      if (h) out.push(`<h4>${inline(h[1])}</h4>`); else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
    if (list) out.push('</ul>');
    return out.join('');
  }

  async function postJSON(url, body) {
    const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    let d = {}; try { d = await r.json(); } catch { /* non-JSON error page */ }
    return { ok: r.ok, status: r.status, data: d };
  }

  /* ---------- busy state for normal forms (advisor, scanner) ---------- */
  document.querySelectorAll('[data-busy-form]').forEach(form => {
    form.addEventListener('submit', () => {
      const b = form.querySelector('button[type=submit]');
      if (b && form.checkValidity()) { b.dataset.label = b.textContent; b.textContent = b.dataset.busyText || 'Working…'; setTimeout(() => { b.disabled = true; }, 0); }
    });
  });
  // Restore the button if the page is restored from the back/forward cache.
  window.addEventListener('pageshow', e => { if (e.persisted) document.querySelectorAll('[data-busy-form] button[type=submit]').forEach(b => { b.disabled = false; if (b.dataset.label) b.textContent = b.dataset.label; }); });

  /* ---------- image upload: preview + drag and drop ---------- */
  const imageInput = $('[data-image-input]');
  if (imageInput) {
    const preview = $('[data-image-preview]'), drop = $('[data-drop]');
    const title = $('[data-upload-title]'), hint = $('[data-upload-hint]'), symbol = $('[data-upload-symbol]');
    const show = () => {
      const file = imageInput.files && imageInput.files[0];
      if (!file) return;
      if (file.size > 10 * 1024 * 1024) { title.textContent = 'File is larger than 10 MB'; hint.textContent = 'Choose a smaller photo.'; imageInput.value = ''; return; }
      if (preview.dataset.url) URL.revokeObjectURL(preview.dataset.url);
      preview.dataset.url = preview.src = URL.createObjectURL(file); preview.hidden = false;
      symbol.hidden = true; title.textContent = file.name; hint.textContent = 'Click to choose a different photo';
    };
    imageInput.addEventListener('change', show);
    if (drop) {
      ['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('drag'); }));
      ['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('drag'); }));
      drop.addEventListener('drop', e => { if (e.dataTransfer.files.length) { imageInput.files = e.dataTransfer.files; show(); } });
    }
  }

  /* ---------- schemes: summary / question ---------- */
  const legalForm = $('[data-legal-form]');
  if (legalForm) {
    legalForm.addEventListener('submit', async e => {
      e.preventDefault();
      const out = $('[data-legal-output]'), button = legalForm.querySelector('button');
      const data = Object.fromEntries(new FormData(legalForm));
      const label = button.textContent;
      button.disabled = true; button.textContent = 'Reading document…';
      out.innerHTML = '<span class="eyebrow">Answer</span><p class="muted-copy">Extracting the PDF text and preparing the explanation…</p>';
      try {
        const { ok, data: result } = await postJSON('/api/legal/summarize', data);
        if (!ok) throw new Error(result.error || result.response || 'Could not read this document.');
        out.innerHTML = `<span class="eyebrow">${data.question ? 'Answer' : 'Summary'} · ${esc(result.provider || 'AI')}</span><div class="answer-copy">${markdown(result.response)}</div>`;
      } catch (err) {
        out.innerHTML = `<div class="notice notice-error" role="alert">${esc(err.message)}</div>`;
      } finally { button.disabled = false; button.textContent = label; }
    });
  }

  /* ---------- site-wide assistant ---------- */
  const chatOpen = $('[data-chat-open]'), chatClose = $('[data-chat-close]'), chatWindow = $('[data-chat-window]');
  const chatForm = $('[data-chat-form]'), chatMessages = $('[data-chat-messages]'), chatLang = $('[data-chat-language]');
  const history = [];
  function showChat() { if (!chatWindow) return; chatWindow.hidden = false; chatOpen.hidden = true; chatForm.elements.message.focus(); }
  function hideChat() { if (!chatWindow) return; chatWindow.hidden = true; chatOpen.hidden = false; chatOpen.focus(); }
  if (chatOpen) chatOpen.addEventListener('click', showChat);
  if (chatClose) chatClose.addEventListener('click', hideChat);
  if (chatWindow) chatWindow.addEventListener('keydown', e => { if (e.key === 'Escape') hideChat(); });
  try { if (chatLang && localStorage.getItem('km-lang')) chatLang.value = localStorage.getItem('km-lang'); } catch { /* storage unavailable */ }
  if (chatLang) chatLang.addEventListener('change', () => { try { localStorage.setItem('km-lang', chatLang.value); } catch { /* ignore */ } });

  function chatLine(text, who) {
    const p = document.createElement('p'); p.className = 'chat-line ' + who; p.textContent = text;
    chatMessages.appendChild(p); chatMessages.scrollTop = chatMessages.scrollHeight; return p;
  }
  async function sendChat(message) {
    const send = chatForm.querySelector('button');
    chatLine(message, 'from-user');
    const sugg = $('[data-chat-suggestions]'); if (sugg) sugg.remove();
    const pending = chatLine('Thinking…', 'from-bot'); send.disabled = true;
    try {
      const { ok, data } = await postJSON('/api/chat', { message, context: 'general', language: chatLang ? chatLang.value : 'English', history });
      if (!ok && !data.response) throw new Error(data.error || 'Assistant request failed.');
      pending.textContent = data.response;
      if (ok) { history.push({ role: 'user', content: message }, { role: 'assistant', content: data.response, reasoning_details: data.reasoning_details || undefined }); if (history.length > 12) history.splice(0, 2); }
    } catch (err) { pending.textContent = err.message; }
    finally { send.disabled = false; chatForm.elements.message.focus(); chatMessages.scrollTop = chatMessages.scrollHeight; }
  }
  if (chatForm) {
    chatForm.addEventListener('submit', e => { e.preventDefault(); const input = chatForm.elements.message, m = input.value.trim(); if (!m) return; input.value = ''; sendChat(m); });
    document.querySelectorAll('[data-chat-suggestions] button').forEach(b => b.addEventListener('click', () => sendChat(b.textContent)));
  }

  /* ---------- rover telemetry ---------- */
  const refreshButton = $('[data-refresh-telemetry]');
  if (refreshButton) {
    const demoButton = $('[data-demo-toggle]');
    let demo = false, demoTimer = null, demoStart = Date.now();
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v ?? '—'; };
    const fmt = (v, s = '') => (v === undefined || v === null || Number.isNaN(v)) ? '—' : `${typeof v === 'number' ? +v.toFixed(2) : v}${s}`;
    const meter = (id, v) => { const el = document.getElementById(id); if (!el) return; el.style.width = (Number.isFinite(v) ? Math.max(0, Math.min(100, v)) : 0) + '%'; el.classList.toggle('low', Number.isFinite(v) && v < 20); };

    function render(d, mode) {              // mode: 'live' | 'offline' | 'demo'
      const online = mode === 'live' || mode === 'demo';
      set('rover-status', mode === 'demo' ? 'Demo preview' : online ? 'Rover connected' : 'Waiting for device');
      set('rover-last-seen', mode === 'demo' ? 'Sample values for layout preview only — not from a real rover.' : d.received_at ? 'Last packet: ' + new Date(d.received_at).toLocaleString() : 'No telemetry received yet.');
      const pill = document.getElementById('rover-pill');
      pill.className = 'connection-pill' + (mode === 'demo' ? ' demo' : online ? ' online' : '');
      pill.innerHTML = '<i></i> ' + (mode === 'demo' ? 'Demo' : online ? 'Online' : 'Offline');
      set('battery-value', fmt(d.battery_percent, '%'));
      set('battery-voltage', d.battery_voltage == null ? 'Voltage not reported' : fmt(d.battery_voltage, ' V'));
      set('tank-a', fmt(d.tank_a_percent, '%')); set('tank-b', fmt(d.tank_b_percent, '%')); set('tank-c', fmt(d.tank_c_percent, '%'));
      meter('battery-meter', d.battery_percent); meter('tank-a-meter', d.tank_a_percent); meter('tank-b-meter', d.tank_b_percent); meter('tank-c-meter', d.tank_c_percent);
      set('rover-state', d.status || '—'); set('latitude', fmt(d.latitude)); set('longitude', fmt(d.longitude));
      set('speed', fmt(d.speed_kmh, ' km/h')); set('area', fmt(d.area_ha, ' ha'));
      set('runtime', d.runtime_seconds == null ? '—' : Math.floor(d.runtime_seconds / 3600) + 'h ' + Math.floor((d.runtime_seconds % 3600) / 60) + 'm');
      const map = document.getElementById('map-link');
      if (Number.isFinite(d.latitude) && Number.isFinite(d.longitude)) { map.href = `https://www.openstreetmap.org/?mlat=${d.latitude}&mlon=${d.longitude}#map=17/${d.latitude}/${d.longitude}`; map.hidden = false; } else map.hidden = true;
      set('telemetry-updated', 'Checked ' + new Date().toLocaleTimeString() + ' · ' + (mode === 'demo' ? 'Demo data' : online ? 'Live device data' : 'No recent device packet'));
    }
    function demoData() {
      const t = (Date.now() - demoStart) / 1000;
      return { battery_percent: Math.max(5, 82 - t / 30), battery_voltage: 12.4, tank_a_percent: 90, tank_b_percent: 65, tank_c_percent: Math.max(0, 40 - t / 40), latitude: 28.61, longitude: 77.20, speed_kmh: 0.4, area_ha: 0.03 + t / 5000, runtime_seconds: 540 + t, status: 'Spraying (demo)' };
    }
    async function refresh() {
      if (demo) return render(demoData(), 'demo');
      try { const r = await fetch('/api/telemetry', { cache: 'no-store' }); const d = await r.json(); render(d, d.connected ? 'live' : 'offline'); }
      catch { set('telemetry-updated', 'Could not read telemetry endpoint.'); }
    }
    refreshButton.addEventListener('click', refresh);
    if (demoButton) demoButton.addEventListener('click', () => {
      demo = !demo; demoStart = Date.now(); demoButton.setAttribute('aria-pressed', String(demo));
      demoButton.textContent = demo ? 'Stop demo preview' : 'Preview with demo data'; refresh();
    });
    refresh(); setInterval(refresh, 5000);
    // demo values animate every second while active
    demoTimer = setInterval(() => { if (demo) render(demoData(), 'demo'); }, 1000);
  }
})();
