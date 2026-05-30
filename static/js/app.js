/* ── Meta-Design Benchmarking — Frontend ──────────────────────────────────── */
'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let columns = [];          // All column names after upload
let lastResultRow = null;  // Last SL result row from engine
let slRunning = false;

// ── Utilities ──────────────────────────────────────────────────────────────
function toast(msg, type = 'success') {
  const box  = document.getElementById('toast-box');
  const body = document.getElementById('toast-body');
  box.className = `toast align-items-center text-white border-0 bg-${type}`;
  body.textContent = msg;
  bootstrap.Toast.getOrCreateInstance(box, { delay: 3500 }).show();
}

function log(msg, cls = 'log-info') {
  const el = document.getElementById('sl-log');
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function showSpinner(btnId) {
  const b = document.getElementById(btnId);
  b.dataset.orig = b.innerHTML;
  b.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Working…';
  b.disabled = true;
}
function hideSpinner(btnId) {
  const b = document.getElementById(btnId);
  b.innerHTML = b.dataset.orig || b.innerHTML;
  b.disabled = false;
}

function getUploadParams() {
  return {
    sep:        document.querySelector('input[name="sep"]:checked')?.value || ',',
    decimal:    document.querySelector('input[name="decimal"]:checked')?.value || '.',
    skip_rows:  document.getElementById('skip-rows').value,
    erase_tab:  document.getElementById('erase-tab').checked,
    erase_quote:document.getElementById('erase-quote').checked,
    erase_pct:  document.getElementById('erase-pct').checked,
  };
}

// ── Tab switching ──────────────────────────────────────────────────────────
document.querySelectorAll('[data-tab]').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    document.querySelectorAll('[data-tab]').forEach(l => l.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    link.classList.add('active');
    document.getElementById('tab-' + link.dataset.tab).classList.add('active');
  });
});

// ── Skip rows slider ───────────────────────────────────────────────────────
document.getElementById('skip-rows').addEventListener('input', function () {
  document.getElementById('skip-val').textContent = this.value;
});

// ── Drop zone ──────────────────────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) {
    fileInput.files = e.dataTransfer.files;
    document.getElementById('file-name').textContent = e.dataTransfer.files[0].name;
  }
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0])
    document.getElementById('file-name').textContent = fileInput.files[0].name;
});

// ── Build FormData ─────────────────────────────────────────────────────────
function buildFormData(action) {
  const file = fileInput.files[0];
  if (!file) { toast('Please select a file first.', 'danger'); return null; }
  const p  = getUploadParams();
  const fd = new FormData();
  fd.append('file',        file);
  fd.append('sep',         p.sep);
  fd.append('decimal',     p.decimal);
  fd.append('skip_rows',   p.skip_rows);
  fd.append('erase_tab',   p.erase_tab);
  fd.append('erase_quote', p.erase_quote);
  fd.append('erase_pct',   p.erase_pct);
  fd.append('action',      action);
  return fd;
}

// ── Preview button ─────────────────────────────────────────────────────────
document.getElementById('btn-preview').addEventListener('click', async () => {
  const fd = buildFormData('preview');
  if (!fd) return;
  showSpinner('btn-preview');
  try {
    const r = await fetch('/api/upload', { method: 'POST', body: fd });
    const d = await r.json();
    if (d.error) { toast(d.error, 'danger'); return; }
    document.getElementById('upload-preview').innerHTML = d.preview_html;
  } catch (err) { toast(err.message, 'danger'); }
  finally { hideSpinner('btn-preview'); }
});

// ── Upload button ──────────────────────────────────────────────────────────
document.getElementById('btn-upload').addEventListener('click', async () => {
  const fd = buildFormData('upload');
  if (!fd) return;
  showSpinner('btn-upload');
  try {
    const r = await fetch('/api/upload', { method: 'POST', body: fd });
    const d = await r.json();
    if (d.error) { toast(d.error, 'danger'); return; }
    document.getElementById('upload-preview').innerHTML = d.preview_html;
    columns = d.columns || [];
    populateAllSelectors(columns);
    toast(d.message || 'Data uploaded!', 'success');
  } catch (err) { toast(err.message, 'danger'); }
  finally { hideSpinner('btn-upload'); }
});

// ── Populate selectors ─────────────────────────────────────────────────────
function populateAllSelectors(cols) {
  const selIds = ['scatter-x','scatter-y','scatter-hue','scatter-size','multi-cols'];
  selIds.forEach(id => {
    const sel = document.getElementById(id);
    sel.innerHTML = '';
    if (id.startsWith('scatter-h') || id.startsWith('scatter-s')) {
      sel.appendChild(new Option('— none —', ''));
    }
    cols.forEach(c => sel.appendChild(new Option(c, c)));
  });

  // Feature / target selectors in benchmarking tab
  ['feat-sel','targ-sel','fixed-targ-sel'].forEach(id => {
    const sel = document.getElementById(id);
    sel.innerHTML = '';
    cols.forEach(c => sel.appendChild(new Option(c, c)));
  });
}

// ── Data Info tab ──────────────────────────────────────────────────────────
document.querySelectorAll('[data-mode]').forEach(btn => {
  btn.addEventListener('click', async function () {
    document.querySelectorAll('[data-mode]').forEach(b => b.classList.remove('active-info-btn'));
    this.classList.add('active-info-btn');
    const r = await fetch('/api/data-info', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: this.dataset.mode }),
    });
    const d = await r.json();
    if (d.error) { toast(d.error, 'danger'); return; }
    document.getElementById('info-content').innerHTML = d.html;
  });
});

// ── Design Space Explorer ──────────────────────────────────────────────────
document.getElementById('graph-type').addEventListener('change', function () {
  const v  = this.value;
  const so = document.getElementById('scatter-opts');
  const mo = document.getElementById('multisel-opts');
  so.classList.toggle('d-none', v !== 'Scatter');
  mo.classList.toggle('d-none', v === 'Scatter' || v === '');
});

document.getElementById('btn-plot').addEventListener('click', async () => {
  const gtype = document.getElementById('graph-type').value;
  if (!gtype) { toast('Choose a graph type first.', 'warning'); return; }

  let payload = { graph_type: gtype };
  if (gtype === 'Scatter') {
    payload.x    = document.getElementById('scatter-x').value;
    payload.y    = document.getElementById('scatter-y').value;
    payload.hue  = document.getElementById('scatter-hue').value;
    payload.size = document.getElementById('scatter-size').value;
  } else {
    payload.columns = getSelectedValues('multi-cols');
  }

  showSpinner('btn-plot');
  try {
    const r = await fetch('/api/plot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const d = await r.json();
    if (d.error) { toast(d.error, 'danger'); return; }
    document.getElementById('plot-output').innerHTML =
      `<img src="data:image/png;base64,${d.image}" class="img-fluid rounded" alt="plot"/>`;
  } catch (err) { toast(err.message, 'danger'); }
  finally { hideSpinner('btn-plot'); }
});

// ── Benchmarking: Feature/Target selects ──────────────────────────────────
document.getElementById('targ-sel').addEventListener('change', function () {
  renderTargetControls(getSelectedValues('targ-sel'), 'targ-controls', 'target', true);
  updateFixedTargetOptions();
});

document.getElementById('fixed-targ-sel').addEventListener('change', function () {
  renderTargetControls(getSelectedValues('fixed-targ-sel'), 'fixed-targ-controls', 'fixed', true);
});

document.getElementById('feat-sel').addEventListener('change', () => {
  updateTargetOptions();
  // updateFixedTargetOptions() is called at the end of updateTargetOptions()
});

const getSelectedValues = id =>
  Array.from(document.getElementById(id).selectedOptions).map(o => o.value);

function repopulateSelect(selId, available, prevSel) {
  const sel = document.getElementById(selId);
  sel.innerHTML = '';
  available.forEach(c => {
    const opt = new Option(c, c);
    if (prevSel.includes(c)) opt.selected = true;
    sel.appendChild(opt);
  });
}

function updateTargetOptions() {
  const featSet   = new Set(getSelectedValues('feat-sel'));
  const available = columns.filter(c => !featSet.has(c));
  const prevSel   = getSelectedValues('targ-sel');
  repopulateSelect('targ-sel', available, prevSel);
  // Re-render controls only for targets still in the available list
  renderTargetControls(prevSel.filter(c => available.includes(c)), 'targ-controls', 'target', true);
  updateFixedTargetOptions();
}

function updateFixedTargetOptions() {
  const usedCols  = new Set([...getSelectedValues('feat-sel'), ...getSelectedValues('targ-sel')]);
  const available = columns.filter(c => !usedCols.has(c));
  repopulateSelect('fixed-targ-sel', available, getSelectedValues('fixed-targ-sel'));
}

/**
 * Render per-target control rows (direction, weight, optional threshold).
 * @param {string[]} targList
 * @param {string}   containerId
 * @param {string}   prefix         'target' | 'fixed'
 * @param {boolean}  withThreshold  show threshold checkbox
 */
function renderTargetControls(targList, containerId, prefix, withThreshold) {
  const container = document.getElementById(containerId);
  // Preserve existing values before clearing
  const existing = {};
  container.querySelectorAll('.target-ctrl-row').forEach(row => {
    const col = row.dataset.col;
    existing[col] = {
      dir:       row.querySelector(`input[name="${prefix}-dir-${encodeURIComponent(col)}"]:checked`)?.value || 'maximize',
      weight:    row.querySelector('[data-field="weight"]')?.value || '1',
      useThresh: row.querySelector('[data-field="use-thresh"]')?.checked || false,
      thresh:    row.querySelector('[data-field="thresh-val"]')?.value || '',
    };
  });

  container.innerHTML = '';
  targList.forEach((col, idx) => {
    const prev  = existing[col] || {};
    const uid   = `${prefix}-${containerId}-${idx}`;  // safe, index-based
    const nameR = `${prefix}-dir-${encodeURIComponent(col)}`;  // radio group name
    const div   = document.createElement('div');
    div.className   = 'target-ctrl-row';
    div.dataset.col = col;

    const threshBlock = withThreshold ? `
      <div class="d-flex align-items-center gap-2 mt-1 flex-wrap">
        <div class="form-check mb-0">
          <input class="form-check-input" type="checkbox" id="uth-${uid}"
                 data-field="use-thresh" ${prev.useThresh ? 'checked' : ''}/>
          <label class="form-check-label" for="uth-${uid}">Use threshold</label>
        </div>
        <div id="tv-${uid}" class="thresh-value ${prev.useThresh ? 'visible' : ''}">
          <input type="number" class="form-control form-control-sm"
                 data-field="thresh-val" value="${prev.thresh || ''}"
                 placeholder="threshold" style="max-width:120px"/>
        </div>
      </div>` : '';

    div.innerHTML = `
      <div class="col-label"></div>
      <div class="d-flex align-items-center gap-3 flex-wrap">
        <div class="form-check form-check-inline mb-0">
          <input class="form-check-input" type="radio" name="${nameR}"
                 id="${uid}-max" value="maximize"
                 ${(prev.dir || 'maximize') === 'maximize' ? 'checked' : ''}/>
          <label class="form-check-label" for="${uid}-max">maximize</label>
        </div>
        <div class="form-check form-check-inline mb-0">
          <input class="form-check-input" type="radio" name="${nameR}"
                 id="${uid}-min" value="minimize"
                 ${prev.dir === 'minimize' ? 'checked' : ''}/>
          <label class="form-check-label" for="${uid}-min">minimize</label>
        </div>
        <div class="d-flex align-items-center gap-1">
          <label class="small mb-0">weight:</label>
          <input type="number" class="form-control form-control-sm" data-field="weight"
                 value="${prev.weight || 1}" min="0" step="0.1" style="max-width:80px"/>
        </div>
      </div>
      ${threshBlock}
    `;
    div.querySelector('.col-label').textContent = col;
    // Wire up threshold checkbox toggle via event listener (safer than inline onchange)
    const uthEl = div.querySelector('[data-field="use-thresh"]');
    if (uthEl) {
      const tvDiv = div.querySelector('.thresh-value');
      uthEl.addEventListener('change', () => {
        if (tvDiv) tvDiv.classList.toggle('visible', uthEl.checked);
      });
    }

    container.appendChild(div);
  });
}

// ── Collect benchmarking config ────────────────────────────────────────────
function collectBenchConfig() {
  const features     = getSelectedValues('feat-sel');
  const targets      = getSelectedValues('targ-sel');
  const fixedTargets = getSelectedValues('fixed-targ-sel');
  const seedRaw      = document.getElementById('random-seed').value.trim();

  function readControls(containerId, cols, prefix) {
    const dirs = {}, weights = {}, thresholds = {};
    cols.forEach(col => {
      const nameR   = `${prefix}-dir-${encodeURIComponent(col)}`;
      const row     = document.querySelector(`#${containerId} [data-col="${CSS.escape(col)}"]`);
      const checked = row ? row.querySelector(`input[name="${nameR}"]:checked`) : null;
      dirs[col]    = checked ? checked.value : 'maximize';
      const wEl    = row ? row.querySelector('[data-field="weight"]')     : null;
      weights[col] = wEl ? parseFloat(wEl.value) || 1.0 : 1.0;
      const utEl   = row ? row.querySelector('[data-field="use-thresh"]') : null;
      const tvEl   = row ? row.querySelector('[data-field="thresh-val"]') : null;
      thresholds[col] = [
        utEl ? utEl.checked : false,
        tvEl ? parseFloat(tvEl.value) || 0 : 0,
      ];
    });
    return { dirs, weights, thresholds };
  }

  const tCtrl  = readControls('targ-controls',       targets,      'target');
  const ftCtrl = readControls('fixed-targ-controls', fixedTargets, 'fixed');

  const selectedModel = document.getElementById('model-sel').value;
  const cfg = {
    features,
    targets,
    fixed_targets:           fixedTargets,
    target_dirs:             tCtrl.dirs,
    fixed_target_dirs:       ftCtrl.dirs,
    target_weights:          tCtrl.weights,
    fixed_target_weights:    ftCtrl.weights,
    target_thresholds:       tCtrl.thresholds,
    fixed_target_thresholds: ftCtrl.thresholds,
    target_quantile:   parseInt(document.getElementById('target-quantile').value),
    init_sample_size:  parseInt(document.getElementById('init-sample').value),
    batch_size:        parseInt(document.getElementById('batch-size').value),
    n_runs:            parseInt(document.getElementById('sl-runs').value),
    sigma:             parseFloat(document.getElementById('sigma-slider').value),
    model:             selectedModel,
    strategy:          document.getElementById('strategy-sel').value,
    random_seed:       seedRaw !== '' ? parseInt(seedRaw) : null,
  };

  // Append LLM config when an LLM-based model is selected
  if (_isLLMModel(selectedModel)) {
    cfg.llm_provider    = document.querySelector('input[name="llm-provider"]:checked')?.value || 'anthropic';
    cfg.llm_api_key     = document.getElementById('llm-api-key').value.trim();
    cfg.llm_model       = document.getElementById('llm-model').value.trim();
    cfg.llm_weight      = parseFloat(document.getElementById('llm-weight').value);
    cfg.llm_max_context = parseInt(document.getElementById('llm-max-context').value) || 25;
    cfg.llm_max_batch   = parseInt(document.getElementById('llm-max-batch').value)   || 40;
  }

  return cfg;
}

// ── Target quantile slider ────────────────────────────────────────────────
document.getElementById('target-quantile').addEventListener('input', function () {
  document.getElementById('tq-val').textContent = this.value;
});

// ── SL runs slider ────────────────────────────────────────────────────────
document.getElementById('sl-runs').addEventListener('input', function () {
  document.getElementById('runs-val').textContent = this.value;
});

// ── Sigma slider ──────────────────────────────────────────────────────────
document.getElementById('sigma-slider').addEventListener('input', function () {
  document.getElementById('sigma-val').textContent = parseFloat(this.value).toFixed(1);
});

// Show / hide sigma and update its label based on strategy
document.getElementById('strategy-sel').addEventListener('change', function () {
  const v = this.value;
  const usesSigma = v.includes('MLI') || v.includes('UCB') || v.includes('Thompson');
  const usesEI    = v.includes('EI');
  document.getElementById('sigma-container').style.opacity = (usesSigma || usesEI) ? '1' : '.35';
  const lbl = v.includes('UCB') ? 'κ (UCB)' : v.includes('EI') ? 'ξ (EI xi)' : 'σ';
  document.getElementById('sigma-label').textContent = lbl;
});

// ── LLM config visibility ─────────────────────────────────────────────────
function _isLLMModel(name) {
  return (typeof LLM_MODEL_NAMES !== 'undefined') && LLM_MODEL_NAMES.includes(name);
}
function _isHybridModel(name) {
  return name.startsWith('Hybrid');
}

document.getElementById('model-sel').addEventListener('change', function () {
  const v       = this.value;
  const isLLM   = _isLLMModel(v);
  const isHybrid = _isHybridModel(v);
  document.getElementById('llm-config-card').classList.toggle('d-none', !isLLM);
  document.getElementById('llm-weight-row').classList.toggle('d-none', !isHybrid);
});

// Auto-fill default model name when provider changes
document.querySelectorAll('input[name="llm-provider"]').forEach(radio => {
  radio.addEventListener('change', function () {
    const modelInput = document.getElementById('llm-model');
    const defaults   = { anthropic: 'claude-sonnet-4-6', openai: 'gpt-4o' };
    // Only overwrite if empty or a known default (don't clobber custom entries)
    const knownDefaults = Object.values(defaults);
    if (!modelInput.value || knownDefaults.includes(modelInput.value)) {
      modelInput.value = defaults[this.value] || '';
    }
  });
});

// LLM blend weight slider
document.getElementById('llm-weight').addEventListener('input', function () {
  document.getElementById('llm-weight-val').textContent = parseFloat(this.value).toFixed(2);
});

// ── LLM validate button ───────────────────────────────────────────────────
document.getElementById('btn-validate-llm').addEventListener('click', async () => {
  const key      = document.getElementById('llm-api-key').value.trim();
  const provider = document.querySelector('input[name="llm-provider"]:checked')?.value || 'anthropic';
  const model    = document.getElementById('llm-model').value.trim();
  const statusEl = document.getElementById('llm-validate-status');

  if (!key) { toast('Enter an API key first.', 'warning'); return; }
  statusEl.textContent = 'Validating…';
  statusEl.className   = 'small text-muted';

  try {
    const r = await fetch('/api/validate-llm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: key, provider, model }),
    });
    const d = await r.json();
    statusEl.textContent = d.message;
    statusEl.className   = d.ok ? 'small text-success' : 'small text-danger';
    if (d.ok) toast('LLM connection verified!', 'success');
    else      toast('LLM connection failed: ' + d.message, 'danger');
  } catch (err) {
    statusEl.textContent = err.message;
    statusEl.className   = 'small text-danger';
  }
});

// ── Show Target Data ──────────────────────────────────────────────────────
document.getElementById('btn-show-ds').addEventListener('click', async () => {
  const cfg = collectBenchConfig();
  if (!cfg.features.length || !cfg.targets.length) {
    toast('Select at least one feature and one target.', 'warning');
    return;
  }
  showSpinner('btn-show-ds');
  try {
    const r = await fetch('/api/target-preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });
    const d = await r.json();
    if (d.error) { toast(d.error, 'danger'); return; }
    const area = document.getElementById('target-preview-area');
    area.classList.remove('d-none');
    document.getElementById('target-count-badge').textContent = `${d.count} target rows`;
    document.getElementById('target-preview-table').innerHTML = d.html;
  } catch (err) { toast(err.message, 'danger'); }
  finally { hideSpinner('btn-show-ds'); }
});

// ── Run Benchmarking ──────────────────────────────────────────────────────
document.getElementById('btn-run').addEventListener('click', async () => {
  if (slRunning) { toast('SL is already running.', 'warning'); return; }

  const cfg = collectBenchConfig();
  if (!cfg.features.length || !cfg.targets.length) {
    toast('Select at least one feature and one target.', 'warning');
    return;
  }

  // Show progress section
  const progressArea = document.getElementById('sl-progress-area');
  progressArea.classList.remove('d-none');
  document.getElementById('sl-log').innerHTML = '';
  document.getElementById('live-plot-area').classList.remove('d-none');

  const bar      = document.getElementById('sl-progress-bar');
  const label    = document.getElementById('sl-progress-label');
  const statusTx = document.getElementById('sl-status-text');
  bar.style.width = '0%';

  // Start SL job on server
  showSpinner('btn-run');
  slRunning = true;
  try {
    const r = await fetch('/api/start-sl', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });
    const d = await r.json();
    if (d.error || d.status !== 'started') {
      toast(d.error || 'Failed to start SL.', 'danger');
      slRunning = false;
      hideSpinner('btn-run');
      return;
    }
  } catch (err) {
    toast(err.message, 'danger');
    slRunning = false;
    hideSpinner('btn-run');
    return;
  }

  // SSE stream
  const es = new EventSource('/api/run-sl');

  es.addEventListener('info', e => {
    const d = JSON.parse(e.data);
    log(d, 'log-info');
  });

  es.addEventListener('warning', e => {
    const d = JSON.parse(e.data);
    log('WARN: ' + d, 'log-warn');
    toast('Warning: ' + d, 'warning');
  });

  es.addEventListener('progress', e => {
    const d = JSON.parse(e.data);
    const pct = Math.round((d.run / d.total) * 100);
    bar.style.width = pct + '%';
    label.textContent = `${d.run} / ${d.total}`;
    if (d.stage === 'running') {
      statusTx.textContent = `Running SL — iteration ${d.run}…`;
    } else if (d.stage === 'done') {
      log(`Run ${d.run}: SL = ${d.tries_sl} exp | Random = ${d.tries_rand} exp`, 'log-ok');
    }
  });

  es.addEventListener('live_plot', e => {
    const d = JSON.parse(e.data);
    const img = document.getElementById('live-plot-img');
    img.src = 'data:image/png;base64,' + d;
  });

  es.addEventListener('final', async e => {
    const d = JSON.parse(e.data);
    lastResultRow = d.result_row;

    // Summary badges
    const s   = d.summary;
    const rBadges = document.getElementById('result-badges');
    rBadges.innerHTML = '';
    const stats = [
      { val: s.sl_mean,   lbl: 'SL cycles (mean)' },
      { val: s.sl_std,    lbl: 'SL cycles (std)' },
      { val: s.rand_mean, lbl: 'Random cycles (mean)' },
      { val: s.r2  != null ? s.r2  : 'N/A', lbl: 'R²' },
      { val: s.mae != null ? s.mae : 'N/A', lbl: 'MAE (norm)' },
      { val: s.p_wilcoxon != null ? s.p_wilcoxon : 'N/A', lbl: 'p (Wilcoxon)' },
    ];
    stats.forEach(st => {
      const col = document.createElement('div');
      col.className = 'col-6 col-sm-4 col-md-2';
      col.innerHTML = `<div class="result-badge-card">
        <div class="stat-val">${st.val}</div>
        <div class="stat-lbl">${st.lbl}</div>
      </div>`;
      rBadges.appendChild(col);
    });
    document.getElementById('results-summary').classList.remove('d-none');

    // Save result row
    if (lastResultRow !== null) {
      const sr = await fetch('/api/save-result', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ result_row: lastResultRow }),
      });
      const sd = await sr.json();
      document.getElementById('results-table').innerHTML = sd.html || '';
    }

    // Final plots
    const plotsDiv = document.getElementById('result-plots');
    plotsDiv.innerHTML = '';
    Object.entries(d.plots || {}).forEach(([, b64]) => {
      plotsDiv.innerHTML += `<img src="data:image/png;base64,${b64}" class="img-fluid rounded shadow-sm mb-3" alt="result plot"/>`;
    });

    // Open results accordion
    const accC = document.getElementById('accC');
    if (!accC.classList.contains('show')) {
      bootstrap.Collapse.getOrCreateInstance(accC).show();
    }
  });

  es.addEventListener('done', () => {
    es.close();
    slRunning = false;
    hideSpinner('btn-run');
    statusTx.textContent = 'Completed ✓';
    bar.classList.remove('progress-bar-animated');
    bar.style.width = '100%';
    toast('Benchmarking complete!', 'success');
  });

  es.addEventListener('error', e => {
    try {
      const d = JSON.parse(e.data);
      log('ERROR: ' + d.message, 'log-err');
      toast('SL error: ' + d.message.split('\n')[0], 'danger');
    } catch (parseErr) {
      log('SL connection error (unparseable event)', 'log-err');
    }
    es.close();
    slRunning = false;
    hideSpinner('btn-run');
  });

  es.onerror = () => {
    if (!slRunning) return;
    es.close();
    slRunning = false;
    hideSpinner('btn-run');
    toast('Connection to server lost.', 'danger');
  };
});

// ── Download results ──────────────────────────────────────────────────────
document.getElementById('btn-download').addEventListener('click', () => {
  window.location.href = '/api/download-results';
});

// ── PDF export ────────────────────────────────────────────────────────────
document.getElementById('btn-pdf').addEventListener('click', () => {
  window.location.href = '/api/download-pdf';
});

// ── Compare All Models ────────────────────────────────────────────────────
let cmpRunning = false;

document.getElementById('btn-compare').addEventListener('click', async () => {
  if (cmpRunning) { toast('Comparison already running.', 'warning'); return; }

  const cfg = collectBenchConfig();
  if (!cfg.features.length || !cfg.targets.length) {
    toast('Select at least one feature and one target.', 'warning');
    return;
  }

  // Show comparison accordion
  const accDItem = document.getElementById('accD-item');
  accDItem.style.display = '';
  bootstrap.Collapse.getOrCreateInstance(document.getElementById('accD')).show();
  document.getElementById('cmp-plot-area').innerHTML  = '';
  document.getElementById('cmp-table-area').innerHTML = '';
  document.getElementById('cmp-progress-bar').style.width = '0%';

  showSpinner('btn-compare');
  cmpRunning = true;

  try {
    const r = await fetch('/api/start-compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...cfg, models_to_compare: null }),
    });
    const d = await r.json();
    if (d.error || d.status !== 'started') {
      toast(d.error || 'Failed to start comparison.', 'danger');
      cmpRunning = false;
      hideSpinner('btn-compare');
      return;
    }
    const total = d.total || 8;

    const es = new EventSource('/api/run-compare');

    es.addEventListener('cmp_progress', e => {
      const d = JSON.parse(e.data);
      const pct = Math.round((d.idx / total) * 100);
      document.getElementById('cmp-progress-bar').style.width = pct + '%';
      document.getElementById('cmp-status').textContent =
        `Completed: ${d.model} (${d.idx}/${total})`;
    });

    es.addEventListener('cmp_model_error', e => {
      const d = JSON.parse(e.data);
      log(`Compare WARN: ${d.model} — ${d.message}`, 'log-warn');
    });

    es.addEventListener('cmp_final', e => {
      const d = JSON.parse(e.data);
      document.getElementById('cmp-plot-area').innerHTML =
        `<img src="data:image/png;base64,${d.plot}" class="img-fluid rounded shadow-sm" alt="comparison"/>`;

      if (d.rows && d.rows.length) {
        const cols = ['Algorithm', 'Utility function', 'Req. dev. cycle (mean)',
                      'Req. dev. cycle (std)', 'Rand. cycles (mean)', 'p-value (Wilcoxon)', 'R²'];
        let tbl = '<div class="table-scroll"><table class="table table-sm data-table"><thead><tr>' +
          cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
        d.rows.forEach(row => {
          tbl += '<tr>' + cols.map(c => `<td>${row[c] ?? '—'}</td>`).join('') + '</tr>';
        });
        tbl += '</tbody></table></div>';
        document.getElementById('cmp-table-area').innerHTML = tbl;
      }
    });

    es.addEventListener('done', () => {
      es.close();
      cmpRunning = false;
      hideSpinner('btn-compare');
      document.getElementById('cmp-status').textContent = 'Comparison complete ✓';
      document.getElementById('cmp-progress-bar').style.width = '100%';
      document.getElementById('cmp-progress-bar').classList.remove('progress-bar-animated');
      toast('Model comparison complete!', 'success');
    });

    es.onerror = () => {
      if (!cmpRunning) return;
      es.close();
      cmpRunning = false;
      hideSpinner('btn-compare');
      toast('Connection lost during comparison.', 'danger');
    };

  } catch (err) {
    toast(err.message, 'danger');
    cmpRunning = false;
    hideSpinner('btn-compare');
  }
});
