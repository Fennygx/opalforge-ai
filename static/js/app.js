// External app script for OpalForge - verbose diagnostic + multi-path model loader
// Preserves UX strings and prediction pipeline; adds detailed console logging
// and attempts multiple common model.json locations to ease deployment issues.

(function () {
  const output = document.getElementById('output');
  const loader = document.getElementById('loader');
  const uploadButton = document.getElementById('upload');
  const fileInput = document.getElementById('fileInput');
  const imgPreview = document.getElementById('imgPreview');
  loader.style.display = 'none';

  let model = null;

  function log(...args) { console.log('[app.js]', ...args); }
  function warn(...args) { console.warn('[app.js]', ...args); }
  function error(...args) { console.error('[app.js]', ...args); }

  async function tryFetchInfo(url) {
    try {
      const res = await fetch(url, { method: 'GET' });
      const info = {
        ok: res.ok,
        status: res.status,
        url: res.url,
        acao: res.headers.get('access-control-allow-origin'),
      };
      // attempt to peek at the start of the body if it's JSON (no heavy download)
      try {
        const txt = await res.clone().text();
        info.preview = txt.slice(0, 1000);
      } catch (e) {
        info.preview = '(preview not available)';
      }
      return info;
    } catch (e) {
      return { ok: false, error: String(e) };
    }
  }

  // Try several likely locations for model.json to be robust to hosting differences.
  const MODEL_CANDIDATES = [
    './model.json',
    '/model.json',
    'model.json',
    './static/model/model.json',
    'static/model/model.json',
    './models/model.json',
    '/static/model/model.json'
  ];

  async function loadModel() {
    loader.style.display = 'block';
    output.textContent = 'Loading model...';
    log('Starting model load sequence...');

    if (typeof tf === 'undefined') {
      loader.style.display = 'none';
      output.textContent = 'TensorFlow.js not loaded. Check network/CSP.';
      error('tf is undefined; TF.js script may be blocked or failed to load.');
      return;
    }
    if (typeof tf.loadLayersModel !== 'function') {
      loader.style.display = 'none';
      output.textContent = 'Incompatible TF.js build: loadLayersModel missing.';
      error('tf.loadLayersModel is not a function. TF.js build missing layers API? tf:', tf);
      return;
    }

    let lastError = null;
    for (const candidate of MODEL_CANDIDATES) {
      log('Probing candidate:', candidate);
      const fetchInfo = await tryFetchInfo(candidate);
      log('Fetch result for', candidate, fetchInfo);

      if (!fetchInfo.ok) {
        warn('Fetch failed or not OK for', candidate, fetchInfo);
        continue;
      }

      // If fetch succeeded, attempt to load via TF API (this will honor CORS for weight shards)
      try {
        log('Attempting tf.loadLayersModel(', candidate, ')');
        const m = await tf.loadLayersModel(candidate);
        model = m;
        loader.style.display = 'none';
        output.textContent = 'Model loaded. Upload an image to start.';
        log('Model loaded successfully from', candidate, m);
        return;
      } catch (e) {
        lastError = e;
        warn('tf.loadLayersModel failed for', candidate, e);
        // continue trying other candidates
      }
    }

    // If we reach here, none of the candidates worked
    loader.style.display = 'none';
    output.textContent = 'Model not loaded. See console for details.';
    error('Model load failed for all candidate paths. Last error:', lastError);
    error('Network probes summary:');
    for (const c of MODEL_CANDIDATES) {
      tryFetchInfo(c).then(info => log('probe', c, info)).catch(() => {});
    }
  }

  async function predictImage(imgElement) {
    if (!model) throw new Error('Model not loaded');
    let logits;
    try {
      logits = tf.tidy(() => {
        const imgTensor = tf.browser.fromPixels(imgElement);
        const resized = tf.image.resizeBilinear(imgTensor, [224, 224]);
        const normalized = resized.toFloat().div(255).expandDims(0);
        const out = model.predict(normalized);
        return Array.isArray(out) ? out[0] : out;
      });

      const data = await logits.data();
      const confidence = Math.max(...data) * 100;
      if (logits.dispose) logits.dispose();
      return confidence;
    } catch (e) {
      try { if (logits && logits.dispose) logits.dispose(); } catch (_) {}
      throw e;
    }
  }

  function showPreview(dataUrl) {
    imgPreview.hidden = false; imgPreview.innerHTML = '';
    const img = new Image();
    img.crossOrigin = 'anonymous'; // harmless for DataURL, required for remote images
    img.src = dataUrl;
    img.alt = 'Uploaded preview';
    img.style.width = '100%'; img.style.height = '100%'; img.style.objectFit = 'cover';
    imgPreview.appendChild(img);
    return img;
  }

  uploadButton.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) { output.textContent = 'No file selected.'; return; }
    if (!model) { output.textContent = 'Model not loaded yet...'; return; }
    loader.style.display = 'block'; output.textContent = 'Processing image...';
    const reader = new FileReader();
    reader.onload = async (ev) => {
      const dataUrl = ev.target.result;
      const img = showPreview(dataUrl);
      img.onload = async () => {
        try {
          const conf = await predictImage(img);
          output.textContent = `Confidence: ${conf.toFixed(2)}% Replica`;
        } catch (err) {
          output.textContent = 'Error processing image.';
          error('Prediction error:', err);
        }
        loader.style.display = 'none';
      };
      img.onerror = () => { loader.style.display = 'none'; output.textContent = 'Error loading image.'; };
    };
    reader.onerror = () => { loader.style.display = 'none'; output.textContent = 'Error reading file.'; };
    reader.readAsDataURL(file);
  });

  window.addEventListener('load', async () => {
    await loadModel();
  });

  // preserve background behavior
  document.addEventListener('mousemove', e => {
    const xRatio = e.clientX / window.innerWidth; const yRatio = e.clientY / window.innerHeight;
    const xPos = Math.round(xRatio * 100); const yPos = Math.round(yRatio * 100);
    document.body.style.background = `linear-gradient(180deg,var(--snow) 60%, var(--navy) 40%), radial-gradient(circle at ${xPos}% ${yPos}%, rgba(184,134,11,0.06), transparent 15%)`;
  });
})();
