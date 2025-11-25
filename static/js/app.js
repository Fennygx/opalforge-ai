// External app script for OpalForge
// Preserves original UI strings and model pipeline exactly while adding memory safety.
//
// Key points:
// - Same preprocessing: resize to [224,224], div 255, expandDims(0)
// - Uses tf.tidy + explicit dispose for logits
// - Sets img.crossOrigin = 'anonymous' (no effect for DataURLs)
// - Keeps original visible strings and "Replica" suffix to preserve UX

(function () {
  const output = document.getElementById('output');
  const loader = document.getElementById('loader');
  const uploadButton = document.getElementById('upload');
  const fileInput = document.getElementById('fileInput');
  const imgPreview = document.getElementById('imgPreview');
  loader.style.display = 'none';

  let model = null;

  async function loadModel(){
    loader.style.display = 'block';
    output.textContent = 'Loading model...';
    try{
      const m = await tf.loadLayersModel('./model.json');
      model = m;
      loader.style.display = 'none';
      output.textContent = 'Model loaded. Upload an image to start.';
    }catch(err){
      loader.style.display = 'none';
      output.textContent = 'Error loading model. Ensure model.json and weights are hosted and accessible via HTTP(S).';
      console.error('Model load error:', err);
    }
  }

  async function predictImage(imgElement){
    if(!model) throw new Error('Model not loaded');
    let logits;
    try{
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
    }catch(e){
      try{ if (logits && logits.dispose) logits.dispose(); }catch(_){}
      throw e;
    }
  }

  function showPreview(dataUrl){
    imgPreview.hidden = false; imgPreview.innerHTML = '';
    const img = new Image();
    // safe to set for remote images; harmless for DataURL
    img.crossOrigin = 'anonymous';
    img.src = dataUrl;
    img.alt = 'Uploaded preview';
    img.style.width='100%'; img.style.height='100%'; img.style.objectFit='cover';
    imgPreview.appendChild(img);
    return img;
  }

  uploadButton.addEventListener('click', ()=>fileInput.click());

  fileInput.addEventListener('change', async (e)=>{
    const file = e.target.files && e.target.files[0];
    if(!file){ output.textContent='No file selected.'; return }
    if(!model){ output.textContent='Model not loaded yet...'; return }
    loader.style.display='block'; output.textContent='Processing image...';
    const reader = new FileReader();
    reader.onload = async(ev)=>{
      const dataUrl = ev.target.result;
      const img = showPreview(dataUrl);
      img.onload = async()=>{
        try{
          const conf = await predictImage(img);
          // preserve original UX suffix "Replica"
          output.textContent = `Confidence: ${conf.toFixed(2)}% Replica`;
        }catch(err){
          output.textContent='Error processing image.';
          console.error('Prediction error:', err);
        }
        loader.style.display='none';
      };
      img.onerror = ()=>{ loader.style.display='none'; output.textContent='Error loading image.'; };
    };
    reader.onerror=()=>{ loader.style.display='none'; output.textContent='Error reading file.'; };
    reader.readAsDataURL(file);
  });

  window.addEventListener('load', async()=>{
    await loadModel();
  });

  // preserve background behavior
  document.addEventListener('mousemove',e=>{
    const xRatio = e.clientX/window.innerWidth;const yRatio = e.clientY/window.innerHeight;
    const xPos = Math.round(xRatio*100);const yPos = Math.round(yRatio*100);
    document.body.style.background = `linear-gradient(180deg,var(--snow) 60%, var(--navy) 40%), radial-gradient(circle at ${xPos}% ${yPos}%, rgba(184,134,11,0.06), transparent 15%)`;
  });
})();
