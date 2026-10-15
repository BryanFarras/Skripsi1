// ==========================================================================
// Component: XAI Navbar & Editorial Modals
// Handles: Editorial Header Links, Modal Dialogs (How It Works, Abstract, Paper)
// ==========================================================================

export function setupNavbar() {
  const btnHowItWorks = document.getElementById('btnHowItWorks');
  const btnAbstract = document.getElementById('btnAbstract');
  const btnPaper = document.getElementById('btnPaper');
  const modalBackdrop = document.getElementById('modalBackdrop');
  const modalTitle = document.getElementById('modalTitle');
  const modalBody = document.getElementById('modalBody');
  const modalCloseBtn = document.getElementById('modalCloseBtn');

  function showModal(title, contentHtml) {
    if (modalTitle) modalTitle.innerText = title;
    if (modalBody) modalBody.innerHTML = contentHtml;
    if (modalBackdrop) modalBackdrop.style.display = 'flex';
  }

  function closeModal() {
    if (modalBackdrop) modalBackdrop.style.display = 'none';
  }

  // 1. "How does it work?" modal
  btnHowItWorks?.addEventListener('click', () => {
    showModal('How Does It Work?', `
      <h4>1. 110-Dimensional Acoustic Signal Extraction</h4>
      <p>Each audio file is decoded at 22,050 Hz and segmented into 5.0-second sliding temporal windows (with 50% overlap). Across each window, the librosa DSP engine extracts 110 physical acoustic dimensions: 20 MFCCs, Delta-MFCCs, 7-band Spectral Contrast, Spectral Rolloff (85% and 95%), Spectral Centroid, Bandwidth, Flatness, ZCR, RMS energy, and Air-band power (&gt;7.7 kHz).</p>
      
      <h4>2. Track-Level Grouped Evaluation Protocol</h4>
      <p>Evaluated across <strong>8,297 acoustic slices</strong> from 241 full tracks using <code>GroupShuffleSplit</code> (75% train / 25% test) to strictly prevent song-chunk leakage. The Random Forest model achieved <strong>99.76% Accuracy</strong>, <strong>99.87% F1-Score</strong>, <strong>99.99% ROC-AUC</strong>, and an <strong>Equal Error Rate (EER) of 1.00%</strong> with <strong>0% False Alarm rate</strong> (245/245 bona-fide human tracks identified correctly).</p>

      <h4>3. Key Acoustic Forensic Finding</h4>
      <p>Global feature importance analysis revealed that <strong>Air-Band Spectral Contrast (Band 6: &gt;7.7 kHz)</strong> is the single most dominant discriminator, accounting for <strong>54.13%</strong> of total model importance (Mean: 39.11%, Std: 15.02%). This mathematically proves that neural audio vocoders (e.g. EnCodec, SoundStream) introduce phase cancellation and contrast collapse in the extreme high frequencies.</p>

      <h4>4. Explainable AI via TreeSHAP</h4>
      <p>Computes exact additive Shapley values $\\phi_i(x)$ satisfying local accuracy $f(x) = \\phi_0 + \\sum \\phi_i(x)$. Features pushing right (red) indicate generative neural vocoder and diffusion artifacts, while features pushing left (blue) confirm natural acoustic dynamics.</p>
    `);
  });

  // 2. "Abstract" modal
  btnAbstract?.addEventListener('click', () => {
    showModal('Thesis Abstract', `
      <h4>Abstrak (Bahasa Indonesia)</h4>
      <p>Perkembangan pesat teknologi Kecerdasan Artifisial (AI) generatif di bidang audio telah memicu lonjakan produksi musik sintetis berkualitas tinggi yang menyerupai karya manusia, sehingga menimbulkan ancaman serius terhadap hak cipta, autentisitas karya seni, dan etika industri musik digital.</p>
      <p>Sebagian besar solusi deteksi yang ada saat ini mengandalkan arsitektur pembelajaran mendalam kotak hitam (<em>black-box deep learning</em>) yang rentan terhadap fenomena korelasi palsu (<em>spurious correlation</em>) dan ketiadaan transparansi matematis. Penelitian ini merancang dan mengimplementasikan sistem deteksi musik sintetis berbasis karakteristik fisik sinyal audio melalui pendekatan <em>Explainable AI</em> (TreeSHAP).</p>
      <p>Berdasarkan evaluasi empiris terhadap <strong>8.297 sampel slice</strong> (241 track utuh) dengan skema <em>Grouped Track-Level Split</em> (tanpa kebocoran data antar potongan lagu), model <strong>Random Forest</strong> menghasilkan performa superior dengan <strong>Akurasi 99,76%</strong>, <strong>F1-Score 99,87%</strong>, <strong>ROC-AUC 99,99%</strong>, dan <strong>Equal Error Rate (EER) sebesar 1,00%</strong> serta <strong>False Alarm Rate 0,00%</strong> (245 dari 245 sampel lagu manusia teridentifikasi dengan tepat). Fitur paling diskriminatif adalah <em>Spectral Contrast Band 6</em> (&gt;7,7 kHz) yang menyumbang 54,13% total signifikansi akibat distorsi representasi laten neural vocoder.</p>
      
      <h4>Abstract (English)</h4>
      <p>This study presents an interpretable forensic framework for detecting AI-generated music versus authentic studio recordings using 110-dimensional DSP acoustic feature extraction and TreeSHAP explainability. Validated on 8,297 slices across 241 full tracks under strict track-level grouping protocols, the model achieves 99.76% accuracy, 99.99% ROC-AUC, 1.00% Equal Error Rate (EER), and 0.00% False Alarm rate, identifying Air-Band Spectral Contrast (&gt;7.7 kHz) as the primary physical discriminator (54.13% importance).</p>
    `);
  });

  // 3. "Paper" modal
  btnPaper?.addEventListener('click', () => {
    showModal('Academic Paper & Model Benchmarks', `
      <h4>Research Details</h4>
      <p><strong>Judul Skripsi:</strong> Rancang Bangun Sistem Deteksi Musik Berbasis Kecerdasan Artifisial Menggunakan Ekstraksi Fitur Akustik dan Explainable AI (TreeSHAP)</p>
      <p><strong>Penulis:</strong> Muhammad Bryan Farras (NPM: 2306230975)</p>
      <p><strong>Institusi:</strong> Departemen Teknik Elektro, Fakultas Teknik, Universitas Indonesia</p>
      <p><strong>Dataset Evaluasi:</strong> 8,297 Slices (241 Full Tracks • 6,196 Train / 2,101 Test • Grouped Track Split)</p>
      
      <h4>Baseline Model Benchmark Comparison</h4>
      <table class="modal-table">
        <thead>
          <tr>
            <th>Classifier Architecture</th>
            <th>Accuracy</th>
            <th>F1-Score</th>
            <th>ROC-AUC</th>
            <th>Equal Error Rate (EER)</th>
            <th>Train Time</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Random Forest (Champion Baseline)</strong></td>
            <td><strong>99.76%</strong></td>
            <td><strong>99.87%</strong></td>
            <td><strong>99.99%</strong></td>
            <td><strong>1.00%</strong></td>
            <td>0.443s</td>
          </tr>
          <tr>
            <td>Support Vector Machine (RBF)</td>
            <td>100.00%</td>
            <td>100.00%</td>
            <td>100.00%</td>
            <td>0.00%</td>
            <td>1.261s</td>
          </tr>
          <tr>
            <td>Logistic Regression (L2)</td>
            <td>100.00%</td>
            <td>100.00%</td>
            <td>100.00%</td>
            <td>0.00%</td>
            <td>0.025s</td>
          </tr>
          <tr>
            <td>Multi-Layer Perceptron (MLP)</td>
            <td>99.19%</td>
            <td>99.54%</td>
            <td>99.70%</td>
            <td>2.10%</td>
            <td>0.764s</td>
          </tr>
        </tbody>
      </table>

      <h4>Top 10 Global Discriminative Acoustic Features</h4>
      <ol style="margin-left: 20px; font-size: 0.82rem; line-height: 1.6; color: var(--c-navy);">
        <li><strong>Air-Band Spectral Contrast Mean (Band 6: &gt;7.7kHz):</strong> 39.11% importance</li>
        <li><strong>Air-Band Contrast Dynamics Std (Band 6):</strong> 15.02% importance</li>
        <li><strong>Vocal Mid-Range Contrast Variance Std (Band 3: ~800Hz-1.7kHz):</strong> 2.62% importance</li>
        <li><strong>Vocal Mid-Range Spectral Contrast Mean (Band 3):</strong> 2.32% importance</li>
        <li><strong>Sub-Bass Spectral Contrast Mean (Band 0: 0-200Hz):</strong> 1.98% importance</li>
        <li><strong>MFCC 1 Mean (Total Energy Distribution):</strong> 1.71% importance</li>
        <li><strong>MFCC 3 Mean (Spectral Curvature & Tilt):</strong> 1.61% importance</li>
        <li><strong>Low-Mid Contrast Dynamics Std (Band 2: ~400Hz-800Hz):</strong> 1.61% importance</li>
        <li><strong>Upper-Mid Spectral Contrast Mean (Band 4: ~1.7kHz-3.5kHz):</strong> 1.38% importance</li>
        <li><strong>MFCC 6 Mean (Upper Vocal Timbral Resonance):</strong> 1.17% importance</li>
      </ol>

      <div style="margin-top: 18px; display: flex; gap: 10px;">
        <a href="/visualizer" class="btn-hero-dark" style="text-decoration: none; font-size: 0.85rem; padding: 8px 16px;">Open Spectrogram Visualizer</a>
        <a href="/docs" target="_blank" class="btn-hero-secondary" style="text-decoration: none; font-size: 0.85rem; padding: 8px 16px;">API Documentation</a>
      </div>
    `);
  });

  modalCloseBtn?.addEventListener('click', closeModal);
  modalBackdrop?.addEventListener('click', (e) => {
    if (e.target === modalBackdrop) closeModal();
  });
}
