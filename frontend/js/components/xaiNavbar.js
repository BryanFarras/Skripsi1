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
      <p>Each audio file is decoded at 22,050 Hz and segmented into 5.0-second sliding temporal windows (with 50% overlap). Across each window, the librosa DSP engine computes 110 physical acoustic dimensions: 20 MFCCs, Delta-MFCCs, 7-band Spectral Contrast, Spectral Rolloff (85% and 95%), Spectral Centroid, Bandwidth, Flatness, ZCR, RMS energy, and Air-band power (&gt;7.7 kHz).</p>
      
      <h4>2. Random Forest Champion Classifier</h4>
      <p>Trained using <code>GroupShuffleSplit</code> to guarantee 0% data leakage across tracks. The champion model achieves 96.67% accuracy and 0.00% Equal Error Rate (EER) distinguishing between bona-fide human studio recordings (MUSDB18) and generative neural audio (Suno AI).</p>

      <h4>3. Explainable AI via TreeSHAP</h4>
      <p>Computes exact additive Shapley values $\\phi_i(x)$ satisfying local accuracy $f(x) = \\phi_0 + \\sum \\phi_i(x)$. Features pushing right (red) indicate generative neural vocoder and diffusion artifacts, while features pushing left (green/teal) confirm natural acoustic dynamics.</p>
    `);
  });

  // 2. "Abstract" modal
  btnAbstract?.addEventListener('click', () => {
    showModal('Thesis Abstract', `
      <h4>Abstrak (Bahasa Indonesia)</h4>
      <p>Perkembangan pesat teknologi Kecerdasan Artifisial (AI) generatif di bidang audio telah memicu lonjakan produksi musik sintetis berkualitas tinggi yang menyerupai karya manusia, sehingga menimbulkan ancaman serius terhadap hak cipta, autentisitas karya seni, dan etika industri musik digital.</p>
      <p>Sebagian besar solusi deteksi yang ada saat ini mengandalkan arsitektur pembelajaran mendalam kotak hitam (<em>black-box deep learning</em>) yang rentan terhadap fenomena korelasi palsu (<em>spurious correlation</em>) dan ketiadaan transparansi matematis. Penelitian ini merancang dan mengimplementasikan sistem deteksi musik sintetis yang akurat, tangguh, dan dapat diinterpretasikan secara transparan berbasis karakteristik fisik sinyal audio melalui pendekatan <em>Explainable AI</em> (TreeSHAP).</p>
      <p>Hasil evaluasi menunjukkan bahwa model <strong>Random Forest</strong> mencapai performa klasifikasi terbaik pada data uji terisolasi dengan <strong>Akurasi 96,67%</strong>, <strong>F1-Score 97,73%</strong>, <strong>ROC-AUC 100,00%</strong>, dan <strong>Equal Error Rate (EER) sebesar 0,00%</strong>.</p>
      
      <h4>Abstract (English)</h4>
      <p>This study presents an interpretable forensic framework for detecting AI-generated music versus authentic studio recordings using 110-dimensional DSP acoustic feature extraction and TreeSHAP explainability, achieving an Equal Error Rate (EER) of 0.00% under strict track-level grouping protocols.</p>
    `);
  });

  // 3. "Paper" modal
  btnPaper?.addEventListener('click', () => {
    showModal('Academic Paper & Model Benchmarks', `
      <h4>Research Details</h4>
      <p><strong>Title:</strong> Rancang Bangun Sistem Deteksi Musik Berbasis Kecerdasan Artifisial Menggunakan Ekstraksi Fitur Akustik dan Explainable AI (TreeSHAP)</p>
      <p><strong>Author:</strong> Muhammad Bryan Farras (NPM: 2306230975)</p>
      <p><strong>Institution:</strong> Departemen Teknik Elektro, Fakultas Teknik, Universitas Indonesia</p>
      
      <h4>Champion Classifier Benchmarks</h4>
      <table class="modal-table">
        <thead>
          <tr>
            <th>Model Architecture</th>
            <th>Accuracy</th>
            <th>F1-Score</th>
            <th>ROC-AUC</th>
            <th>EER</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Random Forest (Champion)</strong></td>
            <td><strong>96.67%</strong></td>
            <td><strong>97.73%</strong></td>
            <td><strong>100.00%</strong></td>
            <td><strong>0.00%</strong></td>
          </tr>
          <tr>
            <td>Support Vector Machine (RBF)</td>
            <td>95.83%</td>
            <td>97.14%</td>
            <td>99.70%</td>
            <td>5.56%</td>
          </tr>
          <tr>
            <td>Logistic Regression (L2)</td>
            <td>96.67%</td>
            <td>97.73%</td>
            <td>99.07%</td>
            <td>3.89%</td>
          </tr>
          <tr>
            <td>Multi-Layer Perceptron (MLP)</td>
            <td>85.83%</td>
            <td>89.70%</td>
            <td>91.52%</td>
            <td>12.78%</td>
          </tr>
        </tbody>
      </table>

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
