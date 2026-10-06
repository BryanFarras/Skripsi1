// ==========================================================================
// Component: XAI Audio Uploader
// Handles: Drag & Drop, File Picker, Sample Demo Loader, Backend API Invocations
// ==========================================================================

export class AudioUploader {
  constructor(onAnalysisSuccess, onShowLoading, onHideLoading) {
    this.onAnalysisSuccess = onAnalysisSuccess;
    this.onShowLoading = onShowLoading;
    this.onHideLoading = onHideLoading;

    this.dom = {
      dropArea: document.getElementById('dropArea'),
      fileInput: document.getElementById('fileInput'),
      browseBtn: document.getElementById('browseBtn'),
      loadDemoBtn: document.getElementById('loadDemoBtn'),
      switchTrackBtn: document.getElementById('switchTrackBtn'),
      uploadPlayer: document.getElementById('uploadPlayer')
    };

    this.init();
  }

  init() {
    // 1. Browse and File Input
    this.dom.browseBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      this.dom.fileInput?.click();
    });

    this.dom.switchTrackBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      this.dom.fileInput?.click();
    });

    this.dom.dropArea?.addEventListener('click', () => {
      if (this.dom.uploadPlayer && this.dom.uploadPlayer.style.display === 'flex') {
        return;
      }
      this.dom.fileInput?.click();
    });

    this.dom.fileInput?.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        this.uploadAndAnalyze(e.target.files[0]);
      }
    });

    // 2. Drag & Drop
    this.dom.dropArea?.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.dom.dropArea.classList.add('dragover');
    });

    this.dom.dropArea?.addEventListener('dragleave', () => {
      this.dom.dropArea.classList.remove('dragover');
    });

    this.dom.dropArea?.addEventListener('drop', (e) => {
      e.preventDefault();
      this.dom.dropArea.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        this.uploadAndAnalyze(e.dataTransfer.files[0]);
      }
    });

    // 3. Demo Button
    this.dom.loadDemoBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      this.analyzeLocalPath('backend/uploads/sample_ai_test.wav');
    });
  }

  async uploadAndAnalyze(file) {
    if (this.onShowLoading) {
      this.onShowLoading(`Analyzing "${file.name}"...`, 'Uploading audio and extracting 110 DSP features...');
    }

    try {
      const rawBuffer = await file.arrayBuffer();
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await res.json();
      if (this.onAnalysisSuccess) {
        this.onAnalysisSuccess(data, rawBuffer);
      }
    } catch (err) {
      alert('Analysis Error: ' + err.message);
    } finally {
      if (this.onHideLoading) this.onHideLoading();
    }
  }

  async analyzeLocalPath(pathStr) {
    if (!pathStr) {
      alert('Please enter a valid file path.');
      return;
    }

    if (this.onShowLoading) {
      this.onShowLoading('Analyzing audio from disk...', 'Computing TreeSHAP Shapley values...');
    }

    try {
      const res = await fetch('/api/analyze-local-path', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path: pathStr })
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
        throw new Error(err.detail || 'Failed to analyze path');
      }

      const data = await res.json();
      if (this.onAnalysisSuccess) {
        this.onAnalysisSuccess(data);
      }
    } catch (err) {
      console.error('Error analyzing local path:', err);
    } finally {
      if (this.onHideLoading) this.onHideLoading();
    }
  }
}
