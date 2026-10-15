// ==========================================================================
// Component: XAI Audio Uploader & Online Stream Analyzer
// Handles: Drag & Drop, File Picker, Sample Demo Loader,
//          Direct Online Streaming from YouTube, Spotify, Apple/iTunes, SoundCloud
// ==========================================================================

export class AudioUploader {
  constructor(onAnalysisSuccess, onShowLoading, onHideLoading) {
    this.onAnalysisSuccess = onAnalysisSuccess;
    this.onShowLoading = onShowLoading;
    this.onHideLoading = onHideLoading;
    this.activeMode = 'upload'; // 'upload' or 'online'

    this.dom = {
      dropArea: document.getElementById('dropArea'),
      fileInput: document.getElementById('fileInput'),
      browseBtn: document.getElementById('browseBtn'),
      loadDemoBtn: document.getElementById('loadDemoBtn'),
      switchTrackBtn: document.getElementById('switchTrackBtn'),
      uploadStandbyContainer: document.getElementById('uploadStandbyContainer'),
      uploadActiveInfo: document.getElementById('uploadActiveInfo'),
      
      // Mode tabs & panels
      tabUploadFile: document.getElementById('tabUploadFile'),
      tabOnlineStream: document.getElementById('tabOnlineStream'),
      uploadPrompt: document.getElementById('uploadPrompt'),
      onlinePrompt: document.getElementById('onlinePrompt'),
      
      // Online stream controls
      onlineUrlInput: document.getElementById('onlineUrlInput'),
      analyzeOnlineBtn: document.getElementById('analyzeOnlineBtn'),
      trackSourceTag: document.getElementById('trackSourceTag'),
      activeSourceIcon: document.getElementById('activeSourceIcon')
    };

    this.init();
  }

  init() {
    // 1. Source Mode Tab Switching
    this.dom.tabUploadFile?.addEventListener('click', (e) => {
      e.stopPropagation();
      this.setMode('upload');
    });

    this.dom.tabOnlineStream?.addEventListener('click', (e) => {
      e.stopPropagation();
      this.setMode('online');
    });

    // 2. Browse and File Input
    this.dom.browseBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      this.dom.fileInput?.click();
    });

    this.dom.switchTrackBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      this.resetToStandby();
    });

    this.dom.dropArea?.addEventListener('click', (e) => {
      // Do not trigger file browse if clicking buttons, inputs, tabs, or online panel
      if (
        e.target.closest('button') ||
        e.target.closest('input') ||
        e.target.closest('.online-stream-panel') ||
        e.target.closest('.source-mode-tabs')
      ) {
        return;
      }
      if (this.activeMode === 'upload') {
        this.dom.fileInput?.click();
      }
    });

    this.dom.fileInput?.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        this.uploadAndAnalyze(e.target.files[0]);
      }
    });

    // 3. Drag & Drop for local files
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
        this.setMode('upload');
        this.uploadAndAnalyze(e.dataTransfer.files[0]);
      }
    });

    // 4. Local Sample Demo Button
    this.dom.loadDemoBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      this.analyzeLocalPath('backend/uploads/sample_ai_test.wav');
    });

    // 5. Online Stream Submit & Enter key
    this.dom.analyzeOnlineBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      this.handleOnlineSubmit();
    });

    this.dom.onlineUrlInput?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        this.handleOnlineSubmit();
      }
    });

    // 6. Quick Demo Chips for Online Streaming
    document.querySelectorAll('.btn-online-chip')?.forEach((chip) => {
      chip.addEventListener('click', (e) => {
        e.stopPropagation();
        const url = chip.getAttribute('data-url');
        if (url) {
          if (this.dom.onlineUrlInput) this.dom.onlineUrlInput.value = url;
          this.analyzeOnlineUrl(url);
        }
      });
    });
  }

  setMode(mode) {
    this.activeMode = mode;
    if (mode === 'upload') {
      this.dom.tabUploadFile?.classList.add('active');
      this.dom.tabOnlineStream?.classList.remove('active');
      if (this.dom.uploadPrompt) this.dom.uploadPrompt.style.display = 'flex';
      if (this.dom.onlinePrompt) this.dom.onlinePrompt.style.display = 'none';
    } else {
      this.dom.tabOnlineStream?.classList.add('active');
      this.dom.tabUploadFile?.classList.remove('active');
      if (this.dom.uploadPrompt) this.dom.uploadPrompt.style.display = 'none';
      if (this.dom.onlinePrompt) this.dom.onlinePrompt.style.display = 'flex';
      this.dom.onlineUrlInput?.focus();
    }
  }

  resetToStandby() {
    if (this.dom.uploadActiveInfo) this.dom.uploadActiveInfo.style.display = 'none';
    if (this.dom.uploadStandbyContainer) this.dom.uploadStandbyContainer.style.display = 'block';
  }

  handleOnlineSubmit() {
    const rawVal = this.dom.onlineUrlInput?.value.trim() || '';
    if (!rawVal) {
      alert('Please enter a YouTube, Spotify, Apple Music/iTunes, or SoundCloud link (or song title).');
      return;
    }
    this.analyzeOnlineUrl(rawVal);
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
      this.updateActiveBadge(data);
      if (this.onAnalysisSuccess) {
        this.onAnalysisSuccess(data, rawBuffer);
      }
    } catch (err) {
      alert('Analysis Error: ' + err.message);
    } finally {
      if (this.onHideLoading) this.onHideLoading();
    }
  }

  async analyzeOnlineUrl(urlOrQuery) {
    if (!urlOrQuery) return;

    let displayHint = urlOrQuery;
    if (urlOrQuery.length > 40) displayHint = urlOrQuery.substring(0, 37) + '...';

    if (this.onShowLoading) {
      this.onShowLoading(
        `Fetching Stream: "${displayHint}"...`,
        'Retrieving online audio from YouTube/Spotify/iTunes/SoundCloud and extracting 110 DSP features...'
      );
    }

    try {
      const res = await fetch('/api/analyze-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: urlOrQuery })
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Online stream analysis failed' }));
        throw new Error(err.detail || 'Failed to stream audio from URL');
      }

      const data = await res.json();
      this.updateActiveBadge(data);
      if (this.onAnalysisSuccess) {
        this.onAnalysisSuccess(data);
      }
    } catch (err) {
      alert('Online Audio Error: ' + err.message);
    } finally {
      if (this.onHideLoading) this.onHideLoading();
    }
  }

  async analyzeLocalPath(pathStr) {
    if (!pathStr) return;

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
      this.updateActiveBadge(data);
      if (this.onAnalysisSuccess) {
        this.onAnalysisSuccess(data);
      }
    } catch (err) {
      console.error('Error analyzing local path:', err);
    } finally {
      if (this.onHideLoading) this.onHideLoading();
    }
  }

  updateActiveBadge(data) {
    if (!data) return;

    if (this.dom.uploadStandbyContainer) this.dom.uploadStandbyContainer.style.display = 'none';
    if (this.dom.uploadActiveInfo) this.dom.uploadActiveInfo.style.display = 'flex';

    const online = data.online_info;
    const tag = this.dom.trackSourceTag;

    if (tag) {
      if (online && online.source) {
        const src = online.source.toUpperCase();
        if (src === 'YOUTUBE') tag.innerText = '🔴 YOUTUBE STREAM';
        else if (src === 'SPOTIFY') tag.innerText = '🟢 SPOTIFY TRACK';
        else if (src === 'ITUNES') tag.innerText = '🟣 APPLE MUSIC / ITUNES';
        else if (src === 'SOUNDCLOUD') tag.innerText = '🟠 SOUNDCLOUD STREAM';
        else tag.innerText = `🌐 ${src} STREAM`;
      } else {
        tag.innerText = '💾 LOCAL AUDIO FILE';
      }
    }
  }
}
