// ==========================================================================
// Component: XAI Metadata Inspector
// Handles: Audio Signal Profile, Metadata Grid, Local Path Analyzers, Exports
// ==========================================================================

export class MetadataInspector {
  constructor(state, onAnalyzePath, onShowLoading, onHideLoading) {
    this.state = state;
    this.onAnalyzePath = onAnalyzePath;
    this.onShowLoading = onShowLoading;
    this.onHideLoading = onHideLoading;

    this.dom = {
      metadataPrompt: document.getElementById('metadataPrompt'),
      metadataActive: document.getElementById('metadataActive'),
      metaValFormat: document.getElementById('metaValFormat'),
      metaValSr: document.getElementById('metaValSr'),
      metaValDuration: document.getElementById('metaValDuration'),
      metaValWindow: document.getElementById('metaValWindow'),
      metaValFeatures: document.getElementById('metaValFeatures'),
      localPathInput: document.getElementById('localPathInput'),
      analyzePathBtn: document.getElementById('analyzePathBtn'),
      localPathInputAlt: document.getElementById('localPathInputAlt'),
      analyzePathBtnAlt: document.getElementById('analyzePathBtnAlt'),
      exportPlotBtn: document.getElementById('exportPlotBtn'),
      exportJsonBtn: document.getElementById('exportJsonBtn')
    };

    this.init();
  }

  init() {
    this.setupLocalPathInputs();
    this.setupExports();
  }

  setupLocalPathInputs() {
    const handleInput = (inputEl) => {
      const p = inputEl?.value.trim();
      if (p && this.onAnalyzePath) {
        this.onAnalyzePath(p);
      }
    };

    this.dom.analyzePathBtn?.addEventListener('click', () => handleInput(this.dom.localPathInput));
    this.dom.localPathInput?.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') handleInput(this.dom.localPathInput);
    });

    this.dom.analyzePathBtnAlt?.addEventListener('click', () => handleInput(this.dom.localPathInputAlt));
    this.dom.localPathInputAlt?.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') handleInput(this.dom.localPathInputAlt);
    });
  }

  setupExports() {
    // Export XAI JSON Report
    this.dom.exportJsonBtn?.addEventListener('click', async () => {
      if (!this.state.activeFileId) return;
      try {
        const res = await fetch(`/api/explain/${this.state.activeFileId}`);
        if (!res.ok) throw new Error('Failed to fetch XAI report');
        const json = await res.json();
        const blob = new Blob([JSON.stringify(json, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `xai_treeshap_report_${this.state.activeFileId}.json`;
        a.click();
        URL.revokeObjectURL(url);
      } catch (e) {
        alert('Export failed: ' + e.message);
      }
    });

    // Export Publication Figure (200 DPI)
    this.dom.exportPlotBtn?.addEventListener('click', async () => {
      if (!this.state.activeFileId) return;
      if (this.onShowLoading) this.onShowLoading('Rendering 200 DPI Skripsi publication figure...');
      try {
        const res = await fetch(`/api/export-plot/${this.state.activeFileId}`, { method: 'POST' });
        if (!res.ok) throw new Error('Export failed');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `skripsi_xai_plot_${this.state.activeFilename}.png`;
        a.click();
        URL.revokeObjectURL(url);
      } catch (e) {
        alert('Plot export failed: ' + e.message);
      } finally {
        if (this.onHideLoading) this.onHideLoading();
      }
    });
  }

  render(metadata, totalSlices) {
    if (!metadata) return;

    if (this.dom.metadataPrompt) this.dom.metadataPrompt.style.display = 'none';
    if (this.dom.metadataActive) this.dom.metadataActive.style.display = 'flex';

    if (this.dom.metaValFormat) {
      this.dom.metaValFormat.innerText = (metadata.format || 'WAV').toUpperCase();
    }
    if (this.dom.metaValSr) {
      this.dom.metaValSr.innerText = `${(metadata.sample_rate / 1000).toFixed(1)} kHz`;
    }
    if (this.dom.metaValDuration) {
      this.dom.metaValDuration.innerText = `${metadata.duration_seconds.toFixed(2)}s`;
    }
    if (this.dom.metaValWindow) {
      this.dom.metaValWindow.innerText = `5.0s (50% Overlap)`;
    }
    if (this.dom.metaValFeatures) {
      this.dom.metaValFeatures.innerText = `110 Acoustic Dims`;
    }
  }
}
