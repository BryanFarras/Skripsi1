// Forensics Diagnostic Summary & Publication Plot Export
import { state, elements, showLoading, hideLoading } from '../state.js';

export function updateDiagnostics(forensics) {
  if (!forensics) return;

  if (forensics.cutoff_detected) {
    if (elements.cutoffBadge) {
      elements.cutoffBadge.innerText = forensics.cutoff_severity;
      elements.cutoffBadge.className = forensics.estimated_cutoff_hz <= 16500 ? 'badge badge-danger' : 'badge badge-warning';
    }
    if (elements.valCutoff) elements.valCutoff.innerText = `${forensics.estimated_cutoff_hz.toFixed(0)} Hz`;
    if (elements.canvasCutoffBadge) elements.canvasCutoffBadge.style.display = 'flex';
    if (elements.cutoffValText) elements.cutoffValText.innerText = `${forensics.estimated_cutoff_hz.toFixed(0)} Hz`;
  } else {
    if (elements.cutoffBadge) {
      elements.cutoffBadge.innerText = 'Natural Spectrum';
      elements.cutoffBadge.className = 'badge badge-normal';
    }
    if (elements.valCutoff) elements.valCutoff.innerText = 'None (>20kHz)';
    if (elements.canvasCutoffBadge) elements.canvasCutoffBadge.style.display = 'none';
  }

  if (elements.valRolloff) elements.valRolloff.innerText = `${forensics.spectral_rolloff_95.toFixed(0)} Hz`;
  if (elements.valCentroid) elements.valCentroid.innerText = `${forensics.spectral_centroid_hz.toFixed(0)} Hz`;
  if (elements.valAirPower) elements.valAirPower.innerText = `${(forensics.high_freq_energy_ratio * 100).toFixed(2)}%`;
  if (elements.forensicNotes) {
    elements.forensicNotes.innerHTML = forensics.forensic_notes.map(n => `<p>• ${n}</p>`).join('');
  }
  if (elements.forensicsSummaryBox) {
    elements.forensicsSummaryBox.style.display = 'block';
  }
}

// Export 200 DPI Skripsi publication figure
export async function exportPlot() {
  if (!state.currentFileId) return;
  showLoading('Rendering 200 DPI Skripsi publication figure...');
  try {
    const res = await fetch(`/api/export-plot/${state.currentFileId}`, { method: 'POST' });
    if (!res.ok) throw new Error('Export failed');
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const filename = (state.analysisData && state.analysisData.metadata) ? state.analysisData.metadata.original_filename : 'audio';
    a.download = `skripsi_visualizer_${filename}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    alert('Error exporting plot: ' + err.message);
  } finally {
    hideLoading();
  }
}
