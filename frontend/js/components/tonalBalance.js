// Tonal Balance Visualizer (Live Multi-Band Energy & Target Reference Corridor)
import { state, freqToX } from '../state.js';

export const BAND_DEFINITIONS = [
  { name: 'Sub', range: [20, 60], color: '#ef4444' },
  { name: 'Bass', range: [60, 250], color: '#f97316' },
  { name: 'Low Mid', range: [250, 2000], color: '#22c55e' },
  { name: 'High Mid', range: [2000, 8000], color: '#06b6d4' },
  { name: 'Treble / Air', range: [8000, 20000], color: '#a855f7' }
];

export function renderTonalCanvas(canvas, tbData, isLive = false) {
  if (!canvas || !tbData) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;

  // Background
  ctx.fillStyle = '#080a0f';
  ctx.fillRect(0, 0, w, h);

  const curveF = (state.analysisData && state.analysisData.fft_spectrum)
    ? state.analysisData.fft_spectrum.frequencies
    : tbData.curve_freqs;

  const curveL = (state.liveMagnitudesDb && state.liveMagnitudesDb.length === curveF.length)
    ? state.liveMagnitudesDb
    : tbData.curve_levels;

  const topY = 32;
  const bottomY = h - 22;
  const contentH = bottomY - topY;

  function tonalDbToY(db) {
    const minDb = -80.0;
    const maxDb = 0.0;
    const clamped = Math.max(minDb, Math.min(maxDb, db));
    return topY + (1.0 - (clamped - minDb) / (maxDb - minDb)) * contentH;
  }

  // Frequency grid
  [60, 250, 1000, 2000, 8000].forEach(f => {
    const x = freqToX(f, w);
    ctx.strokeStyle = '#151c28';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, topY);
    ctx.lineTo(x, bottomY);
    ctx.stroke();

    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(`${f >= 1000 ? f / 1000 + 'k' : f}Hz`, x + 4, h - 6);
  });

  // dB grid
  [-12, -24, -36, -48, -60].forEach(db => {
    const y = tonalDbToY(db);
    ctx.strokeStyle = '#151c28';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();

    ctx.fillStyle = '#475569';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText(`${db}dB`, w - 38, y - 3);
  });

  // Target Reference Corridor (Mastering target zone)
  ctx.fillStyle = 'rgba(255, 255, 255, 0.035)';
  ctx.beginPath();
  for (let i = 0; i < curveF.length; i++) {
    const x = freqToX(curveF[i], w);
    const targetDb = -18 - Math.log10(curveF[i] / 20) * 8;
    const yTop = tonalDbToY(targetDb + 4);
    if (i === 0) ctx.moveTo(x, yTop); else ctx.lineTo(x, yTop);
  }
  for (let i = curveF.length - 1; i >= 0; i--) {
    const x = freqToX(curveF[i], w);
    const targetDb = -18 - Math.log10(curveF[i] / 20) * 8;
    const yBot = tonalDbToY(targetDb - 6);
    ctx.lineTo(x, yBot);
  }
  ctx.closePath();
  ctx.fill();

  // Draw 5 multi-band fills
  BAND_DEFINITIONS.forEach(b => {
    const fMin = b.range[0];
    const fMax = b.range[1];
    ctx.beginPath();
    let started = false, firstX = 0, lastX = 0;

    for (let i = 0; i < curveF.length; i++) {
      if (curveF[i] >= fMin && curveF[i] <= fMax) {
        const x = freqToX(curveF[i], w);
        const y = tonalDbToY(curveL[i]);
        if (!started) { ctx.moveTo(x, y); firstX = x; started = true; }
        else { ctx.lineTo(x, y); }
        lastX = x;
      }
    }
    if (started) {
      ctx.lineTo(lastX, bottomY);
      ctx.lineTo(firstX, bottomY);
      ctx.closePath();
      ctx.fillStyle = b.color + '44';
      ctx.fill();
    }
  });

  // White smooth curve
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2.4;
  ctx.shadowColor = 'rgba(255, 255, 255, 0.4)';
  ctx.shadowBlur = 6;
  ctx.beginPath();
  for (let i = 0; i < curveF.length; i++) {
    const x = freqToX(curveF[i], w);
    const y = tonalDbToY(curveL[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Live HUD: Multi-band energy percentage badges
  const hudY = 18;
  let hudX = 14;
  ctx.font = '10px JetBrains Mono';

  BAND_DEFINITIONS.forEach((b, idx) => {
    const pct = isLive
      ? (state.liveBandPercentages[idx] || 0)
      : (tbData.bands && tbData.bands[idx] && tbData.bands[idx].energy_percent !== undefined
          ? tbData.bands[idx].energy_percent
          : (tbData.bands && tbData.bands[idx] && tbData.bands[idx].energy_pct !== undefined
              ? tbData.bands[idx].energy_pct
              : 0));

    ctx.fillStyle = b.color;
    ctx.fillRect(hudX, hudY - 9, 8, 8);
    ctx.fillStyle = '#e2e8f0';
    const text = `${b.name}: ${Number(pct).toFixed(1)}%`;
    ctx.fillText(text, hudX + 12, hudY - 2);
    hudX += ctx.measureText(text).width + 20;
  });

  // Status badge on top right
  if (isLive) {
    ctx.fillStyle = '#00ffcc';
    ctx.fillText(`● LIVE TONAL (${state.averagingMs}ms avg)`, w - 180, hudY - 2);
  } else {
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(`● PLAYHEAD TONAL BALANCE`, w - 170, hudY - 2);
  }
}
