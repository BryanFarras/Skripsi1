// Equalizer Visualizer (FL Studio Fruity Parametric EQ 2 Style Real-Time FFT)
import { state, freqToX } from '../state.js';

// 7 Parametric EQ 2 Bands
export const PEQ2_BANDS = [
  { id: 1, name: 'SUB', freq: 35, range: [20, 60], color: '#ef4444' },
  { id: 2, name: 'BASS', freq: 110, range: [60, 250], color: '#f97316' },
  { id: 3, name: 'LOW MID', freq: 380, range: [250, 600], color: '#eab308' },
  { id: 4, name: 'MID', freq: 1100, range: [600, 2400], color: '#22c55e' },
  { id: 5, name: 'HIGH MID', freq: 3400, range: [2400, 6000], color: '#06b6d4' },
  { id: 6, name: 'HIGH', freq: 7600, range: [6000, 11000], color: '#3b82f6' },
  { id: 7, name: 'TREBLE', freq: 14500, range: [11000, 20000], color: '#a855f7' }
];

export function renderEqualizerCanvas(canvas, eqData, isLive = false) {
  if (!canvas || !eqData) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;

  // Background - Sleek Studio Console
  ctx.fillStyle = '#080b11';
  ctx.fillRect(0, 0, w, h);

  const topHeaderH = 30;
  const bottomAxisH = 22;
  const rightAxisW = 44;
  const contentW = Math.max(100, w - rightAxisW);
  const contentH = Math.max(50, h - topHeaderH - bottomAxisH);
  const topY = topHeaderH;
  const bottomY = topY + contentH;

  // dB to Y mapping (+12dB down to -80dB)
  function peqDbToY(db) {
    const minDb = -80.0;
    const maxDb = 12.0;
    const clamped = Math.max(minDb, Math.min(maxDb, db));
    return topY + (1.0 - (clamped - minDb) / (maxDb - minDb)) * contentH;
  }

  // 1. Top 7 Band Header Buttons / Pills
  const pillW = contentW / 7;
  PEQ2_BANDS.forEach((b, idx) => {
    const px = idx * pillW;
    const pw = pillW - 2;

    ctx.fillStyle = 'rgba(18, 24, 38, 0.9)';
    ctx.fillRect(px, 3, pw, topHeaderH - 6);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.strokeRect(px, 3, pw, topHeaderH - 6);

    ctx.fillStyle = b.color;
    ctx.fillRect(px, 3, pw, 3);

    ctx.beginPath();
    ctx.arc(px + 10, 15, 6, 0, Math.PI * 2);
    ctx.fillStyle = b.color;
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 8px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${b.id}`, px + 10, 15);

    ctx.textAlign = 'left';
    ctx.font = 'bold 9px JetBrains Mono, monospace';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(b.name, px + 20, 16);
  });
  ctx.textAlign = 'start';
  ctx.textBaseline = 'alphabetic';

  // 2. Frequency Grid Lines & Labels
  const gridFreqs = [20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 10000, 20000];
  const majorFreqs = [20, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000];

  gridFreqs.forEach(f => {
    const x = freqToX(f, contentW);
    const isMajor = [100, 1000, 10000].includes(f);
    ctx.strokeStyle = isMajor ? 'rgba(255, 255, 255, 0.12)' : 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, topY);
    ctx.lineTo(x, bottomY);
    ctx.stroke();
  });

  majorFreqs.forEach(f => {
    const x = freqToX(f, contentW);
    ctx.fillStyle = '#64748b';
    ctx.font = '9px JetBrains Mono, monospace';
    const txt = f >= 1000 ? `${f / 1000}k` : `${f}`;
    ctx.fillText(txt, x - 6, h - 6);
  });

  // 3. dB Grid Lines & Scale Labels
  const dbLines = [12, 6, 0, -6, -12, -18, -36, -60];
  dbLines.forEach(db => {
    const y = peqDbToY(db);
    ctx.strokeStyle = (db === 0) ? 'rgba(255, 255, 255, 0.22)' : 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = (db === 0) ? 1.2 : 1;
    if (db !== 0) ctx.setLineDash([4, 4]); else ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(contentW, y);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = (db === 0) ? '#cbd5e1' : '#64748b';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillText(`${db > 0 ? '+' : ''}${db}dB`, contentW + 6, y + 3);
  });

  const freqs = eqData.frequencies;
  const globalAvgs = eqData.magnitudes_db;

  const avgs = (state.liveMagnitudesDb && state.liveMagnitudesDb.length === freqs.length) ? state.liveMagnitudesDb : globalAvgs;
  const peaks = (state.livePeakHoldDb && state.livePeakHoldDb.length === freqs.length) ? state.livePeakHoldDb : eqData.peaks_db;

  function getDbAtFreq(targetHz, arr) {
    if (targetHz <= freqs[0]) return arr[0];
    if (targetHz >= freqs[freqs.length - 1]) return arr[freqs.length - 1];
    const logMin = Math.log10(freqs[0]);
    const logMax = Math.log10(freqs[freqs.length - 1]);
    const logT = Math.log10(targetHz);
    const pos = ((logT - logMin) / (logMax - logMin)) * (freqs.length - 1);
    const i0 = Math.floor(pos);
    const i1 = Math.min(freqs.length - 1, i0 + 1);
    const frac = pos - i0;
    return arr[i0] * (1 - frac) + arr[i1] * frac;
  }

  // 4. Real-Time Vertical Spectrum Bars
  const barSpacing = 3;
  const numBars = Math.floor(contentW / barSpacing);
  const barW = Math.max(1.5, barSpacing - 1);

  for (let i = 0; i < numBars; i++) {
    const x = i * barSpacing;
    const norm = i / (numBars - 1);
    const logF = Math.log10(20) + norm * (Math.log10(20000) - Math.log10(20));
    const f = Math.pow(10, logF);

    const db = getDbAtFreq(f, avgs);
    const peakDb = getDbAtFreq(f, peaks);

    const barY = peqDbToY(db);
    const peakY = peqDbToY(peakDb);
    const barHeight = bottomY - barY;

    if (barHeight > 1) {
      const grad = ctx.createLinearGradient(0, bottomY, 0, barY);
      grad.addColorStop(0.0, '#5a0204');
      grad.addColorStop(0.35, '#dc2626');
      grad.addColorStop(0.70, '#ea580c');
      grad.addColorStop(0.92, '#f59e0b');
      grad.addColorStop(1.0, '#fef08a');

      ctx.fillStyle = grad;
      ctx.fillRect(x, barY, barW, barHeight);

      if (peakY < barY - 1) {
        ctx.fillStyle = 'rgba(254, 240, 138, 0.75)';
        ctx.fillRect(x, peakY, barW, 1.5);
      }
    }
  }

  // 5. Smooth Spectrum Envelope Curve & Translucent Fill
  ctx.beginPath();
  let firstPt = true;
  for (let i = 0; i < freqs.length; i++) {
    const x = freqToX(freqs[i], contentW);
    const y = peqDbToY(avgs[i]);
    if (firstPt) { ctx.moveTo(x, y); firstPt = false; }
    else { ctx.lineTo(x, y); }
  }
  ctx.lineTo(contentW, bottomY);
  ctx.lineTo(0, bottomY);
  ctx.closePath();

  const curveGrad = ctx.createLinearGradient(0, topY, 0, bottomY);
  curveGrad.addColorStop(0.0, 'rgba(249, 115, 22, 0.22)');
  curveGrad.addColorStop(0.5, 'rgba(220, 38, 38, 0.08)');
  curveGrad.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');
  ctx.fillStyle = curveGrad;
  ctx.fill();

  // Glow line
  ctx.beginPath();
  for (let i = 0; i < freqs.length; i++) {
    const x = freqToX(freqs[i], contentW);
    const y = peqDbToY(avgs[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.8;
  ctx.shadowColor = 'rgba(254, 240, 138, 0.6)';
  ctx.shadowBlur = 8;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // 6. Numbered Band Tokens
  PEQ2_BANDS.forEach(b => {
    const bx = freqToX(b.freq, contentW);
    const bDb = getDbAtFreq(b.freq, avgs);
    const by = peqDbToY(bDb);

    ctx.save();
    ctx.shadowColor = b.color;
    ctx.shadowBlur = 12;

    ctx.beginPath();
    ctx.arc(bx, by, 9, 0, Math.PI * 2);
    ctx.fillStyle = '#0c111a';
    ctx.fill();

    ctx.lineWidth = 2.0;
    ctx.strokeStyle = b.color;
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 9px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${b.id}`, bx, by);

    ctx.font = '8px JetBrains Mono, monospace';
    ctx.fillStyle = b.color;
    const fLabel = b.freq >= 1000 ? `${(b.freq / 1000).toFixed(1)}k` : `${b.freq}Hz`;
    ctx.fillText(fLabel, bx, by + 14);
  });
  ctx.textAlign = 'start';
  ctx.textBaseline = 'alphabetic';

  // 7. Status Badge
  ctx.font = '10px JetBrains Mono, monospace';
  if (isLive) {
    ctx.fillStyle = '#00ffcc';
    ctx.fillText(`● LIVE PARAMETRIC EQ 2 (${state.averagingMs}ms avg)`, contentW - 225, topY + 18);
  } else {
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(`● PLAYHEAD REAL-TIME SPECTRUM`, contentW - 200, topY + 18);
  }
}
