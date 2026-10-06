// ==========================================================================
// Component: XAI Waveform Audio Player
// Handles: Audio Playback, Web Audio Buffer Decoding, Symmetrical Waveform Canvas
// ==========================================================================

export function formatTime(sec) {
  if (isNaN(sec) || sec < 0) return '00:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

export class WaveformAudioPlayer {
  constructor(state) {
    this.state = state;
    this.dom = {
      playBtn: document.getElementById('playBtn'),
      playIcon: document.getElementById('playIcon'),
      filenameEl: document.getElementById('filenameEl'),
      metaDuration: document.getElementById('metaDuration'),
      metaSr: document.getElementById('metaSr'),
      metaSlicesCount: document.getElementById('metaSlicesCount'),
      timecodeEl: document.getElementById('timecodeEl'),
      audioEl: document.getElementById('audioEl'),
      waveformWrap: document.getElementById('waveformWrap'),
      waveformCanvas: document.getElementById('waveformCanvas'),
      waveformPlayhead: document.getElementById('waveformPlayhead'),
      waveformHoverCursor: document.getElementById('waveformHoverCursor'),
      waveformHoverTime: document.getElementById('waveformHoverTime'),
      uploadPrompt: document.getElementById('uploadPrompt'),
      uploadPlayer: document.getElementById('uploadPlayer')
    };

    this.isDraggingSeek = false;
    this.init();
  }

  init() {
    this.setupControls();
    this.setupSeeking();
    window.addEventListener('resize', () => this.drawWaveform());
  }

  setupControls() {
    this.dom.playBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      this.togglePlay();
    });

    this.dom.audioEl?.addEventListener('timeupdate', () => {
      this.state.currentTime = this.dom.audioEl.currentTime;
      this.state.duration = this.dom.audioEl.duration || this.state.duration;

      if (this.dom.timecodeEl) {
        this.dom.timecodeEl.innerText = `${formatTime(this.state.currentTime)} / ${formatTime(this.state.duration)}`;
      }

      this.drawWaveform();
    });

    this.dom.audioEl?.addEventListener('ended', () => {
      this.state.isPlaying = false;
      this.updatePlayIcon();
      this.state.currentTime = 0;
      this.drawWaveform();
    });
  }

  setupSeeking() {
    const seekAt = (clientX) => {
      if (!this.dom.waveformWrap || !this.state.duration) return;
      const rect = this.dom.waveformWrap.getBoundingClientRect();
      const clickX = clientX - rect.left;
      const pct = Math.max(0, Math.min(1, clickX / rect.width));
      this.state.currentTime = pct * this.state.duration;
      if (this.dom.audioEl) {
        this.dom.audioEl.currentTime = this.state.currentTime;
      }
      this.drawWaveform();
    };

    this.dom.waveformWrap?.addEventListener('mousedown', (e) => {
      e.stopPropagation();
      this.isDraggingSeek = true;
      seekAt(e.clientX);
    });

    window.addEventListener('mousemove', (e) => {
      if (this.isDraggingSeek) {
        seekAt(e.clientX);
      }
    });

    window.addEventListener('mouseup', () => {
      this.isDraggingSeek = false;
    });

    // Hover time tooltip
    this.dom.waveformWrap?.addEventListener('mousemove', (e) => {
      if (!this.dom.waveformWrap || !this.state.duration) return;
      const rect = this.dom.waveformWrap.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const pct = Math.max(0, Math.min(1, mouseX / rect.width));
      const hoverTime = pct * this.state.duration;

      if (this.dom.waveformHoverCursor) {
        this.dom.waveformHoverCursor.style.left = `${mouseX}px`;
        this.dom.waveformHoverCursor.style.display = 'block';
      }

      if (this.dom.waveformHoverTime) {
        this.dom.waveformHoverTime.style.left = `${mouseX}px`;
        this.dom.waveformHoverTime.innerText = formatTime(hoverTime);
        this.dom.waveformHoverTime.style.display = 'block';
      }
    });

    this.dom.waveformWrap?.addEventListener('mouseleave', () => {
      if (this.dom.waveformHoverCursor) this.dom.waveformHoverCursor.style.display = 'none';
      if (this.dom.waveformHoverTime) this.dom.waveformHoverTime.style.display = 'none';
    });
  }

  togglePlay() {
    if (!this.dom.audioEl || !this.dom.audioEl.src) return;
    if (this.dom.audioEl.paused) {
      this.dom.audioEl.play().then(() => {
        this.state.isPlaying = true;
        this.updatePlayIcon();
      }).catch(err => console.error('Audio play error:', err));
    } else {
      this.dom.audioEl.pause();
      this.state.isPlaying = false;
      this.updatePlayIcon();
    }
  }

  updatePlayIcon() {
    if (!this.dom.playIcon) return;
    if (this.state.isPlaying) {
      this.dom.playIcon.innerHTML = '<rect x="6" y="4" width="4" height="16" fill="currentColor"></rect><rect x="14" y="4" width="4" height="16" fill="currentColor"></rect>';
    } else {
      this.dom.playIcon.innerHTML = '<polygon points="6 4 20 12 6 20" fill="currentColor"></polygon>';
    }
  }

  async loadTrack(fileId, filename, duration, sampleRate, totalSlices, rawBuffer = null) {
    this.state.activeFileId = fileId;
    this.state.activeFilename = filename;
    this.state.duration = duration;

    // 1. Audio source
    if (this.dom.audioEl) {
      this.dom.audioEl.src = `/api/audio/${fileId}`;
      this.dom.audioEl.load();
      this.state.isPlaying = false;
      this.updatePlayIcon();
    }

    // 2. Track labels
    if (this.dom.filenameEl) this.dom.filenameEl.innerText = filename;
    if (this.dom.metaDuration) this.dom.metaDuration.innerText = `${duration.toFixed(2)}s`;
    if (this.dom.metaSr) this.dom.metaSr.innerText = `${(sampleRate / 1000).toFixed(1)} kHz`;
    if (this.dom.metaSlicesCount) this.dom.metaSlicesCount.innerText = `${totalSlices} Temporal Slices`;
    if (this.dom.timecodeEl) this.dom.timecodeEl.innerText = `00:00 / ${formatTime(duration)}`;

    // 3. Switch view
    if (this.dom.uploadPrompt) this.dom.uploadPrompt.style.display = 'none';
    if (this.dom.uploadPlayer) this.dom.uploadPlayer.style.display = 'flex';

    // 4. Waveform extraction & rendering
    await this.extractAndRenderWaveform(fileId, rawBuffer);
  }

  async extractAndRenderWaveform(fileId, fileBuffer = null) {
    try {
      let arrayBuffer = fileBuffer;
      if (!arrayBuffer && fileId) {
        const resp = await fetch(`/api/audio/${fileId}`);
        if (resp.ok) {
          arrayBuffer = await resp.arrayBuffer();
        }
      }

      if (arrayBuffer) {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const decoded = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
        this.state.waveformPeaks = this.computePeaksFromBuffer(decoded, 160);
        this.drawWaveform();
        return;
      }
    } catch (err) {
      console.warn('Audio decoding fallback to synthetic envelope:', err);
    }

    // Fallback peaks
    this.state.waveformPeaks = this.generateRealisticWaveformPeaks(160);
    this.drawWaveform();
  }

  computePeaksFromBuffer(audioBuffer, numBars = 160) {
    const raw = audioBuffer.getChannelData(0);
    const blockSize = Math.floor(raw.length / numBars);
    const peaks = [];

    for (let i = 0; i < numBars; i++) {
      const start = i * blockSize;
      let max = 0;
      let sum = 0;
      const step = Math.max(1, Math.floor(blockSize / 24));
      let count = 0;

      for (let j = 0; j < blockSize; j += step) {
        const val = Math.abs(raw[start + j] || 0);
        if (val > max) max = val;
        sum += val * val;
        count++;
      }

      const rms = Math.sqrt(sum / Math.max(1, count));
      const amp = Math.max(0.04, max * 0.7 + rms * 0.3);
      peaks.push(amp);
    }

    const maxVal = Math.max(...peaks, 0.05);
    return peaks.map(p => Math.min(1.0, (p / maxVal)));
  }

  generateRealisticWaveformPeaks(numBars = 160) {
    const peaks = [];
    for (let i = 0; i < numBars; i++) {
      const t = i / numBars;
      const env = Math.sin(t * Math.PI) * 0.7 + 0.25;
      const transient = Math.abs(Math.sin(i * 1.3) * Math.cos(i * 2.7)) * 0.45;
      const noise = (Math.sin(i * 9.7) * 0.5 + 0.5) * 0.25;
      const val = Math.max(0.06, Math.min(0.98, env * (0.6 + transient + noise)));
      peaks.push(val);
    }
    return peaks;
  }

  drawWaveform() {
    if (!this.dom.waveformCanvas || !this.dom.waveformWrap) return;
    const canvas = this.dom.waveformCanvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = this.dom.waveformWrap.getBoundingClientRect();
    const width = rect.width || 400;
    const height = rect.height || 78;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);

    const peaks = this.state.waveformPeaks.length > 0 
      ? this.state.waveformPeaks 
      : this.generateRealisticWaveformPeaks(160);

    const numBars = peaks.length;
    const barSpacing = width / numBars;
    const barWidth = Math.max(1.8, barSpacing - 1.2);
    const midY = height / 2;
    const maxBarH = height * 0.86;

    const progressPct = this.state.duration > 0 ? Math.min(1, Math.max(0, this.state.currentTime / this.state.duration)) : 0;
    const progressX = progressPct * width;

    // 1. Subtle horizontal center line
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(253, 240, 213, 0.15)';
    ctx.lineWidth = 1;
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();

    // 2. Symmetrical bars
    for (let i = 0; i < numBars; i++) {
      const x = i * barSpacing;
      const p = peaks[i] || 0.05;
      const barH = Math.max(4, p * maxBarH);
      const topY = midY - barH / 2;
      const isPlayed = x <= progressX;

      ctx.fillStyle = isPlayed ? '#FDF0D5' : 'rgba(253, 240, 213, 0.32)';

      ctx.beginPath();
      ctx.rect(x, topY, barWidth, barH);
      ctx.fill();
    }

    // 3. Playhead needle
    if (this.dom.waveformPlayhead) {
      this.dom.waveformPlayhead.style.left = `${progressX}px`;
      this.dom.waveformPlayhead.style.display = this.state.duration > 0 ? 'block' : 'none';
    }
  }

  seekTo(sec) {
    this.state.currentTime = Math.max(0, Math.min(this.state.duration, sec));
    if (this.dom.audioEl) {
      this.dom.audioEl.currentTime = this.state.currentTime;
      if (this.dom.audioEl.paused) this.togglePlay();
    }
    this.drawWaveform();
  }
}
