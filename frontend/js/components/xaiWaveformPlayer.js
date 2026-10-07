// ==========================================================================
// Component: XAI Waveform Audio Player & 5s Slice Seeker
// Handles: Symmetrical Waveform, Contiguous 5s Segment Division,
//          AI (Red) vs Human (Blue) Segment Styling, and Full-Height Seeking Line
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
    this.timeline = null;
    this.isDraggingSeek = false;

    this.dom = {
      // Audio playback controls
      playBtn: document.getElementById('playBtn'),
      playIcon: document.getElementById('playIcon'),
      timecodeEl: document.getElementById('timecodeEl'),
      audioEl: document.getElementById('audioEl'),
      activeSliceIndicator: document.getElementById('activeSliceIndicator'),
      slicesTotalBadge: document.getElementById('slicesTotalBadge'),

      // Waveform visualizer & Seeker
      waveformWrap: document.getElementById('waveformWrap'),
      waveformCanvas: document.getElementById('waveformCanvas'),
      waveformPlayhead: document.getElementById('waveformPlayhead'),
      waveformHoverCursor: document.getElementById('waveformHoverCursor'),
      waveformHoverTime: document.getElementById('waveformHoverTime'),
      sliceSegmentsHeader: document.getElementById('sliceSegmentsHeader'),
      waveformAxis: document.getElementById('waveformAxis'),

      // Left Upload Card (File info only)
      uploadPrompt: document.getElementById('uploadPrompt'),
      uploadActiveInfo: document.getElementById('uploadActiveInfo'),
      filenameEl: document.getElementById('filenameEl'),
      metaDuration: document.getElementById('metaDuration'),
      metaSr: document.getElementById('metaSr'),
      metaSlicesCount: document.getElementById('metaSlicesCount'),
      dropMetaFormat: document.getElementById('dropMetaFormat')
    };

    this.init();
  }

  init() {
    this.setupControls();
    this.setupSeeking();
    window.addEventListener('resize', () => {
      this.drawWaveform();
      if (this.timeline) {
        this.renderTimeAxis(this.state.duration);
      }
    });
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

    // Hover time tooltip & hover line
    this.dom.waveformWrap?.addEventListener('mousemove', (e) => {
      if (!this.dom.waveformWrap || !this.state.duration) return;
      const rect = this.dom.waveformWrap.getBoundingClientRect();
      const mouseX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
      const pct = mouseX / rect.width;
      const hoverTime = pct * this.state.duration;

      if (this.dom.waveformHoverCursor) {
        this.dom.waveformHoverCursor.style.left = `${mouseX}px`;
        this.dom.waveformHoverCursor.style.display = 'block';
      }

      if (this.dom.waveformHoverTime) {
        this.dom.waveformHoverTime.style.left = `${mouseX}px`;
        const slice = this.getSliceAtTime(hoverTime);
        let tipText = formatTime(hoverTime);
        if (slice) {
          const typeStr = slice.is_ai_flagged ? 'AI' : 'Human';
          const probVal = (slice.ai_prob * 100).toFixed(0);
          tipText = `${formatTime(hoverTime)} • Slice ${slice.slice_idx + 1} (${typeStr} ${probVal}%)`;
        }
        this.dom.waveformHoverTime.innerText = tipText;
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

  async loadTrack(fileId, filename, duration, sampleRate, totalSlices, rawBuffer = null, metadata = null) {
    this.state.activeFileId = fileId;
    this.state.activeFilename = filename;
    this.state.duration = duration;

    // 1. Audio element source
    if (this.dom.audioEl) {
      this.dom.audioEl.src = `/api/audio/${fileId}`;
      this.dom.audioEl.load();
      this.state.isPlaying = false;
      this.updatePlayIcon();
    }

    // 2. Update Left Card: ONLY File Name and Metadata
    if (this.dom.uploadPrompt) this.dom.uploadPrompt.style.display = 'none';
    if (this.dom.uploadActiveInfo) this.dom.uploadActiveInfo.style.display = 'flex';
    if (this.dom.filenameEl) this.dom.filenameEl.innerText = filename;
    if (this.dom.metaDuration) this.dom.metaDuration.innerText = `${duration.toFixed(2)}s`;
    if (this.dom.metaSr) this.dom.metaSr.innerText = `${(sampleRate / 1000).toFixed(1)} kHz`;
    if (this.dom.metaSlicesCount) this.dom.metaSlicesCount.innerText = `${totalSlices} Slices (5.0s)`;
    if (this.dom.dropMetaFormat) {
      const ext = filename.split('.').pop().toUpperCase();
      this.dom.dropMetaFormat.innerText = `${ext} Audio / PCM`;
    }

    // 3. Update Playback Controls
    if (this.dom.timecodeEl) this.dom.timecodeEl.innerText = `00:00 / ${formatTime(duration)}`;
    if (this.dom.slicesTotalBadge) this.dom.slicesTotalBadge.innerText = `${totalSlices} Slices (5.0s)`;

    // 4. Time Axis
    this.renderTimeAxis(duration);

    // 5. Waveform peaks decoding & rendering
    await this.extractAndRenderWaveform(fileId, rawBuffer);
  }

  setTimeline(timeline) {
    this.timeline = timeline;
    this.renderSliceSegmentsHeader(timeline);
    this.renderTimeAxis(this.state.duration);
    this.drawWaveform();
  }

  getSliceAtTime(sec) {
    if (!this.timeline || this.timeline.length === 0) return null;
    for (const s of this.timeline) {
      const st = s.start_sec;
      const end = s.end_sec !== undefined ? s.end_sec : (st + 5.0);
      if (sec >= st && sec <= end) return s;
    }
    return this.timeline[this.timeline.length - 1];
  }

  renderSliceSegmentsHeader(timeline) {
    if (!this.dom.sliceSegmentsHeader) return;
    if (!timeline || timeline.length === 0) {
      this.dom.sliceSegmentsHeader.innerHTML = '';
      return;
    }

    const duration = this.state.duration || (timeline[timeline.length - 1].end_sec || 30);
    let html = '';

    timeline.forEach((s) => {
      const isAi = s.is_ai_flagged;
      const endSec = s.end_sec !== undefined ? s.end_sec : (s.start_sec + 5.0);
      const spanSec = endSec - s.start_sec;
      const widthPct = Math.max(8, (spanSec / duration) * 100);
      const probPct = (s.ai_prob * 100).toFixed(0);
      const badgeClass = isAi ? 'seg-ai' : 'seg-human';
      const labelType = isAi ? 'AI' : 'Human';

      html += `
        <button type="button" 
          class="slice-seg-tab ${badgeClass}" 
          style="flex-basis: ${widthPct}%;" 
          data-start="${s.start_sec}" 
          data-idx="${s.slice_idx}"
          title="Jump to Slice #${s.slice_idx + 1} (${s.start_sec}s - ${endSec}s): ${probPct}% AI Probability">
          <span class="seg-tab-title">Slice ${s.slice_idx + 1}</span>
          <span class="seg-tab-prob">${labelType} ${probPct}%</span>
          <span class="seg-tab-time">${s.start_sec}s-${endSec}s</span>
        </button>
      `;
    });

    this.dom.sliceSegmentsHeader.innerHTML = html;

    // Attach click seekers
    this.dom.sliceSegmentsHeader.querySelectorAll('.slice-seg-tab').forEach((tab) => {
      tab.addEventListener('click', (e) => {
        e.stopPropagation();
        const startSec = parseFloat(tab.getAttribute('data-start') || '0');
        this.seekTo(startSec);
      });
    });

    this.updateActiveSliceVisuals(this.state.currentTime, duration);
  }

  renderTimeAxis(duration) {
    if (!this.dom.waveformAxis || !duration) return;
    const nTicks = Math.ceil(duration / 5.0);
    let html = '';

    for (let i = 0; i <= nTicks; i++) {
      const sec = Math.min(duration, i * 5.0);
      const pct = (sec / duration) * 100;
      html += `
        <div class="axis-tick-wrap" style="left: ${pct}%;">
          <span class="axis-tick-line"></span>
          <span class="axis-tick-label">${sec.toFixed(0)}s</span>
        </div>
      `;
    }

    this.dom.waveformAxis.innerHTML = html;
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
        this.state.waveformPeaks = this.computePeaksFromBuffer(decoded, 180);
        this.drawWaveform();
        return;
      }
    } catch (err) {
      console.warn('Audio decoding fallback to synthetic envelope:', err);
    }

    this.state.waveformPeaks = this.generateRealisticWaveformPeaks(180);
    this.drawWaveform();
  }

  computePeaksFromBuffer(audioBuffer, numBars = 180) {
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

  generateRealisticWaveformPeaks(numBars = 180) {
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
    const width = rect.width || 800;
    const height = rect.height || 96;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);

    const duration = this.state.duration || 5.0;
    const currentTime = this.state.currentTime || 0;
    const progressPct = duration > 0 ? Math.min(1, Math.max(0, currentTime / duration)) : 0;
    const progressX = progressPct * width;

    const timeline = this.timeline || (this.state.mlDetector ? this.state.mlDetector.timeline : null);
    const nSlices = Math.max(1, Math.ceil(duration / 5.0));

    // 1. Draw 5-second Segment Background Tints & Boundary Divider Lines
    for (let i = 0; i < nSlices; i++) {
      const segStartSec = i * 5.0;
      const segEndSec = Math.min(duration, (i + 1) * 5.0);
      const segStartX = (segStartSec / duration) * width;
      const segEndX = (segEndSec / duration) * width;
      const segWidth = segEndX - segStartX;

      const sliceData = this.getSliceAtTime(segStartSec + 0.1);
      const isAi = sliceData ? sliceData.is_ai_flagged : null;

      // Soft background tint per 5s slice
      if (isAi === true) {
        ctx.fillStyle = 'rgba(193, 18, 31, 0.09)'; // Soft AI Red
        ctx.fillRect(segStartX, 0, segWidth, height);
      } else if (isAi === false) {
        ctx.fillStyle = 'rgba(29, 111, 165, 0.09)'; // Soft Human Blue
        ctx.fillRect(segStartX, 0, segWidth, height);
      } else {
        ctx.fillStyle = i % 2 === 0 ? 'rgba(0, 48, 73, 0.02)' : 'rgba(0, 48, 73, 0.05)';
        ctx.fillRect(segStartX, 0, segWidth, height);
      }

      // Vertical 5-second Boundary Divider
      if (i > 0) {
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(0, 48, 73, 0.22)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.moveTo(segStartX, 0);
        ctx.lineTo(segStartX, height);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // 2. Subtle horizontal center line
    const midY = height / 2;
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(0, 48, 73, 0.12)';
    ctx.lineWidth = 1;
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();

    // 3. Symmetrical waveform bars (Red for AI, Blue for Human)
    const peaks = this.state.waveformPeaks.length > 0 
      ? this.state.waveformPeaks 
      : this.generateRealisticWaveformPeaks(180);

    const numBars = peaks.length;
    const barSpacing = width / numBars;
    const barWidth = Math.max(1.8, barSpacing - 1.2);
    const maxBarH = height * 0.84;

    for (let i = 0; i < numBars; i++) {
      const x = i * barSpacing;
      const barTime = (x / width) * duration;
      const p = peaks[i] || 0.05;
      const barH = Math.max(4, p * maxBarH);
      const topY = midY - barH / 2;
      const isPlayed = x <= progressX;

      const sliceData = this.getSliceAtTime(barTime);
      const isAi = sliceData ? sliceData.is_ai_flagged : null;

      // Color selection per user requirement:
      // AI sounding segments: RED (#C1121F)
      // Human sounding segments: BLUE (#1D6FA5)
      if (isAi === true) {
        ctx.fillStyle = isPlayed ? '#C1121F' : 'rgba(193, 18, 31, 0.44)';
      } else if (isAi === false) {
        ctx.fillStyle = isPlayed ? '#1D6FA5' : 'rgba(29, 111, 165, 0.44)';
      } else {
        ctx.fillStyle = isPlayed ? '#003049' : 'rgba(0, 48, 73, 0.35)';
      }

      ctx.beginPath();
      ctx.rect(x, topY, barWidth, barH);
      ctx.fill();
    }

    // 4. Seeking Line (Playhead Needle)
    if (this.dom.waveformPlayhead) {
      this.dom.waveformPlayhead.style.left = `${progressX}px`;
      this.dom.waveformPlayhead.style.display = duration > 0 ? 'block' : 'none';
    }

    // 5. Update active slice badge and segment highlight
    this.updateActiveSliceVisuals(currentTime, duration);
  }

  updateActiveSliceVisuals(currentTime, duration) {
    const activeSlice = this.getSliceAtTime(currentTime);
    if (!activeSlice) return;

    // Update text tag
    if (this.dom.activeSliceIndicator) {
      const isAi = activeSlice.is_ai_flagged;
      const label = isAi ? 'AI' : 'Human';
      const prob = (activeSlice.ai_prob * 100).toFixed(0);
      const endSec = activeSlice.end_sec !== undefined ? activeSlice.end_sec : (activeSlice.start_sec + 5.0);
      this.dom.activeSliceIndicator.innerHTML = `Slice ${activeSlice.slice_idx + 1} • ${activeSlice.start_sec}s - ${endSec}s (${label} ${prob}%)`;
      this.dom.activeSliceIndicator.className = `playback-active-slice-tag ${isAi ? 'tag-ai' : 'tag-human'}`;
    }

    // Update tab active highlight
    if (this.dom.sliceSegmentsHeader) {
      const tabs = this.dom.sliceSegmentsHeader.querySelectorAll('.slice-seg-tab');
      tabs.forEach((tab) => {
        const idx = parseInt(tab.getAttribute('data-idx') || '-1', 10);
        if (idx === activeSlice.slice_idx) {
          tab.classList.add('active-slice');
        } else {
          tab.classList.remove('active-slice');
        }
      });
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
