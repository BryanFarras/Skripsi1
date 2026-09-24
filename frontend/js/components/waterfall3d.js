// 3D Spectral Waterfall Visualizer (Topological Spectral Terrain & 360° Orbit)
import { state, elements, hexToRgba } from '../state.js';
import { safeSeek } from '../audioPlayer.js';

// 3D Waterfall Renderer
export function renderWaterfall3DCanvas(canvas, waterfallData) {
  if (!waterfallData || !waterfallData.slices || waterfallData.slices.length === 0) return;

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

  const slices = waterfallData.slices;
  const numSlices = slices.length;
  const curTime = elements.audioElement ? elements.audioElement.currentTime : 0;

  // Find active slice closest to current playback time
  let activeSliceIdx = 0;
  let minDiff = 999999;
  for (let s = 0; s < numSlices; s++) {
    const diff = Math.abs(slices[s].timestamp_sec - curTime);
    if (diff < minDiff) {
      minDiff = diff;
      activeSliceIdx = s;
    }
  }

  // Precomputed Camera Matrix & Scales (Hoisted outside all loops for maximum 60 FPS performance)
  const cosY = Math.cos(state.camYaw);
  const sinY = Math.sin(state.camYaw);
  const cosP = Math.cos(state.camPitch);
  const sinP = Math.sin(state.camPitch);
  const scaleX = 380 * state.camZoom;
  const scaleY = 220 * state.camZoom;
  const scaleZ = 340 * state.camZoom;
  const cxMid = w * 0.52;
  const cyMid = h * 0.68;
  const dist = 600;

  // Fast 3D Projection Helper
  function project(x, y, z) {
    const cx = (x - 0.45) * scaleX;
    const cy = (y - 0.15) * scaleY;
    const cz = (z - 0.5) * scaleZ;

    const x1 = cx * cosY + cz * sinY;
    const z1 = -cx * sinY + cz * cosY;

    const y1 = cy * cosP - z1 * sinP;
    const z2 = cy * sinP + z1 * cosP;

    const fov = dist / (dist + z2);
    return { sx: cxMid + x1 * fov, sy: cyMid - y1 * fov, depth: z2 };
  }

  // Draw 3D Axes
  const o = project(0, 0, 0);
  const axY = project(0, 1.05, 0);   // volume (y)
  const axX = project(1.05, 0, 0);   // freq (x)
  const axZ = project(0, 0, 1.05);   // t (z)

  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1.5;

  // Y axis line
  ctx.beginPath();
  ctx.moveTo(o.sx, o.sy);
  ctx.lineTo(axY.sx, axY.sy);
  ctx.stroke();

  // X axis line
  ctx.beginPath();
  ctx.moveTo(o.sx, o.sy);
  ctx.lineTo(axX.sx, axX.sy);
  ctx.stroke();

  // Z axis line
  ctx.beginPath();
  ctx.moveTo(o.sx, o.sy);
  ctx.lineTo(axZ.sx, axZ.sy);
  ctx.stroke();

  // Axis Labels
  ctx.font = 'bold 12px JetBrains Mono';
  ctx.fillStyle = '#f8fafc';
  ctx.fillText('volume (y)', axY.sx - 36, axY.sy - 8);
  ctx.fillText('freq (x)', axX.sx + 8, axX.sy + 14);
  ctx.fillText('t (z)', axZ.sx + 8, axZ.sy - 4);

  const isPlaying = elements.audioElement && !elements.audioElement.paused;

  // Ground plane outline
  const floorO = project(0, 0, 0);
  const floorX = project(1, 0, 0);
  const floorXZ = project(1, 0, 1);
  const floorZ = project(0, 0, 1);

  ctx.strokeStyle = isPlaying ? 'rgba(255, 255, 255, 0.08)' : 'rgba(255, 255, 255, 0.04)';
  ctx.lineWidth = 1.0;
  ctx.beginPath();
  ctx.moveTo(floorO.sx, floorO.sy);
  ctx.lineTo(floorX.sx, floorX.sy);
  ctx.lineTo(floorXZ.sx, floorXZ.sy);
  ctx.lineTo(floorZ.sx, floorZ.sy);
  ctx.closePath();
  ctx.stroke();

  // Travel guide line along active slice on ground plane
  if (isPlaying) {
    const curZ = activeSliceIdx / Math.max(1, numSlices - 1);
    const lineStart = project(0, 0, curZ);
    const lineEnd = project(1, 0, curZ);

    ctx.strokeStyle = 'rgba(0, 229, 255, 0.35)';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(lineStart.sx, lineStart.sy);
    ctx.lineTo(lineEnd.sx, lineEnd.sy);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Exact Painter's Depth Sorting
  const sliceDepths = new Float32Array(numSlices);
  for (let s = 0; s < numSlices; s++) {
    sliceDepths[s] = project(0.5, 0, s / Math.max(1, numSlices - 1)).depth;
  }
  const sortedIndices = Array.from({ length: numSlices }, (_, i) => i);
  sortedIndices.sort((a, b) => sliceDepths[b] - sliceDepths[a]);

  const indicesToRender = sortedIndices;
  const isDense = numSlices > 100;

  // Render Slices
  indicesToRender.forEach(sIdx => {
    const slice = slices[sIdx];
    const zNorm = sIdx / Math.max(1, numSlices - 1);
    const mags = slice.magnitudes_db;
    const numBins = mags.length;
    const isActive = (sIdx === activeSliceIdx);

    if (state._wfScreenX.length < numBins) {
      state._wfScreenX = new Float32Array(numBins + 64);
      state._wfScreenY = new Float32Array(numBins + 64);
    }

    const cz = (zNorm - 0.5) * scaleZ;
    const czSinY = cz * sinY;
    const czCosY = cz * cosY;

    let peakX = 0, peakY = 0, maxDb = -999;

    for (let b = 0; b < numBins; b++) {
      const xNorm = b / (numBins - 1);
      const db = mags[b];
      const yNorm = Math.max(0, Math.min(1, (db + 80) / 80));
      const cx = (xNorm - 0.45) * scaleX;
      const cy = (yNorm - 0.15) * scaleY;

      const x1 = cx * cosY + czSinY;
      const z1 = -cx * sinY + czCosY;

      const y1 = cy * cosP - z1 * sinP;
      const z2 = cy * sinP + z1 * cosP;

      const fov = dist / (dist + z2);
      const sx = cxMid + x1 * fov;
      const sy = cyMid - y1 * fov;

      state._wfScreenX[b] = sx;
      state._wfScreenY[b] = sy;

      if (isActive && db > maxDb) {
        maxDb = db;
        peakX = sx;
        peakY = sy;
      }
    }

    if (isActive) {
      // ACTIVE PLAYING FRAME: VIVID NEON HIGHLIGHT
      const bStart = project(0, 0, zNorm);
      const bEnd = project(1, 0, zNorm);

      // 1. Vibrant fill under ribbon
      ctx.beginPath();
      ctx.moveTo(bStart.sx, bStart.sy);
      for (let b = 0; b < numBins; b++) {
        ctx.lineTo(state._wfScreenX[b], state._wfScreenY[b]);
      }
      ctx.lineTo(bEnd.sx, bEnd.sy);
      ctx.closePath();
      ctx.fillStyle = slice.color_hex + '55';
      ctx.fill();

      // 2. Active baseline guide
      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.moveTo(bStart.sx, bStart.sy);
      ctx.lineTo(bEnd.sx, bEnd.sy);
      ctx.stroke();

      // 3. Glowing ridge top line
      ctx.save();
      ctx.shadowColor = slice.color_hex || '#00e5ff';
      ctx.shadowBlur = 14;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3.0;
      ctx.beginPath();
      for (let b = 0; b < numBins; b++) {
        if (b === 0) ctx.moveTo(state._wfScreenX[0], state._wfScreenY[0]);
        else ctx.lineTo(state._wfScreenX[b], state._wfScreenY[b]);
      }
      ctx.stroke();
      ctx.restore();

      // 4. Peak dot & clean floating badge
      ctx.fillStyle = slice.color_hex || '#00e5ff';
      ctx.beginPath();
      ctx.arc(peakX, peakY, 4.5, 0, Math.PI * 2);
      ctx.fill();

      const badgeText = `▶ ${slice.formatted_time}`;
      ctx.font = 'bold 10px JetBrains Mono';
      const badgeW = ctx.measureText(badgeText).width + 10;
      ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(peakX + 4, peakY - 20, badgeW, 16, 4);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.fillText(badgeText, peakX + 9, peakY - 8);
    } else {
      // INACTIVE FRAMES: 100% TRANSPARENT BODY!
      const sliceDist = Math.abs(sIdx - activeSliceIdx);
      let alpha = state.wfInactiveAlpha;
      if (sliceDist < 12) alpha = Math.min(0.60, alpha * 1.8);
      if (sliceDist < 4) alpha = Math.min(0.85, alpha * 2.5);

      ctx.strokeStyle = hexToRgba(slice.color_hex, alpha);
      ctx.lineWidth = isDense ? 0.75 : 1.2;
      ctx.beginPath();
      for (let b = 0; b < numBins; b++) {
        if (b === 0) ctx.moveTo(state._wfScreenX[0], state._wfScreenY[0]);
        else ctx.lineTo(state._wfScreenX[b], state._wfScreenY[b]);
      }
      ctx.stroke();
    }
  });

  // Status HUD indicator
  ctx.font = '10px JetBrains Mono, monospace';
  if (isPlaying) {
    ctx.fillStyle = '#00ffcc';
    ctx.fillText(`● LIVE 3D FRAME: ${slices[activeSliceIdx].formatted_time} (Slice ${activeSliceIdx + 1}/${numSlices})`, w - 320, 24);
  } else {
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(`● 3D WATERFALL (${numSlices} Slices - All Visible)`, w - 240, 24);
  }
}

// 3D Waterfall Click-To-Seek
export function handleWaterfallCanvasClick(e) {
  if (!state.analysisData || !state.analysisData.waterfall_3d || !state.analysisData.waterfall_3d.slices) return;
  const rect = elements.mainWaterfallCanvas.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;
  const w = rect.width;
  const h = rect.height;

  const slices = state.analysisData.waterfall_3d.slices;
  let bestIdx = 0;
  let bestDist = 999999;

  slices.forEach((slice, idx) => {
    const zNorm = idx / Math.max(1, slices.length - 1);
    const cx = (0.5 - 0.45) * 380 * state.camZoom;
    const cy = (0.25 - 0.15) * 220 * state.camZoom;
    const cz = (zNorm - 0.5) * 340 * state.camZoom;

    const cosY = Math.cos(state.camYaw);
    const sinY = Math.sin(state.camYaw);
    const x1 = cx * cosY + cz * sinY;
    const z1 = -cx * sinY + cz * cosY;

    const cosP = Math.cos(state.camPitch);
    const sinP = Math.sin(state.camPitch);
    const y1 = cy * cosP - z1 * sinP;
    const z2 = cy * sinP + z1 * cosP;

    const dist = 600;
    const fov = dist / (dist + z2);

    const sx = w * 0.52 + x1 * fov;
    const sy = h * 0.68 - y1 * fov;

    const d = Math.hypot(clickX - sx, clickY - sy);
    if (d < bestDist) {
      bestDist = d;
      bestIdx = idx;
    }
  });

  if (bestDist < 80) {
    safeSeek(slices[bestIdx].timestamp_sec);
  }
}

// Setup 360° Mouse Orbit, Momentum Inertia, Wheel Zoom, Touch & Preset Controls
export function setupWaterfallOrbitControls(onRedraw) {
  const canvas = elements.mainWaterfallCanvas;
  if (!canvas) return;

  function stopInertia() {
    if (state.inertiaRafId !== null) {
      cancelAnimationFrame(state.inertiaRafId);
      state.inertiaRafId = null;
    }
  }

  function startInertia() {
    stopInertia();
    const FRICTION = 0.88;
    const MIN_VEL = 0.0002;

    function step() {
      state.dragVelX *= FRICTION;
      state.dragVelY *= FRICTION;

      if (Math.abs(state.dragVelX) < MIN_VEL && Math.abs(state.dragVelY) < MIN_VEL) {
        state.inertiaRafId = null;
        return;
      }

      state.camYaw += state.dragVelX;
      state.camPitch += state.dragVelY;
      state.camPitch = ((state.camPitch + Math.PI) % (2 * Math.PI)) - Math.PI;

      if (state.currentView === 'waterfall' && onRedraw) onRedraw();
      state.inertiaRafId = requestAnimationFrame(step);
    }

    state.inertiaRafId = requestAnimationFrame(step);
  }

  // Mouse Down
  canvas.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    stopInertia();
    state.isDragging3D = true;
    state.dragStartX = e.clientX;
    state.dragStartY = e.clientY;
    state.dragVelX = 0; state.dragVelY = 0;
    state.lastDragDx = 0; state.lastDragDy = 0;
    canvas.style.cursor = 'grabbing';
  });

  // Mouse Move
  window.addEventListener('mousemove', (e) => {
    if (!state.isDragging3D) return;
    const dx = e.clientX - state.dragStartX;
    const dy = e.clientY - state.dragStartY;
    state.dragStartX = e.clientX;
    state.dragStartY = e.clientY;

    state.camYaw += dx * 0.008;
    state.camPitch += dy * 0.008;
    state.camPitch = ((state.camPitch + Math.PI) % (2 * Math.PI)) - Math.PI;

    state.lastDragDx = dx * 0.008;
    state.lastDragDy = dy * 0.008;

    if (state.currentView === 'waterfall' && onRedraw) onRedraw();
  });

  // Mouse Up
  window.addEventListener('mouseup', () => {
    if (!state.isDragging3D) return;
    state.isDragging3D = false;
    canvas.style.cursor = 'grab';
    state.dragVelX = state.lastDragDx;
    state.dragVelY = state.lastDragDy;
    if (state.currentView === 'waterfall') startInertia();
  });

  canvas.addEventListener('mouseenter', () => {
    if (!state.isDragging3D) canvas.style.cursor = 'grab';
  });
  canvas.addEventListener('mouseleave', () => {
    if (!state.isDragging3D) canvas.style.cursor = '';
  });

  // Wheel Zoom
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    state.camZoom = Math.max(0.5, Math.min(2.5, state.camZoom - e.deltaY * 0.0012));
    if (state.currentView === 'waterfall' && onRedraw) onRedraw();
  }, { passive: false });

  // Touch Support
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    stopInertia();
    state.isDragging3D = false;
    state.dragVelX = 0; state.dragVelY = 0;

    if (e.touches.length === 1) {
      state.isDragging3D = true;
      state.dragStartX = e.touches[0].clientX;
      state.dragStartY = e.touches[0].clientY;
      state.lastDragDx = 0; state.lastDragDy = 0;
      state.touch1 = e.touches[0];
      state.touch2 = null;
    } else if (e.touches.length === 2) {
      state.isDragging3D = false;
      state.touch1 = e.touches[0];
      state.touch2 = e.touches[1];
      state.pinchStartDist = Math.hypot(
        state.touch2.clientX - state.touch1.clientX,
        state.touch2.clientY - state.touch1.clientY
      );
      state.pinchStartZoom = state.camZoom;
    }
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (e.touches.length === 1 && state.isDragging3D) {
      const dx = e.touches[0].clientX - state.dragStartX;
      const dy = e.touches[0].clientY - state.dragStartY;
      state.dragStartX = e.touches[0].clientX;
      state.dragStartY = e.touches[0].clientY;

      state.camYaw += dx * 0.008;
      state.camPitch += dy * 0.008;
      state.camPitch = ((state.camPitch + Math.PI) % (2 * Math.PI)) - Math.PI;

      state.lastDragDx = dx * 0.008;
      state.lastDragDy = dy * 0.008;
      if (state.currentView === 'waterfall' && onRedraw) onRedraw();
    } else if (e.touches.length === 2 && state.touch1 && state.touch2) {
      const curDist = Math.hypot(
        e.touches[1].clientX - e.touches[0].clientX,
        e.touches[1].clientY - e.touches[0].clientY
      );
      if (state.pinchStartDist > 0) {
        const factor = curDist / state.pinchStartDist;
        state.camZoom = Math.max(0.5, Math.min(2.5, state.pinchStartZoom * factor));
        if (state.currentView === 'waterfall' && onRedraw) onRedraw();
      }
    }
  }, { passive: false });

  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    if (state.isDragging3D) {
      state.isDragging3D = false;
      state.dragVelX = state.lastDragDx;
      state.dragVelY = state.lastDragDy;
      if (state.currentView === 'waterfall') startInertia();
    }
    if (e.touches.length === 0) {
      state.touch1 = null;
      state.touch2 = null;
    }
  }, { passive: false });

  canvas.addEventListener('click', handleWaterfallCanvasClick);

  // Camera Presets
  function setCameraPresetActive(btn) {
    [elements.btnPresetSketch, elements.btnPresetIso, elements.btnPresetFront].forEach(b => {
      if (b) b.classList.remove('active');
    });
    if (btn) btn.classList.add('active');
  }

  if (elements.btnPresetSketch) {
    elements.btnPresetSketch.addEventListener('click', () => {
      stopInertia();
      setCameraPresetActive(elements.btnPresetSketch);
      state.camYaw = -0.68;
      state.camPitch = 0.42;
      state.camZoom = 1.0;
      if (state.currentView === 'waterfall' && onRedraw) onRedraw();
    });
  }

  if (elements.btnPresetIso) {
    elements.btnPresetIso.addEventListener('click', () => {
      stopInertia();
      setCameraPresetActive(elements.btnPresetIso);
      state.camYaw = -Math.PI / 4;
      state.camPitch = 0.61;
      state.camZoom = 1.0;
      if (state.currentView === 'waterfall' && onRedraw) onRedraw();
    });
  }

  if (elements.btnPresetFront) {
    elements.btnPresetFront.addEventListener('click', () => {
      stopInertia();
      setCameraPresetActive(elements.btnPresetFront);
      state.camYaw = 0;
      state.camPitch = 0;
      state.camZoom = 1.0;
      if (state.currentView === 'waterfall' && onRedraw) onRedraw();
    });
  }
}
