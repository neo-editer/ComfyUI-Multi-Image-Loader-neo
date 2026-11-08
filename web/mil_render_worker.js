/**
 * mil_render_worker.js — OffscreenCanvas Web Worker
 *
 * Handles heavy pixel work (crop transforms, fit letterbox/crop, lasso masks)
 * off the main UI thread so ComfyUI / LiteGraph stays responsive.
 *
 * Communication protocol:
 *   Main → Worker:  { jobId, type, imageBlob, params }
 *   Worker → Main:  { jobId, dataUrl }
 *
 * Job types:
 *   "cropTransform"  — full crop/pan/zoom/flip/rotate + lasso composite
 *   "fitPreview"     — simple letterbox or center-crop fit
 */

// ── helpers ─────────────────────────────────────────────────────────────────

function normalizeFitMode(value) {
  const v = String(value ?? "pad").trim().toLowerCase().replace(/[-_ ]+/g, "_");
  const map = {
    stretch: "stretch",
    fill: "stretch",
    resize: "pad",
    pad: "pad",
    letterbox: "pad",
    keep_proportion: "crop",
    crop: "crop",
    original: "original",
  };
  return map[v] ?? "pad";
}

/** Anchor factors (0..1) for keep_proportion cropping. */
function cropAnchorFactors(value) {
  const p = String(value ?? "center").trim().toLowerCase().replace(/[-_ ]+/g, "_");
  if (p === "left") return [0, 0.5];
  if (p === "right") return [1, 0.5];
  if (p === "top") return [0.5, 0];
  if (p === "bottom") return [0.5, 1];
  return [0.5, 0.5];
}

/**
 * Draw lasso ops onto a sub-canvas and composite the bg-fill cutout
 * back onto the main context. All coordinates are normalised [0..1].
 */
function applyLassoMask(ctx, lassoOps, lassoInverted, dw, dh, bgColor) {
  const mw = Math.ceil(dw), mh = Math.ceil(dh);
  if (mw < 1 || mh < 1) return;

  // 1. Build the lasso mask (white = inside lasso)
  const mc = new OffscreenCanvas(mw, mh);
  const mx = mc.getContext("2d");
  for (const op of lassoOps) {
    if (!op.points || op.points.length < 3) continue;
    mx.globalCompositeOperation = op.mode === "add" ? "source-over" : "destination-out";
    mx.fillStyle = "white";
    mx.beginPath();
    mx.moveTo(op.points[0][0] * mw, op.points[0][1] * mh);
    for (let k = 1; k < op.points.length; k++) {
      mx.lineTo(op.points[k][0] * mw, op.points[k][1] * mh);
    }
    mx.closePath();
    mx.fill();
  }
  if (lassoInverted) {
    mx.globalCompositeOperation = "xor";
    mx.fillStyle = "white";
    mx.fillRect(0, 0, mw, mh);
  }
  mx.globalCompositeOperation = "source-over";

  // 2. Build a bg-color rect and punch out the lasso shape
  const oc = new OffscreenCanvas(mw, mh);
  const ox = oc.getContext("2d");
  ox.fillStyle = bgColor;
  ox.fillRect(0, 0, mw, mh);
  ox.globalCompositeOperation = "destination-out";
  ox.drawImage(mc, 0, 0);
  ox.globalCompositeOperation = "source-over";

  // 3. Overlay onto the main canvas
  ctx.drawImage(oc, -dw / 2, -dh / 2);
}

// ── message handler ─────────────────────────────────────────────────────────

self.addEventListener("message", async (event) => {
  const { jobId, type, imageBlob, params } = event.data;

  try {
    const bitmap = await createImageBitmap(imageBlob);
    let dataUrl;

    if (type === "cropTransform") {
      dataUrl = await handleCropTransform(bitmap, params);
    } else if (type === "fitPreview") {
      dataUrl = await handleFitPreview(bitmap, params);
    } else {
      throw new Error(`Unknown job type: ${type}`);
    }

    bitmap.close();
    self.postMessage({ jobId, dataUrl });
  } catch (err) {
    self.postMessage({ jobId, error: err.message });
  }
});


// ── crop transform ──────────────────────────────────────────────────────────

async function handleCropTransform(bitmap, p) {
  const { refW, refH, bgColor, fitMode } = p;
  const mode = normalizeFitMode(fitMode);
  const t = p.transform;

  const cvs = new OffscreenCanvas(refW, refH);
  const ctx = cvs.getContext("2d");

  // Background fill
  ctx.fillStyle = bgColor;
  ctx.fillRect(0, 0, refW, refH);

  // Source crop region
  const hasCR = t.cx != null && (t.cx > 0 || t.cy > 0 || t.cw < 1 || t.ch < 1);
  let srcX = hasCR ? t.cx * bitmap.width : 0;
  let srcY = hasCR ? t.cy * bitmap.height : 0;
  let srcW = hasCR ? t.cw * bitmap.width : bitmap.width;
  let srcH = hasCR ? t.ch * bitmap.height : bitmap.height;

  // ── ASPECT CROP: keep_proportion crops the SOURCE before any scaling ──
  if (mode === "crop") {
    const targetAR = refW / refH, srcAR = srcW / srcH;
    let cw = srcW, ch = srcH;
    if (srcAR > targetAR) cw = srcH * targetAR;
    else ch = srcW / targetAR;
    const [ax, ay] = cropAnchorFactors(p.cropPosition);
    srcX += (srcW - cw) * ax;
    srcY += (srcH - ch) * ay;
    srcW = Math.max(1, cw);
    srcH = Math.max(1, ch);
  }

  // Transform parameters
  const ox = t.ox ?? 0, oy = t.oy ?? 0, sc = t.scale ?? 1;
  const fH = !!(t.flipH), fV = !!(t.flipV);
  const rot = (t.rotate || 0) * Math.PI / 180;
  const cosA = Math.abs(Math.cos(rot)), sinA = Math.abs(Math.sin(rot));

  // Rotated bounding box
  const rW = srcW * cosA + srcH * sinA;
  const rH = srcW * sinA + srcH * cosA;

  // Base fit scale (letterbox vs crop mode)
  const globalScale = p.globalScale ?? 1.0;
  const bf = mode === "crop"
    ? Math.max(refW / rW, refH / rH)
    : Math.min(refW / rW, refH / rH);
  const eff = bf * sc;
  const dw = srcW * eff, dh = srcH * eff;

  ctx.save();
  if (globalScale !== 1.0) {
    ctx.translate(refW / 2, refH / 2);
    ctx.scale(globalScale, globalScale);
    ctx.translate(-refW / 2, -refH / 2);
    
    // Explicit clip guarantees any image parts brought 'inside' the frame by scaling are cropped
    ctx.beginPath();
    ctx.rect(0, 0, refW, refH);
    ctx.clip();
  }

  ctx.translate(refW / 2 + ox * refW, refH / 2 + oy * refH);
  ctx.rotate(rot);
  ctx.scale(fH ? -1 : 1, fV ? -1 : 1);
  ctx.drawImage(bitmap, srcX, srcY, srcW, srcH, -dw / 2, -dh / 2, dw, dh);

  // Lasso mask overlay — skip in Paint mode where the lasso defines a
  // brush-clip region, not a crop mask (mirrors backend _generate_lasso_mask
  // which returns None when lassoIsPaint is set).
  const tOps = t.lassoOps;
  if (tOps && tOps.length > 0 && !t.lassoIsPaint) {
    applyLassoMask(ctx, tOps, t.lassoInverted, dw, dh, bgColor);
  }

  ctx.restore();

  // Convert to data URL via blob → base64
  try {
    const blob = await cvs.convertToBlob({ type: "image/jpeg", quality: 0.92 });
    const arrayBuf = await blob.arrayBuffer();
    const bytes = new Uint8Array(arrayBuf);
    let binary = "";
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    return "data:image/jpeg;base64," + btoa(binary);
  } catch (e) {
    console.warn("[MIL Worker] convertToBlob failed inside handleCropTransform:", e);
    return "";
  }
}


// ── fit preview (letterbox / crop) ──────────────────────────────────────────

async function handleFitPreview(bitmap, p) {
  const { targetW, targetH, mode, bgColor } = p;
  const normalizedMode = normalizeFitMode(mode);

  if (normalizedMode === "original") {
    const cvs = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = cvs.getContext("2d");
    ctx.clearRect(0, 0, bitmap.width, bitmap.height);
    ctx.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height);
    const blob = await cvs.convertToBlob({ type: "image/png" });
    const arrayBuf = await blob.arrayBuffer();
    const bytes = new Uint8Array(arrayBuf);
    let binary = "";
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    return "data:image/png;base64," + btoa(binary);
  }

  const cvs = new OffscreenCanvas(targetW, targetH);
  const ctx = cvs.getContext("2d");

  // Always fill bg first – critical for blank/solid panels in any fit mode
  ctx.fillStyle = bgColor;
  ctx.fillRect(0, 0, targetW, targetH);

  const globalScale = p.globalScale ?? 1.0;
  
  ctx.save();
  if (globalScale !== 1.0) {
    ctx.translate(targetW / 2, targetH / 2);
    ctx.scale(globalScale, globalScale);
    ctx.translate(-targetW / 2, -targetH / 2);
    
    ctx.beginPath();
    ctx.rect(0, 0, targetW, targetH);
    ctx.clip();
  }

  if (normalizedMode === "stretch") {
    ctx.drawImage(bitmap, 0, 0, targetW, targetH);
    ctx.restore();
    return await canvasToDataUrl(cvs);
  }

  if (normalizedMode === "crop") {
    // Crop the source to the target aspect ratio FIRST, then scale to fit.
    const targetAR = targetW / targetH, srcAR = bitmap.width / bitmap.height;
    let sw = bitmap.width, sh = bitmap.height;
    if (srcAR > targetAR) sw = sh * targetAR;
    else sh = sw / targetAR;
    const [ax, ay] = cropAnchorFactors(p.cropPosition);
    const sx = (bitmap.width - sw) * ax, sy = (bitmap.height - sh) * ay;
    ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, targetW, targetH);
    ctx.restore();
    return await canvasToDataUrl(cvs);
  }

  const baseScale = Math.min(targetW / bitmap.width, targetH / bitmap.height);
  const dw = bitmap.width * baseScale, dh = bitmap.height * baseScale;
  const dx = (targetW - dw) / 2, dy = (targetH - dh) / 2;

  ctx.drawImage(bitmap, dx, dy, dw, dh);

  ctx.restore();

  return await canvasToDataUrl(cvs);
}

async function canvasToDataUrl(cvs) {
  try {
    const blob = await cvs.convertToBlob({ type: "image/jpeg", quality: 0.92 });
    const arrayBuf = await blob.arrayBuffer();
    const bytes = new Uint8Array(arrayBuf);
    let binary = "";
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    return "data:image/jpeg;base64," + btoa(binary);
  } catch (e) {
    console.warn("[MIL Worker] convertToBlob failed:", e);
    return "";
  }
}
