/**
 * Multi Image Multi Out / Get Multi Out — frontend extension
 *
 * Multi Image Multi Out: a plain ordered list (no thumbnails) where the row
 * order IS the output order: row 1 → image_1, row 2 → image_2, … (max 10),
 * plus a trailing `images` output carrying the processed batch.
 *
 * Socket count = clamp(max(images, fixed_outputs), 0, 10). Slots with no image
 * are labelled "(empty)" and the backend returns an ExecutionBlocker for them,
 * so whatever is connected downstream is skipped instead of erroring.
 *
 * Get Multi Out: takes that `images` batch and slices it into its own sockets
 * (no reloading/resizing), also driven by fixed_outputs — same blocker
 * behaviour for indices beyond the batch.
 */

import { app } from "../../scripts/app.js";

const NODE_TYPE   = "NeoMultiImageMultiOut";
const GET_NODE_TYPE = "NeoGetMultiOut";
const UPLOAD_URL  = "/multi_image_loader/upload";
const MAX_OUTPUTS = 10;
const LIST_WIDGET = "image_list";
const IMAGES_OUT  = "images";

const NODE_HEADER_H  = 30;
const NODE_SLOT_H    = 22;
const NODE_PADDING_V = 12;

const DROP_TEXT = "Drop images here to replace · click to upload";

// ─── styles ──────────────────────────────────────────────────────────────────

function injectStyles() {
  if (document.getElementById("mimo-style")) return;
  const style = document.createElement("style");
  style.id = "mimo-style";
  style.textContent = `
    .mimo-root{display:flex;flex-direction:column;gap:6px;height:100%;box-sizing:border-box;padding:6px;
      font:12px/1.3 system-ui,-apple-system,sans-serif;color:#d8d8d8;user-select:none}
    .mimo-drop{border:1px dashed #555;border-radius:6px;padding:7px 6px;text-align:center;
      font-size:11px;color:#8f8f8f;cursor:pointer;flex:0 0 auto;transition:background .12s,border-color .12s}
    .mimo-drop:hover{border-color:#6b8fb5;color:#b9c9d8}
    .mimo-drop.over{border-color:#4a9eff;color:#9ecbff;background:rgba(74,158,255,.10)}
    .mimo-list{flex:1 1 auto;overflow-y:auto;overflow-x:hidden;display:flex;flex-direction:column;
      gap:3px;min-height:48px}
    .mimo-empty{color:#6f6f6f;font-size:11px;text-align:center;padding:10px 4px;line-height:1.4}
    .mimo-row{display:flex;align-items:center;gap:6px;background:#2a2a2a;border:1px solid #333;
      border-radius:4px;padding:3px 5px}
    .mimo-idx{flex:0 0 auto;width:22px;height:18px;display:flex;align-items:center;justify-content:center;
      border-radius:3px;background:#3a6ea5;color:#fff;font-weight:600;font-size:10px}
    .mimo-name{flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;
      color:#cfcfcf;direction:rtl;text-align:left}
    .mimo-btn{flex:0 0 auto;width:18px;height:18px;line-height:16px;text-align:center;padding:0;
      border:1px solid #444;border-radius:3px;background:#333;color:#bbb;font-size:10px;cursor:pointer}
    .mimo-btn:hover:not(:disabled){background:#444;color:#fff}
    .mimo-btn:disabled{opacity:.3;cursor:default}
    .mimo-btn.del:hover{background:#7a2c2c;border-color:#a33;color:#fff}
    .mimo-foot{flex:0 0 auto;display:flex;align-items:center;justify-content:space-between;
      font-size:10px;color:#7d7d7d;gap:6px}
    .mimo-clear{font-size:10px;padding:1px 6px;border:1px solid #444;border-radius:3px;
      background:#2c2c2c;color:#9a9a9a;cursor:pointer}
    .mimo-clear:hover{background:#3a2c2c;border-color:#a33;color:#e0b0b0}
  `;
  document.head.appendChild(style);
}

// ─── helpers ─────────────────────────────────────────────────────────────────

async function uploadFiles(files) {
  const fd = new FormData();
  for (const f of files) fd.append("images", f, f.name);
  const resp = await fetch(UPLOAD_URL, { method: "POST", body: fd });
  if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
  const json = await resp.json();
  if (!json.success) throw new Error(json.error || "Unknown upload error");
  return json.files || [];
}

function getListWidget(node) {
  return node.widgets?.find((w) => w.name === LIST_WIDGET);
}

function readFiles(node) {
  const w = getListWidget(node);
  const raw = w?.value ?? "[]";
  if (Array.isArray(raw)) return raw.filter((x) => typeof x === "string");
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}

function writeFiles(node, files) {
  const w = getListWidget(node);
  if (w) w.value = JSON.stringify(files);
}

function addOutputSocket(node, name, type = "IMAGE") {
  node.outputs ||= [];
  try {
    node.addOutput(name, type);
  } catch {
    node.outputs.push({ name, type, links: null });
  }
  // Normalise in case a litegraph build uses (type, name) ordering.
  const slot = node.outputs[node.outputs.length - 1];
  if (slot) {
    slot.name = name;
    slot.type = type;
    if (slot.links === undefined) slot.links = null;
  }
}

function getNumberWidget(node, name, fallback = 0) {
  const w = node.widgets?.find((ww) => ww.name === name);
  const n = Number(w?.value);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * The processed batch output ("images") is always the LAST socket, and the
 * per-image sockets live in front of it. That way adding/removing image sockets
 * never renumbers existing image links.
 */
function ensureImagesOutput(node) {
  node.outputs ||= [];
  const last = node.outputs[node.outputs.length - 1];
  if (last && last.name === IMAGES_OUT) return;

  if (last && last.type === "STRING") {
    // Workflow saved with the old JSON-output version → convert in place.
    last.name = IMAGES_OUT;
    last.type = "IMAGE";
    return;
  }
  // Appending never shifts existing slots.
  addOutputSocket(node, IMAGES_OUT, "IMAGE");
}

/** Fix the batch output's link origin after an image socket was inserted. */
function fixImagesOutputLinks(node) {
  const idx = node.outputs.length - 1;
  const slot = node.outputs[idx];
  if (!slot || slot.name !== IMAGES_OUT || !node.graph) return;
  for (const id of slot.links || []) {
    const link = node.graph._links?.get(id);
    if (link) link.origin_slot = idx;
  }
}

/** Insert a new image socket directly before the trailing batch output. */
function insertImageSocket(node, name) {
  const strIdx = node.outputs.length - 1;
  const slot = { name, type: "IMAGE", links: null };
  node.outputs.splice(strIdx, 0, slot);
  node.onOutputAdded?.(slot);
  fixImagesOutputLinks(node);
  // Links may not be registered yet when this runs during configure.
  setTimeout(() => fixImagesOutputLinks(node), 0);
  return slot;
}

/**
 * Sockets = clamp(max(images, fixed_outputs), 0, 10) image outputs + the
 * always-present processed batch at the end.
 */
function syncOutputs(node, imageCount) {
  node.outputs ||= [];
  ensureImagesOutput(node);

  const fixed = Math.max(0, Math.min(MAX_OUTPUTS, getNumberWidget(node, "fixed_outputs")));
  const want = Math.min(MAX_OUTPUTS, Math.max(imageCount, fixed));

  // Everything except the trailing batch socket is an image socket.
  while (node.outputs.length - 1 > want) removeOutputSocket(node, node.outputs.length - 2);
  while (node.outputs.length - 1 < want) insertImageSocket(node, `image_${node.outputs.length}`);

  node.outputs.forEach((slot, i) => {
    slot.type = "IMAGE";
    slot.name = i === node.outputs.length - 1
      ? IMAGES_OUT
      : (i + 1 <= imageCount ? `image_${i + 1}` : `image_${i + 1} (empty)`);
  });

  node.setDirtyCanvas(true, true);
  app.graph?.setDirtyCanvas(true, true);
}

/** "Get Multi Out": sockets are driven purely by fixed_outputs. */
function syncGetOutputs(node) {
  node.outputs ||= [];
  const fixed = Math.max(0, Math.min(MAX_OUTPUTS, getNumberWidget(node, "fixed_outputs", 5)));

  while (node.outputs.length > fixed) removeOutputSocket(node, node.outputs.length - 1);
  while (node.outputs.length < fixed) addOutputSocket(node, `image_${node.outputs.length + 1}`, "IMAGE");

  node.outputs.forEach((slot, i) => {
    slot.name = `image_${i + 1}`;
    slot.type = "IMAGE";
  });

  node.setDirtyCanvas(true, true);
  app.graph?.setDirtyCanvas(true, true);
}

/**
 * Recover the image list if a workflow saved before `fixed_outputs` existed
 * shifted the positional widgets_values.
 */
function repairLegacyWidgets(node, info) {
  const vals = info?.widgets_values;
  if (!Array.isArray(vals)) return;
  const looksLikeList = (v) => typeof v === "string" && v.trim().startsWith("[");

  const fw = node.widgets?.find((w) => w.name === "fixed_outputs");
  const lw = getListWidget(node);

  if (fw && (looksLikeList(fw.value) || !Number.isFinite(Number(fw.value)))) fw.value = 0;
  if (lw && !looksLikeList(lw.value)) {
    const src = vals.find(looksLikeList);
    if (src) lw.value = src;
  }
}

function removeOutputSocket(node, index) {
  if (typeof node.removeOutput === "function") {
    node.removeOutput(index);
    return;
  }
  const slot = node.outputs?.[index];
  if (slot?.links?.length) {
    for (const linkId of [...slot.links]) {
      const link = node.graph?._links?.get(linkId);
      if (link) node.graph.removeLink(linkId);
    }
  }
  node.outputs.splice(index, 1);
}

// ─── list widget ─────────────────────────────────────────────────────────────

function makeButton(label, title, onClick, disabled = false) {
  const btn = document.createElement("button");
  btn.className = label === "✕" ? "mimo-btn del" : "mimo-btn";
  btn.type = "button";
  btn.textContent = label;
  btn.title = title;
  btn.disabled = disabled;
  btn.addEventListener("pointerdown", (e) => e.stopPropagation());
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    onClick();
  });
  return btn;
}

function createListElement(node) {
  const root = document.createElement("div");
  root.className = "mimo-root";

  const drop = document.createElement("div");
  drop.className = "mimo-drop";
  drop.textContent = DROP_TEXT;

  const list = document.createElement("div");
  list.className = "mimo-list";

  const foot = document.createElement("div");
  foot.className = "mimo-foot";

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.multiple = true;
  fileInput.accept = "image/*";
  fileInput.style.display = "none";

  root.append(drop, list, foot, fileInput);

  // Don't let the canvas hijack scrolling / pointer events inside the list.
  root.addEventListener("wheel", (e) => e.stopPropagation(), { passive: true });

  function commit(files) {
    writeFiles(node, files);
    syncOutputs(node, files.length);
    render();
  }

  function move(index, delta) {
    const files = readFiles(node);
    const target = index + delta;
    if (target < 0 || target >= files.length) return;
    [files[index], files[target]] = [files[target], files[index]];
    commit(files);
  }

  function removeAt(index) {
    const files = readFiles(node);
    files.splice(index, 1);
    commit(files);
  }

  function render() {
    const files = readFiles(node);

    list.replaceChildren();
    if (!files.length) {
      const empty = document.createElement("div");
      empty.className = "mimo-empty";
      empty.textContent = "No images yet — each image becomes one output socket.";
      list.appendChild(empty);
    } else {
      files.forEach((name, i) => {
        const row = document.createElement("div");
        row.className = "mimo-row";

        const idx = document.createElement("span");
        idx.className = "mimo-idx";
        idx.textContent = String(i + 1);
        idx.title = `Output: image_${i + 1}`;

        const label = document.createElement("span");
        label.className = "mimo-name";
        label.textContent = name;
        label.title = name;

        row.append(
          idx,
          label,
          makeButton("↑", "Move up", () => move(i, -1), i === 0),
          makeButton("↓", "Move down", () => move(i, 1), i === files.length - 1),
          makeButton("✕", "Remove image", () => removeAt(i))
        );
        list.appendChild(row);
      });
    }

    foot.replaceChildren();
    const status = document.createElement("span");
    const n = Math.min(files.length, MAX_OUTPUTS);
    const fixed = Math.max(0, Math.min(MAX_OUTPUTS, getNumberWidget(node, "fixed_outputs")));
    const shown = Math.max(n, fixed);
    status.textContent =
      `${files.length} / ${MAX_OUTPUTS} image(s) · ${shown} output${shown === 1 ? "" : "s"}` +
      (fixed ? ` (fixed ${fixed})` : "");
    foot.appendChild(status);

    if (files.length) {
      const clear = document.createElement("button");
      clear.className = "mimo-clear";
      clear.type = "button";
      clear.textContent = "Clear";
      clear.title = "Remove all images";
      clear.addEventListener("pointerdown", (e) => e.stopPropagation());
      clear.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        commit([]);
      });
      foot.appendChild(clear);
    }
  }

  async function addFiles(files) {
    if (!files?.length) return;

    drop.textContent = "Uploading…";
    try {
      const saved = await uploadFiles(files);
      // ComfyUI-style behaviour: a new drop REPLACES the current list.
      const next = saved.slice(0, MAX_OUTPUTS);
      commit(next);
      if (saved.length > MAX_OUTPUTS) {
        drop.textContent = `Only ${MAX_OUTPUTS} images are supported — extras ignored`;
        setTimeout(() => (drop.textContent = DROP_TEXT), 1800);
      }
    } catch (err) {
      console.warn("[MIMO] upload failed:", err);
      drop.textContent = "Upload failed";
      setTimeout(() => (drop.textContent = DROP_TEXT), 1800);
    } finally {
      if (drop.textContent === "Uploading…") drop.textContent = DROP_TEXT;
    }
  }

  // Click / drag'n'drop handling
  drop.addEventListener("pointerdown", (e) => e.stopPropagation());
  drop.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    fileInput.click();
  });
  fileInput.addEventListener("change", async () => {
    const files = [...(fileInput.files || [])];
    fileInput.value = "";
    await addFiles(files);
  });
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.stopPropagation();
    drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", async (e) => {
    e.preventDefault();
    e.stopPropagation();
    drop.classList.remove("over");
    const files = [...(e.dataTransfer?.files || [])].filter((f) => (f.type || "").startsWith("image/"));
    await addFiles(files);
  });

  root._mimoRefresh = render;
  root._mimoAddFiles = addFiles;
  render();
  return root;
}

// ─── extension registration ──────────────────────────────────────────────────

app.registerExtension({
  name: "NeoMultiImageMultiOut",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    nodeType.canvasOnly = true; // force classic canvas rendering (Nodes 2.0 safe)

    const origOnCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      origOnCreated?.apply(this, arguments);
      const node = this;

      injectStyles();

      if (!node.size || node.size[0] < 300) node.setSize([340, 420]);
      const el = createListElement(node);

      node.computeSize = function () {
        return [300, 260];
      };

      const domWidget = node.addDOMWidget("mimo_list", "MultiImageMultiOutWidget", el, {
        getValue() {
          return "";
        },
        setValue() {},
        computeSize(width) {
          const trueWidth = node.size[0];
          let widgetH = 0;
          for (const w of node.widgets || []) {
            if (w.name === "mimo_list") continue;
            widgetH += (w.computeSize ? w.computeSize(trueWidth)[1] : 20) + 4;
          }
          const slotsH = Math.max(node.outputs?.length || 0, 1) * NODE_SLOT_H;
          const chromeH = NODE_HEADER_H + slotsH + widgetH + NODE_PADDING_V;
          const h = Math.max(110, node.size[1] - chromeH);
          el.style.height = `${h}px`;
          return [trueWidth, h];
        },
      });
      node._mimoDom = domWidget;

      // Hide the internal JSON widget (must not be "converted-widget", or it
      // would be dropped from widgets_values and shift the other values).
      setTimeout(() => {
        const w = getListWidget(node);
        if (w) {
          w.computeSize = () => [0, -4];
          w.draw = function () {};
          if (w.inputEl) {
            w.inputEl.style.display = "none";
            w.inputEl.type = "hidden";
          }
          if (w.element) w.element.style.display = "none";
        }
        // Re-sync once all widgets exist (safe for older/newer frontends).
        syncOutputs(node, readFiles(node).length);
        el._mimoRefresh?.();

        // Changing fixed_outputs grows/shrinks the socket list live.
        const fixedWidget = node.widgets?.find((w) => w.name === "fixed_outputs");
        if (fixedWidget) {
          const origCb = fixedWidget.callback;
          fixedWidget.callback = function (...args) {
            origCb?.apply(this, args);
            syncOutputs(node, readFiles(node).length);
            el._mimoRefresh?.();
          };
        }
      }, 0);

      // New nodes start with the 10 declared sockets → trim to current list.
      syncOutputs(node, readFiles(node).length);
      el._mimoRefresh?.();

      node.onDrop = async function (e) {
        e?.preventDefault?.();
        e?.stopPropagation?.();
        try {
          const files = [...(e.dataTransfer?.files || [])].filter((f) =>
            (f.type || "").startsWith("image/")
          );
          if (files.length) await el._mimoAddFiles?.(files);
        } catch (err) {
          console.warn("[MIMO] drop failed:", err);
        }
        return false;
      };

      const origOnResize = node.onResize;
      node.onResize = function () {
        origOnResize?.apply(this, arguments);
        requestAnimationFrame(() => el._mimoRefresh?.());
      };
    };

    // Restore list + sockets when a workflow is loaded.
    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      origOnConfigure?.apply(this, arguments);
      const node = this;
      repairLegacyWidgets(node, info);
      syncOutputs(node, readFiles(node).length);
      node._mimoDom?.element?._mimoRefresh?.();
    };
  },
});

// ─── Get Multi Out ───────────────────────────────────────────────────────────

app.registerExtension({
  name: "NeoGetMultiOut",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== GET_NODE_TYPE) return;

    nodeType.canvasOnly = true;

    const origOnCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      origOnCreated?.apply(this, arguments);
      const node = this;

      if (!node.size || node.size[0] < 240) node.setSize([280, 170]);
      node.computeSize = function () {
        return [240, 140];
      };

      // Trim the 10 declared sockets down to fixed_outputs.
      syncGetOutputs(node);

      setTimeout(() => {
        const fixedWidget = node.widgets?.find((w) => w.name === "fixed_outputs");
        if (fixedWidget) {
          const origCb = fixedWidget.callback;
          fixedWidget.callback = function (...args) {
            origCb?.apply(this, args);
            syncGetOutputs(node);
          };
        }
        syncGetOutputs(node);
      }, 0);
    };

    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      origOnConfigure?.apply(this, arguments);
      syncGetOutputs(this);
    };
  },
});
