# ComfyUI Multi-Image Loader 🖼️

**ComfyUI Multi-Image Loader** is a specialized node extension that enables seamless, interactive batch image loading directly within the ComfyUI interface. Say goodbye to manual folder paths and file-system navigation—just drag, drop, and process.

---

## 🚀 Features

*   **Interactive Drag & Drop**: Upload multiple images directly into the node widget from your local machine.
*   **No Folder Required**: Bypasses the need for pre-defined input folders; images are handled via a dedicated internal upload route.
*   **Unified Batch Tensor**: Automatically combines all uploaded images into a single `IMAGE` batch tensor `[B, H, W, C]`.
*   **Keep Proportion Fit**: The `keep_proportion` widget harmonizes inconsistent image dimensions with `stretch`, `resize`, `pad`, `crop` or `original`, plus a `crop_position` anchor (`center`, `top`, `bottom`, `left`, `right`).
*   **Persistent Storage**: The list of uploaded images is saved directly in your ComfyUI workflow JSON for easy multi-session work.
*   **Saved Image Groups**: The *Multi Image Multi Out* node can save the current image list as a reusable group (`💾 Save`) and restore or delete it later from the dropdown.
*   **Visual Previews**: (Coming Soon/Frontend) Interactive thumbnail previews for managed assets.

---

## 🛠️ Components

1.  **`neo - Load Multiple Images`** (`NeoMultiImageLoader`): the core node for drag-and-drop image batches.
2.  **`neo - Multi Media Loader`** (`NeoMultiMediaLoader`): same node and features, but the drop zone also accepts **videos** (mp4, webm, mov, mkv, avi…) and **animated GIF / WebP**. Videos are decoded into frames with `ffmpeg` and then fed through the exact same pipeline as still images.
3.  **`neo - Get Native Image From List`**: picks one image out of the native (per-image original size) list output.
4.  **`neo - Scale Control`**, **`neo - From Batch to Grid`**: downstream helpers.
5.  **JavaScript Extension**: injects the custom drag-and-drop / file-picker widget into the ComfyUI graph.

### ✂️ Keep proportion (`keep_proportion`)

Controls how each source image is conformed to the shared output canvas.

| Value | Behaviour |
|---|---|
| `stretch` | Force the target size, distorting the aspect ratio |
| `resize` | Scale proportionally to fit inside the canvas |
| `pad` | Scale proportionally, then fill the leftover area with `bg_color` (default) |
| `crop` | Crop the source to the canvas aspect ratio **first**, then scale — no background bars |
| `original` | Keep native pixels, centred on the canvas |

`crop_position` anchors the `crop` step: `center`, `top`, `bottom`, `left`, `right`.
Legacy workflow values (`letterbox`, `fill`) are migrated automatically.

### 💾 Saved image groups (Multi Image Multi Out)

The *Multi Image Multi Out* node keeps a reusable library of image lists:

*   **💾 Save** — stores the current list under the first image's name (deduplicated, up to 50 groups).
*   **Dropdown** — restores a saved group, replacing the node's current list.
*   **🗑** — deletes the selected group.

Groups live in browser `localStorage` (`mimo_groups_v1`), so they are per-machine and
independent of the workflow JSON.

### 🎬 Multi Media Loader — video options

| Widget | Default | Meaning |
|---|---|---|
| `video_fps` | `8.0` | Frames per second sampled from a video |
| `video_max_frames` | `24` | Maximum frames decoded per video |
| `gif_max_frames` | `24` | Maximum frames decoded per animated image |

Decoded frames are cached under `ComfyUI/input/mil_frames/<hash>/`, keyed by the video's
path, size, mtime and sampling settings — re-running does not re-decode.

### 📁 Custom path (`custom_path`)

Optional. Both new uploads **and** filename lookups use it:

* absolute path, e.g. `D:/projects/media`
* or a path relative to `ComfyUI/input`, e.g. `my_media`

Files outside `ComfyUI/input` are served to the UI through the
`/multi_image_loader/media_thumb` route (the standard `/view` endpoint cannot reach them).
Video thumbnails use the same route to show a poster frame.

---

## 📦 Installation

### 1. Manual installation
1.  Navigate to your `ComfyUI/custom_nodes/` folder.
2.  Clone the repository:
    ```bash
    git clone https://github.com/Latentnaut/ComfyUI-Multi-Image-Loader.git
    ```
3.  Restart ComfyUI and refresh your browser.

---

## 📖 Usage

1.  Add the **Load Multiple Images 🖼️** node (found under `image/loaders`).
2.  Click the widget or drag multiple files directly onto the node.
3.  Choose your `resize_mode`:
    *   `resize_to_first`: Resizes all images to match the dimensions of the first one in the batch.
    *   `none`: Attempt to stack (will fail if dimensions differ).
4.  Connect the output `image_batch` to any node that accepts batch image tensors (e.g., VAE Encode, Upscalers).

---

## 🤝 Contributing

Feedback and contributions are welcome! If you encounter issues or have feature requests for the UI, please open an issue on the repository.

*Created by [Latentnaut](https://github.com/Latentnaut/ComfyUI-Multi-Image-Loader)*
