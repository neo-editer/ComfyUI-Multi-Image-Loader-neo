import os
import json
import math
import hashlib
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw
from pathlib import Path
from server import PromptServer
from aiohttp import web
import folder_paths

# ─── Upload route ─────────────────────────────────────────────────────────────

@PromptServer.instance.routes.post("/multi_image_loader/upload")
async def upload_images_handler(request):
    """
    Accepts multipart/form-data with multiple files named 'images'.
    Saves each file to ComfyUI's input folder and returns the list of filenames.
    """
    try:
        reader = await request.multipart()
        saved = []
        input_dir = Path(folder_paths.get_input_directory())
        input_dir.mkdir(parents=True, exist_ok=True)

        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name != "images":
                continue

            filename = part.filename or "upload.png"
            filename = os.path.basename(filename)
            data = await part.read()

            hash_prefix = hashlib.md5(data).hexdigest()[:8]
            stem, ext = os.path.splitext(filename)
            counter = 0
            while True:
                suffix = f"_{counter}" if counter > 0 else ""
                safe_name = f"mil_{hash_prefix}_{stem}{suffix}{ext}"
                dest = input_dir / safe_name
                if not dest.exists():
                    break
                counter += 1
            dest.write_bytes(data)
            saved.append(safe_name)

        return web.json_response({"success": True, "files": saved})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)


# ─── Inpaint route (Content-Aware Fill) ───────────────────────────────────────

@PromptServer.instance.routes.post("/multi_image_loader/inpaint")
async def inpaint_handler(request):
    """Accept base64 image + mask, run cv2.inpaint (Telea), return base64 result."""
    import asyncio, base64, io as _io
    try:
        import cv2
    except ImportError:
        return web.json_response({"error": "opencv-python not installed"}, status=500)

    try:
        data = await request.json()
        img_b64 = data.get("image", "")
        mask_b64 = data.get("mask", "")
        if not img_b64 or not mask_b64:
            return web.Response(status=400, text="Missing image or mask")

        for prefix in ("data:image/png;base64,", "data:image/webp;base64,", "data:image/jpeg;base64,"):
            if img_b64.startswith(prefix):
                img_b64 = img_b64[len(prefix):]
            if mask_b64.startswith(prefix):
                mask_b64 = mask_b64[len(prefix):]

        loop = asyncio.get_event_loop()

        def _run():
            img_pil_raw = Image.open(_io.BytesIO(base64.b64decode(img_b64)))
            has_alpha = img_pil_raw.mode == "RGBA"
            alpha_channel = None
            if has_alpha:
                alpha_channel = img_pil_raw.split()[3]
            img_pil = img_pil_raw.convert("RGB")
            mask_pil = Image.open(_io.BytesIO(base64.b64decode(mask_b64))).convert("L")
            if mask_pil.size != img_pil.size:
                mask_pil = mask_pil.resize(img_pil.size, Image.NEAREST)
            img_cv = np.array(img_pil)[:, :, ::-1].copy()
            mask_cv = 255 - np.array(mask_pil)
            method_str = data.get("method", "telea")
            method = cv2.INPAINT_NS if method_str == "navier-stokes" else cv2.INPAINT_TELEA
            if method_str == "navier-stokes":
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                dilated_mask = cv2.dilate(mask_cv, kernel, iterations=1)
            else:
                dilated_mask = mask_cv
            img_cv[dilated_mask == 255] = [128, 128, 128]
            if method_str == "navier-stokes":
                blurred = cv2.GaussianBlur(img_cv, (0, 0), sigmaX=3)
                orig_cv = np.array(img_pil)[:, :, ::-1].copy()
                img_cv = np.where(dilated_mask[:, :, None] == 255, blurred, orig_cv)
            result = cv2.inpaint(img_cv, dilated_mask, inpaintRadius=7, flags=method)
            result_rgb = result[:, :, ::-1]
            result_pil = Image.fromarray(result_rgb)
            if has_alpha and alpha_channel is not None:
                alpha_np = np.array(alpha_channel)
                mask_np = np.array(mask_cv)
                alpha_np = np.where(mask_np > 128, 255, alpha_np).astype(np.uint8)
                result_pil = result_pil.convert("RGBA")
                result_pil.putalpha(Image.fromarray(alpha_np))
            buf = _io.BytesIO()
            result_pil.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        result_b64 = await loop.run_in_executor(None, _run)
        return web.json_response({"image": result_b64})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# ─── Live-preview route ───────────────────────────────────────────────────────

@PromptServer.instance.routes.post("/multi_image_loader/preview")
async def preview_transform_handler(request):
    """Apply crop transform and return base64 JPEG for live inpaint preview."""
    import asyncio, base64, io as _io

    try:
        data = await request.json()
        filename = data.get("filename", "")
        transform = data.get("transform", {})
        ref_w = max(32, int(data.get("refW", 512)))
        ref_h = max(32, int(data.get("refH", 512)))
        global_scale = float(data.get("globalScale", 1.0))
        fit_mode = data.get("fitMode", "letterbox")
        if fit_mode not in ("letterbox", "crop"):
            fit_mode = "letterbox"

        if not filename:
            return web.Response(status=400, text="Missing filename")

        filename = os.path.basename(filename)
        path = os.path.join(folder_paths.get_input_directory(), filename)
        if not os.path.isfile(path):
            return web.Response(status=404, text=f"File not found: {filename}")

        loop = asyncio.get_event_loop()
        PREVIEW_MAX = 768

        def _run():
            img = Image.open(path)
            has_alpha = (img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info))
            if has_alpha:
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")
            w, h = img.size
            if max(ref_w, ref_h) > PREVIEW_MAX:
                s = PREVIEW_MAX / max(ref_w, ref_h)
                p_ref_w, p_ref_h = max(32, int(ref_w * s)), max(32, int(ref_h * s))
            else:
                p_ref_w, p_ref_h = ref_w, ref_h
            return _apply_crop_transform(img, transform, p_ref_w, p_ref_h, fit_mode=fit_mode, global_scale=global_scale)

        result = await loop.run_in_executor(None, _run)
        buf = _io.BytesIO()
        result.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return web.json_response({"dataUrl": f"data:image/jpeg;base64,{b64}"})
    except Exception as e:
        import traceback
        print(f"[MIL preview] error: {e}\n{traceback.format_exc()}")
        return web.Response(status=500, text=str(e))


# ─── Node ────────────────────────────────────────────────────────────────


def _scale_to_megapixels(img: Image.Image, megapixels: float) -> Image.Image:
    """Scale `img` down if it exceeds `megapixels` * 1 000 000 total pixels.
    Aspect ratio is preserved. Never upscales."""
    if megapixels <= 0:
        return img
    target_px = int(megapixels * 1_000_000)
    w, h = img.size
    if w * h <= target_px:
        return img
    scale = (target_px / (w * h)) ** 0.5
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)


def _compute_canvas_dims(aspect_ratio: str, megapixels: float, first_img_size=None):
    """Return (canvas_w, canvas_h) based on the selected aspect ratio."""
    if aspect_ratio in ("Starred image", "Image Input", "none") or not aspect_ratio:
        return first_img_size
    aw, ah = map(int, aspect_ratio.split(":"))
    if megapixels <= 0 and first_img_size:
        total_px = max(1, first_img_size[0] * first_img_size[1])
    elif megapixels <= 0:
        return None
    else:
        total_px = max(1, int(megapixels * 1_000_000))
    w = max(1, round(math.sqrt(total_px * aw / ah)))
    h = max(1, round(math.sqrt(total_px * ah / aw)))
    return (w, h)


def _compute_individual_ratio_bounds(filtered_filenames, input_dir, megapixels=1, images=None):
    """Compute a shared batch canvas from the max native size across images.
    This preserves each image's native ratio without crop by letterboxing into the shared canvas."""
    sizes = []
    if images is not None:
        sizes.append((max(1, int(images.shape[2])), max(1, int(images.shape[1]))))
    for fname in filtered_filenames:
        fpath = input_dir / fname
        if not fpath.exists():
            continue
        try:
            img = Image.open(fpath)
            img = ImageOps.exif_transpose(img)
            sizes.append(img.size)
        except Exception:
            continue
    if not sizes:
        return 512, 512, 512, 512
    max_w = max(w for w, _ in sizes)
    max_h = max(h for _, h in sizes)
    return max_w, max_h, max_w, max_h


def _parse_bg_color_word(word: str) -> tuple:
    mapping = {"gray": (128, 128, 128), "black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0)}
    return mapping.get(word.strip().lower(), (128, 128, 128))


def _fit_image(img: Image.Image, target_w: int, target_h: int, mode: str, bg_color: tuple = (128, 128, 128), global_scale: float = 1.0) -> Image.Image:
    canvas = Image.new("RGB", (target_w, target_h), bg_color)
    src_w, src_h = img.size
    if mode == "fill":
        resized = img.resize((target_w, target_h), Image.BICUBIC)
        if resized.mode == "RGBA":
            canvas.paste(resized, (0, 0), mask=resized.split()[3])
        else:
            canvas.paste(resized, (0, 0))
    else:
        scale_w = target_w / src_w
        scale_h = target_h / src_h
        if abs(scale_w - scale_h) < 0.005:
            resized = img.resize((target_w, target_h), Image.BICUBIC)
            if resized.mode == "RGBA":
                canvas.paste(resized, (0, 0), mask=resized.split()[3])
            else:
                canvas.paste(resized, (0, 0))
        else:
            scale = min(scale_w, scale_h) if mode == "letterbox" else max(scale_w, scale_h)
            new_w = max(1, round(src_w * scale))
            new_h = max(1, round(src_h * scale))
            resized = img.resize((new_w, new_h), Image.BICUBIC)
            offset_x = (target_w - new_w) // 2
            offset_y = (target_h - new_h) // 2
            if resized.mode == "RGBA":
                canvas.paste(resized, (offset_x, offset_y), mask=resized.split()[3])
            else:
                canvas.paste(resized, (offset_x, offset_y))
    if global_scale != 1.0:
        scaled_w = max(1, round(target_w * global_scale))
        scaled_h = max(1, round(target_h * global_scale))
        shrunk = canvas.resize((scaled_w, scaled_h), Image.BICUBIC)
        final_canvas = Image.new("RGB", (target_w, target_h), bg_color)
        final_canvas.paste(shrunk, ((target_w - scaled_w) // 2, (target_h - scaled_h) // 2))
        return final_canvas
    return canvas


def _apply_crop_region(img: Image.Image, transform: dict) -> Image.Image:
    cx = float(transform.get("cx", 0.0))
    cy = float(transform.get("cy", 0.0))
    cw = float(transform.get("cw", 1.0))
    ch = float(transform.get("ch", 1.0))
    if cx <= 0.0 and cy <= 0.0 and cw >= 1.0 and ch >= 1.0:
        return img
    w, h = img.size
    left = max(0, min(round(cx * w), w - 1))
    top = max(0, min(round(cy * h), h - 1))
    right = max(left + 1, min(round((cx + cw) * w), w))
    bottom = max(top + 1, min(round((cy + ch) * h), h))
    return img.crop((left, top, right, bottom))


def _generate_mask_from_maskops(transform: dict, ref_w: int, ref_h: int) -> Image.Image:
    import base64, io as _io
    ops = transform.get("maskOps", [])
    inverted = bool(transform.get("maskInverted", False))
    if not ops:
        return None
    mask = Image.new("L", (ref_w, ref_h), 0)
    draw = ImageDraw.Draw(mask)
    for op in ops:
        mode = op.get("mode", "add")
        otype = op.get("type", "polygon")
        if otype == "fill" and (op.get("maskFile") or op.get("dataUrl")):
            try:
                fill_img = None
                mask_file = op.get("maskFile")
                if mask_file and isinstance(mask_file, str):
                    fpath = Path(folder_paths.get_input_directory()) / mask_file
                    if fpath.exists():
                        fill_img = Image.open(fpath).convert("L")
                if fill_img is None and op.get("dataUrl"):
                    data_url = op["dataUrl"]
                    _, b64data = data_url.split(",", 1)
                    img_bytes = base64.b64decode(b64data)
                    fill_img = Image.open(_io.BytesIO(img_bytes)).convert("L")
                if fill_img is not None:
                    fill_img = ImageOps.invert(fill_img)
                    fill_img = fill_img.resize((ref_w, ref_h), Image.BICUBIC)
                    mask = fill_img
                    draw = ImageDraw.Draw(mask)
            except Exception:
                pass
            continue
        pts = op.get("pts", [])
        if not pts:
            continue
        fill = 255 if mode == "add" else 0
        if otype == "brush":
            r_norm = float(op.get("r", 0.01))
            r_px = max(1, round(r_norm * ref_w))
            for p in pts:
                px, py = round(p["x"] * ref_w), round(p["y"] * ref_h)
                bbox = [px - r_px, py - r_px, px + r_px, py + r_px]
                draw.ellipse(bbox, fill=fill)
        else:
            if len(pts) < 3:
                continue
            poly_px = [(round(p["x"] * ref_w), round(p["y"] * ref_h)) for p in pts]
            draw.polygon(poly_px, fill=fill)
    if inverted:
        mask = ImageOps.invert(mask)
    return mask


class MultiImageLoader:
    """Loads multiple images into a single batch tensor while preserving available aspect-ratio modes."""

    UPLOADED_FILES_KEY = "image_list"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "images": ("IMAGE",),
                "aspect_ratio_in": ("STRING", {"forceInput": True}),
                "image_list": ("STRING", {"default": "[]"}),
                "fit_mode": (["letterbox", "crop"],),
                "bg_color": (["gray", "black", "white", "red"], {"default": "gray"}),
                "aspect_ratio": (["Starred image", "Image Input", "Aspect Ratio Input", "Individual Ratios", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {"default": "Starred image"}),
                "megapixels": ("INT", {"default": 1, "min": 0, "max": 32, "step": 1, "display": "number"}),
                "thumb_size": (["full", "large", "medium", "small"], {"default": "medium"}),
                "double_click": (["Edit Image", "Edit Pixel", "Mask"], {"default": "Edit Image"}),
                "crop_data": ("STRING", {"default": "{}"}),
                "selected_items": ("STRING", {"default": "[]"}),
                "reference_image": ("STRING", {"default": ""}),
                "grid_columns": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1, "display": "number"}),
                "grid_rows": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1, "display": "number"}),
                "randomize": (["None", "Images", "Selected Images", "Images and Solids", "Selected Images and Solids"], {"default": "None"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "INT", "INT",)
    RETURN_NAMES = ("image_batch", "mask_batch", "image_original", "grid_image", "grid_hires", "native_images", "aspect_ratio_out", "columns_n", "rows_n",)
    FUNCTION = "load_images"
    CATEGORY = "image/loaders"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, images=None, aspect_ratio_in=None, image_list="[]", fit_mode="letterbox", bg_color="gray", aspect_ratio="Starred image", megapixels=1, thumb_size="medium", double_click="Edit Image", crop_data="{}", selected_items="[]", reference_image="", grid_columns=3, grid_rows=3, randomize="None"):
        if randomize != "None":
            return float("NaN")
        master_hash = f"{images.shape}" if images is not None else ""
        ar_in_hash = aspect_ratio_in or ""
        return hashlib.md5((image_list + crop_data + selected_items + reference_image + aspect_ratio + ar_in_hash + fit_mode + bg_color + str(megapixels) + str(grid_columns) + str(grid_rows) + master_hash).encode()).hexdigest()

    def load_images(self, images=None, aspect_ratio_in=None, image_list="[]", fit_mode="letterbox", bg_color="gray", aspect_ratio="Starred image", megapixels=1, thumb_size="medium", double_click="Edit Image", crop_data="{}", selected_items="[]", reference_image="", grid_columns=3, grid_rows=3, randomize="None"):
        if aspect_ratio == "Aspect Ratio Input":
            if aspect_ratio_in and isinstance(aspect_ratio_in, str) and ":" in aspect_ratio_in:
                aspect_ratio = aspect_ratio_in.strip()
            else:
                aspect_ratio = "Starred image"
        resolved_ar = aspect_ratio

        try:
            filenames = json.loads(image_list)
        except Exception:
            filenames = []

        try:
            transforms = json.loads(crop_data) if crop_data else {}
        except Exception:
            transforms = {}

        try:
            selected_filenames = json.loads(selected_items) if selected_items else []
            if not isinstance(selected_filenames, list):
                selected_filenames = []
        except Exception:
            selected_filenames = []

        filtered_filenames = []
        if selected_filenames:
            for fname in filenames:
                if fname in selected_filenames:
                    filtered_filenames.append(fname)
        else:
            filtered_filenames = filenames

        placeholder_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        placeholder_mask = torch.ones((1, 64, 64), dtype=torch.float32)
        if not filtered_filenames and images is None:
            return (placeholder_img, placeholder_mask, placeholder_img, placeholder_img, placeholder_img, [placeholder_img], resolved_ar, grid_columns, grid_rows)

        input_dir = Path(folder_paths.get_input_directory())
        tensors = []
        mask_tensors = []
        orig_tensors = []
        native_image_list = []
        orig_ref_w = orig_ref_h = None

        master_canvas = None
        first_valid_img = None
        badge_resolved = False
        individual_ratio_mode = (aspect_ratio == "Individual Ratios")

        if reference_image:
            fpath = input_dir / reference_image
            if fpath.exists():
                try:
                    img_tmp = Image.open(fpath)
                    img_tmp = ImageOps.exif_transpose(img_tmp)
                    first_valid_img = img_tmp.copy()
                    badge_resolved = True
                    orig_ref_w, orig_ref_h = img_tmp.size
                    t_ref = transforms.get(reference_image)
                    if t_ref:
                        img_tmp_cropped = _apply_crop_region(img_tmp, t_ref)
                        orig_ref_w, orig_ref_h = img_tmp_cropped.size
                    tmp_scaled = _scale_to_megapixels(img_tmp, megapixels)
                    if t_ref:
                        tmp_scaled = _apply_crop_region(tmp_scaled, t_ref)
                    ref_w, ref_h = tmp_scaled.size
                    fixed_canvas = True
                except Exception:
                    pass

        if not badge_resolved and images is not None:
            m_h = images.shape[1]
            m_w = images.shape[2]
            orig_ref_w = m_w
            orig_ref_h = m_h
            if aspect_ratio == "Image Input":
                ref_w, ref_h = m_w, m_h
                orig_ref_w, orig_ref_h = m_w, m_h
                fixed_canvas = True
            elif individual_ratio_mode:
                ref_w, ref_h, orig_ref_w, orig_ref_h = _compute_individual_ratio_bounds(filtered_filenames, input_dir, megapixels, images)
                fixed_canvas = True
            else:
                if megapixels <= 0:
                    c_w, c_h = m_w, m_h
                else:
                    total_px = max(1, megapixels * 1_000_000)
                    ratio = m_w / m_h
                    c_w = max(1, round(math.sqrt(total_px * ratio)))
                    c_h = max(1, round(math.sqrt(total_px / ratio)))
                master_canvas = (c_w, c_h)
                ref_w, ref_h = master_canvas
                fixed_canvas = True

        elif not badge_resolved and aspect_ratio not in ("Starred image", "Image Input", "Individual Ratios", "none") and bool(aspect_ratio):
            fixed_canvas = True
            dims = _compute_canvas_dims(aspect_ratio, megapixels)
            if dims is not None:
                ref_w, ref_h = dims
            else:
                ref_w = ref_h = None
                fixed_canvas = True
        elif not badge_resolved and aspect_ratio == "Individual Ratios":
            fixed_canvas = True
            ref_w, ref_h, orig_ref_w, orig_ref_h = _compute_individual_ratio_bounds(filtered_filenames, input_dir, megapixels, images)
        elif not badge_resolved:
            fixed_canvas = False
            ref_w = ref_h = None

        first_valid_fname = ""
        if first_valid_img is None:
            for fname in filenames:
                fpath = input_dir / fname
                if fpath.exists():
                    try:
                        img_tmp = Image.open(fpath)
                        img_tmp = ImageOps.exif_transpose(img_tmp)
                        first_valid_img = img_tmp.copy()
                        first_valid_fname = fname
                        break
                    except Exception:
                        pass

        if first_valid_img is not None and not badge_resolved:
            first_native_w, first_native_h = first_valid_img.size
            if orig_ref_w is None:
                if master_canvas:
                    native_mp = (first_native_w * first_native_h) / 1_000_000
                    ratio = master_canvas[0] / master_canvas[1]
                    orig_ref_w = max(1, round(math.sqrt(native_mp * 1_000_000 * ratio)))
                    orig_ref_h = max(1, round(math.sqrt(native_mp * 1_000_000 / ratio)))
                elif fixed_canvas:
                    native_mp = (first_native_w * first_native_h) / 1_000_000
                    dims = _compute_canvas_dims(aspect_ratio, native_mp, (first_native_w, first_native_h))
                    if dims is not None:
                        orig_ref_w, orig_ref_h = dims
                    else:
                        orig_ref_w, orig_ref_h = first_native_w, first_native_h
                else:
                    t_fallback = transforms.get(first_valid_fname)
                    if t_fallback:
                        img_tmp_cropped = _apply_crop_region(first_valid_img, t_fallback)
                        orig_ref_w, orig_ref_h = img_tmp_cropped.size
                    else:
                        orig_ref_w = first_native_w
                        orig_ref_h = first_native_h

            if ref_w is None:
                if aspect_ratio not in ("Starred image", "Image Input", "Individual Ratios", "none") and bool(aspect_ratio):
                    dims = _compute_canvas_dims(aspect_ratio, megapixels, (first_native_w, first_native_h))
                    if dims is not None:
                        ref_w, ref_h = dims
                if ref_w is None:
                    tmp_scaled = _scale_to_megapixels(first_valid_img, megapixels)
                    t_fallback = transforms.get(first_valid_fname) if not badge_resolved else None
                    if t_fallback:
                        tmp_scaled = _apply_crop_region(tmp_scaled, t_fallback)
                    ref_w, ref_h = tmp_scaled.size

        if orig_ref_w is None:
            orig_ref_w, orig_ref_h = 512, 512
        if ref_w is None:
            if megapixels <= 0:
                if first_valid_img is not None:
                    ref_w, ref_h = first_valid_img.size
                else:
                    ref_w, ref_h = 512, 512
            else:
                tmp_scaled = _scale_to_megapixels(Image.new("RGB", (512, 512)), megapixels)
                ref_w, ref_h = tmp_scaled.size

        if individual_ratio_mode:
            ref_w, ref_h, orig_ref_w, orig_ref_h = _compute_individual_ratio_bounds(filtered_filenames, input_dir, megapixels, images)

        for fname in filtered_filenames:
            if fname.startswith("__BLANK__"):
                _bg_rgb = _parse_bg_color_word(bg_color)
                img_orig = Image.new("RGB", (orig_ref_w, orig_ref_h), _bg_rgb)
                orig_arr = np.array(img_orig).astype(np.float32) / 255.0
                orig_tensors.append(torch.from_numpy(orig_arr).unsqueeze(0))
                native_image_list.append(torch.from_numpy(orig_arr).unsqueeze(0))
                img_scaled = Image.new("RGB", (ref_w, ref_h), _bg_rgb)
                arr = np.array(img_scaled).astype(np.float32) / 255.0
                tensors.append(torch.from_numpy(arr).unsqueeze(0))
                mask_tensors.append(torch.ones((1, ref_h, ref_w), dtype=torch.float32))
                continue

            fpath = input_dir / fname
            if not fpath.exists():
                continue
            try:
                img = Image.open(fpath)
                img = ImageOps.exif_transpose(img)
                has_alpha = (img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info))
                if has_alpha:
                    img = img.convert("RGBA")
                else:
                    img = img.convert("RGB")

                img_orig = img.copy()
                if img_orig.mode == "RGBA":
                    img_orig_rgb = Image.new("RGB", img_orig.size, _parse_bg_color_word(bg_color))
                    img_orig_rgb.paste(img_orig, (0, 0), mask=img_orig.split()[3])
                    img_orig = img_orig_rgb
                else:
                    img_orig = img_orig.convert("RGB")

                native_arr = np.array(img_orig).astype(np.float32) / 255.0
                native_image_list.append(torch.from_numpy(native_arr).unsqueeze(0))

                t_orig = transforms.get(fname)
                effective_fit_mode = "letterbox" if individual_ratio_mode else fit_mode
                if t_orig:
                    img_orig = _apply_crop_region(img_orig, t_orig)
                    if img_orig.size != (orig_ref_w, orig_ref_h):
                        img_orig = _fit_image(img_orig, orig_ref_w, orig_ref_h, effective_fit_mode, bg_color=_parse_bg_color_word(bg_color), global_scale=1.0)
                elif img_orig.size != (orig_ref_w, orig_ref_h):
                    img_orig = _fit_image(img_orig, orig_ref_w, orig_ref_h, effective_fit_mode, bg_color=_parse_bg_color_word(bg_color), global_scale=1.0)

                if img_orig.mode == "RGBA":
                    canvas_orig = Image.new("RGB", img_orig.size, _parse_bg_color_word(bg_color))
                    canvas_orig.paste(img_orig, (0, 0), mask=img_orig.split()[3])
                    img_orig = canvas_orig
                else:
                    img_orig = img_orig.convert("RGB")
                orig_arr = np.array(img_orig).astype(np.float32) / 255.0
                orig_tensors.append(torch.from_numpy(orig_arr).unsqueeze(0))

                img = _scale_to_megapixels(img, megapixels)
                t = transforms.get(fname)
                _bg_rgb = _parse_bg_color_word(bg_color)
                if t:
                    img = _apply_crop_region(img, t)
                    if img.size != (ref_w, ref_h):
                        img = _fit_image(img, ref_w, ref_h, effective_fit_mode, bg_color=_bg_rgb, global_scale=1.0)
                elif img.size != (ref_w, ref_h):
                    img = _fit_image(img, ref_w, ref_h, effective_fit_mode, bg_color=_bg_rgb, global_scale=1.0)
                if img.mode == "RGBA":
                    canvas_scaled = Image.new("RGB", img.size, _bg_rgb)
                    canvas_scaled.paste(img, (0, 0), mask=img.split()[3])
                    img = canvas_scaled
                else:
                    img = img.convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                tensors.append(torch.from_numpy(arr).unsqueeze(0))

                if t and t.get("maskOps"):
                    mask_img = _generate_mask_from_maskops(t, ref_w, ref_h)
                    if mask_img is not None:
                        mask_arr = np.array(mask_img).astype(np.float32) / 255.0
                        mask_tensors.append(torch.from_numpy(mask_arr).unsqueeze(0))
                    else:
                        mask_tensors.append(torch.ones((1, ref_h, ref_w), dtype=torch.float32))
                else:
                    mask_tensors.append(torch.ones((1, ref_h, ref_w), dtype=torch.float32))
            except Exception:
                continue

        if individual_ratio_mode:
            resolved_ar = "Individual Ratios"
        elif aspect_ratio in ("Starred image", "Image Input", "Aspect Ratio Input", "none") or not aspect_ratio:
            if ref_w and ref_h:
                actual_ratio = ref_w / ref_h
                common_ratios = [(1, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4), (9, 16), (16, 9), (21, 9)]
                best_ratio = None
                min_diff = float("inf")
                for rw, rh in common_ratios:
                    diff = abs(actual_ratio - (rw / rh))
                    if diff < min_diff:
                        min_diff = diff
                        best_ratio = f"{rw}:{rh}"
                resolved_ar = best_ratio or "auto"
            else:
                resolved_ar = "auto"

        if not tensors:
            return (placeholder_img, placeholder_mask, placeholder_img, placeholder_img, placeholder_img, [placeholder_img], resolved_ar, grid_columns, grid_rows)

        batch = torch.cat(tensors, dim=0)
        mask_batch = torch.cat(mask_tensors, dim=0)
        orig_batch = torch.cat(orig_tensors, dim=0) if orig_tensors else placeholder_img
        grid_image = self._make_grid(batch, grid_rows, grid_columns, randomize, image_list, selected_items)
        grid_hires = self._make_grid(orig_batch, grid_rows, grid_columns, randomize, image_list, selected_items)
        return (batch, mask_batch, orig_batch, grid_image, grid_hires, native_image_list, resolved_ar, grid_columns, grid_rows)

    def _make_grid(self, batch, rows, cols, randomize="None", image_list="[]", selected_items="[]"):
        B, H, W, C = batch.shape
        max_images = rows * cols
        grid_items = batch[:max_images]
        actual_b = grid_items.shape[0]
        if randomize != "None" and actual_b > 1:
            try:
                filenames = json.loads(image_list)[:max_images]
            except Exception:
                filenames = []
            try:
                selected = set(json.loads(selected_items)) if selected_items else set()
            except Exception:
                selected = set()
            while len(filenames) < actual_b:
                filenames.append("")
            filenames = filenames[:actual_b]
            is_blank = [f.startswith("__BLANK__") for f in filenames]
            is_selected = [f in selected for f in filenames]
            has_selection = bool(selected)
            if randomize == "Images":
                real_pos = [i for i, b in enumerate(is_blank) if not b]
                shuffled = real_pos[:]
                np.random.shuffle(shuffled)
                result = grid_items.clone()
                for dest, src in zip(real_pos, shuffled):
                    result[dest] = grid_items[src]
                grid_items = result
            elif randomize == "Selected Images":
                if has_selection:
                    sel_real_pos = [i for i, (b, s) in enumerate(zip(is_blank, is_selected)) if not b and s]
                else:
                    sel_real_pos = [i for i, b in enumerate(is_blank) if not b]
                shuffled = sel_real_pos[:]
                np.random.shuffle(shuffled)
                result = grid_items.clone()
                for dest, src in zip(sel_real_pos, shuffled):
                    result[dest] = grid_items[src]
                grid_items = result
            elif randomize == "Images and Solids":
                indices = list(range(actual_b))
                np.random.shuffle(indices)
                grid_items = grid_items[indices]
            elif randomize == "Selected Images and Solids":
                if has_selection:
                    sel_pos = [i for i, s in enumerate(is_selected) if s]
                else:
                    sel_pos = list(range(actual_b))
                shuffled = sel_pos[:]
                np.random.shuffle(shuffled)
                result = grid_items.clone()
                for dest, src in zip(sel_pos, shuffled):
                    result[dest] = grid_items[src]
                grid_items = result
        if actual_b < max_images:
            padding = torch.zeros((max_images - actual_b, H, W, C), dtype=batch.dtype, device=batch.device)
            grid_items = torch.cat([grid_items, padding], dim=0)
        grid_items = grid_items.view(rows, cols, H, W, C)
        grid_items = grid_items.permute(0, 2, 1, 3, 4)
        grid = grid_items.reshape(1, rows * H, cols * W, C)
        return grid


class LoadImagesInGrid(MultiImageLoader):
    @classmethod
    def INPUT_TYPES(cls):
        return MultiImageLoader.INPUT_TYPES()

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("grid_image", "grid_hires",)
    FUNCTION = "compose_grid"
    CATEGORY = "image/loaders"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return MultiImageLoader.IS_CHANGED(*args, **kwargs)

    def compose_grid(self, *args, **kwargs):
        res = self.load_images(*args, **kwargs)
        return (res[3], res[4])


NODE_CLASS_MAPPINGS = {"MultiImageLoader": MultiImageLoader, "LoadImagesInGrid": LoadImagesInGrid}
NODE_DISPLAY_NAME_MAPPINGS = {"MultiImageLoader": "Load Multiple Images 🖼️", "LoadImagesInGrid": "Load Images in Grid (Deprecated) 🖼️"}
