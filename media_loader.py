"""
Multi Media Loader (Neo)
========================

Same behaviour as the multi-image loader, but the media list can also contain
videos and animated GIF/WebP files.

Design note (why this is small):
    Videos and animations are expanded into cached PNG frames on disk, then the
    exact same, already battle-tested pipeline in ``nodes.MultiImageLoader`` is
    invoked. This avoids re-implementing fit modes, aspect-ratio math, masking,
    megapixel budgeting, grids and the native list outputs.
"""

import os
import io
import json
import shutil
import hashlib
import asyncio
import subprocess

import numpy as np
from PIL import Image, ImageOps
from pathlib import Path

from aiohttp import web
from server import PromptServer
import folder_paths

from .nodes import MultiImageLoader


# ─── media type tables ───────────────────────────────────────────────────────

VIDEO_EXTS = {
    ".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v",
    ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".mts", ".3gp",
}
ANIMATED_EXTS = {".gif", ".webp", ".apng"}
IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".apng",
}

CACHE_SUBFOLDER = "mil_frames"


# ─── path helpers ────────────────────────────────────────────────────────────

def _input_dir() -> Path:
    return Path(folder_paths.get_input_directory())


def resolve_media_path(filename: str, custom_path: str = ""):
    """Resolve ``filename`` against the optional custom folder, then input dir."""
    if not filename:
        return None
    rel = Path(filename.replace("\\", "/"))
    if rel.is_absolute() and rel.is_file():
        return rel

    candidates = []
    custom = (custom_path or "").strip()
    if custom:
        base = Path(os.path.expanduser(custom))
        if not base.is_absolute():
            base = _input_dir() / base
        candidates.append(base / rel)
    candidates.append(_input_dir() / rel)

    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _cache_key(path: Path, tag: str) -> str:
    try:
        st = path.stat()
        raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{tag}"
    except OSError:
        raw = f"{path}|{tag}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _ensure_local_copy(path: Path) -> str:
    """Return a path relative to ComfyUI's input dir for ``path``.

    Files already inside the input dir are returned as-is. Files living in a
    custom folder are copied once into ``input/mil_frames/media/`` so the shared
    loader (which only reads from the input dir) can pick them up.
    """
    input_dir = _input_dir()
    try:
        return path.resolve().relative_to(input_dir.resolve()).as_posix()
    except Exception:
        pass

    key = _cache_key(path, "copy")
    dest_dir = input_dir / CACHE_SUBFOLDER / "media"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{key}_{path.name}"
    if not dest.exists():
        shutil.copy2(path, dest)
    return f"{CACHE_SUBFOLDER}/media/{dest.name}"


# ─── frame extraction ────────────────────────────────────────────────────────

def _ffmpeg_bin():
    """Locate an ffmpeg executable, if any."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # optional dependency
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _ppm_header_token(data: bytes, pos: int, end: int):
    """Read one PPM header token (skipping whitespace and # comments)."""
    while pos < end and data[pos:pos + 1].isspace():
        pos += 1
    if pos < end and data[pos:pos + 1] == b"#":
        while pos < end and data[pos:pos + 1] not in (b"\n", b"\r"):
            pos += 1
        return _ppm_header_token(data, pos, end)
    start = pos
    while pos < end and not data[pos:pos + 1].isspace():
        pos += 1
    return int(data[start:pos]), pos


def _parse_ppm_stream(data: bytes):
    """Decode a stream of concatenated binary PPM (P6) images."""
    frames = []
    i, n = 0, len(data)
    while i < n:
        if data[i:i + 2] != b"P6":
            nxt = data.find(b"P6", i + 1)
            if nxt < 0:
                break
            i = nxt
            continue
        try:
            width, pos = _ppm_header_token(data, i + 2, n)
            height, pos = _ppm_header_token(data, pos, n)
            _maxval, pos = _ppm_header_token(data, pos, n)
        except Exception:
            break
        pos += 1  # single whitespace byte after maxval
        size = width * height * 3
        if width < 1 or height < 1 or pos + size > n:
            break
        arr = np.frombuffer(data[pos:pos + size], dtype=np.uint8).reshape(height, width, 3)
        frames.append(Image.fromarray(arr, "RGB"))
        i = pos + size
    return frames


def extract_video_frames(path: Path, fps: float, max_frames: int):
    """Decode up to ``max_frames`` frames sampled at ``fps`` using ffmpeg."""
    exe = _ffmpeg_bin()
    if not exe:
        print("[MML] ffmpeg not found - cannot decode video. "
              "Install ffmpeg or `pip install imageio-ffmpeg`.")
        return []

    fps = max(0.1, float(fps))
    max_frames = max(1, int(max_frames))
    cmd = [
        exe, "-v", "error", "-nostdin",
        "-i", str(path),
        "-vf", f"fps={fps}",
        "-frames:v", str(max_frames),
        "-f", "image2pipe", "-vcodec", "ppm",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True)
    except Exception as e:
        print(f"[MML] ffmpeg failed to start: {e}")
        return []

    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode(errors="ignore").strip()
        print(f"[MML] ffmpeg decode failed for {path.name}: {err[:300]}")
        return []

    return _parse_ppm_stream(proc.stdout)


def extract_animated_frames(path: Path, max_frames: int):
    """Read frames from an animated GIF/WebP/APNG via Pillow."""
    frames = []
    max_frames = max(1, int(max_frames))
    try:
        with Image.open(path) as im:
            total = getattr(im, "n_frames", 1)
            for idx in range(min(total, max_frames)):
                im.seek(idx)
                frames.append(im.convert("RGBA"))
    except Exception as e:
        print(f"[MML] failed to read animation {path.name}: {e}")
        return []
    return frames


def _frame_cache_dir(path: Path, tag: str) -> Path:
    return _input_dir() / CACHE_SUBFOLDER / _cache_key(path, tag)


def _write_frames(frames, cache_dir: Path, rel_prefix: str):
    cache_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for index, frame in enumerate(frames, 1):
        name = f"frame_{index:04d}.png"
        frame.convert("RGB").save(cache_dir / name)
        names.append(f"{rel_prefix}/{name}")
    return names


def expand_media(filename: str, custom_path: str = "", video_fps: float = 8.0,
                 video_max_frames: int = 24, gif_max_frames: int = 24):
    """Expand one media entry into one or more input-dir relative image paths."""
    path = resolve_media_path(filename, custom_path)
    if path is None:
        print(f"[MML] media not found: {filename}")
        return []

    ext = path.suffix.lower()

    if ext in VIDEO_EXTS:
        key = _cache_key(path, f"v{video_fps}_{video_max_frames}")
        cache_dir = _input_dir() / CACHE_SUBFOLDER / key
        existing = sorted(cache_dir.glob("frame_*.png")) if cache_dir.is_dir() else []
        if existing:
            return [f"{CACHE_SUBFOLDER}/{key}/{p.name}" for p in existing]
        frames = extract_video_frames(path, video_fps, video_max_frames)
        if not frames:
            return []
        print(f"[MML] decoded {len(frames)} frame(s) from {path.name} @ {video_fps}fps")
        return _write_frames(frames, cache_dir, f"{CACHE_SUBFOLDER}/{key}")

    if ext in ANIMATED_EXTS:
        try:
            with Image.open(path) as im:
                is_animated = getattr(im, "n_frames", 1) > 1
        except Exception:
            is_animated = False
        if is_animated:
            key = _cache_key(path, f"a{gif_max_frames}")
            cache_dir = _input_dir() / CACHE_SUBFOLDER / key
            existing = sorted(cache_dir.glob("frame_*.png")) if cache_dir.is_dir() else []
            if existing:
                return [f"{CACHE_SUBFOLDER}/{key}/{p.name}" for p in existing]
            frames = extract_animated_frames(path, gif_max_frames)
            if not frames:
                return []
            print(f"[MML] extracted {len(frames)} frame(s) from {path.name}")
            return _write_frames(frames, cache_dir, f"{CACHE_SUBFOLDER}/{key}")

    return [_ensure_local_copy(path)]


def media_poster_frame(path: Path):
    """Return a single representative frame for thumbnails/previews."""
    ext = path.suffix.lower()
    if ext in VIDEO_EXTS:
        frames = extract_video_frames(path, 1.0, 1)
        return frames[0] if frames else None
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if getattr(im, "n_frames", 1) > 1:
                im.seek(0)
            return im.copy()
    except Exception as e:
        print(f"[MML] failed to open {path.name}: {e}")
        return None


# ─── thumbnail route ─────────────────────────────────────────────────────────

@PromptServer.instance.routes.get("/multi_image_loader/media_thumb")
async def media_thumb_handler(request):
    """Serve a poster/thumbnail frame for any supported media file.

    Used by the frontend for video thumbnails and for files that live outside
    ComfyUI's input folder (custom path), which ``/view`` cannot serve.
    """
    filename = request.query.get("filename", "")
    custom_path = request.query.get("custom_path", "")
    try:
        max_px = int(request.query.get("max", "512"))
    except ValueError:
        max_px = 512
    max_px = max(32, min(4096, max_px))

    if not filename:
        return web.Response(status=400, text="Missing filename")

    loop = asyncio.get_event_loop()

    def _run():
        path = resolve_media_path(filename, custom_path)
        if path is None:
            return None
        img = media_poster_frame(path)
        if img is None:
            return None

        if max(img.size) > max_px:
            scale = max_px / max(img.size)
            img = img.resize(
                (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                Image.LANCZOS,
            )

        buf = io.BytesIO()
        if img.mode in ("RGBA", "LA") and path.suffix.lower() not in VIDEO_EXTS:
            img.convert("RGBA").save(buf, format="PNG")
            return buf.getvalue(), "image/png"
        img.convert("RGB").save(buf, format="JPEG", quality=88)
        return buf.getvalue(), "image/jpeg"

    try:
        result = await loop.run_in_executor(None, _run)
    except Exception as e:
        print(f"[MML media_thumb] error: {e}")
        return web.Response(status=500, text=str(e))

    if result is None:
        return web.Response(status=404, text=f"Not found: {filename}")

    body, content_type = result
    return web.Response(body=body, content_type=content_type,
                        headers={"Cache-Control": "no-cache"})


# ─── node ────────────────────────────────────────────────────────────────────

class MultiMediaLoader(MultiImageLoader):
    """Neo multi-media loader: images, GIF/WebP animations and videos.

    Everything else (fit modes, aspect ratios, megapixels, crop/edit data,
    masks, grids, native per-image outputs) is inherited unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = MultiImageLoader.INPUT_TYPES()
        optional = inputs["optional"]
        optional["custom_path"] = ("STRING", {
            "default": "",
            "multiline": False,
            "tooltip": "Optional folder (absolute, or relative to ComfyUI/input) "
                       "used for uploads and for resolving media filenames.",
        })
        optional["video_fps"] = ("FLOAT", {
            "default": 8.0, "min": 0.1, "max": 60.0, "step": 0.1, "display": "number",
            "tooltip": "Frames per second sampled from a video.",
        })
        optional["video_max_frames"] = ("INT", {
            "default": 24, "min": 1, "max": 512, "step": 1, "display": "number",
            "tooltip": "Maximum number of frames decoded per video.",
        })
        optional["gif_max_frames"] = ("INT", {
            "default": 24, "min": 1, "max": 512, "step": 1, "display": "number",
            "tooltip": "Maximum number of frames decoded per animated image.",
        })
        return inputs

    @classmethod
    def IS_CHANGED(cls, images=None, aspect_ratio_in=None, image_list="[]",
                   fit_mode="letterbox", bg_color="gray", aspect_ratio="Starred image",
                   megapixels=1, thumb_size="medium", double_click="Edit Image",
                   crop_data="{}", selected_items="[]", reference_image="",
                   grid_columns=3, grid_rows=3, randomize="None",
                   custom_path="", video_fps=8.0, video_max_frames=24, gif_max_frames=24):
        base = MultiImageLoader.IS_CHANGED(
            images, aspect_ratio_in, image_list, fit_mode, bg_color, aspect_ratio,
            megapixels, thumb_size, double_click, crop_data, selected_items,
            reference_image, grid_columns, grid_rows, randomize,
        )
        extra = f"{custom_path}|{video_fps}|{video_max_frames}|{gif_max_frames}"
        return hashlib.md5(f"{base}|{extra}".encode()).hexdigest()

    def load_images(self, images=None, aspect_ratio_in=None, image_list="[]",
                    fit_mode="letterbox", bg_color="gray", aspect_ratio="Starred image",
                    megapixels=1, thumb_size="medium", double_click="Edit Image",
                    crop_data="{}", selected_items="[]", reference_image="",
                    grid_columns=3, grid_rows=3, randomize="None",
                    custom_path="", video_fps=8.0, video_max_frames=24, gif_max_frames=24):
        try:
            sources = json.loads(image_list) if image_list else []
            if not isinstance(sources, list):
                sources = []
        except Exception:
            sources = []

        try:
            selected = json.loads(selected_items) if selected_items else []
            if not isinstance(selected, list):
                selected = []
        except Exception:
            selected = []

        try:
            transforms = json.loads(crop_data) if crop_data else {}
            if not isinstance(transforms, dict):
                transforms = {}
        except Exception:
            transforms = {}

        expanded = []        # input-dir relative names handed to the base loader
        source_of = {}       # expanded name -> original media name
        reference_frame = reference_image

        for source_name in sources:
            if source_name.startswith("__BLANK__"):
                expanded.append(source_name)
                source_of[source_name] = source_name
                continue

            frame_names = expand_media(
                source_name, custom_path, video_fps, video_max_frames, gif_max_frames
            )
            if not frame_names:
                continue

            for frame_name in frame_names:
                expanded.append(frame_name)
                source_of[frame_name] = source_name

            if reference_image and reference_image == source_name:
                reference_frame = frame_names[0]

        # The UI selection refers to source media; apply it to the expanded frames.
        if selected:
            selected_set = set(selected)
            filtered = [fn for fn in expanded if source_of.get(fn, fn) in selected_set]
            if filtered:
                expanded = filtered

        # Crop/edit transforms are stored per source media - copy them to frames.
        frame_transforms = {}
        for frame_name in expanded:
            source_name = source_of.get(frame_name, frame_name)
            transform = transforms.get(frame_name) or transforms.get(source_name)
            if transform:
                frame_transforms[frame_name] = transform

        return MultiImageLoader.load_images(
            self,
            images=images,
            aspect_ratio_in=aspect_ratio_in,
            image_list=json.dumps(expanded),
            fit_mode=fit_mode,
            bg_color=bg_color,
            aspect_ratio=aspect_ratio,
            megapixels=megapixels,
            thumb_size=thumb_size,
            double_click=double_click,
            crop_data=json.dumps(frame_transforms),
            selected_items="[]",  # selection already resolved above
            reference_image=reference_frame or "",
            grid_columns=grid_columns,
            grid_rows=grid_rows,
            randomize=randomize,
        )


# ─── registrations ───────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "NeoMultiMediaLoader": MultiMediaLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NeoMultiMediaLoader": "neo - Multi Media Loader 🎬",
}
