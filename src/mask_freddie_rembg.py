"""Test rembg as the foreground-extraction tool for the captured Freddie scene.

Mirrors mask_freddie.py but uses rembg's U2Net / ISNet cutout model.
Writes to ../freddie_masked_rembg/ so the result sits next to the SAM 2
output for side-by-side inspection.

For each photo in ../freddie/:
  1. Run rembg.remove() to get an RGBA cutout (alpha channel = foreground mask).
  2. Composite the cutout onto a uniform white background.
  3. Write to ../freddie_masked_rembg/<same_filename>.

Idempotent: skips photos already present in the output directory.
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys
from pathlib import Path


def _preload_cuda12_libs() -> None:
    """Preload CUDA 12 runtime libs from the venv site-packages.

    onnxruntime-gpu 1.26 on PyPI is a CUDA 12 build whose binaries dlopen
    libraries like libcufft.so.11 (CUDA 12's cuFFT). The host system here ships
    CUDA 13 (libcufft.so.12), so the default loader can't satisfy onnxruntime's
    cu12 dependencies. The `nvidia-*-cu12` pip packages bundle the matching
    libs into `site-packages/nvidia/<lib>/lib/`; loading them globally with
    RTLD_GLOBAL puts their symbols on the process's dlopen search path before
    onnxruntime imports. Setting LD_LIBRARY_PATH after the interpreter has
    started would not work - glibc caches it at process start.
    """
    site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia"
    # Order matters: load leaf deps first (cudart, nvJitLink) before the
    # higher-level libs (cublas, cudnn) that depend on them.
    sonames = [
        "libcudart.so.12",
        "libnvJitLink.so.12",
        "libnvrtc.so.12",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcufft.so.11",
        "libcurand.so.10",
        "libcusparse.so.12",
        "libcudnn.so.9",
    ]
    loaded = []
    missing = []
    for soname in sonames:
        hits = glob.glob(str(site / "*" / "lib" / soname))
        if not hits:
            missing.append(soname)
            continue
        try:
            ctypes.CDLL(hits[0], mode=ctypes.RTLD_GLOBAL)
            loaded.append(soname)
        except OSError as e:
            missing.append(f"{soname} ({e})")
    print(f"Preloaded CUDA 12 libs: {len(loaded)} ok"
          + (f", {len(missing)} missing: {missing}" if missing else ""), flush=True)


_preload_cuda12_libs()


import numpy as np
from PIL import Image, ImageOps
from rembg import new_session, remove


# Inverse table for ImageOps.exif_transpose(). Used to put the masked output
# back in the original raw-JPEG pixel orientation: COLMAP's cameras.txt was
# derived from photos whose EXIF was NOT applied (W=4032, H=3024 for these
# 4:3 phone shots), and the notebook loader reads via Image.open() + np.asarray
# which also does not apply EXIF. Saving the masked output in EXIF-applied
# (portrait) orientation would mismatch both.
EXIF_INVERSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_90,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_270,
}


def undo_exif_transpose(pil_img: Image.Image, original_orientation: int) -> Image.Image:
    if original_orientation in (None, 1):
        return pil_img
    method = EXIF_INVERSE.get(original_orientation)
    return pil_img if method is None else pil_img.transpose(method)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "freddie"
DST_DIR = PROJECT_ROOT / "freddie_masked_rembg"

# rembg ships multiple foreground models. isnet-general-use is the newer
# successor to u2net and typically gives cleaner edges on cluttered scenes.
# Swap to "u2net" or "birefnet-general" to compare.
MODEL_NAME = "isnet-general-use"

BACKGROUND_RGB = (255, 255, 255)  # white background after masking

# Sanity bounds on the foreground area fraction. Outside these bounds the mask
# is almost certainly wrong (collapsed to nothing or selected the whole frame).
AREA_FRAC_MIN = 0.02
AREA_FRAC_MAX = 0.70


def main() -> int:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    photos = sorted(p for p in SRC_DIR.iterdir() if p.suffix.upper() in {".JPG", ".JPEG", ".PNG"})
    if not photos:
        print(f"No photos found under {SRC_DIR}.")
        return 1

    print(f"Loading rembg session: {MODEL_NAME}", flush=True)
    # Force GPU; fall back to CPU only if CUDA provider isn't actually working.
    session = new_session(
        model_name=MODEL_NAME,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    print(f"ONNX providers in use: {session.inner_session.get_providers()}", flush=True)

    print(f"Masking {len(photos)} photo(s) from {SRC_DIR} -> {DST_DIR}\n", flush=True)
    print(f"{'#':>3}  {'filename':<20}  {'area%':>6}  status", flush=True)
    print("-" * 50, flush=True)

    area_fracs: list[float] = []
    flagged: list[str] = []
    skipped = 0

    for i, src in enumerate(photos):
        dst = DST_DIR / src.name
        if dst.exists():
            skipped += 1
            print(f"{i:>3}  {src.name:<20}  {'-':>6}  skip (already exists)", flush=True)
            continue

        # Apply the JPEG EXIF orientation tag before rembg sees the image.
        # iPhone photos store the sensor pixels in landscape orientation and
        # rely on the EXIF flag to indicate the intended portrait view; rembg
        # applies that flag internally, so without exif_transpose() here the
        # returned mask is transposed relative to np.array(img).
        img = ImageOps.exif_transpose(Image.open(src).convert("RGB"))
        rgba = remove(img, session=session)
        alpha = np.array(rgba.split()[-1])
        mask = alpha > 127

        H, W = mask.shape
        area_frac = mask.sum() / float(H * W)
        area_fracs.append(area_frac)

        flag = ""
        if not (AREA_FRAC_MIN <= area_frac <= AREA_FRAC_MAX):
            flag = " !! out-of-bounds"
            flagged.append(src.name)

        img_arr = np.array(img)
        out = np.empty_like(img_arr)
        out[...] = BACKGROUND_RGB
        out[mask] = img_arr[mask]
        Image.fromarray(out).save(dst, quality=95)

        print(f"{i:>3}  {src.name:<20}  {area_frac*100:>5.1f}%  ok{flag}", flush=True)

    print("-" * 50, flush=True)
    if area_fracs:
        af = np.array(area_fracs)
        print(f"Masked {len(area_fracs)} photo(s); skipped {skipped} pre-existing.")
        print(f"Area fraction: mean {af.mean()*100:5.1f}% | min {af.min()*100:5.1f}% | max {af.max()*100:5.1f}%")
    else:
        print(f"No new masking performed (all {skipped} outputs already existed).")
    if flagged:
        print(f"\nFlagged for visual inspection ({len(flagged)}): {flagged}")
    print(f"\nMasked photos: {DST_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
