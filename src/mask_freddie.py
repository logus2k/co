"""Mask the Freddie figurine out of the 71 captured photos using SAM 2.

For each photo in ../freddie/:
  1. Run SAM 2 with a single center-point foreground prompt.
  2. Pick the highest-scored returned mask, sanity-check its area.
  3. Composite the masked figurine onto a uniform white background.
  4. Write to ../freddie_masked/<same_filename>.

COLMAP poses derived from the original photos remain valid (same camera
intrinsics, same camera positions), so the existing sparse model at
../data/colmap/freddie/sparse/0/ is reused under ../data/colmap/freddie_masked/.

Idempotent: skips photos already present in the output directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR  = PROJECT_ROOT / "freddie"
DST_DIR  = PROJECT_ROOT / "freddie_masked"
CHECKPOINT = PROJECT_ROOT / "data" / "models" / "sam" / "sam2.1_hiera_large.pt"
MODEL_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"  # resolved by Hydra inside the sam2 package

BACKGROUND_RGB = (255, 255, 255)  # white background after masking

# Sanity bounds on the foreground area fraction. Outside these bounds the mask
# is almost certainly wrong (collapsed to a dot, or selected the whole frame).
AREA_FRAC_MIN = 0.02
AREA_FRAC_MAX = 0.70

# Auto-mask-generator mode. SAM 2 treats the figurine as several separate
# objects (face vs shirt vs pants come back as distinct masks), so any single
# prompt returns only one body part. Instead, enumerate every object mask in
# the image and union the ones whose centroids fall inside a central column,
# i.e. where Freddie was placed (footprint marked at capture time). This
# catches every body part regardless of SAM's part decomposition, while
# centroid filtering rejects the floor and wall masks.
CENTRAL_BOX = (0.20, 0.05, 0.80, 0.92)  # (x1, y1, x2, y2) fractions; mask centroid must lie inside
MASK_AREA_MIN = 0.002  # 0.2% of frame: drop SAM noise
MASK_AREA_MAX = 0.40   # 40% of frame: any single part bigger than this is the floor/wall


def union_central_masks(results: list[dict], hw: tuple[int, int]) -> tuple[np.ndarray, int]:
    """Union every SAM 2 mask whose centroid lies in the central box.

    The figurine comes back as several parts (head, shirt, pants). Each part's
    centroid lies inside CENTRAL_BOX because Freddie was placed on a marked
    footprint at capture time. Floor/wall masks have centroids near the edges
    and get rejected. Tiny noise and the whole-frame mask are filtered by area.
    Returns the unified bool mask plus the number of part-masks unioned.
    """
    H, W = hw
    total = float(H * W)
    x1, y1, x2, y2 = CENTRAL_BOX

    unified = np.zeros((H, W), dtype=bool)
    n_used = 0
    for r in results:
        seg = r["segmentation"]
        area_frac = r["area"] / total
        if not (MASK_AREA_MIN <= area_frac <= MASK_AREA_MAX):
            continue
        ys, xs = np.where(seg)
        if xs.size == 0:
            continue
        cx_frac = xs.mean() / W
        cy_frac = ys.mean() / H
        if not (x1 <= cx_frac <= x2 and y1 <= cy_frac <= y2):
            continue
        unified |= seg
        n_used += 1
    return unified, n_used


def composite_on_white(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """img: (H, W, 3) uint8. mask: (H, W) bool. Returns (H, W, 3) uint8."""
    out = np.empty_like(img)
    out[...] = BACKGROUND_RGB
    out[mask] = img[mask]
    return out


def main() -> int:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    photos = sorted(p for p in SRC_DIR.iterdir() if p.suffix.upper() in {".JPG", ".JPEG", ".PNG"})
    if not photos:
        print(f"No photos found under {SRC_DIR}.")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading SAM 2 from {CHECKPOINT}")
    sam = build_sam2(MODEL_CFG, str(CHECKPOINT), device=device)
    generator = SAM2AutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=2000,
    )

    print(f"Masking {len(photos)} photo(s) from {SRC_DIR} -> {DST_DIR}\n")
    print(f"{'#':>3}  {'filename':<20}  {'area%':>6}  {'parts':>5}  status")
    print("-" * 60)

    area_fracs: list[float] = []
    flagged: list[str] = []
    skipped = 0

    for i, src in enumerate(photos):
        dst = DST_DIR / src.name
        if dst.exists():
            skipped += 1
            print(f"{i:>3}  {src.name:<20}  {'-':>6}  {'-':>3}  skip (already exists)")
            continue

        img = np.array(Image.open(src).convert("RGB"))
        H, W = img.shape[:2]

        with torch.inference_mode():
            results = generator.generate(img)
        mask, n_parts = union_central_masks(results, (H, W))
        area_frac = mask.sum() / float(H * W)
        area_fracs.append(area_frac)

        flag = ""
        if n_parts == 0:
            flag = " !! no central masks found"
            flagged.append(src.name)
        elif not (AREA_FRAC_MIN <= area_frac <= AREA_FRAC_MAX):
            flag = " !! area out-of-bounds"
            flagged.append(src.name)

        out = composite_on_white(img, mask)
        Image.fromarray(out).save(dst, quality=95)
        print(f"{i:>3}  {src.name:<20}  {area_frac*100:>5.1f}%  {n_parts:>5}  ok{flag}")

    print("-" * 60)
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
