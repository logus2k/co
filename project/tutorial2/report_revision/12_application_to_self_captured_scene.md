# 12. Application to a Self-Captured Real-World Scene

Sections 7 to 11 conducted the methodological comparison on the nerf_synthetic dataset, where every image is a clean Blender render and every camera pose is known to floating-point precision. The advantage is methodological rigour; the limitation is that the synthetic substrate says nothing directly about how the validated methods behave on real photographs, where sensor noise, lens distortion, exposure variation, motion blur, and Structure-from-Motion pose-estimation error are all present. This section closes that gap on a single self-captured object and reports the three structural adaptations the transfer from synthetic to real required.

## 12.1 Scene capture and pose recovery

The subject is a 15 cm painted resin Freddie Mercury figurine, captured in 71 photographs by an iPhone 11 Pro at fixed focal length with auto-exposure and auto-focus locked. The shoot used a hand-held spiral in three elevation passes, lighting held constant; the dataset is 71 JPEGs at 4032 × 3024 resolution.

Both reconstruction methods require, for every training image, the camera intrinsics and extrinsics, which on real captures are recovered from the photographs by Structure-from-Motion: feature extraction, exhaustive pairwise matching, and joint bundle adjustment over poses and 3-D points (Schönberger & Frahm 2016). We use COLMAP. The output is each photograph annotated with its recovered pose, plus a sparse cloud of 3-D points seen by multiple cameras.

A first COLMAP pass on the original photographs produces a 30,151-point cloud dominated by the textured hardwood floor and back wall rather than the figurine. As Section 12.4 will show, that distribution wrecks GS initialisation, so we replace every non-figurine pixel with a uniform white background using rembg with the ISNet-general-use cutout model, and re-run COLMAP from scratch on the masked images. SIFT finds no features on uniform white pixels, so every feature point that survives lies on the figurine by construction. The resulting cloud has 24,528 points spanning 3.6 × 3.0 × 1.7 world units centred on Freddie, and is the data Sections 12.2 onward consume.

**Figure 11** - Two captured Freddie photographs (left column) and their rembg foreground-masked counterparts (right column), at two orbit angles.

[Insert the same Figure 11 image used in the Word baseline.]

## 12.2 Training configurations and results

The configurations applied are the per-method winners from the synthetic study, applied without further hyperparameter tuning. NeRF uses the Section 7.2 winner: Adam at η = 10⁻³, L2 loss, 40,000 iterations, 200 × 200. Gaussian Splatting uses the Section 10 baseline: 15,000 iterations, L1+SSIM with α = 0.2, the four implementation details of Section 10.2, 800 × 800. The single scene-dependent intervention is in the GS initialiser: when the scene comes from COLMAP, the random unit-cube cloud is replaced with the COLMAP sparse cloud itself, the canonical Kerbl et al. 2023 recipe; the synthetic-scene results in Sections 10 and 11 are unaffected because the branch never fires for them.

Final test-set numbers: NeRF reaches 17.98 dB PSNR / 0.822 SSIM / 0.283 LPIPS; Gaussian Splatting reaches 29.15 dB / 0.979 / 0.021. GS beats NeRF by 11.17 dB on this scene, reversing the +0.75 dB synthetic Lego ranking. The orbit videos confirm the metric ranking qualitatively: GS reconstructs a sharp, photo-realistic figurine across the trajectory; NeRF produces a recognisable but blurry semi-transparent silhouette whose head and feet dissolve at off-axis viewpoints.

**Table 4** - Per-method PSNR, SSIM, and LPIPS on the synthetic Lego scene and the captured Freddie scene. Higher PSNR and SSIM and lower LPIPS each indicate better reconstruction.

| | NeRF | GS | NeRF − GS |
|---|---|---|---|
| Synthetic / Lego | 22.24 / 0.812 / 0.143 | 21.49 / 0.851 / 0.085 | NeRF +0.75 dB PSNR |
| Captured / Freddie | 17.98 / 0.822 / 0.283 | 29.15 / 0.979 / 0.021 | GS +11.17 dB PSNR |

## 12.3 Three structural adaptations

Reaching these numbers required three adjustments to the synthetic-winning configuration. The synthetic pipeline was validated entirely on Blender renders, which have three properties that do not transfer: images are exactly 800 × 800, the foreground is composited against a clean alpha-zeroed background, and every object sits at the world origin by convention. Each adaptation below addresses one structural mismatch, diagnosed from a specific failure-mode observation.

**Aspect-ratio mismatch in the loader.** The loader resizes every training image to a fixed square via `F.interpolate`. On 800 × 800 synthetic input this is a no-op; on a 4032 × 3024 iPhone capture it uniformly stretches the image to 1:1, breaking the per-pixel angle-versus-focal relationship the volume renderer depends on, so NeRF's MLP converges to a blurry compromise. The fix is a centre-crop to square before the resize, with a corresponding focal rescale. NeRF on the captured scene rises from 14.98 dB to 16.37 dB after this single change; the subsequent foreground-masked SfM lifts it the rest of the way to 17.98 dB.

**Background dominates Structure-from-Motion.** The background-dominated COLMAP cloud described in Section 12.1 is the second mismatch. NeRF degrades gracefully because volume rendering can hedge with semi-transparent fog; GS fails catastrophically because its primitives are initialised from a 3-D point cloud, and on a real-world capture without masking that cloud is dominated by texture-rich background, so the figurine never gets enough density. The fix is the foreground-masked pipeline of Section 12.1.

**Off-origin scene defeats random Gaussian initialisation.** GS places its 30,000 initial Gaussians uniformly in a unit cube around the world origin. For Blender scenes this is correct; for the COLMAP-recovered captured scene the figurine's centre lies at (+1.77, +0.03, −0.32), 3.6 world units offset from the random-init cube. Every initial Gaussian lands in empty space, image-space gradients are zero by ray geometry, no densification fires toward the figurine, and the loss converges to a near-constant white field. Replacing the random cube with the COLMAP sparse cloud (the original Kerbl et al. 2023 recipe) lifts PSNR from 12.33 dB to 29.15 dB on a single rerun, with no other hyperparameter change.

## 12.4 What the captured-scene study delivers

The 11 dB PSNR reversal between synthetic Lego (NeRF +0.75 dB) and captured Freddie (GS +11.17 dB) is not a claim that GS is universally better than NeRF; it is a claim about how initialisation quality interacts with the two representations. NeRF's MLP is implicit in the sense that it never sees the COLMAP point cloud, so a good cloud cannot help it, and NeRF drops from a synthetic 22.24 dB to 17.98 dB on the captured scene, a 4.26 dB generalisation gap explained by real sensor noise, residual JPEG artefacts at mask edges, and small pose-registration errors. Gaussian Splatting rises from 21.49 dB to 29.15 dB because the COLMAP point cloud places every initial Gaussian directly on the figurine's surface and in approximately the right colour. On the synthetic data the random initialisation happens to overlap the object only because Blender's convention places every object at the world origin, which is also the centre of the random-init cube.

The three structural adaptations are themselves the methodological contribution of the captured-scene study. The static-scene assumption with masked input, the aspect-ratio invariance of the loader, and the COLMAP-derived initialisation for the explicit-primitive representation are three properties of the real-world reconstruction problem that the synthetic-data results do not surface, and surfacing them is the substantive reason a captured-scene study was worth doing.
