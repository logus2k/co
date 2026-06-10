# Report revision backlog

Tracks every remaining fix between the current release-candidate PDF (`project/tutorial2/nerf_tutorial2_v6_AC.pdf`, 26 pages) and submission-ready. Items are grouped by severity. Work top-down: address tier A before tier B before tier C.

Each item lists where to find it in the document, the current text, and the replacement text so the edit can be made by find-and-replace in Word and (where applicable) mirrored back into the notebook so the source of truth stays correct.

## Tier A — Blocking (correctness; fix before submission)

### [ ] A1. §8.1 page 12: within-paragraph SD inconsistency

The L1 collapse paragraph reports two different standard deviations for the same value: `14.08 ± 5.87` in the first sentence, then `The 5.85 dB pooled standard deviation` two sentences later. Pick the unrefined value (`5.87`) and use it consistently.

- **Current:** `The 5.85 dB pooled standard deviation reflects that approximately one in three training runs diverges.`
- **Replace with:** `The 5.87 dB pooled standard deviation reflects that approximately one in three training runs diverges.`
- **Mirror to notebook:** same paragraph in the §8 Interpretation Markdown cell (notebook line 4659 area — see [`08_loss_function_comparison.md`](08_loss_function_comparison.md) line 13 for the corresponding text in the reduction draft).

### [ ] A2. §12.3 page 22: garbled aspect-ratio sentence

The first bullet of §12.3 carries leftover text from the previous wording, producing a duplicated phrase that does not parse.

- **Current:** `NeRF on the captured scene rises from 14.98 dB to 16.37 dB after this fix alone NeRF reaches 16.37 dB; combining it with the foreground-masked SfM of §12.1 lifts it to 17.98 dB.`
- **Replace with:** `NeRF on the captured scene rises from 14.98 dB to 16.37 dB after this fix alone; combining it with the foreground-masked SfM of Section 12.1 lifts it to 17.98 dB.`
- This single replacement also resolves item B4 below (the `§12.1` section-sign character) by replacing it with `Section 12.1`.

### [ ] A3. §13 page 23: stray possessive in the L1 paragraph

- **Current:** `A Bayesian's sweep over the learning rate, a cosine schedule, and gradient clipping at norm 1.0 all fail`
- **Replace with:** `A Bayesian sweep over the learning rate, a cosine schedule, and gradient clipping at norm 1.0 all fail`

## Tier B — Polish (style and exposition; should fix)

### [ ] B1. §11.3 / Figure 6 page 19: orphan qualitative-grid figure

Figure 6 (the full-page 6×3 NeRF / GS / GT grid) currently sits at the end of §11.3 without any sentence introducing it. Either add a lead-in or drop the figure.

- **Suggested lead-in** (insert as a final short paragraph at the end of §11.3, page 18, before the figure): `Figure 6 shows three held-out test views per scene rendered by NeRF, Gaussian Splatting, and the ground-truth photograph alongside.`
- **Alternative:** drop Figure 6 entirely. The metric verdict in the body and Figure 5's bar chart already carry the comparison.

### [ ] B2. §9.1 page 15: unbold mid-prose numerical emphasis

The "Perceptual loss (L2 + LPIPS)" bullet contains mid-prose bolds on the LPIPS-drop figures, which violates the report-grade rule of reserving bold for structural labels.

- **Current:** `the substantiated positive. **Lego LPIPS drops 0.146 → 0.099 (−32%); Drums LPIPS drops 0.131 → 0.094 (−28%)**, at the cost of −0.54 dB on Lego PSNR and −0.59 dB on Drums.`
- **Replace with:** `the substantiated positive. Lego LPIPS drops 0.146 → 0.099 (−32%); Drums LPIPS drops 0.131 → 0.094 (−28%), at the cost of −0.54 dB on Lego PSNR and −0.59 dB on Drums.`

### [ ] B3. §12 page 20 Figure 7 caption: indefinite article

- **Current:** `Two Freddie photographs captured using a iPhone 11 Pro (left column)`
- **Replace with:** `Two Freddie photographs captured using an iPhone 11 Pro (left column)`

## Tier C — Optional consistency (nice to have)

### [ ] C1. §12.3 page 22: `§12.1` section-sign character

The section-sign character `§` is not on the user's keyboard and the rest of the document spells "Section X" out.

- **Status:** resolved automatically by the A2 replacement above. Leave unchecked until A2 is applied; check both off together.

### [ ] C2. §13 page 23 closing paragraph: missing "et al."

The closing of §13 names the recipe simply as `the original Kerbl 2023 recipe` while §10, §12.2, and the rest of the document use `Kerbl et al. 2023`. Align to the et al. form.

- **Current:** `the COLMAP sparse cloud itself, the original Kerbl 2023 recipe.`
- **Replace with:** `the COLMAP sparse cloud itself, the original Kerbl et al. 2023 recipe.`

### [ ] C3. §14 page 24 closing: drop emphatic "do"

- **Current:** `A larger-model or larger-dataset version of the project would test whether SGDR, multi-scale, or adaptive view sampling do produce positive results at that scale.`
- **Replace with:** `A larger-model or larger-dataset version of the project would test whether SGDR, multi-scale, or adaptive view sampling produce positive results at that scale.`

### [ ] C4. §10.2 pages 16–17: `800×800` spacing

Two occurrences of `800×800` (no spaces around the ×) where the rest of the document writes `800 × 800` with spaces. Search Word for `800×800` and replace each with `800 × 800`.

- **Pages affected:** §10.2 fourth bullet (page 16, "Empirically too aggressive on Lego at 800×800") and the §10.2 closing sentence (page 17, "15,000 iterations at 800×800").

## How to use this checklist

1. Open the Word document at the first unchecked item's page.
2. Apply the replacement exactly as listed.
3. Tick the box in this file (`[x]` instead of `[ ]`).
4. Move to the next item.
5. After tier A is fully checked, re-export the PDF for a quick visual confirmation of the three blocking fixes before continuing into tier B.

Mirror policy: items A1 also has a corresponding site in the notebook (`src/nerf_tutorial2_v6_AC.ipynb`) so the source of truth stays correct for future re-exports. Apply the same edit there after applying it in Word, in the same session, so the two files do not drift.
