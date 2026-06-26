# Diagram Audit: Review, Recommendations & Remake Plan

## Overview

The paper currently has **5 figures**:

| Fig | Type | Content | Location |
|-----|------|---------|----------|
| 1 | TikZ block diagram | System architecture (data flow) | Line 53–88 |
| 2 | TikZ block diagram | MLP architecture | Line 98–126 |
| 3 | TikZ block diagram | CNN architecture | Line 146–198 |
| 4 | Matplotlib PNG | Magnetic CDF plot | Line 285–290 |
| 5 | Matplotlib PNG (side-by-side) | Dual KalmanNet CDF comparison | Line 312–317 |

---

## Existing Figure Assessment

### Fig. 1 — System Architecture ✅ KEEP (minor resize)

**Verdict: Good — the most important figure.** It correctly shows data flow: PDR → Predict, WiFi+Mag → Innovations → GRU → Update. The block styles are clean, colors are distinct.

> [!NOTE]
> **Minor issue:** Currently at `0.9\columnwidth`. This is fine but could benefit from being exactly `\columnwidth` since it's already compact. No content changes needed.

---

### Fig. 2 — MLP Architecture ⚠️ KEEP (remake for aesthetics)

**Issues identified:**
1. **Overfull hbox warning**: The "Dropout(0.3)" text causes a 1.27pt overflow — cosmetically noticeable at print resolution
2. **Weight labels (W¹, W², W³) are unnecessary**: No other IEEE indoor-positioning paper labels weight matrices on architecture diagrams. They add clutter without information (the reader already knows FC layers have weights).
3. **Box widths are too narrow** (`text width=1.8cm`) — the Dropout text gets squeezed
4. **No dimension annotations**: The figure doesn't show the data dimensions flowing between layers (e.g., N→256→256→M). Good architecture figures show tensor shapes at each stage.

**Recommendation:** Remake with wider boxes (2.2cm), remove W labels, and add dimension annotations above the arrows.

---

### Fig. 3 — CNN Architecture ✅ KEEP (minor refinement)

**Verdict: Accurate and well-structured.** The dashed encoder group box, the branching to dual heads, and the BatchNorm/Dropout labels are all correct.

> [!NOTE]
> **Minor issue:** The spacing between conv blocks is very tight (0.25cm). At column width, the labels may be cramped on some PDF viewers. Consider increasing to 0.35cm.

---

### Fig. 4 — Magnetic CDF Plot ❌ NEEDS REMAKING

**Critical issues:**
1. **Only 1 curve**: A standalone CDF with just one line is not very informative. It doesn't compare anything.
2. **Long tail distorts the plot**: The x-axis extends to 38m (!!) to accommodate a few outliers. This compresses all meaningful data into the left 1/3 of the plot.
3. **Default matplotlib styling**: No grid refinements, default font sizes, default colors. Looks like a debug plot, not a publication figure.
4. **Missing comparisons**: This should compare the window-size sweep (T=50 vs T=84 vs T=134) to justify the T=84 choice, OR compare against a single-frame baseline to demonstrate why sequence matching is superior.

**Recommendation:** Remake as a **window-size comparison CDF** (3-4 curves) with x-axis capped at ~15m, IEEE-appropriate fonts, and tighter formatting.

---

### Fig. 5 — Dual KalmanNet CDF ⚠️ NEEDS RESTYLING

**Issues:**
1. **Too wide for single-column**: The side-by-side layout produces two very narrow sub-panels when constrained to `\columnwidth`. The legend text becomes nearly unreadable at print resolution.
2. **Low DPI apparent**: At column width, the matplotlib rendering looks pixelated. Needs higher DPI or vector format.
3. **Legend placement**: The left panel's legend overlaps the plot area. The right panel's legend is at the bottom-right, partially obscured.
4. **No grid**: IEEE CDF plots typically have subtle gridlines for readability.

**Recommendation:** Either (a) stack vertically (2 rows, 1 column) at `\columnwidth`, or (b) use `figure*` for double-column. I recommend (a) since we already have 5 figures and space is tight.

---

## New Figure Recommendation

> [!IMPORTANT]
> **Proposed: Fig. 6 — Example Trajectory Comparison**

A plot showing a single trajectory with:
- Gray dots = surveyed corridor nodes (environment context)
- Blue line = ground truth path
- Red dashed = PDR-only trajectory (showing cumulative drift)
- Green solid = DualKalmanNet output (showing corrections)

**Why this is essential:** The paper's core claim is that the filter corrects PDR drift using Wi-Fi and magnetic measurements. A trajectory plot is the most intuitive way to demonstrate this — every indoor localization paper at GlobeCom includes one. Currently, the paper has only CDFs (aggregate statistics) with no spatial visualization. Reviewers will expect to *see* a trajectory.

The code already generates this in `stage3_synthetic_eval.py:236-240` — we just need to format it for publication.

---

## Summary of Actions

| Figure | Action | Effort |
|--------|--------|--------|
| Fig. 1 (Architecture) | Keep, resize to `\columnwidth` | Trivial |
| Fig. 2 (MLP) | Remake: wider boxes, remove W labels, add dimensions | Medium |
| Fig. 3 (CNN) | Keep, widen spacing from 0.25→0.35cm | Trivial |
| Fig. 4 (Mag CDF) | **Remake entirely**: multi-curve comparison, cap x-axis, IEEE styling | High |
| Fig. 5 (KalmanNet CDF) | **Restyle**: stack vertically, higher DPI, proper legend placement | Medium |
| Fig. 6 (NEW: Trajectory) | **Create**: example trajectory with GT, PDR, and fusion overlay | Medium |

> [!WARNING]
> With 6 figures in a 5-page paper (6 page limit for GlobeCom), space is tight. If needed, we can merge Fig. 4 (Mag CDF) into Fig. 5 (making it a 3-panel figure: Mag CDF, Full WiFi CDF, Degraded WiFi CDF), saving one figure slot.

---

## Open Questions

1. **Do you want me to merge Fig. 4 + Fig. 5 into one multi-panel figure to save space, or keep them separate?**
2. **For the trajectory plot (Fig. 6), should I use one of the 3 pre-generated example walks from the code, or would you prefer a specific corridor section?**
3. **Do you want the plots in matplotlib (regenerated from data) or should I use TikZ/pgfplots (vector, but harder to iterate)?**
