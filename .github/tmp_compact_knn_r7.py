from pathlib import Path

path = Path("paper/main.tex")
text = path.read_text(encoding="utf-8")

# Only the new R7 figure uses these exact subfigure widths.
count = text.count(r"{0.48\textwidth}")
if count != 3:
    raise SystemExit(f"expected 3 R7 subfigure widths, found {count}")
text = text.replace(r"{0.48\textwidth}", r"{0.44\textwidth}", 3)

old = r'''\caption{KNN baselines and current fusion CDFs. Panel (a) is a separate real static-fingerprint, held-out-device experiment and is not numerically comparable to the synthetic walk errors in (b)--(c). Panels (b)--(c) use exactly the same 250-training/60-test synthetic trajectory generation as KalmanNet. The non-temporal KNN receives the contemporaneous Wi-Fi heatmap fix, magnetic-CNN fix and log-variance, and availability masks, but no PDR motion or recurrent history. Axes, ticks, and legends are regenerated at paper-readable sizes.}'''
new = r'''\caption{KNN baselines and current fusion CDFs. (a) Real static fingerprints with S9+ held out. (b)--(c) Matched 250-training/60-test synthetic walks; the KNN uses current Wi-Fi/magnetic fixes and masks only, without PDR motion or recurrent history.}'''
if old not in text:
    raise SystemExit("R7 caption not found")
text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
