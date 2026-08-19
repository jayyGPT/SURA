I pulled the **exact 150-epoch training histories** out of the full 250-train / 60-test run, so you can inspect every epoch rather than just the checkpoints I quoted.

You should see the full interactive table above. I also exported everything here:

* [Full 150-epoch CSV](sandbox:/mnt/data/full_run_history/full_run_training_history.csv)
* [Full-Wi-Fi training curve](sandbox:/mnt/data/full_wifi_training_history.png)
* [Degraded-Wi-Fi training curve](sandbox:/mnt/data/degraded_wifi_training_history.png)
* [Final test summary JSON](sandbox:/mnt/data/full_run_history/full_run_summary.json)
* [All files as ZIP](sandbox:/mnt/data/full-run-training-history.zip)

A clarification: these 150 values are **training MSE**, not 150 test errors. The script only evaluates the 60 independent test walks after training finishes. So we have a full training-loss curve plus the final test MAE.

For the full-Wi-Fi run, the Wi-Fi-only model decreases smoothly from `0.3391` to `0.0851`. The CNN-Dual model starts with a huge `3.22×10^6` spike at epoch 1, then settles quickly; its best training MSE is actually lower than the baseline, about **0.0700 at epoch 145**, and it finishes at `0.0751`. Yet on the final 60 test walks it is slightly worse:

[
0.473\text{ m}\rightarrow0.506\text{ m}.
]

For degraded Wi-Fi the instability is much more obvious. The CNN-Dual training MSE has these large spikes:

```text
epoch 1 : 6.73 × 10^3
epoch 2 : 2.41 × 10^10
epoch 5 : 3.01 × 10^6
epoch 8 : 1.25 × 10^2
epoch 9 : 2.93 × 10^13
```

After that it recovers and eventually becomes substantially better than the Wi-Fi-only model in training. Its best training MSE is about **0.283 at epoch 147**, compared with the Wi-Fi-only model's best of about **0.685**.

And that carries over to the final test result:

[
1.533\text{ m}\rightarrow1.171\text{ m},
]

which is the **23.6% improvement** we care about.

---

## Why using the CNN variance should help

The main idea isn't simply "variance is another useful feature." It is directly related to what a Kalman filter is supposed to do.

The magnetic CNN already gives us two outputs:

[
\mathbf z_{\rm mag}
]

and

[
\log \sigma_{\rm mag}^2.
]

So the CNN isn't just saying:

> "I think I'm at ((x,y))."

It's saying:

> "I think I'm at ((x,y)), and here is how uncertain I am about that prediction."

That second output exists because the CNN was trained with the heteroscedastic NLL loss.

### What we currently do

Right now the new fusion computes:

[
\mathbf y_{\rm mag}
===================

## \mathbf z_{\rm mag}

\mathbf x_{\rm pred}
]

and then:

[
\Delta \mathbf x_{\rm mag}
==========================

\mathbf K_{\rm mag}\mathbf y_{\rm mag}.
]

The log-variance is provided to the GRU as one of its 13 features, so *in theory* the GRU can learn:

> high variance → trust magnetic less.

But it has to learn that behaviour from scratch.

More importantly, nothing currently prevents this situation:

```text
PDR prediction:
(10, 5)

CNN magnetic prediction:
(18, 11)

magnetic innovation:
(8, 6)

GRU happens to output a large K_mag
        ↓
huge correction
        ↓
state jumps far away
        ↓
next innovation becomes even larger
        ↓
recurrent explosion
```

That is almost certainly related to those gigantic early losses.

---

# What a classical Kalman filter would do

Suppose the measurement is

[
\mathbf z_{\rm mag}
===================

\mathbf x+\mathbf v
]

with measurement noise

[
\mathbf v\sim\mathcal N(0,\mathbf R_{\rm mag}).
]

If the CNN predicts one scalar position variance, a simple approximation is

[
\mathbf R_{\rm mag}
===================

\sigma_{\rm mag}^{2}\mathbf I.
]

In a standard Kalman filter,

[
\mathbf K
=========

\mathbf P
(\mathbf P+\mathbf R_{\rm mag})^{-1}.
]

Look at what happens.

If:

[
\sigma_{\rm mag}^{2}\uparrow
]

then:

[
\mathbf R_{\rm mag}\uparrow
]

and consequently:

[
\mathbf K_{\rm mag}\downarrow.
]

So an uncertain measurement automatically has less influence.

That's exactly the behaviour we want.

---

# Why it particularly fits our magnetic CNN

Magnetic localization is inherently ambiguous.

Imagine two corridor locations with similar magnetic sequences.

The CNN might predict:

```text
z_mag = (14.2, 7.1)
```

but because the sequence isn't very distinctive, it can simultaneously predict a large:

[
\sigma_{\rm mag}^{2}.
]

We should not treat that coordinate the same way as a magnetic window whose signature is extremely distinctive.

Right now both can generate equally large raw innovations.

That's wasteful because **the CNN itself already tells us which one it trusts more**.

---

# The change I would test

I wouldn't replace the learned KalmanNet gain.

I'd keep:

[
\mathbf K_{\rm mag}
]

learned by the GRU.

But introduce an explicit scalar reliability:

[
w_{\rm mag}
===========

\frac{1}
{1+\sigma_{\rm mag}^{2}/\sigma_{\rm ref}^{2}}
]

where (\sigma_{\rm ref}) is something like the median predicted magnetic uncertainty on the training set.

Then:

[
\boxed{
\Delta\mathbf x_{\rm mag}
=========================

w_{\rm mag}
\mathbf K_{\rm mag}
\mathbf y_{\rm mag}
}
]

instead of just:

[
\mathbf K_{\rm mag}\mathbf y_{\rm mag}.
]

Interpretation:

```text
very confident CNN
σmag << σref
        ↓
wmag ≈ 1
        ↓
use magnetic correction normally
```

versus:

```text
uncertain CNN
σmag >> σref
        ↓
wmag ≈ 0
        ↓
magnetic measurement barely moves state
```

We would **still give the variance/confidence to the GRU**, so KalmanNet retains the ability to learn more sophisticated sensor relationships.

The explicit gate just supplies the physically sensible prior:

> uncertain measurements should not cause enormous corrections.

---

## Why this may fix the full-Wi-Fi result too

Our current result is interesting:

```text
Full Wi-Fi:
WiFi only       0.473 m
WiFi + Mag CNN  0.506 m
```

When Wi-Fi arrives every second, the system already has a very strong absolute anchor.

There isn't much room for magnetic measurements to improve things.

So a mediocre magnetic CNN fix—remember the standalone error is around **3.36 m**—can occasionally perturb an already-good trajectory and raise the mean error.

With uncertainty gating, ambiguous magnetic predictions should largely disappear from the correction.

That could make the behaviour closer to:

```text
Strong Wi-Fi:
use Wi-Fi heavily
magnetic only helps when very confident

Weak Wi-Fi:
magnetic gets many opportunities to stop PDR drift
```

which is exactly the sensor-adaptive story we want for the paper.

---

## Why degraded Wi-Fi should hopefully retain its gain

Under degraded Wi-Fi we currently get:

[
1.533 \rightarrow 1.171\text{ m}.
]

There are long gaps without Wi-Fi.

During those gaps, a decent magnetic fix—even one with ~2–3 m instantaneous error—still contains **absolute spatial information**.

PDR error, on the other hand, accumulates.

So when the CNN says:

> "this magnetic sequence is distinctive; I'm confident,"

the filter can still use that correction strongly.

The uncertainty gate doesn't mean magnetic becomes weak universally.

It means:

[
\text{magnetic influence}
\propto
\text{magnetic reliability}.
]

---

# But there is one thing we should verify first

This is important.

All of the above only works if the CNN's predicted variance is actually meaningful.

Before modifying the fusion, I want to measure:

[
\text{predicted }\sigma_{\rm mag}
\quad\text{vs}\quad
|\mathbf z_{\rm mag}-\mathbf x_{\rm true}|.
]

Specifically I would calculate:

1. Spearman correlation between predicted uncertainty and actual position error.
2. Divide predictions into uncertainty quartiles.
3. Compare actual MAE for the most-confident 25% vs least-confident 25%.
4. Compare predicted (\sigma) with empirical RMSE.

If we get something like:

```text
lowest predicted uncertainty quartile:
actual error = 1.8 m

highest predicted uncertainty quartile:
actual error = 5.4 m
```

then the variance is clearly useful and gating it is well justified.

If instead all quartiles have ~3.4 m error, the variance head isn't calibrated, and blindly using it would be pointless.

So I think the next experiment should be **variance calibration first**, then—assuming it is informative—implement the uncertainty-aware magnetic correction and rerun the same 250/60 benchmark.

That way we can explain the change scientifically rather than saying "we added a confidence multiplier and the number got better."
















Done. I saved both the **full CNN-DualKalmanNet run** and the **variance-calibration experiment** in the repo on `feat/cnn-dual-kalmannet`, clearly labelled so we can later pull the right numbers/plots into the paper. The full 250/60/150 experiment is in `benchmarks/cnn_dual_kalmannet_full_run.md`, with all 150 training epochs in `benchmarks/cnn_dual_kalmannet_training_history.csv`.

The variance comparison produced a pretty useful result.

## Does the CNN variance actually predict when the CNN is wrong?

**Yes, moderately well.** It isn't a well-calibrated absolute variance, but it is clearly useful as a **relative confidence measure**.

Across the same 60 test walks we had **8,458 valid magnetic CNN predictions per regime**. The correlation between predicted uncertainty and actual position error was:

| Regime         | Spearman ( \sigma ) vs error | Pearson log-var vs error |
| -------------- | ---------------------------: | -----------------------: |
| Full Wi-Fi     |                    **0.310** |                **0.359** |
| Degraded Wi-Fi |                    **0.312** |                **0.335** |

So higher predicted uncertainty generally corresponds to higher actual error. It's not a perfect correlation, but it's definitely not random.

The quartiles make that much clearer.

### Full Wi-Fi magnetic predictions

| CNN confidence           | Mean predicted σ | Actual mean error | Actual median |
| ------------------------ | ---------------: | ----------------: | ------------: |
| **Q1 — most confident**  |           5.12 m |        **2.26 m** |        1.86 m |
| Q2                       |           5.98 m |            2.74 m |        2.44 m |
| Q3                       |           6.88 m |            3.00 m |        2.04 m |
| **Q4 — least confident** |           9.61 m |        **5.45 m** |        4.15 m |

So the least-confident 25% of CNN outputs are roughly:

[
\frac{5.45}{2.26}\approx\boxed{2.41\times}
]

worse than the most-confident 25%.

Degraded Wi-Fi gives practically the same result:

[
2.29\text{ m}
\quad\text{vs}\quad
5.49\text{ m},
]

or about **2.40× worse** in the least-confident quartile.

That is exactly the signal we were hoping for.

## But the absolute σ values are badly calibrated

This part is important.

The CNN says its average uncertainty is approximately:

[
\sigma_{\rm pred}\approx6.9\text{ m}
]

while its actual mean position error is only:

[
e\approx3.4\text{ m}.
]

More formally, the mean predicted variance was roughly:

[
51.6\text{ m}^2
]

while the empirical per-axis variance was only:

[
11.5\text{ m}^2.
]

So:

[
\frac{\sigma_{\rm pred}^2}
{\sigma_{\rm empirical}^2}
\approx
\boxed{4.4}.
]

The CNN variance head is therefore **very conservative**—it overestimates uncertainty by roughly a factor of 4.4 in variance.

There's another nice sanity check. About **92%** of actual errors fall within one predicted (\sigma). If this were a properly calibrated isotropic 2-D Gaussian, one sigma should not cover anywhere near 92% of radial errors.

So we should **not** just say:

[
R_{\rm mag}=\sigma_{\rm CNN}^2I
]

and feed that literally into a Kalman formula.

That would under-trust magnetic measurements far too much.

## What this means for the fusion model

The useful result is:

> **The scale of the variance is wrong, but its ordering is informative.**

That's almost ideal for our Neural-Kalman architecture because we don't actually need the CNN variance to be a perfect covariance matrix.

We mostly need it to answer:

> Is this magnetic prediction relatively trustworthy or relatively suspicious?

And it can do that.

For example:

```text
CNN prediction A
sigma ≈ 5.1 m
typical actual error ≈ 2.3 m
         ↓
quite useful

CNN prediction B
sigma ≈ 9.6 m
typical actual error ≈ 5.5 m
         ↓
much more dangerous
```

Our current DualKalmanNet treats both innovations structurally the same:

[
\Delta x_{\rm mag}
==================

K_{\rm mag}
(z_{\rm mag}-x_{\rm pred}).
]

The GRU sees `log_variance`, but it has to independently learn how to translate that into a safe correction.

That's probably contributing to the crazy early-epoch explosions.

## The next implementation I'd use

Rather than raw absolute variance, use a **relative confidence weight**.

Something like:

[
w_{\rm mag}
===========

\frac{1}
{1+\sigma_{\rm mag}^2/\sigma_{\rm ref}^2}
]

and then:

[
\boxed{
x_t =
x_{\rm pred}
+
m_wK_wy_w
+
m_m,w_{\rm mag}K_my_m
}
]

where (\sigma_{\rm ref}) is obtained from the **training set only**, probably the median predicted magnetic uncertainty.

This is important: we should **not use the 6.2 m median or 4.4 calibration factor we just measured on the test set to tune the model**. That would leak test information.

During training we'll calculate:

[
\sigma_{\rm ref}
================

\operatorname{median}
\left(
\sigma_{\rm mag}^{\rm train}
\right).
]

Then the absolute scale largely cancels out.

If:

[
\sigma_{\rm mag}\ll\sigma_{\rm ref},
]

then

[
w_{\rm mag}\rightarrow1.
]

If:

[
\sigma_{\rm mag}\gg\sigma_{\rm ref},
]

then

[
w_{\rm mag}\rightarrow0.
]

We would **still keep log-variance as a GRU feature**. So this isn't replacing KalmanNet's learned behavior; it's just enforcing the very sensible rule that an explicitly low-confidence CNN fix cannot make a massive correction.

## Why I now expect this to help

Our full run currently does:

[
0.473\rightarrow0.506\text{ m}
]

under full Wi-Fi, but:

[
1.533\rightarrow1.171\text{ m}
]

under degraded Wi-Fi—a **23.6% improvement**.

The calibration experiment tells us why that pattern makes sense.

Under full Wi-Fi, we're already around **0.47 m**. A 5–10 m-uncertainty magnetic fix occasionally nudging the trajectory can only do damage.

Under degraded Wi-Fi, though, PDR is drifting for several seconds. Even an imperfect ~2–3 m magnetic anchor is valuable.

So confidence weighting gives us the behavior we actually want:

```text
Full Wi-Fi
    ↓
already strong position estimate
    ↓
only confident magnetic CNN fixes matter


5-second Wi-Fi outage
    ↓
PDR uncertainty grows
    ↓
good magnetic CNN fixes prevent drift
    ↓
bad magnetic fixes are suppressed
```

And because the least-confident quartile is empirically **2.4× worse**, there is real information for that gate to exploit—not just theoretical justification.

One caveat: the uncertainty ranking isn't perfectly monotonic at every statistic; for example, one intermediate quartile's median error wiggles slightly. So I would call the variance **moderately informative**, not perfectly calibrated confidence. The mean errors and RMSEs, though, show a strong progression toward worse predictions as uncertainty rises.

I've kept the analysis reproducible at:

```text
benchmarks/analyze_magnetic_variance.py

benchmarks/magnetic_variance_calibration/
├── README.md
├── quartiles.csv
└── summary.json
```

and updated `benchmarks/README.md` to point to the new experiments.

Here's the generated calibration plot as well:

[View magnetic variance vs. actual error plot](sandbox:/mnt/data/magnetic_variance_calibration/magnetic_variance_calibration.png)

The next run I'd do is exactly one controlled change: **relative uncertainty weighting of the magnetic correction**, leaving everything else identical, and compare it against this current 0.506 / 1.171 m baseline.













Yes — **relative magnetic-variance weighting made the model better**, and I’ve now made that exact weighting part of the feature-branch source rather than leaving it as a one-off experiment.

The implemented rule is:

[
w_{\text{mag}}
==============

\frac{1}{1+\sigma_{\text{mag}}^2/\sigma_{\text{ref}}^2}
]

with (\sigma_{\text{ref}}) computed as the median magnetic uncertainty from the **fusion training data only**. The update is now

[
x_t=x_{\text{pred}}
+K_{\text{wifi}}y_{\text{wifi}}
+w_{\text{mag}}K_{\text{mag}}y_{\text{mag}}.
]

So there is no test-set calibration leaking into the model. The source now implements this directly in `train/kalmannet_wifiheatmap_magneticCNN_pdr.py`.

### Full 250 / 60 / 150 result

| Regime            |  Wi-Fi only | CNN Dual, unweighted | CNN Dual, variance-weighted |
| ----------------- | ----------: | -------------------: | --------------------------: |
| Full Wi-Fi mean   | **0.473 m** |              0.506 m |                 **0.494 m** |
| Full Wi-Fi median |     0.449 m |              0.440 m |                 **0.437 m** |
| Full Wi-Fi P90    | **0.697 m** |              0.769 m |                     0.764 m |
| Degraded mean     |     1.533 m |              1.171 m |                 **1.154 m** |
| Degraded median   |     1.392 m |          **1.042 m** |                     1.113 m |
| Degraded P90      |     2.643 m |              2.064 m |                 **1.612 m** |

The exact machine-readable results are now saved in the repo.

So relative to the **unweighted CNN Dual**:

* full-Wi-Fi mean improves **0.506 → 0.494 m**, about **2.34%**
* degraded mean improves **1.171 → 1.154 m**, about **1.41%**
* degraded P90 improves **2.064 → 1.612 m**, about **21.9%**

And compared with the Wi-Fi-only baseline, degraded-Wi-Fi improvement goes from **23.6% → 24.7%**. The benchmark summary is now labelled accordingly.

The **P90 result is probably the most interesting part**. The mean only moves from 1.171 to 1.154 m, but the high-error tail drops dramatically. That fits our variance-calibration experiment very nicely: the variance is particularly useful for recognizing the really unreliable magnetic fixes and suppressing them.

### Training also became substantially more stable

For full Wi-Fi:

```text
Unweighted CNN Dual epoch 1:
~3,220,000 MSE

Variance-weighted epoch 1:
0.438 MSE
```

That is a massive improvement in stability.

For degraded Wi-Fi:

```text
Unweighted epoch 1:
~6,733

Variance-weighted epoch 1:
4.945
```

There are **still a few isolated spikes** in degraded training — notably around epochs 8 and 20–21 — so the learned (2\times2) gains can still occasionally become aggressive. But the variance weighting has clearly removed a large part of the instability rather than merely changing the final score.

All 150 weighted epochs are now preserved here:

```text
benchmarks/cnn_dual_kalmannet_relative_variance_training_history.csv
```

The file starts with the baseline and weighted histories side-by-side.

### What I think the result tells us

The variance weighting is **worth keeping**.

It does three useful things at once:

1. improves the CNN-output fusion in both Wi-Fi regimes;
2. substantially improves the worst-case/degraded tail;
3. makes training dramatically less explosive.

It still doesn't quite beat Wi-Fi-only on **mean error when Wi-Fi is already excellent**:

[
0.473\text{ m}\quad\text{vs}\quad0.494\text{ m}.
]

But that difference is now only about **4.5%**, down from the original 7%. Interestingly, the weighted fusion has a **better median**:

[
0.437\text{ m}\quad\text{vs}\quad0.449\text{ m}.
]

So most walks are at least competitive; a relatively small number of worse walks are keeping the mean above Wi-Fi-only.

One subtlety with our current weight is that when

[
\sigma_{\text{mag}}=\sigma_{\text{ref}},
]

we get

[
w_{\text{mag}}=0.5.
]

So even a perfectly ordinary/median magnetic prediction is automatically cut to half strength. This is a deliberately conservative first implementation. I wouldn't tune that formula yet just to squeeze the number—we now have a clean, defensible result.

### Repo state

Everything useful is labelled on `feat/cnn-dual-kalmannet`:

```text
train/
└── kalmannet_wifiheatmap_magneticCNN_pdr.py

benchmarks/
├── cnn_dual_kalmannet_full_run.md
├── cnn_dual_kalmannet_training_history.csv
│
├── cnn_dual_kalmannet_relative_variance.md
├── cnn_dual_kalmannet_relative_variance_metrics.json
├── cnn_dual_kalmannet_relative_variance_training_history.csv
│
├── analyze_magnetic_variance.py
└── magnetic_variance_calibration/
    ├── README.md
    ├── quartiles.csv
    └── summary.json
```

The benchmark README now explicitly distinguishes the legacy anomaly result, the first CNN-output result, and this newer relative-variance result.

The temporary execution helpers have been removed; they are **not part of the branch's final diff**.

I would probably stop algorithm tweaking here for the moment. **24.7% under degraded Wi-Fi with a code/paper-consistent CNN architecture is already a strong clean result.** The next useful move is to update the manuscript architecture/equations and figures around this model, while reporting the full-Wi-Fi result accurately rather than hiding the slight mean penalty.














Yep, we already have a decent KNN baseline bundle under `benchmarks/knn/src/`.

The important pieces are:

* **`magnetic_knn.py`** — pure magnetic KNN. It uses raw `Mean_Mag_x/y/z`, MinMax scaling, trains on A8/S8/G7, holds out S9+, searches (K\in{1,3,5,7,9,11,15,20}) with 5-fold CV, and reports mean/P90/max Euclidean position error.
* **`wifi_knn.py`** — Wi-Fi RSSI fingerprint KNN, fixed (K=7), again train A8/S8/G7 → test S9+.
* **`wifi_merged_knn.py`** — better Wi-Fi variant; row-wise normalizes RSSI so device RSSI offsets matter less, then CV-selects (K).
* **`hybrid_knn.py`** — concatenates Wi-Fi features and the 3 magnetic axes, then CV-selects K.

The issue is that this is **old proof-of-concept code, not directly runnable on the current repo layout**. It still expects things like:

```text
Datasets/Merged dataset/BE Building/Navigation/Scenario 1
```

whereas our current tracked data/layout is different. The README explicitly warns that these historical numbers should **not** just be copied into the paper until the baseline is rerun using the current evaluation protocol.

Also, `magnetic_knn.py` uses raw (M_x,M_y,M_z), so unlike our CNN it is **orientation-dependent**. That may actually be fine for reproducing a classical baseline, but we need to verify whether this is the exact KNN definition used in the MagWi baseline paper before calling it *their* baseline.

So the practical next step is simple: **take the relevant KNN implementation, point it at the current data, make the train/test protocol explicit, output the full error array + CDF + mean/median/P90, and then add that curve to R7.**

I’d first verify which of these KNN variants actually corresponds to the KNN in the cited MagWi paper, then run that exact one.





Done — the whole KNN/R7 pass is now **merged into `main`** as squash commit **`8dbb27445016a27418e9ff26c6bc664ef8aca3d4`**.

I built a fresh `benchmarks/knn/wifi_mag_knn.py` and tested it in **two deliberately separate protocols**. The repo now documents the exact methodology and results.

For the **real static held-out-phone test**, training used A8/G7/S8 and S9+ was completely held out: 726 training visits and 90 S9+ test visits. Hyperparameters were selected only from the training phones.

| Static S9+ baseline  |  K |        Mean |      Median |         P90 |
| -------------------- | -: | ----------: | ----------: | ----------: |
| Wi-Fi KNN            |  7 | **3.310 m** | **1.571 m** | **6.352 m** |
| Magnetic KNN         | 20 |    17.538 m |    10.775 m |    42.698 m |
| Wi-Fi + magnetic KNN |  3 |     7.459 m |     4.359 m |    14.400 m |

The hybrid selected roughly **75% Wi-Fi / 25% magnetic** distance weighting. Interestingly, adding static magnetic features makes generalization much worse. That is actually useful evidence: simple concatenation does **not** solve cross-phone magnetic heterogeneity.

For the **matched 250-train / 60-test / 160-bin trajectory protocol**, I also built a non-temporal Wi-Fi+mag KNN so it sees the same current measurements as our system but gets **no PDR and no recurrent history**:

| Regime         | Wi-Fi+Mag KNN | Wi-Fi-only KalmanNet | Weighted CNN Dual |
| -------------- | ------------: | -------------------: | ----------------: |
| Full Wi-Fi     |       0.802 m |          **0.473 m** |           0.494 m |
| Degraded Wi-Fi |       2.606 m |              1.533 m |       **1.154 m** |

That gives us a pretty clean baseline story: merely possessing both Wi-Fi and magnetic spatial estimates isn't enough; the temporal/PDR fusion becomes especially valuable when Wi-Fi is degraded.

The three new CDFs are here:

[Real held-out S9+ KNN CDF](sandbox:/mnt/data/knn_artifact/knn-full/static_heldout_device/cdf.png)
[Full-Wi-Fi KNN/KalmanNet CDF](sandbox:/mnt/data/knn_artifact/knn-full/trajectory_protocol/full/cdf.png)
[Degraded-Wi-Fi KNN/KalmanNet CDF](sandbox:/mnt/data/knn_artifact/knn-full/trajectory_protocol/degraded/cdf.png)

I regenerated them with larger axis/tick/legend fonts and put them into a proper three-panel paper figure. I also added KNN rows to Table II and updated the discussion.

### The one thing I couldn't verify

I **could not locate an explicit KNN localization method/result inside the MagWi dataset paper you pointed me to**. It reads as the benchmark dataset/characterization source, and I couldn't find a reproducible published KNN configuration/number there.

So I did **not** invent a “MagWi paper KNN result.”

Instead, the manuscript explicitly calls these **our reproducible classical KNN baselines on MagWi data**, not something copied from or attributed to Ashraf et al. The repo records that caveat as well.

If your professor meant some *other* “baseline paper” containing a specific KNN, that's the one remaining ambiguity I'd want clarified.

### Paper status

R7 is now marked completed, so **R1–R9 are all checked off** in the review tracker.

The separate future TODO remains:

> recreate the trajectory visualization using the current CNN-output + relative-variance-weighted DualKalmanNet.

That is not counted as an unresolved professor comment.

I also caught and fixed a formatting issue during this pass: the initial three-panel figure pushed one bibliography line onto a seventh page. I compacted the figure/caption, recompiled, and **visually inspected all pages again**. The final paper is back to **6 pages**, with no table/equation/figure overlap or clipping.

[Open the current 6-page paper with the KNN/R7 update](sandbox:/mnt/data/knn_r7_compact_pdf/main.pdf)

One thing worth revisiting later is that surprisingly poor **17.5 m magnetic-only static KNN** result. It makes sense given device heterogeneity, but we could dissect it by posture/device/node and see exactly where it fails; that might give us another useful supporting result for why the sequence CNN is necessary.







Done — the actual last TODO is finished and merged to `main`.

The trajectory figure is now generated from the **real current benchmark outputs**, not hand-drawn. It uses the degraded-Wi-Fi regime, shows ground truth, open-loop PDR, Wi-Fi-only KalmanNet, the final CNN DualKalmanNet with relative-variance weighting, and the actual sparse Wi-Fi update times.

To avoid cherry-picking, the script restricts candidates to the **middle 50% of fusion improvements**, then chooses the walk with the most corridor turns. The selected walk is test walk 25; Wi-Fi-only gives **1.947 m**, while the weighted DualKalmanNet gives **1.754 m**.

I also restored the fuller KNN figure/protocol explanation instead of optimizing for the 6-page limit. The manuscript is now **7 pages**, which we can compact later if needed. I compiled it, rendered every page, checked the trajectory figure/table/equations for overlap, and verified the final crop changed only page 7 while pages 1–6 stayed identical.

The review tracker now has **R1–R9 all completed and the separate trajectory TODO completed as well**.

Everything is on `main` at:

`7fefcfd5f7b36a1dbf4d356f3e11b9b97195ec67`

Final outputs:

[Open the finalized paper](sandbox:/mnt/data/SURA_final_session_paper.pdf)

[Open the actual regenerated trajectory figure](sandbox:/mnt/data/SURA_representative_trajectory_current_architecture.png)

So for this session, the professor-review pass is effectively closed. The variance-weighting/full-Wi-Fi behavior is still something we can return to later as a research improvement rather than a review blocker.








*user : 1. the pdr prediction uses the entire test dataset in order to find the correction angle phi-H between device orientation and user orientation and uses the same frozen angle throughout all inferences, this might cause a data leak and is also totally impractical
2. again, the step length is calculated from the training set, whereas it should be a rolling average for both trainnig and inference, using the entire training set in most cases should be illegal, atleast i think so (do correct me if i am wrong)
3. there is a mention of magnetic anomaly A and delta A in section E, it should not be present there
4. the line contatining l-ref overlaps with the left column
5. ig the list of all 13 inputs can presented in a better mannerl, maybe like add breakline after each of the input to make it look like a list
6. also why are we using temporal wifi fixes as input to GRU what role does that play
7. i also want to understaand the cnn works lik ehow exaclty it outputs the variance for the prediction
8. also theres an em dash on page 2 last line
9. also i wanna understand the entire cnn architecture with evidence from code
10. also ORN\_Z was used a lot for pdr but the dataset itseld lacked ORN\_Z so how exaclty we used it 
11. I also wanna deeplu understand the data augmentation pipleline how exaclty was the data created to check for anyflaws



ok so for all of these issues jsut add them to a new todo section in the repo maybe name it proff read for IEEE comnets draft, u may simply or slzighty modify or merge these issues and get ready with like a few lines as a quick view for each for me in the chat


Added a new repo checklist at:

`paper/reviews/prof_read_ieee_comments_draft.md`

under commit `25785463e57e3dd03b3f04fb15f4f7a2366c9626`. I merged related observations where useful, but preserved every concern you raised.

Quick view before we have the proper discussion:

1. **(\phi_h) / heading leakage:** your exact suspicion is slightly different from what the current code does. The **active final KalmanNet experiment does not compute (\phi_h) from the test set at all**. In fact, it does not use `fit_heading_offset()` or real `Orn_z` at all. It creates heading from the synthetic ground-truth path tangent and then adds drift/noise.  The real problem is arguably bigger: **the paper describes a different PDR procedure than the experiment actually used**.

2. **Step length:** using the *training set only* to choose a step-length parameter would not normally be illegal or data leakage. Training-derived calibration is allowed. But it may be impractical for an unseen person if it requires ground-truth path length. More importantly, our active benchmark doesn't currently do what the paper says anyway: it simply hard-codes `STEP_LENGTH_M = 0.65`.  A rolling estimator is worth discussing, but at inference it cannot use true distance; it would need to infer step length from IMU and/or absolute sensor corrections.

3. **(A), (A_{\rm obs}), (\nabla A):** yes. They remain only in the sentence saying that the architecture *doesn't* use them.  Technically it isn't an architecture error anymore, but I agree it unnecessarily resurrects obsolete notation. We should simply remove it.

4. **(\ell_{\rm ref}) overlap:** straightforward formatting TODO. We should split the median definition and (\sigma_{\rm ref}^2=\exp(\ell_{\rm ref})) cleanly over multiple lines and visually inspect the rebuilt PDF.

5. **13 GRU inputs:** agreed. The implementation itself is actually very clear about the eight feature groups totaling 13 scalars.  The paper should probably display them one-per-line or as a compact aligned block rather than burying all thirteen dimensions in one sentence.

6. **Why temporal Wi-Fi difference (\Delta z_{\rm wifi})?** This one has a reasonable purpose. The GRU gets both the current innovation
   [
   z_{\rm wifi}-x_{\rm pred}
   ]
   and
   [
   z_{{\rm wifi},t}-z_{{\rm wifi},\text{previous}}.
   ]
   The first says *how much Wi-Fi disagrees with our current state*; the second says *how much the Wi-Fi system itself suddenly moved since its previous fix*. That helps identify unstable/jumpy Wi-Fi fixes and movement consistency. The code explicitly holds the last available Wi-Fi fix and only computes this feature on Wi-Fi-update steps.  Whether it is truly useful deserves an ablation later.

7. **How CNN variance works:** the network does **not directly calculate variance from some hand formula**. The shared CNN representation feeds a separate learned head `128 → 32 → 1` that predicts (\log\sigma^2).  During training,
   [
   \mathcal L=
   \frac12\frac{|z_{\rm mag}-x_{\rm true}|^2}{\sigma^2}
   +\frac12\log\sigma^2.
   ]
   So difficult examples can get higher uncertainty, reducing their squared-error penalty, but the (\log\sigma^2) term penalizes simply declaring everything uncertain. Importantly, it is **one scalar isotropic position variance**, not separate (x/y) variances.

8. **Page-2 em dash:** recorded as a final copy-edit item. We'll change it once the methodological edits are done because pagination may shift.

9. **Deep CNN architecture walkthrough:** recorded separately. The code is quite traceable: input `[B,T,4]` → Conv1D `4→32, k=7` → pooling → `32→64, k=5` → pooling → `64→128, k=3` → adaptive average pooling → shared 128-D feature → separate position and log-variance heads.  When we discuss this, I'll go layer-by-layer with actual tensor dimensions for (T=84) and explain what each operation means physically.

10. **`Orn_z`:** this is a real paper-consistency problem. The dataset isn't completely devoid of it, but the repository scan found **about 34.8% of all rows missing `Orn_z`**, including substantial missingness in IT Engineering.  More importantly, the active synthetic benchmark does not use `Orn_z` at all. Yet the paper currently explicitly says (\theta_t) comes from `Orn_z`.  We definitely need to resolve that statement.

11. **Data augmentation:** this may be the most important audit. The current path is surveyed nodes → (\varepsilon)-graph → random shortest path → continuous truth → heading **derived from truth** + simulated drift/noise → synthetic gait acceleration → PDR; Wi-Fi scans are sampled from stored fingerprints near the true position; magnetic features are sampled from an interpolated surveyed map and passed through the actual CNN.   One thing I specifically flagged: `setup_environment()` loads the full processed fingerprint DB before constructing the Wi-Fi pool and magnetic map.  We therefore need to investigate whether S9+ data contributes to synthetic sensor generation despite the paper claiming S9+ is held out. That could be a genuine leakage/generalization problem depending on exactly what claim we want to make.

The biggest three to discuss first are therefore **PDR heading**, **step-length methodology**, and the **full augmentation/leakage audit**. Those could affect the scientific claims; the rest are mostly explanation, justification, and presentation.





*user : I got t3wo mails from my prof 

Hi Jay and Utkarsh,
I want to introduce you to Prof. Nilesh Jha (cc'ed), who has recently joined the School of AI, IIT Delhi (he is also an alumnus of IITD, EE-2018 batch). He is an expert on Bayesian signal processing and machine learning.
He has kindly agreed to participate in our ongoing project on hybrid indoor localization.

Please find attached his comments on the current draft. Please address these comments.

We can meet sometime next week to finalize this paper and brainstorm some new ideas for future extensions.
  
Hi Jay and Utkarsh, 
Some more general comments on the draft paper: 
 
1. More accurate statements (citations, etc.) and of contributions. 
2. Mathematical definitions of certain variables can be improved. 
3. A better separation of measurements and preprocessing from the methodology and loss function design. This will improve readability. 
4. A better flow between subsections and modules and also a few grammar corrections. 
 
for now gimme an opinion on the following and just add all of this to the todo each and everthing then we will continue to fix all of them in another session


These emails are actually a **positive development** for the paper. Your supervisor is bringing in someone specifically strong in Bayesian signal processing/ML, says he has agreed to participate in the project, and is already talking about meeting next week to **finalize the paper and brainstorm extensions**. That reads much more like “the project is worth strengthening” than “the approach is fundamentally broken.”

Prof. Jha’s comments are also fairly coherent as a set. He is pushing for a paper that a signal-processing/estimation reader can audit mathematically, rather than one that reads mainly like a deep-learning systems paper. His annotations repeatedly ask for clearer signal definitions, cleaner state-space/KalmanNet mathematics, defensible novelty claims, better separation of preprocessing/method/losses, and less jargon. 

My view of the feedback is:

* **The most important comments are not grammar.** The scientific framing and methodology consistency come first. Our PDR/augmentation questions from your own proofreading are therefore now even more important.
* **The contribution needs reframing, not necessarily changing.** We should not sell “Wi-Fi + magnetic + IMU” itself as novel if related work already combines them. The contribution is more plausibly the *decoupled causal measurement-to-state fusion architecture, learned dual gains, missing-modality handling, CNN magnetic spatial fix, and uncertainty-aware correction*—but every part needs to survive our code audit.
* **The structure should probably change substantially.** I agree with the repeated request to separate: raw measurements → preprocessing/measurement extraction → proposed state-space fusion method → training objectives. Right now Wi-Fi preprocessing, model architecture, targets, loss, and inference are mixed together.
* **Prof. Jha’s mathematical comments should help the paper.** For example, explicitly contrasting analytical EKF gain computation with the GRU-produced KalmanNet gain will make the actual contribution much clearer than saying “fixed linear-Gaussian assumptions fail.”
* **Some annotations are already obsolete.** The attached PDF is the older V3: it still contains the scalar anomaly (A), (\nabla A), the old fusion equations, and the old 0.47/1.07 results.  We should not resurrect those just to answer comments. Instead, we apply the same demand for mathematical precision to our current (z_{\rm mag}), (\ell_{\rm mag}), (w_{\rm mag}), and dual-innovation architecture.
* **The “building-agnostic / generalized / robust” language needs restraint.** This is a very fair comment. The *architecture* can be reusable across buildings while the learned Wi-Fi/magnetic environment models still require building-specific survey/training. Those are different claims.
* **The introduction probably needs the biggest prose rewrite.** His point is basically: establish what earlier systems do, what precise failure remains, why that failure arises mathematically/architecturally, and then state what we change. That will make the paper look substantially more mature.

I added **everything** to the existing proofreading tracker, including all four general email comments and every substantive annotation I could extract from Prof. Jha’s PDF. The tracker now has your existing `P1-P10`, the general `G1-G4`, and detailed `J1-J18` items for his annotated review, including the smaller notation/grammar/figure comments and the obsolete anomaly comments marked as historical rather than silently dropped.

The update is on `main` in commit:

`38b8de0e4f3f4370547c217c4dc685d4b8f5ba1f`

The next-session order I recorded is deliberately **not** “start fixing sentences.” We should do:

**validity/code audit → contribution/literature framing → restructure the paper → mathematical definitions → figures/formatting → final grammar.**

That is the right order because if the PDR or synthetic-data claims change, rewriting those sections beautifully beforehand would just create more work.




P1–P3 are implemented, fully rerun, visually checked, and squash-merged into `main`.

The main commit is **`dcea63de0564a7c3ce622b174b098b61a3e5e0ee`**.

The main changes are:

* **P1 — heading/PDR:** the paper no longer claims the evaluated system uses MagWi `Orn_z` or a ground-truth-calibrated (\phi_h). It now defines a generic noisy heading observation (\hat\theta_t), which matches the actual experiment. The simulator generates it from path tangent + random-walk drift + white noise; the estimator never receives the true heading directly.
* **P2 — step length:** removed the unused training-set calibration story. The experiment now clearly states what the code actually does: a fixed nominal
  [
  L_s=0.65\text{ m}.
  ]
  The source also now calls it `NOMINAL_STEP_LENGTH_M`, so its role is less ambiguous.
* **P3 — augmentation/leakage:** rewrote the setup to explicitly separate **simulator truth** from **estimator inputs**. Wi-Fi and magnetic observations are generated conditional on the latent simulated position, but KalmanNet only sees PDR controls, learned sensor fixes, masks and magnetic uncertainty.
* Added fixed **train seed 1 / test seed 2** plus a SHA-256 trajectory-signature guard. The run fails if an identical binned target trajectory occurs in both sets.
* The full rerun found **0 exact train/test trajectory overlaps** in both regimes.
* Most importantly, we corrected the generalization claim. The fusion environment uses survey fingerprints from `A8, G7, LG G6, LG Q6, S8, S9+`, so the fusion numbers are now explicitly labelled **unseen-trajectory results within a fixed surveyed environment**, not unseen-device results.
* Genuine S9+ device-generalization claims remain only where S9+ is actually held out: the standalone Wi-Fi phone split and real static KNN experiment.
* `docs/architecture/pdr.md` has also been rewritten so it no longer contradicts the actual experiment.
* P1, P2 and P3 are marked resolved in the proofreading tracker.

Crucially, the full **250 train / 60 test / 150 epoch** rerun produced exactly the same headline numbers, so this was genuinely a methodology/provenance correction rather than an algorithm change:

| Regime         | Wi-Fi only | Weighted CNN Dual |
| -------------- | ---------: | ----------------: |
| Full Wi-Fi     |    0.473 m |           0.494 m |
| Degraded Wi-Fi |    1.533 m |       **1.154 m** |

I also rendered and checked the revised PDR/setup pages. They look fine. The previously identified **(\ell_{\rm ref}) overflow is still there**, deliberately untouched because that's P8 rather than P1–P3.

[Open the P1–P3 revised paper](sandbox:/mnt/data/SURA_P1_P3_methodology_preview.pdf)

Next in our agreed order is **P4 + P5**: remove the remaining obsolete anomaly notation, then decide/explain whether (\Delta z_{\rm wifi}) is a justified GRU feature.











Next I’d handle **P4 and P5 together**, but they are very different kinds of fixes.

### P4 — remove old magnetic-anomaly notation

This one is straightforward.

The current paper still says, after defining

[
\mathbf y_{\text{mag}}
======================

\mathbf z_{\text{mag}}-\mathbf x_{\text{pred}},
]

that the active path contains no (A_{\rm obs}), (A(\mathbf x)), or (\nabla A).

That sentence was useful while we were transitioning architectures, but now it just confuses the reader by introducing three symbols that have **no role anywhere in the proposed method**.

I suggest replacing the whole disclaimer with something simple like:

> The magnetic CNN output (\mathbf z_{\rm mag}) therefore serves directly as the magnetic position measurement used by the fusion network.

That's it. No code change and no experiment required.

---

## P5 — why do we give (\Delta \mathbf z_{\rm wifi}) to the GRU?

Current code computes

[
\Delta\mathbf z_{\rm wifi,t}
============================

## \mathbf z_{\rm wifi,t}

\mathbf z_{\rm wifi,prev},
]

where `previous_wifi` is specifically the **last available Wi-Fi fix**. If there is no Wi-Fi update at the current time, the feature is zero because it is multiplied by the Wi-Fi availability mask.

So there are two different Wi-Fi signals:

[
\mathbf y_{\rm wifi,t}
======================

\mathbf z_{\rm wifi,t}-\mathbf x_{\rm pred,t}
]

and

[
\Delta\mathbf z_{\rm wifi,t}
============================

\mathbf z_{\rm wifi,t}-\mathbf z_{\rm wifi,prev}.
]

They answer different questions.

**Innovation** asks:

> “Does Wi-Fi agree with where PDR/filter thinks I am?”

while **Wi-Fi delta** asks:

> “How different is this Wi-Fi estimate from the previous Wi-Fi estimate?”

For example, suppose:

[
\mathbf x_{\rm pred}=(10,5),
\qquad
\mathbf z_{\rm wifi}=(14,5).
]

The innovation is 4 m.

But suppose the previous Wi-Fi fix was:

Case A:

[
\mathbf z_{\rm wifi,prev}=(13.8,5.1)
]

so

[
\Delta z_{\rm wifi}\approx0.22\text{ m}.
]

The Wi-Fi subsystem has been consistently saying “around (14,5),” so perhaps **PDR is drifting**.

Compare Case B:

[
\mathbf z_{\rm wifi,prev}=(10.1,5.0),
]

giving

[
\Delta z_{\rm wifi}\approx3.9\text{ m}.
]

Now the current Wi-Fi measurement suddenly jumped. That might indicate a poor scan/outlier.

That's the intended reason for the feature.

### But I see a genuine weakness

A large (\Delta z_{\rm wifi}) **doesn't necessarily mean Wi-Fi is bad**.

If five seconds pass between Wi-Fi fixes and the person has genuinely walked 5 m, then

[
|\Delta z_{\rm wifi}|\approx5\text{ m}
]

is exactly what should happen.

This matters particularly in our degraded regime, where Wi-Fi arrives every **5 s**.

The GRU can partially distinguish this because it also receives:

* current PDR motion,
* previous filter update,
* current innovation,
* its recurrent hidden state,

and therefore can learn some motion history.

But I don't think we should write in the paper that (\Delta z_{\rm wifi}) “improves robustness” unless we have actual evidence that it does.

---

# What I suggest for P5

A **single clean ablation**.

Keep everything exactly identical and train:

**Current**

[
13\text{-input DualKalmanNet}
]

versus

**No Wi-Fi-delta**

[
11\text{-input DualKalmanNet}
]

by simply removing those two dimensions.

And because the Wi-Fi-only baseline also uses `wifi_delta`—its GRU currently has 9 inputs—we should simultaneously compare:

[
9\text{-input WiFiOnly}
\quad\text{vs}\quad
7\text{-input WiFiOnly}.
]

The Wi-Fi-only implementation confirms it uses the exact same temporal-difference feature.

Same:

* 250 training trajectories
* 60 test trajectories
* seeds 1/2
* 150 epochs
* full Wi-Fi
* degraded Wi-Fi
* all other parameters unchanged.

Then the decision becomes evidence-based:

**If removing it hurts noticeably:** keep (\Delta z_{\rm wifi}), report briefly that an ablation showed the temporal consistency feature was useful.

**If removing it changes essentially nothing:** remove it. The model gets simpler: **13 → 11 inputs**, and our paper becomes easier to explain.

**If removing it improves performance:** definitely remove it.

I slightly suspect it will matter more in full Wi-Fi than degraded Wi-Fi, because at 1 Hz consecutive fixes are reasonably interpretable as short-term consistency, whereas at 5 s the signal is mixed heavily with legitimate user displacement. But that's a hypothesis—we should let the experiment answer it.

So my recommendation for this pass is:

**P4:** remove the anomaly sentence immediately.

**P5:** run the small `with Δz_wifi` vs `without Δz_wifi` ablation before deciding whether the feature belongs in the final architecture.

I would **not change the paper's GRU input count until that ablation finishes**.







Done. **P4 and P5 are resolved and merged into `main`** at commit **`c9172d9701d3c3f92910ccea3817be6bf4f5a4f6`**.

For **P4**, I removed the obsolete (A_{\rm obs}), (A(\mathbf x)), and (\nabla A) discussion entirely. The paper now simply says that the magnetic CNN produces (\mathbf z_{\rm mag}), and this serves directly as the magnetic position measurement for KalmanNet.

For **P5**, we ran the full controlled ablation before deciding. Same 250/60 walks, 150 epochs and same protocol; model initialization and minibatch-shuffle seeds were reset for each paired comparison.

| Model / regime         | With (\Delta z_{\rm wifi}) | Without | no-delta − with-delta |
| ---------------------- | -------------------------: | ------: | --------------------: |
| Wi-Fi-only, full       |                     0.4728 |  0.4678 |             −0.0051 m |
| **CNN Dual, full**     |                 **0.4664** |  0.4945 |         **+0.0280 m** |
| Wi-Fi-only, degraded   |                     1.5708 |  1.5480 |             −0.0228 m |
| **CNN Dual, degraded** |                 **1.1107** |  1.2025 |         **+0.0918 m** |

For the final **CNN DualKalmanNet**, removing Wi-Fi delta makes things worse. In full Wi-Fi the paired 95% CI is **[+0.0016, +0.0545] m**, so there is reasonably clear evidence for retaining it. Degraded Wi-Fi also trends strongly toward keeping it, although its CI ([-0.0224,+0.2060]) crosses zero. The Wi-Fi-only model shows no clear effect.

So we **keep the 13-input GRU**.

I added the actual mathematical definition:

[
\Delta\mathbf z_{{wifi},t}
==========================

## \mathbf z_{{wifi},t}

\mathbf z_{{wifi},t^-},
]

where (t^-) is the **most recent time at which a Wi-Fi fix was available**. If there is no new Wi-Fi fix, the feature is zero.

And the explanation in the paper is deliberately restrained: it provides a **short-term Wi-Fi consistency cue**. The GRU can jointly look at:

[
\underbrace{\mathbf z_{\rm wifi}-\mathbf x_{\rm pred}}_{\text{Wi-Fi vs current prediction}}
]

and

[
\underbrace{\mathbf z_{{wifi},t}-\mathbf z_{{wifi},t^-}}_{\text{current Wi-Fi vs previous Wi-Fi}}
]

along with PDR and its recurrent history. We do **not** claim that a large Wi-Fi delta automatically means an outlier, because during actual movement a large delta may be correct.

I preserved the full reproducible experiment as:

* `benchmarks/ablate_wifi_delta.py`
* `benchmarks/wifi_delta_ablation/README.md`
* `benchmarks/wifi_delta_ablation/metrics.json`

The README also explicitly warns that the ablation's absolute 0.466/1.111 values **do not replace our headline 0.494/1.154 results**. Those changed because the ablation deliberately resets model initialization before every paired training; the scientifically relevant quantities here are the **paired differences**.

P4 and P5 are checked off in the proofreading tracker.

I compiled and rendered the revised 7-page paper. Pages 1–3 were pixel-identical to the previously checked version; I inspected the reflowed pages 4–7. **No new overlaps/clipping.** The already-known (\ell_{\rm ref}) overflow is still present and remains P8—we'll fix that separately.

[Current P4/P5 paper preview](sandbox:/mnt/data/p4p5/main.pdf)

Next I'd do **P6 + P7 together**, because they're really one topic: understand precisely how the magnetic CNN produces both **position and variance**, then make sure the paper explains the architecture correctly from the actual code.








P6 and P7 are done, audited against the actual code, visually checked, and merged into `main` through **PR #8**. Current `main` is:

`6a423ce8679d5e90c83e9b3f417c09e1515dea3c`

The important part is that this audit found a couple of things that were **more scientifically significant than the original proofreading comments**.

### How the magnetic CNN actually works

The CNN does **not** take one magnetic reading and somehow output a position. Its input is an 84-frame sequence with four magnetic features per frame:

[
\mathbf M_t\in\mathbb R^{84\times4},
\qquad
[\text{magN},\text{magV},\text{magH},\text{dip}].
]

For (T=84), the exact code path is

[
[B,84,4]
\rightarrow[B,4,84]
\rightarrow[B,32,84]
\rightarrow[B,32,42]
\rightarrow[B,64,42]
\rightarrow[B,64,21]
\rightarrow[B,128,21]
\rightarrow[B,128].
]

The three convolutions are respectively (4\to32) with kernel 7, (32\to64) with kernel 5, and (64\to128) with kernel 3. The first two are followed by max-pooling, hence (84\to42\to21). Adaptive average pooling then collapses those 21 temporal locations into one 128-dimensional representation.

A useful intuition is that those 128 numbers are the CNN's summary of **“what magnetic pattern did I experience over the last five seconds?”**

That shared 128-D representation is sent into two independent heads:

[
128\rightarrow64\rightarrow2
]

for the magnetic position

[
\mathbf z_{\rm mag}=(x_{\rm mag},y_{\rm mag}),
]

and

[
128\rightarrow32\rightarrow1
]

for the scalar uncertainty score

[
\ell_{\rm mag}.
]

So the uncertainty is **not calculated afterward from the predicted coordinate**. The network looks at the same magnetic sequence that produced the coordinate and learns a second function saying roughly, *“How difficult/reliable does a prediction from a sequence like this tend to be?”*

### The important correction about “variance”

This was the biggest P6 finding.

Previously we were calling

[
\ell_{\rm mag}=\log\sigma_{\rm mag}^2
]

a predicted variance and calling its training loss a heteroscedastic Gaussian NLL.

That interpretation was **too strong**.

The actual code trains with

[
\mathcal L
==========

\frac12
\frac{|\mathbf z_{\rm mag}-\mathbf z_{\rm true}|^2}{q_{\rm mag}}
+
\frac12\ell_{\rm mag},
\qquad
q_{\rm mag}=e^{\ell_{\rm mag}}.
]

The behavior is easy to understand. Suppose the CNN makes a large position error. Increasing (q_{\rm mag}) reduces

[
\frac{|e|^2}{q_{\rm mag}},
]

so the uncertainty head is rewarded for saying “this was difficult.” But if it simply predicts an enormous uncertainty for everything, the

[
+\frac12\ell_{\rm mag}
]

term penalizes it.

For intuition, if (q) were optimized independently for a single fixed error (E=|e|^2),

[
L(q)=\frac{E}{2q}+\frac12\log q,
]

then

[
\frac{dL}{dq}=0
\quad\Rightarrow\quad
q=E.
]

So the quantity naturally behaves like a learned scale related to **squared radial localization error**.

But here is the mathematical problem: if we really had a 2-D isotropic Gaussian

[
\mathbf e\sim\mathcal N(\mathbf0,q\mathbf I_2),
]

its NLL, ignoring constants, would be

[
\frac{|\mathbf e|^2}{2q}
+\log q,
]

**not**

[
\frac{|\mathbf e|^2}{2q}
+\frac12\log q.
]

That factor matters. Therefore our current checkpoint's output is **not properly justified as a calibrated 2-D Cartesian variance or covariance**.

I've consequently changed the paper terminology to:

> **learned scalar log-uncertainty score** (\ell_{\rm mag})

with

[
q_{\rm mag}=e^{\ell_{\rm mag}}
]

being its positive uncertainty scale.

We don't pretend that

[
q_{\rm mag}\mathbf I
]

is a genuine covariance matrix.

This also explains why our previous calibration results looked strange: predicted scales were much larger than actual errors, yet **higher predicted uncertainty reliably corresponded to harder examples**. That is exactly what we need for ranking confidence, even if it is not calibrated probability.

So our final fusion remains completely valid in the narrower form:

[
\ell_{\rm ref}
==============

\operatorname{median}*{\rm training}
\ell*{\rm mag},
]

and

[
w_{\rm mag,t}
=============

\frac{1}
{1+\exp(\ell_{\rm mag,t}-\ell_{\rm ref})}.
]

We're only asking:

> “Is this magnetic fix more or less uncertain than a typical training magnetic fix?”

We are **not** asking the value to be an absolute physical covariance. That interpretation is now consistent everywhere in the manuscript, figures, code comments and calibration documentation.

### Another major thing we discovered: what data the CNN is actually trained on

This also needed correction.

The current CNN is **not trained by taking continuous real MagWi walks and sliding 84-frame windows over them**.

The actual pipeline is:

raw static MagWi magnetic recordings
→ compute `magN`, `magV`, `magH`, `dip`
→ average those features over each static visit
→ subtract each phone's mean feature value
→ aggregate by surveyed spatial node
→ interpolate a four-channel magnetic map
→ generate map-constrained paths
→ bilinearly sample that magnetic map along those paths at 16.7 Hz
→ add measured magnetic noise
→ extract 84-frame windows
→ CNN predicts the position at the final frame.

That is now stated explicitly.

There was also a smaller preprocessing mismatch. The paper said gravity direction came from a **low-pass-filtered accelerometer**, but the actual static fingerprint builder uses

[
\hat{\mathbf a}_t=
\frac{\mathbf a_t}{|\mathbf a_t|}
]

for each individual sample and then computes

[
m_V=\mathbf m_t^T\hat{\mathbf a}_t,
]

[
m_H=\sqrt{\max(m_N^2-m_V^2,0)},
]

[
\delta=\operatorname{atan2}(m_V,m_H).
]

The visit is averaged afterward. The paper now matches that implementation.

### One new issue: P11

The audit uncovered something I don't think we should ignore.

During magnetic-map construction, we perform **per-phone centering**:

[
m' = m-\mu_{\rm phone}.
]

That makes sense for reducing phone-specific magnetometer offsets in the survey database.

But the synthetic fusion experiment starts directly from this already-centered magnetic map.

In an actual deployment with a brand-new phone, where does

[
\mu_{\rm phone}
]

come from?

We currently **do not have a causal online procedure for determining it**.

So an unseen handset cannot simply produce raw `magN/magV/magH/dip` and necessarily be assumed to lie in the same centered feature domain as the CNN.

I've added this as **P11** rather than concealing it.

This doesn't invalidate the current synthetic fusion result. It changes its scope:

> We have demonstrated temporal fusion inside a surveyed, normalized magnetic environment.

We have **not yet demonstrated plug-and-play magnetic localization on an entirely unseen handset**.

That will matter later when we clean up the contribution/generalization claims.

### What changed experimentally?

Nothing.

I deliberately **did not retrain the CNN**. Correcting its loss to a mathematically proper 2-D probabilistic model would mean:

CNN retraining → uncertainty recalibration → KalmanNet retraining → full/degraded rerun → KNN matched baseline rerun → figures/results potentially changing.

That's a legitimate future experiment, not something to slip into a proofreading pass.

So our headline values remain **0.494 m full / 1.154 m degraded**. Only their uncertainty interpretation has become more accurate.

P6 and P7 are now checked off, while P11 is explicitly open.

I also regenerated the existing CDF/trajectory PNGs **from their already-saved predictions**, with no model inference or training, so their legends now say “relative uncertainty” rather than the misleading “relative variance.”

The paper is currently **8 pages**, fully rendered and inspected. The one deliberate known visual defect is still **P8: the (\ell_{\rm ref})/(q_{\rm ref}) equation crossing the column boundary**, which is exactly what we should fix next.

[Open the current P6/P7 paper](sandbox:/mnt/data/SURA_P6_P7_magnetic_CNN_audit.pdf)





Yep — it is now **7 pages**, with the scientific content intact. I avoided shrinking the body font or cutting methodology/results; most of the reduction came from removing repeated prose, tightening float spacing, slightly reducing the CDF figure footprint, and making the trajectory plot a single-column figure. I rendered and inspected all seven pages, including equations, tables, captions, column boundaries, references, and both result figures. I also render-diffed it against the previous 8-page version.

It is merged through **PR #9**, and `main` is now at `80a9344ef3c1ab640e162e72c78da90fcd6e94f1`.

[**Open the new 7-page paper**](sandbox:/mnt/data/SURA_7_page_compressed.pdf)

### Gist of everything we have changed / learned so far

* **Final architecture is now consistently CNN-output-based.** The old scalar magnetic-anomaly formulation is gone from the active paper. The magnetic CNN directly supplies (\mathbf z_{\rm mag}), while Wi-Fi supplies (\mathbf z_{\rm wifi}); PDR predicts motion and DualKalmanNet learns independent (2\times2) gains.

* **Magnetic uncertainty was corrected conceptually.** We discovered that the historical loss
  [
  \frac12\frac{|e|^2}{e^{\ell}}+\frac12\ell
  ]
  is **not exactly the NLL of a 2-D isotropic Gaussian**. Therefore the paper no longer claims that (e^{\ell_{\rm mag}}) is a calibrated Cartesian variance/covariance. It is now correctly called a **learned scalar log-uncertainty score**.

* **The uncertainty is still useful.** Calibration showed that its absolute scale is conservative, but its ordering is informative: high predicted uncertainty tends to correspond to larger actual magnetic errors. Therefore fusion only uses the **relative** quantity
  [
  w_{\rm mag}
  ===========

  \frac{1}{1+\exp(\ell_{\rm mag}-\ell_{\rm ref})},
  ]
  with (\ell_{\rm ref}) computed from training trajectories.

* **The magnetic CNN is now documented exactly from code.** Its path is
  [
  [B,84,4]\rightarrow[B,4,84]
  \rightarrow[B,32,84]\rightarrow[B,32,42]
  \rightarrow[B,64,42]\rightarrow[B,64,21]
  \rightarrow[B,128,21]\rightarrow[B,128],
  ]
  followed by a (128!\to64!\to2) position head and a (128!\to32!\to1) uncertainty head. The input channels are `magN`, `magV`, `magH`, and `dip`.

* **We corrected what magnetic data actually trains the CNN.** It is not trained by sliding windows over raw continuous MagWi walks. Static survey recordings are converted into gravity-referenced features, aggregated into a spatial magnetic map, and then **synthetic/map-constrained paths** are sampled from that map to create causal 84-frame sequences.

* **We corrected the magnetic preprocessing description.** The actual static fingerprint pipeline uses the **instantaneous normalized acceleration vector** as its gravity-direction proxy and subsequently averages over the static visit; the old manuscript's low-pass-accelerometer description did not match that path.

* **PDR was audited.** The final experiment does **not** use MagWi `Orn_z` or a ground-truth-fitted heading offset. For the synthetic benchmark, the simulator takes the true path tangent and corrupts it with drift and white noise; the estimator only receives that noisy heading observation. Likewise, stride length is a fixed (0.65) m rather than something calibrated from test or training trajectory ground truth.

* **The simulator/estimator boundary is now explicit.** Ground truth is available to the simulator for generating noisy observations and to the loss/evaluation function, but it is not fed into KalmanNet. Fusion results are therefore described as **generalization to independent synthetic/map-constrained trajectories inside a surveyed environment**, not continuous real-world GT and not held-out-device fusion performance.

* **Device-generalization claims were separated.** The standalone Wi-Fi experiment and real static KNN test can genuinely hold out S9+. The temporal fusion environment uses survey-derived resources from the available fingerprint database, so its 60-walk result is explicitly **trajectory-generalization**, not unseen-phone generalization.

* **(\Delta\mathbf z_{\rm wifi}) was questioned and experimentally tested.** We kept it after a paired ablation. Removing it worsened CNN DualKalmanNet mean error by about (0.028) m in full Wi-Fi, with a positive paired 95% CI, and by about (0.092) m in degraded Wi-Fi, although the latter CI crossed zero. The paper now describes it conservatively as a **short-term Wi-Fi consistency cue**, not an automatic outlier detector.

* **Relative uncertainty weighting improved the final fusion.** Full Wi-Fi remains approximately parity: Wi-Fi-only (0.473) m mean versus weighted Dual (0.494) m, though the median slightly improves (0.449\to0.437) m. Under degraded Wi-Fi, Wi-Fi-only is (1.533) m and the final weighted Dual is (1.154) m — a **24.7% reduction**. Weighting also reduced the unweighted Dual's degraded P90 from (2.064) m to (1.612) m.

* **We added a classical KNN baseline rather than attributing one incorrectly to MagWi.** The MagWi paper did not contain an identifiable published KNN localization implementation/result, so the manuscript clearly says these are **our reproducible KNN baselines on MagWi data**, not Ashraf et al.'s reported KNN numbers.

* **The real static KNN result exposed cross-device magnetic sensitivity.** Held-out S9+ Wi-Fi-only KNN gets about (3.31) m mean, magnetic-only about (17.54) m, and naive Wi-Fi+magnetic concatenation about (7.46) m. So just appending static magnetic features actually hurts cross-device performance.

* **The matched synthetic KNN baseline also matters.** Non-temporal Wi-Fi+magnetic KNN gets (0.802) m in full Wi-Fi and (2.606) m degraded, substantially behind KalmanNet. That supports the claim that the benefit is not simply “we have two absolute sensor estimates”; **PDR propagation and recurrent temporal context matter**, particularly during sparse Wi-Fi.

* **We restored a scientifically valid trajectory figure.** It uses actual saved experiment predictions, not the earlier generated illustration. The selected walk is a central-performance example: restricted to the interquartile range of fusion improvements and then chosen for high accumulated heading change so several corridor turns are visible. It is explicitly labelled synthetic/map-constrained.

* **A new real deployment gap was uncovered — P11.** Magnetic-map construction subtracts each phone's mean magnetic feature value. Synthetic fusion then samples from that already-centered domain. A brand-new physical handset currently has **no implemented causal procedure for estimating the normalization required to enter that same domain**. This is probably the most important unresolved technical observation from the P6/P7 audit.

* **The known (\ell_{\rm ref}) overflow is now fixed as part of this compression pass.** Equation (18) has been broken into a clean aligned form and visually verified inside the column, so **P8 is resolved**.

* **The 8→7 page compression did not change any experiment.** No model was retrained, no result arrays changed, and no metric changed. I shortened duplicated KNN/result prose and the conclusion, tightened float whitespace, reduced the three-CDF panel footprint slightly, and moved the representative trajectory to a compact single-column layout. PR #9 records exactly those changes.

At this point the immediate proofreading queue is **P9** — make the 13 GRU inputs much easier to audit — and **P10**, the final punctuation/copy-edit item. The technically more consequential open issue is **P11**, the real-phone magnetic centering/deployment question. Beyond those, the new faculty-review round asks for a broader introduction/citation/notation/flow cleanup: more precise contribution claims, clearer mathematical definitions, better separation of preprocessing vs. methodology/loss, and a less jargon-heavy explanation of exactly what prior architectures fail to handle.

So we're in a pretty good position now: **7-page clean working draft, core experimental methodology audited, major misleading claims corrected, and the remaining issues are clearly isolated.**









Finished.

**P9 + P10 are merged into `main` through PR #10**, squash commit:

`fe237679bd3fe6f6c520534d61b72e0a913c9bef`

P9 now shows the GRU inputs as an explicit 8-row table totaling **13 dimensions**, and the audit also made two definitions more code-accurate:

[
\Delta \mathbf{x}_{t-1}
=======================

\mathbf{x}*{t-1}-\mathbf{x}*{t-2}
]

and the actual GRU magnetic-confidence input

[
c_{\mathrm{mag},t}
==================

m_{\mathrm{mag}},
\operatorname{clip}(\ell_{\mathrm{mag},t},-6,8).
]

So we're no longer implying that the raw (\ell_{\rm mag}) goes directly into the GRU.

**P10 is closed.** After the paper's re-pagination, there is no manuscript-authored em dash at the old page-2 location. The `Abstract--` / `Index Terms--` style punctuation comes from IEEEtran itself, so we correctly left that alone. Both P9 and P10 are checked off in the tracker.

I compiled the complete manuscript to **7 pages**, rendered all seven at 180 DPI, and inspected them. The first draft exposed a ~7 pt overflow in the new auxiliary equation, which I fixed by splitting it across two aligned lines. The corrected version has no new P9/P10 overlap or clipping. The only LaTeX overfull warning left is the tiny pre-existing ~1.27 pt `Dropout(0.3)` text inside the Wi-Fi architecture figure, unrelated to this pass.

And regarding the activity that looked unfinished: **repo-side it is definitely finished now**. The earlier problematic validation run has status `completed`, and the final validation run also has status `completed` with `success`.   I also removed all three temporary validation files before merging; PR #10's final diff contained only `paper/main.tex` and the review tracker, and the temporary workflow does not exist on `main`.

If the ChatGPT interface still visually shows some previous activity as running, that would be stale UI state rather than an active GitHub job.

[**Current P9/P10 7-page paper**](sandbox:/mnt/data/SURA_P9_P10_final.pdf)

Next we can move into the **faculty-review changes**. I think the right next block is **G1/J2/J3/J4/J5/J6 together**: audit the introduction, citations, contribution claims and terminology before restructuring the rest of the methodology.











Yes. For the next pass I’d group **G1 + J2–J6** together, because they all boil down to one question:

> **What exactly is our paper claiming to solve, and what exactly is the contribution?**

The current introduction is still one of the weaker parts of the paper. It contains several statements that are broader than what our experiments establish: “generalized,” “optimally fuses,” “robustness,” “real-world deployments,” “standard sequence models frequently memorize routes,” and “learned Bayesian filter.”

I would change the framing substantially, but without changing the method.

### 1. Stop presenting sensor combination itself as the novelty

This is probably Prof. Jha’s most important point.

Our own bibliography already contains:

* **Bolat & Akcakoca (2017):** explicitly *Wi-Fi + magnetic field + inertial navigation*.
* **DeepPositioning (2017):** explicitly Wi-Fi + magnetic fusion.
* **Yu et al. (2023):** neural Wi-Fi/PDR fusion.

So we absolutely should **not** imply:

> “Existing systems use individual modalities, whereas we combine Wi-Fi, magnetic and IMU.”

That would be easy for a reviewer to attack.

Instead, I’d frame the problem as:

> Multiple indoor-localization systems already combine complementary smartphone sensors. Our focus is the **fusion architecture**: how to combine sparse absolute Wi-Fi fixes, sequence-based magnetic fixes, and drifting PDR causally when sensor reliability and availability vary over time.

That is much safer and much closer to what we actually built.

---

### 2. Replace the current “three limitations of deep learning” story

Right now the abstract/introduction says existing approaches suffer from:

* trajectory memorization,
* body-frame magnetic direction dependency,
* non-causal look-ahead bias.

I don't like bundling those together as if we established that these are universal shortcomings of current indoor-localization methods.

**Non-causal Bi-LSTM:** mathematically defensible. A bidirectional model uses future sequence context, so it cannot operate strictly online without modification.

**Body-frame magnetic sensitivity:** also defensible as a motivation for our gravity-referenced features, but we should explain the coordinate issue rather than saying models simply “suffer from heading bias.”

**Trajectory memorization:** this is the weakest claim. We do not currently have an experiment proving that the cited systems memorize trajectories, and the citations in the bibliography do not by themselves establish it.

So I'd remove “trajectory memorization” as a headline claim unless the literature audit finds strong explicit evidence.

Instead, the introduction can make a much cleaner argument:

[
\mathbf x_{t|t-1}
=================

\mathbf x_{t-1}+\mathbf u_t
]

gives a causal PDR prediction, but it drifts.

Then absolute measurements arrive:

[
\mathbf z_{\rm wifi,t},
\qquad
\mathbf z_{\rm mag,t}.
]

Their useful corrections are their innovations:

[
\mathbf y_{\rm wifi,t}
======================

\mathbf z_{\rm wifi,t}-\mathbf x_{t|t-1},
]

[
\mathbf y_{\rm mag,t}
=====================

\mathbf z_{\rm mag,t}-\mathbf x_{t|t-1}.
]

The actual practical problem is:

> **How strongly should each innovation modify the state when measurement quality and availability change with time?**

That directly leads to our DualKalmanNet. Much better flow.

---

### 3. Make the contribution narrower and stronger

Currently we say things such as:

> “generalized, decoupled, strictly causal”

and

> “generic models for any number of APs and spatial nodes, allowing broad applicability.”

I would remove **“generalized.”**

Supporting arbitrary (N) and (M) is useful implementation-wise, but almost every neural architecture can change its input/output layer sizes. It isn't a convincing research contribution.

I'd summarize our contribution around **three actual architectural ideas**:

**A. Decoupled spatial estimation and temporal tracking.**
Wi-Fi and magnetic networks independently convert sensor observations into Cartesian position measurements. KalmanNet then performs temporal fusion instead of learning the complete sensor-to-trajectory mapping end to end.

**B. Dual-innovation learned state update.**
Wi-Fi and magnetic measurements produce separate Cartesian innovations and the GRU predicts independent (2\times2) gains:

[
\mathbf K_{\rm wifi,t},
\qquad
\mathbf K_{\rm mag,t}.
]

Availability masks make the correction explicitly disappear when a modality is absent.

**C. Relative magnetic confidence.**
The magnetic CNN provides (\ell_{\rm mag}), from which we construct a training-normalized relative confidence weight before applying the magnetic correction.

Then the experimental statement:

> Under degraded Wi-Fi, this architecture reduces mean error from 1.533 m for Wi-Fi-only KalmanNet to 1.154 m, a 24.7% reduction on the matched map-constrained trajectory protocol.

That is a **specific contribution + specific evidence** rather than “generalized robust hybrid localization.”

I would not say **“first”**, **“novel”**, or **“state of the art”** until we have done the literature audit.

---

### 4. J3 — the citations need a real audit

This will require more than grammar.

For every introduction claim, we should ask:

> What exact fact does this citation establish?

For example, the first paragraph currently cites `horus` and `hybridwifi` together when saying smartphones rely primarily on Wi-Fi, geomagnetic data and IMUs. But Horus is fundamentally a WLAN localization paper, whereas `hybridwifi` is explicitly the three-modality system.

That's exactly the kind of grouped citation Prof. Jha objected to.

Similarly, this sentence:

> “standard sequence models ... frequently memorize specific spatial routes”

currently sits behind citations to DeepPositioning, MINLOC and a Bi-LSTM magnetic paper, but none of those titles alone establish a **trajectory-memorization failure**.

So during implementation I would actually open the cited papers and verify:

**claim → paper section/evidence → retain/reword/remove.**

We will probably end up with fewer but more meaningful citations.

---

### 5. J4/J5 — don't argue that previous work fails because it was “initialized badly”

Prof. Jha was right to be suspicious of that style of argument.

The paper should not imply:

> previous architecture fails → our initialization/fusion fixes it.

Because then a reviewer can simply ask whether a better initialization would solve the problem.

Our distinction is more structural.

The current architecture has:

[
\mathbf x_{\rm pred}
]

followed by two explicit innovations,

[
\mathbf y_{\rm wifi},
\quad
\mathbf y_{\rm mag},
]

two learned matrix gains,

[
\mathbf K_{\rm wifi},
\quad
\mathbf K_{\rm mag},
]

availability masks,

and explicit relative magnetic confidence.

So the argument should be:

> Our aim is not to replace a poorly initialized conventional estimator; it is to represent **measurement-dependent and history-dependent correction strength** with separate learned gains while retaining an explicit causal state-space prediction/correction structure.

That is a much more defensible distinction.

And our degraded-Wi-Fi experiment is the evidence that this mechanism is useful.

---

### 6. J6 — I would remove “Bayesian filter”

The current architecture section begins:

> “Our proposed architecture functions as a modular, learned Bayesian filter...”

I would change that.

We're not explicitly maintaining

[
p(\mathbf x_t\mid\mathbf z_{1:t})
]

or propagating a posterior probability distribution/covariance in the conventional Bayesian-filter sense.

KalmanNet is inspired by state-space/Kalman filtering, but our network predicts gains directly.

So I'd use:

> **“learned state-space estimator”**

or:

> **“Kalman-inspired learned state-space estimator.”**

That answers Prof. Jha's terminology concern cleanly.

Likewise:

**“hybrid indoor localization”** → define once as *fusion of heterogeneous smartphone sensing modalities* or simply stop overusing “hybrid.”

**“Neural-Kalman”** → use it mainly as architectural description/title.

**“KalmanNet”** → use when referring specifically to the Revach et al. framework.

**“DualKalmanNet”** → our implementation.

I would also remove **Extended Kalman Filter** from the keywords unless we actually discuss/evaluate an EKF. Currently it appears as a keyword despite the final system not being an EKF.

---

## What I think the new introduction should look like

Rather than the present three fairly broad paragraphs, I'd make the logic:

**Paragraph 1 — indoor localization problem + complementary sensors.**
Wi-Fi absolute but sparse/noisy; PDR high-rate but drifting; magnetic structure provides another spatial cue. Mention that prior systems already combine these sensors.

**Paragraph 2 — precise fusion problem.**
The difficulty is not simply obtaining more modalities; it is assigning appropriate correction strength to asynchronous/missing measurements whose reliability varies over time while remaining causal.

**Paragraph 3 — related approaches and exact gap.**
Briefly distinguish direct neural regressors, sequence models and traditional/model-based fusion. Do **not** say all of them fail. Say what design choice motivates our work.

**Paragraph 4 — our method + contributions.**
Decoupled environment models → Cartesian measurements → PDR prediction → dual learned innovations/gains → masks + magnetic relative confidence.

Then one sentence clearly limiting evaluation:

> Temporal fusion is evaluated on independent map-constrained trajectories generated from a fixed surveyed environment; separate experiments evaluate Wi-Fi/static-fingerprint device generalization.

That last sentence may even belong at the end of the contribution paragraph because it makes the claims immediately honest.

---

### What I would **not** touch in this pass

I would **not change any network architecture or rerun experiments** for G1/J2–J6.

This should be a scientific-writing/literature pass:

**verify literature → rewrite intro/abstract/contribution → terminology cleanup → citation corrections.**

After that, I'd tackle **G2/G3/G4 + the remaining J comments**, which is the larger structural reorganization of measurements/preprocessing vs models/losses and the Fig. 1 redesign.

So before implementation, the main decision I'd want from you is whether you're happy with this narrower contribution statement:

> **The contribution is not combining Wi-Fi, magnetic and IMU. It is a causal, decoupled state-space fusion architecture that converts Wi-Fi and magnetic sensing into independent Cartesian innovations, learns separate context-dependent matrix gains, handles missing modalities explicitly, and uses learned magnetic relative confidence to control magnetic corrections.**

I think that's much stronger scientifically than the current “generalized hybrid localization” framing.
