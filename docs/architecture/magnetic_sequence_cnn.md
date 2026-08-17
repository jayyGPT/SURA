# Magnetic sequence CNN: code-backed architecture and uncertainty semantics

This note documents the **active repository implementation** used by the current paper. It is explicit about which quantities come from raw MagWi files, which are survey-derived, which are synthetically generated for model training/evaluation, and what the CNN's scalar uncertainty head can and cannot be interpreted as.

Active sources:

- `tools/fingerprint_builder.py` - raw static-recording feature extraction.
- `train/train_magnetic_sequence.py` - magnetic survey-map construction, synthetic sequence generation, and CNN training.
- `models/magnetic_sequence_cnn.py` - neural-network architecture and training objective.
- `train/kalmannet_wifiheatmap_magneticCNN_pdr.py` - causal use of 84-frame magnetic windows in the final fusion benchmark.

## 1. Survey magnetic features

For every row of a raw static magnetic recording, let

- `m = [Mag_x, Mag_y, Mag_z]`,
- `a = [Acc_x, Acc_y, Acc_z]`,
- `m_N = ||m||`, and
- `a_hat = a / ||a||` whenever `||a|| != 0`.

The processed database uses four gravity-referenced features:

```text
magN = ||m||
magV = m dot a_hat
magH = sqrt(max(magN^2 - magV^2, 0))
dip  = atan2(magV, magH)
```

`tools/fingerprint_builder.py` computes these values row by row and stores their **mean and standard deviation over each static visit**. The magnetic-map trainer therefore consumes visit-level columns `magN_mean`, `magV_mean`, `magH_mean`, and `dip_mean`.

Important implementation detail: the current fingerprint builder uses the **instantaneous normalized acceleration vector** as the gravity-direction proxy. It does not low-pass the accelerometer before computing these processed static features. Averaging over the static visit subsequently reduces some row-level noise, but the low-pass formulation previously written in the manuscript did not match this code path.

## 2. Survey map used to generate magnetic sequences

`train/train_magnetic_sequence.py` does not feed raw continuous MagWi magnetic recordings directly to the CNN. It first builds a four-channel spatial map from the processed static survey database.

For each feature independently:

1. take the visit-level feature mean;
2. subtract that phone's mean value for the same feature (`value - phone_mean`);
3. average the centered values at each surveyed `(x,y)` node;
4. interpolate the node means onto a 1 m Cartesian grid using linear interpolation with nearest-neighbour filling outside the linear hull;
5. estimate a channel noise scale as the median within-node standard deviation of the centered survey values.

The resulting map has four channels corresponding to `magN`, `magV`, `magH`, and `dip`.

The per-phone centering suppresses phone-dependent offsets in the survey map. The current fusion benchmark uses the available surveyed devices to construct this environment resource; it is therefore not presented as a held-out-phone magnetic generalization experiment.

## 3. How the 84-frame training sequences are generated

The standalone magnetic CNN is trained on **survey-derived synthetic/map-constrained sequences**:

1. Build an epsilon-neighbour corridor graph from surveyed coordinates (`epsilon = 1.6 m`).
2. Sample random graph paths of at least 30 m.
3. Interpolate motion along each path at 16.7 Hz with speed sampled from 1.0-1.35 m/s.
4. Bilinearly sample the four-channel magnetic map along the path.
5. Add independent Gaussian noise using the map-estimated per-channel noise scales.
6. Extract fixed-length windows. The current default is `T = 84` frames (about 5.0 s).
7. Use the Cartesian position at the **last frame of the window** as the regression target.

The current training configuration uses 300 generated training walks (`seed=42`) and 60 separately generated test walks (`seed=200`). Training windows use a 5-frame stride and test windows a 10-frame stride.

Thus an input to the CNN is

```text
M_t shape = [batch, 84, 4]
channels  = [magN, magV, magH, dip]
```

In the final fusion benchmark, magnetic fixes are also computed causally: a prediction is emitted only when the 84-frame window ending at the current fusion endpoint is available. No future magnetic frame is included in that window.

## 4. Exact CNN architecture for T = 84

`MagSequenceMatcher` first transposes `[B,T,C]` to PyTorch Conv1D layout `[B,C,T]`.

| Stage | Operation | Output for T=84 |
|---|---|---|
| Input | transpose `[B,84,4] -> [B,4,84]` | `[B,4,84]` |
| Encoder 1 | Conv1D `4 -> 32`, kernel 7, padding 3 + BatchNorm + ReLU | `[B,32,84]` |
| Pool 1 | MaxPool1D(2) | `[B,32,42]` |
| Encoder 2 | Conv1D `32 -> 64`, kernel 5, padding 2 + BatchNorm + ReLU | `[B,64,42]` |
| Pool 2 | MaxPool1D(2) | `[B,64,21]` |
| Encoder 3 | Conv1D `64 -> 128`, kernel 3, padding 1 + BatchNorm + ReLU | `[B,128,21]` |
| Global pooling | AdaptiveAvgPool1D(1) | `[B,128,1]` |
| Shared representation | squeeze temporal dimension | `[B,128]` |

Before global pooling, each local activation in the final convolutional feature map has an effective receptive field of 26 original input frames; adaptive average pooling then aggregates all 21 temporal locations, so the final 128-dimensional representation depends on the complete 84-frame window.

### Position head

```text
128 -> Linear(64) -> ReLU -> Dropout(0.2) -> Linear(2)
```

Output:

```text
z_mag shape = [B,2]
```

This is the 2-D Cartesian magnetic position fix used directly by DualKalmanNet.

### Scalar uncertainty head

The second head is

```text
128 -> Linear(32) -> ReLU -> Linear(1)
```

and returns one unconstrained scalar `ell_mag` per window. The implementation historically names this quantity `log_variance`; define the corresponding positive scale as

```text
q_mag = exp(ell_mag).
```

There are **not** separate x/y variances and there is no learned 2x2 magnetic covariance matrix.

## 5. What the uncertainty-training loss actually means

The active checkpoint was trained with the repository objective

```text
0.5 * ||z_mag - p||^2 / q_mag + 0.5 * ell_mag,
```

with a numerical lower floor of `0.01` applied to `exp(ell_mag)` in the denominator.

This objective encourages the uncertainty head to assign a larger scale to examples with larger position residuals while the `+0.5*ell_mag` term penalizes making the scale arbitrarily large. The position and uncertainty heads share the CNN encoder, so the scale is predicted from the same 84-frame magnetic representation used for localization.

Implementation caveat: the historical code floors `exp(ell_mag)` only in the residual denominator, while the additive `0.5*ell_mag` term itself is not floored. The learned scores in the current calibration runs are far above that floor, but a future probabilistic retraining should use a consistently bounded parameterization if calibrated likelihood semantics are desired.

### Important probabilistic interpretation

The current objective should **not** be described as the exact negative log-likelihood of a 2-D isotropic Gaussian with covariance `sigma^2 I_2`.

For a true two-dimensional isotropic Gaussian, omitting constants, the normalization term would be

```text
0.5 * ||e||^2 / sigma^2 + log(sigma^2),
```

whereas the current implementation uses only `0.5 * log-scale` while applying the scale to the summed 2-D squared error. Consequently, the scalar learned by the current checkpoint is best interpreted as a **relative uncertainty / difficulty scale associated with radial position error**, not as a calibrated per-axis Cartesian variance.

This distinction is important for the paper. The existing uncertainty-calibration experiment shows that the raw scale is conservative in absolute units but useful for ranking magnetic prediction reliability. Therefore the final fusion method uses only **relative uncertainty**:

```text
ell_ref = median training ell_mag
w_mag   = 1 / (1 + exp(ell_mag - ell_ref)).
```

Only the ordering/difference of the learned scores is needed for this weighting; the method does not insert `q_mag` as a literal Kalman covariance.

## 6. Why the final paper keeps the current checkpoint

P6/P7 are an audit of the already reported architecture, not a new model-training experiment. Correcting the loss to a fully specified 2-D Gaussian likelihood would require retraining the magnetic CNN and then rerunning uncertainty calibration and every downstream fusion experiment. The current paper instead makes the narrower claim supported by the existing checkpoint and calibration results: the second head provides a learned **relative magnetic uncertainty score** that is useful for confidence ranking.

A future probabilistic variant can retrain with a mathematically consistent 2-D likelihood (and a consistently bounded log-scale) if calibrated covariance is desired. That is a new experiment rather than a wording fix.