Yes — here is a **clean integration map** for plugging any new dataset into the current architecture.

The description below matches the **active repo + current paper**, not the older drafts.

---

# 1) Big picture

```text
New dataset
   ↓
Convert to one common 2-D map frame (x, y)
   ↓
Create surveyed reference database
   ├─ Wi-Fi fingerprints at known locations
   ├─ Magnetic samples at known locations
   └─ IMU/orientation data for motion model
   ↓
Train 3 pieces
   ├─ Wi-Fi MLP         : RSSI vector → Wi-Fi position z_wifi
   ├─ Magnetic CNN      : 84×4 magnetic window → z_mag, ell_mag
   └─ DualKalmanNet     : PDR + z_wifi + z_mag → final state x_t
   ↓
Run fusion
   ├─ PDR gives prior x_t^-
   ├─ Wi-Fi gives absolute fix
   ├─ Magnetic CNN gives absolute fix + confidence
   └─ KalmanNet fuses them
   ↓
Final 2-D trajectory estimate
```

---

# 2) Minimum data you need

## A. For the **Wi-Fi branch**

You need:

* a set of **known APs / BSSIDs**
* Wi-Fi scans with:

  * AP identifier
  * RSSI value
  * location label `(x, y)` or reference-point ID

### Minimum requirement

At each surveyed location, you must be able to build **one RSSI fingerprint vector**.

---

## B. For the **magnetic branch**

You need:

* **magnetometer**: `mag_x, mag_y, mag_z`
* **accelerometer**: `acc_x, acc_y, acc_z`
* known location `(x, y)` or path position

### Why accelerometer is necessary

The current magnetic features use accelerometer direction as a gravity proxy to compute:

* magnetic norm `m_N`
* vertical component `m_V`
* horizontal component `m_H`
* dip angle `delta`

Without accelerometer, the current magnetic-CNN pipeline **cannot run as-is**.

---

## C. For the **PDR / motion branch**

You need:

* accelerometer stream
* **causal heading** input `theta_hat`

This heading can come from:

* phone orientation / rotation sensor, or
* your own causal heading estimator

### Important

Current PDR assumes heading is already available as an external causal signal.
If your dataset has no heading/orientation, then PDR cannot run unchanged.

---

## D. For training/evaluation

You need:

* a **surveyed indoor map frame**
* enough surveyed nodes to form a connected motion graph
* ideally Wi-Fi + magnetic support over the same region

### Important practical note

For the **current final pipeline**, fully labeled real continuous trajectories are **not mandatory**.
This repo trains the final fusion benchmark using:

* surveyed static database
* synthetic trajectories generated on the survey graph
  rather than raw continuously labeled ground-truth walks.

---

# 3) Clean canonical format you should convert any new dataset into

Instead of forcing raw files to exactly match MagWi, the easiest approach is to convert any new dataset into these **three clean canonical tables/files**.

---

## File 1: `wifi_fingerprints.csv`

One row = one Wi-Fi fingerprint at one surveyed point.

**Required columns**

* `building`
* `node_id`
* `x`
* `y`
* `phone` *(optional but useful)*
* `scan_id`
* one column per AP, e.g.

  * `ap_1`
  * `ap_2`
  * `ap_3`
  * ...
  * `ap_N`

### Values

* each AP column = RSSI in dBm
* if AP absent, store **-100**

### Example

```text
building,node_id,x,y,phone,scan_id,AP_00,AP_01,AP_02,AP_03
BE,17,12.0,4.0,S8,scan_001,-67,-82,-100,-75
BE,18,13.0,4.0,S8,scan_002,-70,-80,-88,-100
```

---

## File 2: `magnetic_samples.csv`

One row = one raw magnetic sample.

**Required columns**

* `building`
* `path_or_visit_id`
* `timestamp`
* `x`
* `y`
* `mag_x`
* `mag_y`
* `mag_z`
* `acc_x`
* `acc_y`
* `acc_z`

**Optional but useful**

* `phone`
* `user`
* `mode`
* `scenario`

### Example

```text
building,path_or_visit_id,timestamp,x,y,mag_x,mag_y,mag_z,acc_x,acc_y,acc_z
BE,visit_012,0.00,12.0,4.0,12.1,-31.4,42.7,0.1,0.2,9.7
BE,visit_012,0.06,12.0,4.0,12.4,-31.0,42.5,0.0,0.1,9.8
```

---

## File 3: `imu_path.csv`

One row = one IMU/PDR sample.

**Required columns**

* `path_id`
* `timestamp`
* `acc_x`
* `acc_y`
* `acc_z`
* `heading`

**Optional**

* `gyro_x, gyro_y, gyro_z`
* `x_true, y_true` *(only if you have real ground truth)*

### Example

```text
path_id,timestamp,acc_x,acc_y,acc_z,heading
walk_001,0.00,0.2,0.1,9.6,1.52
walk_001,0.06,0.4,0.0,10.1,1.50
```

---

# 4) What each architecture block takes in and gives out

## A. Wi-Fi MLP

### Input

A normalized RSSI vector:

$$
\tilde{s}_t \in [0,1]^N
$$

Built from the raw RSSI vector:

* absent AP → `-100`
* clip RSSI to `[-90, -30]`
* rescale to `[0,1]`

### Output

1. **Heatmap probabilities** over all surveyed cells:

   $$
   p_t \in \mathbb{R}^M
   $$
2. **Wi-Fi Cartesian position**

   $$
   z_{\text{wifi},t} \in \mathbb{R}^2
   $$

### Meaning

The Wi-Fi branch gives an **absolute position fix**.
In active fusion, only `z_wifi` is used directly.

---

## B. Magnetic CNN

### Input

A causal window of length `T = 84` frames:

$$
M_t \in \mathbb{R}^{84 \times 4}
$$

Each frame contains 4 magnetic features:

* `m_N`
* `m_V`
* `m_H`
* `delta`

### Output

1. **Magnetic position fix**

   $$
   z_{\text{mag},t} \in \mathbb{R}^2
   $$
2. **Log-uncertainty score**

   $$
   \ell_{\text{mag},t} \in \mathbb{R}
   $$

### Meaning

This branch gives:

* another **absolute position fix**
* a **relative confidence signal** for that fix

The uncertainty is used as a **relative score**, not as a fully calibrated covariance.

---

## C. PDR model

### Input

At raw IMU rate:

* accelerometer magnitude
* heading `theta_hat_n`

### Internal output

Step displacement per sample:

$$
v_n = d_n L_s
\begin{bmatrix}
\cos \hat{\theta}_n \\
\sin \hat{\theta}_n
\end{bmatrix}
$$

### Final output to fusion

Per fusion bin:

$$
u_t \in \mathbb{R}^2
$$

### Meaning

PDR gives **relative motion**, not absolute position.

---

## D. DualKalmanNet fusion

### Inputs at fusion step `t`

* previous state: `x_{t-1}`
* PDR control: `u_t`
* Wi-Fi fix: `z_wifi,t`
* magnetic fix: `z_mag,t`
* magnetic uncertainty: `ell_mag,t`
* availability masks:

  * `m_wifi,t`
  * `m_mag,t`

### First computes

Prior:

$$
x_t^- = x_{t-1} + u_t
$$

Innovations:

$$
y_{\text{wifi},t} = z_{\text{wifi},t} - x_t^-
$$

$$
y_{\text{mag},t} = z_{\text{mag},t} - x_t^-
$$

### GRU feature vector

13-D input:

* masked Wi-Fi innovation (2)
* masked magnetic innovation (2)
* Wi-Fi delta `Δz_wifi` (2)
* PDR control `u_t` (2)
* previous displacement `Δx_{t-1}` (2)
* Wi-Fi mask (1)
* magnetic mask (1)
* clipped magnetic confidence feature (1)

### Outputs

* `K_wifi,t` : `2×2`
* `K_mag,t` : `2×2`
* recurrent hidden state `h_t`

Then final corrected state:

$$
x_t
$$

### Meaning

This is the final estimator.
It learns **how much** to trust Wi-Fi, magnetic, and motion at each step.

---

# 5) Hard constraints for a new dataset

## Must-have

1. **Common 2-D map frame**

   * all modalities must refer to the same `(x,y)` system
   * if data are in latitude/longitude, convert to local metric coordinates first

2. **Wi-Fi fingerprints at known locations**

3. **Magnetometer + accelerometer**

   * both are needed for current magnetic features

4. **Heading/orientation signal**

   * needed for current PDR

5. **Enough spatial coverage**

   * enough nodes to form a connected graph
   * enough overlap between Wi-Fi-supported and magnetic-supported regions

---

## Nice-to-have

* multiple phones
* multiple users
* multiple holding modes
* repeated scans/visits at each node
* denser Wi-Fi coverage
* magnetic data with stable sampling rate near `16.7 Hz`

---

## If missing, what breaks?

| Missing piece                    | Effect                                       |
| -------------------------------- | -------------------------------------------- |
| No Wi-Fi                         | Wi-Fi MLP cannot run                         |
| No accelerometer                 | magnetic features and PDR break              |
| No heading/orientation           | PDR breaks                                   |
| No common map coordinates        | branches cannot be fused consistently        |
| No surveyed nodes / labels       | Wi-Fi and magnetic supervised training break |
| Very poor Wi-Fi/magnetic overlap | fusion benchmark becomes weak or misleading  |

---

# 6) Clean onboarding path for any new dataset

## Option A — easiest, minimal code changes

Convert the new dataset into the same logical structure as the current processed DB:

* one surveyed fingerprint database
* one set of AP columns
* one magnetic map source
* one graph of nodes

Then run:

1. build processed DB
2. train Wi-Fi MLP
3. train magnetic CNN
4. generate synthetic trajectories on the surveyed graph
5. train DualKalmanNet
6. evaluate full Wi-Fi and degraded Wi-Fi

---

## Option B — if raw format is very different

Write a small adapter that converts your raw files into:

* `wifi_fingerprints.csv`
* `magnetic_samples.csv`
* `imu_path.csv`

Then the rest of the architecture can stay conceptually unchanged.

---

# 7) Small final checklist

Before starting on a new building/dataset, check:

* [ ] Do I have a common `(x,y)` frame?
* [ ] Do I have Wi-Fi fingerprints with known locations?
* [ ] Do I have `mag_x, mag_y, mag_z`?
* [ ] Do I have `acc_x, acc_y, acc_z`?
* [ ] Do I have heading/orientation?
* [ ] Can I build 84-frame magnetic windows?
* [ ] Can I form a connected survey-node graph?
* [ ] Is there enough overlap between Wi-Fi and magnetic support?

If all 8 are yes, the current architecture is basically compatible.

---

If you want, next I can turn this into a **repo-ready markdown note** like `docs/NEW_DATASET_INTEGRATION.md`, and after that we can start with **BE Building** first.
