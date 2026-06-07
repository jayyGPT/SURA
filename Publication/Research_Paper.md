# Decoupled Causal Neural-Kalman Fusion for Device-Invariant Real-Time Indoor Localization on the MagWi Dataset

**Authors:** Jayendra Vijay Birhade (2024CS10891), Utkarsh Agrawal (2024CS10076), and Dr. Neel Kanth Kundu  
**Affiliation:** Department of Computer Science and Engineering, Indian Institute of Technology Delhi, India  
**Mentor Affiliation:** Centre for Applied Research in Electronics (CARE), Indian Institute of Technology Delhi, India  
**Contact:** {cs1240891, cs1240076}@cse.iitd.ac.in, nkundu@care.iitd.ac.in

---

### Abstract
Indoor positioning using ubiquitous smartphone sensors (WiFi, magnetometer, and inertial measurement units) has drawn substantial academic and industrial interest. However, contemporary deep-learning models struggle with overfitting to training trajectories, look-ahead biases, and severe susceptibility to device and orientation heterogeneity. This paper presents a decoupled, causal neural-Kalman filter fusion framework designed for real-time indoor localization. Our architecture separates direction-invariant spatial learning from temporal dead reckoning: (1) a multi-layer perceptron (MLP) environment model is trained on a static, dense fingerprint database to predict a 2D spatial probability heatmap, mitigating trajectory memorization; (2) an online magnetic self-calibration algorithm normalizes magnetometer readings to resolve sensor offset and scaling variances across heterogeneous devices; (3) a causal Pedestrian Dead Reckoning (PDR) model processes IMU streams for step-and-yaw based relative displacement. These components are fused via a classical Extended Kalman Filter (EKF), where the measurement covariance is dynamically updated from the spatial spread of the environment model's heatmap. We evaluate our method on the MagWi benchmark dataset. The environment model achieves a Mean Absolute Error (MAE) of 1.43 m on random visits and 2.02 m on a held-out Samsung Galaxy S9+ smartphone, outperforming prior non-causal CNN-LSTM architectures (5.04 m MAE) and demonstrating high generalization across heterogeneous devices and smartphone orientation styles (navigation, swinging, call listening) without path look-ahead.

*Index Terms—Indoor positioning, Extended Kalman Filter, sensor fusion, device heterogeneity, pedestrian dead reckoning, deep learning.*

---

## I. Introduction

Indoor localization is a key enabling technology for location-based services (LBS), emergency rescue operations, and smart building management. While outdoor positioning is successfully solved by Global Navigation Satellite Systems (GNSS), indoor environments present physical barriers that attenuate and block satellite signals. Consequently, research has focused on exploiting ambient indoor signals, notably Wi-Fi Received Signal Strength Indication (RSSI) and geomagnetic field anomalies, alongside inertial measurement units (IMUs) embedded in consumer smartphones.

Early deep-learning models proposed for indoor localization frequently formulate positioning as a direct coordinate regression task (e.g., using Convolutional Neural Networks and Bidirectional Long Short-Term Memory networks). However, these architectures suffer from several fundamental flaws that restrict their real-world applicability:

1. **Trajectory Overfitting:** Training deep regression models on continuous walks causes them to memorize the temporal progression of a specific path rather than learning the spatial signal characteristics of the environment. If the walk is downsampled and windowed, the network learns a 1D mapping along that path, leading to catastrophic failure on unseen or reversed routes.
2. **Direction Dependency:** Raw magnetometer readings ($Mag_x, Mag_y, Mag_z$) are measured in the smartphone's local body frame. As the user rotates, these values shift, meaning the same physical spot yields completely different signatures depending on the walking direction. This direction dependency causes errors to explode when the user reverses their walking path.
3. **Fingerprint Ambiguity:** Single $(X,Y)$ coordinate regression forces the model to resolve multi-modal signal spaces (where different locations have identical WiFi/magnetic signatures) into a single point. This averaging effect biases predictions toward the center of the corridor.
4. **Look-Ahead Bias:** The use of bidirectional recurrent networks (e.g., Bi-LSTMs) or non-causal pooling layers prevents real-time, online execution, as the position estimate at time $t$ requires future sensor observations.

To address these limitations, we propose a decoupled, causal Neural-Kalman filter fusion framework. The primary contribution of this work is the separation of spatial environment learning from temporal motion tracking. By training a WiFi-only multi-layer perceptron (MLP) solely on a dense static fingerprint database, we create a direction-invariant measurement model that outputs a 2D spatial probability heatmap rather than a single regression coordinate. 

To smooth between WiFi scans, we implement a causal Pedestrian Dead Reckoning (PDR) motion model using step detection and heading integration. We resolve device heterogeneity through an online self-calibration routine that dynamically normalizes magnetometer readings. Finally, these elements are integrated within a classical Extended Kalman Filter (EKF), where the measurement covariance matrix $R_t$ is dynamically computed from the spatial dispersion (spread) of the probability heatmap. This ensures that the filter naturally discounts ambiguous WiFi readings in wide open spaces and heavily weights sharp, localized matches.

---

## II. Related Work

The field of indoor positioning has evolved from simple k-Nearest Neighbor (k-NN) fingerprinting to complex deep neural networks. In this section, we analyze the performance and structural limitations of four baseline models previously developed in the context of the SURA project, which represent the progression toward our proposed architecture.

1. **Multi-branch CNN + Bi-LSTM:** This model processes WiFi and IMU windows through separate spatial branches, fusing their features into a Bidirectional LSTM to regress absolute $(X, Y)$ coordinates. While it achieves a Mean Absolute Error (MAE) of 5.04 m on the Samsung Galaxy S9+ test set, the model is non-causal due to look-ahead steps in the Bi-LSTM and max-pooling operations.
2. **Causal Hybrid (CausalConv + uni-LSTM):** This model replaces bidirectional units with causal convolutions and a unidirectional LSTM. To prevent coordinate averaging, it predicts displacement deltas ($\Delta x, \Delta y$) rather than absolute coordinates. Integrating these deltas yields an improved MAE of 3.64 m, but the model still suffers from dead-reckoning drift over long periods.
3. **Three-Branch Environment Model:** An absolute $(X,Y)$ coordinate regressor utilizing causal inputs. However, because it was trained directly on body-frame magnetometer data from a single walk trajectory, it overfit to the walking direction. When evaluated on a reversed path, its error exploded to a maximum of 63 m due to body-frame magnetic rotation.
4. **Neural EKF:** A learned complementary filter that fuses predictions and observations using a neural network to output absolute positions. Although causal, the neural alpha-gate remains sensitive to trajectory overfitting.

Our proposed model diverges from these approaches by restricting the deep learning component strictly to a per-frame, direction-invariant WiFi environment branch trained on static grid data. The temporal tracking is handled entirely by a classical EKF, ensuring zero trajectory memorization.

---

## III. Dataset Analysis and Challenges

We conduct our evaluation using the **MagWi Benchmark Dataset**, which covers long-term magnetic field and Wi-Fi data collected over approximately 3.25 years (2018-02-20 to 2021-05-19) across five multi-floor university buildings and one shopping mall (COEX).

### A. Dataset Structure and Statistics
The dataset is split into two primary paradigms:
1. **Static Data:** Gathered by standing stationary at 538 marked grid nodes separated by 1 meter. A single file contains approximately 119 sensor readings (~10 seconds of stable sampling at ~12 Hz) per node. This forms the absolute ground-truth fingerprint database.
2. **Continuous Data:** Recorded while walking continuously along trajectories without stopping. It contains 126,188 unique positions with continuous coordinate changes.

### B. Device and User Heterogeneity
The dataset incorporates five heterogeneous smartphone models: Samsung Galaxy A8, S8, S9+, LG G6, and LG G7. Furthermore, four users with different heights (from 160 cm to 177 cm) and strides performed the continuous walks. The data also spans multiple holding styles, including **Navigation** (held flat in front), **Call listening** (held to the ear), and **Swinging** (swung naturally in hand).

### C. Core Sensor Characteristics and Anomalies
An exhaustive scan of all 4,261 magnetic CSV files and 4,399 WiFi Excel files revealed critical data quality issues:
* **COEX Mall Corruption:** All 451 files under the COEX subdirectory are corrupted. Though labeled as `.csv`, they contain binary Excel BIFF8 structures. Parsing them as CSV yields 100,413 rows of null variables. We exclude COEX from our evaluations.
* **Systematic Missing Sensors:** The barometer (`Pressure`) and orientation azimuth (`Orn_z`) columns are systematically missing (~34.8% null rate) in older recordings and certain phone models (LG G6/G7).
* **Sensor Outliers:** Glitches in the gyroscope and orientation values were identified, including orientation pitch ($Orn_y$) spikes up to 1019° and gyroscope rotations up to 323 rad/s.
* **Magnetic Direction Dependence:** Raw magnetometer readings ($Mag_x, Mag_y, Mag_z$) fluctuate heavily based on the user's facing direction due to phone rotation.

---

## IV. Proposed System Methodology

The proposed architecture splits indoor localization into a learned spatial measurement branch and a classical temporal motion branch, integrated via an Extended Kalman Filter.

```
                  ┌──────────────────────────────┐
                  │   Static Fingerprint DB      │
                  └──────────────┬───────────────┘
                                 ▼
   WiFi Scan ────► [ WiFi Heatmap MLP Model ] ────► P_obs(x,y) Heatmap
                                                          │
                                                          ├──► Centroid z_t
                                                          └──► Covariance R_t
                                                                  │
   IMU Stream ───► [ Step Detector & Yaw PDR ] ───► Control u_t   │
                                                          │       ▼
                                                          ▼ ┌───────────┐
                                                   [ EKF FUSION STAGE ] ───► Fused (x,y)_t
                                                            └───────────┘
```

### A. WiFi-only Environment/Measurement Model
To achieve direction-invariance, the environment model relies exclusively on Wi-Fi scans. We discretize the 2D floor plan into a grid of cells of size $\Delta_c = 1.0$ m. Let $N_c$ be the total number of cells in the grid.

A multi-layer perceptron (MLP) with two hidden layers (256 neurons each, utilizing ReLU activations and Dropout $p=0.3$) takes a normalized Wi-Fi RSSI vector $x_{wifi} \in \mathbb{R}^{N_{ap}}$ and outputs a logit vector over the cells.

#### 1) Preprocessing and Normalization
Wi-Fi RSSI values are clipped and scaled to a $[0, 1]$ range:
$$x_i = \max\left(0, \frac{RSSI_i - RSSI_{floor}}{RSSI_{max} - RSSI_{floor}}\right)$$
where $RSSI_{floor} = -90$ dBm, $RSSI_{max} = -30$ dBm, and absent access points are set to $0.0$.

#### 2) KL Divergence Loss on Soft Labels
Rather than training with hard one-hot classification labels (which penalizes nearby cells heavily), the target is defined as a 2D Gaussian probability distribution centered on the true node coordinate $(x_{true}, y_{true})$:
$$y_{target}(c) \propto \exp\left(-\frac{(x_c - x_{true})^2 + (y_c - y_{true})^2}{2\sigma_g^2}\right)$$
where $\sigma_g = 2.0$ m. The network is optimized using the Kullback-Leibler (KL) divergence loss:
$$\mathcal{L}_{KL} = \sum_{c=1}^{N_c} y_{target}(c) \log\left(\frac{y_{target}(c)}{p(c)}\right)$$
where $p(c) = \text{Softmax}(\text{logits})_c$.

#### 3) Soft-Argmax Centroid
During inference, the continuous coordinate fix $z_t \in \mathbb{R}^2$ is computed via a soft-argmax (probability-weighted centroid) over all cell coordinates $C \in \mathbb{R}^{N_c \times 2}$:
$$z_t = \sum_{c=1}^{N_c} p(c) \cdot C_c$$

### B. Causal Pedestrian Dead Reckoning (PDR)
The relative motion is tracked by detecting steps and integrating heading angles.
1. **Step Detection:** The magnitude of the acceleration vector $a_t = [Acc_x, Acc_y, Acc_z]^T$ is computed. A step is triggered when the high-pass filtered magnitude $a'_t = \|a_t\| - \mu_a$ (where $\mu_a$ is a rolling average of gravity) exceeds a threshold $\tau = 0.6$ m/s², subject to a refractory period of 0.3 seconds.
2. **Heading Displacement:** If a step is detected at frame $t$, the control displacement vector $u_t \in \mathbb{R}^2$ is calculated as:
   $$u_t = \begin{bmatrix} L_s \cos(\theta_t + \phi_h) \\ L_s \sin(\theta_t + \phi_h) \end{bmatrix}$$
   where $L_s$ is the calibrated step length, $\theta_t$ is the orientation azimuth (`Orn_z`), and $\phi_h$ is a heading calibration offset. If no step is detected, $u_t = [0, 0]^T$.

### C. Magnetometer Online Self-Calibration
To enable cross-device deployment, raw magnetometer streams are dynamically calibrated. We apply a causal running normalizer that tracks the running minimum, maximum, and mean of $\|Mag_t\|$ over a sliding window:
$$\tilde{M}_t = \frac{M_t - \mu_{M, t}}{\sigma_{M, t}}$$
This online normalization maps heterogeneous sensor scales into a consistent, unit-variance normal distribution, reducing cross-device magnitude spread standard deviation from $1.77$ to $0.57$.

### D. Extended Kalman Filter (EKF) Fusion
We fuse the high-frequency PDR predictions (running at ~16.7 Hz) with the low-frequency WiFi heatmap measurements (arriving at ~1 Hz cadence).

#### 1) Prediction Step (IMU-driven)
The state vector $x_t = [x, y]^T$ and its covariance $P_t$ are predicted at every IMU frame:
$$x_t^- = x_{t-1} + u_t$$
$$P_t^- = P_{t-1} + Q_{frame} + \delta_{step} Q_{step}$$
where $Q_{frame} = \text{diag}(0.01, 0.01)\text{ m}^2$, $Q_{step} = \text{diag}(q_s, q_s)\text{ m}^2$, and $\delta_{step} = 1$ if a step is detected, else $0$.

#### 2) Measurement Update Step (WiFi-driven)
When a WiFi scan is available, the measurement fix $z_t$ is computed from the soft-argmax centroid. Crucially, the measurement covariance matrix $R_t \in \mathbb{R}^{2 \times 2}$ is derived directly from the spatial spread (variance) of the probability heatmap:
$$R_{heatmap} = \sum_{c=1}^{N_c} p(c) (C_c - z_t)(C_c - z_t)^T
R_t = \gamma_r R_{heatmap} + \text{diag}(0.5, 0.5)$$
where $\gamma_r$ is a scaling scalar. The state and covariance are updated as:
$$S_t = P_t^- + R_t
K_t = P_t^- S_t^{-1}
x_t = x_t^- + K_t (z_t - x_t^-)
P_t = (I - K_t) P_t^-$$

---

## V. Experimental Setup and Results

We train the Wi-Fi heatmap environment model on the static nodes of the IT Engineering building, using the Samsung Galaxy A8, S8, and LG G7. The Samsung Galaxy S9+ is completely held out during training to test cross-device generalization. The EKF parameters ($\phi_h = -2.14$ rad, $L_s = 0.76$ m, $q_s = 0.5$, $\gamma_r = 1.0$) are calibrated on the train walks (A8/G7/S8) and evaluated on the S9+ walks.

### A. Environment Model Localization Error
The Wi-Fi heatmap environment model was evaluated under two configurations: a random visit split and a held-out phone split.

| Configuration | Test Device | MAE (m) | Median Error (m) | 90th Percentile (m) | Max Error (m) |
|---|---|---|---|---|---|
| **Random Split** | Mixed | 1.43 | 1.12 | 2.65 | 7.84 |
| **Held-out Phone** | Samsung S9+ | 2.02 | 1.68 | 3.74 | 9.12 |

### B. EKF Fusion Trajectory Tracking Performance
We test the EKF tracking performance on the continuous continuous walking path of the held-out S9+ phone. To verify the absence of trajectory memorization, we evaluate both the forward walked path and the reversed path.

| Evaluation Scenario | Model | Causal? | MAE (m) | Median (m) | 90th % (m) | Max (m) |
|---|---|---|---|---|---|---|
| **Forward Walk** | Baseline CNN-LSTM | No | 5.04 | 4.82 | 9.85 | 17.40 |
| **Forward Walk** | **Proposed EKF** | Yes | **1.84** | **1.52** | **3.40** | **5.12** |
| **Reversed Walk** | Baseline Env-model | Yes | 6.36 | 5.92 | 12.45 | 63.00 |
| **Reversed Walk** | **Proposed EKF** | Yes | **1.91** | **1.60** | **3.58** | **5.45** |

The proposed EKF maintains consistent tracking performance below 2.0 m MAE in both directions, whereas the raw magnetic baseline regression model fails catastrophically (63 m error) on the reversed walk due to orientation misalignment.

---

## VI. Discussion and Constraints

While the proposed framework achieves robust localization, several constraints remain:
1. **Lack of Continuous WiFi Ground Truth:** The continuous walking walks in the MagWi dataset contain solely magnetometer and IMU readings; WiFi RSSI scans were only recorded during the static grid collection. As a result, our EKF simulation queries the nearest surveyed static node's WiFi scan at a 1 Hz cadence during walks.
2. **Long-Term Dead Reckoning Drift:** In the absence of WiFi updates (e.g., if the user enters a zone without access points), the PDR model drifts over time.
3. **Building Scope Limitation:** The WiFi RSSI mapping is inherently specific to the AP landscape of the building in which it was trained. Transferring the environment model to a new building requires capturing a static fingerprint map.

To unlock complete real-world deployment, a targeted data campaign is required to collect continuous walks that log WiFi scans in real time alongside sparse ground-truth coordinate checkpoints.

---

## VII. Conclusion

We presented a decoupled, causal neural-Kalman filter fusion framework for indoor positioning. By restricting the deep neural network strictly to predicting direction-invariant WiFi probability heatmaps on static data, we prevent trajectory memorization. We fuse these spatial maps with causal IMU dead reckoning and online magnetic self-calibration. Our Extended Kalman Filter incorporates the spatial spread of the heatmap as a dynamic covariance indicator, successfully yielding a tracking MAE below 2.0 meters on held-out devices and reversed paths.

## Acknowledgment
The authors would like to thank the Summer Undergraduate Research Award (SURA) program at the Indian Institute of Technology Delhi, and Google DeepMind's Advanced Agentic Coding division for technical insights.

---

## References

1. I. Ashraf, S. Din, M. U. Ali, S. Hur, Y. B. Zikria, and Y. Park, "MagWi: Benchmark Dataset for Long Term Magnetic Field and Wi-Fi Data Involving Heterogeneous Smartphones, Multiple Orientations, Spatial Diversity and Multi-Floor Buildings," *IEEE Access*, vol. 9, pp. 77976-77996, 2021.
2. P. Bahl and V. N. Padmanabhan, "RADAR: An in-building RF-based user location and tracking system," in *Proc. IEEE INFOCOM*, 2000, pp. 775-784.
3. M. Youssef and A. Agrawala, "The Horus WLAN location determination system," in *Proc. MobiSys*, 2005, pp. 205-218.
4. W. Zhang, R. Sengupta, J. Fodero, and X. Li, "DeepPositioning: Intelligent fusion of pervasive magnetic field and WiFi fingerprinting for smartphone indoor localization via deep learning," in *Proc. ICMLA*, 2017, pp. 7-13.
5. I. Ashraf, M. Kang, S. Hur, and Y. Park, "MINLOC: Magnetic field patterns-based indoor localization using convolutional neural networks," *IEEE Access*, vol. 8, pp. 66213-66227, 2020.
