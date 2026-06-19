# Walkthrough: Dual-Innovation KalmanNet Evaluation

We have successfully implemented, trained, and evaluated the **Dual-Update KalmanNet**. This model extends the previous Wi-Fi + PDR baseline by learning to dynamically fuse continuous magnetic field sequence measurements using a unified GRU with a dual 2x2 matrix gain structure.

## Summary of Completed Work

### 1. Magnetic Sequence Matcher (`stage2_mag_sequence.py`)
Instead of single-point magnetic matching (which is highly ambiguous), we implemented a 1D-CNN that maps a temporal sliding window of rotation-invariant magnetic features (`magN, magV, magH, dip`) to a spatial coordinate.
- It predicts both the position and an uncertainty estimate using a heteroscedastic NLL loss.
- We ran a window-size sweep to find the optimal temporal context:
  - 3.0s (50 frames): 5.52m MAE
  - **5.0s (84 frames): 3.58m MAE (Best)**
  - 8.0s (134 frames): 3.66m MAE
  - 10.0s (167 frames): 3.86m MAE

![Magnetic Sequence Matcher CDF](/C:/Users/lenovo/.gemini/antigravity-ide/brain/000854a9-60ec-4301-8b6a-43560474a576/stage2_mag_sequence_cdf.png)

### 2. Dual-Innovation KalmanNet (`stage3_dual_kalmannet.py`)
We built a GRU that processes a 13-dimensional input feature vector, taking in both Wi-Fi innovations and gradient-projected magnetic innovations. The GRU outputs two separate $2 \times 2$ matrices ($K_{wifi}$ and $K_{mag}$), applying corrections dynamically based on sensor availability.

> [!TIP]
> **Gradient-Projected Magnetic Innovation**: The filter calculates the scalar magnetic error `(obs - map(pred))` and multiplies it by the spatial gradient of the anomaly map `[∂A/∂x, ∂A/∂y]`. This gives the filter a physically meaningful 2D direction for the magnetic spatial correction.

### 3. Evaluation and Results
We evaluated the Dual KalmanNet against the WiFi-only KalmanNet baseline across 60 held-out synthetic walks under two regimes.

| Regime | Baseline (WiFi+IMU) | DualKalmanNet (WiFi+IMU+Mag) | Improvement |
|--------|---------------------|------------------------------|-------------|
| **Full Wi-Fi** (1 Hz) | $0.55 \pm 0.05$ m | $0.47 \pm 0.03$ m | **+13.4%** |
| **Degraded Wi-Fi** (5s, 40% Drop) | $1.44 \pm 0.18$ m | $1.07 \pm 0.10$ m | **+25.3%** |

![Dual KalmanNet Results CDF](/C:/Users/lenovo/.gemini/antigravity-ide/brain/000854a9-60ec-4301-8b6a-43560474a576/stage3_dual_kalmannet.png)

## Conclusion

The hypothesis was confirmed: while magnetic data adds marginal benefit when high-frequency absolute Wi-Fi fixes are available (+13.4%), it serves as an **excellent bridge across signal outages**. In the degraded scenario (a 5-second gap between Wi-Fi scans), the dense 16.7 Hz magnetic field measurements effectively prevent the IMU Dead-Reckoning from drifting unconstrained, providing a substantial **+25.3% reduction in Mean Absolute Error**.

We have also generated comprehensive [technical documentation](file:///c:/Users/lenovo/Documents/GitHub/SURA/dl_models/dual_kalmannet_docs.md) detailing the architecture and mathematics behind the implementation, which will serve as a strong foundation for the related paper section.

> [!NOTE]
> Progress bars (`tqdm`) and explicit output flushing have been added to the training scripts for future runs to give you better live feedback.
