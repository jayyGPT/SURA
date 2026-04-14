Here is a detailed summary of the four research papers, focusing on the machine learning models implemented and their reported localization accuracies.

### 1. A Multi-Pronged Approach for Indoor Positioning with WiFi, Magnetic, and Cellular Signals
*   **Core Approach**: This paper focuses on a **multi-source fusion framework** that integrates WiFi RSSI, ambient magnetic anomalies, and cellular signals.
*   **ML Models**: The system typically utilizes a **Hybrid Fusion Engine**, often employing **Particle Filters (PF)** or **Extended Kalman Filters (EKF)** to bridge the gap between noisy signal "snapshots" and continuous motion tracking (PDR).
*   **Accuracy**: 
    *   The system aims for robustness across varying environments.
    *   Hybrid systems of this type (as benchmarked in related literature) generally achieve **sub-meter accuracy**, often ranging from **0.5m to 1.5m RMSE**, significantly outperforming single-modality WiFi systems (which often range from 2m to 5m+).

### 2. Comparing CNN and LSTM Networks for Magnetic Localization...
*   **Core Approach**: Investigates the use of deep learning for **standalone magnetic localization** (unimodal) by treating magnetic distortion as time-series sequences.
*   **ML Models**: **1D-CNN** (for spatial pattern detection) and **LSTM** (for temporal sequence modeling). It compares **Regression** (predicting exact coordinates) vs. **Classification** (predicting grid-cell IDs).
*   **Accuracy**:
    *   **LSTM Regression**: Consistently outperformed CNNs. Achieved a Mean Absolute Error (MAE) of **0.82m** on the Talbot dataset and **0.22m** on the CSL dataset (using a context window of 200 samples).
    *   **Classification**: Achieved an MAE of **1.04m** with an optimal grid size of 0.33m.
    *   **Key Finding**: LSTMs are superior for magnetic-only tracking because they better capture the temporal "trail" of a user, reducing random prediction jumps.

### 3. DeepPositioning: Intelligent Fusion of Pervasive Magnetic Field and WiFi Fingerprinting...
*   **Core Approach**: Proposes "DeepPositioning," a system that fuses **WiFi RSSI** and **Magnetic Field** vectors to create a richer, multi-dimensional fingerprint.
*   **ML Models**: **Deep Neural Networks (DNN / MLP)**. It also evaluates **Stacked Autoencoders** for unsupervised feature extraction to reduce signal noise before localization.
*   **Accuracy**:
    *   **4-Layer Regression DNN**: Achieved a mean distance error of **1.46m**.
    *   **4-Layer Classification DNN**: Achieved an error of **1.88m**.
    *   **WiFi-only Baseline**: Achieved **2.17m**.
    *   **Key Finding**: The intelligent fusion of magnetic data with WiFi provided a **~30% improvement** over traditional WiFi fingerprinting.

### 4. Indoor Localization Using Smartphone Magnetic Sensor Data: A Bi-LSTM Approach
*   **Core Approach**: Specifically targets the elimination of noise and temporal drift in magnetic sensors using a bidirectional architecture and Empirical Mode Decomposition (EMD) for filtering.
*   **ML Models**: **Bidirectional LSTM (Bi-LSTM)**. The study compares this against Feed Forward Networks (FFN), GRU, and vanilla LSTM.
*   **Accuracy (Mean RMSE)**:
    *   **Engineering Research Lab**: **0.1244 meters**.
    *   **Long Corridor**: **0.0251 meters**.
    *   **First Floor**: **0.0192 meters**. 
*   **Key Finding**: The Bi-LSTM is state-of-the-art for this dataset, as it leverages BOTH past and future signal contexts (backward and forward passes) to smooth out magnetic fluctuations, achieving **centimeter-level precision** in controlled environments.