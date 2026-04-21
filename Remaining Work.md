# SURA Implementation Plan

## 1. Presentation
### i. Novelty
- Quantum Computing to reduce Complexity
- Using IMU Sensors (never done before)
- Perfect and Efficient Fusion of Mag and Wifi Data
- More Useful to Industry for being Public Dataset 

### ii. Previous Work
#### a. Magnetic Field Only
- Comparing_CNN_and_LSTM_Networks_for_Magnetic_Localization_of_IoT_Devices_and_Pedestrian_Tracking
- Indoor_Localization_Using_Smartphone_Magnetic_Sensor_Data_A_Bi-LSTM_Neural_Network_Approach
- Accurate Localization Method Combining Optimized Hybrid
Neural Networks for Geomagnetic Localization with
Multi-Feature Dead Reckoning

These Papers Focus only the magnetic field properties 

#### b. Wifi Only
- Mitigating_Device_Heterogeneity_for_Enhanced_Indoor_Positioning_System_Performance_Using_Deep_Feature_Learning
- Practical_WiFi_Indoor_Localization_Unleashing_the_Potential_of_GNNs_for_Accuracy_and_Robustness

They only use wifi 

#### c. Mag + Wifi
- DeepPositioning_Intelligent_Fusion_of_Pervasive_Magnetic_Field_and_WiFi_Fingerprinting_for_Smartphone_Indoor_Localization_via_Deep_Learning
- A_multi-pronged_approach_for_indoor_positioning_with_WiFi_magnetic_and_cellular_signals

But these use private datasets which cannot be compared against any other SOTA models 

### iii. Summary
We introduce IMU Sensors along with proper hybridisation of the two fields

## 2. ML Model (Proof of Concept)
- KNN on wifi signals only 
- KNN on magnetic + wifi 
- KNN on magnetic + wifi + IMU sensor data
