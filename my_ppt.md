# The Approach
We used the MagWi dataset to train and test a KNN model with different inputs to follow and reproduce the work done by the research paper.
following are the results we got:

# Results

## Pure Magnetic KNN 
the paper reports the mean euclidean error as the 34.29m, and we achived the error of 15.38m by train a Knn with 5 neighbors.The difference in the results are due to the data preprocessing and the exact train and test split done in the research paper. The paper does not discuss this in details so we took our liberty to choose the train and test split and the data preprocessing steps. As already discussed earlier the dataset covers exhaustive test cases so for this experiment we Chose the BE building and trained the model on A8 G7 and S8 models and tested on the completely different S9+ model. The results show that even across different phone models the magnetic field is a very good feature for indoor localization.

## Pure Wi-Fi KNN 
    Again we used the same train test split and the same phone models as before. The paper reports the mean euclidean error as the 12.86, and we achived the error of 5.93m by training a Knn with 5 neighbors. The results are better than the pure magnetic KNN because Wi-Fi signals are less prone to noise and interference compared to magnetic field. More over the datset has approx 127 unique BSSID's in the BE building which gives a good coverage for the Wi-Fi based localization. The difference in results of the paper could again be due to the data preprocessing and the exact train and test split. We processed the raw WiFi RSS value by performing cross normalization and then setting all the unavailable RSS to 0. Doing this makes the model robust across the different cases as it just takes the relative strength of the available signals. 

## Hybrid Feature KNN
Now for the Hybrid model we trained the KNN model with both the magnetic field and the Wi-Fi RSS values as the features. The paper reports the mean euclidean error as the 4.89m, and we achived the error of 5.72m by training a Knn with 5 neighbors. The results are better than the pure magnetic KNN and the pure Wi-Fi KNN because it combines the strengths of both the features. The small difference between the Wi-Fi and Hybrid model results are due to a naive KNN model that we and the research paper used. The number of WiFi routers are 127 and that compared with the 3 magnetic field values makes the Wi-Fi signals more dominant in the feature space. Still the small improvement shows that the magnetic field is also a good feature for indoor localization. Its potential and use case is displayed in the next model.

## EKF Pipeline (Ours)
The research paper discusses the EKF pipeline in detail and we tried to reproduce the same results. The paper reports the mean euclidean error as the 4.31m, and we achived the error of 4.29m by using the EKF pipeline that has 4 steps. The first step is to calculate the position based on the IMU data provided then we use the Wi-Fi RSS values to find the approximate circle in which the user is by utilizing the earlier trained Wi-Fi KNN model. Then we use the magnetic field values to find the approximate location of the user within that circle by utilizing the earlier trained magnetic KNN model. Finally we use the Kalman filter to fuse the information from the IMU data and the Wi-Fi and magnetic field values to get the final position of the user. 

## Conclusion
Overall we were able to reproduce the results of the research paper and believe that we have a proof of concept for the indoor localization system. Integrating the magnetic fields value along with the Wi-Fi RSS values and the IMU data significantly improves the accuracy of the indoor localization system. The EKF pipeline is a very good approach for indoor localization and can be used to build a robust indoor localization system. 

# Area of improvement and Our Future plans.
This is the only paper that uses a publically available dataset and has worked on this dataset. The raw KNN approach used by the paper is not very good and can be improved significantly. The EKF pipeline is a good approach but it can be improved further. The other papers in this domain either works on just the Wi-Fi data or just the magnetic fields data and some which uses both did not provide the dataset publically. Also this paper is the only paper we could find that uses the IMU data along with the Wi-Fi and magnetic field values. So the field is open for improvements and we are planning to work on this further.

## Proposed Solutions
The magnetic field varies a lot 