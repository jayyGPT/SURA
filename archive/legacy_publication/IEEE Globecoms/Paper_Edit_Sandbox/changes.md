# Requested Changes for the Research Paper

Please list the changes you would like to make to the paper below. You can be as specific or as general as you like.

We will change the intent and core idea of the research paper from being an innovation on the MagWi dataset to beign an innovation in the Indoor Localisation industry and show that we have a model that is commpatible with all the three major data types in the industry i.e Wifi, Magnetic and IMU Sensor Data and that it works even if we have any 1 or 2 of the available inputs then we generalise all of the equations of say any number of Wifi APs and RPs and we give a generic model and then in a later section we discuss the comparison and benchmarks where then we compare it with Datasets famous in the industry like the MagWi dataset (may be MagPi dataset not for now) and then we pull up some stats (if we do not have the comparison right now jsut leave some placeholder ill perform the experiemnet and place if there later) 

Also there are a lot of suggestions in the paper itself that is updated you update all of those in here beofer proceeding 

## Changes:

1. Whenever we show a formula or equation we first describe in detail all the variables and parameters in it and use the same set of notatinos throughout the paper 
2. we make all of the mathematics coherent so it is easy to understand the big picture 
3. We also describe and write out any algori8thm techinques loss funtiohn when we specify it so the paper is somewhat sufficient to undersatnd 
4. also for the MLP part or any of the models we properly describe the input and output and pipeline 
5. we remove the preivous baseline critiques and models for now
 

## Additional Notes (Blue Annotations Extracted from PDF):

1. **Section Restructuring:** Describe the general Hybrid problem first, discuss ML limits, motivate architecture, and move dataset specifics to the evaluation section. (Implemented)
2. **IT Building Justification:** Explain why the IT building was chosen and if the method applies elsewhere. (Implemented in Section III)
3. **Epsilon-graph:** Explain what the $\varepsilon$-graph is. (Implemented in Section III.A)
4. **Nodes Definition:** Clarify what the 132/168 nodes are. (Implemented in Section III.A)
5. **Noise Variance:** Clarify that $8.8^\circ$ is the variance of the simulated angular noise. (Implemented in Section III.A)
6. **CNN/LSTM References:** Provide references for the baseline CNN/LSTM models. (Bypassed: The entire baseline critique section was removed per changes.md rule #5).
7. **MLP Loss & Diagram:** Give the mathematical expression of the loss function used for MLP training and a diagram showing inputs/outputs. (Implemented in Section II.B and Fig. 2)
8. **Mathematical Variables (MLP):** Define the mathematical variables used in the heatmap equations. (Implemented in Section II.B)
9. **Wi-Fi Sparsity:** Clarify if/why Wi-Fi APs are sparse. (Implicitly implemented, defined as 1Hz updates compared to 16.7Hz IMU)
10. **CNN Loss & Diagram:** Give the mathematical expression of the loss function used for CNN training and a diagram showing inputs/outputs. (Implemented in Section II.C and Fig. 3)
11. **Hyperparameters:** Explain hyperparameter choices (like the 84-frame window) in the numerical results or training section. (Implemented in Section III.B)
12. **High Pass Filter:** Explain why a high-pass filter is required for PDR. (Implemented in Section II.D: to isolate dynamic acceleration from gravity)
13. **Theta ($\theta_t$):** Define what $\theta_t$ is. (Implemented in Section II.D: real-time yaw heading angle)
14. **Kalman Variables:** Define variables like $\mathbf{x}, \mathbf{y}_{mag}, \mathbf{z}_{mag}$, etc. (Implemented in Section II.E)
15. **Nabla A ($\nabla A$):** Provide mathematical details for the $\nabla A$ calculation. (Implemented in Section II.E: computed via discrete finite differences)
16. **Training Details:** Give more details on model training, hyperparameter choices, optimization algorithms, and loss evolution plots. (Implemented in Section III.B, with a placeholder for the loss plot)
17. **Robustness Validation:** Add a tabular performance comparison or plot to validate Postural Independence. (Implemented placeholder in Section IV.C)
18. **Conclusion vs Future Work:** Merge the Discussion & Future Work section into the Conclusion. (Implemented in Section V)
