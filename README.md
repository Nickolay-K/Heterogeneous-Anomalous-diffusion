# Estimation of anomalous diffusion exponent and generalised diffusion coefficient for space-time heterogeneous anomalous diffusion 

> The repository is hosting the code for the work "Deep Learning for Heterogeneous Anomalous Dynamics in Cellular and Molecular Biology" by Nickolay Korabel and Thomas Andrew Waigh.

---

## **Description**

A neural network-based method for estimating the time-dependent anomalous exponent α(t) and the generalised diffusion coefficient D(t) from single-particle trajectories.

• The method is based on a tandem of LSTM neural networks: The H-LSTM estimates the time-dependent anomalous exponent α(t) and the D-LSTM estimates the generalised diffusion coefficient D(t).

• Space-time heterogeneities are resolved using a rolling window of 20 data points that slides along the trajectory by one time increment. Within each window, anomalous diffusion is assumed to follow standard fractional Brownian motion with a constant Hurst exponent (related to the anomalous exponent as H = α/2) and a constant generalised diffusion coefficient. The first neural network (H-LSTM), which is trained on FBM trajectories of the same length as the rolling window, estimates the Hurst exponent. This resolves the time-dependent anomalous exponent α(t). The second neural network (D-LSTM) uses the same-size rolling window and the predicted value of the Hurst exponent in each window to estimate the time-dependent generalised diffusion coefficient D(t) from single-particle trajectories.
 
• On average, the NNs achieved a 60% increase in the accuracy of estimating the anomalous exponent α and a 150% increase in the estimation of the generalised diffusion coefficient over time-averaged MSD analysis for short, noisy trajectories with heterogeneous dynamics.
 
• The NNs were validated using synthetic and experimental datasets, including intracellular and cellular motility and microrheology in soft matter systems.

---

## **Usage**

1) Train the H-LSTM model by running the Train_H_LSTM.py script. 

2) Run the Predict_H.py script to estimate the time-dependent Hurst exponent for one or more trajectories.

3) Run the Predict_D.py script to estimate the generalised diffusion coefficient using time-dependent Hurst exponents predicted by H-LSTM.

```
