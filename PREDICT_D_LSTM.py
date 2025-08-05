import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from fbm import FBM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
import time
import os

# ----------------------------
# Parameters
# ----------------------------
ntimes = 20
nn = 10
nn1 = 11
dt = 0.02
nsamples = 27000  # number of FBM trajectories used for D-LSTM training
Tmax = dt * ntimes

# ----------------------------
# Load dataset once
# ----------------------------
d = pd.read_csv('TRAJECTORIES.csv')  # load dataset file with columns "pid,x,y,z,time"
d1 = pd.read_csv('H_LSTM_PREDICTIONS.csv')  # load file with H values predicted by H-LSTM in the form "pid,H_predicted,time"

# ----------------------------
# Output CSV (one for all PIDs)
# ----------------------------
global_output_csv = "D_LSTM_PREDICTIONS.csv"
with open(global_output_csv, "w") as f:
    f.write("PID,time,h_estimate,D_estimate\n")

# ----------------------------
# Loop over PIDs
# ----------------------------
for YY in range(1, 10):
    print(f"\n=== Processing PID {YY} ===")
    start_time = time.time()

    data = d[d['pid'] == YY]
    h_data = d1[d1['pid'] == YY]
    
    if len(data) == 0 or len(h_data) == 0:
        print(f"Skipping PID {YY}: no data found.")
        continue

    x = np.array(data['x'])
    y = np.array(data['y'])
    #z = np.array(data['z'])
    t = np.array(data['t'])
    h_times = np.array(h_data['time'])
    h_values = np.array(h_data['H_predicted'])

    print(f"✓ H range: {np.min(h_values):.4f} to {np.max(h_values):.4f}")

    # Initialize LSTM model for online training
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(40, activation='relu', input_shape=(ntimes, 2), return_sequences=True),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6)

    with open(global_output_csv, "a") as f:
        for idx in tqdm(range(nn, len(x) - nn1), desc=f"PID {YY}"):
            inx = x[(idx - nn):(idx + nn1)]
            iny = y[(idx - nn):(idx + nn1)]
            #inz = z[(idx - nn):(idx + nn1)]
            dx1 = inx[1:] - inx[:-1]
            dx2 = iny[1:] - iny[:-1]
            #dx3 = inz[1:] - inz[:-1]

            sx = np.empty((1, ntimes, 2)) # 2d dataset
            #sx = np.empty((1, ntimes, 3)) # 3d dataset 
            sx[0, :, 0] = dx1
            sx[0, :, 1] = dx2
            #sx[0, :, 2] = dx3

            H_current = np.interp(t[idx], h_times, h_values)
            theoretical_D = d_theory[idx]
            time_point = t[idx]

            # Dynamically train D-LSTM model using H_current
            traindata = np.empty((nsamples, ntimes, 2)) # 2d dataset
            #traindata = np.empty((nsamples, ntimes, 3)) # 3d dataset
            trainlabels = np.empty((nsamples, 1))
            for j in range(nsamples):
                amp = np.random.uniform(1.0, 100.0)
                A = np.sqrt(2 * amp)
                f1 = FBM(n=ntimes, hurst=H_current, length=Tmax).fbm()
                f2 = FBM(n=ntimes, hurst=H_current, length=Tmax).fbm()
                #f3 = FBM(n=ntimes, hurst=H_current, length=Tmax).fbm()
                x1 = f1 * A + np.random.normal(0, 0.001, size=len(f1))
                x2 = f2 * A + np.random.normal(0, 0.001, size=len(f1))
                #x3 = f3 * A + np.random.normal(0, 0.001, size=len(f1))
                dx1 = x1[1:] - x1[:-1]
                dx2 = x2[1:] - x2[:-1]
                #dx3 = x3[1:] - x3[:-1]
                traindata[j, :, 0] = dx1
                traindata[j, :, 1] = dx2
                #traindata[j, :, 2] = dx3
                trainlabels[j] = amp

            model.fit(traindata, trainlabels, epochs=20, batch_size=64, validation_split=0.2,
                      callbacks=[early_stopping, reduce_lr], verbose=0)

            D_predicted = model.predict(sx, verbose=0)[0][0]

            f.write(f"{YY},{time_point:.6f},{H_current:.6f},{D_predicted:.6f}\n")
            f.flush()

    print(f"✓ PID {YY} done in {((time.time() - start_time)/60):.1f} min")

print(f"\n✅ All PIDs processed and saved in {global_output_csv}")
