import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from fbm import FBM
import time
import os

class HurstPredictor:
    def __init__(self, ntimes=20, dt=0.02, base_model_path="LSTM_3d_H_w20.h5"):
        self.ntimes = ntimes
        self.dt = dt
        self.base_model_path = base_model_path
        self.model = load_model(base_model_path)

    def predict_window(self, x_win, y_win, z_win):
        dx = x_win[1:] - x_win[:-1]
        dy = y_win[1:] - y_win[:-1]
        dz = z_win[1:] - z_win[:-1]
        ddx = dx / (np.amax(dx) - np.amin(dx) + 1e-10)
        ddy = dy / (np.amax(dy) - np.amin(dy) + 1e-10)
        ddz = dz / (np.amax(dz) - np.amin(dz) + 1e-10)
        inp = np.empty((1, self.ntimes - 1, 3))
        inp[0, :, 0] = ddx
        inp[0, :, 1] = ddy
        inp[0, :, 2] = ddz
        preds = [self.model.predict(inp, verbose=0)[0][0] for _ in range(5)]
        return np.mean(preds), max(0.0, 1.0 - np.max(np.abs(preds - np.mean(preds))))

    def process_trajectory(self, x, y, z, t, pid, output_file):
        with open(output_file, "a") as f:
            for i in range(0, len(x) - self.ntimes):
                x_win = x[i:i + self.ntimes]
                y_win = y[i:i + self.ntimes]
                z_win = z[i:i + self.ntimes]
                t_mid = t[i + self.ntimes // 2]

                # Initial prediction using base model
                self.model = load_model(self.base_model_path)
                t0 = time.time()
                h_init, conf1 = self.predict_window(x_win, y_win, z_win)
                t1 = time.time()
                print(f"  Window {i}: Initial H = {h_init:.3f} (conf: {conf1:.3f}) in {t1 - t0:.2f}s")

                # Write output
                f.write(f"{pid},{t_mid:.5f},{h_init:.5f}\n")
                f.flush()

if __name__ == "__main__":
    df = pd.read_csv('TRAJECTORIES.csv')
    output_file = "H_LSTM_PREDICTIONS.csv"

    # Write header once
    with open(output_file, "w") as f:
        f.write("pid,time,H_predicted\n")

    for pid in range(1, 350):
        print(f"\nProcessing PID = {pid}")
        data = df[df['pid'] == pid]

        if data.empty:
            print(f"  No data found for PID = {pid}, skipping.")
            continue

        x = np.array(data['x'])
        y = np.array(data['y'])
        z = np.array(data['z'])
        t = np.array(data['t'])

        predictor = HurstPredictor(ntimes=20, dt=0.02, base_model_path="LSTM_3d_H_w20.h5")
        predictor.process_trajectory(x, y, z, t, pid, output_file)
