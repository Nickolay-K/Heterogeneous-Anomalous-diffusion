# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from fbm import FBM

#generate our data for training and testing
nsamples = 77000
ntimes = 20

dt = 0.02  # time increment
Tmax = dt*(ntimes+1)

traindata = np.empty((nsamples,ntimes-1,2)) # 2d
#traindata = np.empty((nsamples,ntimes-1,3)) # 3d
trainlabels = np.empty((nsamples,1))

for i in range(0,nsamples):
    hurst_exp = np.random.uniform(0.01, 0.99)

    amp = np.random.uniform(1, 100)    
    A = np.sqrt(2*amp)
    Tmax = dt*(ntimes)
    
    # Generate fractional Brownian motions
    f1 = FBM(n=ntimes, hurst=hurst_exp, length=Tmax, method='cholesky').fbm()
    f2 = FBM(n=ntimes, hurst=hurst_exp, length=Tmax, method='cholesky').fbm()
    #f3 = FBM(n=ntimes, hurst=hurst_exp, length=Tmax, method='cholesky').fbm()

    # Initialize arrays for the Brownian motions
    fbm1 = np.empty((ntimes,))
    fbm2 = np.empty((ntimes,))
    #fbm3 = np.empty((ntimes,))

    # Compute the Brownian motions
    fbm1[0] = 0
    fbm2[0] = 0
    #fbm3[0] = 0

    for j in range(1, ntimes):
        fbm1[j] = fbm1[j-1] + (f1[j+1] - f1[j]) * A
        fbm2[j] = fbm2[j-1] + (f2[j+1] - f2[j]) * A
        #fbm3[j] = fbm3[j-1] + (f3[j+1] - f3[j]) * A

    # Add noise with the correct length
    x1 = fbm1 + np.random.normal(0, 0.001, size=len(fbm1))
    x2 = fbm2 + np.random.normal(0, 0.001, size=len(fbm2))
    #x3 = fbm3 + np.random.normal(0, 0.001, size=len(fbm3))

    #apply differencing
    dx1 = (x1[1:]-x1[0:-1])
    dx2 = (x2[1:]-x2[0:-1])
    #dx3 = (x3[1:]-x3[0:-1])

    #apply normalization on the data
    ddx = dx1/(np.amax(dx1)-np.amin(dx1))
    ddy = dx2/(np.amax(dx2)-np.amin(dx2))
    #ddz = dx3/(np.amax(dx3)-np.amin(dx3))

    #test dimention
    #print(ddx.shape)

    traindata[i,:,0] = ddx
    traindata[i,:,1] = ddy
    #traindata[i,:,2] = ddz
    trainlabels[i,:] = hurst_exp

testdata = np.empty((nsamples,ntimes-1,2)) # 2d
#testdata = np.empty((nsamples,ntimes-1,3)) # 3d
testlabels = np.empty((nsamples,1))

for i in range(0,nsamples):
    hurst_exp = np.random.uniform(0.01, 0.99)

    amp = np.random.uniform(1, 100)
    A = np.sqrt(2*amp)
    Tmax = dt*(ntimes)
    
    # Generate fractional Brownian motions
    f1 = FBM(n=ntimes, hurst=hurst_exp, length=Tmax, method='cholesky').fbm()
    f2 = FBM(n=ntimes, hurst=hurst_exp, length=Tmax, method='cholesky').fbm()
    #f3 = FBM(n=ntimes, hurst=hurst_exp, length=Tmax, method='cholesky').fbm()

    # Initialize arrays for the Brownian motions
    fbm1 = np.empty((ntimes,))
    fbm2 = np.empty((ntimes,))
    #fbm3 = np.empty((ntimes,))

    # Compute the Brownian motions
    fbm1[0] = 0
    fbm2[0] = 0
    #fbm3[0] = 0

    for j in range(1, ntimes):
        fbm1[j] = fbm1[j-1] + (f1[j+1] - f1[j]) * A
        fbm2[j] = fbm2[j-1] + (f2[j+1] - f2[j]) * A
        #fbm3[j] = fbm3[j-1] + (f3[j+1] - f3[j]) * A

    x1 = fbm1
    x2 = fbm2
    #x3 = fbm3

    #apply differencing and normalization on the data
    dx1 = (x1[1:]-x1[0:-1])
    dx2 = (x2[1:]-x2[0:-1])
    #dx3 = (x3[1:]-x3[0:-1])

    ddx = dx1/(np.amax(dx1)-np.amin(dx1))
    ddy = dx2/(np.amax(dx2)-np.amin(dx2))
    #ddz = dx3/(np.amax(dx3)-np.amin(dx3))

    testdata[i,:,0] = ddx
    testdata[i,:,1] = ddy
    #testdata[i,:,2] = ddz
    testlabels[i,:] = hurst_exp
np.savetxt("LSTM_2d_H_testvalues_w"+str(ntimes)+".csv",testlabels,delimiter=",")

print('training data shape:',traindata.shape,'training labels shape:', trainlabels.shape,'test data shape:',testdata.shape,'test labels shape:',testlabels.shape)

# --- Create and Compile LSTM Model ---
print("\n[STEP 2] Creating and compiling LSTM model...")
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(ntimes-1, 2), return_sequences=True),  # 2d
    #LSTM(64, activation='tanh', input_shape=(ntimes-1, 3), return_sequences=True),  # 3d
    Dropout(0.2),
    LSTM(32, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
model.summary()

# --- Train Model ---
print("\n[STEP 3] Training model...")
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5)
history = model.fit(
    traindata, trainlabels,
    epochs=100,
    batch_size=64,
    validation_data=(testdata, testlabels),
    callbacks=[early_stopping, reduce_lr],
    verbose=2
)

# --- Evaluate and Save Model ---
print("\n[STEP 4] Evaluating and saving model...")
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

loss, mae = model.evaluate(testdata, testlabels, verbose=0)
print(f"Final Test Loss: {loss:.4f}")
print(f"Final Test Mean Absolute Error: {mae:.4f}")

model_path = f"./LSTM_2d_H_w{ntimes}.h5"
print(f"Saving model to {model_path}")
model.save(model_path)

#predict values using data in the testing set
test_predictions = model.predict(testdata)
#save predicted values
np.savetxt("LSTM_2d_H_estimated_w"+str(ntimes)+".csv",test_predictions,delimiter=",")


