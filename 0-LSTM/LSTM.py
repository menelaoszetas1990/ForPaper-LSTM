# Importing the libraries
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential

WHICH_CAPE = 1

# Preprocessing

# Importing the dataset
# Using columns: sog, stw, wspeedbf, wdir, me_power
dataset = pd.read_csv('data/cape' + str(WHICH_CAPE) + '.csv', usecols=[0, 1, 4, 5, 9])

X = dataset.values
Y = dataset.iloc[:, 4].values
Y = Y.reshape(-1, 1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1, shuffle=False)

# Feature Scaling
sc1 = MinMaxScaler()
X_train_scaled = sc1.fit_transform(X_train[:, :])
X_test_scaled = sc1.transform(X_test[:, :])
sc2 = MinMaxScaler()
Y_train_scaled = sc2.fit_transform(Y_train[:, :])
Y_test_scaled = sc2.transform(Y_test[:, :])


# Training the model on the Training set
def to_sequences(_dataset, _seq_size=1):
    x = []
    y = []

    for i in range(len(_dataset) - _seq_size - 1):
        window = _dataset[i:(i + _seq_size), 0:X.shape[1]]
        x.append(window)
        y.append(_dataset[i + _seq_size, 4:])

    return np.array(x), np.array(y)


seq_size = 10  # Number of time steps to look back
# Larger sequences (look further back) may improve forecasting.

# TODO REVIEW FOR ERROR IN LOGIC
trainX, trainY = to_sequences(X_train_scaled, seq_size)
testX, testY = to_sequences(X_test_scaled, seq_size)

print("Shape of training set X: {}".format(trainX.shape))
print("Shape of test set X: {}".format(testX.shape))
print("Shape of training set Y: {}".format(trainY.shape))
print("Shape of test set Y: {}".format(testY.shape))

model = Sequential()

# First Test LSTM_64_32
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))

# Second Test LSTM_64
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))

model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# fit the model
history = model.fit(trainX, trainY, epochs=10, batch_size=64, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

print('Train end')

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions back to pre-scaled values
# This is to compare with original input values
# Since we used MinMaxScaler we can now use scaler.inverse_transform
# to invert the transformation.
trainPredict = sc2.inverse_transform(trainPredict)
trainY = sc2.inverse_transform(trainY)
testPredict = sc2.inverse_transform(testPredict)
testY = sc2.inverse_transform(testY)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[0]))
print('Train Score: %.2f RMSE' % trainScore)

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[0]))
print('Test Score: %.2f RMSE' % testScore)

# shift train predictions for plotting
# we must shift the predictions so that they align on the x-axis with the original dataset.
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[seq_size:len(trainPredict) + seq_size, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(dataset)-len(testPredict):, :] = testPredict

# plot baseline and predictions
plt.plot(dataset['me_power'], label='dataset')
plt.plot(trainPredictPlot[:, 4:], label='trainPredict')
plt.plot(testPredictPlot[:, 4:], label='testPredict')
plt.legend()
plt.show()
print('end')
