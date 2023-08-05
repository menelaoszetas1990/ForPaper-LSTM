# Importing the libraries
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras import Sequential
import csv
import time

WHICH_CAPE = 1

# Preprocessing
# Importing the dataset
# Using columns: sog, stw, wspeedbf, wdir, me_power
dataset = pd.read_csv('../data/cape' + str(WHICH_CAPE) + '.csv', usecols=['sog', 'stw', 'wspeedbf', 'wdir', 'me_power'])

X = dataset[['sog', 'stw', 'wspeedbf', 'wdir', 'me_power']].values
Y = dataset['me_power'].values
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


def lstm_function(_double_lstm=False, _sequence_size=5, _epochs=5, _batch_size=16, _try_number=1):
    # Training the model on the Training set
    def to_sequences(_dataset, _seq_size=1):
        x = []
        y = []
        for i in range(len(_dataset) - _seq_size - 1):
            window = _dataset[i:(i + _seq_size), 0:X.shape[1]]
            x.append(window)
            y.append(_dataset[i + _seq_size, 4:])
        return np.array(x), np.array(y)

    start_time = time.time()*1000

    seq_size = _sequence_size  # Number of time steps to look back
    # Larger sequences (look further back) may improve forecasting.

    trainX, trainY = to_sequences(X_train_scaled, seq_size)
    testX, testY = to_sequences(X_test_scaled, seq_size)

    plt_name = ''

    model = Sequential()

    if _double_lstm:
        # LSTM_64_32
        model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        plt_name += 'LSTM_64_32_'
    else:
        # LSTM_64
        model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
        plt_name += 'LSTM_64_'

    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # fit the model
    history = model.fit(trainX, trainY, epochs=_epochs, batch_size=_batch_size, validation_split=0.1, verbose=1)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    end_time = time.time()*1000

    plt_name += 'SeqSize_{}_EP_{}_BatchSize_{}_'.format(_sequence_size, _epochs, _batch_size)
    plt.figure()
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.savefig('plots/Validation_Train_Loss/' + plt_name + 'ValidationTrainLoss_try{}.png'.format(_try_number))

    # invert predictions back to pre-scaled values
    # This is to compare with original input values
    # Since we used MinMaxScaler we can now use scaler.inverse_transform
    # to invert the transformation.
    trainPredict = sc2.inverse_transform(trainPredict)
    trainY = sc2.inverse_transform(trainY)
    testPredict = sc2.inverse_transform(testPredict)
    testY = sc2.inverse_transform(testY)

    # calculate errors
    trainScoreMAE = mean_absolute_error(trainY, trainPredict)
    trainScoreMSE = mean_squared_error(trainY, trainPredict)
    trainScoreRMSE = math.sqrt(mean_squared_error(trainY, trainPredict))

    testScoreMAE = mean_absolute_error(testY, testPredict)
    testScoreMSE = mean_squared_error(testY, testPredict)
    testScoreRMSE = math.sqrt(mean_squared_error(testY, testPredict))

    # shift train predictions for plotting
    # we must shift the predictions so that they align on the x-axis with the original dataset.
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[seq_size:len(trainPredict) + seq_size, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(dataset) - len(testPredict):, :] = testPredict

    # plot baseline and predictions
    plt.figure()
    plt.plot(dataset['me_power'], label='dataset')
    plt.plot(trainPredictPlot[:, 4:], label='trainPredict')
    plt.plot(testPredictPlot[:, 4:], label='testPredict')
    plt.legend()
    plt.savefig('plots/Prediction_Comparison/' + plt_name + 'PredictionComparison_try{}.png'.format(_try_number))

    writer.writerow({'double_LSTM': _double_lstm, 'sequence_size': _sequence_size, 'epoches': _epochs,
                     'batch_size': _batch_size, 'trainScoreMAE': trainScoreMAE, 'trainScoreMSE': trainScoreMSE,
                     'trainScoreRMSE': trainScoreRMSE, 'testScoreMAE': testScoreMAE, 'testScoreMSE': testScoreMSE,
                     'testScoreRMSE': testScoreRMSE, 'exec_time': end_time - start_time})


if __name__ == '__main__':
    columns = ['double_LSTM', 'sequence_size', 'epoches', 'batch_size',
               'trainScoreMAE', 'trainScoreMSE', 'trainScoreRMSE',
               'testScoreMAE', 'testScoreMSE', 'testScoreRMSE', 'exec_time']

    filename = 'Comparison_Data.csv'
    with open(filename, 'a', encoding='utf-8', newline='') as _output_file:
        writer = csv.DictWriter(_output_file, fieldnames=columns)
        writer.writeheader()

        # for Double LSTM
        double_LSTM = True
        for sequence_size in range(5, 26, 5):
            for epoches in range(5, 11):
                for batch_size in [16, 32, 64]:
                    for tryNum in range(1, 6):
                        lstm_function(double_LSTM, sequence_size, epoches, batch_size, tryNum)

        # for Single LSTM
        double_LSTM = False
        for sequence_size in range(5, 26, 5):
            for epoches in range(5, 11):
                for batch_size in [16, 32, 64]:
                    for tryNum in range(1, 6):
                        lstm_function(double_LSTM, sequence_size, epoches, batch_size, tryNum)

    print('Comparison END')
