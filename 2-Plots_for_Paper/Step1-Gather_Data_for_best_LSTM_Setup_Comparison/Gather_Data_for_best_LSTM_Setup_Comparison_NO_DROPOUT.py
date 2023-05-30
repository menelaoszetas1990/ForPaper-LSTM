# Importing the libraries
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, LearningRateScheduler
import csv
import time


class LSTMPreparation:
    def __init__(self, filename, test_size):
        # Preprocessing
        # Importing the dataset
        # Using columns: sog, stw, wspeedbf, wdir, me_power
        self.filename = filename
        dataset = pd.read_csv('../../data/' + filename + '.csv', usecols=['trim', 'sog', 'stw', 'wspeedbf', 'wdir',
                                                                          'me_power'])

        self.X = dataset[['trim', 'sog', 'stw', 'wspeedbf', 'wdir']].values
        self.y = dataset['me_power'].values.reshape(-1, 1)
        # Splitting the dataset into the Training set and Test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=1, shuffle=False)

        # Feature Scaling
        self.sc1 = MinMaxScaler()
        self.X_train_scaled = self.sc1.fit_transform(self.X_train[:, :])
        self.X_test_scaled = self.sc1.transform(self.X_test[:, :])
        self.sc2 = MinMaxScaler()
        self.y_train_scaled = self.sc2.fit_transform(self.y_train[:, :])
        self.y_test_scaled = self.sc2.transform(self.y_test[:, :])

    def lstm_function(self, writer, max_epochs=30, _learning_rate=0.001, _sequence_size=5, _batch_size=16,
                      _hidden_layers=1, _try_number=1, patience=5):
        # Training the model on the Training set
        def to_sequences(dataset_x, dataset_y, _sequence_size=1):
            x, y = [], []

            for i in range(len(dataset_x) - _sequence_size - 1):
                window = dataset_x[i:(i + _sequence_size), 0:self.X.shape[1]]
                x.append(window)
                y.append(dataset_y[i + _sequence_size])

            return np.array(x), np.array(y)

        start_time = time.time() * 1000

        # Number of time steps to look back
        # Larger sequences (look further back) may improve forecasting.
        seq_size = _sequence_size

        train_X, train_y = to_sequences(self.X_train_scaled, self.y_train_scaled, seq_size)
        test_X, test_y = to_sequences(self.X_test_scaled, self.y_test_scaled, seq_size)

        model = Sequential()
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
        learning_rate = LearningRateScheduler(lambda _: _learning_rate)

        if _hidden_layers >= 3:
            model.add(LSTM(128, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]),
                           return_sequences=True))
            model.add(LSTM(64, activation='relu', return_sequences=True))
            model.add(LSTM(32, activation='relu', return_sequences=True))
            model.add(LSTM(16, activation='relu', return_sequences=False))
        elif _hidden_layers >= 2:
            model.add(LSTM(128, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]),
                           return_sequences=True))
            model.add(LSTM(64, activation='relu', return_sequences=True))
            model.add(LSTM(32, activation='relu', return_sequences=False))
        elif _hidden_layers >= 1:
            model.add(LSTM(128, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]),
                           return_sequences=True))
            model.add(LSTM(64, activation='relu', return_sequences=False))
        else:
            model.add(LSTM(128, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]),
                           return_sequences=False))

        model.add(Dense(self.y_train.shape[1]))

        model.compile(optimizer='adam', loss='mse')
        model.summary()

        # fit the model
        model.fit(train_X, train_y, epochs=max_epochs, batch_size=_batch_size, validation_data=(test_X, test_y),
                  verbose=0, callbacks=[early_stop, learning_rate])

        train_predict = model.predict(train_X)
        test_predict = model.predict(test_X)

        end_time = time.time() * 1000

        train_score_MAE = float("nan")
        train_score_MSE = float("nan")
        train_score_RMSE = float("nan")
        test_score_MAE = float("nan")
        test_score_MSE = float("nan")
        test_score_RMSE = float("nan")
        if not (np.isnan(train_predict).any() or np.isnan(test_predict).any()):
            train_predict = self.sc2.inverse_transform(train_predict)
            train_y = self.sc2.inverse_transform(train_y)
            test_predict = self.sc2.inverse_transform(test_predict)
            test_y = self.sc2.inverse_transform(test_y)

            train_score_MAE = mean_absolute_error(train_y, train_predict)
            train_score_MSE = mean_squared_error(train_y, train_predict)
            train_score_RMSE = math.sqrt(mean_squared_error(train_y, train_predict))
            test_score_MAE = mean_absolute_error(test_y, test_predict)
            test_score_MSE = mean_squared_error(test_y, test_predict)
            test_score_RMSE = math.sqrt(mean_squared_error(test_y, test_predict))

        writer.writerow({'learning_rate': _learning_rate, 'sequence_size': _sequence_size, 'batch_size': _batch_size,
                         'hidden_layers': _hidden_layers, 'train_score_MAE': train_score_MAE,
                         'train_score_MSE': train_score_MSE, 'train_score_RMSE': train_score_RMSE,
                         'test_score_MAE': test_score_MAE, 'test_score_MSE': test_score_MSE,
                         'test_score_RMSE': test_score_RMSE, 'epoches_stopped': early_stop.stopped_epoch,
                         'exec_time': end_time - start_time, 'try': _try_number})

    def run_lstm(self, _learning_rates=None, _sequence_sizes=None, _batch_sizes=None,
                 _hidden_layers=None, tries_number=1, max_epoches_number=30, patience=5):

        if _hidden_layers is None:
            _hidden_layers = [0, 1, 2, 3]
        if _batch_sizes is None:
            _batch_sizes = [16, 32, 64]
        if _sequence_sizes is None:
            _sequence_sizes = [5, 10, 15]
        if _learning_rates is None:
            _learning_rates = [0.1, 0.01, 0.001]

        columns = ['learning_rate', 'sequence_size', 'batch_size', 'hidden_layers', 'train_score_MAE',
                   'train_score_MSE', 'train_score_RMSE', 'test_score_MAE', 'test_score_MSE', 'test_score_RMSE',
                   'epoches_stopped', 'exec_time', 'try']

        _filename = 'Comparison_Data_{}_NO_DROPOUT.csv'.format(self.filename)
        with open(_filename, 'a', encoding='utf-8', newline='') as _output_file:
            writer = csv.DictWriter(_output_file, fieldnames=columns)
            writer.writeheader()

            # for Double LSTM
            for learning_rate in _learning_rates:
                for sequence_size in _sequence_sizes:
                    for batch_size in _batch_sizes:
                        for hidden_layer in _hidden_layers:
                            for tryNum in range(1, tries_number + 1):
                                self.lstm_function(writer, max_epoches_number, learning_rate, sequence_size,
                                                   batch_size, hidden_layer, tryNum, patience)


if __name__ == '__main__':
    dataset_nums = [1, 2, 3, 5, 6, 7]
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    sequence_sizes = [5, 10, 15]
    batch_sizes = [16, 32, 64]
    hidden_layers = [0, 1, 2, 3]
    for dataset_num in dataset_nums:
        test = LSTMPreparation('dataset_{}'.format(dataset_num), 0.1)
        test.run_lstm(_learning_rates=learning_rates, _sequence_sizes=sequence_sizes, _batch_sizes=batch_sizes,
                      _hidden_layers=hidden_layers, tries_number=4, max_epoches_number=10, patience=5)
    print('Comparison END')
