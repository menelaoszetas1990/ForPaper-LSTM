# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


class Figure4:
    def __init__(self):
        self.X_test_set = []
        self.y_test_test = []

    def create_test_dataset(self, _dataset_filenames):
        for dataset_filename in _dataset_filenames:
            _dataset = pd.read_csv('../../data/' + dataset_filename + '.csv', usecols=['trim', 'sog', 'stw', 'wspeedbf',
                                                                                       'wdir', 'me_power'])

            df_X = _dataset[['trim', 'sog', 'stw', 'wspeedbf', 'wdir']].values
            df_y = _dataset['me_power'].values.reshape(-1, 1)
            # Splitting the dataset into the Training set and Test set
            X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=.1, random_state=1, shuffle=False)

            # Feature Scaling
            sc = MinMaxScaler()
            sc.fit_transform(X_train[:, :])
            X_test_scaled = sc.transform(X_test[:, :])
            sc2 = MinMaxScaler()
            sc2.fit_transform(y_train[:, :])
            y_test_scaled = sc2.transform(y_test[:, :])
            self.X_test_set.append(X_test_scaled[-50:, :])
            self.y_test_test.append(y_test_scaled[-50:, :])

    @staticmethod
    def create_models(_dataset_filenames, _max_epochs=10, _learning_rate=0.001, _sequence_size=10, _batch_size=16,
                      _hidden_layers=0):
        # Training the model on the Training set
        def to_sequences(dataset_x, dataset_y, _sequence_size=1):
            x, y = [], []

            for i in range(len(dataset_x) - _sequence_size - 1):
                window = dataset_x[i:(i + _sequence_size), 0:df_X.shape[1]]
                x.append(window)
                y.append(dataset_y[i + _sequence_size])

            return np.array(x), np.array(y)

        for dataset_filename in _dataset_filenames:
            _dataset = pd.read_csv('../../data/' + dataset_filename + '.csv', usecols=['trim', 'sog', 'stw', 'wspeedbf',
                                                                                       'wdir', 'me_power'])

            df_X = _dataset[['trim', 'sog', 'stw', 'wspeedbf', 'wdir']].values
            df_y = _dataset['me_power'].values.reshape(-1, 1)
            # Splitting the dataset into the Training set and Test set
            X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=.1, random_state=1, shuffle=False)

            # Feature Scaling
            sc = MinMaxScaler()
            X_train_scaled = sc.fit_transform(X_train[:, :])
            X_test_scaled = sc.transform(X_test[:, :])
            sc2 = MinMaxScaler()
            y_train_scaled = sc2.fit_transform(y_train[:, :])
            y_test_scaled = sc2.transform(y_test[:, :])

            train_X, train_y = to_sequences(X_train_scaled, y_train_scaled, _sequence_size)
            test_X, test_y = to_sequences(X_test_scaled, y_test_scaled, _sequence_size)

            model = Sequential()
            lr = LearningRateScheduler(lambda _: _learning_rate)

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

            model.add(Dense(y_train.shape[1]))

            model.compile(optimizer='adam', loss='mse')
            model.summary()

            # fit the model
            model.fit(train_X, train_y, epochs=_max_epochs, batch_size=_batch_size,
                      validation_data=(test_X, test_y), verbose=0, callbacks=[lr])

            name = '{}_LR_{}_SS_{}_BS_{}_HL_{}' \
                .format(dataset_filename, _learning_rate, _sequence_size, _batch_size, _hidden_layers)
            model.save('models/' + name)


dataset_nums = [1, 2, 3, 5, 6, 7]
learning_rate = 0.001
hidden_layers = 0
sequence_size = 10

if __name__ == '__main__':
    dataset_filenames = []
    for dataset_num in dataset_nums:
        dataset_filenames.append('dataset_{}'.format(dataset_num))

    test = Figure4()
    test.create_test_dataset(dataset_filenames)
    Figure4.create_models(_dataset_filenames=dataset_filenames, _learning_rate=learning_rate,
                          _hidden_layers=hidden_layers, _sequence_size=sequence_size)

    print('END')
