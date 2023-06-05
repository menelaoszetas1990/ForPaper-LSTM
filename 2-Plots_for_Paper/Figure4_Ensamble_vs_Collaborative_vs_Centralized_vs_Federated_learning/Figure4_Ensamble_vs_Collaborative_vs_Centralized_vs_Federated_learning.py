# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import LearningRateScheduler
from Score1_Model_Predict_Loss_Average_of_Losses import run_score_1
from Score2_Model_Predict_Average_of_Predicts_Loss import run_score_2
from Score3_Centralized_Learning import run_score_3
from Score4_Federated_Learning import run_score_4
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from matplotlib.colors import LinearSegmentedColormap
# import seaborn as sns
from settings import dataset_nums, learning_rate, sequence_size, batch_size, test_data_filename
from settings import hidden_layers_separate_models, hidden_layers_hyper_models
from pickle import dump


class Figure4:
    def __init__(self):
        pass

    @staticmethod
    def create_test_dataset(_dataset_filenames, _test_data_filename):
        test_dataset = pd.DataFrame()
        for dataset_filename in _dataset_filenames:
            _dataset = pd.read_csv('../../data/' + dataset_filename + '.csv', usecols=['trim', 'sog', 'stw', 'wspeedbf',
                                                                                       'wdir', 'me_power'])
            test_dataset = pd.concat([test_dataset, _dataset.tail(50)], ignore_index=True)
            test_dataset.to_csv(_test_data_filename, index=False)

    @staticmethod
    def create_separate_models(_dataset_filenames, _max_epochs=10, _learning_rate=0.001, _sequence_size=10,
                               _batch_size=16, _hidden_layers=0):
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
            sc1 = MinMaxScaler()
            X_train_scaled = sc1.fit_transform(X_train[:, :])
            X_test_scaled = sc1.transform(X_test[:, :])
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
            # model.summary()

            # fit the model
            model.fit(train_X, train_y, epochs=_max_epochs, batch_size=_batch_size,
                      validation_data=(test_X, test_y), verbose=0, callbacks=[lr])

            name = '{}_LR_{}_SS_{}_BS_{}_HL_{}' \
                .format(dataset_filename, _learning_rate, _sequence_size, _batch_size, _hidden_layers)
            model.save('models/' + name)
            # save the scalers
            dump(sc1, open('scalers/' + name + '_sc1', 'wb'))
            dump(sc2, open('scalers/' + name + '_sc2', 'wb'))

    @staticmethod
    def create_hyper_model(_dataset_filenames, _max_epochs=10, _learning_rate=0.001, _sequence_size=10, _batch_size=16,
                           _hidden_layers=0):
        # Training the model on the Training set
        def to_sequences(dataset_x, dataset_y, _sequence_size=1):
            x, y = [], []

            for i in range(len(dataset_x) - _sequence_size - 1):
                window = dataset_x[i:(i + _sequence_size), 0:df_X.shape[1]]
                x.append(window)
                y.append(dataset_y[i + _sequence_size])

            return np.array(x), np.array(y)

        hyper_dataset = pd.DataFrame()
        for dataset_filename in _dataset_filenames:
            _dataset = pd.read_csv('../../data/' + dataset_filename + '.csv', usecols=['trim', 'sog', 'stw', 'wspeedbf',
                                                                                       'wdir', 'me_power'])
            hyper_dataset = pd.concat([hyper_dataset, _dataset], ignore_index=True)

        df_X = hyper_dataset[['trim', 'sog', 'stw', 'wspeedbf', 'wdir']].values
        df_y = hyper_dataset['me_power'].values.reshape(-1, 1)

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=.1, random_state=1, shuffle=False)

        # Feature Scaling
        sc1 = MinMaxScaler()
        X_train_scaled = sc1.fit_transform(X_train[:, :])
        X_test_scaled = sc1.transform(X_test[:, :])
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
        # model.summary()

        # fit the model
        model.fit(train_X, train_y, epochs=_max_epochs, batch_size=_batch_size,
                  validation_data=(test_X, test_y), verbose=0, callbacks=[lr])

        name = 'Hyper_Dataset_LR_{}_SS_{}_BS_{}_HL_{}' \
            .format(_learning_rate, _sequence_size, _batch_size, _hidden_layers)
        model.save('models/' + name)
        # save the scalers
        dump(sc1, open('scalers/' + name + '_sc1', 'wb'))
        dump(sc2, open('scalers/' + name + '_sc2', 'wb'))


if __name__ == '__main__':
    dataset_filenames = []
    for dataset_num in dataset_nums:
        dataset_filenames.append('dataset_{}'.format(dataset_num))

    test = Figure4()
    Figure4.create_test_dataset(dataset_filenames, test_data_filename)
    Figure4.create_separate_models(_dataset_filenames=dataset_filenames, _learning_rate=learning_rate,
                                   _hidden_layers=hidden_layers_separate_models, _batch_size=batch_size,
                                   _sequence_size=sequence_size)
    Figure4.create_hyper_model(_dataset_filenames=dataset_filenames, _learning_rate=learning_rate,
                               _hidden_layers=hidden_layers_hyper_models, _batch_size=batch_size,
                               _sequence_size=sequence_size)
    score_1 = run_score_1()
    print(score_1)
    score_2 = run_score_2()
    print(score_2)
    score_3 = run_score_3()
    print(score_3)
    score_4 = run_score_4()
    print(score_4)

    print('END')
