# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import LearningRateScheduler
from settings import dataset_nums, learning_rate, sequence_size, batch_size, hidden_layers_separate_models
from settings import test_data_filename, number_of_rounds
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pickle import dump


def to_sequences(dataset_x, dataset_y, _sequence_size=1):
    x, y = [], []

    for i in range(len(dataset_x) - _sequence_size - 1):
        window = dataset_x[i:(i + _sequence_size), 0:dataset_x.shape[1]]
        x.append(window)
        y.append(dataset_y[i + _sequence_size])

    return np.array(x), np.array(y)


class Score4:
    dataset = pd.read_csv(test_data_filename, usecols=['trim', 'sog', 'stw', 'wspeedbf', 'wdir', 'me_power'])
    X_test = dataset[['trim', 'sog', 'stw', 'wspeedbf', 'wdir']].values
    y_test = dataset['me_power'].values.reshape(-1, 1)
    # Feature Scaling
    sc1 = MinMaxScaler()
    X_test_scaled = sc1.fit_transform(X_test[:, :])
    sc2 = MinMaxScaler()
    sc2.fit_transform(y_test[:, :])
    test_X, test_y = to_sequences(X_test_scaled, y_test, sequence_size)

    def __init__(self):
        self.global_weights = None
        pass

    @staticmethod
    def global_model_initialization():
        model = Sequential()

        if hidden_layers_separate_models >= 3:
            model.add(LSTM(128, activation='relu', input_shape=(Score4.test_X.shape[1], Score4.test_X.shape[2]),
                           return_sequences=True))
            model.add(LSTM(64, activation='relu', return_sequences=True))
            model.add(LSTM(32, activation='relu', return_sequences=True))
            model.add(LSTM(16, activation='relu', return_sequences=False))
        elif hidden_layers_separate_models >= 2:
            model.add(LSTM(128, activation='relu', input_shape=(Score4.test_X.shape[1], Score4.test_X.shape[2]),
                           return_sequences=True))
            model.add(LSTM(64, activation='relu', return_sequences=True))
            model.add(LSTM(32, activation='relu', return_sequences=False))
        elif hidden_layers_separate_models >= 1:
            model.add(LSTM(128, activation='relu', input_shape=(Score4.test_X.shape[1], Score4.test_X.shape[2]),
                           return_sequences=True))
            model.add(LSTM(64, activation='relu', return_sequences=False))
        else:
            model.add(LSTM(128, activation='relu', input_shape=(Score4.test_X.shape[1], Score4.test_X.shape[2]),
                           return_sequences=False))

        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model.get_weights()

    @staticmethod
    def run_separate_model(_dataset_filename, _global_weights, _max_epochs=10, _learning_rate=0.001, _sequence_size=10,
                           _batch_size=16, _hidden_layers=0):
        _dataset = pd.read_csv('../../data/' + _dataset_filename + '.csv', usecols=['trim', 'sog', 'stw', 'wspeedbf',
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
        model.set_weights(_global_weights)
        # model.summary()

        # fit the model
        model.fit(train_X, train_y, epochs=_max_epochs, batch_size=_batch_size, verbose=0, callbacks=[lr],
                  validation_data=(test_X, test_y))

        return model.get_weights()

    @staticmethod
    def global_model_run(_global_weights, _max_epochs=10, _learning_rate=0.001, _sequence_size=10,
                         _batch_size=16, _hidden_layers=0):
        model = Sequential()

        if _hidden_layers >= 3:
            model.add(LSTM(128, activation='relu', input_shape=(Score4.test_X.shape[1], Score4.test_X.shape[2]),
                           return_sequences=True))
            model.add(LSTM(64, activation='relu', return_sequences=True))
            model.add(LSTM(32, activation='relu', return_sequences=True))
            model.add(LSTM(16, activation='relu', return_sequences=False))
        elif _hidden_layers >= 2:
            model.add(LSTM(128, activation='relu', input_shape=(Score4.test_X.shape[1], Score4.test_X.shape[2]),
                           return_sequences=True))
            model.add(LSTM(64, activation='relu', return_sequences=True))
            model.add(LSTM(32, activation='relu', return_sequences=False))
        elif _hidden_layers >= 1:
            model.add(LSTM(128, activation='relu', input_shape=(Score4.test_X.shape[1], Score4.test_X.shape[2]),
                           return_sequences=True))
            model.add(LSTM(64, activation='relu', return_sequences=False))
        else:
            model.add(LSTM(128, activation='relu', input_shape=(Score4.test_X.shape[1], Score4.test_X.shape[2]),
                           return_sequences=False))

        model.add(Dense(Score4.test_y.shape[1]))

        model.compile(optimizer='adam', loss='mse')
        model.set_weights(_global_weights)

        name = 'Fed_Avg_LR_{}_SS_{}_BS_{}_HL_{}' \
            .format(_learning_rate, _sequence_size, _batch_size, _hidden_layers)
        model.save('models/' + name)
        # save the scalers
        dump(Score4.sc1, open('scalers/' + name + '_sc1', 'wb'))
        dump(Score4.sc2, open('scalers/' + name + '_sc2', 'wb'))

        # fit the model
        test_predict = model.predict(Score4.test_X)
        if not (np.isnan(test_predict).any()):
            test_predict = Score4.sc2.inverse_transform(test_predict)
            test_predicts = dict()
            test_predicts['Pactual'] = Score4.y_test[sequence_size + 1:]
            test_predicts['Ppred'] = test_predict
            return [mean_squared_error(Score4.test_y, test_predict), mean_absolute_error(Score4.test_y, test_predict), \
                   test_predicts]


def run_score_4():
    dataset_filenames = []
    for dataset_num in dataset_nums:
        dataset_filenames.append('dataset_{}'.format(dataset_num))

    test_dataset = Score4()
    test_dataset.global_weights = Score4.global_model_initialization()

    for counter in range(number_of_rounds):
        separate_models_weights = dict()
        for dataset_filename in dataset_filenames:
            separate_models_weights[dataset_filename] = \
                test_dataset.run_separate_model(dataset_filename, test_dataset.global_weights, _max_epochs=10,
                                                _learning_rate=learning_rate, _sequence_size=sequence_size,
                                                _batch_size=batch_size, _hidden_layers=hidden_layers_separate_models)

        # iteration to average them
        for i in range(len(test_dataset.global_weights)):
            for j in range(len(test_dataset.global_weights[i])):
                weight_sum = 0
                for dataset_filename in dataset_filenames:
                    weight_sum += separate_models_weights[dataset_filename][i][j]
                test_dataset.global_weights[i][j] = weight_sum/len(dataset_filenames)

    print('END Score_4')
    return Score4.global_model_run(test_dataset.global_weights, _max_epochs=10, _learning_rate=learning_rate,
                                   _sequence_size=sequence_size, _batch_size=batch_size,
                                   _hidden_layers=hidden_layers_separate_models)
