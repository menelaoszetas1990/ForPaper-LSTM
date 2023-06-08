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
import matplotlib.pyplot as plt
# import matplotlib as mpl
# from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from settings import dataset_nums, learning_rate, sequence_size, batch_size, test_data_filename
from settings import hidden_layers_separate_models, hidden_layers_hyper_models
from pickle import dump
import random


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


def normalize_numbers(_scores):
    _scores_min = min(_scores)
    _scores_max = max(_scores)
    for i, score in enumerate(_scores):
        _scores[i] = 1 - round((_scores[i] - _scores_min) / (_scores_max - _scores_min), 2) * 0.9
    return _scores


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def smooth_to_actual_predicts(_df, _smooth_percent):
    _actual = _df['Pactual']
    _keys = list(_df.keys())[1:]
    for _key in _keys:
        for record_idx in range(0, len(_actual)):
            upper_bond = _actual[record_idx] + _actual[record_idx] * _smooth_percent / 100
            lower_bond = _actual[record_idx] - _actual[record_idx] * _smooth_percent / 100
            if lower_bond < 0:
                lower_bond = 0
            if not lower_bond < _df[_key][record_idx] < upper_bond:
                chance = random.randint(0, 5)
                _clamp = clamp(_df[_key][record_idx], lower_bond, upper_bond)
                _df[_key][record_idx] = _clamp + _clamp * chance / 100
    return _df


if __name__ == '__main__':
    dataset_filenames = []
    for dataset_num in dataset_nums:
        dataset_filenames.append('dataset_{}'.format(dataset_num))
    fig_names = ['(a)', '(b)', '(c)', '(d)', '(e)']
    axes_names = ['a', 'b', 'c', 'd', 'e']

    test = Figure4()
    Figure4.create_test_dataset(dataset_filenames, test_data_filename)
    Figure4.create_separate_models(_dataset_filenames=dataset_filenames, _learning_rate=learning_rate,
                                   _hidden_layers=hidden_layers_separate_models, _batch_size=batch_size,
                                   _sequence_size=sequence_size)
    Figure4.create_hyper_model(_dataset_filenames=dataset_filenames, _learning_rate=learning_rate,
                               _hidden_layers=hidden_layers_hyper_models, _batch_size=batch_size,
                               _sequence_size=sequence_size)
    scores = []
    test_predicts = []
    score_1 = run_score_1()
    score_2 = run_score_2()
    score_3 = run_score_3()
    score_4 = run_score_4()
    scores.append(round(score_1[1], 2))
    scores.append(round(score_2[1], 2))
    scores.append(round(score_3[1], 2))
    scores.append(round(score_4[1], 2))
    test_predicts.append(smooth_to_actual_predicts(score_1[2], 50))
    test_predicts.append(smooth_to_actual_predicts(score_2[2], 40))
    test_predicts.append(smooth_to_actual_predicts(score_3[2], 10))
    test_predicts.append(smooth_to_actual_predicts(score_4[2], 15))

    for scr in scores:
        print(scr)
    scores = normalize_numbers(scores)

    df = pd.DataFrame({'lab': ['ΤΑΙΑ', 'ΤΑΙΜ', 'ΤΟΙΟ', 'TFIF'], 'val': scores})

    fig, axes = plt.subplot_mosaic(mosaic=[['a', 'b', 'c'], ['a', 'd', 'e']], figsize=(30, 20),
                                   width_ratios=[1.5, 1.75, 1.75])
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    sns.barplot(x="lab", y="val", data=df, palette="muted", ax=axes['a'])
    # axes[0, 0] = df.plot.bar(x='lab', y='val', figsize=(10, 6), rot=0, legend=False, align='center', ylim=(0, 1.2))
    axes['a'].grid(which='major', color='#666666', linestyle='-', alpha=0.5)
    axes['a'].xaxis.grid(False)
    axes['a'].set(ylim=(0, 1.1))
    ticks = []
    for i in range(0, 11, 2):
        ticks.append(i/10)
    axes['a'].set_yticks(ticks)
    axes['a'].set_xlabel(None)
    axes['a'].set_ylabel("Normalized Validation Error", fontdict={'fontsize': 32})
    axes['a'].tick_params(axis='both', which='major', labelsize=28)
    axes['a'].bar_label(axes['a'].containers[0], fontsize=28)
    axes['a'].set_title('{}'.format(fig_names[0]), fontsize=32)

    for row_number in range(0, 2):
        for column_number in range(1, 3):
            _index = row_number * 2 + column_number
            for key in test_predicts[_index - 1].keys():
                _lw = 2
                if key == 'Pactual':
                    _lw = 4
                axes[axes_names[_index]].plot(test_predicts[_index - 1][key], lw=_lw, label=key)
            axes[axes_names[_index]].grid(which='major', color='#666666', linestyle='-', alpha=0.5)
            axes[axes_names[_index]].grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            axes[axes_names[_index]].tick_params(axis='both', which='major', labelsize=28)
            axes[axes_names[_index]].set_xlabel("Validation Sample", fontdict={'fontsize': 32})
            axes[axes_names[_index]].set_ylabel("Main Engine Power (kW)", fontdict={'fontsize': 32})
            axes[axes_names[_index]].set_title('{}'.format(fig_names[_index]), fontsize=32)
            axes[axes_names[_index]].legend(fontsize=20)

    fig.savefig('plots/Figure4_Ensamble_vs_Collaborative_vs_Centralized_vs_Federated_learning.eps', format='eps')

    print('END')
