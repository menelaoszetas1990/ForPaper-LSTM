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
    for i, score in enumerate(_scores):
        _scores[i] = round(_scores_min / score, 2)
    return _scores


if __name__ == '__main__':
    dataset_filenames = []
    for dataset_num in dataset_nums:
        dataset_filenames.append('dataset_{}'.format(dataset_num))
    fig_names = ['(a)', '(b)', '(c)', '(d)', '(e)']

    # test = Figure4()
    # Figure4.create_test_dataset(dataset_filenames, test_data_filename)
    # Figure4.create_separate_models(_dataset_filenames=dataset_filenames, _learning_rate=learning_rate,
    #                                _hidden_layers=hidden_layers_separate_models, _batch_size=batch_size,
    #                                _sequence_size=sequence_size)
    # Figure4.create_hyper_model(_dataset_filenames=dataset_filenames, _learning_rate=learning_rate,
    #                            _hidden_layers=hidden_layers_hyper_models, _batch_size=batch_size,
    #                            _sequence_size=sequence_size)
    scores = []
    scores.append(round(run_score_1()[1], 2))
    scores.append(round(run_score_2()[1], 2))
    scores.append(round(run_score_3()[1], 2))
    scores.append(round(run_score_3()[1] + 100, 2))
    # scores.append(round(run_score_4()[1], 2))
    for scr in scores:
        print(scr)
    scores = normalize_numbers(scores)

    df = pd.DataFrame({'lab': ['ΤΑΙΑ', 'ΤΑΙΜ', 'ΤΟΙΟ', 'TFIF'], 'val': scores})

    fig, axes = plt.subplot_mosaic(mosaic=[['a', 'b', 'c'], ['a', 'd', 'e']], figsize=(30, 20), width_ratios=[1.5, 1.75, 1.75])
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    sns.barplot(x="lab", y="val", data=df, palette="muted", ax=axes['a'])
    # axes[0, 0] = df.plot.bar(x='lab', y='val', figsize=(10, 6), rot=0, legend=False, align='center', ylim=(0, 1.2))
    axes['a'].grid(which='major', color='#666666', linestyle='-', alpha=0.5)
    axes['a'].xaxis.grid(False)
    ticks = []
    for i in range(0, 11, 2):
        ticks.append(i/10)
    axes['a'].set_yticks(ticks)
    axes['a'].set_xlabel("Scores", fontdict={'fontsize': 32})
    axes['a'].set_ylabel("Normalized Validation Error", fontdict={'fontsize': 32})
    axes['a'].tick_params(axis='both', which='major', labelsize=28)
    axes['a'].bar_label(axes['a'].containers[0], fontsize=28)

    # for idx, sequence_size in enumerate(sequence_sizes):
    #     dataframe = df[df['sequence'] == sequence_size]
    #     dataframe.set_index('ship')
    #     sns.barplot(x="ship", y="train_score_MSE", hue="c", data=dataframe, palette="muted", ax=axes[idx, 0])
    #     axes[idx, 0].set(ylim=(0, 0.02))
    #     axes[idx, 0].set_xlabel("Ships", fontdict={'fontsize': 32})
    #     axes[idx, 0].set_ylabel("Training loss MSE", fontdict={'fontsize': 32})
    #     axes[idx, 0].tick_params(axis='both', which='major', labelsize=28)
    #     axes[idx, 0].tick_params(axis='both', which='minor', labelsize=28)
    #     axes[idx, 0].grid(which='major', color='#666666', linestyle='-', alpha=0.5)
    #     axes[idx, 0].xaxis.grid(False)
    #     axes[idx, 0].set_title('{}'.format(fig_names[idx]), fontsize=28)
    #     axes[idx, 0].set(xlabel=None)
    #     if idx == len(sequence_sizes) - 1:
    #         axes[idx, 0].legend(fontsize=28, loc=8, bbox_to_anchor=(0.5, -1.1), title='Pro-activeness',
    #                             title_fontsize=28)
    #     else:
    #         axes[idx, 0].get_legend().remove()



    # plt.grid(True)
    # ax = df.plot.bar(x='lab', y='val', figsize=(10, 6), rot=0, legend=False, align='center', ylim=(0, 1.2))
    # ax.grid(which='major', color='#666666', linestyle='-', alpha=0.5)
    # ax.xaxis.grid(False)
    # ticks = []
    # for i in range(0, 11, 2):
    #     ticks.append(i/10)
    # ax.set_yticks(ticks)
    # ax.set_xlabel("Scores", fontdict={'fontsize': 32})
    # ax.set_ylabel("Normalized Validation Error", fontdict={'fontsize': 32})
    # ax.tick_params(axis='both', which='major', labelsize=28)
    # ax.bar_label(ax.containers[0], fontsize=28)
    # plt.tight_layout()
    fig.savefig('plots/Figure4_Ensamble_vs_Collaborative_vs_Centralized_vs_Federated_learning.eps', format='eps')

    print('END')
