# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import LearningRateScheduler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


class Figure3:
    def __init__(self, filename, test_size, _ship_index):
        # Preprocessing
        # Importing the dataset
        # Using columns: sog, stw, wspeedbf, wdir, me_power
        self.filename = filename
        self.ship_index = _ship_index
        _dataset = pd.read_csv('../../data/' + filename + '.csv', usecols=['sog', 'stw', 'wspeedbf', 'wdir',
                                                                           'me_power'])

        self.X = _dataset[['sog', 'stw', 'wspeedbf', 'wdir']].values
        self.y = _dataset['me_power'].values.reshape(-1, 1)
        # Splitting the dataset into the Training set and Test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=1, shuffle=False)

        # Feature Scaling
        self.sc = MinMaxScaler()
        self.X_train_scaled = self.sc.fit_transform(self.X_train[:, :])
        self.X_test_scaled = self.sc.transform(self.X_test[:, :])
        self.sc2 = MinMaxScaler()
        self.y_train_scaled = self.sc2.fit_transform(self.y_train[:, :])
        self.y_test_scaled = self.sc2.transform(self.y_test[:, :])

    def lstm_run(self, _max_epochs=30, _learning_rate=0.001, _sequence_size=10, _batch_size=16, _hidden_layers=1,
                 _future_predict=1):
        # Training the model on the Training set
        def to_sequences(dataset_x, dataset_y, _sequence_size=1):
            x, y = [], []

            for i in range(len(dataset_x) - _sequence_size - 1 - _future_predict - 1):
                window = dataset_x[i:(i + _sequence_size), 0:self.X.shape[1]]
                x.append(window)
                y.append(dataset_y[i + _sequence_size + _future_predict - 1])

            return np.array(x), np.array(y)

        train_X, train_y = to_sequences(self.X_train_scaled, self.y_train_scaled, _sequence_size)
        test_X, test_y = to_sequences(self.X_test_scaled, self.y_test_scaled, _sequence_size)

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

        model.add(Dense(self.y_train.shape[1]))

        model.compile(optimizer='adam', loss='mse')
        model.summary()

        # fit the model
        model.fit(train_X, train_y, epochs=_max_epochs, batch_size=_batch_size,
                  validation_data=(test_X, test_y), verbose=0, callbacks=[lr])

        name = '{}_LR_{}_SS_{}_BS_{}_HL_{}' \
            .format(self.filename, _learning_rate, _sequence_size, _batch_size, _hidden_layers)
        model.save('models/' + name)

        test_predict = model.predict(test_X)

        test_score_MSE = float("nan")
        # test_score_MSE_transformed = float("nan")
        if not np.isnan(test_predict).any():
            # test_predict_transformed = self.sc2.inverse_transform(test_predict)
            # test_y_transformed = self.sc2.inverse_transform(test_y)

            test_score_MSE = mean_squared_error(test_y, test_predict)
            # test_score_MSE_transformed = mean_squared_error(test_y_transformed, test_predict_transformed)

        return {'sequence': _sequence_size, 'ship': 'Ship {}'.format(self.ship_index), 'c': _future_predict,
                'test_score_MSE': test_score_MSE}


if __name__ == '__main__':
    best_lstm_setup = pd.read_csv(
        '../Step2_Getting_the_Best_LSTM_Setup/Comparison_Data_results_NO_DROPOUT.csv', header=0)
    dataset_nums = [1, 2, 3, 5, 6, 7]
    learning_rate = 0.001
    hidden_layer = 1
    sequence_sizes = [5, 10, 15]
    future_predicts = [1, 3, 5]
    fig_names = ['(a)', '(b)', '(c)']
    # for test
    # dataset_nums = [1, 2]
    # learning_rates = [0.01]

    history_dict = []
    for sequence_size in sequence_sizes:
        for idx, dataset in enumerate(dataset_nums):
            ship_index = idx + 1
            test = Figure3('dataset_{}'.format(dataset), 0.1, _ship_index=ship_index)
            for future_predict in future_predicts:
                history_dict.append(test.lstm_run(_max_epochs=10, _learning_rate=learning_rate,
                                                  _sequence_size=sequence_size,
                                                  _batch_size=best_lstm_setup.head(1)['batch_size'][0],
                                                  _hidden_layers=hidden_layer,
                                                  _future_predict=future_predict))

    df = pd.DataFrame.from_dict(history_dict)

    fig, axes = plt.subplots(3, 1, figsize=(30, 20), subplot_kw=dict(ylim=(0, 0.08)))
    plt.grid(True)
    plt.subplots_adjust(bottom=0.25, wspace=0.4, hspace=0.4)
    for idx, sequence_size in enumerate(sequence_sizes):
        dataframe = df[df['sequence'] == sequence_size]
        dataframe.set_index('ship')
        sns.barplot(x="ship", y="test_score_MSE", hue="c", data=dataframe, palette="muted", ax=axes[idx])
        axes[idx].set_xlabel("Ships", fontdict={'fontsize': 32})
        axes[idx].set_ylabel("Training loss MSE", fontdict={'fontsize': 32})
        axes[idx].tick_params(axis='both', which='major', labelsize=28)
        axes[idx].tick_params(axis='both', which='minor', labelsize=28)
        axes[idx].set_title('{}'.format(fig_names[idx]), fontsize=28)
        axes[idx].set(xlabel=None)
        if idx == len(sequence_sizes) - 1:
            axes[idx].legend(fontsize=28, loc=8, bbox_to_anchor=(0.5, -1), title='Pro-activeness', title_fontsize=28)
        else:
            axes[idx].get_legend().remove()
    fig.savefig('plots/Figure-3-Impact_of_Lookback_and_Proactiveness_Ability_NO_DROPOUT.eps',
                format='eps')
    fig.savefig('../plots/Figure3/Figure-3-Impact_of_Lookback_and_Proactiveness_Ability_NO_DROPOUT.eps',
                format='eps')

    print('END')
