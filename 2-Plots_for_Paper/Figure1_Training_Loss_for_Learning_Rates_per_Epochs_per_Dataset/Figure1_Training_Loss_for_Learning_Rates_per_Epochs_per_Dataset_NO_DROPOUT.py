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
import pickle


class Figure1:
    def __init__(self, filename, test_size):
        # Preprocessing
        # Importing the dataset
        # Using columns: sog, stw, wspeedbf, wdir, me_power
        self.filename = filename
        _dataset = pd.read_csv('../../data/' + filename + '.csv', usecols=['trim', 'sog', 'stw', 'wspeedbf', 'wdir',
                                                                           'me_power'])

        self.X = _dataset[['trim', 'sog', 'stw', 'wspeedbf', 'wdir']].values
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

    def lstm_run(self, _max_epochs=30, _learning_rate=0.001, _sequence_size=10, _batch_size=16, _hidden_layers=1):
        # Training the model on the Training set
        def to_sequences(dataset_x, dataset_y, _sequence_size=1):
            x, y = [], []

            for i in range(len(dataset_x) - _sequence_size - 1):
                window = dataset_x[i:(i + _sequence_size), 0:self.X.shape[1]]
                x.append(window)
                y.append(dataset_y[i + _sequence_size])

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
        history = model.fit(train_X, train_y, epochs=_max_epochs, batch_size=_batch_size,
                            validation_data=(test_X, test_y), verbose=0, callbacks=[lr])

        name = '{}_LR_{}_SS_{}_BS_{}_HL_{}_with_trim' \
            .format(self.filename, _learning_rate, _sequence_size, _batch_size, _hidden_layers)
        model.save('models/' + name)

        # plt.figure()
        # plt.plot(history.history['loss'], label='Training loss')
        # plt.plot(history.history['val_loss'], label='Validation loss')
        # plt.legend()
        # plt.savefig('../plots/Figure1/ValidationTrainLoss_{}_{}.png'.format(_learning_rate, self.filename))

        return history


def storeData(_dictionary, _learning_rates, _ship_number):
    db = []
    for lr_idx, _learning_rate in enumerate(_learning_rates):
        db.append([])
        for ep_idx, epoch in enumerate(_dictionary[_learning_rate].epoch):
            # initializing data to be stored in db
            db[lr_idx].append(0)
            db[lr_idx][ep_idx] = _dictionary[_learning_rate].history['loss'][ep_idx]

    # Its important to use binary mode
    dbfile = open('pickle/Figure_1_ship_{}'.format(_ship_number), 'ab')

    # source, destination
    db = np.array(db)
    pickle.dump(db, dbfile)
    dbfile.close()


if __name__ == '__main__':
    best_lstm_setup = pd.read_csv(
        '../Step2_Getting_the_Best_LSTM_Setup/Comparison_Data_results_NO_DROPOUT.csv', header=0)
    dataset_nums = [1, 2, 3, 5, 6, 7]
    learning_rates = [0.01, 0.001, 0.0001]
    fig_names = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    # for test
    # dataset_nums = [1]
    # learning_rates = [0.01, 0.001]

    history_dict = {}
    for idx, dataset in enumerate(dataset_nums):
        test = Figure1('dataset_{}'.format(dataset), 0.1)
        history_dict[dataset] = {}
        for learning_rate in learning_rates:
            history_dict[dataset][learning_rate] = \
                test.lstm_run(_max_epochs=10, _learning_rate=learning_rate,
                              _sequence_size=best_lstm_setup.head(1)['sequence_size'][0],
                              _batch_size=best_lstm_setup.head(1)['batch_size'][0],
                              _hidden_layers=best_lstm_setup.head(1)['hidden_layers'][0])

    fig, axes = plt.subplots(2, 3, figsize=(30, 20), subplot_kw=dict(xlim=(1, 10), ylim=(0, 0.04)))
    plt.grid(True)
    plt.subplots_adjust(bottom=0.2, wspace=0.4, hspace=0.4)

    for row_number in range(0, axes.shape[0]):
        for column_number in range(0, axes.shape[1]):
            _index = row_number * 3 + column_number
            # for test
            # _index = 0
            for learning_rate in learning_rates:
                axes[row_number, column_number].plot(history_dict[dataset_nums[_index]][learning_rate].history['loss'],
                                                     label='Learning Rate {}'.format(learning_rate), lw=4)
            axes[row_number, column_number].grid(which='major', color='#666666', linestyle='-', alpha=0.5)
            # axes[row_number, column_number].minorticks_on()
            # axes[row_number, column_number].grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            axes[row_number, column_number].set_xticks(range(1, 11, 1))
            axes[row_number, column_number].tick_params(axis='both', which='major', labelsize=28)
            # axes[row_number, column_number].tick_params(axis='both', which='minor', labelsize=28)
            axes[row_number, column_number].set_xlabel("Epochs", fontdict={'fontsize': 32})
            axes[row_number, column_number].set_ylabel("Training loss MSE (ship {})".format(_index+1),
                                                       fontdict={'fontsize': 32})
            if row_number == 1 and column_number == 1:
                axes[row_number, column_number].legend(fontsize=28, loc=8, bbox_to_anchor=(0.5, -0.6))
            axes[row_number, column_number].set_title('{}'.format(fig_names[_index]), fontsize=28)

            storeData(history_dict[dataset_nums[_index]], learning_rates, _index)

    # fig.savefig('plots/Figure1_Training_Loss_for_Learning_Rates_per_Epochs_per_Dataset_NO_DROPOUT.eps',
    #             format='eps')
    # fig.savefig('../plots/Figure1/Figure1_Training_Loss_for_Learning_Rates_per_Epochs_per_Dataset_NO_DROPOUT.eps',
    #             format='eps')

    print('END')
