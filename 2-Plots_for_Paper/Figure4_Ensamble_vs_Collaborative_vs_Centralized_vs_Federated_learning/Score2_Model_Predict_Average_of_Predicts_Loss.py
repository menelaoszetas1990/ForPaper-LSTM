# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.models import load_model
from settings import dataset_nums, learning_rate
from settings import sequence_size, batch_size
from settings import hidden_layers_separate_models
from settings import test_data_filename
from pickle import load


def to_sequences(dataset_x, dataset_y, _sequence_size=1):
    x, y = [], []

    for i in range(len(dataset_x) - _sequence_size - 1):
        window = dataset_x[i:(i + _sequence_size), 0:dataset_x.shape[1]]
        x.append(window)
        y.append(dataset_y[i + _sequence_size])

    return np.array(x), np.array(y)


class Score2:
    dataset = pd.read_csv(test_data_filename, usecols=['trim', 'sog', 'stw', 'wspeedbf', 'wdir', 'me_power'])
    X_test = dataset[['trim', 'sog', 'stw', 'wspeedbf', 'wdir']].values
    y_test = dataset['me_power'].values.reshape(-1, 1)
    test_y = {}

    def __init__(self, _model_name):
        self.model = load_model('models/{}'.format(_model_name))
        self.sc1 = load(open('scalers/{}_sc1'.format(_model_name), 'rb'))
        self.sc2 = load(open('scalers/{}_sc2'.format(_model_name), 'rb'))
        # scale x test data
        self.X_test_scaled = self.sc1.transform(Score2.X_test[:, :])
        self.test_X, self.test_y = to_sequences(self.X_test_scaled, Score2.y_test[:, :], sequence_size)
        Score2.test_y = self.test_y

    def model_run(self):
        test_predict = self.model.predict(self.test_X)
        if not (np.isnan(test_predict).any()):
            return self.sc2.inverse_transform(test_predict)


def run_score_2():
    dataset_filenames = []
    losses_MSE = []
    losses_MSE_avg = []
    for idx, dataset_num in enumerate(dataset_nums):
        dataset_filenames.append('dataset_{}'.format(dataset_num))
        model_name = '{}_LR_{}_SS_{}_BS_{}_HL_{}'.format(dataset_filenames[idx], learning_rate, sequence_size,
                                                         batch_size, hidden_layers_separate_models)

        test_dataset = Score2(model_name)
        losses_MSE.append(test_dataset.model_run())

    for i in range(len(losses_MSE[0])):
        _sum = 0
        for idx, dataset_num in enumerate(dataset_nums):
            _sum += losses_MSE[idx][i]
        losses_MSE_avg.append(_sum / len(dataset_nums))

    print('END Score_2')
    return mean_squared_error(Score2.test_y, losses_MSE_avg)
