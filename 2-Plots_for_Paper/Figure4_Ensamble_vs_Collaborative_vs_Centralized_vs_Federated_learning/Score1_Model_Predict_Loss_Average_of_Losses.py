# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from settings import dataset_nums, learning_rate, sequence_size, batch_size, hidden_layers_separate_models
from settings import test_data_filename
from tensorflow.python.keras.models import load_model
from pickle import load


def to_sequences(dataset_x, dataset_y, _sequence_size=1):
    x, y = [], []

    for i in range(len(dataset_x) - _sequence_size - 1):
        window = dataset_x[i:(i + _sequence_size), 0:dataset_x.shape[1]]
        x.append(window)
        y.append(dataset_y[i + _sequence_size])

    return np.array(x), np.array(y)


class Score1:
    dataset = pd.read_csv(test_data_filename, usecols=['trim', 'sog', 'stw', 'wspeedbf', 'wdir', 'me_power'])
    X_test = dataset[['trim', 'sog', 'stw', 'wspeedbf', 'wdir']].values
    y_test = dataset['me_power'].values.reshape(-1, 1)

    def __init__(self, _model_name):
        self.model = load_model('models/{}'.format(_model_name))
        self.sc1 = load(open('scalers/{}_sc1'.format(_model_name), 'rb'))
        self.sc2 = load(open('scalers/{}_sc2'.format(_model_name), 'rb'))
        # scale x test data
        self.X_test_scaled = self.sc1.transform(Score1.X_test[:, :])
        self.test_X, self.test_y = to_sequences(self.X_test_scaled, Score1.y_test[:, :], sequence_size)

    def model_run(self):
        test_predict = self.model.predict(self.test_X)
        if not (np.isnan(test_predict).any()):
            test_predict = self.sc2.inverse_transform(test_predict)
            return mean_squared_error(self.test_y, test_predict), mean_absolute_error(self.test_y, test_predict), \
                   Score1.y_test[sequence_size:], test_predict


def run_score_1():
    dataset_filenames = []
    losses_MSE = []
    losses_MAE = []
    test_predicts = dict()
    for idx, dataset_num in enumerate(dataset_nums):
        dataset_filenames.append('dataset_{}'.format(dataset_num))
        model_name = '{}_LR_{}_SS_{}_BS_{}_HL_{}'.format(dataset_filenames[idx], learning_rate, sequence_size,
                                                         batch_size, hidden_layers_separate_models)

        test_dataset = Score1(model_name)
        losses = test_dataset.model_run()
        losses_MSE.append(losses[0])
        losses_MAE.append(losses[1])
        test_predicts['Pactual'] = losses[2]
        test_predicts['Ppredict{}'.format(idx)] = losses[3]

    print('END Score_1')
    return [sum(losses_MSE) / len(dataset_filenames), sum(losses_MAE) / len(dataset_filenames), test_predicts]
