# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.python.keras.models import load_model
from settings import learning_rate, sequence_size, batch_size, hidden_layers_hyper_models
from pickle import load


def to_sequences(dataset_x, dataset_y, _sequence_size=1):
    x, y = [], []
    for i in range(len(dataset_x) - _sequence_size - 1):
        window = dataset_x[i:(i + _sequence_size), 0:dataset_x.shape[1]]
        x.append(window)
        y.append(dataset_y[i + _sequence_size])
    return np.array(x), np.array(y)


class Score3:
    y_test = None

    def __init__(self, _model_name, _test_dataset_filename):
        dataset = pd.read_csv(_test_dataset_filename, usecols=['trim', 'sog', 'stw', 'wspeedbf', 'wdir', 'me_power'])
        self.X_test = dataset[['trim', 'sog', 'stw', 'wspeedbf', 'wdir']].values
        Score3.y_test = dataset['me_power'].values.reshape(-1, 1)
        self.model = load_model('../Figure4_Ensamble_vs_Collaborative_vs_Centralized_vs_Federated_learning/models/{}'
                                .format(_model_name))
        self.sc1 = load(open('../Figure4_Ensamble_vs_Collaborative_vs_Centralized_vs_Federated_learning/scalers/{}_sc1'
                             .format(_model_name), 'rb'))
        self.sc2 = load(open('../Figure4_Ensamble_vs_Collaborative_vs_Centralized_vs_Federated_learning/scalers/{}_sc2'
                             .format(_model_name), 'rb'))
        # scale x test data
        self.X_test_scaled = self.sc1.transform(self.X_test[:, :])
        self.test_X, self.test_y = to_sequences(self.X_test_scaled, Score3.y_test[:, :], sequence_size)

    def model_run(self):
        test_predict = self.model.predict(self.test_X)
        if not (np.isnan(test_predict).any()):
            test_predict = self.sc2.inverse_transform(test_predict)
            return [mean_squared_error(self.test_y, test_predict), mean_absolute_error(self.test_y, test_predict)]


def run_score_3(_test_dataset_filename):
    model_name = 'Hyper_Dataset_LR_{}_SS_{}_BS_{}_HL_{}'.format(learning_rate, sequence_size, batch_size,
                                                                hidden_layers_hyper_models)
    test_dataset = Score3(model_name, _test_dataset_filename)

    print('END Score_3')
    return test_dataset.model_run()
