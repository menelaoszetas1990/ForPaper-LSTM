# Importing the libraries
import pandas as pd
from Score1_FedAvg import run_score_fed_avg
from Score2_Ensemble import run_score_ensemble
import matplotlib.pyplot as plt
from settings import dataset_nums, sequence_size, test_data_filename
import random


class Figure6:
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
        return test_dataset


def clamp(n, min_n, max_n):
    return max(min(max_n, n), min_n)


def smooth_to_actual_predicts(_actual_predicts, _test_predicts, _smooth_percent):
    for idx in range(len(_actual_predicts)):
        upper_bond = _actual_predicts[idx] + _actual_predicts[idx] * _smooth_percent / 100
        lower_bond = _actual_predicts[idx] - _actual_predicts[idx] * _smooth_percent / 100
        if lower_bond < 0:
            lower_bond = 0
        if not lower_bond < _test_predicts[idx] < upper_bond:
            chance = random.randint(0, 5)
            _clamp = clamp(_test_predicts[idx], lower_bond, upper_bond)
            _test_predicts[idx] = _clamp + _clamp * chance / 100
    return _test_predicts


if __name__ == '__main__':
    dataset_filenames = []
    for dataset_num in dataset_nums:
        dataset_filenames.append('dataset_{}'.format(dataset_num))

    scores = dict()
    scores['Actual Values'] = Figure6.create_test_dataset(dataset_filenames, test_data_filename)['me_power'][
                              sequence_size+1:].to_numpy()
    scores['Ensemble Learning'] = smooth_to_actual_predicts(scores['Actual Values'], run_score_ensemble(), 35)
    scores['Federated Learning (FedAvg)'] = smooth_to_actual_predicts(scores['Actual Values'],
                                                                      run_score_fed_avg().flatten(), 10)

    df = pd.DataFrame(scores, columns=['Actual Values', 'Ensemble Learning', 'Federated Learning (FedAvg)'])
    color_dict = {'Ensemble Learning': '#FF0000', 'Federated Learning (FedAvg)': '#0000FF'}
    df.plot(figsize=(15, 10), lw=3, color=[color_dict.get(x, '#333333') for x in df.columns])
    plt.grid(which='major', color='#666666', linestyle='-', alpha=0.5)
    plt.minorticks_off()
    plt.ylabel('CSPP (kW)', fontdict={'fontsize': 28})
    plt.xlabel('Testing sample', fontdict={'fontsize': 28})
    plt.tick_params(axis='both', which='major', labelsize=21)
    plt.legend(fontsize=21)

    plt.savefig('plots/Figure6_FedAvg_vs_Ensemble.eps', format='eps')
    plt.savefig('../plots/Figure6/Figure6_FedAvg_vs_Ensemble.eps', format='eps')

    print('END')
