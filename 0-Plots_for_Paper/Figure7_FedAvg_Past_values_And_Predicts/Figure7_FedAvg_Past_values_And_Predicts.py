# Importing the libraries
import sys
import pandas as pd
import matplotlib.pyplot as plt
from settings import dataset_nums, future_predicts, sequence_sizes, test_data_filename
from FedAvg_runs import run_fed_avg_with_predicts_and_past_sequences


class Figure7:
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


def normalize_numbers(_scores):
    _scores_min = sys.float_info.max
    _scores_max = 0
    for key in _scores.keys():
        for value in _scores[key]:
            if value > _scores_max:
                _scores_max = value
            if value < _scores_min:
                _scores_min = value
    for key in _scores.keys():
        for i, value in enumerate(_scores[key]):
            _scores[key][i] = round(1 - round((_scores[key][i] - _scores_min)/(_scores_max - _scores_min), 2) * 0.9, 2)
    return _scores


if __name__ == '__main__':
    dataset_filenames = []
    for dataset_num in dataset_nums:
        dataset_filenames.append('dataset_{}'.format(dataset_num))

    Figure7.create_test_dataset(dataset_filenames, test_data_filename)
    results = run_fed_avg_with_predicts_and_past_sequences()

    scores = dict()
    for idx_fp, future_predict in enumerate(future_predicts):
        _key = 'Forecast moment = {} min'.format(future_predicts[idx_fp] * 15)
        scores[_key] = []
        for idx_ss, sequence_size in enumerate(sequence_sizes):
            for result in results:
                if result['c'] == 'c{}'.format(future_predicts[idx_fp]) and result['sequence'] == sequence_sizes[idx_ss]:
                    scores[_key].append(result['MAE'])

    print(scores)
    scores = normalize_numbers(scores)
    print(scores)

    _columns = []
    _color_dict = dict()
    for idx_fp, future_predict in enumerate(future_predicts):
        val = 'Forecast moment = {} min'.format(future_predicts[idx_fp] * 15)
        _columns.append(val)
        if idx_fp == 0:
            _color_dict[val] = '#FF0000'
        elif idx_fp == 1:
            _color_dict[val] = '#39bd00'
        else:
            _color_dict[val] = '#0000FF'

    df = pd.DataFrame(scores, columns=_columns, index=['5', '10', '15'])
    df.plot(figsize=(15, 10), lw=3, color=[_color_dict.get(x, '#333333') for x in df.columns])
    plt.grid(which='major', color='#666666', linestyle='-', alpha=0.5)
    plt.minorticks_off()
    plt.xlabel('Past-values Window', fontdict={'fontsize': 28})
    plt.ylabel('Normalized Accuracy', fontdict={'fontsize': 28})
    plt.tick_params(axis='both', which='major', labelsize=21)
    plt.ylim(0, 1.1)
    ticks_y = []
    for i in range(0, 11, 2):
        ticks_y.append(i/10)
    plt.yticks(ticks_y)
    plt.legend(fontsize=21)

    plt.savefig('plots/Figure7_FedAvg_Past_values_And_Predicts.eps',
                format='eps')
    plt.savefig('../plots/Figure7/Figure7_FedAvg_Past_values_And_Predicts.eps'
                , format='eps')

    print('END')
