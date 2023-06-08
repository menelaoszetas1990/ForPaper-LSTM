# Importing the libraries
import pandas as pd
from Score1_Model_Predict_Loss_Average_of_Losses import run_score_1
from Score2_Model_Predict_Average_of_Predicts_Loss import run_score_2
from Score3_Centralized_Learning import run_score_3
from Score4_Federated_Learning import run_score_4
import matplotlib.pyplot as plt
from settings import test_dataset_nums


class Figure5:
    def __init__(self):
        pass

    @staticmethod
    def create_test_datasets(_dataset_filenames):
        _test_dataset_filenames = []
        for idx, dataset_filename in enumerate(_dataset_filenames):
            _dataset = pd.read_csv('../../data/' + dataset_filename + '.csv', usecols=['trim', 'sog', 'stw', 'wspeedbf',
                                                                                       'wdir', 'me_power']).tail(300)
            _name = 'test_dataset_{}.csv'.format(idx)
            _test_dataset_filenames.append(_name)
            _dataset.to_csv(_name, index=False)
        return _test_dataset_filenames


def normalize_numbers(_scores):
    _scores_min = min(_scores)
    _scores_max = max(_scores)
    for i, score in enumerate(_scores):
        _scores[i] = round(1 - round((_scores[i] - _scores_min) / (_scores_max - _scores_min), 2) * 0.9, 2)
    return _scores


if __name__ == '__main__':
    dataset_filenames = []
    for test_dataset_num in test_dataset_nums:
        dataset_filenames.append('dataset_{}'.format(test_dataset_num))

    test_dataset_filenames = Figure5.create_test_datasets(dataset_filenames)

    scores = dict()
    for idx, test_dataset_filename in enumerate(test_dataset_filenames):
        _key = 'Ship {}'.format(idx + 1)
        scores[_key] = []
        scores[_key].append(run_score_1(test_dataset_filename)[1])
        scores[_key].append(run_score_2(test_dataset_filename)[1])
        scores[_key].append(run_score_3(test_dataset_filename)[1])
        scores[_key].append(run_score_4(test_dataset_filename)[1])

    for _key in scores.keys():
        scores[_key] = normalize_numbers(scores[_key])

    print(scores)

    df = pd.DataFrame(scores, index=['ΤΑΙΑ', 'ΤΑΙΜ', 'ΤΟΙΟ', 'TFIF'])
    df.plot(figsize=(12, 8), lw=4)
    plt.grid(which='major', color='#666666', linestyle='-', alpha=0.5)
    plt.minorticks_off()
    plt.xlabel('Schema', fontdict={'fontsize': 32})
    plt.ylabel('Normalized Validation Error', fontdict={'fontsize': 32})
    plt.tick_params(axis='both', which='major', labelsize=28)
    plt.ylim(0, 1.1)
    ticks = []
    for i in range(0, 11, 2):
        ticks.append(i/10)
    plt.yticks(ticks)
    plt.legend(fontsize=28)

    plt.savefig('plots/Figure5_Ensamble_vs_Collaborative_vs_Centralized_vs_Federated_learning_test_ships.eps',
                format='eps')

    print('END')
