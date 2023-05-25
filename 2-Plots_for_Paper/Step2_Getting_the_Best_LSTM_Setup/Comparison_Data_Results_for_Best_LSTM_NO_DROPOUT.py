import pandas as pd
from functools import reduce
import csv
from Getting_the_Best_LSTM_Setup_per_Dataset_NO_DROPOUT import FindBestLSTMSetupPerDataset


class FindBestLSTMSetupResult:
    def __init__(self, data_filenames, top_lstms_to_check):
        # Preprocessing
        # Importing the datasets
        self.dfs = []
        self.maxHead = 0
        for data_filename in data_filenames:
            df = pd.read_csv(data_filename, header=0).dropna().drop(['position', 'found_at_top',
                                                                     'train_score_MAE', 'train_score_MSE',
                                                                     'train_score_RMSE', 'test_score_MAE',
                                                                     'test_score_MSE', 'test_score_RMSE',
                                                                     'epoches_stopped', 'exec_time'], axis=1) \
                .head(top_lstms_to_check)
            self.dfs.append(df)
            if len(df.axes[0]) > self.maxHead:
                self.maxHead = len(df.axes[0])

    # calculate the dataframes sorted by specific columns that indicate the best value of the column
    def calc_dfs_for_top_best_values(self, head_num):
        data_frames_to_merge = []
        for df in self.dfs:
            data_frames_to_merge.append(df.head(head_num))

        df_merged = reduce(lambda left, right: pd.merge(left, right, on=['learning_rate', 'sequence_size', 'batch_size',
                                                                         'hidden_layers'], how='inner', indicator=True)
                           .assign(**{'Remark Added': lambda d: d['_merge'].eq('both').map({True: 'Yes', False: ''})})
                           .drop(columns='_merge'), data_frames_to_merge)
        return len(df_merged), df_merged

    @staticmethod
    def check_best_lstm_settings_result(best_lstm_setting, new_records):
        records_to_add = []
        for new_record in new_records:
            _found_record = False
            for key in best_lstm_setting.keys():
                if new_record in best_lstm_setting[key][0]:
                    _found_record = True
                    break
            if not _found_record:
                records_to_add.append(new_record)
        return records_to_add

    def export_results(self, result_filename, top_number_of_lstm_settings_to_find=10):
        columns = ['position', 'found_at_top', 'learning_rate', 'sequence_size', 'batch_size', 'hidden_layers']
        with open(result_filename, 'w', encoding='utf-8', newline='') as _output_file:
            writer = csv.DictWriter(_output_file, fieldnames=columns)
            writer.writeheader()

            found = 0
            best_LSTM_settings = {}
            best_LSTM_settings_index = 1
            for i in range(1, self.maxHead):
                result = self.calc_dfs_for_top_best_values(i)
                if result[0] == found:
                    continue
                else:
                    found = result[0]
                    best_LSTM_settings[best_LSTM_settings_index] = \
                        (FindBestLSTMSetupResult.check_best_lstm_settings_result(best_LSTM_settings, result[1]
                                                                                 .to_dict(orient='records')), i)
                    best_LSTM_settings_index += 1
                if found >= top_number_of_lstm_settings_to_find:
                    break
            else:
                print('No shared setup(s) of {} found. Found: {} total'.format(top_number_of_lstm_settings_to_find,
                                                                               found))

            if len(best_LSTM_settings.keys()) > 0:
                for position in best_LSTM_settings.keys():
                    for record in best_LSTM_settings[position][0]:
                        print('\tLearning Rate: \t{:6}, \tSequence Size: \t{:2}, \tBatch Size: \t{:2}, '
                              '\tHidden Layers: \t {:2}'
                              .format(record['learning_rate'], record['sequence_size'], record['batch_size'],
                                      record['hidden_layers']))
                        writer.writerow({'position': position, 'found_at_top': best_LSTM_settings[position][1],
                                         'learning_rate': record['learning_rate'],
                                         'sequence_size': record['sequence_size'],
                                         'batch_size': record['batch_size'],
                                         'hidden_layers': record['hidden_layers']})


if __name__ == '__main__':
    dataset_nums = [1, 2, 3, 5, 6, 7]

    for dataset_num in dataset_nums:
        # read csv
        filename = '../Step1-Gather_Data_for_best_LSTM_Setup_Comparison/Comparison_Data_dataset_{}_NO_DROPOUT.csv'\
            .format(dataset_num)
        test = FindBestLSTMSetupPerDataset(filename)
        test.export_results('Comparison_Data_results_dataset_{}_NO_DROPOUT.csv'.format(dataset_num), 30)

    datasets = []
    for dataset_num in dataset_nums:
        # read csv
        filename = 'Comparison_Data_results_dataset_{}_NO_DROPOUT.csv'\
            .format(dataset_num)
        datasets.append(filename)

    test = FindBestLSTMSetupResult(datasets, 100)
    test.export_results('Comparison_Data_results_NO_DROPOUT.csv', 5)
