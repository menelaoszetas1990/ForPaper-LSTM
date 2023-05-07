import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import csv


# calculate new metric columns for the KPIs
def kpi_calculation(dataframe, x, y):
    return (dataframe[x] + dataframe[y]) / (dataframe[x] * dataframe[y])


# calculate the dataframes sorted by specific columns that indicate the best value of the column
def calc_dfs_for_top_best_values(dataframe, head_num, with_exec_time=False):
    # for execution time
    df_exec_time = dataframe.sort_values(by=['exec_time']).head(head_num)

    # for train scores
    x1 = dataframe.sort_values(by=['trainScoreMAE']).head(head_num)
    x2 = dataframe.sort_values(by=['trainScoreMSE']).head(head_num)
    x3 = dataframe.sort_values(by=['trainScoreRMSE']).head(head_num)
    x4 = dataframe.sort_values(by=['testScoreMAE']).head(head_num)
    x5 = dataframe.sort_values(by=['testScoreMSE']).head(head_num)
    x6 = dataframe.sort_values(by=['testScoreRMSE']).head(head_num)

    # for test scores
    y1 = dataframe.sort_values(by=['KPI_Train_MAE_Exec_time'], ascending=False).head(head_num)
    y2 = dataframe.sort_values(by=['KPI_Train_MSE_Exec_time'], ascending=False).head(head_num)
    y3 = dataframe.sort_values(by=['KPI_Train_RMSE_Exec_time'], ascending=False).head(head_num)
    y4 = dataframe.sort_values(by=['KPI_Test_MAE_Exec_time'], ascending=False).head(head_num)
    y5 = dataframe.sort_values(by=['KPI_Test_MSE_Exec_time'], ascending=False).head(head_num)
    y6 = dataframe.sort_values(by=['KPI_Test_RMSE_Exec_time'], ascending=False).head(head_num)

    # print(df_grouped.sort_values(by=['trainScoreMAE']).head(head_num).index)
    # print(df_grouped.sort_values(by=['trainScoreMSE']).head(head_num).index)
    # print(df_grouped.sort_values(by=['trainScoreRMSE']).head(head_num).index)
    # print(df_grouped.sort_values(by=['testScoreMAE']).head(head_num).index)
    # print(df_grouped.sort_values(by=['testScoreMSE']).head(head_num).index)
    # print(df_grouped.sort_values(by=['testScoreRMSE']).head(head_num).index)
    # print(df_grouped.sort_values(by=['KPI_Train_MAE_Exec_time'], ascending=False).head(head_num).index)
    # print(df_grouped.sort_values(by=['KPI_Train_MSE_Exec_time'], ascending=False).head(head_num).index)
    # print(df_grouped.sort_values(by=['KPI_Train_RMSE_Exec_time'], ascending=False).head(head_num).index)
    # print(df_grouped.sort_values(by=['KPI_Test_MAE_Exec_time'], ascending=False).head(head_num).index)
    # print(df_grouped.sort_values(by=['KPI_Test_MSE_Exec_time'], ascending=False).head(head_num).index)
    # print(df_grouped.sort_values(by=['KPI_Test_RMSE_Exec_time'], ascending=False).head(head_num).index)

    data_frames_to_merge = [x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6]
    if with_exec_time:
        data_frames_to_merge.append(df_exec_time)

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=list(x1.columns), how='inner', indicator=True)
                       .assign(**{'Remark Added': lambda d: d['_merge'].eq('both').map({True: 'Yes', False: ''})})
                       .drop(columns='_merge'), data_frames_to_merge)

    return len(df_merged), df_merged


def check_best_lstm_settings_result(best_lstm_setting, new_records):
    records_to_add = []
    for new_record in new_records:
        _found_record = False
        for key in best_lstm_setting.keys():
            if new_record in best_lstm_setting[key]:
                _found_record = True
                break
        if not _found_record:
            records_to_add.append(new_record)
    return records_to_add


if __name__ == '__main__':
    # read csv
    filename = 'Comparison_Data.csv'
    df = pd.read_csv(filename, header=0)

    df['KPI_Train_MAE_Exec_time'] = kpi_calculation(df, 'trainScoreMAE', 'exec_time')
    df['KPI_Train_MSE_Exec_time'] = kpi_calculation(df, 'trainScoreMSE', 'exec_time')
    df['KPI_Train_RMSE_Exec_time'] = kpi_calculation(df, 'trainScoreRMSE', 'exec_time')
    df['KPI_Test_MAE_Exec_time'] = kpi_calculation(df, 'testScoreMAE', 'exec_time')
    df['KPI_Test_MSE_Exec_time'] = kpi_calculation(df, 'testScoreMSE', 'exec_time')
    df['KPI_Test_RMSE_Exec_time'] = kpi_calculation(df, 'testScoreRMSE', 'exec_time')

    # print(df.head())
    # print(df.info())
    # print(df.describe())

    # group dataframe based on mean values of 5 of every setup:
    # double_LSTM:      LSTM --> single of double
    # sequence_size:    Number of Past Sequences --> 5,10,15,20,25
    # epoches:          Epoches of LSTM --> 5,6,7,8,9,10
    # batch_size:       Batch Sizes of LSTM --> 16,32,64
    df_grouped = df.groupby(['double_LSTM', 'sequence_size', 'epoches', 'batch_size']).mean()
    # print(df_grouped.head())

    columns = ['position', 'double_LSTM', 'sequence_size', 'epoches', 'batch_size',
               'trainScoreMAE', 'trainScoreMSE', 'trainScoreRMSE',
               'testScoreMAE', 'testScoreMSE', 'testScoreRMSE', 'exec_time']
    filename = 'Comparison_Results.csv'
    with open(filename, 'a', encoding='utf-8', newline='') as _output_file:
        writer = csv.DictWriter(_output_file, fieldnames=columns)
        writer.writeheader()

        # find the top number (top_LSTM_setup_setting_num) of LSTM settings
        top_LSTM_setup_setting_num = 10
        take_exec_time_in_consideration = False

        found = 0
        best_LSTM_settings = {}
        best_LSTM_settings_index = 1
        for i in range(1, len(df_grouped)):
            result = calc_dfs_for_top_best_values(df_grouped, i, take_exec_time_in_consideration)
            if result[0] == found:
                continue
            else:
                found = result[0]
                best_LSTM_settings[best_LSTM_settings_index] = \
                    check_best_lstm_settings_result(best_LSTM_settings, result[1].to_dict(orient='records'))
                best_LSTM_settings_index += 1
            if found >= top_LSTM_setup_setting_num:
                break
        else:
            print('No shared setup(s) of {} found. Found: {} total'.format(top_LSTM_setup_setting_num, found))

        if len(best_LSTM_settings.keys()) > 0:
            for position in best_LSTM_settings.keys():
                print('{} position:'.format(position))
                for record in best_LSTM_settings[position]:
                    found_record = df_grouped.loc[(df_grouped['trainScoreMAE'] == record['trainScoreMAE']) &
                                                  (df_grouped['trainScoreMSE'] == record['trainScoreMSE']) &
                                                  (df_grouped['trainScoreRMSE'] == record['trainScoreRMSE']) &
                                                  (df_grouped['testScoreMAE'] == record['testScoreMAE']) &
                                                  (df_grouped['testScoreMSE'] == record['testScoreMSE']) &
                                                  (df_grouped['testScoreRMSE'] == record['testScoreRMSE']) &
                                                  (df_grouped['KPI_Train_MAE_Exec_time']
                                                   == record['KPI_Train_MAE_Exec_time']) &
                                                  (df_grouped['KPI_Train_MSE_Exec_time']
                                                   == record['KPI_Train_MSE_Exec_time']) &
                                                  (df_grouped['KPI_Train_RMSE_Exec_time']
                                                   == record['KPI_Train_RMSE_Exec_time']) &
                                                  (df_grouped['KPI_Test_MAE_Exec_time']
                                                   == record['KPI_Test_MAE_Exec_time']) &
                                                  (df_grouped['KPI_Test_MSE_Exec_time']
                                                   == record['KPI_Test_MSE_Exec_time']) &
                                                  (df_grouped['KPI_Test_RMSE_Exec_time']
                                                   == record['KPI_Test_RMSE_Exec_time']) &
                                                  (df_grouped['exec_time'] == record['exec_time'])]
                    print('\tDouble LSTM: \t{}, \tSequence Size: \t{}, \tEpoches: \t{}, \tBatch Size: \t{}'
                          .format(found_record.index[0][0], found_record.index[0][1], found_record.index[0][2],
                                  found_record.index[0][3]))
                    writer.writerow({'position': position, 'double_LSTM': found_record.index[0][0],
                                     'sequence_size': found_record.index[0][1], 'epoches': found_record.index[0][2],
                                     'batch_size': found_record.index[0][3], 'trainScoreMAE': record['trainScoreMAE'],
                                     'trainScoreMSE': record['trainScoreMSE'],
                                     'trainScoreRMSE': record['trainScoreRMSE'],
                                     'testScoreMAE': record['testScoreMAE'],
                                     'testScoreMSE': record['testScoreMSE'],
                                     'testScoreRMSE': record['testScoreRMSE'],
                                     'exec_time': record['exec_time']})

    # Plots
    def sns_plots_preparation(main_df, axe_num, seq_size, y_axis, _lstm_single_plot_state):
        if _lstm_single_plot_state:
            df_to_plot = main_df.loc[(main_df['sequence_size'] == seq_size) & (main_df['double_LSTM'] == False)]
        else:
            df_to_plot = main_df.loc[(main_df['sequence_size'] == seq_size) & (main_df['double_LSTM'] == True)]

        sns.boxplot(ax=axes[axe_num], x=df_to_plot['epoches'], y=df_to_plot[y_axis], hue=df_to_plot['batch_size'],
                    data=df_to_plot, palette='rainbow')
        axes[axe_num].set_title('Sequence size = {}'.format(seq_size))


    sequences = df['sequence_size'].unique()
    for column in df.columns.drop(['double_LSTM', 'sequence_size', 'epoches', 'batch_size', 'KPI_Train_MAE_Exec_time',
                                   'KPI_Train_MSE_Exec_time', 'KPI_Train_RMSE_Exec_time', 'KPI_Test_MAE_Exec_time',
                                   'KPI_Test_MSE_Exec_time', 'KPI_Test_RMSE_Exec_time']):
        # 1 - Single LSTM taken in consideration
        # 2 - Double LSTM taken in consideration
        for lstm_for_plot in [1, 2]:
            fig, axes = plt.subplots(1, 5, figsize=(25, 10), sharey=True)
            fig.suptitle('{} - Epoches. hue=batch_size'.format(column))

            title = '{} - Epoches. hue=batch_size, Single LSTM'.format(column)
            lstm_single_plot_state = True
            if lstm_for_plot == 2:
                title = '{} - Epoches. hue=batch_size, Double LSTM'.format(column)
                lstm_single_plot_state = False
            fig.suptitle(title)

            for axe_number in range(0, len(axes)):
                sns_plots_preparation(df, axe_number, sequences[axe_number], column, lstm_single_plot_state)
            fig.savefig('plots/Metrics_Plots/{}.png'.format(title))
