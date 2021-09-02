from __future__ import division
import numpy as np
import pandas as pd
import pickle
import os
from math import ceil
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.metrics import r2_score

warnings.simplefilter("ignore")
# colors = ["#3366cc", "#dc3912", "#109618", "#990099", "#ff9900"]
colors = sns.color_palette('muted')
labels = ['Remaining', 'First','Last']

    
    
def density_plot(df, Accuracy_base, Accuracy_LSTM, Accuracy_NG, save_fig, Out_put_name,model_name_list, Mean_or_median):
    model = model_name_list
    sns.set(font_scale=1.5)
    sns.set_style("white", {"legend.frameon": True})
    plt.figure(figsize=(14, 7))
    ax1 = plt.subplot(1, 2, 1)
    cols1 = [df, Accuracy_base, Accuracy_LSTM, Accuracy_NG]
    for i in range(len(cols1)):
        data = cols1[i]['first']
        sns.kdeplot(data, ax=ax1, shade=True, color=colors[i], label=model[i],linewidth=2)
        if Mean_or_median == 'Mean':
            med = data.mean()
        else:
            med = data.median()
        plt.axvline(med, color=colors[i], linestyle='dashed', linewidth=2)
        if i == 1:
            if 'duration' in Out_put_name:
                plt.text(med - 0.02, 3.0, '{}'.format(round(med, 3)),
                         horizontalalignment='right', verticalalignment='center',
                         fontsize=18, color=colors[i])
            else:
                plt.text(med - 0.02, 3.0, '{}%'.format(round(med * 100, 1)),
                         horizontalalignment='right', verticalalignment='center',
                         fontsize=18, color=colors[i])
        elif i== 2:
            if 'duration' in Out_put_name:
                plt.text(med + 0.02, 3.1, '{}'.format(round(med, 3)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])
            else:
                plt.text(med + 0.02, 3.1, '{}%'.format(round(med * 100, 1)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])
        elif i == 0:
            if 'duration' in Out_put_name:
                plt.text(med + 0.02, 3.3, '{}'.format(round(med, 3)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])
            else:
                plt.text(med + 0.02, 3.3, '{}%'.format(round(med * 100, 1)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])
        elif i == 3:
            if 'duration' in Out_put_name:
                plt.text(med - 0.02, 3.0, '{}'.format(round(med, 3)),
                         horizontalalignment='right', verticalalignment='center',
                         fontsize=18, color=colors[i])
            else:
                plt.text(med - 0.02, 3.3, '{}%'.format(round(med * 100, 1)),
                         horizontalalignment='right', verticalalignment='center',
                         fontsize=18, color=colors[i])

    plt.xlim(0, 1.0)
    plt.ylim(0, 3.5)
    if 'location' in Out_put_name:
        ax1.set_xticklabels([str(i) + '%' for i in range(0, 101, 20)])
        plt.xlabel('Prediction accuracy', fontsize=20)
    else:
        plt.xlabel('R'+r'$^2$', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    if 'location' in Out_put_name:
        plt.legend(fontsize=18) #, loc='upper right'
    else:
        plt.legend(fontsize=18)  #
    plt.title('First activities',fontsize=20)
    # plt.text(-0.1, 1.05, '(a)', fontdict={'size': 20, 'weight': 'bold'},
    #          transform=ax1.transAxes)


    ax2 = plt.subplot(1, 2, 2)
    for i in range(len(cols1)):
        data = cols1[i]['Middle']
        sns.kdeplot(data, ax=ax2, shade=True, color=colors[i], label=model[i], linewidth=2)
        if Mean_or_median == 'Mean':
            med = data.mean()
        else:
            med = data.median()
        plt.axvline(med, color=colors[i], linestyle='dashed', linewidth=2, alpha = 1)
        if i == 1:
            if 'duration' in Out_put_name:
                plt.text(med - 0.01, 3.3, '{}'.format(round(med, 3)),
                         horizontalalignment='right', verticalalignment='center',
                         fontsize=18, color=colors[i])
            else:
                plt.text(med + 0.01, 3.3, '{}%'.format(round(med * 100, 1)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])
        elif i== 2:
            if 'duration' in Out_put_name:
                plt.text(med + 0.023, 3.0, '{}'.format(round(med, 3)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])
            else:
                plt.text(med - 0.01, 3.0, '{}%'.format(round(med * 100, 1)),
                         horizontalalignment='right', verticalalignment='center',
                         fontsize=18, color=colors[i])

        elif i == 0:
            if 'duration' in Out_put_name:
                plt.text(med + 0.01, 3.0, '{}'.format(round(med, 3)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])
            else:
                plt.text(med + 0.01, 3.0, '{}%'.format(round(med* 100, 1)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])

        elif i == 3:
            if 'duration' in Out_put_name:
                plt.text(med + 0.01, 3.3, '{}'.format(round(med, 3)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])
            else:
                plt.text(med + 0.01, 3.3, '{}%'.format(round(med* 100, 1)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])
    plt.xlim(0, 1.0)
    plt.ylim(0, 3.5)
    if 'location' in Out_put_name:
        ax2.set_xticklabels([str(i) + '%' for i in range(0, 101, 20)])
        plt.xlabel('Prediction accuracy', fontsize=20)
    else:
        plt.xlabel('R'+r'$^2$', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel('Density', fontsize=20)
    if 'location' in Out_put_name:
        plt.legend(fontsize=18) #, loc='upper right'
    else:
        plt.legend(fontsize=18)  #
    plt.title('Remaining activities',fontsize=20)
    # plt.text(-0.1, 1.05, '(b)', fontdict={'size': 20, 'weight': 'bold'},
    #          transform=ax2.transAxes)
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/' + Out_put_name, dpi=200)


def density_plot_duration_error(df, Accuracy_base, Accuracy_LSTM, save_fig, Out_put_name, model_name_list, Mean_or_median):
    model = model_name_list
    sns.set(font_scale=1.5)
    sns.set_style("white", {"legend.frameon": True})
    plt.figure(figsize=(14, 7))
    ax1 = plt.subplot(1, 2, 1)
    cols1 = [df, Accuracy_base, Accuracy_LSTM]
    for i in range(len(cols1)):
        data = cols1[i]['first']
        sns.kdeplot(data, ax=ax1, shade=True, color=colors[i], label=model[i])
        if Mean_or_median == 'Mean':
            med = data.mean()
        else:
            med = data.median()
        plt.axvline(med, color=colors[i], linestyle='dashed', linewidth=2)
        if i == 0:
            plt.text(med + 0.02, 3.3, '{}%'.format(round(med * 100, 1)),
                     horizontalalignment='left', verticalalignment='center',
                     fontsize=18, color=colors[i])
        elif i == 2:
            if 'duration' in Out_put_name:
                plt.text(med + 0.02, 3.0, '{}%'.format(round(med * 100, 1)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])
            else:
                plt.text(med + 0.02, 3.0, '{}%'.format(round(med * 100, 1)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])
        else:
            plt.text(med - 0.01, 3.3, '{}%'.format(round(med * 100, 1)),
                     horizontalalignment='right', verticalalignment='center',
                     fontsize=18, color=colors[i])
    # plt.xlim(0, 1.0)
    # plt.ylim(0, 3.5)
    # ax1.set_xticklabels([str(i) + '%' for i in range(0, 101, 20)])
    # plt.xlabel('Prediction Accuracy', fontsize=20)
    plt.ylabel('Density (Users)', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    if 'location' in Out_put_name:
        plt.legend(fontsize=18)  # , loc='upper right'
    else:
        plt.legend(fontsize=18, loc='center right')  #
    plt.title('First Activities', fontsize=20)
    # plt.text(-0.1, 1.05, '(a)', fontdict={'size': 20, 'weight': 'bold'},
    #          transform=ax1.transAxes)

    ax2 = plt.subplot(1, 2, 2)
    for i in range(len(cols1)):
        data = cols1[i]['Remaining']
        sns.kdeplot(data, ax=ax2, shade=True, color=colors[i], label=model[i])
        if Mean_or_median == 'Mean':
            med = data.mean()
        else:
            med = data.median()
        plt.axvline(med, color=colors[i], linestyle='dashed', linewidth=2)
        if i == 1:
            if 'duration' in Out_put_name:
                plt.text(med - 0.01, 3.3, '{}%'.format(round(med * 100, 1)),
                         horizontalalignment='right', verticalalignment='center',
                         fontsize=18, color=colors[i])
            else:
                plt.text(med + 0.01, 3.3, '{}%'.format(round(med * 100, 1)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])
        elif i == 2:
            if 'duration' in Out_put_name:
                plt.text(med + 0.023, 3.0, '{}%'.format(round(med * 100, 1)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])
            else:
                plt.text(med - 0.01, 3.0, '{}%'.format(round(med * 100, 1)),
                         horizontalalignment='right', verticalalignment='center',
                         fontsize=18, color=colors[i])

        else:
            plt.text(med + 0.01, 3.3, '{}%'.format(round(med * 100, 1)),
                     horizontalalignment='left', verticalalignment='center',
                     fontsize=18, color=colors[i])
    # plt.xlim(0, 1.0)
    # plt.ylim(0, 3.5)
    # ax2.set_xticklabels([str(i) + '%' for i in range(0, 101, 20)])
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    # plt.xlabel('Prediction Accuracy', fontsize=20)
    plt.ylabel('Density (User-level)', fontsize=20)
    if 'location' in Out_put_name:
        plt.legend(fontsize=18)  # , loc='upper right'
    else:
        plt.legend(fontsize=18, loc='center right')  #
    plt.title('Remaining Activities', fontsize=20)
    # plt.text(-0.1, 1.05, '(b)', fontdict={'size': 20, 'weight': 'bold'},
    #          transform=ax2.transAxes)
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/' + Out_put_name, dpi=200)


def density_plot_not_seperate_mid_first(df, Accuracy_base, Accuracy_LSTM, save_fig, Out_put_name, model_name_list):
    model = model_name_list
    sns.set(font_scale=1.5)
    sns.set_style("white", {"legend.frameon": True})
    plt.figure(figsize=(7, 7))
    ax1 = plt.subplot(1, 1, 1)
    cols1 = [df, Accuracy_base, Accuracy_LSTM]
    for i in range(len(cols1)):
        data = cols1[i]['all']
        sns.kdeplot(data, ax=ax1, shade=True, color=colors[i], label=model[i])
        med = data.mean()
        plt.axvline(med, color=colors[i], linestyle='dashed', linewidth=2)
        if i == 0:
            plt.text(med + 0.02, 3.3, '{}%'.format(round(med * 100, 1)),
                     horizontalalignment='left', verticalalignment='center',
                     fontsize=18, color=colors[i])
        elif i == 2:
            if 'duration' in Out_put_name:
                plt.text(med + 0.02, 3.0, '{}%'.format(round(med * 100, 1)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])
            else:
                plt.text(med + 0.02, 3.0, '{}%'.format(round(med * 100, 1)),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=18, color=colors[i])
        else:
            plt.text(med - 0.01, 3.3, '{}%'.format(round(med * 100, 1)),
                     horizontalalignment='right', verticalalignment='center',
                     fontsize=18, color=colors[i])
    plt.xlim(0, 1.0)
    plt.ylim(0, 3.5)
    ax1.set_xticklabels([str(i) + '%' for i in range(0, 101, 20)])
    plt.xlabel('Prediction Accuracy', fontsize=20)
    plt.ylabel('Density (Users)', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    if 'location' in Out_put_name:
        plt.legend(fontsize=18)  # , loc='upper right'
    else:
        plt.legend(fontsize=18)  #
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/' + Out_put_name, dpi=200)


def data_process_continuous(data):
    error_first_temp = (data['Predict1'].loc[data['activity_index']==0] - data['Ground_truth'].loc[data['activity_index']==0])/3600
    Accuracy_first_temp = sum(np.array(data['Correct'].loc[data['activity_index']==0]))/data['Correct'].loc[data['activity_index']==0].count()
    data_temp = data.loc[data['activity_index']!=0]
    # data_temp = data
    error_Remaining_temp = (data_temp['Predict1'] - data_temp['Ground_truth'])/3600
    Accuracy_temp = sum(np.array(data_temp['Correct']))/data_temp['Correct'].count()
    accuracy_all = sum(np.array(data['Correct']))/data['Correct'].count()
    return error_first_temp, Accuracy_first_temp, error_Remaining_temp, Accuracy_temp,accuracy_all


def calculate_error(result_df):
    # correct error data
    result_df.loc[result_df['Predict_duration'] > 86400, 'Predict_duration'] = 86400
    result_df.loc[result_df['Predict_duration'] <= 0, 'Predict_duration'] = 1
    ######
    result_df['error_sq'] = (result_df['Predict_duration'] - result_df['Ground_truth_duration']) ** 2
    result_df['error_abs'] = np.abs(result_df['Predict_duration'] - result_df['Ground_truth_duration'])
    RMSE = np.sqrt(np.mean(result_df['error_sq']))
    MAPE = np.mean(result_df['error_abs'] / result_df['Ground_truth_duration'])
    MAE = np.mean(result_df['error_abs'])
    if len(result_df) > 0:
        R_sq = r2_score(result_df['Ground_truth_duration'], result_df['Predict_duration'])
    else:
        R_sq = None
    return RMSE, MAPE, MAE, R_sq

def r_sq_for_two_parts(data,y_mean):
    data['RES'] = (data['Ground_truth_duration'] -  data['Predict_duration'])**2
    data['TOT'] = (data['Ground_truth_duration'] -  y_mean)**2
    R_sq = 1 - sum(data['RES'])/sum(data['TOT'])
    return R_sq


def data_process_continuous_R_sq(data):
    _, _, _, R_sq_all = calculate_error(data)
    data_first = data.loc[data['activity_index']==0].copy()
    data_Remaining = data.loc[data['activity_index']!=0].copy()
    mean_y = np.mean(data['Ground_truth_duration'])
    R_sq_first = r_sq_for_two_parts(data_first, mean_y)
    if len(data_Remaining)>0:
        R_sq_Remaining = r_sq_for_two_parts(data_Remaining, mean_y)
    else:
        R_sq_Remaining = None

    return R_sq_first, R_sq_Remaining, R_sq_all

def data_process_continuous_RMSE(data):
    RMSE_all, _, _, _ = calculate_error(data)
    data_first = data.loc[data['activity_index']==0].copy()
    data_Remaining = data.loc[data['activity_index']!=0].copy()
    RMSE_first, _, _, R_sq_first = calculate_error(data_first)
    RMSE_Remaining, _, _, R_sq_Remaining = calculate_error(data_Remaining)
    return RMSE_first, RMSE_Remaining, RMSE_all


def data_process_continuous_MAPE(data):
    _, MAPE_all, _, _ = calculate_error(data)
    data_first = data.loc[data['activity_index']==0].copy()
    data_Remaining = data.loc[data['activity_index']!=0].copy()
    _, MAPE_first, _, R_sq_first = calculate_error(data_first)
    _, MAPE_Remaining, _, R_sq_Remaining = calculate_error(data_Remaining)
    return MAPE_first, MAPE_Remaining, MAPE_all


def data_process_discrete(data):
    error_first_temp = (data['Predict1'].loc[data['activity_index']==0] - data['Ground_truth'].loc[data['activity_index']==0])
    Accuracy_first_temp = sum(np.array(data['Correct'].loc[data['activity_index']==0]))/data['Correct'].loc[data['activity_index']==0].count()
    data_temp = data.loc[data['activity_index']!=0]
    # data_temp = data
    error_Remaining_temp = (data_temp['Predict1'] - data_temp['Ground_truth'])
    Accuracy_temp = sum(np.array(data_temp['Correct']))/data_temp['Correct'].count()
    accuracy_all = sum(np.array(data['Correct'])) / data['Correct'].count()
    return error_first_temp, Accuracy_first_temp, error_Remaining_temp, Accuracy_temp, accuracy_all


def generate_accuracy_file(individual_ID_list, output_fig, duration_error):
    error_list=[]
    total=0
    error_Remaining = pd.DataFrame({'Remaining':[]})
    error_first = pd.DataFrame({'first':[]})
    error_Remaining_base = pd.DataFrame({'Remaining':[]})
    error_first_base = pd.DataFrame({'first':[]})
    Accuracy = {'Card_ID':[], 'Remaining':[],'first':[],'all':[]}
    Accuracy_base = {'Card_ID':[], 'Remaining':[],'first':[],'all':[]}
    Accuracy_LSTM = {'Card_ID': [], 'Remaining': [], 'first': [], 'all': []}
    Accuracy_NG = {'Card_ID': [], 'Remaining': [], 'first': [], 'all': []}
    # data
    Card_ID_used = []
    # individual_ID_list = individual_ID_list[0:80]
    #############IOHMM
    for Card_ID in individual_ID_list:
        # if output_fig == 'duration':
        #     file_name = data_path + 'results/result_' + str(Card_ID) + 'test' + '.csv'
        # else:
        #     file_name = data_path + 'results/result_Location_' + str(Card_ID) + 'test' + '.csv'
        file_name =  data_path + 'results/result_con_dur+loc_' + str(Card_ID) + 'test' + '.csv'
        if os.path.exists(file_name) == False:
            print(Card_ID,'does not exist for IOHMM')
            continue
        else:
            Card_ID_used.append(Card_ID)
        data = pd.read_csv(file_name)
        if output_fig == 'duration':
            if duration_error == 'RMSE':
                R_sq_first, R_sq_Remaining, R_sq_all = data_process_continuous_RMSE(data)
            elif duration_error == 'MAPE':
                R_sq_first, R_sq_Remaining, R_sq_all = data_process_continuous_MAPE(data)
            else:
                R_sq_first, R_sq_Remaining, R_sq_all = data_process_continuous_R_sq(data)

            Accuracy['first'].append(R_sq_first)
            Accuracy['Remaining'].append(R_sq_Remaining)
            Accuracy['all'].append(R_sq_all)
            Accuracy['Card_ID'].append(Card_ID)
        else:
            error_first_temp, Accuracy_first_temp, error_Remaining_temp, Accuracy_temp, accuracy_all = data_process_discrete(data)
            #print (error_first_temp)
            error_first = pd.concat([error_first, error_first_temp], axis = 0)
            error_Remaining = pd.concat([error_Remaining, error_Remaining_temp], axis = 0)
            Accuracy['first'].append(Accuracy_first_temp)
            Accuracy['Remaining'].append(Accuracy_temp)
            Accuracy['all'].append(accuracy_all)
            Accuracy['Card_ID'].append(Card_ID)
        # data
    ############## LSTM
    Card_ID_used_for_base = list(set(Card_ID_used))
    for Card_ID in Card_ID_used_for_base:
        if output_fig == 'duration':
            # file_name = data_path + 'results/result_LSTM' + str(Card_ID) + 'test' + '.csv'
            file_name = data_path + 'results/result_LSTM_con_dur' + str(Card_ID) + 'test' + '.csv'
            #file_name = data_path + 'results/result_NGRAM_con_dur_' + str(Card_ID) + '.csv'
        else:
            file_name = data_path + 'results/result_Location_LSTM' + str(Card_ID) + 'test' + '.csv'
            #file_name = data_path + 'results/result_NGRAM_location_' + str(Card_ID) + '.csv'

        if os.path.exists(file_name) == False:
            print(Card_ID,'does not exist for LSTM')
            continue
        data = pd.read_csv(file_name)
        if output_fig == 'duration':
            if duration_error == 'RMSE':
                R_sq_first, R_sq_Remaining, R_sq_all = data_process_continuous_RMSE(data)
            elif duration_error == 'MAPE':
                R_sq_first, R_sq_Remaining, R_sq_all = data_process_continuous_MAPE(data)
            else:
                R_sq_first, R_sq_Remaining, R_sq_all = data_process_continuous_R_sq(data)
            Accuracy_LSTM['first'].append(R_sq_first)
            Accuracy_LSTM['Remaining'].append(R_sq_Remaining)
            Accuracy_LSTM['all'].append(R_sq_all)
            Accuracy_LSTM['Card_ID'].append(Card_ID)
        else:
            error_first_temp, Accuracy_first_temp, error_Remaining_temp, Accuracy_temp, accuracy_all = data_process_discrete(data)
            #print (error_first_temp)
            error_first = pd.concat([error_first, error_first_temp], axis = 0)
            error_Remaining = pd.concat([error_Remaining, error_Remaining_temp], axis = 0)
            Accuracy_LSTM['first'].append(Accuracy_first_temp)
            Accuracy_LSTM['Remaining'].append(Accuracy_temp)
            Accuracy_LSTM['all'].append(accuracy_all)
            Accuracy_LSTM['Card_ID'].append(Card_ID)


    ############## NG
    Card_ID_used_for_base = list(set(Card_ID_used))
    for Card_ID in Card_ID_used_for_base:
        if output_fig == 'duration':
            # file_name = data_path + 'results/result_LSTM' + str(Card_ID) + 'test' + '.csv'
            #file_name = data_path + 'results/result_LSTM_con_dur' + str(Card_ID) + 'test' + '.csv'
            file_name = data_path + 'results/result_NGRAM_con_dur_' + str(Card_ID) + '.csv'
        else:
            #file_name = data_path + 'results/result_Location_LSTM' + str(Card_ID) + 'test' + '.csv'
            file_name = data_path + 'results/result_NGRAM_location_' + str(Card_ID) + '.csv'

        if os.path.exists(file_name) == False:
            print(Card_ID,'does not exist for NG')
            continue
        data = pd.read_csv(file_name)
        if output_fig == 'duration':
            if duration_error == 'RMSE':
                R_sq_first, R_sq_Remaining, R_sq_all = data_process_continuous_RMSE(data)
            elif duration_error == 'MAPE':
                R_sq_first, R_sq_Remaining, R_sq_all = data_process_continuous_MAPE(data)
            else:
                R_sq_first, R_sq_Remaining, R_sq_all = data_process_continuous_R_sq(data)
            Accuracy_NG['first'].append(R_sq_first)
            Accuracy_NG['Remaining'].append(R_sq_Remaining)
            Accuracy_NG['all'].append(R_sq_all)
            Accuracy_NG['Card_ID'].append(Card_ID)
        else:
            error_first_temp, Accuracy_first_temp, error_Remaining_temp, Accuracy_temp, accuracy_all = data_process_discrete(data)
            #print (error_first_temp)
            error_first = pd.concat([error_first, error_first_temp], axis = 0)
            error_Remaining = pd.concat([error_Remaining, error_Remaining_temp], axis = 0)
            Accuracy_NG['first'].append(Accuracy_first_temp)
            Accuracy_NG['Remaining'].append(Accuracy_temp)
            Accuracy_NG['all'].append(accuracy_all)
            Accuracy_NG['Card_ID'].append(Card_ID)

    ############## MC

    for Card_ID in Card_ID_used_for_base:
        if output_fig == 'duration':
            # file_name = data_path + 'results/result_MC' + str(Card_ID) + '.csv'
            file_name = data_path + 'results/result_LR' + str(Card_ID) + 'test.csv'
        else:
            file_name = data_path + 'results/result_Location_MC' + str(Card_ID) + '.csv'
        if not os.path.exists(file_name):
            print(Card_ID, 'does not exist for Base')
            continue
        data = pd.read_csv(file_name)
        if output_fig == 'duration':
            if duration_error == 'RMSE':
                R_sq_first, R_sq_Remaining, R_sq_all = data_process_continuous_RMSE(data)
            elif duration_error == 'MAPE':
                R_sq_first, R_sq_Remaining, R_sq_all = data_process_continuous_MAPE(data)
            else:
                R_sq_first, R_sq_Remaining, R_sq_all = data_process_continuous_R_sq(data)
            Accuracy_base['first'].append(R_sq_first)
            Accuracy_base['Remaining'].append(R_sq_Remaining)
            Accuracy_base['all'].append(R_sq_all)
            Accuracy_base['Card_ID'].append(Card_ID)
        else:
            error_first_temp, Accuracy_first_temp, error_Remaining_temp, Accuracy_temp, accuracy_all = data_process_discrete(data)
            # print (error_first_temp)
            error_first_base = pd.concat([error_first_base, error_first_temp], axis=0)
            error_Remaining_base = pd.concat([error_Remaining_base, error_Remaining_temp], axis=0)
            Accuracy_base['first'].append(Accuracy_first_temp)
            Accuracy_base['Remaining'].append(Accuracy_temp)
            Accuracy_base['Card_ID'].append(Card_ID)
            Accuracy_base['all'].append(accuracy_all)


    # ====================


    ##############
    Accuracy_IOHMM = pd.DataFrame(Accuracy)
    Accuracy_base = pd.DataFrame(Accuracy_base)
    Accuracy_LSTM = pd.DataFrame(Accuracy_LSTM)
    Accuracy_NG = pd.DataFrame(Accuracy_NG)
    # need to drop na


    return Accuracy_IOHMM, Accuracy_base, Accuracy_LSTM, Accuracy_NG


def Bad_Card_Check(Accuracy_IOHMM, Accuracy_Base, Accuracy_LSTM):
    Accuracy_IOHMM_used = Accuracy_IOHMM.copy()
    Accuracy_Base_used = Accuracy_Base.copy()
    Accuracy_LSTM_used = Accuracy_LSTM.copy()

    print(len(Accuracy_IOHMM_used), len(Accuracy_Base_used),len(Accuracy_LSTM_used))

    Accuracy_IOHMM_used = Accuracy_IOHMM_used.drop_duplicates()
    Accuracy_Base_used = Accuracy_Base_used.drop_duplicates()
    Accuracy_LSTM_used = Accuracy_LSTM_used.drop_duplicates()

    print(len(Accuracy_IOHMM_used), len(Accuracy_Base_used),len(Accuracy_LSTM_used))

    Accuracy_comp = Accuracy_IOHMM_used.merge(Accuracy_Base_used, on = ['Card_ID'])
    Accuracy_comp['first_diff'] = Accuracy_comp['first_x'] - Accuracy_comp['first_y']
    Accuracy_comp = Accuracy_comp.sort_values(['first_diff'])
    print('IOHMM Worst', Accuracy_comp[['Card_ID','first_x','first_y']].head(5))

    Accuracy_comp = Accuracy_LSTM_used.merge(Accuracy_Base_used, on = ['Card_ID'])
    Accuracy_comp['first_diff'] = Accuracy_comp['first_x'] - Accuracy_comp['first_y']
    Accuracy_comp = Accuracy_comp.sort_values(['first_diff'])
    print('LSTM Worst', Accuracy_comp[['Card_ID','first_x','first_y']].head(5))


if __name__ == '__main__':
    data_path = '../data/'
    output_fig_list = ['location'] # location duration,'location'

    CHECK_BAD_CARD_ID = False
    PLOT_DENSITY = True

    _100samples = False
    _500samples = ~_100samples

    Save_fig = 1

    RE_CAL_ACC = False

    for output_fig in output_fig_list:

        if _100samples:
            with open (data_path + 'individual_ID_list_test', 'rb') as fp:
                individual_ID_list = pickle.load(fp)
            file_name_tail = '_100_samples'
        else:
            num_ind = 1000
            with open(data_path + 'individual_ID_list_test_' + str(num_ind) + '.pickle', 'rb') as fp:
                individual_ID_list = pickle.load(fp)
            individual_ID_list = individual_ID_list[0:500]
            file_name_tail = '_500_samples'

        if output_fig == 'duration':
            Out_put_name = 'simu_con_duration_prediction' + file_name_tail + '_add_NG.png'
            model_name_list = ['IOHMM', 'Base-LR', 'Base-LSTM']
        else:
            Out_put_name = 'simu_location_prediction' + file_name_tail + '_add_NG.png'
            model_name_list = ['IOHMM', 'Base-MC', 'Base-LSTM','Base-NG']


        duration_error = 'RMSE' # RMSE MAPE R_sq
        Mean_or_median = 'Mean' # Mean Median
        if RE_CAL_ACC:
            Accuracy_IOHMM, Accuracy_Base, Accuracy_LSTM, Accuracy_NG = generate_accuracy_file(individual_ID_list, output_fig, duration_error = duration_error) #
            with open('../data/plot_acc_add_Ngram.pickle','wb') as f:
                pickle.dump(Accuracy_IOHMM, f)
                pickle.dump(Accuracy_Base, f)
                pickle.dump(Accuracy_LSTM, f)
                pickle.dump(Accuracy_NG, f)
        else:
            with open('../data/plot_acc_add_Ngram.pickle','rb') as f:
                Accuracy_IOHMM = pickle.load(f)
                Accuracy_Base = pickle.load(f)
                Accuracy_LSTM = pickle.load(f)
                Accuracy_NG = pickle.load(f)

        ####
        if CHECK_BAD_CARD_ID:
            Bad_Card_Check(Accuracy_IOHMM, Accuracy_Base, Accuracy_LSTM)
        ########
        if PLOT_DENSITY:
            if output_fig == 'duration':
                density_plot_duration_error(Accuracy_IOHMM, Accuracy_Base, Accuracy_LSTM, save_fig=0, Out_put_name=Out_put_name,
                             model_name_list=model_name_list, Mean_or_median = 'Median') #Mean Median
                # density_plot_not_seperate_mid_first(Accuracy_IOHMM, Accuracy_MC, Accuracy_LSTM, save_fig=0, Out_put_name=Out_put_name,
                #              model_name_list=model_name_list)
                if duration_error == 'R_sq':
                    density_plot(Accuracy_IOHMM, Accuracy_Base, Accuracy_LSTM, save_fig = Save_fig,Out_put_name = Out_put_name, model_name_list = model_name_list, Mean_or_median = Mean_or_median)
            else:
                density_plot(Accuracy_IOHMM, Accuracy_Base, Accuracy_LSTM, Accuracy_NG, save_fig = Save_fig,Out_put_name = Out_put_name, model_name_list = model_name_list,Mean_or_median = Mean_or_median)


