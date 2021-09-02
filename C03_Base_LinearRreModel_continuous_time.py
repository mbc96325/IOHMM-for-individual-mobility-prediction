#
from __future__ import division

import timeit

from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib
from sklearn.linear_model import LinearRegression

from sklearn import linear_model
from sklearn.metrics import r2_score

from scipy.special import logsumexp
import pickle
from copy import deepcopy
import random
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import os



Accurate_duration = []
# filename1='data/activity_index_test.txt'
# file1=open(filename1,'r')
# activity_index_test=eval(file1.read())
activity_index_test = {}


def process_data(data, test_proportion,Card_ID, test_last, dep_var):
    #data['duration'] = np.log(data['duration'])  # log for better modeling
    data.loc[data['duration_last'] == -1, 'duration_last'] = 0  # first activity, assign to 0
    ####generate a new columns: first * mean of first
    data['if_first'] = 0
    data.loc[data['act_ID'] == 0, 'if_first'] = 1
    mean_of_first = np.mean(data.loc[data['if_first'] == 1,'duration'])
    data['dur_first'] = mean_of_first #* data['if_first']
    #################
    column_list = list(data.columns.values)
    location_list = []
    hour_list = []
    for ele in column_list:
        if 'location' in ele:
            location_list.append(ele)
        if 'hour' in ele:
            hour_list.append(ele)
    location_list.remove('location_o')
    location_list.remove('location')
    hour_list.remove('hour')
    hour_list.remove('duration_hour')
    # set covariates to this OLS model
    weather_list=['rain','heavy_rain','sun','cloud','Avrg_Temp','fengli']
    Weekday_list=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    holiday_list=['National_holiday','Observance']
    last_activity=['duration_last','duration_trip']
    previous_trips = ['Last_trip_time_yesterday','N_days_withtrip_past20',
                      'N_consec_days_no_trips','N_trips_yesterday']
    Other = ['if_first']

    Ut_list=weather_list + hour_list + Weekday_list+ location_list + holiday_list +last_activity + previous_trips + Other
    # U1_list=Weekday_list+weather_list + holiday_list
    x_array = np.array(data.loc[:,Ut_list])
    min_max_scaler = preprocessing.MinMaxScaler()
    x_array_minmax = min_max_scaler.fit_transform(x_array)
    print(x_array_minmax.shape)

    min_max_scaler_dep = preprocessing.MinMaxScaler()
    y_minmax = min_max_scaler_dep.fit_transform(np.array(data.loc[:,dep_var]))

    data.loc[:, Ut_list] = x_array_minmax
    data.loc[:, dep_var] = y_minmax

    total_days = data['seq_ID'].max()
    train_days = int(total_days - round(total_days*test_proportion))
    # drop last
    #data = data.loc[data['if_last']!=1]

    if test_last:
        # last 30 days
        data_train = data.loc[data['seq_ID']<=train_days]
        data_test = data.loc[data['seq_ID']>train_days]
    else:
        random.seed(Card_ID)
        test_seq = random.sample(list(range(1,total_days+1)), total_days - train_days)
        data_train = data.loc[~data['seq_ID'].isin(test_seq)]
        data_test = data.loc[data['seq_ID'].isin(test_seq)]

    return min_max_scaler,min_max_scaler_dep, data, data_train, data_test, Ut_list



def gen_sequence(id_df, seq_length, seq_cols):
    '''
    padding with zero
    '''
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,112),(50,192)
    # 0 50 -> from row 0 to row 50
    # 1 51 -> from row 1 to row 51
    # 2 52 -> from row 2 to row 52
    # ...
    # 111 191 -> from row 111 to 191

    for start, stop in zip(range(-seq_length+1, num_elements), range(1, num_elements+1)):
        if start<0: # padding with zero
            padding = np.zeros([-start, data_matrix.shape[1]])
            used_data = data_matrix[0:stop, :]
            yield np.vstack([padding, used_data])
        else:
            yield data_matrix[start:stop, :]


def gen_labels(id_df, seq_length, label):
    # For one id I put all the labels in a single matrix.
    # For example:
    # [[1]
    # [4]
    # [1]
    # [5]
    # [9]
    # ...
    # [200]]
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[0:num_elements, :]

def pre_process_to_LSTM(data_train, data_test, Ut_list, depend_var, sequence_length):


    # test = list(gen_sequence(data_train[data_train['seq_ID'] == 59], sequence_length, Ut_list))

    seq_gen_train = (list(gen_sequence(data_train[data_train['seq_ID'] == idx], sequence_length, Ut_list))
               for idx in data_train['seq_ID'].unique())
    seq_gen_test = (list(gen_sequence(data_test[data_test['seq_ID'] == idx], sequence_length, Ut_list))
               for idx in data_test['seq_ID'].unique())
    seq_array_train = np.concatenate(list(seq_gen_train)).astype(np.float32)
    seq_array_test = np.concatenate(list(seq_gen_test)).astype(np.float32)

    # generate labels
    # val_label = gen_labels(data_train[data_train['seq_ID']==59], sequence_length, depend_var)
    data = data_train.append(data_test)
    # label_gen_train = [gen_labels(data_train[data_train['seq_ID']==idx], sequence_length, depend_var)
    #              for idx in data_train['seq_ID'].unique()]
    # label_gen_test = [gen_labels(data_test[data_test['seq_ID']==idx], sequence_length, depend_var)
    #              for idx in data_test['seq_ID'].unique()]
    dict_label = sorted(list(pd.unique(data.loc[:,depend_var[0]])))
    dict_label2 = {}
    idx = 0
    for key in dict_label:
        dict_label2[key] = idx
        idx += 1
    data['new_dep'] = data[depend_var[0]].apply(lambda x: dict_label2[x])
    label_gen = [gen_labels(data[data['seq_ID']==idx], sequence_length, ['new_dep'])
                 for idx in data['seq_ID'].unique()]
    label_gen = np.concatenate(label_gen).astype(np.int32)


    label_gen = to_categorical(label_gen, num_classes=len(dict_label))
    # label_array_train = np.concatenate(label_gen_train).astype(np.int32)
    # label_array_test = np.concatenate(label_gen_test).astype(np.int32)

    label_array_train = label_gen[0:len(data_train),:]
    label_array_test = label_gen[len(data_train):,:]


    return seq_array_train, seq_array_test, label_array_train, label_array_test, dict_label


def Model(Card_ID):
    file_name_test_results = data_path + 'results/result_LR' + str(Card_ID) + 'test.csv'

    # if Card_ID in activity_index_test:
    # print ('Running model', Card_ID)
    # return
    file_name_train = data_path + 'samples/sample_' + str(Card_ID) + '_201407_201408_all.csv'
    data = pd.read_csv(file_name_train)
    data = data.loc[data['if_last']==0,:] # drop the last one, it will distract the training (because it is manually added)
    test_proportion = 0.2
    data['duration_hour'] = round(data['duration'] / 3600).astype('int')
    depend_var = ['duration']  # regression
    #=========================    #data_preprocessing
    test_last = False
    scaler,scaler_y, data, data_train, data_test, Ut_list = process_data(data, test_proportion,Card_ID, test_last, depend_var)




    # data_train[depend_var] = scaler.fit_transform(data_train[depend_var])
    # data_test[depend_var] = scaler.fit_transform(data_test[depend_var])


    reg = LinearRegression().fit(data_train[Ut_list], data_train[depend_var])

    results_test = data_test.loc[:,['ID','act_ID'] + depend_var]
    results_test['Card_ID'] = Card_ID
    results_test = results_test.reset_index(drop=True)
    results_test['Predict_duration'] = reg.predict(data_test[Ut_list])

    results_test['Predict_duration'] = scaler_y.inverse_transform(results_test['Predict_duration'].values.reshape(-1,1))
    results_test['duration'] = scaler_y.inverse_transform(results_test['duration'].values.reshape(-1,1))

    results_test = results_test.rename(columns = {'act_ID':'activity_index','duration':'Ground_truth_duration'})



    file_name_test_results = data_path + 'results/result_LR' + str(Card_ID) + 'test.csv'
    results_test.to_csv(file_name_test_results, index=False)


    # train

    results_train = data_train.loc[:,['ID','act_ID'] + depend_var]
    results_train['Card_ID'] = Card_ID
    results_train = results_train.reset_index(drop=True)
    results_train['Predict_duration'] = reg.predict(data_train[Ut_list])

    results_train['Predict_duration'] = scaler_y.inverse_transform(results_train['Predict_duration'].values.reshape(-1,1))
    results_train['duration'] = scaler_y.inverse_transform(results_train['duration'].values.reshape(-1,1))

    results_train = results_train.rename(columns = {'act_ID':'activity_index','duration':'Ground_truth_duration'})

    file_name_train_results = data_path + 'results/result_LR' + str(Card_ID) + 'train.csv'
    results_train.to_csv(file_name_train_results, index=False)
    return results_test, results_train

def calculate_accuracy(result_df):
    N_first = result_df['Correct'].loc[result_df['activity_index']==0].count()
    Accuracy_first = result_df['Correct'].loc[(result_df['Correct']==1)&
               (result_df['activity_index']==0)].count()/N_first

    N_middle = result_df['Correct'].loc[result_df['activity_index']!=0].count()
    Accuracy_middle = result_df['Correct'].loc[(result_df['Correct']==1)&
               (result_df['activity_index']!=0)].count()/N_middle

    N_all = result_df['Correct'].count()
    Accuracy_all = result_df['Correct'].loc[result_df['Correct']==1].count()/N_all


    return Accuracy_first, Accuracy_middle, Accuracy_all, N_first, N_middle, N_all


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
    R_sq = r2_score(result_df['Ground_truth_duration'], result_df['Predict_duration'])
    return RMSE, MAPE, MAE, R_sq

if __name__ == '__main__':
    # card_ID = 954394568
    # individual_ID_list_test = [958306207]

    data_path = '../data/'
    # with open(data_path + 'individual_ID_list_test', 'rb') as fp:
    #     individual_ID_list_test = pickle.load(fp)

    num_ind = 1000
    with open(data_path + 'individual_ID_list_test_' + str(num_ind) + '.pickle', 'rb') as fp:
        individual_ID_list_test = pickle.load(fp)
    individual_ID_list_test = individual_ID_list_test[0:500]

    count = 0
    for Card_ID in individual_ID_list_test:
        print('Current Card ID',Card_ID )
        results_test, results_train = Model(Card_ID)
        RMSE_test, MAPE_test, MAE_test, R_sq_test = calculate_error(results_test)
        RMSE_train, MAPE_train, MAE_train, R_sq_train = calculate_error(results_train)
        # result_df_MC = pd.read_csv(data_path + 'results/result_MC' + str(Card_ID) + '.csv')
        # _, _, Accuracy_MC, _, _, _ = calculate_accuracy(result_df_MC)
        # result_df_IOHMM = pd.read_csv(data_path + 'results/result_' + str(Card_ID) + 'test.csv')
        # _, _, Accuracy_IOHMM, _, _, _ = calculate_accuracy(result_df_IOHMM)
        count+=1
        print ('Num_people_processed', count)
        print(Card_ID, 'Total Testing R_sq:', R_sq_test)
        print(Card_ID, 'Total Training R_sq:', R_sq_train)
        print('------****------')
        # pool = multiprocessing.Pool(processes=3)
    # pool.map(Model, individual_ID_list_test)
    # pool.close()
    # print ('Accurate_duration',sum(Accurate_duration)/len(Accurate_duration))

    # filename1='data/activity_index_test.txt'
    # file1=open(filename1,'r')
    # activity_index_test=eval(file1.read())