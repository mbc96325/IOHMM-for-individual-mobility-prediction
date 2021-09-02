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
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation

from IOHMM import UnSupervisedIOHMM
from IOHMM import OLS, DiscreteMNL, CrossEntropyMNL
from IOHMM import forward_backward
from scipy.special import logsumexp
import pickle
from copy import deepcopy
import random
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import os
from sklearn.metrics import r2_score, mean_squared_error

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.utils import to_categorical
from keras import backend as K
from tensorflow.keras import regularizers
from keras.layers import Multiply

from keras.layers import Input, Dense, Add, Concatenate
from keras.models import Model

# with open ('data/individual_ID_list', 'rb') as fp:
# individual_ID_list = pickle.load(fp)

# num_of_test_samples=500
# individual_ID_list_test = [ day_list[i] for i in sorted(random.sample(range(len(day_list)), num_of_test_samples)) ]


Accurate_duration = []
# filename1='data/activity_index_test.txt'
# file1=open(filename1,'r')
# activity_index_test=eval(file1.read())
activity_index_test = {}


def process_data(data, test_proportion,Card_ID, test_last, dep_var):
    #data['duration'] = np.log(data['duration'])  # log for better modeling
    data.loc[data['duration_last'] == -1, 'duration_last'] = 0  # first activity, assign to 0
    column_list = list(data.columns.values)
    location_list = []
    hour_list = []

    data['if_first'] = 0
    data.loc[data['act_ID'] == 0, 'if_first'] = 1

    for ele in column_list:
        if 'location' in ele:
            location_list.append(ele)
        if 'hour' in ele:
            hour_list.append(ele)
    location_list.remove('location_o')
    location_list.remove('location')
    hour_list.remove('hour')
    if 'duration_hour' in hour_list:
        hour_list.remove('duration_hour')
    # set covariates to this OLS model
    weather_list=['rain','heavy_rain','sun','cloud','Avrg_Temp','fengli']
    Weekday_list=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    holiday_list=['National_holiday','Observance']
    last_activity=['duration_last','duration_trip']
    previous_trips = ['Last_trip_time_yesterday','N_days_withtrip_past20',
                      'N_consec_days_no_trips','N_trips_yesterday']
    Other = ['if_first']

    Ut_list = weather_list + hour_list + Weekday_list+ location_list + holiday_list +last_activity + previous_trips + Other

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

    return min_max_scaler, min_max_scaler_dep, data, data_train, data_test, Ut_list



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

    # print(seq_array_train.shape)
    # print(seq_array_test.shape)

    # print(seq_array_train[0])

    label_gen_train = [gen_labels(data_train[data_train['seq_ID']==idx], sequence_length, depend_var)
                 for idx in data_train['seq_ID'].unique()]
    label_gen_test = [gen_labels(data_test[data_test['seq_ID']==idx], sequence_length, depend_var)
                 for idx in data_test['seq_ID'].unique()]

    label_gen_train = np.concatenate(list(label_gen_train)).astype(np.float32)
    label_gen_test = np.concatenate(list(label_gen_test)).astype(np.float32)

    # print(label_gen_train.shape)
    # print(label_gen_test.shape)

    # a=1
    # dict_label = sorted(list(pd.unique(data.loc[:,depend_var[0]])))
    # dict_label2 = {}
    # idx = 0
    # for key in dict_label:
    #     dict_label2[key] = idx
    #     idx += 1
    # data['new_dep'] = data[depend_var[0]].apply(lambda x: dict_label2[x])
    # label_gen = [gen_labels(data[data['seq_ID']==idx], sequence_length, ['new_dep'])
    #              for idx in data['seq_ID'].unique()]
    # label_gen = np.concatenate(label_gen).astype(np.int32)
    #
    #
    # label_gen = to_categorical(label_gen, num_classes=len(dict_label))
    # # label_array_train = np.concatenate(label_gen_train).astype(np.int32)
    # # label_array_test = np.concatenate(label_gen_test).astype(np.int32)
    #
    # label_array_train = label_gen[0:len(data_train),:]
    # label_array_test = label_gen[len(data_train):,:]


    return seq_array_train, seq_array_test, label_gen_train, label_gen_test



def Model_LSTM(Card_ID, RE_RUN, PLOT_HISTORY):
    file_name_test_results = data_path + 'results/result_LSTM_con_dur' + str(Card_ID) + 'test.csv'
    model_path =  'output/LSTM/' + 'model_' + str(Card_ID) + '.h5'
    if not RE_RUN:
        if os.path.exists(file_name_test_results):
            print('Finish model', Card_ID)
            return
    # if Card_ID in activity_index_test:
    # print ('Running model', Card_ID)
    # return
    file_name_train = data_path + 'samples/sample_' + str(Card_ID) + '_201407_201408_all.csv'
    data = pd.read_csv(file_name_train)
    data = data.loc[data['if_last']==0,:] # drop the last one, it will distract the training (because it is manually added)
    test_proportion = 0.2
    #=========================    #data_preprocessing
    test_last = False
    depend_var = ['duration'] ## regression

    scaler, scaler_y, data, data_train, data_test, Ut_list = process_data(data, test_proportion,Card_ID, test_last, depend_var)

    def R_sq(y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))


    sequence_length = 2  # look back period, use 2 because most people only has 2 trips.

    seq_array_train, seq_array_test, label_array_train, label_array_test = pre_process_to_LSTM(data_train, data_test, Ut_list, depend_var, sequence_length)

    # print(seq_array_train.shape, seq_array_test.shape, label_array_train.shape, label_array_test.shape)
    nb_features = seq_array_train.shape[2]

    #===========================##
    # design network
    model = Sequential()
    model.add(LSTM(
        input_shape=(sequence_length, nb_features),
        units=50,
        return_sequences=False
        )) #
    #         kernel_regularizer= regularizers.l1_l2(l1=1e-2, l2=1e-2)
    #         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)
    # kernel_regularizer = regularizers.l2(1e-3),
    # bias_regularizer = regularizers.l2(1e-3)
    model.add(Dropout(0.3))
    # model.add(LSTM(
    #     units=50,
    #     return_sequences=False))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=50))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    #                     kernel_regularizer=regularizers.l2(1e-3),
    #                     bias_regularizer=regularizers.l2(1e-3)
    model.add(Dense(units=1))

    ###################################
    # inp = Input(shape=(sequence_length, nb_features))
    # d = LSTM(units = 64, return_sequences=False)(inp)
    # d = Dropout(0.5)(d)
    # inp2 = Input(shape=(nb_features,))
    # conc = Concatenate()([d, inp2])
    # # out = Dense(units=1,activation='linear')(conc)
    # out = Dense(units=1, activation='linear')(inp2)
    # model = Model(inputs=[inp,inp2], outputs=out)
    # x_inp2_train = data_train[Ut_list].values
    # x_inp2_test = data_test[Ut_list].values
    ###################################
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=[R_sq])
    # print(model.summary())


    # history = model.fit([seq_array_train, x_inp2_train], label_array_train, epochs=200, batch_size=30, verbose=0,
    #                     validation_data=([seq_array_test, x_inp2_test], label_array_test)
    #                     )
    # fit the network

    history = model.fit(seq_array_train, label_array_train, epochs=200, batch_size=30, verbose=0,
                        validation_data=(seq_array_test, label_array_test),
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='R_sq', min_delta=0.03, patience=30, verbose=0,
                                                          mode='max'),
                            keras.callbacks.ModelCheckpoint(model_path, monitor='R_sq', save_best_only=True,
                                                            mode='max', verbose=0)]
                        )

    # history = model.fit([seq_array_train, x_inp2_train], label_array_train, epochs=200, batch_size=30, verbose=0,
    #                     validation_data=([seq_array_test, x_inp2_test], label_array_test),
    #                     callbacks=[
    #                         keras.callbacks.EarlyStopping(monitor='R_sq', min_delta=0.03, patience=30, verbose=0,
    #                                                       mode='max'),
    #                         keras.callbacks.ModelCheckpoint(model_path, monitor='R_sq', save_best_only=True,
    #                                                         mode='max', verbose=0)]
    #                     )

    # history = model.fit(seq_array_train, label_array_train, epochs=200, batch_size=30, verbose=2,
    #                     validation_data=(seq_array_test, label_array_test)
    #                     )

    #####################################plot history

    if PLOT_HISTORY:
        fig_acc = plt.figure(figsize=(10, 10))
        plt.plot(history.history['R_sq'])
        plt.plot(history.history['val_R_sq'])
        plt.title('model accuracy')
        plt.ylabel('R_sq')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.ylim([0,1])
        plt.show()
        # #
        # # summarize history for Loss
        fig_loss = plt.figure(figsize=(10, 10))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    #
    #########################
    # test
    # y_test_pred = model.predict([seq_array_test, x_inp2_test], verbose=0)
    # y_train_pred = model.predict([seq_array_train, x_inp2_train], verbose=0)

    y_test_pred = model.predict(seq_array_test, verbose=0)
    y_train_pred = model.predict(seq_array_train, verbose=0)

    data_train[depend_var] = scaler_y.inverse_transform(data_train[depend_var])
    data_test[depend_var] = scaler_y.inverse_transform(data_test[depend_var])

    y_test_pred = scaler_y.inverse_transform(y_test_pred)
    y_train_pred = scaler_y.inverse_transform(y_train_pred)


    results_test = data_test.loc[:,['ID','duration','act_ID']]
    results_test['Card_ID'] = Card_ID
    results_test = results_test.reset_index(drop=True)

    results_test = results_test.rename(columns = {'duration':'Ground_truth_duration','act_ID':'activity_index'})
    results_test['Predict_duration'] = y_test_pred.reshape(len(results_test))

    num1 = len(results_test)
    results_test = results_test.dropna()
    num2 = len(results_test)
    if num2 < num1:
        print('ID',Card_ID, 'DropNA test',num2 - num1)
    file_name_test_results = data_path + 'results/result_LSTM_con_dur' + str(Card_ID) + 'test.csv'
    results_test.to_csv(file_name_test_results, index=False)


    # train
    results_train = data_train.loc[:,['ID','duration','act_ID']]
    results_train['Card_ID'] = Card_ID
    results_train = results_train.reset_index(drop=True)

    results_train = results_train.rename(columns = {'duration':'Ground_truth_duration','act_ID':'activity_index'})
    results_train['Predict_duration'] = y_train_pred.reshape(len(results_train))

    num1 = len(results_train)
    results_train = results_train.dropna()
    num2 = len(results_train)
    if num2 < num1:
        print('ID',Card_ID, 'DropNA train',num2 - num1)


    file_name_train_results = data_path + 'results/result_LSTM_con_dur' + str(Card_ID) + 'train.csv'
    results_train.to_csv(file_name_train_results,index=False)

    RMSE_test, MAPE_test, MAE_test, R_sq_test = calculate_error(results_test)
    RMSE_train, MAPE_train, MAE_train, R_sq_train = calculate_error(results_train)

    # print('RMSE_train', RMSE_train)
    # print('RMSE_test',RMSE_test)
    #
    # print('R_sq_test',R_sq_test)
    # print('R_sq_train',R_sq_train)

    return R_sq_test, R_sq_train

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
    # individual_ID_list_test = [individual_ID_list_test[23]]
    data_path = '../data/'

    # with open(data_path + 'individual_ID_list_test', 'rb') as fp:
    #     individual_ID_list_test = pickle.load(fp)

    num_ind = 1000
    with open(data_path + 'individual_ID_list_test_' + str(num_ind) + '.pickle', 'rb') as fp:
        individual_ID_list_test = pickle.load(fp)
    individual_ID_list_test = individual_ID_list_test[0:500]

    count = 0
    RE_RUN = True
    PLOT_HISTORY = False
    for Card_ID in individual_ID_list_test:
        print('Current Card ID',Card_ID )
        R_sq_test, R_sq_train = Model_LSTM(Card_ID,RE_RUN, PLOT_HISTORY)
        if R_sq_test is None:
            result_df_MC = pd.read_csv(data_path + 'results/result_LSTM_con_dur' + str(Card_ID) + 'test.csv')
            a=1
        result_df_LR_duration = pd.read_csv(data_path + 'results/result_LR' + str(Card_ID) + 'test.csv')
        RMSE_LR, MAPE_LR, MAE_LR, R_sq_LR  = calculate_error(result_df_LR_duration)

        count+=1
        print ('Num_people_processed', count, 'Total', len(individual_ID_list_test))
        print(Card_ID, 'Total Training R_sq:', R_sq_train)
        print(Card_ID, 'LSTM Dur:', R_sq_test,'LR Dur:', R_sq_LR)
        print('------****------')
        # a=1
        # pool = multiprocessing.Pool(processes=3)
    # pool.map(Model, individual_ID_list_test)
    # pool.close()
    # print ('Accurate_duration',sum(Accurate_duration)/len(Accurate_duration))

    # filename1='data/activity_index_test.txt'
    # file1=open(filename1,'r')
    # activity_index_test=eval(file1.read())