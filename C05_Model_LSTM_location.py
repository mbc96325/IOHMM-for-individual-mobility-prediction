#
from __future__ import division

import timeit
import time
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

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.utils import to_categorical
from keras import backend as K
from tensorflow.keras import regularizers
# with open ('data/individual_ID_list', 'rb') as fp:
# individual_ID_list = pickle.load(fp)

# num_of_test_samples=500
# individual_ID_list_test = [ day_list[i] for i in sorted(random.sample(range(len(day_list)), num_of_test_samples)) ]

Accurate_duration = []
# filename1='data/activity_index_test.txt'
# file1=open(filename1,'r')
# activity_index_test=eval(file1.read())
activity_index_test = {}


def process_data(data, test_proportion,Card_ID, test_last):
    #data['duration'] = np.log(data['duration'])  # log for better modeling
    data.loc[data['duration_last'] == -1, 'duration_last'] = 0  # first activity, assign to 0
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
    # set covariates to this OLS model
    weather_list=['rain','heavy_rain','sun','cloud','Avrg_Temp','fengli']
    Weekday_list=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    holiday_list=['National_holiday','Observance']
    last_activity=['duration_last','duration_trip']
    previous_trips = ['Last_trip_time_yesterday','N_days_withtrip_past20',
                      'N_consec_days_no_trips','N_trips_yesterday']

    Ut_list=weather_list + hour_list + Weekday_list+ location_list + holiday_list +last_activity + previous_trips
    # U1_list=Weekday_list+weather_list + holiday_list
    x_array = np.array(data.loc[:,Ut_list])
    min_max_scaler = preprocessing.MinMaxScaler()
    x_array_minmax = min_max_scaler.fit_transform(x_array)
    print(x_array_minmax.shape)

    weather_list=['rain','heavy_rain','sun','cloud','Avrg_Temp','fengli']
    Weekday_list=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    holiday_list=['National_holiday','Observance']
    last_activity=['duration_last','duration_trip']
    previous_trips = ['Last_trip_time_yesterday','N_days_withtrip_past20',
                      'N_consec_days_no_trips','N_trips_yesterday']
    Ut_list = weather_list + hour_list + Weekday_list + location_list + holiday_list + last_activity + previous_trips
    # U1_list=Weekday_list+weather_list + holiday_list
    data_array = np.array(data.loc[:, Ut_list])
    min_max_scaler = preprocessing.MinMaxScaler()
    array_minmax = min_max_scaler.fit_transform(data_array)
    data.loc[:, Ut_list] = array_minmax
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

    return min_max_scaler, data, data_train, data_test, Ut_list



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


def Model(Card_ID, RE_RUN):
    file_name_test_results = data_path + 'results/result_Location_LSTM' + str(Card_ID) + 'test.csv'
    model_path =  'output/LSTM/' + 'model_Location_' + str(Card_ID) + '.h5'
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
    scaler, data, data_train, data_test, Ut_list = process_data(data, test_proportion,Card_ID, test_last)
    data_train['duration_hour'] = round(data['duration'] / 3600).astype('int')  # classification
    data_test['duration_hour'] = round(data['duration'] / 3600).astype('int')  # classification
    depend_var = ['Next_tapin_station']

    sequence_length = 2  # look back period, use 2 because most people only has 2 trips.

    seq_array_train, seq_array_test, label_array_train, label_array_test, dict_label = pre_process_to_LSTM(data_train, data_test, Ut_list, depend_var, sequence_length)

    # print(seq_array_train.shape, seq_array_test.shape, label_array_train.shape, label_array_test.shape)
    nb_features = seq_array_train.shape[2]
    nb_out = label_array_train.shape[1]
    #===========================
    # design network
    model = Sequential()
    model.add(LSTM(
        input_shape=(sequence_length, nb_features),
        units=50,
        return_sequences=False,
        )) #
    model.add(Dropout(0.05))
    # model.add(Dense(units=50, activation='relu'))
    # model.add(LSTM(
    #     units=50,
    #     return_sequences=False))
    # model.add(Dropout(0.05))
    # opt = keras.optimizers.SGD(lr=1e-2)
    model.add(Dense(units=nb_out, activation='sigmoid',name='output_rank'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())

    # fit the network
    history = model.fit(seq_array_train, label_array_train, epochs=200, batch_size=30, verbose=0,
                        validation_data=(seq_array_test, label_array_test),
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=0,
                                                          mode='max'),
                            keras.callbacks.ModelCheckpoint(model_path, monitor='val_acc', save_best_only=True,
                                                            mode='max', verbose=0)]
                        )

    # history = model.fit(seq_array_train, label_array_train, epochs=200, batch_size=30, verbose=2,
    #                     validation_data=(seq_array_test, label_array_test)
    #                     )

    #####################################plot history

    # fig_acc = plt.figure(figsize=(10, 10))
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # #
    # # summarize history for Loss
    # fig_loss = plt.figure(figsize=(10, 10))
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    #
    #########################
    # test
    if os.path.isfile(model_path):
        estimator = load_model(model_path)
    get_last_layer_output = K.function([estimator.layers[0].input],
                                      [estimator.get_layer('output_rank').output])
    layer_output = get_last_layer_output([seq_array_test])[0]
    top_N = np.min([20, nb_out])

    idx_top_N = np.argsort(-layer_output, axis = 1) # use negative because from small to large
    idx_top_N = idx_top_N[:,0:top_N]
    results = data_test.loc[:,['ID',depend_var[0],'act_ID']]
    results['Card_ID'] = Card_ID
    results = results.reset_index(drop=True)
    predict_topN = [np.array(dict_label)[row_index.astype('int')] for row_index in idx_top_N]
    pred_col = ['Predict' + str(i + 1) for i in range(top_N)]
    results_predict = pd.DataFrame(predict_topN, columns= pred_col)
    results = pd.concat([results, results_predict],axis=1)
    results = results.rename(columns = {depend_var[0]:'Ground_truth','act_ID':'activity_index'})
    results['Correct'] = 0
    results.loc[results['Predict1'] == results['Ground_truth'],'Correct'] = 1
    test_acc = sum(results['Correct'])/len(results)
    if top_N < 20:
        for k in range(top_N+1,20+1):
            results['Predict'+str(k)] = -1

    file_name_test_results = data_path + 'results/result_Location_LSTM' + str(Card_ID) + 'test.csv'
    results.to_csv(file_name_test_results, columns=['ID','Card_ID'] + ['Predict' + str(i + 1) for i in range(20)] + ['Ground_truth', 'Correct', 'activity_index'],index=False)


    # train
    get_last_layer_output = K.function([model.layers[0].input],
                                      [model.get_layer('output_rank').output])
    layer_output = get_last_layer_output([seq_array_train])[0]

    idx_top_N = np.argsort(-layer_output, axis = 1) # use negative because from small to large
    idx_top_N = idx_top_N[:,0:top_N]
    results = data_train.loc[:,['ID',depend_var[0],'act_ID']]
    results['Card_ID'] = Card_ID
    results = results.reset_index(drop=True)
    predict_topN = [np.array(dict_label)[row_index.astype('int')] for row_index in idx_top_N]
    pred_col = ['Predict' + str(i + 1) for i in range(top_N)]
    results_predict = pd.DataFrame(predict_topN, columns= pred_col)
    results = pd.concat([results, results_predict],axis=1)
    results = results.rename(columns = {depend_var[0]:'Ground_truth','act_ID':'activity_index'})
    results['Correct'] = 0
    results.loc[results['Predict1'] == results['Ground_truth'],'Correct'] = 1
    train_acc = sum(results['Correct']) / len(results)
    print('Train accuracy', train_acc)
    if top_N < 20:
        for k in range(top_N+1,20+1):
            results['Predict'+str(k)] = -1
    file_name_train_results = data_path + 'results/result_Location_LSTM' + str(Card_ID) + 'train.csv'
    results.to_csv(file_name_train_results,columns=['ID','Card_ID'] + ['Predict' + str(i + 1) for i in range(20)] + ['Ground_truth', 'Correct', 'activity_index'],index=False)

    return test_acc

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

if __name__ == '__main__':
    # card_ID = 954394568
    # individual_ID_list_test = [958999238]

    data_path = '../data/'
    # with open(data_path + 'individual_ID_list_test', 'rb') as fp:
    #     individual_ID_list_test = pickle.load(fp)

    SHOW_BASELINE = False
    SKIP_RUNNED_MODEL = True

    num_ind = 1000
    with open(data_path + 'individual_ID_list_test_' + str(num_ind) + '.pickle', 'rb') as fp:
        individual_ID_list_test = pickle.load(fp)
    individual_ID_list_test = individual_ID_list_test[0:500]

    count = 0
    RE_RUN = True
    tic = time.time()
    for Card_ID in individual_ID_list_test:
        count+=1
        print('Current Card ID',Card_ID,'count',count, 'total',len(individual_ID_list_test))
        file_name_test_ = data_path + 'results/result_Location_LSTM' + str(Card_ID) + 'test.csv'
        if SKIP_RUNNED_MODEL:
            if os.path.exists(file_name_test_):
                print ('Finish model', Card_ID)
                continue

        test_acc = Model(Card_ID,RE_RUN)
        if test_acc is None:
            result_df = pd.read_csv(data_path + 'results/result_Location_LSTM' + str(Card_ID) + 'test.csv')
            _, _, test_acc, _, _, _ = calculate_accuracy(result_df)
        if SHOW_BASELINE:
            result_df_MC = pd.read_csv(data_path + 'results/result_Location_MC' + str(Card_ID) + '.csv')
            _, _, Accuracy_MC, _, _, _ = calculate_accuracy(result_df_MC)
            result_df_IOHMM = pd.read_csv(data_path + 'results/result_Location_' + str(Card_ID) + 'test.csv')
            _, _, Accuracy_IOHMM, _, _, _ = calculate_accuracy(result_df_IOHMM)
        else:
            Accuracy_MC = -1
            Accuracy_IOHMM = -1

        print ('Num_people_processed', count)
        print(Card_ID, 'Total Testing Accuracy:', test_acc)
        print(Card_ID, 'Base Total Testing Accuracy:', Accuracy_MC)
        print(Card_ID, 'IOHMM Total Testing Accuracy:', Accuracy_IOHMM)
        print('Elapsed time', time.time() - tic)
        print('------****------')
        # pool = multiprocessing.Pool(processes=3)
    print('Total time', time.time() - tic)
    # pool.map(Model, individual_ID_list_test)
    # pool.close()
    # print ('Accurate_duration',sum(Accurate_duration)/len(Accurate_duration))

    # filename1='data/activity_index_test.txt'
    # file1=open(filename1,'r')
    # activity_index_test=eval(file1.read())