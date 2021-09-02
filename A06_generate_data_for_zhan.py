
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



from scipy.special import logsumexp
import pickle
from copy import deepcopy
import random

import os





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

    return None,None, data, data_train, data_test, Ut_list


data_path = '../data/'
# with open(data_path + 'individual_ID_list_test', 'rb') as fp:
#     individual_ID_list_test = pickle.load(fp)

num_ind = 1000
with open(data_path + 'individual_ID_list_test_' + str(num_ind) + '.pickle', 'rb') as fp:
    individual_ID_list_test = pickle.load(fp)
individual_ID_list_test = individual_ID_list_test[0:500]

count = 0


station_idx = pd.read_csv('data_for_zhan/data/station_vocab_hk.csv', header=None)
station_idx.columns = ['MTR_ID', 'Name', 'Line', 'Short_name', 'New_ID']
station_idx['New_ID_str'] = station_idx['New_ID'].astype('int').astype('str')
station_idx_new_to_old_dict = {}
station_idx_old_to_new_dict = {}
for old_id, new_id in zip(station_idx['MTR_ID'], station_idx['New_ID_str']):
    station_idx_old_to_new_dict[old_id] = new_id


USED_TOD_LIST = {}
def retrieve_trip_based_data(data,user_id):
    data_used = data.copy()
    data_used['cum_trip_time'] =  data_used.groupby(['seq_ID'])['duration_trip'].cumsum()
    data_used['cum_duration_time'] = data_used.groupby(['seq_ID'])['duration_last'].cumsum()
    data_used['Next_tapin_time'] = data_used['cum_trip_time'] + data_used['cum_duration_time'] + data_used['duration']
    data_used['Next_tapout_station'] = data_used.shift(-1)['location']
    data_used['trip_time'] = data_used.shift(-1)['duration_trip']
    data_used['Next_tapout_time'] = data_used['Next_tapin_time'] + data_used['trip_time']

    data_used = data_used.loc[data_used['if_last']!=1]

    trip_based = data_used.loc[:,['date','Next_tapin_time','Next_tapin_station','Next_tapout_time','Next_tapout_station']]
    trip_based['user_id'] = user_id

    # only preserve used id
    next_tapin_station_used = []
    next_tapout_station_used = []
    time_hour_used = []

    # revise to Zhan's index

    for key in data_used.columns:
        if 'location_' in key and 'location_o' not in key:
            loc = int(key.split('location_')[1])
            next_tapin_station_used.append(station_idx_old_to_new_dict[loc])
        if 'location_o' in key and key != 'location_o':
            loc_o = int(key.split('location_o')[1])
            next_tapout_station_used.append(station_idx_old_to_new_dict[loc_o])
        if 'hour_' in key:
            time_hour_used.append(str(int(key.split('hour_')[1])))
        #if ''
        #a=1

    trip_based['entry_station_id'] = trip_based['Next_tapin_station'].astype('int')
    trip_based['exit_station_id'] = trip_based['Next_tapout_station'].astype('int')
    USED_TOD_LIST[user_id] = {'o':next_tapin_station_used,'t':time_hour_used,'d':next_tapout_station_used}
    # trip_based.loc[~trip_based['exit_station_id'].isin(next_tapout_station_used), 'exit_station_id'] = -99
    # trip_based.loc[~trip_based['entry_station_id'].isin(next_tapin_station_used), 'entry_station_id'] = -99
    #
    trip_based.loc[trip_based['entry_station_id'] == -1, 'entry_station_id'] = -99
    trip_based.loc[trip_based['exit_station_id'] == -1, 'exit_station_id'] = -99



    trip_based['entry_time'] = np.round(trip_based['Next_tapin_time']/60)
    trip_based['exit_time'] = np.round(trip_based['Next_tapout_time']/60)
    trip_based['entry_date'] = pd.to_datetime(trip_based['date'], origin='unix',format='%Y-%m-%d')
    trip_based['exit_date'] = trip_based['entry_date']
    trip_based['entry_time'] = trip_based['entry_time'].astype('int')
    trip_based['exit_time'] = trip_based['exit_time'].astype('int')
    trip_based['entry_date'] = trip_based['entry_date'].values.astype(np.int64) // 10 ** 9
    trip_based['entry_date'] = trip_based['entry_date'].astype('int')
    trip_based['exit_date'] = trip_based['entry_date']
    trip_based_final = trip_based.loc[:,['user_id','entry_date','entry_time','entry_station_id','exit_time','exit_date','exit_station_id']]

    #a = 1
    return trip_based_final


training_all = []
testing_all = []
all_data = []

#individual_ID_list_test = [958765943]

for Card_ID in individual_ID_list_test:
    file_name_train = data_path + 'samples/sample_' + str(Card_ID) + '_201407_201408_all.csv'
    data = pd.read_csv(file_name_train)
    # data = data.loc[data['if_last'] == 0,
    #        :]  # drop the last one, it will distract the training (because it is manually added)
    test_proportion = 0.2
    data['duration_hour'] = round(data['duration'] / 3600).astype('int')
    depend_var = ['duration']  # regression
    # =========================    #data_preprocessing
    test_last = False
    scaler, scaler_y, data, data_train, data_test, Ut_list = process_data(data, test_proportion, Card_ID, test_last,
                                                                          depend_var)
    # used_col = ['seq_ID','act_ID','location','location_o','date','Next_tapin_station','duration']
    # data_test_no_first = data_test.loc[data_test['if_first']!=1]
    # print(len(data_test_no_first))
    trip_based_final1 = retrieve_trip_based_data(data_train, Card_ID)
    trip_based_final2 = retrieve_trip_based_data(data_test, Card_ID)
    # print(len(trip_based_final2))
    training_all.append(trip_based_final1)
    testing_all.append(trip_based_final2)
    all_data.append(trip_based_final1)
    all_data.append(trip_based_final2)

training_all_df = pd.concat(training_all,sort=False)
testing_all_df = pd.concat(testing_all,sort=False)
all_data_df = pd.concat(all_data,sort=False)


training_all_df = training_all_df.sort_values(['user_id','entry_date','entry_time'])
testing_all_df = testing_all_df.sort_values(['user_id','entry_date','entry_time'])
all_data_df = all_data_df.sort_values(['user_id','entry_date','entry_time'])

training_all_df.to_csv('data_for_zhan/data/training_all_samples_hk.csv',index=False)
testing_all_df.to_csv('data_for_zhan/data/testing_all_samples_hk.csv',index=False)

with open('data_for_zhan/data/USED_TOD_LIST.pickle', 'wb') as fp:
    pickle.dump(USED_TOD_LIST, fp)

all_data_df.to_csv('data_for_zhan/all_samples.csv',index=False)