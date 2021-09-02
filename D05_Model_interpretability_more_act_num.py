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
import copy


from matplotlib.ticker import FormatStrFormatter


from sklearn.metrics import r2_score

data_path = '../data/'
with open (data_path + 'individual_ID_list_test', 'rb') as fp:
    individual_ID_list_test = pickle.load(fp)


def data_process_continuous(data):
    error_first_temp = (data['Predict1'].loc[data['activity_index']==0] - data['Ground_truth'].loc[data['activity_index']==0])/3600
    Accuracy_first_temp = sum(np.array(data['Correct'].loc[data['activity_index']==0]))/data['Correct'].loc[data['activity_index']==0].count()
    data_temp = data.loc[data['activity_index']!=0]
    # data_temp = data
    error_middle_temp = (data_temp['Predict1'] - data_temp['Ground_truth'])/3600
    Accuracy_temp = sum(np.array(data_temp['Correct']))/data_temp['Correct'].count()
    accuracy_all = sum(np.array(data['Correct']))/data['Correct'].count()
    return error_first_temp, Accuracy_first_temp, error_middle_temp, Accuracy_temp,accuracy_all


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

def r_sq_for_two_parts(data,y_mean):
    data['RES'] = (data['Ground_truth_duration'] -  data['Predict_duration'])**2
    data['TOT'] = (data['Ground_truth_duration'] -  y_mean)**2
    R_sq = 1 - sum(data['RES'])/sum(data['TOT'])
    return R_sq


def data_process_continuous_R_sq(data):
    _, _, _, R_sq_all = calculate_error(data)
    data_first = data.loc[data['activity_index']==0].copy()
    data_middle = data.loc[data['activity_index']!=0].copy()
    mean_y = np.mean(data['Ground_truth_duration'])
    R_sq_first = r_sq_for_two_parts(data_first, mean_y)
    R_sq_middle = r_sq_for_two_parts(data_middle, mean_y)

    return R_sq_first, R_sq_middle, R_sq_all

def data_process_continuous_RMSE(data):
    RMSE_all, _, _, _ = calculate_error(data)
    data_first = data.loc[data['activity_index']==0].copy()
    data_middle = data.loc[data['activity_index']!=0].copy()
    RMSE_first, _, _, R_sq_first = calculate_error(data_first)
    RMSE_middle, _, _, R_sq_middle = calculate_error(data_middle)
    return RMSE_first, RMSE_middle, RMSE_all


def data_process_continuous_MAPE(data):
    _, MAPE_all, _, _ = calculate_error(data)
    data_first = data.loc[data['activity_index']==0].copy()
    data_middle = data.loc[data['activity_index']!=0].copy()
    _, MAPE_first, _, R_sq_first = calculate_error(data_first)
    _, MAPE_middle, _, R_sq_middle = calculate_error(data_middle)
    return MAPE_first, MAPE_middle, MAPE_all


def data_process_discrete(data):
    error_first_temp = (data['Predict1'].loc[data['activity_index']==0] - data['Ground_truth'].loc[data['activity_index']==0])
    Accuracy_first_temp = sum(np.array(data['Correct'].loc[data['activity_index']==0]))/data['Correct'].loc[data['activity_index']==0].count()
    data_temp = data.loc[data['activity_index']!=0]
    # data_temp = data
    error_middle_temp = (data_temp['Predict1'] - data_temp['Ground_truth'])
    Accuracy_temp = sum(np.array(data_temp['Correct']))/data_temp['Correct'].count()
    accuracy_all = sum(np.array(data['Correct'])) / data['Correct'].count()
    return error_first_temp, Accuracy_first_temp, error_middle_temp, Accuracy_temp, accuracy_all




def calculate_accuracy(result_df, task = None):
    if task == 'loc':
        RMSE = -1
        MAPE = -1
        MAE = -1
        R_sq = -1
    else:
        # correct error data
        result_df.loc[result_df['Predict_duration'] > 86400, 'Predict_duration'] = 86400
        result_df.loc[result_df['Predict_duration'] <= 0, 'Predict_duration'] = 1
        result_df['error_sq'] = (result_df['Predict_duration'] -  result_df['Ground_truth_duration'])**2
        result_df['error_abs'] = np.abs(result_df['Predict_duration'] -  result_df['Ground_truth_duration'])
        RMSE = np.sqrt(np.mean(result_df['error_sq']))
        MAPE = np.mean(result_df['error_abs']/result_df['Ground_truth_duration'])
        MAE = np.mean(result_df['error_abs'])
        R_sq = r2_score(result_df['Ground_truth_duration'], result_df['Predict_duration'])


    N_first = result_df['Correct'].loc[result_df['activity_index']==0].count()
    Accuracy_first = result_df['Correct'].loc[(result_df['Correct']==1)&
               (result_df['activity_index']==0)].count()/N_first

    N_middle = result_df['Correct'].loc[result_df['activity_index']!=0].count()
    Accuracy_middle = result_df['Correct'].loc[(result_df['Correct']==1)&
               (result_df['activity_index']!=0)].count()/N_middle

    N_all = result_df['Correct'].count()
    Accuracy_all = result_df['Correct'].loc[result_df['Correct']==1].count()/N_all


    return Accuracy_first, Accuracy_middle, Accuracy_all, N_first, N_middle, N_all, RMSE, MAPE, MAE, R_sq




def get_accuracy_and_num_act(individual_ID_list, output_fig, duration_error):
    error_list=[]
    total=0
    error_middle = pd.DataFrame({'middle':[]})
    error_first = pd.DataFrame({'first':[]})
    error_middle_base = pd.DataFrame({'middle':[]})
    error_first_base = pd.DataFrame({'first':[]})
    Accuracy = {'Card_ID':[], 'Middle':[],'first':[],'all':[],'Total_act':[]}
    Accuracy_base = {'Card_ID':[], 'Middle':[],'first':[],'all':[],'Total_act':[]}
    Accuracy_LSTM = {'Card_ID': [], 'Middle': [], 'first': [], 'all': [],'Total_act':[]}
    # data
    Card_ID_used = []
    # individual_ID_list = individual_ID_list[0:80]
    #############IOHMM
    for Card_ID in individual_ID_list:

        file_name =  data_path + 'results/result_con_dur+loc_' + str(Card_ID) + 'test' + '.csv'
        if os.path.exists(file_name) == False:
            print(Card_ID,'does not exist for IOHMM')
            continue
        else:
            Card_ID_used.append(Card_ID)
        data_test = pd.read_csv(file_name)

        file_name =  data_path + 'results/result_con_dur+loc_' + str(Card_ID) + 'train' + '.csv'
        data_train = pd.read_csv(file_name)
        data = pd.concat([data_train, data_test])


        if output_fig == 'duration':
            if duration_error == 'RMSE':
                R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_RMSE(data)
            elif duration_error == 'MAPE':
                R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_MAPE(data)
            else:
                R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_R_sq(data)

            Accuracy['first'].append(R_sq_first)
            Accuracy['Middle'].append(R_sq_middle)
            Accuracy['all'].append(R_sq_all)
            Accuracy['Card_ID'].append(Card_ID)
            Accuracy['Total_act'].append(data['total_activity'].iloc[0])
        else:
            error_first_temp, Accuracy_first_temp, error_middle_temp, Accuracy_temp, accuracy_all = data_process_discrete(data)
            #print (error_first_temp)
            error_first = pd.concat([error_first, error_first_temp], axis = 0)
            error_middle = pd.concat([error_middle, error_middle_temp], axis = 0)
            Accuracy['first'].append(Accuracy_first_temp)
            Accuracy['Middle'].append(Accuracy_temp)
            Accuracy['all'].append(accuracy_all)
            Accuracy['Card_ID'].append(Card_ID)
            Accuracy['Total_act'].append(data['total_activity'].iloc[0])
        # data
    # ############## LSTM
    # Card_ID_used_for_base = list(set(Card_ID_used))
    # for Card_ID in Card_ID_used_for_base:
    #     if output_fig == 'duration':
    #         # file_name = data_path + 'results/result_LSTM' + str(Card_ID) + 'test' + '.csv'
    #         file_name = data_path + 'results/result_LSTM_con_dur' + str(Card_ID) + 'test' + '.csv'
    #     else:
    #         file_name = data_path + 'results/result_Location_LSTM' + str(Card_ID) + 'test' + '.csv'
    #
    #     if os.path.exists(file_name) == False:
    #         print(Card_ID,'does not exist for LSTM')
    #         continue
    #     data = pd.read_csv(file_name)
    #     if output_fig == 'duration':
    #         if duration_error == 'RMSE':
    #             R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_RMSE(data)
    #         elif duration_error == 'MAPE':
    #             R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_MAPE(data)
    #         else:
    #             R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_R_sq(data)
    #         Accuracy_LSTM['first'].append(R_sq_first)
    #         Accuracy_LSTM['Middle'].append(R_sq_middle)
    #         Accuracy_LSTM['all'].append(R_sq_all)
    #         Accuracy_LSTM['Card_ID'].append(Card_ID)
    #
    #     else:
    #         error_first_temp, Accuracy_first_temp, error_middle_temp, Accuracy_temp, accuracy_all = data_process_discrete(data)
    #         #print (error_first_temp)
    #         error_first = pd.concat([error_first, error_first_temp], axis = 0)
    #         error_middle = pd.concat([error_middle, error_middle_temp], axis = 0)
    #         Accuracy_LSTM['first'].append(Accuracy_first_temp)
    #         Accuracy_LSTM['Middle'].append(Accuracy_temp)
    #         Accuracy_LSTM['all'].append(accuracy_all)
    #         Accuracy_LSTM['Card_ID'].append(Card_ID)
    #
    #
    # ############## MC
    #
    # for Card_ID in Card_ID_used_for_base:
    #     if output_fig == 'duration':
    #         # file_name = data_path + 'results/result_MC' + str(Card_ID) + '.csv'
    #         file_name = data_path + 'results/result_LR' + str(Card_ID) + 'test.csv'
    #     else:
    #         file_name = data_path + 'results/result_Location_MC' + str(Card_ID) + '.csv'
    #     # if os.path.exists(file_name) == False:
    #     #     print(Card_ID, 'does not exist for Base')
    #     #     continue
    #     data = pd.read_csv(file_name)
    #     if output_fig == 'duration':
    #         if duration_error == 'RMSE':
    #             R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_RMSE(data)
    #         elif duration_error == 'MAPE':
    #             R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_MAPE(data)
    #         else:
    #             R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_R_sq(data)
    #         Accuracy_base['first'].append(R_sq_first)
    #         Accuracy_base['Middle'].append(R_sq_middle)
    #         Accuracy_base['all'].append(R_sq_all)
    #         Accuracy_base['Card_ID'].append(Card_ID)
    #
    #     else:
    #         error_first_temp, Accuracy_first_temp, error_middle_temp, Accuracy_temp, accuracy_all = data_process_discrete(data)
    #         # print (error_first_temp)
    #         error_first_base = pd.concat([error_first_base, error_first_temp], axis=0)
    #         error_middle_base = pd.concat([error_middle_base, error_middle_temp], axis=0)
    #         Accuracy_base['first'].append(Accuracy_first_temp)
    #         Accuracy_base['Middle'].append(Accuracy_temp)
    #         Accuracy_base['Card_ID'].append(Card_ID)
    #         Accuracy_base['all'].append(accuracy_all)
    #
    #
    # # ====================
    #
    #
    # ##############
    Accuracy_IOHMM = pd.DataFrame(Accuracy)
    # Accuracy_base = pd.DataFrame(Accuracy_base)
    # Accuracy_LSTM = pd.DataFrame(Accuracy_LSTM)

    return Accuracy_IOHMM #, Accuracy_base, Accuracy_LSTM


def print_acc(Card_ID, test_name = ''):

    result_df_IOHMM_location = pd.read_csv(data_path + 'results/result_con_dur+loc_' + str(Card_ID) + 'train.csv')
    Accuracy_IOHMM_loc_train_first, Accuracy_IOHMM_loc_train_middle, Accuracy_IOHMM_loc_train, _, _, _, _, _, _, _ = calculate_accuracy(result_df_IOHMM_location, task='loc')
    result_df_MC_location = pd.read_csv(data_path + 'results/result_Location_LSTM' + str(Card_ID) + 'train.csv')
    Accuracy_LSTM_loc_train_first, Accuracy_LSTM_loc_train_middle, Accuracy_LSTM_loc_train, _, _, _, _, _, _, _ = calculate_accuracy(result_df_MC_location, task='loc')

    result_df_IOHMM_location = pd.read_csv(data_path + 'results/result_con_dur+loc_' + str(Card_ID) + 'test.csv')
    Accuracy_IOHMM_loc_test_first, Accuracy_IOHMM_loc_test_middle, Accuracy_IOHMM_loc_test, _, _, _, _, _, _, _ = calculate_accuracy(result_df_IOHMM_location, task='loc')
    result_df_LSTM_location = pd.read_csv(data_path + 'results/result_Location_LSTM' + str(Card_ID) + 'test.csv')
    Accuracy_LSTM_loc_test_first, Accuracy_LSTM_loc_test_middle, Accuracy_LSTM_loc_test, _, _, _, _, _, _, _ = calculate_accuracy(result_df_LSTM_location, task='loc')
    result_df_MC_location = pd.read_csv(data_path + 'results/result_Location_MC' + str(Card_ID) + '.csv')
    Accuracy_MC_loc_test_first, Accuracy_MC_loc_test_middle, Accuracy_MC_loc_test, _, _, _, _, _, _, _ = calculate_accuracy(result_df_MC_location, task='loc')

    result_df_IOHMM_duration = pd.read_csv(data_path + 'results/result_con_dur+loc_' + str(Card_ID) + 'train.csv')
    R_sq_first_IOHMM_train, R_sq_middle_IOHMM_train, R_sq_all_IOHMM_train = data_process_continuous_R_sq(result_df_IOHMM_duration)
    result_df_LR_duration = pd.read_csv(data_path + 'results/result_LR' + str(Card_ID) + 'train.csv')
    R_sq_first_LR_train, R_sq_middle_LR_train, R_sq_all_LR_train = data_process_continuous_R_sq(result_df_LR_duration)
    result_df_LSTM_duration = pd.read_csv(data_path + 'results/result_LSTM_con_dur' + str(Card_ID) + 'train.csv')
    R_sq_first_LSTM_train, R_sq_middle_LSTM_train, R_sq_all_LSTM_train = data_process_continuous_R_sq(result_df_LSTM_duration)


    result_df_IOHMM_duration = pd.read_csv(data_path + 'results/result_con_dur+loc_' + str(Card_ID) + 'test.csv')
    R_sq_first_IOHMM_test, R_sq_middle_IOHMM_test, R_sq_all_IOHMM_test = data_process_continuous_R_sq(result_df_IOHMM_duration)
    result_df_LR_duration = pd.read_csv(data_path + 'results/result_LR' + str(Card_ID) + 'test.csv')
    R_sq_first_LR_test, R_sq_middle_LR_test, R_sq_all_LR_test = data_process_continuous_R_sq(result_df_LR_duration)
    result_df_LSTM_duration = pd.read_csv(data_path + 'results/result_LSTM_con_dur' + str(Card_ID) + 'test.csv')
    R_sq_first_LSTM_test, R_sq_middle_LSTM_test, R_sq_all_LSTM_test = data_process_continuous_R_sq(result_df_LSTM_duration)


    print('=======Location===========')
    data_save_loc = {}
    data_save_loc['All Training Accuracy'] = [Accuracy_IOHMM_loc_train, 'Not available',Accuracy_LSTM_loc_train]
    data_save_loc['All Testing Accuracy'] = [Accuracy_IOHMM_loc_test, Accuracy_MC_loc_test,Accuracy_LSTM_loc_test]
    data_save_loc['First Training Accuracy'] = [Accuracy_IOHMM_loc_train_first, 'Not available',Accuracy_LSTM_loc_train_first]
    data_save_loc['First Testing Accuracy'] = [Accuracy_IOHMM_loc_test_first, Accuracy_MC_loc_test_first,Accuracy_LSTM_loc_test_first]
    data_save_loc['Middle Training Accuracy'] = [Accuracy_IOHMM_loc_train_middle, 'Not available',Accuracy_LSTM_loc_train_middle]
    data_save_loc['Middle Testing Accuracy'] = [Accuracy_IOHMM_loc_test_middle, Accuracy_MC_loc_test_middle,Accuracy_LSTM_loc_test_middle]

    print(Card_ID, 'All Training Accuracy:', 'IOHMM:',Accuracy_IOHMM_loc_train,'MC','Not available','LSTM:',Accuracy_LSTM_loc_train)

    print(Card_ID, 'All Testing Accuracy:', 'IOHMM:', Accuracy_IOHMM_loc_test, 'MC', Accuracy_MC_loc_test, 'LSTM:',Accuracy_LSTM_loc_test)
    print(Card_ID, 'First Training Accuracy:', 'IOHMM:', Accuracy_IOHMM_loc_train_first, 'MC', 'Not available', 'LSTM:',
          Accuracy_LSTM_loc_train_first)
    print(Card_ID, 'First Testing Accuracy:', 'IOHMM:', Accuracy_IOHMM_loc_test_first, 'MC', Accuracy_MC_loc_test_first, 'LSTM:',Accuracy_LSTM_loc_test_first)
    print(Card_ID, 'Middle Training Accuracy:', 'IOHMM:', Accuracy_IOHMM_loc_train_middle, 'MC', 'Not available', 'LSTM:',
          Accuracy_LSTM_loc_train_middle)
    print(Card_ID, 'Middle Testing Accuracy:', 'IOHMM:', Accuracy_IOHMM_loc_test_middle, 'MC', Accuracy_MC_loc_test_middle, 'LSTM:',Accuracy_LSTM_loc_test_middle)

    print('=======Duration===========')
    data_save_dur = {}
    data_save_dur['All Training R2'] = [R_sq_all_IOHMM_train,R_sq_all_LR_train,R_sq_all_LSTM_train]
    data_save_dur['All Testing R2'] = [R_sq_all_IOHMM_test, R_sq_all_LR_test,R_sq_all_LSTM_test]
    data_save_dur['First Training R2'] = [R_sq_first_IOHMM_train, R_sq_first_LR_train,R_sq_first_LSTM_train]
    data_save_dur['First Testing R2'] = [R_sq_first_IOHMM_test, R_sq_first_LR_test,R_sq_first_LSTM_test]
    data_save_dur['Middle Training R2'] = [R_sq_middle_IOHMM_train, R_sq_middle_LR_train,R_sq_middle_LSTM_train]
    data_save_dur['Middle Testing R2'] = [R_sq_middle_IOHMM_test, R_sq_middle_LR_test,R_sq_middle_LSTM_test]


    print(Card_ID, 'All Training R2:', 'IOHMM:',R_sq_all_IOHMM_train,'LR',R_sq_all_LR_train,'LSTM:',R_sq_all_LSTM_train)
    print(Card_ID, 'All Testing R2:', 'IOHMM:', R_sq_all_IOHMM_test, 'LR', R_sq_all_LR_test, 'LSTM:',R_sq_all_LSTM_test)
    print(Card_ID, 'First Training R2:', 'IOHMM:', R_sq_first_IOHMM_train,'LR',R_sq_first_LR_train, 'LSTM:',
          R_sq_first_LSTM_train)
    print(Card_ID, 'First Testing R2:', 'IOHMM:', R_sq_first_IOHMM_test, 'LR', R_sq_first_LR_test, 'LSTM:',R_sq_first_LSTM_test)
    print(Card_ID, 'Middle Training R2:', 'IOHMM:', R_sq_middle_IOHMM_train, 'LR', R_sq_middle_LR_train, 'LSTM:',
          R_sq_middle_LSTM_train)
    print(Card_ID, 'Middle Testing R2:', 'IOHMM:', R_sq_middle_IOHMM_test, 'LR', R_sq_middle_LR_test, 'LSTM:',R_sq_middle_LSTM_test)

    print('=================')

    data_save_loc_df = pd.DataFrame.from_dict(data_save_loc,orient = 'index',columns=['IOHMM','MC','LSTM'])

    data_save_dur_df = pd.DataFrame.from_dict(data_save_dur,orient = 'index',columns=['IOHMM','LRs','LSTM'])

    data_save_loc_df.to_csv('Test_results_loc_'+ str(Card_ID)+ '_' + test_name + '.csv',index=True)
    data_save_dur_df.to_csv('Test_results_dur_'+ str(Card_ID)+ '_' + test_name + '.csv',index=True)



def get_Ut_list(data):

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
    return Ut_list





def generate_activity_seq(Card_ID, Num_sampling, num_states):
    save_path = 'output/IOHMM/'
    with open(save_path + 'Trained_IOHMM_' + str(Card_ID) +'.pickle', 'rb') as fp:
        save_dict = pickle.load(fp)
        MODEL = save_dict['model']
        data = save_dict['data']
        scaler_x = save_dict['scaler_x']
        scaler_y = save_dict['scaler_y']

    # get start time
    data['start_time'] = data['date_time'].apply(lambda x: int(x.split(' ')[1].split(':')[0]) + int(x.split(' ')[1].split(':')[1])/60 + int(x.split(' ')[1].split(':')[2])/3600) + 4 # start from 4:00
    data['start_location'] = data['location']
    z_var = ['start_time','start_location']

    Dt_h_2 = np.array(sorted(data.loc[:, 'Next_tapin_station'].unique()))

    all_seq = MODEL.dfs_logStates
    # seq_id = 0
    Ut_list = get_Ut_list(all_seq[0][0])
    Ut_list_1 = Ut_list
    Ut_list_2 = Ut_list

    data_set_generated = {'ID':[],'Replicate_id':[],'seq_ID':[],'act_ID':[],'State':[],'Duration':[],
                          'Location':[]}

    Ut_list_list = []
    ID = 0
    for i in range(Num_sampling):
        np.random.seed((i + 100)*3)
        for idx, row in data.iterrows():
            if row['act_ID']==0:
                # Caculate_initial_state:
                X_ini = np.array([row[Ut_list]])
                log_prob_initial = MODEL.model_initial.predict_log_proba(X_ini).reshape(num_states, )
                state = np.random.choice(num_states, size = 1, p=np.exp(log_prob_initial))[0]
                last_state = copy.deepcopy(state)
            else:
                X_tr = np.array([row[Ut_list]])
                log_prob_transition = MODEL.model_transition[last_state].predict_log_proba(X_tr)[0]
                state = np.random.choice(num_states, size=1, p=np.exp(log_prob_transition))[0]
                last_state = copy.deepcopy(state)

            #######duration
            output_id = 0
            X_emi_1 = np.array([row[Ut_list_1]])
            duration_mean = MODEL.model_emissions[state][output_id].predict(X_emi_1)[0][0]
            duration_var = MODEL.model_emissions[state][output_id].get_dispersion(Y_len = 1)[0][0] #
            duration = np.random.normal(duration_mean, np.sqrt(duration_var), 1)[0]
            #########location
            output_id = 1
            X_emi_2 = np.array([row[Ut_list_2]])
            Ut_input_2 = np.repeat(X_emi_2, len(Dt_h_2), axis=0)
            log_prob_location = MODEL.model_emissions[state][output_id].loglike_per_sample(Ut_input_2, Dt_h_2)
            location_idx = np.random.choice(len(Dt_h_2), size=1, p=np.exp(log_prob_location))[0]
            location_station_id = Dt_h_2[location_idx]
            #########
            Ut_list_list.append(row[Ut_list + z_var])
            #########
            ID+= 1
            data_set_generated['ID'].append(ID)
            data_set_generated['Replicate_id'].append(i+1)
            data_set_generated['seq_ID'].append(row['seq_ID'])
            data_set_generated['act_ID'].append(row['act_ID'])
            data_set_generated['State'].append(state)
            data_set_generated['Duration'].append(duration)
            data_set_generated['Location'].append(location_station_id)

    data_set_df = pd.DataFrame(data_set_generated)
    data_set_df['Duration'] = scaler_y.inverse_transform(data_set_df['Duration'].values.reshape(-1, 1))
    data_set_df['Duration'] /= 3600 # to hours
    Ut_df = pd.DataFrame(np.array(Ut_list_list), columns= Ut_list + z_var)
    Ut_df[Ut_list] = scaler_x.inverse_transform(Ut_df[Ut_list].values)
    assert Ut_df.shape[0] == data_set_df.shape[0]
    # print(Ut_df.shape)
    # print(data_set_df.shape)
    final_data = pd.concat([data_set_df, Ut_df], axis=1)
    return final_data


def plot_distribution(Card_ID, data, station_name, num_state, PLOT_GRAPH,bw, save_fig):
    colors = sns.color_palette('Paired')
    font_size = 16
    data = data.merge(station_name, left_on = ['Location'], right_on = ['CODE'], how='left')
    check_an = data.loc[data['CODE'].isna()]
    if len(check_an) > 0:
        print("missing data")
        print('station_id', pd.unique(check_an['Location']))
        exit()
    Num_station_showed = 10


    for i in range(num_state):
        if PLOT_GRAPH['DURATION']:
            # duration
            points = list(data.loc[data['State']==i, 'Duration'])
            sns.set(font_scale=1.5)
            sns.set_style("white", {"legend.frameon": True})
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.kdeplot(points, ax=ax, shade=True, color=colors[0],bw=bw)
            plt.xlim(0, 24)
            # new_ticks2 = list(np.arange(0,0.18,0.03))
            # plt.ylim(0, 0.18)
            #new_ticks.append(30)
            # plt.yticks(new_ticks2)
            #new_ticks = list(range(0,400,60))
            #new_ticks.append(30)
            #plt.xticks(new_ticks)
            plt.xlabel('Activity duration', fontsize=font_size)
            plt.ylabel('Density', fontsize=font_size)
            plt.tight_layout()
            if save_fig == 0:
                plt.show()
            else:
                plt.savefig('img/duration_distribution_act_' + str(i) + '_' + Card_ID + '.png', dpi=200)
        ##########Location
        if PLOT_GRAPH['LOCATION']:
            sns.set(font_scale=1.5)
            sns.set_style("white", {"legend.frameon": True})
            fig, ax = plt.subplots(figsize=(6, 5))
            points = data.loc[data['State'] == i, ['STATION','ID']]
            points_group = points.groupby(['STATION']).count()[['ID']].reset_index()
            points_group = points_group.sort_values(['ID'],ascending=False)
            points_group = points_group.head(Num_station_showed)
            points_group['Prob'] = points_group['ID']/sum(points_group['ID'])
            p = list(points_group['Prob'])
            w = 1
            plt.bar(points_group['STATION'], p, width=w, align='edge', color=colors[1], edgecolor='w', alpha=0.8)
            # plt.hist(S, bins=range(0, 151, 10), normed=True, color=colors[0], edgecolor='w', alpha=0.8)
            # plt.xlim(200, 800)
            # ax1.set_yticks([0.0, 0.005, 0.010, 0.015, 0.020])
            plt.xticks([i + 0.5 for i in range(0, Num_station_showed)], points_group['STATION'], rotation = 0, fontsize = font_size - 2)
            plt.xlabel('Activity end location', fontsize=font_size)
            plt.ylabel('Density', fontsize=font_size)
            plt.tight_layout()
            if save_fig == 0:
                plt.show()
            else:
                plt.savefig('img/location_distribution_act_' + str(i) + '_' + str(Card_ID) + '.png', dpi=200)

        if PLOT_GRAPH['START_TIME']:
            sns.set(font_scale=1.5)
            sns.set_style("white", {"legend.frameon": True})
            fig, ax = plt.subplots(figsize=(6, 5))
            points = data.loc[data['State'] == i,'start_time']
            # a=1
            # # start_time_col = [col for col in points.columns if 'hour' in col]
            # # num_count = []
            # # hour_id_list = []
            # for hour in range(0,24):
            #     hour_id = 'hour_'+str(hour)
            #     if hour_id in start_time_col:
            #         hour_id_list.append(int(hour_id.split('_')[1]))
            #         num_count.append(sum(points[hour_id]))
            #     else:
            #         hour_id_list.append(hour)
            #         num_count.append(0)
            sns.kdeplot(points, ax=ax, shade=True, color=colors[2], label='Activity start time',bw=bw)
            plt.xlim(0, 24)
            # prob = np.array(num_count)/sum(num_count)
            # w = 1
            # plt.bar(hour_id_list, prob, width=w, align='edge', color=colors[0], edgecolor='w', alpha=0.8)
            # plt.hist(S, bins=range(0, 151, 10), normed=True, color=colors[0], edgecolor='w', alpha=0.8)
            # plt.xlim(200, 800)
            # ax1.set_yticks([0.0, 0.005, 0.010, 0.015, 0.020])
            plt.xticks([i for i in range(0,26,2)], range(1,26,2), rotation = 0, fontsize = font_size - 2)
            plt.xlabel('Activity start time', fontsize=font_size)
            plt.ylabel('Density', fontsize=font_size)
            plt.tight_layout()
            if save_fig == 0:
                plt.show()
            else:
                plt.savefig('img/start_time_distribution_act_' + str(i) + '_' + str(Card_ID) + '.png', dpi=200)


        if PLOT_GRAPH['DAY_OF_WEEK']:
            sns.set(font_scale=1.5)
            sns.set_style("white", {"legend.frameon": True})
            fig, ax = plt.subplots(figsize=(6, 5))
            points = data.loc[data['State'] == i]
            num_count = []
            day_of_week_list = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            for dayofweek in day_of_week_list:
                num_count.append(sum(points[dayofweek]))


            prob = np.array(num_count)/sum(num_count)
            w = 1
            day_of_week_list_abbre = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
            plt.bar(day_of_week_list, prob, width=w, align='edge', color=colors[3], edgecolor='w', alpha=0.8)
            # plt.hist(S, bins=range(0, 151, 10), normed=True, color=colors[0], edgecolor='w', alpha=0.8)
            # plt.xlim(200, 800)
            # ax1.set_yticks([0.0, 0.005, 0.010, 0.015, 0.020])
            plt.xticks([i + 0.5 for i in range(len(day_of_week_list))], day_of_week_list_abbre, rotation = 0, fontsize = font_size - 2)
            plt.xlabel('Activity day of week', fontsize=font_size)
            plt.ylabel('Density', fontsize=font_size)
            plt.tight_layout()
            if save_fig == 0:
                plt.show()
            else:
                plt.savefig('img/day_of_week_distribution_act_' + str(i) + '_' + str(Card_ID) + '.png', dpi=200)


        if PLOT_GRAPH['START_LOCATION']:
            sns.set(font_scale=1.5)
            sns.set_style("white", {"legend.frameon": True})
            fig, ax = plt.subplots(figsize=(6, 5))
            points = data.loc[data['State'] == i]
            start_location_col = pd.unique(points['start_location'])
            num_count = []
            name_list = []
            for loc_id in start_location_col:
                num_count.append(len(points.loc[points['start_location']== loc_id]))
                if loc_id == -1:
                    loc_name = 'Null'
                else:
                    loc_name = station_name.loc[station_name['CODE']==loc_id]['STATION'].iloc[0]
                name_list.append(loc_name)
            prob = np.array(num_count)/sum(num_count)

            sort_all = sorted(zip(prob, name_list), key=lambda pair: pair[0],reverse=True)
            prob = [x for x, _ in sort_all]
            name_list = [x for _, x in sort_all]
            name_list = name_list[0:Num_station_showed]
            prob = prob[0:Num_station_showed]
            w = 1
            plt.bar(name_list, prob, width=w, align='edge', color=colors[4], edgecolor='w', alpha=0.8)
            # plt.hist(S, bins=range(0, 151, 10), normed=True, color=colors[0], edgecolor='w', alpha=0.8)
            # plt.xlim(200, 800)
            # ax1.set_yticks([0.0, 0.005, 0.010, 0.015, 0.020])
            plt.xticks([i + 0.5 for i in range(len(name_list))], name_list, rotation = 0, fontsize = font_size - 2)
            plt.xlabel('Activity start location', fontsize=font_size)
            plt.ylabel('Density', fontsize=font_size)
            plt.tight_layout()
            if save_fig == 0:
                plt.show()
            else:
                plt.savefig('img/start_location_distribution_act_' + str(i) + '_' + str(Card_ID) + '.png', dpi=200)



def plot_distribution_subplot(Card_ID, data, station_name, num_state, PLOT_GRAPH, bw, save_fig, act_label = None):
    colors = sns.color_palette('Paired')
    font_size = 16

    data = data.merge(station_name, left_on = ['Location'], right_on = ['CODE'], how='left')
    check_an = data.loc[data['CODE'].isna()]
    if len(check_an) > 0:
        print("missing data")
        print('station_id', pd.unique(check_an['Location']))
        exit()
    Num_station_showed = 7

    sns.set(font_scale=1.5)
    sns.set_style("white", {"legend.frameon": True})
    # fig, axs = plt.subplots(5, num_state, figsize=(15, 18))
    fig, axs = plt.subplots(4, num_state, figsize=(18, 10))


    if act_label is not None and num_state == 3:
        act_label_inv = {v: k for k, v in act_label.items()}
        state_id_list = [act_label_inv['Home'],act_label_inv['Work'],act_label_inv['Other']]
    else:
        state_id_list = range(num_state)

    col_id = -1
    for i in state_id_list:
        col_id += 1
        if PLOT_GRAPH['START_TIME']:

            points = data.loc[data['State'] == i,'start_time']
            # a=1
            # # start_time_col = [col for col in points.columns if 'hour' in col]
            # # num_count = []
            # # hour_id_list = []
            # for hour in range(0,24):
            #     hour_id = 'hour_'+str(hour)
            #     if hour_id in start_time_col:
            #         hour_id_list.append(int(hour_id.split('_')[1]))
            #         num_count.append(sum(points[hour_id]))
            #     else:
            #         hour_id_list.append(hour)
            #         num_count.append(0)

            ax = axs[0, col_id]
            sns.kdeplot(list(points), ax=ax, shade=True, color=colors[2],bw=bw['START_TIME'])
            ax.set_xlim(0, 24)
            # prob = np.array(num_count)/sum(num_count)
            # w = 1
            # plt.bar(hour_id_list, prob, width=w, align='edge', color=colors[0], edgecolor='w', alpha=0.8)
            # plt.hist(S, bins=range(0, 151, 10), normed=True, color=colors[0], edgecolor='w', alpha=0.8)
            # plt.xlim(200, 800)
            # ax1.set_yticks([0.0, 0.005, 0.010, 0.015, 0.020])
            ax.set_xticks([i for i in range(0,28,4)])
            ax.set_xticklabels(range(1,28,4), fontsize = font_size - 2)
            ax.set_xlabel('Activity start time', fontsize=font_size)
            ax.set_ylabel('Density', fontsize=font_size)

        if PLOT_GRAPH['DURATION']:
            # duration
            points = list(data.loc[data['State']==i, 'Duration'])

            ax = axs[1,col_id]
            sns.kdeplot(points, ax=ax, shade=True, color=colors[0],bw=bw['DURATION'])
            ax.set_xlim(0, 24)
            # new_ticks2 = list(np.arange(0,0.18,0.03))
            # plt.ylim(0, 0.18)
            #new_ticks.append(30)
            # plt.yticks(new_ticks2)
            #new_ticks = list(range(0,400,60))
            #new_ticks.append(30)
            #plt.xticks(new_ticks)
            ax.set_xticks([i for i in range(0,28,4)])
            ax.set_xticklabels(range(1,28,4), fontsize = font_size - 2)
            ax.set_xlabel('Activity duration', fontsize=font_size)
            ax.set_ylabel('Density', fontsize=font_size)



        if PLOT_GRAPH['START_LOCATION']:


            ax = axs[2,col_id]

            points = data.loc[data['State'] == i]
            start_location_col = pd.unique(points['start_location'])
            num_count = []
            name_list = []
            for loc_id in start_location_col:
                num_count.append(len(points.loc[points['start_location']== loc_id]))
                if loc_id == -1:
                    loc_name = 'Null'
                else:
                    loc_name = station_name.loc[station_name['CODE']==loc_id]['STATION'].iloc[0]
                name_list.append(loc_name)
            prob = np.array(num_count)/sum(num_count)

            sort_all = sorted(zip(prob, name_list), key=lambda pair: pair[0],reverse=True)
            prob = [x for x, _ in sort_all]
            name_list = [x for _, x in sort_all]
            name_list = name_list[0:Num_station_showed]
            prob = prob[0:Num_station_showed]
            w = 1
            ax.bar(name_list, prob, width=w, align='edge', color=colors[4], edgecolor='w', alpha=0.8)
            # plt.hist(S, bins=range(0, 151, 10), normed=True, color=colors[0], edgecolor='w', alpha=0.8)
            # plt.xlim(200, 800)
            # ax1.set_yticks([0.0, 0.005, 0.010, 0.015, 0.020])
            ax.set_xticks([i + 0.5 for i in range(len(name_list))])
            ax.set_xticklabels(name_list, rotation = 45, fontsize = font_size - 5)
            ax.set_xlabel('Activity start location', fontsize=font_size)
            ax.set_ylabel('Density', fontsize=font_size)

        ##########Location
        if PLOT_GRAPH['LOCATION']:

            ax = axs[3, col_id]
            points = data.loc[data['State'] == i, ['STATION','ID']]
            points_group = points.groupby(['STATION']).count()[['ID']].reset_index()
            points_group = points_group.sort_values(['ID'],ascending=False)
            points_group = points_group.head(Num_station_showed)
            points_group['Prob'] = points_group['ID']/sum(points_group['ID'])
            p = list(points_group['Prob'])
            w = 1
            ax.bar(points_group['STATION'], p, width=w, align='edge', color=colors[1], edgecolor='w', alpha=0.8)
            # plt.hist(S, bins=range(0, 151, 10), normed=True, color=colors[0], edgecolor='w', alpha=0.8)
            # plt.xlim(200, 800)
            # ax1.set_yticks([0.0, 0.005, 0.010, 0.015, 0.020])
            ax.set_xticks([i + 0.5 for i in range(0, len(p))])
            ax.set_xticklabels(points_group['STATION'], rotation = 45, fontsize = font_size - 5)
            ax.set_xlabel('Activity end location', fontsize=font_size)
            ax.set_ylabel('Density', fontsize=font_size)



        # if PLOT_GRAPH['DAY_OF_WEEK']:
        #
        #     ax = axs[4, col_id]
        #     points = data.loc[data['State'] == i]
        #     num_count = []
        #     day_of_week_list = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        #     for dayofweek in day_of_week_list:
        #         num_count.append(sum(points[dayofweek]))
        #
        #
        #     prob = np.array(num_count)/sum(num_count)
        #     w = 1
        #     day_of_week_list_abbre = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        #     ax.bar(day_of_week_list, prob, width=w, align='edge', color=colors[3], edgecolor='w', alpha=0.8)
        #     # plt.hist(S, bins=range(0, 151, 10), normed=True, color=colors[0], edgecolor='w', alpha=0.8)
        #     # plt.xlim(200, 800)
        #     # ax1.set_yticks([0.0, 0.005, 0.010, 0.015, 0.020])
        #     ax.set_xticks([k + 0.5 for k in range(len(day_of_week_list))])
        #     ax.set_xticklabels(day_of_week_list_abbre, rotation = 0, fontsize = font_size - 2)
        #     ax.set_xlabel('Activity day of week', fontsize=font_size)
        #     ax.set_ylabel('Density', fontsize=font_size)
        #     ax.set_xlim([-0.5, len(day_of_week_list)+0.5])


    # if act_label is not None:
    #     cols = ['Home','Work','Other']
    #     #cols = ['1','2','3','4','5','6','7']
    #     for ax, col in zip(axs[0], cols):
    #         ax.set_title(col, fontsize = font_size + 2, fontweight='bold')


    cols = ['1','2','3','4','5','6','7']
    for ax, col in zip(axs[0], cols):
        ax.set_title(col, fontsize = font_size + 2, fontweight='bold')


    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/patterns_distribution_'+ str(Card_ID) + '_more_act.png', dpi=200)



def output_coefficients(Card_ID, num_act, act_label=None):
    save_path = 'output/IOHMM/'
    with open(save_path + 'Trained_IOHMM_' + str(Card_ID) +'.pickle', 'rb') as fp:
        save_dict = pickle.load(fp)
        MODEL = save_dict['model']
        data = save_dict['data']
        scaler_x = save_dict['scaler_x']
        scaler_y = save_dict['scaler_y']

    interested_var = ['rain','sun','Avrg_Temp','fengli','heavy_rain','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday' ,'National_holiday'] #
    duration_task_id = 0
    var_id = [MODEL.covariates_emissions[duration_task_id].index(ele) for ele in interested_var]

    data_save = {'Card_id':[], 'Act_id':[], 'Act_label':[]}
    for _cof in interested_var:
        data_save[_cof] = []


    if act_label is not None and num_act == 3:
        act_label_inv = {v: k for k, v in act_label.items()}
        state_id_list = [act_label_inv['Home'],act_label_inv['Work'],act_label_inv['Other']]
    else:
        state_id_list = range(num_act)

    for act_id in state_id_list:
        data_save['Card_id'].append(Card_ID)
        data_save['Act_id'].append(act_id)
        if act_label is not None:
            data_save['Act_label'].append(act_label[act_id])
        else:
            data_save['Act_label'].append('NA')
        emission_model = MODEL.model_emissions[act_id][duration_task_id]
        coef = emission_model.coef[0][var_id]
        for _cof,v_id in zip(interested_var,range(len(interested_var))):
            data_save[_cof].append(coef[v_id])

    data_save_df = pd.DataFrame(data_save)
    print(data_save_df)
    data_save_df.to_csv('table/' + 'Coef_for_duration_' + str(Card_ID) + '.csv',index=False)

def plot_transition_matrix(Card_ID, final_data, num_act,act_label = None, save_fig=0):
    final_data_shift = final_data.loc[:,['ID','Replicate_id','seq_ID','act_ID','State']].shift(-1)
    final_data['next_state'] = final_data_shift['State']
    final_data['next_Replicate_id'] = final_data_shift['Replicate_id']
    final_data['next_seq_ID'] = final_data_shift['seq_ID']
    final_data['next_act_ID'] = final_data_shift['act_ID'] - 1

    final_data_used = final_data.loc[(final_data['act_ID'] == final_data['next_act_ID']) &
                                     (final_data['Replicate_id'] == final_data['next_Replicate_id']) &
                                     (final_data['seq_ID'] == final_data['next_seq_ID'])]
    prob_matrix = np.zeros((num_act,num_act))

    if act_label is not None and num_act == 3:
        act_label_inv = {v: k for k, v in act_label.items()}
        state_id_list = [act_label_inv['Home'],act_label_inv['Work'],act_label_inv['Other']]
    else:
        state_id_list = range(num_act)

    row_id = -1
    for i in state_id_list:
        row_id+=1
        act_i = final_data_used.loc[final_data_used['State'] == i]
        act_i_next_sum = act_i.groupby(['next_state'])['ID'].count()
        total_act_i = len(act_i)
        prob_i = np.array(act_i_next_sum/total_act_i)

        col_id = -1
        for j in state_id_list:
            col_id+=1
            prob_matrix[row_id,col_id] = prob_i[j]



    f, ax = plt.subplots(figsize=(5, 4))
    # sns.set_theme(style="white")
    if act_label is not None and num_act == 3:
        x_axis_labels = ['Home','Work','Other'] # labels for x-axis
        y_axis_labels = ['Home','Work','Other'] # labels for y-axis
    else:
        x_axis_labels = range(num_act)
        y_axis_labels = range(num_act)

    sns.set(font_scale=1.4)
    g = sns.heatmap(np.array(prob_matrix), annot=True, fmt='.3g', square=True, cmap='Blues', cbar_kws={"shrink": .82},annot_kws={"size": 16})
    ax.set_xticklabels(x_axis_labels, fontsize = 16)
    #
    ax.set_yticks(np.array(range(num_act))[::-1] +0.5, y_axis_labels)
    ax.set_yticklabels(y_axis_labels, fontsize=16, rotation = 0)
    ax.set_ylabel(r'$A_{t-1}$')
    ax.set_xlabel(r'$A_{t}$')
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/Transition_matrix_' + str(Card_ID) + '.png', dpi=200)




if __name__ == '__main__':

    data_path = '../data/'

    # with open (data_path + 'individual_ID_list_test', 'rb') as fp:
    #     individual_ID_list = pickle.load(fp)
    # # individual_ID_list_test_used = [958306207]

    num_ind = 1000
    with open(data_path + 'individual_ID_list_test_' + str(num_ind) + '.pickle', 'rb') as fp:
        individual_ID_list_test = pickle.load(fp)
    individual_ID_list = individual_ID_list_test[0:500]



    OBTAIN_TEST_ID = False

    LOAD_SEQ = True

    COEFF = False

    PLOT_DISTRIBUTION = True

    bw = {'START_TIME': 0.4, 'DURATION': 0.35} # para for kde

    Num_sampling = 20
    # PLOT_GRAPH = {'DURATION': True, 'LOCATION':True,'START_TIME':True, 'DAY_OF_WEEK':True,
    #               'START_LOCATION':True}

    PLOT_GRAPH = {'DURATION': True, 'LOCATION':True,'START_TIME':True,
                  'START_LOCATION':True}
    num_act = 7
    station_name = pd.read_csv('../data/station_id_name_processed.csv')

    save_fig = 1

    if OBTAIN_TEST_ID:
        duration_error = 'R_sq' # RMSE MAPE R_sq
        Mean_or_median = 'Mean' # Mean Median
        output_fig = 'duration'  # 'duration #'location'
        Accuracy_IOHMM_dur = get_accuracy_and_num_act(individual_ID_list, output_fig, duration_error = duration_error) #


        Accuracy_IOHMM_dur_used = Accuracy_IOHMM_dur.loc[Accuracy_IOHMM_dur['Total_act']==num_act]
        Accuracy_IOHMM_dur_used = Accuracy_IOHMM_dur_used.sort_values(['all'],ascending=False)
        a=1
        individual_ID_list_test_used = list(Accuracy_IOHMM_dur_used['Card_ID'].head(20))
        print(individual_ID_list_test_used)
    else:
        individual_ID_list_test_used = [989556685] # [958765943, 910506778, 997602015, 944546044, 910024208, 946346500, 949263007, 954393193, 997872944, 941974385]
        # 994326032 910024208
        # [958765943, 910506778, 997602015, 944546044, 910024208, 946346500, 949263007, 954393193, 997872944, 941974385,
        # 940212591, 997514338, 983832472, 982789143, 994322369, 970814390, 942163209, 954691863, 987865176, 942281154] # duration top 20

        #############
        #num_act = 7 good samples
        # duration top :[989556685, 992373369, 949299508, 991369626, 929949681, 916576827, 995376987, 999105278, 990287425, 972199701, 912222989, 972224825, 970988459, 988754275, 958538925, 934330353]



    if PLOT_DISTRIBUTION:
        for Card_ID in individual_ID_list_test_used:
            # print_acc(Card_ID, test_name = '2')

            if LOAD_SEQ:
                file_name = 'output/IOHMM/generated_seq_' + str(Card_ID) + '.csv'
                if os.path.exists(file_name):
                    final_data = pd.read_csv(file_name)
                else:
                    print('Sequence not generated, generate it now...')
                    final_data = generate_activity_seq(Card_ID, Num_sampling, num_act)
                    final_data.to_csv('output/IOHMM/generated_seq_' + str(Card_ID) + '.csv', index=False)
            else:
                print('start to generate sequence', 'num sampling', Num_sampling)
                final_data = generate_activity_seq(Card_ID, Num_sampling, num_act)
                print('save seq...')
                final_data.to_csv('output/IOHMM/generated_seq_' + str(Card_ID) + '.csv',index=False)
            assert num_act == np.max(final_data['State']) + 1
            print('Current Card ID', Card_ID, 'Num activity', num_act,'Total travel trips', len(final_data)/Num_sampling)
            flag = 1
            for key in PLOT_GRAPH:
                if not PLOT_GRAPH[key]:
                    flag = -1
                    break


            if Card_ID == 994326032:
                act_label = {0: 'Other',1:'Work',2:'Home'}
            elif Card_ID == 910024208:
                act_label = {0: 'Work', 1: 'Home', 2: 'Other'}
            else:
                act_label = None
            if flag == 1: # all figures are available
                plot_distribution_subplot(Card_ID, final_data, station_name, num_act, PLOT_GRAPH, bw = bw, save_fig=save_fig,act_label = act_label)
                # plot_transition_matrix(Card_ID, final_data, num_act, act_label, save_fig=save_fig)
            else:
                plot_distribution(Card_ID, final_data,station_name,num_act,PLOT_GRAPH,bw = bw, save_fig = save_fig)

    if COEFF:
        for Card_ID in individual_ID_list_test_used:
            if Card_ID == 994326032:
                act_label = {0:'Other',1:'Work',2:'Home'}
            elif Card_ID == 910024208:
                act_label = {0: 'Work', 1: 'Home', 2: 'Other'}
            else:
                act_label = None
            output_coefficients(Card_ID, num_act, act_label=act_label)
