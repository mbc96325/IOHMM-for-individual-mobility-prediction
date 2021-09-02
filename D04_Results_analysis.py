import timeit
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import pickle
import os

data_path = '../data/'
with open (data_path + 'individual_ID_list_test', 'rb') as fp:
    individual_ID_list_test = pickle.load(fp)


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





def generate_accuracy_file(individual_ID_list, output_fig, duration_error):
    error_list=[]
    total=0
    error_middle = pd.DataFrame({'middle':[]})
    error_first = pd.DataFrame({'first':[]})
    error_middle_base = pd.DataFrame({'middle':[]})
    error_first_base = pd.DataFrame({'first':[]})
    Accuracy = {'Card_ID':[], 'Middle':[],'first':[],'all':[]}
    Accuracy_base = {'Card_ID':[], 'Middle':[],'first':[],'all':[]}
    Accuracy_LSTM = {'Card_ID': [], 'Middle': [], 'first': [], 'all': []}
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
                R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_RMSE(data)
            elif duration_error == 'MAPE':
                R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_MAPE(data)
            else:
                R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_R_sq(data)

            Accuracy['first'].append(R_sq_first)
            Accuracy['Middle'].append(R_sq_middle)
            Accuracy['all'].append(R_sq_all)
            Accuracy['Card_ID'].append(Card_ID)
        else:
            error_first_temp, Accuracy_first_temp, error_middle_temp, Accuracy_temp, accuracy_all = data_process_discrete(data)
            #print (error_first_temp)
            error_first = pd.concat([error_first, error_first_temp], axis = 0)
            error_middle = pd.concat([error_middle, error_middle_temp], axis = 0)
            Accuracy['first'].append(Accuracy_first_temp)
            Accuracy['Middle'].append(Accuracy_temp)
            Accuracy['all'].append(accuracy_all)
            Accuracy['Card_ID'].append(Card_ID)
        # data
    ############## LSTM
    Card_ID_used_for_base = list(set(Card_ID_used))
    for Card_ID in Card_ID_used_for_base:
        if output_fig == 'duration':
            # file_name = data_path + 'results/result_LSTM' + str(Card_ID) + 'test' + '.csv'
            file_name = data_path + 'results/result_LSTM_con_dur' + str(Card_ID) + 'test' + '.csv'
        else:
            file_name = data_path + 'results/result_Location_LSTM' + str(Card_ID) + 'test' + '.csv'

        if os.path.exists(file_name) == False:
            print(Card_ID,'does not exist for LSTM')
            continue
        data = pd.read_csv(file_name)
        if output_fig == 'duration':
            if duration_error == 'RMSE':
                R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_RMSE(data)
            elif duration_error == 'MAPE':
                R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_MAPE(data)
            else:
                R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_R_sq(data)
            Accuracy_LSTM['first'].append(R_sq_first)
            Accuracy_LSTM['Middle'].append(R_sq_middle)
            Accuracy_LSTM['all'].append(R_sq_all)
            Accuracy_LSTM['Card_ID'].append(Card_ID)
        else:
            error_first_temp, Accuracy_first_temp, error_middle_temp, Accuracy_temp, accuracy_all = data_process_discrete(data)
            #print (error_first_temp)
            error_first = pd.concat([error_first, error_first_temp], axis = 0)
            error_middle = pd.concat([error_middle, error_middle_temp], axis = 0)
            Accuracy_LSTM['first'].append(Accuracy_first_temp)
            Accuracy_LSTM['Middle'].append(Accuracy_temp)
            Accuracy_LSTM['all'].append(accuracy_all)
            Accuracy_LSTM['Card_ID'].append(Card_ID)

    ############## MC

    for Card_ID in Card_ID_used_for_base:
        if output_fig == 'duration':
            # file_name = data_path + 'results/result_MC' + str(Card_ID) + '.csv'
            file_name = data_path + 'results/result_LR' + str(Card_ID) + 'test.csv'
        else:
            file_name = data_path + 'results/result_Location_MC' + str(Card_ID) + '.csv'
        # if os.path.exists(file_name) == False:
        #     print(Card_ID, 'does not exist for Base')
        #     continue
        data = pd.read_csv(file_name)
        if output_fig == 'duration':
            if duration_error == 'RMSE':
                R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_RMSE(data)
            elif duration_error == 'MAPE':
                R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_MAPE(data)
            else:
                R_sq_first, R_sq_middle, R_sq_all = data_process_continuous_R_sq(data)
            Accuracy_base['first'].append(R_sq_first)
            Accuracy_base['Middle'].append(R_sq_middle)
            Accuracy_base['all'].append(R_sq_all)
            Accuracy_base['Card_ID'].append(Card_ID)
        else:
            error_first_temp, Accuracy_first_temp, error_middle_temp, Accuracy_temp, accuracy_all = data_process_discrete(data)
            # print (error_first_temp)
            error_first_base = pd.concat([error_first_base, error_first_temp], axis=0)
            error_middle_base = pd.concat([error_middle_base, error_middle_temp], axis=0)
            Accuracy_base['first'].append(Accuracy_first_temp)
            Accuracy_base['Middle'].append(Accuracy_temp)
            Accuracy_base['Card_ID'].append(Card_ID)
            Accuracy_base['all'].append(accuracy_all)


    # ====================


    ##############
    Accuracy_IOHMM = pd.DataFrame(Accuracy)
    Accuracy_base = pd.DataFrame(Accuracy_base)
    Accuracy_LSTM = pd.DataFrame(Accuracy_LSTM)

    return Accuracy_IOHMM, Accuracy_base, Accuracy_LSTM

if __name__ == '__main__':

    data_path = '../data/'
    with open (data_path + 'individual_ID_list_test', 'rb') as fp:
        individual_ID_list = pickle.load(fp)

    duration_error = 'R_sq'  # RMSE MAPE R_sq
    Mean_or_median = 'Mean'  # Mean Median
    Accuracy_IOHMM, Accuracy_Base, Accuracy_LSTM = generate_accuracy_file(individual_ID_list, output_fig,
                                                                          duration_error=duration_error)  #

