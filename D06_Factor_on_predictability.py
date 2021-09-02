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
from scipy.stats import entropy
from matplotlib.ticker import FormatStrFormatter
import statsmodels.api as sm


from sklearn.metrics import r2_score





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
    # ##############
    Accuracy_IOHMM = pd.DataFrame(Accuracy)
    # Accuracy_base = pd.DataFrame(Accuracy_base)
    # Accuracy_LSTM = pd.DataFrame(Accuracy_LSTM)

    return Accuracy_IOHMM #, Accuracy_base, Accuracy_LSTM



def generate_pred_acc(individual_ID_list, data_path):
    output_fig_list = ['duration','location']
    acc_out = []
    duration_error = 'R_sq'  # RMSE MAPE R_sq
    Mean_or_median = 'Mean'  # Mean Median
    for output_fig in output_fig_list:
        Accuracy_IOHMM = get_accuracy_and_num_act(individual_ID_list, output_fig, duration_error = duration_error) #
        acc_out.append(Accuracy_IOHMM)
        Accuracy_IOHMM.to_csv(data_path + 'all_prediction_accuracy_' + output_fig + '.csv', index = False)

    return acc_out[0], acc_out[1]



def generate_data_for_regression(samples, acc_dur, acc_loc, card_type):

    station_district = pd.read_csv('../data/HK_district_income/station_district.txt')
    station_district = station_district.loc[:,['Station_number','Station_name','ID','ENAME']]
    income_district = pd.read_csv('../data/HK_district_income/HH_income_district_level.csv')

    district_id = pd.read_csv('../data/HK_district_income/district_id.csv')
    eighteen_district_and_three_areas = pd.read_csv('../data/HK_district_income/eighteen_district_and_three_areas.csv')
    district_id = district_id.merge(eighteen_district_and_three_areas,left_on=['ID'],right_on=['Distrcit_id'])
    district_id['Name_upper'] = district_id['Name'].str.upper()


    income_district['District_name_upper'] = income_district['District_name'].str.upper()

    income_district = income_district.merge(district_id[['ID','Name_upper','Big_area']],left_on = ['District_name_upper'],right_on=['Name_upper'])

    station_district_income = station_district.merge(income_district[['District_name_upper','Median_HH_income_all','Big_area']], left_on = ['ENAME'],right_on = ['District_name_upper'], how = 'left')
    assert len(station_district_income.loc[station_district_income['Median_HH_income_all'].isna()]) == 0


    final_output = samples.loc[:,['Card_ID']].drop_duplicates()
    final_output = final_output.sort_values(['Card_ID']).reset_index(drop=True)
    ################ num travel per day mean std
    sample_group = samples.groupby(['Card_ID','seq_ID'])['act_ID'].max().reset_index()
    sample_group = sample_group.rename(columns = {'act_ID':'num_act_in_day'})
    sample_group_used = sample_group.groupby(['Card_ID']).agg({'num_act_in_day': 'mean'}).reset_index()
    sample_group_used = sample_group_used.rename(columns = {'num_act_in_day': 'num_trip_in_day_mean'})
    sample_group_used['num_trip_in_day_std'] = sample_group.groupby(['Card_ID'])['num_act_in_day'].std().reset_index()['num_act_in_day']
    # sample_group.columns = sample_group.columns.droplevel()
    # sample_group = sample_group.rename(columns = {'num_act_in_day/mean':'num_act_in_day_mean', 'num_act_in_day/std': 'num_act_in_day_std'})
    # print(sample_group.columns)
    final_output = final_output.merge(sample_group_used, on = ['Card_ID'])
    ################ num days with travel
    sample_group = samples.groupby(['Card_ID'])['seq_ID'].max().reset_index()
    sample_group = sample_group.rename(columns = {'seq_ID':'num_days_with_travel'})
    final_output = final_output.merge(sample_group, on = ['Card_ID'])

    ############### first act departure time std
    sample_first = samples.loc[samples['act_ID'] == 0]
    sample_group = sample_first.groupby(['Card_ID'])['duration'].std().reset_index() # duration of the first act is the departure time.
    sample_group = sample_group.rename(columns = {'duration':'first_departure_time_std'})
    sample_group['first_departure_time_std'] = sample_group['first_departure_time_std'] / 60 # to min
    final_output = final_output.merge(sample_group, on = ['Card_ID'])

    ############### entropy trip origin, destination, act duration


    temp_dic = {'Card_ID':[],'entropy_act_dur':[],'entropy_trip_origin':[],'entropy_trip_des':[],'home_district':[],'home_big_area':[],'median_hh_income':[]}
    sample_group = samples.groupby('Card_ID')
    for idx, info in sample_group:
        data_used = info.copy()
        data_used['dur_hour'] = data_used['duration'] // 3600
        # dur
        dur_list = list(data_used.loc[data_used['if_last'] == 0, 'dur_hour'])
        unique, counts = np.unique(np.array(dur_list), return_counts=True)
        fre = counts/sum(counts)
        entropy_act_dur = entropy(fre)

        # origin
        origin_list = list(data_used.loc[data_used['location_o']!=-1, 'location_o'])
        unique, counts = np.unique(np.array(origin_list), return_counts=True)
        fre = counts/sum(counts)
        entropy_trip_origin = entropy(fre)

        # destination
        des_list = list(data_used.loc[data_used['location']!=-1, 'location'])
        unique, counts = np.unique(np.array(des_list), return_counts=True)
        fre = counts/sum(counts)
        entropy_trip_des = entropy(fre)


        # first trip origin #inferred home
        first_trip = data_used.loc[data_used['act_ID'] == 1]
        home_loc_list = first_trip.groupby(['location_o'])['seq_ID'].count().reset_index()
        home_loc = home_loc_list.loc[home_loc_list['seq_ID'] == home_loc_list['seq_ID'].max(),'location_o'].iloc[0]
        # then use the ID to match to district + HH income
        district = station_district_income.loc[station_district_income['Station_number'] == home_loc, 'ENAME'].iloc[0]
        Median_income = station_district_income.loc[station_district_income['Station_number'] == home_loc, 'Median_HH_income_all'].iloc[0]
        big_area = station_district_income.loc[station_district_income['Station_number'] == home_loc, 'Big_area'].iloc[0]
        a=1
        temp_dic['Card_ID'].append(idx)
        temp_dic['entropy_act_dur'].append(entropy_act_dur)
        temp_dic['entropy_trip_origin'].append(entropy_trip_origin)
        temp_dic['entropy_trip_des'].append(entropy_trip_des)
        temp_dic['home_district'].append(district)
        temp_dic['home_big_area'].append(big_area)
        temp_dic['median_hh_income'].append(Median_income)


    temp_dic = pd.DataFrame(temp_dic)
    final_output = final_output.merge(temp_dic, on = ['Card_ID'])

    final_output = final_output.merge(acc_dur[['Card_ID','all','Total_act']], on =['Card_ID'])
    final_output = final_output.rename(columns = {'all':'R_sq_dur'})
    final_output = final_output.merge(acc_loc[['Card_ID','all']], on =['Card_ID'])
    final_output = final_output.rename(columns = {'all':'acc_loc'})

    final_output = final_output.merge(card_type, on = ['Card_ID'], how = 'left')

    print(final_output['Card_type'].value_counts())

    final_output['if_student'] = 0
    final_output['if_senior'] = 0

    final_output.loc[final_output['Card_type'] == 'STD', 'if_student'] = 1
    final_output.loc[final_output['Card_type'] == 'SEN', 'if_senior'] = 1

    return final_output


def run_linear_reg(data_path, data):
    data['if_in_HK_island'] = 0
    data['if_in_NT'] = 0
    data.loc[data['home_big_area'] == 'HK','if_in_HK_island'] = 1
    data.loc[data['home_big_area'] == 'NT','if_in_NT'] = 1
    #############duration
    # col_X = ['num_trip_in_day_mean','num_trip_in_day_std','num_days_with_travel','first_departure_time_std',
    #          'entropy_act_dur','if_student', 'if_senior']
    col_X = ['num_trip_in_day_mean','num_trip_in_day_std','num_days_with_travel','first_departure_time_std',
             'if_student', 'if_senior','if_in_HK_island','if_in_NT']
    col_Y = ['R_sq_dur']
    data['median_hh_income'] /= 10000
    X = data.loc[:,col_X].values
    Y = data.loc[:,col_Y].values
    X = sm.add_constant(X)
    est = sm.OLS(Y, X)
    est2 = est.fit()
    results_summary = est2.summary()
    print('Duration', results_summary)
    results_as_html = results_summary.tables[1].as_html()
    table = pd.read_html(results_as_html, header=0, index_col=0)[0]
    table['Variable'] = ['Intercept'] + col_X

    table.to_csv('table/estimate_para_on_R_sq_dur_no_entropy.csv',index=False)
    #############loc
    # col_X = ['num_trip_in_day_mean','num_trip_in_day_std','num_days_with_travel','first_departure_time_std',
    #          'entropy_trip_origin','if_student', 'if_senior']
    col_X = ['num_trip_in_day_mean','num_trip_in_day_std','num_days_with_travel','first_departure_time_std',
             'if_student', 'if_senior','if_in_HK_island','if_in_NT']
    col_Y = ['acc_loc']

    X = data.loc[:,col_X].values
    Y = data.loc[:,col_Y].values
    X = sm.add_constant(X)
    est = sm.OLS(Y, X)
    est2 = est.fit()
    results_summary = est2.summary()
    print('Location', results_summary)
    results_as_html = results_summary.tables[1].as_html()
    table = pd.read_html(results_as_html, header=0, index_col=0)[0]
    table['Variable'] = ['Intercept'] + col_X

    table.to_csv('table/estimate_para_on_acc_loc_no_entropy.csv',index=False)


if __name__ == '__main__':

    data_path = '../data/'

    num_ind = 1000
    with open(data_path + 'individual_ID_list_test_' + str(num_ind) + '.pickle', 'rb') as fp:
        individual_ID_list_test = pickle.load(fp)
    individual_ID_list = individual_ID_list_test[0:500]

    GENERATE_ACCURACY_DATA = False
    GENERATE_REG_DATA = False

    if GENERATE_ACCURACY_DATA:
        acc_dur, acc_loc = generate_pred_acc(individual_ID_list, data_path)
    else:
        acc_dur = pd.read_csv(data_path + 'all_prediction_accuracy_' + 'duration' + '.csv')
        acc_loc = pd.read_csv(data_path + 'all_prediction_accuracy_' + 'location' + '.csv')

    if GENERATE_REG_DATA:
        samples = pd.read_csv(data_path + 'samples/sample_500_all_201407_201408.csv')

        card_type = pd.read_csv(data_path + 'sample_card_type.csv')
        card_type = card_type.rename(columns = {'csc_phy_id': 'Card_ID', 'txn_subtype_co':'Card_type'})
        print(pd.unique(card_type['Card_type']))
        data = generate_data_for_regression(samples, acc_dur, acc_loc, card_type)
        data.to_csv(data_path + 'individual_predict_acc.csv',index=False)
    else:
        data = pd.read_csv(data_path + 'individual_predict_acc.csv')

    count_district = data.groupby(['home_big_area'])['Card_ID'].count().reset_index()
    run_linear_reg(data_path, data)

