#
from __future__ import  division


import timeit

from sklearn import preprocessing
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
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
from sklearn.feature_selection import SelectPercentile, chi2, SelectFromModel, f_regression
from sklearn.svm import LinearSVC
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import r2_score




from sklearn.ensemble import ExtraTreesClassifier
import os




Accurate_duration=[]
#filename1='data/activity_index_test.txt'
#file1=open(filename1,'r')
#activity_index_test=eval(file1.read()) 
activity_index_test = {}



def process_data(Card_ID, data, test_proportion, C, dependent_variables, percent_feature, test_last, model_based_select, SCALAR_DURATION):

    data.loc[data['duration_last']==-1,'duration_last'] = 0 # first activity, assign to 0

    data['if_first'] = 0
    data.loc[data['act_ID'] == 0, 'if_first'] = 1

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
    y = np.array(data.loc[:,dependent_variables])
    print(x_array_minmax.shape)
    if C == -1 and percent_feature == -1:
        Ut_list_1 = []
        Ut_list_2 = []
        Ut_list_new = Ut_list
    else:
        # ============
        if model_based_select:
            if len(dependent_variables) >0:
                lsvc = LinearSVC(C=C, penalty="l1", dual=False).fit(x_array_minmax, y[:,1])
                Feature_select2 = SelectFromModel(lsvc, prefit=True)
            else:
                lsvc = LinearSVC(C = C, penalty="l1", dual=False).fit(x_array_minmax, y)
                Feature_select = SelectFromModel(lsvc, prefit=True)
        #----------
        # clf = ExtraTreesClassifier(n_estimators=50)
        # clf = clf.fit(x_array_minmax, y)
        # Feature_select = SelectFromModel(clf, prefit=True)
        #----------
        else:
            if len(dependent_variables) > 0:
                Feature_select2 = SelectPercentile(chi2, percentile=percent_feature).fit(x_array_minmax, y[:,1])
                Feature_select1 = SelectPercentile(f_regression, percentile=percent_feature).fit(x_array_minmax, y[:,0])
                a=1
            else:
                Feature_select = SelectPercentile(chi2, percentile=percent_feature).fit(x_array_minmax, y)
        # ============
        if len(dependent_variables) > 0:
            # thresh2 = Feature_select2.threshold_
            # X_new2 = Feature_select2.transform(x_array_minmax)
            if model_based_select:
                idx_features2 = Feature_select2.get_support(indices = True)
                num_feature = len(idx_features2)
                clf = LassoCV().fit(x_array_minmax, y[:, 0])
                importance = np.abs(clf.coef_)
                idx_thresh = importance.argsort()[-num_feature]

                threshold = importance[idx_thresh]
                sfm = SelectFromModel(clf, threshold=threshold)
                sfm.fit(x_array_minmax, y[:, 0])
                # X_new1 = sfm.transform(x_array_minmax)
                idx_features1 = sfm.get_support(indices = True)


                used_feature_index = list(set(idx_features2).union(idx_features1))
                Ut_list_new = [Ut_list[i] for i in used_feature_index]
                Ut_list_1 = [Ut_list[i] for i in idx_features1]
                Ut_list_2 = [Ut_list[i] for i in idx_features2]
            else:
                idx_features2 = Feature_select2.get_support(indices = True)
                idx_features1 = Feature_select1.get_support(indices = True)
                # assert len(idx_features1) == len(idx_features2)
                used_feature_index = list(set(idx_features2).union(idx_features1))
                Ut_list_new = [Ut_list[i] for i in used_feature_index]
                Ut_list_1 = [Ut_list[i] for i in idx_features1]
                Ut_list_2 = [Ut_list[i] for i in idx_features2]
        else:
            X_new = Feature_select.transform(x_array_minmax)

    # Ut_list_new = [Ut_list[i] for i in range(len(Ut_list)) if used_feature_index[i]]

    # print(X_new.shape)
    data.loc[:,Ut_list] = x_array_minmax



    if SCALAR_DURATION:
        min_max_scaler_dep = preprocessing.MinMaxScaler()
        data[dependent_variables[0]] = min_max_scaler_dep.fit_transform(data[[dependent_variables[0]]])
    else:
        min_max_scaler_dep = None

    total_days = data['seq_ID'].max()
    train_days = int(total_days - round(total_days*test_proportion))
    if test_last:
        # last 30 days
        data_train = data.loc[data['seq_ID']<=train_days]
        data_test = data.loc[data['seq_ID']>train_days]
    else:
        random.seed(Card_ID)
        test_seq = random.sample(list(range(1,total_days+1)), total_days - train_days)
        data_train = data.loc[~data['seq_ID'].isin(test_seq)]
        data_test = data.loc[data['seq_ID'].isin(test_seq)]
    return min_max_scaler,min_max_scaler_dep, data, data_train, data_test, Ut_list_new, Ut_list_1, Ut_list_2


def predict(sequence, num_states, dependent_variables, Card_ID, data, SHMM, Ut_list, Ut_list_1,Ut_list_2,
            save_info_list, C, percent_feature, save_predicted_rank, scaler_y, SCALAR_DURATION):
    results={}
    show_duration_predict = True
    for info in save_info_list:
        results[info] = []

    Dt_h_2 = np.array(sorted(data.loc[:,dependent_variables[1]].unique()))
    Dt_h_1 = np.array(np.arange(round(min(data['duration'])) - 0.5, round(max(data['duration'])) + 0.5, 0.01)) # candidate duration
    for seq in sequence:
        seq = seq.reset_index(drop=True)
        for idx, row in seq.iterrows():
            if idx == 0:
                X_emi_1 = np.array([row[Ut_list_1]])
                X_emi_2 = np.array([row[Ut_list_2]])
                ############################ location
                X_ini = np.array([row[Ut_list]])
                Log_ini_st = SHMM.model_initial.predict_log_proba(X_ini).reshape(num_states,)
                log_Emission = np.zeros((len(Dt_h_2), num_states))

                Ut_input = np.repeat(X_emi_2, len(Dt_h_2), axis=0)
                for st in range(num_states):
                    # print(Dt_h.shape)
                    # print(X.shape)
                    log_Emission[:, st] = SHMM.model_emissions[st][1].loglike_per_sample(Ut_input, Dt_h_2)
                log_P_temp = log_Emission + Log_ini_st
                P_final = np.sum(np.exp(log_P_temp), axis=1)
                Predict_value = Dt_h_2[np.argmax(P_final)]
                True_value = row[dependent_variables[1]]

                comb_results = [[P_final[i], Dt_h_2[i]] for i in range(len(Dt_h_2))]
                comb_results = sorted(comb_results, reverse=True)

                for i in range(save_predicted_rank):
                    if i >= len(comb_results):
                        rank_name = 'Predict' + str(i+1)
                        results[rank_name].append(-1) # no 20 candidates
                    else:
                        rank_name = 'Predict' + str(i+1)
                        results[rank_name].append(comb_results[i][1])
                # plt.plot(Dt_h,P_final)
                # plt.plot([True_value,True_value],[0,max(P_final)])
                # plt.show()
                results['ID'].append(row['ID'])
                results['Card_ID'].append(Card_ID)
                results['Ground_truth'].append(True_value)
                if Predict_value == True_value:
                    results['Correct'].append(1)
                else:
                    results['Correct'].append(0)
                results['activity_index'].append(idx)
                results['total_activity'].append(num_states)
                results['percent_feature'].append(percent_feature)
                results['C_reg'].append(C)

                ################################################################ continuous duration

                Log_ini_st = SHMM.model_initial.predict_log_proba(X_ini).reshape(num_states,)
                predict_Emission = np.zeros(num_states)
                dispersion = np.zeros(num_states)
                for st in range(num_states):
                    # print(Dt_h.shape)
                    # print(X.shape)
                    predict_Emission[st] = SHMM.model_emissions[st][0].predict(X_emi_1)
                    # dispersion[st] = SHMM.model_emissions[st][0].get_dispersion(Y_len = 1) #
                    # a=1
                P_int_st = np.exp(Log_ini_st)
                Predict_value_mean = sum(P_int_st * predict_Emission)
                # Predict_value_var = sum((P_int_st**2) * dispersion)

                True_value = row[dependent_variables[0]]

                if SCALAR_DURATION:
                    predict_dur = scaler_y.inverse_transform(np.array([Predict_value_mean]).reshape(1, -1))[0][0]
                    true_value = scaler_y.inverse_transform(np.array([True_value]).reshape(1, -1))[0][0]
                    results['Predict_duration'].append(predict_dur)
                    results['Ground_truth_duration'].append(true_value)
                else:
                    results['Predict_duration'].append(Predict_value_mean)
                    results['Ground_truth_duration'].append(True_value)
                # results['Predict_duration_log_std'].append(np.sqrt(Predict_value_var))


            else:
                X_emi_1 = np.array([row[Ut_list_1]])
                X_emi_2 = np.array([row[Ut_list_2]])
                X_ini = np.array([row[Ut_list]])

                ############################ location # second dep
                # calculate log_alpha
                Known_seq = seq.loc[0:idx-1,:]
                n_records = max(Known_seq.index) + 1
                log_prob_initial = Log_ini_st
                log_prob_transition = np.zeros((n_records - 1, num_states, num_states))
                if n_records>1:
                    X_to_transit = np.array(Known_seq.loc[1:,Ut_list])
                    for st in range(num_states):
                        log_prob_transition[:, st, :] = SHMM.model_transition[st].predict_log_proba(X_to_transit)
                    assert log_prob_transition.shape == (n_records - 1, num_states, num_states)
                # emission probability
                log_Emission = np.zeros((n_records, num_states))
                inp_emissions = np.array(Known_seq.loc[0:,Ut_list_1])
                out_emissions = np.array(Known_seq.loc[0:,[dependent_variables[1]]])
                model_collection = [models[1] for models in SHMM.model_emissions]
                # print (model_collection)
                log_Emission += np.vstack([model.loglike_per_sample(
                    inp_emissions.astype('float64'),out_emissions) for model in model_collection]).T
                # print (np.exp(log_Ey))
                # forward backward to calculate posterior
                # print(out_emissions)
                # print(out_emissions.shape)
                # # print ('-----')
                # print(inp_emissions)
                # print(inp_emissions.shape)
                log_gamma, log_epsilon, log_likelihood, log_alpha = forward_backward(
                    log_prob_initial, log_prob_transition, log_Emission, {})
                # ------predict:

                log_alpha_new = np.zeros((num_states,))
                for j in range(num_states):
                    temp_alpha = 0
                    for i in range(num_states):
                        temp_alpha += np.exp(SHMM.model_transition[i].predict_log_proba(X_ini)[0,j] + log_alpha[-1,i])
                        # the first 0 is because we only have one row, so select the first. the second -1 is
                        # because log_alpha has the shape of t,k , where t is the number of timestamps (length) of the sequence.
                        # where k is the number of states of the HMM
                    log_alpha_new[j] = np.log(temp_alpha)
                log_P_D1T_u1Th=np.zeros((num_states,))
                for i in range(num_states):
                    log_P_D1T_u1Th[i]=log_alpha_new[i] - logsumexp(log_alpha_new[:])


                log_Emission = np.zeros((len(Dt_h_2), num_states))
                Ut_input_2 = np.repeat(X_emi_2, len(Dt_h_2), axis=0)
                for st in range(num_states):
                    # print(Dt_h.shape)
                    # print(X.shape)
                    log_Emission[:, st] = SHMM.model_emissions[st][1].loglike_per_sample(Ut_input_2, Dt_h_2)
                log_P_temp = log_Emission + log_P_D1T_u1Th
                P_final = np.sum(np.exp(log_P_temp), axis=1)
                Predict_value = Dt_h_2[np.argmax(P_final)]
                True_value = row[dependent_variables[1]]

                comb_results = [[P_final[i], Dt_h_2[i]] for i in range(len(Dt_h_2))]
                comb_results = sorted(comb_results, reverse=True)

                for i in range(save_predicted_rank):
                    if i >= len(comb_results):
                        rank_name = 'Predict' + str(i+1)
                        results[rank_name].append(-1) # no 20 candidates
                    else:
                        rank_name = 'Predict' + str(i+1)
                        results[rank_name].append(comb_results[i][1])


                results['ID'].append(row['ID'])
                results['Card_ID'].append(Card_ID)
                results['Ground_truth'].append(True_value)
                if Predict_value == True_value:
                    results['Correct'].append(1)
                else:
                    results['Correct'].append(0)
                results['activity_index'].append(idx)
                results['total_activity'].append(num_states)
                results['percent_feature'].append(percent_feature)
                results['C_reg'].append(C)




                ############################ duration
                # calculate log_alpha
                Known_seq = seq.loc[0:idx-1,:]
                n_records = max(Known_seq.index) + 1
                log_prob_initial = Log_ini_st
                log_prob_transition = np.zeros((n_records - 1, num_states, num_states))
                if n_records>1:
                    X_to_transit = np.array(Known_seq.loc[1:,Ut_list])
                    for st in range(num_states):
                        log_prob_transition[:, st, :] = SHMM.model_transition[st].predict_log_proba(X_to_transit)
                    assert log_prob_transition.shape == (n_records - 1, num_states, num_states)
                # emission probability
                log_Emission = np.zeros((n_records, num_states))
                inp_emissions = np.array(Known_seq.loc[0:,Ut_list_1])
                out_emissions = np.array(Known_seq.loc[0:,[dependent_variables[0]]])
                model_collection = [models[0] for models in SHMM.model_emissions]
                # print (model_collection)
                log_Emission += np.vstack([model.loglike_per_sample(
                    inp_emissions.astype('float64'),out_emissions) for model in model_collection]).T
                # print (np.exp(log_Ey))
                # forward backward to calculate posterior
                # print(out_emissions)
                # print(out_emissions.shape)
                # # print ('-----')
                # print(inp_emissions)
                # print(inp_emissions.shape)
                log_gamma, log_epsilon, log_likelihood, log_alpha = forward_backward(
                    log_prob_initial, log_prob_transition, log_Emission, {})
                # ------predict:

                log_alpha_new = np.zeros((num_states,))
                for j in range(num_states):
                    temp_alpha = 0
                    for i in range(num_states):
                        temp_alpha += np.exp(SHMM.model_transition[i].predict_log_proba(X_ini)[0,j] + log_alpha[-1,i])
                        # the first 0 is because we only have one row, so select the first. the second -1 is
                        # because log_alpha has the shape of t,k , where t is the number of timestamps (length) of the sequence.
                        # where k is the number of states of the HMM
                    log_alpha_new[j] = np.log(temp_alpha)
                log_P_D1T_u1Th=np.zeros((num_states,))
                for i in range(num_states):
                    log_P_D1T_u1Th[i]=log_alpha_new[i] - logsumexp(log_alpha_new[:])


                predict_Emission = np.zeros(num_states)
                dispersion = np.zeros(num_states)
                for st in range(num_states):
                    # print(Dt_h.shape)
                    # print(X.shape)
                    predict_Emission[st] = SHMM.model_emissions[st][0].predict(X_emi_1)
                    dispersion[st] = SHMM.model_emissions[st][0].get_dispersion(Y_len = 1) #
                    # a=1
                P_st = np.exp(log_P_D1T_u1Th)
                Predict_value_mean = sum(P_st * predict_Emission)
                Predict_value_var = sum((P_st**2) * dispersion)

                True_value = row[dependent_variables[0]]
                if SCALAR_DURATION:
                    predict_dur = scaler_y.inverse_transform(np.array([Predict_value_mean]).reshape(1, -1))[0][0]
                    true_value = scaler_y.inverse_transform(np.array([True_value]).reshape(1, -1))[0][0]
                    results['Predict_duration'].append(predict_dur)
                    results['Ground_truth_duration'].append(true_value)
                else:
                    results['Predict_duration'].append(Predict_value_mean)
                    results['Ground_truth_duration'].append(True_value)


    return results

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


def Model(Card_ID,percent_feature = -1,C = -1.0, FEATURE_SELECTION = False,SCALAR_DURATION = True, L2_SELECTION = False, l2_penalty = 0.15, SAVE_MODEL = False):

    start_time = timeit.default_timer()
    activity_index_test[Card_ID]=[]

    file_name_train= data_path + 'samples/sample_'+str(Card_ID)+'_201407_201408_all.csv'
    data = pd.read_csv(file_name_train)
    data = data.loc[data['if_last']==0,:] # drop the last one, it will distract the training!
    test_proportion = 0.2
    data['duration_hour'] = round(data['duration'] / 3600).astype('int')
    #----
    dependent_variables = ['duration','Next_tapin_station'] #contiuous
    if percent_feature == -1:
        model_based_select = True
    else:
        model_based_select = False
    test_last = False
    if FEATURE_SELECTION:
        scaler, scaler_y, data, data_train, data_test, Ut_list, Ut_list_1, Ut_list_2 = process_data(Card_ID, data, test_proportion, C, dependent_variables,
                                                                                  percent_feature, test_last, model_based_select, SCALAR_DURATION)
    else:
        scaler, scaler_y, data, data_train, data_test, Ut_list, _, _ = process_data(Card_ID, data, test_proportion, -1, dependent_variables, -1, test_last,
                     model_based_select, SCALAR_DURATION)
        Ut_list_1 = Ut_list
        Ut_list_2 = Ut_list

    save_info_list1 = ['ID', 'Card_ID']
    save_predicted_rank = 20
    save_info_list2 = ['Predict'+str(i+1) for i in range(save_predicted_rank)]
    save_info_list3 = ['Ground_truth','Correct','activity_index','total_activity','percent_feature','C_reg',
                       'Predict_duration','Ground_truth_duration'] #Predict_duration_log_std
    save_info_list = save_info_list1 + save_info_list2 + save_info_list3
    #-------

    range_n_clusters=[3,4,5,6,7]
    silhouette_avg_list=[]
    new_data = data.loc[:,Ut_list + dependent_variables]
    for n_clusters in range_n_clusters:
    
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(new_data)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(new_data, cluster_labels)
        #print("For n_clusters =", n_clusters,
              #"The average silhouette_score is :", silhouette_avg)    
        silhouette_avg_list.append(silhouette_avg)
    num_states = range_n_clusters[silhouette_avg_list.index(max(silhouette_avg_list))]
    #---Fix number of states
    # num_states = 4
    #---
    print ('Num states:', num_states)

    seq_list = list(data_train['seq_ID'].unique())

    sequence = []
    for item in seq_list:
        temp_data = data_train.loc[data_train['seq_ID'] == item]
        temp_data = temp_data.reset_index()
        sequence.append(temp_data)
        del temp_data


    test_seq_list = list(data_test['seq_ID'].unique())
    test_sequence = []
    for item in test_seq_list:
        temp_data = data_test.loc[data_test['seq_ID'] == item]
        temp_data = temp_data.reset_index()
        test_sequence.append(temp_data)
        del temp_data
    
    SHMM = UnSupervisedIOHMM(num_states=num_states, max_EM_iter=70, EM_tol=0.5) # max_EM_iter=70
    
    # This model has only two outputs which is modeled by a linear regression model
    if L2_SELECTION:
        SHMM.set_models(model_emissions = [OLS(alpha = l2_penalty, reg_method = 'l2'), DiscreteMNL(alpha=1, reg_method = 'l2')],
                        model_transition=CrossEntropyMNL(alpha=5, solver='lbfgs',reg_method = 'l2'), #
                        model_initial=CrossEntropyMNL(alpha=5, solver='lbfgs',reg_method = 'l2'))
    else:
        SHMM.set_models(model_emissions = [OLS(alpha = 0.15, reg_method = 'l2'), DiscreteMNL(alpha=1, reg_method = 'l2')],
                        model_transition=CrossEntropyMNL(alpha=5, solver='lbfgs',reg_method = 'l2'), #
                        model_initial=CrossEntropyMNL(alpha=5, solver='lbfgs',reg_method = 'l2'))
    #alpha is the regularization term
    

    SHMM.set_inputs(covariates_initial = Ut_list,
                    covariates_transition = Ut_list,
                    covariates_emissions = [Ut_list_1,Ut_list_2])
          
    
    SHMM.set_outputs([[dependent_variables[0]],[dependent_variables[1]]])
    SHMM.set_data(sequence)
    print('Start training')
    SHMM.train()

    # save model


    end_time = timeit.default_timer()
    print ('Training time:', end_time-start_time)
    #print(SHMM.model_emissions[0][0].coef) # first[] is state, the second[] is the emission model ID
    #print(SHMM.model_emissions[1][0].coef)

    #------Training Accuracy----
    results = predict(sequence, num_states, dependent_variables, Card_ID, data, SHMM, Ut_list, Ut_list_1,Ut_list_2, save_info_list,
                      C,percent_feature, save_predicted_rank, scaler_y, SCALAR_DURATION)
    result_df_train = pd.DataFrame(results)
    num1 = len(result_df_train)
    result_df_train = result_df_train.dropna()
    num2 = len(result_df_train)
    if num2 < num1:
        print('ID',Card_ID, 'DropNA train',num2 - num1)
    Accuracy_first, Accuracy_middle, Accuracy_all_train, N_first, N_middle, N_all, RMSE_train, MAPE, MAE, R_sq_train = calculate_accuracy(result_df_train)
    # print(Card_ID, 'Training Accuracy-First:', Accuracy_first, N_first)
    # print(Card_ID, 'Training Accuracy-Middle:', Accuracy_middle, N_middle)
    # print(Card_ID, 'Total Training Accuracy:', Accuracy_all, N_all)
    # print('--------')
    #------Prediction Accuracy----
    results = predict(test_sequence, num_states, dependent_variables, Card_ID, data, SHMM, Ut_list, Ut_list_1,Ut_list_2,
                      save_info_list, C, percent_feature, save_predicted_rank, scaler_y, SCALAR_DURATION)
    result_df_test=pd.DataFrame(results)
    num1 = len(result_df_test)
    result_df_test = result_df_test.dropna()
    num2 = len(result_df_test)
    if num2 < num1:
        print('ID',Card_ID, 'DropNA test',num2 - num1)
    Accuracy_first, Accuracy_middle, Accuracy_all_test, N_first, N_middle, N_all, RMSE, MAPE, MAE, R_sq = calculate_accuracy(result_df_test)
    # print(Card_ID, 'Testing Accuracy-First:', Accuracy_first, N_first)
    # print(Card_ID, 'Testing Accuracy-Middle:', Accuracy_middle, N_middle)
    # print(Card_ID, 'Total Testing Accuracy:', Accuracy_all, N_all)
    # print('--------')
    Accuracy_test = deepcopy(Accuracy_all_test)
    #--------Base model accuracy

    # print(Card_ID, 'Base Testing Accuracy-First:', Accuracy_first, N_first)
    # print(Card_ID, 'Base Testing Accuracy-Middle:', Accuracy_middle, N_middle)
    # print(Card_ID, 'Base Total Testing Accuracy:', Accuracy_all, N_all)
    # print('--------')
    # result_df.to_csv(file_name.replace('.csv','test.csv'),index=False)
    end2_time=timeit.default_timer()
    print ('Predicting time:', end2_time-end_time)
    print ('--------')
    return Accuracy_test,Accuracy_all_train, result_df_train, result_df_test, RMSE, MAPE, MAE,R_sq, R_sq_train, SHMM, data, scaler_y, scaler


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
    if len(data_middle)>0:
        R_sq_middle = r_sq_for_two_parts(data_middle, mean_y)
    else:
        R_sq_middle = None

    return R_sq_first, R_sq_middle, R_sq_all



if __name__ == '__main__':

    data_path = '../data/'
    # with open(data_path + 'individual_ID_list_test', 'rb') as fp:
    #     individual_ID_list_test = pickle.load(fp)

    num_ind = 1000
    with open(data_path + 'individual_ID_list_test_' + str(num_ind) + '.pickle', 'rb') as fp:
        individual_ID_list_test = pickle.load(fp)
    individual_ID_list_test = individual_ID_list_test[0:500]


    # card_ID = 949480450 #941443407 #931748826 # 949480450 #
    # individual_ID_list_test = [941443407]
    # individual_ID_list_test = [952322469, 994326032, 970538558, 945386614, 916941236]
    # idx = individual_ID_list_test.index(970456037)
    # print(individual_ID_list_test[0])
    # individual_ID_list_test = [individual_ID_list_test[436]]
    C_list = [0.05, 0.06, 0.07] #C_list = [0.05]#C_list = [0.05, 0.06, 0.07] ## #
    percent_feature_list = [30, 40, 50, 60, 70, 80] # percent_feature_list = [30]#percent_feature_list = [30,40,50,60,70,80] #
    OLS_l2_penalty = [0.05, 0.1, 0.15, 0.2]
    #
    FEATURE_SELECTION = False

    L2_SELECTION = True
    SAVE_MODEL = True

    SKIP_RUNNED_MODEL = True

    SCALAR_DURATION = True

    COMPARE_BASELINE = False
    count = 0
    for Card_ID in individual_ID_list_test:
        count +=1
        print('Current card:',Card_ID, 'num',count,'total',len(individual_ID_list_test))
        file_name = data_path + 'results/result_con_dur+loc_' + str(Card_ID) + '.csv'
        if SKIP_RUNNED_MODEL:
            file_name_test_ = file_name.replace('.csv', 'test.csv')
            if os.path.exists(file_name_test_):
                print ('Finish model', Card_ID)
                continue
        # if Card_ID in activity_index_test:
        # print ('Running model', Card_ID)

        if FEATURE_SELECTION:
            # return
            acc_list = []
            acc_list_train = []
            RMSE_list = []
            R_sq_list = []
            R_sq_list_train = []
            data_save = []
            R_sq_first_train = []
            R_sq_first_test = []
            R_sq_middle_train = []
            R_sq_middle_test = []

            for C in C_list:
                percent_feature = -1
                Accuracy_test,Accuracy_train, result_df_train, result_df_test, RMSE, \
                MAPE, MAE, R_sq_test, R_sq_train, IOHMM_MODEL,data, scaler_y, scaler_x  = Model(Card_ID,percent_feature,C,
                                                                      FEATURE_SELECTION, SCALAR_DURATION,SAVE_MODEL = SAVE_MODEL)
                acc_list.append(Accuracy_test)
                acc_list_train.append(Accuracy_train)
                RMSE_list.append(RMSE)
                R_sq_list.append(R_sq_test)
                R_sq_list_train.append(R_sq_train)
                data_save.append((result_df_train, result_df_test))

                R_sq_first, R_sq_middle, _ = data_process_continuous_R_sq(result_df_train)
                R_sq_first_train.append(R_sq_first)
                R_sq_middle_train.append(R_sq_middle)
                R_sq_first, R_sq_middle, _ = data_process_continuous_R_sq(result_df_test)
                R_sq_first_test.append(R_sq_first)
                R_sq_middle_test.append(R_sq_middle)


            for percent_feature in percent_feature_list:
                C = -1
                Accuracy_test,Accuracy_train, result_df_train, result_df_test, RMSE, \
                MAPE, MAE, R_sq_test, R_sq_train, IOHMM_MODEL, data, scaler_y, scaler_x  = Model(Card_ID,percent_feature,C,
                                                                      FEATURE_SELECTION, SCALAR_DURATION, SAVE_MODEL = SAVE_MODEL)
                acc_list.append(Accuracy_test)
                acc_list_train.append(Accuracy_train)
                RMSE_list.append(RMSE)
                R_sq_list.append(R_sq_test)
                R_sq_list_train.append(R_sq_train)
                data_save.append((result_df_train, result_df_test))

                R_sq_first, R_sq_middle, _ = data_process_continuous_R_sq(result_df_train)
                R_sq_first_train.append(R_sq_first)
                R_sq_middle_train.append(R_sq_middle)
                R_sq_first, R_sq_middle, _ = data_process_continuous_R_sq(result_df_test)
                R_sq_first_test.append(R_sq_first)
                R_sq_middle_test.append(R_sq_middle)


            idx_location = np.argsort(-np.array(acc_list))
            idx_duration = np.argsort(-np.array(R_sq_list))
            idx_all = idx_location + idx_duration

            best_idx = list(idx_all).index(min(idx_all))


            result_save_train = data_save[best_idx][0]
            result_save_test = data_save[best_idx][1]
        elif L2_SELECTION:
            acc_list = []
            acc_list_train = []
            RMSE_list = []
            R_sq_list = []
            R_sq_list_train = []
            data_save = []
            R_sq_first_train = []
            R_sq_first_test = []
            R_sq_middle_train = []
            R_sq_middle_test = []
            SAVE_MODEL_LIST = []
            raw_data_list = []
            scaler_y_list = []
            scaler_x_list = []

            for l2_penalty in OLS_l2_penalty:
                Accuracy_test,Accuracy_train, result_df_train, result_df_test, RMSE,\
                MAPE, MAE, R_sq_test, R_sq_train, IOHMM_MODEL, data, scaler_y, scaler_x  = Model(Card_ID, SCALAR_DURATION = SCALAR_DURATION,
                                                                      L2_SELECTION = L2_SELECTION, l2_penalty = l2_penalty, SAVE_MODEL = SAVE_MODEL)
                acc_list.append(Accuracy_test)
                acc_list_train.append(Accuracy_train)
                RMSE_list.append(RMSE)
                R_sq_list.append(R_sq_test)
                R_sq_list_train.append(R_sq_train)
                data_save.append((result_df_train, result_df_test))

                R_sq_first, R_sq_middle, _ = data_process_continuous_R_sq(result_df_train)
                R_sq_first_train.append(R_sq_first)
                R_sq_middle_train.append(R_sq_middle)
                R_sq_first, R_sq_middle, _ = data_process_continuous_R_sq(result_df_test)
                R_sq_first_test.append(R_sq_first)
                R_sq_middle_test.append(R_sq_middle)
                SAVE_MODEL_LIST.append(IOHMM_MODEL)
                scaler_x_list.append(scaler_x)
                scaler_y_list.append(scaler_y)
                raw_data_list.append(data)

            # idx_location = np.argsort(-np.array(acc_list))
            idx_duration = np.argsort(-np.array(R_sq_list))
            # idx_all = idx_location + idx_duration
            idx_all = idx_duration
            best_idx = list(idx_all).index(min(idx_all))
            IOHMM_MODEL = SAVE_MODEL_LIST[best_idx]
            data = raw_data_list[best_idx]
            scaler_y = scaler_y_list[best_idx]
            scaler_x = scaler_x_list[best_idx]
            if SAVE_MODEL:
                save_path = 'output/IOHMM/'
                with open(save_path + 'Trained_IOHMM_'+ str(Card_ID) + '.pickle', 'wb') as fp:

                    save_dic = {'model':IOHMM_MODEL,'data':data, 'scaler_x':scaler_x, 'scaler_y':scaler_y}
                    pickle.dump(save_dic, fp)

            result_save_train = data_save[best_idx][0]
            result_save_test = data_save[best_idx][1]
            result_save_train['l2_penalty'] = OLS_l2_penalty[best_idx]
            result_save_test['l2_penalty'] = OLS_l2_penalty[best_idx]

        else:
            Accuracy_test, Accuracy_train, result_df_train, result_df_test, RMSE, MAPE, MAE,  \
            R_sq_test, R_sq_train, IOHMM_MODEL,data, scaler_y, scaler_x = Model(Card_ID, -1, -1, FEATURE_SELECTION, SCALAR_DURATION, SAVE_MODEL = SAVE_MODEL)
            best_idx = 0
            acc_list = [Accuracy_test]
            acc_list_train = [Accuracy_train]
            RMSE_list = [RMSE]
            R_sq_list = [R_sq_test]
            R_sq_list_train = [R_sq_train]
            result_save_train = result_df_train
            result_save_test = result_df_test

            R_sq_first, R_sq_middle, _ = data_process_continuous_R_sq(result_df_train)
            R_sq_first_train = [R_sq_first]
            R_sq_middle_train = [R_sq_middle]
            R_sq_first, R_sq_middle, _ = data_process_continuous_R_sq(result_df_test)
            R_sq_first_test =  [R_sq_first]
            R_sq_middle_test = [R_sq_middle]
            result_save_train['l2_penalty'] = -1
            result_save_test['l2_penalty'] = -1 # -1 means default values

            if SAVE_MODEL:
                save_path = 'output/IOHMM/'
                with open(save_path + 'Trained_IOHMM_'+ str(Card_ID) + '.pickle', 'wb') as fp:
                    save_dic = {'model':IOHMM_MODEL,'data':data, 'scaler_x':scaler_x, 'scaler_y':scaler_y}
                    pickle.dump(save_dic, fp)

        if COMPARE_BASELINE:
            result_df_MC_location = pd.read_csv(data_path + 'results/result_Location_MC' + str(Card_ID) + '.csv')
            _, _, Accuracy_MC_loc, _, _, _,_, _, _, _ = calculate_accuracy(result_df_MC_location, task = 'loc')

            result_df_MC_location = pd.read_csv(data_path + 'results/result_Location_LSTM' + str(Card_ID) + 'test.csv')
            _, _, Accuracy_LSTM_loc, _, _, _,_, _, _, _ = calculate_accuracy(result_df_MC_location, task = 'loc')

            result_df_LR_duration = pd.read_csv(data_path + 'results/result_LR' + str(Card_ID) + 'test.csv')
            RMSE_LR, MAPE_LR, MAE_LR, R_sq_LR  = calculate_error(result_df_LR_duration)

            R_sq_first_LR, R_sq_middle_LR, _ = data_process_continuous_R_sq(result_df_LR_duration)


            result_df_LSMT_duration = pd.read_csv(data_path + 'results/result_LSTM_con_dur' + str(Card_ID) + 'test.csv')
            RMSE_LSTM, MAPE_LSTM, MAE_LSTM, R_sq_LSTM  = calculate_error(result_df_LSMT_duration)

            R_sq_first_LSTM, R_sq_middle_LSTM, _ = data_process_continuous_R_sq(result_df_LSMT_duration)

        else:
            Accuracy_MC_loc = -1
            Accuracy_LSTM_loc = -1
            R_sq_LR = -1
            R_sq_LSTM = -1


        print ('Num_people_processed', count, 'Total', len(individual_ID_list_test))
        print(Card_ID, 'Total Training Accuracy location:', acc_list_train[best_idx])
        print(Card_ID, 'IOHMM Loc:', acc_list[best_idx], 'MC Loc:', Accuracy_MC_loc,'LSTM Loc:', Accuracy_LSTM_loc)
        print(Card_ID, 'Total Training R_sq:', R_sq_list_train[best_idx])
        print(Card_ID, 'IOHMM Dur:', R_sq_list[best_idx],'LR Dur:', R_sq_LR, 'LSTM Dur',R_sq_LSTM)
        result_save_train.to_csv(file_name.replace('.csv', 'train.csv'), index=False)
        result_save_test.to_csv(file_name.replace('.csv', 'test.csv'), index=False)
        print('------****------')
    #pool = multiprocessing.Pool(processes=3)
    #pool.map(Model, individual_ID_list_test)
    #pool.close()
    #print ('Accurate_duration',sum(Accurate_duration)/len(Accurate_duration))

    #filename1='data/activity_index_test.txt'
    #file1=open(filename1,'r')
    #activity_index_test=eval(file1.read())