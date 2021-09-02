import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import multiprocessing
from datetime import datetime,timedelta
import os
import random
from numpy.random import choice
import math



def get_half_hourID(time):
    Time_sta = (time - datetime.combine(time.date(), datetime.min.time())).total_seconds()
    HourID = math.floor(Time_sta/3600)
    return HourID

def Init_matrix_cal(sample_one, dependent_var,Num_of_time_Seg, location_candidate):
    sample_first = sample_one.loc[sample_one['act_ID'] == 0]
    Num_day = sample_first['act_ID'].count()
    Init_Pro = np.zeros((Num_of_time_Seg,))
    Num_set = Init_Pro.shape[0]
    alpha = 1
    #print (pd.unique(sample_first['hour']))
    for i in range(Num_of_time_Seg):
        Trans_count = len(sample_first.loc[sample_first[dependent_var] == location_candidate[i]])
        Init_Pro[i] = (Trans_count + alpha/Num_set) / (Num_day + alpha)
    return Init_Pro
    
def Trans_matrix_cal(sample_one,dependent_var,Num_of_time_Seg, location_candidate):
    sample_one = sample_one.reset_index(drop=False)
    Trans_matrix = np.zeros((Num_of_time_Seg,Num_of_time_Seg))
    Trans_Pro = np.zeros((Num_of_time_Seg,Num_of_time_Seg))
    Num_set = Trans_matrix.shape[0]
    alpha = 1    
    for i in range(Num_of_time_Seg):
        for j in range(Num_of_time_Seg):
            temp_plus1 = list(sample_one[dependent_var].loc[(sample_one[dependent_var] == location_candidate[i])].index + 1) # trans from i
            max_value = max(sample_one.index.values) + 1
            if max_value in temp_plus1:
                temp_plus1.remove(max_value)
            #print (temp)
            #temp_plus1 = [k+1 for k in temp]
            #print (sample_one.ix[temp,'hour'] == i)
            # print(sample_one.shape)
            Trans_count = sum(sample_one.loc[temp_plus1,dependent_var] == location_candidate[j])
            if Trans_count == 0:
                Trans_matrix[i,j] = 0
            else:
                # only select data belong to same day!
                Trans_row = sample_one.loc[temp_plus1]
                Trans_row = Trans_row.loc[Trans_row.loc[:,dependent_var] == location_candidate[j]] # trans from i to j
                trans_before = sample_one.loc[Trans_row.index - 1,:]
                #print (Trans_row)
                Trans_row = Trans_row.reset_index()
                trans_before = trans_before.reset_index()
                Trans_count = sum(Trans_row.loc[:,'seq_ID'] == trans_before.loc[:,'seq_ID'])
                #print (trans_before)
                #print (Trans_count)
                Trans_matrix[i,j] = Trans_count
    for i in range(Num_of_time_Seg):
        Trans_Pro[i,:] = (Trans_matrix[i,:] + alpha/Num_set) / (sum(Trans_matrix[i,:]) + alpha)
    return Trans_Pro

def visualization(Trans_Pro,Init_Pro):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(Trans_Pro)
    fig.colorbar(cax)  
    plt.show()
    y_pos = range(Num_of_time_Seg)
    plt.bar(y_pos, height = np.array(Init_Pro), align='center', alpha=0.5)
    plt.show()



def process_data(data, test_proportion, test_last):
    data = data.reset_index(drop=True)
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
    return data, data_train, data_test

def Base_Model (Card_ID):
    file_name = data_path + 'results/result_Location_MC'+str(Card_ID)+'.csv'
    # if os.path.exists(file_name)==True:
    #     print ('Finish model', Card_ID)
    #     return

    dependent_variables = 'Next_tapin_station'

    print (Card_ID)
    file_name_train= data_path + 'samples/sample_'+str(Card_ID)+'_201407_201408_all.csv'
    data = pd.read_csv(file_name_train)
    data = data.loc[data['if_last']==0,:] # drop the last one, it will distract the training!

    location_candidate = list(pd.unique(data[dependent_variables]))
    Num_of_time_Seg = len(location_candidate)
    test_proportion = 0.2
    data['duration_hour'] = round(data['duration'] / 3600).astype('int')
    #----

    test_last = False
    data, data_train, data_test = process_data(data, test_proportion, test_last)


    #----****----training--****----
    Trans_Pro = Trans_matrix_cal(data_train,dependent_variables, Num_of_time_Seg, location_candidate)
    Init_Pro = Init_matrix_cal(data_train,dependent_variables,Num_of_time_Seg, location_candidate)
    #visualization(Trans_Pro, Init_Pro)
    #--****---- prediction --****----
    save_info_list1 = ['ID', 'Card_ID']
    save_predicted_rank = 20
    save_info_list2 = ['Predict'+str(i+1) for i in range(save_predicted_rank)]
    save_info_list3 = ['Ground_truth','Correct','activity_index','total_activity']
    save_info_list = save_info_list1 + save_info_list2 + save_info_list3
    #=========
    results={}
    for info in save_info_list:
        results[info] = []
    #=========
    model_validate = data_test.groupby(['seq_ID'])
    #print (last_hour)

    for idx, seq in model_validate:
        count = 0
        for _, row in seq.iterrows():
            count += 1
            if count == 1: # fist activity

                comb_results = [[Init_Pro[i], location_candidate[i]] for i in range(len(Init_Pro))]
                comb_results = sorted(comb_results, reverse=True)

                for i in range(save_predicted_rank):
                    if i >= len(comb_results):
                        rank_name = 'Predict' + str(i+1)
                        results[rank_name].append(-1) # no 20 candidates
                    else:
                        rank_name = 'Predict' + str(i+1)
                        results[rank_name].append(comb_results[i][1])
                predict = comb_results[0][1]
                ground_truth = int(row[dependent_variables])
                results['ID'].append(row['ID'])
                results['Card_ID'].append(Card_ID)

                results['Ground_truth'].append(ground_truth)
                if predict==ground_truth:
                    Correct = 1
                else:
                    Correct = 0
                results['Correct'].append(Correct)
                results['activity_index'].append(row['act_ID'])
                results['total_activity'].append(-1)
                last_index = location_candidate.index(ground_truth)

            if count > 1: #continue seq
                temp = Trans_Pro[last_index,:]
                #print (temp.shape)
                #print (np.where(temp==np.max(temp[:])))


                comb_results = [[temp[i], location_candidate[i]] for i in range(len(temp))]
                comb_results = sorted(comb_results, reverse=True)

                for i in range(save_predicted_rank):
                    if i >= len(comb_results):
                        rank_name = 'Predict' + str(i+1)
                        results[rank_name].append(-1) # no 20 candidates
                    else:
                        rank_name = 'Predict' + str(i+1)
                        results[rank_name].append(comb_results[i][1])
                predict = comb_results[0][1]
                ground_truth = int(row[dependent_variables])
                results['ID'].append(row['ID'])
                results['Card_ID'].append(Card_ID)

                results['Ground_truth'].append(ground_truth)
                if predict==ground_truth:
                    Correct = 1
                else:
                    Correct = 0
                results['Correct'].append(Correct)
                results['activity_index'].append(row['act_ID'])
                results['total_activity'].append(-1)
                last_index = location_candidate.index(ground_truth)


      
    result_df=pd.DataFrame(results)
    result_df.to_csv(file_name,index=False)  
    
    
if __name__ == '__main__':
    data_path = '../data/'


    # with open(data_path + 'individual_ID_list_test', 'rb') as fp:
    #     individual_ID_list_test = pickle.load(fp)


    num_ind = 1000
    with open(data_path + 'individual_ID_list_test_' + str(num_ind) + '.pickle', 'rb') as fp:
        individual_ID_list_test = pickle.load(fp)

    individual_ID_list_test = individual_ID_list_test[0:500]

    for Card_ID in individual_ID_list_test:
        Base_Model(Card_ID)
    #pool = multiprocessing.Pool(processes=5)
    #pool.map(Base_Model, individual_ID_list_test) 
    #pool.close()

        