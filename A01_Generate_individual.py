import pandas as pd
import numpy as np
import copy
import pickle
from datetime import datetime,timedelta
import os
import multiprocessing
import random
data_path = '../data/'
df1 = pd.read_csv(data_path + '201407_new.csv',sep = ';')
df2 = pd.read_csv(data_path + '201408_new.csv',sep = ';')
df3 = pd.read_csv(data_path + '201409_new.csv',sep = ';')
df4 = pd.read_csv(data_path + '201410_new.csv',sep = ';')
df5 = pd.read_csv(data_path + '201411_new.csv',sep = ';')
df6 = pd.read_csv(data_path + '201412_new.csv',sep = ';')
df7 = pd.read_csv(data_path + '201501_new.csv',sep = ';')
df8 = pd.read_csv(data_path + '201502_new.csv',sep = ';')
df9 = pd.read_csv(data_path + '201503_new.csv',sep = ';')
df10 = pd.read_csv(data_path + '201504_new.csv',sep = ';')
df11 = pd.read_csv(data_path + '201505_new.csv',sep = ';')
df12 = pd.read_csv(data_path + '201506_new.csv',sep = ';')
df13 = pd.read_csv(data_path + '201507_new.csv',sep = ';')
df14 = pd.read_csv(data_path + '201508_new.csv',sep = ';')
df15 = pd.read_csv(data_path + '201509_new.csv',sep = ';')
df16 = pd.read_csv(data_path + '201510_new.csv',sep = ';')
df17 = pd.read_csv(data_path + '201511_new.csv',sep = ';')
df18 = pd.read_csv(data_path + '201512_new.csv',sep = ';')
df19 = pd.read_csv(data_path + '201601_new.csv',sep = ';')
df20 = pd.read_csv(data_path + '201602_new.csv',sep = ';')
df21 = pd.read_csv(data_path + '201603_new.csv',sep = ';')
df22 = pd.read_csv(data_path + '201604_new.csv',sep = ';')
df23 = pd.read_csv(data_path + '201605_new.csv',sep = ';')
df24 = pd.read_csv(data_path + '201606_new.csv',sep = ';')
df25 = pd.read_csv(data_path + '201607_new.csv',sep = ';')
df26 = pd.read_csv(data_path + '201608_new.csv',sep = ';')
df27 = pd.read_csv(data_path + '201609_new.csv',sep = ';')
#df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,
                #df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,
                #df21,df22,df23,df24]) # consider 4 month
df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,
                df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,
                df21,df22,df23,df24,df25,df26,df27]) # consider 24 month                
#df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,
                #df11,df12]) # consider 4 month
individual_ID_list=list(df['csc_phy_id'].unique())
with open(data_path+'individual_ID_list', 'wb') as fp:
    pickle.dump(individual_ID_list, fp)
"""
test 
"""
# num_of_test_samples=100
# random.seed(1)
# individual_ID_list_test = [individual_ID_list[i] for i in sorted(random.sample(range(len(individual_ID_list)), num_of_test_samples))]
#
# with open(data_path + 'individual_ID_list_test', 'wb') as fp:
#     pickle.dump(individual_ID_list_test, fp)
#
# with open (data_path + 'individual_ID_list_test', 'rb') as fp:
#     individual_ID_list = pickle.load(fp)
"""
======================================test end
"""

########################

with open ('../data/individual_ID_list', 'rb') as fp:
    individual_ID_list = pickle.load(fp)

#########################





minimum_act = 1200    
def process_data(ID, not_recommend_samples):
    print (ID)
    #if os.path.exists('data/sample_'+str(ID)+'_201407_201408.csv'):
        #continue
    sample_one=copy.deepcopy(df.loc[df['csc_phy_id'] == ID])
    sample_one=sample_one.reset_index()
    sample_one.ix[:,'txn_dt']=pd.to_datetime(sample_one.ix[:,'txn_dt'])-timedelta(hours=4) # set 4:00 AM as the start time of one day
    sample_one=sample_one.sort_values(by=['txn_dt'])
    sample_one=sample_one.reset_index()
    sample_one.loc[:,'new_date'] = sample_one['txn_dt'].apply(lambda x: x.date())
    date_have_trip=sample_one['new_date'].unique()
    #print (date_have_trip)
    #print (sample_one.columns.values)
    count=0
    data={}
    data['duration']=[]
    data['location']=[]
    data['location_o']=[]
    data['duration_trip']=[]
    #data['duration_last']=[]
    data['date_time']=[]
    data['if_last']=[]
    data['error_seq']=[]  
    data['seq_ID']=[]  
    data['act_ID']=[]  
    seq_ID=0
    act_ID=0    
    #print (sample_one)
    for day in date_have_trip:
        #day = datetime(day)
        #print (type(day))
        index = sample_one.index[sample_one['new_date'] == day].tolist()
        count=0
        seq_ID+=1
        act_ID=0

        if len(index)%2 != 0: #must be error day
            data['duration'].append(-1)
            data['location'].append(-1)
            data['location_o'].append(-1)
            data['duration_trip'].append(-1)
            data['date_time'].append(-1)
            data['if_last'].append(-1)
            data['seq_ID'].append(seq_ID)
            data['act_ID'].append(-1)
            data['error_seq'].append(1)
            print('error record')
            continue

        for i in range(len(index)):
            index_int=index[i]
            if i==0: #first activity
                if sample_one.ix[index_int, 'txn_type_co'] == 'ENT':
                    activity_first=(sample_one.ix[index_int, 'txn_dt']-datetime.combine(day, datetime.min.time())).total_seconds()
                    if activity_first > minimum_act: #not transfer
                        time_out = datetime.combine(day, datetime.min.time())
                        location = -1
                        data['duration'].append(activity_first)
                        #data['duration_last'].append(0)
                        data['duration_trip'].append(0)
                        data['location'].append(location)
                        data['location_o'].append(-1)
                        data['date_time'].append(time_out)
                        data['if_last'].append(0)
                        data['seq_ID'].append(seq_ID)
                        data['act_ID'].append(act_ID) 
                        data['error_seq'].append(0)                     
                    time_out=None
                    location=None
                else:
                    data['duration'].append(-1)
                    #data['duration_last'].append(-1)
                    data['duration_trip'].append(-1)
                    data['location'].append(-1)
                    data['location_o'].append(-1)
                    data['date_time'].append(-1)
                    data['if_last'].append(-1)
                    data['seq_ID'].append(seq_ID)
                    data['act_ID'].append(-1)  
                    data['error_seq'].append(1)                        
                    print ('error record')
                    break
                    #activity_first=(sample_one.ix[index_int, 'txn_dt']-datetime.combine(day, datetime.min.time())).total_seconds()
                    #time_out = datetime.combine(day, datetime.min.time())
                    #location = -1
                    #data['duration'].append(activity_first)
                    #data['location'].append(location)
                    #data['date_time'].append(time_out)
                    #data['if_last'].append(0)
                    #time_out=sample_one.ix[index_int, 'txn_dt']
                    #location=sample_one.ix[index_int, 'txn_loc']  
            elif i==len(index)-1: #last activity  
                if sample_one.ix[index_int, 'txn_type_co'] == 'USE':
                    if time_out is not None: # USE USE
                        print ('error record')
                        data['duration'].append(-1)
                        #data['duration_last'].append(-1)
                        data['location'].append(-1)
                        data['location_o'].append(-1)
                        data['duration_trip'].append(-1)
                        data['date_time'].append(-1)
                        data['if_last'].append(-1)
                        data['seq_ID'].append(seq_ID)
                        data['act_ID'].append(-1)  
                        data['error_seq'].append(1)                           
                        break
                        ## use unused information 
                        #time_activity=(sample_one.ix[index_int, 'txn_dt']-time_out).total_seconds()
                        #data['duration'].append(time_activity)
                        #data['location'].append(location)
                        #data['date_time'].append(time_out)
                        #data['if_last'].append(0)
                        #time_out=sample_one.ix[index_int, 'txn_dt']
                        #location=sample_one.ix[index_int, 'txn_loc']
                        ##final
                        #time_activity=(datetime.combine(day, datetime.max.time())-time_out).total_seconds()
                        #data['duration'].append(time_activity)
                        #data['location'].append(location)
                        #data['date_time'].append(time_out)
                        #data['if_last'].append(1)                        
                    else: # ENT USE 
                        time_out=sample_one.ix[index_int, 'txn_dt']
                        location = sample_one.ix[index_int, 'txn_loc']
                        location_o = sample_one.ix[index_int-1, 'txn_loc']
                        time_in = sample_one.ix[index_int-1, 'txn_dt']
                        time_activity=(datetime.combine(day, datetime.max.time())-time_out).total_seconds()
                        if time_activity > minimum_act: #not transfer
                            data['duration'].append(time_activity)
                            #data['duration_last'].append(-1)
                            data['location'].append(location)
                            data['location_o'].append(location_o)
                            data['duration_trip'].append((time_out - time_in).total_seconds())
                            data['date_time'].append(time_out)
                            data['if_last'].append(1)  
                            data['seq_ID'].append(seq_ID)
                            act_ID+=1
                            data['act_ID'].append(act_ID) 
                            data['error_seq'].append(0)                          
                else:
                    print ('error record')
                    data['duration'].append(-1)
                    #data['duration_last'].append(-1)
                    data['location'].append(-1)
                    data['location_o'].append(-1)
                    data['duration_trip'].append(-1)
                    data['date_time'].append(-1)
                    data['if_last'].append(-1)
                    data['seq_ID'].append(seq_ID)
                    data['act_ID'].append(-1)  
                    data['error_seq'].append(1)                           
                    break                    
                    #if time_out is None: # ENT ENT 
                        #print ('error record')
                        #data['duration'].append(-1)
                        #data['location'].append(-1)
                        #data['location_o'].append(-1)
                        #data['date_time'].append(-1)
                        #data['if_last'].append(-1)
                        #data['seq_ID'].append(seq_ID)
                        #data['act_ID'].append(-1)  
                        #data['error_seq'].append(1)                           
                        #break
                        ##time_out=sample_one.ix[index_int-1,'txn_dt']
                        ##location = sample_one.ix[index_int-1,'txn_loc']
                        ##time_activity=(sample_one.ix[index_int, 'txn_dt']-time_out).total_seconds()
                        ##data['duration'].append(time_activity)
                        ##data['location'].append(location)
                        ##data['date_time'].append(time_out)
                        ##data['if_last'].append(0)
                        ##time_out=None
                        ##location=None
                        ### final
                        ##time_out=sample_one.ix[index_int, 'txn_dt']
                        ##location = sample_one.ix[index_int, 'txn_loc']
                        ##time_activity=(datetime.combine(day, datetime.max.time())-time_out).total_seconds()
                        ##data['duration'].append(time_activity)
                        ##data['location'].append(location)
                        ##data['date_time'].append(time_out)
                        ##data['if_last'].append(1)
                        ##time_out=None
                        ##location=None                        
                    #else: # USE ENT 
                        #time_activity=(sample_one.ix[index_int, 'txn_dt']-time_out).total_seconds()
                        #if activity_first > minimum_act: #not transfer
                            #data['duration'].append(time_activity)
                            #data['location'].append(location)
                            #data['location_o'].append(location_o)
                            #data['date_time'].append(time_out)
                            #data['if_last'].append(0)
                            #data['seq_ID'].append(seq_ID)
                            #act_ID+=1
                            #data['act_ID'].append(act_ID) 
                            #data['error_seq'].append(0)                       
                        #time_out=None
                        #location=None
                        ## final
                        #time_out=sample_one.ix[index_int, 'txn_dt']
                        #location = sample_one.ix[index_int, 'txn_loc']
                        #time_activity=(datetime.combine(day, datetime.max.time())-time_out).total_seconds()
                        #if activity_first > minimum_act: #not transfer
                            #data['duration'].append(time_activity)
                            #data['location'].append(location)
                            #data['date_time'].append(time_out)
                            #data['if_last'].append(1)
                            #data['seq_ID'].append(seq_ID)
                            #act_ID+=1
                            #data['act_ID'].append(act_ID) 
                            #data['error_seq'].append(0)                            
                        #time_out=None
                        #location=None                        
            else: #middle actitvi
                if sample_one.ix[index_int, 'txn_type_co'] == 'USE':
                    if time_out is not None: # USE USE
                        print ('error record')
                        data['duration'].append(-1)
                        data['location'].append(-1)
                        data['location_o'].append(-1)
                        data['duration_trip'].append(-1)
                        data['date_time'].append(-1)
                        data['if_last'].append(-1)
                        data['seq_ID'].append(seq_ID)
                        data['act_ID'].append(-1)  
                        data['error_seq'].append(1)                           
                        break
                        #time_activity=(sample_one.ix[index_int, 'txn_dt']-time_out).total_seconds()
                        #data['duration'].append(time_activity)
                        #data['location'].append(location)
                        #data['date_time'].append(time_out)
                        #data['if_last'].append(0)
                        #time_out=sample_one.ix[index_int, 'txn_dt']
                        #location=sample_one.ix[index_int, 'txn_loc']                        
                    else: # ENT USE 
                        time_out = sample_one.ix[index_int, 'txn_dt']
                        location = sample_one.ix[index_int, 'txn_loc']
                        location_o = sample_one.ix[index_int-1, 'txn_loc']
                        time_in = sample_one.ix[index_int-1, 'txn_dt']
                else:
                    if time_out is None: # ENT ENT 
                        print ('error record')
                        data['duration'].append(-1)
                        data['location'].append(-1)
                        data['location_o'].append(-1)
                        data['duration_trip'].append(-1)
                        data['date_time'].append(-1)
                        data['if_last'].append(-1)
                        data['seq_ID'].append(seq_ID)
                        data['act_ID'].append(-1)  
                        data['error_seq'].append(1)                           
                        break
                        #time_out=sample_one.ix[index_int-1,'txn_dt']
                        #location = sample_one.ix[index_int-1,'txn_loc']
                        #time_activity=(sample_one.ix[index_int, 'txn_dt']-time_out).total_seconds()
                        #data['duration'].append(time_activity)
                        #data['location'].append(location)
                        #data['date_time'].append(time_out)
                        #data['if_last'].append(0)
                        #time_out=None
                        #location=None
                    else: # USE ENT 
                        time_activity=(sample_one.ix[index_int, 'txn_dt']-time_out).total_seconds()
                        if time_activity > minimum_act: #not transfer
                            data['duration'].append(time_activity)
                            data['location'].append(location)
                            data['location_o'].append(location_o)
                            data['date_time'].append(time_out)
                            data['duration_trip'].append((time_out - time_in).total_seconds())
                            data['if_last'].append(0)
                            data['seq_ID'].append(seq_ID)
                            act_ID+=1
                            data['act_ID'].append(act_ID) 
                            data['error_seq'].append(0)                         
                        time_out=None
                        location=None                        
    data_df=pd.DataFrame(data)
    data_df_temp = data_df.loc[data_df['error_seq'] == 1]

    total_travel_days = len(pd.unique(data_df['seq_ID']))
    seq_ID_error = list(pd.unique(data_df_temp['seq_ID']))
    total_travel_days_error = len(seq_ID_error)

    if total_travel_days - total_travel_days_error <300:
        not_recommend_samples.append(Card_ID)
    if total_travel_days_error/total_travel_days > 0.05:
        if Card_ID not in not_recommend_samples:
            not_recommend_samples.append(Card_ID)

    data_df = data_df.loc[~data_df['seq_ID'].isin(seq_ID_error)]
    data_df.reset_index()
    #for i in range(len(seq_ID_new)):
        #data_df.loc[data_df['seq_ID'] == seq_ID_new[i]]['seq_ID'] = i+1
    file_name=data_path+'samples/sample_'+str(ID)+'_201407_201408.csv'
    data_df.to_csv(file_name,index=False)
    return not_recommend_samples

if __name__ == '__main__':
    # pool = multiprocessing.Pool(processes=1)
    # pool.map(process_data, individual_ID_list)
    # pool.close()

    not_recommend_samples = []
    for Card_ID in individual_ID_list:
        not_recommend_samples = process_data(Card_ID, not_recommend_samples)

    with open('../data/not_recommend_using_individual_ID.pickle', 'wb') as fp: # due to high error records rate
        pickle.dump(not_recommend_samples, fp)

    # process_data(935962579)