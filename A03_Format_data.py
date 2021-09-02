# -*- coding: utf-8 -*-
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
from itertools import groupby
import multiprocessing
import copy
import os


'''
convert data to each separated day: with trip or without trip

'''
#with open ('data/individual_ID_list_new', 'rb') as fp:
    #individual_ID_list = pickle.load(fp)

#num_of_test_samples=400
#individual_ID_list_test = [individual_ID_list[i] for i in sorted(random.sample(range(len(individual_ID_list)), num_of_test_samples))]
#with open('data/individual_ID_list_test', 'wb') as fp:
    #pickle.dump(individual_ID_list_test, fp)
###print (holiday_info.ix[0,0])
#---------***-------

# with open (data_path + 'individual_ID_list_test', 'rb') as fp:
#     individual_ID_list_test = pickle.load(fp)

def process_data(Card_ID, file_name, weather_info, holiday_info):
    print (Card_ID)

    sample_one=pd.read_csv(file_name)
    
    #print (sample_one.head(10))
    data={}
    location_list=list(pd.unique(sample_one['location']))
    location_list.remove(-1)
    location_fre={}
    
    for locat in location_list:
        location_fre[locat]=sample_one['location'].loc[sample_one['location']==locat].count()
    location_fre = sorted(location_fre.items(), key=lambda kv: kv[1], reverse=True)
    count=0
    total=sum(np.array(location_fre)[:,1])
    #print (total)
    location_final=[]
    for item in location_fre:
        count = count+item[1]
        #print (count)
        if count/total < 0.8: # remain locations with more than 80% trips
            location_final.append(item[0])
        else:
            location_final.append(item[0])
            break
    #print (location_final)
    
    location_list_o=list(pd.unique(sample_one['location_o']))
    location_list_o.remove(-1)
    location_fre={}
    
    for locat in location_list_o:
        location_fre[locat]=sample_one['location_o'].loc[sample_one['location_o']==locat].count()
    location_fre = sorted(location_fre.items(), key=lambda kv: kv[1], reverse=True)
    count=0
    total=sum(np.array(location_fre)[:,1])
    #print (total)
    location_final_o=[]
    for item in location_fre:
        count = count+item[1]
        #print (count)
        if count/total < 0.8: # remain locations with more than 80% trips
            location_final_o.append(item[0])
        else:
            location_final_o.append(item[0])
            break    
    
    #plt.figure()
    #sample_one['duration'].hist(bins=10)
    #plt.show()
    #print (location_list)
    data['ID']=[]
    for locat in location_final:
        key='location_'+str(locat)
        data[key]=[]
    data['ID']=[]
    for locat in location_final_o:
        key='location_o'+str(locat)
        data[key]=[]    
    data['Monday']=[]
    data['Tuesday']=[]
    data['Wednesday']=[]
    data['Thursday']=[]
    data['Friday']=[]
    data['Saturday']=[]
    data['Sunday']=[]
    data['rain']=[]
    data['heavy_rain']=[]

    data['sun']=[]
    data['cloud']=[]
    data['Max_Temp']=[]
    data['Min_Temp']=[]
    data['fengli']=[]
    data['Avrg_Temp']=[]
    data['if_last']=[]
    data['seq_ID']=[]
    data['act_ID']=[]
    data['location']=[]
    data['location_o']=[]
    sample_one['date_time_nostr'] = pd.to_datetime(sample_one['date_time'])
    hour_list=list(pd.unique(sample_one['date_time'].apply(lambda x: x.split(' ')[1].split(':')[0])))
    hour_list_pd=sample_one['date_time'].apply(lambda x: x.split(' ')[1].split(':')[0])
    #print (hour_list_pd)
    hour_list.remove('00')
    hour_list_fre={}
    for hour in hour_list:
        hour_list_fre[hour]=hour_list_pd.loc[hour_list_pd==hour].count()
    hour_list_fre = sorted(hour_list_fre.items(), key=lambda kv: kv[1], reverse=True)
    count=0
    total=0
    for item in hour_list_fre:
        total+=item[1]
    hour_final=[]
    for item in hour_list_fre:
        count = count+item[1]
        #print (count)
        if count/total < 0.8: # remain hours with more than 80% trips
            hour_final.append(item[0])
        else:
            hour_final.append(item[0])
            break
    hour_final.append('00') #used for indicate the first activity
    #print (hour_final)    
    for hour in hour_final:
        key='hour_'+hour
        data[key]=[]
    data['duration']=[]
    data['National_holiday']=[]
    data['Observance']=[]
    data['date']=[]
    data['duration_trip']=[]
    data['duration_last']=[]
    data['date_time'] = []
    data['hour'] = []
    data['N_days_withtrip_past20'] = []
    data['N_consec_days_no_trips'] = []
    data['N_trips_yesterday'] = []
    data['Last_trip_time_yesterday'] = []
    data['Next_tapin_station'] = []
    day_list=[]
    used_seq_ID=[]
    seq_ID=0
    count = 0
    for row in sample_one.iterrows():
        if row[1]['seq_ID'] not in used_seq_ID:
            last_duration = -1
            seq_ID+=1
            used_seq_ID.append(row[1]['seq_ID'])

        if row[1]['if_last'] != 1:
            data['Next_tapin_station'].append(sample_one.loc[row[0]+1,'location_o'])
        else:
            data['Next_tapin_station'].append(-1)

        today = row[1]['date_time_nostr'].date()
        last_20_date = today - pd.Timedelta('20 days')
        records_last_20 = sample_one.loc[(sample_one['date_time_nostr'] >= last_20_date)&
        (sample_one['date_time_nostr'] < today)]
        N_days_withtrip_past20 = len(pd.unique(records_last_20['seq_ID']))
        data['N_days_withtrip_past20'].append(N_days_withtrip_past20)

        last_day_with_trip = sample_one.loc[sample_one['seq_ID']==seq_ID-1]
        if len(last_day_with_trip)>0:
            date_last_trip = last_day_with_trip.iloc[0].loc['date_time_nostr'].date()
            N_consec_days_no_trips = today - date_last_trip
            N_consec_days_no_trips = int(N_consec_days_no_trips.days) - 1
            data['N_consec_days_no_trips'].append(N_consec_days_no_trips)

        else:
            data['N_consec_days_no_trips'].append(0)


        yesterday = today - pd.Timedelta('1 days')
        yesterday_records = sample_one.loc[(sample_one['date_time_nostr'] >= yesterday)&(sample_one['date_time_nostr'] < today)]
        if len(yesterday_records) >0:
            data['N_trips_yesterday'].append(len(yesterday_records) - 1) # -1 because the last trip is a dummy trip
            data['Last_trip_time_yesterday'].append(int(yesterday_records.iloc[-1].loc['date_time_nostr'].hour))
        else:
            data['N_trips_yesterday'].append(0)
            data['Last_trip_time_yesterday'].append(0)

        ID=row[0]+1
        duration = row[1]['duration']
        data['date_time'].append(row[1]['date_time'])
        date_time = row[1]['date_time']
        location = row[1]['location']
        data['location'].append(location)
        data['location_o'].append(row[1]['location_o'])
        data['if_last'].append(row[1]['if_last'])
        data['duration_trip'].append(row[1]['duration_trip'])
        #print (location)
        date_str=date_time.split(' ')[0]
        year=int(date_time.split(' ')[0].split('-')[0])
        month=int(date_time.split(' ')[0].split('-')[1])
        day=int(date_time.split(' ')[0].split('-')[2])
        hh=date_time.split(' ')[1].split(':')[0]
        mm=int(date_time.split(' ')[1].split(':')[1])
        ss=int(date_time.split(' ')[1].split(':')[2])
        date_time_dt=datetime.datetime(year,month,day,int(hh),mm,ss)
        data['date'].append(date_time.split(' ')[0])
        data['hour'].append(hh)
        if date_time.split(' ')[0] not in day_list:
            day_list.append(date_time.split(' ')[0])
        weekday=date_time_dt.weekday() #0 for Mon 6 for Sun
        #print (date_time_dt)
        #print (weekday)
        data['duration'].append(duration)
        data['duration_last'].append(last_duration)
        last_duration = duration
        data['ID'].append(ID)
        data['seq_ID'].append(seq_ID)
        data['act_ID'].append(row[1]['act_ID'])
        for locat in location_final:
            key='location_'+str(locat)
            if locat==location:
                data[key].append(1)
            else:
                data[key].append(0)
        for locat in location_final_o:
            key='location_o'+str(locat)
            if locat==location:
                data[key].append(1)
            else:
                data[key].append(0)        
        if weekday==0:
            data['Monday'].append(1)
        else:
            data['Monday'].append(0)
        if weekday==1:
            data['Tuesday'].append(1)
        else:
            data['Tuesday'].append(0)
        if weekday==2:
            data['Wednesday'].append(1)
        else:
            data['Wednesday'].append(0)
        if weekday==3:
            data['Thursday'].append(1)
        else:
            data['Thursday'].append(0)
        if weekday==4:
            data['Friday'].append(1)
        else:
            data['Friday'].append(0)
        if weekday==5:
            data['Saturday'].append(1)
        else:
            data['Saturday'].append(0)
        if weekday==6:
            data['Sunday'].append(1)
        else:
            data['Sunday'].append(0)
        
        if holiday_info['Holiday_Date'].str.contains(date_time.split(' ')[0]).any():
            row_holiday=holiday_info.loc[holiday_info['Holiday_Date']==date_time.split(' ')[0]].to_dict(orient='list')
            #print (row_holiday)
            if row_holiday['Holiday_Type'][0]=='Observance':
                data['National_holiday'].append(0)
                data['Observance'].append(1)
            elif row_holiday['Holiday_Type'][0]=='National holiday':
                data['National_holiday'].append(1)
                data['Observance'].append(0)                
            else:
                data['National_holiday'].append(0)
                data['Observance'].append(0)  
        else:
            data['National_holiday'].append(0)
            data['Observance'].append(0) 
        #print (weather_info.head())
        #print (type(weather_info['ymd'][0]))
        row_weather=weather_info.loc[weather_info['ymd']==date_str].to_dict(orient='list')
        #print (row_weather)
        weather=row_weather['tianqi'][0]
        if 'rain' in weather:
            data['rain'].append(1)
        else:
            data['rain'].append(0)
        if 'heavy_rain' in weather:
            data['heavy_rain'].append(1)
        else:
            data['heavy_rain'].append(0)  
        if 'cloud' in weather:
            data['cloud'].append(1)
        else:
            data['cloud'].append(0)
        if 'sun' in weather:
            data['sun'].append(1)
        else:
            data['sun'].append(0)
        data['Max_Temp'].append(row_weather['Max_Temp'][0])
        data['Min_Temp'].append(row_weather['Min_Temp'][0])
        data['Avrg_Temp'].append((row_weather['Min_Temp'][0]+row_weather['Max_Temp'][0])/2)
        fengli_list=row_weather['fengli'][0].split('~')
        if len(fengli_list)>1:
            fengli_float=(float(fengli_list[0])+float(fengli_list[1]))/2.0
        else:
            fengli_float=float(fengli_list[0])
        data['fengli'].append(fengli_float)
        '''
        time of a day
        '''
        for hour in hour_final:
            key='hour_'+hour
            if hh==hour:
                data[key].append(1)
            else:
                data[key].append(0)
        
    data_df=pd.DataFrame(data)
    file_name = data_path + 'samples/sample_'+str(Card_ID)+'_201407_201408_all.csv'
    data_df.to_csv(file_name,index=False)


    # num_of_test_days=30
    # V_date_list = [ day_list[i] for i in sorted(random.sample(range(len(day_list)), num_of_test_days)) ]
    # #V_date_list = day_list[-num_of_test_days:]
    # T_date_list = list(set(day_list)^set(V_date_list))
    # #T_date_list = day_list
    # validate_data=data_df.loc[data_df['date'].isin(V_date_list)]
    # train_data=data_df.loc[data_df['date'].isin(T_date_list)]
    # #T_date_list = day_list
    # #print (validate_data)
    # file_name_train=data_path +'samples/sample_'+str(Card_ID)+'_201407_201408_train.csv'
    # train_data.to_csv(file_name_train,index=False)
    # file_name_validate=data_path + 'samples/sample_'+str(Card_ID)+'_201407_201408_validate.csv'
    # validate_data.to_csv(file_name_validate,index=False)
if __name__ == '__main__':
    # pool = multiprocessing.Pool(processes=4)
    # pool.map(process_data, individual_ID_list_test) #
    # pool.close()
    data_path = '../data/'
    RE_GENERATE = True

    num_ind = 1000
    with open(data_path + 'individual_ID_list_test_' + str(num_ind) + '.pickle', 'rb') as fp:
        individual_ID_list_test = pickle.load(fp)

    weather_info = pd.read_csv(data_path + 'weather_201407_20151_HK.csv')
    holiday_info = pd.read_csv(data_path + 'Holiday_2014_2016_HK.csv')

    count = 0
    for Card_ID in individual_ID_list_test:
        count+=1
        print('Current num', count, 'Total', len(individual_ID_list_test))
        file_name_output = data_path + 'samples/sample_' + str(Card_ID) + '_201407_201408_all.csv'
        if RE_GENERATE:
            file_name_input = data_path + 'samples/sample_' + str(Card_ID) + '_201407_201408.csv'
            process_data(Card_ID, file_name_input, weather_info, holiday_info)
        else:
            if os.path.exists(file_name_output):
                print(Card_ID,' exist, skip it...')
                continue
            else:
                file_name_input = data_path + 'samples/sample_' + str(Card_ID) + '_201407_201408.csv'
                process_data(Card_ID, file_name_input, weather_info, holiday_info)