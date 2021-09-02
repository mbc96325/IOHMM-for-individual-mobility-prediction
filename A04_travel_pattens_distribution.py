# -*- coding: utf-8 -*-
import matplotlib
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
from itertools import groupby

import copy
import os

'''
delete samples with less than 200 active days 

'''
#with open ('data/individual_ID_list', 'rb') as fp:
    #individual_ID_list = pickle.load(fp)
##------------****filter out data****------
#individual_ID_list_copy=copy.deepcopy(individual_ID_list)
#for Card_ID in individual_ID_list:
    #file_name='data/sample_'+str(Card_ID)+'_201407_201408.csv'
    #sample_one=pd.read_csv(file_name)
    #sample_one['date_time']=pd.to_datetime(sample_one['date_time'])
    #sample_one["date"]=sample_one["date_time"].apply(lambda x: x.date())
    #date_list=pd.unique(sample_one['date'])
    ##print (date_list)
    #flag=0
    #if len(date_list) < 200:
        #individual_ID_list_copy.remove(Card_ID)
        #continue
    #for date in date_list:
        #trip_per_day=sample_one['date'].loc[sample_one['date']==date].count()-1
        #if trip_per_day>8: # unreasonable card
            #flag=1
            #break
    #if flag==1:
        #individual_ID_list_copy.remove(Card_ID)
    
#with open('data/individual_ID_list_new', 'wb') as fp:
    #pickle.dump(individual_ID_list_copy, fp)


    
colors = ["#3366cc", "#dc3912", "#109618", "#990099", "#ff9900"]
labels = ["All", "First", "Remaining"]
#------------**------
def process_data(Card_ID):
    #print (Card_ID)
    file_name='data/sample_'+str(Card_ID)+'_201407_201408.csv'
    if os.path.exists(file_name)==False:
        return 'nothing'
    #print ('a')
    sample_one=pd.read_csv(file_name)
    sample_one['date_time']=pd.to_datetime(sample_one['date_time'])
    sample_one["date"]=sample_one["date_time"].apply(lambda x: x.date())
    date_list=list(pd.unique(sample_one['date']))
    #print (date_list)
    num_of_Activeday=len(date_list)
    #print(num_of_Activeday)
    location_list=list(pd.unique(sample_one['location']))
    location_list.remove(-1)
    num_of_location=len(location_list)
    #print (num_of_location)
    return (num_of_Activeday,num_of_location)

def process_data_trip_perday(samples):
    samples = samples.loc[samples['if_last']== 0].copy()
    samples['date_time']=pd.to_datetime(samples['date_time'])
    samples["date"]=samples["date_time"].apply(lambda x: x.date())

    samples['duration_hour'] = samples['duration'] / 3600
    samples['trip_start_time'] = samples['date_time'] + pd.to_timedelta(samples['duration'],unit = 'sec')
    samples['trip_start_time_hour'] = samples["trip_start_time"].dt.hour + 4
    samples['trip_start_time_hour'] = samples['trip_start_time_hour'] % 24

    samples_trip_per_day = samples.groupby(['Card_ID', 'date'])['seq_ID'].count().reset_index()
    trip_per_day_list = list(samples_trip_per_day['seq_ID'])

    samples_first = samples.loc[samples['act_ID'] == 0]
    sample_not_first = samples.loc[samples['act_ID'] != 0]

    first_duration_list = list(samples_first['duration_hour'])
    duration_list = list(samples['duration_hour'])
    all_except_duration_list = list(sample_not_first['duration_hour'])
    trip_start_time = list(samples['trip_start_time_hour'])

    num_active_day_list = samples.groupby(['Card_ID'])['date'].nunique().reset_index()
    num_active_day_list = list(num_active_day_list['date'])

    return trip_per_day_list,first_duration_list, duration_list, all_except_duration_list,trip_start_time, num_active_day_list


def plot_data_decription(trip_per_day,active_day,df1,df2,df3,trip_start_time, save_fig):
    import seaborn as sns
    sns.set(font_scale=1.8)
    sns.set_style("whitegrid", {'axes.grid': False, "legend.frameon": True})
    plt.figure(figsize=(14, 10))
    
    
    ax1 = plt.subplot(2, 2, 1)
    hist, bins = np.histogram(active_day, range(300,850,50))
    p = hist.astype(np.float32) / len(active_day)
    w = bins[1] - bins[0]
    plt.bar(bins[:-1], p, width=w, align='edge', color=colors[0], edgecolor='w', alpha=0.8)
    # plt.hist(S, bins=range(0, 151, 10), normed=True, color=colors[0], edgecolor='w', alpha=0.8)
    plt.xlim(300, 800)
    # ax1.set_yticks([0.0, 0.005, 0.010, 0.015, 0.020])
    plt.xlabel('Number of active days', fontsize=18)
    plt.ylabel('Probability', fontsize=18)
    plt.text(-0.1, 1.05, '(a)', fontdict={'size': 18, 'weight': 'bold'},
        transform=ax1.transAxes)      
    
    ax2 = plt.subplot(2, 2, 2)
    n = 8 + 1
    plt.hist(np.array(trip_per_day), bins=range(n) , density = True, color=colors[0], edgecolor='w', alpha=0.8)
    plt.xlim(1, n)
    plt.xlabel('Number of trips per active day', fontsize=18)
    plt.ylabel('Probability', fontsize=18)
    plt.xticks([i + 0.5 for i in range(1, n)], range(1, n))
    plt.text(-0.1, 1.05, '(b)', fontdict={'size': 18, 'weight': 'bold'},
        transform=ax2.transAxes)
    
    

    
    ax3 = plt.subplot(2, 2, 3)
    sns.kdeplot(df1[0][:], ax=ax3, shade=True, color=colors[0], label=labels[0])
    sns.kdeplot(df2[0][:], ax=ax3, shade=True, color=colors[1], label=labels[1])
    sns.kdeplot(df3[0][:], ax=ax3, shade=True, color=colors[2], label=labels[2])

    meda = df1[0][:].mean()
    medb = df2[0][:].mean()
    medc = df3[0][:].mean()

    #plt.axvline(meda, color=colors[0], linestyle='dashed', linewidth=2)
    #plt.axvline(medb, color=colors[1], linestyle='dashed', linewidth=2)
    #plt.axvline(medc, color=colors[2], linestyle='dashed', linewidth=2)

    #plt.text(meda + 0.2, 0.02, 'Mean = {}'.format(round(meda, 1)),
        #horizontalalignment='left', verticalalignment='center',
        #fontsize=16, color=colors[0])
    #plt.text(medb - 0.2, 0.02, 'Mean = {}'.format(round(medb, 1)),
        #horizontalalignment='right', verticalalignment='center',
        #fontsize=16, color=colors[1])
    #plt.text(medc - 0.2, 0.02, 'Mean = {}'.format(round(medb, 1)),
        #horizontalalignment='right', verticalalignment='center',
        #fontsize=16, color=colors[2])    
 

    plt.xlim(0, 24)
    #plt.xlim(0, 3.5)
    #ax.set_xticklabels([str(i) + '%' for i in range(0, 101, 20)])
    plt.xticks(range(0, 24 + 4, 4))
    plt.xlabel('Activity duration (hours)', fontsize=18)
    plt.ylabel('Probability', fontsize=18)
    plt.legend(fontsize=18, loc='upper right')
    
    
    plt.text(-0.1, 1.05, '(c)', fontdict={'size': 18, 'weight': 'bold'},
        transform=ax3.transAxes)
    #plt.show()
    ax4 = plt.subplot(2, 2, 4)
    
    #hist, bins = np.histogram(trip_start_time, range(0,24,1))
    #p = hist.astype(np.float32) / len(trip_start_time)
    #w = bins[1] - bins[0]
    #print (bins)
    #plt.bar(bins[:-1], p, width=w, align='edge', color=colors[0], edgecolor='w', alpha=0.8)
    unique, counts = np.unique(trip_start_time, return_counts=True)
    prob = counts / sum(counts)
    w = unique[1] - unique[0]
    plt.bar(unique, prob, width=w, align='edge', color=colors[0], edgecolor='w', alpha=0.8)

    ax4.set_xlim(0, 23)
    unit_label = 4
    # ax1.set_yticks([0.0, 0.005, 0.010, 0.015, 0.020])
    plt.xticks(np.arange(0, 23 + unit_label, unit_label) + 0.5, tuple([str(i) + '' for i in range(0, 23 + unit_label, unit_label)]))
    #ax4.set_xticklabels([str(i) + ':00' for i in range(0, 30, 6)])
    plt.xlabel('Trip start time', fontsize=18)
    plt.ylabel('Probability', fontsize=18)
    plt.text(-0.1, 1.05, '(d)', fontdict={'size': 18, 'weight': 'bold'},
        transform=ax4.transAxes)
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/travel_patterns.png', dpi=200)
    else:
        plt.show()


if __name__ == '__main__':

    # with open('data/individual_ID_list_new', 'rb') as fp:
    #     individual_ID_list = pickle.load(fp)

    file_name_all = '../data/samples/sample_500_all_201407_201408.csv'
    if not os.path.exists(file_name_all):
        data_path = '../data/'

        num_ind = 1000
        with open(data_path + 'individual_ID_list_test_' + str(num_ind) + '.pickle', 'rb') as fp:
            individual_ID_list_test = pickle.load(fp)
        individual_ID_list = individual_ID_list_test[0:500]

        samples = []
        count = 0
        for Card_ID in individual_ID_list:
            count+=1
            if count % 100 == 0:
                print('Current id', count, 'Total', len(individual_ID_list))
            file_name = '../data/samples/sample_' + str(Card_ID) + '_201407_201408.csv'
            if not os.path.exists(file_name):
                print(Card_ID, 'not available')
                continue
            data_temp = pd.read_csv(file_name)
            data_temp['Card_ID'] = Card_ID
            samples.append(data_temp)
        samples = pd.concat(samples)
        samples.to_csv(file_name_all, index=False)
    else:
        print('Just load data')
        samples = pd.read_csv(file_name_all)


    trip_per_day_list,first_duration_list, duration_list, all_except_duration_list,trip_start_time, num_active_day_list = process_data_trip_perday(samples)


    
    plot_data_decription(trip_per_day_list,num_active_day_list, pd.DataFrame(duration_list,columns=None),pd.DataFrame(first_duration_list,columns=None),
                         pd.DataFrame(all_except_duration_list,columns=None), trip_start_time, save_fig = 1)
    

      
    