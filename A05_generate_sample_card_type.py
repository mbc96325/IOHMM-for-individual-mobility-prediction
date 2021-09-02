import pandas as pd
import numpy as np
import copy
import pickle
from datetime import datetime,timedelta
import os
import multiprocessing
import random


data_path = '../data/'


TEST = True


if TEST:
    df = pd.read_csv(data_path + '201407_new.csv',sep = ';')
else:
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

    df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,
                    df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,
                    df21,df22,df23,df24,df25,df26,df27]) # consider 24 month



df = df.loc[:,['csc_phy_id', 'txn_subtype_co']].drop_duplicates()
df = df.drop_duplicates(['csc_phy_id'])

df.to_csv(data_path + 'sample_card_type.csv',index=False)
    # process_data(935962579)

print(pd.unique(df['txn_subtype_co']))