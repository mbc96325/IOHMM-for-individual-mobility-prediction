import pandas as pd

long_name = pd.read_csv('../data/Station_short_long_name.csv')
station_code = pd.read_excel('../data/Line_Station_Code.xlsx')
# a=1
num_station = len(station_code)
station_code = station_code.merge(long_name[['Short Name','Station Name']], left_on = ['STATION'], right_on = ['Short Name'])
num_station_after = len(station_code)

station_code = station_code.drop_duplicates(['STATION','CODE','Station Name'])
station_code = station_code.reset_index(drop=True)
num_station_final = len(station_code)
station_code = station_code.drop(columns = ['Short Name'])


### add station
data_added = {'LINE':['ISL','ISL','ISL'],'STATION':['SYP','HKU','KDT'],'CODE':[81,82,83],'Station Name':['Sai Ying Pun', 'HKU','Kennedy Town']}

station_code = pd.concat([station_code,pd.DataFrame(data_added)])
station_code = station_code.to_csv('../data/station_id_name_processed.csv',index=False)