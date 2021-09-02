import pandas as pd
import numpy as np


def print_col_median(df):
	print 'Number of rows: {}'.format(df.shape[0])
	columns = list(df.columns.values)
	for col in columns:
		data = df[col][df[col] > -1e-4]
		if col[:2] == 'pp':
			print col, np.log2(data.median())
		else:
			print col, data.median()


df = pd.read_csv('../output/next_trip_prediction_baseline.csv')
print 'Baseline:'
print_col_median(df)

df2 = pd.read_csv('../output/next_trip_prediction.csv')
print 'N-Gram:'
print_col_median(df2)
