import csv
import numpy as np
import random
import multiprocessing as mp
import time
import smartcard_trip_reader as smartcard
import ngram
import ngram_baseline
import os
import pandas as pd
import pickle

start = time.time()
random.seed(111)

trainDataFile = "../data/training_all_samples_hk.csv"
testDataFile = "../data/testing_all_samples_hk.csv"
vocabFile = "../data/station_vocab_hk.csv"

PART_INFO = True

with open('../data/USED_TOD_LIST.pickle', 'rb') as fp:
    USED_TOD_LIST = pickle.load(fp)


def load_stations(filepath):
	rd = csv.reader(open(filepath, 'r'), delimiter=",")
	vocab = []
	for s in rd:
		vocab.append(s[-1])
	return list(set(vocab))


def getStationVocab(filepath):
	return ngram.vocabulary(load_stations(filepath))


def getTimeVocab():
	time_list = [str(i) for i in range(0, 24)]
	return ngram.vocabulary(time_list)


def fitLinearBining(array, domain, num_bins):
	total = np.sum(array)
	linear_bins = np.linspace(domain[0], domain[1], num=num_bins)
	n, bins = np.histogram(array, linear_bins)
	Pkbin = np.array([])
	kbin = np.array([])
	for i in range(len(n)):
		kbin = np.append(kbin, (bins[i + 1] + bins[i]) / 2)
		Pkbin = np.append(Pkbin, n[i] / total)
	return Pkbin, kbin


def split_data(user):
	trainDays = user[0].getDailySequences()
	testDays = user[1].getDailySequences()
	#print(len(testDays))
	trainSeqs = [(k, trainDays[k]) for k in trainDays.keys()]
	testSeqs = [(k, testDays[k]) for k in testDays.keys()]
	return trainSeqs, testSeqs


def build_priorLM(users):
	counter = 0
	corpus = []
	for user in users:
		trainSeqs, testSeqs = split_data(user)
		random.shuffle(trainSeqs)
		corpus.extend(trainSeqs[:30])
		counter += 1
	print("Number of users = {}".format(counter))
	print("number of users days in training set = {}".format(len(corpus)))
	priorLM = ngram.mobilityNgram(corpus, sv, tv, PART_INFO = PART_INFO)
	return priorLM


sv = getStationVocab(vocabFile)
tv = getTimeVocab()
# train_users = smartcard.readPanelData(trainDataFile, vocabFile)
# test_users = smartcard.readPanelData(testDataFile, vocabFile)

train_users = smartcard.readPanelData_baichuan(trainDataFile, vocabFile)
test_users = smartcard.readPanelData_baichuan(testDataFile, vocabFile)

assert(len(train_users) == len(test_users))

station_idx = pd.read_csv(vocabFile, header=None)
station_idx.columns = ['MTR_ID', 'Name', 'Line', 'Short_name', 'New_ID']
station_idx['New_ID_str'] = station_idx['New_ID'].astype('int').astype('str')
station_idx_dict = {}
for old_id, new_id in zip(station_idx['MTR_ID'], station_idx['New_ID_str']):
	station_idx_dict[new_id] = old_id


for i in range(len(train_users)):
	assert(train_users[i].id == test_users[i].id)


users = list(zip(train_users, test_users))
priorLM = build_priorLM(users)


def individual_model_baseline(user):
	trainSeqs, testSeqs = split_data(user)
	userLM = ngram_baseline.mobilityNgram_baseline(trainSeqs, sv, tv, ID=user[0].id)
	return userLM.evaluate(testSeqs)


def individual_model_baseline_agg(users):
	'''
	pool = mp.Pool(processes=20)
	results = pool.map(individual_model_baseline, users)
    '''
	results = []
	for u in users:
		results.append(individual_model_baseline(u))
		# print results[-1].id, results[-1].result1.accu, results[-1].result2.accu
	
	names = ["All", "Time", "Entry", "Exit"]
	pp1 = [r.result1.perp for r in results]
	pred1 = [r.result1.accu for r in results]
	pp = [r.result2.perp for r in results if r.result2.perp is not None]
	pred = [r.result2.accu for r in results if r.result2.accu is not None]
	ppList1 = list(zip(*pp1))
	predList1 = list(zip(*pred1))
	ppList = list(zip(*pp))
	predList = list(zip(*pred))

	print("\tType\tPerplexity\tPrediction\tCount")
	print("Problem 2A:")
	for i in range(4):
		print("\t" + names[i] + "\t" + str(round(np.median(ppList1[i]), 2)) +\
			"\t\t" + str(round(np.median(predList1[i]) * 100, 1)) + "%\t\t" +\
			str(len(ppList1[i])))
	print("Problem 2B:")
	for i in range(4):
		print("\t" + names[i] + "\t" + str(round(np.median(ppList[i]), 2)) +\
			"\t\t" + str(round(np.median(predList[i]) * 100, 1)) + "%\t\t" +\
			str(len(ppList[i])))
	'''
	wt = csv.writer(open('../output/next_trip_prediction_baseline.csv', 'wt'))
	wt.writerow(['uid', 'pp1', 'ppT1', 'ppO1', 'ppD1', 'ac1', 'acT1', 'acO1', 'acD1', 'acT1_t', 'acO1_t', 'acD1_t',
				'pp', 'ppT', 'ppO', 'ppD', 'ac', 'acT', 'acO', 'acD', 'acT_t', 'acO_t', 'acD_t'])
	nan_perp = [-1, -1, -1, -1]
	nan_accu = [-1, -1, -1, -1, -1, -1, -1]
	for r in results:
		if r.result2.perp is not None:
			wt.writerow([r.id] + list(r.result1.perp) + list(r.result1.accu) + list(r.result2.perp) + list(r.result2.accu))
		else:
			wt.writerow([r.id] + list(r.result1.perp) + list(r.result1.accu) + nan_perp + nan_accu)

	wt2 = csv.writer(open('../output/next_trip_time_diff_baseline.csv', 'wt'))
	wt2.writerow(['type', 'predT', 'trueT'])
	for r in results:
		for (predT, trueT) in r.result1.timeDiff:
			wt2.writerow(['A', predT, trueT])
		if r.result2.timeDiff is not None:
			for (predT, trueT) in r.result2.timeDiff:
				wt2.writerow(['B', predT, trueT])

	wt3 = csv.writer(open('../output/next_trip_pred_ranks_baseline.csv', 'wt'))
	wt3.writerow(['type', 'rankT', 'rankO', 'rankD'])
	for r in results:
		tRank1, oRank1, dRank1 = r.result1.predRank
		for i in range(len(tRank1)):
			wt3.writerow(['A', tRank1[i], oRank1[i], dRank1[i]])
		if r.result2.predRank is not None:
			tRank, oRank, dRank = r.result2.predRank
			for j in range(len(tRank)):
				wt3.writerow(['B', tRank[j], oRank[j], dRank[j]])
	'''

def individual_model_eval(user):
	# print os.getpid(), user.id
	trainSeqs, testSeqs = split_data(user)
	userLM = ngram.mobilityNgram(trainSeqs, sv, tv, priorLM, ID=user[0].id)
	return userLM.evaluate(testSeqs)






def individual_model_eval_agg(users):
	pool = mp.Pool(processes=20)
	results = pool.map(individual_model_eval, users)

	names = ["All", "Time", "Entry", "Exit"]
	pp1 = [r.result1.perp for r in results]
	pred1 = [r.result1.accu for r in results]
	pp = [r.result2.perp for r in results if r.result2.perp is not None]
	pred = [r.result2.accu for r in results if r.result2.accu is not None]
	ppList1 = list(zip(*pp1))
	predList1 = list(zip(*pred1))
	ppList = list(zip(*pp))
	predList = list(zip(*pred))

	print("\tType\tPerplexity\tPrediction\tCount")
	print("Problem 2A:")
	for i in range(4):
		print("\t" + names[i] + "\t" + str(round(np.median(ppList1[i]), 2)) +\
			"\t\t" + str(round(np.median(predList1[i]) * 100, 1)) + "%\t\t" +\
			str(len(ppList1[i])))
	print("Problem 2B:")
	for i in range(4):
		print("\t" + names[i] + "\t" + str(round(np.median(ppList[i]), 2)) +\
			"\t\t" + str(round(np.median(predList[i]) * 100, 1)) + "%\t\t" +\
			str(len(ppList[i])))
'''
	wt = csv.writer(open('../output/next_trip_prediction.csv', 'wt'))
	wt.writerow(['uid', 'pp1', 'ppT1', 'ppO1', 'ppD1', 'ac1', 'acT1', 'acO1', 'acD1', 'acT1_t', 'acO1_t', 'acD1_t',
				'pp', 'ppT', 'ppO', 'ppD', 'ac', 'acT', 'acO', 'acD', 'acT_t', 'acO_t', 'acD_t'])
	nan_perp = [-1, -1, -1, -1]
	nan_accu = [-1, -1, -1, -1, -1, -1, -1]
	for r in results:
		if r.result2.perp is not None:
			wt.writerow([r.id] + list(r.result1.perp) + list(r.result1.accu) + list(r.result2.perp) + list(r.result2.accu))
		else:
			wt.writerow([r.id] + list(r.result1.perp) + list(r.result1.accu) + nan_perp + nan_accu)

	wt2 = csv.writer(open('../output/next_trip_time_diff.csv', 'wt'))
	wt2.writerow(['type', 'predT', 'trueT'])
	for r in results:
		for (predT, trueT) in r.result1.timeDiff:
			wt2.writerow(['A', predT, trueT])
		if r.result2.timeDiff is not None:
			for (predT, trueT) in r.result2.timeDiff:
				wt2.writerow(['B', predT, trueT])

	wt3 = csv.writer(open('../output/next_trip_pred_ranks.csv', 'wt'))
	wt3.writerow(['type', 'rankT', 'rankO', 'rankD'])
	for r in results:
		tRank1, oRank1, dRank1 = r.result1.predRank
		for i in range(len(tRank1)):
			wt3.writerow(['A', tRank1[i], oRank1[i], dRank1[i]])
		if r.result2.predRank is not None:
			tRank, oRank, dRank = r.result2.predRank
			for j in range(len(tRank)):
				wt3.writerow(['B', tRank[j], oRank[j], dRank[j]])
'''

def individual_model_eval_baichuan(users):
	count = 0
	for user in users:
		count += 1
		print('current user',user[0].id, 'count',count, 'total',len(users))
		trainSeqs, testSeqs = split_data(user)
		userLM = ngram.mobilityNgram(trainSeqs, sv, tv, priorLM, ID=user[0].id, Station_idx_dict = station_idx_dict, PART_INFO = PART_INFO, USED_TOD_LIST = USED_TOD_LIST)
		# return userLM.evaluate(testSeqs)
		#print(len(testSeqs))
		# for key in testSeqs:
		# 	print(key)
		results_time_first, results_loc_first, results_time_middle, results_loc_middle = userLM.evaluate_baichuan(testSeqs)

		results_path = '../../../data/results/'
		sample_path = '../../../data/samples/'
		card_id = user[0].id

		file_name_LR = results_path + 'result_LR' + str(card_id) + 'test.csv' #results_path + 'result_con_dur+loc_' + str(card_id) + 'test' + '.csv'
		file_name_MC = results_path + 'result_Location_MC' + str(card_id) + '.csv' #results_path + 'result_con_dur+loc_' + str(card_id) + 'test' + '.csv'
		if os.path.exists(file_name_LR) and os.path.exists(file_name_MC):
			p_results_dur = pd.read_csv(file_name_LR)
			p_results_loc = pd.read_csv(file_name_MC)

			act_data_file = sample_path + 'sample_' + card_id + '_201407_201408_all.csv'
			act_data = pd.read_csv(act_data_file)

			first_act_dur = p_results_dur.loc[p_results_dur['activity_index'] == 0]
			mid_act_dur = p_results_dur.loc[p_results_dur['activity_index'] > 0]
			first_act_loc = p_results_loc.loc[p_results_loc['activity_index'] == 0]
			mid_act_loc = p_results_loc.loc[p_results_loc['activity_index'] > 0]

			# print(len(first_act_dur))
			# print(len(first_act_loc))
			# print(len(results_time_first))
			# print(len(results_loc_first))
			# print(len(mid_act_dur))
			# print(len(mid_act_loc))
			# print(len(results_time_middle))
			# print(len(results_loc_middle))
			assert len(first_act_dur) == len(results_time_first)
			assert len(first_act_loc) == len(results_loc_first)
			assert len(mid_act_dur) == len(results_time_middle)
			assert len(mid_act_loc) == len(results_loc_middle)
			results_time_first['ID'] = list(first_act_dur['ID'])
			results_time_first['Ground_truth_duration'] = list(first_act_dur['Ground_truth_duration'])
			results_time_first['activity_index'] = list(first_act_dur['activity_index'])
			results_loc_first['ID'] = list(first_act_loc['ID'])
			results_loc_first['Ground_truth'] = list(first_act_loc['Ground_truth'])
			results_loc_first['activity_index'] = list(first_act_loc['activity_index'])

			results_time_middle['ID'] = list(mid_act_dur['ID'])
			results_time_middle['Ground_truth_duration'] = list(mid_act_dur['Ground_truth_duration'])
			results_time_middle['activity_index'] = list(mid_act_dur['activity_index'])
			results_loc_middle['ID'] = list(mid_act_loc['ID'])
			results_loc_middle['Ground_truth'] = list(mid_act_loc['Ground_truth'])
			results_loc_middle['activity_index'] = list(mid_act_loc['activity_index'])

			#print(type(results_loc_middle.iloc[0, 0]))
			results_time_pred = pd.concat([results_time_first, results_time_middle],sort = False)
			results_time_pred = results_time_pred.sort_values(['ID'])

			t_acc = len(results_time_first.loc[results_time_first['Pred_t'] == results_time_first['Actual_t']])/len(results_time_first)
			print(card_id, 'First T accuracy', t_acc)
			if len(results_time_middle) > 0:
				t_acc = len(results_time_middle.loc[results_time_middle['Pred_t'] == results_time_middle['Actual_t']])/len(results_time_middle)
				print(card_id, 'Middle T accuracy', t_acc)

			results_time_pred_time = results_time_pred.merge(act_data[['ID','date_time','duration_trip','seq_ID']], on = ['ID'])
			temp = results_time_pred_time['date_time'].str.split(' ',expand = True)
			temp2 = temp.iloc[:,1].str.split(':',expand = True)
			results_time_pred_time['last_tap_in_time'] = temp2.iloc[:,0].astype('int') * 3600 + temp2.iloc[:,1].astype('int') * 60 + temp2.iloc[:,2].astype('int')
			results_time_pred_time['Pred_dur'] = results_time_pred_time['Predict_duration'] - results_time_pred_time['last_tap_in_time']
			results_time_pred_time['Pred_dur'] -= results_time_pred_time['duration_trip']
			results_time_pred_time.loc[results_time_pred_time['Pred_dur'] <= 0, 'Pred_dur'] = 3600
			results_time_pred_time['Predict_duration'] = results_time_pred_time['Pred_dur']
			results_time_pred_time = results_time_pred_time.loc[:, results_time_pred.columns]
			results_time_pred_time.to_csv(results_path + 'result_NGRAM_con_dur_' + card_id + '.csv', index=False)

			results_loc_pred = pd.concat([results_loc_first, results_loc_middle],sort = False)
			results_loc_pred = results_loc_pred.sort_values(['ID'])
			results_loc_pred['Correct'] = 0
			results_loc_pred.loc[results_loc_pred['Predict1'] == results_loc_pred['Ground_truth'], 'Correct'] = 1

			results_loc_pred.to_csv(results_path + 'result_NGRAM_location_' + card_id + '.csv',index= False)

			#a=1

			# save

		else:
			print(card_id, 'previous results not exist, run it first before run Ngram')
			exit()


def individual_model_eval_agg2(users):
	results = []
	for u in users:
		results.append(individual_model_eval(u))

	names = ["All", "Time", "Entry", "Exit"]
	pp1 = [r.result1.perp for r in results]
	pred1 = [r.result1.accu for r in results]
	pp = [r.result2.perp for r in results if r.result2.perp is not None]
	pred = [r.result2.accu for r in results if r.result2.accu is not None]
	ppList1 = list(zip(*pp1))
	predList1 = list(zip(*pred1))
	ppList = list(zip(*pp))
	predList = list(zip(*pred))

	print("\tType\tPerplexity\tPrediction\tCount")
	print("Problem 2A:")
	for i in range(4):
		print("\t" + names[i] + "\t" + str(round(np.median(ppList1[i]), 2)) +\
			"\t\t" + str(round(np.median(predList1[i]) * 100, 1)) + "%\t\t" +\
			str(len(ppList1[i])))
	print("Problem 2B:")
	for i in range(4):
		print("\t" + names[i] + "\t" + str(round(np.median(ppList[i]), 2)) +\
			"\t\t" + str(round(np.median(predList[i]) * 100, 1)) + "%\t\t" +\
			str(len(ppList[i])))
	wt = csv.writer(open('../output/next_trip_prediction_hk.csv', 'wt', newline=''))
	wt.writerow(['uid', 'pp1', 'ppT1', 'ppO1', 'ppD1', 'ac1', 'acT1', 'acO1', 'acD1', 'acT1_t', 'acO1_t', 'acD1_t',
				'pp', 'ppT', 'ppO', 'ppD', 'ac', 'acT', 'acO', 'acD', 'acT_t', 'acO_t', 'acD_t'])
	nan_perp = [-1, -1, -1, -1]
	nan_accu = [-1, -1, -1, -1, -1, -1, -1]
	for r in results:
		if r.result2.perp is not None:
			wt.writerow([r.id] + list(r.result1.perp) + list(r.result1.accu) + list(r.result2.perp) + list(r.result2.accu))
		else:
			wt.writerow([r.id] + list(r.result1.perp) + list(r.result1.accu) + nan_perp + nan_accu)



if __name__ == "__main__":
	start = time.time()
	all_u_id = [u[0].id for u in users]
	# 998325943
	for test_id in all_u_id:
		sample_users = [u for u in users if u[0].id == test_id]
		#individual_model_eval_agg2(sample_users)
		individual_model_eval_baichuan(sample_users)
		#individual_model_baseline_agg(sample_users)
		print('Running Time: {} Seconds'.format(str(time.time() - start)))
		print('===================================')
