import csv
import numpy as np
import random
# import os
import multiprocessing as mp
import time
import oyster_reader as oyster
import ngram_online as ngram
# from matplotlib import pyplot as plt


start = time.time()
random.seed(111)

dataFile = "../data/oysterdata.csv"
vocabFile = "../data/station_vocab.csv"


def load_stations(filepath):
	rd = csv.reader(open(filepath, 'rU'), delimiter=",")
	vocab = []
	for s in rd:
		vocab.append(s[-1])
	return list(set(vocab))


def getStationVocab(filepath):
	return ngram.vocabulary(load_stations(filepath))


def getTimeVocab():
	time_list = [str(i) for i in range(3, 27)]
	return ngram.vocabulary(time_list)


def build_priorLM(users):
	counter = 0
	corpus = []
	for user in users:
		activeDays = user.getActiveDayList()
		if len(activeDays) < 60:
			continue
		dailySeqs = user.getDailySequences()
		daySeqs = [(d, dailySeqs[d]) for d in activeDays]
		corpus.extend(daySeqs[:30])
		counter += 1
	print "Number of users = {}".format(counter)
	print "number of users days in training set = {}".format(len(corpus))
	priorLM = ngram.mobilityNgram(corpus, sv, tv)
	return priorLM


sv = getStationVocab(vocabFile)
tv = getTimeVocab()
users = oyster.readPanelData2(dataFile, vocabFile, 10000)
freq_users = [u for u in users if u.getActiveDays() >= 60]
priorLM = build_priorLM(freq_users)


def individual_model_eval(user):
	# print os.getpid(), user.id
	activeDays = user.getActiveDayList()
	dailySeqs = user.getDailySequences()
	daySeqs = [(d, dailySeqs[d]) for d in activeDays]
	userLM = ngram.mobilityNgram(daySeqs[:30], sv, tv, priorLM)
	N = len(daySeqs)
	day_idx = range(30, N)
	test_day_idx = random.sample(day_idx, 30)
	for d in day_idx:
		if d in test_day_idx:
			userLM.evaluate(daySeqs[d])
		else:
			userLM.update(daySeqs[d])
	return userLM.getEvaluationResults()


def individual_model_eval_agg(users):
	pool = mp.Pool(processes=20)
	results = pool.map(individual_model_eval, users)

	names = ["All", "Time", "Entry", "Exit"]
	pp1 = [r.perp1 for r in results]
	pred1 = [r.accu1 for r in results]
	pp = [r.perp for r in results if r.perp is not None]
	pred = [r.accu for r in results if r.accu is not None]
	ppList1 = zip(*pp1)
	predList1 = zip(*pred1)
	ppList = zip(*pp)
	predList = zip(*pred)

	print "\tType\tPerplexity\tPrediction\tCount"
	print "Problem 2A:"
	for i in xrange(4):
		print "\t" + names[i] + "\t" + str(round(np.median(ppList1[i]), 2)) +\
			"\t\t" + str(round(np.median(predList1[i]) * 100, 1)) + "%\t\t" +\
			str(len(ppList1[i]))
	print "Problem 2B:"
	for i in xrange(4):
		print "\t" + names[i] + "\t" + str(round(np.median(ppList[i]), 2)) +\
			"\t\t" + str(round(np.median(predList[i]) * 100, 1)) + "%\t\t" +\
			str(len(ppList[i]))
'''
	wt = csv.writer(open('../output/next_trip_prediction2.csv', 'wt'), delimiter=',')
	wt.writerow(['pp1', 'ppT1', 'ppO1', 'ppD1', 'ac1', 'acT1', 'acO1', 'acD1', 'acT1_t', 'acO1_t', 'acD1_t',
				'pp', 'ppT', 'ppO', 'ppD', 'ac', 'acT', 'acO', 'acD', 'acT_t', 'acO_t', 'acD_t'])
	nan_perp = [-1, -1, -1, -1]
	nan_accu = [-1, -1, -1, -1, -1, -1, -1]
	for r in results:
		if r.perp is not None:
			wt.writerow(list(r.perp1) + list(r.accu1) + list(r.perp) + list(r.accu))
		else:
			wt.writerow(list(r.perp1) + list(r.accu1) + nan_perp + nan_accu)

	wt2 = csv.writer(open('../output/next_trip_time_diff2.csv', 'wt'), delimiter=',')
	wt2.writerow(['type', 'predT', 'trueT'])
	for r in results:
		for (predT, trueT) in r.timeDiff1:
			wt2.writerow(['A', predT, trueT])
		if r.timeDiff is not None:
			for (predT, trueT) in r.timeDiff:
				wt2.writerow(['B', predT, trueT])
'''

def individual_model_eval_agg2(users):
	results = []
	for userid, u in enumerate(users):
		results.append(individual_model_eval(u))
		print userid

	names = ["All", "Time", "Entry", "Exit"]
	pp1 = [r.perp1 for r in results]
	pred1 = [r.accu1 for r in results]
	pp = [r.perp for r in results if r.perp is not None]
	pred = [r.accu for r in results if r.accu is not None]
	ppList1 = zip(*pp1)
	predList1 = zip(*pred1)
	ppList = zip(*pp)
	predList = zip(*pred)

	print "\tType\tPerplexity\tPrediction\tCount"
	print "Problem 2A:"
	for i in xrange(4):
		print "\t" + names[i] + "\t" + str(round(np.median(ppList1[i]), 2)) +\
			"\t\t" + str(round(np.median(predList1[i]) * 100, 1)) + "%\t\t" +\
			str(len(ppList1[i]))
	print "Problem 2B:"
	for i in xrange(4):
		print "\t" + names[i] + "\t" + str(round(np.median(ppList[i]), 2)) +\
			"\t\t" + str(round(np.median(predList[i]) * 100, 1)) + "%\t\t" +\
			str(len(ppList[i]))


if __name__ == "__main__":
	individual_model_eval_agg(freq_users)
	print 'Running Time: {} Seconds'.format(str(time.time() - start))
