import oyster_reader as oyster
import info_theory as it
import time
import numpy as np


def get_max_prob(X):
	return np.max(it.probDbn(X))


def first_trip_frequency(users):
	oProbs = []
	dProbs = []
	tProbs = []
	for u in users:
		oList1 = u.getStationList(stationType='in', order=0)
		dList1 = u.getStationList(stationType='out', order=0)
		tList1 = u.getTripTimeList(60, order=0)
		oProbs.append(get_max_prob(oList1))
		dProbs.append(get_max_prob(dList1))
		tProbs.append(get_max_prob(tList1))
	print np.median(tProbs)
	print np.median(oProbs)
	print np.median(dProbs)


if __name__ == "__main__":
	start = time.time()
	dataFile = "../data/oysterdata.csv"
	vocabFile = "../data/station_vocab.csv"
	users = oyster.readPanelData2(dataFile, vocabFile)
	freq_users = [u for u in users if u.getActiveDays() >= 60]
	first_trip_frequency(freq_users)
	print 'Running Time: {} Seconds'.format(str(time.time() - start))
