import multiprocessing as mp
import time
import os
import oyster_reader as oyster


dataFile = "../data/oysterdata.csv"
vocabFile = "../data/station_vocab.csv"


def individual_model_eval(user):
	print os.getpid(), user.id
	return user.id


def individual_model_eval_agg(users):
	pool = mp.Pool(processes=3)
	results = pool.map(individual_model_eval, users)
	print len(results)


if __name__ == "__main__":
	start = time.time()
	users = oyster.readPanelData2(dataFile, vocabFile, 1000)
	freq_users = [u for u in users if u.getActiveDays() >= 60]
	print 'Number of users: {}'.format(len(freq_users))
	individual_model_eval_agg(freq_users)
	print 'Running Time: {} Seconds'.format(str(time.time() - start))