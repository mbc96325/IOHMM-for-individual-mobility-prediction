import csv
import numpy as np
from collections import Counter


def load_stations(filepath, n=2):
	rd = csv.reader(open(filepath, 'rb'), delimiter=",")
	vocab = []
	for s in rd:
		vocab.append(s[-1])
	return vocab


def buildIndex(wordList):
	vocab, indx = {}, 0
	for w in wordList:
		if w not in vocab.keys():
			vocab[w] = indx
			indx += 1
	return vocab


def getDayOfWeek(daykey):
	day = (daykey - 12052) % 7
	if day == 1:
		return 'MON'
	elif day == 2:
		return 'TUE'
	elif day == 3:
		return 'WED'
	elif day == 4:
		return 'THU'
	elif day == 5:
		return 'FRI'
	elif day == 6:
		return 'SAT'
	elif day == 0:
		return 'SUN'


def getDayType(daykey):
	day = (daykey - 12052) % 7
	if day > 0 and day < 6:
		return 'Weekday'
	else:
		return 'Weekend'


def getKeyByValue(D, v):
	return list(D.keys())[list(D.values()).index(v)]


def isInt(string):
	try:
		int(string)
		return True
	except ValueError:
		return False


def ngramGen(seqs):
	ngramIn = []
	ngramOut = []
	ngramTime = []
	# start_OD = ["START-" + j for j in ["O", "D"]]
	padding = 3 - 1
	start_snt = ['<START-' + str(j) + '>' for j in range(padding, 0, -1)]
	end_snt = ["<STOP>"]
	for seq in seqs:
		day, snt = seq[0], seq[1]
		# start_snt = ["START-" + str(dow)] + start_OD
		s = start_snt + snt + end_snt
		for i in xrange(padding, len(s)):
			if (i - padding + 1) % 3 == 1:
				ngramTime.append(tuple(s[i - 2: i + 1]))
			elif (i - padding + 1) % 3 == 2:
				ngramIn.append(tuple(s[i - 2: i + 1]))
			else:
				ngramOut.append(tuple(s[i - 2: i + 1]))
	return ngramTime, ngramIn, ngramOut


#############################
class mobilityNgram(object):
	def __init__(self, corpus, priorLM=None):
		ngramTime, ngramIn, ngramOut = ngramGen(corpus)
		nTime = len(ngramTime[0])
		nIn = len(ngramIn[0])
		nOut = len(ngramOut[0])
		station_list = load_stations("../../data/station_vocab.csv")
		station_vocab = buildIndex(station_list)
		time_list = [str(i) for i in range(0, 27 * 2)]
		time_list.append("<STOP>")
		time_vocab = buildIndex(time_list)
		self.lmTime = ngramLM(nTime, ngramTime, time_vocab)
		self.lmIn = ngramLM(nIn, ngramIn, station_vocab)
		self.lmOut = ngramLM(nOut, ngramOut, station_vocab)

	def perplexity(self, corpus):
		ngramTime, ngramIn, ngramOut = ngramGen(corpus)
		LLB_T, N_T = self._calc_entropy(ngramTime, self.lmTime)
		LLB_O, N_O = self._calc_entropy(ngramIn, self.lmIn)
		LLB_D, N_D = self._calc_entropy(ngramOut, self.lmOut)
		LLB = LLB_T + LLB_O + LLB_D
		N = N_T + N_O + N_D
		return pow(2.0, -LLB/N), pow(2.0, -LLB_T/N_T),\
			pow(2.0, -LLB_O/N_O), pow(2.0, -LLB_D/N_D)

	def _calc_entropy(self, ngrams, lm):
		LLB = 0.0
		for w in ngrams:
			LLB += np.log2(lm.getProb(w))
		return LLB, len(ngrams)

	def prediction(self, corpus):
		ngramTime, ngramIn, ngramOut = ngramGen(corpus)
		COR_T, N_T = self._pred_eval(ngramTime, self.lmTime)
		COR_O, N_O = self._pred_eval(ngramIn, self.lmIn)
		COR_D, N_D = self._pred_eval(ngramOut, self.lmOut)
		COR = COR_T + COR_O + COR_D
		N = N_T + N_O + N_D
		return COR*1.0/N, COR_T*1.0/N_T, COR_O*1.0/N_O, COR_D*1.0/N_D

	def _pred_eval(self, ngrams, lm):
		cor = 0
		for w in ngrams:
			cor += lm.predict_eval(w)
		return cor, len(ngrams)


#############################
class ngramLM(object):
	def __init__(self, n, ngrams, vocab, alpha=0.001):
		assert n > 1
		self.n = n
		self.vocab = vocab
		self.alpha = alpha
		self.ngramCounts = self._count(ngrams)
		self.probs = treeDict(self.n - 1)

	def _count(self, ngrams):
		ngramCounts = treeDict(self.n - 1)
		V = len(self.vocab)
		counts = Counter(ngrams)
		# Counting
		for ngram in counts.keys():
			prefix = ngram[:-1]
			if ngramCounts.hasKey(prefix) is False:
				ngramCounts.insert(prefix, np.zeros(V))
			ind = self.vocab[ngram[-1]]
			ngramCounts.getValue(prefix)[ind] += counts[ngram]
		return ngramCounts

	def getProbDbn(self, prefix):
		if self.probs.hasKey(prefix):
			return self.probs.getValue(prefix)

		alpha = self.alpha
		V = len(self.vocab)
		probDbn = np.zeros(V) + alpha

		if self.ngramCounts.hasKey(prefix):
			probDbn += self.ngramCounts.getValue(prefix)

		probDbn /= np.sum(probDbn)
		self.probs.insert(prefix, probDbn)

		return probDbn

	def getProb(self, ngram):
		prefix = ngram[:-1]
		probDbn = self.getProbDbn(prefix)
		if hasattr(probDbn, "__getitem__"):
			ind = self.vocab[ngram[-1]]
			return probDbn[ind]
		else:
			return probDbn

	def predict(self, prefix):
		probDbn = self.getProbDbn(prefix)
		predIndx = np.argmax(probDbn)
		return predIndx

	def predict_eval(self, ngram):
		prefix = ngram[:-1]
		predIndx = self.predict(prefix)
		if predIndx == self.vocab[ngram[-1]]:
			return 1
		else:
			# print ngram, getKeyByValue(self.vocab, predIndx)
			return 0


#############################
class unigramLM(object):
	def __init__(self, ngrams, vocab, beta=0.1):
		self.vocab = vocab
		self.beta = beta
		self.ngramCounts = self._learnParas(ngrams)

	def _learnParas(self, ngrams):
		V = len(self.vocab)
		ngram_counts = Counter(ngrams)
		ngramCounts = np.zeros(V)
		# Counting
		for ngram in ngram_counts.keys():
			ind = self.vocab[ngram[-1]]
			ngramCounts[ind] += ngram_counts[ngram]
		return ngramCounts

	def getProbDbn(self):
		V = len(self.vocab)
		beta = self.beta
		probDbn = self.ngramCounts + beta * V * 1e-20
		probDbn /= np.sum(probDbn)
		return probDbn


#############################
def paths(D, cur=()):
	if isinstance(D, dict):
		for k, v in D.iteritems():
			for path in paths(v, cur+(k,)):
				yield path
	else:
		yield cur


def Normalize(D):
	if isinstance(D, dict):
		for v in D.values():
			Normalize(v)
	else:
		D /= np.sum(D)


class treeDict(object):
	def __init__(self, levels):
		self.levels = levels
		self.dict = {}

	def getPath(self, keys):
		if hasattr(keys, "__getitem__"):
			return keys
		else:
			return (keys,)  # Return keys in tuple

	def hasKey(self, keys):
		path = self.getPath(keys)
		assert len(path) == self.levels
		Dict = self.dict
		for i in xrange(self.levels):
			if path[i] in Dict.keys():
				Dict = Dict[path[i]]
			else:
				return False
		return True

	def insert(self, keys, value):
		path = self.getPath(keys)
		assert len(path) == self.levels
		Dict = self.dict
		for i in xrange(self.levels - 1):
			if path[i] not in Dict.keys():
				Dict[path[i]] = {}
			Dict = Dict[path[i]]
		Dict[path[-1]] = value

	def getValue(self, keys):
		path = self.getPath(keys)
		Dict = reduce(lambda d, k: d[k], path[:-1], self.dict)
		return Dict[path[-1]]

	def updateValue(self, keys, val):
		path = self.getPath(keys)
		Dict = reduce(lambda d, k: d[k], path[:-1], self.dict)
		Dict[path[-1]] = val

	def traverseKeys(self):
		return list(paths(self.dict))

	# Special function for probability calculation
	def normalize(self):
		Dict = self.dict
		Normalize(Dict)
