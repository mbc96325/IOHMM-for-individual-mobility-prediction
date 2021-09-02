import numpy as np
from collections import Counter
# from scipy.special import digamma

from ngramGen import ngramGenerator


DECAY_FACTOR = 1.
DECAY_FACTOR_FIRST = 1.


def getKeyByValue(D, v):
	return list(D.keys())[list(D.values()).index(v)]


def isInt(string):
	try:
		int(string)
		return True
	except ValueError:
		return False


def updateCounts(old_counter, new_counter, distance, decay_factor=0.95):
	keys = list(set(old_counter.keys() + new_counter.keys()))
	discount = np.power(decay_factor, distance)
	updated_counter = Counter({k: old_counter[k] * discount + new_counter[k] for k in keys})
	return updated_counter


#############################
class mobilityNgram(object):
	def __init__(self, dayseqs, station_vocab, time_vocab, priorLM=None, optimize=True):
		self.current_time = dayseqs[-1][0]
		ng = ngramGenerator(dayseqs)
		ngramTime1, ngramIn1, ngramOut1 = ng.getNGramsFirst()
		ngramTime, ngramIn, ngramOut = ng.getNGrams()
		self.station_vocab = station_vocab
		self.time_vocab = time_vocab

		if priorLM is None:
			self.lmTimeFirst = ngramLM(ngramTime1, time_vocab)
			self.lmInFirst = ngramLM(ngramIn1, station_vocab)
			self.lmOutFirst = ngramLM(ngramOut1, station_vocab)
			self.lmTime = ngramLM(ngramTime, time_vocab)
			self.lmIn = ngramLM(ngramIn, station_vocab)
			self.lmOut = ngramLM(ngramOut, station_vocab)
		else:
			self.lmTimeFirst = ngramLM(ngramTime1, time_vocab, priorLM.lmTimeFirst,
					timeSmoothing=True, optimize_paras=optimize, alpha=20.0, beta=0.6)
			self.lmInFirst = ngramLM(ngramIn1, station_vocab, priorLM.lmInFirst,
					timeSmoothing=True, optimize_paras=optimize, alpha=5.0, beta=0.6)
			self.lmOutFirst = ngramLM(ngramOut1, station_vocab, priorLM.lmOutFirst,
					timeSmoothing=True, optimize_paras=optimize, alpha=10.0, beta=0.6)
			if len(ngramTime) > 0:
				self.lmTime = ngramLM(ngramTime, time_vocab, priorLM.lmTime,
					timeSmoothing=True, optimize_paras=optimize, alpha=20.0, beta=0.6)
				self.lmIn = ngramLM(ngramIn, station_vocab, priorLM.lmIn,
					timeSmoothing=True, optimize_paras=optimize, alpha=5.0, beta=0.6)
				self.lmOut = ngramLM(ngramOut, station_vocab, priorLM.lmOut,
					timeSmoothing=True, optimize_paras=optimize, alpha=10.0, beta=0.6)
			else:
				self.lmTime = priorLM.lmTime
				self.lmIn = priorLM.lmIn
				self.lmOut = priorLM.lmOut

		# for evaluation
		self.total_preds = 0
		self.right_preds = np.zeros(4)
		self.total_preds_first = 0
		self.right_preds_first = np.zeros(4)

	def update(self, dayseq):
		day, trips = dayseq
		dist = day - self.current_time
		if dist < 0:
			print dist
		ng = ngramGenerator([dayseq])

		ngramTime1, ngramIn1, ngramOut1 = ng.getNGramsFirst()
		self.lmTimeFirst.update(ngramTime1, dist, DECAY_FACTOR_FIRST)
		self.lmInFirst.update(ngramIn1, dist, DECAY_FACTOR_FIRST)
		self.lmOutFirst.update(ngramOut1, dist, DECAY_FACTOR_FIRST)

		ngramTime, ngramIn, ngramOut = ng.getNGrams()
		self.lmTime.update(ngramTime, dist, DECAY_FACTOR)
		self.lmIn.update(ngramIn, dist, DECAY_FACTOR)
		self.lmOut.update(ngramOut, dist, DECAY_FACTOR)

		self.current_time = day

	def evaluate(self, dayseq):
		day, trips = dayseq
		dist = day - self.current_time
		ng = ngramGenerator([dayseq])
		ngramTime1, ngramIn1, ngramOut1 = ng.getNGramsFirst()

		self.lmTimeFirst.evaluate(ngramTime1, True)
		self.lmInFirst.evaluate(ngramIn1)
		self.lmOutFirst.evaluate(ngramOut1)
		self.evaluate_trip_first(ngramTime1, ngramOut1)

		self.lmTimeFirst.update(ngramTime1, dist, DECAY_FACTOR_FIRST)
		self.lmInFirst.update(ngramIn1, dist, DECAY_FACTOR_FIRST)
		self.lmOutFirst.update(ngramOut1, dist, DECAY_FACTOR_FIRST)

		ngramTime, ngramIn, ngramOut = ng.getNGrams()
		if len(ngramTime) > 0:
			self.lmTime.evaluate(ngramTime, True)
			self.lmIn.evaluate(ngramIn)
			self.lmOut.evaluate(ngramOut)
			self.evaluate_trip(ngramOut)

			self.lmTime.update(ngramTime, dist, DECAY_FACTOR)
			self.lmIn.update(ngramIn, dist, DECAY_FACTOR)
			self.lmOut.update(ngramOut, dist, DECAY_FACTOR)

		self.current_time = day

	def getEvaluationResults(self):
		a1 = self.getPredAccuracyFirst()
		perp1 = (self.getPerplexityFirst(), self.lmTimeFirst.getPerplexity(), self.lmInFirst.getPerplexity(), self.lmOutFirst.getPerplexity())
		accu1 = (a1[0], self.lmTimeFirst.getPredAccuracy(), self.lmInFirst.getPredAccuracy(), self.lmOutFirst.getPredAccuracy(), a1[1], a1[2], a1[3])
		timediff1 = self.lmTimeFirst.getTimeDiffs()
		if self.total_preds > 5:
			a = self.getPredAccuracy()
			perp = (self.getPerplexity(), self.lmTime.getPerplexity(), self.lmIn.getPerplexity(), self.lmOut.getPerplexity())
			accu = (a[0], self.lmTime.getPredAccuracy(), self.lmIn.getPredAccuracy(), self.lmOut.getPredAccuracy(), a[1], a[2], a[3])
			timediff = self.lmTime.getTimeDiffs()
		else:
			perp = None
			accu = None
			timediff = None
		# print accu1

		return model_results(perp1, accu1, perp, accu, timediff1, timediff)

	def evaluate_trip(self, ngramOut):
		for i in range(len(ngramOut)):
			prevTrip = ngramOut[i][:3]
			trueTrip = ngramOut[i][-3:]
			self.predict_eval(prevTrip, trueTrip)

	def evaluate_trip_first(self, ngramTime1, ngramOut1):
		for i in range(len(ngramTime1)):
			dayFeatures = ngramTime1[i][:-1]
			trueFirstTrip = ngramOut1[i][-3:]
			self.predict_eval_first(dayFeatures, trueFirstTrip)

	def predict_eval(self, prevTrip, trueTrip, Ks=(10, 5, 10)):
		timeK, inK, outK = Ks
		K = timeK * inK * outK
		probs = np.zeros(K)
		indices = np.zeros((K, 3), dtype=np.int)
		prevT, prevO, prevD = prevTrip
		trueT, trueO, trueD = trueTrip
		timePrefix = (prevO, prevT, prevD)
		predTimeIndx = self.lmTime.predictTop(timePrefix, timeK)
		predTimeProb = self.lmTime.getProbDbn(timePrefix)[predTimeIndx]
		for ti, tindx in enumerate(predTimeIndx):
			tProb = predTimeProb[ti]
			t = self.time_vocab.getWord(tindx)
			inPrefix = (prevT, prevO, t, prevD)
			predInIndx = self.lmIn.predictTop(inPrefix, inK)
			predInProb = self.lmIn.getProbDbn(inPrefix)[predInIndx]
			for oi, oindx in enumerate(predInIndx):
				oProb = predInProb[oi]
				o = self.station_vocab.getWord(oindx)
				outPrefix = (prevT, prevO, prevD, t, o)
				predOutIndx = self.lmOut.predictTop(outPrefix, outK)
				predOutProb = self.lmOut.getProbDbn(outPrefix)[predOutIndx]

				tripProb = tProb * oProb * predOutProb
				start = ti * inK * outK + oi * outK
				end = start + outK
				probs[start:end] = tripProb
				indices[start:end, 0] = tindx
				indices[start:end, 1] = oindx
				indices[start:end, 2] = predOutIndx

		predIndx = indices[np.argmax(probs), :]
		predT = self.time_vocab.getWord(predIndx[0])
		predO = self.station_vocab.getWord(predIndx[1])
		predD = self.station_vocab.getWord(predIndx[2])

		if predT == trueT and predO == trueO and predD == trueD:
			self.right_preds[0] += 1
		if predT == trueT:
			self.right_preds[1] += 1
		if predO == trueO:
			self.right_preds[2] += 1
		if predD == trueD:
			self.right_preds[3] += 1
		self.total_preds += 1

	def predict_eval_first(self, dayFeatures, trueTrip, Ks=(10, 5, 10)):
		timeK, inK, outK = Ks
		K = timeK * inK * outK
		probs = np.zeros(K)
		indices = np.zeros((K, 3), dtype=np.int)
		dow, dayType = dayFeatures
		trueT, trueO, trueD = trueTrip
		timePrefix = dayFeatures
		predTimeIndx = self.lmTimeFirst.predictTop(timePrefix, timeK)
		predTimeProb = self.lmTimeFirst.getProbDbn(timePrefix)[predTimeIndx]
		for ti, tindx in enumerate(predTimeIndx):
			tProb = predTimeProb[ti]
			t = self.time_vocab.getWord(tindx)
			inPrefix = (dow, dayType, t)
			predInIndx = self.lmInFirst.predictTop(inPrefix, inK)
			predInProb = self.lmInFirst.getProbDbn(inPrefix)[predInIndx]
			for oi, oindx in enumerate(predInIndx):
				oProb = predInProb[oi]
				o = self.station_vocab.getWord(oindx)
				outPrefix = (dow, dayType, t, o)
				predOutIndx = self.lmOutFirst.predictTop(outPrefix, outK)
				predOutProb = self.lmOutFirst.getProbDbn(outPrefix)[predOutIndx]

				tripProb = tProb * oProb * predOutProb
				start = ti * inK * outK + oi * outK
				end = start + outK
				probs[start:end] = tripProb
				indices[start:end, 0] = tindx
				indices[start:end, 1] = oindx
				indices[start:end, 2] = predOutIndx

		predIndx = indices[np.argmax(probs), :]
		predT = self.time_vocab.getWord(predIndx[0])
		predO = self.station_vocab.getWord(predIndx[1])
		predD = self.station_vocab.getWord(predIndx[2])

		if predT == trueT and predO == trueO and predD == trueD:
			self.right_preds_first[0] += 1
		if predT == trueT:
			self.right_preds_first[1] += 1
		if predO == trueO:
			self.right_preds_first[2] += 1
		if predD == trueD:
			self.right_preds_first[3] += 1
		self.total_preds_first += 1

	def getPredAccuracyFirst(self):
		return self.right_preds_first / self.total_preds_first

	def getPredAccuracy(self):
		return self.right_preds / self.total_preds

	def getPerplexityFirst(self):
		logLik = self.lmTimeFirst.logLik + self.lmInFirst.logLik + self.lmOutFirst.logLik
		return np.power(2.0, - logLik / self.total_preds_first)

	def getPerplexity(self):
		logLik = self.lmTime.logLik + self.lmIn.logLik + self.lmOut.logLik
		return np.power(2.0, - logLik / self.total_preds)



#############################
class ngramLM(object):
	def __init__(self, ngrams, vocab, priorLM=None, n=None,
					timeSmoothing=False, optimize_paras=False,
					alpha=1.0, beta=0.6):
		if n:
			self.n = n
		else:
			self.n = len(ngrams[0])
		assert self.n > 1
		self.alpha = alpha
		self.beta = beta
		self.vocab = vocab
		self.timeSmoothing = timeSmoothing
		self.optimize_paras = optimize_paras
		self.priorLM = priorLM
		self.backoffLM = self._buildBackoffLM(ngrams)
		self.counts = Counter(ngrams)
		# used for probability calculation
		self.ngramCounts = self._buildTreeCounts()
		# archive calculated probabilities, for performance
		self.probs = treeDict(self.n - 1)
		# for evaluation
		self.total_preds = 0
		self.right_preds = 0
		self.logLik = 0.0
		self.timeDiffs = []

	def _buildBackoffLM(self, ngrams):
		kgrams = [ngram[1:] for ngram in ngrams]
		if self.n > 2:
			if self.n > 3 and self.timeSmoothing is True:
				backoffLM = ngramLM(kgrams, self.vocab,
					timeSmoothing=True, optimize_paras=False,
					alpha=self.alpha, beta=self.beta)
			else:
				backoffLM = ngramLM(kgrams, self.vocab,
					alpha=self.alpha, beta=self.beta)
		else:
			backoffLM = unigramLM(kgrams, self.vocab)
		return backoffLM

	def _buildTreeCounts(self):
		ngramCounts = treeDict(self.n - 1)
		V = self.vocab.get_num_words()
		counts = self.counts
		# Counting
		for ngram in counts.keys():
			prefix = ngram[:-1]
			if ngramCounts.hasKey(prefix) is False:
				ngramCounts.insert(prefix, np.zeros(V))
			ind = self.vocab.getIndex(ngram[-1])
			ngramCounts.getValue(prefix)[ind] += counts[ngram]
		return ngramCounts

	def update(self, ngrams, dist, decay):
		return self.updateTree(ngrams, dist, decay)

	def updateTree(self, ngrams, dist, decay):
		V = self.vocab.get_num_words()
		ngramCounts = self.ngramCounts
		ngramCounts.discount(distance=dist, decay_factor=decay)
		counts = Counter(ngrams)
		for ngram in counts.keys():
			prefix = ngram[:-1]
			if ngramCounts.hasKey(prefix) is False:
				ngramCounts.insert(prefix, np.zeros(V))
			ind = self.vocab.getIndex(ngram[-1])
			ngramCounts.getValue(prefix)[ind] += counts[ngram]
		if self.optimize_paras:
			self.updateHyperParameters(ngrams)
		kgrams = [ngram[1:] for ngram in ngrams]
		self.backoffLM.updateTree(kgrams, dist, decay)
		self.probs = treeDict(self.n - 1)

	def updateCounts(self, ngrams, dist, decay):
		oldCounts = self.counts
		counts = Counter(ngrams)
		self.counts = updateCounts(oldCounts, counts, dist, decay)
		self.ngramCounts = self._buildTreeCounts()
		if self.optimize_paras:
			self.updateHyperParameters(ngrams)
		kgrams = [ngram[1:] for ngram in ngrams]
		self.backoffLM.updateCounts(kgrams, dist, decay)
		self.probs = treeDict(self.n - 1)

	def updateHyperParameters(self, ngrams):
		a = self.alpha
		b = self.beta
		ll = self._computelogLikelihood(ngrams)
		self.alpha = a + 1.0
		self.beta = b
		ll1 = self._computelogLikelihood(ngrams)
		self.alpha = a
		self.beta = b + 0.1
		ll2 = self._computelogLikelihood(ngrams)
		da = ll1 - ll
		db = (ll2 - ll) * 10
		new_a = a + da * 10
		new_b = b + db * 0.1
		if new_a < 1.0:
			new_a = 1.0
		if new_b > 0.95:
			new_b = 0.95
		if new_b < 0.05:
			new_b = 0.05
		self.alpha = new_a
		self.beta = new_b
		# print new_a, new_b, self._computelogLikelihood(ngrams) - ll

	def _computelogLikelihood(self, ngrams):
		logLik = 0.0
		for w in ngrams:
			logLik += np.log2(self.getProb(w))
		return logLik

	def _getPriorProbDbn(self, prefix):
		if self.priorLM is not None:
			return self.priorLM.getProbDbn(prefix)
		else:
			return 0.0

	def _getBackoffProbDbn(self, prefix):
		if self.n > 2:
			return self.backoffLM.getProbDbn(prefix)
		else:
			return self.backoffLM.getProbDbn()

	def getProbDbn(self, prefix):
		if self.probs.hasKey(prefix):
			return self.probs.getValue(prefix)

		alpha = self.alpha
		beta = self.beta
		V = self.vocab.get_num_words()
		probDbn = np.zeros(V)

		if self.ngramCounts.hasKey(prefix):
			probDbn += self.ngramCounts.getValue(prefix)

		if self.timeSmoothing:
			for i, w in enumerate(prefix):
				if isInt(w):
					t = int(w)

					prefix1 = list(prefix)
					prefix1[i] = str(t - 1)
					prefix1 = tuple(prefix1)
					if self.ngramCounts.hasKey(prefix1):
						probDbn += self.ngramCounts.getValue(prefix1)

					prefix2 = list(prefix)
					prefix2[i] = str(t + 1)
					prefix2 = tuple(prefix2)
					if self.ngramCounts.hasKey(prefix2):
						probDbn += self.ngramCounts.getValue(prefix2)

		m = beta * self._getBackoffProbDbn(prefix[1:]) +\
			(1 - beta) * self._getPriorProbDbn(prefix)

		probDbn += alpha * m

		probDbn /= np.sum(probDbn)
		# self.probs.insert(prefix, probDbn)

		return probDbn

	def getProb(self, ngram):
		prefix = ngram[:-1]
		probDbn = self.getProbDbn(prefix)
		if hasattr(probDbn, "__getitem__"):
			ind = self.vocab.getIndex(ngram[-1])
			return probDbn[ind]
		else:
			return probDbn

	def predict(self, prefix):
		probDbn = self.getProbDbn(prefix)
		predIndx = np.argmax(probDbn)
		return predIndx

	def predictTop(self, prefix, K=1):
		probDbn = self.getProbDbn(prefix)
		predIndx = np.argpartition(probDbn, -K)[-K:]
		return predIndx

	def predict_eval(self, ngram):
		prefix = ngram[:-1]
		predIndx = self.predict(prefix)
		if predIndx == self.vocab.getIndex(ngram[-1]):
			self.right_preds += 1
		self.total_preds += 1

	def logLikelihood_eval(self, ngram):
		self.logLik += np.log2(self.getProb(ngram))

	def crossEntropy(self):
		return - self.logLik / self.total_preds

	def getPredAccuracy(self):
		return self.right_preds * 1.0 / self.total_preds

	def getPerplexity(self):
		return np.power(2.0, self.crossEntropy())

	def timeDiff_eval(self, ngram):
		prefix = ngram[:-1]
		predIndx = self.predict(prefix)
		self.timeDiffs.append((predIndx, self.vocab.getIndex(ngram[-1])))

	def getTimeDiffs(self):
		return self.timeDiffs

	def evaluate(self, ngrams, time_eval=False):
		for ngram in ngrams:
			self.predict_eval(ngram)
			self.logLikelihood_eval(ngram)
			if time_eval:
				self.timeDiff_eval(ngram)


#############################
class unigramLM(object):
	def __init__(self, ngrams, vocab, alpha=1e-20):
		self.vocab = vocab
		self.alpha = alpha
		self.counts = Counter(ngrams)
		self.ngramCounts = self._buildTreeCounts()

	def _buildTreeCounts(self):
		V = self.vocab.get_num_words()
		counts = self.counts
		ngramCounts = np.zeros(V)
		# Counting
		for ngram in counts.keys():
			ind = self.vocab.getIndex(ngram[-1])
			ngramCounts[ind] += counts[ngram]
		return ngramCounts

	def updateTree(self, ngrams, dist, decay):
		ngramCounts = self.ngramCounts
		ngramCounts *= np.power(decay, dist)
		counts = Counter(ngrams)
		for ngram in counts.keys():
			ind = self.vocab.getIndex(ngram[-1])
			ngramCounts[ind] += counts[ngram]

	def updateCounts(self, ngrams, dist, decay):
		oldCounts = self.counts
		counts = Counter(ngrams)
		self.counts = updateCounts(oldCounts, counts, dist, decay)
		self.ngramCounts = self._buildTreeCounts()

	def getProbDbn(self):
		V = self.vocab.get_num_words()
		alpha = self.alpha
		probDbn = self.ngramCounts + alpha * V
		probDbn /= np.sum(probDbn)
		return probDbn


#############################
class model_results(object):
	def __init__(self, perp1, accu1, perp, accu, timeDiff1=None, timeDiff=None):
		self.perp = perp
		self.accu = accu
		self.perp1 = perp1
		self.accu1 = accu1
		self.timeDiff = timeDiff
		self.timeDiff1 = timeDiff1


#############################
class vocabulary(object):
	def __init__(self, words):
		self.wordList = words
		self.wordMap = self.buildIndex()

		assert(len(words) == len(set(words)))

		for i, w in enumerate(self.wordList):
			assert(i == self.getIndex(w))

		for w, i in self.wordMap.iteritems():
			assert(self.getWord(i) == w)

	def buildIndex(self):
		word_dict, indx = {}, 0
		for w in self.wordList:
			if w not in word_dict.keys():
				word_dict[w] = indx
				indx += 1
		return word_dict

	def getIndex(self, word):
		return self.wordMap[word]

	def getWord(self, indx):
		return self.wordList[indx]

	def get_num_words(self):
		return len(self.wordList)


#############################
def paths(D, cur=()):
	if isinstance(D, dict):
		for k, v in D.iteritems():
			for path in paths(v, cur + (k,)):
				yield path
	else:
		yield cur


def discountTreeLeafs(D, distance, decay_factor):
	if isinstance(D.values()[0], dict):
		for k, v in D.iteritems():
			discountTreeLeafs(v, distance, decay_factor)
	else:
		for k in D.keys():
			D[k] *= np.power(decay_factor, distance)


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

	def discount(self, distance=1, decay_factor=0.95):
		discountTreeLeafs(self.dict, distance, decay_factor)

	# Special function for probability calculation
	def normalize(self):
		Dict = self.dict
		Normalize(Dict)
