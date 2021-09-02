import numpy as np
from collections import Counter
from functools import reduce
from ngramGen import ngramGenerator_baseline as ngramGenerator


def updateCounts(old_counter, new_counter, discount=0.9):
	keys = list(set(old_counter.keys() + new_counter.keys()))
	updated_counter = Counter({k: old_counter[k] * 0.9 + new_counter[k] for k in keys})
	return updated_counter


#############################
class mobilityNgram_baseline(object):
	def __init__(self, corpus, station_vocab, time_vocab, ID=None):
		ng = ngramGenerator(corpus)
		ngramTime1, ngramIn1, ngramOut1 = ng.getNGramsFirst()
		ngramTime, ngramIn, ngramOut = ng.getNGrams()
		self.station_vocab = station_vocab
		self.time_vocab = time_vocab

		self.lmTimeFirst = unigramLM(ngramTime1, time_vocab)
		self.lmInFirst = unigramLM(ngramIn1, station_vocab)
		self.lmOutFirst = ngramLM(ngramOut1, station_vocab)
		if len(ngramTime) > 0:
			self.lmTime = ngramLM(ngramTime, time_vocab)
			self.lmIn = ngramLM(ngramIn, station_vocab)
			self.lmOut = ngramLM(ngramOut, station_vocab)
		else:
			self.lmTime = ngramLM(ngramTime, time_vocab, n=2)
			self.lmIn = ngramLM(ngramIn, station_vocab, n=2)
			self.lmOut = ngramLM(ngramOut, station_vocab, n=2)

		if ID:
			self.id = ID
		else:
			self.id = None

	def update(self, corpus):
		if len(corpus) > 0:
			ng = ngramGenerator(corpus)
			ngramTime, ngramIn, ngramOut = ng.getNGrams()
			self.lmTime.update(ngramTime)
			self.lmIn.update(ngramIn)
			self.lmOut.update(ngramOut)
		else:
			self.lmTime.update([])
			self.lmIn.update([])
			self.lmOut.update([])

	def evaluate(self, corpus):
		ng = ngramGenerator(corpus)
		perp1 = self.perplexityFirst(ng)
		accu1 = self.predictionFirst(ng)
		timediff1 = self.timeDiffFirst(ng)
		rank1 = self.predRankFirst(ng)
		if len(ng.ngramTime) > 5:
			perp = self.perplexity(ng)
			accu = self.prediction(ng)
			timediff = self.timeDiff(ng)
			rank = self.predRank(ng)
		else:
			perp = None
			accu = None
			timediff = None
			rank = None
		result1 = model_result(perp1, accu1, timediff1, rank1)
		result2 = model_result(perp, accu, timediff, rank)
		return model_results(self.id, result1, result2)

	def perplexity(self, NG):
		ppTrip = self.perplexity_trip(NG)
		ngramTime, ngramIn, ngramOut = NG.getNGrams()
		ppT = self._calc_perplexity(ngramTime, self.lmTime)
		ppO = self._calc_perplexity(ngramIn, self.lmIn)
		ppD = self._calc_perplexity(ngramOut, self.lmOut)
		return ppTrip, ppT, ppO, ppD

	def perplexityFirst(self, NG):
		ppTrip = self.perplexity_trip_first(NG)
		ngramTime, ngramIn, ngramOut = NG.getNGramsFirst()
		ppT = self._calc_perplexity(ngramTime, self.lmTimeFirst)
		ppO = self._calc_perplexity(ngramIn, self.lmInFirst)
		ppD = self._calc_perplexity(ngramOut, self.lmOutFirst)
		return ppTrip, ppT, ppO, ppD

	def _calc_perplexity(self, ngrams, lm):
		N = len(ngrams)
		LLB = 0.0
		for w in ngrams:
			LLB += np.log2(lm.getProb(w))
		return pow(2.0, - LLB / N)

	def prediction(self, NG):
		acTrip = self.pred_eval_trip(NG)
		ngramTime, ngramIn, ngramOut = NG.getNGrams()
		acT = self._pred_eval(ngramTime, self.lmTime)
		acO = self._pred_eval(ngramIn, self.lmIn)
		acD = self._pred_eval(ngramOut, self.lmOut)
		return acTrip[0], acT, acO, acD, acTrip[1], acTrip[2], acTrip[3]

	def predictionFirst(self, NG):
		acTrip = self.pred_eval_trip_first(NG)
		ngramTime, ngramIn, ngramOut = NG.getNGramsFirst()
		acT = self._pred_eval(ngramTime, self.lmTimeFirst)
		acO = self._pred_eval(ngramIn, self.lmInFirst)
		acD = self._pred_eval(ngramOut, self.lmOutFirst)
		return acTrip[0], acT, acO, acD, acTrip[1], acTrip[2], acTrip[3]

	def _pred_eval(self, ngrams, lm):
		cor = 0
		for w in ngrams:
			cor += lm.predict_eval(w)
		return cor * 1.0 / len(ngrams)

	def predRank(self, NG):
		ngramT, ngramO, ngramD = NG.getNGrams()
		predRankT = self.lmTime.getPredRanks(ngramT)
		predRankO = self.lmIn.getPredRanks(ngramO)
		predRankD = self.lmOut.getPredRanks(ngramD)
		return predRankT, predRankO, predRankD

	def predRankFirst(self, NG):
		ngramT, ngramO, ngramD = NG.getNGramsFirst()
		predRankT = self.lmTimeFirst.getPredRanks(ngramT)
		predRankO = self.lmInFirst.getPredRanks(ngramO)
		predRankD = self.lmOutFirst.getPredRanks(ngramD)
		return predRankT, predRankO, predRankD

	def timeDiff(self, NG):
		ngramTime = NG.getNGrams('T')
		return self.lmTime.getTimeDiff(ngramTime)

	def timeDiffFirst(self, NG):
		ngramTime = NG.getNGramsFirst('T')
		return self.lmTimeFirst.getTimeDiff(ngramTime)

	def perplexity_trip(self, NG):
		ngramTime, ngramIn, ngramOut = NG.getNGrams()
		N = len(ngramTime)
		LLB = 0.0
		for i in range(N):
			LLB += np.log2(self.lmTime.getProb(ngramTime[i]))
			LLB += np.log2(self.lmIn.getProb(ngramIn[i]))
			LLB += np.log2(self.lmOut.getProb(ngramOut[i]))
		return pow(2.0, -LLB / N)

	def perplexity_trip_first(self, NG):
		ngramTime, ngramIn, ngramOut = NG.getNGramsFirst()
		N = len(ngramTime)
		LLB = 0.0
		for i in range(N):
			LLB += np.log2(self.lmTimeFirst.getProb(ngramTime[i]))
			LLB += np.log2(self.lmInFirst.getProb(ngramIn[i]))
			LLB += np.log2(self.lmOutFirst.getProb(ngramOut[i]))
		return pow(2.0, -LLB / N)

	def pred_eval_trip(self, NG):
		ac = np.zeros(4)
		ngramTime, ngramIn, ngramOut = NG.getNGrams()
		N = len(ngramTime)
		for i in range(N):
			prevT, T = ngramTime[i]
			prevD = ngramIn[i][0]
			O, D = ngramOut[i]
			prevTrip = (prevT, prevD)
			trueTrip = (T, O, D)
			pred = self.predictTrip(prevTrip, trueTrip)
			for j in range(4):
				ac[j] += pred[j]
		return ac / N

	def pred_eval_trip_first(self, NG):
		ac = np.zeros(4)
		ngramTime, ngramIn, ngramOut = NG.getNGramsFirst()
		N = len(ngramTime)
		for i in range(N):
			T = ngramTime[i][0]
			O, D = ngramOut[i]
			trueTrip = (T, O, D)
			pred = self.predictTripFirst(trueTrip)
			for j in range(4):
				ac[j] += pred[j]
		return ac / N

	def predictTrip(self, prevTrip, trueTrip, Ks=(1, 5, 10)):
		timeK, inK, outK = Ks
		K = timeK * inK * outK
		probs = np.zeros(K)
		indices = np.zeros((K, 3), dtype=np.int)
		prevT, prevD = prevTrip
		trueT, trueO, trueD = trueTrip
		timePrefix = (prevT,)
		predTimeIndx = self.lmTime.predictTop(timePrefix, timeK)
		predTimeProb = self.lmTime.getProbDbn(timePrefix)[predTimeIndx]
		for ti, tindx in enumerate(predTimeIndx):
			tProb = predTimeProb[ti]
			# t = self.time_vocab.getWord(tindx)
			inPrefix = (prevD,)
			predInIndx = self.lmIn.predictTop(inPrefix, inK)
			predInProb = self.lmIn.getProbDbn(inPrefix)[predInIndx]
			for oi, oindx in enumerate(predInIndx):
				oProb = predInProb[oi]
				o = self.station_vocab.getWord(oindx)
				outPrefix = (o,)
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

		acTrip = 0
		acT = 0
		acO = 0
		acD = 0
		if predT == trueT and predO == trueO and predD == trueD:
			acTrip = 1
		if predT == trueT:
			acT = 1
		if predO == trueO:
			acO = 1
		if predD == trueD:
			acD = 1
		return acTrip, acT, acO, acD

	def predictTripFirst(self, trueTrip, Ks=(1, 5, 10)):
		timeK, inK, outK = Ks
		K = timeK * inK * outK
		probs = np.zeros(K)
		indices = np.zeros((K, 3), dtype=np.int)
		trueT, trueO, trueD = trueTrip
		predTimeIndx = self.lmTimeFirst.predictTop(timeK)
		predTimeProb = self.lmTimeFirst.getProbDbn()[predTimeIndx]
		for ti, tindx in enumerate(predTimeIndx):
			tProb = predTimeProb[ti]
			# t = self.time_vocab.getWord(tindx)
			predInIndx = self.lmInFirst.predictTop(inK)
			predInProb = self.lmInFirst.getProbDbn()[predInIndx]
			for oi, oindx in enumerate(predInIndx):
				oProb = predInProb[oi]
				o = self.station_vocab.getWord(oindx)
				outPrefix = (o,)
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

		acTrip = 0
		acT = 0
		acO = 0
		acD = 0
		if predT == trueT and predO == trueO and predD == trueD:
			acTrip = 1
		if predT == trueT:
			acT = 1
		if predO == trueO:
			acO = 1
		if predD == trueD:
			acD = 1
		return acTrip, acT, acO, acD


#############################
class ngramLM(object):
	def __init__(self, ngrams, vocab, priorLM=None, n=None, alpha=1e-3):
		if len(ngrams) > 0:
			self.n = len(ngrams[0])
			assert self.n > 1
			self.vocab = vocab
			self.alpha = alpha
			self.counts = Counter(ngrams)
			# used for probability calculation
			self.ngramCounts = self._count()
		else:
			assert n is not None
			self.n = n
			self.ngramCounts = treeDict(self.n - 1)
		# archive calculated probabilities, for performance
		self.probs = treeDict(self.n - 1)

	def _count(self):
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

	def update(self, ngrams):
		oldCounts = self.counts
		newCounts = Counter(ngrams)
		self.counts = updateCounts(oldCounts, newCounts)
		self.ngramCounts = self._count()
		self.probs = treeDict(self.n - 1)

	def getProbDbn(self, prefix):
		if self.probs.hasKey(prefix):
			return self.probs.getValue(prefix)

		alpha = self.alpha
		V = self.vocab.get_num_words()
		probDbn = np.zeros(V)

		if self.ngramCounts.hasKey(prefix):
			probDbn += self.ngramCounts.getValue(prefix)

		probDbn += alpha

		probDbn /= np.sum(probDbn)
		self.probs.insert(prefix, probDbn)

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
		# predIndx = np.random.choice(self.vocab.get_num_words(), 1, p=probDbn)
		return predIndx

	def predictTop(self, prefix, K=1):
		probDbn = self.getProbDbn(prefix)
		predIndx = np.argpartition(probDbn, -K)[-K:]
		return predIndx

	def predict_eval(self, ngram):
		prefix = ngram[:-1]
		predIndx = self.predict(prefix)
		if predIndx == self.vocab.getIndex(ngram[-1]):
			return 1
		else:
			return 0

	def predict_rank_eval(self, ngram):
		prefix = ngram[:-1]
		probDbn = self.getProbDbn(prefix)
		pred_indices = np.argsort(-probDbn)
		true_index = self.vocab.getIndex(ngram[-1])
		return list(pred_indices).index(true_index)

	def getPredRanks(self, ngrams):
		predRanks = []
		for w in ngrams:
			predRanks.append(self.predict_rank_eval(w))
		return predRanks

	def logLikelihood(self, ngrams):
		logLik = 0.0
		for w in ngrams:
			logLik += np.log2(self.getProb(w))
		return logLik

	def crossEntropy(self, ngrams):
		return - self.logLikelihood(ngrams) / len(ngrams)

	def predAccuracy(self, ngrams):
		rightPred = 0
		for w in ngrams:
			rightPred += self.predict_eval(w)
		predAccuracy = rightPred * 1.0 / len(ngrams)
		return predAccuracy

	def perplexity(self, ngrams):
		return pow(2.0, self.crossEntropy(ngrams))

	def timeDiff(self, ngram):
		prefix = ngram[:-1]
		predIndx = self.predict(prefix)
		return (predIndx, self.vocab.getIndex(ngram[-1]))

	def getTimeDiff(self, ngrams):
		timeDiffs = []
		for w in ngrams:
			timeDiffs.append(self.timeDiff(w))
		return timeDiffs


#############################
class unigramLM(object):
	def __init__(self, ngrams, vocab, alpha=1e-3):
		self.vocab = vocab
		self.alpha = alpha
		self.counts = Counter(ngrams)
		self.ngramCounts = self._learnParas()
		self.probs = None

	def _learnParas(self):
		V = self.vocab.get_num_words()
		counts = self.counts
		ngramCounts = np.zeros(V)
		# Counting
		for ngram in counts.keys():
			ind = self.vocab.getIndex(ngram[-1])
			ngramCounts[ind] += counts[ngram]
		return ngramCounts

	def update(self, ngrams):
		oldCounts = self.counts
		newCounts = Counter(ngrams)
		self.counts = updateCounts(oldCounts, newCounts)
		self.ngramCounts = self._learnParas()
		self.probs = None

	def getProbDbn(self):
		if self.probs is not None:
			return self.probs

		alpha = self.alpha
		probDbn = self.ngramCounts + alpha
		probDbn /= np.sum(probDbn)
		self.probs = probDbn
		return probDbn

	def getProb(self, ngram):
		probDbn = self.getProbDbn()
		if hasattr(probDbn, "__getitem__"):
			ind = self.vocab.getIndex(ngram[-1])
			return probDbn[ind]
		else:
			return probDbn

	def predict(self):
		probDbn = self.getProbDbn()
		predIndx = np.argmax(probDbn)
		# predIndx = np.random.choice(self.vocab.get_num_words(), 1, p=probDbn)
		return predIndx

	def predictTop(self, K=1):
		probDbn = self.getProbDbn()
		predIndx = np.argpartition(probDbn, -K)[-K:]
		return predIndx

	def predict_eval(self, ngram):
		predIndx = self.predict()
		if predIndx == self.vocab.getIndex(ngram[-1]):
			return 1
		else:
			return 0

	def predict_rank_eval(self, ngram):
		probDbn = self.getProbDbn()
		pred_indices = np.argsort(-probDbn)
		true_index = self.vocab.getIndex(ngram[-1])
		return list(pred_indices).index(true_index)

	def getPredRanks(self, ngrams):
		predRanks = []
		for w in ngrams:
			predRanks.append(self.predict_rank_eval(w))
		return predRanks

	def logLikelihood(self, ngrams):
		logLik = 0.0
		for w in ngrams:
			logLik += np.log2(self.getProb(w))
		return logLik

	def crossEntropy(self, ngrams):
		return - self.logLikelihood(ngrams) / len(ngrams)

	def predAccuracy(self, ngrams):
		rightPred = 0
		for w in ngrams:
			rightPred += self.predict_eval(w)
		predAccuracy = rightPred * 1.0 / len(ngrams)
		return predAccuracy

	def perplexity(self, ngrams):
		return pow(2.0, self.crossEntropy(ngrams))

	def timeDiff(self, ngram):
		predIndx = self.predict()
		return (predIndx, self.vocab.getIndex(ngram[-1]))

	def getTimeDiff(self, ngrams):
		timeDiffs = []
		for w in ngrams:
			timeDiffs.append(self.timeDiff(w))
		return timeDiffs


#############################
class model_results(object):
	def __init__(self, ID, result1, result2):
		self.id = ID
		self.result1 = result1
		self.result2 = result2


class model_result(object):
	def __init__(self, perp, accu, timeDiff=None, predRank=None):
		self.perp = perp
		self.accu = accu
		self.timeDiff = timeDiff
		self.predRank = predRank


#############################
class vocabulary(object):
	def __init__(self, words):
		self.wordList = words
		self.wordMap = self.buildIndex()

		assert(len(words) == len(set(words)))

		for i, w in enumerate(self.wordList):
			assert(i == self.getIndex(w))

		for w, i in self.wordMap.items():
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
		for k, v in D.items():
			for path in paths(v, cur + (k,)):
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
		for i in range(self.levels):
			if path[i] in Dict.keys():
				Dict = Dict[path[i]]
			else:
				return False
		return True

	def insert(self, keys, value):
		path = self.getPath(keys)
		assert len(path) == self.levels
		Dict = self.dict
		for i in range(self.levels - 1):
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
