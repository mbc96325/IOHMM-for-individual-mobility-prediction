import numpy as np
from collections import Counter
from scipy.special import digamma
from functools import reduce

from ngramGen import ngramGenerator
import pandas as pd

def getKeyByValue(D, v):
	return list(D.keys())[list(D.values()).index(v)]


def isInt(string):
	try:
		int(string)
		return True
	except ValueError:
		return False


def updateCounts(old_counter, new_counter, discount=0.9):
	keys = list(set(old_counter.keys() + new_counter.keys()))
	updated_counter = Counter({k: old_counter[k] * 0.9 + new_counter[k] for k in keys})
	return updated_counter


def alpha_update(s_init, m, c):
	p = (c + 0.1) / np.sum(c + 0.1)
	K = m.size
	# s_old = - (K - 1) / 2 / np.sum(np.multiply(m, (np.log(p) - np.log(m))))
	s_old = s_init

	for i in range(10):
		s_new = (K - 1) / ((K - 1) / s_old - digamma(s_old) +
			np.sum(np.multiply(m, digamma(s_old * m))) - np.sum(np.multiply(m, np.log(p))))
		if abs(s_new - s_old) < 1e-2:
			return s_new
		s_old = s_new

	s_new = s_old
	print(s_new)
	return s_new


#############################
class mobilityNgram(object):
	def __init__(self, corpus, station_vocab, time_vocab,
		priorLM=None, optimize=False, ID=None, Station_idx_dict = None, PART_INFO = False, USED_TOD_LIST = None):
		self.PART_INFO = PART_INFO
		if ID is not None and USED_TOD_LIST is not None:
			ng = ngramGenerator(corpus, PART_INFO = PART_INFO, USED_TOD_LIST = USED_TOD_LIST[int(ID)])
		else:
			ng = ngramGenerator(corpus, PART_INFO = PART_INFO)
		self.USED_TOD_LIST = USED_TOD_LIST
		ngramTime1, ngramIn1, ngramOut1 = ng.getNGramsFirst()
		ngramTime, ngramIn, ngramOut = ng.getNGrams()
		self.station_vocab = station_vocab
		self.time_vocab = time_vocab
		self.Station_idx_dict = Station_idx_dict
		if priorLM is None:
			self.lmTimeFirst = ngramLM(ngramTime1, time_vocab)
			self.lmInFirst = ngramLM(ngramIn1, station_vocab)
			self.lmOutFirst = ngramLM(ngramOut1, station_vocab)
			self.lmTime = ngramLM(ngramTime, time_vocab)
			self.lmIn = ngramLM(ngramIn, station_vocab)
			self.lmOut = ngramLM(ngramOut, station_vocab)
		else:
			self.lmTimeFirst = ngramLM(ngramTime1, time_vocab, priorLM.lmTimeFirst,
					timeSmoothing=True, optimize_paras=optimize, alpha=25.0, beta=0.6)
			self.lmInFirst = ngramLM(ngramIn1, station_vocab, priorLM.lmInFirst,
					timeSmoothing=True, optimize_paras=optimize, alpha=5.0, beta=0.6)
			self.lmOutFirst = ngramLM(ngramOut1, station_vocab, priorLM.lmOutFirst,
					timeSmoothing=True, optimize_paras=optimize, alpha=10.0, beta=0.6)
			if len(ngramTime) > 0:
				self.lmTime = ngramLM(ngramTime, time_vocab, priorLM.lmTime,
					timeSmoothing=True, optimize_paras=optimize, alpha=25.0, beta=0.6)
				self.lmIn = ngramLM(ngramIn, station_vocab, priorLM.lmIn,
					timeSmoothing=True, optimize_paras=optimize, alpha=5.0, beta=0.6)
				self.lmOut = ngramLM(ngramOut, station_vocab, priorLM.lmOut,
					timeSmoothing=True, optimize_paras=optimize, alpha=10.0, beta=0.6)
			else:
				self.lmTime = priorLM.lmTime
				self.lmIn = priorLM.lmIn
				self.lmOut = priorLM.lmOut

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



	def evaluate_baichuan(self, corpus):
		if self.USED_TOD_LIST is not None:
			ng = ngramGenerator(corpus, TEST_ = False, PART_INFO = self.PART_INFO, USED_TOD_LIST = self.USED_TOD_LIST[int(self.id)])
		else:
			ng = ngramGenerator(corpus, TEST_=False, PART_INFO = self.PART_INFO)
		#print(len(corpus))
		ngramTime, ngramIn, ngramOut = ng.getNGramsFirst()
		assert self.Station_idx_dict is not None
		u_id = self.id

		max_num_pred = 20
		Pred_col = []
		for k in range(max_num_pred):
			Pred_col.append('Predict'+str(k+1))

		results_save_time_first = {'Card_ID':[],'activity_index':[],'Predict_duration':[],'Pred_t':[],'Actual_t':[]}

		for w in ngramTime:
			pred_T = self.lmTimeFirst.predict_eval_baichuan(w, max_avai_num = 20)
			pred_T_time = int(pred_T[0])*3600 + 0.5*3600 # middle of interval
			results_save_time_first['Card_ID'].append(u_id)
			results_save_time_first['activity_index'].append(0)
			results_save_time_first['Predict_duration'].append(pred_T_time)
			results_save_time_first['Pred_t'].append(int(pred_T[0]))
			results_save_time_first['Actual_t'].append(int(w[-1]))
		results_time_first = pd.DataFrame(results_save_time_first)


		pred_O_all = []

		for w in ngramIn:
			pred_O = self.lmInFirst.predict_eval_baichuan(w, max_avai_num = 20)
			pred_O_MTR  = [self.Station_idx_dict[new_id] for new_id in pred_O]
			if len(pred_O_MTR) < 20:
				compensate = [-1] * (20 - len(pred_O_MTR))
				pred_O_MTR += compensate
			pred_O_all.append(pred_O_MTR)


		results_loc_first = pd.DataFrame(pred_O_all, columns=Pred_col)
		results_loc_first['Card_ID'] = u_id
		results_loc_first['activity_index'] = 0



		##############middle
		ngramTime, ngramIn, ngramOut = ng.getNGrams()

		results_save_time_middle = {'Card_ID':[],'activity_index':[],'Predict_duration':[],'Pred_t':[],'Actual_t':[]}

		for w in ngramTime:
			pred_T = self.lmTime.predict_eval_baichuan(w, max_avai_num = 20)
			pred_T_time = int(pred_T[0])*3600 + 0.5*3600 # middle of interval
			results_save_time_middle['Card_ID'].append(u_id)
			results_save_time_middle['activity_index'].append(-1)
			results_save_time_middle['Predict_duration'].append(pred_T_time)
			results_save_time_middle['Pred_t'].append(int(pred_T[0]))
			results_save_time_middle['Actual_t'].append(int(w[-1]))
		results_time_middle = pd.DataFrame(results_save_time_middle)


		pred_O_all = []

		for w in ngramIn:
			pred_O = self.lmIn.predict_eval_baichuan(w, max_avai_num = 20)
			pred_O_MTR  = [self.Station_idx_dict[new_id] for new_id in pred_O]
			if len(pred_O_MTR) < 20:
				compensate = [-1] * (20 - len(pred_O_MTR))
				pred_O_MTR += compensate
			pred_O_all.append(pred_O_MTR)


		results_loc_middle = pd.DataFrame(pred_O_all, columns=Pred_col)
		results_loc_middle['Card_ID'] = u_id
		results_loc_middle['activity_index'] = -1




		return results_time_first, results_loc_first, results_time_middle, results_loc_middle


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



	def predictionTop(self, corpus, K=1):
		ng = ngramGenerator(corpus)
		ngramTime, ngramIn, ngramOut = ng.getNGrams()
		acT = self.lmTime.predAccuracyTop(ngramTime, K)
		acIn = self.lmIn.predAccuracyTop(ngramIn, K)
		acOut = self.lmOut.predAccuracyTop(ngramOut, K)
		return acT, acIn, acOut

	def predictionTopEval(self, corpus, arrayK):
		acT, acIn, acOut = [], [], []
		for k in arrayK:
			t, o, d = self.predictionTop(corpus, k)
			acT.append(t)
			acIn.append(o)
			acOut.append(d)
		return acT, acIn, acOut

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
			prevTrip = ngramOut[i][:3]
			trueTrip = ngramOut[i][-3:]
			pred = self.predictTrip(prevTrip, trueTrip)
			for j in range(4):
				ac[j] += pred[j]
		return ac / N

	def pred_eval_trip_first(self, NG):
		ac = np.zeros(4)
		ngramTime, ngramIn, ngramOut = NG.getNGramsFirst()
		N = len(ngramTime)
		for i in range(N):
			dayFeatures = ngramTime[i][:-1]
			trueTrip = ngramOut[i][-3:]
			pred = self.predictTripFirst(dayFeatures, trueTrip)
			for j in range(4):
				ac[j] += pred[j]
		return ac / N

	def predictTrip(self, prevTrip, trueTrip, Ks=(10, 5, 10)):
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

	def predictTripFirst(self, dayFeatures, trueTrip, Ks=(10, 5, 10)):
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
	def __init__(self, ngrams, vocab, priorLM=None, n=None,
					timeSmoothing=False, optimize_paras=False,
					alpha=1.0, beta=0.6):
		if n:
			self.n = n
		else:
			self.n = len(ngrams[0])
		assert self.n > 1
		self.vocab = vocab
		if optimize_paras:
			self.alpha, self.beta = find_hyperparameters(ngrams, vocab, priorLM,
				alpha, beta, n_iter=5)
			# print self.alpha, self.beta
		else:
			self.alpha = alpha
			self.beta = beta
		self.timeSmoothing = timeSmoothing
		self.priorLM = priorLM
		self.backoffLM = self._buildBackoffLM(ngrams)
		self.counts = Counter(ngrams)
		# used for probability calculation
		self.ngramCounts = self._count()
		# archive calculated probabilities, for performance
		self.probs = treeDict(self.n - 1)

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
		return predIndx

	def predict_eval(self, ngram):
		prefix = ngram[:-1]
		predIndx = self.predict(prefix)
		if predIndx == self.vocab.getIndex(ngram[-1]):
			return 1
		else:
			return 0



	def predict_eval_baichuan(self, ngram, max_avai_num = 20):
		prefix = ngram[:-1]
		probDbn = self.getProbDbn(prefix)

		idx_prob = [(idx_,prob) for idx_,prob in enumerate(probDbn)]
		probDbn_sort = sorted(idx_prob,key=lambda x: x[1], reverse = True)

		pred_ = []

		for k in range(min(max_avai_num,len(probDbn))):
			idx_ = probDbn_sort[k][0]
			#print(self.vocab.wordList)
			pred_.append(self.vocab.getWord(idx_))
		return pred_

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

	def predictTop(self, prefix, K=1):
		probDbn = self.getProbDbn(prefix)
		predIndx = np.argpartition(probDbn, -K)[-K:]
		return predIndx

	def predictTop_eval(self, ngram, K=1):
		prefix = ngram[:-1]
		predIndx = self.predictTop(prefix, K)
		if self.vocab.getIndex(ngram[-1]) in predIndx:
			return 1
		else:
			return 0

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

	def predAccuracyTop(self, ngrams, N=1):
		rightPred = 0
		for w in ngrams:
			rightPred += self.predictTop_eval(w, N)
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
	def __init__(self, ngrams, vocab, alpha=1e-20):
		self.vocab = vocab
		self.alpha = alpha
		self.counts = Counter(ngrams)
		self.ngramCounts = self._learnParas()

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

	def getProbDbn(self):
		V = self.vocab.get_num_words()
		alpha = self.alpha
		probDbn = self.ngramCounts + alpha * V
		probDbn /= np.sum(probDbn)
		return probDbn


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
def find_hyperparameters(ngrams, vocab, priorLM, alpha, beta, n_iter=10):
	N = len(ngrams)
	if N < 10:
		return alpha, beta
	a = alpha
	b = beta
	indices = range(N)
	for i in range(n_iter):
		np.random.shuffle(indices)
		train_data = [ngrams[i] for i in indices[:-5]]
		valid_data = [ngrams[i] for i in indices[-5:]]
		da, db = gradient(train_data, valid_data, vocab, priorLM, a, b)
		a += da * 10 * np.power(0.9, i)
		b += db * 0.1 * np.power(0.9, i)
		if a < 1.0:
			a = 1.0
		if b > 0.95:
			b = 0.95
		if b < 0.05:
			b = 0.05
	return a, b


def gradient(train, valid, vocab, priorLM, alpha, beta):
	lm = ngramLM(train, vocab, priorLM, alpha=alpha, beta=beta)
	ll = lm.logLikelihood(valid)
	lm1 = ngramLM(train, vocab, priorLM, alpha=alpha+1.0, beta=beta)
	ll1 = lm1.logLikelihood(valid)
	lm2 = ngramLM(train, vocab, priorLM, alpha=alpha, beta=beta+0.1)
	ll2 = lm2.logLikelihood(valid)
	return ll1 - ll, (ll2 - ll) * 10


#############################
def paths(D, cur=()):
	if isinstance(D, dict):
		for k, v in D.items():
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
