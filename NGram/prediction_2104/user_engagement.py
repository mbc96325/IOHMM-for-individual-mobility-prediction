import oyster_reader as oyster
from user import getDayOfWeekIndex, isHoliday, isSummer
import csv
import random
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn import linear_model as lm
from sklearn import dummy
from sklearn.model_selection import cross_val_predict as cvp
from sklearn.metrics import accuracy_score, f1_score

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


random.seed(111)


def label_binarize_features(X):
	n, k = X.shape
	for i in xrange(k):
		lbin = LabelBinarizer()
		transformed_X = lbin.fit_transform(X[:, i])
		if transformed_X.shape[1] == 1:
			if len(lbin.classes_) > 1:
				transformed_X = np.concatenate((1 - transformed_X, transformed_X), axis=1)
		if i == 0:
			new_X = transformed_X
		else:
			new_X = np.concatenate((new_X, transformed_X), axis=1)
	return new_X


def cross_entropy(labels, probs):
	logLik = 0.0
	n = labels.size()
	for i in range(n):
		w = labels[i]
		logLik += np.log2(probs[i, w])
	return -logLik / n


class individual_model(object):
	def __init__(self, user):
		self.user = user
		minD, maxD = user.getDayRange()
		self.dayRange = range(minD, maxD + 1)
		minW, maxW = user.getWeekRange()
		self.weekRange = range(minW, maxW + 1)

	def extractDayFeatures(self):
		features = []
		Y = []
		dowLB = LabelBinarizer().fit(range(7))
		activeDays = self.user.getActiveDayList()
		nextInd = 1
		gap = 0
		streak = 1
		for d in self.dayRange[1:]:
			dow = getDayOfWeekIndex(d)
			lastDay = (streak > 0)
			if len(Y) < 20:
				freq = sum(Y) * 1.0 / max(1, len(Y))
			else:
				freq = sum(Y[-20:]) * 1.0 / 20
			holiday = isHoliday(d)
			# summer = isSummer(d)
			features.append([dow, gap, lastDay, freq, holiday])
			if d == activeDays[nextInd]:
				Y.append(1)
				streak += 1
				gap = 0
				nextInd += 1
			else:
				Y.append(0)
				gap += 1
				streak = 0
		features = np.array(features)
		X = np.concatenate((dowLB.transform(features[:, 0]), features[:, 1:]), axis=1)
		return X, np.array(Y)

	def active_day_prediction_eval(self):
		X, Y = self.extractDayFeatures()
		clf = lm.LogisticRegression(C=1.0)
		pred = cvp(clf, X, Y, cv=3)
		proba = cvp(clf, X, Y, cv=3, method='predict_proba')
		clf0 = dummy.DummyClassifier()
		pred0 = cvp(clf0, X, Y, cv=3)
		proba0 = cvp(clf0, X, Y, cv=3, method='predict_proba')
		ac = accuracy_score(Y, pred)
		ac0 = accuracy_score(Y, pred0)
		f1 = f1_score(Y, pred)
		f10 = f1_score(Y, pred0)
		ce = cross_entropy(Y, proba)
		ce0 = cross_entropy(Y, proba0)

		clf.fit(X, Y)
		coef = list(clf.coef_[0])

		return ac, ac0, f1, f10, ce, ce0, coef

	def extractTripFeatures(self):
		features = []
		Y = []
		dailyTrips = self.user.getDailyTrips()
		for day in dailyTrips.keys():
			trips = dailyTrips[day]
			n = len(trips)
			for i, trip in enumerate(trips):
				if i == n - 1:
					y = 0
				else:
					y = 1
				Y.append(y)

				# dow = getDayOfWeekIndex(trip.day)
				h = trip.getHour()
				o = trip.getO()
				d = trip.getD()
				features.append([i + 1, h, o, d])

		X = label_binarize_features(np.array(features))
		Y = np.array(Y)
		return X, Y

	def end_of_day_prediction_eval(self):
		X, Y = self.extractTripFeatures()
		if np.sum(Y) > 10:
			clf0 = dummy.DummyClassifier()
			pred0 = cvp(clf0, X, Y, cv=3)
			proba0 = cvp(clf0, X, Y, cv=3, method='predict_proba')
			clf = lm.LogisticRegression(C=1.0)
			pred = cvp(clf, X, Y, cv=3)
			proba = cvp(clf, X, Y, cv=3, method='predict_proba')
			ac = accuracy_score(Y, pred)
			ac0 = accuracy_score(Y, pred0)
			f1 = f1_score(Y, pred)
			f10 = f1_score(Y, pred0)
			ce = cross_entropy(Y, proba)
			ce0 = cross_entropy(Y, proba0)
			return ac, ac0, f1, f10, ce, ce0
		else:
			return None


def user_engagement_prediction_eval(users):
	print 'User Engagement Prediction'
	wt = csv.writer(open('../output/user_engagement_prediction.csv', 'wt'), delimiter=',')
	wt.writerow(['id', '1a_lr', '1a_dummy', '1b_lr', '1b_dummy'])
	wt2 = csv.writer(open('../output/p1a_coefficients.csv', 'wt'), delimiter=',')
	scores_1a = []
	scores_1b = []
	count = 0
	for u in users:
		model = individual_model(u)
		a = model.active_day_prediction_eval()
		b = model.end_of_day_prediction_eval()
		scores_1a.append(a[:-1])
		if b is not None:
			scores_1b.append(b)
		else:
			b = [-1.0, -1.0]
		wt.writerow([u.id, a[0], a[1], b[0], b[1]])
		wt2.writerow(a[-1])

		count += 1
		if count % 1000 == 0:
			print count

	s1a = zip(*scores_1a)
	s1b = zip(*scores_1b)

	print '1a Performance:'
	print 'Accuracy: {}, {}'.format(np.median(s1a[0]), np.median(s1a[1]))
	print 'F1 Score: {}, {}'.format(np.median(s1a[2]), np.median(s1a[3]))
	print 'Cross Entropy: {}, {}'.format(round(np.median(s1a[4]), 2), round(np.median(s1a[5]), 2))

	print '1b Performance:'
	print 'Accuracy: {}, {}'.format(np.median(s1b[0]), np.median(s1b[1]))
	print 'F1 Score: {}, {}'.format(np.median(s1b[2]), np.median(s1b[3]))
	print 'Cross Entropy: {}, {}'.format(round(np.median(s1b[4]), 2), round(np.median(s1b[5]), 2))


def plot_end_of_day_dbn(users):
	Ys = []
	for u in users:
		model = individual_model(u)
		X, Y = model.extractTripFeatures()
		Ys.append((Y.shape[0] - np.sum(Y)) * 1.0 / Y.shape[0])
		print Ys[-1]
	plt.figure(figsize=(12, 6))
	plt.hist(Ys, bins=10, normed=True)
	plt.savefig('../img/end_of_day_dbn.png', dpi=300)


if __name__ == "__main__":
	dataFile = "../data/oysterdata.csv"
	vocabFile = "../data/station_vocab.csv"
	users = oyster.readPanelData2(dataFile, vocabFile)
	freq_users = [u for u in users if u.getActiveDays() >= 60]
	print 'Number of users: {}'.format(len(freq_users))
	user_engagement_prediction_eval(freq_users)
	# plot_end_of_day_dbn(freq_users)
