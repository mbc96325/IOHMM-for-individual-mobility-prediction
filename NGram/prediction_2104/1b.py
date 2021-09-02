from user import getDayOfWeekIndex
import oyster_reader as oyster

import random
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn import linear_model as lm
from sklearn import dummy
from sklearn.model_selection import cross_val_score as cvs

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


class individual_model(object):
	def __init__(self, user):
		self.user = user
		minD, maxD = user.getDayRange()
		self.dayRange = range(minD, maxD + 1)
		minW, maxW = user.getWeekRange()
		self.weekRange = range(minW, maxW + 1)

	def extractFeatures(self):
		features = []
		Y = []

		dailyTrips = self.user.getDailyTrips()
		for day in dailyTrips.keys():
			trips = dailyTrips[day]
			n = len(trips)
			for i, trip in enumerate(trips):
				if i == n - 1:
					y = 1
				else:
					y = 0
				Y.append(y)

				# dow = getDayOfWeekIndex(trip.day)
				h = trip.getHour()
				o = trip.getO()
				d = trip.getD()
				features.append([i + 1, h, o, d])

		X = label_binarize_features(np.array(features))
		Y = np.array(Y)
		# print Y.shape[0], np.sum(Y), self.user.getActiveDays()
		return X, Y

	def eval(self):
		X, Y = self.extractFeatures()
		clf0 = dummy.DummyClassifier()
		pred0 = cvs(clf0, X, Y, cv=3).mean()
		if Y.shape[0] - np.sum(Y) > 10:
			clf = lm.LogisticRegression(C=1.0)
			pred = cvs(clf, X, Y, cv=3).mean()
			return pred, pred0
		else:
			return pred0, pred0


def end_of_day_prediction(users):
	scores = []
	for u in users:
		model = individual_model(u)
		scores.append(model.eval())

	s = zip(*scores)
	print np.median(s[0]), np.median(s[1])
	'''
	plt.figure(figsize=(12, 6))
	plt.plot(s[2], s[0], 'bo', label='Logistic Regression')
	plt.plot(s[2], s[1], 'ro', label='Dummy Model')
	plt.xlabel('Travel Density (Number of Active Days Per Week)')
	plt.ylabel('Prediction Accuracy')
	plt.legend(loc='lower right')
	plt.savefig('../img/travel_prediction_plot.png', dpi=300)
	'''


if __name__ == "__main__":
	dataFile = "../data/oysterdata.csv"
	vocabFile = "../data/station_vocab.csv"
	users = oyster.readPanelData2(dataFile, vocabFile)
	freq_users = [u for u in users if u.getActiveDays() >= 60]
	print 'Number of users: {}'.format(len(freq_users))
	end_of_day_prediction(freq_users)
