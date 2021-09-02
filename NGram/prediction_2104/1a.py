import oyster_reader as oyster
from user import getDayOfWeekIndex, isHoliday, isSummer
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
			summer = isHoliday(d)
			features.append([dow, gap, lastDay, freq, summer, holiday])
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

	def eval(self):
		X, Y = self.extractDayFeatures()
		clf = lm.LogisticRegression(C=1.0)
		pred = cvs(clf, X, Y, cv=5).mean()
		clf0 = dummy.DummyClassifier()
		pred0 = cvs(clf0, X, Y, cv=5).mean()
		return pred, pred0, self.user.getActiveDays() * 1.0 / len(self.weekRange)


def active_day_prediction(users):
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
	active_day_prediction(freq_users)
