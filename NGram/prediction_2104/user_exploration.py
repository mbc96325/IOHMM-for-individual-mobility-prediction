import math
import numpy as np
import oyster_reader as oyster
from matplotlib import pyplot as plt
import matplotlib.colors as cl
# import seaborn as sns


def plotInterTripTime(users):
	count = 0
	dtFreq = np.zeros(60 * 20)
	for u in users:
		interTripTime = u.getInterTripTime()
		for dt in interTripTime:
			dtFreq[dt] += 1
			count += 1
	dtProb = dtFreq * 1.0 / count

	plt.figure()
	plt.plot(dtProb)
	plt.xlim(0, 18)
	xticks = [60 * i for i in range(19)]
	xlabels = range(19)
	plt.xticks(xticks, xlabels)
	plt.xlabel("Inter-Trip Time (in hours)")
	plt.ylabel("Probability")
	plt.show()


def plotInterTripTimeByOrder(users):
	count = 0
	tFreq = np.zeros((4, 60 * 20))
	for u in users:
		dailyDict = u.getDailyTrips()
		for day, trips in dailyDict.iteritems():
			num_trips = len(trips)
			for i in range(1, num_trips):
				dt = trips[i].getTime() - trips[i - 1].getTime()
				order = min(i - 1, 3)
				tFreq[order, dt] += 1.0
				count += 1
	tProb = tFreq / count

	colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w']
	labels = ['1', '2', '3', '4', '5+']
	plt.figure(figsize=(14, 7))
	for i in xrange(1, 5):
		plt.plot(tProb[i - 1, :], colors[i] + '-', label=labels[i])
	plt.xlim(0, 18)
	xticks = [60 * i for i in range(19)]
	xlabels = range(19)
	plt.xticks(xticks, xlabels)
	plt.xlabel("Inter-Trip Time (in hours)")
	plt.ylabel("Probability")
	plt.legend(loc='upper right')
	plt.show()


def plotTripTime(users):
	count = 0
	tFreq = np.zeros(60 * 27)
	for u in users:
		tripTime = u.getTripTime()
		for dt in tripTime:
			tFreq[dt] += 1
			count += 1
	tProb = tFreq * 1.0 / count

	plt.figure()
	plt.plot(tProb)
	plt.xlim(3, 27)
	xticks = [60 * i for i in range(3, 28)]
	xlabels = range(3, 28)
	plt.xticks(xticks, xlabels)
	plt.xlabel("Time of Day")
	plt.ylabel("Probability")
	plt.show()


def plotTripTimeByOrder(users):
	count = 0
	tFreq = np.zeros((5, 60 * 27))
	for u in users:
		dailyDict = u.getDailyTrips()
		for day, trips in dailyDict.iteritems():
			num_trips = len(trips)
			for i in range(0, num_trips):
				order = min(i, 4)
				tFreq[order, trips[i].getTime()] += 1.0
				count += 1
	tProb = tFreq / count

	colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w']
	labels = ['1', '2', '3', '4', '5+']
	plt.figure(figsize=(14, 7))
	for i in xrange(5):
		plt.plot(tProb[i, :], colors[i] + '-', label=labels[i])
	plt.xlim(3, 27)
	xticks = [60 * i for i in range(3, 28)]
	xlabels = range(3, 28)
	plt.xticks(xticks, xlabels)
	plt.xlabel("Time of Day")
	plt.ylabel("Probability")
	plt.legend(loc='upper left')
	plt.show()


def plotNumStations(users):
	stNum = []
	odNum = []
	for u in users:
		stNum.append(len(u.getStationFreq()))
		odNum.append(len(u.getStationFreq(stationType='od')))
	plt.figure(figsize=(14, 7))
	plt.subplot(121)
	plt.hist(stNum, bins=np.linspace(0, 120, 10))
	plt.xlim(0, 120)
	plt.xlabel('Number of Stations')
	plt.ylabel('Frequency')

	plt.subplot(122)
	plt.plot(stNum, odNum, 'o')
	plt.xlabel('Number of Stations')
	plt.ylabel('Number of OD Pairs')
	plt.show()


def plotStationUsage(users):
	n = 20
	ranks = range(n + 1)
	oRank = []
	dRank = []
	for user in users:
		total = len(user.tripList)
		rankedOs = user.getStationRank(stationType='in')
		shares = [item[1] * 1.0 / total for item in rankedOs[:n]]
		shares = [0] + shares
		if len(shares) < (n + 1):
			shares = shares + [0] * (n + 1 - len(shares))
		dbn = np.cumsum(shares)
		oRank.append(dbn)

		rankedDs = user.getStationRank(stationType='out')
		shares = [item[1] * 1.0 / total for item in rankedDs[:n]]
		shares = [0] + shares
		if len(shares) < (n + 1):
			shares = shares + [0] * (n + 1 - len(shares))
		dbn = np.cumsum(shares)
		dRank.append(dbn)
	plt.figure(figsize=(10, 8))
	YO = np.median(np.array(oRank), axis=0)
	YD = np.median(np.array(dRank), axis=0)
	plt.plot(ranks, YO, c='blue', marker='o', ls='-', label='Entry Station')
	plt.plot(ranks, YD, c='red', marker='^', ls='--', label='Exit Station')
	plt.xlabel('Top-K Most Used Station', fontsize=16)
	plt.ylabel('Proportion (Median)', fontsize=16)
	plt.legend(loc='lower right', fontsize=14)
	plt.grid()
	plt.show()


def mostUsedStations(users):
	inProb = []
	outProb = []
	for u in users:
		nDays = u.getActiveDays()
		mus = u.getStationRank()[0][0]
		inCount = 0
		outCount = 0
		dailyDict = u.getDailyTrips()
		for day, trips in dailyDict.iteritems():
			if trips[0].inStation == mus:
				inCount += 1
			if trips[-1].outStation == mus:
				outCount += 1
			inProb.append(inCount * 1.0 / nDays)
			outProb.append(outCount * 1.0 / nDays)
	print np.median(inProb)
	print np.median(outProb)


def entropy(X):
	ent = 0
	N = len(X)
	labels = set(X)
	for label in labels:
		prob = float(X.count(label)) / N
		if prob > 0:
			ent -= prob * math.log(prob, 2)
	return ent


def gridPlot(users):
	stNum = []

	n = 20
	ranks = range(n + 1)
	oRank = []
	dRank = []

	count = 0
	tFreq = np.zeros(2 * 27)

	oEnt = []
	dEnt = []
	tEnt = []

	for u in users:
		stNum.append(len(set(u.getStationFreq())))

		total = len(u.tripList)
		rankedOs = u.getStationRank(stationType='in')
		oShares = [item[1] * 1.0 / total for item in rankedOs[:n]]
		if len(oShares) < n:
			oShares = oShares + [0] * (n - len(oShares))
		oRank.append(oShares)
		rankedDs = u.getStationRank(stationType='out')
		dShares = [item[1] * 1.0 / total for item in rankedDs[:n]]
		if len(dShares) < n:
			dShares = dShares + [0] * (n - len(dShares))
		dRank.append(dShares)

		tripTime = u.getDiscreteTripTime()
		for dt in tripTime:
			tFreq[dt] += 1
			count += 1

		oEnt.append(entropy(u.getStationList(stationType='in')))
		dEnt.append(entropy(u.getStationList(stationType='out')))
		tEnt.append(entropy(u.getDiscreteTripTime()))

	plt.figure(figsize=(14, 14))

	ax1 = plt.subplot(221)
	plt.hist(stNum, bins=np.linspace(0, 120, 10))
	plt.xlim(0, 120)
	plt.xlabel('Number of Stations (used in a Year)')
	plt.ylabel('Frequency')
	plt.text(-0.15, 1, '(a)', fontdict={'size': 16, 'weight': 'bold'},
		transform=ax1.transAxes)

	ax2 = plt.subplot(222)
	YO = np.median(np.array(oRank), axis=0)
	YD = np.median(np.array(dRank), axis=0)
	plt.plot(ranks[1:], YO, c='blue', marker='o', ls='-', label='Entry Station')
	plt.plot(ranks[1:], YD, c='red', marker='^', ls='--', label='Exit Station')
	plt.xlabel('The K-th Most Used Station')
	plt.ylabel('Proportion (Median)')
	plt.legend(loc='upper right', fontsize=12)
	plt.text(-0.15, 1, '(b)', fontdict={'size': 16, 'weight': 'bold'},
		transform=ax2.transAxes)

	ax3 = plt.subplot(223)
	tProb = tFreq * 1.0 / count
	plt.plot(tProb, 'k-')
	plt.xlim(3 * 2, 26 * 2)
	xticks = [i for i in range(3 * 2, 26 * 2, 2)]
	xlabels = range(3, 26)
	plt.xticks(xticks, xlabels)
	plt.xlabel("Hour of Day")
	plt.ylabel("Probability")
	plt.text(-0.15, 1, '(c)', fontdict={'size': 16, 'weight': 'bold'},
		transform=ax3.transAxes)

	ax4 = plt.subplot(224)
	left = range(1, 4)
	height = [np.median(tEnt), np.median(oEnt), np.median(dEnt)]
	plt.bar(left, height, align='center', color=['k', 'b', 'r'])
	xticks = range(1, 4)
	xlabels = ['Time', 'Entry Station', 'Exit Station']
	plt.xticks(xticks, xlabels)
	plt.ylabel("Entropy")
	plt.text(-0.15, 1, '(d)', fontdict={'size': 16, 'weight': 'bold'},
		transform=ax4.transAxes)

	plt.show()


def plotTripRate(users):
	userDays = 0
	trips = 0
	tRate = []
	for u in users:
		total = len(u.tripList)
		days = u.getActiveDays()
		tRate.append(total * 1.0 / days)
		userDays += days
		trips += total
	print userDays, trips
	plt.hist(tRate)
	plt.show()


def plotActiveDays(users):
	sortedUsers = sorted(users, key=lambda t: t.getActiveDays(), reverse=True)
	vector = np.array([u.getActiveDayVector() for u in sortedUsers])

	fig, ax = plt.subplots(figsize=(9, 3))
	norm = cl.Normalize(vmin=0, vmax=1)
	im = ax.imshow(vector.T, interpolation='nearest', cmap='hot')
	# ax.set_aspect(1. / ax.get_data_ratio())
	ax.set_xlabel('User')
	ax.set_ylabel('Day')
	plt.show()


def plotInterActiveDays(users):
	activeD = []
	deltaD = []
	for u in users:
		activeD.append(u.getActiveDays())
		deltaD.extend(u.getInterActiveDays())
	fig, axes = plt.subplots(figsize=(10, 5))
	plt.subplot(121)
	plt.hist(activeD, bins=range(60, 361, 15), normed=True)
	plt.xlabel('Number of Active Days')
	plt.ylabel('Probability (User)')
	plt.subplot(122)
	plt.hist(np.array(deltaD) - 1, bins=range(0, 20, 1), normed=True)
	plt.xlabel('Number of Days between Active Days')
	plt.ylabel('Probability (User Day)')
	plt.show()


if __name__ == "__main__":
	dataFile = "/Volumes/MobilitySyntax/data/sampleData_2013_reduced.csv"
	vocabFile = "/Volumes/MobilitySyntax/data/station_vocab.csv"
	users = oyster.readPanelData(dataFile, vocabFile)
	users = [u for u in users if u.getActiveDays > 60]
	# plotNumStations(users)
	# plotInterTripTimeByOrder(users)
	# plotTripTimeByOrder(users)
	# plotStationUsage(users)
	# mostUsedStations(users)
	# gridPlot(users)
	# plotTripRate(users)
	# plotActiveDays(users)
	plotInterActiveDays(users)
