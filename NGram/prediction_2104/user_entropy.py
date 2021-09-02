import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.colors as cl
import csv
import pandas as pd

import oyster_reader as oyster
import info_theory as it


matplotlib.use('Agg')


def top_n_probs(X, n):
	probs = sorted(it.probDbn(X), reverse=True)
	if len(probs) < n:
		return probs + [0] * (n - len(probs))
	else:
		return probs[:n]


def time_plot(users):
	T = []
	deltaT = []
	cycleT = []

	for u in users:
		T.extend(u.getTripTimeList())
		deltaT.extend(u.getInterTripTimeList())
		cycleT.extend(u.getRecycleTimeList())

	plt.figure(figsize=(10, 12))

	ax1 = plt.subplot(311)
	h1 = 3
	h2 = 27
	p, x = np.histogram(T, bins=range(h1 * 60, h2 * 60), density=True)
	plt.plot(x[1:], p * 60, '-')
	plt.xlim(h1, h2)
	xticks = [60 * i for i in range(h1, h2 + 1)]
	xlabels = range(h1, h2 + 1)
	plt.xticks(xticks, xlabels)
	plt.xlabel("Time of Day (in Hours)", fontsize=16)
	plt.ylabel("Probability", fontsize=16)
	plt.grid()

	ax2 = plt.subplot(312)
	h1 = 0
	h2 = 80
	p, x = np.histogram(deltaT, bins=range(h1 * 60, h2 * 60), density=True)
	plt.plot(x[1:], p * 60, '-')
	plt.xlim(h1, h2)
	xticks = [60 * i for i in range(h1, h2 + 1, 4)]
	xlabels = range(h1, h2 + 1, 4)
	plt.xticks(xticks, xlabels)
	plt.xlabel("Inter-Trip Time (in Hours)", fontsize=16)
	plt.ylabel("Probability", fontsize=16)
	plt.grid()

	ax3 = plt.subplot(313)
	h1 = 0
	h2 = 24 * 14
	p, x = np.histogram(cycleT, bins=range(h1 * 60, h2 * 60, 60), density=True)
	plt.plot(x[:-1] + (x[1] - x[0]) / 2, p * 60, '-')
	plt.xlim(h1, h2)
	xticks = [60 * i for i in range(h1, h2 + 1, 24)]
	xlabels = range(h1, h2 + 1, 24)
	plt.xticks(xticks, xlabels)
	plt.xlabel("Time between Trips with Same OD (in Hours)", fontsize=16)
	plt.ylabel("Probability", fontsize=16)
	plt.grid()

	plt.savefig('../img/time_plot.png', dpi=300)


def frequency_plot(users):
	oRank = []
	dRank = []
	tRank15 = []
	tRank30 = []
	tRank60 = []

	for u in users:
		oList = u.getStationList(stationType='in')
		dList = u.getStationList(stationType='out')
		tList15 = u.getTripTimeList(15)
		tList30 = u.getTripTimeList(30)
		tList60 = u.getTripTimeList(60)

		oRank.append(top_n_probs(oList, 60))
		dRank.append(top_n_probs(dList, 60))
		tRank15.append(top_n_probs(tList15, 60))
		tRank30.append(top_n_probs(tList30, 30))
		tRank60.append(top_n_probs(tList60, 15))

	o = np.median(np.array(oRank), axis=0)
	d = np.median(np.array(dRank), axis=0)
	t15 = np.median(np.array(tRank15), axis=0)
	t30 = np.median(np.array(tRank30), axis=0)
	t60 = np.median(np.array(tRank60), axis=0)

	plt.figure(figsize=(10, 8))

	ax1 = plt.subplot(121)
	ranks = range(1, 20 + 1)
	plt.plot(ranks, o[:20], c='blue', marker='o', ls='-', label='Entry Station')
	plt.plot(ranks, d[:20], c='red', marker='o', ls='-', label='Exit Station')
	plt.plot(ranks, t15[:20], c='black', marker='^', ls='-', label='Time (15 min)')
	plt.plot(ranks, t30[:20], c='black', marker='*', ls='-', label='Time (30 min)')
	plt.plot(ranks[:15], t60, c='black', marker='x', ls='-', label='Time (1 hour)')
	plt.xlabel('Rank', fontsize=16)
	plt.ylabel('Proportion (Median)', fontsize=16)
	plt.legend(loc='upper right', fontsize=14)
	plt.text(-0.15, 1, '(a)', fontdict={'size': 16, 'weight': 'bold'},
		transform=ax1.transAxes)
	plt.grid()

	ax2 = plt.subplot(122)
	plt.loglog(range(1, 60 + 1), o, c='blue', marker='o', ls='-', label='Entry Station')
	plt.plot(range(1, 60 + 1), d, c='red', marker='o', ls='-', label='Exit Station')
	plt.plot(range(1, 60 + 1), t15, c='black', marker='^', ls='-', label='Time (15 min)')
	plt.plot(range(1, 30 + 1), t30, c='black', marker='*', ls='-', label='Time (30 min)')
	plt.plot(range(1, 15 + 1), t60, c='black', marker='x', ls='-', label='Time (1 hour)')
	plt.xlabel('Rank', fontsize=16)
	plt.ylabel('Proportion (Median)', fontsize=16)
	plt.legend(loc='upper right', fontsize=14)
	plt.text(-0.15, 1, '(b)', fontdict={'size': 16, 'weight': 'bold'},
		transform=ax2.transAxes)
	plt.grid()

	plt.savefig('../img/frequency_plot.png', dpi=300)


def compute_entropy(users):
	ent = []
	for u in users:
		oList = u.getStationList(stationType='in')
		dList = u.getStationList(stationType='out')
		tList = u.getTripTimeList(30)
		# dtList = u.getInterTripTimeList(30)

		o_all = it.entropy(oList)
		d_all = it.entropy(dList)
		t_all = it.entropy(tList)
		# dt_all = it.entropy(dtList)
		ent.append([o_all, d_all, t_all, 'All'])

		o_first = it.entropy(u.getStationList(stationType='in', order=0))
		d_first = it.entropy(u.getStationList(stationType='out', order=0))
		t_first = it.entropy(u.getTripTimeList(30, order=0))
		ent.append([o_first, d_first, t_first, 'First of the Day'])

		o_last = it.entropy(u.getStationList(stationType='in', order=-1))
		d_last = it.entropy(u.getStationList(stationType='out', order=-1))
		t_last = it.entropy(u.getTripTimeList(30, order=-1))
		ent.append([o_last, d_last, t_last, 'Last of the Day'])

		o_weekend = it.entropy(u.getStationList(stationType='in', dayType='Weekend'))
		d_weekend = it.entropy(u.getStationList(stationType='out', dayType='Weekend'))
		t_weekend = it.entropy(u.getTripTimeList(30, dayType='Weekend'))
		ent.append([o_weekend, d_weekend, t_weekend, 'Weekend'])

	wt = csv.writer(open('../output/user_entropy.csv', 'wt'), delimiter=',')
	wt.writerow(['Entry Station', 'Exit Station', 'Time', 'Trip Type'])
	for e in ent:
		wt.writerow(e)


def entropy_plot():
	filepath = '../output/user_entropy.csv'

	df = pd.read_csv(filepath)

	df_long = pd.melt(df, 'Trip Type', var_name='Behavior Aspect', value_name='Entropy')
	print df_long

	import seaborn as sns

	sns.set_style('whitegrid')
	sns.set_context('poster')

	plt.figure(figsize=(10, 8))
	ax1 = plt.subplot(111)
	g = sns.factorplot(x='Behavior Aspect', hue='Trip Type', y='Entropy', data=df_long, kind='box', size=6, aspect=2, legend=False)
	plt.legend(loc='upper right', fontsize=14)

	'''
	ax1 = plt.subplot(121)
	sns.kdeplot(np.array(oEnt), c='blue', ls='-', label='Entry Station')
	sns.kdeplot(np.array(dEnt), c='red', ls='-', label='Exit Station')
	sns.kdeplot(np.array(tEnt), c='black', ls='-', label='Time (30 min)')
	plt.xlabel('Entropy', fontsize=16)
	plt.ylabel('Probability', fontsize=16)
	plt.ylim(0, 1)
	plt.xlim(0, 6)
	plt.text(-0.15, 1, '(a)', fontdict={'size': 16, 'weight': 'bold'},
		transform=ax1.transAxes)

	ax2 = plt.subplot(122)
	sns.kdeplot(np.array(oEnt1), c='blue', ls='-', label='Entry Station')
	sns.kdeplot(np.array(dEnt1), c='red', ls='-', label='Exit Station')
	sns.kdeplot(np.array(tEnt1), c='black', ls='-', label='Time (30 min)')
	plt.xlabel('Entropy', fontsize=16)
	plt.ylabel('Probability', fontsize=16)
	plt.ylim(0, 1)
	plt.xlim(0, 6)
	plt.text(-0.15, 1, '(b)', fontdict={'size': 16, 'weight': 'bold'},
		transform=ax2.transAxes)
	'''
	plt.show()


def entropy_plot2(users):
	ent = []
	ent1 = []
	ent2 = []

	for u in users:
		oList = u.getStationList(stationType='in')
		dList = u.getStationList(stationType='out')
		tList = u.getTripTimeList(30)
		dtList = u.getInterTripTimeList(30)

		o = it.entropy(oList)
		d = it.entropy(dList)
		t = it.entropy(tList)
		dt = it.entropy(dtList)

		oList1 = u.getStationList(stationType='in', order=0)
		dList1 = u.getStationList(stationType='out', order=0)
		tList1 = u.getTripTimeList(30, order=0)
		dtList1 = u.getInterTripTimeList(30, dtType='across')
		# print dtList1

		o1 = it.entropy(oList1)
		d1 = it.entropy(dList1)
		t1 = it.entropy(tList1)
		dt1 = it.entropy(dtList1)

		oList2 = u.getStationList(stationType='in', order='any but first')
		dList2 = u.getStationList(stationType='out', order='any but first')
		tList2 = u.getTripTimeList(30, order='any but first')
		dtList2 = u.getInterTripTimeList(30, dtType='within')
		# print dtList1

		o2 = it.entropy(oList2)
		d2 = it.entropy(dList2)
		t2 = it.entropy(tList2)
		dt2 = it.entropy(dtList2)

		ent.append([o, d, t, dt])
		ent1.append([o1, d1, t1, dt1])
		ent2.append([o2, d2, t2, dt2])

	medEnt = np.median(np.array(ent), axis=0)
	medEnt1 = np.median(np.array(ent1), axis=0)
	medEnt2 = np.median(np.array(ent2), axis=0)

	fig, ax = plt.subplots(figsize=(12, 6))

	index = np.arange(4)
	bar_width = 0.25

	opacity = 0.5

	rects1 = plt.bar(index, medEnt, bar_width,
		alpha=opacity, color='b', label='All')
	rects2 = plt.bar(index + bar_width, medEnt1, bar_width,
		alpha=opacity, color='r', label='First of the Day')
	rects3 = plt.bar(index + 2 * bar_width, medEnt2, bar_width,
		alpha=opacity, color='g', label='Rest of the Day')

	plt.xlabel('Behavior Aspect', fontsize=16)
	plt.ylabel('Entropy (Median)', fontsize=16)
	plt.xticks(index + 1.5 * bar_width, (r'$o_i$', r'$d_i$', r'$t_i$', r'$\Delta t_i$'), fontsize=20)
	plt.legend(loc='upper left', fontsize=14)
	plt.grid()

	plt.savefig('../img/entropy.png', dpi=300)
	plt.show()


def calc_nmi(list1, list2):
	n = len(list1)
	m = len(list2)
	nmi = np.zeros((n, m))
	for i in range(n):
		for j in range(i, m):
			nmi[i, j] = it.normalizedMutualInfo(list1[i], list2[j])
			nmi[j, i] = nmi[i, j]
	return nmi


def compute_mutualInfo(users):
	import random

	random.shuffle(users)
	users = users[:100]

	nmi0 = []
	nmi1 = []
	nmi2 = []

	counter = 0

	for u in users:
		oList = u.getStationList(stationType='in')
		dList = u.getStationList(stationType='out')
		tList = u.getTripTimeList(30)
		# dtList = u.getInterTripTimeList(30)

		dtList = [0] + u.getInterTripTimeList(30)

		for i in range(len(tList)):
			print tList[i], dtList[i]

		x0 = [oList, dList, tList, dtList]
		y0 = [oList, dList, tList, dtList]
		nmi0.append(np.ndarray.flatten(calc_nmi(x0, y0)))
		x1 = [oList[1:], dList[1:], tList[1:], dtList[1:]]
		y1 = [oList[:-1], dList[:-1], tList[:-1], dtList[:-1]]
		nmi1.append(np.ndarray.flatten(calc_nmi(x1, y1)))
		x2 = [oList[2:], dList[2:], tList[2:], dtList[2:]]
		y2 = [oList[:-2], dList[:-2], tList[:-2], dtList[:-2]]
		nmi2.append(np.ndarray.flatten(calc_nmi(x2, y2)))

		counter += 1
		print counter

		print nmi0[0].reshape((4, 4))
		print nmi1[0].reshape((4, 4))
		print nmi2[0].reshape((4, 4))
	'''
	wt0 = csv.writer(open('../../output/nmi0.csv', 'wt'))
	for row in nmi0:
		wt0.writerow(list(row))
	wt1 = csv.writer(open('../../output/nmi1.csv', 'wt'))
	for row in nmi1:
		wt1.writerow(list(row))
	wt2 = csv.writer(open('../../output/nmi2.csv', 'wt'))
	for row in nmi2:
		wt2.writerow(list(row))
	'''


def mutualInfo_plot():
	nmi0 = np.genfromtxt('../output/nmi0.csv', delimiter=',')
	nmi1 = np.genfromtxt('../output/nmi1.csv', delimiter=',')
	nmi2 = np.genfromtxt('../output/nmi2.csv', delimiter=',')

	nmi0 = np.median(nmi0, axis=0).reshape((4, 4))
	nmi1 = np.median(nmi1, axis=0).reshape((4, 4))
	nmi2 = np.median(nmi2, axis=0).reshape((4, 4))
	nmi = [nmi0, nmi1, nmi2]

	fig, axes = plt.subplots(nrows=1, ncols=3)
	norm = cl.Normalize(vmin=0, vmax=1)

	for i in range(3):
		ax = axes.flat[i]
		data = nmi[i]
		im = ax.imshow(data, interpolation='nearest', cmap='hot', norm=norm)
		ax.set_aspect('equal')
		ax.set_yticks(range(4))
		ax.set_yticklabels([r'$o_i$', r'$d_i$', r'$t_i$', r'$\Delta t_i$'])

		ax.xaxis.tick_top()
		ax.set_xticks(range(4))

		ax.xaxis.set_label_position('top')
		if i == 0:
			ax.set_xticklabels([r'$o_i$', r'$d_i$', r'$t_i$', r'$\Delta t_i$'])
		elif i == 1:
			ax.set_xticklabels([r'$o_{i-1}$', r'$d_{i-1}$', r'$t_{i-1}$', r'$\Delta t_{i-1}$'])
		elif i == 2:
			ax.set_xticklabels([r'$o_{i-2}$', r'$d_{i-2}$', r'$t_{i-2}$', r'$\Delta t_{i-2}$'])

	fig.subplots_adjust(right=0.85)
	cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
	fig.colorbar(im, cax=cbar_ax)
	plt.savefig('../img/mutual_info_plot.png', dpi=300)


if __name__ == "__main__":
	dataFile = "../data/oysterdata.csv"
	vocabFile = "../data/station_vocab.csv"

	users = oyster.readPanelData2(dataFile, vocabFile)
	for u in users:
		if u.getActiveDays() < 60:
			users.pop(u)
	print 'Number of users: {}'.format(len(users))
	entropy_plot2(users)
	frequency_plot(users)
	# compute_mutualInfo(users)
	# mutualInfo_plot()
