import oyster_reader as oyster
import info_theory as it

import numpy as np
import csv
import pandas as pd
import time
import multiprocessing as mp

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as cl
import seaborn as sns


colors = ["#3366cc", "#dc3912", "#109618", "#990099", "#ff9900"]
labels = [r'$t$', r'$o$', r'$d$']
# df = pd.read_csv('../output/user_entropy.csv')


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

	# count = 0
	# tFreq = np.zeros((5, 60 * 27))

	S = []
	N = []

	oEnt = []
	dEnt = []
	tEnt = []

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

		S.append(len(set(u.getStationList())))
		dailyTrips = u.getDailyTrips()
		for trips in dailyTrips.values():
			N.append(len(trips))

		oList = u.getStationList(stationType='in')
		dList = u.getStationList(stationType='out')
		tList = u.getTripTimeList(60)

		o = it.entropy(oList)
		d = it.entropy(dList)
		t = it.entropy(tList)

		oList1 = u.getStationList(stationType='in', order=0)
		dList1 = u.getStationList(stationType='out', order=0)
		tList1 = u.getTripTimeList(60, order=0)

		o1 = it.entropy(oList1)
		d1 = it.entropy(dList1)
		t1 = it.entropy(tList1)

		oList2 = u.getStationList(stationType='in', order='any but first')
		dList2 = u.getStationList(stationType='out', order='any but first')
		tList2 = u.getTripTimeList(60, order='any but first')

		o2 = it.entropy(oList2)
		d2 = it.entropy(dList2)
		t2 = it.entropy(tList2)

		oEnt.append([o, o1, o2])
		dEnt.append([d, d1, d2])
		tEnt.append([t, t1, t2])
	'''
		dailyDict = u.getDailyTrips()
		for day, trips in dailyDict.iteritems():
			num_trips = len(trips)
			for i in range(0, num_trips):
				order = min(i, 4)
				tFreq[order, trips[i].getTime()] += 1.0
				count += 1

	tProb = tFreq / count
	'''
	o = np.median(np.array(oRank), axis=0)
	d = np.median(np.array(dRank), axis=0)
	t15 = np.median(np.array(tRank15), axis=0)
	t30 = np.median(np.array(tRank30), axis=0)
	t60 = np.median(np.array(tRank60), axis=0)

	med_oEnt = np.median(np.array(oEnt), axis=0)
	med_dEnt = np.median(np.array(dEnt), axis=0)
	med_tEnt = np.median(np.array(tEnt), axis=0)
	p75_oEnt = np.percentile(np.array(oEnt), 75, axis=0)
	p75_dEnt = np.percentile(np.array(dEnt), 75, axis=0)
	p75_tEnt = np.percentile(np.array(tEnt), 75, axis=0)
	p25_oEnt = np.percentile(np.array(oEnt), 25, axis=0)
	p25_dEnt = np.percentile(np.array(dEnt), 25, axis=0)
	p25_tEnt = np.percentile(np.array(tEnt), 25, axis=0)

	sns.set(font_scale=1.5)
	sns.set_style("whitegrid", {'axes.grid': False, "legend.frameon": True})
	plt.figure(figsize=(16, 12))

	ax1 = plt.subplot(2, 2, 1)
	hist, bins = np.histogram(S, range(0, 151, 10))
	p = hist.astype(np.float32) / len(S)
	w = bins[1] - bins[0]
	plt.bar(bins[:-1], p, width=w, align='edge', color=colors[0], edgecolor='w', alpha=0.8)
	# plt.hist(S, bins=range(0, 151, 10), normed=True, color=colors[0], edgecolor='w', alpha=0.8)
	plt.xlim(0, 150)
	# ax1.set_yticks([0.0, 0.005, 0.010, 0.015, 0.020])
	plt.xlabel('Number of Stations per User', fontsize=16)
	plt.ylabel('Probability', fontsize=16)
	plt.text(-0.1, 1.05, '(a)', fontdict={'size': 18, 'weight': 'bold'},
		transform=ax1.transAxes)

	ax2 = plt.subplot(2, 2, 2)
	n = 8 + 1
	plt.hist(N, bins=range(n), normed=True, color=colors[0], edgecolor='w', alpha=0.8)
	plt.xlim(1, n)
	plt.xlabel('Number of Trips per Active Day', fontsize=16)
	plt.ylabel('Probability', fontsize=16)
	plt.xticks([i + 0.5 for i in range(1, n)], range(1, n))
	plt.text(-0.1, 1.05, '(b)', fontdict={'size': 18, 'weight': 'bold'},
		transform=ax2.transAxes)

	ax3 = plt.subplot(2, 2, 3)
	plt.plot(range(1, 15 + 1), t60, c=colors[0], marker='o', ls='-', label=labels[0] + ' (1 hour)')
	plt.plot(range(1, 30 + 1), t30, c=colors[0], marker='*', ls='-', label=labels[0] + ' (30 min)')
	plt.plot(range(1, 60 + 1), t15, c=colors[0], marker='^', ls='-', label=labels[0] + ' (15 min)')
	plt.loglog(range(1, 60 + 1), o, c=colors[1], marker='o', ls='-', label=labels[1])
	plt.plot(range(1, 60 + 1), d, c=colors[2], marker='o', ls='-', label=labels[2])
	plt.xlim(1, 100)
	plt.ylim(1e-4, 1)
	plt.xlabel(r'Rank ($k$)', fontsize=16)
	plt.ylabel('Probability (Median)', fontsize=16)
	plt.legend(loc='lower left', fontsize=18)
	plt.text(-0.1, 1.05, '(c)', fontdict={'size': 18, 'weight': 'bold'},
		transform=ax3.transAxes)

	ax4 = plt.subplot(2, 2, 4)
	index = np.arange(3)
	bar_width = 0.25
	opacity = 0.5
	rects1 = plt.bar(index, med_tEnt, bar_width, align='edge',
		alpha=opacity, color=colors[0], label=labels[0])
	rects2 = plt.bar(index + bar_width, med_oEnt, bar_width, align='edge',
		alpha=opacity, color=colors[1], label=labels[1])
	rects3 = plt.bar(index + 2 * bar_width, med_dEnt, bar_width, align='edge',
		alpha=opacity, color=colors[2], label=labels[2])
	for ind in index:
		plt.plot([ind + 0.5 * bar_width] * 2, [p25_tEnt[ind], p75_tEnt[ind]], 'k-', linewidth=2)
		plt.plot([ind + 1.5 * bar_width] * 2, [p25_oEnt[ind], p75_oEnt[ind]], 'k-', linewidth=2)
		plt.plot([ind + 2.5 * bar_width] * 2, [p25_dEnt[ind], p75_dEnt[ind]], 'k-', linewidth=2)
		plt.plot([ind + 0.2 * bar_width, ind + 0.8 * bar_width], [p25_tEnt[ind]] * 2, 'k-', linewidth=2)
		plt.plot([ind + 0.2 * bar_width, ind + 0.8 * bar_width], [p75_tEnt[ind]] * 2, 'k-', linewidth=2)
		plt.plot([ind + 1.2 * bar_width, ind + 1.8 * bar_width], [p25_oEnt[ind]] * 2, 'k-', linewidth=2)
		plt.plot([ind + 1.2 * bar_width, ind + 1.8 * bar_width], [p75_oEnt[ind]] * 2, 'k-', linewidth=2)
		plt.plot([ind + 2.2 * bar_width, ind + 2.8 * bar_width], [p25_dEnt[ind]] * 2, 'k-', linewidth=2)
		plt.plot([ind + 2.2 * bar_width, ind + 2.8 * bar_width], [p75_dEnt[ind]] * 2, 'k-', linewidth=2)
	plt.xticks(index + 1.5 * bar_width, ('All Trips', 'First Trip of Day', 'All Except First Trip'), fontsize=14)
	plt.ylim(0, 6)
	plt.ylabel('Entropy', fontsize=16)
	plt.xlabel('Trip Type', fontsize=16)
	ax4.legend(fontsize=18, loc='upper right')
	plt.text(-0.1, 1.05, '(d)', fontdict={'size': 18, 'weight': 'bold'},
		transform=ax4.transAxes)

	plt.savefig('../img/frequency_plot.png', dpi=300)


def user_metrics(users):
	wt = csv.writer(open('../output/user_entropy.csv', 'wt'), delimiter=',')
	wt.writerow(['id', 'days', 'stations', 'trips', 'o_ent', 'd_ent', 't_ent', 'dt_ent'])
	for u in users:
		days = u.getActiveDays()
		stations = len(set(u.getStationList()))
		trips = len(u.getTripList())

		oList = u.getStationList(stationType='in')
		dList = u.getStationList(stationType='out')
		tList = u.getTripTimeList(30)
		dtList = u.getInterTripTimeList(30)

		o = it.entropy(oList)
		d = it.entropy(dList)
		t = it.entropy(tList)
		dt = it.entropy(dtList)

		wt.writerow([u.id, days, stations, trips, o, d, t, dt])


def plot_trip_rate(users):
	S = []
	N = []
	for u in users:
		S.append(len(set(u.getStationList())))
		dailyTrips = u.getDailyTrips()
		for trips in dailyTrips.values():
			N.append(len(trips))

	plt.figure(figsize=(12, 6))
	ax1 = plt.subplot(121)
	plt.hist(S, bins=range(0, 161, 20), normed=True)
	plt.xlim(0, 160)
	plt.xlabel('Number of stations per user')
	plt.ylabel('Proportion')
	plt.text(-0.1, 1.05, '(a)', fontdict={'size': 16, 'weight': 'bold'},
		transform=ax1.transAxes)

	ax2 = plt.subplot(122)
	n = 8 + 1
	plt.hist(N, bins=range(n), normed=True)
	plt.xlim(1, n)
	plt.xlabel('Number of Trips per Day')
	plt.ylabel('Proportion')
	plt.xticks([i + 0.5 for i in range(1, n)], range(1, n))
	plt.text(-0.1, 1.05, '(b)', fontdict={'size': 16, 'weight': 'bold'},
		transform=ax2.transAxes)
	plt.savefig('../img/trip_rate.png', dpi=300)


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
	# plt.show()


def calc_mi(list1, list2, symmetric=False):
	n = len(list1)
	m = len(list2)
	nmi = np.zeros((n, m))
	if symmetric:
		for i in range(n):
			for j in range(i, m):
				nmi[i, j] = it.mutualInfo(list1[i], list2[j])
				nmi[j, i] = nmi[i, j]
	else:
		for i in range(n):
			for j in range(m):
				nmi[i, j] = it.mutualInfo(list1[i], list2[j])
	return nmi


def calc_nmi(list1, list2, symmetric=False):
	n = len(list1)
	m = len(list2)
	nmi = np.zeros((n, m))
	if symmetric:
		for i in range(n):
			for j in range(i, m):
				nmi[i, j] = it.normalizedMutualInfo(list1[i], list2[j])
				nmi[j, i] = nmi[i, j]
	else:
		for i in range(n):
			for j in range(m):
				nmi[i, j] = it.normalizedMutualInfo(list1[i], list2[j])
	return nmi


def compute_mutualInfo_by_user(u):
	tList0 = []
	oList0 = []
	dList0 = []
	tList1_0 = []
	oList1_0 = []
	dList1_0 = []
	tList1_1 = []
	oList1_1 = []
	dList1_1 = []
	tList2_0 = []
	oList2_0 = []
	dList2_0 = []
	tList2_2 = []
	oList2_2 = []
	dList2_2 = []
	dailyTrips = u.getDailyTrips()
	for day, trips in dailyTrips.iteritems():
		n = len(trips)
		ts = [trips[i].getHour() for i in range(n)]
		os = [trips[i].getO() for i in range(n)]
		ds = [trips[i].getD() for i in range(n)]
		tList0.extend(ts)
		oList0.extend(os)
		dList0.extend(ds)
		if n > 1:
			tList1_0.extend(ts[:-1])
			oList1_0.extend(os[:-1])
			dList1_0.extend(ds[:-1])
			tList1_1.extend(ts[1:])
			oList1_1.extend(os[1:])
			dList1_1.extend(ds[1:])
		if n > 2:
			tList2_0.extend(ts[:-2])
			oList2_0.extend(os[:-2])
			dList2_0.extend(ds[:-2])
			tList2_2.extend(ts[2:])
			oList2_2.extend(os[2:])
			dList2_2.extend(ds[2:])
	x0 = [tList0, oList0, dList0]
	mi0 = np.ndarray.flatten(calc_mi(x0, x0, True))
	if len(tList1_0) > 20:
		x1_0 = [tList1_0, oList1_0, dList1_0]
		x1_1 = [tList1_1, oList1_1, dList1_1]
		mi1 = np.ndarray.flatten(calc_mi(x1_1, x1_0))
	else:
		mi1 = None
	if len(tList2_0) > 20:
		x2_0 = [tList2_0, oList2_0, dList2_0]
		x2_2 = [tList2_2, oList2_2, dList2_2]
		mi2 = np.ndarray.flatten(calc_mi(x2_2, x2_0))
	else:
		mi2 = None
	return mi0, mi1, mi2


def compute_mutualInfo(users):
	pool = mp.Pool(processes=20)
	results = pool.map(compute_mutualInfo_by_user, users)
	'''
	results = []
	for u in users:
		results.append(compute_user_nmi(u))
	'''
	wt0 = csv.writer(open('../output/mi0.csv', 'wt'))
	wt1 = csv.writer(open('../output/mi1.csv', 'wt'))
	wt2 = csv.writer(open('../output/mi2.csv', 'wt'))
	for r in results:
		mi0, mi1, mi2 = r
		wt0.writerow(list(mi0))
		if mi1 is not None:
			wt1.writerow(list(mi1))
		if mi2 is not None:
			wt2.writerow(list(mi2))


def mutualInfo_plot():
	mi0 = np.genfromtxt('../output/mi0.csv', delimiter=',')
	mi1 = np.genfromtxt('../output/mi1.csv', delimiter=',')
	mi2 = np.genfromtxt('../output/mi2.csv', delimiter=',')

	print mi0.shape, mi1.shape, mi2.shape

	mat0 = np.nanmedian(mi0, axis=0).reshape((3, 3))
	mat1 = np.nanmedian(mi1, axis=0).reshape((3, 3))
	mat2 = np.nanmedian(mi2, axis=0).reshape((3, 3))
	print mat0
	print mat1
	print mat2

	mat = [mat0, mat1]

	fig, axes = plt.subplots(nrows=1, ncols=2)
	norm = cl.Normalize(vmin=0, vmax=4)

	for i in range(2):
		ax = axes.flat[i]
		data = mat[i]
		im = ax.imshow(data, interpolation='nearest', cmap='hot', norm=norm)
		ax.set_aspect('equal')
		ax.set_yticks(range(3))
		ax.set_yticklabels([r'$t_{i}$', r'$o_{i}$', r'$d_{i}$'])

		ax.xaxis.tick_top()
		ax.set_xticks(range(3))

		ax.xaxis.set_label_position('top')
		if i == 0:
			ax.set_xticklabels([r'$t_{i}$', r'$o_{i}$', r'$d_{i}$'])
		elif i == 1:
			ax.set_xticklabels([r'$t_{i-1}$', r'$o_{i-1}$', r'$d_{i-1}$'])
#		else:
#		 	ax.set_xticklabels([r'$t_{i-2}$', r'$o_{i-2}$', r'$d_{i-2}$'])

		for x in range(3):
			for y in range(3):
				ax.text(y, x, str(round(data[x, y], 2)),
						horizontalalignment='center', verticalalignment='center',
						fontsize=14)

	fig.subplots_adjust(right=0.85)
	cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
	fig.colorbar(im, cax=cbar_ax)
	plt.savefig('../img/mutual_info_plot2.png', dpi=300)


def summary(users):
	n_userdays = 0
	n_trips = 0
	for u in users:
		n_userdays += u.getActiveDays()
		n_trips += len(u.getTripList())
	print 'Number of user-days: {}'.format(n_userdays)
	print 'Number of trips: {}'.format(n_trips)
	print 'Average trip rate: {}'.format(n_trips * 1.0 / n_userdays)


if __name__ == "__main__":
	start = time.time()
	dataFile = "../data/oysterdata.csv"
	vocabFile = "../data/station_vocab.csv"
	users = oyster.readPanelData2(dataFile, vocabFile)
	freq_users = [u for u in users if u.getActiveDays() >= 60]
	# print 'Number of users: {}'.format(len(freq_users))
	# summary(freq_users)
	# user_metrics(freq_users)
	# plot_trip_rate(freq_users)
	frequency_plot(freq_users)
	# entropy_plot()
	# compute_mutualInfo(freq_users)
	# mutualInfo_plot()
	print 'Running Time: {} Seconds'.format(str(time.time() - start))
