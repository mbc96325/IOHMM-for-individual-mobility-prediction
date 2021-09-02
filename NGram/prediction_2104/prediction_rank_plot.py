import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns


colors = ["#3366cc", "#dc3912", "#109618", "#990099", "#ff9900"]
labels = [r'$t$', r'$o$', r'$d$']
df = pd.read_csv('../output/next_trip_pred_ranks.csv')
df0 = pd.read_csv('../output/next_trip_pred_ranks_baseline.csv')
print df.shape, df0.shape


def count_by_rank(data, columns, max_rank=20):
	m = max_rank
	n = len(columns)
	counts = np.zeros((m, n))
	for j, col in enumerate(columns):
		d = data[col].value_counts().to_dict()
		for i in range(m):
			counts[i, j] = d[i]
			if i > 0:
				counts[i, j] += counts[i - 1, j]
	counts /= data.shape[0]
	return counts


def prediction_rank_plot(df, df0):
	columns = ['rankT', 'rankO', 'rankD']
	countsA = count_by_rank(df.loc[df['type'] == 'A'], columns)
	countsB = count_by_rank(df.loc[df['type'] == 'B'], columns)
	countsA0 = count_by_rank(df0.loc[df['type'] == 'A'], columns)
	countsB0 = count_by_rank(df0.loc[df['type'] == 'B'], columns)

	sns.set(font_scale=1.5)
	sns.set_style("whitegrid", {"legend.frameon": True})
	plt.figure(figsize=(16, 8))

	ax1 = plt.subplot(1, 2, 1)
	for i in range(3):
		plt.plot(range(1, 21), countsA[:, i], '-o', color=colors[i], label=labels[i] + r' ($N$-Gram)')
		plt.plot(range(1, 21), countsA0[:, i], '--', color=colors[i], label=labels[i] + ' (2-MC(1))')
	plt.xlim(0, 21)
	plt.ylim(0.4, 1.05)
	plt.xticks(range(0, 21, 5), range(0, 21, 5))
	plt.yticks([i / 100.0 for i in range(40, 101, 10)], [str(i) + '%' for i in range(40, 101, 10)])
	plt.xlabel(r'Prediction Rank ($k$)', fontsize=16)
	plt.ylabel('Cumulative Probability', fontsize=16)
	plt.title('P2A', fontsize=18)
	plt.legend(loc='lower right', fontsize=20)
	plt.text(-0.15, 1.05, '(a)', fontdict={'size': 18, 'weight': 'bold'},
		horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

	ax2 = plt.subplot(1, 2, 2)
	for i in range(3):
		plt.plot(range(1, 21), countsB[:, i], '-o', color=colors[i], label=labels[i] + r' ($N$-Gram)')
		plt.plot(range(1, 21), countsB0[:, i], '--', color=colors[i], label=labels[i] + ' (2-MC(1))')
	plt.xlim(0, 21)
	plt.ylim(0.4, 1.05)
	plt.xticks(range(0, 21, 5), range(0, 21, 5))
	plt.yticks([i / 100.0 for i in range(40, 101, 10)], [str(i) + '%' for i in range(40, 101, 10)])
	plt.xlabel(r'Prediction Rank ($k$)', fontsize=16)
	plt.ylabel('Cumulative Probability', fontsize=16)
	plt.title('P2B', fontsize=18)
	plt.legend(loc='lower right', fontsize=20)
	plt.text(-0.15, 1.05, '(b)', fontdict={'size': 18, 'weight': 'bold'},
		horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

	plt.savefig('../img/next_trip_prediction_rank2.png', dpi=300)


prediction_rank_plot(df, df0)
