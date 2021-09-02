import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


colors = ["#3366cc", "#dc3912", "#109618", "#990099", "#ff9900"]
labels = ['P2A', 'P2B']
df = pd.read_csv('../output/next_trip_time_diff.csv')
df['error'] = df['predT'] - df['trueT']
df['abs_error'] = df['error'].abs()
dfA = df.loc[df['type'] == 'A']
dfB = df.loc[df['type'] == 'B']
print dfA.shape, dfB.shape
print dfA['error'].mean(), dfB['error'].mean()
print dfA['abs_error'].mean(), dfB['abs_error'].mean()
# print dfA.groupby(['predT'])['abs_error'].mean()
# print dfB.groupby(['predT'])['abs_error'].mean()

df2 = pd.read_csv('../output/next_trip_time_diff_baseline.csv')
df2['error'] = df2['predT'] - df2['trueT']
df2['abs_error'] = df2['error'].abs()
dfA2 = df2.loc[df2['type'] == 'A']
dfB2 = df2.loc[df2['type'] == 'B']
print dfA2.shape, dfB2.shape
print dfA2['error'].mean(), dfB2['error'].mean()
print dfA2['abs_error'].mean(), dfB2['abs_error'].mean()


def linearBining(X):
	bins = [i - 20.5 for i in range(42)]
	p, edges = np.histogram(X, bins=bins, density=True)
	k = [i + 0.5 for i in edges[:-1]]
	return p, k


def density_plot():
	sns.set(font_scale=1.5)
	sns.set_style("whitegrid", {"legend.frameon": True})
	plt.figure(figsize=(16, 8))
	ax = plt.subplot(1, 1, 1)
	p1, k1 = linearBining(dfA['error'])
	p2, k2 = linearBining(dfB['error'])
	plt.plot(k1[15: 26], p1[15: 26], '-o', color=colors[0], label=labels[0],
		linewidth=3, markerfacecolor='w', markeredgecolor=colors[0], markeredgewidth=2)
	plt.plot(k2[15: 26], p2[15: 26], '-o', color=colors[1], label=labels[1],
		linewidth=3, markerfacecolor='w', markeredgecolor=colors[1], markeredgewidth=2)

	plt.xlim(-5, 5)
	plt.xlabel(r'$e=t_{predicted}-t_{true}$', fontsize=20)
	plt.ylabel('Trip Density', fontsize=16)
	plt.legend(fontsize=18, loc='upper left')

	plt.text(0.75, 0.95, r'$E[e]$',
		horizontalalignment='right', verticalalignment='center',
		fontdict={'size': 20, 'weight': 'bold'}, transform=ax.transAxes)

	plt.text(0.85, 0.95, '{}'.format(round(dfA['error'].mean(), 2)),
		horizontalalignment='right', verticalalignment='center',
		fontdict={'size': 18, 'weight': 'bold'}, color=colors[0], transform=ax.transAxes)

	plt.text(0.95, 0.95, '{}'.format(round(dfB['error'].mean(), 2)),
		horizontalalignment='right', verticalalignment='center',
		fontdict={'size': 18, 'weight': 'bold'}, color=colors[1], transform=ax.transAxes)

	plt.text(0.75, 0.85, r'$E[\mid e\mid]$',
		horizontalalignment='right', verticalalignment='center',
		fontdict={'size': 20, 'weight': 'bold'}, transform=ax.transAxes)

	plt.text(0.85, 0.85, '{}'.format(round(dfA['abs_error'].mean(), 2)),
		horizontalalignment='right', verticalalignment='center',
		fontdict={'size': 18, 'weight': 'bold'}, color=colors[0], transform=ax.transAxes)

	plt.text(0.95, 0.85, '{}'.format(round(dfB['abs_error'].mean(), 2)),
		horizontalalignment='right', verticalalignment='center',
		fontdict={'size': 18, 'weight': 'bold'}, color=colors[1], transform=ax.transAxes)

	plt.savefig('../img/next_trip_time_diff.png', dpi=300)


def bar_plot():
	sns.set(font_scale=1.5)
	sns.set_style("whitegrid", {"legend.frameon": True})
	plt.figure(figsize=(16, 8))
	ax1 = plt.subplot(1, 2, 1)
	single_bar_plot(dfA['error'], dfA2['error'], ax1, 'P2A', '(a)')
	ax2 = plt.subplot(1, 2, 2)
	single_bar_plot(dfB['error'], dfB2['error'], ax2, 'P2B', '(b)')
	plt.savefig('../img/next_trip_time_diff_bar2.png', dpi=300)


def single_bar_plot(data1, data2, ax, title, fig_ind):
	p1, k1 = linearBining(data1)
	p2, k2 = linearBining(data2)

	index = np.arange(11) + 0.1
	bar_width = 0.4
	opacity = 0.5

	rects1 = plt.bar(index + bar_width / 2, p1[15: 26], bar_width,
		alpha=opacity, color=colors[0], label=r'$N$-Gram')
	rects2 = plt.bar(index + bar_width * 1.5, p2[15: 26], bar_width,
		alpha=opacity, color=colors[1], label='2-MC(1)')

	plt.xlim(0, 11)
	plt.ylim(0, 0.5)
	ticklabels = [str(i) + '%' for i in range(0, 51, 10)]
	ax.set_yticklabels(ticklabels)
	plt.xticks(index + bar_width, range(-5, 6), fontsize=18)
	plt.xlabel(r'$e=t_{predicted}-t_{true}$', fontsize=22)
	plt.ylabel('Probability (Trips)', fontsize=20)
	plt.legend(fontsize=20, loc='upper left')

	plt.text(0.70, 0.95, r'$E[e]$',
		horizontalalignment='right', verticalalignment='center',
		fontdict={'size': 20, 'weight': 'bold'}, transform=ax.transAxes)

	plt.text(0.84, 0.95, '{}'.format(round(data1.mean(), 2)),
		horizontalalignment='right', verticalalignment='center',
		fontdict={'size': 18, 'weight': 'bold'}, color=colors[0], transform=ax.transAxes)

	plt.text(0.98, 0.95, '{}'.format(round(data2.mean(), 2)),
		horizontalalignment='right', verticalalignment='center',
		fontdict={'size': 18, 'weight': 'bold'}, color=colors[1], transform=ax.transAxes)

	plt.text(0.70, 0.85, r'$E[|e|]$',
		horizontalalignment='right', verticalalignment='center',
		fontdict={'size': 20, 'weight': 'bold'}, transform=ax.transAxes)

	plt.text(0.84, 0.85, '{}'.format(round(data1.abs().mean(), 2)),
		horizontalalignment='right', verticalalignment='center',
		fontdict={'size': 18, 'weight': 'bold'}, color=colors[0], transform=ax.transAxes)

	plt.text(0.98, 0.85, '{}'.format(round(data2.abs().mean(), 2)),
		horizontalalignment='right', verticalalignment='center',
		fontdict={'size': 18, 'weight': 'bold'}, color=colors[1], transform=ax.transAxes)

	plt.text(-0.10, 1.05, fig_ind, fontdict={'size': 18, 'weight': 'bold'},
		horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

	plt.title(title)


bar_plot()
