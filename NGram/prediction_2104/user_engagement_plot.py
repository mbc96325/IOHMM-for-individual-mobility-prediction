import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


colors = ["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099"]
labels = ['P1A', 'P1B']
df = pd.read_csv('../output/user_engagement_prediction.csv')
print df['1a_lr'].corr(df['1b_lr'])
'''
coef = np.genfromtxt('../output/p1a_coefficients.csv', delimiter=',')
print np.mean(coef, axis=0)
print np.std(coef, axis=0)
print np.median(coef, axis=0)
print np.percentile(coef, 75, axis=0) - np.percentile(coef, 25, axis=0)
'''

def density_plot(df):
	sns.set(font_scale=1.5)
	sns.set_style("whitegrid", {"legend.frameon": True})
	fig, ax = plt.subplots(figsize=(16, 8))
	sns.kdeplot(df['1a_lr'], ax=ax, shade=True, color=colors[0], label=labels[0])
	sns.kdeplot(df['1b_lr'], ax=ax, shade=True, color=colors[1], label=labels[1])

	meda = df['1a_lr'].median()
	medb = df['1b_lr'].median()
	plt.axvline(meda, color=colors[0], linestyle='dashed', linewidth=2)
	plt.axvline(medb, color=colors[1], linestyle='dashed', linewidth=2)
	plt.text(meda - 0.01, 4.8, 'Median = {}%'.format(round(meda * 100, 1)),
		horizontalalignment='right', verticalalignment='center',
		fontsize=16, color=colors[0])
	plt.text(medb + 0.01, 4.8, 'Median = {}%'.format(round(medb * 100, 1)),
		horizontalalignment='left', verticalalignment='center',
		fontsize=16, color=colors[1])

	plt.xlim(0.4, 1.0)
	ax.set_xticklabels([str(i) + '%' for i in range(40, 101, 10)])
	plt.xlabel('Prediction Accuracy', fontsize=18)
	plt.ylabel('Probability Density (Users)', fontsize=18)
	plt.legend(fontsize=18, loc='upper left')
	plt.savefig('../img/user_engagement_dbn.png', dpi=300)


def scatter_plot(df):
	fig, ax = plt.subplots(figsize=(16, 8))
	sns.kdeplot(df['1a_lr'], df['1b_lr'], shade=True, cmap='Reds')
	plt.xlim(0.4, 1)
	plt.ylim(0.4, 1)
	plt.xlabel('Problem 1A')
	plt.ylabel('Problem 1B')
	plt.savefig('../img/user_engagement_scatter.png', dpi=300)


density_plot(df)
