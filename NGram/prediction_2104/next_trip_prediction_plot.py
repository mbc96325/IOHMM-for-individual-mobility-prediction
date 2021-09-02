import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns


colors = ["#3366cc", "#dc3912", "#109618", "#990099", "#ff9900"]
labels = [r'$t$', r'$o$', r'$d$']
df = pd.read_csv('../output/next_trip_prediction.csv')
print(list(df.columns.values))
print(df.shape)
df2 = pd.read_csv('../output/user_entropy.csv')
print(list(df2.columns.values))
print(df2.shape)
# df['ac1_e'] = df['acT1'] * df['acO1'] * df['acD1']
# df['ac_e'] = df['acT'] * df['acO'] * df['acD']
# print df.describe()


def density_plot2(df):
	sns.set(font_scale=1.8)
	sns.set_style("whitegrid", {"legend.frameon": True})
	plt.figure(figsize=(16, 8))
	ax1 = plt.subplot(1, 2, 1)
	cols1 = ['acT1', 'acO1', 'acD1']
	for i in range(3):
		data = df[cols1[i]]
		sns.kdeplot(data, ax=ax1, shade=True, color=colors[i], label=labels[i])
		med = data.median()
		plt.axvline(med, color=colors[i], linestyle='dashed', linewidth=2)
		if i == 1:
			plt.text(med + 0.01, 3.2, '{}%'.format(round(med * 100, 1)),
				horizontalalignment='left', verticalalignment='center',
				fontsize=18, color=colors[i])
		else:
			plt.text(med - 0.01, 3.2, '{}%'.format(round(med * 100, 1)),
				horizontalalignment='right', verticalalignment='center',
				fontsize=18, color=colors[i])

	plt.xlim(0, 1.0)
	plt.ylim(0, 3.4)
	ax1.set_xticklabels([str(i) + '%' for i in range(0, 101, 20)])
	plt.xlabel('Prediction Accuracy', fontsize=20)
	plt.ylabel('Probability Density (Users)', fontsize=20)
	plt.legend(fontsize=20, loc='upper left')
	plt.title('P2A')
	plt.text(-0.1, 1.05, '(a)', fontdict={'size': 18, 'weight': 'bold'},
		transform=ax1.transAxes)

	ax3 = plt.subplot(1, 2, 2)
	cols3 = ['acT', 'acO', 'acD']
	for i in range(3):
		data = df[cols3[i]][df[cols3[i]] >= 0]
		sns.kdeplot(data, ax=ax3, shade=True, color=colors[i], label=labels[i])
		med = data.median()
		plt.axvline(med, color=colors[i], linestyle='dashed', linewidth=2)
		if i == 1:
			plt.text(med + 0.01, 3.2, '{}%'.format(round(med * 100, 1)),
				horizontalalignment='left', verticalalignment='center',
				fontsize=18, color=colors[i])
		else:
			plt.text(med - 0.01, 3.2, '{}%'.format(round(med * 100, 1)),
				horizontalalignment='right', verticalalignment='center',
				fontsize=18, color=colors[i])

	plt.xlim(0, 1.0)
	plt.ylim(0, 3.4)
	ax3.set_xticklabels([str(i) + '%' for i in range(0, 101, 20)])
	plt.xlabel('Prediction Accuracy', fontsize=18)
	plt.ylabel('Probability Density (Users)', fontsize=18)
	plt.legend(fontsize=20, loc='upper left')
	plt.title('P2B')
	plt.text(-0.1, 1.05, '(b)', fontdict={'size': 18, 'weight': 'bold'},
		transform=ax3.transAxes)

	# plt.show()
	plt.savefig('../img/next_trip_prediction_dbn.png', dpi=300)


def density_plot4(df):
	sns.set(font_scale=1.5)
	sns.set_style("whitegrid", {"legend.frameon": True})
	plt.figure(figsize=(16, 16))
	ax1 = plt.subplot(2, 2, 1)
	cols1 = ['acT1', 'acO1', 'acD1']
	for i in range(3):
		data = df[cols1[i]]
		sns.kdeplot(data, ax=ax1, shade=True, color=colors[i], label=labels[i])
		med = data.median()
		plt.axvline(med, color=colors[i], linestyle='dashed', linewidth=2)
		if i == 1:
			plt.text(med + 0.01, 2.9, '{}%'.format(round(med * 100, 1)),
				horizontalalignment='left', verticalalignment='center',
				fontsize=16, color=colors[i])
		else:
			plt.text(med - 0.01, 2.9, '{}%'.format(round(med * 100, 1)),
				horizontalalignment='right', verticalalignment='center',
				fontsize=16, color=colors[i])

	plt.xlim(0, 1.0)
	plt.ylim(0, 3.0)
	ax1.set_xticklabels([str(i) + '%' for i in range(0, 101, 20)])
	plt.xlabel('Prediction Accuracy', fontsize=16)
	plt.ylabel('Probability Density (Users)', fontsize=16)
	plt.legend(fontsize=18, loc='upper left')
	plt.title('P2A')
	plt.text(-0.1, 1.05, '(a)', fontdict={'size': 18, 'weight': 'bold'},
		transform=ax1.transAxes)

	ax2 = plt.subplot(2, 2, 2)
	cols2 = ['ppT1', 'ppO1', 'ppD1']
	for i in range(3):
		data = np.log2(df[cols2[i]][(df[cols2[i]] >= 0)])
		# data2 = np.log2(data[(df[cols2[i]] < 100)])
		sns.kdeplot(data, ax=ax2, shade=True, color=colors[i], label=labels[i])
		med = data.median()
		plt.axvline(med, color=colors[i], linestyle='dashed', linewidth=2)
		#'''
		if i == 2:
			plt.text(med + 0.1, 0.67, '{}'.format(round(med, 2)),
				horizontalalignment='left', verticalalignment='center',
				fontsize=16, color=colors[i])
		else:
			plt.text(med - 0.1, 0.67, '{}'.format(round(med, 2)),
				horizontalalignment='right', verticalalignment='center',
				fontsize=16, color=colors[i])
		#'''
	plt.xlim(0, 10)
	plt.ylim(0, 0.7)
	plt.xlabel('Cross Entropy', fontsize=16)
	plt.ylabel('Probability Density (Users)', fontsize=16)
	plt.legend(fontsize=18, loc='upper right')
	plt.title('P2A')
	plt.text(-0.1, 1.05, '(b)', fontdict={'size': 18, 'weight': 'bold'},
		transform=ax2.transAxes)

	ax3 = plt.subplot(2, 2, 3)
	cols3 = ['acT', 'acO', 'acD']
	for i in range(3):
		data = df[cols3[i]][df[cols3[i]] >= 0]
		sns.kdeplot(data, ax=ax3, shade=True, color=colors[i], label=labels[i])
		med = data.median()
		plt.axvline(med, color=colors[i], linestyle='dashed', linewidth=2)
		if i == 1:
			plt.text(med + 0.01, 2.9, '{}%'.format(round(med * 100, 1)),
				horizontalalignment='left', verticalalignment='center',
				fontsize=16, color=colors[i])
		else:
			plt.text(med - 0.01, 2.9, '{}%'.format(round(med * 100, 1)),
				horizontalalignment='right', verticalalignment='center',
				fontsize=16, color=colors[i])

	plt.xlim(0, 1.0)
	plt.ylim(0, 3.0)
	ax3.set_xticklabels([str(i) + '%' for i in range(0, 101, 20)])
	plt.xlabel('Prediction Accuracy', fontsize=16)
	plt.ylabel('Probability Density (Users)', fontsize=16)
	plt.legend(fontsize=18, loc='upper left')
	plt.title('P2B')
	plt.text(-0.1, 1.05, '(c)', fontdict={'size': 18, 'weight': 'bold'},
		transform=ax3.transAxes)

	ax4 = plt.subplot(2, 2, 4)
	cols4 = ['ppT', 'ppO', 'ppD']
	for i in range(3):
		data = np.log2(df[cols4[i]][(df[cols4[i]] >= 0)])
		# data2 = data[(df[cols4[i]] < 100)]
		sns.kdeplot(data, ax=ax4, shade=True, color=colors[i], label=labels[i])
		med = data.median()
		plt.axvline(med, color=colors[i], linestyle='dashed', linewidth=2)
		#'''
		if i == 0:
			plt.text(med + 0.1, 0.67, '{}'.format(round(med, 2)),
				horizontalalignment='left', verticalalignment='center',
				fontsize=16, color=colors[i])
		else:
			plt.text(med - 0.1, 0.67, '{}'.format(round(med, 2)),
				horizontalalignment='right', verticalalignment='center',
				fontsize=16, color=colors[i])
		#'''
	plt.xlim(0, 10)
	plt.ylim(0, 0.7)
	plt.xlabel('Cross Entropy', fontsize=16)
	plt.ylabel('Probability Density (Users)', fontsize=16)
	plt.legend(fontsize=18, loc='upper right')
	plt.title('P2B')
	plt.text(-0.1, 1.05, '(d)', fontdict={'size': 18, 'weight': 'bold'},
		transform=ax4.transAxes)

	# plt.show()
	plt.savefig('../img/next_trip_prediction_dbn2.png', dpi=300)


def hist_plot6(df):
	plt.figure(figsize=(18, 12))
	cols6 = ['ppT1', 'ppO1', 'ppD1', 'ppT', 'ppO', 'ppD']
	for i in range(6):
		data = df[cols6[i]][(df[cols6[i]] >= 0) & (df[cols6[i]] < 100)]
		print data.shape
		plt.subplot(2, 3, i + 1)
		plt.hist(data, bins=100)
		plt.title(cols6[i])
	plt.savefig('../img/next_trip_perplexity_dbn.png', dpi=300)

indices = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']


def scatter_plot9(df):
	sns.set(font_scale=1.2)
	sns.set_style("whitegrid")
	plt.figure(figsize=(16, 16))
	ax1 = plt.subplot(3, 3, 1)
	ind = '(a)'
	col1 = 'acT1'
	col2 = 'acO1'
	label1 = 'Problem 2A - Time Prediction'
	label2 = 'Problem 2A - Origin Prediction'
	density_2d(ax1, ind, df, col1, col2, label1, label2)
	#print 'Subplot {} finished'.format(1)
	ax2 = plt.subplot(3, 3, 2)
	ind = '(b)'
	col1 = 'acT1'
	col2 = 'acD1'
	label1 = 'Problem 2A - Time Prediction'
	label2 = 'Problem 2A - Destination Prediction'
	density_2d(ax2, ind, df, col1, col2, label1, label2)
	#print 'Subplot {} finished'.format(2)
	ax3 = plt.subplot(3, 3, 3)
	ind = '(c)'
	col1 = 'acO1'
	col2 = 'acD1'
	label1 = 'Problem 2A - Origin Prediction'
	label2 = 'Problem 2A - Destination Prediction'
	density_2d(ax3, ind, df, col1, col2, label1, label2)
	#print 'Subplot {} finished'.format(3)
	ax4 = plt.subplot(3, 3, 4)
	ind = '(d)'
	col1 = 'acT'
	col2 = 'acO'
	label1 = 'Problem 2B - Time Prediction'
	label2 = 'Problem 2B - Origin Prediction'
	density_2d(ax4, ind, df, col1, col2, label1, label2, True)
	#print 'Subplot {} finished'.format(4)
	ax5 = plt.subplot(3, 3, 5)
	ind = '(e)'
	col1 = 'acT'
	col2 = 'acD'
	label1 = 'Problem 2B - Time Prediction'
	label2 = 'Problem 2B - Destination Prediction'
	density_2d(ax5, ind, df, col1, col2, label1, label2, True)
	#print 'Subplot {} finished'.format(5)
	ax6 = plt.subplot(3, 3, 6)
	ind = '(f)'
	col1 = 'acO'
	col2 = 'acD'
	label1 = 'Problem 2B - Origin Prediction'
	label2 = 'Problem 2B - Destination Prediction'
	density_2d(ax6, ind, df, col1, col2, label1, label2, True)
	#print 'Subplot {} finished'.format(6)
	ax7 = plt.subplot(3, 3, 7)
	ind = '(g)'
	col1 = 'acT1'
	col2 = 'acT'
	label1 = 'Problem 2A - Time Prediction'
	label2 = 'Problem 2B - Time Prediction'
	density_2d(ax7, ind, df, col1, col2, label1, label2, True)
	#print 'Subplot {} finished'.format(7)
	ax8 = plt.subplot(3, 3, 8)
	ind = '(h)'
	col1 = 'acO1'
	col2 = 'acO'
	label1 = 'Problem 2A - Origin Prediction'
	label2 = 'Problem 2B - Origin Prediction'
	density_2d(ax8, ind, df, col1, col2, label1, label2, True)
	#print 'Subplot {} finished'.format(8)
	ax9 = plt.subplot(3, 3, 9)
	ind = '(i)'
	col1 = 'acD1'
	col2 = 'acD'
	label1 = 'Problem 2A - Destination Prediction'
	label2 = 'Problem 2B - Destination Prediction'
	density_2d(ax9, ind, df, col1, col2, label1, label2, True)
	#print 'Subplot {} finished'.format(9)
	plt.tight_layout()
	plt.savefig('../img/next_trip_scatter.png', dpi=300)


def scatter_plot6(df):
	sns.set(font_scale=1.3)
	sns.set_style("whitegrid")
	plt.figure(figsize=(18, 12))
	ax1 = plt.subplot(2, 3, 1)
	ind = '(a)'
	col1 = 'acT1'
	col2 = 'acO1'
	label1 = 'P2A - Time Prediction'
	label2 = 'P2A - Origin Prediction'
	density_2d(ax1, ind, df, col1, col2, label1, label2)
	#print 'Subplot {} finished'.format(1)
	ax2 = plt.subplot(2, 3, 2)
	ind = '(b)'
	col1 = 'acT1'
	col2 = 'acD1'
	label1 = 'P2A - Time Prediction'
	label2 = 'P2A - Destination Prediction'
	density_2d(ax2, ind, df, col1, col2, label1, label2)
	#print 'Subplot {} finished'.format(2)
	ax3 = plt.subplot(2, 3, 3)
	ind = '(c)'
	col1 = 'acO1'
	col2 = 'acD1'
	label1 = 'P2A - Origin Prediction'
	label2 = 'P2A - Destination Prediction'
	density_2d(ax3, ind, df, col1, col2, label1, label2)
	#print 'Subplot {} finished'.format(3)
	ax4 = plt.subplot(2, 3, 4)
	ind = '(d)'
	col1 = 'acT'
	col2 = 'acO'
	label1 = 'P2B - Time Prediction'
	label2 = 'P2B - Origin Prediction'
	density_2d(ax4, ind, df, col1, col2, label1, label2, True)
	#print 'Subplot {} finished'.format(4)
	ax5 = plt.subplot(2, 3, 5)
	ind = '(e)'
	col1 = 'acT'
	col2 = 'acD'
	label1 = 'P2B - Time Prediction'
	label2 = 'P2B - Destination Prediction'
	density_2d(ax5, ind, df, col1, col2, label1, label2, True)
	#print 'Subplot {} finished'.format(5)
	ax6 = plt.subplot(2, 3, 6)
	ind = '(f)'
	col1 = 'acO'
	col2 = 'acD'
	label1 = 'P2B - Origin Prediction'
	label2 = 'P2B - Destination Prediction'
	density_2d(ax6, ind, df, col1, col2, label1, label2, True)
	#print 'Subplot {} finished'.format(6)
	plt.tight_layout()
	plt.savefig('../img/next_trip_scatter2.png', dpi=300)


def density_2d(ax, ind, df, col1, col2, label1, label2, filter=False):
	if filter:
		data1 = df[col1][df[col2] >= 0]
		data2 = df[col2][df[col2] >= 0]
	else:
		data1 = df[col1]
		data2 = df[col2]
	sns.kdeplot(data1, data2, ax=ax, shade=True, cmap='Blues')
	ticklabels = [str(i) + '%' for i in range(0, 101, 20)]
	ax.set_xticklabels(ticklabels)
	ax.set_yticklabels(ticklabels)
	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.xlabel(label1)
	plt.ylabel(label2)
	plt.text(-0.20, 1.0, ind, fontdict={'size': 18, 'weight': 'bold'},
		horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	cor = data1.corr(data2)
	plt.text(0.95, 0.05, 'correlation coeffient = {}'.format(round(cor, 2)),
		horizontalalignment='right', verticalalignment='center',
		fontsize=16, transform=ax.transAxes)
	print '{}, {}: {}'.format(col1, col2, cor)


def trip_prediction_eval_plot(df):
	sns.set(font_scale=1.5)
	sns.set_style("whitegrid", {"legend.frameon": True})
	plt.figure(figsize=(16, 8))
	ax1 = plt.subplot(1, 1, 1)
	data1 = df['ac1']
	data2 = df['ac'][df['ac'] >= 0]
	sns.kdeplot(data1, ax=ax1, shade=True, color=colors[0], label='P2A')
	sns.kdeplot(data2, ax=ax1, shade=True, color=colors[1], label='P2B')
	plt.axvline(data1.median(), color=colors[0], linestyle='dashed', linewidth=2)
	plt.axvline(data2.median(), color=colors[1], linestyle='dashed', linewidth=2)
	plt.text(data1.median() - 0.01, 2.6, '{}%'.format(round(data1.median() * 100, 1)),
		horizontalalignment='right', verticalalignment='center',
		fontsize=16, color=colors[0])
	plt.text(data2.median() + 0.01, 2.6, '{}%'.format(round(data2.median() * 100, 1)),
		horizontalalignment='left', verticalalignment='center',
		fontsize=16, color=colors[1])

	plt.xlim(0, 1.0)
	plt.ylim(0, 2.8)
	ax1.set_xticklabels([str(i) + '%' for i in range(0, 101, 20)])
	plt.xlabel('Prediction Accuracy', fontsize=16)
	plt.ylabel('Probability Density (Users)', fontsize=16)
	plt.legend(fontsize=16, loc='upper right')

	plt.savefig('../img/next_trip_prediction_dbn3.png', dpi=300)


def prediction_comparison_plot(df):

	sns.set(font_scale=1.5)
	sns.set_style("whitegrid", {"legend.frameon": True})
	plt.figure(figsize=(16, 8))
	gs = gridspec.GridSpec(1, 2)

	index = np.arange(3) + 0.2
	bar_width = 0.3
	opacity = 0.5

	ax1 = plt.subplot(gs[0, 0])
	data1 = np.median(df[['acT1_t', 'acO1_t', 'acD1_t']], axis=0)
	data2 = np.median(df[['acT1', 'acO1', 'acD1']], axis=0)
	rects1 = plt.bar(index, data2, bar_width,
		alpha=opacity, color=colors[2], label='Sequential Prediction')
	rects2 = plt.bar(index + bar_width, data1, bar_width,
		alpha=opacity, color=colors[4], label='Simultaneous Prediction')
	plt.xlim(0, 3)
	plt.ylim(0, 1)
	ticklabels = [str(i) + '%' for i in range(0, 101, 20)]
	ax1.set_yticklabels(ticklabels)
	plt.xlabel('Trip Attribute', fontsize=16)
	plt.ylabel('Median Prediction Accuracy', fontsize=16)
	plt.xticks(index + bar_width, (r'$t$', r'$o$', r'$d$'), fontsize=20)
	plt.legend(loc='upper left', fontsize=16)
	plt.text(-0.15, 1.05, '(a)', fontdict={'size': 18, 'weight': 'bold'},
		horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
	plt.title('P2A')

	ax2 = plt.subplot(gs[0, 1])
	data1 = np.median(df[['acT_t', 'acO_t', 'acD_t']], axis=0)
	data2 = np.median(df[['acT', 'acO', 'acD']][df['ppT'] > 0], axis=0)
	rects1 = plt.bar(index, data2, bar_width,
		alpha=opacity, color=colors[2], label='Sequential Prediction')
	rects2 = plt.bar(index + bar_width, data1, bar_width,
		alpha=opacity, color=colors[4], label='Simultaneous Prediction')
	plt.xlim(0, 3)
	plt.ylim(0, 1)
	ticklabels = [str(i) + '%' for i in range(0, 101, 20)]
	ax2.set_yticklabels(ticklabels)
	plt.xlabel('Trip Attribute', fontsize=16)
	plt.ylabel('Median Prediction Accuracy', fontsize=16)
	plt.xticks(index + bar_width, (r'$t$', r'$o$', r'$d$'), fontsize=20)
	plt.legend(loc='upper left', fontsize=16)
	plt.text(-0.15, 1.05, '(b)', fontdict={'size': 18, 'weight': 'bold'},
		horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
	plt.title('P2B')

	# plt.show()
	plt.savefig('../img/next_trip_prediction_dbn4.png', dpi=300)


def density_2d2(ax, ind, df, col1, col2, label1, label2, filter=False, title=None):
	if filter:
		data1 = df[col1][df[col2] >= 0]
		data2 = df[col2][df[col2] >= 0]
	else:
		data1 = df[col1]
		data2 = df[col2]
	sns.kdeplot(data1, data2, ax=ax, shade=True, cmap='Blues')
	ax.plot([0, 1], [0, 1], 'k--')
	ticklabels = [str(i) + '%' for i in range(0, 101, 20)]
	ax.set_xticklabels(ticklabels)
	ax.set_yticklabels(ticklabels)
	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.xlabel(label1)
	plt.ylabel(label2)
	if title:
		plt.title(title)
	plt.text(-0.15, 1.05, ind, fontdict={'size': 18, 'weight': 'bold'},
		horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	cor = data1.corr(data2)
	plt.text(0.95, 0.05, 'correlation coeffient = {}'.format(round(cor, 2)),
		horizontalalignment='right', verticalalignment='center',
		fontsize=16, transform=ax.transAxes)
	print '{}, {}: {}'.format(col1, col2, cor)


def plot_prediction_length(df1, df2, xcol, xlabel):
	sns.set(font_scale=1.3)
	sns.set_style("whitegrid")
	plt.figure(figsize=(18, 12))
	ax1 = plt.subplot(2, 3, 1)
	ind = '(a)'
	col1 = 't_ent'
	col2 = 'acT1'
	label1 = xlabel
	label2 = 'P2A - Time'
	scatter_2d(ax1, ind, df1, df2, col1, col2, label1, label2)
	ax2 = plt.subplot(2, 3, 2)
	ind = '(b)'
	col1 = 'o_ent'
	col2 = 'acO1'
	label1 = xlabel
	label2 = 'P2A - Origin'
	scatter_2d(ax2, ind, df1, df2, col1, col2, label1, label2)
	ax3 = plt.subplot(2, 3, 3)
	ind = '(c)'
	col1 = 'd_ent'
	col2 = 'acD1'
	label1 = xlabel
	label2 = 'P2A - Destination'
	scatter_2d(ax3, ind, df1, df2, col1, col2, label1, label2)
	ax4 = plt.subplot(2, 3, 4)
	ind = '(d)'
	col1 = 't_ent'
	col2 = 'acT'
	label1 = xlabel
	label2 = 'P2B - Time'
	scatter_2d(ax4, ind, df1, df2, col1, col2, label1, label2, True)
	ax5 = plt.subplot(2, 3, 5)
	ind = '(e)'
	col1 = 'o_ent'
	col2 = 'acO'
	label1 = xlabel
	label2 = 'P2B - Origin'
	scatter_2d(ax5, ind, df1, df2, col1, col2, label1, label2, True)
	ax6 = plt.subplot(2, 3, 6)
	ind = '(f)'
	col1 = 'd_ent'
	col2 = 'acD'
	label1 = xlabel
	label2 = 'P2B - Destination'
	scatter_2d(ax6, ind, df1, df2, col1, col2, label1, label2, True)
	plt.tight_layout()
	plt.savefig('../img/next_trip_prediction_length2.png', dpi=300)


def scatter_2d(ax, ind, df1, df2, col1, col2, label1, label2, filter=False, title=None):
	if filter:
		data1 = df1[col1][df2[col2] >= 0]
		data2 = df2[col2][df2[col2] >= 0]
	else:
		data1 = df1[col1]
		data2 = df2[col2]
	ax.plot(data1, data2, 'o')
	ticklabels = [str(i) + '%' for i in range(0, 101, 20)]
	ax.set_yticklabels(ticklabels)
	# plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.xlabel(label1)
	plt.ylabel(label2)
	if title:
		plt.title(title)
	plt.text(-0.15, 1.05, ind, fontdict={'size': 18, 'weight': 'bold'},
		horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	cor = data1.corr(data2)
	plt.text(0.95, 0.05, 'correlation coeffient = {}'.format(round(cor, 2)),
		horizontalalignment='right', verticalalignment='center',
		fontsize=16, transform=ax.transAxes)
	print '{}, {}: {}'.format(col1, col2, cor)


density_plot2(df)
# density_plot4(df)
# scatter_plot6(df)
# trip_prediction_eval_plot(df)
# prediction_comparison_plot(df)
# prediction_rank_plot()
# plot_prediction_length(df2, df, xcol='ent', xlabel='Entropy')
