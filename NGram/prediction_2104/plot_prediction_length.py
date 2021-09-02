import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../output/next_trip_prediction.csv')
print(list(df.columns.values))
print(df.shape)

rd = csv.reader(open('../output/user_entropy.csv', 'rU'), delimiter=',')
print rd.next()
users = {}
for row in rd:
    users[row[0]] = row[1:]

df['days'] = df['uid'].apply(lambda x: int(users[str(x)][0]))
df['trips'] = df['uid'].apply(lambda x: int(users[str(x)][2]))
df['t_ent'] = df['uid'].apply(lambda x: float(users[str(x)][5]))
df['o_ent'] = df['uid'].apply(lambda x: float(users[str(x)][3]))
df['d_ent'] = df['uid'].apply(lambda x: float(users[str(x)][4]))


def plot_prediction_length6(df, xlabel):
    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")
    plt.figure(figsize=(18, 12))
    ax1 = plt.subplot(2, 3, 1)
    ind = '(a)'
    col1 = 't_ent'
    col2 = 'acT1'
    label1 = xlabel
    label2 = 'P2A - Time'
    scatter_2d(ax1, ind, df, col1, col2, label1, label2)
    ax2 = plt.subplot(2, 3, 2)
    ind = '(b)'
    col1 = 'o_ent'
    col2 = 'acO1'
    label1 = xlabel
    label2 = 'P2A - Origin'
    scatter_2d(ax2, ind, df, col1, col2, label1, label2)
    ax3 = plt.subplot(2, 3, 3)
    ind = '(c)'
    col1 = 'd_ent'
    col2 = 'acD1'
    label1 = xlabel
    label2 = 'P2A - Destination'
    scatter_2d(ax3, ind, df, col1, col2, label1, label2)
    ax4 = plt.subplot(2, 3, 4)
    ind = '(d)'
    col1 = 't_ent'
    col2 = 'acT'
    label1 = xlabel
    label2 = 'P2B - Time'
    scatter_2d(ax4, ind, df, col1, col2, label1, label2, True)
    ax5 = plt.subplot(2, 3, 5)
    ind = '(e)'
    col1 = 'o_ent'
    col2 = 'acO'
    label1 = xlabel
    label2 = 'P2B - Origin'
    scatter_2d(ax5, ind, df, col1, col2, label1, label2, True)
    ax6 = plt.subplot(2, 3, 6)
    ind = '(f)'
    col1 = 'd_ent'
    col2 = 'acD'
    label1 = xlabel
    label2 = 'P2B - Destination'
    scatter_2d(ax6, ind, df, col1, col2, label1, label2, True)
    plt.tight_layout()
    plt.savefig('../img/next_trip_prediction_length3.png', dpi=300)


def plot_prediction_length3(df, xlabel):
    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")
    plt.figure(figsize=(18, 6))
    ax1 = plt.subplot(1, 3, 1)
    ind = '(a)'
    col1 = 'trips'
    col2 = 't_ent'
    label1 = xlabel
    label2 = 'Time Entropy'
    scatter_2d(ax1, ind, df, col1, col2, label1, label2)
    ax2 = plt.subplot(1, 3, 2)
    ind = '(b)'
    col1 = 'trips'
    col2 = 'o_ent'
    label1 = xlabel
    label2 = 'Origin Entropy'
    scatter_2d(ax2, ind, df, col1, col2, label1, label2)
    ax3 = plt.subplot(1, 3, 3)
    ind = '(c)'
    col1 = 'trips'
    col2 = 'd_ent'
    label1 = xlabel
    label2 = 'Destination Entropy'
    scatter_2d(ax3, ind, df, col1, col2, label1, label2)
    plt.tight_layout()
    plt.savefig('../img/next_trip_prediction_length4.png', dpi=300)


def scatter_2d(ax, ind, df, col1, col2, label1, label2, filter=False, title=None):
    if filter:
        data1 = df[col1][df[col2] >= 0]
        data2 = df[col2][df[col2] >= 0]
    else:
        data1 = df[col1]
        data2 = df[col2]
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


# density_plot2(df)
# density_plot4(df)
# scatter_plot6(df)
# trip_prediction_eval_plot(df)
# prediction_comparison_plot(df)
# prediction_rank_plot()
plot_prediction_length6(df, xlabel='Entropy')
