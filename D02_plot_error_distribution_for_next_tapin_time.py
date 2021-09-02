import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os


# colors = ["#3366cc", "#dc3912", "#109618", "#990099", "#ff9900"]
colors = sns.color_palette('muted')
labels = ['Remaining', 'First','Last']
model = ['IOHMM','Base-LR','Base-LSTM']

def linearBining(X):
    X_new = X // 1800
    label, counts = np.unique(X_new,return_counts=True)
    return label, counts



def bar_plot(dfA,dfA2,dfA3,dfB,dfB2,dfB3, Out_put_name, save_fig):
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid", {"legend.frameon": True})
    plt.figure(figsize=(14, 7))
    ax1 = plt.subplot(1, 2, 1)
    single_bar_plot(dfA['error'], dfA2['error'],dfA3['error'], ax1, 'First activities', '(a)')
    ax2 = plt.subplot(1, 2, 2)
    single_bar_plot(dfB['error'], dfB2['error'], dfB3['error'], ax2, 'Remaining activities', '(b)')
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/' + Out_put_name, dpi=200)
    else:
        plt.show()


def single_bar_plot(data1, data2, data3, ax, title, fig_ind):
    label1, counts1 = linearBining(data1)
    label2, counts2 = linearBining(data2)
    label3, counts3 = linearBining(data3)

    index = label1
    bar_width = 0.3
    opacity = 0.7
    p1 = counts1 / sum(counts1)
    p2 = counts2 / sum(counts2)
    p3 = counts3 / sum(counts3)

    first_N = 9
    label1 = label1[:first_N]
    p1 = p1[:first_N]
    p2 = p2[:first_N]
    p3 = p3[:first_N]

    cdf_p1 = np.cumsum(p1)
    cdf_p2 = np.cumsum(p2)
    cdf_p3 = np.cumsum(p3)




    rects1 = plt.bar(label1, p1, bar_width,
                     alpha=opacity, color=colors[0], label=model[0])
    rects2 = plt.bar(label1 + bar_width, p2, bar_width,
                     alpha=opacity, color=colors[1], label=model[1])

    rects3 = plt.bar(label1 + 2*bar_width, p3, bar_width,
                     alpha=opacity, color=colors[2], label=model[2])

    plt.plot(label1+ bar_width , cdf_p1, color=colors[0], marker='^', ls='-')
    plt.plot(label1 + bar_width, cdf_p2, color=colors[1], marker='^', ls='-')
    plt.plot(label1 + bar_width, cdf_p3, color=colors[2], marker='^', ls='-')

    plt.plot([-5,-5],[0,0], marker= '^', ls='-', color = 'k', label = 'CDF')


    plt.xlim(-0.5, first_N + 1*bar_width)
    plt.ylim(0, 1)
    ticklabels = [str(i) + '%' for i in range(0, 101, 20)]
    plt.yticks(np.arange(0, 1.01, 0.2), ticklabels,fontsize=20)
    plt.xticks(label1 + bar_width, ['<0.5','0.5-1','1-1.5','1.5-2','2-2.5','2.5-3','3-3.5','3.5-4','4-4.5'], fontsize=20, rotation = 30)

    plt.yticks(fontsize=20)
    plt.xlabel('Absolute errors (hours)', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.legend(fontsize=18, loc = 'center right')

    # plt.text(0.70, 0.95, r'$E[e]$',
    #          horizontalalignment='right', verticalalignment='center',
    #          fontdict={'size': 20, 'weight': 'bold'}, transform=ax.transAxes)
    #
    # plt.text(0.84, 0.95, '{}'.format(round(data1.mean(), 2)),
    #          horizontalalignment='right', verticalalignment='center',
    #          fontdict={'size': 18, 'weight': 'bold'}, color=colors[0], transform=ax.transAxes)
    #
    # plt.text(0.98, 0.95, '{}'.format(round(data2.mean(), 2)),
    #          horizontalalignment='right', verticalalignment='center',
    #          fontdict={'size': 18, 'weight': 'bold'}, color=colors[1], transform=ax.transAxes)
    #
    # plt.text(0.70, 0.85, r'$E[|e|]$',
    #          horizontalalignment='right', verticalalignment='center',
    #          fontdict={'size': 20, 'weight': 'bold'}, transform=ax.transAxes)
    #
    # plt.text(0.84, 0.85, '{}'.format(round(data1.abs().mean(), 2)),
    #          horizontalalignment='right', verticalalignment='center',
    #          fontdict={'size': 18, 'weight': 'bold'}, color=colors[0], transform=ax.transAxes)
    #
    # plt.text(0.98, 0.85, '{}'.format(round(data2.abs().mean(), 2)),
    #          horizontalalignment='right', verticalalignment='center',
    #          fontdict={'size': 18, 'weight': 'bold'}, color=colors[1], transform=ax.transAxes)
    #
    # plt.text(-0.10, 1.05, fig_ind, fontdict={'size': 18, 'weight': 'bold'},
    #          horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.title(title)





def data_process_continuous(data):
    data['abs_error'] = np.abs(data['Ground_truth_duration'] - data['Predict_duration'])
    data_first = data.loc[data['activity_index'] == 0]
    data_Remaining = data.loc[data['activity_index'] != 0]
    error_first_temp = list(data_first['abs_error'])   # data_first['Predicted'] - data_first['Ground_truth_duration']
    error_Remaining_temp = list(data_Remaining['abs_error'])
    return error_first_temp,None, error_Remaining_temp,None,None


if __name__ == '__main__':
    data_path = '../data/'

    # with open (data_path + 'individual_ID_list_test', 'rb') as fp:
    #     individual_ID_list = pickle.load(fp)

    # Error distribution only works for duration!

    num_ind = 1000
    with open(data_path + 'individual_ID_list_test_' + str(num_ind) + '.pickle', 'rb') as fp:
        individual_ID_list_test = pickle.load(fp)
    individual_ID_list = individual_ID_list_test[0:500]

    OUTPUT = 'duration' # location duration

    error_list=[]
    total=0
    error_first = pd.DataFrame({'error':[]})
    error_first_base = pd.DataFrame({'error': []})
    error_first_LSTM = pd.DataFrame({'error': []})
    error_Remaining = pd.DataFrame({'error': []})
    error_Remaining_base = pd.DataFrame({'error': []})
    error_Remaining_LSTM = pd.DataFrame({'error': []})


    Accuracy = {'Card_ID':[], 'Remaining':[],'first':[],'all':[]}
    Accuracy_base = {'Card_ID':[], 'Remaining':[],'first':[],'all':[]}
    Accuracy_LSTM = {'Card_ID': [], 'Remaining': [], 'first': [], 'all': []}
    # data
    Card_ID_used = []
    # individual_ID_list = individual_ID_list[0:50]
    for Card_ID in individual_ID_list:

        file_name = data_path + 'results/result_con_dur+loc_'+ str(Card_ID) + 'test'+'.csv'
        #
        if not os.path.exists(file_name):
            print(Card_ID,'does not exist for IOHMM')
            continue
        else:
            Card_ID_used.append(Card_ID)
        data = pd.read_csv(file_name)
        error_first_temp, Accuracy_first_temp, error_Remaining_temp, Accuracy_temp, accuracy_all = data_process_continuous(data)
        #print (error_first_temp)
        error_first = pd.concat([error_first, pd.DataFrame({'error':list(error_first_temp)}).reset_index(drop = True)],axis=0)
        error_Remaining = pd.concat([error_Remaining, pd.DataFrame({'error':list(error_Remaining_temp)}).reset_index(drop = True)],axis=0)

        # data

    #############
    for Card_ID in individual_ID_list:

        file_name = data_path + 'results/result_LSTM_con_dur' + str(Card_ID) + 'test' + '.csv'

        # file_name = data_path + 'results/result_Location_'+ str(Card_ID) + 'test'+'.csv'
        if not os.path.exists(file_name):
            print(Card_ID,'does not exist for LSTM')
            continue

        data = pd.read_csv(file_name)
        error_first_temp, Accuracy_first_temp, error_Remaining_temp, Accuracy_temp, accuracy_all = data_process_continuous(data)
        #print (error_first_temp)
        error_first_LSTM = pd.concat([error_first_LSTM, pd.DataFrame({'error':list(error_first_temp)}).reset_index(drop = True)],axis=0)
        error_Remaining_LSTM = pd.concat([error_Remaining_LSTM, pd.DataFrame({'error':list(error_Remaining_temp)}).reset_index(drop = True)],axis=0)


    ##############
    for Card_ID in individual_ID_list:

        file_name = data_path + 'results/result_LR' + str(Card_ID) + 'test.csv'

        if not os.path.exists(file_name):
            print(Card_ID,'does not exist for Base')
            continue

        data = pd.read_csv(file_name)
        error_first_temp, Accuracy_first_temp, error_Remaining_temp, Accuracy_temp, accuracy_all = data_process_continuous(data)
        error_first_base = pd.concat([error_first_base, pd.DataFrame({'error':list(error_first_temp)}).reset_index(drop = True)],axis=0)
        error_Remaining_base = pd.concat([error_Remaining_base, pd.DataFrame({'error':list(error_Remaining_temp)}).reset_index(drop = True)],axis=0)

    # ====================

    Out_put_name = 'duration_error_distribution.png'
    dfA = error_first
    dfA2 = error_first_base
    dfA3 = error_first_LSTM
    dfB = error_Remaining
    dfB2 = error_Remaining_base
    dfB3 = error_Remaining_LSTM
    bar_plot(dfA, dfA2,dfA3, dfB, dfB2,dfB3, Out_put_name, save_fig = 1)



