import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# colors = ["#3366cc", "#dc3912", "#109618", "#990099", "#ff9900"]
colors = sns.color_palette('muted')



def obtain_rank_prob(results,max_rank):
    for k in range(max_rank):
        name = 'Predict' + str(k+1)
        correct_name = 'Correct' + str(k+1)
        results[correct_name] = 0
        results.loc[results[name] == results['Ground_truth'], correct_name] = 1

    PA = results.loc[results['activity_index'] == 0]
    PB = results.loc[results['activity_index'] != 0]
    cum_proA = []
    cum_proB = []
    for k in range(max_rank):
        total_correct_A = 0
        total_correct_B = 0
        for j in range(k+1):
            correct_name = 'Correct' + str(j + 1)
            total_correct_A += len(PA.loc[PA[correct_name]==1])
            total_correct_B += len(PB.loc[PB[correct_name] == 1])

        cum_proA.append(total_correct_A/len(PA)*100)
        cum_proB.append(total_correct_B/len(PB)*100)
    return cum_proA,cum_proB



def prediction_rank_plot(IOHMM, Base, LSTM, NG, file_name_tail, save_fig):
    # columns = ['rankT', 'rankO', 'rankD']

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid", {"legend.frameon": True})
    plt.figure(figsize=(14, 7))

    ax1 = plt.subplot(1, 2, 1)
    sign = ['-o','-s']
    i = 1
    plt.plot(range(1, 21), IOHMM[0][i], sign[i], color=colors[0], label='IOHMM',alpha = 0.8)
    plt.plot(range(1, 21), Base[0][i], sign[i], color=colors[1], label='Base-MC',alpha = 0.8)
    plt.plot(range(1, 21), LSTM[0][i], sign[i], color=colors[2], label='Base-LSTM',alpha = 0.8)
    plt.plot(range(1, 21), NG[0][i], sign[i], color=colors[3], label='Base-NG',alpha = 0.8)
    plt.xlim(0, 21)
    plt.ylim(70, 102)
    plt.xticks(range(0, 21, 5), range(0, 21, 5), fontsize = 20)
    plt.yticks([i for i in range(70, 102, 10)], [str(i) + '%' for i in range(70, 102, 10)], fontsize = 20)
    plt.xlabel(r'Prediction rank ($k$)', fontsize=20)
    plt.ylabel('Cumulative probability', fontsize=20)
    plt.title('First activities', fontsize=18)
    plt.legend(loc='lower right', fontsize=18)
    # plt.text(-0.15, 1.05, '(a)', fontdict={'size': 18, 'weight': 'bold'},
    #     horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

    ax2 = plt.subplot(1, 2, 2)

    plt.plot(range(1, 21), IOHMM[1][i],sign[i], color=colors[0], label='IOHMM',alpha = 0.8)
    plt.plot(range(1, 21), Base[1][i], sign[i], color=colors[1], label='Base-MC',alpha = 0.8)
    plt.plot(range(1, 21), LSTM[1][i], sign[i], color=colors[2], label='Base-LSTM',alpha = 0.8)
    plt.plot(range(1, 21), NG[1][i], sign[i], color=colors[3], label='Base-NG',alpha = 0.8)

    plt.xlim(0, 21)
    plt.ylim(30, 102)
    plt.xticks(range(0, 21, 5), range(0, 21, 5),fontsize = 20)
    plt.yticks([i for i in range(30, 102, 10)], [str(i) + '%' for i in range(30, 102, 10)], fontsize = 20)
    plt.xlabel(r'Prediction rank ($k$)', fontsize=20)
    plt.ylabel('Cumulative probability', fontsize=20)
    plt.title('Remaining activities', fontsize=18)
    plt.legend(loc='lower right', fontsize=18)
    # plt.text(-0.15, 1.05, '(b)', fontdict={'size': 18, 'weight': 'bold'},
    #     horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/prediction_rank_' + file_name_tail + '_add_NG.png', dpi=200)

if __name__ == '__main__':

    #####ONLY FOR LOCATION

    #RE_CAL_ACC = T

    data_path = '../data/'
    # with open (data_path + 'individual_ID_list_test', 'rb') as fp:
    #     individual_ID_list = pickle.load(fp)

    num_ind = 1000
    with open(data_path + 'individual_ID_list_test_' + str(num_ind) + '.pickle', 'rb') as fp:
        individual_ID_list = pickle.load(fp)
    individual_ID_list = individual_ID_list[0:500]
    file_name_tail = '_500_samples'


    data_IOHMM = pd.DataFrame()
    data_IOHMM_location = pd.DataFrame()
    data_base = pd.DataFrame()
    data_base_location = pd.DataFrame()
    data_LSTM = pd.DataFrame()
    data_LSTM_location = pd.DataFrame()

    data_NG = pd.DataFrame()
    data_NG_location = pd.DataFrame()

    Card_ID_used =[]
    for Card_ID in individual_ID_list:
        file_name = data_path + 'results/result_con_dur+loc_' + str(Card_ID) + 'test.csv'
        if not os.path.exists(file_name):
            print(Card_ID,'does not exist for IOHMM')
            continue
        else:
            if Card_ID not in Card_ID_used:
                Card_ID_used.append(Card_ID)
        data = pd.read_csv(file_name)
        data_IOHMM = data_IOHMM.append(data)
        #----------------------------------------
        file_name = data_path + 'results/result_LR' + str(Card_ID) + 'test.csv'
        if not os.path.exists(file_name):
            print(Card_ID,'does not exist for MC')
            continue

        data = pd.read_csv(file_name)
        data_base = data_base.append(data)
        #----------------------------------------
        file_name = data_path + 'results/result_LSTM_con_dur' + str(Card_ID) + 'test' + '.csv'
        if not os.path.exists(file_name):
            print(Card_ID,'does not exist for LSTM')
            continue

        data = pd.read_csv(file_name)
        data_LSTM = data_LSTM.append(data)


        #################################################
        ################################################
        ################################################
        ################################################
        file_name = data_path + 'results/result_con_dur+loc_' + str(Card_ID) + 'test.csv'
        if not os.path.exists(file_name):
            print(Card_ID,'does not exist for IOHMM')
            continue
        else:
            if Card_ID not in Card_ID_used:
                Card_ID_used.append(Card_ID)
        data = pd.read_csv(file_name)
        data_IOHMM_location = data_IOHMM_location.append(data)
        #---------------------------

        file_name = data_path + 'results/result_Location_MC'+ str(Card_ID) + '.csv'
        if not os.path.exists(file_name):
            print(Card_ID,'does not exist for MC')
            continue

        data = pd.read_csv(file_name)
        data_base_location = data_base_location.append(data)

        #---------------------------

        file_name = data_path + 'results/result_Location_LSTM'+ str(Card_ID) +'test'+'.csv'
        if not os.path.exists(file_name):
            print(Card_ID,'does not exist for LSTM')
            continue

        data = pd.read_csv(file_name)
        data_LSTM_location = data_LSTM_location.append(data)


        #---------------------------NG

        file_name = data_path + 'results/result_NGRAM_location_' + str(Card_ID) + '.csv'
        if not os.path.exists(file_name):
            print(Card_ID,'does not exist for NG')
            continue

        data = pd.read_csv(file_name)
        data_NG_location = data_NG_location.append(data)




    IOHMM_cum = [[0,0],[0,0]]
    # IOHMM_cum[0][0],IOHMM_cum[1][0]  = obtain_rank_prob(data_IOHMM, max_rank=20)
    IOHMM_cum[0][1], IOHMM_cum[1][1] = obtain_rank_prob(data_IOHMM_location, max_rank=20)
    base_cum = [[0,0], [0,0]]
    # base_cum[0][0],base_cum[1][0]  = obtain_rank_prob(data_base, max_rank=20)
    base_cum[0][1], base_cum[1][1] = obtain_rank_prob(data_base_location, max_rank=20)
    LSTM_cum = [[0,0], [0,0]]
    # LSTM_cum[0][0],LSTM_cum[1][0]  = obtain_rank_prob(data_LSTM, max_rank=20)
    LSTM_cum[0][1], LSTM_cum[1][1] = obtain_rank_prob(data_LSTM_location, max_rank=20)

    NG_cum = [[0,0], [0,0]]
    # LSTM_cum[0][0],LSTM_cum[1][0]  = obtain_rank_prob(data_LSTM, max_rank=20)
    NG_cum[0][1], NG_cum[1][1] = obtain_rank_prob(data_NG_location, max_rank=20)

    prediction_rank_plot(IOHMM_cum, base_cum, LSTM_cum, NG_cum, file_name_tail, save_fig = 1)
