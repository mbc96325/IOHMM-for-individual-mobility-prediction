import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
colors = ["#3366cc", "#dc3912", "#109618", "#990099", "#ff9900"]

save_fig = True

data = pd.read_csv('../data/individual_predict_acc.csv')
plt.figure(figsize=(7, 5))
act_num = list(data['Total_act'])
ax2 = plt.subplot(1, 1, 1)
max_act = 8
min_act = 3
delta = 0.3
plt.hist(np.array(act_num), bins=range(3, max_act + 1), density=True, color=colors[2], edgecolor='w', alpha=0.8)
plt.xlim(min_act - delta, max_act + delta)
plt.xlabel('Estimated number of hidden activities', fontsize=18)
plt.ylabel('Probability', fontsize=18)
plt.yticks(fontsize = 18)
plt.xticks([i + 0.5 for i in range(3, max_act)], range(3, max_act), fontsize = 18)
plt.tight_layout()

if save_fig:
    plt.savefig('img/num_hidden_act_distribution.png', dpi=200)
else:
    plt.show()


