from matplotlib import pyplot as plt


plt.figure(figsize=(15, 5))

ax1 = plt.subplot(131)

tPP = [7.16, 6.42, 5.97, 6.08, 6.22]
oPP = [3.23, 2.95, 2.87, 2.93, 3.02]
dPP = [6.03, 5.58, 5.41, 5.42, 6.66]

x = range(2, 7)
plt.plot(x, tPP, 'k-o', label='Time Prediction')
plt.plot(x, oPP, 'b-^', label='Entry Station Prediction')
plt.plot(x, dPP, 'r-s', label='Exit Station Prediction')
plt.xlim(1, 7)
plt.ylim(2, 8)
plt.xlabel('n')
plt.ylabel('Perplexity')
plt.legend(loc='upper right', fontsize=12)
plt.text(-0.15, 1, '(a)', fontdict={'size': 16, 'weight': 'bold'},
		transform=ax1.transAxes)


ax2 = plt.subplot(132)

tPP = [7.77, 7.11, 6.81, 6.6, 6.45, 6.37, 6.28, 6.24, 6.2, 6.17,
		6.15, 6.14, 6.12, 6.1, 6.07, 6.05, 6.04, 6.03, 6.02, 6.01,
		6, 6, 6, 6, 6]
oPP = [3.09, 2.99, 2.98, 2.95, 2.94, 2.93, 2.94, 2.95, 2.96, 2.97,
		2.99, 2.99, 2.99, 2.99, 3, 3.01, 3.02, 3.03, 3.05, 3.05,
		3.06, 3.08, 3.1, 3.11, 3.11]
dPP = [6.43, 6, 5.77, 5.66, 5.62, 5.58, 5.55, 5.52, 5.5, 5.53,
		5.58, 5.61, 5.57, 5.55, 5.54, 5.55, 5.56, 5.58, 5.59, 5.59,
		5.6, 5.62, 5.64, 5.65, 5.67]

x = range(1, 26)
plt.plot(x, tPP, 'k-o', label='Time Prediction')
plt.plot(x, oPP, 'b-^', label='Entry Station Prediction')
plt.plot(x, dPP, 'r-s', label='Exit Station Prediction')
plt.xlim(0, 26)
plt.ylim(2, 8)
plt.xlabel('alpha')
plt.ylabel('Perplexity')
plt.legend(loc='upper right', fontsize=12)
plt.text(-0.15, 1, '(b)', fontdict={'size': 16, 'weight': 'bold'},
		transform=ax2.transAxes)


ax3 = plt.subplot(133)

tPP = [6.6, 6.28, 6.12, 6.06, 5.99, 5.97, 6.04, 6.16, 6.48]
oPP = [3.29, 3.1, 3.02, 2.98, 2.94, 2.93, 2.97, 3.04, 3.15]
dPP = [7.57, 6.53, 6.01, 5.7, 5.53, 5.42, 5.36, 5.38, 5.58]

x = [i / 10.0 for i in range(1, 10)]
plt.plot(x, tPP, 'k-o', label='Time Prediction')
plt.plot(x, oPP, 'b-^', label='Entry Station Prediction')
plt.plot(x, dPP, 'r-s', label='Exit Station Prediction')
plt.xlim(0, 1)
plt.ylim(2, 8)
plt.xlabel('beta')
plt.ylabel('Perplexity')
plt.legend(loc='upper right', fontsize=12)
plt.text(-0.15, 1, '(c)', fontdict={'size': 16, 'weight': 'bold'},
		transform=ax3.transAxes)

plt.show()
