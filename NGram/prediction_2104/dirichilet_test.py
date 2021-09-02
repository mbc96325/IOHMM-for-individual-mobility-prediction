import numpy as np
from scipy.special import digamma


def alpha_update(m, p):
	K = m.size
	s_old = - (K - 1) / 2 / np.sum(np.multiply(m, (np.log(p) - np.log(m))))
	print s_old
	count = 0
	for i in range(100):
		s_new = (K - 1) / ((K - 1)/s_old - digamma(s_old) +
			np.sum(np.multiply(m, digamma(s_old * m))) - np.sum(np.multiply(m, np.log(p))))
		if abs(s_new - s_old) < 1e-2:
			print count
			return s_new
		s_old = s_new
		count += 1
	print count
	return s_new


if __name__ == '__main__':
	m = np.array([0.1, 0.2, 0.3, 0.4])
	p = np.array([0.001, 0.001, 0.499, 0.499])
	print alpha_update(m, p)
