import numpy as np
import math


def probDbn(X):
	'''
	X is a list of (repeatable) labels
	returns a list of probabilities, corresponding to each of the possible label
	'''
	X = np.array(X)
	return [np.mean(X == c) for c in set(X)]


def jointProbDbn(X, Y):
	setX = set(X)
	setY = set(Y)
	X = np.array(X)
	Y = np.array(Y)
	probs = np.empty((len(setX), len(setY)))
	for i, x in enumerate(setX):
		for j, y in enumerate(setY):
			probs[i, j] = np.mean(np.logical_and(X == x, Y == y))
	return probs


def entropy(X):
	probs = probDbn(X)
	return -np.sum(p * np.log2(p) for p in probs)


def jointEntropy(X, Y):
	probs = jointProbDbn(X, Y)
	return -np.sum(p * np.log2(p) for p in np.nditer(probs) if p > 0)


def conditionalEntropy(X, Y):
	'''
	H(X|Y) = H(X, Y) - H(Y) = H(X) - MI(X, Y)
	'''
	return jointEntropy(X, Y) - entropy(Y)


def mutualInfo(X, Y):
	'''
	MI(X, Y) = H(X) + H(Y) - H(X, Y)
	'''
	return entropy(X) + entropy(Y) - jointEntropy(X, Y)


def normalizedMutualInfo(X, Y):
	'''
	NMI(X, Y) = MI(X, Y) / (H(X) * H(Y))^0.5
	'''
	return mutualInfo(X, Y) / math.sqrt(entropy(X) * entropy(Y))


def infoGainRatio(X, Y):
	return mutualInfo(X, Y) / entropy(X)


if __name__ == '__main__':
	x = ['a', 'b', 'b', 'a', 'a', 'a', 'a', 'b']
	y = [3, 1, 1, 3, 2, 3, 2, 5]
	print jointEntropy(x, y)
	print mutualInfo(x, y)
	print entropy(x) - conditionalEntropy(x, y)
	print normalizedMutualInfo(x, y)
