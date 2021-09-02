import numpy as np
import matplotlib.pyplot as plt
import oyster_reader as oyster


dataFile = "../../data/sampleData_2013_reduced.csv"


if __name__ == "__main__":
	X = []
	users = oyster.readPanelData(dataFile)

	def getKey(u):
		return u.getActiveDays()

	users = sorted(users, key=getKey, reverse=True)

	for user in users:
		X.append(user.getActiveDayArray())
	X = np.array(X)
	print np.sum(X)

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111)
	plt.imshow(X)
	ax.set_aspect('equal')
	plt.show()

