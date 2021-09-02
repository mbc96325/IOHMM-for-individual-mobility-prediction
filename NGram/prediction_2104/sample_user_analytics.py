import oyster_reader as oyster
import user
import csv


if __name__ == '__main__':
	dataFile = "../output/sampleUsers.csv"
	vocabFile = "../data/station_vocab.csv"
	users = oyster.readPanelData2(dataFile, vocabFile)
	freq_users = [u for u in users if u.getActiveDays() >= 60]
	print 'Number of users: {}'.format(len(freq_users))
	for i, u in enumerate(users):
		filename = '../output/user{}.csv'.format(i + 1)
		wt = csv.writer(open(filename, 'wt'))
		wt.writerow(['origin', 'destination', 'count'])
		D = u.getStationDict()

		ods = u.getStationList(stationType='od')
		odDict = user.wordListToFreqDict(ods)
		sortedODs = user.sortFreqDict(odDict)
		n = len(sortedODs)
		for i in range(n):
			od, c = sortedODs[n - i - 1]
			wt.writerow([D[od[0]], D[od[1]], c])
