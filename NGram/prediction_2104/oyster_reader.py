import csv

from trip import trip
from user import user


class panelDataReader:
	def __init__(self, file):
		self.reader = csv.reader(open(file), delimiter=",")
		self.header = self.reader.next()
		self.lastRecord = None

	def nextRecord(self):
		try:
			line = next(self.reader)
		except StopIteration:
			line = None
		self.lastRecord = line
		return line

	def nextUserRecords(self):
		userIndex = self.header.index("prestigeid")
		records = []
		if self.lastRecord is None:
			if self.nextRecord() is None:
				return None

		firstRecord = self.lastRecord
		records.append(firstRecord)
		while True:
			prevID = self.lastRecord[userIndex]
			nextRecord = self.nextRecord()
			if nextRecord is not None and prevID == nextRecord[userIndex]:
				records.append(nextRecord)
			else:
				break

		if len(records) > 0:
			return records
		else:
			return None


def readPanelData(file, stationDictPath=None):
	print 'Importing users...'
	if stationDictPath is not None:
		stationDict = stationNameDict(stationDictPath)
	else:
		stationDict = None

	panelReader = panelDataReader(file)
	# indices for later use
	headers = panelReader.header
	userIndex = headers.index("prestigeid")
	dayIndex = headers.index("daykey")
	transIndex = headers.index("transactiontype")
	timeIndex = headers.index("transactiontime")
	entryIndex = headers.index("stationoffirstentrykey")
	exitIndex = headers.index("stationofexitkey")

	X = []
	counter = 0
	userRecords = panelReader.nextUserRecords()
	# If there is more data to be processed
	while userRecords is not None:
		userID = userRecords[0][userIndex]  # new user ID
		userTripList = []  # new user tripList
		prevTap = None
		for i in xrange(len(userRecords)):
			tap = userRecords[i]
			# convert day and time to integer
			tap[dayIndex] = int(tap[dayIndex])
			tap[timeIndex] = int(tap[timeIndex])
			# If transaction time ealier than 3am, count it as previous day
			if tap[timeIndex] < 60 * 3:
				tap[dayIndex] -= 1
				tap[timeIndex] += 60 * 24
			# Add new trip
			if tap[transIndex] == "61":
				if prevTap is None:
					prevTap = tap
					continue  # go to nex iteration to see if it is a complete trip
				else:
					# a new trip missing tap-out
					newTrip = trip(day=prevTap[dayIndex],
									o=prevTap[entryIndex],
									d=prevTap[exitIndex],  # outStation="-1"
									ot=prevTap[timeIndex],
									dt=-1)
					prevTap = tap
			elif tap[transIndex] == "62":
				if prevTap is None:
					# a new trp missing tap-in
					newTrip = trip(day=tap[dayIndex],
									o=tap[entryIndex],
									d=tap[exitIndex],
									ot=-1,
									dt=tap[timeIndex])
				elif prevTap[dayIndex] == tap[dayIndex] and \
					prevTap[entryIndex] == tap[entryIndex]:
					# a complete new trip
					newTrip = trip(day=tap[dayIndex],
									o=tap[entryIndex],
									d=tap[exitIndex],
									ot=prevTap[timeIndex],
									dt=tap[timeIndex])
					prevTap = None
			if newTrip is not None:
				# Exclude trips with same entry and exit stations
				if newTrip.o == newTrip.d:
					continue
				# Exclude trips with unvalid hours
				if newTrip.getHour() > 26 or newTrip.getHour() < 3:
					continue
				# Exclude repetitive records
				if len(userTripList) > 0:
					if newTrip.day == userTripList[-1].day and \
						newTrip.getTime() == userTripList[-1].getTime():
						continue
				# Convert station id to station name
				if stationDict is not None:
					newTrip.o = stationDict[newTrip.o]
					newTrip.d = stationDict[newTrip.d]
				# Exclude incomplete trips with unknown OD pairs
				if newTrip.incomplete() is True or\
					newTrip.o == "Unknown" or newTrip.d == "Unknown":
					newTrip = None
					continue
				userTripList.append(newTrip)  # Add the new trip to the user's list
				newTrip = None
		# Define a new user and add to a user list
		newUser = user(userID, tripList=userTripList)
		X.append(newUser)
		# Print progress (number of users processed...)
		counter += 1
		if counter % 100 == 0:
			print counter
		if counter == 1000:
			return X
		# Get next user's transactions, and start over
		userRecords = panelReader.nextUserRecords()
	return X


def readPanelData2(file, stationDictPath=None, limit=None):
	print 'Importing users...'
	if stationDictPath is not None:
		stationDict = stationNameDict(stationDictPath)
	else:
		stationDict = None

	panelReader = panelDataReader(file)
	# indices for later use
	headers = panelReader.header
	userIndex = headers.index("prestigeid")
	dayIndex = headers.index("daykey")
	transIndex = headers.index("transactiontype")
	entryTimeIndex = headers.index("timeoffirstentry")
	exitTimeIndex = headers.index("transactiontime")
	entryIndex = headers.index("stationoffirstentry")
	exitIndex = headers.index("nlc")

	X = []
	counter = 0
	userRecords = panelReader.nextUserRecords()
	# If there is more data to be processed
	while userRecords is not None:
		userID = userRecords[0][userIndex]  # new user ID
		userTripList = []  # new user tripList
		for i in xrange(len(userRecords)):
			tap = userRecords[i]
			if tap[transIndex] == "62":
				# convert day and time to integer
				daykey = int(tap[dayIndex])
				inTime = int(tap[entryTimeIndex])
				outTime = int(tap[exitTimeIndex])
				inStation = tap[entryIndex]
				outStation = tap[exitIndex]

				# Add new trip
				if inStation != "-1" and outStation != "-1":
					# If transaction time ealier than 3am, count it as previous day
					# if inTime < 60 * 3:
						# daykey -= 1
						# inTime += 60 * 24
						# outTime += 60 * 24
					# define a new trip object
					newTrip = trip(day=daykey,
									o=inStation,
									d=outStation,
									ot=inTime,
									dt=outTime)

					# Exclude trips with same entry and exit stations
					if newTrip.o == newTrip.d:
						continue
					# Exclude trips with unvalid hours
					if newTrip.getHour() > 26 or newTrip.getHour() < 3:
						continue
					# Convert station id to station name
					if stationDict is not None:
						newTrip.o = stationDict[newTrip.o]
						newTrip.d = stationDict[newTrip.d]
					# Exclude incomplete trips with unknown OD pairs
					if newTrip.incomplete() is True or\
						newTrip.o == "Unknown" or newTrip.d == "Unknown":
						newTrip = None
						continue
					# If two trips with same starting time, use the latest one
					if len(userTripList) > 0:
						if newTrip.day == userTripList[-1].day and \
							newTrip.getAbsTime() == userTripList[-1].getAbsTime():
							userTripList[-1] = newTrip
							continue
					# Data errors
					if len(userTripList) > 0:
						if newTrip.day == userTripList[-1].day and \
							newTrip.getAbsTime() < userTripList[-1].getAbsTime():
							continue
					userTripList.append(newTrip)  # Add the new trip to the user's list
		# Define a new user and add to a user list
		userTripList.sort
		newUser = user(userID, tripList=userTripList)
		X.append(newUser)
		# Print progress (number of users processed...)
		counter += 1
		if counter % 10000 == 0:
			print counter
		if limit:
			if counter >= limit:
				return X
		# Get next user's transactions, and start over
		userRecords = panelReader.nextUserRecords()
	return X


def stationNameDict(filepath):
	rd = csv.reader(open(filepath, 'rU'), delimiter=",")
	Dict = {}
	for s in rd:
		Dict[s[0]] = s[-1]
	# Special situations
	# Dict["-1"] = "Unknown"
	# Dict["0"] = "Unknown"
	return Dict


def filter_by_userid(filepath, ids):
	rd = csv.reader(open(filepath, 'rU'))
	headers = rd.next()
	userIndex = headers.index("prestigeid")

	filename = '../output/sampleUsers.csv'
	wt = csv.writer(open(filename, 'wt'))
	wt.writerow(headers)
	for row in rd:
		userid = row[userIndex]
		if userid in ids:
			wt.writerow(row)


if __name__ == '__main__':
	dataFile = "../data/oysterdata.csv"
	# ids = ['1797223601', '1537232329']
	ids = ['1837931289', '1297299286']
	filter_by_userid(dataFile, ids)
