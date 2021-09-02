REF_DAY = 15614  # Oct 1 2012, Monday
HOLIDAYS = [12054, 12141, 12144, 12179, 12200, 12291, 12412, 12413,
			12419, 12526, 12529, 12543, 12564, 12655, 12777, 12778,
			12784, 12876, 12879, 12907, 12928, 13026, 13142, 13143,
			13149, 13233, 13236, 13271, 13299, 13390, 13508, 13509]


def getDayType(day):
	d = (day - REF_DAY) % 7
	if d < 5:
		return 'Weekday'
	else:
		return 'Weekend'


def getDayOfWeek(daykey):
	day = (daykey - REF_DAY) % 7
	if day == 0:
		return 'MON'
	elif day == 1:
		return 'TUE'
	elif day == 2:
		return 'WED'
	elif day == 3:
		return 'THU'
	elif day == 4:
		return 'FRI'
	elif day == 5:
		return 'SAT'
	elif day == 6:
		return 'SUN'


def getDayOfWeekIndex(daykey):
	return (daykey - REF_DAY) % 7


def isHoliday(daykey):
	if daykey in HOLIDAYS:
		return True
	else:
		return False


def isSummer(daykey):
	n_years = int(len(HOLIDAYS) / 8)
	for i in range(n_years):
		if daykey > HOLIDAYS[i * 8 + 4] and daykey < HOLIDAYS[i * 8 + 5]:
			return True
	return False


def getWeekID(dayID):
	# A week starts on Sunday and ends on Saturday
	return int((dayID - REF_DAY) / 7)


def getDayID(weekID):
	return weekID * 7 + REF_DAY


def wordListToFreqDict(wordlist):
	wordfreq = [wordlist.count(p) for p in wordlist]
	return dict(zip(wordlist, wordfreq))


def sortFreqDict(freqdict):
	aux = [(key, freqdict[key])for key in freqdict]
	return sorted(aux, key=lambda item: item[1], reverse=True)


class user(object):
	def __init__(self, ID, tripList=None):
		self.id = ID
		self.tripList = tripList

	def addTrip(self, trip):
		self.tripList.append(trip)

	def getTripList(self, dayType=None, order=None):
		if dayType is None and order is None:
			return self.tripList
		else:
			trips = []
			dailyTrips = self.getDailyTrips()
			activeDays = self.getActiveDayList(dayType)
			if order is None:
				for day in activeDays:
					trips.extend(dailyTrips[day])
			else:
				for day in activeDays:
					if order == 'any but first':
						trips.extend(dailyTrips[day][1:])
					if order < len(dailyTrips[day]):
						trips.append(dailyTrips[day][order])
			return trips

	def getDailyTrips(self):
		dailyDict = {}
		for t in self.getTripList():
			if t.day not in dailyDict.keys():
				dailyDict[t.day] = [t]
			else:
				dailyDict[t.day].append(t)
		return dailyDict

	def getActiveDayList(self, dayType=None):
		days = list(set([t.day for t in self.getTripList()]))
		if dayType:
			for i, d in enumerate(days):
				if getDayType(d) != dayType:
					days.remove(d)
		return sorted(days)

	def getActiveDays(self, dayType=None):
		days = list(set([t.day for t in self.getTripList()]))
		if dayType:
			for d in days:
				if getDayType(d) != dayType:
					days.remove(d)
		return len(days)

	def getStationList(self, stationType="all", dayType=None, order=None):
		try:
			stationType in ["all", "in", "out", "od"]
		except:
			raise ValueError("stationType value not valid!")

		tripList = self.getTripList(dayType, order)
		stationList = []
		if stationType == "all":
			for t in tripList:
				stationList.extend([t.o, t.d])
		elif stationType == "in":
			stationList = [t.o for t in tripList]
		elif stationType == "out":
			stationList = [t.d for t in tripList]
		elif stationType == "od":
			stationList = [t.getOD(asTuple=True) for t in tripList]

		return stationList

	def getStationDict(self, stationType="all"):
		'''
		Return a dictionary whose key = station name and value = rank
		'''
		statList = self.getStationList(stationType)
		freqDict = wordListToFreqDict(statList)
		sortedStats = sortFreqDict(freqDict)
		statDict = {}
		for i, s in enumerate(sortedStats):
			statDict[s[0]] = i + 1
		return statDict

	def getTripTimeList(self, unit=1, dayType=None, order=None):
		tripList = self.getTripList(dayType, order)
		return [t.getTime(unit) for t in tripList]

	def getInterTripTimeList(self, unit=1, dtType=None, dayType=None, order=None):
		deltaT = []
		tripList = self.getTripList(dayType, order)
		if dtType is None:
			for i in range(1, len(tripList)):
				deltaT.append(tripList[i].getAbsTime(unit) - tripList[i - 1].getAbsTime(unit))
		elif dtType == 'within':
			for i in range(1, len(tripList)):
				if tripList[i].day == tripList[i - 1].day:
					deltaT.append(tripList[i].getTime(unit) - tripList[i - 1].getTime(unit))
		elif dtType == 'across':
			for i in range(1, len(tripList)):
				if tripList[i].day != tripList[i - 1].day:
					deltaT.append(tripList[i].getAbsTime(unit) - tripList[i - 1].getAbsTime(unit))
		else:
			raise ValueError("dtType value not valid!")
		return deltaT

	def getDailySequences(self):
		seqDict = {}
		for t in self.getTripList():
			if t.day not in seqDict.keys():
				seqDict[t.day] = t.syntaxExpression()
			else:
				seqDict[t.day].extend(t.syntaxExpression())
		return seqDict

	def getWeeklyDailySequences(self):
		seqDict = {}
		dailySeqs = self.getDailySequences()
		days = self.getActiveDayList()
		for d in days:
			w = getWeekID(d)
			if w not in seqDict.keys():
				seqDict[w] = []
			seqDict[w].append((d, dailySeqs[d]))
		return seqDict

	def getDailySequencesWithHeadway(self):
		dailyDict = self.getDailyTrips()
		seqDict = {}
		for day, trips in dailyDict.iteritems():
			seq = trips[0].syntaxExpression()
			num_trips = len(trips)
			for i in range(1, num_trips):
				deltaT = (trips[i].getTime() - trips[i - 1].getTime()) * 1.0 / 60
				seq.extend([int(deltaT), trips[i].o, trips[i].d])
			seqDict[day] = seq
		return seqDict

	def getODTrips(self):
		odDict = {}
		for t in self.getTripList():
			if t.getOD() not in odDict.keys():
				odDict[t.getOD()] = [t]
			else:
				odDict[t.getOD()].append(t)
		return odDict

	def getRecycleTimeList(self):
		cycleT = []
		odDict = self.getODTrips()
		for trips in odDict.values():
			if len(trips) > 1:
				cycleT.extend([trips[i].getAbsTime() - trips[i - 1].getAbsTime() for i in range(1, len(trips))])
		return cycleT

	def getDayRange(self):
		days = self.getActiveDayList()
		minD = days[0]
		maxD = days[-1]
		return minD, maxD

	def getWeekRange(self):
		minD, maxD = self.getDayRange()
		return getWeekID(minD), getWeekID(maxD)

	def getActiveDayVector(self):
		dayRange = range(REF_DAY + 366, REF_DAY + 366 + 365)
		vector = [0] * len(dayRange)
		activeDays = self.getActiveDayList()
		for i, d in enumerate(dayRange):
			if d in activeDays:
				vector[i] = 1
		return vector

	def getInterActiveDays(self):
		days = self.getActiveDayList()
		return [days[i] - days[i - 1] for i in range(1, len(days))]
