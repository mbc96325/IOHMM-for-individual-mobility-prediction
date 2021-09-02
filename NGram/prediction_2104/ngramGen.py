def getDayOfWeek(daykey):
	day = (daykey - 12052) % 7
	if day == 1:
		return 'MON'
	elif day == 2:
		return 'TUE'
	elif day == 3:
		return 'WED'
	elif day == 4:
		return 'THU'
	elif day == 5:
		return 'FRI'
	elif day == 6:
		return 'SAT'
	elif day == 0:
		return 'SUN'


def getDayType(daykey):
	day = (daykey - 12052) % 7
	if day > 0 and day < 6:
		return 'Weekday'
	else:
		return 'Weekend'


class ngramGenerator(object):
	def __init__(self, seqs, n=None, TEST_ = False, PART_INFO = False, USED_TOD_LIST = None):
		self.ngramIn = None
		self.ngramOut = None
		self.ngramTime = None
		self.ngramIn1 = None
		self.ngramOut1 = None
		self.ngramTime1 = None
		if n:
			self.ngramGen(seqs, n)
		else:
			self.mobilityGramGen(seqs, TEST_, PART_INFO, USED_TOD_LIST)

	def ngramGen(self, seqs, n):
		ngramIn = []
		ngramOut = []
		ngramTime = []
		# start_OD = ["START-" + j for j in ["O", "D"]]
		padding = n - 1
		start = ['<START-' + str(j) + '>' for j in range(padding, 0, -1)]
		for seq in seqs:
			snt = seq[1]
			s = start + snt
			for i in range(padding, len(s)):
				if (i - padding + 1) % 3 == 1:
					ngramTime.append(tuple(s[i - n + 1: i + 1]))
				elif (i - padding + 1) % 3 == 2:
					ngramIn.append(tuple(s[i - n + 1: i + 1]))
				else:
					ngramOut.append(tuple(s[i - n + 1: i + 1]))

		self.ngramTime = ngramTime
		self.ngramIn = ngramIn
		self.ngramOut = ngramOut

	def mobilityGramGen(self, seqs, TEST_ = False, PART_INFO = False, USED_TOD_LIST = None):
		ngramIn1 = []
		ngramOut1 = []
		ngramTime1 = []
		ngramIn = []
		ngramOut = []
		ngramTime = []
		#print(len(seqs))
		# start = '<START>'
		if TEST_:
			test = {'entry':[],'exit':[],'time':[],'day':[]}

		#seq = seqs[2]
		for seq in seqs:
			day = seq[0]
			trips = seq[1]
			dow = getDayOfWeek(day)
			dayType = getDayType(day)
			t = [trips[j] for j in range(len(trips)) if (j + 1) % 3 == 1]
			o = [trips[j] for j in range(len(trips)) if (j + 1) % 3 == 2]
			d = [trips[j] for j in range(len(trips)) if (j + 1) % 3 == 0]
			for i in range(len(t)):
				if i == 0:  # The first trip of the day
					if PART_INFO:
						ngramTime1.append((dow, dayType, t[i]))
						ngramIn1.append((dow, dayType, o[i]))
						ngramOut1.append((dow, dayType, t[i], o[i], d[i]))
					else:
						ngramTime1.append((dow, dayType, t[i]))
						ngramIn1.append((dow, dayType, t[i], o[i]))
						ngramOut1.append((dow, dayType, t[i], o[i], d[i]))
					if TEST_:
						test['time'].append(t[i])
						test['entry'].append(o[i])
						test['exit'].append(d[i])
						test['day'].append(day)
				else:
					if PART_INFO:
						if USED_TOD_LIST is not None:
							if o[i-1] in USED_TOD_LIST['o']:
								o_info = o[i - 1]
							else:
								o_info = -99
							#t_info = t[i - 1]
							if t[i-1] in USED_TOD_LIST['t']:
								t_info = t[i - 1]
							else:
								t_info = -1

							if d[i-1] in USED_TOD_LIST['d']:
								d_info = d[i - 1]
							else:
								d_info = -99
						else:
							o_info = o[i - 1]
							t_info = t[i - 1]
							d_info = d[i - 1]
						ngramTime.append((o_info, t_info, d_info, t[i]))
						ngramIn.append((t_info, o_info, d_info, o[i]))
						ngramOut.append((t[i - 1], o[i - 1], d[i - 1], t[i], o[i], d[i]))
					else:
						ngramTime.append((o[i - 1], t[i - 1], d[i - 1], t[i]))
						ngramIn.append((t[i - 1], o[i - 1], t[i], d[i - 1], o[i]))
						ngramOut.append((t[i - 1], o[i - 1], d[i - 1], t[i], o[i], d[i]))
					if TEST_:
						test['time'].append(t[i])
						test['entry'].append(o[i])
						test['exit'].append(d[i])
						test['day'].append(day)


		#
		if TEST_:
			import pandas as pd
			test_df = pd.DataFrame(test)
			test_df.to_csv('test.csv')


		self.ngramTime1 = ngramTime1
		self.ngramIn1 = ngramIn1
		self.ngramOut1 = ngramOut1

		self.ngramTime = ngramTime
		self.ngramIn = ngramIn
		self.ngramOut = ngramOut

	def getNGrams(self, gramType='TOD'):
		if gramType == 'TOD':
			return self.ngramTime, self.ngramIn, self.ngramOut
		elif gramType == 'T':
			return self.ngramTime
		elif gramType == 'O':
			return self.ngramIn
		elif gramType == 'D':
			return self.ngramOut

	def getNGramsFirst(self, gramType='TOD'):
		if gramType == 'TOD':
			return self.ngramTime1, self.ngramIn1, self.ngramOut1
		elif gramType == 'T':
			return self.ngramTime1
		elif gramType == 'O':
			return self.ngramIn1
		elif gramType == 'D':
			return self.ngramOut1


class ngramGenerator_baseline(object):
	def __init__(self, seqs):
		self.ngramIn = None
		self.ngramOut = None
		self.ngramTime = None
		self.ngramIn1 = None
		self.ngramOut1 = None
		self.ngramTime1 = None
		self.ngramGen(seqs)

	def ngramGen(self, seqs):
		ngramIn1 = []
		ngramOut1 = []
		ngramTime1 = []
		ngramIn = []
		ngramOut = []
		ngramTime = []

		for seq in seqs:
			trips = seq[1]
			t = [trips[i] for i in range(len(trips)) if (i + 1) % 3 == 1]
			o = [trips[i] for i in range(len(trips)) if (i + 1) % 3 == 2]
			d = [trips[i] for i in range(len(trips)) if (i + 1) % 3 == 0]

			for i in range(len(t)):
				if i == 0:  # The first trip of the day
					ngramTime1.append((t[i],))
					ngramIn1.append((o[i],))
					ngramOut1.append((o[i], d[i]))
				else:
					ngramTime.append((t[i - 1], t[i]))
					ngramIn.append((d[i - 1], o[i]))
					ngramOut.append((o[i], d[i]))

		self.ngramTime1 = ngramTime1
		self.ngramIn1 = ngramIn1
		self.ngramOut1 = ngramOut1

		self.ngramTime = ngramTime
		self.ngramIn = ngramIn
		self.ngramOut = ngramOut

	def getNGrams(self, gramType='TOD'):
		if gramType == 'TOD':
			return self.ngramTime, self.ngramIn, self.ngramOut
		elif gramType == 'T':
			return self.ngramTime
		elif gramType == 'O':
			return self.ngramIn
		elif gramType == 'D':
			return self.ngramOut

	def getNGramsFirst(self, gramType='TOD'):
		if gramType == 'TOD':
			return self.ngramTime1, self.ngramIn1, self.ngramOut1
		elif gramType == 'T':
			return self.ngramTime1
		elif gramType == 'O':
			return self.ngramIn1
		elif gramType == 'D':
			return self.ngramOut1
