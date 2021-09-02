class trip(object):
	def __init__(self, day, o, d, ot, dt):
		self.day = day  # int
		self.o = o  # string
		self.d = d  # string
		self.ot = ot  # int (in min)
		self.dt = dt  # int (in min)

	def __str__(self):
		return str([self.day, self.o, self.d, self.ot, self.dt])

	def _missing_start(self):
		return (self.o == -1 or self.ot == -1)

	def _missing_end(self):
		return (self.d == -1 or self.dt == -1)

	def incomplete(self):
		return self._missing_start() or self._missing_end()

	def _get_time(self):
		if self.ot != -1:
			return self.ot
		else:
			return self.dt

	def getTime(self, unit=1):
		if unit == 1:
			return self._get_time()
		else:
			return int(self._get_time() / unit)

	def getAbsTime(self, unit=1):
		return self.getTime(unit) + int(self.day * 1440 / unit)

	def getHour(self):
		return self.getTime(60)

	def getO(self):
		return self.o

	def getD(self):
		return self.d

	def getOD(self, asTuple=False):
		if asTuple:
			return (self.o, self.d)
		else:
			return self.o + '-' + self.d

	def syntaxExpression(self):
		return [str(self.getTime(60)), self.getO(), self.getD()]
