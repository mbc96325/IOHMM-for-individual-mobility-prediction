# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:36:39 2021

@author: zhanzhao
"""

import csv

from trip import trip
from user import user


def stationNameDict(filepath):
	rd = csv.reader(open(filepath, 'rU'), delimiter=",")
	Dict = {}
	for s in rd:
		Dict[s[0]] = s[-1]
	# Special situations
	# Dict["-1"] = "Unknown"
	# Dict["0"] = "Unknown"
	return Dict


class panelDataReader:
	def __init__(self, file):
		self.reader = csv.reader(open(file), delimiter=",")
		self.header = next(self.reader)
		self.lastRecord = None

	def nextRecord(self):
		try:
			line = next(self.reader)
		except StopIteration:
			line = None
		self.lastRecord = line
		return line

	def nextUserRecords(self):
		userIndex = self.header.index("user_id")
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





def readPanelData_baichuan(file, stationDictPath=None, limit=None):
	print('Importing users...')
	if stationDictPath is not None:
		stationDict = stationNameDict(stationDictPath)
	else:
		stationDict = None

	panelReader = panelDataReader(file)
	# indices for later use
	headers = panelReader.header
	userIndex = headers.index("user_id")
	entryDateIndex = headers.index("entry_date")
	entryTimeIndex = headers.index("entry_time")
	#exitDateIndex = headers.index("exit_date")
	exitTimeIndex = headers.index("exit_time")
	entryStationIndex = headers.index("entry_station_id")
	exitStationIndex = headers.index("exit_station_id")

	X = []
	counter = 0
	userRecords = panelReader.nextUserRecords()
	# If there is more data to be processed
	while userRecords is not None:
		userID = userRecords[0][userIndex]  # new user ID
		userTripList = []  # new user tripList
		for i in range(len(userRecords)):
			tap = userRecords[i]
			# convert day and time to integer
			daykey = int(int(tap[entryDateIndex]) / 3600 / 24)
			inTime = int(tap[entryTimeIndex])
			outTime = int(tap[exitTimeIndex])
			inStation = tap[entryStationIndex]
			outStation = tap[exitStationIndex]

			# Add new trip
			#if inStation != "-1" and outStation != "-1":
			if True:
				# define a new trip object
				newTrip = trip(day=daykey,
								o=inStation,
								d=outStation,
								ot=inTime,
								dt=outTime)

				# Exclude trips with same entry and exit stations
				# if newTrip.o == newTrip.d:
				# 	continue

				# Convert station id to station name
				if stationDict is not None:
					newTrip.o = stationDict[newTrip.o]
					newTrip.d = stationDict[newTrip.d]


				# Exclude incomplete trips with unknown OD pairs
				# if newTrip.incomplete() is True or\
				# 	newTrip.o == "Unknown" or newTrip.d == "Unknown":
				# 	newTrip = None
				# 	continue

				#
				# # If two trips with same starting time, use the latest one
				# if len(userTripList) > 0:
				# 	if newTrip.day == userTripList[-1].day and \
				# 		newTrip.getAbsTime() == userTripList[-1].getAbsTime():
				# 		userTripList[-1] = newTrip
				# 		continue
				#

				# # Data errors
				# if len(userTripList) > 0:
				# 	if newTrip.day == userTripList[-1].day and \
				# 		newTrip.getAbsTime() < userTripList[-1].getAbsTime():
				# 		continue
				#

				userTripList.append(newTrip)  # Add the new trip to the user's list
		# Define a new user and add to a user list
		userTripList.sort
		newUser = user(userID, tripList=userTripList)
		X.append(newUser)
		# Print progress (number of users processed...)
		counter += 1
		if counter % 10000 == 0:
			print(counter)
			if limit:
				if counter >= limit:
					return X
		# Get next user's transactions, and start over
		userRecords = panelReader.nextUserRecords()
	return X

def readPanelData(file, stationDictPath=None, limit=None):
	print('Importing users...')
	if stationDictPath is not None:
		stationDict = stationNameDict(stationDictPath)
	else:
		stationDict = None

	panelReader = panelDataReader(file)
	# indices for later use
	headers = panelReader.header
	userIndex = headers.index("user_id")
	entryDateIndex = headers.index("entry_date")
	entryTimeIndex = headers.index("entry_time")
	#exitDateIndex = headers.index("exit_date")
	exitTimeIndex = headers.index("exit_time")
	entryStationIndex = headers.index("entry_station_id")
	exitStationIndex = headers.index("exit_station_id")

	X = []
	counter = 0
	userRecords = panelReader.nextUserRecords()
	# If there is more data to be processed
	while userRecords is not None:
		userID = userRecords[0][userIndex]  # new user ID
		userTripList = []  # new user tripList
		for i in range(len(userRecords)):
			tap = userRecords[i]
			# convert day and time to integer
			daykey = int(int(tap[entryDateIndex]) / 3600 / 24)
			inTime = int(tap[entryTimeIndex])
			outTime = int(tap[exitTimeIndex])
			inStation = tap[entryStationIndex]
			outStation = tap[exitStationIndex]

			# Add new trip
			if inStation != "-1" and outStation != "-1":
				# define a new trip object
				newTrip = trip(day=daykey,
								o=inStation,
								d=outStation,
								ot=inTime,
								dt=outTime)

				# Exclude trips with same entry and exit stations
				if newTrip.o == newTrip.d:
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
			print(counter)
			if limit:
				if counter >= limit:
					return X
		# Get next user's transactions, and start over
		userRecords = panelReader.nextUserRecords()
	return X