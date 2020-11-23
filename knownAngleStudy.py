import numpy as np
import newMCS as MCS
import matplotlib.pyplot as plt

# Parameters
truePosInfo = "500_mu_1_GeV_start_beam-entry_dir_35_-45_truePosInfo.txt"
recoPosInfo = "500_mu_1_GeV_start_beam-entry_dir_35_-45_recoPosInfo.txt"
maxEventNum = 500
eventList = [0,1] # List of events we are working with
ignoreEventList = True

# Sort trajectory points corresponding to each event into a List
#events = MCS.OrganizeTrueDataIntoEvents(truePosInfo)
events = MCS.OrganizeRecoDataIntoEvents(recoPosInfo)

# Remove all events that are not in eventList, unless ignoreEventList is True
if not ignoreEventList:
	tempEvents = []
	for eventNum in np.arange(0,len(events)):
		if eventNum in eventList:
			tempEvents.append(events[eventNum])
			
	events = tempEvents

# If you want to know the eventNum of events[eventIndex], use eventList[eventIndex]
# This code here allows this to be true, regardless of ignoreEventList
if ignoreEventList:
	eventList = [i for i in np.arange(0,maxEventNum)]

# Sort each track's data into segments, forcing the segment length to be exactly the same
# sortedTrueData Organizational structure:
# [[[[[x1,y1,z1],trajectoryPoint2,trajectoryPoint3,...],segment2,segment3,...],track2,track3,...],event2,event3,...]
sortedTrueData = []
for eventIndex in np.arange(0,len(events)):
	event = events[eventIndex]
	sortedTrueData.append([])
	for trackIndex in np.arange(0,len(event)):
		track = event[trackIndex]
		sortedData = MCS.OrganizeTrackDataIntoSegments(track,14.0,False)
		sortedTrueData[eventIndex].append(sortedData)

# Get Barycenters of each segment in each track in each event
# Does not get the barycenter of the final segment, which is not 14-cm
trueBarycentersList = []
for eventIndex in np.arange(0,len(sortedTrueData)):
	event = sortedTrueData[eventIndex]
	trueBarycentersList.append([])
	for trackIndex in np.arange(0,len(event)):
		track = event[trackIndex]
		trueBarycentersList[eventIndex].append(MCS.GetBarycenters(track))

# Get Linear Fit Parameters of each segment in each track in each event
# Does not get the fit parameters of the final segment, which is not 14-cm
trueLinearFitParametersList = []
for eventIndex in np.arange(0,len(sortedTrueData)):
	event = sortedTrueData[eventIndex]
	trueLinearFitParametersList.append([])
	for trackIndex in np.arange(0,len(event)):
		track = event[trackIndex]
		trueLinearFitParametersList[eventIndex].append(MCS.GetLinearFitParameters(track))

# Get Polygonal Angles from each segment in each track, using the barycenters
truePolygonalAngles = []
for eventIndex in np.arange(0,len(sortedTrueData)):
	event = sortedTrueData[eventIndex]
	truePolygonalAngles.append([])
	for trackIndex in np.arange(0,len(event)):
		barycenters = trueBarycentersList[eventIndex][trackIndex]
		truePolygonalAngles[eventIndex].append(MCS.GetPolygonalAngles(track,barycenters))

# Get Linear Angles from each segment in each event, using the linear fit parameters
trueLinearAngles = []
for eventIndex in np.arange(0,len(sortedTrueData)):
	event = sortedTrueData[eventIndex]
	trueLinearAngles.append([])
	for trackIndex in np.arange(0,len(event)):
		parameters = trueLinearFitParametersList[eventIndex][trackIndex]
		trueLinearAngles[eventIndex].append(MCS.GetLinearAngles(event,parameters))

# Recreate sigmaRMS, sigmaHL, and sigmaRES analysis plots
truePolygonalSigmaRMS_vals = MCS.GetSigmaRMS_vals(truePolygonalAngles)

# Recreate final MCS Momentum Estimation Methods:
# Likelihoods
# Momentum Loss
"""
#"""
