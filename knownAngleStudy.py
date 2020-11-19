import numpy as np
import newMCS as MCS

# Parameters
truePosInfo = "500_mu_1_GeV_start_beam-entry_dir_35_-45_truePosInfo.txt"
recoPosInfo = "500_mu_1_GeV_start_beam-entry_dir_35_-45_recoPosInfo.txt"
maxEventNum = 500
eventList = [0] # List of events we are working with
ignoreEventList = False

# Sort trajectory points corresponding to each event into a List
#events = MCS.OrganizeTrueDataIntoEvents(truePosInfo)
events = MCS.OrganizeDataIntoEvents(recoPosInfo)

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
# Sort each event's data into segments, forcing the segment length to be exactly the same
# sortedTrueData Organizational structure:
# [[[[x1,y1,z1],trajectoryPoint2,trajectoryPoint3,...],segment2,segment3,...],event2,event3,...]
sortedTrueData = []
for eventIndex in np.arange(0,len(events)):
	event = events[eventIndex]
	sortedData = MCS.OrganizeEventDataIntoSegments(event,14.0,False)
	sortedTrueData.append(sortedData)

# Get Barycenters of each segment in each event
# Does not get the barycenter of the final segment, which is not 14-cm
trueBarycentersList = []
for eventIndex in np.arange(0,len(sortedTrueData)):
	event = sortedTrueData[eventIndex]
	trueBarycentersList.append(MCS.GetBarycenters(event))

# Get Linear Fit Parameters of each segment in each event
# Does not get the fit parameters of the final segment, which is not 14-cm
trueLinearFitParametersList = []
for eventIndex in np.arange(0,len(sortedTrueData)):
	event = sortedTrueData[eventIndex]
	trueLinearFitParametersList.append(MCS.GetLinearFitParameters(event))

# Get Polygonal Angles from each segment in each event, using the barycenters
truePolygonalAngles = []
for eventIndex in np.arange(0,len(sortedTrueData)):
	event = sortedTrueData[eventIndex]
	barycenters = trueBarycentersList[eventIndex]
	truePolygonalAngles.append(MCS.GetPolygonalAngles(event,barycenters))

# Get Linear Angles from each segment in each event, using the linear fit parameters
trueLinearAngles = []
for eventIndex in np.arange(0,len(sortedTrueData)):
	event = sortedTrueData[eventIndex]
	parameters = trueLinearFitParametersList[eventIndex]
	trueLinearAngles.append(MCS.GetLinearAngles(event,parameters))

print("True Linear Parameters:")
for parametersIndex in np.arange(0,len(trueLinearFitParametersList[0])):
	print(parametersIndex, trueLinearFitParametersList[0][parametersIndex])

print()
print("True Linear Angles:")
for index in np.arange(0,len(np.transpose(trueLinearAngles[0]))):
	print(index, np.transpose(trueLinearAngles[0])[index])

print()
print("True Polygonal Angles:")
for index in np.arange(0,len(np.transpose(truePolygonalAngles[0]))):
	print(index, np.transpose(truePolygonalAngles[0])[index])
