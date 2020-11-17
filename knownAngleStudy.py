import numpy as np
import newMCS as MCS

# Parameters
truePosInfo = "500_mu_1_GeV_start_beam-entry_dir_35_-45_truePosInfo.txt"
maxEventNum = 500
eventList = [0] # List of events we are working with
ignoreEventList = False

# Sort trajectory points corresponding to each event into a List
events = MCS.OrganizeTrueDataIntoEvents(truePosInfo)

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
	sortedData = MCS.OrganizeEventDataIntoSegments(event,14.0,True)
	sortedTrueData.append(sortedData)

# Get Barycenters of each event
trueBarycenters = []
for eventIndex in np.arange(0,len(sortedTrueData)):
	event = sortedTrueData[eventIndex]
	trueBarycenters.append(MCS.GetBarycenters(event))
