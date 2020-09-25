import numpy as np

# Returned data structure:
# [[[[x1,x2,x3...],[y1,y2,y3,...],[z1,z2,z3,...]],segment2,segment3,...],event2,event3,...]
def OrganizeEventData(fileName,eventsToDraw,drawAllEvents):
	recoEventNum,recoSegmentNum,recoPointNum,recoX,recoY,recoZ = np.loadtxt(fileName,unpack = True,skiprows=1)

	# Convert to integers to be used as list index
	recoEventNum = recoEventNum.astype(int)
	recoSegmentNum = recoSegmentNum.astype(int)
	recoPointNum = recoPointNum.astype(int)
	 
	# Get the maximums
	recoMaxSeg = np.amax(recoSegmentNum)
	maxEventNum = np.amax(recoEventNum)				# Currently not used
	# Events begin counting at 1 and go to maxEventNum.
	# Segments begin counting at 0 and go to recoMaxSeg for that event. (for the events that contained the maximum, for the other events we will later chop off the empty data)

	# Setting some initial values.
	segmentNum = recoSegmentNum[0] 					# Initial entry's initial segment num (should be 0)
	eventNum = recoEventNum[0]					# Initial entry's event num (normally 1)

	recoData = [[[[],[],[]] for segment in np.arange(0,recoMaxSeg+1)] for event in np.arange(0,maxEventNum)] # +1 comes from the fact that segments go from 0 to recoMaxSeg, meaning there are recoMaxSeg+1 total segments!
	# Build the recoData list using the following structure: (easily read from right to left)
	# [[[[x1,x2,x3...],[y1,y2,y3,...],[z1,z2,z3,...]],segment2,segment3,...],event2,event3,...]
	# Recall that events start at 1 but segments start at 0

	for entry in np.arange(0,len(recoEventNum)):	# Loop through every entry
		eventNum = recoEventNum[entry]
		if (eventNum-1 in eventsToDraw) or drawAllEvents:				# -1 comes from events starting to count at 1, not 0
			segmentNum = recoSegmentNum[entry]
			
			recoData[eventNum-1][segmentNum][0].append(recoX[entry])	# -1 comes from events starting to count at 1, not 0
			recoData[eventNum-1][segmentNum][1].append(recoY[entry])
			recoData[eventNum-1][segmentNum][2].append(recoZ[entry])

	# From here on, events start counting at zero because we get the index of the event from the position in the recoData list.

	# Remove the entries of recoData that are empty. (when that event have a number of segments less than recoMaxSeg)
	emptySegment = [[],[],[]]
	for event in recoData:
		try:
			while True:
				event.remove(emptySegment)
		except ValueError:
			pass
		
	return recoData

# Returned data structure:
# [[[x,y,z],barycenter2,barycenter3,...],event2, event3,....]
def GetBarycenters(recoData,eventsToDraw,drawAllEvents):
	barycenters = [[] for event in recoData]
	for eventNum in np.arange(0,len(recoData)):
		if (eventNum in eventsToDraw) or drawAllEvents:
			for segmentNum in np.arange(0,len(recoData[eventNum])-1):
				barycenters[eventNum].append([np.average(recoData[eventNum][segmentNum][0]),np.average(recoData[eventNum][segmentNum][1]),np.average(recoData[eventNum][segmentNum][2])])
	
	return barycenters

# Returned data structures:
# [[[A,B,C,D],segment2,segment3,...],event2,event3,...]
# where [A,B,C,D] is defined by the linear fit of each segment.
# z = A*x + B
# y = C*z + D
def GetLinearFitParameters(recoData,eventsToDraw,drawAllEvents):
	linearFitParameters = [[[] for segment in event if segment != event[len(event)-1]] for event in recoData] # The if statement is there because we won't take the linear fit of the last, non-14 cm segment

	for eventNum in np.arange(0,len(recoData)):

		if (eventNum in eventsToDraw) or drawAllEvents:
			event = recoData[eventNum]
		
			for segmentNum in np.arange(0,len(event)-1):		# The -1 comes from not doing the last segment since it's not 14 cm.
				
				# Used in least squares regression
				SSxz = 0
				SSxx = 0
				SSzy = 0
				SSzz = 0

				# Segment data
				segment = event[segmentNum]
				#xAvg = barycenters[eventNum][segmentNum][0]
				#yAvg = barycenters[eventNum][segmentNum][1]
				#zAvg = barycenters[eventNum][segmentNum][2]
				
				xTrajectoryPoints = segment[0]
				yTrajectoryPoints = segment[1]
				zTrajectoryPoints = segment[2]
				
				xAvg = np.mean(xTrajectoryPoints)
				yAvg = np.mean(yTrajectoryPoints)
				zAvg = np.mean(zTrajectoryPoints)
				
				for x,y,z in zip(xTrajectoryPoints,yTrajectoryPoints,zTrajectoryPoints):
					SSxz += (x-xAvg)*(z-zAvg)
					SSxx += (x-xAvg)*(x-xAvg)
					SSzy += (y-yAvg)*(z-zAvg)
					SSzz += (z-zAvg)*(z-zAvg)
				
				# Calculate linear fit parameters
				# z = A*x+B
				# y = C*z + D
				A = SSxz / SSxx
				B = zAvg - A*xAvg
				C = SSzy / SSzz
				D = yAvg - C*zAvg
				linearFitParameters[eventNum][segmentNum] = [A,B,C,D]
	return linearFitParameters

# Returned data structures:
# [[[[x1,y1,z1],[x2,y2,z2]],linearSegment2,linearSegment3,...],event2, event3, ...]
#    \___VV___/ \___VV___/
#      Point 1    Point 2
# The distance between Points 1 and 2 will be controlled by the parameter segmentLength, so that we can easily visualize the angles.
def GetLinearFitEndPoints(linearFitParameters,barycenters,segmentLength,eventsToDraw,drawAllEvents):
	linearFitEndPoints = [[[] for segment in event] for event in linearFitParameters]

	for eventNum in np.arange(0,len(linearFitParameters)):

		if (eventNum in eventsToDraw) or drawAllEvents:
			event = linearFitParameters[eventNum]
			
			for segmentNum in np.arange(0,len(event)):
				# Get the first and last (x,y,z) of this segment.

				# z = A*x + B
				# y = C*z + D
				A = linearFitParameters[eventNum][segmentNum][0]
				B = linearFitParameters[eventNum][segmentNum][1]
				C = linearFitParameters[eventNum][segmentNum][2]
				D = linearFitParameters[eventNum][segmentNum][3]
				
				# Segment Directional Length Projections given the linear fit parameters
				deltaX = segmentLength / np.sqrt(1 + A**2 + (A*C)**2)
				deltaZ = A*deltaX
				deltaY = C*deltaZ
				
				xAvg = barycenters[eventNum][segmentNum][0]
				yAvg = barycenters[eventNum][segmentNum][1]
				zAvg = barycenters[eventNum][segmentNum][2]
				
				xPlus = xAvg + deltaX / 2
				xMinus = xAvg - deltaX / 2
				zPlus = A*xPlus + B
				zMinus = A*xMinus + B
				yPlus = C*zPlus + D
				yMinus = C*zMinus + D
				
				linearFitEndPoints[eventNum][segmentNum].append([[xMinus,yMinus,zMinus],[xPlus,yPlus,zPlus]])

	return linearFitEndPoints
