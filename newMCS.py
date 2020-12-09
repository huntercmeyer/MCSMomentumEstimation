import numpy as np
from scipy.stats import norm

# TODO: Add Documentation
# Description
#
# Input:
#
# Output:
#
# [[[[x1,y1,z1],trajectoryPoint2,trajectoryPoint3,...],track2,track3,...],event2,event3,...]
# Will contain multiple tracks if each track was longer than 100 cm
def OrganizeRecoDataIntoEvents(fileName):
	eventNum,trackNum,xVals,yVals,zVals = np.loadtxt(fileName,unpack = True,skiprows=1)
	
	# Convert to integers to be used as list index
	eventNum = eventNum.astype(int)
	trackNum = trackNum.astype(int)
	maxEventNum = np.amax(eventNum)
	
	eventData = [[] for event in np.arange(0,maxEventNum)]
	previousEvent = 0
	previousTrack = 0
	eventData[eventNum[0]-1].append([])
	for entry in np.arange(0,len(eventNum)):
		event = eventNum[entry] - 1 # The -1 comes from events starting to count at 1, not 0
		track = trackNum[entry]
		if track > previousTrack or event > previousEvent:
			eventData[event].append([])
		if xVals[entry] != -999.0:
			eventData[event][track].append([xVals[entry],yVals[entry],zVals[entry]])
		
		previousEvent = event
		previousTrack = track

	return eventData

# Will be a function just like the reco one, except will sort away the momentum info
# And will separate into individual MCParticle containers, as opposed to Track containers.
# They're the same thing, so there's no point unless I was willing to make classes and things, which I'm not.
# This will allow everything else to work basically the same
def OrganizeTrueDataIntoEvents(fileName):
	eventNum,particleNum,momentum,xVals,yVals,zVals = np.loadtxt(fileName,unpack = True,skiprows=1)
	
	# Convert to integers to be used as list index
	eventNum = eventNum.astype(int)
	particleNum = particleNum.astype(int)
	maxEventNum = np.amax(eventNum)

	eventData = [[] for event in range(0,maxEventNum)]
	momentumData = [[] for event in range(0,maxEventNum)]
	
	previousEvent = 0
	previousParticle = 0
	eventData[eventNum[0]-1].append([]) # The -1 comes from events starting to count at 1, not 0
	momentumData[eventNum[0]-1].append([])
	for entry in range(0,len(eventNum)):
		event = eventNum[entry] - 1 # The -1 comes from events starting to count at 1, not 0
		particle = particleNum[entry]
		if particle > previousParticle or event > previousEvent:
			eventData[event].append([])
			momentumData[event].append([])
		eventData[event][particle].append([xVals[entry],yVals[entry],zVals[entry]])
		momentumData[event][particle].append(momentum[entry])
		
		previousEvent = event
		previousParticle = particle
	
	return (eventData, momentumData)

# Returned Organizational structure:
# [[[x1,y1,z1],trajectoryPoint2,trajectoryPoint3],segment2,segment3,...] = iTrack
def OrganizeTrackDataIntoSegments(track,segmentLength = 14.0,forceSegmentLength = False):
	# List of segments for this track
	segments = []

	# Loop through every trajectory point in this track.
	length = 0
	previousLength = 0
	previousDiff = 0
	segmentPoints = []
	segmentNum = 0
	segments.append([track[0]])
	for trajectoryPointNum in np.arange(1,len(track)):
		currentPoint = track[trajectoryPointNum]
		previousPoint = track[trajectoryPointNum-1]
		diff = np.sqrt((currentPoint[0]-previousPoint[0])**2 + (currentPoint[1]-previousPoint[1])**2 + (currentPoint[2]-previousPoint[2])**2)
		length += diff
		if length > 30:
			print(diff)
			print(previousDiff)
		if length > segmentLength and previousLength <= segmentLength:
			# Calculate the point that makes this 14 cm.
			# Add it to the list for this segment
			# Increase the segmentNum by 1, then add the new point to that segment.  Then add the current point to this and set the length to be correct.
			if forceSegmentLength:
				virtualPoint = []
				
				# Calculate the virutal point.
				multiplier = (segmentLength - previousLength)/(diff)
				virtualPoint.append(multiplier*(currentPoint[0]-previousPoint[0])+previousPoint[0]) # x_v
				virtualPoint.append(multiplier*(currentPoint[1]-previousPoint[1])+previousPoint[1]) # y_v
				virtualPoint.append(multiplier*(currentPoint[2]-previousPoint[2])+previousPoint[2]) # z_v
				
				# Fill last point of this segment: The virtual point.
				segments[segmentNum].append(virtualPoint)
				
				# Update to the next segment.
				segmentNum += 1
				
				# Fill the first point of the next segment: The virtual point, then fill the current point
				segments.append([])
				segments[segmentNum].append(virtualPoint)
				segments[segmentNum].append(currentPoint)
				
				# Correct the length to be the distance from the virtual point to the current point.
				length -= segmentLength
			else:
				if segmentLength - previousLength >= length - segmentLength:
					segments[segmentNum].append(currentPoint)
					segmentNum += 1
					segments.append([])
					segments[segmentNum].append(currentPoint)
					length = 0
				else:
					segmentNum += 1
					segments.append([])
					segments[segmentNum].append(previousPoint)
					segments[segmentNum].append(currentPoint)
					length = diff

		elif diff > 14:
			print("Solve")
			# Need to solve this issue.
			# is this even happening?
		else:
			segments[segmentNum].append(currentPoint)
		
		previousLength = length
		previousDiff = diff

	return segments

# TODO: Clean by grouping momentum and position data into one structure (class?)
# 	Per traj. point would be simplest...
# TODO: Every time we change OrganizeTrackDataIntoSegments, reflect changes here!
# Returned Organizational structure:
# [[[x1,y1,z1],trajectoryPoint2,trajectoryPoint3],segment2,segment3,...] = iParticle
def OrganizeParticleDataIntoSegments(particle,particleMomentum,segmentLength = 14.0,forceSegmentLength = False):
	# List of segments for this particle
	segments = []
	segmentMomentumAverages = [] # Only for 14-cm segments

	# Loop through every trajectory point in this particle.
	length = 0
	previousLength = 0
	previousDiff = 0
	segmentPoints = []
	segmentNum = 0
	segments.append([particle[0]])
	
	segmentNumPoints = 1
	currentMomentum = particleMomentum[0]
	segmentMomentumSum = currentMomentum
	
	for trajectoryPointNum in np.arange(1,len(particle)):
		currentPoint = particle[trajectoryPointNum]
		previousPoint = particle[trajectoryPointNum-1]
		diff = np.sqrt((currentPoint[0]-previousPoint[0])**2 + (currentPoint[1]-previousPoint[1])**2 + (currentPoint[2]-previousPoint[2])**2)
		length += diff

		currentMomentum = particleMomentum[trajectoryPointNum]
		#print("seg num = ", segmentNum, " curr mom = ",currentMomentum, )
		if length >= 2*segmentLength:
			
			# This is based on the assumption that length is always less than 3*segmentLength
			# We need to add two virtual points in this block
			if forceSegmentLength:
				virtualPoint1 = []
				multiplier1 = (segmentLength - previousLength)/diff
				virtualPoint1.append(multiplier1*(currentPoint[0]-previousPoint[0])+previousPoint[0]) #x_v1
				virtualPoint1.append(multiplier1*(currentPoint[1]-previousPoint[1])+previousPoint[1]) #y_v1
				virtualPoint1.append(multiplier1*(currentPoint[2]-previousPoint[2])+previousPoint[2]) #z_v1
				
				virtualPoint2 = []
				multiplier2 = (2*segmentLength - previousLength)/diff
				virtualPoint2.append(multiplier2*(currentPoint[0]-previousPoint[0])+previousPoint[0]) #x_v2
				virtualPoint2.append(multiplier2*(currentPoint[1]-previousPoint[1])+previousPoint[1]) #y_v2
				virtualPoint2.append(multiplier2*(currentPoint[2]-previousPoint[2])+previousPoint[2]) #z_v2
				
				segments[segmentNum].append(virtualPoint1)
				
				segments.append([])
				segmentNum += 1
				segments[segmentNum].extend([virtualPoint1,virtualPoint2])
				segmentMomentumSum = previousMomentum + currentMomentum
				segmentMomentumAverages.append(segmentMomentumSum / 2)
				
				segments.append([])
				segmentNum += 1
				segments[segmentNum].append(virtualPoint2)
				segments[segmentNum].append(currentPoint)
				length -= 2*segmentLength
				segmentMomentumSum = currentMomentum
			else:
				segmentMomentumAverages.append(segmentMomentumSum / segmentNumPoints)
				
				segments.append([])
				segmentNum += 1
				segments[segmentNum].append(previousPoint)
				segments[segmentNum].append(currentPoint)
				segmentMomentumSum = previousMomentum + currentMomentum
				segmentMomentumAverages.append(segmentMomentumSum / 2)
				
				segments.append([])
				segmentNum += 1
				segments[segmentNum].append(currentPoint)
				length = 0
				segmentMomentumSum = currentMomentum
				
		elif length > segmentLength and previousLength <= segmentLength:
			# Calculate the point that makes this 14 cm.
			# Add it to the list for this segment
			# Increase the segmentNum by 1, then add the new point to that segment.  Then add the current point to this and set the length to be correct.
			if forceSegmentLength:
				virtualPoint = []
				
				# Calculate the virutal point.
				multiplier = (segmentLength - previousLength)/(diff)
				virtualPoint.append(multiplier*(currentPoint[0]-previousPoint[0])+previousPoint[0]) # x_v
				virtualPoint.append(multiplier*(currentPoint[1]-previousPoint[1])+previousPoint[1]) # y_v
				virtualPoint.append(multiplier*(currentPoint[2]-previousPoint[2])+previousPoint[2]) # z_v
				
				# Fill last point of this segment: The virtual point.
				segments[segmentNum].append(virtualPoint)
				
				# Update to the next segment.
				segmentNum += 1
				
				# Fill the first point of the next segment: The virtual point, then fill the current point
				segments.append([])
				segments[segmentNum].append(virtualPoint)
				segments[segmentNum].append(currentPoint)
				
				# Correct the length to be the distance from the virtual point to the current point.
				length -= segmentLength
				
				segmentMomentumAverages.append(segmentMomentumSum / segmentNumPoints)
				segmentMomentumSum = currentMomentum # Reset for next segment
				segmentNumPoints = 1 # Reset for next segment
			else:
				if segmentLength - previousLength >= length - segmentLength:
					segments[segmentNum].append(currentPoint)
					segmentNum += 1
					segments.append([])
					segments[segmentNum].append(currentPoint)
					length = 0
					
					segmentMomentumSum += currentMomentum
					segmentNumPoints += 1
					#print("One",segmentNumPoints, segmentMomentumSum / segmentNumPoints)
					segmentMomentumAverages.append(segmentMomentumSum / segmentNumPoints)
					segmentMomentumSum = currentMomentum # Reset for next segment
					segmentNumPoints = 1 # Reset for next segment
				else:
					segmentNum += 1
					segments.append([])
					segments[segmentNum].append(previousPoint)
					segments[segmentNum].append(currentPoint)
					length = diff
					
					#print("Two",segmentNumPoints, segmentMomentumSum / segmentNumPoints)
					segmentMomentumAverages.append(segmentMomentumSum / segmentNumPoints)
					segmentMomentumSum = previousMomentum + currentMomentum # Reset for next segment
					segmentNumPoints = 2 # Reset for next segment
			
		else:
			segments[segmentNum].append(currentPoint)

			segmentMomentumSum += currentMomentum # true momentum of this point
			segmentNumPoints += 1
		
		previousLength = length
		previousDiff = diff
		previousMomentum = currentMomentum

	return segments, segmentMomentumAverages

# Returned Organizational structure:
# [[x_avg,y_avg,z_avg],segment2,segment3,...]
def GetBarycenters(track):
	barycenters = []
	for segmentNum in np.arange(0,len(track)-1): # -1 so that we don't take the barycenter of the last, non-14 cm segment.
		segment = track[segmentNum]
		transposedSegment = np.transpose(segment)
		
		xAvg = np.average(transposedSegment[0])
		yAvg = np.average(transposedSegment[1])
		zAvg = np.average(transposedSegment[2])
		
		barycenters.append([xAvg,yAvg,zAvg])
	
	return barycenters

# Returned Organizational structure:
# [[A,B,C,D],segment2,segment3,...]
def GetLinearFitParameters(track):
	linearFitParameters = []
	
	for segmentNum in np.arange(0,len(track)-1): # -1 so that we don't take the fit of the last non-14cm segment.
		
		# Segment data
		segment = track[segmentNum]
		transposedSegment = np.transpose(segment)
		
		xTrajectoryPoints = transposedSegment[0]
		yTrajectoryPoints = transposedSegment[1]
		zTrajectoryPoints = transposedSegment[2]
		
		xAvg = np.mean(xTrajectoryPoints)
		yAvg = np.mean(yTrajectoryPoints)
		zAvg = np.mean(zTrajectoryPoints)
		
		# Used in least squares regression
		SSxz = 0
		SSxx = 0
		SSzy = 0
		SSzz = 0
		
		# Least squares regression
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

		linearFitParameters.append([A,B,C,D])
		
	return linearFitParameters

# Get Polygonal Angles between segments
def GetPolygonalAngles(firstPoint, barycenters):
	thetaXZprimeList = []
	thetaYZprimeList = []

	x = barycenters[0][0] - firstPoint[0]
	y = barycenters[0][1] - firstPoint[1]
	z = barycenters[0][2] - firstPoint[2]
	thetaXZ = np.arctan(x/z)
	thetaYZ = np.arctan(y/z)
	
	for i in np.arange(0,len(barycenters)-1):
		v = [0,0,1]
		vPrime = [np.tan(thetaXZ),np.tan(thetaYZ),1]
		vPrime = (vPrime / np.linalg.norm(vPrime)).tolist()
		alpha = -np.arccos(np.dot(vPrime,v))
		u = np.cross(v,vPrime)
		u = (u / np.linalg.norm(u)).tolist()
		
		qs = np.cos(alpha/2)
		qx = np.sin(alpha/2)*u[0]
		qy = np.sin(alpha/2)*u[1]
		qz = np.sin(alpha/2)*u[2]
	
		x = barycenters[i+1][0] - barycenters[i][0]
		y = barycenters[i+1][1] - barycenters[i][1]
		z = barycenters[i+1][2] - barycenters[i][2]
	
		XYZ = [x,y,z]
		M = [[qs*qs+qx*qx-qy*qy-qz*qz,2*(qx*qy-qs*qz),2*(qs*qy+qx*qz)],[2*(qs*qz+qx*qy),qs*qs-qx*qx+qy*qy-qz*qz,2*(qy*qz-qs*qx)],[2*(qx*qz-qs*qy),2*(qs*qx+qy*qz),qs*qs-qx*qx-qy*qy+qz+qz]]
		
		XYZprime = np.dot(M,XYZ)
		thetaXZprime = np.arctan(XYZprime[0] / XYZprime[2])
		thetaYZprime = np.arctan(XYZprime[1] / XYZprime[2])
		
		thetaXZprimeList.append(thetaXZprime)
		thetaYZprimeList.append(thetaYZprime)

		thetaXZ = np.arctan(x/z)
		thetaYZ = np.arctan(y/z)

	return [thetaXZprimeList, thetaYZprimeList]

# Get Linear Angles between segments
# Doesn't actually use the track function parameter!
def GetLinearAngles(linearFitParameters):
	thetaXZprimeList = []
	thetaYZprimeList = []
	
	x = 1
	z = linearFitParameters[0][0]
	y = linearFitParameters[0][2]*linearFitParameters[0][0]
	thetaXZ = np.arctan(x/z)
	thetaYZ = np.arctan(y/z)
	
	for parametersIndex in np.arange(1,len(linearFitParameters)):
		parameters = linearFitParameters[parametersIndex]
	
		v = [0,0,1]
		vPrime = [np.tan(thetaXZ),np.tan(thetaYZ),1]
		vPrime = (vPrime / np.linalg.norm(vPrime)).tolist()
		alpha = -np.arccos(np.dot(vPrime,v))
		u = np.cross(v,vPrime)
		u = (u / np.linalg.norm(u)).tolist()
		
		qs = np.cos(alpha/2)
		qx = np.sin(alpha/2)*u[0]
		qy = np.sin(alpha/2)*u[1]
		qz = np.sin(alpha/2)*u[2]
		
		x = 1
		z = parameters[0]
		y = parameters[2]*parameters[0]
		
		XYZ = [x,y,z]
		M = [[qs*qs+qx*qx-qy*qy-qz*qz,2*(qx*qy-qs*qz),2*(qs*qy+qx*qz)],[2*(qs*qz+qx*qy),qs*qs-qx*qx+qy*qy-qz*qz,2*(qy*qz-qs*qx)],[2*(qx*qz-qs*qy),2*(qs*qx+qy*qz),qs*qs-qx*qx-qy*qy+qz+qz]]
		
		XYZprime = np.dot(M,XYZ)
		thetaXZprime = np.arctan(XYZprime[0] / XYZprime[2])
		thetaYZprime = np.arctan(XYZprime[1] / XYZprime[2])
		
		thetaXZprimeList.append(thetaXZprime)
		thetaYZprimeList.append(thetaYZprime)

		thetaXZ = np.arctan(x/z)
		thetaYZ = np.arctan(y/z)
	
	return [thetaXZprimeList, thetaYZprimeList]

# Provide angleList = [[[[thetaXZprime1,thetaXZprime2,thetaXZprime3,...],[thetaYZprime1,thetaYZprime2,thetaYZprime3,...],track2,track3,...],event2,event3,...]
# This function will group the data from lots of events,tracks, and segments and will group the data such that all angle measurements from segment1 will be grouped together, all angle measurements corresponding to segment 2 will be grouped together, and so on
# Returned format: [[[thetaXZprime_segment1_1, thetaXZprime_segment1_2, ...], XZangleMeasurementsXZ2, XZangleMeasurements3, ...],[[thetaYZprime_segment1_1, thetaYZprime_segment1_2, ...], YZangleMeasurements2, YZangleMeasurements3, ...]]
def GroupAngleDataIntoSegments(angleList):
	# Get the maximum angle num for building the data structure
	maxAngleNum = 0
	for event in angleList:
		for track in event:
			maxAngleNumForThisEvent = len(track[0])
			if maxAngleNum < maxAngleNumForThisEvent:
				maxAngleNum = maxAngleNumForThisEvent
	
	# Build the data structure that will host the angle data
	transposedThetaXZprimeList = [[] for angle in np.arange(0,maxAngleNum)]
	transposedThetaYZprimeList = [[] for angle in np.arange(0,maxAngleNum)]
	
	# Transpose the input data so it can be properly processed
	for eventIndex in np.arange(0,len(angleList)):
		event = angleList[eventIndex]
		for trackIndex in np.arange(0,len(event)):
			thetaXZprimeList,thetaYZprimeList = event[trackIndex]
			for angleIndex in np.arange(0,len(thetaXZprimeList)):
				transposedThetaXZprimeList[angleIndex].append(thetaXZprimeList[angleIndex])
				transposedThetaYZprimeList[angleIndex].append(thetaYZprimeList[angleIndex])
				
	return [transposedThetaXZprimeList, transposedThetaYZprimeList]



# Perform sigmaRMS, sigmaHL, and sigmaRES analysis
# Gaussian distribution of x with standard deviation of sigma
# This may need to be used in a scipy.optimize.curve_fit function (for uncertainties in PDF)
def Gauss(x,sigma,a,x_c):
	return a*np.exp(-(x-x_c)**2 / (2*sigma**2))

# Provide angleList = [[[thetaXZprime_segment1_1, thetaXZprime_segment1_2, ...], XZangleMeasurementsXZ2, XZangleMeasurements3, ...],[[thetaYZprime_segment1_1, thetaYZprime_segment1_2, ...], YZangleMeasurements2, YZangleMeasurements3, ...]]
def GetSigmaRMS_vals(groupedAngleMeasurements):
	sigmaRMS_vals = []
	"""
	# Get the maximum angle num for building the data structure
	maxAngleNum = 0
	for event in angleList:
		for track in event:
			maxAngleNumForThisEvent = len(track[0])
			if maxAngleNum < maxAngleNumForThisEvent:
				maxAngleNum = maxAngleNumForThisEvent
	
	# Build the data structure that will host the angle data
	transposedThetaXZprimeList = [[] for angle in np.arange(0,maxAngleNum)]
	transposedThetaYZprimeList = [[] for angle in np.arange(0,maxAngleNum)]
	
	# Transpose the input data so it can be properly processed
	for eventIndex in np.arange(0,len(angleList)):
		event = angleList[eventIndex]
		for trackIndex in np.arange(0,len(event)):
			thetaXZprimeList,thetaYZprimeList = event[trackIndex]
			for angleIndex in np.arange(0,len(thetaXZprimeList)):
				transposedThetaXZprimeList[angleIndex].append(thetaXZprimeList[angleIndex])
				transposedThetaYZprimeList[angleIndex].append(thetaYZprimeList[angleIndex])
	"""
	# Calculate sigmaRMS for each segment based on transposed input data
	for angleListIndex in np.arange(0,len(groupedAngleMeasurements[0])):
		thetaXZprime_mu, thetaXZprime_stdev = norm.fit(groupedAngleMeasurements[0][angleListIndex])
		thetaYZprime_mu, thetaYZprime_stdev = norm.fit(groupedAngleMeasurements[1][angleListIndex])
		
		# Manually check that sigmaRMS values are correct by crosschecking with Root
		"""print()
		print("N =", len(groupedAngleMeasurements[0][angleListIndex]))
		
		print()
		print("ThetaXZprime sigmaRMS vals")
		print(angleListIndex, ":")
		print("mu =", thetaXZprime_mu)
		print("stdev =", thetaXZprime_stdev*1000)
		
		print()
		print("ThetaYZprime sigmaRMS vals")
		print(angleListIndex, ":")
		print("mu =", thetaYZprime_mu)
		print("stdev =", thetaYZprime_stdev*1000)
		
		print()
		"""
		
		sigmaRMS_vals.append([thetaXZprime_stdev,thetaYZprime_stdev])

	return sigmaRMS_vals

# Implement MCS methods for estimating momentum
