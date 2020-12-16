import numpy as np
import newMCS as MCS
import matplotlib.pyplot as plt
import sys

# MARK: Parameters
truePosInfo = "500_mu_1_GeV_start_beam-entry_dir_35_-45_truePosInfo.txt"
recoPosInfo = "500_mu_1_GeV_start_beam-entry_dir_35_-45_recoPosInfo.txt"
maxEventNum = 500
eventList = [0,1,2,3,4,5,6,7,8,9] # List of events we are working with
ignoreEventList = True
titleFontSize = 8
forceRecoSegmentLength = True
forceTrueSegmentLength = True

# Modify eventList if ignoreEventList = True
# 	If you want to know the eventNum of events[eventIndex], use eventList[eventIndex]
#	This code here allows this to be true, regardless of ignoreEventList
if ignoreEventList:
	eventList = [i for i in np.arange(0,maxEventNum)]

################################################################
# MARK: Section 1: Analyze Reco Tracks
################################################################

# ==============================================================
# Process recoPosInfo so that it is separated by events and by tracks (segments are not yet formed)
# recoEvents data format:
# [[[[x1,y1,z1],trajectoryPoint2,trajectoryPoint3,...],track2,track3,...],event2,event3,...]
# ==============================================================

# Process for all events from recoPosInfo inputFile
recoEvents = MCS.OrganizeRecoDataIntoEvents(recoPosInfo)

# Extra processing to remove events that are not in the eventList, unless ignoreEventList = True
if not ignoreEventList:
	tempEvents = []
	for eventNum in np.arange(0,len(recoEvents)):
		if eventNum in eventList:
			tempEvents.append(recoEvents[eventNum])
			
	recoEvents = tempEvents

# ==============================================================
# Process recoEvents into segmentedRecoData
# segmentedRecoData data format:
# [[[[[x1,y1,z1],trajectoryPoint2,trajectoryPoint3,...],segment2,segment3,...],track2,track3,...],event2,event3,...]
# (just like recoEvents, except the list of trajectory points are now segmented
# Force segmentLength by setting forceRecoSegmentLength = True in parameters at the top
# ==============================================================

segmentedRecoData = []
for eventIndex in np.arange(0,len(recoEvents)):
	event = recoEvents[eventIndex]
	segmentedRecoData.append([])
	for trackIndex in np.arange(0,len(event)):
		track = event[trackIndex]
		sortedData = MCS.OrganizeTrackDataIntoSegments(track,14.0,forceRecoSegmentLength)
		segmentedRecoData[eventIndex].append(sortedData)

# ==============================================================
# Calculate segment lengths
# recoSegmentLengthsList data format:
# [[[length1,length2,length3,...],track2,track3,...],event2,event3,...] (Linear Lengths)
# ==============================================================

recoSegmentLengthsList = []
for eventNum in range(0,len(segmentedRecoData)):
	event = segmentedRecoData[eventNum]
	recoSegmentLengthsList.append([]) # Append an Event
	
	for trackNum in range(0,len(event)):
		track = event[trackNum]
		recoSegmentLengthsList[eventNum].append([]) # Append a Track
		
		for segment in track:
			segmentLength = 0
			#print(segment)
			for trajectoryPointNum in range(1,len(segment)):
				previousPoint = segment[trajectoryPointNum-1]
				currentPoint = segment[trajectoryPointNum]
				#print(previousPoint)
				#print(currentPoint)
				diff = np.sqrt((previousPoint[0]-currentPoint[0])**2+(previousPoint[1]-currentPoint[1])**2+(previousPoint[2]-currentPoint[2])**2)
				segmentLength += diff
			
			recoSegmentLengthsList[eventNum][trackNum].append(segmentLength)

# ==============================================================
# Process segmentedRecoData to form recoBarycentersList
# recoBarycentersList data format:
# [[[[x_avg,y_avg,z_avg],segment2,segment3,...],track2,track3,...],event2,event3,...]
# Does not get the fit parameters of the final segment, which is not 14-cm
# ==============================================================

recoBarycentersList = []
for eventIndex in np.arange(0,len(segmentedRecoData)):
	event = segmentedRecoData[eventIndex]
	recoBarycentersList.append([])
	for trackIndex in np.arange(0,len(event)):
		track = event[trackIndex]
		recoBarycentersList[eventIndex].append(MCS.GetBarycenters(track))

# ==============================================================
# Calculate recoPolygonalLengthsList
# recoPolygonalLengthsList data format:
# [[[length1,length2,length3,...],track2,track3,...],event2,event3,...] (Polygonal Lengths)
# ==============================================================
recoPolygonalLengthsList = []
for eventIndex in range(0,len(recoBarycentersList)):
	event = recoBarycentersList[eventIndex]
	recoPolygonalLengthsList.append([])
	for trackIndex in range(0,len(event)):
		track = recoBarycentersList[trackIndex]
		
		firstPoint = segmentedRecoData[eventIndex][trackIndex][0][0]
		segmentLength = np.sqrt((firstPoint[0]-track[0][0])**2 + (firstPoint[1]-track[0][1])**2 + (firstPoint[1]-track[0][2])**2)
		for barycenterIndex in range(1,len(track)):
			previousBarycenter = track[barycenterIndex-1]
			currentBarycenter = track[barycenterIndex]
			segmentLength += np.sqrt((previousBarycenter[0]-currentBarycenter[0])**2 + (previousBarycenter[1]-currentBarycenter[2])**2 + (previousBarycenter[2]-currentBarycenter[2])**2)
		
		recoPolygonalLengthsList[eventIndex].append(segmentLength)

# ==============================================================
# Process segmentedRecoData to form recoLinearFitParametersList
# recoLinearFitParametersList data format:
# [[[[A,B,C,D],segment2,segment3,...],track2,track3,...],event2,event3,...]
# Does not get the fit parameters of the final segment, which is not 14-cm
# ==============================================================

recoLinearFitParametersList = []
for eventIndex in np.arange(0,len(segmentedRecoData)):
	event = segmentedRecoData[eventIndex]
	recoLinearFitParametersList.append([])
	for trackIndex in np.arange(0,len(event)):
		track = event[trackIndex]
		recoLinearFitParametersList[eventIndex].append(MCS.GetLinearFitParameters(track))

# ==============================================================
# Process recoBarycentersList to form recoPolygonalAnglesList
# recoPolygonalAnglesList data format:
# [[[[thetaXZprime1,thetaXZprime2,thetaXZprime3,...],thetaYZprimeList],track2,track3,...],event2,event3,...] (polygonal angles)
# ==============================================================
recoPolygonalAnglesList = []
for eventIndex in np.arange(0,len(segmentedRecoData)):
	event = segmentedRecoData[eventIndex]
	recoPolygonalAnglesList.append([])
	for trackIndex in np.arange(0,len(event)):
		firstPoint = event[trackIndex][0][0]
		barycenters = recoBarycentersList[eventIndex][trackIndex]
		recoPolygonalAnglesList[eventIndex].append(MCS.GetPolygonalAngles(firstPoint,barycenters))

# ==============================================================
# Process recoLinearFitParametersList to form recoLinearAnglesList
# recoPolygonalAnglesList data format:
# [[[[thetaXZprime1,thetaXZprime2,thetaXZprime3,...],thetaYZprimeList],track2,track3,...],event2,event3,...] (linear angles)
# ==============================================================

recoLinearAnglesList = []
for eventIndex in np.arange(0,len(segmentedRecoData)):
	event = segmentedRecoData[eventIndex]
	recoLinearAnglesList.append([])
	for trackIndex in np.arange(0,len(event)):
		parameters = recoLinearFitParametersList[eventIndex][trackIndex]
		recoLinearAnglesList[eventIndex].append(MCS.GetLinearAngles(parameters))

# ==============================================================
# Calculate linear log likelihoods list using the linear angles and linear segment lengths
# recoLinearLogLikelihoodsList data format:
# [[[(p,-ln(L)),logLLH2,logLLH3,...],track2,track3,...],event2,event3,...]
# [p,-ln(L)] are a pair of momentum and the associated log likelihood
# ==============================================================

recoLinearLogLikelihoodsList = []
for eventIndex in range(0,len(recoLinearAnglesList)): #recoLinearAnglesList[0]
	eventAngleData = recoLinearAnglesList[eventIndex]
	eventSegmentLengthData = recoSegmentLengthsList[eventIndex]
	if len(eventAngleData) != len(eventSegmentLengthData):
		sys.exit("Lengths of list not matching (1)")
	recoLinearLogLikelihoodsList.append([])
	for trackIndex in range(0,len(eventAngleData)):
		print("this")
		trackAngleData = eventAngleData[trackIndex]
		trackSegmentLengthData = eventSegmentLengthData[trackIndex]
		if len(trackAngleData[0]) != len(trackSegmentLengthData)-2:
			sys.exit("Lengths of list not matching (2)")
		
		thetaXZprimeList = trackAngleData[0]
		thetaYZprimeList = trackAngleData[1]
		if len(thetaXZprimeList) != len(thetaYZprimeList):
			sys.exit("Lengths of list not matching (3)")
		
		possibleMomentumList = np.arange(0.1,6.0,0.01)
		logLLH_List = MCS.GetLogLikelihoods(thetaXZprimeList, thetaYZprimeList, trackSegmentLengthData[1:], possibleMomentumList)
		if len(logLLH_List) != len(possibleMomentumList):
			sys.exit("Lengths of list not matching (4)")
		recoLinearLogLikelihoodsList[eventIndex].append(logLLH_List)

# ==============================================================
# Calculate Linear MCSMomentum List using the linear log likelihoods
# recoLinearMCSMomentumList data format:
# [[track1_MCSMomentum,track2_MCSMomentum,track3_MCSMomentum,...],event2,event3,...]
# ==============================================================
recoLinearMCSMomentumList = []
for eventIndex in range(0,len(recoLinearLogLikelihoodsList)):
	eventLogLikelihoods = recoLinearLogLikelihoodsList[eventIndex]
	recoLinearMCSMomentumList.append([])
	for trackIndex in range(0,len(eventLogLikelihoods)):
		trackLogLikelihoods = eventLogLikelihoods[trackIndex]
		
		tranposed = np.transpose(trackLogLikelihoods)
		minIndex = np.argmin(tranposed[1])
		momentum = trackLogLikelihoods[minIndex][0]
		momentum += MCS.deltaP(1000*momentum,14)
		recoLinearMCSMomentumList[eventIndex].append(momentum)
		
		print("Event: ", eventIndex, "Track: ", trackIndex, "Momentum: ", momentum)

print(recoLinearMCSMomentumList)
# TODO: Get Polygonal Lengths List
# TODO: recoPolyLogLikelihoodsList
# TODO: recoPolyMCSMomentumList
# TODO: trueLinearLogLikelihoodsList
# TODO: trueLinearMCSMomentumList
# TODO: truePolyLogLikelihoodsList
# TODO: truePolyMCSMomentumList
# TODO: Plot MCS Momentum vs. True Momentum (v easy)
# TODO: Calculate and Plot sigmaHL vs. Segment Number
# TODO: Calculate and Plot sigmaRES vs. Segment Number

################################################################
# MARK: Section 2: Analyze True Tracks (MCParticle's)
################################################################

# ==============================================================
# Process truePosInfo so that it is separated by events and by particles (segments are not yet formed)
# trueEvents data format:
# [[[[x1,y1,z1],trajectoryPoint2,trajectoryPoint3,...],particle2,particle3,...],event2,event3,...]
#
# Also separate momentum info
# trueMomentumData format:
# [[[p1,p2,p3,...],particle2,particle3,...],event2,event3,...]
# where p#, the # corresponds to the trajectory point
# ==============================================================

# Process for all events from truePosInfo inputFile
trueEvents, trueMomentumData = MCS.OrganizeTrueDataIntoEvents(truePosInfo)

# Extra processing to remove events that are not in the eventList, unless ignoreEventList = True
if not ignoreEventList:
	tempEvents = []
	tempMomentumData = []
	for eventNum in np.arange(0,len(trueEvents)):
		if eventNum in eventList:
			tempEvents.append(trueEvents[eventNum])
			tempMomentumData.append(trueMomentumData[eventNum])
			
	trueEvents = tempEvents
	trueMomentumData = tempMomentumData
	
# ==============================================================
# Process trueEvents into segmentedTrueData
# segmentedTrueData data format:
# [[[[[x1,y1,z1],trajectoryPoint2,trajectoryPoint3,...],segment2,segment3,...],particle2,particle3,...],event2,event3,...]
# (just like trueEvents, except the list of trajectory points are now segmented
# Force segmentLength by setting forceRecoSegmentLength = True in parameters at the top
#
# Average segmented momentum data into segmentAverageMomentumData (obv not including virtual points)
# segmentAverageMomentumData data format:
# [[[segMom1,segMom2,segMom3,...],particle2,particle3,...],event2,event3,...]
# ==============================================================

segmentedTrueData = []
segmentAverageMomentumData = []
for eventIndex in np.arange(0,len(trueEvents)):
	event = trueEvents[eventIndex]
	segmentedTrueData.append([])
	segmentAverageMomentumData.append([])
	for particleIndex in np.arange(0,len(event)):
		particle = event[particleIndex]
		particleMomentum = trueMomentumData[eventIndex][particleIndex]
		sortedPositionData, sortedMomentumAverages = MCS.OrganizeParticleDataIntoSegments(particle,particleMomentum,14.0,forceTrueSegmentLength)
		segmentedTrueData[eventIndex].append(sortedPositionData)
		segmentAverageMomentumData[eventIndex].append(sortedMomentumAverages)

# ==============================================================
# Calculate segment lengths
# trueSegmentLengthsList data format:
# [[[length1,length2,length3,...],particle2,particle3,...],event2,event3,...]
# ==============================================================

trueSegmentLengthsList = []
for eventNum in range(0,len(segmentedTrueData)):
	event = segmentedTrueData[eventNum]
	trueSegmentLengthsList.append([]) # Append an Event
	
	for particleNum in range(0,len(event)):
		trueSegmentLengthsList[eventNum].append([]) # Append a Particle
		
		for segment in track:
			segmentLength = 0
			for trajectoryPointNum in range(1,len(segment)):
				previousPoint = segment[trajectoryPointNum-1]
				currentPoint = segment[trajectoryPointNum]
				
				diff = np.sqrt((previousPoint[0]-currentPoint[0])**2+(previousPoint[1]-currentPoint[1])**2+(previousPoint[2]-currentPoint[2])**2)
				segmentLength += diff
			
			trueSegmentLengthsList[eventNum][particleNum].append(segmentLength)

# ==============================================================
# Process segmentedTrueData to form trueBarycentersList
# trueBarycentersList data format:
# [[[[x_avg,y_avg,z_avg],segment2,segment3,...],particle2,particl3,...],event2,event3,...]
# Does not get the fit parameters of the final segment, which is not 14-cm
# ==============================================================

trueBarycentersList = []
for eventIndex in np.arange(0,len(segmentedTrueData)):
	event = segmentedTrueData[eventIndex]
	trueBarycentersList.append([])
	for particleIndex in np.arange(0,len(event)):
		particle = event[particleIndex]
		trueBarycentersList[eventIndex].append(MCS.GetBarycenters(particle))

# ==============================================================
# Process segmentedTrueData to form trueLinearFitParametersList
# trueLinearFitParametersList data format:
# [[[[A,B,C,D],segment2,segment3,...],particle2,particle3,...],event2,event3,...]
# Does not get the fit parameters of the final segment, which is not 14-cm
# ==============================================================

trueLinearFitParametersList = []
for eventIndex in np.arange(0,len(segmentedTrueData)):
	event = segmentedTrueData[eventIndex]
	trueLinearFitParametersList.append([])
	for particleIndex in np.arange(0,len(event)):
		particle = event[particleIndex]
		trueLinearFitParametersList[eventIndex].append(MCS.GetLinearFitParameters(particle))

# ==============================================================
# Process trueBarycentersList to form truePolygonalAnglesList
# truePolygonalAnglesList data format:
# [[[[thetaXZprime1,thetaXZprime2,thetaXZprime3,...],thetaYZprimeList],particle2,particle3,...],event2,event3,...] (polygonal angles)
# ==============================================================
truePolygonalAnglesList = []
for eventIndex in np.arange(0,len(segmentedTrueData)):
	event = segmentedTrueData[eventIndex]
	truePolygonalAnglesList.append([])
	for particleIndex in np.arange(0,len(event)):
		firstPoint = event[particleIndex][0][0]
		barycenters = trueBarycentersList[eventIndex][particleIndex]
		truePolygonalAnglesList[eventIndex].append(MCS.GetPolygonalAngles(firstPoint,barycenters))

# ==============================================================
# Process trueLinearFitParametersList to form trueLinearAnglesList
# truePolygonalAnglesList data format:
# [[[[thetaXZprime1,thetaXZprime2,thetaXZprime3,...],thetaYZprimeList],particle2,particle3,...],event2,event3,...] (linear angles)
# ==============================================================

trueLinearAnglesList = []
for eventIndex in np.arange(0,len(segmentedTrueData)):
	event = segmentedTrueData[eventIndex]
	trueLinearAnglesList.append([])
	for particleIndex in np.arange(0,len(event)):
		parameters = trueLinearFitParametersList[eventIndex][particleIndex]
		trueLinearAnglesList[eventIndex].append(MCS.GetLinearAngles(parameters))
"""
# Print checks!
for eventIndex in range(0,len(segmentedTrueData)):
	particlePositions = segmentedTrueData[eventIndex][0]
	particleMomentums = segmentAverageMomentumData[eventIndex][0]
	particleBarycenters = trueBarycentersList[eventIndex][0]
	particleLinearFitParameters = trueLinearFitParametersList[eventIndex][0]
	particlePolyAngles = truePolygonalAnglesList[eventIndex][0]
	particleLinearAngles = trueLinearAnglesList[eventIndex][0]
	
	print("Event: ", eventIndex)
	print(len(particlePositions),"compare with",len(segmentedRecoData[eventIndex][0]))
	print(len(particleMomentums))
	print(len(particleBarycenters))
	print(len(particleLinearFitParameters))
	print(len(particlePolyAngles[0]))
	print(len(particleLinearAngles[0]))
	
	print()

for eventIndex in range(0,len(segmentedTrueData)):
	event = segmentedTrueData[eventIndex]
	print("Event:",eventIndex)
	for particle in event:
		for segmentNum in range(0,len(particle)):
			segment = particle[segmentNum]
			segmentLength = 0
			for trajectoryPointNum in range(1,len(segment)):
				previousPoint = segment[trajectoryPointNum-1]
				currentPoint = segment[trajectoryPointNum]
				
				diff = np.sqrt((previousPoint[0]-currentPoint[0])**2+(previousPoint[1]-currentPoint[1])**2+(previousPoint[2]-currentPoint[2])**2)
				segmentLength += diff
			if segmentLength > 28:
				print("Segment: ", segmentNum, "length = ",segmentLength)

for eventIndex in range(0,len(segmentedRecoData)):
	event = segmentedRecoData[eventIndex]
	print("Event:",eventIndex)
	for track in event:
		for segmentNum in range(0,len(track)):
			segment = track[segmentNum]
			segmentLength = 0
			for trajectoryPointNum in range(1,len(segment)):
				previousPoint = segment[trajectoryPointNum-1]
				currentPoint = segment[trajectoryPointNum]
				
				diff = np.sqrt((previousPoint[0]-currentPoint[0])**2+(previousPoint[1]-currentPoint[1])**2+(previousPoint[2]-currentPoint[2])**2)
				segmentLength += diff
			if segmentLength > 15:
				print("Segment: ", segmentNum, "length = ",segmentLength)
"""

# There are several events where weird stuff happens, but it's necessary to worry about now.
# TODO: Plot # of segments, specifically with or without forced seg lengths

# Implement MCS Momentum Estimation Method for True Particles

################################################################
# MARK: Section 3: sigmaRMS, sigmaHL, and sigmaRES analysis
################################################################

# ==============================================================
# Reprocess recoPolygonalAnglesList and recoLinearAnglesList so that they're grouped by segment, rather than by track/event
# This reduces the size of the array to the form:
# [[[thetaXZprime_segment1_1,thetaXZprime_segment1_2,...],XZangleMeasurementsXZ2,XZangleMeasurements3,...],segmentedThetaYZprimeList]
# We lose track and event separation from this!
# ==============================================================

recoGroupedPolyAngleMeasurements = MCS.GroupAngleDataIntoSegments(recoPolygonalAnglesList)
recoGroupedLinearAngleMeasurements = MCS.GroupAngleDataIntoSegments(recoLinearAnglesList)

trueGroupedPolyAngleMeasurements = MCS.GroupAngleDataIntoSegments(truePolygonalAnglesList)
trueGroupedLinearAngleMeasurements = MCS.GroupAngleDataIntoSegments(trueLinearAnglesList)

# Recreate sigmaRMS, sigmaHL, and sigmaRES analysis plots
recoPolygonalSigmaRMS_vals = MCS.GetSigmaRMS_vals(recoGroupedPolyAngleMeasurements)
recoLinearSigmaRMS_vals = MCS.GetSigmaRMS_vals(recoGroupedLinearAngleMeasurements)

truePolygonalSigmaRMS_vals = MCS.GetSigmaRMS_vals(trueGroupedPolyAngleMeasurements)
trueLinearSigmaRMS_vals = MCS.GetSigmaRMS_vals(trueGroupedLinearAngleMeasurements)

################################################################
# MARK: Section 4: Angle Plotting
################################################################

# Plot angle measurements for each segment
# First need to make the arrays to plots
poly_thetaXZprime_vals = recoGroupedPolyAngleMeasurements[0]
poly_thetaYZprime_vals = recoGroupedPolyAngleMeasurements[1]
poly_xVals = []
yVals_polyThetaXZprime = []
yVals_polyThetaYZprime = []
for segmentNum in range(0,len(poly_thetaXZprime_vals)):
	poly_xVals.extend([segmentNum for _ in range(0,len(poly_thetaXZprime_vals[segmentNum]))])
	yVals_polyThetaXZprime.extend(1000*x for x in poly_thetaXZprime_vals[segmentNum])
	yVals_polyThetaYZprime.extend(1000*x for x in poly_thetaYZprime_vals[segmentNum])

linear_thetaXZprime_vals = recoGroupedLinearAngleMeasurements[0]
linear_thetaYZprime_vals = recoGroupedLinearAngleMeasurements[1]
linear_xVals = []
yVals_linearThetaXZprime = []
yVals_linearThetaYZprime = []
for segmentNum in range(0,len(linear_thetaXZprime_vals)):
	linear_xVals.extend([segmentNum for _ in range(0,len(linear_thetaXZprime_vals[segmentNum]))])
	yVals_linearThetaXZprime.extend(1000*x for x in linear_thetaXZprime_vals[segmentNum])
	yVals_linearThetaYZprime.extend(1000*x for x in linear_thetaYZprime_vals[segmentNum])

fig = plt.figure()
ax1 = plt.subplot(221) # Poly XZprime
ax3 = plt.subplot(223) # Poly YZprime
ax2 = plt.subplot(222) # Linear XZprime
ax4 = plt.subplot(224) # Linear YZprime

ax1.hist2d(poly_xVals,yVals_polyThetaXZprime, bins = [np.amax(poly_xVals),400], range = [[0,np.amax(poly_xVals)],[-200,200]], cmin = 1)
ax1.set_title("Poly ThetaXZprime", fontsize = titleFontSize)
ax1.set_xlabel("Segment #")
ax1.set_ylabel("ThetaXZprime")

ax3.hist2d(poly_xVals,yVals_polyThetaYZprime, bins = [np.amax(poly_xVals),400], range = [[0,np.amax(poly_xVals)],[-200,200]], cmin = 1)
ax3.set_title("Poly ThetaYZprime", fontsize = titleFontSize)
ax3.set_xlabel("Segment #")
ax3.set_ylabel("ThetaYZprime")

ax2.hist2d(linear_xVals,yVals_linearThetaXZprime, bins = [np.amax(linear_xVals),400], range = [[0,np.amax(linear_xVals)],[-200,200]], cmin = 1)
ax2.set_title("Linear ThetaXZprime", fontsize = titleFontSize)
ax2.set_xlabel("Segment #")
ax2.set_ylabel("ThetaXZprime")

ax4.hist2d(linear_xVals,yVals_linearThetaYZprime, bins = [np.amax(linear_xVals),400], range = [[0,np.amax(linear_xVals)],[-200,200]], cmin = 1)
ax4.set_title("Linear ThetaYZprime", fontsize = titleFontSize)
ax4.set_xlabel("Segment #")
ax4.set_ylabel("ThetaYZprime")

plt.tight_layout()
plt.savefig("angles.png",bbox_inches = 'tight')


# Plot sigmaRMS

fig = plt.figure()
ax1 = plt.subplot(221) # PolyXZ
ax3 = plt.subplot(223) # PolyYZ
ax2 = plt.subplot(222) # LinearXZ
ax4 = plt.subplot(224) # LinearYZ

ax1.plot(range(0,len(np.transpose(recoPolygonalSigmaRMS_vals)[0])), np.transpose(recoPolygonalSigmaRMS_vals)[0],'o')
ax1.set_title("Poly ThetaXZprime sigmaRMS vs. Segment #", fontsize = titleFontSize)
ax1.set_xlabel("Segment #")
ax1.set_ylabel("sigmaRMS")

ax3.plot(range(0,len(np.transpose(recoPolygonalSigmaRMS_vals)[1])), np.transpose(recoPolygonalSigmaRMS_vals)[1],'o')
ax3.set_title("Poly ThetaYZprime sigmaRMS vs. Segment #", fontsize = titleFontSize)
ax3.set_xlabel("Segment #")
ax3.set_ylabel("sigmaRMS")

ax2.plot(range(0,len(np.transpose(recoLinearSigmaRMS_vals)[0])), np.transpose(recoLinearSigmaRMS_vals)[0],'o')
ax2.set_title("Linear ThetaXZprime sigmaRMS vs. Segment #", fontsize = titleFontSize)
ax2.set_xlabel("Segment #")
ax2.set_ylabel("sigmaRMS")

ax4.plot(range(0,len(np.transpose(recoLinearSigmaRMS_vals)[1])), np.transpose(recoLinearSigmaRMS_vals)[1],'o')
ax4.set_title("Linear ThetaYZprime sigmaRMS vs. Segment #", fontsize = titleFontSize)
ax4.set_xlabel("Segment #")
ax4.set_ylabel("sigmaRMS")

plt.tight_layout()
plt.savefig("sigmaRMS.png", bbox_inches = 'tight')

# Recreate final MCS Momentum Estimation Methods:
# Likelihoods
# Momentum Loss
"""
#"""

# Future check:
# Check if the same points in each segment for a segment with big diff

################################################################
# MARK: Section 5: Momentum Plotting
################################################################
fig = plt.figure()
ax1 = plt.subplot(121) # Linear
ax2 = plt.subplot(122) # Polygonal

# Collapse recoLinearMCSMomentumList down to a single array, from one that is separated by tracks
recoLinearMCSMomentum_vals = []
N = 0
for eventIndex in range(0,len(recoLinearMCSMomentumList)):
	event = recoLinearMCSMomentumList[eventIndex]
	recoLinearMCSMomentum_vals.extend(event)
	N += len(event)

print(N)
print(len(recoLinearMCSMomentum_vals))
print(recoLinearMCSMomentum_vals)
ax1.hist(recoLinearMCSMomentum_vals, bins = 100)
plt.savefig("linearMCSMomentum.png", bbox_inches = 'tight')
