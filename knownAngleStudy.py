# TODO: Add a READ_ME

import numpy as np
import newMCS as MCS
import printerMCS
import matplotlib.pyplot as plt
import sys
import os
import shutil

# MARK: Parameters
sampleDescription = "500_mu_1_GeV_start_beam-entry_dir_35_-45" # description of sample being analyzed
samplePath = "KAS_pictures/" + sampleDescription + "/" # path of sample, not an example path
truePosInfo = sampleDescription + "_truePosInfo.txt"
recoPosInfo = sampleDescription + "_recoPosInfo.txt"
maxEventNum = 500
eventList = range(0,5) # List of events we are working with
ignoreEventList = False
titleFontSize = 8
forceRecoSegmentLength = False
forceTrueSegmentLength = False

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
# Includes final non-14 cm segment
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
			for trajectoryPointNum in range(1,len(segment)):
				previousPoint = segment[trajectoryPointNum-1]
				currentPoint = segment[trajectoryPointNum]
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
		barycenters = event[trackIndex]
		recoPolygonalLengthsList[eventIndex].append([])
		firstPoint = segmentedRecoData[eventIndex][trackIndex][0][0]
		segmentLength = np.sqrt((firstPoint[0]-barycenters[0][0])**2 + (firstPoint[1]-barycenters[0][1])**2 + (firstPoint[2]-barycenters[0][2])**2)
		recoPolygonalLengthsList[eventIndex][trackIndex].append(segmentLength)
		
		for barycenterIndex in range(1,len(barycenters)):
			previousBarycenter = barycenters[barycenterIndex-1]
			currentBarycenter = barycenters[barycenterIndex]
			segmentLength = np.sqrt((previousBarycenter[0]-currentBarycenter[0])**2 + (previousBarycenter[1]-currentBarycenter[1])**2 + (previousBarycenter[2]-currentBarycenter[2])**2)
			
			recoPolygonalLengthsList[eventIndex][trackIndex].append(segmentLength)

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
for eventIndex in range(0,len(recoLinearAnglesList)):
	eventAngleData = recoLinearAnglesList[eventIndex]
	eventSegmentLengthData = recoSegmentLengthsList[eventIndex]
	if len(eventAngleData) != len(eventSegmentLengthData):
		sys.exit("Lengths of list not matching (1)")
	recoLinearLogLikelihoodsList.append([])
	for trackIndex in range(0,len(eventAngleData)):
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
		
		print("Reco Linear Event: ", eventIndex, "Track: ", trackIndex, "Momentum: ", momentum)

# ==============================================================
# Calculate polygonal log likelihoods list using the polygonal angles and polygonal segment lengths
# recoPolyLogLikelihoodsList data format:
# [[[(p,-ln(L)),logLLH2,logLLH3,...],track2,track3,...],event2,event3,...]
# [p,-ln(L)] are a pair of momentum and the associated log likelihood
# ==============================================================
recoPolyLogLikelihoodsList = []
for eventIndex in range(0,len(recoPolygonalAnglesList)):
	eventAngleData = recoPolygonalAnglesList[eventIndex]
	eventSegmentLengthData = recoPolygonalLengthsList[eventIndex]
	if len(eventAngleData) != len(eventSegmentLengthData):
		sys.exit("Lengths of list not matching (1)")
	recoPolyLogLikelihoodsList.append([])
	for trackIndex in range(0,len(eventAngleData)):
		trackAngleData = eventAngleData[trackIndex]
		trackSegmentLengthData = eventSegmentLengthData[trackIndex]
		if len(trackAngleData[0]) != len(trackSegmentLengthData)-1:
			sys.exit("Lengths of list not matching (2)")
		
		thetaXZprimeList = trackAngleData[0]
		thetaYZprimeList = trackAngleData[1]
		if len(thetaXZprimeList) != len(thetaYZprimeList):
			sys.exit("Lengths of list not matching (3)")
		
		possibleMomentumList = np.arange(0.1,6.0,0.01)
		logLLH_List = MCS.GetLogLikelihoods(thetaXZprimeList, thetaYZprimeList, trackSegmentLengthData[1:], possibleMomentumList)
		if len(logLLH_List) != len(possibleMomentumList):
			sys.exit("Lengths of list not matching (4)")
		recoPolyLogLikelihoodsList[eventIndex].append(logLLH_List)

# ==============================================================
# Calculate Polygonal MCSMomentum List using the polygonal log likelihoods
# recoPolyMCSMomentumList data format:
# [[track1_MCSMomentum,track2_MCSMomentum,track3_MCSMomentum,...],event2,event3,...]
# ==============================================================
recoPolyMCSMomentumList = []
for eventIndex in range(0,len(recoPolyLogLikelihoodsList)):
	eventLogLikelihoods = recoPolyLogLikelihoodsList[eventIndex]
	recoPolyMCSMomentumList.append([])
	for trackIndex in range(0,len(eventLogLikelihoods)):
		trackLogLikelihoods = eventLogLikelihoods[trackIndex]
		
		tranposed = np.transpose(trackLogLikelihoods)
		minIndex = np.argmin(tranposed[1])
		momentum = trackLogLikelihoods[minIndex][0]
		momentum += MCS.deltaP(1000*momentum,14)
		recoPolyMCSMomentumList[eventIndex].append(momentum)
		
		print("Reco Poly Event: ", eventIndex, "Track: ", trackIndex, "Momentum: ", momentum)

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
		particle = event[particleNum]
		trueSegmentLengthsList[eventNum].append([]) # Append a particle
		
		for segment in particle:
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
# Calculate truePolygonalLengthsList
# truePolygonalLengthsList data format:
# [[[length1,length2,length3,...],particle2,particle3,...],event2,event3,...] (Polygonal Lengths)
# ==============================================================

truePolygonalLengthsList = []
for eventIndex in range(0,len(trueBarycentersList)):
	event = trueBarycentersList[eventIndex]
	truePolygonalLengthsList.append([])
	for particleIndex in range(0,len(event)):
		barycenters = event[particleIndex]
		truePolygonalLengthsList[eventIndex].append([])
		firstPoint = segmentedTrueData[eventIndex][particleIndex][0][0]
		segmentLength = np.sqrt((firstPoint[0]-barycenters[0][0])**2 + (firstPoint[1]-barycenters[0][1])**2 + (firstPoint[2]-barycenters[0][2])**2)
		truePolygonalLengthsList[eventIndex][particleIndex].append(segmentLength)
		
		for barycenterIndex in range(1,len(barycenters)):
			previousBarycenter = barycenters[barycenterIndex-1]
			currentBarycenter = barycenters[barycenterIndex]
			segmentLength = np.sqrt((previousBarycenter[0]-currentBarycenter[0])**2 + (previousBarycenter[1]-currentBarycenter[1])**2 + (previousBarycenter[2]-currentBarycenter[2])**2)
			
			truePolygonalLengthsList[eventIndex][particleIndex].append(segmentLength)

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

# There are several events where weird stuff happens, but it's necessary to worry about now.
# TODO: Plot # of segments, specifically with or without forced seg lengths

# ==============================================================
# Calculate linear log likelihoods list using the linear angles and linear segment lengths
# trueLinearLogLikelihoodsList data format:
# [[[(p,-ln(L)),logLLH2,logLLH3,...],particle2,particle3,...],event2,event3,...]
# [p,-ln(L)] are a pair of momentum and the associated log likelihood
# ==============================================================

trueLinearLogLikelihoodsList = []
for eventIndex in range(0,len(trueLinearAnglesList)):
	eventAngleData = trueLinearAnglesList[eventIndex]
	eventSegmentLengthData = trueSegmentLengthsList[eventIndex]
	if len(eventAngleData) != len(eventSegmentLengthData):
		sys.exit("Lengths of list not matching (1)")
	trueLinearLogLikelihoodsList.append([])
	for particleIndex in range(0,len(eventAngleData)):
		particleAngleData = eventAngleData[particleIndex]
		particleSegmentLengthData = eventSegmentLengthData[particleIndex]
		if len(particleAngleData[0]) != len(particleSegmentLengthData)-2:
			sys.exit("Lengths of list not matching (2)")
		
		thetaXZprimeList = particleAngleData[0]
		thetaYZprimeList = particleAngleData[1]
		if len(thetaXZprimeList) != len(thetaYZprimeList):
			sys.exit("Lengths of list not matching (3)")
		
		possibleMomentumList = np.arange(0.1,6.0,0.01)
		logLLH_List = MCS.GetLogLikelihoods(thetaXZprimeList, thetaYZprimeList, particleSegmentLengthData[1:], possibleMomentumList)
		if len(logLLH_List) != len(possibleMomentumList):
			sys.exit("Lengths of list not matching (4)")
		trueLinearLogLikelihoodsList[eventIndex].append(logLLH_List)

# ==============================================================
# Calculate Linear MCSMomentum List using the linear log likelihoods
# trueLinearMCSMomentumList data format:
# [[particle1_MCSMomentum,particle2_MCSMomentum,particle3_MCSMomentum,...],event2,event3,...]
# ==============================================================
trueLinearMCSMomentumList = []
for eventIndex in range(0,len(trueLinearLogLikelihoodsList)):
	eventLogLikelihoods = trueLinearLogLikelihoodsList[eventIndex]
	trueLinearMCSMomentumList.append([])
	for particleIndex in range(0,len(eventLogLikelihoods)):
		particleLogLikelihoods = eventLogLikelihoods[particleIndex]
		
		tranposed = np.transpose(particleLogLikelihoods)
		minIndex = np.argmin(tranposed[1])
		momentum = particleLogLikelihoods[minIndex][0]
		momentum += MCS.deltaP(1000*momentum,14)
		trueLinearMCSMomentumList[eventIndex].append(momentum)
		
		print("True Linear Event: ", eventIndex, "Particle: ", particleIndex, "Momentum: ", momentum)

# ==============================================================
# Calculate polygonal log likelihoods list using the polygonal angles and polygonal segment lengths
# truePolyLogLikelihoodsList data format:
# [[[(p,-ln(L)),logLLH2,logLLH3,...],particle2,particle3,...],event2,event3,...]
# [p,-ln(L)] are a pair of momentum and the associated log likelihood
# ==============================================================
truePolyLogLikelihoodsList = []
for eventIndex in range(0,len(truePolygonalAnglesList)):
	eventAngleData = truePolygonalAnglesList[eventIndex]
	eventSegmentLengthData = truePolygonalLengthsList[eventIndex]
	if len(eventAngleData) != len(eventSegmentLengthData):
		sys.exit("Lengths of list not matching (1)")
	truePolyLogLikelihoodsList.append([])
	for particleIndex in range(0,len(eventAngleData)):
		particleAngleData = eventAngleData[particleIndex]
		particleSegmentLengthData = eventSegmentLengthData[particleIndex]
		if len(particleAngleData[0]) != len(particleSegmentLengthData)-1:
			sys.exit("Lengths of list not matching (2)")
		
		thetaXZprimeList = particleAngleData[0]
		thetaYZprimeList = particleAngleData[1]
		if len(thetaXZprimeList) != len(thetaYZprimeList):
			sys.exit("Lengths of list not matching (3)")
		
		possibleMomentumList = np.arange(0.1,6.0,0.01)
		logLLH_List = MCS.GetLogLikelihoods(thetaXZprimeList, thetaYZprimeList, particleSegmentLengthData[1:], possibleMomentumList)
		if len(logLLH_List) != len(possibleMomentumList):
			sys.exit("Lengths of list not matching (4)")
		truePolyLogLikelihoodsList[eventIndex].append(logLLH_List)

# ==============================================================
# Calculate Polygonal MCSMomentum List using the polygonal log likelihoods
# truePolyMCSMomentumList data format:
# [[particle1_MCSMomentum,particle2_MCSMomentum,particle3_MCSMomentum,...],event2,event3,...]
# ==============================================================
truePolyMCSMomentumList = []
for eventIndex in range(0,len(truePolyLogLikelihoodsList)):
	eventLogLikelihoods = truePolyLogLikelihoodsList[eventIndex]
	truePolyMCSMomentumList.append([])
	for particleIndex in range(0,len(eventLogLikelihoods)):
		particleLogLikelihoods = eventLogLikelihoods[particleIndex]
		
		tranposed = np.transpose(particleLogLikelihoods)
		minIndex = np.argmin(tranposed[1])
		momentum = particleLogLikelihoods[minIndex][0]
		momentum += MCS.deltaP(1000*momentum,14)
		truePolyMCSMomentumList[eventIndex].append(momentum)
		
		print("True Poly Event: ", eventIndex, "Particle: ", particleIndex, "Momentum: ", momentum)

################################################################
# MARK: Section 3: sigmaRMS, sigmaHL, and sigmaRES analysis
################################################################

# ==============================================================
# Reprocess recoPolygonalAnglesList and recoLinearAnglesList so that they're grouped by segment, rather than by track/event
# This reduces the shape of the array to the form:
# [[[thetaXZprime_segment1_1,thetaXZprime_segment1_2,...],XZangleMeasurementsXZ2,XZangleMeasurements3,...],segmentedThetaYZprimeList]
# We lose track and event separation from this!
# ==============================================================

recoGroupedPolyAngleMeasurements = MCS.GroupAngleDataIntoSegments(recoPolygonalAnglesList)
recoGroupedLinearAngleMeasurements = MCS.GroupAngleDataIntoSegments(recoLinearAnglesList)

trueGroupedPolyAngleMeasurements = MCS.GroupAngleDataIntoSegments(truePolygonalAnglesList)
trueGroupedLinearAngleMeasurements = MCS.GroupAngleDataIntoSegments(trueLinearAnglesList)

# Recreate sigmaRMS, sigmaHL, and sigmaRES analysis plots
# Get sigmaHL values
# Sort true momentum values into segments just like the angles previously (reduce array shape)
maxSegmentNum = 0
for event in segmentAverageMomentumData:
	for particle in event:
		maxSegmentNumForThisEvent = len(particle)
		if maxSegmentNum < maxSegmentNumForThisEvent:
			maxSegmentNum = maxSegmentNumForThisEvent

groupedMomentumVals = [[] for segment in range(0,maxSegmentNum)]

for eventIndex in range(0,len(segmentAverageMomentumData)):
	event = segmentAverageMomentumData[eventIndex]
	for particleIndex in range(0,len(event)):
		particleMomentumVals = event[particleIndex]
		
		for segmentIndex in range(0,len(particleMomentumVals)):
			groupedMomentumVals[segmentIndex].append(particleMomentumVals[segmentIndex])

# Array has been reduced to form groupedMomentumVals, not use to calculate sigmaHL_vals
averageSegmentMomentumList = [np.mean(x) for x in groupedMomentumVals]
sigmaHL_vals = [MCS.highland(x, 14) for x in averageSegmentMomentumList] # TODO: Change this to use the correct length! for p,l in zip(averageSegmentMomentumList,lengthList[1:])???? Verify it's the same segment and same lengths or zip(averageSegmentMomentumList[1:], lengthList[2:] (or even [1:] idk!)
choppedSigmaHL_vals = sigmaHL_vals[1:] # used in sigmaRES_vals calculations

# Get sigmaRMS values
recoPolygonalSigmaRMS_vals = MCS.GetSigmaRMS_vals(recoGroupedPolyAngleMeasurements)
recoLinearSigmaRMS_vals = MCS.GetSigmaRMS_vals(recoGroupedLinearAngleMeasurements)

truePolygonalSigmaRMS_vals = MCS.GetSigmaRMS_vals(trueGroupedPolyAngleMeasurements)
trueLinearSigmaRMS_vals = MCS.GetSigmaRMS_vals(trueGroupedLinearAngleMeasurements)

# Get sigmaRES values:
# TODO: Add Documentation
recoPoly_sigmaRES_vals = MCS.GetSigmaRES_vals(recoPolygonalSigmaRMS_vals, choppedSigmaHL_vals)
recoLinear_sigmaRES_vals = MCS.GetSigmaRES_vals(recoLinearSigmaRMS_vals, choppedSigmaHL_vals)
truePoly_sigmaRES_vals = MCS.GetSigmaRES_vals(truePolygonalSigmaRMS_vals, choppedSigmaHL_vals)
trueLinear_sigmaRES_vals = MCS.GetSigmaRES_vals(trueLinearSigmaRMS_vals, choppedSigmaHL_vals)
# TODO: Potential restructure to be just like sigmaRMS? or change sigmaRMS to fit this?

recoPolyXZ_sigmaRES_vals, recoPolyYZ_sigmaRES_vals = recoPoly_sigmaRES_vals
recoLinearXZ_sigmaRES_vals, recoLinearYZ_sigmaRES_vals = recoLinear_sigmaRES_vals
truePolyXZ_sigmaRES_vals, truePolyYZ_sigmaRES_vals = truePoly_sigmaRES_vals
trueLinearXZ_sigmaRES_vals, trueLinearYZ_sigmaRES_vals = trueLinear_sigmaRES_vals

# print sigmaHL (it's important to note that choppedSigmaHL_vals is used in calculating sigmaRES_vals)
printerMCS.printSigmaHL_vals(sigmaHL_vals)

# print sigmaRMS
printerMCS.printSigmaRMS_vals(recoPolygonalSigmaRMS_vals, "recoPoly")
printerMCS.printSigmaRMS_vals(recoLinearSigmaRMS_vals, "recoLinear")
printerMCS.printSigmaRMS_vals(truePolygonalSigmaRMS_vals, "truePoly")
printerMCS.printSigmaRMS_vals(trueLinearSigmaRMS_vals, "trueLinear")

# print sigmaRES
printerMCS.printSigmaRES_vals(recoPoly_sigmaRES_vals,"recoPoly")
printerMCS.printSigmaRES_vals(recoLinear_sigmaRES_vals,"recoLinear")
printerMCS.printSigmaRES_vals(truePoly_sigmaRES_vals,"truePoly")
printerMCS.printSigmaRES_vals(trueLinear_sigmaRES_vals,"trueLinear")

# print sigmaRES averages
printerMCS.printAverageSigmaRES_vals([(recoPolyXZ_sigmaRES_vals,"recoPolyXZ"),(recoPolyYZ_sigmaRES_vals,"recoPolyYZ"),(recoLinearXZ_sigmaRES_vals,"recoLinearXZ"),(recoLinearYZ_sigmaRES_vals,"recoLinearYZ"),(truePolyXZ_sigmaRES_vals,"truePolyXZ"),(truePolyYZ_sigmaRES_vals,"truePolyYZ"),(trueLinearXZ_sigmaRES_vals,"trueLinearXZ"),(trueLinearYZ_sigmaRES_vals,"trueLinearYZ")])

# Make/Clear the directory before plotting and saving:
if os.path.exists(samplePath):
	shutil.rmtree(samplePath)
os.mkdir(samplePath)

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
ax1.set_title("Reconstructed Polygonal Angles", fontsize = titleFontSize)
ax1.set_xlabel("Segment #")
ax1.set_ylabel("$\\theta_{XZ}'$ (mrad)")

ax3.hist2d(poly_xVals,yVals_polyThetaYZprime, bins = [np.amax(poly_xVals),400], range = [[0,np.amax(poly_xVals)],[-200,200]], cmin = 1)
ax3.set_title("Reconstructed Polygonal Angles", fontsize = titleFontSize)
ax3.set_xlabel("Segment #")
ax3.set_ylabel("$\\theta_{YZ}'$ (mrad)")

ax2.hist2d(linear_xVals,yVals_linearThetaXZprime, bins = [np.amax(linear_xVals),400], range = [[0,np.amax(linear_xVals)],[-200,200]], cmin = 1)
ax2.set_title("Reconstructed Linear Fit Angles", fontsize = titleFontSize)
ax2.set_xlabel("Segment #")
ax2.set_ylabel("$\\theta_{XZ}'$ (mrad)")

ax4.hist2d(linear_xVals,yVals_linearThetaYZprime, bins = [np.amax(linear_xVals),400], range = [[0,np.amax(linear_xVals)],[-200,200]], cmin = 1)
ax4.set_title("Reconstructed Linear Fit Angles", fontsize = titleFontSize)
ax4.set_xlabel("Segment #")
ax4.set_ylabel("$\\theta_{YZ}'$ (mrad)")

plt.tight_layout()
plt.savefig(samplePath + "recoAnglesVsSegmentNumber.png",bbox_inches = 'tight')
# TODO: Add a caption? and suptitle?

# Plot Reconstructed sigmaRMS & sigmaHL vs. Segment Number

fig = plt.figure()
ax1 = plt.subplot(221) # PolyXZ
ax2 = plt.subplot(222) # PolyYZ
ax3 = plt.subplot(223) # LinearXZ
ax4 = plt.subplot(224) # LinearYZ

fig.suptitle("Reconstructed Angular Distribution Parameters vs. Segment Number")

ax1.plot(range(0,len(np.transpose(recoPolygonalSigmaRMS_vals)[0])), np.transpose(recoPolygonalSigmaRMS_vals)[0],'o', label = "$\sigma_{RMS}$")
ax1.plot(range(0,len(choppedSigmaHL_vals)), choppedSigmaHL_vals, '+', label = "$\sigma_{HL}$")
ax1.set_title("Polygonal $\\theta_{XZ}'$", fontsize = titleFontSize)
ax1.set_xlabel("Segment #")
ax1.set_ylabel("Angular Distribution Paras (rad)", fontsize = 'x-small')
ax1.set_ylim(bottom = 0)
ax1.legend()

ax2.plot(range(0,len(np.transpose(recoPolygonalSigmaRMS_vals)[1])), np.transpose(recoPolygonalSigmaRMS_vals)[1],'o', label = "$\sigma_{RMS}$")
ax2.plot(range(0,len(choppedSigmaHL_vals)), choppedSigmaHL_vals, '+', label = "$\sigma_{HL}$")
ax2.set_title("Polygonal $\\theta_{YZ}'$", fontsize = titleFontSize)
ax2.set_xlabel("Segment #")
ax2.set_ylabel("Angular Distribution Paras (rad)", fontsize = 'x-small')
ax2.set_ylim(bottom = 0)
ax2.legend()

ax3.plot(range(0,len(np.transpose(recoLinearSigmaRMS_vals)[0])), np.transpose(recoLinearSigmaRMS_vals)[0],'o', label = "$\sigma_{RMS}$")
ax3.plot(range(0,len(choppedSigmaHL_vals)), choppedSigmaHL_vals, '+', label = "$\sigma_{HL}$")
ax3.set_title("Linear $\\theta_{XZ}'$", fontsize = titleFontSize)
ax3.set_xlabel("Segment #")
ax3.set_ylabel("Angular Distribution Paras (rad)", fontsize = 'x-small')
ax3.set_ylim(bottom = 0)
ax3.legend()

ax4.plot(range(0,len(np.transpose(recoLinearSigmaRMS_vals)[1])), np.transpose(recoLinearSigmaRMS_vals)[1],'o', label = "$\sigma_{RMS}$")
ax4.plot(range(0,len(choppedSigmaHL_vals)), choppedSigmaHL_vals, '+', label = "$\sigma_{HL}$")
ax4.set_title("Linear $\\theta_{YZ}'$", fontsize = titleFontSize)
ax4.set_xlabel("Segment #")
ax4.set_ylabel("Angular Distribution Paras (rad)", fontsize = 'x-small')
ax4.set_ylim(bottom = 0)
ax4.legend()

fig.tight_layout(rect = (0, 0.03, 1, 0.95))
plt.savefig(samplePath + "recoAngularDistributionParametersVsSegmentNumber.png", bbox_inches = 'tight')

# Plot reeconstructed sigmaRES vs. Segment Number
fig = plt.figure()
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

fig.suptitle("Reconstructed $\sigma_{RES}$ vs. Segment Number")

ax1.plot(range(0,len(recoPolyXZ_sigmaRES_vals)), recoPolyXZ_sigmaRES_vals, 'o')
ax1.set_title("Polygonal $\\theta_{XZ}'$", fontsize = titleFontSize)
ax1.set_xlabel("Segment #")
ax1.set_ylabel("sigmaRES (rad)")
ax1.set_ylim(bottom = 0)

ax2.plot(range(0,len(recoPolyYZ_sigmaRES_vals)), recoPolyYZ_sigmaRES_vals, 'o')
ax2.set_title("Polygonal $\\theta_{YZ}'$", fontsize = titleFontSize)
ax2.set_xlabel("Segment #")
ax2.set_ylabel("sigmaRES (rad)")
ax2.set_ylim(bottom = 0)

ax3.plot(range(0,len(recoLinearXZ_sigmaRES_vals)), recoLinearXZ_sigmaRES_vals, 'o')
ax3.set_title("Linear $\\theta_{XZ}'$", fontsize = titleFontSize)
ax3.set_xlabel("Segment #")
ax3.set_ylabel("sigmaRES (rad)")
ax3.set_ylim(bottom = 0)

ax4.plot(range(0,len(recoLinearYZ_sigmaRES_vals)), recoLinearYZ_sigmaRES_vals, 'o')
ax4.set_title("Linear $\\theta_{YZ}'$", fontsize = titleFontSize)
ax4.set_xlabel("Segment #")
ax4.set_ylabel("sigmaRES (rad)")
ax4.set_ylim(bottom = 0)

fig.tight_layout(rect = (0, 0.03, 1, 0.95))
plt.savefig(samplePath + "recoSigmaRESVsSegmentNumber.png", bbox_inches = 'tight')

# Plot reconstructed average sigmaRES values
fig = plt.figure()
ax = plt.subplot()

fig.suptitle("Reconstructed Average $\sigma_{RES}$")

titles = ["Linear $\\theta_{XZ}'$", "Linear $\\theta_{YZ}'$", "Poly $\\theta_{XZ}'$", "Poly $\\theta_{YZ}'$"]
recoSigmaRES_averages = [np.nanmean(recoLinearXZ_sigmaRES_vals), np.nanmean(recoLinearYZ_sigmaRES_vals), np.nanmean(recoPolyXZ_sigmaRES_vals), np.nanmean(recoPolyYZ_sigmaRES_vals)]

ax.bar(titles,recoSigmaRES_averages)
ax.set_xlabel("MCS Method")
ax.set_ylabel("Average $\sigma_{RES}$ (rad)")

fig.tight_layout(rect = (0, 0.03, 1, 0.95))
plt.savefig(samplePath + "recoAverageSigmaRES.png", bbox_inches = 'tight')

# True Angle Plots:
# Plot True sigmaRMS & sigmaHL vs. Segment Number

fig = plt.figure()
ax1 = plt.subplot(221) # PolyXZ
ax2 = plt.subplot(222) # PolyYZ
ax3 = plt.subplot(223) # LinearXZ
ax4 = plt.subplot(224) # LinearYZ

fig.suptitle("True Angular Distribution Parameters vs. Segment Number")

ax1.plot(range(0,len(np.transpose(truePolygonalSigmaRMS_vals)[0])), np.transpose(truePolygonalSigmaRMS_vals)[0],'o', label = "$\sigma_{RMS}$")
ax1.plot(range(0,len(choppedSigmaHL_vals)), choppedSigmaHL_vals, '+', label = "$\sigma_{HL}$")
ax1.set_title("Polygonal $\\theta_{XZ}'$", fontsize = titleFontSize)
ax1.set_xlabel("Segment #")
ax1.set_ylabel("Angular Distribution Paras (rad)", fontsize = 'x-small')
ax1.set_ylim(bottom = 0)
ax1.legend()

ax2.plot(range(0,len(np.transpose(truePolygonalSigmaRMS_vals)[1])), np.transpose(truePolygonalSigmaRMS_vals)[1],'o', label = "$\sigma_{RMS}$")
ax2.plot(range(0,len(choppedSigmaHL_vals)), choppedSigmaHL_vals, '+', label = "$\sigma_{HL}$")
ax2.set_title("Polygonal $\\theta_{YZ}'$", fontsize = titleFontSize)
ax2.set_xlabel("Segment #")
ax2.set_ylabel("Angular Distribution Paras (rad)", fontsize = 'x-small')
ax2.set_ylim(bottom = 0)
ax2.legend()

ax3.plot(range(0,len(np.transpose(trueLinearSigmaRMS_vals)[0])), np.transpose(trueLinearSigmaRMS_vals)[0],'o', label = "$\sigma_{RMS}$")
ax3.plot(range(0,len(choppedSigmaHL_vals)), choppedSigmaHL_vals, '+', label = "$\sigma_{HL}$")
ax3.set_title("Linear $\\theta_{XZ}'$", fontsize = titleFontSize)
ax3.set_xlabel("Segment #")
ax3.set_ylabel("Angular Distribution Paras (rad)", fontsize = 'x-small')
ax3.set_ylim(bottom = 0)
ax3.legend()

ax4.plot(range(0,len(np.transpose(trueLinearSigmaRMS_vals)[1])), np.transpose(trueLinearSigmaRMS_vals)[1],'o', label = "$\sigma_{RMS}$")
ax4.plot(range(0,len(choppedSigmaHL_vals)), choppedSigmaHL_vals, '+', label = "$\sigma_{HL}$")
ax4.set_title("Linear $\\theta_{YZ}'$", fontsize = titleFontSize)
ax4.set_xlabel("Segment #")
ax4.set_ylabel("Angular Distribution Paras (rad)", fontsize = 'x-small')
ax4.set_ylim(bottom = 0)
ax4.legend()

fig.tight_layout(rect = (0, 0.03, 1, 0.95))
plt.savefig(samplePath + "trueAngularDistributionParametersVsSegmentNumber.png", bbox_inches = 'tight')

# Plot true sigmaRES vs. Segment Number
fig = plt.figure()
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

fig.suptitle("True $\sigma_{RES}$ vs. Segment Number")

ax1.plot(range(0,len(truePolyXZ_sigmaRES_vals)), truePolyXZ_sigmaRES_vals, 'o')
ax1.set_title("Polygonal $\\theta_{XZ}'$", fontsize = titleFontSize)
ax1.set_xlabel("Segment #")
ax1.set_ylabel("sigmaRES (rad)")
ax1.set_ylim(bottom = 0)

ax2.plot(range(0,len(truePolyYZ_sigmaRES_vals)), truePolyYZ_sigmaRES_vals, 'o')
ax2.set_title("Polygonal $\\theta_{YZ}'$", fontsize = titleFontSize)
ax2.set_xlabel("Segment #")
ax2.set_ylabel("sigmaRES (rad)")
ax2.set_ylim(bottom = 0)

ax3.plot(range(0,len(trueLinearXZ_sigmaRES_vals)), trueLinearXZ_sigmaRES_vals, 'o')
ax3.set_title("Linear $\\theta_{XZ}'$", fontsize = titleFontSize)
ax3.set_xlabel("Segment #")
ax3.set_ylabel("sigmaRES (rad)")
ax3.set_ylim(bottom = 0)

ax4.plot(range(0,len(trueLinearYZ_sigmaRES_vals)), trueLinearYZ_sigmaRES_vals, 'o')
ax4.set_title("Linear $\\theta_{YZ}'$", fontsize = titleFontSize)
ax4.set_xlabel("Segment #")
ax4.set_ylabel("sigmaRES (rad)")
ax4.set_ylim(bottom = 0)

fig.tight_layout(rect = (0, 0.03, 1, 0.95))
plt.savefig(samplePath + "trueSigmaRESVsSegmentNumber.png", bbox_inches = 'tight')

# Plot true average sigmaRES values
fig = plt.figure()
ax = plt.subplot()

fig.suptitle("True Average $\sigma_{RES}$")

titles = ["Linear $\\theta_{XZ}'$", "Linear $\\theta_{YZ}'$", "Poly $\\theta_{XZ}'$", "Poly $\\theta_{YZ}'$"]
trueSigmaRES_averages = [np.nanmean(trueLinearXZ_sigmaRES_vals), np.nanmean(trueLinearYZ_sigmaRES_vals), np.nanmean(truePolyXZ_sigmaRES_vals), np.nanmean(truePolyYZ_sigmaRES_vals)]

ax.bar(titles,trueSigmaRES_averages)
ax.set_xlabel("MCS Method")
ax.set_ylabel("Average $\sigma_{RES}$ (rad)")

fig.tight_layout(rect = (0, 0.03, 1, 0.95))
plt.savefig(samplePath + "trueAverageSigmaRES.png", bbox_inches = 'tight')

################################################################
# MARK: Section 5: Momentum Plotting
################################################################

# ==============================================================
# Reco MCS Momentum Plotting
# ==============================================================

# Collapse recoLinearMCSMomentumList down to a single array, from one that is separated by tracks
recoLinearMCSMomentum_vals = []
recoPolyMCSMomentum_vals = []
for eventIndex in range(0,len(recoLinearMCSMomentumList)):
	linearMCSMomentum_vals = recoLinearMCSMomentumList[eventIndex] # Array of track MCS momentum
	polyMCSMomentum_vals = recoPolyMCSMomentumList[eventIndex]
	recoLinearMCSMomentum_vals.extend(linearMCSMomentum_vals)
	recoPolyMCSMomentum_vals.extend(polyMCSMomentum_vals)

fig = plt.figure()
ax1 = plt.subplot(121) # Linear
ax2 = plt.subplot(122) # Polygonal

ax1.hist(recoLinearMCSMomentum_vals, bins = 100)
ax2.hist(recoPolyMCSMomentum_vals, bins = 100)
plt.savefig(samplePath + "recoMCSMomentum.png", bbox_inches = 'tight')

# ==============================================================
# True MCS Momentum Plotting
# ==============================================================

# Collapse trueLinearMCSMomentumList down to a single array, from one that is separated by tracks
trueLinearMCSMomentum_vals = []
truePolyMCSMomentum_vals = []
for eventIndex in range(0,len(trueLinearMCSMomentumList)):
	linearMCSMomentum_vals = trueLinearMCSMomentumList[eventIndex] # Array of track MCS momentum
	polyMCSMomentum_vals = truePolyMCSMomentumList[eventIndex]
	trueLinearMCSMomentum_vals.extend(linearMCSMomentum_vals)
	truePolyMCSMomentum_vals.extend(polyMCSMomentum_vals)

fig = plt.figure()
ax1 = plt.subplot(121) # Linear
ax2 = plt.subplot(122) # Polygonal

ax1.hist(trueLinearMCSMomentum_vals, bins = 100)
ax2.hist(truePolyMCSMomentum_vals, bins = 100)
plt.savefig(samplePath + "trueMCSMomentum.png", bbox_inches = 'tight')

# Still need to plot MCS Momentum vs. True Momentum (initial and maybe even vs. segment number, that could be interesting)

print()
print("knownAngleStudy.py has completed for sample:\n" + sampleDescription)

# TODO: Future check:
# Check if the same points in each segment for a segment with big diff

# MARK: Note to self:
"""
NOW:
Directly compare sigmaRMS, sigmaHL, and sigmaRES with ROOT values.
Plot MCS Momentum vs. True Momentum (are we getting the exact same values as ROOT?)
Calculate percent and fractional momentum bias & resolution

Soon:
Analyze how curvy tracks factor into MCS Momentum and sigmaRES calculations.
Analyze how virtual points may or may not help MCS Momentum and sigmaRES calculations.
Analyze how virtual points affect # of segment comparisons.
Analyze the effect of sigmaRES values on MCS Momentum calculations

Future:
Continually split this code into different files.
	Specifically put print statements in a separate file and plotting functions in a separate file
Propagate sigmaRES uncertainty.
Wrap data into classes to serve as containers
"""
