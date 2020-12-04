import numpy as np
import newMCS as MCS
import matplotlib.pyplot as plt
import sys

# Parameters
truePosInfo = "500_mu_1_GeV_start_beam-entry_dir_35_-45_truePosInfo.txt"
recoPosInfo = "500_mu_1_GeV_start_beam-entry_dir_35_-45_recoPosInfo.txt"
maxEventNum = 500
eventList = [0,1] # List of events we are working with
ignoreEventList = True
titleFontSize = 8
forceRecoSegmentLength = False

################################################################
# Analyze Reco Tracks
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

# Modify eventList if ignoreEventList = True
# 	If you want to know the eventNum of events[eventIndex], use eventList[eventIndex]
#	This code here allows this to be true, regardless of ignoreEventList
if ignoreEventList:
	eventList = [i for i in np.arange(0,maxEventNum)]

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
# Reprocess recoPolygonalAnglesList and recoLinearAnglesList so that they're grouped by segment, rather than by track/event
# This reduces the size of the array to the form:
# [[[thetaXZprime_segment1_1,thetaXZprime_segment1_2,...],XZangleMeasurementsXZ2,XZangleMeasurements3,...],segmentedThetaYZprimeList]
# We lose track and event separation from this!
# ==============================================================

recoGroupedPolyAngleMeasurements = MCS.GroupAngleDataIntoSegments(recoPolygonalAnglesList)
recoGroupedLinearAngleMeasurements = MCS.GroupAngleDataIntoSegments(recoLinearAnglesList)

# Recreate sigmaRMS, sigmaHL, and sigmaRES analysis plots
recoPolygonalSigmaRMS_vals = MCS.GetSigmaRMS_vals(recoGroupedPolyAngleMeasurements)
recoLinearSigmaRMS_vals = MCS.GetSigmaRMS_vals(recoGroupedLinearAngleMeasurements)

################################################################
# Analyze True Tracks (MCParticle's)
################################################################

trueEvents, trueMomentumData = MCS.OrganizeTrueDataIntoEvents(truePosInfo)

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
