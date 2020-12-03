import numpy as np
import newMCS as MCS
import matplotlib.pyplot as plt

# Parameters
truePosInfo = "500_mu_1_GeV_start_beam-entry_dir_35_-45_truePosInfo.txt"
recoPosInfo = "500_mu_1_GeV_start_beam-entry_dir_35_-45_recoPosInfo.txt"
maxEventNum = 500
eventList = [0,1] # List of events we are working with
ignoreEventList = True
titleFontSize = 8

# Sort trajectory points corresponding to each event into a List
# events = MCS.OrganizeTrueDataIntoEvents(truePosInfo)
recoEvents = MCS.OrganizeRecoDataIntoEvents(recoPosInfo)

# Remove all events that are not in eventList, unless ignoreEventList is True
if not ignoreEventList:
	tempEvents = []
	for eventNum in np.arange(0,len(recoEvents)):
		if eventNum in eventList:
			tempEvents.append(recoEvents[eventNum])
			
	recoEvents = tempEvents

# If you want to know the eventNum of events[eventIndex], use eventList[eventIndex]
# This code here allows this to be true, regardless of ignoreEventList
if ignoreEventList:
	eventList = [i for i in np.arange(0,maxEventNum)]

# Sort each track's data into segments, forcing the segment length to be exactly the same
# sortedRecoData Organizational structure:
# [[[[[x1,y1,z1],trajectoryPoint2,trajectoryPoint3,...],segment2,segment3,...],track2,track3,...],event2,event3,...]
sortedRecoData = []
for eventIndex in np.arange(0,len(recoEvents)):
	event = recoEvents[eventIndex]
	sortedRecoData.append([])
	for trackIndex in np.arange(0,len(event)):
		track = event[trackIndex]
		sortedData = MCS.OrganizeTrackDataIntoSegments(track,14.0,False)
		sortedRecoData[eventIndex].append(sortedData)

# Get Barycenters of each segment in each track in each event
# Does not get the barycenter of the final segment, which is not 14-cm
recoBarycentersList = []
for eventIndex in np.arange(0,len(sortedRecoData)):
	event = sortedRecoData[eventIndex]
	recoBarycentersList.append([])
	for trackIndex in np.arange(0,len(event)):
		track = event[trackIndex]
		recoBarycentersList[eventIndex].append(MCS.GetBarycenters(track))

# Get Linear Fit Parameters of each segment in each track in each event
# Does not get the fit parameters of the final segment, which is not 14-cm
recoLinearFitParametersList = []
for eventIndex in np.arange(0,len(sortedRecoData)):
	event = sortedRecoData[eventIndex]
	recoLinearFitParametersList.append([])
	for trackIndex in np.arange(0,len(event)):
		track = event[trackIndex]
		recoLinearFitParametersList[eventIndex].append(MCS.GetLinearFitParameters(track))

# Get Polygonal Angles from each segment in each track, using the barycenters
recoPolygonalAngles = []
for eventIndex in np.arange(0,len(sortedRecoData)):
	event = sortedRecoData[eventIndex]
	recoPolygonalAngles.append([])
	for trackIndex in np.arange(0,len(event)):
		barycenters = recoBarycentersList[eventIndex][trackIndex]
		recoPolygonalAngles[eventIndex].append(MCS.GetPolygonalAngles(track,barycenters))

# Get Linear Angles from each segment in each event, using the linear fit parameters
recoLinearAngles = []
for eventIndex in np.arange(0,len(sortedRecoData)):
	event = sortedRecoData[eventIndex]
	recoLinearAngles.append([])
	for trackIndex in np.arange(0,len(event)):
		parameters = recoLinearFitParametersList[eventIndex][trackIndex]
		recoLinearAngles[eventIndex].append(MCS.GetLinearAngles(event,parameters))

# Group data from separate tracks (across all events) into a format for sigmaRMS processing.
# Group XZ measurements into a List that contains a List corresponding to segments.  Each entry in that List is a List that contains the angle measurements from different tracks that all correspond to that segment.
# Do the same for YZ and return in a List so that they're together
groupedPolyAngleMeasurements = MCS.GroupAngleDataIntoSegments(recoPolygonalAngles)
groupedLinearAngleMeasurements = MCS.GroupAngleDataIntoSegments(recoLinearAngles)

# Plot angle measurements for each segment
# First need to make the arrays to plots
poly_thetaXZprime_vals = groupedPolyAngleMeasurements[0]
poly_thetaYZprime_vals = groupedPolyAngleMeasurements[1]
poly_xVals = []
yVals_polyThetaXZprime = []
yVals_polyThetaYZprime = []
for segmentNum in range(0,len(poly_thetaXZprime_vals)):
	poly_xVals.extend([segmentNum for _ in range(0,len(poly_thetaXZprime_vals[segmentNum]))])
	yVals_polyThetaXZprime.extend(1000*x for x in poly_thetaXZprime_vals[segmentNum])
	yVals_polyThetaYZprime.extend(1000*x for x in poly_thetaYZprime_vals[segmentNum])

linear_thetaXZprime_vals = groupedLinearAngleMeasurements[0]
linear_thetaYZprime_vals = groupedLinearAngleMeasurements[1]
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

# Recreate sigmaRMS, sigmaHL, and sigmaRES analysis plots
truePolygonalSigmaRMS_vals = MCS.GetSigmaRMS_vals(groupedPolyAngleMeasurements)
print(truePolygonalSigmaRMS_vals)

trueLinearSigmaRMS_vals = MCS.GetSigmaRMS_vals(groupedLinearAngleMeasurements)
print(trueLinearSigmaRMS_vals)

fig = plt.figure()
ax1 = plt.subplot(221) # PolyXZ
ax3 = plt.subplot(223) # PolyYZ
ax2 = plt.subplot(222) # LinearXZ
ax4 = plt.subplot(224) # LinearYZ

ax1.plot(range(0,len(np.transpose(truePolygonalSigmaRMS_vals)[0])), np.transpose(truePolygonalSigmaRMS_vals)[0],'o')
ax1.set_title("Poly ThetaXZprime sigmaRMS vs. Segment #", fontsize = titleFontSize)
ax1.set_xlabel("Segment #")
ax1.set_ylabel("sigmaRMS")

ax3.plot(range(0,len(np.transpose(truePolygonalSigmaRMS_vals)[1])), np.transpose(truePolygonalSigmaRMS_vals)[1],'o')
ax3.set_title("Poly ThetaYZprime sigmaRMS vs. Segment #", fontsize = titleFontSize)
ax3.set_xlabel("Segment #")
ax3.set_ylabel("sigmaRMS")

ax2.plot(range(0,len(np.transpose(trueLinearSigmaRMS_vals)[0])), np.transpose(trueLinearSigmaRMS_vals)[0],'o')
ax2.set_title("Linear ThetaXZprime sigmaRMS vs. Segment #", fontsize = titleFontSize)
ax2.set_xlabel("Segment #")
ax2.set_ylabel("sigmaRMS")

ax4.plot(range(0,len(np.transpose(trueLinearSigmaRMS_vals)[1])), np.transpose(trueLinearSigmaRMS_vals)[1],'o')
ax4.set_title("Linear ThetaYZprime sigmaRMS vs. Segment #", fontsize = titleFontSize)
ax4.set_xlabel("Segment #")
ax4.set_ylabel("sigmaRMS")

plt.tight_layout()
plt.savefig("sigmaRMS.png", bbox_inches = 'tight')

# First need to ask how to get momentum for segment due to virtual points

# Recreate final MCS Momentum Estimation Methods:
# Likelihoods
# Momentum Loss
"""
#"""

# Future check:
# Check if the same points in each segment for a segment with big diff
