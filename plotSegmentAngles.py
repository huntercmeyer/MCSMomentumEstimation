""" READ_ME
Written by Hunter Meyer, hmeyer5@lsu.edu

How to use this script: python3 plotSegmentAngles.py
Parameters you can change:
	markerSize (Float)
	drawAllEvents (Bool)
	eventToDraw (Int)
	colors (List of Color Strings)
	recoFileName (String)
	screenshot1path (String)
	screenshot2path (String)
	outputFileName (String)

recoFileName must have to following contents:
recoEventNum, recoSegmentNum, recoPointNum, recoX, recoY, and recoZ
	Each entry is a trajectory point that describes has an X,Y,Z position, as well as some identifying
	information to separate the trajectory points based on event and segment.  recoPointNum is not used.
	
This script will convert the 1D arrays gotten from np.loadtxt(recoFileName, unpack = True, skiprows = 1) to this data structure:
recoData = [[[[x1,x2,x3...],[y1,y2,y3,...],[z1,z2,z3,...]],[segment2],[segment3],...],[event2],[event3],...]
	This is a 4D list, with entries of events, that contain entries of segments, that each have X,Y,Z lists that contain the
	X,Y, and Z data for all trajectory points corresponding to that segment in that event.
	
	If you would instead like the list of trajectory points in a segment in an event, you must transpose recoData[eventNum][segmentNum].

The barycenters are stored in this data structure:
barycenters = [[[[x1,x2,x3...],[y1,y2,y3,...],[z1,z2,z3,...]],[segment2],[segment3],...],[event2],[event3],...]
The polygonal segments are the lines connecting these barycenters

The Linear Segments are stored in this data structure:
linearFitParameters = [[[A,B,C,D],segment2,segment3,...],event2,event3,...] where z = A*x + B and y = C*z + D
linearFitEndPoints = [[[[x1,y1,z1],[x2,y2,z2]],linearSegment2,linearSegment3,...],event2, event3, ...]
						 \___VV___/ \___VV___/
						   Point 1    Point 2
The linear segment is the line connecting these two points.
There are N segments, N-1 14 cm segments, N-1 barycenters, and N-1 linear fits
There are N+1 segment start points (because the last segment start point is actually the last trajectory point in the detector (it's not the start of a segment but it's stored with the trajectory points that are), but that is out of the scope of this script.

At the end of this script the polygonal and linear segments are plotted.
Please forward any questions or concerns to Hunter Meyer, hmeyer5@lsu.edu, and let me know if any comments or documentation are unclear.
"""

import numpy as np
import math as math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import MCS

# Parameters you can change, see above READ_ME
markerSize = 0.7
drawAllEvents = False
eventsToDraw = [209] # 209 is in the main figure, events start counting at 0 for this (unlike recoEventNum data)
colors = ['blue','red','green','yellow','orange','purple','magenta','lime','deeppink']
blackOrGray = ['black','gray']
inputFileName = "normalRecoPosInfoLastSegmentIncluded.txt"
screenshot1path = "beeEventDisplayScreenshots/screenshot11.png"
screenshot2path = "beeEventDisplayScreenshots/screenshot12.png"
outputFileName = "segmentAngles.png"

# Reading the data from inputFileName, we will have data that has no organization, but we can organize it into a readable data structure that better suits our situation.

# Organize Event Data into a more readable data structure (4D list)
recoData = MCS.OrganizeEventData(inputFileName,eventsToDraw,drawAllEvents)
# Structure:
# recoData = [[[[x1,x2,x3...],[y1,y2,y3,...],[z1,z2,z3,...]],segment2,segment3,...],event2,event3,...]
# You can access trajectory point trajectoryPointNum from event eventNum and segment segmentNum using:
# trajectoryPoint = np.transpose(recoData[eventNum][segmentNum])[trajectoryPointNum]
# trajectoryPoint will look like [x_i,y_i,z_i] where i indicates the trajectory point number.

# Calculate the barycenter of each segment and store into an organized 3D list
barycenters = MCS.GetBarycenters(recoData,eventsToDraw,drawAllEvents)
# Structure:
# barycenters = [[[x,y,z],barycenter2,barycenter3,...],event2, event3,....]
# barycenter = barycenters[eventNum][segmentNum] will return the [x,y,z] of this barycenter

############################################################
# Section 4 - Get Linear Fits for Linear Segments
############################################################

# What I am about to do:
# Using recoData, get the linear fits of each segment (we will get a list that has 4 entries corresponding to the fit parameters in the 2 projections)
# Using the linear fits and the barycenters, we will calculate 2 points for each segment that are equally distanced from the barycenter (centroid) correspond to the linear fits.
# We will end up with this data structure:
# [[[[x1,y1,z1],[x2,y2,z2]],linearSegment2,linearSegment3,...],event2, event3, ...]
#    \___VV___/ \___VV___/
#      Point 1    Point 2
# The distance between Points 1 and 2 will be controlled by a parameter, so that we can easily visualize the angles.

linearFitParameters = MCS.GetLinearFitParameters(recoData,eventsToDraw,drawAllEvents)
linearFitEndPoints = MCS.GetLinearFitEndPoints(linearFitParameters,barycenters,eventsToDraw,drawAllEvents)

############################################################
# Section 5 - Plotting
############################################################

fig = plt.figure()
# Top Left - X vs. Z Polygonal Segments
ax1 = plt.subplot(221)
ax1.set_title("Polygonal Segments X vs. Z")
ax1.set_xlabel("Z (cm)")
ax1.set_ylabel("X (cm)")

# Bottom Left - Y vs. Z Polygonal Segments
ax2 = plt.subplot(223)
ax2.set_title("Polygonal Segments Y vs. Z")
ax2.set_xlabel("Z (cm)")
ax2.set_ylabel("Y (cm)")

# Top Right - X vs. Z Linear Segments
ax3 = plt.subplot(222)
ax3.set_title("Linear Segments X vs. Z")
ax3.set_xlabel("Z (cm)")
ax3.set_ylabel("X (cm)")

# Bottom Right - Y vs. Z Linear Segments
ax4 = plt.subplot(224)
ax4.set_title("Linear Segments Y vs. Z")
ax4.set_xlabel("Z (cm)")
ax4.set_ylabel("Y (cm)")

for eventNum in np.arange(0,len(recoData)):

	if (eventNum in eventsToDraw) or drawAllEvents:
		# List of barycenters for this event
		xBarycenters = np.transpose(barycenters[eventNum]).tolist()[0]
		yBarycenters = np.transpose(barycenters[eventNum]).tolist()[1]
		zBarycenters = np.transpose(barycenters[eventNum]).tolist()[2]
	
		for segmentNum in np.arange(0,len(recoData[eventNum])):
			# Raw trajectory points
			segment = recoData[eventNum][segmentNum]
			x = segment[0]
			y = segment[1]
			z = segment[2]
			
			# Plot segments
			# Background of polygonal segments:
			ax1.plot(z,x,'o',markerSize = 5*markerSize, color = colors[segmentNum % len(colors)])
			ax2.plot(z,y,'o',markerSize = 5*markerSize, color = colors[segmentNum % len(colors)])
			# Background of linear segments:
			ax3.plot(z,x,'o',markerSize = 5*markerSize, color = colors[segmentNum % len(colors)])
			ax4.plot(z,y,'o',markerSize = 5*markerSize, color = colors[segmentNum % len(colors)])

			# If not the last segment (non-14 cm)
			if segmentNum != len(recoData[eventNum])-1:
				# Barycenter for this segment
				xBarycenter = xBarycenters[segmentNum]
				yBarycenter = yBarycenters[segmentNum]
				zBarycenter = zBarycenters[segmentNum]
				# Plot Barycenters
				ax1.plot(zBarycenter,xBarycenter,'o',markerSize = 10*markerSize, color = 'k')
				ax2.plot(zBarycenter,yBarycenter,'o',markerSize = 10*markerSize, color = 'k')
				# Plot Linear Segments (connected linearFitEndPoints) using list of end points
				xLinearFitEndPoints = np.transpose(linearFitEndPoints[eventNum][segmentNum]).tolist()[0]
				yLinearFitEndPoints = np.transpose(linearFitEndPoints[eventNum][segmentNum]).tolist()[1]
				zLinearFitEndPoints = np.transpose(linearFitEndPoints[eventNum][segmentNum]).tolist()[2]
				
				ax3.plot(zLinearFitEndPoints,xLinearFitEndPoints,markerSize = markerSize/2,color = blackOrGray[segmentNum % 2])
				ax4.plot(zLinearFitEndPoints,yLinearFitEndPoints,markerSize = markerSize/2,color = blackOrGray[segmentNum % 2])

		# Plot Polygonal Segments (connected barycenters) using list of barycenters
		ax1.plot(zBarycenters,xBarycenters,markerSize = markerSize/2, color = 'gray')
		ax2.plot(zBarycenters,yBarycenters,markerSize = markerSize/2, color = 'gray')
		
plt.tight_layout()

# Using this dpi for a very clear picture with lots of zoom available.
fig.dpi = 1200
plt.savefig(outputFileName,bbox_inches = 'tight')

# This is the default dpi
fig.dpi = 100
plt.show()
