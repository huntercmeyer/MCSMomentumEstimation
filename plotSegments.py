""" READ_ME
Written by Hunter Meyer

How to use this script: python3 plotSegments.py
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
	
This script will generate a figure with a 2x2 grid of images.  The two on the left will be screenshots (taken from a 3D event display).
	You can input the path to the screenshots with the screenshot1path and screenshot2path parameters
The two images on the right will be plots of the 2D projections of either a reconstructed track for a single event, denoted eventToDraw or all of the events, if drawAllEvents is True.
The 2D projections will have each segment color determined by the colors list.

The figure will be saved in a file named by the outputFileName parameter.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters you can change, see above READ_ME
markerSize = 0.1
drawAllEvents = False
eventsToDraw = [209]
colors = ['blue','red','green','yellow','orange','purple','magenta','lime','deeppink']
recoFileName = "normalRecoPosInfo.txt"
screenshot1path = "beeEventDisplayScreenshots/screenshot11.png"
screenshot2path = "beeEventDisplayScreenshots/screenshot12.png"
outputFileName = "eventDisplay.png"

# Draw reconstructed segments, labelled by color

recoEventNum,recoSegmentNum,recoPointNum,recoX,recoY,recoZ = np.loadtxt(recoFileName,unpack = True,skiprows=1)

# Convert to integers to be used as list index
recoEventNum = recoEventNum.astype(int)
recoSegmentNum = recoSegmentNum.astype(int)
recoPointNum = recoPointNum.astype(int)
 
# Get the maximums
recoMaxSeg = np.amax(recoSegmentNum)
maxEventNum = np.amax(recoEventNum)
# Events begin counting at 1 and go to maxEventNum.
# Segments begin counting at 0 and go to recoMaxSeg for that event. (for the events that contained the maximum, for the other events we will later chop off the empty data)

recoData = [[[[],[],[]] for segment in np.arange(0,recoMaxSeg+1)] for event in np.arange(0,maxEventNum)] # +1 comes from the fact that segments go from 0 to recoMaxSeg, meaning there are recoMaxSeg+1 total segments!
# Build the recoData list using the following structure: (easily read from right to left)
# [[[[x1,x2,x3...],[y1,y2,y3,...],[z1,z2,z3,...]],[segment2],[segment3],...],[event2],[event3],...]
# Recall that events start at 1 but segments start at 0

for entry in np.arange(0,len(recoEventNum)):	# Loop through every entry
	eventNum = recoEventNum[entry]
	if (eventNum - 1 in eventsToDraw) or drawAllEvents:					# -1 comes from events starting
		segmentNum = recoSegmentNum[entry]
		
		recoData[eventNum-1][segmentNum][0].append(recoX[entry])	# -1 comes from events starting to count at 1, not 0
		recoData[eventNum-1][segmentNum][1].append(recoY[entry])
		recoData[eventNum-1][segmentNum][2].append(recoZ[entry])
print(recoData[209])
# Remove the entries of recoData that are empty.
emptySegment = [[],[],[]]
for event in recoData:
	try:
		while True:
			event.remove(emptySegment)
	except ValueError:
		pass

fig = plt.figure()
ax1 = plt.subplot(221) # Top left
ax2 = plt.subplot(223) # Bottom left
ax3 = plt.subplot(222) # Top Right
ax4 = plt.subplot(224) # Bottom Right

ax1.imshow(mpimg.imread(screenshot1path))
ax1.axis('off')
ax2.imshow(mpimg.imread(screenshot2path))
ax2.axis('off')

ax1.set_title("Bee Event Display screenshots")

# X vs. Z Box
ax3.plot((-0.5,0.5,695,695,-0.5),(-350,350,350,-350,-350),":r")
ax3.set_title("X vs. Z")
ax3.set_xlabel("Z (cm)")
ax3.set_ylabel("X (cm)")

# Y vs. Z Box
ax4.plot((-0.5,0.5,695,695,-0.5),(0,608,608,0,0),":r")
ax4.set_title("Y vs. Z")
ax4.set_xlabel("Z (cm)")
ax4.set_ylabel("Y (cm)")

ax3.get_xaxis().set_label_coords(1,-0.1)
ax3.get_yaxis().set_label_coords(-0.15,1)

ax4.get_xaxis().set_label_coords(1,-0.1)
ax4.get_yaxis().set_label_coords(-0.15,1)

for eventNum in np.arange(0,len(recoData)):
	if (eventNum in eventsToDraw) or drawAllEvents:
		event = recoData[eventNum]
		for segmentNum in np.arange(0,len(event)):
			segment = event[segmentNum]
			x = segment[0]
			y = segment[1]
			z = segment[2]
			ax3.plot(z,x,'o',markerSize = markerSize,color = colors[segmentNum % len(colors)])
			ax4.plot(z,y,'o',markerSize = markerSize,color = colors[segmentNum % len(colors)])

plt.tight_layout()

#plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# Using this dpi for a very clear picture with lots of zoom available.
fig.dpi = 1200
plt.savefig(outputFileName,bbox_inches = 'tight')

# This is the default dpi
fig.dpi = 100
plt.show()
