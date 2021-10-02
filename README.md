This is an outdated repository. This work has been implemented in C++ in the LArSoft library.
You can view the latest work at: https://github.com/kasettisiva/MCSanalysis in protoduneana/protoduneana/singlephase/MCSanalysis.

# MCSMomentumEstimation
Work used in multiple Coulomb scattering momentum estimation analysis.

In plotSegments.py & plotSegmentAngles.py, you must specify an input file.  The input files are not located in this repository because they were too large.
If you need one, contact Hunter Meyer, hmeyer5@lsu.edu.

MCS.py:

old functions:
OrganizeEventData(fileName,eventsToDraw,drawAllEvents):
	Organizes event data, when the tracks have already been split into segments.
	returns: [[[[x1,x2,x3...],[y1,y2,y3,...],[z1,z2,z3,...]],segment2,segment3,...],event2,event3,...]

GetBarycenters(recoData,eventsToDraw,drawAllEvents):
	Takes recoData and computes the barycenters of each segment
	returns: [[[x,y,z],barycenter2,barycenter3,...],event2, event3,....]

GetLinearFitParameters(recoData,eventsToDraw,drawAllEvents):
	Takes recoData and computes the linear fit parameters of each segment
	returns: [[[A,B,C,D],segment2,segment3,...],event2,event3,...]

GetLinearFitEndPoints(linearFitParameters,barycenters,segmentLength,eventsToDraw,drawAllEvents):
	Takes data and calculates the two end points that would be required to calculate the end points of a 14-cm segment where it's center goes through the barycenter of this segment
	returns: [[[[x1,y1,z1],[x2,y2,z2]],linearSegment2,linearSegment3,...],event2, event3, ...]

new functions:
// May add a function to go ahead and separate the tracks into segments, basically form recoData in another way, but for now not going to waste the time.

InsertVirtualPointsToForceSegmentLength(recoData,segmentLength,eventsToDraw,drawAllEvents):
