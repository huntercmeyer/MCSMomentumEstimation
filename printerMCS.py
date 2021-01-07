""" READ_ME
This file will host a set of functions that will accept MCS data and print it
in a way that is easy to look at and compare with other data.

I will add documentation about what I expect from these print statements in relation to one another,
such as when comparing the lengths of certain lists that have to do with segments.
"""
# TODO: Add more documentation

import numpy as np

# Provide sigmaHL_vals of data structure:
# A 1D list.
def printSigmaHL_vals(sigmaHL_vals):
	print()
	print("sigmaHL_vals:")
	print("Length:",len(sigmaHL_vals))
	# Length expectation: One less than the number of true track segments. (doesn't include 14-cm)
		# sigmaHL is calculated for the first segment, but the sigmaHL list is usually passed to other functions using a chopped version of itself [1:].
	
	for index in range(0,len(sigmaHL_vals)):
		sigmaHL = round(sigmaHL_vals[index],4)
		print("i:",index,"\tsigmaHL:",sigmaHL)
	
	print()

def printSigmaRMS_vals(sigmaRMS_vals, description):
	print()
	print(description,"sigmaRMS_vals:")
	print("Length:",len(sigmaRMS_vals))
	
	for index in range(0,len(sigmaRMS_vals)):
		sigmaRMS_XZ = round(sigmaRMS_vals[index][0],4)
		sigmaRMS_YZ = round(sigmaRMS_vals[index][1],4)
		print("i:",index,"\tsigmaRMS_XZ:",sigmaRMS_XZ,"\tsigmaRMS_YZ:",sigmaRMS_YZ)
	
	print()

def printSigmaRES_vals(sigmaRES_vals, description):
	print()
	print(description,"sigmaRES_vals:")
	print("Length:",len(sigmaRES_vals))
	
	for index in range(0,len(sigmaRES_vals[0])):
		sigmaRES_XZ = round(sigmaRES_vals[0][index],4)
		sigmaRES_YZ = round(sigmaRES_vals[1][index],4)
		print("i:",index,"\tsigmaRES_XZ:",sigmaRES_XZ,"\tsigmaRES_YZ:",sigmaRES_YZ)
	
	print()

# listOfSigmaRES_vals_with_descriptions is a List of tuples, where the first value in that tuple
# is a list of sigmaRES_vals and the second value is a description of that list of sigmaRES_vals
def printAverageSigmaRES_vals(listOfSigmaRES_vals_with_descriptions):
	print()
	print("Average sigmaRES_vals:")
	
	for index in range(0,len(listOfSigmaRES_vals_with_descriptions)):
		sigmaRES_vals = listOfSigmaRES_vals_with_descriptions[index][0]
		description = listOfSigmaRES_vals_with_descriptions[index][1]
		
		sigmaRES_average = round(np.nanmean(sigmaRES_vals),4)
		nanCount = len([x for x in sigmaRES_vals if np.isnan(x)])
		print(description + ":\t",sigmaRES_average,"\tnan count:",nanCount,"out of",len(sigmaRES_vals))

	print()
