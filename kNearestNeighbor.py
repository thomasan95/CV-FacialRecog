import os, sys
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
from scipy import spatial
from scipy import stats

from util import load_subset

path = '/Users/thomasan/Documents/UCSD/CSE152/HW5/data/yaleBfaces/'
trainingSet, trainingLabels = load_subset([0])

# Changing lists to numpy arrays
trainingSet = np.asarray(trainingSet)
trainingLabels = np.asarray(trainingLabels)

# Reshaping each image into a column vector
numTrain = trainingSet.shape[0]
imRow = trainingSet.shape[1]
imCol = trainingSet.shape[2]
trainingSet = np.reshape(trainingSet, (numTrain, imRow*imCol))
oneNeighborErr = np.zeros((4,1))
oneNeighborAcc = np.zeros((4,1))

threeNeighborErr = np.zeros((4,1))
threeNeighborAcc = np.zeros((4,1))

fiveNeighborErr = np.zeros((4,1))
fiveNeighborAcc = np.zeros((4,1))

# errorPercentages = np.zeros((3,5,1))
# accPercentages = np.zeros((3,5,1))

Neighbors = np.array([1, 3, 5])
for i in range(4):

    counter = -1
    
    for k in Neighbors:
        counter = counter + 1
        # Loading every data set
        dataSet, labelSet = load_subset([i+1])
        
        # Switching classify image sets into numpy arrays
        dataSet = np.asarray(dataSet)
        labelSet = np.asarray(labelSet)
    
        # Changing each image into a column vector
        numImg = dataSet.shape[0]
        trRow = dataSet.shape[1]
        trCol = dataSet.shape[2]
        dataSet = np.reshape(dataSet, (numImg,trRow*trCol))
        classifiedLabels = np.zeros(labelSet.shape)
        
        numErr, total = 0, 0
        # Create arrays to store distances and the sorted Indexes 
        distances = np.zeros((dataSet.shape[0], trainingSet.shape[0]))
        sortedIndex = np.zeros((dataSet.shape[0], trainingSet.shape[0]))
        
        for m in range(dataSet.shape[0]):
            lowVal = float("inf")
            label = 0
            for n in range(trainingSet.shape[0]):
                # l1 norm
                abs_diff = np.sum(np.absolute(np.subtract(dataSet[m,:],trainingSet[n,:])))
                distance = abs_diff
                
                # l2 norm
                # diff = np.sum(np.square(np.subtract(dataSet[m,:],trainingSet[n,:])))
                # distance = np.sqrt(diff)
                distances[m,n] = distance

            # Grab indexes that would sort array then grab closest k neighbors    
            sortedIndex[m,:] = np.argsort(distances[m,:])
            kNeighbors = sortedIndex[m,0:k]
            labels = np.zeros(kNeighbors.shape)

            for a in range(labels.shape[0]):
            	index = int(kNeighbors[a])
            	labels[a] = trainingLabels[index]
            if k > 1:
            	modeAndCount = stats.mode(labels)
            	mode = modeAndCount[0]
            	counts = modeAndCount[1]
            	if counts[0] == 1:
            		continue
            	else:
            		l = int(mode[0])
            		classifiedLabels[m] = l
            		total = total + 1
            		if l != labelSet[m]:
            			numErr = numErr + 1
            else:
            	classifiedLabels[m] = labels[0]
            	total = total + 1
            	if labels[0] != labelSet[m]:
            		numErr = numErr + 1
        # print i, counter
        # errorPercentages[i,0] = numErr/float(total)
        # accPercentages[i,0] = 1-errorPercentages[i,0]
        # print errorPercentages[i,:]
        if k == 1:
        	oneNeighborErr[i,0] = numErr/float(total)
        	oneNeighborAcc[i,0] = 1 - oneNeighborErr[i,0]
        elif k == 3:
        	threeNeighborErr[i,0] = numErr/float(total)
        	threeNeighborAcc[i,0] = 1 - threeNeighborErr[i,0]
       	elif k == 5:
       		fiveNeighborErr[i,0] = numErr/float(total)
       		fiveNeighborAcc[i,0] = 1 - fiveNeighborErr[i,0]

print oneNeighborErr
print threeNeighborErr
print fiveNeighborErr

# print oneNeighborAcc
# print threeNeighborAcc
# print fiveNeighborAcc
