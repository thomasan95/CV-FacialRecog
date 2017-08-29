import os, sys
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
from scipy import spatial
import math

from util import load_subset, draw_faces

path = '/Users/thomasan/Documents/UCSD/CSE152/HW5/data/yaleBfaces/'
trainingSet, trainingLabels = load_subset([0])

dispTrainSet, dispTrainLabels = load_subset([0])

# Changing lists to numpy arrays
trainingSet = np.asarray(trainingSet)
trainingLabels = np.asarray(trainingLabels)

# Reshaping each image into a column vector
numTrain = trainingSet.shape[0]
imRow = trainingSet.shape[1]
imCol = trainingSet.shape[2]
trainingSet = np.reshape(trainingSet, (numTrain, imRow*imCol))
# print trainingSet.shape
errorPercentages = np.zeros((4,1))
accPercentages = np.zeros((4,1))

for i in range(4):
    counter = 0
    # Loading every data set
    dataSet, labelSet = load_subset([i+1])
    dispDataSet, dispLabelSet = load_subset([i+1])
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
    for m in range(dataSet.shape[0]):
        lowVal = float("inf")
        label = 0
        col = 0
        for n in range(trainingSet.shape[0]):

            diff = np.sum(np.square(np.subtract(dataSet[m,:],trainingSet[n,:])))
            distance = np.sqrt(diff)

            if distance < lowVal:
                lowVal = distance
                label = trainingLabels[n]
                col = n

        classifiedLabels[m] = label
        if label != labelSet[m]:
            print "Classification error", m, " as", label, labelSet[m]
    total = classifiedLabels.shape[0]
    errorPercentages[i,0] = np.sum(classifiedLabels != labelSet)/float(total)
    accPercentages[i,0] = np.sum(classifiedLabels == labelSet)/float(total)

print errorPercentages
print accPercentages
    

