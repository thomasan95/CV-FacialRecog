import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import spatial
from scipy.stats import mode

from util import load_subset

def eigenTrain(trainset, k):
    # Takes input N x d matrix from subset 0
    N = trainset.shape[0]
    d = trainset.shape[1]
    
    meanTrain = np.mean(trainset,0)
    centeredTrain = trainset - meanTrain
    
    # covTrain = np.cov(centeredTrain.T)
    
    U, S, V = np.linalg.svd(trainset.T)
    # Tossing out first 4 eigenvectors
    # U = U[:,4:]
    eigVectors = U[:,:k]
    
    W = eigVectors.T
    return W, meanTrain
    
def eigenTest(trainset, trainlabels, testset, W, mu, k):
    # print trainset.shape, trainlabels.shape, testset.shape, W.shape, mu, k
    Ytrain = np.dot(trainset, W.T)
    Ytest = np.dot(testset, W.T)

    classlabels = np.zeros(testset.shape[0])

    for m in range(Ytest.shape[0]):
        lowVal = float("inf")
        label = 0
        for n in range(Ytrain.shape[0]):
            diff = np.sum(np.square(np.subtract(Ytest[m,:],Ytrain[n,:])))
            distance = np.sqrt(diff)
            if distance < lowVal:
                lowVal = distance
                label = trainlabels[n]
        classlabels[m] = label
    return classlabels



trainSet, trainLabels = load_subset([0])
trainSet, trainLabels = np.asarray(trainSet), np.asarray(trainLabels)
trainSet = np.reshape(trainSet, (70, 2500))

###############################################################################
'''
Executing Part 1 of problem 3
'''
###############################################################################
# W, mu = eigenTrain(trainSet, 20)
# print W.shape, mu.shape
# eigImages = np.reshape(W, (20, 50, 50))
# print eigImages.shape
# images = np.zeros((500,100))
# counter = -1
# for i in range(10):
#     for j in range(2):
#         counter += 1
#         images[i*50:i*50+50,j*50:j*50+50] = eigImages[counter,:,:]
# plt.imshow(images, cmap='gray')
# plt.show()


###############################################################################
'''
Executing with 1 image from each person
'''
###############################################################################
# Splice subset to grab every 7th picture,
# corresponds to first pic of every person

# tenImages = trainSet[0::7,:]
# for k in range(1,11):
#     W, mean = eigenTrain(tenImages, k)
#     Y = np.dot(W,tenImages.T)
#     reProj = np.dot(Y.T, W) + mean
#     reProj = np.reshape(reProj, (10,50,50))
#     images = np.zeros((50,500))
#     counter = -1
#     for i in range(10):
#         counter += 1
#         images[:,i*50:i*50+50] = reProj[counter,:,:]
#
#     plt.figure(1)
#     plt.imshow(Y, cmap='gray')
#     plt.figure(2)
#     plt.imshow(images, cmap='gray')
#     plt.show()
#
#     # Use raw input to space out figures
#     raw_input("Press Enter to continue...")
#
#     plt.close('all')
#     print W.shape, tenImages.shape


###############################################################################
'''
Executing eigenTest
'''
###############################################################################
errorRates = np.zeros((4,20))
accRates = np.zeros((4,20))
for k in range(1,21):
    for i in range(4):
        # Loading testset with label and

        testSet, testLabel = load_subset([i+1])
        testSet, testLabel = np.asarray(testSet), np.asarray(testLabel)

        numImg = testSet.shape[0]
        nrow = testSet.shape[1]
        ncol = testSet.shape[2]

        testSet = np.reshape(testSet, (numImg,nrow*ncol))
        W, mu = eigenTrain(trainSet, k)
        classlabels = eigenTest(trainSet, trainLabels, testSet, W, mu, k)
        total = classlabels.shape[0]

        accPercent = np.sum(classlabels == testLabel)/float(total)
        errPercent = np.sum(classlabels != testLabel)/float(total)
        errorRates[i,k-1] = errPercent
        accRates[i,k-1] = accPercent

plt.plot(errorRates[0, :], label="subset 1")
plt.plot(errorRates[1, :], label="subset 2")
plt.plot(errorRates[2, :], label="subset 3")
plt.plot(errorRates[3, :], label="subset 4")

plt.legend(loc='upper right')
plt.xlabel('K = 1-20')
plt.xticks(np.arange(1,21))
plt.ylabel('Error Percentage')
plt.show()