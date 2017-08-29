import numpy as np
import matplotlib.pyplot as plt


from util import load_subset
from EigenFaces import eigenTrain, eigenTest

def fisherTrain(trainset, trainlabels, c):
    trainset = np.asarray(trainset)
    trainlabels = np.asarray(trainlabels)
    meanTrain = np.mean(trainset, 0)

    [N, d] = trainset.shape

    # Computer Wpca for the first N-c eigenvectors
    [W_pca, mu_pca] = eigenTrain(trainset, N-c)
    trainset_centered_pca = trainset - mu_pca

    proj_pca = np.dot(trainset_centered_pca, W_pca.T)

    # Grab the number of classes or value of unique
    # labels
    num_classes = np.unique(trainlabels)
    [N2, d2] = proj_pca.shape
    Sw = np.zeros((d2, d2))
    Sb = np.zeros((d2, d2))
    mean_proj = np.mean(proj_pca, 0)
    # Iteratively find Sw and Sb going through classes
    for i in num_classes:
        # grab means of trainset and class

        train_class_i = proj_pca[np.where(trainlabels == i)[0], :]
        num_sample = train_class_i.shape[0]

        mean_class = np.mean(train_class_i, 0)

        # Calculate Sw and Sb from their respective equations
        Sb = Sb + num_sample * np.dot((mean_class - mean_proj).T, (mean_class - mean_proj))
        Sw = Sw + np.dot((train_class_i-mean_class).T, (train_class_i-mean_class))

    # Grab eigvectors and values linalg because might not be able to use svm
    eigvalues, eigvectors = np.linalg.eig(np.linalg.inv(Sw)*Sb)
    # eigval and vectors shapes are 60x60, and 60x1
    pos = np.argsort(-eigvalues.real)
    pos = pos[0:c-1]
    # # shapes are 60x60 and 60x1
    eigvectors = eigvectors[:, pos]

    W_fld = np.copy(eigvectors[:, 0:c].real).T

    W_fisher = np.dot(W_fld, W_pca)
    return W_fisher, meanTrain

# Load up training set and labels
trainSet, trainLabels = load_subset([0])
trainSet, trainLabels = np.asarray(trainSet), np.asarray(trainLabels)
trainSet = np.reshape(trainSet, (70, 2500))

# Initialize c to 10
c = 10
Wfisher, mu_fisher = fisherTrain(trainSet, trainLabels, c)
errorRates = np.zeros((4,9))
accRates = np.zeros((4,9))

for k in range(1,10):
    print k
    for i in range(4):
        # Loading testset with label and
        testSet, testLabel = load_subset([i+1])
        testSet, testLabel = np.asarray(testSet), np.asarray(testLabel)

        numImg = testSet.shape[0]
        nrow = testSet.shape[1]
        ncol = testSet.shape[2]

        testSet = np.reshape(testSet, (numImg,nrow*ncol))
        W, mu = fisherTrain(trainSet, trainLabels, k)

        classlabels = eigenTest(trainSet, trainLabels, testSet, W, mu, k)
        total = classlabels.shape[0]

        accPercent = np.sum(classlabels == testLabel)/float(total)
        errPercent = np.sum(classlabels != testLabel)/float(total)
        errorRates[i, k-1] = errPercent
        accRates[i, k-1] = accPercent

plt.plot(errorRates[0, :], label="subset 1")
plt.plot(errorRates[1, :], label="subset 2")
plt.plot(errorRates[2, :], label="subset 3")
plt.plot(errorRates[3, :], label="subset 4")


plt.legend(loc='upper right')
plt.xlabel('K Values 1-9')
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.ylabel('Error Percentage')
plt.show()



# fisher_images = np.reshape(Wfisher, (9, 50, 50))
# images = np.zeros((450,50))
# counter = -1
# for i in range(9):
#     counter += 1
#     images[i*50:i*50+50, :] = fisher_images[counter, :, :]
# plt.imshow(images, cmap='gray')
# plt.show()