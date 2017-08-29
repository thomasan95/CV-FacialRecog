NearestNeighbor.py
This file calculated the nearet neighbors for the Yale faces dataset. The file then
displays the error rates and accuracy rates to terminal.

kNearestNeighbor.py
This file calculates the nearest neighbors and error rates/accuracy rates for 
neighbors k=1,3,5. It displays the rates for all of them to terminal.

EigenFaces.py
def eigenTrain(trainset, k)
	This function takes in the train set and 'k' which is the number of
	eigenvalues to consider. The function performs PCA on this and 
	through svd and then returns the W transformation matrix along
	with the data set mean.
	return W, mu
def eigenTest(trainset, trainlabels, testset, W, mu, k)
	This function takes in the trainset and train labels, test set, W
	mu and k. The function proojects the testset onto the different
	dimension space received from W from eigenTrain. It then performs 
	the nearest neighbor classifier in this space and returns the class labels 
	stored in a matrix.
	return classlabels[]

FisherFaces.py
def fisherTrain(trainset, trainlabels, c)
	This function takes in the trainset and 'c' value to specify the number
	of fisher vectors to compute. This function computes the fisher vectors
	by first calculating the cross class and in class variances and then computing
	the eigenvectors form those. Then we return the transformation matrix which is
	c-1 dimensions.
	return W, mu