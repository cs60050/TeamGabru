import pickle
import numpy as np


def load_dataset():
	print("Loading")
	trainfile = open('matrix_train.pkl', "rb")

	
	testfile = open('matrix_test_labelled.pkl',"rb")

	trainset = pickle.load(trainfile)
	print("Train loaded")
	testset = pickle.load(testfile)

	print('All loaded')

	print(testset[0])

	testset, y_test = testset[:,0], testset[:,1]

	print(np.shape(testset), np.shape(y_test))

	trainset, y_train = trainset[:,0], trainset[:,1]


	print(np.shape(trainset))

	X = trainset.tolist()
	for t in testset:
		X.append(t)

	X = np.asarray(X)

	y = np.append(y_train, y_test)

	print(y[10:100])

	print(np.bincount(y))

	print(np.shape(X), np.shape(y))


load_dataset()