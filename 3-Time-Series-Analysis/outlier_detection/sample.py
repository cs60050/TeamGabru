from sklearn.datasets.samples_generator import make_blobs, make_classification
import numpy as np
import codecs
import pickle
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import random

class MeanShiftAnamoly():


	def __init__(self, W, thres):
		self.sample = []
		self.W = W
		self.s = 0
		self.thres = thres



	def reservoir_sample(self, new_point):
		if(len(self.sample) < self.W):
			self.sample.append(new_point)
			self.s += 1
			return
		self.s += 1
		j = random.randint(1,self.s)
		if(j < self.W):
			self.sample[j] = new_point


	def novelty_detector(self, newpoints):


		stored = np.asarray(self.sample)
		newpoints = np.asarray(newpoints)
		print(np.shape(stored))
		print(np.shape(newpoints))

		X = np.append(stored, newpoints, axis=0)

		

		bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=1000)
		ms = MeanShift(bandwidth=5*bandwidth, bin_seeding=True)
		ms.fit(X)

		labels = ms.labels_
		cluster_centers = ms.cluster_centers_

		labels_unique = np.unique(labels)
		n_clusters_ = len(labels_unique)

		print(n_clusters_)

		cluster_weights = np.bincount(labels)

		total_wieght = np.sum(cluster_weights)

		cluster_novelty = 1 - cluster_weights / total_wieght

		print(cluster_novelty)

		novelties = [cluster_novelty[c] for c in labels[:-len(newpoints)]]

		# # Plot result
		# import matplotlib.pyplot as plt
		# from itertools import cycle

		# plt.figure(1)
		# plt.clf()

		# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
		# for k, col in zip(range(n_clusters_), colors):
		#     my_members = labels == k
		#     cluster_center = cluster_centers[k]
		#     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
		#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
		#              markeredgecolor='k', markersize=14)
		# plt.title('Estimated number of clusters: %d' % n_clusters_)
		# plt.show()

		for pt in newpoints:
			self.reservoir_sample(pt)

		return novelties


	def online_detect(self, new_points):

		novelties = self.novelty_detector(new_points)

		return novelties

		anamoly = np.zeros(len(new_points))

		for i in range(len(new_points)):
			if(novelties[i] < self.thres):
				anamoly[i] = 0
			else:
				anamoly[i] = 1

		return anamoly, novelties



def test():
	detector =  MeanShiftAnamoly(600, 0.5)

	novelty = []

	print("Loading")
	trainfile = open('matrix_train.pkl', "rb")

	
	testfile = open('matrix_test_labelled.pkl',"rb")

	trainset = pickle.load(trainfile)
	print("Train loaded")
	testset = pickle.load(testfile)

	print('All loaded')

	print(np.shape(testset))
	exit(0)


	for frame in trainset[:600]:
		detector.reservoir_sample(frame)


	alll = np.append(trainset[:600], testset, axis=0)

	novelty_all = detector.online_detect(alll)

	plt.plot(novelty_all)
	plt.show()

	exit(0)


	for frame in trainset[600:]:
		novelty.append(detector.online_detect([frame])[1][0])

	for frame in testset:
		novelty.append(detector.online_detect([frame])[1][0])

	print(novelty[5:10])

	plt.plot(novelty)
	plt.show()




def test_on_synthetic_data():
	X, _ = make_classification(n_samples=10000,n_features=2, n_redundant=0, weights=[0.99,0.01])

	for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000]:



		labels, cluster_novelty = novelty_detector(X[:i], X[i:i+1000])
	# pt_novelty = cluster_novelty[labels]

	# import matplotlib.pyplot as plt

	# plt.plot(X, pt_novelty, 'ro')
	# plt.show()


test()

