import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from sklearn import preprocessing


mypath = "Train/"
from os import listdir
from os.path import isfile, join
files = [join(mypath,f) for f in listdir(mypath) if isfile(join(mypath, f))]


# for f in files:
# 	print f

vectors = []

for f in files :
	file1 =  open(f,'r').read().strip('\n').split('\n')
	print f
	for j in file1:
		vectors.append([float(x) for x in j.strip(',').split(',')])

print "done"

# for i in vectors :
# 	print i	
X = vectors
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
ms = KMeans(n_clusters=100)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_	
print labels
print cluster_centers
