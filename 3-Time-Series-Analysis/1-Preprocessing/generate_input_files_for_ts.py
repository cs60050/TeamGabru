from __future__ import print_function
import pickle
import os
import datetime

import numpy as np
from sklearn.decomposition import PCA

test_path = "/home/other/13EC10063/Anomaly/features/feature_matrix/"
destpath = "/home/other/13EC10063/Anomaly/Input_TS/"

timestamp = "/home/other/13EC10063/Anomaly/features/timestamp/"

def main():

	files = os.listdir(test_path)

	for filename in files:

		test_file = open(test_path+filename,'rb')

		matrix = pickle.load(test_file)


		pca = PCA(n_components=300)
		print(len(matrix[0]))
		pca.fit(matrix)

		red_matrix = pca.transform(matrix)

		red_matrix = red_matrix.tolist()
		
		print(len(red_matrix[0]))

		str_arr = []

		if not os.path.exists(timestamp+filename):
			os.makedirs(timestamp+filename)


		timestamp_file = open(timestamp+filename+"/data.txt",'w')
		a = datetime.datetime(2016,1,1,11,34,00)
		for i,elem in enumerate(red_matrix):
			a = a + datetime.timedelta(0,60)
			
			print(a.strftime("%Y-%m-%d %H:%M:%S"), file = timestamp_file)
			elem_norm = elem
			
			for ii,in_elem in enumerate(elem_norm):
				# s = ""

				s = '"'+str(i)+'",'+a.strftime("%Y-%m-%d %H:%M:%S")+','+str(in_elem)

				try:
					str_arr[ii] = str_arr[ii]+'\n'+s
				except:
					str_arr.append("")
					str_arr[ii] = str_arr[ii]+'\n'+s
		if not os.path.exists(destpath+filename):
			os.makedirs(destpath+filename)

		for idx,elem in enumerate(str_arr):
			destfile = open(destpath+filename+'/'+str(idx)+'.csv','w')
			print('"","timestamp","count"',end='', file=destfile)
			print(elem, file = destfile)

		print(filename)
		# print(a.strftime("%Y-%m-%d %H:%M:%S")+','+str(i))

if __name__ == "__main__":main()