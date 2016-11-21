from __future__ import print_function

import codecs
import pickle
import os
import numpy as np
from random import shuffle

basepath_Test = "/home/other/13EC10063/Anomaly/features/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/"
# basepath_Train = "/home/other/13EC10063/Anomaly/features/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/"

# destpath_train = "/home/other/13EC10063/Anomaly/features/matrix_train_2.pkl"
destpath_test = "/home/other/13EC10063/Anomaly/features/feature_matrix/"

destpath_map = "/home/other/13EC10063/Anomaly/features/feature_map/"

def main():

	folders = os.listdir(basepath_Test)
	folders.sort()
	for foldername in folders:

		files = os.listdir(basepath_Test+foldername)

		files.sort()

		destfile_test = open(destpath_test+foldername,'wb')

		destfile_map = open(destpath_map+foldername, 'wb')

		matrix_test = []
		for idx,filename in enumerate(files):

			print(str(idx)+'\t'+foldername+'\t'+filename, file = destfile_map)

			file = codecs.open(basepath_Test+foldername+'/'+filename,'r','utf-8')


			s = []
			for row in file:
				s.append(np.float(float(row.strip())))
			s = np.array(s)
			matrix_test.append(s)
			print(len(s))
			print(foldername, filename)
	
		matrix_test = np.array(matrix_test)
		pickle.dump(matrix_test, destfile_test)



	# print(matrix[0][1])

	# shuffle(matrix)

	# print(matrix[0][1])

	# print(len(matrix))
	# print(matrix, file = destfile)





if __name__ == "__main__":main()
