import codecs
import pickle
import os
import numpy as np

basepath = "UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/"


destpath = "matrix_train.pkl"

def main():
	folders = os.listdir(basepath)
	folders = sorted(folders)
	matrix = []

	# tfile = open('gt_test.txt', "r")
	# ytest = []
	# for line in tfile:
	# 	ranges = line.strip().split(",")
	# 	rs = np.zeros(200)
	# 	for r in ranges:
	# 		s, e = r.strip().split(":")
	# 		s,e  = int(s), int(e)
	# 		rs[s:e+1] = 1
	# 	ytest.append(rs)


	for foldername in folders:
		files = os.listdir(basepath+foldername)
		files = sorted(files)

		number = int(foldername[-3:])
		#ranges = ytest[number-1]

		for filename in files:
			file = open(basepath+foldername+'/'+filename,'r')


			s = []
			for row in file:
				s.append(np.float(float(row.strip())))
			s = np.array(s)
			framenumber = int(filename[-7:-4])

			#label = ranges[framenumber-1]
			label = 0

			matrix.append((s,label))
			print(foldername, filename)



	destfile = open(destpath,'wb')

	matrix = np.array(matrix)
	pickle.dump(matrix, destfile)

	# print(matrix, file = destfile)





if __name__ == "__main__":main()
