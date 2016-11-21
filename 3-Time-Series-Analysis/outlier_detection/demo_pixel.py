import cv2
import pickle
import numpy as np
import os
import glob

testfolders = [24,32]

basepath = "/home/prabhat/Documents/Anomaly/dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test"




def process_atom(bins, magnitude, fmask, tag_image, out, image_out, clf, atom_shape=[10,10,5]):

	bin_count = np.zeros(9, np.uint8)
	h,w, t = bins.shape
	for i in range(0,h,atom_shape[0]):
		for j in range(0, w, atom_shape[1]):
			i_end = min(h, i+10)
			j_end = min(w, j+10)

			# Get the atom for bins
			atom_bins = bins[i:i_end, j:j_end].flatten()

			# Average magnitude
			atom_mag = magnitude[i:i_end, j:j_end].flatten().mean()
			atom_fmask = fmask[i:i_end, j:j_end].flatten()
			
			# Count of foreground values
			f_cnt = np.count_nonzero(atom_fmask)

			# Get the direction bins values
			hs, _ = np.histogram(atom_bins, np.arange(10))
			
			# get the tag atom
			#tag_atom = tag_image[i:i_end, j:j_end].flatten() 
			#print(tag_atom)
			#ones = np.count_nonzero(tag_atom)
			# zeroes = len(tag_atom) - ones
			# tag = 1
			# if(ones < 10):
			# 	tag = 0
			features = hs.tolist()
			features.extend([f_cnt, atom_mag])
			y = clf.predict([features])[0]
			if(y == 1):
				image_out[i:i_end, j:j_end] = (0,0,255)
				
	return 0


def getFeaturesFromVideo(imagelist, taglist, out, clf, mag_threshold=1e-3, atom_shape=[10,10,5]):

	
	
	
	# first frame in grayscale
	prevgray = cv2.imread(imagelist.__next__(), cv2.IMREAD_GRAYSCALE)
	taglist.__next__()

	
	(rH, rW) = prevgray.shape[:2]
	blank_image = np.zeros((rH,rW,3), np.uint8)






	## Background extractor
	fgbg = cv2.createBackgroundSubtractorMOG2()

	h, w = prevgray.shape[:2]

	bins = np.zeros((h, w, atom_shape[2]), np.uint8)
	mag = np.zeros((h, w, atom_shape[2]), np.float32)
	fmask = np.zeros((h,w,atom_shape[2]), np.uint8)
	tag_img = np.zeros((h,w,atom_shape[2]), np.uint8)

	time = 0

	# Go through all the frames of the video
	while True:
		#Read next frame
		try:
			img = cv2.imread(imagelist.__next__(), cv2.IMREAD_GRAYSCALE)
		except:
			print("done")
			break
		#img = cv.imread(imagelist.__next__(), cv2.IMREAD_GRAYSCALE)

		# Read Tagged image
		tag_img_ = cv2.imread(taglist.__next__(), cv2.IMREAD_GRAYSCALE)

		tag_img[...,time] = tag_img_


		# cv2.imshow('image',img)
		# cv2.imshow('tag', tag_img_)
		# cv2.waitKey(5)

		# Get foreground/background
		fmask[...,time] = fgbg.apply(img)


		#Convert to grayscale
		gray = img
		# Calculate Optical Flow for all pixels in the image
		# Parameters : 
		# 		prevgray = prev frame
		#		gray     = current frame
		#       levels, winsize, iterations, poly_n, poly_sigma, flag
		# 		0.5 - image pyramid or simple image scale
		#		3 - no of pyramid levels
		# 		15 - window size
		#		3 - no of iterations
		#		5 - Polynomial degree epansion
		#		1.2 - standard deviation to smooth used derivatives
		# 		0 - flag
		flow = cv2.calcOpticalFlowFarneback(prevgray, gray,None, 0.5, 3, 15, 3, 5, 1.2, 0)
		## Flow contains vx, vy for each pixel

		
		# Calculate direction and magnitude
		height, width = flow.shape[:2]
		fx, fy = flow[:,:,0], flow[:,:,1]


		# Calculate direction qunatized into 8 directions
		angle = ((np.arctan2(fy, fx+1) + 2*np.pi)*180)% 360
		binno = np.ceil(angle/45)

		# Calculate magnitude
		magnitude = np.sqrt(fx*fx+fy*fy)

		# Add to zero bin if magnitude below a certain threshold
		#if(magnitude < mag_threshold):
		binno[magnitude < mag_threshold] = 0

		bins[...,time] = binno
		mag[..., time] = magnitude
		time = time + 1

		if(time == 5):
			time = 0
			process_atom(bins, mag, fmask, tag_img, out, blank_image, clf)

		prevgray = gray

	if(time > 0):
		process_atom(bins,mag,fmask, tag_img, out)
	return blank_image



for testf in testfolders:
	testpath = basepath + "/Test0"+str(testf)
	images = sorted(glob.glob(testpath + "/*.tif"))


	cnt = len(images)
	filename = "/home/prabhat/Documents/sem7/ML/outlier_detectin/DecisionTree.pkl"

	clf = pickle.load(open(filename, "rb"))
	for i in range(2,cnt-1):
		destfile = basepath+"/Test0"+str(testf)+"/d"+str(i)+".png"
		orig = cv2.imread(images[i], cv2.IMREAD_UNCHANGED)
		orig = cv2.cvtColor(orig,cv2.COLOR_GRAY2BGR)

		sample_images = images[i-2:i+3]

		mask = getFeaturesFromVideo(sample_images,[], None, clf)
		added = cv2.addWeighted( orig, 0.8, blank_image, 0.2, 0.0)
		cv2.imwrite(destfile, added)





