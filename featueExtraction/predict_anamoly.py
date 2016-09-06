
from __future__ import print_function
import cv2
import numpy as np
import pickle
import os

# destpath = os.path.dirname(os.path.abspath(__file__))+"/Output/10/Ped1"

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def process_atom(bins, magnitude, fmask, classifier, atom_shape=[10,10,5]):

	bin_count = np.zeros(9, np.uint8)
	h,w, t = bins.shape
	tagged = np.zeros((h, w, 1), np.uint8)
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

			features = hs.tolist()
			features.extend([f_cnt, atom_mag])

			tag = classifier.predict(features)[0]

			tagged[i:i_end, j:j_end] = tag * 255

	cv2.imshow("tag", tagged)
	cv2.waitKey(5)


	return 0


def getPredictionForVideo(video_link, classifier, mag_threshold=1e-3, atom_shape=[10,10,5]):

    cam = cv2.VideoCapture(video_link)
    # first frame in grayscale
    ret, prev = cam.read()
    print(ret)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
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
        ret, img = cam.read()
        #img = cv.imread(imagelist.__next__(), cv2.IMREAD_GRAYSCALE)
        cv2.imshow('image',img)
        cv2.waitKey(5)
        # Get foreground/background
        fmask[...,time] = fgbg.apply(img)
        #Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
            process_atom(bins, mag, fmask, classifier)

        prevgray = gray

    if(time > 0):
        process_atom(bins,mag,fmask, classifier)



CLASSIFIER_DIR = "../ML-Model/TrainedClassifiers/10/PED 1/"
video = input("Enter video file name, 0 for webcam")
classifier_name = input("Enter the classifier name")

clf = pickle.load(open(CLASSIFIER_DIR + "/" + 'Decision Trees.pkl', 'rb'))

getPredictionForVideo(video, clf)
