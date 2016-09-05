import sys
import video
import cv2
import numpy as np

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

try: fn = sys.argv[1]
except: fn = 0

atom_shape = [10,10,5]

cam = cv2.VideoCapture(fn)

# first frame in grayscale
ret, prev = cam.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Go through all the frames of the video

mag_threshold = 5

while True:
	#Read next frame
	ret, img = cam.read()
	#Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Calculate Optical for all pixels in the image
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
	flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
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

	hsv = np.zeros((height, width, 3), np.uint8)
	hsv[...,0] = np.minimum(binno*40, 255)
	hsv[...,1] = 255
	hsv[...,2] = 255
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	cv2.imshow('flow', draw_flow(gray, flow))
	cv2.imshow('flow HSV', bgr)
	ch = 0xFF & cv2.waitKey(5)
	if ch == 27:
		break

	prevgray = gray

