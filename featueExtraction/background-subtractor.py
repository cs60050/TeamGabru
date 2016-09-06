import numpy as np
import cv2
import os

basepath = os.path.dirname(os.path.abspath(__file__))+"/Sample-Videos/"

def background_subtractor(video_link,method="MOG"):
	cap = cv2.VideoCapture(video_link)
	if method == "MOG":
		cap = cv2.createBackgroundSubtractorMOG()
	elif method == "MOG2":
		fgbg = cv2.createBackgroundSubtractorMOG2()
	elif method=="GMG":
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		fgbg = cv2.createBackgroundSubtractorGMG()


	while(1):
		ret, frame = cap.read()
		fgmask = fgbg.apply(frame)

		if method == "GMG":
			fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

		cv2.imshow('frame',fgmask)
		print(fgmask)
		k=cv2.waitKey(30) & 0xff
		if k == 27:
			break
	cap.release()
	cv2.destroyAllWindows()


def main():
	background_subtractor(basepath+"/VIRAT_S_010005_02_000177_000203.mp4","MOG2")


if __name__ == "__main__":main()
