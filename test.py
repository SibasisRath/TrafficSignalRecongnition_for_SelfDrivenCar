from keras.models import load_model
from skimage import io
from skimage import transform
from skimage import exposure
import numpy as np
import cv2
import os
from collections import deque
import numpy as np
import argparse
import imutils
firstFrame = None
labelNames = open(os.path.join("dataset", "signnames.csv")).read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]
model=load_model('trafficnet.keras')
'''
image = io.imread('image.png')
image = transform.resize(image, (32, 32))
image = exposure.equalize_adapthist(image, clip_limit=0.1)
			# preprocess the image by scaling it to the range [0, 1]
image = image.astype("float32") / 255.0
image = np.expand_dims(image, axis=0)
			# make predictions using the traffic sign recognizer CNN
preds = model.predict(image)
j = preds.argmax(axis=1)[0]
label = labelNames[j]
print("output",label)
'''
vs = cv2.VideoCapture(0)
while True:
    # grab the current frame and initialize the occupied/unoccupied
	# text
	frame = vs.read()
	
	frame = frame[1]
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < 5000:
			continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
		image = frame
		image = transform.resize(image, (32, 32))
		image = exposure.equalize_adapthist(image, clip_limit=0.1)
        # preprocess the image by scaling it to the range [0, 1]
		image = image.astype("float32") / 255.0
		image = np.expand_dims(image, axis=0)
        # make predictions using the traffic sign recognizer CNN
		preds = model.predict(image)
		j = preds.argmax(axis=1)[0]
		label = labelNames[j]
		print(label)
	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
vs.release()
cv2.destroyAllWindows()