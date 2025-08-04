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
labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]
model=load_model('trafficnet.h5')

image = io.imread('14.jpg')
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
