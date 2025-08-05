# Import necessary packages
from keras.models import load_model
from skimage import transform, exposure, io
import numpy as np
import cv2
import imutils
import os

# --- 1. SETUP: Load Model and Labels (Your code was correct here) ---
print("[INFO] Loading model and label names...")
labelNames = open(os.path.join("dataset", "signnames.csv")).read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]
model = load_model('trafficnet.keras')

# Initialize the video stream and a variable for the first frame of motion detection
print("[INFO] Starting video stream...")
vs = cv2.VideoCapture(0)
firstFrame = None
cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
cv2.namedWindow("Threshold Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Frame Delta", cv2.WINDOW_NORMAL)
# --- 2. MAIN LOOP: Process video frames ---
while True:
    # Grab the current frame
    ret, frame = vs.read()

    # If the frame was not grabbed, then we have reached the end of the stream
    if not ret:
        print("Error: Could not read frame from video stream.")
        break

    # Resize the frame to a manageable size and convert it to grayscale for motion detection
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # If the first frame is None, initialize it. This is the background reference.
    if firstFrame is None:
        firstFrame = gray
        continue

    # --- 3. MOTION DETECTION ---
    # Compute the difference between the current frame and the first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes, then find contours
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # --- 4. CLASSIFICATION ON DETECTED OBJECTS ---
    # Loop over the contours (the detected moving objects)
    for c in cnts:
        # If the contour is too small, ignore it
        if cv2.contourArea(c) < 1000: # You can adjust this threshold
            continue

        # Compute the bounding box for the contour, which is our Region of Interest (ROI)
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Extract the ROI from the *original color frame*
        roi = frame[y:y+h, x:x+w]

        # --- 5. PREPROCESSING THE ROI FOR THE MODEL ---
        # This pipeline MUST MATCH the one used during training.
        # Note: skimage works with RGB images, so we convert the BGR roi to RGB.
        try:
            image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            image = transform.resize(image, (32, 32))
            image = exposure.equalize_adapthist(image, clip_limit=0.1)
            
            # Scale pixel intensities to the range [0, 1]
            image = image.astype("float32") / 255.0
            
            # Add a batch dimension so we can pass it through the model
            image = np.expand_dims(image, axis=0)

            # Make a prediction on the ROI
            preds = model.predict(image)
            j = preds.argmax(axis=1)[0]
            label = labelNames[j]

            # Draw the bounding box and the prediction on the original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            # This try-except block prevents crashes if the ROI is invalid (e.g., has zero size)
            print(f"[WARNING] Could not process ROI: {e}")
            continue

    # --- 6. DISPLAY RESULTS ---
    # Show the output frame and the motion detection masks
    cv2.imshow("Video Feed", frame)
    cv2.imshow("Threshold Mask", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key is pressed, break from the loop
    if key == ord("q"):
        break

# --- 7. CLEANUP ---
print("[INFO] Cleaning up...")
vs.release()
cv2.destroyAllWindows()
