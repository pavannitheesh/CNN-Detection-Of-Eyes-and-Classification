import cv2
import numpy as np
import os
from keras.models import load_model
from pygame import mixer
import time

# Initialize the alarm sound and face/eye cascades
mixer.init()
sound = mixer.Sound('alarm/beep-04.wav')
model = load_model(r'C:\Desktop\MINOR\CNN-Detection-Of-Eyes-and-Classification\Models\cnnfinal.h5')
face = cv2.CascadeClassifier('../haarcascade_files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('../haarcascade_files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('../haarcascade_files/haarcascade_righteye_2splits.xml')
#model = load_model('data set/data/cnnfinal.keras')  # Adjust this path to your model file

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize counters and thickness
count = 0
score = 0
thicc = 2
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
rpred = [99]
lpred = [99]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face, left eye, and right eye
    faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Draw rectangle around face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    # Process right eye
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24)) / 255.0
        r_eye = np.expand_dims(r_eye.reshape(24, 24, -1), axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=1)
        break

    # Process left eye
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24)) / 255.0
        l_eye = np.expand_dims(l_eye.reshape(24, 24, -1), axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=1)
        break

    # Check if both eyes are closed
    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Sleepy!", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Alert", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0

    # Show score on the screen
    cv2.putText(frame, f'Score: {score}', (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Activate alarm and increase thickness if sleepy
    if score > 5:  # Set threshold as needed
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
        if thicc < 16:
            thicc += 2
        else:
            thicc = 2
        try:
            sound.play()
        except:
            pass
    else:
        sound.stop()

    # Display the frame
    cv2.imshow('Driver Drowsiness Detector', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
