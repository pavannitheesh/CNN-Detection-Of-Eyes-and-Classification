import os
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from pygame import mixer
from PIL import Image
# Load pre-trained model for eye detection
model = load_model(r'C:\Desktop\MINOR\CNN-Detection-Of-Eyes-and-Classification\Models\cnnfinal.h5')
face = cv2.CascadeClassifier('../haarcascade_files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('../haarcascade_files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('../haarcascade_files/haarcascade_righteye_2splits.xml')
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
mixer.init()
sound = mixer.Sound('../alarm/beep-04.wav')
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
path = os.getcwd()
# Streamlit app setup
st.title("Driver Drowsiness Detection")
st.write("Detects if the driver's eyes are open or closed in real time.")
# OpenCV video capture
run = st.checkbox("Run Live Detection")
stop_button = st.button("Stop Live Detection")
  # Set 0 for default webcam
# Loop for live feed
cap = cv2.VideoCapture(0)
if run:
    stframe = st.empty()  # Placeholder for video frame
    while True:
        if stop_button:
            break
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break
        height,width = frame.shape[:2]
        # Convert frame to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)
        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
        # Process eyes for drowsiness detection
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
        for (x, y, w, h) in right_eye:
            r_eye=frame[y:y+h,x:x+w]
            count=count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            predict_x=model.predict(r_eye) 
            rpred=np.argmax(predict_x,axis=1)
            if(rpred[0]==1):
                lbl='Alert' 
            if(rpred[0]==0):
                lbl='Sleepy!'
            break
        for (x, y, w, h) in left_eye:
            l_eye=frame[y:y+h,x:x+w]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            predict_x=model.predict(l_eye) 
            lpred=np.argmax(predict_x,axis=1)
            if(lpred[0]==1):
                lbl='Alert'   
            if(lpred[0]==0):
                lbl='Sleepy!'
            break
        # Display the frame
        if(rpred[0]==0 and lpred[0]==0):
            score=score+1
            cv2.putText(frame,"Sleepy!",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
        else:
            score=score-1
            cv2.putText(frame,"Alert",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)


        if(score<0):
            score=0   
        cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        if(score>3):
        #person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path,'image.jpg'),frame)
            try:
                sound.play()

            except:  # isplaying = False
                pass
            if(thicc<16):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Streamlit
        stframe.image(frame_rgb, channels="RGB")
        # Exit on keypress 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Release resources
cap.release()
cv2.destroyAllWindows()
