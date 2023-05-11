from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.pyplot as plt

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier =load_model('./Emotion_Detection.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
emotions = {'Angry': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}

cap = cv2.VideoCapture(0)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plt.ion()

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            # make a prediction on the ROI, then lookup the class
            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            emotions[label] += 1
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

    axs[0].cla()
    axs[0].bar(emotions.keys(), emotions.values())
    axs[0].set_title('Emotions Distribution')

    axs[1].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Emotion Detector')
    axs[1].axis('off')
    
    plt.draw()
    plt.pause(0.01)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
