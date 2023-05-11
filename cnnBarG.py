from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.pyplot as plt

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./Emotion_Detection.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
correct_predictions = [0] * len(class_labels)
total_predictions = [0] * len(class_labels)

cap = cv2.VideoCapture(0)

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
            print("\nprediction = ",preds)
            label = class_labels[preds.argmax()]
            print("\nprediction max = ",preds.argmax())
            print("\nlabel = ",label)
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

            # update the correct and total predictions for the current emotion
            idx = preds.argmax()
            if label == class_labels[idx]:
                correct_predictions[idx] += 1
            total_predictions[idx] += 1
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n\n")
    
    # show the video feed with emotion labels
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# calculate accuracy for each emotion
accuracy = [correct / total if total > 0 else 0 for correct, total in zip(correct_predictions, total_predictions)]

# plot the bar graph
fig, ax = plt.subplots()
ax.bar(class_labels, accuracy)
ax.set_xlabel('Emotion')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.0)
ax.set_title('Emotion Recognition Accuracy')
plt.show()
