from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import os

# load images and labels from a directory
data_dir = './validation'
class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
images = []
labels = []
for class_label in class_labels:
    class_dir = os.path.join(data_dir, class_label)
    for filename in os.listdir(class_dir):
        img_path = os.path.join(class_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        images.append(img)
        labels.append(class_labels.index(class_label))

# convert images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# create a face classifier and emotion detection model
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
emotion_model = load_model('./Emotion_Detection.h5')

# initialize lists to store true and predicted labels for each face
true_labels = []
predicted_labels = []

# loop through each image and detect faces and emotions
for i in range(len(images)):
    # detect faces in the image
    gray = images[i]
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # loop through each face and predict the emotion
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_model.predict(roi)[0]
        label = class_labels[preds.argmax()]

        # store the true and predicted labels for this face
        true_label = class_labels[labels[i]]
        true_labels.append(true_label)
        predicted_labels.append(label)

# create a confusion matrix using scikit-learn
conf_mat = confusion_matrix(true_labels, predicted_labels, labels=class_labels)

# plot the confusion matrix using seaborn
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
