from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
import cv2
import numpy as np
import os
import joblib
from tqdm import tqdm

# Set the directory of the dataset and haar cascades
DATASET_DIR = './DS'
CASCADE_FILE = './haarcascade_frontalface_default.xml'

# Set the list of emotions to recognize
EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprise']

# Define the size of the images to use for training and testing
IMG_SIZE = (48, 48)

# Initialize arrays to store training and testing data and labels
X_train = []
y_train = []
X_test = []
y_test = []

# Loop through each emotion and load the images into the training and testing arrays
for i, emotion in enumerate(EMOTIONS):
    # Set the directory of the current emotion
    emotion_dir = os.path.join(DATASET_DIR, emotion)
    
    # Load the images into the training array
    train_dir =os.path.join('./DS/train',emotion)
    for filename in tqdm(os.listdir(train_dir), desc='Loading training images for ' + emotion):
        filepath = os.path.join(train_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        X_train.append(img)
        y_train.append(i)
    
    # Load the images into the testing array
    test_dir = os.path.join('./DS/test',emotion)
    for filename in tqdm(os.listdir(test_dir), desc='Loading testing images for ' + emotion):
        filepath = os.path.join(test_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        X_test.append(img)
        y_test.append(i)

# Convert the training and testing data and labels into numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Flatten the images into 1D arrays for training and testing
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Create a support vector machine (SVM) model and fit it to the training data
svm = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3, random_state=42)
print('Training the SVM model...')
n_epochs = 10
batch_size = 32
n_batches = int(np.ceil(len(X_train) / batch_size))

for epoch in range(n_epochs):
    for batch in tqdm(range(n_batches), desc='Epoch {}/{}'.format(epoch+1, n_epochs)):
        start_idx = batch * batch_size
        end_idx = min((batch+1) * batch_size, len(X_train))
        batch_X = X_train[start_idx:end_idx]
        batch_y = y_train[start_idx:end_idx]
        svm.partial_fit(batch_X, batch_y, classes=np.unique(y_train))

# Save the trained SVM model to a file
model_file = 'svmNew_model.joblib'
joblib.dump(svm, model_file)

