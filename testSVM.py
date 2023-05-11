import cv2
import joblib
import numpy as np

# Load the trained SVM model from the saved file
model_file = './svmNew_model.joblib'
svm = joblib.load(model_file)

# Set the list of emotions to recognize
EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprise']

# Initialize the Haar cascade classifier for face detection
CASCADE_FILE = './haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_FILE)

# Initialize the video capture device
cap = cv2.VideoCapture(0)

# Loop through each frame of the video capture
while True:
    # Capture the frame from the video capture device
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Extract the facial region from the grayscale frame
        face = gray[y:y+h, x:x+w]
        
        # Resize the facial region to the same size used for training the model
        face = cv2.resize(face, (48, 48))
        
        # Convert the resized facial region to a 1D array
        face = face.reshape(1, -1)
        
        # Use the SVM model to predict the emotion of the facial region
        prediction = svm.predict(face)[0]
        emotion = EMOTIONS[prediction]
        
        # Draw a rectangle around the detected face and display the predicted emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Facial Emotion Recognition', frame)
    
    # Wait for a key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
