import cv2
import face_recognition
import numpy as np
import os

# Create a directory for storing all the training images, if already exits do nothing !!!
os.makedirs("faces", exist_ok = True)

# Get the name of the image
name = input("Enter the name: ")

# Initialize video capturing
cap = cv2.VideoCapture(0)

# Capture frames in a loop
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture!!!")
        break

    # Display the captured frame
    cv2.imshow ("Press 'c' to Capture or 'q' to quit", frame)

    # Capture the frame when 'c' is pressed
    if cv2.waitKey(1) & 0xFF == ord('c'):
        img_path = f'faces/{name}.jpg'
        cv2.imwrite(img_path, frame) # Save the original frame
        print (f"Image {name} saved successfully at: {img_path}")
        
        # Convert the frame to RGB, because open_cv default is BGR
        img_rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rbg) # Returns a list of face encodings
        
        if encodings:
            # Save the encodings of the image as a numpy array
            np.save(f'face/{name}_encoding.npy', encodings[0]) # Save the first face appeared in the image
            print(f"Encoding saved for {name}")
    
        else:
            print ("No face detected, Try again.")
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
        
    
        
    