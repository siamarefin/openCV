import cv2
from simple_facerec import SimpleFacerec

# Initialize SimpleFacerec
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Initialize webcam (try different indices if needed: 0, 1, 2)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera. Check the camera index or connection.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret or frame is None:
        print("Error: Failed to capture a frame.")
        break

    # Detect known faces in the frame
    face_locations, face_names = sfr.detect_known_faces(frame)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
