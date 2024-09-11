import face_recognition
import cv2
import os
import glob
import numpy as np


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for faster processing
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Load encoding images from the provided path.
        :param images_path: Path to the folder containing images.
        """
        # Load all images from the folder
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print(f"{len(images_path)} encoding images found.")

        # Loop over each image path and process
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract the file name without extension
            basename = os.path.basename(img_path)
            filename, ext = os.path.splitext(basename)

            # Get face encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Store the encoding and name
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)

        print("Encoding images loaded successfully.")

    def detect_known_faces(self, frame):
        """
        Detect faces in a frame and match with known faces.
        :param frame: Frame in which faces are to be detected.
        :return: Detected face locations and names.
        """
        # Resize the frame for faster face detection
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)

        # Convert the frame to RGB color format
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Check if the face matches any known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Find the known face with the smallest distance
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Scale back face locations to the original frame size
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing

        return face_locations.astype(int), face_names
