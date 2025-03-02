import os
import cv2
import numpy as np

# Initialize face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def prepare_training_data(data_folder_path):
    faces = []
    labels = []
    label_dict = {}  # mapping numeric label to person's name
    current_label = 0

    # Loop through each folder (each person's images)
    for person_name in os.listdir(data_folder_path):
        person_folder = os.path.join(data_folder_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        label_dict[current_label] = person_name

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect the face in the image
            faces_rects = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5)
            if len(faces_rects) == 0:
                continue  # no face found

            # Assume the first detected face is the subject
            (x, y, w, h) = faces_rects[0]
            face = gray[y:y+h, x:x+w]

            faces.append(face)
            labels.append(current_label)

        current_label += 1
    return faces, labels, label_dict


# Example usage: update the path to your dataset
data_folder_path = r'downloaded_images'
faces, labels, label_dict = prepare_training_data(data_folder_path)
print(f"Total faces: {len(faces)}, Total labels: {len(labels)}")

# Create the LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

<<<<<<< HEAD
# test_image = faces[1]  # Use the first training image, for instance
# predicted_label, confidence = face_recognizer.predict(test_image)
# print(f"Predicted: {label_dict[predicted_label]}, Confidence: {confidence}")
=======
test_image = faces[1]  # Use the first training image, for instance
predicted_label, confidence = face_recognizer.predict(test_image)
print(f"Predicted: {label_dict[predicted_label]}, Confidence: {confidence}")
>>>>>>> 7d385272d74a047e6509f1d4160a3d5a95200fe6
