import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from Trainer import face_recognizer, faces, labels, label_dict
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import openai

openai.api_key = "your_openai_api_key_here"


def preprocess_face(face_img, size=(200, 200)):
    """Resize the face image to a fixed size."""
    return cv2.resize(face_img, size)

def compute_lbp_hist(image, numPoints=8, radius=1, grid_x=8, grid_y=8):
    """
    Compute the Local Binary Pattern (LBP) histogram for a grayscale image.
    The image is divided into a grid and a histogram is computed for each cell.
    """
    lbp = local_binary_pattern(image, numPoints, radius, method="uniform")
    h, w = image.shape
    cell_h = h // grid_y
    cell_w = w // grid_x
    hist_features = []
    for i in range(grid_y):
        for j in range(grid_x):
            cell = lbp[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            hist, _ = np.histogram(cell.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            hist_features.extend(hist)
    return np.array(hist_features)

def chi_square_distance(histA, histB, eps=1e-10):
    """Compute the Chi-Square distance between two histograms."""
    return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))


stream_url = "https://192.168.176.166:8080/video"
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("Error: Could not open video stream from phone.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

root = tk.Tk()
root.title("Continuous Best Match Display")
face_popup = tk.Toplevel(root)
face_popup.title("Best Matching Training Face")
face_label = tk.Label(face_popup)
face_label.pack()

# --- Persistence Variables ---
persistent_label = None
persistent_start_time = None
persistent_triggered = False
best_match = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from stream")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rects = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    if len(faces_rects) > 0:
        largest_face = max(faces_rects, key=lambda r: r[2] * r[3])
        (x, y, w, h) = largest_face

        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_resized = preprocess_face(roi_gray, size=(200, 200))

        predicted_label, confidence = face_recognizer.predict(roi_resized)
        predicted_name = label_dict.get(predicted_label, "Unknown")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'{predicted_name}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        candidate_faces = []
        for idx, train_face in enumerate(faces):
            if labels[idx] == predicted_label:
                candidate_faces.append(preprocess_face(train_face, size=(200, 200)))
        test_hist = compute_lbp_hist(roi_resized, numPoints=8, radius=1, grid_x=8, grid_y=8)
        best_distance = float("inf")
        best_match = None
        for candidate in candidate_faces:
            candidate_hist = compute_lbp_hist(candidate, numPoints=8, radius=1, grid_x=8, grid_y=8)
            distance = chi_square_distance(test_hist, candidate_hist)
            if distance < best_distance:
                best_distance = distance
                best_match = candidate

        # --- Continuously Update the Best Match Popup ---
        if best_match is not None:
            rgb_image = cv2.cvtColor(best_match, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            photo = ImageTk.PhotoImage(pil_image)
            face_label.config(image=photo)
            face_label.image = photo 

        if persistent_label is None or persistent_label != predicted_label:
            persistent_label = predicted_label
            persistent_start_time = time.time()
            persistent_triggered = False
        else:
            elapsed = time.time() - persistent_start_time
            if elapsed >= 4 and not persistent_triggered:
                persistent_triggered = True
                messagebox.showinfo("Result", 
                    f"Congratulations! You are most likely {predicted_name}.\nConfidence: {confidence:.2f}")
                prompt = f"how was {predicted_name} responsible for the Uighur camps."
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a knowledgeable assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=150,
                        temperature=0.7,
                    )
                    crimes_info = response.choices[0].message.content.strip()
                    messagebox.showinfo("Crimes", crimes_info)
                except Exception as e:
                    messagebox.showerror("OpenAI API Error", str(e))
    else:
        persistent_label = None
        persistent_start_time = None
        persistent_triggered = False

    cv2.imshow("Face Detection and Recognition", frame)
    root.update_idletasks()
    root.update()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
root.destroy()
