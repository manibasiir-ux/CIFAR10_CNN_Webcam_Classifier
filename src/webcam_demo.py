import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.preprocessing import image
import numpy as np
import cv2
from keras import layers
from keras import models
import matplotlib.pyplot as plt
from keras import optimizers
from keras.optimizers import MultiOptimizer
from pathlib import Path
import json

model_path = Path(
    "models/final_model.keras"
)

# Below this confidence the model says "unknown" instead of guessing (0 turns it off)
unknown_threshold = 0.70

webcam_model = keras.models.load_model(
    str(model_path),
    compile=False
)

print("Model input shape:", webcam_model.input_shape)
print("Model output shape:", webcam_model.output_shape)




# Class names saved by train.py
with open("models/class_names.json") as f:
    class_names = json.load(f)

assert len(class_names) == webcam_model.output_shape[-1]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open the webcam")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read camera frame")
            break

        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(rgb_frame, (150, 150))

        img = img.astype("float32")

        # Add a batch dimension: (1, 150, 150, 3)
        img = np.expand_dims(img, axis=0)

        predictions = webcam_model.predict(
            img, verbose=0
        )

        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])

        if confidence < unknown_threshold:
            label = f"unknown (best guess {predicted_class} {confidence:.0%})"
            color = (0, 165, 255)
        else:
            label = f"{predicted_class}: {confidence:.1%}"
            color = (0, 255, 0)

        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.imshow("My CNN Webcam Classifier", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Release camera and close window
    cap.release()
    cv2.destroyAllWindows()
