from tensorflow import keras
import numpy as np
import cv2
import json
import sys
from pathlib import Path

# Classify image files without a webcam:
#   python src/predict.py my_photo.jpg another_photo.png

model_path = Path("models/final_model.keras")

# Below this confidence the model says "unknown" instead of guessing.
unknown_threshold = 0.70

image_paths = sys.argv[1:]

model = keras.models.load_model(
    str(model_path),
    compile=False
)

with open("models/class_names.json") as f:
    class_names = json.load(f)

for image_path in image_paths:
    frame = cv2.imread(image_path)

    if frame is None:
        print(image_path, ": could not read image")
        continue

    # Same preprocessing as the webcam demo
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb_frame, (150, 150))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index])

    if confidence < unknown_threshold:
        print(f"{image_path}: unknown (best guess {predicted_class} {confidence:.0%})")
    else:
        print(f"{image_path}: {predicted_class} {confidence:.1%}")
