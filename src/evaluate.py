import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Test accuracy, confusion matrix, per-class precision/recall,
# and how the "unknown" threshold changes the results.

Validation_dir = 'data/cifar10/validation'
Test_dir = 'data/cifar10/test'
Reports_dir = Path('reports')

batch_size = 32
img_height = 150
img_width = 150

thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

model = keras.models.load_model(
    "models/final_model.keras",
    compile=False
)

with open("models/class_names.json") as f:
    class_names = json.load(f)

val_ds = tf.keras.utils.image_dataset_from_directory(
  Validation_dir,
  seed=123,
  image_size=(img_height, img_width),
  label_mode="categorical",
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  Test_dir,
  seed=123,
  image_size=(img_height, img_width),
  label_mode="categorical",
  batch_size=batch_size)

# Predict batch by batch and keep the true labels in the same order
val_probs, val_labels = [], []
for images, labels in val_ds:
    val_probs.append(model.predict(images, verbose=0))
    val_labels.append(np.argmax(labels.numpy(), axis=1))
val_probs = np.concatenate(val_probs)
val_labels = np.concatenate(val_labels)

test_probs, test_labels = [], []
for images, labels in test_ds:
    test_probs.append(model.predict(images, verbose=0))
    test_labels.append(np.argmax(labels.numpy(), axis=1))
test_probs = np.concatenate(test_probs)
test_labels = np.concatenate(test_labels)

test_pred = np.argmax(test_probs, axis=1)
test_accuracy = float(np.mean(test_pred == test_labels))
print(f"Test accuracy: {test_accuracy:.4f} on {len(test_labels)} images")

# Confusion matrix: rows = true class, columns = predicted class
num_classes = len(class_names)
confusion = np.zeros((num_classes, num_classes), dtype=int)
for true_label, pred_label in zip(test_labels, test_pred):
    confusion[true_label, pred_label] += 1

print(f"\n{'class':<12}{'precision':>10}{'recall':>8}")
per_class = {}
for i, name in enumerate(class_names):
    precision = confusion[i, i] / max(confusion[:, i].sum(), 1)
    recall = confusion[i, i] / max(confusion[i, :].sum(), 1)
    per_class[name] = {"precision": float(precision), "recall": float(recall)}
    print(f"{name:<12}{precision:>10.3f}{recall:>8.3f}")

# For each threshold: how many images still get an answer, and how accurate those answers are
print("\nUnknown threshold (validation set):")
print(f"{'threshold':>10}{'answered':>10}{'acc on answered':>17}")
threshold_table = []
val_conf = np.max(val_probs, axis=1)
val_pred = np.argmax(val_probs, axis=1)
for t in thresholds:
    answered = val_conf >= t
    if answered.any():
        acc = float(np.mean(val_pred[answered] == val_labels[answered]))
    else:
        acc = float("nan")
    threshold_table.append({"threshold": t, "answered": float(answered.mean()), "accuracy_on_answered": acc})
    print(f"{t:>10.2f}{answered.mean():>10.1%}{acc:>17.1%}")

Reports_dir.mkdir(exist_ok=True)
with open(Reports_dir / "metrics.json", "w") as f:
    json.dump({
        "test_accuracy": test_accuracy,
        "test_images": int(len(test_labels)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "unknown_threshold_validation": threshold_table,
    }, f, indent=2)

plt.figure(figsize=(8, 7))
plt.imshow(confusion, cmap="Blues")
plt.xticks(range(num_classes), class_names, rotation=45, ha="right")
plt.yticks(range(num_classes), class_names)
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, confusion[i, j], ha="center", va="center", fontsize=8,
                 color="white" if confusion[i, j] > confusion.max() / 2 else "black")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion matrix, test set (accuracy {test_accuracy:.2%})")
plt.tight_layout()
plt.savefig(Reports_dir / "confusion_matrix.png", dpi=150)
plt.show()
