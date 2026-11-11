import numpy as np
from keras.datasets import cifar10
from PIL import Image
from pathlib import Path

# Downloads CIFAR-10 and writes it as PNG folders, the layout train.py expects:
#   data/cifar10/train/      40,000 images
#   data/cifar10/validation/ 10,000 images (1,000 per class, taken from the official training set)
#   data/cifar10/test/       10,000 images (the official test set)

Data_dir = Path("data/cifar10")

class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
val_per_class = 1000

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.ravel()
y_test = y_test.ravel()

# Pick 1,000 validation images per class with a fixed seed
rng = np.random.default_rng(123)
val_idx = []
for c in range(len(class_names)):
    class_idx = np.flatnonzero(y_train == c)
    val_idx.append(rng.choice(class_idx, val_per_class, replace=False))
val_idx = np.concatenate(val_idx)

train_mask = np.ones(len(y_train), dtype=bool)
train_mask[val_idx] = False

splits = [
    ("train", x_train[train_mask], y_train[train_mask]),
    ("validation", x_train[val_idx], y_train[val_idx]),
    ("test", x_test, y_test),
]

for split_name, images, labels in splits:
    for i, (img, label) in enumerate(zip(images, labels)):
        class_dir = Data_dir / split_name / class_names[int(label)]
        class_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(class_dir / f"{split_name}_{i:05d}.png")
    print(split_name, len(labels), "images")

print("Done:", Data_dir)
