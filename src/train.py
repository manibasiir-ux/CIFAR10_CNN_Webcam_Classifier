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

Train_dir = 'data/cifar10/train'
Validation_dir = 'data/cifar10/validation'
Test_dir = 'data/cifar10/test'

model = models.Sequential()
model.add(layers.Input(shape=(150, 150, 3)))

model.add(layers.RandomFlip("horizontal")),
model.add(layers.RandomRotation(0.03)),
model.add(layers.RandomZoom(0.1)),
model.add(layers.RandomTranslation(0.1, 0.1))
model.add(layers.Rescaling(1./255)),

model.add(layers.Conv2D(12, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(34, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss= 'categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=0.001),
              metrics=['acc'])

batch_size = 32
img_height = 150
img_width = 150

train_ds = tf.keras.utils.image_dataset_from_directory(
  Train_dir,
  seed=123,
  image_size=(img_height, img_width),
  label_mode="categorical",
  batch_size=batch_size)




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




class_names = train_ds.class_names
print(class_names)

# Save the class names so webcam_demo.py works without the dataset
with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)


history = model.fit(
      train_ds,
      epochs=20,
      validation_data=val_ds)

test_loss, test_accuracy = model.evaluate( test_ds)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

model.save("models/my_cnn_First_model.keras")
