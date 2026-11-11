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

Train_dir = 'data/cifar10/train'
Validation_dir = 'data/cifar10/validation'
Test_dir = 'data/cifar10/test'

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

saved_model = keras.models.load_model(
    "models/my_cnn_First_model.keras"
)

conv_base = keras.Sequential(
    saved_model.layers[:-4],
    name="conv_base"
)

conv_base.summary()

final_model = models.Sequential()

final_model.add(conv_base)

conv_base.trainable = True

for layer in conv_base.layers:
    layer.trainable = False

selected_indices = [7, 11, 13]

for i in selected_indices:
    conv_base.layers[i].trainable = True


final_model.add(layers.GlobalAveragePooling2D())
final_model.add(layers.Dense(128, activation='relu'))
final_model.add(layers.Dense(10, activation='relu'))
final_model.add(layers.Dense(10, activation='softmax'))

# Build the model so the new Dense layers have weights before they are mapped
# to optimizers below (without this, a fresh run crashes in fit() with KeyError)
final_model.build((None, 150, 150, 3))




optimizers = [tf.keras.optimizers.Adam(learning_rate=0.0012),
              tf.keras.optimizers.Adam(learning_rate=0.001),
              tf.keras.optimizers.RMSprop(learning_rate=0.001),
              tf.keras.optimizers.RMSprop(learning_rate=0.0009)]

optimizers_and_layers = [(optimizers[0], conv_base.layers[7]),
                         (optimizers[1], conv_base.layers[11]),
                         (optimizers[2], conv_base.layers[13]),
                         (optimizers[3], final_model.layers[-3]),
                         (optimizers[3], final_model.layers[-2]),
                         (optimizers[3], final_model.layers[-1])
                         ]
optimizer_map = {}

for optimizer, layer_group in optimizers_and_layers:
    if not isinstance(layer_group, list):
        layer_group = [layer_group]

    for layer in layer_group:
        for variable in layer.trainable_weights:
            optimizer_map[variable.path] = optimizer

def select_optimizer(variable):
    return optimizer_map[variable.path]

for layer in [final_model.layers[-3], final_model.layers[-2],final_model.layers[-1]]:
    for variable in layer.trainable_weights:
        optimizer_map[variable.path] = optimizers[3]

missing = [
    variable.path
    for variable in final_model.trainable_weights
    if variable.path not in optimizer_map
]

print("Missing after update:", missing)


opt = MultiOptimizer(select_optimizer)

final_model.compile(
    optimizer=opt,
    loss="categorical_crossentropy",
    metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")])


fine_tune_epoches = 10
loss0, accuracy0 = final_model.evaluate(val_ds)

final_history = final_model.fit(train_ds,
                    epochs=fine_tune_epoches,
                    validation_data=val_ds)


test_loss, test_accuracy = final_model.evaluate(test_ds)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


epochs = range(1, len(final_history.history['loss']) + 1)

plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(
    epochs,
    final_history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    epochs,
    final_history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.title('Fine-Tuning: Training and Validation Accuracy')

plt.subplot(2, 1, 2)

plt.plot(
    epochs,
    final_history.history['loss'],
    label='Training Loss'
)

plt.plot(
    epochs,
    final_history.history['val_loss'],
    label='Validation Loss'
)

plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.xlabel('Epoch')
plt.title('Fine-Tuning: Training and Validation Loss')

plt.tight_layout()
plt.show()


final_model.save("models/final_model.keras")
