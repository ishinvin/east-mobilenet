import logging, os
logging.disable(logging.WARNING)
logging.disable(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import keras.api._v2.keras as keras
from keras import layers

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from time import time

BATCH_SIZE = 32
IMG_HEIGHT = 20
IMG_WIDTH = 20
# ajustar consoante a localizacao
TRAIN_PATH = "letters/train"
VALIDATION_PATH = "letters/validation"

# Number of classes: a -> z
NUM_CLASSES = 26

train_ds = tf.keras.utils.image_dataset_from_directory(
  TRAIN_PATH,
  color_mode='grayscale',
  labels='inferred',
  label_mode = 'categorical',
  subset="training",
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
  VALIDATION_PATH,
  color_mode='grayscale',
  labels='inferred',
  label_mode = 'categorical',
  subset="validation",
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)



labels = train_ds.class_names
print(labels)


train_ds = train_ds.cache()
val_ds = val_ds.cache()



model = tf.keras.models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.Conv2D(16, 3, padding='same', activation='relu'), 
    layers.Dropout(0.2),
    layers.MaxPooling2D(), 
    layers.Conv2D(32, 3, padding='same', activation='relu'), 
    layers.Dropout(0.2),
    layers.MaxPooling2D(), 
    layers.Conv2D(64, 3, padding='same', activation='relu'), 
    layers.Dropout(0.2),
    layers.MaxPooling2D(), 
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), 
                metrics=['accuracy'])

model.summary()

EPOCHS = 4

start = time()

history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
end = time()

y_pred = model.predict(val_ds)
y_pred = tf.argmax(y_pred, axis=1)

y_true = tf.concat([y for x, y in val_ds], axis=0)
y_true = tf.argmax(y_true, axis=1)

# SAVE MODEL (COMMENTED ON PURPOSE)
#tf.saved_model.save(model, "alphaNet/1/")

print("\nTraining time: ", end - start)

cm = confusion_matrix(y_true, y_pred)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

print(f"Accuracy: {acc[-1]} , Validation Accuracy: {val_acc[-1]}")


plt.figure(2, figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()
