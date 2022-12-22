# -*- coding: utf-8 -*-
"""PDD-VGG16-Script-

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZhuGZQm6TA3_aPju456jp7NkRQwRty7i
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from google.colab import drive
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

drive.mount('/content/drive')
def load_data(ROOT_DIR,IMAGE_SIZE,BATCH_SIZE):
  datagen_train = ImageDataGenerator(rescale = 1./255)  
  datagen_valid = ImageDataGenerator(rescale = 1./255)
  datagen_test = ImageDataGenerator(rescale = 1./255)
  train_it = datagen_train.flow_from_directory(
      ROOT_DIR+"/train",
      target_size=(IMAGE_SIZE, IMAGE_SIZE),
      color_mode="rgb",
      class_mode="categorical",
      batch_size=BATCH_SIZE,
  )
  valid_it = datagen_valid.flow_from_directory(
      ROOT_DIR+"/val",
      target_size=(IMAGE_SIZE, IMAGE_SIZE),
      color_mode="rgb",
      class_mode="categorical",
      batch_size=BATCH_SIZE,
  )
  test_it = datagen_test.flow_from_directory(
      ROOT_DIR+"/test",
      target_size=(IMAGE_SIZE, IMAGE_SIZE),
      color_mode="rgb",
      class_mode="categorical",
      batch_size=BATCH_SIZE,
  )
  return train_it, valid_it, test_it
def vgg16_model(ROOT_DIR,IMAGE_SIZE,CHANNELS,BATCH_SIZE,EPOCHS,NUMBER_CLASSES):
  train_it, valid_it, test_it = load_data(ROOT_DIR,IMAGE_SIZE,BATCH_SIZE)
  base_model = keras.applications.VGG16(
    weights='imagenet',  
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
    include_top=False)
  base_model.trainable = False
  inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
  x = base_model(inputs, training=False)
  x = keras.layers.GlobalAveragePooling2D()(x)
  outputs = keras.layers.Dense(NUMBER_CLASSES)(x)
  model = keras.Model(inputs, outputs)
  model.summary()
  model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.CategoricalAccuracy()])
  history=model.fit(train_it,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=valid_it,
          shuffle=True,
          initial_epoch=0,
          steps_per_epoch=train_it.n // train_it.batch_size,
          validation_steps=valid_it.n // valid_it.batch_size,
          validation_batch_size=BATCH_SIZE,
          )

  return history
def plot_history(history,EPOCHS):
  accuracy = history.history['categorical_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  val_accuracy = history.history['val_categorical_accuracy']

  plt.figure(figsize=(14, 14))
  plt.subplot(2, 2, 1)
  plt.plot(range(1), accuracy, label='Training Accuracy')
  plt.plot(range(1), val_accuracy, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Accuracy : Training vs. Validation ')

  plt.subplot(2, 2, 2)
  plt.plot(range(1), loss, label='Training Loss')
  plt.plot(range(1), val_loss, label='Validation Loss')
  plt.title('Loss : Training vs. Validation ')
  plt.legend(loc='upper right')
  plt.show()
# earlyStopping, learning rate , optim etc etc