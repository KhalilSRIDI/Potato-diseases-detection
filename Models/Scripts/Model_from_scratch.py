import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import cv2
import os
import matplotlib.image as mpimg
import random
from sklearn import preprocessing
import tensorflow.keras as keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping,TensorBoard
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten,Input,MaxPooling2D, Activation,Dropout,GlobalAveragePooling2D,Average,AveragePooling2D,GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.applications.xception import Xception
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image_dataset_from_directory
import itertools
from sklearn.metrics import confusion_matrix




def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
SIZE = 256
SEED_TRAINING = 121
SEED_TESTING = 197
SEED_VALIDATION = 164
CHANNELS = 3
n_classes = 3
EPOCHS = 40
BATCH_SIZE = 32
input_shape = (SIZE, SIZE, CHANNELS)
early_stopping_monitor = EarlyStopping(patience=8,monitor='val_loss')

train_dir = "/content/drive/MyDrive/_train_test_val/train"
val_dir = "/content/drive/MyDrive/_train_test_val/val"
test_dir  = "/content/drive/MyDrive/_train_test_val/test"

train_data = image_dataset_from_directory(train_dir,label_mode="categorical",image_size = (SIZE,SIZE),batch_size = BATCH_SIZE,seed = 42,shuffle = True)
val_data = image_dataset_from_directory(val_dir,label_mode="categorical",image_size = (SIZE,SIZE),batch_size = BATCH_SIZE,seed = 42,shuffle = False)
test_data = image_dataset_from_directory(test_dir,label_mode="categorical",image_size = (SIZE,SIZE),batch_size = BATCH_SIZE,seed = 42,shuffle = False)

train_data = train_data.prefetch(tf.data.AUTOTUNE)
test_data = test_data.prefetch(tf.data.AUTOTUNE)
val_data = val_data.prefetch(tf.data.AUTOTUNE)

modelFS = keras.models.Sequential([
        keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation ='relu'),
        keras.layers.Dense(n_classes, activation='softmax')
    ])

modelFS.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
checkpointer = ModelCheckpoint('best_model1.h5',monitor='accuracy',verbose=1,save_best_only=True,save_weights_only=True)

modelFS.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics  = ['accuracy']
    )

historyFS = modelFS.fit(train_data,epochs=40,steps_per_epoch=len(train_data),validation_data = val_data,validation_steps = int(0.25*len(val_data)),verbose=1,callbacks=[learning_rate_reduction,early_stopping_monitor,checkpointer])

accuracy = historyFS.history['accuracy']
loss = historyFS.history['loss']
val_loss = historyFS.history['val_loss']
val_accuracy = historyFS.history['val_accuracy']

plt.figure(figsize=(14, 14))
plt.subplot(2, 2, 1)
plt.plot(range(13), accuracy, label='Training Accuracy')
plt.plot(range(13), val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy : Training vs. Validation ')

plt.subplot(2, 2, 2)
plt.plot(range(13), loss, label='Training Loss')
plt.plot(range(13), val_loss, label='Validation Loss')
plt.title('Loss : Training vs. Validation Model From Scratch')
plt.legend(loc='upper right')
plt.show()

modelFS.save('modelFS.h5')
modelFS.save_weights('modelFSweights.h5')

Y_pred = modelFS.predict(test_data, test_data.samples / BATCH_SIZE)
val_preds = np.argmax(Y_pred, axis=1)
val_trues =test_generator.classes

print(classification_report(val_trues, val_preds))

confusion_matrix = pd.crosstab(val_trues, val_preds, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)

predictions = modelFS.predict(
      x=test_data
    , batch_size=32
    , verbose=1
)

rounded_predictions=np.argmax(predictions,axis=-1)

test_dataset = [(example.numpy(), label.numpy()) for example, label in test_data]


def get_labels_from_tfdataset(tfdataset, batched=False):

    labels = list(map(lambda x: x[1], tfdataset)) # Get labels

    if not batched:
        return tf.concat(labels, axis=0) # concat the list of batched labels

    return labels

array_test1=get_labels_from_tfdataset(test_data)


test_labels =np.array([])

for i in array_test1:
  result=np.where(i==1)[0][0]
  test_labels=np.append(test_labels,result)
test_labels.astype(int)
print(test_labels)



cm = confusion_matrix(y_true=test_labels,y_pred=rounded_predictions)

cm_plot_labels = ['Early_blight','Healthy','Late_blight']

plot_confusion_matrix(cm=cm,classes=cm_plot_labels,title="confusion matrix")
