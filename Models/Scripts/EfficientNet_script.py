import itertools
from os import listdir

import cv2
import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.layers import *
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  img_to_array)


def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)

def preprocess_image(image_path):
    show_image(image_path)
    image = image_utils.load_img(image_path, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = image.reshape(1,224,224,3)
    image = preprocess_input(image)
    return image

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
    
cm_plot_labels = ['Early_blight','Healthy','Late_blight']
early_stopping_monitor = EarlyStopping(patience=5,monitor='val_accuracy')
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=0.00001)
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
depth=3
IMG_SIZE = 224
NUM_CLASSES = 3

def load_data(train_dir = "/content/drive/MyDrive/Private Data Split_train_test_val/train",val_dir = "/content/drive/MyDrive/Private Data Split_train_test_val/val",test_dir  = "/content/drive/MyDrive/Private Data Split_train_test_val/test"):
    """
    function to load the data
    
    Keyword arguments:
    train_dir -- the path for the dataset to use for training
    test_dir -- the path for the dataset to use for testing
    val_dir -- the path for the dataset to use for validation

    Return: return training set, testing set and validation set
    """
     
    train_data = image_dataset_from_directory(train_dir,label_mode="categorical",image_size = (IMG_SIZE,IMG_SIZE),batch_size = BS,seed = 42,shuffle = True)
    val_data = image_dataset_from_directory(val_dir,label_mode="categorical",image_size = (IMG_SIZE,IMG_SIZE),batch_size = BS,seed = 42,shuffle = False)
    test_data = image_dataset_from_directory(test_dir,label_mode="categorical",image_size = (IMG_SIZE,IMG_SIZE),batch_size = BS,seed = 42,shuffle = False)
    
    return train_data,val_data,test_data

def efficientNet_Setup(trainable=False):
    """function to create the efficient net B5 model 
    
    Keyword arguments:
    trainable -- set the model to trainable or not (default False)
    Return: returns the model
    """
    
    model = EfficientNetB5(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, depth), weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = trainable

    # Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
    # Compile
    model = tf.keras.Model(model.input, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(beta_1 = 0.9, beta_2 = 0.999, decay = 0.0,learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]) 
    return model


#evaluate model
def model_evaluation(model,testing_set,batch_size=BS):
    """function to evaluate the given model
    
    Keyword arguments:
    model -- the model to evaluate 
    testing_set -- the testing set to predict on
    batch_size -- (int) the batch size to use ( default BS variable)
    Return: return_description
    """
    
    predictions = model.predict(x=testing_set, batch_size=batch_size, verbose=1)  
    rounded_predictions=np.argmax(predictions,axis=-1)
    return rounded_predictions

def get_labels_from_tfdataset(tfdataset, batched=False):
    
    labels = list(map(lambda x: x[1], tfdataset)) # Get labels 

    if not batched:
        return tf.concat(labels, axis=0) # concat the list of batched labels

    return labels

#load the data
training_set,validation_set,testing_set=load_data(0.2,True,15,True,0.2)

#create the model instance
model = efficientNet_Setup(True)

#train the model
history = model.fit(training_set,validation_data = validation_set,epochs=EPOCHS,verbose=1,callbacks=[learning_rate_reduction,early_stopping_monitor])

#plot model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#plot model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#save model
model.save('trained_model_colored.h5')

rp=model_evaluation(model,testing_set,32)
array_test1=get_labels_from_tfdataset(testing_set)
test_labels =np.array([])
for i in array_test1:
    result=np.where(i==1)[0][0]
    test_labels=np.append(test_labels,result)
test_labels.astype(int)

cm = confusion_matrix(y_true=test_labels,y_pred=rp)
plot_confusion_matrix(cm=cm,classes=cm_plot_labels,title="confusion matrix")