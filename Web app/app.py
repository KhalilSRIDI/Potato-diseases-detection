import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask


app = Flask(__name__)

app.secret_key = "secret key"


# loading models
model = tf.keras.models.load_model('modelFS.h5')
disease = ['Early_blight', 'Healthy', 'Late_blight']


def load_image(img_path):

    img = image.load_img(img_path, target_size=(256, 256))
    # (height, width, channels)
    img_tensor = image.img_to_array(img)
    # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    return img_tensor


def prediction(img_path):
    # load the image
    new_image = load_image(img_path)

    # predict the image label
    pred_disease = model.predict(new_image)
    print(pred_disease)
    # label generation
    disease_labels = np.array(pred_disease)
    disease_labels[disease_labels >= 0.6] = 1
    disease_labels[disease_labels < 0.6] = 0
    disease_index = np.argmax(disease_labels)

    return disease_index


@app.route('/about.html')
def about():
    return render_template('about.html')


@app.route('/contact.html')
def contact():
    return render_template('contact.html')


@app.route('/products.html')
def products():
    return render_template('products.html')


@app.route('/services.html')
def services():
    return render_template('services.html')


@app.route('/predict.html')
def render_predict():
    return render_template('predict.html')


@app.route('/diseases.html')
def diseases():
    return render_template('diseases.html')


@app.route('/prevention.html')
def prevention():
    return render_template('prevention.html')


@app.route('/treatment.html')
def treatment():
    return render_template('treatment.html')


@app.route('/index.html')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join('static/', filename)
    file.save(file_path)
    predict_d = prediction(file_path)
    prediction_str = "\rThe leaf is : "+disease[predict_d]
    return render_template('predict.html', prediction_text='{}'.format(prediction_str))


if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
