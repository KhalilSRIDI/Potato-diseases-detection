
import tensorflow as tf
# Load the trained TensorFlow model.
model_to_convert = tf.keras.models.load_model("trained_model_colored (2).h5")
# Convert the model to TensorFlow Lite.
converter = tf.lite.TFLiteConverter.from_keras_model(model_to_convert)
tflite_model = converter.convert()

# Save the TensorFlow Lite model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)