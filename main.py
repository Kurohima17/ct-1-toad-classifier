import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable OneDNN
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load and preprocess image
def load_image(img_path):
    """
    Load and preprocess image
    :param img_path: path to image
    :return: processed tensor
    """
    # Instantiate local variables
    img_height = 256
    img_width = 256

    img = image.load_img(img_path, target_size=(img_height, img_width)) # Load image
    img = image.img_to_array(img) # Convert image to numpy array
    img = np.expand_dims(img, axis=0) # Add batch dimension
    img = tf.keras.applications.xception.preprocess_input(img) # Preprocess image for Xception model
    return img

# Predict whether image is cane toad or not, extracting confidence level
def predict_image(img_path, debug=False):
    """
    Predict whether image is cane toad or not, extracting confidence level
    :param img_path: path to image
    :return: prediction
    """
    # Load model
    if debug:
        print("loading model...")
    model = tf.keras.models.load_model('ct-1v2.keras')

    # Load image
    if debug:
        print("loading image...")
    img = load_image(img_path)

    # Make prediction
    if debug:
        print("making prediction...")
    prediction = model.predict(img)
    if debug:
        print(prediction.shape)

    if debug:
        print("decoding prediction...")
    predictionLabel = decode_prediction(prediction[0][0])

    return predictionLabel

def decode_prediction(prediction):
    """
    Decode prediction
    :param prediction: raw Xception prediction
    :return: JSON string 
    """
    is_canetoad = prediction > 0.5
    decoded_pred = {"is_canetoad": is_canetoad, "confidence": prediction}
    return decoded_pred

# Main function
if __name__ == "__main__":
    # Instantiate local variables
    img_path = "data/canetoad/1.jpg"

    # Predict image
    pred = predict_image(img_path, debug=True)
    print(pred)