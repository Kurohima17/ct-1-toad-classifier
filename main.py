import os
import tensorflow as tf

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

    image = tf.io.read_file(img_path) # Load image
    image = tf.image.decode_jpeg(image, channels=3) # Decode image
    image = tf.image.resize(image, [img_height, img_width]) # Resize image to 256x256
    image /= 255.0  # normalize to [0,1] range
    return image

# Predict whether image is cane toad or not, extracting confidence level
def predict_image(img_path):
    """
    Predict whether image is cane toad or not, extracting confidence level
    :param img_path: path to image
    :return: prediction
    """
    # Load model
    model = tf.keras.models.load_model('ct-1.keras')

    # Load image
    img = load_image(img_path)

    # Make prediction
    prediction = model.predict(tf.expand_dims(img, 0))

    return prediction

# Main function
if __name__ == "__main__":
    # Instantiate local variables
    img_path = "data/canetoad/9.jpeg"

    # Predict image
    prediction = predict_image(img_path)

    # Print prediction
    if prediction[0][0] > 0.5:
        print("Cane toad")
    else:
        print("Not cane toad") 