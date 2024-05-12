import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable OneDNN
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# Author: Ditiphatra (Hima) Chanarithichai
# Cereated for: FIT5120 - Industry Experience Project

# Last update: 12 May 2024

# References
# https://www.tensorflow.org/datasets/api_docs/python/tfds
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling2D
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
# https://www.tensorflow.org/api_docs/python/tf/keras/Model

# This script contain code snippets recommended by ChatGPT and Copilot - revised by author

# Define parameters for dataset builder
# batch_size = len(os.listdir('data/canetoad')) # length of dataset
img_height = 256
img_width = 256
data_dir = 'data'

def load_dataset(data_dir, test_split=0.2, image_size=(224, 224), batch_size=32):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=test_split,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=test_split,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )

    # Get class names and number of classes
    class_names = train_dataset.class_names
    num_classes = len(class_names)

    # Normalise image
    train_dataset = train_dataset.map(normalize_img)
    test_dataset = test_dataset.map(normalize_img)

    return train_dataset, test_dataset, num_classes

def normalize_img(image, label):
    # Convert the datasets to float32 and normalize the images
    return tf.cast(image, tf.float32) / 255., label

def create_few_shot_model(input_shape, num_classes):
    # Create a few-shot learning model based on Xception CNN
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

def few_shot_learning(train_dataset, test_dataset, num_classes, num_shot):
    input_shape = train_dataset.element_spec[0].shape[1:]  # Get the input shape from the dataset
    model = create_few_shot_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Convert the entire dataset to NumPy arrays
    images, labels = next(iter(train_dataset.batch(len(train_dataset))))
    images = images.numpy().reshape(-1, *input_shape)
    labels = labels.numpy().reshape(-1)
    
    # Prepare support set and query set from train_dataset
    support_set_images = []
    support_set_labels = []
    query_set_images = []
    query_set_labels = []

    for class_index in range(num_classes):
        class_indices = np.where(labels == class_index)[0]
        np.random.shuffle(class_indices)
        support_indices = class_indices[:num_shot]
        query_indices = class_indices[num_shot:]

        for class_index in range(num_classes):
            # Get indices of images for the current class
            class_indices = np.where(labels == class_index)[0]
            np.random.shuffle(class_indices)
            support_indices = class_indices[:num_shot]
            query_indices = class_indices[num_shot:]

            # Append images and labels to support set and query set
            for index in support_indices:
                support_set_images.append(images[index])
                support_set_labels.append(labels[index])
            for index in query_indices:
                query_set_images.append(images[index])
                query_set_labels.append(labels[index])

    support_set_images = np.array(support_set_images)
    support_set_labels = np.array(support_set_labels)
    query_set_images = np.array(query_set_images)
    query_set_labels = np.array(query_set_labels)

    # Convert support set and query set labels to one-hot encoding
    support_set_labels = tf.one_hot(support_set_labels, depth=num_classes)
    query_set_labels = tf.one_hot(query_set_labels, depth=num_classes)

    # Train the model
    model.fit(support_set_images, support_set_labels, epochs=5, batch_size=32, verbose=1)

    # Evaluate on the query set
    loss, accuracy = model.evaluate(query_set_images, query_set_labels, verbose=0)
    print(f'Few-shot learning accuracy: {accuracy}')

    return model, accuracy

def best_model(train_dataset, test_dataset, num_classes, num_shots):
    # Reset best accuracy and best model
    best_accuracy = 0
    best_model = None
    best_num_shot = 0

    # Find best model based on number of shots
    for num_shot in num_shots:
        model, accuracy = few_shot_learning(train_dataset, test_dataset, num_classes, num_shot)
        if accuracy > best_accuracy:
            best_num_shot = num_shot
            best_accuracy = accuracy
            best_model = model

    # Report best model accuracy
    print(f'Best model accuracy: {best_accuracy}')

    return best_model

def main():
    # Load dataset
    train_dataset, test_dataset, num_classes = load_dataset(data_dir)

    # Perform few-shot learning
    num_shots = [1, 2, 3, 4, 5]
    model = best_model(train_dataset, test_dataset, num_classes, num_shots)

    # Save the best model
    model.save('few_shot_model.keras')

if __name__ == '__main__':
    main()
