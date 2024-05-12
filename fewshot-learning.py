import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable OneDNN
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import image_dataset_from_directory
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# Example usage
# Assuming you have images and labels loaded as numpy arrays
# images.shape = (num_samples, height, width, num_channels)
# labels.shape = (num_samples,)
# num_classes = total number of classes
# num_shot = number of examples per class for support set

# Define parameters for dataset builder
# batch_size = len(os.listdir('data/canetoad')) # length of dataset
img_height = 256
img_width = 256
data_dir = 'data'

def load_dataset(data_dir, test_split=0.2, image_size=(224, 224), batch_size=32):
    train_dataset = image_dataset_from_directory(
        data_dir,
        validation_split=test_split,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )

    test_dataset = image_dataset_from_directory(
        data_dir,
        validation_split=test_split,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)

    return train_dataset, test_dataset, num_classes

def create_few_shot_model(input_shape, num_classes):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

def few_shot_learning(train_dataset, test_dataset, num_classes, num_shot):
    input_shape = train_dataset.element_spec[0].shape[1:]  # Get the input shape from the dataset
    model = create_few_shot_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Prepare support set and query set from train_dataset
    support_set_images = []
    support_set_labels = []
    query_set_images = []
    query_set_labels = []

    for images, labels in train_dataset:
        for class_index in range(num_classes):
            class_indices = tf.where(tf.equal(labels, class_index))[:, 0]
            support_indices = tf.random.shuffle(class_indices)[:num_shot]
            query_indices = tf.sets.difference([class_indices], [support_indices])

            support_set_images.extend(images[support_indices])
            support_set_labels.extend(labels[support_indices])
            query_set_images.extend(images[query_indices])
            query_set_labels.extend(labels[query_indices])

    support_set_images = np.array(support_set_images)
    support_set_labels = np.array(support_set_labels)
    query_set_images = np.array(query_set_images)
    query_set_labels = np.array(query_set_labels)

    # Convert support set labels to one-hot encoding
    support_set_labels = tf.one_hot(support_set_labels, depth=num_classes)

    # Train the model
    model.fit(support_set_images, support_set_labels, epochs=5, batch_size=32, verbose=1)

    # Evaluate on the query set
    loss, accuracy = model.evaluate(query_set_images, query_set_labels, verbose=0)
    print(f'Few-shot learning accuracy: {accuracy}')

def main():
    # Load dataset
    train_dataset, test_dataset, num_classes = load_dataset(data_dir)

    # Perform few-shot learning
    num_shot = 2
    few_shot_learning(train_dataset, test_dataset, num_classes, num_shot)

if __name__ == '__main__':
    main()
