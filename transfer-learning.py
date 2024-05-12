import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable OneDNN
import tensorflow as tf

# Define parameters for dataloader
batch_size = len(os.listdir('data/canetoad')) # length of dataset
img_height = 256
img_width = 256
data_dir = 'data'

def build_dataset():
    global train_ds, val_ds, data_augmentation

    # Create training dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    seed=123,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

    # Create validation dataset
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    seed=123,
    subset="validation",
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

    # Apply data augmentation to training dataset
    data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"), # Randomly flip images horizontally
    tf.keras.layers.RandomRotation(0.1) # Randomly rotate images by 10%
    ])

def build_model():
    # Create datasets
    build_dataset()

    # Create base model using Xception
    base_model = tf.keras.applications.Xception(
      weights='imagenet',  # Load weights pre-trained on ImageNet.
      input_shape=(img_height, img_width, 3),
      include_top=False)  # Do not include the ImageNet classifier at the top.

    # Freeze the convolutional base
    base_model.trainable = False

    # Create batch
    image_batch, label_batch = next(iter(train_ds))
    feature_batch = base_model(image_batch)

    # Add classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    # Add prediction layer
    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    # Build model
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def train_model(lr=0.00001, epochs=50):
    # Build model
    model = build_model()

    # Compile model
    lr = lr
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train model
    epochs = epochs
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )
    return model

def main():
    model = train_model()
    model.evaluate(val_ds)

if __name__ == '__main__':
    main()