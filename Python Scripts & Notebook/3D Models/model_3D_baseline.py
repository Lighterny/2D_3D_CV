def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


import argparse
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import subprocess
import sys

if __name__ == '__main__':
    install('keras_metrics')
    import keras_metrics

    parser = argparse.ArgumentParser()
    # hyperparameters set in os environment

    parser.add_argument('--gpu-count', type=int,
                        default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])

    parser.add_argument('--training', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str,
                        default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()

    epochs = 50
    lr = 0.0001
    batch_size = 1
    # l1 = args.l1
    # wd = args.wd

    gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.training
    validation_dir = args.validation

    # Calling data
    x_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
    y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
    x_val = np.load(os.path.join(validation_dir, 'val.npz'))['image']
    y_val = np.load(os.path.join(validation_dir, 'val.npz'))['label']

    img_width, img_height, img_depth = 128, 128, 64
    # Tensorflow needs image channels last, e.g. (batch size, width, height, channels)
    # Tensorflow needs image channels last, e.g. (batch size, width, height, channels)
    #    if K.image_data_format() == 'channels_last':
    #        x_train = x_train.reshape(x_train.shape[0],
    #                                  img_width,
    #                                  img_height,
    #                                  img_depth)
    #        x_val = x_val.reshape(x_val.shape[0],
    #                              img_width,
    #                              img_height,
    #                              img_depth)
    #        y_train = y_train.reshape(y_train.shape[0], )
    #        y_val = y_val.reshape(y_val.shape[0], )
    #
    #        input_shape = (img_width, img_height, img_depth, 1)
    #        batch_norm_axis = -1
    #    else:
    #
    #        print('Channels first, exiting')
    #        exit(-1)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'test samples')


    def preprocess(volume, label):
        """Process validation data by only adding a channel."""
        volume = tf.expand_dims(volume, axis=3)
        return volume, label


    # Define data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_dataset = (
        train_loader.shuffle(len(x_train))
            .map(preprocess)
            .batch(1)
            .prefetch(1)
    )
    val_dataset = (
        validation_loader.shuffle(len(x_val))
            .map(preprocess)
            .batch(1)
            .prefetch(1)
    )

    print('Training Dataset Shape:', train_dataset)
    print('Validation Dataset Shape:', val_dataset)


    def get_model(width=128, height=128, depth=64):
        """Build a 3D convolutional neural network model."""

        inputs = keras.Input((width, height, depth, 1))

        x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
        x = layers.MaxPool3D(pool_size=2)(x)
        # x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        # x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        # x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        # x = layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(units=512, activation="relu")(x)
        # x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(units=1, activation="sigmoid")(x)

        # Define the model.
        model = keras.Model(inputs, outputs, name="ALight3DCNN2")
        return model


    model = get_model(width=128, height=128, depth=64)

    # Compile model.
    # learning_rate = 0.0001
    # Compile model.
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=['accuracy',
                 keras_metrics.precision(),
                 keras_metrics.f1_score(),
                 keras_metrics.recall(),
                 tf.keras.metrics.AUC()],
    )

    # Define callbacks.
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "Alight3D_CB_CP.h5", save_best_only=True)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                         patience=40)

    # Train the model, doing validation at the end of each epoch
    # epochs = 50
    # batch_size = 5

    # train model
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=50,
                        shuffle=True,
                        verbose=1,
                        callbacks=[checkpoint_cb, early_stopping_cb],
                        )

    # For BatchNormalization layers
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())

    print(model.summary())
    score = model.evaluate(val_dataset, verbose=1)
    print('Validation Loss    :', score[0])
    print('Validation Accuracy:', score[1])
    print('Validation F1 Score:', score[2])
    print('Validation Precision:', score[3])
    print('Validation Recall:', score[4])
    print('Validation AUC:', score[5])

    # save Keras model for Tensorflow Serving
    print('Saving model to:', model_dir)
    model.save(os.path.join(model_dir, '1'))
    model.save(os.path.join(model_dir, 'ALight_3D_CNN_baseline.h5'))
