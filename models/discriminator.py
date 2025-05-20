import tensorflow as tf
from keras import layers

SIZE = 128

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape=(SIZE, SIZE, 3)),
        layers.Conv2D(64, 4, 2, 'same', use_bias=False, kernel_initializer='he_normal'),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, 4, 2, 'same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2D(256, 4, 2, 'same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2D(512, 4, 2, 'same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    return model
