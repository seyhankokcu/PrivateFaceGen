import tensorflow as tf
from keras import layers

latent_dim = 100

def build_generator():
    model = tf.keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(8 * 8 * 256, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, 4, 2, 'same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(64, 4, 2, 'same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(32, 4, 2, 'same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(3, 4, 2, 'same', activation='tanh', dtype='float32')
    ])
    return model
