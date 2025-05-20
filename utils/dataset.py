import tensorflow as tf
import os
import re

SIZE = 128
batch_size = 32

def load_and_preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [SIZE, SIZE])
    img = (img - 127.5) / 127.5
    return img

def create_dataset(dataset_path):
    files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    files = sorted(files, key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', x)])
    files = [f for f in files if 'seed9090.png' not in f]
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
    return dataset
