import tensorflow as tf
import matplotlib.pyplot as plt
from models.generator import build_generator
from utils.perceptual_loss import perceptual_loss

SIZE = 128
latent_dim = 100

def preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [SIZE, SIZE])
    img = (img - 127.5) / 127.5
    return img

def find_latent_vector(input_image, generator, iterations=2000, learning_rate=0.05):
    latent_vector = tf.Variable(tf.random.normal([1, latent_dim]))
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for step in range(iterations):
        with tf.GradientTape() as tape:
            generated = generator(latent_vector, training=False)
            mse_loss = tf.reduce_mean(tf.square(generated - input_image))
            perc_loss = perceptual_loss(input_image, generated)
            loss = mse_loss + 0.1 * perc_loss
        grads = tape.gradient(loss, latent_vector)
        optimizer.apply_gradients([(grads, latent_vector)])
        if step % 200 == 0:
            print(f"Step {step}: Loss = {loss.numpy():.4f}")
    return latent_vector

def generate_similar_image(path):
    input_image = preprocess_image(path)
    input_image = tf.expand_dims(input_image, 0)
    generator = build_generator()
    generator.load_weights('models/generator_optimized.keras')
    latent_vector = find_latent_vector(input_image, generator)
    similar_image = generator(latent_vector, training=False)
    input_image = (input_image[0] + 1) / 2.0
    similar_image = (similar_image[0] + 1) / 2.0
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title('Input')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(similar_image)
    plt.title('Generated')
    plt.axis('off')
    plt.show()
