import tensorflow as tf
from models.generator import build_generator
from models.discriminator import build_discriminator
from utils.dataset import create_dataset
from utils.plot import plot_generated_images
from utils.losses import generator_loss, discriminator_loss

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

latent_dim = 100
batch_size = 32

generator = build_generator()
discriminator = build_discriminator()

gen_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
disc_opt = tf.keras.optimizers.Adam(4e-4, beta_1=0.5)

dataset = create_dataset('data')

@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, latent_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(fake_output, real_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def train(epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for image_batch in dataset:
            train_step(image_batch)
        if (epoch + 1) % 5 == 0:
            plot_generated_images(generator, latent_dim, 5, epoch + 1)
    generator.save('models/generator_optimized.keras')
    discriminator.save('models/discriminator_optimized.keras')

if __name__ == '__main__':
    train(50)
