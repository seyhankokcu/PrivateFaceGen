import matplotlib.pyplot as plt
import tensorflow as tf
import os

def plot_generated_images(generator, latent_dim, square=5, epoch=0, save_dir='outputs', display=False):
    os.makedirs(save_dir, exist_ok=True)
    noise = tf.random.normal([square * square, latent_dim])
    generated_images = generator(noise, training=False)
    generated_images = (generated_images + 1) / 2.0

    plt.figure(figsize=(10, 10))
    for i in range(square * square):
        plt.subplot(square, square, i + 1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.suptitle(f'Epoch {epoch}', fontsize=16)

    if epoch > 0:
        plt.savefig(os.path.join(save_dir, f'epoch_{epoch}.png'))
    if display:
        plt.show()
    else:
        plt.close()
