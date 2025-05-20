def perceptual_loss(input_img, generated_img):
    return tf.reduce_mean(tf.square(input_img - generated_img))
