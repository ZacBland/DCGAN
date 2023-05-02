

import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, Dropout, Dense
from tensorflow.python.keras import Model, Sequential



def build_generator(latent_dim=128) -> Sequential:
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Dense(32*32*256, use_bias=False, input_shape=(latent_dim, )))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    
    model.add(tf.keras.layers.Reshape((32,32,256)))
    assert model.output_shape == (None, 32, 32, 256)
    
    model.add(tf.keras.layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    
    model.add(tf.keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    
    model.add(tf.keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    
    model.add(tf.keras.layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, 256, 256, 3)
    
    return model

def build_discriminator() -> Sequential:
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2D(256, (5,5), strides=(5,5), padding='same', input_shape=[256,256,3]))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(512, (5,5), strides=(2,2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    
    return model

@tf.function
def train_step(batch_size, noise_dim, images,generator, discriminator, cross_entropy, generator_optimizer, discriminator_optimizer) -> None:
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output,cross_entropy)
        disc_loss = discriminator_loss(real_output, fake_output,cross_entropy)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def discriminator_loss(real, fake, cross_entropy):
    real_loss = cross_entropy(tf.ones_like(real), real)
    fake_loss = cross_entropy(tf.zeros_like(fake), fake)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake, cross_entropy):
    return cross_entropy(tf.ones_like(fake), fake)

from PIL import Image
import matplotlib.pyplot as plt
def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    #img = Image.fromarray((predictions[0, :, :, :]*255).numpy().astype('uint8'))
    #img.save("gen_image_single/image_at_epoch_{:04d}.png".format(epoch))
       
    fig = plt.figure(figsize=(4, 4), facecolor="black")

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i, :, :, :]*255).numpy().astype('uint8'))
        plt.axis('off')

    plt.savefig('gen_images/image_at_epoch_{:04d}.png'.format(epoch), dpi=1200)
    #  plt.show()
    plt.close(fig)
    

import PIL
def display_image(epoch_no):
    return PIL.Image.open('gen_images/image_at_epoch_{:04d}.png'.format(epoch_no))