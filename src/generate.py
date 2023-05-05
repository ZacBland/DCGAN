import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt

from zipfile import ZipFile

LATENT_DIM = 100

def generate_plot():
    """
    Opens a Plot of 4x4 images that are randomly generated from the model. Click anywhere on the plot to generate new images.
    """
    with ZipFile('./models.zip', 'r') as zf:
        zf.extract('models/genenerator_monet_4750.h5')
        generator = tf.keras.models.load_model('models/genenerator_monet_4750.h5', compile=False)
        shutil.rmtree('models/')

    num_examples_to_generate = 16

    class Main(object):
        def clear(self):
            plt.clf()
        def redraw(self):
            self.clear()
            generated_img = generator(self.noise, training=False)
            for i in range(generated_img.shape[0]):
                plt.subplot(4, 4, i+1)
                plt.imshow((generated_img[i, :, :, :]*255).numpy().astype('uint8'))
                plt.axis('off')
            plt.draw()
        def on_click(self, event):
            self.noise = tf.random.normal([num_examples_to_generate, LATENT_DIM])
            print("clicked")
            self.redraw()
        def run(self):
            self.redraw()
            plt.connect('button_press_event', self.on_click)
            plt.show()

        def __init__(self):
            self.noise = tf.random.normal([num_examples_to_generate, LATENT_DIM])

    m=Main()
    m.run()

from PIL import Image

def generate_images(dir, num_of_imgs=100):
    """
    Function that generates new images based on model

    Args:
        dir (String): Directory folder that images should be saved to
        num_of_imgs (Integer): Number of images that would like to be saved
    """
    
    with ZipFile('./models.zip', 'r') as zf:
        zf.extract('models/genenerator_monet_4750.h5')
        generator = tf.keras.models.load_model('models/genenerator_monet_4750.h5', compile=False)
        shutil.rmtree('models/')

    #Creates array of newly generated images
    noise = tf.random.normal([num_of_imgs, LATENT_DIM])
    generated_imgs = generator(noise, training=False)

    if not os.path.exists(dir):
        os.mkdir(dir)
    
    print("Saving Images...")
    progbar = tf.keras.utils.Progbar(len(generated_imgs))
    for i in range(len(generated_imgs)):
        img = Image.fromarray((generated_imgs[i, :, :, :]*255).numpy().astype('uint8'))
        img.save(os.path.join(dir, "new_img_{}.png".format(i)))
        progbar.update(i)

    print(f"\nImages saved to {dir}!")

                 
