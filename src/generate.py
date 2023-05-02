import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt

from zipfile import ZipFile

with ZipFile('./models.zip', 'r') as zf:
    zf.extract('models/genenerator_monet_4750.h5')
    generator = tf.keras.models.load_model('models/genenerator_monet_4750.h5', compile=False)
    shutil.rmtree('models/')

LATENT_DIM = 100
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