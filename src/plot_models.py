from src import build_discriminator, build_generator
import tensorflow as tf

LATENT_DIM = 100
gen_model = build_generator(latent_dim=LATENT_DIM)
tf.keras.utils.plot_model(gen_model, to_file='./generator_plot.png', show_shapes=True)

dis_model = build_discriminator()
tf.keras.utils.plot_model(dis_model, to_file='./discriminator_plot.png', show_shapes=True)