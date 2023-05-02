from src import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))


from PIL import Image
import os, os.path


#Dataset preprocessing
imgs = []
DATASET_PATH = "./dataset/"
VALID_TYPES = [".jpg", ".png", ".jpeg"]

progbar = tf.keras.utils.Progbar(len(os.listdir(DATASET_PATH)))
i = 0
for file in os.listdir(DATASET_PATH):
    #get file extension
    ext = os.path.splitext(file)[1]
    
    if ext.lower() not in VALID_TYPES:
        continue
    
    img = np.array(Image.open(os.path.join(DATASET_PATH, file)).resize((256,256)))

    if img.shape == (256,256,3):
        temp = np.asarray(Image.open(os.path.join(DATASET_PATH, file)))
        imgs.append(temp[:256,:256])
    i+=1
    progbar.update(i)
    
tensor = np.array(imgs)
#Normalize images to [0, 1]
tensor = (tensor) / 255

print(f"Size of tensor is {len(tensor)}")
print(tensor[5].shape)

print(tensor[0].shape)
imgplot = plt.imshow(tensor[0])
#plt.show()

BATCH_SIZE = 16
BUFFER_SIZE = 10000
LATENT_DIM = 100

#create dataset
train_dataset = tf.data.Dataset.from_tensor_slices(tensors=tensor).shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)

#build models
gen_model = build_generator(latent_dim=LATENT_DIM)
gen_model.summary()
noise = tf.random.normal([1, LATENT_DIM])
generated_img = gen_model(noise, training=False)
#plt.imshow(generated_img[0, :, :, 0])
#plt.show()

dis_model = build_discriminator()
dis_model.summary()

decision = dis_model(generated_img)
print(decision)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

gen_opt = tf.keras.optimizers.Adam(1e-4)
dis_opt = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    gen_opt=gen_opt,
    dis_opt=dis_opt,
    gen_model=gen_model,
    dis_model=dis_model
)

print("\n\n\nCHECKING CHECKPOINT!")
latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
checkpoint.restore(latest)

print("\n\n\n")

EPOCHS=5000
IMG_SAVE_INTERVAL = 10
CHKP_SAVE_INTERVAL = 250
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, LATENT_DIM])

import time
for epoch in range(EPOCHS):
    start = time.time()
    
    progbar = tf.keras.utils.Progbar(len(train_dataset)-1)
    for i, image_batch in enumerate(train_dataset):
        gen_loss, disc_loss = train_step(BATCH_SIZE, LATENT_DIM, image_batch, gen_model, dis_model, cross_entropy, gen_opt, dis_opt)
        progbar.update(i)
        
    
    if(epoch) % IMG_SAVE_INTERVAL == 0:
        generate_and_save_images(gen_model, epoch+1, seed)
        
    if(epoch) % CHKP_SAVE_INTERVAL == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
        gen_model.save("./training_checkpoints/genenerator_{}_{:.4f}.h5".format("monet",epoch))
        dis_model.save("./training_checkpoints/discriminator_{}_{:.4f}.h5".format("monet",epoch))
    
    print('\nTime for epoch {} is {:.5f} sec'.format(epoch, time.time()-start))
    print('Dis Loss: {:.5f}\nGen Loss: {:.5f}'.format(float(disc_loss), float(gen_loss)))
    print()


from IPython import display
display.clear_output(wait=False)
generate_and_save_images(gen_model,
                         EPOCHS,
                         seed)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
display_image(EPOCHS)


