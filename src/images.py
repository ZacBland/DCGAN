import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def create_avi_from_dir(dir):
    frames = [os.path.join(dir,image) for image in os.listdir(dir)]

    frame = cv2.imread(frames[0])

    height, width, layers = frame.shape

    video = cv2.VideoWriter("training.avi", 0, 24, (width, height))

    progbar = tf.keras.utils.Progbar(len(frames)-1)

    for i, frame in enumerate(frames[1:]):
        video.write(cv2.imread(frame))
        progbar.update(i)

    cv2.destroyAllWindows()
    video.release()
    
    
import imageio
from PIL import Image
def create_gif_from_dir(dir):
    
    dirs = [os.path.join(dir,image) for image in os.listdir(dir)]
    frames = []

    progbar = tf.keras.utils.Progbar(len(dirs))
    for i, file_path in enumerate(dirs):
        img = imageio.imread(file_path)
        img = Image.fromarray(img).resize((480,480))
        frames.append(img)
        progbar.update(i)

    imageio.mimsave('./test.gif', ims=frames, duration=1)
    
    
    
create_gif_from_dir("./images/")