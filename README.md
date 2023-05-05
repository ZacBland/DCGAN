# Deep Convolutional GAN (DCGAN) for Monet Art

This repository contains an implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) for generating Monet-style art. The DCGAN is a type of generative model that uses deep convolutional neural networks to learn the features of a given dataset and generate new samples from that dataset. In this case, we use the DCGAN to generate images that resemble the style of the famous painter Claude Monet.

## Dataset

The dataset used to train the DCGAN contains 300 256x256px images of Monet's paintings. These were gathered through the ["I'm Something of a Painter Myself" Kaggle competition](https://www.kaggle.com/competitions/gan-getting-started/data).

## Dependencies

This implementation of the DCGAN was developed using Python 3.9.12 and the following libraries:
- tensorflow==2.12.0
- imageio==2.28.0
- matplotlib==3.7.1
- numpy==1.23.5
- opencv_python==4.7.0.72
- Pillow==9.5.0

To install these dependencies, please run the following command:
```
pip install -r requirements.txt
```

## Usage

### Train

To train the DCGAN, first download the Monet Dataset from the previously mentioned Kaggle Competition and extract it to a directory named `dataset/`. The allowed image types are `.jpg`, `.png`, and `.jpeg`. Then, run the following command: 
```
python main.py
```

The `main.py` script will train the DCGAN based on the dataset given. This train function will save a 4x4 plot of generated images every 10 epochs to a directory called `images/` This script can also train images other than Monets, however be sure to either have 256x256 images or change the architecture's input to take in the desired image size.

### Generate

There is a function within the `src/` folder called generate. Before running this function, please download the model.zip from this [link](https://drive.google.com/file/d/1tMgNib8BymXr5MGEqTKMfLrfDqobD0Vd/view?usp=sharing) and move model.zip into the root folder of the directory.

There are two functions within `generate.py`, `generate_plot()` and `generate_imgs()`. `generate_plot()` will open a matplotlib plot of a 4x4 grid of newly generated images. `generate_imgs()` will save a specified number of newly generated images to a given folder.

To run these functions, either add one of the following commands to the bottom of the file or run from a seperate file:
```
generate_plot()
generate_imgs(dir="folder/of/your/choosing", num_of_imgs=7000)
```

## Results

The following is a gif of the training throughout the 5000 epochs for this current model:
![Training Gif](https://github.com/ZacBland/DCGAN/blob/main/data/training.gif)

As can be seen, the DCGAN is able to produce fairly high-quality images that resemble the style of Monet's paintings. However, there is still room for improvement, and thus further training and testing to the current architecture may produce better results.

