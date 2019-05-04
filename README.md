## cnn-networks

This repo contains the code for implemeting a _convolutional neural network_ (CNN). The arcitecture of build.py is based on the 
[LeNet-5](https://engmrk.com/lenet-5-a-classic-cnn-architecture/). The code is mostly based off of  this [keras 
tutorial](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py).

### Modules

#### digits_data.py
Reads in the image data. Provides access to raw data to be fed to plt.imshow() as well as reshaped data to be fed into the CNN input layer. Provides 
both training and test images and labels. Imported in other modules as dd.

#### build_cnn.py
Uses tensorflow.keras to train a CNN  based on the lenet-5 arch. except for differences in the activation functions. Saves model to specified 
ROOTPATH.

#### transformations.py
Shows the input digit, the kernel, and then the feature map created. The output plots can be found in the screenshots folder.
