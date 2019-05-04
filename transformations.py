import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K 
from digits_data import image_train, image_tr_rshp


class CreateRow:
    """ Class to visualize the transformations that take place due to C1.

        Args:
            images: np.array - the data containing the original images
                *shape=(n_images, height_of_image, width_of_image)
            kernels: list - the list of kernels
            feature_maps: list - the list of feature maps

        Returns:
            instance of a 6x3 subplot

        Methods:
            new_row:
                Args:
                    row_num: int - valid values are 0-5
                    im_loc: int - the image to show (keep consistent)
                    ker_fmap_loc: int - the kernel and corresponding f_map
                                 to plot

                Returns:
                    updated instance of the 6x3 subplot
    """

    def __init__(self, model, images, im_loc):
        self.fig, self.axs = plt.subplots(6, 3)
        self.images = images
        self.im_loc = im_loc

        #get weights of c1 (i.e. get the kernels)
        self.c1 = model.layers[0]
        self.c1_Wb = self.c1.get_weights()
        self.c1_W = self.c1_Wb[0]
        self.kernels = [self.c1_W[:, :, 0, i] for i in range(self.c1_W.shape[3])]

        #get the corresponding maps
        X = image_tr_rshp[self.im_loc] #same figure, dif maps
        X = np.expand_dims(X, axis=0)
        self.c1_output = K.function([self.c1.input], [self.c1.output])
        self.feature_maps = [self.c1_output(X)[0][:, :, :, i][0] for i in range(self.c1_W.shape[3])]


    def new_row(self,  row_num, ker_fmap_loc):
        imc0 = self.axs[row_num, 0].imshow(self.images[self.im_loc], cmap='binary')
        self.fig.colorbar(imc0, ax=self.axs[row_num, 0])
        imc1 = self.axs[row_num, 1].imshow(self.kernels[ker_fmap_loc], cmap='binary') #don't want to norm this one bc the neg values are valueable
        self.fig.colorbar(imc1, ax=self.axs[row_num, 1])
        imc2 = self.axs[row_num, 2].imshow(self.feature_maps[ker_fmap_loc], cmap='binary', vmin=0, vmax=1)
        self.fig.colorbar(imc2, ax=self.axs[row_num, 2])


if __name__ == '__main__':

    lenet5 = tf.keras.models.load_model('lenet_5.h5py')

    grid = CreateRow(lenet5, image_train, im_loc=0)
    grid.new_row(row_num=0, ker_fmap_loc=0)
    grid.new_row(row_num=1, ker_fmap_loc=1)
    grid.new_row(row_num=2, ker_fmap_loc=2)
    grid.new_row(row_num=3, ker_fmap_loc=3)
    grid.new_row(row_num=4, ker_fmap_loc=4)
    grid.new_row(row_num=5, ker_fmap_loc=5)

    plt.show()
