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

    def __init__(self, images, kernels, feature_maps):
        self.fig, self.axs = plt.subplots(6, 3)
        self.images = images
        self.kernels = kernels
        self.feature_maps = feature_maps

    def new_row(self,  row_num, im_loc, ker_fmap_loc):
        imc0 = self.axs[row_num, 0].imshow(self.images[im_loc], cmap='binary')
        self.fig.colorbar(imc0, ax=self.axs[row_num, 0])
        imc1 = self.axs[row_num, 1].imshow(self.kernels[ker_fmap_loc], cmap='binary') #don't want to norm this one bc the neg values are valueable
        self.fig.colorbar(imc1, ax=self.axs[row_num, 1])
        imc2 = self.axs[row_num, 2].imshow(self.feature_maps[ker_fmap_loc], cmap='binary', vmin=0, vmax=1)
        self.fig.colorbar(imc2, ax=self.axs[row_num, 2])


lenet5 = tf.keras.models.load_model('lenet_5.h5py')

#get weights of c1 (i.e. get the kernels)
c1 = lenet5.layers[0]
c1_Wb = c1.get_weights()
c1_W = c1_Wb[0]
kernels = [c1_W[:, :, 0, i] for i in range(c1_W.shape[3])]

#get the corresponding maps
X = image_tr_rshp[0] #same figure, dif maps
X = np.expand_dims(X, axis=0)
c1_output = K.function([c1.input],
                       [c1.output])

fmaps = [c1_output(X)[0][:, :, :, i][0] for i in range(c1_W.shape[3])]

if __name__ == '__main__':
    
    print(image_train.shape)
    grid = CreateRow(image_train, kernels, fmaps)
    grid.new_row(row_num=0, im_loc=0, ker_fmap_loc=0)
    grid.new_row(row_num=1, im_loc=0, ker_fmap_loc=1)
    grid.new_row(row_num=2, im_loc=0, ker_fmap_loc=2)
    grid.new_row(row_num=3, im_loc=0, ker_fmap_loc=3)
    grid.new_row(row_num=4, im_loc=0, ker_fmap_loc=4)
    grid.new_row(row_num=5, im_loc=0, ker_fmap_loc=5)

    plt.show()
