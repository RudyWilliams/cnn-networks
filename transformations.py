import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K 
from digits_data import image_train, image_tr_rshp

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
    
    fig, axs = plt.subplots(6, 3)

    im00 = axs[0, 0].imshow(image_train[0], cmap='binary')
    fig.colorbar(im00, ax=axs[0,0])
    im01 = axs[0, 1].imshow(kernels[0], cmap='binary') #don't want to norm this one bc the neg values are valueable
    fig.colorbar(im01, ax=axs[0,1])
    im02 = axs[0, 2].imshow(fmaps[0], cmap='binary', vmin=0, vmax=1)
    fig.colorbar(im02, ax=axs[0,2])

    im10 = axs[1, 0].imshow(image_train[0], cmap='binary')
    fig.colorbar(im10, ax=axs[1,0])
    im11 = axs[1, 1].imshow(kernels[1], cmap='binary') #don't want to norm this one bc the neg values are valueable
    fig.colorbar(im11, ax=axs[1,1])
    im12 = axs[1, 2].imshow(fmaps[1], cmap='binary', vmin=0, vmax=1)
    fig.colorbar(im12, ax=axs[1,2])

    im20 = axs[2, 0].imshow(image_train[0], cmap='binary')
    fig.colorbar(im20, ax=axs[2,0])
    im21 = axs[2, 1].imshow(kernels[2], cmap='binary') #don't want to norm this one bc the neg values are valueable
    fig.colorbar(im21, ax=axs[2,1])
    im22 = axs[2, 2].imshow(fmaps[2], cmap='binary', vmin=0, vmax=1)
    fig.colorbar(im22, ax=axs[2,2])

    im30 = axs[3, 0].imshow(image_train[0], cmap='binary')
    fig.colorbar(im30, ax=axs[3,0])
    im31 = axs[3, 1].imshow(kernels[3], cmap='binary') #don't want to norm this one bc the neg values are valueable
    fig.colorbar(im31, ax=axs[3,1])
    im32 = axs[3, 2].imshow(fmaps[3], cmap='binary', vmin=0, vmax=1)
    fig.colorbar(im32, ax=axs[3,2])

    im40 = axs[4, 0].imshow(image_train[0], cmap='binary')
    fig.colorbar(im40, ax=axs[4,0])
    im41 = axs[4, 1].imshow(kernels[4], cmap='binary') #don't want to norm this one bc the neg values are valueable
    fig.colorbar(im41, ax=axs[4,1])
    im42 = axs[4, 2].imshow(fmaps[4], cmap='binary', vmin=0, vmax=1)
    fig.colorbar(im42, ax=axs[4,2])

    im50 = axs[5, 0].imshow(image_train[0], cmap='binary')
    fig.colorbar(im50, ax=axs[5,0])
    im51 = axs[5, 1].imshow(kernels[5], cmap='binary') #don't want to norm this one bc the neg values are valueable
    fig.colorbar(im51, ax=axs[5,1])
    im52 = axs[5, 2].imshow(fmaps[5], cmap='binary', vmin=0, vmax=1)
    fig.colorbar(im52, ax=axs[5,2])




    plt.show()
    
    