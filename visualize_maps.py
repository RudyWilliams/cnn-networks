import tensorflow as tf
from tensorflow import keras

lenet5 = tf.keras.models.load_model('lenet_5.h5py')

c1_W_b =  lenet5.layers[0].get_weights()
print(len(c1_W_b))
for _ in c1_W_b:
    print('----------------')
    print(_.shape)
print('===============')
c1W = c1_W_b[0]

kernel_one = c1W[:,:,0,0]
#c1_W_b has the weight matrix for c1 layer along with bias terms
#[ [ [[0 1 2 3 4 5]]    #value for each map
#    [[0 1 2 3 4 5]]
#    ...                #5 of these (dim of kernel)
#    [[0 1 2 3 4 5]] ] 
#  ...                  #5 of these (other dim of kernel)
#  [ [[...        ]]
#    ...
#    [[...        ]] ] ]
#
#bias looks like
#[0 1 2 3 4 5] #one for each map

from tensorflow.keras import backend as K 
import digits_data as dd 
import numpy as np
import matplotlib.pyplot as plt

# plt.imshow(kernel_one, cmap='binary', vmin=0, vmax=1)
# plt.colorbar()
# plt.grid(False)
# plt.show()


c1_output = K.function([lenet5.layers[0].input],
                       [lenet5.layers[0].output])

#print(c1_output(dd.image_te_rshp)[0].shape)
#all training images are thrown through so the shape
#is (10000, 28, 28, 6)

X = dd.image_tr_rshp[0]
X = np.expand_dims(X, axis=0)
fm_0 = c1_output([X])[0][:,:,:,0] #the last index controls what feature map it is
#the first feature map shows the 'underbelly' of the numbers
fm_0 = fm_0[0]
# plt.imshow(fm_0, cmap='binary')
# plt.colorbar()
# plt.grid(False)
# plt.show()

