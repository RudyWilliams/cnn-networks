"""
    This module loads the data.
    Provide a clean way to access the raw data (for showing humans).
    Provide a way to access manipulated shapes (for showing tf).
"""
import tensorflow as tf


data = tf.keras.datasets.mnist.load_data()
train, test = data
image_train, y_train = train
image_test, y_test = test
#make each value be from 0 to 1
image_train = image_train / 255
image_test = image_test / 255
n_classes = 10
width, height = image_train.shape[1::]

image_tr_rshp = image_train.reshape(-1, width, height, 1)
image_te_rshp = image_test.reshape(-1, width, height, 1)
y_tr_hot = tf.keras.utils.to_categorical(y_train, 10)
y_te_hot = tf.keras.utils.to_categorical(y_test, 10)

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    print(f'Running TensorFlow v {tf.__version__}')
    print(f'Training data: {image_train.shape[1::]}')
    print(f'Training labels: {y_train.shape}')
    print(f'Test data: {image_test.shape}')
    print(f'Test labels: {y_test.shape}')
    ## 60,000 training images of shape (28, 28, 1) #bw images
    ## 10,000 test images

    #plot an image to show the datas form
    #plt.figure(figsize=(1,2))
    plt.imshow(image_train[0], cmap='binary')
    plt.colorbar()
    plt.grid(False)
    plt.show()

    
