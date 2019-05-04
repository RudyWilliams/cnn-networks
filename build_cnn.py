import os
import tensorflow as tf
from tensorflow import keras
import digits_data as dd


#hyperparams/initializations 
batch_size = 100
epochs = 12
input_shape = (dd.width, dd.height, 1)
loss = keras.losses.categorical_crossentropy
optimizer = keras.optimizers.Adadelta()
metrics = ['accuracy']


model = keras.Sequential()

c1 = keras.layers.Conv2D(filters=6,
                   kernel_size=(5,5),
                   padding='same',
                   activation='relu',
                   input_shape=input_shape)
#input shape is only needed for first layer
mp1 = keras.layers.MaxPool2D(pool_size=(2,2)) #reduces to 14x14
## show what these images look like after this layer
model.add(c1)
model.add(mp1)

c2 = keras.layers.Conv2D(filters=16,
                   kernel_size=(5,5),
                   padding='valid', #reduces to 10x10
                   activation='relu')
mp2 = keras.layers.MaxPool2D(pool_size=(2,2)) #reduces to 5x5
model.add(c2)
model.add(mp2)

model.add(keras.layers.Flatten()) #shape of 5x5 x16 feature maps = 400
fc1 = keras.layers.Dense(120, activation='relu')
fc2 = keras.layers.Dense(84, activation='relu')
output_layer = keras.layers.Dense(dd.n_classes, activation='softmax')
model.add(fc1)
model.add(fc2)
model.add(output_layer)

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

model.fit(dd.image_tr_rshp, dd.y_tr_hot,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(dd.image_te_rshp, dd.y_te_hot))

score = model.evaluate(dd.image_te_rshp, dd.y_te_hot, verbose=0)

#save model
def save_model_at(ROOTPATH):
    keras.models.save_model(model, os.path.join(ROOTPATH, 'lenet_5.h5py'))

save_model_at('C:\\Users\\rudyw\\cnn-networks\\')


if __name__ == '__main__':
    print(f'Test loss: {score[0]}')
    print(f'Test accuracy: {score[1]}')