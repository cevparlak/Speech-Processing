'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np # Matrix and vector computation package
import scipy.io as sio

batch = 2
nclasses = 11
epoch = 100

x_train1 = sio.loadmat('ti_train_raw_wave_batch_2D_batch.mat')
y_train1 = sio.loadmat('ti_train_raw_wave_batchlabels.mat')
data1 = x_train1.get('data') 
label1 = y_train1.get('batchlabels') 
x_train = np.array(data1) # For converting to numpy array
y_train = np.array(label1)-1 # For converting to numpy array

x_test1 = sio.loadmat('ti_test_raw_wave_batch_2D_batch.mat')
y_test1 = sio.loadmat('ti_test_raw_wave_batchlabels.mat')
data1 = x_test1.get('data') 
label1 = y_test1.get('batchlabels') 
x_test = np.array(data1) # For converting to numpy array
y_test = np.array(label1)-1 # For converting to numpy array

# input image dimensions
img_rows, img_cols = 30, 500
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, nclasses)
y_test = keras.utils.to_categorical(y_test, nclasses)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nclasses, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
##history=model.fit(X_train, y_train, epochs=nepoch,  validation_split = 0.1, batch_size = batch, verbose = 1, shuffle = 1)
##history=model.fit(img, lab, batch_size = batch,epochs=epoch,verbose=1)
history=model.fit(x_train, y_train,
          batch_size=batch,
          epochs=epoch,
          verbose=1,
          validation_data=(x_test, y_test))
    
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])