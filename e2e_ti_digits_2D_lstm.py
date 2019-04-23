import numpy as np # Matrix and vector computation package
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.utils import np_utils

import scipy.io as sio

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

batch = 2
epoch=100
hidden_units = 50
classes = 11

img=x_train
lab=y_train
img_test=x_test
lab_test=y_test
lab = np_utils.to_categorical(lab, classes)
lab_test = np_utils.to_categorical(lab_test, classes)
print(img.shape[1:])
model = Sequential()
model.add(LSTM(hidden_units,input_shape =img.shape[1:], batch_size = batch))
model.add(Dense(classes))
model.add(Activation('softmax'))

model.compile(optimizer = 'Adam', loss='mean_squared_error', metrics = ['accuracy'])
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])
#model.fit(img, lab, batch_size = batch,epochs=epoch,verbose=1)
model.fit(img, lab,
          batch_size=batch,
          epochs=epoch,
          verbose=1,
          validation_data=(img_test, lab_test))

scores = model.evaluate(img_test, lab_test, batch_size=batch)

predictions = model.predict(img_test, batch_size = batch)

print('LSTM test score:', scores[0])
print('LSTM test accuracy:', scores[1])