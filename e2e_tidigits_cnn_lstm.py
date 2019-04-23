'''
Basic demonstration of the capabilities of the CRNN using TimeDistributed
wrapper. Processes an MNIST image (or blank square) at each time step and
sums the digits. Learning is based on the sum of the digits, not explicit
labels on each digit.
'''
from __future__ import print_function
import numpy as np

import keras.backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.utils import np_utils
# for reproducibility
np.random.seed(2016)
K.set_image_dim_ordering('tf')

# define some run parameters
batch_size = 10
nb_epochs = 1
maxToAdd = 1
hidden_units = 100
size1 = 30
size2 = 500
nclasses=11;
# the data, shuffled and split between train and test sets
#(X_train_raw, y_train_temp), (X_test_raw, y_test_temp) = mnist.load_data()

import scipy.io as sio

x_train1 = sio.loadmat('ti_train_raw_wave_batch_2D_batch.mat')   # 2068x30x500
y_train1 = sio.loadmat('ti_train_raw_wave_batchlabels.mat')
data1 = x_train1.get('data') 
label1 = y_train1.get('batchlabels') 
x_train = np.array(data1) # For converting to numpy array
y_train = np.array(label1)-1 # For converting to numpy array

x_test1 = sio.loadmat('ti_test_raw_wave_batch_2D_batch.mat')   # 2468x30x500
y_test1 = sio.loadmat('ti_test_raw_wave_batchlabels.mat')
data1 = x_test1.get('data') 
label1 = y_test1.get('batchlabels') 
x_test = np.array(data1) # For converting to numpy array
y_test = np.array(label1)-1 # For converting to numpy array

nsamples1=2068
nsamples2=2486
x_train=x_train[0:nsamples1,:]
x_test=x_test[0:nsamples2,:]

y_train=y_train[0:nsamples1,:]
y_test=y_test[0:nsamples2,:]

X_train_raw=x_train
X_test_raw=x_test
y_train_temp=y_train
y_test_temp=y_test
# basic image processing
#X_train_raw = X_train_raw.astype('float32')
#X_test_raw = X_test_raw.astype('float32')
#X_train_raw /= 255
#X_test_raw /= 255

print('X_train_raw shape:', X_train_raw.shape)
print(X_train_raw.shape[0], 'train samples')
print(X_test_raw.shape[0], 'test samples')
print("Building model")

# define our time-distributed setup
inp = Input(shape=(maxToAdd, size1, size2, 1))
x = TimeDistributed(Conv2D(32, (8, 8), padding='valid', activation='relu'))(inp)
x = TimeDistributed(Conv2D(64, (8, 8), padding='valid', activation='relu'))(x)
x = TimeDistributed(Flatten())(x)
x = LSTM(units=100, return_sequences=True)(x)
x = LSTM(units=50, return_sequences=False)(x)
x = Dropout(.2)(x)
x = Dense(nclasses)(x)
model = Model(inp, x)
model.summary()
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta(),  metrics = ['accuracy'])
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adam(),
#              metrics=['accuracy'])

# run epochs of sampling data then training
for ep in range(0, nb_epochs):
    X_train = []
    y_train = []
    X_test  = []
    y_test  = []

    examplesPer = nsamples1
    X_train = np.zeros((examplesPer, maxToAdd, size1, size2, 1))
    for i in range(0, examplesPer):
        # initialize a training example of max_num_time_steps,im_size,im_size
        output = np.zeros((maxToAdd, size1, size2, 1))
        # decide how many MNIST images to put in that tensor
#        numToAdd = int(np.ceil(np.random.rand()*maxToAdd))
#        print(numToAdd)
        # sample that many images
        indices = i;# np.random.choice(X_train_raw.shape[0], size=numToAdd)
        example = X_train_raw[indices]
        # sum up the outputs for new output
        exampleY = y_train_temp[indices]
        output[0, :, :, 0] = example
        X_train[i, :, :, :, :] = output
        y_train.append((exampleY))
    X_train = np.array(X_train)
    y_train = np.array(y_train)
          
    examplesPer = nsamples2
    X_test = np.zeros((examplesPer, maxToAdd, size1, size2, 1))
    for i in range(0, examplesPer):
        output = np.zeros((maxToAdd, size1, size2, 1))
    #    numToAdd = int(np.ceil(np.random.rand()*maxToAdd))
        indices = i;#np.random.choice(X_test_raw.shape[0], size=numToAdd)
        example = X_test_raw[indices]
        exampleY = y_test_temp[indices]
        output[0, :, :, 0] = example
        X_test[i, :, :, :, :] = output
        y_test.append((exampleY))
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)        
         
    y_train = np_utils.to_categorical(y_train, nclasses)
    y_test = np_utils.to_categorical(y_test, nclasses)
    if ep == 0:
        print("X_train shape: ", X_train.shape)
        print("y_train shape: ", y_train.shape)
        print("X_test shape: ", X_test.shape)
        print("y_test shape: ", y_test.shape)        

#    model.fit(X_train, y_train, batch_size=batch_size, epochs=10, verbose=1)
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=100,
              verbose=1,
              validation_data=(X_test, y_test))
# Test the model
preds = model.predict(X_test)
# print the results of the test
print(np.sum(np.sqrt(np.mean([ (y_test[i] - preds[i][0])**2 for i in range(0,len(preds)) ]))))
print("naive guess", np.sum(np.sqrt(np.mean([ (y_test[i] - np.mean(y_test))**2 for i in range(0,len(y_test)) ]))))

# save the model
#jsonstring  = model.to_json()
#with open("../models/basicRNN.json",'wb') as f:
#    f.write(jsonstring)
#model.save_weights("../models/basicRNN.h5",overwrite=True)