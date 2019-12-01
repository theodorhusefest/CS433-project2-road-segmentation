

# Imports

import os,sys
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True' # Need this to work on Theos mac

from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Concatenate, Input, Dropout

class UNET():

    def __init__(self, x_train, y_train, x_test, y_test, layers = 2, dropout_rate = 0.5):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.IMAGE_SIZE = x_train[0].shape[0]
        self.IMAGE_SHAPE = x_train[0].shape
        self.layers = layers
        self.dropout_rate = dropout_rate
        self.model = None


    def build_model(self):
        print('Building model with {} layers'.format(self.layers))
        filter_sizes = np.append(np.flip([int(self.IMAGE_SIZE/2**(i)) for i in range(self.layers)]), self.IMAGE_SIZE*2)

        print('Filtersizes being used in UNET: {}'.format(filter_sizes))

        inputs = Input(self.IMAGE_SHAPE)
        pool = inputs

        convs = []

        # Contracting 
        for i in range(self.layers):

            print('Bulding contraction layers at layer: {} and filtersize: {}'.format(i+1, filter_sizes[i]))
            if i == self.layers - 1: # If this is last iteration
                conv, pool = self.contract(pool, filter_sizes[i], dropout= True, dropout_rate= self.dropout_rate)
            else:
                conv, pool = self.contract(pool, filter_sizes[i], dropout= False)
            
            # Save convolution for expanding phase
            convs.append(conv)

        # Bottleneck
        print('Building bottleneck at layer: {} and filtersize: {}'.format(i+1, filter_sizes[i]))

        conv = self.bottleneck(pool, filter_sizes[-1])

        # Expansion
        for i in range(self.layers-1 , -1, -1):
            print('Building expansion at layer: {} and filtersize: {}'.format(i+1, filter_sizes[i]))

            conv = self.expand(conv, convs[i], filter_sizes[i])

        outputs = Conv2D(1, 1, padding= 'same', activation='sigmoid')(conv)

        self.model = Model(inputs, outputs)
        print("Compiling model...")
        self.model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ["acc"])



    def contract(self, x, filter_size, kernel_size = 3, padding = 'same', strides = 1, dropout= False, dropout_rate = 0.5):
        conv = Conv2D(filter_size, kernel_size, padding=padding, strides=strides, activation='relu')(x)
        conv = Conv2D(filter_size, kernel_size, padding=padding, strides=strides, activation='relu')(conv)
        if dropout:
            conv = Dropout(0.5)(conv)
            
        pool = MaxPool2D(pool_size = (2,2))(conv)
        
        return conv, pool

    def expand(self, x, contract_conv, filter_size, kernel_size = 3, padding = 'same', strides = 1):
        up = UpSampling2D(size = (2, 2))(x)
        concat = Concatenate()([up, contract_conv])

        conv = Conv2D(filter_size, kernel_size, padding=padding, strides=strides, activation='relu')(concat)
        conv = Conv2D(filter_size, kernel_size, padding=padding, strides=strides, activation='relu')(conv)

        return conv

    def bottleneck(self, x, filter_size, kernel_size = 3, padding = 'same', strides = 1):
        conv = Conv2D(filter_size, kernel_size, padding= padding, strides= strides, activation= 'relu')(x)
        conv = Conv2D(filter_size, kernel_size, padding= padding, strides= strides, activation= 'relu')(conv)
        return conv

    def describe_model(self):
        if self.model == None:
            print('Cannot find any model')
        else:
            self.model.summary()


    def train_model(self, epochs, batch_size):
        print()
        print('Training model')
        self.model.fit(x= self.x_train, y = self.y_train,  
                       validation_data =(self.x_test, self.y_test), 
                       epochs=epochs, batch_size = batch_size)


    def save_weights(self, filename):
        print()
        print('Saving Model')
        self.model.save_weights(filename)
        
    def load_weights(self, filename):
        print()
        print('Saving Model')
        self.model.load_weights(filename)
        
    def predict(self):
        