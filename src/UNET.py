


import os
import numpy as np
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK']='True' # Need this to work on Theos mac

from src.metrics import f1, recall, precision

from keras.models import Model, Sequential

from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Concatenate, Input, Dropout, LeakyReLU

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau, TensorBoard

class UNET():

    def __init__(self, args, image_shape = (400, 400, 3), layers = 2):

        self.args = args
        self.IMAGE_SIZE = image_shape[0]
        self.IMAGE_SHAPE = image_shape
        self.layers = layers
        self.dropout_rate = 0
        self.model = None
        self.history = None
        self.lrelu = lambda x: LeakyReLU(alpha=0.1)(x)
        self.activation = self.lrelu

        self.dir_ = self.args.job_dir + self.args.job_name
        self.weights_dir = self.dir_ + '/weights'

        """
        if not os.path.isdir(self.dir_):
            os.mkdir(self.dir_)
        
        if not os.path.isdir(self.weights_dir):
            os.mkdir(self.weights_dir)
        """


    def build_model(self):
        """
        Builds the model for a general number of layers.
        Contains three phases, contracting, bottleneck and expansion.
        """

        print('Building model with {} layers'.format(self.layers))
        filter_sizes = [2**(6+i) for i in range(self.layers + 1)]
        print('Filtersizes being used in UNET: {}'.format(filter_sizes))

        
        inputs = Input(self.IMAGE_SHAPE)
        pool = inputs

        convs = []

        # Contracting 
        for i in range(self.layers):

            print('Bulding contraction layers at layer: {} and filtersize: {}'.format(i+1, filter_sizes[i]))
            if i == self.layers - 1: # If this is last iteration
                conv, pool = self.contract(pool, filter_sizes[i], dropout= True)
            else:
                conv, pool = self.contract(pool, filter_sizes[i], dropout= True)
            
            # Save convolution for expanding phase
            convs.append(conv)

        # Bottleneck
        print('Building bottleneck at layer: {} and filtersize: {}'.format(self.layers, filter_sizes[-1]))

        conv = self.bottleneck(pool, filter_sizes[-1])

        # Expansion
        for i in range(self.layers-1 , -1, -1):
            print('Building expansion at layer: {} and filtersize: {}'.format(i+1, filter_sizes[i]))

            conv = self.expand(conv, convs[i], filter_sizes[i], dropout= True)

        conv = Conv2D(filter_sizes[0], (3, 3), padding='same', activation= self.activation, kernel_initializer="he_normal")(conv)
        outputs = Conv2D(1, (1,1), padding= 'same', activation='sigmoid')(conv)


        self.model = Model(inputs, outputs)
        print("Compiling model...")
        opt = Adam(learning_rate=0.02)
        self.model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = [f1, precision, recall, 'accuracy'])
        print("Model compiled.")


    def contract(self, x, filter_size, kernel_size = (3, 3), padding = 'same', strides = 1, dropout= False):
        """
        Contracting phase of the model.
        Consists of two layers with convoluton, before a before a pooling phase which reduces the dimentionality.
        The last contraction before the bottleneck there is a dropout-phase.

        params:
            x: data to be contracted
            filter_size: w
        """

        conv = Conv2D(filter_size, kernel_size, padding=padding, strides=strides, activation=self.activation, kernel_initializer="he_normal")(x)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filter_size, kernel_size, padding=padding, strides=strides, activation=self.activation, kernel_initializer="he_normal")(conv)
        conv = BatchNormalization()(conv)
        if dropout:
            conv = Dropout(self.dropout_rate)(conv)

        pool = MaxPool2D(pool_size = (2,2))(conv)
        
        return conv, pool


    def expand(self, x, contract_conv, filter_size, kernel_size = (3, 3), padding = 'same', strides = 1, dropout = False):
        up = UpSampling2D(size = (2, 2))(x)
        concat = Concatenate(axis = 3)([contract_conv, up])

        conv = Conv2D(filter_size, kernel_size, padding=padding, strides=strides, activation=self.activation, kernel_initializer="he_normal")(concat)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filter_size, kernel_size, padding=padding, strides=strides, activation=self.activation, kernel_initializer="he_normal")(conv)
        conv = BatchNormalization()(conv)
        if dropout:
            conv = Dropout(self.dropout_rate)(conv)
        return conv


    def bottleneck(self, x, filter_size, kernel_size = (3, 3), padding = 'same', strides = 1):
        conv = Conv2D(filter_size, kernel_size, padding= padding, strides= strides, activation= self.activation, kernel_initializer="he_normal")(x)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filter_size, kernel_size, padding= padding, strides= strides, activation= self.activation, kernel_initializer="he_normal")(conv)
        conv = BatchNormalization()(conv)
        conv = Dropout(self.dropout_rate)(conv)

        return conv 


    def describe_model(self):
        if self.model == None:
            print('Cannot find any model')
        else:
            self.model.summary()


    def train_model(self, x_train, y_train, x_val, y_val, epochs, batch_size):
        print()
        print('Training model')
        self.model.fit(x= x_train, y = y_train,  
                       validation_data =(x_val, y_val), 
                       epochs=epochs, batch_size = batch_size)


    def train_generator(self, datagen, x_train, y_train, x_val, y_val, epochs, batch_size):
        print('Training using generator')


        filepath= self.weights_dir + '/epoch{epoch:02d}_F1{val_f1:.2f}' + datetime.now().strftime("%d_%H.%M") + '.h5'
        #logs_path = self.args.job_name + '.csv'

        checkpoint = ModelCheckpoint(filepath, monitor='val_f1', verbose=1, period=10, save_weights_only= True, save_best_only=True, mode='max')
        earlystop = EarlyStopping(monitor='val_f1', verbose=1, patience=30, mode='max', restore_best_weights= True)
        reduceLR = ReduceLROnPlateau(monitor='loss', verbose= 1, patience = 2, mode='min', factor=0.8, min_delta=0.005, min_lr=0.0001)
        #tensorboard = TensorBoard(self.dir_, histogram_freq=1, batch_size=64, write_graph=True)
        #csv_logger = CSVLogger(logs_path, append=True)
        callbacks_list = [checkpoint, reduceLR, earlystop]

        self.history = self.model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size),
                                validation_data = (x_val, y_val),
                                steps_per_epoch = len(x_train)/batch_size, epochs = epochs,
                                callbacks=callbacks_list)

        self.save_model()


    def save_model(self, filename  = ''):
        print("Saving Model")
    
        filepath= self.weights_dir + '/FINISHED' + datetime.now().strftime("%d_%H.%M") + '.h5'
        self.model.save_weights(filepath)


    def load_weights(self, model_filename, weights_filename):
        print()
        print('Loading Model')
        #self.model.load('./models/' + model_filename)
        self.model.load_weights('./models/' + weights_filename)
        
        
    def predict(self, x_test):
        print()
        print('Predicting on {} images'.format(x_test.shape[0]))
        self.model.predict(x_test)

    def get_model(self):
        return self.model