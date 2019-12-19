


import os
import numpy as np
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK']='True' # Need this to work on Theos mac

from src.metrics import f1, recall, precision

from keras.models import Model, Sequential

from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Concatenate, Input, Dropout, LeakyReLU

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras.utils import multi_gpu_model

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau, TensorBoard

class UNET():

    def __init__(self, args, image_shape = (400, 400, 3), depth = 2):

        self.args = args
        self.IMAGE_SHAPE = image_shape
        self.depth = depth
        self.model = None
        self.lrelu = lambda x: LeakyReLU(alpha=0.01)(x)
        self.activation = self.lrelu

    def build_model(self, num_gpus):
        """
        Builds the model for a general number of depth.
        Contains three phases, contracting, bottleneck and expansion.
        """

        filter_sizes = [2**(2+i) for i in range(self.depth + 1)]   # [64, 128, 256, 512, 1024]

        inputs = Input(self.IMAGE_SHAPE)
        pool = inputs

        convs = []

        # Contracting 
        for i in range(self.depth):
            conv, pool = self.contract(pool, filter_sizes[i])
            # Save convolution for expanding phase
            convs.append(conv)

        conv = self.bottleneck(pool, filter_sizes[-1])

        # Expansion
        for i in range(self.depth-1 , -1, -1):
            conv = self.expand(conv, convs[i], filter_sizes[i])

        conv = Conv2D(filter_sizes[0], (3, 3), padding='same', activation= self.activation, kernel_initializer="he_normal")(conv)
        outputs = Conv2D(1, (1,1), padding= 'same', activation='sigmoid')(conv)

        self.model = Model(inputs, outputs)

        if num_gpus > 1:
            self.model = multi_gpu_model(self.model, num_gpus)
            print("Model running on {} GPU's.".format(num_gpus))

        opt = Adam(learning_rate=0.02) 
        self.model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = [f1, precision, recall, 'accuracy'])


    def contract(self, x, filter_size, kernel_size = (3, 3), padding = 'same', strides = 1):
        """
        Contracting phase of the model.
        Consists of two layers with convoluton, before a before a max pooling which reduces the dimentionality by two.
        BatchNormalization is added after each activation-function to prevent overfitting.
        """

        conv = Conv2D(filter_size, kernel_size, padding=padding, strides=strides, activation=self.activation, kernel_initializer="he_normal")(x)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filter_size, kernel_size, padding=padding, strides=strides, activation=self.activation, kernel_initializer="he_normal")(conv)
        conv = BatchNormalization()(conv)

        pool = MaxPool2D(pool_size = (2,2))(conv)
        
        return conv, pool


    def expand(self, x, contract_conv, filter_size, kernel_size = (3, 3), padding = 'same', strides = 1):
        """
        Expanding phase of the model.
        First an up-sampling which doubles the dimentionality, before two convolutinal layers.
        Consists of two layers with convoluton, before a before a pooling phase which reduces the dimentionality.
        BatchNormalization is added after each activation-function to prevent overfitting.
        """
        up = UpSampling2D(size = (2, 2))(x)
        concat = Concatenate(axis = 3)([contract_conv, up])

        conv = Conv2D(filter_size, kernel_size, padding=padding, strides=strides, activation=self.activation, kernel_initializer="he_normal")(concat)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filter_size, kernel_size, padding=padding, strides=strides, activation=self.activation, kernel_initializer="he_normal")(conv)
        conv = BatchNormalization()(conv)

        return conv


    def bottleneck(self, x, filter_size, kernel_size = (3, 3), padding = 'same', strides = 1):
        """
        The bottleneck is the lowest level in the model.
        Two convolutions followed by batchNormalization
        """
        conv = Conv2D(filter_size, kernel_size, padding= padding, strides= strides, activation= self.activation, kernel_initializer="he_normal")(x)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filter_size, kernel_size, padding= padding, strides= strides, activation= self.activation, kernel_initializer="he_normal")(conv)
        conv = BatchNormalization()(conv)

        return conv 

    def describe_model(self):
        """
        Produces a summary if model is available
        """
        if self.model == None:
            print('Cannot find any model')
        else:
            self.model.summary()


    def train_generator(self, datagen, x_train, y_train, x_val, y_val, epochs, batch_size):
        """
        Function that trains the model.
        Takes in train and validation data.
        Also creates callbacks
        """

        if self.args.load_best:
            print('Loading Best Model')
            self.model.load_weights('./models/best_model_F10.985.h5')
            return

        print('Starting Training')
        # Set path from command line args
        self.dir_ = self.args.job_dir + self.args.job_name
        filepath= self.dir_ + '/epoch{epoch:02d}_F1{val_f1:.2f}_' + datetime.now().strftime("%H.%M") + '.h5'

        # Saves the model each time validation f1 increases
        checkpoint = ModelCheckpoint(filepath, monitor='val_f1', verbose=1, period=1, 
                                        save_weights_only= True, 
                                        save_best_only=True, mode='max')

        # Quits training if val_f1 is not increasing for 20 epochs
        earlystop = EarlyStopping(monitor='val_f1', verbose=1, patience=20,
                                         mode='max', restore_best_weights= True)

        # Reduces learning rate if the loss is non-decreasing for 3 epochs
        reduceLR = ReduceLROnPlateau(monitor='loss', verbose= 1, patience = 3,
                                         mode='min', factor=0.9, min_delta=0.001, min_lr=0.00001)

        # Keeps a log of training
        tensorboard = TensorBoard(self.dir_, histogram_freq=1, write_graph=True)
        callbacks_list = [checkpoint, reduceLR, earlystop, tensorboard]

        # Train model
        self.model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size),
                                validation_data = (x_val, y_val),
                                steps_per_epoch = len(x_train)/batch_size, epochs = epochs,
                                callbacks=callbacks_list)

        # Saves the model after training
        self.save_model()


    def save_model(self):
        """
        Saves model to given filepath.
        """
        filepath= self.dir_ + '/FINISHED' + datetime.now().strftime("%d_%H.%M") + '.h5'
        self.model.save_weights(filepath)
