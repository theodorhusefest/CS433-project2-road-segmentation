import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable TensorFlow debugging info 

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from src.UNET import UNET
from src.helpers import get_args
from src.preprocessing import data_generator, prepare_labels
from src.create_submission import create_submission

# Constants
EPOCHS = 100
BATCH_SIZE = 64
DEPTH = 4

PATCH_SIZE = 200
PADDING = 14
NUM_IMAGES = 100
TRAIN_TEST_RATIO = 0.8
ROTATIONS = range(1, 91, 90)

# Check if there are GPU available to train on
num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", num_gpus)

# Load data
x_tr, x_te, y_tr, y_te = data_generator(PATCH_SIZE, train_test_ratio = TRAIN_TEST_RATIO, 
                        num_images = NUM_IMAGES, rotation_degs= ROTATIONS, 
                        padding_size=PADDING)

print('Loaded {} patches for x_train, and {} for x_test.'.format(len(x_tr), len(x_te)))

# Initialize Keras Imagegenerator
datagen = ImageDataGenerator()
datagen.fit(x_tr)

args = get_args()

# Builds the UNET
UNET = UNET(args, image_shape = x_tr[0].shape, depth = DEPTH)
UNET.build_model(num_gpus)
UNET.describe_model()

UNET.train_generator(datagen, x_tr, y_tr, x_te, y_te, epochs = EPOCHS, batch_size = BATCH_SIZE)

create_submission('submisson.csv', UNET.model, patch_size = PATCH_SIZE, padding_size= PADDING)