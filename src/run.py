import numpy as np
import tensorflow as tf

from src.UNET import UNET
from src.helpers import get_args
from src.preprocessing import data_generator, prepare_labels
from keras.preprocessing.image import ImageDataGenerator

# Check if there are GPU available to train on
num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", num_gpus)

# Load data
x_tr, x_te, y_tr, y_te = data_generator(64, train_test_ratio = 0.80, 
                        num_images = 2, rotation_degs= range(1, 91, 50), 
                        padding_size=28, download_from_cloud=True)

print('Loaded {} patches for x_train, and {} for x_test.'.format(len(x_tr), len(x_te)))

# Initialize Keras Imagegenerator
datagen = ImageDataGenerator()
datagen.fit(x_tr)

args = get_args()

# Builds the UNET
UNET = UNET(args, image_shape = x_tr[0].shape, layers = 4)
UNET.build_model(num_gpus)
UNET.describe_model()

UNET.train_generator(datagen, x_tr, y_tr, x_te, y_te, epochs = 250, batch_size = 64)

