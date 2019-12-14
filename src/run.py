import numpy as np
import argparse

from src.preprocessing import data_generator
from src.UNET import UNET

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", num_gpus)


x_tr, x_te, y_tr, y_te = data_generator(400, train_test_ratio = 0.8 ,num_images = 100, rotation_degs= [45], padding_size=104, download_from_cloud=True)
print()
print('Loaded {} patches for x_train, and {} for x_test.'.format(len(x_tr), len(x_te)))

def get_args():
    """
    Parses arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        default = './models/'
    )
    parser.add_argument(
        '--job-name',
        type=str,
        default='padded_400_filt_6_lay4'
    )
    args, _ = parser.parse_known_args()
    return args


def fix_labels(y):
    y[y >= 0.5] = 1
    y[y < 0.5] = 0

    return y.astype(int)

y_tr = fix_labels(y_tr)
y_te = fix_labels(y_te)


datagen = ImageDataGenerator(
)


datagen.fit(x_tr)
print(x_tr[0].shape)

args = get_args()

UNET = UNET(args, image_shape = x_tr[0].shape, layers = 4)
UNET.build_model(num_gpus)
UNET.describe_model()

UNET.train_generator(datagen, x_tr, y_tr, x_te, y_te, epochs = 250, batch_size = 64)

