import numpy as np
import argparse

from src.preprocessing import data_generator
from src.UNET import UNET

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


x_tr, x_te, y_tr, y_te = data_generator(128, num_images = 100, rotation_degs= range(15, 360, 15), download_from_cloud=True)
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

UNET = UNET(args, image_shape = x_tr[0].shape, layers = 2)
UNET.build_model()
UNET.describe_model()

UNET.train_generator(datagen, x_tr, y_tr, x_te, y_te, epochs = 100, batch_size = 64)
