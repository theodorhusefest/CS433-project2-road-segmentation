from importlib import reload
import src
reload(src)

import numpy as np

from src.preprocessing import data_generator
from src.UNET import UNET

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


x_tr, x_te, y_tr, y_te = data_generator(128, num_images = 100, rotation_degs=range(3,360,3))
print()
print('Loaded {} patches for x_train, and {} for x_test.'.format(len(x_tr), len(x_te)))

#from skimage.transform import resize
#NEW_RES = 64

#def lower_res(x, channels):
#    return np.asarray([resize(x[i], (NEW_RES, NEW_RES, channels) ) for i in rsange(len(x))])

#x_tr = lower_res(x_train, 3)
#x_te = lower_res(x_test, 3)
#y_tr = lower_res(y_train, 1)
#y_te = lower_res(y_test, 1)

def fix_labels(y):
    y[y >= 0.5] = 1
    y[y < 0.5] = 0

    return to_categorical(y.astype(int))

y_tr = fix_labels(y_tr)
y_te = fix_labels(y_te)


datagen = ImageDataGenerator(
    featurewise_std_normalization = True 
)


datagen.fit(x_tr)
print(x_tr[0].shape)

UNET = UNET(image_shape = x_tr[0].shape, layers = 2)
UNET.build_model()
UNET.describe_model()

UNET.train_generator(datagen, x_tr, y_tr, x_te, y_te, epochs = 5, batch_size = 32)

UNET.save_model()
