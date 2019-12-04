from importlib import reload
import src
reload(src)

from src.preprocessing import data_generator
from src.UNET import UNET

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

x_train, x_test, y_train, y_test = data_generator(80, num_images = 10, rotation_degs=[])
print()
print('Loaded {} patches for x_train, and {} for x_test.'.format(len(x_train), len(x_test)))

def fix_labels(y):
    y[y >= 0.5] = 1
    y[y < 0.5] = 0

    return to_categorical(y.astype(int))

y_tr = fix_labels(y_train)
y_te = fix_labels(y_test)


datagen = ImageDataGenerator(
    featurewise_std_normalization = True 
)

datagen.fit(x_train)
print(x_train[0].shape)

UNET = UNET(image_shape = x_train[0].shape, layers = 2)
UNET.build_model()
UNET.describe_model()

UNET.train_generator(datagen, x_train, y_tr, x_test, y_te, epochs = 2, batch_size = 32)

UNET.save_model()
