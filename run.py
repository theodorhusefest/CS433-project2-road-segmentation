from importlib import reload
import src
reload(src)

from src.preprocessing import data_generator
from src.UNET import UNET
from keras.preprocessing.image import ImageDataGenerator

x_train, x_test, y_train, y_test = data_generator(80, num_images = 10, rotation_degs=[15, 30])
print()
print('Loaded {} patches for x_train, and {} for x_test.'.format(len(x_train), len(x_test)))


datagen = ImageDataGenerator(
    featurewise_std_normalization = True 
)

datagen.fit(x_train)

UNET = UNET(image_shape = x_train[0].shape, layers = 1)
UNET.build_model()
UNET.describe_model()

UNET.train_generator(datagen, x_train, y_train, x_test, y_test, epochs = 2, batch_size = 5)

