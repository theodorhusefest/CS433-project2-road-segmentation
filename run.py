
import os
import numpy as np
from src.helpers import load_image
from skimage.transform import resize

def shape_training_data(imgs, gt_imgs):
    SIZE = (256, 256)
    n = len(imgs)
     
    imgs = [resize(img, SIZE) for img in imgs]
    gt_imgs = [resize(gt_img, SIZE) for gt_img in gt_imgs]
    
    gt_imgs = [gt_imgs[i].reshape(256, 256, 1) for i in range(n)]
    gt_imgs = [np.around(gt_imgs[i]) for i in range(n)]
    
    return np.asarray(imgs), np.asarray(gt_imgs)

def load_training_data(datapath, num_samples):
    
    image_dir = datapath + "images/"
    files = os.listdir(image_dir)
    n = min(num_samples, len(files)) # Load maximum 20 images
    
    print("Loading " + str(n) + " images")
    imgs = [load_image(image_dir + files[i]) for i in range(n)]

    gt_dir = datapath + "groundtruth/"
    print("Loading " + str(n) + " images")
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

    return shape_training_data(imgs, gt_imgs)



x, y = load_training_data('data/training/', 20)

x_tr = x[0:10, :, : , :]
x_te = x[10:15, :, : , :]

y_tr = y[0:10, :, : , :]
y_te = y[10:15, :, : , :]
x_tr.shape, y_tr.shape, x_te.shape, y_te.shape




from imp import reload
import unet_class
reload(unet_class)
import unet_class


unet = unet_class.UNET(image_shape = (256, 256, 3), layers = 1)


unet.build_model()

unet.describe_model()



epochs = 2
batch_size = 2
#unet.train_model(x_tr, y_tr, x_te, y_te, epochs, batch_size)

unet.save_model('model_1layer.h5')
#unet.save_weights('UNET_1layer.h5')
