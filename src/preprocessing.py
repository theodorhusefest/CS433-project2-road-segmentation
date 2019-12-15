
""" Contains all functions used in preprocessing""" 


import os
import numpy as np
from six.moves import urllib
from scipy import ndimage, misc
import matplotlib.image as mpimg
from skimage.transform import resize

from google.cloud import storage
from zipfile import ZipFile


def data_generator(patch_size, num_images = 100, train_test_ratio = 0.8, rotation_degs = [], download_from_cloud= False, padding_size = 14,
                   DATAPATH = "gs://cs433-ml/data/training.zip"):
    """
    Generate data from images, contruct patches and split data into train/test set
    
    Returns:
        x_train, x_test, y_train, y_test
    """
    
    if download_from_cloud:
        BUCKETNAME = 'cs433-ml'
        DESTINATION = './temp2'

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKETNAME)
        blob = bucket.blob('data/training.zip')
        blob.download_to_filename('temp.zip')

        with ZipFile('temp.zip', 'r') as zipObj:
            zipObj.extractall('temp2')

    else:
        DESTINATION = './data'
        
    x_imgs = []
    y_imgs = []

    # Load all images
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        x_image_filename = DESTINATION + '/training/images/' + imageid + '.png'
        y_image_filename = DESTINATION + '/training/groundtruth/' + imageid + '.png'

        if os.path.isfile(x_image_filename) and os.path.isfile(y_image_filename):
            x_img = mpimg.imread(x_image_filename)
            x_img = add_padding(x_img, padding_size, 3)
            x_imgs.append(x_img)

            y_img = mpimg.imread(y_image_filename)
            y_img = y_img.reshape((y_img.shape[0], y_img.shape[1], 1))
            y_img = add_padding(y_img, padding_size, 1)
            y_imgs.append(y_img)

        else:
            print('File ' + x_image_filename + ' does not exist') 

    
    num_images = len(x_imgs)
    IMG_WIDTH = x_imgs[0].shape[0]
    IMG_HEIGHT = x_imgs[0].shape[1]

    assert (x_imgs[0].shape[0] - padding_size*2)%patch_size == 0 , "patch size is not multiple of image width/height"

    x_train, x_test, y_train, y_test = patches_split(x_imgs, y_imgs, patch_size, train_test_ratio, padding_size)
    
    for deg in rotation_degs:
        x_rotated_imgs = []
        y_rotated_imgs = []
        for i in range(num_images):
            tmp_x = rotation_crop(ndimage.rotate(x_imgs[i], deg, reshape=True, mode='mirror'), IMG_WIDTH, IMG_HEIGHT)
            tmp_y = rotation_crop(ndimage.rotate(y_imgs[i], deg, reshape=True, mode='mirror'), IMG_WIDTH, IMG_HEIGHT)
            x_rotated_imgs.append(tmp_x)
            x_rotated_imgs.append(np.flip(tmp_x,0))
            x_rotated_imgs.append(np.flip(tmp_x,1))
            if i < 30:
                x_rotated_imgs.append(np.flip(x_imgs[i],0))
            y_rotated_imgs.append(tmp_y)
            y_rotated_imgs.append(np.flip(tmp_y,0))
            y_rotated_imgs.append(np.flip(tmp_y,1))

            if i < 30:
                y_rotated_imgs.append(np.flip(y_imgs[i],0))
            
        x_train_rot, x_test_rot, y_train_rot, y_test_rot = patches_split(x_rotated_imgs, y_rotated_imgs, 
                                                                         patch_size, train_test_ratio, padding_size)
        
        x_train = np.concatenate([x_train, x_train_rot])
        y_train = np.concatenate([y_train, y_train_rot])

        x_test = np.concatenate([x_test, x_test_rot])
        y_test = np.concatenate([y_test, y_test_rot])



    #y_train = np.asarray([y.reshape(patch_size, patch_size, 1) for y in y_train])
    #y_test = np.asarray([y.reshape(patch_size, patch_size, 1) for y in y_test])

    return x_train, x_test, y_train, y_test


def rotation_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    padding = (imgwidth - w)//2
    for i in range(0,imgheight,h):
        if (i + h > imgheight):
            continue 
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j+padding:j+w+padding, i+padding:i+h+padding]
            else:
                im_patch = im[j+padding:j+w+padding, i+padding:i+h+padding,: ]
            list_patches.append(im_patch)
    return list_patches[0]

def lower_res(x, channels, res):
    return np.asarray(resize(x, (res, res, channels)))

def add_padding(img, padding_size, channels):
    padded_img = np.zeros((img.shape[0] + padding_size*2,
                           img.shape[1] + padding_size*2,
                           channels))
    for channel in range(channels):
        padded_img[:,:,channel] = np.pad(img[:,:,channel],
                                         ((padding_size, padding_size),(padding_size, padding_size)), 
                                         'symmetric')
    return padded_img

def img_crop(im, w, h, p):
    list_patches = []
    imgwidth = im.shape[0] - p*2
    imgheight = im.shape[1] - p*2
    is_2d = len(im.shape) < 3
    for i in range(p,imgheight+p,h):
        for j in range(p,imgwidth+p,w):
            if is_2d:
                im_patch = im[j-p:j+w+p, i-p:i+h+p]
            else:
                im_patch = im[j-p:j+w+p, i-p:i+h+p, :]
            list_patches.append(im_patch)
    return list_patches

def prepare_labels(y):
    """
    Converts greyscale image into binary values. 1 = road, 0 = not-road
    """
    y[y >= 0.5] = 1
    y[y < 0.5] = 0

    return y.astype(int)

def patches_split(x, y, patch_size, split, padding_size):
    assert len(x) == len(y), "Length of x and y has to be the same"
    perm = np.random.permutation(len(x))
    split_perm = int(len(x)*split)
    train_perm = perm[:split_perm]
    test_perm = perm[split_perm:]
    
    x_tr_img_patches = [img_crop(x[i], patch_size, patch_size, padding_size) for i in train_perm]
    x_train = np.asarray([x_tr_img_patches[i][j] for i in range(len(x_tr_img_patches)) for j in range(len(x_tr_img_patches[i]))])
    x_te_img_patches = [img_crop(x[i], patch_size, patch_size, padding_size) for i in test_perm]
    x_test = np.asarray([x_te_img_patches[i][j] for i in range(len(x_te_img_patches)) for j in range(len(x_te_img_patches[i]))])
    
    y_tr_img_patches = [img_crop(y[i], patch_size, patch_size, padding_size) for i in train_perm]
    y_train = np.asarray([y_tr_img_patches[i][j] for i in range(len(y_tr_img_patches)) for j in range(len(y_tr_img_patches[i]))])
    y_te_img_patches = [img_crop(y[i], patch_size, patch_size, padding_size) for i in test_perm]
    y_test = np.asarray([y_te_img_patches[i][j] for i in range(len(y_te_img_patches)) for j in range(len(y_te_img_patches[i]))])
    
    return x_train, x_test, y_train, y_test



def patches_to_images(patches, patch_size, img_side_len=400):
    """
    takes an array of patches and integer of patch_size
    returns an array of images
    """
    assert patches.shape[0]%(img_side_len/patch_size)**2==0, "Uneven number of patches given image and patch size"
    
    num_patches_img = (img_side_len/patch_size) ** 2
    num_imgs = int(patches.shape[0]/num_patches_img)
    imgs = []
    
    
    tot_index = 0
    for img in range(num_imgs):
        img_index = 0
        image = []
        for row in range(int(np.sqrt(num_patches_img))):
            img_row = []
            for col in range(int(np.sqrt(num_patches_img))):
                if len(img_row)==0:
                    img_row = patches[tot_index]
                else:
                    img_row = np.append(img_row, patches[tot_index], axis=0)
                tot_index += 1
            if len(image)==0:
                image = img_row
            else:
                image = np.append(image, img_row, axis=1)
        imgs.append(image)
        
    return np.asarray(imgs)
