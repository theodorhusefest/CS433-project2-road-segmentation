
""" Contains all basic helperfunctions, e.g load_image()"""

import os,sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_image(image_path):
    """
    Uses matplotlib to import an image from filename.
    params: 
        image_path: path to image
        
    returns:
        data: image as numpy array
    """
    
    data = mpimg.imread(image_path)
    return data

def img_float_to_uint8(img):
    """
    Converts all cell-values to uint8 (range(0-255))
    params: 
        img: img with float cell-values
        
    returns:
        rimg: img with cell-values range(0-255)
    """

    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    """
    Concatenate two images of same size two easier compare.
    Images must be the same size.
    params: 
        img: img to compare
        gt_img: img to compare (usually groundtrouth image of first img)
        
    returns:
        cimg: concatenated image
    """
    
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
        
    return cimg

def img_crop(img, w:int, h:int):
    """
    Crops img to wanted width and height
    params: 
        img: img to crop
        w: width of output img
        h: height of output img
        
    returns:
        cropped_img: concatenated image
    """
    
    cropped_img = []
    imgwidth = img.shape[0]
    imgheight = img.shape[1]
    is_2d = len(img.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                img_patch = img[j:j+w, i:i+h]
            else:
                img_patch = img[j:j+w, i:i+h, :]
            cropped_img.append(img_patch)
            
    return cropped_img


def get_args():
    """
    Parses arguments passed in command line.
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
        default=''
    )
    args, _ = parser.parse_known_args()
    return args

