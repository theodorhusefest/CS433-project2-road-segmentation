import os
import cv2
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize

from src.mask_to_submission import masks_to_submission
from src.preprocessing import add_padding, rotation_crop, img_crop, patches_to_images

def create_submission(submission_name, model, padding_size = 14, patch_size = 100):
    """
    Function to create submissionfile
    """
    
    # Load images
    test_set = load_test_img(padding_size = padding_size)
    
    # Get patches 
    img_patches = [img_crop(test_set[i], patch_size, patch_size, padding_size) for i in range(len(test_set))]
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    print('Shape of img_patches', img_patches.shape)
    # Predict on given model
    pred_patches = model.predict(img_patches)
    
    # Reassemble to img
    cropped_pred = [rotation_crop(pred_patch, patch_size, patch_size) for pred_patch in pred_patches]
    predictions = patches_to_images(np.asarray(cropped_pred), patch_size, img_side_len = 600)
    
    # Fix labels
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0
    
    # Scale back up
    predictions = change_res(predictions, 1, 608)
    
    # Save predictions as imgs, and keep the names
    image_names = save_test_img(predictions)
    
    masks_to_submission(submission_name, image_names)
    
    print('Succesfully created submission.')

def load_test_img(filepath = './data/test_set_images/', img_size = 600, padding_size = 14):
    """
    Loads all test images on and returns them as
    """
    test_imgs =[]
    
    # Load all images
    num_images = 50
    for i in range(1, num_images+1):
        test_id = 'test_' + str(i)
        image_path = filepath + test_id + '/' + test_id + '.png'

        if os.path.isfile(image_path):
            test_img = change_res(mpimg.imread(image_path), 3, img_size)
            test_img = add_padding(test_img, padding_size, 3)
            test_imgs.append(test_img)
        else:
            print('File ' + image_path + ' does not exist') 

    return np.asarray(test_imgs)


def save_test_img(pred, filepath = './predictions/'):
    """ 
    Saves predicted test images to predictions-folder
    """
    
    image_names = []
    
    # Load all images
    num_images = 50
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
        
    for i in range(1, num_images+1):
        test_id = 'test_' + str(i)
        
        if not os.path.isdir(filepath + test_id):
            os.mkdir(filepath + test_id)
            
        image_path = filepath + test_id + '/' + test_id + '.png'
        image_names.append(image_path)
        cv2.imwrite(image_path, pred[i-1], [cv2.IMWRITE_PNG_BILEVEL, 1])
        
    return image_names

def change_res(x, channels, res):
    """
    Helper file to change resolution of photo
    """
    return np.asarray(resize(x, (res, res, channels)))
