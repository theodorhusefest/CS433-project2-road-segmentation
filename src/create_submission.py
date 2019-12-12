import os
import cv2
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize

from src.mask_to_submission import masks_to_submission


def create_submission(submission_name, model):
    """
    Function to create submissionfile
    
    """
    
    # Load images
    test_set = load_test_img()
    
    # Predict on given model
    predictions = model.predict(test_set)
    
    # Fix labels
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0
    
    # Scale back up
    predictions = np.asarray([resize(predictions[i], (608, 608, 1)) for i in range(len(predictions))])
    
    # Save predictions as imgs, and keep the names
    image_names = save_test_img(predictions)
    
    masks_to_submission(submission_name, image_names)
    
    print('\nSuccesfully created submission.')
    

    
def lower_res(x, channels, res):
    """
    Helper file to change resultion of photo
    """
    return np.asarray(resize(x, (res, res, channels)))


def load_test_img(filepath = './data/test_set_images/', patch_size = 128):
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
            test_img = lower_res(mpimg.imread(image_path), 3, patch_size)
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


