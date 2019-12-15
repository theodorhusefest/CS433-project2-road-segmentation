"""
Helper module to predict when multiple GPU's have been used to train in cloud (need multiple CPU to predict as well).

Reads weights-file and test-images from Google Cloud, predicts and loads up predictions to same Google Cloud Bucket.
Saves predictions, not CSV-file.

"""
import argparse
import numpy as np
from zipfile import ZipFile

from google.cloud import storage

from src.UNET import UNET
from src.helpers import get_args
from src.create_submission import create_submission

# Parses arguments from command line
args = get_args()

# Set constants
BUCKETNAME = 'cs433-ml'
IMAGE_SHAPE = (256, 256, 3)
DIR = args.job_dir

# Connect to Google Cloud Bucket
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKETNAME)

blob_imgs = bucket.blob('data/test_set_images.zip')
blob_imgs.download_to_filename('test_set_images.zip')

with ZipFile('test_set_images.zip', 'r') as zipObj:
    zipObj.extractall('test_set_images')

# Build model
UNET = UNET(args, image_shape = IMAGE_SHAPE, layers = 4)
UNET.build_model(num_gpus= 4)
model = UNET.get_model()


# Read weights file from Bucket
blob_weights = bucket.blob('keras-job-dir/padded_200_filt_6_lay4_final/epoch26_F10.98_13.50.h5')
blob_weights.download_to_filename('weights.h5')

# Load the model with weights and predict
model.load_weights('weights.h5')
img_patches, cropped_pred = create_submission('test_sub.csv', model, patch_size=200, padding_size=28)
np.save('./cropped_pred.npy', cropped_pred)

# Save predictions back to cloud
up_blob = bucket.blob('predictions')
up_blob.upload_from_filename('./cropped_pred.npy')

