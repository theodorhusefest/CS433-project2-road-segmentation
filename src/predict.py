import argparse
import numpy as np

from src.UNET import UNET
from src.create_submission import create_submission

from google.cloud import storage
from zipfile import ZipFile

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
    parser.add_argument(
        '--job-name',
        type=str,
        default=''
    )
    args, _ = parser.parse_known_args()
    return args

args = get_args()


BUCKETNAME = 'cs433-ml'
IMAGE_SHAPE = (256, 256, 3)
DIR = args.job_dir

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKETNAME)
blob = bucket.blob('data/test_set_images.zip')
blob.download_to_filename('test_set_images.zip')

with ZipFile('test_set_images.zip', 'r') as zipObj:
    zipObj.extractall('test_set_images')


UNET = UNET(args, image_shape = IMAGE_SHAPE, layers = 4)
UNET.build_model(num_gpus= 4)
model = UNET.get_model()


blob_weights = bucket.blob('keras-job-dir/no_padd_400_filt_6_lay4/epoch35_F10.9715_11.55.h5')
blob_weights.download_to_filename('weights.h5')

model.load_weights('weights.h5')

img_patches, cropped_pred = create_submission('test_sub.csv', model, patch_size=200, padding_size=28)
np.save('./cropped_pred.npy', cropped_pred)

up_blob = bucket.blob('predictions')

up_blob.upload_from_filename('./cropped_pred.npy')



