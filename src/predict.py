import argparse

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
IMAGE_SHAPE = (128, 128, 3)
DIR = args.job_dir

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKETNAME)
blob = bucket.blob('data/test_set_images.zip')
blob.download_to_filename('test_set_images.zip')

with ZipFile('test_set_images.zip', 'r') as zipObj:
    zipObj.extractall('test_set_images')


UNET = UNET(args, image_shape = IMAGE_SHAPE, layers = 4)
UNET.build_model()
model = UNET.get_model()

model.load_weights(DIR + 'padded_filt_6_lay4/weights/epoch30_F10.9113_19.21.h5')

create_submission(DIR + 'test_sub.csv', model)

