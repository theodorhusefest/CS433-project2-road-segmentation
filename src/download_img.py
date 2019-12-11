import numpy as np
from google.cloud import storage
import matplotlib.image as mpimg
from zipfile import ZipFile

BUCKETNAME = 'cs433-ml'
DESTINATION = './temp2'

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKETNAME)
blob = bucket.blob('data/training.zip')
blob.download_to_filename('temp.zip')

with ZipFile('temp.zip', 'r') as zipObj:
    zipObj.extractall('temp2')


# Load all images
num_images = 100

x_imgs = []
for i in range(1, num_images+1):
    imageid = "satImage_%.3d" % i
    x_img = DESTINATION + '/training/images/' + imageid + '.png'
    y_img = DESTINATION +'/training/groundtruth/' + imageid + '.png'
    x_img  = mpimg.imread(x_img)
    x_imgs.append(x_img)


print(len(x_imgs))