"""
Run this script to prepare the Omniglot dataset from the raw Omniglot dataset that is found at
https://github.com/brendenlake/omniglot/tree/master/python.

This script prepares an enriched version of Omniglot the same as is used in the Matching Networks and Prototypical
Networks papers.

1. Augment classes with rotations in multiples of 90 degrees.
2. Downsize images to 28x28
3. Uses background and evaluation sets present in the raw dataset
"""
from skimage import io
from skimage import transform
import zipfile
import shutil
import os

# Parameters
kamon_location = '/data/output/segmentation_extended/cropping/training/200018823'
output_shape = (100, 100)

print('Processing kamon dataset...')
for img_path in os.listdir(kamon_location):
        if(os.path.isfile(os.path.join(kamon_location, img_path))):
            img = io.imread(kamon_location+'/'+img_path)
            img_width = img.shape[0]
            img_height = img.shape[1]
            
            cropping_factors = [0.9, 0.8]
            for cropping_factor in cropping_factors:
                new_img_width = cropping_factor*img_width
                new_img_height = cropping_factor*img_height
                half_width_diff = int((img_width-new_img_width)/2)
                half_height_diff = int((img_height-new_img_height)/2)
                
                new_img = img[half_width_diff:(img_width-half_width_diff), half_height_diff:(img_height-half_height_diff)].copy()
                new_img = transform.resize(new_img, output_shape, anti_aliasing=True)
                #new_img = (img - img.min()) / (img.max() - img.min())
                io.imsave(kamon_location+'/scaled_{}/'.format(cropping_factor)+img_path, new_img)