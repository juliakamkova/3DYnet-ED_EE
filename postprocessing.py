
import nibabel as nib
import numpy as np
import os

from skimage import measure
from skimage.measure import regionprops
from skimage.measure import label 

import argparse
from pathlib import Path

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def save_vols(dir, output, file_name):
    image = nib.Nifti1Image(output, np.eye(4))
    name = os.path.join (dir, file_name)
    #print(name)
    nib.save(image, name)

def post_liver(mymatrix):
    mymatrix [mymatrix < 0] = 0
    new = mymatrix > 0
    new2 =  mymatrix > 1
    mymatrix = new.astype(int) + new2.astype(int)
    mask = mymatrix > 0
    mask_kidney = getLargestCC(mask)
    mymatrix = mask_kidney * mymatrix
    return mymatrix




# input_folder = 'C:\\Users\\julkam\\Documents\\GitHub\\pytorch_unet\\comet_server\\version_1\\inference'
# output_folder = Path('C:\\Users\\julkam\\Documents\\GitHub\\pytorch_unet\\comet_server\\version_1\postprocessing')
# output_folder.mkdir(parents=True, exist_ok=True)

# x_filenames =  os.listdir (input_folder)
# for name in (x_filenames):
#     dir_path = os.path.join (input_folder, name)
#     img_orig = nib.load (dir_path)
#     mymatrix = img_orig.get_fdata()
#     mymatrix = post_liver(mymatrix)

#     header = nib.Nifti1Header()
#     image = nib.Nifti1Image(mymatrix, img_orig.affine, header )
#     save_path = os.path.join (output_folder, name)
#     print('just checking maybe we are here', save_path)
#     nib.save(image, save_path)