#!/usr/bin/env python3

import torch
from monai.visualize.img2tensorboard import add_animated_gif
import nibabel as nib 
import numpy as np
from pathlib import Path
import torch.nn.functional as F



def save_as_vol(output, full_name, outdir ):
    #checking the desired size of the prediction and saving with the same name and size
    prediction = output.detach().cpu().argmax(dim=0).unsqueeze(0).unsqueeze(0).float()
    filename = Path(full_name).name
    new_vol = nib.load(full_name)
    img_3D = new_vol.get_fdata()
    new_dim =  img_3D.shape
    #print(new_dim)
    output_orig_size = F.interpolate(prediction, size=new_dim, mode = 'trilinear')
    mask2 = output_orig_size.squeeze().detach().cpu().numpy()

    path_mask = Path(outdir)/('inference')
    path_mask.mkdir(parents=True, exist_ok=True)
    path_mask1 = path_mask / filename

    header = nib.Nifti1Header()
    check_mask = nib.Nifti1Image(mask2, new_vol.affine, header)
    nib.save( check_mask, path_mask1 )
    print(f'Input: {filename}  saved as {path_mask1}')

def generate_gallery(input, mask, prediction, writer, name_tag, global_step):

    # There is no batch here, it is a single volume sample!
    # Mask: background is black, class 1 is gray, class 2 is white, ...
    # Prediction: rounded, bacground = black, class 1 is gray, etc.

    input = input.detach().cpu().to(torch.float32).squeeze()
    mask = mask.detach().cpu().to(torch.float32).squeeze()
    prediction = prediction.detach().cpu().argmax(dim=0).float()
    #prediction  = prediction  / mask.max() #number of classes

    #print('images shape from evaluation file :',input.shape, mask.shape, prediction.shape)
    gallery = torch.cat((input, mask, prediction), dim=1)  # make it 64 (depth, axial), 192 (gt, mask, pred), 64   
    #WT101 X.cat(): Use rearrange(X, '...->...') from https://github.com/arogozhnikov/einops
    gallery = torch.stack((gallery, gallery, gallery), dim=0).numpy()  # make a fake grayscale image: R,G,B = intensity
    #WT101 X.stack(): Use rearrange(X, '...->...') from https://github.com/arogozhnikov/einops

#    print('gallery shape:',gallery.shape)
    add_animated_gif(
                    writer=writer,
                    tag=name_tag,
                    image_tensor=gallery,
                    max_out=255,
                    scale_factor=255.,
                    global_step=global_step
                    )


def save_batch(mask, name):
    """
    #Saving batch that is tensor to nii volume
    #input: batch; name which will be saved
    """
    mask1 = mask.detach().cpu().numpy()#.squeeze()
    #Use rearrange(X, '...->...') from https://github.com/arogozhnikov/einops
    print('saving volume with a name ', name)
    header = nib.Nifti1Header()
    image = nib.Nifti1Image(mask1.astype(int), np.eye(4), header)
    nib.save(image, name)
        
