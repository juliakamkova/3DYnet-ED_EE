import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import nibabel as nib
import utils
import monai
import einops
import matplotlib.pyplot as plt
import evaluation
import postprocessing


# change

def inference_save(net, inputdir, outdir, volumes, params):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    net = net.to(device)

    transforms = []
    for class_params in params.inference.augmentation:
        # print('checking which augmentation is loading ', class_params)
        transforms.extend(utils.class_loader(class_params))
    infer_transform = monai.transforms.Compose(transforms)

    if inputdir is None:
        volumes = Path(volumes)
        print(volumes)
        infer_dicts = [{'image': str(volumes)}]
    else:
        inputdir = Path(inputdir)
        print('checking the directory', inputdir)
        volumes = sorted(Path(inputdir).glob('*.nii*'))
        print(volumes)
        infer_dicts = [{'image': str(image_name)} for image_name in volumes]

    infer_ds = monai.data.CacheDataset(data=infer_dicts, transform=infer_transform)

    for index, batch in enumerate(infer_ds):

        input = batch['image'].unsqueeze(0).to(device)  # .to(torch.float32)
        print('input size', input.size())

        output = net.forward(input)

        output = output[0, ...]
        vol_name = infer_dicts[index]["image"]
        mask = torch.argmax(output, axis=0).unsqueeze(0).unsqueeze(0).float()
        print('after argmax', mask.shape)

        filename = Path(vol_name).name
        new_vol = nib.load(vol_name)
        img_3D = new_vol.get_fdata()
        new_dim = img_3D.shape
        print('new dim', new_dim)
        output_orig_size = F.interpolate(mask, size=new_dim, mode='trilinear')
        mask2 = output_orig_size.squeeze().detach().cpu().numpy()
        mask2 = postprocessing.post_liver(mask2)

        header = nib.Nifti1Header()
        image = nib.Nifti1Image(mask2.astype(int), new_vol.affine, header)

        if outdir is None:
            path_mask = params['log']['configdir'] / Path('inference')
            path_mask.mkdir(parents=True, exist_ok=True)
        else:
            path_mask = Path(outdir) / ('inference')
            path_mask.mkdir(parents=True, exist_ok=True)

        # path_mask.mkdir(parents=True, exist_ok=True)
        path_mask1 = path_mask / filename
        nib.save(image, path_mask1)
        print(f'Input: {filename}  saved to: {path_mask1}')

      