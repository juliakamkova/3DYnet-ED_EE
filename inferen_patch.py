#import numpy as np
#import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import utils
import monai
import nibabel as nib
#from monai.data import CacheDataset,
from monai.data import CacheDataset,list_data_collate, decollate_batch, Dataset
from monai.metrics import compute_meandice , DiceMetric
from monai.inferers import sliding_window_inference
#import glob
from pathlib import Path
import evaluation
from monai.transforms import AsDiscrete, Compose, EnsureType
from pytorch_lightning.callbacks import LearningRateMonitor
# import networks
from box import Box


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
                volumes = sorted(Path(inputdir).glob('*.nii*'))
                print(volumes)
                infer_dicts = [{'image': str(image_name)} for image_name in volumes]

        infer_ds = monai.data.Dataset(data=infer_dicts, transform=infer_transform)
        infer_loader = torch.utils.data.DataLoader(infer_ds, batch_size=1)

        for i, infer_data in enumerate(infer_loader):
        #for infer_data in infer_loader:
                infer_inputs = infer_data["image"].to(device)
                #print(f"batch_data image: {infer_data['image'].shape}")

                #print('params.data.patch', params.data.patch)
                roi_size = list(params.data.patch)
                sw_batch_size = 1

                #input = batch['image'].unsqueeze(0).to(device)#.to(torch.float32)
                #output = net.forward(input)
                #infer_data = batch["image"].unsqueeze(0).to(device)
                with torch.no_grad():
                        val_output = sliding_window_inference(infer_inputs, roi_size, sw_batch_size, net)
                val_output = val_output[0,...]
                mask = torch.argmax(val_output, axis=0).int()
                mask2 = mask.detach().cpu().numpy()
                #print(mask2.shape)

                #print('checking image name' ,Path(infer_dicts[i]["image"]).name)
                new_vol = nib.load(infer_dicts[i]["image"])
                img_3D = new_vol.get_fdata()
                header = nib.Nifti1Header()
                image = nib.Nifti1Image(mask2.astype(int), new_vol.affine, header)

                filename = Path(infer_dicts[i]["image"]).name
                if outdir is None:
                    path_mask = params['log']['configdir']/Path('inference')
                    path_mask.mkdir(parents=True, exist_ok=True)
                else:
                    path_mask = Path(outdir)/('inference')
                    path_mask.mkdir(parents=True, exist_ok=True)

                path_mask1 = path_mask / filename
                nib.save(image, path_mask1)
                print(f'Input: {filename}  saved to: {path_mask1}')

