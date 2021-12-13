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


def inference_save (net, inputdir, outdir, volumes, params ):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
   
    net = net.to(device)

    transforms = []
    for class_params in params.inference.augmentation:
        #print('checking which augmentation is loading ', class_params)
        transforms.extend(utils.class_loader(class_params))   
    infer_transform = monai.transforms.Compose(transforms)

    if inputdir is None:
        volumes = Path(volumes)
        print(volumes)
        infer_dicts = [{'image':str(volumes)}]
    else:
        inputdir = Path(inputdir)
        volumes = sorted(Path(inputdir).glob('*.nii*'))
        print(volumes)
        infer_dicts = [{'image':str(image_name)} for image_name in volumes ]
        
    infer_ds = monai.data.CacheDataset(data = infer_dicts, transform=infer_transform)    



    for index, batch in enumerate(infer_ds):

        input = batch['image'].unsqueeze(0).to(device)#.to(torch.float32)
        print(input.size())
        output = net.forward(input)

        output = output[0,...]
        vol_name=infer_dicts[index]["image"]
        mask = torch.argmax(output, axis=0).unsqueeze(0).unsqueeze(0).float()
        #print('after argmax', mask.shape)

        filename = Path(vol_name).name 
        new_vol = nib.load(vol_name)
        img_3D = new_vol.get_fdata()
        new_dim =  img_3D.shape
        #print(new_dim)
        output_orig_size = F.interpolate(mask, size=new_dim, mode = 'trilinear')
        mask2 = output_orig_size.squeeze().detach().cpu().numpy()
        mask2 = postprocessing.post_liver(mask2)
    
        header = nib.Nifti1Header()
        image = nib.Nifti1Image(mask2.astype(int), new_vol.affine, header)

        if outdir is None:
            path_mask = params['log']['configdir']/Path('inference')
            path_mask.mkdir(parents=True, exist_ok=True)
        else:
            path_mask = Path(outdir)/('inference')
            path_mask.mkdir(parents=True, exist_ok=True)

        #path_mask.mkdir(parents=True, exist_ok=True)
        path_mask1 = path_mask / filename
        nib.save(image, path_mask1)
        print(f'Input: {filename}  saved to: {path_mask1}')



        # #trying doing in a better way
        # roi_size = (64, 64, 32)
        # sw_batch_size = 1
        #
        # input = batch['image'].unsqueeze(0).to(device)#.to(torch.float32)
        # output = net.forward(input)
        # val_data = val_data["image"].to(device)
        # val_output = sliding_window_inference(
        #     val_data, roi_size, sw_batch_size, model)
        # # plot the slice [:, :, 80]
        # plt.figure("check", (20, 4))
        # plt.subplot(1, 5, 1)
        # plt.title(f"image {i}")
        # plt.imshow(val_data.detach().cpu()[0, 0, :, :, 80], cmap="gray")
        # plt.subplot(1, 5, 2)
        # plt.title(f"argmax {i}")
        # argmax = AsDiscrete(argmax=True)(val_output)
        # plt.imshow(argmax.detach().cpu()[0, 0, :, :, 80])
        # plt.subplot(1, 5, 3)
        # plt.title(f"largest {i}")
        # largest = KeepLargestConnectedComponent(applied_labels=[1])(argmax)
        # plt.imshow(largest.detach().cpu()[0, 0, :, :, 80])
        # plt.subplot(1, 5, 4)
        # plt.title(f"contour {i}")
        # contour = LabelToContour()(largest)
        # plt.imshow(contour.detach().cpu()[0, 0, :, :, 80])
        # plt.subplot(1, 5, 5)
        # plt.title(f"map image {i}")
        # map_image = contour + val_data
        # plt.imshow(map_image.detach().cpu()[0, 0, :, :, 80], cmap="gray")
        # plt.show()

        #end of trying