
For working with __GITHUB__ check __github_commands.txt__

# LITS_segmentation

__1.Create an env__  
Create a Conda env   
*Python==3.8.2  
GPU  
conda install -c anaconda cudatoolkit  
conda install -c anaconda cudnn  
*Pytorch==1.9.0 (https://pytorch.org/get-started/previous-versions/)  
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge  
      
conda install --file lits_requirements.txt  
pip install -r pip_req.txt  
    
  
__2.Organize the data__  
├── LITS_img/       # CT volumes/     
│   ├── train/  
│   │   ├──001.nii.gz    
│   ├── valid/  
│   └── test/  
│   └── excluded/ *here comes lits volumes without tumor class on it (could also go to test)*
│

├──LITS_seg/  #GT with __THE SAME__ names as in LITS_img folder  
│   ├── train/  
    │   ├──001.nii.gz   
│   ├── valid/  
│   └── test/  
│   └── excluded/  	#91, 89, 87, 41, 47, 38, 34, 32 


__3. Create a .yaml file__    
(e.g yaml_files/lits_gab_unet.yaml)  
    -specify where the data in  
    -where to save checkpoints and tensorboard logs  
    -specify the network  
    -specify the loss function  
    -specify the optimizer and lr  
    -Transformations and patch size  
    
  
__4. Run *main_augm_stable.py*__
```
    4.1 For training
    $ python main_augm_stable.py -c yaml_files\lits_gab_unet.yaml --train

    4.2 To restore training
    $ python main_augm_stable.py -c yaml_files\lits_gab_unet.yaml --restore -w checpoint_name.ckpt
    
    4.3 Inference
    $ python main_augm_stable.py -c yaml_files\lits_gab_unet.yaml --infer -w checpoint_name.ckpt -in_dir path_to_folder
    $ python main_augm_stable.py -c yaml_files\lits_gab_unet.yaml --infer -w checpoint_name.ckpt volumes path_to_volume.nii
    
    could also specify where to save with --save_dir
```

