# LITS_segmentation

__1.Create an env__  
conda create --name <env> --file lits_requirements.txt
    

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
    -specify where the data is
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
    
    [] TODO 4.3 Inference
```

