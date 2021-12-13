# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:15:31 2020

utils for model checking 
@author: julkam
"""
from pathlib import Path
import importlib

def class_loader(d, extra_args=(), extra_kwargs={}):
    result = []
    for key in d.keys():
        *parts, function_name = key.split('.')
        for idx, part in enumerate(parts):
            name = '.'.join(parts[0:idx+1])
            module = importlib.import_module(name)
        fun = getattr(module, function_name)
        params = tuple(d[key].get('args',tuple()))+extra_args
        kw_params = d[key].get('kwargs', dict())
        obj = fun(*params, **kw_params, **extra_kwargs)
        result.append(obj)
    return result

def version(module):
    if isinstance(module, (list,tuple,set)):
        result = {}
        for mod in module:
            result = {**result, **version(mod)}
        return result

    try:
        return {module.__name__: str(module.version.__version__)}
    except:
        pass
    try:
        return {module.__name__: str(module.__version__)}
    except:
        pass
    try:
        return {module.__name__: str(module.version.version)}
    except:
        pass
    try:
        return {module.__name__: str(module.version)}
    except:
        pass    
    return {module: 'unknown version'}

def split_data(img_dir):
    """
    Split list of names into 3 subsets: train(80%), validation(10%), test(10%)
    
    Args:
        img_dir (string) : location of volumes
    Returns:
        x_train_filenames: type(list)
        x_val_filenames: type(list) 
        x_test_filenames: type(list)

    """
    dirname = Path(img_dir) 
    train_fold = dirname / 'train'
    valid_fold = dirname / 'valid'
    test_fold = dirname / 'test'
    x_train_filenames =[ str(x.name) for x in train_fold.iterdir()]
    x_val_filenames =[ str(x.name) for x in valid_fold.iterdir()]
    x_test_filenames =[ str(x.name) for x in test_fold.iterdir()]

    return x_train_filenames, x_val_filenames, x_test_filenames 


# def compute_class_freqs(label, num_clas):
#     """
#     Compute frequences for each class in the label
    
#     Args:
#         label (np.array): matrix of labels, size (h*w*l)
#         num_clas (int) : number of label classes in the segmentation
#     Returns:
#         frenquncies (np.array): frequences for each of the label, size(num_clas)

#     """
#     labels = labels.to_numpy()
#     _, freq = np.unique(labels, return_counts=True)
#     return freq
    
