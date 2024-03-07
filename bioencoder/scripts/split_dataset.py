#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import shutil
import random

from bioencoder.core import utils

#%%

def split_dataset(
        image_dir, 
        mode="flat",
        val_percent=0.1, 
        max_ratio=7,
        min_per_class=20,
        random_seed=42,
        dry_run=False,
        overwrite=False,
        **kwargs,
        ):

    """
    Split input dataset into a training set and a validation set.

    Parameters
    ----------
    image_dir : str
        path images (already sorted into class specific sub-folders).
    mode : str, optional
        type of split:
            - "flat":
            - "random": 
            - "fixed": 
    val_percent : float, optional
        train/valitation split. The default is 0.1.
    max_ratio : int, optional
        The maximium ratio between the least and most abdundant class. Images 
        in the most abundant class beyond this ratio will be excluded by random 
        selection. The default is 7.
    min_per_class : TYPE, optional
        DESCRIPTION. The default is 20.
    dry_run : TYPE, optional
        DESCRIPTION. The default is False.
    overwrite : TYPE, optional
        DESCRIPTION. The default is False.
    random_seed : int, optional
        Random seed for the selection of the training and validation images. 
        The default is 42.

    Returns
    -------
    None.

    """
    
    ## load bioencoer configmin_images
    config = utils.load_config(kwargs.get("bioencoder_config_path"))
    root_dir = config.root_dir
    run_name = config.run_name
    
    ## directory management
    dataset_directory = os.path.join(root_dir, "data", run_name)     
    train_directory = os.path.join(dataset_directory, "train")
    val_directory = os.path.join(dataset_directory, "val")
    if not dry_run:
        if os.path.exists(dataset_directory) and overwrite==True:
            print(f"removing {dataset_directory} (ow=True)")
            shutil.rmtree(dataset_directory)
        os.makedirs(train_directory)
        os.makedirs(val_directory)
    else:
        print("\n[dry run - not actually copying files]\n")
    
    ## get images 
    class_names = os.listdir(image_dir)
    all_n_images = [len(os.listdir(os.path.join(image_dir, cls))) for cls in class_names]  
    
    print(f"Number of images per class prior to balancing: {all_n_images} ({sum(all_n_images)} total)")
    assert (
        any(num < min_per_class for num in all_n_images) is False
    ), f"Each class must contain at least {min_per_class} images. Please remove classes with fewer images."

    ## check for max ratio
    min_num = min(all_n_images)
    max_num = max_ratio * min_num
    print(f"Minimum number of images per class: {min_num} * max ratio {max_ratio} = {int(max_num)} max per class")

    all_n_images_balanced = [min(max_num, num) for num in all_n_images]
    print(f"Number of images per class after balancing: {all_n_images_balanced} ({sum(all_n_images_balanced)} total)")

    if mode == "flat":
        
        ## feedback
        n_images_val = int(sum(all_n_images_balanced) * val_percent)
        class_n_images_val = int(n_images_val / len(class_names))
        print(f"Mode \"flat\": {n_images_val} validation images in total, min. {class_n_images_val} per class - processing:\n")
        
        ## collect image paths and apply split
        for class_name, class_n_images in zip(class_names, all_n_images_balanced):
            print(f"Processing class {class_name}...")
            class_dir = os.path.join(image_dir, class_name)
            random.seed(random_seed)
            class_images_selection = random.sample(os.listdir(class_dir), class_n_images)          

            ## check min training imgs
            n_train_imgs = len(class_images_selection) - class_n_images_val
            if n_train_imgs < 50:
                print(f"Warning: {class_name} contains fewer than 50 images for training ({n_train_imgs}) - excluding this class!")
                continue
            if class_n_images_val > n_train_imgs :
                print(f"Warning: {class_name} contains fewer images for training ({n_train_imgs}) than for validation ({class_n_images_val}) .")
                
            ## apply split 
            val_set = class_images_selection[:class_n_images_val]
            train_set = class_images_selection[class_n_images_val:]    
            if not dry_run:
                for image_set, target_dir in zip([val_set, train_set], [val_directory, train_directory]):
                    dest_dir = os.path.join(target_dir, class_name)
                    os.makedirs(dest_dir, exist_ok=True)
                    for image_name in image_set:
                        shutil.copy(os.path.join(class_dir, image_name), dest_dir)
                
    if mode == "random":

        ## feedback
        n_images_val = int(sum(all_n_images_balanced) * val_percent)
        print(f"Mode \"random\": {n_images_val} validation images in total, randomly selected across all classes - processing:\n")
        
        ## collect image paths and subsample to balance     
        class_images_selection = []
        for class_name, class_n_images in zip(class_names, all_n_images_balanced):
            print(f"Processing class {class_name}...")
            class_dir = os.path.join(image_dir, class_name)
            random.seed(random_seed)
            class_images_selection = class_images_selection + [os.path.join(class_dir, image_name) for image_name in random.sample(os.listdir(class_dir), class_n_images)]
            
        ## subample from balanced set and apply split
        random.seed(random_seed)
        val_set = random.sample(class_images_selection, n_images_val)
        train_set = set(class_images_selection) - set(val_set)
        if not dry_run:
            for image_set, target_dir in zip([val_set, train_set], [val_directory, train_directory]):
                for image_path in image_set:
                    class_name = os.path.basename(os.path.dirname(image_path))
                    dest_dir = os.path.join(target_dir, class_name)
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(image_path, dest_dir)
                
            final_val_set = dict(zip(os.listdir(val_directory), [len(os.listdir(os.path.join(val_directory, class_dir))) for class_dir in os.listdir(val_directory)]))
            print(f"validation set: {final_val_set}")
                
    elif mode == "fixed":
        
        ## feedback
        all_n_images_val = [int(n_images * val_percent) for n_images in all_n_images_balanced] 
        print(f"Mode \"fixed\": {sum(all_n_images_val)} validation images in total. {dict(zip(class_names, all_n_images_val))} - processing:\n")
        
        ## collect image paths and apply split
        for class_name, class_n_images in zip(class_names, all_n_images_balanced):
            print(f"Processing class {class_name}...")
            class_dir = os.path.join(image_dir, class_name)
            random.seed(random_seed)
            class_images_selection = random.sample(os.listdir(class_dir), class_n_images)
            val_set = class_images_selection[:int(class_n_images * val_percent)]
            train_set = class_images_selection[int(class_n_images * val_percent):]
            if not dry_run:
                for image_set, target_dir in zip([val_set, train_set], [val_directory, train_directory]):
                    dest_dir = os.path.join(target_dir, class_name)
                    os.makedirs(dest_dir, exist_ok=True)
                    for image_name in image_set:
                        shutil.copy(os.path.join(class_dir, image_name), dest_dir)
                

        
def cli():
       
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir", 
        type=str, 
        help="path to the image directory"
    )
    parser.add_argument(
        "--val-percent",
        type=float,
        help="percentage of images to be retained for validation (default=0.1)",
        default=0.1,
    )
    parser.add_argument(
        "--ratio",
        type=float,
        help="maximum ratio between the number of images in the most common class and the least commong one",
        default=7,
    )
    args = parser.parse_args()
    
    split_dataset(args.image_dir, args.val_percent, args.ratio)



if __name__ == "__main__":
    
    cli()

