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
    Splits a dataset of images into training and validation subsets based on the specified criteria.
    This function is designed to operate on datasets organized into class-specific subfolders within a given directory.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing subfolders of images, where each subfolder represents a class.
    mode : str, optional
        Specifies the strategy for splitting the dataset:
            - "flat": Calculating split to the most abundant class (after applying max_ratio), and then applying it to all classes 
            - "random": Randomly selects images across all classes to form the validation set, disregarding class balance.
            - "fixed": Ensures each class contributes a fixed proportion to the validation set, based on `val_percent`.
        Default is "flat".
    val_percent : float, optional
        Proportion of the dataset to allocate to the validation set, expressed as a decimal.
        Default is 0.1 (10%).
    max_ratio : int, optional
        Maximum allowed ratio between the most and least abundant classes in the dataset. Excess images in larger classes are randomly discarded to adhere to this ratio.
        Default is 7.
    min_per_class : int, optional
        Minimum number of images required in each class to perform the split. If any class has fewer images, an error is raised.
        Default is 20.
    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility in splits.
        Default is 42.
    dry_run : bool, optional
        If True, performs a trial run where the split is simulated but no files are actually moved or copied.
        Default is False.
    overwrite : bool, optional
        If True, allows the function to overwrite existing directories and files in the target split directories. If False, the function will not overwrite any files and will raise an error if file conflicts exist.
        Default is False.
    **kwargs : dict, optional
        Additional keyword arguments for extending function functionality or providing configurations, such as paths to configuration files.
        
    Raises
    ------
    AssertionError
        If any class contains fewer images than `min_per_class`, or if the directory structure is not as expected.
    IOError
        If `overwrite` is False and target directories for the split already exist.

    Notes
    -----
    The dataset directory structure should follow this pattern:
        /path/to/dataset/
            /class1/
            /class2/
            ...

    Examples
    --------
    This would simulate (dry_run=True) a random split where 15% of the total images are allocated to the validation set,
    and sampling randomly acrosss all classes (mode="random"):
        bioencoder.split_dataset("/path/to/dataset", mode="random", val_percent=0.15, dry_run=True)

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
            if n_train_imgs < min_per_class:
                print(f"Warning: {class_name} contains fewer than the set min value per class ({min_per_class}) - excluding this class!")
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
                
    elif mode == "random":

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
    parser.add_argument("--image-dir", type=str, help="Path to the images directory sorted into class-specific subfolders.")
    parser.add_argument("--mode", type=str, choices=['flat', 'random', 'fixed'], default='flat', help="Type of dataset split to perform.")
    parser.add_argument("--val_percent", type=float, default=0.1, help="Percentage of data to use as validation set.")
    parser.add_argument("--max_ratio", type=int, default=7, help="Maximum ratio between the most and least abundant classes.")
    parser.add_argument("--min_per_class", type=int, default=20, help="Minimum number of images per class.")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed for random number generator.")
    parser.add_argument("--dry_run", action='store_true', help="Run without making any changes.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files without asking.")
    args = parser.parse_args()
    
    split_dataset(
         args.image_dir, 
         mode=args.mode,
         val_percent=args.val_percent,
         max_ratio=args.max_ratio,
         min_per_class=args.min_per_class,
         random_seed=args.random_seed,
         dry_run=args.dry_run,
         overwrite=args.overwrite,
     )

if __name__ == "__main__":
    
    cli()

