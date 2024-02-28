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
    min_images : TYPE, optional
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
    
    ## load bioencoer config
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
            shutil.rmtree(dir)
        else:
            os.makedirs(train_directory)
            os.makedirs(val_directory)
    else:
        print("\n[dry run - not actually copying files]\n")
    
    ## get images 
    classes = os.listdir(image_dir)
    images_per_class = [len(os.listdir(os.path.join(image_dir, cls))) for cls in classes]  
    
    print(f"Number of images per class prior to balancing: {images_per_class} ({sum(images_per_class)} total)")
    assert (
        any(num < min_per_class for num in images_per_class) is False
    ), f"Each class must contain at least {min_per_class} images. Please remove classes with fewer images."

    ## check for max ratio
    min_num = min(images_per_class)
    max_num = max_ratio * min_num
    print(f"Minimum number of images per class: {min_num} * max ratio {max_ratio} = {int(max_num)} max per class")

    images_per_class = [min(max_num, num) for num in images_per_class]
    print(f"Number of images per class after balancing: {images_per_class} ({sum(images_per_class)} total)")

    if mode == "flat":
        
        ## feedback
        num_val_images = int(val_percent * sum(images_per_class))
        val_images_per_class = num_val_images // len(classes)
        print(f"Mode \"flat\": {num_val_images} validation images in total, min. {val_images_per_class} per class - processing:\n")
        
        for class_, num_class_images in zip(classes, images_per_class):
            print(f"Processing class {class_}...")
            class_dir = os.path.join(image_dir, class_)
            img_paths = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]

            ## check min training imgs
            n_train_imgs = num_class_images - val_images_per_class
            if n_train_imgs < 50:
                print(f"Warning: {class_} contains fewer than 50 images for training ({n_train_imgs}) - excluding this class!")
                continue
            if val_images_per_class > n_train_imgs :
                print(f"Warning: {class_} contains fewer images for training ({n_train_imgs}) than for validation ({val_images_per_class}) .")

            ## check if too many
            if num_class_images <= max_num:
                images_to_use = img_paths
            else:
                print(f"Warning: {class_} contains more than {max_num} images. Only {max_num} images will be used.")
                random.seed(random_seed)
                images_to_use = random.sample(img_paths, max_num)
                
            ## randomly select from each class
            random.seed(random_seed)
            val_images = random.sample(images_to_use, val_images_per_class)
            for img_path in images_to_use:
                if img_path in val_images:
                    dest_dir = os.path.join(val_directory, class_)
                else:
                    dest_dir = os.path.join(train_directory, class_)
                if not dry_run:
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(img_path, dest_dir)
                
    elif mode == "random":

        ## feedback
        sample_size = max(1, int(sum(images_per_class) * val_percent))
        print(f"Mode \"random\": {sample_size} validation images in total, randomly selected across all classes - processing:\n")
        
        ## collect image paths            
        img_paths = []
        for class_, num_class_images in zip(classes, images_per_class):
            print(f"Processing class {class_}...")
            class_dir = os.path.join(image_dir, class_)
            random.seed(random_seed)
            img_paths = img_paths + [os.path.join(class_dir, img) for img in random.sample(os.listdir(class_dir),num_class_images)]
            
        # Sample random files
        random.seed(random_seed)
        val_set = random.sample(img_paths, sample_size)
        train_set = set(img_paths) - set(val_set)
        
        for img_set, target_dir in zip(
                [val_set, train_set], 
                [val_directory, train_directory]):
            for img_path in img_set:
                class_ = os.path.basename(os.path.dirname(img_path))
                dest_dir = os.path.join(target_dir, class_)
            if not dry_run:
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(img_path, dest_dir)
                
        final_val_set = dict(zip(os.listdir(val_directory), [len(os.listdir(os.path.join(val_directory, class_dir))) for class_dir in os.listdir(val_directory)]))
        print(f"validation set: {final_val_set}")
                
    elif mode == "fixed":
        
        ## feedback
        images_per_class_val = [int(class_imgs * val_percent) for class_imgs in images_per_class] 
        print(f"Mode \"fixed\": {sum(images_per_class_val)} validation images in total. {dict(zip(classes, images_per_class_val))} - processing:\n")
        
        ## collect image paths and apply split
        img_paths, val_set = [], []
        for class_, num_class_images in zip(classes, images_per_class_val):
            print(f"Processing class {class_}...")
            class_dir = os.path.join(image_dir, class_)
            img_paths = img_paths + [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
            val_set = val_set + [os.path.join(class_dir, img) for img in random.sample(os.listdir(class_dir), num_class_images)]
            
        train_set = set(img_paths) - set(val_set)
        
        for img_set, target_dir in zip(
                [val_set, train_set], 
                [val_directory, train_directory]):
            for img_path in img_set:
                class_ = os.path.basename(os.path.dirname(img_path))
                dest_dir = os.path.join(target_dir, class_)
            if not dry_run:
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(img_path, dest_dir)
        

        
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

