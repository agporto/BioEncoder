import os
import shutil
import random
from pathlib import Path

def split_data(root_dir, val_percent):
    dataset_directory = os.path.join(Path(root_dir).parent, "biosup")
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)

    train_directory = os.path.join(dataset_directory, "train")
    val_directory = os.path.join(dataset_directory, "val")
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(val_directory, exist_ok=True)

    classes = os.listdir(root_dir)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    num_images = sum([len(os.listdir(os.path.join(root_dir, cls))) for cls in classes])
    num_val_images = int(val_percent * num_images)
    num_images_per_class = num_val_images // len(classes)

    for class_ in classes:
        class_dir = os.path.join(root_dir, class_)
        img_paths = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
        num_class_images = len(img_paths)
        if num_class_images//2 < num_images_per_class:
            print(f"Warning: {class_} contains fewer than {2*num_images_per_class} images and will be ignored.")
            continue
        val_images = random.sample(img_paths, num_images_per_class)
        for img_path in img_paths:
            if img_path in val_images:
                dest_dir = os.path.join(val_directory, class_)
            else:
                dest_dir = os.path.join(train_directory, class_)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(img_path, dest_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="path to the root directory")
    parser.add_argument("--val-percent", type=float, help="percentage of images to be retained for validation (default=0.1)", default=0.1)
    args = parser.parse_args()
    split_data(args.dataset, args.val_percent)