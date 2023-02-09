import os
import shutil
import random
from pathlib import Path

def split_data(root_dir, val_percent, max_ratio=7):
    dataset_directory = os.path.join(Path(root_dir).parent, "biosup")
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)

    train_directory = os.path.join(dataset_directory, "train")
    val_directory = os.path.join(dataset_directory, "val")
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(val_directory, exist_ok=True)

    classes = os.listdir(root_dir)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    images_per_class = [len(os.listdir(os.path.join(root_dir, cls))) for cls in classes]
    print(f"Number of images per class prior to balancing: {images_per_class}")

    assert any(num < 20 for num in images_per_class) is False, "Each class must contain at least 50 images. Please remove classes with fewer images."

    min_number = min(images_per_class)
    print(f"Minimum number of images per class: {min_number}")
      
    images_per_class = [min(max_ratio * min_number, num) for num in images_per_class]
    print(f"Number of images per class after balancing: {images_per_class}")

    num_images = sum(images_per_class)
    num_val_images = int(val_percent * num_images)
    val_images_per_class = num_val_images // len(classes)
    print (f"Number of images per class reserved for validation: {val_images_per_class}")


    for class_ in classes:
        print(f"Processing class {class_}...")
        class_dir = os.path.join(root_dir, class_)
        img_paths = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
        num_class_images = len(img_paths)
        if num_class_images//2 < val_images_per_class:
            print(f"Warning: {class_} contains fewer than {2*val_images_per_class} images and will be ignored.")
            continue
        if num_class_images <= max_ratio * min_number:
            images_to_use = img_paths
        else:
            print(f"Warning: {class_} contains more than {max_ratio * min_number} images. Only {max_ratio * min_number} images will be used.")
            images_to_use = random.sample(img_paths, max_ratio * min_number)
        val_images = random.sample(images_to_use, val_images_per_class)
        for img_path in images_to_use:
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