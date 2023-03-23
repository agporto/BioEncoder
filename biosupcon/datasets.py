import torchvision
import os
import numpy as np


class BioDataset(torchvision.datasets.ImageFolder):
    """Custom dataset for bio images.

    This dataset inherits from `torchvision.datasets.ImageFolder` and adds a flag to indicate whether to
    perform the second stage of data augmentation.
    """

    def __init__(self, data_dir, transform, second_stage):
        """
        Args:
            data_dir (str): Path to the directory containing the images.
            transform (callable): Transformation to be applied to the images.
            second_stage (bool): Flag indicating whether to perform the second stage of data augmentation.
        """
        super().__init__(root=data_dir, transform=transform)

        self.second_stage = second_stage
        self.transform = transform

    def __getitem__(self, idx):
        """
        Returns:
            tuple: Tuple of the transformed image and its label.
        """
        path, label = self.samples[idx]
        img = self.loader(path)
        image = np.asarray(img)

        if self.second_stage:
            image = self.transform(image=image)["image"]
        else:
            image = self.transform(image)

        return image, label


def create_dataset(data_dir, train, transform, second_stage):
    """
    Creates an instance of the `BioDataset` class for either the training or validation set.

    Args:
        data_dir (str): Path to the directory containing the images.
        train (bool): Flag indicating whether to create the training set.
        transform (callable): Transformation to be applied to the images.
        second_stage (bool): Flag indicating whether to perform the second stage of data augmentation.

    Returns:
        BioDataset: Instance of the `BioDataset` class.
    """
    # Set the path to the directory containing the images
    path = os.path.join(data_dir, "train" if train else "val")
    # Return an instance of the `BioDataset` class
    return BioDataset(path, transform, second_stage)
