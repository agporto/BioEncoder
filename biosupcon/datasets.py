import torchvision
import os
import numpy as np

class BioDataset(torchvision.datasets.ImageFolder):
    def __init__(self, data_dir, transform, second_stage):
        super().__init__(root=data_dir, transform=transform)
        
        self.second_stage = second_stage
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        image = np.asarray(img)

        if self.second_stage:
            image = self.transform(image=image)['image']
        else:
            image = self.transform(image)
        
        return image, label


def create_dataset(data_dir, train, transform, second_stage):
    path = os.path.join(data_dir, 'train' if train else 'val')
    return BioDataset(path, transform, second_stage)



