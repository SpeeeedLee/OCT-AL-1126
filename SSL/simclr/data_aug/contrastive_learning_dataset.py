
# https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html
# We should not do AutoContrast here, because AutoContrast means doing nothinh when we already normalize imgae to 0 and 255

import os
import numpy as np
from torchvision.transforms import transforms
from torchvision import transforms, datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size, scale=(0.3, 0.7)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(), # 加入at 2025/12/2 
                                              transforms.RandomAutocontrast(),
                                              transforms.RandomEqualize(),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'train': lambda: datasets.ImageFolder(self.root_folder, 
                                                              transform=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(256),
                                                                                                        n_views))}
        
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

    def get_val_dataset(self, n_views):
        """Held-out contrastive validation set: the SAME val-half used by classification
        get_data() — sibling 'val/' folder, stratified 50/50 split (random_state=42),
        take the val half (test half stays reserved). Uses the same SimCLR augment pipeline.
        Returns None if the val folder is absent."""
        val_root = os.path.join(os.path.dirname(self.root_folder.rstrip("/")), "val")
        if not os.path.isdir(val_root):
            return None
        targets = np.array(datasets.ImageFolder(val_root).targets)
        val_idx, _ = train_test_split(np.arange(len(targets)), test_size=0.5,
                                      stratify=targets, random_state=42)
        full = datasets.ImageFolder(
            val_root, transform=ContrastiveLearningViewGenerator(
                self.get_simclr_pipeline_transform(256), n_views))
        return Subset(full, val_idx.tolist())