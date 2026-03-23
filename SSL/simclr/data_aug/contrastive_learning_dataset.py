
# https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html
# We should not do AutoContrast here, because AutoContrast means doing nothinh when we already normalize imgae to 0 and 255

from torchvision.transforms import transforms
from torchvision import transforms, datasets
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