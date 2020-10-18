import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from augment.rand_augment import RandAugmentTransform
from augment.cutout import CutoutTransform


# Augmentation types
weakly_augment = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(0, translate=(0.125, 0.125)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
    ]
)

strongly_augment = transforms.Compose(
    [
        CutoutTransform(),
        RandAugmentTransform(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
    ]
)

normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
    ]
)


class Augmented(Dataset):
    def __init__(self, data, *augmentations):
        self.data = data
        self.augmentations = augmentations

    def __getitem__(self, index):
        x, y = self.data[index]
        xAug = [augmentation(x) for augmentation in self.augmentations]
        return (*xAug, y)

    def __len__(self):
        return len(self.data)


def create_dataset_split(dataset, labeled_size, validation_size):
    """Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results with the seed.
    """

    total_size = labeled_size + validation_size

    part1, part2, part3 = torch.utils.data.random_split(dataset, [labeled_size, len(dataset) - total_size, validation_size])

    labeled_dataset = Augmented(part1, weakly_augment)

    # In the FixMatch paper they mention that they add the labeled dataset to the unlabeled dataset. However, because we want measure
    # the accuracy on the unlabeled samples, training on it would bias the metric, and therefore we don't add the labeled data.
    # However, to add the labeled data change this to the following line:
    # unlabled_dataset = unlabeled(ConcatDataset([part1, part2]), weakly_augment, strongly_augment)
    unlabled_dataset = Augmented(part2, weakly_augment, strongly_augment)

    validation_dataset = Augmented(part3, normalize)

    return labeled_dataset, unlabled_dataset, validation_dataset
