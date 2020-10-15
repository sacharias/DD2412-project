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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

strongly_augment = transforms.Compose(
    [
        CutoutTransform(),
        RandAugmentTransform(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)


class unlabeled(Dataset):
    def __init__(self, data, weaklyAug, stronglyAug):
        self.data = data
        self.weaklyAug = weaklyAug
        self.stronglyAug = stronglyAug

    def __getitem__(self, index):
        x, y = self.data[index]

        # For some reason it converts the image to a tensor,
        # so have to convert it back to an PIL to apply transformation
        img = transforms.ToPILImage()(x)
        xW, xS = self.weaklyAug(img), self.stronglyAug(img)
        return xW, xS, y

    def __len__(self):
        return len(self.data)


def create_dataset_split(dataset, labeled_size, validation_size, seed=1):
    """Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results with the seed.
    """

    totalSize = labeled_size + validation_size

    part1, part2, part3 = torch.utils.data.random_split(dataset, [labeled_size, len(dataset) - totalSize, validation_size])#, generator=torch.Generator().manual_seed(seed))

    part1.dataset.transform = weakly_augment

    labeled_dataset = part1

    # In the FixMatch paper they mention that they add the labeled dataset to the unlabeled dataset. However, because we want measure
    # the accuracy on the unlabeled samples, training on it would bias the metric, and therefore we don't add the labeled data.
    # However, to add the labeled data change this to the following line:
    # unlabled_dataset = unlabeled(ConcatDataset([part1, part2]), weakly_augment, strongly_augment)
    unlabled_dataset = unlabeled(part2, weakly_augment, strongly_augment)

    part3.dataset.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    validation_dataset = part3

    return labeled_dataset, unlabled_dataset, validation_dataset
